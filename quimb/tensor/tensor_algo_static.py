"""DMRG-like variational algorithms, but in tensor network language.
"""
import numpy as np
import scipy.sparse.linalg as spla

from ..utils import progbar
from ..accel import prod
from ..linalg.base_linalg import eigsys
from .tensor_core import Tensor, TensorNetwork, tensor_contract
from .tensor_gen import MPS_rand_state
from .tensor_1d import align_inner


class EffectiveHamLinearOperator(spla.LinearOperator):
    """Get a linear operator - something that replicates the matrix-vector
    operation - for an arbitrary *uncontracted* hamiltonian operator, e.g:

         / | | \
        L--H-H--R
         \ | | /

    This can then be supplied to scipy's sparse linear algebra routines.

    Parameters
    ----------
    TN_ham : TensorNetwork
        A representation of the hamiltonian
    upper_inds : sequence of hashable
        The upper inds of the effective hamiltonian network.
    lower_inds : sequence of hashable
        The lower inds of the effective hamiltonian network. These should be
        ordered the same way as ``upper_inds``.
    dims : tuple of int
        The dimensions corresponding to the inds.
    """

    def __init__(self, TN_ham, upper_inds, lower_inds, dims):
        self.eff_ham_tensors = TN_ham["__ham__"]
        self.upper_inds = upper_inds
        self.lower_inds = lower_inds
        self.dims = dims
        self.d = prod(dims)
        super().__init__(dtype=complex, shape=(self.d, self.d))

    def _matvec(self, vec):
        v = Tensor(vec.reshape(*self.dims), inds=self.upper_inds)
        v_out = tensor_contract(*self.eff_ham_tensors, v,
                                output_inds=self.lower_inds).data
        return v_out.reshape(*vec.shape)


class Moving1SiteEnv:
    """Helper class for efficiently moving the effective 'environment' of a
    single site in a 1D tensor network. E.g. for ``start='left'`` this
    initialzes the right environments like so:

        n - 1: o-o-o-     -o-o-o
               | | |       | | |
               H-H-H- ... -H-H-H
               | | |       | | |
               o-o-o-     -o-o-o

        n - 2: o-o-o-     -o-o\
               | | |       | | o
               H-H-H- ... -H-H-H
               | | |       | | o
               o-o-o-     -o-o/

        n - 3: o-o-o-     -o\
               | | |       | oo
               H-H-H- ... -H-HH
               | | |       | oo
               o-o-o-     -o/

        ...

        0    : o\
               | oo   ooo
               H-HH...HHH
               | oo   ooo
               o/

    which can then be used to efficiently generate the left environments as
    each site is updated.

    Parameters
    ----------
    tn : TensorNetwork
        A 1d-ish tensor network.
    n : int
        Number of sites.
    start : {'left', 'right'}
        Which side to start at.
    """

    def __init__(self, tn, n, start):
        self.n = n
        self.start = start

        if start == 'left':
            first_i = n - 1
            sweep = reversed(range(0, n - 1))
            previous_step = 1
            self.site = 0
        elif start == 'right':
            first_i = 0
            sweep = range(1, n)
            previous_step = -1
            self.site = n - 1
        else:
            raise ValueError("'start' must be one of {'left', 'right'}.")

        self.envs = {first_i: tn.copy(virtual=True)}
        for i in sweep:
            env = self.envs[i + previous_step].copy(virtual=True)
            limit = min(n - 1, i + 2) if (start == 'left') else max(0, i - 2)
            env ^= slice(limit, i)
            self.envs[i] = env

    def move_right(self):
        i = self.site

        if i >= 2:
            # replace left env with new effective left env
            for j in range(i - 1):
                del self.envs[i].site[j]
            self.envs[i] |= self.envs[i - 1].site[i - 2]

        if i >= 1:
            # contract left env with new minimized, canonized site
            self.envs[i] ^= slice(max(0, i - 2), i)

        self.site += 1

    def move_left(self):
        i = self.site

        if i <= self.n - 3:
            # replace right env with new effective right env
            for j in reversed(range(i + 2, self.n)):
                del self.envs[i].site[j]
            self.envs[i] |= self.envs[i + 1].site[i + 2]

        if i <= self.n - 2:
            # contract right env with new minimized, canonized site
            self.envs[i] ^= slice(min(self.n - 1, i + 2), i)

        self.site -= 1

    def __getitem__(self, i):
        return self.envs[i]


class DMRG1:
    """Single site, fixed bond-dimension variational groundstate search.

    Parameters
    ----------
    ham : MatrixProductOperator
        The hamiltonian in MPO form.
    bond_dim : int
        The bond-dimension of the MPS to optimize.

    Attributes
    ----------
    k : MatrixProductState
        The current, optimized state.
    energies : list of float
        The list of energies after each sweep.
    """

    def __init__(self, ham, bond_dim):
        self.n = ham.nsites
        self.bond_dim = bond_dim
        self.k = MPS_rand_state(self.n, bond_dim)
        self.b = self.k.H

        # Tag the various bits for contraction.
        ham.add_tag("__ham__")

        # Line up and overlap
        align_inner(self.k, self.b, ham)

        # want to contract this multiple times while
        #   manipulating k -> make virtual
        self.TN_energy = TensorNetwork([self.b, ham, self.k], virtual=True)
        self.energies = []
        self.site_id = ham.site_tag_id

    def update_with_eff_gs(self, eff_ham, i, dense=False):
        """Find the effective tensor groundstate of:


                      /|\
            >->->->->- | -<-<-<-<-<-<-<-<          |||
            | | | | |  |  | | | | | | | |         / | \
            H-H-H-H-H--H--H-H-H-H-H-H-H-H   =    L--H--R
            | | | | | i|  | | | | | | | |         \i| /
            >->->->->- | -<-<-<-<-<-<-<-<          |||
                      \|/

        And insert it back into the states ``k`` and ``b``, and thus
        ``TN_energy``.
        """
        if dense:
            # contract remaining hamiltonian and get its dense representation
            eff_ham = (eff_ham ^ '__ham__')['__ham__']
            eff_ham.fuse((('lower', self.b.site[i].inds),
                          ('upper', self.k.site[i].inds)), inplace=True)
            op = eff_ham.data
        else:
            op = EffectiveHamLinearOperator(eff_ham, dims=self.k.site[i].shape,
                                            upper_inds=self.k.site[i].inds,
                                            lower_inds=self.b.site[i].inds)

        eff_e, eff_gs = spla.eigs(op, k=1, which='SR')
        self.k.site[i].data = eff_gs
        self.b.site[i].data = eff_gs.conj()
        return eff_e

    def sweep_right(self, canonize=True, eff_ham_dense=False):
        """Perform a sweep of optimizations rightwards:

              optimize -->
                .
            >->-o-<-<-<-<-<-<-<-<-<-<-<-<-<
            | | | | | | | | | | | | | | | |
            H-H-H-H-H-H-H-H-H-H-H-H-H-H-H-H
            | | | | | | | | | | | | | | | |
            >->-o-<-<-<-<-<-<-<-<-<-<-<-<-<

        After the sweep the state is left canonized.

        Parameters
        ----------
        canonize : bool, optional
            Right canonize first. Set to False if already right-canonized.
        eff_ham_dense : bool, optional
            Solve the inner eigensystem using a dense representation of the
            effective hamiltonian. Can be quicker for small bond_dim.
        """
        if canonize:
            self.k.right_canonize(bra=self.b)

        eff_envs = Moving1SiteEnv(self.TN_energy, self.n, start='left')

        for i in range(0, self.n):
            eff_envs.move_right()
            en = self.update_with_eff_gs(eff_envs[i], i, dense=eff_ham_dense)

            if i < self.n - 1:
                self.k.left_canonize_site(i, bra=self.b)

        return en

    def sweep_left(self, canonize=True, eff_ham_dense=False):
        """Perform a sweep of optimizations leftwards:

                            <-- optimize
                                      .
            >->->->->->->->->->->->->-o-<-<
            | | | | | | | | | | | | | | | |
            H-H-H-H-H-H-H-H-H-H-H-H-H-H-H-H
            | | | | | | | | | | | | | | | |
            >->->->->->->->->->->->->-o-<-<

        After the sweep the state is right canonized.

        Parameters
        ----------
        canonize : bool, optional
            Left canonize first. Set to False if already right-canonized.
        eff_ham_dense : bool, optional
            Solve the inner eigensystem using a dense representation of the
            effective hamiltonian. Can be quicker for small bond_dim.
        """
        if canonize:
            self.k.left_canonize(bra=self.b)

        eff_envs = Moving1SiteEnv(self.TN_energy, self.n, start='right')

        for i in reversed(range(0, self.n)):
            eff_envs.move_left()
            en = self.update_with_eff_gs(eff_envs[i], i, dense=eff_ham_dense)

            if i > 0:
                self.k.right_canonize_site(i, bra=self.b)

        return en

    def solve(self, max_sweeps=4, eff_ham_dense="AUTO"):
        """Sweep a number of times.
        """
        # choose a rough value at which dense effective ham should not be used
        if eff_ham_dense == "AUTO":
            eff_ham_dense = self.bond_dim < 20

        for _ in progbar(range(max_sweeps)):
            self.energies.append(self.sweep_right(eff_ham_dense=eff_ham_dense))

        return self.energies[-1], self.k


class DMRGX:
    """Class implmenting DMRG-X [1], whereby local effective energy eigenstates
    are chosen to maximise overlap with the previous step's state, leading to
    convergence on an mid-spectrum eigenstate of the full hamiltonian, as long
    as it is perturbatively close to the original state.

    [1] Khemani, V., Pollmann, F. & Sondhi, S. L. Obtaining Highly Excited
    Eigenstates of Many-Body Localized Hamiltonians by the Density Matrix
    Renormalization Group Approach. Phys. Rev. Lett. 116, 247204 (2016).

    Parameters
    ----------
    ham : MatrixProductOperator
        The hamiltonian in MPO form, should have ~area-law eigenstates.
    p0 : MatrixProductState
        The intial MPS guess, e.g. a computation basis state.
    bond_dim : int
        The bond dimension to find the state for.

    Attributes
    ----------
    k : MatrixProductState
        The current, optimized state.
    energies : list of float
        The list of energies after each sweep.
    """

    def __init__(self, ham, p0, bond_dim):
        self.n = ham.nsites
        self.k = p0.expand_bond_dimension(bond_dim)
        self.b = self.k.H

        # Tag the various bits for contraction.
        ham.add_tag("__ham__")

        # Line up and overlap
        align_inner(self.k, self.b, ham)

        # want to contract this multiple times while
        #   manipulating k -> make virtual
        self.TN_energy = TensorNetwork([self.b, ham, self.k], virtual=True)

        self.energies = []
        self.site_id = ham.site_tag_id

    def update_with_best_evec(self, eff_ham, eff_ovlp, i):
        """Like ``update_with_eff_gs``, but re-insert all eigenvectors, then
        choose the one with best overlap with ``eff_evlp``.
        """

        # contract remaining hamiltonian and get its dense representation
        eff_ham = (eff_ham ^ '__ham__')['__ham__']
        eff_ham.fuse((('lower', self.b.site[i].inds),
                      ('upper', self.k.site[i].inds)), inplace=True)
        op = eff_ham.data

        # eigen-decompose and reshape eigenvectors thus:  |
        #                                                 E
        #                                                /|\
        evals, evecs = eigsys(op)
        evecs = np.asarray(evecs).reshape(*self.k.site[i].shape, -1)

        # update tensor at site i with all evecs -> need dummy index
        tnsr = self.k.site[i]
        tnsr.update(data=evecs, inds=(*tnsr.inds, '__ev_ind__'))

        # find the index of the highest overlap eigenvector, by contracting:
        #
        #           |
        #     o-o-o-E-o-o-o-o-o-o-o
        #     | | | | | | | | | | |
        #     O-O-O-O-O-O-O-O-O-O-O  <- state from previous step
        #
        best = np.argmax(np.abs((eff_ovlp ^ ...).data))

        # update site i with the data and drop dummy index too
        tnsr.update(data=evecs[..., best], inds=tnsr.inds[:-1])

        # update the bra -> it only needs the new, conjugated data.
        self.b.site[i].data = evecs[..., best].conj()
        return evals[best]

    def sweep_right(self, canonize=True):
        self.old_k = self.k.copy().H
        TN_overlap = TensorNetwork([self.k, self.old_k], virtual=True)

        if canonize:
            self.k.right_canonize(bra=self.b)

        enrg_envs = Moving1SiteEnv(self.TN_energy, self.n, start='left')
        ovlp_envs = Moving1SiteEnv(TN_overlap, self.n, start='left')

        for i in range(0, self.n):
            enrg_envs.move_right()
            ovlp_envs.move_right()
            en = self.update_with_best_evec(enrg_envs[i], ovlp_envs[i], i)
            if i < self.n - 1:
                self.k.left_canonize_site(i, bra=self.b)

        return en

    def sweep_left(self, canonize=True):
        self.old_k = self.k.copy().H
        TN_overlap = TensorNetwork([self.k, self.old_k], virtual=True)

        if canonize:
            self.k.left_canonize(bra=self.b)

        enrg_envs = Moving1SiteEnv(self.TN_energy, self.n, start='right')
        ovlp_envs = Moving1SiteEnv(TN_overlap, self.n, start='right')

        for i in range(0, self.n):
            enrg_envs.move_left()
            ovlp_envs.move_left()
            en = self.update_with_best_evec(enrg_envs[i], ovlp_envs[i], i)
            if i < self.n - 1:
                self.k.right_canonize_site(i, bra=self.b)

        return en
