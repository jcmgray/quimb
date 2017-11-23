"""DMRG-like variational algorithms, but in tensor network language.
"""
import numpy as np
import scipy.sparse.linalg as spla
import itertools

from ..utils import progbar
from ..accel import prod
from ..linalg.base_linalg import eigsys, seigsys
from .tensor_core import Tensor, TensorNetwork, tensor_contract
from .tensor_gen import MPS_rand_state
from .tensor_1d import TN_1D_align


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
    initialzes the right environments like so::

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
    which : {'SR', 'LR'}, optional
        Whether to search for smallest or largest real part eigenvectors.

    Attributes
    ----------
    k : MatrixProductState
        The current, optimized state.
    energies : list of float
        The list of energies after each sweep.
    """

    def __init__(self, ham, bond_dim, which='SR'):
        self.n = ham.nsites
        self.bond_dim = bond_dim
        self.which = which
        self.k = MPS_rand_state(self.n, bond_dim)
        self.b = self.k.H
        self.ham = ham.copy()

        # Tag the various bits for contraction.
        self.ham.add_tag("__ham__")

        # Line up and overlap
        TN_1D_align(self.k, self.ham, self.b, inplace=True)

        # want to contract this multiple times while
        #   manipulating k -> make virtual
        self.TN_energy = self.b | self.ham | self.k
        self.energies = [self.TN_energy ^ ...]

    def update_with_eff_gs(self, eff_ham, i, dense=False):
        """Find the effective tensor groundstate of::


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

        eff_e, eff_gs = seigsys(op, k=1, which=self.which)
        eff_gs = eff_gs.A
        self.k.site[i].data = eff_gs
        self.b.site[i].data = eff_gs.conj()
        return eff_e

    def sweep_right(self, canonize=True, eff_ham_dense=False):
        """Perform a sweep of optimizations rightwards::

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
        """Perform a sweep of optimizations leftwards::

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

    def solve(self, tol=1e-9, max_sweeps=10, sweep_sequence='R',
              eff_ham_dense="AUTO", compress=False, compress_opts=None):
        """Solve the system with a sequence of sweeps, up to a certain
        absolute tolerance in the energy or maximum number of sweeps.

        Parameters
        ----------
        tol : float, optional
            The absolute tolerance to converge energy to.
        max_sweeps : int, optional
            The maximum number of sweeps to perform.
        sweep_sequence : str, optional
            String made of 'L' and 'R' defining the sweep sequence, e.g 'RRL'.
            The sequence will be repeated until ``max_sweeps`` is reached.
        compress : bool, optional
            Whether to compress the state after each sweep (but then expand to
            ``bond_dim`` before each iteration).
        eff_ham_dense : "AUTO" or bool, optional
            Whether to use a dense representation of the effective hamiltonians
            or a tensor contraction based linear operation representation.
        """
        # choose a rough value at which dense effective ham should not be used
        if eff_ham_dense == "AUTO":
            eff_ham_dense = self.bond_dim < 20

        compress_opts = {} if compress_opts is None else dict(compress_opts)

        RLs = itertools.cycle(sweep_sequence)
        previous_LR = '0'

        for i in progbar(range(max_sweeps)):
            LR = next(RLs)
            # if last sweep was opposite direction no need to canonize
            canonize = False if LR + previous_LR in {'LR', 'RL'} else True

            if LR == 'R':
                self.energies.append(self.sweep_right(
                    eff_ham_dense=eff_ham_dense, canonize=canonize))
            elif LR == 'L':
                self.energies.append(self.sweep_left(
                    eff_ham_dense=eff_ham_dense, canonize=canonize))

            if compress:
                self.k.compress(form=LR, bra=self.b, **compress_opts)

            if abs(self.energies[-2] - self.energies[-1]) < tol:
                break

            if compress and i < max_sweeps - 1:
                self.k.expand_bond_dimension(
                    self.bond_dim, inplace=True, bra=self.b)

            previous_LR = LR

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
        self.k = p0.expand_bond_dimension(bond_dim)  # copies as well
        self.b = self.k.H
        self.ham = ham.copy()
        self.ham.add_tag("__ham__")

        # Line up and overlap
        TN_1D_align(self.k, self.ham, self.b, inplace=True)

        # want to contract this multiple times while
        #   manipulating k -> make virtual
        self.TN_energy = self.b | self.ham | self.k
        self.energies = [self.TN_energy ^ ...]

        # Want to keep track of energy variance as well
        var_ham1 = self.ham.copy()
        var_ham1.upper_ind_id = self.k.site_ind_id
        var_ham1.lower_ind_id = "__var_ham{}__"
        var_ham2 = self.ham.copy()
        var_ham2.upper_ind_id = "__var_ham{}__"
        var_ham2.lower_ind_id = self.b.site_ind_id
        self.TN_en_var2 = self.k | var_ham1 | var_ham2 | self.b
        self.variances = [(self.TN_en_var2 ^ ...) - self.energies[-1]**2]

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

        # find the index of the highest overlap eigenvector, by contracting::
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

        for i in reversed(range(0, self.n)):
            enrg_envs.move_left()
            ovlp_envs.move_left()
            en = self.update_with_best_evec(enrg_envs[i], ovlp_envs[i], i)
            if i > 0:
                self.k.right_canonize_site(i, bra=self.b)

        return en

    def solve(self, vtol=1e-9, max_sweeps=10, sweep_sequence='R'):
        """Solve the system with a sequence of sweeps, up to a certain
        absolute tolerance in the energy variance, i.e. ``<E^2> - <E>^2``,
        or maximum number of sweeps.

        Parameters
        ----------
        vtol : float, optional
            The absolute tolerance to converge energy variance to.
        max_sweeps : int, optional
            The maximum number of sweeps to perform.
        sweep_sequence : str, optional
            String made of 'L' and 'R' defining the sweep sequence, e.g 'RRL'.
            The sequence will be repeated until ``max_sweeps`` is reached.
        """
        RLs = itertools.cycle(sweep_sequence)
        previous_LR = '0'

        for i in progbar(range(max_sweeps)):
            LR = next(RLs)
            # if last sweep was opposite direction no need to canonize
            canonize = False if LR + previous_LR in {'LR', 'RL'} else True

            if LR == 'R':
                self.energies.append(self.sweep_right(canonize=canonize))
            elif LR == 'L':
                self.energies.append(self.sweep_left(canonize=canonize))

            # update the variances
            self.variances.append(
                (self.TN_en_var2 ^ ...) - self.energies[-1]**2)

            if self.variances[-1] < vtol:
                break

            previous_LR = LR

        return self.energies[-1], self.k
