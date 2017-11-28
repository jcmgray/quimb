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


class EffHamOp(spla.LinearOperator):
    """Get a linear operator - something that replicates the matrix-vector
    operation - for an arbitrary *uncontracted* hamiltonian operator, e.g:

         / | | \
        L--H-H--R  <- tensors should be tagged with "__ham__"
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


class MovingEnvironment:
    """Helper class for efficiently moving the effective 'environment' of a
    few sites in a 1D tensor network. E.g. for ``start='left', bsz=2``, this
    initializes the right environments like so::

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

        0    : o-o\
               | | oo   ooo
               H-H-HH...HHH
               | | oo   ooo
               o-o/

    which can then be used to efficiently generate the left environments as
    each site is updated. For example if ``bsz=2`` and the environements have
    been shifted many sites into the middle, then ``MovingEnvironment[i]``
    returns something like::

             <---> bsz sites
             /o-o\
        ooooo | | ooooooo
        HHHHH-H-H-HHHHHHH
        ooooo | | ooooooo
             \o-o/
        0 ... i i+1 ... n-1

    Does not necessarily need to be an operator overlap tensor network. Useful
    for any kind of sweep where only local tensor updates are being made. Note
     that *only* the current site is completely up-to-date.

    Parameters
    ----------
    tn : TensorNetwork
        A 1d-ish tensor network.
    n : int
        Number of sites.
    start : {'left', 'right'}
        Which side to start at, e.g. 'left' for diagram above.
    bsz : int
        Number of local sites to keep un-contracted from the environment.
        Defaults to 1, but would be 2 for DMRG2 etc.
    """

    def __init__(self, tn, n, start, bsz=1):
        self.n = n
        self.start = start
        self.bsz = bsz
        self.num_blocks = self.n - self.bsz + 1

        if start == 'left':
            initial_j = n - self.bsz
            sweep = reversed(range(0, self.num_blocks - 1))
            previous_step = 1
            self.pos = 0
        elif start == 'right':
            initial_j = 0
            sweep = range(1, self.num_blocks)
            previous_step = -1
            self.pos = n - self.bsz
        else:
            raise ValueError("'start' must be one of {'left', 'right'}.")

        self.envs = {initial_j: tn.copy(virtual=True)}
        for j in sweep:
            # get the env from previous step
            env = self.envs[j + previous_step].copy(virtual=True)

            # contract it with one more site, no-op if near end, to get jth env
            if start == 'left':
                env ^= slice(j + self.bsz, min(n, j + self.bsz + 2))
            else:
                env ^= slice(max(0, j - 2), j)
            self.envs[j] = env

    def move_right(self):
        i = self.pos + 1

        if i > 1:
            # replace left env with new effective left env
            for j in range(i - 1):
                del self.envs[i].site[j]
            self.envs[i] |= self.envs[i - 1].site[i - 2]

        if i > 0:
            # contract left env with new minimized, canonized site
            self.envs[i] ^= slice(max(0, i - 2), i)

        self.pos += 1

    def move_left(self):
        i = self.pos - 1

        if i < self.n - self.bsz - 1:
            # replace right env with new effective right env
            for j in range(self.n - 1, i + self.bsz, -1):
                del self.envs[i].site[j]
            self.envs[i] |= self.envs[i + 1].site[i + self.bsz + 1]

        if i < self.n - self.bsz:
            # contract right env with new minimized, canonized site
            self.envs[i] ^= slice(min(self.n - 1, i + self.bsz + 1),
                                  i + self.bsz - 1)
        self.pos -= 1

    def move_to(self, i):
        if not (0 <= i < self.num_blocks):
            raise ValueError("Condition 0 <= {} < {} not satisfied"
                             "".format(i, self.num_blocks))
        if i < self.pos:
            while self.pos != i:
                self.move_left()
        else:
            while self.pos != i:
                self.move_right()

    def __call__(self):
        return self.envs[self.pos]


class DMRG:
    """Single site, fixed bond-dimension variational groundstate search.
    Some initialising arguments act as defaults, but can be overidden with
    each solve or sweep.

    Parameters
    ----------
    ham : MatrixProductOperator
        The hamiltonian in MPO form.
    bond_dim : int or sequence of ints.
        The bond-dimension of the MPS to optimize. If ``bsz > 1``, then this
        corresponds to the maximum bond dimension when splitting the effective
        local groundstate. If a sequence is supplied then successive sweeps
        iterate through, then repeate the final value. E.g.
        ``[16, 32, 64] -> (16, 32, 64, 64, 64, ...)```.
    bsz : {1, 2}
        Number of sites to optimize for locally.
    which : {'SR', 'LR'}, optional
        Whether to search for smallest or largest real part eigenvectors.
    compress_opts : dict-like
        Options to

    Attributes
    ----------
    k : MatrixProductState
        The current, optimized state.
    energies : list of float
        The list of energies after each sweep.
    """

    def __init__(self, ham, bond_dim, bsz=1, which='SR', compress_opts=None):
        self.n = ham.nsites
        self.bond_dim = bond_dim
        self.phys_dim = ham.phys_dim(0)
        self.bsz = bsz
        self.which = which
        self.compress_opts = ({} if compress_opts is None else
                              dict(compress_opts))

        # create internal states and ham
        self.k = MPS_rand_state(self.n, bond_dim, phys_dim=self.phys_dim)
        self.b = self.k.H
        self.ham = ham.copy()
        self.ham.add_tag("__ham__")

        # Line up and overlap for energy calc
        self.k.align(self.ham, self.b, inplace=True)

        # want to contract this multiple times while
        #   manipulating k/b -> make virtual
        self.TN_energy = self.b | self.ham | self.k
        self.energies = [self.TN_energy ^ ...]

    def update_local_gs_1site(self, eff_ham, i, direction, dense=False):
        """Find the single site effective tensor groundstate of::


                      /|\
            >->->->->- | -<-<-<-<-<-<-<-<          /|\
            | | | | |  |  | | | | | | | |         / | \
            H-H-H-H-H--H--H-H-H-H-H-H-H-H   =    L--H--R
            | | | | | i|  | | | | | | | |         \i| /
            >->->->->- | -<-<-<-<-<-<-<-<          \|/
                      \|/

        And insert it back into the states ``k`` and ``b``, and thus
        ``TN_energy``.
        """
        uix = self.k.site[i].inds
        lix = self.b.site[i].inds
        dims = self.k.site[i].shape

        if dense:
            # contract remaining hamiltonian and get its dense representation
            eff_ham = (eff_ham ^ '__ham__')['__ham__']
            eff_ham.fuse((('lower', lix), ('upper', uix)), inplace=True)
            op = eff_ham.data
        else:
            op = EffHamOp(eff_ham, dims=dims, upper_inds=uix, lower_inds=lix)

        eff_e, eff_gs = seigsys(op, k=1, which=self.which,
                                v0=self.k.site[i].data)
        eff_gs = eff_gs.A
        self.k.site[i].data = eff_gs
        self.b.site[i].data = eff_gs.conj()

        if (direction == 'right') and (i < self.n - 1):
            self.k.left_canonize_site(i, bra=self.b)
        elif (direction == 'left') and (i > 0):
            self.k.right_canonize_site(i, bra=self.b)
        return eff_e

    def update_local_gs_2site(self, eff_ham, i, direction, dense=False):
        """Find the 2-site effective tensor groundstate of::


                      /| |\
            >->->->->- | | -<-<-<-<-<-<-<-<          /| |\
            | | | | |  | |  | | | | | | | |         / | | \
            H-H-H-H-H--H-H--H-H-H-H-H-H-H-H   =    L--H-H--R
            | | | | |  i i+1| | | | | | | |         \ | | /
            >->->->->- | | -<-<-<-<-<-<-<-<          \| |/
                      \| |/                           i i+1

        And insert it back into the states ``k`` and ``b``, and thus
        ``TN_energy``.
        """
        # Sort out the dims and inds of::
        #
        #   ---O---O---
        #      |   |
        #
        u_bond_ind = self.k.bond(i, i + 1)
        dims_L, uix_L = zip(*(
            (d, ix)
            for d, ix in zip(self.k.site[i].shape, self.k.site[i].inds)
            if ix != u_bond_ind
        ))
        dims_R, uix_R = zip(*(
            (d, ix)
            for d, ix in zip(self.k.site[i + 1].shape, self.k.site[i + 1].inds)
            if ix != u_bond_ind
        ))
        uix = uix_L + uix_R

        l_bond_ind = self.b.bond(i, i + 1)
        lix_L = tuple(i for i in self.b.site[i].inds if i != l_bond_ind)
        lix_R = tuple(i for i in self.b.site[i + 1].inds if i != l_bond_ind)
        lix = lix_L + lix_R

        dims = dims_L + dims_R

        # form the local operator to find ground-state of
        if dense:
            # contract remaining hamiltonian and get its dense representation
            eff_ham = (eff_ham ^ '__ham__')['__ham__']
            eff_ham.fuse((('lower', lix), ('upper', uix)), inplace=True)
            op = eff_ham.data
        else:
            op = EffHamOp(eff_ham, upper_inds=uix, lower_inds=lix, dims=dims)

        # find the 2-site local groundstate
        # XXX: contract sites for v0
        eff_e, eff_gs = seigsys(op, k=1, which=self.which)

        # split the two site local groundstate
        T_AB = Tensor(eff_gs.A.reshape(dims), uix)
        L, R = T_AB.split(left_inds=uix_L, get='arrays', absorb=direction)

        self.k.site[i].update(data=L, inds=(*uix_L, u_bond_ind))
        self.b.site[i].update(data=L.conj(), inds=(*lix_L, l_bond_ind))
        self.k.site[i + 1].update(data=R, inds=(u_bond_ind, *uix_R))
        self.b.site[i + 1].update(data=R.conj(), inds=(l_bond_ind, *lix_R))

        return eff_e

    def update_local_gs(self, eff_ham, i, dense, direction):
        return {
            1: self.update_local_gs_1site,
            2: self.update_local_gs_2site,
        }[self.bsz](eff_ham, i, direction=direction, dense=dense)

    def sweep_right(self, canonize=True, dense=False, verbose=False):
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
        dense : bool, optional
            Solve the inner eigensystem using a dense representation of the
            effective hamiltonian. Can be quicker for small bond_dim.
        """
        if canonize:
            self.k.right_canonize(bra=self.b)

        eff_envs = MovingEnvironment(self.TN_energy, self.n, 'left', self.bsz)

        sweep = range(0, self.n - self.bsz + 1)
        if verbose:
            sweep = progbar(sweep, ncols=80)

        for i in sweep:
            eff_envs.move_to(i)
            en = self.update_local_gs(
                eff_envs(), i, direction='right', dense=dense)

        return en

    def sweep_left(self, canonize=True, dense=False, verbose=False):
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
        dense : bool, optional
            Solve the inner eigensystem using a dense representation of the
            effective hamiltonian. Can be quicker for small bond_dim.
        """
        if canonize:
            self.k.left_canonize(bra=self.b)

        eff_envs = MovingEnvironment(self.TN_energy, self.n, 'right', self.bsz)

        sweep = reversed(range(0, self.n - self.bsz + 1))
        if verbose:
            sweep = progbar(sweep, ncols=80)

        for i in sweep:
            eff_envs.move_to(i)
            en = self.update_local_gs(
                eff_envs(), i, direction='left', dense=dense)

        return en

    def solve(self,
              tol=1e-9,
              max_sweeps=10,
              sweep_sequence='R',
              dense="AUTO",
              compress_opts=None,
              verbose=0):
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
        dense : "AUTO" or bool, optional
            Whether to use a dense representation of the effective hamiltonians
            or a tensor contraction based linear operation representation.
        """
        # choose a rough value at which dense effective ham should not be used
        if dense == "AUTO":
            dense = (self.phys_dim * self.bsz * self.bond_dim**2) < 800

        verbose = {False: 0, True: 1}.get(verbose, verbose)

        compress_opts = {} if compress_opts is None else dict(compress_opts)

        RLs = itertools.cycle(sweep_sequence)
        previous_LR = '0'

        for i in range(max_sweeps):
            LR = next(RLs)
            # if last sweep was opposite direction no need to canonize
            canonize = False if LR + previous_LR in {'LR', 'RL'} else True
            sweep_opts = {'dense': dense,
                          'canonize': canonize,
                          'verbose': verbose}

            if verbose:
                print(f"Sweep {i + 1} to {LR}:")

            if LR == 'R':
                self.energies.append(self.sweep_right(**sweep_opts))
            elif LR == 'L':
                self.energies.append(self.sweep_left(**sweep_opts))

            if verbose:
                if verbose > 1:
                    self.k.plot()
                print(f"Energy: {np.asscalar(self.energies[-1])}", end="")
            if abs(self.energies[-2] - self.energies[-1]) < tol:
                if verbose:
                    print(" - converged!")
                break
            else:
                if verbose:
                    print(" - not converged")

            previous_LR = LR

        return self.energies[-1], self.k


class DMRG1(DMRG):

    def __init__(self, ham, bond_dim, which='SR'):
        super().__init__(ham, bond_dim, which=which, bsz=1)


class DMRG2(DMRG):

    def __init__(self, ham, max_bond_dim, which='SR'):
        super().__init__(ham, bond_dim=max_bond_dim, which=which, bsz=2)


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
        self.k.align(self.ham, self.b, inplace=True)

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
        """Like ``update_local_gs``, but re-insert all eigenvectors, then
        choose the one with best overlap with ``eff_ovlp``.
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

        enrg_envs = MovingEnvironment(self.TN_energy, self.n, start='left')
        ovlp_envs = MovingEnvironment(TN_overlap, self.n, start='left')

        for i in range(0, self.n):
            enrg_envs.move_to(i)
            ovlp_envs.move_to(i)
            en = self.update_with_best_evec(enrg_envs(), ovlp_envs(), i)
            if i < self.n - 1:
                self.k.left_canonize_site(i, bra=self.b)

        return en

    def sweep_left(self, canonize=True):
        self.old_k = self.k.copy().H
        TN_overlap = TensorNetwork([self.k, self.old_k], virtual=True)

        if canonize:
            self.k.left_canonize(bra=self.b)

        enrg_envs = MovingEnvironment(self.TN_energy, self.n, start='right')
        ovlp_envs = MovingEnvironment(TN_overlap, self.n, start='right')

        for i in reversed(range(0, self.n)):
            enrg_envs.move_to(i)
            ovlp_envs.move_to(i)
            en = self.update_with_best_evec(enrg_envs(), ovlp_envs(), i)
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

        for _ in progbar(range(max_sweeps)):
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
