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
        dtype = self.eff_ham_tensors[0].dtype
        self.upper_inds = upper_inds
        self.lower_inds = lower_inds
        self.dims = dims
        self.d = prod(dims)
        super().__init__(dtype=dtype, shape=(self.d, self.d))

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
        """Get the current environment.
        """
        return self.envs[self.pos]


# --------------------------------------------------------------------------- #
#                                  DMRG Base                                  #
# --------------------------------------------------------------------------- #

class DMRG:
    """Single site, fixed bond-dimension variational groundstate search.
    Some initialising arguments act as defaults, but can be overidden with
    each solve or sweep.

    Parameters
    ----------
    ham : MatrixProductOperator
        The hamiltonian in MPO form.
    bond_dims : int or sequence of ints.
        The bond-dimension of the MPS to optimize. If ``bsz > 1``, then this
        corresponds to the maximum bond dimension when splitting the effective
        local groundstate. If a sequence is supplied then successive sweeps
        iterate through, then repeate the final value. E.g.
        ``[16, 32, 64] -> (16, 32, 64, 64, 64, ...)```.
    cutoffs : dict-like
        The cutoff threshold(s) to use when compressing. If a sequence is
        supplied then successive sweeps iterate through, then repeate the final
        value. E.g. ``[1e-5, 1e-7, 1e-9] -> (1e-5, 1e-7, 1e-9, 1e-9, ...)```.
    bsz : {1, 2}
        Number of sites to optimize for locally i.e. DMRG1 or DMRG2.
    which : {'SA', 'LA'}, optional
        Whether to search for smallest or largest real part eigenvectors.

    Attributes
    ----------
    energy : float
        The current most optimized energy.
    state : MatrixProductState
        The current, optimized state.
    energies : list of float
        The list of energies after each sweep.
    opts : dict
        Advanced options e.g. relating to the inner eigensolve or compression.
    """

    def __init__(self, ham, bond_dims,
                 bsz=1, cutoffs=1e-8, which='SA', p0=None):
        self.n = ham.nsites
        self.phys_dim = ham.phys_dim(0)
        self.bsz = bsz
        self.which = which
        self._set_bond_dim_seq(bond_dims)
        self._set_cutoff_seq(cutoffs)

        # create internal states and ham
        if p0 is not None:
            self._k = p0.copy()
        else:
            dtype = ham.site[0].dtype
            self._k = MPS_rand_state(self.n, self._bond_dim0, self.phys_dim,
                                     dtype=dtype)
        self._b = self._k.H
        self.ham = ham.copy()
        self.ham.add_tag("__ham__")

        # Line up and overlap for energy calc
        self._k.align(self.ham, self._b, inplace=True)

        # want to contract this multiple times while
        #   manipulating k/b -> make virtual
        self.TN_energy = self._b | self.ham | self._k
        self.energies = [self.TN_energy ^ ...]

        self.opts = {
            'eff_eig_bkd': "AUTO",
            'eff_eig_tol': 1e-1,
            'eff_eig_ncv': 4,
            'eff_eig_maxiter': None,
            'eff_eig_dense': None,
            'compress_method': 'svd',
            'compress_cutoff_mode': 'sum2',
        }

    def _set_bond_dim_seq(self, bond_dims):
        bds = (bond_dims,) if isinstance(bond_dims, int) else tuple(bond_dims)
        self._bond_dim0 = bds[0]
        self._bond_dims = itertools.chain(bds, itertools.repeat(bds[-1]))

    def _set_cutoff_seq(self, cutoffs):
        bds = (cutoffs,) if isinstance(cutoffs, float) else tuple(cutoffs)
        self._cutoffs = itertools.chain(bds, itertools.repeat(bds[-1]))

    @property
    def energy(self):
        return self.energies[-1]

    @property
    def state(self):
        return self._k.copy()

    # -------------------- standard DMRG update methods --------------------- #

    def _seigsys(self, op, v0=None):
        """Find single eigenpair, using all the internal settings.
        """
        return seigsys(
            op, k=1, which=self.which, v0=v0,
            backend=self.opts['eff_eig_bkd'],
            ncv=self.opts['eff_eig_ncv'],
            tol=self.opts['eff_eig_tol'],
            maxiter=self.opts['eff_eig_maxiter'])

    def update_local_state_1site(self, eff_ham, i, direction, **compress_opts):
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
        uix = self._k.site[i].inds
        lix = self._b.site[i].inds
        dims = self._k.site[i].shape

        # choose a rough value at which dense effective ham should not be used
        dense = self.opts['eff_eig_dense']
        if dense is None:
            dense = prod(dims) < 800

        if dense:
            # contract remaining hamiltonian and get its dense representation
            eff_ham = (eff_ham ^ '__ham__')['__ham__']
            eff_ham.fuse((('lower', lix), ('upper', uix)), inplace=True)
            op = eff_ham.data
        else:
            op = EffHamOp(eff_ham, dims=dims, upper_inds=uix, lower_inds=lix)

        eff_e, eff_gs = self._seigsys(op, v0=self._k.site[i].data)

        eff_gs = eff_gs.A
        self._k.site[i].data = eff_gs
        self._b.site[i].data = eff_gs.conj()

        if (direction == 'right') and (i < self.n - 1):
            self._k.left_compress_site(i, bra=self._b, **compress_opts)
        elif (direction == 'left') and (i > 0):
            self._k.right_compress_site(i, bra=self._b, **compress_opts)

        return eff_e[0]

    def update_local_state_2site(self, eff_ham, i, direction, **compress_opts):
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
        u_bond_ind = self._k.bond(i, i + 1)
        dims_L, uix_L = zip(*(
            (d, ix)
            for d, ix in zip(self._k.site[i].shape, self._k.site[i].inds)
            if ix != u_bond_ind
        ))
        dims_R, uix_R = zip(*(
            (d, ix)
            for d, ix in zip(self._k.site[i + 1].shape,
                             self._k.site[i + 1].inds)
            if ix != u_bond_ind
        ))
        uix = uix_L + uix_R

        l_bond_ind = self._b.bond(i, i + 1)
        lix_L = tuple(i for i in self._b.site[i].inds if i != l_bond_ind)
        lix_R = tuple(i for i in self._b.site[i + 1].inds if i != l_bond_ind)
        lix = lix_L + lix_R

        dims = dims_L + dims_R

        # choose a rough value at which dense effective ham should not be used
        dense = self.opts['eff_eig_dense']
        if dense is None:
            dense = prod(dims) < 800

        # form the local operator to find ground-state of
        if dense:
            # contract remaining hamiltonian and get its dense representation
            eff_ham = (eff_ham ^ '__ham__')['__ham__']
            eff_ham.fuse((('lower', lix), ('upper', uix)), inplace=True)
            op = eff_ham.data
        else:
            op = EffHamOp(eff_ham, upper_inds=uix, lower_inds=lix, dims=dims)

        # find the 2-site local groundstate using previous as initial guess
        v0 = self._k.site[i].contract(self._k.site[i + 1],
                                      output_inds=uix).data

        eff_e, eff_gs = self._seigsys(op, v0=v0)

        # split the two site local groundstate
        T_AB = Tensor(eff_gs.A.reshape(dims), uix)
        L, R = T_AB.split(left_inds=uix_L, get='arrays', absorb=direction,
                          **compress_opts)

        self._k.site[i].update(data=L, inds=(*uix_L, u_bond_ind))
        self._b.site[i].update(data=L.conj(), inds=(*lix_L, l_bond_ind))
        self._k.site[i + 1].update(data=R, inds=(u_bond_ind, *uix_R))
        self._b.site[i + 1].update(data=R.conj(), inds=(l_bond_ind, *lix_R))

        return eff_e[0]

    def update_local_state(self, eff_ham, i, **update_opts):
        return {
            1: self.update_local_state_1site,
            2: self.update_local_state_2site,
        }[self.bsz](eff_ham, i, **update_opts)

    def sweep_right(self, canonize=True, verbose=False, **update_opts):
        """Perform a sweep of optimizations rightwards::

              optimize -->
                ...
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
            self._k.right_canonize(bra=self._b)

        eff_envs = MovingEnvironment(self.TN_energy, self.n, 'left', self.bsz)

        sweep = range(0, self.n - self.bsz + 1)
        if verbose:
            sweep = progbar(sweep, ncols=80)

        for i in sweep:
            eff_envs.move_to(i)
            en = self.update_local_state(
                eff_envs(), i, direction='right', **update_opts)

        return en

    def sweep_left(self, canonize=True, verbose=False, **update_opts):
        """Perform a sweep of optimizations leftwards::

                            <-- optimize
                                      ...
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
            self._k.left_canonize(bra=self._b)

        eff_envs = MovingEnvironment(self.TN_energy, self.n, 'right', self.bsz)

        sweep = reversed(range(0, self.n - self.bsz + 1))
        if verbose:
            sweep = progbar(sweep, ncols=80)

        for i in sweep:
            eff_envs.move_to(i)
            en = self.update_local_state(
                eff_envs(), i, direction='left', **update_opts)

        return en

    def sweep(self, direction, **sweep_opts):
        if direction == 'L':
            en = self.sweep_left(**sweep_opts)
        elif direction == 'R':
            en = self.sweep_right(**sweep_opts)
        return en

    # ----------------- overloadable 'plugin' style methods ----------------- #

    def _compute_pre_sweep(self):
        """Compute this before each sweep.
        """
        pass

    def _print_pre_sweep(self, i, LR, bd, ctf, verbose=0):
        """Print this before each sweep.
        """
        if verbose > 0:
            msg = f"SWEEP-{i + 1}, direction={LR}, max_bond={bd}, cutoff:{ctf}"
            print(msg, flush=True)

    def _compute_post_sweep(self):
        """Compute this after each sweep.
        """
        pass

    def _print_post_sweep(self, converged, verbose=0):
        """Print this after each sweep.
        """
        if verbose > 1:
            self._k.plot()
        if verbose > 0:
            msg = f"Energy: {self.energy} ... " + ("converged!" if converged
                                                   else "not converged")
            print(msg, flush=True)

    def _check_convergence(self, tol):
        """By default check the aboslute change in energy.
        """
        return abs(self.energies[-2] - self.energies[-1]) < tol

    # -------------------------- main solve driver -------------------------- #

    def solve(self,
              tol=1e-8,
              bond_dims=None,
              cutoffs=None,
              sweep_sequence='R',
              max_sweeps=10,
              verbose=0):
        """Solve the system with a sequence of sweeps, up to a certain
        absolute tolerance in the energy or maximum number of sweeps.

        Parameters
        ----------
        tol : float, optional
            The absolute tolerance to converge energy to.
        bond_dims : int or sequence of int
            Overide the initial/current bond_dim sequence.
        cutoffs : float of sequence of float
            Overide the initial/current cutoff sequence.
        sweep_sequence : str, optional
            String made of 'L' and 'R' defining the sweep sequence, e.g 'RRL'.
            The sequence will be repeated until ``max_sweeps`` is reached.
        max_sweeps : int, optional
            The maximum number of sweeps to perform.
        """
        verbose = {False: 0, True: 1}.get(verbose, verbose)

        # Possibly overide the default bond dimension and cutoff sequences.
        if bond_dims is not None:
            self._set_bond_dim_seq(bond_dims)
        if cutoffs is not None:
            self._set_cutoff_seq(cutoffs)

        RLs = itertools.cycle(sweep_sequence)
        previous_LR = '0'

        for i in range(max_sweeps):
            # Get the next direction, bond dimension and cutoff
            LR, bd, ctf = next(RLs), next(self._bond_dims), next(self._cutoffs)
            self._print_pre_sweep(i, LR, bd, ctf, verbose=verbose)

            # if last sweep was in opposite direction no need to canonize
            canonize = False if LR + previous_LR in {'LR', 'RL'} else True

            # need to manually expand bond dimension for DMRG1
            if self.bsz == 1:
                self._k.expand_bond_dimension(bd, bra=self._b)

            # inject all options and defaults
            sweep_opts = {
                'canonize': canonize,
                'max_bond': bd,
                'cutoff': ctf,
                'cutoff_mode': self.opts['compress_cutoff_mode'],
                'method': self.opts['compress_method'],
                'verbose': verbose
            }

            self._compute_pre_sweep()
            self.energies += [self.sweep(direction=LR, **sweep_opts)]
            self._compute_post_sweep()

            converged = self._check_convergence(tol)

            self._print_post_sweep(converged, verbose=verbose)

            if converged:
                break
            previous_LR = LR

        return converged


class DMRG1(DMRG):
    """Simple alias of one site ``DMRG``.
    """
    __doc__ += DMRG.__doc__

    def __init__(self, ham, bond_dims, cutoffs=1e-8, which='SA'):
        super().__init__(ham, bond_dims=bond_dims, cutoffs=cutoffs,
                         which=which, bsz=1)


class DMRG2(DMRG):
    """Simple alias of two site ``DMRG``.
    """
    __doc__ += DMRG.__doc__

    def __init__(self, ham, bond_dims, cutoffs=1e-8, which='SA'):
        super().__init__(ham, bond_dims=bond_dims, cutoffs=cutoffs,
                         which=which, bsz=2)


class DMRGX(DMRG):
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
    bond_dims : int
        The bond dimension to find the state for.

    Attributes
    ----------
    k : MatrixProductState
        The current, optimized state.
    energies : list of float
        The list of energies after each sweep.
    """

    def __init__(self, ham, p0, bond_dims, cutoffs=1e-8, bsz=1, ):
        super().__init__(ham, bond_dims=bond_dims, p0=p0, bsz=bsz,
                         cutoffs=cutoffs)
        # Want to keep track of energy variance as well
        var_ham1 = self.ham.copy()
        var_ham1.upper_ind_id = self._k.site_ind_id
        var_ham1.lower_ind_id = "__var_ham{}__"
        var_ham2 = self.ham.copy()
        var_ham2.upper_ind_id = "__var_ham{}__"
        var_ham2.lower_ind_id = self._b.site_ind_id
        self.TN_en_var2 = self._k | var_ham1 | var_ham2 | self._b
        self.variances = [(self.TN_en_var2 ^ ...) - self.energies[-1]**2]

    @property
    def variance(self):
        return self.variances[-1]

    def update_local_state_1site(self, eff_ham, eff_ovlp, i, direction,
                                 dense=True, **compress_opts):
        """Like ``update_local_state``, but re-insert all eigenvectors, then
        choose the one with best overlap with ``eff_ovlp``.
        """
        # contract remaining hamiltonian and get its dense representation
        eff_ham = (eff_ham ^ '__ham__')['__ham__']
        eff_ham.fuse((('lower', self._b.site[i].inds),
                      ('upper', self._k.site[i].inds)), inplace=True)
        op = eff_ham.data

        # eigen-decompose and reshape eigenvectors thus::
        #
        #    |'__ev_ind__'
        #    E
        #   /|\
        #
        evals, evecs = eigsys(op)
        evecs = np.asarray(evecs).reshape(*self._k.site[i].shape, -1)

        # update tensor at site i with all evecs -> need dummy index
        tnsr = self._k.site[i]
        tnsr.update(data=evecs, inds=(*tnsr.inds, '__ev_ind__'))

        # find the index of the highest overlap eigenvector, by contracting::
        #
        #           |'__ev_ind__'
        #     o-o-o-E-o-o-o-o-o-o-o
        #     | | | | | | | | | | |
        #     0-0-0-0-0-0-0-0-0-0-0  <- state from previous step
        #
        best = np.argmax(np.abs((eff_ovlp ^ ...).data))

        # update site i with the data and drop dummy index too
        tnsr.update(data=evecs[..., best], inds=tnsr.inds[:-1])

        # update the bra -> it only needs the new, conjugated data.
        self._b.site[i].data = evecs[..., best].conj()

        if (direction == 'right') and (i < self.n - 1):
            self._k.left_compress_site(i, bra=self._b, **compress_opts)
        elif (direction == 'left') and (i > 0):
            self._k.right_compress_site(i, bra=self._b, **compress_opts)

        return evals[best]

    def update_local_state_2site(self, eff_ham, eff_ovlp, i, direction,
                                 dense=True, **compress_opts):
        raise NotImplementedError("2-site DMRGX not implemented yet.")

    def update_local_state(self, eff_ham, eff_ovlp, i, **update_opts):
        return {
            1: self.update_local_state_1site,
            2: self.update_local_state_2site,
        }[self.bsz](eff_ham, eff_ovlp, i, **update_opts)

    def sweep_right(self, canonize=True, verbose=False, **update_opts):
        self.old_k = self._k.copy().H
        TN_overlap = TensorNetwork([self._k, self.old_k], virtual=True)

        if canonize:
            self._k.right_canonize(bra=self._b)

        enrg_envs = MovingEnvironment(self.TN_energy, self.n, start='left')
        ovlp_envs = MovingEnvironment(TN_overlap, self.n, start='left')

        sweep = range(0, self.n - self.bsz + 1)
        if verbose:
            sweep = progbar(sweep, ncols=80)

        for i in sweep:
            enrg_envs.move_to(i)
            ovlp_envs.move_to(i)
            en = self.update_local_state(
                enrg_envs(), ovlp_envs(), i, direction='right', **update_opts)

        return en

    def sweep_left(self, canonize=True, **update_opts):
        self.old_k = self._k.copy().H
        TN_overlap = TensorNetwork([self._k, self.old_k], virtual=True)

        if canonize:
            self._k.left_canonize(bra=self._b)

        enrg_envs = MovingEnvironment(self.TN_energy, self.n, start='right')
        ovlp_envs = MovingEnvironment(TN_overlap, self.n, start='right')

        for i in reversed(range(0, self.n)):
            enrg_envs.move_to(i)
            ovlp_envs.move_to(i)
            en = self.update_local_state(
                enrg_envs(), ovlp_envs(), i, direction='left', **update_opts)

        return en

    def _compute_post_sweep(self):
        en_var = (self.TN_en_var2 ^ ...) - self.energies[-1]**2
        self.variances.append(en_var)

    def _print_post_sweep(self, converged, verbose=0):
        if verbose > 1:
            self._k.plot()
        if verbose > 0:
            print(f"Energy={self.energy}, Variance={self.variance}",
                  end="", flush=True)
            if converged:
                print(" ... converged!", flush=True)
            else:
                print(" ... not converged", flush=True)

    def _check_convergence(self, tol):
        return self.variances[-1] < tol
