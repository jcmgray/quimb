"""DMRG-like variational algorithms, but in tensor network language.
"""

import numpy as np
import itertools

from ..utils import progbar
from ..accel import prod
from ..linalg.base_linalg import eigsys, seigsys
from .tensor_core import (
    Tensor,
    TensorNetwork,
    tensor_contract,
    TNLinearOperator,
    find_shared_inds,
)


class MovingEnvironment:
    r"""Helper class for efficiently moving the effective 'environment' of a
    few sites in a 1D tensor network. E.g. for ``begin='left', bsz=2``, this
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
    been shifted many sites into the middle, then ``MovingEnvironment()``
    returns something like::

             <---> bsz sites
             /o-o\
        ooooo | | ooooooo
        HHHHH-H-H-HHHHHHH
        ooooo | | ooooooo
             \o-o/
        0 ... i i+1 ... n-1

    For periodic systems ``MovingEnvironment`` approximates the 'long
    way round' transfer matrices. E.g consider replacing segment B
    (to arbitrary precision) with an SVD::

        /-----------------------------------------------\
        +-A-A-A-A-A-A-A-A-A-A-A-A-B-B-B-B-B-B-B-B-B-B-B-+
          | | | | | | | | | | | | | | | | | | | | | | |           -->
        +-A-A-A-A-A-A-A-A-A-A-A-A-B-B-B-B-B-B-B-B-B-B-B-+
        \-----------------------------------------------/

        +~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~+
        |   /-A-A-A-A-A-A-A-A-A-A-A-A-\   |                       -->
        +~<BL | | | | | | | | | | | | BR>~+
            \-A-A-A-A-A-A-A-A-A-A-A-A-/
              ^                     ^
        segment_start          segment_stop - 1

        +~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~+
        |   /-A-A-\                        |                      -->
        +~<BL | |  AAAAAAAAAAAAAAAAAAAABR>~+
            \-A-A-/
              ...
            <-bsz->

        +~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~+
        |               /-A-A-\           |                       -->
        +~<BLAAAAAAAAAAA  | |  AAAAAAABR>~+
                        \-A-A-/
                          i i+1
             -----sweep--------->

    Can then contract and store left and right environments for efficient
    sweeping just as in non-periodic case. If the segment is long enough (50+)
    sites, often only 1 singular value is needed, and thus the efficiency is
    the same as for OBC.

    Parameters
    ----------
    tn : TensorNetwork
        The 1D tensor network, should be closed, i.e. an overlap of some sort.
    begin : {'left', 'right'}
        Which side to start at and sweep from.
    bsz : int
        The number of sites that form the the 'non-environment',
        e.g. 2 for DMRG2.
    ssz : float or int, optional
        The size of the segment to use, if float, the proportion. Default: 1/2.
    eps : float, optional
        The tolerance to approximate the transfer matrix with. See
        :meth:`~quimb.tensor.TensorNetwork.replace_with_svd`.

    Notes
    -----
    Does not necessarily need to be an operator overlap tensor network. Useful
    for any kind of sweep where only local tensor updates are being made. Note
    that *only* the current site is completely up-to-date and can be modified
    with changes meant to propagate.
    """

    def __init__(self, tn, begin, bsz, cyclic=False, ssz=0.5, eps=1e-8):
        self.n = tn.nsites
        self.begin = begin
        self.bsz = bsz
        self.cyclic = cyclic
        self.eps = eps
        self.tn = tn.copy(virtual=True)
        self.site_tag = lambda i: tn.structure.format(i % self.n)

        if self.cyclic:
            if isinstance(ssz, float):
                self.ssz = int(self.n * ssz)
            else:
                self.ssz = ssz

            start, stop = {'left': (0, self.ssz),
                           'right': (self.n - self.ssz, self.n)}[begin]

        else:
            start, stop = (0, self.n - self.bsz + 1)

        self.init_segment(begin, start, stop)

    def init_segment(self, begin, start, stop):
        """Initialize the environments in ``range(start, stop)`` so that one
        can start sweeping from the side defined by ``begin``.
        """
        if (start >= self.n) or (stop < 0):
            start, stop = start % self.n, stop % self.n

        self.segment = range(start, stop)

        self.prepare_LR_envs(start, stop + self.bsz // 2)

        if begin == 'left':
            self.envs = {stop - 1: self.tnc}
            for i in reversed(range(start, stop - 1)):
                env = self.envs[i + 1].copy(virtual=True)
                env ^= ['_RIGHT', self.site_tag(i + self.bsz)]
                self.envs[i] = env
            self.pos = start

        elif begin == 'right':
            self.envs = {start: self.tnc}
            for i in range(start + 1, stop):
                env = self.envs[i - 1].copy(virtual=True)
                env ^= ['_LEFT', self.site_tag(i - 1)]
                self.envs[i] = env
            self.pos = stop - 1

        else:
            raise ValueError("``begin`` must be 'left' or 'right'.")

    def prepare_LR_envs(self, start, stop):
        """Compress and label the effective env not in ``range(start, stop)``
        if cyclic, else just add some dummy left and right end pieces.
        """
        self.tnc = self.tn.copy(virtual=True)
        ltags = {'_LEFT', *self.tnc.select(start).tags}
        rtags = {'_RIGHT', *self.tnc.select(stop - 1).tags}

        if not self.cyclic:
            # generate dummy left and right envs
            self.tnc |= Tensor(1.0, (), ltags)
            self.tnc |= Tensor(1.0, (), rtags)
            # self.tnc.add_tag('_LEFT', 0, mode='any')
            # self.tnc.add_tag('_RIGHT', -1, mode='any')
            return

        lix = find_shared_inds(self.tnc[start - 1], self.tnc[start])
        where = slice(start, stop)
        self.tnc.replace_with_svd(where, left_inds=lix, eps=self.eps,
                                  mode='!any', ltags=ltags, rtags=rtags,
                                  inplace=True, method='isvd', keep_tags=False)

    def move_right(self):
        i = (self.pos + 1) % self.n

        # generate a new segment if we go over the border
        if i not in self.segment:
            if not self.cyclic:
                raise ValueError("For OBC, ``0 <= position <= n - bsz``.")
            self.init_segment('left', i, i + self.ssz)
        else:
            self.pos = i % self.n

        i0 = self.segment.start

        if i >= i0 + 2:
            # delete the old left environment
            where = ['_LEFT'] + [self.site_tag(i) for i in range(i0, i - 1)]
            self.envs[i].delete(where, mode='any')

            # insert the updated left env from previous step
            self.envs[i] |= self.envs[i - 1]['_LEFT']

        if i >= i0 + 1:
            # contract left env with updated site just to left
            self.envs[i] ^= ['_LEFT', self.site_tag(i - 1)]

    def move_left(self):
        i = (self.pos - 1) % self.n

        # generate a new segment if we go over the border
        if i not in self.segment:
            if not self.cyclic:
                raise ValueError("For OBC, ``0 <= position <= n - bsz``.")
            self.init_segment('right', i - self.ssz + 1, i + 1)
        else:
            self.pos = i % self.n

        iN = self.segment.stop

        if i <= iN - 3:
            # delete the old right environment
            where = ['_RIGHT'] + [self.site_tag(i) for i in
                                  range(i + self.bsz + 1, iN + self.bsz - 1)]
            self.envs[i].delete(where, mode='any')

            # insert the updated right env from previous step
            self.envs[i] |= self.envs[i + 1]['_RIGHT']

        if i <= iN - 2:
            # contract right env with updated site just to right
            self.envs[i] ^= ['_RIGHT', self.site_tag(i + self.bsz)]

    def move(self, direction):
        {'left': self.move_left, 'right': self.move_right}[direction]()

    def move_to(self, i):

        if self.cyclic:
            # to take account of PBC, rescale so that current pos == n // 2,
            #     then work out if desired i is lower or higher
            ri = (i + (self.n // 2 - self.pos)) % self.n
            direction = 'left' if ri <= self.n // 2 else 'right'
        else:
            direction = 'left' if i < self.pos else 'right'

        while self.pos != i % self.n:
            self.move(direction)

    def __call__(self):
        """Get the current environment.
        """
        return self.envs[self.pos]


# --------------------------------------------------------------------------- #
#                                  DMRG Base                                  #
# --------------------------------------------------------------------------- #

def parse_2site_inds_dims(k, b, i):
    r"""Sort out the dims and inds of::

        ---O---O---
           |   |

    For use in 2 site algorithms.
    """
    u_bond_ind = k.bond(i, i + 1)
    dims_L, uix_L = zip(*(
        (d, ix)
        for d, ix in zip(k[i].shape, k[i].inds)
        if ix != u_bond_ind
    ))
    dims_R, uix_R = zip(*(
        (d, ix)
        for d, ix in zip(k[i + 1].shape,
                         k[i + 1].inds)
        if ix != u_bond_ind
    ))
    uix = uix_L + uix_R

    l_bond_ind = b.bond(i, i + 1)
    lix_L = tuple(i for i in b[i].inds if i != l_bond_ind)
    lix_R = tuple(i for i in b[i + 1].inds if i != l_bond_ind)
    lix = lix_L + lix_R

    dims = dims_L + dims_R

    return dims, lix_L, lix_R, lix, uix_L, uix_R, uix, l_bond_ind, u_bond_ind


class DMRG:
    r"""Density Matrix Renormalization Group variational groundstate search.
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
        ``[16, 32, 64] -> (16, 32, 64, 64, 64, ...)``.
    cutoffs : dict-like
        The cutoff threshold(s) to use when compressing. If a sequence is
        supplied then successive sweeps iterate through, then repeate the final
        value. E.g. ``[1e-5, 1e-7, 1e-9] -> (1e-5, 1e-7, 1e-9, 1e-9, ...)``.
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
                 bsz=2, cutoffs=1e-9, which='SA', p0=None):
        self.n = ham.nsites
        self.phys_dim = ham.phys_dim()
        self.bsz = bsz
        self.which = which
        self.cyclic = ham.cyclic
        self._set_bond_dim_seq(bond_dims)
        self._set_cutoff_seq(cutoffs)

        # create internal states and ham
        if p0 is not None:
            self._k = p0.copy()
        else:
            self._k = ham.rand_state(self._bond_dim0)
        self._b = self._k.H
        self.ham = ham.copy()
        self._k.add_tag("_KET")
        self._b.add_tag("_BRA")
        self.ham.add_tag("_HAM")

        # Line up and overlap for energy calc
        self._k.align(self.ham, self._b, inplace=True)

        # want to contract this multiple times while
        #   manipulating k/b -> make virtual
        self.TN_energy = self._b | self.ham | self._k
        self.energies = [self.TN_energy ^ ...]

        # if cyclic need to keep track of normalization
        if self.cyclic:
            eye = self.ham.identity()
            eye.add_tag('_EYE')
            self.TN_norm = self._b | eye | self._k

        self.opts = {
            'eff_eig_tol': 1e-3,
            'eff_eig_ncv': 4,
            'eff_eig_bkd': None,
            'eff_eig_maxiter': None,
            'eff_eig_dense': None,
            'eff_eig_EPSType': 'krylovschur',
            'compress_method': 'svd',
            'compress_cutoff_mode': 'sum2',
            'default_sweep_sequence': 'R',
            'bond_expand_rand_strength': 1e-9,
            'periodic_compress_tol': 1e-8,
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

    def _compress_after_1site_update(self, direction, i, **compress_opts):
        """Compress a site having updated it.
        """
        if (direction == 'right') and (i < self.n - 1):
            self._k.left_compress_site(i, bra=self._b, **compress_opts)
        elif (direction == 'left') and (i > 0):
            self._k.right_compress_site(i, bra=self._b, **compress_opts)

    def _seigsys(self, A, B=None, v0=None):
        """Find single eigenpair, using all the internal settings.
        """
        backend = self.opts['eff_eig_bkd']
        if backend in (None, 'AUTO') and B is not None:
            backend = 'scipy'

        return seigsys(
            A, k=1, B=B, which=self.which, v0=v0,
            backend=backend,
            EPSType=self.opts['eff_eig_EPSType'],
            ncv=self.opts['eff_eig_ncv'],
            tol=self.opts['eff_eig_tol'],
            maxiter=self.opts['eff_eig_maxiter'])

    def _update_local_state_1site(self, i, direction, **compress_opts):
        r"""Find the single site effective tensor groundstate of::

            >->->->->-/|\-<-<-<-<-<-<-<-<          /|\
            | | | | |  |  | | | | | | | |         / | \
            H-H-H-H-H--H--H-H-H-H-H-H-H-H   =    L--H--R
            | | | | | i|  | | | | | | | |         \i| /
            >->->->->-\|/-<-<-<-<-<-<-<-<          \|/

        And insert it back into the states ``k`` and ``b``, and thus
        ``TN_energy``.
        """
        uix, lix = self._k[i].inds, self._b[i].inds
        dims = self._k[i].shape

        # choose a rough value at which dense effective ham should not be used
        dense = self.opts['eff_eig_dense']
        if dense is None:
            dense = prod(dims) < 800

        if self.cyclic:
            B = (self._eff_norm ^ '_EYE')['_EYE'].to_dense(lix, uix)
            # B = TNLinearOperator(self._eff_norm['_EYE'], ldims=dims,
            #                      rdims=dims, left_inds=lix, right_inds=uix)
            # B = (B.conj().T + B) / 2
            B += 1e-12 * np.eye(B.shape[0])
        else:
            B = None

        if dense:
            # contract remaining hamiltonian and get its dense representation
            A = (self._eff_ham ^ '_HAM')['_HAM'].to_dense(lix, uix)
        else:
            A = TNLinearOperator(self._eff_ham['_HAM'], ldims=dims, rdims=dims,
                                 left_inds=lix, right_inds=uix)

        eff_e, eff_gs = self._seigsys(A, B=B, v0=self._k[i].data.ravel())

        eff_gs = eff_gs.A
        self._k[i].data = eff_gs
        self._b[i].data = eff_gs.conj()

        self._compress_after_1site_update(direction, i, **compress_opts)
        return eff_e[0]

    def _update_local_state_2site(self, i, direction, **compress_opts):
        r"""Find the 2-site effective tensor groundstate of::

            >->->->->-/| |\-<-<-<-<-<-<-<-<          /| |\
            | | | | |  | |  | | | | | | | |         / | | \
            H-H-H-H-H--H-H--H-H-H-H-H-H-H-H   =    L--H-H--R
            | | | | |  i i+1| | | | | | | |         \ | | /
            >->->->->-\| |/-<-<-<-<-<-<-<-<          \| |/
                                                 i i+1

        And insert it back into the states ``k`` and ``b``, and thus
        ``TN_energy``.
        """
        dims, lix_L, lix_R, lix, uix_L, uix_R, uix, l_bond_ind, u_bond_ind = \
            parse_2site_inds_dims(self._k, self._b, i)

        # choose a rough value at which dense effective ham should not be used
        dense = self.opts['eff_eig_dense']
        if dense is None:
            dense = prod(dims) < 800

        if self.cyclic:
            B = (self._eff_norm ^ '_EYE')['_EYE'].to_dense(lix, uix)
            # B = TNLinearOperator(self._eff_norm['_EYE'], ldims=dims,
            #                      rdims=dims, left_inds=lix, right_inds=uix)
            # B = (B.conj().T + B) / 2
            B += 1e-12 * np.eye(B.shape[0])
        else:
            B = None

        # form the local operator to find ground-state of
        if dense:
            # contract remaining hamiltonian and get its dense representation
            A = (self._eff_ham ^ '_HAM')['_HAM'].to_dense(lix, uix)
        else:
            A = TNLinearOperator(self._eff_ham['_HAM'], ldims=dims, rdims=dims,
                                 left_inds=lix, right_inds=uix)

        # find the 2-site local groundstate using previous as initial guess
        v0 = self._k[i].contract(self._k[i + 1], output_inds=uix).data.ravel()

        eff_e, eff_gs = self._seigsys(A, B=B, v0=v0)

        # split the two site local groundstate
        T_AB = Tensor(eff_gs.A.reshape(dims), uix)
        L, R = T_AB.split(left_inds=uix_L, get='arrays', absorb=direction,
                          **compress_opts)

        self._k[i].modify(data=L, inds=(*uix_L, u_bond_ind))
        self._b[i].modify(data=L.conj(), inds=(*lix_L, l_bond_ind))
        self._k[i + 1].modify(data=R, inds=(u_bond_ind, *uix_R))
        self._b[i + 1].modify(data=R.conj(), inds=(l_bond_ind, *lix_R))

        return eff_e[0]

    def _update_local_state(self, i, **update_opts):
        return {
            1: self._update_local_state_1site,
            2: self._update_local_state_2site,
        }[self.bsz](i, **update_opts)

    def sweep(self, direction, canonize=True, verbose=False, **update_opts):
        r"""Perform a sweep of optimizations rightwards (`direction='R'`)::

              optimize -->
                ...
            >->-o-<-<-<-<-<-<-<-<-<-<-<-<-<
            | | | | | | | | | | | | | | | |
            H-H-H-H-H-H-H-H-H-H-H-H-H-H-H-H
            | | | | | | | | | | | | | | | |
            >->-o-<-<-<-<-<-<-<-<-<-<-<-<-<

        or leftwards (`direction='L'`)::

                            <-- optimize
                                      ...
            >->->->->->->->->->->->->-o-<-<
            | | | | | | | | | | | | | | | |
            H-H-H-H-H-H-H-H-H-H-H-H-H-H-H-H
            | | | | | | | | | | | | | | | |
            >->->->->->->->->->->->->-o-<-<

        After the sweep the state is left or right canonized respectively.

        Parameters
        ----------
        direction : {'R', 'L'}
            Sweep from left to right (->) or right to left (<-) respectively.
        canonize : bool, optional
            Canonize the state first, not needed if doing alternate sweeps.
        verbose : bool, optional
            Show a progress bar for the sweep.
        update_opts :
            Supplied to ``self._update_local_state``.
        """
        if canonize:
            {'R': self._k.right_canonize,
             'L': self._k.left_canonize}[direction](bra=self._b)

        direction, eff_start, sweep = {
            'R': ('right', 'left', range(0, self.n - self.bsz + 1)),
            'L': ('left', 'right', reversed(range(0, self.n - self.bsz + 1))),
        }[direction]

        eff_args = {'begin': eff_start, 'bsz': self.bsz, 'cyclic': self.cyclic}
        eff_hams = MovingEnvironment(self.TN_energy, **eff_args)
        if self.cyclic:
            eff_norms = MovingEnvironment(self.TN_norm, **eff_args)

        if verbose:
            sweep = progbar(sweep, ncols=80, total=self.n - self.bsz + 1)

        for i in sweep:
            if self.cyclic:
                # self._k.canonize_cyclic(slice(i, i + self.bsz), bra=self._b)
                eff_norms.move_to(i)
                self._eff_norm = eff_norms()

            eff_hams.move_to(i)
            self._eff_ham = eff_hams()

            en = self._update_local_state(
                i, direction=direction, **update_opts)

        return en

    def sweep_right(self, canonize=True, verbose=False, **update_opts):
        return self.sweep(direction='R', canonize=canonize,
                          verbose=verbose, **update_opts)

    def sweep_left(self, canonize=True, verbose=False, **update_opts):
        return self.sweep(direction='L', canonize=canonize,
                          verbose=verbose, **update_opts)

    # ----------------- overloadable 'plugin' style methods ----------------- #

    def _print_pre_sweep(self, i, LR, bd, ctf, verbose=0):
        """Print this before each sweep.
        """
        if verbose > 0:
            msg = "SWEEP-{}, direction={}, max_bond={}, cutoff:{}"
            print(msg.format(i + 1, LR, bd, ctf), flush=True)

    def _compute_post_sweep(self):
        """Compute this after each sweep.
        """
        pass

    def _print_post_sweep(self, converged, verbose=0):
        """Print this after each sweep.
        """
        if verbose > 1:
            self._k.show()
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
              sweep_sequence=None,
              max_sweeps=8,
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

        # Possibly overide the default bond dimension, cutoff, LR sequences.
        if bond_dims is not None:
            self._set_bond_dim_seq(bond_dims)
        if cutoffs is not None:
            self._set_cutoff_seq(cutoffs)
        if sweep_sequence is None:
            sweep_sequence = self.opts['default_sweep_sequence']

        RLs = itertools.cycle(sweep_sequence)
        previous_LR = '0'

        for i in range(max_sweeps):
            # Get the next direction, bond dimension and cutoff
            LR, bd, ctf = next(RLs), next(self._bond_dims), next(self._cutoffs)
            if LR == 'X':
                import random
                LR = random.choice(('R', 'L'))
            self._print_pre_sweep(i, LR, bd, ctf, verbose=verbose)

            # if last sweep was in opposite direction no need to canonize
            canonize = False if LR + previous_LR in {'LR', 'RL'} else True
            # need to manually expand bond dimension for DMRG1
            if self.bsz == 1:
                self._k.expand_bond_dimension(
                    bd, bra=self._b,
                    rand_strength=self.opts['bond_expand_rand_strength'])

            # inject all options and defaults
            sweep_opts = {
                'canonize': canonize,
                'max_bond': bd,
                'cutoff': ctf,
                'cutoff_mode': self.opts['compress_cutoff_mode'],
                'method': self.opts['compress_method'],
                'verbose': verbose
            }

            # perform sweep, computations and convergence test
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

    def __init__(self, ham, which='SA',
                 bond_dims=(8, 16, 32, 64), cutoffs=1e-8):
        super().__init__(ham, bond_dims=bond_dims, cutoffs=cutoffs,
                         which=which, bsz=1)


class DMRG2(DMRG):
    """Simple alias of two site ``DMRG``.
    """
    __doc__ += DMRG.__doc__

    def __init__(self, ham, which='SA',
                 bond_dims=(10, 20, 50, 100), cutoffs=1e-8):
        super().__init__(ham, bond_dims=bond_dims, cutoffs=cutoffs,
                         which=which, bsz=2)


# --------------------------------------------------------------------------- #
#                                    DMRGX                                    #
# --------------------------------------------------------------------------- #

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
    bond_dims : int or sequence of int
        See :class:`DMRG`.
    cutoffs : float or sequence of float
        See :class:`DMRG`.

    Attributes
    ----------
    k : MatrixProductState
        The current, optimized state.
    energies : list of float
        The list of energies after each sweep.
    """

    def __init__(self, ham, p0, bond_dims, cutoffs=1e-8, bsz=1):
        super().__init__(ham, bond_dims=bond_dims, p0=p0, bsz=bsz,
                         cutoffs=cutoffs)
        # Want to keep track of energy variance as well
        var_ham1 = self.ham.copy()
        var_ham2 = self.ham.copy()
        var_ham1.upper_ind_id = self._k.site_ind_id
        var_ham1.lower_ind_id = "__ham2{}__"
        var_ham2.upper_ind_id = "__ham2{}__"
        var_ham2.lower_ind_id = self._b.site_ind_id
        self.TN_energy2 = self._k | var_ham1 | var_ham2 | self._b
        self.variances = [(self.TN_energy2 ^ ...) - self.energies[-1]**2]
        self._target_energy = self.energies[-1]

        self.opts = {
            'eff_eig_partial_cutoff': 2**11,
            'eff_eig_partial_k': 0.02,
            'eff_eig_tol': 1e-1,
            'overlap_thresh': 2 / 3,
            'compress_method': 'svd',
            'compress_cutoff_mode': 'sum2',
            'default_sweep_sequence': 'RRLL',
            'bond_expand_rand_strength': 1e-9,
        }

    @property
    def variance(self):
        return self.variances[-1]

    def _update_local_state_1site_dmrgx(self, i, direction, **compress_opts):
        """Like ``_update_local_state``, but re-insert all eigenvectors, then
        choose the one with best overlap with ``eff_ovlp``.
        """
        uix = self._k[i].inds
        lix = self._b[i].inds
        dims = self._k[i].shape

        # contract remaining hamiltonian and get its dense representation
        A = (self._eff_ham ^ '_HAM')['_HAM'].to_dense(lix, uix)

        # eigen-decompose and reshape eigenvectors thus::
        #
        #    |'__ev_ind__'
        #    E
        #   /|\
        #
        D = prod(dims)
        if D <= self.opts['eff_eig_partial_cutoff']:
            evals, evecs = eigsys(A)
        else:
            if isinstance(self.opts['eff_eig_partial_k'], float):
                k = int(self.opts['eff_eig_partial_k'] * D)
            else:
                k = self.opts['eff_eig_partial_k']

            evals, evecs = seigsys(
                A, sigma=self._target_energy, v0=self._k[i].data,
                k=k, tol=self.opts['eff_eig_tol'], backend='scipy')

        evecs = np.asarray(evecs).reshape(*dims, -1)
        evecs_c = evecs.conj()

        # update tensor at site i with all evecs -> need dummy index
        ki = self._k[i]
        bi = self._b[i]
        ki.modify(data=evecs, inds=(*uix, '__ev_ind__'))

        # find the index of the highest overlap eigenvector, by contracting::
        #
        #           |'__ev_ind__'
        #     o-o-o-E-o-o-o-o-o-o-o
        #     | | | | | | | | | | |
        #     0-0-0-0-0-0-0-0-0-0-0  <- state from previous step
        #
        # choose the eigenvectors with best overlap
        overlaps = np.abs((self._eff_ovlp ^ ...).data)

        if self.opts['overlap_thresh'] == 1:
            # just choose the maximum overlap state
            best = np.argmax(overlaps)
        else:
            # else simulteneously reduce energy variance as well
            best_overlaps, = np.where(
                overlaps > np.max(overlaps) * self.opts['overlap_thresh'])

            if len(best_overlaps) == 1:
                # still only one good overlapping eigenvector -> choose that
                best, = best_overlaps
            else:
                # reduce down to the candidate eigenpairs
                evals = evals[best_overlaps]
                evecs = evecs[..., best_overlaps]
                evecs_c = evecs_c[..., best_overlaps]

                # need bra site in place with extra dimension to calc variance
                ki.modify(data=evecs)
                bi.modify(data=evecs_c, inds=(*lix, '__ev_ind__'))

                # now find the variances of the best::
                #
                #           |'__ev_ind__'
                #     o-o-o-E-o-o-o                |'__ev_ind__'  ^2
                #     | | | | | | |          o-o-o-E-o-o-o
                #     H-H-H-H-H-H-H          | | | | | | |
                #     | | | | | | |    -     H-H-H-H-H-H-H
                #     H-H-H-H-H-H-H          | | | | | | |
                #     | | | | | | |          o-o-o-E-o-o-o
                #     o-o-o-E-o-o-o                |'__ev_ind__'
                #           |'__ev_ind__'
                #
                # use einsum notation to get diagonal of left hand term
                en2 = tensor_contract(*self._eff_ham2.tensors,
                                      output_inds=['__ev_ind__']).data

                # then find minimum variance
                best = np.argmin(en2 - evals**2)

        # update site i with the data and drop dummy index too
        ki.modify(data=evecs[..., best], inds=uix)
        bi.modify(data=evecs_c[..., best], inds=lix)
        # store the current effective energy for possibly targeted seigsys
        self._target_energy = evals[best]

        self._compress_after_1site_update(direction, i, **compress_opts)
        return evals[best]

    def _update_local_state_2site_dmrgx(self, i, direction, **compress_opts):
        raise NotImplementedError("2-site DMRGX not implemented yet.")
        dims, lix_L, lix_R, lix, uix_L, uix_R, uix, l_bond_ind, u_bond_ind = \
            parse_2site_inds_dims(self._k, self._b, i)

        # contract remaining hamiltonian and get its dense representation
        eff_ham = (self._eff_ham ^ '_HAM')['_HAM']
        eff_ham.fuse((('lower', lix), ('upper', uix)), inplace=True)
        A = eff_ham.data

        # eigen-decompose and reshape eigenvectors thus::
        #
        #    ||'__ev_ind__'
        #    EE
        #   /||\
        #
        D = prod(dims)
        if D <= self.opts['eff_eig_partial_cutoff']:
            evals, evecs = eigsys(A)
        else:
            if isinstance(self.opts['eff_eig_partial_k'], float):
                k = int(self.opts['eff_eig_partial_k'] * D)
            else:
                k = self.opts['eff_eig_partial_k']

            # find the 2-site local state using previous as initial guess
            v0 = self._k[i].contract(self._k[i + 1], output_inds=uix).data

            evals, evecs = seigsys(
                A, sigma=self.energies[-1], v0=v0,
                k=k, tol=self.opts['eff_eig_tol'], backend='scipy')

    def _update_local_state_dmrgx(self, i, **update_opts):
        return {
            1: self._update_local_state_1site_dmrgx,
            2: self._update_local_state_2site_dmrgx,
        }[self.bsz](i, **update_opts)

    def sweep(self, direction, canonize=True, verbose=False, **update_opts):
        """Perform a sweep of the algorithm.

        Parameters
        ----------
        direction : {'R', 'L'}
            Sweep from left to right (->) or right to left (<-) respectively.
        canonize : bool, optional
            Canonize the state first, not needed if doing alternate sweeps.
        verbose : bool, optional
            Show a progress bar for the sweep.
        update_opts :
            Supplied to ``self._update_local_state``.
        """
        old_k = self._k.copy().H
        TN_overlap = TensorNetwork([self._k, old_k], virtual=True)

        if canonize:
            {'R': self._k.right_canonize,
             'L': self._k.left_canonize}[direction](bra=self._b)

        direction, eff_start, sweep = {
            'R': ('right', 'left', range(0, self.n - self.bsz + 1)),
            'L': ('left', 'right', reversed(range(0, self.n - self.bsz + 1))),
        }[direction]

        eff_args = {'begin': eff_start, 'bsz': self.bsz, 'cyclic': self.cyclic}
        eff_hams = MovingEnvironment(self.TN_energy, **eff_args)
        eff_ham2s = MovingEnvironment(self.TN_energy2, **eff_args)
        eff_ovlps = MovingEnvironment(TN_overlap, **eff_args)

        if verbose:
            sweep = progbar(sweep, ncols=80, total=self.n - self.bsz + 1)

        for i in sweep:
            eff_hams.move_to(i)
            eff_ham2s.move_to(i)
            eff_ovlps.move_to(i)
            self._eff_ham = eff_hams()
            self._eff_ovlp = eff_ovlps()
            self._eff_ham2 = eff_ham2s()
            en = self._update_local_state_dmrgx(
                i, direction=direction, **update_opts)

        return en

    def _compute_post_sweep(self):
        en_var = (self.TN_energy2 ^ ...) - self.energies[-1]**2
        self.variances.append(en_var)

    def _print_post_sweep(self, converged, verbose=0):
        if verbose > 1:
            self._k.show()
        if verbose > 0:
            msg = (f"Energy={self.energy}, Variance={self.variance} ... " +
                   "converged!" if converged else "not converged")
            print(msg, flush=True)

    def _check_convergence(self, tol):
        return self.variance < tol
