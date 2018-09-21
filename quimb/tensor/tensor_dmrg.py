"""DMRG-like variational algorithms, but in tensor network language.
"""

import itertools
import numpy as np

from ..utils import progbar
from ..core import prod
from ..linalg.base_linalg import eigh, IdentityLinearOperator
from .tensor_core import (
    Tensor,
    TensorNetwork,
    tensor_contract,
    TNLinearOperator,
    _asarray,
)


def get_default_opts(cyclic=False):
    """Get the default advanced settings for DMRG.

    Returns
    -------
    default_sweep_sequence : str
        How to sweep. Will be repeated, e.g. "RRL" -> RRLRRLRRL..., default: R.
    bond_compress_method : {'svd', 'eig', ...}
        Method used to compress sites after update.
    bond_compress_cutoff_mode : {'sum2', 'abs', 'rel'}
        How to perform compression truncation.
    bond_expand_rand_strength : float
        In DMRG1, strength of randomness to expand bonds with. Needed to avoid
        singular matrices after expansion.
    local_eig_tol : float
        Relative tolerance to solve inner eigenproblem to, larger = quicker but
        more unstable, default: 1e-3. Note this can be much looser than the
        overall tolerance, the starting point for each local solve is the
        previous state, and the overall accuracy comes from multiple sweeps.
    local_eig_ncv : int
        Number of inner eigenproblem lanczos vectors. Smaller can mean quicker.
    local_eig_backend : {None, 'AUTO', 'SCIPY', 'SLEPC'}
        Which to backend to use for the inner eigenproblem. None or 'AUTO' to
        choose best. Generally ``'SLEPC'`` best if available for large
        problems, but it can't currently handle ``LinearOperator`` Neff as well
        as ``'lobpcg'``.
    local_eig_maxiter : int
        Maximum number of inner eigenproblem iterations.
    local_eig_ham_dense : bool
        Force dense representation of the effective hamiltonian.
    local_eig_EPSType : {'krylovschur', 'gd', 'jd', ...}
        Eigensovler tpye if ``local_eig_backend='slepc'``.
    local_eig_norm_dense : bool
        Force dense representation of the effective norm.
    periodic_segment_size : float or int
        How large (as a proportion if float) to make the 'segments' in periodic
        DMRG. During a sweep everything outside this (the 'long way round') is
        compressed so the effective energy and norm can be efficiently formed.
        Tradeoff: longer segments means having to compress less, but also
        having a shorter 'long way round', meaning that it needs a larger bond
        to represent it and can be 'pseudo-orthogonalized' less effectively.
        0.5 is the largest fraction that makes sense. Set to >= 1.0 to not
        use segmentation at all, which is better for small systems.
    periodic_compress_method : {'isvd', 'svds'}
        Which method to perform the transfer matrix compression with.
    periodic_compress_norm_eps : float
        Precision to compress the norm transfer matrix in periodic systems.
    periodic_compress_ham_eps : float
        Precision to compress the energy transfer matrix in periodic systems.
    periodic_compress_max_bond : int
        The maximum bond to use when compressing transfer matrices.
    periodic_nullspace_fudge_factor : float
        Factor to add to ``Heff`` and ``Neff`` to remove nullspace.
    periodic_canonize_inv_tol : float
        When psuedo-orthogonalizing, an inverse gauge is generated that can be
        very ill-conditioned. This factor controls cutting off the small
        singular values of the gauge to stop this.
    periodic_orthog_tol : float
        When psuedo-orthogonalizing, if the local norm is within this
        distance to 1 (pseudo-orthogonoalized), then the generalized eigen
        decomposition is *not* used, which is much more efficient. If set too
        large the total normalization can become unstable.
    """
    return {
        'default_sweep_sequence': 'R',
        'bond_compress_method': 'svd',
        'bond_compress_cutoff_mode': 'rel' if cyclic else 'sum2',
        'bond_expand_rand_strength': 1e-6,
        'local_eig_tol': 1e-3,
        'local_eig_ncv': 4,
        'local_eig_backend': None,
        'local_eig_maxiter': None,
        'local_eig_EPSType': None,
        'local_eig_ham_dense': None,
        'local_eig_norm_dense': None,
        'periodic_segment_size': 1 / 2,
        'periodic_compress_method': 'isvd',
        'periodic_compress_norm_eps': 1e-6,
        'periodic_compress_ham_eps': 1e-6,
        'periodic_compress_max_bond': -1,
        'periodic_nullspace_fudge_factor': 1e-12,
        'periodic_canonize_inv_tol': 1e-10,
        'periodic_orthog_tol': 1e-6,
    }


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
          | | | | | | | | | | | | | | | | | | | | | | |           ==>
        +-A-A-A-A-A-A-A-A-A-A-A-A-B-B-B-B-B-B-B-B-B-B-B-+
        \-----------------------------------------------/

        +~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~+
        |   /-A-A-A-A-A-A-A-A-A-A-A-A-\   |                       ==>
        +~<BL | | | | | | | | | | | | BR>~+
            \-A-A-A-A-A-A-A-A-A-A-A-A-/
              ^                     ^
        segment_start          segment_stop - 1

        +~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~+
        |   /-A-A-\                        |                      ==>
        +~<BL | |  AAAAAAAAAAAAAAAAAAAABR>~+
            \-A-A-/
              ...
            <-bsz->

        +~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~+
        |               /-A-A-\           |                       ==>
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
    cyclic : bool, optional
        Whether this is a periodic ``MovingEnvironment``.
    segment_callbacks : sequence of callable, optional
        Functions with signature ``callback(start, stop, self.begin)``, to be
        called every time a new segment is initialized.
    method : {'isvd', 'svds', ...}, optional
        How to performt the transfer matrix compression. See
        :meth:`~quimb.tensor.TensorNetwork.replace_with_svd`.
    max_bond : , optional
        If > 0, the maximum bond of the compressed transfer matrix.
    norm : bool, optional
        If True, treat this ``MovingEnvironment`` as the state overlap, which
        enables a few extra checks.

    Notes
    -----
    Does not necessarily need to be an operator overlap tensor network. Useful
    for any kind of sweep where only local tensor updates are being made. Note
    that *only* the current site is completely up-to-date and can be modified
    with changes meant to propagate.
    """

    def __init__(self, tn, begin, bsz, *, cyclic=False, segment_callbacks=None,
                 ssz=0.5, eps=1e-8, method='isvd', max_bond=-1, norm=False):

        self.tn = tn.copy(virtual=True)
        self.begin = begin
        self.bsz = bsz
        self.cyclic = cyclic

        if callable(segment_callbacks):
            self.segment_callbacks = (segment_callbacks,)
        else:
            self.segment_callbacks = segment_callbacks

        self.n = tn.nsites
        self.structure = tn.structure

        if self.cyclic:
            self.eps = eps
            self.method = method
            self.max_bond = max_bond
            self.norm = norm
            self.bond_sizes = []

            if isinstance(ssz, float):
                # this logic essentially makes sure that segments prefer
                #     overshooting e.g ssz=1/3 with n=100 produces segments of
                #     length 34, to avoid a final segement of length 1.
                self._ssz = int(self.n * ssz + self.n % int(1 / ssz))
            else:
                self._ssz = ssz

            self.segmented = self._ssz < self.n
            # will still split system in half but no compression or callbacks
            if not self.segmented:
                self._ssz = int(self.n / 2 + self.n % 2)

            start, stop = {
                'left': (0, self._ssz),
                'right': (self.n - self._ssz, self.n)
            }[begin]
        else:
            self.segmented = False
            start, stop = (0, self.n - self.bsz + 1)

        self.init_segment(begin, start, stop)

    def site_tag(self, i):
        return self.structure.format(i % self.n)

    def init_segment(self, begin, start, stop):
        """Initialize the environments in ``range(start, stop)`` so that one
        can start sweeping from the side defined by ``begin``.
        """
        if (start >= self.n) or (stop < 0):
            start, stop = start % self.n, stop % self.n

        self.segment = range(start, stop)
        self.init_non_segment(start, stop + self.bsz // 2)

        if begin == 'left':

            tags_initital = ['_RIGHT'] + [self.site_tag(stop - 1 + b)
                                          for b in range(self.bsz)]
            self.envs = {stop - 1: self.tnc.select(tags_initital, which='any')}

            for i in reversed(range(start, stop - 1)):
                # add a new site to previous env, and contract one site
                self.envs[i] = self.envs[i + 1].copy(virtual=True)
                self.envs[i] |= self.tnc.select(i)
                self.envs[i] ^= ('_RIGHT', self.site_tag(i + self.bsz))

            self.envs[i] |= self.tnc['_LEFT']
            self.pos = start

        elif begin == 'right':

            tags_initital = ['_LEFT'] + [self.site_tag(start + b)
                                         for b in range(self.bsz)]
            self.envs = {start: self.tnc.select(tags_initital, which='any')}

            for i in range(start + 1, stop):
                # add a new site to previous env, and contract one site
                self.envs[i] = self.envs[i - 1].copy(virtual=True)
                self.envs[i] |= self.tnc.select(i + self.bsz - 1)
                self.envs[i] ^= ('_LEFT', self.site_tag(i - 1))

            self.envs[i] |= self.tnc['_RIGHT']
            self.pos = stop - 1

        else:
            raise ValueError("``begin`` must be 'left' or 'right'.")

    def init_non_segment(self, start, stop):
        """Compress and label the effective env not in ``range(start, stop)``
        if cyclic, else just add some dummy left and right end pieces.
        """
        self.tnc = self.tn.copy(virtual=True)

        if not self.segmented:
            if not self.cyclic:
                # generate dummy left and right envs
                self.tnc |= Tensor(1.0, (), {'_LEFT'}).astype(self.tn.dtype)
                self.tnc |= Tensor(1.0, (), {'_RIGHT'}).astype(self.tn.dtype)
                return

            # if cyclic just contract other section and tag
            self.tnc |= Tensor(1.0, (), {'_LEFT'}).astype(self.tn.dtype)
            self.tnc.contract(slice(stop, start + self.n), inplace=True)
            self.tnc.add_tag('_RIGHT', where=stop + 1)
            return

        # replicate all tags on end pieces apart from site number
        ltags = {'_LEFT', *self.tnc.select(start - 1).tags}
        ltags.remove(self.site_tag(start - 1))
        rtags = {'_RIGHT', *self.tnc.select(stop).tags}
        rtags.remove(self.site_tag(stop))

        # for example, pseudo orthogonalization if cyclic
        if self.segment_callbacks is not None:
            for callback in self.segment_callbacks:
                callback(start, stop, self.begin)

        opts = {
            'keep_tags': False,
            'ltags': ltags,
            'rtags': rtags,
            'eps': self.eps,
            'method': self.method,
            'max_bond': self.max_bond,
            'inplace': True,
        }

        self.tnc.replace_section_with_svd(start, stop, which='!any', **opts)

        self.bond_sizes.append(
            self.tnc['_LEFT'].shared_bond_size(self.tnc['_RIGHT']))

        if self.norm:
            # ensure that expectation still = 1 after approximation
            # section left can still be pretty long so do structured contract
            tnn = self.tnc.copy()
            tnn ^= ['_LEFT', self.site_tag(start)]
            tnn ^= ['_RIGHT', self.site_tag(stop - 1)]
            norm = (tnn ^ slice(start, stop)) ** 0.5

            self.tnc['_LEFT'] /= norm
            self.tnc['_RIGHT'] /= norm

    def move_right(self):
        i = (self.pos + 1) % self.n

        # generate a new segment if we go over the border
        if i not in self.segment:
            if not self.cyclic:
                raise ValueError("For OBC, ``0 <= position <= n - bsz``.")
            self.init_segment('left', i, i + self._ssz)
        else:
            self.pos = i % self.n

        i0 = self.segment.start

        if i >= i0 + 1:
            # insert the updated left env from previous step
            # contract left env with updated site just to left
            new_left = self.envs[i - 1].select(
                ['_LEFT', self.site_tag(i - 1)], which='any')
            self.envs[i] |= new_left ^ all

    def move_left(self):
        i = (self.pos - 1) % self.n

        # generate a new segment if we go over the border
        if i not in self.segment:
            if not self.cyclic:
                raise ValueError("For OBC, ``0 <= position <= n - bsz``.")
            self.init_segment('right', i - self._ssz + 1, i + 1)
        else:
            self.pos = i % self.n

        iN = self.segment.stop

        if i <= iN - 2:
            # insert the updated right env from previous step
            # contract right env with updated site just to right
            new_right = self.envs[i + 1].select(
                ['_RIGHT', self.site_tag(i + self.bsz)], which='any')
            self.envs[i] |= new_right ^ all

    def move_to(self, i):
        """Move this effective environment to site ``i``.
        """

        if self.cyclic:
            # to take account of PBC, rescale so that current pos == n // 2,
            #     then work out if desired i is lower or higher
            ri = (i + (self.n // 2 - self.pos)) % self.n
            direction = 'left' if ri <= self.n // 2 else 'right'
        else:
            direction = 'left' if i < self.pos else 'right'

        while self.pos != i % self.n:
            {'left': self.move_left, 'right': self.move_right}[direction]()

    def __call__(self):
        """Get the current environment.
        """
        return self.envs[self.pos]


def get_cyclic_canonizer(k, b, inv_tol=1e-10):
    """Get a function to use as a callback for ``MovingEnvironment`` that
    approximately orthogonalizes the segments of periodic MPS.
    """
    def cyclic_canonizer(start, stop, begin):
        k.canonize_cyclic(slice(start, stop), bra=b, inv_tol=inv_tol)
        if begin == 'left':
            k.right_canonize(start=stop - 1, stop=start, bra=b)
        else:
            k.left_canonize(start=start, stop=stop - 1, bra=b)

    return cyclic_canonizer


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


class DMRGError(Exception):
    pass


class DMRG:
    r"""Density Matrix Renormalization Group variational groundstate search.
    Some initialising arguments act as defaults, but can be overidden with
    each solve or sweep. See :func:`~quimb.tensor.tensor_dmrg.get_default_opts`
    for the list of advanced options initialized in the ``opts`` attribute.

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
    p0 : MatrixProductState, optional
        If given, use as the initial state.

    Attributes
    ----------
    state : MatrixProductState
        The current, optimized state.
    energy : float
        The current most optimized energy.
    energies : list of float
        The total energy after each sweep.
    local_energies : list of list of float
        The local energies per sweep: ``local_energies[i, j]`` contains the
        local energy found at the jth step of the (i+1)th sweep.
    total_energies : list of list of float
        The total energies per sweep: ``local_energies[i, j]`` contains the
        total energy after the jth step of the (i+1)th sweep.
    opts : dict
        Advanced options e.g. relating to the inner eigensolve or compression,
        see :func:`~quimb.tensor.tensor_dmrg.get_default_opts`.
    (bond_sizes_ham) : list[list[int]]
        If cyclic, the sizes of the energy environement transfer matrix bonds,
        per segment, per sweep.
    (bond_sizes_norm) : list[list[int]]
        If cyclic, the sizes of the norm environement transfer matrix bonds,
        per segment, per sweep.
    """

    def __init__(self, ham, bond_dims, cutoffs=1e-9,
                 bsz=2, which='SA', p0=None):
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
        self._k.align_(self.ham, self._b)

        # want to contract this multiple times while
        #   manipulating k/b -> make virtual
        self.TN_energy = self._b | self.ham | self._k
        self.energies = []
        self.local_energies = []
        self.total_energies = []

        # if cyclic need to keep track of normalization
        if self.cyclic:
            eye = self.ham.identity()
            eye.add_tag('_EYE')
            self.TN_norm = self._b | eye | self._k

            self.bond_sizes_ham = []
            self.bond_sizes_norm = []

        self.opts = get_default_opts(self.cyclic)

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
        copy = self._k.copy()
        copy.drop_tags('_KET')
        return copy

    # -------------------- standard DMRG update methods --------------------- #

    def _canonize_after_1site_update(self, direction, i):
        """Compress a site having updated it. Also serves to move the
        orthogonality center along.
        """
        if (direction == 'right') and ((i < self.n - 1) or self.cyclic):
            self._k.left_canonize_site(i, bra=self._b)
        elif (direction == 'left') and ((i > 0) or self.cyclic):
            self._k.right_canonize_site(i, bra=self._b)

    def _eigs(self, A, B=None, v0=None):
        """Find single eigenpair, using all the internal settings.
        """
        # intercept generalized eigen
        backend = self.opts['local_eig_backend']
        if (backend is None) and (B is not None):
            backend = 'LOBPCG'

        return eigh(
            A, k=1, B=B, which=self.which, v0=v0,
            backend=backend,
            EPSType=self.opts['local_eig_EPSType'],
            ncv=self.opts['local_eig_ncv'],
            tol=self.opts['local_eig_tol'],
            maxiter=self.opts['local_eig_maxiter'],
            fallback_to_scipy=True)

    def print_energy_info(self, Heff=None, loc_gs=None):
        sweep_num = len(self.energies) + 1
        full_en = self.TN_energy ^ ...
        effv_en = self._eff_ham ^ all

        if Heff is None:
            site_en = "N/A"
        else:
            site_en = np.asscalar(loc_gs.H @ (Heff @ loc_gs))

        print("Sweep {} -- fullE={} effcE={} siteE={}"
              "".format(sweep_num, full_en, effv_en, site_en))

    def print_norm_info(self, i=None):
        sweep_num = len(self.energies) + 1
        full_n = self._k.H @ self._k

        if self.cyclic:
            effv_n = self._eff_norm ^ all
        else:
            effv_n = 'OBC'

        if i is None:
            site_norm = [self._k[i].H @ self._k[i] for i in range(self.n)]
        else:
            site_norm = self._k[i].H @ self._k[i]

        print("Sweep {} -- fullN={} effvN={} siteN={}"
              "".format(sweep_num, full_n, effv_n, site_norm))

    def form_local_ops(self, i, dims, lix, uix):
        """Construct the effective Hamiltonian, and if needed, norm.
        """
        if self.cyclic:
            self._eff_norm = self.ME_eff_norm()
        self._eff_ham = self.ME_eff_ham()

        # choose a rough value at which dense effective ham should not be used
        dense = self.opts['local_eig_ham_dense']
        if dense is None:
            dense = prod(dims) < 800

        dims_inds = {'ldims': dims, 'rdims': dims,
                     'left_inds': lix, 'right_inds': uix}

        # form effective hamiltonian
        if dense:
            # contract remaining hamiltonian and get its dense representation
            Heff = (self._eff_ham ^ '_HAM')['_HAM'].to_dense(lix, uix)
        else:
            Heff = TNLinearOperator(self._eff_ham['_HAM'], **dims_inds)

        # form effective norm
        if self.cyclic:
            fudge = self.opts['periodic_nullspace_fudge_factor']

            neff_dense = self.opts['local_eig_norm_dense']
            if neff_dense is None:
                neff_dense = dense

            # Check if site already pseudo-orthonogal
            site_norm = self._k[i:i + self.bsz].H @ self._k[i:i + self.bsz]
            if abs(site_norm - 1) < self.opts['periodic_orthog_tol']:
                Neff = None

            # else contruct RHS normalization operator
            elif neff_dense:
                Neff = (self._eff_norm ^ '_EYE')['_EYE'].to_dense(lix, uix)
                np.fill_diagonal(Neff, Neff.diagonal() + fudge)
                np.fill_diagonal(Heff, Heff.diagonal() + fudge**0.5)
            else:
                Neff = TNLinearOperator(self._eff_norm['_EYE'], **dims_inds)
                Neff += IdentityLinearOperator(Neff.shape[0], fudge)
                Heff += IdentityLinearOperator(Heff.shape[0], fudge**0.5)

        else:
            Neff = None

        return Heff, Neff

    def post_check(self, i, Neff, loc_gs, loc_en, loc_gs_old):
        """Perform some checks on the output of the local eigensolve.
        """
        if self.cyclic:
            # pseudo-orthogonal
            if Neff is None:
                # just perform leading correction to norm from site_norm
                site_norm = self._k[i:i + self.bsz].H @ self._k[i:i + self.bsz]
                loc_gs *= site_norm ** 0.5
                loc_en *= site_norm
                return loc_en, loc_gs

            loc_en -= self.opts['periodic_nullspace_fudge_factor']**0.5

            # this is helpful for identifying badly behaved numerics
            Neffnorm = np.asscalar(loc_gs.H @ (Neff @ loc_gs))
            if abs(Neffnorm - 1) > 10 * self.opts['local_eig_tol']:
                raise DMRGError("Effective norm diverged to {}, check "
                                "that Neff is positive?".format(Neffnorm))

        return loc_en, loc_gs

    def _update_local_state_1site(self, i, direction, **compress_opts):
        r"""Find the single site effective tensor groundstate of::

            >->->->->-/|\-<-<-<-<-<-<-<-<          /|\       <-- uix
            | | | | |  |  | | | | | | | |         / | \
            H-H-H-H-H--H--H-H-H-H-H-H-H-H   =    L--H--R
            | | | | | i|  | | | | | | | |         \i| /
            >->->->->-\|/-<-<-<-<-<-<-<-<          \|/       <-- lix

        And insert it back into the states ``k`` and ``b``, and thus
        ``TN_energy``.
        """
        uix, lix = self._k[i].inds, self._b[i].inds
        dims = self._k[i].shape

        # get local operators
        Heff, Neff = self.form_local_ops(i, dims, lix, uix)

        # get the old local groundstate to use as initial guess
        loc_gs_old = self._k[i].data.ravel()

        # find the local energy and groundstate
        loc_en, loc_gs = self._eigs(Heff, B=Neff, v0=loc_gs_old)

        # perform some minor checks and corrections
        loc_en, loc_gs = self.post_check(i, Neff, loc_gs, loc_en, loc_gs_old)

        # insert back into state and all tensor networks viewing it
        loc_gs = loc_gs.A.reshape(dims)
        self._k[i].modify(data=loc_gs)
        self._b[i].modify(data=loc_gs.conj())

        # normalize - necessary due to loose tolerance eigensolve
        if self.cyclic:
            norm = (self._eff_norm ^ all) ** 0.5
            self._k[i].modify(data=self._k[i].data / norm)
            self._b[i].modify(data=self._b[i].data / norm)

        tot_en = self._eff_ham ^ all

        self._canonize_after_1site_update(direction, i)

        return np.asscalar(loc_en), tot_en

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

        # get local operators
        Heff, Neff = self.form_local_ops(i, dims, lix, uix)

        # get the old 2-site local groundstate to use as initial guess
        loc_gs_old = self._k[i].contract(self._k[i + 1]).to_dense(uix)

        # find the 2-site local groundstate and energy
        loc_en, loc_gs = self._eigs(Heff, B=Neff, v0=loc_gs_old)

        # perform some minor checks and corrections
        loc_en, loc_gs = self.post_check(i, Neff, loc_gs, loc_en, loc_gs_old)

        # split the two site local groundstate
        T_AB = Tensor(loc_gs.A.reshape(dims), uix)
        L, R = T_AB.split(left_inds=uix_L, get='arrays', absorb=direction,
                          right_inds=uix_R, **compress_opts)

        # insert back into state and all tensor networks viewing it
        self._k[i].modify(data=L, inds=(*uix_L, u_bond_ind))
        self._b[i].modify(data=L.conj(), inds=(*lix_L, l_bond_ind))
        self._k[i + 1].modify(data=R, inds=(u_bond_ind, *uix_R))
        self._b[i + 1].modify(data=R.conj(), inds=(l_bond_ind, *lix_R))

        # normalize due to compression and insert factor at the correct site
        if self.cyclic:
            #   Right         Left
            #   i  i+1        i  i+1
            # -->~~o--  or  --o~~<--
            #   |  |          |  |
            norm = (self.ME_eff_norm() ^ all) ** 0.5
            next_site = {'right': i + 1, 'left': i}[direction]

            self._k[next_site].modify(data=self._k[next_site].data / norm)
            self._b[next_site].modify(data=self._b[next_site].data / norm)

        tot_en = self._eff_ham ^ all

        return np.asscalar(loc_en), tot_en

    def _update_local_state(self, i, **update_opts):
        """Move envs to site ``i`` and dispatch to the correct local updater.
        """
        if self.cyclic:
            # move effective norm first as it can trigger canonize_cyclic etc.
            self.ME_eff_norm.move_to(i)

        self.ME_eff_ham.move_to(i)

        return {
            1: self._update_local_state_1site,
            2: self._update_local_state_2site,
        }[self.bsz](i, **update_opts)

    def sweep(self, direction, canonize=True, verbosity=0, **update_opts):
        r"""Perform a sweep of optimizations, either rightwards::

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
        verbosity : {0, 1, 2}, optional
            Show a progress bar for the sweep.
        update_opts :
            Supplied to ``self._update_local_state``.
        """
        if canonize:
            {'R': self._k.right_canonize,
             'L': self._k.left_canonize}[direction](bra=self._b)

        n, bsz = self.n, self.bsz

        direction, begin, sweep = {
            ('R', False): ('right', 'left', range(0, n - bsz + 1)),
            ('L', False): ('left', 'right', range(n - bsz, -1, -1)),
            ('R', True): ('right', 'left', range(0, n)),
            ('L', True): ('left', 'right', range(n - 1, -1, -1)),
        }[direction, self.cyclic]

        if verbosity:
            sweep = progbar(sweep, ncols=80, total=len(sweep))

        env_opts = {'begin': begin, 'bsz': bsz, 'cyclic': self.cyclic,
                    'ssz': self.opts['periodic_segment_size'],
                    'method': self.opts['periodic_compress_method'],
                    'max_bond': self.opts['periodic_compress_max_bond']}

        if self.cyclic:
            # setup moving norm environment
            nm_opts = {
                **env_opts, 'norm': True,
                'eps': self.opts['periodic_compress_norm_eps'],
                'segment_callbacks': get_cyclic_canonizer(
                    self._k, self._b,
                    inv_tol=self.opts['periodic_canonize_inv_tol']),
            }
            self.ME_eff_norm = MovingEnvironment(self.TN_norm, **nm_opts)

        # setup moving energy environment
        en_opts = {**env_opts, 'eps': self.opts['periodic_compress_ham_eps']}
        self.ME_eff_ham = MovingEnvironment(self.TN_energy, **en_opts)

        # perform the sweep, collecting local and total energies
        local_ens, tot_ens = zip(*[
            self._update_local_state(i, direction=direction, **update_opts)
            for i in sweep
        ])

        if verbosity:
            sweep.close()

        self.local_energies.append(local_ens)
        self.total_energies.append(tot_ens)

        if self.cyclic:
            self.bond_sizes_ham.append(self.ME_eff_ham.bond_sizes)
            self.bond_sizes_norm.append(self.ME_eff_norm.bond_sizes)

        return tot_ens[-1]

    def sweep_right(self, canonize=True, verbosity=0, **update_opts):
        return self.sweep(direction='R', canonize=canonize,
                          verbosity=verbosity, **update_opts)

    def sweep_left(self, canonize=True, verbosity=0, **update_opts):
        return self.sweep(direction='L', canonize=canonize,
                          verbosity=verbosity, **update_opts)

    # ----------------- overloadable 'plugin' style methods ----------------- #

    @staticmethod
    def _print_pre_sweep(i, LR, bd, ctf, verbosity=0):
        """Print this before each sweep.
        """
        if verbosity > 0:
            msg = "SWEEP-{}, direction={}, max_bond={}, cutoff:{}"
            print(msg.format(i + 1, LR, bd, ctf), flush=True)

    def _compute_post_sweep(self):
        """Compute this after each sweep.
        """
        pass

    def _print_post_sweep(self, converged, verbosity=0):
        """Print this after each sweep.
        """
        if verbosity > 1:
            self._k.show()
        if verbosity > 0:
            msg = "Energy: {} ... {}".format(self.energy, "converged!" if
                                             converged else "not converged.")
            print(msg, flush=True)

    def _check_convergence(self, tol):
        """By default check the absolute change in energy.
        """
        if len(self.energies) < 2:
            return False
        return abs(self.energies[-2] - self.energies[-1]) < tol

    # -------------------------- main solve driver -------------------------- #

    def solve(self,
              tol=1e-4,
              bond_dims=None,
              cutoffs=None,
              sweep_sequence=None,
              max_sweeps=10,
              verbosity=0):
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
        verbosity : {0, 1, 2}, optional
            How much information to print about progress.

        Returns
        -------
        converged : bool
            Whether the algorithm has converged to ``tol`` yet.
        """
        verbosity = int(verbosity)

        # Possibly overide the default bond dimension, cutoff, LR sequences.
        if bond_dims is not None:
            self._set_bond_dim_seq(bond_dims)
        if cutoffs is not None:
            self._set_cutoff_seq(cutoffs)
        if sweep_sequence is None:
            sweep_sequence = self.opts['default_sweep_sequence']

        RLs = itertools.cycle(sweep_sequence)
        previous_LR = '0'

        for _ in range(max_sweeps):
            # Get the next direction, bond dimension and cutoff
            LR, bd, ctf = next(RLs), next(self._bond_dims), next(self._cutoffs)
            self._print_pre_sweep(len(self.energies), LR,
                                  bd, ctf, verbosity=verbosity)

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
                'cutoff_mode': self.opts['bond_compress_cutoff_mode'],
                'method': self.opts['bond_compress_method'],
                'verbosity': verbosity,
            }

            # perform sweep, any plugin computations
            self.energies.append(self.sweep(direction=LR, **sweep_opts))
            self._compute_post_sweep()

            # check convergence
            converged = self._check_convergence(tol)
            self._print_post_sweep(converged, verbosity=verbosity)
            if converged:
                break

            previous_LR = LR

        return converged


class DMRG1(DMRG):
    """Simple alias of one site ``DMRG``.
    """
    __doc__ += DMRG.__doc__

    def __init__(self, ham, which='SA', bond_dims=None, cutoffs=1e-8, p0=None):

        if bond_dims is None:
            bond_dims = range(10, 1001, 10)

        super().__init__(ham, bond_dims=bond_dims, cutoffs=cutoffs,
                         which=which, p0=p0, bsz=1)


class DMRG2(DMRG):
    """Simple alias of two site ``DMRG``.
    """
    __doc__ += DMRG.__doc__

    def __init__(self, ham, which='SA', bond_dims=None, cutoffs=1e-8, p0=None):

        if bond_dims is None:
            bond_dims = [8, 16, 32, 64, 128, 256, 512, 1024]

        super().__init__(ham, bond_dims=bond_dims, cutoffs=cutoffs,
                         which=which, p0=p0, bsz=2)


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
        The initial MPS guess, e.g. a computation basis state.
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
        self.energies.append(self.TN_energy ^ ...)
        self.variances = [(self.TN_energy2 ^ ...) - self.energies[-1]**2]
        self._target_energy = self.energies[-1]

        self.opts = {
            'local_eig_partial_cutoff': 2**11,
            'local_eig_partial_k': 0.02,
            'local_eig_tol': 1e-1,
            'overlap_thresh': 2 / 3,
            'bond_compress_method': 'svd',
            'bond_compress_cutoff_mode': 'sum2',
            'default_sweep_sequence': 'RRLL',
            'bond_expand_rand_strength': 1e-9,
        }

    @property
    def variance(self):
        return self.variances[-1]

    def form_local_ops(self, i, dims, lix, uix):
        self._eff_ham = self.ME_eff_ham()
        self._eff_ovlp = self.ME_eff_ovlp()
        self._eff_ham2 = self.ME_eff_ham2()

        Heff = (self._eff_ham ^ '_HAM')['_HAM'].to_dense(lix, uix)

        return Heff

    def _update_local_state_1site_dmrgx(self, i, direction, **compress_opts):
        """Like ``_update_local_state``, but re-insert all eigenvectors, then
        choose the one with best overlap with ``eff_ovlp``.
        """
        uix, lix = self._k[i].inds, self._b[i].inds
        dims = self._k[i].shape

        # contract remaining hamiltonian and get its dense representation
        Heff = self.form_local_ops(i, dims, lix, uix)

        # eigen-decompose and reshape eigenvectors thus::
        #
        #    |'__ev_ind__'
        #    E
        #   /|\
        #
        D = prod(dims)
        if D <= self.opts['local_eig_partial_cutoff']:
            evals, evecs = eigh(Heff)
        else:
            if isinstance(self.opts['local_eig_partial_k'], float):
                k = int(self.opts['local_eig_partial_k'] * D)
            else:
                k = self.opts['local_eig_partial_k']

            evals, evecs = eigh(
                Heff, sigma=self._target_energy, v0=self._k[i].data,
                k=k, tol=self.opts['local_eig_tol'], backend='scipy')

        evecs = _asarray(evecs).reshape(*dims, -1)
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
        overlaps = np.abs((self._eff_ovlp ^ all).data)

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
        # store the current effective energy for possibly targeted eigh
        self._target_energy = evals[best]

        tot_en = self._eff_ham ^ all

        self._canonize_after_1site_update(direction, i)

        return evals[best], tot_en

    # def _update_local_state_2site_dmrgx(self, i, direction, **compress_opts):
    #     raise NotImplementedError("2-site DMRGX not implemented yet.")
    #     dims, lix_L, lix_R, lix, uix_L, uix_R, uix, l_bond_ind, u_bond_ind =\
    #         parse_2site_inds_dims(self._k, self._b, i)

    #     # contract remaining hamiltonian and get its dense representation
    #     eff_ham = (self._eff_ham ^ '_HAM')['_HAM']
    #     eff_ham.fuse_((('lower', lix), ('upper', uix)))
    #     A = eff_ham.data

    #     # eigen-decompose and reshape eigenvectors thus::
    #     #
    #     #    ||'__ev_ind__'
    #     #    EE
    #     #   /||\
    #     #
    #     D = prod(dims)
    #     if D <= self.opts['local_eig_partial_cutoff']:
    #         evals, evecs = eigh(A)
    #     else:
    #         if isinstance(self.opts['local_eig_partial_k'], float):
    #             k = int(self.opts['local_eig_partial_k'] * D)
    #         else:
    #             k = self.opts['local_eig_partial_k']

    #         # find the 2-site local state using previous as initial guess
    #         v0 = self._k[i].contract(self._k[i + 1], output_inds=uix).data

    #         evals, evecs = eigh(
    #             A, sigma=self.energies[-1], v0=v0,
    #             k=k, tol=self.opts['local_eig_tol'], backend='scipy')

    def _update_local_state(self, i, **update_opts):
        self.ME_eff_ham.move_to(i)
        self.ME_eff_ham2.move_to(i)
        self.ME_eff_ovlp.move_to(i)

        return {
            1: self._update_local_state_1site_dmrgx,
            # 2: self._update_local_state_2site_dmrgx,
        }[self.bsz](i, **update_opts)

    def sweep(self, direction, canonize=True, verbosity=0, **update_opts):
        """Perform a sweep of the algorithm.

        Parameters
        ----------
        direction : {'R', 'L'}
            Sweep from left to right (->) or right to left (<-) respectively.
        canonize : bool, optional
            Canonize the state first, not needed if doing alternate sweeps.
        verbosity : {0, 1, 2}, optional
            Show a progress bar for the sweep.
        update_opts :
            Supplied to ``self._update_local_state``.
        """
        old_k = self._k.copy().H
        TN_overlap = TensorNetwork([self._k, old_k], virtual=True)

        if canonize:
            {'R': self._k.right_canonize,
             'L': self._k.left_canonize}[direction](bra=self._b)

        direction, begin, sweep = {
            'R': ('right', 'left', range(0, self.n - self.bsz + 1)),
            'L': ('left', 'right', reversed(range(0, self.n - self.bsz + 1))),
        }[direction]

        eff_opts = {'begin': begin, 'bsz': self.bsz, 'cyclic': self.cyclic}
        self.ME_eff_ham = MovingEnvironment(self.TN_energy, **eff_opts)
        self.ME_eff_ham2 = MovingEnvironment(self.TN_energy2, **eff_opts)
        self.ME_eff_ovlp = MovingEnvironment(TN_overlap, **eff_opts)

        if verbosity:
            sweep = progbar(sweep, ncols=80, total=self.n - self.bsz + 1)

        local_ens, tot_ens = zip(*[
            self._update_local_state(i, direction=direction, **update_opts)
            for i in sweep
        ])

        self.local_energies.append(local_ens)
        self.total_energies.append(tot_ens)

        return tot_ens[-1]

    def _compute_post_sweep(self):
        en_var = (self.TN_energy2 ^ ...) - self.energies[-1]**2
        self.variances.append(en_var)

    def _print_post_sweep(self, converged, verbosity=0):
        if verbosity > 1:
            self._k.show()
        if verbosity > 0:
            msg = "Energy={}, Variance={} ... {}"
            msg = msg.format(self.energy, self.variance, "converged!"
                             if converged else "not converged.")
            print(msg, flush=True)

    def _check_convergence(self, tol):
        return self.variance < tol
