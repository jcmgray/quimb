"""Classes and algorithms related to 1D tensor networks.
"""

import re
import functools
from math import log2
from numbers import Integral

import numpy as np
import opt_einsum as oe

from ..utils import three_line_multi_print, pairwise
import quimb as qu
from .tensor_core import (
    Tensor,
    TensorNetwork,
    rand_uuid,
    bonds,
    tags2set,
    get_tags,
    _asarray,
    _ndim,
)


def align_TN_1D(*tns, ind_ids=None, inplace=False):
    r"""Align an arbitrary number of 1D tensor networks in a stack-like
    geometry::

        a-a-a-a-a-a-a-a-a-a-a-a-a-a-a-a-a-a
        | | | | | | | | | | | | | | | | | | <- ind_ids[0] (defaults to 1st id)
        b-b-b-b-b-b-b-b-b-b-b-b-b-b-b-b-b-b
        | | | | | | | | | | | | | | | | | | <- ind_ids[1]
                       ...
        | | | | | | | | | | | | | | | | | | <- ind_ids[-2]
        y-y-y-y-y-y-y-y-y-y-y-y-y-y-y-y-y-y
        | | | | | | | | | | | | | | | | | | <- ind_ids[-1]
        z-z-z-z-z-z-z-z-z-z-z-z-z-z-z-z-z-z

    Parameters
    ----------
    tns : sequence of TensorNetwork1D
        The 1D TNs to align.
    ind_ids : None, or sequence of str
        String with format specifiers to id each level of sites with. Will be
        automatically generated like ``(tns[0].site_ind_id, "__ind_a{}__",
        "__ind_b{}__", ...)`` if not given.
    """
    if not inplace:
        tns = [tn.copy() for tn in tns]

    if ind_ids is None:
        ind_ids = ([tns[0].site_ind_id] +
                   ["__ind_{}".format(oe.get_symbol(i)) + "{}__"
                    for i in range(len(tns) - 2)])
    else:
        ind_ids = tuple(ind_ids)

    for i, tn in enumerate(tns):
        if isinstance(tn, TensorNetwork1DVector):
            if i == 0:
                tn.site_ind_id = ind_ids[i]
            elif i == len(tns) - 1:
                tn.site_ind_id = ind_ids[i - 1]
            else:
                raise ValueError("An 1D TN vector can only be aligned as the "
                                 "first or last TN in a sequence.")

        elif isinstance(tn, MatrixProductOperator):
            tn.upper_ind_id = ind_ids[i - 1]
            tn.lower_ind_id = ind_ids[i]

        else:
            raise ValueError("Can only align MPS and MPOs currently.")

    return tns


def expec_TN_1D(*tns, compress=None, eps=1e-15):
    """Compute the expectation of several 1D TNs, using transfer matrix
    compression if any are periodic.

    Parameters
    ----------
    tns : sequence of TensorNetwork1D
        The MPS and MPO to find expectation of. Should start and begin with
        an MPS e.g. ``(MPS, MPO, ...,  MPS)``.
    compress : {None, False, True}, optional
        Whether to perform transfer matrix compression on cyclic systems. If
        set to ``None`` (the default), decide heuristically.
    eps : float, optional
        The accuracy of the transfer matrix compression.

    Returns
    -------
    x : float
        The expectation value.
    """
    expec_tn = TensorNetwork(align_TN_1D(*tns))

    # if OBC or <= 0.0 specified use exact contraction
    cyclic = any(tn.cyclic for tn in tns)
    if not cyclic:
        compress = False

    n = expec_tn.nsites
    isflat = all(isinstance(tn, TensorNetwork1DFlat) for tn in tns)

    # work out whether to compress, could definitely be improved ...
    if compress is None and isflat:
        # compression only worth it for long, high bond dimension TNs.
        total_bd = qu.prod(tn.bond_size(0, 1) for tn in tns)
        compress = (n >= 100) and (total_bd >= 1000)

    if compress:
        expec_tn.replace_section_with_svd(1, n, eps=eps, inplace=True)
        return expec_tn ^ all

    return expec_tn ^ ...


def gate_TN_1D(tn, G, where, contract=False, tags=None,
               propagate_tags='sites', inplace=False,
               cur_orthog=None, **compress_opts):
    r"""Act with the gate ``g`` on sites ``where``, maintaining the outer
    indices of the 1D tensor netowork::

        contract=False     contract=True
            ...                  ...         <- where
        0-0-0-0-0-0-0      0-0-0-GGG-0-0-0
        | | | | | | |      | | | / \ | | |
            GGG
            | |

    By default, site tags will be propagated to the gate tensors, identifying
    a 'light cone'.

    Parameters
    ----------
    G : array
        A square array to act with on sites ``where``. It should have twice the
        number of dimensions as the number of sites. The second half of these
        will be contracted with the MPS, and the first half indexed with the
        correct ``site_ind_id``. Sites are read left to right from the shape.
    where : int or sequence of int
        Where the gate should act.
    contract, {False, True, 'swap+split'}, optional
        Contract the gate into the MPS, or leave it uncontracted.
    tags : str or sequence of str, optional
        Tag the new gate tensor with these tags.
    propagate_tags : {'sites', 'register', False, True}, optional
        Add any tags from the sites to the new gate tensor (only matters if
        ``contract=False`` else tags are merged anyway):

            - If ``'sites'``, then only propagate tags matching e.g. 'I{}' and
              ignore all others. I.e. just propagate the lightcone.
            - If ``'register'``, then only propagate tags matching the sites of
              where this gate was actually applied. I.e. ignore the lightcone,
              just keep track of which 'registers' the gate was applied to.
            - If ``False``, propagate nothing.
            - If ``True``, propagate all atags.

    inplace, bool, optional
        Perform the gate in place.

    Returns
    -------
    MatrixProductState

    See Also
    --------
    MatrixProductState.gate_split

    Examples
    --------
    >>> p = MPS_rand_state(3, 7)
    >>> p.gate_(spin_operator('X'), where=1, tags=['GX'])
    >>> p
    <MatrixProductState(tensors=4, structure='I{}', nsites=3)>

    >>> p.outer_inds()
    ('k0', 'k1', 'k2')
    """
    psi = tn if inplace else tn.copy()

    dp = psi.phys_dim()
    tags = tags2set(tags)

    if isinstance(where, Integral):
        where = (where,)
    ng = len(where)

    shape_matches_2d = (_ndim(G) == 2) and (G.shape[1] == dp ** ng)
    shape_maches_nd = all(d == dp for d in G.shape)

    if shape_matches_2d:
        G = _asarray(G).reshape([dp] * 2 * ng)
    elif not shape_maches_nd:
        raise ValueError("Gate with shape {} doesn't match sites {}"
                         "".format(G.shape, where))

    if contract == 'swap+split' and ng > 1:
        if ng > 2:
            raise ValueError("Can't use auto-swap gate for more than 2 sites.")
        psi.gate_with_auto_swap(G, where, cur_orthog=cur_orthog,
                                inplace=True, **compress_opts)
        return psi

    bnds = [rand_uuid() for _ in range(ng)]
    site_ix = [psi.site_ind(i) for i in where]
    gate_ix = site_ix + bnds

    psi.reindex_(dict(zip(site_ix, bnds)))

    # get the sites that used to have the physical indices
    site_tids = psi._get_tids_from_inds(bnds, which='any')

    TG = Tensor(G, gate_ix, tags=tags)

    if contract:
        # pop the sites, contract, then re-add
        pts = [psi._pop_tensor(tid) for tid in site_tids]
        psi |= TG.contract(*pts)
    else:
        psi |= TG
        if propagate_tags:
            if propagate_tags == 'register':
                old_tags = {psi.site_tag(i) for i in where}
            else:
                old_tags = get_tags(psi.tensor_map[tid] for tid in site_tids)

            if propagate_tags == 'sites':
                # use regex to take tags only matching e.g. 'I0', 'I13'
                rex = re.compile(psi.structure.format("\d+"))
                old_tags = {t for t in old_tags if rex.match(t)}

            TG.modify(tags=TG.tags | old_tags)

    return psi


def rand_padder(vector, pad_width, iaxis, kwargs):
    """Helper function for padding tensor with random entries.
    """
    rand_strength = kwargs.get('rand_strength')
    if pad_width[0]:
        vector[:pad_width[0]] = rand_strength * qu.randn(pad_width[0],
                                                         dtype='float32')
    if pad_width[1]:
        vector[-pad_width[1]:] = rand_strength * qu.randn(pad_width[1],
                                                          dtype='float32')
    return vector


class TensorNetwork1D:
    """Base class for tensor networks with a one-dimensional structure.
    """

    @property
    def site_tag_id(self):
        """The string specifier for tagging each site of this 1D TN.
        """
        return self._site_tag_id

    def site_tag(self, i):
        """The name of the tag specifiying the tensor at site ``i``.
        """
        return self.site_tag_id.format(i % self.nsites)

    @property
    def site_tags(self):
        """An ordered tuple of the actual site tags.
        """
        return tuple(self.site_tag(i) for i in self.sites)


class TensorNetwork1DVector:
    """1D Tensor network which overall is like a vector with a single type of
    site ind.
    """

    def reindex_sites(self, new_id, where=None, inplace=False):
        """Update the physical site index labels to a new string specifier.
        Note that this doesn't change the stored id string with the TN.

        Parameters
        ----------
        new_id : str
            A string with a format placeholder to accept an int, e.g. "ket{}".
        where : None or slice
            Which sites to update the index labels on. If ``None`` (default)
            all sites.
        inplace : bool
            Whether to reindex in place.
        """
        if where is None:
            indices = self.sites
        elif isinstance(where, slice):
            indices = self.slice2sites(where)
        else:
            indices = where

        return self.reindex({self.site_ind(i): new_id.format(i)
                             for i in indices}, inplace=inplace)

    def _get_site_ind_id(self):
        return self._site_ind_id

    def _set_site_ind_id(self, new_id):
        if self._site_ind_id != new_id:
            self.reindex_sites(new_id, inplace=True)
            self._site_ind_id = new_id

    site_ind_id = property(_get_site_ind_id, _set_site_ind_id,
                           doc="The string specifier for the physical indices")

    def site_ind(self, i):
        if isinstance(i, Integral):
            i = i % self.nsites
        return self.site_ind_id.format(i)

    @property
    def site_inds(self):
        """An ordered tuple of the actual physical indices.
        """
        return tuple(self.site_ind(i) for i in self.sites)

    def to_dense(self, *inds_seq):
        """Return the dense ket version of this 1D vector, i.e. a
        ``qarray`` with shape (-1, 1).
        """
        if not inds_seq:
            # just use list of site indices
            return TensorNetwork.to_dense(self, self.site_inds).reshape(-1, 1)

        return TensorNetwork.to_dense(self, self.site_inds)

    def phys_dim(self, i=None):
        if i is None:
            i = self.sites[0]
        return self.ind_size(self.site_ind(i))

    @functools.wraps(gate_TN_1D)
    def gate(self, *args, inplace=False, **kwargs):
        return gate_TN_1D(self, *args, inplace=inplace, **kwargs)

    gate_ = functools.partialmethod(gate, inplace=True)

    @functools.wraps(align_TN_1D)
    def align(self, *args, inplace=False, **kwargs):
        return align_TN_1D(self, *args, inplace=inplace, **kwargs)

    align_ = functools.partialmethod(align, inplace=True)

    @functools.wraps(expec_TN_1D)
    def expec(self, *args, **kwargs):
        return expec_TN_1D(self, *args, **kwargs)

    def correlation(self, A, i, j, B=None, **expec_opts):
        """Correlation of operator ``A`` between ``i`` and ``j``.

        Parameters
        ----------
        A : array
            The operator to act with, can be multi site.
        i : int or sequence of int
            The first site(s).
        j : int or sequence of int
            The second site(s).
        expec_opts
            Supplied to :func:`~quimb.tensor.tensor_1d.expec_TN_1D`.

        Returns
        -------
        C : float
            The correlation ``<A(i)> + <A(j)> - <A(ij)>``.

        Examples
        --------
        >>> ghz = (MPS_computational_state('0000') +
        ...        MPS_computational_state('1111')) / 2**0.5
        >>> ghz.correlation(pauli('Z'), 0, 1)
        1.0
        >>> ghz.correlation(pauli('Z'), 0, 1, B=pauli('X'))
        0.0
        """
        if B is None:
            B = A

        pA = self.gate(A, i, contract=True)
        cA = self.expec(pA, **expec_opts)

        pB = self.gate(B, j, contract=True)
        cB = self.expec(pB, **expec_opts)

        pAB = pA.gate_(B, j, contract=True)
        cAB = self.expec(pAB, **expec_opts)

        return cAB - cA * cB


class TensorNetwork1DFlat:
    """1D Tensor network which has a flat structure.
    """

    def _left_decomp_site(self, i, bra=None, **split_opts):
        T1, T2 = self[i], self[i + 1]
        rix, lix = T1.filter_bonds(T2)

        Q, R = T1.split(lix, get='tensors', right_inds=rix, **split_opts)
        R = R @ T2

        Q.transpose_like_(T1)
        R.transpose_like_(T2)

        self[i].modify(data=Q.data)
        self[i + 1].modify(data=R.data)

        if bra is not None:
            bra[i].modify(data=Q.data.conj())
            bra[i + 1].modify(data=R.data.conj())

    def _right_decomp_site(self, i, bra=None, **split_opts):
        T1, T2 = self[i], self[i - 1]
        lix, rix = T1.filter_bonds(T2)

        L, Q = T1.split(lix, get='tensors', right_inds=rix, **split_opts)
        L = T2 @ L

        L.transpose_like_(T2)
        Q.transpose_like_(T1)

        self[i - 1].modify(data=L.data)
        self[i].modify(data=Q.data)

        if bra is not None:
            bra[i - 1].modify(data=L.data.conj())
            bra[i].modify(data=Q.data.conj())

    def left_canonize_site(self, i, bra=None):
        r"""Left canonize this TN's ith site, inplace::

                i                i
               -o-o-            ->-s-
            ... | | ...  ==> ... | | ...

        Parameters
        ----------
        i : int
            Which site to canonize. The site at i + 1 also absorbs the
            non-isometric part of the decomposition of site i.
        bra : None or matching TensorNetwork to self, optional
            If set, also update this TN's data with the conjugate canonization.
        """
        self._left_decomp_site(i, bra=bra, method='qr')

    def right_canonize_site(self, i, bra=None):
        r"""Right canonize this TN's ith site, inplace::

                  i                i
               -o-o-            -s-<-
            ... | | ...  ==> ... | | ...

        Parameters
        ----------
        i : int
            Which site to canonize. The site at i - 1 also absorbs the
            non-isometric part of the decomposition of site i.
         bra : None or matching TensorNetwork to self, optional
            If set, also update this TN's data with the conjugate canonization.
        """
        self._right_decomp_site(i, bra=bra, method='lq')

    def left_canonize(self, stop=None, start=None, normalize=False, bra=None):
        r"""Left canonize all or a portion of this TN. If this is a MPS,
        this implies that::

                          i              i
            >->->->->->->-o-o-         +-o-o-
            | | | | | | | | | ...  =>  | | | ...
            >->->->->->->-o-o-         +-o-o-

        Parameters
        ----------
        start : int, optional
            If given, the site to start left canonizing at.
        stop : int, optional
            If given, the site to stop left canonizing at.
        normalize : bool, optional
            Whether to normalize the state, only works for OBC.
        bra : MatrixProductState, optional
            If supplied, simultaneously left canonize this MPS too, assuming it
            to be the conjugate state.
        """
        if start is None:
            start = 0
        if stop is None:
            stop = self.nsites - 1

        for i in range(start, stop):
            self.left_canonize_site(i, bra=bra)

        if normalize:
            factor = self[-1].norm()
            self[-1] /= factor
            if bra is not None:
                bra[-1] /= factor

    def right_canonize(self, stop=None, start=None, normalize=False, bra=None):
        r"""Right canonize all or a portion of this TN. If this is a MPS,
        this implies that::

                   i                           i
                -o-o-<-<-<-<-<-<-<          -o-o-+
             ... | | | | | | | | |   ->  ... | | |
                -o-o-<-<-<-<-<-<-<          -o-o-+


        Parameters
        ----------
        start : int, optional
            If given, the site to start right canonizing at.
        stop : int, optional
            If given, the site to stop right canonizing at.
        normalize : bool, optional
            Whether to normalize the state.
        bra : MatrixProductState, optional
            If supplied, simultaneously right canonize this MPS too, assuming
            it to be the conjugate state.
        """
        if start is None:
            start = self.nsites - 1
        if stop is None:
            stop = 0

        for i in range(start, stop, -1):
            self.right_canonize_site(i, bra=bra)

        if normalize:
            factor = self[0].norm()
            self[0] /= factor
            if bra is not None:
                bra[0] /= factor

    def canonize_cyclic(self, i, bra=None, method='isvd', inv_tol=1e-10):
        """Bring this MatrixProductState into (possibly only approximate)
        canonical form at site(s) ``i``.

        Parameters
        ----------
        i :  int or slice
            The site or range of sites to make canonical.
        bra : MatrixProductState, optional
            Simultaneously canonize this state as well, assuming it to be the
            co-vector.
        method : {'isvd', 'svds', ...}, optional
            How to perform the lateral compression.
        inv_tol : float, optional
            Tolerance with which to invert the gauge.
        """
        if isinstance(i, Integral):
            start, stop = i, i + 1
        elif isinstance(i, slice):
            start, stop = i.start, i.stop
        else:
            start, stop = min(i), max(i) + 1
            if tuple(i) != tuple(range(start, stop)):
                raise ValueError("Parameter ``i`` should be an integer or "
                                 "contiguous block of integers, got {}."
                                 "".format(i))

        k = self.copy()
        b = k.H
        k.add_tag('_KET')
        b.add_tag('_BRA')
        kb = k & b

        # approximate the rest of the chain with a separable transfer operator
        kbc = kb.replace_section_with_svd(start, stop, eps=0.0, which='!any',
                                          method=method, max_bond=1,
                                          ltags='_LEFT', rtags='_RIGHT')

        EL = kbc['_LEFT'].squeeze()
        # explicitly symmetrize to hermitian
        EL.modify(data=(EL.data + EL.data.conj().T) / 2)
        # split into upper 'ket' part and lower 'bra' part, symmetric
        EL_lix, = EL.bonds(kbc[k.site_tag(start), '_BRA'])
        _, x = EL.split(EL_lix, method='eigh', cutoff=-1, get='arrays')

        ER = kbc['_RIGHT'].squeeze()
        # explicitly symmetrize to hermitian
        ER.modify(data=(ER.data + ER.data.conj().T) / 2)
        # split into upper 'ket' part and lower 'bra' part, symmetric
        ER_lix, = ER.bonds(kbc[k.site_tag(stop - 1), '_BRA'])
        _, y = ER.split(ER_lix, method='eigh', cutoff=-1, get='arrays')

        self.insert_gauge(x.T, start - 1, start, tol=inv_tol)
        self.insert_gauge(y.T, stop, stop - 1, tol=inv_tol)

        if bra is not None:
            for i in (start - 1, start, stop, stop - 1):
                bra[i].modify(data=self[i].data.conj())

    def shift_orthogonality_center(self, current, new, bra=None):
        """Move the orthogonality center of this MPS.

        Parameters
        ----------
        current : int
            The current orthogonality center.
        new : int
            The target orthogonality center.
        bra : MatrixProductState, optional
            If supplied, simultaneously move the orthogonality center of this
            MPS too, assuming it to be the conjugate state.
        """
        if new > current:
            for i in range(current, new):
                self.left_canonize_site(i, bra=bra)
        else:
            for i in range(current, new, -1):
                self.right_canonize_site(i, bra=bra)

    def canonize(self, where, cur_orthog='calc', bra=None):
        r"""Mixed canonize this TN. If this is a MPS, this implies that::

                          i                      i
            >->->->->- ->-o-<- -<-<-<-<-<      +-o-+
            | | | | |...| | |...| | | | |  ->  | | |
            >->->->->- ->-o-<- -<-<-<-<-<      +-o-+

        You can also supply a set of indices to orthogonalize around, and a
        current location of the orthogonality center for efficiency::

                  current                              where
                  .......                              .....
            >->->-c-c-c-c-<-<-<-<-<-<      >->->->->->-w-w-w-<-<-<-<
            | | | | | | | | | | | | |  ->  | | | | | | | | | | | | |
            >->->-c-c-c-c-<-<-<-<-<-<      >->->->->->-w-w-w-<-<-<-<
               cmin     cmax                           i   j

        This would only move ``cmin`` to ``i`` and ``cmax`` to ``j`` if
        necessary.

        Parameters
        ----------
        where : int or sequence of int
            Which site(s) to orthogonalize around. If a sequence of int then
            make sure that section from min(where) to max(where) is orthog.
        cur_orthog : int, sequence of int, or 'calc'
            If given, the current site(s), so as to shift the orthogonality
            ceneter as efficiently as possible. If 'calc', calculate the
            current orthogonality center.
        bra : MatrixProductState, optional
            If supplied, simultaneously mixed canonize this MPS too, assuming
            it to be the conjugate state.
        """
        if isinstance(where, int):
            i = j = where
        else:
            i, j = min(where), max(where)

        if cur_orthog == 'calc':
            cur_orthog = self.calc_current_orthog_center()

        if cur_orthog is not None:
            if isinstance(cur_orthog, int):
                cmin = cmax = cur_orthog
            else:
                cmin, cmax = min(cur_orthog), max(cur_orthog)

            if cmax > j:
                self.shift_orthogonality_center(cmax, j, bra=bra)
            if cmin < i:
                self.shift_orthogonality_center(cmin, i, bra=bra)

        else:
            self.left_canonize(i, bra=bra)
            self.right_canonize(j, bra=bra)

        return self

    def left_compress_site(self, i, bra=None, **compress_opts):
        """Left compress this 1D TN's ith site, such that the site is then
        left unitary with its right bond (possibly) reduced in dimension.

        Parameters
        ----------
        i : int
            Which site to compress.
        bra : None or matching TensorNetwork to self, optional
            If set, also update this TN's data with the conjugate compression.
        compress_opts
            Supplied to :meth:`Tensor.split`.
        """
        compress_opts['absorb'] = compress_opts.get('absorb', 'right')
        self._left_decomp_site(i, bra=bra, **compress_opts)

    def right_compress_site(self, i, bra=None, **compress_opts):
        """Right compress this 1D TN's ith site, such that the site is then
        right unitary with its left bond (possibly) reduced in dimension.

        Parameters
        ----------
        i : int
            Which site to compress.
        bra : None or matching TensorNetwork to self, optional
            If set, update this TN's data with the conjugate compression.
        compress_opts
            Supplied to :meth:`Tensor.split`.
        """
        compress_opts['absorb'] = compress_opts.get('absorb', 'left')
        self._right_decomp_site(i, bra=bra, **compress_opts)

    def left_compress(self, start=None, stop=None, bra=None, **compress_opts):
        """Compress this 1D TN, from left to right, such that it becomes
        left-canonical (unless ``absorb != 'right'``).

        Parameters
        ----------
        start : int, optional
            Site to begin compressing on.
        stop : int, optional
            Site to stop compressing at (won't itself be an isometry).
        bra : None or TensorNetwork like this one, optional
            If given, update this TN as well, assuming it to be the conjugate.
        compress_opts
            Supplied to :meth:`Tensor.split`.
        """
        if start is None:
            start = 0
        if stop is None:
            stop = self.nsites - 1

        for i in range(start, stop):
            self.left_compress_site(i, bra=bra, **compress_opts)

    def right_compress(self, start=None, stop=None, bra=None, **compress_opts):
        """Compress this 1D TN, from right to left, such that it becomes
        right-canonical (unless ``absorb != 'left'``).

        Parameters
        ----------
        start : int, optional
            Site to begin compressing on.
        stop : int, optional
            Site to stop compressing at (won't itself be an isometry).
        bra : None or TensorNetwork like this one, optional
            If given, update this TN as well, assuming it to be the conjugate.
        compress_opts
            Supplied to :meth:`Tensor.split`.
        """
        if start is None:
            start = self.nsites - 1
        if stop is None:
            stop = 0

        for i in range(start, stop, -1):
            self.right_compress_site(i, bra=bra, **compress_opts)

    def compress(self, form=None, **compress_opts):
        """Compress this 1D Tensor Network, possibly into canonical form.

        Parameters
        ----------
        form : {None, 'flat', 'left', 'right'} or int
            Output form of the TN. ``None`` left canonizes the state first for
            stability reasons, then right_compresses (default). ``'flat'``
            tries to distribute the singular values evenly -- state will not
            be canonical. ``'left'`` and ``'right'`` put the state into left
            and right canonical form respectively without a prior sweep, or an
            int will put the state into mixed canonical form at that site.
        compress_opts
            Supplied to :meth:`Tensor.split`.
        """
        if form is None:
            self.left_canonize(bra=compress_opts.get('bra', None))
            self.right_compress(**compress_opts)

        elif isinstance(form, Integral):
            self.left_compress(stop=form, **compress_opts)
            self.right_compress(stop=form, **compress_opts)

        elif form == 'left':
            self.left_compress(**compress_opts)
        elif form == 'right':
            self.right_compress(**compress_opts)

        elif form == 'flat':
            compress_opts['absorb'] = 'both'
            self.right_compress(stop=self.nsites // 2, **compress_opts)
            self.left_compress(stop=self.nsites // 2, **compress_opts)

        else:
            raise ValueError("Form specifier {} not understood, should be "
                             "either 'left', 'right', 'flat' or an int "
                             "specifiying a new orthog center.".format(form))

    def bond(self, i, j):
        """Get the name of the index defining the bond between sites i and j.
        """
        bond, = self[i].bonds(self[j])
        return bond

    def bond_size(self, i, j):
        """Return the size of the bond between site ``i`` and ``j``.
        """
        b_ix = self.bond(i, j)
        return self[i].ind_size(b_ix)

    def bond_sizes(self):
        bnd_szs = [self.bond_size(i, i + 1) for i in range(self.nsites - 1)]
        if self.cyclic:
            bnd_szs.append(self.bond_size(-1, 0))
        return bnd_szs

    def fuse_multibonds(self, inplace=False):
        """Fuse any double/triple etc bonds between neighbours
        """
        tn = self if inplace else self.copy()

        for i, j in pairwise(tn.sites):
            T1, T2 = tn[i], tn[j]
            dbnds = tuple(T1.bonds(T2))
            T1.fuse_({dbnds[0]: dbnds})
            T2.fuse_({dbnds[0]: dbnds})

        return tn

    def singular_values(self, i, cur_orthog=None, method='svd'):
        r"""Find the singular values associated with the ith bond::

            ....L....   i
            o-o-o-o-o-l-o-o-o-o-o-o-o-o-o-o-o
            | | | | |   | | | | | | | | | | |
                   i-1  ..........R..........

        Leaves the 1D TN in mixed canoncial form at bond ``i``.

        Parameters
        ----------
        i : int
            Which bond, or equivalently, the number of sites in the
            left partition.
        cur_orthog : int
            If given, the known current orthogonality center, to speed up the
            mixed canonization, e.g. if sweeping this function from left to
            right would use ``i - 1``.

        Returns
        -------
        svals : 1d-array
            The singular values.
        """
        if not (0 < i < self.nsites):
            raise ValueError("Need 0 < i < {}, got i={}."
                             .format(self.nsites, i))

        self.canonize(i, cur_orthog)

        Tm1 = self[i]
        left_inds = Tm1.bonds(self[i - 1])
        return Tm1.singular_values(left_inds, method=method)

    def expand_bond_dimension(self, new_bond_dim, inplace=True, bra=None,
                              rand_strength=0.0):
        """Expand the bond dimensions of this 1D tensor network to at least
        ``new_bond_dim``.

        Parameters
        ----------
        new_bond_dim : int
            Minimum bond dimension to expand to.
        inplace : bool, optional
            Whether to perform the expansion in place.
        bra : MatrixProductState, optional
            Mirror the changes to ``bra`` inplace, treating it as the conjugate
            state.
        rand_strength : float, optional
            If ``rand_strength > 0``, fill the new tensor entries with gaussian
            noise of strength ``rand_strength``.

        Returns
        -------
        MatrixProductState
        """
        if inplace:
            expanded = self
        else:
            expanded = self.copy()

        for i in self.sites:
            tensor = expanded[i]
            to_expand = []

            if i > 0 or self.cyclic:
                to_expand.append(self.bond(i - 1, i))
            if i < self.nsites - 1 or self.cyclic:
                to_expand.append(self.bond(i, i + 1))

            pads = [(0, 0) if i not in to_expand else
                    (0, max(new_bond_dim - d, 0))
                    for d, i in zip(tensor.shape, tensor.inds)]

            if rand_strength > 0:
                edata = np.pad(tensor.data, pads, mode=rand_padder,
                               rand_strength=rand_strength)
            else:
                edata = np.pad(tensor.data, pads, mode='constant')

            tensor.modify(data=edata)

            if bra is not None:
                bra[i].modify(data=tensor.data.conj())

        return expanded

    def count_canonized(self):
        if self.cyclic:
            return 0, 0

        ov = self.H & self
        num_can_l = 0
        num_can_r = 0

        def isidentity(x):
            d = x.shape[0]
            if x.dtype in ('float32', 'complex64'):
                rtol, atol = 1e-5, 1e-6
                idtty = np.eye(d, dtype='float32')
            else:
                rtol, atol = 1e-9, 1e-11
                idtty = np.eye(d, dtype='float64')
            return np.allclose(x, idtty, rtol=rtol, atol=atol)

        for i in range(self.nsites - 1):
            ov ^= slice(max(0, i - 1), i + 1)
            x = ov[i].data
            if isidentity(x):
                num_can_l += 1
            else:
                break

        for j in reversed(range(i + 1, self.nsites)):
            ov ^= slice(j, min(self.nsites, j + 2))
            x = ov[j].data
            if isidentity(x):
                num_can_r += 1
            else:
                break

        return num_can_l, num_can_r

    def calc_current_orthog_center(self):
        """Calculate the site(s) of the current orthogonality center.

        Returns
        -------
        int or (int, int)
            The site, or min/max, around which this MPS is orthogonal.
        """
        lo, ro = self.count_canonized()
        i, j = lo, self.nsites - ro - 1
        return i if i == j else i, j

    def as_cyclic(self, inplace=False):
        """Convert this flat, 1D, TN into cyclic form by adding a dummy bond
        between the first and last sites.
        """
        tn = self if inplace else self.copy()

        # nothing to do
        if tn.cyclic:
            return tn

        tn.add_bond(0, -1)
        tn.cyclic = True
        return tn

    def show(self, max_width=None):
        l1 = ""
        l2 = ""
        l3 = ""
        num_can_l, num_can_r = self.count_canonized()
        for i in range(len(self.sites) - 1):
            bdim = self.bond_size(self.sites[i], self.sites[i + 1])
            strl = len(str(bdim))
            l1 += " {}".format(bdim)
            l2 += (">" if i < num_can_l else
                   "<" if i >= self.nsites - num_can_r else
                   "o") + ("-" if bdim < 100 else "=") * strl
            l3 += "|" + " " * strl
            strl = len(str(bdim))

        l1 += " "
        l2 += "<" if num_can_r > 0 else "o"
        l3 += "|"

        if self.cyclic:
            bdim = self.bond_size(self.sites[0], self.sites[-1])
            bnd_str = ("-" if bdim < 100 else "=") * strl
            l1 = " {}{}{} ".format(bdim, l1, bdim)
            l2 = "+{}{}{}+".format(bnd_str, l2, bnd_str)
            l3 = " {}{}{} ".format(" " * strl, l3, " " * strl)

        three_line_multi_print(l1, l2, l3, max_width=max_width)


class MatrixProductState(TensorNetwork1DVector,
                         TensorNetwork1DFlat,
                         TensorNetwork1D,
                         TensorNetwork):
    """Initialise a matrix product state, with auto labelling and tagging.

    Parameters
    ----------
    arrays : sequence of arrays
        The tensor arrays to form into a MPS.
    shape : str, optional
        String specifying layout of the tensors. E.g. 'lrp' (the default)
        indicates the shape corresponds left-bond, right-bond, physical index.
        End tensors have either 'l' or 'r' dropped from the string.
    site_ind_id : str
        A string specifiying how to label the physical site indices. Should
        contain a ``'{}'`` placeholder. It is used to generate the actual
        indices like: ``map(site_ind_id.format, range(len(arrays)))``.
    site_tag_id : str
        A string specifiying how to tag the tensors at each site. Should
        contain a ``'{}'`` placeholder. It is used to generate the actual tags
        like: ``map(site_tag_id.format, range(len(arrays)))``.
    tags : str or sequence of str, optional
        Global tags to attach to all tensors.
    bond_name : str, optional
        The base name of the bond indices, onto which uuids will be added.
    """

    _EXTRA_PROPS = ('_site_ind_id', '_site_tag_id', 'cyclic')

    def __init__(self, arrays, *, shape='lrp', tags=None, bond_name="",
                 site_ind_id='k{}', site_tag_id='I{}', sites=None, nsites=None,
                 **tn_opts):

        # short-circuit for copying MPSs
        if isinstance(arrays, MatrixProductState):
            super().__init__(arrays)
            for ep in MatrixProductState._EXTRA_PROPS:
                setattr(self, ep, getattr(arrays, ep))
            return

        arrays = tuple(arrays)

        if sites is None:
            if nsites is None:
                nsites = len(arrays)
            sites = range(nsites)

        # process site indices
        self._site_ind_id = site_ind_id
        site_inds = map(site_ind_id.format, sites)

        # process site tags
        self._site_tag_id = site_tag_id
        site_tags = map(site_tag_id.format, sites)

        if tags is not None:
            if isinstance(tags, str):
                tags = {tags}
            else:
                tags = set(tags)

            site_tags = tuple({st} | tags for st in site_tags)

        self.cyclic = (_ndim(arrays[0]) == 3)

        # transpose arrays to 'lrp' order.
        def gen_orders():
            lp_ord = tuple(shape.replace('r', "").find(x) for x in 'lp')
            lrp_ord = tuple(shape.find(x) for x in 'lrp')
            rp_ord = tuple(shape.replace('l', "").find(x) for x in 'rp')
            yield lp_ord if not self.cyclic else lrp_ord
            for _ in range(len(sites) - 2):
                yield lrp_ord
            yield rp_ord if not self.cyclic else lrp_ord

        def gen_inds():
            cyc_bond = (rand_uuid(base=bond_name),) if self.cyclic else ()

            nbond = rand_uuid(base=bond_name)
            yield cyc_bond + (nbond, next(site_inds))
            pbond = nbond
            for _ in range(len(sites) - 2):
                nbond = rand_uuid(base=bond_name)
                yield (pbond, nbond, next(site_inds))
                pbond = nbond
            yield (pbond,) + cyc_bond + (next(site_inds),)

        def gen_tensors():
            for array, site_tag, inds, order in zip(arrays, site_tags,
                                                    gen_inds(), gen_orders()):
                yield Tensor(array.transpose(*order), inds=inds, tags=site_tag)

        super().__init__(gen_tensors(), structure=site_tag_id, sites=sites,
                         nsites=nsites, check_collisions=False, **tn_opts)

    @classmethod
    def from_TN(cls, tn, site_ind_id, site_tag_id,
                cyclic=False, inplace=False):
        """Convert a ``TensorNetwork`` into a ``MatrixProductState``, assuming
        it has the appropirate underlying structure.

        Parameters
        ----------
        tn : TensorNetwork
            The tensor network to convert.
        site_ind_id : str
            The string formatter that specifies the site indices -- it should
            match what the tensors of ``tn`` already have.
        site_tag_id : str
            The string formatter that specifies the site tags -- it should
            match what the tensors of ``tn`` already have.
        inplace : bool, optional
            If True, perform the conversion in-place.
        """
        if not inplace:
            tn = tn.copy()
        tn.__class__ = cls
        tn._site_ind_id = site_ind_id
        tn._site_tag_id = site_tag_id
        tn.cyclic = cyclic
        return tn

    @classmethod
    def from_dense(cls, psi, dims, site_ind_id='k{}',
                   site_tag_id='I{}', **split_opts):
        """Create a ``MatrixProductState`` directly from a dense vector

        Parameters
        ----------
        psi : array_like
            The dense state to convert to MPS from.
        dims : sequence of int
            Physical subsystem dimensions of each site.
        site_ind_id : str, optional
            How to index the physical sites, see
            :class:`~quimb.tensor.tensor_1d.MatrixProductState`.
        site_tag_id : str, optional
            How to tag the physical sites, see
            :class:`~quimb.tensor.tensor_1d.MatrixProductState`.
        split_opts
            Supplied to :func:`~quimb.tensor.tensor_core.tensor_split` to
            in order to partition the dense vector into tensors.

        Returns
        -------
        MatrixProductState

        Examples
        --------

            >>> dims = [2, 2, 2, 2, 2, 2]
            >>> psi = rand_ket(prod(dims))
            >>> mps = MatrixProductState.from_dense(psi, dims)
            >>> mps.show()
             2 4 8 4 2
            o-o-o-o-o-o
            | | | | | |
        """
        n = len(dims)
        inds = [site_ind_id.format(i) for i in range(n)]

        T = Tensor(psi.A.reshape(dims), inds=inds)

        def gen_tensors():
            #           split
            #       <--  : yield
            #            : :
            #     OOOOOOO--O-O-O
            #     |||||||  | | |
            #     .......
            #    left_inds
            TM = T
            for i in range(n - 1, 0, -1):
                TM, TR = TM.split(left_inds=inds[:i], get='tensors',
                                  rtags={site_tag_id.format(i)}, **split_opts)
                yield TR
            TM.tags.add(site_tag_id.format(0))
            yield TM

        tn = TensorNetwork(gen_tensors(), structure='I{}')
        return cls.from_TN(tn, site_ind_id, site_tag_id)

    def imprint(self, other):
        """Cast ``other`` into a ``MatrixProductState`` like ``self``.
        """
        for p in MatrixProductState._EXTRA_PROPS:
            setattr(other, p, getattr(self, p))
        other.__class__ = MatrixProductState

    def add_MPS(self, other, inplace=False, compress=False, **compress_opts):
        """Add another MatrixProductState to this one.
        """
        N = self.nsites

        if N != other.nsites:
            raise ValueError("Can't add MPS with another of different length.")

        new_mps = self if inplace else self.copy()

        for i in new_mps.sites:
            t1, t2 = new_mps[i], other[i]

            if set(t1.inds) != set(t2.inds):
                # Need to use bonds to match indices
                reindex_map = {}

                if i > 0 or self.cyclic:
                    pair = ((i - 1) % N, i)
                    reindex_map[other.bond(*pair)] = new_mps.bond(*pair)

                if i < new_mps.nsites - 1 or self.cyclic:
                    pair = (i, (i + 1) % N)
                    reindex_map[other.bond(*pair)] = new_mps.bond(*pair)

                t2 = t2.reindex(reindex_map)

            t1.direct_product(t2, inplace=True, sum_inds=new_mps.site_ind(i))

        if compress:
            new_mps.compress(**compress_opts)

        return new_mps

    def __add__(self, other):
        """MPS addition.
        """
        return self.add_MPS(other, inplace=False)

    def __iadd__(self, other):
        """In-place MPS addition.
        """
        return self.add_MPS(other, inplace=True)

    def __sub__(self, other):
        """MPS subtraction.
        """
        return self.add_MPS(other * -1, inplace=False)

    def __isub__(self, other):
        """In-place MPS subtraction.
        """
        return self.add_MPS(other * -1, inplace=True)

    def normalize(self, bra=None, eps=1e-15, insert=None):
        """Normalize this MPS, optional with co-vector ``bra``. For periodic
        MPS this uses transfer matrix SVD approximation with precision ``eps``
        in order to be efficient. Inplace.

        Parameters
        ----------
        bra : MatrixProductState, optional
            If given, normalize this MPS with the same factor.
        eps : float, optional
            If cyclic, precision to approximation transfer matrix with.
            Default: 1e-14.
        insert : int, optional
            Insert the corrective normalization on this site, random if
            not given.

        Returns
        -------
        old_norm : float
            The old norm ``self.H @ self``.
        """
        norm = expec_TN_1D(self.H, self, eps=eps)

        if insert is None:
            insert = -1

        self[insert].modify(data=self[insert].data / norm ** 0.5)
        if bra is not None:
            bra[insert].modify(data=bra[insert].data / norm ** 0.5)

        return norm

    def gate_split(self, G, where, inplace=False, **compress_opts):
        r"""Apply a two-site gate and then split resulting tensor to retrieve a
        MPS form::

            -o-o-A-B-o-o-
             | | | | | |            -o-o-GGG-o-o-           -o-o-X~Y-o-o-
             | | GGG | |     ==>     | | | | | |     ==>     | | | | | |
             | | | | | |                 i j                     i j
                 i j

        As might be found in TEBD.

        Parameters
        ----------
        G : array
            The gate, with shape ``(d**2, d**2)`` for physical dimension ``d``.
        where : (int, int)
            Indices of the sites to apply the gate to.
        compress_opts
            Supplied to :func:`~quimb.tensor.tensor_split`.

        See Also
        --------
        gate, gate_with_auto_swap
        """
        tn = self if inplace else self.copy()

        i, j = where

        Ti, Tj = tn[i], tn[j]
        ix_i, ix_j = tn.site_ind(i), tn.site_ind(j)

        # Make Tensor of gate
        d = tn.phys_dim(i)
        TG = Tensor(_asarray(G).reshape(d, d, d, d),
                    inds=("_tmpi", "_tmpj", ix_i, ix_j))

        # Contract gate into the two sites
        TG = TG.contract(Ti, Tj)
        TG.reindex_({"_tmpi": ix_i, "_tmpj": ix_j})

        # Split the tensor
        _, left_ix = Ti.filter_bonds(Tj)
        nTi, nTj = TG.split(left_inds=left_ix, get='tensors', **compress_opts)

        # make sure the new data shape matches and reinsert
        Ti.modify(data=nTi.transpose_like_(Ti).data)
        Tj.modify(data=nTj.transpose_like_(Tj).data)

        return tn

    gate_split_ = functools.partialmethod(gate_split, inplace=True)

    def swap_sites_with_compress(self, i, j, cur_orthog=None,
                                 inplace=False, **compress_opts):
        """Swap sites ``i`` and ``j`` by contracting, then splitting with the
        physical indices swapped.

        Parameters
        ----------
        i : int
            The first site to swap.
        j : int
            The second site to swap.
        cur_orthog : int, sequence of int, or 'calc'
            If known, the current orthogonality center.
        inplace : bond, optional
            Perform the swaps inplace.
        compress_opts
            Supplied to :func:`~quimb.tensor.tensor_core.tensor_split`.
        """
        i, j = sorted((i, j))
        if i + 1 != j:
            raise ValueError("Sites aren't adjacent.")

        mps = self if inplace else self.copy()
        mps.canonize((i, j), cur_orthog)

        # get site tensors and indices
        ix_i, ix_j = map(mps.site_ind, (i, j))
        Ti, Tj = mps[i], mps[j]
        _, unshared = Ti.filter_bonds(Tj)

        # split the contracted tensor, swapping the site indices
        Tij = Ti @ Tj
        lix = [i for i in unshared if i != ix_i] + [ix_j]
        sTi, sTj = Tij.split(lix, get='tensors', **compress_opts)

        # reindex and tranpose the tensors to directly update original tensors
        sTi.reindex_({ix_j: ix_i})
        sTj.reindex_({ix_i: ix_j})
        sTi.transpose_like_(Ti)
        sTj.transpose_like_(Tj)

        Ti.modify(data=sTi.data)
        Tj.modify(data=sTj.data)

        return mps

    def swap_site_to(self, i, f, cur_orthog=None,
                     inplace=False, **compress_opts):
        r"""Swap site ``i`` to site ``f``, compressing the bond after each
        swap::

                  i       f
            0 1 2 3 4 5 6 7 8 9      0 1 2 4 5 6 7 3 8 9
            o-o-o-x-o-o-o-o-o-o      o-o-o-o-o-o-o-x-o-o
            | | | | | | | | | |  ->  | | | | | | | | | |


        Parameters
        ----------
        i : int
            The site to move.
        f : int
            The new location for site ``i``.
        cur_orthog : int, sequence of int, or 'calc'
            If known, the current orthogonality center.
        inplace : bond, optional
            Perform the swaps inplace.
        compress_opts
            Supplied to :func:`~quimb.tensor.tensor_core.tensor_split`.
        """
        mps = self if inplace else self.copy()

        if i == f:
            return mps
        if i < f:
            js = range(i, f)
        if f < i:
            js = range(i - 1, f - 1, -1)

        for j in js:
            mps.swap_sites_with_compress(
                j, j + 1, inplace=True, cur_orthog=cur_orthog, **compress_opts)
            cur_orthog = (j, j + 1)

        return mps

    def gate_with_auto_swap(self, G, where, inplace=False,
                            cur_orthog=None, **compress_opts):
        """Perform a two site gate on this MPS by, if necessary, swapping and
        compressing the sites until they are adjacent, using ``gate_split``,
        then unswapping the sites back to their original position.

        Parameters
        ----------
        G : array
            The gate, with shape ``(d**2, d**2)`` for physical dimension ``d``.
        where : (int, int)
            Indices of the sites to apply the gate to.
        cur_orthog : int, sequence of int, or 'calc'
            If known, the current orthogonality center.
        inplace : bond, optional
            Perform the swaps inplace.
        compress_opts
            Supplied to :func:`~quimb.tensor.tensor_core.tensor_split`.

        See Also
        --------
        gate, gate_split
        """
        mps = self if inplace else self.copy()

        i, j = sorted(where)
        need2swap = i + 1 != j

        # move j site adjacent to i site
        if need2swap:
            mps.swap_site_to(j, i + 1, cur_orthog=cur_orthog,
                             inplace=True, **compress_opts)
            cur_orthog = (i + 1, i + 2)

        # make sure sites are orthog center, then apply and split
        mps.canonize((i, i + 1), cur_orthog)
        mps.gate_split_(G, (i, i + 1), **compress_opts)

        # move j site back to original position
        if need2swap:
            mps.swap_site_to(i + 1, j, cur_orthog=(i, i + 1),
                             inplace=True, **compress_opts, )

        return mps

    def magnetization(self, i, direction='Z', cur_orthog=None):
        """Compute the magnetization at site ``i``.
        """
        if self.cyclic:
            msg = ("``magnetization`` currently makes use of orthogonality for"
                   " efficiencies sake, for cyclic systems is it still "
                   "possible to compute as a normal expectation.")
            raise NotImplementedError(msg)

        self.canonize(i, cur_orthog)

        # +-k-+
        # | O |
        # +-b-+

        Tk = self[i]
        ind1, ind2 = self.site_ind(i), '__tmp__'
        Tb = Tk.H.reindex({ind1: ind2})

        O_data = qu.spin_operator(direction, S=(self.phys_dim(i) - 1) / 2)
        TO = Tensor(O_data, inds=(ind1, ind2))

        return Tk.contract(TO, Tb)

    def schmidt_values(self, i, cur_orthog=None, method='svd'):
        r"""Find the schmidt values associated with the bipartition of this
        MPS between sites on either site of ``i``. In other words, ``i`` is the
        number of sites in the left hand partition::

            ....L....   i
            o-o-o-o-o-S-o-o-o-o-o-o-o-o-o-o-o
            | | | | |   | | | | | | | | | | |
                   i-1  ..........R..........

        The schmidt values, ``S``, are the singular values associated with the
        ``(i - 1, i)`` bond, squared, provided the MPS is mixed canonized at
        one of those sites.

        Parameters
        ----------
        i : int
            The number of sites in the left partition.
        cur_orthog : int
            If given, the known current orthogonality center, to speed up the
            mixed canonization.

        Returns
        -------
        S : 1d-array
            The schmidt values.
        """
        if self.cyclic:
            raise NotImplementedError

        return self.singular_values(i, cur_orthog, method=method)**2

    def entropy(self, i, cur_orthog=None, method='svd'):
        """The entropy of bipartition between the left block of ``i`` sites and
        the rest.

        Parameters
        ----------
        i : int
            The number of sites in the left partition.
        cur_orthog : int
            If given, the known current orthogonality center, to speed up the
            mixed canonization.

        Returns
        -------
        float
        """
        if self.cyclic:
            msg = ("For cyclic systems, try explicitly computing the entropy "
                   "of the (compressed) reduced density matrix.")
            raise NotImplementedError(msg)

        S = self.schmidt_values(i, cur_orthog=cur_orthog, method=method)
        S = S[S > 0.0]
        return np.sum(-S * np.log2(S))

    def schmidt_gap(self, i, cur_orthog=None, method='svd'):
        """The schmidt gap of bipartition between the left block of ``i`` sites
        and the rest.

        Parameters
        ----------
        i : int
            The number of sites in the left partition.
        cur_orthog : int
            If given, the known current orthogonality center, to speed up the
            mixed canonization.

        Returns
        -------
        float
        """
        if self.cyclic:
            raise NotImplementedError

        S = self.schmidt_values(i, cur_orthog=cur_orthog, method=method)

        if len(S) == 1:
            return S[0]

        return S[0] - S[1]

    def partial_trace(self, keep, upper_ind_id="b{}", rescale_sites=True):
        r"""Partially trace this matrix product state, producing a matrix
        product operator.

        Parameters
        ----------
        keep : sequence of int or slice
            Indicies of the sites to keep.
        upper_ind_id : str, optional
            The ind id of the (new) 'upper' inds, i.e. the 'bra' inds.
        rescale_sites : bool, optional
            If ``True`` (the default), then the kept sites will be rescaled to
            ``(0, 1, 2, ...)`` etc. rather than keeping their original site
            numbers.

        Returns
        -------
        rho : MatrixProductOperator
            The density operator in MPO form.
        """
        p_bra = self.copy()
        p_bra.reindex_sites(upper_ind_id, where=keep, inplace=True)
        rho = self.H & p_bra
        # now have e.g:
        #     | |     |   |
        # o-o-o-o-o-o-o-o-o
        # | |     | |   |
        # o-o-o-o-o-o-o-o-o
        #     | |     |   |

        if isinstance(keep, slice):
            keep = self.slice2sites(keep)

        keep = sorted(keep)
        n = len(keep)

        for i in self.sites:
            if i in keep:
                #      |
                #     -o-             |
                # ... -o- ... -> ... -O- ...
                #     i|             i|
                rho ^= self.site_tag(i)
            else:
                #        |
                #     -o-o-              |
                # ...  |    ... -> ... -OO- ...
                #     -o-o-              |i+1
                #      i |i+1
                if i < self.nsites - 1:
                    rho >>= [self.site_tag(i), self.site_tag(i + 1)]
                else:
                    rho >>= [self.site_tag(i), self.site_tag(max(keep))]

                rho.drop_tags(self.site_tag(i))

        # transpose upper and lower tags to match other MPOs
        rho = MatrixProductOperator.from_TN(
            rho, lower_ind_id=upper_ind_id, upper_ind_id=self.site_ind_id,
            cyclic=self.cyclic, site_tag_id=self.site_tag_id, inplace=True)

        if rescale_sites:
            # e.g. [3, 4, 5, 7, 9] -> [0, 1, 2, 3, 4]
            retag, reind = {}, {}
            for new, old in enumerate(keep):
                retag[self.site_tag(old)] = self.site_tag(new)
                reind[rho.lower_ind(old)] = rho.lower_ind(new)
                reind[rho.upper_ind(old)] = rho.upper_ind(new)

            rho.retag_(retag)
            rho.reindex_(reind)

            rho.nsites = n
            rho.sites = range(n)
        else:
            rho.sites = keep

        rho.fuse_multibonds(inplace=True)
        return rho

    def ptr(self, keep, upper_ind_id="b{}", rescale_sites=True):
        """Alias of :meth:`~quimb.tensor.MatrixProductState.partial_trace`.
        """
        return self.partial_trace(keep, upper_ind_id,
                                  rescale_sites=rescale_sites)

    def bipartite_schmidt_state(self, sz_a, get='ket', cur_orthog=None):
        r"""Compute the reduced state for a bipartition of an OBC MPS, in terms
        of the minimal left/right schmidt basis::

                A            B
            .........     ...........
            >->->->->--s--<-<-<-<-<-<    ->   +-s-+
            | | | | |     | | | | | |         |   |
           k0 k1...                          kA   kB

        Parameters
        ----------
        sz_a : int
            The number of sites in subsystem A, must be ``0 < sz_a < N``.
        get : {'ket', 'rho', 'ket-dense', 'rho-dense'}, optional
            Get the:

            - 'ket': vector form as tensor.
            - 'rho': density operator form, i.e. vector outer product
            - 'ket-dense': like 'ket' but return ``numpy.matrix``.
            - 'rho-dense': like 'rho' but return ``numpy.matrix``.

        cur_orthog : int, optional
            If given, take as the current orthogonality center so as to
            efficienctly move it a minimal distance.
        """
        if self.cyclic:
            raise NotImplementedError("MPS must have OBC.")

        s = np.diag(self.singular_values(sz_a, cur_orthog=cur_orthog))

        if 'dense' in get:
            kd = qu.qarray(s.reshape(-1, 1))
            if 'ket' in get:
                return kd
            elif 'rho' in get:
                return kd @ kd.H

        else:
            k = Tensor(s, (self.site_ind('A'), self.site_ind('B')))
            if 'ket' in get:
                return k
            elif 'rho' in get:
                return k & k.reindex({'kA': 'bA', 'kB': 'bB'})

    @staticmethod
    def _do_lateral_compress(mps, kb, section, leave_short, ul, ll, heps,
                             hmethod, hmax_bond, verbosity, compressed,
                             **compress_opts):

        #           section
        #   ul -o-o-o-o-o-o-o-o-o-       ul -\       /-
        #       | | | | | | | | |   ==>       0~~~~~0
        #   ll -o-o-o-o-o-o-o-o-o-       ll -/   :   \-
        #                                      hmax_bond

        if leave_short:
            # if section is short doesn't make sense to lateral compress
            #     work out roughly when this occurs by comparing bond size
            left_sz = mps.bond_size(section[0] - 1, section[0])
            right_sz = mps.bond_size(section[-1], section[-1] + 1)

            if mps.phys_dim() ** len(section) <= left_sz * right_sz:
                if verbosity >= 1:
                    print("Leaving lateral compress of section '{}' as it is "
                          "too short: length={}, eff size={}."
                          .format(section, len(section), left_sz * right_sz))
                return

        if verbosity >= 1:
            print("Laterally compressing section {}. Using options: "
                  "eps={}, method={}, max_bond={}"
                  .format(section, heps, hmethod, hmax_bond))

        section_tags = map(mps.site_tag, section)
        kb.replace_with_svd(section_tags, (ul, ll), heps, inplace=True,
                            ltags='_LEFT', rtags='_RIGHT', method=hmethod,
                            max_bond=hmax_bond, **compress_opts)

        compressed.append(section)

    @staticmethod
    def _do_vertical_decomp(mps, kb, section, sysa, sysb, compressed, ul, ur,
                            ll, lr, vmethod, vmax_bond, veps, verbosity,
                            **compress_opts):
        if section == sysa:
            label = 'A'
        elif section == sysb:
            label = 'B'
        else:
            return

        section_tags = [mps.site_tag(i) for i in section]

        if section in compressed:

            #                    ----U----             |  <- vmax_bond
            #  -\      /-            /             ----U----
            #    L~~~~R     ==>      \       ==>
            #  -/      \-            /             ----D----
            #                    ----D----             |  <- vmax_bond

            # try and choose a sensible method
            if vmethod is None:
                left_sz = mps.bond_size(section[0] - 1, section[0])
                right_sz = mps.bond_size(section[-1], section[-1] + 1)
                if left_sz * right_sz <= 2**13:
                    # cholesky is not rank revealing
                    vmethod = 'eigh' if vmax_bond else 'cholesky'
                else:
                    vmethod = 'isvd'

            if verbosity >= 1:
                print("Performing vertical decomposition of section {}, "
                      "using options: eps={}, method={}, max_bond={}."
                      .format(label, veps, vmethod, vmax_bond))

            # do vertical SVD
            kb.replace_with_svd(
                section_tags, (ul, ur), right_inds=(ll, lr), eps=veps,
                ltags='_UP', rtags='_DOWN', method=vmethod, inplace=True,
                max_bond=vmax_bond, **compress_opts)

            # cut joined bond by reindexing to upper- and lower- ind_id.
            kb.cut_bond((mps.site_tag(section[0]), '_UP'),
                        (mps.site_tag(section[0]), '_DOWN'),
                        "_tmp_ind_u{}".format(label),
                        "_tmp_ind_l{}".format(label))

        else:
            # just unfold and fuse physical indices:
            #                              |
            #   -A-A-A-A-A-A-A-        -AAAAAAA-
            #    | | | | | | |   ===>
            #   -A-A-A-A-A-A-A-        -AAAAAAA-
            #                              |

            if verbosity >= 1:
                print("Just vertical unfolding section {}.".format(label))

            kb, sec = kb.partition(section_tags, inplace=True)
            sec_l, sec_u = sec.partition('_KET', inplace=True)
            T_UP = (sec_u ^ all)
            T_UP.add_tag('_UP')
            T_UP.fuse_({"_tmp_ind_u{}".format(label):
                        [mps.site_ind(i) for i in section]})
            T_DN = (sec_l ^ all)
            T_DN.add_tag('_DOWN')
            T_DN.fuse_({"_tmp_ind_l{}".format(label):
                        [mps.site_ind(i) for i in section]})
            kb |= T_UP
            kb |= T_DN

    def partial_trace_compress(self, sysa, sysb, eps=1e-8,
                               method=('isvd', None), max_bond=(None, 1024),
                               leave_short=True, renorm=True,
                               lower_ind_id='b{}', verbosity=0,
                               **compress_opts):
        r"""Perform a compressed partial trace using singular value
        lateral then vertical decompositions of transfer matrix products::


                    .....sysa......     ...sysb....
            o-o-o-o-A-A-A-A-A-A-A-A-o-o-B-B-B-B-B-B-o-o-o-o-o-o-o-o-o
            | | | | | | | | | | | | | | | | | | | | | | | | | | | | |

                                      ==> form inner product

                    ...............     ...........
            o-o-o-o-A-A-A-A-A-A-A-A-o-o-B-B-B-B-B-B-o-o-o-o-o-o-o-o-o
            | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
            o-o-o-o-A-A-A-A-A-A-A-A-o-o-B-B-B-B-B-B-o-o-o-o-o-o-o-o-o

                                      ==> lateral SVD on each section

                      .....sysa......     ...sysb....
                      /\             /\   /\         /\
              ... ~~~E  A~~~~~~~~~~~A  E~E  B~~~~~~~B  E~~~ ...
                      \/             \/   \/         \/

                                      ==> vertical SVD and unfold on A & B

                              |                 |
                      /-------A-------\   /-----B-----\
              ... ~~~E                 E~E             E~~~ ...
                      \-------A-------/   \-----B-----/
                              |                 |

        With various special cases including OBC or end spins included in
        subsytems.


        Parameters
        ----------
        sysa :  sequence of int
            The sites, which should be contiguous, defining subsystem A.
        sysb :  sequence of int
            The sites, which should be contiguous, defining subsystem B.
        eps : float or (float, float), optional
            Tolerance(s) to use when compressing the subsystem transfer
            matrices and vertically decomposing.
        method : str or (str, str), optional
            Method(s) to use for laterally compressing the state then
            vertially compressing subsytems.
        max_bond : int or (int, int), optional
            The maximum bond to keep for laterally compressing the state then
            vertially compressing subsytems.
        leave_short : bool, optional
            If True (the default), don't try to compress short sections.
        renorm : bool, optional
            If True (the default), renomalize the state so that ``tr(rho)==1``.
        lower_ind_id : str, optional
            The index id to create for the new density matrix, the upper_ind_id
            is automatically taken as the current site_ind_id.
        compress_opts : dict, optional
            If given, supplied to ``partial_trace_compress`` to govern how
            singular values are treated. See ``tensor_split``.
        verbosity : {0, 1}, optional
            How much information to print while performing the compressed
            partial trace.

        Returns
        -------
        rho_ab : TensorNetwork
            Density matrix tensor network with
            ``outer_inds = ('k0', 'k1', 'b0', 'b1')`` for example.
        """
        N = self.nsites

        if (len(sysa) + len(sysb) == N) and not self.cyclic:
            return self.bipartite_schmidt_state(len(sysa), get='rho')

        # parse horizontal and vertical svd tolerances and methods
        try:
            heps, veps = eps
        except (ValueError, TypeError):
            heps = veps = eps
        try:
            hmethod, vmethod = method
        except (ValueError, TypeError):
            hmethod = vmethod = method
        try:
            hmax_bond, vmax_bond = max_bond
        except (ValueError, TypeError):
            hmax_bond = vmax_bond = max_bond

        # the sequence of sites in each of the 'environment' sections
        envm = range(max(sysa) + 1, min(sysb))
        envl = range(0, min(sysa))
        envr = range(max(sysb) + 1, N)

        # spread norm, and if not cyclic put in mixed canonical form, taking
        # care that the orthogonality centre is in right place to use identity
        k = self.copy()
        k.left_canonize()
        k.right_canonize(max(sysa) + (bool(envm) or bool(envr)))

        # form the inner product
        b = k.conj()
        k.add_tag('_KET')
        b.add_tag('_BRA')
        kb = k | b

        # label the various partitions
        names = ('_ENVL', '_SYSA', '_ENVM', '_SYSB', '_ENVR')
        for name, where in zip(names, (envl, sysa, envm, sysb, envr)):
            if where:
                kb.add_tag(name, where=map(self.site_tag, where), which='any')

        if self.cyclic:
            # can combine right and left envs
            sections = [envm, sysa, sysb, (*envr, *envl)]
        else:
            sections = [envm]
            # if either system includes end, can ignore and use identity
            if 0 not in sysa:
                sections.append(sysa)
            if N - 1 not in sysb:
                sections.append(sysb)

        # ignore empty sections
        sections = list(filter(len, sections))

        # figure out the various indices
        ul_ur_ll_lrs = []
        for section in sections:

            #          ...section[i]....
            #   ul[i] -o-o-o-o-o-o-o-o-o- ur[i]
            #          | | | | | | | | |
            #   ll[i] -o-o-o-o-o-o-o-o-o- lr[i]

            st_left = self.site_tag(section[0] - 1)
            st_right = self.site_tag(section[0])
            ul, = bonds(kb['_KET', st_left], kb['_KET', st_right])
            ll, = bonds(kb['_BRA', st_left], kb['_BRA', st_right])

            st_left = self.site_tag(section[-1])
            st_right = self.site_tag(section[-1] + 1)
            ur, = bonds(kb['_KET', st_left], kb['_KET', st_right])
            lr, = bonds(kb['_BRA', st_left], kb['_BRA', st_right])

            ul_ur_ll_lrs.append((ul, ur, ll, lr))

        # lateral compress sections if long
        compressed = []
        for section, (ul, _, ll, _) in zip(sections, ul_ur_ll_lrs):
            self._do_lateral_compress(self, kb, section, leave_short, ul, ll,
                                      heps, hmethod, hmax_bond, verbosity,
                                      compressed, **compress_opts)

        # vertical compress and unfold system sections only
        for section, (ul, ur, ll, lr) in zip(sections, ul_ur_ll_lrs):
            self._do_vertical_decomp(self, kb, section, sysa, sysb, compressed,
                                     ul, ur, ll, lr, vmethod, vmax_bond, veps,
                                     verbosity, **compress_opts)

        if not self.cyclic:
            # check if either system is at end, and thus reduces to identities
            #
            #  A-A-A-A-A-A-A-m-m-m-            \-m-m-m-
            #  | | | | | | | | | |  ...  ==>     | | |  ...
            #  A-A-A-A-A-A-A-m-m-m-            /-m-m-m-
            #
            if 0 in sysa:
                # get neighbouring tensor
                if envm:
                    try:
                        TU = TD = kb['_ENVM', '_LEFT']
                    except KeyError:
                        # didn't lateral compress
                        TU = kb['_ENVM', '_KET', self.site_tag(envm[0])]
                        TD = kb['_ENVM', '_BRA', self.site_tag(envm[0])]
                else:
                    TU = kb['_SYSB', '_UP']
                    TD = kb['_SYSB', '_DOWN']
                ubnd, = kb['_KET', self.site_tag(sysa[-1])].bonds(TU)
                lbnd, = kb['_BRA', self.site_tag(sysa[-1])].bonds(TD)

                # delete the A system
                kb.delete('_SYSA')
                kb.reindex_({ubnd: "_tmp_ind_uA", lbnd: "_tmp_ind_lA"})
            else:
                # or else replace the left or right envs with identites since
                #
                #  >->->->-A-A-A-A-           +-A-A-A-A-
                #  | | | | | | | |  ...  ==>  | | | | |
                #  >->->->-A-A-A-A-           +-A-A-A-A-
                #
                kb.replace_with_identity('_ENVL', inplace=True)

            if N - 1 in sysb:
                # get neighbouring tensor
                if envm:
                    try:
                        TU = TD = kb['_ENVM', '_RIGHT']
                    except KeyError:
                        # didn't lateral compress
                        TU = kb['_ENVM', '_KET', self.site_tag(envm[-1])]
                        TD = kb['_ENVM', '_BRA', self.site_tag(envm[-1])]
                else:
                    TU = kb['_SYSA', '_UP']
                    TD = kb['_SYSA', '_DOWN']
                ubnd, = kb['_KET', self.site_tag(sysb[0])].bonds(TU)
                lbnd, = kb['_BRA', self.site_tag(sysb[0])].bonds(TD)

                # delete the B system
                kb.delete('_SYSB')
                kb.reindex_({ubnd: "_tmp_ind_uB", lbnd: "_tmp_ind_lB"})
            else:
                kb.replace_with_identity('_ENVR', inplace=True)

        kb.reindex_({
            '_tmp_ind_uA': self.site_ind('A'),
            '_tmp_ind_lA': lower_ind_id.format('A'),
            '_tmp_ind_uB': self.site_ind('B'),
            '_tmp_ind_lB': lower_ind_id.format('B'),
        })

        if renorm:
            # normalize
            norm = kb.trace(['kA', 'kB'], ['bA', 'bB'])

            ts = []
            tags = kb.tags

            # check if we have system A
            if '_SYSA' in tags:
                ts.extend(kb[sysa[0]])

            # check if we have system B
            if '_SYSB' in tags:
                ts.extend(kb[sysb[0]])

            # If we dont' have either (OBC with both at ends) use middle envm
            if len(ts) == 0:
                ts.extend(kb[envm[0]])

            nt = len(ts)

            # now spread the norm out among tensors
            for t in ts:
                t.modify(data=t.data / norm**(1 / nt))

        return kb

    def logneg_subsys(self, sysa, sysb, compress_opts=None,
                      approx_spectral_opts=None, verbosity=0,
                      approx_thresh=2**12):
        r"""Compute the logarithmic negativity between subsytem blocks, e.g.::

                               sysa         sysb
                             .........       .....
            ... -o-o-o-o-o-o-A-A-A-A-A-o-o-o-B-B-B-o-o-o-o-o-o-o- ...
                 | | | | | | | | | | | | | | | | | | | | | | | |

        Parameters
        ----------
        sysa :  sequence of int
            The sites, which should be contiguous, defining subsystem A.
        sysb :  sequence of int
            The sites, which should be contiguous, defining subsystem B.
        eps : float, optional
            Tolerance to use when compressing the subsystem transfer matrices.
        method : str or (str, str), optional
            Method(s) to use for laterally compressing the state then
            vertially compressing subsytems.
        compress_opts : dict, optional
            If given, supplied to ``partial_trace_compress`` to govern how
            singular values are treated. See ``tensor_split``.
        approx_spectral_opts
            Supplied to :func:`~quimb.approx_spectral_function`.

        Returns
        -------
        ln : float
            The logarithmic negativity.

        See Also
        --------
        MatrixProductState.partial_trace_compress, approx_spectral_function
        """
        if not self.cyclic and (len(sysa) + len(sysb) == self.nsites):
            # pure bipartition with OBC
            psi = self.bipartite_schmidt_state(len(sysa), get='ket-dense')
            d = round(psi.shape[0]**0.5)
            return qu.logneg(psi, [d, d])

        if compress_opts is None:
            compress_opts = {}
        if approx_spectral_opts is None:
            approx_spectral_opts = {}

        # set the default verbosity for each method
        compress_opts.setdefault('verbosity', verbosity)
        approx_spectral_opts.setdefault('verbosity', verbosity)

        # form the compressed density matrix representation
        rho_ab = self.partial_trace_compress(sysa, sysb, **compress_opts)

        # view it as an operator
        rho_ab_pt_lo = rho_ab.aslinearoperator(['kA', 'bB'], ['bA', 'kB'])

        if rho_ab_pt_lo.shape[0] <= approx_thresh:
            tr_norm = qu.norm(rho_ab_pt_lo.to_dense(), 'tr')
        else:
            # estimate its spectrum and sum the abs(eigenvalues)
            tr_norm = qu.approx_spectral_function(
                rho_ab_pt_lo, abs, **approx_spectral_opts)

        # clip below 0
        return max(0, log2(tr_norm))


class MatrixProductOperator(TensorNetwork1DFlat,
                            TensorNetwork1D,
                            TensorNetwork):
    """Initialise a matrix product operator, with auto labelling and tagging.

    Parameters
    ----------
    arrays : sequence of arrays
        The tensor arrays to form into a MPO.
    shape : str, optional
        String specifying layout of the tensors. E.g. 'lrud' (the default)
        indicates the shape corresponds left-bond, right-bond, 'up' physical
        index, 'down' physical index.
        End tensors have either 'l' or 'r' dropped from the string.
    upper_ind_id : str
        A string specifiying how to label the upper physical site indices.
        Should contain a ``'{}'`` placeholder. It is used to generate the
        actual indices like: ``map(upper_ind_id.format, range(len(arrays)))``.
    lower_ind_id : str
        A string specifiying how to label the lower physical site indices.
        Should contain a ``'{}'`` placeholder. It is used to generate the
        actual indices like: ``map(lower_ind_id.format, range(len(arrays)))``.
    site_tag_id : str
        A string specifiying how to tag the tensors at each site. Should
        contain a ``'{}'`` placeholder. It is used to generate the actual tags
        like: ``map(site_tag_id.format, range(len(arrays)))``.
    tags : str or sequence of str, optional
        Global tags to attach to all tensors.
    bond_name : str, optional
        The base name of the bond indices, onto which uuids will be added.
    """

    _EXTRA_PROPS = ('_upper_ind_id', '_lower_ind_id', '_site_tag_id', 'cyclic')

    def __init__(self, arrays, shape='lrud', site_tag_id='I{}', tags=None,
                 upper_ind_id='k{}', lower_ind_id='b{}', bond_name="",
                 sites=None, nsites=None, **tn_opts):
        # short-circuit for copying
        if isinstance(arrays, MatrixProductOperator):
            super().__init__(arrays)
            for ep in MatrixProductOperator._EXTRA_PROPS:
                setattr(self, ep, getattr(arrays, ep))
            return

        arrays = tuple(arrays)

        if sites is None:
            if nsites is None:
                nsites = len(arrays)
            sites = range(nsites)

        # process site indices
        self._upper_ind_id = upper_ind_id
        self._lower_ind_id = lower_ind_id
        upper_inds = map(upper_ind_id.format, sites)
        lower_inds = map(lower_ind_id.format, sites)

        # process site tags
        self._site_tag_id = site_tag_id
        site_tags = map(site_tag_id.format, sites)
        if tags is not None:
            if isinstance(tags, str):
                tags = (tags,)
            else:
                tags = tuple(tags)

            site_tags = tuple((st,) + tags for st in site_tags)

        self.cyclic = (_ndim(arrays[0]) == 4)

        # transpose arrays to 'lrud' order.
        def gen_orders():
            lud_ord = tuple(shape.replace('r', "").find(x) for x in 'lud')
            rud_ord = tuple(shape.replace('l', "").find(x) for x in 'rud')
            lrud_ord = tuple(map(shape.find, 'lrud'))
            yield rud_ord if not self.cyclic else lrud_ord
            for _ in range(len(sites) - 2):
                yield lrud_ord
            yield lud_ord if not self.cyclic else lrud_ord

        def gen_inds():
            cyc_bond = (rand_uuid(base=bond_name),) if self.cyclic else ()

            nbond = rand_uuid(base=bond_name)
            yield (*cyc_bond, nbond, next(upper_inds), next(lower_inds))
            pbond = nbond
            for _ in range(len(sites) - 2):
                nbond = rand_uuid(base=bond_name)
                yield (pbond, nbond, next(upper_inds), next(lower_inds))
                pbond = nbond
            yield (pbond, *cyc_bond, next(upper_inds), next(lower_inds))

        def gen_tensors():
            for array, site_tag, inds, order in zip(arrays, site_tags,
                                                    gen_inds(), gen_orders()):

                yield Tensor(array.transpose(*order), inds=inds, tags=site_tag)

        super().__init__(gen_tensors(), structure=site_tag_id, sites=sites,
                         nsites=nsites, check_collisions=False, **tn_opts)

    @classmethod
    def from_TN(cls, tn, upper_ind_id, lower_ind_id, site_tag_id,
                cyclic=False, inplace=False):
        """Convert a TensorNetwork into a MatrixProductOperator, assuming it
        has the appropirate underlying structure.

        Parameters
        ----------
        tn : TensorNetwork
            The tensor network to convert.
        upper_ind_id : str
            The string formatter that specifies the upper indices -- it should
            match what the tensors of ``tn`` already have.
        lower_ind_id : str
            The string formatter that specifies the lower indices -- it should
            match what the tensors of ``tn`` already have.
        site_tag_id : str
            The string formatter that specifies the site tags -- it should
            match what the tensors of ``tn`` already have.
        inplace : bool, optional
            If True, perform the conversion in-place.
        """
        if not inplace:
            tn = tn.copy()
        tn.__class__ = cls
        tn._upper_ind_id = upper_ind_id
        tn._lower_ind_id = lower_ind_id
        tn._site_tag_id = site_tag_id
        tn.cyclic = cyclic
        return tn

    def imprint(self, other):
        """Cast ``other`` into a ``MatrixProductOperator`` like ``self``.
        """
        for p in MatrixProductOperator._EXTRA_PROPS:
            setattr(other, p, getattr(self, p))
        other.__class__ = MatrixProductOperator

    def reindex_lower_sites(self, new_id, where=None, inplace=False):
        """Update the lower site index labels to a new string specifier.

        Parameters
        ----------
        new_id : str
            A string with a format placeholder to accept an int, e.g. "ket{}".
        where : None or slice
            Which sites to update the index labels on. If ``None`` (default)
            all sites.
        inplace : bool
            Whether to reindex in place.
        """
        if where is None:
            start = 0
            stop = self.nsites
        else:
            start = 0 if where.start is None else where.start
            stop = self.nsites if where.stop is ... else where.stop

        return self.reindex({self.lower_ind(i): new_id.format(i)
                             for i in range(start, stop)}, inplace=inplace)

    def reindex_upper_sites(self, new_id, where=None, inplace=False):
        """Update the upper site index labels to a new string specifier.

        Parameters
        ----------
        new_id : str
            A string with a format placeholder to accept an int, e.g. "ket{}".
        where : None or slice
            Which sites to update the index labels on. If ``None`` (default)
            all sites.
        inplace : bool
            Whether to reindex in place.
        """
        if where is None:
            start = 0
            stop = self.nsites
        else:
            start = 0 if where.start is None else where.start
            stop = self.nsites if where.stop is ... else where.stop

        return self.reindex({self.upper_ind(i): new_id.format(i)
                             for i in range(start, stop)}, inplace=inplace)

    def _get_lower_ind_id(self):
        return self._lower_ind_id

    def _set_lower_ind_id(self, new_id):
        if new_id == self._upper_ind_id:
            raise ValueError("Setting the same upper and lower index ids will"
                             " make the two ambiguous.")

        if self._lower_ind_id != new_id:
            self.reindex_lower_sites(new_id, inplace=True)
            self._lower_ind_id = new_id

    lower_ind_id = property(_get_lower_ind_id, _set_lower_ind_id,
                            doc="The string specifier for the lower phyiscal "
                            "indices")

    def lower_ind(self, i):
        """The name of the lower ('ket') index at site ``i``.
        """
        return self.lower_ind_id.format(i)

    def _get_upper_ind_id(self):
        return self._upper_ind_id

    def _set_upper_ind_id(self, new_id):
        if new_id == self._lower_ind_id:
            raise ValueError("Setting the same upper and lower index ids will"
                             " make the two ambiguous.")

        if self._upper_ind_id != new_id:
            self.reindex_upper_sites(new_id, inplace=True)
            self._upper_ind_id = new_id

    upper_ind_id = property(_get_upper_ind_id, _set_upper_ind_id,
                            doc="The string specifier for the upper phyiscal "
                            "indices")

    def upper_ind(self, i):
        """The name of the upper ('bra') index at site ``i``.
        """
        return self.upper_ind_id.format(i)

    def add_MPO(self, other, inplace=False, compress=False, **compress_opts):
        """Add another MatrixProductState to this one.
        """
        N = self.nsites

        if N != other.nsites:
            raise ValueError("Can't add MPO with another of different length."
                             "Got lengths {} and {}".format(N, other.nsites))

        summed = self if inplace else self.copy()

        for i in summed.sites:
            t1, t2 = summed[i], other[i]

            if set(t1.inds) != set(t2.inds):
                # Need to use bonds to match indices
                reindex_map = {}

                if i > 0 or self.cyclic:
                    pair = ((i - 1) % N, i)
                    reindex_map[other.bond(*pair)] = summed.bond(*pair)

                if i < summed.nsites - 1 or self.cyclic:
                    pair = (i, (i + 1) % N)
                    reindex_map[other.bond(*pair)] = summed.bond(*pair)

                t2 = t2.reindex(reindex_map)

            sum_inds = (summed.upper_ind(i), summed.lower_ind(i))
            t1.direct_product(t2, inplace=True, sum_inds=sum_inds)

        if compress:
            summed.compress(**compress_opts)

        return summed

    def _apply_mps(self, other, compress=True, **compress_opts):
        # import pdb; pdb.set_trace()
        A, x = self.copy(), other.copy()

        # align the indices
        A.upper_ind_id = "__tmp{}__"
        A.lower_ind_id = x.site_ind_id
        x.reindex_sites("__tmp{}__", inplace=True)

        # form total network and contract each site
        x |= A
        for i in range(x.nsites):
            x ^= x.site_tag(i)

        x.fuse_multibonds(inplace=True)

        # optionally compress
        if compress:
            x.compress(**compress_opts)

        return x

    def _apply_mpo(self, other, compress=False, **compress_opts):
        A, B = self.copy(), other.copy()

        # align the indices and combine into a ladder
        A.lower_ind_id = B.lower_ind_id
        B.lower_ind_id = "__tmp{}__"
        A.upper_ind_id = "__tmp{}__"
        both = A | B

        # contract each pair of tensors at each site
        for i in range(A.nsites):
            both ^= A.site_tag(i)

        # convert back to MPO and fuse the double bonds
        out = MatrixProductOperator.from_TN(
            both, upper_ind_id=B.upper_ind_id, lower_ind_id=A.lower_ind_id,
            inplace=True, cyclic=self.cyclic, site_tag_id=A.site_tag_id)

        out.fuse_multibonds(inplace=True)

        # optionally compress
        if compress:
            out.compress(**compress_opts)

        return out

    def apply(self, other, compress=False, **compress_opts):
        r"""Act with this MPO on another MPO or MPS, such that the resulting
        object has the same tensor network structure/indices as ``other``.

        For an MPS::

            other: x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x
                   | | | | | | | | | | | | | | | | | |
             self: A-A-A-A-A-A-A-A-A-A-A-A-A-A-A-A-A-A
                   | | | | | | | | | | | | | | | | | |

                                   -->

              out: y=y=y=y=y=y=y=y=y=y=y=y=y=y=y=y=y=y
                   | | | | | | | | | | | | | | | | | |   <- other.site_ind_id

        For an MPO::

                   | | | | | | | | | | | | | | | | | |
            other: B-B-B-B-B-B-B-B-B-B-B-B-B-B-B-B-B-B
                   | | | | | | | | | | | | | | | | | |
             self: A-A-A-A-A-A-A-A-A-A-A-A-A-A-A-A-A-A
                   | | | | | | | | | | | | | | | | | |

                                   -->

                   | | | | | | | | | | | | | | | | | |   <- other.upper_ind_id
              out: C=C=C=C=C=C=C=C=C=C=C=C=C=C=C=C=C=C
                   | | | | | | | | | | | | | | | | | |   <- other.lower_ind_id

        The resulting TN will have the same structure/indices as ``other``, but
        probably with larger bonds (depending on compression).


        Parameters
        ----------
        other : MatrixProductOperator or MatrixProductState
            The object to act on.
        compress : bool, optional
            Whether to compress the resulting object.
        compress_opts
            Supplied to :meth:`TensorNetwork1DFlat.compress`.

        Returns
        -------
        MatrixProductOperator or MatrixProductState
        """
        if isinstance(other, MatrixProductState):
            return self._apply_mps(other, compress=compress, **compress_opts)
        elif isinstance(other, MatrixProductOperator):
            return self._apply_mpo(other, compress=compress, **compress_opts)
        else:
            raise TypeError("Can only Dot with a MatrixProductOperator or a "
                            "MatrixProductState, got {}".format(type(other)))

    def trace(self, left_inds=None, right_inds=None):
        """Take the trace of this MPO.
        """
        if left_inds is None:
            left_inds = map(self.upper_ind, range(self.nsites))
        if right_inds is None:
            right_inds = map(self.lower_ind, range(self.nsites))
        return super().trace(left_inds, right_inds)

    def partial_transpose(self, sysa, inplace=False):
        """Perform the partial transpose on this MPO by swapping the bra and
        ket indices on sites in ``sysa``.

        Parameters
        ----------
        sysa : sequence of int or int
            The sites to transpose indices on.
        inplace : bool, optional
            Whether to perform the partial transposition inplace.

        Returns
        -------
        MatrixProductOperator
        """
        tn = self if inplace else self.copy()

        if isinstance(sysa, Integral):
            sysa = (sysa,)

        tmp_ind_id = "__tmp_{}__"

        tn.reindex_({tn.upper_ind(i): tmp_ind_id.format(i) for i in sysa})
        tn.reindex_({tn.lower_ind(i): tn.upper_ind(i) for i in sysa})
        tn.reindex_({tmp_ind_id.format(i): tn.lower_ind(i) for i in sysa})
        return tn

    def __add__(self, other):
        """MPO addition.
        """
        return self.add_MPO(other, inplace=False)

    def __iadd__(self, other):
        """In-place MPO addition.
        """
        return self.add_MPO(other, inplace=True)

    def __sub__(self, other):
        """MPO subtraction.
        """
        return self.add_MPO(-1 * other, inplace=False)

    def __isub__(self, other):
        """In-place MPO subtraction.
        """
        return self.add_MPO(-1 * other, inplace=True)

    @property
    def lower_inds(self):
        """An ordered tuple of the actual lower physical indices.
        """
        return tuple(self.lower_ind(i) for i in self.sites)

    @property
    def upper_inds(self):
        """An ordered tuple of the actual upper physical indices.
        """
        return tuple(self.upper_ind(i) for i in self.sites)

    def to_dense(self, *inds_seq):
        if inds_seq:
            lix, rix = inds_seq
        else:
            lix, rix = self.lower_inds, self.upper_inds

        data = self.contract(...).fuse((('lower', lix), ('upper', rix))).data
        d = int(data.size**0.5)
        return qu.qarray(data.reshape(d, d))

    def phys_dim(self, i=None):
        if i is None:
            i = self.sites[0]
        return self[i].ind_size(self.upper_ind(i))

    def rand_state(self, bond_dim, **mps_opts):
        """Get a random vector matching this MPO.
        """
        return qu.tensor.MPS_rand_state(self.nsites, bond_dim, self.phys_dim(),
                                        dtype=self.dtype, cyclic=self.cyclic,
                                        **mps_opts)

    def identity(self, **mpo_opts):
        """Get a identity matching this MPO.
        """
        return qu.tensor.MPO_identity_like(self, **mpo_opts)

    def show(self, max_width=None):
        l1 = ""
        l2 = ""
        l3 = ""
        num_can_l, num_can_r = self.count_canonized()
        for i in range(len(self.sites) - 1):
            bdim = self.bond_size(self.sites[i], self.sites[i + 1])
            strl = len(str(bdim))
            l1 += "|{}".format(bdim)
            l2 += (">" if i < num_can_l else
                   "<" if i >= self.nsites - num_can_r else
                   "O") + ("-" if bdim < 100 else "=") * strl
            l3 += "|" + " " * strl

        l1 += "|"
        l2 += "<" if num_can_r > 0 else "O"
        l3 += "|"

        if self.cyclic:
            bdim = self.bond_size(self.sites[0], self.sites[-1])
            bnd_str = ("-" if bdim < 100 else "=") * strl
            l1 = " {}{}{} ".format(bdim, l1, bdim)
            l2 = "+{}{}{}+".format(bnd_str, l2, bnd_str)
            l3 = " {}{}{} ".format(" " * strl, l3, " " * strl)

        three_line_multi_print(l1, l2, l3, max_width=max_width)


class Dense1D(TensorNetwork1DVector,
              TensorNetwork1D,
              TensorNetwork):
    """Mimics other 1D tensor network structures, but really just keeps the
    full state in a single tensor. This allows e.g. applying gates in the same
    way for quantum circuit simulation as lazily represented hilbert spaces.

    Parameters
    ----------
    array : array_like
        The full hilbert space vector - assumed to be made of equal hilbert
        spaces each of size ``phys_dim`` and will be reshaped as such.
    phys_dim : int, optional
        The hilbert space size of each site, default: 2.
    tags : sequence of str, optional
        Extra tags to add to the tensor network.
    site_ind_id : str, optional
        String formatter describing how to label the site indices.
    site_tag_id : str, optional
        String formatter describing how to label the site tags.
    sites : sequence of int, optional
        The actual sites represented by this tensor network, defaults to
        ``range(nsites)``.
    nsites : int, optional
        The total number of sites, inferred from ``array`` if not given.
    tn_opts
        Supplied to :class:`~quimb.tensor.tensor_core.TensorNetwork`.
    """

    _EXTRA_PROPS = ('_site_ind_id', '_site_tag_id')

    def __init__(self, array, phys_dim=2, tags=None,
                 site_ind_id='k{}', site_tag_id='I{}',
                 sites=None, nsites=None,
                 **tn_opts):

        # copy short-circuit
        if isinstance(array, Dense1D):
            super().__init__(array)
            for ep in Dense1D._EXTRA_PROPS:
                setattr(self, ep, getattr(array, ep))
            return

        # work out number of sites and sub-dimensions etc.
        if nsites is None:
            nsites = qu.infer_size(array, base=phys_dim)
        if sites is None:
            sites = range(nsites)
        dims = [phys_dim] * len(sites)

        # process site indices
        self._site_ind_id = site_ind_id
        site_inds = map(site_ind_id.format, sites)

        # process site tags
        self._site_tag_id = site_tag_id
        site_tags = set(map(site_tag_id.format, sites))

        if tags is not None:
            if isinstance(tags, str):
                tags = {tags}
            else:
                tags = set(tags)
            site_tags = site_tags | tags

        T = Tensor(_asarray(array).reshape(*dims),
                   inds=site_inds, tags=site_tags)

        super().__init__([T], structure=site_tag_id, sites=sites,
                         nsites=nsites, check_collisions=False, **tn_opts)
