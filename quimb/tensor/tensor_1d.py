"""Classes and algorithms related to 1d tensor networks.
"""
import functools
import copy
import numpy as np
from ..utils import three_line_multi_print, pairwise
from .tensor_core import (
    Tensor,
    TensorNetwork,
    rand_uuid,
)
try:
    from opt_einsum import parser
except ImportError:
    pass


def align_TN_1D(*tns, ind_ids=None, inplace=False):
    """Align an arbitrary number of 1D tensor networks in a stack-like
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
    tns : sequence of MatrixProductState and MatrixProductOperator
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
                   ["__ind_{}".format(parser.einsum_symbols[i]) + "{}__"
                    for i in range(len(tns) - 2)])
    else:
        ind_ids = tuple(ind_ids)

    for i, tn in enumerate(tns):
        if isinstance(tn, MatrixProductState):
            if i == 0:
                tn.site_ind_id = ind_ids[i]
            elif i == len(tns) - 1:
                tn.site_ind_id = ind_ids[i - 1]
            else:
                raise ValueError("An MPS can only be aligned as the first or "
                                 "last TN in a sequence.")

        elif isinstance(tn, MatrixProductOperator):
            tn.upper_ind_id = ind_ids[i - 1]
            tn.lower_ind_id = ind_ids[i]

        else:
            raise ValueError("Can only align MPS and MPOs currently.")

    return tns


class TensorNetwork1D(TensorNetwork):
    """Base class for tensor networks with a one-dimensional structure.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def site_tag_id(self):
        """The string specifier for tagging each site of this 1D TN.
        """
        return self._site_tag_id

    def site_tag(self, i):
        """The name of the tag specifiying the tensor at site ``i``.
        """
        return self.site_tag_id.format(i)

    @property
    def site_tags(self):
        """An ordered tuple of the actual site tags.
        """
        return tuple(self.site_tag(i) for i in self.sites)

    def _left_decomp_site(self, i, bra=None, **split_opts):
        T1 = self.site[i]
        T2 = self.site[i + 1]

        t1_inds_set = set(T1.inds)
        t2_inds_set = set(T2.inds)

        old_shared_bond, = t1_inds_set & t2_inds_set
        left_inds = t1_inds_set - t2_inds_set

        Q, R = T1.split(left_inds, get='tensors', **split_opts)
        R = R @ T2

        new_shared_bond, = (j for j in Q.inds if j not in t1_inds_set)
        Q.reindex({new_shared_bond: old_shared_bond}, inplace=True)
        Q.transpose(*T1.inds, inplace=True)
        R.reindex({new_shared_bond: old_shared_bond}, inplace=True)
        R.transpose(*T2.inds, inplace=True)

        self.site[i].modify(data=Q.data)
        self.site[i + 1].modify(data=R.data)

        if bra is not None:
            bra.site[i].modify(data=Q.data.conj())
            bra.site[i + 1].modify(data=R.data.conj())

    def _right_decomp_site(self, i, bra=None, **split_opts):
        T1 = self.site[i]
        T2 = self.site[i - 1]

        t1_inds_set = set(T1.inds)
        t2_inds_set = set(T2.inds)

        left_inds = t1_inds_set & t2_inds_set
        old_shared_bond, = left_inds

        L, Q = T1.split(left_inds, get='tensors', **split_opts)
        L = T2 @ L

        new_shared_bond, = (j for j in Q.inds if j not in t1_inds_set)
        L.reindex({new_shared_bond: old_shared_bond}, inplace=True)
        L.transpose(*T2.inds, inplace=True)
        Q.reindex({new_shared_bond: old_shared_bond}, inplace=True)
        Q.transpose(*T1.inds, inplace=True)

        self.site[i - 1].modify(data=L.data)
        self.site[i].modify(data=Q.data)

        if bra is not None:
            bra.site[i - 1].modify(data=L.data.conj())
            bra.site[i].modify(data=Q.data.conj())

    def left_canonize_site(self, i, bra=None):
        """Left canonize this TN's ith site, inplace::

                i                i
               -o-o-            ->-s-
            ... | | ...  --> ... | | ...

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
        """Right canonize this TN's ith site, inplace::

                  i                i
               -o-o-            -s-<-
            ... | | ...  --> ... | | ...

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
        """Left canonize all or a portion of this TN. If this is a MPS,
        this implies that::

                          i              i
            >->->->->->->-o-o-         +-o-o-
            | | | | | | | | | ...  ->  | | | ...
            >->->->->->->-o-o-         +-o-o-

        Parameters
        ----------
        start : int, optional
            If given, the site to start left canonizing at.
        stop : int, optional
            If given, the site to stop left canonizing at.
        normalize : bool, optional
            Whether to normalize the state.
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
            factor = self.site[-1].norm()
            self.site[-1] /= factor
            if bra is not None:
                bra.site[-1] /= factor

    def right_canonize(self, stop=None, start=None, normalize=False, bra=None):
        """Right canonize all or a portion of this TN. If this is a MPS,
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
            factor = self.site[0].norm()
            self.site[0] /= factor
            if bra is not None:
                bra.site[0] /= factor

    def canonize(self, orthogonality_center, bra=None):
        """Mixed canonize this TN. If this is a MPS, this implies that:

                          i                      i
            >->->->->- ->-o-<- -<-<-<-<-<      +-o-+
            | | | | |...| | |...| | | | |  ->  | | |
            >->->->->- ->-o-<- -<-<-<-<-<      +-o-+

        Parameters
        ----------
        orthogonality_center : int, optional
            Which site to orthogonalize around.
        bra : MatrixProductState, optional
            If supplied, simultaneously mixed canonize this MPS too, assuming
            it to be the conjugate state.
        """
        self.left_canonize(stop=orthogonality_center, bra=bra)
        self.right_canonize(stop=orthogonality_center, bra=bra)

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

        elif isinstance(form, int):
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
        bond, = self.site[i].shared_inds(self.site[j])
        return bond

    def bond_dim(self, i, j):
        """Return the size of the bond between site ``i`` and ``j``.
        """
        b_ix = self.bond(i, j)
        return self.site[i].ind_size(b_ix)

    def fuse_multibonds(self, inplace=False):
        """Fuse any double/triple etc bonds between neighbours
        """
        tn = self if inplace else self.copy()

        for i, j in pairwise(tn.sites):
            T1, T2 = tn.site[i], tn.site[j]
            dbnds = T1.shared_inds(T2)
            T1.fuse({dbnds[0]: dbnds}, inplace=True)
            T2.fuse({dbnds[0]: dbnds}, inplace=True)

        return tn

    def singular_values(self, i, current_orthog_centre=None, method='svd'):
        """Find the singular values associated with the ith bond::

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
        current_orthog_centre : int
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

        if current_orthog_centre is None:
            self.canonize(i)
        else:
            self.shift_orthogonality_center(current_orthog_centre, i)

        Tm1 = self.site[i]
        left_inds = Tm1.shared_inds(self.site[i - 1])
        return Tm1.singular_values(left_inds, method=method)

    def expand_bond_dimension(self, new_bond_dim, inplace=True, bra=None):
        """Expand the bond dimensions of this 1D tensor network to at least
        ``new_bond_dim``.
        """
        if inplace:
            expanded = self
        else:
            expanded = self.copy()

        for i in self.sites:
            tensor = expanded.site[i]
            to_expand = []

            if i > 0:
                to_expand.append(self.bond(i - 1, i))
            if i < self.nsites - 1:
                to_expand.append(self.bond(i, i + 1))

            pads = [(0, 0) if i not in to_expand else
                    (0, max(new_bond_dim - d, 0))
                    for d, i in zip(tensor.shape, tensor.inds)]

            tensor.modify(data=np.pad(tensor.data, pads, mode='constant'))

            if bra is not None:
                bra.site[i].modify(data=tensor.data.conj())

        return expanded

    def count_canonized(self, **allclose_opts):
        ov = self.H & self
        num_can_l = 0
        num_can_r = 0

        for i in range(self.nsites - 1):
            ov ^= slice(0, i + 1)
            x = ov.site[i].data
            if np.allclose(x, np.eye(x.shape[0]), **allclose_opts):
                num_can_l += 1
            else:
                break

        for j in reversed(range(i + 1, self.nsites)):
            ov ^= slice(j, ...)
            x = ov.site[j].data
            if np.allclose(x, np.eye(x.shape[0]), **allclose_opts):
                num_can_r += 1
            else:
                break

        return num_can_l, num_can_r

    def show(self, max_width=None):
        l1 = ""
        l2 = ""
        l3 = ""
        num_can_l, num_can_r = self.count_canonized()
        for i in range(self.nsites - 1):
            bdim = self.bond_dim(i, i + 1)
            strl = len(str(bdim))
            l1 += " {}".format(bdim)
            l2 += (">" if i < num_can_l else
                   "<" if i >= self.nsites - num_can_r else
                   "o") + ("-" if bdim < 100 else "=") * strl
            l3 += "|" + " " * strl

        l2 += "<" if num_can_r > 0 else "o"
        l3 += "|"

        three_line_multi_print(l1, l2, l3, max_width=max_width)


class MatrixProductState(TensorNetwork1D):
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
    tags : str or sequence of hashable, optional
        Global tags to attach to all tensors.
    bond_name : str, optional
        The base name of the bond indices, onto which uuids will be added.
    """

    def __init__(self, arrays, *, shape='lrp', tags=None, bond_name="",
                 site_ind_id='k{}', site_tag_id='I{}', sites=None, nsites=None,
                 **tn_opts):

        # short-circuit for copying MPSs
        if isinstance(arrays, MatrixProductState):
            super().__init__(arrays)
            self._site_ind_id = copy.copy(arrays.site_ind_id)
            self._site_tag_id = copy.copy(arrays.site_tag_id)
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

        cyclic = (arrays[0].ndim == 3)

        # transpose arrays to 'lrp' order.
        def gen_orders():
            lp_ord = tuple(shape.replace('r', "").find(x) for x in 'lp')
            lrp_ord = tuple(shape.find(x) for x in 'lrp')
            rp_ord = tuple(shape.replace('l', "").find(x) for x in 'rp')
            yield lp_ord if not cyclic else lrp_ord
            for _ in range(len(sites) - 2):
                yield lrp_ord
            yield rp_ord if not cyclic else lrp_ord

        def gen_inds():
            cyc_bond = (rand_uuid(base=bond_name),) if cyclic else ()

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

    _EXTRA_PROPS = ('_site_ind_id', '_site_tag_id')

    def imprint(self, other):
        """Cast ``other'' into a ``MatrixProductState'' like ``self''.
        """
        for p in MatrixProductState._EXTRA_PROPS:
            setattr(other, p, getattr(self, p))
        other.__class__ = MatrixProductState

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
            indices = range(0, self.nsites)
        elif isinstance(where, slice):
            start = 0 if where.start is None else where.start
            stop = self.nsites if where.stop is ... else where.stop
            indices = range(start, stop)
        else:
            indices = where

        return self.reindex({self.site_ind(i): new_id.format(i)
                             for i in indices}, inplace=inplace)

    def _get_site_ind_id(self):
        return self._site_ind_id

    def _set_site_ind_id(self, new_id):
        self.reindex_sites(new_id, inplace=True)
        self._site_ind_id = new_id

    site_ind_id = property(_get_site_ind_id, _set_site_ind_id,
                           doc="The string specifier for the physical indices")

    def site_ind(self, i):
        return self.site_ind_id.format(i)

    @property
    def site_inds(self):
        """An ordered tuple of the actual physical indices.
        """
        return tuple(self.site_ind(i) for i in self.sites)

    def add_MPS(self, other, inplace=False, compress=False, **compress_opts):
        """Add another MatrixProductState to this one.
        """
        if self.nsites != other.nsites:
            raise ValueError("Can't add MPS with another of different length.")

        summed = self if inplace else self.copy()

        for i in summed.sites:
            summed_tensor = summed.site[i]
            other_tensor = other.site[i]

            if set(summed_tensor.inds) != set(other_tensor.inds):
                # Need to use bonds to match indices
                reindex_map = {}
                if i > 0:
                    reindex_map[other.bond(i - 1, i)] = summed.bond(i - 1, i)
                if i < summed.nsites - 1:
                    reindex_map[other.bond(i, i + 1)] = summed.bond(i, i + 1)
                other_tensor = other_tensor.reindex(reindex_map)

            summed_tensor.direct_product(other_tensor, inplace=True,
                                         sum_inds=summed.site_ind(i))

        if compress:
            summed.compress(**compress_opts)

        return summed

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

    def schmidt_values(self, i, current_orthog_centre=None, method='svd'):
        """Find the schmidt values associated with the bipartition of this
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
        current_orthog_centre : int
            If given, the known current orthogonality center, to speed up the
            mixed canonization.

        Returns
        -------
        S : 1d-array
            The schmidt values.
        """
        return self.singular_values(i, current_orthog_centre, method=method)**2

    def entropy(self, i, current_orthog_centre=None, method='svd'):
        """The entropy of bipartition between the left block of ``i`` sites and
        the rest.

        Parameters
        ----------
        i : int
            The number of sites in the left partition.
        current_orthog_centre : int
            If given, the known current orthogonality center, to speed up the
            mixed canonization.

        Returns
        -------
        float
        """
        S = self.schmidt_values(i, current_orthog_centre=current_orthog_centre,
                                method=method)
        S = S[S > 0.0]
        return np.sum(-S * np.log2(S))

    def schmidt_gap(self, i, current_orthog_centre=None, method='svd'):
        """The schmidt gap of bipartition between the left block of ``i`` sites
        and the rest.

        Parameters
        ----------
        i : int
            The number of sites in the left partition.
        current_orthog_centre : int
            If given, the known current orthogonality center, to speed up the
            mixed canonization.

        Returns
        -------
        float
        """
        S = self.schmidt_values(i, current_orthog_centre=current_orthog_centre,
                                method=method)
        return S[0] - S[1]

    def partial_trace(self, keep, upper_ind_id="b{}", rescale_sites=True):
        """Partially trace this matrix product state, producing a matrix
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
            keep = range(*self.parse_tag_slice(keep))

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
        rho = view_TN_as_MPO(rho, lower_ind_id=upper_ind_id,
                             upper_ind_id=self.site_ind_id,
                             site_tag_id=self.site_tag_id, inplace=True)

        if rescale_sites:
            # e.g. [3, 4, 5, 7, 9] -> [0, 1, 2, 3, 4]
            tag_map, ind_map = {}, {}
            for new, old in enumerate(keep):
                tag_map[self.site_tag(old)] = self.site_tag(new)
                ind_map[rho.lower_ind(old)] = rho.lower_ind(new)
                ind_map[rho.upper_ind(old)] = rho.upper_ind(new)

            rho.retag(tag_map, inplace=True)
            rho.reindex(ind_map, inplace=True)

            rho.nsites = n
            rho.sites = range(n)
        else:
            rho.sites = keep

        rho.fuse_multibonds(inplace=True)
        return rho

    @functools.wraps(partial_trace)
    def ptr(self, keep, upper_ind_id="b{}", rescale_sites=True):
        return self.partial_trace(keep, upper_ind_id,
                                  rescale_sites=rescale_sites)

    def to_dense(self):
        """Return the dense ket version of this MPS, i.e. a ``numpy.matrix``
        with shape (-1, 1).
        """
        return np.asmatrix(self.contract(...)
                           .fuse({'all': self.site_inds})
                           .data.reshape(-1, 1))

    def phys_dim(self, i=None):
        if i is None:
            i = self.sites[0]
        return self.site[i].ind_size(self.site_ind(i))

    @functools.wraps(align_TN_1D)
    def align(self, *args, inplace=True):
        return align_TN_1D(self, *args, inplace=inplace)


class MatrixProductOperator(TensorNetwork1D):
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
    tags : str or sequence of hashable, optional
        Global tags to attach to all tensors.
    bond_name : str, optional
        The base name of the bond indices, onto which uuids will be added.
    """

    def __init__(self, arrays, shape='lrud', site_tag_id='I{}', tags=None,
                 upper_ind_id='k{}', lower_ind_id='b{}', bond_name="",
                 sites=None, nsites=None, **tn_opts):
        # short-circuit for copying
        if isinstance(arrays, MatrixProductOperator):
            super().__init__(arrays)
            self._upper_ind_id = copy.copy(arrays.upper_ind_id)
            self._lower_ind_id = copy.copy(arrays.lower_ind_id)
            self._site_tag_id = copy.copy(arrays.site_tag_id)
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

        # transpose arrays to 'lrud' order.
        def gen_orders():
            lud_ord = tuple(shape.replace('r', "").find(x) for x in 'lud')
            rud_ord = tuple(shape.replace('l', "").find(x) for x in 'rud')
            lrud_ord = tuple(map(shape.find, 'lrud'))
            yield lud_ord
            for _ in range(len(sites) - 2):
                yield lrud_ord
            yield rud_ord

        def gen_inds():
            nbond = rand_uuid(base=bond_name)
            yield (nbond, next(upper_inds), next(lower_inds))
            pbond = nbond
            for _ in range(len(sites) - 2):
                nbond = rand_uuid(base=bond_name)
                yield (pbond, nbond, next(upper_inds), next(lower_inds))
                pbond = nbond
            yield (pbond, next(upper_inds), next(lower_inds))

        def gen_tensors():
            for array, site_tag, inds, order in zip(arrays, site_tags,
                                                    gen_inds(), gen_orders()):

                yield Tensor(array.transpose(*order), inds=inds, tags=site_tag)

        super().__init__(gen_tensors(), structure=site_tag_id, sites=sites,
                         nsites=nsites, check_collisions=False, **tn_opts)

    _EXTRA_PROPS = ('_upper_ind_id', '_lower_ind_id', '_site_tag_id')

    def imprint(self, other):
        """Cast ``other'' into a ``MatrixProductOperator'' like ``self''.
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
        if self.nsites != other.nsites:
            raise ValueError("Can't add MPO with another of different length.")

        if inplace:
            summed = self
        else:
            summed = self.copy()

        for i in summed.sites:
            summed_tensor = summed.site[i]
            other_tensor = other.site[i]

            if set(summed_tensor.inds) != set(other_tensor.inds):
                # Need to use bonds to match indices
                reindex_map = {}
                if i > 0:
                    reindex_map[other.bond(i - 1, i)] = summed.bond(i - 1, i)
                if i < summed.nsites - 1:
                    reindex_map[other.bond(i, i + 1)] = summed.bond(i, i + 1)
                other_tensor = other_tensor.reindex(reindex_map)

            summed_tensor.direct_product(other_tensor, inplace=True,
                                         sum_inds=(summed.upper_ind(i),
                                                   summed.lower_ind(i)))

        if compress:
            summed.compress(**compress_opts)

        return summed

    def _apply_mpo(self, other, compress=False, **compress_opts):
        """This MPO acting on another MPO ladder style.
        """
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
        out = view_TN_as_MPO(both, inplace=True, site_tag_id=A.site_tag_id,
                             upper_ind_id=B.upper_ind_id,
                             lower_ind_id=A.lower_ind_id)

        out.fuse_multibonds(inplace=True)

        # optionally compress
        if compress:
            out.compress(**compress_opts)

        return out

    def apply(self, other, compress=False, **compress_opts):
        """Act with this MPO on another MPO or MPS, such that the resulting
        object has the same tensor network structure/indices as the input.

        Parameters
        ----------
        other : MatrixProductOperator or MatrixProductState
            The object to act on.
        compress : bool, optional
            Whether to compress the resulting object.
        compress_opts
            Supplied to :meth:`TensorNetwork1D.compress`.
        """
        if isinstance(other, MatrixProductOperator):
            return self._apply_mpo(other, compress=compress, **compress_opts)
        else:
            raise TypeError("Can only Dot with a MatrixProductOperator or a "
                            "MatrixProductState, got {}".format(type(other)))

    def trace(self):
        traced = self.copy()
        traced.upper_ind_id = traced.lower_ind_id
        return traced ^ ...

    def partial_transpose(self, sysa, inplace=False):
        """Perform the partial tranpose on this MPO by swapping the bra and ket
        indices on sites in ``sysa``.

        Parameters
        ----------
        sysa : sequence of int or int
            The sites to tranpose indices on.
        inplace : bool, optional
            Whether to perform the partial transposition inplace.

        Returns
        -------
        MatrixProductOperator
        """
        tn = self if inplace else self.copy()

        if isinstance(sysa, int):
            sysa = (sysa,)

        tmp_ind_id = "__tmp_{}__"

        tn.reindex({tn.upper_ind(i): tmp_ind_id.format(i)
                    for i in sysa}, inplace=True)
        tn.reindex({tn.lower_ind(i): tn.upper_ind(i)
                    for i in sysa}, inplace=True)
        tn.reindex({tmp_ind_id.format(i): tn.lower_ind(i)
                    for i in sysa}, inplace=True)
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

    def to_dense(self):
        data = self.contract(...).fuse((('lower', self.lower_inds),
                                        ('upper', self.upper_inds))).data
        d = int(data.size**0.5)
        return np.matrix(data.reshape(d, d))

    def phys_dim(self, i=None):
        if i is None:
            i = self.sites[0]
        return self.site[i].ind_size(self.upper_ind(i))

    def show(self, max_width=None):
        l1 = ""
        l2 = ""
        l3 = ""
        num_can_l, num_can_r = self.count_canonized()
        for i in range(len(self.sites) - 1):
            bdim = self.bond_dim(self.sites[i], self.sites[i + 1])
            strl = len(str(bdim))
            l1 += "|{}".format(bdim)
            l2 += (">" if i < num_can_l else
                   "<" if i >= self.nsites - num_can_r else
                   "O") + ("-" if bdim < 100 else "=") * strl
            l3 += "|" + " " * strl

        l1 += "|"
        l2 += "<" if num_can_r > 0 else "O"
        l3 += "|"

        three_line_multi_print(l1, l2, l3, max_width=max_width)


def view_TN_as_MPO(tn, upper_ind_id, lower_ind_id, site_tag_id, inplace=False):
    """Convert a TensorNetwork into a MatrixProductOperator, assuming it has
    the appropirate underlying structure.

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
    tn.__class__ = MatrixProductOperator
    tn._upper_ind_id = upper_ind_id
    tn._lower_ind_id = lower_ind_id
    tn._site_tag_id = site_tag_id
    return tn
