"""Classes and algorithms related to 1d tensor networks.
"""
import functools
import copy
import numpy as np
from ..utils import three_line_multi_print
from .tensor_core import (
    _einsum_symbols,
    Tensor,
    TensorNetwork,
    rand_uuid,
    tensor_direct_product,
)


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
                   ["__ind_{}".format(_einsum_symbols[i]) + "{}__"
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
    """
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

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

        self.site[i].update(data=Q._data)
        self.site[i + 1].update(data=R._data)

        if bra is not None:
            bra.site[i].update(data=Q._data.conj())
            bra.site[i + 1].update(data=R._data.conj())

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

        self.site[i - 1].update(data=L._data)
        self.site[i].update(data=Q._data)

        if bra is not None:
            bra.site[i - 1].update(data=L._data.conj())
            bra.site[i].update(data=Q._data.conj())

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

    def left_canonize(self, start=None, stop=None, normalize=False, bra=None):
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

    def right_canonize(self, start=None, stop=None, normalize=False, bra=None):
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

    def left_compress(self, start=None, stop=None, bra=None,
                      current_orthog_centre=None, **compress_opts):
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
        current_orthog_centre : int, optional
            The current orthogonality center, if known, to speed things up.
        compress_opts
            Supplied to :meth:`Tensor.split`.
        """
        if start is None:
            start = 0
        if stop is None:
            stop = self.nsites - 1

        for i in range(start, stop):
            self.left_compress_site(i, bra=bra, **compress_opts)

    def right_compress(self, start=None, stop=None, bra=None,
                       current_orthog_centre=None, **compress_opts):
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
        current_orthog_centre : int, optional
            The current orthogonality center, if known, to speed things up.
        compress_opts
            Supplied to :meth:`Tensor.split`.
        """
        if start is None:
            start = self.nsites - 1
        if stop is None:
            stop = 0

        for i in range(start, stop, -1):
            self.right_compress_site(i, bra=bra, **compress_opts)

    def compress(self, form='flat', **compress_opts):
        """Compress this 1D Tensor Network, possibly into canonical form.

        Parameters
        ----------
        form : {'flat', 'left', 'right'} or int
            Output form of the TN. ``'flat'`` tries to distribute the singular
            values evenly, but state willl not be canonical (default).
            ``'left'`` and ``'right'`` put the state into left and right
            canonical form respectively, or an int will put the state into
            mixed canonical form at that site.
        compress_opts
            Supplied to :meth:`Tensor.split`.
        """
        if isinstance(form, int):
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
        inds_i = self.site[i].inds
        inds_j = self.site[j].inds
        bond, = (i for i in inds_i if i in inds_j)
        return bond

    def bond_dim(self, i, j):
        """Return the size of the bond between site ``i`` and ``j``.
        """
        b_ix = self.bond(i, j)
        return self.site[i].ind_size(b_ix)

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
        left_inds = set(Tm1.inds) & set(self.site[i - 1].inds)
        return Tm1.singular_values(left_inds, method=method)

    def expand_bond_dimension(self, new_bond_dim, inplace=True, bra=None):
        """Expand the bond dimensions of this 1D tensor network to at least
        ``new_bond_dim``.
        """
        if inplace:
            expanded = self
        else:
            expanded = self.copy()

        for i in range(self.nsites):
            tensor = expanded.site[i]
            to_expand = []

            if i > 0:
                to_expand.append(self.bond(i - 1, i))
            if i < self.nsites - 1:
                to_expand.append(self.bond(i, i + 1))

            pads = [(0, 0) if i not in to_expand else
                    (0, max(new_bond_dim - d, 0))
                    for d, i in zip(tensor.shape, tensor.inds)]

            tensor.update(data=np.pad(tensor._data, pads, mode='constant'))

            if bra is not None:
                bra.site[i].update(data=tensor.data.conj())

        return expanded

    def count_canonized(self, **allclose_opts):
        ov = self.H & self
        num_can_l = 0
        num_can_r = 0

        # import pdb; pdb.set_trace()

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
                 site_ind_id='k{}', site_tag_id='I{}', **tn_opts):

        # short-circuit for copying MPSs
        if isinstance(arrays, MatrixProductState):
            super().__init__(arrays)
            self._site_ind_id = copy.copy(arrays._site_ind_id)
            self._site_tag_id = copy.copy(arrays._site_tag_id)
            return

        arrays = tuple(arrays)
        nsites = len(arrays)

        # process site indices
        self._site_ind_id = site_ind_id
        site_inds = map(site_ind_id.format, range(nsites))

        # process site tags
        self._site_tag_id = site_tag_id
        site_tags = map(site_tag_id.format, range(nsites))

        if tags is not None:
            if isinstance(tags, str):
                tags = {tags}
            else:
                tags = set(tags)

            site_tags = tuple({st} | tags for st in site_tags)

        # TODO: figure out cyclic or not
        # TODO: allow open ends non-cyclic

        # transpose arrays to 'lrp' order.
        def gen_orders():
            lp_ord = tuple(shape.replace('r', "").find(x) for x in 'lp')
            lrp_ord = tuple(shape.find(x) for x in 'lrp')
            rp_ord = tuple(shape.replace('l', "").find(x) for x in 'rp')
            yield lp_ord
            for _ in range(nsites - 2):
                yield lrp_ord
            yield rp_ord

        def gen_inds():
            nbond = rand_uuid(base=bond_name)
            yield (nbond, next(site_inds))
            pbond = nbond
            for _ in range(nsites - 2):
                nbond = rand_uuid(base=bond_name)
                yield (pbond, nbond, next(site_inds))
                pbond = nbond
            yield (pbond, next(site_inds))

        def gen_tensors():
            for array, site_tag, inds, order in zip(arrays, site_tags,
                                                    gen_inds(), gen_orders()):
                yield Tensor(array.transpose(*order), inds=inds, tags=site_tag)

        super().__init__(gen_tensors(), structure=site_tag_id,
                         nsites=nsites, check_collisions=False, **tn_opts)

    def reindex_sites(self, new_id, where=None, inplace=False):
        """Update the physical site index labels to a new string specifier.

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

        return self.reindex({self.site_ind_id.format(i): new_id.format(i)
                             for i in range(start, stop)}, inplace=inplace)

    def _get_site_ind_id(self):
        return self._site_ind_id

    def _set_site_ind_id(self, new_id):
        self.reindex_sites(new_id, inplace=True)
        self._site_ind_id = new_id

    site_ind_id = property(_get_site_ind_id, _set_site_ind_id,
                           doc="The string specifier for the physical indices")

    @property
    def site_inds(self):
        """An ordered tuple of the actual physical indices.
        """
        return tuple(self.site_ind_id.format(i) for i in range(self.nsites))

    @property
    def site_tag_id(self):
        """The string specifier for tagging each site of this MPS.
        """
        return self._site_tag_id

    @property
    def site_tags(self):
        """An ordered tuple of the actual site tags.
        """
        return tuple(self.site_tag_id.format(i) for i in range(self.nsites))

    def add_MPS(self, other, inplace=False, compress=False, **compress_opts):
        """Add another MatrixProductState to this one.
        """
        if self.nsites != other.nsites:
            raise ValueError("Can't add MPS with another of different length.")

        if inplace:
            summed = self
        else:
            summed = self.copy()

        for i in range(summed.nsites):
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

            tensor_direct_product(summed_tensor, other_tensor, inplace=True,
                                  sum_inds=summed.site_ind_id.format(i))

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

    def to_dense(self):
        """Return the dense ket version of this MPS, i.e. a ``numpy.matrix``
        with shape (-1, 1).
        """
        return np.asmatrix(self.contract(...)
                           .fuse({'all': self.site_inds})
                           .data.reshape(-1, 1))

    def phys_dim(self, i=0):
        return self.site[i].ind_size(self.site_ind_id.format(i))

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
                 **kwargs):
        # short-circuit for copying
        if isinstance(arrays, MatrixProductOperator):
            super().__init__(arrays)
            self._upper_ind_id = copy.copy(arrays._upper_ind_id)
            self._lower_ind_id = copy.copy(arrays._lower_ind_id)
            self._site_tag_id = copy.copy(arrays._site_tag_id)
            return

        arrays = tuple(arrays)
        nsites = len(arrays)

        # process site indices
        self._upper_ind_id = upper_ind_id
        self._lower_ind_id = lower_ind_id
        upper_inds = map(upper_ind_id.format, range(nsites))
        lower_inds = map(lower_ind_id.format, range(nsites))

        # process site tags
        self._site_tag_id = site_tag_id
        site_tags = map(site_tag_id.format, range(nsites))
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
            for _ in range(nsites - 2):
                yield lrud_ord
            yield rud_ord

        def gen_inds():
            nbond = rand_uuid(base=bond_name)
            yield (nbond, next(upper_inds), next(lower_inds))
            pbond = nbond
            for _ in range(nsites - 2):
                nbond = rand_uuid(base=bond_name)
                yield (pbond, nbond, next(upper_inds), next(lower_inds))
                pbond = nbond
            yield (pbond, next(upper_inds), next(lower_inds))

        def gen_tensors():
            for array, site_tag, inds, order in zip(arrays, site_tags,
                                                    gen_inds(), gen_orders()):

                yield Tensor(array.transpose(*order), inds=inds, tags=site_tag)

        super().__init__(gen_tensors(), structure=site_tag_id,
                         nsites=nsites, check_collisions=False, **kwargs)

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

        return self.reindex({self.lower_ind_id.format(i): new_id.format(i)
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

        return self.reindex({self.upper_ind_id.format(i): new_id.format(i)
                             for i in range(start, stop)}, inplace=inplace)

    def _get_lower_ind_id(self):
        return self._lower_ind_id

    def _set_lower_ind_id(self, new_id):
        self.reindex_lower_sites(new_id, inplace=True)
        self._lower_ind_id = new_id

    lower_ind_id = property(_get_lower_ind_id, _set_lower_ind_id,
                            doc="The string specifier for the lower phyiscal "
                            "indices")

    def _get_upper_ind_id(self):
        return self._upper_ind_id

    def _set_upper_ind_id(self, new_id):
        self.reindex_upper_sites(new_id, inplace=True)
        self._upper_ind_id = new_id

    upper_ind_id = property(_get_upper_ind_id, _set_upper_ind_id,
                            doc="The string specifier for the upper phyiscal "
                            "indices")

    def add_MPO(self, other, inplace=False, compress=False, **compress_opts):
        """Add another MatrixProductState to this one.
        """
        if self.nsites != other.nsites:
            raise ValueError("Can't add MPO with another of different length.")

        if inplace:
            summed = self
        else:
            summed = self.copy()

        for i in range(summed.nsites):
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

            tensor_direct_product(summed_tensor, other_tensor, inplace=True,
                                  sum_inds=(summed.upper_ind_id.format(i),
                                            summed.lower_ind_id.format(i)))

        if compress:
            summed.compress(**compress_opts)

        return summed

    def trace(self):
        traced = self.copy()
        traced.upper_ind_id = traced.lower_ind_id
        return traced ^ ...

    def __add__(self, other):
        """MPO addition.
        """
        return self.add_MPO(other, inplace=False)

    def __iadd__(self, other):
        """In-place MPO addition.
        """
        return self.add_MPO(other, inplace=True)

    @property
    def lower_inds(self):
        """An ordered tuple of the actual lower physical indices.
        """
        return tuple(self.lower_ind_id.format(i) for i in range(self.nsites))

    @property
    def upper_inds(self):
        """An ordered tuple of the actual upper physical indices.
        """
        return tuple(self.upper_ind_id.format(i) for i in range(self.nsites))

    @property
    def site_tag_id(self):
        """The string specifier for tagging each site of this MPO.
        """
        return self._site_tag_id

    @property
    def site_tags(self):
        """An ordered tuple of the actual site tags.
        """
        return tuple(self.site_tag_id.format(i) for i in range(self.nsites))

    def to_dense(self):
        data = self.contract(...).fuse((('lower', self.lower_inds),
                                        ('upper', self.upper_inds))).data
        d = int(data.size**0.5)
        return np.matrix(data.reshape(d, d))

    def phys_dim(self, i=0):
        return self.site[i].ind_size(self.upper_ind_id.format(i))

    def show(self, max_width=None):
        l1 = ""
        l2 = ""
        l3 = ""
        for i in range(self.nsites - 1):
            bdim = self.bond_dim(i, i + 1)
            strl = len(str(bdim))
            l1 += "|{}".format(bdim)
            l2 += "O" + ("-" if bdim < 100 else "=") * strl
            l3 += "|" + " " * strl

        l1 += "|"
        l2 += "O"
        l3 += "|"

        three_line_multi_print(l1, l2, l3, max_width=max_width)
