"""Classes and algorithms related to 1d tensor networks.
"""
import copy
import numpy as np
from .tensor_core import Tensor, TensorNetwork, rand_uuid


class MatrixProductState(TensorNetwork):
    """Initialise a matrix product state, with auto labelling and tagging.

    Parameters
    ----------
    *arrays : sequence of arrays
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
                 site_ind_id='k{}', site_tag_id='i{}', **kwargs):

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

        super().__init__(gen_tensors(), contract_strategy=site_tag_id,
                         nsites=nsites, check_collisions=False, **kwargs)

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

    def left_canonize_site(self, i, bra=None):
        """Left canonize this MPS' ith site, inplace.

        Parameters
        ----------
        i : int
            Which site to canonize. The site at i + 1 also absorbs the
            non-isometric part of the decomposition of site i.
        bra : None or MatrixProductState, optional
            If given, simultaneously left canonize site i of this MPS, assuming
            it to hold the conjugate state.
        """
        T1 = self.site[i]
        T2 = self.site[i + 1]

        t1_inds_set = set(T1.inds)
        t2_inds_set = set(T2.inds)

        old_shared_bond, = t1_inds_set & t2_inds_set
        left_inds = t1_inds_set - t2_inds_set

        Q, R = T1.split(left_inds, method='qr', get='tensors')
        R = R @ T2

        new_shared_bond, = (j for j in Q.inds if j not in t1_inds_set)
        Q.reindex({new_shared_bond: old_shared_bond}, inplace=True)
        Q.transpose(*T1.inds, inplace=True)
        R.reindex({new_shared_bond: old_shared_bond}, inplace=True)
        R.transpose(*T2.inds, inplace=True)

        self.site[i]._data = Q._data
        self.site[i + 1]._data = R._data

        if bra is not None:
            bra.site[i]._data = Q._data.conj()
            bra.site[i + 1]._data = R._data.conj()

    def right_canonize_site(self, i, bra=None):
        """Right canonize this MPS' ith site, inplace.

        Parameters
        ----------
        i : int
            Which site to canonize. The site at i - 1 also absorbs the
            non-isometric part of the decomposition of site i.
        bra : None or MatrixProductState, optional
            If given, simultaneously right canonize site i of this MPS,
            assuming it to hold the conjugate state.
        """
        T1 = self.site[i]
        T2 = self.site[i - 1]

        t1_inds_set = set(T1.inds)
        t2_inds_set = set(T2.inds)

        left_inds = t1_inds_set & t2_inds_set
        old_shared_bond, = left_inds

        L, Q = T1.split(left_inds, method='lq', get='tensors')
        L = T2 @ L

        new_shared_bond, = (j for j in Q.inds if j not in t1_inds_set)
        L.reindex({new_shared_bond: old_shared_bond}, inplace=True)
        L.transpose(*T2.inds, inplace=True)
        Q.reindex({new_shared_bond: old_shared_bond}, inplace=True)
        Q.transpose(*T1.inds, inplace=True)

        self.site[i - 1]._data = L._data
        self.site[i]._data = Q._data

        if bra is not None:
            bra.site[i - 1]._data = L._data.conj()
            bra.site[i]._data = Q._data.conj()

    def left_canonize(self, start=None, stop=None, normalize=False, bra=None):
        """Left canonize all or a portion of this MPS, such that:

                          i              i
            +-+-+-+-+-+-+-o-o-         +-o-o-
            | | | | | | | | | ...  ->  | | | ...
            +-+-+-+-+-+-+-o-o-         +-o-o-

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
        """Right canonize all or a portion of this MPS, such that:

                   i                           i
                -o-o-+-+-+-+-+-+-+          -o-o-+
             ... | | | | | | | | |   ->  ... | | |
                -o-o-+-+-+-+-+-+-+          -o-o-+


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
        """Mixed canonize this MPS, such that:

                          i                      i
            +-+-+-+-+- -+-o-+- -+-+-+-+-+      +-o-+
            | | | | |...| | |...| | | | |  ->  | | |
            +-+-+-+-+- -+-o-+- -+-+-+-+-+      +-o-+

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
            If supplied, simultaneously move the orthogonality cente this MPS
            too, assuming it to be the conjugate state.
        """
        if new > current:
            for i in range(current, new):
                self.left_canonize_site(i, bra=bra)
        else:
            for i in range(current, new, -1):
                self.right_canonize_site(i, bra=bra)

    def schmidt_values(self, i, current_orthog_centre=None, method='svd'):
        """Find the schmidt values associated with the bipartition of this
        MPS between sites on either site of ``i``. In other words, ``i`` is the
        number of sites in the left hand partition:

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
        if not (0 < i < self.nsites):
            raise ValueError("Need 0 < i < {}, got i={}."
                             .format(self.nsites, i))

        if current_orthog_centre is None:
            self.canonize(i)
        else:
            self.shift_orthogonality_center(current_orthog_centre, i)

        Tm1 = self.site[i]
        left_inds = set(Tm1.inds) & set(self.site[i - 1].inds)
        S = Tm1.singular_values(left_inds, method=method)**2
        return S

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


class MatrixProductOperator(TensorNetwork):
    """Initialise a matrix product operator, with auto labelling and tagging.

    Parameters
    ----------
    *arrays : sequence of arrays
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

    def __init__(self, arrays, shape='lrud', site_tag_id='i{}', tags=None,
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

        super().__init__(gen_tensors(), contract_strategy=site_tag_id,
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
        self.reindex_sites(new_id, inplace=True)
        self._lower_ind_id = new_id

    lower_ind_id = property(_get_lower_ind_id, _set_lower_ind_id,
                            doc="The string specifier for the lower phyiscal "
                            "indices")

    def _get_upper_ind_id(self):
        return self._upper_ind_id

    def _set_upper_ind_id(self, new_id):
        self.reindex_sites(new_id, inplace=True)
        self._upper_ind_id = new_id

    upper_ind_id = property(_get_upper_ind_id, _set_upper_ind_id,
                            doc="The string specifier for the upper phyiscal "
                            "indices")

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


def align_inner(mps_ket, mps_bra, mpo=None):
    """Align two MPS, with or without a sandwiched MPO, so that they form an
    overlap/expectation tensor network.

    Parameters
    ----------
    mps_ket : MatrixProductState
        A state.
    mps_bra : MatrixProductState
        Another state, notionally the 'bra'.
    mpo : None or MatrixProductOperator
        If given, sandwich this operator between the two MPS.
    """
    if mpo is None:
        if mps_ket.site_ind_id != mps_bra.site_ind_id:
            mps_bra.site_ind_id = mps_ket.site_ind_id
            return

    if mps_ket.site_ind_id != mpo.upper_ind_id:
        mps_ket.site_ind_id = mpo.upper_ind_id

    if mps_bra.site_ind_id != mpo.lower_ind_id:
        mps_bra.site_ind_id = mpo.lower_ind_id
