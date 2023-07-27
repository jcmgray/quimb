"""Classes and algorithms related to arbitrary geometry tensor networks.
"""

import functools
from operator import add

from autoray import do, dag

from ..utils import check_opt, deprecated, ensure_dict
from ..utils import progbar as Progbar
from .tensor_core import (
    TensorNetwork,
    get_symbol,
    rand_uuid,
    oset,
    tags_to_oset,
)


def get_coordinate_formatter(ndims):
    return ",".join("{}" for _ in range(ndims))


def tensor_network_align(*tns, ind_ids=None, trace=False, inplace=False):
    r"""Align an arbitrary number of tensor networks in a stack-like geometry::

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
    tns : sequence of TensorNetwork
        The TNs to align, should be structured and either effective 'vectors'
        (have a ``site_ind_id``) or 'operators' (have a ``up_ind_id`` and
        ``lower_ind_id``).
    ind_ids : None, or sequence of str
        String with format specifiers to id each level of sites with. Will be
        automatically generated like ``(tns[0].site_ind_id, "__ind_a{}__",
        "__ind_b{}__", ...)`` if not given.
    inplace : bool
        Whether to modify the input tensor networks inplace.

    Returns
    -------
    tns_aligned : sequence of TensorNetwork
    """
    if not inplace:
        tns = [tn.copy() for tn in tns]

    n = len(tns)
    coordinate_formatter = get_coordinate_formatter(tns[0]._NDIMS)

    if ind_ids is None:
        if hasattr(tns[0], "site_ind_id"):
            ind_ids = [tns[0].site_ind_id]
        else:
            ind_ids = [tns[0].lower_ind_id]
        ind_ids.extend(
            f"__ind_{get_symbol(i)}{coordinate_formatter}__"
            for i in range(n - 2)
        )
    else:
        ind_ids = tuple(ind_ids)

    for i, tn in enumerate(tns):
        if hasattr(tn, "site_ind_id"):
            if i == 0:
                tn.site_ind_id = ind_ids[i]
            elif i == n - 1:
                tn.site_ind_id = ind_ids[i - 1]
            else:
                raise ValueError("An TN 'vector' can only be aligned as the "
                                 "first or last TN in a sequence.")

        elif hasattr(tn, "upper_ind_id") and hasattr(tn, "lower_ind_id"):
            if i != 0:
                tn.upper_ind_id = ind_ids[i - 1]
            if i != n - 1:
                tn.lower_ind_id = ind_ids[i]

        else:
            raise ValueError("Can only align vectors and operators currently.")

    if trace:
        tns[-1].lower_ind_id = tns[0].upper_ind_id

    return tns


def tensor_network_apply_op_vec(
    tn_op,
    tn_vec,
    compress=False,
    **compress_opts
):
    """Apply a general a general tensor network representing an operator (has
    ``up_ind_id`` and ``lower_ind_id``) to a tensor network representing a
    vector (has ``site_ind_id``), by contracting each pair of tensors at each
    site then compressing the resulting tensor network. How the compression
    takes place is determined by the type of tensor network passed in. The
    returned tensor network has the same site indices as ``tn_vec``, and it is
    the ``lower_ind_id`` of ``tn_op`` that is contracted.

    Parameters
    ----------
    tn_op : TensorNetwork
        The tensor network representing the operator.
    tn_vec : TensorNetwork
        The tensor network representing the vector.
    compress : bool
        Whether to compress the resulting tensor network.
    compress_opts
        Options to pass to ``tn_vec.compress``.

    Returns
    -------
    tn_op_vec : TensorNetwork
    """
    A, x = tn_op.copy(), tn_vec.copy()

    # align the indices
    coordinate_formatter = get_coordinate_formatter(A._NDIMS)
    A.lower_ind_id = f"__tmp{coordinate_formatter}__"
    A.upper_ind_id = x.site_ind_id
    x.reindex_sites_(f"__tmp{coordinate_formatter}__")

    # form total network and contract each site
    x |= A
    for site in x.gen_sites_present():
        x ^= site

    x.fuse_multibonds_()
    # optionally compress
    if compress:
        x.compress(**compress_opts)

    return x



class TensorNetworkGen(TensorNetwork):
    """A tensor network which notionally has a single tensor per 'site',
    though these could be labelled arbitrarily could also be linked in an
    arbitrary geometry by bonds.
    """

    _NDIMS = 1
    _EXTRA_PROPS = (
        "_sites",
        "_site_tag_id",
    )

    def _compatible_arbgeom(self, other):
        """Check whether ``self`` and ``other`` represent the same set of
        sites and are tagged equivalently.
        """
        return isinstance(other, TensorNetworkGen) and all(
            getattr(self, e, 0) == getattr(other, e, 1)
            for e in TensorNetworkGen._EXTRA_PROPS
        )

    def combine(self, other, *, virtual=False, check_collisions=True):
        """Combine this tensor network with another, returning a new tensor
        network. If the two are compatible, cast the resulting tensor network
        to a :class:`TensorNetworkGen` instance.

        Parameters
        ----------
        other : TensorNetworkGen or TensorNetwork
            The other tensor network to combine with.
        virtual : bool, optional
            Whether the new tensor network should copy all the incoming tensors
            (``False``, the default), or view them as virtual (``True``).
        check_collisions : bool, optional
            Whether to check for index collisions between the two tensor
            networks before combining them. If ``True`` (the default), any
            inner indices that clash will be mangled.

        Returns
        -------
        TensorNetworkGen or TensorNetwork
        """
        new = super().combine(
            other, virtual=virtual, check_collisions=check_collisions
        )
        if self._compatible_arbgeom(other):
            new.view_as_(TensorNetworkGen, like=self)
        return new

    @property
    def nsites(self):
        """The total number of sites.
        """
        return len(self._sites)

    def gen_site_coos(self):
        """Generate the coordinates of all sites, same as ``self.sites``.
        """
        return self._sites

    @property
    def sites(self):
        """Tuple of the possible sites in this tensor network.
        """
        sites = getattr(self, "_sites", None)
        if sites is None:
            sites = tuple(self.gen_site_coos())
        return sites

    def _get_site_set(self):
        """The set of all sites.
        """
        if getattr(self, "_site_set", None) is None:
            self._site_set = set(self.sites)
        return self._site_set

    def gen_sites_present(self):
        """Generate the sites which are currently present (e.g. if a local view
        of a larger tensor network), based on whether their tags are present.

        Examples
        --------

            >>> tn = qtn.TN3D_rand(4, 4, 4, 2)
            >>> tn_sub = tn.select_local('I1,2,3', max_distance=1)
            >>> list(tn_sub.gen_sites_present())
            [(0, 2, 3), (1, 1, 3), (1, 2, 2), (1, 2, 3), (1, 3, 3), (2, 2, 3)]

        """
        return (
            site for site in self.gen_site_coos()
            if self.site_tag(site) in self.tag_map
        )

    @property
    def site_tag_id(self):
        """The string specifier for tagging each site of this tensor network.
        """
        return self._site_tag_id

    def site_tag(self, site):
        """The name of the tag specifiying the tensor at ``site``.
        """
        return self.site_tag_id.format(site)

    def retag_sites(self, new_id, where=None, inplace=False):
        """Modify the site tags for all or some tensors in this tensor network
        (without changing the ``site_tag_id``).

        Parameters
        ----------
        new_id : str
            A string with a format placeholder to accept a site, e.g. "S{}".
        where : None or sequence
            Which sites to update the index labels on. If ``None`` (default)
            all sites.
        inplace : bool
            Whether to retag in place.
        """
        if where is None:
            where = self.gen_sites_present()

        return self.retag(
            {self.site_tag(x): new_id.format(x) for x in where},
            inplace=inplace,
        )

    @property
    def site_tags(self):
        """All of the site tags.
        """
        if getattr(self, "_site_tags", None) is None:
            self._site_tags = tuple(map(self.site_tag, self.gen_site_coos()))
        return self._site_tags

    @property
    def site_tags_present(self):
        """All of the site tags still present in this tensor network.
        """
        return tuple(map(self.site_tag, self.gen_sites_present()))

    @site_tag_id.setter
    def site_tag_id(self, new_id):
        if self._site_tag_id != new_id:
            self.retag_sites(new_id, inplace=True)
            self._site_tag_id = new_id
            self._site_tags = None

    def retag_all(self, new_id, inplace=False):
        """Retag all sites and change the ``site_tag_id``.
        """
        tn = self if inplace else self.copy()
        tn.site_tag_id = new_id
        return tn

    retag_all_ = functools.partialmethod(retag_all, inplace=True)

    def _get_site_tag_set(self):
        """The oset of all site tags.
        """
        if getattr(self, "_site_tag_set", None) is None:
            self._site_tag_set = set(self.site_tags)
        return self._site_tag_set

    def filter_valid_site_tags(self, tags):
        """Get the valid site tags from ``tags``.
        """
        return oset(sorted(self._get_site_tag_set().intersection(tags)))

    def maybe_convert_coo(self, x):
        """Check if ``x`` is a valid site and convert to the corresponding site
        tag if so, else return ``x``.
        """
        try:
            if x in self._get_site_set():
                return self.site_tag(x)
        except TypeError:
            pass
        return x

    def gen_tags_from_coos(self, coos):
        """Generate the site tags corresponding to the given coordinates.
        """
        return map(self.site_tag, coos)

    def _get_tids_from_tags(self, tags, which="all"):
        """This is the function that lets coordinates such as ``site`` be
        used for many 'tag' based functions.
        """
        tags = self.maybe_convert_coo(tags)
        return super()._get_tids_from_tags(tags, which=which)

    def reset_cached_properties(self):
        """Reset any cached properties, one should call this when changing the
        actual geometry of a TN inplace, for example.
        """
        self._site_set = None
        self._site_tag_set = None
        self._site_tags = None

    @functools.wraps(tensor_network_align)
    def align(self, *args, inplace=False, **kwargs):
        return tensor_network_align(self, *args, inplace=inplace, **kwargs)

    align_ = functools.partialmethod(align, inplace=True)


def gauge_product_boundary_vector(
    tn,
    tags,
    which="all",
    max_bond=1,
    smudge=1e-6,
    canonize_distance=0,
    select_local_distance=None,
    select_local_opts=None,
    **contract_around_opts,
):
    tids = tn._get_tids_from_tags(tags, which)

    # form the double layer tensor network - this is the TN we will
    #     generate the actual gauges with
    if select_local_distance is None:
        # use the whole tensor network ...
        outer_inds = tn.outer_inds()
        dtn = tn.H & tn
    else:
        # ... or just a local patch
        select_local_opts = ensure_dict(select_local_opts)
        ltn = tn._select_local_tids(
            tids,
            max_distance=select_local_distance,
            virtual=False,
            **select_local_opts,
        )
        outer_inds = ltn.outer_inds()
        dtn = ltn.H | ltn

    # get all inds in the tagged region
    region_inds = set.union(*(set(tn.tensor_map[tid].inds) for tid in tids))

    # contract all 'physical' indices so that we have a single layer TN
    #     outside region and double layer sandwich inside region
    for ix in outer_inds:
        if (ix in region_inds) or (ix not in dtn.ind_map):
            # 1st condition - don't contract region sandwich
            # 2nd condition - if local selecting, will get multibonds so
            #     some indices already contracted
            continue
        dtn.contract_ind(ix)

    # form the single layer boundary of double layer tagged region
    dtids = dtn._get_tids_from_tags(tags, which)
    dtn._contract_around_tids(
        dtids,
        min_distance=1,
        max_bond=max_bond,
        canonize_distance=canonize_distance,
        **contract_around_opts,
    )

    # select this boundary and compress to ensure it is a product operator
    dtn = dtn._select_without_tids(dtids, virtual=True)
    dtn.compress_all_(max_bond=1)
    dtn.squeeze_()

    # each tensor in the boundary should now have exactly two inds
    #     connecting to the top and bottom of the tagged region double
    #     layer. Iterate over these, inserting the gauge into the original
    #     tensor network that would turn each of these boundary tensors
    #     into identities.
    for t in dtn:
        (ix,) = [i for i in t.inds if i in region_inds]
        _, s, VH = do("linalg.svd", t.data)
        s = s + smudge
        G = do("reshape", s ** 0.5, (-1, 1)) * VH
        Ginv = dag(VH) * do("reshape", s ** -0.5, (1, -1))

        tid_l, tid_r = sorted(tn.ind_map[ix], key=lambda tid: tid in tids)
        tn.tensor_map[tid_l].gate_(Ginv.T, ix)
        tn.tensor_map[tid_r].gate_(G, ix)

    return tn


_VALID_GATE_PROPAGATE = {'sites', 'register', False, True}
_LAZY_GATE_CONTRACT = {
    False,
    'split-gate',
    'swap-split-gate',
    'auto-split-gate',
}


class TensorNetworkGenVector(TensorNetworkGen):
    """A tensor network which notionally has a single tensor and outer index
    per 'site', though these could be labelled arbitrarily and could also be
    linked in an arbitrary geometry by bonds.
    """

    _EXTRA_PROPS = (
        "_sites",
        "_site_tag_id",
        "_site_ind_id",
    )

    @property
    def site_ind_id(self):
        """The string specifier for the physical indices.
        """
        return self._site_ind_id

    def site_ind(self, site):
        return self.site_ind_id.format(site)

    @property
    def site_inds(self):
        """Return a tuple of all site indices.
        """
        if getattr(self, "_site_inds", None) is None:
            self._site_inds = tuple(map(self.site_ind, self.gen_site_coos()))
        return self._site_inds

    @property
    def site_inds_present(self):
        """All of the site inds still present in this tensor network.
        """
        return tuple(map(self.site_ind, self.gen_sites_present()))

    def reset_cached_properties(self):
        """Reset any cached properties, one should call this when changing the
        actual geometry of a TN inplace, for example.
        """
        self._site_inds = None
        return super().reset_cached_properties()

    def reindex_sites(self, new_id, where=None, inplace=False):
        """Modify the site indices for all or some tensors in this vector
        tensor network (without changing the ``site_ind_id``).

        Parameters
        ----------
        new_id : str
            A string with a format placeholder to accept a site, e.g. "ket{}".
        where : None or sequence
            Which sites to update the index labels on. If ``None`` (default)
            all sites.
        inplace : bool
            Whether to reindex in place.
        """
        if where is None:
            where = self.gen_sites_present()

        return self.reindex(
            {self.site_ind(x): new_id.format(x) for x in where},
            inplace=inplace,
        )

    @site_ind_id.setter
    def site_ind_id(self, new_id):
        if self._site_ind_id != new_id:
            self.reindex_sites_(new_id)
            self._site_ind_id = new_id
            self._site_inds = None

    def reindex_all(self, new_id, inplace=False):
        """Reindex all physical sites and change the ``site_ind_id``.
        """
        tn = self if inplace else self.copy()
        tn.site_ind_id = new_id
        return tn

    reindex_all_ = functools.partialmethod(reindex_all, inplace=True)

    def gen_inds_from_coos(self, coos):
        """Generate the site inds corresponding to the given coordinates.
        """
        return map(self.site_ind, coos)

    def phys_dim(self, site=None):
        """Get the physical dimension of ``site``, defaulting to the first site
        if not specified.
        """
        if site is None:
            site = next(iter(self.gen_sites_present()))
        return self.ind_size(self.site_ind(site))

    def to_dense(
        self,
        *inds_seq,
        to_qarray=False,
        to_ket=None,
        **contract_opts
    ):
        """Contract this tensor network 'vector' into a dense array. By
        default, turn into a 'ket' ``qarray``, i.e. column vector of shape
        ``(d, 1)``.

        Parameters
        ----------
        inds_seq : sequence of sequences of str
            How to group the site indices into the dense array. By default,
            use a single group ordered like ``sites``, but only containing
            those sites which are still present.
        to_qarray : bool
            Whether to turn the dense array into a ``qarray``, if the backend
            would otherwise be ``'numpy'``.
        to_ket : None or str
            Whether to reshape the dense array into a ket (shape ``(d, 1)``
            array). If ``None`` (default), do this only if the ``inds_seq`` is
            not supplied.
        contract_opts
            Options to pass to
            :meth:`~quimb.tensor.tensor_core.TensorNewtork.contract`.

        Returns
        -------
        array
        """
        if not inds_seq:
            inds_seq = (self.site_inds_present,)
            if to_ket is None:
                to_ket = True

        x = TensorNetwork.to_dense(
            self, *inds_seq,
            to_qarray=to_qarray,
            **contract_opts
        )

        if to_ket:
            x = do("reshape", x, (-1, 1))

        return x

    to_qarray = functools.partialmethod(to_dense, to_qarray=True)

    def gate(
        self, G, where,
        contract=False,
        tags=None,
        propagate_tags=False,
        info=None,
        inplace=False,
        **compress_opts,
    ):
        r"""Apply a gate to this vector tensor network at sites ``where``. This
        is essentially a wrapper around
        :meth:`~quimb.tensor.tensor_core.TensorNetwork.gate_inds` apart from
        ``where`` can be specified as a list of sites, and tags can be
        optionally, intelligently propagated to the new gate tensor.

        .. math::

            | \psi \rangle \rightarrow G_\mathrm{where} | \psi \rangle

        Parameters
        ----------
        G : array_ike
            The gate array to apply, should match or be factorable into the
            shape ``(*phys_dims, *phys_dims)``.
        where : node or sequence[node]
            The sites to apply the gate to.
        contract : {False, True, 'split', 'reduce-split', 'split-gate',
                    'swap-split-gate', 'auto-split-gate'}, optional
            How to apply the gate, see
            :meth:`~quimb.tensor.tensor_core.TensorNetwork.gate_inds`.
        tags : str or sequence of str, optional
            Tags to add to the new gate tensor.
        propagate_tags : {False, True, 'register', 'sites'}, optional
            Whether to propagate tags to the new gate tensor::

                - False: no tags are propagated
                - True: all tags are propagated
                - 'register': only site tags corresponding to ``where`` are
                  added.
                - 'sites': all site tags on the current sites are propgated,
                  resulting in a lightcone like tagging.

        info : None or dict, optional
            Used to store extra optional information such as the singular
            values if not absorbed.
        inplace : bool, optional
            Whether to perform the gate operation inplace on the tensor network
            or not.
        compress_opts
            Supplied to :func:`~quimb.tensor.tensor_core.tensor_split` for any
            ``contract`` methods that involve splitting. Ignored otherwise.

        Returns
        -------
        TensorNetworkGenVector

        See Also
        --------
        TensorNetwork.gate_inds
        """
        check_opt('propagate_tags', propagate_tags, _VALID_GATE_PROPAGATE)

        tn = self if inplace else self.copy()

        if not isinstance(where, (tuple, list)):
            where = (where,)
        inds = tuple(map(tn.site_ind, where))

        # potentially add tags from current tensors to the new ones,
        # only do this if we are lazily adding the gate tensor(s)
        if (
            (contract in _LAZY_GATE_CONTRACT) and
            (propagate_tags in (True, 'sites'))
        ):
            old_tags = oset.union(
                *(t.tags for t in tn._inds_get(*inds))
            )
            if propagate_tags == 'sites':
                old_tags = tn.filter_valid_site_tags(old_tags)

            tags = tags_to_oset(tags)
            tags.update(old_tags)

        # perform the actual gating
        tn.gate_inds_(
            G, inds, contract=contract, tags=tags, info=info, **compress_opts
        )

        # possibly add tags based on where the gate was applied
        if propagate_tags == 'register':
            for ix, site in zip(inds, where):
                t, = tn._inds_get(ix)
                t.add_tag(tn.site_tag(site))

        return tn

    gate_ = functools.partialmethod(gate, inplace=True)

    def gate_simple_(self, G, where, gauges, renorm=True, **gate_opts):
        """Apply a gate to this vector tensor network at sites ``where``, using
        simple update style gauging of the tensors first, as supplied in
        ``gauges``. The new singular values for the bond are reinserted into
        ``gauges``.

        Parameters
        ----------
        G : array_like
            The gate to be applied.
        where : node or sequence[node]
            The sites to apply the gate to.
        gauges : dict[str, array_like]
            The store of gauge bonds, the keys being indices and the values
            being the vectors. Only bonds present in this dictionary will be
            used.
        renorm : bool, optional
            Whether to renormalise the singular after the gate is applied,
            before reinserting them into ``gauges``.
        """
        gate_opts.setdefault("absorb", None)
        gate_opts.setdefault("contract", "reduce-split")

        site_tags = tuple(map(self.site_tag, where))
        tn_where = self.select_any(site_tags)

        with tn_where.gauge_simple_temp(gauges, ungauge_inner=False):
            info = {}
            tn_where.gate_(G, where, info=info, **gate_opts)

            # inner ungauging is performed by tracking the new singular values
            (((_, ix), s),) = info.items()
            if renorm:
                s = s / s[0]
            gauges[ix] = s

        return self

    def gate_fit_local_(
        self,
        G,
        where,
        max_distance=0,
        fillin=0,
        gauges=None,
        **fit_opts,
    ):
        # select a local neighborhood of tensors
        tids = self._get_tids_from_tags(
            tuple(map(self.site_tag, where)), 'any'
        )
        if len(tids) == 2:
            tids = self._get_string_between_tids(*tids)

        k = self._select_local_tids(
            tids,
            max_distance=max_distance,
            fillin=fillin,
            virtual=True,
        )

        if gauges:
            outer, inner = k.gauge_simple_insert(gauges)

        Gk = k.gate(G, where)
        k.fit_(Gk, **fit_opts)

        if gauges:
            k.gauge_simple_remove(outer, inner)

        if gauges is not None:
            k.gauge_all_simple_(gauges=gauges)

    def local_expectation_cluster(
        self,
        G,
        where,
        normalized=True,
        max_distance=0,
        fillin=False,
        gauges=None,
        optimize="auto",
        max_bond=None,
        rehearse=False,
        **contract_opts,
    ):
        r"""Approximately compute a single local expectation value of the gate
        ``G`` at sites ``where``, either treating the environment beyond
        ``max_distance`` as the identity, or using simple update style bond
        gauges as supplied in ``gauges``.

        This selects a local neighbourhood of tensors up to distance
        ``max_distance`` away from ``where``, then traces over dangling bonds
        after potentially inserting the bond gauges, to form an approximate
        version of the reduced density matrix.

        .. math::

            \langle \psi | G | \psi \rangle
            \approx
            \frac{
            \mathrm{Tr} [ G \tilde{\rho}_\mathrm{where} ]
            }{
            \mathrm{Tr} [ \tilde{\rho}_\mathrm{where} ]
            }

        assuming ``normalized==True``.

        Parameters
        ----------
        G : array_like
            The gate to compute the expecation of.
        where : node or sequence[node]
            The sites to compute the expectation at.
        normalized : bool, optional
            Whether to locally normalize the result, i.e. divide by the
            expectation value of the identity.
        max_distance : int, optional
            The maximum graph distance to include tensors neighboring ``where``
            when computing the expectation. The default 0 means only the
            tensors at sites ``where`` are used.
        fillin : bool or int, optional
            When selecting the local tensors, whether and how many times to
            'fill-in' corner tensors attached multiple times to the local
            region. On a lattice this fills in the corners. See
            :meth:`~quimb.tensor.tensor_core.TensorNetwork.select_local`.
        gauges : dict[str, array_like], optional
            The store of gauge bonds, the keys being indices and the values
            being the vectors. Only bonds present in this dictionary will be
            used.
        optimize : str or PathOptimizer, optional
            The contraction path optimizer to use, when exactly contracting the
            local tensors.
        max_bond : None or int, optional
            If specified, use compressed contraction.
        rehearse : {False, 'tn', 'tree', True}, optional
            Whether to perform the computations or not::

                - False: perform the computation.
                - 'tn': return the tensor networks of each local expectation,
                  without running the path optimizer.
                - 'tree': run the path optimizer and return the
                  ``cotengra.ContractonTree`` for each local expectation.
                - True: run the path optimizer and return the ``PathInfo`` for
                  each local expectation.

        Returns
        -------
        expectation : float
        """
        # select a local neighborhood of tensors
        tids = self._get_tids_from_tags(
            tuple(map(self.site_tag, where)), 'any'
        )

        if len(tids) == 2:
            tids = self._get_string_between_tids(*tids)

        k = self._select_local_tids(
            tids,
            max_distance=max_distance,
            fillin=fillin,
            virtual=False,
        )

        if gauges is not None:
            # gauge the region with simple update style bond gauges
            k.gauge_simple_insert(gauges)

        if max_bond is not None:
            return k.local_expectation(
                G=G,
                where=where,
                max_bond=max_bond,
                optimize=optimize,
                normalized=normalized,
                rehearse=rehearse,
                **contract_opts
            )

        return k.local_expectation_exact(
            G=G,
            where=where,
            optimize=optimize,
            normalized=normalized,
            rehearse=rehearse,
            **contract_opts
        )

    local_expectation_simple = deprecated(
        local_expectation_cluster,
        'local_expectation_simple',
        'local_expectation_cluster',
    )

    def compute_local_expectation_cluster(
        self,
        terms,
        *,
        max_distance=0,
        fillin=False,
        normalized=True,
        gauges=None,
        optimize="auto",
        max_bond=None,
        return_all=False,
        rehearse=False,
        executor=None,
        progbar=False,
        **contract_opts,
    ):
        r"""Compute all local expectations of the given terms, either treating
        the environment beyond ``max_distance`` as the identity, or using
        simple update style bond gauges as supplied in ``gauges``.

        This selects a local neighbourhood of tensors up to distance
        ``max_distance`` away from each term's sites, then traces over
        dangling bonds after potentially inserting the bond gauges, to form
        an approximate version of the reduced density matrix.

        .. math::

            \sum_\mathrm{i}
            \langle \psi | G_\mathrm{i} | \psi \rangle
            \approx
            \sum_\mathrm{i}
            \frac{
            \mathrm{Tr} [ G_\mathrm{i} \tilde{\rho}_\mathrm{i} ]
            }{
            \mathrm{Tr} [ \tilde{\rho}_\mathrm{i} ]
            }

        assuming ``normalized==True``.

        Parameters
        ----------
        terms : dict[node or (node, node), array_like]
            The terms to compute the expectation of, with keys being the sites
            and values being the local operators.
        max_distance : int, optional
            The maximum graph distance to include tensors neighboring each
            term's sites when computing the expectation. The default 0 means
            only the tensors at sites of each term are used.
        fillin : bool or int, optional
            When selecting the local tensors, whether and how many times to
            'fill-in' corner tensors attached multiple times to the local
            region. On a lattice this fills in the corners. See
            :meth:`~quimb.tensor.tensor_core.TensorNetwork.select_local`.
        normalized : bool, optional
            Whether to locally normalize the result, i.e. divide by the
            expectation value of the identity. This implies that a different
            normalization factor is used for each term.
        gauges : dict[str, array_like], optional
            The store of gauge bonds, the keys being indices and the values
            being the vectors. Only bonds present in this dictionary will be
            used.
        optimize : str or PathOptimizer, optional
            The contraction path optimizer to use, when exactly contracting the
            local tensors.
        max_bond : None or int, optional
            If specified, use compressed contraction.
        return_all : bool, optional
            Whether to return all results, or just the summed expectation.
        rehearse : {False, 'tn', 'tree', True}, optional
            Whether to perform the computations or not::

                - False: perform the computation.
                - 'tn': return the tensor networks of each local expectation,
                  without running the path optimizer.
                - 'tree': run the path optimizer and return the
                  ``cotengra.ContractonTree`` for each local expectation.
                - True: run the path optimizer and return the ``PathInfo`` for
                  each local expectation.

        executor : Executor, optional
            If supplied compute the terms in parallel using this executor.
        progbar : bool, optional
            Whether to show a progress bar.
        contract_opts
            Supplied to
            :meth:`~quimb.tensor.tensor_core.TensorNetwork.contract`.

        Returns
        -------
        expecs : float or dict[node or (node, node), float]
            If ``return_all==False``, return the summed expectation value of
            the given terms. Otherwise, return a dictionary mapping each term's
            location to the expectation value.
        """
        return _compute_expecs_maybe_in_parallel(
            fn=_tn_local_expectation_cluster,
            tn=self,
            terms=terms,
            return_all=return_all,
            executor=executor,
            progbar=progbar,
            normalized=normalized,
            max_distance=max_distance,
            fillin=fillin,
            gauges=gauges,
            optimize=optimize,
            rehearse=rehearse,
            max_bond=max_bond,
            **contract_opts,
        )

    compute_local_expectation_simple = deprecated(
        compute_local_expectation_cluster,
        'compute_local_expectation_simple',
        'compute_local_expectation_cluster',
    )

    def local_expectation_exact(
        self,
        G,
        where,
        optimize="auto-hq",
        normalized=True,
        rehearse=False,
        **contract_opts,
    ):
        """Compute the local expectation of operator ``G`` at site(s) ``where``
        by exactly contracting the full overlap tensor network.
        """
        k_inds = tuple(map(self.site_ind, where))
        b_inds = tuple(map("_bra{}".format, where))
        b = self.conj().reindex_(dict(zip(k_inds, b_inds)))
        tn = (b | self)

        if rehearse:
            if rehearse == 'tn':
                return tn
            if rehearse == 'tree':
                return tn.contraction_tree(
                    optimize, output_inds=k_inds + b_inds)
            if rehearse:
                return tn.contraction_info(
                    optimize, output_inds=k_inds + b_inds)

        rho = tn.to_dense(k_inds, b_inds, optimize=optimize, **contract_opts)
        expec = do("tensordot", rho, G, axes=((0, 1), (1, 0)))
        if normalized:
            expec = expec / do("trace", rho)

        return expec

    def compute_local_expectation_exact(
        self,
        terms,
        optimize="auto-hq",
        *,
        normalized=True,
        return_all=False,
        rehearse=False,
        executor=None,
        progbar=False,
        **contract_opts,
    ):
        """Compute the local expectations of many operators,
        by exactly contracting the full overlap tensor network.

        Parameters
        ----------
        terms : dict[node or (node, node), array_like]
            The terms to compute the expectation of, with keys being the sites
            and values being the local operators.
        optimize : str or PathOptimizer, optional
            The contraction path optimizer to use, when exactly contracting the
            full tensor network.
        normalized : bool, optional
            Whether to normalize the result.
        return_all : bool, optional
            Whether to return all results, or just the summed expectation.
        rehearse : {False, 'tn', 'tree', True}, optional
            Whether to perform the computations or not::

                - False: perform the computation.
                - 'tn': return the tensor networks of each local expectation,
                  without running the path optimizer.
                - 'tree': run the path optimizer and return the
                  ``cotengra.ContractonTree`` for each local expectation.
                - True: run the path optimizer and return the ``PathInfo`` for
                  each local expectation.

        executor : Executor, optional
            If supplied compute the terms in parallel using this executor.
        progbar : bool, optional
            Whether to show a progress bar.
        contract_opts
            Supplied to
            :meth:`~quimb.tensor.tensor_core.TensorNetwork.contract`.

        Returns
        -------
        expecs : float or dict[node or (node, node), float]
            If ``return_all==False``, return the summed expectation value of
            the given terms. Otherwise, return a dictionary mapping each term's
            location to the expectation value.
        """
        return _compute_expecs_maybe_in_parallel(
            fn=_tn_local_expectation_exact,
            tn=self,
            terms=terms,
            return_all=return_all,
            executor=executor,
            progbar=progbar,
            optimize=optimize,
            normalized=normalized,
            rehearse=rehearse,
            **contract_opts,
        )

    def partial_trace(
        self,
        keep,
        max_bond,
        optimize,
        flatten=True,
        reduce=False,
        normalized=True,
        symmetrized='auto',
        rehearse=False,
        method='contract_compressed',
        **contract_compressed_opts,
    ):
        """Partially trace this tensor network state, keeping only the sites in
        ``keep``, using compressed contraction.

        Parameters
        ----------
        keep : iterable of hashable
            The sites to keep.
        max_bond : int
            The maximum bond dimensions to use while compressed contracting.
        optimize : str or PathOptimizer, optional
            The contraction path optimizer to use, should specifically generate
            contractions paths designed for compressed contraction.
        flatten : {False, True, 'all'}, optional
            Whether to force 'flattening' (contracting all physical indices) of
            the tensor network before  contraction, whilst this makes the TN
            generally more complex to contract, the accuracy is usually
            improved. If ``'all'`` also flatten the tensors in ``keep``.
        reduce : bool, optional
            Whether to first 'pull' the physical indices off their respective
            tensors using QR reduction. Experimental.
        normalized : bool, optional
            Whether to normalize the reduced density matrix at the end.
        symmetrized : {'auto', True, False}, optional
            Whether to symmetrize the reduced density matrix at the end. This
            should be unecessary if ``flatten`` is set to ``True``.
        rehearse : {False, 'tn', 'tree', True}, optional
            Whether to perform the computation or not::

                - False: perform the computation.
                - 'tn': return the tensor network without running the path
                  optimizer.
                - 'tree': run the path optimizer and return the
                  ``cotengra.ContractonTree``..
                - True: run the path optimizer and return the ``PathInfo``.

        contract_compressed_opts : dict, optional
            Additional keyword arguments to pass to
            :meth:`~quimb.tensor.tensor_core.TensorNetwork.contract_compressed`.

        Returns
        -------
        rho : array_like
            The reduce density matrix of sites in ``keep``.
        """
        if symmetrized == 'auto':
            symmetrized = not flatten

        # form the partial trace
        k_inds = tuple(map(self.site_ind, keep))

        k = self.copy()
        if reduce:
            k.reduce_inds_onto_bond(*k_inds, tags='__BOND__', drop_tags=True)

        b_inds = tuple(map("_bra{}".format, keep))
        b = k.conj().reindex_(dict(zip(k_inds, b_inds)))

        tn = (b | k)
        output_inds = k_inds + b_inds

        if flatten:
            for site in self.gen_site_coos():
                if (site not in keep) or (flatten == 'all'):
                    # check if site exists still to permit e.g. local methods
                    # to use this same logic
                    tag = tn.site_tag(site)
                    if tag in tn.tag_map:
                        tn ^= tag

        tn.fuse_multibonds_()

        if method == 'contract_compressed':

            if reduce:
                output_inds = None
                tn, tn_reduced = tn.partition('__BOND__', inplace=True)

            if rehearse:
                if rehearse == 'tn':
                    return tn
                if rehearse == 'tree':
                    return tn.contraction_tree(
                        optimize, output_inds=output_inds)
                if rehearse:
                    return tn.contraction_info(
                        optimize, output_inds=output_inds)

            t_rho = tn.contract_compressed(
                optimize,
                max_bond=max_bond,
                output_inds=output_inds,
                **contract_compressed_opts,
            )

            if reduce:
                t_rho |= tn_reduced

            rho = t_rho.to_dense(k_inds, b_inds)

        elif method == 'contract_around':
            tn.contract_around_(
                tuple(map(self.site_tag, keep)), 'any',
                max_bond=max_bond,
                **contract_compressed_opts
            )

            if rehearse:
                if rehearse == 'tn':
                    return tn
                if rehearse == 'tree':
                    return tn.contraction_tree(
                        optimize, output_inds=output_inds)
                if rehearse:
                    return tn.contraction_info(
                        optimize, output_inds=output_inds)

            rho = tn.to_dense(
                k_inds, b_inds, optimize=optimize,
            )

        else:
            raise ValueError(f"Unknown method: {method}.")

        if symmetrized:
            rho = (rho + dag(rho)) / 2

        if normalized:
            rho = rho / do("trace", rho)

        return rho

    def local_expectation(
        self,
        G,
        where,
        max_bond,
        optimize,
        flatten=True,
        normalized=True,
        symmetrized='auto',
        reduce=False,
        rehearse=False,
        **contract_compressed_opts,
    ):
        """Compute the local expectation of operator ``G`` at site(s) ``where``
        by approximately contracting the full overlap tensor network.

        Parameters
        ----------
        G : array_like
            The local operator to compute the expectation of.
        where : node or sequence of nodes
            The sites to compute the expectation for.
        max_bond : int
            The maximum bond dimensions to use while compressed contracting.
        optimize : str or PathOptimizer, optional
            The contraction path optimizer to use, should specifically generate
            contractions paths designed for compressed contraction.
        method : {'rho', 'rho-reduced'}, optional
            The method to use to compute the expectation value.
        flatten : bool, optional
            Whether to force 'flattening' (contracting all physical indices) of
            the tensor network before  contraction, whilst this makes the TN
            generally more complex to contract, the accuracy is usually much
            improved.
        normalized : bool, optional
            If computing via `partial_trace`, whether to normalize the reduced
            density matrix at the end.
        symmetrized : {'auto', True, False}, optional
            If computing via `partial_trace`, whether to symmetrize the reduced
            density matrix at the end. This should be unecessary if ``flatten``
            is set to ``True``.
        rehearse : {False, 'tn', 'tree', True}, optional
            Whether to perform the computation or not::

                - False: perform the computation.
                - 'tn': return the tensor network without running the path
                  optimizer.
                - 'tree': run the path optimizer and return the
                  ``cotengra.ContractonTree``..
                - True: run the path optimizer and return the ``PathInfo``.

        contract_compressed_opts : dict, optional
            Additional keyword arguments to pass to
            :meth:`~quimb.tensor.tensor_core.TensorNetwork.contract_compressed`.

        Returns
        -------
        expec : float
        """
        rho = self.partial_trace(
            keep=where,
            max_bond=max_bond,
            optimize=optimize,
            flatten=flatten,
            reduce=reduce,
            normalized=normalized,
            symmetrized=symmetrized,
            rehearse=rehearse,
            **contract_compressed_opts,
        )
        if rehearse:
            return rho

        return do("tensordot", rho, G, axes=((0, 1), (1, 0)))

    def compute_local_expectation(
        self,
        terms,
        max_bond,
        optimize,
        *,
        flatten=True,
        normalized=True,
        symmetrized='auto',
        reduce=False,
        return_all=False,
        rehearse=False,
        executor=None,
        progbar=False,
        **contract_compressed_opts,
    ):
        """Compute the local expectations of many local operators, by
        approximately contracting the full overlap tensor network.

        Parameters
        ----------
        terms : dict[node or (node, node), array_like]
            The terms to compute the expectation of, with keys being the sites
            and values being the local operators.
        max_bond : int
            The maximum bond dimension to use during contraction.
        optimize : str or PathOptimizer
            The compressed contraction path optimizer to use.
        method : {'rho', 'rho-reduced'}, optional
            The method to use to compute the expectation value.

                - 'rho': compute the expectation value via the reduced density
                  matrix.
                - 'rho-reduced': compute the expectation value via the reduced
                  density matrix, having reduced the physical indices onto the
                  bonds first.

        flatten : bool, optional
            Whether to force 'flattening' (contracting all physical indices) of
            the tensor network before  contraction, whilst this makes the TN
            generally more complex to contract, the accuracy can often be much
            improved.
        normalized : bool, optional
            Whether to locally normalize the result.
        symmetrized : {'auto', True, False}, optional
            Whether to symmetrize the reduced density matrix at the end. This
            should be unecessary if ``flatten`` is set to ``True``.
        return_all : bool, optional
            Whether to return all results, or just the summed expectation. If
            ``rehease is not False``, this is ignored and a dict is always
            returned.
        rehearse : {False, 'tn', 'tree', True}, optional
            Whether to perform the computations or not::

                - False: perform the computation.
                - 'tn': return the tensor networks of each local expectation,
                  without running the path optimizer.
                - 'tree': run the path optimizer and return the
                  ``cotengra.ContractonTree`` for each local expectation.
                - True: run the path optimizer and return the ``PathInfo`` for
                  each local expectation.

        executor : Executor, optional
            If supplied compute the terms in parallel using this executor.
        progbar : bool, optional
            Whether to show a progress bar.
        contract_compressed_opts
            Supplied to
            :meth:`~quimb.tensor.tensor_core.TensorNetwork.contract_compressed`.

        Returns
        -------
        expecs : float or dict[node or (node, node), float]
            If ``return_all==False``, return the summed expectation value of
            the given terms. Otherwise, return a dictionary mapping each term's
            location to the expectation value.
        """
        return _compute_expecs_maybe_in_parallel(
            fn=_tn_local_expectation,
            tn=self,
            terms=terms,
            return_all=return_all,
            executor=executor,
            progbar=progbar,
            max_bond=max_bond,
            optimize=optimize,
            normalized=normalized,
            symmetrized=symmetrized,
            reduce=reduce,
            flatten=flatten,
            rehearse=rehearse,
            **contract_compressed_opts,
        )

    compute_local_expectation_rehearse = functools.partialmethod(
        compute_local_expectation, rehearse=True)

    compute_local_expectation_tn = functools.partialmethod(
        compute_local_expectation, rehearse='tn')


class TensorNetworkGenOperator(TensorNetworkGen):
    """A tensor network which notionally has a single tensor and two outer
    indices per 'site', though these could be labelled arbitrarily and could
    also be linked in an arbitrary geometry by bonds.
    """

    _EXTRA_PROPS = (
        "_sites",
        "_site_tag_id",
        "_upper_ind_id",
        "_lower_ind_id",
    )

    @property
    def upper_ind_id(self):
        """The string specifier for the upper phyiscal indices.
        """
        return self._upper_ind_id

    @upper_ind_id.setter
    def upper_ind_id(self, new_id):
        if new_id == self._lower_ind_id:
            raise ValueError("Setting the same upper and upper index ids will"
                             " make the two ambiguous.")

        if self._upper_ind_id != new_id:
            self.reindex_upper_sites_(new_id)
            self._upper_ind_id = new_id
            self._upper_inds = None

    def upper_ind(self, site):
        """Get the upper physical index name of ``site``.
        """
        return self.upper_ind_id.format(site)

    @property
    def upper_inds(self):
        """Return a tuple of all upper indices.
        """
        if getattr(self, "_upper_inds", None) is None:
            self._upper_inds = tuple(map(self.upper_ind, self.gen_site_coos()))
        return self._upper_inds

    @property
    def upper_inds_present(self):
        """Return a tuple of all upper indices still present in the tensor
        network.
        """
        return tuple(map(self.upper_ind, self.gen_sites_present()))

    @property
    def lower_ind_id(self):
        """The string specifier for the lower phyiscal indices.
        """
        return self._lower_ind_id

    @lower_ind_id.setter
    def lower_ind_id(self, new_id):
        if new_id == self._upper_ind_id:
            raise ValueError("Setting the same upper and lower index ids will"
                             " make the two ambiguous.")

        if self._lower_ind_id != new_id:
            self.reindex_lower_sites_(new_id)
            self._lower_ind_id = new_id
            self._lower_inds = None

    def lower_ind(self, site):
        """Get the lower physical index name of ``site``.
        """
        return self.lower_ind_id.format(site)

    @property
    def lower_inds(self):
        """Return a tuple of all lower indices.
        """
        if getattr(self, "_lower_inds", None) is None:
            self._lower_inds = tuple(map(self.lower_ind, self.gen_site_coos()))
        return self._lower_inds

    @property
    def lower_inds_present(self):
        """Return a tuple of all lower indices still present in the tensor
        network.
        """
        return tuple(map(self.lower_ind, self.gen_sites_present()))

    def to_dense(
        self,
        *inds_seq,
        to_qarray=False,
        **contract_opts
    ):
        """Contract this tensor network 'operator' into a dense array.

        Parameters
        ----------
        inds_seq : sequence of sequences of str
            How to group the site indices into the dense array. By default,
            use a single group ordered like ``sites``, but only containing
            those sites which are still present.
        to_qarray : bool
            Whether to turn the dense array into a ``qarray``, if the backend
            would otherwise be ``'numpy'``.
        contract_opts
            Options to pass to
            :meth:`~quimb.tensor.tensor_core.TensorNewtork.contract`.

        Returns
        -------
        array
        """
        if not inds_seq:
            inds_seq = (self.upper_inds_present, self.lower_inds_present)

        return TensorNetwork.to_dense(
            self, *inds_seq,
            to_qarray=to_qarray,
            **contract_opts
        )

    to_qarray = functools.partialmethod(to_dense, to_qarray=True)

    def phys_dim(self, site=None, which='upper'):
        """Get the physical dimension of ``site``.
        """
        if site is None:
            site = next(iter(self.gen_sites_present()))

        if which == 'upper':
            return self[site].ind_size(self.upper_ind(site))

        if which == 'lower':
            return self[site].ind_size(self.lower_ind(site))


def _compute_expecs_maybe_in_parallel(
    fn,
    tn,
    terms,
    return_all=False,
    executor=None,
    progbar=False,
    **kwargs,
):
    """Unified helper function for the various methods that compute many
    expectations, possibly in parallel, possibly with a progress bar.
    """
    if not isinstance(terms, dict):
        terms = dict(terms.items())

    if executor is None:
        results = (fn(tn, G, where, **kwargs) for where, G in terms.items())
    else:
        if hasattr(executor, 'scatter'):
            tn = executor.scatter(tn)

        futures = [
            executor.submit(fn, tn, G, where, **kwargs)
            for where, G in terms.items()
        ]
        results = (future.result() for future in futures)

    if progbar:
        results = Progbar(results, total=len(terms))

    expecs = dict(zip(terms.keys(), results))

    if return_all or kwargs.get('rehearse', False):
        return expecs

    return functools.reduce(add, expecs.values())


def _tn_local_expectation(tn, *args, **kwargs):
    """Define as function for pickleability.
    """
    return tn.local_expectation(*args, **kwargs)


def _tn_local_expectation_cluster(tn, *args, **kwargs):
    """Define as function for pickleability.
    """
    return tn.local_expectation_cluster(*args, **kwargs)


def _tn_local_expectation_exact(tn, *args, **kwargs):
    """Define as function for pickleability.
    """
    return tn.local_expectation_exact(*args, **kwargs)

