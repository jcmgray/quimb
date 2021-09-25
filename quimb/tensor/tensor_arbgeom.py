import functools
from operator import add

from autoray import do, dag

from ..utils import check_opt, ensure_dict
from ..utils import progbar as Progbar
from .tensor_core import TensorNetwork


class TensorNetworkGen(TensorNetwork):
    """A tensor network which notionally has a single tensor per 'site',
    though these could be labelled arbitrarily could also be linked in an
    arbitrary geometry by bonds.
    """

    _EXTRA_PROPS = (
        "_sites",
        "_site_tag_id",
    )

    def _compatible_arbgeom(self, other):
        """Check whether ``self`` and ``other`` represent the same set of
        sites and are tagged equivalently.
        """
        return isinstance(other, TensorNetworkGen) and all(
            getattr(self, e) == getattr(other, e)
            for e in TensorNetworkGen._EXTRA_PROPS
        )

    def __and__(self, other):
        new = super().__and__(other)
        if self._compatible_arbgeom(other):
            new.view_as_(TensorNetworkGen, like=self)
        return new

    def __or__(self, other):
        new = super().__or__(other)
        if self._compatible_arbgeom(other):
            new.view_as_(TensorNetworkGen, like=self)
        return new

    @property
    def sites(self):
        """The sites of this arbitrary geometry tensor network.
        """
        return self._sites

    @property
    def nsites(self):
        """The total number of sites.
        """
        return len(self._sites)

    @property
    def site_tag_id(self):
        """The string specifier for tagging each site of this generic TN.
        """
        return self._site_tag_id

    def site_tag(self, site):
        """The name of the tag specifiying the tensor at ``site``.
        """
        return self.site_tag_id.format(site)

    @property
    def site_tags(self):
        """All of the site tags.
        """
        return tuple(map(self.site_tag, self.sites))

    def maybe_convert_coo(self, x):
        """Check if ``x`` is a tuple of two ints and convert to the
        corresponding site tag if so.
        """
        if x in self.sites:
            return self.site_tag(x)
        return x

    def _get_tids_from_tags(self, tags, which="all"):
        """This is the function that lets coordinates such as ``site`` be
        used for many 'tag' based functions.
        """
        tags = self.maybe_convert_coo(tags)
        return super()._get_tids_from_tags(tags, which=which)


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


class TensorNetworkGenVector(TensorNetworkGen, TensorNetwork):
    """A tensor network which notionally has a single tensor and outer index
    per 'site', though these could be labelled arbitrarily could also be linked
    in an arbitrary geometry by bonds.
    """

    _EXTRA_PROPS = (
        "_sites",
        "_site_tag_id",
        "_site_ind_id",
    )

    @property
    def site_ind_id(self):
        return self._site_ind_id

    def site_ind(self, site):
        return self.site_ind_id.format(site)

    def gate(self, G, where, inplace=False, **gate_opts):
        r"""Apply a gate to this vector tensor network at sites ``where``.

        .. math::

            | \psi \rangle \rightarrow G_\mathrm{where} | \psi \rangle

        Parameters
        ----------
        G : array_like
            The gate to be applied.
        where : node or sequence[node]
            The sites to apply the gate to.
        inplace : bool, optional
            Whether to apply the gate in place.
        gate_opts
            Keyword arguments to be passed to
            :func:`~quimb.tensor.tensor_core.TensorNetwork.gate_inds`.
        """
        if not isinstance(where, (tuple, list)):
            where = (where,)
        inds = tuple(map(self.site_ind, where))
        return self.gate_inds(G, inds, inplace=inplace, **gate_opts)

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

    def local_expectation_simple(
        self,
        G,
        where,
        normalized=True,
        max_distance=0,
        gauges=None,
        optimize="auto",
        rehearse=False,
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
        gauges : dict[str, array_like], optional
            The store of gauge bonds, the keys being indices and the values
            being the vectors. Only bonds present in this dictionary will be
            used.
        optimize : str or PathOptimizer, optional
            The contraction path optimizer to use, when exactly contracting the
            local tensors.

        Returns
        -------
        expectation : float
        """

        # select a local neighborhood of tensors
        site_tags = tuple(map(self.site_tag, where))
        k = self.select_local(
            site_tags, "any", max_distance=max_distance, virtual=False,
        )

        if gauges is not None:
            # gauge the region with simple update style bond gauges
            k.gauge_simple_insert(gauges)

        k_inds = tuple(map(self.site_ind, where))
        b_inds = tuple(map("_bra{}".format, where))
        b = k.conj().reindex_(dict(zip(k_inds, b_inds)))

        tn = (b | k)

        if rehearse:
            if rehearse == 'tn':
                return tn
            if rehearse == 'tree':
                return tn.contraction_tree(
                    optimize, output_inds=k_inds + b_inds)
            if rehearse:
                return tn.contraction_info(
                    optimize, output_inds=k_inds + b_inds)

        rho = tn.to_dense(k_inds, b_inds, optimize=optimize)
        expec = do("trace", rho @ G)
        if normalized:
            expec = expec / do("trace", rho)

        return expec

    def compute_local_expectation_simple(
        self,
        terms,
        *,
        max_distance=0,
        normalized=True,
        gauges=None,
        optimize="auto",
        return_all=False,
        rehearse=False,
        progbar=False,
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
        return_all : bool, optional
            Whether to return all results, or just the summed expectation.

        Returns
        -------
        expecs : float or dict[node or (node, node), float]
            If ``return_all==False``, return the summed expectation value of
            the given terms. Otherwise, return a dictionary mapping each term's
            location to the expectation value.
        """
        expecs = {}

        if progbar:
            items = Progbar(terms.items())
        else:
            items = terms.items()

        for where, G in items:
            expecs[where] = self.local_expectation_simple(
                G,
                where,
                normalized=normalized,
                max_distance=max_distance,
                gauges=gauges,
                optimize=optimize,
                rehearse=rehearse,
            )

        if return_all or rehearse:
            return expecs

        return functools.reduce(add, expecs.values())

    def local_expectation_exact(
        self, G, where, optimize="auto-hq", normalized=True, rehearse=False,
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

        rho = tn.to_dense(k_inds, b_inds, optimize=optimize)
        expec = do("trace", rho @ G)
        if normalized:
            expec = expec / do("trace", rho)

        return expec

    def compute_local_expectation_exact(
        self, terms, optimize="auto-hq", normalized=True, return_all=False,
        rehearse=False, progbar=False,
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
                - True: run the path optimizer and return the
                  ``cotengra.ContractonTree`` for each local expectation.
                - True: run the path optimizer and return the ``PathInfo`` for
                  each local expectation.

        Returns
        -------
        expecs : float or dict[node or (node, node), float]
            If ``return_all==False``, return the summed expectation value of
            the given terms. Otherwise, return a dictionary mapping each term's
            location to the expectation value.
        """
        expecs = {}

        if progbar:
            items = Progbar(terms.items())
        else:
            items = terms.items()

        for where, G in items:
            expecs[where] = self.local_expectation_exact(
                G, where, optimize=optimize,
                normalized=normalized, rehearse=rehearse
            )

        if return_all or rehearse:
            return expecs
        return functools.reduce(add, expecs.values())

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
                - True: run the path optimizer and return the
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

        if reduce:
            k = self.copy()
            k.reduce_inds_onto_bond(*k_inds, tags='__BOND__', drop_tags=True)
        else:
            k = self

        b_inds = tuple(map("_bra{}".format, keep))
        b = k.conj().reindex_(dict(zip(k_inds, b_inds)))

        tn = b & k
        output_inds = k_inds + b_inds

        if flatten:
            for site in self.sites:
                if (site not in keep) or (flatten == 'all'):
                    tn ^= site
            if reduce and (flatten == 'all'):
                tn ^= '__BOND__'

        if rehearse:
            if rehearse == 'tn':
                return tn
            if rehearse == 'tree':
                return tn.contraction_tree(optimize, output_inds=output_inds)
            if rehearse:
                return tn.contraction_info(optimize, output_inds=output_inds)

        t_rho = tn.contract_compressed(
            optimize,
            max_bond=max_bond,
            output_inds=output_inds,
            **contract_compressed_opts,
        )

        rho = t_rho.to_dense(k_inds, b_inds)

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
        method='rho',
        flatten=True,
        normalized=True,
        symmetrized='auto',
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
                - True: run the path optimizer and return the
                  ``cotengra.ContractonTree``..
                - True: run the path optimizer and return the ``PathInfo``.

        contract_compressed_opts : dict, optional
            Additional keyword arguments to pass to
            :meth:`~quimb.tensor.tensor_core.TensorNetwork.contract_compressed`.

        Returns
        -------
        expec : float
        """
        check_opt('method', method, ('rho', 'rho-reduced'))
        reduce = method == 'rho-reduced'

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

        return do("trace", rho @ G)

    def compute_local_expectation(
        self,
        terms,
        max_bond,
        optimize,
        method='rho',
        flatten=True,
        normalized=True,
        symmetrized='auto',
        return_all=False,
        rehearse=False,
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
        normalized : bool, optional
            Whether to locally normalize the result.
        flatten : bool, optional
            Whether to force 'flattening' (contracting all physical indices) of
            the tensor network before  contraction, whilst this makes the TN
            generally more complex to contract, the accuracy is usually much
            improved.
        rehearse : {False, 'tn', 'tree', True}, optional
            Whether to perform the computations or not::

                - False: perform the computation.
                - 'tn': return the tensor networks of each local expectation,
                  without running the path optimizer.
                - True: run the path optimizer and return the
                  ``cotengra.ContractonTree`` for each local expectation.
                - True: run the path optimizer and return the ``PathInfo`` for
                  each local expectation.

        return_all : bool, optional
            Whether to return all results, or just the summed expectation. If
            ``rehease is not False``, this is ignored and a dict is always
            returned.

        Returns
        -------
        expecs : float or dict[node or (node, node), float]
            If ``return_all==False``, return the summed expectation value of
            the given terms. Otherwise, return a dictionary mapping each term's
            location to the expectation value.
        """
        expecs = {}

        if progbar:
            items = Progbar(terms.items())
        else:
            items = terms.items()

        for where, G in items:
            expecs[where] = self.local_expectation(
                G,
                where,
                max_bond,
                optimize=optimize,
                normalized=normalized,
                symmetrized=symmetrized,
                flatten=flatten,
                method=method,
                rehearse=rehearse,
                **contract_compressed_opts,
            )

        if return_all or rehearse:
            return expecs
        return functools.reduce(add, expecs.values())

    compute_local_expectation_rehearse = functools.partialmethod(
        compute_local_expectation, rehearse=True)

    compute_local_expectation_tn = functools.partialmethod(
        compute_local_expectation, rehearse='tn')
