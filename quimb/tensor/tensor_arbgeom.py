import functools
from operator import add

from autoray import do, dag

from ..utils import ensure_dict
from .tensor_core import TensorNetwork


class TensorNetworkGen(TensorNetwork):
    """A tensor network which notionally has a single tensor per 'site',
    though these could be labelled arbitrarily could also be linked in an
    arbitrary geometry by bonds.
    """

    _EXTRA_PROPS = (
        '_sites',
        '_site_tag_id',
    )

    def _compatible_arbgeom(self, other):
        """Check whether ``self`` and ``other`` represent the same set of
        sites and are tagged equivalently.
        """
        return (
            isinstance(other, TensorNetworkGen) and
            all(getattr(self, e) == getattr(other, e)
                for e in TensorNetworkGen._EXTRA_PROPS)
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

    def _get_tids_from_tags(self, tags, which='all'):
        """This is the function that lets coordinates such as ``site`` be
        used for many 'tag' based functions.
        """
        tags = self.maybe_convert_coo(tags)
        return super()._get_tids_from_tags(tags, which=which)


def gauge_product_boundary_vector(
    tn,
    tags,
    which='all',
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
            tids, max_distance=select_local_distance,
            virtual=False, **select_local_opts)
        outer_inds = ltn.outer_inds()
        dtn = ltn.H | ltn

    # get all inds in the tagged region
    region_inds = set.union(
        *(set(tn.tensor_map[tid].inds) for tid in tids))

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
        **contract_around_opts)

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
        ix, = [i for i in t.inds if i in region_inds]
        _, s, VH = do('linalg.svd', t.data)
        s = s + smudge
        G = do('reshape', s**0.5, (-1, 1)) * VH
        Ginv = dag(VH) * do('reshape', s**-0.5, (1, -1))

        tid_l, tid_r = sorted(tn.ind_map[ix], key=lambda tid: tid in tids)
        tn.tensor_map[tid_l].gate_(Ginv.T, ix)
        tn.tensor_map[tid_r].gate_(G, ix)

    return tn


class TensorNetworkGenVector(TensorNetworkGen,
                             TensorNetwork):
    """A tensor network which notionally has a single tensor and outer index
    per 'site', though these could be labelled arbitrarily could also be linked
    in an arbitrary geometry by bonds.
    """

    _EXTRA_PROPS = (
        '_sites',
        '_site_tag_id',
        '_site_ind_id',
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
        gate_opts.setdefault('absorb', None)
        gate_opts.setdefault('contract', 'reduce-split')

        where_tags = tuple(map(self.site_tag, where))
        tn_where = self.select_any(where_tags)

        with tn_where.gauge_simple_temp(gauges, ungauge_inner=False):
            info = {}
            tn_where.gate_(G, where, info=info, **gate_opts)

            # inner ungauging is performed by tracking the new singular values
            ((_, ix), s), = info.items()
            if renorm:
                s = s / s[0]
            gauges[ix] = s

        return self

    def local_expectation_simple(
        self,
        G,
        where,
        normalized=False,
        max_distance=0,
        gauges=None,
        optimize='auto',
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
        where_tags = tuple(map(self.site_tag, where))
        k = self.select_local(
            where_tags, 'any',
            max_distance=max_distance,
            virtual=False,
        )

        if gauges is not None:
            # gauge the region with simple update style bond gauges
            k.gauge_simple_insert(gauges)

        b = k.H
        if normalized:
            # compute <b|k> locally
            nfact = (b | k).contract(all, optimize=optimize)
        else:
            nfact = None

        # now compute <b|G|k> locally
        k.gate_(G, where)
        ex = (b | k).contract(all, optimize=optimize)

        if nfact is not None:
            return ex / nfact
        return ex

    def compute_local_expectation_simple(
        self,
        terms,
        *,
        max_distance=0,
        normalized=False,
        gauges=None,
        optimize='auto',
        return_all=False,
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
        for where, G in terms.items():
            expecs[where] = self.local_expectation_simple(
                G, where,
                normalized=normalized,
                max_distance=max_distance,
                gauges=gauges,
                optimize=optimize,
            )

        if return_all:
            return expecs

        return functools.reduce(add, expecs.values())

    def local_expectation_exact(
        self, G, where, optimize='auto-hq', normalized=False,
    ):
        """Compute the local expectation of operator ``G`` at site(s) ``where``
        by exactly contracting the full overlap tensor network.
        """
        if not normalized:
            Gk = self.gate(G, where)
            b = self.H
            return (b | Gk).contract(all, optimize=optimize)

        k_inds = tuple(map(self.site_ind, where))
        b_inds = tuple(f'_bra{site}' for site in where)
        b = self.H.reindex_(dict(zip(k_inds, b_inds)))

        rho = (b | self).contract(all, optimize=optimize)

        rho = TensorNetwork([rho])
        nfact = rho.trace(k_inds, b_inds)
        rho.gate_inds_(G, k_inds)
        expec = rho.trace(k_inds, b_inds)
        return expec / nfact

    def compute_local_expectation_exact(
        self, terms,
        optimize='auto-hq',
        normalized=False,
        return_all=False,
    ):
        """Compute the local expectation of operator ``G`` at site(s) ``where``
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

        Returns
        -------
        expecs : float or dict[node or (node, node), float]
            If ``return_all==False``, return the summed expectation value of
            the given terms. Otherwise, return a dictionary mapping each term's
            location to the expectation value.
        """
        expecs = {}
        for where, G in terms.items():
            expecs[where] = self.local_expectation_exact(
                G, where, optimize=optimize, normalized=False
            )

        if normalized:
            nfact = (self & self.H).contract(all, optimize=optimize)
            if return_all:
                return {where: x / nfact for where, x in expecs.items()}
            return functools.reduce(add, expecs.values()) / nfact

        if return_all:
            return expecs
        return functools.reduce(add, expecs.values())
