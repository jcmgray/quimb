from ...utils import check_opt
from ..tensor_arbgeom import TensorNetworkGen, TensorNetworkGenVector
from .fermion_core import FermionTensorNetwork, FermionTensor

class FermionTensorNetworkGen(FermionTensorNetwork,
                              TensorNetworkGen):


    _EXTRA_PROPS = (
        '_sites',
        '_site_tag_id',
    )

    def __and__(self, other):
        new = super().__and__(other)
        if self._compatible_arbgeom(other):
            new.view_as_(FermionTensorNetworkGen, like=self)
        return new

    def __or__(self, other):
        new = super().__or__(other)
        if self._compatible_arbgeom(other):
            new.view_as_(FermionTensorNetworkGen, like=self)
        return new


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
    pass


class FermionTensorNetworkGenVector(
        FermionTensorNetworkGen,
        TensorNetworkGenVector):
    """A tensor network which notionally has a single tensor and outer index
    per 'site', though these could be labelled arbitrarily could also be linked
    in an arbitrary geometry by bonds.
    """

    _EXTRA_PROPS = (
        '_sites',
        '_site_tag_id',
        '_site_ind_id',
    )

    def gate_simple_(self, G, where, gauges, renorm=True, **gate_opts):
        gate_opts.setdefault('absorb', None)
        gate_opts.setdefault('contract', 'reduce-split')

        where_tags = tuple(map(self.site_tag, where))
        tn_where = self.select_any(where_tags)

        with tn_where.gauge_simple_temp(gauges, ungauge_inner=False):
            info = {}
            tn_where.gate_(G, where, info=info, **gate_opts)
            # inner ungauging is performed by tracking the new singular values
            ((_, ixs), s), = info.items()
            if renorm:
                s = s / s.norm()
            gauges.pop((ixs[1], ixs[0]), None)
            gauges[ixs] = s
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
        # select a local neighborhood of tensors
        where_tags = tuple(map(self.site_tag, where))
        tn = self.copy()

        k = tn.select_local(
            where_tags, 'any',
            max_distance=max_distance,
            virtual=True,
        )

        if gauges is not None:
            # gauge the region with simple update style bond gauges
            k.gauge_simple_insert(gauges)

        # reconstructing the 'simple' network even though they
        # might not be continugously ordered in original network
        k = k.copy(force=True)

        b = k.H
        if normalized:
            # compute <b|k> locally
            nfact = (k & b).contract(all, optimize=optimize)
        else:
            nfact = None

        # now compute <b|G|k> locally
        k.gate_(G, where)
        ex = (k | b).contract(all, optimize=optimize)

        if nfact is not None:
            return ex / nfact
        return ex

    def local_expectation_exact(
        self, G, where, optimize='auto-hq', normalized=False,
    ):
        """Compute the local expectation of operator ``G`` at site(s) ``where``
        by exactly contracting the full overlap tensor network.
        """
        if not normalized:
            Gk = self.gate(G, where)
            b = self.H
            return (Gk | b).contract(all, optimize=optimize)

        k_inds = tuple(map(self.site_ind, where))
        b_inds = tuple(f'_bra{site}' for site in where)
        b = self.H.reindex_(dict(zip(k_inds, b_inds)))

        Gop = FermionTensor(G.copy(), inds=b_inds+k_inds)
        new_G = b.fermion_space.move_past(Gop).data

        rho = (self & b).contract(all, optimize=optimize)
        # make sure the bra indices lie ahead of ket indices
        rho.transpose_(*b_inds, *k_inds)
        rho = FermionTensorNetwork([rho])
        nfact = rho.trace(k_inds, b_inds)
        rho.gate_inds_(new_G, k_inds)
        expec = rho.trace(k_inds, b_inds)
        return expec / nfact

    def local_expectation(
        self,
        G,
        where,
        max_bond,
        optimize,
        normalized=True,
        flatten=True,
        method='rho',
        rehearse=False,
        symmetrized="auto",
        **contract_compressed_opts,
    ):
        check_opt('method', method, ('rho', 'rho-reduced'))
        reduce = method == 'rho-reduced'

        # form the partial trace
        k_inds = tuple(map(self.site_ind, where))

        if reduce:
            k = self.copy()
            k.reduce_inds_onto_bond(*k_inds, tags='__BOND__', drop_tags=True)
        else:
            k = self

        b_inds = tuple(map("_bra{}".format, where))
        b = k.H.reindex_(dict(zip(k_inds, b_inds)))

        Gop = FermionTensor(G.copy(), inds=b_inds+k_inds)
        new_G = b.fermion_space.move_past(Gop).data

        tn = k & b

        output_inds = k_inds + b_inds

        if flatten:
            for site in self.sites:
                if site in where:
                    continue
                tn ^= site
            if reduce:
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

        t_rho.transpose_(*b_inds, *k_inds)
        t_rho = FermionTensorNetwork([t_rho])
        if normalized:
            nfact = t_rho.trace(k_inds, b_inds)
        else:
            nfact = 1

        t_rho.gate_inds_(new_G, k_inds)
        expec = t_rho.trace(k_inds, b_inds) / nfact
        return expec
