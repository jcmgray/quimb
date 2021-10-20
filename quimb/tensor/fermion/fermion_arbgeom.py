from typing import _SpecialForm
from ...utils import check_opt
from ..tensor_arbgeom import TensorNetworkGen, TensorNetworkGenVector
from .fermion_core import FermionTensorNetwork, FermionTensor
import functools

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
        normalized=True,
        max_distance=0,
        fillin=False,
        gauges=None,
        optimize="auto",
        max_bond=None,
        rehearse=False,
        **contract_opts,
    ):
        # select a local neighborhood of tensors
        site_tags = tuple(map(self.site_tag, where))
        k = self.select_local(site_tags, "any", max_distance=max_distance,
                              fillin=fillin, virtual=False)

        if gauges is not None:
            # gauge the region with simple update style bond gauges
            k.gauge_simple_insert(gauges)
        k = k.copy(force=True)

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
        b = self.H.reindex_(dict(zip(k_inds, b_inds)))
        Gop = FermionTensor(G.copy(), inds=b_inds+k_inds)
        new_G = b.fermion_space.move_past(Gop).data
        tn = self & b
        output_inds = b_inds + k_inds
        if rehearse:
            if rehearse == 'tn':
                return tn
            if rehearse == 'tree':
                return tn.contraction_tree(
                    optimize, output_inds=output_inds)
            if rehearse:
                return tn.contraction_info(
                    optimize, output_inds=output_inds)

        rho = tn.contract(all, optimize=optimize, output_inds=output_inds)
        # make sure the bra indices lie ahead of ket indices
        rho = FermionTensorNetwork([rho])
        if normalized:
            nfact = rho.trace(k_inds, b_inds)
        else:
            nfact = 1
        rho.gate_inds_(new_G, k_inds)
        expec = rho.trace(k_inds, b_inds) / nfact
        return expec

    def partial_trace(
        self,
        keep,
        max_bond,
        optimize,
        flatten=True,
        reduce=False,
        normalized=True,
        symmetrized=False,
        rehearse=False,
        **contract_compressed_opts,
    ):
        if symmetrized:
            raise NotImplementedError

        # form the partial trace
        k_inds = tuple(map(self.site_ind, keep))

        k = self.copy()
        if reduce:
            k.reduce_inds_onto_bond(*k_inds, tags='__BOND__', drop_tags=True)

        b_inds = tuple(map("_bra{}".format, keep))
        b = k.H.reindex_(dict(zip(k_inds, b_inds)))

        tn = k | b
        output_inds = b_inds + k_inds

        if flatten:
            for site in self.sites:
                if (site not in keep) or (flatten == 'all'):
                    # check if site exists still to permit e.g. local methods
                    # to use this same logic
                    tag = tn.site_tag(site)
                    if tag in tn.tag_map:
                        tn ^= tag
            if reduce and (flatten == 'all'):
                tn ^= '__BOND__'

        if rehearse:
            if rehearse == 'tn':
                return tn
            if rehearse == 'tree':
                return tn.contraction_tree(optimize, output_inds=output_inds)
            if rehearse:
                return tn.contraction_info(optimize, output_inds=output_inds)

        rho = tn.contract_compressed(
            optimize,
            max_bond=max_bond,
            output_inds=output_inds,
            **contract_compressed_opts,
        )

        rho.transpose_(*output_inds)
        t_rho = FermionTensorNetwork([rho], virtual=True)
        if normalized:
            nfact = t_rho.trace(k_inds, b_inds)
            rho.modify(data=rho.data/nfact)
        return t_rho

    def local_expectation(
        self,
        G,
        where,
        max_bond,
        optimize,
        method='rho',
        flatten=True,
        normalized=True,
        symmetrized=False,
        rehearse=False,
        **contract_compressed_opts,
    ):

        check_opt('method', method, ('rho', 'rho-reduced'))
        reduce = (method == 'rho-reduced')

        k_inds = tuple(map(self.site_ind, where))
        b_inds = tuple(map("_bra{}".format, where))
        b = self.H.reindex_(dict(zip(k_inds, b_inds)))
        Gop = FermionTensor(G.copy(), inds=b_inds+k_inds)
        new_G = b.fermion_space.move_past(Gop).data

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

        rho.gate_inds_(new_G, k_inds)
        expec = rho.trace(k_inds, b_inds)
        return expec

    compute_local_expectation = functools.partialmethod(
        TensorNetworkGenVector.compute_local_expectation, symmetrized=False)

    compute_local_expectation_rehearse = functools.partialmethod(
        compute_local_expectation, rehearse=True)

    compute_local_expectation_tn = functools.partialmethod(
        compute_local_expectation, rehearse='tn')
