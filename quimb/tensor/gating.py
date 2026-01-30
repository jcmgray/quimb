"""Functionality relating to *gating* tensor networks, i.e. applying a 'gate' -
typically a local operator described by a raw array - to one or more indices in
the network, maintaining the outer index structure by appropriate reindexing.
"""

import autoray as ar

from .tensor_core import (
    PArray,
    PTensor,
    Tensor,
    TensorNetwork,
    check_opt,
    group_inds,
    isblocksparse,
    prod,
    rand_uuid,
    tags_to_oset,
    tensor_contract,
)


def _tensor_network_gate_inds_basic(
    tn: TensorNetwork,
    G,
    inds,
    ng,
    tags,
    contract,
    isparam,
    info,
    **compress_opts,
):
    tags = tags_to_oset(tags)

    if (ng == 1) and contract:
        # single site gate, eagerly applied so contract in directly ->
        # useful short circuit  as it maintains the index structure exactly
        (ix,) = inds
        (t,) = tn._inds_get(ix)
        t.gate_(G, ix)
        t.add_tag(tags)
        return tn

    # new indices to join old physical sites to new gate
    bnds = [rand_uuid() for _ in range(ng)]
    reindex_map = dict(zip(inds, bnds))

    # tensor representing the gate
    if isparam:
        TG = PTensor.from_parray(
            G, inds=(*inds, *bnds), tags=tags, left_inds=bnds
        )
    else:
        TG = Tensor(G, inds=(*inds, *bnds), tags=tags, left_inds=bnds)

    if contract is False:
        # we just attach gate to the network, no contraction:
        #
        #       │   │      <- site_ix
        #       GGGGG
        #       │╱  │╱     <- bnds
        #     ──●───●──
        #      ╱   ╱
        #
        tn.reindex_(reindex_map)
        tn |= TG
        return tn

    tids = tn._get_tids_from_inds(inds, "any")

    if (contract is True) or (len(tids) == 1):
        # everything is contracted, no need to split anything:
        #
        #       │╱│╱
        #     ──GGG──
        #      ╱ ╱
        #
        tn.reindex_(reindex_map)

        # get the sites that used to have the physical indices
        site_tids = tn._get_tids_from_inds(bnds, which="any")

        # pop the sites, contract, then re-add
        pts = [tn.pop_tensor(tid) for tid in site_tids]
        tn |= tensor_contract(*pts, TG)

        return tn

    # else we have a two index, two tensor, gate with some splitting:
    return _tensor_network_gate_inds_eager_split(
        tn,
        inds,
        contract,
        reindex_map,
        TG,
        info,
        compress_opts,
    )


def _tensor_network_gate_inds_eager_split(
    tn,
    inds,
    contract,
    reindex_map,
    TG,
    info,
    compress_opts,
):
    # get the two tensors and their current shared indices etc.
    ixl, ixr = inds
    tl, tr = tn._inds_get(ixl, ixr)

    # TODO: handle possible creation or fusing of bond here?
    bnds_l, (bix,), bnds_r = group_inds(tl, tr)

    # NOTE: disabled for block sparse, where reduced split is always important
    # for keeping charge distributions across tensors stable
    if ((len(bnds_l) <= 2) and (len(bnds_r) <= 2)) and not isblocksparse(
        TG.data
    ):
        # reduce split is likely redundant (i.e. contracting pair
        # and splitting just as cheap as performing QR reductions)
        contract = "split"

    if contract == "split":
        # contract everything and then split back apart:
        #
        #       │╱  │╱         │╱  │╱
        #     ──GGGGG──  ->  ──G~~~G──
        #      ╱   ╱          ╱   ╱
        #

        # contract with new gate tensor
        tlGr = tensor_contract(
            tl.reindex(reindex_map), tr.reindex(reindex_map), TG
        )

        # decompose back into two tensors
        tln, *maybe_svals, trn = tlGr.split(
            left_inds=bnds_l,
            right_inds=bnds_r,
            bond_ind=bix,
            get="tensors",
            **compress_opts,
        )

    if contract == "reduce-split":
        # move physical inds on reduced tensors
        #
        #       │   │             │ │
        #       GGGGG             GGG
        #       │╱  │╱   ->     ╱ │ │   ╱
        #     ──●───●──      ──▶──●─●──◀──
        #      ╱   ╱          ╱       ╱
        #
        tmp_bix_l = rand_uuid()
        tl_Q, tl_R = tl.split(
            left_inds=None,
            right_inds=[bix, ixl],
            method="qr",
            bond_ind=tmp_bix_l,
        )
        tmp_bix_r = rand_uuid()
        tr_L, tr_Q = tr.split(
            left_inds=[bix, ixr],
            right_inds=None,
            method="lq",
            bond_ind=tmp_bix_r,
        )

        # contract reduced tensors with gate tensor
        #
        #          │ │
        #          GGG                │ │
        #        ╱ │ │   ╱    ->    ╱ │ │   ╱
        #     ──▶──●─●──◀──      ──▶──LGR──◀──
        #      ╱       ╱          ╱       ╱
        #
        tlGr = tensor_contract(
            tl_R.reindex(reindex_map), tr_L.reindex(reindex_map), TG
        )

        # split to find new reduced factors
        #
        #          │ │                │ │
        #        ╱ │ │   ╱    ->    ╱ │ │   ╱
        #     ──▶──LGR──◀──      ──▶──L~R──◀──
        #      ╱       ╱          ╱       ╱
        #
        tl_R, *maybe_svals, tr_L = tlGr.split(
            left_inds=[tmp_bix_l, ixl],
            right_inds=[tmp_bix_r, ixr],
            bond_ind=bix,
            get="tensors",
            **compress_opts,
        )

        # absorb reduced factors back into site tensors
        #
        #         │   │            │    │
        #        ╱│   │  ╱         │╱   │╱
        #     ──▶─L~~~R─◀──  ->  ──●~~~~●──
        #      ╱       ╱          ╱    ╱
        #
        tln = tl_Q @ tl_R
        trn = tr_L @ tr_Q

    # if singular values are returned (``absorb=None``) check if we should
    #     return them further via ``info``, e.g. for ``SimpleUpdate`
    if maybe_svals and (info is not None):
        s = next(iter(maybe_svals)).data
        info["singular_values", bix] = s

    # update original tensors
    tl.modify(data=tln.transpose_like_(tl).data)
    tl.add_tag(TG.tags)
    tr.modify(data=trn.transpose_like_(tr).data)
    tr.add_tag(TG.tags)


def _tensor_network_gate_inds_lazy_split(
    tn: TensorNetwork,
    G,
    inds,
    ng,
    tags,
    contract,
    **compress_opts,
):
    lix = [f"l{i}" for i in range(ng)]
    rix = [f"r{i}" for i in range(ng)]

    TG = Tensor(data=G, inds=lix + rix, tags=tags, left_inds=rix)

    # check if we should split multi-site gates (which may result in an easier
    #     tensor network to contract if we use compression)
    if contract in ("split-gate", "auto-split-gate"):
        #  | |       | |
        #  GGG  -->  G~G
        #  | |       | |
        tnG_spat = TG.split(("l0", "r0"), bond_ind="b", **compress_opts)

    # sometimes it is worth performing the decomposition *across* the gate,
    #     effectively introducing a SWAP
    if contract in ("swap-split-gate", "auto-split-gate"):
        #            \ /
        #  | |        X
        #  GGG  -->  / \
        #  | |       G~G
        #            | |
        tnG_swap = TG.split(("l0", "r1"), bond_ind="b", **compress_opts)

    # like 'split-gate' but check the rank for swapped indices also, and if no
    #     rank reduction, simply don't swap
    if contract == "auto-split-gate":
        #            | |      \ /
        #  | |       | |       X           | |
        #  GGG  -->  G~G  or  / \   or ... GGG
        #  | |       | |      G~G          | |
        #            | |      | |
        spat_rank = tnG_spat.ind_size("b")
        swap_rank = tnG_swap.ind_size("b")

        if swap_rank < spat_rank:
            contract = "swap-split-gate"
        elif spat_rank < prod(G.shape[:ng]):
            contract = "split-gate"
        else:
            # else no rank reduction available - leave as ``contract=False``.
            contract = False

    if contract == "swap-split-gate":
        tnG = tnG_swap
    elif contract == "split-gate":
        tnG = tnG_spat
    else:
        tnG = TG

    return tn.gate_inds_with_tn_(inds, tnG, rix, lix)


_BASIC_GATE_CONTRACT = {
    False,
    True,
    "split",
    "reduce-split",
}
_SPLIT_GATE_CONTRACT = {
    "auto-split-gate",
    "split-gate",
    "swap-split-gate",
}
_VALID_GATE_CONTRACT = _BASIC_GATE_CONTRACT | _SPLIT_GATE_CONTRACT


def maybe_factor_gate(
    G,
    inds,
    xp=None,
    tn=None,
):
    """Possibly reshape gate ``G``, if it has been supplied as a matrix with
    'fused' dimensions, into a tensor with separate physical dimensions.
    """
    if xp is None:
        xp = ar.get_namespace(G)

    ng = len(inds)
    ndimG = xp.ndim(G)

    if ndimG != 2 * ng:
        # gate supplied as matrix, factorize it

        if isblocksparse(G) or (tn is None):
            # can't simply infer shape -> guess all same size
            # the gate should be supplied as a tensor to avoid this
            dg = round(xp.size(G) ** (1 / (2 * ng)))
            gate_shape = (dg,) * (2 * ng)
            G = xp.reshape(G, gate_shape)
        else:
            # can infer required shape from physical dimensions
            dims = tuple(tn.ind_size(ix) for ix in inds)
            G = xp.reshape(G, dims * 2)

    return G


def tensor_network_gate_inds(
    self: TensorNetwork,
    G,
    inds,
    contract=False,
    tags=None,
    info=None,
    inplace=False,
    **compress_opts,
):
    r"""Apply a local 'gate' ``G``, given as a raw array, to a group of
    indices ``inds``, as if applying ``G @ x``. The indices are propagated to
    the outside to maintain the original index structure of the tensor network.

    Parameters
    ----------
    G : array_ike
        The gate array to apply, should match or be factorable into the
        shape ``(*phys_dims, *phys_dims)``.
    inds : str or sequence or str,
        The index or indices to apply the gate to.
    contract : {False, True, 'split', 'reduce-split', 'split-gate',
                'swap-split-gate', 'auto-split-gate'}, optional
        How to apply the gate:

        - ``False``: gate is added to network lazily and nothing is contracted,
          tensor network structure is thus not maintained.
        - ``True``: gate is contracted eagerly with all tensors involved,
          tensor network structure is thus only maintained if gate acts on a
          single site only.
        - ``'split'``: contract all involved tensors then split the result back
          into two.
        - ``'reduce-split'``: factor the two physical indices into 'R-factors'
          using QR decompositions on the original site tensors, then contract
          the gate, split it and reabsorb each side. Cheaper than ``'split'``
          when the tensors on either side have at least 3 bonds.
        - ``'split-gate'``: lazily add the gate as with ``False``, but split
          the gate tensor spatially.
        - ``'swap-split-gate'``: lazily add the gate as with ``False``, but
          split the gate as if an extra SWAP has been applied.
        - ``'auto-split-gate'``: lazily add the gate as with ``False``, but
          maybe apply one of the above options depending on whether they result
          in a rank reduction.

        The named methods are relevant for two site gates only, for single site
        gates they use the ``contract=True`` option which also maintains the
        structure of the TN. See below for a pictorial description of each
        method.
    tags : str or sequence of str, optional
        Tags to add to the new gate tensor.
    info : None or dict, optional
        Used to store extra optional information such as the singular values if
        not absorbed.
    inplace : bool, optional
        Whether to perform the gate operation inplace on the tensor network or
        not.
    compress_opts
        Supplied to :func:`~quimb.tensor.tensor_core.tensor_split` for any
        ``contract`` methods that involve splitting. Ignored otherwise.

    Returns
    -------
    G_tn : TensorNetwork

    Notes
    -----

    The ``contract`` options look like the following (for two site gates).

    ``contract=False``::

          .   .  <- inds
          │   │
          GGGGG
          │╱  │╱
        ──●───●──
         ╱   ╱

    ``contract=True``::

          │╱  │╱
        ──GGGGG──
         ╱   ╱

    ``contract='split'``::

          │╱  │╱          │╱  │╱
        ──GGGGG──  ==>  ──G┄┄┄G──
         ╱   ╱           ╱   ╱
          <SVD>

    ``contract='reduce-split'``::

          │   │             │ │
          GGGGG             GGG               │ │
          │╱  │╱   ==>     ╱│ │  ╱   ==>     ╱│ │  ╱          │╱  │╱
        ──●───●──       ──▶─●─●─◀──       ──▶─GGG─◀──  ==>  ──G┄┄┄G──
         ╱   ╱           ╱     ╱           ╱     ╱           ╱   ╱
        <QR> <LQ>                            <SVD>

    For one site gates when one of the above 'split' methods is supplied
    ``contract=True`` is assumed.

    ``contract='split-gate'``::

          │   │ <SVD>
          G~~~G
          │╱  │╱
        ──●───●──
         ╱   ╱

    ``contract='swap-split-gate'``::

           ╲ ╱
            ╳
           ╱ ╲ <SVD>
          G~~~G
          │╱  │╱
        ──●───●──
         ╱   ╱

    ``contract='auto-split-gate'`` chooses between the above two and ``False``,
    depending on whether either results in a lower rank.

    """
    check_opt("contract", contract, _VALID_GATE_CONTRACT)

    tn = self if inplace else self.copy()

    G = maybe_factor_gate(G, inds, tn=tn)
    gatesplitting = contract in _SPLIT_GATE_CONTRACT
    ng = len(inds)

    if (
        # if single ind, gate splitting methods are same as contract=False
        (gatesplitting and (ng == 1))
        or
        # or for 3+ sites, treat auto as no splitting
        ((contract == "auto-split-gate") and (ng > 2))
    ):
        gatesplitting = False
        contract = False

    isparam = isinstance(G, PArray)
    if isparam:
        if contract == "auto-split-gate":
            # simply don't split
            gatesplitting = False
            contract = False
        elif contract and ng > 1:
            raise ValueError(
                "For a parametrized gate acting on more than one site "
                "``contract`` must be false to preserve the array shape."
            )

    if gatesplitting:
        # possible splitting of gate itself involved
        if ng > 2:
            raise ValueError(f"`contract='{contract}'` invalid for >2 sites.")

        _tensor_network_gate_inds_lazy_split(
            tn, G, inds, ng, tags, contract, **compress_opts
        )
    else:
        # no splitting of the *gate on its own* involved
        _tensor_network_gate_inds_basic(
            tn, G, inds, ng, tags, contract, isparam, info, **compress_opts
        )

    return tn


def _tensor_network_gate_sandwich_inds_eager_split(
    tn,
    G,
    Gconj,
    inds_upper,
    inds_lower,
    contract,
    tags,
    info,
    **compress_opts,
):
    new_upper = [rand_uuid() for _ in inds_upper]
    new_lower = [rand_uuid() for _ in inds_lower]
    reindex_map = dict(zip(inds_upper, new_upper))
    reindex_map.update(dict(zip(inds_lower, new_lower)))

    # wrap the gates as tensors
    TGu = Tensor(G, inds=(*inds_upper, *new_upper))
    TGl = Tensor(Gconj, inds=(*inds_lower, *new_lower))

    # get the two tensors to be gated
    kixl, kixr = inds_upper
    bixl, bixr = inds_lower
    tl, tr = tn._inds_get(kixl, bixl, kixr, bixr)
    bnds_l, (bix,), bnds_r = group_inds(tl, tr)

    # NOTE: disabled for block sparse, where reduced split is always important
    # for keeping charge distributions across tensors stable
    if ((len(bnds_l) <= 3) and (len(bnds_r) <= 3)) and not isblocksparse(G):
        # reduce split is likely redundant (i.e. contracting pair
        # and splitting just as cheap as performing QR reductions)
        contract = "split"

    if contract == "split":
        # contract everything and then split back apart:
        #
        #       │╱  │╱         │╱  │╱
        #     ──GGGGG──  ->  ──●~~~●──
        #      ╱│  ╱│         ╱│  ╱│
        #

        # contract with new gate tensor
        tlGr = tensor_contract(
            tl.reindex(reindex_map),
            tr.reindex(reindex_map),
            TGu,
            TGl,
        )

        # decompose back into two tensors
        tln, *maybe_svals, trn = tlGr.split(
            left_inds=bnds_l,
            right_inds=bnds_r,
            bond_ind=bix,
            get="tensors",
            **compress_opts,
        )

    if contract == "reduce-split":
        # move physical inds on reduced tensors:
        #
        #       │   │             │ │
        #       GGGGG             GGG
        #       │╱  │╱          ╱ │ │   ╱
        #     ──●───●──  ->  ──▶──●─●──◀──
        #      ╱│  ╱│         ╱   │ │ ╱
        #       G†G†G             G†G
        #       │   │             │ │
        #
        tmp_bix_l = rand_uuid()
        tl_Q, tl_R = tl.split(
            left_inds=None,
            right_inds=[bix, kixl, bixl],
            method="qr",
            bond_ind=tmp_bix_l,
        )
        tmp_bix_r = rand_uuid()
        tr_L, tr_Q = tr.split(
            left_inds=[bix, kixr, bixr],
            right_inds=None,
            method="lq",
            bond_ind=tmp_bix_r,
        )

        # contract reduced tensors with gate tensors:
        #
        #          │ │
        #          GGG                │ │
        #        ╱ │ │   ╱          ╱ │ │   ╱
        #     ──▶━━●─●━━◀──  ->  ──▶━━LGR━━◀──
        #      ╱   │ │  ╱         ╱   │ │ ╱
        #          G†G                │ │
        #          │ │
        #
        tlGr = tensor_contract(
            tl_R.reindex(reindex_map), tr_L.reindex(reindex_map), TGu, TGl
        )

        # split to find new reduced factors:
        #
        #          │ │               │   │
        #        ╱ │ │   ╱          ╱│   │  ╱
        #     ──▶━━LGR━━◀──  ->  ──▶━L~~~R━◀──
        #      ╱   │ │ ╱          ╱  │   │╱
        #          │ │               │   │
        #
        tl_R, *maybe_svals, tr_L = tlGr.split(
            left_inds=[tmp_bix_l, kixl, bixl],
            right_inds=[tmp_bix_r, kixr, bixr],
            bond_ind=bix,
            get="tensors",
            **compress_opts,
        )

        # absorb reduced factors back into site tensors:
        #
        #          │ │             │   │
        #        ╱ │ │   ╱         │╱  │╱
        #     ──▶━━L~R━━◀──  ->  ──●~~~●──
        #      ╱   │ │ ╱          ╱│  ╱│
        #          │ │             │   │
        #
        tln = tl_Q @ tl_R
        trn = tr_L @ tr_Q

    # if singular values are returned (``absorb=None``) check if we should
    #     return them further via ``info``, e.g. for ``SimpleUpdate`
    if maybe_svals and (info is not None):
        s = next(iter(maybe_svals)).data
        info["singular_values", bix] = s

    # update original tensors
    tl.modify(data=tln.transpose_like_(tl).data)
    tl.add_tag(tags)
    tr.modify(data=trn.transpose_like_(tr).data)
    tr.add_tag(tags)


def tensor_network_gate_sandwich_inds(
    self: TensorNetwork,
    G,
    inds_upper,
    inds_lower,
    contract=False,
    tags=None,
    tags_upper=None,
    tags_lower=None,
    info=None,
    inplace=False,
    **compress_opts,
):
    """Apply a 'sandwich' gate, ``G`` to two groups of indices in a tensor
    network, i.e. applying ``G @ x @ G†`` to the indices given respectively by
    ``inds_upper`` and ``inds_lower``, then propagating them to the outside, to
    maintain the original index structure.
    """

    tn = self if inplace else self.copy()

    xp = ar.get_namespace(G)

    G = maybe_factor_gate(G, inds_upper, xp, tn)
    Gconj = xp.conj(G)

    tags = tags_to_oset(tags)
    tags_upper = tags | tags_to_oset(tags_upper)
    tags_lower = tags | tags_to_oset(tags_lower)

    if len(inds_upper) == 2 and contract in ("split", "reduce-split"):
        # need to combine both gates before splitting
        _tensor_network_gate_sandwich_inds_eager_split(
            tn,
            G=G,
            Gconj=Gconj,
            inds_upper=inds_upper,
            inds_lower=inds_lower,
            contract=contract,
            tags=tags | tags_upper | tags_lower,
            info=info,
            **compress_opts,
        )
    else:
        for G_ul, inds_ul, tags_ul in (
            (G, inds_upper, tags_upper),
            (Gconj, inds_lower, tags_lower),
        ):
            tensor_network_gate_inds(
                tn,
                G=G_ul,
                inds=inds_ul,
                contract=contract,
                tags=tags_ul,
                info=info,
                inplace=True,
                **compress_opts,
            )

    return tn
