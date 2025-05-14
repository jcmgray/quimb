"""Miscellaneous experimental functions."""

import functools

from quimb.tensor.tensor_core import (
    Tensor,
    _parse_split_opts,
    bonds,
    check_opt,
    dag,
    decomp,
    do,
    ensure_dict,
    group_inds,
    oset,
    rand_uuid,
    reshape,
    tensor_canonize_bond,
    tensor_compress_bond,
    tensor_contract,
    tensor_fuse_squeeze,
    tensor_make_single_bond,
    tensor_split,
)


def rand_reduce(self, ind, rand_fn=None, inplace=False):
    """Contract a random vector with ``ind``, removing it from this tensor.

    Parameters
    ----------
    ind : str
        The index to contract with.
    inplace : bool, optional
        Whether to perform the reduction inplace.

    Returns
    -------
    Tensor
    """
    t = self if inplace else self.copy()

    d = t.ind_size(ind)

    if rand_fn is None:
        r = do("random.normal", size=d, like=t.data)
        r /= do("linalg.norm", r)
    else:
        r = rand_fn(d)

    axis = t.inds.index(ind)
    new_inds = t.inds[:axis] + t.inds[axis + 1 :]
    t.modify(
        apply=lambda x: do("tensordot", x, r, axes=((axis,), (0,))),
        inds=new_inds,
    )
    return t


rand_reduce_ = functools.partialmethod(rand_reduce, inplace=True)


def _compress_between_local_reflect(
    self,
    tid1,
    tid2,
    method="svd",
    absorb="both",
    max_bond=None,
    cutoff=1e-10,
    cutoff_mode="rel",
    renorm=None,
    max_distance=1,
    smudge=1e-6,
    power=0.5,
    optimize=None,
    info=None,
):
    from quimb.tensor.decomp import svd

    t1, t2 = self._tids_get(tid1, tid2)

    # separate a local spanning tree into left and right components
    lset = oset([tid1])
    rset = oset([tid2])
    if max_distance > 0:
        for a, b, _ in self.get_tree_span(
            [tid1, tid2], max_distance=max_distance, inwards=False
        ):
            (lset if b in lset else rset).add(a)

    # for each tree, contract with dual to form reflected environment
    (bix,) = bonds(t1, t2)
    bix_dual = rand_uuid()

    tnl = self._select_tids(lset, virtual=False)
    tnr = self._select_tids(rset, virtual=False)

    tn_El = tnl & tnl.reindex({bix: bix_dual}).conj_()
    El = tn_El.to_dense([bix_dual], [bix], optimize=optimize)
    tn_Er = tnr & tnr.reindex({bix: bix_dual}).conj_()
    Er = tn_Er.to_dense([bix], [bix_dual], optimize=optimize)

    # decompose the environment matrices to use as a temporary gauge
    #     - this is like doing SVD(unreflected environment) but much
    #     cheaper since we only need the left singular vectors.
    sl2, Ul = do("linalg.eigh", El)
    sr2, Ur = do("linalg.eigh", Er)

    # we are free transform the singular values as inverse is just 1 / s
    sl = (sl2 + smudge * sl2[0]) ** power
    sr = (sr2 + smudge * sr2[0]) ** power
    sl = sl / do("max", sl)
    sr = sr / do("max", sr)

    # form the central reduced bond matrix & perform the actual truncation
    C = (reshape(sl, (-1, 1)) * dag(Ul)) @ (Ur * reshape(sr, (1, -1)))
    opts = _parse_split_opts(
        method, cutoff, absorb, max_bond, cutoff_mode, renorm
    )
    Lc, sc, Rc = svd(C, **opts)

    # combine the 'ungauge' with the central compressors and insert
    Cl = (Ul * reshape(1 / sl, (1, -1))) @ Lc
    Cr = Rc @ (reshape(1 / sr, (-1, 1)) * dag(Ur))

    t1.gate_(Cl.T, bix)
    t2.gate_(Cr, bix)

    if sc is not None and info is not None:
        info["singular_values"] = sc


def _compute_tensor_pair_env(
    self,
    tids,
    select_local_distance=None,
    select_local_opts=None,
    max_bond=None,
    cutoff=None,
    contract_around_distance=1,
    contract_around_opts=None,
    include=None,
    exclude=None,
):
    """Compute the local TN around ``tid``, maybe approximately."""
    # the TN we will start with
    if select_local_distance is include is exclude is None:
        # ... either the full TN
        tn_env = self.copy()

    else:
        # ... or just a local patch of the TN (with dangling bonds removed)
        select_local_opts = ensure_dict(select_local_opts)
        select_local_opts.setdefault("reduce_outer", "svd")

        tn_env = self._select_local_tids(
            tids,
            max_distance=select_local_distance,
            virtual=False,
            include=include,
            exclude=exclude,
            **select_local_opts,
        )

        # not propagated by _select_local_tids
        tn_env.exponent = self.exponent

    # possibly boundary contract in to make the env smaller
    if max_bond is not None:
        contract_around_opts = ensure_dict(contract_around_opts)
        tn_env._contract_around_tids(
            tids,
            max_bond=max_bond,
            cutoff=cutoff,
            min_distance=contract_around_distance,
            **contract_around_opts,
        )

    return tn_env


def _compress_between_full_gauge_tids(
    self,
    tid1,
    tid2,
    max_bond=None,
    cutoff=0.0,
    absorb="both",
    select_local_distance=None,
    select_local_opts=None,
    env_max_bond="max_bond",
    env_cutoff="cutoff",
    contract_around_distance=1,
    contract_around_opts=None,
    env_optimize="auto-hq",
    env_split_opts=None,
    smudge=1e-12,
    include=None,
    exclude=None,
    **compress_opts,
):
    r"""Use the full environment to try and temporarily gauge and compress.

    ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
    ┗━━━━━●━━━━━●━━━━━━━━U━s━VH━┛  (1) form and split the env
        ┊  1  ┊  2  ┊     ┈┈┈┈┈┈
        lix   bix   rix     env

    ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
    ┗Gi━G━●━━━━━●━━G━Gi━━U━s━VH━┛  (2) gauge the tensors using split
        ->       <-

    ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
    ┗━Gi━━○━━━━━○━━Gi━━━━U━s━VH━┛   (3) compress the gauged tensors
                *

    ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
    ┗━Gi━━○─────○━━Gi━━━━U━s━VH━┛   (4) ungauge the compressed tensors
        ->       <-

    ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
    ┗━━━━━●─────●━━━━━━━━U━s━VH━┛
                            ┈┈┈┈┈┈
                            env unchanged

    """
    if env_max_bond == "max_bond":
        env_max_bond = max_bond
    if env_cutoff == "cutoff":
        env_cutoff = cutoff

    ta = self.tensor_map[tid1]
    tb = self.tensor_map[tid2]

    tensor_fuse_squeeze(ta, tb)
    lix, bix, rix = group_inds(ta, tb)
    if not bix:
        return

    tn_env = self._compute_tensor_pair_env(
        (tid1, tid2),
        select_local_distance=select_local_distance,
        select_local_opts=select_local_opts,
        max_bond=env_max_bond,
        cutoff=env_cutoff,
        contract_around_distance=contract_around_distance,
        contract_around_opts=contract_around_opts,
        include=include,
        exclude=exclude,
    )

    # now we form an effective linear operator of the env
    tn_env = tn_env._select_without_tids((tid1, tid2))
    E = tn_env.aslinearoperator(rix, lix, optimize=env_optimize)

    # then we split it
    env_split_opts = ensure_dict(env_split_opts)
    env_split_opts.setdefault("cutoff", 0.0)
    U, s, VH = tensor_split(
        E,
        left_inds=lix,
        right_inds=rix,
        get="arrays",
        absorb=None,
        **env_split_opts,
    )

    # renormalize the env singular values for stability
    s = s / do("max", s) + smudge

    # create the gauges
    U = reshape(U, (-1, s.size))
    VH = reshape(VH, (s.size, -1))
    s_1_2 = s ** (1 / 2)
    s_1_2_inv = s ** (-1 / 2)
    gauge_a = reshape(s_1_2, (-1, 1)) * VH
    guage_a_inv = dag(VH) * reshape(s_1_2_inv, (1, -1))
    gauge_b = U * reshape(s_1_2, (1, -1))
    gauge_b_inv = reshape(s_1_2_inv, (-1, 1)) * dag(U)

    # gauge the raw arrays
    X_a = ta.to_dense(lix, bix)
    X_b = tb.to_dense(bix, rix)
    X_gauged_a = gauge_a @ X_a
    X_gauged_b = X_b @ gauge_b

    # perform the actual compression
    tga = Tensor(X_gauged_a, ("env_left_bond", "bond"))
    tgb = Tensor(X_gauged_b, ("bond", "env_right_bond"))
    tensor_compress_bond(
        tga, tgb, max_bond=max_bond, cutoff=cutoff, **compress_opts
    )

    # ungauge the compressed arrays
    X_compressed_a = guage_a_inv @ tga.data
    X_compressed_b = tgb.data @ gauge_b_inv

    # reshape the arrays to insert back into the original TN
    bix_size = tga.ind_size("bond")

    # create a tensor for the new compressed data, and update original ta
    tca_inds = lix + bix
    tca_shape = [bix_size if ix in bix else ta.ind_size(ix) for ix in tca_inds]
    tca = Tensor(reshape(X_compressed_a, tca_shape), tca_inds)
    tca.transpose_like_(ta)
    ta.modify(data=tca.data)

    # create a tensor for the new compressed data, and update original tb
    tcb_inds = bix + rix
    tcb_shape = [bix_size if ix in bix else tb.ind_size(ix) for ix in tcb_inds]
    tcb = Tensor(reshape(X_compressed_b, tcb_shape), tcb_inds)
    tcb.transpose_like_(tb)
    tb.modify(data=tcb.data)

    if absorb != "both":
        tensor_canonize_bond(ta, tb, absorb=absorb)


def gauge_product_boundary_vector(
    self,
    tags,
    which="all",
    max_bond=1,
    smudge=1e-6,
    canonize_distance=0,
    select_local_distance=None,
    select_local_opts=None,
    inplace=False,
    **contract_around_opts,
):
    # this is what we are going to eventually gauge
    tn = self if inplace else self.copy()
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
        s = s + smudge * s[0]
        G = reshape(s**0.5, (-1, 1)) * VH
        Ginv = dag(VH) * reshape(s**-0.5, (1, -1))

        tid_l, tid_r = sorted(tn.ind_map[ix], key=lambda tid: tid in tids)
        tn.tensor_map[tid_l].gate_(Ginv.T, ix)
        tn.tensor_map[tid_r].gate_(G, ix)

    return tn


def _compress_between_virtual_dense_tids(
    self,
    tidl,
    tidr,
    max_bond,
    cutoff,
    r,
    absorb="both",
    include=None,
    exclude=None,
    span_opts=None,
    contract_opts=None,
    optimize="auto-hq",
    **compress_opts,
):
    check_opt("absorb", absorb, ("both",))

    tl = self.tensor_map[tidl]
    tr = self.tensor_map[tidr]
    _, bix, _ = tensor_make_single_bond(tl, tr)

    # get a local region, but divide into left and right parts
    span_opts = ensure_dict(span_opts)
    span_opts["max_distance"] = r
    span_opts["exclude"] = exclude
    span_opts["include"] = include
    span_opts["inwards"] = False
    tree = self.get_tree_span([tidl, tidr], **span_opts)

    tidsl = oset([tidl])
    tidsr = oset([tidr])
    for tid_outer, tid_inner, _ in tree:
        if tid_inner in tidsl:
            tidsl.add(tid_outer)
        else:
            tidsr.add(tid_outer)

    # contract the squared region factors
    contract_opts = ensure_dict(contract_opts)
    contract_opts["optimize"] = optimize

    # compute reduced factor via XdagX for the left environment
    tnl = self._select_tids(tidsl, virtual=False)
    dl = self.inds_size(ix for ix in tnl.outer_inds() if ix != bix)
    dr = self.ind_size(bix)
    tn_XX = tnl.H.reindex({bix: "__conjbix__"}) & tnl
    XX = tn_XX.to_dense(["__conjbix__"], [bix], **contract_opts)
    Rl = decomp.squared_op_to_reduced_factor(XX, dl, dr)

    # compute reduced factor via XdagX for the right environment
    tnr = self._select_tids(tidsr, virtual=False)
    dl = self.inds_size(ix for ix in tnr.outer_inds() if ix != bix)
    dr = self.ind_size(bix)
    tn_XX = tnr.H.reindex({bix: "__conjbix__"}) & tnr
    XX = tn_XX.to_dense(["__conjbix__"], [bix], **contract_opts)
    Rr = decomp.squared_op_to_reduced_factor(XX, dl, dr)

    compress_opts["max_bond"] = max_bond
    compress_opts["cutoff"] = cutoff

    # compute the oblique projectors from the reduced factors
    Pl, Pr = decomp.compute_oblique_projectors(Rl, Rr.T, **compress_opts)

    # absorb the projectors into the tensors to perform the compression
    tl.gate_(Pl.T, bix)
    tr.gate_(Pr, bix)


def _compress_between_virtual_duotree_tids(
    self,
    tidl,
    tidr,
    max_bond,
    cutoff,
    r,
    absorb="both",
    include=None,
    exclude=None,
    span_opts=None,
    duo_exclude="tensor",
    **compress_opts,
):
    check_opt("absorb", absorb, ("both",))
    check_opt("duo_exclude", duo_exclude, ("tensor", "bond"))

    span_opts = ensure_dict(span_opts)
    span_opts["max_distance"] = r
    span_opts["include"] = include
    compress_opts["max_bond"] = max_bond
    compress_opts["cutoff"] = cutoff

    tl = self.tensor_map[tidl]
    tr = self.tensor_map[tidr]
    _, bix, _ = tensor_make_single_bond(tl, tr)

    if duo_exclude == "bond":
        # only exclude the actual shared bond from gauging
        tntree = self.copy()
        tntree._cut_between_tids(tidl, tidr, "__templ__", "__tempr__")
        treel = tntree.get_tree_span([tidl], exclude=exclude, **span_opts)
        treer = tntree.get_tree_span([tidr], exclude=exclude, **span_opts)

    else:  # duo_exclude == "tensor"
        # only exclude the other tensor from gauging
        if exclude is None:
            exclude_left = (tidr,)
            exclude_right = (tidl,)
        else:
            if not isinstance(exclude, oset):
                exclude = oset(exclude)
            exclude_left = exclude | oset([tidr])
            exclude_right = exclude | oset([tidl])

        treel = self.get_tree_span([tidl], exclude=exclude_left, **span_opts)
        treer = self.get_tree_span([tidr], exclude=exclude_right, **span_opts)

    # compute reduced factors from trees
    (Rl,) = self._compute_tree_gauges(treel, [(tidl, bix)])
    (Rr,) = self._compute_tree_gauges(treer, [(tidr, bix)])
    Pl, Pr = decomp.compute_oblique_projectors(Rl, Rr.T, **compress_opts)
    tl.gate_(Pl.T, bix)
    tr.gate_(Pr, bix)


def contract_greedily(
    self,
    max_score=0,
    max_contractions=float("inf"),
    cands=None,
    inplace=False,
):
    import heapq

    tn = self if inplace else self.copy()

    def _check_ind(ind):
        tids = tn.ind_map[ind]
        if len(tids) != 2:
            return
        ta, tb = tn._tids_get(*tids)
        sizea, sizeb = ta.size, tb.size
        oinds = tn.compute_contracted_inds(*tids)
        sizeab = tn.inds_size(oinds)
        score = sizeab - sizea - sizeb
        heapq.heappush(cands, (score, *tids))

    if cands is None:
        cands = []

    if len(cands) == 0:
        for ind in tn.ind_map:
            _check_ind(ind)

    c = 0
    while cands:
        # get the 'best' contraction
        score, tida, tidb = heapq.heappop(cands)

        if not (tida in tn.tensor_map and tidb in tn.tensor_map):
            # tensor a or b already contracted
            continue
        if score > max_score:
            # contraction isn't good enough, all others are also worse
            break

        # perform the contraction
        ta = tn.pop_tensor(tida)
        tb = tn.pop_tensor(tidb)
        tab = tensor_contract(ta, tb, preserve_tensor=True)
        tn |= tab

        # check how many contractions we've done
        c += 1
        if c >= max_contractions:
            break

        # add new potential candidates
        for ind in tab.inds:
            _check_ind(ind)

    return tn


contract_greedily_ = functools.partialmethod(contract_greedily, inplace=True)
