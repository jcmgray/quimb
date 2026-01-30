from quimb.tensor.tnag.core import *


def virtual_bond_operator(
    self,
    ind,
    operators=None,
    flatten=False,
    rehearse=False,
    **contract_opts,
):
    operators = {} if operators is None else dict(operators)

    ket = self.copy()
    tida, tidb = sorted(ket.ind_map[ind])

    # cut the bond open and create dual
    ixkl, ixkr, ixbl, ixbr = (rand_uuid() for _ in range(4))
    ket._cut_between_tids(tida, tidb, ixkl, ixkr)
    bra = ket.H.reindex_({ixkl: ixbl, ixkr: ixbr})

    # optionally apply list of operators to ket only
    for where, G in operators.items():
        ket.gate_(G, where)

    # form partial trace
    overlap = ket | bra
    if flatten:
        for site in overlap.gen_sites_present():
            overlap ^= site

    output_inds = (ixbl, ixbr, ixkl, ixkr)
    if rehearse == "tree":
        return overlap.contraction_tree(
            optimize=contract_opts.get("optimize", None),
            output_inds=output_inds,
        )

    # contract to matrix that maps cut bra bond to cut ket bond
    return overlap.contract_compressed(
        output_inds=output_inds,
        **contract_opts,
    ).to_dense([ixbl, ixbr], [ixkl, ixkr])


def virtual_site_operator(
    self,
    site,
    operators=None,
    flatten=False,
    max_bond=None,
    rehearse=False,
    **contract_opts,
):
    operators = {} if operators is None else dict(operators)

    ket = self.copy()
    (tid,) = ket._get_tids_from_tags(site)
    t = ket.pop_tensor(tid)

    # once the tensor is removed, the physical site is gone - re-add later
    phys_ind = self.site_ind(site)
    k_ix = tuple(ix for ix in t.inds if ix != phys_ind)
    reindex = {ix: rand_uuid() for ix in k_ix}
    b_ix = tuple(reindex.values())
    bra = ket.conj().reindex_(reindex)

    if phys_ind not in t.inds:
        dangling = None
    elif site in operators:
        # missing physical bond is the operator
        dangling = operators.pop(site)
    else:
        # missing physical bond is just a identity
        d = t.ind_size(phys_ind)
        dangling = do("eye", d, dtype=t.data.dtype, like=t.data)

    # apply other operators to ket
    for where, G in operators.items():
        t = ket[where]
        t.gate_(G, ket.site_ind(where))

    # form partial trace
    overlap = ket | bra
    if flatten:
        for site in overlap.gen_sites_present():
            overlap ^= site
        overlap.fuse_multibonds_()

    output_inds = (*b_ix, *k_ix)
    if rehearse == "tree":
        return overlap.contraction_tree(
            optimize=contract_opts.get("optimize", None),
            output_inds=output_inds,
        )

    X = overlap.contract_compressed(
        max_bond=max_bond,
        output_inds=output_inds,
        **contract_opts,
    ).to_dense(b_ix, k_ix)

    if dangling is not None:
        X = do("kron", X, dangling)

    return X


def product_expectation_exact(
    self,
    Gs,
    where,
    optimize="auto-hq",
    normalized=True,
    rehearse=False,
    **contract_opts,
):
    k = self.copy()
    b = k.conj()

    for site, G in zip(where, Gs):
        k.gate_(G, (site,), contract=True)

    tn = b & k

    if rehearse:
        return handle_rehearse(rehearse, tn, optimize, output_inds=())

    expec = tn.contract(
        optimize=optimize,
        **contract_opts,
    )

    if normalized:
        k = self.copy()
        tn = b & k
        nfact = tn.contract(
            optimize=optimize,
            **contract_opts,
        )
        expec = expec / nfact

    return expec


def product_expectation_cluster(
    self,
    Gs,
    where,
    normalized=True,
    max_distance=0,
    fillin=False,
    gauges=None,
    optimize="auto",
    rehearse=False,
    **contract_opts,
):
    # select a local neighborhood of tensors
    tids = self._get_tids_from_tags(tuple(map(self.site_tag, where)), "any")

    k = self._select_local_tids(
        tids,
        max_distance=max_distance,
        fillin=fillin,
        virtual=False,
    )

    if gauges is not None:
        # gauge the region with simple update style bond gauges
        k.gauge_simple_insert(gauges)

    return k.product_expectation_exact(
        Gs=Gs,
        where=where,
        optimize=optimize,
        normalized=normalized,
        rehearse=rehearse,
        **contract_opts,
    )
