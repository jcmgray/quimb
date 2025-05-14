# Tensor


def cp_decompose(
    self,
    rank,
    bond_ind=None,
    **kwargs,
):
    """Construct a CANDECOMP/PARAFAC decomposition of this tensor via
    fitting.
    """
    t0 = self.copy()

    # adding some noise to the initial tensor (not just factors) makes this
    # much more stable when very symmetric XXX: use fixed seed?
    noise_scale = self.largest_element() * 0.01
    noise = noise_scale * do("random.normal", size=self.shape, like=t0.data)
    t0.modify(data=t0.data + noise)

    if bond_ind is None:
        bond_ind = rand_uuid()

    # construct out initial guess
    tn = TensorNetwork()
    for ix in t0.inds:
        tn |= t0.split(
            left_inds=[ix],
            bond_ind=bond_ind,
            max_bond=rank,
            cutoff=0.0,
            get="tensors",
        )[0]

    # fit and return the new TN
    return tn.fit_(self, **kwargs)


# TensorNetwork


def cp_simplify(
    self,
    atol=1e-12,
    equalize_norms=False,
    cache=None,
    inplace=False,
    **cp_opts,
):
    tn = self if inplace else self.copy()

    # we don't want to repeatedly check the CP decompositions of the
    #     same tensor as we cycle through simplification methods
    if cache is None:
        cache = set()

    for tid, t in tuple(tn.tensor_map.items()):
        # id's are reused when objects go out of scope -> use tid as well
        cache_key = ("cp", tid, id(t.data))
        if cache_key in cache:
            continue

        if t.ndim < 3:
            cache.add(cache_key)
            continue

        try:
            # all dimensions need to be the same
            (rank,) = set(t.shape)
        except ValueError:
            cache.add(cache_key)
            continue

        # compute the decomposition
        tcp = t.cp_decompose(rank, **cp_opts)

        if any(np.any(~np.isfinite(t.data)) for t in tcp):
            raise ValueError("bad value")

        # check if its exact up to atol
        err = tcp.distance(t)
        if err < atol:
            if equalize_norms:
                tcp.equalize_norms(value=equalize_norms)

            tn.pop_tensor(tid)
            tn |= tcp
        else:
            cache.add(cache_key)

    return tn


cp_simplify_ = functools.partialmethod(cp_simplify, inplace=True)
