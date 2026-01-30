import functools
import random
from itertools import chain, count, cycle

import autoray as ar

from quimb.gen.operators import swap
from quimb.tensor.tensor_core import (
    Tensor,
    bonds,
    rand_uuid,
    tags_to_oset,
    tensor_contract,
)
from quimb.tensor.tn1d.core import maybe_factor_gate_into_tensor
from quimb.utils import pairwise


def manhattan_distance(coo_a, coo_b):
    return sum(abs(coo_a[i] - coo_b[i]) for i in range(2))


def gate_string_split_(
    TG,
    where,
    string,
    original_ts,
    bonds_along,
    reindex_map,
    site_ix,
    info,
    **compress_opts,
):
    # by default this means singuvalues are kept in the string 'blob' tensor
    compress_opts.setdefault("absorb", "right")

    # the outer, neighboring indices of each tensor in the string
    neighb_inds = []

    # tensors we are going to contract in the blob, reindex some to attach gate
    contract_ts = []

    for t, coo in zip(original_ts, string):
        neighb_inds.append(tuple(ix for ix in t.inds if ix not in bonds_along))
        contract_ts.append(t.reindex(reindex_map) if coo in where else t)

    # form the central blob of all sites and gate contracted
    blob = tensor_contract(*contract_ts, TG)

    regauged = []

    # one by one extract the site tensors again from each end
    inner_ts = [None] * len(string)
    i = 0
    j = len(string) - 1

    while True:
        lix = neighb_inds[i]
        if i > 0:
            lix += (bonds_along[i - 1],)

        # the original bond we are restoring
        bix = bonds_along[i]

        # split the blob!
        inner_ts[i], *maybe_svals, blob = blob.split(
            left_inds=lix, get="tensors", bond_ind=bix, **compress_opts
        )

        # if singular values are returned (``absorb=None``) check if we should
        #     return them via ``info``, e.g. for ``SimpleUpdate`
        if maybe_svals and info is not None:
            s = next(iter(maybe_svals)).data
            coo_pair = tuple(sorted((string[i], string[i + 1])))
            info["singular_values", coo_pair] = s

            # regauge the blob but record so as to unguage later
            if i != j - 1:
                blob.multiply_index_diagonal_(bix, s)
                regauged.append((i + 1, bix, s))

        # move inwards along string, terminate if two ends meet
        i += 1
        if i == j:
            inner_ts[i] = blob
            break

        # extract at end of string
        lix = neighb_inds[j]
        if j < len(string) - 1:
            lix += (bonds_along[j],)

        # the original bond we are restoring
        bix = bonds_along[j - 1]

        # split the blob!
        inner_ts[j], *maybe_svals, blob = blob.split(
            left_inds=lix, get="tensors", bond_ind=bix, **compress_opts
        )

        # if singular values are returned (``absorb=None``) check if we should
        #     return them via ``info``, e.g. for ``SimpleUpdate`
        if maybe_svals and info is not None:
            s = next(iter(maybe_svals)).data
            coo_pair = tuple(sorted((string[j - 1], string[j])))
            info["singular_values", coo_pair] = s

            # regauge the blob but record so as to unguage later
            if j != i + 1:
                blob.multiply_index_diagonal_(bix, s)
                regauged.append((j - 1, bix, s))

        # move inwards along string, terminate if two ends meet
        j -= 1
        if j == i:
            inner_ts[j] = blob
            break

    # ungauge the site tensors along bond if necessary
    for i, bix, s in regauged:
        t = inner_ts[i]
        t.multiply_index_diagonal_(bix, s**-1)

    # transpose to match original tensors and update original data
    for to, tn in zip(original_ts, inner_ts):
        tn.transpose_like_(to)
        to.modify(data=tn.data)


def gate_string_reduce_split_(
    TG,
    where,
    string,
    original_ts,
    bonds_along,
    reindex_map,
    site_ix,
    info,
    **compress_opts,
):
    # by default this means singuvalues are kept in the string 'blob' tensor
    compress_opts.setdefault("absorb", "right")

    # indices to reduce, first and final include physical indices for gate
    inds_to_reduce = [(bonds_along[0], site_ix[0])]
    for b1, b2 in pairwise(bonds_along):
        inds_to_reduce.append((b1, b2))
    inds_to_reduce.append((bonds_along[-1], site_ix[-1]))

    # tensors that remain on the string sites and those pulled into string
    outer_ts, inner_ts = [], []
    for coo, rix, t in zip(string, inds_to_reduce, original_ts):
        tq, tr = t.split(
            left_inds=None, right_inds=rix, method="qr", get="tensors"
        )
        outer_ts.append(tq)
        inner_ts.append(tr.reindex_(reindex_map) if coo in where else tr)

    # contract the blob of gate with reduced tensors only
    blob = tensor_contract(*inner_ts, TG)

    regauged = []

    # extract the new reduced tensors sequentially from each end
    i = 0
    j = len(string) - 1

    while True:
        # extract at beginning of string
        lix = bonds(blob, outer_ts[i])
        if i == 0:
            lix.add(site_ix[0])
        else:
            lix.add(bonds_along[i - 1])

        # the original bond we are restoring
        bix = bonds_along[i]

        # split the blob!
        inner_ts[i], *maybe_svals, blob = blob.split(
            left_inds=lix, get="tensors", bond_ind=bix, **compress_opts
        )

        # if singular values are returned (``absorb=None``) check if we should
        #     return them via ``info``, e.g. for ``SimpleUpdate`
        if maybe_svals and info is not None:
            s = next(iter(maybe_svals)).data
            coo_pair = tuple(sorted((string[i], string[i + 1])))
            info["singular_values", coo_pair] = s

            # regauge the blob but record so as to unguage later
            if i != j - 1:
                blob.multiply_index_diagonal_(bix, s)
                regauged.append((i + 1, bix, s))

        # move inwards along string, terminate if two ends meet
        i += 1
        if i == j:
            inner_ts[i] = blob
            break

        # extract at end of string
        lix = bonds(blob, outer_ts[j])
        if j == len(string) - 1:
            lix.add(site_ix[-1])
        else:
            lix.add(bonds_along[j])

        # the original bond we are restoring
        bix = bonds_along[j - 1]

        # split the blob!
        inner_ts[j], *maybe_svals, blob = blob.split(
            left_inds=lix, get="tensors", bond_ind=bix, **compress_opts
        )

        # if singular values are returned (``absorb=None``) check if we should
        #     return them via ``info``, e.g. for ``SimpleUpdate`
        if maybe_svals and info is not None:
            s = next(iter(maybe_svals)).data
            coo_pair = tuple(sorted((string[j - 1], string[j])))
            info["singular_values", coo_pair] = s

            # regauge the blob but record so as to unguage later
            if j != i + 1:
                blob.multiply_index_diagonal_(bix, s)
                regauged.append((j - 1, bix, s))

        # move inwards along string, terminate if two ends meet
        j -= 1
        if j == i:
            inner_ts[j] = blob
            break

    # reabsorb the inner reduced tensors into the sites
    new_ts = [
        tensor_contract(ts, tr, output_inds=to.inds)
        for to, ts, tr in zip(original_ts, outer_ts, inner_ts)
    ]

    # ungauge the site tensors along bond if necessary
    for i, bix, s in regauged:
        t = new_ts[i]
        t.multiply_index_diagonal_(bix, s**-1)

    # update originals
    for to, t in zip(original_ts, new_ts):
        to.modify(data=t.data)


def gate_2d_long_range(
    self,
    G,
    where,
    contract=False,
    tags=None,
    inplace=False,
    info=None,
    long_range_use_swaps=False,
    long_range_path_sequence=None,
    **compress_opts,
):
    psi = self if inplace else self.copy()

    ng = len(where)

    dp = psi.phys_dim(*where[0])
    tags = tags_to_oset(tags)

    # allow a matrix to be reshaped into a tensor if it factorizes
    #     i.e. (4, 4) assumed to be two qubit gate -> (2, 2, 2, 2)
    G = maybe_factor_gate_into_tensor(G, dp, ng, where)

    site_ix = [psi.site_ind(i, j) for i, j in where]
    # new indices to join old physical sites to new gate
    bnds = [rand_uuid() for _ in range(ng)]
    reindex_map = dict(zip(site_ix, bnds))

    TG = Tensor(G, inds=site_ix + bnds, tags=tags, left_inds=bnds)

    # following are all based on splitting tensors to maintain structure
    ij_a, ij_b = where

    # parse the argument specifying how to find the path between
    # non-nearest neighbours
    if long_range_path_sequence is not None:
        # make sure we can index
        long_range_path_sequence = tuple(long_range_path_sequence)
        # if the first element is a str specifying move sequence, e.g.
        #     ('v', 'h')
        #     ('av', 'bv', 'ah', 'bh')  # using swaps
        manual_lr_path = not isinstance(long_range_path_sequence[0], str)
        # otherwise assume a path has been manually specified, e.g.
        #     ((1, 2), (2, 2), (2, 3), ... )
        #     (((1, 1), (1, 2)), ((4, 3), (3, 3)), ...)  # using swaps
    else:
        manual_lr_path = False

    # check if we are not nearest neighbour and need to swap first
    if long_range_use_swaps:
        if manual_lr_path:
            *swaps, final = long_range_path_sequence
        else:
            # find a swap path
            *swaps, final = gen_long_range_swap_path(
                ij_a, ij_b, sequence=long_range_path_sequence
            )

        # move the sites together
        SWAP = get_swap(
            dp, dtype=ar.get_dtype_name(G), backend=ar.infer_backend(G)
        )
        for pair in swaps:
            psi.gate_(SWAP, pair, contract=contract, absorb="right")

        compress_opts["info"] = info
        compress_opts["contract"] = contract

        # perform actual gate also compressing etc on 'way back'
        psi.gate_(G, final, **compress_opts)

        compress_opts.setdefault("absorb", "both")
        for pair in reversed(swaps):
            psi.gate_(SWAP, pair, **compress_opts)

        return psi

    if manual_lr_path:
        string = long_range_path_sequence
    else:
        string = tuple(
            gen_long_range_path(*where, sequence=long_range_path_sequence)
        )

    # the tensors along this string, which will be updated
    original_ts = [psi[coo] for coo in string]

    # the len(string) - 1 indices connecting the string
    bonds_along = [
        next(iter(bonds(t1, t2))) for t1, t2 in pairwise(original_ts)
    ]

    if contract == "split":
        #
        #       │╱  │╱          │╱  │╱
        #     ──GGGGG──  ==>  ──G┄┄┄G──
        #      ╱   ╱           ╱   ╱
        #
        gate_string_split_(
            TG,
            where,
            string,
            original_ts,
            bonds_along,
            reindex_map,
            site_ix,
            info,
            **compress_opts,
        )

    elif contract == "reduce-split":
        #
        #       │   │             │ │
        #       GGGGG             GGG               │ │
        #       │╱  │╱   ==>     ╱│ │  ╱   ==>     ╱│ │  ╱          │╱  │╱
        #     ──●───●──       ──>─●─●─<──       ──>─GGG─<──  ==>  ──G┄┄┄G──
        #      ╱   ╱           ╱     ╱           ╱     ╱           ╱   ╱
        #    <QR> <LQ>                            <SVD>
        #
        gate_string_reduce_split_(
            TG,
            where,
            string,
            original_ts,
            bonds_along,
            reindex_map,
            site_ix,
            info,
            **compress_opts,
        )

    return psi


def gen_long_range_path(ij_a, ij_b, sequence=None):
    """Generate a string of coordinates, in order, from ``ij_a`` to ``ij_b``.

    Parameters
    ----------
    ij_a : (int, int)
        Coordinate of site 'a'.
    ij_b : (int, int)
        Coordinate of site 'b'.
    sequence : None, iterable of {'v', 'h'}, or 'random', optional
        What order to cycle through and try and perform moves in, 'v', 'h'
        standing for move vertically and horizontally respectively. The default
        is ``('v', 'h')``.

    Returns
    -------
    generator[tuple[int]]
        The path, each element is a single coordinate.
    """
    ia, ja = ij_a
    ib, jb = ij_b
    di = ib - ia
    dj = jb - ja

    # nearest neighbour
    if abs(di) + abs(dj) == 1:
        yield ij_a
        yield ij_b
        return

    if sequence is None:
        poss_moves = cycle(("v", "h"))
    elif sequence == "random":
        poss_moves = (random.choice("vh") for _ in count())
    else:
        poss_moves = cycle(sequence)

    yield ij_a

    for move in poss_moves:
        if abs(di) + abs(dj) == 1:
            yield ij_b
            return

        if (move == "v") and (di != 0):
            # move a vertically
            istep = min(max(di, -1), +1)
            new_ij_a = (ia + istep, ja)
            yield new_ij_a
            ij_a = new_ij_a
            ia += istep
            di -= istep
        elif (move == "h") and (dj != 0):
            # move a horizontally
            jstep = min(max(dj, -1), +1)
            new_ij_a = (ia, ja + jstep)
            yield new_ij_a
            ij_a = new_ij_a
            ja += jstep
            dj -= jstep


def gen_long_range_swap_path(ij_a, ij_b, sequence=None):
    """Generate the coordinates or a series of swaps that would bring ``ij_a``
    and ``ij_b`` together.

    Parameters
    ----------
    ij_a : (int, int)
        Coordinate of site 'a'.
    ij_b : (int, int)
        Coordinate of site 'b'.
    sequence : None, it of {'av', 'bv', 'ah', 'bh'}, or 'random', optional
        What order to cycle through and try and perform moves in, 'av', 'bv',
        'ah', 'bh' standing for move 'a' vertically, 'b' vertically, 'a'
        horizontally', and 'b' horizontally respectively. The default is
        ``('av', 'bv', 'ah', 'bh')``.

    Returns
    -------
    generator[tuple[tuple[int]]]
        The path, each element is two coordinates to swap.
    """
    ia, ja = ij_a
    ib, jb = ij_b
    di = ib - ia
    dj = jb - ja

    # nearest neighbour
    if abs(di) + abs(dj) == 1:
        yield (ij_a, ij_b)
        return

    if sequence is None:
        poss_moves = cycle(("av", "bv", "ah", "bh"))
    elif sequence == "random":
        poss_moves = (random.choice(("av", "bv", "ah", "bh")) for _ in count())
    else:
        poss_moves = cycle(sequence)

    for move in poss_moves:
        if (move == "av") and (di != 0):
            # move a vertically
            istep = min(max(di, -1), +1)
            new_ij_a = (ia + istep, ja)
            yield (ij_a, new_ij_a)
            ij_a = new_ij_a
            ia += istep
            di -= istep

        elif (move == "bv") and (di != 0):
            # move b vertically
            istep = min(max(di, -1), +1)
            new_ij_b = (ib - istep, jb)
            # need to make sure final gate is applied correct way
            if new_ij_b == ij_a:
                yield (ij_a, ij_b)
            else:
                yield (ij_b, new_ij_b)
            ij_b = new_ij_b
            ib -= istep
            di -= istep

        elif (move == "ah") and (dj != 0):
            # move a horizontally
            jstep = min(max(dj, -1), +1)
            new_ij_a = (ia, ja + jstep)
            yield (ij_a, new_ij_a)
            ij_a = new_ij_a
            ja += jstep
            dj -= jstep

        elif (move == "bh") and (dj != 0):
            # move b horizontally
            jstep = min(max(dj, -1), +1)
            new_ij_b = (ib, jb - jstep)
            # need to make sure final gate is applied correct way
            if new_ij_b == ij_a:
                yield (ij_a, ij_b)
            else:
                yield (ij_b, new_ij_b)
            ij_b = new_ij_b
            jb -= jstep
            dj -= jstep

        if di == dj == 0:
            return


def swap_path_to_long_range_path(swap_path, ij_a):
    """Generates the ordered long-range path - a sequence of coordinates - from
    a (long-range) swap path - a sequence of coordinate pairs.
    """
    sites = set(chain(*swap_path))
    return sorted(sites, key=lambda ij_b: manhattan_distance(ij_a, ij_b))


@functools.lru_cache(8)
def get_swap(dp, dtype, backend):
    SWAP = swap(dp, dtype=dtype)
    return ar.do("array", SWAP, like=backend)
