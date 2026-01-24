"""Generic methods for compressing 1D-like tensor networks, where the tensor
network can locally have arbitrary structure and outer indices.

- [x] the direct method
- [x] the density matrix method
- [x] the zip-up method
- [x] the zip-up first method
- [x] the 1-site variational fit method, including sums of tensor networks
- [x] the 2-site variational fit method, including sums of tensor networks
- [x] the local projector method (CTMRG and HOTRG style)
- [x] the autofit method (via non-1d specific ALS or autodiff)

"""

import collections
import functools
import itertools
import warnings

from autoray import do

from ..gen.rand import randn
from .tensor_arbgeom import tensor_network_apply_op_vec
from .tensor_arbgeom_compress import tensor_network_ag_compress
from .tensor_builder import TN_matching, rand_tensor
from .tensor_core import (
    Tensor,
    TensorNetwork,
    bonds,
    ensure_dict,
    oset,
    rand_uuid,
    tensor_contract,
)


def enforce_1d_like(tn, site_tags=None, fix_bonds=True, inplace=False):
    """Check that ``tn`` is 1D-like with OBC, i.e. 1) that each tensor has
    exactly one of the given ``site_tags``. If not, raise a ValueError. 2) That
    there are no hyper indices. And 3) that there are only bonds within sites
    or between nearest neighbor sites. This issue can be optionally
    automatically fixed by inserting a string of identity tensors.

    Parameters
    ----------
    tn : TensorNetwork
        The tensor network to check.
    site_tags : sequence of str, optional
        The tags to use to group and order the tensors from ``tn``. If not
        given, uses ``tn.site_tags``.
    fix_bonds : bool, optional
        Whether to fix the bond structure by inserting identity tensors.
    inplace : bool, optional
        Whether to perform the fix inplace or not.

    Raises
    ------
    ValueError
        If the tensor network is not 1D-like.
    """
    tn = tn if inplace else tn.copy()

    if site_tags is None:
        site_tags = tn.site_tags

    tag_to_site = {tag: i for i, tag in enumerate(site_tags)}
    tid_to_site = {}

    def _check_tensor_site(tid, t):
        if tid in tid_to_site:
            return tid_to_site[tid]

        sites = []
        for tag in t.tags:
            site = tag_to_site.get(tag, None)
            if site is not None:
                sites.append(site)
        if len(sites) != 1:
            raise ValueError(
                f"{t} does not have one site tag, it has {sites}."
            )

        return sites[0]

    for ix, tids in list(tn.ind_map.items()):
        if len(tids) == 1:
            # assume outer
            continue
        elif len(tids) != 2:
            raise ValueError(
                f"TN has a hyper index, {ix}, connecting more than 2 tensors."
            )

        tida, tidb = tids
        ta = tn.tensor_map[tida]
        tb = tn.tensor_map[tidb]

        # get which single site each tensor belongs too
        sa = _check_tensor_site(tida, ta)
        sb = _check_tensor_site(tidb, tb)
        if sa > sb:
            sa, sb = sb, sa

        if sb - sa > 1:
            if not fix_bonds:
                raise ValueError(
                    f"Tensor {ta} and {tb} are not nearest "
                    "neighbors, and `fix_bonds=False`."
                )

            # not 1d like: bond is not nearest neighbor
            # but can insert identites along string to fix
            data = do("eye", ta.ind_size(ix), like=ta.data)

            ixl = ix
            for i in range(sa + 1, sb):
                ixr = rand_uuid()
                tn |= Tensor(
                    data=data,
                    inds=[ixl, ixr],
                    tags=site_tags[i],
                )
                ixl = ixr

            tb.reindex_({ix: ixl})

    return tn


def possibly_permute_(tn, permute_arrays):
    # possibly put the array indices in canonical order (e.g. when MPS or MPO)
    if permute_arrays and hasattr(tn, "permute_arrays"):
        if permute_arrays is True:
            # use default order
            tn.permute_arrays()
        else:
            # use given order
            tn.permute_arrays(permute_arrays)


def tensor_network_1d_compress_direct(
    tn,
    max_bond=None,
    cutoff=1e-10,
    site_tags=None,
    normalize=False,
    canonize=True,
    cutoff_mode="rsum2",
    permute_arrays=True,
    optimize="auto-hq",
    sweep_reverse=False,
    equalize_norms=False,
    inplace=False,
    **compress_opts,
):
    """Compress a 1D-like tensor network using the 'direct' or 'naive' method,
    that is, explicitly contracting site-wise to form a MPS-like TN,
    canonicalizing in one direction, then compressing in the other. This has
    the same scaling as the density matrix (dm) method, but a larger prefactor.
    It can still be faster for small bond dimensions however, and is
    potentially higher precision since it works in the space of singular values
    directly rather than singular values squared. It is not quite optimal in
    terms of error due to the compounding errors of the SVDs.

    Parameters
    ----------
    tn : TensorNetwork
        The tensor network to compress. Every tensor should have exactly one of
        the site tags. Each site can have multiple tensors and output indices.
    max_bond : int
        The maximum bond dimension to compress to.
    cutoff : float, optional
        A dynamic threshold for discarding singular values when compressing.
    site_tags : sequence of str, optional
        The tags to use to group and order the tensors from ``tn``. If not
        given, uses ``tn.site_tags``. The tensor network built will have one
        tensor per site, in the order given by ``site_tags``.
    normalize : bool, optional
        Whether to normalize the final tensor network, making use of the fact
        that the output tensor network is in right canonical form.
    canonize : bool, optional
        Whether to canonicalize the network in one direction before compressing
        in the other.
    cutoff_mode : {"rsum2", "rel", ...}, optional
        The mode to use when truncating the singular values of the decomposed
        tensors. See :func:`~quimb.tensor.tensor_split`.
    permute_arrays : bool or str, optional
        Whether to permute the array indices of the final tensor network into
        canonical order. If ``True`` will use the default order, otherwise if a
        string this specifies a custom order.
    optimize : str, optional
        The contraction path optimizer to use.
    sweep_reverse : bool, optional
        Whether to sweep in the reverse direction, resulting in a left
        canonical form instead of right canonical.
    equalize_norms : bool, optional
        Whether to renormalize the tensors during the compression procedure.
        If ``True`` the gathered exponent will be redistributed equally among
        the tensors. If a float, all tensors will be renormalized to this
        value, and the gathered exponent is tracked in ``tn.exponent`` of the
        returned tensor network.
    inplace : bool, optional
        Whether to perform the compression inplace or not.
    compress_opts
        Supplied to :func:`~quimb.tensor.tensor_split`.

    Returns
    -------
    TensorNetwork
        The compressed tensor network, with canonical center at
        ``site_tags[0]`` ('right canonical' form) or ``site_tags[-1]`` ('left
        canonical' form) if ``sweep_reverse``.
    """
    if site_tags is None:
        site_tags = tn.site_tags
    if sweep_reverse:
        site_tags = tuple(reversed(site_tags))

    new = enforce_1d_like(tn, site_tags=site_tags, inplace=inplace)

    # contract the first site group
    new.contract_tags_(site_tags[0], optimize=optimize)

    # sweep right
    for i in range(1, len(site_tags)):
        # contract the next site group
        new.contract_tags_(site_tags[i], optimize=optimize)
        #     │ │ │ │ │ │ │ │ │ │
        #     ▶━▶━▶━▶═○─○─○─○─○─○
        #              ╲│ │ │ │ │
        #           : : ○─○─○─○─○
        #         i-1 i

        if canonize:
            # shift canonical center rightwards
            new.canonize_between(
                site_tags[i - 1], site_tags[i], equalize_norms=equalize_norms
            )
            #     │ │ │ │ │ │ │ │ │ │
            #     ▶━▶━▶━▶━▶─○─○─○─○─○
            #              ╲│ │ │ │ │
            #           : : ○─○─○─○─○
            #         i-1 i

    # sweep left
    for i in range(len(site_tags) - 1, 0, -1):
        # compress and shift canonical center leftwards
        new.compress_between(
            site_tags[i - 1],
            site_tags[i],
            absorb="left",
            reduced="right",
            max_bond=max_bond,
            cutoff=cutoff,
            cutoff_mode=cutoff_mode,
            equalize_norms=equalize_norms,
            **compress_opts,
        )
        #     │ │ │ │ │ │ │ │ │ │
        #     ▶━▶━▶━▶━▶━▶━○─◀─◀─◀
        #                      :
        #                 : :  max_bond
        #               i-1 i

    if normalize:
        # make use of the fact that the output is in right canonical form
        t0 = new[site_tags[0]]
        t0.normalize_()
        new.exponent = 0.0
    elif equalize_norms is True:
        # redistribute the exponent equally among the tensors
        new.equalize_norms_()

    # possibly put the array indices in canonical order (e.g. when MPS or MPO)
    possibly_permute_(new, permute_arrays)

    return new


def _form_final_tn_from_tensor_sequence(
    tn,
    ts,
    normalize,
    sweep_reverse,
    permute_arrays,
    equalize_norms,
    inplace,
    tags_per_site=None,
):
    if tags_per_site is not None:
        for t, tags in zip(ts, tags_per_site):
            t.modify(tags=tags)

    if normalize:
        # in right canonical form already
        ts[0].normalize_()

    if sweep_reverse:
        # this is purely cosmetic for ordering tensor_map dict entries
        ts.reverse()

    # form the final TN
    if inplace:
        # simply replace all tensors
        new = tn
        new.remove_all_tensors()
        new.add(ts, virtual=True)
    else:
        new = TensorNetwork(ts, virtual=True)
        # cast as whatever the input was e.g. MPS, MPO
        new.view_like_(tn)
        new.exponent = tn.exponent

    # possibly put the array indices in canonical order (e.g. when MPS or MPO)
    possibly_permute_(new, permute_arrays)

    # XXX: do better than simply waiting til the end to equalize norms
    if equalize_norms is True:
        new.equalize_norms_()
    elif equalize_norms:
        new.equalize_norms_(value=equalize_norms)

    return new


def tensor_network_1d_compress_dm(
    tn,
    max_bond=None,
    cutoff=1e-10,
    site_tags=None,
    normalize=False,
    cutoff_mode="rsum1",
    permute_arrays=True,
    optimize="auto-hq",
    sweep_reverse=False,
    canonize=True,
    equalize_norms=False,
    strip_exponent=False,
    inplace=False,
    **compress_opts,
):
    """Compress any 1D-like tensor network using the 'density matrix' method
    (https://tensornetwork.org/mps/algorithms/denmat_mpo_mps/).

    While this has the same scaling as the direct method, in practice it can
    often be faster, especially at large bond dimensions. Potentially there are
    some situations where the direct method is more stable with regard to
    precision, since the density matrix method works in the 'squared' picture.

    Parameters
    ----------
    tn : TensorNetwork
        The tensor network to compress. Every tensor should have exactly one of
        the site tags. Each site can have multiple tensors and output indices.
    max_bond : int
        The maximum bond dimension to compress to.
    cutoff : float, optional
        The truncation error to use when compressing the double layer tensor
        network.
    site_tags : sequence of str, optional
        The tags to use to group and order the tensors from ``tn``. If not
        given, uses ``tn.site_tags``. The tensor network built will have one
        tensor per site, in the order given by ``site_tags``.
    normalize : bool, optional
        Whether to normalize the final tensor network, making use of the fact
        that the output tensor network is in right canonical form.
    cutoff_mode : {"rsum1", "rel", ...}, optional
        The mode to use when truncating the singular values of the decomposed
        tensors. See :func:`~quimb.tensor.tensor_split`. Note for the density
        matrix method the default 'rsum1' mode acts like 'rsum2' for the direct
        method due to truncating in the squared space.
    permute_arrays : bool or str, optional
        Whether to permute the array indices of the final tensor network into
        canonical order. If ``True`` will use the default order, otherwise if a
        string this specifies a custom order.
    optimize : str, optional
        The contraction path optimizer to use.
    sweep_reverse : bool, optional
        Whether to sweep in the reverse direction, resulting in a left
        canonical form instead of right canonical.
    canonize : bool, optional
        Dummy argument to match the signature of other compression methods.
    equalize_norms : bool or float, optional
        Whether to equalize the norms of the tensors after compression. If an
        explicit value is give, then the norms will be set to that value, and
        the overall scaling factor will be accumulated into `.exponent`.
    strip_exponent : bool, optional
        If `True`, the intermediate tensors are normalized during contraction,
        and the overall scaling factor will be accumulated into `.exponent`.
    inplace : bool, optional
        Whether to perform the compression inplace or not.
    compress_opts
        Supplied to :func:`~quimb.tensor.tensor_split`.

    Returns
    -------
    TensorNetwork
        The compressed tensor network, with canonical center at
        ``site_tags[0]`` ('right canonical' form) or ``site_tags[-1]`` ('left
        canonical' form) if ``sweep_reverse``.
    """
    if not canonize:
        warnings.warn("`canonize=False` is ignored for the `dm` method.")

    if site_tags is None:
        site_tags = tn.site_tags
    if sweep_reverse:
        site_tags = tuple(reversed(site_tags))
    N = len(site_tags)

    ket = enforce_1d_like(tn, site_tags=site_tags, inplace=inplace)

    # partition outer indices, and create conjugate bra indices
    ket_site_inds = []
    bra_site_inds = []
    ketbra_indmap = {}
    for tag in site_tags:
        k_inds_i = []
        b_inds_i = []
        for kix in ket.select(tag)._outer_inds & ket._outer_inds:
            bix = rand_uuid()
            k_inds_i.append(kix)
            b_inds_i.append(bix)
            ketbra_indmap[kix] = bix
        ket_site_inds.append(tuple(k_inds_i))
        bra_site_inds.append(tuple(b_inds_i))

    bra = ket.H
    # doing this means forming the norm doesn't do its own mangling
    bra.mangle_inner_()
    # form the overlapping double layer TN
    norm = bra & ket
    # open the bra's indices back up
    bra.reindex_(ketbra_indmap)

    # construct dense left environments
    # (n.b. K, B generally *collection* of tensors)
    #
    #     K══K══K══K══            ╔═K══          ╔═
    #     │  │  │  │       ->    LE │     = ... LE
    #     B══B══B══B══            ╚═B══          ╚═
    #     0  1  2  3            i-1 i            i
    #
    left_envs = {}
    left_envs[1] = norm.select(site_tags[0]).contract(
        preserve_tensor=True,
        drop_tags=True,
        optimize=optimize,
    )
    left_envs[1].normalize_()
    for i in range(2, N):
        left_envs[i] = tensor_contract(
            left_envs[i - 1],
            *norm.select_tensors(site_tags[i - 1]),
            preserve_tensor=True,
            drop_tags=True,
            optimize=optimize,
        )
        left_envs[i].normalize_()

    # build projectors and right environments
    Us = [None] * N
    right_env_ket = None
    right_env_bra = None
    new_bonds = collections.defaultdict(rand_uuid)
    exponent = 0.0

    for i in range(N - 1, 0, -1):
        # form the reduced density matrix
        #
        #                          ket_site_inds[i]
        #          │   ┃           :  new_bonds["k", i + 1]
        #       ╔══K══REk          │  ┃
        #      LE            =>    rhoi
        #       ╚══B══REb          │  ┃
        #          │   ┃           :  new_bonds["b", i + 1]
        #                          bra_site_inds[i]
        #     i-1  i  i+1
        #
        rho_tensors = [
            left_envs[i],
            *ket.select_tensors(site_tags[i]),
            *bra.select_tensors(site_tags[i]),
        ]
        left_inds = list(ket_site_inds[i])
        right_inds = list(bra_site_inds[i])
        if right_env_ket is not None:
            rho_tensors.extend([right_env_ket, right_env_bra])
            left_inds.append(new_bonds["k", i + 1])
            right_inds.append(new_bonds["b", i + 1])

        # contract and then split it
        #
        #                    │  ┃
        #     │  ┃           UUUU
        #     │  ┃            ┃
        #     rhoi    =>      s    ... bix, with size max_bond
        #     │  ┃            ┃
        #     │  ┃           UHUH
        #                    │  ┃
        #
        rhoi = tensor_contract(
            *rho_tensors,
            preserve_tensor=True,
            optimize=optimize,
        )

        # XXX: fermionic, see fix in squared_op_to_reduced_factor

        U, s, UH = rhoi.split(
            left_inds=left_inds,
            right_inds=right_inds,
            method="eigh",
            positive=1,
            max_bond=max_bond,
            cutoff=cutoff,
            cutoff_mode=cutoff_mode,
            get="tensors",
            absorb=None,
            **compress_opts,
        )

        # turn bond into 'virtual right' indices
        #
        #     │  ┃
        #     UUUU
        #      ┃   ...new_bonds["k", i]
        #     ✂
        #      ┃   ...new_bonds["b", i]
        #     UHUH
        #     │  ┃
        #
        (bix,) = s.inds
        U.reindex_({bix: new_bonds["k", i]})
        UH.reindex_({bix: new_bonds["b", i]})
        Us[i] = U

        # attach the unitaries to the right environments and contract
        #
        #         ┃
        #       UUUUU†             new_bonds["k", i]
        #       │   ┃     =>       ┃
        #     ══K══REk          ══REk
        #
        #       i   i+1            i
        #
        #     ══B══REb          ══REb
        #       │   ┃     =>       ┃
        #       UHUHU†             new_bonds["b", i]
        #         ┃
        #
        right_ket_tensors = [*ket.select_tensors(site_tags[i]), U.H]
        right_bra_tensors = [*bra.select_tensors(site_tags[i]), UH.H]
        if right_env_ket is not None:
            # we have already done one move -> have right envs
            right_ket_tensors.append(right_env_ket)
            right_bra_tensors.append(right_env_bra)

        if strip_exponent:
            right_env_ket, result_exponent = tensor_contract(
                *right_ket_tensors,
                preserve_tensor=True,
                drop_tags=True,
                optimize=optimize,
                strip_exponent=strip_exponent,
            )
            exponent += result_exponent
        else:
            right_env_ket = tensor_contract(
                *right_ket_tensors,
                preserve_tensor=True,
                drop_tags=True,
                optimize=optimize,
            )
        # TODO: could compute this just as conjugated and relabelled ket env
        if strip_exponent:
            right_env_bra, _ = tensor_contract(
                *right_bra_tensors,
                preserve_tensor=True,
                drop_tags=True,
                optimize=optimize,
                strip_exponent=strip_exponent,
            )
        else:
            right_env_bra = tensor_contract(
                *right_bra_tensors,
                preserve_tensor=True,
                drop_tags=True,
                optimize=optimize,
            )

    # form the final site
    #
    #     │   ┏━━         │
    #     K══REk     =    U━━
    #     0   1           0
    #
    Us[0] = tensor_contract(
        *ket.select_tensors(site_tags[0]),
        right_env_ket,
        optimize=optimize,
        preserve_tensor=True,
    )

    new = _form_final_tn_from_tensor_sequence(
        tn,
        Us,
        normalize,
        sweep_reverse,
        permute_arrays,
        equalize_norms,
        inplace,
    )
    if strip_exponent:
        new.exponent += exponent

    return new


def tensor_network_1d_compress_zipup(
    tn,
    max_bond=None,
    cutoff=1e-10,
    site_tags=None,
    canonize=True,
    normalize=False,
    cutoff_mode="rsum2",
    permute_arrays=True,
    optimize="auto-hq",
    sweep_reverse=False,
    equalize_norms=False,
    strip_exponent=False,
    inplace=False,
    **compress_opts,
):
    """Compress a 1D-like tensor network using the 'zip-up' algorithm due to
    'Minimally Entangled Typical Thermal State Algorithms', E.M. Stoudenmire &
    Steven R. White (https://arxiv.org/abs/1002.1305). The returned tensor
    network will have one tensor per site, in the order given by ``site_tags``,
    with canonical center at ``site_tags[0]`` ('right' canonical form).

    The zipup algorithm scales better than the direct and density matrix
    methods when multiple tensors are present at each site (such as MPO-MPS
    multiplication), but is less accurate due to the compressions taking place
    in an only pseudo-canonical gauge. It generally also only makes sense in
    the fixed bond dimension case, as opposed to relying on a specific
    `cutoff` only.

    Parameters
    ----------
    tn : TensorNetwork
        The tensor network to compress. Every tensor should have exactly one of
        the site tags. Each site can have multiple tensors and output indices.
    max_bond : int
        The maximum bond dimension to compress to.
    cutoff : float, optional
        A dynamic threshold for discarding singular values when compressing.
    site_tags : sequence of str, optional
        The tags to use to group and order the tensors from ``tn``. If not
        given, uses ``tn.site_tags``. The tensor network built will have one
        tensor per site, in the order given by ``site_tags``.
    canonize : bool, optional
        Whether to pseudo canonicalize the initial tensor network.
    normalize : bool, optional
        Whether to normalize the final tensor network, making use of the fact
        that the output tensor network is in right canonical form.
    cutoff_mode : {"rsum2", "rel", ...}, optional
        The mode to use when truncating the singular values of the decomposed
        tensors. See :func:`~quimb.tensor.tensor_split`.
    permute_arrays : bool or str, optional
        Whether to permute the array indices of the final tensor network into
        canonical order. If ``True`` will use the default order, otherwise if a
        string this specifies a custom order.
    optimize : str, optional
        The contraction path optimizer to use.
    sweep_reverse : bool, optional
        Whether to sweep in the reverse direction, resulting in a left
        canonical form instead of right canonical.
    equalize_norms : bool or float, optional
        Whether to equalize the norms of the tensors after compression. If an
        explicit value is give, then the norms will be set to that value, and
        the overall scaling factor will be accumulated into `.exponent`.
    strip_exponent : bool, optional
        If `True`, the intermediate tensors are normalized during contraction,
        and the overall scaling factor will be accumulated into `.exponent`.
    inplace : bool, optional
        Whether to perform the compression inplace or not.
    compress_opts
        Supplied to :func:`~quimb.tensor.tensor_split`.

    Returns
    -------
    TensorNetwork
        The compressed tensor network, with canonical center at
        ``site_tags[0]`` ('right canonical' form) or ``site_tags[-1]`` ('left
        canonical' form) if ``sweep_reverse``.
    """
    if site_tags is None:
        site_tags = tn.site_tags
    if sweep_reverse:
        site_tags = tuple(reversed(site_tags))
    N = len(site_tags)

    tn = enforce_1d_like(tn, site_tags=site_tags, inplace=inplace)

    # calculate the local site (outer) indices
    site_inds = [
        tuple(tn.select(tag)._outer_inds & tn._outer_inds) for tag in site_tags
    ]

    if canonize:
        # put in 'pseudo' left canonical form:
        # (NB: diagrams assume MPO-MPS but algorithm is agnostic)
        #
        #     │ │ │ │ │ │ │ │ │ │
        #     ▶─▶─▶─▶─▶─▶─▶─▶─▶─○  MPO
        #     │ │ │ │ │ │ │ │ │ │
        #     ▶─▶─▶─▶─▶─▶─▶─▶─▶─○  MPS
        #
        tn = tn.canonize_around_(site_tags[-1], equalize_norms=equalize_norms)

    # zip along the bonds
    ts = [None] * N
    bix = None
    Us = None
    exponent = 0.0
    for i in range(N - 1, 0, -1):
        #          U*s VH
        #      │ │     │ │
        #     ─▶─▶──□━━◀━◀━
        #      │ │ ╱    :
        #     ─▶─▶    max_bond
        #        i
        #        .... contract
        if Us is None:
            # first site
            C = tensor_contract(
                *tn.select_tensors(site_tags[i]), optimize=optimize
            )
        else:
            if strip_exponent:
                C, result_exponent = tensor_contract(
                    Us, *tn.select_tensors(site_tags[i]), 
                    optimize=optimize,
                    strip_exponent=strip_exponent
                )
                exponent += result_exponent
            else:
                C = tensor_contract(
                    Us, *tn.select_tensors(site_tags[i]), optimize=optimize
                )
        #         i
        #      │  │    │ │
        #     ─▶──□━━━━◀━◀━
        #      │ ╱   :
        #     ─▶  :  bix
        #         C
        right_inds = list(site_inds[i])
        if bix is not None:
            right_inds.append(bix)

        # the new bond index, keep track for above
        bix = rand_uuid()

        Us, VH = C.split(
            left_inds=None,
            right_inds=right_inds,
            max_bond=max_bond,
            cutoff=cutoff,
            cutoff_mode=cutoff_mode,
            absorb="left",
            bond_ind=bix,
            get="tensors",
            **compress_opts,
        )
        Us.drop_tags()
        ts[i] = VH
        #           i
        #      │    │  │ │
        #     ─▶──□━◀━━◀━◀━
        #      │ ╱
        #     ─▶  : :
        #       U*s VH

    # compute final site
    ts[0] = tensor_contract(
        Us, *tn.select_tensors(site_tags[0]), optimize=optimize
    )

    new = _form_final_tn_from_tensor_sequence(
        tn,
        ts,
        normalize,
        sweep_reverse,
        permute_arrays,
        equalize_norms,
        inplace,
    )
    if strip_exponent:
        new.exponent += exponent

    return new


def _do_direct_sweep(
    tn,
    site_tags,
    max_bond,
    cutoff,
    cutoff_mode,
    equalize_norms,
    normalize,
    permute_arrays,
    **compress_opts,
):
    for i in range(len(site_tags) - 1, 0, -1):
        # compress and shift canonical center
        tn.compress_between(
            site_tags[i - 1],
            site_tags[i],
            absorb="left",
            reduced="right",
            max_bond=max_bond,
            cutoff=cutoff,
            cutoff_mode=cutoff_mode,
            equalize_norms=equalize_norms,
            **compress_opts,
        )

    if normalize:
        # make use of the fact that the output is in right canonical form
        tn[site_tags[-1]].normalize_()

    # possibly put the array indices in canonical order (e.g. when MPS or MPO)
    possibly_permute_(tn, permute_arrays)

    if equalize_norms is True:
        tn.equalize_norms_()
    elif equalize_norms:
        tn.equalize_norms_(value=equalize_norms)

    return tn


def tensor_network_1d_compress_zipup_oversample(
    tn,
    max_bond=None,
    max_bond_oversample=None,
    cutoff=1e-10,
    cutoff_oversample=None,
    site_tags=None,
    canonize=True,
    normalize=False,
    cutoff_mode="rsum2",
    permute_arrays=True,
    optimize="auto-hq",
    sweep_reverse=False,
    equalize_norms=False,
    strip_exponent=False,
    inplace=False,
    **compress_opts,
):
    """Compress this 1D-like tensor network using the 'zip-up first' algorithm,
    that is, first compressing the tensor network to a larger bond dimension
    using the 'zip-up' algorithm, then compressing to the desired bond
    dimension using a direct sweep.

    Depending on the value of ``max_bond`` and ``max_bond_oversample``, this
    can be scale better than the direct and density matrix methods, but reach
    close to the same accuracy. As with the 'zip-up' method, there is no
    advantage unless there are multiple tensors per site, and it generally only
    makes sense in the fixed bond dimension case, as opposed to relying on a
    specific `cutoff` only.

    Parameters
    ----------
    tn : TensorNetwork
        The tensor network to compress. Every tensor should have exactly one of
        the site tags. Each site can have multiple tensors and output indices.
    max_bond : int
        The final maximum bond dimension to compress to.
    max_bond_oversample : int, optional
        The intermediate maximum bond dimension to compress to using the
        'zip-up' algorithm. If not given and `max_bond` is, this is set as
        twice the target bond dimension, ``2 * max_bond``.
    cutoff : float, optional
        A dynamic threshold for discarding singular values when compressing.
    cutoff_oversample : float, optional
        A dynamic threshold for discarding singular values when compressing to
        the intermediate bond dimension using the 'zip-up' algorithm. If not
        given, this is set to the same as ``cutoff`` if a maximum bond is
        given, else ``cutoff / 10``.
    site_tags : sequence of str, optional
        The tags to use to group and order the tensors from ``tn``. If not
        given, uses ``tn.site_tags``. The tensor network built will have one
        tensor per site, in the order given by ``site_tags``.
    canonize : bool, optional
        Whether to pseudo canonicalize the initial tensor network.
    normalize : bool, optional
        Whether to normalize the final tensor network, making use of the fact
        that the output tensor network is in right canonical form.
    cutoff_mode : {"rsum2", "rel", ...}, optional
        The mode to use when truncating the singular values of the decomposed
        tensors. See :func:`~quimb.tensor.tensor_split`.
    permute_arrays : bool or str, optional
        Whether to permute the array indices of the final tensor network into
        canonical order. If ``True`` will use the default order, otherwise if a
        string this specifies a custom order.
    optimize : str, optional
        The contraction path optimizer to use.
    sweep_reverse : bool, optional
        Whether to sweep in the reverse direction, resulting in a left
        canonical form instead of right canonical.
    equalize_norms : bool or float, optional
        Whether to equalize the norms of the tensors after compression. If an
        explicit value is give, then the norms will be set to that value, and
        the overall scaling factor will be accumulated into `.exponent`.
    strip_exponent : bool, optional
        If `True`, the intermediate tensors are normalized during contraction,
        and the overall scaling factor will be accumulated into `.exponent`.
    inplace : bool, optional
        Whether to perform the compression inplace or not.
    compress_opts
        Supplied to :func:`~quimb.tensor.tensor_split`.

    Returns
    -------
    TensorNetwork
        The compressed tensor network, with canonical center at
        ``site_tags[0]`` ('right canonical' form) or ``site_tags[-1]`` ('left
        canonical' form) if ``sweep_reverse``.
    """
    if max_bond_oversample is None:
        if max_bond is not None:
            max_bond_oversample = 2 * max_bond
        else:
            warnings.warn(
                "Both `max_bond` and `max_bond_oversample` are None, "
                "typically relying on `cutoff` and `cutoff_oversample` only is"
                " not recommended for this method."
            )
    elif isinstance(max_bond_oversample, float):
        if max_bond is not None:
            # assume a multiplier
            max_bond_oversample = round(max_bond * max_bond_oversample)
        else:
            raise ValueError(
                "`max_bond_oversample` cannot be "
                "a float (a multiplier) if `max_bond` is None."
            )
    else:
        # assume manually supplied, `max_bond` is not needed
        max_bond_oversample = int(max_bond_oversample)

    if cutoff_oversample is None:
        if max_bond is None:
            # fully dynamic mode
            cutoff_oversample = cutoff / 10
        else:
            # assume max_bond limited
            cutoff_oversample = cutoff

    if site_tags is None:
        site_tags = tn.site_tags
    if sweep_reverse:
        site_tags = tuple(reversed(site_tags))

    # yields right canonical form w.r.t site_tags
    tn = tensor_network_1d_compress_zipup(
        tn,
        max_bond=max_bond_oversample,
        cutoff=cutoff_oversample,
        site_tags=site_tags,
        canonize=canonize,
        cutoff_mode=cutoff_mode,
        optimize=optimize,
        sweep_reverse=True,
        equalize_norms=equalize_norms,
        strip_exponent=strip_exponent,
        inplace=inplace,
        **compress_opts,
    )
    # direct sweep in other direction
    return _do_direct_sweep(
        tn,
        site_tags,
        max_bond,
        cutoff,
        cutoff_mode,
        equalize_norms,
        normalize,
        permute_arrays,
        **compress_opts,
    )


def tensor_network_1d_compress_src(
    tn,
    max_bond,
    cutoff=0.0,
    site_tags=None,
    normalize=False,
    noise_mode="separable",
    permute_arrays=True,
    sweep_reverse=False,
    canonize=True,
    equalize_norms=False,
    strip_exponent=False,
    inplace=False,
    **contract_opts,
):
    """Compress any 1D-like tensor network using 'Successive Randomized
    Compression' (SRC) https://arxiv.org/abs/2504.06475.

    Parameters
    ----------
    tn : TensorNetwork
        The tensor network to compress. Every tensor should have exactly one of
        the site tags. Each site can have multiple tensors and output indices.
    max_bond : int
        The maximum bond dimension to compress to.
    cutoff : float, optional
        Unused for SRC, use ``max_bond``.
    site_tags : sequence of str, optional
        The tags to use to group and order the tensors from ``tn``. If not
        given, uses ``tn.site_tags``. The tensor network built will have one
        tensor per site, in the order given by ``site_tags``.
    normalize : bool, optional
        Whether to normalize the final tensor network, making use of the fact
        that the output tensor network is in right canonical form.
    noise_mode : {'separable', 'symmetric', 'joint'}, optional
        How to generate the random noise tensors. 'separable' generates a
        random vector for each outer index. 'symmetric' reuses the same random
        vector for all outer indices on a site. 'joint' generates a single
        random tensor for all outer indices on a site.
    permute_arrays : bool or str, optional
        Whether to permute the array indices of the final tensor network into
        canonical order. If ``True`` will use the default order, otherwise if a
        string this specifies a custom order.
    sweep_reverse : bool, optional
        Whether to sweep in the reverse direction, resulting in a left
        canonical form instead of right canonical.
    canonize : bool, optional
        Whether to pseudo canonicalize the initial tensor network.
    equalize_norms : bool or float, optional
        Whether to equalize the norms of the tensors after compression. If an
        explicit value is give, then the norms will be set to that value, and
        the overall scaling factor will be accumulated into `.exponent`.
    strip_exponent : bool, optional
        If `True`, the intermediate tensors are normalized during contraction,
        and the overall scaling factor will be accumulated into `.exponent`.
    inplace : bool, optional
        Whether to perform the compression inplace or not.
    contract_opts
        Supplied to :func:`~quimb.tensor.tensor_contract`.

    Returns
    -------
    TensorNetwork
        The compressed tensor network, with canonical center at
        ``site_tags[0]`` ('right canonical' form) or ``site_tags[-1]`` ('left
        canonical' form) if ``sweep_reverse``.
    """
    # TODO: customizable noise and seed etc. [ ]

    if not canonize:
        warnings.warn("`canonize=False` is ignored for the `src` method.")

    if max_bond is None:
        raise ValueError("`max_bond` must be given for the `src` method.")

    if cutoff != 0.0:
        warnings.warn(
            "`cutoff` is ignored for the `src` method, use `max_bond` instead."
        )

    contract_opts["drop_tags"] = True

    # batch index for the noise samples
    Bix = rand_uuid()
    if site_tags is None:
        site_tags = tn.site_tags
    if sweep_reverse:
        site_tags = tuple(reversed(site_tags))
    L = len(site_tags)

    # first we segment the tensor network into local sites
    local_tns = [None] * L
    local_inds = [None] * L
    local_bonds = [None] * (L - 1)
    output_inds = tn._outer_inds
    for i in range(L):
        # local network
        local_tns[i] = tni = tn.select(site_tags[i])
        # outer indices found on this site
        local_inds[i] = oset.from_dict(tni.ind_map).intersection(output_inds)
        if i > 0:
            local_bonds[i - 1] = bonds(local_tns[i - 1], tni)

    # first we form the left environment tensors with sampling noise
    left_envs = {}
    for i in range(1, L):
        # get random sampling tensors for the previous site
        if noise_mode == "separable":
            # one stack of noise vectors for each leg on this site
            tws = [
                rand_tensor(
                    shape=(max_bond, tn.ind_size(ix)),
                    inds=(Bix, ix),
                    tags=(site_tags[i - 1],),
                    dist="rademacher",
                )
                for ix in local_inds[i - 1]
            ]
        elif noise_mode == "symmetric":
            # reuse the same noise for all output legs on this site
            shape = (max_bond, tn.ind_size(next(iter(local_inds[i - 1]))))
            data = randn(shape, dist="rademacher")
            tws = [
                Tensor(data=data, inds=(Bix, ix), tags=(site_tags[i - 1],))
                for ix in local_inds[i - 1]
            ]
        elif noise_mode == "joint":
            # one big noise tensor for all output legs on this site
            tws = [
                rand_tensor(
                    shape=(
                        max_bond,
                        *(tn.ind_size(ix) for ix in local_inds[i - 1]),
                    ),
                    inds=(Bix, *local_inds[i - 1]),
                    tags=(site_tags[i - 1],),
                    dist="rademacher",
                )
            ]
        else:
            raise ValueError(
                f"Unknown noise mode {noise_mode!r}, "
                "should be one of 'separable', 'symmetric', or 'joint'."
            )
        # contract it with the previous site and environment
        #
        #     bix
        #      /\       (MPO-MPS example)
        #     |  w
        #     |  |
        #    LE--i--
        #    LE  |    bonds
        #    LE--i--
        #
        ts = [*local_tns[i - 1], *tws]
        if i > 1:
            ts.append(left_envs[i - 1])
        left_envs[i] = tensor_contract(
            *ts,
            output_inds=(
                # batch index is hyper, so we have to specify it
                Bix,
                # other outputs are any shared bonds between sites
                *local_bonds[i - 1],
            ),
            **contract_opts,
        )
        left_envs[i].normalize_()

    # then we sweep in from the right
    Us = [None] * L
    right_envs = {}
    exponent = 0.0
    for i in range(L - 1, 0, -1):
        # contract the environments with the current site
        #
        #   bix
        #     |  |  |
        #    LE--i--RE     (MPO-MPS example)
        #    LE  |  RE
        #    LE--i--RE
        #
        ts = [left_envs[i], *local_tns[i]]
        if i < L - 1:
            ts.append(right_envs[i])
        t = tensor_contract(*ts, **contract_opts)

        # perform QR to get the projector!
        #
        #   bix
        #     |  |  |
        #     R--QQQQ
        #
        tq, _ = t.split(
            left_inds=None,  # auto calc
            right_inds=(Bix,),
            method="qr",
            get="tensors",
        )
        Us[i] = tq

        # now we contract the projector with one more site
        # to form the right environment tensor
        #
        #       QQQQ--
        #       |  |      (MPO-MPS example)
        #     --i--RE
        #       |  RE
        #     --i--RE
        #
        ts = [*local_tns[i], tq.conj()]
        if i < L - 1:
            # include the right environment
            ts.append(right_envs[i])
        if strip_exponent:
            right_envs[i - 1], result_exponent = tensor_contract(*ts, 
                                                                 strip_exponent=strip_exponent, 
                                                                 **contract_opts)
            exponent += result_exponent
        else:
            right_envs[i - 1] = tensor_contract(*ts, **contract_opts)

    # handle the final tensor
    Us[0] = (local_tns[0] | right_envs[0]).contract(all, **contract_opts)

    new = _form_final_tn_from_tensor_sequence(
        tn,
        Us,
        normalize,
        sweep_reverse,
        permute_arrays,
        equalize_norms,
        inplace,
        tags_per_site=[ltn.tags for ltn in local_tns],
    )
    if strip_exponent:
        new.exponent += exponent

    return new


def tensor_network_1d_compress_src_oversample(
    tn,
    max_bond,
    max_bond_oversample=None,
    cutoff=1e-10,
    cutoff_oversample=0.0,
    noise_mode="separable",
    site_tags=None,
    canonize=True,
    normalize=False,
    cutoff_mode="rsum2",
    permute_arrays=True,
    optimize="auto-hq",
    sweep_reverse=False,
    equalize_norms=False,
    strip_exponent=False,
    inplace=False,
    **compress_opts,
):
    """Compress this 1D-like tensor network using the 'src first' algorithm,
    i.e. src with oversampling, that is, first compressing the tensor network
    to a larger bond dimension using the 'src' (successive randomized
    compression, https://arxiv.org/abs/2504.06475) algorithm, then compressing
    to the desired bond dimension using a direct sweep.

    Parameters
    ----------
    tn : TensorNetwork
        The tensor network to compress. Every tensor should have exactly one of
        the site tags. Each site can have multiple tensors and output indices.
    max_bond : int
        The final maximum bond dimension to compress to.
    max_bond_oversample : int, optional
        The intermediate maximum bond dimension to compress to using the
        'src' algorithm. If not given and `max_bond` is, this is set as
        ``max(round(1.5 * max_bond), max_bond + 10)``. If given as a float,
        this is assumed to be a multiplier of `max_bond`.
    cutoff : float, optional
         A dynamic threshold for discarding singular values when compressing
        during the final direct sweep.
    cutoff_oversample : float, optional
        Unused for SRC.
    noise_mode : {'separable', 'symmetric', 'joint'}, optional
        How to generate the random noise tensors for the SRC step.
    site_tags : sequence of str, optional
        The tags to use to group and order the tensors from ``tn``. If not
        given, uses ``tn.site_tags``. The tensor network built will have one
        tensor per site, in the order given by ``site_tags``.
    canonize : bool, optional
        Whether to pseudo canonicalize the initial tensor network.
    normalize : bool, optional
        Whether to normalize the final tensor network, making use of the fact
        that the output tensor network is in right canonical form.
    cutoff_mode : {"rsum2", "rel", ...}, optional
        The mode to use when truncating the singular values of the decomposed
        tensors during the final direct sweep.
    permute_arrays : bool or str, optional
        Whether to permute the array indices of the final tensor network into
        canonical order. If ``True`` will use the default order, otherwise if a
        string this specifies a custom order.
    optimize : str, optional
        The contraction path optimizer to use.
    sweep_reverse : bool, optional
        Whether to sweep in the reverse direction, resulting in a left
        canonical form instead of right canonical.
    equalize_norms : bool or float, optional
        Whether to equalize the norms of the tensors after compression. If an
        explicit value is give, then the norms will be set to that value, and
        the overall scaling factor will be accumulated into `.exponent`.
    strip_exponent : bool, optional
        If `True`, the intermediate tensors are normalized during contraction,
        and the overall scaling factor will be accumulated into `.exponent`.
    inplace : bool, optional
        Whether to perform the compression inplace or not.
    compress_opts
        Supplied to :func:`~quimb.tensor.tensor_split` during the final direct
        sweep.

    Returns
    -------
    TensorNetwork
        The compressed tensor network.
    """
    if max_bond is None:
        raise ValueError(
            "`max_bond` must be given for the `src-oversample` method."
        )

    if max_bond_oversample is None:
        # sensible default oversampling (taken from paper)
        max_bond_oversample = max(round(1.5 * max_bond), max_bond + 10)
    elif isinstance(max_bond_oversample, float):
        # assume a multiplier
        max_bond_oversample = round(max_bond * max_bond_oversample)
    else:
        # assume manually supplied
        max_bond_oversample = int(max_bond_oversample)

    if not canonize:
        warnings.warn(
            "`canonize=False` is ignored for the `src-oversample` method."
        )

    if site_tags is None:
        site_tags = tn.site_tags
    if sweep_reverse:
        site_tags = tuple(reversed(site_tags))

    # yields right canonical form w.r.t site_tags
    tn = tensor_network_1d_compress_src(
        tn,
        max_bond=max_bond_oversample,
        cutoff=cutoff_oversample,
        site_tags=site_tags,
        noise_mode=noise_mode,
        normalize=False,  # handled after direct sweep
        permute_arrays=False,  # handle after direct sweep
        sweep_reverse=True,  # handled above, opposite to direct sweep
        equalize_norms=equalize_norms,
        strip_exponent=strip_exponent,
        optimize=optimize,
        inplace=inplace,
    )
    # direct sweep in other direction
    return _do_direct_sweep(
        tn,
        site_tags,
        max_bond,
        cutoff,
        cutoff_mode,
        equalize_norms,
        normalize,
        permute_arrays,
        **compress_opts,
    )


def tensor_network_1d_compress_srcmps(
    tn,
    max_bond,
    cutoff=0.0,
    tn_fit=None,
    site_tags=None,
    normalize=False,
    permute_arrays=True,
    sweep_reverse=False,
    canonize=True,
    equalize_norms=False,
    inplace=False,
    strip_exponent=False,
    **contract_opts,
):
    """Compress any 1D-like tensor network using 'Successive Randomized
    Compression' (SRC) https://arxiv.org/abs/2504.06475 but using an random or
    supplied MPS as the 'sampling noise'.

    Parameters
    ----------
    tn : TensorNetwork
        The tensor network to compress. Every tensor should have exactly one of
        the site tags. Each site can have multiple tensors and output indices.
    max_bond : int
        The maximum bond dimension to compress to.
    cutoff : float, optional
        Unused for SRC, use ``max_bond``.
    tn_fit : TensorNetwork, dict, or str, optional
        The MPS specificing the sampling noise (its bond dimension effectively
        sets the compression rank). If not given, a random MPS with bond
        dimension ``max_bond`` is used. If a string or dict, this is used to
        construct the MPS from ``tn``.
    site_tags : sequence of str, optional
        The tags to use to group and order the tensors from ``tn``. If not
        given, uses ``tn.site_tags``. The tensor network built will have one
        tensor per site, in the order given by ``site_tags``.
    normalize : bool, optional
        Whether to normalize the final tensor network, making use of the fact
        that the output tensor network is in right canonical form.
    permute_arrays : bool or str, optional
        Whether to permute the array indices of the final tensor network into
        canonical order. If ``True`` will use the default order, otherwise if a
        string this specifies a custom order.
    canonize : bool, optional
        Whether to pseudo canonicalize the initial tensor network.
    sweep_reverse : bool, optional
        Whether to sweep in the reverse direction, resulting in a left
        canonical form instead of right canonical.
    equalize_norms : bool or float, optional
        Whether to equalize the norms of the tensors after compression. If an
        explicit value is give, then the norms will be set to that value, and
        the overall scaling factor will be accumulated into `.exponent`.
    strip_exponent : bool, optional
        If `True`, the intermediate tensors are normalized during contraction,
        and the overall scaling factor will be accumulated into `.exponent`.
    inplace : bool, optional
        Whether to perform the compression inplace or not.
    contract_opts
        Supplied to :func:`~quimb.tensor.tensor_contract`.

    Returns
    -------
    TensorNetwork
        The compressed tensor network.
    """
    # TODO: customizable noise and seed etc.
    # TODO: handle arbitrary outer indices

    if not canonize:
        warnings.warn("`canonize=False` is ignored for the `src` method.")
    if cutoff != 0.0:
        warnings.warn(
            "`cutoff` is ignored for the `src` method, use `max_bond` instead."
        )

    contract_opts["optimize"] = "auto"
    contract_opts["drop_tags"] = True

    if site_tags is None:
        site_tags = tn.site_tags
    if sweep_reverse:
        site_tags = tuple(reversed(site_tags))
    L = len(site_tags)

    local_tns = [tn.select(site_tags[i]) for i in range(L)]

    # parse the sampling MPS
    if not isinstance(tn_fit, TensorNetwork):
        if tn_fit is None:
            tn_fit = TN_matching(
                tn,
                max_bond=max_bond,
                site_tags=site_tags,
            )
        else:
            if isinstance(tn_fit, str):
                tn_fit = {"method": tn_fit}
            tn_fit.setdefault("max_bond", max_bond)
            tn_fit.setdefault("cutoff", cutoff)
            tn_fit.setdefault("site_tags", site_tags)
            tn_fit = tensor_network_1d_compress(tn, **tn_fit)

    # get the bonds along the sampling MPS
    bonds = [
        next(iter(tn_fit[site_tags[i]].bonds(tn_fit[site_tags[i + 1]])))
        for i in range(L - 1)
    ]

    # compute the left environments
    left_envs = {}
    for i in range(1, L):
        # contract it with the previous site and environment
        #
        #    LE--w--    (MPO-MPS example)
        #    LE  |
        #    LE--i--
        #    LE  |
        #    LE--i--
        #
        ts = [*local_tns[i - 1], tn_fit[site_tags[i - 1]]]
        if i >= 2:
            ts.append(left_envs[i - 1])
        left_envs[i] = tensor_contract(*ts, **contract_opts)
        left_envs[i].normalize_()

    # then we sweep in from the right
    Us = [None] * L
    right_envs = {}
    for i in range(L - 1, 0, -1):
        # contract the environments with the current site
        #
        #       bonds[i-1]
        #    LE--
        #    LE   |  |
        #    LE---i--RE     (MPO-MPS example)
        #    LE   |  RE
        #    LE---i--RE
        #
        ts = [left_envs[i], *local_tns[i]]
        if i < L - 1:
            ts.append(right_envs[i])

        t = tensor_contract(*ts, **contract_opts)

        # perform QR to get the projector!
        #
        #   bonds[i-1]
        #     |
        #     |  |  |
        #     R--QQQQ
        #
        tq, _ = t.split(
            left_inds=None,  # auto calc
            right_inds=(bonds[i - 1],),
            method="qr",
            get="tensors",
        )
        Us[i] = tq

        # now we contract the projector with one more site
        # to form the right environment tensor
        #
        #       QQQQ--
        #       |  |      (MPO-MPS example)
        #     --i--RE
        #       |  RE
        #     --i--RE
        #
        ts = [*local_tns[i], tq.conj()]
        if i < L - 1:
            # include the right environment
            ts.append(right_envs[i])
        if strip_exponent:
            right_envs[i - 1], result_exponent = tensor_contract(*ts, 
                                                                 strip_exponent=strip_exponent, 
                                                                 **contract_opts)
            exponent += result_exponent
        else:
            right_envs[i - 1] = tensor_contract(*ts, **contract_opts)

    # handle the final tensor
    Us[0] = (local_tns[0] | right_envs[0]).contract(all, **contract_opts)

    new = _form_final_tn_from_tensor_sequence(
        tn,
        Us,
        normalize,
        sweep_reverse,
        permute_arrays,
        equalize_norms,
        inplace,
        tags_per_site=[ltn.tags for ltn in local_tns],
    )
    if strip_exponent:
        new.exponent += exponent

    return new


def tensor_network_1d_compress_srcmps_oversample(
    tn,
    max_bond,
    max_bond_oversample=None,
    cutoff=1e-10,
    cutoff_oversample=0.0,
    tn_fit=None,
    site_tags=None,
    canonize=True,
    normalize=False,
    cutoff_mode="rsum2",
    permute_arrays=True,
    optimize="auto-hq",
    sweep_reverse=False,
    equalize_norms=False,
    strip_exponent=False,
    inplace=False,
    **compress_opts,
):
    """Compress this 1D-like tensor network using the 'srcmps-oversample'
    algorithm, i.e. src with oversampling and matrix product state noise, that
    is, first compressing the tensor network to a larger bond dimension using
    the 'src' (successive randomized compression,
    https://arxiv.org/abs/2504.06475) algorithm, then compressing to the
    desired bond dimension using a direct sweep.

    Parameters
    ----------
    tn : TensorNetwork
        The tensor network to compress. Every tensor should have exactly one of
        the site tags. Each site can have multiple tensors and output indices.
    max_bond : int
        The final maximum bond dimension to compress to.
    max_bond_oversample : int, optional
        The intermediate maximum bond dimension to compress to using the
        'srcmps' algorithm. If not given and `max_bond` is, this is set as
        ``max(round(1.5 * max_bond), max_bond + 10)``. If given as a float,
        this is assumed to be a multiplier of `max_bond`.
    cutoff : float, optional
         A dynamic threshold for discarding singular values when compressing
        during the final direct sweep.
    cutoff_oversample : float, optional
        Unused for SRC.
    tn_fit : TensorNetwork, optional
        The MPS specificing the sampling noise (its bond dimension effectively
        sets the compression rank).
    site_tags : sequence of str, optional
        The tags to use to group and order the tensors from ``tn``. If not
        given, uses ``tn.site_tags``. The tensor network built will have one
        tensor per site, in the order given by ``site_tags``.
    canonize : bool, optional
        Whether to pseudo canonicalize the initial tensor network.
    normalize : bool, optional
        Whether to normalize the final tensor network, making use of the fact
        that the output tensor network is in right canonical form.
    cutoff_mode : {"rsum2", "rel", ...}, optional
        The mode to use when truncating the singular values of the decomposed
        tensors during the final direct sweep.
    permute_arrays : bool or str, optional
        Whether to permute the array indices of the final tensor network into
        canonical order. If ``True`` will use the default order, otherwise if a
        string this specifies a custom order.
    optimize : str, optional
        The contraction path optimizer to use.
    sweep_reverse : bool, optional
        Whether to sweep in the reverse direction, resulting in a left
        canonical form instead of right canonical.
    equalize_norms : bool or float, optional
        Whether to equalize the norms of the tensors after compression. If an
        explicit value is give, then the norms will be set to that value, and
        the overall scaling factor will be accumulated into `.exponent`.
    strip_exponent : bool, optional
        If `True`, the intermediate tensors are normalized during contraction,
        and the overall scaling factor will be accumulated into `.exponent`.
    inplace : bool, optional
        Whether to perform the compression inplace or not.
    compress_opts
        Supplied to :func:`~quimb.tensor.tensor_split` during the final direct
        sweep.

    Returns
    -------
    TensorNetwork
        The compressed tensor network.
    """
    if max_bond is None:
        raise ValueError(
            "`max_bond` must be given for the `srcmps-oversample` method."
        )

    if max_bond_oversample is None:
        # sensible default oversampling
        max_bond_oversample = max(round(1.5 * max_bond), max_bond + 10)
    elif isinstance(max_bond_oversample, float):
        # assume a multiplier
        max_bond_oversample = round(max_bond * max_bond_oversample)
    else:
        # assume manually supplied
        max_bond_oversample = int(max_bond_oversample)

    if not canonize:
        warnings.warn(
            "`canonize=False` is ignored for the `srcmps-oversample` method."
        )

    if site_tags is None:
        site_tags = tn.site_tags
    if sweep_reverse:
        site_tags = tuple(reversed(site_tags))

    # yields right canonical form w.r.t site_tags
    tn = tensor_network_1d_compress_srcmps(
        tn,
        max_bond=max_bond_oversample,
        cutoff=cutoff_oversample,
        tn_fit=tn_fit,
        site_tags=site_tags,
        normalize=False,  # handled after direct sweep
        permute_arrays=False,  # handle after direct sweep
        sweep_reverse=True,  # handled above, opposite to direct sweep
        equalize_norms=equalize_norms,
        strip_exponent=strip_exponent,
        optimize=optimize,
        inplace=inplace,
    )
    # direct sweep in other direction
    return _do_direct_sweep(
        tn,
        site_tags,
        max_bond,
        cutoff,
        cutoff_mode,
        equalize_norms,
        normalize,
        permute_arrays,
        **compress_opts,
    )


# ---------------------------- fitting methods ------------------------------ #


def _tn1d_fit_sum_sweep_1site(
    tn_fit: TensorNetwork,
    tn_overlaps,
    site_tags,
    max_bond=None,
    cutoff=0.0,
    envs=None,
    prepare=True,
    reverse=False,
    compute_tdiff=True,
    optimize="auto-hq",
):
    """Core sweep of the 1-site 1D fit algorithm."""

    if cutoff != 0.0:
        raise ValueError("Non-zero `cutoff` not supported for 1-site fit.")

    N = len(site_tags)
    K = len(tn_overlaps)

    fermion = tn_fit.isfermionic()
    if fermion:
        raise NotImplementedError(
            "Fermionic 1-site fitting not implemented, use 2-site (bsz=2)."
        )

    if max_bond is not None:
        current_bond_dim = tn_fit.max_bond()
        if current_bond_dim < max_bond:
            tn_fit.expand_bond_dimension_(max_bond)
            prepare = True

    if envs is None:
        envs = {}
        prepare = True

    if prepare:
        for k in range(K):
            envs.setdefault(("L", 0, k), TensorNetwork())
            envs.setdefault(("R", N - 1, k), TensorNetwork())

        if not reverse:
            # move canonical center to left
            tn_fit.canonize_around_(site_tags[0])
            # compute each of K right environments
            for i in reversed(range(N - 1)):
                site_r = site_tags[i + 1]
                for k, tn_overlap in enumerate(tn_overlaps):
                    tni = envs["R", i + 1, k] | tn_overlap.select(site_r)
                    envs["R", i, k] = tni.contract(all, optimize=optimize)
        else:
            # move canonical center to right
            tn_fit.canonize_around_(site_tags[-1])
            # compute each of K left environments
            for i in range(1, N):
                site_l = site_tags[i - 1]
                for k, tn_overlap in enumerate(tn_overlaps):
                    tni = envs["L", i - 1, k] | tn_overlap.select(site_l)
                    envs["L", i, k] = tni.contract(all, optimize=optimize)

    # track the maximum change in any tensor norm
    max_tdiff = -1.0

    sweep = range(N)
    if reverse:
        sweep = reversed(sweep)

    for i in sweep:
        site = site_tags[i]

        if not reverse:
            if i > 0:
                # move canonical center
                site_l = site_tags[i - 1]
                tn_fit.canonize_between(site_l, site)

                # recalculate K left environments
                for k, tn_overlap in enumerate(tn_overlaps):
                    tni = envs["L", i - 1, k] | tn_overlap.select(site_l)
                    envs["L", i, k] = tni.contract(all, optimize=optimize)
        else:
            if i < N - 1:
                # move canonical center
                site_r = site_tags[i + 1]
                tn_fit.canonize_between(site_r, site)

                # recalculate right environment
                for k, tn_overlap in enumerate(tn_overlaps):
                    tni = envs["R", i + 1, k] | tn_overlap.select(site_r)
                    envs["R", i, k] = tni.contract(all, optimize=optimize)

        tfi = tn_fit[site_tags[i]]
        tfinew = None

        for k, tn_overlap in enumerate(tn_overlaps):
            # form local overlap
            tnik = (
                envs["L", i, k]
                | tn_overlap.select_any(site_tags[i])
                | envs["R", i, k]
            )

            # remove old tensor
            del tnik["__FIT__", site]

            # contract its new value, maintaining index order
            tfiknew = tnik.contract(
                all, optimize=optimize, output_inds=tfi.inds
            )

            # sum into fitted tensor
            if tfinew is None:
                tfinew = tfiknew
            else:
                tfinew += tfiknew

        tfinew.conj_()

        if compute_tdiff:
            # track change in tensor norm
            dt = tfi.distance_normalized(tfinew)
            max_tdiff = max(max_tdiff, dt)

        # reinsert into all viewing tensor networks
        tfi.modify(data=tfinew.data)

    return max_tdiff


def _tn1d_fit_sum_sweep_2site(
    tn_fit,
    tn_overlaps,
    site_tags,
    max_bond=None,
    cutoff=1e-10,
    envs=None,
    prepare=True,
    reverse=False,
    optimize="auto-hq",
    compute_tdiff=True,
    **compress_opts,
):
    """Core sweep of the 2-site 1D fit algorithm."""

    N = len(site_tags)
    K = len(tn_overlaps)

    if envs is None:
        envs = {}
        prepare = True

    if prepare:
        for k in range(K):
            envs.setdefault(("L", 0, k), TensorNetwork())
            envs.setdefault(("R", N - 1, k), TensorNetwork())

        if not reverse:
            # move canonical center to left
            tn_fit.canonize_around_(site_tags[0])
            # compute each of K right environments
            for i in range(N - 2, 0, -1):
                site_r = site_tags[i + 1]
                for k, tn_overlap in enumerate(tn_overlaps):
                    tni = envs["R", i + 1, k] | tn_overlap.select(site_r)
                    envs["R", i, k] = tni.contract(all, optimize=optimize)
        else:
            # move canonical center to right
            tn_fit.canonize_around_(site_tags[-1])
            # compute each of K left environments
            for i in range(1, N - 1):
                site_l = site_tags[i - 1]
                for k, tn_overlap in enumerate(tn_overlaps):
                    tni = envs["L", i - 1, k] | tn_overlap.select(site_l)
                    envs["L", i, k] = tni.contract(all, optimize=optimize)

    # track the maximum change in any tensor norm
    max_tdiff = -1.0

    sweep = range(N - 1)
    if reverse:
        sweep = reversed(sweep)

    for i in sweep:
        site0 = site_tags[i]
        site1 = site_tags[i + 1]

        if not reverse:
            if i > 0:
                site_l = site_tags[i - 1]
                # recalculate K left environments
                for k, tn_overlap in enumerate(tn_overlaps):
                    tni = envs["L", i - 1, k] | tn_overlap.select(site_l)
                    envs["L", i, k] = tni.contract(all, optimize=optimize)
        else:
            if i < N - 2:
                site_r = site_tags[i + 2]
                # recalculate right environment
                for k, tn_overlap in enumerate(tn_overlaps):
                    tni = envs["R", i + 2, k] | tn_overlap.select(site_r)
                    envs["R", i + 1, k] = tni.contract(all, optimize=optimize)

        tfi0 = tn_fit[site0]
        tfi1 = tn_fit[site1]
        (bond,) = tfi0.bonds(tfi1)
        left_inds = tuple(ix for ix in tfi0.inds if ix != bond)
        right_inds = tuple(ix for ix in tfi1.inds if ix != bond)
        tfinew = None

        fermion = tfi0.isfermionic()

        for k, tn_overlap in enumerate(tn_overlaps):
            # form local overlap
            tnik = (
                envs["L", i, k]
                | tn_overlap.select_any((site0, site1))
                | envs["R", i + 1, k]
            )

            if fermion:
                # keep track of the environment legs for possible phase flips later
                left_env_ind, right_env_ind = None, None
                if type(envs["L", i, k]) is Tensor:
                    (left_env_ind,) = tfi0.bonds(envs["L", i, k])
                if type(envs["R", i + 1, k]) is Tensor:
                    (right_env_ind,) = tfi1.bonds(envs["R", i + 1, k])

            # remove old tensors
            del tnik["__FIT__", site0]
            del tnik["__FIT__", site1]

            # contract its new value, maintaining index order
            tfiknew = tnik.contract(
                all, optimize=optimize, output_inds=left_inds + right_inds
            )

            if fermion:
                # flip the dual indices of the environment legs if needed
                lind_id, rind_id = None, None
                lind_id = (
                    tfiknew.inds.index(left_env_ind)
                    if left_env_ind is not None
                    else None
                )
                rind_id = (
                    tfiknew.inds.index(right_env_ind)
                    if right_env_ind is not None
                    else None
                )
                if lind_id is not None and rind_id is not None:
                    if tfiknew.data.duals[lind_id]:
                        tfiknew.data.phase_flip(lind_id, inplace=True)
                    else:
                        tfiknew.data.phase_flip(rind_id, inplace=True)
                elif lind_id is not None and rind_id is None:
                    if tfiknew.data.duals[lind_id]:
                        tfiknew.data.phase_flip(lind_id, inplace=True)
                elif lind_id is None and rind_id is not None:
                    if tfiknew.data.duals[rind_id]:
                        tfiknew.data.phase_flip(rind_id, inplace=True)

            # sum into fitted tensor
            if tfinew is None:
                tfinew = tfiknew
            else:
                tfinew += tfiknew

        if fermion:
            # when conjugating tn_fit (a ftn) the dual outer indices will be flipped (phase_dual=True)
            # but ftensor.conj() assumes phase_dual=False by default
            # we need to manually flip the dual 'physical' indices in tfinew before conjugating
            data = tfinew.data
            dual_outer_axs = tuple(
                ax
                for ax, ix in enumerate(tfinew.inds)
                if ix not in (left_env_ind, right_env_ind)
                and data.indices[ax].dual  # or tfixnew.data.duals[ax]
            )
            if dual_outer_axs:
                tfinew.modify(data=data.phase_flip(*dual_outer_axs))

        tfinew.conj_()

        tfinew0, tfinew1 = tfinew.split(
            max_bond=max_bond,
            cutoff=cutoff,
            absorb="left" if reverse else "right",
            left_inds=left_inds,
            right_inds=right_inds,
            bond_ind=bond,
            get="tensors",
            **compress_opts,
        )

        if compute_tdiff:
            # track change in tensor norm
            dt = (tfi0 | tfi1).distance_normalized(tfinew0 | tfinew1)
            max_tdiff = max(max_tdiff, dt)

        # reinsert into all viewing tensor networks
        tfinew0.transpose_like_(tfi0)
        tfinew1.transpose_like_(tfi1)
        tfi0.modify(data=tfinew0.data, left_inds=tfinew0.left_inds)
        tfi1.modify(data=tfinew1.data, left_inds=tfinew1.left_inds)

        if fermion:
            # deal with the global signs generated during conjugation
            for ts in (tfi0 | tfi1).tensors:
                if sum(m.parity for m in ts.data.dummy_modes) % 2 == 1:
                    ts.data.phase_global(inplace=True)

    return max_tdiff


def tensor_network_1d_compress_fit(
    tns,
    max_bond=None,
    cutoff=None,
    tn_fit=None,
    bsz="auto",
    initial_bond_dim=8,
    max_iterations=10,
    tol=0.0,
    site_tags=None,
    cutoff_mode="rsum2",
    sweep_sequence="RL",
    normalize=False,
    permute_arrays=True,
    optimize="auto-hq",
    canonize=True,
    sweep_reverse=False,
    equalize_norms=False,
    inplace_fit=False,
    inplace=False,
    progbar=False,
    **compress_opts,
):
    """Compress any 1D-like (can have multiple tensors per site) tensor network
    or sum of tensor networks to an exactly 1D (one tensor per site) tensor
    network of bond dimension `max_bond` using the 1-site or 2-site variational
    fitting (or 'DMRG-style') method. The tensor network(s) can have arbitrary
    inner and outer structure.

    This method has the lowest scaling of the standard 1D compression methods
    and can also provide the most accurate compression, but the actual speed
    and accuracy depend on the number of iterations required and initial guess,
    making it a more 'hands-on' method.

    It's also the only method to support fitting to a sum of tensor networks
    directly, rather than having to forming the explicitly summed TN first.

    Parameters
    ----------
    tns : TensorNetwork or Sequence[TensorNetwork]
        The tensor network or tensor networks to compress. Each tensor network
        should have the same outer index structure, and within each tensor
        network every tensor should have exactly one of the site tags.
    max_bond : int
        The maximum bond dimension to compress to. If not given, this is set
        as the maximum bond dimension of the initial guess tensor network, if
        any, else infinite for ``bsz=2``.
    cutoff : float, optional
        A dynamic threshold for discarding singular values when compressing.
        This is only relevant for the 2-site sweeping algorithm (``bsz=2``),
        where it defaults to 1e-10.
    tn_fit : TensorNetwork, dict, or str, optional
        An initial guess for the compressed tensor network. It should matching
        outer indices and site tags with ``tn``. If a `dict`, this is assumed
        to be options to supply to `tensor_network_1d_compress` to construct
        the initial guess, inheriting various defaults like `initial_bond_dim`.
        If a string, e.g. ``"zipup"``, this is shorthand for that compression
        method with default settings. If not given, a random 1D tensor network
        will be used.
    bsz : {"auto", 1, 2}, optional
        The size of the block to optimize while sweeping. If ``"auto"``, this
        will be inferred from the value of ``max_bond`` and ``cutoff``.
    initial_bond_dim : int, optional
        The initial bond dimension to use when creating the initial guess. This
        is only relevant if ``tn_fit`` is not given. For each sweep the allowed
        bond dimension is doubled, up to ``max_bond``. For 1-site this occurs
        via explicit bond expansion, while for 2-site it occurs during the
        2-site tensor decomposition.
    max_iterations : int, optional
        The maximum number of variational sweeps to perform.
    tol : float, optional
        The convergence tolerance, in terms of local tensor distance
        normalized. If zero, there will be exactly ``max_iterations`` sweeps.
    site_tags : sequence of str, optional
        The tags to use to group and order the tensors from ``tn``. If not
        given, uses ``tn.site_tags``. The tensor network built will have one
        tensor per site, in the order given by ``site_tags``.
    cutoff_mode : {"rsum2", "rel", ...}, optional
        The mode to use when truncating the singular values of the decomposed
        tensors. See :func:`~quimb.tensor.tensor_split`, if using the 2-site
        sweeping algorithm.
    sweep_sequence : str, optional
        The sequence of sweeps to perform, e.g. ``"LR"`` means first sweep left
        to right, then right to left. The sequence is cycled. The final
        canonical form of the output tensor network depends on the last sweep
        direction and ``sweep_reverse``.
    normalize : bool, optional
        Whether to normalize the final tensor network, making use of the fact
        that the output tensor network is in left or right canonical form.
    permute_arrays : bool or str, optional
        Whether to permute the array indices of the final tensor network into
        canonical order. If ``True`` will use the default order, otherwise if a
        string this specifies a custom order.
    optimize : str, optional
        The contraction path optimizer to use.
    canonize : bool, optional
        Dummy argument to match the signature of other compression methods.
    sweep_reverse : bool, optional
        Whether to sweep in the reverse direction, swapping whether the final
        tensor network is in right or left canonical form, which also depends
        on the last sweep direction.
    equalize_norms : bool or float, optional
        Whether to equalize the norms of the tensors after compression. If an
        explicit value is give, then the norms will be set to that value, and
        the overall scaling factor will be accumulated into `.exponent`.
    inplace_fit : bool, optional
        Whether to perform the compression inplace on the initial guess tensor
        network, ``tn_fit``, if supplied.
    inplace : bool, optional
        Whether to perform the compression inplace on the target tensor network
        supplied, or ``tns[0]`` if a sequence to sum is supplied.
    progbar : bool, optional
        Whether to show a progress bar. Note the progress bar shows the maximum
        change of any single tensor norm, *not* the global change in norm or
        truncation error.
    compress_opts
        Supplied to :func:`~quimb.tensor.tensor_split`, if using the 2-site
        sweeping algorithm.

    Returns
    -------
    TensorNetwork
        The compressed tensor network. Depending on ``sweep_reverse`` and the
        last sweep direction, the canonical center will be at either L:
        ``site_tags[0]`` or R: ``site_tags[-1]``, or the opposite if
        ``sweep_reverse``.
    """
    if not canonize:
        warnings.warn("`canonize=False` is ignored for the `fit` method.")

    if isinstance(tns, TensorNetwork):
        # fit to single tensor network
        tns = (tns,)
    else:
        # fit to sum of tensor networks
        tns = tuple(tns)

    # how to partition the tensor network(s)
    if site_tags is None:
        site_tags = next(
            tn.site_tags for tn in tns if hasattr(tn, "site_tags")
        )

    tns = tuple(
        enforce_1d_like(tn, site_tags=site_tags, inplace=inplace) for tn in tns
    )

    # choose the block size of the sweeping function
    if bsz == "auto":
        if max_bond is not None:
            if (cutoff is None) or (cutoff == 0.0):
                # max_bond specified, no cutoff -> 1-site
                bsz = 1
            else:
                # max_bond and cutoff specified -> 2-site
                bsz = 2
        else:
            if cutoff == 0.0:
                # no max_bond or cutoff -> 1-site
                bsz = 1
            else:
                # no max_bond, but cutoff -> 2-site
                bsz = 2
    f_sweep = {
        1: _tn1d_fit_sum_sweep_1site,
        2: _tn1d_fit_sum_sweep_2site,
    }[bsz]

    if cutoff is None:
        # set default cutoff
        cutoff = 1e-10 if bsz == 2 else 0.0

    if bsz == 2:
        compress_opts["cutoff_mode"] = cutoff_mode

    # choose our initial guess
    if not isinstance(tn_fit, TensorNetwork):
        if max_bond is None:
            if bsz == 1:
                raise ValueError(
                    "Need to specify at least one of `max_bond` "
                    "or `tn_fit` when using 1-site sweeping."
                )
            max_bond = float("inf")
            current_bond_dim = initial_bond_dim
        else:
            # don't start larger than the target bond dimension
            current_bond_dim = min(initial_bond_dim, max_bond)

        # if we are only doing a small number of iterations, we need to make
        # sure the doubling logic can actually reach max_bond
        max_increase = 2 ** (max_iterations)
        current_bond_dim = max(
            current_bond_dim,
            max_bond // max_increase + bool(max_bond % max_increase),
        )

        if tn_fit is None:
            # random initial guess
            if max_iterations <= 1:
                # no need to generate a random guess and expand it
                current_bond_dim = max_bond

            tn_fit = TN_matching(
                tns[0], max_bond=current_bond_dim, site_tags=site_tags
            )
        else:
            if isinstance(tn_fit, str):
                tn_fit = {"method": tn_fit}
            tn_fit.setdefault("max_bond", current_bond_dim)
            tn_fit.setdefault("cutoff", cutoff)
            tn_fit.setdefault("site_tags", site_tags)
            tn_fit.setdefault("optimize", optimize)
            tn_fit = tensor_network_1d_compress(tns[0], **tn_fit)
            # we generated it so we can always do inplace fitting
            inplace_fit = True
    else:
        # a guess was supplied
        current_bond_dim = tn_fit.max_bond()
        if max_bond is None:
            # assume we want to limit bond dimension to the initial guess
            max_bond = current_bond_dim

    # choose to conjugate the smaller, fitting network
    tn_fit = tn_fit.conj(inplace=inplace_fit)
    tn_fit.add_tag("__FIT__")
    # note these are all views of `tn_fit` and thus will update as it does
    tn_overlaps = [(tn_fit | tn) for tn in tns]

    if any(tn_overlap.outer_inds() for tn_overlap in tn_overlaps):
        raise ValueError(
            "The outer indices of one or more of "
            "`tns` and `tn_fit` don't seem to match."
        )

    sweeps = itertools.cycle(sweep_sequence)
    if max_iterations is None:
        its = itertools.count()
    else:
        its = range(max_iterations)

    envs = {}
    old_direction = ""

    if progbar:
        from quimb.utils import progbar as ProgBar

        its = ProgBar(its, total=max_iterations)

    # whether to compute the maximum change in tensor norm
    compute_tdiff = (tol != 0.0) or progbar

    try:
        for i in its:
            next_direction = next(sweeps)
            reverse = {"R": False, "L": True}[next_direction]
            if sweep_reverse:
                reverse = not reverse

            if current_bond_dim < max_bond:
                # double bond dimension, up to max_bond
                current_bond_dim = min(2 * current_bond_dim, max_bond)

            # perform a single sweep
            max_tdiff = f_sweep(
                tn_fit,
                tn_overlaps,
                max_bond=current_bond_dim,
                cutoff=cutoff,
                envs=envs,
                prepare=(i == 0) or (next_direction == old_direction),
                site_tags=site_tags,
                reverse=reverse,
                optimize=optimize,
                compute_tdiff=compute_tdiff,
                **compress_opts,
            )

            if progbar:
                its.set_description(f"max_tdiff={max_tdiff:.2e}")
            if tol != 0.0 and max_tdiff < tol:
                # converged
                break

            old_direction = next_direction
    except KeyboardInterrupt:
        pass
    finally:
        if progbar:
            its.close()

    tn_fit.drop_tags("__FIT__")
    tn_fit.conj_()

    if normalize:
        if reverse:
            tn_fit[site_tags[0]].normalize_()
        else:
            tn_fit[site_tags[-1]].normalize_()

    if inplace:
        tn0 = tns[0]
        tn0.remove_all_tensors()
        tn0.add_tensor_network(
            tn_fit, virtual=not inplace_fit, check_collisions=False
        )
        tn_fit = tn0

    # possibly put the array indices in canonical order (e.g. when MPS or MPO)
    possibly_permute_(tn_fit, permute_arrays)

    # XXX: do better than simply waiting til the end to equalize norms
    if equalize_norms is True:
        tn_fit.equalize_norms_()
    elif equalize_norms:
        tn_fit.equalize_norms_(value=equalize_norms)

    return tn_fit


def tensor_network_1d_compress_fit_guess(
    tn,
    guess,
    max_bond=None,
    cutoff=1e-10,
    cutoff_fit=0.0,
    bsz=1,
    max_iterations=8,
    canonize=True,
    **kwargs,
):
    """Compress any 1D-like (can have multiple tensors per site) tensor network
    to an exactly 1D (one tensor per site) tensor network of bond dimension
    `max_bond` using by default 1-site variational fitting (or 'DMRG-style')
    method starting with a non-random guess tensor network, e.g. from the cheap
    zip-up or projector methods.

    Parameters
    ----------
    tn : TensorNetwork
        The tensor network to compress. Every tensor should have exactly one of
        the site tags. Each site can have multiple tensors and output indices.
    guess : str
        The method to use to generate the initial guess.
    max_bond : int
        The maximum bond dimension to compress to.
    cutoff : float, optional
        The cutoff used for the *guess* compression method.
    cutoff_fit : float, optional
        The cutoff used for the *fit* compression method.
    bsz : int, optional
        The block size to use for the variational fitting sweep.
    max_iterations : int, optional
        The maximum number of variational sweeps to perform.
    canonize : bool, optional
        Whether to canonicalize the guess tensor network.
    kwargs
        Supplied to :func:`tensor_network_1d_compress_fit`.

    Returns
    -------
    TensorNetwork
        The compressed tensor network.
    """
    tn_fit = {
        "method": guess,
        # use cutoff in guess, but not in fitting
        "cutoff": cutoff,
        "canonize": canonize,
    }

    return tensor_network_1d_compress_fit(
        tn,
        max_bond=max_bond,
        cutoff=cutoff_fit,
        tn_fit=tn_fit,
        bsz=bsz,
        max_iterations=max_iterations,
        inplace_fit=True,
        **kwargs,
    )


tensor_network_1d_compress_fit_zipup = functools.partial(
    tensor_network_1d_compress_fit_guess, guess="zipup"
)
tensor_network_1d_compress_fit_projector = functools.partial(
    tensor_network_1d_compress_fit_guess, guess="projector"
)


def tensor_network_1d_compress_fit_oversample(
    tn,
    max_bond=None,
    max_bond_oversample=None,
    cutoff=1e-10,
    cutoff_fit=0.0,
    bsz=1,
    site_tags=None,
    canonize=True,
    normalize=False,
    cutoff_mode="rsum2",
    permute_arrays=True,
    optimize="auto-hq",
    sweep_reverse=False,
    equalize_norms=False,
    inplace=False,
    **compress_opts,
):
    """Compress this 1D-like tensor network using the 'fit-oversample'
    algorithm, that is, first compressing the tensor network to a larger bond
    dimension using the variational 'fit' algorithm, then compressing to the
    desired bond dimension using a direct sweep.

    Parameters
    ----------
    tn : TensorNetwork
        The tensor network to compress. Every tensor should have exactly one of
        the site tags. Each site can have multiple tensors and output indices.
    max_bond : int
        The final maximum bond dimension to compress to.
    max_bond_oversample : int, optional
        The intermediate maximum bond dimension to compress to using the
        'fit' algorithm. If not given and `max_bond` is, this is set as
        ``max(round(1.5 * max_bond), max_bond + 10)``. If given as a float,
        this is assumed to be a multiplier of `max_bond`.
    cutoff : float, optional
        A dynamic threshold for discarding singular values when compressing
        during the final direct sweep.
    cutoff_fit : float, optional
        A dynamic threshold for discarding singular values when compressing to
        the intermediate bond dimension using the 'fit' algorithm (only
        relevant if ``bsz=2``).
    bsz : {1, 2}, optional
        The block size to use for the variational fitting sweep.
    site_tags : sequence of str, optional
        The tags to use to group and order the tensors from ``tn``. If not
        given, uses ``tn.site_tags``. The tensor network built will have one
        tensor per site, in the order given by ``site_tags``.
    canonize : bool, optional
        Whether to pseudo canonicalize the initial tensor network.
    normalize : bool, optional
        Whether to normalize the final tensor network, making use of the fact
        that the output tensor network is in right canonical form.
    cutoff_mode : {"rsum2", "rel", ...}, optional
        The mode to use when truncating the singular values of the decomposed
        tensors. See :func:`~quimb.tensor.tensor_split`.
    permute_arrays : bool or str, optional
        Whether to permute the array indices of the final tensor network into
        canonical order. If ``True`` will use the default order, otherwise if a
        string this specifies a custom order.
    optimize : str, optional
        The contraction path optimizer to use.
    sweep_reverse : bool, optional
        Whether to sweep in the reverse direction, resulting in a left
        canonical form instead of right canonical.
    equalize_norms : bool or float, optional
        Whether to equalize the norms of the tensors after compression. If an
        explicit value is give, then the norms will be set to that value, and
        the overall scaling factor will be accumulated into `.exponent`.
    inplace : bool, optional
        Whether to perform the compression inplace or not.
    compress_opts
        Supplied to :func:`~quimb.tensor.tensor_split`.

    Returns
    -------
    TensorNetwork
        The compressed tensor network, with canonical center at
        ``site_tags[0]`` ('right canonical' form) or ``site_tags[-1]`` ('left
        canonical' form) if ``sweep_reverse``.
    """
    if max_bond is None:
        raise ValueError(
            "`max_bond` must be given for the `fit-oversample` method."
        )

    if max_bond_oversample is None:
        # sensible default oversampling
        max_bond_oversample = max(round(1.5 * max_bond), max_bond + 10)
    elif isinstance(max_bond_oversample, float):
        # assume a multiplier
        max_bond_oversample = round(max_bond * max_bond_oversample)
    else:
        # assume manually supplied
        max_bond_oversample = int(max_bond_oversample)

    if not canonize:
        warnings.warn(
            "`canonize=False` is ignored for the `fit-oversample` method."
        )

    if site_tags is None:
        site_tags = tn.site_tags
    if sweep_reverse:
        site_tags = tuple(reversed(site_tags))

    # yields right canonical form w.r.t site_tags
    tn = tensor_network_1d_compress_fit(
        tn,
        max_bond=max_bond_oversample,
        cutoff=cutoff_fit,
        bsz=bsz,
        max_iterations=1,
        site_tags=site_tags,
        normalize=False,  # handled after direct sweep
        permute_arrays=False,  # handle after direct sweep
        sweep_sequence="R",  # to ensure right canonical form
        sweep_reverse=False,  # handled above, opposite to direct sweep
        equalize_norms=equalize_norms,
        optimize=optimize,
        inplace=inplace,
    )

    # direct sweep in other direction
    return _do_direct_sweep(
        tn,
        site_tags,
        max_bond,
        cutoff,
        cutoff_mode,
        equalize_norms,
        normalize,
        permute_arrays,
        **compress_opts,
    )


# ------------------------ interface to all methods ------------------------- #


_TN1D_COMPRESS_METHODS = {
    "direct": tensor_network_1d_compress_direct,
    "dm": tensor_network_1d_compress_dm,
    "zipup": tensor_network_1d_compress_zipup,
    "zipup-first": tensor_network_1d_compress_zipup_oversample,
    "zipup-oversample": tensor_network_1d_compress_zipup_oversample,
    "src": tensor_network_1d_compress_src,
    "src-first": tensor_network_1d_compress_src_oversample,
    "src-oversample": tensor_network_1d_compress_src_oversample,
    "srcmps": tensor_network_1d_compress_srcmps,
    "srcmps-first": tensor_network_1d_compress_srcmps_oversample,
    "srcmps-oversample": tensor_network_1d_compress_srcmps_oversample,
    "fit": tensor_network_1d_compress_fit,
    "fit-zipup": tensor_network_1d_compress_fit_zipup,
    "fit-projector": tensor_network_1d_compress_fit_projector,
    "fit-oversample": tensor_network_1d_compress_fit_oversample,
}


def tensor_network_1d_compress(
    tn,
    max_bond=None,
    cutoff=1e-10,
    method="dm",
    site_tags=None,
    canonize=True,
    permute_arrays=True,
    optimize="auto-hq",
    sweep_reverse=False,
    equalize_norms=False,
    compress_opts=None,
    inplace=False,
    **kwargs,
):
    """Compress a 1D-like tensor network using the specified method.

    Parameters
    ----------
    tn : TensorNetwork
        The tensor network to compress. Every tensor should have exactly one of
        the site tags. Each site can have multiple tensors and output indices.
    max_bond : int
        The maximum bond dimension to compress to.
    cutoff : float, optional
        A dynamic threshold for discarding singular values when compressing.
    method : str, optional
        The compression method to use. The options are:

        - ``"direct"`` : direct SVD sweep.
        - ``"dm"`` : density matrix method.
        - ``"zipup"`` : zip-up method.
        - ``"zipup-first"`` or ``"zipup-oversample"`` : zip-up with
          oversampling.
        - ``"src"``: successive randomized compression method.
        - ``"src-first"`` or ``"src-oversample"`` : successive
          randomized compression with oversampling.
        - ``"srcmps"``: successive randomized compression method using an MPS
          as the sampling noise.
        - ``"srcmps-first"`` or ``"srcmps-oversample"`` : successive
          randomized compression using an MPS as the sampling noise with
          oversampling.
        - ``"fit"`` : variational fitting (DMRG-style) method.
        - ``"fit-{method}"`` : variational fitting starting from a guess
          generated using `{method}`, e.g. ``"fit-zipup"``.
        - ``"fit-oversample"`` : variational fitting with oversampling.
        - any other method supported by :func:`tensor_network_ag_compress`.

        Oversampling methods first compress to `max_bond_oversample` (typically
        chosen as 1.5 or 2x `max_bond`) using the specified method, then
        compress to `max_bond` using a direct sweep.
    site_tags : sequence of str, optional
        The tags to use to group and order the tensors from ``tn``. If not
        given, uses ``tn.site_tags``. The tensor network built will have one
        tensor per site, in the order given by ``site_tags``.
    canonize : bool, optional
        Whether to perform canonicalization, pseudo or otherwise depending on
        the method, before compressing. Ignored for ``method='dm'`` and
        ``method='fit'``.
    permute_arrays : bool or str, optional
        Whether to permute the array indices of the final tensor network into
        canonical order. If ``True`` will use the default order, otherwise if a
        string this specifies a custom order.
    optimize : str, optional
        The contraction path optimizer to use.
    sweep_reverse : bool, optional
        Whether to sweep in the reverse direction, resulting in a left
        canonical form instead of right canonical (for the fit method, this
        also depends on the last sweep direction).
    equalize_norms : bool or float, optional
        Whether to equalize the norms of the tensors after compression. If an
        explicit value is give, then the norms will be set to that value, and
        the overall scaling factor will be accumulated into `.exponent`.
    inplace : bool, optional
        Whether to perform the compression inplace.
    kwargs
        Supplied to the chosen compression method.

    Returns
    -------
    TensorNetwork
    """
    compress_opts = compress_opts or {}

    f_tn1d = _TN1D_COMPRESS_METHODS.get(method, None)
    if f_tn1d is not None:
        # 1D specific compression methods
        return f_tn1d(
            tn,
            max_bond=max_bond,
            cutoff=cutoff,
            site_tags=site_tags,
            canonize=canonize,
            permute_arrays=permute_arrays,
            optimize=optimize,
            sweep_reverse=sweep_reverse,
            equalize_norms=equalize_norms,
            inplace=inplace,
            **compress_opts,
            **kwargs,
        )

    # generic tensor network compression methods
    if sweep_reverse:
        warnings.warn(
            "sweep_reverse has no effect for arbitrary geometry (AG) methods."
        )

    tnc = tensor_network_ag_compress(
        tn,
        max_bond=max_bond,
        cutoff=cutoff,
        method=method,
        site_tags=site_tags,
        canonize=canonize,
        optimize=optimize,
        equalize_norms=equalize_norms,
        inplace=inplace,
        **compress_opts,
        **kwargs,
    )

    if permute_arrays:
        possibly_permute_(tnc, permute_arrays)

    return tnc


# --------------- MPO-MPS gating using 1D compression methods --------------- #


def mps_gate_with_mpo_lazy(mps, mpo, inplace=False):
    """Apply an MPO to an MPS lazily, i.e. nothing is contracted, but the new
    TN object has the same outer indices as the original MPS.
    """
    return tensor_network_apply_op_vec(
        A=mpo, x=mps, contract=False, inplace=inplace
    )


def mps_gate_with_mpo_direct(
    mps,
    mpo,
    max_bond=None,
    cutoff=1e-10,
    inplace=False,
    **compress_opts,
):
    """Apply an MPO to an MPS using the boundary compression method, that is,
    explicitly contracting site-wise to form a MPS-like TN, canonicalizing in
    one direction, then compressing in the other. This has the same scaling as
    the density matrix (dm) method, but a larger prefactor. It can still be
    faster for small bond dimensions however, and is potentially higher
    precision since it works in the space of singular values directly rather
    than singular values squared. It is not quite optimal in terms of error due
    to the compounding errors of the SVDs.

    Parameters
    ----------
    mps : MatrixProductState
        The MPS to gate.
    mpo : MatrixProductOperator
        The MPO to gate with.
    max_bond : int
        The maximum bond dimension to compress to.
    cutoff : float, optional
        A dynamic threshold for discarding singular values when compressing.
    compress_opts
        Supplied to :func:`~quimb.tensor.tensor_split`.
    """
    # form the double layer tensor network
    tn = mps_gate_with_mpo_lazy(mps, mpo, inplace=inplace)

    # directly compress it without first contracting site-wise
    return tensor_network_1d_compress_direct(
        tn,
        max_bond=max_bond,
        cutoff=cutoff,
        inplace=inplace,
        **compress_opts,
    )


def mps_gate_with_mpo_dm(
    mps,
    mpo,
    max_bond=None,
    cutoff=1e-10,
    inplace=False,
    **compress_opts,
):
    """Gate this MPS with an MPO, using the density matrix compression method.

    Parameters
    ----------
    mps : MatrixProductState
        The MPS to gate.
    mpo : MatrixProductOperator
        The MPO to gate with.
    max_bond : int, optional
        The maximum bond dimension to keep when compressing the double layer
        tensor network, if any.
    cutoff : float, optional
        The truncation error to use when compressing the double layer tensor
        network, if any.
    compress_opts
        Supplied to :func:`~quimb.tensor.tensor_split`.
    """
    # form the double layer tensor network
    tn = mps_gate_with_mpo_lazy(mps, mpo, inplace=inplace)

    # directly compress it without first contracting site-wise
    return tensor_network_1d_compress_dm(
        tn, max_bond, cutoff, inplace=inplace, **compress_opts
    )


def mps_gate_with_mpo_zipup(
    mps,
    mpo,
    max_bond=None,
    cutoff=1e-10,
    canonize=True,
    optimize="auto-hq",
    **compress_opts,
):
    """Apply an MPO to an MPS using the 'zip-up' algorithm due to
    'Minimally Entangled Typical Thermal State Algorithms', E.M. Stoudenmire &
    Steven R. White (https://arxiv.org/abs/1002.1305).

    Parameters
    ----------
    mps : MatrixProductState
        The MPS to gate.
    mpo : MatrixProductOperator
        The MPO to gate with.
    max_bond : int
        The maximum bond dimension to compress to.
    cutoff : float, optional
        A dynamic threshold for discarding singular values when compressing.
    site_tags : sequence of str, optional
        The tags to use to group and order the tensors from ``tn``. If not
        given, uses ``tn.site_tags``. The tensor network built will have one
        tensor per site, in the order given by ``site_tags``.
    canonize : bool, optional
        Whether to pseudo canonicalize the initial tensor network.
    normalize : bool, optional
        Whether to normalize the final tensor network, making use of the fact
        that the output tensor network is in right canonical form.
    permute_arrays : bool or str, optional
        Whether to permute the array indices of the final tensor network into
        canonical order. If ``True`` will use the default order, otherwise if a
        string this specifies a custom order.
    optimize : str, optional
        The contraction path optimizer to use.
    compress_opts
        Supplied to :func:`~quimb.tensor.tensor_split`.

    Returns
    -------
    MatrixProductState
        The compressed MPS, in right canonical form.
    """
    # form double layer
    tn = mps_gate_with_mpo_lazy(mps, mpo)

    # compress it using zip-up
    return tensor_network_1d_compress_zipup(
        tn,
        max_bond=max_bond,
        cutoff=cutoff,
        canonize=canonize,
        optimize=optimize,
        **compress_opts,
    )


def mps_gate_with_mpo_zipup_first(
    mps,
    mpo,
    max_bond=None,
    max_bond_oversample=None,
    cutoff=1e-10,
    cutoff_oversample=None,
    canonize=True,
    optimize="auto-hq",
    **compress_opts,
):
    """Apply an MPO to an MPS by first using the zip-up method with a larger
    bond dimension, then doing a regular compression sweep to the target final
    bond dimension. This avoids forming an intermediate MPS with bond dimension
    ``mps.max_bond() * mpo.max_bond()``.

    Parameters
    ----------
    mps : MatrixProductState
        The MPS to gate.
    mpo : MatrixProductOperator
        The MPO to gate with.
    max_bond : int
        The target final bond dimension.
    max_bond_oversample : int, optional
        The maximum bond dimension to use when zip-up compressing the double
        layer tensor network. If not given, defaults to ``2 * max_bond``.
        Needs to be smaller than ``mpo.max_bond()`` for any savings.
    cutoff : float, optional
        The truncation error to use when performing the final regular
        compression sweep.
    cutoff_oversample : float, optional
        The truncation error to use when performing the zip-up compression.
    canonize : bool, optional
        Whether to pseudo canonicalize the initial tensor network.
    optimize : str, optional
        The contraction path optimizer to use.
    compress_opts
        Supplied to :func:`~quimb.tensor.tensor_split` (both the zip-up and
        final sweep).

    Returns
    -------
    MatrixProductState
        The compressed MPS, in right canonical form.
    """
    new = mps_gate_with_mpo_lazy(mps, mpo)
    return tensor_network_1d_compress_zipup_oversample(
        new,
        max_bond=max_bond,
        max_bond_oversample=max_bond_oversample,
        cutoff=cutoff,
        cutoff_oversample=cutoff_oversample,
        canonize=canonize,
        optimize=optimize,
        **compress_opts,
    )


def mps_gate_with_mpo_fit(mps, mpo, max_bond, **kwargs):
    """Gate an MPS with an MPO using the variational fitting or DMRG-style
    method.

    Parameters
    ----------
    mps : MatrixProductState
        The MPS to gate.
    mpo : MatrixProductOperator
        The MPO to gate with.
    max_bond : int
        The maximum bond dimension to compress to.

    Returns
    -------
    MatrixProductState
        The gated MPS.
    """
    tn = mps_gate_with_mpo_lazy(mps, mpo)
    return tensor_network_1d_compress_fit(tn, max_bond, **kwargs)


def mps_gate_with_mpo_autofit(
    self,
    mpo,
    max_bond,
    cutoff=0.0,
    init_guess=None,
    **fit_opts,
):
    """Fit a MPS to a MPO applied to an MPS using geometry generic versions
    of either ALS or autodiff. This is usually much less efficient that using
    the 1D specific methods.

    Some nice alternatives to the default fit_opts:

        - method="autodiff"
        - method="als", solver="lstsq"

    """
    if cutoff != 0.0:
        raise ValueError("cutoff must be zero for fitting")

    target = mps_gate_with_mpo_lazy(self, mpo)

    if init_guess is None:
        ansatz = self.copy()
        ansatz.expand_bond_dimension_(max_bond)
    else:
        raise NotImplementedError

    return ansatz.fit_(target, **fit_opts)


def mps_gate_with_mpo_projector(
    self,
    mpo,
    max_bond,
    cutoff=1e-10,
    canonize=True,
    canonize_opts=None,
    inplace=False,
    **compress_opts,
):
    """Apply an MPO to an MPS using local projectors, in the style of CTMRG
    or HOTRG, without using information beyond the neighboring 4 tensors.
    """
    tn = mps_gate_with_mpo_lazy(self, mpo)

    if canonize:
        # precondition
        canonize_opts = ensure_dict(canonize_opts)
        tn.gauge_all_(**canonize_opts)

    tn_calc = tn.copy()

    for i in range(tn.L - 1):
        ltags = (tn.site_tag(i),)
        rtags = (tn.site_tag(i + 1),)

        tn_calc.insert_compressor_between_regions_(
            ltags,
            rtags,
            new_ltags=ltags,
            new_rtags=rtags,
            max_bond=max_bond,
            cutoff=cutoff,
            insert_into=tn,
            bond_ind=self.bond(i, i + 1),
            **compress_opts,
        )

    if inplace:
        for i in range(tn.L):
            ti = self[i]
            data = tensor_contract(
                *tn[i], output_inds=ti.inds, optimize="auto-hq"
            ).data
            ti.modify(data=data)

    else:
        for i in range(tn.L):
            tn.contract_tags_(
                tn.site_tag(i),
                output_inds=self[i].inds,
                optimize="auto-hq",
            )

        tn.view_like_(self)

    return tn
