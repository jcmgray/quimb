"""Generic methods for compressing arbitrary geometry tensor networks, where
the tensor network can locally have arbitrary structure and outer indices.

- [x] projector
- [x] l2bp
- [x] local early
- [x] local late
- [x] superorthogonal

"""

from ..utils import ensure_dict
from .tensor_arbgeom import create_lazy_edge_map
from .tensor_core import choose_local_compress_gauge_settings


def tensor_network_ag_compress_projector(
    tn,
    max_bond=None,
    cutoff=1e-10,
    site_tags=None,
    canonize=True,
    canonize_opts=None,
    lazy=False,
    optimize="auto-hq",
    equalize_norms=False,
    inplace=False,
    **compress_opts,
):
    """Compress an arbtrary geometry tensor network, with potentially multiple
    tensors per site, using locally computed projectors.

    Very loosely, this is like a generalization HOTRG.

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
        The tags to use to group the tensors from ``tn``. If not
        given, uses ``tn.site_tags``. The tensor network built will have one
        tensor per site.
    canonize : bool, optional
        Whether to pseudo canonicalize the initial tensor network.
    canonize_opts
        Supplied to :meth:`~quimb.tensor.tensor_core.TensorNetwork.gauge_all`.
    lazy : bool, optional
        Whether to leave the computed projectors uncontracted, default: False.
    optimize : str, optional
        The contraction path optimizer to use.
    equalize_norms : bool or float, optional
        Whether to equalize the norms of the tensors after compression. If an
        explicit value is give, then the norms will be set to that value, and
        the overall scaling factor will be accumulated into `.exponent`.
    inplace : bool, optional
        Whether to perform the compression inplace.
    compress_opts
        Supplied to :func:`~quimb.tensor.tensor_split`.

    Returns
    -------
    TensorNetwork
    """
    tn = tn if inplace else tn.copy()

    if site_tags is None:
        site_tags = tn.site_tags

    edges, _ = create_lazy_edge_map(tn, site_tags)

    if canonize:
        # optionally precondition the uncontracted network
        canonize_opts = ensure_dict(canonize_opts)
        tn.gauge_all_(
            equalize_norms=equalize_norms,
            **canonize_opts
        )

    # then compute projectors using local information
    tn_calc = tn.copy()
    for taga, tagb in edges:
        tn_calc.insert_compressor_between_regions_(
            [taga],
            [tagb],
            max_bond=max_bond,
            cutoff=cutoff,
            insert_into=tn,
            new_ltags=[taga],
            new_rtags=[tagb],
            optimize=optimize,
            **compress_opts,
        )

    if not lazy:
        # then contract each site with all surrounding projectors
        for st in site_tags:
            tn.contract_(st, optimize=optimize)

    # XXX: do better than simply waiting til the end to equalize norms
    if equalize_norms is True:
        tn.equalize_norms_()
    elif equalize_norms:
        tn.equalize_norms_(value=equalize_norms)

    return tn


def tensor_network_ag_compress_local_early(
    tn,
    max_bond=None,
    cutoff=1e-10,
    site_tags=None,
    canonize=True,
    tree_gauge_distance=None,
    canonize_distance=None,
    canonize_after_distance=None,
    mode="auto",
    optimize="auto-hq",
    equalize_norms=False,
    inplace=False,
    **compress_opts,
):
    """Compress an arbtrary geometry tensor network, with potentially multiple
    tensors per site, using explicit contraction followed by immediate
    ('early') compression. In other words, contractions are interleaved with
    compressions.

    Very loosely, this is like a generalization of the 'zip-up' algorithm in
    1D, but for arbitrary geometry.


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
        The tags to use to group the tensors from ``tn``. If not
        given, uses ``tn.site_tags``. The tensor network built will have one
        tensor per site.
    canonize : bool, optional
        Whether to locally gauge before each compression, defaults to True.
    tree_gauge_distance : int, optional
        The distance to locally gauge to before each compression. Defaults to
        3.
    canonize_distance : int, optional
        The distance to canonize to before each compression, by default this
        is set by ``tree_gauge_distance``.
    canonize_after_distance : int, optional
        The distance to canonize to after each compression, by default this
        is set by ``tree_gauge_distance``, depending on ``mode``.
    mode : {'auto', 'basic', 'virtual-tree', ...}, optional
        The mode to use for the local gauging. If 'auto' will default to
        virtual tree gauging, or basic if `tree_gauge_distance` is 0.
    optimize : str, optional
        The contraction path optimizer to use.
    equalize_norms : bool or float, optional
        Whether to equalize the norms of the tensors after compression. If an
        explicit value is give, then the norms will be set to that value, and
        the overall scaling factor will be accumulated into `.exponent`.
    inplace : bool, optional
        Whether to perform the compression inplace.
    compress_opts
        Supplied to
        :meth:`~quimb.tensor.tensor_core.TensorNetwork.compress_between`.

    Returns
    -------
    TensorNetwork
    """
    tnc = tn if inplace else tn.copy()

    if site_tags is None:
        site_tags = tnc.site_tags

    _, neighbors = create_lazy_edge_map(tnc, site_tags)

    canonize_distance, canonize_after_distance, mode = (
        choose_local_compress_gauge_settings(
            canonize,
            tree_gauge_distance,
            canonize_distance,
            canonize_after_distance,
            mode,
        )
    )

    st0 = next(iter(site_tags))
    seen = {st0}
    queue = [st0]

    while queue:
        # process sites in a breadth-first manner
        taga = queue.pop(0)

        for tagb in neighbors[taga]:
            if tagb not in seen:
                queue.append(tagb)
                seen.add(tagb)

        # contract this site
        tnc.contract_(taga, optimize=optimize)

        # then immediately compress around it
        (tida,) = tnc._get_tids_from_tags(taga)
        for tidb in tnc._get_neighbor_tids(tida):
            tnc._compress_between_tids(
                tida,
                tidb,
                max_bond=max_bond,
                cutoff=cutoff,
                canonize_distance=canonize_distance,
                canonize_after_distance=canonize_after_distance,
                mode=mode,
                **compress_opts,
            )

    if equalize_norms is True:
        tnc.equalize_norms_()
    elif equalize_norms:
        tnc.equalize_norms_(value=equalize_norms)

    return tnc


def tensor_network_ag_compress_local_late(
    tn,
    max_bond=None,
    cutoff=1e-10,
    site_tags=None,
    canonize=True,
    tree_gauge_distance=None,
    canonize_distance=None,
    canonize_after_distance=None,
    mode="auto",
    optimize="auto-hq",
    equalize_norms=False,
    inplace=False,
    **compress_opts,
):
    """Compress an arbtrary geometry tensor network, with potentially multiple
    tensors per site, by explicitly contracting all sites first and then
    ('late') locally compressing. In other words, all contractions happen, then
    all compressions happen.

    Very loosely, this is like a generalization of the 'direct' algorithm in
    1D, but for arbitrary geometry.

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
        The tags to use to group the tensors from ``tn``. If not
        given, uses ``tn.site_tags``. The tensor network built will have one
        tensor per site.
    canonize : bool, optional
        Whether to locally gauge before each compression, defaults to True.
    tree_gauge_distance : int, optional
        The distance to locally gauge to before each compression. Defaults to
        3.
    canonize_distance : int, optional
        The distance to canonize to before each compression, by default this
        is set by ``tree_gauge_distance``.
    canonize_after_distance : int, optional
        The distance to canonize to after each compression, by default this
        is set by ``tree_gauge_distance``, depending on ``mode``.
    mode : {'auto', 'basic', 'virtual-tree', ...}, optional
        The mode to use for the local gauging. If 'auto' will default to
        virtual tree gauging, or basic if `tree_gauge_distance` is 0.
    optimize : str, optional
        The contraction path optimizer to use.
    equalize_norms : bool or float, optional
        Whether to equalize the norms of the tensors after compression. If an
        explicit value is give, then the norms will be set to that value, and
        the overall scaling factor will be accumulated into `.exponent`.
    inplace : bool, optional
        Whether to perform the compression inplace.
    compress_opts
        Supplied to
        :meth:`~quimb.tensor.tensor_core.TensorNetwork.compress_between`.

    Returns
    -------
    TensorNetwork
    """
    tnc = tn if inplace else tn.copy()

    if site_tags is None:
        site_tags = tnc.site_tags

    for st in site_tags:
        tnc.contract_(st, optimize=optimize)

    tnc.compress_all_(
        max_bond=max_bond,
        cutoff=cutoff,
        canonize=canonize,
        tree_gauge_distance=tree_gauge_distance,
        canonize_distance=canonize_distance,
        canonize_after_distance=canonize_after_distance,
        mode=mode,
        **compress_opts,
    )

    if equalize_norms is True:
        tnc.equalize_norms_()
    elif equalize_norms:
        tnc.equalize_norms_(value=equalize_norms)

    return tnc


def tensor_network_ag_compress_superorthogonal(
    tn,
    max_bond=None,
    cutoff=1e-10,
    site_tags=None,
    canonize=True,
    optimize="auto-hq",
    equalize_norms=False,
    inplace=False,
    **compress_opts,
):
    """Compress an arbtrary geometry tensor network, with potentially multiple
    tensors per site, using the 'superorthogonal' / 'Vidal' / quasi-canonical
    / 'simple update' gauge for compression. This is the same gauge as used in
    L2BP, but the intermediate tensor network is explicitly constructed.

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
        The tags to use to group the tensors from ``tn``. If not
        given, uses ``tn.site_tags``. The tensor network built will have one
        tensor per site.
    canonize : bool, optional
        Whether to locally gauge before each compression, defaults to True.
    optimize : str, optional
        The contraction path optimizer to use.
    equalize_norms : bool or float, optional
        Whether to equalize the norms of the tensors after compression. If an
        explicit value is give, then the norms will be set to that value, and
        the overall scaling factor will be accumulated into `.exponent`.
    inplace : bool, optional
        Whether to perform the compression inplace.
    compress_opts
        Supplied to
        :meth:`~quimb.tensor.tensor_core.TensorNetwork.compress_all_simple`.

    Returns
    -------
    TensorNetwork
    """
    tnc = tn if inplace else tn.copy()

    if site_tags is None:
        site_tags = tnc.site_tags

    for st in site_tags:
        tnc.contract_(st, optimize=optimize)

    tnc.fuse_multibonds_()

    if not canonize:
        # turn off gauging effect
        compress_opts.setdefault("max_iterations", 1)
        compress_opts.setdefault("tol", 0.0)
    else:
        compress_opts.setdefault("max_iterations", 1000)
        compress_opts.setdefault("tol", 5e-6)

    tnc.compress_all_simple_(
        max_bond=max_bond,
        cutoff=cutoff,
        **compress_opts,
    )

    if equalize_norms is True:
        tnc.equalize_norms_()
    elif equalize_norms:
        tnc.equalize_norms_(value=equalize_norms)

    return tnc


def tensor_network_ag_compress_l2bp(
    tn,
    max_bond=None,
    cutoff=1e-10,
    site_tags=None,
    canonize=True,
    damping=0.0,
    local_convergence=True,
    update="sequential",
    optimize="auto-hq",
    equalize_norms=False,
    inplace=False,
    **compress_opts,
):
    """Compress an arbitrary geometry tensor network, with potentially multiple
    tensors per site, using lazy 2-norm belief propagation.

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
        The tags to use to group the tensors from ``tn``. If not
        given, uses ``tn.site_tags``. The tensor network built will have one
        tensor per site.
    canonize : bool, optional
        Whether to locally gauge before each compression, defaults to True.
    damping : float, optional
        How much to dampen message updates, to help convergence, defaults to 0.
    local_convergence : bool, optional
        Whether to use local convergence criteria, defaults to True.
    update : {'parallel', 'sequential'}, optional
        Whether to update all messages in parallel or sequentially, defaults to
        'parallel'.
    optimize : str, optional
        The contraction path optimizer to use.
    equalize_norms : bool or float, optional
        Whether to equalize the norms of the tensors after compression. If an
        explicit value is give, then the norms will be set to that value, and
        the overall scaling factor will be accumulated into `.exponent`.
    inplace : bool, optional
        Whether to perform the compression inplace.
    compress_opts
        Supplied to
        :func:`~quimb.tensor.belief_propagation.l2bp.compress_l2bp`.

    Returns
    -------
    TensorNetwork
    """
    from quimb.tensor.belief_propagation.l2bp import compress_l2bp

    if not canonize:
        compress_opts.setdefault("max_iterations", 1)

    tnc = compress_l2bp(
        tn,
        max_bond=max_bond,
        cutoff=cutoff,
        site_tags=site_tags,
        damping=damping,
        local_convergence=local_convergence,
        update=update,
        optimize=optimize,
        inplace=inplace,
        **compress_opts,
    )

    if equalize_norms is True:
        tnc.equalize_norms_()
    elif equalize_norms:
        tnc.equalize_norms_(value=equalize_norms)

    return tnc


_TNAG_COMPRESS_METHODS = {
    "local-early": tensor_network_ag_compress_local_early,
    "local-late": tensor_network_ag_compress_local_late,
    "projector": tensor_network_ag_compress_projector,
    "su": tensor_network_ag_compress_superorthogonal,
    "superorthogonal": tensor_network_ag_compress_superorthogonal,
    "l2bp": tensor_network_ag_compress_l2bp,
}


def tensor_network_ag_compress(
    tn,
    max_bond,
    cutoff=1e-10,
    method="local-early",
    site_tags=None,
    canonize=True,
    optimize="auto-hq",
    equalize_norms=False,
    inplace=False,
    **kwargs,
):
    """Compress an arbitrary geometry tensor network, with potentially multiple
    tensors per site.

    Parameters
    ----------
    tn : TensorNetwork
        The tensor network to compress. Every tensor should have exactly one of
        the site tags. Each site can have multiple tensors and output indices.
    max_bond : int
        The maximum bond dimension to compress to.
    cutoff : float, optional
        A dynamic threshold for discarding singular values when compressing.
    method : {'local-early', 'local-late', 'projector', 'superorthogonal', 'l2bp'}, optional
        The compression method to use:

        - 'local-early': explicitly contract each site and interleave with
          immediate compression, see
          :func:`~quimb.tensor.tensor_arbgeom_compress.tensor_network_ag_compress_local_early`.
        - 'local-late': explicitly contract all sites and then compress, see
          :func:`~quimb.tensor.tensor_arbgeom_compress.tensor_network_ag_compress_local_late`.
        - 'projector': use locally computed projectors, see
          :func:`~quimb.tensor.tensor_arbgeom_compress.tensor_network_ag_compress_projector`.
        - 'superorthogonal': use the 'superorthogonal' gauge, see
          :func:`~quimb.tensor.tensor_arbgeom_compress.tensor_network_ag_compress_superorthogonal`.
        - 'l2bp': use lazy 2-norm belief propagation, see
          :func:`~quimb.tensor.tensor_arbgeom_compress.tensor_network_ag_compress_l2bp`.

    site_tags : sequence of str, optional
        The tags to use to group the tensors from ``tn``. If not
        given, uses ``tn.site_tags``. The tensor network built will have one
        tensor per site.
    canonize : bool, optional
        Whether to perform canonicalization, pseudo or otherwise depending on
        the method, before compressing.
    optimize : str, optional
        The contraction path optimizer to use.
    equalize_norms : bool or float, optional
        Whether to equalize the norms of the tensors after compression. If an
        explicit value is give, then the norms will be set to that value, and
        the overall scaling factor will be accumulated into `.exponent`.
    inplace : bool, optional
        Whether to perform the compression inplace.
    kwargs
        Supplied to the chosen compression method.
    """
    return _TNAG_COMPRESS_METHODS[method](
        tn,
        max_bond=max_bond,
        cutoff=cutoff,
        site_tags=site_tags,
        canonize=canonize,
        optimize=optimize,
        equalize_norms=equalize_norms,
        inplace=inplace,
        **kwargs,
    )
