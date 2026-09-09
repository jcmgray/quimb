from ..tn1d.compress import possibly_permute_
from ..tnag.compress import tensor_network_ag_compress

_TN2D_COMPRESS_METHODS = {}


def tensor_network_2d_compress(
    tn,
    max_bond=None,
    cutoff=1e-10,
    method="local-early",
    site_tags=None,
    canonize=True,
    permute_arrays=True,
    optimize="auto-hq",
    equalize_norms=False,
    compress_opts=None,
    inplace=False,
    **kwargs,
):
    """Compress a 2D-like tensor network using the specified method.

    Parameters
    ----------
    tn : TensorNetwork
        The tensor network to compress. Every tensor should have exactly one of
        the site tags. Each site can have multiple tensors and output indices.
    max_bond : int
        The maximum bond dimension to compress to.
    cutoff : float, optional
        A dynamic threshold for discarding singular values when compressing.
    method : str or callable, optional
        The compression method to use. A callable is passed the same arguments
        as a built-in 2D method.
    site_tags : sequence of str or tag groups, optional
        Tags that identify and order sites. Defaults to ``tn.site_tags``.
        Each item can group tags as described by
        :func:`~quimb.tensor.parse_site_tag_groups`. The output has one tensor
        per item.
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
    inplace : bool, optional
        Whether to perform the compression inplace.
    kwargs
        Supplied to the chosen compression method.

    Returns
    -------
    TensorNetwork
    """
    compress_opts = compress_opts or {}

    if callable(method):
        f_tn2d = method
    else:
        f_tn2d = _TN2D_COMPRESS_METHODS.get(method, None)

    if f_tn2d is not None:
        return f_tn2d(
            tn,
            max_bond=max_bond,
            cutoff=cutoff,
            site_tags=site_tags,
            canonize=canonize,
            permute_arrays=permute_arrays,
            optimize=optimize,
            equalize_norms=equalize_norms,
            inplace=inplace,
            compress_opts=compress_opts,
            **kwargs,
        )

    # try arbitrary geometry methods
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
        compress_opts=compress_opts,
        **kwargs,
    )

    if permute_arrays:
        possibly_permute_(tnc, permute_arrays)

    return tnc
