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
    method : {"direct", "dm", "zipup", "zipup-first", "fit", "projector"}
        The compression method to use.
    site_tags : sequence of sequence of str, optional
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
    inplace : bool, optional
        Whether to perform the compression inplace.
    kwargs
        Supplied to the chosen compression method.

    Returns
    -------
    TensorNetwork
    """
    compress_opts = compress_opts or {}

    try:
        return _TN2D_COMPRESS_METHODS[method](
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
    except KeyError:
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
