"""Tools for computing distances between and fitting tensor networks."""
from autoray import dag, do

from .contraction import contract_strategy
from ..utils import check_opt


def tensor_network_distance(
    tnA,
    tnB,
    xAA=None,
    xAB=None,
    xBB=None,
    method="auto",
    normalized=False,
    **contract_opts,
):
    r"""Compute the Frobenius norm distance between two tensor networks:

    .. math::

            D(A, B)
            = | A - B |_{\mathrm{fro}}
            = \mathrm{Tr} [(A - B)^{\dagger}(A - B)]^{1/2}
            = ( \langle A | A \rangle - 2 \mathrm{Re} \langle A | B \rangle|
            + \langle B | B \rangle ) ^{1/2}

    which should have matching outer indices. Note the default approach to
    computing the norm is precision limited to about ``eps**0.5`` where ``eps``
    is the precision of the data type, e.g. ``1e-8`` for float64. This is due
    to the subtraction in the above expression.

    Parameters
    ----------
    tnA : TensorNetwork or Tensor
        The first tensor network operator.
    tnB : TensorNetwork or Tensor
        The second tensor network operator.
    xAA : None or scalar
        The value of ``A.H @ A`` if you already know it (or it doesn't matter).
    xAB : None or scalar
        The value of ``A.H @ B`` if you already know it (or it doesn't matter).
    xBB : None or scalar
        The value of ``B.H @ B`` if you already know it (or it doesn't matter).
    method : {'auto', 'overlap', 'dense'}, optional
        How to compute the distance. If ``'overlap'``, the default, the
        distance will be computed as the sum of overlaps, without explicitly
        forming the dense operators. If ``'dense'``, the operators will be
        directly formed and the norm computed, which can be quicker when the
        exterior dimensions are small. If ``'auto'``, the dense method will
        be used if the total operator (outer) size is ``<= 2**16``.
    normalized : bool, optional
        If ``True``, then normalize the distance by the norm of the two
        operators, i.e. ``2 * D(A, B) / (|A| + |B|)``. The resulting distance
        lies between 0 and 2 and is more useful for assessing convergence.
    contract_opts
        Supplied to :meth:`~quimb.tensor.tensor_core.TensorNetwork.contract`.

    Returns
    -------
    D : float
    """
    check_opt("method", method, ("auto", "dense", "overlap"))

    tnA = tnA.as_network()
    tnB = tnB.as_network()

    oix = tnA.outer_inds()
    if set(oix) != set(tnB.outer_inds()):
        raise ValueError(
            "Can only compute distance between tensor "
            "networks with matching outer indices."
        )

    if method == "auto":
        d = tnA.inds_size(oix)
        if d <= 1 << 16:
            method = "dense"
        else:
            method = "overlap"

    # directly from vectorizations of both
    if method == "dense":
        tnA = tnA.contract(..., output_inds=oix, preserve_tensor=True)
        tnB = tnB.contract(..., output_inds=oix, preserve_tensor=True)

    # overlap method
    if xAA is None:
        xAA = (tnA | tnA.H).contract(..., **contract_opts)
    if xAB is None:
        xAB = (tnA | tnB.H).contract(..., **contract_opts)
    if xBB is None:
        xBB = (tnB | tnB.H).contract(..., **contract_opts)

    dAB = do("abs", xAA - 2 * do("real", xAB) + xBB) ** 0.5

    if normalized:
        dAB *= 2 / (do("abs", xAA)**0.5 + do("abs", xBB)**0.5)

    return dAB




def tensor_network_fit_autodiff(
    tn,
    tn_target,
    steps=1000,
    tol=1e-9,
    autodiff_backend="autograd",
    contract_optimize="auto-hq",
    distance_method="auto",
    inplace=False,
    progbar=False,
    **kwargs,
):
    """Optimize the fit of ``tn`` with respect to ``tn_target`` using
    automatic differentation. This minimizes the norm of the difference
    between the two tensor networks, which must have matching outer indices,
    using overlaps.

    Parameters
    ----------
    tn : TensorNetwork
        The tensor network to fit.
    tn_target : TensorNetwork
        The target tensor network to fit ``tn`` to.
    steps : int, optional
        The maximum number of autodiff steps.
    tol : float, optional
        The target norm distance.
    autodiff_backend : str, optional
        Which backend library to use to perform the gradient computation.
    contract_optimize : str, optional
        The contraction path optimized used to contract the overlaps.
    distance_method : {'auto', 'dense', 'overlap'}, optional
        Supplied to :func:`~quimb.tensor.tensor_core.tensor_network_distance`,
        controls how the distance is computed.
    inplace : bool, optional
        Update ``tn`` in place.
    progbar : bool, optional
        Show a live progress bar of the fitting process.
    kwargs
        Passed to :class:`~quimb.tensor.tensor_core.optimize.TNOptimizer`.

    See Also
    --------
    tensor_network_distance, tensor_network_fit_als
    """
    from .optimize import TNOptimizer
    from .tensor_core import tensor_network_distance

    xBB = (tn_target | tn_target.H).contract(
        ...,
        output_inds=(),
        optimize=contract_optimize,
    )

    tnopt = TNOptimizer(
        tn=tn,
        loss_fn=tensor_network_distance,
        loss_constants={"tnB": tn_target, "xBB": xBB},
        loss_kwargs={"method": distance_method, "optimize": contract_optimize},
        autodiff_backend=autodiff_backend,
        progbar=progbar,
        **kwargs,
    )

    tn_fit = tnopt.optimize(steps, tol=tol)

    if not inplace:
        return tn_fit

    for t1, t2 in zip(tn, tn_fit):
        t1.modify(data=t2.data)

    return tn


def _tn_fit_als_core(
    var_tags,
    tnAA,
    tnAB,
    xBB,
    tol,
    contract_optimize,
    steps,
    enforce_pos,
    pos_smudge,
    solver="solve",
    progbar=False,
):
    from .tensor_core import group_inds

    # shared intermediates + greedy = good reuse of contractions
    with contract_strategy(contract_optimize):
        # prepare each of the contractions we are going to repeat
        env_contractions = []
        for tg in var_tags:
            # varying tensor and conjugate in norm <A|A>
            tk = tnAA["__KET__", tg]
            tb = tnAA["__BRA__", tg]

            # get inds, and ensure any bonds come last, for linalg.solve
            lix, bix, rix = group_inds(tb, tk)
            tk.transpose_(*rix, *bix)
            tb.transpose_(*lix, *bix)

            # form TNs with 'holes', i.e. environment tensors networks
            A_tn = tnAA.select((tg,), "!all")
            y_tn = tnAB.select((tg,), "!all")

            env_contractions.append((tk, tb, lix, bix, rix, A_tn, y_tn))

        if tol != 0.0:
            old_d = float("inf")

        if progbar:
            import tqdm

            pbar = tqdm.trange(steps)
        else:
            pbar = range(steps)

        # the main iterative sweep on each tensor, locally optimizing
        for _ in pbar:
            for tk, tb, lix, bix, rix, A_tn, y_tn in env_contractions:
                Ni = A_tn.to_dense(lix, rix)
                Wi = y_tn.to_dense(rix, bix)

                if enforce_pos:
                    el, ev = do("linalg.eigh", Ni)
                    el = do("clip", el, el[-1] * pos_smudge, None)
                    Ni_p = ev * do("reshape", el, (1, -1)) @ dag(ev)
                else:
                    Ni_p = Ni

                if solver == "solve":
                    x = do("linalg.solve", Ni_p, Wi)
                elif solver == "lstsq":
                    x = do("linalg.lstsq", Ni_p, Wi, rcond=pos_smudge)[0]

                x_r = do("reshape", x, tk.shape)
                # n.b. because we are using virtual TNs -> updates propagate
                tk.modify(data=x_r)
                tb.modify(data=do("conj", x_r))

            # assess | A - B | for convergence or printing
            if (tol != 0.0) or progbar:
                xAA = do("trace", dag(x) @ (Ni @ x))  # <A|A>
                xAB = do("trace", do("real", dag(x) @ Wi))  # <A|B>
                d = do("abs", (xAA - 2 * xAB + xBB)) ** 0.5
                if abs(d - old_d) < tol:
                    break
                old_d = d

            if progbar:
                pbar.set_description(str(d))


def tensor_network_fit_als(
    tn,
    tn_target,
    tags=None,
    steps=100,
    tol=1e-9,
    solver="solve",
    enforce_pos=False,
    pos_smudge=None,
    tnAA=None,
    tnAB=None,
    xBB=None,
    contract_optimize="greedy",
    inplace=False,
    progbar=False,
):
    """Optimize the fit of ``tn`` with respect to ``tn_target`` using
    alternating least squares (ALS). This minimizes the norm of the difference
    between the two tensor networks, which must have matching outer indices,
    using overlaps.

    Parameters
    ----------
    tn : TensorNetwork
        The tensor network to fit.
    tn_target : TensorNetwork
        The target tensor network to fit ``tn`` to.
    tags : sequence of str, optional
        If supplied, only optimize tensors matching any of given tags.
    steps : int, optional
        The maximum number of ALS steps.
    tol : float, optional
        The target norm distance.
    solver : {'solve', 'lstsq', ...}, optional
        The underlying driver function used to solve the local minimization,
        e.g. ``numpy.linalg.solve`` for ``'solve'`` with ``numpy`` backend.
    enforce_pos : bool, optional
        Whether to enforce positivity of the locally formed environments,
        which can be more stable.
    pos_smudge : float, optional
        If enforcing positivity, the level below which to clip eigenvalues
        for make the local environment positive definite.
    tnAA : TensorNetwork, optional
        If you have already formed the overlap ``tn.H & tn``, maybe
        approximately, you can supply it here. The unconjugated layer should
        have tag ``'__KET__'`` and the conjugated layer ``'__BRA__'``. Each
        tensor being optimized should have tag ``'__VAR{i}__'``.
    tnAB : TensorNetwork, optional
        If you have already formed the overlap ``tn_target.H & tn``, maybe
        approximately, you can supply it here. Each tensor being optimized
        should have tag ``'__VAR{i}__'``.
    xBB : float, optional
        If you have already know, have computed ``tn_target.H @ tn_target``,
        or it doesn't matter, you can supply the value here.
    contract_optimize : str, optional
        The contraction path optimized used to contract the local environments.
        Note ``'greedy'`` is the default in order to maximize shared work.
    inplace : bool, optional
        Update ``tn`` in place.
    progbar : bool, optional
        Show a live progress bar of the fitting process.

    Returns
    -------
    TensorNetwork

    See Also
    --------
    tensor_network_fit_autodiff, tensor_network_distance
    """
    # mark the tensors we are going to optimize
    tna = tn.copy()
    tna.add_tag("__KET__")

    if tags is None:
        to_tag = tna
    else:
        to_tag = tna.select_tensors(tags, "any")

    var_tags = []
    for i, t in enumerate(to_tag):
        var_tag = f"__VAR{i}__"
        t.add_tag(var_tag)
        var_tags.append(var_tag)

    # form the norm of the varying TN (A) and its overlap with the target (B)
    if tnAA is None:
        tnAA = tna | tna.H.retag_({"__KET__": "__BRA__"})
    if tnAB is None:
        tnAB = tna | tn_target.H

    if (tol != 0.0) and (xBB is None):
        # <B|B>
        xBB = (tn_target | tn_target.H).contract(
            ...,
            optimize=contract_optimize,
            output_inds=(),
        )

    if pos_smudge is None:
        pos_smudge = max(tol, 1e-15)

    _tn_fit_als_core(
        var_tags=var_tags,
        tnAA=tnAA,
        tnAB=tnAB,
        xBB=xBB,
        tol=tol,
        contract_optimize=contract_optimize,
        steps=steps,
        enforce_pos=enforce_pos,
        pos_smudge=pos_smudge,
        solver=solver,
        progbar=progbar,
    )

    if not inplace:
        tn = tn.copy()

    for t1, t2 in zip(tn, tna):
        # transpose so only thing changed in original TN is data
        t2.transpose_like_(t1)
        t1.modify(data=t2.data)

    return tn
