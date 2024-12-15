"""Tools for computing distances between and fitting tensor networks."""

from autoray import backend_like, compose, dag, do

from ..utils import check_opt
from .tensor_core import TNLinearOperator
from .contraction import contract_strategy


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
    normalized : bool or str, optional
        If ``True``, then normalize the distance by the norm of the two
        operators, i.e. ``D(A, B) * 2 / (|A| + |B|)``. The resulting distance
        lies between 0 and 2 and is more useful for assessing convergence.
        If ``'infidelity'``, compute the normalized infidelity
        ``1 - |<A|B>|^2 / (|A| |B|)``, which can be faster to optimize e.g.,
        but does not take into account normalization.
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
        tnA = tnA.contract(output_inds=oix, preserve_tensor=True)
        tnB = tnB.contract(output_inds=oix, preserve_tensor=True)

    # overlap method
    if xAA is None:
        xAA = (tnA | tnA.H).contract(**contract_opts)
    if xAB is None:
        xAB = (tnA | tnB.H).contract(**contract_opts)
    if xBB is None:
        xBB = (tnB | tnB.H).contract(**contract_opts)

    if normalized == "infidelity":
        # compute normalized infidelity
        return 1 - do("abs", xAB**2 / (xAA * xBB))

    if normalized == "infidelity_sqrt":
        # compute normalized sqrt infidelity
        return 1 - do("abs", xAB / (xAA * xBB) ** 0.5)

    if normalized == "squared":
        return (
            do("abs", xAA + xBB - 2 * do("real", xAB))
            # divide by average norm-squared of A and B
            * 2
            / (do("abs", xAA) + do("abs", xBB))
        ) ** 0.5

    dAB = do("abs", xAA + xBB - 2 * do("real", xAB)) ** 0.5

    if normalized:
        # divide by average norm of A and B
        dAB = dAB * 2 / (do("abs", xAA) ** 0.5 + do("abs", xBB) ** 0.5)

    return dAB


def tensor_network_fit_autodiff(
    tn,
    tn_target,
    steps=1000,
    tol=1e-9,
    autodiff_backend="autograd",
    contract_optimize="auto-hq",
    distance_method="auto",
    normalized="squared",
    xBB=None,
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
    normalized : bool or str, optional
        If ``True``, then normalize the distance by the norm of the two
        operators, i.e. ``D(A, B) * 2 / (|A| + |B|)``. The resulting distance
        lies between 0 and 2 and is more useful for assessing convergence.
        If ``'infidelity'``, compute the normalized infidelity
        ``1 - |<A|B>|^2 / (|A| |B|)``, which can be faster to optimize e.g.,
        but does not take into account normalization.
    xBB : float, optional
        If you already know, have computed ``tn_target.H @ tn_target``, or
        don't care about the overall scale of the norm distance, you can supply
        a value here.
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

    if xBB is None:
        xBB = (tn_target | tn_target.H).contract(
            output_inds=(),
            optimize=contract_optimize,
        )

    tnopt = TNOptimizer(
        tn=tn,
        loss_fn=tensor_network_distance,
        loss_constants={"tnB": tn_target, "xBB": xBB},
        loss_kwargs={
            "method": distance_method,
            "optimize": contract_optimize,
            "normalized": normalized,
        },
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


@compose
def vdot_broadcast(x, y):
    return do("sum", x * do("conj", y), axis=0)


def conjugate_gradient(A, b, x0=None, tol=1e-10, maxiter=1000):
    """
    Conjugate Gradient solver for complex matrices/linear operators.

    Parameters
    ----------
    A : operator_like
        The matrix or linear operator.
    B : array_like
        The right-hand side vector.
    x0 : array_like, optional
        Initial guess for the solution.
    tol : float, optional
        Tolerance for convergence.
    maxiter : int, optional
        Maximum number of iterations.

    Returns:
    --------
    x : array_like
        The solution vector.
    """
    if x0 is None:
        x0 = do("zeros_like", b)

    x = x0
    r = p = b - A @ x

    rsold = vdot_broadcast(r, r)

    for _ in range(maxiter):
        Ap = A @ p
        alpha = rsold / vdot_broadcast(p, Ap)
        x = x + alpha * p
        r = r - alpha * Ap
        rsnew = vdot_broadcast(r, r)
        if do("all", do("sqrt", rsnew)) < tol:
            break
        p = r + (rsnew / rsold) * p
        rsold = rsnew

    return x


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
    iterative=False,
    iterative_solver="cg",
    iterative_maxiter=2,
    progbar=False,
):
    from .tensor_core import group_inds

    backend = next(iter(tnAA.tensors)).backend

    # shared intermediates + greedy = good reuse of contractions
    with contract_strategy(contract_optimize), backend_like(backend):
        # prepare each of the contractions we are going to repeat
        env_contractions = []
        for tg in var_tags:
            # varying tensor and conjugate in norm <A|A>
            tk = tnAA["__KET__", tg]
            tb = tnAA["__BRA__", tg]

            # get inds, and ensure any bonds come last, for linalg.solve
            lix, bix, rix = group_inds(tk, tb)
            tk.transpose_(*lix, *bix)
            tb.transpose_(*rix, *bix)

            # form TNs with 'holes', i.e. environment tensors networks
            A_tn = tnAA.select((tg,), "!all")
            y_tn = tnAB.select((tg,), "!all")

            env_contractions.append((tk, tb, lix, bix, rix, A_tn, y_tn))

        if tol != 0.0 or progbar:
            old_d = float("inf")

        if progbar:
            import tqdm

            pbar = tqdm.trange(steps)
        else:
            pbar = range(steps)

        # the main iterative sweep on each tensor, locally optimizing
        for _ in pbar:
            for tk, tb, lix, bix, rix, A_tn, y_tn in env_contractions:
                if iterative:
                    Ni = TNLinearOperator(
                        A_tn,
                        left_inds=rix + bix,
                        right_inds=lix + bix,
                        ldims=tb.shape,
                        rdims=tk.shape,
                    )
                    bi = y_tn.to_dense((*rix, *bix))
                    x0 = tk.to_dense((*lix, *bix))

                    if iterative_solver is None:
                        x = conjugate_gradient(
                            Ni, bi, x0=x0, tol=tol, maxiter=iterative_maxiter
                        )
                    else:
                        x = do(
                            f"scipy.sparse.linalg.{iterative_solver}",
                            Ni,
                            bi,
                            x0=x0,
                            rtol=tol,
                            maxiter=iterative_maxiter,
                        )[0]

                else:
                    # form local normalization and local overlap
                    Ni = A_tn.to_dense(rix, lix)
                    bi = y_tn.to_dense(rix, bix)

                    if enforce_pos:
                        el, V = do("linalg.eigh", Ni)
                        elmax = do("max", el)
                        el = do("clip", el, elmax * pos_smudge, None)
                        # can solve directly using eigendecomposition
                        x = V @ ((dag(V) @ bi) / do("reshape", el, (-1, 1)))
                    else:
                        Ni_p = Ni

                        if solver is None:
                            x0 = tk.to_dense(lix, bix)
                            x = conjugate_gradient(
                                Ni_p,
                                bi,
                                x0=x0,
                                tol=tol,
                                maxiter=iterative_maxiter,
                            )
                        if solver == "solve":
                            x = do("linalg.solve", Ni_p, bi)
                        elif solver == "lstsq":
                            x = do("linalg.lstsq", Ni_p, bi, rcond=pos_smudge)[
                                0
                            ]

                x_r = do("reshape", x, tk.shape)
                # n.b. because we are using virtual TNs -> updates propagate
                tk.modify(data=x_r)
                tb.modify(data=do("conj", x_r))

            # assess | A - B | (normalized) for convergence or printing
            if (tol != 0.0) or progbar:
                dagx = dag(x)

                if x.ndim == 2:
                    xAA = do("trace", do("real", dagx @ (Ni @ x)))  # <A|A>
                    xAB = do("trace", do("real", dagx @ bi))  # <A|B>
                else:
                    xAA = do("real", dagx @ (Ni @ x))
                    xAB = do("real", dagx @ bi)

                d = abs(xAA + xBB - 2 * xAB) ** 0.5 * 2 / (xAA**0.5 + xBB**0.5)
                if abs(d - old_d) < tol:
                    break
                old_d = d

            if progbar:
                pbar.set_description(f"{d:.3g}")


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
    **kwargs,
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
    tn_fit = tn.copy()
    tn_fit.add_tag("__KET__")

    if tags is None:
        to_tag = tn_fit
    else:
        to_tag = tn_fit.select_tensors(tags, "any")

    var_tags = []
    for i, t in enumerate(to_tag):
        var_tag = f"__VAR{i}__"
        t.add_tag(var_tag)
        var_tags.append(var_tag)

    # form the norm of the varying TN (A) and its overlap with the target (B)
    tn_fit_conj = tn_fit.conj().retag_({"__KET__": "__BRA__"})
    if tnAA is None:
        tnAA = tn_fit | tn_fit_conj
    if tnAB is None:
        tnAB = tn_target | tn_fit_conj

    if (tol != 0.0 or progbar) and (xBB is None):
        # <B|B>
        xBB = (tn_target | tn_target.H).contract(
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
        **kwargs,
    )

    if not inplace:
        tn = tn.copy()

    for t1, t2 in zip(tn, tn_fit):
        # transpose so only thing changed in original TN is data
        t2.transpose_like_(t1)
        t1.modify(data=t2.data)

    return tn


def tensor_network_fit_tree(
    tn,
    tn_target,
    tags=None,
    steps=100,
    tol=1e-9,
    ordering=None,
    xBB=None,
    istree=True,
    contract_optimize="auto-hq",
    inplace=False,
    progbar=False,
):
    """Fit `tn` to `tn_target`, assuming that `tn` has tree structure (i.e. a
    single path between any two sites) and matching outer structure to
    `tn_target`. The tree structure allows a canonical form that greatly
    simplifies the normalization and least squares minimization. Note that no
    structure is assumed about `tn_target`, and so for example no partial
    contractions reused.

    Parameters
    ----------
    tn : TensorNetwork
        The tensor network to fit, it should have a tree structure and outer
        indices matching `tn_target`.
    tn_target : TensorNetwork
        The target tensor network to fit ``tn`` to.
    tags : sequence of str, optional
        If supplied, only optimize tensors matching any of given tags.
    steps : int, optional
        The maximum number of ALS steps.
    tol : float, optional
        The target norm distance.
    ordering : sequence of int, optional
        The order in which to optimize the tensors, if None will be computed
        automatically using a hierarchical clustering.
    xBB : float, optional
        If you have already know, have computed ``tn_target.H @ tn_target``,
        or it doesn't matter, you can supply the value here. It matters only
        for the overall scale of the norm distance.
    contract_optimize : str, optional
        A contraction path strategy or optimizer for contracting the local
        environments.
    inplace : bool, optional
        Fit ``tn`` in place.
    progbar : bool, optional
        Show a live progress bar of the fitting process.

    Returns
    -------
    TensorNetwork
    """
    if xBB is None:
        xBB = (tn_target | tn_target.H).contract(
            optimize=contract_optimize,
            output_inds=(),
        )

    tn_fit = tn.conj(inplace=inplace)
    tnAB = tn_fit | tn_target

    if ordering is None:
        if tags is not None:
            tids = tn_fit._get_tids_from_tags(tags, "any")
        else:
            tids = None
        ordering = tn_fit.compute_hierarchical_ordering(tids)

    # prepare contractions
    env_contractions = []
    for i, tid in enumerate(ordering):
        tn_hole = tnAB.copy(virtual=True)
        ti = tn_hole.pop_tensor(tid)
        # we'll need to canonicalize along path from the last tid to this one
        tid_prev = ordering[(i - 1) % len(ordering)]
        path = tn_fit.get_path_between_tids(tid_prev, tid)
        canon_pairs = [
            (path.tids[j], path.tids[j + 1]) for j in range(len(path))
        ]
        env_contractions.append((tid, tn_hole, ti, canon_pairs))

    # initial canonicalization around first tensor
    tn_fit._canonize_around_tids([ordering[0]])

    if progbar:
        import tqdm

        pbar = tqdm.trange(steps)
    else:
        pbar = range(steps)

    old_d = float("inf")

    for _ in pbar:
        for tid, tn_hole, ti, canon_pairs in env_contractions:
            if istree:
                # move canonical center to tid
                for tidi, tidj in canon_pairs:
                    tn_fit._canonize_between_tids(tidi, tidj)
            else:
                # pseudo canonicalization
                tn_fit._canonize_around_tids([tid])

            # get the new conjugate tensor
            ti_new = tn_hole.contract(output_inds=ti.inds, optimize="auto-hq")
            ti_new.conj_()
            # modify the data
            ti.modify(data=ti_new.data)

        if tol != 0.0 or progbar:
            # canonicalized form enable simpler distance computation
            xAA = ti.norm() ** 2  # == xAB
            d = 2 * abs(xBB - xAA) ** 0.5 / (xBB**0.5 + xAA**0.5)
            if abs(d - old_d) < tol:
                break
            old_d = d

        if progbar:
            pbar.set_description(f"{d:.3g}")

    return tn_fit.conj_()
