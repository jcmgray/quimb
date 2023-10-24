"""Hyper, vectorized, 1-norm, belief propagation.
"""

import autoray as ar

from quimb.tensor.contraction import array_contract
from .bp_common import (
    BeliefPropagationCommon,
    compute_all_index_marginals_from_messages,
    contract_hyper_messages,
    initialize_hyper_messages,
    maybe_get_thread_pool,
)


def initialize_messages_batched(tn, messages=None):
    """Initialize batched messages for belief propagation, as the uniform
    distribution.
    """
    if messages is None:
        messages = initialize_hyper_messages(tn)

    backend = ar.infer_backend(next(iter(messages.values())))
    _stack = ar.get_lib_fn(backend, "stack")
    _array = ar.get_lib_fn(backend, "array")

    # prepare index messages
    batched_inputs_m = {}
    input_locs_m = {}
    output_locs_m = {}
    for ix, tids in tn.ind_map.items():
        rank = len(tids)
        try:
            batch = batched_inputs_m[rank]
        except KeyError:
            batch = batched_inputs_m[rank] = [[] for _ in range(rank)]

        for i, tid in enumerate(tids):
            batch_i = batch[i]
            # position in the stack
            b = len(batch_i)
            input_locs_m[tid, ix] = (rank, i, b)
            output_locs_m[ix, tid] = (rank, i, b)
            batch_i.append(messages[tid, ix])

    # prepare tensor messages
    batched_tensors = {}
    batched_inputs_t = {}
    input_locs_t = {}
    output_locs_t = {}
    for tid, t in tn.tensor_map.items():
        rank = t.ndim
        if rank == 0:
            continue

        try:
            batch = batched_inputs_t[rank]
            batch_t = batched_tensors[rank]
        except KeyError:
            batch = batched_inputs_t[rank] = [[] for _ in range(rank)]
            batch_t = batched_tensors[rank] = []

        for i, ix in enumerate(t.inds):
            batch_i = batch[i]
            # position in the stack
            b = len(batch_i)
            input_locs_t[ix, tid] = (rank, i, b)
            output_locs_t[tid, ix] = (rank, i, b)
            batch_i.append(messages[ix, tid])

        batch_t.append(t.data)

    # stack messages in into single arrays
    for batched_inputs in (batched_inputs_m, batched_inputs_t):
        for key, batch in batched_inputs.items():
            batched_inputs[key] = _stack(
                tuple(_stack(batch_i) for batch_i in batch)
            )
    for rank, tensors in batched_tensors.items():
        batched_tensors[rank] = _stack(tensors)

    # make numeric masks for updating output to input messages
    masks_m = {}
    masks_t = {}
    for masks, input_locs, output_locs in [
        (masks_m, input_locs_m, output_locs_t),
        (masks_t, input_locs_t, output_locs_m),
    ]:
        for pair in input_locs:
            (ranki, ii, bi) = input_locs[pair]
            (ranko, io, bo) = output_locs[pair]
            key = (ranki, ranko)
            try:
                maskin, maskout = masks[key]
            except KeyError:
                maskin, maskout = masks[key] = [], []
            maskin.append([ii, bi])
            maskout.append([io, bo])

        for key, (maskin, maskout) in masks.items():
            masks[key] = _array(maskin), _array(maskout)

    return (
        batched_inputs_m,
        batched_inputs_t,
        batched_tensors,
        input_locs_m,
        input_locs_t,
        masks_m,
        masks_t,
    )


def _compute_all_hyperind_messages_tree_batched(bm):
    """ """
    ndim = len(bm)

    if ndim == 2:
        # shortcut for 'bonds', which just swap places
        return ar.do("flip", bm, (0,))

    backend = ar.infer_backend(bm)
    _prod = ar.get_lib_fn(backend, "prod")
    _empty_like = ar.get_lib_fn(backend, "empty_like")

    bmo = _empty_like(bm)
    queue = [(tuple(range(ndim)), 1, bm)]

    while queue:
        js, x, bm = queue.pop()

        ndim = len(bm)
        if ndim == 1:
            # reached single message
            bmo[js[0]] = x
            continue
        elif ndim == 2:
            # shortcut for 2 messages left
            bmo[js[0]] = x * bm[1]
            bmo[js[1]] = bm[0] * x
            continue

        # else split in two and contract each half
        k = ndim // 2
        jl, jr = js[:k], js[k:]
        bml, bmr = bm[:k], bm[k:]

        # contract the right messages to get new left array
        xl = x * _prod(bmr, axis=0)

        # contract the left messages to get new right array
        xr = _prod(bml, axis=0) * x

        # add the queue for possible further halving
        queue.append((jl, xl, bml))
        queue.append((jr, xr, bmr))

    return bmo


def _compute_all_hyperind_messages_prod_batched(bm, smudge_factor=1e-12):
    """ """
    backend = ar.infer_backend(bm)
    _prod = ar.get_lib_fn(backend, "prod")
    _reshape = ar.get_lib_fn(backend, "reshape")

    ndim = len(bm)
    if ndim == 2:
        # shortcut for 'bonds', which just swap
        return ar.do("flip", bm, (0,))

    combined = _prod(bm, axis=0)
    return _reshape(combined, (1, *ar.shape(combined))) / (bm + smudge_factor)


def _compute_all_tensor_messages_tree_batched(bx, bm):
    """Compute all output messages for a stacked tensor and messages."""
    backend = ar.infer_backend_multi(bx, bm)
    _stack = ar.get_lib_fn(backend, "stack")

    ndim = len(bm)
    mouts = [None for _ in range(ndim)]
    queue = [(tuple(range(ndim)), bx, bm)]

    while queue:
        js, bx, bm = queue.pop()

        ndim = len(bm)
        if ndim == 1:
            # reached single message
            mouts[js[0]] = bx
            continue
        elif ndim == 2:
            # shortcut for 2 messages left
            mouts[js[0]] = array_contract(
                arrays=(bx, bm[1]),
                inputs=(("X", "a", "b"), ("X", "b")),
                output=("X", "a"),
                backend=backend,
            )
            mouts[js[1]] = array_contract(
                arrays=(bm[0], bx),
                inputs=(("X", "a"), ("X", "a", "b")),
                output=("X", "b"),
                backend=backend,
            )
            continue

        # else split in two and contract each half
        k = ndim // 2
        jl, jr = js[:k], js[k:]
        ml, mr = bm[:k], bm[k:]

        # contract the right messages to get new left array
        xl = array_contract(
            arrays=(bx, *(mr[i] for i in range(mr.shape[0]))),
            inputs=((-1, *js), *((-1, j) for j in jr)),
            output=(-1, *jl),
            backend=backend,
        )

        # contract the left messages to get new right array
        xr = array_contract(
            arrays=(bx, *(ml[i] for i in range(ml.shape[0]))),
            inputs=((-1, *js), *((-1, j) for j in jl)),
            output=(-1, *jr),
            backend=backend,
        )

        # add the queue for possible further halving
        queue.append((jl, xl, ml))
        queue.append((jr, xr, mr))

    return _stack(tuple(mouts))


def _compute_all_tensor_messages_prod_batched(bx, bm, smudge_factor=1e-12):
    backend = ar.infer_backend_multi(bx, bm)
    _einsum = ar.get_lib_fn(backend, "einsum")
    _stack = ar.get_lib_fn(backend, "stack")

    ndim = len(bm)
    x_inds = (-1, *range(ndim))
    m_inds = [(-1, i) for i in range(ndim)]
    bmx = array_contract(
        arrays=(bx, *bm),
        inputs=(x_inds, *m_inds),
        output=x_inds,
    )

    bminv = 1 / (bm + smudge_factor)

    mouts = []
    for i in range(ndim):
        # sum all but ith index, apply inverse gate to that
        mouts.append(
            array_contract(
                arrays=(bmx, bminv[i]),
                inputs=(x_inds, m_inds[i]),
                output=m_inds[i],
            )
        )

    return _stack(mouts)


def _compute_output_single_t(
    bm,
    bx,
    _reshape,
    _sum,
    smudge_factor=1e-12,
):
    # tensor messages
    bmo = _compute_all_tensor_messages_tree_batched(bx, bm)
    # bmo = _compute_all_tensor_messages_prod_batched(bx, bm, smudge_factor)
    # normalize
    bmo /= _reshape(_sum(bmo, axis=-1), (*ar.shape(bmo)[:-1], 1))
    return bmo


def _compute_output_single_m(bm, _reshape, _sum, smudge_factor=1e-12):
    # index messages
    # bmo = _compute_all_hyperind_messages_tree_batched(bm)
    bmo = _compute_all_hyperind_messages_prod_batched(bm, smudge_factor)
    # normalize
    bmo /= _reshape(_sum(bmo, axis=-1), (*ar.shape(bmo)[:-1], 1))
    return bmo


def _compute_outputs_batched(
    batched_inputs,
    batched_tensors=None,
    smudge_factor=1e-12,
    _pool=None,
):
    """Given stacked messsages and tensors, compute stacked output messages."""
    backend = ar.infer_backend(next(iter(batched_inputs.values())))
    _sum = ar.get_lib_fn(backend, "sum")
    _reshape = ar.get_lib_fn(backend, "reshape")

    if batched_tensors is not None:
        # tensor messages
        f = _compute_output_single_t
        f_args = {
            rank: (bm, batched_tensors[rank], _reshape, _sum, smudge_factor)
            for rank, bm in batched_inputs.items()
        }
    else:
        # index messages
        f = _compute_output_single_m
        f_args = {
            rank: (bm, _reshape, _sum, smudge_factor)
            for rank, bm in batched_inputs.items()
        }

    batched_outputs = {}
    if _pool is None:
        # sequential process
        for rank, args in f_args.items():
            batched_outputs[rank] = f(*args)
    else:
        # parallel process
        for rank, args in f_args.items():
            batched_outputs[rank] = _pool.submit(f, *args)
        for key, fut in batched_outputs.items():
            batched_outputs[key] = fut.result()

    return batched_outputs


def _update_output_to_input_single_batched(
    bi,
    bo,
    maskin,
    maskout,
    _max,
    _sum,
    _abs,
    damping=0.0,
):
    # do a vectorized update
    select_in = (maskin[:, 0], maskin[:, 1], slice(None))
    select_out = (maskout[:, 0], maskout[:, 1], slice(None))
    bim = bi[select_in]
    bom = bo[select_out]

    if damping > 0.0:
        bim = (1 - damping) * bom + damping * bim

    # first check the change
    dm = _max(_sum(_abs(bim - bom), axis=-1))

    # update the input
    bi[select_in] = bom

    return dm


def _update_outputs_to_inputs_batched(
    batched_inputs, batched_outputs, masks, damping=0.0, _pool=None
):
    """Update the stacked input messages from the stacked output messages."""
    backend = ar.infer_backend(next(iter(batched_outputs.values())))
    _max = ar.get_lib_fn(backend, "max")
    _sum = ar.get_lib_fn(backend, "sum")
    _abs = ar.get_lib_fn(backend, "abs")

    f = _update_output_to_input_single_batched
    f_args = (
        (
            batched_inputs[ranki],
            batched_outputs[ranko],
            maskin,
            maskout,
            _max,
            _sum,
            _abs,
            damping,
        )
        for (ranki, ranko), (maskin, maskout) in masks.items()
    )

    if _pool is None:
        # sequential process
        dms = (f(*args) for args in f_args)
    else:
        # parallel process
        futs = [_pool.submit(f, *args) for args in f_args]
        dms = (fut.result() for fut in futs)

    return max(dms)


def _extract_messages_from_inputs_batched(
    batched_inputs_m,
    batched_inputs_t,
    input_locs_m,
    input_locs_t,
):
    """Get all messages as a dict from the batch stacked input form."""
    messages = {}
    for pair, (rank, i, b) in input_locs_m.items():
        messages[pair] = batched_inputs_m[rank][i, b, :]
    for pair, (rank, i, b) in input_locs_t.items():
        messages[pair] = batched_inputs_t[rank][i, b, :]
    return messages


def iterate_belief_propagation_batched(
    batched_inputs_m,
    batched_inputs_t,
    batched_tensors,
    masks_m,
    masks_t,
    smudge_factor=1e-12,
    damping=0.0,
    _pool=None,
):
    """ """
    # compute tensor messages
    batched_outputs_t = _compute_outputs_batched(
        batched_inputs=batched_inputs_t,
        batched_tensors=batched_tensors,
        smudge_factor=smudge_factor,
        _pool=_pool,
    )
    # update the index input messages
    t_max_dm = _update_outputs_to_inputs_batched(
        batched_inputs_m,
        batched_outputs_t,
        masks_m,
        damping=damping,
        _pool=_pool,
    )

    # compute index messages
    batched_outputs_m = _compute_outputs_batched(
        batched_inputs=batched_inputs_m,
        batched_tensors=None,
        smudge_factor=smudge_factor,
        _pool=_pool,
    )
    # update the tensor input messages
    m_max_dm = _update_outputs_to_inputs_batched(
        batched_inputs_t,
        batched_outputs_m,
        masks_t,
        damping=damping,
        _pool=_pool,
    )
    return batched_inputs_m, batched_inputs_t, max(t_max_dm, m_max_dm)


class HV1BP(BeliefPropagationCommon):
    """Object interface for hyper, vectorized, 1-norm, belief propagation. This
    is the fast version of belief propagation possible when there are many,
    small, matching tensor sizes.

    Parameters
    ----------
    tn : TensorNetwork
        The tensor network to run BP on.
    messages : dict, optional
        Initial messages to use, if not given then uniform messages are used.
    smudge_factor : float, optional
        A small number to add to the denominator of messages to avoid division
        by zero. Note when this happens the numerator will also be zero.
    thread_pool : bool or int, optional
        Whether to use a thread pool for parallelization, if ``True`` use the
        default number of threads, if an integer use that many threads.
    """

    def __init__(
        self,
        tn,
        messages=None,
        smudge_factor=1e-12,
        damping=0.0,
        thread_pool=False,
    ):
        self.tn = tn
        self.backend = next(t.backend for t in tn)
        self.smudge_factor = smudge_factor
        self.damping = damping
        self.pool = maybe_get_thread_pool(thread_pool)
        (
            self.batched_inputs_m,
            self.batched_inputs_t,
            self.batched_tensors,
            self.input_locs_m,
            self.input_locs_t,
            self.masks_m,
            self.masks_t,
        ) = initialize_messages_batched(tn, messages)

    def iterate(self, **kwargs):
        (
            self.batched_inputs_m,
            self.batched_inputs_t,
            max_dm,
        ) = iterate_belief_propagation_batched(
            self.batched_inputs_m,
            self.batched_inputs_t,
            self.batched_tensors,
            self.masks_m,
            self.masks_t,
            damping=self.damping,
            smudge_factor=self.smudge_factor,
            _pool=self.pool,
        )
        return None, None, max_dm

    def get_messages(self):
        """Get messages in individual form from the batched stacks."""
        return _extract_messages_from_inputs_batched(
            self.batched_inputs_m,
            self.batched_inputs_t,
            self.input_locs_m,
            self.input_locs_t,
        )

    def contract(self, strip_exponent=False):
        return contract_hyper_messages(
            self.tn,
            self.get_messages(),
            strip_exponent=strip_exponent,
            backend=self.backend,
        )


def contract_hv1bp(
    tn,
    messages=None,
    max_iterations=1000,
    tol=5e-6,
    smudge_factor=1e-12,
    damping=0.0,
    strip_exponent=False,
    info=None,
    progbar=False,
):
    """Estimate the contraction of ``tn`` with hyper, vectorized, 1-norm
    belief propagation, via the exponential of the Bethe free entropy.

    Parameters
    ----------
    tn : TensorNetwork
        The tensor network to run BP on, can have hyper indices.
    messages : dict, optional
        Initial messages to use, if not given then uniform messages are used.
    max_iterations : int, optional
        The maximum number of iterations to perform.
    tol : float, optional
        The convergence tolerance for messages.
    smudge_factor : float, optional
        A small number to add to the denominator of messages to avoid division
        by zero. Note when this happens the numerator will also be zero.
    damping : float, optional
        The damping factor to use, 0.0 means no damping.
    strip_exponent : bool, optional
        Whether to strip the exponent from the final result. If ``True``
        then the returned result is ``(mantissa, exponent)``.
    info : dict, optional
        If specified, update this dictionary with information about the
        belief propagation run.
    progbar : bool, optional
        Whether to show a progress bar.

    Returns
    -------
    scalar or (scalar, float)
    """
    bp = HV1BP(
        tn,
        messages=messages,
        damping=damping,
        smudge_factor=smudge_factor,
    )
    bp.run(
        max_iterations=max_iterations,
        tol=tol,
        info=info,
        progbar=progbar,
    )
    return bp.contract(strip_exponent=strip_exponent)


def run_belief_propagation_hv1bp(
    tn,
    messages=None,
    max_iterations=1000,
    tol=5e-6,
    damping=0.0,
    smudge_factor=1e-12,
    info=None,
    progbar=False,
):
    """Run belief propagation on a tensor network until it converges.

    Parameters
    ----------
    tn : TensorNetwork
        The tensor network to run BP on.
    messages : dict, optional
        The current messages. For every index and tensor id pair, there should
        be a message to and from with keys ``(ix, tid)`` and ``(tid, ix)``.
        If not given, then messages are initialized as uniform.
    max_iterations : int, optional
        The maximum number of iterations to run for.
    tol : float, optional
        The convergence tolerance.
    damping : float, optional
        The damping factor to use, 0.0 means no damping.
    smudge_factor : float, optional
        A small number to add to the denominator of messages to avoid division
        by zero. Note when this happens the numerator will also be zero.
    info : dict, optional
        If specified, update this dictionary with information about the
        belief propagation run.
    progbar : bool, optional
        Whether to show a progress bar.

    Returns
    -------
    messages : dict
        The final messages.
    converged : bool
        Whether the algorithm converged.
    """
    bp = HV1BP(
        tn, messages=messages, damping=damping, smudge_factor=smudge_factor
    )
    bp.run(max_iterations=max_iterations, tol=tol, info=info, progbar=progbar)
    return bp.get_messages(), bp.converged


def sample_hv1bp(
    tn,
    messages=None,
    output_inds=None,
    max_iterations=1000,
    tol=1e-2,
    damping=0.0,
    smudge_factor=1e-12,
    bias=False,
    seed=None,
    progbar=False,
):
    """Sample all indices of a tensor network using repeated belief propagation
    runs and decimation.

    Parameters
    ----------
    tn : TensorNetwork
        The tensor network to sample.
    messages : dict, optional
        The current messages. For every index and tensor id pair, there should
        be a message to and from with keys ``(ix, tid)`` and ``(tid, ix)``.
        If not given, then messages are initialized as uniform.
    output_inds : sequence of str, optional
        The indices to sample. If not given, then all indices are sampled.
    max_iterations : int, optional
        The maximum number of iterations for each message passing run.
    tol : float, optional
        The convergence tolerance for each message passing run.
    smudge_factor : float, optional
        A small number to add to each message to avoid zeros. Making this large
        is similar to adding a temperature, which can aid convergence but
        likely produces less accurate marginals.
    bias : bool or float, optional
        Whether to bias the sampling towards the largest marginal. If ``False``
        (the default), then indices are sampled proportional to their
        marginals. If ``True``, then each index is 'sampled' to be its largest
        weight value always. If a float, then the local probability
        distribution is raised to this power before sampling.
    thread_pool : bool, int or ThreadPoolExecutor, optional
        Whether to use a thread pool for parallelization. If an integer, then
        this is the number of threads to use. If ``True``, then the number of
        threads is set to the number of cores. If a ``ThreadPoolExecutor``,
        then this is used directly.
    seed : int, optional
        A random seed to use for the sampling.
    progbar : bool, optional
        Whether to show a progress bar.

    Returns
    -------
    config : dict[str, int]
        The sample configuration, mapping indices to values.
    tn_config : TensorNetwork
        The tensor network with all index values (or just those in
        `output_inds` if supllied) selected. Contracting this tensor network
        (which will just be a sequence of scalars if all index values have been
        sampled) gives the weight of the sample, e.g. should be 1 for a SAT
        problem and valid assignment.
    omega : float
        The probability of choosing this sample (i.e. product of marginal
        values). Useful possibly for importance sampling.
    """
    import numpy as np

    rng = np.random.default_rng(seed)

    tn_config = tn.copy()

    if messages is None:
        messages = initialize_hyper_messages(tn_config)

    if output_inds is None:
        output_inds = tn_config.ind_map.keys()
    output_inds = set(output_inds)

    config = {}
    omega = 1.0

    if progbar:
        import tqdm

        pbar = tqdm.tqdm(total=len(output_inds))
    else:
        pbar = None

    while output_inds:
        messages, _ = run_belief_propagation_hv1bp(
            tn_config,
            messages,
            max_iterations=max_iterations,
            tol=tol,
            damping=damping,
            smudge_factor=smudge_factor,
        )

        marginals = compute_all_index_marginals_from_messages(
            tn_config, messages
        )

        # choose most peaked marginal
        ix, p = max(
            (m for m in marginals.items() if m[0] in output_inds),
            key=lambda ix_p: max(ix_p[1]),
        )

        if bias is False:
            # sample the value according to the marginal
            v = rng.choice(np.arange(p.size), p=p)
        elif bias is True:
            v = np.argmax(p)
            # in some sense omega is really 1.0 here
        else:
            # bias towards larger marginals by raising to a power
            p = p**bias
            p /= np.sum(p)
            v = np.random.choice(np.arange(p.size), p=p)

        omega *= p[v]
        config[ix] = v

        # clean up messages
        for tid in tn_config.ind_map[ix]:
            del messages[ix, tid]
            del messages[tid, ix]

        # remove index
        tn_config.isel_({ix: v})
        output_inds.remove(ix)

        if progbar:
            pbar.update(1)
            pbar.set_description(f"{ix}->{v}", refresh=False)

    if progbar:
        pbar.close()

    return config, tn_config, omega
