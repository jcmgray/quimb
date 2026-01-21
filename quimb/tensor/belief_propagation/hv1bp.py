"""Hyper, vectorized, 1-norm, belief propagation."""

import autoray as ar

from quimb.tensor.contraction import array_contract

from .bp_common import (
    BeliefPropagationCommon,
    compute_all_index_marginals_from_messages,
    contract_hyper_messages,
    initialize_hyper_messages,
    maybe_get_thread_pool,
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
            arrays=(bx, *(mr[p] for p in range(mr.shape[0]))),
            inputs=((-1, *js), *((-1, j) for j in jr)),
            output=(-1, *jl),
            backend=backend,
        )

        # contract the left messages to get new right array
        xr = array_contract(
            arrays=(bx, *(ml[p] for p in range(ml.shape[0]))),
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
    _stack = ar.get_lib_fn(backend, "stack")

    ndim = len(bm)
    x_inds = (-1, *range(ndim))
    m_inds = [(-1, p) for p in range(ndim)]
    bmx = array_contract(
        arrays=(bx, *bm),
        inputs=(x_inds, *m_inds),
        output=x_inds,
    )

    bminv = 1 / (bm + smudge_factor)

    mouts = []
    for p in range(ndim):
        # sum all but ith index, apply inverse gate to that
        mouts.append(
            array_contract(
                arrays=(bmx, bminv[p]),
                inputs=(x_inds, m_inds[p]),
                output=m_inds[p],
            )
        )

    return _stack(mouts)


def _compute_output_single_t(
    bm,
    bx,
    normalize,
    smudge_factor=1e-12,
):
    # tensor messages
    bmo = _compute_all_tensor_messages_tree_batched(bx, bm)
    # bmo = _compute_all_tensor_messages_prod_batched(bx, bm, smudge_factor)
    normalize(bmo)
    return bmo


def _compute_output_single_m(bm, normalize, smudge_factor=1e-12):
    # index messages
    # bmo = _compute_all_hyperind_messages_tree_batched(bm)
    bmo = _compute_all_hyperind_messages_prod_batched(bm, smudge_factor)
    normalize(bmo)
    return bmo


def _update_output_to_input_single_batched(
    batched_input,
    batched_output,
    mask,
    _distance_fn,
    damping=0.0,
):
    # do a vectorized update
    select_in = (mask[0, 0, :], mask[0, 1, :], slice(None))
    select_out = (mask[1, 0, :], mask[1, 1, :], slice(None))
    bim = batched_input[select_in]
    bom = batched_output[select_out]

    # pre-damp distance
    mdiff = _distance_fn(bim, bom)

    if damping != 0.0:
        bom = damping * bim + (1 - damping) * bom

    # # post-damp distance
    # mdiff = _distance_fn(bim, bom)

    # update the input
    batched_input[select_in] = bom

    return mdiff


def _gather_zb(zb, power=1.0):
    """Given a vector of local contraction estimates `zb`, compute their
    product, avoiding underflow/overflow by accumulating the sign and exponent
    separately.

    Parameters
    ----------
    zb : array
        The local contraction estimates.
    power : float, optional
        Raise the final result to this power.

    Returns
    -------
    sign : float
        The accumulated sign or phase.
    exponent : float
        The accumulated exponent.
    """
    zb_mag = ar.do("abs", zb)
    zb_phase = zb / zb_mag

    # accumulate sign and exponent separately
    sign = ar.do("prod", zb_phase)
    exponent = ar.do("sum", ar.do("log10", zb_mag))

    if power != 1.0:
        sign **= power
        exponent *= power

    return sign, exponent


def _contract_index_region_single(bm):
    # take product over input position and sum over variable
    zb = ar.do("sum", ar.do("prod", bm, axis=0), axis=1)
    # that just leaves broadcast dimension to take product over
    return _gather_zb(zb)


def _contract_tensor_region_single(rank, batched_tensors, batched_inputs_t):
    bt = batched_tensors[rank]
    bms = batched_inputs_t[rank]
    # contract every tensor of rank `rank` with its messages
    zb = array_contract(
        [bt, *bms],
        inputs=[tuple(range(-1, rank))] + [(-1, r) for r in range(rank)],
        output=(-1,),
    )
    return _gather_zb(zb)


def _contract_messages_pair_single(
    ranki,
    ranko,
    mask,
    batched_inputs_m,
    batched_inputs_t,
):
    bmm = batched_inputs_m[ranki]
    bmt = batched_inputs_t[ranko]
    select_in = (mask[0, 0, :], mask[0, 1, :], slice(None))
    select_out = (mask[1, 0, :], mask[1, 1, :], slice(None))

    bml = bmm[select_in]
    bmr = bmt[select_out]

    zb = array_contract(
        [bml, bmr],
        inputs=[(-1, 0), (-1, 0)],
        output=(-1,),
    )

    # individual message reasons having counting factor -1
    # i.e. we are dividing by all of them
    return _gather_zb(zb, power=-1.0)


def _extract_messages_from_inputs_batched(
    batched_inputs_m,
    batched_inputs_t,
    input_locs_m,
    input_locs_t,
):
    """Get all messages as a dict from the batch stacked input form."""
    messages = {}
    for pair, (rank, p, b) in input_locs_m.items():
        messages[pair] = batched_inputs_m[rank][p, b, :]
    for pair, (rank, p, b) in input_locs_t.items():
        messages[pair] = batched_inputs_t[rank][p, b, :]
    return messages


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
    damping : float or callable, optional
        The damping factor to apply to messages. This simply mixes some part
        of the old message into the new one, with the final message being
        ``damping * old + (1 - damping) * new``. This makes convergence more
        reliable but slower.
    update : {'sequential', 'parallel'}, optional
        Whether to update messages sequentially (newly computed messages are
        immediately used for other updates in the same iteration round) or in
        parallel (all messages are comptued using messages from the previous
        round only). Sequential generally helps convergence but parallel can
        possibly converge to differnt solutions.
    normalize : {'L1', 'L2', 'L2phased', 'Linf', callable}, optional
        How to normalize messages after each update. If None choose
        automatically. If a callable, it should take a message and return the
        normalized message. If a string, it should be one of 'L1', 'L2',
        'L2phased', 'Linf' for the corresponding norms. 'L2phased' is like 'L2'
        but also normalizes the phase of the message, by default used for
        complex dtypes.
    distance : {'L1', 'L2', 'L2phased', 'Linf', 'cosine', callable}, optional
        How to compute the distance between messages to check for convergence.
        If None choose automatically. If a callable, it should take two
        messages and return the distance. If a string, it should be one of
        'L1', 'L2', 'L2phased', 'Linf', or 'cosine' for the corresponding
        norms. 'L2phased' is like 'L2' but also normalizes the phases of the
        messages, by default used for complex dtypes if phased normalization is
        not already being used.
    smudge_factor : float, optional
        A small number to add to the denominator of messages to avoid division
        by zero. Note when this happens the numerator will also be zero.
    thread_pool : bool or int, optional
        Whether to use a thread pool for parallelization, if ``True`` use the
        default number of threads, if an integer use that many threads.
    contract_every : int, optional
        If not None, 'contract' (via BP) the tensor network every
        ``contract_every`` iterations. The resulting values are stored in
        ``zvals`` at corresponding points ``zval_its``.
    inplace : bool, optional
        Whether to perform any operations inplace on the input tensor network.
    """

    def __init__(
        self,
        tn,
        *,
        messages=None,
        damping=0.0,
        update="parallel",
        normalize="L2",
        distance="L2",
        smudge_factor=1e-12,
        thread_pool=False,
        contract_every=None,
        inplace=False,
    ):
        super().__init__(
            tn,
            damping=damping,
            update=update,
            normalize=normalize,
            distance=distance,
            contract_every=contract_every,
            inplace=inplace,
        )

        if update != "parallel":
            raise ValueError("Only parallel update supported.")

        self.smudge_factor = smudge_factor
        self.pool = maybe_get_thread_pool(thread_pool)
        self.initialize_messages_batched(messages)

    @property
    def normalize(self):
        return self._normalize

    @normalize.setter
    def normalize(self, normalize):
        if callable(normalize):
            # custom normalization function
            _normalize = normalize
        elif normalize == "L1":
            _abs = ar.get_lib_fn(self.backend, "abs")
            _sum = ar.get_lib_fn(self.backend, "sum")

            def _normalize(bx):
                bxn = _sum(_abs(bx), axis=-1, keepdims=True)
                bx /= bxn

        elif normalize == "L2":
            _abs = ar.get_lib_fn(self.backend, "abs")
            _sum = ar.get_lib_fn(self.backend, "sum")

            def _normalize(bx):
                bxn = _sum(_abs(bx) ** 2, axis=-1, keepdims=True) ** 0.5
                bx /= bxn

        elif normalize == "Linf":
            _abs = ar.get_lib_fn(self.backend, "abs")
            _max = ar.get_lib_fn(self.backend, "max")

            def _normalize(bx):
                bxn = _max(_abs(bx), axis=-1, keepdims=True)
                bx /= bxn

        else:
            raise ValueError(f"Unrecognized normalize={normalize}")

        self._normalize = _normalize

    @property
    def distance(self):
        return self._distance

    @distance.setter
    def distance(self, distance):
        if callable(distance):
            # custom normalization function
            _distance_fn = distance

        elif distance == "L1":
            _abs = ar.get_lib_fn(self.backend, "abs")
            _sum = ar.get_lib_fn(self.backend, "sum")
            _max = ar.get_lib_fn(self.backend, "max")

            def _distance_fn(bx, by):
                return _max(_sum(_abs(bx - by), axis=-1))

        elif distance == "L2":
            _abs = ar.get_lib_fn(self.backend, "abs")
            _sum = ar.get_lib_fn(self.backend, "sum")
            _max = ar.get_lib_fn(self.backend, "max")

            def _distance_fn(bx, by):
                return _max(_sum(_abs(bx - by) ** 2, axis=-1)) ** 0.5

        elif distance == "Linf":
            _abs = ar.get_lib_fn(self.backend, "abs")
            _max = ar.get_lib_fn(self.backend, "max")

            def _distance_fn(bx, by):
                return _max(_abs(bx - by))

        else:
            raise ValueError(f"Unrecognized distance={distance}")

        self._distance = distance
        self._distance_fn = _distance_fn

    def initialize_messages_batched(self, messages=None):
        _stack = ar.get_lib_fn(self.backend, "stack")
        _array = ar.get_lib_fn(self.backend, "array")

        if isinstance(messages, dict):
            # 'dense' (i.e. non-batch) messages explicitly supplied
            message_init_fn = None
        elif callable(messages):
            # custom message initialization function
            message_init_fn = messages
            messages = None
        elif messages == "dense":
            # explicitly create dense messages first
            message_init_fn = None
            messages = initialize_hyper_messages(self.tn)
        elif messages is None:
            # default to uniform messages
            message_init_fn = ar.get_lib_fn(self.backend, "ones")
        else:
            raise ValueError(f"Unrecognized messages={messages}")

        # here we are stacking all contractions with matching rank
        #
        #     rank: number of incident messages to a tensor or hyper index
        #     pos (p): which of those messages we are (0, 1, ..., rank - 1)
        #     batch position (b): which position in the stack we are
        #
        #     _m = messages incident to indices
        #     _t = messages incident to tensors

        # prepare index messages
        batched_inputs_m = {}
        input_locs_m = {}
        output_locs_m = {}
        shapes_m = {}

        for ix, tids in self.tn.ind_map.items():
            # all updates of the same rank can be performed simultaneously
            rank = len(tids)
            try:
                batch = batched_inputs_m[rank]
                shape = shapes_m[rank]
            except KeyError:
                batch = batched_inputs_m[rank] = [[] for _ in range(rank)]
                shape = shapes_m[rank] = [rank, 0, self.tn.ind_size(ix)]

            # batch index
            b = shape[1]
            for p, tid in enumerate(tids):
                if message_init_fn is None:
                    # we'll stack the messages later
                    batch[p].append(messages[tid, ix])
                input_locs_m[tid, ix] = (rank, p, b)
                output_locs_m[ix, tid] = (rank, p, b)

            # increment batch index
            shape[1] = b + 1

        # prepare tensor messages
        batched_tensors = {}
        batched_inputs_t = {}
        input_locs_t = {}
        output_locs_t = {}
        shapes_t = {}

        for tid, t in self.tn.tensor_map.items():
            # all updates of the same rank can be performed simultaneously
            rank = t.ndim
            if rank == 0:
                # floating scalars are not updated
                continue

            try:
                batch = batched_inputs_t[rank]
                batch_t = batched_tensors[rank]
                shape = shapes_t[rank]
            except KeyError:
                batch = batched_inputs_t[rank] = [[] for _ in range(rank)]
                batch_t = batched_tensors[rank] = []
                shape = shapes_t[rank] = [rank, 0, t.shape[0]]

            # batch index
            b = shape[1]
            for p, ix in enumerate(t.inds):
                if message_init_fn is None:
                    # we'll stack the messages later
                    batch[p].append(messages[ix, tid])
                input_locs_t[ix, tid] = (rank, p, b)
                output_locs_t[tid, ix] = (rank, p, b)

            batch_t.append(t.data)
            # increment batch index
            shape[1] = b + 1

        # combine or create batch message arrays
        for batched_inputs, shapes in zip(
            (batched_inputs_m, batched_inputs_t), (shapes_m, shapes_t)
        ):
            for rank, batch in batched_inputs.items():
                if isinstance(messages, dict):
                    # stack given messages into single arrays
                    batched_inputs[rank] = _stack(
                        tuple(_stack(batch_p) for batch_p in batch)
                    )
                else:
                    # create message arrays directly
                    batched_inputs[rank] = message_init_fn(shapes[rank])

        # stack all tensors of each rank into a single array
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
                (ranki, pi, bi) = input_locs[pair]
                (ranko, po, bo) = output_locs[pair]
                # we can vectorize over all distinct pairs of ranks
                key = (ranki, ranko)
                try:
                    ma_pi, ma_po, ma_bi, ma_bo = masks[key]
                except KeyError:
                    ma_pi, ma_po, ma_bi, ma_bo = masks[key] = [], [], [], []

                ma_pi.append(pi)
                ma_bi.append(bi)
                ma_po.append(po)
                ma_bo.append(bo)

            for key, (ma_pi, ma_po, ma_bi, ma_bo) in masks.items():
                # first dimension is in/out
                # second dimension is position or batch
                # third dimension is stack index
                mask = _array([[ma_pi, ma_bi], [ma_po, ma_bo]])
                masks[key] = mask

        self.batched_inputs_m = batched_inputs_m
        self.batched_inputs_t = batched_inputs_t
        self.batched_tensors = batched_tensors
        self.input_locs_m = input_locs_m
        self.input_locs_t = input_locs_t
        self.masks_m = masks_m
        self.masks_t = masks_t

    @property
    def messages(self):
        return (self.batched_inputs_m, self.batched_inputs_t)

    @messages.setter
    def messages(self, messages):
        self.batched_inputs_m, self.batched_inputs_t = messages

    def _compute_outputs_batched(
        self,
        batched_inputs,
        batched_tensors=None,
    ):
        """Given stacked messsages and optionally tensors, compute stacked
        output messages, possibly using parallel pool.
        """

        if batched_tensors is not None:
            # tensor messages
            f = _compute_output_single_t
            f_args = {
                rank: (bm, batched_tensors[rank], self.normalize)
                for rank, bm in batched_inputs.items()
            }
        else:
            # index messages
            f = _compute_output_single_m
            f_args = {
                rank: (bm, self.normalize, self.smudge_factor)
                for rank, bm in batched_inputs.items()
            }

        batched_outputs = {}
        if self.pool is None:
            # sequential process
            for rank, args in f_args.items():
                batched_outputs[rank] = f(*args)
        else:
            # parallel process
            for rank, args in f_args.items():
                batched_outputs[rank] = self.pool.submit(f, *args)
            for key, fut in batched_outputs.items():
                batched_outputs[key] = fut.result()

        return batched_outputs

    def _update_outputs_to_inputs_batched(
        self,
        batched_inputs,
        batched_outputs,
        masks,
    ):
        """Update the stacked input messages from the stacked output messages."""
        f = _update_output_to_input_single_batched
        f_args = (
            (
                batched_inputs[ranki],
                batched_outputs[ranko],
                mask,
                self._distance_fn,
                self.damping,
            )
            for (ranki, ranko), mask in masks.items()
        )

        if self.pool is None:
            # sequential process
            mdiffs = (f(*args) for args in f_args)
        else:
            # parallel process
            futs = [self.pool.submit(f, *args) for args in f_args]
            mdiffs = (fut.result() for fut in futs)

        return max(mdiffs)

    def iterate(self, tol=None):
        # first we compute new tensor output messages
        batched_outputs_t = self._compute_outputs_batched(
            batched_inputs=self.batched_inputs_t,
            batched_tensors=self.batched_tensors,
        )
        # update the index input messages with these
        t_max_mdiff = self._update_outputs_to_inputs_batched(
            self.batched_inputs_m,
            batched_outputs_t,
            self.masks_m,
        )

        # compute index messages
        batched_outputs_m = self._compute_outputs_batched(
            batched_inputs=self.batched_inputs_m,
        )
        # update the tensor input messages
        m_max_mdiff = self._update_outputs_to_inputs_batched(
            self.batched_inputs_t,
            batched_outputs_m,
            self.masks_t,
        )
        return max(t_max_mdiff, m_max_mdiff)

    def get_messages_dense(self):
        """Get messages in individual form from the batched stacks."""
        return _extract_messages_from_inputs_batched(
            self.batched_inputs_m,
            self.batched_inputs_t,
            self.input_locs_m,
            self.input_locs_t,
        )

    def get_messages(self):
        import warnings

        warnings.warn(
            "get_messages() is deprecated, or in the future it might return "
            "the batch messages, use get_messages_dense() instead.",
            DeprecationWarning,
        )

        return self.get_messages_dense()

    def contract(self, strip_exponent=False, check_zero=False):
        """Estimate the contraction of the tensor network using the current
        messages. Uses batched vectorized contractions for speed.

        Parameters
        ----------
        strip_exponent : bool, optional
            Whether to strip the exponent from the final result. If ``True``
            then the returned result is ``(mantissa, exponent)``.
        check_zero : bool, optional
            Whether to check for zero values and return zero early. Currently
            ``True`` is not implemented for HV1BP.

        Returns
        -------
        scalar or (scalar, float)
        """
        if check_zero:
            raise NotImplementedError("check_zero not implemented for HV1BP.")

        fn_args = []
        # for each rank contract index region estimate
        for bm in self.batched_inputs_m.values():
            fn_args.append((_contract_index_region_single, (bm,)))
        # for each rank contract tensor region estimate
        for rank in self.batched_tensors:
            fn_args.append(
                (
                    _contract_tensor_region_single,
                    (rank, self.batched_tensors, self.batched_inputs_t),
                )
            )
        # for each pair of ranks contract messages pair
        # region estimate which we divide by (power=-1.0)
        for (ranki, ranko), mask in self.masks_m.items():
            fn_args.append(
                (
                    _contract_messages_pair_single,
                    (
                        ranki,
                        ranko,
                        mask,
                        self.batched_inputs_m,
                        self.batched_inputs_t,
                    ),
                )
            )

        if self.pool is None:
            results = [fn(*args) for fn, args in fn_args]
        else:
            futs = [self.pool.submit(fn, *args) for fn, args in fn_args]
            results = [fut.result() for fut in futs]

        sign = 1.0
        exponent = 0.0
        for s, e in results:
            sign *= s
            exponent += e

        if strip_exponent:
            return sign, exponent
        else:
            return sign * 10**exponent

    def contract_dense(self, strip_exponent=False, check_zero=True):
        """Slow contraction via explicit extranting individual dense messages.
        This supports check_zero=True and may be useful for debugging.
        """
        return contract_hyper_messages(
            self.tn,
            self.get_messages_dense(),
            strip_exponent=strip_exponent,
            check_zero=check_zero,
            backend=self.backend,
        )


def contract_hv1bp(
    tn,
    messages=None,
    max_iterations=1000,
    tol=5e-6,
    damping=0.0,
    diis=False,
    update="parallel",
    normalize="L2",
    distance="L2",
    tol_abs=None,
    tol_rolling_diff=None,
    smudge_factor=1e-12,
    strip_exponent=False,
    check_zero=False,
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
    damping : float, optional
        The damping factor to use, 0.0 means no damping.
    diis : bool or dict, optional
        Whether to use direct inversion in the iterative subspace to
        help converge the messages by extrapolating to low error guesses.
        If a dict, should contain options for the DIIS algorithm. The
        relevant options are {`max_history`, `beta`, `rcond`}.
    update : {'parallel'}, optional
        Whether to update messages sequentially or in parallel.
    normalize : {'L1', 'L2', 'L2phased', 'Linf', callable}, optional
        How to normalize messages after each update. If None choose
        automatically. If a callable, it should take a message and return the
        normalized message. If a string, it should be one of 'L1', 'L2',
        'L2phased', 'Linf' for the corresponding norms. 'L2phased' is like 'L2'
        but also normalizes the phase of the message, by default used for
        complex dtypes.
    distance : {'L1', 'L2', 'L2phased', 'Linf', 'cosine', callable}, optional
        How to compute the distance between messages to check for convergence.
        If None choose automatically. If a callable, it should take two
        messages and return the distance. If a string, it should be one of
        'L1', 'L2', 'L2phased', 'Linf', or 'cosine' for the corresponding
        norms. 'L2phased' is like 'L2' but also normalizes the phases of the
        messages, by default used for complex dtypes if phased normalization is
        not already being used.
    tol_abs : float, optional
        The absolute convergence tolerance for maximum message update
        distance, if not given then taken as ``tol``.
    tol_rolling_diff : float, optional
        The rolling mean convergence tolerance for maximum message update
        distance, if not given then taken as ``tol``. This is used to stop
        running when the messages are just bouncing around the same level,
        without any overall upward or downward trends, roughly speaking.
    smudge_factor : float, optional
        A small number to add to the denominator of messages to avoid division
        by zero. Note when this happens the numerator will also be zero.
    strip_exponent : bool, optional
        Whether to strip the exponent from the final result. If ``True``
        then the returned result is ``(mantissa, exponent)``.
    check_zero : bool, optional
        Whether to check for zero values and return zero early.
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
        update=update,
        normalize=normalize,
        distance=distance,
        smudge_factor=smudge_factor,
    )
    bp.run(
        max_iterations=max_iterations,
        tol=tol,
        diis=diis,
        tol_abs=tol_abs,
        tol_rolling_diff=tol_rolling_diff,
        info=info,
        progbar=progbar,
    )
    return bp.contract(
        strip_exponent=strip_exponent,
        check_zero=check_zero,
    )


def run_belief_propagation_hv1bp(
    tn,
    messages=None,
    max_iterations=1000,
    tol=5e-6,
    damping=0.0,
    diis=False,
    update="parallel",
    normalize="L2",
    distance="L2",
    tol_abs=None,
    tol_rolling_diff=None,
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
    diis : bool or dict, optional
        Whether to use direct inversion in the iterative subspace to
        help converge the messages by extrapolating to low error guesses.
        If a dict, should contain options for the DIIS algorithm. The
        relevant options are {`max_history`, `beta`, `rcond`}.
    update : {'parallel'}, optional
        Whether to update messages sequentially or in parallel.
    normalize : {'L1', 'L2', 'L2phased', 'Linf', callable}, optional
        How to normalize messages after each update. If None choose
        automatically. If a callable, it should take a message and return the
        normalized message. If a string, it should be one of 'L1', 'L2',
        'L2phased', 'Linf' for the corresponding norms. 'L2phased' is like 'L2'
        but also normalizes the phase of the message, by default used for
        complex dtypes.
    distance : {'L1', 'L2', 'L2phased', 'Linf', 'cosine', callable}, optional
        How to compute the distance between messages to check for convergence.
        If None choose automatically. If a callable, it should take two
        messages and return the distance. If a string, it should be one of
        'L1', 'L2', 'L2phased', 'Linf', or 'cosine' for the corresponding
        norms. 'L2phased' is like 'L2' but also normalizes the phases of the
        messages, by default used for complex dtypes if phased normalization is
        not already being used.
    tol_abs : float, optional
        The absolute convergence tolerance for maximum message update
        distance, if not given then taken as ``tol``.
    tol_rolling_diff : float, optional
        The rolling mean convergence tolerance for maximum message update
        distance, if not given then taken as ``tol``. This is used to stop
        running when the messages are just bouncing around the same level,
        without any overall upward or downward trends, roughly speaking.
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
        tn,
        messages=messages,
        damping=damping,
        smudge_factor=smudge_factor,
        update=update,
        normalize=normalize,
        distance=distance,
    )
    bp.run(
        max_iterations=max_iterations,
        tol=tol,
        diis=diis,
        tol_abs=tol_abs,
        tol_rolling_diff=tol_rolling_diff,
        info=info,
        progbar=progbar,
    )
    return bp.get_messages_dense(), bp.converged


def sample_hv1bp(
    tn,
    messages=None,
    output_inds=None,
    max_iterations=1000,
    tol=1e-2,
    damping=0.0,
    diis=False,
    update="parallel",
    normalize="L2",
    distance="L2",
    tol_abs=None,
    tol_rolling_diff=None,
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
    damping : float, optional
        The damping factor to use, 0.0 means no damping.
    diis : bool or dict, optional
        Whether to use direct inversion in the iterative subspace to
        help converge the messages by extrapolating to low error guesses.
        If a dict, should contain options for the DIIS algorithm. The
        relevant options are {`max_history`, `beta`, `rcond`}.
    update : {'parallel'}, optional
        Whether to update messages sequentially or in parallel.
    normalize : {'L1', 'L2', 'L2phased', 'Linf', callable}, optional
        How to normalize messages after each update. If None choose
        automatically. If a callable, it should take a message and return the
        normalized message. If a string, it should be one of 'L1', 'L2',
        'L2phased', 'Linf' for the corresponding norms. 'L2phased' is like 'L2'
        but also normalizes the phase of the message, by default used for
        complex dtypes.
    distance : {'L1', 'L2', 'L2phased', 'Linf', 'cosine', callable}, optional
        How to compute the distance between messages to check for convergence.
        If None choose automatically. If a callable, it should take two
        messages and return the distance. If a string, it should be one of
        'L1', 'L2', 'L2phased', 'Linf', or 'cosine' for the corresponding
        norms. 'L2phased' is like 'L2' but also normalizes the phases of the
        messages, by default used for complex dtypes if phased normalization is
        not already being used.
    tol_abs : float, optional
        The absolute convergence tolerance for maximum message update
        distance, if not given then taken as ``tol``.
    tol_rolling_diff : float, optional
        The rolling mean convergence tolerance for maximum message update
        distance, if not given then taken as ``tol``. This is used to stop
        running when the messages are just bouncing around the same level,
        without any overall upward or downward trends, roughly speaking.
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
            diis=diis,
            update=update,
            normalize=normalize,
            distance=distance,
            tol_abs=tol_abs,
            tol_rolling_diff=tol_rolling_diff,
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
