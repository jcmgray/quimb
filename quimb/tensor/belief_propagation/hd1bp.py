"""Hyper dense belief propagation for arbitrary `quimb` tensor networks. This
is the classic 1-norm version of belief propagation, which treats the tensor
network directly as a factor graph. Messages are processed one at a time.

TODO:

- [ ] implement 'touching', so that only necessary messages are updated
- [ ] implement sequential update

"""

import autoray as ar

import quimb.tensor as qtn

from .bp_common import (
    BeliefPropagationCommon,
    combine_local_contractions,
    compute_all_index_marginals_from_messages,
    contract_hyper_messages,
    initialize_hyper_messages,
    prod,
)


def compute_all_hyperind_messages_prod(ms, smudge_factor=1e-12):
    """Given set of messages ``ms`` incident to a single index, compute the
    corresponding next output messages, using the 'product' implementation.
    """
    if len(ms) == 2:
        # shortcut for 2 messages
        return [ms[1], ms[0]]

    x = prod(ms)
    return [x / (m + smudge_factor) for m in ms]


def compute_all_hyperind_messages_tree(ms):
    """Given set of messages ``ms`` incident to a single index, compute the
    corresponding next output messages, using the 'tree' implementation.
    """
    ndim = len(ms)
    if len(ms) == 2:
        # shortcut for 2 messages
        return [ms[1], ms[0]]

    mouts = [None for _ in range(ndim)]
    queue = [(tuple(range(ndim)), 1, ms)]

    while queue:
        js, x, ms = queue.pop()

        ndim = len(ms)
        if ndim == 1:
            # reached single message
            mouts[js[0]] = x
            continue
        elif ndim == 2:
            # shortcut for 2 messages left
            mouts[js[0]] = x * ms[1]
            mouts[js[1]] = ms[0] * x
            continue

        # else split in two and contract each half
        k = ndim // 2
        jl, jr = js[:k], js[k:]
        ml, mr = ms[:k], ms[k:]

        # contract the right messages to get new left array
        xl = prod((*mr, x))

        # contract the left messages to get new right array
        xr = prod((*ml, x))

        # add the queue for possible further halving
        queue.append((jl, xl, ml))
        queue.append((jr, xr, mr))

    return mouts


def compute_all_tensor_messages_shortcuts(x, ms, ndim):
    if ndim == 2:
        # shortcut for 2 messages
        return [x @ ms[1], ms[0] @ x]
    elif ndim == 1:
        # shortcut for single message
        return [x]
    elif ndim == 0:
        # shortcut for no messages
        return []


def compute_all_tensor_messages_prod(
    x,
    ms,
    backend=None,
    smudge_factor=1e-12,
):
    """Given set of messages ``ms`` incident to tensor with data ``x``, compute
    the corresponding next output messages, using the 'prod' implementation.
    """
    ndim = len(ms)
    if ndim <= 2:
        return compute_all_tensor_messages_shortcuts(x, ms, ndim)

    js = tuple(range(ndim))

    mx = qtn.array_contract(
        arrays=(x, *ms), inputs=(js, *((j,) for j in js)), output=js
    )
    mouts = []

    for j, g in enumerate(ms):
        mouts.append(
            qtn.array_contract(
                arrays=(mx, 1 / (g + smudge_factor)),
                inputs=(js, (j,)),
                output=(j,),
                backend=backend,
            )
        )

    return mouts


def compute_all_tensor_messages_tree(x, ms, backend=None):
    """Given set of messages ``ms`` incident to tensor with data ``x``, compute
    the corresponding next output messages, using the 'tree' implementation.
    """
    ndim = len(ms)
    if ndim <= 2:
        return compute_all_tensor_messages_shortcuts(x, ms, ndim)

    mouts = [None for _ in range(ndim)]
    queue = [(tuple(range(ndim)), x, ms)]

    while queue:
        js, x, ms = queue.pop()

        ndim = len(ms)
        if ndim == 1:
            # reached single message
            mouts[js[0]] = x
            continue
        elif ndim == 2:
            # shortcut for 2 messages left
            mouts[js[0]] = x @ ms[1]
            mouts[js[1]] = ms[0] @ x
            continue

        # else split in two and contract each half
        k = ndim // 2
        jl, jr = js[:k], js[k:]
        ml, mr = ms[:k], ms[k:]

        # contract the right messages to get new left array
        xl = qtn.array_contract(
            arrays=(x, *mr),
            inputs=(js, *((j,) for j in jr)),
            output=jl,
            backend=backend,
        )

        # contract the left messages to get new right array
        xr = qtn.array_contract(
            arrays=(x, *ml),
            inputs=(js, *((j,) for j in jl)),
            output=jr,
            backend=backend,
        )

        # add the queue for possible further halving
        queue.append((jl, xl, ml))
        queue.append((jr, xr, mr))

    return mouts


class HD1BP(BeliefPropagationCommon):
    """Object interface for hyper, dense, 1-norm belief propagation. This is
    standard belief propagation in tensor network form.

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
    inplace : bool, optional
        Whether to perform any operations inplace on the input tensor network.
    """

    def __init__(
        self,
        tn,
        *,
        messages=None,
        damping=0.0,
        update="sequential",
        normalize=None,
        distance=None,
        smudge_factor=1e-12,
        inplace=False,
    ):
        super().__init__(
            tn,
            damping=damping,
            update=update,
            normalize=normalize,
            distance=distance,
            inplace=inplace,
        )

        self.smudge_factor = smudge_factor

        if callable(messages):
            messages = initialize_hyper_messages(
                tn, fill_fn=messages, smudge_factor=smudge_factor
            )
        elif messages is None:
            messages = initialize_hyper_messages(
                tn, smudge_factor=smudge_factor
            )
        self.messages = messages

    def iterate(self, tol=None):
        if self.update == "sequential":
            new_messages = self.messages
        else:
            new_messages = {}

        def _normalize_and_insert(key, new_m, max_mdiff):
            # normalize and insert
            new_m = self._normalize_fn(new_m)
            old_m = self.messages[key]

            # pre-damp distance
            mdiff = self._distance_fn(old_m, new_m)

            if self.damping:
                new_m = self._damping_fn(old_m, new_m)

            # # post-damp distance
            # mdiff = self._distance_fn(old_m, new_m)

            max_mdiff = max(mdiff, max_mdiff)

            # set and return the max diff so far
            new_messages[key] = new_m
            return max_mdiff

        max_mdiff = 0.0

        # hyper index messages
        for ix, tids in self.tn.ind_map.items():
            ms = compute_all_hyperind_messages_prod(
                [self.messages[tid, ix] for tid in tids],
                self.smudge_factor,
            )
            for tid, m in zip(tids, ms):
                max_mdiff = _normalize_and_insert((ix, tid), m, max_mdiff)

        if self.update == "parallel":
            self.messages.update(new_messages)
            new_messages.clear()

        # tensor messages
        for tid, t in self.tn.tensor_map.items():
            inds = t.inds
            ms = compute_all_tensor_messages_tree(
                t.data,
                [self.messages[ix, tid] for ix in inds],
            )
            for ix, m in zip(inds, ms):
                max_mdiff = _normalize_and_insert((tid, ix), m, max_mdiff)

        if self.update == "parallel":
            self.messages.update(new_messages)

        return max_mdiff

    def get_gauged_tn(self):
        """Assuming the supplied tensor network has no hyper or dangling
        indices, gauge it by inserting the BP-approximated transfer matrix
        eigenvectors, which may be complex. The BP-contraction of this gauged
        network is then simply the product of zeroth entries of each tensor.
        """
        tng = self.tn.copy()
        for ind, tids in self.tn.ind_map.items():
            tida, tidb = tids
            ka = (ind, tida)
            kb = (ind, tidb)
            ma = self.messages[ka]
            mb = self.messages[kb]

            el, ev = ar.do("linalg.eig", ar.do("outer", ma, mb))
            k = ar.do("argsort", -ar.do("abs", el))
            ev = ev[:, k]
            Uinv = ev
            U = ar.do("linalg.inv", ev)
            tng._insert_gauge_tids(U, tida, tidb, Uinv)
        return tng

    def contract(self, strip_exponent=False, check_zero=True):
        """Estimate the total contraction, i.e. the exponential of the 'Bethe
        free entropy'.
        """
        return contract_hyper_messages(
            self.tn,
            self.messages,
            strip_exponent=strip_exponent,
            check_zero=check_zero,
            backend=self.backend,
        )

    def normalize_messages(self):
        """Normalize all messages such that the 'region contraction' of a
        single hyper index is 1.
        """
        for ind, tids in self.tn.ind_map.items():
            ms = [self.messages[tid, ind] for tid in tids]
            overlap = qtn.array_contract(ms, [(0,) for _ in ms], [])
            overlap **= 1 / len(ms)
            for tid, m in zip(tids, ms):
                self.messages[tid, ind] = m / overlap

    def get_cluster(self, r, virtual=True, autocomplete=True):
        """Get the tensor network of a region ``r``, with all boundary
        messages attached.

        Parameters
        ----------
        r : sequence of int or str
            The region to get, given as a sequence of indices or tensor ids.
        virtual : bool, optional
            Whether the view the original tensors (`virtual=True`, the default)
            or take copies (`virtual=False`).
        autocomplete : bool, optional
            Whether to automatically include all indices attached to the
            tensors in the region, or just the ones given in ``r``.

        Returns
        -------
        TensorNetwork
        """
        rtids = set()
        rinds = set()
        for x in r:
            if isinstance(x, int):
                rtids.add(x)
                if autocomplete:
                    rinds.update(self.tn.tensor_map[x].inds)
            else:
                rinds.add(x)

        tnr = self.tn._select_tids(rtids, virtual=virtual)
        for ind in rinds:
            # attach all messages coming from tensors outside the cluster
            for ntid in self.tn.ind_map[ind]:
                if ntid not in rtids:
                    tnr |= qtn.Tensor(
                        data=self.messages[ntid, ind], inds=(ind,)
                    )

        return tnr

    def contract_gloop_expand(
        self,
        gloops=None,
        strip_exponent=False,
        check_zero=True,
        optimize="auto-hq",
        progbar=False,
        **contract_otps,
    ):
        from .regions import RegionGraph

        # if we normalized messages we can ignore all index-only regions
        self.normalize_messages()

        if isinstance(gloops, int):
            gloops = tuple(self.tn.gen_gloops(gloops))

        rg = RegionGraph(gloops)

        if progbar:
            import tqdm

            regions = tqdm.tqdm(rg.regions)
        else:
            regions = rg.regions

        zvals = []
        for r in regions:
            # XXX: autoreduce intersecting clusters to gloops?
            cr = rg.get_count(r)
            # either we autocomplete above or we do it here per region
            tnr = self.get_cluster(r, autocomplete=True)
            zr = tnr.contract(
                output_inds=(), optimize=optimize, **contract_otps
            )
            zvals.append((zr, cr))

        return combine_local_contractions(
            zvals,
            strip_exponent=strip_exponent,
            check_zero=check_zero,
            backend=self.backend,
        )


def contract_hd1bp(
    tn,
    messages=None,
    max_iterations=1000,
    tol=5e-6,
    damping=0.0,
    diis=False,
    update="sequential",
    normalize=None,
    distance=None,
    tol_abs=None,
    tol_rolling_diff=None,
    smudge_factor=1e-12,
    strip_exponent=False,
    check_zero=True,
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
        The damping parameter to use, defaults to no damping.
    diis : bool or dict, optional
        Whether to use direct inversion in the iterative subspace to
        help converge the messages by extrapolating to low error guesses.
        If a dict, should contain options for the DIIS algorithm. The
        relevant options are {`max_history`, `beta`, `rcond`}.
    update : {'sequential', 'parallel'}, optional
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
    bp = HD1BP(
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


def run_belief_propagation_hd1bp(
    tn,
    messages=None,
    max_iterations=1000,
    tol=5e-6,
    damping=0.0,
    smudge_factor=1e-12,
    info=None,
    progbar=False,
):
    """Run belief propagation on a tensor network until it converges. This
    is the basic version that does not vectorize contractions.

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
    bp = HD1BP(
        tn, messages=messages, damping=damping, smudge_factor=smudge_factor
    )
    bp.run(max_iterations=max_iterations, tol=tol, info=info, progbar=progbar)
    return bp.messages, bp.converged


def sample_hd1bp(
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
        messages, _ = run_belief_propagation_hd1bp(
            tn_config,
            messages,
            max_iterations=max_iterations,
            tol=tol,
            damping=damping,
            smudge_factor=smudge_factor,
            progbar=True,
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
