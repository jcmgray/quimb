import autoray as ar
import quimb.tensor as qtn

from .bp_common import (
    BeliefPropagationCommon,
    combine_local_contractions,
)


class D2BP(BeliefPropagationCommon):
    """Dense (as in one tensor per site) 2-norm (as in for wavefunctions and
    operators) belief propagation. Allows messages reuse. This version assumes
    no hyper indices (i.e. a standard PEPS like tensor network).

    Potential use cases for D2BP and a PEPS like tensor network are:

        - globally compressing it from bond dimension ``D`` to ``D'``
        - eagerly applying gates and locally compressing back to ``D``
        - sampling configurations
        - estimating the norm of the tensor network


    Parameters
    ----------
    tn : TensorNetwork
        The tensor network to form the 2-norm of and run BP on.
    messages : dict[(str, int), array_like], optional
        The initial messages to use, effectively defaults to all ones if not
        specified.
    output_inds : set[str], optional
        The indices to consider as output (dangling) indices of the tn.
        Computed automatically if not specified.
    optimize : str or PathOptimizer, optional
        The path optimizer to use when contracting the messages.
    local_convergence : bool, optional
        Whether to allow messages to locally converge - i.e. if all their
        input messages have converged then stop updating them.
    contract_opts
        Other options supplied to ``cotengra.array_contract``.
    """

    def __init__(
        self,
        tn,
        messages=None,
        output_inds=None,
        optimize="auto-hq",
        local_convergence=True,
        damping=0.0,
        **contract_opts,
    ):
        from quimb.tensor.contraction import array_contract_expression

        self.tn = tn
        self.contract_opts = contract_opts
        self.contract_opts.setdefault("optimize", optimize)
        self.local_convergence = local_convergence
        self.damping = damping

        if output_inds is None:
            self.output_inds = set(self.tn.outer_inds())
        else:
            self.output_inds = set(output_inds)

        self.backend = next(t.backend for t in tn)
        _abs = ar.get_lib_fn(self.backend, "abs")
        _sum = ar.get_lib_fn(self.backend, "sum")

        def _normalize(x):
            return x / _sum(x)

        def _distance(x, y):
            return _sum(_abs(x - y))

        self._normalize = _normalize
        self._distance = _distance

        if messages is None:
            self.messages = {}
        else:
            self.messages = messages

        # record which messages touch each others, for efficient updates
        self.touch_map = {}
        self.touched = set()
        self.exprs = {}

        # populate any messages
        for ix, tids in self.tn.ind_map.items():
            if ix in self.output_inds:
                continue

            tida, tidb = tids
            jx = ix + "*"
            ta, tb = self.tn._tids_get(tida, tidb)

            for tid, t, t_in in ((tida, ta, tb), (tidb, tb, ta)):
                this_touchmap = []
                for nx in t.inds:
                    if nx in self.output_inds or nx == ix:
                        continue
                    # where this message will be sent on to
                    (tidn,) = (n for n in self.tn.ind_map[nx] if n != tid)
                    this_touchmap.append((nx, tidn))
                self.touch_map[ix, tid] = this_touchmap

                if (ix, tid) not in self.messages:
                    tm = (t_in.reindex({ix: jx}).conj_() @ t_in).data
                    self.messages[ix, tid] = self._normalize(tm.data)

        # for efficiency setup all the contraction expressions ahead of time
        for ix, tids in self.tn.ind_map.items():
            if ix in self.output_inds:
                continue

            for tida, tidb in (sorted(tids), sorted(tids, reverse=True)):
                ta = self.tn.tensor_map[tida]
                kix = ta.inds
                bix = tuple(
                    i if i in self.output_inds else i + "*" for i in kix
                )
                inputs = [kix, bix]
                data = [ta.data, ta.data.conj()]
                shapes = [ta.shape, ta.shape]
                for i in kix:
                    if (i != ix) and i not in self.output_inds:
                        inputs.append((i + "*", i))
                        data.append((i, tida))
                        shapes.append(self.messages[i, tida].shape)

                expr = array_contract_expression(
                    inputs=inputs,
                    output=(ix + "*", ix),
                    shapes=shapes,
                    **self.contract_opts,
                )
                self.exprs[ix, tidb] = expr, data

    def update_touched_from_tids(self, *tids):
        """Specify that the messages for the given ``tids`` have changed."""
        for tid in tids:
            t = self.tn.tensor_map[tid]
            for ix in t.inds:
                if ix in self.output_inds:
                    continue
                (ntid,) = (n for n in self.tn.ind_map[ix] if n != tid)
                self.touched.add((ix, ntid))

    def update_touched_from_tags(self, tags, which="any"):
        """Specify that the messages for the messages touching ``tags`` have
        changed.
        """
        tids = self.tn._get_tids_from_tags(tags, which)
        self.update_touched_from_tids(*tids)

    def update_touched_from_inds(self, inds, which="any"):
        """Specify that the messages for the messages touching ``inds`` have
        changed.
        """
        tids = self.tn._get_tids_from_inds(inds, which)
        self.update_touched_from_tids(*tids)

    def iterate(self, tol=5e-6):
        """Perform a single iteration of dense 2-norm belief propagation."""

        if (not self.local_convergence) or (not self.touched):
            # assume if asked to iterate that we want to check all messages
            self.touched.update(self.exprs.keys())

        ncheck = len(self.touched)
        new_messages = {}
        while self.touched:
            key = self.touched.pop()
            expr, data = self.exprs[key]
            m = expr(*data[:2], *(self.messages[mkey] for mkey in data[2:]))
            # enforce hermiticity and normalize
            m = m + ar.dag(m)
            m = self._normalize(m)

            if self.damping > 0.0:
                m = self._normalize(
                    # new message
                    (1 - self.damping) * m
                    +
                    # old message
                    self.damping * self.messages[key]
                )

            new_messages[key] = m

        # process modified messages
        nconv = 0
        max_mdiff = -1.0
        for key, m in new_messages.items():
            mdiff = float(self._distance(m, self.messages[key]))

            if mdiff > tol:
                # mark touching messages for update
                self.touched.update(self.touch_map[key])
            else:
                nconv += 1

            max_mdiff = max(max_mdiff, mdiff)
            self.messages[key] = m

        return nconv, ncheck, max_mdiff

    def compute_marginal(self, ind):
        """Compute the marginal for the index ``ind``."""
        (tid,) = self.tn.ind_map[ind]
        t = self.tn.tensor_map[tid]

        arrays = [t.data, ar.do("conj", t.data)]
        k_input = []
        b_input = []
        m_inputs = []
        for j, jx in enumerate(t.inds, 1):
            k_input.append(j)

            if jx == ind:
                # output index -> take diagonal
                output = (j,)
                b_input.append(j)
            else:
                try:
                    # partial trace with message
                    m = self.messages[jx, tid]
                    arrays.append(m)
                    b_input.append(-j)
                    m_inputs.append((-j, j))
                except KeyError:
                    # direct partial trace
                    b_input.append(j)

        p = qtn.array_contract(
            arrays,
            inputs=(tuple(k_input), tuple(b_input), *m_inputs),
            output=output,
            **self.contract_opts,
        )
        p = ar.do("real", p)
        return p / ar.do("sum", p)

    def contract(self, strip_exponent=False):
        """Estimate the total contraction, i.e. the 2-norm.

        Parameters
        ----------
        strip_exponent : bool, optional
            Whether to strip the exponent from the final result. If ``True``
            then the returned result is ``(mantissa, exponent)``.

        Returns
        -------
        scalar or (scalar, float)
        """
        tvals = []

        for tid, t in self.tn.tensor_map.items():
            arrays = [t.data, ar.do("conj", t.data)]
            k_input = []
            b_input = []
            m_inputs = []
            for i, ix in enumerate(t.inds, 1):
                k_input.append(i)
                if ix in self.output_inds:
                    b_input.append(i)
                else:
                    b_input.append(-i)
                    m_inputs.append((-i, i))
                    arrays.append(self.messages[ix, tid])

            inputs = (tuple(k_input), tuple(b_input), *m_inputs)
            output = ()
            tval = qtn.array_contract(
                arrays, inputs, output, **self.contract_opts
            )
            tvals.append(tval)

        mvals = []
        for ix, tids in self.tn.ind_map.items():
            if ix in self.output_inds:
                continue
            tida, tidb = tids
            ml = self.messages[ix, tidb]
            mr = self.messages[ix, tida]
            mval = qtn.array_contract(
                (ml, mr), ((1, 2), (1, 2)), (), **self.contract_opts
            )
            mvals.append(mval)

        return combine_local_contractions(
            tvals, mvals, self.backend, strip_exponent=strip_exponent
        )

    def compress(
        self,
        max_bond,
        cutoff=0.0,
        cutoff_mode=4,
        renorm=0,
        inplace=False,
    ):
        """Compress the initial tensor network using the current messages."""
        tn = self.tn if inplace else self.tn.copy()

        for ix, tids in tn.ind_map.items():
            if len(tids) != 2:
                continue
            tida, tidb = tids

            # messages are left and right factors squared already
            ta = tn.tensor_map[tida]
            dm = ta.ind_size(ix)
            dl = ta.size // dm
            ml = self.messages[ix, tidb]
            Rl = qtn.decomp.squared_op_to_reduced_factor(
                ml, dl, dm, right=True
            )

            tb = tn.tensor_map[tidb]
            dr = tb.size // dm
            mr = self.messages[ix, tida].T
            Rr = qtn.decomp.squared_op_to_reduced_factor(
                mr, dm, dr, right=False
            )

            # compute the compressors
            Pl, Pr = qtn.decomp.compute_oblique_projectors(
                Rl,
                Rr,
                max_bond=max_bond,
                cutoff=cutoff,
                cutoff_mode=cutoff_mode,
                renorm=renorm,
            )

            # contract the compressors into the tensors
            tn.tensor_map[tida].gate_(Pl.T, ix)
            tn.tensor_map[tidb].gate_(Pr, ix)

            # update messages with projections
            if inplace:
                new_Ra = Rl @ Pl
                new_Rb = Pr @ Rr
                self.messages[ix, tidb] = ar.dag(new_Ra) @ new_Ra
                self.messages[ix, tida] = new_Rb @ ar.dag(new_Rb)

        return tn


def contract_d2bp(
    tn,
    messages=None,
    output_inds=None,
    optimize="auto-hq",
    local_convergence=True,
    damping=0.0,
    max_iterations=1000,
    tol=5e-6,
    strip_exponent=False,
    info=None,
    progbar=False,
    **contract_opts,
):
    """Estimate the norm squared of ``tn`` using dense 2-norm belief
    propagation.

    Parameters
    ----------
    tn : TensorNetwork
        The tensor network to form the 2-norm of and run BP on.
    messages : dict[(str, int), array_like], optional
        The initial messages to use, effectively defaults to all ones if not
        specified.
    max_iterations : int, optional
        The maximum number of iterations to perform.
    tol : float, optional
        The convergence tolerance for messages.
    output_inds : set[str], optional
        The indices to consider as output (dangling) indices of the tn.
        Computed automatically if not specified.
    optimize : str or PathOptimizer, optional
        The path optimizer to use when contracting the messages.
    local_convergence : bool, optional
        Whether to allow messages to locally converge - i.e. if all their
        input messages have converged then stop updating them.
    damping : float, optional
        The damping parameter to use, defaults to no damping.
    strip_exponent : bool, optional
        Whether to strip the exponent from the final result. If ``True``
        then the returned result is ``(mantissa, exponent)``.
    info : dict, optional
        If specified, update this dictionary with information about the
        belief propagation run.
    progbar : bool, optional
        Whether to show a progress bar.
    contract_opts
        Other options supplied to ``cotengra.array_contract``.

    Returns
    -------
    scalar or (scalar, float)
    """
    bp = D2BP(
        tn,
        messages=messages,
        output_inds=output_inds,
        optimize=optimize,
        local_convergence=local_convergence,
        damping=damping,
        **contract_opts,
    )
    bp.run(
        max_iterations=max_iterations,
        tol=tol,
        info=info,
        progbar=progbar,
    )
    return bp.contract(strip_exponent=strip_exponent)


def compress_d2bp(
    tn,
    max_bond,
    cutoff=0.0,
    cutoff_mode="rsum2",
    renorm=0,
    messages=None,
    output_inds=None,
    optimize="auto-hq",
    local_convergence=True,
    damping=0.0,
    max_iterations=1000,
    tol=5e-6,
    inplace=False,
    info=None,
    progbar=False,
    **contract_opts,
):
    """Compress the tensor network ``tn`` using dense 2-norm belief
    propagation.

    Parameters
    ----------
    tn : TensorNetwork
        The tensor network to form the 2-norm of, run BP on and then compress.
    max_bond : int
        The maximum bond dimension to compress to.
    cutoff : float, optional
        The cutoff to use when compressing.
    cutoff_mode : int, optional
        The cutoff mode to use when compressing.
    messages : dict[(str, int), array_like], optional
        The initial messages to use, effectively defaults to all ones if not
        specified.
    max_iterations : int, optional
        The maximum number of iterations to perform.
    tol : float, optional
        The convergence tolerance for messages.
    output_inds : set[str], optional
        The indices to consider as output (dangling) indices of the tn.
        Computed automatically if not specified.
    optimize : str or PathOptimizer, optional
        The path optimizer to use when contracting the messages.
    damping : float, optional
        The damping parameter to use, defaults to no damping.
    local_convergence : bool, optional
        Whether to allow messages to locally converge - i.e. if all their
        input messages have converged then stop updating them.
    inplace : bool, optional
        Whether to perform the compression inplace.
    info : dict, optional
        If specified, update this dictionary with information about the
        belief propagation run.
    progbar : bool, optional
        Whether to show a progress bar.
    contract_opts
        Other options supplied to ``cotengra.array_contract``.

    Returns
    -------
    TensorNetwork
    """
    bp = D2BP(
        tn,
        messages=messages,
        output_inds=output_inds,
        optimize=optimize,
        local_convergence=local_convergence,
        damping=damping,
        **contract_opts,
    )
    bp.run(
        max_iterations=max_iterations,
        tol=tol,
        info=info,
        progbar=progbar,
    )
    return bp.compress(
        max_bond=max_bond,
        cutoff=cutoff,
        cutoff_mode=cutoff_mode,
        renorm=renorm,
        inplace=inplace,
    )


def sample_d2bp(
    tn,
    output_inds=None,
    messages=None,
    max_iterations=100,
    tol=1e-2,
    bias=None,
    seed=None,
    local_convergence=True,
    progbar=False,
    **contract_opts,
):
    """Sample a configuration from ``tn`` using dense 2-norm belief
    propagation.

    Parameters
    ----------
    tn : TensorNetwork
        The tensor network to sample from.
    output_inds : set[str], optional
        Which indices to sample.
    messages : dict[(str, int), array_like], optional
        The initial messages to use, effectively defaults to all ones if not
        specified.
    max_iterations : int, optional
        The maximum number of iterations to perform, per marginal.
    tol : float, optional
        The convergence tolerance for messages.
    bias : float, optional
        Bias the sampling towards more locally likely bit-strings. This is
        done by raising the probability of each bit-string to this power.
    seed : int, optional
        A random seed for reproducibility.
    local_convergence : bool, optional
        Whether to allow messages to locally converge - i.e. if all their
        input messages have converged then stop updating them.
    progbar : bool, optional
        Whether to show a progress bar.
    contract_opts
        Other options supplied to ``cotengra.array_contract``.

    Returns
    -------
    config : dict[str, int]
        The sampled configuration, a mapping of output indices to values.
    tn_config : TensorNetwork
        The tensor network with the sampled configuration applied.
    omega : float
        The BP probability of the sampled configuration.
    """
    import numpy as np

    if output_inds is None:
        output_inds = tn.outer_inds()

    rng = np.random.default_rng(seed)
    config = {}
    omega = 1.0

    tn = tn.copy()
    bp = D2BP(
        tn,
        messages=messages,
        local_convergence=local_convergence,
        **contract_opts,
    )
    bp.run(max_iterations=max_iterations, tol=tol)

    marginals = dict.fromkeys(output_inds)

    if progbar:
        import tqdm

        pbar = tqdm.tqdm(total=len(marginals))
    else:
        pbar = None

    while marginals:
        for ix in marginals:
            marginals[ix] = bp.compute_marginal(ix)

        ix, p = max(marginals.items(), key=lambda x: max(x[1]))
        p = ar.to_numpy(p)

        if bias is not None:
            # bias distribution towards more locally likely bit-strings
            p = p**bias
            p /= np.sum(p)

        v = rng.choice([0, 1], p=p)
        config[ix] = v
        del marginals[ix]

        tids = tuple(tn.ind_map[ix])
        tn.isel_({ix: v})

        omega *= p[v]
        if progbar:
            pbar.update(1)
            pbar.set_description(f"{ix}->{v}", refresh=False)

        bp = D2BP(
            tn,
            messages=bp.messages,
            local_convergence=local_convergence,
            **contract_opts,
        )
        bp.update_touched_from_tids(*tids)
        bp.run(tol=tol, max_iterations=max_iterations)

    if progbar:
        pbar.close()

    return config, tn, omega
