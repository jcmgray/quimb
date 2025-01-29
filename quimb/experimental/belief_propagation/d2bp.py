import contextlib

import autoray as ar

import quimb.tensor as qtn
from quimb.utils import oset

from .bp_common import (
    BeliefPropagationCommon,
    combine_local_contractions,
    normalize_message_pair,
)
from .regions import RegionGraph


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
    local_convergence : bool, optional
        Whether to allow messages to locally converge - i.e. if all their
        input messages have converged then stop updating them.
    contract_opts
        Other options supplied to ``cotengra.array_contract``.
    """

    def __init__(
        self,
        tn,
        *,
        messages=None,
        output_inds=None,
        optimize="auto-hq",
        damping=0.0,
        update="sequential",
        normalize=None,
        distance=None,
        inplace=False,
        local_convergence=True,
        **contract_opts,
    ):
        super().__init__(
            tn=tn,
            damping=damping,
            update=update,
            normalize=normalize,
            distance=distance,
            inplace=inplace,
        )

        self.contract_opts = contract_opts
        self.contract_opts.setdefault("optimize", optimize)
        self.local_convergence = local_convergence

        if output_inds is None:
            self.output_inds = set(self.tn.outer_inds())
        else:
            self.output_inds = set(output_inds)

        if messages is None:
            self.messages = {}
        else:
            self.messages = messages

        # record which messages touch each others, for efficient updates
        self.touch_map = {}
        self.touched = oset()
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
                    m = (t_in.reindex({ix: jx}).conj_() @ t_in).data
                    self.messages[ix, tid] = self._normalize_fn(m)

        # for efficiency setup all the contraction expressions ahead of time
        for ix, tids in self.tn.ind_map.items():
            if ix not in self.output_inds:
                self.build_expr(ix)

    def build_expr(self, ix):
        from quimb.tensor.contraction import array_contract_expression

        tids = self.tn.ind_map[ix]

        for tida, tidb in (sorted(tids), sorted(tids, reverse=True)):
            ta = self.tn.tensor_map[tida]
            kix = ta.inds
            bix = tuple(i if i in self.output_inds else i + "*" for i in kix)
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
        nconv = 0
        max_mdiff = -1.0
        new_touched = oset()

        def _compute_m(key):
            expr, data = self.exprs[key]
            m = expr(*data[:2], *(self.messages[mkey] for mkey in data[2:]))
            # enforce hermiticity and normalize
            return self._normalize_fn(m + ar.dag(m))

        def _update_m(key, new_m):
            nonlocal nconv, max_mdiff

            old_m = self.messages[key]

            # pre-damp distance
            mdiff = self._distance_fn(old_m, new_m)

            if self.damping:
                new_m = self.fn_damping(old_m, new_m)

            # # post-damp distance
            # mdiff = self._distance_fn(old_m, new_m)

            if mdiff > tol:
                # mark touching messages for update
                new_touched.update(self.touch_map[key])
            else:
                nconv += 1
            max_mdiff = max(max_mdiff, mdiff)
            self.messages[key] = new_m

        if self.update == "parallel":
            new_messages = {}
            # compute all new messages
            while self.touched:
                key = self.touched.pop()
                new_messages[key] = _compute_m(key)
            # insert all new messages
            for key, new_m in new_messages.items():
                _update_m(key, new_m)

        elif self.update == "sequential":
            # compute each new message and immediately re-insert it
            while self.touched:
                key = self.touched.pop()
                new_m = _compute_m(key)
                _update_m(key, new_m)

        self.touched = new_touched

        return {
            "nconv": nconv,
            "ncheck": ncheck,
            "max_mdiff": max_mdiff,
        }

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

    def normalize_message_pairs(self):
        """Normalize a pair of messages such that `<mi|mj> = 1` and
        `<mi|mi> = <mj|mj>` (but in general != 1).
        """
        _reshape = ar.get_lib_fn(self.backend, "reshape")

        for ix, tids in self.tn.ind_map.items():
            if len(tids) != 2:
                continue
            tida, tidb = tids
            ml = self.messages[ix, tida]
            mr = self.messages[ix, tidb]

            nml, nmr = normalize_message_pair(
                _reshape(ml, (-1,)),
                _reshape(mr, (-1,)),
            )

            self.messages[ix, tida] = _reshape(nml, ml.shape)
            self.messages[ix, tidb] = _reshape(nmr, mr.shape)

    def contract(
        self,
        strip_exponent=False,
        check_zero=True,
    ):
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
        zvals = []

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
            zvals.append((tval, 1))

        for ix, tids in self.tn.ind_map.items():
            if ix in self.output_inds:
                continue
            tida, tidb = tids
            ml = self.messages[ix, tidb]
            mr = self.messages[ix, tida]
            mval = qtn.array_contract(
                (ml, mr), ((1, 2), (1, 2)), (), **self.contract_opts
            )
            # counting factor is -1 i.e. divide by the message
            zvals.append((mval, -1))

        return combine_local_contractions(
            zvals,
            backend=self.backend,
            strip_exponent=strip_exponent,
            check_zero=check_zero,
        )

    def contract_cluster_expansion(
        self,
        clusters=None,
        autocomplete=True,
        optimize="auto-hq",
        strip_exponent=False,
        check_zero=True,
        info=None,
        progbar=False,
        **contract_opts,
    ):
        self.normalize_message_pairs()

        if isinstance(clusters, int):
            max_cluster_size = clusters
            clusters = None
        else:
            max_cluster_size = None

        if clusters is None:
            clusters = tuple(
                self.tn.gen_regions(max_region_size=max_cluster_size)
            )
        else:
            clusters = tuple(clusters)

        rg = RegionGraph(clusters, autocomplete=autocomplete)

        for tid in self.tn.tensor_map:
            rg.add_region([tid])

        if info is None:
            info = {}
        info.setdefault("contractions", {})
        contractions = info["contractions"]

        zvals = []

        if progbar:
            import tqdm

            it = tqdm.tqdm(rg.regions)
        else:
            it = rg.regions

        for region in it:
            counting_factor = rg.get_count(region)

            if counting_factor == 0:
                continue

            try:
                zr = contractions[region]
            except KeyError:
                k = self.tn._select_tids(region, virtual=False)
                b = k.conj()
                # apply gauge by contracting messages into ket layer
                for oix in k.outer_inds():
                    if oix in self.output_inds:
                        continue
                    (tid,) = k.ind_map[oix]
                    m = self.messages[oix, tid]
                    t = k.tensor_map[tid]
                    t.gate_(m, oix)
                zr = (k | b).contract(
                    optimize=optimize,
                    **contract_opts,
                )
                contractions[region] = zr

            zvals.append((zr, counting_factor))

        return combine_local_contractions(
            zvals,
            backend=self.backend,
            strip_exponent=strip_exponent,
            check_zero=check_zero,
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

    def gauge_insert(self, tn, smudge=1e-12):
        """Insert the sqrt of messages on the boundary of a part of the main BP
        TN.

        Parameters
        ----------
        tn : TensorNetwork
            The tensor network to insert the messages into.
        smudge : float, optional
            Smudge factor to avoid numerical issues, the eigenvalues of the
            messages are clipped to be at least the largest eigenvalue times
            this factor.

        Returns
        -------
        list[tuple[Tensor, str, array_like]]
            The sequence of tensors, indices and inverse gauges to apply to
            reverse the gauges applied.
        """
        outer = []

        _eigh = ar.get_lib_fn(self.backend, "linalg.eigh")
        _clip = ar.get_lib_fn(self.backend, "clip")
        _sqrt = ar.get_lib_fn(self.backend, "sqrt")

        for ix in tn.outer_inds():
            # get the tensor and dangling index
            (tid,) = tn.ind_map[ix]
            try:
                m = self.messages[ix, tid]
            except KeyError:
                # could be phsyical index or not generated yet
                continue
            t = tn.tensor_map[tid]

            # compute the 'square root' of the message
            s2, W = _eigh(m)
            s2 = _clip(s2, s2[-1] * smudge, None)
            s = _sqrt(s2)
            msqrt = qtn.decomp.ldmul(s, ar.dag(W))
            msqrt_inv = qtn.decomp.rddiv(W, s)
            t.gate_(msqrt, ix)
            outer.append((t, ix, msqrt_inv))

        return outer

    @contextlib.contextmanager
    def gauge_temp(self, tn, ungauge_outer=True):
        """Context manager to temporarily gauge a tensor network, presumably a
        subnetwork of the main BP network, using the current messages, and then
        un-gauge it afterwards.

        Parameters
        ----------
        tn : TensorNetwork
            The tensor network to gauge.
        ungauge_outer : bool, optional
            Whether to un-gauge the outer indices of the tensor network.
        """
        outer = self.gauge_insert(tn)
        try:
            yield outer
        finally:
            if ungauge_outer:
                for t, ix, msqrt_inv in outer:
                    t.gate_(msqrt_inv, ix)

    def gate_(
        self,
        G,
        where,
        max_bond=None,
        cutoff=0.0,
        cutoff_mode="rsum2",
        renorm=0,
        tn=None,
        **gate_opts,
    ):
        """Apply a gate to the tensor network at the specified sites, using
        the current messages to gauge the tensors.
        """
        if len(where) == 1:
            # single site gate
            self.tn.gate_(G, where, contract=True)
            return

        gate_opts.setdefault("contract", "reduce-split")

        if tn is None:
            tn = self.tn
        site_tags = tuple(map(tn.site_tag, where))
        tn_where = tn.select_any(site_tags)

        with self.gauge_temp(tn_where):
            # contract and split the gate
            tn_where.gate_(
                G,
                where,
                max_bond=max_bond,
                cutoff=cutoff,
                cutoff_mode=cutoff_mode,
                renorm=renorm,
                **gate_opts,
            )

            # update the messages for this bond
            taga, tagb = site_tags
            (tida,) = tn._get_tids_from_tags(taga)
            (tidb,) = tn._get_tids_from_tags(tagb)
            ta = tn.tensor_map[tida]
            tb = tn.tensor_map[tidb]
            lix, (ix,), rix = qtn.group_inds(ta, tb)

            # make use of the fact that we already have gauged tensors
            A = ta.to_dense(lix, (ix,))
            B = tb.to_dense((ix,), rix)
            ma = ar.dag(A) @ A
            mb = B @ ar.dag(B)

            shape_changed = self.messages[ix, tidb].shape != ma.shape

            self.messages[ix, tidb] = ma
            self.messages[ix, tida] = mb

            # mark the sites as touched
            self.update_touched_from_tids(tida, tidb)
            if shape_changed:
                # rebuild the contraction expressions if shapes changed
                for cix in (*lix, ix, *rix):
                    if cix not in self.output_inds:
                        self.build_expr(cix)


def contract_d2bp(
    tn,
    *,
    messages=None,
    output_inds=None,
    max_iterations=1000,
    tol=5e-6,
    damping=0.0,
    diis=False,
    update="sequential",
    normalize=None,
    distance=None,
    tol_abs=None,
    tol_rolling_diff=None,
    local_convergence=True,
    optimize="auto-hq",
    strip_exponent=False,
    check_zero=True,
    info=None,
    progbar=False,
    **contract_opts,
):
    """Estimate the norm squared of ``tn`` using dense 2-norm belief
    propagation (no hyper indices).

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
    local_convergence : bool, optional
        Whether to allow messages to locally converge - i.e. if all their
        input messages have converged then stop updating them.
    optimize : str or PathOptimizer, optional
        The path optimizer to use when contracting the messages.
    strip_exponent : bool, optional
        Whether to return the mantissa and exponent separately.
    check_zero : bool, optional
        Whether to check for zero values and return zero early.
    info : dict, optional
        If supplied, the following information will be added to it:
        ``converged`` (bool), ``iterations`` (int), ``max_mdiff`` (float),
        ``rolling_abs_mean_diff`` (float).
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
        update=update,
        normalize=normalize,
        distance=distance,
        **contract_opts,
    )
    bp.run(
        max_iterations=max_iterations,
        diis=diis,
        tol=tol,
        tol_abs=tol_abs,
        tol_rolling_diff=tol_rolling_diff,
        info=info,
        progbar=progbar,
    )
    return bp.contract(
        strip_exponent=strip_exponent,
        check_zero=check_zero,
    )


def compress_d2bp(
    tn,
    max_bond,
    cutoff=0.0,
    cutoff_mode="rsum2",
    renorm=0,
    messages=None,
    output_inds=None,
    max_iterations=1000,
    tol=5e-6,
    damping=0.0,
    diis=False,
    update="sequential",
    normalize=None,
    distance=None,
    tol_abs=None,
    tol_rolling_diff=None,
    local_convergence=True,
    optimize="auto-hq",
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
    renorm : float, optional
        Whether to renormalize the singular values when compressing.
    messages : dict[(str, int), array_like], optional
        The initial messages to use, effectively defaults to all ones if not
        specified.
    output_inds : set[str], optional
        The indices to consider as output (dangling) indices of the tn.
        Computed automatically if not specified.
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
    local_convergence : bool, optional
        Whether to allow messages to locally converge - i.e. if all their
        input messages have converged then stop updating them.
    optimize : str or PathOptimizer, optional
        The path optimizer to use when contracting the messages.
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
        damping=damping,
        update=update,
        normalize=normalize,
        distance=distance,
        local_convergence=local_convergence,
        inplace=inplace,
        **contract_opts,
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
    optimize="auto-hq",
    damping=0.0,
    diis=False,
    update="sequential",
    normalize=None,
    distance=None,
    tol_abs=None,
    tol_rolling_diff=None,
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
    optimize : str or PathOptimizer, optional
        The path optimizer to use when contracting the messages.
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
        optimize=optimize,
        damping=damping,
        update=update,
        normalize=normalize,
        distance=distance,
        local_convergence=local_convergence,
        **contract_opts,
    )
    bp.run(
        max_iterations=max_iterations,
        tol=tol,
        diis=diis,
        tol_abs=tol_abs,
        tol_rolling_diff=tol_rolling_diff,
    )

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
            messages=messages,
            optimize=optimize,
            damping=damping,
            update=update,
            normalize=normalize,
            distance=distance,
            local_convergence=local_convergence,
            **contract_opts,
        )
        bp.update_touched_from_tids(*tids)
        bp.run(
            max_iterations=max_iterations,
            tol=tol,
            diis=diis,
            tol_abs=tol_abs,
            tol_rolling_diff=tol_rolling_diff,
        )

    if progbar:
        pbar.close()

    return config, tn, omega
