"""Dense 2-norm belief propagation for standard PEPS like tensor networks, with
one tensor per site and no hyper indices. This is the basic 'quantum' BP.

TODO:
- [ ] cache gauges computed from messages until out-of-date
- [ ] fix fermionic compress and gate (non-hermitian messages currently?)
- [ ] store conditioned messages separately?
"""

import contextlib
import functools
import itertools
import operator

import autoray as ar

import quimb.tensor as qtn
from quimb.utils import ensure_dict, oset

from .bp_common import (
    BeliefPropagationCommon,
    combine_local_contractions,
    normalize_message_pair,
    process_loop_series_expansion_weights,
)
from .regions import gen_region_counts


def _parse_global_gloops(tn, gloops=None):
    if isinstance(gloops, int):
        max_size = gloops
        gloops = None
    else:
        max_size = None

    if gloops is None:
        gloops = tuple(tn.gen_gloops(max_size=max_size))
    else:
        gloops = tuple(gloops)

    return gloops


@functools.lru_cache(maxsize=128)
def _get_message_conditioner(power=1.0, smudge=0.0, backend=None):
    """Get a function to condition squared BP messages spectrally."""

    if power == 1.0:
        if smudge == 0.0:

            def conditioner(m):
                return m

        else:
            if backend is None:
                _eigh = ar.DoFunc("linalg.eigh")
                _clip = ar.DoFunc("clip")
                _sqrt = ar.DoFunc("sqrt")
            else:
                _eigh = ar.get_lib_fn(backend, "linalg.eigh")
                _clip = ar.get_lib_fn(backend, "clip")
                _sqrt = ar.get_lib_fn(backend, "sqrt")

            def conditioner(m):
                el, ev = _eigh(m)
                el = _clip(el, 0.0, None)
                el = _sqrt(el) + smudge
                el = el**2
                return ev @ qtn.decomp.ldmul(el, ar.dag(ev))

    else:
        if smudge == 0.0:
            if backend is None:
                _eigh = ar.DoFunc("linalg.eigh")
                _clip = ar.DoFunc("clip")
            else:
                _eigh = ar.get_lib_fn(backend, "linalg.eigh")
                _clip = ar.get_lib_fn(backend, "clip")

            def conditioner(m):
                el, ev = _eigh(m)
                el = _clip(el, 0.0, None)
                el = el**power
                return ev @ qtn.decomp.ldmul(el, ar.dag(ev))

        else:
            if backend is None:
                _eigh = ar.DoFunc("linalg.eigh")
                _clip = ar.DoFunc("clip")
                _sqrt = ar.DoFunc("sqrt")
            else:
                _eigh = ar.get_lib_fn(backend, "linalg.eigh")
                _clip = ar.get_lib_fn(backend, "clip")
                _sqrt = ar.get_lib_fn(backend, "sqrt")

            def conditioner(m):
                el, ev = _eigh(m)
                el = _clip(el, 0.0, None)
                el = _sqrt(el) + smudge
                el = el ** (2 * power)
                return ev @ qtn.decomp.ldmul(el, ar.dag(ev))

    return conditioner


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
    power : float, optional
        The power used to condition the square-root message spectrum. Each
        message eigenvalue ``el`` is transformed to
        ``(sqrt(max(el, 0)) + smudge) ** (2 * power)``.
    smudge : float, optional
        The value added to the square-root message spectrum before applying
        ``power`` and reconstructing the squared message.
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
    contract_every : int, optional
        If not None, 'contract' (via BP) the tensor network every
        ``contract_every`` iterations. The resulting values are stored in
        ``zvals`` at corresponding points ``zval_its``.
    inplace : bool, optional
        Whether to perform any operations inplace on the input tensor network.
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
        power=1.0,
        smudge=0.0,
        normalize=None,
        distance=None,
        local_convergence=True,
        contract_every=None,
        inplace=False,
        **contract_opts,
    ):
        super().__init__(
            tn=tn,
            damping=damping,
            update=update,
            normalize=normalize,
            distance=distance,
            contract_every=contract_every,
            inplace=inplace,
        )
        self.contract_opts = contract_opts
        self.contract_opts.setdefault("optimize", optimize)
        self.local_convergence = local_convergence
        self._power = power
        self._smudge = smudge
        self._update_message_conditioner()

        if output_inds is None:
            self.output_inds = set(self.tn.outer_inds())
        else:
            self.output_inds = set(output_inds)

        if messages is None:
            self.messages = {}
        else:
            self.messages = messages

        self._initialize_contract_expressions()

    def _initialize_contract_expressions(self):
        # record which messages touch each others, for efficient updates
        self.touch_map = {}
        self.touched = oset()
        self.exprs = {}
        self.index_dual_map = {}
        self.tensor_dual_map = {}
        for tid in self.tn.tensor_map:
            self._init_tid(tid)

    def _update_message_conditioner(self):
        self._message_conditioner = _get_message_conditioner(
            self._power,
            self._smudge,
            self.backend,
        )
        if hasattr(self, "touched"):
            # all messages potentially need to be recomputed
            self.touched.update(self.exprs)

    @property
    def power(self):
        """The power used to condition the square-root message spectrum.

        Setting this marks all message expressions for recomputation.
        """
        return self._power

    @power.setter
    def power(self, power):
        if power != self._power:
            self._power = power
            self._update_message_conditioner()

    @property
    def smudge(self):
        """The value added to the square-root message spectrum.

        Setting this marks all message expressions for recomputation.
        """
        return self._smudge

    @smudge.setter
    def smudge(self, smudge):
        if smudge != self._smudge:
            self._smudge = smudge
            self._update_message_conditioner()

    def _init_tid(self, tid):
        """Setup any missing input messages and build contraction expressions
        for output message updates for each bond around tensor at `tid`.
        """
        from quimb.tensor.contraction import array_contract_expression

        t = self.tn.tensor_map[tid]
        ix_neighbors = {}
        axs_output = []

        # first build initial messages from ix->tid
        for ax, ix in enumerate(t.inds):
            if ix in self.output_inds:
                # output index -> directly contract without message
                # if fermionic we may need to phase flip -> record
                axs_output.append(ax)
                continue

            # bond index -> mangle for bra
            try:
                ixc = self.index_dual_map[ix]
            except KeyError:
                ixc = qtn.rand_uuid()
                self.index_dual_map[ix] = ixc

            # get neighbor tid
            tidn = next(tidn for tidn in self.tn.ind_map[ix] if tidn != tid)
            ix_neighbors[ix] = tidn

            if (ix, tid) not in self.messages:
                # only create missing messages
                # fermions: use select here to generate initial cluster phases
                k = self.tn._select_tids([tidn], virtual=False)
                b = k.conj().reindex({ix: ixc})
                m = (b | k).to_dense((ixc,), (ix,))
                m = self._message_conditioner(m)
                m = self._normalize_fn(m)
                self.messages[ix, tid] = m

            # make sure touch_map entry exists
            self.touch_map.setdefault((ix, tid), {})

        t_dag = t.conj().reindex_(self.index_dual_map)
        if t_dag.isfermionic():
            # need to phase dual output indices only
            data = t_dag.data
            axs_phase = tuple(
                ax for ax in axs_output if not data.indices[ax].dual
            )
            if axs_phase:
                t_dag.modify(data=data.phase_flip(*axs_phase))

        self.tensor_dual_map[tid] = t_dag
        kix = t.inds
        bix = t_dag.inds

        # then we build contraction expressions for tid->ix message updates
        for ix_next, tid_next in ix_neighbors.items():
            inputs = [bix, kix]
            data = [t_dag.data, t.data]
            shapes = [t_dag.shape, t.shape]
            # define the messages
            for k, b in zip(kix, bix):
                if k == ix_next:
                    # the message output *bond* index
                    output = (b, k)
                elif k != b:
                    # inner bond with associated message -> attach
                    inputs.append((b, k))
                    data.append((k, tid))
                    shapes.append(self.messages[k, tid].shape)
                    # also populate touch_map for this message:
                    # (k->tid) propagates to (ix_next->tid_next)
                    self.touch_map[k, tid][ix_next, tid_next] = None
                # else:
                # global output -> directly traced without message

            expr = array_contract_expression(
                inputs=inputs,
                output=output,
                shapes=shapes,
                **self.contract_opts,
            )
            self.exprs[ix_next, tid_next] = expr, data

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
            x, xc, *mkeys = data
            ms = [self.messages[mkey] for mkey in mkeys]

            # contract update!
            m = expr(x, xc, *ms)

            # for stability enforce hermiticity
            m = m + ar.dag(m)

            # possibly condition the message spectrum
            m = self._message_conditioner(m)

            # finally normalize the message
            return self._normalize_fn(m)

        def _update_m(key, new_m):
            nonlocal nconv, max_mdiff

            old_m = self.messages[key]

            # pre-damp distance
            mdiff = self._distance_fn(old_m, new_m)

            if self.damping:
                new_m = self._damping_fn(old_m, new_m)

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

    def local_tensor_contract(self, tid):
        """Contract the local region of the tensor at ``tid``."""
        t = self.tn.tensor_map[tid]
        t_dag = self.tensor_dual_map[tid]
        arrays = [t.data, t_dag.data]
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
                m = self.messages[ix, tid]
                arrays.append(m)

        inputs = (tuple(k_input), tuple(b_input), *m_inputs)
        output = ()
        return qtn.array_contract(arrays, inputs, output, **self.contract_opts)

    def normalize_tensors(self, strip_exponent=True):
        """Normalize the tensors in the tensor network such that their 2-norm
        is 1. If ``strip_exponent`` is ``True`` then accrue the phase and
        exponent (log10) into the ``sign`` and ``exponent`` attributes of the
        D2BP object (the default), contract methods can then reinsert these
        factors when returning the final result.
        """
        for tid, t in self.tn.tensor_map.items():
            tval = self.local_tensor_contract(tid)
            tabs = ar.do("abs", tval)
            tsgn = tval / tabs
            tlog = ar.do("log10", tabs)
            nfact = (tsgn * tabs) ** 0.5
            t /= nfact
            # keep cached dual tensor in sync
            self.tensor_dual_map[tid] /= ar.do("conj", nfact)
            if strip_exponent:
                self.sign = tsgn * self.sign
                self.exponent = tlog + self.exponent

    def contract(
        self,
        strip_exponent=False,
        check_zero=True,
        **kwargs,
    ):
        """Contract the frobenius norm squared of the target tensor network via
        BP.

        Parameters
        ----------
        strip_exponent : bool, optional
            Whether to strip the exponent from the final result. If ``True``
            then the returned result is ``(mantissa, exponent)``.
        check_zero : bool, optional
            Whether to check for zero values and return zero early.

        Returns
        -------
        scalar or (scalar, float)
        """
        zvals = []

        for tid in self.tn.tensor_map:
            tval = self.local_tensor_contract(tid)
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
            mantissa=self.sign**2,
            exponent=self.exponent * 2,
            **kwargs,
        )

    def get_cluster_excited(
        self,
        tids=None,
        partial_trace_map=(),
        exclude=(),
    ):
        """Get the local norm tensor network for ``tids`` with BP messages
        inserted on the boundary and excitation projectors inserted on the
        inner bonds. See arxiv.org/abs/2409.03108 for more details.

        Parameters
        ----------
        tids : iterable of hashable
            The tensor ids to include in the cluster.
        partial_trace_map : dict[str, str], optional
            A remapping of ket indices to bra indices to perform an effective
            partial trace.
        exclude : iterable of str, optional
            A set of bond indices to exclude from inserting excitation
            projectors on, e.g. when forming a reduced density matrix.

        Returns
        -------
        TensorNetwork
        """
        stn = self.tn._select_tids(tids)

        kixmaps = {tid: {} for tid in stn.tensor_map}
        bixmaps = {tid: {} for tid in stn.tensor_map}
        exc_ixs = {}
        bms = []
        ems = []

        for ix, tids in stn.ind_map.items():
            if ix in self.output_inds:
                # physical traced index
                if ix in partial_trace_map:
                    (tid,) = tids
                    bixmaps[tid][ix] = partial_trace_map[ix]

            elif ix in exclude:
                # excluded bond -> simply rename bra
                bix = qtn.rand_uuid()
                for tid in tids:
                    bixmaps[tid][ix] = bix

            elif ix in stn._inner_inds:
                # internal index
                for tid in tids:
                    kix = qtn.rand_uuid()
                    bix = qtn.rand_uuid()
                    kixmaps[tid][ix] = kix
                    bixmaps[tid][ix] = bix
                    # store labels for excitation projector, (bra, ket)
                    exc_ixs.setdefault(ix, {})[tid] = (bix, kix)
                ems.append((ix, tids))

            else:
                # boundary index
                (tid,) = tids
                kix = qtn.rand_uuid()
                bix = qtn.rand_uuid()
                kixmaps[tid][ix] = kix
                bixmaps[tid][ix] = bix
                bms.append((ix, tid))

        tn = qtn.TensorNetwork()

        # add bra and ket tensors
        for tid in stn.tensor_map:
            t = stn.tensor_map[tid]
            tn |= t.reindex(kixmaps[tid])
            tn |= t.conj().reindex(bixmaps[tid])

        # add boundary message tensors
        for ix, tid in bms:
            data = self.messages[ix, tid]
            # message index ordering is (bra, ket)
            inds = (bixmaps[tid][ix], kixmaps[tid][ix])
            tn |= qtn.Tensor(data, inds)

        # add inner exitation projector message tensors
        with ar.backend_like(self.backend):
            for ix, tids in ems:
                tidl, tidr = tids
                ml = self.messages[ix, tidl]
                mr = self.messages[ix, tidr]

                # form outer product
                p0 = ar.do("einsum", "i,j->ij", ml.reshape(-1), mr.reshape(-1))
                # subtract from identity
                pe = ar.do("eye", ar.do("shape", p0)[0]) - p0
                # reshape back into 4-tensor
                pe = ar.do(
                    "reshape",
                    pe,
                    ar.do("shape", ml) + ar.do("shape", mr),
                )
                inds = (*exc_ixs[ix][tidl], *exc_ixs[ix][tidr])
                tn |= qtn.Tensor(pe, inds)

        return tn

    def contract_loop_series_expansion(
        self,
        gloops=None,
        multi_excitation_correct=True,
        tol_correction=1e-12,
        maxiter_correction=100,
        strip_exponent=False,
        optimize="auto-hq",
        **contract_opts,
    ):
        """Contract the norm of the tensor network using the same procedure as
        in https://arxiv.org/abs/2409.03108 - "Loop Series Expansions for
        Tensor Networks".

        Parameters
        ----------
        gloops : int or iterable of tuples, optional
            The gloop sizes to use. If an integer, then generate all gloop
            sizes up to this size. If a tuple, then use these gloops.
        multi_excitation_correct : bool, optional
            Whether to use the multi-excitation correction. If ``True``, then
            the free energy is refined iteratively until self consistent.
        tol_correction : float, optional
            The tolerance for the multi-excitation correction.
        maxiter_correction : int, optional
            The maximum number of iterations for the multi-excitation
            correction.
        strip_exponent : bool, optional
            Whether to strip the exponent from the final result. If ``True``
            then the returned result is ``(mantissa, exponent)``.
        optimize : str or PathOptimizer, optional
            The path optimizer to use when contracting the messages.
        contract_opts
            Other options supplied to ``TensorNetwork.contract``.
        """
        self.normalize_message_pairs()
        # accrues BP estimate into self.sign and self.exponent
        self.normalize_tensors()

        gloops = _parse_global_gloops(self.tn, gloops)

        weights = {}
        for gloop in gloops:
            # get local tensor network with boundary
            # messages and excitation projectors inserted
            etn = self.get_cluster_excited(gloop)
            # contract it to get local weight!
            weights[tuple(gloop)] = etn.contract(
                optimize=optimize, **contract_opts
            )

        return process_loop_series_expansion_weights(
            weights,
            mantissa=self.sign,
            exponent=self.exponent,
            multi_excitation_correct=multi_excitation_correct,
            tol_correction=tol_correction,
            maxiter_correction=maxiter_correction,
            strip_exponent=strip_exponent,
        )

    def partial_trace_loop_series_expansion(
        self,
        where,
        gloops=None,
        normalized=True,
        grow_from="alldangle",
        strict_size=False,
        multi_excitation_correct=True,
        optimize="auto-hq",
        **contract_opts,
    ):
        """Compute the reduced density matrix for the sites specified by
        ``where`` using the loop series expansion method from
        https://arxiv.org/abs/2409.03108 - "Loop Series Expansions for Tensor
        Networks".

        Parameters
        ----------
        where : sequence[hashable]
            The sites to from the reduced density matrix of.
        gloops : int or iterable of tuples, optional
            The generalized loops to use, or an integer to automatically
            generate all up to a certain size. If none use the smallest non-
            trivial size.
        normalized : bool, optional
            Whether to normalize the final density matrix.
        grow_from : {'alldangle', 'all', 'any'}, optional
            How to grow the generalized loops from the specified ``where``:

            - 'alldangle': clusters up to max size, where target sites are
              allowed to dangle.
            - 'all': clusters where loop, up to max size, has to include *all*
              target sites.
            - 'any': clusters where loop, up to max size, can include *any* of
              the target sites. Remaining target sites are added as extras.

            By default 'alldangle'.
        strict_size : bool, optional
            Whether to enforce the maximum size of the generalized loops, only
            relevant for `grow_from="any"`.
        multi_excitation_correct : bool, optional
            Whether to use the multi-excitation correction. If ``True``, then
            the free energy is refined iteratively until self consistent.
        optimize : str or PathOptimizer, optional
            The path optimizer to use when contracting the messages.
        contract_opts
            Other options supplied to ``TensorNetwork.contract``.
        """
        self.normalize_message_pairs()
        self.normalize_tensors()

        tags = [self.tn.site_tag(coo) for coo in where]
        tids = self.tn._get_tids_from_tags(tags, "any")

        # get a mapping of ket indices to bra indices on target sites
        kix = [self.tn.site_ind(coo) for coo in where]
        bix = [qtn.rand_uuid() for _ in where]
        partial_trace_map = dict(zip(kix, bix))
        output_inds = (*kix, *bix)

        # generate the generalized loops relevant to the target sites
        gloops = self.tn.get_local_gloops(
            tids=tids,
            gloops=gloops,
            grow_from=grow_from,
            strict_size=strict_size,
        )
        # the base (BP) region, including target sites only
        r0 = frozenset(tids)

        # get internal indices of the base region to exclude
        # from inserting excited space projectors on
        inner_bonds = self.tn._select_tids(tids).inner_inds()

        # get loop excited reduced density matrices
        rho_es = {}
        for gloop in gloops:
            etn = self.get_cluster_excited(
                gloop, exclude=inner_bonds, partial_trace_map=partial_trace_map
            )
            rho_e = etn.contract(
                output_inds=output_inds, optimize=optimize, **contract_opts
            )
            rho_e = rho_e.to_dense(kix, bix)

            if (normalized == "local") and gloop != r0:
                rho_e /= 1 + ar.do("trace", rho_e)

            rho_es[gloop] = rho_e

        if multi_excitation_correct:
            # trace of each density matrix is its corresponding
            # loops contribution to the norm free energy
            weights = {
                gloop: ar.do("trace", rho_e) for gloop, rho_e in rho_es.items()
            }
            # remove the BP contribution (= minimal region)
            weights.pop(r0)
            # compute exponential suppresion factors
            corrections = process_loop_series_expansion_weights(
                weights, return_all=True
            )
            # add back in the BP contribution
            corrections[r0] = 1.0

            # weighted sum
            rho = functools.reduce(
                operator.add,
                (
                    rho_e * corrections[gloop]
                    for gloop, rho_e in rho_es.items()
                ),
            )
        else:
            rho = functools.reduce(operator.add, rho_es.values())

        if normalized:
            rho /= ar.do("trace", rho)
        elif (self.sign, self.exponent) != (1.0, 0.0):
            # have been accrued into by normalize_tensors most likely
            rho *= self.sign * 10**self.exponent

        return rho

    def contract_gloop_expand(
        self,
        gloops=None,
        autocomplete=True,
        optimize="auto-hq",
        strip_exponent=False,
        check_zero=True,
        info=None,
        progbar=False,
        **contract_opts,
    ):
        self.normalize_message_pairs()

        gloops = _parse_global_gloops(self.tn, gloops)

        if info is None:
            info = {}
        info.setdefault("contractions", {})
        contractions = info["contractions"]

        region_counts = gen_region_counts(
            itertools.chain(gloops, ((tid,) for tid in self.tn.tensor_map)),
            autocomplete=autocomplete,
        )

        if progbar:
            import tqdm

            region_counts = tqdm.tqdm(region_counts)

        zvals = []
        for region, counting_factor in region_counts:
            try:
                zr = contractions[region]
            except KeyError:
                tnr = self.get_cluster_norm(region)
                zr = tnr.contract(optimize=optimize, **contract_opts)
                contractions[region] = zr
            zvals.append((zr, counting_factor))

        return combine_local_contractions(
            zvals,
            mantissa=self.sign**2,
            exponent=self.exponent * 2,
            backend=self.backend,
            strip_exponent=strip_exponent,
            check_zero=check_zero,
        )

    def compress(
        self,
        max_bond,
        cutoff=0.0,
        cutoff_mode="rsum2",
        renorm=0,
        reduce_opts=None,
        compress_opts=None,
        inplace=False,
        **kwargs,
    ):
        """Compress the initial tensor network using the current messages."""
        tn = self.tn if inplace else self.tn.copy()

        reduce_opts = ensure_dict(reduce_opts)
        compress_opts = kwargs | ensure_dict(compress_opts)
        compress_opts.setdefault("max_bond", max_bond)
        compress_opts.setdefault("cutoff", cutoff)
        compress_opts.setdefault("cutoff_mode", cutoff_mode)
        compress_opts.setdefault("renorm", renorm)

        for ix, tids in tn.ind_map.items():
            if len(tids) != 2:
                continue
            tida, tidb = tids

            # messages are left and right factors squared already
            ta = tn.tensor_map[tida]
            dim_bond = ta.ind_size(ix)
            dim_left = ta.size // dim_bond
            ml = self.messages[ix, tidb]
            Rl = qtn.decomp.squared_op_to_reduced_factor(
                ml, dim_left, dim_bond, right=True, **reduce_opts
            )

            tb = tn.tensor_map[tidb]
            dim_right = tb.size // dim_bond
            mr = self.messages[ix, tida].T
            Rr = qtn.decomp.squared_op_to_reduced_factor(
                mr, dim_bond, dim_right, right=False, **reduce_opts
            )

            # compute the compressors
            Pl, Pr = qtn.decomp.compute_oblique_projectors(
                Rl, Rr, **compress_opts
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

        if inplace:
            # tensor data has been modified
            self._initialize_contract_expressions()

        return tn

    def gauge_symmetric(self, inplace=False, **kwargs):
        """Gauge the tensor network symmetrically using the current messages.

        This applies the full-rank oblique projectors associated with each
        pair of messages, absorbing the effective singular values equally
        into both tensors.

        Parameters
        ----------
        inplace : bool, optional
            Whether to gauge the tensor network held by this BP instance.
        kwargs
            Additional options supplied when computing the oblique projectors.

        Returns
        -------
        TensorNetwork
        """
        kwargs.setdefault("max_bond", None)
        kwargs.setdefault("cutoff", 0.0)
        kwargs.setdefault("absorb", "both")
        return self.compress(inplace=inplace, **kwargs)

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
        yield outer
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

            self.messages[ix, tidb] = ma
            self.messages[ix, tida] = mb

        # mark the sites as touched
        self.update_touched_from_tids(tida, tidb)
        # rebuild the local contraction expressions
        self._init_tid(tida)
        self._init_tid(tidb)

    def get_cluster_norm(
        self,
        tids,
        partial_trace_map=(),
    ):
        """Get the local norm tensor network for ``tids`` with BP messages
        inserted on the boundary. Optionally open some physical indices up to
        perform an effective partial trace.

        Parameters
        ----------
        tids : iterable of hashable
            The tensor ids to include in the cluster.
        partial_trace_map : dict[str, str], optional
            A remapping of ket indices to bra indices to perform an effective
            partial trace.

        Returns
        -------
        TensorNetwork
        """
        k = self.tn._select_tids(tids, virtual=False)
        b = qtn.TensorNetwork(self.tensor_dual_map[tid] for tid in tids)

        if partial_trace_map:
            # open up the bra indices
            b.reindex_(partial_trace_map)

        tn_cluster = b | k

        for ix in k.outer_inds():
            if (ix not in partial_trace_map) and (ix not in self.output_inds):
                # dangling index -> attach message
                (tid,) = k.ind_map[ix]
                ixc = self.index_dual_map[ix]
                tm = qtn.Tensor(self.messages[ix, tid], inds=(ixc, ix))
                tn_cluster |= tm

        return tn_cluster

    def partial_trace(
        self,
        where,
        normalized=True,
        tids_region=None,
        get="matrix",
        bra_ind_id=None,
        optimize="auto-hq",
        **contract_opts,
    ):
        """Get the reduced density matrix for the sites specified by ``where``,
        with the remaining network approximated by messages on the boundary.

        Parameters
        ----------
        where : sequence[hashable]
            The sites to from the reduced density matrix of.
        get : {'tn', 'tensor', 'array', 'matrix'}, optional
            The type of object to return. If 'tn', return the uncontracted
            tensor network object. If 'tensor', return the labelled density
            operator as a `Tensor`. If 'array', return the unfused raw array
            with 2 * len(where) dimensions. If 'matrix', fuse the ket and bra
            indices and return this 2D matrix.
        bra_ind_id : str, optional
            If ``get="tn"``, how to label the bra indices. If None, use the
            default based on the current site_ind_id.
        optimize : str or PathOptimizer, optional
            The path optimizer to use when contracting the tensor network.
        contract_opts
            Other options supplied to ``TensorNetwork.contract``.

        Returns
        -------
        TensorNetwork or Tensor or array
        """
        # get a mapping of ket indices to bra indices on target sites
        if bra_ind_id is None:
            bra_ind_id = "b" + self.tn.site_ind_id[1:]
        bra_ind_starmap = bra_ind_id.count("{}") > 1
        kix = [self.tn.site_ind(coo) for coo in where]
        if bra_ind_starmap:
            bix = [bra_ind_id.format(*coo) for coo in where]
        else:
            bix = [bra_ind_id.format(coo) for coo in where]
        output_inds = (*kix, *bix)
        partial_trace_map = dict(zip(kix, bix))

        # get target region
        tags = [self.tn.site_tag(coo) for coo in where]

        if tids_region is None:
            tids_region = self.tn._get_tids_from_tags(tags, "any")
        tn = self.get_cluster_norm(
            tids_region, partial_trace_map=partial_trace_map
        )

        if get == "tn":
            return tn

        t = tn.contract(
            output_inds=output_inds, optimize=optimize, **contract_opts
        )

        if normalized:
            t /= t.trace(kix, bix)

        if get == "tensor":
            return t
        elif get == "array":
            return t.data
        elif get == "matrix":
            return t.to_dense(kix, bix)
        else:
            raise ValueError(f"Unknown get option: {get}")

    def partial_trace_gloop_expand(
        self,
        where,
        gloops=None,
        combine="sum",
        normalized=True,
        grow_from="alldangle",
        strict_size=False,
        optimize="auto-hq",
        **contract_opts,
    ):
        """Compute a reduced density matrix for the sites specified by
        ``where`` using the generalized loop cluster expansion.

        Parameters
        ----------
        where : sequence[hashable]
            The sites to from the reduced density matrix of.
        gloops : int or iterable of tuples, optional
            The generalized loops to use, or an integer to automatically
            generate all up to a certain size. If none use the smallest non-
            trivial size.
        combine : {'sum', 'prod'}, optional
            How to combine the contributions from each generalized loop. If
            'sum', use coefficient weighted addition. If 'prod', use power
            weighted multiplication.
        normalized : bool or {"local", "separate"}, optional
            Whether to normalize the density matrix. If True or "local",
            normalize each cluster density matrix by its trace. If "separate",
            normalize the final density matrix by its trace (usually less
            accurate). If False, do not normalize.
        grow_from : {'alldangle', 'all', 'any'}, optional
            How to grow the generalized loops from the specified ``where``:

            - 'alldangle': clusters up to max size, where target sites are
              allowed to dangle.
            - 'all': clusters where loop, up to max size, has to include *all*
              target sites.
            - 'any': clusters where loop, up to max size, can include *any* of
              the target sites. Remaining target sites are added as extras.

            By default 'alldangle'.
        strict_size : bool, optional
            Whether to enforce the maximum size of the generalized loops, only
            relevant for `grow_from="any"`.
        optimize : str or PathOptimizer, optional
            The path optimizer to use when contracting the tensor network.
        contract_opts
            Other options supplied to ``TensorNetwork.contract``.
        """
        tags = [self.tn.site_tag(coo) for coo in where]
        tids = self.tn._get_tids_from_tags(tags, "any")

        if normalized is True:
            normalized = "local"

        gloops = self.tn.get_local_gloops(
            tids=tids,
            gloops=gloops,
            grow_from=grow_from,
            strict_size=strict_size,
        )

        rhos = []
        for region, cr in gen_region_counts(gloops):
            rho_r = self.partial_trace(
                where,
                tids_region=region,
                normalized=False,
                get="matrix",
                optimize=optimize,
                **contract_opts,
            )

            if normalized == "local":
                rho_r /= ar.do("trace", rho_r)

            rhos.append((rho_r, cr))

        if combine == "sum":
            rho = functools.reduce(
                operator.add, (cr * rho_r for rho_r, cr in rhos)
            )
        elif combine == "prod":
            rho = functools.reduce(
                operator.mul, (rho_r**cr for rho_r, cr in rhos)
            )
        else:
            raise ValueError(f"Unknown combine option: {combine}")

        if (normalized == "separate") or (normalized and combine == "prod"):
            rho /= ar.do("trace", rho)

        if (not normalized) and ((self.sign, self.exponent) != (1.0, 0.0)):
            # have been accrued into by normalize_tensors most likely
            rho *= self.sign * 10**self.exponent

        return rho


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
        inplace=True,
    )


def gauge_d2bp(
    tn,
    *,
    messages=None,
    output_inds=None,
    power=1.0,
    smudge=0.0,
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
    reduce_opts=None,
    compress_opts=None,
    inplace=False,
    info=None,
    progbar=False,
    **contract_opts,
):
    """Gauge a tensor network using dense 2-norm belief propagation into the
    'symmetric' gauge. This is equivalent to simple update gauging where the
    singular values are absorbed equally into both tensors finally.

    Parameters
    ----------
    tn : TensorNetwork
        The tensor network to form the 2-norm of, run BP on, and then gauge.
    messages : dict[(str, int), array_like], optional
        The initial messages to use.
    output_inds : set[str], optional
        The indices to consider as output indices of the tensor network.
    power : float, optional
        The power used to condition the square-root message spectrum. Each
        message eigenvalue ``el`` is transformed to
        ``(sqrt(max(el, 0)) + smudge) ** (2 * power)``.
    smudge : float, optional
        The value added to the square-root message spectrum before applying
        ``power`` and reconstructing the squared message.
    max_iterations : int, optional
        The maximum number of BP iterations.
    tol : float, optional
        The convergence tolerance for messages.
    damping : float, optional
        The damping parameter to use.
    diis : bool or dict, optional
        Whether to use direct inversion in the iterative subspace.
    update : {'sequential', 'parallel'}, optional
        Whether to update messages sequentially or in parallel.
    normalize : str or callable, optional
        How to normalize messages after each update.
    distance : str or callable, optional
        How to compute the distance between messages.
    tol_abs : float, optional
        The absolute convergence tolerance.
    tol_rolling_diff : float, optional
        The rolling mean convergence tolerance.
    local_convergence : bool, optional
        Whether to allow messages to locally converge.
    optimize : str or PathOptimizer, optional
        The path optimizer to use when contracting messages.
    reduce_opts : dict, optional
        Options supplied when converting squared messages to reduced factors.
    compress_opts : dict, optional
        Options supplied when computing the symmetric oblique projectors.
    inplace : bool, optional
        Whether to gauge the input tensor network in place.
    info : dict, optional
        Store information about the BP run in this dictionary.
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
        power=power,
        smudge=smudge,
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
    return bp.gauge_symmetric(
        reduce_opts=reduce_opts,
        compress_opts=compress_opts,
        inplace=True,
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
