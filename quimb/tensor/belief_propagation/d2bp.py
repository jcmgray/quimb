"""Dense 2-norm belief propagation for standard PEPS like tensor networks, with
one tensor per site and no hyper indices. This is the basic 'quantum' BP.

A fermionic message has two representations, differing by a single
``phase_flip(0)``, which negates the odd-parity sectors of the bra axis and is
its own inverse. The flip is intrinsic to the doubled contraction.

- ``C`` is the raw contraction of the bra with the ket over everything except
  one bond, leaving that bond's bra and ket legs open, in ``(bra, ket)`` order.
  A BP update expression returns ``C``.
- ``M = C.phase_flip(0)`` is the stored message, and the representation in
  which symmray's fermionic ``eigh()`` reports a nonnegative spectrum.

Storing ``M`` lets conditioning, gauge insertion and hermitization treat a
message as an operator. To use it as an environment, insert ``M`` directly
between the bra and ket, using the usual fermionic contraction rules. Message
updates and direct environment contractions do not require square-root
factors. Gauge insertion still computes them. autoray's default ``gram``
composes conjugation and contraction, and symmray overrides it to build ``M``
directly and cancel paired conjugate dummy modes.

Convert between the representations with ``phase_flip(0)``:

- after a BP update, convert the result ``C`` to ``M`` before storing it
- before calling ``squared_op_to_reduced_factor``, convert ``M`` to ``C``.
  That routine also applies a separate flip based on the bond orientation
- when contracting the two messages on a bond to a scalar, convert only one
  of them to ``C``

TODO: cache gauges computed from messages until out of date.
"""

import contextlib
import functools
import itertools
import operator

import autoray as ar

import quimb.tensor as qtn
from quimb.tensor.array_ops import isfermionic
from quimb.tensor.networking import NetworkPatch
from quimb.tensor.tnag.core import (
    contract_reduced_density_matrix,
    get_bra_inds,
)
from quimb.utils import check_opt, ensure_dict, oset

from .bp_common import (
    BeliefPropagationCommon,
    combine_local_contractions,
    normalize_message_pair,
    parse_gloops_edge_induced,
    process_loop_series_expansion_weights,
)
from .regions import gen_region_counts


def _parse_global_gloops(tn, gloops=None):
    if isinstance(gloops, (int, str)):
        max_size = gloops
        gloops = None
    else:
        max_size = None

    if gloops is None:
        gloops = tuple(tn.gen_gloops(max_size=max_size))
    else:
        gloops = tuple(gloops)

    return gloops


def _message_to_reduced_factor(m, *args, **kwargs):
    if isfermionic(m):
        # reduced-factor extraction expects C rather than the stored M
        m = m.phase_flip(0)
    return qtn.decomp.squared_op_to_reduced_factor(m, *args, **kwargs)


@functools.lru_cache(maxsize=128)
def _get_message_conditioner(power=1.0, smudge=0.0, backend=None):
    """Get a function that conditions squared BP messages spectrally. Return
    ``None`` if the function would do nothing."""

    if power == 1.0:
        if smudge == 0.0:
            # trivial conditioning
            return None

        else:
            if backend is None:
                _eigh = ar.DoFunc("linalg.eigh")
                _clip = ar.DoFunc("clip")
                _sqrt = ar.DoFunc("sqrt")
                _max = ar.DoFunc("max")
            else:
                _eigh = ar.get_lib_fn(backend, "linalg.eigh")
                _clip = ar.get_lib_fn(backend, "clip")
                _sqrt = ar.get_lib_fn(backend, "sqrt")
                _max = ar.get_lib_fn(backend, "max")

            def conditioner(m):
                el, ev = _eigh(m)
                el = _clip(el, 0.0, None)
                el = _sqrt(el)
                el = el + smudge * _max(el)
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
                _max = ar.DoFunc("max")
            else:
                _eigh = ar.get_lib_fn(backend, "linalg.eigh")
                _clip = ar.get_lib_fn(backend, "clip")
                _sqrt = ar.get_lib_fn(backend, "sqrt")
                _max = ar.get_lib_fn(backend, "max")

            def conditioner(m):
                el, ev = _eigh(m)
                el = _clip(el, 0.0, None)
                el = _sqrt(el)
                el = el + smudge * _max(el)
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
    diis : bool or dict, optional
        Whether to use direct inversion in the iterative subspace to help
        converge the messages by extrapolating to low error guesses. If a
        dict, should contain options for the DIIS algorithm. The relevant
        options are {`max_history`, `beta`, `rcond`}.
    update : {'sequential', 'parallel'}, optional
        Whether to update messages sequentially (newly computed messages are
        immediately used for other updates in the same iteration round) or in
        parallel (all messages are comptued using messages from the previous
        round only). Sequential generally helps convergence but parallel can
        possibly converge to differnt solutions.
    power : float, optional
        Condition each message with this power when D2BP inserts it as an
        environment. D2BP transforms each square-root message eigenvalue
        ``s = sqrt(max(el, 0))`` to ``(s + smudge * max(s)) ** (2 * power)``.
        D2BP stores the raw messages, and keeps each conditioned copy after
        it computes it.
    smudge : float, optional
        Add this value to the square-root message spectrum before D2BP
        applies ``power`` and squares the spectrum. It is relative to the
        largest square-root message eigenvalue.
    normalize : {'L1', 'L2', 'L2phased', 'Linf', callable}, optional
        How to normalize messages after each update. If None choose
        automatically. If a callable, it should take a message and return the
        normalized message. If a string, it should be one of 'L1', 'L2',
        'L2phased', 'Linf' for the corresponding norms. 'L2phased' is like 'L2'
        but also normalizes the phase of the message, by default used for
        complex dtypes.
        Fermionic messages default to Frobenius norm normalization without
        changing their global phase.
    distance : {'L1', 'L2', 'L2phased', 'Linf', 'cosine', callable}, optional
        How to compute the distance between messages to check for convergence.
        If None choose automatically. If a callable, it should take two
        messages and return the distance. If a string, it should be one of
        'L1', 'L2', 'L2phased', 'Linf', or 'cosine' for the corresponding
        norms. 'L2phased' is like 'L2' but also normalizes the phases of the
        messages, by default used for complex dtypes if phased normalization is
        not already being used.
        Fermionic messages default to the Frobenius norm of their difference.
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
        diis=False,
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
        self._fermionic = tn.isfermionic()
        if self._fermionic:
            # phase normalization can negate positive fermionic operators
            if normalize is None:
                normalize = "L2"
            if distance is None:
                distance = "L2"
        super().__init__(
            tn=tn,
            damping=damping,
            diis=diis,
            update=update,
            normalize=normalize,
            distance=distance,
            contract_every=contract_every,
            inplace=inplace,
        )
        self.contract_opts = contract_opts
        self.contract_opts.setdefault("optimize", optimize)
        self.local_convergence = local_convergence
        self._messages_conditioned = {}
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
        # the conditioned copies are out of date
        self._messages_conditioned.clear()
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
        """The relative value added to the square-root message spectrum.

        Setting this marks all message expressions for recomputation.
        """
        return self._smudge

    @smudge.setter
    def smudge(self, smudge):
        if smudge != self._smudge:
            self._smudge = smudge
            self._update_message_conditioner()

    @property
    def messages(self):
        """The raw messages, keyed by ``(ix, tid)``: the message *to* tensor
        ``tid`` along index ``ix``. If you change these directly, note that
        D2BP keeps the conditioned copies separately. See
        ``_get_message_conditioned`` and ``_messages_conditioned``.

        Fermionic messages are positive fermionic operators in ``(bra, ket)``
        order, with conjugate dummy modes removed. They connect the bra and
        ket copies and can be contracted into either side first.
        Their spectra from ``eigh()`` are positive, although their dense block
        matrices can have negative odd-sector eigenvalues.
        """
        return self._messages

    @messages.setter
    def messages(self, messages):
        self._messages = messages
        self._messages_conditioned.clear()

    def _get_message_conditioned(self, key):
        """Get the conditioned version of message ``key``, for insertion as
        an environment. D2BP computes the copy only if it does not have one,
        then keeps it.
        """
        if self._message_conditioner is None:
            return self._messages[key]
        try:
            mc = self._messages_conditioned[key]
        except KeyError:
            mc = self._message_conditioner(self._messages[key])
            self._messages_conditioned[key] = mc
        return mc

    def get_message(self, key, power=None, smudge=None):
        """Get a message conditioned for insertion as an environment.

        Parameters
        ----------
        key : tuple[str, int]
            The directed message key ``(ix, tid)``.
        power : float, optional
            Condition the square-root message spectrum with this power. If
            you do not supply it, use this instance's iteration power.
        smudge : float, optional
            Add this value to the square-root message spectrum. It is
            relative to the largest square-root message eigenvalue. If you do
            not supply it, use this instance's iteration smudge.

        Returns
        -------
        array_like
        """
        if power is None:
            power = self.power
        if smudge is None:
            smudge = self.smudge

        if (power == self.power) and (smudge == self.smudge):
            return self._get_message_conditioned(key)

        m = self.messages[key]
        conditioner = _get_message_conditioner(
            power,
            smudge,
            self.backend,
        )
        if conditioner is None:
            return m
        return conditioner(m)

    def _init_tid(self, tid):
        """Setup any missing input messages and build contraction expressions
        for output message updates for each bond around tensor at `tid`.
        """
        from quimb.tensor.contraction import array_contract_expression

        t = self.tn.tensor_map[tid]
        ix_neighbors = {}

        # first build initial messages from ix->tid
        for ix in t.inds:
            if ix in self.output_inds:
                # output index -> directly contract without message
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
                kt = self.tn.tensor_map[tidn]
                m = ar.do("gram", kt.data, axes=kt.inds.index(ix))
                m = self._normalize_fn(m)
                self.messages[ix, tid] = m

            # make sure touch_map entry exists
            self.touch_map.setdefault((ix, tid), {})

        # include the norm-conjugation phases on every leg of this local bra
        t_dag = t.conj(output_inds=t.inds)
        t_dag.reindex_(self.index_dual_map)

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
            # insert the conditioned (e.g. powered) input messages
            ms = [self._get_message_conditioned(mkey) for mkey in mkeys]

            # contract update!
            m = expr(x, xc, *ms)
            if self._fermionic:
                # the expression returns C, while messages store M
                m = m.phase_flip(0)

            # for stability enforce hermiticity
            m = m + ar.dag(m)

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
            # the conditioned copy is out of date
            self._messages_conditioned.pop(key, None)

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
            self._messages_conditioned.pop((ix, tida), None)
            self._messages_conditioned.pop((ix, tidb), None)

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
            # refresh both the dual tensor and captured expression inputs
            self._init_tid(tid)
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
            if self._fermionic:
                # the scalar overlap takes C against the opposing stored M
                ml = ml.phase_flip(0)
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
        """Build the local loop excitation norm tensor network for ``tids``.

        Insert BP messages on boundary bonds and excitation projectors on
        all or selected inner bonds. See https://arxiv.org/abs/2409.03108.

        Parameters
        ----------
        tids : iterable of hashable or NetworkPatch
            The region ``tids``. A
            :class:`~quimb.tensor.networking.NetworkPatch` also selects the
            subset of inner excited bonds. Non-excited inner bonds are treated
            like boundary bonds. See
            :func:`~quimb.tensor.networking.gen_gloops_edge_induced`.
        partial_trace_map : dict[str, str], optional
            Map ket indices to bra indices for an effective partial trace.
        exclude : iterable of str, optional
            Bond indices that do not receive excitation projectors, e.g. when
            forming a reduced density matrix.

        Returns
        -------
        TensorNetwork
        """
        if isinstance(tids, NetworkPatch):
            excited = tids.inds
            tids = tids.tids
        else:
            # excite every inner bond
            excited = None

        stn = self.tn._select_tids(tids)

        kixmaps = {tid: {} for tid in stn.tensor_map}
        bixmaps = {tid: {} for tid in stn.tensor_map}
        exc_ixs = {}
        bms = []
        ems = []

        for ix, ix_tids in stn.ind_map.items():
            if ix in self.output_inds:
                # physical traced index
                if ix in partial_trace_map:
                    (tid,) = ix_tids
                    bixmaps[tid][ix] = partial_trace_map[ix]

            elif ix in exclude:
                # excluded bond -> simply rename bra
                bix = qtn.rand_uuid()
                for tid in ix_tids:
                    bixmaps[tid][ix] = bix

            elif ix in stn._inner_inds:
                # internal index
                for tid in ix_tids:
                    kix = qtn.rand_uuid()
                    bix = qtn.rand_uuid()
                    kixmaps[tid][ix] = kix
                    bixmaps[tid][ix] = bix
                    # store labels for excitation projector, (bra, ket)
                    exc_ixs.setdefault(ix, {})[tid] = (bix, kix)

                if (excited is None) or (ix in excited):
                    ems.append((ix, ix_tids))
                else:
                    # messages cut this bond like a boundary bond
                    for tid in ix_tids:
                        bms.append((ix, tid))

            else:
                # boundary index
                (tid,) = ix_tids
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

        # add inner excitation projectors
        with ar.backend_like(self.backend):
            for ix, ix_tids in ems:
                tidl, tidr = ix_tids
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
        gloops : None, int, "min" or iterable, optional
            Loops or regions to use. An integer generates all loops up to that
            size. ``None`` and ``"min"`` use an automatic size. An iterable
            can contain regions of ``tids`` or ``NetworkPatch`` objects.
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

        gloops = parse_gloops_edge_induced(self.tn, gloops)

        weights = {}
        for gloop in gloops:
            etn = self.get_cluster_excited(gloop)
            w = etn.contract(optimize=optimize, **contract_opts)
            # loops with the same tensor support use one suppression factor
            key = tuple(sorted(gloop.tids))
            weights[key] = weights.get(key, 0.0) + w

        return process_loop_series_expansion_weights(
            weights,
            num_tensors=self.tn.num_tensors,
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
        grow_from="all",
        strict_size=False,
        multi_excitation_correct=True,
        optimize="auto-hq",
        allow_dangling=True,
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
        gloops : None, int, "min" or iterable of tuples, optional
            The generalized loops to use, an integer to generate all loops up
            to that size, or ``None``/``"min"`` for the automatic size, see
            :func:`~quimb.tensor.networking.gen_gloops`.
        normalized : bool, optional
            Whether to normalize the final density matrix.
        grow_from : {'all', 'any'}, optional
            How to grow the generalized loops from the specified ``where``:

            - 'all': the loop, up to max size, must include *all* target sites.
            - 'any': the loop, up to max size, can include *any* of the target
              sites. The other target sites are added as extras.

        allow_dangling : bool, optional
            Whether target sites can have fewer than two internal bonds in
            yielded regions.
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
        bix = get_bra_inds(self.tn, where)
        partial_trace_map = dict(zip(kix, bix))
        output_inds = (*kix, *bix)

        # generate the generalized loops relevant to the target sites
        gloops = self.tn.get_local_gloops(
            tids=tids,
            gloops=gloops,
            grow_from=grow_from,
            strict_size=strict_size,
            allow_dangling=allow_dangling,
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
                weights,
                num_tensors=self.tn.num_tensors,
                return_all=True,
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
        power=1.0,
        smudge=0.0,
        reduce_opts=None,
        compress_opts=None,
        inplace=False,
        **kwargs,
    ):
        """Compress the tensor network using the current messages.

        Parameters
        ----------
        max_bond : int
            The maximum bond dimension to compress to.
        cutoff : float, optional
            A dynamic singular value cutoff to use when compressing.
        cutoff_mode : str, optional
            The mode for cutoff compression.
        renorm : float, optional
            Whether to renormalize the singular values when compressing.
        power : float, optional
            Condition the message spectra with this power before you compute
            the reduced factors and projectors. This tempers the environment
            weight. It is independent of the iteration ``power``.
        smudge : float, optional
            Add this value to the square-root message spectra before you
            apply ``power``. It is relative to the largest square-root
            eigenvalue of each message.
        reduce_opts : dict, optional
            Supplied to :func:`~quimb.tensor.tensor_core.tensor_split` for the
            reduction step. Values set here take precedence over any defaults.
        compress_opts : dict, optional
            Supplied to :func:`~quimb.tensor.tensor_core.tensor_split`. Values set
            here take precedence over any defaults.
        inplace : bool, optional
            Whether to compress in place or return a new tensor network.
        kwargs
            Extra keyword arguments are combined into `compress_opts`, though
            existing items in `compress_opts` take precedence over `kwargs`.

        Returns
        -------
        TensorNetwork
            The compressed tensor network.
        """
        tn = self.tn if inplace else self.tn.copy()

        reduce_opts = ensure_dict(reduce_opts)
        compress_opts = kwargs | ensure_dict(compress_opts)
        compress_opts.setdefault("max_bond", max_bond)
        compress_opts.setdefault("cutoff", cutoff)
        compress_opts.setdefault("cutoff_mode", cutoff_mode)
        compress_opts.setdefault("renorm", renorm)

        conditioner = _get_message_conditioner(
            power, smudge, backend=self.backend
        )

        for ix, tids in tn.ind_map.items():
            if len(tids) != 2:
                continue
            tida, tidb = tids

            # messages are left and right factors squared already
            ta = tn.tensor_map[tida]
            dim_bond = ta.ind_size(ix)
            dim_left = ta.size // dim_bond
            ml_raw = self.messages[ix, tidb]
            ml = ml_raw if conditioner is None else conditioner(ml_raw)
            Ra = _message_to_reduced_factor(
                ml, dim_left, dim_bond, right=True, **reduce_opts
            )

            tb = tn.tensor_map[tidb]
            dim_right = tb.size // dim_bond
            mr_raw = self.messages[ix, tida]
            mr_raw = ar.do("transpose", mr_raw)
            mr = mr_raw if conditioner is None else conditioner(mr_raw)
            Rb = _message_to_reduced_factor(
                mr, dim_bond, dim_right, right=False, **reduce_opts
            )

            # compute the compressors
            Pa, Pb = qtn.decomp.compute_oblique_projectors(
                Ra, Rb, **compress_opts
            )

            # contract the compressors into the tensors
            tn.tensor_map[tida].gate_(Pa.T, ix)
            tn.tensor_map[tidb].gate_(Pb, ix)

            # update messages with projections
            if inplace:
                if conditioner is not None:
                    # messages are stored raw, so project the raw factors
                    Ra = _message_to_reduced_factor(
                        ml_raw, dim_left, dim_bond, right=True, **reduce_opts
                    )
                    Rb = _message_to_reduced_factor(
                        mr_raw, dim_bond, dim_right, right=False, **reduce_opts
                    )
                new_Ra = Ra @ Pa
                self.messages[ix, tidb] = ar.do("gram", new_Ra, axes=1)

                new_Rb = Pb @ Rb
                self.messages[ix, tida] = ar.do("gram", new_Rb, axes=0)

                self._messages_conditioned.pop((ix, tidb), None)
                self._messages_conditioned.pop((ix, tida), None)

        if inplace:
            # tensor data has been modified
            self._initialize_contract_expressions()

        return tn

    def gauge_symmetric(self, power=1.0, smudge=0.0, inplace=False, **kwargs):
        """Gauge the tensor network symmetrically using the current messages.

        This applies the full-rank oblique projectors associated with each
        pair of messages, absorbing the effective singular values equally
        into both tensors.

        Parameters
        ----------
        power : float, optional
            Condition the message spectra with this power before you compute
            the projectors. This tempers the environment weight. It is
            independent of the iteration ``power``.
        smudge : float, optional
            Add this value to the square-root message spectra before you
            apply ``power``. It is relative to the largest square-root
            eigenvalue of each message.
        inplace : bool, optional
            Whether to gauge the tensor network held by this BP instance.
        kwargs
            Additional options supplied to
            :meth:`~quimb.tensor.belief_propagation.d2bp.D2BP.compress`.

        Returns
        -------
        TensorNetwork
        """
        kwargs.setdefault("max_bond", None)
        kwargs.setdefault("cutoff", 0.0)
        kwargs.setdefault("absorb", "both")
        return self.compress(
            power=power, smudge=smudge, inplace=inplace, **kwargs
        )

    def gauge_insert(
        self,
        tn,
        power=1.0,
        smudge=1e-12,
        return_gauges="inverse",
    ):
        """Insert the sqrt of messages on the boundary of a part of the main BP
        TN.

        Parameters
        ----------
        tn : TensorNetwork
            The tensor network to insert the messages into.
        power : float, optional
            Condition the smudged square-root message spectrum with this
            power.
        smudge : float, optional
            Add this value to the square-root message spectrum before this
            method applies ``power``. It is relative to the largest
            square-root message eigenvalue.
        return_gauges : {"raw", "inverse", None}, optional
            Whether to return the message factors applied, their inverses, or
            nothing. The default is ``"inverse"``.

        Returns
        -------
        list[tuple[Tensor, str, array_like]] or None
            If requested, the sequence of tensors, indices, and raw or inverse
            message factors.
        """
        check_opt("return_gauges", return_gauges, ("raw", "inverse", None))

        _eigh = ar.get_lib_fn(self.backend, "linalg.eigh")
        _clip = ar.get_lib_fn(self.backend, "clip")
        _sqrt = ar.get_lib_fn(self.backend, "sqrt")
        _max = ar.get_lib_fn(self.backend, "max")

        outer = [] if return_gauges is not None else None

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
            s = _sqrt(_clip(s2, 0.0, None))
            if smudge != 0.0:
                s = s + smudge * _max(s)
            if power != 1.0:
                s = s**power
            msqrt = qtn.decomp.ldmul(s, ar.dag(W))
            t.gate_(msqrt, ix)
            if return_gauges == "raw":
                outer.append((t, ix, msqrt))
            elif return_gauges == "inverse":
                msqrt_inv = qtn.decomp.rddiv(W, s)
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
        the current messages to gauge the tensors. A distinct tensor network
        is not supported because the messages and contraction expressions are
        tied to this instance's managed network.
        """
        if tn is None:
            tn = self.tn
        elif tn is not self.tn:
            raise ValueError(
                "D2BP.gate_ can only update its managed tensor network."
            )

        if len(where) == 1:
            # single site gate
            tn.gate_(G, where, contract=True)
            tids = tn._get_tids_from_tags(tn.site_tag(where[0]))
            self.update_touched_from_tids(*tids)
            for tid in tids:
                self._init_tid(tid)
            return

        site_tags = tuple(map(tn.site_tag, where))
        tn_where = tn.select_any(site_tags)
        gate_opts.setdefault("contract", "reduce-split")

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
            _, (ix,), _ = qtn.group_inds(ta, tb)

            # make use of the fact that we already have gauged tensors
            ma = ar.do("gram", ta.data, axes=ta.inds.index(ix))
            mb = ar.do("gram", tb.data, axes=tb.inds.index(ix))

            self.messages[ix, tidb] = ma
            self.messages[ix, tida] = mb
            self._messages_conditioned.pop((ix, tidb), None)
            self._messages_conditioned.pop((ix, tida), None)

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

        # conjugate the cluster jointly, phasing only its outer legs
        b = k.conj().reindex(self.index_dual_map)

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
            The sites to keep in the reduced density matrix.
        normalized : bool or "return", optional
            Whether to normalize the result to unit trace. If "return", return
            the trace separately, without dividing by it.
            Ignored if ``get="tn"``, which returns the unnormalized network
            without a separate trace.
        tids_region : sequence[int], optional
            Tensor tids to contract exactly, with messages on the boundary. By
            default use only the tensors at ``where``.
        get : {'matrix', 'array', 'tensor', 'tn'}, optional
            How to return the reduced density matrix:

            - 'matrix': a dense matrix, with the ket sites fused into rows
              and the bra sites fused into columns.
            - 'array': the raw array, with one axis per ket site then one
              axis per bra site.
            - 'tensor': a :class:`~quimb.tensor.tensor_core.Tensor` with the
              ket and bra indices.
            - 'tn': the uncontracted tensor network.

        bra_ind_id : str, optional
            Format string for bra indices. By default replace a leading ``k``
            with ``b``, or prepend ``b``. See
            :func:`~quimb.tensor.tnag.core.get_bra_inds`.
        optimize : str or PathOptimizer, optional
            The path optimizer to use when contracting the tensor network.
        contract_opts
            Other options supplied to ``TensorNetwork.contract``.

        Returns
        -------
        array or Tensor or TensorNetwork or (array, float) or (Tensor, float)
        """
        # get a mapping of ket indices to bra indices on target sites
        kix = [self.tn.site_ind(coo) for coo in where]
        if bra_ind_id is None:
            bix = get_bra_inds(self.tn, where, warn=get in ("tn", "tensor"))
        elif bra_ind_id.count("{}") > 1:
            bix = [bra_ind_id.format(*coo) for coo in where]
        else:
            bix = [bra_ind_id.format(coo) for coo in where]
        partial_trace_map = dict(zip(kix, bix))

        # get target region
        tags = [self.tn.site_tag(coo) for coo in where]

        if tids_region is None:
            tids_region = self.tn._get_tids_from_tags(tags, "any")
        tn = self.get_cluster_norm(
            tids_region, partial_trace_map=partial_trace_map
        )

        return contract_reduced_density_matrix(
            tn,
            kix,
            bix,
            normalized=normalized,
            get=get,
            optimize=optimize,
            **contract_opts,
        )

    def partial_trace_gloop_expand(
        self,
        where,
        gloops=None,
        combine="sum",
        normalized=True,
        grow_from="all",
        strict_size=False,
        optimize="auto-hq",
        allow_dangling=True,
        **contract_opts,
    ):
        """Compute a reduced density matrix for the sites specified by
        ``where`` using the generalized loop cluster expansion.

        Parameters
        ----------
        where : sequence[hashable]
            The sites to from the reduced density matrix of.
        gloops : None, int, "min" or iterable of tuples, optional
            The generalized loops to use, an integer to generate all loops up
            to that size, or ``None``/``"min"`` for the automatic size, see
            :func:`~quimb.tensor.networking.gen_gloops`.
        combine : {'sum', 'prod'}, optional
            How to combine the contributions from each generalized loop. If
            'sum', use coefficient weighted addition. If 'prod', use power
            weighted multiplication.
        normalized : bool or {"local", "separate"}, optional
            Whether to normalize the density matrix. If True or "local",
            normalize each cluster density matrix by its trace. If "separate",
            normalize the final density matrix by its trace (usually less
            accurate). If False, do not normalize.
        grow_from : {'all', 'any'}, optional
            How to grow the generalized loops from the specified ``where``:

            - 'all': the loop, up to max size, must include *all* target sites.
            - 'any': the loop, up to max size, can include *any* of the target
              sites. The other target sites are added as extras.

        allow_dangling : bool, optional
            Whether target sites can have fewer than two internal bonds in
            yielded regions.
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
            allow_dangling=allow_dangling,
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


def converge_d2bp(
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
    contract_every=None,
    optimize="auto-hq",
    inplace=False,
    info=None,
    progbar=False,
    **contract_opts,
):
    """Construct and run dense 2-norm belief propagation, returning the
    resulting instance.

    Parameters
    ----------
    tn : TensorNetwork
        The tensor network to form the 2-norm of and run BP on.
    messages : dict[(str, int), array_like], optional
        The initial messages to use.
    output_inds : set[str], optional
        The indices to consider as output indices of the tensor network.
    power : float, optional
        Condition each message with this power when D2BP inserts it into an
        update contraction.
    smudge : float, optional
        Add this value to the square-root message spectrum before D2BP
        applies ``power``. It is relative to the largest square-root message
        eigenvalue.
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
    contract_every : int, optional
        Compute and store the BP contraction every this many iterations.
    optimize : str or PathOptimizer, optional
        The path optimizer to use when contracting messages.
    inplace : bool, optional
        Whether the BP instance should use the input tensor network directly.
    info : dict, optional
        Store information about the BP run in this dictionary.
    progbar : bool, optional
        Whether to show a progress bar.
    contract_opts
        Other options supplied to ``cotengra.array_contract``.

    Returns
    -------
    D2BP
        The belief propagation instance after running to convergence or the
        iteration limit.
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
        contract_every=contract_every,
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
    return bp


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
    bp = converge_d2bp(
        tn,
        messages=messages,
        output_inds=output_inds,
        max_iterations=max_iterations,
        tol=tol,
        diis=diis,
        tol_abs=tol_abs,
        tol_rolling_diff=tol_rolling_diff,
        optimize=optimize,
        local_convergence=local_convergence,
        damping=damping,
        update=update,
        normalize=normalize,
        distance=distance,
        info=info,
        progbar=progbar,
        **contract_opts,
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
    power=1.0,
    smudge=0.0,
    gauge_power=1.0,
    gauge_smudge=0.0,
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
        A dynamic singular value cutoff to use when compressing.
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
    power : float, optional
        Condition each message with this power when D2BP inserts it into an
        update contraction. D2BP transforms each square-root message
        eigenvalue ``s = sqrt(max(el, 0))`` to
        ``(s + smudge * max(s)) ** (2 * power)``. D2BP stores the raw
        messages.
    smudge : float, optional
        Add this value to the square-root message spectrum before D2BP
        applies ``power`` and squares the spectrum. It is relative to the
        largest square-root message eigenvalue.
    gauge_power : float, optional
        Condition the converged message spectra with this power when
        computing the compression projectors. This tempers the environment
        weight. It is independent of ``power``.
    gauge_smudge : float, optional
        Add this value to the converged square-root message spectra before
        applying ``gauge_power``. It is independent of ``smudge``.
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
    bp = converge_d2bp(
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
        max_iterations=max_iterations,
        tol=tol,
        diis=diis,
        tol_abs=tol_abs,
        tol_rolling_diff=tol_rolling_diff,
        info=info,
        progbar=progbar,
        **contract_opts,
    )
    return bp.compress(
        max_bond=max_bond,
        cutoff=cutoff,
        cutoff_mode=cutoff_mode,
        renorm=renorm,
        power=gauge_power,
        smudge=gauge_smudge,
        inplace=True,
    )


def gauge_d2bp(
    tn,
    *,
    messages=None,
    output_inds=None,
    power=1.0,
    smudge=0.0,
    gauge_power=1.0,
    gauge_smudge=0.0,
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
        Condition each message with this power when D2BP inserts it into an
        update contraction. D2BP transforms each square-root message
        eigenvalue ``s = sqrt(max(el, 0))`` to
        ``(s + smudge * max(s)) ** (2 * power)``. D2BP stores the raw
        messages.
    smudge : float, optional
        Add this value to the square-root message spectrum before D2BP
        applies ``power`` and squares the spectrum. It is relative to the
        largest square-root message eigenvalue.
    gauge_power : float, optional
        Condition the converged message spectra with this power when
        computing the final gauging projectors. This tempers the environment
        weight. It is independent of ``power``.
    gauge_smudge : float, optional
        Add this value to the converged square-root message spectra before
        applying ``gauge_power``. It is independent of ``smudge``.
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
    bp = converge_d2bp(
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
        max_iterations=max_iterations,
        tol=tol,
        diis=diis,
        tol_abs=tol_abs,
        tol_rolling_diff=tol_rolling_diff,
        info=info,
        progbar=progbar,
        **contract_opts,
    )
    return bp.gauge_symmetric(
        power=gauge_power,
        smudge=gauge_smudge,
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
