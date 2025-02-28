"""Belief propagation for standard tensor networks. This:

- assumes no hyper indices, only standard bonds.
- assumes a single ('dense') tensor per site
- works directly on the '1-norm' i.e. scalar tensor network

This is the simplest version of belief propagation, and is useful for
simple investigations.
"""

import autoray as ar

from quimb.tensor import Tensor, TensorNetwork, rand_uuid
from quimb.tensor.contraction import array_contract
from quimb.utils import oset

from .bp_common import (
    BeliefPropagationCommon,
    combine_local_contractions,
    normalize_message_pair,
)
from .hd1bp import (
    compute_all_tensor_messages_tree,
)


def initialize_messages(tn, fill_fn=None):
    messages = {}
    for ix, tids in tn.ind_map.items():
        if len(tids) != 2:
            continue
        tida, tidb = tids

        for tid_from, tid_to in [(tida, tidb), (tidb, tida)]:
            t_from = tn.tensor_map[tid_from]
            if fill_fn is not None:
                d = t_from.ind_size(ix)
                m = fill_fn((d,))
            else:
                m = array_contract(
                    arrays=(t_from.data,),
                    inputs=(tuple(range(t_from.ndim)),),
                    output=(t_from.inds.index(ix),),
                )
            messages[ix, tid_to] = m

    return messages


class D1BP(BeliefPropagationCommon):
    """Dense (as in one tensor per site) 1-norm (as in for 'classical' systems)
    belief propagation algorithm. Allows message reuse. This version assumes no
    hyper indices (i.e. a standard tensor network). This is the simplest
    version of belief propagation.

    Parameters
    ----------
    tn : TensorNetwork
        The tensor network to run BP on.
    messages : dict[(str, int), array_like], optional
        The initial messages to use, effectively defaults to all ones if not
        specified.
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
    contract_every : int, optional
        If not None, 'contract' (via BP) the tensor network every
        ``contract_every`` iterations. The resulting values are stored in
        ``zvals`` at corresponding points ``zval_its``.
    inplace : bool, optional
        Whether to perform any operations inplace on the input tensor network.

    Attributes
    ----------
    tn : TensorNetwork
        The target tensor network.
    messages : dict[(str, int), array_like]
        The current messages. The key is a tuple of the index and tensor id
        that the message is being sent to.
    key_pairs : dict[(str, int), (str, int)]
        A dictionary mapping the key of a message to the key of the message
        propagating in the opposite direction.
    """

    def __init__(
        self,
        tn: TensorNetwork,
        *,
        messages=None,
        damping=0.0,
        update="sequential",
        normalize=None,
        distance=None,
        local_convergence=True,
        message_init_function=None,
        contract_every=None,
        inplace=False,
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

        self.local_convergence = local_convergence

        if messages is None:
            self.messages = initialize_messages(self.tn, message_init_function)
        else:
            self.messages = messages

        # record which messages touch which tids, for efficient updates
        self.touched = oset()
        self.key_pairs = {}
        for ix, tids in tn.ind_map.items():
            if len(tids) != 2:
                continue
            tida, tidb = tids
            self.key_pairs[ix, tidb] = (ix, tida)
            self.key_pairs[ix, tida] = (ix, tidb)

    def iterate(self, tol=5e-6):
        if (not self.local_convergence) or (not self.touched):
            # assume if asked to iterate that we want to check all messages
            self.touched = oset(self.tn.tensor_map)

        ncheck = len(self.touched)
        nconv = 0
        max_mdiff = -1.0
        new_touched = oset()

        def _compute_ms(tid):
            t = self.tn.tensor_map[tid]
            new_ms = compute_all_tensor_messages_tree(
                t.data,
                [self.messages[ix, tid] for ix in t.inds],
                self.backend,
            )
            new_ms = [self._normalize_fn(m) for m in new_ms]
            new_ks = [self.key_pairs[ix, tid] for ix in t.inds]

            return new_ks, new_ms

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
                # mark distination tid for update
                new_touched.add(key[1])
            else:
                nconv += 1

            max_mdiff = max(max_mdiff, mdiff)
            self.messages[key] = new_m

        if self.update == "sequential":
            # compute each new message and immediately re-insert it
            while self.touched:
                tid = self.touched.pop()
                keys, new_ms = _compute_ms(tid)
                for key, new_m in zip(keys, new_ms):
                    _update_m(key, new_m)

        elif self.update == "parallel":
            new_data = {}
            # compute all new messages
            while self.touched:
                tid = self.touched.pop()
                keys, new_ms = _compute_ms(tid)
                for key, new_m in zip(keys, new_ms):
                    new_data[key] = new_m
            # insert all new messages
            for key, new_m in new_data.items():
                _update_m(key, new_m)

        self.touched = new_touched

        return {
            "nconv": nconv,
            "ncheck": ncheck,
            "max_mdiff": max_mdiff,
        }

    def normalize_message_pairs(self):
        """Normalize all messages such that for each bond `<m_i|m_j> = 1` and
        `<m_i|m_i> = <m_j|m_j>` (but in general != 1).
        """
        for ix, tids in self.tn.ind_map.items():
            if len(tids) != 2:
                continue
            tida, tidb = tids
            mi = self.messages[ix, tida]
            mj = self.messages[ix, tidb]
            mi, mj = normalize_message_pair(mi, mj)
            self.messages[ix, tida] = mi
            self.messages[ix, tidb] = mj

    def normalize_tensors(self, strip_exponent=True):
        """Normalize every local tensor contraction so that it equals 1. Gather
        the overall normalization factor into ``self.exponent`` and the sign
        into ``self.sign`` by default.

        Parameters
        ----------
        strip_exponent : bool, optional
            Whether to collect the sign and exponent. If ``False`` then the
            value of the BP contraction is set to 1.
        """
        for tid, t in self.tn.tensor_map.items():
            tval = self.local_tensor_contract(tid)
            tabs = ar.do("abs", tval)
            tsgn = tval / tabs
            tlog = ar.do("log10", tabs)
            t /= tsgn * tabs
            if strip_exponent:
                self.sign = tsgn * self.sign
                self.exponent = tlog + self.exponent

    def get_gauged_tn(self):
        """Gauge the original TN by inserting the BP-approximated transfer
        matrix eigenvectors, which may be complex. The BP-contraction of this
        gauged network is then simply the product of zeroth entries of each
        tensor.
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

    def get_cluster(self, tids):
        """Get the region of tensors given by `tids`, with the messages
        on the border contracted in, removing those dangling indices.

        Parameters
        ----------
        tids : sequence of int
            The tensor ids forming a region.

        Returns
        -------
        TensorNetwork
        """
        # take copy as we are going contract messages in
        tnr = self.tn._select_tids(tids, virtual=False)
        oixr = tnr.outer_inds()
        for ix in oixr:
            # get the tensor this index belongs to
            (tid,) = tnr._get_tids_from_inds(ix)
            t = tnr.tensor_map[tid]
            # contract the message in, removing index
            t.vector_reduce_(ix, self.messages[ix, tid])
        return tnr

    def local_tensor_contract(self, tid):
        """Contract the messages around tensor ``tid``."""
        t = self.tn.tensor_map[tid]
        arrays = [t.data]
        inputs = [tuple(range(t.ndim))]
        for i, ix in enumerate(t.inds):
            m = self.messages[ix, tid]
            arrays.append(m)
            inputs.append((i,))

        return array_contract(
            arrays=arrays,
            inputs=inputs,
            output=(),
        )

    def local_message_contract(self, ix):
        """Contract the messages at index ``ix``."""
        tids = self.tn.ind_map[ix]
        if len(tids) != 2:
            return None
        tida, tidb = tids
        return self.messages[ix, tida] @ self.messages[ix, tidb]

    def contract(
        self,
        strip_exponent=False,
        check_zero=True,
        **kwargs,
    ):
        """Estimate the contraction of the tensor network."""

        zvals = [
            (self.local_tensor_contract(tid), 1) for tid in self.tn.tensor_map
        ] + [(self.local_message_contract(ix), -1) for ix in self.tn.ind_map]

        return combine_local_contractions(
            zvals,
            backend=self.backend,
            strip_exponent=strip_exponent,
            check_zero=check_zero,
            mantissa=self.sign,
            exponent=self.exponent,
            **kwargs,
        )

    def contract_with_loops(
        self,
        max_loop_length=None,
        min_loop_length=1,
        optimize="auto-hq",
        strip_exponent=False,
        check_zero=True,
        **contract_opts,
    ):
        """Estimate the contraction of the tensor network, including loop
        corrections.
        """
        self.normalize_message_pairs()
        self.normalize_tensors()

        zvals = []

        for loop in self.tn.gen_paths_loops(max_loop_length=max_loop_length):
            if len(loop) < min_loop_length:
                continue

            # get the loop local patch
            ltn = self.tn.select_path(loop)

            # attach boundary messages

            for ix, tids in tuple(ltn.ind_map.items()):
                if ix in loop:
                    continue

                elif len(tids) == 1:
                    # outer index -> cap it with messages
                    (tid,) = tids
                    ltn |= Tensor(self.messages[ix, tid], [ix])

                else:
                    # non-loop internal index -> cut it with messages
                    tida, tidb = tids
                    ma = self.messages[ix, tida]
                    mb = self.messages[ix, tidb]
                    lix = rand_uuid()
                    rix = rand_uuid()
                    ltn._cut_between_tids(tida, tidb, lix, rix)
                    ltn |= Tensor(ma, [lix])
                    ltn |= Tensor(mb, [rix])

            zvals.append((ltn.contract(optimize=optimize, **contract_opts), 1))

        return combine_local_contractions(
            zvals,
            backend=self.backend,
            strip_exponent=strip_exponent,
            check_zero=check_zero,
            mantissa=self.sign,
            exponent=self.exponent,
        )

    def contract_gloop_expand(
        self,
        gloops=None,
        autocomplete=True,
        strip_exponent=False,
        check_zero=True,
        optimize="auto-hq",
        **contract_opts,
    ):
        from .regions import RegionGraph

        if isinstance(gloops, int):
            max_size = gloops
            gloops = None
        else:
            max_size = None

        if gloops is None:
            gloops = tuple(self.tn.gen_gloops(max_size=max_size))
        else:
            gloops = tuple(gloops)

        rg = RegionGraph(gloops, autocomplete=autocomplete)

        zvals = []
        for r in rg.regions:
            c = rg.get_count(r)
            tnr = self.get_cluster(r)
            zr = tnr.contract(optimize=optimize, **contract_opts)

            zvals.append((zr, c))

        return combine_local_contractions(
            zvals,
            backend=self.backend,
            strip_exponent=strip_exponent,
            check_zero=check_zero,
            mantissa=self.sign,
            exponent=self.exponent,
        )


def contract_d1bp(
    tn,
    *,
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
    strip_exponent=False,
    check_zero=True,
    info=None,
    progbar=False,
    **contract_opts,
):
    """Estimate the contraction of standard tensor network ``tn`` using dense
    1-norm belief propagation.

    Parameters
    ----------
    tn : TensorNetwork
        The tensor network to contract, it should have no dangling or hyper
        indices.
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
    """
    bp = D1BP(
        tn,
        damping=damping,
        local_convergence=local_convergence,
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
