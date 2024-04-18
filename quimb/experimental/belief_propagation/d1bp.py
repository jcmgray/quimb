"""Belief propagation for standard tensor networks. This:

- assumes no hyper indices, only standard bonds.
- assumes a single ('dense') tensor per site
- works directly on the '1-norm' i.e. scalar tensor network

This is the simplest version of belief propagation, and is useful for
simple investigations.
"""

import autoray as ar

from quimb.tensor.contraction import array_contract
from quimb.utils import oset

from .bp_common import (
    BeliefPropagationCommon,
    combine_local_contractions,
)
from .hd1bp import (
    compute_all_tensor_messages_tree,
)


def initialize_messages(tn, fill_fn=None):

    backend = ar.infer_backend(next(t.data for t in tn))
    _sum = ar.get_lib_fn(backend, "sum")

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
            messages[ix, tid_to] = m / _sum(m)

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
    damping : float, optional
        The damping factor to use, 0.0 means no damping.
    update : {'sequential', 'parallel'}, optional
        Whether to update messages sequentially or in parallel.
    local_convergence : bool, optional
        Whether to allow messages to locally converge - i.e. if all their
        input messages have converged then stop updating them.
    fill_fn : callable, optional
        If specified, use this function to fill in the initial messages.

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
        tn,
        messages=None,
        damping=0.0,
        update="sequential",
        local_convergence=True,
        message_init_function=None,
    ):
        self.tn = tn
        self.damping = damping
        self.local_convergence = local_convergence
        self.update = update

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
            new_ms = [self._normalize(m) for m in new_ms]
            new_ks = [self.key_pairs[ix, tid] for ix in t.inds]

            return new_ks, new_ms

        def _update_m(key, data):
            nonlocal nconv, max_mdiff

            m = self.messages[key]
            if self.damping != 0.0:
                data = (1 - self.damping) * data + self.damping * m

            mdiff = float(self._distance(m, data))
            if mdiff > tol:
                # mark distination tid for update
                new_touched.add(key[1])
            else:
                nconv += 1

            max_mdiff = max(max_mdiff, mdiff)
            self.messages[key] = data

        if self.update == "sequential":
            # compute each new message and immediately re-insert it
            while self.touched:
                tid = self.touched.pop()
                keys, new_ms = _compute_ms(tid)
                for key, data in zip(keys, new_ms):
                    _update_m(key, data)

        elif self.update == "parallel":
            new_data = {}
            # compute all new messages
            while self.touched:
                tid = self.touched.pop()
                keys, new_ms = _compute_ms(tid)
                for key, data in zip(keys, new_ms):
                    new_data[key] = data
            # insert all new messages
            for key, data in new_data.items():
                _update_m(key, data)

        self.touched = new_touched
        return nconv, ncheck, max_mdiff

    def normalize_messages(self):
        """Normalize all messages such that for each bond `<m_i|m_j> = 1` and
        `<m_i|m_i> = <m_j|m_j>` (but in general != 1).
        """
        for ix, tids in self.tn.ind_map.items():
            if len(tids) != 2:
                continue
            tida, tidb = tids
            mi = self.messages[ix, tida]
            mj = self.messages[ix, tidb]
            nij = abs(mi @ mj)**0.5
            nii = (mi @ mi)**0.25
            njj = (mj @ mj)**0.25
            self.messages[ix, tida] = mi / (nij * nii / njj)
            self.messages[ix, tidb] = mj / (nij * njj / nii)

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

            el, ev = ar.do('linalg.eig', ar.do('outer', ma, mb))
            k = ar.do('argsort', -ar.do('abs', el))
            ev = ev[:, k]
            Uinv = ev
            U = ar.do('linalg.inv', ev)
            tng._insert_gauge_tids(U, tida, tidb, Uinv)
        return tng

    def contract(self, strip_exponent=False):
        tvals = []
        for tid, t in self.tn.tensor_map.items():
            arrays = [t.data]
            inputs = [tuple(range(t.ndim))]
            for i, ix in enumerate(t.inds):
                m = self.messages[ix, tid]
                arrays.append(m)
                inputs.append((i,))
            tvals.append(
                array_contract(
                    arrays=arrays,
                    inputs=inputs,
                    output=(),
                )
            )

        mvals = []
        for ix, tids in self.tn.ind_map.items():
            if len(tids) != 2:
                continue
            tida, tidb = tids
            mvals.append(
                self.messages[ix, tida] @ self.messages[ix, tidb]
            )

        return combine_local_contractions(
            tvals, mvals, self.backend, strip_exponent=strip_exponent
        )



def contract_d1bp(
    tn,
    max_iterations=1000,
    tol=5e-6,
    damping=0.0,
    update="sequential",
    local_convergence=True,
    strip_exponent=False,
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
        The maximum number of iterations to run for.
    tol : float, optional
        The convergence tolerance for messages.
    damping : float, optional
        The damping parameter to use, defaults to no damping.
    update : {'sequential', 'parallel'}, optional
        Whether to update messages sequentially or in parallel.
    local_convergence : bool, optional
        Whether to allow messages to locally converge - i.e. if all their
        input messages have converged then stop updating them.
    strip_exponent : bool, optional
        Whether to strip the exponent from the final result. If ``True``
        then the returned result is ``(mantissa, exponent)``.
    info : dict, optional
        If specified, update this dictionary with information about the
        belief propagation run.
    progbar : bool, optional
        Whether to show a progress bar.
    """
    bp = D1BP(
        tn,
        damping=damping,
        local_convergence=local_convergence,
        update=update,
        **contract_opts,
    )
    bp.run(
        max_iterations=max_iterations,
        tol=tol,
        info=info,
        progbar=progbar,
    )
    return bp.contract(
        strip_exponent=strip_exponent,
    )
