import autoray as ar

import quimb.tensor as qtn

from .bp_common import (
    BeliefPropagationCommon,
    create_lazy_community_edge_map,
    combine_local_contractions,
)


class L1BP(BeliefPropagationCommon):
    """Lazy 1-norm belief propagation. BP is run between groups of tensors
    defined by ``site_tags``. The message updates are lazy contractions.

    Parameters
    ----------
    tn : TensorNetwork
        The tensor network to run BP on.
    site_tags : sequence of str, optional
        The tags identifying the sites in ``tn``, each tag forms a region,
        which should not overlap. If the tensor network is structured, then
        these are inferred automatically.
    damping : float, optional
        The damping parameter to use, defaults to no damping.
    local_convergence : bool, optional
        Whether to allow messages to locally converge - i.e. if all their
        input messages have converged then stop updating them.
    optimize : str or PathOptimizer, optional
        The path optimizer to use when contracting the messages.
    contract_opts
        Other options supplied to ``cotengra.array_contract``.
    """

    def __init__(
        self,
        tn,
        site_tags=None,
        damping=0.0,
        local_convergence=True,
        update="parallel",
        optimize="auto-hq",
        message_init_function=None,
        **contract_opts,
    ):
        self.backend = next(t.backend for t in tn)
        self.damping = damping
        self.local_convergence = local_convergence
        self.update = update
        self.optimize = optimize
        self.contract_opts = contract_opts

        if site_tags is None:
            self.site_tags = tuple(tn.site_tags)
        else:
            self.site_tags = tuple(site_tags)

        (
            self.edges,
            self.neighbors,
            self.local_tns,
            self.touch_map,
        ) = create_lazy_community_edge_map(tn, site_tags)
        self.touched = set()

        self._abs = ar.get_lib_fn(self.backend, "abs")
        self._max = ar.get_lib_fn(self.backend, "max")
        self._sum = ar.get_lib_fn(self.backend, "sum")
        _real = ar.get_lib_fn(self.backend, "real")
        _argmax = ar.get_lib_fn(self.backend, "argmax")
        _reshape = ar.get_lib_fn(self.backend, "reshape")
        self._norm = ar.get_lib_fn(self.backend, "linalg.norm")

        def _normalize(x):
            return x / self._sum(x)
            # return x / self._norm(x)
            # return x / self._max(x)
            # fx = _reshape(x, (-1,))
            # return x / fx[_argmax(self._abs(_real(fx)))]

        def _distance(x, y):
            return self._sum(self._abs(x - y))

        self._normalize = _normalize
        self._distance = _distance

        # for each meta bond create initial messages
        self.messages = {}
        for pair, bix in self.edges.items():
            # compute leftwards and rightwards messages
            for i, j in (sorted(pair), sorted(pair, reverse=True)):
                tn_i = self.local_tns[i]
                # initial message just sums over dangling bonds

                if message_init_function is None:
                    tm = tn_i.contract(
                        all,
                        output_inds=bix,
                        optimize=self.optimize,
                        drop_tags=True,
                        **self.contract_opts,
                    )
                    # normalize
                    tm.modify(apply=self._normalize)
                else:
                    shape = tuple(tn_i.ind_size(ix) for ix in bix)
                    tm = qtn.Tensor(
                        data=message_init_function(shape),
                        inds=bix,
                    )

                self.messages[i, j] = tm

        # compute the contractions
        self.contraction_tns = {}
        for pair, bix in self.edges.items():
            # for each meta bond compute left and right contractions
            for i, j in (sorted(pair), sorted(pair, reverse=True)):
                tn_i = self.local_tns[i].copy()
                # attach incoming messages to dangling bonds
                tks = [
                    self.messages[k, i] for k in self.neighbors[i] if k != j
                ]
                # virtual so we can modify messages tensors inplace
                tn_i_to_j = qtn.TensorNetwork((tn_i, *tks), virtual=True)
                self.contraction_tns[i, j] = tn_i_to_j

    def iterate(self, tol=5e-6):
        if (not self.local_convergence) or (not self.touched):
            # assume if asked to iterate that we want to check all messages
            self.touched.update(
                pair for edge in self.edges for pair in (edge, edge[::-1])
            )

        ncheck = len(self.touched)
        nconv = 0
        max_mdiff = -1.0
        new_touched = set()

        def _compute_m(key):
            i, j = key
            bix = self.edges[(i, j) if i < j else (j, i)]
            tn_i_to_j = self.contraction_tns[i, j]
            tm_new = tn_i_to_j.contract(
                all,
                output_inds=bix,
                optimize=self.optimize,
                **self.contract_opts,
            )
            return self._normalize(tm_new.data)

        def _update_m(key, data):
            nonlocal nconv, max_mdiff

            tm = self.messages[key]

            if self.damping != 0.0:
                data = (1 - self.damping) * data + self.damping * tm.data

            mdiff = float(self._distance(tm.data, data))

            if mdiff > tol:
                # mark touching messages for update
                new_touched.update(self.touch_map[key])
            else:
                nconv += 1

            max_mdiff = max(max_mdiff, mdiff)
            tm.modify(data=data)

        if self.update == "parallel":
            new_data = {}
            # compute all new messages
            while self.touched:
                key = self.touched.pop()
                new_data[key] = _compute_m(key)
            # insert all new messages
            for key, data in new_data.items():
                _update_m(key, data)

        elif self.update == "sequential":
            # compute each new message and immediately re-insert it
            while self.touched:
                key = self.touched.pop()
                data = _compute_m(key)
                _update_m(key, data)

        self.touched = new_touched
        return nconv, ncheck, max_mdiff

    def contract(self, strip_exponent=False):
        tvals = []
        for site, tn_ic in self.local_tns.items():
            if site in self.neighbors:
                tval = qtn.tensor_contract(
                    *tn_ic,
                    *(self.messages[k, site] for k in self.neighbors[site]),
                    optimize=self.optimize,
                    **self.contract_opts,
                )
            else:
                # site exists but has no neighbors
                tval = tn_ic.contract(
                    all,
                    output_inds=(),
                    optimize=self.optimize,
                    **self.contract_opts,
                )
            tvals.append(tval)

        mvals = []
        for i, j in self.edges:
            mval = qtn.tensor_contract(
                self.messages[i, j],
                self.messages[j, i],
                optimize=self.optimize,
                **self.contract_opts,
            )
            mvals.append(mval)

        return combine_local_contractions(
            tvals, mvals, self.backend, strip_exponent=strip_exponent
        )


def contract_l1bp(
    tn,
    max_iterations=1000,
    tol=5e-6,
    site_tags=None,
    damping=0.0,
    local_convergence=True,
    update="parallel",
    optimize="auto-hq",
    strip_exponent=False,
    info=None,
    progbar=False,
    **contract_opts,
):
    """Estimate the contraction of ``tn`` using lazy 1-norm belief propagation.

    Parameters
    ----------
    tn : TensorNetwork
        The tensor network to contract.
    max_iterations : int, optional
        The maximum number of iterations to perform.
    tol : float, optional
        The convergence tolerance for messages.
    site_tags : sequence of str, optional
        The tags identifying the sites in ``tn``, each tag forms a region. If
        the tensor network is structured, then these are inferred
        automatically.
    damping : float, optional
        The damping parameter to use, defaults to no damping.
    local_convergence : bool, optional
        Whether to allow messages to locally converge - i.e. if all their
        input messages have converged then stop updating them.
    optimize : str or PathOptimizer, optional
        The path optimizer to use when contracting the messages.
    progbar : bool, optional
        Whether to show a progress bar.
    strip_exponent : bool, optional
        Whether to strip the exponent from the final result. If ``True``
        then the returned result is ``(mantissa, exponent)``.
    info : dict, optional
        If specified, update this dictionary with information about the
        belief propagation run.
    contract_opts
        Other options supplied to ``cotengra.array_contract``.
    """
    bp = L1BP(
        tn,
        site_tags=site_tags,
        damping=damping,
        local_convergence=local_convergence,
        update=update,
        optimize=optimize,
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
