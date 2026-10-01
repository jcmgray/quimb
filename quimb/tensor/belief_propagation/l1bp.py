import quimb.tensor as qtn
from quimb.tensor.networking import SweepScheduler

from .bp_common import (
    BeliefPropagationCommon,
    combine_local_contractions,
    create_lazy_community_edge_map,
)


class L1BP(BeliefPropagationCommon):
    """Lazy 1-norm belief propagation. BP is run between groups of tensors
    defined by ``site_tags``. The message updates are lazy contractions.

    Parameters
    ----------
    tn : TensorNetwork
        The tensor network to run BP on.
    site_tags : sequence of str, optional
        Tags for sites in ``tn``. Each tag should select a region, and
        regions must not overlap. Infer tags for structured networks.
    damping : float or callable, optional
        Mix old and new messages as ``damping * old + (1 - damping) * new``.
        This can make convergence more reliable but slower.
    diis : bool or dict, optional
        Use direct inversion in the iterative subspace (DIIS) to estimate
        messages with lower error. A dict can contain DIIS options:
        ``max_history``, ``beta``, ``rcond``.
    update : {'sequential', 'parallel'}, optional
        With 'sequential', use new messages in later updates of the same
        sweep. With 'parallel', use only messages from the previous sweep.
        Sequential updates usually help convergence. Parallel updates can
        converge to different solutions.
    sweep_order : {'auto', 'alternate', 'fixed', None}, optional
        Sequential update order. Use 'alternate' to reverse each sweep,
        starting reversed, or 'fixed' to keep the same order. With 'auto',
        alternate on graphs with no loops, else use a fixed order. With None,
        use reverse insertion order. See :class:`.SweepScheduler`.
    normalize : {'L1', 'L2', 'L2phased', 'Linf', callable}, optional
        Norm to use after each update. With None, choose automatically.
        Callables should take a message and return its normalized form.
        'L2phased' also normalizes the phase, and is the default for complex
        dtypes.
    distance : {'L1', 'L2', 'L2phased', 'Linf', 'cosine', callable}, optional
        Distance between messages to check convergence. With None, choose
        automatically. Callables should take two messages and return their
        distance. 'L2phased' normalizes their phases before computing the L2
        distance. It is the default for complex dtypes when normalization
        does not already fix the phase.
    local_convergence : bool, optional
        Stop updating messages whose input messages have converged.
    optimize : str or PathOptimizer, optional
        The path optimizer to use when contracting the messages.
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
        site_tags=None,
        *,
        damping=0.0,
        diis=False,
        update="sequential",
        sweep_order="auto",
        normalize=None,
        distance=None,
        local_convergence=True,
        optimize="auto-hq",
        message_init_function=None,
        contract_every=None,
        inplace=False,
        **contract_opts,
    ):
        super().__init__(
            tn,
            damping=damping,
            diis=diis,
            update=update,
            normalize=normalize,
            distance=distance,
            contract_every=contract_every,
            inplace=inplace,
        )

        self.local_convergence = local_convergence
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
        self.sweeper = SweepScheduler(
            nodes=self.site_tags,
            edges=tuple(self.edges),
            # rank messages by their source site
            key_to_nodes=lambda key: key[:1],
            sweep_order=sweep_order,
        )

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
                    tm.modify(apply=self._normalize_fn)
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
        """Perform one round of message passing."""
        if (not self.local_convergence) or (not self.sweeper.touched):
            # check all messages when none are marked for update
            self.sweeper.touch(
                pair for edge in self.edges for pair in (edge, edge[::-1])
            )

        ncheck = 0
        nconv = 0
        max_mdiff = -1.0

        def _compute_m(key):
            nonlocal ncheck
            ncheck += 1
            i, j = key
            bix = self.edges[(i, j) if i < j else (j, i)]
            tn_i_to_j = self.contraction_tns[i, j]
            tm_new = tn_i_to_j.contract(
                all,
                output_inds=bix,
                optimize=self.optimize,
                **self.contract_opts,
            )
            return self._normalize_fn(tm_new.data)

        def _update_m(key, data):
            nonlocal nconv, max_mdiff

            tm = self.messages[key]

            # measure change before damping
            mdiff = self._distance_fn(data, tm.data)

            if self.damping:
                data = self._damping_fn(data, tm.data)

            if mdiff > tol:
                # mark touching messages for update
                self.sweeper.touch(self.touch_map[key])
            else:
                nconv += 1

            max_mdiff = max(max_mdiff, mdiff)
            tm.modify(data=data)

        if self.update == "parallel":
            new_data = {}
            # compute all messages before updating any
            for key in self.sweeper.sweep():
                new_data[key] = _compute_m(key)
            for key, data in new_data.items():
                _update_m(key, data)

        elif self.update == "sequential":
            # use each new message in subsequent updates
            for key in self.sweeper.sweep():
                data = _compute_m(key)
                _update_m(key, data)

        return {
            "nconv": nconv,
            "ncheck": ncheck,
            "npending": len(self.sweeper.touched),
            "max_mdiff": max_mdiff,
        }

    def contract(self, strip_exponent=False, check_zero=True, **kwargs):
        """Contract the target tensor network via lazy belief propagation using
        the current messages.
        """
        zvals = []
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
            zvals.append((tval, 1))

        for i, j in self.edges:
            mval = qtn.tensor_contract(
                self.messages[i, j],
                self.messages[j, i],
                optimize=self.optimize,
                **self.contract_opts,
            )
            # power / counting factor is -1 for messages
            zvals.append((mval, -1))

        return combine_local_contractions(
            zvals,
            backend=self.backend,
            strip_exponent=strip_exponent,
            check_zero=check_zero,
            mantissa=self.sign,
            exponent=self.exponent,
            **kwargs,
        )

    def normalize_message_pairs(self):
        """Normalize all messages such that for each bond `<m_i|m_j> = 1` and
        `<m_i|m_i> = <m_j|m_j>` (but in general != 1).
        """
        for i, j in self.edges:
            tmi = self.messages[i, j]
            tmj = self.messages[j, i]
            nij = abs(tmi @ tmj) ** 0.5
            nii = (tmi @ tmi) ** 0.25
            njj = (tmj @ tmj) ** 0.25
            tmi /= nij * nii / njj
            tmj /= nij * njj / nii


def contract_l1bp(
    tn,
    max_iterations=1000,
    tol=5e-6,
    site_tags=None,
    damping=0.0,
    update="sequential",
    diis=False,
    local_convergence=True,
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
    update : {'parallel', 'sequential'}, optional
        Whether to update all messages in parallel or sequentially.
    local_convergence : bool, optional
        Stop updating messages whose input messages have converged.
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
        diis=diis,
        info=info,
        progbar=progbar,
    )
    return bp.contract(
        strip_exponent=strip_exponent,
    )
