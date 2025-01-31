import math

import autoray as ar

import quimb.tensor as qtn
from quimb.utils import oset

from .bp_common import (
    BeliefPropagationCommon,
    combine_local_contractions,
    create_lazy_community_edge_map,
)


def _identity(x):
    return x


class L2BP(BeliefPropagationCommon):
    """Lazy (as in multiple uncontracted tensors per site) 2-norm (as in for
    wavefunctions and operators) belief propagation.

    Parameters
    ----------
    tn : TensorNetwork
        The tensor network to form the 2-norm of and run BP on.
    site_tags : sequence of str, optional
        The tags identifying the sites in ``tn``, each tag forms a region,
        which should not overlap. If the tensor network is structured, then
        these are inferred automatically.
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
    symmetrize : bool or callable, optional
        Whether to symmetrize the messages, i.e. for each message ensure that
        it is hermitian with respect to its bra and ket indices. If a callable
        it should take a message and return the symmetrized message.
    local_convergence : bool, optional
        Whether to allow messages to locally converge - i.e. if all their
        input messages have converged then stop updating them.
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
        update="sequential",
        normalize=None,
        distance=None,
        symmetrize=True,
        local_convergence=True,
        optimize="auto-hq",
        contract_every=None,
        inplace=False,
        **contract_opts,
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
        self.touched = oset()

        # these are all settable properties
        self.symmetrize = symmetrize

        # initialize messages
        self.messages = {}

        for pair, bix in self.edges.items():
            cix = tuple(ix + "_l2bp*" for ix in bix)
            remapper = dict(zip(bix, cix))
            output_inds = cix + bix

            # compute leftwards and righwards messages
            for i, j in (sorted(pair), sorted(pair, reverse=True)):
                tn_i = self.local_tns[i]
                tn_i2 = tn_i & tn_i.conj().reindex_(remapper)
                tm = tn_i2.contract(
                    all,
                    output_inds=output_inds,
                    optimize=self.optimize,
                    drop_tags=True,
                    **self.contract_opts,
                )
                tm.modify(apply=self._symmetrize_fn)
                tm.modify(apply=self._normalize_fn)
                self.messages[i, j] = tm

        # initialize contractions
        self.contraction_tns = {}
        for pair, bix in self.edges.items():
            for i, j in (sorted(pair), sorted(pair, reverse=True)):
                # form the ket side and messages
                tn_i_left = self.local_tns[i]
                # get other incident nodes which aren't j
                ks = [k for k in self.neighbors[i] if k != j]
                tks = [self.messages[k, i] for k in ks]

                # form the 'bra' side
                tn_i_right = tn_i_left.conj()
                # get the bonds that attach the bra to messages
                outer_bix = {
                    ix for k in ks for ix in self.edges[tuple(sorted((k, i)))]
                }
                # need to reindex to join message bonds, and create bra outputs
                remapper = {}
                for ix in tn_i_right.ind_map:
                    if ix in bix:
                        # bra outputs
                        remapper[ix] = ix + "_l2bp**"
                    elif ix in outer_bix:
                        # messages connected
                        remapper[ix] = ix + "_l2bp*"
                    # remaining indices are either internal and will be mangled
                    # or global outer indices and will be contracted directly

                tn_i_right.reindex_(remapper)

                self.contraction_tns[i, j] = qtn.TensorNetwork(
                    (tn_i_left, *tks, tn_i_right), virtual=True
                )

    @property
    def symmetrize(self):
        return self._symmetrize

    @symmetrize.setter
    def symmetrize(self, symmetrize):
        if callable(symmetrize):
            # explicit function
            self._symmetrize = True
            self._symmetrize_fn = symmetrize

        elif symmetrize:
            # default symmetrization
            _transpose = ar.get_lib_fn(self.backend, "transpose")
            _conj = ar.get_lib_fn(self.backend, "conj")

            def _symmetrize_fn(x):
                N = ar.ndim(x)
                perm = (*range(N // 2, N), *range(0, N // 2))
                # XXX: do this blockwise for block/fermi arrays?
                return x + _conj(_transpose(x, perm))

            self._symmetrize = True
            self._symmetrize_fn = _symmetrize_fn

        else:
            # no symmetrization
            self._symmetrize = False
            self._symmetrize_fn = _identity

    def iterate(self, tol=5e-6):
        if (not self.local_convergence) or (not self.touched):
            # assume if asked to iterate that we want to check all messages
            self.touched.update(
                pair for edge in self.edges for pair in (edge, edge[::-1])
            )

        ncheck = len(self.touched)
        nconv = 0
        max_mdiff = -1.0
        new_touched = oset()

        def _compute_m(key):
            i, j = key
            bix = self.edges[(i, j) if i < j else (j, i)]
            cix = tuple(ix + "_l2bp**" for ix in bix)
            output_inds = cix + bix

            tn_i_to_j = self.contraction_tns[i, j]

            tm_new = tn_i_to_j.contract(
                all,
                output_inds=output_inds,
                drop_tags=True,
                optimize=self.optimize,
                **self.contract_opts,
            )
            tm_new.modify(apply=self._symmetrize_fn)
            tm_new.modify(apply=self._normalize_fn)
            return tm_new.data

        def _update_m(key, data):
            nonlocal nconv, max_mdiff

            tm = self.messages[key]

            # pre-damp distance
            mdiff = self._distance_fn(data, tm.data)

            if self.damping:
                data = self.fn_damping(data, tm.data)

            # # post-damp distance
            # mdiff = self._distance_fn(data, tm.data)

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

        return {
            "nconv": nconv,
            "ncheck": ncheck,
            "max_mdiff": max_mdiff,
        }

    def normalize_message_pairs(self):
        """Normalize all messages such that for each bond `<m_i|m_j> = 1` and
        `<m_i|m_i> = <m_j|m_j>` (but in general != 1). This is different to
        normalizing each message.
        """
        for i, j in self.edges:
            tmi = self.messages[i, j]
            tmj = self.messages[j, i]
            nij = (tmi @ tmj) ** 0.5
            nii = (tmi @ tmi) ** 0.25
            njj = (tmj @ tmj) ** 0.25
            tmi /= nij * nii / njj
            tmj /= nij * njj / nii

    def contract(self, strip_exponent=False, check_zero=True):
        """Estimate the contraction of the norm squared using the current
        messages.
        """
        zvals = []
        for i, ket in self.local_tns.items():
            # we allow missing keys here for tensors which are just
            # disconnected but still appear in local_tns
            ks = self.neighbors.get(i, ())
            bix = [ix for k in ks for ix in self.edges[tuple(sorted((k, i)))]]
            bra = ket.H.reindex_({ix: ix + "_l2bp*" for ix in bix})
            tni = qtn.TensorNetwork(
                (
                    ket,
                    *(self.messages[k, i] for k in ks),
                    bra,
                )
            )
            z = tni.contract(all, optimize=self.optimize, **self.contract_opts)
            zvals.append((z, 1))

        for i, j in self.edges:
            z = (self.messages[i, j] & self.messages[j, i]).contract(
                all,
                optimize=self.optimize,
                **self.contract_opts,
            )
            # power / counting factor is -1 for messages, i.e. divide
            zvals.append((z, -1))

        return combine_local_contractions(
            zvals,
            backend=self.backend,
            strip_exponent=strip_exponent,
            check_zero=check_zero,
        )

    def partial_trace(
        self,
        site,
        normalized=True,
        optimize="auto-hq",
    ):
        example_tn = next(tn for tn in self.local_tns.values())

        site_tag = example_tn.site_tag(site)
        ket_site_ind = example_tn.site_ind(site)

        ks = self.neighbors[site_tag]
        tn_rho_i = self.local_tns[site_tag].copy()
        tn_bra_i = tn_rho_i.H

        for k in ks:
            tn_rho_i &= self.messages[k, site_tag]

        outer_bix = {
            ix for k in ks for ix in self.edges[tuple(sorted((k, site_tag)))]
        }

        ind_changes = {}
        for ix in tn_bra_i.ind_map:
            if ix == ket_site_ind:
                # open up the site index
                bra_site_ind = ix + "_l2bp**"
                ind_changes[ix] = bra_site_ind
            if ix in outer_bix:
                # attach bra message indices
                ind_changes[ix] = ix + "_l2bp*"
        tn_bra_i.reindex_(ind_changes)

        tn_rho_i &= tn_bra_i

        rho_i = tn_rho_i.to_dense(
            [ket_site_ind],
            [bra_site_ind],
            optimize=optimize,
            **self.contract_opts,
        )
        if normalized:
            rho_i = rho_i / ar.do("trace", rho_i)

        return rho_i

    def compress(
        self,
        tn,
        max_bond=None,
        cutoff=5e-6,
        cutoff_mode="rsum2",
        renorm=0,
        lazy=False,
    ):
        """Compress the state ``tn``, assumed to matched this L2BP instance,
        using the messages stored.
        """
        for (i, j), bix in self.edges.items():
            tml = self.messages[i, j]
            tmr = self.messages[j, i]

            bix_sizes = [tml.ind_size(ix) for ix in bix]
            dm = math.prod(bix_sizes)

            ml = ar.reshape(tml.data, (dm, dm))
            dl = self.local_tns[i].outer_size() // dm
            Rl = qtn.decomp.squared_op_to_reduced_factor(
                ml, dl, dm, right=True
            )

            mr = ar.reshape(tmr.data, (dm, dm)).T
            dr = self.local_tns[j].outer_size() // dm
            Rr = qtn.decomp.squared_op_to_reduced_factor(
                mr, dm, dr, right=False
            )

            Pl, Pr = qtn.decomp.compute_oblique_projectors(
                Rl,
                Rr,
                cutoff_mode=cutoff_mode,
                renorm=renorm,
                max_bond=max_bond,
                cutoff=cutoff,
            )

            Pl = ar.do("reshape", Pl, (*bix_sizes, -1))
            Pr = ar.do("reshape", Pr, (-1, *bix_sizes))

            ltn = tn.select(i)
            rtn = tn.select(j)

            new_lix = [qtn.rand_uuid() for _ in bix]
            new_rix = [qtn.rand_uuid() for _ in bix]
            new_bix = [qtn.rand_uuid()]
            ltn.reindex_(dict(zip(bix, new_lix)))
            rtn.reindex_(dict(zip(bix, new_rix)))

            # ... and insert the new projectors in place
            tn |= qtn.Tensor(Pl, inds=new_lix + new_bix, tags=(i,))
            tn |= qtn.Tensor(Pr, inds=new_bix + new_rix, tags=(j,))

        if not lazy:
            for st in self.site_tags:
                try:
                    tn.contract_tags_(
                        st, optimize=self.optimize, **self.contract_opts
                    )
                except KeyError:
                    pass

        return tn


def contract_l2bp(
    tn,
    site_tags=None,
    damping=0.0,
    update="sequential",
    local_convergence=True,
    optimize="auto-hq",
    max_iterations=1000,
    tol=5e-6,
    strip_exponent=False,
    info=None,
    progbar=False,
    **contract_opts,
):
    """Estimate the norm squared of ``tn`` using lazy belief propagation.

    Parameters
    ----------
    tn : TensorNetwork
        The tensor network to estimate the norm squared of.
    site_tags : sequence of str, optional
        The tags identifying the sites in ``tn``, each tag forms a region.
    damping : float, optional
        The damping parameter to use, defaults to no damping.
    update : {'parallel', 'sequential'}, optional
        Whether to update all messages in parallel or sequentially.
    local_convergence : bool, optional
        Whether to allow messages to locally converge - i.e. if all their
        input messages have converged then stop updating them.
    optimize : str or PathOptimizer, optional
        The contraction strategy to use.
    max_iterations : int, optional
        The maximum number of iterations to perform.
    tol : float, optional
        The convergence tolerance for messages.
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
    """
    bp = L2BP(
        tn,
        site_tags=site_tags,
        damping=damping,
        update=update,
        local_convergence=local_convergence,
        optimize=optimize,
        **contract_opts,
    )
    bp.run(
        max_iterations=max_iterations,
        tol=tol,
        info=info,
        progbar=progbar,
    )
    return bp.contract(strip_exponent=strip_exponent)


def compress_l2bp(
    tn,
    max_bond,
    cutoff=0.0,
    cutoff_mode="rsum2",
    max_iterations=1000,
    tol=5e-6,
    site_tags=None,
    damping=0.0,
    update="sequential",
    local_convergence=True,
    optimize="auto-hq",
    lazy=False,
    inplace=False,
    info=None,
    progbar=False,
    **contract_opts,
):
    """Compress ``tn`` using lazy belief propagation, producing a tensor
    network with a single tensor per site.

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
        Whether to allow messages to locally converge - i.e. if all their
        input messages have converged then stop updating them.
    optimize : str or PathOptimizer, optional
        The path optimizer to use when contracting the messages.
    lazy : bool, optional
        Whether to perform the compression lazily, i.e. to leave the computed
        compression projectors uncontracted.
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
    tnc = tn if inplace else tn.copy()

    bp = L2BP(
        tnc,
        site_tags=site_tags,
        damping=damping,
        update=update,
        local_convergence=local_convergence,
        optimize=optimize,
        **contract_opts,
    )
    bp.run(
        max_iterations=max_iterations,
        tol=tol,
        info=info,
        progbar=progbar,
    )
    return bp.compress(
        tnc,
        max_bond=max_bond,
        cutoff=cutoff,
        cutoff_mode=cutoff_mode,
        lazy=lazy,
    )
