import math

import autoray as ar

import quimb.tensor as qtn
from .bp_common import (
    BeliefPropagationCommon,
    create_lazy_community_edge_map,
    combine_local_contractions,
)


class L2BP(BeliefPropagationCommon):
    """A simple class to hold all the data for a L2BP run."""

    def __init__(
        self,
        tn,
        site_tags=None,
        damping=0.0,
        local_convergence=True,
        update="parallel",
        optimize="auto-hq",
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

        _abs = ar.get_lib_fn(self.backend, "abs")
        _sum = ar.get_lib_fn(self.backend, "sum")
        _transpose = ar.get_lib_fn(self.backend, "transpose")
        _conj = ar.get_lib_fn(self.backend, "conj")

        def _normalize(x):
            return x / _sum(x)

        def _symmetrize(x):
            N = ar.ndim(x)
            perm = (*range(N // 2, N), *range(0, N // 2))
            return x + _conj(_transpose(x, perm))

        def _distance(x, y):
            return _sum(_abs(x - y))

        self._normalize = _normalize
        self._symmetrize = _symmetrize
        self._distance = _distance

        # initialize messages
        self.messages = {}

        for pair, bix in self.edges.items():
            cix = tuple(ix + "*" for ix in bix)
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
                tm.modify(apply=self._symmetrize)
                tm.modify(apply=self._normalize)
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
                        remapper[ix] = ix + "**"
                    elif ix in outer_bix:
                        # messages connected
                        remapper[ix] = ix + "*"
                    # remaining indices are either internal and will be mangled
                    # or global outer indices and will be contracted directly

                tn_i_right.reindex_(remapper)

                self.contraction_tns[i, j] = qtn.TensorNetwork(
                    (tn_i_left, *tks, tn_i_right), virtual=True
                )

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
            cix = tuple(ix + "**" for ix in bix)
            output_inds = cix + bix

            tn_i_to_j = self.contraction_tns[i, j]

            tm_new = tn_i_to_j.contract(
                all,
                output_inds=output_inds,
                drop_tags=True,
                optimize=self.optimize,
                **self.contract_opts,
            )
            tm_new.modify(apply=self._symmetrize)
            tm_new.modify(apply=self._normalize)
            return tm_new.data

        def _update_m(key, data):
            nonlocal nconv, max_mdiff

            tm = self.messages[key]

            if self.damping > 0.0:
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
        """Estimate the contraction of the norm squared using the current
        messages.
        """
        tvals = []
        for i, ket in self.local_tns.items():
            # we allow missing keys here for tensors which are just
            # disconnected but still appear in local_tns
            ks = self.neighbors.get(i, ())
            bix = [ix for k in ks for ix in self.edges[tuple(sorted((k, i)))]]
            bra = ket.H.reindex_({ix: ix + "*" for ix in bix})
            tni = qtn.TensorNetwork(
                (
                    ket,
                    *(self.messages[k, i] for k in ks),
                    bra,
                )
            )
            tvals.append(
                tni.contract(all, optimize=self.optimize, **self.contract_opts)
            )

        mvals = []
        for i, j in self.edges:
            mvals.append(
                (self.messages[i, j] & self.messages[j, i]).contract(
                    all,
                    optimize=self.optimize,
                    **self.contract_opts,
                )
            )

        return combine_local_contractions(
            tvals, mvals, self.backend, strip_exponent=strip_exponent
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
                bra_site_ind = ix + "**"
                ind_changes[ix] = bra_site_ind
            if ix in outer_bix:
                # attach bra message indices
                ind_changes[ix] = ix + "*"
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
