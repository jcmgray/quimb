
    def _compute_oblique_projectors_between_full_env_exact_tids(
        self,
        tida,
        tidb,
        max_bond=None,
        cutoff=1e-10,
        fit_steps=0,
        **fit_opts,
    ):
        k = self.copy()

        ta = k.tensor_map[tida]
        tb = k.tensor_map[tidb]

        # names for bonds
        (bix,) = ta.bonds(tb)
        ix_kl = rand_uuid()
        ix_kr = rand_uuid()
        ix_bl = rand_uuid()
        ix_br = rand_uuid()
        ebix = rand_uuid()

        # make norm network, but with ket and bra bond cut open
        k._cut_between_tids(tida, tidb, ix_kl, ix_kr)
        b = k.conj().reindex_({ix_kl: ix_bl, ix_kr: ix_br})
        ftn = k | b

        # contract full environment
        ft = ftn.contract(
            all,
            output_inds=[ix_kl, ix_kr, ix_bl, ix_br],
            optimize="auto-hq",
        )

        # factor positive environment
        ekt, _ = ft.split(
            left_inds=[ix_kl, ix_kr],
            right_inds=[ix_bl, ix_br],
            get="tensors",
            bond_ind=ebix,
            method="eigh",
        )

        ek = ft.trace([ix_bl], [ix_br]).to_dense([ix_kl], [ix_kr])
        U, _, VH = do("linalg.svd", ek)
        Pl = do("conj", U[:, :max_bond])
        Pr = do("conj", VH[:max_bond, :])

        # # compute left reduced factor
        # Rl = ekt.compute_reduced_factor("right", [ebix, ix_kr], [ix_kl])

        # # compute right reduced factor
        # Rr = ekt.compute_reduced_factor("left", [ix_kr], [ix_kl, ebix])

        # # compute compressed projectors
        # Pl, Pr = compute_oblique_projectors(
        #     Rl,
        #     Rr,
        #     max_bond=max_bond,
        #     cutoff=cutoff,
        # )

        if fit_steps:
            tPl = Tensor(Pl, inds=[ix_kl, bix], tags="Pl")
            tPr = Tensor(Pr, inds=[bix, ix_kr], tags="Pr")
            tn_fit = ekt | tPl | tPr
            tn_target = ekt.trace(ix_kl, ix_kr)
            tn_fit.fit_(
                tn_target,
                tags=["Pl", "Pr"],
                steps=fit_steps,
                **fit_opts,
            )
            Pl = tPl.data
            Pr = tPr.data

        return Pl, Pr

    def _compress_between_full_env_exact_tids(
        self,
        tida,
        tidb,
        max_bond=None,
        cutoff=1e-10,
        fit_steps=0,
        inplace=False,
        **fit_opts,
    ):
        tn = self if inplace else self.copy()

        ta = tn.tensor_map[tida]
        tb = tn.tensor_map[tidb]

        _, bix, _ = tensor_make_single_bond(ta, tb)

        Pl, Pr = tn._compute_oblique_projectors_between_full_env_exact_tids(
            tida,
            tidb,
            max_bond=max_bond,
            cutoff=cutoff,
            fit_steps=fit_steps,
            **fit_opts,
        )

        ta.gate_(Pl, bix, transposed=True)
        tb.gate_(Pr, bix)

        return tn

    def _compress_between_cluster_env_tids(
        self,
        tida,
        tidb,
        max_bond=None,
        cutoff=1e-10,
        fit_steps=10,
        max_distance=1,
        fillin=False,
        gauges=None,
        power=1,
        smudge=1e-12,
        inplace=False,
        **fit_opts,
    ):
        tn = self if inplace else self.copy()
        k = tn._select_local_tids(
            [tida, tidb],
            max_distance=max_distance,
            fillin=fillin,
        )

        if gauges is not None:
            outer, inner = k.gauge_simple_insert(
                gauges,
                power=power,
                smudge=smudge,
            )
        else:
            outer = inner = None

        k._compress_between_full_env_exact_tids(
            tida,
            tidb,
            max_bond=max_bond,
            cutoff=cutoff,
            fit_steps=fit_steps,
            inplace=True,
            **fit_opts,
        )

        if gauges is not None:
            # undo outer gauge
            k.gauge_simple_remove(outer)
            # inner gauges are simply out of date and removed
            for _, ix, _ in inner:
                gauges.pop(ix)

            # regauge
            tn._gauge_local_tids(
                (tida, tidb),
                max_distance=max_distance,
                method="simple",
                gauges=gauges,
                power=power,
                smudge=smudge,
            )
            # tn.gauge_all_simple_(
            #     100,
            #     1e-9,
            #     gauges=gauges,
            #     touched_tids=(tida, tidb),
            # )

        return tn

    def gate_cluster_environment(
        self,
        G,
        where,
        max_bond=None,
        cutoff=1e-10,
        fit_steps=10,
        max_distance=1,
        power=1,
        smudge=1e-12,
        gauges=None,
        inplace=False,
        **fit_opts,
    ):
        tn = self if inplace else self.copy()

        # apply gate without truncation
        if gauges is not None:
            tn.gate_simple_(
                G,
                where,
                gauges=gauges,
                max_bond=None,
                power=power,
                smudge=smudge,
            )
        else:
            tn.gate_(
                G,
                where,
                contract="reduce-split",
                max_bond=None,
            )

        tags = tuple(map(tn.site_tag, where))
        tida, tidb = tn._get_tids_from_tags(tags, "any")

        # compress corresponding bond using cluster environment
        return tn._compress_between_cluster_env_tids(
            tida,
            tidb,
            max_bond=max_bond,
            cutoff=cutoff,
            fit_steps=fit_steps,
            max_distance=max_distance,
            gauges=gauges,
            power=power,
            smudge=smudge,
            inplace=True,
            **fit_opts,
        )

