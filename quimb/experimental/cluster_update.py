"""Implementation of arbitrary geometry wavefunction cluster update.
"""
import functools

from quimb.tensor.tensor_core import ensure_dict, bonds, bonds_size, do
from quimb.tensor.tensor_arbgeom_tebd import SimpleUpdateGen


def gate_inds_nn_fit(
    self,
    G,
    ind1,
    ind2,
    max_bond=None,
    method="als",
    pregauge=2,
    init_simple_guess=True,
    steps=10,
    fit_opts=None,
    contract_opts=None,
    inplace=False,
):
    """Gate two nearest neighbor outer indices, using full fitting of
    reduced tensors with respect to the environment. This is more accurate
    than a simple reduced gate when restricting the bond dimension.

    Parameters
    ----------
    G : array_like
        The gate to fit.
    ind1, ind2 : str
        The indices to gate.
    max_bond : int, optional
        The maximum bond dimension to use. If ``None``, use the maximum
        bond dimension that the tensors currently share.
    method : {'als', 'autodiff'}, optional
        The method to use for fitting.
    pregauge : int, optional
        How many times to locally canonize from the purified environment
        tensor to both the left and right reduced tensors.
    init_simple_guess : bool, optional
        Whether to use a 'simple update' guess for the initial guess. This
        can be quite high quality already if pregauging is used.
    steps : int, optional
        The number of steps to use for fitting, can be ``0`` in which case
        the initial guess is used, which in conjuction with the
        envinronment pregauging can still be quite high quality.
    inplace : bool, optional
        Whether to update the tensor network in place.
    contract_opts
        Supplied to
        :meth:`~quimb.tensor.tensor_core.TensorNetwork.contract`.
    """
    fit_opts = ensure_dict(fit_opts)
    contract_opts = ensure_dict(contract_opts)

    ket = self.copy()

    # move indices onto shared bond so environment can be contracted
    ket.reduce_inds_onto_bond(
        ind1, ind2, combine=False, ndim_cutoff=0, tags="__REDUCED__"
    )

    ket.add_tag("__KET__")
    bra = ket.conj().retag_({"__KET__": "__BRA__"})
    norm = ket | bra

    # contract environment -> all but reduced tensors
    norm.contract_tags_("__REDUCED__", "!any", **contract_opts)

    # get tensors and bond names
    (tide,) = norm._get_tids_from_tags("__REDUCED__", "!any")
    (te,) = norm._tids_get(tide)
    (tid_k1,) = norm._get_tids_from_inds(ind1) & norm._get_tids_from_tags(
        "__KET__"
    )
    (tid_k2,) = norm._get_tids_from_inds(ind2) & norm._get_tids_from_tags(
        "__KET__"
    )
    (tid_b1,) = norm._get_tids_from_inds(ind1) & norm._get_tids_from_tags(
        "__BRA__"
    )
    (tid_b2,) = norm._get_tids_from_inds(ind2) & norm._get_tids_from_tags(
        "__BRA__"
    )
    tk1 = norm.tensor_map[tid_k1]
    tk2 = norm.tensor_map[tid_k2]
    tb1 = norm.tensor_map[tid_b1]
    tb2 = norm.tensor_map[tid_b2]

    (ix_ek1,) = bonds(te, tk1)
    (ix_ek2,) = bonds(te, tk2)
    (ix_eb1,) = bonds(te, tb1)
    (ix_eb2,) = bonds(te, tb2)

    if max_bond is None:
        max_bond = bonds_size(tk1, tk2)

    # split environment to get purification
    _, tek = te.split(
        left_inds=[ix_eb1, ix_eb2],
        right_inds=[ix_ek1, ix_ek2],
        method="svd",
    )

    #   ┌────────────────┐
    # ┌─┴┐      ┌──┐     │   :
    # │k1├──────┤k2├───┐ │   : this is the purification
    # └─┬┘      └─┬┘   ├─┴┐  : we'll fit to gated version of itself
    #   │         │    │ek│  :
    #   │ind1     │    └─┬┘
    #   │     ind2│    ┌─┴┐
    #   │         │    │eb│
    # ┌─┴┐      ┌─┴┐   ├─┬┘
    # │b1├──────┤b2├───┘ │
    # └─┬┘      └──┘     │
    #   └────────────────┘

    for _ in range(int(pregauge)):
        # perform some conditioning: local gauging from Q -> R tensors
        for ind, ix_ek, tk in [
            (ind1, ix_ek1, tk1),
            (ind2, ix_ek2, tk2),
        ]:
            R = tek.split(
                left_inds=None,
                right_inds=[ix_ek],
                method="qr",
                get="arrays",
            )[1]

            Rinv = do("linalg.inv", R)
            # get Q tensor in uncontracted TN
            (tidkq,) = ket._get_tids_from_inds(
                ix_ek
            ) - ket._get_tids_from_inds(ind)
            tkq = ket.tensor_map[tidkq]
            # need to keep environment and bare tensors in sync
            tkq.gate_(Rinv.T, ix_ek)
            tek.gate_(Rinv.T, ix_ek)
            tk.gate_(R, ix_ek)

    # form purification
    tnl = tek | tk1 | tk2
    # form gated purification
    tnl_target = tnl.gate_inds(G, (ind1, ind2), contract=True)

    if init_simple_guess:
        # maybe initialize with simple guess
        tnl.gate_inds_(
            G, (ind1, ind2), contract="split", max_bond=max_bond, cutoff=0.0
        )

    if steps:
        # perform the actual fitting, specifying only the reduced tensors
        tnl.fit_(
            tnl_target,
            tags=["__REDUCED__"],
            method=method,
            steps=steps,
            **fit_opts,
        )

    # re-absorb reduced factors
    ket.contract_ind(ix_ek1)
    ket.contract_ind(ix_ek2)

    # TN we will return
    new = self if inplace else self.copy()
    (tn1,) = new._inds_get(ind1)
    (tn2,) = new._inds_get(ind2)

    # permute to match original tensors
    (t1,) = ket._inds_get(ind1)
    (t2,) = ket._inds_get(ind2)
    t1.transpose_like_(tn1)
    t2.transpose_like_(tn2)

    tn1.modify(data=t1.data)
    tn2.modify(data=t2.data)

    return new


gate_inds_nn_fit_ = functools.partialmethod(gate_inds_nn_fit, inplace=True)

# TensorNetwork.gate_inds_nn_fit =gate_inds_nn_fit
# TensorNetwork.gate_inds_nn_fit_ = gate_inds_nn_fit


class ClusterUpdateNNGen(SimpleUpdateGen):
    """Cluster update for arbitrary geometry nearest neighbor hamiltonians.
    This keeps track of simple update style gauges, in order to approximately
    partial trace beyond ``cluster_radius`` and form an approximate environment
    for two nearest neighbor sites that be used to fit the gate with higher
    quality than simple update only.
    """

    def __init__(
        self,
        psi0,
        ham,
        tau=0.01,
        D=None,
        cluster_radius=1,
        cluster_fillin=0,
        gauge_smudge=1e-6,
        imag=True,
        gate_opts=None,
        ordering=None,
        second_order_reflect=False,
        compute_energy_every=None,
        compute_energy_final=True,
        compute_energy_opts=None,
        compute_energy_fn=None,
        compute_energy_per_site=False,
        callback=None,
        keep_best=False,
        progbar=True,
    ):
        super().__init__(
            psi0=psi0,
            ham=ham,
            tau=tau,
            D=D,
            imag=imag,
            gate_opts=gate_opts,
            ordering=ordering,
            second_order_reflect=second_order_reflect,
            compute_energy_every=compute_energy_every,
            compute_energy_final=compute_energy_final,
            compute_energy_opts=compute_energy_opts,
            compute_energy_fn=compute_energy_fn,
            compute_energy_per_site=compute_energy_per_site,
            callback=callback,
            keep_best=keep_best,
            progbar=progbar,
        )
        self.cluster_radius = cluster_radius
        self.cluster_fillin = cluster_fillin
        self.gauge_smudge = gauge_smudge

        # override some default TEBDGen gate_opts
        self.gate_opts.pop("cutoff")
        self.gate_opts.pop("contract")

    def gate(self, U, where):
        taga, tagb = self._psi.gen_tags_from_coos(where)
        inda, indb = self._psi.gen_inds_from_coos(where)

        # get the local cluster
        psi_local = self._psi.select_local(
            (taga, tagb),
            "any",
            max_distance=self.cluster_radius,
            fillin=self.cluster_fillin,
            virtual=True,
        )

        # temporarily gauge it with 'simple' gauges
        with psi_local.gauge_simple_temp(
            self.gauges,
            smudge=self.gauge_smudge,
        ):
            # fit the gate to the gauged local cluster
            gate_inds_nn_fit(
                psi_local,
                U,
                inda,
                indb,
                **self.gate_opts,
            )

        # update nearest gauges for the modified tensors
        self._psi.gauge_local_(
            (taga, tagb),
            "any",
            max_distance=1,
            method="simple",
            gauges=self.gauges,
            smudge=self.gauge_smudge,
        )

        # perform some conditioning
        self._psi.equalize_norms_(1.0)
        self._psi.exponent = 0.0
