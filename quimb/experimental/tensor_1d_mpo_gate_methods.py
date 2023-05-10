"""Methods for acting with an MPO on an MPS.


TODO:

- [ ] implement early compress boundary method
- [ ] find out why projector method is slower than expected
- [ ] density matrix method

"""
from quimb.tensor.tensor_core import (
    ensure_dict, rand_uuid, tensor_contract,
    )


def mps_gate_with_mpo_boundary(
    self,
    mpo,
    max_bond,
    cutoff=0.0,
):
    return mpo.apply(self, compress=True, max_bond=max_bond, cutoff=cutoff)


def mps_gate_with_mpo_lazy(self, mpo):
    """Apply an MPO to an MPS lazily, i.e. nothing is contracted, but the new
    TN object has the same outer indices as the original MPS.
    """
    mps_calc = self.copy()
    mpo_calc = mpo.copy()

    outerid = self.site_ind_id
    innerid = rand_uuid() + "{}"

    mps_calc.site_ind_id = innerid
    mpo_calc.lower_ind_id = innerid
    mpo_calc.upper_ind_id = outerid

    mps_calc |= mpo_calc

    return mps_calc


def mps_gate_with_mpo_fit(
    self,
    mpo,
    max_bond,
    cutoff=0.0,
    init_guess=None,
    **fit_opts,
):
    """Fit a MPS to a MPO applied to an MPS using either ALS or autodiff.

    Some nice alternatives to the default fit_opts:

        - method="autodiff"
        - method="als", solver="lstsq"

    """
    if cutoff != 0.0:
        raise ValueError("cutoff must be zero for fitting")

    target = mps_gate_with_mpo_lazy(self, mpo)

    if init_guess is None:
        ansatz = self.copy()
        ansatz.expand_bond_dimension_(max_bond)
    else:
        raise NotImplementedError

    return ansatz.fit_(target, **fit_opts)


def mps_gate_with_mpo_projector(
    self,
    mpo,
    max_bond,
    cutoff=0.0,
    canonize=False,
    canonize_opts=None,
    inplace=False,
    **compress_opts,
):
    tn = mps_gate_with_mpo_lazy(self, mpo)

    if canonize:
        # precondition
        canonize_opts = ensure_dict(canonize_opts)
        tn.gauge_all_(**canonize_opts)

    tn_calc = tn.copy()

    for i in range(tn.L - 1):
        ltags = (tn.site_tag(i),)
        rtags = (tn.site_tag(i + 1),)
        tn_calc.insert_compressor_between_regions_(
            ltags,
            rtags,
            new_ltags=ltags,
            new_rtags=rtags,
            max_bond=max_bond,
            cutoff=cutoff,
            insert_into=tn,
            bond_ind=self.bond(i, i + 1),
            **compress_opts
        )

    if inplace:
        for i in range(tn.L):
            ti = self[i]
            data = tensor_contract(
                *tn[i], output_inds=ti.inds,
                optimize="auto-hq"
            ).data
            ti.modify(data=data)

    else:
        for i in range(tn.L):
            tn.contract_tags_(
                tn.site_tag(i), output_inds=self[i].inds,
                optimize="auto-hq",
            )

        tn.view_like_(self)

    return tn
