"""Methods for acting with an MPO on an MPS.


TODO:

- [x] density matrix method
- [x] zip-up method
- [ ] implement early compress boundary method
- [ ] find out why projector method is slower than expected
- [ ] left/right compressed optimal projector method

"""
from quimb.tensor.tensor_core import (
    ensure_dict,
    rand_uuid,
    tensor_contract,
    TensorNetwork,
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

    mps_calc._site_ind_id = outerid

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
            **compress_opts,
        )

    if inplace:
        for i in range(tn.L):
            ti = self[i]
            data = tensor_contract(
                *tn[i], output_inds=ti.inds, optimize="auto-hq"
            ).data
            ti.modify(data=data)

    else:
        for i in range(tn.L):
            tn.contract_tags_(
                tn.site_tag(i),
                output_inds=self[i].inds,
                optimize="auto-hq",
            )

        tn.view_like_(self)

    return tn


def tensor_1d_compress_dm(
    self,
    max_bond=None,
    cutoff=1e-10,
    optimize="auto-hq",
    normalize=False,
    **compress_opts,
):
    ket = self.copy()
    bra = ket.H
    # doing this means forming the norm doesn't do its own mangling
    bra.mangle_inner_()
    # form the overlapping double layer TN
    norm = bra & ket
    # open the bra's indices back up
    bra.reindex_all_("__b{}")

    # construct dense left environments
    left_envs = {}
    left_envs[1] = norm.select(0).contract(optimize=optimize, drop_tags=True)
    for i in range(2, self.L):
        left_envs[i] = tensor_contract(
            left_envs[i - 1],
            *norm.select(i - 1),
            optimize=optimize,
            drop_tags=True,
        )

    # build projectors and right environments
    Us = []
    right_env_ket = None
    right_env_bra = None
    for i in range(self.L - 1, 0, -1):
        # form the reduced density matrix
        rho_tensors = [left_envs[i], *ket.select(i), *bra.select(i)]
        left_inds = [ket.site_ind(i)]
        right_inds = [bra.site_ind(i)]
        if right_env_ket is not None:
            rho_tensors.extend([right_env_ket, right_env_bra])
            left_inds.append(f"__kr{i + 1}")
            right_inds.append(f"__br{i + 1}")

        # contract and then split it
        rhoi = tensor_contract(*rho_tensors, optimize=optimize)
        U, s, UH = rhoi.split(
            left_inds=left_inds,
            right_inds=right_inds,
            method="eigh",
            max_bond=max_bond,
            cutoff=cutoff,
            get="tensors",
            absorb=None,
            **compress_opts,
        )

        # turn bond into 'virtual right' indices
        (bix,) = s.inds
        U.reindex_({bix: f"__kr{i}"})
        UH.reindex_({bix: f"__br{i}"})
        Us.append(U)

        # attach the unitaries to the right environments and contract
        right_ket_tensors = [*ket.select(i), U.H]
        right_bra_tensors = [*bra.select(i), UH.H]
        if right_env_ket is not None:
            right_ket_tensors.append(right_env_ket)
            right_bra_tensors.append(right_env_bra)

        right_env_ket = tensor_contract(
            *right_ket_tensors, optimize=optimize, drop_tags=True
        )
        # TODO: could compute this just as conjugated and relabelled ket env
        right_env_bra = tensor_contract(
            *right_bra_tensors, optimize=optimize, drop_tags=True
        )

    # form the final site
    U0 = tensor_contract(*ket.select(0), right_env_ket, optimize=optimize)

    if normalize:
        # in right canonical form already
        U0.normalize_()

    new = TensorNetwork([U0] + Us[::-1])
    # cast as whatever the input was e.g. MPS
    new.view_like_(self)
    # this puts the array indices in canonical order
    new.permute_arrays()

    return new


def mps_gate_with_mpo_dm(
    mps,
    mpo,
    max_bond=None,
    cutoff=1e-10,
    **compress_opts,
):
    """Gate this MPS with an MPO, using the density matrix compression method.

    Parameters
    ----------
    mps : MatrixProductState
        The MPS to gate.
    mpo : MatrixProductOperator
        The MPO to gate with.
    max_bond : int, optional
        The maximum bond dimension to keep when compressing the double layer
        tensor network, if any.
    cutoff : float, optional
        The truncation error to use when compressing the double layer tensor
        network, if any.
    compress_opts
        Supplied to :func:`~quimb.tensor.tensor_split`.
    """
    # form the double layer tensor network
    target = mps_gate_with_mpo_lazy(mps, mpo)

    # directly compress it without first contracting site-wise
    return tensor_1d_compress_dm(target, max_bond, cutoff, **compress_opts)


def mps_gate_with_mpo_zipup(
    mps,
    mpo,
    max_bond=None,
    cutoff=1e-10,
    canonize=True,
    optimize="auto-hq",
    **compress_opts,
):
    """
    "Minimally Entangled Typical Thermal State Algorithms", E.M. Stoudenmire &
    Steven R. White (https://arxiv.org/abs/1002.1305).
    """
    mps = mps.copy()
    mpo = mpo.copy()

    if canonize:
        # put in 'pseudo' right canonical form:
        #
        #     │ │ │ │ │ │ │ │ │ │
        #     ○─◀─◀─◀─◀─◀─◀─◀─◀─◀  MPO
        #     │ │ │ │ │ │ │ │ │ │
        #     ○─◀─◀─◀─◀─◀─◀─◀─◀─◀  MPS
        #
        mps.right_canonize()
        mpo.right_canonize()

    # form double layer
    tn = mps_gate_with_mpo_lazy(mps, mpo)

    # zip along the bonds
    Us = []
    bix = None
    sVH = None
    for i in range(tn.L - 1):
        #             sVH
        #     │ │ │ │     │ │ │ │ │ │ │
        #     ▶═▶═▶═▶══□──◀─◀─◀─◀─◀─◀─◀
        #        :      ╲ │ │ │ │ │ │ │
        #   max_bond      ◀─◀─◀─◀─◀─◀─◀
        #                 i
        #              .... contract
        if sVH is None:
            # first site
            C = tn.select(i).contract(optimize=optimize)
        else:
            C = (sVH | tn.select(i)).contract(optimize=optimize)
        #                i
        #     │ │ │ │    │  │ │ │ │ │ │
        #     ▶═▶═▶═▶════□──◀─◀─◀─◀─◀─◀
        #             :   ╲ │ │ │ │ │ │
        #           bix  :  ◀─◀─◀─◀─◀─◀
        #               split
        left_inds = [mps.site_ind(i)]
        if bix is not None:
            left_inds.append(bix)

        # the new bond index
        bix = rand_uuid()

        U, sVH = C.split(
            left_inds,
            max_bond=max_bond,
            cutoff=cutoff,
            absorb='right',
            bond_ind=bix,
            get='tensors',
            **compress_opts,
        )
        sVH.drop_tags()
        Us.append(U)
        #              i
        #     │ │ │ │  │    │ │ │ │ │ │
        #     ▶═▶═▶═▶══▶═□──◀─◀─◀─◀─◀─◀
        #                 ╲ │ │ │ │ │ │
        #              : :  ◀─◀─◀─◀─◀─◀
        #              U sVH

    Us.append((sVH | tn.select(tn.L - 1)).contract(optimize=optimize))

    new = TensorNetwork(Us)
    # cast as whatever the input was e.g. MPS
    new.view_like_(mps)
    # this puts the array indices in canonical order
    new.permute_arrays()

    return new
