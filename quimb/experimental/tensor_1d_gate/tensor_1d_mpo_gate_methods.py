"""Methods for compressing any 1D-like tensor networks and via these acting
with an MPO on an MPS.


- [x] the bi-directional density matrix method

"""

from autoray import do

from quimb.tensor import decomp
from quimb.tensor.tensor_core import (
    Tensor,
    rand_uuid,
    tensor_contract,
)
from quimb.tensor.tn1d.compress import mps_gate_with_mpo_lazy


def tensor_1d_compress_bidm(
    self,
    max_bond=None,
    cutoff=1e-10,
    optimize="auto-hq",
    normalize=False,
    inplace=False,
    **compress_opts,
):
    # XXX: broken for complex tensors
    # XXX: less accurate that direct/dm: simulteneous optimal projectors for
    # uncompressed state compound to be less accuracte that sequence of optimal
    # projectors for compressed state
    tn = self if inplace else self.copy()

    # form norm tensor network, with explicit index mangling
    ket = tn.copy()
    bra = ket.H
    bra.mangle_inner_()
    norm = bra & ket

    left_env = None
    info = {}
    reduced_factors = {}

    # compute left environments and reduced factors
    for i in range(1, tn.L):
        if left_env is None:
            left_env_tensors = (*norm.select(0),)
        else:
            left_env_tensors = (left_env, *norm.select(i - 1))

        left_env = t = tensor_contract(
            *left_env_tensors,
            optimize=optimize,
            drop_tags=True,
        )

        lix = t.inds[: t.ndim // 2]
        rix = t.inds[t.ndim // 2 :]
        XX = t.to_dense(lix, rix)
        R = decomp.squared_op_to_reduced_factor(XX, 2**i, XX.shape[1])
        reduced_factors[i - 1, i, "L"] = R
        info[i - 1, i] = t.shape[t.ndim // 2 :], rix

    right_env = None
    for i in range(tn.L - 2, -1, -1):
        if right_env is None:
            right_env_tensors = (*norm.select(tn.L - 1),)
        else:
            right_env_tensors = (right_env, *norm.select(i + 1))

        right_env = t = tensor_contract(
            *right_env_tensors,
            optimize=optimize,
            drop_tags=True,
        )
        lix = t.inds[: t.ndim // 2]
        rix = t.inds[t.ndim // 2 :]
        XX = t.to_dense(lix, rix)
        R = decomp.squared_op_to_reduced_factor(
            XX, XX.shape[0], 2 ** (tn.L - i - 1), False
        )
        reduced_factors[i, i + 1, "R"] = R

    for i in range(tn.L - 1):
        Rl = reduced_factors.pop((i, i + 1, "L"))
        Rr = reduced_factors.pop((i, i + 1, "R"))

        Pl, Pr = decomp.compute_oblique_projectors(
            Rl,
            Rr,
            max_bond=max_bond,
            cutoff=cutoff,
            **compress_opts,
        )

        bix_sizes, bix = info[i, i + 1]

        Pl = do("reshape", Pl, (*bix_sizes, -1))
        Pr = do("reshape", Pr, (-1, *bix_sizes))

        ltn = tn.select(i)
        rtn = tn.select(i + 1)

        # finally cut the bonds
        new_lix = [rand_uuid() for _ in bix]
        new_rix = [rand_uuid() for _ in bix]
        new_bix = [rand_uuid()]
        ltn.reindex_(dict(zip(bix, new_lix)))
        rtn.reindex_(dict(zip(bix, new_rix)))

        # ... and insert the new projectors in place
        new_ltags = [tn.site_tag(i)]
        new_rtags = [tn.site_tag(i + 1)]
        tn |= Tensor(Pl, inds=new_lix + new_bix, tags=new_ltags)
        tn |= Tensor(Pr, inds=new_bix + new_rix, tags=new_rtags)

    for i in range(tn.L):
        tn ^= tn.site_tag(i)

    return tn


def mps_gate_with_mpo_bidm(mps, mpo, max_bond=None, cutoff=1e-10):
    tn = mps_gate_with_mpo_lazy(mps, mpo)
    return tensor_1d_compress_bidm(
        tn,
        max_bond=max_bond,
        cutoff=cutoff,
        inplace=True,
    )
