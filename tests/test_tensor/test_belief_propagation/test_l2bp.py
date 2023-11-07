import pytest

import quimb.tensor as qtn
from quimb.experimental.belief_propagation.l2bp import (
    contract_l2bp,
    compress_l2bp,
)


@pytest.mark.parametrize("dtype", ["float32", "complex64"])
def test_contract_tree_exact(dtype):
    psi = qtn.TN_rand_tree(20, 3, 2, dtype=dtype)
    norm2 = psi.H @ psi
    info = {}
    norm2_bp = contract_l2bp(psi, info=info, progbar=True)
    assert info["converged"]
    assert norm2_bp == pytest.approx(norm2, rel=5e-6)


@pytest.mark.parametrize("dtype", ["float32", "complex64"])
def test_contract_loopy_approx(dtype):
    peps = qtn.PEPS.rand(3, 4, 3, dtype=dtype, seed=42)
    norm_ex = peps.H @ peps
    info = {}
    norm_bp = contract_l2bp(peps, damping=0.1, info=info, progbar=True)
    assert info["converged"]
    assert norm_bp == pytest.approx(norm_ex, rel=0.2)


@pytest.mark.parametrize("damping", [0.0, 0.1])
@pytest.mark.parametrize("dtype", ["float32", "complex64"])
def test_compress_loopy(damping, dtype):
    peps = qtn.PEPS.rand(3, 4, 3, seed=42, dtype=dtype)
    # test that using the BP compression gives better fidelity than purely
    # local, naive compression scheme
    peps_c1 = peps.compress_all(max_bond=2)
    info = {}
    peps_c2 = compress_l2bp(
        peps, max_bond=2, damping=damping, info=info, progbar=True
    )
    assert info["converged"]
    fid1 = peps_c1.H @ peps_c2
    fid2 = peps_c2.H @ peps_c2
    assert abs(fid2) > abs(fid1)


@pytest.mark.parametrize("dtype", ["float32", "complex64"])
def test_contract_double_layer_tree_exact(dtype):
    # generate a random binary tree
    edges = qtn.edges_tree_rand(10, max_degree=3, seed=42)
    # generate a random tree product state and operator on this tree
    tps = qtn.TN_from_edges_rand(
        edges, 3, phys_dim=2, site_ind_id="k{}", dtype=dtype
    )
    tpo = qtn.TN_from_edges_rand(
        edges, 3, phys_dim=2, site_ind_id=("k{}", "b{}"), dtype=dtype
    )
    # join into double layer tree
    tn = qtn.tensor_network_apply_op_vec(tpo, tps, contract=False)
    assert tn.num_tensors == 20

    norm_ex = tn.H @ tn
    info = {}
    norm_bp = contract_l2bp(tn, info=info, progbar=True)
    assert info["converged"]

    assert norm_bp == pytest.approx(norm_ex, rel=1e-6)


@pytest.mark.parametrize("dtype", ["float32", "complex64"])
@pytest.mark.parametrize("damping", [0.0, 0.1])
@pytest.mark.parametrize("update", ["parallel", "sequential"])
def test_compress_double_layer_loopy(dtype, damping, update):
    peps = qtn.PEPS.rand(3, 4, bond_dim=3, seed=42, dtype=dtype)
    pepo = qtn.PEPO.rand(3, 4, bond_dim=2, seed=42, dtype=dtype)

    tn_lazy = qtn.tensor_network_apply_op_vec(pepo, peps, contract=False)
    assert tn_lazy.num_tensors == 24

    # compress using basic local compression
    tn_eager = qtn.tensor_network_apply_op_vec(pepo, peps, contract=True)
    assert tn_eager.num_tensors == 12
    tn_eager.compress_all_(max_bond=3)
    fid_basic = abs(tn_eager.H @ tn_lazy)

    # compress using BP
    info = {}
    tn_bp = compress_l2bp(
        tn_lazy,
        max_bond=3,
        damping=damping,
        update=update,
        info=info,
        progbar=True,
    )
    assert info["converged"]
    assert tn_bp.num_tensors == 12

    # assert we did better than basic local compression
    fid_bp = abs(tn_bp.H @ tn_lazy)
    assert fid_bp > fid_basic
