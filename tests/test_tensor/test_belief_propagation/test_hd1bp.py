import pytest

import quimb as qu
import quimb.tensor as qtn
import quimb.tensor.belief_propagation as qbp


@pytest.mark.parametrize("damping", [0.0, 0.1])
def test_contract_hyper(damping):
    htn = qtn.HTN_random_ksat(3, 50, alpha=2.0, seed=42, mode="dense")
    info = {}
    num_solutions = qbp.contract_hd1bp(
        htn, damping=damping, info=info, progbar=True
    )
    assert info["converged"]
    assert num_solutions == pytest.approx(309273226, rel=0.1)


@pytest.mark.parametrize("normalize", ["L1", "L2", "Linf"])
def test_contract_tree_exact(normalize):
    tn = qtn.TN_rand_tree(20, 3)
    Z = tn.contract()
    info = {}
    Z_bp = qbp.contract_hd1bp(
        tn,
        info=info,
        normalize=normalize,
        progbar=True,
    )
    assert info["converged"]
    assert Z == pytest.approx(Z_bp, rel=1e-12)


@pytest.mark.parametrize("dtype", ["float32", "complex64"])
def test_contract_with_exponent(dtype):
    tn = qtn.TN_rand_tree(10, 3, max_degree=4, seed=42, dtype=dtype)
    Zex = tn.contract()
    tn.equalize_norms_(1.7)
    assert tn.exponent
    bp = qbp.HD1BP(tn)
    bp.run()
    assert bp.contract() == pytest.approx(Zex, rel=1e-5)


@pytest.mark.parametrize("damping", [0.0, 0.1])
@pytest.mark.parametrize("diis", [False, True])
def test_contract_normal(damping, diis):
    tn = qtn.TN2D_from_fill_fn(lambda s: qu.randn(s, dist="uniform"), 6, 6, 2)
    Z = tn.contract()
    info = {}
    Z_bp = qbp.contract_hd1bp(
        tn, damping=damping, diis=diis, info=info, progbar=True
    )
    assert info["converged"]
    assert Z == pytest.approx(Z_bp, rel=1e-1)


@pytest.mark.parametrize("damping", [0.0, 0.1])
def test_sample(damping):
    nvars = 20
    htn = qtn.HTN_random_ksat(3, nvars, alpha=2.0, seed=42, mode="dense")
    config, tn_config, omega = qbp.sample_hd1bp(
        htn, damping=damping, seed=42, progbar=True
    )
    assert len(config) == nvars
    assert tn_config.num_indices == 0
    assert tn_config.contract() == pytest.approx(1.0)
    assert 0.0 < omega < 1.0


def test_get_gauged_tn():
    tn = qtn.TN2D_from_fill_fn(lambda s: qu.randn(s, dist="uniform"), 6, 6, 2)
    Z = tn.contract()
    bp = qbp.HD1BP(tn)
    bp.run()
    Zbp = bp.contract()
    assert Z == pytest.approx(Zbp, rel=1e-1)
    tn_gauged = bp.get_gauged_tn()
    Zg = qu.prod(array.item(0) for array in tn_gauged.arrays)
    assert Z == pytest.approx(Zg, rel=1e-1)
