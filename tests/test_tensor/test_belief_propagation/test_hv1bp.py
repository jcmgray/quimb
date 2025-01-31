import pytest

import quimb as qu
import quimb.tensor as qtn
import quimb.tensor.belief_propagation as qbp


@pytest.mark.parametrize("damping", [0.0, 0.1, 0.5])
@pytest.mark.parametrize("diis", [False, True])
def test_contract_hyper(damping, diis):
    htn = qtn.HTN_random_ksat(3, 50, alpha=2.0, seed=42, mode="dense")
    info = {}
    num_solutions = qbp.contract_hv1bp(
        htn, damping=damping, diis=diis, info=info, progbar=True
    )
    assert info["converged"]
    assert num_solutions == pytest.approx(309273226, rel=0.1)


@pytest.mark.parametrize("messages", [None, "dense", "random"])
def test_contract_tree_exact(messages):
    tn = qtn.TN_rand_tree(20, 3)
    Z = tn.contract()
    info = {}

    if messages == "random":

        def messages(shape):
            return qu.randn(shape, dist="uniform")

    Z_bp = qbp.contract_hv1bp(tn, messages=messages, info=info, progbar=True)
    assert info["converged"]
    assert Z == pytest.approx(Z_bp, rel=1e-12)


@pytest.mark.parametrize("damping", [0.0, 0.1, 0.5])
@pytest.mark.parametrize("diis", [False, True])
def test_contract_normal(damping, diis):
    tn = qtn.TN2D_from_fill_fn(lambda s: qu.randn(s, dist="uniform"), 6, 6, 2)
    Z = tn.contract()
    info = {}
    Z_bp = qbp.contract_hv1bp(
        tn, damping=damping, diis=diis, info=info, progbar=True
    )
    assert info["converged"]
    assert Z == pytest.approx(Z_bp, rel=1e-1)


@pytest.mark.parametrize("damping", [0.0, 0.1])
def test_sample(damping):
    nvars = 20
    htn = qtn.HTN_random_ksat(3, nvars, alpha=2.0, seed=42, mode="dense")
    config, tn_config, omega = qbp.sample_hv1bp(
        htn, damping=damping, seed=42, progbar=True
    )
    assert len(config) == nvars
    assert tn_config.num_indices == 0
    assert tn_config.contract() == pytest.approx(1.0)
    assert 0.0 < omega < 1.0
