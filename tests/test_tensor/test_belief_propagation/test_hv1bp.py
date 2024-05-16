import pytest

import quimb as qu
import quimb.tensor as qtn
from quimb.experimental.belief_propagation.hv1bp import (
    contract_hv1bp,
    sample_hv1bp,
)


@pytest.mark.parametrize("damping", [0.0, 0.1, 0.5])
def test_contract_hyper(damping):
    htn = qtn.HTN_random_ksat(3, 50, alpha=2.0, seed=42, mode="dense")
    info = {}
    num_solutions = contract_hv1bp(
        htn, damping=damping, info=info, progbar=True
    )
    assert info["converged"]
    assert num_solutions == pytest.approx(309273226, rel=0.1)


def test_contract_tree_exact():
    tn = qtn.TN_rand_tree(20, 3)
    Z = tn.contract()
    info = {}
    Z_bp = contract_hv1bp(tn, info=info, progbar=True)
    assert info["converged"]
    assert Z == pytest.approx(Z_bp, rel=1e-12)


@pytest.mark.parametrize("damping", [0.0, 0.1, 0.5])
def test_contract_normal(damping):
    tn = qtn.TN2D_from_fill_fn(lambda s: qu.randn(s, dist="uniform"), 6, 6, 2)
    Z = tn.contract()
    info = {}
    Z_bp = contract_hv1bp(tn, damping=damping, info=info, progbar=True)
    assert info["converged"]
    assert Z == pytest.approx(Z_bp, rel=1e-1)


@pytest.mark.parametrize("damping", [0.0, 0.1])
def test_sample(damping):
    nvars = 20
    htn = qtn.HTN_random_ksat(3, nvars, alpha=2.0, seed=42, mode="dense")
    config, tn_config, omega = sample_hv1bp(
        htn, damping=damping, seed=42, progbar=True
    )
    assert len(config) == nvars
    assert tn_config.num_indices == 0
    assert tn_config.contract() == pytest.approx(1.0)
    assert 0.0 < omega < 1.0
