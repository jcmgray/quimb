import pytest

import quimb as qu
import quimb.tensor as qtn
import quimb.tensor.belief_propagation as qbp


@pytest.mark.parametrize("local_convergence", [False, True])
def test_contract_tree_exact(local_convergence):
    tn = qtn.TN_rand_tree(20, 3)
    Z = tn.contract()
    info = {}
    Z_bp = qbp.contract_d1bp(
        tn, info=info, local_convergence=local_convergence, progbar=True
    )
    assert info["converged"]
    assert Z == pytest.approx(Z_bp, rel=1e-12)


@pytest.mark.parametrize("damping", [0.0, 0.1])
@pytest.mark.parametrize("diis", [False, True])
def test_contract_normal(damping, diis):
    tn = qtn.TN2D_from_fill_fn(lambda s: qu.randn(s, dist="uniform"), 6, 6, 2)
    Z = tn.contract()
    info = {}
    Z_bp = qbp.contract_d1bp(
        tn, damping=damping, diis=diis, info=info, progbar=True
    )
    assert info["converged"]
    assert Z == pytest.approx(Z_bp, rel=1e-1)


def test_get_gauged_tn():
    tn = qtn.TN2D_from_fill_fn(lambda s: qu.randn(s, dist="uniform"), 6, 6, 2)
    Z = tn.contract()
    bp = qbp.D1BP(tn)
    bp.run()
    Zbp = bp.contract()
    assert Z == pytest.approx(Zbp, rel=1e-1)
    tn_gauged = bp.get_gauged_tn()
    Zg = qu.prod(array.item(0) for array in tn_gauged.arrays)
    assert Z == pytest.approx(Zg, rel=1e-1)
