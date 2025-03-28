import numpy as np
import pytest

import quimb.tensor as qtn
from quimb.tensor.contraction import _CONTRACT_BACKEND, _TENSOR_LINOP_BACKEND


def test_tensor_contract_strip_exponent():
    tn = qtn.TN_rand_reg(10, 3, 3, dtype=complex)
    z0 = tn.contract()
    m1, e1 = qtn.tensor_contract(*tn, strip_exponent=True)
    assert m1 * 10**e1 == pytest.approx(z0)
    # test tn.exponent is reinserted
    tn.equalize_norms_(value=1.0)
    z2 = tn.contract()
    assert z2 == pytest.approx(z0)
    # test tn.exponent is reinserted with strip exponent
    m3, e3 = tn.contract(strip_exponent=True)
    assert m3 * 10**e3 == pytest.approx(z0)


@pytest.mark.parametrize("strip_exponent", [False, True])
@pytest.mark.parametrize("equalize_norms", [False, 1.0, True])
@pytest.mark.parametrize("inplace", [False, True])
def test_contract_tags_strip_exponent(
    strip_exponent,
    equalize_norms,
    inplace,
):
    tn = qtn.TN_rand_reg(8, 3, 2)
    Zex = tn.contract()

    if inplace:
        tnc = tn.copy()
        tnc.contract_tags_(
            all,
            strip_exponent=strip_exponent,
            equalize_norms=equalize_norms,
        )
        Z = tnc.arrays[0] * 10**tnc.exponent

    else:
        Z = tn.contract_tags(
            all,
            strip_exponent=strip_exponent,
            equalize_norms=equalize_norms,
        )
        if strip_exponent:
            Z = Z[0] * 10 ** Z[1]

    assert Z == pytest.approx(Zex, rel=1e-3)


@pytest.mark.parametrize("strip_exponent", [False, True])
@pytest.mark.parametrize("equalize_norms", [False, 1.0, True])
@pytest.mark.parametrize("inplace", [False, True])
def test_contract_cumulative_strip_exponent(
    strip_exponent,
    equalize_norms,
    inplace,
):
    mps = qtn.MPS_rand_state(7, 3)
    tn = mps.make_norm()
    Zex = tn.contract()

    assert tn._CONTRACT_STRUCTURED

    if inplace:
        tnc = tn.copy()
        tnc.contract_(
            ...,
            strip_exponent=strip_exponent,
            equalize_norms=equalize_norms,
        )
        Z = tnc.arrays[0] * 10**tnc.exponent

    else:
        Z = tn.contract(
            ...,
            strip_exponent=strip_exponent,
            equalize_norms=equalize_norms,
        )
        if strip_exponent:
            Z = Z[0] * 10 ** Z[1]

    assert Z == pytest.approx(Zex, rel=1e-3)


@pytest.mark.parametrize("strip_exponent", [False, True])
@pytest.mark.parametrize("equalize_norms", [False, 1.0, True])
@pytest.mark.parametrize("inplace", [False, True])
def test_contract_compressed_strip_exponent(
    strip_exponent,
    equalize_norms,
    inplace,
):
    L = 6
    tn = qtn.TN2D_rand(L, L, 2, seed=42, dist="uniform")
    Zex = tn.contract()

    if inplace:
        tnc = tn.copy()
        tnc.contract_(
            optimize="greedy-compressed",
            max_bond=4,
            strip_exponent=strip_exponent,
            equalize_norms=equalize_norms,
            progbar=True,
        )
        Z = tnc.arrays[0] * 10**tnc.exponent
    else:
        Z = tn.contract(
            optimize="greedy-compressed",
            max_bond=4,
            strip_exponent=strip_exponent,
            equalize_norms=equalize_norms,
            progbar=True,
        )
        if strip_exponent:
            Z = Z[0] * 10 ** Z[1]

        if equalize_norms == 1.0:
            # undefined
            return

    assert Z == pytest.approx(Zex, rel=1e-3)


class TestContractOpts:
    def test_contract_strategy(self):
        assert qtn.get_contract_strategy() == "greedy"
        with qtn.contract_strategy("auto"):
            assert qtn.get_contract_strategy() == "auto"
        assert qtn.get_contract_strategy() == "greedy"

    def test_contract_backend(self):
        assert qtn.get_contract_backend() == _CONTRACT_BACKEND
        with qtn.contract_backend("cupy"):
            assert qtn.get_contract_backend() == "cupy"
        assert qtn.get_contract_backend() == _CONTRACT_BACKEND

    def test_tensor_linop_backend(self):
        assert qtn.get_tensor_linop_backend() == _TENSOR_LINOP_BACKEND
        with qtn.tensor_linop_backend("cupy"):
            assert qtn.get_tensor_linop_backend() == "cupy"
        assert qtn.get_tensor_linop_backend() == _TENSOR_LINOP_BACKEND

    def test_contract_cache(self):
        import cotengra as ctg

        info = {"num_calls": 0}

        def my_custom_opt(inputs, output, size_dict, memory_limit=None):
            info["num_calls"] += 1
            return [(0, 1)] * (len(inputs) - 1)

        ctg.register_preset("quimb_test_opt", my_custom_opt)

        tn = qtn.MPS_rand_state(4, 3) & qtn.MPS_rand_state(4, 3)
        assert tn.contract(
            all, optimize="quimb_test_opt", get="expression"
        ) is tn.contract(all, optimize="quimb_test_opt", get="expression")
        assert info["num_calls"] == 1

        assert info["num_calls"] == 1


@pytest.mark.parametrize("around", ["I3,3", "I0,0", "I1,2"])
@pytest.mark.parametrize("equalize_norms", [False, True])
@pytest.mark.parametrize("gauge_boundary_only", [False, True])
def test_contract_approx_with_gauges(
    around, equalize_norms, gauge_boundary_only
):
    rng = np.random.default_rng(42)
    tn = qtn.TN2D_from_fill_fn(
        lambda shape: rng.uniform(size=shape, low=-0.5), 7, 7, 4
    )
    Zex = tn ^ ...
    Z = tn.contract_around(
        around,
        max_bond=8,
        gauges=True,
        gauge_boundary_only=gauge_boundary_only,
        tree_gauge_distance=2,
        equalize_norms=equalize_norms,
        progbar=True,
    )
    assert Z == pytest.approx(Zex, rel=1e-2)
