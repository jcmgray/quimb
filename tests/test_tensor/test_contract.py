import numpy as np
import pytest

import quimb.tensor as qtn
from quimb.tensor.contraction import _CONTRACT_BACKEND, _TENSOR_LINOP_BACKEND

from . import pytorch_case, requires_symmray


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


@pytest.mark.parametrize("insert_exponent", [False, True])
@pytest.mark.parametrize("strip_exponent", [False, True])
@pytest.mark.parametrize("equalize_norms", [False, 1.0, True])
@pytest.mark.parametrize("inplace", [False, True])
def test_contract_cumulative_strip_exponent(
    insert_exponent,
    strip_exponent,
    equalize_norms,
    inplace,
):
    mps = qtn.MPS_rand_state(7, 3)
    tn = mps.make_norm()
    Zex = tn.contract()

    if insert_exponent:
        mps.equalize_norms_(2.0)
        tn = mps.make_norm()
        assert tn.exponent != 0.0

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


def _contract_route(tn, route, **kwargs):
    if route == "all":
        return tn.contract(all, **kwargs)
    if route == "tags":
        return tn.contract_tags(all, **kwargs)
    if route == "structured":
        assert tn._CONTRACT_STRUCTURED
        return tn.contract(..., **kwargs)
    raise ValueError(route)


@pytest.mark.parametrize("route", ["all", "tags", "structured"])
@pytest.mark.parametrize("insert_exponent", [False, True])
class TestCheckZero:
    @staticmethod
    def to_backend(tn, backend):
        if backend == "torch":
            import torch

            tn.apply_to_arrays(torch.tensor)
        return tn

    @staticmethod
    def make_mps(backend):
        if backend == "symmray":
            import symmray as sr

            return sr.MPS_abelian_rand(
                "Z2", 6, bond_dim=4, subsizes="equal", seed=42
            )
        return qtn.MPS_rand_state(6, 4, seed=42)

    @pytest.mark.parametrize(
        "backend",
        [
            "numpy",
            pytorch_case,
            pytest.param("symmray", marks=requires_symmray),
        ],
    )
    def test_zero_tensor(self, route, insert_exponent, backend):
        psi = self.make_mps(backend)
        psi[2].modify(apply=lambda x: 0 * x)
        tn = self.to_backend(psi.make_norm(), backend)
        if insert_exponent:
            tn.exponent = 3.0
        m, e = _contract_route(tn, route, strip_exponent=True, check_zero=True)
        assert float(abs(m)) == 0.0
        assert float(e) == 0.0

    @pytest.mark.parametrize("backend", ["numpy", pytorch_case])
    def test_cancellation(self, route, insert_exponent, backend):
        a = qtn.MPS_product_state([np.array([1.0, 1.0])] * 4)
        b = qtn.MPS_product_state(
            [np.array([1.0, -1.0])] + [np.array([1.0, 1.0])] * 3
        )
        tn = self.to_backend(a.make_overlap(b), backend)
        if insert_exponent:
            tn.exponent = 3.0
        m, e = _contract_route(tn, route, strip_exponent=True, check_zero=True)
        assert float(abs(m)) == 0.0
        assert float(e) == 0.0

    @pytest.mark.parametrize(
        "backend",
        [
            "numpy",
            pytorch_case,
            pytest.param("symmray", marks=requires_symmray),
        ],
    )
    def test_tiny_nonzero(self, route, insert_exponent, backend):
        psi = self.make_mps(backend)
        Zex = psi.make_norm().contract()
        psi.multiply_each_(1e-40)
        tn = self.to_backend(psi.make_norm(), backend)
        if insert_exponent:
            tn.exponent = 3.0
        m, e = _contract_route(tn, route, strip_exponent=True, check_zero=True)
        # 12 tensors each scaled by 1e-40, which would underflow
        shift = 480 - 3 * insert_exponent
        assert complex(m) * 10 ** (float(e) + shift) == pytest.approx(Zex)


def test_check_zero_compressed_not_implemented():
    tn = qtn.TN2D_rand(3, 3, 2, seed=42)
    with pytest.raises(NotImplementedError):
        tn.contract(max_bond=2, strip_exponent=True, check_zero=True)


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
