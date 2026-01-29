import importlib

import numpy as np
import pytest

import quimb.tensor as qtn

requires_autograd = pytest.mark.skipif(
    importlib.util.find_spec("autograd") is None,
    reason="autograd not installed",
)


@pytest.mark.parametrize("method", ("auto", "dense", "overlap"))
@pytest.mark.parametrize(
    "normalized",
    (
        True,
        False,
        "squared",
        "infidelity",
        "infidelity_sqrt",
    ),
)
def test_tensor_network_distance(method, normalized):
    n = 6
    A = qtn.TN_rand_reg(n=n, reg=3, D=2, phys_dim=2, dtype=complex)
    Ad = A.to_dense([f"k{i}" for i in range(n)])
    B = qtn.TN_rand_reg(n=6, reg=3, D=2, phys_dim=2, dtype=complex)
    Bd = B.to_dense([f"k{i}" for i in range(n)])
    d1 = np.linalg.norm(Ad - Bd)
    d2 = A.distance(B, method=method, normalized=normalized)
    if normalized:
        assert 0 <= d2 <= 2
    else:
        assert d1 == pytest.approx(d2)


@pytest.mark.parametrize("method", ("auto", "dense", "overlap"))
@pytest.mark.parametrize(
    "normalized",
    (
        True,
        False,
        "squared",
        "infidelity",
        "infidelity_sqrt",
    ),
)
def test_distance_contract_structured_with_exponents(method, normalized):
    a = qtn.MPS_rand_state(5, 3, seed=42)
    b = qtn.MPS_rand_state(5, 3, seed=43)
    dex = a.distance(b, method=method, normalized=normalized)
    a.equalize_norms_(1.0)
    assert a.exponent != 0.0
    b.equalize_norms_(1.0)
    assert b.exponent != 0.0
    d = a.distance(b, method=method, normalized=normalized)
    assert d == pytest.approx(dex)


@pytest.mark.parametrize(
    "opts",
    [
        dict(method="als", dense_solve=False),
        dict(method="als", dense_solve=False, solver="lgmres"),
        dict(
            method="als",
            dense_solve=True,
            enforce_pos=False,
            solver_dense="lstsq",
        ),
        dict(method="als", dense_solve=True, enforce_pos=True),
        dict(method="tree"),
        pytest.param(
            dict(method="autodiff", distance_method="dense"),
            marks=requires_autograd,
        ),
        pytest.param(
            dict(method="autodiff", distance_method="overlap"),
            marks=requires_autograd,
        ),
    ],
)
@pytest.mark.parametrize("dtype", ("float64", "complex128"))
def test_fit_mps(opts, dtype):
    k1 = qtn.MPS_rand_state(5, 3, seed=666, dtype=dtype)
    k2 = qtn.MPS_rand_state(5, 3, seed=667, dtype=dtype)
    assert k1.distance_normalized(k2) > 1e-3
    k1.fit_(k2, progbar=True, **dict(opts))
    assert k1.distance_normalized(k2) < 1e-3


@pytest.mark.parametrize(
    "opts",
    [
        dict(method="als", dense_solve=False),
        dict(method="als", dense_solve=False, solver="lgmres"),
        dict(
            method="als",
            dense_solve=True,
            enforce_pos=False,
            solver_dense="lstsq",
        ),
        dict(method="als", dense_solve=True, enforce_pos=True),
        pytest.param(
            dict(method="autodiff", distance_method="dense"),
            marks=requires_autograd,
        ),
        pytest.param(
            dict(method="autodiff", distance_method="overlap"),
            marks=requires_autograd,
        ),
    ],
)
@pytest.mark.parametrize("dtype", ("float64", "complex128"))
def test_fit_rand_reg(opts, dtype):
    r1 = qtn.TN_rand_reg(5, 4, D=2, seed=666, phys_dim=2, dtype=dtype)
    k2 = qtn.MPS_rand_state(5, 3, seed=667, dtype=dtype)
    assert r1.distance(k2) > 1e-3
    r1.fit_(k2, progbar=True, **dict(opts))
    assert r1.distance(k2) < 1e-3


@pytest.mark.parametrize(
    "opts",
    [
        dict(method="als", dense_solve=False),
        dict(method="als", dense_solve=False, solver="lgmres"),
        dict(
            method="als",
            dense_solve=True,
            enforce_pos=False,
            solver_dense="lstsq",
        ),
        dict(method="als", dense_solve=True, enforce_pos=True),
        dict(method="tree"),
        pytest.param(
            dict(method="autodiff", distance_method="dense"),
            marks=requires_autograd,
        ),
        pytest.param(
            dict(method="autodiff", distance_method="overlap"),
            marks=requires_autograd,
        ),
    ],
)
@pytest.mark.parametrize("dtype", ("float64", "complex128"))
def test_fit_partial_tags(opts, dtype):
    k1 = qtn.MPS_rand_state(5, 3, seed=666, dtype=dtype)
    k2 = qtn.MPS_rand_state(5, 3, seed=667, dtype=dtype)
    d0 = k1.distance(k2)
    tags = ["I0", "I2", "I4"]
    k1f = k1.fit(k2, tol=1e-3, tags=tags, progbar=True, **dict(opts))
    assert k1f.distance(k2) < d0
    if opts["method"] != "tree":
        assert (k1f[0] - k1[0]).norm() > 1e-12
        assert (k1f[1] - k1[1]).norm() < 1e-12
        assert (k1f[2] - k1[2]).norm() > 1e-12
        assert (k1f[3] - k1[3]).norm() < 1e-12
        assert (k1f[4] - k1[4]).norm() > 1e-12


def test_hyper_distance():
    tna = qtn.HTN_rand(10, 3, 2, 2, 2, 2, 2, seed=0, dtype="complex128")
    tnb = qtn.HTN_rand(10, 3, 2, 2, 2, 2, 2, seed=1, dtype="complex128")
    oix = [f"k{i}" for i in range(4)]
    xa = tna.to_dense(oix)
    xb = tnb.to_dense(oix)
    assert tna.norm(output_inds=oix) == pytest.approx(np.linalg.norm(xa))
    assert tnb.norm(output_inds=oix) == pytest.approx(np.linalg.norm(xb))
    assert tnb.overlap(tna, output_inds=oix) == pytest.approx(np.vdot(xa, xb))
    assert tna.distance(tnb, output_inds=oix) == pytest.approx(
        np.linalg.norm(xa - xb)
    )
