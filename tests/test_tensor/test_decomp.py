import numpy as np
import pytest
from numpy.testing import assert_allclose

import quimb as qu


def test_sgn_convention():
    from quimb.tensor.decomp import sgn

    assert sgn(1) == 1
    assert sgn(2.0) == 1
    assert sgn(-1) == -1
    assert sgn(-2.0) == -1
    assert sgn(0) == 1
    assert sgn(0.0) == 1
    assert sgn(0.0 + 0.0j) == 1
    assert sgn(1.0 + 2.0j) != 1
    assert sgn(1.0 + 2.0j) != -1
    assert abs(sgn(1.0 + 2.0j)) == pytest.approx(1)


@pytest.mark.parametrize(
    "dtype", ["float64", "float32", "complex128", "complex64"]
)
def test_qr_stabilized_sign_bug(dtype):
    from quimb.tensor.decomp import qr_stabilized

    for _ in range(10):
        Q = qu.rand_uni(4, dtype=dtype)
        R = qu.rand_matrix(4, dtype=dtype)

        # make R strictly upper triangular
        ii, jj = np.indices(R.shape)
        R[ii >= jj] = 0.0

        X = Q @ R
        Q2, _, R2 = qr_stabilized(X)

        assert abs(np.linalg.norm((Q2 @ R2) - X)) < (
            1e-12 if dtype in ("float64", "complex128") else 1e-6
        )


@pytest.mark.parametrize("da", [5, 7])
@pytest.mark.parametrize("db", [5, 7])
@pytest.mark.parametrize("k", [-1, 6, 8])
@pytest.mark.parametrize(
    "absorb", ["U", "s", "VH", "Us", "sVH", "U,s,VH", "U,sVH", "Us,VH"]
)
@pytest.mark.parametrize("descending", [True, False])
def test_decomp_svd_via_eig(
    da,
    db,
    k,
    absorb,
    descending,
):
    from quimb.tensor.decomp import svd_via_eig

    # turn test-case into reproducible seed
    seed = qu.utils.hash_kwargs_to_int(
        da=da, db=db, k=k, absorb=absorb, descending=descending
    )

    rng = np.random.default_rng(seed)
    x = rng.uniform(size=(da, db))
    x /= np.linalg.norm(x)

    Ux, sx, VHx = np.linalg.svd(x, full_matrices=False)

    U, s, VH = svd_via_eig(x, max_bond=k, absorb=absorb, descending=descending)
    if 0 < k < min(da, db):
        sx = sx[:k]
        Ux = Ux[:, :k]
        VHx = VHx[:k, :]
    if not descending:
        sx = sx[::-1]
        Ux = Ux[:, ::-1]
        VHx = VHx[::-1, :]

    if absorb in ("U", "U,s,VH", "U,sVH"):
        assert U is not None
        assert U.shape == (da, min(da, db, k) if k > 0 else min(da, db))
        Udag = np.conj(np.transpose(U))
        assert_allclose(Udag @ U, np.eye(U.shape[1]), atol=1e-9)
        assert_allclose(np.abs(Udag @ Ux), np.eye(U.shape[1]), atol=1e-9)

    if absorb in ("s", "U,s,VH"):
        assert s is not None
        assert s.shape == (min(da, db, k) if k > 0 else min(da, db),)
        assert_allclose(s, sx, atol=1e-9)
        if descending:
            assert np.all(s[:-1] >= s[1:])
        else:
            assert np.all(s[:-1] <= s[1:])

    if absorb in ("VH", "Us,VH", "U,s,VH"):
        assert VH is not None
        assert VH.shape == (min(da, db, k) if k > 0 else min(da, db), db)
        V = np.conj(np.transpose(VH))
        assert_allclose(VH @ V, np.eye(VH.shape[0]), atol=1e-9)
        assert_allclose(np.abs(VHx @ V), np.eye(VH.shape[0]), atol=1e-9)

    if absorb in ("Us", "Us,VH"):
        assert_allclose((U.conj().T @ U), np.diag(sx**2), atol=1e-9)

    if absorb in ("sVH", "U,sVH"):
        assert_allclose((VH @ VH.conj().T), np.diag(sx**2), atol=1e-9)

    if absorb in ("Us,VH", "U,sVH", "U,s,VH"):
        if absorb == "U,s,VH":
            U = U @ np.diag(s)
        if k > min(da, db):
            assert_allclose(U @ VH, x, atol=1e-9)
        else:
            # low rank approx
            assert np.linalg.norm(x - (U @ VH)) < 0.2
