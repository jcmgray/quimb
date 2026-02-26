import numpy as np
import pytest
from numpy.testing import assert_allclose

import quimb as qu
from quimb.tensor.decomp import _compute_number_svals_to_keep_numba


def test_trim_singular_vals():
    s = np.array([3.0, 2.0, 1.0, 0.1])
    assert _compute_number_svals_to_keep_numba(s, 0.5, 1) == 3
    assert _compute_number_svals_to_keep_numba(s, 0.5, 2) == 2
    assert _compute_number_svals_to_keep_numba(s, 2, 3) == 2
    assert _compute_number_svals_to_keep_numba(s, 5.02, 3) == 1


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


@pytest.mark.parametrize("da", [5, 7])
@pytest.mark.parametrize("db", [5, 7])
@pytest.mark.parametrize("k", [-1, 4])
@pytest.mark.parametrize(
    "absorb", ["U,s,VH", "U,sVH", "Us,VH", "U", "VH", "Us", "sVH", "s"]
)
def test_svd_rand(da, db, k, absorb):
    from quimb.tensor.decomp import svd_rand_truncated

    seed = qu.utils.hash_kwargs_to_int(da=da, db=db, k=k, absorb=absorb)
    rng = np.random.default_rng(seed)
    x = rng.uniform(size=(da, db))
    x /= np.linalg.norm(x)

    rank = min(da, db) if k < 0 else min(da, db, k)

    U, s, VH = svd_rand_truncated(x, absorb=absorb, max_bond=k, seed=seed + 1)

    # left isometric factor (U is column-orthonormal)
    if absorb in ("U,s,VH", "U,sVH", "U"):
        assert U is not None
        assert U.shape == (da, rank)
        assert_allclose(U.conj().T @ U, np.eye(rank), atol=1e-9)

    # right isometric factor (VH is row-orthonormal)
    if absorb in ("U,s,VH", "Us,VH", "VH"):
        assert VH is not None
        assert VH.shape == (rank, db)
        assert_allclose(VH @ VH.conj().T, np.eye(rank), atol=1e-9)

    # singular values present and non-negative, descending
    if absorb in ("U,s,VH", "s"):
        assert s is not None
        assert s.shape == (rank,)
        assert np.all(s >= 0)
        if rank > 1:
            assert np.all(s[:-1] >= s[1:])

    # shape-only: s absorbed into left, right not returned
    if absorb == "Us":
        assert U is not None and U.shape == (da, rank)
        assert s is None and VH is None

    # shape-only: s absorbed into right, left not returned
    if absorb == "sVH":
        assert VH is not None and VH.shape == (rank, db)
        assert s is None and U is None

    # absorbed-s factor shapes (non-isometric side)
    if absorb == "Us,VH":
        assert U is not None and U.shape == (da, rank)
    if absorb == "U,sVH":
        assert VH is not None and VH.shape == (rank, db)

    # reconstruction
    if absorb == "U,s,VH":
        assert np.linalg.norm(x - U @ np.diag(s) @ VH) < 0.5
    if absorb in ("Us,VH", "U,sVH"):
        assert U is not None and VH is not None
        assert np.linalg.norm(x - U @ VH) < 0.5


def test_svd_rand_seed_reproducible():
    from quimb.tensor.decomp import svd_rand_truncated

    rng = np.random.default_rng(0)
    x = rng.normal(size=(6, 8))
    x /= np.linalg.norm(x)

    # same seed -> identical outputs
    U1, s1, VH1 = svd_rand_truncated(x, absorb=None, max_bond=4, seed=42)
    U2, s2, VH2 = svd_rand_truncated(x, absorb=None, max_bond=4, seed=42)
    assert_allclose(s1, s2)
    assert_allclose(U1, U2)
    assert_allclose(VH1, VH2)

    # different seeds -> different sketch -> different U/VH (no power iterations)
    U3, _, _ = svd_rand_truncated(
        x, absorb=None, max_bond=4, seed=42, num_iterations=0
    )
    U4, _, _ = svd_rand_truncated(
        x, absorb=None, max_bond=4, seed=99, num_iterations=0
    )
    assert not np.allclose(U3, U4)


@pytest.mark.parametrize("da,db", [(4, 8), (8, 4), (6, 6)])
@pytest.mark.parametrize("right", [True, False, None])
def test_svd_rand_right_param(right, da, db):
    from quimb.tensor.decomp import svd_rand_truncated

    seed = qu.utils.hash_kwargs_to_int(right=right, da=da, db=db)
    rng = np.random.default_rng(seed)
    x = rng.uniform(size=(da, db))
    x /= np.linalg.norm(x)

    rank = 3
    U, s, VH = svd_rand_truncated(
        x, absorb=None, max_bond=rank, right=right, seed=seed + 1
    )

    assert U.shape == (da, rank)
    assert s.shape == (rank,)
    assert VH.shape == (rank, db)
    assert np.all(s >= 0)
    assert_allclose(U.conj().T @ U, np.eye(rank), atol=1e-9)
    assert_allclose(VH @ VH.conj().T, np.eye(rank), atol=1e-9)
    assert np.linalg.norm(x - U @ np.diag(s) @ VH) < 0.5


def test_svd_rand_truncated_warns():
    from quimb.tensor.decomp import svd_rand_truncated

    rng = np.random.default_rng(0)
    x = rng.uniform(size=(6, 6))
    with pytest.warns(UserWarning, match="inefficient"):
        svd_rand_truncated(x, max_bond=None)


def test_svd_rand_via_array_split():
    from quimb.tensor.decomp import array_split

    rng = np.random.default_rng(42)
    x = rng.uniform(size=(8, 5))
    x /= np.linalg.norm(x)

    U, s, VH = array_split(x, method="svd:rand", absorb=None, max_bond=4)

    assert U.shape == (8, 4)
    assert s.shape == (4,)
    assert VH.shape == (4, 5)
    assert np.all(s >= 0)
    assert np.linalg.norm(x - U @ np.diag(s) @ VH) < 0.5
