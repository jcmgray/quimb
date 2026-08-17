import importlib.util

import autoray as ar
import numpy as np
import pytest
from numpy.testing import assert_allclose

import quimb as qu
from quimb.tensor.decomp import (
    _compute_number_svals_to_keep_numba,
    array_split,
    parse_split_opts,
)

from . import (
    jax_case,
    pytorch_case,
    tensorflow_case,
)


def test_parse_split_opts():
    method, opts = parse_split_opts()
    assert method == "svd"
    assert opts["absorb"] == 0
    assert opts["cutoff"] != 0.0
    assert opts["max_bond"] == -1

    method, opts = parse_split_opts("qr", "auto", max_bond=None, cutoff=0.0)
    assert method == "qr"
    assert opts["absorb"] == 1

    method, opts = parse_split_opts("auto", "right", max_bond=None, cutoff=0.0)
    assert method == "qr"
    assert opts["absorb"] == 1

    method, opts = parse_split_opts("lq", "auto", max_bond=None, cutoff=0.0)
    assert method == "qr"
    assert opts["absorb"] == -1

    method, opts = parse_split_opts("auto", "left", max_bond=None, cutoff=0.0)
    assert method == "qr"
    assert opts["absorb"] == -1

    method, opts = parse_split_opts(
        "lq:cholesky", "auto", max_bond=None, cutoff=0.0
    )
    assert method == "qr:cholesky"
    assert opts["absorb"] == -1


def test_trim_singular_vals():
    s = np.array([3.0, 2.0, 1.0, 0.1])
    assert _compute_number_svals_to_keep_numba(s, 0.5, 1) == 3
    assert _compute_number_svals_to_keep_numba(s, 0.5, 2) == 2
    assert _compute_number_svals_to_keep_numba(s, 2, 3) == 2
    assert _compute_number_svals_to_keep_numba(s, 5.02, 3) == 1


class TestSafeInverse:
    def test_matches_exact_inverse_away_from_zero(self):
        from quimb.tensor.decomp import safe_inverse

        x = np.array([3.0, 1.0, 1e-3])
        assert_allclose(safe_inverse(x), 1 / x, rtol=1e-14)
        assert_allclose(safe_inverse(x, power=0.5), x**-0.5, rtol=1e-14)

    def test_zeros_map_to_zero(self):
        from quimb.tensor.decomp import safe_inverse

        x = np.array([2.0, 0.0, 0.0])
        assert_allclose(safe_inverse(x), [0.5, 0.0, 0.0])
        assert_allclose(safe_inverse(x, power=0.5), [2**-0.5, 0.0, 0.0])
        # all zero, no scale to normalize by
        assert_allclose(safe_inverse(np.zeros(3)), np.zeros(3))
        assert_allclose(safe_inverse(np.zeros(3), power=0.5), np.zeros(3))

    @pytest.mark.parametrize("dtype", ["float32", "float64"])
    @pytest.mark.parametrize("scale", [1e-20, 1.0, 1e20])
    def test_no_overflow_at_extreme_scales(self, dtype, scale):
        # squaring x directly over- or underflows at these scales
        from quimb.tensor.decomp import safe_inverse

        x = np.array([1.0, 1e-3, 0.0], dtype=dtype) * scale
        xinv = safe_inverse(x)
        assert np.isfinite(xinv).all()
        assert_allclose(xinv[:2], 1 / x[:2], rtol=1e-5)
        xisqrt = safe_inverse(x, power=0.5)
        assert np.isfinite(xisqrt).all()
        assert_allclose(xisqrt[:2], x[:2] ** -0.5, rtol=1e-5)

    def test_batched_damps_per_vector(self):
        from quimb.tensor.decomp import safe_inverse

        # second row is tiny only relative to the first, not to itself
        x = np.array([[1.0, 1e-30], [1e-30, 1e-60]])
        xinv = safe_inverse(x)
        # damped rather than 1e30, and bounded by 1 / (2 * eps)
        assert xinv[0, 1] < 0.5 / np.finfo(x.dtype).eps
        assert xinv[1, 0] == pytest.approx(1e30)

    @pytest.mark.parametrize("cutoff", [None, 1e-8])
    def test_numba_matches_generic(self, cutoff):
        from quimb.tensor.decomp import safe_inverse, safe_inverse_numba

        rng = np.random.default_rng(42)
        x = np.sort(rng.uniform(size=(3, 4, 5)))[..., ::-1]
        x[1, 2, 3:] = 0.0
        assert_allclose(safe_inverse_numba(x, cutoff), safe_inverse(x, cutoff))

    @pytest.mark.parametrize(
        "backend", ["numpy", jax_case, tensorflow_case, pytorch_case]
    )
    def test_backends(self, backend):
        from quimb.tensor.decomp import safe_inverse

        xp = ar.get_namespace(backend)
        x = xp.asarray(np.array([2.0, 1e-3, 0.0]))
        assert_allclose(ar.to_numpy(safe_inverse(x)), [0.5, 1e3, 0.0])
        assert_allclose(
            ar.to_numpy(safe_inverse(x, power=0.5)),
            [2**-0.5, 1e3**0.5, 0.0],
            rtol=1e-5,
        )

    @pytest.mark.parametrize("backend", [pytorch_case])
    def test_backend_specific_dtype(self, backend):
        # numpy has no bfloat16, the cutoff must come from torch's finfo
        from quimb.tensor.decomp import safe_inverse

        xp = ar.get_namespace(backend)
        eps = xp.finfo(xp.bfloat16).eps
        x = xp.asarray([1.0, eps], dtype=xp.bfloat16)
        # the entry right at the cutoff is damped to half its inverse
        xinv = ar.to_numpy(safe_inverse(x).to(xp.float32))
        assert xinv[1] == pytest.approx(0.5 / eps, rel=0.05)

    def test_lazy(self):
        # lazy arrays have neither `finfo` nor `keepdims`
        from quimb.tensor.decomp import safe_inverse

        x = ar.lazy.array(np.array([2.0, 1e-3, 0.0]))
        assert_allclose(safe_inverse(x).compute(), [0.5, 1e3, 0.0])

    @pytest.mark.skipif(
        importlib.util.find_spec("symmray") is None,
        reason="symmray not installed",
    )
    @pytest.mark.parametrize("power", [1.0, 0.5])
    def test_block_sparse(self, power):
        # block sparse: one vector, no axis aware max, expand_dims or where
        import symmray as sr

        from quimb.tensor.decomp import safe_inverse

        inds = [sr.BlockIndex({0: 2, 1: 3}, dual=d) for d in (0, 1)]
        x = sr.Z2Array.random(inds, seed=42)
        _u, s, _vh = ar.do("linalg.svd", x)

        sinv = safe_inverse(s, power=power)
        assert_allclose(
            ar.to_numpy(sinv.to_dense()),
            safe_inverse(ar.to_numpy(s.to_dense()), power=power),
        )


@pytest.mark.parametrize("absorb", ["both", None, "left", "right"])
def test_oblique_projectors_exactly_low_rank(absorb):
    # with cutoff=0.0 the zero singular values are kept, so must not blow up
    from quimb.tensor.decomp import compute_oblique_projectors

    rng = np.random.default_rng(0)
    d, rank = 12, 5
    Rl = np.hstack([rng.standard_normal((d, rank)), np.zeros((d, d - rank))])
    Rr = np.vstack([rng.standard_normal((rank, d)), np.zeros((d - rank, d))])

    out = compute_oblique_projectors(
        Rl, Rr, max_bond=None, cutoff=0.0, absorb=absorb
    )
    if absorb is None:
        Pl, s, Pr = out
        P = (Pl * s) @ Pr
    else:
        Pl, Pr = out
        P = Pl @ Pr

    assert np.isfinite(P).all()
    # inserting the projectors should be the identity on the non-null space
    assert_allclose(Rl @ P @ Rr, Rl @ Rr, atol=1e-12)


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
    "backend", ["numpy", jax_case, tensorflow_case, pytorch_case]
)
@pytest.mark.parametrize("shape", [(8, 5), (5, 5), (5, 8)])
@pytest.mark.parametrize("stabilized", [True, False])
@pytest.mark.parametrize(
    "method, absorb",
    [
        ("qr", "right"),
        ("qr", "lorthog"),
        ("qr", "rfactor"),
        ("lq", "left"),
        ("lq", "lfactor"),
        ("lq", "rorthog"),
    ],
)
@pytest.mark.parametrize("dtype", ["complex128", "float32"])
def test_qr_lq_stabilized(backend, shape, stabilized, method, absorb, dtype):
    xp = ar.get_namespace(backend)
    rng = xp.random.default_rng(0)
    a = rng.uniform(size=shape)
    a = xp.astype(a, dtype)
    left, _, right = array_split(
        a, method=method, stabilized=stabilized, absorb=absorb
    )

    is_qr = method == "qr"
    orthog = left if is_qr else right
    factor = right if is_qr else left
    no_orthog = absorb in ("rfactor", "lfactor")
    no_factor = absorb in ("lorthog", "rorthog")

    if not no_orthog:
        assert orthog is not None
        if is_qr:
            assert orthog.shape == (shape[0], min(shape))
            assert_allclose(
                xp.conj(xp.swapaxes(orthog, -2, -1)) @ orthog,
                xp.eye(min(shape)),
                rtol=1e-6,
                atol=1e-6,
            )
        else:
            assert orthog.shape == (min(shape), shape[1])
            assert_allclose(
                orthog @ xp.conj(xp.swapaxes(orthog, -2, -1)),
                xp.eye(min(shape)),
                rtol=1e-6,
                atol=1e-6,
            )
    else:
        assert orthog is None

    if not no_factor:
        assert factor is not None
        if is_qr:
            assert factor.shape == (min(shape), shape[1])
        else:
            assert factor.shape == (shape[0], min(shape))
        if stabilized:
            assert_allclose(
                xp.sgn(xp.diag(factor)),
                xp.array(1.0, dtype=factor.dtype),
                rtol=1e-6,
                atol=1e-6,
            )
    else:
        assert factor is None

    if absorb == "both":
        assert_allclose(left @ right, a, rtol=1e-6, atol=1e-6)


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


QR_METHODS = (
    "auto",
    "qr",
    "lq",
    "qr:cholesky",
    "svd",
    "svd:eig",
    "svd:rand",
)
LQ_METHODS = (
    "auto",
    "lq",
    "qr",
    "lq:cholesky",
    "svd",
    "svd:eig",
    "svd:rand",
)


@pytest.mark.parametrize(
    "methods, absorb",
    [
        (QR_METHODS, "right"),
        (LQ_METHODS, "left"),
    ],
)
@pytest.mark.parametrize("m, n", [(8, 5), (5, 5), (5, 8)])
@pytest.mark.parametrize("dtype", ["float64", "complex128"])
def test_qr_lq_methods(methods, absorb, m, n, dtype):
    for method in methods:
        if "cholesky" in method:
            if absorb == "right" and m < n:
                continue
            if absorb == "left" and m > n:
                continue

        rng = np.random.default_rng(42)
        x = rng.standard_normal((m, n))
        if dtype == "complex128":
            x = x + 1j * rng.standard_normal((m, n))

        k = min(m, n)
        opts = {}
        if "rand" in method:
            opts["max_bond"] = k

        left, _, right = array_split(x, method=method, absorb=absorb, **opts)

        assert left.shape == (m, k)
        assert right.shape == (k, n)
        assert_allclose(left @ right, x, atol=1e-10)

        if absorb == "right":
            assert_allclose(left.conj().T @ left, np.eye(k), atol=1e-10)
        else:
            assert_allclose(right @ right.conj().T, np.eye(k), atol=1e-10)


@pytest.mark.parametrize("method", QR_METHODS + LQ_METHODS)
@pytest.mark.parametrize("absorb", ["left", "right"])
def test_qr_lq_methods_truncated(method, absorb):
    if "qr" in method or "lq" in method:
        pytest.skip("no truncation support")

    rng = np.random.default_rng(42)
    x = rng.standard_normal((8, 6))

    max_bond = 4
    opts = {"max_bond": max_bond}

    if "rand" in method:
        opts["oversample"] = 10
        opts["num_iterations"] = 2

    left, _, right = array_split(x, method=method, absorb=absorb, **opts)

    if absorb == "right":
        assert left.shape == (8, max_bond)
        assert right.shape == (max_bond, 6)
        assert_allclose(left.conj().T @ left, np.eye(max_bond), atol=1e-10)
    else:
        assert left.shape == (8, max_bond)
        assert right.shape == (max_bond, 6)
        assert_allclose(right @ right.conj().T, np.eye(max_bond), atol=1e-10)

    assert np.linalg.norm(x - left @ right) < 2.0


_BASE_METHODS = ["auto", "qr", "lq", "svd", "svd:eig", "svd:rand"]

_PARTIAL_ABSORB_CONFIG = {
    # absorb: (extra_methods, cholesky_skip_condition, null_side, check_orthog)
    #   null_side: "left" means left output is None
    #   cholesky_skip: "m<n" or "m>n"
    "rfactor": (
        ["qr:cholesky"],
        "m<n",
        "left",
        False,
    ),
    "lfactor": (
        ["qr:cholesky", "lq:cholesky"],
        "m>n",
        "right",
        False,
    ),
    "rorthog": (
        ["qr:cholesky", "lq:cholesky"],
        "m>n",
        "left",
        True,
    ),
    "lorthog": (
        ["qr:cholesky"],
        "m<n",
        "right",
        True,
    ),
}


@pytest.mark.parametrize(
    "absorb", ["rfactor", "lfactor", "rorthog", "lorthog"]
)
@pytest.mark.parametrize("m, n", [(8, 5), (5, 5), (5, 8)])
@pytest.mark.parametrize("dtype", ["float64", "complex128"])
def test_partial_factor_methods(absorb, m, n, dtype):
    extra_methods, chol_skip, null_side, check_orthog = _PARTIAL_ABSORB_CONFIG[
        absorb
    ]
    methods = _BASE_METHODS + extra_methods

    for method in methods:
        if "cholesky" in method:
            if chol_skip == "m<n" and m < n:
                continue
            if chol_skip == "m>n" and m > n:
                continue

        rng = np.random.default_rng(42)
        x = rng.standard_normal((m, n))
        if dtype == "complex128":
            x = x + 1j * rng.standard_normal((m, n))

        k = min(m, n)
        opts = {}
        if "rand" in method:
            opts["max_bond"] = k

        left, _, right = array_split(x, method=method, absorb=absorb, **opts)

        if null_side == "left":
            assert left is None
            assert right.shape == (k, n)
            if check_orthog:
                assert_allclose(right @ right.conj().T, np.eye(k), atol=1e-10)
        else:
            assert right is None
            assert left.shape == (m, k)
            if check_orthog:
                assert_allclose(left.conj().T @ left, np.eye(k), atol=1e-10)


@pytest.mark.parametrize(
    "backend", ["numpy", jax_case, tensorflow_case, pytorch_case]
)
@pytest.mark.parametrize("method", ["svd", "svd:eig", "svd:rand"])
@pytest.mark.parametrize("max_bond", [None, 4])
@pytest.mark.parametrize("absorb", ["both", "left", "right", None])
def test_batch_svd(backend, method, max_bond, absorb):
    xp = ar.get_namespace(backend)
    rng = xp.random.default_rng(42)
    x = rng.uniform(size=(3, 5, 7))

    kwargs = {
        "method": method,
        "max_bond": max_bond,
        "absorb": absorb,
    }
    if method == "svd:rand":
        kwargs["seed"] = rng

    left, s, right = array_split(x, **kwargs)

    if max_bond is None:
        k = 5
    else:
        k = 4

    assert left is None or left.shape == (3, 5, k)
    assert s is None or s.shape == (3, k)
    assert right is None or right.shape == (3, k, 7)


@pytest.mark.parametrize(
    "backend",
    [
        "numpy",
        "autoray.lazy",
        jax_case,
        tensorflow_case,
        pytorch_case,
    ],
)
@pytest.mark.parametrize(
    "method, shape, expected_shapes",
    [
        ("qr", (3, 7, 5), ((3, 7, 5), (3, 5), (3, 5, 5))),
        ("qr:cholesky", (3, 7, 5), ((3, 7, 5), (3, 5), (3, 5, 5))),
        ("lq", (3, 5, 7), ((3, 5, 5), (3, 5), (3, 5, 7))),
        ("lq:cholesky", (3, 5, 7), ((3, 5, 5), (3, 5), (3, 5, 7))),
    ],
)
def test_batch_qr_lq(backend, method, shape, expected_shapes):
    if backend == "autoray.lazy":
        backend = "numpy"
    xp = ar.get_namespace(backend)
    rng = xp.random.default_rng(42)
    x = rng.uniform(size=shape)

    left, s, right = array_split(x, method)
    left_shape, s_shape, right_shape = expected_shapes
    assert left is None or left.shape == left_shape
    assert s is None or s.shape == s_shape
    assert right is None or right.shape == right_shape


@pytest.mark.parametrize(
    "backend", ["numpy", jax_case, tensorflow_case, pytorch_case]
)
@pytest.mark.parametrize("max_bond", [None, 4])
@pytest.mark.parametrize("absorb", ["both", "left", "right", None])
def test_batch_eigh(backend, max_bond, absorb):
    xp = ar.get_namespace(backend)
    rng = xp.random.default_rng(42)
    # build a batch of symmetric (hermitian) matrices: x @ x.T
    a = rng.uniform(size=(3, 6, 6))
    x = a @ xp.swapaxes(a, -2, -1)

    left, s, right = array_split(
        x, method="eigh", max_bond=max_bond, absorb=absorb
    )

    k = 4 if max_bond is not None else 6

    assert left is None or left.shape == (3, 6, k)
    assert s is None or s.shape == (3, k)
    assert right is None or right.shape == (3, k, 6)


def test_eigh_shift():
    x = np.diag([4.0, 1.0, 0.0])
    trace = np.trace(x)

    _, s0, _ = array_split(
        x,
        method="eigh",
        absorb="s",
        cutoff=0.0,
        positive=1,
    )
    _, s_false, _ = array_split(
        x,
        method="eigh",
        absorb="s",
        cutoff=0.0,
        positive=1,
        shift=False,
    )
    _, s_float, _ = array_split(
        x,
        method="eigh",
        absorb="s",
        cutoff=0.0,
        positive=1,
        shift=0.1,
    )
    _, s_true, _ = array_split(
        x,
        method="eigh",
        absorb="s",
        cutoff=0.0,
        positive=1,
        shift=True,
    )

    assert_allclose(s_false, s0)
    assert_allclose(s_float, s0 + 0.1 * trace)
    assert_allclose(s_true, s0 + np.finfo(x.dtype).eps * trace)
