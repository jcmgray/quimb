import pytest
import numpy as np
from numpy.testing import assert_allclose

from quimb import (
    prod,
    rand_ket,
    rand_pos,
    rand_herm,
    partial_transpose,
    eigvalsh,
    neel_state,
    ham_heis,
    logneg,
    negativity,
    entropy,
    can_use_mpi_pool,
)
from quimb.utils import last

from quimb.linalg.approx_spectral import (
    lazy_ptr_ppt_linop,
    construct_lanczos_tridiag,
    lanczos_tridiag_eig,
    approx_spectral_function,
    lazy_ptr_linop,
    tr_abs_approx,
    tr_exp_approx,
    tr_sqrt_approx,
    tr_xlogx_approx,
    entropy_subsys_approx,
    logneg_subsys_approx,
    negativity_subsys_approx,
    norm_fro,
    norm_fro_approx,
)

MPI_PARALLEL = [False]
if can_use_mpi_pool():
    MPI_PARALLEL.append(True)


np.random.seed(42)

DIMS = [5, 6, 7]
DIMS_MB = [2] * 11


@pytest.fixture
def psi_abc():
    return rand_ket(prod(DIMS))


@pytest.fixture
def psi_ab():
    return rand_ket(prod(DIMS[:-1]))


@pytest.fixture
def psi_ab_mat():
    return np.concatenate(
        [rand_ket(prod(DIMS[:-1])) for _ in range(6)],
        axis=1)


@pytest.fixture
def psi_mb_abc():
    return rand_ket(prod(DIMS_MB))


@pytest.fixture
def psi_mb_ab():
    return rand_ket(prod(DIMS_MB[:7]))


@pytest.fixture
def psi_mb_ab_mat():
    return np.concatenate(
        [rand_ket(prod(DIMS_MB[:7])) for _ in range(6)],
        axis=1)


class TestLazyTensorEval:

    # ------------------------- Just partial trace -------------------------- #

    def test_lazy_ptr_dot_simple_linear_op(self, psi_abc, psi_ab):
        rho_ab = psi_abc.ptr(DIMS, [0, 1])
        psi_out_expected = rho_ab @ psi_ab
        lo = lazy_ptr_linop(psi_abc, DIMS, [0, 1])
        assert hasattr(lo, "H")
        assert lo.dtype == complex
        assert lo.shape == (DIMS[0] * DIMS[1], DIMS[0] * DIMS[1])
        psi_out_got = lo @ psi_ab
        assert_allclose(psi_out_expected, psi_out_got)

    def test_lazy_ptr_dot_mat_simple_linear_op(self, psi_abc, psi_ab_mat):
        rho_ab = psi_abc.ptr(DIMS, [0, 1])
        psi_out_expected = rho_ab @ psi_ab_mat
        lo = lazy_ptr_linop(psi_abc, DIMS, [0, 1])
        psi_out_got = lo @ psi_ab_mat
        assert_allclose(psi_out_expected, psi_out_got)

    def test_lazy_ptr_dot_manybody_linear_op(self, psi_mb_abc, psi_mb_ab):
        sysa = [0, 1, 2, 3, 7, 8, 9]
        rho_ab = psi_mb_abc.ptr(DIMS_MB, sysa)
        psi_out_expected = rho_ab @ psi_mb_ab
        lo = lazy_ptr_linop(psi_mb_abc, DIMS_MB, sysa=sysa)
        assert hasattr(lo, "H")
        assert lo.dtype == complex
        assert lo.shape == (128, 128)
        psi_out_got = lo @ psi_mb_ab
        assert_allclose(psi_out_expected, psi_out_got)

    def test_lazy_ptr_dot_mat_manybody_linear_op(self, psi_mb_abc,
                                                 psi_mb_ab_mat):
        sysa = [0, 1, 2, 3, 7, 8, 9]
        rho_ab = psi_mb_abc.ptr(DIMS_MB, sysa)
        psi_out_expected = rho_ab @ psi_mb_ab_mat
        lo = lazy_ptr_linop(psi_mb_abc, DIMS_MB, sysa=sysa)
        psi_out_got = lo @ psi_mb_ab_mat
        assert_allclose(psi_out_expected, psi_out_got)

    # ----------------- partial trace and partial transpose ----------------- #

    def test_lazy_ptr_ppt_dot_linear_op(self, psi_abc, psi_ab):
        rho_ab = psi_abc.ptr(DIMS, [0, 1])
        rho_ab_pt = partial_transpose(rho_ab, DIMS[:-1])
        psi_out_expected = rho_ab_pt @ psi_ab
        lo = lazy_ptr_ppt_linop(psi_abc, DIMS, 0, 1)
        assert hasattr(lo, "H")
        assert lo.dtype == complex
        assert lo.shape == (DIMS[0] * DIMS[1], DIMS[0] * DIMS[1])
        psi_out_got = lo @ psi_ab
        assert_allclose(psi_out_expected, psi_out_got)

    def test_lazy_ptr_ppt_dot_mat_linear_op(self, psi_abc, psi_ab_mat):
        rho_ab = psi_abc.ptr(DIMS, [0, 1])
        rho_ab_pt = partial_transpose(rho_ab, DIMS[:-1])
        psi_out_expected = rho_ab_pt @ psi_ab_mat
        lo = lazy_ptr_ppt_linop(psi_abc, DIMS, 0, 1)
        psi_out_got = lo @ psi_ab_mat
        assert_allclose(psi_out_expected, psi_out_got)

    def test_lazy_ptr_ppt_dot_manybody_linear_op(self, psi_mb_abc, psi_mb_ab):
        sysa = [0, 1, 7, 8]
        sysb = [2, 3, 9]
        rho_ab = psi_mb_abc.ptr(DIMS_MB, sysa + sysb)
        rho_ab = partial_transpose(rho_ab, [2] * 7, sysa=(0, 1, 4, 5))
        psi_out_expected = rho_ab @ psi_mb_ab
        lo = lazy_ptr_ppt_linop(psi_mb_abc, DIMS_MB, sysa, sysb)
        assert hasattr(lo, "H")
        assert lo.dtype == complex
        assert lo.shape == (128, 128)
        psi_out_got = lo.dot(psi_mb_ab)
        assert_allclose(psi_out_expected, psi_out_got)

    def test_lazy_ptr_ppt_dot_mat_manybody_linear_op(self, psi_mb_abc,
                                                     psi_mb_ab_mat):
        sysa = [0, 1, 7, 8]
        sysb = [2, 3, 9]
        rho_ab = psi_mb_abc.ptr(DIMS_MB, sysa + sysb)
        rho_ab = partial_transpose(rho_ab, [2] * 7, sysa=(0, 1, 4, 5))
        psi_out_expected = rho_ab @ psi_mb_ab_mat
        lo = lazy_ptr_ppt_linop(psi_mb_abc, DIMS_MB, sysa, sysb)
        psi_out_got = lo.dot(psi_mb_ab_mat)
        assert psi_out_got.shape[1] > 1
        assert_allclose(psi_out_expected, psi_out_got)


# --------------------- Test lanczos spectral functions --------------------- #

def np_sqrt(x):
    out = np.empty_like(x)
    mtz = x > 0
    out[~mtz] = 0.0
    out[mtz] = np.sqrt(x[mtz])
    return out


def np_xlogx(x):
    out = np.empty_like(x)
    mtz = x > 0
    out[~mtz] = 0.0
    out[mtz] = x[mtz] * np.log2(x[mtz])
    return out


class TestLanczosApprox:
    @pytest.mark.parametrize("bsz", [1, 2, 5])
    def test_construct_lanczos_tridiag(self, bsz):
        a = rand_herm(2**5)
        alpha, beta, scaling = last(
            construct_lanczos_tridiag(a, bsz=bsz, K=20))
        assert alpha.shape == (20,)
        assert beta.shape == (20,)

        el, ev = lanczos_tridiag_eig(alpha, beta)
        assert el.shape == (20,)
        assert el.dtype == float
        assert ev.shape == (20, 20)
        assert ev.dtype == float

    @pytest.mark.parametrize("bsz", [1, 2, 5])
    def test_construct_lanczos_tridiag_beta_breakdown(self, bsz):
        a = rand_herm(8)
        alpha, beta, scaling = last(construct_lanczos_tridiag(a, bsz=bsz, K=9))
        assert alpha.shape == (8,)
        assert beta.shape == (8,)

        el, ev = lanczos_tridiag_eig(alpha, beta)
        assert el.shape == (8,)
        assert el.dtype == float
        assert ev.shape == (8, 8)
        assert ev.dtype == float

    @pytest.mark.parametrize("mpi", MPI_PARALLEL)
    @pytest.mark.parametrize("bsz", [1, 2, 5])
    @pytest.mark.parametrize(
        "fn_matrix_rtol",
        [
            (np.abs, rand_herm, 1e-1),
            (np.sqrt, rand_pos, 1e-1),
            (np.log1p, rand_pos, 2e-1),
            (np.exp, rand_herm, 1e-1),
        ]
    )
    def test_approx_spectral_function(self, fn_matrix_rtol, bsz, mpi):
        fn, matrix, rtol = fn_matrix_rtol
        a = matrix(2**7)
        pos = fn == np.sqrt
        actual_x = sum(fn(eigvalsh(a)))
        approx_x = approx_spectral_function(a, fn, mpi=mpi, pos=pos,
                                            bsz=bsz, verbosity=2)
        assert_allclose(actual_x, approx_x, rtol=rtol)

    @pytest.mark.parametrize("bsz", [1, 2, 5])
    @pytest.mark.parametrize("dist", ['gaussian', 'phase', 'rademacher'])
    @pytest.mark.parametrize(
        "fn_matrix_rtol",
        [
            (np.abs, rand_herm, 2e-1),
            (np.sqrt, rand_pos, 2e-1),
            (np.log1p, rand_pos, 3e-1),
            (np.exp, rand_herm, 2e-1),
        ]
    )
    def test_approx_spectral_function_with_v0(self, dist, fn_matrix_rtol, bsz):
        fn, matrix, rtol = fn_matrix_rtol
        a = matrix(2**7)
        actual_x = sum(fn(eigvalsh(a)))
        # check un-normalized state work properly
        v0 = (neel_state(7) + neel_state(7, down_first=True))
        v0 = v0.A.reshape(-1)
        pos = fn == np.sqrt
        v0_opts = {'dist': dist, 'orthog': True}
        approx_x = approx_spectral_function(a, fn, v0=v0, pos=pos, bsz=bsz,
                                            K=20, verbosity=2, v0_opts=v0_opts)
        assert_allclose(actual_x, approx_x, rtol=rtol)

    @pytest.mark.parametrize("bsz", [1, 2, 5])
    @pytest.mark.parametrize("fn_approx_rtol",
                             [(np_sqrt, tr_sqrt_approx, 0.3),
                              (np.exp, tr_exp_approx, 0.03),
                              (np_xlogx, tr_xlogx_approx, 0.3)])
    def test_approx_spectral_function_ptr_lin_op(self, fn_approx_rtol,
                                                 psi_abc, psi_ab, bsz):
        fn, approx, rtol = fn_approx_rtol
        sysa = [0, 1]
        rho_ab = psi_abc.ptr(DIMS, sysa)
        actual_x = sum(fn(eigvalsh(rho_ab)))
        lo = lazy_ptr_linop(psi_abc, DIMS, sysa)
        approx_x = approx(lo, R=50, bsz=bsz, verbosity=2)
        assert_allclose(actual_x, approx_x, rtol=rtol)

    @pytest.mark.parametrize("bsz", [1, 2, 5])
    @pytest.mark.parametrize("fn_approx_rtol",
                             [(np.exp, tr_exp_approx, 5e-2),
                              (np.abs, tr_abs_approx, 1e-1)])
    def test_approx_spectral_function_ptr_ppt_lin_op(self, fn_approx_rtol,
                                                     psi_abc, psi_ab, bsz):
        fn, approx, rtol = fn_approx_rtol
        rho_ab_ppt = partial_transpose(psi_abc.ptr(DIMS, [0, 1]), DIMS[:-1], 0)
        actual_x = sum(fn(eigvalsh(rho_ab_ppt)))
        lo = lazy_ptr_ppt_linop(psi_abc, DIMS, sysa=0, sysb=1)
        approx_x = approx_spectral_function(lo, fn, K=20, R=20, bsz=bsz)
        assert_allclose(actual_x, approx_x, rtol=rtol)

    @pytest.mark.parametrize("bsz", [1, 2, 5])
    def test_approx_spectral_subspaces_with_heis_partition(self, bsz):
        h = ham_heis(10, sparse=True)
        beta = 0.01
        actual_Z = sum(np.exp(-beta * eigvalsh(h.A)))
        approx_Z = tr_exp_approx(-beta * h, bsz=bsz)
        assert_allclose(actual_Z, approx_Z, rtol=3e-2)


# ------------------------ Test specific quantities ------------------------- #

class TestSpecificApproxQuantities:

    @pytest.mark.parametrize("bsz", [1, 2, 5])
    def test_entropy_approx_simple(self, psi_abc, bsz):
        np.random.seed(42)
        rho_ab = psi_abc.ptr(DIMS, [0, 1])
        actual_e = entropy(rho_ab)
        approx_e = entropy_subsys_approx(psi_abc, DIMS, [0, 1],
                                         bsz=bsz, R=20)
        assert_allclose(actual_e, approx_e, rtol=2e-1)

    @pytest.mark.parametrize("bsz", [1, 2, 5])
    def test_entropy_approx_many_body(self, psi_mb_abc, bsz):
        rho_ab = psi_mb_abc.ptr(DIMS_MB, [0, 1, 7, 8, 2, 3, 9])
        actual_e = entropy(rho_ab)
        approx_e = entropy_subsys_approx(psi_mb_abc, DIMS_MB,
                                         [0, 1, 7, 8, 2, 3, 9], bsz=bsz)
        assert_allclose(actual_e, approx_e, rtol=2e-1)

    @pytest.mark.parametrize("bsz", [1, 2, 5])
    def test_logneg_approx_simple(self, psi_abc, bsz):
        rho_ab = psi_abc.ptr(DIMS, [0, 1])
        actual_ln = logneg(rho_ab, DIMS[:-1], 0)
        approx_ln = logneg_subsys_approx(psi_abc, DIMS, 0, 1, bsz=bsz)
        assert_allclose(actual_ln, approx_ln, rtol=2e-1)

    @pytest.mark.parametrize("bsz", [1, 2, 5])
    def test_logneg_approx_many_body(self, psi_mb_abc, bsz):
        sysa = [0, 1, 7, 8]
        sysb = [2, 3, 9]
        rho_ab = psi_mb_abc.ptr(DIMS_MB, sysa + sysb)
        actual_ln = logneg(rho_ab, [2] * 7, sysa=(0, 1, 4, 5))
        approx_ln = logneg_subsys_approx(psi_mb_abc, DIMS_MB,
                                         sysa=sysa, sysb=sysb, bsz=bsz)
        assert_allclose(actual_ln, approx_ln, rtol=1e-1)

    @pytest.mark.parametrize("bsz", [1, 2, 5])
    def test_negativity_approx_simple(self, psi_abc, bsz):
        rho_ab = psi_abc.ptr(DIMS, [0, 1])
        actual_neg = negativity(rho_ab, DIMS[:-1], 0)
        approx_neg = negativity_subsys_approx(psi_abc, DIMS, 0, 1, bsz=bsz)
        assert_allclose(actual_neg, approx_neg, rtol=2e-1)

    @pytest.mark.parametrize("bsz", [1, 2, 5])
    def test_norm_fro_approx(self, bsz):
        A = rand_herm(2**5)
        actual_norm_fro = norm_fro(A)
        approx_norm_fro = norm_fro_approx(A, tol=1e-2, bsz=bsz)
        assert_allclose(actual_norm_fro, approx_norm_fro, rtol=1e-1)
