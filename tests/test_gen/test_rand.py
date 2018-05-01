import pytest
from numpy.testing import assert_allclose, assert_almost_equal
import numpy as np
import scipy.sparse as sp
from quimb import (
    isherm,
    ispos,
    eye,
    tr,
    ptr,
    expec,
    eigvalsh,
    mutual_information,
    logneg,
    rand_matrix,
    rand_herm,
    rand_pos,
    rand_rho,
    rand_ket,
    rand_uni,
    rand_haar_state,
    gen_rand_haar_states,
    rand_mix,
    rand_product_state,
    rand_matrix_product_state,
    rand_seperable,
)


dtypes = [np.float32, np.float64, np.complex128, np.complex64]


@pytest.mark.parametrize('dtype', dtypes)
class TestRandMatrix:
    def test_rand_matrix(self, dtype):
        a = rand_matrix(3, scaled=True, dtype=dtype)
        assert a.shape == (3, 3)
        assert type(a) == np.matrix
        assert a.dtype == dtype

    def test_rand_matrix_sparse(self, dtype):
        a = rand_matrix(3, sparse=True, dtype=dtype)
        assert a.shape == (3, 3)
        assert type(a) == sp.csr_matrix
        assert a.dtype == dtype

    def test_rand_matrix_sparse_density(self, dtype):
        a = rand_matrix(3, sparse=True, density=1 / 9, dtype=dtype)
        assert a.nnz == 1
        a = rand_matrix(3, sparse=True, density=7 / 9, dtype=dtype)
        assert a.nnz == 7

    def test_rand_matrix_bsr(self, dtype):
        a = rand_matrix(10, sparse=True, density=0.2, stype='bsr', dtype=dtype)
        assert a.shape == (10, 10)
        assert type(a) == sp.bsr_matrix
        assert a.dtype == dtype

    @pytest.mark.parametrize('sparse', [False, True])
    def test_seed(self, dtype, sparse):
        a = rand_matrix(10, sparse=sparse, dtype=dtype, seed=42)
        b = rand_matrix(10, sparse=sparse, dtype=dtype, seed=42)
        if sparse:
            assert_allclose(a.data, b.data)
        else:
            assert_allclose(a, b)


@pytest.mark.parametrize('dtype', dtypes)
class TestRandHerm:
    def test_rand_herm(self, dtype):
        a = rand_herm(3, dtype=dtype)
        assert a.shape == (3, 3)
        assert type(a) == np.matrix
        assert a.dtype == dtype
        assert_allclose(a, a.H)
        evals = np.linalg.eigvalsh(a)
        assert_allclose(evals.imag, [0, 0, 0], atol=1e-14)

    def test_rand_herm_sparse(self, dtype):
        a = rand_herm(3, sparse=True, density=0.3, dtype=dtype)
        assert a.shape == (3, 3)
        assert type(a) == sp.csr_matrix
        assert isherm(a)
        assert a.dtype == dtype
        evals = np.linalg.eigvalsh(a.A)
        assert_allclose(evals.imag, [0, 0, 0], atol=1e-14)


@pytest.mark.parametrize('dtype', dtypes)
class TestRandPos:
    def test_rand_pos(self, dtype):
        a = rand_pos(3, dtype=dtype)
        assert ispos(a)
        assert a.shape == (3, 3)
        assert type(a) == np.matrix
        assert a.dtype == dtype
        evals = np.linalg.eigvalsh(a)
        assert_allclose(evals.imag, [0, 0, 0], atol=1e-7)
        assert np.all(evals.real >= 0)

    def test_rand_pos_sparse(self, dtype):
        a = rand_pos(3, sparse=True, density=0.3, dtype=dtype)
        assert a.shape == (3, 3)
        assert type(a) == sp.csr_matrix
        assert a.dtype == dtype
        evals = np.linalg.eigvalsh(a.A)
        assert_allclose(evals.imag, [0, 0, 0], atol=1e-7)
        assert np.all(evals.real >= -1e-15)


@pytest.mark.parametrize('dtype', dtypes)
class TestRandRho:
    def test_rand_rho(self, dtype):
        rho = rand_rho(3, dtype=dtype)
        assert rho.shape == (3, 3)
        assert type(rho) == np.matrix
        assert rho.dtype == dtype
        assert_almost_equal(tr(rho), 1.0)

    def test_rand_rho_sparse(self, dtype):
        rho = rand_rho(3, sparse=True, density=0.3, dtype=dtype)
        assert rho.shape == (3, 3)
        assert type(rho) == sp.csr_matrix
        assert rho.dtype == dtype
        assert_almost_equal(tr(rho), 1.0)


@pytest.mark.parametrize('dtype', dtypes)
class TestRandUni:
    def test_rand_uni(self, dtype):
        u = rand_uni(3, dtype=dtype)
        assert u.shape == (3, 3)
        assert type(u) == np.matrix
        assert u.dtype == dtype
        # low tolerances for float32 etc
        assert_allclose(eye(3), u @ u.H, atol=1e-7, rtol=1e-5)
        assert_allclose(eye(3), u.H @ u, atol=1e-7, rtol=1e-5)


class TestRandKet:
    def test_rand_ket(self):
        ket = rand_ket(3)
        assert ket.shape == (3, 1)
        assert type(ket) == np.matrix
        assert_almost_equal(tr(ket.H @ ket), 1.0)


class TestRandHaarState:
    def test_rand_haar_state(self):
        ket = rand_haar_state(3)
        assert ket.shape == (3, 1)
        assert type(ket) == np.matrix
        assert_almost_equal(tr(ket.H @ ket), 1.0)

    def test_gen_rand_haar_states(self):
        kets = [*gen_rand_haar_states(3, 6)]
        for ket in kets:
            assert ket.shape == (3, 1)
            assert type(ket) == np.matrix
            assert_almost_equal(tr(ket.H @ ket), 1.0)


class TestRandMix:
    @pytest.mark.parametrize("mode", ['rand', 'haar'])
    def test_rand_mix(self, mode):
        rho = rand_mix(3, mode=mode)
        assert rho.shape == (3, 3)
        assert type(rho) == np.matrix
        assert_almost_equal(tr(rho), 1.0)
        mixedness = tr(rho @ rho)
        assert mixedness < 1.0


class TestRandProductState:
    def test_rand_product_state(self):
        a = rand_product_state(3)
        assert a.shape[0] == 2**3
        assert_almost_equal((a.H @ a)[0, 0].real, 1.0)
        assert_almost_equal(mutual_information(a, [2, 2, 2], 0, 1), 0.0)
        assert_almost_equal(mutual_information(a, [2, 2, 2], 1, 2), 0.0)
        assert_almost_equal(mutual_information(a, [2, 2, 2], 0, 2), 0.0)


class TestRandMPS:
    @pytest.mark.parametrize("cyclic", (True, False))
    @pytest.mark.parametrize("d_n_b_e", [
        (2, 4, 5, 16),
        (2, 4, 1, 16),
        (3, 3, 7, 27),
    ])
    def test_shape(self, d_n_b_e, cyclic):
        d, n, b, e = d_n_b_e
        psi = rand_matrix_product_state(d, n, b, cyclic=cyclic)
        assert psi.shape == (e, 1)

        assert_allclose(expec(psi, psi), 1.0)

    @pytest.mark.parametrize("cyclic", (True, False))
    @pytest.mark.parametrize("bond_dim", (1, 2, 3))
    def test_rank(self, bond_dim, cyclic):
        psi = rand_matrix_product_state(
            2, 10, bond_dim, cyclic=cyclic)
        rhoa = ptr(psi, [2] * 10, [0, 1, 2, 3])
        el = eigvalsh(rhoa)
        # bond_dim squared as cyclic mps is generated
        assert sum(el > 1e-12) == bond_dim ** (2 if cyclic else 1)


class TestRandSeperable:

    def test_entanglement(self):
        rho = rand_seperable([2, 3, 2], 10)
        assert_almost_equal(tr(rho), 1.0)
        assert isherm(rho)

        assert logneg(rho, [2, 6]) < 1e-12
        assert logneg(rho, [6, 2]) < 1e-12

        rho_a = ptr(rho, [2, 3, 2], 1)

        el = eigvalsh(rho_a)
        assert np.all(el < 1 - 1e-12)
        assert np.all(el > 1e-12)
