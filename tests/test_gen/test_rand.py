import importlib

import pytest
from numpy.testing import assert_allclose
import numpy as np
import scipy.sparse as sp

import quimb as qu


found_randomgen = bool(importlib.util.find_spec('randomgen'))
reason = "randomgen not installed."
randomgen_mark = pytest.mark.skipif(not found_randomgen, reason=reason)


dtypes = [np.float32, np.float64, np.complex128, np.complex64]


class TestRandn:

    @pytest.mark.parametrize(
        'dtype', dtypes + [float, complex, 'f8', 'f4', 'c8', 'c16', 'float32',
                           'float64', 'complex64', 'complex128'])
    def test_basic(self, dtype):
        x = qu.randn((2, 3, 4), dtype=dtype)
        assert x.shape == (2, 3, 4)
        assert x.dtype == np.dtype(dtype)

    @pytest.mark.parametrize("num_threads", [2, 3])
    @pytest.mark.parametrize("dist", ['uniform', 'normal', 'exp'])
    def test_multithreaded(self, num_threads, dist):
        x = qu.randn((2, 3, 4), dist=dist, num_threads=num_threads, seed=42)
        y = qu.randn((2, 3, 4), dist=dist, num_threads=num_threads, seed=42)
        assert_allclose(x, y)

    def test_can_seed(self):
        assert_allclose(qu.randn(5, seed=42), qu.randn(5, seed=42))

    def test_scale_and_loc(self):
        x = qu.randn(1000, scale=100, loc=50, dtype=float, seed=42)
        assert_allclose(np.mean(x), 50, rtol=1e-1)
        assert_allclose(np.std(x), 100, rtol=1e-1)

    @pytest.mark.parametrize("dtype,tol", [
        ('complex64', 1e-5),
        ('complex128', 1e-11),
    ])
    def test_rand_phase(self, dtype, tol):
        x = qu.gen.rand.rand_phase(10, dtype=dtype)
        assert x.dtype == dtype
        assert_allclose(np.abs(x), np.ones(10), rtol=tol)

    @pytest.mark.parametrize("dtype", ['float32', 'float64',
                                       'complex64', 'complex128'])
    def test_rand_rademacher(self, dtype):
        x = qu.gen.rand.rand_rademacher(10, dtype=dtype)
        assert x.dtype == dtype
        assert_allclose(np.abs(x), np.ones(10))

    @pytest.mark.parametrize('bitgen', [
        'MT19937',
        'PCG64',
        'Philox',
        'SFC64',
        pytest.param('JSF', marks=randomgen_mark),
        pytest.param('SFMT', marks=randomgen_mark),
        pytest.param('Xoshiro256', marks=randomgen_mark),
        pytest.param('Xoshiro512', marks=randomgen_mark),
    ])
    def test_set_bitgen(self, bitgen):
        x0 = qu.randn(3, seed=42)
        qu.set_rand_bitgen(bitgen)
        x1 = qu.randn(3, seed=42)
        assert not np.allclose(x0, x1) or bitgen == 'PCG64'  # <- default
        x2 = qu.randn(3, seed=42)
        assert_allclose(x1, x2)
        qu.set_rand_bitgen(None)
        x3 = qu.randn(3, seed=42)
        assert_allclose(x0, x3)


@pytest.mark.parametrize('dtype', dtypes)
class TestRandMatrix:
    def test_rand_matrix(self, dtype):
        a = qu.rand_matrix(3, scaled=True, dtype=dtype)
        assert a.shape == (3, 3)
        assert type(a) == qu.qarray
        assert a.dtype == dtype

    def test_rand_matrix_sparse(self, dtype):
        a = qu.rand_matrix(3, sparse=True, dtype=dtype)
        assert a.shape == (3, 3)
        assert type(a) == sp.csr_matrix
        assert a.dtype == dtype

    def test_rand_matrix_sparse_density(self, dtype):
        a = qu.rand_matrix(3, sparse=True, density=1 / 9, dtype=dtype)
        assert a.nnz == 1
        a = qu.rand_matrix(3, sparse=True, density=7 / 9, dtype=dtype)
        assert a.nnz == 7

    def test_rand_matrix_bsr(self, dtype):
        a = qu.rand_matrix(10, sparse=True, density=0.2,
                           stype='bsr', dtype=dtype)
        assert a.shape == (10, 10)
        assert type(a) == sp.bsr_matrix
        assert a.dtype == dtype

    @pytest.mark.parametrize('sparse', [False, True])
    def test_seed(self, dtype, sparse):
        a = qu.rand_matrix(10, sparse=sparse, dtype=dtype, seed=42)
        b = qu.rand_matrix(10, sparse=sparse, dtype=dtype, seed=42)
        if sparse:
            assert_allclose(a.data, b.data)
        else:
            assert_allclose(a, b)


@pytest.mark.parametrize('dtype', dtypes)
class TestRandHerm:
    def test_rand_herm(self, dtype):
        a = qu.rand_herm(3, dtype=dtype)
        assert a.shape == (3, 3)
        assert type(a) == qu.qarray
        assert a.dtype == dtype
        assert_allclose(a, a.H)
        evals = qu.eigvalsh(a)
        assert_allclose(evals.imag, [0, 0, 0], atol=1e-14)

    def test_rand_herm_sparse(self, dtype):
        a = qu.rand_herm(3, sparse=True, density=0.3, dtype=dtype)
        assert a.shape == (3, 3)
        assert type(a) == sp.csr_matrix
        assert qu.isherm(a)
        assert a.dtype == dtype
        evals = qu.eigvalsh(a.A)
        assert_allclose(evals.imag, [0, 0, 0], atol=1e-14)


@pytest.mark.parametrize('dtype', dtypes)
class TestRandPos:
    def test_rand_pos(self, dtype):
        a = qu.rand_pos(3, dtype=dtype)
        assert qu.ispos(a)
        assert a.shape == (3, 3)
        assert type(a) == qu.qarray
        assert a.dtype == dtype
        evals = qu.eigvalsh(a)
        assert_allclose(evals.imag, [0, 0, 0], atol=1e-7)
        assert np.all(evals.real >= 0)

    def test_rand_pos_sparse(self, dtype):
        a = qu.rand_pos(3, sparse=True, density=0.3, dtype=dtype)
        assert a.shape == (3, 3)
        assert type(a) == sp.csr_matrix
        assert a.dtype == dtype
        evals = qu.eigvalsh(a.A)
        assert_allclose(evals.imag, [0, 0, 0], atol=1e-7)
        assert np.all(evals.real >= -1e-15)


@pytest.mark.parametrize('dtype', dtypes)
class TestRandRho:
    def test_rand_rho(self, dtype):
        rho = qu.rand_rho(3, dtype=dtype)
        assert rho.shape == (3, 3)
        assert type(rho) == qu.qarray
        assert rho.dtype == dtype
        assert_allclose(qu.tr(rho), 1.0)

    def test_rand_rho_sparse(self, dtype):
        rho = qu.rand_rho(3, sparse=True, density=0.3, dtype=dtype)
        assert rho.shape == (3, 3)
        assert type(rho) == sp.csr_matrix
        assert rho.dtype == dtype
        assert_allclose(qu.tr(rho), 1.0)


@pytest.mark.parametrize('dtype', dtypes)
class TestRandUni:
    def test_rand_uni(self, dtype):
        u = qu.rand_uni(3, dtype=dtype)
        assert u.shape == (3, 3)
        assert type(u) == qu.qarray
        assert u.dtype == dtype
        # low tolerances for float32 etc
        assert_allclose(qu.eye(3), u @ u.H, atol=1e-7, rtol=1e-5)
        assert_allclose(qu.eye(3), u.H @ u, atol=1e-7, rtol=1e-5)


class TestRandKet:
    def test_rand_ket(self):
        ket = qu.rand_ket(3)
        assert ket.shape == (3, 1)
        assert type(ket) == qu.qarray
        assert_allclose(qu.tr(ket.H @ ket), 1.0)


class TestRandHaarState:
    def test_rand_haar_state(self):
        ket = qu.rand_haar_state(3)
        assert ket.shape == (3, 1)
        assert type(ket) == qu.qarray
        assert_allclose(qu.tr(ket.H @ ket), 1.0)

    def test_gen_rand_haar_states(self):
        kets = [*qu.gen_rand_haar_states(3, 6)]
        for ket in kets:
            assert ket.shape == (3, 1)
            assert type(ket) == qu.qarray
            assert_allclose(qu.tr(ket.H @ ket), 1.0)


class TestRandMix:
    @pytest.mark.parametrize("mode", ['rand', 'haar'])
    def test_rand_mix(self, mode):
        rho = qu.rand_mix(3, mode=mode)
        assert rho.shape == (3, 3)
        assert type(rho) == qu.qarray
        assert_allclose(qu.tr(rho), 1.0)
        mixedness = qu.tr(rho @ rho)
        assert mixedness < 1.0


class TestRandProductState:
    def test_rand_product_state(self):
        a = qu.rand_product_state(3)
        assert a.shape[0] == 2**3
        assert (a.H @ a)[0, 0].real == pytest.approx(1.0)
        assert qu.mutinf(a, [2, 2, 2], 0, 1) == pytest.approx(0.0)
        assert qu.mutinf(a, [2, 2, 2], 1, 2) == pytest.approx(0.0)
        assert qu.mutinf(a, [2, 2, 2], 0, 2) == pytest.approx(0.0)


class TestRandMPS:
    @pytest.mark.parametrize("cyclic", (True, False))
    @pytest.mark.parametrize("d_n_b_e", [
        (2, 4, 5, 16),
        (2, 4, 1, 16),
        (3, 3, 7, 27),
    ])
    def test_shape(self, d_n_b_e, cyclic):
        d, n, b, e = d_n_b_e
        psi = qu.rand_matrix_product_state(n, b, d, cyclic=cyclic)
        assert psi.shape == (e, 1)

        assert_allclose(qu.expec(psi, psi), 1.0)

    @pytest.mark.parametrize("cyclic", (True, False))
    @pytest.mark.parametrize("bond_dim", (1, 2, 3))
    def test_rank(self, bond_dim, cyclic):
        psi = qu.rand_matrix_product_state(
            10, bond_dim, cyclic=cyclic)
        rhoa = qu.ptr(psi, [2] * 10, [0, 1, 2, 3])
        el = qu.eigvalsh(rhoa)
        # bond_dim squared as cyclic mps is generated
        assert sum(el > 1e-12) == bond_dim ** (2 if cyclic else 1)


class TestRandSeperable:

    def test_entanglement(self):
        rho = qu.rand_seperable([2, 3, 2], 10)
        assert_allclose(qu.tr(rho), 1.0)
        assert qu.isherm(rho)

        assert qu.logneg(rho, [2, 6]) < 1e-12
        assert qu.logneg(rho, [6, 2]) < 1e-12

        rho_a = qu.ptr(rho, [2, 3, 2], 1)

        el = qu.eigvalsh(rho_a)
        assert np.all(el < 1 - 1e-12)
        assert np.all(el > 1e-12)


class TestRandMERA:

    @pytest.mark.parametrize("invariant", [False, True])
    @pytest.mark.parametrize("dtype", ['float32', 'float64',
                                       'complex64', 'complex128'])
    def test_simple(self, invariant, dtype):
        m = qu.rand_mera(8, invariant=invariant, dtype=dtype)
        assert m.dtype == dtype
        assert m.H @ m == pytest.approx(1.0)
        assert m.shape == (256, 1)
