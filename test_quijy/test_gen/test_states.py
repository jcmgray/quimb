import numpy as np
from pytest import raises
from numpy.testing import assert_allclose
from quijy.core import tr, eye, chop
from quijy.solve import eigvals, eigsys, groundstate
from quijy.gen import basis_vec, sig, thermal_state, ham_j1j2, rand_herm


class TestBasisVec:
    def test_basis_vec(self):
        x = basis_vec(1, 2)
        assert_allclose(x, np.matrix([[0.], [1.]]))
        x = basis_vec(1, 2, qtype='b')
        assert_allclose(x, np.matrix([[0., 1.]]))

    def test_basis_vec_sparse(self):
        x = basis_vec(4, 100, sparse=True)
        assert x[4, 0] == 1.
        assert x.nnz == 1
        assert x.dtype == complex


class TestThermalState:
    def test_thermal_state_normalization(self):
        full = rand_herm(2**4)
        for beta in (0, 0.5, 1, 10):
            rhoth = thermal_state(full, beta)
            assert_allclose(tr(rhoth), 1)

    def test_thermal_state_tuple(self):
        full = rand_herm(2**4)
        l, v = eigsys(full)
        for beta in (0, 0.5, 1, 10):
            rhoth1 = thermal_state(full, beta)
            rhoth2 = thermal_state((l, v), beta)
            assert_allclose(rhoth1, rhoth2)

    def test_thermal_state_hot(self):
        full = rand_herm(2**4)
        rhoth = chop(thermal_state(full, 0.0))
        assert_allclose(rhoth, eye(2**4) / 2**4)

    def test_thermal_state_cold(self):
        full = ham_j1j2(4, j2=0.1253)
        rhoth = thermal_state(full, 100)
        gs = groundstate(full)
        assert_allclose(tr(gs.H @ rhoth @ gs), 1.0, rtol=1e-4)

    def test_thermal_state_precomp(self):
        full = rand_herm(2**4)
        beta = 0.624
        rhoth1 = thermal_state(full, beta)
        func = thermal_state(full, None, precomp_func=True)
        rhoth2 = func(beta)
        assert_allclose(rhoth1, rhoth2)
