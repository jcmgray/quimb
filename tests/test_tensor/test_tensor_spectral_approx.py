import pytest
import numpy as np
from numpy.testing import assert_allclose

from quimb import approx_spectral_function, eigvals
from quimb.tensor import MPO_rand_herm, MPO_ham_heis, DMRG2
from quimb.tensor.tensor_approx_spectral import construct_lanczos_tridiag_MPO


class TestTensorSpectralApprox:

    def test_simple(self):
        A = MPO_rand_herm(10, 7)
        for x in construct_lanczos_tridiag_MPO(A, 5):
            pass

    @pytest.mark.parametrize("fn", [abs, np.cos, lambda x: np.sin(x)**2])
    def test_approx_fn(self, fn):
        A = MPO_rand_herm(10, 7, normalize=True)
        xe = sum(fn(eigvals(A.to_dense())))
        xf = approx_spectral_function(A, fn, R=10)
        assert_allclose(xe, xf, rtol=0.2)

    def test_realistic(self):
        ham = MPO_ham_heis(20)
        dmrg = DMRG2(ham, bond_dims=8)
        dmrg.solve()
        rho_ab = dmrg.state.ptr(range(4, 16))

        xe = rho_ab.trace()
        xf = approx_spectral_function(rho_ab, lambda x: x, R=10)
        assert_allclose(xe, xf, rtol=0.5, atol=0.001)
