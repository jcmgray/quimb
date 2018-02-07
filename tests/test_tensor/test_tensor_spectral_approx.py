import pytest
import numpy as np
from numpy.testing import assert_allclose

from quimb import approx_spectral_function, eigvals
from quimb.tensor import MPO_rand_herm, MPO_ham_heis, DMRG2, MPS_rand_state
from quimb.tensor.tensor_approx_spectral import (
    construct_lanczos_tridiag_MPO,
    EMPS_rand_state,
    MPSPTPT,
)


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
        dmrg = DMRG2(ham, bond_dims=4)
        dmrg.solve()
        rho_ab = dmrg.state.ptr(range(6, 14))

        xe = rho_ab.trace()
        xf = approx_spectral_function(rho_ab, lambda x: x, R=20)
        assert_allclose(xe, xf, rtol=0.5, atol=0.001)


sysa_sysb_configs = ([(3, 4, 5), (7, 8, 9)],
                     [(0, 1), (4, 5, 6)],
                     [(8,), (9, 10)],
                     [(5, 6), (10, 11)])


class TestEMPS:

    @pytest.mark.parametrize("sysa,sysb", sysa_sysb_configs)
    def test_init(self, sysa, sysb):
        e = EMPS_rand_state(sysa, sysb, nsites=12, bond_dim=7)
        assert_allclose(e.H @ e, 1.0)


class TestMPSPTPT:

    @pytest.mark.parametrize("sysa,sysb", sysa_sysb_configs)
    def test_init(self, sysa, sysb):
        p = MPS_rand_state(12, 7)
        pX = MPSPTPT(p, sysa=sysa, sysb=sysb)

        assert len(pX.X.tensors) == (2 * len(sysa) +
                                     2 * len(sysb) +
                                     int(max(sysa) + 1 != min(sysb)))
