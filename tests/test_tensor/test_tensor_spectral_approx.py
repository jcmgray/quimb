from math import log2
import pytest
import numpy as np
from numpy.testing import assert_allclose

from quimb import (
    seed_rand,
    approx_spectral_function,
    eigvalsh,
    logneg_subsys,
)
from quimb.tensor import MPO_rand_herm, MPO_ham_heis, DMRG2
from quimb.tensor.tensor_approx_spectral import construct_lanczos_tridiag_MPO


np.random.seed(42)


# XXX: these all need improvement


class TestMPOSpectralApprox:

    def test_constructing_tridiag_works(self):
        A = MPO_rand_herm(10, 7)
        for _ in construct_lanczos_tridiag_MPO(A, 5):
            pass

    @pytest.mark.parametrize("fn", [abs, np.cos, lambda x: np.sin(x)**2])
    def test_approx_fn(self, fn):
        A = MPO_rand_herm(10, 7, normalize=True)
        xe = sum(fn(eigvalsh(A.to_dense())))
        xf = approx_spectral_function(A, fn, tol=0.1, verbosity=2)
        assert_allclose(xe, xf, rtol=0.5)

    def test_realistic(self):
        seed_rand(42)
        ham = MPO_ham_heis(20)
        dmrg = DMRG2(ham, bond_dims=[2, 4])
        dmrg.solve()
        rho_ab = dmrg.state.ptr(range(6, 14))
        xf = approx_spectral_function(rho_ab, lambda x: x,
                                      tol=0.1, verbosity=2)
        assert_allclose(1.0, xf, rtol=0.6, atol=0.001)

    def test_realistic_ent(self):
        n = 12
        sysa, sysb = range(3, 6), range(6, 8)
        sysab = (*sysa, *sysb)

        ham = MPO_ham_heis(n)
        dmrg = DMRG2(ham, bond_dims=[10])
        dmrg.solve()

        psi0 = dmrg.state.to_dense()
        lne = logneg_subsys(psi0, [2] * n, sysa=sysa, sysb=sysb)

        rho_ab = dmrg.state.ptr(sysab, rescale_sites=True)
        rho_ab_pt = rho_ab.partial_transpose(range(3))
        lnx = log2(approx_spectral_function(rho_ab_pt, abs,
                                            tol=0.1, verbosity=2))
        assert_allclose(lne, lnx, rtol=0.6, atol=0.1)
