from math import log2
import pytest
import numpy as np
from numpy.testing import assert_allclose

from quimb import (
    approx_spectral_function,
    eigvals,
    ham_heis,
    groundstate,
    logneg_subsys,
)
from quimb.tensor import MPO_rand_herm, MPO_ham_heis, DMRG2, MPS_rand_state
from quimb.tensor.tensor_approx_spectral import (
    construct_lanczos_tridiag_MPO,
    EEMPS_rand_state,
    PTPTLazyMPS,
    construct_lanczos_tridiag_PTPTLazyMPS,
)


class TestMPOSpectralApprox:

    def test_constructing_tridiag_works(self):
        A = MPO_rand_herm(10, 7)
        for _ in construct_lanczos_tridiag_MPO(A, 5):
            pass

    @pytest.mark.parametrize("fn", [abs, np.cos, lambda x: np.sin(x)**2])
    def test_approx_fn(self, fn):
        A = MPO_rand_herm(10, 7, normalize=True)
        xe = sum(fn(eigvals(A.to_dense())))
        xf = approx_spectral_function(A, fn, tol=0.1, verbosity=2)
        assert_allclose(xe, xf, rtol=0.5)

    def test_realistic(self):
        ham = MPO_ham_heis(20)
        dmrg = DMRG2(ham, bond_dims=4)
        dmrg.solve()
        rho_ab = dmrg.state.ptr(range(6, 14))
        xf = approx_spectral_function(rho_ab, lambda x: x,
                                      tol=0.1, verbosity=2)
        assert_allclose(1.0, xf, rtol=0.5, atol=0.001)

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
        assert_allclose(lne, lnx, rtol=0.5, atol=0.1)


sysa_sysb_configs = ([(3, 4, 5), (7, 8, 9)],
                     [(0, 1), (4, 5, 6)],
                     [(7, 8,), (9, 10)],
                     [(5, 6), (10, 11)])


class TestEEMPS:

    @pytest.mark.parametrize("sysa,sysb", sysa_sysb_configs)
    def test_init(self, sysa, sysb):
        e = EEMPS_rand_state(sysa, sysb, nsites=12, bond_dim=7)
        assert_allclose(e.H @ e, 1.0)
        assert e.sites == (*sysa, *sysb)

    @pytest.mark.parametrize("sysa,sysb", sysa_sysb_configs)
    def test_add_EEMPS(self, sysa, sysb):
        x = EEMPS_rand_state(sysa, sysb, nsites=12, bond_dim=7)
        y = EEMPS_rand_state(sysa, sysb, nsites=12, bond_dim=9)
        z = x + y
        assert_allclose(z.H @ z, x.H @ x + x.H @ y + y.H @ y + y.H @ x)


class TestPTPTLazyMPS:

    @pytest.mark.parametrize("sysa,sysb", sysa_sysb_configs)
    def test_init(self, sysa, sysb):
        p = MPS_rand_state(12, 7)
        pX = PTPTLazyMPS(p, sysa=sysa, sysb=sysb)
        assert len(pX.TN.tensors) == (2 * len(sysa) +
                                      2 * len(sysb) +
                                      int(max(sysa) + 1 != min(sysb)))

        pt = pX.TN.reindex({
            pX.upper_ind_id.format(i): pX.lower_ind_id.format(i)
            for i in (*pX.sysa, *pX.sysb)
        })
        assert_allclose(pt.contract_tags(...), 1.0)

    def test_apply(self):
        psi_abc = MPS_rand_state(20, bond_dim=7)
        X = PTPTLazyMPS(psi_abc, sysa=range(5, 10), sysb=range(12, 16))
        v0 = EEMPS_rand_state(X.sysa, X.sysb, nsites=X.TN.nsites, bond_dim=7)
        vf = X.apply(v0)

        assert vf.site_ind_id == 'k{}'
        assert vf.site_tag_id == 'I{}'

        v0d, Xd, vfd = map(lambda x: x.to_dense(), (v0, X, vf))
        assert_allclose(vfd, Xd @ v0d)


class TestPTPTLazyMPSSpectralApprox:

    @pytest.mark.parametrize("sysa,sysb", sysa_sysb_configs)
    def test_construct_tridiag_works(self, sysa, sysb):
        p = MPS_rand_state(12, 7)
        pX = PTPTLazyMPS(p, sysa=sysa, sysb=sysb)

        for _ in construct_lanczos_tridiag_PTPTLazyMPS(pX, 5):
            pass

    def test_realistic(self):
        ham = MPO_ham_heis(20)
        dmrg = DMRG2(ham, bond_dims=16)
        dmrg.solve()
        sysa, sysb = range(2, 9), range(12, 18)
        rho_ab_pt = PTPTLazyMPS(dmrg.state, sysa, sysb)
        xf = approx_spectral_function(rho_ab_pt, lambda x: x,
                                      tol=0.1, verbosity=2)
        assert_allclose(1, xf, rtol=0.5, atol=0.1)

    def test_realistic_ent(self):
        n = 12
        sysa, sysb = range(3, 6), range(6, 8)

        ham = MPO_ham_heis(n)
        dmrg = DMRG2(ham, bond_dims=[10, 20, 40, 80])
        dmrg.solve()
        rho_ab_pt = PTPTLazyMPS(dmrg.state, sysa, sysb)

        psi0 = dmrg.state.to_dense()
        lne = logneg_subsys(psi0, [2] * n, sysa=sysa, sysb=sysb)
        lnx = log2(approx_spectral_function(rho_ab_pt, abs, tol=0.1,
                                            verbosity=2))
        assert_allclose(lne, lnx, rtol=0.5, atol=0.1)


class TestPartialTraceCompress:

    @pytest.mark.parametrize("n", [12])
    @pytest.mark.parametrize("l", [1, 2, 3, 5])
    @pytest.mark.parametrize("gap", [0, 1, 2])
    def test_heisenberg(self, n, l, gap):

        ham = MPO_ham_heis(n)
        dmrg = DMRG2(ham)
        dmrg.solve()

        g = gap // 2
        m = n // 2 + g + gap % 2

        sysa = range(m - l - g, m - g)
        sysb = range(m, m + l)

        rho_ab = dmrg.state.partial_trace_compress(sysa, sysb)

        rho_ab_pt_lo = rho_ab.aslinearoperator(['k0', 'b1'], ['b0', 'k1'])

        ln = log2(approx_spectral_function(rho_ab_pt_lo, abs))

        # exact
        lne = logneg_subsys(groundstate(ham_heis(n, cyclic=False)),
                            [2] * n, sysa, sysb)

        assert_allclose(lne, ln, rtol=0.05, atol=0.05)
