import pytest
from pytest import approx

import quimb as qu
import quimb.tensor as qtn


class TestTEBD:

    def test_setup_and_sweep(self):
        n = 10
        H_int = qu.ham_heis(n=2, cyclic=False)
        psi0 = qtn.MPS_neel_state(n, dtype=complex)
        tebd = qtn.TEBD(psi0, H_int, dt=0.05)
        assert tebd.pt.bond_size(0, 1) == 1
        tebd.sweep('right', 1 / 2)
        assert tebd.pt.count_canonized() == (n - 1, 0)
        tebd.sweep('left', 1 / 2)
        assert tebd.pt.count_canonized() == (0, n - 1)
        assert tebd.pt.bond_size(0, 1) > 1
        assert not tebd._queued_sweep

    @pytest.mark.parametrize('order', [2, 4])
    @pytest.mark.parametrize('dt,tol', [
        (0.0759283, None),
        (None, 1e-4),
        (0.0759283, 1e-4),
        (None, None),
    ])
    def test_evolve_obc(self, order, dt, tol):
        n = 10
        tf = 2
        psi0 = qtn.MPS_neel_state(n)
        H_int = qu.ham_heis(2, cyclic=False)

        if dt and tol:
            with pytest.raises(ValueError):
                qtn.TEBD(psi0, H_int, dt=dt, tol=tol)
            return

        tebd = qtn.TEBD(psi0, H_int, dt=dt, tol=tol)

        tebd.split_opts['cutoff'] = 0.0

        if (dt is None and tol is None):
            with pytest.raises(ValueError):
                tebd.update_to(tf, order=order)
            return

        tebd.update_to(tf, order=order)
        assert tebd.t == approx(tf)
        assert not tebd._queued_sweep

        dpsi0 = psi0.to_dense()
        dham = qu.ham_heis(n=n, sparse=True, cyclic=False)
        evo = qu.Evolution(dpsi0, dham)
        evo.update_to(tf)

        assert qu.expec(evo.pt, tebd.pt.to_dense()) == approx(1, rel=1e-5)

    @pytest.mark.parametrize('dt,tol', [
        (0.0659283, None),
        (None, 1e-5),
    ])
    def test_at_times(self, dt, tol):
        n = 10
        psi0 = qtn.MPS_neel_state(n)
        H_int = qu.ham_heis(2, cyclic=False)
        tebd = qtn.TEBD(psi0, H_int, dt=dt, tol=tol)
        assert tebd.H.special_sites == set()

        for pt in tebd.at_times([0.1, 0.2, 0.3, 0.4, 0.5]):
            assert pt.H @ pt == approx(1, rel=1e-5)

        assert tebd.err <= 1e-5

    def test_NNI_and_single_site_terms(self):
        n = 10
        psi0 = qtn.MPS_neel_state(n)
        H_nni = qtn.NNI_ham_XY(n, bz=0.9)
        assert H_nni.special_sites == {(8, 9)}
        tebd = qtn.TEBD(psi0, H_nni)
        tebd.update_to(1.0, tol=1e-5)
        assert abs(psi0.H @ tebd.pt) < 1.0
        assert tebd.pt.entropy(5) > 0.0

        psi0_dns = qu.neel_state(n)
        H_dns = qu.ham_XY(10, jxy=1.0, bz=0.9, cyclic=False)
        evo = qu.Evolution(psi0_dns, H_dns)
        evo.update_to(1.0)

        assert qu.expec(tebd.pt.to_dense(), evo.pt) == pytest.approx(1.0)

    def test_NNI_and_single_site_terms_heis(self):
        n = 10
        psi0 = qtn.MPS_neel_state(n)
        H_nni = qtn.NNI_ham_heis(n, j=(0.7, 0.8, 0.9), bz=0.337)
        tebd = qtn.TEBD(psi0, H_nni)
        tebd.update_to(1.0, tol=1e-5)
        assert abs(psi0.H @ tebd.pt) < 1.0
        assert tebd.pt.entropy(5) > 0.0

        psi0_dns = qu.neel_state(n)
        H_dns = qu.ham_heis(10, j=(0.7, 0.8, 0.9), b=0.337, cyclic=False)
        evo = qu.Evolution(psi0_dns, H_dns)
        evo.update_to(1.0)

        assert qu.expec(tebd.pt.to_dense(), evo.pt) == pytest.approx(1.0)

    def test_non_trans_invar(self):
        n = 10
        tf = 1.0
        p0 = qtn.MPS_rand_state(n, bond_dim=1)
        H = qtn.NNI_ham_mbl(n, dh=1.7, cyclic=False, run=42)
        print(H)
        assert H.special_sites == {(i, i + 1) for i in range(n)}
        tebd = qtn.TEBD(p0, H)
        tebd.update_to(tf, tol=1e-3)

        p0d = p0.to_dense()
        Hd = qu.ham_mbl(n, dh=1.7, cyclic=False, run=42, sparse=True)
        evo = qu.Evolution(p0d, Hd)
        evo.update_to(tf)

        assert qu.expec(tebd.pt.to_dense(), evo.pt) == pytest.approx(1.0)
