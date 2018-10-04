import pytest
from pytest import approx
import numpy as np

import quimb as qu
import quimb.tensor as qtn
from quimb.tensor.tensor_tebd import OTOC

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

    @pytest.mark.parametrize('cyclic', [False, True])
    @pytest.mark.parametrize('order', [2, 4])
    @pytest.mark.parametrize('dt,tol', [
        (0.0759283, None),
        (None, 1e-4),
        (0.0759283, 1e-4),
        (None, None),
    ])
    def test_evolve_obc_pbc(self, order, dt, tol, cyclic):
        n = 10
        tf = 2
        psi0 = qtn.MPS_neel_state(n, cyclic=cyclic)
        H_int = qu.ham_heis(2, cyclic=False)  # this is just the interaction

        if dt and tol:
            with pytest.raises(ValueError):
                qtn.TEBD(psi0, H_int, dt=dt, tol=tol)
            return

        tebd = qtn.TEBD(psi0, H_int, dt=dt, tol=tol)
        assert tebd.cyclic == cyclic
        tebd.split_opts['cutoff'] = 1e-10

        if (dt is None and tol is None):
            with pytest.raises(ValueError):
                tebd.update_to(tf, order=order)
            return

        tebd.update_to(tf, order=order)
        assert tebd.t == approx(tf)
        assert not tebd._queued_sweep

        dpsi0 = psi0.to_dense()
        dham = qu.ham_heis(n=n, sparse=True, cyclic=cyclic)
        evo = qu.Evolution(dpsi0, dham)
        evo.update_to(tf)

        assert qu.expec(evo.pt, tebd.pt.to_dense()) == approx(1, rel=1e-5)

    @pytest.mark.parametrize('cyclic', [False, True])
    @pytest.mark.parametrize('dt,tol', [
        (0.0659283, None),
        (None, 1e-5),
    ])
    def test_at_times(self, dt, tol, cyclic):
        n = 10
        psi0 = qtn.MPS_neel_state(n, cyclic=cyclic)
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
        H = qtn.NNI_ham_mbl(n, dh=1.7, cyclic=False, seed=42)
        print(H)
        assert H.special_sites == {(i, i + 1) for i in range(n)}
        tebd = qtn.TEBD(p0, H)
        tebd.update_to(tf, tol=1e-3)

        p0d = p0.to_dense()
        Hd = qu.ham_mbl(n, dh=1.7, cyclic=False, seed=42, sparse=True)
        evo = qu.Evolution(p0d, Hd)
        evo.update_to(tf)

        assert qu.expec(tebd.pt.to_dense(), evo.pt) == pytest.approx(1.0)

    @pytest.mark.parametrize('cyclic', [False, True])
    def test_ising_model_with_field(self, cyclic):

        p = qtn.MPS_computational_state('0000100000', cyclic=cyclic)
        pd = p.to_dense()

        H_nni = qtn.NNI_ham_ising(10, j=4, bx=1, cyclic=cyclic)
        H_mpo = qtn.MPO_ham_ising(10, j=4, bx=1, cyclic=cyclic)
        H = qu.ham_ising(10, jz=4, bx=1, cyclic=cyclic)

        tebd = qtn.TEBD(p, H_nni, tol=1e-6)
        tebd.split_opts['cutoff'] = 1e-9
        tebd.split_opts['cutoff_mode'] = 'rel'
        evo = qu.Evolution(pd, H)

        e0 = qu.expec(pd, H)
        e0_mpo = qtn.expec_TN_1D(p.H, H_mpo, p)

        assert e0_mpo == pytest.approx(e0)

        tf = 2
        ts = np.linspace(0, tf, 21)
        evo.update_to(tf)

        for pt in tebd.at_times(ts):
            assert isinstance(pt, qtn.MatrixProductState)
            assert (pt.H @ pt) == pytest.approx(1.0, rel=1e-5)

        assert (qu.expec(tebd.pt.to_dense(), evo.pt) ==
                pytest.approx(1.0, rel=1e-5))

        ef_mpo = qtn.expec_TN_1D(tebd.pt.H, H_mpo, tebd.pt)
        assert ef_mpo == pytest.approx(e0, 1e-5)


def test_OTOC():
    L = 10
    psi0 = qtn.MPS_computational_state('0'*L, cyclic=True)
    H1 = qtn.NNI_ham_ising(L, j=4, bx=0, cyclic=True)
    H_back1 = qtn.NNI_ham_ising(L, j=-4, bx=0, cyclic=True)
    H2 = qtn.NNI_ham_ising(L, j=4, bx=3, cyclic=True)
    H_back2 = qtn.NNI_ham_ising(L, j=-4, bx=3, cyclic=True)

    assert OTOC(psi0, H1, H_back1, 0, 5) == pytest.approx(1.0)
    assert OTOC(psi0, H1, H_back1, 1, 5) == pytest.approx(1.0)
    x = OTOC(psi0, H2, H_back2, 1, 5)
    y = OTOC(psi0, H2, H_back2, 1, 4)
    assert abs(x-y) < 1e-4
    assert OTOC(psi0, H2, H_back2, 2, 5) == pytest.approx(0.1, 0.05)
