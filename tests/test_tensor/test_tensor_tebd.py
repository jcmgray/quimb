import pytest
from pytest import approx
import numpy as np

import quimb as qu
import quimb.tensor as qtn
from quimb.tensor.tensor_1d_tebd import OTOC_local


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
    @pytest.mark.parametrize('n', [5, 6])
    def test_evolve_obc_pbc(self, n, order, dt, tol, cyclic):
        tf = 1.0 if cyclic else 2
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

        assert qu.expec(evo.pt, tebd.pt.to_dense()) == approx(1, rel=1e-3 if
                                                              cyclic else 1e-5)

    @pytest.mark.parametrize('cyclic', [False, True])
    @pytest.mark.parametrize('order', [2, 4])
    @pytest.mark.parametrize('dt,tol', [
        (0.0759283, None),
        (None, 1e-4),
        (0.0759283, 1e-4),
        (None, None),
    ])
    @pytest.mark.parametrize('n', [5, 6])
    def test_imag_evolve_obc_pbc(self, n, order, dt, tol, cyclic):
        tf = 2
        ground = qtn.MPS_computational_state('0' * n, cyclic=cyclic)
        excited = qtn.MPS_computational_state(('01' * n)[:n], cyclic=cyclic)
        psi0 = (ground + excited) / 2**0.5
        H = qtn.ham_1d_ising(n, j=-1, cyclic=cyclic)

        if dt and tol:
            with pytest.raises(ValueError):
                qtn.TEBD(psi0, H, dt=dt, tol=tol, imag=True)
            return

        tebd = qtn.TEBD(psi0, H, dt=dt, tol=tol, imag=True)
        assert tebd.cyclic == cyclic
        tebd.split_opts['cutoff'] = 1e-10

        if (dt is None and tol is None):
            with pytest.raises(ValueError):
                tebd.update_to(tf, order=order)
            return

        tebd.update_to(tf, order=order)
        assert tebd.t == approx(tf)
        assert not tebd._queued_sweep

        H_mpo = qtn.MPO_ham_ising(n, j=-1, cyclic=cyclic)
        E_ground = qtn.expec_TN_1D(ground.H, H_mpo, ground)
        E_excited = qtn.expec_TN_1D(excited.H, H_mpo, excited)

        psi1 = (np.exp(-tf * E_ground) * ground +
                np.exp(-tf * E_excited) * excited)
        psi1 /= np.sqrt(np.exp(-2 * tf * E_ground) +
                        np.exp(-2 * tf * E_excited))

        assert qtn.expec_TN_1D(psi1.H, tebd.pt) == approx(1, rel=1e-5)

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

        for pt in tebd.at_times([0.1, 0.2, 0.3, 0.4, 0.5]):
            assert pt.H @ pt == approx(1, rel=1e-5)

        assert tebd.err <= 1e-5

    def test_local_ham_1d_and_single_site_terms(self):
        n = 10
        psi0 = qtn.MPS_neel_state(n)
        lham_1d = qtn.ham_1d_XY(n, bz=0.9)
        tebd = qtn.TEBD(psi0, lham_1d)
        tebd.update_to(1.0, tol=1e-5)
        assert abs(psi0.H @ tebd.pt) < 1.0
        assert tebd.pt.entropy(5) > 0.0

        psi0_dns = qu.neel_state(n)
        H_dns = qu.ham_XY(10, jxy=1.0, bz=0.9, cyclic=False)
        evo = qu.Evolution(psi0_dns, H_dns)
        evo.update_to(1.0)

        assert qu.expec(tebd.pt.to_dense(), evo.pt) == pytest.approx(1.0)

    def test_local_ham_1d_and_single_site_terms_heis(self):
        n = 10
        psi0 = qtn.MPS_neel_state(n)
        lham_1d = qtn.ham_1d_heis(n, j=(0.7, 0.8, 0.9), bz=0.337)
        tebd = qtn.TEBD(psi0, lham_1d)
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
        H = qtn.ham_1d_mbl(n, dh=1.7, cyclic=False, seed=42)
        print(H)
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

        lham_1d = qtn.ham_1d_ising(10, j=4, bx=1, cyclic=cyclic)
        H_mpo = qtn.MPO_ham_ising(10, j=4, bx=1, cyclic=cyclic)
        H = qu.ham_ising(10, jz=4, bx=1, cyclic=cyclic)

        tebd = qtn.TEBD(p, lham_1d, tol=1e-6)
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


def test_OTOC_local():
    L = 10
    psi0 = qtn.MPS_computational_state('0' * L, cyclic=True)
    H1 = qtn.ham_1d_ising(L, j=4, bx=0, cyclic=True)
    H_back1 = qtn.ham_1d_ising(L, j=-4, bx=0, cyclic=True)
    H2 = qtn.ham_1d_ising(L, j=4, bx=1, cyclic=True)
    H_back2 = qtn.ham_1d_ising(L, j=-4, bx=-1, cyclic=True)
    A = qu.pauli('z')
    ts = np.linspace(1, 2, 2)
    OTOC_t = []
    for OTOC in OTOC_local(psi0, H1, H_back1, ts, 5, A, tol=1e-5,
                           split_opts={'cutoff': 1e-5, 'cutoff_mode': 'rel'},
                           initial_eigenstate='check'):
        OTOC_t += [OTOC]
    assert OTOC_t[0] == pytest.approx(1.0)
    assert OTOC_t[1] == pytest.approx(1.0)
    x_t = []
    for x in OTOC_local(psi0, H2, H_back2, ts, 5, A, tol=1e-5,
                        split_opts={'cutoff': 1e-5, 'cutoff_mode': 'rel'},
                        initial_eigenstate='check'):
        x_t += [x]
    assert x_t[0] == pytest.approx(0.52745, rel=1e-4, abs=1e-9)
    assert x_t[1] == pytest.approx(0.70440, rel=1e-4, abs=1e-9)
