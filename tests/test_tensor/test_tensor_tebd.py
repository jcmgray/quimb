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

        for pt in tebd.at_times([0.1, 0.2, 0.3, 0.4, 0.5]):
            assert pt.H @ pt == approx(1, rel=1e-5)

        assert tebd.err <= 1e-5
