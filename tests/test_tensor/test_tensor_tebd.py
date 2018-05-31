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

    @pytest.mark.parametrize('order', [2, 4])
    def test_evolve_obc(self, order):
        n = 10
        tf = 5
        psi0 = qtn.MPS_neel_state(n)
        H_int = qu.ham_heis(2, cyclic=False)
        tebd = qtn.TEBD(psi0, H_int, dt=0.098741)
        tebd.split_opts['cutoff'] = 0.0
        tebd.update_to(tf, order=order)
        assert tebd.t == approx(tf)

        dpsi0 = psi0.to_dense()
        dham = qu.ham_heis(n=n, sparse=True, cyclic=False)
        evo = qu.Evolution(dpsi0, dham)
        evo.update_to(tf)

        assert qu.expec(evo.pt, tebd.pt.to_dense()) == approx(1, rel=1e-5)
