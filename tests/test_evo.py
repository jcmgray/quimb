from pytest import fixture, mark, raises

from math import pi, gcd, cos
from functools import reduce

import numpy as np
from numpy.testing import assert_allclose

import quimb as qu
from quimb.evo import (
    schrodinger_eq_ket,
    schrodinger_eq_dop,
    schrodinger_eq_dop_vectorized,
    lindblad_eq,
    lindblad_eq_vectorized,
)
from .test_linalg.test_slepc_linalg import slepc4py_test


@fixture
def psi_dot():
    psi = qu.rand_ket(3)
    ham = 10 * qu.rand_herm(3)
    psid = -1.0j * (ham @ psi)
    return psi, ham, psid


@fixture
def spsi_dot():
    psi = qu.rand_ket(3)
    ham = qu.rand_herm(3, sparse=True, density=0.5)
    psid = -1.0j * (ham @ psi)
    return psi, ham, psid


@fixture
def rho_dot():
    rho = qu.rand_rho(3)
    ham = qu.rand_herm(3)
    rhod = -1.0j * (ham @ rho - rho @ ham)
    return rho, ham, rhod


@fixture
def srho_dot():
    rho = qu.rand_rho(3)
    ham = qu.rand_herm(3, sparse=True, density=0.5)
    rhod = -1.0j * (ham @ rho - rho @ ham)
    return rho, ham, rhod


@fixture
def rho_dot_ls():
    np.random.seed(1)
    rho = qu.rand_rho(3)
    ham = qu.rand_herm(3)
    gamma = 0.7
    ls = [qu.rand_matrix(3) for _ in range(3)]
    rhodl = -1.0j * (ham @ rho - rho @ ham)
    for l in ls:
        rhodl += gamma * (l @ rho @ l.H)
        rhodl -= gamma * 0.5 * (rho @ l.H @ l)
        rhodl -= gamma * 0.5 * (l.H @ l @ rho)
    return rho, ham, gamma, ls, rhodl


@fixture
def srho_dot_ls():
    rho = qu.rand_rho(3)
    ham = qu.rand_herm(3, sparse=True, density=0.5)
    gamma = 0.7
    ls = [qu.rand_matrix(3, sparse=True, density=0.5) for _ in range(3)]
    rhodl = -1.0j * (ham @ rho - rho @ ham)
    for l in ls:
        rhodl += gamma * (l @ rho @ l.H)
        rhodl -= gamma * 0.5 * (rho @ l.H @ l)
        rhodl -= gamma * 0.5 * (l.H @ l @ rho)
    return rho, ham, gamma, ls, rhodl


# --------------------------------------------------------------------------- #
# Evolution equation tests                                                    #
# --------------------------------------------------------------------------- #

class TestSchrodingerEqKet:
    def test_ket_matrix(self, psi_dot):
        psi, ham, psid = psi_dot
        foo = schrodinger_eq_ket(ham)
        psid2 = foo(None, psi)
        assert_allclose(psid, psid2)

    def test_ket_1darray(self, psi_dot):
        psi, ham, psid = psi_dot
        foo = schrodinger_eq_ket(ham)
        psid2 = foo(None, psi.A.reshape(-1)).reshape(-1, 1)
        assert_allclose(psid, psid2)

    def test_ket_matrix_sparse(self, spsi_dot):
        psi, ham, psid = spsi_dot
        foo = schrodinger_eq_ket(ham)
        psid2 = foo(None, psi)
        assert_allclose(psid, psid2)

    def test_ket_1darray_sparse(self, spsi_dot):
        psi, ham, psid = spsi_dot
        foo = schrodinger_eq_ket(ham)
        psid2 = foo(None, psi.A.reshape(-1)).reshape(-1, 1)
        assert_allclose(psid, psid2)


class TestSchrodingerEqDop:
    def test_dop_matrix(self, rho_dot):
        rho, ham, rhod = rho_dot
        foo = schrodinger_eq_dop(ham)
        rhod2 = foo(None, rho.A).reshape(3, 3)
        assert_allclose(rhod, rhod2)

    def test_dop_1darray(self, rho_dot):
        rho, ham, rhod = rho_dot
        foo = schrodinger_eq_dop(ham)
        rhod2 = foo(None, rho.A.reshape(-1)).reshape(3, 3)
        assert_allclose(rhod, rhod2)

    def test_dop_matrix_sparse(self, srho_dot):
        rho, ham, rhod = srho_dot
        foo = schrodinger_eq_dop(ham)
        rhod2 = foo(None, rho.A).reshape(3, 3)
        assert_allclose(rhod, rhod2, atol=1e-12)

    def test_dop_1darray_sparse(self, srho_dot):
        rho, ham, rhod = srho_dot
        foo = schrodinger_eq_dop(ham)
        rhod2 = foo(None, rho.A.reshape(-1)).reshape(3, 3)
        assert_allclose(rhod, rhod2, atol=1e-12)


class TestSchrodingerEqDopVec:
    def test_dop_1darray(self, rho_dot):
        rho, ham, rhod = rho_dot
        foo = schrodinger_eq_dop_vectorized(ham)
        rhod2 = foo(None, rho.A.reshape(-1)).reshape(3, 3)
        assert_allclose(rhod, rhod2)

    def test_dop_1darray_sparse(self, srho_dot):
        rho, ham, rhod = srho_dot
        foo = schrodinger_eq_dop_vectorized(ham)
        rhod2 = foo(None, rho.A.reshape(-1)).reshape(3, 3)
        assert_allclose(rhod, rhod2, atol=1e-12)


class TestLindbladEq:
    def test_matrix(self, rho_dot_ls):
        rho, ham, gamma, ls, rhod = rho_dot_ls
        foo = lindblad_eq(ham, ls, gamma)
        rhod2 = foo(None, rho).reshape(3, 3)
        assert_allclose(rhod, rhod2)

    def test_1darray(self, rho_dot_ls):
        rho, ham, gamma, ls, rhod = rho_dot_ls
        foo = lindblad_eq(ham, ls, gamma)
        rhod2 = foo(None, rho.A.reshape(-1)).reshape(3, 3)
        assert_allclose(rhod, rhod2)

    def test_matrix_sparse(self, srho_dot_ls):
        rho, ham, gamma, ls, rhod = srho_dot_ls
        foo = lindblad_eq(ham, ls, gamma)
        rhod2 = foo(None, rho).reshape(3, 3)
        assert_allclose(rhod, rhod2)

    def test_1darray_sparse(self, srho_dot_ls):
        rho, ham, gamma, ls, rhod = srho_dot_ls
        foo = lindblad_eq(ham, ls, gamma)
        rhod2 = foo(None, rho.A.reshape(-1)).reshape(3, 3)
        assert_allclose(rhod, rhod2)


class TestLindbladEqVec:
    def test_1darray(self, rho_dot_ls):
        rho, ham, gamma, ls, rhod = rho_dot_ls
        foo = lindblad_eq_vectorized(ham, ls, gamma)
        rhod2 = foo(None, rho.A.reshape(-1)).reshape(3, 3)
        assert_allclose(rhod, rhod2)

    def test_1darray_sparse(self, srho_dot_ls):
        rho, ham, gamma, ls, rhod = srho_dot_ls
        foo = lindblad_eq_vectorized(ham, ls, gamma)
        rhod2 = foo(None, rho.A.reshape(-1)).reshape(3, 3)
        assert_allclose(rhod, rhod2)


# --------------------------------------------------------------------------- #
# Evolution class tests                                                       #
# --------------------------------------------------------------------------- #


@fixture
def ham_rcr_psi():
    # Define a random hamiltonian with a known recurrence time
    d = 3
    np.random.seed(1)
    ems = np.random.randint(1, 6, d)
    ens = np.random.randint(1, 6, d)  # eigenvalues as rational numbers
    # numerator lowest common divisor
    LCD = reduce(gcd, ems)
    # denominator lowest common multiple
    LCM = reduce(lambda a, b: a * b // gcd(a, b), ens)
    trc = 2 * pi * LCM / LCD
    evals = np.array(ems) / np.array(ens)
    v = qu.rand_uni(d)
    ham = v @ np.diag(evals) @ v.H
    p0 = qu.rand_ket(d)
    tm = 0.573 * trc
    pm = v @ np.diag(np.exp(-1.0j * tm * evals)) @ v.H @ p0
    return ham, trc, p0, tm, pm


class TestEvolution:
    @mark.parametrize("sparse, presolve",
                      [(False, False),
                       (True, False),
                       (False, True)])
    def test_evo_ham_dense_ket_solve(self, ham_rcr_psi, sparse, presolve):
        ham, trc, p0, tm, pm = ham_rcr_psi
        ham = qu.qu(ham, sparse=sparse)
        if presolve:
            l, v = qu.eigh(ham)
            sim = qu.Evolution(p0, (l, v))
            assert sim._solved
        else:
            sim = qu.Evolution(p0, ham, method='solve')
        sim.update_to(tm)
        assert_allclose(sim.pt, pm)
        assert qu.expec(sim.pt, p0) < 1.0
        sim.update_to(trc)
        assert_allclose(sim.pt, p0)
        assert isinstance(sim.pt, qu.qarray)
        assert sim.t == trc

    @mark.parametrize("dop", [False, True])
    @mark.parametrize("sparse", [False, True])
    @mark.parametrize("method", ["solve", "integrate", 'expm', 'bad'])
    @mark.parametrize("timedep", [False, True])
    def test_evo_ham(self, ham_rcr_psi, sparse, dop, method, timedep):
        ham, trc, p0, tm, pm = ham_rcr_psi
        if dop:
            if method == 'expm':
                # XXX: not implemented
                return
            p0 = p0 @ p0.H
            pm = pm @ pm.H

        if method == 'bad':
            with raises(ValueError):
                qu.Evolution(p0, ham, method=method)
            return

        ham = qu.qu(ham, sparse=sparse)

        if timedep:
            # fake a time dependent ham by making it callable
            ham_object, ham = ham, (lambda t: ham_object)

        if timedep and (method in ('expm', 'solve')):
            with raises(TypeError):
                qu.Evolution(p0, ham, method=method)
            return

        sim = qu.Evolution(p0, ham, method=method)
        sim.update_to(tm)
        assert_allclose(sim.pt, pm, rtol=1e-4, atol=1e-6)
        assert qu.expec(sim.pt, p0) < 1.0
        sim.update_to(trc)
        assert_allclose(sim.pt, p0, rtol=1e-4, atol=1e-6)
        assert isinstance(sim.pt, qu.qarray)
        assert sim.t == trc

    @mark.parametrize("dop", [False, True])
    @mark.parametrize("linop", [False, True])
    def test_evo_timedep_adiabatic(self, dop, linop):
        L = 6
        T = 20

        H1 = qu.ham_mbl(L, dh=1.0, seed=4, sparse=True)
        gs1 = qu.groundstate(H1)
        H2 = qu.ham_mbl(L, dh=1.0, seed=5, sparse=True)
        gs2 = qu.groundstate(H2)

        if linop:
            import scipy.sparse.linalg as spla

            H1 = spla.aslinearoperator(H1)
            H2 = spla.aslinearoperator(H2)

        # make sure two ground states are different
        assert qu.fidelity(gs1, gs2) < 0.5

        # linearly interpolate from one ham to the other
        def ham(t):
            return (1 - t / T) * H1 + (t / T) * H2

        if linop:
            assert isinstance(ham(0.3), spla.LinearOperator)

        if dop:
            p0 = qu.dop(gs1)
        else:
            p0 = gs1

        evo = qu.Evolution(p0, ham, progbar=True)
        evo.update_to(T)

        # final state should now overlap much more with second hamiltonian GS
        assert qu.fidelity(evo.pt, gs1) < 0.5
        assert qu.fidelity(evo.pt, gs2) > 0.99

    def test_evo_at_times(self):
        ham = qu.ham_heis(2, cyclic=False)
        p0 = qu.up() & qu.down()
        sim = qu.Evolution(p0, ham, method='solve')
        ts = np.linspace(0, 10)
        for t, pt in zip(ts, sim.at_times(ts)):
            x = cos(t)
            y = qu.expec(pt, qu.ikron(qu.pauli('z'), [2, 2], 0))
            assert_allclose(x, y, atol=1e-15)

    @mark.parametrize("qtype", ['ket', 'dop'])
    @mark.parametrize("method", ['solve', 'integrate', 'expm'])
    def test_evo_compute_callback(self, qtype, method):
        ham = qu.ham_heis(2, cyclic=False)
        p0 = qu.qu(qu.up() & qu.down(), qtype=qtype)

        def some_quantity(t, pt):
            return t, qu.logneg(pt)

        evo = qu.Evolution(p0, ham, method=method, compute=some_quantity)
        manual_lns = []
        for pt in evo.at_times(np.linspace(0, 1, 6)):
            manual_lns.append(qu.logneg(pt))
        ts, lns = zip(*evo.results)
        assert len(lns) >= len(manual_lns)
        # check a specific value of logneg at t=0.8 was computed automatically
        checked = False
        for t, ln in zip(ts, lns):
            if abs(t - 0.8) < 1e-12:
                assert abs(ln - manual_lns[4]) < 1e-12
                checked = True
        assert checked

    @mark.parametrize("qtype", ['ket', 'dop'])
    @mark.parametrize("method", ['solve', 'integrate', 'expm'])
    def test_evo_multi_compute(self, method, qtype):

        ham = qu.ham_heis(2, cyclic=False)
        p0 = qu.qu(qu.up() & qu.down(), qtype=qtype)

        def some_quantity(t, _):
            return t

        def some_other_quantity(_, pt):
            return qu.logneg(pt)

        evo = qu.Evolution(p0, ham, method=method,
                           compute={'t': some_quantity,
                                    'logneg': some_other_quantity})
        manual_lns = []
        for pt in evo.at_times(np.linspace(0, 1, 6)):
            manual_lns.append(qu.logneg(pt))
        ts = evo.results['t']
        lns = evo.results['logneg']
        assert len(lns) >= len(manual_lns)
        # check a specific value of logneg at t=0.8 was computed automatically
        checked = False
        for t, ln in zip(ts, lns):
            if abs(t - 0.8) < 1e-12:
                assert abs(ln - manual_lns[4]) < 1e-12
                checked = True
        assert checked

    @slepc4py_test
    def test_expm_krylov_expokit(self):
        ham = qu.rand_herm(100, sparse=True, density=0.8)
        psi = qu.rand_ket(100)
        evo_exact = qu.Evolution(psi, ham, method='solve')
        evo_krylov = qu.Evolution(psi, ham, method='expm',
                                  expm_backend='slepc-krylov')
        evo_expokit = qu.Evolution(psi, ham, method='expm',
                                   expm_backend='slepc-expokit')
        ts = np.linspace(0, 100, 21)
        for p1, p2, p3 in zip(evo_exact.at_times(ts),
                              evo_krylov.at_times(ts),
                              evo_expokit.at_times(ts)):
            assert abs(qu.expec(p1, p2) - 1) < 1e-9
            assert abs(qu.expec(p1, p3) - 1) < 1e-9

    def test_progbar_update_to_integrate(self, capsys):
        ham = qu.ham_heis(2, cyclic=False)
        p0 = qu.up() & qu.down()
        sim = qu.Evolution(p0, ham, method='integrate', progbar=True)
        sim.update_to(100)
        # check something as been printed
        _, err = capsys.readouterr()
        assert err and "%" in err

    def test_progbar_at_times_solve(self, capsys):
        ham = qu.ham_heis(2, cyclic=False)
        p0 = qu.up() & qu.down()
        sim = qu.Evolution(p0, ham, method='solve', progbar=True)
        for _ in sim.at_times(np.linspace(0, 100, 11)):
            pass
        # check something as been printed
        _, err = capsys.readouterr()
        assert err and "%" in err

    def test_progbar_at_times_expm(self, capsys):
        ham = qu.ham_heis(2, cyclic=False)
        p0 = qu.up() & qu.down()
        sim = qu.Evolution(p0, ham, method='expm', progbar=True)
        for _ in sim.at_times(np.linspace(0, 100, 11)):
            pass
        # check something as been printed
        _, err = capsys.readouterr()
        assert err and "%" in err
