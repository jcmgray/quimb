from pytest import fixture, mark

from math import pi, gcd, cos
from functools import reduce

import numpy as np
from numpy.testing import assert_allclose

from quimb import (
    qu, eigsys,
    rand_ket,
    rand_rho,
    rand_herm,
    rand_matrix,
    rand_uni,
    overlap,
    ham_heis,
    up,
    down,
    eyepad,
    sig,
)
from quimb.evo import (
    schrodinger_eq_ket,
    schrodinger_eq_dop,
    schrodinger_eq_dop_vec,
    lindblad_eq,
    lindblad_eq_vec,
    QuEvo,
)


@fixture
def psi_dot():
    psi = rand_ket(3)
    ham = 10 * rand_herm(3)
    psid = -1.0j * (ham @ psi)
    return psi, ham, psid


@fixture
def spsi_dot():
    psi = rand_ket(3)
    ham = rand_herm(3, sparse=True, density=0.5)
    psid = -1.0j * (ham @ psi)
    return psi, ham, psid


@fixture
def rho_dot():
    rho = rand_rho(3)
    ham = rand_herm(3)
    rhod = -1.0j * (ham @ rho - rho @ ham)
    return rho, ham, rhod


@fixture
def srho_dot():
    rho = rand_rho(3)
    ham = rand_herm(3, sparse=True, density=0.5)
    rhod = -1.0j * (ham @ rho - rho @ ham)
    return rho, ham, rhod


@fixture
def rho_dot_ls():
    np.random.seed(1)
    rho = rand_rho(3)
    ham = rand_herm(3)
    gamma = 0.7
    ls = [rand_matrix(3) for _ in range(3)]
    rhodl = -1.0j * (ham @ rho - rho @ ham)
    for l in ls:
        rhodl += gamma * (l @ rho @ l.H)
        rhodl -= gamma * 0.5 * (rho @ l.H @ l)
        rhodl -= gamma * 0.5 * (l.H @ l @ rho)
    return rho, ham, gamma, ls, rhodl


@fixture
def srho_dot_ls():
    rho = rand_rho(3)
    ham = rand_herm(3, sparse=True, density=0.5)
    gamma = 0.7
    ls = [rand_matrix(3, sparse=True, density=0.5) for _ in range(3)]
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
        foo = schrodinger_eq_dop_vec(ham)
        rhod2 = foo(None, rho.A.reshape(-1)).reshape(3, 3)
        assert_allclose(rhod, rhod2)

    def test_dop_1darray_sparse(self, srho_dot):
        rho, ham, rhod = srho_dot
        foo = schrodinger_eq_dop_vec(ham)
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
        foo = lindblad_eq_vec(ham, ls, gamma)
        rhod2 = foo(None, rho.A.reshape(-1)).reshape(3, 3)
        assert_allclose(rhod, rhod2)

    def test_1darray_sparse(self, srho_dot_ls):
        rho, ham, gamma, ls, rhod = srho_dot_ls
        foo = lindblad_eq_vec(ham, ls, gamma)
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
    v = rand_uni(d)
    ham = v @ np.diag(evals) @ v.H
    p0 = rand_ket(d)
    tm = 0.573 * trc
    pm = v @ np.diag(np.exp(-1.0j * tm * evals)) @ v.H @ p0
    return ham, trc, p0, tm, pm


class TestQuEvo:
    @mark.parametrize("sparse, presolve",
                      [(False, False),
                       (True, False),
                       (False, True)])
    def test_quevo_ham_dense_ket_solve(self, ham_rcr_psi, sparse, presolve):
        ham, trc, p0, tm, pm = ham_rcr_psi
        ham = qu(ham, sparse=sparse)
        if presolve:
            l, v = eigsys(ham)
            sim = QuEvo(p0, (l, v))
            assert sim.solved
        else:
            sim = QuEvo(p0, ham, solve=True)
        sim.update_to(tm)
        assert_allclose(sim.pt, pm)
        assert overlap(sim.pt, p0) < 1.0
        sim.update_to(trc)
        assert_allclose(sim.pt, p0)
        assert isinstance(sim.pt, np.matrix)
        assert sim.t == trc

    @mark.parametrize("dop", [False, True])
    @mark.parametrize("sparse", [False, True])
    @mark.parametrize("solve", [False, True])
    def test_quevo_ham(self, ham_rcr_psi, sparse, dop, solve):
        ham, trc, p0, tm, pm = ham_rcr_psi
        if dop:
            p0 = p0 @ p0.H
            pm = pm @ pm.H
        ham = qu(ham, sparse=sparse)
        sim = QuEvo(p0, ham, solve=solve)
        sim.update_to(tm)
        assert_allclose(sim.pt, pm, rtol=1e-4)
        assert overlap(sim.pt, p0) < 1.0
        sim.update_to(trc)
        assert_allclose(sim.pt, p0, rtol=1e-4)
        assert isinstance(sim.pt, np.matrix)
        assert sim.t == trc

    def test_quevo_at_times(self):
        ham = ham_heis(2, cyclic=False)
        p0 = up() & down()
        sim = QuEvo(p0, ham, solve=True)
        ts = np.linspace(0, 10)
        for t, pt in zip(ts, sim.at_times(ts)):
            x = cos(4 * t)
            y = overlap(pt, eyepad(sig('z'), [2, 2], 0))
            assert_allclose(x, y, atol=1e-15)
