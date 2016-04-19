from math import pi, gcd
from functools import reduce
from pytest import fixture
import numpy as np
from numpy.testing import assert_allclose
from quijy import (
    qjf,
    rand_ket,
    rand_rho,
    rand_herm,
    rand_matrix,
    rand_uni,
)
from quijy.evo import (
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
    # Define a hamiltonian with a known recurrence time
    def lcm(a, b):
        return a * b // gcd(a, b)

    def lcms(*args):
        return reduce(lcm, args)

    def gcds(*args):
        return reduce(gcd, args)

    d = 3
    ems = np.random.randint(1, 6, d)
    LCD = gcds(*ems)
    ens = np.random.randint(1, 10, d)
    LCM = lcms(*ens)
    trc = 2 * LCM/LCD
    l = np.array(ems) / np.array(ens)
    v = rand_uni(d)
    ham = v @ np.diag(l) @ v.H
    p0 = qjf([[0.123], [0.456j], [0.789]], qtype='ket', normalized=True)
    tm = 0.573 * trc
    pm = v @ np.diag(np.exp(-1.0j * tm * l)) @ v.H @ p0
    return ham, trc, p0, tm, pm


class TestQuEvo:
    def test_quevo_ham_dense_ket(self, ham_rcr):
        ham, t_recur, p0, p57 = ham_rcr
        sim = QuEvo(p0, ham, solve=False)
        sim.update_to(57 * pi)
        assert_allclose(p57, sim.pt, rtol=1e-6, atol=1e-12)
        sim.update_to(t_recur * pi)
        assert_allclose(p0, sim.pt, rtol=1e-6, atol=1e-12)
