from nose.tools import eq_, ok_
from nose.tools import assert_almost_equal as aeq_
import numpy as np
from numpy.testing import assert_allclose
from quijy.core import eye, chop, tr
from quijy.calc import mutual_information
from quijy.rand import (rand_uni, rand_product_state, rand_matrix,
                        rand_herm, rand_pos, rand_rho, rand_ket,
                        rand_haar_state, gen_rand_haar_states,
                        rand_mix)


def test_rand_matrix():
    a = rand_matrix(5)
    eq_(a.shape, (5, 5))
    eq_(type(a), np.matrix)


def test_rand_herm():
    a = rand_herm(5)
    eq_(a.shape, (5, 5))
    eq_(type(a), np.matrix)
    assert_allclose(a, a.H)
    l = np.linalg.eigvals(a)
    assert_allclose(l.imag, [0, 0, 0, 0, 0], atol=1e-15)


def test_rand_pos():
    a = rand_pos(5)
    eq_(a.shape, (5, 5))
    eq_(type(a), np.matrix)
    l = np.linalg.eigvals(a)
    assert_allclose(l.imag, [0, 0, 0, 0, 0], atol=1e-15)
    ok_(np.all(l.real >= 0))


def test_rand_rho():
    rho = rand_rho(5)
    eq_(rho.shape, (5, 5))
    eq_(type(rho), np.matrix)
    aeq_(tr(rho), 1.0)


def test_rand_ket():
    ket = rand_ket(5)
    eq_(ket.shape, (5, 1))
    eq_(type(ket), np.matrix)
    aeq_(tr(ket.H @ ket), 1.0)


def test_rand_uni():
    u = rand_uni(5)
    eq_(u.shape, (5, 5))
    eq_(type(u), np.matrix)
    assert_allclose(eye(5), chop(u @ u.H, inplace=False))
    assert_allclose(eye(5), chop(u.H @ u, inplace=False))


def test_rand_haar_state():
    ket = rand_haar_state(5)
    eq_(ket.shape, (5, 1))
    eq_(type(ket), np.matrix)
    aeq_(tr(ket.H @ ket), 1.0)


def test_gen_rand_haar_states():
    kets = [*gen_rand_haar_states(3, 6)]
    for ket in kets:
        eq_(ket.shape, (3, 1))
        eq_(type(ket), np.matrix)
        aeq_(tr(ket.H @ ket), 1.0)


def test_rand_mix():
    rho = rand_mix(5)
    eq_(rho.shape, (5, 5))
    eq_(type(rho), np.matrix)
    aeq_(tr(rho), 1.0)
    ok_(tr(rho @ rho) < 1.0)


def test_rand_product_state():
    a = rand_product_state(3)
    eq_(a.shape[0], 2**3)
    aeq_((a.H @ a)[0, 0].real, 1.0)
    aeq_(mutual_information(a, [2, 2, 2], 0, 1), 0.0)
    aeq_(mutual_information(a, [2, 2, 2], 1, 2), 0.0)
    aeq_(mutual_information(a, [2, 2, 2], 0, 2), 0.0)
