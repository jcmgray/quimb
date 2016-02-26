from numpy.testing import assert_allclose, assert_almost_equal
import numpy as np
from quijy.core import eye, chop, tr
from quijy.calc import mutual_information
from quijy.rand import (rand_uni, rand_product_state, rand_matrix,
                        rand_herm, rand_pos, rand_rho, rand_ket,
                        rand_haar_state, gen_rand_haar_states,
                        rand_mix)


def test_rand_matrix():
    a = rand_matrix(5, scaled=True)
    assert a.shape == (5, 5)
    assert type(a) == np.matrix


def test_rand_herm():
    a = rand_herm(5)
    assert a.shape == (5, 5)
    assert type(a) == np.matrix
    assert_allclose(a, a.H)
    l = np.linalg.eigvals(a)
    assert_allclose(l.imag, [0, 0, 0, 0, 0], atol=1e-15)


def test_rand_pos():
    a = rand_pos(5)
    assert a.shape == (5, 5)
    assert type(a) == np.matrix
    l = np.linalg.eigvals(a)
    assert_allclose(l.imag, [0, 0, 0, 0, 0], atol=1e-15)
    assert np.all(l.real >= 0)


def test_rand_rho():
    rho = rand_rho(5)
    assert rho.shape == (5, 5)
    assert type(rho) == np.matrix
    assert_almost_equal(tr(rho), 1.0)


def test_rand_ket():
    ket = rand_ket(5)
    assert ket.shape == (5, 1)
    assert type(ket) == np.matrix
    assert_almost_equal(tr(ket.H @ ket), 1.0)


def test_rand_uni():
    u = rand_uni(5)
    assert u.shape == (5, 5)
    assert type(u) == np.matrix
    assert_allclose(eye(5), chop(u @ u.H, inplace=False))
    assert_allclose(eye(5), chop(u.H @ u, inplace=False))


def test_rand_haar_state():
    ket = rand_haar_state(5)
    assert ket.shape == (5, 1)
    assert type(ket) == np.matrix
    assert_almost_equal(tr(ket.H @ ket), 1.0)


def test_gen_rand_haar_states():
    kets = [*gen_rand_haar_states(3, 6)]
    for ket in kets:
        assert ket.shape == (3, 1)
        assert type(ket) == np.matrix
        assert_almost_equal(tr(ket.H @ ket), 1.0)


def test_rand_mix():
    rho = rand_mix(5)
    assert rho.shape == (5, 5)
    assert type(rho) == np.matrix
    assert_almost_equal(tr(rho), 1.0)
    mixedness = tr(rho @ rho)
    assert mixedness < 1.0


def test_rand_product_state():
    a = rand_product_state(3)
    assert a.shape[0] == 2**3
    assert_almost_equal((a.H @ a)[0, 0].real, 1.0)
    assert_almost_equal(mutual_information(a, [2, 2, 2], 0, 1), 0.0)
    assert_almost_equal(mutual_information(a, [2, 2, 2], 1, 2), 0.0)
    assert_almost_equal(mutual_information(a, [2, 2, 2], 0, 2), 0.0)
