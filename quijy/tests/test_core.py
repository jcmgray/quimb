import scipy.sparse as sp
import numpy as np
from numpy.testing import assert_allclose
from nose.tools import assert_almost_equal, eq_, ok_
from quijy.core import *


# Quijify
def test_quijify_vector_create():
    x = [1, 2, 3j]
    p = quijify(x, qtype='ket')
    eq_(type(p), np.matrix)
    eq_(p.dtype, np.complex)
    eq_(p.shape, (3, 1))
    p = quijify(x, qtype='bra')
    eq_(p.shape, (1, 3))
    assert_almost_equal(p[0,2], -3.0j)

def test_quijify_dop_create():
    x = np.random.randn(3, 3)
    p = quijify(x, qtype='dop')
    eq_(type(p), np.matrix)
    eq_(p.dtype, np.complex)
    eq_(p.shape, (3, 3))

def test_quijify_convert_vector_to_dop():
    x = [1, 2, 3j]
    p = quijify(x, qtype='r')
    assert_allclose(p, np.matrix([[1.+0.j,  2.+0.j,  0.-3.j],
                                  [2.+0.j,  4.+0.j,  0.-6.j],
                                  [0.+3.j,  0.+6.j,  9.+0.j]]))


def test_quijify_chopped():
    x = [9e-16, 1]
    p = quijify(x, 'k', chopped=False)
    ok_(p[0, 0] != 0.0)
    p = quijify(x, 'k', chopped=True)
    eq_(p[0, 0], 0.0)


def test_quijify_normalized():
    x = [3j, 4j]
    p = quijify(x, 'k', normalized=False)
    assert_almost_equal(tr(p.H @ p), 25.)
    p = quijify(x, 'k', normalized=True)
    assert_almost_equal(tr(p.H @ p), 1.)
    p = quijify(x, 'dop', normalized=True)
    assert_almost_equal(tr(p), 1.)

def test_quijify_sparse_create():
    x = [[1, 0], [3, 0]]
    p = quijify(x, 'dop', sparse=False)
    eq_(type(p), np.matrix)
    p = quijify(x, 'dop', sparse=True)
    eq_(type(p), sp.csr_matrix)
    eq_(p.dtype, np.complex)
    eq_(p.nnz, 2)


def test_quijify_sparse_convert_to_dop():
    x = [1, 0, 9e-16, 0, 3j]
    p = quijify(x, 'ket', sparse=True)
    q = quijify(p, 'dop', sparse=True)
    eq_(q.shape, (5, 5))
    eq_(q.nnz, 9)
    assert_almost_equal(q[4, 4], 9.)
    q = quijify(p, 'dop', sparse=True, normalized=True)
    assert_almost_equal(tr(q), 1.)


# Shape checks
def test_ket():
    x = qjf([[1],[0]])
    ok_(isket(x))
    ok_(not isbra(x))
    ok_(not isop(x))


def test_bra():
    x = qjf([[1 , 0]])
    ok_(not isket(x))
    ok_(isbra(x))
    ok_(not isop(x))


def test_op():
    x = qjf([[1 , 0]])
    ok_(not isket(x))
    ok_(isbra(x))
    ok_(not isop(x))
