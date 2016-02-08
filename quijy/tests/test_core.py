from quijy.core import *
import numpy as np
import scipy.sparse as sp
from nose.tools import assert_almost_equal, assert_equal
from numpy.testing import assert_allclose


def test_quijify_vector_create():
    x = [1, 2, 3j]
    p = quijify(x, qtype='ket')
    assert_equal(type(p), np.matrix)
    assert_equal(p.dtype, np.complex)
    assert_equal(p.shape, (3, 1))
    p = quijify(x, qtype='bra')
    assert_equal(p.shape, (1, 3))
    assert_almost_equal(p[0,2], -3.0j)

def test_quijify_dop_create():
    x = np.random.randn(3, 3)
    p = quijify(x, qtype='dop')
    assert_equal(type(p), np.matrix)
    assert_equal(p.dtype, np.complex)
    assert_equal(p.shape, (3, 3))

def test_quijify_convert_vector_to_dop():
    x = [1, 2, 3j]
    p = quijify(x, qtype='r')
    assert_allclose(p, np.matrix([[1.+0.j,  2.+0.j,  0.-3.j],
                                  [2.+0.j,  4.+0.j,  0.-6.j],
                                  [0.+3.j,  0.+6.j,  9.+0.j]]))


def test_quijify_chopped():
    x = [9e-16, 1]
    p = quijify(x, 'k', chopped=False)
    assert(p[0, 0] != 0.0)
    p = quijify(x, 'k', chopped=True)
    assert_equal(p[0, 0], 0.0)


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
    assert_equal(type(p), np.matrix)
    p = quijify(x, 'dop', sparse=True)
    assert_equal(type(p), sp.csr_matrix)
    assert_equal(p.dtype, np.complex)
    assert_equal(p.nnz, 2)


def test_quijify_sparse_convert_to_dop():
    x = [1, 0, 9e-16, 0, 3j]
    p = quijify(x, 'ket', sparse=True)
    q = quijify(p, 'dop', sparse=True)
    assert_equal(q.shape, (5, 5))
    assert_equal(q.nnz, 9)
    assert_almost_equal(q[4, 4], 9.)
    q = quijify(p, 'dop', sparse=True, normalized=True)
    assert_almost_equal(tr(q), 1.)
