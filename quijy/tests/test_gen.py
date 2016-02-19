import numpy as np
from nose.tools import eq_
from numpy.testing import assert_allclose
from quijy.gen import basis_vec


def test_basis_vec():
    x = basis_vec(1, 2)
    assert_allclose(x, np.matrix([[0.], [1.]]))
    x = basis_vec(1, 2, qtype='b')
    assert_allclose(x, np.matrix([[0., 1.]]))


def test_basis_vec_sparse():
    x = basis_vec(4, 100, sparse=True)
    eq_(x[4, 0], 1.)
    eq_(x.nnz, 1)
    eq_(x.dtype, complex)
