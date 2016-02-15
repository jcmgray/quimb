import numpy as np
import scipy.sparse as sp
from nose.tools import ok_, eq_, assert_almost_equal
from numpy.testing import assert_allclose
from quijy.gen import *


def test_basis_vec():
    x = basis_vec(1, 2)
    assert_allclose(x, np.matrix([[0.],
                                  [1.]]))
    x = basis_vec(1, 2, qtype='b')
    assert_allclose(x, np.matrix([[0., 1.]]))

def test_basis_vec_sparse():
    x = basis_vec(4, 100, sparse=True)
    eq_(x[4, 0], 1.)
    eq_(x.nnz, 1)
    eq_(x.dtype, complex)
