import numpy as np
from pytest import raises
from numpy.testing import assert_allclose
from quijy.gen import basis_vec, sig
from quijy.solve import eigvals


def test_basis_vec():
    x = basis_vec(1, 2)
    assert_allclose(x, np.matrix([[0.], [1.]]))
    x = basis_vec(1, 2, qtype='b')
    assert_allclose(x, np.matrix([[0., 1.]]))


def test_basis_vec_sparse():
    x = basis_vec(4, 100, sparse=True)
    assert x[4, 0] == 1.
    assert x.nnz == 1
    assert x.dtype == complex


def test_sig_dim2():
    for dir in (1, 'x', 'X',
                2, 'y', 'Y',
                3, 'z', 'Z'):
        x = sig(dir)
        assert_allclose(eigvals(x), [-1, 1])


def test_sig_dim3():
    for dir in (1, 'x', 'X',
                2, 'y', 'Y',
                3, 'z', 'Z'):
        x = sig(dir, dim=3)
        assert_allclose(eigvals(x), [-1, 0, 1],
                        atol=1e-15)


def test_sig_bad_dim():
    with raises(KeyError):
        sig('x', 4)


def test_sig_bad_dir():
    with raises(KeyError):
        sig('w', 2)
