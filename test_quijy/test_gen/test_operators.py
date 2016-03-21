from pytest import raises
import numpy as np
from numpy.testing import assert_allclose
from quijy.core import issparse
from quijy.solve import eigvals
from quijy.gen.operators import sig, controlled


class TestSig:
    def test_sig_dim2(self):
        for dir in (1, 'x', 'X',
                    2, 'y', 'Y',
                    3, 'z', 'Z'):
            x = sig(dir)
            assert_allclose(eigvals(x), [-1, 1])

    def test_sig_dim3(self):
        for dir in (1, 'x', 'X',
                    2, 'y', 'Y',
                    3, 'z', 'Z'):
            x = sig(dir, dim=3)
            assert_allclose(eigvals(x), [-1, 0, 1],
                            atol=1e-15)

    def test_sig_bad_dim(self):
        with raises(KeyError):
            sig('x', 4)

    def test_sig_bad_dir(self):
        with raises(KeyError):
            sig('w', 2)


class TestControlledZ:
    def test_controlled_z_dense(self):
        cz = controlled('z')
        assert_allclose(cz, np.diag([1, 1, 1, -1]))

    def test_controlled_z_sparse(self):
        cz = controlled('z', sparse=True)
        assert(issparse(cz))
        assert_allclose(cz.A, np.diag([1, 1, 1, -1]))
