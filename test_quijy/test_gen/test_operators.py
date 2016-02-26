from pytest import raises
from numpy.testing import assert_allclose
from quijy.solve import eigvals
from quijy.gen import sig


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
