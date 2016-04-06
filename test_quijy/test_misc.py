import numpy as np
from quijy.misc import matrixify, realify


class TestMatrixify:
    def test_matrixify(self):
        def foo(n):
            return np.random.randn(n, n)
        a = foo(2)
        assert not isinstance(a, np.matrix)

        @matrixify
        def foo(n):
            return np.random.randn(n, n)
        a = foo(2)
        assert isinstance(a, np.matrix)


class TestRealify:
    def test_realify(self):
        def foo(a, b):
            return a + 1j * b
        a = foo(1e15, 1)
        assert a.real == 1e15
        assert a.imag == 1

        @realify
        def foo(a, b):
            return a + 1j * b
        a = foo(1e15, 1)
        assert a.real == 1e15
        assert a.imag == 0
