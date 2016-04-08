import numpy as np
from numpy.testing import assert_allclose
from quijy import rand_matrix, rand_ket
from quijy.accel import (
    matrixify,
    realify,
    mul_dense,
    dot_dense,
    idot,
    vdot,
    ldmul,
    rdmul,
    outer,
)


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


class TestMulDense:
    def test_mul_same(self):
        a = rand_matrix(5)
        b = rand_matrix(5)
        ca = mul_dense(a, b)
        assert isinstance(ca, np.matrix)
        cn = np.multiply(a, b)
        assert_allclose(ca, cn)

    def test_mul_broadcast(self):
        a = rand_matrix(5)
        b = rand_ket(5)
        ca = mul_dense(a, b)
        assert isinstance(ca, np.matrix)
        cn = np.multiply(a, b)
        assert_allclose(ca, cn)
        ca = mul_dense(a.H, b)
        assert isinstance(ca, np.matrix)
        cn = np.multiply(a.H, b)
        assert_allclose(ca, cn)


class TestDotDense:
    def test_dot_matrix(self):
        a = rand_matrix(5)
        b = rand_matrix(5)
        ca = dot_dense(a, b)
        assert isinstance(ca, np.matrix)
        cn = a @ b
        assert_allclose(ca, cn)

    def test_dot_ket(self):
        a = rand_matrix(5)
        b = rand_ket(5)
        ca = dot_dense(a, b)
        assert isinstance(ca, np.matrix)
        cn = a @ b
        assert_allclose(ca, cn)


class TestIdot:
    def test_multiarg_mats(self):
        a, b, c = rand_matrix(5), rand_matrix(5), rand_matrix(5)
        d = idot(a, b, c)
        assert isinstance(d, np.matrix)
        assert_allclose(d, a @ b @ c)

    def test_multiarg_vecs(self):
        a, b, c = rand_matrix(5), rand_matrix(5), rand_ket(5)
        d = idot(a, b, c)
        assert isinstance(d, np.matrix)
        assert_allclose(d, a @ b @ c)

    def test_multiarg_closed(self):
        a, b, c = rand_matrix(5), rand_matrix(5), rand_ket(5)
        d = idot(c.H, a, b, c)
        assert isinstance(d, np.matrix)
        assert_allclose(d, c.H @ a @ b @ c)


class TestAccelVdot:
    def test_accel_vdot(self):
        a = rand_ket(5)
        b = rand_ket(5)
        ca = vdot(a, b)
        cn = (a.H @ b)[0, 0]
        assert_allclose(ca, cn)


class TestFastDiagMul:
    def test_ldmul_small(self):
        n = 4
        vec = np.random.randn(2**n)
        mat = rand_matrix(2**n)
        a = ldmul(vec, mat)
        b = np.diag(vec) @ mat
        assert isinstance(a, np.matrix)
        assert_allclose(a, b)

    def test_ldmul_large(self):
        n = 9
        vec = np.random.randn(2**n)
        mat = rand_matrix(2**n)
        a = ldmul(vec, mat)
        b = np.diag(vec) @ mat
        assert isinstance(a, np.matrix)
        assert_allclose(a, b)

    def test_rdmul_small(self):
        n = 4
        vec = np.random.randn(2**n)
        mat = rand_matrix(2**n)
        a = rdmul(mat, vec)
        b = mat @ np.diag(vec)
        assert isinstance(a, np.matrix)
        assert_allclose(a, b)

    def test_rdmul_large(self):
        n = 9
        vec = np.random.randn(2**n)
        mat = rand_matrix(2**n)
        a = rdmul(mat, vec)
        b = mat @ np.diag(vec)
        assert isinstance(a, np.matrix)
        assert_allclose(a, b)


class TestOuter:
    def test_outer_ket_ket(self):
        a, b = rand_ket(5), rand_ket(5)
        c = outer(a, b)
        assert isinstance(c, np.matrix)
        d = np.multiply(a, b.T)
        assert_allclose(c, d)

    def test_outer_ket_bra(self):
        a, b = rand_ket(5), rand_ket(5)
        c = outer(a, b.H)
        assert isinstance(c, np.matrix)
        d = np.multiply(a, b.H)
        assert_allclose(c, d)

    def test_outer_bra_ket(self):
        a, b = rand_ket(5), rand_ket(5)
        c = outer(a.H, b)
        assert isinstance(c, np.matrix)
        d = np.multiply(a.H.T, b.T)
        assert_allclose(c, d)

    def test_outer_bra_bra(self):
        a, b = rand_ket(5), rand_ket(5)
        c = outer(a.H, b.H)
        assert isinstance(c, np.matrix)
        d = np.multiply(a.H.T, b.H)
        assert_allclose(c, d)
