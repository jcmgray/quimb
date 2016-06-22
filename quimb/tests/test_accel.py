from pytest import fixture, mark
import numpy as np
from numpy.testing import assert_allclose
import scipy.sparse as sp
from .. import rand_matrix, rand_ket
from ..accel import (
    matrixify, realify,
    issparse, isket, isop, isbra, isherm,
    mul, dot, vdot, rdot, ldmul, rdmul, outer,
    kron_dense, kron_dense_big, kron_sparse, kron, kronpow,
    explt,
    idot, calc_dot_type, calc_dot_weight_func_out,
)


sparse_types = ("csr", "bsr", "csc", "coo")


@fixture
def test_objs():
    d = 5
    od1 = rand_matrix(d)
    od2 = rand_matrix(d)
    os1 = rand_matrix(d, sparse=True, density=0.5)
    os2 = rand_matrix(d, sparse=True, density=0.5)
    kd1 = rand_ket(d)
    kd2 = rand_ket(d)
    ld = np.random.randn(d) + 1.0j * np.random.randn(d)
    return od1, od2, os1, os2, kd1, kd2, ld


@fixture
def d1():
    return rand_matrix(3)


@fixture
def d2():
    return rand_matrix(3)


@fixture
def d3():
    return rand_matrix(3)


@fixture
def s1():
    return rand_matrix(3, sparse=True, density=0.5)


@fixture
def s2():
    return rand_matrix(3, sparse=True, density=0.5)


@fixture
def s1nnz():
    return rand_matrix(3, sparse=True, density=0.75)


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
        a = foo(1, 1e-15)
        assert a.real == 1
        assert a.imag == 1e-15

        @realify
        def foo(a, b):
            return a + 1j * b
        a = foo(1, 1e-15)
        assert a.real == 1
        assert a.imag == 0


class TestShapes:
    def test_sparse(self):
        x = np.matrix([[1], [0]])
        assert not issparse(x)
        x = sp.csr_matrix(x)
        assert issparse(x)

    def test_ket(self):
        x = np.matrix([[1], [0]])
        assert(isket(x))
        assert(not isbra(x))
        assert(not isop(x))
        x = sp.csr_matrix(x)
        assert(isket(x))
        assert(not isbra(x))
        assert(not isop(x))

    def test_bra(self):
        x = np.matrix([[1, 0]])
        assert(not isket(x))
        assert(isbra(x))
        assert(not isop(x))
        x = sp.csr_matrix(x)
        assert(not isket(x))
        assert(isbra(x))
        assert(not isop(x))

    def test_op(self):
        x = np.matrix([[1, 0], [0, 0]])
        assert(not isket(x))
        assert(not isbra(x))
        assert(isop(x))
        x = sp.csr_matrix(x)
        assert(not isket(x))
        assert(not isbra(x))
        assert(isop(x))

    def test_isherm(self):
        a = np.matrix([[1.0, 2.0 + 3.0j],
                       [2.0 - 3.0j, 1.0]])
        assert(isherm(a))
        a = np.matrix([[1.0, 2.0 - 3.0j],
                       [2.0 - 3.0j, 1.0]])
        assert(not isherm(a))

    def test_isherm_sparse(self):
        a = sp.csr_matrix([[1.0, 2.0 + 3.0j],
                           [2.0 - 3.0j, 1.0]])
        assert(isherm(a))
        a = sp.csr_matrix([[1.0, 2.0 - 3.0j],
                           [2.0 - 3.0j, 1.0]])
        assert(not isherm(a))


class TestMul:
    def test_mul_dense_same(self, test_objs):
        a, b, _, _, _, _, _ = test_objs
        ca = mul(a, b)
        assert isinstance(ca, np.matrix)
        cn = np.multiply(a, b)
        assert_allclose(ca, cn)

    def test_mul_broadcast(self, test_objs):
        a, _, _, _, b, _, _ = test_objs
        ca = mul(a, b)
        assert isinstance(ca, np.matrix)
        cn = np.multiply(a, b)
        assert_allclose(ca, cn)
        ca = mul(a.H, b)
        assert isinstance(ca, np.matrix)
        cn = np.multiply(a.H, b)
        assert_allclose(ca, cn)

    def test_mul_sparse(self, test_objs):
        _, _, a, b, _, _, _ = test_objs
        cq = mul(a, b)
        cn = a.A * b.A
        assert issparse(cq)
        assert_allclose(cq.A, cn)
        cq = mul(b.A, a)
        cn = b.A * a.A
        assert issparse(cq)
        assert_allclose(cq.A, cn)

    def test_mul_sparse_broadcast(self, test_objs):
        _, _, a, _, b, _, _ = test_objs
        ca = mul(a, b)
        cn = np.multiply(a.A, b)
        assert_allclose(ca.A, cn)
        ca = mul(a.H, b)
        cn = np.multiply(a.H.A, b)
        assert_allclose(ca.A, cn)


class TestDot:
    def test_dot_matrix(self, test_objs):
        a, b, _, _, _, _, _ = test_objs
        ca = dot(a, b)
        assert isinstance(ca, np.matrix)
        cn = a @ b
        assert_allclose(ca, cn)

    def test_dot_ket(self, test_objs):
        a, _, _, _, b, _, _ = test_objs
        ca = dot(a, b)
        assert isinstance(ca, np.matrix)
        cn = a @ b
        assert_allclose(ca, cn)

    def test_dot_sparse_sparse(self, test_objs):
        _, _, a, b, _, _, _ = test_objs
        cq = dot(a, b)
        cn = a @ b
        assert issparse(cq)
        assert_allclose(cq.A, cn.A)

    def test_dot_sparse_dense(self, test_objs):
        _, _, a, _, b, _, _ = test_objs
        cq = dot(a, b)
        cn = a @ b
        assert not issparse(cq)
        assert_allclose(cq.A, cn)

    def test_dot_sparse_dense_ket(self, test_objs):
        _, _, a, _, b, _, _ = test_objs
        cq = dot(a, b)
        cn = a @ b
        assert not issparse(cq)
        assert isket(cq)
        assert_allclose(cq.A, cn)


class TestAccelVdot:
    def test_accel_vdot(self, test_objs):
        _, _, _, _, a, b, _ = test_objs
        ca = vdot(a, b)
        cn = (a.H @ b)[0, 0]
        assert_allclose(ca, cn)


class TestAccelRdot:
    def test_accel_rdot(self, test_objs):
        _, _, _, _, a, b, _ = test_objs
        cq = rdot(a.H, b)
        cn = (a.H @ b)[0, 0]
        assert_allclose(cq, cn)


class TestFastDiagMul:
    def test_ldmul_small(self, test_objs):
        mat, _, _, _, _, _, vec = test_objs
        a = ldmul(vec, mat)
        b = np.diag(vec) @ mat
        assert isinstance(a, np.matrix)
        assert_allclose(a, b)

    def test_ldmul_large(self):
        vec = np.random.randn(501)
        mat = rand_matrix(501)
        a = ldmul(vec, mat)
        b = np.diag(vec) @ mat
        assert isinstance(a, np.matrix)
        assert_allclose(a, b)

    def test_ldmul_sparse(self, test_objs):
        _, _, mat, _, _, _, vec = test_objs
        assert issparse(mat)
        a = ldmul(vec, mat)
        b = np.diag(vec) @ mat.A
        assert issparse(a)
        assert_allclose(a.A, b)

    def test_rdmul_small(self, test_objs):
        mat, _, _, _, _, _, vec = test_objs
        a = rdmul(mat, vec)
        b = mat @ np.diag(vec)
        assert isinstance(a, np.matrix)
        assert_allclose(a, b)

    def test_rdmul_large(self):
        vec = np.random.randn(501)
        mat = rand_matrix(501)
        a = rdmul(mat, vec)
        b = mat @ np.diag(vec)
        assert isinstance(a, np.matrix)
        assert_allclose(a, b)

    def test_rdmul_sparse(self, test_objs):
        _, _, mat, _, _, _, vec = test_objs
        a = rdmul(mat, vec)
        b = mat.A @ np.diag(vec)
        assert issparse(a)
        assert_allclose(a.A, b)


class TestOuter:
    def test_outer_ket_ket(self, test_objs):
        _, _, _, _, a, b, _ = test_objs
        c = outer(a, b)
        assert isinstance(c, np.matrix)
        d = np.multiply(a, b.T)
        assert_allclose(c, d)

    def test_outer_ket_bra(self, test_objs):
        _, _, _, _, a, b, _ = test_objs
        c = outer(a, b.H)
        assert isinstance(c, np.matrix)
        d = np.multiply(a, b.H)
        assert_allclose(c, d)

    def test_outer_bra_ket(self, test_objs):
        _, _, _, _, a, b, _ = test_objs
        c = outer(a.H, b)
        assert isinstance(c, np.matrix)
        d = np.multiply(a.H.T, b.T)
        assert_allclose(c, d)

    def test_outer_bra_bra(self, test_objs):
        _, _, _, _, a, b, _ = test_objs
        c = outer(a.H, b.H)
        assert isinstance(c, np.matrix)
        d = np.multiply(a.H.T, b.H)
        assert_allclose(c, d)


class TestKron:
    @mark.parametrize("func", [kron_dense, kron_dense_big])
    def test_kron_dense(self, d1, d2, func):
        x = func(d1, d2)
        assert d1.shape == (3, 3)
        assert d2.shape == (3, 3)
        xn = np.kron(d1, d2)
        assert_allclose(x, xn)
        assert isinstance(x, np.matrix)

    def test_kron_multi_args(self, d1, d2, d3):
        assert_allclose(kron(d1), d1)
        assert_allclose(kron(d1, d2, d3),
                        np.kron(np.kron(d1, d2), d3))

    def test_kron_mixed_types(self, d1, s1):
        assert_allclose(kron(d1, s1).A,
                        (sp.kron(d1, s1, 'csr')).A)
        assert_allclose(kron(s1, s1).A,
                        (sp.kron(s1, s1, 'csr')).A)

    def test_kronpow(self, d1):
        x = d1 & d1 & d1
        y = kronpow(d1, 3)
        assert_allclose(x, y)


class TestKronSparseFormats:
    def test_sparse_sparse_auto(self, s1):
        c = kron_sparse(s1, s1)
        assert c.format == 'csr'

    def test_sparse_dense_auto(self, s1, d1):
        c = kron_sparse(s1, d1)
        assert c.format == 'bsr'

    def test_dense_sparse_auto(self, s1, d1):
        c = kron_sparse(d1, s1)
        assert c.format == 'csr'

    def test_sparse_sparsennz(self, s1, s1nnz):
        c = kron_sparse(s1, s1nnz)
        assert c.format == 'csr'

    @mark.parametrize("sformat", sparse_types)
    def test_sparse_sparse_to_sformat(self, s1, sformat):
        c = kron_sparse(s1, s1, sformat=sformat)
        assert c.format == sformat

    @mark.parametrize("sformat", (None,) + sparse_types)
    def test_many_args_dense_last(self, s1, s2, d1, sformat):
        c = kron(s1, s2, d1, sformat=sformat)
        assert c.format == (sformat if sformat is not None else "bsr")

    @mark.parametrize("sformat", (None,) + sparse_types)
    def test_many_args_dense_not_last(self, s1, s2, d1, sformat):
        c = kron(d1, s1, s2, sformat=sformat)
        assert c.format == (sformat if sformat is not None else "csr")
        c = kron(s1, d1, s2, sformat=sformat)
        assert c.format == (sformat if sformat is not None else "csr")

    @mark.parametrize("sformat", (None,) + sparse_types)
    def test_many_args_dense_last_coo_construct(self, s1, s2, d1, sformat):
        c = kron(s1, s2, d1, sformat=sformat, coo_construct=True)
        assert c.format == (sformat if sformat is not None else "csr")

    @mark.parametrize("sformat", (None,) + sparse_types)
    def test_many_args_dense_not_last_coo_construct(self, s1, s2, d1, sformat):
        c = kron(s1, d1, s2, sformat=sformat, coo_construct=True)
        assert c.format == (sformat if sformat is not None else "csr")
        c = kron(d1, s1, s2, sformat=sformat, coo_construct=True)
        assert c.format == (sformat if sformat is not None else "csr")


class TestCalcDotType:
    def test_scalar(self):
        assert calc_dot_type(1) == 'c'
        assert calc_dot_type(1.) == 'c'
        assert calc_dot_type(1.0j) == 'c'

    def test_1d_array(self, test_objs):
        _, _, _, _, _, _, l = test_objs
        assert calc_dot_type(l) == 'l'

    def test_ket(self, test_objs):
        _, _, _, _, k, _, _ = test_objs
        assert calc_dot_type(k) == 'k'

    def test_bra(self, test_objs):
        _, _, _, _, k, _, _ = test_objs
        assert calc_dot_type(k.H) == 'b'

    def test_op(self, test_objs):
        od, _, os, _, _, _, _ = test_objs
        assert calc_dot_type(od) == 'o'
        assert calc_dot_type(os) == 'os'


class TestCalcDotWeightFuncOut:
    def test_ket(self):
        for z, w in zip(("k", "b", "o", "os", "l", "c"),
                        (11, 12, 21, 23, 25, 32)):
            assert calc_dot_weight_func_out(z, "k")[0] == w


class TestExplt:
    def test_small(self):
        l = np.random.randn(3)
        en = np.exp(-1.0j * l * 7)
        eq = explt(l, 7)
        assert_allclose(eq, en)


# --------------------------------------------------------------------------- #
# Test Intelligent chaining of operations                                     #
# --------------------------------------------------------------------------- #

class TestIdot:
    def test_multiarg_mats(self, test_objs):
        od1, od2, os1, os2, kd1, kd2, ld = test_objs
        dq = idot(kd1.H, od1, os2, os1, ld, od2, kd2)
        dn = (kd1.H @ od1 @ os2.A @ os1.A @ np.diag(ld) @ od2 @ kd2)[0, 0]
        assert_allclose(dq, dn)

    def test_multiarg_vecs(self):
        a, b, c = rand_matrix(5), rand_matrix(5), rand_ket(5)
        d = idot(a, b, c)
        assert isinstance(d, np.matrix)
        assert_allclose(d, a @ b @ c)

    def test_multiarg_closed(self):
        a, b, c = rand_matrix(5), rand_matrix(5), rand_ket(5)
        d = idot(c.H, a, b, c)
        assert np.isscalar(d)
        assert_allclose(d, c.H @ a @ b @ c)

    # 0, 1, 2, 3, 4, 5, 6
    # a, b, c, d, e, f, g
    #   m, n, o, p, q, r
    #   0, 1, 2, 3, 4, 5

    # 0, 1, 2, 3, 4, 5
    # a, b, c, X, f, g
    #   m, n, Y, Z, r
    #   0, 1, 2, 3, 4
