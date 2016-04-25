from pytest import fixture
import numpy as np
from numpy.testing import assert_allclose
import scipy.sparse as sp
from quijy import rand_matrix, rand_ket
from quijy.accel import (
    matrixify,
    realify,
    issparse,
    isket,
    isop,
    isbra,
    isherm,
    mul,
    dot,
    vdot,
    rdot,
    ldmul,
    rdmul,
    outer,
    kron_dense,
    kron_dense_big,
    kron,
    kronpow,
    explt,
    idot,
    calc_dot_type,
    calc_dot_weight_func_out,
)


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
    def test_kron_dense(self):
        a = rand_matrix(3)
        b = rand_matrix(3)
        c = kron_dense(a, b)
        npc = np.kron(a, b)
        assert_allclose(c, npc)
        assert isinstance(c, np.matrix)

    def test_kron_dense_big(self):
        a = rand_matrix(3)
        b = rand_matrix(3)
        c = kron_dense_big(a, b)
        npc = np.kron(a, b)
        assert_allclose(c, npc)
        assert isinstance(c, np.matrix)

    def test_kron_multi_args(self):
        a = rand_matrix(3)
        b = rand_matrix(3)
        c = rand_matrix(3)
        assert_allclose(kron(a), a)
        assert_allclose(kron(a, b, c),
                        np.kron(np.kron(a, b), c))

    def test_kron_mixed_types(self):
        a = rand_ket(4)
        b = rand_ket(4, sparse=True)
        assert_allclose(kron(a, b).A,
                        (sp.kron(a, b, 'csr')).A)
        assert_allclose(kron(b, b).A,
                        (sp.kron(b, b, 'csr')).A)

    def test_kronpow(self):
        a = rand_matrix(2)
        b = a & a & a
        c = kronpow(a, 3)
        assert_allclose(b, c)


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
