from pytest import fixture, mark, raises
import numpy as np
from numpy.testing import assert_allclose
import scipy.sparse as sp

from quimb import (
    qu,
    rand_matrix,
    rand_ket,
)
from quimb.core import (
    qarray,
    ensure_qarray,
    issparse,
    isdense,
    isket,
    isop,
    isbra,
    isvec,
    isherm,
    mul,
    dot,
    vdot,
    rdot,
    ldmul,
    rdmul,
    outer,
    explt,
    make_immutable,
    realify,
    dot_sparse,
    par_dot_csr_matvec,
    kron_dense,
    kron_sparse,
)
from quimb.core import kron, kronpow


# ----------------------------- FIXTURES ------------------------------------ #

_SPARSE_FORMATS = ("csr", "bsr", "csc", "coo")
_TEST_SZ = 4


@fixture
def mat_d():
    return rand_matrix(_TEST_SZ)


@fixture
def mat_d2():
    return rand_matrix(_TEST_SZ)


@fixture
def mat_d3():
    return rand_matrix(_TEST_SZ)


@fixture
def mat_s():
    return rand_matrix(_TEST_SZ, sparse=True, density=0.5)


@fixture
def mat_s2():
    return rand_matrix(_TEST_SZ, sparse=True, density=0.5)


@fixture
def ket_d():
    return rand_ket(_TEST_SZ)


@fixture
def ket_d2():
    return rand_ket(_TEST_SZ)


@fixture
def l1d():
    return np.random.randn(_TEST_SZ) + 1.0j * np.random.randn(_TEST_SZ)


@fixture
def mat_s_nnz():
    return rand_matrix(_TEST_SZ, sparse=True, density=0.75)


# --------------------------------------------------------------------------- #
#                                  TESTS                                      #
# --------------------------------------------------------------------------- #

class TestMakeImmutable():
    def test_dense(self):
        mat = qu([[1, 2], [3, 4]])
        make_immutable(mat)
        with raises(ValueError):
            mat[-1, -1] = 1

    @mark.parametrize("stype", _SPARSE_FORMATS)
    def test_sparse(self, stype):
        mat = qu([[1, 2], [3, 4]], stype=stype)
        make_immutable(mat)
        if stype in {'csr', 'csc'}:
            with raises(ValueError):
                mat[-1, -1] = 1


class TestEnsureQarray:
    def test_ensure_qarray(self):
        def foo(n):
            return np.random.randn(n, n)
        a = foo(2)
        assert not isinstance(a, qarray)

        @ensure_qarray
        def foo2(n):
            return np.random.randn(n, n)
        a = foo2(2)
        assert isinstance(a, qarray)


class TestRealify:
    def test_realify(self):
        def foo(a, b):
            return a + 1j * b
        a = foo(1, 1e-15)
        assert a.real == 1
        assert a.imag == 1e-15

        @realify
        def foo2(a, b):
            return a + 1j * b
        a = foo2(1, 1e-15)
        assert a.real == 1
        assert a.imag == 0

    def test_wrong_type(self):
        @realify
        def foo(a, b):
            return str(a) + str(b)
        assert foo(1, 2) == '12'


class TestShapes:
    def test_sparse(self):
        x = np.array([[1], [0]])
        assert not issparse(x)
        assert isdense(x)
        x = sp.csr_matrix(x)
        assert issparse(x)

    def test_ket(self):
        x = np.array([[1], [0]])
        assert(isket(x))
        assert(not isbra(x))
        assert(not isop(x))
        assert isvec(x)
        x = sp.csr_matrix(x)
        assert(isket(x))
        assert isvec(x)
        assert(not isbra(x))
        assert(not isop(x))

    def test_bra(self):
        x = np.array([[1, 0]])
        assert(not isket(x))
        assert(isbra(x))
        assert(not isop(x))
        assert isvec(x)
        x = sp.csr_matrix(x)
        assert(not isket(x))
        assert(isbra(x))
        assert(not isop(x))
        assert isvec(x)

    def test_op(self):
        x = np.array([[1, 0], [0, 0]])
        assert(not isket(x))
        assert(not isbra(x))
        assert(isop(x))
        assert (not isvec(x))
        x = sp.csr_matrix(x)
        assert(not isket(x))
        assert(not isbra(x))
        assert(isop(x))
        assert (not isvec(x))

    def test_isherm(self):
        a = np.array([[1.0, 2.0 + 3.0j],
                      [2.0 - 3.0j, 1.0]])
        assert(isherm(a))
        a = np.array([[1.0, 2.0 - 3.0j],
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
    def test_mul_dense_same(self, mat_d, mat_d2):
        ca = mul(mat_d, mat_d2)
        assert isinstance(ca, qarray)
        cn = np.multiply(mat_d, mat_d2)
        assert_allclose(ca, cn)

    def test_mul_broadcast(self, mat_d, ket_d):
        ca = mul(mat_d, ket_d)
        assert isinstance(ca, qarray)
        cn = np.multiply(mat_d, ket_d)
        assert_allclose(ca, cn)
        ca = mul(mat_d.H, ket_d)
        assert isinstance(ca, qarray)
        cn = np.multiply(mat_d.H, ket_d)
        assert_allclose(ca, cn)

    def test_mul_sparse(self, mat_s, mat_s2):
        cq = mul(mat_s, mat_s2)
        cn = mat_s.A * mat_s2.A
        assert issparse(cq)
        assert_allclose(cq.A, cn)
        cq = mul(mat_s2.A, mat_s)
        cn = mat_s2.A * mat_s.A
        assert issparse(cq)
        assert_allclose(cq.A, cn)

    def test_mul_sparse_broadcast(self, mat_s, ket_d):
        ca = mul(mat_s, ket_d)
        cn = np.multiply(mat_s.A, ket_d)
        assert_allclose(ca.A, cn)
        ca = mul(mat_s.H, ket_d)
        cn = np.multiply(mat_s.H.A, ket_d)
        assert_allclose(ca.A, cn)


class TestDot:
    def test_dot_matrix(self, mat_d, mat_d2):
        ca = dot(mat_d, mat_d2)
        assert isinstance(ca, qarray)
        cn = mat_d @ mat_d2
        assert_allclose(ca, cn)

    def test_dot_ket(self, mat_d, ket_d):
        ca = dot(mat_d, ket_d)
        assert isinstance(ca, qarray)
        cn = mat_d @ ket_d
        assert_allclose(ca, cn)

    def test_dot_sparse_sparse(self, mat_s, mat_s2):
        cq = dot(mat_s, mat_s2)
        cn = mat_s @ mat_s2
        assert issparse(cq)
        assert_allclose(cq.A, cn.A)

    def test_dot_sparse_dense(self, mat_s, ket_d):
        cq = dot(mat_s, ket_d)
        assert isinstance(cq, qarray)
        cq = mat_s @ ket_d
        assert isinstance(cq, qarray)
        cn = mat_s._mul_vector(ket_d)
        assert not issparse(cq)
        assert isdense(cq)
        assert_allclose(cq.A.ravel(), cn)

    def test_dot_sparse_dense_ket(self, mat_s, ket_d):
        cq = dot(mat_s, ket_d)
        cn = mat_s @ ket_d
        assert not issparse(cq)
        assert isdense(cq)
        assert isket(cq)
        assert_allclose(cq.A, cn)

    def test_par_dot_csr_matvec(self, mat_s, ket_d):
        x = par_dot_csr_matvec(mat_s, ket_d)
        y = dot_sparse(mat_s, ket_d)
        assert x.dtype == complex
        assert x.shape == (_TEST_SZ, 1)
        assert isinstance(x, qarray)
        assert_allclose(x, y)

    def test_par_dot_csr_matvec_Array(self, mat_s, ket_d):
        x = par_dot_csr_matvec(mat_s, np.asarray(ket_d).reshape(-1))
        y = dot_sparse(mat_s, ket_d)
        assert x.dtype == complex
        assert x.shape == (_TEST_SZ,)
        assert_allclose(y, x.reshape(-1, 1))


class TestAccelVdot:
    def test_accel_vdot(self, ket_d, ket_d2):
        ca = vdot(ket_d, ket_d2)
        cn = (ket_d.H @ ket_d2)[0, 0]
        assert_allclose(ca, cn)


class TestAccelRdot:
    def test_accel_rdot(self, ket_d, ket_d2):
        cq = rdot(ket_d.H, ket_d2)
        cn = (ket_d.H @ ket_d2)[0, 0]
        assert_allclose(cq, cn)


class TestFastDiagMul:
    def test_ldmul_small(self, mat_d, l1d):
        a = ldmul(l1d, mat_d)
        b = np.diag(l1d) @ mat_d
        assert isinstance(a, qarray)
        assert_allclose(a, b)

    def test_ldmul_large(self):
        vec = np.random.randn(501)
        mat = rand_matrix(501)
        a = ldmul(vec, mat)
        b = np.diag(vec) @ mat
        assert isinstance(a, qarray)
        assert_allclose(a, b)

    def test_ldmul_sparse(self, mat_s, l1d):
        assert issparse(mat_s)
        a = ldmul(l1d, mat_s)
        b = np.diag(l1d) @ mat_s.A
        assert issparse(a)
        assert_allclose(a.A, b)

    def test_rdmul_small(self, mat_d, l1d):
        a = rdmul(mat_d, l1d)
        b = mat_d @ np.diag(l1d)
        assert isinstance(a, qarray)
        assert_allclose(a, b)

    def test_rdmul_large(self):
        vec = np.random.randn(501)
        mat = rand_matrix(501)
        a = rdmul(mat, vec)
        b = mat @ np.diag(vec)
        assert isinstance(a, qarray)
        assert_allclose(a, b)

    def test_rdmul_sparse(self, mat_s, l1d):
        a = rdmul(mat_s, l1d)
        b = mat_s.A @ np.diag(l1d)
        assert issparse(a)
        assert_allclose(a.A, b)


class TestOuter:
    def test_outer_ket_ket(self, ket_d, ket_d2):
        c = outer(ket_d, ket_d2)
        assert isinstance(c, qarray)
        d = np.multiply(ket_d, ket_d2.T)
        assert_allclose(c, d)

    def test_outer_ket_bra(self, ket_d, ket_d2):
        c = outer(ket_d, ket_d2.H)
        assert isinstance(c, qarray)
        d = np.multiply(ket_d, ket_d2.H)
        assert_allclose(c, d)

    def test_outer_bra_ket(self, ket_d, ket_d2):
        c = outer(ket_d.H, ket_d2)
        assert isinstance(c, qarray)
        d = np.multiply(ket_d.H.T, ket_d2.T)
        assert_allclose(c, d)

    def test_outer_bra_bra(self, ket_d, ket_d2):
        c = outer(ket_d.H, ket_d2.H)
        assert isinstance(c, qarray)
        d = np.multiply(ket_d.H.T, ket_d2.H)
        assert_allclose(c, d)


class TestExplt:
    def test_small(self):
        evals = np.random.randn(3)
        en = np.exp(-1.0j * evals * 7)
        eq = explt(evals, 7)
        assert_allclose(eq, en)


# --------------------------------------------------------------------------- #
# Kronecker (tensor) product tests                                            #
# --------------------------------------------------------------------------- #

class TestKron:
    @mark.parametrize("big", [False, True])
    def test_kron_dense(self, mat_d, mat_d2, big):
        x = kron_dense(mat_d, mat_d2, par_thresh=0 if big else 1e100)
        assert mat_d.shape == (_TEST_SZ, _TEST_SZ)
        assert mat_d2.shape == (_TEST_SZ, _TEST_SZ)
        xn = np.kron(mat_d, mat_d2)
        assert_allclose(x, xn)
        assert isinstance(x, qarray)

    def test_kron_multi_args(self, mat_d, mat_d2, mat_d3):
        assert_allclose(kron(mat_d), mat_d)
        assert_allclose(kron(mat_d, mat_d2, mat_d3),
                        np.kron(np.kron(mat_d, mat_d2), mat_d3))

    def test_kron_mixed_types(self, mat_d, mat_s):
        assert_allclose(kron(mat_d, mat_s).A,
                        (sp.kron(mat_d, mat_s, 'csr')).A)
        assert_allclose(kron(mat_s, mat_s).A,
                        (sp.kron(mat_s, mat_s, 'csr')).A)


class TestKronSparseFormats:
    def test_sparse_sparse_auto(self, mat_s):
        c = kron_sparse(mat_s, mat_s)
        assert c.format == 'csr'

    def test_sparse_dense_auto(self, mat_s, mat_d):
        c = kron_sparse(mat_s, mat_d)
        assert c.format == 'bsr'

    def test_dense_sparse_auto(self, mat_s, mat_d):
        c = kron_sparse(mat_d, mat_s)
        assert c.format == 'csr'

    def test_sparse_sparsennz(self, mat_s, mat_s_nnz):
        c = kron_sparse(mat_s, mat_s_nnz)
        assert c.format == 'csr'

    @mark.parametrize("stype", _SPARSE_FORMATS)
    def test_sparse_sparse_to_sformat(self, mat_s, stype):
        c = kron_sparse(mat_s, mat_s, stype=stype)
        assert c.format == stype

    @mark.parametrize("stype", (None,) + _SPARSE_FORMATS)
    def test_many_args_dense_last(self, mat_s, mat_s2, mat_d, stype):
        c = kron(mat_s, mat_s2, mat_d, stype=stype)
        assert c.format == (stype if stype is not None else "bsr")

    @mark.parametrize("stype", (None,) + _SPARSE_FORMATS)
    def test_many_args_dense_not_last(self, mat_s, mat_s2, mat_d, stype):
        c = kron(mat_d, mat_s, mat_s2, stype=stype)
        assert c.format == (stype if stype is not None else "csr")
        c = kron(mat_s, mat_d, mat_s2, stype=stype)
        assert c.format == (stype if stype is not None else "csr")

    @mark.parametrize("stype", (None,) + _SPARSE_FORMATS)
    def test_many_args_dense_last_coo_construct(self, mat_s, mat_s2, mat_d,
                                                stype):
        c = kron(mat_s, mat_s2, mat_d, stype=stype, coo_build=True)
        assert c.format == (stype if stype is not None else "csr")

    @mark.parametrize("stype", (None,) + _SPARSE_FORMATS)
    def test_many_args_dense_not_last_coo_construct(self, mat_s, mat_s2, mat_d,
                                                    stype):
        c = kron(mat_s, mat_d, mat_s2, stype=stype, coo_build=True)
        assert c.format == (stype if stype is not None else "csr")
        c = kron(mat_d, mat_s, mat_s2, stype=stype, coo_build=True)
        assert c.format == (stype if stype is not None else "csr")


class TestKronPow:
    def test_dense(self, mat_d):
        x = mat_d & mat_d & mat_d
        y = kronpow(mat_d, 3)
        assert_allclose(x, y)

    def test_sparse(self, mat_s):
        x = mat_s & mat_s & mat_s
        y = kronpow(mat_s, 3)
        assert_allclose(x.A, y.A)

    @mark.parametrize("stype", _SPARSE_FORMATS)
    def test_sparse_formats(self, stype, mat_s):
        x = mat_s & mat_s & mat_s
        y = kronpow(mat_s, 3, stype=stype)
        assert y.format == stype
        assert_allclose(x.A, y.A)

    @mark.parametrize("sformat_in", _SPARSE_FORMATS)
    @mark.parametrize("stype", (None,) + _SPARSE_FORMATS)
    def test_sparse_formats_coo_construct(self, sformat_in, stype, mat_s):
        mat_s = mat_s.asformat(sformat_in)
        x = mat_s & mat_s & mat_s
        y = kronpow(mat_s, 3, stype=stype, coo_build=True)
        assert y.format == stype if stype is not None else "sformat_in"
        assert_allclose(x.A, y.A)
