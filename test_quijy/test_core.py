from itertools import combinations
import scipy.sparse as sp
import numpy as np
from pytest import raises
from numpy.testing import assert_allclose, assert_almost_equal
from quijy.gen import (bell_state, rand_rho, rand_matrix, rand_ket, up, plus,
                       yplus, sig)
from quijy.calc import mutual_information
from quijy.core import (quijify, qjf, isbra, isket, isop, tr, isherm,
                        trace, trace_dense, trace_sparse, nmlz, kron_dense,
                        kron, kronpow, coo_map, coo_compress, eye, eyepad,
                        permute, chop, ldmul, rdmul,
                        infer_size, issparse, matrixify, realify, issmall,
                        accel_mul, accel_dot, accel_vdot, inner, trace_lose,
                        trace_keep, partial_trace, perm_pad)


class TestQuijify:
    def test_quijify_vector_create(self):
        x = [1, 2, 3j]
        p = quijify(x, qtype='ket')
        assert(type(p) == np.matrix)
        assert(p.dtype == np.complex)
        assert(p.shape == (3, 1))
        p = quijify(x, qtype='bra')
        assert(p.shape == (1, 3))
        assert_almost_equal(p[0, 2], -3.0j)

    def test_quijify_dop_create(self):
        x = np.random.randn(3, 3)
        p = quijify(x, qtype='dop')
        assert(type(p) == np.matrix)
        assert(p.dtype == np.complex)
        assert(p.shape == (3, 3))

    def test_quijify_convert_vector_to_dop(self):
        x = [1, 2, 3j]
        p = quijify(x, qtype='r')
        assert_allclose(p, np.matrix([[1.+0.j,  2.+0.j,  0.-3.j],
                                      [2.+0.j,  4.+0.j,  0.-6.j],
                                      [0.+3.j,  0.+6.j,  9.+0.j]]))

    def test_quijify_chopped(self):
        x = [9e-16, 1]
        p = quijify(x, 'k', chopped=False)
        assert(p[0, 0] != 0.0)
        p = quijify(x, 'k', chopped=True)
        assert(p[0, 0] == 0.0)

    def test_quijify_normalized(self):
        x = [3j, 4j]
        p = quijify(x, 'k', normalized=False)
        assert_almost_equal(tr(p.H @ p), 25.)
        p = quijify(x, 'k', normalized=True)
        assert_almost_equal(tr(p.H @ p), 1.)
        p = quijify(x, 'dop', normalized=True)
        assert_almost_equal(tr(p), 1.)

    def test_quijify_sparse_create(self):
        x = [[1, 0], [3, 0]]
        p = quijify(x, 'dop', sparse=False)
        assert(type(p) == np.matrix)
        p = quijify(x, 'dop', sparse=True)
        assert(type(p) == sp.csr_matrix)
        assert(p.dtype == np.complex)
        assert(p.nnz == 2)

    def test_quijify_sparse_convert_to_dop(self):
        x = [1, 0, 9e-16, 0, 3j]
        p = quijify(x, 'ket', sparse=True)
        q = quijify(p, 'dop', sparse=True)
        assert(q.shape == (5, 5))
        assert(q.nnz == 9)
        assert_almost_equal(q[4, 4], 9.)
        q = quijify(p, 'dop', sparse=True, normalized=True)
        assert_almost_equal(tr(q), 1.)


class TestShapes:
    def test_sparse(self):
        x = qjf([[1], [0]])
        assert not issparse(x)
        x = qjf([[1], [0]], sparse=True)
        assert issparse(x)

    def test_ket(self):
        x = qjf([[1], [0]])
        assert(isket(x))
        assert(not isbra(x))
        assert(not isop(x))
        x = qjf([[1], [0]], sparse=True)
        assert(isket(x))
        assert(not isbra(x))
        assert(not isop(x))

    def test_bra(self):
        x = qjf([[1, 0]])
        assert(not isket(x))
        assert(isbra(x))
        assert(not isop(x))
        x = qjf([[1, 0]], sparse=True)
        assert(not isket(x))
        assert(isbra(x))
        assert(not isop(x))

    def test_op(self):
        x = qjf([[1, 0], [0, 1]])
        assert(not isket(x))
        assert(not isbra(x))
        assert(isop(x))
        x = qjf([[1, 0], [0, 1]], sparse=True)
        assert(not isket(x))
        assert(not isbra(x))
        assert(isop(x))

    def test_isherm(self):
        a = qjf([[1.0, 2.0 + 3.0j],
                 [2.0 - 3.0j, 1.0]])
        assert(isherm(a))
        a = qjf([[1.0, 2.0 - 3.0j],
                 [2.0 - 3.0j, 1.0]])
        assert(not isherm(a))

    def test_isherm_sparse(self):
        a = qjf([[1.0, 2.0 + 3.0j],
                 [2.0 - 3.0j, 1.0]], sparse=True)
        assert(isherm(a))
        a = qjf([[1.0, 2.0 - 3.0j],
                 [2.0 - 3.0j, 1.0]], sparse=True)
        assert(not isherm(a))

    def test_issmall(self):
        a = qjf([1, 2, 3], 'ket')
        assert issmall(a, 4)
        assert not issmall(a, 2)
        a = qjf([1, 2, 3], 'dop')
        assert issmall(a, 4)
        assert not issmall(a, 2)


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


class TestInferSize:
    def test_infer_size(self):
        p = rand_ket(8)
        assert infer_size(p) == 3
        p = rand_ket(16)
        assert infer_size(p) == 4

    def test_infer_size_base(self):
        p = rand_ket(9)
        assert infer_size(p, 3) == 2
        p = rand_ket(81)
        assert infer_size(p, 3) == 4


class TestTrace:
    def test_trace_dense(self):
        a = qjf([[2, 1], [4, 5]])
        assert(trace(a) == 7)
        a = qjf([[2, 1], [4, 5j]])
        assert(trace(a) == 2 + 5j)
        assert(a.tr.__code__.co_code == trace_dense.__code__.co_code)

    def test_sparse_trace(self):
        a = qjf([[2, 1], [0, 5]], sparse=True)
        assert(trace(a) == 7)
        a = qjf([[2, 1], [4, 5j]], sparse=True)
        assert(trace(a) == 2 + 5j)
        assert(a.tr.__code__.co_code == trace_sparse.__code__.co_code)


class TestNormalize:
    def test_normalize_ket(self):
        a = qjf([1, -1j], 'ket')
        b = nmlz(a, inplace=False)
        assert_almost_equal(trace(b.H @ b), 1.0)
        assert_almost_equal(trace(a.H @ a), 2.0)

    def test_normalize_bra(self):
        a = qjf([1, -1j], 'bra')
        b = nmlz(a, inplace=False)
        assert_almost_equal(trace(b @ b.H), 1.0)

    def test_normalize_dop(self):
        a = qjf([1, -1j], 'dop')
        b = nmlz(a, inplace=False)
        assert_almost_equal(trace(b), 1.0)

    def test_normalize_inplace_ket(self):
        a = qjf([1, -1j], 'ket')
        a.nmlz(inplace=True)
        assert_almost_equal(trace(a.H @ a), 1.0)

    def test_normalize_inplace_bra(self):
        a = qjf([1, -1j], 'bra')
        a.nmlz(inplace=True)
        assert_almost_equal(trace(a @ a.H), 1.0)

    def test_normalize_inplace_dop(self):
        a = qjf([1, -1j], 'dop')
        b = nmlz(a, inplace=True)
        assert_almost_equal(trace(a), 1.0)
        assert_almost_equal(trace(b), 1.0)


class TestKron:
    def test_kron_dense(self):
        a = rand_matrix(3)
        b = rand_matrix(3)
        c = kron_dense(a, b)
        npc = np.kron(a, b)
        assert_allclose(c, npc)

    def test_kron_multi_args(self):
        a = rand_matrix(3)
        b = rand_matrix(3)
        c = rand_matrix(3)
        assert(kron() == 1)
        assert_allclose(kron(a), a)
        assert_allclose(kron(a, b, c),
                        np.kron(np.kron(a, b), c))

    def test_kron_mixed_types(self):
        a = qjf([1, 2, 3, 4], 'ket')
        b = qjf([0, 1, 0, 2], 'ket', sparse=True)
        assert_allclose(kron(a, b).A,
                        (sp.kron(a, b, 'csr')).A)
        assert_allclose(kron(b, b).A,
                        (sp.kron(b, b, 'csr')).A)

    def test_kronpow(self):
        a = rand_matrix(2)
        b = a & a & a
        c = kronpow(a, 3)
        assert_allclose(b, c)


class TestCooMap:
    def test_coo_map_1d(self):
        dims = [10, 11, 12, 13]
        coos = (1, 2, 3)
        ndims, ncoos = coo_map(dims, coos)
        assert_allclose(ndims[ncoos], (11, 12, 13))
        coos = ([-1], [2], [5])
        with raises(ValueError):
            ndims, ncoos = coo_map(dims, coos)
        ndims, ncoos = coo_map(dims, coos, cyclic=True)
        assert_allclose(ndims[ncoos], (13, 12, 11))
        ndims, ncoos = coo_map(dims, coos, trim=True)
        assert_allclose(ndims[ncoos], [12])

    def test_coo_map2d(self):
        dims = [[200, 201, 202, 203],
                [210, 211, 212, 213]]
        coos = ((1, 2), (1, 3), (0, 3))
        ndims, ncoos = coo_map(dims, coos)
        assert_allclose(ndims[ncoos], (212, 213, 203))
        coos = ((-1, 1), (1, 2), (3, 4))
        with raises(ValueError):
            ndims, ncoos = coo_map(dims, coos)
        ndims, ncoos = coo_map(dims, coos, cyclic=True)
        assert_allclose(ndims[ncoos], (211, 212, 210))
        ndims, ncoos = coo_map(dims, coos, trim=True)
        assert_allclose(ndims[ncoos], [212])

    def test_coo_map_3d(self):
        dims = [[[3000, 3001, 3002],
                 [3010, 3011, 3012],
                 [3020, 3021, 3022]],
                [[3100, 3101, 3102],
                 [3110, 3111, 3112],
                 [3120, 3121, 3122]]]
        coos = ((0, 0, 2), (1, 1, 2), (1, 2, 0))
        ndims, ncoos = coo_map(dims, coos)
        assert_allclose(ndims[ncoos], (3002, 3112, 3120))
        coos = ((0, -1, 2), (1, 2, 2), (4, -1, 3))
        with raises(ValueError):
            ndims, ncoos = coo_map(dims, coos)
        ndims, ncoos = coo_map(dims, coos, cyclic=True)
        assert_allclose(ndims[ncoos], (3022, 3122, 3020))
        ndims, ncoos = coo_map(dims, coos, trim=True)
        assert_allclose(ndims[ncoos], [3122])


class TestCooCompress:
    def test_coo_compress_edge(self):
        dims = [2, 3, 2, 4, 5]
        coos = [0, 4]
        ndims, ncoos = coo_compress(dims, coos)
        assert ndims == [2, 24, 5]
        assert ncoos == [0, 2]

    def test_coo_compress_middle(self):
        dims = [5, 3, 2, 5, 4, 3, 2]
        coos = [1, 2, 3, 5]
        ndims, ncoos = coo_compress(dims, coos)
        assert ndims == [5, 30, 4, 3, 2]
        assert ncoos == [1, 3]

    def test_coo_compress_single(self):
        dims = [5, 3, 2, 5, 4, 3, 2]
        coos = 3
        ndims, ncoos = coo_compress(dims, coos)
        assert ndims == [30, 5, 24]
        assert ncoos == [1]


class TestEye:
    def test_eye_dense(self):
        a = eye(3, sparse=False)
        assert a.shape == (3, 3)
        assert isinstance(a, np.matrix)
        assert a.dtype == complex

    def test_eye_sparse(self):
        a = eye(3, sparse=True)
        assert a.shape == (3, 3)
        assert isinstance(a, sp.csr_matrix)
        assert a.dtype == complex


class TestEyepad:
    def test_eyepad_basic(self):
        a = rand_matrix(2)
        i = eye(2)
        dims = [2, 2, 2]
        b = eyepad([a], dims, [0])
        assert_allclose(b, a & i & i)
        b = eyepad([a], dims, [1])
        assert_allclose(b, i & a & i)
        b = eyepad([a], dims, [2])
        assert_allclose(b, i & i & a)
        b = eyepad([a], dims, [0, 2])
        assert_allclose(b, a & i & a)
        b = eyepad([a], dims, [0, 1, 2])
        assert_allclose(b, a & a & a)

    def test_eyepad_mid_multi(self):
        a = [rand_matrix(2) for i in range(3)]
        i = eye(2)
        dims = [2, 2, 2, 2, 2, 2]
        inds = [1, 2, 4]
        b = eyepad(a, dims, inds)
        assert_allclose(b, i & a[0] & a[1] & i & a[2] & i)

    def test_eyepad_mid_multi_reverse(self):
        a = [rand_matrix(2) for i in range(3)]
        i = eye(2)
        dims = [2, 2, 2, 2, 2, 2]
        inds = [5, 4, 1]
        b = eyepad(a, dims, inds)
        assert_allclose(b, i & a[2] & i & i & a[1] & a[0])

    def test_eyepad_auto(self):
        a = rand_matrix(2)
        i = eye(2)
        b = eyepad([a], (2, -1, 2), [1])
        assert_allclose(b, i & a & i)

    def test_eyepad_ndarrays(self):
        a = rand_matrix(2)
        i = eye(2)
        b = eyepad([a], np.array([2, 2, 2]), [0, 2])
        assert_allclose(b, a & i & a)
        b = eyepad([a], [2, 2, 2], np.array([0, 2]))
        assert_allclose(b, a & i & a)

    def test_eyepad_overlap(self):
        a = [rand_matrix(4) for i in range(2)]
        dims1 = [2, 2, 2, 2, 2, 2]
        dims2 = [2, 4, 4, 2]
        b = eyepad(a, dims1, [1, 2, 3, 4])
        c = eyepad(a, dims2, [1, 2])
        assert_allclose(c, b)
        dims2 = [4, 2, 2, 4]
        b = eyepad(a, dims1, [0, 1, 4, 5])
        c = eyepad(a, dims2, [0, 3])
        assert_allclose(c, b)

    def test_eyepad_holey_overlap(self):
        a = rand_matrix(8)
        dims1 = (2, 2, 2, 2, 2)
        dims2 = (2, 8, 2)
        b = eyepad(a, dims1, (1, 3))
        c = eyepad(a, dims2, 1)
        assert_allclose(b, c)
        dims1 = (2, 2, 2, 2, 2)
        dims2 = (2, 2, 8)
        b = eyepad(a, dims1, (2, 4))
        c = eyepad(a, dims2, 2)
        assert_allclose(b, c)
        dims1 = (2, 2, 2, 2, 2)
        dims2 = (8, 2, 2)
        b = eyepad(a, dims1, (0, 2))
        c = eyepad(a, dims2, 0)
        assert_allclose(b, c)

    def test_eyepad_sparse(self):
        i = eye(2, sparse=True)
        a = qjf(rand_matrix(2), sparse=True)
        b = eyepad(a, [2, 2, 2], 1)  # infer sparse
        assert(issparse(b))
        assert_allclose(b.A, (i & a & i).A)
        a = rand_matrix(2)
        b = eyepad(a, [2, 2, 2], 1, sparse=True)  # explicit sparse
        assert(issparse(b))
        assert_allclose(b.A, (i & a & i).A)

    def test_eyepad_2d_simple(self):
        a = (rand_matrix(2), rand_matrix(2))
        dims = ((2, 3), (3, 2))
        inds = ((0, 0), (1, 1))
        b = eyepad(a, dims, inds)
        assert b.shape == (36, 36)
        assert_allclose(b, a[0] & eye(9) & a[1])


class TestPermPad:
    def test_perm_pad_dop_spread(self):
        a = rand_rho(4)
        b = perm_pad(a, [2, 2, 2], [0, 2])
        c = (a & eye(2)).A.reshape([2, 2, 2, 2, 2, 2])  \
                          .transpose([0, 2, 1, 3, 5, 4])  \
                          .reshape([8, 8])
        assert_allclose(b, c)

    def test_perm_pad_dop_reverse(self):
        a = rand_rho(4)
        b = perm_pad(a, [2, 2, 2], [2, 0])
        c = (a & eye(2)).A.reshape([2, 2, 2, 2, 2, 2])  \
                          .transpose([1, 2, 0, 4, 5, 3])  \
                          .reshape([8, 8])
        assert_allclose(b, c)


class TestPermute:
    def test_permute_ket(self):
        a = up() & plus() & yplus()
        b = permute(a, [2, 2, 2], [2, 0, 1])
        assert_allclose(b, yplus() & up() & plus())

    def test_permute_op(self):
        a = sig('x') & sig('y') & sig('z')
        b = permute(a, [2, 2, 2], [2, 0, 1])
        assert_allclose(b, sig('z') & sig('x') & sig('y'))

    def test_entangled_permute(self):
        dims = [2, 2, 2]
        a = bell_state(0) & up()
        assert_allclose(mutual_information(a, dims, 0, 1), 2.)
        b = permute(a, dims, [1, 2, 0])
        assert_allclose(mutual_information(b, dims, 0, 1), 0., atol=1e-12)
        assert_allclose(mutual_information(b, dims, 0, 2), 2.)

    def test_permute_sparse_ket(self):
        dims = [3, 2, 5, 4]
        a = rand_ket(np.prod(dims), sparse=True, density=0.5)
        b = permute(a, dims, [3, 1, 2, 0])
        c = permute(a.A, dims, [3, 1, 2, 0])
        assert_allclose(b.A, c)

    def test_permute_sparse_op(self):
        dims = [3, 2, 5, 4]
        a = rand_rho(np.prod(dims), sparse=True, density=0.5)
        b = permute(a, dims, [3, 1, 2, 0])
        c = permute(a.A, dims, [3, 1, 2, 0])
        assert_allclose(b.A, c)


class TestPartialTraceDense:
    def test_partial_trace_basic(self):
        a = rand_rho(2**2)
        b = partial_trace(a, [2, 2], 0)
        assert isinstance(b, np.matrix)
        assert isherm(b)
        assert_allclose(tr(b), 1.0)

    def test_ptr_compare_to_manual(self):
        a = rand_rho(2**2)
        b = partial_trace(a, [2, 2], 0)
        c = a.A.reshape([2, 2, 2, 2]).trace(axis1=1, axis2=3)
        assert_allclose(b, c)
        b = partial_trace(a, [2, 2], 1)
        c = a.A.reshape([2, 2, 2, 2]).trace(axis1=0, axis2=2)
        assert_allclose(b, c)

    def test_partial_trace_early_return(self):
        a = qjf([0.5, 0.5, 0.5, 0.5], 'ket')
        b = partial_trace(a, [2, 2], [0, 1])
        assert_allclose(a @ a.H, b)
        a = qjf([0.5, 0.5, 0.5, 0.5], 'dop')
        b = partial_trace(a, [2, 2], [0, 1])
        assert_allclose(a, b)

    def test_partial_trace_return_type(self):
        a = qjf([0, 2**-0.5, 2**-0.5, 0], 'ket')
        b = partial_trace(a, [2, 2], 1)
        assert(type(b) == np.matrix)
        a = qjf([0, 2**-0.5, 2**-0.5, 0], 'dop')
        b = partial_trace(a, [2, 2], 1)
        assert(type(b) == np.matrix)

    def test_partial_trace_single_ket(self):
        dims = [2, 3, 4]
        a = np.random.randn(np.prod(dims), 1)
        for i, dim in enumerate(dims):
            b = partial_trace(a, dims, i)
            assert(b.shape[0] == dim)

    def test_partial_trace_multi_ket(self):
        dims = [2, 3, 4]
        a = np.random.randn(np.prod(dims), 1)
        for i1, i2 in combinations([0, 1, 2], 2):
            b = partial_trace(a, dims, [i1, i2])
            assert(b.shape[1] == dims[i1] * dims[i2])

    def test_partial_trace_dop_product_state(self):
        dims = [3, 2, 4, 2, 3]
        ps = [rand_rho(dim) for dim in dims]
        pt = kron(*ps)
        for i, dim in enumerate(dims):
            p = partial_trace(pt, dims, i)
            assert_allclose(p, ps[i])

    def test_partial_trace_bell_states(self):
        for lab in ('psi-', 'psi+', 'phi-', 'phi+'):
            psi = bell_state(lab, qtype='dop')
            rhoa = partial_trace(psi, [2, 2], 0)
            assert_allclose(rhoa, eye(2)/2)

    def test_partial_trace_supply_ndarray(self):
        a = rand_rho(2**3)
        dims = np.array([2, 2, 2])
        keep = np.array(1)
        b = partial_trace(a, dims, keep)
        assert(b.shape[0] == 2)


class TestTraceLose:
    def test_rps(self):
        a, b, c = (rand_rho(2, sparse=True, density=0.5),
                   rand_rho(3, sparse=True, density=0.5),
                   rand_rho(2, sparse=True, density=0.5))
        abc = a & b & c
        pab = trace_lose(abc, [2, 3, 2], 2)
        assert_allclose(pab, (a & b).A)
        pac = trace_lose(abc, [2, 3, 2], 1)
        assert_allclose(pac, (a & c).A)
        pbc = trace_lose(abc, [2, 3, 2], 0)
        assert_allclose(pbc, (b & c).A)

    def test_bell_state(self):
        a = bell_state('psi-', sparse=True)
        b = trace_lose(a @ a.H, [2, 2], 0)
        assert_allclose(b, eye(2) / 2)
        b = trace_lose(a @ a.H, [2, 2], 1)
        assert_allclose(b, eye(2) / 2)

    def test_vs_ptr(self):
        a = rand_rho(6, sparse=True, density=0.5)
        b = trace_lose(a, [2, 3], 1)
        c = partial_trace(a.A, [2, 3], 0)
        assert_allclose(b, c)
        b = trace_lose(a, [2, 3], 0)
        c = partial_trace(a.A, [2, 3], 1)
        assert_allclose(b, c)

    def test_vec_dense(self):
        a = rand_ket(4)
        b = trace_lose(a, [2, 2], 1)
        c = partial_trace(a.A, [2, 2], 0)
        assert_allclose(b, c)
        b = trace_lose(a, [2, 2], 0)
        c = partial_trace(a.A, [2, 2], 1)
        assert_allclose(b, c)


class TestTraceKeep:
    def test_rps(self):
        a, b, c = (rand_rho(2, sparse=True, density=0.5),
                   rand_rho(3, sparse=True, density=0.5),
                   rand_rho(2, sparse=True, density=0.5))
        abc = a & b & c
        pc = trace_keep(abc, [2, 3, 2], 2)
        assert_allclose(pc, c.A)
        pb = trace_keep(abc, [2, 3, 2], 1)
        assert_allclose(pb, b.A)
        pa = trace_keep(abc, [2, 3, 2], 0)
        assert_allclose(pa, a.A)

    def test_bell_state(self):
        a = bell_state('psi-', sparse=True)
        b = trace_keep(a @ a.H, [2, 2], 0)
        assert_allclose(b, eye(2) / 2)
        b = trace_keep(a @ a.H, [2, 2], 1)
        assert_allclose(b, eye(2) / 2)

    def test_vs_ptr(self):
        a = rand_rho(6, sparse=True, density=0.5)
        b = trace_keep(a, [2, 3], 0)
        c = partial_trace(a.A, [2, 3], 0)
        assert_allclose(b, c)
        b = trace_keep(a, [2, 3], 1)
        c = partial_trace(a.A, [2, 3], 1)
        assert_allclose(b, c)

    def test_vec_dense(self):
        a = rand_ket(4)
        b = trace_keep(a, [2, 2], 0)
        c = partial_trace(a.A, [2, 2], 0)
        assert_allclose(b, c)
        b = trace_keep(a, [2, 2], 1)
        c = partial_trace(a.A, [2, 2], 1)
        assert_allclose(b, c)


class TestPartialTraceSparse:
    def test_partial_trace_sparse_basic(self):
        a = rand_rho(4)
        b = partial_trace(a, [2, 2], 0)
        assert type(b) == np.matrix
        assert isherm(b)
        assert_allclose(tr(b), 1.0)

    def test_partial_trace_simple_single(self):
        a = rand_rho(12, sparse=True, density=0.5)
        dims = [2, 3, 2]
        b = partial_trace(a, dims, 1)
        c = a.A.reshape([*dims, *dims])  \
             .trace(axis1=2, axis2=5)  \
             .trace(axis1=0, axis2=2)
        assert_allclose(c, b)

    def test_partial_trace_simple_double(self):
        a = rand_rho(12, sparse=True, density=0.5)
        dims = [2, 3, 2]
        b = partial_trace(a, dims, [0, 2])
        c = partial_trace(a.A, dims, [0, 2])
        assert_allclose(b, c)
        b = partial_trace(a, dims, [1, 2])
        c = partial_trace(a.A, dims, [1, 2])
        assert_allclose(b, c)

    def test_partial_trace_simple_ket(self):
        a = rand_ket(12, sparse=True, density=0.5)
        dims = [2, 3, 2]
        b = partial_trace(a, dims, [0, 1])
        c = partial_trace(a.A, dims, [0, 1])
        assert_allclose(b, c)


class TestChop:
    def test_chop_inplace(self):
        a = qjf([-1j, 0.1+0.2j])
        chop(a, tol=0.11, inplace=True)
        assert_allclose(a, qjf([-1j, 0.2j]))
        # Sparse
        a = qjf([-1j, 0.1+0.2j], sparse=True)
        chop(a, tol=0.11, inplace=True)
        b = qjf([-1j, 0.2j], sparse=True)
        assert((a != b).nnz == 0)

    def test_chop_inplace_dop(self):
        a = qjf([1, 0.1], 'dop')
        chop(a, tol=0.11, inplace=True)
        assert_allclose(a, qjf([1, 0], 'dop'))
        a = qjf([1, 0.1], 'dop', sparse=True)
        chop(a, tol=0.11, inplace=True)
        b = qjf([1, 0.0], 'dop', sparse=True)
        assert((a != b).nnz == 0)

    def test_chop_copy(self):
        a = qjf([-1j, 0.1+0.2j])
        b = chop(a, tol=0.11, inplace=False)
        assert_allclose(a, qjf([-1j, 0.1+0.2j]))
        assert_allclose(b, qjf([-1j, 0.2j]))
        # Sparse
        a = qjf([-1j, 0.1+0.2j], sparse=True)
        b = chop(a, tol=0.11, inplace=False)
        ao = qjf([-1j, 0.1+0.2j], sparse=True)
        bo = qjf([-1j, 0.2j], sparse=True)
        assert((a != ao).nnz == 0)
        assert((b != bo).nnz == 0)


class TestAccelMul:
    def test_accel_mul_same(self):
        a = rand_matrix(5)
        b = rand_matrix(5)
        ca = accel_mul(a, b)
        assert isinstance(ca, np.matrix)
        cn = np.multiply(a, b)
        assert_allclose(ca, cn)

    def test_accel_mul_broadcast(self):
        a = rand_matrix(5)
        b = rand_ket(5)
        ca = accel_mul(a, b)
        assert isinstance(ca, np.matrix)
        cn = np.multiply(a, b)
        assert_allclose(ca, cn)
        ca = accel_mul(a.H, b)
        assert isinstance(ca, np.matrix)
        cn = np.multiply(a.H, b)
        assert_allclose(ca, cn)


class TestAccelDot:
    def test_accel_dot_matrix(self):
        a = rand_matrix(5)
        b = rand_matrix(5)
        ca = accel_dot(a, b)
        assert isinstance(ca, np.matrix)
        cn = a @ b
        assert_allclose(ca, cn)

    def test_accel_dot_ket(self):
        a = rand_matrix(5)
        b = rand_ket(5)
        ca = accel_dot(a, b)
        assert isinstance(ca, np.matrix)
        cn = a @ b
        assert_allclose(ca, cn)


class TestAccelVdot:
    def test_accel_vdot(self):
        a = rand_ket(5)
        b = rand_ket(5)
        ca = accel_vdot(a, b)
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


class TestInner:
    def test_inner_vec_vec_dense(self):
        a = qjf([[1], [2j], [3]])
        b = qjf([[1j], [2], [3j]])
        c = inner(a, b)
        assert not isinstance(c, complex)
        assert_allclose(c, 36)

    def test_inner_vec_op_dense(self):
        a = qjf([[1], [2j], [3]], 'dop')
        b = qjf([[1j], [2], [3j]])
        c = inner(a, b)
        assert not isinstance(c, complex)
        assert_allclose(c, 36)

    def test_inner_op_vec_dense(self):
        a = qjf([[1], [2j], [3]])
        b = qjf([[1j], [2], [3j]], 'dop')
        c = inner(a, b)
        assert not isinstance(c, complex)
        assert_allclose(c, 36)

    def test_inner_op_op_dense(self):
        a = qjf([[1], [2j], [3]], 'dop')
        b = qjf([[1j], [2], [3j]], 'dop')
        c = inner(a, b)
        assert not isinstance(c, complex)
        assert_allclose(c, 36)

    def test_inner_vec_vec_sparse(self):
        a = qjf([[1], [2j], [3]], sparse=True)
        b = qjf([[1j], [2], [3j]])
        c = inner(a, b)
        assert not isinstance(c, complex)
        assert_allclose(c, 36)

    def test_inner_vec_op_sparse(self):
        a = qjf([[1], [2j], [3]], 'dop', sparse=True)
        b = qjf([[1j], [2], [3j]], sparse=True)
        c = inner(a, b)
        assert not isinstance(c, complex)
        assert_allclose(c, 36)

    def test_inner_op_vec_sparse(self):
        a = qjf([[1], [2j], [3]])
        b = qjf([[1j], [2], [3j]], 'dop', sparse=True)
        c = inner(a, b)
        assert not isinstance(c, complex)
        assert_allclose(c, 36)

    def test_inner_op_op_sparse(self):
        a = qjf([[1], [2j], [3]], 'dop', sparse=True)
        b = qjf([[1j], [2], [3j]], 'dop', sparse=True)
        c = inner(a, b)
        assert not isinstance(c, complex)
        assert_allclose(c, 36)
