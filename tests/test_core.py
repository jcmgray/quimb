import itertools

from pytest import fixture, raises, mark
import scipy.sparse as sp
import numpy as np
from numpy.testing import assert_allclose, assert_almost_equal

import quimb as qu

stypes = ("csr", "csc", "bsr", "coo")


@fixture
def od1():
    return qu.rand_matrix(3)


@fixture
def os1():
    return qu.rand_matrix(3, sparse=True, density=0.5)


class TestSparseMatrix:
    @mark.parametrize("stype", stypes)
    def test_simple(self, stype):
        a = qu.core.sparse_matrix([[0, 3], [1, 2]], stype)
        assert a.format == stype
        assert a.dtype == complex


class TestQuimbify:
    def test_vector_create(self):
        x = [1, 2, 3j]
        p = qu.qu(x, qtype='ket')
        assert(type(p) == qu.qarray)
        assert(p.dtype == complex)
        assert(p.shape == (3, 1))
        p = qu.qu(x, qtype='bra')
        assert(p.shape == (1, 3))
        assert_almost_equal(p[0, 2], -3.0j)

    def test_dop_create(self):
        x = np.random.randn(3, 3)
        p = qu.qu(x, qtype='dop')
        assert(type(p) == qu.qarray)
        assert(p.dtype == complex)
        assert(p.shape == (3, 3))

    def test_convert_vector_to_dop(self):
        x = [1, 2, 3j]
        p = qu.qu(x, qtype='r')
        assert_allclose(p, qu.qarray([[1. + 0.j, 2. + 0.j, 0. - 3.j],
                                      [2. + 0.j, 4. + 0.j, 0. - 6.j],
                                      [0. + 3.j, 0. + 6.j, 9. + 0.j]]))

    def test_chopped(self):
        x = [9e-16, 1]
        p = qu.qu(x, 'k', chopped=False)
        assert(p[0, 0] != 0.0)
        p = qu.qu(x, 'k', chopped=True)
        assert(p[0, 0] == 0.0)

    def test_normalized(self):
        x = [3j, 4j]
        p = qu.qu(x, 'k', normalized=False)
        assert_almost_equal(qu.tr(p.H @ p), 25.)
        p = qu.qu(x, 'k', normalized=True)
        assert_almost_equal(qu.tr(p.H @ p), 1.)
        p = qu.qu(x, 'dop', normalized=True)
        assert_almost_equal(qu.tr(p), 1.)

    def test_sparse_create(self):
        x = [[1, 0], [3, 0]]
        p = qu.qu(x, 'dop', sparse=False)
        assert(type(p) == qu.qarray)
        p = qu.qu(x, 'dop', sparse=True)
        assert(type(p) == sp.csr_matrix)
        assert(p.dtype == complex)
        assert(p.nnz == 2)

    def test_sparse_convert_to_dop(self):
        x = [1, 0, 9e-16, 0, 3j]
        p = qu.qu(x, 'ket', sparse=True)
        q = qu.qu(p, 'dop', sparse=True)
        assert(q.shape == (5, 5))
        assert(q.nnz == 9)
        assert_almost_equal(q[4, 4], 9.)
        q = qu.qu(p, 'dop', sparse=True, normalized=True)
        assert_almost_equal(qu.tr(q), 1.)

    @mark.parametrize("qtype, shape, out",
                      (("bra", (1, 4), [[1, 0, 2, -3j]]),
                       ("ket", (4, 1), [[1], [0], [2], [3j]]),
                       ("dop", (4, 4), [[1, 0, 2, -3j],
                                        [0, 0, 0, 0],
                                        [2, 0, 4, -6j],
                                        [3j, 0, 6j, 9]])))
    @mark.parametrize("format_in", stypes)
    @mark.parametrize("format_out", (None,) + stypes)
    @mark.parametrize("dtype", [float, complex, np.float_, np.complex_])
    def test_reshape_sparse(self, qtype, shape, out,
                            format_in, format_out, dtype):
        import warnings

        in_ = [[1], [0], [2], [3j]]
        x = qu.core.sparse_matrix(in_, stype=format_in)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            y = qu.qu(x, qtype=qtype, stype=format_out, dtype=dtype)

        assert y.shape == shape
        assert y.dtype == dtype
        if format_out is None:
            format_out = format_in
        assert y.format == format_out

        if np.issubdtype(dtype, np.floating):
            assert_allclose(y.A, np.real(out), atol=1e-12)
        else:
            assert_allclose(y.A, out)

    @mark.parametrize("qtype, shape, out",
                      (("bra", (1, 4), [[1, 0, 2, -3j]]),
                       ("ket", (4, 1), [[1], [0], [2], [3j]]),
                       ("dop", (4, 4), [[1, 0, 2, -3j],
                                        [0, 0, 0, 0],
                                        [2, 0, 4, -6j],
                                        [3j, 0, 6j, 9]])))
    @mark.parametrize("format_out", (None,) + stypes)
    def test_dense_to_sparse_format(self, qtype, shape, out, format_out):
        x = [[1], [0], [2], [3j]]
        y = qu.qu(x, qtype=qtype, stype=format_out, sparse=True)
        assert y.shape == shape
        assert y.dtype == complex
        if format_out is None:
            format_out = "csr"
        assert y.format == format_out
        assert_allclose(y.A, out)

    @mark.parametrize("qtype, shape",
                      (["bra", (1, 4)],
                       ["ket", (4, 1)],
                       ["dop", (4, 4)]))
    @mark.parametrize("format_out", stypes)
    def test_give_sformat_only(self, qtype, shape, format_out):
        x = [[1], [0], [2], [3j]]
        y = qu.qu(x, qtype=qtype, stype=format_out)
        assert qu.issparse(y)
        assert y.shape == shape
        assert y.format == format_out


class TestInferSize:
    @mark.parametrize("d,base,n",
                      ([8, 2, 3],
                       [16, 2, 4],
                       [9, 3, 2],
                       [81, 3, 4]))
    def test_infer_size(self, d, base, n):
        p = qu.rand_ket(d)
        assert qu.infer_size(p, base) == n

    def test_raises(self):
        p = qu.rand_ket(2) & qu.rand_ket(3)
        with raises(ValueError):
            qu.infer_size(p, base=2)


class TestTrace:
    @mark.parametrize("inpt, outpt",
                      ([[[2, 1], [4, 5]], 7],
                       [[[2, 1], [4, 5j]], 2 + 5j]))
    @mark.parametrize("sparse, func",
                      ([False, qu.core._trace_dense],
                       [True, qu.core._trace_sparse]))
    def test_simple(self, inpt, outpt, sparse, func):
        a = qu.qu(inpt, sparse=sparse)
        assert(qu.trace(a) == outpt)


class TestITrace:
    @mark.parametrize("axes", [(0, 1), ((0,), (1,))])
    def test_axes_types(self, axes):
        a = qu.rand_matrix(4)
        b = qu.itrace(a, axes)
        assert_allclose(b, np.trace(a))

    def test_complex_dims(self):
        a = np.random.rand(4, 3, 2, 2, 4, 3)
        atr = qu.itrace(a, ((0, 1, 2), (4, 5, 3)))
        btr = np.trace(np.trace(np.trace(a, axis1=1, axis2=5),
                                axis1=1, axis2=2))
        assert_allclose(atr, btr)


class TestNormalize:
    def test_normalize_ket(self):
        a = qu.qu([1, -1j], 'ket')
        b = qu.nmlz(a, inplace=False)
        assert_almost_equal(qu.trace(b.H @ b), 1.0)
        assert_almost_equal(qu.trace(a.H @ a), 2.0)

    def test_normalize_bra(self):
        a = qu.qu([1, -1j], 'bra')
        b = qu.nmlz(a, inplace=False)
        assert_almost_equal(qu.trace(b @ b.H), 1.0)

    def test_normalize_dop(self):
        a = qu.qu([1, -1j], 'dop')
        b = qu.nmlz(a, inplace=False)
        assert_almost_equal(qu.trace(b), 1.0)

    def test_normalize_inplace_ket(self):
        a = qu.qu([1, -1j], 'ket')
        a.nmlz(inplace=True)
        assert_almost_equal(qu.trace(a.H @ a), 1.0)

    def test_normalize_inplace_bra(self):
        a = qu.qu([1, -1j], 'bra')
        a.nmlz(inplace=True)
        assert_almost_equal(qu.trace(a @ a.H), 1.0)

    def test_normalize_inplace_dop(self):
        a = qu.qu([1, -1j], 'dop')
        b = qu.nmlz(a, inplace=True)
        assert_almost_equal(qu.trace(a), 1.0)
        assert_almost_equal(qu.trace(b), 1.0)


class TestDimMap:
    @mark.parametrize("numpy", [False, True])
    def test_1d(self, numpy):
        dims = [10, 11, 12, 13]
        coos = (1, 2, 3)
        if numpy:
            dims, coos = np.asarray(dims), np.asarray(coos)
        ndims, ncoos = qu.dim_map(dims, coos)
        assert_allclose([ndims[c] for c in ncoos], (11, 12, 13))
        coos = ([-1], [2], [5])
        with raises(ValueError):
            ndims, ncoos = qu.dim_map(dims, coos)
        ndims, ncoos = qu.dim_map(dims, coos, cyclic=True)
        assert_allclose([ndims[c] for c in ncoos], (13, 12, 11))
        ndims, ncoos = qu.dim_map(dims, coos, trim=True)
        assert_allclose([ndims[c] for c in ncoos], [12])

    def test_2d(self):
        dims = [[200, 201, 202, 203],
                [210, 211, 212, 213]]
        coos = ((1, 2), (1, 3), (0, 3))
        ndims, ncoos = qu.dim_map(dims, coos)
        assert_allclose([ndims[c] for c in ncoos], (212, 213, 203))
        coos = ((-1, 1), (1, 2), (3, 4))
        with raises(ValueError):
            ndims, ncoos = qu.dim_map(dims, coos)
        ndims, ncoos = qu.dim_map(dims, coos, cyclic=True)
        assert_allclose([ndims[c] for c in ncoos], (211, 212, 210))
        ndims, ncoos = qu.dim_map(dims, coos, trim=True)
        assert_allclose([ndims[c] for c in ncoos], [212])

    def test_3d(self):
        dims = [[[3000, 3001, 3002],
                 [3010, 3011, 3012],
                 [3020, 3021, 3022]],
                [[3100, 3101, 3102],
                 [3110, 3111, 3112],
                 [3120, 3121, 3122]]]
        coos = ((0, 0, 2), (1, 1, 2), (1, 2, 0))
        ndims, ncoos = qu.dim_map(dims, coos)
        assert_allclose([ndims[c] for c in ncoos], (3002, 3112, 3120))
        coos = ((0, -1, 2), (1, 2, 2), (4, -1, 3))
        with raises(ValueError):
            ndims, ncoos = qu.dim_map(dims, coos)
        ndims, ncoos = qu.dim_map(dims, coos, cyclic=True)
        assert_allclose([ndims[c] for c in ncoos], (3022, 3122, 3020))
        ndims, ncoos = qu.dim_map(dims, coos, trim=True)
        assert_allclose([ndims[c] for c in ncoos], [3122])


class TestDimCompress:
    def test_edge(self):
        dims = [2, 3, 2, 4, 5]
        coos = [0, 4]
        ndims, ncoos = qu.dim_compress(dims, coos)
        assert ndims == (2, 24, 5)
        assert ncoos == (0, 2)

    def test_middle(self):
        dims = [5, 3, 2, 5, 4, 3, 2]
        coos = [1, 2, 3, 5]
        ndims, ncoos = qu.dim_compress(dims, coos)
        assert ndims == (5, 30, 4, 3, 2)
        assert ncoos == (1, 3)

    def test_single(self):
        dims = [5, 3, 2, 5, 4, 3, 2]
        coos = 3
        ndims, ncoos = qu.dim_compress(dims, coos)
        assert ndims == (30, 5, 24)
        assert ncoos == (1,)

    @mark.parametrize("dims, inds, ndims, ninds",
                      [([2, 2], [0, 1], (4,), (0,)),
                       ([4], [0], (4,), (0,))])
    def test_tiny(self, dims, inds, ndims, ninds):
        dims, inds = qu.dim_compress(dims, inds)
        assert dims == ndims
        assert inds == ninds


class TestEye:
    def test_eye_dense(self):
        a = qu.eye(3, sparse=False)
        assert a.shape == (3, 3)
        assert isinstance(a, qu.qarray)
        assert a.dtype == complex

    def test_eye_sparse(self):
        a = qu.eye(3, sparse=True)
        assert a.shape == (3, 3)
        assert isinstance(a, sp.csr_matrix)
        assert a.dtype == complex


class TestKron:
    @mark.parametrize("parallel", [True, False])
    def test_kron_basic(self, parallel):
        a = qu.rand_ket(2)
        b = qu.rand_ket(4)
        c = qu.rand_ket(4)
        d = qu.rand_ket(5)
        t = qu.kron(a, b, c, d, parallel=parallel)
        assert_allclose(t, a & b & c & d)

    @mark.parametrize("sparse", [True, False])
    @mark.parametrize("ri,rf", ([0, 4], [75, 89], [150, 168], [0, 168]))
    def test_kron_ownership(self, sparse, ri, rf):
        dims = [7, 2, 4, 3]
        ops = [qu.rand_matrix(d, sparse=sparse) for d in dims]
        X1 = qu.kron(*ops)[ri:rf, :]
        X2 = qu.kron(*ops, ownership=(ri, rf))
        assert_allclose(X1.A, X2.A)


class Testikron:
    def test_basic(self):
        a = qu.rand_matrix(2)
        i = qu.eye(2)
        dims = [2, 2, 2]
        b = qu.ikron([a], dims, [0])
        assert_allclose(b, a & i & i)
        b = qu.ikron([a], dims, [1])
        assert_allclose(b, i & a & i)
        b = qu.ikron([a], dims, [2])
        assert_allclose(b, i & i & a)
        b = qu.ikron([a], dims, [0, 2])
        assert_allclose(b, a & i & a)
        b = qu.ikron([a], dims, [0, 1, 2])
        assert_allclose(b, a & a & a)

    def test_mid_multi(self):
        a = [qu.rand_matrix(2) for i in range(3)]
        i = qu.eye(2)
        dims = [2, 2, 2, 2, 2, 2]
        inds = [1, 2, 4]
        b = qu.ikron(a, dims, inds)
        assert_allclose(b, i & a[0] & a[1] & i & a[2] & i)

    def test_mid_multi_reverse(self):
        a = [qu.rand_matrix(2) for i in range(3)]
        i = qu.eye(2)
        dims = [2, 2, 2, 2, 2, 2]
        inds = [5, 4, 1]
        b = qu.ikron(a, dims, inds)
        assert_allclose(b, i & a[2] & i & i & a[1] & a[0])

    def test_auto(self):
        a = qu.rand_matrix(2)
        i = qu.eye(2)
        b = qu.ikron([a], (2, -1, 2), [1])
        assert_allclose(b, i & a & i)

    def test_ndarrays(self):
        a = qu.rand_matrix(2)
        i = qu.eye(2)
        b = qu.ikron([a], np.array([2, 2, 2]), [0, 2])
        assert_allclose(b, a & i & a)
        b = qu.ikron([a], [2, 2, 2], np.array([0, 2]))
        assert_allclose(b, a & i & a)

    def test_overlap(self):
        a = [qu.rand_matrix(4) for i in range(2)]
        dims1 = [2, 2, 2, 2, 2, 2]
        dims2 = [2, 4, 4, 2]
        b = qu.ikron(a, dims1, [1, 2, 3, 4])
        c = qu.ikron(a, dims2, [1, 2])
        assert_allclose(c, b)
        dims2 = [4, 2, 2, 4]
        b = qu.ikron(a, dims1, [0, 1, 4, 5])
        c = qu.ikron(a, dims2, [0, 3])
        assert_allclose(c, b)

    def test_holey_overlap(self):
        a = qu.rand_matrix(8)
        dims1 = (2, 2, 2, 2, 2)
        dims2 = (2, 8, 2)
        b = qu.ikron(a, dims1, (1, 3))
        c = qu.ikron(a, dims2, 1)
        assert_allclose(b, c)
        dims1 = (2, 2, 2, 2, 2)
        dims2 = (2, 2, 8)
        b = qu.ikron(a, dims1, (2, 4))
        c = qu.ikron(a, dims2, 2)
        assert_allclose(b, c)
        dims1 = (2, 2, 2, 2, 2)
        dims2 = (8, 2, 2)
        b = qu.ikron(a, dims1, (0, 2))
        c = qu.ikron(a, dims2, 0)
        assert_allclose(b, c)

    def test_sparse(self):
        i = qu.eye(2, sparse=True)
        a = qu.qu(qu.rand_matrix(2), sparse=True)
        b = qu.ikron(a, [2, 2, 2], 1)  # infer sparse
        assert(qu.issparse(b))
        assert_allclose(b.A, (i & a & i).A)
        a = qu.rand_matrix(2)
        b = qu.ikron(a, [2, 2, 2], 1, sparse=True)  # explicit sparse
        assert(qu.issparse(b))
        assert_allclose(b.A, (i & a & i).A)

    def test_2d_simple(self):
        a = (qu.rand_matrix(2), qu.rand_matrix(2))
        dims = ((2, 3), (3, 2))
        inds = ((0, 0), (1, 1))
        b = qu.ikron(a, dims, inds)
        assert b.shape == (36, 36)
        assert_allclose(b, a[0] & qu.eye(9) & a[1])

    @mark.parametrize("stype", (None,) + stypes)
    @mark.parametrize("pos", [0, 1, 2,
                              (0,), (1,), (2,),
                              (0, 1),
                              (1, 2),
                              (0, 2)])
    @mark.parametrize("coo_build", [False, True])
    def test_sparse_format_outputs(self, os1, stype, pos, coo_build):
        x = qu.ikron(os1, [3, 3, 3], pos,
                     stype=stype, coo_build=coo_build)
        assert x.format == "csr" if stype is None else stype

    @mark.parametrize("stype", (None,) + stypes)
    @mark.parametrize("pos", [0, 1, 2,
                              (0,), (1,), (2,),
                              (0, 1),
                              (1, 2),
                              (0, 2)])
    @mark.parametrize("coo_build", [False, True])
    def test_sparse_format_outputs_with_dense(self, od1, stype, pos,
                                              coo_build):
        x = qu.ikron(od1, [3, 3, 3], pos, sparse=True,
                     stype=stype, coo_build=coo_build)
        try:
            default = "bsr" if (2 in pos and not coo_build) else "csr"
        except TypeError:
            default = "bsr" if (pos == 2 and not coo_build) else "csr"
        assert x.format == default if stype is None else stype

    @mark.parametrize("sparse", [True, False])
    @mark.parametrize("ri,rf", ([0, 4], [75, 89], [150, 168], [0, 168]))
    def test_ikron_ownership(self, sparse, ri, rf):
        dims = [7, 2, 4, 3]
        X = qu.rand_matrix(2, sparse=sparse)
        Y = qu.rand_matrix(3, sparse=sparse)
        X1 = qu.ikron((X, Y), dims, (1, 3))[ri:rf, :]
        X2 = qu.ikron((X, Y), dims, (1, 3), ownership=(ri, rf))
        assert_allclose(X1.A, X2.A)


class TestPermikron:
    def test_dop_spread(self):
        a = qu.rand_rho(4)
        b = qu.pkron(a, [2, 2, 2], [0, 2])
        c = ((a & qu.eye(2)).A
             .reshape([2, 2, 2, 2, 2, 2])
             .transpose([0, 2, 1, 3, 5, 4])
             .reshape([8, 8]))
        assert_allclose(b, c)

    def test_dop_reverse(self):
        a = qu.rand_rho(4)
        b = qu.pkron(a, np.array([2, 2, 2]), [2, 0])
        c = ((a & qu.eye(2)).A.reshape([2, 2, 2, 2, 2, 2])
                            .transpose([1, 2, 0, 4, 5, 3])
                            .reshape([8, 8]))
        assert_allclose(b, c)

    def test_dop_reverse_sparse(self):
        a = qu.rand_rho(4, sparse=True, density=0.5)
        b = qu.pkron(a, np.array([2, 2, 2]), [2, 0])
        c = ((a & qu.eye(2)).A.reshape([2, 2, 2, 2, 2, 2])
                            .transpose([1, 2, 0, 4, 5, 3])
                            .reshape([8, 8]))
        assert_allclose(b.A, c)


class TestPermute:
    def test_permute_ket(self):
        a = qu.up() & qu.plus() & qu.yplus()
        b = qu.permute(a, [2, 2, 2], [2, 0, 1])
        assert_allclose(b, qu.yplus() & qu.up() & qu.plus())

    def test_permute_op(self):
        a = qu.pauli('x') & qu.pauli('y') & qu.pauli('z')
        b = qu.permute(a, [2, 2, 2], [2, 0, 1])
        assert_allclose(b, qu.pauli('z') & qu.pauli('x') & qu.pauli('y'))

    def test_entangled_permute(self):
        dims = [2, 2, 2]
        a = qu.bell_state(0) & qu.up()
        assert_allclose(qu.mutinf_subsys(a, dims, 0, 1), 2.)
        b = qu.permute(a, dims, [1, 2, 0])
        assert_allclose(qu.mutinf_subsys(b, dims, 0, 1), 0., atol=1e-12)
        assert_allclose(qu.mutinf_subsys(b, dims, 0, 2), 2.)

    def test_permute_sparse_ket(self):
        dims = [3, 2, 5, 4]
        a = qu.rand_ket(qu.prod(dims), sparse=True, density=0.5)
        b = qu.permute(a, dims, [3, 1, 2, 0])
        c = qu.permute(a.A, dims, [3, 1, 2, 0])
        assert_allclose(b.A, c)

    def test_permute_sparse_op(self):
        dims = [3, 2, 5, 4]
        a = qu.rand_rho(qu.prod(dims), sparse=True, density=0.5)
        b = qu.permute(a, dims, [3, 1, 2, 0])
        c = qu.permute(a.A, dims, [3, 1, 2, 0])
        assert_allclose(b.A, c)


class TestPartialTraceDense:
    def test_partial_trace_basic(self):
        a = qu.rand_rho(2**2)
        b = qu.partial_trace(a, [2, 2], 0)
        assert isinstance(b, qu.qarray)
        assert qu.isherm(b)
        assert_allclose(qu.tr(b), 1.0)

    def test_ptr_compare_to_manual(self):
        a = qu.rand_rho(2**2)
        b = qu.partial_trace(a, [2, 2], 0)
        c = a.A.reshape([2, 2, 2, 2]).trace(axis1=1, axis2=3)
        assert_allclose(b, c)
        b = qu.partial_trace(a, [2, 2], 1)
        c = a.A.reshape([2, 2, 2, 2]).trace(axis1=0, axis2=2)
        assert_allclose(b, c)

    def test_partial_trace_early_return(self):
        a = qu.qu([0.5, 0.5, 0.5, 0.5], 'ket')
        b = qu.partial_trace(a, [2, 2], [0, 1])
        assert_allclose(a @ a.H, b)
        a = qu.qu([0.5, 0.5, 0.5, 0.5], 'dop')
        b = qu.partial_trace(a, [2, 2], [0, 1])
        assert_allclose(a, b)

    def test_partial_trace_return_type(self):
        a = qu.qu([0, 2**-0.5, 2**-0.5, 0], 'ket')
        b = qu.partial_trace(a, [2, 2], 1)
        assert(type(b) == qu.qarray)
        a = qu.qu([0, 2**-0.5, 2**-0.5, 0], 'dop')
        b = qu.partial_trace(a, [2, 2], 1)
        assert(type(b) == qu.qarray)

    def test_partial_trace_single_ket(self):
        dims = [2, 3, 4]
        a = np.random.randn(qu.prod(dims), 1)
        for i, dim in enumerate(dims):
            b = qu.partial_trace(a, dims, i)
            assert(b.shape[0] == dim)

    def test_partial_trace_multi_ket(self):
        dims = [2, 3, 4]
        a = np.random.randn(qu.prod(dims), 1)
        for i1, i2 in itertools.combinations([0, 1, 2], 2):
            b = qu.partial_trace(a, dims, [i1, i2])
            assert(b.shape[1] == dims[i1] * dims[i2])

    def test_partial_trace_dop_product_state(self):
        dims = [3, 2, 4, 2, 3]
        ps = [qu.rand_rho(dim) for dim in dims]
        pt = qu.kron(*ps)
        for i, dim in enumerate(dims):
            p = qu.partial_trace(pt, dims, i)
            assert_allclose(p, ps[i])

    def test_partial_trace_bell_states(self):
        for lab in ('psi-', 'psi+', 'phi-', 'phi+'):
            psi = qu.bell_state(lab, qtype='dop')
            rhoa = qu.partial_trace(psi, [2, 2], 0)
            assert_allclose(rhoa, qu.eye(2) / 2)

    def test_partial_trace_supply_ndarray(self):
        a = qu.rand_rho(2**3)
        dims = np.array([2, 2, 2])
        keep = np.array(1)
        b = qu.partial_trace(a, dims, keep)
        assert(b.shape[0] == 2)

    def test_partial_trace_order_doesnt_matter(self):
        a = qu.rand_rho(2**3)
        dims = np.array([2, 2, 2])
        b1 = qu.partial_trace(a, dims, [0, 2])
        b2 = qu.partial_trace(a, dims, [2, 0])
        assert_allclose(b1, b2)


class TestTraceLose:
    def test_rps(self):
        a, b, c = (qu.rand_rho(2, sparse=True, density=0.5),
                   qu.rand_rho(3, sparse=True, density=0.5),
                   qu.rand_rho(2, sparse=True, density=0.5))
        abc = a & b & c
        pab = qu.core._trace_lose(abc, [2, 3, 2], 2)
        assert_allclose(pab, (a & b).A)
        pac = qu.core._trace_lose(abc, [2, 3, 2], 1)
        assert_allclose(pac, (a & c).A)
        pbc = qu.core._trace_lose(abc, [2, 3, 2], 0)
        assert_allclose(pbc, (b & c).A)

    def test_bell_state(self):
        a = qu.bell_state('psi-', sparse=True)
        b = qu.core._trace_lose(a @ a.H, [2, 2], 0)
        assert_allclose(b, qu.eye(2) / 2)
        b = qu.core._trace_lose(a @ a.H, [2, 2], 1)
        assert_allclose(b, qu.eye(2) / 2)

    def test_vs_ptr(self):
        a = qu.rand_rho(6, sparse=True, density=0.5)
        b = qu.core._trace_lose(a, [2, 3], 1)
        c = qu.partial_trace(a.A, [2, 3], 0)
        assert_allclose(b, c)
        b = qu.core._trace_lose(a, [2, 3], 0)
        c = qu.partial_trace(a.A, [2, 3], 1)
        assert_allclose(b, c)

    def test_vec_dense(self):
        a = qu.rand_ket(4)
        b = qu.core._trace_lose(a, [2, 2], 1)
        c = qu.partial_trace(a.A, [2, 2], 0)
        assert_allclose(b, c)
        b = qu.core._trace_lose(a, [2, 2], 0)
        c = qu.partial_trace(a.A, [2, 2], 1)
        assert_allclose(b, c)


class TestTraceKeep:
    def test_rps(self):
        a, b, c = (qu.rand_rho(2, sparse=True, density=0.5),
                   qu.rand_rho(3, sparse=True, density=0.5),
                   qu.rand_rho(2, sparse=True, density=0.5))
        abc = a & b & c
        pc = qu.core._trace_keep(abc, [2, 3, 2], 2)
        assert_allclose(pc, c.A)
        pb = qu.core._trace_keep(abc, [2, 3, 2], 1)
        assert_allclose(pb, b.A)
        pa = qu.core._trace_keep(abc, [2, 3, 2], 0)
        assert_allclose(pa, a.A)

    def test_bell_state(self):
        a = qu.bell_state('psi-', sparse=True)
        b = qu.core._trace_keep(a @ a.H, [2, 2], 0)
        assert_allclose(b, qu.eye(2) / 2)
        b = qu.core._trace_keep(a @ a.H, [2, 2], 1)
        assert_allclose(b, qu.eye(2) / 2)

    def test_vs_ptr(self):
        a = qu.rand_rho(6, sparse=True, density=0.5)
        b = qu.core._trace_keep(a, [2, 3], 0)
        c = qu.partial_trace(a.A, [2, 3], 0)
        assert_allclose(b, c)
        b = qu.core._trace_keep(a, [2, 3], 1)
        c = qu.partial_trace(a.A, [2, 3], 1)
        assert_allclose(b, c)

    def test_vec_dense(self):
        a = qu.rand_ket(4)
        b = qu.core._trace_keep(a, [2, 2], 0)
        c = qu.partial_trace(a.A, [2, 2], 0)
        assert_allclose(b, c)
        b = qu.core._trace_keep(a, [2, 2], 1)
        c = qu.partial_trace(a.A, [2, 2], 1)
        assert_allclose(b, c)


class TestPartialTraceSparse:
    def test_partial_trace_sparse_basic(self):
        a = qu.rand_rho(4)
        b = qu.partial_trace(a, [2, 2], 0)
        assert type(b) == qu.qarray
        assert qu.isherm(b)
        assert_allclose(qu.tr(b), 1.0)

    def test_partial_trace_simple_single(self):
        a = qu.rand_rho(12, sparse=True, density=0.5)
        dims = [2, 3, 2]
        b = qu.partial_trace(a, dims, 1)
        c = a.A.reshape([*dims, *dims])  \
             .trace(axis1=2, axis2=5)  \
             .trace(axis1=0, axis2=2)
        assert_allclose(c, b)

    def test_partial_trace_simple_double(self):
        a = qu.rand_rho(12, sparse=True, density=0.5)
        dims = [2, 3, 2]
        b = qu.partial_trace(a, dims, [0, 2])
        c = qu.partial_trace(a.A, dims, [0, 2])
        assert_allclose(b, c)
        b = qu.partial_trace(a, dims, [1, 2])
        c = qu.partial_trace(a.A, dims, [1, 2])
        assert_allclose(b, c)

    def test_partial_trace_simple_ket(self):
        a = qu.rand_ket(12, sparse=True, density=0.5)
        dims = [2, 3, 2]
        b = qu.partial_trace(a, dims, [0, 1])
        c = qu.partial_trace(a.A, dims, [0, 1])
        assert_allclose(b, c)


class TestChop:
    def test_chop_inplace(self):
        a = qu.qu([-1j, 0.1 + 0.2j])
        qu.chop(a, tol=0.11, inplace=True)
        assert_allclose(a, qu.qu([-1j, 0.2j]))
        # Sparse
        a = qu.qu([-1j, 0.1 + 0.2j], sparse=True)
        qu.chop(a, tol=0.11, inplace=True)
        b = qu.qu([-1j, 0.2j], sparse=True)
        assert((a != b).nnz == 0)

    def test_chop_inplace_dop(self):
        a = qu.qu([1, 0.1], 'dop')
        qu.chop(a, tol=0.11, inplace=True)
        assert_allclose(a, qu.qu([1, 0], 'dop'))
        a = qu.qu([1, 0.1], 'dop', sparse=True)
        qu.chop(a, tol=0.11, inplace=True)
        b = qu.qu([1, 0.0], 'dop', sparse=True)
        assert((a != b).nnz == 0)

    def test_chop_copy(self):
        a = qu.qu([-1j, 0.1 + 0.2j])
        b = qu.chop(a, tol=0.11, inplace=False)
        assert_allclose(a, qu.qu([-1j, 0.1 + 0.2j]))
        assert_allclose(b, qu.qu([-1j, 0.2j]))
        # Sparse
        a = qu.qu([-1j, 0.1 + 0.2j], sparse=True)
        b = qu.chop(a, tol=0.11, inplace=False)
        ao = qu.qu([-1j, 0.1 + 0.2j], sparse=True)
        bo = qu.qu([-1j, 0.2j], sparse=True)
        assert((a != ao).nnz == 0)
        assert((b != bo).nnz == 0)


class TestExpec:
    @mark.parametrize("qtype1", ['ket', 'dop'])
    @mark.parametrize("spars1", [True, False])
    @mark.parametrize("qtype2", ['ket', 'dop'])
    @mark.parametrize("spars2", [True, False])
    def test_all(self, qtype1, spars1, qtype2, spars2):
        a = qu.qu([[1], [2j], [3]], qtype=qtype1, sparse=spars1)
        b = qu.qu([[1j], [2], [3j]], qtype=qtype2, sparse=spars2)
        c = qu.expec(a, b)
        assert not isinstance(c, complex)
        assert_allclose(c, 36)

    @mark.parametrize("qtype", ['ket', 'dop'])
    @mark.parametrize("sparse", [True, False])
    @mark.parametrize("s", ['x', 'y', 'z'])
    def test_negative_expec(self, qtype, sparse, s):
        a = qu.singlet(qtype=qtype)
        b = qu.pauli(s, sparse=sparse) & qu.pauli(s, sparse=sparse)
        assert_allclose(qu.expec(a, b), -1)


class TestNumbaFuncs:

    @mark.parametrize("size", [300, 3000, (300, 5), (3000, 5)])
    @mark.parametrize("X_dtype", ['float32', 'float64',
                                  'complex64', 'complex128'])
    @mark.parametrize("c_dtype", ['float32', 'float64'])
    def test_subtract_update(
        self, size, X_dtype, c_dtype,
    ):
        X = qu.randn(size, dtype=X_dtype)
        Y = qu.randn(size, dtype=X_dtype)
        c = qu.randn(1, dtype=c_dtype).item()
        res = X - c * Y
        qu.core.subtract_update_(X, c, Y)
        assert_allclose(res, X)

    @mark.parametrize("size", [300, 3000, (300, 5), (3000, 5)])
    @mark.parametrize("X_dtype", ['float32', 'float64',
                                  'complex64', 'complex128'])
    @mark.parametrize("c_dtype", ['float32', 'float64'])
    def test_divide_update(
        self, size, X_dtype, c_dtype,
    ):
        X = qu.randn(size, dtype=X_dtype)
        Y = np.empty_like(X)
        c = qu.randn(1, dtype=c_dtype).item()
        res = X / c
        qu.core.divide_update_(X, c, Y)
        assert_allclose(res, Y, rtol=1e-6)
