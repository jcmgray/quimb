import itertools

from pytest import fixture, raises, mark
import scipy.sparse as sp
import numpy as np
from numpy.testing import assert_allclose, assert_almost_equal

from quimb import (
    issparse,
    isherm,
    kron,
    prod,
    mutual_information,
    bell_state,
    rand_rho,
    rand_matrix,
    rand_ket,
    up,
    plus,
    yplus,
    sig,
    singlet,
    qu,
    infer_size,
    itrace,
    trace,
    tr,
    nmlz,
    dim_map,
    dim_compress,
    eye,
    eyepad,
    perm_eyepad,
    permute,
    partial_trace,
    chop,
    expec,
)
from quimb.core import (
    sparse_matrix,
    _trace_dense,
    _trace_sparse,
    _trace_lose,
    _trace_keep,
)

stypes = ("csr", "csc", "bsr", "coo")


@fixture
def od1():
    return rand_matrix(3)


@fixture
def os1():
    return rand_matrix(3, sparse=True, density=0.5)


class TestSparseMatrix:
    @mark.parametrize("stype", stypes)
    def test_simple(self, stype):
        a = sparse_matrix([[0, 3], [1, 2]], stype)
        assert a.format == stype
        assert a.dtype == complex


class TestQuimbify:
    def test_vector_create(self):
        x = [1, 2, 3j]
        p = qu(x, qtype='ket')
        assert(type(p) == np.matrix)
        assert(p.dtype == np.complex)
        assert(p.shape == (3, 1))
        p = qu(x, qtype='bra')
        assert(p.shape == (1, 3))
        assert_almost_equal(p[0, 2], -3.0j)

    def test_dop_create(self):
        x = np.random.randn(3, 3)
        p = qu(x, qtype='dop')
        assert(type(p) == np.matrix)
        assert(p.dtype == np.complex)
        assert(p.shape == (3, 3))

    def test_convert_vector_to_dop(self):
        x = [1, 2, 3j]
        p = qu(x, qtype='r')
        assert_allclose(p, np.matrix([[1. + 0.j, 2. + 0.j, 0. - 3.j],
                                      [2. + 0.j, 4. + 0.j, 0. - 6.j],
                                      [0. + 3.j, 0. + 6.j, 9. + 0.j]]))

    def test_chopped(self):
        x = [9e-16, 1]
        p = qu(x, 'k', chopped=False)
        assert(p[0, 0] != 0.0)
        p = qu(x, 'k', chopped=True)
        assert(p[0, 0] == 0.0)

    def test_normalized(self):
        x = [3j, 4j]
        p = qu(x, 'k', normalized=False)
        assert_almost_equal(tr(p.H @ p), 25.)
        p = qu(x, 'k', normalized=True)
        assert_almost_equal(tr(p.H @ p), 1.)
        p = qu(x, 'dop', normalized=True)
        assert_almost_equal(tr(p), 1.)

    def test_sparse_create(self):
        x = [[1, 0], [3, 0]]
        p = qu(x, 'dop', sparse=False)
        assert(type(p) == np.matrix)
        p = qu(x, 'dop', sparse=True)
        assert(type(p) == sp.csr_matrix)
        assert(p.dtype == np.complex)
        assert(p.nnz == 2)

    def test_sparse_convert_to_dop(self):
        x = [1, 0, 9e-16, 0, 3j]
        p = qu(x, 'ket', sparse=True)
        q = qu(p, 'dop', sparse=True)
        assert(q.shape == (5, 5))
        assert(q.nnz == 9)
        assert_almost_equal(q[4, 4], 9.)
        q = qu(p, 'dop', sparse=True, normalized=True)
        assert_almost_equal(tr(q), 1.)

    @mark.parametrize("qtype, shape, out",
                      (("bra", (1, 4), [[1, 0, 2, -3j]]),
                       ("ket", (4, 1), [[1], [0], [2], [3j]]),
                       ("dop", (4, 4), [[1, 0, 2, -3j],
                                        [0, 0, 0, 0],
                                        [2, 0, 4, -6j],
                                        [3j, 0, 6j, 9]])))
    @mark.parametrize("format_in", stypes)
    @mark.parametrize("format_out", (None,) + stypes)
    def test_reshape_sparse(self, qtype, shape, out, format_in, format_out):
        x = sparse_matrix([[1], [0], [2], [3j]], format_in)
        y = qu(x, qtype=qtype, stype=format_out)
        assert y.shape == shape
        assert y.dtype == complex
        if format_out is None:
            format_out = format_in
        assert y.format == format_out
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
        y = qu(x, qtype=qtype, stype=format_out, sparse=True)
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
        y = qu(x, qtype=qtype, stype=format_out)
        assert issparse(y)
        assert y.shape == shape
        assert y.format == format_out


class TestInferSize:
    @mark.parametrize("d,base,n",
                      ([8, 2, 3],
                       [16, 2, 4],
                       [9, 3, 2],
                       [81, 3, 4]))
    def test_infer_size(self, d, base, n):
        p = rand_ket(d)
        assert infer_size(p, base) == n

    def test_raises(self):
        p = rand_ket(2) & rand_ket(3)
        with raises(ValueError):
            infer_size(p, base=2)


class TestTrace:
    @mark.parametrize("inpt, outpt",
                      ([[[2, 1], [4, 5]], 7],
                       [[[2, 1], [4, 5j]], 2 + 5j]))
    @mark.parametrize("sparse, func",
                      ([False, _trace_dense],
                       [True, _trace_sparse]))
    def test_simple(self, inpt, outpt, sparse, func):
        a = qu(inpt, sparse=sparse)
        assert(trace(a) == outpt)
        assert(a.tr.__code__.co_code == func.__code__.co_code)


class TestITrace:
    @mark.parametrize("axes", [(0, 1), ((0,), (1,))])
    def test_axes_types(self, axes):
        a = rand_matrix(4)
        b = itrace(a, axes)
        assert_allclose(b, np.trace(a))

    def test_complex_dims(self):
        a = np.random.rand(4, 3, 2, 2, 4, 3)
        atr = itrace(a, ((0, 1, 2), (4, 5, 3)))
        btr = np.trace(np.trace(np.trace(a, axis1=1, axis2=5),
                                axis1=1, axis2=2))
        assert_allclose(atr, btr)


class TestNormalize:
    def test_normalize_ket(self):
        a = qu([1, -1j], 'ket')
        b = nmlz(a, inplace=False)
        assert_almost_equal(trace(b.H @ b), 1.0)
        assert_almost_equal(trace(a.H @ a), 2.0)

    def test_normalize_bra(self):
        a = qu([1, -1j], 'bra')
        b = nmlz(a, inplace=False)
        assert_almost_equal(trace(b @ b.H), 1.0)

    def test_normalize_dop(self):
        a = qu([1, -1j], 'dop')
        b = nmlz(a, inplace=False)
        assert_almost_equal(trace(b), 1.0)

    def test_normalize_inplace_ket(self):
        a = qu([1, -1j], 'ket')
        a.nmlz(inplace=True)
        assert_almost_equal(trace(a.H @ a), 1.0)

    def test_normalize_inplace_bra(self):
        a = qu([1, -1j], 'bra')
        a.nmlz(inplace=True)
        assert_almost_equal(trace(a @ a.H), 1.0)

    def test_normalize_inplace_dop(self):
        a = qu([1, -1j], 'dop')
        b = nmlz(a, inplace=True)
        assert_almost_equal(trace(a), 1.0)
        assert_almost_equal(trace(b), 1.0)


class TestDimMap:
    @mark.parametrize("numpy", [False, True])
    def test_1d(self, numpy):
        dims = [10, 11, 12, 13]
        coos = (1, 2, 3)
        if numpy:
            dims, coos = np.asarray(dims), np.asarray(coos)
        ndims, ncoos = dim_map(dims, coos)
        assert_allclose([ndims[c] for c in ncoos], (11, 12, 13))
        coos = ([-1], [2], [5])
        with raises(ValueError):
            ndims, ncoos = dim_map(dims, coos)
        ndims, ncoos = dim_map(dims, coos, cyclic=True)
        assert_allclose([ndims[c] for c in ncoos], (13, 12, 11))
        ndims, ncoos = dim_map(dims, coos, trim=True)
        assert_allclose([ndims[c] for c in ncoos], [12])

    def test_2d(self):
        dims = [[200, 201, 202, 203],
                [210, 211, 212, 213]]
        coos = ((1, 2), (1, 3), (0, 3))
        ndims, ncoos = dim_map(dims, coos)
        assert_allclose([ndims[c] for c in ncoos], (212, 213, 203))
        coos = ((-1, 1), (1, 2), (3, 4))
        with raises(ValueError):
            ndims, ncoos = dim_map(dims, coos)
        ndims, ncoos = dim_map(dims, coos, cyclic=True)
        assert_allclose([ndims[c] for c in ncoos], (211, 212, 210))
        ndims, ncoos = dim_map(dims, coos, trim=True)
        assert_allclose([ndims[c] for c in ncoos], [212])

    def test_3d(self):
        dims = [[[3000, 3001, 3002],
                 [3010, 3011, 3012],
                 [3020, 3021, 3022]],
                [[3100, 3101, 3102],
                 [3110, 3111, 3112],
                 [3120, 3121, 3122]]]
        coos = ((0, 0, 2), (1, 1, 2), (1, 2, 0))
        ndims, ncoos = dim_map(dims, coos)
        assert_allclose([ndims[c] for c in ncoos], (3002, 3112, 3120))
        coos = ((0, -1, 2), (1, 2, 2), (4, -1, 3))
        with raises(ValueError):
            ndims, ncoos = dim_map(dims, coos)
        ndims, ncoos = dim_map(dims, coos, cyclic=True)
        assert_allclose([ndims[c] for c in ncoos], (3022, 3122, 3020))
        ndims, ncoos = dim_map(dims, coos, trim=True)
        assert_allclose([ndims[c] for c in ncoos], [3122])


class TestDimCompress:
    def test_edge(self):
        dims = [2, 3, 2, 4, 5]
        coos = [0, 4]
        ndims, ncoos = dim_compress(dims, coos)
        assert ndims == (2, 24, 5)
        assert ncoos == (0, 2)

    def test_middle(self):
        dims = [5, 3, 2, 5, 4, 3, 2]
        coos = [1, 2, 3, 5]
        ndims, ncoos = dim_compress(dims, coos)
        assert ndims == (5, 30, 4, 3, 2)
        assert ncoos == (1, 3)

    def test_single(self):
        dims = [5, 3, 2, 5, 4, 3, 2]
        coos = 3
        ndims, ncoos = dim_compress(dims, coos)
        assert ndims == (30, 5, 24)
        assert ncoos == (1,)

    @mark.parametrize("dims, inds, ndims, ninds",
                      [([2, 2], [0, 1], (4,), (0,)),
                       ([4], [0], (4,), (0,))])
    def test_tiny(self, dims, inds, ndims, ninds):
        dims, inds = dim_compress(dims, inds)
        assert dims == ndims
        assert inds == ninds


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
    def test_basic(self):
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

    def test_mid_multi(self):
        a = [rand_matrix(2) for i in range(3)]
        i = eye(2)
        dims = [2, 2, 2, 2, 2, 2]
        inds = [1, 2, 4]
        b = eyepad(a, dims, inds)
        assert_allclose(b, i & a[0] & a[1] & i & a[2] & i)

    def test_mid_multi_reverse(self):
        a = [rand_matrix(2) for i in range(3)]
        i = eye(2)
        dims = [2, 2, 2, 2, 2, 2]
        inds = [5, 4, 1]
        b = eyepad(a, dims, inds)
        assert_allclose(b, i & a[2] & i & i & a[1] & a[0])

    def test_auto(self):
        a = rand_matrix(2)
        i = eye(2)
        b = eyepad([a], (2, -1, 2), [1])
        assert_allclose(b, i & a & i)

    def test_ndarrays(self):
        a = rand_matrix(2)
        i = eye(2)
        b = eyepad([a], np.array([2, 2, 2]), [0, 2])
        assert_allclose(b, a & i & a)
        b = eyepad([a], [2, 2, 2], np.array([0, 2]))
        assert_allclose(b, a & i & a)

    def test_overlap(self):
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

    def test_holey_overlap(self):
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

    def test_sparse(self):
        i = eye(2, sparse=True)
        a = qu(rand_matrix(2), sparse=True)
        b = eyepad(a, [2, 2, 2], 1)  # infer sparse
        assert(issparse(b))
        assert_allclose(b.A, (i & a & i).A)
        a = rand_matrix(2)
        b = eyepad(a, [2, 2, 2], 1, sparse=True)  # explicit sparse
        assert(issparse(b))
        assert_allclose(b.A, (i & a & i).A)

    def test_2d_simple(self):
        a = (rand_matrix(2), rand_matrix(2))
        dims = ((2, 3), (3, 2))
        inds = ((0, 0), (1, 1))
        b = eyepad(a, dims, inds)
        assert b.shape == (36, 36)
        assert_allclose(b, a[0] & eye(9) & a[1])

    @mark.parametrize("stype", (None,) + stypes)
    @mark.parametrize("pos", [0, 1, 2,
                              (0,), (1,), (2,),
                              (0, 1),
                              (1, 2),
                              (0, 2)])
    @mark.parametrize("coo_build", [False, True])
    def test_sparse_format_outputs(self, os1, stype, pos, coo_build):
        x = eyepad(os1, [3, 3, 3], pos,
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
        x = eyepad(od1, [3, 3, 3], pos, sparse=True,
                   stype=stype, coo_build=coo_build)
        try:
            default = "bsr" if (2 in pos and not coo_build) else "csr"
        except TypeError:
            default = "bsr" if (pos == 2 and not coo_build) else "csr"
        assert x.format == default if stype is None else stype


class TestPermEyepad:
    def test_dop_spread(self):
        a = rand_rho(4)
        b = perm_eyepad(a, [2, 2, 2], [0, 2])
        c = (a & eye(2)).A.reshape([2, 2, 2, 2, 2, 2])  \
                          .transpose([0, 2, 1, 3, 5, 4])  \
                          .reshape([8, 8])
        assert_allclose(b, c)

    def test_dop_reverse(self):
        a = rand_rho(4)
        b = perm_eyepad(a, np.array([2, 2, 2]), [2, 0])
        c = (a & eye(2)).A.reshape([2, 2, 2, 2, 2, 2])  \
                          .transpose([1, 2, 0, 4, 5, 3])  \
                          .reshape([8, 8])
        assert_allclose(b, c)

    def test_dop_reverse_sparse(self):
        a = rand_rho(4, sparse=True, density=0.5)
        b = perm_eyepad(a, np.array([2, 2, 2]), [2, 0])
        c = (a & eye(2)).A.reshape([2, 2, 2, 2, 2, 2])  \
                          .transpose([1, 2, 0, 4, 5, 3])  \
                          .reshape([8, 8])
        assert_allclose(b.A, c)


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
        a = rand_ket(prod(dims), sparse=True, density=0.5)
        b = permute(a, dims, [3, 1, 2, 0])
        c = permute(a.A, dims, [3, 1, 2, 0])
        assert_allclose(b.A, c)

    def test_permute_sparse_op(self):
        dims = [3, 2, 5, 4]
        a = rand_rho(prod(dims), sparse=True, density=0.5)
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
        a = qu([0.5, 0.5, 0.5, 0.5], 'ket')
        b = partial_trace(a, [2, 2], [0, 1])
        assert_allclose(a @ a.H, b)
        a = qu([0.5, 0.5, 0.5, 0.5], 'dop')
        b = partial_trace(a, [2, 2], [0, 1])
        assert_allclose(a, b)

    def test_partial_trace_return_type(self):
        a = qu([0, 2**-0.5, 2**-0.5, 0], 'ket')
        b = partial_trace(a, [2, 2], 1)
        assert(type(b) == np.matrix)
        a = qu([0, 2**-0.5, 2**-0.5, 0], 'dop')
        b = partial_trace(a, [2, 2], 1)
        assert(type(b) == np.matrix)

    def test_partial_trace_single_ket(self):
        dims = [2, 3, 4]
        a = np.random.randn(prod(dims), 1)
        for i, dim in enumerate(dims):
            b = partial_trace(a, dims, i)
            assert(b.shape[0] == dim)

    def test_partial_trace_multi_ket(self):
        dims = [2, 3, 4]
        a = np.random.randn(prod(dims), 1)
        for i1, i2 in itertools.combinations([0, 1, 2], 2):
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
            assert_allclose(rhoa, eye(2) / 2)

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
        pab = _trace_lose(abc, [2, 3, 2], 2)
        assert_allclose(pab, (a & b).A)
        pac = _trace_lose(abc, [2, 3, 2], 1)
        assert_allclose(pac, (a & c).A)
        pbc = _trace_lose(abc, [2, 3, 2], 0)
        assert_allclose(pbc, (b & c).A)

    def test_bell_state(self):
        a = bell_state('psi-', sparse=True)
        b = _trace_lose(a @ a.H, [2, 2], 0)
        assert_allclose(b, eye(2) / 2)
        b = _trace_lose(a @ a.H, [2, 2], 1)
        assert_allclose(b, eye(2) / 2)

    def test_vs_ptr(self):
        a = rand_rho(6, sparse=True, density=0.5)
        b = _trace_lose(a, [2, 3], 1)
        c = partial_trace(a.A, [2, 3], 0)
        assert_allclose(b, c)
        b = _trace_lose(a, [2, 3], 0)
        c = partial_trace(a.A, [2, 3], 1)
        assert_allclose(b, c)

    def test_vec_dense(self):
        a = rand_ket(4)
        b = _trace_lose(a, [2, 2], 1)
        c = partial_trace(a.A, [2, 2], 0)
        assert_allclose(b, c)
        b = _trace_lose(a, [2, 2], 0)
        c = partial_trace(a.A, [2, 2], 1)
        assert_allclose(b, c)


class TestTraceKeep:
    def test_rps(self):
        a, b, c = (rand_rho(2, sparse=True, density=0.5),
                   rand_rho(3, sparse=True, density=0.5),
                   rand_rho(2, sparse=True, density=0.5))
        abc = a & b & c
        pc = _trace_keep(abc, [2, 3, 2], 2)
        assert_allclose(pc, c.A)
        pb = _trace_keep(abc, [2, 3, 2], 1)
        assert_allclose(pb, b.A)
        pa = _trace_keep(abc, [2, 3, 2], 0)
        assert_allclose(pa, a.A)

    def test_bell_state(self):
        a = bell_state('psi-', sparse=True)
        b = _trace_keep(a @ a.H, [2, 2], 0)
        assert_allclose(b, eye(2) / 2)
        b = _trace_keep(a @ a.H, [2, 2], 1)
        assert_allclose(b, eye(2) / 2)

    def test_vs_ptr(self):
        a = rand_rho(6, sparse=True, density=0.5)
        b = _trace_keep(a, [2, 3], 0)
        c = partial_trace(a.A, [2, 3], 0)
        assert_allclose(b, c)
        b = _trace_keep(a, [2, 3], 1)
        c = partial_trace(a.A, [2, 3], 1)
        assert_allclose(b, c)

    def test_vec_dense(self):
        a = rand_ket(4)
        b = _trace_keep(a, [2, 2], 0)
        c = partial_trace(a.A, [2, 2], 0)
        assert_allclose(b, c)
        b = _trace_keep(a, [2, 2], 1)
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
        a = qu([-1j, 0.1 + 0.2j])
        chop(a, tol=0.11, inplace=True)
        assert_allclose(a, qu([-1j, 0.2j]))
        # Sparse
        a = qu([-1j, 0.1 + 0.2j], sparse=True)
        chop(a, tol=0.11, inplace=True)
        b = qu([-1j, 0.2j], sparse=True)
        assert((a != b).nnz == 0)

    def test_chop_inplace_dop(self):
        a = qu([1, 0.1], 'dop')
        chop(a, tol=0.11, inplace=True)
        assert_allclose(a, qu([1, 0], 'dop'))
        a = qu([1, 0.1], 'dop', sparse=True)
        chop(a, tol=0.11, inplace=True)
        b = qu([1, 0.0], 'dop', sparse=True)
        assert((a != b).nnz == 0)

    def test_chop_copy(self):
        a = qu([-1j, 0.1 + 0.2j])
        b = chop(a, tol=0.11, inplace=False)
        assert_allclose(a, qu([-1j, 0.1 + 0.2j]))
        assert_allclose(b, qu([-1j, 0.2j]))
        # Sparse
        a = qu([-1j, 0.1 + 0.2j], sparse=True)
        b = chop(a, tol=0.11, inplace=False)
        ao = qu([-1j, 0.1 + 0.2j], sparse=True)
        bo = qu([-1j, 0.2j], sparse=True)
        assert((a != ao).nnz == 0)
        assert((b != bo).nnz == 0)


class TestOverlap:
    @mark.parametrize("qtype1", ['ket', 'dop'])
    @mark.parametrize("spars1", [True, False])
    @mark.parametrize("qtype2", ['ket', 'dop'])
    @mark.parametrize("spars2", [True, False])
    def test_all(self, qtype1, spars1, qtype2, spars2):
        a = qu([[1], [2j], [3]], qtype=qtype1, sparse=spars1)
        b = qu([[1j], [2], [3j]], qtype=qtype2, sparse=spars2)
        c = expec(a, b)
        assert not isinstance(c, complex)
        assert_allclose(c, 36)

    @mark.parametrize("qtype", ['ket', 'dop'])
    @mark.parametrize("sparse", [True, False])
    @mark.parametrize("s", ['x', 'y', 'z'])
    def test_negative_expec(self, qtype, sparse, s):
        a = singlet(qtype=qtype)
        b = sig(s, sparse=sparse) & sig(s, sparse=sparse)
        assert_allclose(expec(a, b), -1)
