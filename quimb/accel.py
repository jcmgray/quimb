"""Accelerated helper numerical functions.

These in general do not need to be called directly.
"""
# TODO: merge kron, eyepad --> tensor

import cmath
import functools
import operator

import numpy as np
import scipy.sparse as sp
from numba import njit, vectorize
from numexpr import evaluate
from cytoolz import partition_all

import os
for env_var in ['QUIMB_NUM_THREAD_WORKERS',
                'QUIMB_NUM_PROCS',
                'OMP_NUM_THREADS']:
    if env_var in os.environ:
        _NUM_THREAD_WORKERS = int(os.environ[env_var])
        _NUM_THREAD_WORKERS_SET = True
        break
    _NUM_THREAD_WORKERS_SET = False

if not _NUM_THREAD_WORKERS_SET:
    import psutil
    _NUM_THREAD_WORKERS = psutil.cpu_count(logical=False)


class CacheThreadPool(object):
    """
    """

    def __init__(self, func):
        self._settings = '__UNINITIALIZED__'
        self._pool_fn = func

    def __call__(self, num_threads=None):
        # convert None to default so caches the same
        if num_threads is None:
            num_threads = _NUM_THREAD_WORKERS
        # first call
        if self._settings == '__UNINITIALIZED__':
            self._pool = self._pool_fn(num_threads)
            self._settings = num_threads
        # new type of pool requested
        elif self._settings != num_threads:
            self._pool.shutdown()
            self._pool = self._pool_fn(num_threads)
            self._settings = num_threads
        return self._pool


@CacheThreadPool
def get_thread_pool(num_workers):
    from concurrent.futures import ThreadPoolExecutor
    return ThreadPoolExecutor(num_workers)


def par_reduce(fn, seq, nthreads=_NUM_THREAD_WORKERS):
    """Parallel reduce.

    Parameters
    ----------
    fn : callable
        Two argument function to reduce with.
    seq : sequence
        Sequence to reduce.
    nthreads : int, optional
        The number of threads to reduce with in parallel.

    Returns
    -------
    depends on ``fn`` and ``seq``.

    Notes
    -----
    This has a several hundred microsecond overhead.
    """
    if nthreads == 1:
        return functools.reduce(fn, seq)

    pool = get_thread_pool(nthreads)  # cached

    def _sfn(x):
        """Single call of `fn`, but accounts for the fact
        that can be passed a single item, in which case
        it should not perform the binary operation.
        """
        if len(x) == 1:
            return x[0]
        return fn(*x)

    def _inner_preduce(x):
        """Splits the sequence into pairs and possibly one
        singlet, on each of which `fn` is performed to create
        a new sequence.
        """
        if len(x) < 3:
            return _sfn(x)
        paired_x = partition_all(2, x)
        new_x = tuple(pool.map(_sfn, paired_x))
        return _inner_preduce(new_x)

    return _inner_preduce(tuple(seq))


def prod(xs):
    """Product (as in multiplication) of an iterable.
    """
    return functools.reduce(operator.mul, xs, 1)


def make_immutable(mat):
    """Make matrix read only, in-place.

    Parameters
    ----------
    mat : sparse or dense matrix
        Matrix to make immutable.
    """
    if issparse(mat):
        mat.data.flags.writeable = False
        if mat.format in {'csr', 'csc', 'bsr'}:
            mat.indices.flags.writeable = False
            mat.indptr.flags.writeable = False
        elif mat.format == 'coo':
            mat.row.flags.writeable = False
            mat.col.flags.writeable = False
    else:
        mat.flags.writeable = False


# --------------------------------------------------------------------------- #
# Decorators for standardizing output                                         #
# --------------------------------------------------------------------------- #

def matrixify(fn):
    """Decorator that wraps output as a numpy.matrix.
    """
    @functools.wraps(fn)
    def matrixified_fn(*args, **kwargs):
        out = fn(*args, **kwargs)
        if not isinstance(out, np.matrix):
            return np.asmatrix(out)
        return out
    return matrixified_fn


def realify(fn, imag_tol=1.0e-12):
    """Decorator that drops ``fn``'s output imaginary part if very small.
    """
    @functools.wraps(fn)
    def realified_fn(*args, **kwargs):
        x = fn(*args, **kwargs)
        try:
            return x.real if abs(x.imag) < abs(x.real) * imag_tol else x
        except AttributeError:
            return x
    return realified_fn


def zeroify(fn, tol=1e-14):
    """Decorator that rounds ``fn``'s output to zero if very small.
    """
    @functools.wraps(fn)
    def zeroified_f(*args, **kwargs):
        x = fn(*args, **kwargs)
        return 0.0 if abs(x) < tol else x
    return zeroified_f


def upcast(fn):
    """Decorator to make sure the types of two numpy arguments match.
    """
    def upcasted_fn(a, b):
        if a.dtype == b.dtype:
            return fn(a, b)
        else:
            common = np.common_type(a, b)
            return fn(a.astype(common), b.astype(common))

    return upcasted_fn


# --------------------------------------------------------------------------- #
# Type and shape checks                                                       #
# --------------------------------------------------------------------------- #

def isket(qob):
    """Checks if ``qob`` is in ket form -- a matrix column.
    """
    return qob.shape[0] > 1 and qob.shape[1] == 1  # Column vector check


def isbra(qob):
    """Checks if ``qob`` is in bra form -- a matrix row.
    """
    return qob.shape[0] == 1 and qob.shape[1] > 1  # Row vector check


def isop(qob):
    """Checks if ``qob`` is an operator -- a square matrix.
    """
    m, n = qob.shape
    return m == n and m > 1  # Square matrix check


def isvec(qob):
    """Checks if ``qob`` is row-vector, column-vector or one-dimensional.
    """
    shp = qob.shape
    return len(shp) == 1 or (len(shp) == 2 and (shp[0] == 1 or shp[1] == 1))


def issparse(qob):
    """Checks if ``qob`` is sparse.
    """
    return isinstance(qob, sp.spmatrix)


def isherm(qob):
    """Checks if ``qob`` is hermitian.

    Parameters
    ----------
    qob : dense or sparse matrix
        Matrix to check.

    Returns
    -------
    bool
    """
    return ((qob != qob.H).nnz == 0 if issparse(qob) else
            np.allclose(qob, qob.H))


# --------------------------------------------------------------------------- #
# Core accelerated numeric functions                                          #
# --------------------------------------------------------------------------- #

@matrixify
@njit
def mul_dense(x, y):  # pragma: no cover
    """Numba-accelerated element-wise multiplication of two dense matrices.
    """
    return x * y


def mul(x, y):
    """Element-wise multiplication, dispatched to correct dense or sparse
    function.

    Parameters
    ----------
    x : dense or sparse matrix
        First array.
    y : dense or sparse matrix
        Second array.

    Returns
    -------
    dense or sparse matrix
        Element wise product of ``x`` and ``y``.
    """
    # TODO: add sparse, dense -> sparse w/ broadcasting
    if issparse(x):
        return x.multiply(y)
    elif issparse(y):
        return y.multiply(x)
    return mul_dense(x, y)


@matrixify
@njit
def dot_dense(a, b):  # pragma: no cover
    """Accelerated dense dot product of matrices
    """
    return a @ b


@njit(nogil=True)  # pragma: no cover
def dot_csr_matvec(data, indptr, indices, vec, out, k1k2):
    """Sparse csr matrix-vector dot-product, only acting on range(k1, k2).
    """
    for i in range(*k1k2):
        ri = indptr[i]
        rf = indptr[i + 1]
        isum = 0.0j
        for j in range(rf - ri):
            ri_j = ri + j
            isum += data[ri_j] * vec[indices[ri_j]]
        out[i] = isum


def par_dot_csr_matvec(mat, vec, nthreads=_NUM_THREAD_WORKERS):
    """Parallel sparse csr-matrix vector dot product.

    Parameters
    ----------
    mat : sparse csr-matrix
        Matrix.
    vec : dense vector
        Vector.
    nthreads : int, optional
        Perform in parallel with this many threads.

    Returns
    -------
    dense vector
        Result of ``mat @ vec``.

    Notes
    -----
    The main bottleneck for sparse matrix vector product is memory access,
    as such this function is only beneficial for very large matrices.
    """
    vec_shape = vec.shape
    vec_matrix = True if isinstance(vec, np.matrix) else False

    sz = mat.shape[0]
    out = np.empty(sz, dtype=vec.dtype)
    sz_chnk = (sz // nthreads) + 1
    slices = tuple((i * sz_chnk, min((i + 1) * sz_chnk, sz))
                   for i in range(nthreads))

    fn = functools.partial(dot_csr_matvec,
                           mat.data,
                           mat.indptr,
                           mat.indices,
                           np.asarray(vec).reshape(-1), out)

    pool = get_thread_pool(nthreads)
    tuple(pool.map(fn, slices))

    if out.shape != vec_shape:
        out = out.reshape(*vec_shape)
    if vec_matrix:
        out = np.asmatrix(out)

    return out


def dot_sparse(a, b):
    """Dot product for sparse matrix, dispatching to parallel for v large nnz.
    """
    use_parallel = (issparse(a) and
                    isvec(b) and
                    a.nnz > 500000 and
                    _NUM_THREAD_WORKERS > 1)
    if use_parallel:  # pragma: no cover
        return par_dot_csr_matvec(a, b)
    return a @ b


def dot(a, b):
    """Matrix multiplication, dispatched to dense or sparse functions.

    Parameters
    ----------
    a : dense or sparse matrix
        First array.
    b : dense or sparse matrix
        Second array.

    Returns
    -------
    dense or sparse matrix
        Dot product of ``a`` and ``b``.
    """
    if issparse(a) or issparse(b):
        return dot_sparse(a, b)
    return dot_dense(a, b)


@realify
@upcast
@njit
def vdot(a, b):  # pragma: no cover
    """Accelerated 'Hermitian' inner product of two vectors.

    In other words, ``b`` here will be conjugated by the function.
    """
    return np.vdot(a.reshape(-1), b.reshape(-1))


@realify
@njit
def rdot(a, b):  # pragma: no cover
    """Real dot product of two dense vectors.

    Here, ``b`` will *not* be conjugated before the inner product.
    """
    a, b = a.reshape((1, -1)), b.reshape((-1, 1))
    return (a @ b)[0, 0]


@njit
def reshape_for_ldmul(vec):  # pragma: no cover
    """Reshape a vector to be broadcast multiplied against a matrix in a way
    that replicates left diagonal matrix multiplication.
    """
    d = vec.size
    return d, vec.reshape(d, 1)


@matrixify
def l_diag_dot_dense(diag, mat):
    """Dot product of digonal matrix (with only diagonal supplied) and dense
    matrix.
    """
    d, diag = reshape_for_ldmul(diag)
    return evaluate("diag * mat") if d > 500 else mul_dense(diag, mat)


def l_diag_dot_sparse(diag, mat):
    """Dot product of digonal matrix (with only diagonal supplied) and sparse
    matrix.
    """
    return sp.diags(diag) @ mat


def ldmul(diag, mat):
    """Accelerated left diagonal multiplication using numexp.

    Equivalent to ``numpy.diag(diag) @ mat``, but faster than numpy
    for n > ~ 500.

    Parameters
    ----------
    diag : vector or 1d-array
        Vector representing the diagonal of a matrix.
    mat : dense or sparse matrix
        A normal (non-diagonal) matrix.

    Returns
    -------
    dense or sparse matrix
        Dot product of the matrix whose diagonal is ``diag`` and ``mat``.
    """
    if issparse(mat):
        return l_diag_dot_sparse(diag, mat)
    return l_diag_dot_dense(diag, mat)


@njit
def reshape_for_rdmul(vec):  # pragma: no cover
    """Reshape a vector to be broadcast multiplied against a matrix in a way
    that replicates right diagonal matrix multiplication.
    """
    d = vec.size
    return d, vec.reshape(1, d)


@matrixify
def r_diag_dot_dense(mat, diag):
    """Dot product of dense matrix and digonal matrix (with only diagonal
    supplied).
    """
    d, diag = reshape_for_rdmul(diag)
    return evaluate("mat * diag") if d > 500 else mul_dense(mat, diag)


def r_diag_dot_sparse(mat, diag):
    """Dot product of sparse matrix and digonal matrix (with only diagonal
    supplied).
    """
    return mat @ sp.diags(diag)


def rdmul(mat, diag):
    """Accelerated left diagonal multiplication using numexpr.

    Equivalent to ``mat @ numpy.diag(diag)``, but faster than numpy
    for n > ~ 500.

    Parameters
    ----------
    mat : dense or sparse matrix
        A normal (non-diagonal) matrix.
    diag : vector or 1d-array
        Vector representing the diagonal of a matrix.

    Returns
    -------
    dense or sparse matrix
        Dot product of ``mat`` and the matrix whose diagonal is ``diag``.
    """
    if issparse(mat):
        return r_diag_dot_sparse(mat, diag)
    return r_diag_dot_dense(mat, diag)


@njit
def reshape_for_outer(a, b):  # pragma: no cover
    """Reshape two vectors for an outer product.
    """
    d = a.size
    return d, a.reshape(d, 1), b.reshape(1, d)


def outer(a, b):
    """Outer product between two vectors (no conjugation).
    """
    d, a, b = reshape_for_outer(a, b)
    return mul_dense(a, b) if d < 500 else np.asmatrix(evaluate('a * b'))


@vectorize(nopython=True)
def explt(l, t):  # pragma: no cover
    """Complex exponenital as used in solution to schrodinger equation.
    """
    return cmath.exp((-1.0j * t) * l)


# --------------------------------------------------------------------------- #
# Kronecker (tensor) product                                                  #
# --------------------------------------------------------------------------- #

@njit
def reshape_for_kron(a, b):  # pragma: no cover
    """Reshape two arrays for a 'broadcast' tensor (kronecker) product.

    Returns the expected new dimensions as well.
    """
    m, n = a.shape
    p, q = b.shape
    a = a.reshape((m, 1, n, 1))
    b = b.reshape((1, p, 1, q))
    return a, b, m * p, n * q


@matrixify
@njit
def kron_dense(a, b):  # pragma: no cover
    """Tensor (kronecker) product of two dense arrays.
    """
    a, b, mp, nq = reshape_for_kron(a, b)
    return (a * b).reshape((mp, nq))


@matrixify
def kron_dense_big(a, b):
    """Parallelized (using numpexpr) tensor (kronecker) product for two
    dense arrays.
    """
    a, b, mp, nq = reshape_for_kron(a, b)
    return evaluate('a * b').reshape((mp, nq))


def kron_sparse(a, b, stype=None):
    """Sparse tensor (kronecker) product,

    Output format can be specified or will be automatically determined.
    """
    if stype is None:
        stype = ("bsr" if isinstance(b, np.ndarray) or b.format == 'bsr' else
                 b.format if isinstance(a, np.ndarray) else
                 "csc" if a.format == "csc" and b.format == "csc" else
                 "csr")

    return sp.kron(a, b, format=stype)


def kron_dispatch(a, b, stype=None):
    """Kronecker product of two arrays, dispatched based on dense/sparse and
    also size of product.
    """
    if issparse(a) or issparse(b):
        return kron_sparse(a, b, stype=stype)
    elif a.size * b.size > 23000:  # pragma: no cover
        return kron_dense_big(a, b)
    else:
        return kron_dense(a, b)


# --------------------------------------------------------------------------- #
# MONKEY-PATCHES                                                              #
# --------------------------------------------------------------------------- #

# Map unused & symbol to tensor product
np.matrix.__and__ = kron_dispatch
sp.csr_matrix.__and__ = kron_dispatch
sp.bsr_matrix.__and__ = kron_dispatch
sp.csc_matrix.__and__ = kron_dispatch
sp.coo_matrix.__and__ = kron_dispatch
