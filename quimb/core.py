"""Core functions for manipulating quantum objects.
"""

import os
import math
import cmath
import operator
import itertools
import functools
from numbers import Integral

import numpy as np
from numpy.matlib import zeros
import scipy.sparse as sp
from numexpr import evaluate
from cytoolz import partition_all


# --------------------------------------------------------------------------- #
#                            Accelerated Functions                            #
# --------------------------------------------------------------------------- #

for env_var in ['QUIMB_NUM_THREAD_WORKERS',
                'QUIMB_NUM_PROCS',
                'OMP_NUM_THREADS']:
    if env_var in os.environ:
        _NUM_THREAD_WORKERS = int(os.environ[env_var])
        break
else:
    import psutil
    _NUM_THREAD_WORKERS = psutil.cpu_count(logical=False)

os.environ['NUMBA_NUM_THREADS'] = str(_NUM_THREAD_WORKERS)

# need to set NUMBA_NUM_THREADS first
import numba as nb  # noqa

_NUMBA_CACHE = {
    'True': True, 'False': False,
}[os.environ.get('QUIMB_NUMBA_CACHE', 'True')]

njit = functools.partial(nb.njit, cache=_NUMBA_CACHE)
"""Numba no-python jit, but obeying cache setting."""

njit_nocache = functools.partial(nb.njit, cache=False)
"""No cache alias of njit."""

vectorize = functools.partial(nb.vectorize, cache=_NUMBA_CACHE)
"""Numba vectorize, but obeying cache setting."""

vectorize_nocache = functools.partial(nb.vectorize, cache=False)
"""No cache alias of vectorize."""


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
def get_thread_pool(num_workers=None):
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
        if len(x) <= 2:
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
    """Make array read only, in-place.

    Parameters
    ----------
    mat : sparse or dense array
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


class qarray(np.ndarray):
    """Thin subclass of :class:`numpy.ndarray` with some convenient quantum
    linear algebra related methods and attributes (``.H``, ``&``, etc.), and
    matrix-like preservation of at least 2-dimensions so as to distiguish
    kets and bras.
    """

    def __new__(cls, data, dtype=None, order=None):
        return np.asarray(data, dtype=dtype, order=order).view(cls)

    @property
    def H(self):
        if issubclass(self.dtype.type, np.complexfloating):
            return self.conjugate().transpose()
        else:
            return self.transpose()

    @property
    def A(self):
        return np.asarray(self)

    def __and__(self, other):
        return kron_dispatch(self, other)

    def normalize(self, inplace=True):
        return normalize(self, inplace=inplace)

    def nmlz(self, inplace=True):
        return normalize(self, inplace=inplace)

    def chop(self, inplace=True):
        return chop(self, inplace=inplace)

    def tr(self):
        return _trace_dense(self)

    def partial_trace(self, dims, keep):
        return partial_trace(self, dims, keep)

    def ptr(self, dims, keep):
        return partial_trace(self, dims, keep)

    def __str__(self):
        with np.printoptions(precision=6, linewidth=120):
            return super().__str__()

    def __repr__(self):
        with np.printoptions(precision=6, linewidth=120):
            return super().__repr__()

# --------------------------------------------------------------------------- #
# Decorators for standardizing output                                         #
# --------------------------------------------------------------------------- #


def ensure_qarray(fn):
    """Decorator that wraps output as a ``qarray``.
    """

    @functools.wraps(fn)
    def qarray_fn(*args, **kwargs):
        out = fn(*args, **kwargs)
        if not isinstance(out, qarray):
            return qarray(out)
        return out

    return qarray_fn


def realify_scalar(x, imag_tol=1e-12):
    try:
        return x.real if abs(x.imag) < abs(x.real) * imag_tol else x
    except AttributeError:
        return x


def realify(fn, imag_tol=1e-12):
    """Decorator that drops ``fn``'s output imaginary part if very small.
    """
    @functools.wraps(fn)
    def realified_fn(*args, **kwargs):
        return realify_scalar(fn(*args, **kwargs), imag_tol=imag_tol)

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

def dag(qob):
    """Conjugate transpose.
    """
    try:
        return qob.H
    except AttributeError:
        return qob.conj().T


def isket(qob):
    """Checks if ``qob`` is in ket form -- an array column.
    """
    return qob.shape[0] > 1 and qob.shape[1] == 1  # Column vector check


def isbra(qob):
    """Checks if ``qob`` is in bra form -- an array row.
    """
    return qob.shape[0] == 1 and qob.shape[1] > 1  # Row vector check


def isop(qob):
    """Checks if ``qob`` is an operator.
    """
    s = qob.shape
    return len(s) == 2 and (s[0] > 1) and (s[1] > 1)


def isvec(qob):
    """Checks if ``qob`` is row-vector, column-vector or one-dimensional.
    """
    shp = qob.shape
    return len(shp) == 1 or (len(shp) == 2 and (shp[0] == 1 or shp[1] == 1))


def issparse(qob):
    """Checks if ``qob`` is explicitly sparse.
    """
    return isinstance(qob, sp.spmatrix)


def isdense(qob):
    """Checks if ``qob`` is explicitly dense.
    """
    return isinstance(qob, np.ndarray)


def isreal(qob, **allclose_opts):
    """Checks if ``qob`` is approximately real.
    """
    data = qob.data if issparse(qob) else qob

    # check dtype
    if np.isrealobj(data):
        return True

    # else check explicitly
    return np.allclose(data.imag, 0.0, **allclose_opts)


def allclose_sparse(A, B, **allclose_opts):

    if A.shape != B.shape:
        return False

    r1, c1, v1 = sp.find(A)
    r2, c2, v2 = sp.find(B)
    index_match = np.array_equal(r1, r2) & np.array_equal(c1, c2)

    if not index_match:
        return False

    return np.allclose(v1, v2, **allclose_opts)


def isherm(qob, **allclose_opts):
    """Checks if ``qob`` is hermitian.

    Parameters
    ----------
    qob : dense or sparse operator
        Matrix to check.

    Returns
    -------
    bool
    """
    if issparse(qob):
        return allclose_sparse(qob, dag(qob), **allclose_opts)
    else:
        return np.allclose(qob, dag(qob), **allclose_opts)


def ispos(qob, tol=1e-15):
    """Checks if the dense hermitian ``qob`` is approximately positive
    semi-definite, using the cholesky decomposition.

    Parameters
    ----------
    qob : dense operator
        Matrix to check.

    Returns
    -------
    bool
    """
    try:
        np.linalg.cholesky(qob + tol * np.eye(qob.shape[0]))
        return True
    except np.linalg.LinAlgError:
        return False


# --------------------------------------------------------------------------- #
# Core accelerated numeric functions                                          #
# --------------------------------------------------------------------------- #

def _nb_complex_base(real, imag):
    return real + 1j * imag


_cmplx_sigs = ['complex64(float32, float32)', 'complex128(float64, float64)']
_nb_complex_seq = vectorize(_cmplx_sigs)(_nb_complex_base)
_nb_complex_par = vectorize_nocache(
    _cmplx_sigs, target='parallel')(_nb_complex_base)


def complex_array(real, imag):
    """Accelerated creation of complex array.
    """
    if real.size > 50000:
        return _nb_complex_par(real, imag)
    return _nb_complex_seq(real, imag)


@ensure_qarray
@upcast
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
    x : dense or sparse operator
        First array.
    y : dense or sparse operator
        Second array.

    Returns
    -------
    dense or sparse operator
        Element wise product of ``x`` and ``y``.
    """
    # dispatch to sparse methods
    if issparse(x):
        return x.multiply(y)
    elif issparse(y):
        return y.multiply(x)

    return mul_dense(x, y)


def _nb_subtract_update_base(X, c, Z):
    return X - c * Z


_sbtrct_sigs = ['float32(float32, float32, float32)',
                'float64(float64, float64, float64)',
                'complex64(complex64, float32, complex64)',
                'complex128(complex128, float64, complex128)']
_nb_subtract_update_seq = vectorize(
    _sbtrct_sigs, target='cpu')(_nb_subtract_update_base)
_nb_subtract_update_par = vectorize_nocache(
    _sbtrct_sigs, target='parallel')(_nb_subtract_update_base)


def subtract_update_(X, c, Y):
    """Accelerated inplace computation of ``X -= c * Y``. This is mainly
    for Lanczos iteration.
    """
    if X.size > 2048:
        _nb_subtract_update_par(X, c, Y, out=X)
    else:
        _nb_subtract_update_seq(X, c, Y, out=X)


def _nb_divide_update_base(X, c):
    return X / c


_divd_sigs = ['float32(float32, float32)',
              'float64(float64, float64)',
              'complex64(complex64, float32)',
              'complex128(complex128, float64)']
_nb_divide_update_seq = vectorize(
    _divd_sigs, target='cpu')(_nb_divide_update_base)
_nb_divide_update_par = vectorize_nocache(
    _divd_sigs, target='parallel')(_nb_divide_update_base)


def divide_update_(X, c, out):
    """Accelerated computation of ``X / c`` into ``out``.
    """
    if X.size > 2048:
        _nb_divide_update_par(X, c, out=out)
    else:
        _nb_divide_update_seq(X, c, out=out)


@njit(parallel=True)  # pragma: no cover
def _dot_csr_matvec_prange(data, indptr, indices, vec, out):
    for i in nb.prange(vec.size):
        isum = 0.0
        for j in range(indptr[i], indptr[i + 1]):
            isum += data[j] * vec[indices[j]]
        out[i] = isum


def par_dot_csr_matvec(A, x):
    """Parallel sparse csr-matrix vector dot product.

    Parameters
    ----------
    A : scipy.sparse.csr_matrix
        Operator.
    x : dense vector
        Vector.

    Returns
    -------
    dense vector
        Result of ``A @ x``.

    Notes
    -----
    The main bottleneck for sparse matrix vector product is memory access,
    as such this function is only beneficial for pretty large matrices.
    """
    y = np.empty(x.size, np.find_common_type([], [A.dtype, x.dtype]))
    _dot_csr_matvec_prange(A.data, A.indptr, A.indices, x.ravel(), y)
    y.shape = x.shape
    if isinstance(x, qarray):
        y = qarray(y)
    return y


def dot_sparse(a, b):
    """Dot product for sparse matrix, dispatching to parallel for v large nnz.
    """
    out = a @ b

    if isdense(out) and (isinstance(b, qarray) or isinstance(a, qarray)):
        out = qarray(out)

    return out


def dot(a, b):
    """Matrix multiplication, dispatched to dense or sparse functions.

    Parameters
    ----------
    a : dense or sparse operator
        First array.
    b : dense or sparse operator
        Second array.

    Returns
    -------
    dense or sparse operator
        Dot product of ``a`` and ``b``.
    """
    if issparse(a) or issparse(b):
        return dot_sparse(a, b)
    try:
        return a.dot(b)
    except AttributeError:
        return a @ b


@realify
def vdot(a, b):
    """Accelerated 'Hermitian' inner product of two arrays. In other words,
    ``b`` here will be conjugated by the function.
    """
    return np.vdot(a.ravel(), b.ravel())


@realify
@upcast
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


@ensure_qarray
def l_diag_dot_dense(diag, mat):
    """Dot product of diagonal matrix (with only diagonal supplied) and dense
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


@ensure_qarray
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
    return mul_dense(a, b) if d < 500 else qarray(evaluate('a * b'))


@vectorize
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


@ensure_qarray
@upcast
@njit
def kron_dense(a, b):  # pragma: no cover
    """Tensor (kronecker) product of two dense arrays.
    """
    a, b, mp, nq = reshape_for_kron(a, b)
    return (a * b).reshape((mp, nq))


@ensure_qarray
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
#                                Core Functions                               #
# --------------------------------------------------------------------------- #

_SPARSE_CONSTRUCTORS = {"csr": sp.csr_matrix,
                        "bsr": sp.bsr_matrix,
                        "csc": sp.csc_matrix,
                        "coo": sp.coo_matrix}


def sparse_matrix(data, stype="csr", dtype=complex):
    """Construct a sparse matrix of a particular format.

    Parameters
    ----------
    data : array_like
        Fed to scipy.sparse constructor.
    stype : {'csr', 'csc', 'coo', 'bsr'}, optional
        Sparse format.

    Returns
    -------
    scipy sparse matrix
        Of format ``stype``.
    """
    return _SPARSE_CONSTRUCTORS[stype](data, dtype=dtype)


_EXPEC_METHODS = {
    # [isop(a), isop(b), issparse(a) or issparse(b)]
    (0, 0, 0): lambda a, b: abs(vdot(a, b))**2,
    (0, 1, 0): lambda a, b: vdot(a, b @ a),
    (1, 0, 0): lambda a, b: vdot(b, a @ b),
    (1, 1, 0): lambda a, b: _trace_dense(a @ b),
    (0, 0, 1): lambda a, b: abs(dot(dag(a), b)[0, 0])**2,
    (0, 1, 1): realify(lambda a, b: dot(dag(a), dot(b, a))[0, 0]),
    (1, 0, 1): realify(lambda a, b: dot(dag(b), dot(a, b))[0, 0]),
    (1, 1, 1): lambda a, b: _trace_sparse(dot(a, b)),
}


def expectation(a, b):
    """'Expectation' between a vector/operator and another vector/operator.

    The 'operator' inner product between ``a`` and ``b``, but also for vectors.
    This means that for consistency:

    - for two vectors it will be the absolute expec squared ``|<a|b><b|a>|``,
      *not* ``<a|b>``.
    - for a vector and an operator its will be ``<a|b|a>``
    - for two operators it will be the Hilbert-schmidt inner product
      ``tr(A @ B)``

    In this way ``expectation(a, b) == expectation(dop(a), b) ==
    expectation(dop(a), dop(b))``.

    Parameters
    ----------
    a : vector or operator
        First state or operator - assumed to be ket if vector.
    b : vector or operator
        Second state or operator - assumed to be ket if vector.

    Returns
    -------
    x : float
        'Expectation' of ``a`` with ``b``.
    """
    return _EXPEC_METHODS[isop(a), isop(b), issparse(a) or issparse(b)](a, b)


expec = expectation
"""Alias for :func:`expectation`."""


def normalize(qob, inplace=True):
    """Normalize a quantum object.

    Parameters
    ----------
    qob : dense or sparse vector or operator
        Quantum object to normalize.
    inplace : bool, optional
        Whether to act inplace on the given operator.

    Returns
    -------
    dense or sparse vector or operator
        Normalized quantum object.
    """
    if not inplace:
        qob = qob.copy()

    if isop(qob):
        n_factor = trace(qob)
    else:
        n_factor = expectation(qob, qob)**0.25

    qob[:] /= n_factor
    return qob


normalize_ = functools.partial(normalize, inplace=True)


def chop(qob, tol=1.0e-15, inplace=True):
    """Set small values of a dense or sparse array to zero.

    Parameters
    ----------
    qob : dense or sparse vector or operator
        Quantum object to chop.
    tol : float, optional
        Fraction of ``max(abs(qob))`` to chop below.
    inplace : bool, optional
        Whether to act on input array or return copy.

    Returns
    -------
    dense or sparse vector or operator
        Chopped quantum object.
    """
    minm = np.abs(qob).max() * tol  # minimum value tolerated
    if not inplace:
        qob = qob.copy()
    if issparse(qob):
        qob.data.real[np.abs(qob.data.real) < minm] = 0.0
        qob.data.imag[np.abs(qob.data.imag) < minm] = 0.0
        qob.eliminate_zeros()
    else:
        qob.real[np.abs(qob.real) < minm] = 0.0
        qob.imag[np.abs(qob.imag) < minm] = 0.0
    return qob


chop_ = functools.partial(chop, inplace=True)


def quimbify(data, qtype=None, normalized=False, chopped=False,
             sparse=None, stype=None, dtype=complex):
    """Converts data to 'quantum' i.e. complex matrices, kets being columns.

    Parameters
    ----------
    data : dense or sparse array_like
        Array describing vector or operator.
    qtype : {``'ket'``, ``'bra'`` or ``'dop'``}, optional
        Quantum object type output type. Note that if an operator is given
        as ``data`` and ``'ket'`` or ``'bra'`` as ``qtype``, the operator
        will be unravelled into a column or row vector.
    sparse : bool, optional
        Whether to convert output to sparse a format.
    normalized : bool, optional
        Whether to normalise the output.
    chopped : bool, optional
        Whether to trim almost zero entries of the output.
    stype : {``'csr'``, ``'csc'``, ``'bsr'``, ``'coo'``}, optional
        Format of output matrix if sparse, defaults to ``'csr'``.

    Returns
    -------
    dense or sparse vector or operator

    Notes
    -----
    1. Will unravel an array if ``'ket'`` or ``'bra'`` given.
    2. Will conjugate if ``'bra'`` given.
    3. Will leave operators as is if ``'dop'`` given, but construct one if
       vector given with the assumption that it was a ket.

    Examples
    --------

    Create a ket (column vector):

    >>> qu([1, 2j, 3])
    qarray([[1.+0.j],
            [0.+2.j],
            [3.+0.j]])

    Create a single precision bra (row vector):

    >>> qu([1, 2j, 3], qtype='bra', dtype='complex64')
    qarray([[1.-0.j, 0.-2.j, 3.-0.j]], dtype=complex64)

    Create a density operator from a vector:

    >>> qu([1, 2j, 3], qtype='dop')
    qarray([[1.+0.j, 0.-2.j, 3.+0.j],
            [0.+2.j, 4.+0.j, 0.+6.j],
            [3.+0.j, 0.-6.j, 9.+0.j]])

    Create a sparse density operator:

    >>> qu([1, 0, 0], sparse=True, qtype='dop')
    <3x3 sparse matrix of type '<class 'numpy.complex128'>'
        with 1 stored elements in Compressed Sparse Row format>
    """

    sparse_input = issparse(data)
    sparse_output = ((sparse) or
                     (sparse_input and sparse is None) or
                     (sparse is None and stype))
    # Infer output sparse format from input if necessary
    if sparse_input and sparse_output and stype is None:
        stype = data.format

    if (qtype is None) and (np.ndim(data) == 1):
        # assume quimbify simple list -> ket
        qtype = 'ket'

    if qtype is not None:
        # Must be dense to reshape
        data = qarray(data.A if sparse_input else data)
        if qtype in ("k", "ket"):
            data = data.reshape((prod(data.shape), 1))
        elif qtype in ("b", "bra"):
            data = data.reshape((1, prod(data.shape))).conj()
        elif qtype in ("d", "r", "rho", "op", "dop") and isvec(data):
            data = dot(quimbify(data, "ket"), quimbify(data, "bra"))
        data = data.astype(dtype)

    # Just cast as qarray
    elif not sparse_output:
        data = qarray(data.A if sparse_input else data, dtype=dtype)

    # Check if already sparse matrix, or wanted to be one
    if sparse_output:
        data = sparse_matrix(data, dtype=dtype,
                             stype=(stype if stype is not None else "csr"))

    # Optionally normalize and chop small components
    if normalized:
        normalize_(data)
    if chopped:
        chop_(data)

    return data


qu = quimbify
"""Alias of :func:`quimbify`."""

ket = functools.partial(quimbify, qtype='ket')
"""Convert an object into a ket."""

bra = functools.partial(quimbify, qtype='bra')
"""Convert an object into a bra."""

dop = functools.partial(quimbify, qtype='dop')
"""Convert an object into a density operator."""

sparse = functools.partial(quimbify, sparse=True)
"""Convert an object into sparse form."""


def infer_size(p, base=2):
    """Infer the size, i.e. number of 'sites' in a state.

    Parameters
    ----------
    p : vector or operator
        An array representing a state with a shape attribute.
    base : int, optional
        Size of the individual states that ``p`` is composed of, e.g. this
        defauts 2 for qubits.

    Returns
    -------
    int
        Number of composite systems.

    Examples
    --------
    >>> infer_size(singlet() & singlet())
    4

    >>> infersize(rand_rho(5**3), base=5)
    3
    """
    sz = math.log(max(p.shape), base)

    if sz % 1 > 1e-13:
        raise ValueError("This state does not seem to be composed of sites"
                         "of equal size {}.".format(base))

    return int(sz)


@realify
@njit
def _trace_dense(op):  # pragma: no cover
    """Trace of a dense operator.
    """
    x = 0.0
    for i in range(op.shape[0]):
        x += op[i, i]
    return x


@realify
def _trace_sparse(op):
    """Trace of a sparse operator.
    """
    return np.sum(op.diagonal())


def trace(mat):
    """Trace of a dense or sparse operator.

    Parameters
    ----------
    mat : operator
        Operator, dense or sparse.

    Returns
    -------
    x : float
        Trace of ``mat``
    """
    return _trace_sparse(mat) if issparse(mat) else _trace_dense(mat)


@ensure_qarray
def _identity_dense(d, dtype=complex):
    """Returns a dense, identity of given dimension ``d`` and type ``dtype``.
    """
    return np.eye(d, dtype=dtype)


def _identity_sparse(d, stype="csr", dtype=complex):
    """Returns a sparse, complex identity of order d.
    """
    return sp.eye(d, dtype=dtype, format=stype)


def identity(d, sparse=False, stype="csr", dtype=complex):
    """Return identity of size d in complex format, optionally sparse.

    Parameters
    ----------
    d : int
        Dimension of identity.
    sparse : bool, optional
        Whether to output in sparse form.
    stype : str, optional
        If sparse, what format to use.

    Returns
    -------
    id : qarray or sparse matrix
        Identity operator.
    """
    if sparse:
        return _identity_sparse(d, stype=stype, dtype=dtype)

    return _identity_dense(d, dtype=dtype)


eye = identity
"""Alias for :func:`identity`."""

speye = functools.partial(identity, sparse=True)
"""Sparse identity."""


def _kron_core(*ops, stype=None, coo_build=False, parallel=False):
    """Core kronecker product for a sequence of objects.
    """
    tmp_stype = "coo" if coo_build or stype == "coo" else None
    reducer = par_reduce if parallel else functools.reduce
    return reducer(functools.partial(kron_dispatch, stype=tmp_stype), ops)


def dynal(x, bases):
    """Generate 'dynamic decimal' for ``x`` given ``dims``.

    Examples
    --------
    >>> dims = [13, 2, 7, 3, 10]
    >>> prod(dims)  # total hilbert space size
    5460

    >>> x = 3279
    >>> drep = list(dyn_bin(x, dims))  # dyn bases repr
    >>> drep
    [7, 1, 4, 0, 9]

    >>> bs_szs = [prod(dims[i + 1:]) for i in range(len(dims))]
    >>> bs_szs
    [420, 210, 30, 10, 1]

    >>> # reconstruct x
    >>> sum(d * b for d, b in zip(drep, bs_szs))
    3279
    """
    bs_szs = [prod(bases[i + 1:]) for i in range(len(bases))]

    for b in bs_szs:
        div = x // b
        yield div
        x -= div * b


def gen_matching_dynal(ri, rf, dims):
    """Return the matching dynal part of ``ri`` and ``rf``, plus the first pair
    that don't match.
    """
    for d1, d2 in zip(dynal(ri, dims), dynal(rf, dims)):
        if d1 == d2:
            yield (d1, d2)
        else:
            yield (d1, d2)
            break


def gen_ops_maybe_sliced(ops, ix):
    """Take ``ops`` and slice the first few, according to the length of ``ix``
    and with ``ix``, and leave the rest.
    """
    for op, i in itertools.zip_longest(ops, ix):
        if i is not None:
            d1, d2 = i
            # can't slice coo matrices
            if sp.isspmatrix_coo(op):
                yield op.tocsr()[slice(d1, d2 + 1), :].tocoo()
            else:
                yield op[slice(d1, d2 + 1), :]
        else:
            yield op


def kron(*ops, stype=None, coo_build=False, parallel=False, ownership=None):
    """Tensor (kronecker) product of variable number of arguments.

    Parameters
    ----------
    ops : sequence of vectors or matrices
        Objects to be tensored together.
    stype : str, optional
        Desired output format if resultant object is sparse. Should be one
        of {``'csr'``, ``'bsr'``, ``'coo'``, ``'csc'``}. If ``None``, infer
        from input matrices.
    coo_build : bool, optional
        Whether to force sparse construction to use the ``'coo'``
        format (only for sparse matrices in the first place.).
    parallel : bool, optional
        Perform a parallel reduce on the operators, can be quicker.
    ownership : (int, int), optional
        If given, only construct the rows in ``range(*ownership)``. Such that
        the  final operator is actually ``X[slice(*ownership), :]``. Useful for
        constructing operators in parallel, e.g. for MPI.

    Returns
    -------
    X : dense or sparse vector or operator
        Tensor product of ``ops``.

    Notes
    -----
    1. The product is performed as ``(a & (b & (c & ...)))``

    Examples
    --------
    Simple example:

    >>> a = np.array([[1, 2], [3, 4]])
    >>> b = np.array([[1., 1.1], [1.11, 1.111]])
    >>> kron(a, b)
    qarray([[1.   , 1.1  , 2.   , 2.2  ],
            [1.11 , 1.111, 2.22 , 2.222],
            [3.   , 3.3  , 4.   , 4.4  ],
            [3.33 , 3.333, 4.44 , 4.444]])

    Partial construction of rows:

    >>> ops = [rand_matrix(2, sparse=True) for _ in range(10)]
    >>> kron(*ops, ownership=(256, 512))
    <256x1024 sparse matrix of type '<class 'numpy.complex128'>'
            with 13122 stored elements in Compressed Sparse Row format>
    """
    core_kws = {'coo_build': coo_build, 'stype': stype, 'parallel': parallel}

    if ownership is None:
        X = _kron_core(*ops, **core_kws)
    else:
        ri, rf = ownership
        dims = [op.shape[0] for op in ops]

        D = prod(dims)
        if not ((0 <= ri < D) and ((0 < rf <= D))):
            raise ValueError(
                "Ownership ({}, {}) not in range [0-{}].".format(ri, rf, D))

        matching_dyn = tuple(gen_matching_dynal(ri, rf - 1, dims))
        sliced_ops = list(gen_ops_maybe_sliced(ops, matching_dyn))
        X = _kron_core(*sliced_ops, **core_kws)

        # check if the kron has naturally oversliced
        if matching_dyn:
            mtchn_bs = [prod(dims[i + 1:]) for i in range(len(matching_dyn))]
            coeffs_bases = tuple(zip(mtchn_bs, matching_dyn))
            ri_got = sum(d * b[0] for d, b in coeffs_bases)
            rf_got = sum(d * b[1] for d, b in coeffs_bases) + mtchn_bs[-1]
        else:
            ri_got, rf_got = 0, D

        # slice the desired rows only using the difference between indices
        di, df = ri - ri_got, rf - rf_got
        if di or df:
            # we can't slice 'coo' matrices -> convert to 'csr'
            if sp.isspmatrix_coo(X):
                X = X.tocsr()
            X = X[di:(None if df == 0 else df), :]

    if stype is not None:
        return X.asformat(stype)
    if coo_build or (issparse(X) and X.format == "coo"):
        return X.asformat("csr")

    return X


def kronpow(a, p, **kron_opts):
    """Returns `a` tensored with itself `p` times

    Equivalent to ``reduce(lambda x, y: x & y, [a] * p)``.

    Parameters
    ----------
    a : dense or sparse vector or operator
        Object to tensor power.
    p : int
        Tensor power.
    kron_opts :
        Supplied to :func:`~quimb.kron`.

    Returns
    -------
    dense or sparse vector or operator
    """
    ops = (a,) * p
    return kron(*ops, **kron_opts)


def _find_shape_of_nested_int_array(x):
    """Take a n-nested list/tuple of integers and find its array shape.
    """
    shape = [len(x)]
    sub_x = x[0]
    while not np.issubdtype(type(sub_x), np.integer):
        shape.append(len(sub_x))
        sub_x = sub_x[0]
    return tuple(shape)


def _dim_map_1d(sza, coos):
    for coo in coos:
        if 0 <= coo < sza:
            yield coo
        else:
            raise ValueError("One or more coordinates out of range.")


def _dim_map_1dtrim(sza, coos):
    return (coo for coo in coos if (0 <= coo < sza))


def _dim_map_1dcyclic(sza, coos):
    return (coo % sza for coo in coos)


def _dim_map_2dcyclic(sza, szb, coos):
    return (szb * (coo[0] % sza) + coo[1] % szb for coo in coos)


def _dim_map_2dtrim(sza, szb, coos):
    for coo in coos:
        x, y = coo
        if 0 <= x < sza and 0 <= y < szb:
            yield szb * x + y


def _dim_map_2d(sza, szb, coos):
    for coo in coos:
        x, y = coo
        if 0 <= x < sza and 0 <= y < szb:
            yield szb * x + y
        else:
            raise ValueError("One or more coordinates out of range.")


def _dim_map_nd(szs, coos, cyclic=False, trim=False):
    strides = [1]
    for sz in szs[-1:0:-1]:
        strides.insert(0, sz * strides[0])
    if cyclic:
        coos = ((c % sz for c, sz in zip(coo, szs)) for coo in coos)
    elif trim:
        coos = (c for c in coos if all(x == x % sz for x, sz in zip(c, szs)))
    elif not all(all(c == c % sz for c, sz in zip(coo, szs)) for coo in coos):
        raise ValueError("One or more coordinates out of range.")
    return (sum(c * m for c, m in zip(coo, strides)) for coo in coos)


_dim_mapper_methods = {(1, False, False): _dim_map_1d,
                       (1, False, True): _dim_map_1dtrim,
                       (1, True, False): _dim_map_1dcyclic,
                       (2, False, False): _dim_map_2d,
                       (2, False, True): _dim_map_2dtrim,
                       (2, True, False): _dim_map_2dcyclic}


def dim_map(dims, coos, cyclic=False, trim=False):
    """Flatten 2d+ dimensions and coordinates.

    Maps multi-dimensional coordinates and indices to flat arrays in a
    regular way. Wraps or deletes coordinates beyond the system size
    depending on parameters ``cyclic`` and ``trim``.

    Parameters
    ----------
    dims : nested tuple of int
        Multi-dim array of systems' internal dimensions.
    coos : list of tuples of int
        Array of coordinate tuples to convert
    cyclic : bool, optional
        Whether to automatically wrap coordinates beyond system size or
        delete them.
    trim : bool, optional
        If True, any coordinates beyond dimensions will be deleted,
        overidden by cyclic.

    Returns
    -------
    flat_dims : tuple
        Flattened version of ``dims``.
    inds : tuple
        Indices corresponding to the original coordinates.

    Examples
    --------

    >>> dims = [[2, 3], [4, 5]]
    >>> coords = [(0, 0), (1, 1)]
    >>> flat_dims, inds = dim_map(dims, coords)
    >>> flat_dims
    (2, 3, 4, 5)
    >>> inds
    (0, 3)

    >>> dim_map(dims, [(2, 0), (-1, 1)], cyclic=True)
    ((2, 3, 4, 5), (0, 3))
    """
    # Figure out shape of dimensions given
    if isinstance(dims, np.ndarray):
        szs = dims.shape
        ndim = dims.ndim
    else:
        szs = _find_shape_of_nested_int_array(dims)
        ndim = len(szs)

    # Ensure `coos` in right format for 1d (i.e. not single tuples)
    if ndim == 1:
        if isinstance(coos, np.ndarray):
            coos = coos.ravel()
        elif not isinstance(coos[0], Integral):
            coos = (c[0] for c in coos)

    # Map coordinates to indices
    try:
        inds = _dim_mapper_methods[(ndim, cyclic, trim)](*szs, coos)
    except KeyError:
        inds = _dim_map_nd(szs, coos, cyclic, trim)

    # Ravel dims
    while ndim > 1:
        dims = itertools.chain.from_iterable(dims)
        ndim -= 1

    return tuple(dims), tuple(inds)


# numba decorator can't cache generator
@njit_nocache  # pragma: no cover
def _dim_compressor(dims, inds):  # pragma: no cover
    """Helper function for ``dim_compress`` that does the heavy lifting.

    Parameters
    ----------
    dims : sequence of int
        The subsystem dimensions.
    inds : sequence of int
        The indices of the 'marked' subsystems.

    Returns
    -------
    generator of (int, int)
        Sequence of pairs of new dimension subsystem with marked flag {0, 1}.
    """
    blocksize_id = blocksize_op = 1
    autoplace_count = 0
    for i, dim in enumerate(dims):
        if dim < 0:
            if blocksize_op > 1:
                yield (blocksize_op, 1)
                blocksize_op = 1
            elif blocksize_id > 1:
                yield (blocksize_id, 0)
                blocksize_id = 1
            autoplace_count += dim
        elif i in inds:
            if blocksize_id > 1:
                yield (blocksize_id, 0)
                blocksize_id = 1
            elif autoplace_count < 0:
                yield (autoplace_count, 1)
                autoplace_count = 0
            blocksize_op *= dim
        else:
            if blocksize_op > 1:
                yield (blocksize_op, 1)
                blocksize_op = 1
            elif autoplace_count < 0:
                yield (autoplace_count, 1)
                autoplace_count = 0
            blocksize_id *= dim
    yield ((blocksize_op, 1) if blocksize_op > 1 else
           (blocksize_id, 0) if blocksize_id > 1 else
           (autoplace_count, 1))


def dim_compress(dims, inds):
    """Compress neighbouring subsytem dimensions.

    Take some dimensions and target indices and compress both, i.e.
    merge adjacent dimensions that are both either in ``dims`` or not. For
    example, if tensoring an operator onto a single site, with many sites
    the identity, treat these as single large identities.

    Parameters
    ----------
    dims : tuple of int
        List of system's dimensions - 1d or flattened (e.g. with
        ``dim_map``).
    inds: tuple of int
        List of target indices, i.e. dimensions not to merge.

    Returns
    -------
    dims : tuple of int
        New compressed dimensions.
    inds : tuple of int
        New indexes corresponding to the compressed dimensions. These are
        guaranteed to now be alternating i.e. either (0, 2, ...) or
        (1, 3, ...).

    Examples
    --------
    >>> dims = [2] * 10
    >>> inds = [3, 4]
    >>> compressed_dims, compressed_inds = dim_compress(dims, inds)
    >>> compressed_dims
    (8, 4, 32)
    >>> compressed_inds
    (1,)
    """
    if isinstance(inds, Integral):
        inds = (inds,)

    dims, inds = zip(*_dim_compressor(dims, inds))
    inds = tuple(i for i, b in enumerate(inds) if b)

    return dims, inds


def ikron(ops, dims, inds, sparse=None, stype=None,
          coo_build=False, parallel=False, ownership=None):
    """Tensor an operator into a larger space by padding with identities.

    Automatically placing a large operator over several dimensions is allowed
    and a list of operators can be given which are then placed cyclically.

    Parameters
    ----------
    op : operator or sequence of operators
        Operator(s) to place into the tensor space. If more than one, these
        are cyclically placed at each of the ``dims`` specified by ``inds``.
    dims : sequence of int or nested sequences of int
        The subsystem dimensions. If treated as an array, should have the same
        number of dimensions as the system.
    inds : tuple of int, or sequence of tuple of int
        Indices, or coordinates, of the dimensions to place operator(s) on.
        Each dimension specified can be smaller than the size of ``op`` (as
        long as it factorizes it).
    sparse : bool, optional
        Whether to construct the new operator in sparse form.
    stype : str, optional
        If sparse, which format to use for the output.
    coo_build : bool, optional
        Whether to build the intermediary matrices using the ``'coo'``
        format - can be faster to build sparse in this way, then
        convert to chosen format, including dense.
    parallel : bool, optional
        Whether to build the operator in parallel using threads (only good
        for big (d > 2**16) operators).
    ownership : (int, int), optional
        If given, only construct the rows in ``range(*ownership)``. Such that
        the  final operator is actually ``X[slice(*ownership), :]``. Useful for
        constructing operators in parallel, e.g. for MPI.

    Returns
    -------
    qarray or sparse matrix
        Operator such that ops act on ``dims[inds]``.

    See Also
    --------
    kron, pkron

    Examples
    --------
    Place an operator between two identities:

    >>> IZI = ikron(pauli('z'), [2, 2, 2], 1)
    >>> np.allclose(IZI, eye(2) & pauli('z') & eye(2))
    True

    Overlay a large operator on several sites:

    >>> rho_ab = rand_rho(4)
    >>> rho_abc = ikron(rho_ab, [5, 2, 2, 7], [1, 2])  # overlay both 2s
    >>> rho_abc.shape
    (140, 140)

    Place an operator at specified sites, regardless of size:

    >>> A = rand_herm(5)
    >>> ikron(A, [2, -1, 2, -1, 2, -1], [1, 3, 5]).shape
    (1000, 1000)
    """
    # TODO: test 2d+ dims and coos
    # TODO: simplify  with compress coords?
    # TODO: allow -1 in dims to auto place *without* ind? one or other

    # Make sure `ops` islist
    if isinstance(ops, (np.ndarray, sp.spmatrix)):
        ops = (ops,)

    # Make sure dimensions and coordinates have been flattenened.
    if np.ndim(dims) > 1:
        dims, inds = dim_map(dims, inds)
    # Make sure `inds` is list
    elif np.ndim(inds) == 0:
        inds = (inds,)

    # Infer sparsity from list of ops
    if sparse is None:
        sparse = any(issparse(op) for op in ops)

    # Create a sorted list of operators with their matching index
    inds, ops = zip(*sorted(zip(inds, itertools.cycle(ops))))
    inds, ops = set(inds), iter(ops)

    # can't slice "coo" format so use "csr" if ownership specified
    eye_kws = {'sparse': sparse, 'stype': "csr" if ownership else "coo"}

    def gen_ops():
        cff_id = 1  # keeps track of compressing adjacent identities
        cff_ov = 1  # keeps track of overlaying op on multiple dimensions
        for ind, dim in enumerate(dims):

            # check if op should be placed here
            if ind in inds:

                # check if need preceding identities
                if cff_id > 1:
                    yield eye(cff_id, **eye_kws)
                    cff_id = 1  # reset cumulative identity size

                # check if first subsystem in placement block
                if cff_ov == 1:
                    op = next(ops)
                    sz_op = op.shape[0]

                # final dim (of block or total) -> place op
                if cff_ov * dim == sz_op or dim == -1:
                    yield op
                    cff_ov = 1
                # accumulate sub-dims
                else:
                    cff_ov *= dim

            # check if midway through placing operator over several subsystems
            elif cff_ov > 1:
                cff_ov *= dim

            # else accumulate adjacent identites
            else:
                cff_id *= dim

        # check if trailing identity needed
        if cff_id > 1:
            yield eye(cff_id, **eye_kws)

    return kron(*gen_ops(), stype=stype, coo_build=coo_build,
                parallel=parallel, ownership=ownership)


@ensure_qarray
def _permute_dense(p, dims, perm):
    """Permute the subsytems of a dense array.
    """
    p, perm = np.asarray(p), np.asarray(perm)
    d = prod(dims)

    if isop(p):
        return (p.reshape([*dims, *dims])
                .transpose([*perm, *(perm + len(dims))])
                .reshape([d, d]))

    return (p.reshape(dims)
            .transpose(perm)
            .reshape([d, 1]))


def _permute_sparse(a, dims, perm):
    """Permute the subsytems of a sparse matrix.
    """
    perm, dims = np.asarray(perm), np.asarray(dims)

    # New dimensions & stride (i.e. product of preceding dimensions)
    new_dims = dims[perm]
    odim_stride = np.multiply.accumulate(dims[::-1])[::-1] // dims
    ndim_stride = np.multiply.accumulate(new_dims[::-1])[::-1] // new_dims

    # Range of possible coordinates for each subsys
    coos = (tuple(range(dim)) for dim in dims)

    # Complete basis using coordinates for current and new dimensions
    basis = np.asarray(tuple(itertools.product(*coos, repeat=1)))
    oinds = np.sum(odim_stride * basis, axis=1)
    ninds = np.sum(ndim_stride * basis[:, perm], axis=1)

    # Construct permutation matrix and apply it to state
    perm_mat = sp.coo_matrix((np.ones(a.shape[0]), (ninds, oinds))).tocsr()
    if isop(a):
        return dot(dot(perm_mat, a), dag(perm_mat))
    return dot(perm_mat, a)


def permute(p, dims, perm):
    """Permute the subsytems of state or opeator.

    Parameters
    ----------
    p : vector or operator
        State or operator to permute.
    dims : tuple of int
        Internal dimensions of the system.
    perm : tuple of int
        New order of indexes ``range(len(dims))``.

    Returns
    -------
    pp : vector or operator
        Permuted state or operator.

    See Also
    --------
    pkron

    Examples
    --------

    >>> IX = speye(2) & pauli('X', sparse=True)
    >>> XI = permute(IX, dims=[2, 2], perm=[1, 0])
    >>> np.allclose(XI.A, pauli('X') & eye(2))
    True
    """
    if issparse(p):
        return _permute_sparse(p, dims, perm)
    return _permute_dense(p, dims, perm)


def pkron(op, dims, inds, **ikron_opts):
    # TODO: multiple ops
    # TODO: coo map, coo compress
    # TODO: sparse, stype, coo_build?
    """Advanced, padded tensor product.

    Construct an operator such that ``op`` acts on ``dims[inds]``, and allow it
    to be arbitrarily split and reversed etc., in other words, permute and then
    tensor it into a larger space.

    Parameters
    ----------
    ops : matrix-like or tuple of matrix-like
        Operator to place into the tensor space.
    dims : tuple of int
        Dimensions of tensor space.
    inds : tuple of int
        Indices of the dimensions to place operators on. If multiple
        operators are specified, ``inds[1]`` corresponds to ``ops[1]`` and
        so on.
    sparse : bool, optional
        Whether to construct the new operator in sparse form.
    stype : str, optional
        If sparse, which format to use for the output.
    coo_build : bool, optional
        Whether to build the intermediary matrices using the ``'coo'``
        format - can be faster to build sparse in this way, then
        convert to chosen format, including dense.

    Returns
    -------
    operator
        Operator such that ops act on ``dims[inds]``.

    See Also
    --------
    ikron, permute

    Examples
    --------

    Here we take an operator that acts on spins 0 and 1 with X and Z, and
    transform it to act on spins 2 and 0 -- i.e. reverse it and sandwich an
    identity between the two sites it acts on.

    >>> XZ = pauli('X') & pauli('Z')
    >>> ZIX = pkron(XZ, dims=[2, 3, 2], inds=[2, 0])
    >>> np.allclose(ZIX, pauli('Z') & eye(3) & pauli('X'))
    True
    """
    dims, inds = np.asarray(dims), np.asarray(inds)

    # total number of subsytems and size
    n = len(dims)
    sz = prod(dims)

    # dimensions of space where op should be placed, and its total size
    dims_in = dims[inds]
    sz_in = prod(dims_in)

    # construct pre-permuted full operator
    b = ikron(op, [sz_in, sz // sz_in], 0, **ikron_opts)

    # inverse of inds
    inds_out, dims_out = zip(
        *((i, x) for i, x in enumerate(dims) if i not in inds))

    # current order and dimensions of system
    p = [*inds, *inds_out]
    dims_cur = (*dims_in, *dims_out)

    # find inverse permutation
    ip = np.empty(n, dtype=np.int)
    ip[p] = np.arange(n)

    return permute(b, dims_cur, ip)


def ind_complement(inds, n):
    """Return the indices below ``n`` not contained in ``inds``.
    """
    return tuple(i for i in range(n) if i not in inds)


def itrace(a, axes=(0, 1)):
    """General tensor trace, i.e. multiple contractions, for a dense array.

    Parameters
    ----------
    a : numpy.ndarray
        Tensor to trace.
    axes : (2,) int or (2,) array of int
        - (2,) int: Perform trace on the two indices listed.
        - (2,) array of int: Trace out first sequence of indices with second
          sequence indices.

    Returns
    -------
    numpy.ndarray
        The tensor remaining after tracing out the specified axes.

    See Also
    --------
    trace, partial_trace

    Examples
    --------
    Trace out a single pair of dimensions:

    >>> a = randn(2, 3, 4, 2, 3, 4)
    >>> itrace(a, axes=(0, 3)).shape
    (3, 4, 3, 4)

    Trace out multiple dimensions:

    >>> itrace(a, axes=([1, 2], [4, 5])).shape
    (2, 2)
    """
    # Single index pair to trace out
    if isinstance(axes[0], Integral):
        return np.trace(a, axis1=axes[0], axis2=axes[1])
    elif len(axes[0]) == 1:
        return np.trace(a, axis1=axes[0][0], axis2=axes[1][0])

    # Multiple index pairs to trace out
    gone = set()
    for axis1, axis2 in zip(*axes):
        # Modify indices to adjust for traced out dimensions
        mod1 = sum(x < axis1 for x in gone)
        mod2 = sum(x < axis2 for x in gone)
        gone |= {axis1, axis2}
        a = np.trace(a, axis1=axis1 - mod1, axis2=axis2 - mod2)
    return a


@ensure_qarray
def _partial_trace_dense(p, dims, keep):
    """Perform partial trace of a dense matrix.
    """
    if isinstance(keep, Integral):
        keep = (keep,)
    if isvec(p):  # p = psi
        p = np.asarray(p).reshape(dims)
        lose = ind_complement(keep, len(dims))
        p = np.tensordot(p, p.conj(), (lose, lose))
        d = int(p.size**0.5)
        return p.reshape((d, d))
    else:
        p = np.asarray(p).reshape((*dims, *dims))
        total_dims = len(dims)
        lose = ind_complement(keep, total_dims)
        lose2 = tuple(ind + total_dims for ind in lose)
        p = itrace(p, (lose, lose2))
    d = int(p.size**0.5)
    return p.reshape((d, d))


def _trace_lose(p, dims, lose):
    """Simple partial trace where the single subsytem at ``lose``
    is traced out.
    """
    p = p if isop(p) else dot(p, dag(p))
    dims = np.asarray(dims)
    e = dims[lose]
    a = prod(dims[:lose])
    b = prod(dims[lose + 1:])
    rhos = zeros(shape=(a * b, a * b), dtype=np.complex128)
    for i in range(a * b):
        for j in range(i, a * b):
            i_i = e * b * (i // b) + (i % b)
            i_f = e * b * (i // b) + (i % b) + (e - 1) * b + 1
            j_i = e * b * (j // b) + (j % b)
            j_f = e * b * (j // b) + (j % b) + (e - 1) * b + 1
            rhos[i, j] = trace(p[i_i:i_f:b, j_i:j_f:b])
            if j != i:
                rhos[j, i] = rhos[i, j].conjugate()
    return rhos


def _trace_keep(p, dims, keep):
    """Simple partial trace where the single subsytem
    at ``keep`` is kept.
    """
    p = p if isop(p) else dot(p, dag(p))
    dims = np.asarray(dims)
    s = dims[keep]
    a = prod(dims[:keep])
    b = prod(dims[keep + 1:])
    rhos = zeros(shape=(s, s), dtype=np.complex128)
    for i in range(s):
        for j in range(i, s):
            for k in range(a):
                i_i = b * i + s * b * k
                i_f = b * i + s * b * k + b
                j_i = b * j + s * b * k
                j_f = b * j + s * b * k + b
                rhos[i, j] += trace(p[i_i:i_f, j_i:j_f])
            if j != i:
                rhos[j, i] = rhos[i, j].conjugate()
    return rhos


def _partial_trace_simple(p, dims, keep):
    """Simple partial trace made up of consecutive single subsystem partial
    traces, augmented by 'compressing' the dimensions each time.
    """
    p = p if isop(p) else dot(p, dag(p))
    dims, keep = dim_compress(dims, keep)
    if len(keep) == 1:
        return _trace_keep(p, dims, *keep)
    lmax = max(enumerate(dims),
               key=lambda ix: (ix[0] not in keep) * ix[1])[0]
    p = _trace_lose(p, dims, lmax)
    dims = (*dims[:lmax], *dims[lmax + 1:])
    keep = {(ind if ind < lmax else ind - 1) for ind in keep}
    return _partial_trace_simple(p, dims, keep)


def partial_trace(p, dims, keep):
    """Partial trace of a dense or sparse state.

    Parameters
    ----------
    p : ket or density operator
        State to perform partial trace on - can be sparse.
    dims : sequence of int or nested sequences of int
        The subsystem dimensions. If treated as an array, should have the same
        number of dimensions as the system.
    keep : int, sequence of int or sequence of tuple[int]
        Index or indices of subsytem(s) to keep. If a sequence of integer
        tuples, each should be a coordinate such that the length matches the
        number of dimensions of the system.

    Returns
    -------
    rho : qarray
        Density operator of subsytem dimensions ``dims[keep]``.

    See Also
    --------
    itrace

    Examples
    --------
    Trace out single subsystem of a ket:

    >>> psi = bell_state('psi-')
    >>> ptr(psi, [2, 2], keep=0)  # expect identity
    qarray([[ 0.5+0.j,  0.0+0.j],
            [ 0.0+0.j,  0.5+0.j]])

    Trace out multiple subsystems of a density operator:

    >>> rho_abc = rand_rho(3 * 4 * 5)
    >>> rho_ab = partial_trace(rho_abc, [3, 4, 5], keep=[0, 1])
    >>> rho_ab.shape
    (12, 12)

    Trace out qutrits from a 2D system:

    >>> psi_abcd = rand_ket(3 ** 4)
    >>> dims = [[3, 3],
    ...         [3, 3]]
    >>> keep = [(0, 0), (1, 1)]
    >>> rho_ac = partial_trace(psi_abcd, dims, keep)
    >>> rho_ac.shape
    (9, 9)
    """
    # map 2D+ systems into flat hilbert space
    try:
        ndim = dims.ndim
    except AttributeError:
        ndim = len(_find_shape_of_nested_int_array(dims))

    if ndim >= 2:
        dims, keep = dim_map(dims, keep)

    if issparse(p):
        return _partial_trace_simple(p, dims, keep)

    return _partial_trace_dense(p, dims, keep)


# --------------------------------------------------------------------------- #
# MONKEY-PATCHES                                                              #
# --------------------------------------------------------------------------- #


nmlz = normalize
"""Alias for :func:`normalize`."""

tr = trace
"""Alias for :func:`trace`."""

ptr = partial_trace
"""Alias for :func:`partial_trace`."""

sp.csr_matrix.nmlz = nmlz

sp.csr_matrix.tr = _trace_sparse
sp.csc_matrix.tr = _trace_sparse
sp.coo_matrix.tr = _trace_sparse
sp.bsr_matrix.tr = _trace_sparse

sp.csr_matrix.ptr = _partial_trace_simple
sp.csc_matrix.ptr = _partial_trace_simple
sp.coo_matrix.ptr = _partial_trace_simple
sp.bsr_matrix.ptr = _partial_trace_simple

sp.csr_matrix.__and__ = kron_dispatch
sp.bsr_matrix.__and__ = kron_dispatch
sp.csc_matrix.__and__ = kron_dispatch
sp.coo_matrix.__and__ = kron_dispatch


def csr_mulvec_wrap(fn):
    """Dispatch sparse csr-vector multiplication to parallel method.
    """
    @functools.wraps(fn)
    def csr_mul_vector(A, x):
        if A.nnz > 50000 and _NUM_THREAD_WORKERS > 1:
            return par_dot_csr_matvec(A, x)
        else:
            y = fn(A, x)
            if isinstance(x, qarray):
                y = qarray(y)
            return y

    return csr_mul_vector


def sp_mulvec_wrap(fn):
    """Scipy sparse doesn't call __array_finalize__ so need to explicitly
    make sure qarray input -> qarray output.
    """
    @functools.wraps(fn)
    def qarrayed_fn(self, other):
        out = fn(self, other)
        if isinstance(other, qarray):
            out = qarray(out)
        return out

    return qarrayed_fn


sp.csr_matrix._mul_vector = csr_mulvec_wrap(sp.csr_matrix._mul_vector)
sp.csc_matrix._mul_vector = sp_mulvec_wrap(sp.csc_matrix._mul_vector)
sp.coo_matrix._mul_vector = sp_mulvec_wrap(sp.coo_matrix._mul_vector)
sp.bsr_matrix._mul_vector = sp_mulvec_wrap(sp.bsr_matrix._mul_vector)

sp.csr_matrix._mul_multivector = sp_mulvec_wrap(sp.csr_matrix._mul_multivector)
sp.csc_matrix._mul_multivector = sp_mulvec_wrap(sp.csc_matrix._mul_multivector)
sp.coo_matrix._mul_multivector = sp_mulvec_wrap(sp.coo_matrix._mul_multivector)
sp.bsr_matrix._mul_multivector = sp_mulvec_wrap(sp.bsr_matrix._mul_multivector)
