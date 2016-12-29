"""Core accelerated numerical functions
"""
# TODO: merge kron, eyepad --> tensor
# TODO: finish idot with rpn

import cmath
import functools
import psutil
import threading
import operator

import numpy as np
import scipy.sparse as sp
from numba import jit, vectorize
from numexpr import evaluate


try:
    import os
    _NUM_THREADS = int(os.environ['OMP_NUM_THREADS'])
except KeyError:
    _NUM_THREADS = psutil.cpu_count()

accel = functools.partial(jit, nopython=True, cache=False)


def prod(xs):
    """Product of an iterable.
    """
    return functools.reduce(operator.mul, xs, 1)


def make_immutable(mat):
    """Make sure matrix cannot be changed, for dense and sparse.
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
    """To decorate functions returning ndarrays.
    """
    @functools.wraps(fn)
    def matrixified_fn(*args, **kwargs):
        return np.asmatrix(fn(*args, **kwargs))
    return matrixified_fn


def realify(fn, imag_tol=1.0e-14):
    """To decorate functions that should return float for small complex.
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
    """To decorate functions that compute close to zero answers.
    """
    @functools.wraps(fn)
    def zeroified_f(*args, **kwargs):
        x = fn(*args, **kwargs)
        return 0.0 if abs(x) < tol else x
    return zeroified_f


# --------------------------------------------------------------------------- #
# Type and shape checks                                                       #
# --------------------------------------------------------------------------- #

def isket(qob):
    """Checks if matrix is in ket form, i.e. a matrix column.
    """
    return qob.shape[0] > 1 and qob.shape[1] == 1  # Column vector check


def isbra(qob):
    """Checks if matrix is in bra form, i.e. a matrix row.
    """
    return qob.shape[0] == 1 and qob.shape[1] > 1  # Row vector check


def isop(qob):
    """Checks if matrix is an operator, i.e. a square matrix.
    """
    m, n = qob.shape
    return m == n and m > 1  # Square matrix check


def isvec(qob):
    """Checks if object is row-vector, column-vector or one-dimensional.
    """
    shp = qob.shape
    return len(shp) == 1 or (len(shp) == 2 and (shp[0] == 1 or shp[1] == 1))


def issparse(qob):
    """Checks if an object is sparse.
    """
    return isinstance(qob, sp.spmatrix)


def isherm(qob):
    """Checks if matrix is hermitian, for sparse or dense.
    """
    return ((qob != qob.H).nnz == 0 if issparse(qob) else
            np.allclose(qob, qob.H))


# --------------------------------------------------------------------------- #
# Core accelerated numeric functions                                          #
# --------------------------------------------------------------------------- #

@matrixify
@accel
def _mul_dense(x, y):  # pragma: no cover
    """Accelerated element-wise multiplication of two matrices
    """
    return x * y


def mul(x, y):
    """Dispatch to element wise multiplication.
    """
    # TODO: add sparse, dense -> sparse w/ broadcasting
    if issparse(x):
        return x.multiply(y)
    elif issparse(y):
        return y.multiply(x)
    return _mul_dense(x, y)


@matrixify
@accel
def _dot_dense(a, b):  # pragma: no cover
    """Accelerated dense dot product of matrices
    """
    return a @ b


@accel(nogil=True)  # pragma: no cover
def _dot_csr_matvec(data, indptr, indices, vec, out, k1, k2):
    """Sparse csr matrix-vector dot-product, only acting on range(k1, k2).
    """
    for i in range(k1, k2):
        ri = indptr[i]
        rf = indptr[i + 1]
        isum = 0.0j
        for j in range(rf - ri):
            ri_j = ri + j
            isum += data[ri_j] * vec[indices[ri_j]]
        out[i] = isum


def _par_dot_csr_matvec(mat, vec, nthreads=_NUM_THREADS):
    """Parallel sparse csr matrix vector dot product.
    """
    sz = mat.shape[0]
    out = np.empty(sz, dtype=vec.dtype)
    sz_chnk = (sz // nthreads) + 1
    slices = tuple((i * sz_chnk, min((i + 1) * sz_chnk, sz))
                   for i in range(nthreads))

    fn = functools.partial(_dot_csr_matvec,
                           mat.data,
                           mat.indptr,
                           mat.indices,
                           vec.A.reshape(-1), out)

    thrds = tuple(threading.Thread(target=fn, args=kslice)
                  for kslice in slices)

    for t in thrds:
        t.start()
    for t in thrds:
        t.join()
    return out.reshape(-1, 1)


def _dot_sparse(a, b):
    """Dot product for sparse matrix, dispatching to parallel v large nnz.
    """
    if (issparse(a) and isvec(b) and a.nnz > 500000):  # pragma: no cover
        return _par_dot_csr_matvec(a, b)
    return a @ b


def dot(a, b):
    """Matrix multiplication, dispatched to dense method.
    """
    if issparse(a) or issparse(b):
        return _dot_sparse(a, b)
    return _dot_dense(a, b)


@realify
@accel
def vdot(a, b):  # pragma: no cover
    """Accelerated 'Hermitian' inner product of two vectors.
    """
    return np.vdot(a.reshape(-1), b.reshape(-1))


@realify
@accel
def rdot(a, b):  # pragma: no cover
    """Real dot product of two dense vectors.
    """
    a, b = a.reshape((1, -1)), b.reshape((-1, 1))
    return (a @ b)[0, 0]


@accel
def _reshape_for_ldmul(vec):  # pragma: no cover
    """Reshape a vector to be broadcast multiplied against a matrix in a way
    that replicates left diagonal matrix multiplication.
    """
    d = vec.size
    return d, vec.reshape(d, 1)


@matrixify
def _l_diag_dot_dense(vec, mat):
    d, vec = _reshape_for_ldmul(vec)
    return evaluate("vec * mat") if d > 500 else _mul_dense(vec, mat)


def _l_diag_dot_sparse(vec, mat):
    return sp.diags(vec) @ mat


def ldmul(vec, mat):
    """Accelerated left diagonal multiplication using numexpr,
    faster than numpy for n > ~ 500.

    Parameters
    ----------
        vec: vector of diagonal matrix, can be array
        mat: matrix

    Returns
    -------
        mat: np.matrix
    """
    if issparse(mat):
        return _l_diag_dot_sparse(vec, mat)
    return _l_diag_dot_dense(vec, mat)


@accel
def _reshape_for_rdmul(vec):  # pragma: no cover
    """Reshape a vector to be broadcast multiplied against a matrix in a way
    that replicates right diagonal matrix multiplication.
    """
    d = vec.size
    return d, vec.reshape(1, d)


@matrixify
def _r_diag_dot_dense(mat, vec):
    d, vec = _reshape_for_rdmul(vec)
    return evaluate("mat * vec") if d > 500 else _mul_dense(mat, vec)


def _r_diag_dot_sparse(mat, vec):
    return mat @ sp.diags(vec)


def rdmul(mat, vec):
    """ Accelerated right diagonal multiplication using numexpr,
    faster than numpy for n > ~ 500.

    Parameters
    ----------
        mat: matrix
        vec: vector of diagonal matrix, can be array

    Returns
    -------
        mat: np.matrix """
    if issparse(mat):
        return _r_diag_dot_sparse(mat, vec)
    return _r_diag_dot_dense(mat, vec)


@accel
def _reshape_for_outer(a, b):  # pragma: no cover
    """Reshape two vectors for an outer product
    """
    d = a.size
    return d, a.reshape(d, 1), b.reshape(1, d)


def outer(a, b):
    """Outer product between two vectors (no conjugation).
    """
    d, a, b = _reshape_for_outer(a, b)
    return _mul_dense(a, b) if d < 500 else np.asmatrix(evaluate('a * b'))


@vectorize(nopython=True)
def explt(l, t):  # pragma: no cover
    """Complex exponenital as used in solution to schrodinger equation.
    """
    return cmath.exp((-1.0j * t) * l)


# --------------------------------------------------------------------------- #
# Kronecker (tensor) product                                                  #
# --------------------------------------------------------------------------- #

@accel
def _reshape_for_kron(a, b):  # pragma: no cover
    m, n = a.shape
    p, q = b.shape
    a = a.reshape((m, 1, n, 1))
    b = b.reshape((1, p, 1, q))
    return a, b, m * p, n * q


@matrixify
@jit(nopython=True)
def _kron_dense(a, b):  # pragma: no cover
    a, b, mp, nq = _reshape_for_kron(a, b)
    return (a * b).reshape((mp, nq))


@matrixify
def _kron_dense_big(a, b):
    a, b, mp, nq = _reshape_for_kron(a, b)
    return evaluate('a * b').reshape((mp, nq))


def _kron_sparse(a, b, stype=None):
    """Sparse tensor product, output format can be specified or will be
    automatically determined.
    """
    if stype is None:
        stype = ("bsr" if isinstance(b, np.ndarray) or b.format == 'bsr' else
                 b.format if isinstance(a, np.ndarray) else
                 "csc" if a.format == "csc" and b.format == "csc" else
                 "csr")

    return sp.kron(a, b, format=stype)


def _kron_dispatch(a, b, stype=None):
    if issparse(a) or issparse(b):
        return _kron_sparse(a, b, stype=stype)
    elif a.size * b.size > 23000:  # pragma: no cover
        return _kron_dense_big(a, b)
    else:
        return _kron_dense(a, b)


def kron(*ops, stype=None, coo_build=False):
    """Tensor product of variable number of arguments.

    Parameters
    ----------
        ops: objects to be tensored together
        stype: desired output format if resultant object is sparse.
        coo_build: whether to force sparse construction to use the 'coo'
            format (only for sparse matrices in the first place.).

    Returns
    -------
        operator

    Notes
    -----
         1. The product is performed as (a * (b * (c * ...)))
    """
    opts = {"stype": "coo" if coo_build or stype == "coo" else None}

    def _inner_kron(ops, _l):
        if _l == 1:
            return ops[0]
        a, b = ops[0], _inner_kron(ops[1:], _l - 1)
        return _kron_dispatch(a, b, **opts)

    x = _inner_kron(ops, len(ops))

    if stype is not None:
        return x.asformat(stype)
    if coo_build or (issparse(x) and x.format == "coo"):
        return x.asformat("csr")
    return x


def kronpow(a, p, stype=None, coo_build=False):
    """Returns `a` tensored with itself `p` times
    """
    return kron(*(a for _ in range(p)), stype=stype, coo_build=coo_build)


# --------------------------------------------------------------------------- #
# MONKEY-PATCHES                                                              #
# --------------------------------------------------------------------------- #

# Unused & symbol to tensor product
np.matrix.__and__ = _kron_dispatch
sp.csr_matrix.__and__ = _kron_dispatch
sp.bsr_matrix.__and__ = _kron_dispatch
sp.csc_matrix.__and__ = _kron_dispatch
sp.coo_matrix.__and__ = _kron_dispatch
