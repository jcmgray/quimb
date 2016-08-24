"""
Core accelerated numerical functions
"""
# TODO: merge kron, eyepad --> tensor
# TODO: finish idot with rpn


import cmath
import functools

import numpy as np
import scipy.sparse as sp
from scipy.sparse import issparse
from numba import jit, vectorize
from numexpr import evaluate


accel = functools.partial(jit, nopython=True, cache=True)


# --------------------------------------------------------------------------- #
# Decorators for standardizing output                                         #
# --------------------------------------------------------------------------- #

def matrixify(f):
    """ To decorate functions returning ndarrays. """
    def matrixified_f(*args, **kwargs):
        return np.asmatrix(f(*args, **kwargs))
    return matrixified_f


def realify(f, imag_tol=1.0e-14):
    """ To decorate functions that should return float for small complex. """
    def realified_f(*args, **kwargs):
        x = f(*args, **kwargs)
        if isinstance(x, complex):
            return x.real if abs(x.imag) < abs(x.real) * imag_tol else x
        return x
    return realified_f


def zeroify(f, tol=1e-14):
    """ To decorate functions that compute close to zero answers. """
    def zeroified_f(*args, **kwargs):
        x = f(*args, **kwargs)
        return 0.0 if abs(x) < tol else x
    return zeroified_f


# --------------------------------------------------------------------------- #
# Type and shape checks                                                       #
# --------------------------------------------------------------------------- #

def isket(qob):
    """ Checks if matrix is in ket form, i.e. a matrix column. """
    return qob.shape[0] > 1 and qob.shape[1] == 1  # Column vector check


def isbra(qob):
    """ Checks if matrix is in bra form, i.e. a matrix row. """
    return qob.shape[0] == 1 and qob.shape[1] > 1  # Row vector check


def isop(qob):
    """ Checks if matrix is an operator, i.e. a square matrix. """
    m, n = qob.shape
    return m == n and m > 1  # Square matrix check


def isherm(qob):
    """ Checks if matrix is hermitian, for sparse or dense. """
    return ((qob != qob.H).nnz == 0 if issparse(qob) else
            np.allclose(qob, qob.H))


# --------------------------------------------------------------------------- #
# Core accelerated numeric functions                                          #
# --------------------------------------------------------------------------- #

@matrixify
@accel
def mul_dense(x, y):  # pragma: no cover
    """ Accelerated element-wise multiplication of two matrices """
    return x * y


def mul(x, y):
    """ Dispatch to element wise multiplication. """
    # TODO: add sparse, dense -> sparse w/ broadcasting
    if issparse(x):
        return x.multiply(y)
    elif issparse(y):
        return y.multiply(x)
    return mul_dense(x, y)


@matrixify
@accel
def dot_dense(a, b):  # pragma: no cover
    """ Accelerated dot_dense product of matrices  """
    return a @ b


def dot_sparse(a, b):
    return a @ b


def dot(a, b):
    """ Matrix multiplication, dispatched to dense method. """
    if issparse(a) or issparse(b):
        return dot_sparse(a, b)
    return dot_dense(a, b)


@realify
@accel
def vdot(a, b):  # pragma: no cover
    """ Accelerated 'Hermitian' inner product of two vectors. """
    return np.vdot(a.reshape(-1), b.reshape(-1))


@realify
@accel
def rdot(a, b):  # pragma: no cover
    """ Real dot product of two dense vectors. """
    a, b = a.reshape((1, -1)), b.reshape((-1, 1))
    return (a @ b)[0, 0]


@accel
def reshape_for_ldmul(vec):  # pragma: no cover
    """ Reshape a vector to be broadcast multiplied against a matrix in a way
    that replicates left diagonal matrix multiplication. """
    d = vec.size
    return d, vec.reshape(d, 1)


@matrixify
def l_diag_dot_dense(vec, mat):
    d, vec = reshape_for_ldmul(vec)
    return evaluate("vec * mat") if d > 500 else mul_dense(vec, mat)


def l_diag_dot_sparse(vec, mat):
    return sp.diags(vec) @ mat


def ldmul(vec, mat):
    """ Accelerated left diagonal multiplication using numexpr,
    faster than numpy for n > ~ 500.

    Parameters
    ----------
        vec: vector of diagonal matrix, can be array
        mat: matrix

    Returns
    -------
        mat: np.matrix """
    if issparse(mat):
        return l_diag_dot_sparse(vec, mat)
    return l_diag_dot_dense(vec, mat)


@accel
def reshape_for_rdmul(vec):  # pragma: no cover
    """ Reshape a vector to be broadcast multiplied against a matrix in a way
    that replicates right diagonal matrix multiplication. """
    d = vec.size
    return d, vec.reshape(1, d)


@matrixify
def r_diag_dot_dense(mat, vec):
    d, vec = reshape_for_rdmul(vec)
    return evaluate("mat * vec") if d > 500 else mul_dense(mat, vec)


def r_diag_dot_sparse(mat, vec):
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
        return r_diag_dot_sparse(mat, vec)
    return r_diag_dot_dense(mat, vec)


@accel
def reshape_for_outer(a, b):  # pragma: no cover
    """ Reshape two vectors for an outer product """
    d = a.size
    return d, a.reshape(d, 1), b.reshape(1, d)


def outer(a, b):
    """ Outer product between two vectors (no conjugation). """
    d, a, b = reshape_for_outer(a, b)
    return mul_dense(a, b) if d < 500 else np.asmatrix(evaluate('a * b'))


@vectorize(nopython=True)
def explt(l, t):  # pragma: no cover
    """ Complex exponenital as used in solution to schrodinger equation. """
    return cmath.exp((-1.0j * t) * l)


# --------------------------------------------------------------------------- #
# Kronecker (tensor) product                                                  #
# --------------------------------------------------------------------------- #

@accel
def reshape_for_kron(a, b):  # pragma: no cover
    m, n = a.shape
    p, q = b.shape
    a = a.reshape((m, 1, n, 1))
    b = b.reshape((1, p, 1, q))
    return a, b, m*p, n*q


@matrixify
@accel
def kron_dense(a, b):  # pragma: no cover
    a, b, mp, nq = reshape_for_kron(a, b)
    return (a * b).reshape((mp, nq))


@matrixify
def kron_dense_big(a, b):
    a, b, mp, nq = reshape_for_kron(a, b)
    return evaluate('a * b').reshape((mp, nq))


def kron_sparse(a, b, stype=None):
    """  Sparse tensor product, output format can be specified or will be
    automatically determined. """
    if stype is None:
        stype = ("bsr" if isinstance(b, np.ndarray) or b.format == 'bsr' else
                 b.format if isinstance(a, np.ndarray) else
                 "csc" if a.format == "csc" and b.format == "csc" else
                 "csr")

    return sp.kron(a, b, format=stype)


def kron_dispatch(a, b, stype=None):
    if issparse(a) or issparse(b):
        return kron_sparse(a, b, stype=stype)
    elif a.size * b.size > 23000:  # pragma: no cover
        return kron_dense_big(a, b)
    else:
        return kron_dense(a, b)


def kron(*ops, stype=None, coo_build=False):
    """
    Tensor product of variable number of arguments.

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
        a, b = ops[0], _inner_kron(ops[1:], _l-1)
        return kron_dispatch(a, b, **opts)

    x = _inner_kron(ops, len(ops))

    if stype is not None:
        return x.asformat(stype)
    if coo_build or (issparse(x) and x.format == "coo"):
        return x.asformat("csr")
    return x


# Monkey-patch unused & symbol to tensor product
np.matrix.__and__ = kron_dispatch
sp.csr_matrix.__and__ = kron_dispatch
sp.bsr_matrix.__and__ = kron_dispatch
sp.csc_matrix.__and__ = kron_dispatch
sp.coo_matrix.__and__ = kron_dispatch


def kronpow(a, p, stype=None, coo_build=False):
    """ Returns `a` tensored with itself `p` times """
    return kron(*(a for _ in range(p)), stype=stype, coo_build=coo_build)
