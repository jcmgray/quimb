from cmath import exp
import numpy as np
import scipy.sparse as sp
from numba import jit, vectorize
from numexpr import evaluate


# --------------------------------------------------------------------------- #
# Decorators for standardizing output                                         #
# --------------------------------------------------------------------------- #

def matrixify(foo):
    """ To decorate functions returning ndarrays. """
    def matrixified_foo(*args, **kwargs):
        return np.asmatrix(foo(*args, **kwargs))
    return matrixified_foo


def realify(foo, imag_tol=1.0e-14):
    """ To decorate functions that should return float for small complex. """
    def realified_foo(*args, **kwargs):
        x = foo(*args, **kwargs)
        return x.real if abs(x.imag) < abs(x.real) * imag_tol else x
    return realified_foo


# --------------------------------------------------------------------------- #
# Type and shape checks                                                       #
# --------------------------------------------------------------------------- #

def issparse(x):
    """ Checks if object is scipy sparse format. """
    return isinstance(x, sp.spmatrix)


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
@jit(nopython=True)
def mul_dense(x, y):  # pragma: no cover
    """ Accelerated element-wise multiplication of two matrices """
    return x * y


def mul(x, y):
    """ Dispatch to element wise multiplication. """
    # TODO: add sparse, dense -> sparse w/ broadcasting
    if issparse(x) or issparse(y):
        return x.multiply(y)
    return mul_dense(x, y)


@matrixify
@jit(nopython=True)
def dot_dense(x, y):  # pragma: no cover
    """ Accelerated dot_dense product of matrices  """
    return x @ y


def dot(x, y):
    """ Matrix multiplication, dispatched to dense method. """
    if issparse(x) or issparse(y):
        return x @ y
    return dot_dense(x, y)


@realify
@jit(nopython=True)
def vdot(a, b):  # pragma: no cover
    """ Accelerated 'Hermitian' inner product of two vectors. """
    return np.vdot(a.ravel(), b.ravel())


@realify
@jit(nopython=True)
def rdot(a, b):  # pragma: no cover
    """ Real dot product of two dense vectors. """
    d = max(a.shape[0], a.shape[1])
    a, b = a.reshape((1, d)), b.reshape((d, 1))
    return (a @ b)[0, 0]


@jit(nopython=True)
def reshape_for_ldmul(vec):  # pragma: no cover
    """ Reshape a vector to be broadcast multiplied against a matrix in a way
    that replicates left diagonal matrix multiplication. """
    d = vec.size
    return d, vec.reshape(d, 1)


@matrixify
def l_diag_dot_dense(vec, mat):
    d, vec = reshape_for_ldmul(vec)
    return evaluate("vec * mat") if d > 500 else mul_dense(vec, mat)


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
        return sp.diags(vec) @ mat
    return l_diag_dot_dense(vec, mat)


@jit(nopython=True)
def reshape_for_rdmul(vec):  # pragma: no cover
    """ Reshape a vector to be broadcast multiplied against a matrix in a way
    that replicates right diagonal matrix multiplication. """
    d = vec.size
    return d, vec.reshape(1, d)


@matrixify
def r_diag_dot_dense(mat, vec):
    d, vec = reshape_for_rdmul(vec)
    return evaluate("mat * vec") if d > 500 else mul_dense(mat, vec)


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
        return mat @ sp.diags(vec)
    return r_diag_dot_dense(mat, vec)


@jit(nopython=True)
def reshape_for_outer(a, b):  # pragma: no cover
    """ Reshape two vectors for an outer product """
    d = a.size
    return d, a.reshape(d, 1), b.reshape(1, d)


def outer(a, b):
    """ Outer product between two vectors (no conjugation). """
    d, a, b = reshape_for_outer(a, b)
    return mul_dense(a, b) if d < 500 else np.asmatrix(evaluate('a * b'))


@vectorize(nopython=True)
def explt(l, t):
    """ Complex exponenital as used in solution to schrodinger equation. """
    return exp((-1.0j * t) * l)


# --------------------------------------------------------------------------- #
# Intelligent chaining of operations                                          #
# --------------------------------------------------------------------------- #

def calc_dot_weight(x, y):
    """ Assign a 'weight' to a particular dot product based on the sparsity
    of the two objects and their shape. Overall, dimension reducing operations
    are favoured and within each sector forward sparse matrix multiplication
    is also favoured due to the CSR-format used. Higher weights imply a higher
    cost of performing that operation early.

    Parameters
    ----------
        x, y: sparse or dense operators, kets, or bras. 1D arrays are assumed
            to be operators in diagonal form.

    Returns
    -------
        weight: integer representing the rough cost of performing dot. """
    weight_map = {
        # bra @ ket
        (0, -1, 1): 11,
        (1, -1, 1): 12,
        (2, -1, 1): 13,
        (3, -1, 1): 14,
        # op @ ket
        (0, 0, 1): 21,
        (1, 0, 1): 22,
        (2, 0, 1): 23,
        (3, 0, 1): 24,
        # ket | op, leaving until vdot at end
        # (0, 1, 0):
        # bra @ op
        (0, -1, 0): 31,
        (1, -1, 0): 32,
        (2, -1, 0): 33,
        (3, -1, 0): 34,
        # op @ op
        (0, 0, 0): 41,
        (1, 0, 0): 42,
        (2, 0, 0): 43,
        (3, 0, 0): 44,
        # ket @ bra
        (0, 1, -1): 51,
        (1, 1, -1): 52,
        (2, 1, -1): 53,
        (3, 1, -1): 54,
    }
    return weight_map[issparse(x) + 2 * issparse(y),
                      0 if x.ndim == 1 else isket(x) - isbra(x),
                      0 if y.ndim == 1 else isket(y) - isbra(y)]


def calc_dot_weights(*args):
    """ Find pairwise `calc_dot_weight` for a list of objects """
    def gen_pair_weights():
        arg1 = args[0]
        for arg2 in args[1:]:
            yield calc_dot_weight(arg1, arg2)
            arg1 = arg2
    return [*gen_pair_weights()]


def calc_dot_func(x, y):
    """ Return the correct function to efficiently compute the dot product
    between x and y. """
    func_map = {
        (1, 1): vdot,  # two kets -> assume inner product wanted
        (-1, 1): rdot,  # already conjugated so use real inner product
        (-1, 0): dot,
        (0, 0): dot,
        (0, 1): dot,
        (-1, 2): rdmul,
        (0, 2): rdmul,
        (2, 0): ldmul,
        (2, 1): ldmul,
        (2, 2): mul_dense,
        (1, -1): outer,
    }
    xkey = 2 if x.ndim == 1 else isket(x) - isbra(x)
    ykey = 2 if y.ndim == 1 else isket(y) - isbra(y)
    return func_map[(xkey, ykey)]


def idot(*args, weights=None):
    # TODO: combine weight and func and output
    #   triple dict, pre-calculate entire tree
    """ Accelerated and intelligent dot product of multiple objects. """
    n = len(args)
    if n == 1:
        return args[0]
    if n == 2:
        dot_func = calc_dot_func(*args)
        return dot_func(*args)
    if weights is None:
        weights = calc_dot_weights(*args)
    # Find best dot to do
    ind, _ = min(enumerate(weights), key=lambda p: p[1])
    args = [*args[:ind], idot(args[ind], args[ind+1]), *args[ind+2:]]
    nweights = [*weights[:ind-1],
                *calc_dot_weights(args[ind-1:ind+2]),
                *weights[ind+2:]]
    return idot(*args, weights=nweights)
