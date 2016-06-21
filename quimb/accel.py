"""
Core accelerated numerical functions
"""
# TODO: merge kron, eyepad --> tensor
# TODO: finish idot with rpn


from cmath import exp
from functools import partial
import numpy as np
import scipy.sparse as sp
from scipy.sparse import issparse
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


def realify(foo, imag_tol=1.0e-13):
    """ To decorate functions that should return float for small complex. """
    def realified_foo(*args, **kwargs):
        x = foo(*args, **kwargs)
        return x.real if abs(x.imag) < abs(x.real) * imag_tol else x
    return realified_foo

accel = partial(jit, nopython=True, cache=True)


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


def kron_sparse(a, b, format=None):
    """
    Sparse tensor product
    """
    if format is None:
        format = ("bsr" if isinstance(b, np.ndarray) or b.format == 'bsr' else
                  b.format if isinstance(a, np.ndarray) else
                  "csc" if a.format == "csc" and b.format == "csc" else
                  "csr")

    return sp.kron(a, b, format=format)


def kron_dispatch(a, b, format=None):
        if issparse(a) or issparse(b):
            return kron_sparse(a, b, format=format)
        elif a.size * b.size > 23000:  # pragma: no cover
            return kron_dense_big(a, b)
        else:
            return kron_dense(a, b)


def kron(*ops, format=None, coo_construct=False):
    """
    Tensor product of variable number of arguments.

    Parameters
    ----------
        ops: objects to be tensored together
        format: desired output format if resultant object is sparse.
        coo_construct: whether to force sparse construction to use the 'coo'
            format,

    Returns
    -------
        operator

    Notes
    -----
         1. The product is performed as (a * (b * (c * ...)))
    """
    cfrmt = "coo" if coo_construct else None

    def kronner(ops, _l):
        if _l == 1:
            return ops[0]
        elif _l == 2:
            return kron_dispatch(ops[0], ops[1], format=cfrmt)
        else:
            return kron_dispatch(ops[0], kronner(ops[1:], _l-1), format=cfrmt)

    x = kronner(ops, len(ops))
    if issparse(x) and format is not None:
        x = x.asformat(format)
    return x


# Monkey-patch unused & symbol to tensor product
np.matrix.__and__ = kron_dispatch
sp.csr_matrix.__and__ = kron_dispatch
sp.bsr_matrix.__and__ = kron_dispatch
sp.csc_matrix.__and__ = kron_dispatch


def kronpow(a, pwr):
    """ Returns `a` tensored with itself pwr times """
    return kron(*(a for _ in range(pwr)))


@vectorize(nopython=True)
def explt(l, t):  # pragma: no cover
    """ Complex exponenital as used in solution to schrodinger equation. """
    return exp((-1.0j * t) * l)


# --------------------------------------------------------------------------- #
# Intelligent chaining of operations                                          #
# --------------------------------------------------------------------------- #

def calc_dot_type(x):
    """ Assign a label to a object to take part in a dot product. """
    if np.isscalar(x):
        s = "c"
    elif x.ndim == 1:
        s = "l"
    elif isket(x):
        s = "k"
    elif isbra(x):
        s = "b"
    else:
        s = "o"
    return s + "s" if issparse(x) else s


def calc_dot_weight_func_out(s1, s2):
    # columns: weight | function | output-type
    wfo = {
        # vec inner
        "kk": (11, vdot, "c"),
        "bk": (12, rdot, "c"),
        # op @ vec
        "ok": (21, dot_dense, "k"),
        "bo": (22, dot_dense, "b"),
        "osk": (23, dot_sparse, "k"),
        "bos": (24, dot_sparse, "b"),
        "lk": (25, l_diag_dot_dense, "k"),
        "bl": (26, r_diag_dot_dense, "b"),
        # const mult
        "cc": (31, lambda a, b: a * b, "c"),
        "ck": (32, lambda a, b: a * b, "k"),
        "kc": (33, lambda a, b: a * b, "k"),
        "cb": (34, lambda a, b: a * b, "b"),
        "bc": (35, lambda a, b: a * b, "b"),
        "cl": (36, lambda a, b: a * b, "l"),
        "lc": (37, lambda a, b: a * b, "l"),
        "cos": (38, lambda a, b: a * b, "os"),
        "osc": (39, lambda a, b: a * b, "os"),
        "co": (48, mul_dense, "o"),
        "oc": (49, mul_dense, "o"),
        # op @ op
        "los": (41, l_diag_dot_sparse, "os"),
        "osl": (42, r_diag_dot_sparse, "os"),
        "osos": (43, dot_sparse, "os"),
        "lo": (51, l_diag_dot_dense, "o"),
        "ol": (52, r_diag_dot_dense, "o"),
        "oso": (53, dot_sparse, "o"),
        "oos": (54, dot_sparse, "o"),
        "oo": (55, dot_dense, "o"),
        # vec outer
        ("k", "b"): (61, outer, "o"),
    }
    return wfo[s1 + s2]


def calc_wfo_pairs(ss):
    s1 = ss[0]
    for s2 in ss[1:]:
        yield calc_dot_weight_func_out(s1, s2)
        s1 = s2


def calc_dot_tree(ops):
    n = len(ops)
    ss = [calc_dot_type(op) for op in ops]
    wfos = [*calc_wfo_pairs(ss)]
    for _ in range(n-1):
        imin, (_, f, o) = min(enumerate(wfos), key=lambda p: p[1][0])
        yield imin, f
        del ss[imin]
        ss[imin] = o
        del wfos[imin]
        wfos[imin-1:imin+1] = [*calc_wfo_pairs(ss[imin-1:imin+2])]


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
