import numpy as np
from numba import jit
from numexpr import evaluate
from .misc import matrixify, realify


@matrixify
@jit(nopython=True)
def mul(x, y):  # pragma: no cover
    """ Accelerated element-wise multiplication of two matrices """
    return x * y


@matrixify
@jit(nopython=True)
def dot(x, y):  # pragma: no cover
    """ Accelerated dot product of matrices  """
    return x @ y


@matrixify
def idot(*args):
    """ Accelerated and intelligent dot product of multiple matrices, runs
    backwards over. """
    n = len(args)
    x = args[0]
    if n == 1:
        return x
    y = idot(*args[1:])
    return dot(x, y)


@realify
@jit(nopython=True)
def vdot(a, b):  # pragma: no cover
    """ Accelerated 'Hermitian' inner product of two vectors. """
    return np.vdot(a.ravel(), b.ravel())


@jit(nopython=True)
def reshape_for_ldmul(vec):  # pragma: no cover
    """ Reshape a vector to be broadcast multiplied against a matrix in a way
    that replicates left diagonal matrix multiplication. """
    d = vec.size
    return d, vec.reshape(d, 1)


@matrixify
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
    d, vec = reshape_for_ldmul(vec)
    return evaluate("vec * mat") if d > 500 else mul(vec, mat)


@jit(nopython=True)
def reshape_for_rdmul(vec):  # pragma: no cover
    """ Reshape a vector to be broadcast multiplied against a matrix in a way
    that replicates right diagonal matrix multiplication. """
    d = vec.size
    return d, vec.reshape(1, d)


@matrixify
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
    d, vec = reshape_for_rdmul(vec)
    return evaluate("mat * vec") if d > 500 else mul(mat, vec)


@jit(nopython=True)
def reshape_for_outer(a, b):  # pragma: no cover
    """ Reshape two vectors for an outer product """
    d = a.size
    return d, a.reshape(d, 1), b.reshape(1, d)


def outer(a, b):
    """ Outer product between two vectors (no conjugation). """
    d, a, b = reshape_for_outer(a, b)
    return mul(a, b) if d < 500 else np.asmatrix(evaluate('a * b'))
