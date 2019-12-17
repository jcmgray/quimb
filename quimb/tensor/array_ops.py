"""Backend agnostic array operations.
"""
import itertools

import numpy
from autoray import do, reshape, transpose, dag

from ..linalg.base_linalg import norm_fro_dense


def asarray(array):
    if isinstance(array, numpy.matrix) or not hasattr(array, 'shape'):
        return numpy.asarray(array)
    return array


def ndim(array):
    try:
        return array.ndim
    except AttributeError:
        return len(array.shape)


# ------------- miscelleneous other backend agnostic functions -------------- #

def iscomplex(x):
    if not hasattr(x, 'dtype'):
        return isinstance(x, complex)
    return 'complex' in str(x.dtype)


def norm_fro(x):
    if isinstance(x, numpy.ndarray):
        return norm_fro_dense(x.reshape(-1))
    try:
        return do('linalg.norm', reshape(x, [-1]), 2)
    except AttributeError:
        return do('sum', do('multiply', do('conj', x), x)) ** 0.5


def sensibly_scale(x):
    """Take an array and scale it *very* roughly such that random tensor
    networks consisting of such arrays do not have gigantic norms.
    """
    return x / norm_fro(x)**(1.5 / ndim(x))


def _unitize_qr(x):
    """Perform isometrization using the QR decomposition.
    """
    fat = x.shape[0] < x.shape[1]
    if fat:
        x = transpose(x)

    Q = do('linalg.qr', x)[0]
    if fat:
        Q = transpose(Q)

    return Q


def _unitize_svd(x):
    fat = x.shape[0] < x.shape[1]
    if fat:
        x = transpose(x)

    Q = do('linalg.svd', x)[0]
    if fat:
        Q = transpose(Q)

    return Q


def _unitize_exp(x):
    r"""Perform isometrization using the using anti-symmetric matrix
    exponentiation.

    .. math::

            U_A = \exp{A - A^\dagger}

    If ``x`` is rectangular it is completed with zeros first.
    """
    m, n = x.shape
    d = max(m, n)
    x = do('pad', x, [[0, d - m], [0, d - n]], 'constant', constant_values=0.0)
    expx = do('linalg.expm', x - dag(x))
    return expx[:m, :n]


def _unitize_modified_gram_schmidt(A):
    """Perform isometrization explicitly using the modified Gram Schmidt
    procedure.
    """
    m, n = A.shape

    thin = m > n
    if thin:
        A = do('transpose', A)

    Q = []
    for j in range(0, min(m, n)):

        q = A[j, :]
        for i in range(0, j):
            rij = do('tensordot', do('conj', Q[i]), q, 1)
            q = q - rij * Q[i]

        Q.append(q / do('linalg.norm', q, 2))

    Q = do('stack', Q, axis=0, like=A)

    if thin:
        Q = do('transpose', Q)

    return Q


_UNITIZE_METHODS = {
    'qr': _unitize_qr,
    'svd': _unitize_svd,
    'exp': _unitize_exp,
    'mgs': _unitize_modified_gram_schmidt,
}


def unitize(x, method='qr'):
    """Generate a isometric (or unitary if square) matrix from array ``x``.

    Parameters
    ----------
    x : array
        The matrix to generate the isometry from.
    method : {'qr', 'exp', 'mgs'}, optional
        The method used to generate the isometry. Note ``'qr'`` is the fastest
        and most robust but, for example, some libraries cannot back-propagate
        through it.
    """
    return _UNITIZE_METHODS[method](x)


def find_diag_axes(x, atol=1e-12):
    """Try and find a pair of axes of ``x`` in which it is diagonal.

    Parameters
    ----------
    x : array-like
        The array to search.
    atol : float, optional
        Tolerance with which to compare to zero.

    Returns
    -------
    tuple[int] or None
        The two axes if found else None.

    Examples
    --------

        >>> x = np.array([[[1, 0], [0, 2]],
        ...               [[3, 0], [0, 4]]])
        >>> find_diag_axes(x)
        (1, 2)

    Which means we can reduce ``x`` without loss of information to:

        >>> np.einsum('abb->ab', x)
        array([[1, 2],
               [3, 4]])

    """
    shape = x.shape
    indxrs = do('indices', shape, like=x)

    for i, j in itertools.combinations(range(len(shape)), 2):
        if shape[i] != shape[j]:
            continue
        if do('allclose', x[indxrs[i] != indxrs[j]], 0.0, atol=atol):
            return (i, j)
    return None


def find_antidiag_axes(x, atol=1e-12):
    """Try and find a pair of axes of ``x`` in which it is anti-diagonal.

    Parameters
    ----------
    x : array-like
        The array to search.
    atol : float, optional
        Tolerance with which to compare to zero.

    Returns
    -------
    tuple[int] or None
        The two axes if found else None.

    Examples
    --------

        >>> x = np.array([[[0, 1], [0, 2]],
        ...               [[3, 0], [4, 0]]])
        >>> find_antidiag_axes(x)
        (0, 2)

    Which means we can reduce ``x`` without loss of information to:

        >>> np.einsum('aba->ab', x[::-1, :, :])
        array([[3, 4],
               [1, 2]])

    as long as we flip the order of dimensions on other tensors corresponding
    to the the same index.
    """
    shape = x.shape
    indxrs = do('indices', shape, like=x)

    for i, j in itertools.combinations(range(len(shape)), 2):
        di, dj = shape[i], shape[j]
        if di != dj:
            continue
        if do('allclose', x[indxrs[i] != dj - 1 - indxrs[j]], 0.0, atol=atol):
            return (i, j)
    return None


def find_columns(x, atol=1e-12):
    """Try and find columns of axes which are zero apart from a single index.

    Parameters
    ----------
    x : array-like
        The array to search.
    atol : float, optional
        Tolerance with which to compare to zero.

    Returns
    -------
    tuple[int] or None
        If found, the first integer is which axis, and the second is which
        column of that axis, else None.

    Examples
    --------

        >>> x = np.array([[[0, 1], [0, 2]],
        ...               [[0, 3], [0, 4]]])
        >>> find_columns(x)
        (2, 1)

    Which means we can happily slice ``x`` without loss of information to:

        >>> x[:, :, 1]
        array([[1, 2],
               [3, 4]])

    """
    shape = x.shape
    indxrs = do('indices', shape, like=x)

    for i in range(len(shape)):
        for j in range(shape[i]):
            if do('allclose', x[indxrs[i] != j], 0.0, atol=atol):
                return (i, j)

    return None


class PArray:
    """Simple array-like object that lazily generates the actual array by
    calling a function with a set of parameters.

    Parameters
    ----------
    fn : callable
        The function that generates the tensor data from ``params``.
    params : sequence of numbers
        The initial parameters supplied to the generating function like
        ``fn(params)``.

    See Also
    --------
    PTensor
    """

    def __init__(self, fn, params, shape=None):
        self._fn = fn
        self._params = asarray(params)
        self._shape = shape

    @property
    def fn(self):
        return self._fn

    @property
    def params(self):
        return self._params

    @params.setter
    def params(self, x):
        self._params = asarray(x)

    @property
    def data(self):
        self._data = self._fn(self._params)
        return self._data

    @property
    def shape(self):
        if self._shape is None:
            self._shape = self.data.shape
        return self._shape

    @property
    def ndim(self):
        return len(self.shape)
