"""Backend agnostic array operations.
"""
import itertools

import numpy
from autoray import do, reshape, transpose, dag, infer_backend, get_dtype_name

from ..core import njit, qarray
from ..utils import compose
from ..linalg.base_linalg import norm_fro_dense


def asarray(array):
    """Maybe convert data for a tensor to use. If ``array`` already has a
    ``.shape`` attribute, i.e. looks like an array, it is left as-is. Else the
    elements are inspected to see which libraries' array constructor should be
    used, defaulting to ``numpy`` if everything is builtin or numpy numbers.
    """
    if isinstance(array, (numpy.matrix, qarray)):
        # if numpy make sure array not subclass
        return numpy.asarray(array)

    if hasattr(array, 'shape'):
        # otherwise don't touch things which are already array like
        return array

    # else we some kind of possibly nested python iterable -> inspect items
    backends = set()

    def _nd_py_iter(x):
        if isinstance(x, str):
            # handle recursion error
            return x

        backend = infer_backend(x)
        if backend != 'builtins':
            # don't iterate any non-builtin containers
            backends.add(backend)
            return x

        # is some kind of python container or element -> iterate or return
        try:
            return tuple(_nd_py_iter(sub) for sub in x)
        except TypeError:
            return x

    nested_tup = _nd_py_iter(array)

    # numpy and builtin elements treat as basic
    backends -= {"builtins", "numpy"}
    if not backends:
        backend = "numpy"
    else:
        backend, = backends

    return do('array', nested_tup, like=backend)


def ndim(array):
    try:
        return array.ndim
    except AttributeError:
        return len(array.shape)


# ------------- miscelleneous other backend agnostic functions -------------- #

def iscomplex(x):
    if infer_backend(x) == 'builtins':
        return isinstance(x, complex)
    return 'complex' in get_dtype_name(x)


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

    Q = do('stack', tuple(Q), axis=0, like=A)

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


@njit
def _numba_find_diag_axes(x, atol=1e-12):  # pragma: no cover
    """Numba-compiled array diagonal axis finder.

    Parameters
    ----------
    x : numpy.ndarray
        The array to search for diagonal axes.
    atol : float
        The tolerance with which to compare to zero.

    Returns
    -------
    diag_axes : set[tuple[int]]
        The set of pairs of axes which are diagonal.
    """

    # create the set of pairs of matching size axes
    diag_axes = set()
    for d1 in range(x.ndim - 1):
        for d2 in range(d1 + 1, x.ndim):
            if x.shape[d1] == x.shape[d2]:
                diag_axes.add((d1, d2))

    # enumerate through every array entry, eagerly invalidating axis pairs
    for index, val in numpy.ndenumerate(x):
        for d1, d2 in list(diag_axes):
            if (index[d1] != index[d2]) and (abs(val) > atol):
                diag_axes.remove((d1, d2))

        # all pairs invalid, nothing left to do
        if len(diag_axes) == 0:
            break

    return diag_axes


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
    if len(shape) < 2:
        return None

    backend = infer_backend(x)

    # use numba-accelerated version for numpy arrays
    if backend == 'numpy':
        diag_axes = _numba_find_diag_axes(x, atol=atol)
        if diag_axes:
            # make it determinstic
            return min(diag_axes)
        return None
    indxrs = do('indices', shape, like=backend)

    for i, j in itertools.combinations(range(len(shape)), 2):
        if shape[i] != shape[j]:
            continue
        if do('allclose', x[indxrs[i] != indxrs[j]], 0.0,
              atol=atol, like=backend):
            return (i, j)
    return None


@njit
def _numba_find_antidiag_axes(x, atol=1e-12):  # pragma: no cover
    """Numba-compiled array antidiagonal axis finder.

    Parameters
    ----------
    x : numpy.ndarray
        The array to search for anti-diagonal axes.
    atol : float
        The tolerance with which to compare to zero.

    Returns
    -------
    antidiag_axes : set[tuple[int]]
        The set of pairs of axes which are anti-diagonal.
    """

    # create the set of pairs of matching size axes
    antidiag_axes = set()
    for i in range(x.ndim - 1):
        for j in range(i + 1, x.ndim):
            if x.shape[i] == x.shape[j]:
                antidiag_axes.add((i, j))

    # enumerate through every array entry, eagerly invalidating axis pairs
    for index, val in numpy.ndenumerate(x):
        for i, j in list(antidiag_axes):
            d = x.shape[i]
            if (index[i] != d - 1 - index[j]) and (abs(val) > atol):
                antidiag_axes.remove((i, j))

        # all pairs invalid, nothing left to do
        if len(antidiag_axes) == 0:
            break

    return antidiag_axes


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
    if len(shape) < 2:
        return None

    backend = infer_backend(x)

    # use numba-accelerated version for numpy arrays
    if backend == 'numpy':
        antidiag_axes = _numba_find_antidiag_axes(x, atol=atol)
        if antidiag_axes:
            # make it determinstic
            return min(antidiag_axes)
        return None

    indxrs = do('indices', shape, like=backend)

    for i, j in itertools.combinations(range(len(shape)), 2):
        di, dj = shape[i], shape[j]
        if di != dj:
            continue
        if do('allclose', x[indxrs[i] != dj - 1 - indxrs[j]], 0.0,
              atol=atol, like=backend):
            return (i, j)
    return None


@njit
def _numba_find_columns(x, atol=1e-12):  # pragma: no cover
    """Numba-compiled single non-zero column axis finder.

    Parameters
    ----------
    x :  array
        The array to search.
    atol : float, optional
        Absolute tolerance to compare to zero with.

    Returns
    -------
    set[tuple[int]]
        Set of pairs (axis, index) defining lone non-zero columns.
    """

    # possible pairings of axis + index
    column_pairs = set()
    for ax, d in enumerate(x.shape):
        for i in range(d):
            column_pairs.add((ax, i))

    # enumerate over all array entries, invalidating potential column pairs
    for index, val in numpy.ndenumerate(x):
        if abs(val) > atol:
            for ax, i in enumerate(index):
                for pax, pi in list(column_pairs):
                    if ax == pax and pi != i:
                        column_pairs.remove((pax, pi))

        # all potential pairs invalidated
        if not len(column_pairs):
            break

    return column_pairs


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
    if len(shape) < 1:
        return None

    backend = infer_backend(x)

    # use numba-accelerated version for numpy arrays
    if backend == 'numpy':
        columns_pairs = _numba_find_columns(x, atol)
        if columns_pairs:
            return min(columns_pairs)
        return None

    indxrs = do('indices', shape, like=backend)

    for i in range(len(shape)):
        for j in range(shape[i]):
            if do('allclose', x[indxrs[i] != j], 0.0, atol=atol, like=backend):
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
        self.fn = fn
        self.params = params
        self._shape = shape
        self._shape_fn_id = id(fn)

    def copy(self):
        new = PArray(self.fn, self.params, self.shape)
        new._data = self._data  # for efficiency
        return new

    @property
    def fn(self):
        return self._fn

    @fn.setter
    def fn(self, x):
        self._fn = x
        self._data = None

    @property
    def params(self):
        return self._params

    @params.setter
    def params(self, x):
        self._params = asarray(x)
        self._data = None

    @property
    def data(self):
        if self._data is None:
            self._data = self._fn(self._params)
        return self._data

    @property
    def shape(self):
        # if we haven't calculated shape or have updated function, get shape
        _shape_fn_id = id(self.fn)
        if (self._shape is None) or (self._shape_fn_id != _shape_fn_id):
            self._shape = self.data.shape
            self._shape_fn_id = _shape_fn_id
        return self._shape

    @property
    def ndim(self):
        return len(self.shape)

    def add_function(self, g):
        """Chain the new function ``g`` on top of current function ``f`` like
        ``g(f(params))``.
        """
        f = self.fn
        self.fn = compose(g, f)
