"""Backend agnostic array operations.
"""
import functools
import itertools

import numpy
from autoray import (
    compose,
    do,
    get_dtype_name,
    get_lib_fn,
    infer_backend,
    reshape,
)

from ..core import njit, qarray
from ..utils import compose as fn_compose
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
            return list(_nd_py_iter(sub) for sub in x)
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


@functools.lru_cache(2**14)
def calc_fuse_perm_and_shape(shape, axes_groups):
    ndim = len(shape)

    # which group does each axis appear in, if any
    num_groups = len(axes_groups)
    ax2group = {ax: g for g, axes in enumerate(axes_groups) for ax in axes}

    # the permutation will be the same for every block: precalculate
    # n.b. all new groups will be inserted at the *first fused axis*
    position = min(g for gax in axes_groups for g in gax)
    axes_before = tuple(
        ax for ax in range(position)
        if ax2group.setdefault(ax, None) is None
    )
    axes_after = tuple(
        ax for ax in range(position, ndim)
        if ax2group.setdefault(ax, None) is None
    )
    perm = (*axes_before, *(ax for g in axes_groups for ax in g), *axes_after)

    # track where each axis will be in the new array
    new_axes = {ax: ax for ax in axes_before}
    for i, g in enumerate(axes_groups):
        for ax in g:
            new_axes[ax] = position + i
    for i, ax in enumerate(axes_after):
        new_axes[ax] = position + num_groups + i
    new_ndim = len(axes_before) + num_groups + len(axes_after)

    new_shape = [1] * new_ndim
    for i, d in enumerate(shape):
        g = ax2group[i]
        new_ax = new_axes[i]
        if g is None:
            # not fusing, new value is just copied
            new_shape[new_ax] = d
        else:
            # fusing: need to accumulate
            new_shape[new_ax] *= d

    if all(i == ax for i, ax in enumerate(perm)):
        # no need to transpose
        perm = None

    new_shape = tuple(new_shape)
    if shape == new_shape:
        # no need to reshape
        new_shape = None

    return perm, new_shape


@compose
def fuse(x, *axes_groups, backend=None):
    """Fuse the give group or groups of axes. The new fused axes will be
    inserted at the minimum index of any fused axis (even if it is not in
    the first group). For example, ``fuse(x, [5, 3], [7, 2, 6])`` will
    produce an array with axes like::

        groups inserted at axis 2, removed beyond that.
                ......<--
        (0, 1, g0, g1, 4, 8, ...)
                |   |
                |   g1=(7, 2, 6)
                g0=(5, 3)

    Parameters
    ----------
    axes_groups : sequence of sequences of int
        The axes to fuse. Each group of axes will be fused into a single
        axis.
    """
    if backend is None:
        backend = infer_backend(x)
    _transpose = get_lib_fn(backend, "transpose")
    _reshape = get_lib_fn(backend, "reshape")

    axes_groups =tuple(map(tuple, axes_groups))
    if not any(axes_groups):
        return x

    shape = tuple(map(int, x.shape))
    perm, new_shape = calc_fuse_perm_and_shape(shape, axes_groups)

    if perm is not None:
        x = _transpose(x, perm)
    if new_shape is not None:
        x = _reshape(x, new_shape)

    return x


def ndim(array):
    """The number of dimensions of an array.
    """
    try:
        return array.ndim
    except AttributeError:
        return len(array.shape)


# ------------- miscelleneous other backend agnostic functions -------------- #

def iscomplex(x):
    """Does ``x`` have a complex dtype?
    """
    if infer_backend(x) == 'builtins':
        return isinstance(x, complex)
    return 'complex' in get_dtype_name(x)


@compose
def norm_fro(x):
    """The frobenius norm of an array.
    """
    try:
        return do("linalg.norm", reshape(x, (-1,)))
    except AttributeError:
        return do("sum", do("abs", x)**2) ** 0.5


norm_fro.register("numpy", norm_fro_dense)


def sensibly_scale(x):
    """Take an array and scale it *very* roughly such that random tensor
    networks consisting of such arrays do not have gigantic norms.
    """
    return x / norm_fro(x)**(1.5 / ndim(x))


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

    __slots__ = ('_fn', '_params', '_data', '_shape', '_shape_fn_id')

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
        self.fn = fn_compose(g, f)
