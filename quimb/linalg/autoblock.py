import numpy as np
import numba

from ..core import njit, pnjit, qarray


@njit
def get_nz(A):  # pragma: no cover
    return np.nonzero(A)


@njit(['void(int32, int32, List(Set(int32)))',
       'void(int64, int64, List(Set(int64)))'])
def _add_to_groups(i, j, groups):  # pragma: no cover
    for group in groups:
        if i in group:
            group.add(j)
            return
        if j in group:
            group.add(i)
            return
    # pair is not in a sector yet - create new one
    groups.append({i, j})


@njit(['List(List(int32))(int32[:], int32[:], int_)',
       'List(List(int64))(int64[:], int64[:], int_)'])
def compute_blocks(ix, jx, d):  # pragma: no cover
    """Find the charge sectors (blocks in matrix terms) given element
    coordinates ``ix`` and ``jx`` and total size ``d``.

    Parameters
    ----------
    ix : array of int
        The row coordinates of non-zero elements.
    jx : array of int
        The column coordinates of non-zero elements.
    d : int
        The total size of the operator.

    Returns
    -------
    sectors : list[list[int]]
        The list of charge sectors. Each element is itself a sorted list of the
        basis numbers that make up that sector. The permutation that would
        block diagonalize the operator is then ``np.concatenate(sectors)``.

    Examples
    --------

    >>> H = ham_hubbard_hardcore(4, sparse=True)
    >>> ix, jx = H.nonzero()
    >>> d = H.shape[0]
    >>> sectors = compute_blocks(ix, jx, d)
    >>> sectors
    [[0], [1, 2, 4, 8], [3, 5, 6, 9, 10, 12], [7, 11, 13, 14], [15]]
    """
    groups = [{ix[0], jx[0]}]

    # go through actual nz
    for i, j in zip(ix, jx):
        _add_to_groups(i, j, groups)

    # make sure kernel added as subspace
    for i in range(d):
        _add_to_groups(i, i, groups)

    # sort indices in each group and groups by first element
    return sorted([sorted(g) for g in groups])


@pnjit
def subselect(A, p):  # pragma: no cover
    """Select only the intersection of rows and columns of ``A`` matching the
    basis indices ``p``. Faster than double numpy slicing.

    Parameters
    ----------
    A : 2D-array
        Dense matrix to select from.
    p : sequence of int
        The basis indices to select.

    Returns
    -------
    B : 2D-array
        The matrix, of size ``(len(p), len(p))``.

    Examples
    --------
    >>> A = np.arange(25).reshape(5, 5)
    >>> A
    array([[ 0,  1,  2,  3,  4],
           [ 5,  6,  7,  8,  9],
           [10, 11, 12, 13, 14],
           [15, 16, 17, 18, 19],
           [20, 21, 22, 23, 24]])

    >>> subselect(A, [1, 3])
    array([[ 6,  8],
           [16, 18]])
    """
    dp = len(p)
    out = np.empty((dp, dp), dtype=A.dtype)

    for i in numba.prange(dp):
        for j in numba.prange(dp):
            out[i, j] = A[p[i], p[j]]

    return out


@pnjit
def subselect_set(A, B, p):  # pragma: no cover
    """Set only the intersection of rows and colums of ``A`` matching the
    basis indices ``p`` to ``B``.

    Parameters
    ----------
    A : array with shape (d, d)
        The matrix to set elements in.
    B : array with shape (dp, dp)
        The matrix to set elements from.
    p : sequence of size dp
        The basis indices.

    Examples
    --------
    >>> A = np.zeros((5, 5))
    >>> B = np.random.randn(3, 3)
    >>> p = [0, 2, 4]
    >>> subselect_set(A, B, p)
    array([[-0.31888218,  0.        ,  0.39293245,  0.        ,  0.21822712],
           [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ],
           [ 0.66674486,  0.        ,  1.03388035,  0.        ,  1.7319345 ],
           [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ],
           [-0.94542733,  0.        , -0.37211882,  0.        ,  0.51951555]])
    """
    dp = len(p)

    for i in numba.prange(dp):
        for j in numba.prange(dp):
            A[p[i], p[j]] = B[i, j]


# XXX: want to cache this eventaully -> need parallel+cache numba support?
@njit
def _eigh_autoblocked(A, sort=True):  # pragma: no cover
    d = A.shape[0]

    # allocate output arrays
    el = np.empty(d)
    ev = np.zeros_like(A)

    # find non-zero elements and group into charge sectors
    ix, jx = get_nz(A)
    gs = compute_blocks(ix, jx, d)
    gs = [np.array(g) for g in gs]

    # diagonalize each charge sector seperately
    for i, g in enumerate(gs):
        ng = len(g)

        # check if trivial
        if ng == 1:
            el[g[0]] = A[g[0], g[0]].real
            ev[g[0], g[0]] = 1.0
            continue

        # else diagonalize just the block
        sub_el, sub_ev = np.linalg.eigh(subselect(A, g))

        # set the correct eigenpairs in the output
        el[g] = sub_el
        subselect_set(ev, sub_ev, g)

    # sort into ascending eigenvalue order
    if sort:
        so = np.argsort(el)
        el[:] = el[so]
        ev[:, :] = ev[:, so]

    return el, ev


# XXX: want to cache this eventaully -> need parallel+cache numba support?
@njit
def _eigvalsh_autoblocked(A, sort=True):  # pragma: no cover
    # as above but ignore eigenvector for extra speed
    d = A.shape[0]

    el = np.empty(d)

    ix, jx = get_nz(A)
    gs = compute_blocks(ix, jx, d)
    gs = [np.array(g) for g in gs]

    for i, g in enumerate(gs):
        if len(g) == 1:
            el[g[0]] = A[g[0], g[0]]
            continue

        el[g] = np.linalg.eigvalsh(subselect(A, g))

    if sort:
        el[:] = np.sort(el)

    return el


def eigensystem_autoblocked(A, sort=True, return_vecs=True, isherm=True):
    """Perform Hermitian eigen-decomposition, automatically identifying and
    exploiting symmetries appearing in the current basis as block diagonals
    formed via permutation of rows and columns. The whole process is
    accelerated using ``numba``.

    Parameters
    ----------
    A : array_like
        The operator to eigen-decompose.
    sort : bool, optional
        Whether to sort into ascending order, default True.
    isherm : bool, optional
        Whether ``A`` is hermitian, default True.
    return_vecs : bool, optional
        Whether to return the eigenvectors, default True.

    Returns
    -------
    evals : 1D-array
        The eigenvalues.
    evecs : qarray
        If ``return_vecs=True``, the eigenvectors.
    """
    if not isherm:
        err_msg = "Non-hermitian autoblocking not implemented yet."
        raise NotImplementedError(err_msg)

    if not return_vecs:
        return _eigvalsh_autoblocked(A, sort=sort)

    el, ev = _eigh_autoblocked(A, sort=sort)
    return el, qarray(ev)
