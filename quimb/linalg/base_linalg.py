"""Backend agnostic functions for solving matrices either fully or partially.
"""
import functools
import warnings

import numpy as np
import scipy.linalg as sla
import scipy.sparse.linalg as spla

from ..utils import raise_cant_find_library_function
from ..core import qarray, dag, issparse, isdense, vdot, ldmul
from .numpy_linalg import (
    eig_numpy,
    eigs_numpy,
    svds_numpy,
)

from .scipy_linalg import (
    eigs_scipy,
    eigs_lobpcg,
    svds_scipy,
)
from . import SLEPC4PY_FOUND

if SLEPC4PY_FOUND:
    from .mpi_launcher import (
        eigs_slepc_spawn,
        mfn_multiply_slepc_spawn,
        svds_slepc_spawn,
    )
    from .slepc_linalg import eigs_slepc, svds_slepc, mfn_multiply_slepc
else:  # pragma: no cover
    eigs_slepc = raise_cant_find_library_function("slepc4py")
    eigs_slepc_spawn = raise_cant_find_library_function("slepc4py")
    svds_slepc = raise_cant_find_library_function("slepc4py")
    svds_slepc_spawn = raise_cant_find_library_function("slepc4py")
    mfn_multiply_slepc = raise_cant_find_library_function("slepc4py")
    mfn_multiply_slepc_spawn = raise_cant_find_library_function("slepc4py")


# --------------------------------------------------------------------------- #
#                          Partial eigendecomposition                         #
# --------------------------------------------------------------------------- #

def choose_backend(A, k, int_eps=False, B=None):
    """Pick a backend automatically for partial decompositions.
    """
    # LinOps -> not possible to simply convert to dense or use MPI processes
    A_is_linop = isinstance(A, spla.LinearOperator)
    B_is_linop = isinstance(B, spla.LinearOperator)

    # small array or large part of subspace requested
    small_d_big_k = A.shape[0] ** 2 / k < (10000 if int_eps else 2000)

    if small_d_big_k and not (A_is_linop or B_is_linop):
        return "NUMPY"

    # slepc seems faster for sparse, dense and LinearOperators
    if SLEPC4PY_FOUND and not B_is_linop:

        # only spool up an mpi pool for big sparse matrices though
        if issparse(A) and A.nnz > 10000:
            return 'SLEPC'

        return 'SLEPC-NOMPI'

    return 'SCIPY'


_EIGS_METHODS = {
    'NUMPY': eigs_numpy,
    'SCIPY': eigs_scipy,
    'LOBPCG': eigs_lobpcg,
    'SLEPC': eigs_slepc_spawn,
    'SLEPC-NOMPI': eigs_slepc,
}


def eigensystem_partial(A, k, isherm, *, B=None, which=None, return_vecs=True,
                        sigma=None, ncv=None, tol=None, v0=None, sort=True,
                        backend=None, fallback_to_scipy=False, **backend_opts):
    """Return a few eigenpairs from an operator.

    Parameters
    ----------
    A : sparse, dense or linear operator
        The operator to solve for.
    k : int
        Number of eigenpairs to return.
    isherm : bool
        Whether to use hermitian solve or not.
    B : sparse, dense or linear operator, optional
        If given, the RHS operator defining a generalized eigen problem.
    which : {'SA', 'LA', 'LM', 'SM', 'TR'}
        Where in spectrum to take eigenvalues from (see
        :func:``scipy.sparse.linalg.eigsh``)
    return_vecs : bool, optional
        Whether to return the eigenvectors.
    sigma : float, optional
        Which part of spectrum to target, implies which='TR' if which is None.
    ncv : int, optional
        number of lanczos vectors, can use to optimise speed
    tol : None or float
        Tolerance with which to find eigenvalues.
    v0 : None or 1D-array like
        An initial vector guess to iterate with.
    sort : bool, optional
        Whether to explicitly sort by ascending eigenvalue order.
    backend : {'AUTO', 'NUMPY', 'SCIPY',
               'LOBPCG', 'SLEPC', 'SLEPC-NOMPI'}, optional
        Which solver to use.
    fallback_to_scipy : bool, optional
        If an error occurs and scipy is not being used, try using scipy.
    backend_opts
        Supplied to the backend solver.

    Returns
    -------
    elk : (k,) array
        The ``k`` eigenvalues.
    evk : (d, k) array
        Array with ``k`` eigenvectors as columns if ``return_vecs``.
    """
    settings = {
        'k': k,
        'B': B,
        'which': ("SA" if (which is None) and (sigma is None) else
                  "TR" if (which is None) and (sigma is not None) else
                  which),
        'return_vecs': return_vecs,
        'sigma': sigma,
        'isherm': isherm,
        'ncv': ncv,
        'sort': sort,
        'tol': tol,
        'v0': v0,
    }

    # Choose backend to perform the decompostion
    bkd = 'AUTO' if backend is None else backend.upper()
    if bkd == 'AUTO':
        bkd = choose_backend(A, k, sigma is not None, B=B)

    try:
        return _EIGS_METHODS[bkd](A, **settings, **backend_opts)

    # sometimes e.g. lobpcg fails, worth trying scipy
    except Exception as e:  # pragma: no cover
        if fallback_to_scipy and (bkd != 'SCIPY'):
            warnings.warn(
                "`eigensystem_partial` with backend '{}' failed, trying again "
                "with scipy. Set ``fallback_to_scipy=False`` to avoid this and"
                " see the full error. ".format(bkd))

            return eigs_scipy(A, **settings, **backend_opts)
        else:
            raise e


# --------------------------------------------------------------------------- #
#                        Full eigendecomposition                              #
# --------------------------------------------------------------------------- #

def eigensystem(A, isherm, *, k=-1, sort=True, return_vecs=True, **kwargs):
    """Find all or some eigenpairs of an operator.

    Parameters
    ----------
    A : operator
        The operator to decompose.
    isherm : bool
        Whether the operator is assumed to be hermitian or not.
    k : int, optional
        If negative, find all eigenpairs, else perform partial
        eigendecomposition and find ``k`` pairs. See
        :func:`~quimb.linalg.base_linalg.eigensystem_partial`.
    sort : bool, optional
        Whether to sort the eigenpairs in ascending eigenvalue order.
    kwargs
        Supplied to the backend function.

    Returns
    -------
    el : (k,) array
        Eigenvalues.
    ev : (d, k) array
        Corresponding eigenvectors as columns of array, such that
        ``ev @ diag(el) @ ev.H == A``.
    """
    if k < 0:
        return eig_numpy(A, isherm=isherm, sort=sort,
                         return_vecs=return_vecs, **kwargs)

    return eigensystem_partial(A, k=k, isherm=isherm, sort=sort,
                               return_vecs=return_vecs, **kwargs)


eig = functools.partial(eigensystem, isherm=False, return_vecs=True)
eigh = functools.partial(eigensystem, isherm=True, return_vecs=True)
eigvals = functools.partial(eigensystem, isherm=False, return_vecs=False)
eigvalsh = functools.partial(eigensystem, isherm=True, return_vecs=False)


@functools.wraps(eigensystem)
def eigenvectors(A, isherm, *, sort=True, **kwargs):
    return eigensystem(A, isherm=isherm, sort=sort, **kwargs)[1]


eigvecs = functools.partial(eigenvectors, isherm=False)
eigvecsh = functools.partial(eigenvectors, isherm=True)


def groundstate(ham, **kwargs):
    """Alias for finding lowest eigenvector only.
    """
    return eigvecsh(ham, k=1, which='SA', **kwargs)


def groundenergy(ham, **kwargs):
    """Alias for finding lowest eigenvalue only.
    """
    return eigvalsh(ham, k=1, which='SA', **kwargs)[0]


def bound_spectrum(A, backend='auto', **kwargs):
    """Return the smallest and largest eigenvalue of hermitian operator ``A``.
    """
    el_min = eigvalsh(A, k=1, which='SA', backend=backend, **kwargs)[0]
    el_max = eigvalsh(A, k=1, which='LA', backend=backend, **kwargs)[0]
    return el_min, el_max


def _rel_window_to_abs_window(el_min, el_max, w_0, w_sz=None):
    """Convert min/max eigenvalues and relative window to absolute values.

    Parameters
    ----------
    el_min : float
        Smallest eigenvalue.
    el_max : float
        Largest eigenvalue.
    w_0 : float [0.0 - 1.0]
        Relative window centre.
    w_sz : float, optional
        Relative window width.

    Returns
    -------
    l_0[, l_min, l_max]:
        Absolute value of centre of window, lower and upper intervals if a
        window size is specified.
    """
    el_range = el_max - el_min
    el_w_0 = el_min + w_0 * el_range
    if w_sz is not None:
        el_w_min = el_w_0 - w_sz * el_range / 2
        el_w_max = el_w_0 + w_sz * el_range / 2
        return el_w_0, el_w_min, el_w_max
    return el_w_0


def eigh_window(A, w_0, k, w_sz=None, backend='AUTO',
                return_vecs=True, offset_const=1 / 104729, **kwargs):
    """ Return mid-spectrum eigenpairs from a hermitian operator.

    Parameters
    ----------
    A : (d, d) operator
        Operator to retrieve eigenpairs from.
    w_0 : float [0.0, 1.0]
        Relative window centre to retrieve eigenpairs from.
    k : int
        Target number of eigenpairs to retrieve.
    w_sz : float, optional
        Relative maximum window width within which to keep eigenpairs.
    backend : str, optional
        Which :func:`~quimb.eigh` backend to use.
    return_vecs : bool, optional
        Whether to return eigenvectors as well.
    offset_const : float, optional
        Small fudge factor (relative to window range) to avoid 1 / 0 issues.

    Returns
    -------
    el : (k,) array
        Eigenvalues around w_0.
    ev : (d, k) array
        The eigenvectors, if ``return_vecs=True``.
    """
    w_sz = w_sz if w_sz is not None else 1.1

    if isdense(A) or backend.upper() == 'NUMPY':
        if return_vecs:
            lk, vk = eigh(A.A if issparse(A) else A, **kwargs)
        else:
            lk = eigvalsh(A.A if issparse(A) else A, **kwargs)

        lmin, lmax = lk[0], lk[-1]
        l_w0, l_wmin, l_wmax = _rel_window_to_abs_window(lmin, lmax, w_0, w_sz)

    else:
        lmin, lmax = bound_spectrum(A, backend=backend, **kwargs)
        l_w0, l_wmin, l_wmax = _rel_window_to_abs_window(lmin, lmax, w_0, w_sz)
        l_w0 += (lmax - lmin) * offset_const  # for 1/0 issues

        if return_vecs:
            lk, vk = eigh(A, k=k, sigma=l_w0, backend=backend, **kwargs)
        else:
            lk = eigvalsh(A, k=k, sigma=l_w0, backend=backend, **kwargs)

    # Trim eigenpairs from beyond window
    in_window = (lk > l_wmin) & (lk < l_wmax)

    if return_vecs:
        return lk[in_window], vk[:, in_window]

    return lk[in_window]


def eigvalsh_window(*args, **kwargs):
    """Alias for only finding the eigenvalues in a relative window.
    """
    return eigh_window(*args, return_vecs=False, **kwargs)


def eigvecsh_window(*args, **kwargs):
    """Alias for only finding the eigenvectors in a relative window.
    """
    return eigh_window(*args, return_vecs=True, **kwargs)[1]


# -------------------------------------------------------------------------- #
# Partial singular value decomposition                                       #
# -------------------------------------------------------------------------- #

def svd(A, return_vecs=True):
    """Compute full singular value decomposition of an operator, using numpy.

    Parameters
    ----------
    A : (m, n) array
        The operator.
    return_vecs : bool, optional
        Whether to return the singular vectors.

    Returns
    -------
    U : (m, k) array
        Left singular vectors (if ``return_vecs=True``) as columns.
    s : (k,) array
        Singular values.
    VH : (k, n) array
        Right singular vectors (if ``return_vecs=True``) as rows.
    """
    try:
        return np.linalg.svd(A, full_matrices=False, compute_uv=return_vecs)

    except np.linalg.linalg.LinAlgError:  # pragma: no cover
        warnings.warn("Numpy SVD failed, trying again with different driver.")
        return sla.svd(A, full_matrices=False, compute_uv=return_vecs,
                       lapack_driver='gesvd')


_SVDS_METHODS = {
    'SLEPC': svds_slepc_spawn,
    'SLEPC-NOMPI': svds_slepc,
    'NUMPY': svds_numpy,
    'SCIPY': svds_scipy,
}


def svds(A, k, ncv=None, return_vecs=True, backend='AUTO', **kwargs):
    """Compute the partial singular value decomposition of an operator.

    Parameters
    ----------
    A : dense, sparse or linear operator
        The operator to decompose.
    k : int, optional
        number of singular value (triplets) to retrieve
    ncv : int, optional
        Number of lanczos vectors to use performing decomposition.
    return_vecs : bool, optional
        Whether to return the left and right vectors
    backend : {'AUTO', 'SCIPY', 'SLEPC', 'SLEPC-NOMPI', 'NUMPY'}, optional
        Which solver to use to perform decomposition.

    Returns
    -------
    (Uk,) sk (, VHk) :
        Singular value(s) (and vectors) such that ``Uk @ np.diag(sk) @ VHk``
        approximates ``A``.
    """
    settings = {
        'k': k,
        'ncv': ncv,
        'return_vecs': return_vecs}

    bkd = (choose_backend(A, k, False) if backend in {'auto', 'AUTO'} else
           backend.upper())
    svds_func = _SVDS_METHODS[bkd.upper()]

    return svds_func(A, **settings, **kwargs)


# -------------------------------------------------------------------------- #
# Norms and other quantities based on decompositions                         #
# -------------------------------------------------------------------------- #

def norm_2(A, **kwargs):
    """Return the 2-norm of operator, ``A``, i.e. the largest singular value.
    """
    return svds(A, k=1, return_vecs=False, **kwargs)[0]


def norm_fro_dense(A):
    """Frobenius norm for dense matrices
    """
    return vdot(A, A).real**0.5


def norm_fro_sparse(A):
    return vdot(A.data, A.data).real**0.5


def norm_trace_dense(A, isherm=True):
    """Returns the trace norm of operator ``A``, that is,
    the sum of the absolute eigenvalues.
    """
    return abs(eigensystem(A, return_vecs=False, isherm=isherm)).sum()


def norm(A, ntype=2, **kwargs):
    """Operator norms.

    Parameters
    ----------
    A : operator
        The operator to find norm of.
    ntype : str
        Norm to calculate, if any of:

        - {2, '2', 'spectral'}: largest singular value
        - {'f', 'fro'}: frobenius norm
        - {'t', 'nuc', 'tr', 'trace'}: sum of singular values

    Returns
    -------
    x : float
        The operator norm.
    """
    types = {'2': '2', 2: '2', 'spectral': '2',
             'f': 'f', 'fro': 'f',
             't': 't', 'trace': 't', 'nuc': 't', 'tr': 't'}
    methods = {('2', 0): norm_2,
               ('2', 1): norm_2,
               ('t', 0): norm_trace_dense,
               ('f', 0): norm_fro_dense,
               ('f', 1): norm_fro_sparse}
    return methods[(types[ntype], issparse(A))](A, **kwargs)


# --------------------------------------------------------------------------- #
#                               Matrix functions                              #
# --------------------------------------------------------------------------- #

def expm(A, herm=False):
    """Matrix exponential, can be accelerated if explicitly hermitian.

    Parameters
    ----------
    A : dense or sparse operator
        Operator to exponentiate.
    herm : bool, optional
        If True (not default), and ``A`` is dense, digonalize the matrix
        in order to perform the exponential.
    """
    if issparse(A):
        # convert to and from csc to suppress scipy warning
        return spla.expm(A.tocsc()).tocsr()
    elif not herm:
        return qarray(spla.expm(A))
    else:
        evals, evecs = eigh(A)
        return evecs @ ldmul(np.exp(evals), dag(evecs))


_EXPM_MULTIPLY_METHODS = {
    'SCIPY': spla.expm_multiply,
    'SLEPC': functools.partial(mfn_multiply_slepc_spawn, fntype='exp'),
    'SLEPC-KRYLOV': functools.partial(
        mfn_multiply_slepc_spawn, fntype='exp', MFNType='KRYLOV'),
    'SLEPC-EXPOKIT': functools.partial(
        mfn_multiply_slepc_spawn, fntype='exp', MFNType='EXPOKIT'),
    'SLEPC-NOMPI': functools.partial(mfn_multiply_slepc, fntype='exp'),
}


def expm_multiply(mat, vec, backend="AUTO", **kwargs):
    """Compute the action of ``expm(mat)`` on ``vec``.

    Parameters
    ----------
    mat : operator
        Operator with which to act with exponential on ``vec``.
    vec : vector-like
        Vector to act with exponential of operator on.
    backend : {'AUTO', 'SCIPY', 'SLEPC', 'SLEPC-KRYLOV', 'SLEPC-EXPOKIT'}
        Which backend to use.
    kwargs
        Supplied to backend function.

    Returns
    -------
    vector
        Result of ``expm(mat) @ vec``.
    """
    if backend == 'AUTO':
        if SLEPC4PY_FOUND and vec.size > 2**10:
            backend = 'SLEPC'
        else:
            backend = 'SCIPY'

    return _EXPM_MULTIPLY_METHODS[backend.upper()](mat, vec, **kwargs)


def sqrtm(A, herm=True):
    """Matrix square root, can be accelerated if explicitly hermitian.

    Parameters
    ----------
    A : dense array
        Operator to take square root of.
    herm : bool, optional
        If True (the default), and ``A`` is dense, digonalize the matrix
        in order to take the square root.

    Returns
    -------
    array
    """
    if issparse(A):
        raise NotImplementedError("No sparse sqrtm available.")
    elif not herm:
        return qarray(sla.sqrtm(A))
    else:
        evals, evecs = eigh(A)
        return evecs @ ldmul(np.sqrt(evals.astype(complex)), dag(evecs))


class IdentityLinearOperator(spla.LinearOperator):
    """Get a ``LinearOperator`` representation of the identity operator,
    scaled by ``factor``.

    Parameters
    ----------
    size : int
        The size of the identity.
    factor : float
        The coefficient of the identity.

    Examples
    --------

    >>> I3 = IdentityLinearOperator(100, 1/3)
    >>> p = rand_ket(100)
    >>> np.allclose(I3 @ p, p / 3)
    True
    """

    def __init__(self, size, factor=1):
        self.factor = factor
        super().__init__(dtype=np.array(factor).dtype, shape=(size, size))

    def _matvec(self, vec):
        return self.factor * vec

    def _rmatvec(self, vec):
        return self.factor * vec

    def _matmat(self, mat):
        return self.factor * mat


class Lazy:
    """A simple class representing an unconstructed matrix. This can be passed
    to, for example, MPI workers, who can then construct the matrix themselves.
    The main function ``fn`` should ideally take an ``ownership`` keyword to
    avoid forming every row.

    This is essentially like using ``functools.partial`` and assigning the
    ``shape`` attribute.

    Parameters
    ----------
    fn : callable
        A function that constructs an operator.
    shape :
        Shape of the constructed operator.
    args
        Supplied to ``fn``.
    kwargs
        Supplied to ``fn``.

    Returns
    -------
    Lazy : callable

    Examples
    --------
    Setup the lazy operator:

    >>> H_lazy = Lazy(ham_heis, n=10, shape=(2**10, 2**10), sparse=True)
    >>> H_lazy
    <Lazy(ham_heis, shape=(1024, 1024), dtype=None)>

    Build a matrix slice (usually done automatically by e.g. ``eigs``):

    >>> H_lazy(ownership=(256, 512))
    <256x1024 sparse matrix of type '<class 'numpy.float64'>'
            with 1664 stored elements in Compressed Sparse Row format>
    """

    def __init__(self, fn, *args, shape=None, factor=None, **kwargs):
        if shape is None:
            raise TypeError("`shape` must be specified.")
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.shape = shape
        self.factor = factor
        self.dtype = None

    def __imul__(self, x):
        if self.factor is None:
            self.factor = x
        else:
            self.factor = self.factor * x

    def __mul__(self, x):
        if self.factor is not None:
            x = x * self.factor
        return Lazy(self.fn, *self.args, shape=self.shape,
                    factor=x, **self.kwargs)

    def __rmul__(self, x):
        return self.__mul__(x)

    def __call__(self, **kwargs):
        A = self.fn(*self.args, **self.kwargs, **kwargs)

        # check if any prefactors have been set
        if self.factor is not None:
            # try inplace first
            try:
                A *= self.factor
            except (ValueError, TypeError):
                A = self.factor * A

        # helpful to store dtype once constructed
        self.dtype = A.dtype
        return A

    def __repr__(self):
        s = "<Lazy({}, shape={}{}{})>"

        s_dtype = (', dtype={}'.format(self.dtype)
                   if self.dtype is not None else '')
        s_factor = (', factor={}'.format(self.factor)
                    if self.factor is not None else '')

        return s.format(self.fn.__name__, self.shape, s_dtype, s_factor)
