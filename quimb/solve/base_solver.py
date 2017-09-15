"""Backend agnostic functions for solving matrices either fully or partially.
"""

# TODO: restart eigen and svd - up to tol
# TODO: test non-herm

import functools
import numpy as np
import numpy.linalg as nla
import scipy.linalg as sla
import scipy.sparse.linalg as spla

from ..utils import raise_cant_find_library_function
from ..accel import issparse, vdot, dot_dense, ldmul
from .numpy_solver import (
    eigsys_numpy,
    eigvals_numpy,
    seigsys_numpy,
    numpy_svds,
)
from .scipy_solver import seigsys_scipy, scipy_svds
from . import SLEPC4PY_FOUND

if SLEPC4PY_FOUND:
    from .mpi_spawner import seigsys_slepc_spawn, mfn_multiply_slepc_spawn
    from .slepc_solver import slepc_svds
else:
    seigsys_slepc_spawn = raise_cant_find_library_function("slepc4py")
    slepc_svds = raise_cant_find_library_function("slepc4py")
    mfn_multiply_slepc_spawn = raise_cant_find_library_function("slepc4py")


# --------------------------------------------------------------------------- #
#                        Full eigendecomposition                              #
# --------------------------------------------------------------------------- #

_EIGSYS_METHODS = {
    'NUMPY': eigsys_numpy,
}


def eigsys(a, sort=True, isherm=True, backend='NUMPY', **kwargs):
    """Find all eigenpairs of a dense matrix.

    Parameters
    ----------
        a : matrix-like
            The matrix of decompose.
        sort : bool, optional
            Whether to sort the eigenpairs in ascending eigenvalue order.
        isherm : bool, optional
            Whether the matrix is assumed to be hermitian or not.
        backend : {'numpy'}, optional
            Which backend to use to solve the system.
        **kwargs
            Supplied to the backend function.

    Returns
    -------
        evals : 1d-array
            Eigenvalues.
        evecs : np.matrix
            Corresponding eigenvectors as columns of matrix, such that
            ``evecs @ evals @ evecs.H == a``.
    """
    fn = _EIGSYS_METHODS[backend.upper()]
    return fn(a, sort=sort, isherm=isherm, **kwargs)


_EIGVALS_METHODS = {
    'NUMPY': eigvals_numpy,
}


def eigvals(a, sort=True, isherm=True, backend='numpy', **kwargs):
    """Find all eigenvalues of dense matrix.

    Parameters
    ----------
        a : matrix-like
            The matrix to find eigenvalues of.
        sort : bool, optional
            Whether to sort the eigenvalues in ascending order.
        isherm : bool, optional
            Whether the matrix is assumed to be hermitian or not.
        backend : {'numpy'}, optional
            Which backend to use to solve the system.
        **kwargs
            Supplied to the backend function.

    Returns
    -------
        evals : 1d-array
            Eigenvalues.
    """
    fn = _EIGVALS_METHODS[backend.upper()]
    return fn(a, sort=sort, isherm=isherm, **kwargs)


def eigvecs(a, sort=True, isherm=True, backend='numpy', **kwargs):
    """Find all eigenvectors of a dense matrix.

    Parameters
    ----------
        a : matrix-like
            The matrix of decompose.
        sort : bool, optional
            Whether to sort the eigenpairs in ascending eigenvalue order.
        isherm : bool, optional
            Whether the matrix is assumed to be hermitian or not.
        backend : {'numpy'}, optional
            Which backend to use to solve the system.
        **kwargs
            Supplied to the backend function.

    Returns
    -------
        evecs : np.matrix
            Eigenvectors as columns of matrix.
    """
    return eigsys(a, sort=sort, isherm=isherm, backend=backend, **kwargs)[1]


# --------------------------------------------------------------------------- #
#                          Partial eigendecomposition                         #
# --------------------------------------------------------------------------- #


_SEIGSYS_METHODS = {
    'NUMPY': seigsys_numpy,
    'DENSE': seigsys_numpy,
    'SCIPY': seigsys_scipy,
    'SLEPC': seigsys_slepc_spawn,
}


def _choose_backend(a, k, int_eps=False):
    """Pick a backend automatically for partial decompositions.
    """
    # small matrix or large part of subspace requested
    small_d_big_k = a.shape[0] ** 2 / k < (10000 if int_eps else 2000)
    return ("NUMPY" if small_d_big_k else
            "SLEPC" if SLEPC4PY_FOUND and issparse(a) else
            "SCIPY")


def seigsys(a, k=6, which=None, return_vecs=True, sigma=None,
            isherm=True, ncv=None, sort=True, backend='AUTO', **kwargs):
    """Returns a few eigenpairs from a possibly sparse hermitian operator

    Parameters
    ----------
        a: matrix, probably sparse, hermitian
        k: number of eigenpairs to return
        which: where in spectrum to take eigenvalues from (see scipy eigsh)
        nvc: number of lanczos vectors, can use to optimise speed

    Returns
    -------
        lk: array of eigenvalues
        vk: matrix of eigenvectors as columns
    """
    settings = {
        'k': k,
        'which': ("SA" if which is None and sigma is None else
                  "TR" if which is None and sigma is not None else
                  which),
        'return_vecs': return_vecs,
        'sigma': sigma,
        'isherm': isherm,
        'ncv': ncv,
        'sort': sort}

    # Choose backend to perform the decompostion
    bkd = backend.upper()
    if bkd == 'AUTO':
        bkd = _choose_backend(a, k, sigma is not None)

    return _SEIGSYS_METHODS[bkd](a, **settings, **kwargs)


def seigvals(a, k=6, **kwargs):
    """Seigsys alias for finding eigenvalues only.
    """
    return seigsys(a, k=k, return_vecs=False, **kwargs)


def seigvecs(a, k=6, **kwargs):
    """Seigsys alias for finding eigenvectors only.
    """
    return seigsys(a, k=k, return_vecs=True, **kwargs)[1]


def groundstate(ham, **kwargs):
    """Alias for finding lowest eigenvector only.
    """
    return seigvecs(ham, k=1, which='SA', **kwargs)


def groundenergy(ham, **kwargs):
    """Alias for finding lowest eigenvalue only.
    """
    return seigvals(ham, k=1, which='SA', **kwargs)[0]


def bound_spectrum(a, backend='auto', **kwargs):
    """Return the smallest and largest eigenvalue of operator `a`.
    """
    el_min = seigvals(a, k=1, which='SA', backend=backend, **kwargs)[0]
    el_max = seigvals(a, k=1, which='LA', backend=backend, **kwargs)[0]
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
        w_sz : float (None)
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


def eigsys_window(a, w_0, w_n=6, w_sz=None, backend='AUTO',
                  return_vecs=True, offset_const=1 / 104729, **kwargs):
    """ Return eigenpairs internally from a hermitian matrix.

    Parameters
    ----------
        a : operator
            Operator to retrieve eigenpairs from.
        w_0 : float [0.0 - 1.0]
            Relative window centre to retrieve eigenpairs from.
        w_n : int
            Target number of eigenpairs to retrieve.
        w_sz : float (optional)
            Relative maximum window width within which to keep eigenpairs.
        backend : str
        return_vecs : bool
        offset_const : float

    Returns
    -------
        ls: eigenvalues around w_0
    """
    w_sz = w_sz if w_sz is not None else 1.1

    if not issparse(a) or backend == "dense":
        if return_vecs:
            lk, vk = eigsys(a.A if issparse(a) else a, **kwargs)
        else:
            lk = eigvals(a.A if issparse(a) else a, **kwargs)
        lmin, lmax = lk[0], lk[-1]
        l_w0, l_wmin, l_wmax = _rel_window_to_abs_window(lmin, lmax, w_0, w_sz)

    else:
        lmin, lmax = bound_spectrum(a, **kwargs)
        l_w0, l_wmin, l_wmax = _rel_window_to_abs_window(lmin, lmax, w_0, w_sz)
        l_w0 += (lmax - lmin) * offset_const  # for 1/0 issues
        if return_vecs:
            lk, vk = seigsys(a, k=w_n, sigma=l_w0, backend=backend, **kwargs)
        else:
            lk = seigvals(a, k=w_n, sigma=l_w0, backend=backend, **kwargs)

    # Trim eigenpairs from beyond window
    in_window = (lk > l_wmin) & (lk < l_wmax)
    if return_vecs:
        return lk[in_window], vk[:, in_window]
    return lk[in_window]


def eigvals_window(*args, **kwargs):
    """Alias for only finding the eigenvalues in a relative window.
    """
    return eigsys_window(*args, return_vecs=False, **kwargs)


def eigvecs_window(*args, **kwargs):
    """Alias for only finding the eigenvectors in a relative window.
    """
    return eigsys_window(*args, return_vecs=True, **kwargs)[1]


# -------------------------------------------------------------------------- #
# Partial singular value decomposition                                       #
# -------------------------------------------------------------------------- #

def svd(a, return_vecs=True):
    """Compute full singular value decomposition of matrix.
    """
    return nla.svd(a, full_matrices=False, compute_uv=return_vecs)


def svds(a, k=6, ncv=None, return_vecs=True, backend='AUTO', **kwargs):
    """Compute the partial singular value decomposition of a matrix.

    Parameters
    ----------
        a: operator to partially decompose
        k: number of singular value (triplets) to retrieve
        ncv: number of lanczos vectors to use in decomposition
        return_vecs: whether to return the left and right vectors
        backend: which solver to use to perform decomposition

    Returns
    -------
        (uk,) sk (, vk): singular value(s) (and vectors)
    """
    settings = {
        'k': k,
        'ncv': ncv,
        'return_vecs': return_vecs}
    bkd = (_choose_backend(a, k, False) if backend in {'auto', 'AUTO'} else
           backend.upper())
    svds_func = (slepc_svds if bkd == 'SLEPC' else
                 numpy_svds if bkd in {'NUMPY', 'DENSE'} else
                 scipy_svds if bkd == 'SCIPY' else
                 None)
    return svds_func(a, **settings, **kwargs)


# -------------------------------------------------------------------------- #
# Norms and other quantities based on decompositions                         #
# -------------------------------------------------------------------------- #

def norm_2(a, **kwargs):
    """Return the 2-norm of matrix, a, i.e. the largest singular value.
    """
    return svds(a, k=1, return_vecs=False, **kwargs)[0]


def norm_fro_dense(a):
    """Frobenius norm for dense matrices
    """
    return vdot(a, a).real**0.5


def norm_fro_sparse(a):
    return vdot(a.data, a.data).real**0.5


def norm_trace_dense(a, isherm=True):
    """Returns the trace norm of operator a, that is,
    the sum of abs eigvals.
    """
    return np.sum(np.absolute(eigvals(a, sort=False, isherm=isherm)))


def norm(a, ntype=2, **kwargs):
    """Operator norms.

    Parameters
    ----------
        a: matrix, dense or sparse
        ntype: norm to calculate

    Returns
    -------
        x: matrix norm
    """
    types = {'2': '2', 2: '2', 'spectral': '2',
             'f': 'f', 'fro': 'f',
             't': 't', 'trace': 't', 'nuc': 't', 'tr': 't'}
    methods = {('2', 0): norm_2,
               ('2', 1): norm_2,
               ('t', 0): norm_trace_dense,
               ('f', 0): norm_fro_dense,
               ('f', 1): norm_fro_sparse}
    return methods[(types[ntype], issparse(a))](a, **kwargs)


# --------------------------------------------------------------------------- #
#                               Matrix functions                              #
# --------------------------------------------------------------------------- #

def expm(a, herm=False):
    """Matrix exponential, can be accelerated if explicitly hermitian.

    Parameters
    ----------
    a : dense or sparse matrix
        Matrix to exponentiate.
    herm : bool, optional
        If True (not default), and ``a`` is dense, digonalize the matrix
        in order to perform the exponential.

    Returns
    -------
    matrix
    """
    if issparse(a):
        # convert to and from csc to suppress scipy warning
        return spla.expm(a.tocsc()).tocsr()
    elif not herm:
        return np.asmatrix(spla.expm(a))
    else:
        evals, evecs = eigsys(a)
        return dot_dense(evecs, ldmul(np.exp(evals), evecs.H))


_EXPM_MULTIPLY_METHODS = {
    'SCIPY': spla.expm_multiply,
    'SLEPC': functools.partial(mfn_multiply_slepc_spawn, fntype='exp'),
}


def expm_multiply(mat, vec, backend="AUTO", **kwargs):
    """Compute the action of ``expm(mat)`` on ``vec``.

    Parameters
    ----------
    mat : matrix-like
        Matrix to exponentiate.
    vec : vector-like
        Vector to act with exponential of matrix on.
    backend : {'AUTO', 'SCIPY', 'SLEPC'}, optional
        Which backend to use.
    kwargs
        Supplied to backend function.

    Returns
    -------
    vector
        Result of ``expm(mat) @ vec``.
    """
    return _EXPM_MULTIPLY_METHODS[backend.upper()](mat, vec, **kwargs)


def sqrtm(a, herm=True):
    """Matrix square root, can be accelerated if explicitly hermitian.

    Parameters
    ----------
    a : dense or sparse matrix
        Matrix to take square root of.
    herm : bool, optional
        If True (the default), and ``a`` is dense, digonalize the matrix
        in order to take the square root.

    Returns
    -------
    matrix
    """
    if issparse(a):
        raise NotImplementedError("No sparse sqrtm available.")
    elif not herm:
        return np.asmatrix(sla.sqrtm(a))
    else:
        evals, evecs = eigsys(a)
        return dot_dense(evecs, ldmul(np.sqrt(evals.astype(complex)),
                                      evecs.H))
