"""Functions for solving matrices either fully or partially.
Note that the eigendecompositions here all assume a
hermitian matrix and sort the eigenvalues in ascending
algebraic order by default.
"""

# TODO: restart eigen and svd - up to tol
# TODO: test non-herm
# TODO: factor out numpy
# TODO: add petsc / elemental ...
# TODO: fix slepc svds instability

import numpy as np
import numpy.linalg as nla

from ..accel import issparse, vdot
from .numpy_solver import numpy_seigsys, numpy_svds
from .scipy_solver import scipy_seigsys, scipy_svds
from . import SLEPC4PY_FOUND
if SLEPC4PY_FOUND:
    from .slepc_solver import slepc_seigsys, slepc_svds


# -------------------------------------------------------------------------- #
# Full eigendecomposition methods for dense matrices                         #
# -------------------------------------------------------------------------- #

def eigsys(a, sort=True, isherm=True):
    """Find all eigenpairs of dense, hermitian matrix.

    Parameters
    ----------
        a: hermitian matrix
        sort: whether to sort the eigenpairs in ascending eigenvalue order

    Returns
    -------
        evals: array of eigenvalues
        evecs: corresponding eigenvectors as columns of matrix
    """
    evals, evecs = nla.eigh(a) if isherm else nla.eig(a)
    if sort:
        sortinds = np.argsort(evals)
        return evals[sortinds], np.asmatrix(evecs[:, sortinds])
    return evals, np.asmatrix(evecs)


def eigvals(a, sort=True, isherm=True):
    """Find all eigenvalues of dense, hermitian matrix

    Parameters
    ----------
        a: hermitian matrix
        sort: whether to sort the eigenvalues in ascending order

    Returns
    -------
        evals: array of eigenvalues
    """
    evals = nla.eigvalsh(a) if isherm else nla.eigvals(a)
    return np.sort(evals) if sort else evals


def eigvecs(a, sort=True, isherm=True):
    """Find all eigenvectors of dense, hermitian matrix

    Parameters
    ----------
        a: hermitian matrix
        sort: whether to sort the eigenvectors in ascending eigenvalue order

    Returns
    -------
        evecs: eigenvectors as columns of matrix
    """
    return eigsys(a, sort=sort, isherm=isherm)[1]


# -------------------------------------------------------------------------- #
# iterative methods for partial eigendecompision                             #
# -------------------------------------------------------------------------- #


def choose_backend(a, k, int_eps=False):
    """Pick a backend automatically for partial decompositions.
    """
    # small matrix or large part of subspace requested
    small_d_big_k = a.shape[0] ** 2 / k < (10000 if int_eps else 2000)
    # avoid using slepc for dense matrices and inner eigenvectors
    slepc_suitable = not int_eps or issparse(a)
    return ("NUMPY" if small_d_big_k else
            "SLEPC" if SLEPC4PY_FOUND and slepc_suitable else
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
        bkd = choose_backend(a, k, sigma is not None)
    seig_func = (slepc_seigsys if bkd == 'SLEPC' else
                 numpy_seigsys if bkd in {'NUMPY', 'DENSE'} else
                 scipy_seigsys if bkd == 'SCIPY' else
                 None)
    return seig_func(a, **settings, **kwargs)


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
    bkd = (choose_backend(a, k, False) if backend in {'auto', 'AUTO'} else
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
