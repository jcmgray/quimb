"""
Functions for solving matrices either fully or partially.
Note that the eigendecompositions here all assume a
hermitian matrix and sort the eigenvalues in ascending
algebraic order by default. Use explicit numpy/scipy linalg
routines for non-hermitian matrices.
"""

import numpy as np
import numpy.linalg as nla
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from .core import qjf


# -------------------------------------------------------------------------- #
# Full eigendecomposition methods for dense matrices                         #
# -------------------------------------------------------------------------- #

def eigsys(a, sort=True):
    """ Find all eigenpairs of dense, hermitian matrix.
    Input:
        a: hermitian matrix
        sort: whether to sort the eigenpairs in ascending eigenvalue order
    Returns:
        l: array of eigenvalues
        v: corresponding eigenvectors as columns of matrix
    """
    l, v = nla.eigh(a)
    if sort:
        sortinds = np.argsort(l)
        return l[sortinds], np.asmatrix(v[:, sortinds])
    else:
        return l, v


def eigvals(a, sort=True):
    """ Find all eigenvalues of dense matrix
    Input:
        a: hermitian matrix
        sort: whether to sort the eigenvalues in ascending order
    Returns:
        l: array of eigenvalues
    """
    l = nla.eigvalsh(a)
    return np.sort(l) if sort else l


def eigvecs(a, sort=True):
    """ Find all eigenvectors of dense matrix
    Input:
        a: hermitian matrix
        sort: whether to sort the eigenvectors in ascending eigenvalue order
    Returns:
        v: eigenvectors as columns of matrix
    """
    l, v = eigsys(a, sort=sort)
    return v


# -------------------------------------------------------------------------- #
# iterative methods for partial eigendecompision                             #
# -------------------------------------------------------------------------- #

def seigsys(a, k=6, which='SA', ncv=None, return_vecs=True, **kwargs):
    """ Returns a few eigenpairs from a possibly sparse hermitian operator
    Inputs:
        a: matrix, probably sparse, hermitian
        k: number of eigenpairs to return
        which: where in spectrum to take eigenvalues from (see scipy eigsh)
        nvc: number of lanczos vectors, can use to optimise speed
    Returns:
        lk: array of eigenvalues
        vk: matrix of eigenvectors as columns
    """
    n = a.shape[0]
    sparse = sp.issparse(a)
    if not sparse and n <= 500:
        # TODO: select which from nla full spectrum
        if return_vecs:
            lk, vk = eigsys(a)
            return lk[:k], vk[:, :k]
        else:
            lk = eigvals(a)
            return lk[:k]
    else:
        ncv = choose_ncv(k, n) if ncv is None else ncv
        if return_vecs:
            lk, vk = spla.eigsh(a, k=k, which=which, ncv=ncv, **kwargs)
            sortinds = np.argsort(lk)
            return lk[sortinds], np.asmatrix(vk[:, sortinds])
        else:
            lk = spla.eigsh(a, k=k, which=which, ncv=ncv,
                            return_eigenvectors=False, **kwargs)
            return np.sort(lk)


def seigvals(a, k=6, **kwargs):
    return seigsys(a, k=k, return_vecs=False, **kwargs)


def seigvecs(a, k=6, **kwargs):
    _, v = seigsys(a, k=k, return_vecs=True, **kwargs)
    return v


def groundstate(ham):
    """ Alias for finding lowest eigenvector only. """
    return seigvecs(ham, k=1, which='SA')


def groundenergy(ham):
    """ Alias for finding lowest eigenvalue only. """
    return seigvals(ham, k=1, which='SA')


# -------------------------------------------------------------------------- #
# iterative methods for partial singular value decomposition                 #
# -------------------------------------------------------------------------- #

def svds(a, k=6, ncv=None, return_vecs=True, **kwargs):
    """ Compute a number of singular value pairs """
    n = a.shape[0]
    sparse = sp.issparse(a)
    if not sparse and n <= 500:  # small and dense --> use full decomposition
        if return_vecs:
            uk, sk, vtk = nla.svd(a)
            return uk[:, 0:k], sk[0:k], vtk[0:k, :]
        else:
            return nla.svd(a, compute_uv=False)[:k]
    else:
        ncv = choose_ncv(k, n) if ncv is None else ncv
        if return_vecs:
            uk, sk, vtk = spla.svds(a, k=k, ncv=ncv, **kwargs)
            so = np.argsort(-sk)
            return qjf(uk[:, so]), sk[so], qjf(vtk[so, :])
        else:
            sk = spla.svds(a, k=k, ncv=ncv,
                           return_singular_vectors=False, **kwargs)
            return sk[np.argsort(-sk)]


def norm2(a):
    """ Return the 2-norm of matrix, a, i.e. the largest singular value. """
    return svds(a, k=1, return_vecs=False)[0]


def choose_ncv(k, n):
    """ Optimise number of lanczos vectors for iterative methods
    Args:
        k: number of target eigenvalues/singular values
        n: matrix size
    Returns:
        ncv: number of lanczos vectors to use
    """
    return min(max(20, 2 * k + 1), n)
