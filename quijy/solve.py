"""
Functions for solving matrices either fully or partially
"""

import numpy as np
import numpy.linalg as nla
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from quijy.core import qonvert


def eigsys(a, sort=True):
    """ Find sorted eigenpairs of matrix
    Input:
        a: hermitian matrix
        sort: whether to sort the eigenpairs in ascending eigenvalue order
    Returns:
        l: array of eigenvalues, if sorted, by ascending algebraic order
        v: corresponding eigenvectors as columns of matrix
    """
    l, v = nla.eigh(a)
    if sort:
        sortinds = np.argsort(l)
        return l[sortinds], qonvert(v[:, sortinds])
    else:
        return l, v


def eigvals(a, sort=True):
    """ Find sorted eigenvalues of matrix
    Input:
        a: hermitian matrix
    Returns:
        l: array of eigenvalues, if sorted, by ascending algebraic order
    """
    l = nla.eigvalsh(a)
    return np.sort(l) if sort else l


def eigvecs(a, sort=True):
    """ Find sorted eigenvectors of matrix
    Input:
        a: hermitian matrix
    Returns:
        v: eigenvectors as columns of matrix, if sorted, by ascending
        eigenvalue order
    """
    l, v = nla.eigh(a)
    return qonvert(v[:, np.argsort(l)]) if sort else qonvert(v)


def calcncv(k, n, sparse):
    """ Optimise number of lanczos vectors for...
    Args:
        k: number of target eigenvalues/singular values
        n: matrix size
        sparse:  if matrix is sparse
        #TODO: sparsity?
    Returns:
        ncv: number of lanczos vectors to use
    """
    if sparse:
        ncv = max(8, 2 * k + 2)
    else:
        ncv = max(10, n//2**5 - 1, k * 2 + 2)
    return ncv
    pass


def seigsys(a, k=1, which='SA', ncv=None, **kwargs):
    """
    Returns a few eigenpairs from a possibly sparse hermitian operator
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
    if not sparse and n <= 500:  # Small dense matrices can use full decomp.
        lk, vk = eigsys(a)
        return lk[0:k], vk[:, 0:k]
    else:
        ncv = calcncv(k, n, sparse) if ncv is None else ncv
        lk, vk = spla.eigsh(a, k=k, which=which, ncv=ncv, **kwargs)
        return lk, qonvert(vk)


def seigvals(a, k=1, which='SA', ncv=None, **kwargs):
    n = a.shape[0]
    sparse = sp.issparse(a)
    if not sparse and n <= 500:  # Small dense matrices can use full decomp.
        lk = eigvals(a)
        return lk[0:k]
    else:
        ncv = calcncv(k, n, sparse) if ncv is None else ncv
        return spla.eigsh(a, k=k, which=which, ncv=ncv,
                          return_eigenvectors=False, **kwargs)


def seigvecs(a, k=1, which='SA', ncv=None, **kwargs):
    l, v = seigsys(a, k, which, ncv, **kwargs)
    return v


def groundstate(ham):
    """ Alias for finding lowest eigenvector only. """
    return seigvecs(a)


def groundenergy(ham):
    """ Alias for finding lowest eigenvalue only. """
    return seigvals(ham)


def svds(a, k=1, ncv=None, **kwargs):
    """
    Compute a number of singular values
    """
    n = a.shape[0]
    sparse = sp.issparse(a)
    if not sparse and n <= 500:
        u, s, vt = nla.svd(a)
    else:
        ncv = calcncv(k, n, sparse) if ncv is None else ncv
        u, s, vt = spla.svds(a, k=k, ncv=ncv, **kwargs)
    return qonvert(u), s, qonvert(vt)


def norm(a):
    """
    Return the 2-norm of matrix, a, i.e. the largest singular value.
    """
    return svds(a, k=1, return_singular_vectors=False)[0]
