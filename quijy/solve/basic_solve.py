""" Functions for solving matrices either fully or partially.
Note that the eigendecompositions here all assume a
hermitian matrix and sort the eigenvalues in ascending
algebraic order by default. Use explicit numpy/scipy linalg
routines for non-hermitian matrices. """
# TODO: test non-herm

import numpy as np
import numpy.linalg as nla
import scipy.sparse.linalg as spla
from numba import jit
from ..accel import issparse, vdot
from ..core import qjf


# -------------------------------------------------------------------------- #
# Full eigendecomposition methods for dense matrices                         #
# -------------------------------------------------------------------------- #

def eigsys(a, sort=True, isherm=True):
    """ Find all eigenpairs of dense, hermitian matrix.

    Parameters
    ----------
        a: hermitian matrix
        sort: whether to sort the eigenpairs in ascending eigenvalue order

    Returns
    -------
        l: array of eigenvalues
        v: corresponding eigenvectors as columns of matrix """
    l, v = nla.eigh(a) if isherm else nla.eig(a)
    if sort:
        sortinds = np.argsort(l)
        return l[sortinds], np.asmatrix(v[:, sortinds])
    return l, v


def eigvals(a, sort=True, isherm=True):
    """ Find all eigenvalues of dense, hermitian matrix

    Parameters
    ----------
        a: hermitian matrix
        sort: whether to sort the eigenvalues in ascending order

    Returns
    -------
        l: array of eigenvalues """
    l = nla.eigvalsh(a) if isherm else nla.eigvals(a)
    return np.sort(l) if sort else l


def eigvecs(a, sort=True, isherm=True):
    """ Find all eigenvectors of dense, hermitian matrix

    Parameters
    ----------
        a: hermitian matrix
        sort: whether to sort the eigenvectors in ascending eigenvalue order

    Returns
    -------
        v: eigenvectors as columns of matrix """
    _, v = eigsys(a, sort=sort, isherm=isherm)
    return v


# -------------------------------------------------------------------------- #
# iterative methods for partial eigendecompision                             #
# -------------------------------------------------------------------------- #

@jit(nopython=True)
def choose_ncv(k, n):  # pragma: no cover
    """ Optimise number of lanczos vectors for iterative methods

    Parameters
    ----------
        k: number of target eigenvalues/singular values
        n: matrix size

    Returns
    -------
        ncv: number of lanczos vectors to use """
    return min(max(20, 2 * k + 1), n)


def seigsys(a, k=6, which=None, return_vecs=True, sigma=None,
            isherm=True, ncv=None, **kwargs):
    """ Returns a few eigenpairs from a possibly sparse hermitian operator

    Parameters
    ----------
        a: matrix, probably sparse, hermitian
        k: number of eigenpairs to return
        which: where in spectrum to take eigenvalues from (see scipy eigsh)
        nvc: number of lanczos vectors, can use to optimise speed

    Returns
    -------
        lk: array of eigenvalues
        vk: matrix of eigenvectors as columns """
    n = a.shape[0]
    if which is None:
        which = 'SA' if sigma is None else 'LM'
    if not issparse(a) and n <= 500 and which == 'SA':
        if return_vecs:
            lk, vk = eigsys(a, isherm=isherm)
            return lk[:k], vk[:, :k]
        else:
            lk = eigvals(a, isherm=isherm)
            return lk[:k]
    else:
        ncv = choose_ncv(k, n) if ncv is None else ncv
        seig_func = spla.eigsh if isherm else spla.eigs
        lvk = seig_func(a, k=k, which=which, ncv=ncv, sigma=sigma,
                        return_eigenvectors=return_vecs, **kwargs)
        if return_vecs:
            sortinds = np.argsort(lvk[0])
            return lvk[0][sortinds], np.asmatrix(lvk[1][:, sortinds])
        else:
            return np.sort(lvk)


def seigvals(a, k=6, **kwargs):
    """ Seigsys alias for finding eigenvalues only. """
    return seigsys(a, k=k, return_vecs=False, **kwargs)


def seigvecs(a, k=6, **kwargs):
    """ Seigsys alias for finding eigenvectors only. """
    _, v = seigsys(a, k=k, return_vecs=True, **kwargs)
    return v


def groundstate(ham):
    """ Alias for finding lowest eigenvector only. """
    return seigvecs(ham, k=1, which='SA')


def groundenergy(ham):
    """ Alias for finding lowest eigenvalue only. """
    return seigvals(ham, k=1, which='SA')[0]


# -------------------------------------------------------------------------- #
# iterative methods for partial singular value decomposition                 #
# -------------------------------------------------------------------------- #

def svd(a, return_vecs=True):
    """ Compute full singular value decomposiont of matrix. """
    return nla.svd(a, full_matrices=False, compute_uv=return_vecs)


def svds(a, k=6, ncv=None, return_vecs=True, **kwargs):
    """ Compute a number of singular value pairs """
    n = a.shape[0]
    sparse = issparse(a)
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


# -------------------------------------------------------------------------- #
# Norms and other quantities based on decompositions                         #
# -------------------------------------------------------------------------- #

def norm_2(a):
    """ Return the 2-norm of matrix, a, i.e. the largest singular value. """
    return svds(a, k=1, return_vecs=False)[0]


def norm_fro_dense(a):
    """ Frobenius norm for dense matrices """
    return vdot(a, a).real**0.5


def norm_fro_sparse(a):
    return vdot(a.data, a.data).real**0.5


def norm_trace_dense(a, isherm=True):
    """ Returns the trace norm of operator a, that is,
    the sum of abs eigvals. """
    return np.sum(np.absolute(eigvals(a, sort=False, isherm=isherm)))


def norm(a, ntype=2):
    """ Operator norms.

    Parameters
    ----------
        a: matrix, dense or sparse
        ntype: norm to calculate

    Returns
    -------
        x: matrix norm """
    keys = {'2': '2', 2: '2', 'spectral': '2',
            'f': 'f', 'fro': 'f',
            't': 't', 'trace': 't', 'nuc': 't', 'tr': 't'}
    methods = {('2', 0): norm_2,
               ('2', 1): norm_2,
               ('t', 0): norm_trace_dense,
               ('f', 0): norm_fro_dense,
               ('f', 1): norm_fro_sparse}
    return methods[(keys[ntype], issparse(a))](a)
