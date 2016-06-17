""" Functions for solving matrices either fully or partially.
Note that the eigendecompositions here all assume a
hermitian matrix and sort the eigenvalues in ascending
algebraic order by default. Use explicit numpy/scipy linalg
routines for non-hermitian matrices. """
# TODO: test non-herm

import numpy as np
import numpy.linalg as nla

from ..accel import issparse, vdot
from .scipy_solver import scipy_seigsys, scipy_svds

from . import slepc4py_found
if slepc4py_found():
    from .slepc_solver import slepc_seigsys, slepc_svds


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


def seigsys(a, k=6, which=None, return_vecs=True, sigma=None,
            isherm=True, ncv=None, sort=True, backend='scipy', **kwargs):
    """
    Returns a few eigenpairs from a possibly sparse hermitian operator

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
    # TODO: autodense for n < 500
    settings = {
        'k': k,
        'which': which,
        'return_vecs': return_vecs,
        'sigma': sigma,
        'isherm': isherm,
        'ncv': ncv,
        'sort': sort}
    seig_func = (slepc_seigsys if backend.lower() == 'slepc' else
                 scipy_seigsys)
    return seig_func(a, **settings, **kwargs)


def seigvals(a, k=6, **kwargs):
    """ Seigsys alias for finding eigenvalues only. """
    return seigsys(a, k=k, return_vecs=False, **kwargs)


def seigvecs(a, k=6, **kwargs):
    """ Seigsys alias for finding eigenvectors only. """
    _, v = seigsys(a, k=k, return_vecs=True, **kwargs)
    return v


def groundstate(ham, **kwargs):
    """ Alias for finding lowest eigenvector only. """
    return seigvecs(ham, k=1, which='SA', **kwargs)


def groundenergy(ham, **kwargs):
    """ Alias for finding lowest eigenvalue only. """
    return seigvals(ham, k=1, which='SA', **kwargs)[0]


# -------------------------------------------------------------------------- #
# iterative methods for partial singular value decomposition                 #
# -------------------------------------------------------------------------- #

def svd(a, return_vecs=True):
    """ Compute full singular value decomposition of matrix. """
    return nla.svd(a, full_matrices=False, compute_uv=return_vecs)


def svds(a, k=6, ncv=None, return_vecs=True, backend='scipy', **kwargs):
    """
    Compute a number of singular value pairs
    """
    # TODO: autodense for n < 500
    settings = {
        'k': k,
        'ncv': ncv,
        'return_vecs': return_vecs}
    svd_func = (slepc_svds if backend.lower() == 'slepc' else
                scipy_svds)
    return svd_func(a, **settings, **kwargs)


# -------------------------------------------------------------------------- #
# Norms and other quantities based on decompositions                         #
# -------------------------------------------------------------------------- #

def norm_2(a, **kwargs):
    """ Return the 2-norm of matrix, a, i.e. the largest singular value. """
    return svds(a, k=1, return_vecs=False, **kwargs)[0]


def norm_fro_dense(a):
    """ Frobenius norm for dense matrices """
    return vdot(a, a).real**0.5


def norm_fro_sparse(a):
    return vdot(a.data, a.data).real**0.5


def norm_trace_dense(a, isherm=True):
    """ Returns the trace norm of operator a, that is,
    the sum of abs eigvals. """
    return np.sum(np.absolute(eigvals(a, sort=False, isherm=isherm)))


def norm(a, ntype=2, **kwargs):
    """ Operator norms.

    Parameters
    ----------
        a: matrix, dense or sparse
        ntype: norm to calculate

    Returns
    -------
        x: matrix norm """
    types = {'2': '2', 2: '2', 'spectral': '2',
             'f': 'f', 'fro': 'f',
             't': 't', 'trace': 't', 'nuc': 't', 'tr': 't'}
    methods = {('2', 0): norm_2,
               ('2', 1): norm_2,
               ('t', 0): norm_trace_dense,
               ('f', 0): norm_fro_dense,
               ('f', 1): norm_fro_sparse}
    return methods[(types[ntype], issparse(a))](a, **kwargs)
