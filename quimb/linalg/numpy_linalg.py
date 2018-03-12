"""Numpy base linear algebra.
"""

import numpy as np
import numpy.linalg as nla
import scipy.linalg as scla

from ..accel import issparse


def eigsys_numpy(a, sort=True, isherm=True):
    """Numpy based dense eigensolve.
    """
    if isherm:
        evals, evecs = nla.eigh(a)
    else:
        evals, evecs = nla.eig(a)

    if sort:
        sortinds = np.argsort(evals)
        return evals[sortinds], np.asmatrix(evecs[:, sortinds])

    return evals, np.asmatrix(evecs)


def eigvals_numpy(a, sort=True, isherm=True):
    """Numpy based dense eigenvalues.
    """
    if isherm:
        evals = nla.eigvalsh(a)
    else:
        evals = nla.eigvals(a)

    if sort:
        return np.sort(evals)

    return evals


def sort_inds(a, method, sigma=None):
    """Return the sorting inds of a list

    Parameters
    ----------
        a : array_like
            List to base sort on.
        method : str
            Method of sorting list, one of
                * "LM" - Largest magnitude first
                * "SM" - Smallest magnitude first
                * "SA" - Smallest algebraic first
                * "SR" - Smallest real part first
                * "SI" - Smallest imaginary part first
                * "LA" - Largest algebraic first
                * "LR" - Largest real part first
                * "LI" - Largest imaginary part first
                * "TM" - Magnitude closest to target sigma first
                * "TR" - Real part closest to target sigma first
                * "TI" - Imaginary part closest to target sigma first
        sigma : float, optional
            The target if method={"TM", "TR", or "TI"}.

    Returns
    -------
        inds : array of int
            Indices that would sort `a` based on `method`
    """
    _SORT_FUNCS = {
        "LM": lambda a: -abs(a),
        "SM": lambda a: -abs(1 / a),
        "SA": lambda a: a,
        "SR": lambda a: a.real,
        "SI": lambda a: a.imag,
        "LA": lambda a: -a,
        "LR": lambda a: -a.real,
        "LI": lambda a: -a.imag,
        "TM": lambda a: -1 / abs(abs(a) - sigma),
        "TR": lambda a: -1 / abs(a.real - sigma),
        "TI": lambda a: -1 / abs(a.imag - sigma),
    }
    return np.argsort(_SORT_FUNCS[method.upper()](a))


_DENSE_EIG_METHODS = {
    (True, True, False): nla.eigh,
    (True, False, False): nla.eigvalsh,
    (False, True, False): nla.eig,
    (False, False, False): nla.eigvals,
    (True, True, True): scla.eigh,
    (True, False, True): scla.eigvalsh,
    (False, True, True): scla.eig,
    (False, False, True): scla.eigvals,
}


def seigsys_numpy(A, k=6, B=None, which=None, return_vecs=True, sigma=None,
                  isherm=True, sort=True, **eig_opts):
    """Partial eigen-decomposition using numpy's dense linear algebra.

    Parameters
    ----------
    A : matrix-like
        Operator to partially eigen-decompose.
    k : int, optional
        Number of eigenpairs to return.
    B : matrix-like
        If given, the RHS matrix defining a generalized eigen problem.
    which : str, optional
        Which part of the spectrum to target.
    return_vecs : bool, optional
        Whether to return eigenvectors.
    sigma : None or float, optional
        Target eigenvalue.
    isherm : bool, optional
        Whether `a` is hermitian.
    sort : bool, optional
        Whether to sort reduced list of eigenpairs into ascending order.
    eig_opts
        Settings to pass to numpy.eig... functions.

    Returns
    -------
        lk, (vk): k eigenvalues (and eigenvectors) sorted according to which
    """
    generalized = B is not None

    eig_fn = _DENSE_EIG_METHODS[(isherm, return_vecs, generalized)]

    if generalized:
        eig_opts['b'] = B

    # these might be given by seigsys but not relevant for numpy
    eig_opts.pop('ncv', None)
    eig_opts.pop('v0', None)
    eig_opts.pop('tol', None)
    eig_opts.pop('maxiter', None)
    eig_opts.pop('EPSType', None)

    if return_vecs:
        # get all eigenpairs
        evals, evecs = eig_fn(A.A if issparse(A) else A, **eig_opts)

        # sort and trim according to which k we want
        sk = sort_inds(evals, method=which, sigma=sigma)[:k]
        evals, evecs = evals[sk], np.asmatrix(evecs[:, sk])

        # also potentially sort into ascending order
        if sort:
            so = np.argsort(evals)
            return evals[so], evecs[:, so]

        return evals, evecs

    else:
        # get all eigenvalues
        evals = eig_fn(A.A if issparse(A) else A, **eig_opts)

        # sort and trim according to which k we want
        sk = sort_inds(evals, method=which, sigma=sigma)[:k]
        evals = evals[sk]

        # also potentially sort into ascending order
        return np.sort(evals) if sort else evals


def numpy_svds(a, k=6, return_vecs=True, **_):
    """Partial singular value decomposition using numpys (full) singular value
    decomposition.

    Parameters
    ----------
        a: operator decompose
        k: number of singular value triplets to retrieve
        return_vecs: whether to return the computed vecs or values only
        ncv: redundant, for compatibility only.

    Returns
    -------
        (uk,) sk (, vkt): singlar value triplets
    """
    if return_vecs:
        uk, sk, vkt = nla.svd(a.A if issparse(a) else a, compute_uv=True)
        return np.asmatrix(uk[:, :k]), sk[:k], np.asmatrix(vkt[:k, :])
    else:
        sk = nla.svd(a.A if issparse(a) else a, compute_uv=False)
        return sk[:k]
