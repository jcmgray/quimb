import numpy as np
import numpy.linalg as nla

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


def seigsys_numpy(a, k=6, which=None, return_vecs=True, sigma=None,
                  isherm=True, sort=True, **kwargs):
    """Partial eigen-decomposition using numpy's dense linear algebra.

    Parameters
    ----------
        a: operator to partially eigen-decompose
        k: number of eigenpairs to return
        which: which part of the spectrum to target
        return_vecs: whether to return eigenvectors
        sigma: target eigenvalue
        isherm: whether `a` is hermitian
        sort: whether to sort reduced list of eigenpairs into ascending order
        **kwargs: settings to pass to numpy.eig... functions

    Returns
    -------
        lk, (vk): k eigenvalues (and eigenvectors) sorted according to which
    """
    efunc = {(True, True): nla.eigh,
             (True, False): nla.eigvalsh,
             (False, True): nla.eig,
             (False, False): nla.eigvals}[(isherm, return_vecs)]

    kwargs.pop('ncv', None)  # might be given by seigsys but not used

    if return_vecs:
        evals, evecs = efunc(a.A if issparse(a) else a, **kwargs)
        sk = sort_inds(evals, method=which, sigma=sigma)[:k]
        evals, evecs = evals[sk], np.asmatrix(evecs[:, sk])
        if sort:
            so = np.argsort(evals)
            return evals[so], evecs[:, so]
        return evals, evecs
    else:
        evals = efunc(a.A if issparse(a) else a, **kwargs)
        sk = sort_inds(evals, method=which, sigma=sigma)[:k]
        evals = evals[sk]
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
