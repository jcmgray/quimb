import numpy as np
import scipy.sparse.linalg as spla

from ..accel import accel


@accel
def choose_ncv(k, n):  # pragma: no cover
    """ Optimise number of lanczos vectors for iterative methods

    Parameters
    ----------
        k: number of target eigenvalues/singular values
        n: matrix size

    Returns
    -------
        ncv: number of lanczos vectors to use
    """
    return min(max(20, 2 * k + 1), n)


def scipy_seigsys(a, k=6, which=None, return_vecs=True, sigma=None,
                  isherm=True, ncv=None, sort=True, **kwargs):
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
        vk: matrix of eigenvectors as columns
    """
    settings = {
        'k': k,
        'which': ('LM' if 'T' in which.upper() and sigma is not None else
                  'SA' if which is None and sigma is None else
                  'LM' if which is None and sigma is not None else
                  which),
        'sigma': sigma,
        'ncv': choose_ncv(k, a.shape[0]) if ncv is None else ncv,
        'return_eigenvectors': return_vecs}
    seig_func = spla.eigsh if isherm else spla.eigs
    if return_vecs:
        lk, vk = seig_func(a, **settings, **kwargs)
        sortinds = np.argsort(lk)
        return lk[sortinds], np.asmatrix(vk[:, sortinds])
    else:
        lk = seig_func(a, **settings, **kwargs)
        return np.sort(lk) if sort else lk


def scipy_svds(a, k=6, ncv=None, return_vecs=True, **kwargs):
    """ Compute a number of singular value pairs """
    settings = {
        'k': k,
        'ncv': choose_ncv(k, a.shape[0]) if ncv is None else ncv,
        'return_singular_vectors': return_vecs}
    if return_vecs:
        uk, sk, vtk = spla.svds(a, **settings, **kwargs)
        so = np.argsort(-sk)
        return np.asmatrix(uk[:, so]), sk[so], np.asmatrix(vtk[so, :])
    else:
        sk = spla.svds(a, **settings, **kwargs)
        return sk[np.argsort(-sk)]
