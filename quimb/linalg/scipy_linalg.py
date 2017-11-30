"""Scipy based linear algebra.
"""

import numpy as np
import scipy.sparse.linalg as spla


def seigsys_scipy(a, k=6, *, which=None, return_vecs=True, sigma=None,
                  isherm=True, sort=True, tol=None, **eigs_opts):
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
    # Options that might get passed that scipy doesn't support
    eigs_opts.pop('EPSType', None)

    # convert certain options for scipy
    settings = {
        'k': k,
        'which': ('SA' if (which is None) and (sigma is None) else
                  'LM' if (which is None) and (sigma is not None) else
                  # For target using shift-invert scipy requires 'LM' ->
                  'LM' if ('T' in which.upper()) and (sigma is not None) else
                  which),
        'sigma': sigma,
        'return_eigenvectors': return_vecs,
        'tol': 0 if tol is None else tol
    }

    fn = spla.eigsh if isherm else spla.eigs

    if return_vecs:
        lk, vk = fn(a, **settings, **eigs_opts)
        sortinds = np.argsort(lk)
        return lk[sortinds], np.asmatrix(vk[:, sortinds])
    else:
        lk = fn(a, **settings, **eigs_opts)
        return np.sort(lk) if sort else lk


def scipy_svds(a, k=6, *, return_vecs=True, **kwargs):
    """Compute a number of singular value pairs
    """
    settings = {
        'k': k,
        'return_singular_vectors': return_vecs
    }
    if return_vecs:
        uk, sk, vtk = spla.svds(a, **settings, **kwargs)
        so = np.argsort(-sk)
        return np.asmatrix(uk[:, so]), sk[so], np.asmatrix(vtk[so, :])
    else:
        sk = spla.svds(a, **settings, **kwargs)
        return sk[np.argsort(-sk)]
