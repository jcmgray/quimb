"""Scipy based linear algebra.
"""

import numpy as np
import scipy.sparse.linalg as spla


def seigsys_scipy(A, k, *, B=None, which=None, return_vecs=True, sigma=None,
                  isherm=True, sort=True, tol=None, **eigs_opts):
    """Returns a few eigenpairs from a possibly sparse hermitian operator

    Parameters
    ----------
    A : sparse matrix-like, dense matrix-like, or LinearOperator
        The operator to solve for.
    k : int, optional
        Number of eigenpairs to return
    B : sparse matrix-like, dense matrix-like, or LinearOperator, optional
        If given, the RHS matrix defining a generalized eigen problem.
    which : str, optional
        where in spectrum to take eigenvalues from (see scipy eigsh)
    ncv: int, optional
        Number of lanczos vectors, can use to optimise speed

    Returns
    -------
    lk[, vk] : numpy.ndarray[, numpy.matrix]
        lk: array of eigenvalues, vk: matrix of eigenvectors as columns.
    """
    # Options that might get passed that scipy doesn't support
    eigs_opts.pop('EPSType', None)

    # convert certain options for scipy
    settings = {
        'k': k,
        'M': B,
        'which': ('SA' if (which is None) and (sigma is None) else
                  'LM' if (which is None) and (sigma is not None) else
                  # For target using shift-invert scipy requires 'LM' ->
                  'LM' if ('T' in which.upper()) and (sigma is not None) else
                  which),
        'sigma': sigma,
        'return_eigenvectors': return_vecs,
        'tol': 0 if tol is None else tol
    }

    eig_fn = spla.eigsh if isherm else spla.eigs

    if return_vecs:
        lk, vk = eig_fn(A, **settings, **eigs_opts)
        sortinds = np.argsort(lk)
        return lk[sortinds], np.asmatrix(vk[:, sortinds])
    else:
        lk = eig_fn(A, **settings, **eigs_opts)
        return np.sort(lk) if sort else lk


def seigsys_lobpcg(A, k, *, B=None, v0=None, which=None, return_vecs=True,
                   sigma=None, isherm=True, sort=True, **lobpcg_opts):
    """Interface to scipy's lobpcg eigensolver, which can be good for
    generalized eigenproblems with matrix-free operators. Seems to a be a bit
    innacurate though (e.g. on the order of ~ 1e-6 for eigenvalues). Also only
    takes real, symmetric problems, targeting smallest eigenvalues.

    Note that the slepc eigensolver also has a lobpcg backend
    (``EPSType='lobpcg'``) which accepts complex input and is more accurate -
    though seems slower.

    Parameters
    ----------
    A : operator (n, n)
        The main operator, can be dense, sparse or a LinearOperator.
    k : int, optional
        Number of eigenvalues to find.
    B : operator (n, n), optional
        The RHS operator.
    v0 : array_like (n, k), optional
        The initial subspace to iterate with.
    which : {'SA', 'LA'}, optional
        Find the smallest or largest eigenvalues.
    return_vecs : bool, optional
        Whether to return the eigenvectors found.
    sort : bool, optional
        Whether to ensure the eigenvalues are sorted in ascending value.
    lobpcg_opts
        Supplied to :func:`scipy.sparse.linagl.lobpcg`.

    Returns
    -------
    lk : array_like (k,)
        The eigenvalues.
    vk : array_like (n, k)
        The eigenvectors, if `return_vecs=True`.

    See Also
    --------
    seigsys_scipy, seigsys_numpy, seigsys_slepc
    """
    if not isherm:
        raise ValueError("lobpcg can only solve symmetric problems.")

    if sigma is not None:
        raise ValueError("lobpcg can only solve extremal eigenvalues.")

    if not np.issubdtype(A.dtype, np.floating):
        raise ValueError("lobpcg can only solve real problems.")

    if (B is not None) and (not np.issubdtype(B.dtype, np.floating)):
        raise ValueError("lobpcg can only solve real problems.")

    # remove invalid options for lobpcg
    lobpcg_opts.pop('ncv', None)
    lobpcg_opts.pop('EPSType', None)

    # convert some arguments and defaults
    lobpcg_opts.setdefault('maxiter', 30)
    if lobpcg_opts['maxiter'] is None:
        lobpcg_opts['maxiter'] = 30
    largest = {'SA': False, 'LA': True}[which]

    n = A.shape[0]

    if v0 is None:
        v0 = np.random.choice([1.0, -1.0], size=(n, k))
    else:
        v0 = v0.reshape(n, -1)

    if v0.shape[1] != k:
        v0 = np.concatenate(v0, np.random.randn(n, k - v0.shape[1]), axis=1)

    lk, vk = spla.lobpcg(A=A, X=v0, B=B, largest=largest, **lobpcg_opts)

    if return_vecs:
        sortinds = np.argsort(lk)
        return lk[sortinds], np.asmatrix(vk[:, sortinds])
    else:
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
