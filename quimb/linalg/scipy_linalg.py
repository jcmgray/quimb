"""Scipy based linear algebra.
"""

import numpy as np
import scipy.sparse.linalg as spla

import quimb as qu


def maybe_sort_and_project(lk, vk, P, sort=True):
    if sort:
        sortinds = np.argsort(lk)
        lk, vk = lk[sortinds], np.asmatrix(vk[:, sortinds])

    # map eigenvectors out of subspace
    if P is not None:
        vk = P @ vk

    return lk, vk


def eigs_scipy(A, k, *, B=None, which=None, return_vecs=True, sigma=None,
               isherm=True, sort=True, P=None, tol=None, **eigs_opts):
    """Returns a few eigenpairs from a possibly sparse hermitian operator

    Parameters
    ----------
    A : dense-matrix, sparse-matrix, LinearOperator or quimb.Lazy
        The operator to solve for.
    k : int
        Number of eigenpairs to return
    B : dense-matrix, sparse-matrix, LinearOperator or quimb.Lazy, optional
        If given, the RHS matrix (which should be positive) defining a
        generalized eigen problem.
    which : str, optional
        where in spectrum to take eigenvalues from (see
        :func:`scipy.sparse.linalg.eigsh`).
    return_vecs : bool, optional
        Whether to return the eigenvectors as well.
    sigma : float, optional
        Shift, if targeting interior eigenpairs.
    isherm : bool, optional
        Whether ``A`` is hermitian.
    P : dense-matrix, sparse-matrix, LinearOperator or quimb.Lazy, optional
        Perform the eigensolve in the subspace defined by this projector.
    sort : bool, optional
        Whether to ensure the eigenvalues are sorted in ascending value.
    eigs_opts
        Supplied to :func:`scipy.sparse.linalg.eigsh` or
        :func:`scipy.sparse.linalg.eigs`.

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

    if isinstance(A, qu.Lazy):
        A = A()
    if isinstance(B, qu.Lazy):
        B = B()
    if isinstance(P, qu.Lazy):
        P = P()

    eig_fn = spla.eigsh if isherm else spla.eigs

    # project into subspace
    if P is not None:
        A = P.H @ (A @ P)

    if return_vecs:
        lk, vk = eig_fn(A, **settings, **eigs_opts)
        return maybe_sort_and_project(lk, vk, P, sort)
    else:
        lk = eig_fn(A, **settings, **eigs_opts)
        return np.sort(lk) if sort else lk


def eigs_lobpcg(A, k, *, B=None, v0=None, which=None, return_vecs=True,
                sigma=None, isherm=True, P=None, sort=True, **lobpcg_opts):
    """Interface to scipy's lobpcg eigensolver, which can be good for
    generalized eigenproblems with matrix-free operators. Seems to a be a bit
    innacurate though (e.g. on the order of ~ 1e-6 for eigenvalues). Also only
    takes real, symmetric problems, targeting smallest eigenvalues (though
    scipy will soon have complex support, and its easy to add oneself).

    Note that the slepc eigensolver also has a lobpcg backend
    (``EPSType='lobpcg'``) which accepts complex input and is more accurate -
    though seems slower.

    Parameters
    ----------
    A : dense-matrix, sparse-matrix, LinearOperator or callable
        The operator to solve for.
    k : int
        Number of eigenpairs to return
    B : dense-matrix, sparse-matrix, LinearOperator or callable, optional
        If given, the RHS matrix (which should be positive) defining a
        generalized eigen problem.
    v0 : array_like (d, k), optional
        The initial subspace to iterate with.
    which : {'SA', 'LA'}, optional
        Find the smallest or largest eigenvalues.
    return_vecs : bool, optional
        Whether to return the eigenvectors found.
    P : dense-matrix, sparse-matrix, LinearOperator or callable, optional
        Perform the eigensolve in the subspace defined by this projector.
    sort : bool, optional
        Whether to ensure the eigenvalues are sorted in ascending value.
    lobpcg_opts
        Supplied to :func:`scipy.sparse.linagl.lobpcg`.

    Returns
    -------
    lk : array_like (k,)
        The eigenvalues.
    vk : array_like (d, k)
        The eigenvectors, if `return_vecs=True`.

    See Also
    --------
    eigs_scipy, eigs_numpy, eigs_slepc
    """
    if not isherm:
        raise ValueError("lobpcg can only solve symmetric problems.")

    if sigma is not None:
        raise ValueError("lobpcg can only solve extremal eigenvalues.")

    # remove invalid options for lobpcg
    lobpcg_opts.pop('ncv', None)
    lobpcg_opts.pop('EPSType', None)

    # convert some arguments and defaults
    lobpcg_opts.setdefault('maxiter', 30)
    if lobpcg_opts['maxiter'] is None:
        lobpcg_opts['maxiter'] = 30
    largest = {'SA': False, 'LA': True}[which]

    if isinstance(A, qu.Lazy):
        A = A()
    if isinstance(B, qu.Lazy):
        B = B()
    if isinstance(P, qu.Lazy):
        P = P()

    # project into subspace
    if P is not None:
        A = P.H @ (A @ P)

    d = A.shape[0]

    # set up the initial subsspace to iterate with
    if v0 is None:
        v0 = np.random.choice([1.0, -1.0], size=(d, k))
    else:
        # check if intial space should be projected too
        if P is not None and v0.shape[0] != d:
            v0 = P.H @ v0

        v0 = v0.reshape(d, -1)

        # if not enough initial states given, flesh out with random
        if v0.shape[1] != k:
            v0 = np.hstack(v0, np.random.randn(d, k - v0.shape[1]))

    lk, vk = spla.lobpcg(A=A, X=v0, B=B, largest=largest, **lobpcg_opts)

    if return_vecs:
        return maybe_sort_and_project(lk, vk, P, sort)
    else:
        return np.sort(lk) if sort else lk


def svds_scipy(a, k=6, *, return_vecs=True, **svds_opts):
    """Compute a number of singular value pairs
    """
    settings = {
        'k': k,
        'return_singular_vectors': return_vecs
    }
    if return_vecs:
        uk, sk, vtk = spla.svds(a, **settings, **svds_opts)
        so = np.argsort(-sk)
        return np.asmatrix(uk[:, so]), sk[so], np.asmatrix(vtk[so, :])
    else:
        sk = spla.svds(a, **settings, **svds_opts)
        return sk[np.argsort(-sk)]
