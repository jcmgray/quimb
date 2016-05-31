"""
Interface to slepc4py for solving advanced eigenvalue problems.
"""
# TODO: delete solver or keep and extend
# TODO: FEAST / other contour solvers?
# TODO: exponential, sqrt etc.
# TODO: region for eps in middle, both ciss and normal
# TODO: handle dense matrices / full decomp

import numpy as np
import scipy.sparse as sp
from petsc4py import PETSc
from slepc4py import SLEPc


def scipy_to_petsc_csr(a):
    """ Convert a scipy sparse matrix to the relevant PETSc type, currently
    only supports csr and bsr formats. """
    if sp.isspmatrix_csr(a):
        b = PETSc.Mat().createAIJ(size=a.shape,
                                  csr=(a.indptr, a.indices, a.data))
    elif sp.isspmatrix_bsr(a):
        b = PETSc.Mat().createBAIJ(size=a.shape, bsize=a.blocksize,
                                   csr=(a.indptr, a.indices, a.data))
    else:
        b = PETSc.Mat().createDense(size=a.shape, array=a)
    return b


def init_eigensolver(which="LM", sigma=None, isherm=True, etype="krylovschur",
                     st_opts_dict={}, tol=None, max_it=None):
    """ Create an advanced eigensystem solver

    Parameters
    ----------
        sigma: target eigenvalue
        isherm: whether problem is hermitian or not

    Returns
    -------
        SLEPc solver ready to be called. """
    slepc_isherm = {
        True: SLEPc.EPS.ProblemType.HEP,
        False: SLEPc.EPS.ProblemType.NHEP,
    }
    scipy_to_slepc_which = {
        "LM": SLEPc.EPS.Which.LARGEST_MAGNITUDE,
        "SM": SLEPc.EPS.Which.SMALLEST_MAGNITUDE,
        "LR": SLEPc.EPS.Which.LARGEST_REAL,
        "LA": SLEPc.EPS.Which.LARGEST_REAL,
        "SR": SLEPc.EPS.Which.SMALLEST_REAL,
        "SA": SLEPc.EPS.Which.SMALLEST_REAL,
        "LI": SLEPc.EPS.Which.LARGEST_IMAGINARY,
        "SI": SLEPc.EPS.Which.SMALLEST_IMAGINARY,
        "TM": SLEPc.EPS.Which.TARGET_MAGNITUDE,
        "TR": SLEPc.EPS.Which.TARGET_REAL,
        "TI": SLEPc.EPS.Which.TARGET_IMAGINARY,
    }
    eigensolver = SLEPc.EPS()
    eigensolver.create()
    if sigma is not None:
        which = "TM"
        eigensolver.setST(init_spectral_inverter(**st_opts_dict))
        eigensolver.setTarget(sigma)
    eigensolver.setType(etype)
    eigensolver.setProblemType(slepc_isherm[isherm])
    eigensolver.setWhichEigenpairs(scipy_to_slepc_which[which.upper()])
    eigensolver.setConvergenceTest(SLEPc.EPS.Conv.ABS)
    eigensolver.setTolerances(tol=tol, max_it=max_it)
    return eigensolver


def init_spectral_inverter(ptype="lu", ppackage="mumps", ktype="preonly",
                           stype="sinvert"):
    """ Create a slepc spectral transformation object. """
    # Preconditioner and linear solver
    P = PETSc.PC()
    P.create()
    P.setType(ptype)
    P.setFactorSolverPackage(ppackage)
    # Krylov subspace
    K = PETSc.KSP()
    K.create()
    K.setPC(P)
    K.setType(ktype)
    # Spectral transformer
    S = SLEPc.ST()
    S.create()
    S.setKSP(K)
    S.setType(stype)
    return S


def aeigsys(a, k=6, which="SR", sigma=None, isherm=True, return_vecs=True,
            sort=True, ncv=None, etype="krylovschur", return_all_conv=False,
            st_opts_dict={}, tol=None, max_it=None):
    """ Solve a matrix using the advanced eigensystem solver

    Parameters
    ----------
        a: sparse matrix in csr format
        k: number of requested eigenpairs
        sigma: target eigenvalue
        isherm: whether problem is hermitian or not
        return_vecs: whether to return the eigenvectors
        sort: whether to sort the eigenpairs in ascending real value
        etype: SLEPc eigensolver type to use
        return_all_conv: whether to return converged eigenpairs beyond
            requested subspace size
        st_opts_dict: options to send to the eigensolver internal inverter

    Returns
    -------
        lk: eigenvalues
        vk: corresponding eigenvectors (if return_vecs == True)"""
    eigensolver = init_eigensolver(which=which, sigma=sigma, isherm=isherm,
                                   etype=etype, st_opts_dict=st_opts_dict,
                                   tol=tol, max_it=max_it)
    pa = scipy_to_petsc_csr(a)
    eigensolver.setOperators(pa)
    eigensolver.setDimensions(k, ncv)
    eigensolver.solve()
    nconv = eigensolver.getConverged()
    assert nconv >= k
    k = nconv if return_all_conv else k
    lk = np.asarray([eigensolver.getEigenvalue(i) for i in range(k)])
    lk = lk.real if isherm else lk
    if return_vecs:
        vk = np.empty((a.shape[0], k), dtype=complex)
        for i, v in enumerate(eigensolver.getInvariantSubspace()[:k]):
            vk[:, i] = v
        if sort:
            sortinds = np.argsort(lk)
            return lk[sortinds], np.asmatrix(vk[:, sortinds])
    return np.sort(lk) if sort else lk


def aeigvals(a, k=6, **kwargs):
    """ Aeigsys alias for finding eigenvalues only. """
    return aeigsys(a, k=k, return_vecs=False, **kwargs)


def aeigvecs(a, k=6, **kwargs):
    """ Aeigsys alias for finding eigenvectors only. """
    _, v = aeigsys(a, k=k, return_vecs=True, **kwargs)
    return v


def agroundstate(ham):
    """ Alias for finding lowest eigenvector only. """
    return aeigvecs(ham, k=1, which='SA')


def agroundenergy(ham):
    """ Alias for finding lowest eigenvalue only. """
    return aeigvals(ham, k=1, which='SA')[0]


def asvds(a, k=1, stype="cross", extra_vals=False, ncv=None,
          tol=None, max_it=None):
    """ Find the singular values for sparse matrix `a`.

    Parameters
    ----------
        a: sparse matrix in csr format
        k: number of requested singular values
        method: solver method to use, options ["cross", "cyclic", "lanczos",
            "trlanczos"]

    Returns
    -------
        ds: singular values """
    svd_solver = SLEPc.SVD()
    svd_solver.create()
    svd_solver.setType(stype)
    svd_solver.setDimensions(nsv=k, ncv=ncv)
    svd_solver.setTolerances(tol=tol, max_it=max_it)
    svd_solver.setOperator(scipy_to_petsc_csr(a))
    svd_solver.solve()
    nconv = svd_solver.getConverged()
    assert nconv >= k
    k = nconv if extra_vals else k
    ds = [svd_solver.getValue(i) for i in range(k)]
    svd_solver.destroy()
    return ds
