"""Interface to slepc4py for solving advanced eigenvalue problems.
"""
# TODO: fix mpi abort errors with multiprocessing
# TODO: set number of processes (using mpi4py?)
# TODO: delete solver or keep and extend
# TODO: FEAST / other contour solvers?
# TODO: exponential, sqrt etc.
# TODO: region for eps in middle, both ciss and normal
# TODO: mumps set icntrl(14): catch error infog(1)=-9 and resatrt
# TODO: cache storage

import petsc4py
import slepc4py
from petsc4py import PETSc
from slepc4py import SLEPc
import numpy as np
import scipy.sparse as sp

petsc4py.init()
slepc4py.init()


def convert_to_petsc(mat):
    """Convert a matrix to the relevant PETSc type, currently
    only supports csr, bsr, vectors and dense matrices formats.
    """
    if sp.isspmatrix_csr(mat):
        mat.sort_indices()
        csr = (mat.indptr, mat.indices, mat.data)
        pmat = PETSc.Mat().createAIJ(size=mat.shape, csr=csr)
    elif sp.isspmatrix_bsr(mat):
        mat.sort_indices()
        csr = (mat.indptr, mat.indices, mat.data)
        pmat = PETSc.Mat().createBAIJ(size=mat.shape, bsize=mat.blocksize,
                                      csr=csr)
    elif mat.ndim == 1:
        pmat = PETSc.Vec().createWithArray(mat)
    else:
        pmat = PETSc.Mat().createDense(size=mat.shape, array=mat)
    return pmat


def new_petsc_vec(n):
    """Create an empty complex petsc vector of size `n`.
    """
    a = np.empty(n, dtype=complex)
    return PETSc.Vec().createWithArray(a)


def _init_spectral_inverter(ptype="lu", ppackage="mumps", ktype="preonly",
                            stype="sinvert"):
    """Create a slepc spectral transformation object with specified solver.
    """
    # Preconditioner and linear solver
    P = PETSc.PC().create()
    P.setType(ptype)
    P.setFactorSolverPackage(ppackage)
    P.setFromOptions()
    # Krylov subspace
    K = PETSc.KSP().create()
    K.setPC(P)
    K.setType(ktype)
    K.setFromOptions()
    # Spectral transformer
    S = SLEPc.ST().create()
    S.setKSP(K)
    S.setType(stype)
    S.setFromOptions()
    return S


_SCIPY_TO_SLEPC_WHICH = {
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


def _init_eigensolver(which='LM', sigma=None, isherm=True,
                      EPSType="krylovschur", st_opts_dict=None, tol=None,
                      max_it=None):
    """Create an advanced eigensystem solver

    Parameters
    ----------
        sigma: target eigenvalue
        isherm: whether problem is hermitian or not

    Returns
    -------
        SLEPc solver ready to be called.
    """
    if st_opts_dict is None:
        st_opts_dict = dict()
    slepc_isherm = {True: SLEPc.EPS.ProblemType.HEP,
                    False: SLEPc.EPS.ProblemType.NHEP}
    eigensolver = SLEPc.EPS().create()
    if sigma is not None:
        which = "TR"
        eigensolver.setST(_init_spectral_inverter(**st_opts_dict))
        eigensolver.setTarget(sigma)
    eigensolver.setType(EPSType)
    eigensolver.setProblemType(slepc_isherm[isherm])
    eigensolver.setWhichEigenpairs(_SCIPY_TO_SLEPC_WHICH[which.upper()])
    eigensolver.setConvergenceTest(SLEPc.EPS.Conv.ABS)
    eigensolver.setTolerances(tol=tol, max_it=max_it)
    eigensolver.setFromOptions()
    return eigensolver


def slepc_seigsys(a, k=6, which=None, return_vecs=True, sigma=None,
                  isherm=True, ncv=None, sort=True, EPSType="krylovschur",
                  return_all_conv=False, st_opts_dict=None, tol=None,
                  max_it=None):
    """Solve a matrix using the advanced eigensystem solver

    Parameters
    ----------
        a: sparse matrix in csr format
        k: number of requested eigenpairs
        sigma: target eigenvalue
        isherm: whether problem is hermitian or not
        return_vecs: whether to return the eigenvectors
        sort: whether to sort the eigenpairs in ascending real value
        EPSType: SLEPc eigensolver type to use
        return_all_conv: whether to return converged eigenpairs beyond
            requested subspace size
        st_opts_dict: options to send to the eigensolver internal inverter

    Returns
    -------
        lk: eigenvalues
        vk: corresponding eigenvectors (if return_vecs == True)
    """
    if st_opts_dict is None:
        st_opts_dict = dict()
    eps_settings = {
        'which': 'SR' if which is None else which,
        'sigma': sigma,
        'isherm': isherm,
        'EPSType': EPSType,
        'tol': tol,
        'max_it': max_it,
    }
    eigensolver = _init_eigensolver(**eps_settings, **st_opts_dict)
    eigensolver.setOperators(convert_to_petsc(a))
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
        eigensolver.destroy()
        if sort:
            sortinds = np.argsort(lk)
            lk, vk = lk[sortinds], np.asmatrix(vk[:, sortinds])
        return lk, vk
    eigensolver.destroy()
    return np.sort(lk) if sort else lk


def slepc_svds(a, k=6, ncv=None, return_vecs=True,
               SVDType="cross", extra_vals=False, tol=None, max_it=None):
    """Find the singular values for sparse matrix `a`.

    Parameters
    ----------
        a: sparse matrix in csr format
        k: number of requested singular values
        method: solver method to use, options ["cross", "cyclic", "lanczos",
            "trlanczos"]

    Returns
    -------
        sk: singular values
    """
    svd_solver = SLEPc.SVD().create()
    svd_solver.setType(SVDType)
    svd_solver.setDimensions(nsv=k, ncv=ncv)
    svd_solver.setTolerances(tol=tol, max_it=max_it)
    petsc_a = convert_to_petsc(a)
    svd_solver.setOperator(petsc_a)
    svd_solver.setFromOptions()
    svd_solver.solve()

    nconv = svd_solver.getConverged()
    assert nconv >= k
    k = nconv if extra_vals else k

    if return_vecs:
        sk = np.empty(k, dtype=float)
        uk = np.empty((a.shape[0], k), dtype=complex)
        vtk = np.empty((k, a.shape[1]), dtype=complex)

        u, v = petsc_a.getVecs()

        for i in range(k):
            sk[i] = svd_solver.getSingularTriplet(i, u, v)
            uk[:, i] = u.getArray()
            vtk[i, :] = v.getArray().conjugate()
            so = np.argsort(-sk)
        svd_solver.destroy()
        return np.asmatrix(uk[:, so]), sk[so], np.asmatrix(vtk[so, :])
    else:
        sk = np.asarray([svd_solver.getValue(i) for i in range(k)])
        svd_solver.destroy()
        return sk[np.argsort(-sk)]
