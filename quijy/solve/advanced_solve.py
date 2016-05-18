"""
Interface to slepc4py for solving advanced eigenvalue problems.
"""
# TODO: documentation
# TODO: get eigenvectors
# TODO: mimick seivals interface
# TODO: delete solver or keep and extend
# TODO: FEAST / other solvers
from petsc4py import PETSc
from slepc4py import SLEPc
import numpy as np


def scipy_to_petsc_csr(a):
    """ Convert a scipy.sp_csrmatrix to PETSc csr matrix. """
    return PETSc.Mat().createAIJ(size=a.shape,
                                 csr=(a.indptr, a.indices, a.data))


def init_eigensolver(which="LM", sigma=None, isherm=True, extra_evals=False):
    """ Create an advanced eigensystem solver

    Parameters
    ----------
        sigma: target eigenvalue
        isherm: whether problem is hermitian or not
        extra_evals: whether to return converged eigenpairs beyond requested
            subspace size

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
        "SR": SLEPc.EPS.Which.SMALLEST_REAL,
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
        eigensolver.setST(init_spectral_inverter())
        eigensolver.setTarget(sigma)
    eigensolver.setType('krylovschur')
    eigensolver.setProblemType(slepc_isherm[isherm])
    eigensolver.setWhichEigenpairs(scipy_to_slepc_which[which])
    eigensolver.setConvergenceTest(SLEPc.EPS.Conv.ABS)
    return eigensolver


def init_spectral_inverter():
    """ Create a slepc spectral transformation object. """
    # LINEAR SOLVER AND PRECONDITIONER
    P = PETSc.PC()
    P.create()
    P.setType('lu')
    P.setFactorSolverPackage('mumps')
    K = PETSc.KSP()
    K.create()
    K.setPC(P)
    K.setType('preonly')
    # SPECTRAL TRANSFORMER
    S = SLEPc.ST()
    S.create()
    S.setKSP(K)
    S.setType('sinvert')
    return S


def aeigsys(a, k=6, which="LM", sigma=None, isherm=True,
            extra_evals=False):
    """ Solve a matrix using the advanced eigensystem solver

    Parameters
    ----------
        a: sparse matrix in csr format
        k: number of requested eigenpairs
        sigma: target eigenvalue
        isherm: whether problem is hermitian or not
        extra_evals: whether to return converged eigenpairs beyond requested
            subspace size

    Returns
    -------
        lk: eigenvalues """
    eigensolver = init_eigensolver(which=which, sigma=sigma, isherm=isherm,
                                   extra_evals=extra_evals)
    eigensolver.setOperators(scipy_to_petsc_csr(a))
    eigensolver.setDimensions(k)
    eigensolver.solve()
    nconv = eigensolver.getConverged()
    assert nconv >= k
    k = nconv if extra_evals else k
    l = np.asarray([eigensolver.getEigenvalue(i).real for i in range(k)])
    l = np.sort(l)
    eigensolver.destroy()
    return l


def asvds(a, k=1, method='cross', extra_vals=False):
    """ Find the singular values for sparse matrix `a`.

    Parameters
    ----------
        a: sparse matrix in csr format
        k: number of requested singular values
        method: solver method to use, options ['cross', 'cyclic', 'lanczos',
            'trlanczos']

    Returns
    -------
        ds: singular values """
    svd_solver = SLEPc.SVD()
    svd_solver.create()
    svd_solver.setType('cross')
    svd_solver.setDimensions(k)
    svd_solver.setOperator(scipy_to_petsc_csr(a))
    svd_solver.solve()
    nconv = svd_solver.getConverged()
    assert nconv >= k
    k = nconv if extra_vals else k
    ds = [svd_solver.getValue(i) for i in range(k)]
    svd_solver.destroy()
    return ds
