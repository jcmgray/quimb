"""

"""
# TODO: get eigenvectors
# TODO: delete solver or keep and extend
# TODO: FEAST / other solvers
from petsc4py import PETSc
from slepc4py import SLEPc
import numpy as np


def scipy_to_petsc_csr(a):
    b = PETSc.Mat().createAIJ(size=a.shape, csr=(a.indptr, a.indices, a.data))
    return b


def internal_eigsys_solver(k=6, sigma=0.0, tol=None):
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
    S.setShift(sigma)
    # EIGENSOLVER
    E = SLEPc.EPS()
    E.create()
    E.setST(S)
    E.setProblemType(SLEPc.EPS.ProblemType.HEP)
    E.setType('krylovschur')
    E.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_MAGNITUDE)
    E.setTarget(sigma)
    E.setDimensions(k)
    E.setConvergenceTest(SLEPc.EPS.Conv.ABS)
    E.setTolerances(tol=tol, max_it=None)
    return E


def internal_eigvals(h, k=6, sigma=0.0, tol=None, extra_evals=False):
    A = scipy_to_petsc_csr(h)
    E = internal_eigsys_solver(k=k, sigma=sigma, tol=tol)
    E.setOperators(A)
    E.solve()
    nconv = E.getConverged()
    assert nconv >= k
    k = nconv if extra_evals else k
    l = np.asarray([E.getEigenvalue(i).real for i in range(k)]).real
    l = np.sort(l)
    E.destroy()
    return l
