"""Interface to slepc4py for solving advanced eigenvalue problems.
"""
# TODO: FEAST / other contour solvers?
# TODO: expm, sqrtm

import numpy as np
import scipy.sparse as sp


def cache_first_call(fn):
    """Wrap a function such that it returns its first call's result,
    regardless of later calls with different arguments.
    """

    def wrapped(*args, **kwargs):
        if wrapped.result is None:
            wrapped.result = fn(*args, **kwargs)
            return wrapped.result
        else:
            return wrapped.result

    wrapped.result = None
    return wrapped


@cache_first_call
def get_default_comm():
    """Define the default communicator.
    """
    from mpi4py import MPI
    return MPI.COMM_SELF


@cache_first_call
def init_petsc_and_slepc(comm=None):
    """Make sure petsc is initialized with comm before slepc.
    """
    if comm is None:
        comm = get_default_comm()
    import petsc4py
    petsc4py.init(args=['-no_signal_handler'], comm=comm)
    from petsc4py import PETSc
    import slepc4py
    slepc4py.init(args=['-no_signal_handler'])
    from slepc4py import SLEPc
    return PETSc, SLEPc


@cache_first_call
def _get_petsc(comm=None):
    """Cache petsc module import to allow lazy start.
    """
    return init_petsc_and_slepc(comm=comm)[0]


@cache_first_call
def _get_slepc(comm=None):
    """Cache slepc module import to allow lazy start.
    """
    return init_petsc_and_slepc(comm=comm)[1]


def mpi_partition(n, comm=None):
    """
    """
    PETSc = _get_petsc(comm=comm)
    comm = PETSc.COMM_WORLD
    # Get size and position within comm
    mpi_size = comm.Get_size()
    mpi_rank = comm.Get_rank()
    # Equally partition
    block_size = n // mpi_size
    row_start = mpi_rank * block_size
    if mpi_rank != n - 1:
        row_finish = (mpi_rank + 1) * block_size
    else:
        row_finish = n
    return row_start, row_finish


def convert_to_petsc(mat, comm=None):
    """Convert a matrix to the relevant PETSc type, currently
    only supports csr, bsr, vectors and dense matrices formats.
    """
    PETSc = _get_petsc(comm=comm)
    comm = PETSc.COMM_WORLD

    # Sparse compressed row matrix
    if sp.isspmatrix_csr(mat):
        mat.sort_indices()
        if comm.Get_size() > 1:
            ri, rf = mpi_partition(mat.shape[0], comm=comm)
            csr = (mat.indptr[ri:rf + 1] - mat.indptr[ri],
                   mat.indices[mat.indptr[ri]:mat.indptr[rf]],
                   mat.data[mat.indptr[ri]:mat.indptr[rf]])
        else:
            csr = (mat.indptr, mat.indices, mat.data)
        pmat = PETSc.Mat().createAIJ(size=mat.shape, csr=csr, comm=comm)
    # Sparse block row matrix
    elif sp.isspmatrix_bsr(mat):
        mat.sort_indices()
        csr = (mat.indptr, mat.indices, mat.data)
        pmat = PETSc.Mat().createBAIJ(size=mat.shape, bsize=mat.blocksize,
                                      csr=csr, comm=comm)
    # Dense vector
    elif mat.ndim == 1:
        pmat = PETSc.Vec().createWithArray(mat, comm=comm)
    # Dense matrix
    else:
        pmat = PETSc.Mat().createDense(size=mat.shape, array=mat, comm=comm)
    pmat.assemble()
    return pmat


def new_petsc_vec(n, comm=None):
    """Create an empty complex petsc vector of size `n`.
    """
    PETSc = _get_petsc(comm=comm)
    comm = PETSc.COMM_WORLD
    a = np.empty(n, dtype=complex)
    return PETSc.Vec().createWithArray(a, comm=comm)


def _init_spectral_inverter(ptype="lu",
                            ppackage="mumps",
                            ktype="preonly",
                            stype="sinvert",
                            comm=None):
    """Create a slepc spectral transformation object with specified solver.
    """
    PETSc = _get_petsc(comm=comm)
    SLEPc = _get_slepc(comm=comm)
    comm = PETSc.COMM_WORLD
    # Preconditioner and linear solver
    P = PETSc.PC().create(comm=comm)
    P.setType(ptype)
    P.setFactorSolverPackage(ppackage)
    P.setFromOptions()
    # Krylov subspace
    K = PETSc.KSP().create(comm=comm)
    K.setPC(P)
    K.setType(ktype)
    K.setFromOptions()
    # Spectral transformer
    S = SLEPc.ST().create(comm=comm)
    S.setKSP(K)
    S.setType(stype)
    S.setFromOptions()
    return S


_WHICH_SCIPY_TO_SLEPC = {
    "LM": 'LARGEST_MAGNITUDE',
    "SM": 'SMALLEST_MAGNITUDE',
    "LR": 'LARGEST_REAL',
    "LA": 'LARGEST_REAL',
    "SR": 'SMALLEST_REAL',
    "SA": 'SMALLEST_REAL',
    "LI": 'LARGEST_IMAGINARY',
    "SI": 'SMALLEST_IMAGINARY',
    "TM": 'TARGET_MAGNITUDE',
    "TR": 'TARGET_REAL',
    "TI": 'TARGET_IMAGINARY',
}


def _which_scipy_to_slepc(which):
    SLEPc = _get_slepc()
    return getattr(SLEPc.EPS.Which, _WHICH_SCIPY_TO_SLEPC[which.upper()])


def _init_eigensolver(which='LM', sigma=None, isherm=True,
                      EPSType="krylovschur", st_opts_dict=(), tol=None,
                      max_it=None, comm=None):
    """Create an advanced eigensystem solver

    Parameters
    ----------
        sigma: target eigenvalue
        isherm: whether problem is hermitian or not

    Returns
    -------
        SLEPc solver ready to be called.
    """
    SLEPc = _get_slepc(comm=comm)
    comm = SLEPc.COMM_WORLD
    eigensolver = SLEPc.EPS().create(comm=comm)
    if sigma is not None:
        which = "TR"
        eigensolver.setST(_init_spectral_inverter(comm=comm,
                                                  **dict(st_opts_dict)))
        eigensolver.setTarget(sigma)
    eigensolver.setType(EPSType)
    eigensolver.setProblemType(SLEPc.EPS.ProblemType.HEP if isherm else
                               SLEPc.EPS.ProblemType.NHEP)
    eigensolver.setWhichEigenpairs(_which_scipy_to_slepc(which))
    eigensolver.setConvergenceTest(SLEPc.EPS.Conv.ABS)
    eigensolver.setTolerances(tol=tol, max_it=max_it)
    eigensolver.setFromOptions()
    return eigensolver


def slepc_seigsys(a, k=6, which=None, return_vecs=True, sigma=None,
                  isherm=True, ncv=None, sort=True, EPSType="krylovschur",
                  return_all_conv=False, st_opts_dict=(), tol=None,
                  max_it=None, comm=None):
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
    eigensolver = _init_eigensolver(which=which if which else 'SR',
                                    sigma=sigma,
                                    isherm=isherm,
                                    EPSType=EPSType,
                                    tol=tol,
                                    max_it=max_it,
                                    st_opts_dict=st_opts_dict,
                                    comm=comm)
    eigensolver.setOperators(convert_to_petsc(a, comm=comm))
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
    else:
        eigensolver.destroy()
        return np.sort(lk) if sort else lk


def _init_svd_solver(SVDType='cross', tol=None, max_it=None, comm=None):
    SLEPc = _get_slepc(comm=comm)
    comm = SLEPc.COMM_WORLD
    svd_solver = SLEPc.SVD().create(comm=comm)
    svd_solver.setType(SVDType)
    svd_solver.setTolerances(tol=tol, max_it=max_it)
    svd_solver.setFromOptions()
    return svd_solver


def slepc_svds(a, k=6, ncv=None, return_vecs=True, SVDType='cross',
               extra_vals=False, tol=None, max_it=None, comm=None):
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
    svd_solver = _init_svd_solver(SVDType=SVDType, tol=tol,
                                  max_it=max_it, comm=comm)
    petsc_a = convert_to_petsc(a, comm=comm)
    svd_solver.setOperator(petsc_a)
    svd_solver.setDimensions(nsv=k, ncv=ncv)
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
