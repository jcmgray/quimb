"""Interface to slepc4py for solving advanced eigenvalue problems.
"""
# TODO: FEAST / other contour solvers?
# TODO: expm, sqrtm

import numpy as np
import scipy.sparse as sp


# --------------------------------------------------------------------------- #
#                          LAZY LOAD MPI/PETSc/SLEPc                          #
# --------------------------------------------------------------------------- #


def get_default_comm():
    """Define the default communicator.
    """
    from mpi4py import MPI
    return MPI.COMM_SELF


class CacheOnComm(object):
    """
    """

    def __init__(self, comm_fn):
        self._comm = '__UNINITIALIZED__'
        self._comm_fn = comm_fn

    def __call__(self, comm=None):
        # resolve default comm
        if comm is None:
            comm = get_default_comm()
        # first call or called with different comm
        if self._comm is not comm:
            self._result = self._comm_fn(comm=comm)
            self._comm = comm
        return self._result


@CacheOnComm
def init_petsc_and_slepc(comm=None):
    """Make sure petsc is initialized with comm before slepc.
    """
    import petsc4py
    petsc4py.init(args=['-no_signal_handler'], comm=comm)
    from petsc4py import PETSc
    import slepc4py
    slepc4py.init(args=['-no_signal_handler'])
    from slepc4py import SLEPc
    return PETSc, SLEPc


@CacheOnComm
def get_petsc(comm=None):
    """Cache petsc module import to allow lazy start.
    """
    return init_petsc_and_slepc(comm=comm)[0]


@CacheOnComm
def get_slepc(comm=None):
    """Cache slepc module import to allow lazy start.
    """
    return init_petsc_and_slepc(comm=comm)[1]


# --------------------------------------------------------------------------- #
#                               PETSc FUNCTIONS                               #
# --------------------------------------------------------------------------- #


def convert_to_petsc(mat, comm=None):
    """Convert a matrix to the relevant PETSc type, currently
    only supports csr, bsr, vectors and dense matrices formats.
    """
    # TODO: split kwarg, --> assume matrix already sliced
    # TODO: bsr, dense, vec

    PETSc = get_petsc(comm=comm)
    comm = PETSc.COMM_WORLD
    mpi_sz = comm.Get_size()
    pmat = PETSc.Mat()

    if mpi_sz > 1:
        pmat.create(comm=comm)
        pmat.setSizes(mat.shape)
        pmat.setFromOptions()
        pmat.setUp()
        ri, rf = pmat.getOwnershipRange()

    # Sparse compressed row matrix
    if sp.isspmatrix_csr(mat):
        mat.sort_indices()
        if mpi_sz > 1:
            csr = (mat.indptr[ri:rf + 1] - mat.indptr[ri],
                   mat.indices[mat.indptr[ri]:mat.indptr[rf]],
                   mat.data[mat.indptr[ri]:mat.indptr[rf]])
        else:
            csr = (mat.indptr, mat.indices, mat.data)
        pmat.createAIJ(size=mat.shape, nnz=mat.nnz, csr=csr, comm=comm)

    # Sparse block row matrix
    elif sp.isspmatrix_bsr(mat):
        mat.sort_indices()
        if mpi_sz > 1:
            csr = (mat.indptr[ri:rf + 1] - mat.indptr[ri],
                   mat.indices[mat.indptr[ri]:mat.indptr[rf]],
                   mat.data[mat.indptr[ri]:mat.indptr[rf]])
        else:
            csr = (mat.indptr, mat.indices, mat.data)
        pmat.createBAIJ(size=mat.shape, bsize=mat.blocksize,
                        nnz=mat.nnz, csr=csr, comm=comm)

    # Dense matrix
    else:
        if mpi_sz > 1:
            pmat.createDense(size=mat.shape, array=mat[ri:rf, :], comm=comm)
        else:
            pmat.createDense(size=mat.shape, array=mat, comm=comm)

    pmat.assemble()
    return pmat


def new_petsc_vec(n, comm=None):
    """Create an empty complex petsc vector of size `n`.
    """
    PETSc = get_petsc(comm=comm)
    comm = PETSc.COMM_WORLD
    a = np.empty(n, dtype=complex)
    return PETSc.Vec().createWithArray(a, comm=comm)


# --------------------------------------------------------------------------- #
#                               SLEPc FUNCTIONS                               #
# --------------------------------------------------------------------------- #

def _init_spectral_inverter(ptype="lu",
                            ppackage="mumps",
                            ktype="preonly",
                            stype="sinvert",
                            comm=None):
    """Create a slepc spectral transformation object with specified solver.
    """
    PETSc = get_petsc(comm=comm)
    SLEPc = get_slepc(comm=comm)
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
    SLEPc = get_slepc()
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
    SLEPc = get_slepc(comm=comm)
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
    if comm is None:
        comm = get_default_comm()

    eigensolver = _init_eigensolver(
        which=("SA" if which is None and sigma is None else
               "TR" if which is None and sigma is not None else
               which),
        sigma=sigma,
        isherm=isherm,
        EPSType=EPSType,
        tol=tol,
        max_it=max_it,
        st_opts_dict=st_opts_dict,
        comm=comm)

    pa = convert_to_petsc(a, comm=comm)
    ri, rf = pa.getOwnershipRange()

    eigensolver.setOperators(pa)
    eigensolver.setDimensions(k, ncv)
    eigensolver.solve()
    nconv = eigensolver.getConverged()
    assert nconv >= k
    k = nconv if return_all_conv else k

    rank = comm.Get_rank()

    if return_vecs:
        vec, _ = pa.getVecs()
        vk = np.empty((rf - ri, k), dtype=complex)
        for i in range(k):
            eigensolver.getEigenvector(i, vec)
            vk[:, i] = vec.getArray()

        # Master only
        if rank == 0:
            # pre-allocate array for whole eigenvectors and set local data
            nvk = np.empty((a.shape[0], k), dtype=complex)
            nvk[ri:rf, :] = vk
            # get ownership ranges and data from worker processes
            for i in range(1, comm.Get_size()):
                ji, jf = comm.recv(source=i, tag=11)
                comm.Recv(nvk[ji:jf, :], source=i, tag=42)
            vk = nvk

        # Worker only
        else:
            # send ownership range
            comm.send((ri, rf), dest=0, tag=11)
            # send local portion of eigenvectors as buffer
            comm.Send(vk, dest=0, tag=42)

    if rank != 0:
        eigensolver.destroy()
        return None

    lk = np.asarray([eigensolver.getEigenvalue(i) for i in range(k)])
    lk = lk.real if isherm else lk

    if return_vecs:
        if sort:
            sortinds = np.argsort(lk)
            lk, vk = lk[sortinds], np.asmatrix(vk[:, sortinds])
        res = lk, np.asmatrix(vk)
    else:
        res = np.sort(lk) if sort else lk

    eigensolver.destroy()
    return res


# ----------------------------------- SVD ----------------------------------- #

def _init_svd_solver(SVDType='cross', tol=None, max_it=None, comm=None):
    SLEPc = get_slepc(comm=comm)
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
    if comm is None:
        comm = get_default_comm()

    svd_solver = _init_svd_solver(SVDType=SVDType, tol=tol,
                                  max_it=max_it, comm=comm)
    petsc_a = convert_to_petsc(a, comm=comm)
    svd_solver.setOperator(petsc_a)
    svd_solver.setDimensions(nsv=k, ncv=ncv)
    svd_solver.solve()

    if comm and comm.Get_rank() > 0:
        svd_solver.destroy()
        return

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
