"""Interface to slepc4py for solving advanced eigenvalue problems.
"""
# TODO: FEAST / other contour solvers?
# TODO: pre-sliced and/or un-constructed matrices?

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
        if self._comm != comm:
            self._result = self._comm_fn(comm=comm)
            self._comm = comm
        return self._result, comm


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

class PetscLinearOperatorContext:
    def __init__(self, lo):
        self.lo = lo

    def mult(self, _, x, y):
        y[:] = self.lo.matvec(x)


def linear_operator_2_petsc_shell(lo, comm=None):
    PETSc, comm = get_petsc(comm=comm)
    context = PetscLinearOperatorContext(lo)
    A = PETSc.Mat().createPython(lo.shape, context, comm=comm)
    A.setUp()
    return A


def slice_sparse_matrix_to_components(mat, ri, rf):
    """Slice the matrix `mat` between indices `ri` and `rf` -- for csr or bsr.
    """
    return (mat.indptr[ri:rf + 1] - mat.indptr[ri],
            mat.indices[mat.indptr[ri]:mat.indptr[rf]],
            mat.data[mat.indptr[ri]:mat.indptr[rf]])


def convert_mat_to_petsc(mat, comm=None):
    """Convert a matrix to the relevant PETSc type, currently
    only supports csr, bsr and dense matrices formats.

    Parameters
    ----------
        mat : matrix-like
            Matrix, dense or sparse.
        comm : mpi4py.MPI.Comm instance
            The mpi communicator.

    Returns
    -------
        pmat : petsc4py.PETSc.Mat
            The matrix in petsc form - only the local part if running
            across several mpi processes.
    """
    if isinstance(mat, sp.linalg.LinearOperator):
        return linear_operator_2_petsc_shell(mat, comm=comm)

    PETSc, comm = get_petsc(comm=comm)
    mpi_sz = comm.Get_size()
    pmat = PETSc.Mat()

    if mpi_sz > 1:
        pmat.create(comm=comm)
        pmat.setSizes(mat.shape)
        pmat.setFromOptions()
        pmat.setUp()
        ri, rf = pmat.getOwnershipRange()

    # Sparse block row matrix
    if sp.isspmatrix_bsr(mat):
        mat.sort_indices()
        if mpi_sz > 1:
            csr = slice_sparse_matrix_to_components(mat, ri, rf)
        else:
            csr = (mat.indptr, mat.indices, mat.data)
        pmat.createBAIJ(size=mat.shape, bsize=mat.blocksize,
                        nnz=mat.nnz, csr=csr, comm=comm)

    # Sparse compressed row matrix
    elif sp.isspmatrix_csr(mat):
        mat.sort_indices()
        if mpi_sz > 1:
            csr = slice_sparse_matrix_to_components(mat, ri, rf)
        else:
            csr = (mat.indptr, mat.indices, mat.data)
        pmat.createAIJ(size=mat.shape, nnz=mat.nnz, csr=csr, comm=comm)

    # Dense matrix
    else:
        if mpi_sz > 1:
            pmat.createDense(size=mat.shape, array=mat[ri:rf, :], comm=comm)
        else:
            pmat.createDense(size=mat.shape, array=mat, comm=comm)

    pmat.assemble()
    return pmat


def convert_vec_to_petsc(vec, comm=None):
    """Convert a vector/array to the PETSc form.

    Parameters
    ----------
        vec : vector-like
            Numpy array, will be unravelled to one dimension.
        comm : mpi4py.MPI.Comm instance
            The mpi communicator.

    Returns
    -------
        pvec : petsc4py.PETSc.Vec
            The vector in petsc form - only the local part if running
            across several mpi processes.
    """
    PETSc, comm = get_petsc(comm=comm)
    mpi_sz = comm.Get_size()
    pvec = PETSc.Vec()

    flat_vec = np.asarray(vec).reshape(-1)

    if mpi_sz > 1:
        pvec.create(comm=comm)
        pvec.setSizes(vec.size)
        pvec.setFromOptions()
        pvec.setUp()
        ri, rf = pvec.getOwnershipRange()
        pvec.createWithArray(flat_vec[ri:rf], comm=comm)
    else:
        pvec.createWithArray(flat_vec, comm=comm)

    return pvec


def new_petsc_vec(d, comm=None):
    """Create an empty complex petsc vector of size `d`.

    Parameters
    ----------
        d : int
            Dimension of vector, i.e. the global size.
        comm : mpi4py.MPI.Comm instance
            The mpi communicator.

    Returns
    -------
        pvec : petsc4py.PETSc.Vec
            An empty vector in petsc form - only the local part if running
            across several mpi processes.
    """
    PETSc, comm = get_petsc(comm=comm)
    pvec = PETSc.Vec()
    pvec.create(comm=comm)
    pvec.setSizes(d)
    pvec.setFromOptions()
    pvec.setUp()
    return pvec


def gather_petsc_array(x, comm, out_shape=None, matrix=False):
    """Gather the petsc vector/matrix `x` to a single array on the master
    process, assuming that owernership is sliced along the first dimension.

    Parameters
    ----------
        x : petsc4py.PETSc Mat or Vec
            Distributed petsc array to gather.
        comm : mpi4py.MPI.COMM
            MPI communicator
        out_shape : tuple, optional
            If not None, reshape the output array to this.
        matrix : bool, optional
            Whether to convert the array to a np.matrix.

    Returns
    -------
        np.array or np.matrix on master, None on workers (rank > 0)
    """
    # get local numpy array
    lx = x.getArray()
    ox = np.empty(2, dtype=int)
    ox[:] = x.getOwnershipRange()

    # master only
    if comm.Get_rank() == 0:

        # create total array
        ax = np.empty(x.getSize(), dtype=complex)
        # set master's portion
        ax[ox[0]:ox[1], ...] = lx

        # get ownership ranges and data from worker processes
        for i in range(1, comm.Get_size()):
            comm.Recv(ox, source=i, tag=11)

            # receive worker's part of ouput vector
            comm.Recv(ax[ox[0]:ox[1], ...], source=i, tag=42)

        if out_shape is not None:
            ax = ax.reshape(*out_shape)
        if matrix:
            ax = np.asmatrix(ax)

    # Worker only
    else:
        # send ownership range
        comm.Send(ox, dest=0, tag=11)
        # send local portion of eigenvectors as buffer
        comm.Send(lx, dest=0, tag=42)
        ax = None

    return ax


# --------------------------------------------------------------------------- #
#                               SLEPc FUNCTIONS                               #
# --------------------------------------------------------------------------- #

def _init_spectral_inverter(STType="sinvert",
                            KSPType="preonly",
                            PType="lu",
                            PFactorSolverPackage="mumps",
                            comm=None):
    """Create a slepc spectral transformation object with specified solver.
    """
    PETSc, comm = get_petsc(comm=comm)
    SLEPc, comm = get_slepc(comm=comm)
    # Preconditioner and linear solver
    P = PETSc.PC().create(comm=comm)
    P.setType(PType)
    P.setFactorSolverPackage(PFactorSolverPackage)
    P.setFromOptions()
    # Krylov subspace
    K = PETSc.KSP().create(comm=comm)
    K.setPC(P)
    K.setType(KSPType)
    K.setFromOptions()
    # Spectral transformer
    S = SLEPc.ST().create(comm=comm)
    S.setType(STType)
    S.setKSP(K)
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
    SLEPc = get_slepc()[0]
    return getattr(SLEPc.EPS.Which, _WHICH_SCIPY_TO_SLEPC[which.upper()])


def _init_eigensolver(k=6, which='LM', sigma=None, isherm=True,
                      EPSType="krylovschur", st_opts_dict=(), tol=None,
                      max_it=None, ncv=None, comm=None):
    """Create an advanced eigensystem solver

    Parameters
    ----------
    sigma :
        Target eigenvalue.
    isherm :
        Whether problem is hermitian or not.

    Returns
    -------
        SLEPc solver ready to be called.
    """
    SLEPc, comm = get_slepc(comm=comm)

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
    eigensolver.setDimensions(k, ncv)
    eigensolver.setFromOptions()
    return eigensolver


def seigsys_slepc(mat, k=6, which=None, return_vecs=True, sigma=None,
                  isherm=True, ncv=None, sort=True, EPSType=None,
                  return_all_conv=False, st_opts_dict=(), tol=None,
                  max_it=None, comm=None):
    """Solve a matrix using the advanced eigensystem solver

    Parameters
    ----------
    mat : sparse matrix in csr format
        Operator to solve.
    k : int, optional
        Number of requested eigenpairs.
    which : {"LM": "SM", "LR", "LA", "SR", "SA", "LI", "SI", "TM", "TR", "TI"}
        Which eigenpairs to target. See :func:`scipy.sparse.linalg.eigs`.
    sigma : float, optional
        Target eigenvalue, implies ``which='TR'`` if this is not set but
        ``sigma`` is.
    isherm : bool, optional
        Whether problem is hermitian or not.
    return_vecs : bool, optional
        Whether to return the eigenvectors.
    sort : bool, optional
        Whether to sort the eigenpairs in ascending real value.
    EPSType : {"krylovschur", ...}, optional
        SLEPc eigensolver type to use, see slepc4py.EPSType.
    return_all_conv : bool, optional
        Whether to return converged eigenpairs beyond requested subspace size
    st_opts_dict : dict, optional
        options to send to the eigensolver internal inverter.

    Returns
    -------
    lk : array
        The eigenvalues.
    vk : np.matrix
        Corresponding eigenvectors (if return_vecs == True)
    """
    if comm is None:
        comm = get_default_comm()

    # Need different defaults for interior eigensearch of shell matrix
    if (
            isinstance(mat, sp.linalg.LinearOperator) and
            (sigma is not None) and
            (EPSType is None) and
            st_opts_dict is ()
    ):
        # Note : probably very slow compared to converting to explicit matrix!
        EPSType = 'gd'
        st_opts_dict = {'STType': 'precond',
                        'KSPType': 'preonly',
                        'PType': 'none'}

    eigensolver = _init_eigensolver(
        k=k,
        which=("SA" if (which is None) and (sigma is None) else
               "TR" if (which is None) and (sigma is not None) else
               which),
        sigma=sigma,
        isherm=isherm,
        EPSType="krylovschur" if EPSType is None else EPSType,
        tol=tol,
        max_it=max_it,
        ncv=ncv,
        st_opts_dict=st_opts_dict,
        comm=comm,
    )

    mat = convert_mat_to_petsc(mat, comm=comm)
    eigensolver.setOperators(mat)
    eigensolver.solve()

    nconv = eigensolver.getConverged()
    k = nconv if return_all_conv else k
    if nconv < k:
        raise RuntimeError("SLEPC eigs did not find enough singular triplets, "
                           "wanted: {}, found: {}.".format(k, nconv))

    rank = comm.Get_rank()

    if rank == 0:
        lk = np.asarray([eigensolver.getEigenvalue(i) for i in range(k)])
        lk = lk.real if isherm else lk
    else:
        res = None

    if return_vecs:
        pvec = mat.getVecLeft()

        def get_vecs_local():
            for i in range(k):
                eigensolver.getEigenvector(i, pvec)
                yield gather_petsc_array(pvec, comm=comm, out_shape=(-1, 1))

        lvecs = list(get_vecs_local())
        if rank == 0:
            vk = np.concatenate(lvecs, axis=1)
            if sort:
                sortinds = np.argsort(lk)
                lk, vk = lk[sortinds], np.asmatrix(vk[:, sortinds])
            res = lk, np.asmatrix(vk)
    else:
        if rank == 0:
            res = np.sort(lk) if sort else lk

    eigensolver.destroy()
    return res


# ----------------------------------- SVD ----------------------------------- #

def _init_svd_solver(nsv=6, SVDType='cross', tol=None, max_it=None,
                     ncv=None, comm=None):
    SLEPc, comm = get_slepc(comm=comm)
    svd_solver = SLEPc.SVD().create(comm=comm)
    svd_solver.setType(SVDType)
    svd_solver.setTolerances(tol=tol, max_it=max_it)
    svd_solver.setDimensions(nsv=nsv, ncv=ncv)
    svd_solver.setFromOptions()
    return svd_solver


def svds_slepc(mat, k=6, ncv=None, return_vecs=True, SVDType='cross',
               return_all_conv=False, tol=None, max_it=None, comm=None):
    """Find the singular values for sparse matrix `a`.

    Parameters
    ----------
    mat : sparse matrix in csr format
        The matrix to solve.
    k : int
        Number of requested singular values.
    method : {"cross", "cyclic", "lanczos", "trlanczos"}
        Solver method to use.

    Returns
    -------
    (uk,) sk (, vtk,) : (np.matrix,) np.array (, np.matrix,)
        Singular values, or if return_vecs=True, the left, unitary matrix
        singular values, and and transposed right unitary matrix, such that
        ``a ~ uk @ diag(sk) @ vtk``.
    """
    if comm is None:
        comm = get_default_comm()

    mat = convert_mat_to_petsc(mat, comm=comm)

    svd_solver = _init_svd_solver(nsv=k, SVDType=SVDType, tol=tol,
                                  max_it=max_it, ncv=ncv, comm=comm)
    svd_solver.setOperator(mat)
    svd_solver.solve()

    nconv = svd_solver.getConverged()
    k = nconv if return_all_conv else k
    if nconv < k:
        raise RuntimeError("SLEPC svds did not find enough singular triplets, "
                           "wanted: {}, found: {}.".format(k, nconv))

    rank = comm.Get_rank()

    if return_vecs:

        def usv_getter():
            v, u = mat.createVecs()
            for i in range(k):
                s = svd_solver.getSingularTriplet(i, u, v)
                lu = gather_petsc_array(u, comm=comm, out_shape=(-1, 1))
                lv = gather_petsc_array(v, comm=comm, out_shape=(1, -1))
                yield lu, s, lv

        lus, sk, lvs = zip(*usv_getter())
        sk = np.asarray(sk)

        if rank == 0:
            uk = np.concatenate(lus, axis=1)
            vtk = np.concatenate(lvs, axis=0).conjugate()
            so = np.argsort(-sk)
            res = np.asmatrix(uk[:, so]), sk[so], np.asmatrix(vtk[so, :])
    else:
        if rank == 0:
            sk = np.asarray([svd_solver.getValue(i) for i in range(k)])
            res = sk[np.argsort(-sk)]

    svd_solver.destroy()
    return res if rank == 0 else None


# ------------------------ matrix multiply function ------------------------- #

def mfn_multiply_slepc(mat, vec,
                       fntype='exp',
                       MFNType='AUTO',
                       comm=None,
                       isherm=False):
    """Compute the action of ``func(mat) @ vec``.

    Parameters
    ----------
    mat : matrix-like
        Matrix to compute function action of.
    vec : vector-like
        Vector to compute matrix function action on.
    func : {'exp', 'sqrt', 'log'}, optional
        Function to use.
    MFNType : {'krylov', 'expokit'}, optional
        Method of computing the matrix function action, 'expokit' is only
        available for func='exp'.
    comm : mpi4py.MPI.Comm instance, optional
        The mpi communicator.
    isherm : bool, optional
        If `mat` is known to be hermitian, this might speed things up in
        some circumstances.

    Returns
    -------
    fvec : np.matrix
        The vector output of ``func(mat) @ vec``.
    """
    SLEPc, comm = get_slepc(comm=comm)

    mat = convert_mat_to_petsc(mat, comm=comm)
    if isherm:
        mat.setOption(mat.Option.HERMITIAN, True)
    vec = convert_vec_to_petsc(vec, comm=comm)
    out = new_petsc_vec(vec.size, comm=comm)

    if MFNType.upper() == 'AUTO':
        if vec.size > 2**16:
            MFNType = 'KRYLOV'
        else:
            MFNType = 'EXPOKIT'

    # set up the matrix function options and objects
    mfn = SLEPc.MFN().create(comm=comm)
    mfn.setType(getattr(SLEPc.MFN.Type, MFNType.upper()))
    mfn_fn = mfn.getFN()
    mfn_fn.setType(getattr(SLEPc.FN.Type, fntype.upper()))
    mfn_fn.setScale(1.0, 1.0)
    mfn.setFromOptions()

    mfn.setOperator(mat)
    # 'solve' / perform the matrix function
    mfn.solve(vec, out)

    # --> gather the (distributed) petsc vector to a numpy matrix on master
    all_out = gather_petsc_array(
        out, comm=comm, out_shape=(-1, 1), matrix=True)

    mfn.destroy()
    return all_out
