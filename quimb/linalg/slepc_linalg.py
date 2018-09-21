"""Interface to slepc4py for solving advanced eigenvalue problems.
"""

import numpy as np
import scipy.sparse as sp

import quimb as qu


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
    import os
    petsc_arch = os.environ.get("PETSC_ARCH", None)

    import petsc4py
    petsc4py.init(args=['-no_signal_handler'], arch=petsc_arch, comm=comm)
    import slepc4py
    slepc4py.init(args=['-no_signal_handler'], arch=petsc_arch)
    return petsc4py.PETSc, slepc4py.SLEPc


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
        self.real = lo.dtype in (float, np.float_)

    def mult(self, _, x, y):
        y[:] = self.lo.matvec(x[:])

    def multHermitian(self, _, x, y):
        y[:] = self.lo.rmatvec(x[:])


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
    mat : dense, sparse, LinearOperator or Lazy matrix.
        The operator to convert.
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

    # retrieve before mat is possibly built into sliced matrix
    shape = mat.shape

    pmat.create(comm=comm)
    pmat.setSizes(shape)
    pmat.setFromOptions()
    pmat.setUp()
    ri, rf = pmat.getOwnershipRange()

    # only consider the operator already sliced if owns whole
    sliced = (mpi_sz == 1)
    if isinstance(mat, qu.Lazy):
        # operator hasn't been constructed yet
        try:
            # try and and lazily construct with slicing
            mat = mat(ownership=(ri, rf))
            sliced = True
        except TypeError:
            mat = mat()

    # Sparse compressed or block row matrix
    if sp.issparse(mat):
        mat.sort_indices()

        if sliced:
            csr = (mat.indptr, mat.indices, mat.data)
        else:
            csr = slice_sparse_matrix_to_components(mat, ri, rf)

        if sp.isspmatrix_csr(mat):
            pmat.createAIJ(size=shape, nnz=mat.nnz, csr=csr, comm=comm)
        elif sp.isspmatrix_bsr(mat):
            pmat.createBAIJ(size=shape, bsize=mat.blocksize,
                            nnz=mat.nnz, csr=csr, comm=comm)

    # Dense matrix
    else:
        if mpi_sz > 1 and not sliced:
            pmat.createDense(size=shape, array=mat[ri:rf, :], comm=comm)
        else:
            pmat.createDense(size=shape, array=mat, comm=comm)

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

    # get shape before slicing
    size = max(vec.shape)

    pvec.create(comm=comm)
    pvec.setSizes(size)
    pvec.setFromOptions()
    pvec.setUp()
    ri, rf = pvec.getOwnershipRange()

    # only consider the vector already sliced if owns whole
    sliced = (mpi_sz == 1)
    if isinstance(vec, qu.Lazy):
        # vector hasn't been constructed yet
        try:
            # try and and lazily construct with slicing
            vec = vec(ownership=(ri, rf))
            sliced = True
        except TypeError:
            vec = vec()

    array = np.asarray(vec).reshape(-1)

    if not sliced:
        array = array[ri:rf]

    pvec.createWithArray(array, comm=comm)
    return pvec


def new_petsc_vec(d, comm=None):
    """Create an empty petsc vector of size `d`.

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


def gather_petsc_array(x, comm, out_shape=None):
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

    Returns
    -------
    gathered : np.array master, None on workers (rank > 0)
    """
    # get local numpy array
    lx = x.getArray()
    ox = np.empty(2, dtype=int)
    ox[:] = x.getOwnershipRange()

    # master only
    if comm.Get_rank() == 0:

        # create total array
        ax = np.empty(x.getSize(), dtype=lx.dtype)
        # set master's portion
        ax[ox[0]:ox[1], ...] = lx

        # get ownership ranges and data from worker processes
        for i in range(1, comm.Get_size()):
            comm.Recv(ox, source=i, tag=11)

            # receive worker's part of ouput vector
            comm.Recv(ax[ox[0]:ox[1], ...], source=i, tag=42)

        if out_shape is not None:
            ax = ax.reshape(*out_shape)

    # Worker only
    else:
        # send ownership range
        comm.Send(ox, dest=0, tag=11)
        # send local portion of eigenvectors as buffer
        comm.Send(lx, dest=0, tag=42)
        ax = None

    return ax


def normalize_real_part(vecs, transposed=False):
    """Take the real part of a set of vectors and normalize it. This is used
    for returning real eigenvectors even when the PETSc scalar type is complex.
    """
    k = vecs.shape[0] if transposed else vecs.shape[1]

    vecs = np.ascontiguousarray(vecs.real)
    for i in range(k):
        where = (i, Ellipsis) if transposed else (Ellipsis, i)
        qu.nmlz(vecs[where], inplace=True)

    return vecs

# --------------------------------------------------------------------------- #
#                               SLEPc FUNCTIONS                               #
# --------------------------------------------------------------------------- #


def _init_krylov_subspace(comm=None, tol=None, maxiter=None,
                          KSPType="preonly",
                          PCType="lu",
                          PCFactorSolverType="mumps",
                          ):
    """Initialise a krylov subspace and preconditioner.
    """
    PETSc, comm = get_petsc(comm=comm)
    K = PETSc.KSP().create(comm=comm)
    K.setType(KSPType)
    K.setTolerances(rtol=tol, max_it=maxiter)

    if PCType:
        PC = K.getPC()
        PC.setType(PCType)
        if PCFactorSolverType:
            PC.setFactorSolverType(PCFactorSolverType)
        PC.setFactorShift(PETSc.Mat.FactorShiftType.POSITIVE_DEFINITE)
        PC.setFromOptions()
        K.setPC(PC)

    K.setFromOptions()
    return K


def _init_spectral_inverter(STType="sinvert",
                            KSPType="preonly",
                            PCType="lu",
                            PCFactorSolverType="mumps",
                            comm=None):
    """Create a slepc spectral transformation object with specified solver.
    """
    SLEPc, comm = get_slepc(comm=comm)
    S = SLEPc.ST().create(comm=comm)
    S.setType(STType)
    # set the krylov subspace and preconditioner.
    if KSPType:
        K = _init_krylov_subspace(
            KSPType=KSPType, PCType=PCType, comm=comm,
            PCFactorSolverType=PCFactorSolverType)
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
    "ALL": 'ALL',
}


def _which_scipy_to_slepc(which):
    SLEPc = get_slepc()[0]
    return getattr(SLEPc.EPS.Which, _WHICH_SCIPY_TO_SLEPC[which.upper()])


def _init_eigensolver(k=6, which='LM', sigma=None, isherm=True, isgen=False,
                      EPSType=None, st_opts=None, tol=None, A_is_linop=False,
                      B_is_linop=False, maxiter=None, ncv=None, l_win=None,
                      comm=None):
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

    if st_opts is None:
        st_opts = {}

    if l_win is not None:
        EPSType = {'ciss': 'ciss', None: 'ciss'}[EPSType]
        which = 'ALL'
        rg = eigensolver.getRG()
        rg.setType(SLEPc.RG.Type.INTERVAL)
        rg.setIntervalEndpoints(*l_win, -0.1, 0.1)
    else:
        eigensolver.setDimensions(k, ncv)

    internal = (sigma is not None) or (l_win is not None)

    # pick right backend
    if EPSType is None:
        if (isgen and B_is_linop) or (internal and A_is_linop):
            EPSType = 'gd'
        else:
            EPSType = 'krylovschur'

    # set some preconditioning defaults for 'gd' and 'lobpcg'
    if EPSType in ('gd', 'lobpcg', 'blopex', 'primme'):
        st_opts.setdefault('STType', 'precond')
        st_opts.setdefault('KSPType', 'preonly')
        st_opts.setdefault('PCType', 'none')

    # set some preconditioning defaults for 'jd'
    elif EPSType == 'jd':
        st_opts.setdefault('STType', 'precond')
        st_opts.setdefault('KSPType', 'bcgs')
        st_opts.setdefault('PCType', 'none')

    # set the spectral inverter / preconditioner.
    if st_opts or internal:
        st = _init_spectral_inverter(comm=comm, **st_opts)
        eigensolver.setST(st)

        # NB: `setTarget` must be called *after* `setST`.
        if sigma is not None:
            which = "TR"
            eigensolver.setTarget(sigma)

            if A_is_linop:
                st.setMatMode(SLEPc.ST.MatMode.SHELL)

    _EPS_PROB_TYPES = {
        (False, False): SLEPc.EPS.ProblemType.NHEP,
        (True, False): SLEPc.EPS.ProblemType.HEP,
        (False, True): SLEPc.EPS.ProblemType.GNHEP,
        (True, True): SLEPc.EPS.ProblemType.GHEP,
    }

    eigensolver.setType(EPSType)
    eigensolver.setProblemType(_EPS_PROB_TYPES[(isherm, isgen)])
    eigensolver.setWhichEigenpairs(_which_scipy_to_slepc(which))
    eigensolver.setConvergenceTest(SLEPc.EPS.Conv.REL)
    eigensolver.setTolerances(tol=tol, max_it=maxiter)
    eigensolver.setFromOptions()
    return eigensolver


def eigs_slepc(A, k, *, B=None, which=None, sigma=None, isherm=True, P=None,
               v0=None, ncv=None, return_vecs=True, sort=True, EPSType=None,
               return_all_conv=False, st_opts=None, tol=None, maxiter=None,
               l_win=None, comm=None):
    """Solve a matrix using the advanced eigensystem solver

    Parameters
    ----------
    A : dense-matrix, sparse-matrix, LinearOperator or callable
        Operator to solve.
    k : int, optional
        Number of requested eigenpairs.
    B : dense-matrix, sparse-matrix, LinearOperator or callable
        The RHS operator defining a generalized eigenproblem.
    which : {"LM": "SM", "LR", "LA", "SR", "SA", "LI", "SI", "TM", "TR", "TI"}
        Which eigenpairs to target. See :func:`scipy.sparse.linalg.eigs`.
    sigma : float, optional
        Target eigenvalue, implies ``which='TR'`` if this is not set but
        ``sigma`` is.
    isherm : bool, optional
        Whether problem is hermitian or not.
    P : dense-matrix, sparse-matrix, LinearOperator or callable
        Perform the eigensolve in the subspace defined by this projector.
    v0 : 1D-array like, optional
        Initial iteration vector, e.g., informed guess at eigenvector.
    ncv : int, optional
        Subspace size, defaults to ``min(20, 2 * k)``.
    return_vecs : bool, optional
        Whether to return the eigenvectors.
    sort : bool, optional
        Whether to sort the eigenpairs in ascending real value.
    EPSType : {"krylovschur", 'gd', 'lobpcg', 'jd', 'ciss'}, optional
        SLEPc eigensolver type to use, see slepc4py.EPSType.
    return_all_conv : bool, optional
        Whether to return converged eigenpairs beyond requested subspace size
    st_opts : dict, optional
        options to send to the eigensolver internal inverter.
    tol : float, optional
        Tolerance.
    maxiter : int, optional
        Maximum number of iterations.
    comm : mpi4py communicator, optional
        MPI communicator, defaults to ``COMM_SELF`` for a single process solve.

    Returns
    -------
    lk : (k,) array
        The eigenvalues.
    vk : (m, k) array
        Corresponding eigenvectors (if ``return_vecs=True``).
    """
    if comm is None:
        comm = get_default_comm()

    A_is_linop = isinstance(A, sp.linalg.LinearOperator)
    B_is_linop = isinstance(B, sp.linalg.LinearOperator)
    isgen = B is not None

    eigensolver = _init_eigensolver(
        which=("SA" if (which is None) and (sigma is None) else
               "TR" if (which is None) and (sigma is not None) else which),
        EPSType=EPSType, k=k, sigma=sigma, isherm=isherm, tol=tol, ncv=ncv,
        maxiter=maxiter, st_opts=st_opts, comm=comm, l_win=l_win, isgen=isgen,
        A_is_linop=A_is_linop, B_is_linop=B_is_linop)

    # set up the initial operators and solver
    pA = convert_mat_to_petsc(A, comm=comm)
    pB = convert_mat_to_petsc(B, comm=comm) if isgen else None

    if P is not None:
        pP = convert_mat_to_petsc(P, comm=comm)
        pA = pP.transposeMatMult(pA.matMult(pP))

    eigensolver.setOperators(pA, pB)
    if v0 is not None:
        eigensolver.setInitialSpace(convert_vec_to_petsc(v0, comm=comm))
    eigensolver.solve()

    # work out how many eigenpairs to retrieve
    nconv = eigensolver.getConverged()
    k = nconv if (return_all_conv or l_win is not None) else k
    if nconv < k:
        raise RuntimeError("SLEPC eigs did not find enough eigenpairs, "
                           "wanted: {}, found: {}.".format(k, nconv))

    # get eigenvalues
    rank = comm.Get_rank()
    if rank == 0:
        lk = np.asarray([eigensolver.getEigenvalue(i) for i in range(k)])
        lk = lk.real if isherm else lk
    else:
        res = None

    # gather eigenvectors
    if return_vecs:
        pvec = pA.getVecLeft()

        def get_vecs_local():
            for i in range(k):
                eigensolver.getEigenvector(i, pvec)

                if P is not None:
                    pvecP = pP.getVecLeft()
                    pP.mult(pvec, pvecP)

                yield gather_petsc_array(
                    pvecP if P is not None else pvec,
                    comm=comm, out_shape=(-1, 1))

        lvecs = tuple(get_vecs_local())
        if rank == 0:
            vk = np.concatenate(lvecs, axis=1)
            if sort:
                sortinds = np.argsort(lk)
                lk, vk = lk[sortinds], vk[:, sortinds]

            # check if input matrix was real -> use real output when
            #   petsc compiled with complex scalars and thus outputs complex
            convert = (isherm and np.issubdtype(A.dtype, np.floating) and
                       np.issubdtype(vk.dtype, np.complexfloating))
            if convert:
                vk = normalize_real_part(vk)

            res = lk, qu.qarray(vk)
    elif rank == 0:
        res = np.sort(lk) if sort else lk

    eigensolver.destroy()
    return res


# ----------------------------------- SVD ----------------------------------- #

def _init_svd_solver(nsv=6, SVDType='cross', tol=None, maxiter=None,
                     ncv=None, comm=None):
    SLEPc, comm = get_slepc(comm=comm)
    svd_solver = SLEPc.SVD().create(comm=comm)
    svd_solver.setType(SVDType)
    svd_solver.setTolerances(tol=tol, max_it=maxiter)
    svd_solver.setDimensions(nsv=nsv, ncv=ncv)
    svd_solver.setFromOptions()
    return svd_solver


def svds_slepc(A, k=6, ncv=None, return_vecs=True, SVDType='cross',
               return_all_conv=False, tol=None, maxiter=None, comm=None):
    """Find the singular values for sparse matrix `a`.

    Parameters
    ----------
    A : sparse matrix in csr format
        The matrix to solve.
    k : int
        Number of requested singular values.
    method : {"cross", "cyclic", "lanczos", "trlanczos"}
        Solver method to use.

    Returns
    -------
    U : (m, k) array
        Left singular vectors (if ``return_vecs=True``) as columns.
    s : (k,) array
        Singular values.
    VH : (k, n) array
        Right singular vectors (if ``return_vecs=True``) as rows.
    """
    if comm is None:
        comm = get_default_comm()

    pA = convert_mat_to_petsc(A, comm=comm)

    svd_solver = _init_svd_solver(nsv=k, SVDType=SVDType, tol=tol,
                                  maxiter=maxiter, ncv=ncv, comm=comm)
    svd_solver.setOperator(pA)
    svd_solver.solve()

    nconv = svd_solver.getConverged()
    k = nconv if return_all_conv else k
    if nconv < k:
        raise RuntimeError("SLEPC svds did not find enough singular triplets, "
                           "wanted: {}, found: {}.".format(k, nconv))

    rank = comm.Get_rank()

    if return_vecs:
        def usv_getter():
            v, u = pA.createVecs()
            for i in range(k):
                s = svd_solver.getSingularTriplet(i, u, v)
                lu = gather_petsc_array(u, comm=comm, out_shape=(-1, 1))
                lv = gather_petsc_array(v, comm=comm, out_shape=(1, -1))
                yield lu, s, lv

        lus, sk, lvs = zip(*usv_getter())
        sk = np.asarray(sk)

        if rank == 0:
            uk = qu.qarray(np.concatenate(lus, axis=1))
            vtk = qu.qarray(np.concatenate(lvs, axis=0).conjugate())

            # # check if input matrix was real -> use real output when
            # #   petsc compiled with complex scalars and thus outputs complex
            # convert = (np.issubdtype(A.dtype, np.floating) and
            #            np.issubdtype(uk.dtype, np.complexfloating))
            # if convert:
            #     uk = normalize_real_part(uk)
            #     vtk = normalize_real_part(uk, transposed=True)

            res = uk, sk, vtk
    else:
        res = np.asarray([svd_solver.getValue(i) for i in range(k)])

    comm.Barrier()
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
    mat : operator
        Operator to compute function action of.
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
    fvec : array
        The vector output of ``func(mat) @ vec``.
    """
    SLEPc, comm = get_slepc(comm=comm)

    mat = convert_mat_to_petsc(mat, comm=comm)

    if isherm:
        mat.setOption(mat.Option.HERMITIAN, True)

    vec = convert_vec_to_petsc(vec, comm=comm)
    out = new_petsc_vec(vec.size, comm=comm)

    if MFNType.upper() == 'AUTO':
        if (fntype == 'exp') and (vec.size <= 2**16):
            MFNType = 'EXPOKIT'
        else:
            MFNType = 'KRYLOV'

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
    all_out = gather_petsc_array(out, comm=comm, out_shape=(-1, 1))

    comm.Barrier()
    mfn.destroy()
    return all_out


# -------------------- solve linear system of equations --------------------- #

def lookup_ksp_error(i):
    """Look up PETSc error to print when raising after not converging.
    """
    PETSc = get_petsc()
    _KSP_DIVERGED_REASONS = {i: error for error, i in
                             PETSc.KSP.ConvergedReason.__dict__.items()
                             if isinstance(i, int)}
    return _KSP_DIVERGED_REASONS[i]


def ssolve_slepc(A, y, isherm=True, comm=None, maxiter=None, tol=None,
                 KSPType='preonly',
                 PCType='lu',
                 PCFactorSolverType="mumps",
                 ):
    if comm is None:
        comm = get_default_comm()
    A = convert_mat_to_petsc(A, comm=comm)
    if isherm:
        A.setOption(A.Option.HERMITIAN, isherm)
    x = A.createVecRight()
    out_shape = y.shape
    y = convert_vec_to_petsc(y, comm=comm)

    ksp = _init_krylov_subspace(
        KSPType=KSPType, PCType=PCType, comm=comm, maxiter=maxiter, tol=tol,
        PCFactorSolverType=PCFactorSolverType)

    ksp.setOperators(A)
    ksp.solve(y, x)

    converged_reason = ksp.getConvergedReason()
    if converged_reason < 0:
        raise RuntimeError("PETSc KSP solve did not converge, reason: {}"
                           "".format(lookup_ksp_error(converged_reason)))

    x = gather_petsc_array(x, comm=comm, out_shape=out_shape)
    comm.Barrier()
    ksp.destroy()
    return x
