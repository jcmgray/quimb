"""Use lanczos tri-diagonalization to approximate the spectrum of any operator
which has an efficient represenation of its linear action on a vector.
"""
# TODO: error estimates
# TODO: more advanced tri-diagonalization method?
# TODO: tol and/or max number of steps

import functools
from math import sqrt, log2, exp

import numpy as np
import scipy.linalg as scla
import scipy.sparse.linalg as spla
try:
    # opt_einsum is highly recommended as until numpy 1.14 einsum contractions
    # do not use BLAS.
    import opt_einsum
    contract = opt_einsum.contract

    def contract_path(*args, optimize='optimal', memory_limit=-1, **kwargs):
        return opt_einsum.contract_path(
            *args, path=optimize, memory_limit=memory_limit, **kwargs)
except ImportError:
    def contract(*args, optimize='optimal', **kwargs):
        return np.einsum(
            *args, optimize=optimize, **kwargs)

    def contract_path(*args, optimize='optimal', memory_limit=-1, **kwargs):
        return np.einsum_path(
            *args, optimize=(optimize, memory_limit), **kwargs)
from ..accel import prod, vdot
from ..utils import int2tup


class HuskArray(np.ndarray):
    """Just an ndarray with only shape defined, so as to allow caching on shape
    alone.
    """

    def __init__(self, shape):
        self.shape = shape


# --------------------------------------------------------------------------- #
#                   'Lazy' represenation tensor contractions                  #
# --------------------------------------------------------------------------- #

@functools.lru_cache(128)
def get_cntrct_inds_ptr_dot(ndim_ab, sysa, matmat=False):
    """Find the correct integer contraction labels for ``lazy_ptr_dot``.

    Parameters
    ----------
    ndim_ab : int
        The total number of subsystems (dimensions) in 'ab'.
    sysa : int or sequence of int, optional
            Index(es) of the 'a' subsystem(s) to keep.
    matmat : bool, optional
        Whether to output indices corresponding to a matrix-vector or
        matrix-matrix opertion.

    Returns
    -------
    inds_a_ket : sequence of int
        The tensor index labels for the ket on subsystem 'a'.
    inds_ab_bra : sequence of int
        The tensor index labels for the bra on subsystems 'ab'.
    inds_ab_ket : sequence of int
        The tensor index labels for the ket on subsystems 'ab'.
    """
    inds_a_ket = []
    inds_ab_bra = []
    inds_ab_ket = []

    upper_inds = iter(range(ndim_ab, 2 * ndim_ab))

    for i in range(ndim_ab):
        if i in sysa:
            inds_a_ket.append(i)
            inds_ab_bra.append(i)
            inds_ab_ket.append(next(upper_inds))
        else:
            inds_ab_bra.append(i)
            inds_ab_ket.append(i)

    if matmat:
        inds_a_ket.append(2 * ndim_ab)

    return tuple(inds_a_ket), tuple(inds_ab_bra), tuple(inds_ab_ket)


@functools.lru_cache(128)
def prepare_lazy_ptr_dot(psi_a_shape, dims=None, sysa=0):
    """Pre-calculate the arrays and indexes etc for ``lazy_ptr_dot``.
    """
    mat_size = psi_a_shape[1] if len(psi_a_shape) > 1 else 1

    # convert to tuple so can always cache
    sysa = int2tup(sysa)

    ndim_ab = len(dims)
    inds_a_ket, inds_ab_bra, inds_ab_ket = get_cntrct_inds_ptr_dot(
        ndim_ab, sysa, matmat=mat_size > 1)

    dims_a = tuple(d for i, d in enumerate(dims) if i in sysa)
    if mat_size > 1:
        dims_a = dims_a + (mat_size,)

    return dims_a, inds_a_ket, inds_ab_bra, inds_ab_ket


@functools.lru_cache(128)
def get_path_lazy_ptr_dot(psi_ab_tensor_shape, psi_a_tensor_shape,
                          inds_a_ket, inds_ab_bra, inds_ab_ket):
    return contract_path(
        HuskArray(psi_a_tensor_shape), inds_a_ket,
        HuskArray(psi_ab_tensor_shape), inds_ab_bra,
        HuskArray(psi_ab_tensor_shape), inds_ab_ket,
        optimize='optimal',
        memory_limit=-1,
    )[0]


def do_lazy_ptr_dot(psi_ab_tensor, psi_a_tensor,
                    inds_a_ket, inds_ab_bra, inds_ab_ket,
                    path, out=None):
    """Perform ``lazy_ptr_dot`` with the pre-calculated indexes etc from
    ``prepare_lazy_ptr_dot``.
    """
    if out is None:
        out = np.empty_like(psi_a_tensor)

    return contract(
        psi_a_tensor, inds_a_ket,
        psi_ab_tensor.conjugate(), inds_ab_bra,
        psi_ab_tensor, inds_ab_ket,
        optimize=path,
        out=out,
    )


def lazy_ptr_dot(psi_ab, psi_a, dims=None, sysa=0, out=None):
    r"""Perform the 'lazy' evalution of ``ptr(psi_ab, ...) @ psi_a``,
    that is, contract the tensor diagram in an efficient way that does not
    necessarily construct the explicit reduced density matrix. In tensor
    diagram notation:
    ``
      ( | )
    +-------+
    | psi_a |   ______
    +_______+  /      \
       a|      |b     |
    +-------------+   |
    |  psi_ab.H   |   |
    +_____________+   |
                      |
    +-------------+   |
    |   psi_ab    |   |
    +_____________+   |
       a|      |b     |
        |      \______/
    ``

    Parameters
    ----------
    psi_ab : ket
        State to partially trace and dot with another ket, with
        size ``prod(dims)``.
    psi_a : ket or sequence of kets
        State to act on with the dot product, of size ``prod(dims[sysa])``, or
        several states to act on at once, then rectangular matrix of size
        ``(prod(dims[sysa]), nstates)``.
    dims : sequence of int, optional
        The sub dimensions of ``psi_ab``, inferred as bipartite if not given,
        i.e. ``(psi_a.size, psi_ab.size // psi_a.size)``.
    sysa : int or sequence of int, optional
        Index(es) of the 'a' subsystem(s) to keep.
    out : array, optional
        If provided, the calculation is done into this array.

    Returns
    -------
    ket
    """
    if dims is None:
        da = psi_a.shape[0]
        d = psi_ab.size
        dims = (da, d // da)
    else:
        dims = int2tup(dims)
    sysa = int2tup(sysa)

    # prepare shapes and indexes -- cached
    dims_a, inds_a_ket, inds_ab_bra, inds_ab_ket = \
        prepare_lazy_ptr_dot(psi_a.shape, dims=dims, sysa=sysa)
    psi_ab_tensor = np.asarray(psi_ab).reshape(dims)
    psi_a_tensor = np.asarray(psi_a).reshape(dims_a)

    # find the optimal path -- cached
    path = get_path_lazy_ptr_dot(
        psi_ab_tensor.shape, psi_a_tensor.shape,
        inds_a_ket, inds_ab_bra, inds_ab_ket)

    # perform the contraction
    return do_lazy_ptr_dot(
        psi_ab_tensor, psi_a_tensor,
        inds_a_ket, inds_ab_bra, inds_ab_ket,
        path=path, out=out).reshape(psi_a.shape)


class LazyPtrOperator(spla.LinearOperator):
    """A linear operator representing action of partially tracing a bipartite
    state, then multiplying another 'unipartite' state.

    Parameters
    ----------
    psi_ab : ket
        State to partially trace and dot with another ket, with
        size ``prod(dims)``.
    dims : sequence of int, optional
        The sub dimensions of ``psi_ab``.
    sysa : int or sequence of int, optional
        Index(es) of the 'a' subsystem(s) to keep.
    """

    def __init__(self, psi_ab, dims, sysa):
        self.psi_ab_tensor = np.asarray(psi_ab).reshape(dims)
        self.dims = int2tup(dims)
        self.sysa = int2tup(sysa)
        dims_a = [d for i, d in enumerate(dims) if i in self.sysa]
        sz_a = prod(dims_a)
        super().__init__(dtype=psi_ab.dtype, shape=(sz_a, sz_a))

    def _matvec(self, vecs):
        # prepare shapes and indexes -- cached
        dims_a, inds_a_ket, inds_ab_bra, inds_ab_ket = \
            prepare_lazy_ptr_dot(vecs.shape, dims=self.dims, sysa=self.sysa)

        # have to do this each time?
        psi_a_tensor = np.asarray(vecs).reshape(dims_a)

        # find the optimal path -- cached
        path = get_path_lazy_ptr_dot(
            self.psi_ab_tensor.shape, psi_a_tensor.shape,
            inds_a_ket, inds_ab_bra, inds_ab_ket)

        # perform the contraction
        return do_lazy_ptr_dot(
            self.psi_ab_tensor, psi_a_tensor,
            inds_a_ket, inds_ab_bra, inds_ab_ket,
            path=path).reshape(vecs.shape)

    def _matmat(self, vecs):
        return self._matvec(vecs)

    def _adjoint(self):
        return self.__class__(self.psi_ab_tensor.conjugate(),
                              self.dims, self.sysa)


@functools.lru_cache(128)
def get_cntrct_inds_ptr_ppt_dot(ndim_abc, sysa, sysb, matmat=False):
    """Find the correct integer contraction labels for ``lazy_ptr_ppt_dot``.

    Parameters
    ----------
    ndim_abc : int
        The total number of subsystems (dimensions) in 'abc'.
    sysa : int or sequence of int, optional
        Index(es) of the 'a' subsystem(s) to keep, with respect to all
        the dimensions, ``dims``, (i.e. pre-partial trace).
    sysa : int or sequence of int, optional
        Index(es) of the 'b' subsystem(s) to keep, with respect to all
        the dimensions, ``dims``, (i.e. pre-partial trace).
    matmat : bool, optional
        Whether to output indices corresponding to a matrix-vector or
        matrix-matrix opertion.

    Returns
    -------
    inds_ab_ket : sequence of int
        The tensor index labels for the ket on subsystems 'ab'.
    inds_abc_bra : sequence of int
        The tensor index labels for the bra on subsystems 'abc'.
    inds_abc_ket : sequence of int
        The tensor index labels for the ket on subsystems 'abc'.
    inds_out : sequence of int
        The tensor indices of the resulting ket, important as these might
        no longer be ordered.
    """
    inds_ab_ket = []
    inds_abc_bra = []
    inds_abc_ket = []
    inds_out = []

    upper_inds = iter(range(ndim_abc, 2 * ndim_abc))

    for i in range(ndim_abc):
        if i in sysa:
            up_ind = next(upper_inds)
            inds_ab_ket.append(i)
            inds_abc_bra.append(up_ind)
            inds_abc_ket.append(i)
            inds_out.append(up_ind)
        elif i in sysb:
            up_ind = next(upper_inds)
            inds_ab_ket.append(up_ind)
            inds_abc_bra.append(up_ind)
            inds_abc_ket.append(i)
            inds_out.append(i)
        else:
            inds_abc_bra.append(i)
            inds_abc_ket.append(i)

    if matmat:
        inds_ab_ket.append(2 * ndim_abc)
        inds_out.append(2 * ndim_abc)

    return (tuple(inds_ab_ket), tuple(inds_abc_bra),
            tuple(inds_abc_ket), tuple(inds_out))


@functools.lru_cache(128)
def prepare_lazy_ptr_ppt_dot(psi_ab_shape, dims, sysa, sysb):
    """Pre-calculate the arrays and indexes etc for ``lazy_ptr_ppt_dot``.
    """
    mat_size = psi_ab_shape[1] if len(psi_ab_shape) > 1 else 1

    inds_ab_ket, inds_abc_bra, inds_abc_ket, inds_out = \
        get_cntrct_inds_ptr_ppt_dot(len(dims), sysa, sysb, matmat=mat_size > 1)

    dims_ab = tuple(d for i, d in enumerate(dims)
                    if (i in sysa) or (i in sysb))
    if mat_size > 1:
        dims_ab = dims_ab + (mat_size,)

    return dims_ab, inds_ab_ket, inds_abc_bra, inds_abc_ket, inds_out


@functools.lru_cache(128)
def get_path_lazy_ptr_ppt_dot(psi_abc_tensor_shape, psi_ab_tensor_shape,
                              inds_ab_ket, inds_abc_bra,
                              inds_abc_ket, inds_out):
    return contract_path(
        HuskArray(psi_ab_tensor_shape), inds_ab_ket,
        HuskArray(psi_abc_tensor_shape), inds_abc_bra,
        HuskArray(psi_abc_tensor_shape), inds_abc_ket,
        inds_out,
        optimize='optimal',
        memory_limit=-1,
    )[0]


def do_lazy_ptr_ppt_dot(psi_ab_tensor, psi_abc_tensor,
                        inds_ab_ket, inds_abc_bra, inds_abc_ket, inds_out,
                        path, out=None):
    if out is None:
        out = np.empty_like(psi_ab_tensor)

    # must have ``inds_out`` as resulting indices are not ordered
    # in the same way as input due to partial tranpose.
    return contract(
        psi_ab_tensor, inds_ab_ket,
        psi_abc_tensor.conjugate(), inds_abc_bra,
        psi_abc_tensor, inds_abc_ket,
        inds_out,
        optimize=path,
        out=out,
    )


def lazy_ptr_ppt_dot(psi_abc, psi_ab, dims, sysa, sysb, out=None):
    r"""Perform the 'lazy' evalution of
    ``partial_transpose(ptr(psi_abc, ...)) @ psi_ab``, that is, contract the
    tensor diagram in an efficient way that does not necessarily construct
    the explicit reduced density matrix. For a tripartite system, the partial
    trace is with respect to ``c``, while the partial tranpose is with
    respect to ``a/b``. In tensor diagram notation:
    ``
         ( | )
    +--------------+
    |   psi_ab     |
    +______________+  _____
     a|  ____   b|   /     \
      | /   a\   |   |c    |
      | | +-------------+  |
      | | |  psi_abc.H  |  |
      \ / +-------------+  |
       X                   |
      / \ +-------------+  |
      | | |   psi_abc   |  |
      | | +-------------+  |
      | \____/a  |b  |c    |
     a|          |   \_____/
    ``

    Parameters
    ----------
    psi_abc : ket
        State to partially trace, partially tranpose, then dot with another
        ket, with size ``prod(dims)``.
    psi_ab : ket, sequence of kets
        State to act on with the dot product, of size
        , or series of states to ac
        State to act on with the dot product, of size
        ``prod(dims[sysa] + dims[sysb])``, or several states to act on at once,
        then rectangular matrix of size
        ``(prod(dims[sysa] + dims[sysb]), nstates)``.
    dims : sequence of int
        The sub dimensions of ``psi_abc``.
    sysa : int or sequence of int, optional
        Index(es) of the 'a' subsystem(s) to keep, with respect to all
        the dimensions, ``dims``, (i.e. pre-partial trace).
    sysa : int or sequence of int, optional
        Index(es) of the 'b' subsystem(s) to keep, with respect to all
        the dimensions, ``dims``, (i.e. pre-partial trace).
    out : array, optional
        If provided, the calculation is done into this array.

    Returns
    -------
    ket
    """
    # convert to tuple so can always cache
    dims, sysa, sysb = int2tup(dims), int2tup(sysa), int2tup(sysb)

    # prepare shapes and indexes -- cached
    dims_ab, inds_ab_ket, inds_abc_bra, inds_abc_ket, inds_out = \
        prepare_lazy_ptr_ppt_dot(psi_ab.shape, dims, sysa, sysb)
    psi_ab_tensor = np.asarray(psi_ab).reshape(dims_ab)
    psi_abc_tensor = np.asarray(psi_abc).reshape(dims)

    # find the optimal path -- cached
    path = get_path_lazy_ptr_ppt_dot(
        psi_abc_tensor.shape, psi_ab_tensor.shape,
        inds_ab_ket, inds_abc_bra, inds_abc_ket, inds_out)

    # perform contraction
    return do_lazy_ptr_ppt_dot(
        psi_ab_tensor, psi_abc_tensor,
        inds_ab_ket, inds_abc_bra, inds_abc_ket, inds_out,
        path, out=out).reshape(psi_ab.shape)


class LazyPtrPptOperator(spla.LinearOperator):
    """A linear operator representing action of partially tracing a tripartite
    state, partially transposing the remaining bipartite state, then
    multiplying another bipartite state.

    Parameters
    ----------
    psi_abc : ket
        State to partially trace, partially tranpose, then dot with another
        ket, with size ``prod(dims)``.
        ``prod(dims[sysa] + dims[sysb])``.
    dims : sequence of int
        The sub dimensions of ``psi_abc``.
    sysa : int or sequence of int, optional
        Index(es) of the 'a' subsystem(s) to keep, with respect to all
        the dimensions, ``dims``, (i.e. pre-partial trace).
    sysa : int or sequence of int, optional
        Index(es) of the 'b' subsystem(s) to keep, with respect to all
        the dimensions, ``dims``, (i.e. pre-partial trace).
    """

    def __init__(self, psi_abc, dims, sysa, sysb):
        self.psi_abc_tensor = np.asarray(psi_abc).reshape(dims)
        self.dims = int2tup(dims)
        self.sysa, self.sysb = int2tup(sysa), int2tup(sysb)
        sys_ab = self.sysa + self.sysb
        sz_ab = prod([d for i, d in enumerate(dims) if i in sys_ab])
        super().__init__(dtype=psi_abc.dtype, shape=(sz_ab, sz_ab))

    def _matvec(self, vecs):
        # prepare shapes and indexes -- cached
        dims_ab, inds_ab_ket, inds_abc_bra, inds_abc_ket, inds_out = \
            prepare_lazy_ptr_ppt_dot(vecs.shape, self.dims,
                                     self.sysa, self.sysb)

        # do each time
        psi_ab_tensor = np.asarray(vecs).reshape(dims_ab)

        # find the optimal path -- cached
        path = get_path_lazy_ptr_ppt_dot(
            self.psi_abc_tensor.shape, psi_ab_tensor.shape,
            inds_ab_ket, inds_abc_bra, inds_abc_ket, inds_out)

        # perform contraction
        return do_lazy_ptr_ppt_dot(
            psi_ab_tensor, self.psi_abc_tensor,
            inds_ab_ket, inds_abc_bra, inds_abc_ket, inds_out,
            path).reshape(vecs.shape)

    def _matmat(self, vecs):
        return self._matvec(vecs)

    def _adjoint(self):
        return self.__class__(self.psi_abc_tensor.conjugate(), self.dims,
                              self.sysa, self.sysb)


# --------------------------------------------------------------------------- #
#                         Lanczos tri-diag technique                          #
# --------------------------------------------------------------------------- #

K_DEFAULT = 20
R_DEFAULT = 10
BSZ_DEFAULT = 1
BETA_TOL = 1e-9


def inner(a, b):
    """Inner product between two vectors
    """
    return vdot(a, b).real


def norm_fro(a):
    """'Frobenius' norm of a vector.
    """
    return sqrt(inner(a, a))


def construct_lanczos_tridiag(A, K=K_DEFAULT, v0=None, bsz=BSZ_DEFAULT):
    """Construct the tridiagonal lanczos matrix using only matvec operators.

    Parameters
    ----------
    A : matrix-like or linear operator
        The operator to approximate, must implement ``.dot`` method to compute
        its action on a vector.
    v0 : vector, optional
        The starting vector to iterate with, default to random.
    K : int, optional
        The number of iterations and thus rank of the matrix to find.

    Returns
    -------
    alpha : sequence of float of length k
        The diagonal entries of the lanczos matrix.
    beta : sequence of float of length k
        The off-diagonal entries of the lanczos matrix, with the last entry
        the 'look' forward value.
    scaling : float
        How to scale the overall weights.
    """
    if isinstance(A, np.matrix):
        A = np.asarray(A)
    d = A.shape[0]

    if bsz == 1:
        v_shp = (d,)
    else:
        v_shp = (d, bsz)

    alpha = np.zeros(K + 1)
    beta = np.zeros(K + 2)
    beta[1] = sqrt(prod(v_shp))  # by construction

    if v0 is None:
        V = np.random.choice([-1, 1, 1j, -1j], v_shp)
        V /= beta[1]  # normalize
    else:
        V = v0.astype(np.complex128)
        V /= norm_fro(V)  # normalize (make sure has unit variance)
    Vm1 = np.zeros_like(V)

    for j in range(1, K + 1):
        Vt = A.dot(V)
        Vt -= beta[j] * Vm1
        alpha[j] = inner(V, Vt)
        Vt -= alpha[j] * V
        beta[j + 1] = norm_fro(Vt)

        # check for convergence
        if abs(beta[j + 1]) < BETA_TOL:
            break

        Vm1[...] = V[...]
        np.divide(Vt, beta[j + 1], out=V)

    return alpha[1:j + 1], beta[2:j + 2], beta[1]**2 / bsz


def lanczos_tridiag_eig(alpha, beta, check_finite=True):
    """Find the eigen-values and -vectors of the Lanczos triadiagonal matrix.

    Parameters
    ----------
    alpha : array of float
        The diagonal.
    beta : array of float
        The k={-1, 1} off-diagonal. Only first ``len(alpha) - 1`` entries used.
    """
    Tk_banded = np.empty((2, alpha.size))
    Tk_banded[1, -1] = 0.0  # sometimes can get nan here? -> breaks eig_banded
    Tk_banded[0, :] = alpha
    Tk_banded[1, :beta.size] = beta
    return scla.eig_banded(Tk_banded, lower=True, check_finite=check_finite)


def calc_trace_fn_tridiag(tl, tv, fn, pos=True):
    return sum(fn(max(tl[i], 0.0) if pos else tl[i]) * tv[0, i]**2
               for i in range(tl.size))


def approx_spectral_function(A, fn,
                             K=K_DEFAULT, R=R_DEFAULT, bsz=BSZ_DEFAULT,
                             v0=None, pos=False):
    """Approximate a spectral function, that is, the quantity ``Tr(fn(A))``.

    Parameters
    ----------
    A : matrix-like or LinearOperator
        Operator to approximate spectral function for. Should implement
        ``A.dot(vec)``.
    fn : callable
        Scalar function with with to act on approximate eigenvalues.
    K : int, optional
        The size of the tri-diagonal lanczos matrix to form. Cost of algorithm
        scales linearly with ``K``.
    R : int, optional
        The number of repeats with different initial random vectors to perform.
        Increasing this should increase accuracy as ``sqrt(R)``. Cost of
        algorithm thus scales linearly with ``R``.
    v0 : vector, or callable
        Initial vector to iterate with, sets ``R=1`` if given. If callable, the
        function to produce a random intial vector (sequence).
    pos : bool, optional
        If True, make sure any approximate eigenvalues are positive by
        clipping below 0.

    Returns
    -------
    scalar
        The approximate value ``Tr(fn(a))``.
    """
    if (v0 is not None) and not callable(v0):
        R = 1

    def gen_vals():
        for _ in range(R):
            alpha, beta, scaling = construct_lanczos_tridiag(
                A, K=K, bsz=bsz, v0=v0() if callable(v0) else v0)

            # First bound
            Gf = scaling * calc_trace_fn_tridiag(*lanczos_tridiag_eig(
                alpha, beta, check_finite=False), fn=fn, pos=pos)

            # Check if already converged (i.e. found non-null space)
            if abs(beta[-1]) < BETA_TOL:
                yield Gf

            else:
                beta[-1] = beta[0]
                # second bound
                Rf = scaling * calc_trace_fn_tridiag(*lanczos_tridiag_eig(
                    np.append(alpha, alpha[0]), beta, check_finite=False),
                    fn=fn, pos=pos)

                yield (Gf + Rf) / 2  # mean of lower and upper bounds

    return sum(gen_vals()) / R  # take average over repeats


tr_abs_approx = functools.partial(approx_spectral_function, fn=abs)
tr_exp_approx = functools.partial(approx_spectral_function, fn=exp)
tr_sqrt_approx = functools.partial(approx_spectral_function, fn=sqrt, pos=True)
tr_xlogx_approx = functools.partial(
    approx_spectral_function, fn=lambda x: x * log2(x) if x > 0 else 0.0)


# --------------------------------------------------------------------------- #
#                             Specific quantities                             #
# --------------------------------------------------------------------------- #

def entropy_subsys_approx(psi_ab, dims, sysa, **kwargs):
    """Approximate the (Von Neumann) entropy of a pure state's subsystem.
    psi_ab : ket
        Bipartite state to partially trace and find entopy of.
    dims : sequence of int, optional
        The sub dimensions of ``psi_ab``.
    sysa : int or sequence of int, optional
        Index(es) of the 'a' subsystem(s) to keep.
    K : int, optional
        The size of the tri-diagonal lanczos matrix to form. Cost of algorithm
        scales linearly with ``K``.
    R : int, optional
        The number of repeats with different initial random vectors to perform.
        Increasing this should increase accuracy as ``sqrt(R)``. Cost of
        algorithm thus scales linearly with ``R``.
    v0 :
        Initial vector to iterate with, sets ``R=1`` if given.
    """
    lo = LazyPtrOperator(psi_ab, dims=dims, sysa=sysa)
    return - tr_xlogx_approx(lo, **kwargs)


def norm_ppt_subsys_approx(psi_abc, dims, sysa, sysb, **kwargs):
    """Estimate the norm of the partial tranpose of a pure state's subsystem.
    """
    lo = LazyPtrPptOperator(psi_abc, dims=dims, sysa=sysa, sysb=sysb)
    return tr_abs_approx(lo, **kwargs)


def logneg_subsys_approx(*args, **kwargs):
    """Estimate the logarithmic negativity of a pure state's subsystem.

    Parameters
    ----------
    psi_abc : ket
        Pure tripartite state, for which estimate the entanglement between
        'a' and 'b'.
    dims : sequence of int
        The sub dimensions of ``psi_abc``.
    sysa : int or sequence of int, optional
        Index(es) of the 'a' subsystem(s) to keep, with respect to all
        the dimensions, ``dims``, (i.e. pre-partial trace).
    sysa : int or sequence of int, optional
        Index(es) of the 'b' subsystem(s) to keep, with respect to all
        the dimensions, ``dims``, (i.e. pre-partial trace).
    K : int, optional
        The size of the tri-diagonal lanczos matrix to form. Cost of algorithm
        scales linearly with ``K``.
    R : int, optional
        The number of repeats with different initial random vectors to perform.
        Increasing this should increase accuracy as ``sqrt(R)``. Cost of
        algorithm thus scales linearly with ``R``.
    v0 :
        Initial vector to iterate with, sets ``R=1`` if given.
    """
    return max(log2(norm_ppt_subsys_approx(*args, **kwargs)), 0.0)


def negativity_subsys_approx(*args, **kwargs):
    """Estimate the negativity of a pure state's subsystem.

    Parameters
    ----------
    psi_abc : ket
        Pure tripartite state, for which estimate the entanglement between
        'a' and 'b'.
    dims : sequence of int
        The sub dimensions of ``psi_abc``.
    sysa : int or sequence of int, optional
        Index(es) of the 'a' subsystem(s) to keep, with respect to all
        the dimensions, ``dims``, (i.e. pre-partial trace).
    sysa : int or sequence of int, optional
        Index(es) of the 'b' subsystem(s) to keep, with respect to all
        the dimensions, ``dims``, (i.e. pre-partial trace).
    K : int, optional
        The size of the tri-diagonal lanczos matrix to form. Cost of algorithm
        scales linearly with ``K``.
    R : int, optional
        The number of repeats with different initial random vectors to perform.
        Increasing this should increase accuracy as ``sqrt(R)``. Cost of
        algorithm thus scales linearly with ``R``.
    v0 :
        Initial vector to iterate with, sets ``R=1`` if given.
    """
    return max((norm_ppt_subsys_approx(*args, **kwargs) - 1) / 2, 0.0)
