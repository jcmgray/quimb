"""Use lanczos tri-diagonalization to approximate the spectrum of any operator
which has an efficient represenation of its linear action on a vector.
"""
# TODO: error estimates
# TODO: more advanced tri-diagonalization method?

import functools
from math import sqrt, log2, exp

import numpy as np
import scipy.linalg as scla
import scipy.sparse.linalg as spla
try:
    from opt_einsum import contract as contract
except ImportError:
    from numpy import contract

from ..accel import prod
from ..utils import int2tup


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

    return inds_a_ket, inds_ab_bra, inds_ab_ket


def lazy_ptr_dot(psi_ab, psi_a, dims=None, sysa=0):
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

    Returns
    -------
    ket
    """
    mat_size = psi_a.shape[1] if len(psi_a.shape) > 1 else 1

    if dims is None:
        da = psi_a.shape[0]
        d = psi_ab.size
        dims = (da, d // da)

    # convert to tuple so can always cache
    sysa = int2tup(sysa)

    ndim_ab = len(dims)
    inds_a_ket, inds_ab_bra, inds_ab_ket = get_cntrct_inds_ptr_dot(
        ndim_ab, sysa, matmat=mat_size > 1)

    dims_a = [d for i, d in enumerate(dims) if i in sysa]
    if mat_size > 1:
        dims_a.append(mat_size)

    psi_a_tensor = np.asarray(psi_a).reshape(dims_a)
    psi_ab_tensor = np.asarray(psi_ab).reshape(dims)

    return contract(
        psi_a_tensor, inds_a_ket,
        psi_ab_tensor.conjugate(), inds_ab_bra,
        psi_ab_tensor, inds_ab_ket,
        optimize=True,
    ).reshape(psi_a.shape)


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
        self.psi_ab = psi_ab
        self.dims = dims
        self.sysa = int2tup(sysa)
        dims_a = [d for i, d in enumerate(dims) if i in self.sysa]
        sz_a = prod(dims_a)
        super().__init__(dtype=psi_ab.dtype, shape=(sz_a, sz_a))

    def _matvec(self, vec):
        return lazy_ptr_dot(self.psi_ab, vec, self.dims, self.sysa)

    def _matmat(self, vecs):
        return lazy_ptr_dot(self.psi_ab, vecs, self.dims, self.sysa)

    def _adjoint(self):
        return self.__class__(self.psi_ab.conjugate(), self.dims, self.sysa)


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

    return inds_ab_ket, inds_abc_bra, inds_abc_ket, inds_out


def lazy_ptr_ppt_dot(psi_abc, psi_ab, dims, sysa, sysb):
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

    Returns
    -------
    ket
    """
    mat_size = psi_ab.shape[1] if len(psi_ab.shape) > 1 else 1

    # convert to tuple so can always cache
    sysa, sysb = int2tup(sysa), int2tup(sysb)

    inds_ab_ket, inds_abc_bra, inds_abc_ket, inds_out = \
        get_cntrct_inds_ptr_ppt_dot(len(dims), sysa, sysb, matmat=mat_size > 1)

    dims_ab = [d for i, d in enumerate(dims) if (i in sysa) or (i in sysb)]
    if mat_size > 1:
        dims_ab.append(mat_size)
    psi_ab_tensor = np.asarray(psi_ab).reshape(dims_ab)
    psi_abc_tensor = np.asarray(psi_abc).reshape(dims)

    # must have ``inds_out`` as resulting indices are not ordered
    # in the same way as input due to partial tranpose.
    return contract(
        psi_ab_tensor, inds_ab_ket,
        psi_abc_tensor.conjugate(), inds_abc_bra,
        psi_abc_tensor, inds_abc_ket,
        inds_out,
        optimize=True,
    ).reshape(psi_ab.shape)


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
        self.psi_abc = psi_abc
        self.dims = dims
        self.sysa, self.sysb = int2tup(sysa), int2tup(sysb)
        sys_ab = self.sysa + self.sysb
        sz_ab = prod([d for i, d in enumerate(dims) if i in sys_ab])
        super().__init__(dtype=psi_abc.dtype, shape=(sz_ab, sz_ab))

    def _matvec(self, vec):
        return lazy_ptr_ppt_dot(self.psi_abc, vec, self.dims,
                                self.sysa, self.sysb)

    def _matmat(self, vecs):
        return lazy_ptr_ppt_dot(self.psi_abc, vecs, self.dims,
                                self.sysa, self.sysb)

    def _adjoint(self):
        return self.__class__(self.psi_abc.conjugate(), self.dims,
                              self.sysa, self.sysb)


# --------------------------------------------------------------------------- #
#                         Lanczos tri-diag technique                          #
# --------------------------------------------------------------------------- #

M_DEFAULT = 20
R_DEFAULT = 20


def construct_lanczos_tridiag(A, M=M_DEFAULT, v0=None):
    """Construct the tridiagonal lanczos matrix using only matvec operators.

    Parameters
    ----------
    A : matrix-like or linear operator
        The operator to approximate, must implement ``.dot`` method to compute
        its action on a vector.
    v0 : vector, optional
        The starting vector to iterate with, default to random.
    M : int, optional
        The number of iterations and thus rank of the matrix to find.

    Returns
    -------
    alpha : sequence of float of length k
        The diagonal entries of the lanczos matrix.
    beta : sequence of float of length k - 1
        The off-diagonal entries of the lanczos matrix.
    """
    if isinstance(A, np.matrix):
        A = np.asarray(A)

    d = A.shape[0]

    alpha = np.zeros(M + 1)
    beta = np.zeros(M + 2)
    vk = np.empty((d, M + 2), dtype=np.complex128)
    vk[:, 0] = 0.0j

    # initialize & normalize the starting vector
    if v0 is None:
        vk[:, 1] = np.random.randn(d)
        vk[:, 1] += 1.0j * np.random.randn(d)
    else:
        vk[:, 1] = v0
    vk[:, 1] /= sqrt(np.vdot(vk[:, 1], vk[:, 1]).real)

    # construct the krylov subspace
    for k in range(1, M + 1):
        wk = A.dot(vk[:, k])
        wk -= beta[k] * vk[:, k - 1]
        alpha[k] = np.vdot(wk, vk[:, k]).real
        wk -= alpha[k] * vk[:, k]
        beta[k + 1] = sqrt(np.vdot(wk, wk).real)
        if beta[k + 1] < 1e-12:  # converged
            break
        np.divide(wk, beta[k + 1], out=vk[:, k + 1])

    return alpha[1:k + 1], beta[2:k + 1], A.shape[0]


def lanczos_tridiag_eig(alpha, beta, check_finite=True):
    """Find the eigen-values and -vectors of the Lanczos triadiagonal matrix.
    """
    Tk_banded = np.empty((2, alpha.size))
    Tk_banded[1, -1] = 0.0  # sometimes can get nan here? -> breaks eig_banded
    Tk_banded[0, :] = alpha
    Tk_banded[1, :-1] = beta
    return scla.eig_banded(Tk_banded, lower=True, check_finite=check_finite)


def calc_trace_fn_tridiag(tl, tv, fn, pos=True):
    return sum(fn(max(tl[i], 0.0) if pos else tl[i]) * tv[0, i]**2
               for i in range(tl.size))


def approx_spectral_function(A, fn, M=M_DEFAULT, R=R_DEFAULT,
                             v0=None, pos=False):
    """Approximate a spectral function, that is, the quantity ``Tr(fn(A))``.

    Parameters
    ----------
    A : matrix-like or LinearOperator
        Operator to approximate spectral function for. Should implement
        ``A.dot(vec)``.
    fn : callable
        Scalar function with with to act on approximate eigenvalues.
    M : int, optional
        The size of the tri-diagonal lanczos matrix to form. Cost of algorithm
        scales linearly with ``M``.
    R : int, optional
        The number of repeats with different initial random vectors to perform.
        Increasing this should increase accuracy as ``sqrt(R)``. Cost of
        algorithm thus scales linearly with ``R``.
    v0 :
        Initial vector to iterate with, sets ``R=1`` if given.
    pos : bool, optional
        If True, make sure any approximate eigenvalues are positive by
        clipping below 0.

    Returns
    -------
    scalar
        The approximate value ``Tr(fn(a))``.
    """
    if v0 is not None:
        R = 1

    def gen_vals():
        for _ in range(R):
            alpha, beta, scaling = construct_lanczos_tridiag(A, M=M, v0=v0)
            tl, tv = lanczos_tridiag_eig(alpha, beta, check_finite=False)
            yield scaling * calc_trace_fn_tridiag(tl, tv, fn, pos=pos)

    return sum(gen_vals()) / R


tr_abs_approx = functools.partial(approx_spectral_function, fn=abs)
tr_exp_approx = functools.partial(approx_spectral_function, fn=exp)
tr_sqrt_approx = functools.partial(approx_spectral_function, fn=sqrt, pos=True)
tr_xlogx_approx = functools.partial(approx_spectral_function,
                                    fn=lambda x: x * log2(x) if x > 0 else 0.0)


# --------------------------------------------------------------------------- #
#                             Specific quantities                             #
# --------------------------------------------------------------------------- #

def entropy_subsys_approx(psi_ab, dims, sysa,
                          M=M_DEFAULT, R=R_DEFAULT, v0=None):
    """Approximate the (Von Neumann) entropy of a pure state's subsystem.
    psi_ab : ket
        Bipartite state to partially trace and find entopy of.
    dims : sequence of int, optional
        The sub dimensions of ``psi_ab``.
    sysa : int or sequence of int, optional
        Index(es) of the 'a' subsystem(s) to keep.
    M : int, optional
        The size of the tri-diagonal lanczos matrix to form. Cost of algorithm
        scales linearly with ``M``.
    R : int, optional
        The number of repeats with different initial random vectors to perform.
        Increasing this should increase accuracy as ``sqrt(R)``. Cost of
        algorithm thus scales linearly with ``R``.
    v0 :
        Initial vector to iterate with, sets ``R=1`` if given.
    """
    lo = LazyPtrOperator(psi_ab, dims=dims, sysa=sysa)
    return - tr_xlogx_approx(lo, M=M, R=R, v0=v0)


def norm_ppt_subsys_approx(psi_abc, dims, sysa, sysb,
                           M=M_DEFAULT, R=R_DEFAULT, v0=None):
    """Estimate the norm of the partial tranpose of a pure state's subsystem.
    """
    lo = LazyPtrPptOperator(psi_abc, dims=dims, sysa=sysa, sysb=sysb)
    return tr_abs_approx(lo, M=M, R=R, v0=v0)


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
    M : int, optional
        The size of the tri-diagonal lanczos matrix to form. Cost of algorithm
        scales linearly with ``M``.
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
    M : int, optional
        The size of the tri-diagonal lanczos matrix to form. Cost of algorithm
        scales linearly with ``M``.
    R : int, optional
        The number of repeats with different initial random vectors to perform.
        Increasing this should increase accuracy as ``sqrt(R)``. Cost of
        algorithm thus scales linearly with ``R``.
    v0 :
        Initial vector to iterate with, sets ``R=1`` if given.
    """
    return max((norm_ppt_subsys_approx(*args, **kwargs) - 1) / 2, 0.0)
