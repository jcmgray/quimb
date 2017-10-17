"""Use lanczos tri-diagonalization to approximate the spectrum of any operator
which has an efficient representation of its linear action on a vector.
"""
import functools
from math import sqrt, log2, exp

import numpy as np
import scipy.linalg as scla
import scipy.sparse.linalg as spla
from ..core import ptr
from ..tensor_networks import einsum, einsum_path, HuskArray
from ..accel import prod, vdot
from ..utils import int2tup


# --------------------------------------------------------------------------- #
#                  'Lazy' representation tensor contractions                  #
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
    return einsum_path(
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

    return einsum(
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
    diagram notation::

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
    return einsum_path(
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
    return einsum(
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
    respect to ``a/b``. In tensor diagram notation::

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

def inner(a, b):
    """Inner product between two vectors
    """
    return vdot(a, b).real


def norm_fro(a):
    """'Frobenius' norm of a vector.
    """
    return sqrt(inner(a, a))


def construct_lanczos_tridiag(
        A,
        K,
        v0=None,
        bsz=1,
        beta_tol=1e-6):
    """Construct the tridiagonal lanczos matrix using only matvec operators.
    This is a generator that iteratively yields the alpha and beta digaonals
    at each step.

    Parameters
    ----------
    A : matrix-like or linear operator
        The operator to approximate, must implement ``.dot`` method to compute
        its action on a vector.
    K : int, optional
        The maximum number of iterations and thus rank of the matrix to find.
    v0 : vector, optional
        The starting vector to iterate with, default to random.
    bsz : int, optional
        The block size (number of columns) of random vectors to iterate with.
    beta_tol : float, optional
        The 'breakdown' tolerance. If the next beta ceofficient in the lanczos
        matrix is less that this, implying that the full non-null space has
        been found, terminate early.

    Yields
    ------
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
        if abs(beta[j + 1]) < beta_tol:
            yield alpha[1:j + 1], beta[2:j + 2], beta[1]**2 / bsz
            break

        Vm1[...] = V[...]
        np.divide(Vt, beta[j + 1], out=V)
        yield (np.copy(alpha[1:j + 1]),
               np.copy(beta[2:j + 2]),
               np.copy(beta[1])**2 / bsz)


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


def ext_per_trim(x, p=0.6, s=1.0):
    """Extended percentile trimmed-mean. Makes the mean robust to asymmetric
    outliers, while using all data when it is nicely clustered. This can be
    visualized roughly as:

            |--------|=========|--------|
        x     x   xx xx xxxxx xxx   xx     x      x           x

    Where the inner range contains the central ``p`` proportion of the data,
    and the outer ranges entends this by a factor of ``s`` either side.

    Parameters
    ----------
    x : array
        Data to trim.
    p : Proportion of data used to define the 'central' percentile.
        For example, p=0.5 gives the inter-quartile range.
    s : Include data up to this factor times the central 'percentile' range
        away from the central percentile itself.

    Returns
    xt : array
        Trimmed data.
    """
    lb = np.percentile(x, 100 * (1 - p) / 2)
    ub = np.percentile(x, 100 * (1 + p) / 2)
    ib = ub - lb

    x = np.array(x)
    trimmed_x = x[(lb - s * ib < x) & (x < ub + s * ib)]

    return trimmed_x


def approx_spectral_function(
        A, fn,
        tol=5e-3,
        K=100,
        R=100,
        bsz=10,
        v0=None,
        pos=False,
        tau=1e-3,
        tol_scale=1,
        beta_tol=1e-6,
        mean_p=0.7,
        mean_s=1.0):
    """Approximate a spectral function, that is, the quantity ``Tr(fn(A))``.

    Parameters
    ----------
    A : matrix-like or LinearOperator
        Operator to approximate spectral function for. Should implement
        ``A.dot(vec)``.
    fn : callable
        Scalar function with with to act on approximate eigenvalues.
    tol : float, optional
        Convergence tolerance threshold for error on mean of repeats. This can
        pretty much be relied on as the overall accuracy. See also
        ``tol_scale`` and ``tau``. Default: 0.5%.
    K : int, optional
        The size of the tri-diagonal lanczos matrix to form. Cost of algorithm
        scales linearly with ``K``. If ``tau`` is non-zero, this is the
        maximum size matrix to form. Default: 100.
    R : int, optional
        The number of repeats with different initial random vectors to perform.
        Increasing this should increase accuracy as ``sqrt(R)``. Cost of
        algorithm thus scales linearly with ``R``. If ``tol`` is non-zero, this
        is the maximum number of repeats. Default: 100.
    bsz : int, optional
        Number of simultenous vector columns to use at once, 1 equating to the
        standard lanczos method. If ``bsz > 1`` then ``A`` must implement
        matrix-matrix multiplication. This is a more performant way of
        essentially increasing ``R``, at the cost of more memory. Default: 10.
    v0 : vector, or callable
        Initial vector to iterate with, sets ``R=1`` if given. If callable, the
        function to produce a random intial vector (sequence).
    pos : bool, optional
        If True, make sure any approximate eigenvalues are positive by
        clipping below 0.
    tau : float, optional
        The relative tolerance required for a single lanczos run to converge.
        This needs to be small enough that each estimate with a single random
        vector produces an unbiased sample of the operators spectrum.
        Default: 0.05%.
    tol_scale : float, optional
        This sets the overall expected scale of each estimate, so that an
        absolute tolerance can be used for values near zero. Default: 1.
    beta_tol : float, optional
        The 'breakdown' tolerance. If the next beta ceofficient in the lanczos
        matrix is less that this, implying that the full non-null space has
        been found, terminate early. Default: 1e-6.
    mean_p : float, optional
        Factor for robustly finding mean and err of repeat estimates,
        see :func:`ext_per_trim`.
    mean_s : float, optional
        Factor for robustly finding mean and err of repeat estimates,
        see :func:`ext_per_trim`.

    Returns
    -------
    scalar
        The approximate value ``Tr(fn(a))``.
    """
    if (v0 is not None) and not callable(v0):
        R = 1

    def single_random_estimate():
        estimate = None

        # iteratively build the lanczos matrix, checking for convergence
        for alpha, beta, scaling in construct_lanczos_tridiag(
                A, K=K, bsz=bsz, beta_tol=beta_tol,
                v0=v0() if callable(v0) else v0):

            # First bound
            Gf = scaling * calc_trace_fn_tridiag(*lanczos_tridiag_eig(
                alpha, beta, check_finite=False), fn=fn, pos=pos)

            # check for break-down convergence (e.g. found entire non-null)
            if abs(beta[-1]) < beta_tol:
                estimate = Gf
                break

            # second bound
            beta[-1] = beta[0]
            Rf = scaling * calc_trace_fn_tridiag(*lanczos_tridiag_eig(
                np.append(alpha, alpha[0]), beta, check_finite=False),
                fn=fn, pos=pos)

            # check for error bound convergence
            if abs(Rf - Gf) < 2 * tau * (abs(Gf) + tol_scale):
                estimate = (Gf + Rf) / 2
                break

        # didn't converge, use best estimate
        if estimate is None:
            estimate = (Gf + Rf) / 2

        return estimate

    estimate = None
    samples = []

    for r, sample in ((r, single_random_estimate()) for r in range(R)):
        samples.append(sample)

        # wait a few iterations before checking error on mean breakout
        if r >= 3:  # i.e. 4 samples
            xtrim = ext_per_trim(samples, p=mean_p, s=mean_s)
            estimate, sdev = np.mean(xtrim), np.std(xtrim)

            err = sdev / r ** 0.5

            if err < tol * (abs(estimate) + tol_scale):
                return estimate

    return np.mean(samples) if estimate is None else estimate


@functools.wraps(approx_spectral_function)
def tr_abs_approx(*args, **kwargs):
    return approx_spectral_function(*args, fn=abs, **kwargs)


@functools.wraps(approx_spectral_function)
def tr_exp_approx(*args, **kwargs):
    return approx_spectral_function(*args, fn=exp, **kwargs)


@functools.wraps(approx_spectral_function)
def tr_sqrt_approx(*args, **kwargs):
    return approx_spectral_function(*args, fn=sqrt, pos=True, **kwargs)


def xlogx(x):
    return x * log2(x) if x > 0 else 0.0


@functools.wraps(approx_spectral_function)
def tr_xlogx_approx(*args, **kwargs):
    return approx_spectral_function(*args, fn=xlogx, **kwargs)


# --------------------------------------------------------------------------- #
#                             Specific quantities                             #
# --------------------------------------------------------------------------- #

def entropy_subsys_approx(psi_ab, dims, sysa, **kwargs):
    """Approximate the (Von Neumann) entropy of a pure state's subsystem.

    Parameters
    ----------
    psi_ab : ket
        Bipartite state to partially trace and find entopy of.
    dims : sequence of int, optional
        The sub dimensions of ``psi_ab``.
    sysa : int or sequence of int, optional
        Index(es) of the 'a' subsystem(s) to keep.
    **kwargs
        See :func:`approx_spectral_function`.
    """
    lo = LazyPtrOperator(psi_ab, dims=dims, sysa=sysa)
    return - tr_xlogx_approx(lo, **kwargs)


def tr_sqrt_subsys_approx(psi_ab, dims, sysa, **kwargs):
    """Approximate the trace sqrt of a pure state's subsystem.

    Parameters
    ----------
    psi_ab : ket
        Bipartite state to partially trace and find trace sqrt of.
    dims : sequence of int, optional
        The sub dimensions of ``psi_ab``.
    sysa : int or sequence of int, optional
        Index(es) of the 'a' subsystem(s) to keep.
    **kwargs
        See :func:`approx_spectral_function`.
    """
    lo = LazyPtrOperator(psi_ab, dims=dims, sysa=sysa)
    return tr_sqrt_approx(lo, **kwargs)


def norm_ppt_subsys_approx(psi_abc, dims, sysa, sysb, **kwargs):
    """Estimate the norm of the partial tranpose of a pure state's subsystem.
    """
    lo = LazyPtrPptOperator(psi_abc, dims=dims, sysa=sysa, sysb=sysb)
    return tr_abs_approx(lo, **kwargs)


def logneg_subsys_approx(psi_abc, dims, sysa, sysb, **kwargs):
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
    **kwargs
        See :func:`approx_spectral_function`.
    """
    return max(log2(norm_ppt_subsys_approx(psi_abc, dims, sysa, sysb,
                                           **kwargs)), 0.0)


def negativity_subsys_approx(psi_abc, dims, sysa, sysb, **kwargs):
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
    **kwargs
        See :func:`approx_spectral_function`.
    """
    return max((norm_ppt_subsys_approx(psi_abc, dims, sysa, sysb,
                                       **kwargs) - 1) / 2, 0.0)


def gen_bipartite_spectral_fn(exact_fn, approx_fn, pure_default):
    """Generate a function that computes a spectral quantity of the subsystem
    of a pure state. Automatically computes for the smaller subsystem, or
    switches to the approximate method for large subsystems.

    Parameters
    ----------
    exact_fn : callable
        The function that computes the quantity on a density matrix, with
        signature: ``exact_fn(rho_a, rank=...)``.
    approx_fn : callable
        The function that approximately computes the quantity using a lazy
        representation of the whole system. With signature
        ``approx_fn(psi_ab, dims, sysa, **approx_opts)``.
    pure_default : float
        The default value when the whole state is the subsystem.

    Returns
    -------
    bipartite_spectral_fn : callable
        The function, with signature:
        ``(psi_ab, dims, sysa, approx_thresh=2**12, **approx_opts)``
    """
    def bipartite_spectral_fn(psi_ab, dims, sysa, approx_thresh=2**12,
                              **approx_opts):
        sysa = int2tup(sysa)
        sz_a = prod(d for i, d in enumerate(dims) if i in sysa)
        sz_b = prod(dims) // sz_a

        # pure state
        if sz_b == 1:
            return pure_default

        # also check if system b is smaller, since spectrum is same for both
        if sz_b < sz_a:
            # if so swap things around
            sz_a, sz_b = sz_b, sz_a
            sysb = [i for i in range(len(dims)) if i not in sysa]
            sysa = sysb

        # check whether to use approx lanczos method
        if (approx_thresh is not None) and (sz_a >= approx_thresh):
            return approx_fn(psi_ab, dims, sysa, **approx_opts)

        rho_a = ptr(psi_ab, dims, sysa)
        return exact_fn(rho_a)

    return bipartite_spectral_fn
