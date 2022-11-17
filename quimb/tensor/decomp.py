import functools

import numpy as np
import scipy.sparse.linalg as spla
import scipy.linalg.interpolative as sli
from autoray import (
    do,
    reshape,
    dag,
    infer_backend,
    astype,
    get_dtype_name,
    compose,
    backend_like,
)

from ..core import njit, generated_jit
from ..linalg import base_linalg
from ..linalg import rand_linalg


# some convenience functions for multiplying diagonals


def rdmul(x, d):
    """Right-multiplication a matrix by a vector representing a diagonal."""
    return x * reshape(d, (1, -1))


def rddiv(x, d):
    """Right-multiplication of a matrix by a vector representing an inverse
    diagonal.
    """
    return x / reshape(d, (1, -1))


def ldmul(d, x):
    """Left-multiplication a matrix by a vector representing a diagonal."""
    return x * reshape(d, (-1, 1))


def lddiv(d, x):
    """Left-multiplication of a matrix by a vector representing an inverse
    diagonal.
    """
    return x / reshape(d, (-1, 1))


@njit  # pragma: no cover
def dag_numba(x):
    return np.conjugate(x.T)


@njit  # pragma: no cover
def rdmul_numba(x, d):
    return x * d.reshape(1, -1)


@njit  # pragma: no cover
def rddiv_numba(x, d):
    return x / d.reshape(1, -1)


@njit  # pragma: no cover
def ldmul_numba(d, x):
    return x * d.reshape(-1, 1)


@njit  # pragma: no cover
def lddiv_numba(d, x):
    return x / d.reshape(-1, 1)


@compose
def sgn(x):
    """Get the 'sign' of ``x``, such that ``x / sgn(x)`` is real and
    non-negative.
    """
    return x / (do('abs', x) + (x == 0.0))


@sgn.register("numpy")
@generated_jit
def sgn_numba(x):
    from numba import types
    if hasattr(x, "dtype"):
        dtype = x.dtype
    else:
        dtype = x
    if isinstance(dtype, types.Complex):
        return lambda x: x / (np.abs(x) + (x == 0.0))
    else:
        return lambda x: np.sign(x) + (x == 0.0)


def _trim_and_renorm_svd_result(
    U, s, VH, cutoff, cutoff_mode, max_bond, absorb, renorm
):
    """Give full SVD decomposion result ``U``, ``s``, ``VH``, optionally trim,
    renormalize, and absorb the singular values. See ``svd_truncated`` for
    details.
    """
    if (cutoff > 0.0) or (renorm > 0):
        if cutoff_mode == 1:  # 'abs'
            n_chi = do("count_nonzero", s > cutoff)

        elif cutoff_mode == 2:  # 'rel'
            n_chi = do("count_nonzero", s > cutoff * s[0])

        elif cutoff_mode in (3, 4, 5, 6):
            if cutoff_mode in (3, 4):
                pow = 2
            else:
                pow = 1

            sp = s**pow
            csp = do("cumsum", sp, 0)
            tot = csp[-1]

            if cutoff_mode in (4, 6):
                n_chi = do("count_nonzero", csp < (1 - cutoff) * tot) + 1
            else:
                n_chi = do("count_nonzero", (tot - csp) > cutoff) + 1

        n_chi = max(n_chi, 1)
        if max_bond > 0:
            n_chi = min(n_chi, max_bond)

    elif max_bond > 0:
        # only maximum bond specified
        n_chi = max_bond
    else:
        # neither maximum bond dimension nor cutoff specified
        n_chi = s.shape[0]

    if n_chi < s.shape[0]:
        s = s[:n_chi]
        U = U[:, :n_chi]
        VH = VH[:n_chi, :]

        if renorm > 0:
            norm = (tot / csp[n_chi - 1]) ** (1 / pow)
            s *= norm

    # XXX: tensorflow can't multiply mixed dtypes
    if infer_backend(s) == "tensorflow":
        dtype = get_dtype_name(U)
        if "complex" in dtype:
            s = astype(s, dtype)

    if absorb is None:
        return U, s, VH
    if absorb == -1:
        U = rdmul(U, s)
    elif absorb == 1:
        VH = ldmul(s, VH)
    else:
        s = do("sqrt", s)
        U = rdmul(U, s)
        VH = ldmul(s, VH)

    return U, None, VH


@compose
def svd_truncated(
    x,
    cutoff=-1.0,
    cutoff_mode=3,
    max_bond=-1,
    absorb=0,
    renorm=0,
    backend=None,
):
    """Truncated svd or raw array ``x``.

    Parameters
    ----------
    cutoff : float
        Singular value cutoff threshold, if ``cutoff <= 0.0``, then only
        ``max_bond`` is used.
    cutoff_mode : {1, 2, 3, 4, 5, 6}
        How to perform the trim:

            - 1: ['abs'], trim values below ``cutoff``
            - 2: ['rel'], trim values below ``s[0] * cutoff``
            - 3: ['sum2'], trim s.t. ``sum(s_trim**2) < cutoff``.
            - 4: ['rsum2'], trim s.t. ``sum(s_trim**2) < sum(s**2) * cutoff``.
            - 5: ['sum1'], trim s.t. ``sum(s_trim**1) < cutoff``.
            - 6: ['rsum1'], trim s.t. ``sum(s_trim**1) < sum(s**1) * cutoff``.

    max_bond : int
        An explicit maximum bond dimension, use -1 for none.
    absorb : {-1, 0, 1, None}
        How to absorb the singular values. -1: left, 0: both, 1: right and
        None: don't absorb (return).
    renorm : {0, 1}
        Whether to renormalize the singular values (depends on `cutoff_mode`).
    """
    with backend_like(backend):
        U, s, VH = do("linalg.svd", x)
        return _trim_and_renorm_svd_result(
            U, s, VH, cutoff, cutoff_mode, max_bond, absorb, renorm
        )


@njit(["i4(f4[:], f4, i4)", "i4(f8[:], f8, i4)"])  # pragma: no cover
def _compute_number_svals_to_keep_numba(s, cutoff, cutoff_mode):
    """Find the number of singular values to keep of ``s`` given ``cutoff`` and
    ``cutoff_mode``.
    """
    if cutoff_mode == 1:  # 'abs'
        n_chi = np.sum(s > cutoff)

    elif cutoff_mode == 2:  # 'rel'
        n_chi = np.sum(s > cutoff * s[0])

    elif cutoff_mode in (3, 4, 5, 6):
        if cutoff_mode in (3, 4):
            pow = 2
        else:
            pow = 1

        target = cutoff
        if cutoff_mode in (4, 6):
            target *= np.sum(s**pow)

        n_chi = s.size
        ssum = 0.0
        for i in range(s.size - 1, -1, -1):
            s2 = s[i] ** pow
            if not np.isnan(s2):
                ssum += s2
            if ssum > target:
                break
            n_chi -= 1

    return max(n_chi, 1)


@njit(["f4(f4[:], i4, f4)", "f8(f8[:], i4, f8)"])  # pragma: no cover
def _compute_svals_renorm_factor_numba(s, n_chi, renorm):
    """Find the normalization constant for ``s`` such that the new sum squared
    of the ``n_chi`` largest values equals the sum squared of all the old ones.
    """
    s_tot_keep = 0.0
    s_tot_lose = 0.0
    for i in range(s.size):
        s2 = s[i] ** renorm
        if not np.isnan(s2):
            if i < n_chi:
                s_tot_keep += s2
            else:
                s_tot_lose += s2
    return ((s_tot_keep + s_tot_lose) / s_tot_keep) ** (1 / renorm)


@njit  # pragma: no cover
def _trim_and_renorm_svd_result_numba(
    U, s, VH, cutoff, cutoff_mode, max_bond, absorb, renorm
):
    """Accelerate version of ``_trim_and_renorm_svd_result``."""
    if (cutoff > 0.0) or (renorm > 0):
        n_chi = _compute_number_svals_to_keep_numba(s, cutoff, cutoff_mode)

        if max_bond > 0:
            n_chi = min(n_chi, max_bond)

        if n_chi < s.size:
            if renorm > 0:
                s = s[:n_chi] * _compute_svals_renorm_factor_numba(
                    s, n_chi, renorm
                )
            else:
                s = s[:n_chi]

            U = U[:, :n_chi]
            VH = VH[:n_chi, :]

    elif (max_bond != -1) and (max_bond < s.shape[0]):
        U = U[:, :max_bond]
        s = s[:max_bond]
        VH = VH[:max_bond, :]

    s = np.ascontiguousarray(s)

    if absorb is None:
        return U, s, VH
    elif absorb == -1:
        U = rdmul_numba(U, s)
    elif absorb == 1:
        VH = ldmul_numba(s, VH)
    else:
        s **= 0.5
        U = rdmul_numba(U, s)
        VH = ldmul_numba(s, VH)

    return U, None, VH


@svd_truncated.register("numpy")
@njit  # pragma: no cover
def svd_truncated_numba(
    x, cutoff=-1.0, cutoff_mode=3, max_bond=-1, absorb=0, renorm=0
):
    """Accelerated version of ``svd_truncated`` for numpy arrays."""
    U, s, VH = np.linalg.svd(x, full_matrices=False)
    return _trim_and_renorm_svd_result_numba(
        U, s, VH, cutoff, cutoff_mode, max_bond, absorb, renorm
    )


def svdvals(x):
    """SVD-decomposition, but return singular values only."""
    return np.linalg.svd(x, full_matrices=False, compute_uv=False)


@njit  # pragma: no cover
def _svd_via_eig_truncated_numba(
    x, cutoff=-1.0, cutoff_mode=3, max_bond=-1, absorb=0, renorm=0
):
    """SVD-split via eigen-decomposition."""
    if x.shape[0] > x.shape[1]:
        # Get sU, V
        s2, V = np.linalg.eigh(dag_numba(x) @ x)
        U = x @ V
        VH = dag_numba(V)
        # small negative eigenvalues turn into nan when sqrtd
        s2[s2 < 0.0] = 0.0
        s = np.sqrt(s2)
        U = rddiv_numba(U, s)
    else:
        # Get U, sV
        s2, U = np.linalg.eigh(x @ dag_numba(x))
        VH = dag_numba(U) @ x
        s2[s2 < 0.0] = 0.0
        s = np.sqrt(s2)
        VH = lddiv_numba(s, VH)

    # we need singular values and vectors in descending order
    U, s, VH = U[:, ::-1], s[::-1], VH[::-1, :]

    return _trim_and_renorm_svd_result_numba(
        U, s, VH, cutoff, cutoff_mode, max_bond, absorb, renorm
    )


def svd_via_eig_truncated(
    x, cutoff=-1.0, cutoff_mode=3, max_bond=-1, absorb=0, renorm=0
):
    if isinstance(x, np.ndarray):
        return _svd_via_eig_truncated_numba(
            x, cutoff, cutoff_mode, max_bond, absorb, renorm
        )

    if x.shape[0] > x.shape[1]:
        # Get sU, V
        s2, V = do("linalg.eigh", dag(x) @ x)
        U = x @ V
        VH = dag(V)
        # small negative eigenvalues turn into nan when sqrtd
        s2 = do("clip", s2, 0.0, None)
        s = do("sqrt", s2)
        U = rddiv(U, s)
    else:
        # Get U, sV
        s2, U = do("linalg.eigh", x @ dag(x))
        VH = dag(U) @ x
        s2 = do("clip", s2, 0.0, None)
        s = do("sqrt", s2)
        VH = lddiv(s, VH)

    # we need singular values and vectors in descending order
    U, s, VH = do("flip", U, (1,)), do("flip", s, (0,)), do("flip", VH, (0,))

    return _trim_and_renorm_svd_result(
        U, s, VH, cutoff, cutoff_mode, max_bond, absorb, renorm
    )


@njit  # pragma: no cover
def svdvals_eig(x):  # pragma: no cover
    """SVD-decomposition via eigen, but return singular values only."""
    if x.shape[0] > x.shape[1]:
        s2 = np.linalg.eigvalsh(dag_numba(x) @ x)
    else:
        s2 = np.linalg.eigvalsh(x @ dag_numba(x))

    s2[s2 < 0.0] = 0.0
    return s2[::-1] ** 0.5


@njit  # pragma: no cover
def eigh(x, cutoff=-1.0, cutoff_mode=3, max_bond=-1, absorb=0, renorm=0):
    """SVD-decomposition, using hermitian eigen-decomposition, only works if
    ``x`` is hermitian.
    """
    s, U = np.linalg.eigh(x)
    s, U = s[::-1], U[:, ::-1]  # make sure largest singular value first

    V = ldmul_numba(sgn_numba(s), dag_numba(U))
    s = np.abs(s)
    return _trim_and_renorm_svd_result_numba(
        U, s, V, cutoff, cutoff_mode, max_bond, absorb, renorm
    )


def _choose_k(x, cutoff, max_bond):
    """Choose the number of singular values to target."""
    d = min(x.shape)

    if cutoff != 0.0:
        k = rand_linalg.estimate_rank(
            x, cutoff, k_max=None if max_bond < 0 else max_bond
        )
    else:
        k = min(d, max_bond)

    # if computing more than half of spectrum then just use dense method
    return "full" if k > d // 2 else k


def svds(x, cutoff=0.0, cutoff_mode=2, max_bond=-1, absorb=0, renorm=0):
    """SVD-decomposition using iterative methods. Allows the
    computation of only a certain number of singular values, e.g. max_bond,
    from the get-go, and is thus more efficient. Can also supply
    ``scipy.sparse.linalg.LinearOperator``.
    """
    k = _choose_k(x, cutoff, max_bond)

    if k == "full":
        if not isinstance(x, np.ndarray):
            x = x.to_dense()
        return svd_truncated(x, cutoff, cutoff_mode, max_bond, absorb)

    U, s, VH = base_linalg.svds(x, k=k)
    return _trim_and_renorm_svd_result_numba(
        U, s, VH, cutoff, cutoff_mode, max_bond, absorb, renorm
    )


def isvd(x, cutoff=0.0, cutoff_mode=2, max_bond=-1, absorb=0, renorm=0):
    """SVD-decomposition using interpolative matrix random methods. Allows the
    computation of only a certain number of singular values, e.g. max_bond,
    from the get-go, and is thus more efficient. Can also supply
    ``scipy.sparse.linalg.LinearOperator``.
    """
    k = _choose_k(x, cutoff, max_bond)

    if k == "full":
        if not isinstance(x, np.ndarray):
            x = x.to_dense()
        return svd_truncated(x, cutoff, cutoff_mode, max_bond, absorb)

    U, s, V = sli.svd(x, k)
    VH = dag_numba(V)
    return _trim_and_renorm_svd_result_numba(
        U, s, VH, cutoff, cutoff_mode, max_bond, absorb, renorm
    )


def _rsvd_numpy(x, cutoff=0.0, cutoff_mode=2, max_bond=-1, absorb=0, renorm=0):
    if max_bond > 0:
        if cutoff > 0.0:
            # adapt and block
            U, s, VH = rand_linalg.rsvd(x, cutoff, k_max=max_bond)
        else:
            U, s, VH = rand_linalg.rsvd(x, max_bond)
    else:
        U, s, VH = rand_linalg.rsvd(x, cutoff)

    return _trim_and_renorm_svd_result_numba(
        U, s, VH, cutoff, cutoff_mode, max_bond, absorb, renorm
    )


def rsvd(x, cutoff=0.0, cutoff_mode=2, max_bond=-1, absorb=0, renorm=0):
    """SVD-decomposition using randomized methods (due to Halko). Allows the
    computation of only a certain number of singular values, e.g. max_bond,
    from the get-go, and is thus more efficient. Can also supply
    ``scipy.sparse.linalg.LinearOperator``.
    """
    if isinstance(x, (np.ndarray, spla.LinearOperator)):
        return _rsvd_numpy(x, cutoff, cutoff_mode, max_bond, absorb, renorm)

    U, s, VH = do("linalg.rsvd", x, max_bond)
    return _trim_and_renorm_svd_result(
        U, s, VH, cutoff, cutoff_mode, max_bond, absorb, renorm
    )


def eigsh(x, cutoff=0.0, cutoff_mode=2, max_bond=-1, absorb=0, renorm=0):
    """SVD-decomposition using iterative hermitian eigen decomp, thus assuming
    that ``x`` is hermitian. Allows the computation of only a certain number of
    singular values, e.g. max_bond, from the get-go, and is thus more
    efficient. Can also supply ``scipy.sparse.linalg.LinearOperator``.
    """
    k = _choose_k(x, cutoff, max_bond)

    if k == "full":
        if not isinstance(x, np.ndarray):
            x = x.to_dense()
        return eigh(x, cutoff, cutoff_mode, max_bond, absorb)

    s, U = base_linalg.eigh(x, k=k)
    s, U = s[::-1], U[:, ::-1]  # make sure largest singular value first
    V = ldmul_numba(sgn(s), dag_numba(U))
    s = np.abs(s)
    return _trim_and_renorm_svd_result_numba(
        U, s, V, cutoff, cutoff_mode, max_bond, absorb, renorm
    )


@compose
def qr_stabilized(x, backend=None):
    """QR-decomposition, with stabilized R factor."""
    with backend_like(backend):
        Q, R = do("linalg.qr", x)
        # stabilize the diagonal of R
        rd = do("diag", R)
        s = sgn(rd)
        Q = rdmul(Q, s)
        R = ldmul(s, R)
        return Q, None, R


@qr_stabilized.register("numpy")
@njit  # pragma: no cover
def qr_stabilized_numba(x):
    Q, R = np.linalg.qr(x)
    for i in range(R.shape[0]):
        rii = R[i, i]
        si = sgn_numba(rii)
        if si != 1.0:
            Q[:, i] *= si
            R[i, i:] *= si
    return Q, None, R


@compose
def lq_stabilized(x, backend=None):
    with backend_like(backend):
        Q, _, L = qr_stabilized(do("transpose", x))
        return do("transpose", L), None, do("transpose", Q)


@lq_stabilized.register("numpy")
@njit  # pragma: no cover
def lq_stabilized_numba(x):
    Q, _, L = qr_stabilized_numba(x.T)
    return L.T, None, Q.T



@njit  # pragma: no cover
def _cholesky_numba(x, cutoff=-1, cutoff_mode=3, max_bond=-1, absorb=0):
    """SVD-decomposition, using cholesky decomposition, only works if
    ``x`` is positive definite.
    """
    L = np.linalg.cholesky(x)
    return L, None, dag_numba(L)


def cholesky(x, cutoff=-1, cutoff_mode=3, max_bond=-1, absorb=0):
    try:
        return _cholesky_numba(x, cutoff, cutoff_mode, max_bond, absorb)
    except np.linalg.LinAlgError as e:
        if cutoff < 0:
            raise e
        # try adding cutoff identity - assuming it is approx allowable error
        xi = x + 2 * cutoff * np.eye(x.shape[0])
        return _cholesky_numba(xi, cutoff, cutoff_mode, max_bond, absorb)


# ------ similarity transforms for compressing effective environments ------- #


def _similarity_compress_eig(X, max_bond, renorm):
    # eigen decompose X -> V w V^-1
    el, ev = do("linalg.eig", X)
    evi = do("linalg.inv", ev)

    # choose largest abs value eigenpairs
    sel = do("argsort", do("abs", el))[-max_bond:]
    Cl = ev[:, sel]
    Cr = evi[sel, :]

    if renorm:
        trace_old = do("sum", el)
        trace_new = do("sum", el[sel])
        Cl = Cl * trace_old / trace_new

    return Cl, Cr


@njit(
    [
        "(c8[:,:], i4, i4)",
        "(c16[:,:], i4, i4)",
    ]
)  # pragma: no cover
def _similarity_compress_eig_numba(X, max_bond, renorm):
    el, ev = np.linalg.eig(X)
    evi = np.linalg.inv(ev)
    sel = np.argsort(np.abs(el))[-max_bond:]
    Cl = ev[:, sel]
    Cr = evi[sel, :]
    if renorm:
        trace_old = np.sum(el)
        trace_new = np.sum(el[sel])
        Cl = Cl * trace_old / trace_new
    return Cl, Cr


def _similarity_compress_eigh(X, max_bond, renorm):
    XX = (X + dag(X)) / 2
    el, ev = do("linalg.eigh", XX)
    sel = do("argsort", do("abs", el))[-max_bond:]
    Cl = ev[:, sel]
    Cr = dag(Cl)
    if renorm:
        trace_old = do("trace", X)
        trace_new = do("trace", Cr @ (X @ Cl))
        Cl = Cl * trace_old / trace_new
    return Cl, Cr


@njit  # pragma: no cover
def _similarity_compress_eigh_numba(X, max_bond, renorm):
    XX = (X + dag_numba(X)) / 2
    el, ev = np.linalg.eigh(XX)
    sel = np.argsort(-np.abs(el))[:max_bond]
    Cl = ev[:, sel]
    Cr = dag_numba(Cl)
    if renorm:
        trace_old = np.trace(X)
        trace_new = np.trace(Cr @ (X @ Cl))
        Cl = Cl * trace_old / trace_new
    return Cl, Cr


def _similarity_compress_svd(X, max_bond, renorm, asymm):
    U, _, VH = do("linalg.svd", X)
    U = U[:, :max_bond]

    Cl = U
    if asymm:
        VH = VH[:max_bond, :]
        Cr = dag(U)
        Cl = dag(VH)
    else:
        Cr = dag(U)

    if renorm:
        # explicitly maintain trace value
        trace_old = do("trace", X)
        trace_new = do("trace", Cr @ (X @ Cl))
        Cl = Cl * (trace_old / trace_new)

    return Cl, Cr


@njit  # pragma: no cover
def _similarity_compress_svd_numba(X, max_bond, renorm, asymm):
    U, _, VH = np.linalg.svd(X)
    U = U[:, :max_bond]
    Cl = U

    if asymm:
        VH = VH[:max_bond, :]
        Cr = dag_numba(U)
        Cl = dag_numba(VH)
    else:
        Cr = dag_numba(U)

    if renorm:
        trace_old = np.trace(X)
        trace_new = np.trace(Cr @ (X @ Cl))
        Cl = Cl * trace_old / trace_new
    return Cl, Cr


def _similarity_compress_biorthog(X, max_bond, renorm):
    U, s, VH = do("linalg.svd", X)

    B = U[:, :max_bond]
    AH = VH[:max_bond, :]

    Uab, sab, VHab = do("linalg.svd", AH @ B)
    sab = (sab + 1e-12 * do("max", sab)) ** -0.5
    sab_inv = do("reshape", sab, (1, -1))
    P = Uab * sab_inv
    Q = dag(VHab) * sab_inv

    Cl = B @ Q
    Cr = dag(P) @ AH

    if renorm:
        trace_old = do("trace", X)
        trace_new = do("trace", Cr @ (X @ Cl))
        Cl = Cl * trace_old / trace_new

    return Cl, Cr


@njit  # pragma: no cover
def _similarity_compress_biorthog_numba(X, max_bond, renorm):
    U, s, VH = np.linalg.svd(X)

    B = U[:, :max_bond]
    AH = VH[:max_bond, :]

    Uab, sab, VHab = np.linalg.svd(AH @ B)

    # smudge factor
    sab += 1e-12 * np.max(sab)
    sab **= -0.5

    sab_inv = sab.reshape((1, -1))
    P = Uab * sab_inv
    Q = dag_numba(VHab) * sab_inv

    Cl = B @ Q
    Cr = dag_numba(P) @ AH

    if renorm:
        trace_old = np.trace(X)
        trace_new = np.trace(Cr @ (X @ Cl))
        Cl = Cl * trace_old / trace_new

    return Cl, Cr


_similarity_compress_fns = {
    ("eig", False): _similarity_compress_eig,
    ("eig", True): _similarity_compress_eig_numba,
    ("eigh", False): _similarity_compress_eigh,
    ("eigh", True): _similarity_compress_eigh_numba,
    ("svd", False): functools.partial(_similarity_compress_svd, asymm=0),
    ("svd", True): functools.partial(_similarity_compress_svd_numba, asymm=0),
    ("biorthog", False): _similarity_compress_biorthog,
    ("biorthog", True): _similarity_compress_biorthog_numba,
}


def similarity_compress(X, max_bond, renorm=True, method="eigh"):
    if method == "eig":
        if get_dtype_name(X) == "float64":
            X = astype(X, "complex128")
        elif get_dtype_name(X) == "float32":
            X = astype(X, "complex64")

    isnumpy = isinstance(X, np.ndarray)
    # if isnumpy:
    #     X = np.ascontiguousarray(X)
    fn = _similarity_compress_fns[method, isnumpy]
    return fn(X, max_bond, int(renorm))


@compose
def isometrize_qr(x, backend=None):
    """Perform isometrization using the QR decomposition.
    """
    with backend_like(backend):
        fat = x.shape[0] < x.shape[1]
        if fat:
            x = do("transpose", x)
        Q, R = do('linalg.qr', x)
        # stabilize qr by fixing diagonal of R in canonical, positive form (we
        # don't actaully do anything to R, just absorb the necessary sign -> Q)
        rd = do('diag', R)
        s = do('sign', rd) + (rd == 0)
        Q = Q * reshape(s, (1, -1))
        if fat:
            Q = do("transpose", Q)
        return Q


@compose
def isometrize_svd(x, backend=None):
    with backend_like(backend):
        fat = x.shape[0] < x.shape[1]
        if fat:
            x = do("transpose", x)

        Q = do('linalg.svd', x)[0]
        if fat:
            Q = do("transpose", Q)

        return Q


@compose
def isometrize_exp(x, backend):
    r"""Perform isometrization using the using anti-symmetric matrix
    exponentiation.

    .. math::

            U_A = \exp{A - A^\dagger}

    If ``x`` is rectangular it is completed with zeros first.
    """
    with backend_like(backend):
        m, n = x.shape
        d = max(m, n)
        x = do(
            'pad', x, [[0, d - m], [0, d - n]], 'constant', constant_values=0.0
        )
        expx = do('linalg.expm', x - dag(x))
        return expx[:m, :n]


@compose
def isometrize_modified_gram_schmidt(A, backend=None):
    """Perform isometrization explicitly using the modified Gram Schmidt
    procedure.
    """
    with backend_like(backend):
        m, n = A.shape

        thin = m > n
        if thin:
            A = do('transpose', A)

        Q = []
        for j in range(0, min(m, n)):

            q = A[j, :]
            for i in range(0, j):
                rij = do('tensordot', do('conj', Q[i]), q, 1)
                q = q - rij * Q[i]

            Q.append(q / do('linalg.norm', q, 2))

        Q = do('stack', tuple(Q), axis=0, like=A)

        if thin:
            Q = do('transpose', Q)

        return Q


_ISOMETRIZE_METHODS = {
    'qr': isometrize_qr,
    'svd': isometrize_svd,
    'exp': isometrize_exp,
    'mgs': isometrize_modified_gram_schmidt,
}


def isometrize(x, method='qr'):
    """Generate a isometric (or unitary if square) matrix from array ``x``.

    Parameters
    ----------
    x : array
        The matrix to project into isometrix form.
    method : {'qr', 'svd', 'exp', 'mgs'}, optional
        The method used to generate the isometry. Note ``'qr'`` is the fastest
        and most robust but, for example, some libraries cannot back-propagate
        through it.
    """
    return _ISOMETRIZE_METHODS[method](x)
