"""Functions for decomposing and projecting matrices."""

import functools
import operator
import warnings

import cotengra as ctg
import numpy as np
import scipy.linalg as scla
import scipy.linalg.interpolative as sli
import scipy.sparse.linalg as spla
from autoray import (
    astype,
    backend_like,
    compose,
    dag,
    do,
    get_dtype_name,
    get_lib_fn,
    infer_backend,
    lazy,
    reshape,
)

from ..core import njit
from ..linalg import base_linalg, rand_linalg
from .array_ops import isblocksparse, isfermionic


# some convenience functions for multiplying diagonals


@compose
def rdmul(x, d):
    """Right-multiplication a matrix by a vector representing a diagonal."""
    return x * d[None, :]


@compose
def rddiv(x, d):
    """Right-multiplication of a matrix by a vector representing an inverse
    diagonal.
    """
    return x / d[None, :]


@compose
def ldmul(d, x):
    """Left-multiplication a matrix by a vector representing a diagonal."""
    return x * d[:, None]


@compose
def lddiv(d, x):
    """Left-multiplication of a matrix by a vector representing an inverse
    diagonal.
    """
    return x / d[:, None]


@njit  # pragma: no cover
def dag_numba(x):
    return np.conjugate(x.T)


@njit  # pragma: no cover
def rdmul_numba(x, d):
    return x * d[None, :]


@njit  # pragma: no cover
def rddiv_numba(x, d):
    return x / d[None, :]


@njit  # pragma: no cover
def ldmul_numba(d, x):
    return x * d[:, None]


@njit  # pragma: no cover
def lddiv_numba(d, x):
    return x / d[:, None]


@compose
def sgn(x):
    """Get the 'sign' of ``x``, such that ``x / sgn(x)`` is real and
    non-negative.
    """
    x0 = do("equal", x, 0.0)
    return (x + x0) / (do("abs", x) + x0)


@sgn.register("numpy")
@njit  # pragma: no cover
def sgn_numba(x):
    x0 = x == 0.0
    return (x + x0) / (np.abs(x) + x0)


@sgn.register("tensorflow")
def sgn_tf(x):
    with backend_like(x):
        x0 = do("cast", do("equal", x, 0.0), x.dtype)
        xa = do("cast", do("abs", x), x.dtype)
        return (x + x0) / (xa + x0)


_CUTOFF_MODE_MAP = {
    1: 1,
    "abs": 1,
    2: 2,
    "rel": 2,
    3: 3,
    "sum2": 3,
    4: 4,
    "rsum2": 4,
    5: 5,
    "sum1": 5,
    6: 6,
    "rsum1": 6,
}


_ABSORB_MAP = {
    None: None,
    -1: -1,
    0: 0,
    1: 1,
    "left": -1,
    "both": 0,
    "right": 1,
}


def _trim_and_renorm_svd_result(
    U,
    s,
    VH,
    cutoff,
    cutoff_mode,
    max_bond,
    absorb,
    renorm,
    use_abs=False,
):
    """Give full SVD decomposion result ``U``, ``s``, ``VH``, optionally trim,
    renormalize, and absorb the singular values. See ``svd_truncated`` for
    details.
    """
    if use_abs:
        sabs = do("abs", s)
    else:
        # assume already all positive
        sabs = s

    d = do("shape", sabs)[0]

    if (cutoff > 0.0) or (renorm > 0):
        if cutoff_mode == 1:  # 'abs'
            n_chi = do("count_nonzero", sabs > cutoff)

        elif cutoff_mode == 2:  # 'rel'
            n_chi = do("count_nonzero", sabs > cutoff * sabs[0])

        elif cutoff_mode in (3, 4, 5, 6):
            if cutoff_mode in (3, 4):
                pow = 2
                sp = sabs**pow
            else:
                pow = 1
                sp = sabs

            csp = do("cumsum", sp, 0)
            tot = csp[-1]

            if cutoff_mode in (4, 6):
                n_chi = do("count_nonzero", csp < (1 - cutoff) * tot) + 1
            else:
                n_chi = do("count_nonzero", (tot - csp) > cutoff) + 1

        n_chi = max(n_chi, 1)
        if max_bond > 0:
            # need to take both cutoff and max bond into account
            n_chi = min(n_chi, max_bond)

    elif max_bond > 0:
        # only maximum bond specified
        n_chi = max_bond
    else:
        # neither maximum bond dimension nor cutoff specified
        n_chi = d

    if n_chi < d:
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
    cutoff_mode=4,
    max_bond=-1,
    absorb=0,
    renorm=0,
    backend=None,
):
    """Truncated svd or raw array ``x``.

    Parameters
    ----------
    cutoff : float, optional
        Singular value cutoff threshold, if ``cutoff <= 0.0``, then only
        ``max_bond`` is used.
    cutoff_mode : {1, 2, 3, 4, 5, 6}, optional
        How to perform the trim:

            - 1: ['abs'], trim values below ``cutoff``
            - 2: ['rel'], trim values below ``s[0] * cutoff``
            - 3: ['sum2'], trim s.t. ``sum(s_trim**2) < cutoff``.
            - 4: ['rsum2'], trim s.t. ``sum(s_trim**2) < sum(s**2) * cutoff``.
            - 5: ['sum1'], trim s.t. ``sum(s_trim**1) < cutoff``.
            - 6: ['rsum1'], trim s.t. ``sum(s_trim**1) < sum(s**1) * cutoff``.

    max_bond : int, optional
        An explicit maximum bond dimension, use -1 for none.
    absorb : {-1, 0, 1, None}, optional
        How to absorb the singular values. -1: left, 0: both, 1: right and
        None: don't absorb (return).
    renorm : {0, 1}, optional
        Whether to renormalize the singular values (depends on `cutoff_mode`).
    """
    absorb = _ABSORB_MAP[absorb]
    cutoff_mode = _CUTOFF_MODE_MAP[cutoff_mode]

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

    raise_power = renorm >= 2

    for i in range(s.size):
        s2 = s[i]
        if raise_power:
            s2 **= renorm

        if not np.isnan(s2):
            if i < n_chi:
                s_tot_keep += s2
            else:
                s_tot_lose += s2

    f = (s_tot_keep + s_tot_lose) / s_tot_keep
    if raise_power:
        f **= 1 / renorm

    return f


@njit  # pragma: no cover
def _trim_and_renorm_svd_result_numba(
    U,
    s,
    VH,
    cutoff,
    cutoff_mode,
    max_bond,
    absorb,
    renorm,
    use_abs=False,
):
    """Accelerated version of ``_trim_and_renorm_svd_result``."""

    if use_abs:
        sabs = np.abs(s)
    else:
        sabs = s

    if (cutoff > 0.0) or (renorm > 0):
        # need to dynamically truncate
        n_chi = _compute_number_svals_to_keep_numba(sabs, cutoff, cutoff_mode)

        if max_bond > 0:
            n_chi = min(n_chi, max_bond)

        if n_chi < s.size:
            if renorm > 0:
                f = _compute_svals_renorm_factor_numba(sabs, n_chi, renorm)
                s = s[:n_chi] * f
            else:
                s = s[:n_chi]

            U = U[:, :n_chi]
            VH = VH[:n_chi, :]

    elif (max_bond != -1) and (max_bond < s.shape[0]):
        # only maximum bond specified
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


@njit  # pragma: no cover
def svd_truncated_numba(
    x, cutoff=-1.0, cutoff_mode=4, max_bond=-1, absorb=0, renorm=0
):
    """Accelerated version of ``svd_truncated`` for numpy arrays."""
    U, s, VH = np.linalg.svd(x, full_matrices=False)

    return _trim_and_renorm_svd_result_numba(
        U, s, VH, cutoff, cutoff_mode, max_bond, absorb, renorm
    )


@svd_truncated.register("numpy")
def svd_truncated_numpy(
    x, cutoff=-1.0, cutoff_mode=4, max_bond=-1, absorb=0, renorm=0
):
    """Numpy version of ``svd_truncated``, trying the accelerated version
    first, then falling back to the more stable scipy version.
    """
    absorb = _ABSORB_MAP[absorb]
    cutoff_mode = _CUTOFF_MODE_MAP[cutoff_mode]
    try:
        return svd_truncated_numba(
            x, cutoff, cutoff_mode, max_bond, absorb, renorm
        )
    except ValueError as e:  # pragma: no cover
        warnings.warn(f"Got: {e}, falling back to scipy gesvd driver.")
        U, s, VH = scla.svd(x, full_matrices=False, lapack_driver="gesvd")
        return _trim_and_renorm_svd_result_numba(
            U, s, VH, cutoff, cutoff_mode, max_bond, absorb, renorm
        )


@svd_truncated.register("autoray.lazy")
@lazy.core.lazy_cache("svd_truncated")
def svd_truncated_lazy(
    x,
    cutoff=-1.0,
    cutoff_mode=4,
    max_bond=-1,
    absorb=0,
    renorm=0,
):
    if cutoff != 0.0:
        raise ValueError("Can't handle dynamic cutoffs in lazy mode.")

    m, n = x.shape
    k = min(m, n)
    if max_bond > 0:
        k = min(k, max_bond)

    lsvdt = x.to(
        fn=get_lib_fn(x.backend, "svd_truncated"),
        args=(x, cutoff, cutoff_mode, max_bond, absorb, renorm),
        shape=(3,),
    )

    U = lsvdt.to(operator.getitem, (lsvdt, 0), shape=(m, k))
    if absorb is None:
        s = lsvdt.to(operator.getitem, (lsvdt, 1), shape=(k,))
    else:
        s = None
    VH = lsvdt.to(operator.getitem, (lsvdt, 2), shape=(k, n))

    return U, s, VH


@compose
def lu_truncated(
    x,
    cutoff=-1.0,
    cutoff_mode=4,
    max_bond=-1,
    absorb=0,
    renorm=0,
    backend=None,
):
    if absorb != 0:
        raise NotImplementedError(
            f"Can't handle absorb{absorb} in lu_truncated."
        )
    elif renorm != 0:
        raise NotImplementedError(
            f"Can't handle renorm={renorm} in lu_truncated."
        )
    elif max_bond != -1:
        # use argsort(sl * su) to handle this?
        raise NotImplementedError(
            f"Can't handle max_bond={max_bond} in lu_truncated."
        )

    with backend_like(backend):
        PL, U = do("scipy.linalg.lu", x, permute_l=True)

        sl = do("sum", do("abs", PL), axis=0)
        su = do("sum", do("abs", U), axis=1)

        if cutoff_mode == 2:
            abs_cutoff_l = cutoff * do("max", sl)
            abs_cutoff_u = cutoff * do("max", su)
        elif cutoff_mode == 1:
            abs_cutoff_l = abs_cutoff_u = cutoff
        else:
            raise NotImplementedError(
                f"Can't handle cutoff_mode={cutoff_mode} in lu_truncated."
            )

        idx = (sl > abs_cutoff_l) & (su > abs_cutoff_u)

        PL = PL[:, idx]
        U = U[idx, :]

        return PL, None, U


def svdvals(x):
    """SVD-decomposition, but return singular values only."""
    return np.linalg.svd(x, full_matrices=False, compute_uv=False)


@njit  # pragma: no cover
def _svd_via_eig_truncated_numba(
    x, cutoff=-1.0, cutoff_mode=4, max_bond=-1, absorb=0, renorm=0
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
    x, cutoff=-1.0, cutoff_mode=4, max_bond=-1, absorb=0, renorm=0
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


@compose
def eigh_truncated(
    x,
    cutoff=-1.0,
    cutoff_mode=4,
    max_bond=-1,
    absorb=0,
    renorm=0,
    positive=0,
    backend=None,
):
    with backend_like(backend):
        s, U = do("linalg.eigh", x)

        # make sure largest singular value first
        if not positive:
            idx = do("argsort", -do("abs", s))
            s, U = s[idx], U[:, idx]
        else:
            # assume all positive, simply reverse
            s = do("flip", s)
            U = do("flip", U, axis=1)

        VH = dag(U)

        # XXX: better to absorb phase in V and return positive 'values'?
        # V = ldmul(sgn(s), dag(U))
        # s = do("abs", s)

        return _trim_and_renorm_svd_result(
            U,
            s,
            VH,
            cutoff,
            cutoff_mode,
            max_bond,
            absorb,
            renorm,
            use_abs=True,
        )


@eigh_truncated.register("numpy")
@njit  # pragma: no cover
def eigh_truncated_numba(
    x,
    cutoff=-1.0,
    cutoff_mode=4,
    max_bond=-1,
    absorb=0,
    renorm=0,
    positive=0,
):
    """SVD-decomposition, using hermitian eigen-decomposition, only works if
    ``x`` is hermitian.
    """
    s, U = np.linalg.eigh(x)

    # make sure largest singular value first
    if not positive:
        k = np.argsort(-np.abs(s))
        s, U = s[k], U[:, k]
    else:
        s = s[::-1]
        U = U[:, ::-1]
    VH = dag_numba(U)

    # XXX: better to absorb phase in V and return positive 'values'?
    # VH = ldmul_numba(sgn_numba(s), dag_numba(U))
    # s = np.abs(s)

    return _trim_and_renorm_svd_result_numba(
        U, s, VH, cutoff, cutoff_mode, max_bond, absorb, renorm, use_abs=True
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


def svds(x, cutoff=0.0, cutoff_mode=4, max_bond=-1, absorb=0, renorm=0):
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


def isvd(x, cutoff=0.0, cutoff_mode=4, max_bond=-1, absorb=0, renorm=0):
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


def _rsvd_numpy(x, cutoff=0.0, cutoff_mode=4, max_bond=-1, absorb=0, renorm=0):
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


def rsvd(x, cutoff=0.0, cutoff_mode=4, max_bond=-1, absorb=0, renorm=0):
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


def eigsh(x, cutoff=0.0, cutoff_mode=4, max_bond=-1, absorb=0, renorm=0):
    """SVD-decomposition using iterative hermitian eigen decomp, thus assuming
    that ``x`` is hermitian. Allows the computation of only a certain number of
    singular values, e.g. max_bond, from the get-go, and is thus more
    efficient. Can also supply ``scipy.sparse.linalg.LinearOperator``.
    """
    k = _choose_k(x, cutoff, max_bond)

    if k == "full":
        if not isinstance(x, np.ndarray):
            x = x.to_dense()
        return eigh_truncated(x, cutoff, cutoff_mode, max_bond, absorb)

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
        Q = rdmul(Q, do("conj", s))
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
            Q[:, i] *= np.conj(si)
            R[i, i:] *= si
    return Q, None, R


@qr_stabilized.register("autoray.lazy")
@lazy.core.lazy_cache("qr_stabilized")
def qr_stabilized_lazy(x):
    m, n = x.shape
    k = min(m, n)
    lqrs = x.to(
        fn=get_lib_fn(x.backend, "qr_stabilized"),
        args=(x,),
        shape=(3,),
    )
    Q = lqrs.to(operator.getitem, (lqrs, 0), shape=(m, k))
    R = lqrs.to(operator.getitem, (lqrs, 2), shape=(k, n))
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
def _cholesky_numba(x, cutoff=-1, cutoff_mode=4, max_bond=-1, absorb=0):
    """SVD-decomposition, using cholesky decomposition, only works if
    ``x`` is positive definite.
    """
    L = np.linalg.cholesky(x)
    return L, None, dag_numba(L)


def cholesky(x, cutoff=-1, cutoff_mode=4, max_bond=-1, absorb=0):
    try:
        return _cholesky_numba(x, cutoff, cutoff_mode, max_bond, absorb)
    except np.linalg.LinAlgError as e:
        if cutoff < 0:
            raise e
        # try adding cutoff identity - assuming it is approx allowable error
        xi = x + 2 * cutoff * np.eye(x.shape[0])
        return _cholesky_numba(xi, cutoff, cutoff_mode, max_bond, absorb)


@compose
def polar_right(x):
    """Polar decomposition of ``x``."""
    W, s, VH = do("linalg.svd", x)
    U = W @ VH
    P = dag(VH) @ ldmul(s, VH)
    return U, None, P


@polar_right.register("numpy")
@njit  # pragma: no cover
def polar_right_numba(x):
    W, s, VH = np.linalg.svd(x, full_matrices=0)
    U = W @ VH
    P = dag_numba(VH) @ ldmul_numba(s, VH)
    return U, None, P


@compose
def polar_left(x):
    """Polar decomposition of ``x``."""
    W, s, VH = do("linalg.svd", x)
    U = W @ VH
    P = rdmul(W, s) @ dag(W)
    return P, None, U


@polar_left.register("numpy")
@njit  # pragma: no cover
def polar_left_numba(x):
    W, s, VH = np.linalg.svd(x, full_matrices=0)
    U = W @ VH
    P = rdmul_numba(W, s) @ dag_numba(W)
    return P, None, U


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


def similarity_compress(X, max_bond, renorm=False, method="eigh"):
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
    """Perform isometrization using the QR decomposition."""
    with backend_like(backend):
        Q, R = do("linalg.qr", x)
        # stabilize qr by fixing diagonal of R in canonical, positive form (we
        # don't actaully do anything to R, just absorb the necessary sign -> Q)
        rd = do("diag", R)
        s = do("sign", rd) + (rd == 0)
        Q = Q * reshape(s, (1, -1))
        return Q


@compose
def isometrize_svd(x, backend=None):
    """Perform isometrization using the SVD decomposition."""
    U, _, VH = do("linalg.svd", x, like=backend)
    return U @ VH


@compose
def isometrize_exp(x, backend):
    r"""Perform isometrization using anti-symmetric matrix exponentiation.

    .. math::

            U_A = \exp \left( X - X^\dagger \right)

    If ``x`` is rectangular it is completed with zeros first.
    """
    with backend_like(backend):
        m, n = x.shape
        d = max(m, n)
        x = do(
            "pad", x, [[0, d - m], [0, d - n]], "constant", constant_values=0.0
        )
        x = x - dag(x)
        Q = do("scipy.linalg.expm", x)
        return Q[:m, :n]


@compose
def isometrize_cayley(x, backend):
    r"""Perform isometrization using an anti-symmetric Cayley transform.

    .. math::

            U_A = (I + \dfrac{A}{2})(I - \dfrac{A}{2})^{-1}

    where :math:`A = X - X^\dagger`. If ``x`` is rectangular it is completed
    with zeros first.
    """
    with backend_like(backend):
        m, n = x.shape
        d = max(m, n)
        x = do(
            "pad", x, [[0, d - m], [0, d - n]], "constant", constant_values=0.0
        )
        x = x - dag(x)
        x = x / 2.0
        Id = do("eye", d, like=x)
        Q = do("linalg.solve", Id - x, Id + x)
        return Q[:m, :n]


@compose
def isometrize_modified_gram_schmidt(A, backend=None):
    """Perform isometrization explicitly using the modified Gram Schmidt
    procedure (this is slow but a useful reference).
    """
    with backend_like(backend):
        Q = []
        for j in range(A.shape[1]):
            q = A[:, j]
            for i in range(0, j):
                rij = do("tensordot", do("conj", Q[i]), q, 1)
                q = q - rij * Q[i]
            Q.append(q / do("linalg.norm", q))
        Q = do("stack", tuple(Q), axis=1)
        return Q


@compose
def isometrize_householder(X, backend=None):
    with backend_like(backend):
        X = do("tril", X, -1)
        tau = 2.0 / (1.0 + do("sum", do("conj", X) * X, 0))
        Q = do("linalg.householder_product", X, tau)
        return Q


def isometrize_torch_householder(x):
    """Isometrize ``x`` using the Householder reflection method, as implemented
    by the ``torch_householder`` package.
    """
    from torch_householder import torch_householder_orgqr

    return torch_householder_orgqr(x)


_ISOMETRIZE_METHODS = {
    "qr": isometrize_qr,
    "svd": isometrize_svd,
    "mgs": isometrize_modified_gram_schmidt,
    "exp": isometrize_exp,
    "cayley": isometrize_cayley,
    "householder": isometrize_householder,
    "torch_householder": isometrize_torch_householder,
}


def isometrize(x, method="qr"):
    """Generate an isometric (or unitary if square) / orthogonal matrix from
    array ``x``.

    Parameters
    ----------
    x : array
        The matrix to project into isometrix form.
    method : str, optional
        The method used to generate the isometry. The options are:

        - "qr": use the Q factor of the QR decomposition of ``x`` with the
          constraint that the diagonal of ``R`` is positive.
        - "svd": uses ``U @ VH`` of the SVD decomposition of ``x``. This is
          useful for finding the 'closest' isometric matrix to ``x``, such as
          when it has been expanded with noise etc. But is less stable for
          differentiation / optimization.
        - "exp": use the matrix exponential of ``x - dag(x)``, first
          completing ``x`` with zeros if it is rectangular. This is a good
          parametrization for optimization, but more expensive for non-square
          ``x``.
        - "cayley": use the Cayley transform of ``x - dag(x)``, first
          completing ``x`` with zeros if it is rectangular. This is a good
          parametrization for optimization (one the few compatible with
          `HIPS/autograd` e.g.), but more expensive for non-square ``x``.
        - "householder": use the Householder reflection method directly. This
          requires that the backend implements "linalg.householder_product".
        - "torch_householder": use the Householder reflection method directly,
          using the ``torch_householder`` package. This requires that the
          package is installed and that the backend is ``"torch"``. This is
          generally the best parametrizing method for "torch" if available.
        - "mgs": use a python implementation of the modified Gram Schmidt
          method directly. This is slow if not compiled but a useful reference.

        Not all backends support all methods or differentiating through all
        methods.

    Returns
    -------
    Q : array
        The isometrization / orthogonalization of ``x``.
    """
    m, n = x.shape
    fat = m < n
    if fat:
        x = do("transpose", x)
    Q = _ISOMETRIZE_METHODS[method](x)
    if fat:
        Q = do("transpose", Q)
    return Q


@compose
def squared_op_to_reduced_factor(x2, dl, dr, right=True):
    """Given the square, ``x2``, of an operator ``x``, compute either the left
    or right reduced factor matrix of the unsquared operator ``x`` with
    original shape ``(dl, dr)``.
    """
    if right:
        if dl < dr:
            # know exactly low-rank, so truncate
            keep = dl
        else:
            keep = -1
    else:
        if dl > dr:
            # know exactly low-rank, so truncate
            keep = dr
        else:
            keep = -1

    try:
        # attempt faster hermitian eigendecomposition
        U, s2, VH = eigh_truncated(
            x2,
            max_bond=keep,
            cutoff=0.0,
            absorb=None,
            positive=1,  # know positive
        )
        # might have negative eigenvalues due to numerical error from squaring
        s2 = do("clip", s2, 0.0, None)

    except Exception as e:
        warnings.warn(
            "squared_op_to_reduced_factor: eigh_truncated failed"
            f" with error: {e}, falling back to svd_truncated.",
            RuntimeWarning,
        )

        # fallback to SVD if maybe badly conditioned etc.
        U, s2, VH = svd_truncated(
            x2,
            max_bond=keep,
            cutoff=0.0,
            absorb=None,
        )

    s = do("sqrt", s2)
    if right:
        factor = ldmul(s, VH)
    else:  # 'left'
        factor = rdmul(U, s)

    return factor


@squared_op_to_reduced_factor.register("numpy")
@njit
def squared_op_to_reduced_factor_numba(x2, dl, dr, right=True):
    s2, W = np.linalg.eigh(x2)

    if right:
        if dl < dr:
            # know exactly low-rank, so truncate
            keep = dl
        else:
            keep = None
    else:
        if dl > dr:
            # know exactly low-rank, so truncate
            keep = dr
        else:
            keep = None

    if keep is not None:
        # outer dimension smaller -> exactly low-rank
        s2 = s2[-keep:]
        W = W[:, -keep:]

    # might have negative eigenvalues due to numerical error from squaring
    s2 = np.clip(s2, 0.0, None)
    s = np.sqrt(s2)

    if right:
        factor = ldmul_numba(s, dag_numba(W))
    else:  # 'left'
        factor = rdmul_numba(W, s)

    return factor


def compute_oblique_projectors(
    Rl, Rr, max_bond, cutoff, absorb="both", cutoff_mode=4, **compress_opts
):
    """Compute the oblique projectors for two reduced factor matrices that
    describe a gauge on a bond. Concretely, assuming that ``Rl`` and ``Rr`` are
    the reduced factor matrices for local operator ``A``, such that:

    .. math::

        A = Q_L R_L R_R Q_R

    with ``Q_L`` and ``Q_R`` isometric matrices, then the optimal inner
    truncation is given by:

    .. math::

        A' = Q_L P_L P_R' Q_R

    Parameters
    ----------
    Rl : array
        The left reduced factor matrix.
    Rr : array
        The right reduced factor matrix.

    Returns
    -------
    Pl : array
        The left oblique projector.
    Pr : array
        The right oblique projector.
    """
    if max_bond is None:
        max_bond = -1

    absorb = _ABSORB_MAP[absorb]
    cutoff_mode = _CUTOFF_MODE_MAP[cutoff_mode]

    Ut, st, VHt = svd_truncated(
        Rl @ Rr,
        max_bond=max_bond,
        cutoff=cutoff,
        absorb=None,
        cutoff_mode=cutoff_mode,
        **compress_opts,
    )

    if absorb is None:
        Pl = Rr @ rddiv(dag(VHt), st)
        Pr = lddiv(st, dag(Ut)) @ Rl
        return Pl, st, Pr

    elif absorb == 0:
        st_sqrt = do("sqrt", st)

        # then form the 'oblique' projectors
        Pl = Rr @ rddiv(dag(VHt), st_sqrt)
        Pr = lddiv(st_sqrt, dag(Ut)) @ Rl

    elif absorb == -1:
        Pl = Rr @ dag(VHt)
        Pr = lddiv(st, dag(Ut)) @ Rl

    elif absorb == 1:
        Pl = Rr @ rddiv(dag(VHt), st)
        Pr = dag(Ut) @ Rl
    else:
        raise ValueError(f"Unrecognized absorb={absorb}.")

    return Pl, Pr


def compute_bondenv_projectors(
    E,
    max_bond,
    cutoff=0.0,
    absorb="both",
    max_iterations=100,
    tol=1e-10,
    solver="solve",
    solver_maxiter=4,
    prenormalize=False,
    condition=True,
    enforce_pos=True,
    pos_smudge=1e-10,
    init="svd",
    info=None,
):
    """Given 4D environment tensor of a bond, iteratively compute projectors
    that compress the bond dimension to `max_bond`, minimizing the distance in
    terms of frobenius norm. If absorb!="both" and cutoff!=0.0 then a final
    truncated SVD is also performed on the final projector pair.

    N.B. This is experimental and not working for e.g. fermions yet.

    Parameters
    ----------
    E : array
        The 4D environment tensor of a bond. The dimensions should be arranged
        as (ket-left, ket-right, bra-left, bra-right).
    max_bond : int
        The maximum bond dimension to compress to.
    cutoff : float, optional
        The singular value cutoff to use.
    absorb : {'both', 'left', 'right', None}, optional
        How to absorb the effective singular values into the tensors.
    max_iterations : int, optional
        The maximum number of iterations to use when fitting the projectors.
    tol : float, optional
        The target tolerance to reach when fitting the projectors.
    solver : {'solve', None, str}, optional
        The solver to use inside the fitting loop. If None will use a custom
        conjugate gradient method. Else can be any of the iterative solvers
        supported by ``scipy.sparse.linalg`` such as 'gmres', 'bicgstab', etc.
    solver_maxiter : int, optional
        The maximum number of iterations to use for the *inner* solver, i.e.
        per fitting step, only for iterative `solver` args.
    prenormalize : bool, optional
        Whether to prenormalize the environment tensor such that its full
        contraction before compression is 1. Recommended for stability when
        the normalization does not matter.
    condition : bool or "iso", optional
        Whether to condition the projectors after each fitting step. If
        ``True``, their norms will be simply matched. If ``"iso"``, then they
        are gauged each time such that the previous tensor is isometric.
        Recommended for stability.
    enforce_pos : bool, optional
        Whether to enforce the environment tensor to be positive semi-definite
        by symmetrizing and clipping negative eigenvalues. Recommended for
        stability.
    pos_smudge : float, optional
        The value to clip negative eigenvalues to when enforcing positivity,
        relative to the largest eigenvalue.
    init : {'svd', 'eigh', 'random', 'reduced'}, optional
        How to initialize the compression projectors. The options are:

        - 'svd': use a truncated SVD of the environment tensor with the bra
          bond traced out.
        - 'eigh': use a similarity compression of the environment tensor with
          the bra bond traced out.
        - 'random': use random projectors.
        - 'reduced': split the environment into bra and ket parts, then
          canonize one half left and right to get the reduced factors.

    info : dict, optional
        If provided, will store information about the fitting process here.
        The keys 'iterations' and 'distance' will contain the final number of
        iterations and distance reached respectively.

    Returns
    -------
    Pl : array
        The left projector.
    Pr : array
        The right projector.
    """
    backend = infer_backend(E)
    _conj = get_lib_fn(backend, "conj")
    _fuse = get_lib_fn(backend, "fuse")
    _reshape = get_lib_fn(backend, "reshape")

    absorb = _ABSORB_MAP[absorb]

    if solver == "solve":
        _solve = get_lib_fn(backend, "linalg.solve")
        use_x0 = False
    elif solver is None:
        from .fitting import conjugate_gradient

        _solve = conjugate_gradient
        use_x0 = True

    blocksparse = isblocksparse(E)
    fermionic = isfermionic(E)

    if fermionic:
        if E.indices[2].dual:
            E = E.phase_flip(2)
        else:
            E = E.phase_flip(3)

    if prenormalize:
        E = E / ctg.array_contract((E,), (("K", "K", "B", "B"),), ())

    if enforce_pos:
        with backend_like(backend):
            Ea = do("fuse", E, (0, 1), (2, 3))
            Ea = (Ea + dag(Ea)) / 2
            el, ev = do("linalg.eigh", Ea)
            lmax = do("max", el)
            el = do("clip", el + lmax * pos_smudge, lmax * pos_smudge, None)
            Ea = do("multiply_diagonal", ev, el, axis=1) @ dag(ev)
            E = do("reshape", Ea, E.shape)

    # current bond dim
    d = E.shape[0]
    # environment with bra indices traced out (i.e. half uncompressed)
    Ek = ctg.array_contract((E,), (("kl", "kr", "X", "X"),), ("kl", "kr"))
    # for distance calculation, compute <A|A>, which is constant
    yAA = do("abs", ctg.array_contract((Ek,), (("X", "X"),), ()))

    # initial guess for projectors

    if init == "svd":
        Pl, _, Pr = svd_truncated(
            Ek,
            absorb=None,
            max_bond=max_bond,
            cutoff=1e-15,
            cutoff_mode=2,
        )
        Pl = _conj(Pl)
        Pr = _conj(Pr)

    elif init == "eigh":
        Pl, Pr = similarity_compress(Ek, max_bond)

    elif init == "random":
        if backend == "torch":
            import torch

            Pl = torch.randn(d, max_bond, dtype=E.dtype, device=E.device)
            Pr = torch.linalg.pinv(Pl)
        else:
            Pl = do("random.normal", size=(d, max_bond), like=E)
            Pr = do("linalg.pinv", Pl)

    elif init == "reduced":
        from .tensor_core import Tensor

        ft = Tensor(E, ["kl", "kr", "bl", "br"])
        # factor positive environment
        ekt, _ = ft.split(
            left_inds=["kl", "kr"],
            right_inds=["bl", "br"],
            get="tensors",
            bond_ind="b",
            method="eigh",
        )
        # compute left reduced factor
        Rl = ekt.compute_reduced_factor("right", ["b", "kr"], ["kl"])

        # compute right reduced factor
        Rr = ekt.compute_reduced_factor("left", ["kr"], ["kl", "b"])

        # compute compressed projectors
        Pl, Pr = compute_oblique_projectors(
            Rl,
            Rr,
            max_bond=max_bond,
            cutoff=cutoff,
        )

    else:
        raise ValueError(f"Unrecognized init={init}.")

    # E, Pl = do("align_axes", E, Pl, ((0,), (0,)))
    # E, Pr = do("align_axes", E, Pr, ((1,), (1,)))
    # E, Pl = do("align_axes", E, Pl, ((2,), (0,)))
    # E, Pr = do("align_axes", E, Pr, ((3,), (1,)))
    # Ek, Pl = do("align_axes", Ek, Pl, ((0,), (0,)))
    # Ek, Pr = do("align_axes", Ek, Pr, ((1,), (1,)))

    def _distance(xc, x, A, b):
        yAB = (xc @ b).real
        yBB = abs(xc @ (A @ x))
        return 2 * (yAA + yBB - 2 * yAB) ** 0.5 / (yAA**0.5 + yBB**0.5)

    old_diff = None
    new_diff = None

    if use_x0:
        xl0 = _conj(_fuse(Pl, (0, 1)))
        xr0 = _conj(_fuse(Pr, (0, 1)))
    else:
        xl0 = xr0 = None

    for it in range(max_iterations):
        if condition == "iso":
            Lr, _, Pr = lq_stabilized(Pr)
            Pl = Pl @ Lr

        elif condition:
            # match projector norms
            nrml = do("linalg.norm", Pl)
            nrmr = do("linalg.norm", Pr)
            Pl = Pl * (nrmr**0.5 / nrml**0.5)
            Pr = Pr * (nrml**0.5 / nrmr**0.5)

        # solve for left projector
        #      ┌────┐             ┌────┐
        #          ┌┴─┐               ┌┴─┐
        #          │Pr│               │Pr│
        #          └┬─┘               └┬─┘
        #     ┌┴────┴┐           ┌┴────┴┐
        #     │  E   │   x    =  │  Ek  │
        #     └┬────┬┘           └┬────┬┘
        #          ┌┴─┐           │    │
        #      ?   │Pr│*          └────┘
        #          └┬─┘
        #      └────┘
        A = ctg.array_contract(
            [E, Pr, _conj(Pr)],
            [("kl", "krX", "bl", "brX"), ("kr", "krX"), ("br", "brX")],
            ("kl", "kr", "bl", "br"),
        )
        b = ctg.array_contract(
            [Ek, Pr],
            [("kl", "krX"), ("kr", "krX")],
            ("kl", "kr"),
        )

        if blocksparse:
            A, b = do("align_axes", A, b, axes=((0, 1), (0, 1)))
            A, b = do("align_axes", A, b, axes=((2, 3), (0, 1)))

        # get pre-fuse shape as `d` might have changed
        Pl_shape = b.shape
        b = _fuse(b, (0, 1))
        A = _fuse(A, (0, 1), (2, 3))

        if use_x0:
            x = _solve(A, b, x0=xl0, maxiter=solver_maxiter)
            xl0 = x
        else:
            x = _solve(A, b)

        xc = _conj(x)
        Pl = _reshape(xc, Pl_shape)

        # solve for right projector
        #      ┌────┐            ┌────┐
        #     ┌┴─┐              ┌┴─┐
        #     │Pl│              │Pl│
        #     └┬─┘              └┬─┘
        #     ┌┴─────┐          ┌┴─────┐
        #     │  E   │  x ?  =  │  Ek  │
        #     └┬─────┘          └┬────┬┘
        #     ┌┴─┐               │    │
        #     │Pl│*              └────┘
        #     └┬─┘
        #      └────┘
        if condition == "iso":
            Pl, _, Rl = qr_stabilized(Pl)
            Pr = Rl @ Pr

        b = ctg.array_contract(
            [Ek, Pl],
            [("klX", "kr"), ("klX", "kl")],
            ("kl", "kr"),
        )
        A = ctg.array_contract(
            [E, Pl, _conj(Pl)],
            [("klX", "kr", "blX", "br"), ("klX", "kl"), ("blX", "bl")],
            ("kl", "kr", "bl", "br"),
        )

        if blocksparse:
            A, b = do("align_axes", A, b, axes=((0, 1), (0, 1)))
            A, b = do("align_axes", A, b, axes=((2, 3), (0, 1)))

        # get pre-fuse shape as `d` might have changed
        Pr_shape = b.shape
        b = _fuse(b, (0, 1))
        A = _fuse(A, (0, 1), (2, 3))

        if use_x0:
            x = _solve(A, b, x0=xr0, maxiter=solver_maxiter)
            xr0 = x
        else:
            x = _solve(A, b)

        xc = _conj(x)
        Pr = _reshape(xc, Pr_shape)

        # check for convergence
        if tol != 0.0:
            new_diff = _distance(xc, x, A, b)
            if old_diff is not None and abs(new_diff - old_diff) < tol:
                break
            old_diff = new_diff

    if info is not None:
        info["distance"] = new_diff
        info["iterations"] = it + 1

    if fermionic:
        # reflip
        if Pr.indices[0].dual:
            Pr.phase_flip(0, inplace=True)
        else:
            Pl.phase_flip(1, inplace=True)

    if not ((absorb == 0) and (cutoff == 0.0)):
        # should/can do this on reduced factors?
        Pl, svals, Pr = svd_truncated(
            Pl @ Pr,
            max_bond=max_bond,
            cutoff=cutoff,
            absorb=absorb,
        )
    else:
        # svals already absorbed on both sides, and no dynamic cutoff
        svals = None

    return Pl, svals, Pr
