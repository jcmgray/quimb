"""Functions for decomposing and projecting matrices, purely at the level of
arrays (rather than wrapped as Tensors). The includes the main entry point
``array_split``, which dispatches to a registered split driver based on the
method.
"""

import functools
import inspect
import operator
import warnings

import cotengra as ctg
import numpy as np
import scipy.linalg as scla
import scipy.linalg.interpolative as sli
import scipy.sparse.linalg as spla
from autoray import (
    astype,
    compose,
    dag,
    get_dtype_name,
    get_lib_fn,
    get_namespace,
    infer_backend,
    lazy,
)

from ..core import njit
from ..linalg import base_linalg, rand_linalg
from ..utils import ensure_dict, parse_info_extras
from .array_ops import isblocksparse, isfermionic


def array_split(
    x,
    method="auto",
    absorb="auto",
    max_bond=None,
    cutoff=1e-10,
    cutoff_mode="rsum2",
    renorm=None,
    info=None,
    **kwargs,
):
    """Split a 2D array (or batch of 2D arrays) into left and right factors
    (and possibly singular values) using a matrix decomposition. This is the
    main array-level entry point, dispatching to a registered split driver
    based on ``method``, with ``absorb`` controlling the output form of the
    decomposition, and ``max_bond``, ``cutoff`` and others controlling
    static and dynamic truncation respectively.

    Parameters
    ----------
    x : array_like
        The array or batch of arrays to split, of shape ``(..., m, n)``.
    method : str, optional
        The decomposition method to use:

        - ``'auto'``: use the default method based on ``absorb`` and whether
          truncation is requested via ``max_bond`` and/or ``cutoff``.

        The five main methods are:

        - ``'svd'``: full SVD, allowing all truncation options.
        - ``'svd:eig'``: full SVD via eigendecomposition, allowing all
          truncation options. This can be faster than the standard SVD, but
          entails some loss of precision.
        - ``'svd:rand'``: low-rank SVD via randomized projection, allows
          (and is only beneficial for) static truncation
        - ``'qr'``: QR decomposition, by default left factor is isometric.
        - ``'qr:cholesky'``: QR decomposition via Cholesky factorization, by
          default left factor is isometric. This can be faster than the
          standard QR, but entails some loss of precision.

        All of these can return different forms (and take various shortucts)
        based on ``absorb``. Other methods are:

        - ``'lq'``: LQ decomposition, right factor is isometric. This is simply
          an alias for ``method='qr'`` with ``absorb='left'``.
        - ``'eigh'``: full eigen-decomposition, array must be hermitian.
        - ``'eigsh'``: iterative eigen-decomposition, array must be hermitian.
        - ``'svds'``: iterative SVD, allows truncation.
        - ``'isvd'``: iterative SVD using interpolative methods, allows
          truncation (see :mod:`scipy.linalg.interpolative`).
        - ``'rsvd'``: randomized iterative SVD with truncation (alternative
          implementation).
        - ``'lu'``: full LU decomposition, allows truncation. Favors sparsity
          but is not rank optimal.
        - ``'polar_right'``: polar decomposition as ``A = U @ P``.
        - ``'polar_left'``: polar decomposition as ``A = P @ U``.
        - ``'cholesky'``: cholesky decomposition, array must be positive
          definite.

    absorb : str or None, optional
        Desired form the decomposition / where to absorb the singular- or
        eigen- values. Common options are:

        - ``'auto'``: use the method's default absorb mode.
        - ``'both'`` / ``'Usq,sqVH'``: absorb ``sqrt(s)`` into both factors.
        - ``'left'`` / ``'Us,VH'``: absorb ``s`` into the left factor,
          leaving the right factor isometric (LQ-like).
        - ``'right'`` / ``'U,sVH'``: absorb ``s`` into the right factor,
          leaving the left factor isometric (QR-like).
        - ``None`` / ``'U,s,VH'``: return ``s`` unabsorbed.

        Additional options for returning partial results (e.g. only one
        factor, unrequested factors returned as ``None``):

        - ``'lorthog'`` / ``'U'``: return only the left isometric factor.
        - ``'rorthog'`` / ``'VH'``: return only the right isometric factor.
        - ``'lfactor'`` / ``'Us'``: return only the left factor with ``s``
          absorbed (the L in an LQ decomposition).
        - ``'rfactor'`` / ``'sVH'``: return only the right factor with ``s``
          absorbed (the R in a QR decomposition).
        - ``'s'``: return only the singular values.

        Note in all cases unrequested factors are returned as ``None``.
    max_bond : int or None, optional
        The maximum bond dimension (number of singular values) to keep.
        ``None`` means no limit.
    cutoff : float, optional
        Threshold for discarding singular values, only used by methods that
        support dynamic truncation.
    cutoff_mode : {'rsum2', 'rel', 'abs', 'sum2', 'rsum1', 'sum1'}, optional
        How to interpret ``cutoff`` when discarding singular values:

        - ``'rel'``: values less than ``cutoff * s[0]`` discarded.
        - ``'abs'``: values less than ``cutoff`` discarded.
        - ``'sum2'``: sum squared of values discarded must be ``< cutoff``.
        - ``'rsum2'``: sum squared of values discarded must be less than
          ``cutoff`` times the total sum of squared values.
        - ``'sum1'``: sum values discarded must be ``< cutoff``.
        - ``'rsum1'``: sum of values discarded must be less than ``cutoff``
          times the total sum of values.

    renorm : int or bool, optional
        Whether to renormalize the kept singular values to maintain the
        Frobenius or nuclear norm. ``0`` or ``False`` means no renormalization.
        ``True`` automatically picks the power based on ``cutoff_mode``.
    info : dict or None, optional
        If a dict is passed, store truncation info in the dict. Currently only
        supports the key 'error' for the truncation error, which is only
        computed if ``method in {"svd", "svd:eig"}``.
    **kwargs
        Additional keyword arguments passed to the underlying split driver.

    Returns
    -------
    left : array or None
        The left factor, or ``None`` if not requested by ``absorb``.
    s : array or None
        The singular/eigen values, or ``None`` if absorbed into the factors.
    right : array or None
        The right factor, or ``None`` if not requested by ``absorb``.
    """

    method, split_opts = parse_split_opts(
        method,
        absorb,
        max_bond,
        cutoff,
        cutoff_mode,
        renorm,
    )

    if info is not None:
        # inject info dict for extra info returns (e.g. truncation error)
        kwargs["info"] = info

    if method in _DENSE_ONLY_METHODS and isinstance(x, spla.LinearOperator):
        x = x.to_dense()

    return _SPLIT_FNS[method](x, **split_opts, **kwargs)


def array_svals(x, method="svd", **kwargs):
    """Compute the singular values of a 2D array without forming the full
    decomposition, using a registered singular value driver.

    Parameters
    ----------
    x : array_like
        The 2D array whose singular values to compute.
    method : str, optional
        The method to use. Must have a registered singular value driver (e.g.
        ``'svd'`` or ``'svd:eig'``).
    **kwargs
        Additional keyword arguments passed to the underlying driver.

    Returns
    -------
    s : array
        The singular values in descending order.
    """
    method = parse_method(method)
    return _SPLIT_VALUES_FNS[method](x, **kwargs)


# mode aliases, and conversion to enum for numba functions
get_U_s_VH = None  # 'full'
get_s = 2  # 'svals'
get_Usq = -12  # 'lsqrt'
get_VH = -11  # 'rorthog'
get_Us = -10  # 'lfactor'
get_Us_VH = -1  # absorb 'left'
get_Usq_sqVH = 0  # absorb 'both'
get_U_sVH = 1  # absorb 'right'
get_U = 10  # 'lorthog'
get_sVH = 11  # 'rfactor'
get_sqVH = 12  # 'rsqrt'
_ABSORB_MAP = {}
for mode, aliases in [
    (None, ["U,s,VH"]),
    (get_s, ["s"]),  # 2
    (get_Usq, ["lsqrt"]),  # -12
    (get_VH, ["VH", "rorthog"]),  # -11
    (get_Us, ["Us", "lfactor"]),  # -10
    (get_Us_VH, ["Us,VH", "left"]),  # -1
    (get_Usq_sqVH, ["Usq,sqVH", "both"]),  # 0
    (get_U_sVH, ["U,sVH", "right"]),  # 1
    (get_U, ["U", "lorthog"]),  # 10
    (get_sVH, ["sVH", "rfactor"]),  # 11
    (get_sqVH, ["sqVH", "rsqrt"]),  # 12
]:
    _ABSORB_MAP[mode] = mode
    for alias in aliases:
        _ABSORB_MAP[alias] = mode

_ABSORB_TRANSPOSE_MAP = {
    None: None,
    get_s: get_s,
    get_Usq: get_sqVH,
    get_VH: get_U,
    get_Us: get_sVH,
    get_Us_VH: get_U_sVH,
    get_Usq_sqVH: get_Usq_sqVH,
    get_U_sVH: get_Us_VH,
    get_U: get_VH,
    get_sVH: get_Us,
    get_sqVH: get_Usq,
}

_RETURNS_LEFT_ABSORBS = {
    get_U_s_VH,
    get_Usq,
    get_Us,
    get_Us_VH,
    get_Usq_sqVH,
    get_U_sVH,
    get_U,
}
_RETURNS_RIGHT_ABSORBS = {
    get_U_s_VH,
    get_VH,
    get_Us_VH,
    get_Usq_sqVH,
    get_U_sVH,
    get_sVH,
    get_sqVH,
}


# cutoff_mode aliases, and conversion to enum for numba functions
cutoff_mode_abs = 1
cutoff_mode_rel = 2
cutoff_mode_sum2 = 3
cutoff_mode_rsum2 = 4
cutoff_mode_sum1 = 5
cutoff_mode_rsum1 = 6
_CUTOFF_MODE_MAP = {}
for mode, aliases in [
    (cutoff_mode_abs, ["abs"]),
    (cutoff_mode_rel, ["rel"]),
    (cutoff_mode_sum2, ["sum2"]),
    (cutoff_mode_rsum2, ["rsum2"]),
    (cutoff_mode_sum1, ["sum1"]),
    (cutoff_mode_rsum1, ["rsum1"]),
]:
    _CUTOFF_MODE_MAP[mode] = mode
    for alias in aliases:
        _CUTOFF_MODE_MAP[alias] = mode

_MAX_BOND_LOOKUP = {None: -1}
_CUTOFF_LOOKUP = {None: -1.0}
_RENORM_LOOKUP = {
    cutoff_mode_sum2: 2,
    cutoff_mode_rsum2: 2,
    cutoff_mode_sum1: 1,
    cutoff_mode_rsum1: 1,
}


def parse_method(method):
    if method == "eig":
        warnings.warn(
            "`method='eig'` has been renamed to `method='svd:eig'` for "
            "consistency. In future it might apply the non-hermitian "
            "eigendecomposition instead of the SVD via eig, use 'svd:eig' "
            "to keep the current behaviour.",
            FutureWarning,
        )
        method = "svd:eig"
    return method


@functools.cache
def parse_method_absorb(
    method="auto",
    absorb="auto",
    truncation=True,
):
    """Parse the method and absorb options. Possibly resolve "auto" settings
    to defaults, and  convert from string aliases to numeric codes.
    """
    # handle deprecation aliases
    method = parse_method(method)

    # first we resolve method
    if method == "auto":
        if truncation:
            # if truncating always default to SVD
            method = "svd"
        else:
            if absorb == "auto":
                # ... don't really have enough information to choose
                warnings.warn(
                    "No method, absorb, or truncation options specified. "
                    "Defaulting to `method='svd'` and `absorb='both'`.",
                )
                method = "svd"
            else:
                # determine method based on absorb
                absorb = _ABSORB_MAP[absorb]

                dont_need_singular_values = absorb in (
                    get_U_sVH,
                    get_U,
                    get_sVH,
                    get_Us_VH,
                    get_Us,
                    get_VH,
                )

                if dont_need_singular_values:
                    # note qr handles lq-like options via absorb
                    method = "qr"
                else:
                    method = "svd"

    # lq methods are simply qr with different default absorb
    if method.startswith("lq"):
        method = "qr" + method[2:]
        if absorb == "auto":
            absorb = "left"

    # then we resolve absorb
    if absorb == "auto":
        # determined by method
        absorb = _DEFAULT_ABSORB[method]
    else:
        # normalize to numeric code
        absorb = _ABSORB_MAP[absorb]

    return method, absorb


@functools.cache
def parse_split_opts(
    method="auto",
    absorb="auto",
    max_bond=None,
    cutoff=1e-10,
    cutoff_mode="rsum2",
    renorm=None,
):
    """Automatically select method and absorb, convert defaults and settings to
    numeric type for numba, and only supply valid options for given method.
    """
    # first normalize main truncation opts
    max_bond = _MAX_BOND_LOOKUP.get(max_bond, max_bond)
    cutoff = _CUTOFF_LOOKUP.get(cutoff, cutoff)

    # note: default options are truncating
    truncation = (max_bond > 0) or (cutoff > 0.0)

    # first we resolve the the possibly automatically chosen method and absorb
    method, absorb = parse_method_absorb(method, absorb, truncation=truncation)

    # finally we check which options to inject based on method capabilities
    signature = inspect.signature(_SPLIT_FNS[method])
    opts = dict()

    if "absorb" in signature.parameters:
        opts["absorb"] = absorb
    else:
        if absorb is None:
            # no singular value options should be supplied
            raise ValueError(
                "You can't return the singular values separately when "
                "`method='{}'`.".format(method)
            )

    if "max_bond" in signature.parameters:
        opts["max_bond"] = max_bond

    if "cutoff" in signature.parameters:
        opts["cutoff"] = cutoff

        if "cutoff_mode" in signature.parameters:
            cutoff_mode = _CUTOFF_MODE_MAP[cutoff_mode]
            opts["cutoff_mode"] = cutoff_mode

            # renorm doubles up as the power used to renormalize
            if "renorm" in signature.parameters:
                if renorm is True:
                    # match renorm power to cutoff mode
                    renorm = _RENORM_LOOKUP.get(cutoff_mode, 0)
                    opts["renorm"] = renorm
                else:
                    # turn off, or use explicitly supplied power
                    opts["renorm"] = 0 if renorm is None else renorm

    return method, opts


@functools.cache
def parse_split_left_right_isom(method="auto", absorb="auto"):
    """Based on split method and absorb mode, determine whether the left and
    right factors are isometric, and can be marked as such.
    """
    # NOTE: truncation can't (?) affect parsed absorb output
    method, absorb = parse_method_absorb(method, absorb)
    left_isom = absorb in (get_U_s_VH, get_U_sVH, get_U)
    right_isom = absorb in (get_U_s_VH, get_Us_VH, get_VH)
    return left_isom, right_isom


_SPLIT_FNS = {}
_SPLIT_VALUES_FNS = {}
_DENSE_ONLY_METHODS = set()
_DEFAULT_ABSORB = {}


def register_split_driver(name, sparse=False, default_absorb=get_Usq_sqVH):
    """Decorator to register functions which can decompose a matrix, and sets
    various flags specifying its capabilities and thus which options are valid
    to supply in ``array_split``.

    Parameters
    ----------
    name : str
        The name of the method, corresponding to the ``method`` argument of
        ``array_split``.
    truncation : {"none", "static", "dynamic"}, optional
        Whether the method can handle truncation, and if so, whether it can
        only handle static truncation (i.e. `max_bond`) or dynamic truncation
        as well (i.e. `cutoff` and `cutoff_mode`).
    sparse : bool, optional
        Whether the method can handle sparse arrays directly.
    """
    if not sparse:
        _DENSE_ONLY_METHODS.add(name)

    _DEFAULT_ABSORB[name] = default_absorb

    def decorator(fn):
        _SPLIT_FNS[name] = fn
        return fn

    return decorator


def register_svals_driver(name):
    """Decorator to register functions which can compute the singular values of
    a matrix without computing the full decomposition, which is used in
    ``array_svals``.

    Parameters
    ----------
    name : str
        The name of the method, corresponding to the ``method`` argument of
        ``array_svals``.
    """

    def decorator(fn):
        _SPLIT_VALUES_FNS[name] = fn
        return fn

    return decorator


# -------------------------- actual split drivers --------------------------- #


@njit  # pragma: no cover
def dag_numba(x):
    return np.conjugate(x.T)


def safe_inverse(s, cutoff):
    return s / (s**2 + cutoff**2)
    # NOTE: other options, possibly less gradient friendly
    # return 1 / s
    # return 1 / (s + (s == 0.0))
    # return np.where(s > cutoff, 1 / np.maximum(s, cutoff), 0.0)


@njit  # pragma: no cover
def safe_inverse_numba(s, cutoff):
    return s / (s**2 + cutoff**2)


# some convenience functions for multiplying diagonals


@compose
def rdmul(x, d):
    """Right-multiplication a matrix by a vector representing a diagonal."""
    return x * d[..., None, :]


@njit  # pragma: no cover
def rdmul_numba(x, d):
    return x * d[..., None, :]


@compose
def rddiv(x, d):
    """Right-multiplication of a matrix by a vector representing an inverse
    diagonal.
    """
    return x / d[..., None, :]


@njit  # pragma: no cover
def rddiv_numba(x, d):
    return x / d[None, :]


@compose
def ldmul(d, x):
    """Left-multiplication a matrix by a vector representing a diagonal."""
    return x * d[..., :, None]


@njit  # pragma: no cover
def ldmul_numba(d, x):
    return x * d[..., :, None]


@compose
def lddiv(d, x):
    """Left-multiplication of a matrix by a vector representing an inverse
    diagonal.
    """
    return x / d[..., :, None]


@njit  # pragma: no cover
def lddiv_numba(d, x):
    return x / d[..., :, None]


@compose
def sgn(x):
    """Get the 'sign' of ``x``, such that ``x / sgn(x)`` is real and
    non-negative.
    """
    xp = get_namespace(x)
    x0 = xp.equal(x, 0.0)
    return (x + x0) / (xp.abs(x) + x0)


@sgn.register("numpy")
@njit  # pragma: no cover
def sgn_numba(x):
    x0 = x == 0.0
    return (x + x0) / (np.abs(x) + x0)


@sgn.register("tensorflow")
def sgn_tf(x):
    xp = get_namespace(x)
    x0 = xp.cast(xp.equal(x, 0.0), x.dtype)
    xa = xp.cast(xp.abs(x), x.dtype)
    return (x + x0) / (xa + x0)


# ----------------------------------- svd ----------------------------------- #


def _do_absorb(U, s, VH, absorb=None, xp=None):
    if absorb is None:  # 'full'
        return U, s, VH
    if xp is None:
        xp = get_namespace(U)
    if absorb == get_Usq_sqVH:  # 'both'
        sq = xp.sqrt(s)
        return rdmul(U, sq), None, ldmul(sq, VH)
    if absorb == get_U_sVH:  # 'right'
        return U, None, ldmul(s, VH)
    if absorb == get_Us_VH:  # 'left'
        return rdmul(U, s), None, VH
    if absorb == get_sVH:  # 'rfactor'
        return None, None, ldmul(s, VH)
    if absorb == get_Us:  # 'lfactor'
        return rdmul(U, s), None, None
    if absorb == get_U:  # 'lorthog'
        return U, None, None
    if absorb == get_VH:  # 'rorthog'
        return None, None, VH
    if absorb == get_Usq:  # 'lsqrt'
        sq = xp.sqrt(s)
        return rdmul(U, sq), None, None
    if absorb == get_sqVH:  # 'rsqrt'
        sq = xp.sqrt(s)
        return None, None, ldmul(sq, VH)
    if absorb == get_s:  # 'svals'
        return None, s, None
    raise ValueError(f"Invalid absorb mode: {absorb}")


@njit  # pragma: no cover
def _do_absorb_numba(U, s, VH, absorb):
    if absorb is None:
        # get_U_s_VH - return as-is
        return U, s, VH
    if absorb == get_Usq_sqVH:  # 'both'
        sq = np.sqrt(s)
        return rdmul_numba(U, sq), None, ldmul_numba(sq, VH)
    if absorb == get_U_sVH:  # 'right'
        return U, None, ldmul_numba(s, VH)
    if absorb == get_Us_VH:  # 'left'
        return rdmul_numba(U, s), None, VH
    if absorb == get_sVH:  # 'rfactor'
        return None, None, ldmul_numba(s, VH)
    if absorb == get_Us:  # 'lfactor'
        return rdmul_numba(U, s), None, None
    if absorb == get_U:  # 'lorthog'
        return U, None, None
    if absorb == get_VH:  # 'rorthog'
        return None, None, VH
    if absorb == get_Usq:  # 'lsqrt'
        sq = np.sqrt(s)
        return rdmul_numba(U, sq), None, None
    if absorb == get_sqVH:  # 'rsqrt'
        sq = np.sqrt(s)
        return None, None, ldmul_numba(sq, VH)
    if absorb == get_s:  # 'svals'
        return None, s, None
    return None, None, None


def _trim_and_renorm_svd_result(
    U,
    s,
    VH,
    cutoff=-1.0,
    cutoff_mode=cutoff_mode_rsum2,
    max_bond=-1,
    absorb=get_Usq_sqVH,
    renorm=0,
    use_abs=False,
    info=None,
    xp=None,
):
    """Give full SVD decomposion result ``U``, ``s``, ``VH``, optionally trim,
    renormalize, and absorb the singular values. See ``svd_truncated`` for
    details.
    """
    if xp is None:
        xp = get_namespace(U)

    info = parse_info_extras(info, ("error",))

    if use_abs:
        sabs = xp.abs(s)
    else:
        # assume already all positive
        sabs = s

    *batch_dims, d = xp.shape(sabs)

    if (cutoff > 0.0) or (renorm > 0):
        # need to dynamically truncate based on spectrum
        if cutoff_mode == cutoff_mode_abs:
            n_chi = xp.count_nonzero(sabs > cutoff, axis=-1)

        elif cutoff_mode == cutoff_mode_rel:
            n_chi = xp.count_nonzero(sabs > cutoff * sabs[..., 0:1], axis=-1)

        elif cutoff_mode in (
            cutoff_mode_sum2,
            cutoff_mode_rsum2,
            cutoff_mode_sum1,
            cutoff_mode_rsum1,
        ):
            # TODO: if we are truncating a batch result, we might want to treat
            # all singular values together, assuming block diagonal form
            if cutoff_mode in (cutoff_mode_sum2, cutoff_mode_rsum2):
                pow = 2
                sp = sabs**pow
            else:
                pow = 1
                sp = sabs

            csp = xp.cumsum(sp, axis=-1)
            tot = csp[..., -1:]

            if cutoff_mode in (cutoff_mode_rsum1, cutoff_mode_rsum2):
                above_thresh = csp < tot * (1 - cutoff)
            else:
                above_thresh = csp < tot - cutoff

            n_chi = xp.count_nonzero(above_thresh, axis=-1) + 1

        if batch_dims:
            # batch -> take maximum bond needed for any single batch
            n_chi = xp.max(n_chi)

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
        # some truncation
        s = s[..., :n_chi]
        U = U[..., :, :n_chi]
        VH = VH[..., :n_chi, :]

        if renorm > 0:
            norm = (tot / csp[n_chi - 1]) ** (1 / pow)
            s *= norm

        if "error" in info:
            info["error"] = xp.sqrt(xp.sum(sabs[..., n_chi:] ** 2, axis=-1))

    elif "error" in info:
        # no truncation
        info["error"] = 0.0

    # XXX: tensorflow can't multiply mixed dtypes
    if infer_backend(s) == "tensorflow":
        dtype = get_dtype_name(U)
        if "complex" in dtype:
            s = astype(s, dtype)

    return _do_absorb(U, s, VH, absorb=absorb, xp=xp)


@register_split_driver("svd")
@compose
def svd_truncated(
    x,
    cutoff=-1.0,
    cutoff_mode=cutoff_mode_rsum2,
    max_bond=-1,
    absorb=get_Usq_sqVH,
    renorm=0,
    info=None,
):
    """Singular value decomposition of raw 2d array ``x``, with optional
    truncation based on `max_bond` and/or dynamically on `cutoff`.

    Parameters
    ----------
    cutoff : float, optional
        Singular value cutoff threshold, if ``cutoff <= 0.0``, then only
        ``max_bond`` is used.
    cutoff_mode : {1, 2, 3, 4, 5, 6}, optional
        How to perform the truncation based on ``cutoff``:

        - 1 / 'abs': trim values below ``cutoff``
        - 2 / 'rel': trim values below ``s[0] * cutoff``
        - 3 / 'sum2': trim s.t. ``sum(s_trim**2) < cutoff``.
        - 4 / 'rsum2': trim s.t. ``sum(s_trim**2) < sum(s**2) * cutoff``.
        - 5 / 'sum1': trim s.t. ``sum(s_trim**1) < cutoff``.
        - 6 / 'rsum1': trim s.t. ``sum(s_trim**1) < sum(s**1) * cutoff``.

    max_bond : int, optional
        An explicit maximum bond dimension, use -1 for none.
    absorb : int or None, optional
        How to absorb the singular values, as a pre-converted numeric code
        (``get_Us_VH=-1``: left, ``get_Usq_sqVH=0``: both,
        ``get_U_sVH=1``: right, ``None``: return separately). Use
        ``array_split`` with string aliases (e.g. ``'left'``, ``'both'``,
        ``'right'``, ``None``) for a friendlier interface.
    renorm : int, optional
        Whether to renormalize the kept singular values. ``0`` means
        no renormalization, ``1`` maintains the trace norm, ``2``
        maintains the Frobenius norm.
    info : dict or None, optional
        If a dict is passed, store truncation info in the dict. Currently only
        supports the key 'error' for the truncation error.

    Returns
    -------
    left : array_like or None
    s : array_like or None
    right : array_like or None
    """
    absorb = _ABSORB_MAP[absorb]
    cutoff_mode = _CUTOFF_MODE_MAP[cutoff_mode]

    xp = get_namespace(x)
    U, s, VH = xp.linalg.svd(x)
    return _trim_and_renorm_svd_result(
        U,
        s,
        VH,
        cutoff=cutoff,
        cutoff_mode=cutoff_mode,
        max_bond=max_bond,
        absorb=absorb,
        renorm=renorm,
        info=info,
        xp=xp,
    )


@njit(["i4(f4[:], f4, i4)", "i4(f8[:], f8, i4)"])  # pragma: no cover
def _compute_number_svals_to_keep_numba(s, cutoff, cutoff_mode):
    """Find the number of singular values to keep of ``s`` given ``cutoff`` and
    ``cutoff_mode``.
    """
    if cutoff_mode == cutoff_mode_abs:
        n_chi = np.sum(s > cutoff)

    elif cutoff_mode == cutoff_mode_rel:
        n_chi = np.sum(s > cutoff * s[0])

    elif cutoff_mode in (
        cutoff_mode_sum2,
        cutoff_mode_rsum2,
        cutoff_mode_sum1,
        cutoff_mode_rsum1,
    ):
        if cutoff_mode in (cutoff_mode_sum2, cutoff_mode_rsum2):
            pow = 2
        else:
            pow = 1

        target = cutoff
        if cutoff_mode in (cutoff_mode_rsum2, cutoff_mode_rsum1):
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
    calc_error=False,
):
    """Accelerated version of ``_trim_and_renorm_svd_result``."""

    if use_abs:
        sabs = np.abs(s)
    else:
        sabs = s

    # default values
    if calc_error:
        error = 0.0
    else:
        error = None

    if (cutoff > 0.0) or (renorm > 0):
        # need to dynamically truncate
        n_chi = _compute_number_svals_to_keep_numba(sabs, cutoff, cutoff_mode)

        if max_bond > 0:
            # bond dimension limited by both cutoff and max_bond
            n_chi = min(n_chi, max_bond)

        if n_chi < s.size:
            # some truncation needed
            if calc_error:
                error = np.sqrt(np.sum(sabs[n_chi:] ** 2))

            if renorm > 0:
                f = _compute_svals_renorm_factor_numba(sabs, n_chi, renorm)
                s = s[:n_chi] * f
            else:
                s = s[:n_chi]

            U = U[:, :n_chi]
            VH = VH[:n_chi, :]

    elif (max_bond != -1) and (max_bond < s.shape[0]):
        # some truncation, but only maximum bond specified
        if calc_error:
            error = np.sqrt(np.sum(sabs[max_bond:] ** 2))

        U = U[:, :max_bond]
        s = s[:max_bond]
        VH = VH[:max_bond, :]

    s = np.ascontiguousarray(s)

    U, s, VH = _do_absorb_numba(U, s, VH, absorb)

    return U, s, VH, error


@njit  # pragma: no cover
def svd_truncated_numba(
    x,
    cutoff=-1.0,
    cutoff_mode=cutoff_mode_rsum2,
    max_bond=-1,
    absorb=get_Usq_sqVH,
    renorm=0,
    calc_error=False,
):
    """Accelerated version of ``svd_truncated`` for numpy arrays."""
    U, s, VH = np.linalg.svd(x, full_matrices=False)

    return _trim_and_renorm_svd_result_numba(
        U,
        s,
        VH,
        cutoff,
        cutoff_mode,
        max_bond,
        absorb,
        renorm,
        calc_error=calc_error,
    )


@svd_truncated.register("numpy")
def svd_truncated_numpy(
    x,
    cutoff=-1.0,
    cutoff_mode=cutoff_mode_rsum2,
    max_bond=-1,
    absorb=get_Usq_sqVH,
    renorm=0,
    info=None,
):
    """Numpy version of ``svd_truncated``, trying the accelerated version
    first, then falling back to the more stable scipy version.
    """
    if x.ndim > 2:
        # XXX: batch truncation not implemented in numba version yet
        return svd_truncated._default_fn(
            x,
            cutoff=cutoff,
            cutoff_mode=cutoff_mode,
            max_bond=max_bond,
            absorb=absorb,
            renorm=renorm,
            info=info,
        )

    absorb = _ABSORB_MAP[absorb]
    cutoff_mode = _CUTOFF_MODE_MAP[cutoff_mode]

    info = parse_info_extras(info, ("error",))
    calc_error = "error" in info

    try:
        U, s, VH, error = svd_truncated_numba(
            x,
            cutoff,
            cutoff_mode,
            max_bond,
            absorb,
            renorm,
            calc_error=calc_error,
        )
    except ValueError as e:  # pragma: no cover
        warnings.warn(f"Got: {e}, falling back to scipy gesvd driver.")
        U, s, VH = scla.svd(x, full_matrices=False, lapack_driver="gesvd")

        U, s, VH, error = _trim_and_renorm_svd_result_numba(
            U,
            s,
            VH,
            cutoff,
            cutoff_mode,
            max_bond,
            absorb,
            renorm,
            calc_error=calc_error,
        )

    if calc_error:
        info["error"] = error

    return U, s, VH


@svd_truncated.register("autoray.lazy")
@lazy.core.lazy_cache("svd_truncated")
def svd_truncated_lazy(
    x,
    cutoff=-1.0,
    cutoff_mode=cutoff_mode_rsum2,
    max_bond=-1,
    absorb=get_Usq_sqVH,
    renorm=0,
):
    if cutoff != 0.0:
        raise ValueError("Can't handle dynamic cutoffs in lazy mode.")

    *b, m, n = x.shape
    k = min(m, n)
    if max_bond > 0:
        k = min(k, max_bond)

    lsvdt = x.to(
        fn=get_lib_fn(x.backend, "svd_truncated"),
        args=(x, cutoff, cutoff_mode, max_bond, absorb, renorm),
        shape=(3,),
    )

    U = lsvdt.to(operator.getitem, (lsvdt, 0), shape=(*b, m, k))
    if absorb is None:
        s = lsvdt.to(
            operator.getitem,
            (lsvdt, 1),
            shape=(*b, k),
        )
    else:
        s = None
    VH = lsvdt.to(operator.getitem, (lsvdt, 2), shape=(*b, k, n))

    return U, s, VH


@register_svals_driver("svd")
def svdvals(x):
    """SVD-decomposition, but return singular values only."""
    return np.linalg.svd(x, full_matrices=False, compute_uv=False)


# ------------------------------- svd via eig ------------------------------- #


def svd_via_eig(
    x,
    absorb=None,
    max_bond=-1,
    descending=True,
    right=None,
):
    """Singular value decomposition of raw 2d array ``x``, via hermitian
    eigen-decomposition of the Gram matrix (xdag @ x or x @ xdag), with
    static truncation (``max_bond`` only) and various ``absorb`` return
    options, each with their own shortcuts.

    Parameters
    ----------
    x : array-like
        The 2d array to decompose.
    absorb : str or None, optional
        What to compute / where to absorb the singular values:

        - ``None`` / ``'U,s,VH'``: return ``s`` as the middle element of the
          3-tuple, unabsorbed.
        - ``'both'`` / ``'Usq,sqVH'``: absorb ``sqrt(s)`` into both factors.
        - ``'left'`` / ``'Us,VH'``: absorb ``s`` into the left factor,
          leaving the right factor isometric (LQ-like).
        - ``'right'`` / ``'U,sVH'``: absorb ``s`` into the right factor,
          leaving the left factor isometric (QR-like).
        - ``'lorthog'`` / ``'U'``: return only the left isometric factor.
        - ``'rorthog'`` / ``'VH'``: return only the right isometric factor.
        - ``'lfactor'`` / ``'Us'``: return only the left factor with ``s``
          absorbed (the L in an LQ decomposition).
        - ``'rfactor'`` / ``'sVH'``: return only the right factor with ``s``
          absorbed (the R in a QR decomposition).
        - ``'s'``: return only the singular values.

    max_bond : int, optional
        An explicit maximum bond dimension, use -1 for no truncation.
    descending : bool, optional
        Whether to return singular values in descending order, default False.
    right : bool, optional
        Whether to use decompose (xdag @ x: True) or (x @ xdag: False).
        If None (default), then will choose based on shape of x, and if square,
        then based on what is requested in ``absorb``.

    Returns
    -------
    U : array or None
        Left singular vectors, possibly with ``s`` absorbed, or ``None`` if not
        requested.
    s : array or None
        Singular values, or ``None`` if not requested.
    VH : array or None
        Right singular vectors, possibly with ``s`` absorbed, or ``None``
        if not requested.
    """
    xp = get_namespace(x)
    *_, m, n = xp.shape(x)
    xdag = xp.conj(xp.swapaxes(x, -2, -1))
    absorb = _ABSORB_MAP[absorb]

    if right is None:
        if m > n:
            right = True
        elif m < n:
            right = False
        else:
            # avoid division if possible
            right = absorb in (get_VH, get_sVH, get_sqVH, get_Us_VH)

    if right:
        # tall: eigendecompose xdag @ x
        s2, V = xp.linalg.eigh(xdag @ x)
        if 0 < max_bond < min(m, n):
            s2 = s2[..., -max_bond:]
            V = V[..., :, -max_bond:]
        if descending:
            # maybe match svd convention, by default eigh is *ascending*
            s2 = xp.flip(s2, axis=-1)
            V = xp.flip(V, axis=-1)
        s2 = xp.clip(s2, 0.0, None)

        if absorb == get_s:  # 'svals'
            s = xp.sqrt(s2)
            return None, s, None
        if absorb == get_VH:  # 'rorthog'
            VH = xp.conj(xp.swapaxes(V, -2, -1))
            return None, None, VH
        if absorb == get_sVH:  # 'rfactor'
            VH = xp.conj(xp.swapaxes(V, -2, -1))
            s = xp.sqrt(s2)
            sVH = s[..., :, None] * VH
            return None, None, sVH
        if absorb == get_sqVH:  # 'rsqrt'
            sq = xp.sqrt(xp.sqrt(s2))
            VH = xp.conj(xp.swapaxes(V, -2, -1))
            sqVH = sq[..., :, None] * VH
            return None, None, sqVH

        Us = x @ V
        if absorb == get_Us:  # 'lfactor'
            return Us, None, None
        if absorb == get_Us_VH:  # 'left'
            VH = xp.conj(xp.swapaxes(V, -2, -1))
            return Us, None, VH

        # for all other options we need U
        s = xp.sqrt(s2)
        eps = xp.finfo(s.dtype).eps
        if descending:
            smax = s[..., 0:1]
        else:
            smax = s[..., -1:]
        cutoff = smax * eps * max(m, n)
        sinv = safe_inverse(s, cutoff)
        U = Us * sinv[..., None, :]

        if absorb == get_U:  # 'lorthog'
            return U, None, None
        if absorb == get_Usq:  # 'lsqrt'
            sq = xp.sqrt(s)
            Usq = U * sq[..., None, :]
            return Usq, None, None

        # need U and VH for all remaining options
        VH = xp.conj(xp.swapaxes(V, -2, -1))
        if absorb == get_U_s_VH:  # 'full'
            return U, s, VH
        if absorb == get_U_sVH:  # 'right'
            sVH = s[..., :, None] * VH
            return U, None, sVH
        if absorb == get_Usq_sqVH:  # 'both'
            sq = xp.sqrt(s)
            Usq = U * sq[..., None, :]
            sqVH = sq[..., :, None] * VH
            return Usq, None, sqVH

    else:
        # wide: eigendecompose x @ xdag
        s2, U = xp.linalg.eigh(x @ xdag)
        if 0 < max_bond < min(m, n):
            s2 = s2[..., -max_bond:]
            U = U[..., :, -max_bond:]
        if descending:
            # maybe match svd convention, by default eigh is *ascending*
            s2 = xp.flip(s2, axis=-1)
            U = xp.flip(U, axis=-1)
        s2 = xp.clip(s2, 0.0, None)

        if absorb == get_s:  # 'svals'
            s = xp.sqrt(s2)
            return None, s, None
        if absorb == get_U:  # 'lorthog'
            return U, None, None
        if absorb == get_Us:  # 'lfactor'
            s = xp.sqrt(s2)
            Us = U * s[..., None, :]
            return Us, None, None
        if absorb == get_Usq:  # 'lsqrt'
            sq = xp.sqrt(xp.sqrt(s2))
            Usq = U * sq[..., None, :]
            return Usq, None, None

        sVH = xp.conj(xp.swapaxes(U, -2, -1)) @ x
        if absorb == get_sVH:  # 'rfactor'
            return None, None, sVH
        if absorb == get_U_sVH:  # 'right'
            return U, None, sVH

        # for all other options we need VH
        s = xp.sqrt(s2)
        eps = xp.finfo(s.dtype).eps
        if descending:
            smax = s[..., 0:1]
        else:
            smax = s[..., -1:]
        cutoff = smax * eps * max(m, n)
        sinv = safe_inverse(s, cutoff)
        VH = sinv[..., :, None] * sVH

        if absorb == get_VH:  # 'rorthog'
            return None, None, VH
        if absorb == get_U_s_VH:  # 'full'
            return U, s, VH
        if absorb == get_Us_VH:  # 'left'
            Us = U * s[..., None, :]
            return Us, None, VH
        sq = xp.sqrt(s)
        sqVH = sq[..., :, None] * VH
        if absorb == get_Usq_sqVH:  # 'both'
            Usq = U * sq[..., None, :]
            return Usq, None, sqVH
        if absorb == get_sqVH:  # 'rsqrt'
            return None, None, sqVH

    raise ValueError(f"Invalid absorb mode: {absorb}")


@register_split_driver("svd:eig")
@compose
def svd_via_eig_truncated(
    x,
    cutoff=-1.0,
    cutoff_mode=cutoff_mode_rsum2,
    max_bond=-1,
    absorb=get_Usq_sqVH,
    renorm=0,
    info=None,
):
    """Singular value decomposition of raw 2d array ``x``, via hermitian
    eigen-decomposition of the Gram matrix, with optional truncation based on
    `max_bond` and/or dynamically on `cutoff`.

    Parameters
    ----------
    cutoff : float, optional
        Singular value cutoff threshold, if ``cutoff <= 0.0``, then only
        ``max_bond`` is used.
    cutoff_mode : {1, 2, 3, 4, 5, 6}, optional
        How to perform the truncation based on ``cutoff``:

        - 1 / 'abs': trim values below ``cutoff``
        - 2 / 'rel': trim values below ``s[0] * cutoff``
        - 3 / 'sum2': trim s.t. ``sum(s_trim**2) < cutoff``.
        - 4 / 'rsum2': trim s.t. ``sum(s_trim**2) < sum(s**2) * cutoff``.
        - 5 / 'sum1': trim s.t. ``sum(s_trim**1) < cutoff``.
        - 6 / 'rsum1': trim s.t. ``sum(s_trim**1) < sum(s**1) * cutoff``.

    max_bond : int, optional
        An explicit maximum bond dimension, use -1 for none.
    absorb : int or None, optional
        How to absorb the singular values, as a pre-converted numeric code
        (``get_Us_VH=-1``: left, ``get_Usq_sqVH=0``: both,
        ``get_U_sVH=1``: right, ``None``: return separately). Use
        ``array_split`` with string aliases (e.g. ``'left'``, ``'both'``,
        ``'right'``, ``None``) for a friendlier interface.
    renorm : int, optional
        Whether to renormalize the kept singular values. ``0`` means
        no renormalization, ``1`` maintains the trace norm, ``2``
        maintains the Frobenius norm.
    info : dict or None, optional
        If a dict is passed, store truncation info in the dict. Currently only
        supports the key 'error' for the truncation error.

    Returns
    -------
    left : array_like or None
    s : array_like or None
    right : array_like or None
    """
    absorb = _ABSORB_MAP[absorb]
    info = parse_info_extras(info, ("error",))
    need_full_spectrum = (cutoff > 0.0) or (renorm > 0) or "error" in info

    if need_full_spectrum:
        # 1. compute full svd ...
        U, s, VH = svd_via_eig(
            x,
            absorb=None,
            max_bond=-1,
            descending=True,
        )
        # 2. ... then truncate separately
        return _trim_and_renorm_svd_result(
            U,
            s,
            VH,
            cutoff=cutoff,
            cutoff_mode=cutoff_mode,
            max_bond=max_bond,
            absorb=absorb,
            renorm=renorm,
            info=info,
        )
    else:
        # perform in one step, possibly taking advantage of shortcuts
        return svd_via_eig(
            x, absorb=absorb, max_bond=max_bond, descending=False
        )


@njit  # pragma: no cover
def _svd_via_eig_numba(
    x,
    absorb=get_U_s_VH,
    max_bond=-1,
    descending=True,
    right=None,
):
    """Numba-accelerated SVD via eigendecomposition of the Gram matrix,
    with static truncation (``max_bond`` only) and all ``absorb`` return
    options, each with their own shortcuts. Mirrors the structure of
    ``svd_via_eig``.

    Parameters
    ----------
    x : array-like
        The 2d array to decompose.
    absorb : int or None, optional
        Numeric absorb code controlling what to compute / return.
    max_bond : int, optional
        An explicit maximum bond dimension, use -1 for no truncation.
    descending : bool, optional
        Whether to return singular values in descending order.

    Returns
    -------
    U : array or None
    s : array or None
    VH : array or None
    """
    m, n = x.shape

    if right is None:
        if m > n:
            right = True
        elif m < n:
            right = False
        else:
            # square: avoid division if possible
            right = (absorb is not None) and (
                absorb == get_Us or absorb == get_Us_VH or absorb == get_VH
            )
    else:
        right = bool(right)

    if right:
        # tall: eigendecompose xdag @ x
        xx = dag_numba(x) @ x
        s2, V = np.linalg.eigh(xx)
        if 0 < max_bond < min(m, n):
            s2 = s2[-max_bond:]
            V = np.ascontiguousarray(V[:, -max_bond:])
        if descending:
            s2 = s2[::-1]
            V = np.ascontiguousarray(V[:, ::-1])
        s2 = np.maximum(s2, 0.0)
        if absorb == get_s:  # 'svals'
            return None, np.sqrt(s2), None
        if absorb == get_VH:  # 'rorthog'
            return None, None, dag_numba(V)
        if absorb == get_sVH:  # 'rfactor'
            return None, None, ldmul_numba(np.sqrt(s2), dag_numba(V))
        V = np.ascontiguousarray(V)
        Us = x @ V
        if absorb == get_Us:  # 'lfactor'
            return Us, None, None
        if absorb == get_Us_VH:  # 'left'
            return Us, None, dag_numba(V)
        s = np.sqrt(s2)

        eps = np.finfo(s.dtype).eps
        cutoff = np.max(s) * eps * max(m, n)
        sinv = safe_inverse_numba(s, cutoff)
        U = Us * sinv[None, :]

        if absorb == get_U:  # 'lorthog'
            return U, None, None
        VH = dag_numba(V)
        if absorb is None:  # 'full'
            return U, s, VH
        if absorb == get_U_sVH:  # 'right'
            return U, None, ldmul_numba(s, VH)
        sq = np.sqrt(s)
        if absorb == get_Usq_sqVH:  # 'both'
            return rdmul_numba(U, sq), None, ldmul_numba(sq, VH)
        if absorb == get_Usq:  # 'lsqrt'
            return rdmul_numba(U, sq), None, None
        if absorb == get_sqVH:  # 'rsqrt'
            return None, None, ldmul_numba(sq, VH)
    else:
        # wide: eigendecompose x @ xdag
        xx = x @ dag_numba(x)
        s2, U = np.linalg.eigh(xx)
        if 0 < max_bond < min(m, n):
            s2 = s2[-max_bond:]
            U = np.ascontiguousarray(U[:, -max_bond:])
        if descending:
            s2 = s2[::-1]
            U = np.ascontiguousarray(U[:, ::-1])
        # clip small/negative eigenvalues
        s2 = np.maximum(s2, 0.0)
        if absorb == get_s:  # 'svals'
            return None, np.sqrt(s2), None
        if absorb == get_U:  # 'lorthog'
            return U, None, None
        if absorb == get_Us:  # 'lfactor'
            return rdmul_numba(U, np.sqrt(s2)), None, None
        U = np.ascontiguousarray(U)
        sVH = dag_numba(U) @ x
        if absorb == get_sVH:  # 'rfactor'
            return None, None, sVH
        if absorb == get_U_sVH:  # 'right'
            return U, None, sVH
        s = np.sqrt(s2)

        eps = np.finfo(s.dtype).eps
        cutoff = np.max(s) * eps * max(m, n)
        sinv = safe_inverse_numba(s, cutoff)
        VH = sinv[:, None] * sVH

        if absorb == get_VH:  # 'rorthog'
            return None, None, VH
        if absorb is None:  # 'full'
            return U, s, VH
        if absorb == get_Us_VH:  # 'left'
            return rdmul_numba(U, s), None, VH
        sq = np.sqrt(s)
        if absorb == get_Usq_sqVH:  # 'both'
            return rdmul_numba(U, sq), None, ldmul_numba(sq, VH)
        if absorb == get_Usq:  # 'lsqrt'
            return rdmul_numba(U, sq), None, None
        if absorb == get_sqVH:  # 'rsqrt'
            return None, None, ldmul_numba(sq, VH)

    # fallback (should not be reached)
    return None, None, None


@njit  # pragma: no cover
def _svd_via_eig_truncated_numba(
    x,
    cutoff=-1.0,
    cutoff_mode=cutoff_mode_rsum2,
    max_bond=-1,
    absorb=get_Usq_sqVH,
    renorm=0,
    calc_error=False,
):
    """SVD-split via eigen-decomposition, with optional dynamic truncation.
    Uses ``_svd_via_eig_numba`` shortcuts when the full spectrum is not
    needed.
    """
    need_full_spectrum = (cutoff > 0.0) or (renorm > 0) or calc_error

    if need_full_spectrum:
        # 1. compute full svd ...
        U, s, VH = _svd_via_eig_numba(
            x,
            absorb=None,
            max_bond=-1,
            descending=True,
        )
        # 2. ... then truncate separately
        return _trim_and_renorm_svd_result_numba(
            U,
            s,
            VH,
            cutoff=cutoff,
            cutoff_mode=cutoff_mode,
            max_bond=max_bond,
            absorb=absorb,
            renorm=renorm,
            calc_error=calc_error,
        )
    else:
        # perform in one step, possibly taking advantage of shortcuts
        U, s, VH = _svd_via_eig_numba(
            x,
            absorb=absorb,
            max_bond=max_bond,
            descending=False,
        )
        return U, s, VH, None


@svd_via_eig_truncated.register("numpy")
def svd_via_eig_truncated_numpy(
    x,
    cutoff=-1.0,
    cutoff_mode=cutoff_mode_rsum2,
    max_bond=-1,
    absorb=get_Usq_sqVH,
    renorm=0,
    info=None,
):
    if x.ndim > 2:
        # XXX: batch truncation not implemented in numba version yet
        return svd_via_eig_truncated._default_fn(
            x,
            cutoff=cutoff,
            cutoff_mode=cutoff_mode,
            max_bond=max_bond,
            absorb=absorb,
            renorm=renorm,
            info=info,
        )

    info = parse_info_extras(info, ("error",))

    calc_error = "error" in info
    U, s, VH, error = _svd_via_eig_truncated_numba(
        x,
        cutoff,
        cutoff_mode,
        max_bond,
        absorb,
        renorm,
        calc_error=calc_error,
    )

    if calc_error:
        info["error"] = error

    return U, s, VH


@register_svals_driver("svd:eig")
@njit  # pragma: no cover
def svdvals_eig(x):
    """SVD-decomposition via eigen, but return singular values only."""
    if x.shape[0] > x.shape[1]:
        s2 = np.linalg.eigvalsh(dag_numba(x) @ x)
    else:
        s2 = np.linalg.eigvalsh(x @ dag_numba(x))

    s2[s2 < 0.0] = 0.0
    return np.sqrt(s2[::-1])


# -------------------------------- rand svd --------------------------------- #


@register_split_driver("svd:rand")
@compose
def svd_rand_truncated(
    x,
    max_bond,
    absorb=get_Usq_sqVH,
    oversample=10,
    num_iterations=2,
    method_lorthog="qr",
    method_reduced="svd",
    right=None,
    lorthog_opts=None,
    reduced_opts=None,
    seed=None,
):
    """Singular value decomposition of raw 2d array ``x``, via randomized
    sketching, with static truncation (``max_bond`` only) and various
    ``absorb`` return options, each with their own shortcuts. The speedup
    over full SVD is proportional to the truncation.

    Parameters
    ----------
    x : array_like
        The 2d array to decompose.
    max_bond : int
        An explicit maximum bond dimension / target rank for the randomized
        sketch. You can use ``None`` or a negative value to indicate no
        truncation, though this is not recommended as there is no speedup.
    absorb : str or None, optional
        What to compute / where to absorb the singular values:

        - ``None`` / ``'U,s,VH'``: return ``s`` as the middle element of the
          3-tuple, unabsorbed.
        - ``'both'`` / ``'Usq,sqVH'``: absorb ``sqrt(s)`` into both factors.
        - ``'left'`` / ``'Us,VH'``: absorb ``s`` into the left factor,
          leaving the right factor isometric (LQ-like).
        - ``'right'`` / ``'U,sVH'``: absorb ``s`` into the right factor,
          leaving the left factor isometric (QR-like).
        - ``'lorthog'`` / ``'U'``: return only the left isometric factor.
        - ``'rorthog'`` / ``'VH'``: return only the right isometric factor.
        - ``'lfactor'`` / ``'Us'``: return only the left factor with ``s``
          absorbed (the L in an LQ decomposition).
        - ``'rfactor'`` / ``'sVH'``: return only the right factor with ``s``
          absorbed (the R in a QR decomposition).
        - ``'s'``: return only the singular values.

    oversample : int, optional
        Extra dimensions for the random sketch to improve accuracy,
        default 10.
    num_iterations : int, optional
        Number of power iterations to improve accuracy, default 2.
    method_reduced : str, optional
        The decomposition method to use for the reduced matrix, e.g.
        ``'svd'``, ``'svd:eig'``, etc. Default is ``'svd'``.
    method_lorthog : str, optional
        The decomposition method to use for forming the orthonormal basis
        of the random sketch, e.g. ``'qr'``, ``'svd:eig'``, ``'svd'``,
        etc. Default is ``'qr'``.
    right : bool, optional
        Whether to sketch from the right or left. If None (default), then will
        choose based on value of ``absorb``.
    lorthog_opts : dict or None, optional
        Extra options to pass to the decomposition method for forming the
        orthonormal basis of the random sketch.
    reduced_opts : dict or None, optional
        Extra options to pass to the decomposition method for the reduced
        matrix.
    seed : int, Generator or None, optional
        Random seed or existing generator for reproducibility.

    Returns
    -------
    left : array_like or None
    s : array_like or None
    right : array_like or None
    """
    absorb = _ABSORB_MAP[absorb]
    if max_bond is None:
        max_bond = -1

    lorthog_opts = ensure_dict(lorthog_opts)
    lorthog_opts.setdefault("method", method_lorthog)

    xp = get_namespace(x)
    *batch_dims, m, n = xp.shape(x)

    # determine target rank and sketch size
    if max_bond < 0:
        warnings.warn(
            "Using 'svd:rand' without `max_bond` is inefficient, "
            "consider simply using 'svd' or 'svd:eig' instead."
        )
        k = min(m, n)
    else:
        k = min(m, n, max_bond)
    k_sketch = min(m, n, k + oversample)

    if right is None:
        # avoid svd on reduced factor if possible
        if absorb in (get_U_sVH, get_U, get_sVH):
            right = True
        elif absorb in (get_Us_VH, get_Us, get_VH):
            right = False
        else:
            # note unlike svd via eig, tall vs wide is secondary consideration
            right = m > n

    rng = xp.random.default_rng(seed)

    if right:
        # tall: sketch from the right
        omega = rng.normal(size=(*batch_dims, n, k_sketch))
        y = x @ omega
        if num_iterations:
            xdag = xp.conj(xp.swapaxes(x, -2, -1))
            for _ in range(num_iterations):
                y = xdag @ y
                y = x @ y

        # form orthonormal basis of column space / 'U'
        Q, _, _ = array_split(y, absorb=get_U, **lorthog_opts)

        # X ≈ Q @ B, maybe shortcut for some absorb if no truncation needed
        if k >= k_sketch:
            if absorb == get_U_sVH:  # 'right'
                return Q, None, xp.conj(xp.swapaxes(Q, -2, -1)) @ x
            if absorb == get_sVH:  # 'rfactor'
                return None, None, xp.conj(xp.swapaxes(Q, -2, -1)) @ x
            if absorb == get_U:  # 'lorthog'
                return Q, None, None

        # form reduced factor
        B = xp.conj(xp.swapaxes(Q, -2, -1)) @ x
    else:
        # wide: sketch from the left
        omega = rng.normal(size=(*batch_dims, k_sketch, m))
        y = omega @ x
        if num_iterations:
            xdag = xp.conj(xp.swapaxes(x, -2, -1))
            for _ in range(num_iterations):
                y = y @ xdag
                y = y @ x
        y = xp.conj(xp.swapaxes(y, -2, -1))

        # form orthonormal basis of row space / 'V'
        Q, _, _ = array_split(y, absorb=get_U, **lorthog_opts)

        # X ≈ B @ Qdag, maybe shortcut for some absorb if no truncation needed
        if k >= k_sketch:
            if absorb == get_Us_VH:  # 'left'
                return x @ Q, None, xp.conj(xp.swapaxes(Q, -2, -1))
            if absorb == get_Us:  # 'lfactor'
                return x @ Q, None, None
            if absorb == get_VH:  # 'rorthog'
                return None, None, xp.conj(xp.swapaxes(Q, -2, -1))

        # form reduced factor
        B = x @ Q

    reduced_opts = ensure_dict(reduced_opts)
    reduced_opts.setdefault("method", method_reduced)
    reduced_opts.setdefault("cutoff", 0.0)

    # decompose and maybe further truncate reduced matrix
    U, s, VH = array_split(B, absorb=absorb, max_bond=k, **reduced_opts)

    # expand back out from reduced space
    if (U is not None) and right:
        U = Q @ U
    if (VH is not None) and not right:
        VH = VH @ xp.conj(xp.swapaxes(Q, -2, -1))

    return U, s, VH


# ---------------------------------- eigh ----------------------------------- #


@register_split_driver("eigh")
@compose
def eigh_truncated(
    x,
    cutoff=-1.0,
    cutoff_mode=cutoff_mode_rsum2,
    max_bond=-1,
    absorb=get_Usq_sqVH,
    renorm=0,
    positive=0,
):
    """SVD-like decomposition using hermitian eigen-decomposition, only works
    if ``x`` is hermitian.

    Returns
    -------
    left : array_like or None
    s : array_like or None
    right : array_like or None
    """
    xp = get_namespace(x)
    s, U = xp.linalg.eigh(x)

    # make sure largest singular value first
    if not positive:
        idx = xp.argsort(-xp.abs(s), axis=-1)
        s = xp.take_along_axis(s, idx, axis=-1)
        U = xp.take_along_axis(U, idx[..., None, :], axis=-1)
    else:
        # assume all positive, simply reverse
        s = xp.flip(s, axis=-1)
        U = xp.flip(U, axis=-1)

        if absorb in (get_Usq_sqVH, get_Usq, get_sqVH):
            # operator assumed positive, but small negative eignvalues
            # will cause problems when taking sqrt, so clip to zero
            s = xp.clip(s, 0.0, None)

    VH = xp.conj(xp.swapaxes(U, -2, -1))

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
        xp=xp,
    )


@njit  # pragma: no cover
def eigh_truncated_numba(
    x,
    cutoff=-1.0,
    cutoff_mode=cutoff_mode_rsum2,
    max_bond=-1,
    absorb=get_Usq_sqVH,
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

        if absorb == get_Usq_sqVH or absorb == get_Usq or absorb == get_sqVH:
            # operator assumed positive, but small negative eignvalues
            # will cause problems when taking sqrt, so clip to zero
            s[s < 0.0] = 0.0

    VH = dag_numba(U)

    # XXX: better to absorb phase in V and return positive 'values'?
    # VH = ldmul_numba(sgn_numba(s), dag_numba(U))
    # s = np.abs(s)

    U, s, VH, _ = _trim_and_renorm_svd_result_numba(
        U, s, VH, cutoff, cutoff_mode, max_bond, absorb, renorm, use_abs=True
    )

    return U, s, VH


@eigh_truncated.register("numpy")
def eigh_truncated_numpy(
    x,
    cutoff=-1.0,
    cutoff_mode=cutoff_mode_rsum2,
    max_bond=-1,
    absorb=get_Usq_sqVH,
    renorm=0,
    positive=0,
):
    if x.ndim > 2:
        # XXX: batch truncation not implemented in numba version yet
        return eigh_truncated._default_fn(
            x,
            cutoff=cutoff,
            cutoff_mode=cutoff_mode,
            max_bond=max_bond,
            absorb=absorb,
            renorm=renorm,
            positive=positive,
        )
    return eigh_truncated_numba(
        x, cutoff, cutoff_mode, max_bond, absorb, renorm, positive
    )


# ---------------------- QR and LQ like decompositions ---------------------- #


@register_split_driver("qr", default_absorb=get_U_sVH)
@compose
def qr_stabilized(x, absorb=get_U_sVH, stabilized=True, **kwargs):
    """QR-decomposition, with stabilized R factor.

    Parameters
    ----------
    x : array_like
        The 2d array to decompose.
    absorb : str or int, optional
        What form to compute / where to 'absorb' the singular values:

        - ``"right"`` / ``'U,sVH'``: usual QR decomposition.
        - ``"lorthog"`` / ``"U"``: just the left isometric factor (Q).
        - ``"rfactor"`` / ``"sVH"``: just the right factor (R).
        - ``"left"`` / ``'Us,VH'``: LQ-like decomposition.
        - ``"lfactor"`` / ``"Us"``: just the left factor (L) of LQ.
        - ``"rorthog"`` / ``"VH"``: just the right isometric factor (Q) of LQ.

    stabilized : bool, optional
        Whether to stabilize the diagonal of R by setting its phase to 1 and
        absorbing the phase into Q. This can improve the numerical stability
        of gradients for example.
    kwargs
        Extra kwargs to pass to the underlying QR function.

    Returns
    -------
    left : array_like
        The left isometric factor (Q).
    s : None
    right : array_like
        The right upper triangular factor (R).
    """
    absorb = _ABSORB_MAP[absorb]

    if absorb in (get_U_sVH, get_U, get_sVH):
        transposed = False
    elif absorb in (get_Us_VH, get_Us, get_VH):
        # performing LQ-like decomposition
        transposed = True
    else:
        raise ValueError(f"Invalid absorb mode for qr_stabilized: {absorb}")

    xp = get_namespace(x)

    if transposed:
        x = xp.swapaxes(x, -2, -1)
        absorb = _ABSORB_TRANSPOSE_MAP[absorb]

    Q, R = xp.linalg.qr(x, **kwargs)

    if not stabilized:
        # set factors directly
        if absorb == get_U:
            right = None
        else:
            right = R
        if absorb == get_sVH:
            left = None
        else:
            left = Q
    else:
        # stabilize the diagonal of R -> set phase to 1, absorbing into Q
        rd = xp.diagonal(R, axis1=-2, axis2=-1)
        phase = sgn(rd)

        if absorb != get_sVH:
            left = Q * xp.conj(phase)[..., None, :]
        else:
            left = None

        if absorb != get_U:
            right = phase[..., :, None] * R
        else:
            right = None

    if transposed:
        # swap and transpose back
        QT, LT = left, right
        if LT is not None:
            left = xp.swapaxes(LT, -2, -1)
        else:
            left = None
        if QT is not None:
            right = xp.swapaxes(QT, -2, -1)
        else:
            right = None

    return left, None, right


@njit  # pragma: no cover
def _qr_stabilized_numba(x, absorb=get_U_sVH, stabilized=True):
    Q, R = np.linalg.qr(x)

    if stabilized:
        if absorb == get_U:
            for i in range(R.shape[0]):
                rii = R[i, i]
                phase = sgn_numba(rii)
                if phase != 1.0:
                    Q[:, i] *= np.conj(phase)
        elif absorb == get_sVH:
            for i in range(R.shape[0]):
                rii = R[i, i]
                phase = sgn_numba(rii)
                if phase != 1.0:
                    R[i, i:] *= phase
        else:
            for i in range(R.shape[0]):
                rii = R[i, i]
                phase = sgn_numba(rii)
                if phase != 1.0:
                    Q[:, i] *= np.conj(phase)
                    R[i, i:] *= phase

    if absorb == get_U:
        return Q, None, None

    if absorb == get_sVH:
        return None, None, R

    return Q, None, R


@njit  # pragma: no cover
def _lq_stabilized_numba(x, absorb=get_Us_VH, stabilized=True):
    if absorb == get_Us:
        absorb_t = get_sVH
    elif absorb == get_VH:
        absorb_t = get_U
    else:
        absorb_t = get_U_sVH
    Q, _, L = _qr_stabilized_numba(x.T, absorb=absorb_t, stabilized=stabilized)
    if absorb == get_Us:
        return L.T, None, None
    elif absorb == get_VH:
        return None, None, Q.T
    else:
        return L.T, None, Q.T


@qr_stabilized.register("numpy")
def qr_stabilized_numpy(x, absorb=get_U_sVH, stabilized=True, **kwargs):
    if x.ndim > 2:
        # XXX: batch not implemented in numba version yet
        return qr_stabilized._default_fn(
            x,
            absorb=absorb,
            stabilized=stabilized,
            **kwargs,
        )

    if absorb in (get_U_sVH, get_U, get_sVH):
        # can be handled directly in qr
        return _qr_stabilized_numba(x, absorb=absorb, stabilized=stabilized)
    elif absorb in (get_Us_VH, get_Us, get_VH):
        # perform LQ via transposed QR
        return _lq_stabilized_numba(x, absorb=absorb, stabilized=stabilized)
    else:
        raise ValueError(f"Invalid absorb mode for qr_stabilized: {absorb}")


@qr_stabilized.register("autoray.lazy")
@lazy.core.lazy_cache("qr_stabilized")
def qr_stabilized_lazy(x, absorb=get_U_sVH, **kwargs):
    *batch_dims, m, n = x.shape
    k = min(m, n)
    kwargs["absorb"] = absorb
    lqrs = x.to(
        fn=get_lib_fn(x.backend, "qr_stabilized"),
        args=(x,),
        kwargs=kwargs,
        shape=(3,),
    )
    if absorb in _RETURNS_LEFT_ABSORBS:
        Q = lqrs.to(operator.getitem, (lqrs, 0), shape=(*batch_dims, m, k))
    else:
        Q = None
    if absorb in _RETURNS_RIGHT_ABSORBS:
        R = lqrs.to(operator.getitem, (lqrs, 2), shape=(*batch_dims, k, n))
    else:
        R = None
    return Q, None, R


# ---------------------- cholesky based decompositions ---------------------- #


def _cholesky_maybe_with_diag_shift(x, absorb=get_Usq_sqVH, shift=0.0):
    xp = get_namespace(x)

    if shift < 0.0:
        # auto compute
        shift = xp.finfo(x.dtype).eps

    if shift > 0.0:
        trace = xp.trace(x, axis1=-2, axis2=-1)[..., None, None]
        try:
            # if possible avoid accumulating regulization gradient
            trace = xp.stop_gradient(trace)
        except (ImportError, AttributeError):
            pass
        x = x + shift * trace * xp.eye(x.shape[-1])

    if absorb == get_sqVH:
        # can compute the right factor directly using upper
        return None, None, xp.linalg.cholesky(x, upper=True)

    left = xp.linalg.cholesky(x, upper=False)

    if absorb == get_Usq:
        return left, None, None

    if absorb == get_Usq_sqVH:
        # compute upper from lower
        right = xp.conj(left).swapaxes(-2, -1)
        return left, None, right

    raise ValueError(
        f"Invalid absorb={absorb} in cholesky_regularized. Should be one "
        "of 'both'/'get_Usq_sqVH', 'lsqrt'/'get_Usq' or rsqrt'/'get_sqVH'."
    )


@register_split_driver("cholesky")
@compose
def cholesky_regularized(x, absorb=get_Usq_sqVH, shift=True):
    """Cholesky decomposition, only works if ``x`` is positive definite. The
    ``shift`` parameter controls optional regularization for close to
    singular matrices.

    Parameters
    ----------
    x : array_like
        The 2D array to decompose.
    absorb : int, optional
        How to absorb the factors. The valid options are:

        - ``"both"`` or ``get_Usq_sqVH``: return (L, None, L^H)
        - ``"lsqrt"`` or ``get_Usq``: return (L, None, None)
        - ``"rsqrt"`` or ``get_sqVH``: return (None, None, L^H)

    shift : "auto", bool or float, optional
        Whether to add a small shift to the diagonal of ``x`` for
        regularization. The valid options are:

        - ``True``: always add a small shift computed as above.
        - ``"auto"``: try without shift, if it fails, then add a small shift
          computed as ``trace(x) * eps``, ``eps`` is the machine epsilon for
          the dtype of ``x``.
        - ``False``: never add a shift, just try the Cholesky decomposition
          directly.
        - float: use the provided value as a relative shift, i.e. add
          ``shift * trace(x)`` to the diagonal of ``x``.

    Returns
    -------
    left : array_like or None
        The lower triangular Cholesky factor (L).
    s : None
    right : array_like or None
        The conjugate transpose of L (L^H).
    """
    absorb = _ABSORB_MAP[absorb]

    if shift == "auto":
        try:
            # try without shift
            return _cholesky_maybe_with_diag_shift(x, absorb, shift=0.0)
        except Exception as e:
            warnings.warn(
                f"Cholesky decomposition failed with error: {e}. "
                "retrying with small regularization added to the diagonal."
            )
            return _cholesky_maybe_with_diag_shift(x, absorb, shift=-1.0)

    shift = {False: 0.0, True: -1.0}.get(shift, shift)
    return _cholesky_maybe_with_diag_shift(x, absorb, shift=shift)


@njit  # pragma: no cover
def _cholesky_regularized_numba(x, absorb=get_Usq_sqVH, shift=-1.0):
    if shift < 0.0:
        # auto compute
        shift = np.finfo(x.dtype).eps

    if shift > 0.0:
        shift = shift * np.trace(x)
        x = x.copy()  # avoid modifying input in-place
        for i in range(x.shape[0]):
            x[i, i] += shift

    L = np.linalg.cholesky(x)

    if absorb == get_Usq:
        return L, None, None
    if absorb == get_sqVH:
        return None, None, dag_numba(L)
    # absorb == get_Usq_sqVH
    return L, None, dag_numba(L)


@cholesky_regularized.register("numpy")
def cholesky_regularized_numpy(x, absorb=get_Usq_sqVH, shift=True):
    if x.ndim > 2:
        # XXX:
        return cholesky_regularized._default_fn(x, absorb=absorb, shift=shift)

    absorb = _ABSORB_MAP[absorb]
    if shift == "auto":
        try:
            # try without shift
            return _cholesky_regularized_numba(x, absorb, shift=0.0)
        except Exception as e:
            warnings.warn(
                f"Cholesky decomposition failed with error: {e}. "
                "retrying with small regularization added to the diagonal."
            )
            return _cholesky_regularized_numba(x, absorb, shift=-1.0)
    shift = {False: 0.0, True: -1.0}.get(shift, shift)
    return _cholesky_regularized_numba(x, absorb, shift=shift)


@register_split_driver("qr:cholesky", default_absorb=get_U_sVH)
@compose
def qr_via_cholesky(x, absorb=get_Us_VH, shift=True, solve_triangular=True):
    """Perform a QR-like or LQ-like decomposition via Cholesky decomposition
    of the Gram matrix.
    """
    absorb = _ABSORB_MAP[absorb]

    if absorb in (get_U_sVH, get_U, get_sVH):
        # performing QR-like decomposition
        #     NOTE: the cholesky method naturally gives us LQ-like
        #     decomposition so *this* is the case we transpose
        transposed = True
    elif absorb in (get_Us_VH, get_Us, get_VH):
        # performing LQ-like decomposition
        transposed = False
    else:
        raise ValueError(f"Invalid absorb mode for qr_via_cholesky: {absorb}")

    xp = get_namespace(x)

    if transposed:
        absorb = _ABSORB_TRANSPOSE_MAP[absorb]
        # avoid transposing twice
        xT = xp.swapaxes(x, -2, -1)
        xx = xT @ xp.conj(x)
        x = xT
    else:
        xx = x @ xp.conj(xp.swapaxes(x, -2, -1))

    *_, m, n = xp.shape(x)
    if m > n:
        warnings.warn(
            f"qr_via_cholesky not well-defined for tall matrices ({m} > {n})."
        )

    L, _, _ = cholesky_regularized(xx, absorb=get_Usq, shift=shift)

    if absorb != get_Us:
        # need Q / rorthog
        if solve_triangular:
            right = xp.scipy.linalg.solve_triangular(L, x, lower=True)
        else:
            right = xp.linalg.solve(L, x)
    else:
        right = None

    if absorb != get_VH:
        # need lfactor
        left = L
    else:
        left = None

    if transposed:
        # swap and transpose back
        RT, QT = left, right
        if QT is not None:
            left = xp.swapaxes(QT, -2, -1)
        else:
            left = None
        if RT is not None:
            right = xp.swapaxes(RT, -2, -1)
        else:
            right = None

    return left, None, right


# ------------------------ iterative decompositions ------------------------- #


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


@register_split_driver("svds", sparse=True)
def svds(
    x,
    cutoff=0.0,
    cutoff_mode=cutoff_mode_rsum2,
    max_bond=-1,
    absorb=get_Usq_sqVH,
    renorm=0,
):
    """SVD-decomposition using iterative methods. Allows the
    computation of only a certain number of singular values, e.g. max_bond,
    from the get-go, and is thus more efficient. Can also supply
    ``scipy.sparse.linalg.LinearOperator``.

    Returns
    -------
    left : array_like or None
    s : array_like or None
    right : array_like or None
    """
    k = _choose_k(x, cutoff, max_bond)

    if k == "full":
        if not isinstance(x, np.ndarray):
            x = x.to_dense()
        return svd_truncated(x, cutoff, cutoff_mode, max_bond, absorb)

    U, s, VH = base_linalg.svds(x, k=k)
    U, s, VH, _ = _trim_and_renorm_svd_result_numba(
        U, s, VH, cutoff, cutoff_mode, max_bond, absorb, renorm
    )
    return U, s, VH


@register_split_driver("isvd", sparse=True)
def isvd(
    x,
    cutoff=0.0,
    cutoff_mode=cutoff_mode_rsum2,
    max_bond=-1,
    absorb=get_Usq_sqVH,
    renorm=0,
):
    """SVD-decomposition using interpolative matrix random methods. Allows the
    computation of only a certain number of singular values, e.g. max_bond,
    from the get-go, and is thus more efficient. Can also supply
    ``scipy.sparse.linalg.LinearOperator``.

    Returns
    -------
    left : array_like or None
    s : array_like or None
    right : array_like or None
    """
    k = _choose_k(x, cutoff, max_bond)

    if k == "full":
        if not isinstance(x, np.ndarray):
            x = x.to_dense()
        return svd_truncated(x, cutoff, cutoff_mode, max_bond, absorb)

    U, s, V = sli.svd(x, k)
    VH = dag_numba(V)
    U, s, VH, _ = _trim_and_renorm_svd_result_numba(
        U, s, VH, cutoff, cutoff_mode, max_bond, absorb, renorm
    )
    return U, s, VH


def _rsvd_numpy(
    x,
    cutoff=0.0,
    cutoff_mode=cutoff_mode_rsum2,
    max_bond=-1,
    absorb=get_Usq_sqVH,
    renorm=0,
):
    if max_bond > 0:
        if cutoff > 0.0:
            # adapt and block
            U, s, VH = rand_linalg.rsvd(x, cutoff, k_max=max_bond)
        else:
            U, s, VH = rand_linalg.rsvd(x, max_bond)
    else:
        U, s, VH = rand_linalg.rsvd(x, cutoff)

    U, s, VH, _ = _trim_and_renorm_svd_result_numba(
        U, s, VH, cutoff, cutoff_mode, max_bond, absorb, renorm
    )
    return U, s, VH


@register_split_driver("rsvd", sparse=True)
def rsvd(
    x,
    cutoff=0.0,
    cutoff_mode=cutoff_mode_rsum2,
    max_bond=-1,
    absorb=get_Usq_sqVH,
    renorm=0,
):
    """SVD-decomposition using randomized methods (due to Halko). Allows the
    computation of only a certain number of singular values, e.g. max_bond,
    from the get-go, and is thus more efficient. Can also supply
    ``scipy.sparse.linalg.LinearOperator``.

    Returns
    -------
    left : array_like or None
    s : array_like or None
    right : array_like or None
    """
    if isinstance(x, (np.ndarray, spla.LinearOperator)):
        return _rsvd_numpy(x, cutoff, cutoff_mode, max_bond, absorb, renorm)

    xp = get_namespace(x)
    U, s, VH = xp.linalg.rsvd(x, max_bond)
    return _trim_and_renorm_svd_result(
        U,
        s,
        VH,
        cutoff,
        cutoff_mode,
        max_bond,
        absorb,
        renorm,
        xp=xp,
    )


@register_split_driver("eigsh", sparse=True)
def eigsh(
    x,
    cutoff=0.0,
    cutoff_mode=cutoff_mode_rsum2,
    max_bond=-1,
    absorb=get_Usq_sqVH,
    renorm=0,
):
    """SVD-decomposition using iterative hermitian eigen decomp, thus assuming
    that ``x`` is hermitian. Allows the computation of only a certain number of
    singular values, e.g. max_bond, from the get-go, and is thus more
    efficient. Can also supply ``scipy.sparse.linalg.LinearOperator``.

    Returns
    -------
    left : array_like or None
    s : array_like or None
    right : array_like or None
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
    U, s, V, _ = _trim_and_renorm_svd_result_numba(
        U, s, V, cutoff, cutoff_mode, max_bond, absorb, renorm
    )
    return U, s, V


# ------------------------ misc other decompositions ------------------------ #


@register_split_driver("lu")
@compose
def lu_truncated(
    x,
    cutoff=-1.0,
    cutoff_mode=cutoff_mode_rsum2,
    max_bond=-1,
    absorb=get_Usq_sqVH,
    renorm=0,
):
    """LU-decomposition with optional truncation.

    Returns
    -------
    left : array_like
        The permuted lower triangular factor (PL).
    s : None
    right : array_like
        The upper triangular factor (U).
    """
    if absorb != get_Usq_sqVH:
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

    xp = get_namespace(x)
    PL, U = xp.scipy.linalg.lu(x, permute_l=True)

    sl = xp.sum(xp.abs(PL), axis=0)
    su = xp.sum(xp.abs(U), axis=1)

    if cutoff_mode == 2:
        abs_cutoff_l = cutoff * xp.max(sl)
        abs_cutoff_u = cutoff * xp.max(su)
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


@register_split_driver("polar_right", default_absorb=get_U_sVH)
@compose
def polar_right(x):
    """Polar decomposition of ``x`` as x = U @ P, where U is unitary and P is
    positive semidefinite.

    Returns
    -------
    left : array_like
        The unitary factor (U).
    s : None
    right : array_like
        The positive semidefinite factor (P).
    """
    xp = get_namespace(x)
    W, s, VH = xp.linalg.svd(x)
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


@register_split_driver("polar_left", default_absorb=get_Us_VH)
@compose
def polar_left(x):
    """Polar decomposition of ``x`` as x = P @ U, where U is unitary and P is
    positive semidefinite.

    Returns
    -------
    left : array_like
        The positive semidefinite factor (P).
    s : None
    right : array_like
        The unitary factor (U).
    """
    xp = get_namespace(x)
    W, s, VH = xp.linalg.svd(x)
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
    xp = get_namespace(X)
    # eigen decompose X -> V w V^-1
    el, ev = xp.linalg.eig(X)
    evi = xp.linalg.inv(ev)

    # choose largest abs value eigenpairs
    sel = xp.argsort(xp.abs(el))[-max_bond:]
    Cl = ev[:, sel]
    Cr = evi[sel, :]

    if renorm:
        trace_old = xp.sum(el)
        trace_new = xp.sum(el[sel])
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
    xp = get_namespace(X)
    XX = (X + dag(X)) / 2
    el, ev = xp.linalg.eigh(XX)
    sel = xp.argsort(xp.abs(el))[-max_bond:]
    Cl = ev[:, sel]
    Cr = dag(Cl)
    if renorm:
        trace_old = xp.trace(X)
        trace_new = xp.trace(Cr @ (X @ Cl))
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
    xp = get_namespace(X)
    U, _, VH = xp.linalg.svd(X)
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
        trace_old = xp.trace(X)
        trace_new = xp.trace(Cr @ (X @ Cl))
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
    xp = get_namespace(X)
    U, s, VH = xp.linalg.svd(X)

    B = U[:, :max_bond]
    AH = VH[:max_bond, :]

    Uab, sab, VHab = xp.linalg.svd(AH @ B)
    sab = (sab + 1e-12 * xp.max(sab)) ** -0.5
    sab_inv = xp.reshape(sab, (1, -1))
    P = Uab * sab_inv
    Q = dag(VHab) * sab_inv

    Cl = B @ Q
    Cr = dag(P) @ AH

    if renorm:
        trace_old = xp.trace(X)
        trace_new = xp.trace(Cr @ (X @ Cl))
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
def isometrize_qr(x):
    """Perform isometrization using the QR decomposition."""
    xp = get_namespace(x)
    Q, R = xp.linalg.qr(x)
    # stabilize qr by fixing diagonal of R in canonical, positive form (we
    # don't actaully do anything to R, just absorb the necessary sign -> Q)
    rd = xp.diag(R)
    s = sgn(rd)
    Q = Q * xp.reshape(s, (1, -1))
    return Q


@compose
def isometrize_svd(x):
    """Perform isometrization using the SVD decomposition."""
    xp = get_namespace(x)
    U, _, VH = xp.linalg.svd(x)
    return U @ VH


@compose
def isometrize_exp(x):
    r"""Perform isometrization using anti-symmetric matrix exponentiation.

    .. math::

            U_A = \exp \left( X - X^\dagger \right)

    If ``x`` is rectangular it is completed with zeros first.
    """
    xp = get_namespace(x)
    m, n = x.shape
    d = max(m, n)
    x = xp.pad(x, [[0, d - m], [0, d - n]], "constant", constant_values=0.0)
    x = x - dag(x)
    Q = xp.scipy.linalg.expm(x)
    return Q[:m, :n]


@compose
def isometrize_cayley(x):
    r"""Perform isometrization using an anti-symmetric Cayley transform.

    .. math::

            U_A = (I + \dfrac{A}{2})(I - \dfrac{A}{2})^{-1}

    where :math:`A = X - X^\dagger`. If ``x`` is rectangular it is completed
    with zeros first.
    """
    xp = get_namespace(x)
    m, n = x.shape
    d = max(m, n)
    x = xp.pad(x, [[0, d - m], [0, d - n]], "constant", constant_values=0.0)
    x = x - dag(x)
    x = x / 2.0
    Id = xp.eye(d)
    Q = xp.linalg.solve(Id - x, Id + x)
    return Q[:m, :n]


@compose
def isometrize_modified_gram_schmidt(A):
    """Perform isometrization explicitly using the modified Gram Schmidt
    procedure (this is slow but a useful reference).
    """
    xp = get_namespace(A)
    Q = []
    for j in range(A.shape[1]):
        q = A[:, j]
        for i in range(0, j):
            rij = xp.tensordot(xp.conj(Q[i]), q, 1)
            q = q - rij * Q[i]
        Q.append(q / xp.linalg.norm(q))
    Q = xp.stack(tuple(Q), axis=1)
    return Q


@compose
def isometrize_householder(X):
    xp = get_namespace(X)
    X = xp.tril(X, -1)
    tau = 2.0 / (1.0 + xp.sum(xp.conj(X) * X, 0))
    Q = xp.linalg.householder_product(X, tau)
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
    xp = get_namespace(x)
    m, n = x.shape
    fat = m < n
    if fat:
        x = xp.transpose(x)
    Q = _ISOMETRIZE_METHODS[method](x)
    if fat:
        Q = xp.transpose(Q)
    return Q


@compose
def squared_op_to_reduced_factor(
    x2,
    dl,
    dr,
    right=True,
    method="eigh",
    **kwargs,
):
    """Given the square, ``x2``, of an operator ``x``, compute either the left
    or right reduced factor matrix of the unsquared operator ``x`` with
    original shape ``(dl, dr)``.

    If ``right=True``, compute the right factor, ``s @ Vdag``, assuming ``x2``
    is given as ``x2 = dag(x) @ x = V @ s^2 @ Vdag``, otherwise
    compute the left factor, ``U @ s``, assuming
    ``x2 = x @ dag(x) = U @ s^2 @ Udag``.

    Parameters
    ----------
    x2 : array
        The squared operator, either ``dag(x) @ x`` or ``x @ dag(x)``.
    dl : int
        The original left dimension of the unsquared operator ``x``.
    dr : int
        The original right dimension of the unsquared operator ``x``.
    right : bool, optional
        Whether to compute the right factor (``s @ Vdag``) or left factor
        (``U @ s``).
    method : str, optional
        The method to use for the decomposition.
    kwargs
        Additional keyword arguments to pass to the decomposition method.
        For example ``shift`` for the ``cholesky`` method.

    Returns
    -------
    factor : array
        The reduced factor matrix of the unsquared operator ``x``.
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

    if isfermionic(x2) and x2.indices[1].dual:
        x2 = x2.phase_flip(1)

    # only compute factor we need
    if right:
        absorb = get_sqVH
    else:
        absorb = get_Usq

    if method == "cholesky":
        if keep != -1:
            # XXX: fallback to eigh here?
            warnings.warn(
                "Operator is exactly low rank, but cholesky method "
                "doesn't support truncation, so ignoring `max_bond`."
            )

        lsqrt, _, rsqrt = cholesky_regularized(
            x2,
            absorb=absorb,
            **kwargs,
        )

    else:
        if method == "eigh":
            kwargs.setdefault("positive", 1)

        lsqrt, _, rsqrt = array_split(
            x2,
            max_bond=keep,
            cutoff=0.0,
            absorb=absorb,
            method=method,
            **kwargs,
        )

    if right:
        return rsqrt
    else:
        return lsqrt


def compute_oblique_projectors(
    Rl,
    Rr,
    max_bond=None,
    cutoff=0.0,
    absorb="both",
    cutoff_mode="rsum2",
    method="svd",
    **compress_opts,
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
    max_bond : int, optional
        The maximum bond dimension to compress to.
    cutoff : float, optional
        The singular value cutoff to use.
    absorb : str or None, optional
        Where to absorb the effective singular values into the projectors.
        The options are:

        - ``'both'`` / ``'Usq,sqVH'``: absorb ``sqrt(s)`` into both
          projectors.
        - ``'left'`` / ``'Us,VH'``: absorb ``s`` into the left projector,
          leaving the right projector isometric (LQ-like).
        - ``'right'`` / ``'U,sVH'``: absorb ``s`` into the right projector,
          leaving the left projector isometric (QR-like).
        - ``None`` / ``'U,s,VH'``: return ``s`` as a separate middle element.

    cutoff_mode : str, optional
        The cutoff mode to use when applying the cutoff.
    method : str, optional
        The method to use for the SVD decomposition when computing the
        projectors.
    compress_opts
        Additional keyword arguments to pass to the SVD decomposition method.

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

    Ut, st, VHt = array_split(
        Rl @ Rr,
        max_bond=max_bond,
        cutoff=cutoff,
        absorb=None,
        cutoff_mode=cutoff_mode,
        method=method,
        **compress_opts,
    )

    if absorb is None:
        Pl = Rr @ rddiv(dag(VHt), st)
        Pr = lddiv(st, dag(Ut)) @ Rl
        return Pl, st, Pr

    elif absorb == get_Usq_sqVH:
        st_sqrt = get_namespace(st).sqrt(st)

        # then form the 'oblique' projectors
        Pl = Rr @ rddiv(dag(VHt), st_sqrt)
        Pr = lddiv(st_sqrt, dag(Ut)) @ Rl

    elif absorb == get_Us_VH:
        Pl = Rr @ dag(VHt)
        Pr = lddiv(st, dag(Ut)) @ Rl

    elif absorb == get_U_sVH:
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
    absorb : str or None, optional
        Where to absorb the effective singular values into the projectors:

        - ``'both'`` / ``'Usq,sqVH'``: absorb ``sqrt(s)`` into both
          projectors.
        - ``'left'`` / ``'Us,VH'``: absorb ``s`` into the left projector,
          leaving the right projector isometric (LQ-like).
        - ``'right'`` / ``'U,sVH'``: absorb ``s`` into the right projector,
          leaving the left projector isometric (QR-like).
        - ``None`` / ``'U,s,VH'``: return ``s`` as a separate middle element.
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
        xp = get_namespace(E)
        Ea = xp.fuse(E, (0, 1), (2, 3))
        Ea = (Ea + dag(Ea)) / 2
        el, ev = xp.linalg.eigh(Ea)
        lmax = xp.max(el)
        el = xp.clip(el + lmax * pos_smudge, lmax * pos_smudge, None)
        Ea = xp.multiply_diagonal(ev, el, axis=1) @ dag(ev)
        E = xp.reshape(Ea, E.shape)

    # current bond dim
    d = E.shape[0]
    # environment with bra indices traced out (i.e. half uncompressed)
    Ek = ctg.array_contract((E,), (("kl", "kr", "X", "X"),), ("kl", "kr"))
    # for distance calculation, compute <A|A>, which is constant
    xp = get_namespace(E)
    yAA = xp.abs(ctg.array_contract((Ek,), (("X", "X"),), ()))

    # initial guess for projectors

    if init == "svd":
        Pl, _, Pr = svd_truncated(
            Ek,
            absorb=None,
            max_bond=max_bond,
            cutoff=1e-15,
            cutoff_mode=cutoff_mode_rel,
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
            Pl = xp.random.normal(size=(d, max_bond))
            Pr = xp.linalg.pinv(Pl)

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
            Lr, _, Pr = qr_stabilized(Pr, absorb="left")
            Pl = Pl @ Lr

        elif condition:
            # match projector norms
            nrml = xp.linalg.norm(Pl)
            nrmr = xp.linalg.norm(Pr)
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
            A, b = xp.align_axes(A, b, axes=((0, 1), (0, 1)))
            A, b = xp.align_axes(A, b, axes=((2, 3), (0, 1)))

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
            A, b = xp.align_axes(A, b, axes=((0, 1), (0, 1)))
            A, b = xp.align_axes(A, b, axes=((2, 3), (0, 1)))

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

    if not ((absorb == get_Usq_sqVH) and (cutoff == 0.0)):
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
