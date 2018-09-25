"""Core tensor network tools.
"""
import os
import re
import copy
import uuid
import string
import weakref
import operator
import functools
import itertools
import collections
from numbers import Integral

from cytoolz import (unique, concat, frequencies,
                     partition_all, merge_with, valmap)
import numpy as np
import opt_einsum as oe
import scipy.linalg as scla
import scipy.sparse.linalg as spla
import scipy.linalg.interpolative as sli

from ..core import qarray, prod, njit, realify_scalar, vdot
from ..linalg.base_linalg import norm_fro_dense
from ..linalg.rand_linalg import rsvd, estimate_rank
from ..utils import functions_equal, has_cupy


def _get_contract_expr(contract_str, *shapes, **kwargs):
    # choose how large intermediate arrays can be
    kwargs.setdefault('memory_limit', -1)
    return oe.contract_expression(contract_str, *shapes, **kwargs)


_get_contract_expr_cached = functools.lru_cache(4096)(_get_contract_expr)


def get_contract_expr(contract_str, *shapes, **kwargs):
    """Get an callable expression that will evaluate ``contract_str`` based on
    ``shapes``. Cache the result if no constant tensors are involved.
    """
    # can only cache if the expression does not involve constant tensors
    if kwargs.get('constants', None):
        return _get_contract_expr(contract_str, *shapes, **kwargs)

    return _get_contract_expr_cached(contract_str, *shapes, **kwargs)


_CONTRACT_BACKEND = 'numpy'
_TENSOR_LINOP_BACKEND = 'cupy' if has_cupy() else 'numpy'


def get_contract_backend():
    """Get the default backend used for tensor contractions, via 'opt_einsum'.

    See Also
    --------
    set_contract_backend, get_tensor_linop_backend, set_tensor_linop_backend,
    tensor_contract
    """
    return _CONTRACT_BACKEND


def set_contract_backend(backend):
    """Set the default backend used for tensor contractions, via 'opt_einsum'.

    See Also
    --------
    get_contract_backend, set_tensor_linop_backend, get_tensor_linop_backend,
    tensor_contract
    """
    global _CONTRACT_BACKEND
    _CONTRACT_BACKEND = backend


def get_tensor_linop_backend():
    """Get the default backend used for tensor network linear operators, via
    'opt_einsum'. This is different from the default contraction backend as
    the contractions are likely repeatedly called many times.

    See Also
    --------
    set_tensor_linop_backend, set_contract_backend, get_contract_backend,
    TNLinearOperator
    """
    return _TENSOR_LINOP_BACKEND


def set_tensor_linop_backend(backend):
    """Set the default backend used for tensor network linear operators, via
    'opt_einsum'. This is different from the default contraction backend as
    the contractions are likely repeatedly called many times.

    See Also
    --------
    get_tensor_linop_backend, set_contract_backend, get_contract_backend,
    TNLinearOperator
    """
    global _TENSOR_LINOP_BACKEND
    _TENSOR_LINOP_BACKEND = backend


# --------------------------------------------------------------------------- #
#                                Tensor Funcs                                 #
# --------------------------------------------------------------------------- #

def set_union(sets):
    """Non variadic version of set.union.
    """
    return set.union(*sets)


class OrderedCounter(collections.Counter, collections.OrderedDict):
    pass


def _gen_output_inds(all_inds):
    """Generate the output, i.e. unique, indices from the set ``inds``. Raise
    if any index found more than twice.
    """
    cnts = OrderedCounter(all_inds)
    for ind, freq in cnts.items():
        if freq > 2:
            raise ValueError("The index {} appears more "
                             "than twice!".format(ind))
        elif freq == 1:
            yield ind


def _maybe_map_indices_to_alphabet(a_ix, i_ix, o_ix):
    """``einsum`` need characters a-z,A-Z or equivalent numbers.
    Do this early, and allow *any* index labels.

    Parameters
    ----------
    a_ix : set
        All of the input indices.
    i_ix : sequence of sequence
        The input indices per tensor.
    o_ix : list of int
        The output indices.

    Returns
    -------
    contract_str : str
        The string to feed to einsum/contract.
    """
    amap = {ix: oe.get_symbol(i) for i, ix in enumerate(a_ix)}
    in_str = ("".join(amap[i] for i in ix) for ix in i_ix)
    out_str = "".join(amap[o] for o in o_ix)

    return ",".join(in_str) + "->" + out_str


def tensor_contract(*tensors, output_inds=None, get=None,
                    backend=None, **contract_opts):
    """Efficiently contract multiple tensors, combining their tags.

    Parameters
    ----------
    *tensors : sequence of Tensor
        The tensors to contract.
    output_inds : sequence
        If given, the desired order of output indices, else defaults to the
        order they occur in the input indices.
    get : {None, 'expression'}, optional
        If ``'expression'``, return the expression that performs the
        contraction, for e.g. inspection of the order chosen.
    backend  {'numpy', 'cupy', 'tensorflow', 'theano', ...}, optional
        Which backend to use to perform the contraction. Must be a valid
        ``opt_einsum`` backend with the relevant library installed.

    Returns
    -------
    scalar or Tensor
    """
    if backend is None:
        backend = _CONTRACT_BACKEND

    i_ix = tuple(t.inds for t in tensors)  # input indices per tensor
    a_ix = tuple(concat(i_ix))  # list of all input indices

    if output_inds is None:
        # sort output indices  by input order for efficiency and consistency
        o_ix = tuple(_gen_output_inds(a_ix))
    else:
        o_ix = output_inds

    # possibly map indices into the range needed by opt- einsum
    contract_str = _maybe_map_indices_to_alphabet([*unique(a_ix)], i_ix, o_ix)

    if get == 'path':
        ops = (t.data for t in tensors)
        return oe.contract_path(contract_str, *ops)[1]

    if get == 'expression':
        # account for possible constant tensors
        cnst = contract_opts.get('constants', ())
        ops = (t.data if i in cnst else t.shape for i, t in enumerate(tensors))
        expression = get_contract_expr(contract_str, *ops, **contract_opts)
        return expression

    # perform the contraction
    shapes = (t.shape for t in tensors)
    expression = get_contract_expr(contract_str, *shapes, **contract_opts)
    o_array = expression(*(t.data for t in tensors), backend=backend)

    if not o_ix:
        if isinstance(o_array, np.ndarray):
            o_array = np.asscalar(o_array)
        return realify_scalar(o_array)

    # unison of all tags
    o_tags = set_union(t.tags for t in tensors)

    return Tensor(data=o_array, inds=o_ix, tags=o_tags)


# generate a random base to avoid collisions on difference processes ...
r_bs_str = str(uuid.uuid4())[:6]
# but then make the list orderable to help contraction caching
RAND_UUIDS = map("".join, itertools.product(string.hexdigits, repeat=7))


def rand_uuid(base=""):
    """Return a guaranteed unique, shortish identifier, optional appended
    to ``base``.

    Examples
    --------
    >>> rand_uuid()
    '_2e1dae1b'

    >>> rand_uuid('virt-bond')
    'virt-bond_bf342e68'
    """
    return base + "_" + r_bs_str + next(RAND_UUIDS)


@njit(['i4(f4[:], f4, i4)', 'i4(f8[:], f8, i4)'])  # pragma: no cover
def _trim_singular_vals(s, cutoff, cutoff_mode):
    """Find the number of singular values to keep of ``s`` given ``cutoff`` and
    ``cutoff_mode``.

    Parameters
    ----------
    s : array
        Singular values.
    cutoff : float
        Cutoff.
    cutoff_mode : {1, 2, 3}
        How to perfrm the trim:

            - 1: ['abs'], trim values below ``cutoff``
            - 2: ['rel'], trim values below ``s[0] * cutoff``
            - 3: ['sum2'], trim s.t. ``sum(s_trim**2) < cutoff``.
            - 4: ['rsum2'], trim s.t. ``sum(s_trim**2) < sum(s**2) * cutoff``.
    """
    if cutoff_mode == 1:
        n_chi = np.sum(s > cutoff)

    elif cutoff_mode == 2:
        n_chi = np.sum(s > cutoff * s[0])

    elif (cutoff_mode == 3) or (cutoff_mode == 4):

        target = cutoff
        if cutoff_mode == 4:
            target *= np.sum(s**2)

        n_chi = s.size
        s2s = 0.0
        for i in range(s.size - 1, -1, -1):
            s2 = s[i]**2
            if not np.isnan(s2):
                s2s += s2
            if s2s > target:
                break
            n_chi -= 1
    else:
        raise ValueError("``cutoff_mode`` not one of {1, 2, 3, 4}.")

    return max(n_chi, 1)


@njit(['f4(f4[:], i4)', 'f8(f8[:], i4)'])  # pragma: no cover
def _renorm_singular_vals(s, n_chi):
    """Find the normalization constant for ``s`` such that the new sum squared
    of the ``n_chi`` largest values equals the sum squared of all the old ones.
    """
    s_tot_keep = 0.0
    s_tot_lose = 0.0
    for i in range(s.size):
        s2 = s[i]**2
        if not np.isnan(s2):
            if i < n_chi:
                s_tot_keep += s2
            else:
                s_tot_lose += s2
    return ((s_tot_keep + s_tot_lose) / s_tot_keep)**0.5


@njit  # pragma: no cover
def _trim_and_renorm_SVD(U, s, V, cutoff, cutoff_mode, max_bond, absorb):
    if cutoff > 0.0:
        n_chi = _trim_singular_vals(s, cutoff, cutoff_mode)

        if max_bond > 0:
            n_chi = min(n_chi, max_bond)

        if n_chi < s.size:
            norm = _renorm_singular_vals(s, n_chi)
            s = s[:n_chi] * norm
            U = U[..., :n_chi]
            V = V[:n_chi, ...]

    s = np.ascontiguousarray(s)

    if absorb == -1:
        U *= s.reshape((1, -1))
    elif absorb == 1:
        V *= s.reshape((-1, 1))
    else:
        s **= 0.5
        U *= s.reshape((1, -1))
        V *= s.reshape((-1, 1))

    return U, V


@njit  # pragma: no cover
def _array_split_svd_nb(x, cutoff=-1.0, cutoff_mode=3, max_bond=-1, absorb=0):
    """SVD-decomposition.
    """
    U, s, V = np.linalg.svd(x, full_matrices=False)
    return _trim_and_renorm_SVD(U, s, V, cutoff, cutoff_mode, max_bond, absorb)


def _array_split_svd_alt(x, cutoff=-1.0, cutoff_mode=3, max_bond=-1, absorb=0):
    """SVD-decompt using alternate scipy driver.
    """
    U, s, V = scla.svd(x, full_matrices=False, lapack_driver='gesvd')
    return _trim_and_renorm_SVD(U, s, V, cutoff, cutoff_mode, max_bond, absorb)


def _array_split_svd(x, cutoff=-1.0, cutoff_mode=3, max_bond=-1, absorb=0):
    args = (x, cutoff, cutoff_mode, max_bond, absorb)

    try:
        return _array_split_svd_nb(*args)

    except (scla.LinAlgError, ValueError) as e:  # pragma: no cover

        if isinstance(e, scla.LinAlgError) or 'converge' in str(e):
            import warnings
            warnings.warn("TN SVD failed, trying again with alternate driver.")

            return _array_split_svd_alt(*args)

        raise e


def _array_split_svdvals(x):
    """SVD-decomposition, but return singular values only.
    """
    return np.linalg.svd(x, full_matrices=False, compute_uv=False)


@njit  # pragma: no cover
def dag(x):
    """Hermitian conjugate.
    """
    return np.conjugate(x).T


@njit  # pragma: no cover
def _array_split_eig(x, cutoff=-1.0, cutoff_mode=3, max_bond=-1, absorb=0):
    """SVD-split via eigen-decomposition.
    """
    if x.shape[0] > x.shape[1]:
        # Get sU, V
        s2, V = np.linalg.eigh(dag(x) @ x)
        U = x @ V
        V = dag(V)

        # Check if want U, sV
        if absorb == 1:
            s = s2**0.5
            U /= s.reshape((1, -1))
            V *= s.reshape((-1, 1))
        # Or sqrt(s)U, sqrt(s)V
        elif absorb == 0:
            sqrts = s2**0.25
            U /= sqrts.reshape((1, -1))
            V *= sqrts.reshape((-1, 1))

    else:
        # Get U, sV
        s2, U = np.linalg.eigh(x @ dag(x))
        V = dag(U) @ x

        # Check if want sU, V
        if absorb == -1:
            s = s2**0.5
            U *= s.reshape((1, -1))
            V /= s.reshape((-1, 1))

        # Or sqrt(s)U, sqrt(s)V
        elif absorb == 0:
            sqrts = s2**0.25
            U *= sqrts.reshape((1, -1))
            V /= sqrts.reshape((-1, 1))

    # eigh produces ascending eigenvalue order -> slice opposite to svd
    if cutoff > 0.0:
        s = s2[::-1]**0.5
        n_chi = _trim_singular_vals(s, cutoff, cutoff_mode)

        if max_bond > 0:
            n_chi = min(n_chi, max_bond)

        if n_chi < s.size:
            norm = _renorm_singular_vals(s, n_chi)
            U = U[..., -n_chi:]
            V = V[-n_chi:, ...]

            if absorb == -1:
                U *= norm
            elif absorb == 0:
                U *= norm**0.5
                V *= norm**0.5
            else:
                V *= norm

    return U, V


@njit
def _array_split_svdvals_eig(x):  # pragma: no cover
    """SVD-decomposition via eigen, but return singular values only.
    """
    if x.shape[0] > x.shape[1]:
        s2 = np.linalg.eigvalsh(dag(x) @ x)
    else:
        s2 = np.linalg.eigvalsh(x @ dag(x))
    return s2**0.5


@njit  # pragma: no cover
def _array_split_eigh(x, cutoff=-1.0, cutoff_mode=3, max_bond=-1, absorb=0):
    """SVD-decomposition, using hermitian eigen-decomposition, only works if
    ``x`` is hermitian.
    """
    s, U = np.linalg.eigh(x)
    s, U = s[::-1], U[:, ::-1]  # make sure largest singular value first

    V = np.sign(s).reshape(-1, 1) * dag(U)
    s = np.abs(s)
    return _trim_and_renorm_SVD(U, s, V, cutoff, cutoff_mode, max_bond, absorb)


@njit  # pragma: no cover
def _numba_cholesky(x, cutoff=-1, cutoff_mode=3, max_bond=-1, absorb=0):
    """SVD-decomposition, using cholesky decomposition, only works if
    ``x`` is positive definite.
    """
    L = np.linalg.cholesky(x)
    return L, dag(L)


def _array_split_cholesky(x, cutoff=-1, cutoff_mode=3, max_bond=-1, absorb=0):
    try:
        return _numba_cholesky(x, cutoff, cutoff_mode, max_bond, absorb)
    except np.linalg.LinAlgError as e:
        if cutoff < 0:
            raise e
        # try adding cutoff identity - assuming it is approx allowable error
        xi = x + 2 * cutoff * np.eye(x.shape[0])
        return _numba_cholesky(xi, cutoff, cutoff_mode, max_bond, absorb)


def _choose_k(x, cutoff, max_bond):
    """Choose the number of singular values to target.
    """
    d = min(x.shape)

    if cutoff != 0.0:
        k = estimate_rank(x, cutoff, k_max=None if max_bond < 0 else max_bond)
    else:
        k = min(d, max_bond)

    # if computing more than half of spectrum then just use dense method
    return 'full' if k > d // 2 else k


def _array_split_svds(x, cutoff=0.0, cutoff_mode=2, max_bond=-1, absorb=0):
    """SVD-decomposition using iterative methods. Allows the
    computation of only a certain number of singular values, e.g. max_bond,
    from the get-go, and is thus more efficient. Can also supply
    ``scipy.sparse.linalg.LinearOperator``.
    """
    k = _choose_k(x, cutoff, max_bond)

    if k == 'full':
        if not isinstance(x, np.ndarray):
            x = x.to_dense()
        return _array_split_svd(x, cutoff, cutoff_mode, max_bond, absorb)

    U, s, V = spla.svds(x, k=k)
    return _trim_and_renorm_SVD(U, s, V, cutoff, cutoff_mode, max_bond, absorb)


def _array_split_isvd(x, cutoff=0.0, cutoff_mode=2, max_bond=-1, absorb=0):
    """SVD-decomposition using interpolative matrix random methods. Allows the
    computation of only a certain number of singular values, e.g. max_bond,
    from the get-go, and is thus more efficient. Can also supply
    ``scipy.sparse.linalg.LinearOperator``.
    """
    k = _choose_k(x, cutoff, max_bond)

    if k == 'full':
        if not isinstance(x, np.ndarray):
            x = x.to_dense()
        return _array_split_svd(x, cutoff, cutoff_mode, max_bond, absorb)

    U, s, V = sli.svd(x, k)
    V = V.conj().T
    return _trim_and_renorm_SVD(U, s, V, cutoff, cutoff_mode, max_bond, absorb)


def _array_split_rsvd(x, cutoff=0.0, cutoff_mode=2, max_bond=-1, absorb=0):
    """SVD-decomposition using randomized methods (due to Halko). Allows the
    computation of only a certain number of singular values, e.g. max_bond,
    from the get-go, and is thus more efficient. Can also supply
    ``scipy.sparse.linalg.LinearOperator``.
    """
    U, s, V = rsvd(x, cutoff)
    return _trim_and_renorm_SVD(U, s, V, cutoff, cutoff_mode, max_bond, absorb)


def _array_split_eigsh(x, cutoff=0.0, cutoff_mode=2, max_bond=-1, absorb=0):
    """SVD-decomposition using iterative hermitian eigen decomp, thus assuming
    that ``x`` is hermitian. Allows the computation of only a certain number of
    singular values, e.g. max_bond, from the get-go, and is thus more
    efficient. Can also supply ``scipy.sparse.linalg.LinearOperator``.
    """
    k = _choose_k(x, cutoff, max_bond)

    if k == 'full':
        if not isinstance(x, np.ndarray):
            x = x.to_dense()
        return _array_split_eigh(x, cutoff, cutoff_mode, max_bond, absorb)

    s, U = spla.eigsh(x, k=k)
    s, U = s[::-1], U[:, ::-1]  # make sure largest singular value first
    V = np.sign(s).reshape(-1, 1) * dag(U)
    s = np.abs(s)
    return _trim_and_renorm_SVD(U, s, V, cutoff, cutoff_mode, max_bond, absorb)


@njit  # pragma: no cover
def _array_split_qr(x):
    """QR-decomposition.
    """
    Q, R = np.linalg.qr(x)
    return Q, R


@njit  # pragma: no cover
def _array_split_lq(x):
    """LQ-decomposition.
    """
    Q, L = np.linalg.qr(x.T)
    return L.T, Q.T


def tensor_split(T, left_inds, method='svd', max_bond=None, absorb='both',
                 cutoff=1e-10, cutoff_mode='sum2', get=None, bond_ind=None,
                 ltags=None, rtags=None, right_inds=None):
    """Decompose this tensor into two tensors.

    Parameters
    ----------
    T : Tensor
        The tensor to split.
    left_inds : str or sequence of str
        The index or sequence of inds, which ``tensor`` should already have, to
        split to the 'left'.
    method : str, optional
        How to split the tensor, only some methods allow bond truncation:

            - 'svd': full SVD, allows truncation.
            - 'eig': full SVD via eigendecomp, allows truncation.
            - 'svds': iterative svd, allows truncation.
            - 'isvd': iterative svd using interpolative methods, allows
              truncation.
            - 'qr': full QR decomposition.
            - 'lq': full LR decomposition.
            - 'eigh': full eigen-decomposition, tensor must he hermitian.
            - 'eigsh': iterative eigen-decomposition, tensor must he hermitian.
            - 'cholesky': full cholesky decomposition, tensor must be positive.

    max_bond: None or int
        If integer, the maxmimum number of singular values to keep, regardless
        of ``cutoff``.
    absorb = {'both', 'left', 'right'}
        Whether to absorb the singular values into both, the left or right
        unitary matrix respectively.
    cutoff : float, optional
        The threshold below which to discard singular values, only applies to
        SVD and eigendecomposition based methods (not QR, LQ, or cholesky).
    cutoff_mode : {'sum2', 'rel', 'abs', 'rsum2'}
        Method with which to apply the cutoff threshold:

            - 'rel': values less than ``cutoff * s[0]`` discarded.
            - 'abs': values less than ``cutoff`` discarded.
            - 'sum2': sum squared of values discarded must be ``< cutoff``.
            - 'rsum2': sum squared of values discarded must be less than
              ``cutoff`` times the total sum squared values.

    get : {None, 'arrays', 'tensors', 'values'}
        If given, what to return instead of a TN describing the split. The
        default, ``None``, returns a TensorNetwork of the two tensors.
    bond_ind : str, optional
        Explicitly name the new bond, else a random one will be generated.
    ltags : sequence of str, optional
        Add these new tags to the left tensor.
    rtags : sequence of str, optional
        Add these new tags to the right tensor.
    right_inds : sequence of str, optional
        Explicitly give the right indices, otherwise they will be worked out.
        This is a minor performance feature.

    Returns
    -------
    TensorNetwork or (Tensor, Tensor) or (array, array) or 1D-array
        Respectively if get={None, 'tensors', 'arrays', 'values'}.
    """
    if isinstance(left_inds, str):
        left_inds = (left_inds,)
    else:
        left_inds = tuple(left_inds)

    if right_inds is None:
        right_inds = tuple(x for x in T.inds if x not in left_inds)

    TT = T.transpose(*left_inds, *right_inds)

    left_dims = TT.shape[:len(left_inds)]
    right_dims = TT.shape[len(left_inds):]

    array = TT.data.reshape(prod(left_dims), prod(right_dims))

    if get == 'values':
        return {'svd': _array_split_svdvals,
                'eig': _array_split_svdvals_eig}[method](array)

    opts = {}
    if method not in ('qr', 'lq'):
        # Convert defaults and settings to numeric type for numba funcs
        opts['cutoff'] = {None: -1.0}.get(cutoff, cutoff)
        opts['absorb'] = {'left': -1, 'both': 0, 'right': 1}[absorb]
        opts['max_bond'] = {None: -1}.get(max_bond, max_bond)
        opts['cutoff_mode'] = {'abs': 1, 'rel': 2,
                               'sum2': 3, 'rsum2': 4}[cutoff_mode]

    left, right = {
        'svd': _array_split_svd,
        'eig': _array_split_eig,
        'qr': _array_split_qr,
        'lq': _array_split_lq,
        'eigh': _array_split_eigh,
        'cholesky': _array_split_cholesky,
        'isvd': _array_split_isvd,
        'svds': _array_split_svds,
        'rsvd': _array_split_rsvd,
        'eigsh': _array_split_eigsh,
    }[method](array, **opts)

    left = left.reshape(*left_dims, -1)
    right = right.reshape(-1, *right_dims)

    if get == 'arrays':
        return left, right

    bond_ind = rand_uuid() if bond_ind is None else bond_ind

    ltags, rtags = tags2set(ltags) | T.tags, tags2set(rtags) | T.tags

    Tl = Tensor(data=left, inds=(*left_inds, bond_ind), tags=ltags)
    Tr = Tensor(data=right, inds=(bond_ind, *right_inds), tags=rtags)

    if get == 'tensors':
        return Tl, Tr

    return TensorNetwork((Tl, Tr), check_collisions=False)


def tensor_compress_bond(T1, T2, **compress_opts):
    r"""Inplace compress between the two single tensors. It follows the
    following steps to minimize the size of SVD performed::

        a)|   |        b)|            |        c)|       |
        --1---2--  ->  --1L~~1R--2L~~2R--  ->  --1L~~M~~2R--
          |   |          |   ......   |          |       |
         <*> <*>              >  <                  <*>

                  d)|            |        e)|     |
              ->  --1L~~ML~~MR~~2R--  ->  --1C~~~2C--
                    |....    ....|          |     |
                     >  <    >  <              ^compressed bond
    """
    s_ix, t1_ix = T1.filter_bonds(T2)

    if not s_ix:
        raise ValueError("The tensors specified don't share an bond.")
    # a) -> b)
    T1_L, T1_R = T1.split(left_inds=t1_ix, get='tensors',
                          absorb='right', **compress_opts)
    T2_L, T2_R = T2.split(left_inds=s_ix, get='tensors',
                          absorb='left', **compress_opts)
    # b) -> c)
    M = (T1_R @ T2_L)
    M.drop_tags()
    # c) -> d)
    M_L, M_R = M.split(left_inds=T1_L.bonds(M), get='tensors',
                       absorb='both', **compress_opts)

    # make sure old bond being used
    ns_ix, = M_L.bonds(M_R)
    M_L.reindex_({ns_ix: s_ix[0]})
    M_R.reindex_({ns_ix: s_ix[0]})

    # d) -> e)
    T1C = T1_L.contract(M_L, output_inds=T1.inds)
    T2C = M_R.contract(T2_R, output_inds=T2.inds)

    # update with the new compressed data
    T1.modify(data=T1C.data)
    T2.modify(data=T2C.data)


def tensor_add_bond(T1, T2):
    """Inplace addition of a dummy bond between ``T1`` and ``T2``.
    """
    bnd = rand_uuid()
    T1.modify(data=T1.data[..., np.newaxis], inds=(*T1.inds, bnd))
    T2.modify(data=T2.data[..., np.newaxis], inds=(*T2.inds, bnd))


def array_direct_product(X, Y, sum_axes=()):
    """Direct product of two numpy.ndarrays.

    Parameters
    ----------
    X : numpy.ndarray
        First tensor.
    Y : numpy.ndarray
        Second tensor, same shape as ``X``.
    sum_axes : sequence of int
        Axes to sum over rather than direct product, e.g. physical indices when
        adding tensor networks.

    Returns
    -------
    Z : numpy.ndarray
        Same shape as ``X`` and ``Y``, but with every dimension the sum of the
        two respective dimensions, unless it is included in ``sum_axes``.
    """

    if isinstance(sum_axes, Integral):
        sum_axes = (sum_axes,)

    # parse the intermediate and final shape doubling the size of any axes that
    #   is not to be summed, and preparing slices with which to add X, Y.
    final_shape = []
    selectorX = []
    selectorY = []

    for i, (d1, d2) in enumerate(zip(X.shape, Y.shape)):
        if i not in sum_axes:
            final_shape.append(d1 + d2)
            selectorX.append(slice(0, d1))
            selectorY.append(slice(d1, None))
        else:
            if d1 != d2:
                raise ValueError("Can only add sum tensor indices of the same "
                                 "size.")
            final_shape.append(d1)
            selectorX.append(slice(None))
            selectorY.append(slice(None))

    new_type = np.find_common_type((X.dtype, Y.dtype), ())
    Z = np.zeros(final_shape, dtype=new_type)

    # Add tensors to the diagonals
    Z[tuple(selectorX)] += X
    Z[tuple(selectorY)] += Y

    return Z


def tensor_direct_product(T1, T2, sum_inds=(), inplace=False):
    """Direct product of two Tensors. Any axes included in ``sum_inds`` must be
    the same size and will be summed over rather than concatenated. Summing
    over contractions of TensorNetworks equates to contracting a TensorNetwork
    made of direct products of each set of tensors. I.e. (a1 @ b1) + (a2 @ b2)
    == (a1 (+) a2) @ (b1 (+) b2).

    Parameters
    ----------
    T1 : Tensor
        The first tensor.
    T2 : Tensor
        The second tensor, with matching indices and dimensions to ``T1``.
    sum_inds : sequence of str, optional
        Axes to sum over rather than combine, e.g. physical indices when
        adding tensor networks.
    inplace : bool, optional
        Whether to modify ``T1`` inplace.

    Returns
    -------
    Tensor
        Like ``T1``, but with each dimension doubled in size if not
        in ``sum_inds``.
    """
    if isinstance(sum_inds, (str, Integral)):
        sum_inds = (sum_inds,)

    if T2.inds != T1.inds:
        T2 = T2.transpose(*T1.inds)

    sum_axes = tuple(T1.inds.index(ind) for ind in sum_inds)

    if inplace:
        new_T = T1
    else:
        new_T = T1.copy()

    # XXX: add T2s tags?
    new_T.modify(data=array_direct_product(T1.data, T2.data,
                                           sum_axes=sum_axes))
    return new_T


def bonds(t1, t2):
    """Getting any indices connecting the Tensor(s) or TensorNetwork(s) ``t1``
    and ``t2``.
    """
    if isinstance(t1, Tensor):
        ix1 = set(t1.inds)
    else:
        ix1 = set(concat(t.inds for t in t1))

    if isinstance(t2, Tensor):
        ix2 = set(t2.inds)
    else:
        ix2 = set(concat(t.inds for t in t2))

    return ix1 & ix2


def bonds_size(t1, t2):
    """Get the size of the bonds linking tensors or tensor networks ``t1`` and
    ``t2``.
    """
    return prod(t1.ind_size(ix) for ix in bonds(t1, t2))


def get_tags(ts):
    """Return all the tags in found in ``ts``.

    Parameters
    ----------
    ts :  Tensor, TensorNetwork or sequence of either
        The objects to combine tags from.
    """
    if isinstance(ts, (TensorNetwork, Tensor)):
        ts = (ts,)

    return set().union(*[t.tags for t in ts])


def tags2set(tags):
    """Parse a ``tags`` argument into a set - leave if already one.
    """
    if isinstance(tags, set):
        return tags
    elif tags is None:
        return set()
    elif isinstance(tags, str):
        return {tags}
    else:
        return set(tags)


# --------------------------------------------------------------------------- #
#                                Tensor Class                                 #
# --------------------------------------------------------------------------- #

def _asarray(array):
    if isinstance(array, np.matrix) or not hasattr(array, 'shape'):
        return np.asarray(array)
    return array


def _ndim(array):
    try:
        return array.ndim
    except AttributeError:
        return len(array.shape)


class Tensor(object):
    """A labelled, tagged ndarray.

    Parameters
    ----------
    data : numpy.ndarray
        The n-dimensional data.
    inds : sequence of str
        The index labels for each dimension. Must match the number of
        dimensions of ``data``.
    tags : sequence of str
        Tags with which to identify and group this tensor. These will
        be converted into a ``set``.
    """

    def __init__(self, data, inds, tags=None):
        # a new or copied Tensor always has no owners
        self.owners = {}

        # Short circuit for copying Tensors
        if isinstance(data, Tensor):
            self._data = data.data
            self._inds = data.inds
            self._tags = data.tags.copy()
            return

        self._data = _asarray(data)
        self._inds = tuple(inds)

        if _ndim(self._data) != len(self.inds):
            raise ValueError(
                "Wrong number of inds, {}, supplied for array"
                " of shape {}.".format(self.inds, self._data.shape))

        self._tags = tags2set(tags)

    def copy(self, deep=False):
        """
        """
        if deep:
            return copy.deepcopy(self)
        else:
            return Tensor(self, None)

    __copy__ = copy

    @property
    def data(self):
        return self._data

    @property
    def inds(self):
        return self._inds

    @property
    def tags(self):
        return self._tags

    def add_owner(self, tn, tid):
        """Add ``tn`` as owner of this Tensor - it's tag and ind maps will
        be updated whenever this tensor is retagged or reindexed.
        """
        self.owners[hash(tn)] = (weakref.ref(tn), tid)

    def remove_owner(self, tn):
        """Remove ``tn`` as owner of this Tensor.
        """
        try:
            del self.owners[hash(tn)]
        except KeyError:
            pass

    def check_owners(self):
        """Check if this tensor is 'owned' by any alive TensorNetworks. Also
        trim any weakrefs to dead TensorNetwork.
        """
        # first parse out dead owners
        for k in tuple(self.owners):
            if not self.owners[k][0]():
                del self.owners[k]

        return len(self.owners) > 0

    def modify(self, data=None, inds=None, tags=None):
        """Overwrite the data of this tensor in place.

        Parameters
        ----------
        data : array, optional
            New data.
        inds : sequence of str, optional
            New tuple of indices.
        tags : sequence of str, optional
            New tags.
        """
        if data is not None:
            self._data = _asarray(data)

        if inds is not None:
            # if this tensor has owners, update their ``ind_map``.
            if self.check_owners():
                for ref, tid in self.owners.values():
                    ref()._modify_tensor_inds(self.inds, inds, tid)

            self._inds = tuple(inds)

        if tags is not None:
            # if this tensor has owners, update their ``tag_map``.
            if self.check_owners():
                for ref, tid in self.owners.values():
                    ref()._modify_tensor_tags(self.tags, tags, tid)

            self._tags = tags2set(tags)

        if len(self.inds) != _ndim(self.data):
            raise ValueError("Mismatch between number of data dimensions and "
                             "number of indices supplied.")

    def add_tag(self, tag):
        """Add a tag to this tensor. Unlike ``self.tags.add`` this also updates
        any TensorNetworks viewing this Tensor.
        """
        self.modify(tags=set.union(self.tags, {tag}))

    def conj(self, inplace=False):
        """Conjugate this tensors data (does nothing to indices).
        """
        if inplace:
            self._data = self._data.conj()
            return self
        else:
            return Tensor(self._data.conj(), self.inds, self.tags)

    conj_ = functools.partialmethod(conj, inplace=True)

    @property
    def H(self):
        """Conjugate this tensors data (does nothing to indices).
        """
        return self.conj()

    @property
    def shape(self):
        return self._data.shape

    @property
    def ndim(self):
        return _ndim(self._data)

    @property
    def size(self):
        return self._data.size

    @property
    def dtype(self):
        return self._data.dtype

    def isreal(self):
        return np.issubdtype(self.dtype, np.floating)

    def iscomplex(self):
        return np.issubdtype(self.dtype, np.complexfloating)

    def astype(self, dtype, inplace=False):
        """Change the type of this tensor to ``dtype``.
        """
        T = self if inplace else self.copy()
        T.modify(data=self.data.astype(dtype))
        return T

    def ind_size(self, ind):
        """Return the size of dimension corresponding to ``ind``.
        """
        return self.shape[self.inds.index(ind)]

    def shared_bond_size(self, other):
        """Get the total size of the shared index(es) with ``other``.
        """
        return bonds_size(self, other)

    def inner_inds(self):
        """
        """
        ind_freqs = frequencies(self.inds)
        return tuple(i for i in self.inds if ind_freqs[i] == 2)

    def transpose(self, *output_inds, inplace=False):
        """Transpose this tensor.

        Parameters
        ----------
        output_inds : sequence of str
            The desired output sequence of indices.
        inplace : bool, optional
            Perform the tranposition inplace.

        Returns
        -------
        tt : Tensor
            The transposed tensor.

        See Also
        --------
        transpose_like
        """
        tn = self if inplace else self.copy()

        output_inds = tuple(output_inds)  # need to re-use this.

        if set(tn.inds) != set(output_inds):
            raise ValueError("'output_inds' must be permutation of the "
                             "current tensor indices, but {} != {}"
                             .format(set(tn.inds), set(output_inds)))

        current_ind_map = {ind: i for i, ind in enumerate(tn.inds)}
        out_shape = tuple(current_ind_map[i] for i in output_inds)

        tn.modify(data=tn.data.transpose(*out_shape), inds=output_inds)
        return tn

    transpose_ = functools.partialmethod(transpose, inplace=True)

    def transpose_like(self, other, inplace=False):
        """Transpose this tensor to match the indices of ``other``, allowing
        for one index to be different. E.g. if
        ``self.inds = ('a', 'b', 'c', 'x')`` and
        ``other.inds = ('b', 'a', 'd', 'c')`` then 'x' will be aligned with 'd'
        and the output inds will be ``('b', 'a', 'x', 'c')``

        Parameters
        ----------
        other : Tensor
            The tensor to match.
        inplace : bool, optional
            Perform the tranposition inplace.

        Returns
        -------
        tt : Tensor
            The transposed tensor.

        See Also
        --------
        transpose
        """
        tn = self if inplace else self.copy()
        diff_ix = set(tn.inds) - set(other.inds)

        if len(diff_ix) > 1:
            raise ValueError("More than one index don't match, the transpose "
                             "is therefore not well-defined.")

        # if their indices match, just plain transpose
        if not diff_ix:
            tn.transpose_(*other.inds)

        else:
            di, = diff_ix
            new_ix = (i if i in tn.inds else di for i in other.inds)
            tn.transpose_(*new_ix)

        return tn

    transpose_like_ = functools.partialmethod(transpose_like, inplace=True)

    @functools.wraps(tensor_contract)
    def contract(self, *others, output_inds=None, **opts):
        return tensor_contract(self, *others, output_inds=output_inds, **opts)

    @functools.wraps(tensor_direct_product)
    def direct_product(self, other, sum_inds=(), inplace=False):
        return tensor_direct_product(
            self, other, sum_inds=sum_inds, inplace=inplace)

    @functools.wraps(tensor_split)
    def split(self, *args, **kwargs):
        return tensor_split(self, *args, **kwargs)

    def singular_values(self, left_inds, method='svd'):
        """Return the singular values associated with splitting this tensor
        according to ``left_inds``.

        Parameters
        ----------
        left_inds : sequence of str
            A subset of this tensors indices that defines 'left'.
        method : {'svd', 'eig'}
            Whether to use the SVD or eigenvalue decomposition to get the
            singular values.

        Returns
        -------
        1d-array
            The singular values.
        """
        return self.split(left_inds=left_inds, method=method, get='values')

    def entropy(self, left_inds, method='svd'):
        """Return the entropy associated with splitting this tensor
        according to ``left_inds``.

        Parameters
        ----------
        left_inds : sequence of str
            A subset of this tensors indices that defines 'left'.
        method : {'svd', 'eig'}
            Whether to use the SVD or eigenvalue decomposition to get the
            singular values.

        Returns
        -------
        float
        """
        el = self.singular_values(left_inds=left_inds, method=method)**2
        el = el[el > 0.0]
        return np.sum(-el * np.log2(el))

    def retag(self, retag_map, inplace=False):
        """Rename the tags of this tensor, optionally, in-place.

        Parameters
        ----------
        retag_map : dict-like
            Mapping of pairs ``{old_tag: new_tag, ...}``.
        inplace : bool, optional
            If ``False`` (the default), a copy of this tensor with the changed
            tags will be returned.
        """
        new = self if inplace else self.copy()
        new.modify(tags={retag_map.get(tag, tag) for tag in new.tags})
        return new

    retag_ = functools.partialmethod(retag, inplace=True)

    def reindex(self, index_map, inplace=False):
        """Rename the indices of this tensor, optionally in-place.

        Parameters
        ----------
        index_map : dict-like
            Mapping of pairs ``{old_ind: new_ind, ...}``.
        inplace : bool, optional
            If ``False`` (the default), a copy of this tensor with the changed
            inds will be returned.
        """
        new = self if inplace else self.copy()
        new.modify(inds=tuple(index_map.get(ind, ind) for ind in new.inds))
        return new

    reindex_ = functools.partialmethod(reindex, inplace=True)

    def fuse(self, fuse_map, inplace=False):
        """Combine groups of indices into single indices.

        Parameters
        ----------
        fuse_map : dict_like or sequence of tuples.
            Mapping like: ``{new_ind: sequence of existing inds, ...}`` or an
            ordered mapping like ``[(new_ind_1, old_inds_1), ...]`` in which
            case the output tensor's fused inds will be ordered. In both cases
            the new indices are created at the beginning of the tensor's shape.

        Returns
        -------
        Tensor
            The transposed, reshaped and re-labeled tensor.
        """
        tn = self if inplace else self.copy()

        if isinstance(fuse_map, dict):
            new_fused_inds, fused_inds = zip(*fuse_map.items())
        else:
            new_fused_inds, fused_inds = zip(*fuse_map)

        unfused_inds = tuple(i for i in tn.inds if not
                             any(i in fs for fs in fused_inds))

        # transpose tensor to bring groups of fused inds to the beginning
        tn.transpose_(*concat(fused_inds), *unfused_inds)

        # for each set of fused dims, group into product, then add remaining
        dims = iter(tn.shape)
        dims = [prod(next(dims) for _ in fs) for fs in fused_inds] + list(dims)

        # create new tensor with new + remaining indices
        tn.modify(data=tn.data.reshape(*dims),
                  inds=(*new_fused_inds, *unfused_inds))
        return tn

    fuse_ = functools.partialmethod(fuse, inplace=True)

    def to_dense(self, *inds_seq):
        """Convert this Tensor into an dense array, with a single dimension
        for each of inds in ``inds_seqs``. E.g. to convert several sites
        into a density matrix: ``T.to_dense(('k0', 'k1'), ('b0', 'b1'))``.
        """
        fused = self.fuse([(str(i), ix) for i, ix in enumerate(inds_seq)])
        return qarray(fused.data)

    def squeeze(self, inplace=False):
        """Drop any singlet dimensions from this tensor.
        """
        t = self if inplace else self.copy()
        new_shape, new_inds = zip(
            *((d, i) for d, i in zip(self.shape, self.inds) if d > 1))
        if len(t.inds) != len(new_inds):
            t.modify(data=t.data.reshape(new_shape), inds=new_inds)
        return t

    squeeze_ = functools.partialmethod(squeeze, inplace=True)

    def norm(self):
        """Frobenius norm of this tensor.
        """
        return norm_fro_dense(self.data.reshape(-1))

    def symmetrize(self, ind1, ind2, inplace=False):
        """Hermitian symmetrize this tensor for indices ``ind1`` and ``ind2``.
        I.e. ``T = (T + T.conj().T) / 2``, where the transpose is taken only
        over the specified indices.
        """
        T = self if inplace else self.copy()
        Hinds = [{ind1: ind2, ind2: ind1}.get(i, i) for i in self.inds]
        TH = T.conj().transpose(*Hinds)
        T.modify(data=(T.data + TH.data) / 2)
        return T

    def almost_equals(self, other, **kwargs):
        """Check if this tensor is almost the same as another.
        """
        same_inds = (set(self.inds) == set(other.inds))
        if not same_inds:
            return False
        otherT = other.transpose(*self.inds)
        return np.allclose(self.data, otherT.data, **kwargs)

    def drop_tags(self, tags=None):
        """Drop certain tags, defaulting to all, from this tensor.
        """
        if tags is None:
            tags = self.tags
        else:
            tags = tags2set(tags)

        self.modify(tags=self.tags - tags)

    def bonds(self, other):
        """Return a tuple of the shared indices between this tensor
        and ``other``.
        """
        return bonds(self, other)

    def filter_bonds(self, other):
        """Sort this tensor's indices into a list of those that it shares and
        doesn't share with another tensor.

        Parameters
        ----------
        other : Tensor
            The other tensor.

        Returns
        -------
        shared, unshared : (tuple[str], tuple[str])
            The shared and unshared indices.
        """
        shared = []
        unshared = []
        for i in self.inds:
            if i in other.inds:
                shared.append(i)
            else:
                unshared.append(i)
        return shared, unshared

    def __and__(self, other):
        """Combine with another ``Tensor`` or ``TensorNetwork`` into a new
        ``TensorNetwork``.
        """
        return TensorNetwork((self, other))

    def __or__(self, other):
        """Combine virtually (no copies made) with another ``Tensor`` or
        ``TensorNetwork`` into a new ``TensorNetwork``.
        """
        return TensorNetwork((self, other), virtual=True)

    def __matmul__(self, other):
        """Explicitly contract with another tensor.
        """
        return tensor_contract(self, other)

    def graph(self, *args, **kwargs):
        """Plot a graph of this tensor and its indices.
        """
        TensorNetwork((self,)).graph(*args, **kwargs)

    def __getstate__(self):
        # This allows pickling, since the copy has no weakrefs.
        return self.copy().__dict__

    def __setstate(self, state):
        self.__dict__ = state.copy()

    def __repr__(self):
        return "Tensor(shape={}, inds={}, tags={})".format(
            self.data.shape,
            self.inds,
            self.tags)


# ------------------------- Add ufunc like methods -------------------------- #

def _make_promote_array_func(op, meth_name):

    @functools.wraps(getattr(np.ndarray, meth_name))
    def _promote_array_func(self, other):
        """Use standard array func, but make sure Tensor inds match.
        """
        if isinstance(other, Tensor):

            if set(self.inds) != set(other.inds):
                raise ValueError(
                    "The indicies of these two tensors do not "
                    "match: {} != {}".format(self.inds, other.inds))

            otherT = other.transpose(*self.inds)

            return Tensor(
                data=op(self.data, otherT.data), inds=self.inds,
                tags=self.tags | other.tags)
        else:
            return Tensor(data=op(self.data, other),
                          inds=self.inds, tags=self.tags)

    return _promote_array_func


for meth_name, op in [('__add__', operator.__add__),
                      ('__sub__', operator.__sub__),
                      ('__mul__', operator.__mul__),
                      ('__pow__', operator.__pow__),
                      ('__truediv__', operator.__truediv__)]:
    setattr(Tensor, meth_name, _make_promote_array_func(op, meth_name))


def _make_rhand_array_promote_func(op, meth_name):

    @functools.wraps(getattr(np.ndarray, meth_name))
    def _rhand_array_promote_func(self, other):
        """Right hand operations -- no need to check ind equality first.
        """
        return Tensor(data=op(other, self.data),
                      inds=self.inds, tags=self.tags)

    return _rhand_array_promote_func


for meth_name, op in [('__radd__', operator.__add__),
                      ('__rsub__', operator.__sub__),
                      ('__rmul__', operator.__mul__),
                      ('__rpow__', operator.__pow__),
                      ('__rtruediv__', operator.__truediv__)]:
    setattr(Tensor, meth_name, _make_rhand_array_promote_func(op, meth_name))


class TNLinearOperator(spla.LinearOperator):
    r"""Get a linear operator - something that replicates the matrix-vector
    operation - for an arbitrary uncontracted TensorNetwork, e.g::

                 : --O--O--+ +-- :                 --+
                 :   |     | |   :                   |
                 : --O--O--O-O-- :    acting on    --V
                 :   |     |     :                   |
                 : --+     +---- :                 --+
        left_inds^               ^right_inds

    This can then be supplied to scipy's sparse linear algebra routines.
    The ``left_inds`` / ``right_inds`` convention is that the linear operator
    will have shape matching ``(*left_inds, *right_inds)``, so that the
    ``right_inds`` are those that will be contracted in a normal
    matvec / matmat operation::

        _matvec =    --0--v    , _rmatvec =     v--0--

    Parameters
    ----------
    tns : sequence of Tensors or TensorNetwork
        A representation of the hamiltonian
    left_inds : sequence of str
        The 'left' inds of the effective hamiltonian network.
    right_inds : sequence of str
        The 'right' inds of the effective hamiltonian network. These should be
        ordered the same way as ``left_inds``.
    ldims : tuple of int, or None
        The dimensions corresponding to left_inds. Will figure out if None.
    rdims : tuple of int, or None
        The dimensions corresponding to right_inds. Will figure out if None.
    is_conj : bool, optional
        Whether this object should represent the *adjoint* operator.

    See Also
    --------
    TNLinearOperator1D
    """

    def __init__(self, tns, left_inds, right_inds, ldims=None, rdims=None,
                 backend=None, is_conj=False):
        self.backend = _TENSOR_LINOP_BACKEND if backend is None else backend

        if isinstance(tns, TensorNetwork):
            self._tensors = tns.tensors

            if ldims is None or rdims is None:
                ix_sz = tns.ind_sizes()
                ldims = tuple(ix_sz[i] for i in left_inds)
                rdims = tuple(ix_sz[i] for i in right_inds)

        else:
            self._tensors = tuple(tns)

            if ldims is None or rdims is None:
                ix_sz = dict(zip(concat((t.inds, t.shape) for t in tns)))
                ldims = tuple(ix_sz[i] for i in left_inds)
                rdims = tuple(ix_sz[i] for i in right_inds)

        self.left_inds, self.right_inds = left_inds, right_inds
        self.ldims, ld = ldims, prod(ldims)
        self.rdims, rd = rdims, prod(rdims)

        self._kws = {'get': 'expression'}

        # if recent opt_einsum specify constant tensors
        if hasattr(oe.backends, 'evaluate_constants'):
            self._kws['constants'] = range(len(self._tensors))
            self._ins = ()
        else:
            self._ins = tuple(t.data for t in self._tensors)

        # conjugate inputs/ouputs rather all tensors if necessary
        self.is_conj = is_conj
        self._conj_linop = None
        self._adjoint_linop = None
        self._transpose_linop = None
        self._contractors = {}

        super().__init__(dtype=self._tensors[0].dtype, shape=(ld, rd))

    def _matvec(self, vec):
        in_data = vec.reshape(*self.rdims)

        if self.is_conj:
            in_data = in_data.conj()

        # cache the contractor
        if 'matvec' not in self._contractors:
            # generate a expression that acts directly on the data
            iT = Tensor(in_data, inds=self.right_inds)
            self._contractors['matvec'] = tensor_contract(
                *self._tensors, iT, output_inds=self.left_inds, **self._kws)

        fn = self._contractors['matvec']
        out_data = fn(*self._ins, in_data, backend=self.backend)

        if self.is_conj:
            out_data = out_data.conj()

        return out_data.ravel()

    def _matmat(self, mat):
        d = mat.shape[-1]
        in_data = mat.reshape(*self.rdims, d)

        if self.is_conj:
            in_data = in_data.conj()

        # for matmat need different contraction scheme for different d sizes
        key = "matmat_{}".format(d)

        # cache the contractor
        if key not in self._contractors:
            # generate a expression that acts directly on the data
            iT = Tensor(in_data, inds=(*self.right_inds, '_mat_ix'))
            o_ix = (*self.left_inds, '_mat_ix')
            self._contractors[key] = tensor_contract(
                *self._tensors, iT, output_inds=o_ix, **self._kws)

        fn = self._contractors[key]
        out_data = fn(*self._ins, in_data, backend=self.backend)

        if self.is_conj:
            out_data = out_data.conj()

        return out_data.reshape(-1, d)

    def copy(self, conj=False, transpose=False):
        if transpose:
            inds = self.right_inds, self.left_inds
            dims = self.rdims, self.ldims
        else:
            inds = self.left_inds, self.right_inds
            dims = self.ldims, self.rdims

        if conj:
            is_conj = not self.is_conj
        else:
            is_conj = self.is_conj

        return TNLinearOperator(self._tensors, *inds, *dims,
                                is_conj=is_conj, backend=self.backend)

    def conj(self):
        if self._conj_linop is None:
            self._conj_linop = self.copy(conj=True)
        return self._conj_linop

    def _transpose(self):
        if self._transpose_linop is None:
            self._transpose_linop = self.copy(transpose=True)
        return self._transpose_linop

    def _adjoint(self):
        """Hermitian conjugate of this TNLO.
        """
        # cache the adjoint
        if self._adjoint_linop is None:
            self._adjoint_linop = self.copy(conj=True, transpose=True)
        return self._adjoint_linop

    def to_dense(self, *inds_seq):
        """Convert this TNLinearOperator into a dense array, defaulting to
        grouping the left and right indices respectively.
        """
        if self.is_conj:
            ts = (t.conj() for t in self._tensors)
        else:
            ts = self._tensors

        if not inds_seq:
            inds_seq = self.left_inds, self.right_inds

        return tensor_contract(*ts).to_dense(*inds_seq)

    @property
    def A(self):
        return self.to_dense()

    def astype(self, dtype):
        """Convert this ``TNLinearOperator`` to type ``dtype``.
        """
        return TNLinearOperator(
            (t.astype(dtype) for t in self._tensors),
            left_inds=self.left_inds, right_inds=self.right_inds,
            ldims=self.ldims, rdims=self.rdims, backend=self.backend,
        )


# --------------------------------------------------------------------------- #
#                            Tensor Network Class                             #
# --------------------------------------------------------------------------- #

class TensorNetwork(object):
    r"""A collection of (as yet uncontracted) Tensors.

    Parameters
    ----------
    ts : sequence of Tensor or TensorNetwork
        The objects to combine. The new network will copy these (but not the
        underlying data) by default. For a *view* set ``virtual=True``.
    structure : str, optional
        A string, with integer format specifier, that describes how to range
        over the network's tags in order to contract it. Also allows integer
        indexing rather than having to explcitly use tags.
    structure_bsz : int, optional
        How many sites to group together when auto contracting. Eg for 3 (with
        the dotted lines denoting vertical strips of tensors to be
        contracted)::

            .....       i        ........ i        ...i.
            O-O-O-O-O-O-O-        /-O-O-O-O-        /-O-
            | | | | | | |   ->   1  | | | |   ->   2  |   ->  etc.
            O-O-O-O-O-O-O-        \-O-O-O-O-        \-O-

        Should not require tensor contractions with more than 52 unique
        indices.
    nsites : int, optional
        The total number of sites, if explicitly known. This will be calculated
        using `structure` if needed but not specified. When the network is not
        dense in sites, i.e. ``sites != range(nsites)``, this should be the
        total number of sites the network is embedded in::

            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10  :-> nsites=10
            .  .  .  .  .  .  .  .  .  .  .
                  0--0--0--------0--0         :-> sites=(2, 3, 4, 7, 8)
                  |  |  |        |  |

    sites : sequence of int, optional
        The indices of the sites present in this network, defaults to
        ``range(nsites)``. But could be e.g. ``[0, 1, 4, 5, 7]`` if some sites
        have been removed.
    check_collisions : bool, optional
        If True, the default, then Tensors and TensorNetworks with double
        indices which match another Tensor or TensorNetworks double indices
        will have those indices' names mangled. Can be explicitly turned off
        when it is known that no collisions will take place -- i.e. when not
        adding any new tensors.
    virtual : bool, optional
        Whether the TensorNetwork should be a *view* onto the tensors it is
        given, or a copy of them. E.g. if a virtual TN is constructed, any
        changes to a Tensor's indices or tags will propagate to all TNs viewing
        that Tensor.

    Attributes
    ----------
    tensor_map : dict
        Mapping of unique ids to tensors, like``{tensor_id: tensor, ...}``.
        I.e. this is where the tensors are 'stored' by the network.
    tag_map : dict
        Mapping of tags to a set of tensor ids which have those tags. I.e.
        ``{tag: {tensor_id_1, tensor_id_2, ...}}``. Thus to select those
        tensors could do: ``map(tensor_map.__getitem__, tag_map[tag])``.
    ind_map : dict
        Like ``tag_map`` but for indices. So ``ind_map[ind]]`` returns the
        tensor ids of those tensors with ``ind``.
    """

    def __init__(self, ts, *,
                 virtual=False,
                 structure=None,
                 structure_bsz=None,
                 nsites=None,
                 sites=None,
                 check_collisions=True):

        # short-circuit for copying TensorNetworks
        if isinstance(ts, TensorNetwork):
            self.structure = ts.structure
            self.nsites = ts.nsites
            self.sites = ts.sites
            self.structure_bsz = ts.structure_bsz
            self.tag_map = valmap(lambda tid: tid.copy(), ts.tag_map)
            self.ind_map = valmap(lambda tid: tid.copy(), ts.ind_map)
            self.tensor_map = {}
            for tid, t in ts.tensor_map.items():
                self.tensor_map[tid] = t if virtual else t.copy()
                self.tensor_map[tid].add_owner(self, tid)
            return

        # parameters
        self.structure = structure
        self.structure_bsz = structure_bsz
        self.nsites = nsites
        self.sites = sites

        # internal structure
        self.tensor_map = {}
        self.tag_map = {}
        self.ind_map = {}

        inner_inds = set()
        for t in ts:
            self.add(t, virtual=virtual, inner_inds=inner_inds,
                     check_collisions=check_collisions)

        if self.structure:
            # set the list of indices of sites which are present
            if self.sites is None:
                if self.nsites is None:
                    self.nsites = self.calc_nsites()
                self.sites = range(self.nsites)
            else:
                if self.nsites is None:
                    raise ValueError("The total number of sites, ``nsites`` "
                                     "must be specified when a custom subset, "
                                     "i.e. ``sites``, is.")

            # set default blocksize
            if self.structure_bsz is None:
                self.structure_bsz = 10

    def _combine_properties(self, other):
        props_equals = (('structure', lambda u, v: u == v),
                        ('nsites', lambda u, v: u == v),
                        ('structure_bsz', lambda u, v: u == v),
                        ('contract_structured_all', functions_equal))

        for prop, equal in props_equals:

            # check whether to inherit ... or compare properties
            u, v = getattr(self, prop, None), getattr(other, prop, None)

            if v is not None:
                # don't have prop yet -> inherit
                if u is None:
                    setattr(self, prop, v)

                # both have prop, and don't match -> raise
                elif not equal(u, v):
                    raise ValueError(
                        "Conflicting values found on tensor networks for "
                        "property {}. First value: {}, second value: {}"
                        .format(prop, u, v))

    def __and__(self, other):
        """Combine this tensor network with more tensors, without contracting.
        Copies the tensors.
        """
        return TensorNetwork((self, other))

    def __or__(self, other):
        """Combine this tensor network with more tensors, without contracting.
        Views the constituent tensors.
        """
        return TensorNetwork((self, other), virtual=True)

    # ------------------------------- Methods ------------------------------- #

    def copy(self, virtual=False, deep=False):
        """Copy this ``TensorNetwork``. If ``deep=False``, (the default), then
        everything but the actual numeric data will be copied.
        """
        if deep:
            return copy.deepcopy(self)
        return self.__class__(self, virtual=virtual)

    __copy__ = copy

    @staticmethod
    def _add_tid(xs, x_map, tid):
        """Add tid to the relevant map.
        """
        for x in xs:
            if x in x_map:
                x_map[x].add(tid)
            else:
                x_map[x] = {tid}

    @staticmethod
    def _remove_tid(xs, x_map, tid):
        """Remove tid from the relevant map.
        """
        for x in xs:
            try:
                tids = x_map[x]
                tids.discard(tid)
                if not tids:
                    # tid was last tensor -> delete entry
                    del x_map[x]
            except KeyError:
                # tid already removed from x entry - e.g. repeated index
                pass

    def add_tensor(self, tensor, tid=None, virtual=False):
        """Add a single tensor to this network - mangle its tid if neccessary.
        """
        # check for tid conflict
        if (tid is None) or (tid in self.tensor_map):
            tid = rand_uuid(base="_T")

        # add tensor to the main index
        T = tensor if virtual else tensor.copy()
        self.tensor_map[tid] = T
        T.add_owner(self, tid)

        # add its tid to the relevant tags and inds, or create new entries
        self._add_tid(T.tags, self.tag_map, tid)
        self._add_tid(T.inds, self.ind_map, tid)

    def add_tensor_network(self, tn, virtual=False, check_collisions=True,
                           inner_inds=None):
        """
        """
        self._combine_sites(tn)
        self._combine_properties(tn)

        if check_collisions:  # add tensors individually
            if inner_inds is None:
                inner_inds = set(self.inner_inds())

            # check for matching inner_indices -> need to re-index
            tn_iix = set(tn.inner_inds())
            b_ix = inner_inds & tn_iix

            if b_ix:
                g_ix = tn_iix - inner_inds
                new_inds = {rand_uuid() for _ in range(len(b_ix))}
                reind = dict(zip(b_ix, new_inds))
                inner_inds |= new_inds
                inner_inds |= g_ix
            else:
                inner_inds |= tn_iix

            # add tensors, reindexing if necessary
            for tid, tsr in tn.tensor_map.items():
                if b_ix and any(i in reind for i in tsr.inds):
                    tsr = tsr.reindex(reind, inplace=virtual)
                self.add_tensor(tsr, virtual=virtual, tid=tid)

        else:  # directly add tensor/tag indexes
            for tid, tsr in tn.tensor_map.items():
                T = tsr if virtual else tsr.copy()
                self.tensor_map[tid] = T
                T.add_owner(self, tid)

            self.tag_map = merge_with(set_union, self.tag_map, tn.tag_map)
            self.ind_map = merge_with(set_union, self.ind_map, tn.ind_map)

    def add(self, t, virtual=False, check_collisions=True, inner_inds=None):
        """Add Tensor, TensorNetwork or sequence thereof to self.
        """
        if isinstance(t, (tuple, list)):
            for each_t in t:
                self.add(each_t, inner_inds=inner_inds, virtual=virtual,
                         check_collisions=check_collisions)
            return

        istensor = isinstance(t, Tensor)
        istensornetwork = isinstance(t, TensorNetwork)

        if not (istensor or istensornetwork):
            raise TypeError("TensorNetwork should be called as "
                            "`TensorNetwork(ts, ...)`, where each "
                            "object in 'ts' is a Tensor or "
                            "TensorNetwork.")

        if istensor:
            self.add_tensor(t, virtual=virtual)
        else:
            self.add_tensor_network(t, virtual=virtual, inner_inds=inner_inds,
                                    check_collisions=check_collisions)

    def __iand__(self, tensor):
        """Inplace, but non-virtual, addition of a Tensor or TensorNetwork to
        this network. It should not have any conflicting indices.
        """
        self.add(tensor, virtual=False)
        return self

    def __ior__(self, tensor):
        """Inplace, virtual, addition of a Tensor or TensorNetwork to this
        network. It should not have any conflicting indices.
        """
        self.add(tensor, virtual=True)
        return self

    def _modify_tensor_tags(self, old, new, tid):
        self._remove_tid((o for o in old if o not in new), self.tag_map, tid)
        self._add_tid((n for n in new if n not in old), self.tag_map, tid)

    def _modify_tensor_inds(self, old, new, tid):
        self._remove_tid((o for o in old if o not in new), self.ind_map, tid)
        self._add_tid((n for n in new if n not in old), self.ind_map, tid)

    def calc_nsites(self):
        """Calculate how many tags there are which match ``structure``.
        """
        return len(re.findall(self.structure.format("(\d+)"), str(self.tags)))

    @staticmethod
    @functools.lru_cache(8)
    def regex_for_calc_sites_cached(structure):
        return re.compile(structure.format("(\d+)"))

    def calc_sites(self):
        """Calculate with sites this TensorNetwork contain based on its
        ``structure``.
        """
        rgx = self.regex_for_calc_sites_cached(self.structure)
        matches = rgx.findall(str(self.tags))
        sites = sorted(map(int, matches))

        # check if can convert to contiguous range
        mn, mx = min(sites), max(sites) + 1
        if len(sites) == mx - mn:
            sites = range(mn, mx)

        return sites

    def _combine_sites(self, other):
        """Correctly combine the sites list of two TNs.
        """
        if (self.sites != other.sites) and (other.sites is not None):
            if self.sites is None:
                self.sites = other.sites
            else:
                self.sites = tuple(sorted(set(self.sites) | set(other.sites)))

                mn, mx = min(self.sites), max(self.sites) + 1
                if len(self.sites) == mx - mn:
                    self.sites = range(mn, mx)

    def _pop_tensor(self, tid):
        """Remove a tensor from this network, returning said tensor.
        """
        # pop the tensor itself
        t = self.tensor_map.pop(tid)

        # remove the tid from the tag and ind maps
        self._remove_tid(t.tags, self.tag_map, tid)
        self._remove_tid(t.inds, self.ind_map, tid)

        # remove this tensornetwork as an owner
        t.remove_owner(self)

        return t

    def delete(self, tags, which='all'):
        """Delete any tensors which match all or any of ``tags``.

        Parameters
        ----------
        tags : str or sequence of str
            The tags to match.
        which : {'all', 'any'}, optional
            Whether to match all or any of the tags.
        """
        tids = self._get_tids_from_tags(tags, which=which)
        for tid in tuple(tids):
            self._pop_tensor(tid)

    def add_tag(self, tag, where=None, which='all'):
        """Add tag to every tensor in this network, or if ``where`` is
        specified, the tensors matching those tags -- i.e. adds the tag to
        all tensors in ``self.select_tensors(where, which=which)``.
        """
        tids = self._get_tids_from_tags(where, which=which)

        for tid in tids:
            self.tensor_map[tid].add_tag(tag)

    def drop_tags(self, tags):
        """Remove a tag from any tensors in this network which have it.
        Inplace operation.

        Parameters
        ----------
        tags : str or sequence of str
            The tag or tags to drop.
        """
        tags = tags2set(tags)

        for t in self:
            t.drop_tags(tags)

    def retag(self, tag_map, inplace=False):
        """Rename tags for all tensors in this network, optionally in-place.

        Parameters
        ----------
        tag_map : dict-like
            Mapping of pairs ``{old_tag: new_tag, ...}``.
        inplace : bool, optional
            Perform operation inplace or return copy (default).
        """
        tn = self if inplace else self.copy()

        # get ids of tensors which have any of the tags
        tids = tn._get_tids_from_tags(tag_map.keys(), which='any')

        for tid in tids:
            t = tn.tensor_map[tid]
            t.retag_(tag_map)

        return tn

    retag_ = functools.partialmethod(retag, inplace=True)

    def reindex(self, index_map, inplace=False):
        """Rename indices for all tensors in this network, optionally in-place.

        Parameters
        ----------
        index_map : dict-like
            Mapping of pairs ``{old_ind: new_ind, ...}``.
        """
        tn = self if inplace else self.copy()

        tids = set_union(tn.ind_map.get(ix, set()) for ix in index_map)

        for tid in tids:
            T = tn.tensor_map[tid]
            T.reindex_(index_map)

        return tn

    reindex_ = functools.partialmethod(reindex, inplace=True)

    def conj(self, inplace=False):
        """Conjugate all the tensors in this network (leaves all indices).
        """
        new_tn = self if inplace else self.copy()

        for t in new_tn:
            t.conj_()

        return new_tn

    conj_ = functools.partialmethod(conj, inplace=True)

    @property
    def H(self):
        """Conjugate all the tensors in this network (leaves all indices).
        """
        return self.conj()

    def multiply(self, x, inplace=False):
        """Scalar multiplication of this tensor network with ``x``.
        """
        multiplied = self if inplace else self.copy()
        tensor = next(iter(multiplied))
        tensor.modify(data=tensor.data * x)
        return multiplied

    def __mul__(self, other):
        """Scalar multiplication.
        """
        return self.multiply(other, inplace=False)

    def __rmul__(self, other):
        """Right side scalar multiplication.
        """
        return self.multiply(other, inplace=False)

    def __imul__(self, other):
        """Inplace scalar multiplication.
        """
        return self.multiply(other, inplace=True)

    def __truediv__(self, other):
        """Scalar division.
        """
        return self.multiply(1 / other, inplace=False)

    def __itruediv__(self, other):
        """Inplace scalar division.
        """
        return self.multiply(1 / other, inplace=True)

    @property
    def tensors(self):
        return tuple(self)

    def tensors_sorted(self):
        """Return a tuple of tensors sorted by their respective tags, such that
        the tensors of two networks with the same tag structure can be
        iterated over pairwise.
        """
        ts_and_sorted_tags = [(t, sorted(t.tags)) for t in self]
        ts_and_sorted_tags.sort(key=lambda x: x[1])
        return tuple(x[0] for x in ts_and_sorted_tags)

    def __iter__(self):
        return iter(self.tensor_map.values())

    def apply_to_arrays(self, fn):
        """Modify every tensor's array inplace by applying ``fn`` to it.
        """
        for t in self:
            t.modify(data=fn(t.data))

    # ----------------- selecting and splitting the network ----------------- #

    def slice2sites(self, tag_slice):
        """Take a slice object, and work out its implied start, stop and step,
        taking into account cyclic boundary conditions.

        Examples
        --------
        Normal slicing:

            >>> p = MPS_rand_state(10, bond_dim=7)
            >>> p.slice2sites(slice(5))
            (0, 1, 2, 3, 4)

            >>> p.slice2sites(slice(4, 8))
            (4, 5, 6, 7)

        Slicing from end backwards:

            >>> p.slice2sites(slice(..., -3, -1))
            (9, 8)

        Slicing round the end:

            >>> p.slice2sites(slice(7, 12))
            (7, 8, 9, 0, 1)

            >>> p.slice2sites(slice(-3, 2))
            (7, 8, 9, 0, 1)

        If the start point is > end point (*before* modulo n), then step needs
        to be negative to return anything.
        """
        if tag_slice.start is None:
            start = 0
        elif tag_slice.start is ...:
            if tag_slice.step == -1:
                start = self.nsites - 1
            else:
                start = -1
        else:
            start = tag_slice.start

        if tag_slice.stop in (..., None):
            stop = self.nsites
        else:
            stop = tag_slice.stop

        step = 1 if tag_slice.step is None else tag_slice.step

        return tuple(s % self.nsites for s in range(start, stop, step))

    def site_tag(self, i):
        """Get the tag corresponding to site ``i``, taking into account
        periodic boundary conditions.
        """
        return self.structure.format(i % self.nsites)

    def sites2tags(self, sites):
        """Take a integer or slice and produce the correct set of tags.

        Parameters
        ----------
        sites : int or slice
            The site(s). If ``slice``, non inclusive of end.

        Returns
        -------
        tags : set
            The correct tags describing those sites.
        """
        if isinstance(sites, Integral):
            return {self.site_tag(sites)}
        elif isinstance(sites, slice):
            return set(map(self.structure.format, self.slice2sites(sites)))
        else:
            raise TypeError("``sites2tags`` needs an integer or a slice"
                            ", but got {}".format(sites))

    def _get_tids_from(self, xmap, xs, which):
        inverse = which[0] == '!'
        if inverse:
            which = which[1:]

        combine = {'all': set.intersection, 'any': set.union}[which]
        tid_sets = (xmap[x] for x in xs)
        tids = combine(*tid_sets)

        if inverse:
            return set(self.tensor_map) - tids

        return tids

    def _get_tids_from_tags(self, tags, which='all'):
        """Return the set of tensor ids that match ``tags``.

        Parameters
        ----------
        tags : seq or str, str, None, ..., int, slice
            Tag specifier(s).
        which : {'all', 'any', '!all', '!any'}
            How to select based on the tags, if:

            - 'all': get ids of tensors matching all tags
            - 'any': get ids of tensors matching any tags
            - '!all': get ids of tensors *not* matching all tags
            - '!any': get ids of tensors *not* matching any tags

        Returns
        -------
        set[str]
        """
        if tags in (None, ..., all):
            return set(self.tensor_map)
        elif isinstance(tags, (Integral, slice)):
            tags = self.sites2tags(tags)
        else:
            tags = tags2set(tags)

        return self._get_tids_from(self.tag_map, tags, which)

    def _get_tids_from_inds(self, inds, which='all'):
        """Like ``_get_tids_from_tags`` but specify inds instead.
        """
        inds = tags2set(inds)
        return self._get_tids_from(self.ind_map, inds, which)

    def select_tensors(self, tags, which='all'):
        """Return the sequence of tensors that match ``tags``. If
        ``which='all'``, each tensor must contain every tag. If
        ``which='any'``, each tensor can contain any of the tags.

        Parameters
        ----------
        tags : str or sequence of str
            The tag or tag sequence.
        which : {'all', 'any'}
            Whether to require matching all or any of the tags.

        Returns
        -------
        tagged_tensors : tuple of Tensor
            The tagged tensors.

        See Also
        --------
        select, select_neighbors, partition, partition_tensors
        """
        tids = self._get_tids_from_tags(tags, which=which)
        return tuple(self.tensor_map[n] for n in tids)

    def select(self, tags, which='all'):
        """Get a TensorNetwork comprising tensors that match all or any of
        ``tags``, inherit the network properties/structure from ``self``.
        This returns a view of the tensors not a copy.

        Parameters
        ----------
        tags : str or sequence of str
            The tag or tag sequence.
        which : {'all', 'any'}
            Whether to require matching all or any of the tags.

        Returns
        -------
        tagged_tensors : tuple of Tensor
            The tagged tensors.

        See Also
        --------
        select_tensors, select_neighbors, partition, partition_tensors
        """
        tagged_tids = self._get_tids_from_tags(tags, which=which)
        ts = (self.tensor_map[n] for n in tagged_tids)

        tn = TensorNetwork(ts, check_collisions=False, virtual=True,
                           structure=self.structure, nsites=self.nsites,
                           structure_bsz=self.structure_bsz)

        if self.structure is not None:
            tn.sites = tn.calc_sites()

        return tn

    def select_neighbors(self, tags, which='any'):
        """Select any neighbouring tensors to those specified by ``tags``.self

        Parameters
        ----------
        tags : sequence of str, int
            Tags specifying tensors.
        which : {'any', 'all'}, optional
            How to select tensors based on ``tags``.

        Returns
        -------
        tuple[Tensor]
            The neighbouring tensors.

        See Also
        --------
        select_tensors, partition_tensors
        """

        # find all the inds in the tagged portion
        tagged_tids = self._get_tids_from_tags(tags, which)
        tagged_ts = (self.tensor_map[tid] for tid in tagged_tids)
        inds = set(concat(t.inds for t in tagged_ts))

        # find all tensors with those inds, and remove the initial tensors
        inds_tids = set_union(self.ind_map[i] for i in inds)
        neighbour_tids = inds_tids - tagged_tids

        return tuple(self.tensor_map[tid] for tid in neighbour_tids)

    def __getitem__(self, tags):
        """Get the tensor(s) associated with ``tags``.

        Parameters
        ----------
        tags : str or sequence of str
            The tags used to select the tensor(s).

        Returns
        -------
        Tensor or sequence of Tensors
        """
        if isinstance(tags, slice):
            return self.select(self.sites2tags(tags), which='any')

        elif isinstance(tags, Integral):
            tensors = self.select_tensors(self.sites2tags(tags), which='any')

        else:
            tensors = self.select_tensors(tags, which='all')

        if len(tensors) == 0:
            raise KeyError("Couldn't find any tensors "
                           "matching {}.".format(tags))

        if len(tensors) == 1:
            return tensors[0]

        return tensors

    def __setitem__(self, tags, tensor):
        """Set the single tensor uniquely associated with ``tags``.
        """
        tids = self._get_tids_from_tags(tags, which='all')
        if len(tids) != 1:
            raise KeyError("'TensorNetwork.__setitem__' is meant for a single "
                           "existing tensor only - found {} with tag(s) '{}'."
                           .format(len(tids), tags))

        if not isinstance(tensor, Tensor):
            raise TypeError("Can only set value with a new 'Tensor'.")

        tid, = tids

        # check if tags match, else need to modify TN structure
        if self.tensor_map[tid].tags != tensor.tags:
            self._pop_tensor(tid)
            self.add_tensor(tensor, tid, virtual=True)
        else:
            self.tensor_map[tid] = tensor

    def __delitem__(self, tags):
        """Delete any tensors which have all of ``tags``.
        """
        tids = self._get_tids_from_tags(tags, which='all')
        for tid in tuple(tids):
            self._pop_tensor(tid)

    def partition_tensors(self, tags, inplace=False, which='any'):
        """Split this TN into a list of tensors containing any or all of
        ``tags`` and a ``TensorNetwork`` of the the rest.

        Parameters
        ----------
        tags : sequence of str
            The list of tags to filter the tensors by. Use ``...``
            (``Ellipsis``) to filter all.
        inplace : bool, optional
            If true, remove tagged tensors from self, else create a new network
            with the tensors removed.
        which : {'all', 'any'}
            Whether to require matching all or any of the tags.

        Returns
        -------
        (u_tn, t_ts) : (TensorNetwork, tuple of Tensors)
            The untagged tensor network, and the sequence of tagged Tensors.

        See Also
        --------
        partition, select, select_tensors
        """
        tagged_tids = self._get_tids_from_tags(tags, which=which)

        # check if all tensors have been tagged
        if len(tagged_tids) == len(self.tensor_map):
            return None, self.tensor_map.values()

        # Copy untagged to new network, and pop tagged tensors from this
        untagged_tn = self if inplace else self.copy()
        tagged_ts = tuple(map(untagged_tn._pop_tensor, sorted(tagged_tids)))

        return untagged_tn, tagged_ts

    def partition(self, tags, which='any', inplace=False, calc_sites=True):
        """Split this TN into two, based on which tensors have any or all of
        ``tags``. Unlike ``partition_tensors``, both results are TNs which
        inherit the structure of the initial TN.

        Parameters
        ----------
        tags : sequence of str
            The tags to split the network with.
        which : {'any', 'all'}
            Whether to split based on matching any or all of the tags.
        inplace : bool
            If True, actually remove the tagged tensors from self.
        calc_sites : bool
            If True, calculate which sites belong to which network.

        Returns
        -------
        untagged_tn, tagged_tn : (TensorNetwork, TensorNetwork)
            The untagged and tagged tensor networs.

        See Also
        --------
        partition_tensors, select, select_tensors
        """
        tagged_tids = self._get_tids_from_tags(tags, which=which)

        kws = {'check_collisions': False, 'structure': self.structure,
               'structure_bsz': self.structure_bsz, 'nsites': self.nsites}

        if inplace:
            t1 = self
            t2s = [t1._pop_tensor(tid) for tid in tagged_tids]
            t2 = TensorNetwork(t2s, **kws)

        else:  # rebuild both -> quicker
            t1s, t2s = [], []
            for tid, tensor in self.tensor_map.items():
                (t2s if tid in tagged_tids else t1s).append(tensor)

            t1, t2 = TensorNetwork(t1s, **kws), TensorNetwork(t2s, **kws)

        if calc_sites and self.structure is not None:
            t1.sites = t1.calc_sites()
            t2.sites = t2.calc_sites()

        return t1, t2

    def replace_with_identity(self, where, which='any', inplace=False):
        r"""Replace all tensors marked by ``where`` with an
        identity. E.g. if ``X`` denote ``where`` tensors::


            ---1  X--X--2---         ---1---2---
               |  |  |  |      ==>          |
               X--X--X  |                   |

        Parameters
        ----------
        where : tag or seq of tags
            Tags specifying the tensors to replace.
        which : {'any', 'all'}
            Whether to replace tensors matching any or all the tags ``where``.
        inplace : bool
            Perform operation in place.

        Returns
        -------
        TensorNetwork
            The TN, with section replaced with identity.

        See Also
        --------
        replace_with_svd
        """
        tn = self if inplace else self.copy()

        if not where:
            return tn

        (dl, il), (dr, ir) = TensorNetwork(
            self.select_tensors(where, which=which)).outer_dims_inds()

        if dl != dr:
            raise ValueError(
                "Can only replace_with_identity when the remaining indices "
                "have matching dimensions, but {} != {}.".format(dl, dr))

        tn.delete(where, which=which)

        tn.reindex_({il: ir})
        return tn

    def replace_with_svd(self, where, left_inds, eps, *, which='any',
                         right_inds=None, method='isvd', max_bond=None,
                         ltags=None, rtags=None, keep_tags=True,
                         start=None, stop=None, inplace=False):
        r"""Replace all tensors marked by ``where`` with an iteratively
        constructed SVD. E.g. if ``X`` denote ``where`` tensors::

                                    :__       ___:
            ---X  X--X  X---        :  \     /   :
               |  |  |  |      ==>  :   U~s~VH---:
            ---X--X--X--X---        :__/     \   :
                  |     +---        :         \__:
                  X              left_inds       :
                                             right_inds

        Parameters
        ----------
        where : tag or seq of tags
            Tags specifying the tensors to replace.
        left_inds : ind or sequence of inds
            The indices defining the left hand side of the SVD.
        eps : float
            The tolerance to perform the SVD with, affects the number of
            singular values kept. See
            :func:`quimb.linalg.rand_linalg.estimate_rank`.
        which : {'any', 'all', '!any', '!all'}, optional
            Whether to replace tensors matching any or all the tags ``where``,
            prefix with '!' to invert the selection.
        right_inds : ind or sequence of inds, optional
            The indices defining the right hand side of the SVD, these can be
            automatically worked out, but for hermitian decompositions the
            order is important and thus can be given here explicitly.
        method : {'isvd', 'eig', 'eigh', 'svd', 'svds', 'eigsh', 'cholesky'}
            How to perform the decomposition, if not an iterative method
            ('isvd', 'svds', 'eigsh'), the subnetwork dense tensor will be
            formed first.
        max_bond : int, optional
            The maximum bond to keep, defaults to no maximum (-1).
        ltags : sequence of str, optional
            Tags to add to the left tensor.
        rtags : sequence of str, optional
            Tags to add to the right tensor.
        keep_tags : bool, optional
            Whether to propagate tags found in the subnetwork to both new
            tensors or drop them, defaults to ``True``.
        start : int, optional
            If given, assume can use ``TNLinearOperator1D``.
        stop :  int, optional
            If given, assume can use ``TNLinearOperator1D``.
        inplace : bool, optional
            Perform operation in place.

        Returns
        -------

        See Also
        --------
        replace_with_identity
        """
        leave, svd_section = self.partition(where, which=which,
                                            inplace=inplace, calc_sites=False)

        tags = svd_section.tags if keep_tags else set()
        ltags = tags2set(ltags)
        rtags = tags2set(rtags)

        if right_inds is None:
            # compute
            right_inds = tuple(i for i in svd_section.outer_inds()
                               if i not in left_inds)

        if (start is None) and (stop is None):
            A = svd_section.aslinearoperator(left_inds=left_inds,
                                             right_inds=right_inds)
        else:
            # check if need to invert start stop as well
            if '!' in which:
                start, stop = stop, start + self.nsites
                left_inds, right_inds = right_inds, left_inds
                ltags, rtags = rtags, ltags

            A = TNLinearOperator1D(svd_section, start=start, stop=stop,
                                   left_inds=left_inds, right_inds=right_inds)

        left_shp, right_shp = A.ldims, A.rdims

        opts = {'max_bond': -1 if max_bond is None else max_bond}

        if method in ('svd', 'eig', 'eigh', 'cholesky'):
            if not isinstance(A, np.ndarray):
                A = A.to_dense()

        U, V = {
            'svd': _array_split_svd,
            'eig': _array_split_eig,
            'eigh': _array_split_eigh,
            'cholesky': _array_split_cholesky,
            'isvd': _array_split_isvd,
            'svds': _array_split_svds,
            'rsvd': _array_split_rsvd,
            'eigsh': _array_split_eigsh,
        }[method](A, cutoff=eps, **opts)

        U = U.reshape(*left_shp, -1)
        V = V.reshape(-1, *right_shp)

        new_bnd = rand_uuid()

        # Add the new, compressed tensors back in
        leave |= Tensor(U, inds=(*left_inds, new_bnd), tags=tags | ltags)
        leave |= Tensor(V, inds=(new_bnd, *right_inds), tags=tags | rtags)

        return leave

    def replace_section_with_svd(self, start, stop, eps,
                                 **replace_with_svd_opts):
        """Take a 1D tensor network, and replace a section with a SVD.
        See :meth:`~quimb.tensor.tensor_core.TensorNetwork.replace_with_svd`.

        Parameters
        ----------
        start : int
            Section start index.
        stop : int
            Section stop index, not included itself.
        eps : float
            Precision of SVD.
        replace_with_svd_opts
            Supplied to
            :meth:`~quimb.tensor.tensor_core.TensorNetwork.replace_with_svd`.

        Returns
        -------
        TensorNetwork
        """
        return self.replace_with_svd(
            where=slice(start, stop), start=start, stop=stop,
            left_inds=bonds(self[start - 1], self[start]), eps=eps,
            **replace_with_svd_opts)

    def convert_to_zero(self):
        """Inplace conversion of this network to an all zero tensor network.
        """
        outer_inds = self.outer_inds()

        for T in self:
            new_shape = tuple(d if i in outer_inds else 1
                              for d, i in zip(T.shape, T.inds))
            T.modify(data=np.zeros(new_shape, dtype=T.dtype))

    def compress_between(self, tags1, tags2, **compress_opts):
        """Compress the bond between the two single tensors in this network
        specified by ``tags1`` and ``tags2`` using ``tensor_compress_bond``.
        This is an inplace operation.
        """
        n1, = self._get_tids_from_tags(tags1, which='all')
        n2, = self._get_tids_from_tags(tags2, which='all')
        tensor_compress_bond(self.tensor_map[n1], self.tensor_map[n2])

    def compress_all(self, **compress_opts):
        """Inplace compress all bonds in this network.
        """
        for T1, T2 in itertools.combinations(self.tensors, 2):
            try:
                tensor_compress_bond(T1, T2, **compress_opts)
            except ValueError:
                continue
            except ZeroDivisionError:
                self.convert_to_zero()
                break

    def add_bond(self, tags1, tags2):
        """Inplace addition of a dummmy (size 1) bond between the single
        tensors specified by by ``tags1`` and ``tags2``.
        """
        tid1, = self._get_tids_from_tags(tags1, which='all')
        tid2, = self._get_tids_from_tags(tags2, which='all')

        T1, T2 = self.tensor_map[tid1], self.tensor_map[tid2]
        tensor_add_bond(T1, T2)

        bnd, = bonds(T1, T2)
        self.ind_map[bnd] = {tid1, tid2}

    def cut_bond(self, left_tags, right_tags, left_ind, right_ind):
        """Cut the bond between the tensors specified by ``left_tags`` and
        ``right_tags``, giving them the new inds ``left_ind`` and
        ``right_ind`` respectively.
        """
        tid_l, = self._get_tids_from_tags(left_tags)
        tid_r, = self._get_tids_from_tags(right_tags)

        TL, TR = self.tensor_map[tid_l], self.tensor_map[tid_r]
        bnd, = bonds(TL, TR)

        TL.reindex_({bnd: left_ind})
        TR.reindex_({bnd: right_ind})

    def insert_operator(self, A, where1, where2, tags=None, inplace=False):
        r"""Insert an operator on the bond between the specified tensors,
        e.g.::

              |   |              |   |
            --1---2--    ->    --1-A-2--
              |                  |

        Parameters
        ----------
        A : array
            The operator to insert.
        where1 : str, sequence of str, or int
            The tags defining the 'left' tensor.
        where2 : str, sequence of str, or int
            The tags defining the 'right' tensor.
        tags : str or sequence of str
            Tags to add to the new operator's tensor.
        inplace : bool, optional
            Whether to perform the insertion inplace.
        """
        tn = self if inplace else self.copy()

        d = A.shape[0]

        T1, T2 = tn[where1], tn[where2]
        bnd, = bonds(T1, T2)
        db = T1.ind_size(bnd)

        if d != db:
            raise ValueError("This operator has dimension {} but needs "
                             "dimension {}.".format(d, db))

        # reindex one tensor, and add a new A tensor joining the bonds
        nbnd = rand_uuid()
        T2.reindex_({bnd: nbnd})
        TA = Tensor(A, inds=(bnd, nbnd), tags=tags)
        tn |= TA

        return tn

    def insert_gauge(self, U, where1, where2, Uinv=None, tol=1e-10):
        """Insert the gauge transformation ``U @ U^-1`` into the bond between
        the tensors, ``T1`` and ``T2``, defined by ``where1`` and ``where2``.
        The resulting tensors at those locations will be ``T1 @ U^-1`` and
        ``T2 @ U``.

        Parameters
        ----------
        U : np.ndarray
            The gauge to insert.
        where1 : str, sequence of str, or int
            Tags defining the location of the 'left' tensor.
        where2 : str, sequence of str, or int
            Tags defining the location of the 'right' tensor.
        Uinv : np.ndarray
            The inverse gauge, ``U @ Uinv == Uinv @ U == eye``, to insert.
            If not given will be calculated using :func:`numpy.linalg.inv`.
        """
        n1, = self._get_tids_from_tags(where1, which='all')
        n2, = self._get_tids_from_tags(where2, which='all')
        T1, T2 = self.tensor_map[n1], self.tensor_map[n2]
        bnd, = T1.bonds(T2)

        if Uinv is None:
            Uinv = np.linalg.inv(U)

            # if we get wildly larger inverse due to singular U, try pseudo-inv
            if vdot(Uinv, Uinv) / vdot(U, U) > 1 / tol:
                Uinv = np.linalg.pinv(U, rcond=tol**0.5)

            # if still wildly larger inverse raise an error
            if vdot(Uinv, Uinv) / vdot(U, U) > 1 / tol:
                raise np.linalg.LinAlgError("Ill conditioned inverse.")

        T1Ui = Tensor(Uinv, inds=('__dummy__', bnd)) @ T1
        T2U = Tensor(U, inds=(bnd, '__dummy__')) @ T2

        T1Ui.transpose_like_(T1)
        T2U.transpose_like_(T2)

        T1.modify(data=T1Ui.data)
        T2.modify(data=T2U.data)

    # ----------------------- contracting the network ----------------------- #

    def contract_tags(self, tags, inplace=False, which='any', **opts):
        """Contract the tensors that match any or all of ``tags``.

        Parameters
        ----------
        tags : sequence of str
            The list of tags to filter the tensors by. Use ``...``
            (``Ellipsis``) to contract all.
        inplace : bool, optional
            Whether to perform the contraction inplace.
        which : {'all', 'any'}
            Whether to require matching all or any of the tags.

        Returns
        -------
        TensorNetwork, Tensor or scalar
            The result of the contraction, still a ``TensorNetwork`` if the
            contraction was only partial.

        See Also
        --------
        contract, contract_cumulative, contract_structured
        """
        untagged_tn, tagged_ts = self.partition_tensors(
            tags, inplace=inplace, which=which)

        if not tagged_ts:
            raise ValueError("No tags were found - nothing to contract. "
                             "(Change this to a no-op maybe?)")

        contracted = tensor_contract(*tagged_ts, **opts)

        if untagged_tn is None:
            return contracted

        untagged_tn.add_tensor(contracted, virtual=True)
        return untagged_tn

    def contract_cumulative(self, tags_seq, inplace=False, **opts):
        """Cumulative contraction of tensor network. Contract the first set of
        tags, then that set with the next set, then both of those with the next
        and so forth. Could also be described as an manually ordered
        contraction of all tags in ``tags_seq``.

        Parameters
        ----------
        tags_seq : sequence of sequence of str
            The list of tag-groups to cumulatively contract.
        inplace : bool, optional
            Whether to perform the contraction inplace.

        Returns
        -------
        TensorNetwork, Tensor or scalar
            The result of the contraction, still a ``TensorNetwork`` if the
            contraction was only partial.

        See Also
        --------
        contract, contract_tags, contract_structured
        """
        tn = self if inplace else self.copy()
        c_tags = set()

        for tags in tags_seq:
            # accumulate tags from each contractions
            c_tags |= tags2set(tags)

            # peform the next contraction
            tn = tn.contract_tags(c_tags, inplace=True, which='any', **opts)

            if isinstance(tn, Tensor) or np.isscalar(tn):
                # nothing more to contract
                break

        return tn

    def contract_structured(self, tag_slice, inplace=False, **opts):
        """Perform a structured contraction, translating ``tag_slice`` from a
        ``slice`` or `...` to a cumulative sequence of tags.

        Parameters
        ----------
        tag_slice : slice or ...
            The range of sites, or `...` for all.
        inplace : bool, optional
            Whether to perform the contraction inplace.

        Returns
        -------
        TensorNetwork, Tensor or scalar
            The result of the contraction, still a ``TensorNetwork`` if the
            contraction was only partial.

        See Also
        --------
        contract, contract_tags, contract_cumulative
        """
        # check for all sites
        if tag_slice is ...:

            # check for a custom structured full contract sequence
            if hasattr(self, "contract_structured_all"):
                return self.contract_structured_all(
                    self, inplace=inplace, **opts)

            # else slice over all sites
            tag_slice = slice(0, self.nsites)

        # filter sites by the slice, but also which sites are present at all
        sites = self.slice2sites(tag_slice)
        tags_seq = (self.structure.format(s) for s in sites if s in self.sites)

        # partition sites into `structure_bsz` groups
        if self.structure_bsz > 1:
            tags_seq = partition_all(self.structure_bsz, tags_seq)

        # contract each block of sites cumulatively
        return self.contract_cumulative(tags_seq, inplace=inplace, **opts)

    def contract(self, tags=..., inplace=False, **opts):
        """Contract some, or all, of the tensors in this network. This method
        dispatches to ``contract_structured`` or ``contract_tags``.

        Parameters
        ----------
        tags : sequence of str
            Any tensors with any of these tags with be contracted. Set to
            ``...`` (``Ellipsis``) to contract all tensors, the default.
        inplace : bool, optional
            Whether to perform the contraction inplace.
        opts
            Passed to ``tensor_contract``.

        Returns
        -------
        TensorNetwork, Tensor or scalar
            The result of the contraction, still a ``TensorNetwork`` if the
            contraction was only partial.

        See Also
        --------
        contract_structured, contract_tags, contract_cumulative
        """

        # Check for a structured strategy for performing contraction...
        if self.structure is not None:

            # but only use for total or slice tags
            if (tags is ...) or isinstance(tags, slice):
                return self.contract_structured(tags, inplace=inplace, **opts)

        # Else just contract those tensors specified by tags.
        return self.contract_tags(tags, inplace=inplace, **opts)

    def contraction_complexity(self):
        """Compute the 'contraction complexity' of this tensor network. This
        is simply defined as the maximum tensor rank produced during the
        'greedy' (so potentially only pseudo-optimal) contraction sequence.
        """
        expr = self.contract(all, get='expression')
        return max(len(c[2].split('->')[-1]) for c in expr.contraction_list)

    def __rshift__(self, tags_seq):
        """Overload of '>>' for TensorNetwork.contract_cumulative.
        """
        return self.contract_cumulative(tags_seq)

    def __irshift__(self, tags_seq):
        """Overload of '>>=' for inplace TensorNetwork.contract_cumulative.
        """
        return self.contract_cumulative(tags_seq, inplace=True)

    def __xor__(self, tags):
        """Overload of '^' for TensorNetwork.contract.
        """
        return self.contract(tags)

    def __ixor__(self, tags):
        """Overload of '^=' for inplace TensorNetwork.contract.
        """
        return self.contract(tags, inplace=True)

    def __matmul__(self, other):
        """Overload "@" to mean full contraction with another network.
        """
        return TensorNetwork((self, other)) ^ ...

    @functools.wraps(TNLinearOperator)
    def aslinearoperator(self, left_inds, right_inds,
                         ldims=None, rdims=None, backend=None):
        return TNLinearOperator(self, left_inds, right_inds,
                                ldims, rdims, backend=backend)

    def trace(self, left_inds, right_inds):
        """Trace over ``left_inds`` joined with ``right_inds``
        """
        tn = self.reindex({u: l for u, l in zip(left_inds, right_inds)})
        return tn.contract_tags(...)

    def to_dense(self, *inds_seq):
        """Convert this network into an dense array, with a single dimension
        for each of inds in ``inds_seqs``. E.g. to convert several sites
        into a density matrix: ``TN.to_dense(('k0', 'k1'), ('b0', 'b1'))``.
        """
        return (self ^ ...).to_dense(*inds_seq)

    # --------------- information about indices and dimensions -------------- #

    def _check_internal(self):
        for tid, t in self.tensor_map.items():
            for ix in t.inds:
                if tid not in self.ind_map[ix]:
                    raise ValueError("inds wrong")
            for tg in t.tags:
                if tid not in self.tag_map[tg]:
                    raise ValueError("tags wrong")

    @property
    def tags(self):
        return set(self.tag_map.keys())

    def all_inds(self):
        """Return a tuple of all indices (with repetition) in this network.
        """
        return tuple(self.ind_map)

    def inner_inds(self):
        """Tuple of all inner indices, i.e. those that appear twice.
        """
        return tuple(i for i, tids in self.ind_map.items() if len(tids) == 2)

    def outer_inds(self):
        """Tuple of exterior indices, i.e. those that appear once.
        """
        return tuple(i for i, tids in self.ind_map.items() if len(tids) == 1)

    def ind_size(self, ind):
        """Find the size of ``ind``.
        """
        tid = next(iter(self.ind_map[ind]))
        return self.tensor_map[tid].ind_size(ind)

    def ind_sizes(self):
        """Get dict of each index mapped to its size.
        """
        return {i: self.ind_size(i) for i in self.ind_map}

    def outer_dims_inds(self):
        """Get the 'outer' pairs of dimension and indices, i.e. as if this
        tensor network was fully contracted.
        """
        return tuple((self.ind_size(i), i) for i in self.outer_inds())

    def squeeze(self, fuse=False, inplace=False):
        """Drop singlet bonds and dimensions from this tensor network. If
        ``fuse=True`` also fuse all multibonds between tensors.
        """
        tn = self if inplace else self.copy()
        for t in tn:
            t.squeeze_()

        if fuse:
            tn.fuse_multibonds(inplace=True)

        return tn

    squeeze_ = functools.partialmethod(squeeze, inplace=True)

    def fuse_multibonds(self, inplace=False):
        """Fuse any multi-bonds (more than one index shared by the same pair
        of tensors) into a single bond.
        """
        tn = self if inplace else self.copy()

        for T1, T2 in itertools.combinations(tn.tensors, 2):
            dbnds = tuple(T1.bonds(T2))
            if dbnds:
                T1.fuse_({dbnds[0]: dbnds})
                T2.fuse_({dbnds[0]: dbnds})

        return tn

    def max_bond(self):
        """Return the size of the largest bond in this network.
        """
        return max(max(t.shape) for t in self)

    @property
    def shape(self):
        """Actual, i.e. exterior, shape of this TensorNetwork.
        """
        return tuple(di[0] for di in self.outer_dims_inds())

    @property
    def dtype(self):
        """The dtype of this TensorNetwork, note this just randomly samples the
        dtype of *one* tensor and thus assumes they all have the same dtype.
        """
        return next(iter(self)).dtype

    def isreal(self):
        return np.issubdtype(self.dtype, np.floating)

    def iscomplex(self):
        return np.issubdtype(self.dtype, np.complexfloating)

    def astype(self, dtype, inplace=False):
        """Convert the type of all tensors in this network to ``dtype``.
        """
        TN = self if inplace else self.copy()
        for t in TN:
            t.astype(dtype, inplace=True)
        return TN

    # ------------------------------ printing ------------------------------- #

    def graph(self, color=None, show_inds=None, show_tags=None, node_size=None,
              iterations=200, k=None, fix=None, figsize=(6, 6), legend=True,
              return_fig=False, **plot_opts):
        """Plot this tensor network as a networkx graph using matplotlib,
        with edge width corresponding to bond dimension.

        Parameters
        ----------
        color : sequence of tags, optional
            If given, uniquely color any tensors which have each of the tags.
            If some tensors have more than of the tags, only one color will
        show_inds : bool, optional
            Explicitly turn on labels for each tensors indices.
        show_tags : bool, optional
            Explicitly turn on labels for each tensors tags.
        iterations : int, optional
            How many iterations to perform when when finding the best layout
            using node repulsion. Ramp this up if the graph is drawing messily.
        k : float, optional
            The optimal distance between nodes.
        fix : dict[tags, (float, float)], optional
            Used to specify actual relative positions for each tensor node.
            Each key should be a sequence of tags that uniquely identifies a
            tensor, and each value should be a x, y coordinate tuple.
        figsize : tuple of int
            The size of the drawing.
        legend : bool, optional
            Whether to draw a legend for the colored tags.
        node_size : None
            How big to draw the tensors.
        plot_opts
            Supplied to ``networkx.draw``.
        """
        import networkx as nx
        import matplotlib.pyplot as plt
        import math

        # build the graph
        G = nx.Graph()
        ts = list(self.tensors)
        n = len(ts)

        if show_inds is None:
            show_inds = (n <= 20)
            show_tags = (n <= 20)

        if fix is None:
            fix = {}
        else:
            # find range with which to scale spectral points with
            xmin, xmax, ymin, ymax = (
                f(fix.values(), key=lambda xy: xy[i])[i]
                for f, i in [(min, 0), (max, 0), (min, 1), (max, 1)])
            if xmin == xmax:
                xmin, xmax = xmin - 1, xmax + 1
            if ymin == ymax:
                ymin, ymax = ymin - 1, ymax + 1
            xymin, xymax = min(xmin, ymin), max(xmax, ymax)

        # identify tensors by tid
        fix_tids = {}
        for tags_or_ind, pos in tuple(fix.items()):
            try:
                tid, = self._get_tids_from_tags(tags_or_ind)
                fix_tids[tid] = pos
            except KeyError:
                # assume index
                fix_tids["ext{}".format(tags_or_ind)] = pos

        labels = {}
        fixed_positions = {}

        for i, (tid, t1) in enumerate(self.tensor_map.items()):

            if tid in fix_tids:
                fixed_positions[i] = fix_tids[tid]

            if not t1.inds:
                # is a scalar
                G.add_node(i)
                continue

            for ix in t1.inds:
                found_ind = False

                # check to see if index is linked to another tensor
                for j in range(0, n):
                    if j == i:
                        continue

                    t2 = ts[j]
                    if ix in t2.inds:
                        found_ind = True
                        G.add_edge(i, j, weight=t1.shared_bond_size(t2))

                # else it must be an 'external' index
                if not found_ind:
                    ext_lbl = "ext{}".format(ix)
                    G.add_edge(i, ext_lbl, weight=t1.ind_size(ix))

                    # optionally label the external index
                    if show_inds:
                        labels[ext_lbl] = ix

                    if ext_lbl in fix_tids:
                        fixed_positions[ext_lbl] = fix_tids[ext_lbl]

        edge_weights = [x[2]['weight'] for x in G.edges(data=True)]

        # color the nodes
        if color is None:
            colors = {}
        elif isinstance(color, str):
            colors = {color: plt.get_cmap('tab10').colors[0]}
        else:
            # choose longest nice seq of colors
            if len(color) > 10:
                rgbs = plt.get_cmap('tab20').colors
            else:
                rgbs = plt.get_cmap('tab10').colors
            # extend
            extras = [plt.get_cmap(i).colors
                      for i in ('Dark2', 'Set2', 'Set3', 'Accent', 'Set1')]
            # but also resort to random if too long
            rand_colors = (tuple(np.random.rand(3)) for _ in range(9999999999))
            rgbs = concat((rgbs, *extras, rand_colors))
            # ordered to reliably overcolor tags
            colors = collections.OrderedDict(zip(color, rgbs))

        for i, t1 in enumerate(ts):
            G.node[i]['color'] = None
            for col_tag in colors:
                if col_tag in t1.tags:
                    G.node[i]['color'] = colors[col_tag]
            # optionally label the tensor's tags
            if show_tags:
                labels[i] = str(t1.tags)

        # Set the size of the nodes, so that dangling inds appear so.
        # Also set the colors of any tagged tensors.
        if node_size is None:
            node_size = 1000 / n**0.7

        szs = []
        crs = []
        for nd in G.nodes():
            if isinstance(nd, str):
                szs += [0]
                crs += [(1.0, 1.0, 1.0)]
            else:
                szs += [node_size]
                if G.node[nd]['color'] is not None:
                    crs += [G.node[nd]['color']]
                else:
                    crs += [(0.6, 0.6, 0.6)]

        edge_weights = [math.log2(d) for d in edge_weights]

        fig = plt.figure(figsize=figsize)

        # use spectral layout as starting point
        pos0 = nx.spectral_layout(G)
        # scale points to fit with specified positions
        if fix:
            # but update with fixed positions
            pos0.update(valmap(lambda xy: np.array(
                (2 * (xy[0] - xymin) / (xymax - xymin) - 1,
                 2 * (xy[1] - xymin) / (xymax - xymin) - 1)), fixed_positions))
            fixed = fixed_positions.keys()
        else:
            fixed = None

        # and then relax remaining using spring layout
        pos = nx.spring_layout(G, pos=pos0, fixed=fixed,
                               k=k, iterations=iterations)

        nx.draw(G, node_size=szs, node_color=crs, pos=pos, labels=labels,
                with_labels=True, width=edge_weights, **plot_opts)
        plt.gca().set_aspect('equal')

        # create legend
        if colors and legend:
            handles = []
            for color in colors.values():
                handles += [plt.Line2D([0], [0], marker='o', color=color,
                                       linestyle='', markersize=10)]

            # needed in case '_' is the first character
            lbls = [" {}".format(l) for l in colors]

            plt.legend(handles, lbls, ncol=max(int(len(handles) / 20), 1),
                       loc='center left', bbox_to_anchor=(1, 0.5))

        if return_fig:
            return fig
        else:
            plt.show()

    def __getstate__(self):
        # This allows pickling, by removing all tensor owner weakrefs
        d = self.__dict__.copy()
        d['tensor_map'] = {
            k: t.copy() for k, t in d['tensor_map'].items()
        }
        return d

    def __setstate__(self, state):
        # This allows picklings, by restoring the returned TN as owner
        self.__dict__ = state.copy()
        for t in self.__dict__['tensor_map'].values():
            t.add_owner(self, tid=rand_uuid(base="_T"))

    def __str__(self):
        return "{}([{}{}{}]{}{})".format(
            self.__class__.__name__,
            os.linesep,
            "".join(["    " + repr(t) + "," + os.linesep
                     for t in self.tensors[:-1]]),
            "    " + repr(self.tensors[-1]) + "," + os.linesep,
            ", structure='{}'".format(self.structure) if
            self.structure is not None else "",
            ", nsites={}".format(self.nsites) if
            self.nsites is not None else "")

    def __repr__(self):
        rep = "<{}(tensors={}".format(self.__class__.__name__,
                                      len(self.tensor_map))
        if self.structure:
            rep += ", structure='{}', nsites={}".format(self.structure,
                                                        self.nsites)

        return rep + ")>"


class TNLinearOperator1D(spla.LinearOperator):
    r"""A 1D tensor network linear operator like::

                 start                 stop - 1
                   .                     .
                 :-O-O-O-O-O-O-O-O-O-O-O-O-:                 --+
                 : | | | | | | | | | | | | :                   |
                 :-H-H-H-H-H-H-H-H-H-H-H-H-:    acting on    --V
                 : | | | | | | | | | | | | :                   |
                 :-O-O-O-O-O-O-O-O-O-O-O-O-:                 --+
        left_inds^                         ^right_inds

    Like :class:`~quimb.tensor.tensor_core.TNLinearOperator`, but performs a
    structured contract from one end to the other than can handle very long
    chains possibly more efficiently by contracting in blocks from one end.


    Parameters
    ----------
    tn : TensorNetwork
        The tensor network to turn into a ``LinearOperator``.
    left_inds : sequence of str
        The left indicies.
    right_inds : sequence of str
        The right indicies.
    start : int
        Index of starting site.
    stop : int
        Index of stopping site (does not include this site).
    ldims : tuple of int, optional
        If known, the dimensions corresponding to ``left_inds``.
    rdims : tuple of int, optional
        If known, the dimensions corresponding to ``right_inds``.

    See Also
    --------
    TNLinearOperator
    """

    def __init__(self, tn, left_inds, right_inds, start, stop,
                 ldims=None, rdims=None, is_conj=False, is_trans=False):
        self.tn = tn
        self.start, self.stop = start, stop

        if ldims is None or rdims is None:
            ind_sizes = tn.ind_sizes()
            ldims = tuple(ind_sizes[i] for i in left_inds)
            rdims = tuple(ind_sizes[i] for i in right_inds)

        self.left_inds, self.right_inds = left_inds, right_inds
        self.ldims, ld = ldims, prod(ldims)
        self.rdims, rd = rdims, prod(rdims)

        # conjugate inputs/ouputs rather all tensors if necessary
        self.is_conj = is_conj
        self.is_trans = is_trans
        self._conj_linop = None
        self._adjoint_linop = None
        self._transpose_linop = None

        super().__init__(dtype=self.tn.dtype, shape=(ld, rd))

    def _matvec(self, vec):
        in_data = vec.reshape(*self.rdims)

        if self.is_conj:
            in_data = in_data.conj()

        if self.is_trans:
            i, f, s = self.start, self.stop, 1
        else:
            i, f, s = self.stop - 1, self.start - 1, -1

        # add the vector to the right of the chain
        tnc = self.tn | Tensor(in_data, self.right_inds, tags=['_VEC'])

        # absorb it into the rightmost site
        tnc ^= ['_VEC', self.tn.site_tag(i)]

        # then do a structured contract along the whole chain
        out_T = tnc ^ slice(i, f, s)

        out_data = out_T.transpose_(*self.left_inds).data.ravel()
        if self.is_conj:
            out_data = out_data.conj()

        return out_data

    def _matmat(self, mat):
        d = mat.shape[-1]
        in_data = mat.reshape(*self.rdims, d)

        if self.is_conj:
            in_data = in_data.conj()

        if self.is_trans:
            i, f, s = self.start, self.stop, 1
        else:
            i, f, s = self.stop - 1, self.start - 1, -1

        # add the vector to the right of the chain
        in_ix = (*self.right_inds, '_mat_ix')
        tnc = self.tn | Tensor(in_data, inds=in_ix, tags=['_VEC'])

        # absorb it into the rightmost site
        tnc ^= ['_VEC', self.tn.site_tag(i)]

        # then do a structured contract along the whole chain
        out_T = tnc ^ slice(i, f, s)

        out_ix = (*self.left_inds, '_mat_ix')
        out_data = out_T.transpose_(*out_ix).data.reshape(-1, d)
        if self.is_conj:
            out_data = out_data.conj()

        return out_data

    def copy(self, conj=False, transpose=False):

        if transpose:
            inds = (self.right_inds, self.left_inds)
            dims = (self.rdims, self.ldims)
            is_trans = not self.is_trans
        else:
            inds = (self.left_inds, self.right_inds)
            dims = (self.ldims, self.rdims)
            is_trans = self.is_trans

        if conj:
            is_conj = not self.is_conj
        else:
            is_conj = self.is_conj

        return TNLinearOperator1D(self.tn, *inds, self.start, self.stop, *dims,
                                  is_conj=is_conj, is_trans=is_trans)

    def conj(self):
        if self._conj_linop is None:
            self._conj_linop = self.copy(conj=True)
        return self._conj_linop

    def _transpose(self):
        if self._transpose_linop is None:
            self._transpose_linop = self.copy(transpose=True)
        return self._transpose_linop

    def _adjoint(self):
        """Hermitian conjugate of this TNLO.
        """
        # cache the adjoint
        if self._adjoint_linop is None:
            self._adjoint_linop = self.copy(conj=True, transpose=True)
        return self._adjoint_linop

    def to_dense(self):
        T = self.tn ^ slice(self.start, self.stop)

        if self.is_conj:
            T = T.conj()

        return T.to_dense(self.left_inds, self.right_inds)

    @property
    def A(self):
        return self.to_dense()
