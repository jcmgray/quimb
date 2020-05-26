"""Core tensor network tools.
"""
import os
import re
import copy
import uuid
import math
import string
import weakref
import operator
import functools
import itertools
import contextlib
import collections
from numbers import Integral

import numpy as np
import opt_einsum as oe
import scipy.sparse.linalg as spla
from cytoolz import unique, concat, frequencies, partition_all, merge_with
from autoray import (do, conj, reshape, transpose, astype,
                     infer_backend, get_dtype_name)

from ..core import qarray, prod, realify_scalar, vdot, common_type
from ..utils import check_opt, functions_equal
from ..gen.rand import randn, seed_rand
from . import decomp
from .array_ops import (iscomplex, norm_fro, unitize, ndim, asarray, PArray,
                        find_diag_axes, find_antidiag_axes, find_columns)
from .graphing import graph

_DEFAULT_CONTRACTION_STRATEGY = 'greedy'


def cost_rank(s12, s1, s2, k12, k1, k2):
    rank_new = len(k12)
    rank_old = max(len(k1), len(k2))
    # second entry is to break ties
    return rank_new - rank_old, rank_old


greedy_rank = functools.partial(oe.paths.greedy, cost_fn=cost_rank)
oe.paths.register_path_fn('greedy-rank', greedy_rank)


def get_contract_strategy():
    r"""Get the default contraction strategy - the option supplied as
    ``optimize`` to ``opt_einsum``.
    """
    return _DEFAULT_CONTRACTION_STRATEGY


def set_contract_strategy(strategy):
    """Get the default contraction strategy - the option supplied as
    ``optimize`` to ``opt_einsum``.
    """
    global _DEFAULT_CONTRACTION_STRATEGY
    _DEFAULT_CONTRACTION_STRATEGY = strategy


@contextlib.contextmanager
def contract_strategy(strategy):
    """A context manager to temporarily set the default contraction strategy
    supplied as ``optimize`` to ``opt_einsum``.
    """
    orig_strategy = get_contract_strategy()
    try:
        yield set_contract_strategy(strategy)
    finally:
        set_contract_strategy(orig_strategy)


def _get_contract_path(eq, *shapes, **kwargs):
    """Get the contraction path - sequence of integer pairs.
    """
    return tuple(oe.contract_path(eq, *shapes, shapes=True, **kwargs)[0])


def _get_contract_expr(eq, *shapes, **kwargs):
    """Get the contraction expression - callable taking raw arrays.
    """
    return oe.contract_expression(eq, *shapes, **kwargs)


def _get_contract_info(eq, *shapes, **kwargs):
    """Get the contraction ipath info - object containing various information.
    """
    return oe.contract_path(eq, *shapes, shapes=True, **kwargs)[1]


_CONTRACT_PATH_CACHE = None


_CONTRACT_FNS = {
    # key: (get, cache)
    ('path', False): _get_contract_path,
    ('path', True): functools.lru_cache(2**12)(_get_contract_path),
    ('expr', False): _get_contract_expr,
    ('expr', True): functools.lru_cache(2**12)(_get_contract_expr),
    ('info', False): _get_contract_info,
    ('info', True): functools.lru_cache(2**12)(_get_contract_info),
}


def set_contract_path_cache(
    directory=None,
    in_mem_cache_size=2**12,
):
    """Specify an directory to cache all contraction paths to, if a directory
    is specified ``diskcache`` (https://pypi.org/project/diskcache/) will be
    used to write all contraction expressions / paths to.

    Parameters
    ----------
    directory : None or path, optimize
        If None (the default), don't use any disk caching. If a path, supply it
        to ``diskcache.Cache`` to use as the persistent store.
    in_mem_cache_size_expr : int, optional
        The size of the in memory cache to use for contraction expressions.
    in_mem_cache_size_path : int, optional
        The size of the in memory cache to use for contraction paths.
    """
    global _CONTRACT_PATH_CACHE

    if _CONTRACT_PATH_CACHE is not None:
        _CONTRACT_PATH_CACHE.close()

    if directory is None:
        _CONTRACT_PATH_CACHE = None
        path_fn = _get_contract_path
    else:
        # for size reasons we only cache actual path to disk
        import diskcache
        _CONTRACT_PATH_CACHE = diskcache.Cache(directory)
        path_fn = _CONTRACT_PATH_CACHE.memoize()(_get_contract_path)

    # second layer of in memory caching applies to all functions
    _CONTRACT_FNS['path', True] = (
        functools.lru_cache(in_mem_cache_size)(path_fn))
    _CONTRACT_FNS['expr', True] = (
        functools.lru_cache(in_mem_cache_size)(_get_contract_expr))
    _CONTRACT_FNS['info', True] = (
        functools.lru_cache(in_mem_cache_size)(_get_contract_info))


def get_contraction(eq, *shapes, cache=True, get='expr', **kwargs):
    """Get an callable expression that will evaluate ``eq`` based on
    ``shapes``. Cache the result if no constant tensors are involved.
    """
    optimize = kwargs.pop('optimize', _DEFAULT_CONTRACTION_STRATEGY)

    # can't cache if using constants
    if 'constants' in kwargs:
        expr_fn = _CONTRACT_FNS['expr', False]
        expr = expr_fn(eq, *shapes, optimize=optimize, **kwargs)
        return expr

    # make sure explicit paths are hashable
    if isinstance(optimize, list):
        optimize = tuple(optimize)

    # can't cache if using a reusable path-optimizer
    cache &= not isinstance(optimize, oe.paths.PathOptimizer)

    # make sure shapes are hashable (e.g. for tensorflow arrays)
    if cache:
        try:
            hash(shapes[0])
        except TypeError:
            shapes = tuple(tuple(map(int, s)) for s in shapes)

    # get the path, unless explicitly given already
    if not isinstance(optimize, tuple):
        path_fn = _CONTRACT_FNS['path', cache]
        path = path_fn(eq, *shapes, optimize=optimize, **kwargs)
    else:
        path = optimize

    if get == 'expr':
        expr_fn = _CONTRACT_FNS['expr', cache]
        expr = expr_fn(eq, *shapes, optimize=path, **kwargs)
        return expr

    if get == 'info':
        info_fn = _CONTRACT_FNS['info', cache]
        info = info_fn(eq, *shapes, optimize=path, **kwargs)
        return info

    if get == 'path':
        return path


try:
    from opt_einsum.contract import infer_backend as _oe_infer_backend
    del _oe_infer_backend
    _CONTRACT_BACKEND = 'auto'
    _TENSOR_LINOP_BACKEND = 'auto'
except ImportError:
    _CONTRACT_BACKEND = 'numpy'
    _TENSOR_LINOP_BACKEND = 'numpy'


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


@contextlib.contextmanager
def contract_backend(backend):
    """A context manager to temporarily set the default backend used for tensor
    contractions, via 'opt_einsum'.
    """
    orig_backend = get_contract_backend()
    try:
        yield set_contract_backend(backend)
    finally:
        set_contract_backend(orig_backend)


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


@contextlib.contextmanager
def tensor_linop_backend(backend):
    """A context manager to temporarily set the default backend used for tensor
    network linear operators, via 'opt_einsum'.
    """
    orig_backend = get_tensor_linop_backend()
    try:
        yield set_tensor_linop_backend(backend)
    finally:
        set_tensor_linop_backend(orig_backend)


# --------------------------------------------------------------------------- #
#                                Tensor Funcs                                 #
# --------------------------------------------------------------------------- #


def utup(x):
    """Construct a tuple of the unique elements of ``x``.
    """
    return tuple(unique(x))


def empty_utup():
    """The empty 'unique tuple'.
    """
    return tuple()


def utup_add(xs, y):
    """Add element ``y`` to tuple ``xs`` if it doesn't already contain it,
    returning a new tuple.
    """
    if y in xs:
        return xs
    return xs + (y,)


def utup_discard(xs, y):
    """If element ``y`` is found in ``xs`` remove it, returning a new tuple.
    """
    try:
        i = xs.index(y)
        return xs[:i] + xs[i + 1:]
    except ValueError:
        return xs


def utup_union(xs):
    """Unique tuple, non variadic version of set.union.
    """
    return tuple(unique(concat(xs)))


def utup_intersection(xs):
    """Unique tuple, non variadic version of set.intersection.
    """
    x0, *x1s = xs
    return tuple(el for el in x0 if all(el in x for x in x1s))


def utup_difference(x, y):
    """Return a tuple with any elements in ``y`` removed from ``x``.
    """
    return tuple(el for el in x if el not in y)


def tags_to_utup(tags):
    """Parse a ``tags`` argument into a unique tuple.
    """
    if tags is None:
        return ()
    elif isinstance(tags, str):
        return (tags,)
    else:
        return utup(tags)


def _gen_output_inds(all_inds):
    """Generate the output, i.e. unique, indices from the set ``inds``. Raise
    if any index found more than twice.
    """
    cnts = collections.Counter(all_inds)
    for ind, freq in cnts.items():
        if freq > 2:
            raise ValueError(
                f"The index {ind} appears more than twice! If this is "
                "intentionally a 'hyper' tensor network you will need to "
                "explicitly supply `output_inds` when contracting for example."
            )
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
    eq : str
        The string to feed to einsum/contract.
    """
    amap = {ix: oe.get_symbol(i) for i, ix in enumerate(a_ix)}
    in_str = ("".join(amap[i] for i in ix) for ix in i_ix)
    out_str = "".join(amap[o] for o in o_ix)

    return ",".join(in_str) + "->" + out_str


_VALID_CONTRACT_GET = {None, 'expression', 'path-info', 'symbol-map'}


def tensor_contract(*tensors, output_inds=None, get=None,
                    backend=None, **contract_opts):
    """Efficiently contract multiple tensors, combining their tags.

    Parameters
    ----------
    tensors : sequence of Tensor
        The tensors to contract.
    output_inds : sequence of str
        If given, the desired order of output indices, else defaults to the
        order they occur in the input indices.
    get : {None, 'expression', 'path-info', 'opt_einsum'}, optional
        What to return. If:

            * ``None`` (the default) - return the resulting scalar or Tensor.
            * ``'expression'`` - return the ``opt_einsum`` expression that
              performs the contraction and operates on the raw arrays.
            * ``'symbol-map'`` - return the dict mapping ``opt_einsum`` symbols
              to tensor indices.
            * ``'path-info'`` - return the full ``opt_einsum`` path object with
              detailed information such as flop cost. The symbol-map is also
              added to the ``quimb_symbol_map`` attribute.

    backend : {'numpy', 'cupy', 'tensorflow', 'theano', 'dask', ...}, optional
        Which backend to use to perform the contraction. Must be a valid
        ``opt_einsum`` backend with the relevant library installed.
    contract_opts
        Passed to ``opt_einsum.contract_expression`` or
        ``opt_einsum.contract_path``.

    Returns
    -------
    scalar or Tensor
    """
    check_opt('get', get, _VALID_CONTRACT_GET)

    if backend is None:
        backend = _CONTRACT_BACKEND

    i_ix = tuple(t.inds for t in tensors)  # input indices per tensor
    total_ix = tuple(concat(i_ix))  # list of all input indices
    all_ix = utup(total_ix)

    if output_inds is None:
        # sort output indices by input order for efficiency and consistency
        o_ix = tuple(_gen_output_inds(total_ix))
    else:
        o_ix = tuple(output_inds)

    # possibly map indices into the range needed by opt-einsum
    eq = _maybe_map_indices_to_alphabet(all_ix, i_ix, o_ix)

    if get == 'symbol-map':
        return {oe.get_symbol(i): ix for i, ix in enumerate(all_ix)}

    if get == 'path-info':
        ops = (t.shape for t in tensors)
        path_info = get_contraction(eq, *ops, get='info', **contract_opts)
        path_info.quimb_symbol_map = {
            oe.get_symbol(i): ix for i, ix in enumerate(all_ix)
        }
        return path_info

    if get == 'expression':
        # account for possible constant tensors
        cnst = contract_opts.get('constants', ())
        ops = (t.data if i in cnst else t.shape for i, t in enumerate(tensors))
        expression = get_contraction(eq, *ops, **contract_opts)
        return expression

    # perform the contraction
    shapes = (t.shape for t in tensors)
    expression = get_contraction(eq, *shapes, **contract_opts)
    o_array = expression(*(t.data for t in tensors), backend=backend)

    if not o_ix:
        if isinstance(o_array, np.ndarray):
            o_array = realify_scalar(o_array.item(0))
        return o_array

    # union of all tags
    o_tags = utup_union(t.tags for t in tensors)

    return Tensor(data=o_array, inds=o_ix, tags=o_tags)


# generate a random base to avoid collisions on difference processes ...
_RAND_PREFIX = str(uuid.uuid4())[:6]
# but then make the list orderable to help contraction caching
_RAND_ALPHABET = string.ascii_uppercase + string.ascii_lowercase
RAND_UUIDS = map("".join, itertools.product(_RAND_ALPHABET, repeat=5))


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
    return base + "_" + _RAND_PREFIX + next(RAND_UUIDS)


_VALID_SPLIT_GET = {None, 'arrays', 'tensors', 'values'}
_FULL_SPLIT_METHODS = {'svd', 'eig', 'eigh'}
_RANK_HIDDEN_METHODS = {'qr', 'lq', 'cholesky'}
_DENSE_ONLY_METHODS = {'svd', 'eig', 'eigh', 'cholesky', 'qr', 'lq'}
_CUTOFF_LOOKUP = {None: -1.0}
_ABSORB_LOOKUP = {'left': -1, 'both': 0, 'right': 1, None: None}
_MAX_BOND_LOOKUP = {None: -1}
_CUTOFF_MODES = {'abs': 1, 'rel': 2, 'sum2': 3,
                 'rsum2': 4, 'sum1': 5, 'rsum1': 6}
_RENORM_LOOKUP = {'sum2': 2, 'rsum2': 2, 'sum1': 1, 'rsum1': 1}


def _parse_split_opts(method, cutoff, absorb, max_bond, cutoff_mode, renorm):
    opts = dict()

    if method in _RANK_HIDDEN_METHODS:
        if absorb is None:
            raise ValueError(
                "You can't return the singular values separately when "
                "`method='{}'`.".format(method))

        # options are only relevant for handling singular values
        return opts

    # convert defaults and settings to numeric type for numba funcs
    opts['cutoff'] = _CUTOFF_LOOKUP.get(cutoff, cutoff)
    opts['absorb'] = _ABSORB_LOOKUP[absorb]
    opts['max_bond'] = _MAX_BOND_LOOKUP.get(max_bond, max_bond)
    opts['cutoff_mode'] = _CUTOFF_MODES[cutoff_mode]

    # renorm doubles up as the power used to renormalize
    if (method in _FULL_SPLIT_METHODS) and (renorm is None):
        opts['renorm'] = _RENORM_LOOKUP.get(cutoff_mode, 0)
    else:
        opts['renorm'] = 0 if renorm is None else int(renorm)

    return opts


def tensor_split(
    T,
    left_inds,
    method='svd',
    get=None,
    absorb='both',
    max_bond=None,
    cutoff=1e-10,
    cutoff_mode='rel',
    renorm=None,
    ltags=None,
    rtags=None,
    stags=None,
    bond_ind=None,
    right_inds=None,
):
    """Decompose this tensor into two tensors.

    Parameters
    ----------
    T : Tensor or TNLinearOperator
        The tensor (network) to split.
    left_inds : str or sequence of str
        The index or sequence of inds, which ``T`` should already have, to
        split to the 'left'. You can supply ``None`` here if you supply
        ``right_inds`` instead.
    method : str, optional
        How to split the tensor, only some methods allow bond truncation:

            - ``'svd'``: full SVD, allows truncation.
            - ``'eig'``: full SVD via eigendecomp, allows truncation.
            - ``'svds'``: iterative svd, allows truncation.
            - ``'isvd'``: iterative svd using interpolative methods, allows
              truncation.
            - ``'rsvd'`` : randomized iterative svd with truncation.
            - ``'eigh'``: full eigen-decomposition, tensor must he hermitian.
            - ``'eigsh'``: iterative eigen-decomposition, tensor must be
              hermitian.
            - ``'qr'``: full QR decomposition.
            - ``'lq'``: full LR decomposition.
            - ``'cholesky'``: full cholesky decomposition, tensor must be
              positive.

    get : {None, 'arrays', 'tensors', 'values'}
        If given, what to return instead of a TN describing the split:

            - ``None``: a tensor network of the two (or three) tensors.
            - ``'arrays'``: the raw data arrays as a tuple ``(l, r)`` or
              ``(l, s, r)`` depending on ``absorb``.
            - ``'tensors '``: the new tensors as a tuple ``(Tl, Tr)`` or
              ``(Tl, Ts, Tr)`` depending on ``absorb``.
            - ``'values'``: only compute and return the singular values ``s``.

    absorb : {'both', 'left', 'right', None}, optional
        Whether to absorb the singular values into both, the left, or the right
        unitary matrix respectively, or neither. If neither (``absorb=None``)
        then the singular values will be returned separately in their own
        1D tensor or array. In that case if ``get=None`` the tensor network
        returned will have a hyperedge corresponding to the new bond index
        connecting three tensors. If ``get='tensors'`` or ``get='arrays'`` then
        a tuple like ``(left, s, right)`` is returned.
    max_bond : None or int
        If integer, the maxmimum number of singular values to keep, regardless
        of ``cutoff``.
    cutoff : float, optional
        The threshold below which to discard singular values, only applies to
        rank revealing methods (not QR, LQ, or cholesky).
    cutoff_mode : {'sum2', 'rel', 'abs', 'rsum2'}
        Method with which to apply the cutoff threshold:

            - ``'rel'``: values less than ``cutoff * s[0]`` discarded.
            - ``'abs'``: values less than ``cutoff`` discarded.
            - ``'sum2'``: sum squared of values discarded must be ``< cutoff``.
            - ``'rsum2'``: sum squared of values discarded must be less than
              ``cutoff`` times the total sum of squared values.
            - ``'sum1'``: sum values discarded must be ``< cutoff``.
            - ``'rsum1'``: sum of values discarded must be less than
              ``cutoff`` times the total sum of values.

    renorm : {None, bool, or int}, optional
        Whether to renormalize the kept singular values, assuming the bond has
        a canonical environment, corresponding to maintaining the Frobenius
        norm or trace. If ``None`` (the default) then this is automatically
        turned on only for ``cutoff_method in {'sum2', 'rsum2', 'sum1',
        'rsum1'}`` with ``method in {'svd', 'eig', 'eigh'}``.
    ltags : sequence of str, optional
        Add these new tags to the left tensor.
    rtags : sequence of str, optional
        Add these new tags to the right tensor.
    stags : sequence of str, optional
        Add these new tags to the singular value tensor.
    bond_ind : str, optional
        Explicitly name the new bond, else a random one will be generated.
    right_inds : sequence of str, optional
        Explicitly give the right indices, otherwise they will be worked out.
        This is a minor performance feature.

    Returns
    -------
    TensorNetwork or tuple[Tensor] or tuple[array] or 1D-array
        Depending on if ``get`` is ``None``, ``'tensors'``, ``'arrays'``, or
        ``'values'``. In the first three cases, if ``absorb`` is set, then the
        returned objects correspond to ``(left, right)`` whereas if
        ``absorb=None`` the returned objects correspond to
        ``(left, singular_values, right)``.
    """
    check_opt('get', get, _VALID_SPLIT_GET)

    if left_inds is None:
        left_inds = utup_difference(T.inds, right_inds)
    else:
        left_inds = tags_to_utup(left_inds)

    if right_inds is None:
        right_inds = utup_difference(T.inds, left_inds)

    if isinstance(T, spla.LinearOperator):
        left_dims = T.ldims
        right_dims = T.rdims

        if method in _DENSE_ONLY_METHODS:
            array = T.to_dense()
        else:
            array = T
    else:
        TT = T.transpose(*left_inds, *right_inds)

        left_dims = TT.shape[:len(left_inds)]
        right_dims = TT.shape[len(left_inds):]

        array = reshape(TT.data, (prod(left_dims), prod(right_dims)))

    if get == 'values':
        return {'svd': decomp._svdvals,
                'eig': decomp._svdvals_eig}[method](array)

    opts = _parse_split_opts(
        method, cutoff, absorb, max_bond, cutoff_mode, renorm)

    split_fn = {
        'svd': decomp._svd,
        'eig': decomp._eig,
        'qr': decomp._qr,
        'lq': decomp._lq,
        'eigh': decomp._eigh,
        'cholesky': decomp._cholesky,
        'isvd': decomp._isvd,
        'svds': decomp._svds,
        'rsvd': decomp._rsvd,
        'eigsh': decomp._eigsh,
    }[method]

    # ``s`` itself will be None unless ``absorb=None`` is specified
    left, s, right = split_fn(array, **opts)
    left = reshape(left, (*left_dims, -1))
    right = reshape(right, (-1, *right_dims))

    if get == 'arrays':
        if absorb is None:
            return left, s, right
        return left, right

    bond_ind = rand_uuid() if bond_ind is None else bond_ind
    ltags = utup_union((tags_to_utup(ltags), T.tags))
    rtags = utup_union((tags_to_utup(rtags), T.tags))

    Tl = Tensor(data=left, inds=(*left_inds, bond_ind), tags=ltags)
    Tr = Tensor(data=right, inds=(bond_ind, *right_inds), tags=rtags)

    if absorb is None:
        stags = utup_union((tags_to_utup(stags), T.tags))
        Ts = Tensor(data=s, inds=(bond_ind,), tags=stags)
        tensors = (Tl, Ts, Tr)
    else:
        tensors = (Tl, Tr)

    if get == 'tensors':
        return tensors

    return TensorNetwork(tensors, check_collisions=False)


def tensor_canonize_bond(T1, T2, **split_opts):
    r"""Inplace 'canonization' of two tensors. This gauges the bond between
    the two such that ``T1`` is isometric::

          |   |          |   |          |   |
        --1---2--  =>  -->~R-2--  =>  -->~~~O--
          |   |          |   |          |   |
          .                ...
         <QR>              contract

    Parameters
    ----------
    T1 : Tensor
        The tensor to be isometrized.
    T2 : Tensor
        The tensor to absorb the R-factor into.
    split_opts
        Supplied to :func:`~quimb.tensor.tensor_core.tensor_split`, with
        modified defaults of ``method=='qr'`` and ``absorb='right'``.
    """
    split_opts.setdefault('method', 'qr')
    split_opts.setdefault('absorb', 'right')

    shared_ix, left_env_ix = T1.filter_bonds(T2)
    if not shared_ix:
        raise ValueError("The tensors specified don't share an bond.")
    elif len(shared_ix) > 1:
        # fuse multibonds
        T1.fuse_({shared_ix[0]: shared_ix})
        T2.fuse_({shared_ix[0]: shared_ix})

    new_T1, tRfact = T1.split(left_env_ix, get='tensors', **split_opts)
    new_T2 = T2.contract(tRfact)

    new_T1.transpose_like_(T1)
    new_T2.transpose_like_(T2)

    T1.modify(data=new_T1.data)
    T2.modify(data=new_T2.data)


def tensor_compress_bond(T1, T2, absorb='both', **compress_opts):
    r"""Inplace compress between the two single tensors. It follows the
    following steps to minimize the size of SVD performed::

        a)|   |        b)|            |        c)|       |
        --1---2--  ->  --1L~~1R--2L~~2R--  ->  --1L~~M~~2R--
          |   |          |   ......   |          |       |
         <*> <*>              >  <                  <*>
         QR   LQ                                    SVD

                  d)|            |        e)|     |
              ->  --1L~~ML~~MR~~2R--  ->  --1C~~~2C--
                    |....    ....|          |     |
                     >  <    >  <              ^compressed bond
    """
    shared_ix, left_env_ix = T1.filter_bonds(T2)
    if not shared_ix:
        raise ValueError("The tensors specified don't share an bond.")
    elif len(shared_ix) > 1:
        # fuse multibonds
        T1.fuse_({shared_ix[0]: shared_ix})
        T2.fuse_({shared_ix[0]: shared_ix})
        shared_ix = (shared_ix[0],)

    # a) -> b)
    T1_L, T1_R = T1.split(left_inds=left_env_ix, right_inds=shared_ix,
                          get='tensors', method='qr')
    T2_L, T2_R = T2.split(left_inds=shared_ix, get='tensors', method='lq')
    # b) -> c)
    M = (T1_R @ T2_L)
    M.drop_tags()
    # c) -> d)
    M_L, M_R = M.split(left_inds=T1_L.bonds(M), get='tensors',
                       absorb=absorb, **compress_opts)

    # make sure old bond being used
    ns_ix, = M_L.bonds(M_R)
    M_L.reindex_({ns_ix: shared_ix[0]})
    M_R.reindex_({ns_ix: shared_ix[0]})

    # d) -> e)
    T1C = T1_L.contract(M_L, output_inds=T1.inds)
    T2C = M_R.contract(T2_R, output_inds=T2.inds)

    # update with the new compressed data
    T1.modify(data=T1C.data)
    T2.modify(data=T2C.data)


def tensor_balance_bond(t1, t2, smudge=1e-6):
    """Gauge the bond between two tensors such that the norm of the 'columns'
    of the tensors on each side is the same for each index of the bond.

    Parameters
    ----------
    t1 : Tensor
        The first tensor, should share a single index with ``t2``.
    t2 : Tensor
        The second tensor, should share a single index with ``t1``.
    smudge : float, optional
        Avoid numerical issues by 'smudging' the correctional factor by this
        much - the gauging introduced is still exact.
    """
    ix, = bonds(t1, t2)
    x = tensor_contract(t1.H, t1, output_inds=[ix]).data
    y = tensor_contract(t2.H, t2, output_inds=[ix]).data
    s = (x + smudge) / (y + smudge)
    t1.multiply_index_diagonal_(ix, s**-0.25)
    t2.multiply_index_diagonal_(ix, s**+0.25)


def new_bond(T1, T2, size=1, name=None, axis1=0, axis2=0):
    """Inplace addition of a new bond between tensors ``T1`` and ``T2``. The
    size of the new bond can be specified, in which case the new array parts
    will be filled with zeros.

    Parameters
    ----------
    T1 : Tensor
        First tensor to modify.
    T2 : Tensor
        Second tensor to modify.
    size : int, optional
        Size of the new dimension.
    name : str, optional
        Name for the new index.
    axis1 : int, optional
        Position on the first tensor for the new dimension.
    axis2 : int, optional
        Position on the second tensor for the new dimension.
    """
    if name is None:
        name = rand_uuid()

    T1.new_ind(name, size=size, axis=axis1)
    T2.new_ind(name, size=size, axis=axis2)


def array_direct_product(X, Y, sum_axes=()):
    """Direct product of two arrays.

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

    padX = []
    padY = []
    for i, (d1, d2) in enumerate(zip(X.shape, Y.shape)):
        if i not in sum_axes:
            padX.append((0, d2))
            padY.append((d1, 0))
        else:
            if d1 != d2:
                raise ValueError("Can only add sum tensor "
                                 "indices of the same size.")
            padX.append((0, 0))
            padY.append((0, 0))

    pX = do('pad', X, padX, mode='constant')
    pY = do('pad', Y, padY, mode='constant')

    return pX + pY


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
        ix1 = t1.inds
    else:
        ix1 = utup_union(t.inds for t in t1)

    if isinstance(t2, Tensor):
        ix2 = t2.inds
    else:
        ix2 = utup_union(t.inds for t in t2)

    return utup_intersection((ix1, ix2))


def bonds_size(t1, t2):
    """Get the size of the bonds linking tensors or tensor networks ``t1`` and
    ``t2``.
    """
    return prod(t1.ind_size(ix) for ix in bonds(t1, t2))


def connect(t1, t2, ax1, ax2):
    """Connect two tensors by setting a shared index for the specified
    dimensions. This is an inplace operation that will also affect any tensor
    networks viewing these tensors.

    Parameters
    ----------
    t1 : Tensor
        The first tensor.
    t2 :
        The second tensor.
    ax1 : int
        The dimension (axis) to connect on the first tensor.
    ax2 : int
        The dimension (axis) to connect on the second tensor.

    Examples
    --------

        >>> X = rand_tensor([2, 3], inds=['a', 'b'])
        >>> Y = rand_tensor([3, 4], inds=['c', 'd'])

        >>> tn = (X | Y)  # is *view* of tensors (``&`` would copy them)
        >>> print(tn)
        TensorNetwork([
            Tensor(shape=(2, 3), inds=('a', 'b'), tags=()),
            Tensor(shape=(3, 4), inds=('c', 'd'), tags=()),
        ])

        >>> connect(X, Y, 1, 0)  # modifies tensors *and* viewing TN
        >>> print(tn)
        TensorNetwork([
            Tensor(shape=(2, 3), inds=('a', '_e9021e0000002'), tags=()),
            Tensor(shape=(3, 4), inds=('_e9021e0000002', 'd'), tags=()),
        ])

        >>>  tn ^ all
        Tensor(shape=(2, 4), inds=('a', 'd'), tags=())

    """
    d1, d2 = t1.shape[ax1], t2.shape[ax2]
    if d1 != d2:
        raise ValueError(f"Index sizes don't match: {d1} != {d2}.")

    new_ind = rand_uuid()

    ind1 = t1.inds[ax1]
    ind2 = t2.inds[ax2]
    t1.reindex_({ind1: new_ind})
    t2.reindex_({ind2: new_ind})


def get_tags(ts):
    """Return all the tags in found in ``ts``.

    Parameters
    ----------
    ts :  Tensor, TensorNetwork or sequence of either
        The objects to combine tags from.
    """
    if isinstance(ts, (TensorNetwork, Tensor)):
        ts = (ts,)

    return utup_union((t.tags for t in ts))


# --------------------------------------------------------------------------- #
#                                Tensor Class                                 #
# --------------------------------------------------------------------------- #

class Tensor(object):
    """A labelled, tagged ndarray. The index labels are used instead of
    axis numbers to identify dimensions, and are preserved through operations.

    Parameters
    ----------
    data : numpy.ndarray
        The n-dimensional data.
    inds : sequence of str
        The index labels for each dimension. Must match the number of
        dimensions of ``data``.
    tags : sequence of str, optional
        Tags with which to identify and group this tensor. These will
        be converted into a ``set``.
    left_inds : sequence of str, optional
        Which, if any, indices to group as 'left' indices of an effective
        matrix. This can be useful, for example, when automatically applying
        unitary constraints to impose a certain flow on a tensor network but at
        the atomistic (Tensor) level.

    Examples
    --------

    Basic construction:

        >>> from quimb import randn
        >>> from quimb.tensor import Tensor
        >>> X = Tensor(randn((2, 3, 4)), inds=['a', 'b', 'c'], tags={'X'})
        >>> Y = Tensor(randn((3, 4, 5)), inds=['b', 'c', 'd'], tags={'Y'})

    Indices are automatically aligned, and tags combined, when contracting:

        >>> X @ Y
        Tensor(shape=(2, 5), inds=('a', 'd'), tags={'Y', 'X'})

    """

    def __init__(self, data=1.0, inds=(), tags=None, left_inds=None):
        # a new or copied Tensor always has no owners
        self.owners = dict()

        # Short circuit for copying Tensors
        if isinstance(data, Tensor):
            self._data = data.data
            self._inds = data.inds
            self._tags = data.tags
            self._left_inds = data.left_inds
            return

        self._data = asarray(data)
        self._inds = tuple(inds)
        self._tags = tags_to_utup(tags)
        self._left_inds = tuple(left_inds) if left_inds is not None else None

        nd = ndim(self._data)
        if nd != len(self.inds):
            raise ValueError(
                f"Wrong number of inds, {self.inds}, supplied for array"
                f" of shape {self._data.shape}.")

        if self.left_inds and any(i not in self.inds for i in self.left_inds):
            raise ValueError(f"The 'left' indices {self.left_inds} are not "
                             f"found in {self.inds}.")

    def copy(self, deep=False):
        """Copy this tensor. Note by default (``deep=False``), the underlying
        array will *not* be copied.
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

    @property
    def left_inds(self):
        return self._left_inds

    @left_inds.setter
    def left_inds(self, left_inds):
        self._left_inds = tuple(left_inds) if left_inds is not None else None

    def add_owner(self, tn, tid):
        """Add ``tn`` as owner of this Tensor - it's tag and ind maps will
        be updated whenever this tensor is retagged or reindexed.
        """
        self.owners[hash(tn)] = (weakref.ref(tn), tid)

    def remove_owner(self, tn):
        """Remove TensorNetwork ``tn`` as an owner of this Tensor.
        """
        try:
            del self.owners[hash(tn)]
        except KeyError:
            pass

    def check_owners(self):
        """Check if this tensor is 'owned' by any alive TensorNetworks. Also
        trim any weakrefs to dead TensorNetworks.
        """
        # first parse out dead owners
        for k in tuple(self.owners):
            if not self.owners[k][0]():
                del self.owners[k]

        return len(self.owners) > 0

    def modify(self, **kwargs):
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
        if 'data' in kwargs:
            self._data = asarray(kwargs.pop('data'))

        if 'inds' in kwargs:
            inds = tuple(kwargs.pop('inds'))

            # if this tensor has owners, update their ``ind_map``, but only if
            #     the indices are actually being changed not just permuted
            if (set(self.inds) != set(inds)) and self.check_owners():
                for ref, tid in self.owners.values():
                    ref()._modify_tensor_inds(self.inds, inds, tid)

            self._inds = inds

        if 'tags' in kwargs:
            tags = tags_to_utup(kwargs.pop('tags'))

            # if this tensor has owners, update their ``tag_map``.
            if self.check_owners():
                for ref, tid in self.owners.values():
                    ref()._modify_tensor_tags(self.tags, tags, tid)

            self._tags = tags

        if 'left_inds' in kwargs:
            self.left_inds = kwargs.pop('left_inds')

        if kwargs:
            raise ValueError(f"Option(s) {kwargs} not valid.")

        if len(self.inds) != ndim(self.data):
            raise ValueError("Mismatch between number of data dimensions and "
                             "number of indices supplied.")

        if self.left_inds and any(i not in self.inds for i in self.left_inds):
            raise ValueError(f"The 'left' indices {self.left_inds} are "
                             f"not found in {self.inds}.")

    def isel(self, selectors, inplace=False):
        """Select specific values for some dimensions/indices of this tensor,
        thereby removing them. Analogous to ``X[:, :, 3, :, :]`` with arrays.

        Parameters
        ----------
        selectors : dict[str, int]
            Mapping of index(es) to which value to take.
        inplace : bool, optional
            Whether to select inplace or not.

        Returns
        -------
        Tensor

        Examples
        --------
        >>> T = rand_tensor((2, 3, 4), inds=('a', 'b', 'c'))
        >>> T.isel({'b': -1})
        Tensor(shape=(2, 4), inds=('a', 'c'), tags=())

        See Also
        --------
        TensorNetwork.isel
        """
        T = self if inplace else self.copy()

        new_inds = tuple(ix for ix in self.inds if ix not in selectors)

        data_loc = tuple(selectors.get(ix, slice(None)) for ix in self.inds)
        new_data = self.data[data_loc]

        T.modify(data=new_data, inds=new_inds, left_inds=None)
        return T

    isel_ = functools.partialmethod(isel, inplace=True)

    def add_tag(self, tag):
        """Add a tag to this tensor. Unlike ``self.tags.add`` this also updates
        any TensorNetworks viewing this Tensor.
        """
        self.modify(tags=utup_add(self.tags, tag))

    def expand_ind(self, ind, size):
        """Inplace increase the size of the dimension of ``ind``, the new array
        entries will be filled with zeros.

        Parameters
        ----------
        name : str
            Name of the index to expand.
        size : int, optional
            Size of the expanded index.
        """
        if ind not in self.inds:
            raise ValueError(f"Tensor has no index '{ind}'.")

        size_current = self.ind_size(ind)
        pads = [
            (0, size - size_current) if i == ind else (0, 0)
            for i in self.inds
        ]
        self.modify(data=do('pad', self.data, pads, mode='constant'))

    def new_ind(self, name, size=1, axis=0):
        """Inplace add a new index - a named dimension. If ``size`` is
        specified to be greater than one then the new array entries will be
        filled with zeros.

        Parameters
        ----------
        name : str
            Name of the new index.
        size : int, optional
            Size of the new index.
        axis : int, optional
            Position of the new index.
        """
        new_inds = list(self.inds)
        new_inds.insert(axis, name)

        new_data = do('expand_dims', self.data, axis=axis)

        self.modify(data=new_data, inds=new_inds)
        if size > 1:
            self.expand_ind(name, size)

    new_bond = new_bond

    def conj(self, inplace=False):
        """Conjugate this tensors data (does nothing to indices).
        """
        t = self if inplace else self.copy()
        data = t.data
        if iscomplex(data):
            t.modify(data=conj(data))
        return t

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
        return ndim(self._data)

    @property
    def size(self):
        return self._data.size

    @property
    def dtype(self):
        return self._data.dtype

    def iscomplex(self):
        return iscomplex(self.data)

    def astype(self, dtype, inplace=False):
        """Change the type of this tensor to ``dtype``.
        """
        T = self if inplace else self.copy()
        T.modify(data=astype(self.data, dtype))
        return T

    astype_ = functools.partialmethod(astype, inplace=True)

    def ind_size(self, ind):
        """Return the size of dimension corresponding to ``ind``.
        """
        return int(self.shape[self.inds.index(ind)])

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
        t = self if inplace else self.copy()

        output_inds = tuple(output_inds)  # need to re-use this.

        if set(t.inds) != set(output_inds):
            raise ValueError("'output_inds' must be permutation of the current"
                             f" tensor indices, but {set(t.inds)} != "
                             f"{set(output_inds)}")

        current_ind_map = {ind: i for i, ind in enumerate(t.inds)}
        out_shape = tuple(current_ind_map[i] for i in output_inds)

        t.modify(data=transpose(t.data, out_shape), inds=output_inds)
        return t

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
        t = self if inplace else self.copy()
        diff_ix = set(t.inds) - set(other.inds)

        if len(diff_ix) > 1:
            raise ValueError("More than one index don't match, the transpose "
                             "is therefore not well-defined.")

        # if their indices match, just plain transpose
        if not diff_ix:
            t.transpose_(*other.inds)

        else:
            di, = diff_ix
            new_ix = (i if i in t.inds else di for i in other.inds)
            t.transpose_(*new_ix)

        return t

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
        return do('sum', -el * do('log2', el))

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
        new.modify(tags=(retag_map.get(tag, tag) for tag in new.tags))
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

        new_inds = tuple(index_map.get(ind, ind) for ind in new.inds)

        if self.left_inds:
            new_left_inds = (index_map.get(ind, ind) for ind in self.left_inds)
        else:
            new_left_inds = self.left_inds

        new.modify(inds=new_inds, left_inds=new_left_inds)

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
        t = self if inplace else self.copy()

        if isinstance(fuse_map, dict):
            new_fused_inds, fused_inds = zip(*fuse_map.items())
        else:
            new_fused_inds, fused_inds = zip(*fuse_map)

        unfused_inds = tuple(i for i in t.inds if not
                             any(i in fs for fs in fused_inds))

        # transpose tensor to bring groups of fused inds to the beginning
        t.transpose_(*concat(fused_inds), *unfused_inds)

        # for each set of fused dims, group into product, then add remaining
        dims = iter(t.shape)
        dims = [prod(next(dims) for _ in fs) for fs in fused_inds] + list(dims)

        # create new tensor with new + remaining indices
        #     + drop 'left' marked indices since they might be fused
        t.modify(data=reshape(t.data, dims),
                 inds=(*new_fused_inds, *unfused_inds), left_inds=None)

        return t

    fuse_ = functools.partialmethod(fuse, inplace=True)

    def to_dense(self, *inds_seq):
        """Convert this Tensor into an dense array, with a single dimension
        for each of inds in ``inds_seqs``. E.g. to convert several sites
        into a density matrix: ``T.to_dense(('k0', 'k1'), ('b0', 'b1'))``.
        """
        x = self.fuse([(str(i), ix) for i, ix in enumerate(inds_seq)]).data
        if isinstance(x, np.ndarray):
            return qarray(x)
        return x

    def squeeze(self, inplace=False):
        """Drop any singlet dimensions from this tensor.
        """
        t = self if inplace else self.copy()
        new_shape, new_inds = zip(
            *((d, i) for d, i in zip(self.shape, self.inds) if d > 1))

        new_left_inds = (
            None if self.left_inds is None else
            (i for i in self.left_inds if i in new_inds)
        )

        if len(t.inds) != len(new_inds):
            t.modify(
                data=reshape(t.data, new_shape),
                inds=new_inds,
                left_inds=new_left_inds,
            )

        return t

    squeeze_ = functools.partialmethod(squeeze, inplace=True)

    def norm(self):
        """Frobenius norm of this tensor.
        """
        return norm_fro(self.data)

    def normalize(self, inplace=False):
        T = self if inplace else self.copy()
        T.modify(data=T.data / T.norm())
        return T

    normalize_ = functools.partialmethod(normalize, inplace=True)

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

    def unitize(self, left_inds=None, inplace=False, method='qr'):
        r"""Make this tensor unitary (or isometric) with respect to
        ``left_inds``. The underlying method is set by ``method``.

        Parameters
        ----------
        left_inds : sequence of str
            The indices to group together and treat as the left hand side of a
            matrix.
        inplace : bool, optional
            Whether to perform the unitization inplace.
        method : {'qr', 'exp', 'mgs'}, optional
            How to generate the unitary matrix. The options are:

            - 'qr': use a QR decomposition directly.
            - 'exp': exponential the padded, anti-hermitian part of the array
            - 'mgs': use a explicit modified-gram-schmidt procedure

            Generally, 'qr' is the fastest and best approach, however currently
            ``tensorflow`` cannot back-propagate through for instance, making
            the other two methods necessary.

        Returns
        -------
        Tensor
        """
        if left_inds is None:
            if self.left_inds is None:
                raise ValueError(
                    "You must specify `left_inds` since this tensor does not "
                    "have any indices marked automatically as such in the "
                    "attribute `left_inds`.")
            else:
                left_inds = self.left_inds

        # partition indices into left and right
        L_inds = list(left_inds)
        R_inds = [ix for ix in self.inds if ix not in L_inds]

        # if the tensor is an effective vector, we can just normalize
        if (len(L_inds) == 0) or (len(R_inds) == 0):
            return self.normalize(inplace=inplace)

        LR_inds = L_inds + R_inds

        # fuse this tensor into a matrix and 'isometrize' it
        x = self.to_dense(L_inds, R_inds)
        x = unitize(x, method=method)

        # turn the array back into a tensor
        x = reshape(x, [self.ind_size(ix) for ix in LR_inds])
        Tu = Tensor(x, inds=LR_inds, tags=self.tags)

        if inplace:
            # XXX: do self.transpose_like_(Tu) or Tu.transpose_like_(self)?
            self.modify(data=Tu.data, inds=Tu.inds)
            Tu = self

        return Tu

    unitize_ = functools.partialmethod(unitize, inplace=True)

    def randomize(self, dtype=None, inplace=False, **randn_opts):
        """Randomize the entries of this tensor.

        Parameters
        ----------
        dtype : {None, str}, optional
            The data type of the random entries. If left as the default
            ``None``, then the data type of the current array will be used.
        inplace : bool, optional
            Whether to perform the randomization inplace, by default ``False``.
        randn_opts
            Supplied to :func:`~quimb.gen.rand.randn`.

        Returns
        -------
        Tensor
        """
        t = self if inplace else self.copy()

        if dtype is None:
            dtype = t.dtype

        t.modify(data=randn(t.shape, dtype=dtype, **randn_opts))
        return t

    randomize_ = functools.partialmethod(randomize, inplace=True)

    def flip(self, ind, inplace=False):
        """Reverse the axis on this tensor corresponding to ``ind``. Like
        performing e.g. ``X[:, :, ::-1, :]``.
        """
        if ind not in self.inds:
            raise ValueError(f"Can't find index {ind} on this tensor.")

        t = self if inplace else self.copy()
        flipper = tuple(
            slice(None, None, -1) if i == ind else slice(None) for i in t.inds
        )
        t.modify(data=self.data[flipper])
        return t

    flip_ = functools.partialmethod(flip, inplace=True)

    def multiply_index_diagonal(self, ind, x, inplace=False):
        """Multiply this tensor by 1D array ``x`` as if it were a diagonal
        tensor being contracted into index ``ind``.
        """
        t = self if inplace else self.copy()
        x_broadcast = reshape(x, [(-1 if i == ind else 1) for i in t.inds])
        t.modify(data=t.data * x_broadcast)
        return t

    multiply_index_diagonal_ = functools.partialmethod(
        multiply_index_diagonal, inplace=True)

    def almost_equals(self, other, **kwargs):
        """Check if this tensor is almost the same as another.
        """
        same_inds = (set(self.inds) == set(other.inds))
        if not same_inds:
            return False
        otherT = other.transpose(*self.inds)
        return do('allclose', self.data, otherT.data, **kwargs)

    def drop_tags(self, tags=None):
        """Drop certain tags, defaulting to all, from this tensor.
        """
        if tags is None:
            self.modify(tags=empty_utup())
        else:
            self.modify(tags=utup_difference(self.tags, tags_to_utup(tags)))

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
        return self.contract(other)

    def graph(self, *args, **kwargs):
        """Plot a graph of this tensor and its indices.
        """
        graph(TensorNetwork((self,)), *args, **kwargs)

    def __getstate__(self):
        # This allows pickling, since the copy has no weakrefs.
        return self.copy().__dict__

    def __setstate__(self, state):
        self.__dict__ = state.copy()

    def __repr__(self):
        return (f"{self.__class__.__name__}("
                f"shape={tuple(map(int, self.data.shape))}, "
                f"inds={self.inds}, "
                f"tags={tuple(self.tags)})")

    def __str__(self):
        s = self.__repr__()[:-1]
        s += (f", backend='{infer_backend(self.data)}'"
              f", dtype='{get_dtype_name(self.data)}')")
        return s


# ------------------------- Add ufunc like methods -------------------------- #

def _make_promote_array_func(op, meth_name):

    @functools.wraps(getattr(np.ndarray, meth_name))
    def _promote_array_func(self, other):
        """Use standard array func, but make sure Tensor inds match.
        """
        if isinstance(other, Tensor):

            if set(self.inds) != set(other.inds):
                raise ValueError("The indicies of these two tensors do not "
                                 f"match: {self.inds} != {other.inds}")

            otherT = other.transpose(*self.inds)

            return Tensor(
                data=op(self.data, otherT.data), inds=self.inds,
                tags=utup_union((self.tags, other.tags)))
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

    _EXTRA_PROPS = ()

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
            self.tag_map = ts.tag_map.copy()
            self.ind_map = ts.ind_map.copy()
            self.tensor_map = dict()
            for tid, t in ts.tensor_map.items():
                self.tensor_map[tid] = t if virtual else t.copy()
                self.tensor_map[tid].add_owner(self, tid)
            for ep in ts.__class__._EXTRA_PROPS:
                setattr(self, ep, getattr(ts, ep))
            return

        # parameters
        self.structure = structure
        self.structure_bsz = structure_bsz
        self.nsites = nsites
        self.sites = sites

        # internal structure
        self.tensor_map = dict()
        self.tag_map = dict()
        self.ind_map = dict()

        self._inner_inds = empty_utup()
        for t in ts:
            self.add(t, virtual=virtual, check_collisions=check_collisions)
        self._inner_inds = None

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
                    raise ValueError("Conflicting values found on tensor "
                                     f"networks for property {prop}. First "
                                     f"value: {u}, second value: {v}")

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

    _EXTRA_PROPS = ()

    @classmethod
    def from_TN(cls, tn, like=None, inplace=False, **kwargs):
        """Construct a specific tensor network subclass (i.e. one with some
        promise about structure/geometry and tags/inds such as an MPS) from
        a generic tensor network which should have that structure already.

        Parameters
        ----------
        cls : class
            The TensorNetwork subclass to convert ``tn`` to.
        tn : TensorNetwork
            The TensorNetwork to convert.
        like : TensorNetwork, optional
            If specified, try and retrieve the neccesary attribute values from
            this tensor network.
        inplace : bool, optional
            Whether to perform the conversion inplace or not.
        kwargs
            Extra properties of the TN subclass that should be specified.
        """
        new_tn = tn if inplace else tn.copy()

        for prop in cls._EXTRA_PROPS:
            # equate real and private property name
            prop_name = prop.lstrip('_')

            # get value from kwargs
            if prop_name in kwargs:
                setattr(new_tn, prop, kwargs.pop(prop_name))

            # get value from another manually specified TN
            elif (like is not None) and hasattr(like, prop_name):
                setattr(new_tn, prop, getattr(like, prop_name))

            # get value directly from TN
            elif hasattr(tn, prop_name):
                setattr(new_tn, prop, getattr(tn, prop_name))

            else:
                raise ValueError(
                    f"You need to specify '{prop_name}' for the tensor network"
                    f" class {cls}, and ensure that it correctly corresponds "
                    f"to the structure of the tensor network supplied, since "
                    f"it cannot be found as an attribute on the TN: {tn}.")

        if kwargs:
            raise ValueError(
                f"Options {kwargs} are invalid for the class {cls}.")

        new_tn.__class__ = cls
        return new_tn

    def view_as(self, cls, inplace=False, **kwargs):
        """View this tensor network as subclass ``cls``.
        """
        return cls.from_TN(self, inplace=inplace, **kwargs)

    view_as_ = functools.partialmethod(view_as, inplace=True)

    def view_like(self, like, inplace=False, **kwargs):
        """View this tensor network as the same subclass ``cls`` as ``like``
        inheriting its extra properties as well.
        """
        return self.view_as(like.__class__, like=like,
                            inplace=inplace, **kwargs)

    view_like_ = functools.partialmethod(view_like, inplace=True)

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
                x_map[x] = utup_add(x_map[x], tid)
            else:
                x_map[x] = (tid,)

    @staticmethod
    def _remove_tid(xs, x_map, tid):
        """Remove tid from the relevant map.
        """
        for x in xs:
            try:
                tids = utup_discard(x_map[x], tid)
                if not tids:
                    # tid was last tensor -> delete entry
                    del x_map[x]
                else:
                    x_map[x] = tids
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

    def add_tensor_network(self, tn, virtual=False, check_collisions=True):
        """
        """
        self._combine_sites(tn)
        self._combine_properties(tn)

        if check_collisions:  # add tensors individually
            if getattr(self, '_inner_inds', None) is None:
                self._inner_inds = self.inner_inds()

            # check for matching inner_indices -> need to re-index
            other_inner_ix = tn.inner_inds()
            clash_ix = utup_intersection((self._inner_inds, other_inner_ix))

            if clash_ix:
                can_keep_ix = utup_difference(other_inner_ix, self._inner_inds)
                new_inds = tuple(rand_uuid() for _ in range(len(clash_ix)))
                reind = dict(zip(clash_ix, new_inds))
                self._inner_inds += new_inds + can_keep_ix
            else:
                self._inner_inds += other_inner_ix

            # add tensors, reindexing if necessary
            for tid, tsr in tn.tensor_map.items():
                if clash_ix and any(i in reind for i in tsr.inds):
                    tsr = tsr.reindex(reind, inplace=virtual)
                self.add_tensor(tsr, virtual=virtual, tid=tid)

        else:  # directly add tensor/tag indexes
            for tid, tsr in tn.tensor_map.items():
                T = tsr if virtual else tsr.copy()
                self.tensor_map[tid] = T
                T.add_owner(self, tid)

            self.tag_map = merge_with(utup_union, self.tag_map, tn.tag_map)
            self.ind_map = merge_with(utup_union, self.ind_map, tn.ind_map)

    def add(self, t, virtual=False, check_collisions=True):
        """Add Tensor, TensorNetwork or sequence thereof to self.
        """
        if isinstance(t, (tuple, list)):
            for each_t in t:
                self.add(each_t, virtual=virtual,
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
            self.add_tensor_network(t, virtual=virtual,
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
        self._remove_tid(utup_difference(old, new), self.ind_map, tid)
        self._add_tid(utup_difference(new, old), self.ind_map, tid)

    @property
    def num_tensors(self):
        """The total number of tensors in the tensor network.
        """
        return len(self.tensor_map)

    @property
    def num_indices(self):
        """The total number of indices in the tensor network.
        """
        return len(self.ind_map)

    def calc_nsites(self):
        """Calculate how many tags there are which match ``structure``.
        """
        return len(
            re.findall(self.structure.format(r"(\d+)"), ",".join(self.tags))
        )

    @staticmethod
    @functools.lru_cache(8)
    def regex_for_calc_sites_cached(structure):
        return re.compile(structure.format(r"(\d+)"))

    def calc_sites(self):
        """Calculate with sites this TensorNetwork contain based on its
        ``structure``.
        """
        rgx = self.regex_for_calc_sites_cached(self.structure)
        matches = rgx.findall(",".join(self.tags))
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
        tags = tags_to_utup(tags)

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

        tids = utup_union(tn.ind_map.get(ix, ()) for ix in index_map)

        for tid in tids:
            T = tn.tensor_map[tid]
            T.reindex_(index_map)

        return tn

    reindex_ = functools.partialmethod(reindex, inplace=True)

    def mangle_inner_(self, append=None, which=None):
        """Generate new index names for internal bonds, meaning that when this
        tensor network is combined with another, there should be no collisions.

        Parameters
        ----------
        append : None or str, optional
            Whether and what to append to the indices to perform the mangling.
            If ``None`` a whole new random UUID will be generated.
        which : sequence of str, optional
            Which indices to rename, if ``None`` (the default), all inner
            indices.
        """
        if which is None:
            which = self.inner_inds()

        if append is None:
            reindex_map = {ix: rand_uuid() for ix in which}
        else:
            reindex_map = {ix: ix + append for ix in which}

        self.reindex_(reindex_map)
        return self

    def conj(self, mangle_inner=False, inplace=False):
        """Conjugate all the tensors in this network (leaves all indices).
        """
        tn = self if inplace else self.copy()

        for t in tn:
            t.conj_()

        if mangle_inner:
            append = None if mangle_inner is True else str(mangle_inner)
            tn.mangle_inner_(append)

        return tn

    conj_ = functools.partialmethod(conj, inplace=True)

    @property
    def H(self):
        """Conjugate all the tensors in this network (leaves all indices).
        """
        return self.conj()

    def multiply(self, x, inplace=False, spread_over=8):
        """Scalar multiplication of this tensor network with ``x``.

        Parameters
        ----------
        x : scalar
            The number to multiply this tensor network by.
        inplace : bool, optional
            Whether to perform the multiplication inplace.
        spread_over : int, optional
            How many tensors to try and spread the multiplication over, in
            order that the effect of multiplying by a very large or small
            scalar is not concentrated.
        """
        multiplied = self if inplace else self.copy()

        if spread_over == 'all':
            spread_over = self.num_tensors
        else:
            spread_over = min(self.num_tensors, spread_over)

        if spread_over == 1:
            x_sign = 1.0
            x_spread = x
        else:
            # take care of sign of real scalars so as to keep real
            if iscomplex(x):
                x_sign = 1.0
            else:
                x_sign = do('sign', x)
                x = abs(x)

            x_spread = x ** (1 / spread_over)

        tensors = iter(multiplied)
        for i in range(spread_over):
            tensor = next(tensors)

            # take into account a negative factor with single minus sign
            if i == 0:
                tensor.modify(data=tensor.data * (x_sign * x_spread))
            else:
                tensor.modify(data=tensor.data * x_spread)

        return multiplied

    multiply_ = functools.partialmethod(multiply, inplace=True)

    def multiply_each(self, x, inplace=False):
        """Scalar multiplication of each tensor in this
        tensor network with ``x``. If trying to spread a
        multiplicative factor ``fac`` uniformly over all tensors in the
        network and the number of tensors is large, then calling
        ``multiply(fac)`` can be inaccurate due to precision loss.
        If one has a routine that can precisely compute the ``x``
        to be applied to each tensor, then this function avoids
        the potential inaccuracies in ``multiply()``.

        Parameters
        ----------
        x : scalar
            The number that multiplies each tensor in the network
        inplace : bool, optional
            Whether to perform the multiplication inplace.
        """
        multiplied = self if inplace else self.copy()

        for t in multiplied.tensors:
            t.modify(data=t.data * x)

        return multiplied

    multiply_each_ = functools.partialmethod(multiply_each, inplace=True)

    def __mul__(self, other):
        """Scalar multiplication.
        """
        return self.multiply(other)

    def __rmul__(self, other):
        """Right side scalar multiplication.
        """
        return self.multiply(other)

    def __imul__(self, other):
        """Inplace scalar multiplication.
        """
        return self.multiply_(other)

    def __truediv__(self, other):
        """Scalar division.
        """
        return self.multiply(other**-1)

    def __itruediv__(self, other):
        """Inplace scalar division.
        """
        return self.multiply_(other**-1)

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
            return (self.site_tag(sites),)

        elif isinstance(sites, slice):
            return tuple(map(self.structure.format, self.slice2sites(sites)))
        else:
            raise TypeError("``sites2tags`` needs an integer or a slice"
                            f", but got {sites}")

    def _get_tids_from(self, xmap, xs, which):
        inverse = which[0] == '!'
        if inverse:
            which = which[1:]

        combine = {'all': utup_intersection, 'any': utup_union}[which]
        tid_sets = (xmap[x] for x in xs)
        tids = combine(tid_sets)

        if inverse:
            return utup_difference(self.tensor_map, tids)

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
            return tuple(self.tensor_map)
        elif isinstance(tags, (Integral, slice)):
            tags = self.sites2tags(tags)
        else:
            tags = tags_to_utup(tags)

        return self._get_tids_from(self.tag_map, tags, which)

    def _get_tids_from_inds(self, inds, which='all'):
        """Like ``_get_tids_from_tags`` but specify inds instead.
        """
        inds = tags_to_utup(inds)
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

    select_any = functools.partialmethod(select, which='any')
    select_all = functools.partialmethod(select, which='all')

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
        inds = utup_union(t.inds for t in tagged_ts)

        # find all tensors with those inds, and remove the initial tensors
        inds_tids = utup_union(self.ind_map[i] for i in inds)
        neighbour_tids = utup_difference(inds_tids, tagged_tids)

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
            return self.select_any(self.sites2tags(tags))

        elif isinstance(tags, Integral):
            tensors = self.select_tensors(self.sites2tags(tags), which='any')

        else:
            tensors = self.select_tensors(tags, which='all')

        if len(tensors) == 0:
            raise KeyError(f"Couldn't find any tensors matching {tags}.")

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
        if len(tagged_tids) == self.num_tensors:
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
                f"have matching dimensions, but {dl} != {dr}.")

        tn.delete(where, which=which)

        tn.reindex_({il: ir})
        return tn

    def replace_with_svd(self, where, left_inds, eps, *, which='any',
                         right_inds=None, method='isvd', max_bond=None,
                         absorb='both', cutoff_mode='rel', renorm=None,
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
        method : str, optional
            How to perform the decomposition, if not an iterative method
            the subnetwork dense tensor will be formed first, see
            :func:`~quimb.tensor.tensor_core.tensor_split` for options.
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

        tags = svd_section.tags if keep_tags else empty_utup()
        ltags = tags_to_utup(ltags)
        rtags = tags_to_utup(rtags)

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

        ltags = utup_union((tags, ltags))
        rtags = utup_union((tags, rtags))

        TL, TR = tensor_split(A, left_inds=left_inds, right_inds=right_inds,
                              method=method, cutoff=eps, absorb=absorb,
                              max_bond=max_bond, cutoff_mode=cutoff_mode,
                              renorm=renorm, ltags=ltags, rtags=rtags)

        leave |= TL
        leave |= TR

        return leave

    replace_with_svd_ = functools.partialmethod(replace_with_svd, inplace=True)

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
            T.modify(data=do('zeros', new_shape, dtype=T.dtype, like=T.data))

    def contract_between(self, tags1, tags2, **contract_opts):
        """Contract the two tensors specified by ``tags1`` and ``tags2``
        respectively. This is an inplace operation. No-op if the tensor
        specified by ``tags1`` and ``tags2`` is the same tensor.

        Parameters
        ----------
        tags1 :
            Tags uniquely identifying the first tensor.
        tags2 : str or sequence of str
            Tags uniquely identifying the second tensor.
        contract_opts
            Supplied to :func:`~quimb.tensor.tensor_core.tensor_contract`.
        """
        tid1, = self._get_tids_from_tags(tags1, which='all')
        tid2, = self._get_tids_from_tags(tags2, which='all')

        # allow no-op for same tensor specified twice ('already contracted')
        if tid1 == tid2:
            return

        T1 = self._pop_tensor(tid1)
        T2 = self._pop_tensor(tid2)

        self |= tensor_contract(T1, T2, **contract_opts)

    def _compress_between_tids(self, tid1, tid2, canonize_distance=None,
                               **compress_opts):
        Tl = self.tensor_map[tid1]
        Tr = self.tensor_map[tid2]

        if canonize_distance:
            self._canonize_around_tids(
                (tid1, tid2), max_distance=canonize_distance)

        tensor_compress_bond(Tl, Tr, **compress_opts)

    def compress_between(self, tags1, tags2, canonize_distance=None,
                         **compress_opts):
        r"""Compress the bond between the two single tensors in this network
        specified by ``tags1`` and ``tags2`` using
        :func:`~quimb.tensor.tensor_core.tensor_compress_bond`::

              |    |    |    |           |    |    |    |
            ==●====●====●====●==       ==●====●====●====●==
             /|   /|   /|   /|          /|   /|   /|   /|
              |    |    |    |           |    |    |    |
            ==●====1====2====●==  ==>  ==●====L----R====●==
             /|   /|   /|   /|          /|   /|   /|   /|
              |    |    |    |           |    |    |    |
            ==●====●====●====●==       ==●====●====●====●==
             /|   /|   /|   /|          /|   /|   /|   /|

        This is an inplace operation. The compression is unlikely to be optimal
        with respect to the frobenius norm, unless the TN is already
        canonicalized at the two tensors. The ``absorb`` kwarg can be
        specified to yield an isometry on either the left or right resulting
        tensors.

        Parameters
        ----------
        tags1 :
            Tags uniquely identifying the first ('left') tensor.
        tags2 : str or sequence of str
            Tags uniquely identifying the second ('right') tensor.
        compress_opts
            Supplied to :func:`~quimb.tensor.tensor_core.tensor_compress_bond`.

        See Also
        --------
        canonize_between
        """
        tid1, = self._get_tids_from_tags(tags1, which='all')
        tid2, = self._get_tids_from_tags(tags2, which='all')
        self._compress_between_tids(
            tid1, tid2, canonize_distance=canonize_distance, **compress_opts)

    def compress_all(self, inplace=False, **compress_opts):
        """Inplace compress all bonds in this network.
        """
        tn = self if inplace else self.copy()
        tn.fuse_multibonds_()

        for ix, tids in tn.ind_map.items():
            if len(tids) != 2:
                continue

            T1, T2 = (tn.tensor_map[tid] for tid in tids)
            try:
                tensor_compress_bond(T1, T2, **compress_opts)
            except ValueError:
                continue
            except ZeroDivisionError:
                tn.convert_to_zero()
                break

        return tn

    compress_all_ = functools.partialmethod(compress_all, inplace=True)

    def _canonize_between_tids(self, tid1, tid2, **canonize_opts):
        Tl = self.tensor_map[tid1]
        Tr = self.tensor_map[tid2]
        tensor_canonize_bond(Tl, Tr, **canonize_opts)

    def canonize_between(self, tags1, tags2, **canonize_opts):
        r"""'Canonize' the bond between the two single tensors in this network
        specified by ``tags1`` and ``tags2`` using ``tensor_canonize_bond``::

              |    |    |    |           |    |    |    |
            --●----●----●----●--       --●----●----●----●--
             /|   /|   /|   /|          /|   /|   /|   /|
              |    |    |    |           |    |    |    |
            --●----1----2----●--  ==>  --●---->~~~~R----●--
             /|   /|   /|   /|          /|   /|   /|   /|
              |    |    |    |           |    |    |    |
            --●----●----●----●--       --●----●----●----●--
             /|   /|   /|   /|          /|   /|   /|   /|


        This is an inplace operation. This can only be used to put a TN into
        truly canonical form if the geometry is a tree, such as an MPS.

        Parameters
        ----------
        tags1 :
            Tags uniquely identifying the first ('left') tensor, which will
            become an isometry.
        tags2 : str or sequence of str
            Tags uniquely identifying the second ('right') tensor.
        canonize_opts
            Supplied to :func:`~quimb.tensor.tensor_core.tensor_canonize_bond`.

        See Also
        --------
        compress_between
        """
        tid1, = self._get_tids_from_tags(tags1, which='all')
        tid2, = self._get_tids_from_tags(tags2, which='all')
        self._canonize_between_tids(tid1, tid2, **canonize_opts)

    def _canonize_around_tids(self, tids, max_distance=None,
                              inplace=True, **canonize_opts):
        tn = self if inplace else self.copy()

        border = tids
        all_tids = tuple(tn.tensor_map)
        remaining = utup_difference(all_tids, border)
        border_d = [(x, 0) for x in border]

        # the sequence of tensors pairs to canonize
        seq = []

        while border_d:

            tid, d = border_d.pop(0)
            if (max_distance is not None) and d >= max_distance:
                continue

            ix = tn.tensor_map[tid].inds

            # get un-canonized neighbors
            neighbor_tids = utup_union(tn.ind_map[i] for i in ix)
            neighbor_tids = utup_intersection((neighbor_tids, remaining))

            for n_tid in neighbor_tids:
                remaining = utup_discard(remaining, n_tid)
                border_d.append((n_tid, d + 1))
                seq.append((n_tid, tid))

        # want to start furthest away
        seq.reverse()

        for tid1, tid2 in seq:
            T1 = tn.tensor_map[tid1]
            T2 = tn.tensor_map[tid2]
            tensor_canonize_bond(T1, T2, **canonize_opts)

        return tn

    def canonize_around(self, tags, which='all', max_distance=None,
                        inplace=False, **canonize_opts):
        r"""Expand a locally canonical region around ``tags``::

                      --●---●--
                    |   |   |   |
                  --●---v---v---●--
                |   |   |   |   |   |
              --●--->---v---v---<---●--
            |   |   |   |   |   |   |   |
            ●--->--->---O---O---<---<---●
            |   |   |   |   |   |   |   |
              --●--->---^---^---^---●--
                |   |   |   |   |   |
                  --●---^---^---●--
                    |   |   |   |
                      --●---●--

                             <=====>
                             max_distance = 2 e.g.

        Shown on a grid here but applicable to arbitrary geometry. This is a
        way of gauging a tensor network that results in a canonical form if the
        geometry is described by a tree (e.g. an MPS or TTN). The canonizations
        proceed inwards via QR decompositions.

        The sequence generated by round-robin expanding the boundary of the
        originally specified tensors - it will only be unique for trees.

        Parameters
        ----------
        tags : str, or sequence  or str
            Tags defining which set of tensors to locally canonize around.
        which : {'all', 'any', '!all', '!any'}, optional
            How select the tensors based on tags.
        max_distance : None or int, optional
            How far, in terms of graph distance, to canonize tensors.
        inplace : bool, optional
            Whether to perform

        """
        # the set of tensor tids that are in the 'bulk'
        border = self._get_tids_from_tags(tags, which=which)
        return self._canonize_around_tids(border, max_distance=max_distance,
                                          inplace=inplace, **canonize_opts)

    canonize_around_ = functools.partialmethod(canonize_around, inplace=True)

    def new_bond(self, tags1, tags2, **opts):
        """Inplace addition of a dummmy (size 1) bond between the single
        tensors specified by by ``tags1`` and ``tags2``.

        Parameters
        ----------
        tags1 : sequence of str
            Tags identifying the first tensor.
        tags2 : sequence of str
            Tags identifying the second tensor.
        opts
            Supplied to :func:`~quimb.tensor.tensor_core.new_bond`.

        See Also
        --------
        new_bond
        """
        tid1, = self._get_tids_from_tags(tags1, which='all')
        tid2, = self._get_tids_from_tags(tags2, which='all')
        new_bond(self.tensor_map[tid1], self.tensor_map[tid2], **opts)

    def cut_between(self, left_tags, right_tags, left_ind, right_ind):
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

    def isel(self, selectors, inplace=False):
        """Select specific values for some dimensions/indices of this tensor
        network, thereby removing them.

        Parameters
        ----------
        selectors : dict[str, int]
            Mapping of index(es) to which value to take.
        inplace : bool, optional
            Whether to select inplace or not.

        Returns
        -------
        TensorNetwork

        See Also
        --------
        Tensor.isel
        """
        tn = self if inplace else self.copy()

        for tid in utup_union(map(self.ind_map.__getitem__, selectors)):
            tn.tensor_map[tid].isel_(selectors)

        return tn

    isel_ = functools.partialmethod(isel, inplace=True)

    def cut_iter(self, *inds):
        """Cut and iterate over one or more indices in this tensor network.
        Each network yielded will have that index removed, and the sum of all
        networks will equal the original network. This works by iterating over
        the product of all combinations of each bond supplied to ``isel``.
        As such, the number of networks produced is exponential in the number
        of bonds cut.

        Parameters
        ----------
        inds : sequence of str
            The bonds to cut.

        Yields
        ------
        TensorNetwork


        Examples
        --------

        Here we'll cut the two extra bonds of a cyclic MPS and sum the
        contraction of the resulting 49 OBC MPS norms:

            >>> psi = MPS_rand_state(10, bond_dim=7, cyclic=True)
            >>> norm = psi.H & psi
            >>> bnds = bonds(norm[0], norm[-1])
            >>> sum(tn ^ all for tn in norm.cut_iter(*bnds))
            1.0

        See Also
        --------
        TensorNetwork.isel, TensorNetwork.cut_between
        """
        ranges = [range(self.ind_size(ix)) for ix in inds]
        for which in itertools.product(*ranges):
            selector = dict(zip(inds, which))
            yield self.isel(selector)

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
            raise ValueError(f"This operator has dimension {d} but needs "
                             f"dimension {db}.")

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
        U : array
            The gauge to insert.
        where1 : str, sequence of str, or int
            Tags defining the location of the 'left' tensor.
        where2 : str, sequence of str, or int
            Tags defining the location of the 'right' tensor.
        Uinv : array
            The inverse gauge, ``U @ Uinv == Uinv @ U == eye``, to insert.
            If not given will be calculated using :func:`numpy.linalg.inv`.
        """
        n1, = self._get_tids_from_tags(where1, which='all')
        n2, = self._get_tids_from_tags(where2, which='all')
        T1, T2 = self.tensor_map[n1], self.tensor_map[n2]
        bnd, = T1.bonds(T2)

        if Uinv is None:
            Uinv = do('linalg.inv', U)

            # if we get wildly larger inverse due to singular U, try pseudo-inv
            if vdot(Uinv, Uinv) / vdot(U, U) > 1 / tol:
                Uinv = do('linalg.pinv', U, rcond=tol**0.5)

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
        c_tags = empty_utup()

        for tags in tags_seq:
            # accumulate tags from each contractions
            c_tags = utup_union((c_tags, tags_to_utup(tags)))

            # peform the next contraction
            tn = tn.contract_tags(c_tags, inplace=True, which='any', **opts)

            if not isinstance(tn, TensorNetwork):
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
            Whether to perform the contraction inplace. This is only valid
            if not all tensors are contracted (which doesn't produce a TN).
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
        if tags is all:
            return tensor_contract(*self, **opts)

        # Check for a structured strategy for performing contraction...
        if self.structure is not None:

            # but only use for total or slice tags
            if (tags is ...) or isinstance(tags, slice):
                return self.contract_structured(tags, inplace=inplace, **opts)

        # Else just contract those tensors specified by tags.
        return self.contract_tags(tags, inplace=inplace, **opts)

    contract_ = functools.partialmethod(contract, inplace=True)

    def contraction_width(self, **contract_opts):
        """Compute the 'contraction width' of this tensor network. This
        is defined as log2 of the maximum tensor size produced during the
        contraction sequence. If every index in the network has dimension 2
        this corresponds to the maximum rank tensor produced.
        """
        path_info = self.contract(all, get='path-info', **contract_opts)
        return math.log2(path_info.largest_intermediate)

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

    def aslinearoperator(self, left_inds, right_inds,
                         ldims=None, rdims=None, backend=None):
        return TNLinearOperator(self, left_inds, right_inds,
                                ldims, rdims, backend=backend)

    def trace(self, left_inds, right_inds):
        """Trace over ``left_inds`` joined with ``right_inds``
        """
        tn = self.reindex({u: l for u, l in zip(left_inds, right_inds)})
        return tn.contract_tags(...)

    def to_dense(self, *inds_seq, **contract_opts):
        """Convert this network into an dense array, with a single dimension
        for each of inds in ``inds_seqs``. E.g. to convert several sites
        into a density matrix: ``TN.to_dense(('k0', 'k1'), ('b0', 'b1'))``.
        """
        tags = contract_opts.pop('tags', all)
        T = self.contract(
            tags, output_inds=tuple(concat(inds_seq)), **contract_opts)
        return T.to_dense(*inds_seq)

    # --------------- information about indices and dimensions -------------- #

    @property
    def tags(self):
        return tuple(self.tag_map.keys())

    def all_inds(self):
        """Return a tuple of all indices (with repetition) in this network.
        """
        return tuple(self.ind_map)

    def inner_inds(self):
        """Tuple of interior indices, assumed to be any indices that appear
        twice or more (this only holds generally for non-hyper tensor
        networks).
        """
        return tuple(i for i, tids in self.ind_map.items() if len(tids) == 2)

    def outer_inds(self):
        """Tuple of exterior indices, assumed to be any lone indices (this only
        holds generally for non-hyper tensor networks).
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
            tn.fuse_multibonds_()

        return tn

    squeeze_ = functools.partialmethod(squeeze, inplace=True)

    def unitize(self, mode='error', inplace=False, method='qr'):
        """
        """
        tn = self if inplace else self.copy()
        for t in tn:
            if (t.left_inds is None) and (mode == 'error'):
                raise ValueError("The tensor {} doesn't have left indices "
                                 "marked using the `left_inds` attribute.")
            t.unitize_(method=method)
        return tn

    unitize_ = functools.partialmethod(unitize, inplace=True)

    def randomize(self, dtype=None, seed=None, inplace=False, **randn_opts):
        """Randomize every tensor in this TN - see
        :meth:`quimb.tensor.tensor_core.Tensor.randomize`.

        Parameters
        ----------
        dtype : {None, str}, optional
            The data type of the random entries. If left as the default
            ``None``, then the data type of the current array will be used.
        seed : None or int, optional
            Seed for the random number generator.
        inplace : bool, optional
            Whether to perform the randomization inplace, by default ``False``.
        randn_opts
            Supplied to :func:`~quimb.gen.rand.randn`.

        Returns
        -------
        TensorNetwork
        """
        tn = self if inplace else self.copy()

        if seed is not None:
            seed_rand(seed)

        for t in tn:
            t.randomize_(dtype=dtype, **randn_opts)

        return tn

    randomize_ = functools.partialmethod(randomize, inplace=True)

    def equalize_norms(self, value=None, inplace=False):
        """Make the Frobenius norm of every tensor in this TN equal without
        changing the overall value if ``value=None``, or set the norm of every
        tensor to ``value`` by scalar multiplication only.
        """
        tn = self if inplace else self.copy()

        norms = [t.norm() for t in tn]

        if value is None:
            backend = infer_backend(norms[0])
            value = do('power',
                       math.e,
                       do('mean',
                          do('log',
                             do('array', norms, like=backend))), like=backend)

        for t, xi in zip(tn, norms):
            t.modify(data=(value / xi) * t.data)

        return tn

    equalize_norms_ = functools.partialmethod(equalize_norms, inplace=True)

    def balance_bonds(self, inplace=False):
        """
        """
        tn = self if inplace else self.copy()

        for ix, tids in tn.ind_map.items():
            if len(tids) != 2:
                continue
            tid1, tid2 = tids
            t1, t2 = [tn.tensor_map[x] for x in (tid1, tid2)]
            tensor_balance_bond(t1, t2)

        return tn

    balance_bonds_ = functools.partialmethod(balance_bonds, inplace=True)

    def fuse_multibonds(self, inplace=False):
        """Fuse any multi-bonds (more than one index shared by the same pair
        of tensors) into a single bond.
        """
        tn = self if inplace else self.copy()

        seen = collections.defaultdict(list)
        for ix, tids in tn.ind_map.items():

            # only want to fuse inner bonds
            if len(tids) > 1:
                seen[utup(tids)].append(ix)

        for tidset, ixs in seen.items():
            if len(ixs) > 1:
                for tid in tidset:
                    self.tensor_map[tid].fuse_({ixs[0]: ixs})

        return tn

    fuse_multibonds_ = functools.partialmethod(fuse_multibonds, inplace=True)

    def flip(self, inds, inplace=False):
        """Flip the dimension corresponding to indices ``inds`` on all tensors
        that share it.
        """
        tn = self if inplace else self.copy()

        if isinstance(inds, str):
            inds = (inds,)

        for ind in inds:
            tids = tn.ind_map[ind]
            for tid in tids:
                tn.tensor_map[tid].flip_(ind)

        return tn

    flip_ = functools.partialmethod(flip, inplace=True)

    def rank_simplify(self, inplace=False, optimize='greedy-rank',
                      **contract_opts):
        """Simplify this tensor network by performing contractions that don't
        increase the rank of any tensors. This may need to be run several times
        since the full contraction path generated is not necessarily greedily
        ordered in terms of rank increases.

        Parameters
        ----------
        inplace : bool, optional
            Whether to perform the rand reduction inplace.
        optimize : str, opt_einsum.PathOptimizer
            How to choose the the full contraction, which will be performed
            up until the point that any tensor ranks start to increase.
        contract_opts
            Supplied to :func:`tensor_contract` to generate the full path, from
            which initial rank reducing contraction steps will be taken.

        Returns
        -------
        TensorNetwork

        See Also
        --------
        full_simplify, column_reduce, diagonal_reduce
        """
        tn = self if inplace else self.copy()

        # first remove floating scalar tensors
        scalars = []
        for tid, t in tuple(tn.tensor_map.items()):
            if len(t.inds) == 0:
                tn._pop_tensor(tid)
                scalars.append(t.data)
        if scalars:
            tn.multiply_(prod(scalars))

        # then contract rank reducing tensors
        info = tn.contract(all, get='path-info',
                           optimize=optimize, **contract_opts)
        contract_tids = list(tn.tensor_map)

        for c in info.contraction_list:
            # whilst contractions don't increase rank perform them
            lhs, output_i = c[2].split('->')
            inputs_i = lhs.split(',')
            if len(output_i) > max(map(len, inputs_i)):
                break

            # get the tensors that the contraction corresponds to
            tids = [contract_tids.pop(x) for x in sorted(c[0], reverse=True)]
            Ts = [tn._pop_tensor(tid) for tid in tids]

            # find the correct output indices and perform the contraction
            oix = [info.quimb_symbol_map[x] for x in output_i]
            T = tensor_contract(*Ts, output_inds=oix)

            # handle the case when a scalar is created
            if not isinstance(T, Tensor):
                T = Tensor(T)

            # add the new tensor back into the network
            tid_ij = rand_uuid(base="_T")
            tn.add_tensor(T, tid=tid_ij, virtual=True)
            contract_tids.append(tid_ij)

        return tn

    rank_simplify_ = functools.partialmethod(rank_simplify, inplace=True)

    def diagonal_reduce(self, inplace=False, output_inds=None, atol=1e-12):
        """Find tensors with diagonal structure and collapse those axes. This
        will create a tensor 'hyper' network with indices repeated 2+ times, as
        such, output indices should be explicitly supplied when contracting, as
        they can no longer be automatically inferred. For example:

            >>> tn_diag = tn.diagonal_reduce()
            >>> tn_diag.contract(all, output_inds=[])

        Parameters
        ----------
        inplace, bool, optional
            Whether to perform the diagonal reduction inplace.
        output_inds : sequence of str, optional
            Which indices to explicitly consider as outer legs of the tensor
            network and thus not replace. If not given, these will be taken as
            all the indices that appear once.
        atol : float, optional
            When identifying diagonal tensors, the absolute tolerance with
            which to compare to zero with.

        Returns
        -------
        TensorNetwork

        See Also
        --------
        full_simplify, rank_simplify, antidiag_gauge, column_reduce
        """
        tn = self if inplace else self.copy()

        if output_inds is None:
            output_inds = set(self.outer_inds())

        queue = list(tn.tensors)
        while queue:
            t = queue.pop()
            ij = find_diag_axes(t.data, atol=atol)

            # no diagonals
            if ij is None:
                continue

            i, j = ij
            ix_i, ix_j = t.inds[i], t.inds[j]
            if ix_j in output_inds:
                if ix_i in output_inds:
                    # both indices are outer indices - leave them
                    continue
                # just j is, make sure j -> i
                ixmap = {ix_i: ix_j}
            else:
                ixmap = {ix_j: ix_i}

            # e.g. if `ij == (0, 2)` then here we want 'abcd -> abad -> abd'
            tmp_inds = [ixmap.get(ix, ix) for ix in t.inds]
            new_inds = list(unique(tmp_inds))
            eq = _maybe_map_indices_to_alphabet(new_inds, [tmp_inds], new_inds)

            # extract the diagonal and update the tensor
            new_data = do('einsum', eq, t.data, like=t.data)
            t.modify(data=new_data, inds=new_inds, left_inds=None)

            # update wherever else the changed index appears (e.g. 'c' above)
            tn.reindex_(ixmap)

            # tensor might still have diagonal indices
            queue.append(t)

        return tn

    diagonal_reduce_ = functools.partialmethod(diagonal_reduce, inplace=True)

    def antidiag_gauge(self, inplace=False, output_inds=None, atol=1e-12):
        """Flip the order of any bonds connected to antidiagonal tensors.
        Whilst this is just a gauge fixing (with the gauge being the flipped
        identity) it then allows ``diagonal_reduce`` to then simplify those
        indices.

        Parameters
        ----------
        inplace, bool, optional
            Whether to perform the antidiagonal gauging inplace.
        output_inds : sequence of str, optional
            Which indices to explicitly consider as outer legs of the tensor
            network and thus not flip. If not given, these will be taken as
            all the indices that appear once.
        atol : float, optional
            When identifying antidiagonal tensors, the absolute tolerance with
            which to compare to zero with.

        Returns
        -------
        TensorNetwork

        See Also
        --------
        full_simplify, rank_simplify, diagonal_reduce, column_reduce
        """
        tn = self if inplace else self.copy()

        if output_inds is None:
            output_inds = set(self.outer_inds())

        done = set()

        queue = list(tn.tensor_map)
        while queue:
            tid = queue.pop()
            t = tn.tensor_map[tid]
            ij = find_antidiag_axes(t.data, atol=atol)

            # tensor not anti-diagonal
            if ij is None:
                continue

            # work out which, if any, index to flip
            i, j = ij
            ix_i, ix_j = t.inds[i], t.inds[j]
            if ix_i in output_inds:
                if ix_j in output_inds:
                    # both are output indices, don't flip
                    continue
                # don't flip i as it is an output index
                ix_flip = ix_j
            else:
                ix_flip = ix_i

            # can get caught in loop unless we only flip once
            if ix_flip in done:
                continue

            # only flip one index
            tn.flip_([ix_flip])
            done.add(ix_flip)
            queue.append(tid)

        return tn

    antidiag_gauge_ = functools.partialmethod(antidiag_gauge, inplace=True)

    def column_reduce(self, inplace=False, output_inds=None, atol=1e-12):
        """Find bonds on this tensor network which have tensors where all but
        one column (of the respective index) is non-zero, allowing the
        'cutting' of that bond.

        Parameters
        ----------
        inplace, bool, optional
            Whether to perform the column reductions inplace.
        output_inds : sequence of str, optional
            Which indices to explicitly consider as outer legs of the tensor
            network and thus not slice. If not given, these will be taken as
            all the indices that appear once.
        atol : float, optional
            When identifying singlet column tensors, the absolute tolerance
            with which to compare to zero with.

        Returns
        -------
        TensorNetwork

        See Also
        --------
        full_simplify, rank_simplify, diagonal_reduce, antidiag_gauge
        """
        tn = self if inplace else self.copy()

        if output_inds is None:
            output_inds = set(self.outer_inds())

        queue = list(tn.tensor_map)
        while queue:
            tid = queue.pop()
            t = tn.tensor_map[tid]
            ax_i = find_columns(t.data, atol=atol)

            # not singlet columns
            if ax_i is None:
                continue

            ax, i = ax_i
            ind = t.inds[ax]

            # don't want to modify 'outer' shape of TN
            if ind in output_inds:
                continue

            tn.isel_({ind: i})
            queue.append(tid)

        return tn

    column_reduce_ = functools.partialmethod(column_reduce, inplace=True)

    def full_simplify(self, seq='DRAC', inplace=False, output_inds=None,
                      atol=1e-12, **rank_simplify_opts):
        """Perform a series of tensor network 'simplifications' in a loop until
        there is no more reduction in the number of tensors or indices. Note
        that apart from rank-reduction, the simplification methods make use of
        the non-zero structure of the tensors, and thus changes to this will
        potentially produce different simplifications.

        Parameters
        ----------
        seq : str, optional
            Which simplifications and which order to perform them in.

                * ``'D'`` : stands for ``diagonal_reduce``
                * ``'R'`` : stands for ``rank_simplify``
                * ``'A'`` : stands for ``antidiag_gauge``
                * ``'C'`` : stands for ``column_reduce``

            If you want to keep the tensor network 'simple', i.e. with no
            hyperedges, then don't use ``'D'`` (moreover ``'A'`` is redundant).
        inplace : bool, optional
            Whether to perform the simplification inplace.
        output_inds : sequence of str, optional
            Explicitly set which indices of the tensor network are output
            indices and thus should not be modified.
        atol : float, optional
            The absolute tolerance when indentifying zero entries of tensors.
        rank_simplify_opts
            Supplied to ``rank_simplify``.

        Returns
        -------
        TensorNetwork

        See Also
        --------
        diagonal_reduce, rank_simplify, antidiag_gauge, column_reduce
        """
        tn = self if inplace else self.copy()
        tn.squeeze_()

        # all the methods
        if output_inds is None:
            output_inds = self.outer_inds()

        # for the index trick reductions, faster to supply set
        ix_o = set(output_inds)

        # keep simplifying until the number of tensors and indices equalizes
        old_nt, old_ni = -1, -1
        nt, ni = tn.num_tensors, tn.num_indices
        while (nt, ni) != (old_nt, old_ni):
            for meth in seq:
                if meth == 'D':
                    tn.diagonal_reduce_(output_inds=ix_o, atol=atol)
                elif meth == 'R':
                    tn.rank_simplify_(output_inds=output_inds,
                                      **rank_simplify_opts)
                elif meth == 'A':
                    tn.antidiag_gauge_(output_inds=ix_o, atol=atol)
                elif meth == 'C':
                    tn.column_reduce_(output_inds=ix_o, atol=atol)
                else:
                    raise ValueError(f"'{meth}' is not a valid simplify type.")

            old_nt, old_ni = nt, ni
            nt, ni = tn.num_tensors, tn.num_indices

        return tn

    full_simplify_ = functools.partialmethod(full_simplify, inplace=True)

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
        """The dtype of this TensorNetwork, this is the minimal common type
        of all the tensors data.
        """
        return common_type(*self)

    def iscomplex(self):
        return iscomplex(self)

    def astype(self, dtype, inplace=False):
        """Convert the type of all tensors in this network to ``dtype``.
        """
        TN = self if inplace else self.copy()
        for t in TN:
            t.astype(dtype, inplace=True)
        return TN

    astype_ = functools.partialmethod(astype, inplace=True)

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
        rep = f"<{self.__class__.__name__}("
        rep += f"tensors={self.num_tensors}"
        rep += f", indices={self.num_indices}"
        if self.structure:
            rep += f", structure='{self.structure}', nsites={self.nsites}"

        return rep + ")>"

    graph = graph


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
                ix_sz = dict(concat((zip(t.inds, t.shape) for t in tns)))
                ldims = tuple(ix_sz[i] for i in left_inds)
                rdims = tuple(ix_sz[i] for i in right_inds)

        self.left_inds, self.right_inds = left_inds, right_inds
        self.ldims, ld = ldims, prod(ldims)
        self.rdims, rd = rdims, prod(rdims)
        self.tags = utup_union((t.inds for t in self._tensors))

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
        self._contractors = dict()

        super().__init__(dtype=self._tensors[0].dtype, shape=(ld, rd))

    def _matvec(self, vec):
        in_data = reshape(vec, self.rdims)

        if self.is_conj:
            in_data = conj(in_data)

        # cache the contractor
        if 'matvec' not in self._contractors:
            # generate a expression that acts directly on the data
            iT = Tensor(in_data, inds=self.right_inds)
            self._contractors['matvec'] = tensor_contract(
                *self._tensors, iT, output_inds=self.left_inds, **self._kws)

        fn = self._contractors['matvec']
        out_data = fn(*self._ins, in_data, backend=self.backend)

        if self.is_conj:
            out_data = conj(out_data)

        return out_data.ravel()

    def _matmat(self, mat):
        d = mat.shape[-1]
        in_data = reshape(mat, (*self.rdims, d))

        if self.is_conj:
            in_data = conj(in_data)

        # for matmat need different contraction scheme for different d sizes
        key = f"matmat_{d}"

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
            out_data = conj(out_data)

        return reshape(out_data, (-1, d))

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

    def to_dense(self, *inds_seq, **contract_opts):
        """Convert this TNLinearOperator into a dense array, defaulting to
        grouping the left and right indices respectively.
        """
        if self.is_conj:
            ts = (t.conj() for t in self._tensors)
        else:
            ts = self._tensors

        if not inds_seq:
            inds_seq = self.left_inds, self.right_inds

        return tensor_contract(*ts, **contract_opts).to_dense(*inds_seq)

    @functools.wraps(tensor_split)
    def split(self, **kwargs):
        return tensor_split(self, left_inds=self.left_inds,
                            right_inds=self.right_inds, **kwargs)

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
        self.tags = self.tn.tags

        # conjugate inputs/ouputs rather all tensors if necessary
        self.is_conj = is_conj
        self.is_trans = is_trans
        self._conj_linop = None
        self._adjoint_linop = None
        self._transpose_linop = None

        super().__init__(dtype=self.tn.dtype, shape=(ld, rd))

    def _matvec(self, vec):
        in_data = reshape(vec, self.rdims)

        if self.is_conj:
            in_data = conj(in_data)

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
            out_data = conj(out_data)

        return out_data

    def _matmat(self, mat):
        d = mat.shape[-1]
        in_data = reshape(mat, (*self.rdims, d))

        if self.is_conj:
            in_data = conj(in_data)

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
        out_data = reshape(out_T.transpose_(*out_ix).data, (-1, d))
        if self.is_conj:
            out_data = conj(out_data)

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


class PTensor(Tensor):
    """A tensor whose data array is lazily generated from a set of parameters
    and a function.

    Parameters
    ----------
    fn : callable
        The function that generates the tensor data from ``params``.
    params : sequence of numbers
        The initial parameters supplied to the generating function like
        ``fn(params)``.
    inds : optional
        Should match the shape of ``fn(params)``,
        see :class:`~quimb.tensor.tensor_core.Tensor`.
    tags : optional
        See :class:`~quimb.tensor.tensor_core.Tensor`.
    left_inds : optional
        See :class:`~quimb.tensor.tensor_core.Tensor`.
    conj : optional
        Whether to treat this ``Tensor`` as lazily conjugated, such that when
        the ``.data`` attribute is accessed it will be conjugated as well.

    See Also
    --------
    PTensor
    """

    def __init__(self, fn, params, inds=(), tags=None,
                 left_inds=None, conj=False):
        super().__init__(
            PArray(fn, params), inds=inds, tags=tags, left_inds=left_inds)
        self.is_conj = conj

    @classmethod
    def from_parray(cls, parray, *args, **kwargs):
        return cls(parray.fn, parray.params, *args, **kwargs)

    def copy(self):
        """Copy this parametrized tensor.
        """
        return PTensor(
            fn=self.fn,
            params=self.params,
            inds=self.inds,
            tags=self.tags,
            left_inds=self.left_inds,
            conj=self.is_conj,
        )

    @property
    def _data(self):
        """Make ``_data`` read-only and handle conjugation lazily.
        """
        data_fn_params = self._parray.data
        if self.is_conj:
            return conj(data_fn_params)
        return data_fn_params

    @_data.setter
    def _data(self, x):
        if not isinstance(x, PArray):
            raise ValueError(
                "You can only update the data of a ``PTensor`` with an "
                "``PArray``. Alternatively you can convert this ``PTensor to "
                "a normal ``Tensor`` with ``t.unparametrize()``")
        self.is_conj = False
        self._parray = x

    @property
    def data(self):
        return self._data

    @property
    def fn(self):
        return self._parray.fn

    @property
    def params(self):
        return self._parray.params

    @params.setter
    def params(self, x):
        self._parray.params = x

    def conj(self, inplace=False):
        """Conjugate this parametrized tensor - done lazily whenever the
        ``.data`` attribute is accessed.
        """
        t = self if inplace else self.copy()
        t.is_conj = not self.is_conj
        return t

    conj_ = functools.partialmethod(conj, inplace=True)

    def unparametrize(self):
        """Turn this PTensor into a normal Tensor.
        """
        return Tensor(self)
