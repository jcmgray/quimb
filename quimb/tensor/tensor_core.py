"""Core tensor network tools.
"""
import os
import copy
import uuid
import math
import string
import weakref
import operator
import functools
import itertools
import threading
import contextlib
import collections
from numbers import Integral

import numpy as np
import opt_einsum as oe
import scipy.sparse.linalg as spla
from autoray import (do, conj, reshape, transpose, astype,
                     infer_backend, get_dtype_name, dag)

from ..core import (qarray, prod, realify_scalar, vdot, make_immutable)
from ..utils import (check_opt, oset, concat, frequencies, unique,
                     valmap, ensure_dict, LRU, gen_bipartitions)
from ..gen.rand import randn, seed_rand, rand_matrix, rand_uni
from . import decomp
from .array_ops import (iscomplex, norm_fro, unitize, ndim, asarray, PArray,
                        find_diag_axes, find_antidiag_axes, find_columns)
from .drawing import draw_tn


try:
    from autoray import get_common_dtype
except ImportError:
    from ..core import common_type as get_common_dtype


_CONTRACT_STRATEGY = 'greedy'
_TEMP_CONTRACT_STRATEGIES = collections.defaultdict(list)


def get_contract_strategy():
    r"""Get the default contraction strategy - the option supplied as
    ``optimize`` to ``opt_einsum``.
    """
    if not _TEMP_CONTRACT_STRATEGIES:
        # shortcut for when no temp strategies are in use
        return _CONTRACT_STRATEGY

    thread_id = threading.get_ident()
    if thread_id not in _TEMP_CONTRACT_STRATEGIES:
        return _CONTRACT_STRATEGY

    temp_strategies = _TEMP_CONTRACT_STRATEGIES[thread_id]
    # empty list -> not in context manager -> use default strategy
    if not temp_strategies:
        # clean up to allow above shortcuts
        del _TEMP_CONTRACT_STRATEGIES[thread_id]
        return _CONTRACT_STRATEGY

    # use most recently set strategy for this threy
    return temp_strategies[-1]


def set_contract_strategy(strategy):
    """Get the default contraction strategy - the option supplied as
    ``optimize`` to ``opt_einsum``.
    """
    global _CONTRACT_STRATEGY
    _CONTRACT_STRATEGY = strategy


@contextlib.contextmanager
def contract_strategy(strategy, set_globally=False):
    """A context manager to temporarily set the default contraction strategy
    supplied as ``optimize`` to ``opt_einsum``. By default, this only sets the
    contract strategy for the current thread.

    Parameters
    ----------
    set_globally : bool, optimize
        Whether to set the strategy just for this thread, or for all threads.
        If you are entering the context, *then* using multithreading, you might
        want ``True``.
    """
    if set_globally:
        orig_strategy = get_contract_strategy()
        set_contract_strategy(strategy)
        try:
            yield
        finally:
            set_contract_strategy(orig_strategy)
    else:
        thread_id = threading.get_ident()
        temp_strategies = _TEMP_CONTRACT_STRATEGIES[thread_id]
        temp_strategies.append(strategy)
        try:
            yield
        finally:
            temp_strategies.pop()


def _get_contract_path(eq, *shapes, **kwargs):
    """Get the contraction path - sequence of integer pairs.
    """

    # construct the internal opt_einsum data
    lhs, output = eq.split('->')
    inputs = lhs.split(',')

    # nothing to optimize in this case
    nterms = len(inputs)
    if nterms <= 2:
        return (tuple(range(nterms)),)

    size_dict = {}
    for ix, d in zip(concat(inputs), concat(shapes)):
        size_dict[ix] = d

    # get the actual path generating function
    optimize = kwargs.pop('optimize', get_contract_strategy())
    if isinstance(optimize, str):
        optimize = oe.paths.get_path_fn(optimize)

    kwargs.setdefault('memory_limit', None)

    # this way we get to avoid constructing the full PathInfo object
    path = optimize(inputs, output, size_dict, **kwargs)
    return tuple(path)


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


def _get_contraction(eq, shapes, optimize, cache, get, **kwargs):
    # don't cache path if using a 'single-shot' path-optimizer
    #     (you may want to run these several times, each time improving path)
    cache_path = cache and not isinstance(optimize, oe.paths.PathOptimizer)

    # get the path, unless explicitly given already
    if not isinstance(optimize, tuple):
        path_fn = _CONTRACT_FNS['path', cache_path]
        path = path_fn(eq, *shapes, optimize=optimize, **kwargs)
    else:
        path = optimize

    if get == 'path':
        return path

    if get == 'expr':
        expr_fn = _CONTRACT_FNS['expr', cache]
        expr = expr_fn(eq, *shapes, optimize=path, **kwargs)
        return expr

    if get == 'info':
        info_fn = _CONTRACT_FNS['info', cache]
        info = info_fn(eq, *shapes, optimize=path, **kwargs)
        return info


def get_contraction(eq, *shapes, cache=True, get='expr',
                    optimize=None, **kwargs):
    """Get an callable expression that will evaluate ``eq`` based on
    ``shapes``. Cache the result if no constant tensors are involved.
    """
    if optimize is None:
        optimize = get_contract_strategy()

    # can't cache if using constants
    if 'constants' in kwargs:
        expr_fn = _CONTRACT_FNS['expr', False]
        expr = expr_fn(eq, *shapes, optimize=optimize, **kwargs)
        return expr

    # make sure shapes are hashable + concrete python ints
    if not (
        isinstance(shapes[0], tuple) and
        isinstance(next(concat(shapes), 1), int)
    ):
        shapes = tuple(tuple(map(int, s)) for s in shapes)

    # make sure explicit paths are hashable
    if isinstance(optimize, list):
        optimize = tuple(optimize)

    return _get_contraction(eq, shapes, optimize, cache, get, **kwargs)


try:
    from opt_einsum.contract import infer_backend as _oe_infer_backend
    del _oe_infer_backend
    _CONTRACT_BACKEND = 'auto'
    _TENSOR_LINOP_BACKEND = 'auto'
except ImportError:
    _CONTRACT_BACKEND = 'numpy'
    _TENSOR_LINOP_BACKEND = 'numpy'


_TEMP_CONTRACT_BACKENDS = collections.defaultdict(list)
_TEMP_TENSOR_LINOP_BACKENDS = collections.defaultdict(list)


def get_contract_backend():
    """Get the default backend used for tensor contractions, via 'opt_einsum'.

    See Also
    --------
    set_contract_backend, get_tensor_linop_backend, set_tensor_linop_backend,
    tensor_contract
    """
    if not _TEMP_CONTRACT_BACKENDS:
        return _CONTRACT_BACKEND

    thread_id = threading.get_ident()
    if thread_id not in _TEMP_CONTRACT_BACKENDS:
        return _CONTRACT_BACKEND

    temp_backends = _TEMP_CONTRACT_BACKENDS[thread_id]
    if not temp_backends:
        del _TEMP_CONTRACT_BACKENDS[thread_id]
        return _CONTRACT_BACKEND

    return temp_backends[-1]


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
def contract_backend(backend, set_globally=False):
    """A context manager to temporarily set the default backend used for tensor
    contractions, via 'opt_einsum'. By default, this only sets the contract
    backend for the current thread.

    Parameters
    ----------
    set_globally : bool, optimize
        Whether to set the backend just for this thread, or for all threads. If
        you are entering the context, *then* using multithreading, you might
        want ``True``.
    """
    if set_globally:
        orig_backend = get_contract_backend()
        set_contract_backend(backend)
        try:
            yield
        finally:
            set_contract_backend(orig_backend)
    else:
        thread_id = threading.get_ident()
        temp_backends = _TEMP_CONTRACT_BACKENDS[thread_id]
        temp_backends.append(backend)
        try:
            yield
        finally:
            temp_backends.pop()


def get_tensor_linop_backend():
    """Get the default backend used for tensor network linear operators, via
    'opt_einsum'. This is different from the default contraction backend as
    the contractions are likely repeatedly called many times.

    See Also
    --------
    set_tensor_linop_backend, set_contract_backend, get_contract_backend,
    TNLinearOperator
    """
    if not _TEMP_TENSOR_LINOP_BACKENDS:
        return _TENSOR_LINOP_BACKEND

    thread_id = threading.get_ident()
    if thread_id not in _TEMP_TENSOR_LINOP_BACKENDS:
        return _TENSOR_LINOP_BACKEND

    temp_backends = _TEMP_TENSOR_LINOP_BACKENDS[thread_id]
    if not temp_backends:
        del _TEMP_TENSOR_LINOP_BACKENDS[thread_id]
        return _TENSOR_LINOP_BACKEND

    return temp_backends[-1]


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
def tensor_linop_backend(backend, set_globally=False):
    """A context manager to temporarily set the default backend used for tensor
    network linear operators, via 'opt_einsum'. By default, this
    only sets the contract backend for the current thread.

    Parameters
    ----------
    set_globally : bool, optimize
        Whether to set the backend just for this thread, or for all threads. If
        you are entering the context, *then* using multithreading, you might
        want ``True``.
    """
    if set_globally:
        orig_backend = get_tensor_linop_backend()
        set_tensor_linop_backend(backend)
        try:
            yield
        finally:
            set_tensor_linop_backend(orig_backend)
    else:
        thread_id = threading.get_ident()
        temp_backends = _TEMP_TENSOR_LINOP_BACKENDS[thread_id]
        temp_backends.append(backend)
        try:
            yield
        finally:
            temp_backends.pop()


# --------------------------------------------------------------------------- #
#                                Tensor Funcs                                 #
# --------------------------------------------------------------------------- #

def oset_union(xs):
    """Non-variadic ordered set union taking any sequence of iterables.
    """
    return oset(concat(xs))


def oset_intersection(xs):
    return functools.reduce(operator.and_, xs)


def tags_to_oset(tags):
    """Parse a ``tags`` argument into an ordered set.
    """
    if tags is None:
        return oset()
    elif isinstance(tags, (str, int)):
        return oset((tags,))
    elif isinstance(tags, oset):
        return tags.copy()
    else:
        return oset(tags)


def sortedtuple(x):
    return tuple(sorted(x))


def _gen_output_inds(all_inds):
    """Generate the output, i.e. unique, indices from the set ``inds``. Raise
    if any index found more than twice.
    """
    for ind, freq in frequencies(all_inds).items():
        if freq > 2:
            raise ValueError(
                f"The index {ind} appears more than twice! If this is "
                "intentionally a 'hyper' tensor network you will need to "
                "explicitly supply `output_inds` when contracting for example."
            )
        elif freq == 1:
            yield ind


@functools.lru_cache(2**12)
def get_symbol(i):
    """Get the 'ith' symbol.
    """
    return oe.get_symbol(i)


def empty_symbol_map():
    """Get a default dictionary that will populate with symbol entries as they
    are accessed.
    """
    return collections.defaultdict(map(get_symbol, itertools.count()).__next__)


@functools.lru_cache(2**12)
def _inds_to_eq(inputs, output):
    """Turn input and output indices of any sort into a single 'equation'
    string where each index is a single 'symbol' (unicode character).

    Parameters
    ----------
    inputs : sequence of sequence of str
        The input indices per tensor.
    output : sequence of str
        The output indices.

    Returns
    -------
    eq : str
        The string to feed to einsum/contract.
    """
    symbol_get = empty_symbol_map().__getitem__
    in_str = ("".join(map(symbol_get, inds)) for inds in inputs)
    out_str = "".join(map(symbol_get, output))
    return ",".join(in_str) + f"->{out_str}"


_VALID_CONTRACT_GET = {None, 'expression', 'path', 'path-info', 'symbol-map'}


def tensor_contract(
    *tensors,
    output_inds=None,
    get=None,
    backend=None,
    preserve_tensor=False,
    **contract_opts
):
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

    backend : {'auto', 'numpy', 'jax', 'cupy', 'tensorflow', ...}, optional
        Which backend to use to perform the contraction. Must be a valid
        ``opt_einsum`` backend with the relevant library installed.
    preserve_tensor : bool, optional
        Whether to return a tensor regardless of whether the output object
        is a scalar (has no indices) or not.
    contract_opts
        Passed to ``opt_einsum.contract_expression`` or
        ``opt_einsum.contract_path``.

    Returns
    -------
    scalar or Tensor
    """
    if backend is None:
        backend = get_contract_backend()

    inds_i = tuple(t.inds for t in tensors)  # input indices per tensor

    if output_inds is None:
        # sort output indices by input order for efficiency and consistency
        inds_out = tuple(_gen_output_inds(concat(inds_i)))
    else:
        inds_out = tuple(output_inds)

    # possibly map indices into the range needed by opt-einsum
    eq = _inds_to_eq(inds_i, inds_out)

    if get is not None:
        check_opt('get', get, _VALID_CONTRACT_GET)

        if get == 'symbol-map':
            return {
                get_symbol(i): ix
                for i, ix in enumerate(unique(concat(inds_i)))
            }

        if get == 'path':
            ops = (t.shape for t in tensors)
            return get_contraction(eq, *ops, get='path', **contract_opts)

        if get == 'path-info':
            ops = (t.shape for t in tensors)
            path_info = get_contraction(eq, *ops, get='info', **contract_opts)
            path_info.quimb_symbol_map = {
                get_symbol(i): ix
                for i, ix in enumerate(unique(concat(inds_i)))
            }
            return path_info

        if get == 'expression':
            # account for possible constant tensors
            cnst = contract_opts.get('constants', ())
            ops = (t.data if i in cnst else t.shape
                   for i, t in enumerate(tensors))
            expression = get_contraction(eq, *ops, **contract_opts)
            return expression

    # perform the contraction
    shapes = (t.shape for t in tensors)
    expression = get_contraction(eq, *shapes, **contract_opts)
    o_array = expression(*(t.data for t in tensors), backend=backend)

    if not inds_out and not preserve_tensor:
        if isinstance(o_array, np.ndarray):
            o_array = realify_scalar(o_array.item(0))
        return o_array

    # union of all tags
    o_tags = oset.union(*(t.tags for t in tensors))

    return Tensor(data=o_array, inds=inds_out, tags=o_tags)


# generate a random base to avoid collisions on difference processes ...
_RAND_PREFIX = str(uuid.uuid4())[:6]
# but then make the list orderable to help contraction caching
_RAND_ALPHABET = string.ascii_uppercase + string.ascii_lowercase
RAND_UUIDS = map(
    "".join,
    itertools.chain.from_iterable(
        itertools.product(_RAND_ALPHABET, repeat=repeat)
        for repeat in itertools.count(5)
    )
)


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
    return f"{base}_{_RAND_PREFIX}{next(RAND_UUIDS)}"


_VALID_SPLIT_GET = {None, 'arrays', 'tensors', 'values'}
_SPLIT_FNS = {
    'svd': decomp.svd,
    'eig': decomp.eig,
    'qr': decomp.qr,
    'lq': decomp.lq,
    'eigh': decomp.eigh,
    'cholesky': decomp.cholesky,
    'isvd': decomp.isvd,
    'svds': decomp.svds,
    'rsvd': decomp.rsvd,
    'eigsh': decomp.eigsh,
}
_SPLIT_VALUES_FNS = {'svd': decomp.svdvals, 'eig': decomp.svdvals_eig}
_FULL_SPLIT_METHODS = {'svd', 'eig', 'eigh'}
_RANK_HIDDEN_METHODS = {'qr', 'lq', 'cholesky'}
_DENSE_ONLY_METHODS = {'svd', 'eig', 'eigh', 'cholesky', 'qr', 'lq'}
_ISOM_METHODS = {'svd', 'eig', 'eigh', 'isvd', 'svds', 'rsvd', 'eigsh'}
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
        a canonical environment, corresponding to maintaining the frobenius
        or nuclear norm. If ``None`` (the default) then this is automatically
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
        left_inds = oset(T.inds) - oset(right_inds)
    else:
        left_inds = tags_to_oset(left_inds)

    if right_inds is None:
        right_inds = oset(T.inds) - oset(left_inds)
    else:
        right_inds = tags_to_oset(right_inds)

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
        return _SPLIT_VALUES_FNS[method](array)

    opts = _parse_split_opts(
        method, cutoff, absorb, max_bond, cutoff_mode, renorm)

    # ``s`` itself will be None unless ``absorb=None`` is specified
    left, s, right = _SPLIT_FNS[method](array, **opts)
    left = reshape(left, (*left_dims, -1))
    right = reshape(right, (-1, *right_dims))

    if get == 'arrays':
        if absorb is None:
            return left, s, right
        return left, right

    bond_ind = rand_uuid() if bond_ind is None else bond_ind
    ltags = T.tags | tags_to_oset(ltags)
    rtags = T.tags | tags_to_oset(rtags)

    Tl = Tensor(data=left, inds=(*left_inds, bond_ind), tags=ltags)
    Tr = Tensor(data=right, inds=(bond_ind, *right_inds), tags=rtags)

    if absorb is None:
        stags = T.tags | tags_to_oset(stags)
        Ts = Tensor(data=s, inds=(bond_ind,), tags=stags)
        tensors = (Tl, Ts, Tr)
    else:
        tensors = (Tl, Tr)

    # work out if we have created left and/or right isometric tensors
    left_isom = ((method == 'qr') or (method in _ISOM_METHODS and
                                      absorb in (None, 'right')))
    right_isom = ((method == 'lq') or (method in _ISOM_METHODS and
                                       absorb in (None, 'left')))
    if left_isom:
        Tl.modify(left_inds=left_inds)
    if right_isom:
        Tr.modify(left_inds=right_inds)

    if get == 'tensors':
        return tensors

    return TensorNetwork(tensors, virtual=True)


def tensor_canonize_bond(
    T1,
    T2,
    absorb='right',
    gauges=None,
    gauge_smudge=1e-6,
    **split_opts
):
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
    absorb : {'right', 'left', 'both', None}, optional
        Which tensor to effectively absorb the singular values into.
    split_opts
        Supplied to :func:`~quimb.tensor.tensor_core.tensor_split`, with
        modified defaults of ``method=='qr'`` and ``absorb='right'``.
    """
    check_opt('absorb', absorb, ('left', 'both', 'right'))

    if absorb == 'both':
        # same as doing reduced compression with no truncation
        split_opts.setdefault('cutoff', 0.0)
        return tensor_compress_bond(
            T1, T2, gauges=gauges, gauge_smudge=gauge_smudge, **split_opts)

    split_opts.setdefault('method', 'qr')
    if absorb == 'left':
        T1, T2 = T2, T1

    lix, bix, _ = tensor_make_single_bond(T1, T2, gauges=gauges)
    if not bix:
        raise ValueError("The tensors specified don't share an bond.")

    if (T1.left_inds is not None) and set(T1.left_inds) == set(lix):
        # tensor is already isometric with respect to shared bonds
        return

    if gauges is not None:
        # gauge outer and inner but only revert outer
        absorb = None
        tn = T1 | T2
        outer, _ = tn.gauge_simple_insert(gauges, smudge=gauge_smudge)
        gauges.pop(bix, None)

    new_T1, tRfact = T1.split(lix, get='tensors', **split_opts)
    new_T2 = tRfact @ T2

    new_T1.transpose_like_(T1)
    new_T2.transpose_like_(T2)

    T1.modify(data=new_T1.data, left_inds=lix)
    T2.modify(data=new_T2.data)

    if gauges is not None:
        tn.gauge_simple_remove(outer=outer)


def tensor_compress_bond(
    T1,
    T2,
    reduced=True,
    absorb='both',
    gauges=None,
    gauge_smudge=1e-6,
    info=None,
    **compress_opts
):
    r"""Inplace compress between the two single tensors. It follows the
    following steps to minimize the size of SVD performed::

        a)│   │        b)│        │        c)│       │
        ━━●━━━●━━  ->  ━━>━━○━━○━━<━━  ->  ━━>━━━M━━━<━━
          │   │          │  ....  │          │       │
         <*> <*>          contract              <*>
         QR   LQ            -><-                SVD

                  d)│            │        e)│   │
              ->  ━━>━━━ML──MR━━━<━━  ->  ━━●───●━━
                    │....    ....│          │   │
                  contract  contract          ^compressed bond
                    -><-      -><-

    Parameters
    ----------
    T1 : Tensor
        The left tensor.
    T2 : Tensor
        The right tensor.
    max_bond : int or None, optional
        The maxmimum bond dimension.
    cutoff : float, optional
        The singular value cutoff to use.
    reduced : bool, optional
        Whether to perform the QR reduction as above or not.
    absorb : {'both', 'left', 'right', None}, optional
        Where to absorb the singular values after decomposition.
    info : None or dict, optional
        A dict for returning extra information such as the singular values.
    compress_opts :
        Supplied to :func:`~quimb.tensor.tensor_core.tensor_split`.
    """

    lix, bix, rix = tensor_make_single_bond(T1, T2, gauges=gauges)
    if not bix:
        raise ValueError("The tensors specified don't share an bond.")

    if gauges is not None:
        absorb = None
        tn = T1 | T2
        outer, _ = tn.gauge_simple_insert(gauges, smudge=gauge_smudge)

    if reduced is True:
        # a) -> b)
        T1_L, T1_R = T1.split(
            left_inds=lix, right_inds=bix,
            get='tensors', method='qr')
        T2_L, T2_R = T2.split(
            left_inds=bix, right_inds=rix,
            get='tensors', method='lq')

        # b) -> c)
        M = T1_R @ T2_L
        # c) -> d)
        M_L, *s, M_R = M.split(
            left_inds=T1_L.bonds(M), bond_ind=bix,
            get='tensors', absorb=absorb, **compress_opts)

        # d) -> e)
        T1C = T1_L.contract(M_L, output_inds=T1.inds)
        T2C = M_R.contract(T2_R, output_inds=T2.inds)

    elif reduced == 'lazy':
        compress_opts.setdefault('method', 'isvd')
        T12 = TNLinearOperator((T1, T2), lix, rix)
        T1C, *s, T2C = T12.split(get='tensors', absorb=absorb, **compress_opts)
        T1C.transpose_like_(T1)
        T2C.transpose_like_(T2)

    else:
        T12 = T1 @ T2
        T1C, *s, T2C = T12.split(left_inds=lix, get='tensors',
                                 absorb=absorb, **compress_opts)
        T1C.transpose_like_(T1)
        T2C.transpose_like_(T2)

    # update with the new compressed data
    T1.modify(data=T1C.data)
    T2.modify(data=T2C.data)

    if absorb == 'right':
        T1.modify(left_inds=lix)
    elif absorb == 'left':
        T2.modify(left_inds=rix)

    if s and info is not None:
        info['singular_values'] = s[0].data

    if gauges is not None:
        tn.gauge_simple_remove(outer=outer)
        g = s[0].data
        fact = g[0]
        g = g / fact
        gauges[bix] = g
        fact_1_2 = fact**0.5
        T1 *= fact_1_2
        T2 *= fact_1_2


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


def tensor_make_single_bond(t1, t2, gauges=None):
    """If two tensors share multibonds, fuse them together and return the left
    indices, bond if it exists, and right indices. Handles simple ``gauges``.
    """
    left, shared, right = group_inds(t1, t2)
    nshared = len(shared)

    if nshared == 0:
        return left, None, right

    bond = next(iter(shared))
    if nshared > 1:
        t1.fuse_({bond: shared})
        t2.fuse_({bond: shared})
        if gauges is not None:
            gauges[bond] = functools.reduce(
                lambda x, y: do("kron", x, y),
                (gauges.pop(ix) for ix in shared)
            )

    return left, bond, right


def tensor_fuse_squeeze(t1, t2, squeeze=True, gauges=None):
    """If ``t1`` and ``t2`` share more than one bond fuse it, and if the size
    of the shared dimenion(s) is 1, squeeze it. Inplace operation.
    """
    _, ind0, _ = tensor_make_single_bond(t1, t2, gauges=gauges)

    if squeeze and t1.ind_size(ind0) == 1:
        t1.squeeze_(include=(ind0,))
        t2.squeeze_(include=(ind0,))

        if gauges is not None:
            s0_1_2 = gauges.pop(ind0).item()**0.5
            t1 *= s0_1_2
            t2 *= s0_1_2


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


def rand_padder(vector, pad_width, iaxis, kwargs):
    """Helper function for padding tensor with random entries.
    """
    rand_strength = kwargs.get('rand_strength')
    if pad_width[0]:
        vector[:pad_width[0]] = rand_strength * randn(pad_width[0],
                                                      dtype='float32')
    if pad_width[1]:
        vector[-pad_width[1]:] = rand_strength * randn(pad_width[1],
                                                       dtype='float32')
    return vector


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

    new_data = array_direct_product(T1.data, T2.data, sum_axes=sum_axes)
    new_T.modify(data=new_data)

    return new_T


def tensor_network_sum(tnA, tnB):
    """Sum of two tensor networks, whose indices should match exactly, using
    direct products.

    Parameters
    ----------
    tnA : TensorNetwork
        The first tensor network.
    tnB : TensorNetwork
        The second tensor network.

    Returns
    -------
    TensorNetwork
        The sum of ``tnA`` and ``tnB``, with increased bond dimensions.
    """
    oix = tnA.outer_inds()

    ts = []
    for t1, t2 in zip(tnA, tnB):

        if set(t1.inds) != set(t2.inds):
            raise ValueError("Can only sum TNs with exactly matching indices.")

        sum_inds = [ix for ix in t1.inds if ix in oix]
        ts.append(tensor_direct_product(t1, t2, sum_inds))

    return TensorNetwork(ts).view_like_(tnA)


def bonds(t1, t2):
    """Getting any indices connecting the Tensor(s) or TensorNetwork(s) ``t1``
    and ``t2``.
    """
    if isinstance(t1, Tensor):
        ix1 = oset(t1.inds)
    else:
        ix1 = oset_union(t.inds for t in t1)

    if isinstance(t2, Tensor):
        ix2 = oset(t2.inds)
    else:
        ix2 = oset_union(t.inds for t in t2)

    return ix1 & ix2


def bonds_size(t1, t2):
    """Get the size of the bonds linking tensors or tensor networks ``t1`` and
    ``t2``.
    """
    return prod(t1.ind_size(ix) for ix in bonds(t1, t2))


def group_inds(t1, t2):
    """Group bonds into left only, shared, and right only.
    """
    left_inds, shared_inds, right_inds = [], [], []

    for ix in t1.inds:
        if ix in t2.inds:
            shared_inds.append(ix)
        else:
            left_inds.append(ix)
    for ix in t2.inds:
        if ix not in shared_inds:
            right_inds.append(ix)

    return left_inds, shared_inds, right_inds


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

    return oset_union(t.tags for t in ts)


def maybe_unwrap(t, preserve_tensor=False, equalize_norms=False):
    """Maybe unwrap a ``TensorNetwork`` or ``Tensor`` into a ``Tensor`` or
    scalar, depending on how many tensors and indices it has.
    """
    if isinstance(t, TensorNetwork):
        if equalize_norms is True:
            # this also redistributes the any collected norm exponent
            t.equalize_norms_()
        if t.num_tensors != 1:
            return t
        t, = t.tensor_map.values()
    if preserve_tensor or t.ndim != 0:
        return t
    return t.data


def tensor_network_distance(
    tnA,
    tnB,
    xAA=None,
    xAB=None,
    xBB=None,
    method='auto',
    **contract_opts,
):
    r"""Compute the Frobenius norm distance between two tensor networks:

    .. math::

            D(A, B)
            = | A - B |_{\mathrm{fro}}
            = \mathrm{Tr} [(A - B)^{\dagger}(A - B)]^{1/2}
            = ( \langle A | A \rangle - 2 \mathrm{Re} \langle A | B \rangle|
            + \langle B | B \rangle ) ^{1/2}

    which should have a matching external indices.

    Parameters
    ----------
    tnA : TensorNetwork or Tensor
        The first tensor network operator.
    tnB : TensorNetwork or Tensor
        The second tensor network operator.
    xAA : None or scalar
        The value of ``A.H @ A`` if you already know it (or it doesn't matter).
    xAB : None or scalar
        The value of ``A.H @ B`` if you already know it (or it doesn't matter).
    xBB : None or scalar
        The value of ``B.H @ B`` if you already know it (or it doesn't matter).
    method : {'auto', 'overlap', 'dense'}, optional
        How to compute the distance. If ``'overlap'``, the default, the
        distance will be computed as the sum of overlaps, without explicitly
        forming the dense operators. If ``'dense'``, the operators will be
        directly formed and the norm computed, which can be quicker when the
        exterior dimensions are small. If ``'auto'``, the dense method will
        be used if the total operator (outer) size is ``<= 2**16``.
    contract_opts
        Supplied to :meth:`~quimb.tensor.tensor_core.TensorNetwork.contract`.

    Returns
    -------
    D : float
    """
    check_opt('method', method, ('auto', 'dense', 'overlap'))

    oix = tnA.outer_inds()
    if set(oix) != set(tnB.outer_inds()):
        raise ValueError(
            "Can only fit tensor networks with matching outer indices.")

    if method == 'auto':
        d = prod(map(tnA.ind_size, oix))
        if d <= 1 << 16:
            method = 'dense'
        else:
            method = 'overlap'

    # directly form vectorizations of both
    if method == 'dense':
        A = tnA.to_dense(oix, to_qarray=False)
        B = tnB.to_dense(oix, to_qarray=False)
        return do('linalg.norm', A - B)

    # overlap method
    if xAA is None:
        xAA = (tnA | tnA.H).contract(all, **contract_opts)
    if xAB is None:
        xAB = (tnA | tnB.H).contract(all, **contract_opts)
    if xBB is None:
        xBB = (tnB | tnB.H).contract(all, **contract_opts)

    return do('abs', xAA - 2 * do('real', xAB) + xBB)**0.5


def tensor_network_fit_autodiff(
    tn,
    tn_target,
    steps=1000,
    tol=1e-9,
    autodiff_backend='autograd',
    contract_optimize='auto-hq',
    distance_method='auto',
    inplace=False,
    progbar=False,
    **kwargs
):
    """Optimize the fit of ``tn`` with respect to ``tn_target`` using
    automatic differentation. This minimizes the norm of the difference
    between the two tensor networks, which must have matching outer indices,
    using overlaps.

    Parameters
    ----------
    tn : TensorNetwork
        The tensor network to fit.
    tn_target : TensorNetwork
        The target tensor network to fit ``tn`` to.
    steps : int, optional
        The maximum number of autodiff steps.
    tol : float, optional
        The target norm distance.
    autodiff_backend : str, optional
        Which backend library to use to perform the gradient computation.
    contract_optimize : str, optional
        The contraction path optimized used to contract the overlaps.
    distance_method : {'auto', 'dense', 'overlap'}, optional
        Supplied to :func:`~quimb.tensor.tensor_core.tensor_network_distance`,
        controls how the distance is computed.
    inplace : bool, optional
        Update ``tn`` in place.
    progbar : bool, optional
        Show a live progress bar of the fitting process.
    kwargs
        Passed to :class:`~quimb.tensor.tensor_core.optimize.TNOptimizer`.

    See Also
    --------
    tensor_network_distance, tensor_network_fit_als
    """
    from .optimize import TNOptimizer

    xBB = (tn_target | tn_target.H).contract(all, optimize=contract_optimize)

    tnopt = TNOptimizer(
        tn=tn,
        loss_fn=tensor_network_distance,
        loss_constants={'tnB': tn_target, 'xBB': xBB},
        loss_kwargs={'method': distance_method, 'optimize': contract_optimize},
        autodiff_backend=autodiff_backend,
        progbar=progbar,
        **kwargs)

    tn_fit = tnopt.optimize(steps, tol=tol)

    if not inplace:
        return tn_fit

    for t1, t2 in zip(tn, tn_fit):
        t1.modify(data=t2.data)

    return tn


def tensor_network_fit_als(
    tn,
    tn_target,
    tags=None,
    steps=100,
    tol=1e-9,
    solver='solve',
    enforce_pos=False,
    pos_smudge=None,
    tnAA=None,
    tnAB=None,
    xBB=None,
    contract_optimize='greedy',
    inplace=False,
    progbar=False,
):
    """Optimize the fit of ``tn`` with respect to ``tn_target`` using
    alternating least squares (ALS). This minimizes the norm of the difference
    between the two tensor networks, which must have matching outer indices,
    using overlaps.

    Parameters
    ----------
    tn : TensorNetwork
        The tensor network to fit.
    tn_target : TensorNetwork
        The target tensor network to fit ``tn`` to.
    tags : sequence of str, optional
        If supplied, only optimize tensors matching any of given tags.
    steps : int, optional
        The maximum number of ALS steps.
    tol : float, optional
        The target norm distance.
    solver : {'solve', 'lstsq', ...}, optional
        The underlying driver function used to solve the local minimization,
        e.g. ``numpy.linalg.solve`` for ``'solve'`` with ``numpy`` backend.
    enforce_pos : bool, optional
        Whether to enforce positivity of the locally formed environments,
        which can be more stable.
    pos_smudge : float, optional
        If enforcing positivity, the level below which to clip eigenvalues
        for make the local environment positive definite.
    tnAA : TensorNetwork, optional
        If you have already formed the overlap ``tn.H & tn``, maybe
        approximately, you can supply it here. The unconjugated layer should
        have tag ``'__KET__'`` and the conjugated layer ``'__BRA__'``. Each
        tensor being optimized should have tag ``'__VAR{i}__'``.
    tnAB : TensorNetwork, optional
        If you have already formed the overlap ``tn_target.H & tn``, maybe
        approximately, you can supply it here. Each tensor being optimized
        should have tag ``'__VAR{i}__'``.
    xBB : float, optional
        If you have already know, have computed ``tn_target.H @ tn_target``,
        or it doesn't matter, you can supply the value here.
    contract_optimize : str, optional
        The contraction path optimized used to contract the local environments.
        Note ``'greedy'`` is the default in order to maximize shared work.
    inplace : bool, optional
        Update ``tn`` in place.
    progbar : bool, optional
        Show a live progress bar of the fitting process.

    Returns
    -------
    TensorNetwork

    See Also
    --------
    tensor_network_fit_autodiff, tensor_network_distance
    """
    # mark the tensors we are going to optimize
    tna = tn.copy()
    tna.add_tag('__KET__')

    if tags is None:
        to_tag = tna
    else:
        to_tag = tna.select_tensors(tags, 'any')

    tagged = []
    for i, t in enumerate(to_tag):
        var_tag = f'__VAR{i}__'
        t.add_tag(var_tag)
        tagged.append(var_tag)

    # form the norm of the varying TN (A) and its overlap with the target (B)
    if tnAA is None:
        tnAA = tna | tna.H.retag_({'__KET__': '__BRA__'})
    if tnAB is None:
        tnAB = tna | tn_target.H

    if pos_smudge is None:
        pos_smudge = max(tol, 1e-15)

    # want to cache from sweep to sweep but also not infinitely
    cachesize = len(tagged) * (tn.num_tensors + tn_target.num_tensors)
    cache = LRU(maxsize=cachesize)

    # shared intermediates + greedy = good reuse of contractions
    with oe.shared_intermediates(cache), contract_strategy(contract_optimize):

        # prepare each of the contractions we are going to repeat
        env_contractions = []
        for tg in tagged:
            # varying tensor and conjugate in norm <A|A>
            tk = tnAA['__KET__', tg]
            tb = tnAA['__BRA__', tg]

            # get inds, and ensure any bonds come last, for linalg.solve
            lix, bix, rix = group_inds(tb, tk)
            tk.transpose_(*rix, *bix)
            tb.transpose_(*lix, *bix)

            # form TNs with 'holes', i.e. environment tensors networks
            A_tn = tnAA.select((tg,), '!all')
            y_tn = tnAB.select((tg,), '!all')

            env_contractions.append((tk, tb, lix, bix, rix, A_tn, y_tn))

        if tol != 0.0:
            old_d = float('inf')
            if xBB is None:
                # compute this so tracking real norm distance is accurate
                xBB = (tn_target | tn_target.H) ^ all  # <B|B>

        if progbar:
            import tqdm
            pbar = tqdm.trange(steps)
        else:
            pbar = range(steps)

        # the main iterative sweep on each tensor, locally optimizing
        for _ in pbar:
            for (tk, tb, lix, bix, rix, A_tn, y_tn) in env_contractions:
                Ni = A_tn.to_dense(lix, rix)
                Wi = y_tn.to_dense(rix, bix)

                if enforce_pos:
                    el, ev = do('linalg.eigh', Ni)
                    el = do('clip', el, el[-1] * pos_smudge, None)
                    Ni_p = ev * do('reshape', el, (1, -1)) @ dag(ev)
                else:
                    Ni_p = Ni

                if solver == 'solve':
                    x = do('linalg.solve', Ni_p, Wi)
                elif solver == 'lstsq':
                    x = do('linalg.lstsq', Ni_p, Wi, rcond=pos_smudge)[0]

                x_r = do('reshape', x, tk.shape)
                # n.b. because we are using virtual TNs -> updates propagate
                tk.modify(data=x_r)
                tb.modify(data=do('conj', x_r))

            # assess | A - B | for convergence or printing
            if (tol != 0.0) or progbar:
                xAA = do('trace', dag(x) @ (Ni @ x))  # <A|A>
                xAB = do('trace', do('real', dag(x) @ Wi))  # <A|B>
                d = do('abs', (xAA - 2 * xAB + xBB))**0.5
                if abs(d - old_d) < tol:
                    break
                old_d = d

            if progbar:
                pbar.set_description(str(d))

    if not inplace:
        tn = tn.copy()

    for t1, t2 in zip(tn, tna):
        # transpose so only thing changed in original TN is data
        t2.transpose_like_(t1)
        t1.modify(data=t2.data)

    return tn


# --------------------------------------------------------------------------- #
#                                Tensor Class                                 #
# --------------------------------------------------------------------------- #

class Tensor(object):
    """A labelled, tagged n-dimensional array. The index labels are used
    instead of axis numbers to identify dimensions, and are preserved through
    operations. The tags are used to identify the tensor within networks, and
    are combined when tensors are contracted together.

    Parameters
    ----------
    data : numpy.ndarray
        The n-dimensional data.
    inds : sequence of str
        The index labels for each dimension. Must match the number of
        dimensions of ``data``.
    tags : sequence of str, optional
        Tags with which to identify and group this tensor. These will
        be converted into a ``oset``.
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

    __slots__ = ('_data', '_inds', '_tags', '_left_inds', '_owners')

    def __init__(self, data=1.0, inds=(), tags=None, left_inds=None):
        # a new or copied Tensor always has no owners
        self._owners = dict()

        # Short circuit for copying Tensors
        if isinstance(data, Tensor):
            self._data = data.data
            self._inds = data.inds
            self._tags = data.tags.copy()
            self._left_inds = data.left_inds
            return

        self._data = asarray(data)
        self._inds = tuple(inds)
        self._tags = tags_to_oset(tags)
        self._left_inds = tuple(left_inds) if left_inds is not None else None

        nd = ndim(self._data)
        if nd != len(self.inds):
            raise ValueError(
                f"Wrong number of inds, {self.inds}, supplied for array"
                f" of shape {self._data.shape}.")

        if self.left_inds and any(i not in self.inds for i in self.left_inds):
            raise ValueError(f"The 'left' indices {self.left_inds} are not "
                             f"found in {self.inds}.")

    def copy(self, deep=False, virtual=False):
        """Copy this tensor.

        .. note::

            By default (``deep=False``), the underlying array will *not* be
            copied.

        Parameters
        ----------
        deep : bool, optional
            Whether to copy the underlying data as well.
        virtual : bool, optional
            To conveniently mimic the behaviour of taking a virtual copy of
            tensor network, this simply returns ``self``.
        """
        if not (deep or virtual):
            return self.__class__(self, None)

        if deep and virtual:
            raise ValueError("Copy can't be both deep and virtual.")

        if virtual:
            return self

        if deep:
            return copy.deepcopy(self)

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

    @property
    def owners(self):
        return self._owners

    def add_owner(self, tn, tid):
        """Add ``tn`` as owner of this Tensor - it's tag and ind maps will
        be updated whenever this tensor is retagged or reindexed.
        """
        self._owners[hash(tn)] = (weakref.ref(tn), tid)

    def remove_owner(self, tn):
        """Remove TensorNetwork ``tn`` as an owner of this Tensor.
        """
        try:
            del self._owners[hash(tn)]
        except KeyError:
            pass

    def check_owners(self):
        """Check if this tensor is 'owned' by any alive TensorNetworks. Also
        trim any weakrefs to dead TensorNetworks.
        """
        # first parse out dead owners
        for k in tuple(self._owners):
            if not self._owners[k][0]():
                del self._owners[k]

        return len(self._owners) > 0

    def _apply_function(self, fn):
        self._data = fn(self.data)

    def modify(self, **kwargs):
        """Overwrite the data of this tensor in place.

        Parameters
        ----------
        data : array, optional
            New data.
        apply : callable, optional
            A function to apply to the current data. If `data` is also given
            this is applied subsequently.
        inds : sequence of str, optional
            New tuple of indices.
        tags : sequence of str, optional
            New tags.
        left_inds : sequence of str, optional
            New grouping of indices to be 'on the left'.
        """
        if 'data' in kwargs:
            self._data = asarray(kwargs.pop('data'))
            self._left_inds = None

        if 'apply' in kwargs:
            self._apply_function(kwargs.pop('apply'))
            self._left_inds = None

        if 'inds' in kwargs:
            inds = tuple(kwargs.pop('inds'))
            # if this tensor has owners, update their ``ind_map``, but only if
            #     the indices are actually being changed not just permuted
            old_inds = oset(self.inds)
            new_inds = oset(inds)
            if (old_inds != new_inds) and self.check_owners():
                for ref, tid in self._owners.values():
                    ref()._modify_tensor_inds(old_inds, new_inds, tid)

            self._inds = inds
            self._left_inds = None

        if 'tags' in kwargs:
            tags = tags_to_oset(kwargs.pop('tags'))
            # if this tensor has owners, update their ``tag_map``.
            if self.check_owners():
                for ref, tid in self._owners.values():
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
        T.modify(apply=lambda x: x[data_loc], inds=new_inds, left_inds=None)
        return T

    isel_ = functools.partialmethod(isel, inplace=True)

    def add_tag(self, tag):
        """Add a tag to this tensor. Unlike ``self.tags.add`` this also updates
        any TensorNetworks viewing this Tensor.
        """
        # TODO: make this more efficient with inplace |= ?
        self.modify(tags=itertools.chain(self.tags, (tag,)))

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

        # list.insert has different behavior to expand_dims for -ve. axis
        if axis < 0:
            axis = len(new_inds) + axis + 1

        new_inds.insert(axis, name)

        new_data = do('expand_dims', self.data, axis=axis)

        self.modify(data=new_data, inds=new_inds)
        if size > 1:
            self.expand_ind(name, size)

    new_bond = new_bond

    def new_ind_with_identity(self, name, left_inds, right_inds, axis=0):
        """Inplace add a new index, where the newly stacked array entries form
        the identity from ``left_inds`` to ``right_inds``. Selecting 0 or 1 for
        the new index ``name`` thus is like 'turning off' this tensor if viewed
        as an operator.

        Parameters
        ----------
        name : str
            Name of the new index.
        left_inds : tuple[str]
            Names of the indices forming the left hand side of the operator.
        right_inds : tuple[str]
            Names of the indices forming the right hand side of the operator.
            The dimensions of these must match those of ``left_inds``.
        axis : int, optional
            Position of the new index.
        """
        ldims = tuple(map(self.ind_size, left_inds))
        x_id = do('eye', prod(ldims), dtype=self.dtype, like=self.data)
        x_id = do('reshape', x_id, ldims + ldims)
        t_id = Tensor(x_id, inds=left_inds + right_inds)
        t_id.transpose_(*self.inds)
        new_data = do('stack', (self.data, t_id.data), axis=axis)
        new_inds = list(self.inds)
        new_inds.insert(axis, name)
        self.modify(data=new_data, inds=new_inds)

    def conj(self, inplace=False):
        """Conjugate this tensors data (does nothing to indices).
        """
        t = self if inplace else self.copy()
        t.modify(apply=conj)
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
        return len(self._inds)

    @property
    def size(self):
        # more robust than calling _data.size (e.g. for torch) - consider
        # adding do('size', x) to autoray?
        return prod(self.shape)

    @property
    def dtype(self):
        return self._data.dtype

    def iscomplex(self):
        return iscomplex(self.data)

    def astype(self, dtype, inplace=False):
        """Change the type of this tensor to ``dtype``.
        """
        T = self if inplace else self.copy()
        if T.dtype != dtype:
            T.modify(apply=lambda data: astype(data, dtype))
        return T

    astype_ = functools.partialmethod(astype, inplace=True)

    def max_dim(self):
        """Return the maximum size of any dimension, or 1 if scalar.
        """
        if self.ndim == 0:
            return 1
        return max(self.shape)

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
        """Transpose this tensor - permuting the order of both the data *and*
        the indices. This operation is mainly for ensuring a certain data
        layout since for most operations the specific order of indices doesn't
        matter.

        Note to compute the tranditional 'transpose' of an operator within a
        contraction for example, you would just use reindexing not this.

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
        transpose_like, reindex
        """
        t = self if inplace else self.copy()

        output_inds = tuple(output_inds)  # need to re-use this.

        if set(t.inds) != set(output_inds):
            raise ValueError("'output_inds' must be permutation of the current"
                             f" tensor indices, but {set(t.inds)} != "
                             f"{set(output_inds)}")

        current_ind_map = {ind: i for i, ind in enumerate(t.inds)}
        perm = tuple(current_ind_map[i] for i in output_inds)

        t.modify(apply=lambda x: transpose(x, perm), inds=output_inds)
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

    def trace(
        self,
        left_inds,
        right_inds,
        preserve_tensor=False,
        inplace=False
    ):
        """Trace index or indices ``left_inds`` with ``right_inds``, removing
        them.

        Parameters
        ----------
        left_inds : str or sequence of str
            The left indices to trace, order matching ``right_inds``.
        right_inds : str or sequence of str
            The right indices to trace, order matching ``left_inds``.
        preserve_tensor : bool, optional
            If ``True``, a tensor will be returned even if no indices remain.
        inplace : bool, optional
            Perform the trace inplace.

        Returns
        -------
        z : Tensor or scalar
        """
        t = self if inplace else self.copy()

        if isinstance(left_inds, str):
            left_inds = (left_inds,)
        if isinstance(right_inds, str):
            right_inds = (right_inds,)

        if len(left_inds) != len(right_inds):
            raise ValueError(f"Can't trace {left_inds} with {right_inds}.")

        remap = {}
        for lix, rix in zip(left_inds, right_inds):
            remap[lix] = lix
            remap[rix] = lix

        old_inds, new_inds = [], []
        for ix in t.inds:
            nix = remap.pop(ix, None)
            if nix is not None:
                old_inds.append(nix)
            else:
                old_inds.append(ix)
                new_inds.append(ix)

        if remap:
            raise ValueError(f"Indices {tuple(remap)} not found.")

        old_inds, new_inds = tuple(old_inds), tuple(new_inds)

        eq = _inds_to_eq((old_inds,), new_inds)
        t.modify(apply=lambda x: do('einsum', eq, x, like=x),
                 inds=new_inds, left_inds=None)

        if not preserve_tensor and not new_inds:
            data_out = t.data
            if isinstance(data_out, np.ndarray):
                data_out = realify_scalar(data_out.item())
            return data_out

        return t

    def sum_reduce(self, ind, inplace=False):
        """Sum over index ``ind``, removing it from this tensor.

        Parameters
        ----------
        ind : str
            The index to sum over.
        inplace : bool, optional
            Whether to perform the reduction inplace.

        Returns
        -------
        Tensor
        """
        t = self if inplace else self.copy()
        axis = t.inds.index(ind)
        new_inds = t.inds[:axis] + t.inds[axis + 1:]
        t.modify(apply=lambda x: do('sum', x, axis=axis), inds=new_inds)
        return t

    sum_reduce_ = functools.partialmethod(sum_reduce, inplace=True)

    def collapse_repeated(self, inplace=False):
        """Take the diagonals of any repeated indices, such that each index
        only appears once.
        """
        t = self if inplace else self.copy()

        old_inds = t.inds
        new_inds = tuple(unique(old_inds))
        if len(old_inds) == len(new_inds):
            return t

        eq = _inds_to_eq((old_inds,), new_inds)
        t.modify(apply=lambda x: do('einsum', eq, x, like=x),
                 inds=new_inds, left_inds=None)

        return t

    collapse_repeated_ = functools.partialmethod(
        collapse_repeated, inplace=True)

    @functools.wraps(tensor_contract)
    def contract(self, *others, output_inds=None, **opts):
        return tensor_contract(self, *others, output_inds=output_inds, **opts)

    @functools.wraps(tensor_direct_product)
    def direct_product(self, other, sum_inds=(), inplace=False):
        return tensor_direct_product(
            self, other, sum_inds=sum_inds, inplace=inplace)

    direct_product_ = functools.partialmethod(direct_product, inplace=True)

    @functools.wraps(tensor_split)
    def split(self, *args, **kwargs):
        return tensor_split(self, *args, **kwargs)

    @functools.wraps(tensor_network_distance)
    def distance(self, other, **contract_opts):
        return tensor_network_distance(self, other, **contract_opts)

    def gate(self, G, ind, inplace=False, **contract_opts):
        """Gate this tensor - contract a matrix into one of its indices without
        changing its indices. Unlike ``contract``, ``G`` is a raw array and the
        tensor remains with the same set of indices.

        Parameters
        ----------
        G : 2D array_like
            The matrix to gate the tensor index with.
        ind : str
            Which index to apply the gate to.

        Returns
        -------
        Tensor

        Examples
        --------

        Create a random tensor of 4 qubits:

            >>> t = qtn.rand_tensor(
            ...    shape=[2, 2, 2, 2],
            ...    inds=['k0', 'k1', 'k2', 'k3'],
            ... )

        Create another tensor with an X gate applied to qubit 2:

            >>> Gt = t.gate(qu.pauli('X'), 'k2')

        The contraction of these two tensors is now the expectation of that
        operator:

            >>> t.H @ Gt
            -4.108910576149794

        """
        t = self if inplace else self.copy()
        G_inds = ['__tmp__', ind]
        out = ['__tmp__' if ix == ind else ix for ix in t.inds]
        new_data = oe.contract(G, G_inds, t.data, t.inds, out, **contract_opts)
        t.modify(data=new_data)
        return t

    gate_ = functools.partialmethod(gate, inplace=True)

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
                 inds=(*new_fused_inds, *unfused_inds))

        return t

    fuse_ = functools.partialmethod(fuse, inplace=True)

    def unfuse(self, unfuse_map, shape_map, inplace=False):
        """Reshape single indices into groups of multiple indices

        Parameters
        ----------
        unfuse_map : dict_like or sequence of tuples.
            Mapping like: ``{existing_ind: sequence of new inds, ...}`` or an
            ordered mapping like ``[(old_ind_1, new_inds_1), ...]`` in which
            case the output tensor's new inds will be ordered. In both cases
            the new indices are created at the old index's position of the
            tensor's shape
        shape_map : dict_like or sequence of tuples
            Mapping like: ``{old_ind: new_ind_sizes, ...}`` or an
            ordered mapping like ``[(old_ind_1, new_ind_sizes_1), ...]``.

        Returns
        -------
        Tensor
            The transposed, reshaped and re-labeled tensor
        """
        t = self if inplace else self.copy()

        if isinstance(unfuse_map, dict):
            old_inds, new_unfused_inds = zip(*unfuse_map.items())
        else:
            old_inds, new_unfused_inds = zip(*unfuse_map)

        # for each set of fused dims, group into product, then add remaining
        new_inds = [[i] for i in t.inds]
        new_dims = [[i] for i in t.shape]
        for ix in range(len(old_inds)):
            ind_pos = t.inds.index(old_inds[ix])
            new_inds[ind_pos] = new_unfused_inds[ix]
            new_dims[ind_pos] = shape_map[old_inds[ix]]

        # flatten new_inds, new_dims
        new_inds = tuple(itertools.chain(*new_inds))
        new_dims = tuple(itertools.chain(*new_dims))

        try:
            new_left_inds = []
            for ix in t.left_inds:
                try:
                    new_left_inds.extend(unfuse_map[ix])
                except KeyError:
                    new_left_inds.append(ix)
        except TypeError:
            new_left_inds = None

        # create new tensor with new + remaining indices
        #     + updated 'left' marked indices assuming all unfused left inds
        #       remain 'left' marked
        t.modify(data=reshape(t.data, new_dims),
                 inds=new_inds, left_inds=new_left_inds)

        return t

    unfuse_ = functools.partialmethod(unfuse, inplace=True)

    def to_dense(self, *inds_seq, to_qarray=True):
        """Convert this Tensor into an dense array, with a single dimension
        for each of inds in ``inds_seqs``. E.g. to convert several sites
        into a density matrix: ``T.to_dense(('k0', 'k1'), ('b0', 'b1'))``.
        """
        x = self.fuse([(str(i), ix) for i, ix in enumerate(inds_seq)]).data
        if (infer_backend(x) == 'numpy') and to_qarray:
            return qarray(x)
        return x

    def squeeze(self, include=None, inplace=False):
        """Drop any singlet dimensions from this tensor.

        Parameters
        ----------
        inplace : bool, optional
            Whether modify the original or return a new tensor.
        include : sequence of str, optional
            Only squeeze dimensions with indices in this list.

        Returns
        -------
        Tensor
        """
        t = self if inplace else self.copy()

        # handles case of scalar as well
        if 1 not in t.shape:
            return t

        new_shape_new_inds = [
            (d, i) for d, i in zip(self.shape, self.inds)
            if (d > 1) or (include is not None and i not in include)
        ]

        if not new_shape_new_inds:
            # squeezing everything -> can't unzip `new_shape_new_inds`
            new_inds = ()
            new_data = reshape(t.data, ())
        else:
            new_shape, new_inds = zip(*new_shape_new_inds)
            new_data = reshape(t.data, new_shape)

        new_left_inds = (
            None if self.left_inds is None else
            (i for i in self.left_inds if i in new_inds)
        )

        if len(t.inds) != len(new_inds):
            t.modify(data=new_data, inds=new_inds, left_inds=new_left_inds)

        return t

    squeeze_ = functools.partialmethod(squeeze, inplace=True)

    def largest_element(self):
        r"""Return the largest element, in terms of absolute magnitude, of this
        tensor.
        """
        return do('max', do('abs', self.data))

    def norm(self):
        r"""Frobenius norm of this tensor:

        .. math::

            \|t\|_F = \sqrt{\mathrm{Tr} \left(t^{\dagger} t\right)}

        where the trace is taken over all indices. Equivalent to the square
        root of the sum of squared singular values across any partition.
        """
        return norm_fro(self.data)

    def normalize(self, inplace=False):
        T = self if inplace else self.copy()
        T.modify(data=T.data / T.norm(), left_inds=T.left_inds)
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
        Tu = self.__class__(
            x, inds=LR_inds, tags=self.tags, left_inds=left_inds
        )

        if inplace:
            # XXX: do self.transpose_like_(Tu) or Tu.transpose_like_(self)?
            self.modify(data=Tu.data, inds=Tu.inds, left_inds=Tu.left_inds)
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
        t.modify(apply=lambda x: x[flipper])
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
            self.modify(tags=oset())
        else:
            self.modify(tags=self.tags - tags_to_oset(tags))

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

    def __imul__(self, other):
        self.modify(apply=lambda x: x * other)
        return self

    def __itruediv__(self, other):
        self.modify(apply=lambda x: x / other)
        return self

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

    def draw(self, *args, **kwargs):
        """Plot a graph of this tensor and its indices.
        """
        draw_tn(TensorNetwork((self,)), *args, **kwargs)

    graph = draw

    def __getstate__(self):
        # This allows pickling, since the copy has no weakrefs.
        return (self._data, self._inds, self._tags, self._left_inds)

    def __setstate__(self, state):
        self._data, self._inds, tags, self._left_inds = state
        self._tags = tags.copy()
        self._owners = {}

    def __repr__(self):
        return (f"{self.__class__.__name__}("
                f"shape={tuple(map(int, self.data.shape))}, "
                f"inds={self.inds}, "
                f"tags={self.tags})")

    def __str__(self):
        s = self.__repr__()[:-1]
        s += (f", backend='{infer_backend(self.data)}'"
              f", dtype='{get_dtype_name(self.data)}')")
        return s


@functools.lru_cache(128)
def _make_copy_ndarray(d, ndim, dtype=float):
    c = np.zeros([d] * ndim, dtype=dtype)
    for i in range(d):
        c[(i,) * ndim] = 1
    make_immutable(c)
    return c


def COPY_tensor(d, inds, tags=None, dtype=float):
    """Get the tensor representing the COPY operation with dimension size
    ``d`` and number of dimensions ``len(inds)``, with exterior indices
    ``inds``.

    Parameters
    ----------
    d : int
        The size of each dimension.
    inds : sequence of str
        The exterior index names for each dimension.
    tags : None or sequence of str, optional
        Tag the tensor with these.
    dtype : str, optional
        Data type to create the underlying numpy array with.

    Returns
    -------
    Tensor
        The tensor describing the MPS, of size ``d**len(inds)``.
    """
    ndim = len(inds)
    return Tensor(_make_copy_ndarray(d, ndim, dtype), inds, tags)


def COPY_mps_tensors(d, inds, tags=None, dtype=float):
    """Get the set of MPS tensors representing the COPY tensor with dimension
    size ``d`` and number of dimensions ``len(inds)``, with exterior indices
    ``inds``.

    Parameters
    ----------
    d : int
        The size of each dimension.
    inds : sequence of str
        The exterior index names for each dimension.
    tags : None or sequence of str, optional
        Tag the tensors with these.
    dtype : str, optional
        Data type to create the underlying numpy array with.

    Returns
    -------
    List[Tensor]
        The ``len(inds)`` tensors describing the MPS, with physical legs
        ordered as supplied in ``inds``.
    """
    ndim = len(inds)
    if ndim <= 3:
        # no saving from dense to MPS -> ([d, d], [d, d, d], [d, d])
        return [COPY_tensor(d, inds, tags, dtype)]

    bonds = collections.defaultdict(rand_uuid)

    sub_inds = (inds[0], bonds[0, 1])
    ts = [COPY_tensor(d, sub_inds, tags, dtype)]
    for i in range(1, ndim - 1):
        sub_inds = (bonds[i - 1, i], bonds[i, i + 1], inds[i])
        ts.append(COPY_tensor(d, inds=sub_inds, tags=tags, dtype=dtype))
    sub_inds = (bonds[ndim - 2, ndim - 1], inds[-1])
    ts.append(COPY_tensor(d, inds=sub_inds, tags=tags, dtype=dtype))

    return ts


def COPY_tree_tensors(d, inds, tags=None, dtype=float, ssa_path=None):
    """Get the set of tree tensors representing the COPY tensor with dimension
    size ``d`` and number of dimensions ``len(inds)``, with exterior indices
    ``inds``. The tree is generated by cycling through pairs.

    Parameters
    ----------
    d : int
        The size of each dimension.
    inds : sequence of str
        The exterior index names for each dimension.
    tags : None or sequence of str, optional
        Tag the tensors with these.
    dtype : str, optional
        Data type to create the underlying numpy array with.

    Returns
    -------
    List[Tensor]
        The ``len(inds) - 2`` tensors describing the TTN, with physical legs
        ordered as supplied in ``inds``.
    """
    if ssa_path is None:
        ssa_path = ((2 * i, 2 * i + 1) for i in itertools.count())
    else:
        ssa_path = iter(ssa_path)

    ts = []
    remaining = set(inds)
    ssa_leaves = list(inds)

    while len(remaining) > 3:
        k1, k2 = next(ssa_path)
        ix1 = ssa_leaves[k1]
        ix2 = ssa_leaves[k2]
        ix12 = rand_uuid()
        ssa_leaves.append(ix12)
        ts.append(COPY_tensor(d, (ix1, ix2, ix12), tags, dtype))
        remaining.symmetric_difference_update((ix1, ix2, ix12))

    ts.append(COPY_tensor(d, sorted(remaining), tags, dtype))
    return ts


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
    virtual : bool, optional
        Whether the TensorNetwork should be a *view* onto the tensors it is
        given, or a copy of them. E.g. if a virtual TN is constructed, any
        changes to a Tensor's indices or tags will propagate to all TNs viewing
        that Tensor.
    check_collisions : bool, optional
        If True, the default, then ``TensorNetwork`` instances with double
        indices which match another ``TensorNetwork`` instances double indices
        will have those indices' names mangled. Can be explicitly turned off
        when it is known that no collisions will take place -- i.e. when not
        adding any new tensors.

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
    exponent : float
        A scalar prefactor for the tensor network, stored in base 10 like
        ``10**exponent``. This is mostly for conditioning purposes and will be
        ``0.0`` unless you use use ``equalize_norms(value)`` or
        ``tn.strip_exponent(tid_or_tensor)``.
    """

    _EXTRA_PROPS = ()
    _CONTRACT_STRUCTURED = False

    def __init__(self, ts, *, virtual=False, check_collisions=True):

        # short-circuit for copying TensorNetworks
        if isinstance(ts, TensorNetwork):
            self.tag_map = valmap(lambda tids: tids.copy(), ts.tag_map)
            self.ind_map = valmap(lambda tids: tids.copy(), ts.ind_map)
            self.tensor_map = dict()
            for tid, t in ts.tensor_map.items():
                self.tensor_map[tid] = t if virtual else t.copy()
                self.tensor_map[tid].add_owner(self, tid)
            self._inner_inds = ts._inner_inds.copy()
            self._outer_inds = ts._outer_inds.copy()
            self._tid_counter = ts._tid_counter
            self.exponent = ts.exponent
            for ep in ts.__class__._EXTRA_PROPS:
                setattr(self, ep, getattr(ts, ep))
            return

        # internal structure
        self._tid_counter = 0
        self.tensor_map = dict()
        self.tag_map = dict()
        self.ind_map = dict()
        self._inner_inds = oset()
        self._outer_inds = oset()
        self.exponent = 0.0
        for t in ts:
            self.add(t, virtual=virtual, check_collisions=check_collisions)

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

    def _link_tags(self, tags, tid):
        """Link ``tid`` to each of ``tags``.
        """
        for tag in tags:
            if tag in self.tag_map:
                self.tag_map[tag].add(tid)
            else:
                self.tag_map[tag] = oset((tid,))

    def _unlink_tags(self, tags, tid):
        """"Unlink ``tid`` from each of ``tags``.
        """
        for tag in tags:
            try:
                tids = self.tag_map[tag]
                tids.discard(tid)
                if not tids:
                    # tid was last tensor -> delete entry
                    del self.tag_map[tag]
            except KeyError:
                # tid already removed from x entry - e.g. repeated index
                pass

    def _link_inds(self, inds, tid):
        """Link ``tid`` to each of ``inds``.
        """
        for ind in inds:
            if ind in self.ind_map:
                self.ind_map[ind].add(tid)
                self._outer_inds.discard(ind)
                self._inner_inds.add(ind)
            else:
                self.ind_map[ind] = oset((tid,))
                self._outer_inds.add(ind)

    def _unlink_inds(self, inds, tid):
        """"Unlink ``tid`` from each of ``inds``.
        """
        for ind in inds:
            try:
                tids = self.ind_map[ind]
                tids.discard(tid)
                occurences = len(tids)
                if occurences == 0:
                    # tid was last tensor -> delete entry
                    del self.ind_map[ind]
                    self._outer_inds.discard(ind)
                elif occurences == 1:
                    self._inner_inds.discard(ind)
                    self._outer_inds.add(ind)
            except KeyError:
                # tid already removed from x entry - e.g. repeated index
                pass

    def _reset_inner_outer(self, inds):
        for ind in inds:
            occurences = len(self.ind_map[ind])
            if occurences == 1:
                self._inner_inds.discard(ind)
                self._outer_inds.add(ind)
            else:
                self._inner_inds.add(ind)
                self._outer_inds.discard(ind)

    def _next_tid(self):
        # N.B. safer? previous behavior -> return rand_uuid('_T')
        while self._tid_counter in self.tensor_map:
            self._tid_counter = self._tid_counter + 1
        return self._tid_counter

    def add_tensor(self, tensor, tid=None, virtual=False):
        """Add a single tensor to this network - mangle its tid if neccessary.
        """
        # check for tid conflict
        if (tid is None) or (tid in self.tensor_map):
            tid = self._next_tid()

        # add tensor to the main index
        T = tensor if virtual else tensor.copy()
        self.tensor_map[tid] = T
        T.add_owner(self, tid)

        # add its tid to the relevant tag and inds maps, or create new entries
        self._link_tags(T.tags, tid)
        self._link_inds(T.inds, tid)

    def add_tensor_network(self, tn, virtual=False, check_collisions=True):
        """
        """
        if check_collisions:  # add tensors individually
            # check for matching inner_indices -> need to re-index
            clash_ix = self._inner_inds & tn._inner_inds
            reind = {ix: rand_uuid() for ix in clash_ix}
        else:
            clash_ix = False
            reind = None

        # add tensors, reindexing if necessary
        for tid, tsr in tn.tensor_map.items():
            if clash_ix and any(i in reind for i in tsr.inds):
                tsr = tsr.reindex(reind, inplace=virtual)
            self.add_tensor(tsr, virtual=virtual, tid=tid)

        self.exponent = self.exponent + tn.exponent

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

    def make_tids_consecutive(self, tid0=0):
        """Reset the `tids` - node identifies - to be consecutive integers.
        """
        tids = tuple(self.tensor_map.keys())
        ts = tuple(map(self._pop_tensor, tids))
        self._tid_counter = tid0
        self.add(ts, virtual=True)

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
        self._unlink_tags(old - new, tid)
        self._link_tags(new - old, tid)

    def _modify_tensor_inds(self, old, new, tid):
        self._unlink_inds(old - new, tid)
        self._link_inds(new - old, tid)

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

    def _pop_tensor(self, tid):
        """Remove a tensor from this network, returning said tensor.
        """
        # pop the tensor itself
        t = self.tensor_map.pop(tid)

        # remove the tid from the tag and ind maps
        self._unlink_tags(t.tags, tid)
        self._unlink_inds(t.inds, tid)

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
        tags = tags_to_oset(tags)

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

        tids = oset_union(tn.ind_map.get(ix, oset()) for ix in index_map)

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

    def largest_element(self):
        """Return the 'largest element', in terms of absolute magnitude, of
        this tensor network. This is defined as the product of the largest
        elements of each tensor in the network, which would be the largest
        single term occuring if the TN was summed explicitly.
        """
        return prod(t.largest_element() for t in self)

    def norm(self, **contract_opts):
        r"""Frobenius norm of this tensor network. Computed by exactly
        contracting the TN with its conjugate:

        .. math::

            \|T\|_F = \sqrt{\mathrm{Tr} \left(T^{\dagger} T\right)}

        where the trace is taken over all indices. Equivalent to the square
        root of the sum of squared singular values across any partition.
        """
        norm = self.conj() | self
        return norm.contract(**contract_opts)**0.5

    def make_norm(
        self,
        mangle_append='*',
        layer_tags=('KET', 'BRA'),
        return_all=False,
    ):
        """Make the norm tensor network of this tensor network ``tn.H & tn``.

        Parameters
        ----------
        mangle_append : {str, False or None}, optional
            How to mangle the inner indices of the bra.
        layer_tags : (str, str), optional
            The tags to identify the top and bottom.
        return_all : bool, optional
            Return the norm, the ket and the bra.
        """
        ket = self.copy()
        ket.add_tag(layer_tags[0])

        bra = ket.retag({layer_tags[0]: layer_tags[1]})
        bra.conj_(mangle_append)

        norm = ket | bra

        if return_all:
            return norm, ket, bra
        return norm

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
                tensor.modify(apply=lambda data: data * (x_sign * x_spread))
            else:
                tensor.modify(apply=lambda data: data * x_spread)

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
            t.modify(apply=lambda data: data * x)

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

    def __iter__(self):
        return iter(self.tensor_map.values())

    @property
    def tensors(self):
        """Get the tuple of tensors in this tensor network.
        """
        return tuple(self.tensor_map.values())

    @property
    def arrays(self):
        """Get the tuple of raw arrays containing all the tensor network data.
        """
        return tuple(t.data for t in self)

    def get_symbol_map(self):
        """Get the mapping of the current indices to ``einsum`` style single
        unicode characters. The symbols are generated in the order they appear
        on the tensors.

        See Also
        --------
        get_equation, get_inputs_output_size_dict
        """
        symbol_map = empty_symbol_map()
        for t in self:
            for ix in t.inds:
                symbol_map[ix]
        return symbol_map

    def get_equation(self, output_inds=None):
        """Get the 'equation' describing this tensor network, in ``einsum``
        style with a single unicode letter per index. The symbols are generated
        in the order they appear on the tensors.

        Parameters
        ----------
        output_inds : None or sequence of str, optional
            Manually specify which are the output indices.

        Returns
        -------
        eq : str

        Examples
        --------

            >>> tn = qtn.TN_rand_reg(10, 3, 2)
            >>> tn.get_equation()
            'abc,dec,fgb,hia,jke,lfk,mnj,ing,omd,ohl->'

        See Also
        --------
        get_symbol_map, get_inputs_output_size_dict
        """
        if output_inds is None:
            output_inds = self.outer_inds()
        inputs_inds = tuple(t.inds for t in self)
        return _inds_to_eq(inputs_inds, output_inds)

    def get_inputs_output_size_dict(self, output_inds=None):
        """Get a tuple of ``inputs``, ``output`` and ``size_dict`` suitable for
        e.g. passing to path optimizers. The symbols are generated in the order
        they appear on the tensors.

        Parameters
        ----------
        output_inds : None or sequence of str, optional
            Manually specify which are the output indices.

        Returns
        -------
        inputs : tuple[str]
        output : str
        size_dict : dict[str, ix]

        See Also
        --------
        get_symbol_map, get_equation
        """
        eq = self.get_equation(output_inds=output_inds)
        lhs, output = eq.split('->')
        inputs = lhs.split(',')
        size_dict = {}
        for term, t in zip(inputs, self):
            for k, d in zip(term, t.shape):
                size_dict[k] = int(d)
        return inputs, output, size_dict

    def geometry_hash(self, output_inds=None, strict_index_order=False):
        """A hash of this tensor network's shapes & geometry. A useful check
        for determinism. Moreover, if this matches for two tensor networks then
        they can be contracted using the same tree for the same cost. Order of
        tensors matters for this - two isomorphic tensor networks with shuffled
        tensor order will not have the same hash value. Permuting the indices
        of individual of tensors or the output does not matter unless you set
        ``strict_index_order=True``.

        Parameters
        ----------
        output_inds : None or sequence of str, optional
            Manually specify which indices are output indices and their order,
            otherwise assumed to be all indices that appear once.
        strict_index_order : bool, optional
            If ``False``, then the permutation of the indices of each tensor
            and the output does not matter.

        Returns
        -------
        str

        Examples
        --------

        If we transpose some indices, then only the strict hash changes:

            >>> tn = qtn.TN_rand_reg(100, 3, 2, seed=0)
            >>> tn.geometry_hash()
            '18c702b2d026dccb1a69d640b79d22f3e706b6ad'

            >>> tn.geometry_hash(strict_index_order=True)
            'c109fdb43c5c788c0aef7b8df7bb83853cf67ca1'

            >>> t = tn['I0']
            >>> t.transpose_(t.inds[2], t.inds[1], t.inds[0])
            >>> tn.geometry_hash()
            '18c702b2d026dccb1a69d640b79d22f3e706b6ad'

            >>> tn.geometry_hash(strict_index_order=True)
            '52c32c1d4f349373f02d512f536b1651dfe25893'


        """
        import pickle
        import hashlib

        inputs, output, size_dict = self.get_inputs_output_size_dict(
            output_inds=output_inds,
        )

        if strict_index_order:
            return hashlib.sha1(pickle.dumps((
                tuple(map(tuple, inputs)),
                tuple(output),
                sortedtuple(size_dict.items())
            ))).hexdigest()

        edges = collections.defaultdict(list)
        for ix in output:
            edges[ix].append(-1)
        for i, term in enumerate(inputs):
            for ix in term:
                edges[ix].append(i)

        # then sort edges by each's incidence nodes
        canonical_edges = sortedtuple(map(sortedtuple, edges.values()))

        return hashlib.sha1(pickle.dumps((
            canonical_edges, sortedtuple(size_dict.items())
        ))).hexdigest()

    def tensors_sorted(self):
        """Return a tuple of tensors sorted by their respective tags, such that
        the tensors of two networks with the same tag structure can be
        iterated over pairwise.
        """
        ts_and_sorted_tags = [(t, sorted(t.tags)) for t in self]
        ts_and_sorted_tags.sort(key=lambda x: x[1])
        return tuple(x[0] for x in ts_and_sorted_tags)

    def apply_to_arrays(self, fn):
        """Modify every tensor's array inplace by applying ``fn`` to it.
        """
        for t in self:
            t.modify(apply=fn)

    # ----------------- selecting and splitting the network ----------------- #

    def _get_tids_from(self, xmap, xs, which):
        inverse = which[0] == '!'
        if inverse:
            which = which[1:]

        combine = {
            'all': oset_intersection,
            'any': oset_union,
        }[which]

        tid_sets = tuple(xmap[x] for x in xs)
        if not tid_sets:
            tids = oset()
        else:
            tids = combine(tid_sets)

        if inverse:
            return oset(self.tensor_map) - tids

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
        else:
            tags = tags_to_oset(tags)

        return self._get_tids_from(self.tag_map, tags, which)

    def _get_tids_from_inds(self, inds, which='all'):
        """Like ``_get_tids_from_tags`` but specify inds instead.
        """
        inds = tags_to_oset(inds)
        return self._get_tids_from(self.ind_map, inds, which)

    def _tids_get(self, *tids):
        """Convenience function that generates unique tensors from tids.
        """
        seen = set()
        sadd = seen.add
        tmap = self.tensor_map
        for tid in tids:
            if tid not in seen:
                yield tmap[tid]
                sadd(tid)

    def _inds_get(self, *inds):
        """Convenience function that generates unique tensors from inds.
        """
        seen = set()
        sadd = seen.add
        tmap = self.tensor_map
        imap = self.ind_map
        for ind in inds:
            for tid in imap.get(ind, ()):
                if tid not in seen:
                    yield tmap[tid]
                    sadd(tid)

    def _tags_get(self, *tags):
        """Convenience function that generates unique tensors from tags.
        """
        seen = set()
        sadd = seen.add
        tmap = self.tensor_map
        gmap = self.tag_map
        for tag in tags:
            for tid in gmap.get(tag, ()):
                if tid not in seen:
                    yield tmap[tid]
                    sadd(tid)

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

    def _select_tids(self, tids, virtual=True):
        """Get a copy or a virtual copy (doesn't copy the tensors) of this
        ``TensorNetwork``, only with the tensors corresponding to ``tids``.
        """
        tn = TensorNetwork(())
        for tid in tids:
            tn.add_tensor(self.tensor_map[tid], tid=tid, virtual=virtual)
        tn.view_like_(self)
        return tn

    def _select_without_tids(self, tids, virtual=True):
        """Get a copy or a virtual copy (doesn't copy the tensors) of this
        ``TensorNetwork``, without the tensors corresponding to ``tids``.
        """
        tn = self.copy(virtual=virtual)
        for tid in tids:
            tn._pop_tensor(tid)
        return tn

    def select(self, tags, which='all', virtual=True):
        """Get a TensorNetwork comprising tensors that match all or any of
        ``tags``, inherit the network properties/structure from ``self``.
        This returns a view of the tensors not a copy.

        Parameters
        ----------
        tags : str or sequence of str
            The tag or tag sequence.
        which : {'all', 'any'}
            Whether to require matching all or any of the tags.
        virtual : bool, optional
            Whether the returned tensor network views the same tensors (the
            default) or takes copies (``virtual=False``) from ``self``.

        Returns
        -------
        tagged_tn : TensorNetwork
            A tensor network containing the tagged tensors.

        See Also
        --------
        select_tensors, select_neighbors, partition, partition_tensors
        """
        tagged_tids = self._get_tids_from_tags(tags, which=which)
        return self._select_tids(tagged_tids, virtual=virtual)

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
        inds = oset_union(t.inds for t in tagged_ts)

        # find all tensors with those inds, and remove the initial tensors
        inds_tids = oset_union(self.ind_map[i] for i in inds)
        neighbour_tids = inds_tids - tagged_tids

        return tuple(self.tensor_map[tid] for tid in neighbour_tids)

    def _select_local_tids(
        self,
        tids,
        max_distance=1,
        fillin=False,
        reduce_outer=None,
        inwards=False,
        virtual=True,
        include=None,
        exclude=None,
    ):
        span = self.get_tree_span(
            tids, max_distance=max_distance,
            include=include, exclude=exclude, inwards=inwards,
        )
        local_tids = oset(tids)
        for s in span:
            local_tids.add(s[0])
            local_tids.add(s[1])

        for _ in range(int(fillin)):
            connectivity = frequencies(
                tid_n
                for tid in local_tids
                for tid_n in self._get_neighbor_tids(tid)
                if tid_n not in local_tids
            )
            for tid_n, cnt in connectivity.items():
                if cnt >= 2:
                    local_tids.add(tid_n)

        tn_sl = self._select_tids(local_tids, virtual=virtual)

        # optionally remove/reduce outer indices that appear outside `tag`
        if reduce_outer == 'sum':
            for ix in tn_sl.outer_inds():
                tid_edge, = tn_sl.ind_map[ix]
                if tid_edge in tids:
                    continue
                tn_sl.tensor_map[tid_edge].sum_reduce_(ix)

        elif reduce_outer == 'svd':
            for ix in tn_sl.outer_inds():
                # get the tids that stretch across the border
                tid_out, tid_in = sorted(
                    self.ind_map[ix], key=tn_sl.tensor_map.__contains__)

                # rank-1 decompose the outer tensor
                l, r = self.tensor_map[tid_out].split(
                    left_inds=None, right_inds=[ix],
                    max_bond=1, get='arrays', absorb='left')

                # absorb the factor into the inner tensor to remove that ind
                tn_sl.tensor_map[tid_in].gate_(r, ix).squeeze_(include=[ix])

        elif reduce_outer == 'svd-sum':
            for ix in tn_sl.outer_inds():
                # get the tids that stretch across the border
                tid_out, tid_in = sorted(
                    self.ind_map[ix], key=tn_sl.tensor_map.__contains__)

                # full-rank decompose the outer tensor
                l, r = self.tensor_map[tid_out].split(
                    left_inds=None, right_inds=[ix],
                    max_bond=None, get='arrays', absorb='left')

                # absorb the factor into the inner tensor then sum over it
                tn_sl.tensor_map[tid_in].gate_(r, ix).sum_reduce_(ix)

        elif reduce_outer == 'reflect':
            tn_sl |= tn_sl.H

        return tn_sl

    def select_local(
        self,
        tags,
        which='all',
        max_distance=1,
        fillin=False,
        reduce_outer=None,
        virtual=True,
        include=None,
        exclude=None,
    ):
        r"""Select a local region of tensors, based on graph distance
        ``max_distance`` to any tagged tensors.

        Parameters
        ----------
        tags : str or sequence of str
            The tag or tag sequence defining the initial region.
        which : {'all', 'any', '!all', '!any'}, optional
            Whether to require matching all or any of the tags.
        max_distance : int, optional
            The maximum distance to the initial tagged region.
        fillin : bool or int, optional
            Once the local region has been selected based on graph distance,
            whether and how many times to 'fill-in' corners by adding tensors
            connected multiple times. For example, if ``R`` is an initially
            tagged tensor and ``x`` are locally selected tensors::

                  fillin=0       fillin=1       fillin=2

                 | | | | |      | | | | |      | | | | |
                -o-o-x-o-o-    -o-x-x-x-o-    -x-x-x-x-x-
                 | | | | |      | | | | |      | | | | |
                -o-x-x-x-o-    -x-x-x-x-x-    -x-x-x-x-x-
                 | | | | |      | | | | |      | | | | |
                -x-x-R-x-x-    -x-x-R-x-x-    -x-x-R-x-x-

        reduce_outer : {'sum', 'svd', 'svd-sum', 'reflect'}, optional
            Whether and how to reduce any outer indices of the selected region.
        virtual : bool, optional
            Whether the returned tensor network should be a view of the tensors
            or a copy (``virtual=False``).
        include : sequence of int, optional
            Only include tensor with these ``tids``.
        exclude : sequence of int, optional
            Only include tensor without these ``tids``.

        Returns
        -------
        TensorNetwork
        """
        check_opt('reduce_outer', reduce_outer,
                  (None, 'sum', 'svd', 'svd-sum', 'reflect'))

        return self._select_local_tids(
            tids=self._get_tids_from_tags(tags, which),
            max_distance=max_distance,
            fillin=fillin,
            reduce_outer=reduce_outer,
            virtual=virtual,
            include=include,
            exclude=exclude)

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
            return self.select_any(self.maybe_convert_coo(tags))

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
        self._pop_tensor(tid)
        self.add_tensor(tensor, tid=tid, virtual=True)

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

    def partition(self, tags, which='any', inplace=False):
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

        Returns
        -------
        untagged_tn, tagged_tn : (TensorNetwork, TensorNetwork)
            The untagged and tagged tensor networs.

        See Also
        --------
        partition_tensors, select, select_tensors
        """
        tagged_tids = self._get_tids_from_tags(tags, which=which)

        kws = {'check_collisions': False}

        if inplace:
            t1 = self
            t2s = [t1._pop_tensor(tid) for tid in tagged_tids]
            t2 = TensorNetwork(t2s, **kws)
            t2.view_like_(self)

        else:  # rebuild both -> quicker
            t1s, t2s = [], []
            for tid, tensor in self.tensor_map.items():
                (t2s if tid in tagged_tids else t1s).append(tensor)

            t1, t2 = TensorNetwork(t1s, **kws), TensorNetwork(t2s, **kws)
            t1.view_like_(self)
            t2.view_like_(self)

        return t1, t2

    def _split_tensor_tid(self, tid, left_inds, **split_opts):
        t = self._pop_tensor(tid)
        tl, tr = t.split(left_inds=left_inds, get='tensors', **split_opts)
        self.add_tensor(tl)
        self.add_tensor(tr)
        return self

    def split_tensor(
        self,
        tags,
        left_inds,
        **split_opts,
    ):
        """Split the single tensor uniquely identified by ``tags``, adding the
        resulting tensors from the decomposition back into the network. Inplace
        operation.
        """
        tid, = self._get_tids_from_tags(tags, which='all')
        self._split_tensor_tid(tid, left_inds, **split_opts)

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
        TensorNetwork

        See Also
        --------
        replace_with_identity
        """
        leave, svd_section = self.partition(where, which=which,
                                            inplace=inplace)

        tags = svd_section.tags if keep_tags else oset()
        ltags = tags_to_oset(ltags)
        rtags = tags_to_oset(rtags)

        if right_inds is None:
            # compute
            right_inds = tuple(i for i in svd_section.outer_inds()
                               if i not in left_inds)

        if (start is None) and (stop is None):
            A = svd_section.aslinearoperator(left_inds=left_inds,
                                             right_inds=right_inds)
        else:
            from .tensor_1d import TNLinearOperator1D

            # check if need to invert start stop as well
            if '!' in which:
                start, stop = stop, start + self.L
                left_inds, right_inds = right_inds, left_inds
                ltags, rtags = rtags, ltags

            A = TNLinearOperator1D(svd_section, start=start, stop=stop,
                                   left_inds=left_inds, right_inds=right_inds)

        ltags = tags | ltags
        rtags = tags | rtags

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

    def _contract_between_tids(
        self,
        tid1,
        tid2,
        equalize_norms=False,
        gauges=None,
        **contract_opts,
    ):
        # allow no-op for same tensor specified twice ('already contracted')
        if tid1 == tid2:
            return

        output_inds = self.compute_contracted_inds(tid1, tid2)
        t1 = self._pop_tensor(tid1)
        t2 = self._pop_tensor(tid2)

        if gauges is not None:
            for ix in bonds(t1, t2):
                # about to contract so don't need to balance gauge on both
                g = gauges.pop(ix)
                t1.multiply_index_diagonal_(ix, g)

        t12 = tensor_contract(
            t1, t2,
            output_inds=output_inds,
            preserve_tensor=True,
            **contract_opts,
        )
        self.add_tensor(t12, tid=tid2, virtual=True)

        # maybe control norm blow-up by stripping the new tensor exponent
        if equalize_norms:
            self.strip_exponent(tid2, equalize_norms)

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
        self._contract_between_tids(tid1, tid2, **contract_opts)

    def contract_ind(self, ind, output_inds=None, **contract_opts):
        """Contract tensors connected by ``ind``.
        """
        tids = tuple(self._get_tids_from_inds(ind))
        output_inds = self.compute_contracted_inds(
            *tids, output_inds=output_inds)
        tnew = tensor_contract(
            *map(self._pop_tensor, tids), output_inds=output_inds,
            preserve_tensor=True, **contract_opts
        )
        self.add_tensor(tnew, tid=tids[0], virtual=True)

    def gate_inds(
        self,
        G,
        inds,
        contract=False,
        tags=None,
        info=None,
        inplace=False,
        **compress_opts,
    ):
        """Apply the 'gate' ``G`` to indices ``inds``, propagating them to the
        outside, as if applying ``G @ x``.

        Parameters
        ----------
        G : array_ike
            The gate array to apply, should match or be factorable into the
            shape ``(*phys_dims, *phys_dims)``.
        inds : str or sequence or str,
            The index or indices to apply the gate to.
        contract : {False, True, 'split', 'reduce-split'}, optional
            How to apply the gate:

                - False: gate is added to network and nothing is contracted,
                  tensor network structure is thus not maintained.
                - True: gate is contracted with all tensors involved, tensor
                  network structure is thus only maintained if gate acts on a
                  single site only.
                - 'split': contract all involved tensors then split the result
                  back into two.
                - 'reduce-split': factor the two physical indices into
                  'R-factors' using QR decompositions on the original site
                  tensors, then contract the gate, split it and reabsorb each
                  side. Much cheaper than ``'split'``.

            The final two methods are relevant for two site gates only, for
            single site gates they use the ``contract=True`` option which also
            maintains the structure of the TN. See below for a pictorial
            description of each method.
        tags : str or sequence of str, optional
            Tags to add to the new gate tensor.
        info : None or dict, optional
            Used to store extra optional information such as the singular
            values if not absorbed.
        inplace : bool, optional
            Whether to perform the gate operation inplace on the tensor
            network or not.
        compress_opts
            Supplied to :func:`~quimb.tensor.tensor_core.tensor_split` for any
            ``contract`` methods that involve splitting. Ignored otherwise.

        Returns
        -------
        G_tn : TensorNetwork

        Notes
        -----

        The ``contract`` options look like the following (for two site gates).

        ``contract=False``::

              .   .  <- inds
              │   │
              GGGGG
              │╱  │╱
            ──●───●──
             ╱   ╱

        ``contract=True``::

              │╱  │╱
            ──GGGGG──
             ╱   ╱

        ``contract='split'``::

              │╱  │╱          │╱  │╱
            ──GGGGG──  ==>  ──G┄┄┄G──
             ╱   ╱           ╱   ╱
             <SVD>

        ``contract='reduce-split'``::

               │   │             │ │
               GGGGG             GGG               │ │
               │╱  │╱   ==>     ╱│ │  ╱   ==>     ╱│ │  ╱          │╱  │╱
             ──●───●──       ──>─●─●─<──       ──>─GGG─<──  ==>  ──G┄┄┄G──
              ╱   ╱           ╱     ╱           ╱     ╱           ╱   ╱
            <QR> <LQ>                            <SVD>

        For one site gates when one of the 'split' methods is supplied
        ``contract=True`` is assumed.
        """
        check_opt("contract", contract, (False, True, 'split', 'reduce-split'))

        tn = self if inplace else self.copy()

        if isinstance(inds, str):
            inds = (inds,)

        ng = len(inds)
        if (ng == 1) and contract:
            # single site gate, eagerly applied so contract in directly ->
            # useful short circuit  as it maintains the index structure exactly
            ix, = inds
            t, = tn.ind_map[ix]
            t.gate_(G, ix)
            return tn

        ndimG = ndim(G)
        ds = [tn.ind_size(ix) for ix in inds]

        if ndimG != 2 * ng:
            # gate supplied as matrix, factorize it
            G = reshape(G, ds * 2)

        for i, d in enumerate(G.shape):
            if d != ds[i % ng]:
                raise ValueError(
                    f"Gate with shape {G.shape} doesn't match indices {inds} "
                    f"with dimensions {ds}. "
                )

        # new indices to join old physical sites to new gate
        bnds = [rand_uuid() for _ in range(ng)]
        reindex_map = dict(zip(inds, bnds))

        # tensor representing the gate
        tags = tags_to_oset(tags)
        tG = Tensor(G, inds=(*inds, *bnds), tags=tags, left_inds=bnds)

        if contract is False:
            #
            #       │   │      <- site_ix
            #       GGGGG
            #       │╱  │╱     <- bnds
            #     ──●───●──
            #      ╱   ╱
            #
            tn.reindex_(reindex_map)
            tn |= tG
            return tn

        tids = self._get_tids_from_inds(inds, 'any')

        if (contract is True) or (len(tids) == 1):
            #
            #       │╱  │╱
            #     ──GGGGG──
            #      ╱   ╱
            #
            tn.reindex_(reindex_map)

            # get the sites that used to have the physical indices
            site_tids = tn._get_tids_from_inds(bnds, which='any')

            # pop the sites, contract, then re-add
            pts = [tn._pop_tensor(tid) for tid in site_tids]
            tn |= tensor_contract(*pts, tG)

            return tn

        # get the two tensors and their current shared indices etc.
        ixl, ixr = inds
        tl, tr = tn._inds_get(ixl, ixr)
        bnds_l, (bix,), bnds_r = group_inds(tl, tr)

        if contract == 'split':
            #
            #       │╱  │╱         │╱  │╱
            #     ──GGGGG──  ->  ──G~~~G──
            #      ╱   ╱          ╱   ╱
            #

            # contract with new gate tensor
            tlGr = tensor_contract(
                tl.reindex(reindex_map),
                tr.reindex(reindex_map),
                tG)

            # decompose back into two tensors
            tln, *maybe_svals, trn = tlGr.split(
                left_inds=bnds_l, right_inds=bnds_r,
                bond_ind=bix, get='tensors', **compress_opts)

        if contract == 'reduce-split':
            # move physical inds on reduced tensors
            #
            #       │   │             │ │
            #       GGGGG             GGG
            #       │╱  │╱   ->     ╱ │ │   ╱
            #     ──●───●──      ──>──●─●──<──
            #      ╱   ╱          ╱       ╱
            #
            tmp_bix_l = rand_uuid()
            tl_Q, tl_R = tl.split(left_inds=None, right_inds=[bix, ixl],
                                  method='qr', bond_ind=tmp_bix_l)
            tmp_bix_r = rand_uuid()
            tr_L, tr_Q = tr.split(left_inds=[bix, ixr], right_inds=None,
                                  method='lq', bond_ind=tmp_bix_r)

            # contract reduced tensors with gate tensor
            #
            #          │ │
            #          GGG                │ │
            #        ╱ │ │   ╱    ->    ╱ │ │   ╱
            #     ──>──●─●──<──      ──>──LGR──<──
            #      ╱       ╱          ╱       ╱
            #
            tlGr = tensor_contract(
                tl_R.reindex(reindex_map),
                tr_L.reindex(reindex_map),
                tG)

            # split to find new reduced factors
            #
            #          │ │                │ │
            #        ╱ │ │   ╱    ->    ╱ │ │   ╱
            #     ──>──LGR──<──      ──>──L=R──<──
            #      ╱       ╱          ╱       ╱
            #
            tl_R, *maybe_svals, tr_L = tlGr.split(
                left_inds=[tmp_bix_l, ixl], right_inds=[tmp_bix_r, ixr],
                bond_ind=bix, get='tensors', **compress_opts)

            # absorb reduced factors back into site tensors
            #
            #          │ │             │   │
            #        ╱ │ │   ╱         │╱  │╱
            #     ──>──L=R──<──  ->  ──●───●──
            #      ╱       ╱          ╱   ╱
            #
            tln = tl_Q @ tl_R
            trn = tr_L @ tr_Q

        # if singular values are returned (``absorb=None``) check if we should
        #     return them via ``info``, e.g. for ``SimpleUpdate`
        if maybe_svals and info is not None:
            s = next(iter(maybe_svals)).data
            info['singular_values', bix] = s

        # update original tensors
        tl.modify(data=tln.transpose_like_(tl).data)
        tr.modify(data=trn.transpose_like_(tr).data)

        return tn

    gate_inds_ = functools.partialmethod(gate_inds, inplace=True)


    def _compute_bond_env(
        self, tid1, tid2,
        select_local_distance=None,
        select_local_opts=None,
        max_bond=None,
        cutoff=None,
        method='contract_around',
        contract_around_opts=None,
        contract_compressed_opts=None,
        optimize='auto-hq',
        include=None,
        exclude=None,
    ):
        """Compute the local tensor environment of the bond(s), if cut,
        between two tensors.
        """
        # the TN we will start with
        if select_local_distance is include is exclude is None:
            # ... either the full TN
            tn_env = self.copy()
        else:
            # ... or just a local patch of the TN (with dangling bonds removed)
            select_local_opts = ensure_dict(select_local_opts)
            select_local_opts.setdefault('reduce_outer', 'svd')

            tn_env = self._select_local_tids(
                (tid1, tid2), max_distance=select_local_distance,
                virtual=False, include=include, exclude=exclude,
                **select_local_opts)

            # not propagated by _select_local_tids
            tn_env.exponent = self.exponent

        # cut the bond between the two target tensors in the local TN
        t1 = tn_env.tensor_map[tid1]
        t2 = tn_env.tensor_map[tid2]
        bond, = t1.bonds(t2)
        lcut = rand_uuid()
        rcut = rand_uuid()
        t1.reindex_({bond: lcut})
        t2.reindex_({bond: rcut})

        if max_bond is not None:
            if method == 'contract_around':
                tn_env._contract_around_tids(
                    (tid1, tid2), max_bond=max_bond, cutoff=cutoff,
                    **ensure_dict(contract_around_opts))

            elif method == 'contract_compressed':
                tn_env.contract_compressed_(
                    max_bond=max_bond, cutoff=cutoff,
                    **ensure_dict(contract_compressed_opts))

            else:
                raise ValueError(f'Unknown method: {method}')

        return tn_env.to_dense([lcut], [rcut], optimize=optimize)

    def _compress_between_full_bond_tids(
        self,
        tid1,
        tid2,
        max_bond,
        cutoff=0.0,
        absorb='both',
        renorm=False,
        method='eigh',
        select_local_distance=None,
        select_local_opts=None,
        env_max_bond='max_bond',
        env_cutoff='cutoff',
        env_method='contract_around',
        contract_around_opts=None,
        contract_compressed_opts=None,
        env_optimize='auto-hq',
        include=None,
        exclude=None,
    ):
        if env_max_bond == 'max_bond':
            env_max_bond = max_bond
        if env_cutoff == 'cutoff':
            env_cutoff = cutoff

        ta = self.tensor_map[tid1]
        tb = self.tensor_map[tid2]

        # handle multibonds and no shared bonds
        _, bond, _ = tensor_make_single_bond(ta, tb)
        if not bond:
            return

        E = self._compute_bond_env(
            tid1, tid2,
            select_local_distance=select_local_distance,
            select_local_opts=select_local_opts,
            max_bond=env_max_bond,
            cutoff=env_cutoff,
            method=env_method,
            contract_around_opts=contract_around_opts,
            contract_compressed_opts=contract_compressed_opts,
            optimize=env_optimize,
            include=include,
            exclude=exclude,
        )

        Cl, Cr = decomp.similarity_compress(
            E, max_bond, method=method, renorm=renorm)

        # absorb them into the tensors to compress this bond
        ta.gate_(Cr, bond)
        tb.gate_(Cl.T, bond)

        if absorb != 'both':
            tensor_canonize_bond(ta, tb, absorb=absorb)

    def _compress_between_local_fit(
        self,
        tid1,
        tid2,
        max_bond,
        cutoff=0.0,
        absorb='both',
        method='als',
        select_local_distance=1,
        select_local_opts=None,
        include=None,
        exclude=None,
        **fit_opts
    ):
        if cutoff != 0.0:
            import warnings
            warnings.warn("Non-zero cutoff ignored by local fit compress.")

        select_local_opts = ensure_dict(select_local_opts)
        tn_loc_target = self._select_local_tids(
            (tid1, tid2),
            max_distance=select_local_distance, virtual=False,
            include=include, exclude=exclude, **select_local_opts)

        tn_loc_compress = tn_loc_target.copy()
        tn_loc_compress._compress_between_tids(
            tid1, tid2, max_bond=max_bond, cutoff=0.0)

        tn_loc_opt = tn_loc_compress.fit_(
            tn_loc_target, method=method, **fit_opts)

        for tid, t in tn_loc_opt.tensor_map.items():
            self.tensor_map[tid].modify(data=t.data)

        if absorb != 'both':
            self._canonize_between_tids(tid1, tid2, absorb=absorb)

    def _compress_between_tids(
        self,
        tid1,
        tid2,
        max_bond=None,
        cutoff=1e-10,
        absorb='both',
        canonize_distance=None,
        canonize_opts=None,
        canonize_after_distance=None,
        canonize_after_opts=None,
        mode='basic',
        equalize_norms=False,
        gauges=None,
        gauge_smudge=1e-6,
        callback=None,
        **compress_opts
    ):
        ta = self.tensor_map[tid1]
        tb = self.tensor_map[tid2]

        lix, bix, rix = tensor_make_single_bond(ta, tb, gauges=gauges)
        if not bix:
            return

        if (max_bond is not None) and (cutoff == 0.0):
            lsize = prod(map(self.ind_size, lix))
            rsize = prod(map(self.ind_size, rix))
            if (lsize <= max_bond) or (rsize <= max_bond):
                # special case - fixing any orthonormal basis for the left or
                # right tensor (whichever has smallest outer dimensions) will
                # produce the required compression without any SVD
                compress_absorb = 'right' if lsize <= rsize else 'left'
                tensor_canonize_bond(
                    ta, tb, absorb=compress_absorb,
                    gauges=gauges, gauge_smudge=gauge_smudge)

                if absorb != compress_absorb:
                    tensor_canonize_bond(
                        ta, tb, absorb=absorb,
                        gauges=gauges, gauge_smudge=gauge_smudge)

                if equalize_norms:
                    self.strip_exponent(tid1, equalize_norms)
                    self.strip_exponent(tid2, equalize_norms)

                return

        if canonize_distance:
            # gauge around pair by absorbing QR factors along bonds
            canonize_opts = ensure_dict(canonize_opts)
            canonize_opts.setdefault('equalize_norms', equalize_norms)
            self._canonize_around_tids(
                (tid1, tid2),
                gauges=gauges,
                max_distance=canonize_distance,
                **canonize_opts
            )

        compress_opts['max_bond'] = max_bond
        compress_opts['cutoff'] = cutoff
        compress_opts['absorb'] = absorb
        if gauges is not None:
            compress_opts['gauges'] = gauges
            compress_opts['gauge_smudge'] = gauge_smudge

        if mode == 'basic':
            tensor_compress_bond(ta, tb, **compress_opts)

        elif mode == 'full-bond':
            self._compress_between_full_bond_tids(tid1, tid2, **compress_opts)


        elif mode == 'local-fit':
            self._compress_between_local_fit(tid1, tid2, **compress_opts)

        else:
            # assume callable
            mode(self, tid1, tid2, **compress_opts)

        if equalize_norms:
            self.strip_exponent(tid1, equalize_norms)
            self.strip_exponent(tid2, equalize_norms)

        if canonize_after_distance:
            # 'undo' the inwards canonization
            canonize_after_opts = ensure_dict(canonize_after_opts)
            self._gauge_local_tids(
                tids=(tid1, tid2),
                max_distance=canonize_after_distance,
                gauges=gauges,
                **canonize_after_opts
            )

        if callback is not None:
            callback(self, (tid1, tid2))

    def compress_between(
        self,
        tags1,
        tags2,
        max_bond=None,
        cutoff=1e-10,
        absorb='both',
        canonize_distance=0,
        canonize_opts=None,
        equalize_norms=False,
        **compress_opts,
    ):
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
        max_bond : int or None, optional
            The maxmimum bond dimension.
        cutoff : float, optional
            The singular value cutoff to use.
        canonize_distance : int, optional
            How far to locally canonize around the target tensors first.
        canonize_opts : None or dict, optional
            Other options for the local canonization.
        equalize_norms : bool or float, optional
            If set, rescale the norms of all tensors modified to this value,
            stripping the rescaling factor into the ``exponent`` attribute.
        compress_opts
            Supplied to :func:`~quimb.tensor.tensor_core.tensor_compress_bond`.

        See Also
        --------
        canonize_between
        """
        tid1, = self._get_tids_from_tags(tags1, which='all')
        tid2, = self._get_tids_from_tags(tags2, which='all')

        self._compress_between_tids(
            tid1, tid2,
            max_bond=max_bond,
            cutoff=cutoff,
            absorb=absorb,
            canonize_distance=canonize_distance,
            canonize_opts=canonize_opts,
            equalize_norms=equalize_norms,
            **compress_opts)

    def compress_all(self, inplace=False, **compress_opts):
        """Inplace compress all bonds in this network.
        """
        tn = self if inplace else self.copy()
        tn.fuse_multibonds_()

        for ix in tuple(tn.ind_map):
            try:
                tid1, tid2 = tn.ind_map[ix]
            except (ValueError, KeyError):
                # not a bond, or index already compressed away
                continue
            tn._compress_between_tids(tid1, tid2, **compress_opts)

        return tn

    compress_all_ = functools.partialmethod(compress_all, inplace=True)

    def _canonize_between_tids(
        self,
        tid1,
        tid2,
        absorb='right',
        equalize_norms=False,
        **canonize_opts,
    ):
        Tl = self.tensor_map[tid1]
        Tr = self.tensor_map[tid2]
        tensor_canonize_bond(Tl, Tr, absorb=absorb, **canonize_opts)

        if equalize_norms:
            self.strip_exponent(tid1, equalize_norms)
            self.strip_exponent(tid2, equalize_norms)

    def canonize_between(self, tags1, tags2, absorb='right', **canonize_opts):
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
        absorb : {'left', 'both', 'right'}, optional
            Which side of the bond to absorb the non-isometric operator.
        canonize_opts
            Supplied to :func:`~quimb.tensor.tensor_core.tensor_canonize_bond`.

        See Also
        --------
        compress_between
        """
        tid1, = self._get_tids_from_tags(tags1, which='all')
        tid2, = self._get_tids_from_tags(tags2, which='all')
        self._canonize_between_tids(tid1, tid2, absorb=absorb, **canonize_opts)

    def reduce_inds_onto_bond(self, inda, indb, tags=None, drop_tags=False):
        """Use QR factorization to 'pull' the indices ``inda`` and ``indb`` off
        of their respective tensors and onto the bond between them. This is an
        inplace operation.
        """
        tida, = self._get_tids_from_inds(inda)
        tidb, = self._get_tids_from_inds(indb)
        ta, tb = self._tids_get(tida, tidb)
        bix = bonds(ta, tb)

        if ta.ndim > 3:
            self._split_tensor_tid(
                tida, left_inds=None, right_inds=[inda, *bix], method='qr')
            # get new location of ind
            tida, = self._get_tids_from_inds(inda)
        else:
            drop_tags = False

        if tb.ndim > 3:
            self._split_tensor_tid(
                tidb, left_inds=None, right_inds=[indb, *bix], method='qr')
            # get new location of ind
            tidb, = self._get_tids_from_inds(indb)
        else:
            drop_tags = False

        # contract the reduced factors and get the tensor
        self._contract_between_tids(tida, tidb)
        tab, = self._inds_get(inda, indb)

        # modify with the desired tags
        tags = tags_to_oset(tags)
        if drop_tags:
            tab.modify(tags=tags)
        else:
            tab.modify(tags=tab.tags | tags)

    def _get_neighbor_tids(self, tids, exclude_inds=()):
        """Get the tids of tensors connected to the tensor at ``tid``.
        """
        tids = tags_to_oset(tids)

        neighbors = oset_union(
            self.ind_map[ind]
            for tid in tids
            for ind in self.tensor_map[tid].inds
            if ind not in exclude_inds
        )

        # discard rather than remove to account for scalar ``tid`` tensor
        neighbors -= tids

        return neighbors

    def subgraphs(self, virtual=False):
        """Split this tensor network into disconneceted subgraphs.

        Parameters
        ----------
        virtual : bool, optional
            Whether the tensor networks should view the original tensors or
            not - by default take copies.

        Returns
        -------
        list[TensorNetwork]
        """
        groups = []
        tids = oset(self.tensor_map)

        # check all nodes
        while tids:

            # get a remaining node
            tid0 = tids.popright()
            queue = [tid0]
            group = oset(queue)

            while queue:
                # expand it until no neighbors
                tid = queue.pop()
                for tid_n in self._get_neighbor_tids(tid):
                    if tid_n in group:
                        continue
                    else:
                        group.add(tid_n)
                        queue.append(tid_n)

            # remove current subgraph and continue
            tids -= group
            groups.append(group)

        return [self._select_tids(group, virtual=virtual) for group in groups]

    def get_tree_span(
        self,
        tids,
        min_distance=0,
        max_distance=None,
        include=None,
        exclude=None,
        ndim_sort='max',
        distance_sort='min',
        sorter=None,
        weight_bonds=True,
        inwards=True,
    ):
        """Generate a tree on the tensor network graph, fanning out from the
        tensors identified by ``tids``, up to a maximum of ``max_distance``
        away. The tree can be visualized with
        :meth:`~quimb.tensor.tensor_core.TensorNetwork.draw_tree_span`.

        Parameters
        ----------
        tids : sequence of str
            The nodes that define the region to span out of.
        min_distance : int, optional
            Don't add edges to the tree until this far from the region. For
            example, ``1`` will not include the last merges from neighboring
            tensors in the region defined by ``tids``.
        max_distance : None or int, optional
            Terminate branches once they reach this far away. If ``None`` there
            is no limit,
        include : sequence of str, optional
            If specified, only ``tids`` specified here can be part of the tree.
        exclude : sequence of str, optional
            If specified, ``tids`` specified here cannot be part of the tree.
        ndim_sort : {'min', 'max', 'none'}, optional
            When expanding the tree, how to choose what nodes to expand to
            next, once connectivity to the current surface has been taken into
            account.
        distance_sort : {'min', 'max', 'none'}, optional
            When expanding the tree, how to choose what nodes to expand to
            next, once connectivity to the current surface has been taken into
            account.
        weight_bonds : bool, optional
            Whether to weight the 'connection' of a candidate tensor to expand
            out to using bond size as well as number of bonds.

        Returns
        -------
        list[(str, str, int)]
            The ordered list of merges, each given as tuple ``(tid1, tid2, d)``
            indicating merge ``tid1 -> tid2`` at distance ``d``.

        See Also
        --------
        draw_tree_span
        """
        # current tensors in the tree -> we will grow this
        region = oset(tids)

        # check if we should only allow a certain set of nodes
        if include is None:
            include = oset(self.tensor_map)
        elif not isinstance(include, oset):
            include = oset(include)

        allowed = include - region

        # check if we should explicitly ignore some nodes
        if exclude is not None:
            if not isinstance(exclude, oset):
                exclude = oset(exclude)
            allowed -= exclude

        # possible merges of neighbors into the region
        candidates = []

        # actual merges we have performed, defining the tree
        merges = {}

        # distance to the original region
        distances = {tid: 0 for tid in region}

        # how many times (or weight) that neighbors are connected to the region
        connectivity = collections.defaultdict(lambda: 0)

        # given equal connectivity compare neighbors based on
        #      min/max distance and min/max ndim
        distance_coeff = {'min': -1, 'max': 1, 'none': 0}[distance_sort]
        ndim_coeff = {'min': -1, 'max': 1, 'none': 0}[ndim_sort]

        def _check_candidate(tid_surface, tid_neighb):
            """Check the expansion of ``tid_surface`` to ``tid_neighb``.
            """
            if (tid_neighb in region) or (tid_neighb not in allowed):
                # we've already absorbed it, or we're not allowed to
                return

            if tid_neighb not in distances:
                # defines a new spanning tree edge
                merges[tid_neighb] = tid_surface
                # graph distance to original region
                new_d = distances[tid_surface] + 1
                distances[tid_neighb] = new_d
                if (max_distance is None) or (new_d <= max_distance):
                    candidates.append(tid_neighb)

            # keep track of how connected to the current surface potential new
            # nodes are
            if weight_bonds:
                connectivity[tid_neighb] += math.log2(bonds_size(
                    self.tensor_map[tid_surface], self.tensor_map[tid_neighb]
                ))
            else:
                connectivity[tid_neighb] += 1

        if sorter is None:
            def _sorter(t):
                # how to pick which tensor to absorb into the expanding surface
                # here, choose the candidate that is most connected to current
                # surface, breaking ties with how close it is to the original
                # region, and how many dimensions it has
                return (
                    connectivity[t],
                    ndim_coeff * self.tensor_map[t].ndim,
                    distance_coeff * distances[t],
                )
        else:
            _sorter = functools.partial(
                sorter, tn=self, distances=distances,
                connectivity=connectivity)

        # setup the initial region and candidate nodes to expand to
        for tid_surface in region:
            for tid_next in self._get_neighbor_tids(tid_surface):
                _check_candidate(tid_surface, tid_next)

        # generate the sequence of tensor merges
        seq = []
        while candidates:
            # choose the *highest* scoring candidate
            candidates.sort(key=_sorter)
            tid_surface = candidates.pop()
            region.add(tid_surface)

            if distances[tid_surface] > min_distance:
                # checking distance allows the innermost merges to be ignored,
                # for example, to contract an environment around a region
                seq.append(
                    (tid_surface, merges[tid_surface], distances[tid_surface])
                )

            # check all the neighbors of the tensor we've just expanded to
            for tid_next in self._get_neighbor_tids(tid_surface):
                _check_candidate(tid_surface, tid_next)

        if inwards:
            # make the sequence of merges flow inwards
            seq.reverse()

        return seq

    def _draw_tree_span_tids(
        self,
        tids,
        span=None,
        min_distance=0,
        max_distance=None,
        include=None,
        exclude=None,
        ndim_sort='max',
        distance_sort='min',
        sorter=None,
        weight_bonds=True,
        color='order',
        colormap='Spectral',
        **draw_opts,
    ):
        tn = self.copy()

        tix = oset()
        ds = oset()

        if span is None:
            span = tn.get_tree_span(
                tids,
                min_distance=min_distance,
                max_distance=max_distance,
                include=include,
                exclude=exclude,
                ndim_sort=ndim_sort,
                distance_sort=distance_sort,
                sorter=sorter,
                weight_bonds=weight_bonds)

        for i, (tid1, tid2, d) in enumerate(span):
            # get the tensors on either side of this tree edge
            t1, t2 = tn.tensor_map[tid1], tn.tensor_map[tid2]

            # get the ind(s) connecting them
            tix |= oset(bonds(t1, t2))

            if color == 'distance':
                # tag the outer tensor with distance ``d``
                t1.add_tag(f'D{d}')
                ds.add(d)
            elif color == 'order':
                d = len(span) - i
                t1.add_tag(f'D{d}')
                ds.add(d)

        if colormap is not None:
            if isinstance(colormap, str):
                import matplotlib.cm
                cmap = getattr(matplotlib.cm, colormap)
            else:
                cmap = colormap
            custom_colors = cmap(np.linspace(0, 1, len(ds)))
        else:
            custom_colors = None

        draw_opts.setdefault('legend', False)
        draw_opts.setdefault('edge_color', (0.85, 0.85, 0.85))
        draw_opts.setdefault('highlight_inds', tix)
        draw_opts.setdefault('custom_colors', custom_colors)

        return tn.draw(color=[f'D{d}' for d in sorted(ds)], **draw_opts)

    def draw_tree_span(
        self,
        tags,
        which='all',
        min_distance=0,
        max_distance=None,
        include=None,
        exclude=None,
        ndim_sort='max',
        distance_sort='min',
        weight_bonds=True,
        color='order',
        colormap='Spectral',
        **draw_opts,
    ):
        """Visualize a generated tree span out of the tensors tagged by
        ``tags``.

        Parameters
        ----------
        tags : str or sequence of str
            Tags specifiying a region of tensors to span out of.
        which : {'all', 'any': '!all', '!any'}, optional
            How to select tensors based on the tags.
        min_distance : int, optional
            See :meth:`~quimb.tensor.tensor_core.TensorNetwork.get_tree_span`.
        max_distance : None or int, optional
            See :meth:`~quimb.tensor.tensor_core.TensorNetwork.get_tree_span`.
        include : sequence of str, optional
            See :meth:`~quimb.tensor.tensor_core.TensorNetwork.get_tree_span`.
        exclude : sequence of str, optional
            See :meth:`~quimb.tensor.tensor_core.TensorNetwork.get_tree_span`.
        distance_sort : {'min', 'max'}, optional
            See :meth:`~quimb.tensor.tensor_core.TensorNetwork.get_tree_span`.
        color : {'order', 'distance'}, optional
            Whether to color nodes based on the order of the contraction or the
            graph distance from the specified region.
        colormap : str
            The name of a ``matplotlib`` colormap to use.

        See Also
        --------
        get_tree_span
        """
        return self._draw_tree_span_tids(
            self._get_tids_from_tags(tags, which=which),
            min_distance=min_distance,
            max_distance=max_distance,
            include=include,
            exclude=exclude,
            ndim_sort=ndim_sort,
            distance_sort=distance_sort,
            weight_bonds=weight_bonds,
            color=color,
            colormap=colormap,
            **draw_opts)

    graph_tree_span = draw_tree_span

    def _canonize_around_tids(
        self,
        tids,
        min_distance=0,
        max_distance=None,
        include=None,
        exclude=None,
        span_opts=None,
        absorb='right',
        gauge_links=False,
        link_absorb='both',
        inwards=True,
        **canonize_opts
    ):
        span_opts = ensure_dict(span_opts)
        seq = self.get_tree_span(
            tids,
            min_distance=min_distance,
            max_distance=max_distance,
            include=include,
            exclude=exclude,
            inwards=inwards,
            **span_opts)

        if gauge_links:
            # if specified we first gauge *between* the branches
            branches = oset()
            merges = oset()
            links = oset()

            # work out which bonds are branch-to-branch
            for tid1, tid2, d in seq:
                branches.add(tid1)
                merges.add(frozenset((tid1, tid2)))

            for tid1 in branches:
                for tid1_neighb in self._get_neighbor_tids(tid1):
                    if tid1_neighb not in branches:
                        # connects to out of tree -> ignore
                        continue
                    link = frozenset((tid1, tid1_neighb))
                    if link in merges:
                        # connects along tree not between branches -> ignore
                        continue
                    links.add(link)

            # do a simple update style gauging of each link
            for _ in range(int(gauge_links)):
                for tid1, tid2 in links:
                    self._canonize_between_tids(
                        tid1, tid2, absorb=link_absorb, **canonize_opts)

        # gauge inwards *along* the branches
        for tid1, tid2, _ in seq:
            self._canonize_between_tids(
                tid1, tid2, absorb=absorb, **canonize_opts)

        return self

    def canonize_around(
        self,
        tags,
        which='all',
        min_distance=0,
        max_distance=None,
        include=None,
        exclude=None,
        span_opts=None,
        absorb='right',
        gauge_links=False,
        link_absorb='both',
        equalize_norms=False,
        inplace=False,
        **canonize_opts
    ):
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
        min_distance : int, optional
            How close, in terms of graph distance, to canonize tensors away.
            See :meth:`~quimb.tensor.tensor_core.TensorNetwork.get_tree_span`.
        max_distance : None or int, optional
            How far, in terms of graph distance, to canonize tensors away.
            See :meth:`~quimb.tensor.tensor_core.TensorNetwork.get_tree_span`.
        include : sequence of str, optional
            How to build the spanning tree to canonize along.
            See :meth:`~quimb.tensor.tensor_core.TensorNetwork.get_tree_span`.
        exclude : sequence of str, optional
            How to build the spanning tree to canonize along.
            See :meth:`~quimb.tensor.tensor_core.TensorNetwork.get_tree_span`.
        distance_sort  {'min', 'max'}, optional
            How to build the spanning tree to canonize along.
            See :meth:`~quimb.tensor.tensor_core.TensorNetwork.get_tree_span`.
        absorb : {'right', 'left', 'both'}, optional
            As we canonize inwards from tensor A to tensor B which to absorb
            the singular values into.
        gauge_links : bool, optional
            Whether to gauge the links *between* branches of the spanning tree
            generated (in a Simple Update like fashion).
        link_absorb : {'both', 'right', 'left'}, optional
            If performing the link gauging, how to absorb the singular values.
        equalize_norms : bool or float, optional
            Scale the norms of tensors acted on to this value, accumulating the
            log10 scaled factors in ``self.exponent``.
        inplace : bool, optional
            Whether to perform the canonization inplace.

        Returns
        -------
        TensorNetwork

        See Also
        --------
        get_tree_span
        """
        tn = self if inplace else self.copy()

        # the set of tensor tids that are in the 'bulk'
        border = tn._get_tids_from_tags(tags, which=which)

        return tn._canonize_around_tids(
            border,
            min_distance=min_distance,
            max_distance=max_distance,
            include=include,
            exclude=exclude,
            span_opts=span_opts,
            absorb=absorb,
            gauge_links=gauge_links,
            link_absorb=link_absorb,
            equalize_norms=equalize_norms,
            **canonize_opts)

    canonize_around_ = functools.partialmethod(canonize_around, inplace=True)

    def gauge_all_canonize(
        self,
        max_iterations=5,
        absorb='both',
        equalize_norms=False,
        inplace=False,
        **kwargs,
    ):
        """Iterative gauge all the bonds in this tensor network with a basic
        'canonization' strategy.
        """
        tn = self if inplace else self.copy()

        for _ in range(max_iterations):
            for ind in tuple(tn.ind_map.keys()):
                try:
                    tid1, tid2 = tn.ind_map[ind]
                except (KeyError, ValueError):
                    # fused multibond (removed) or not a bond (len(tids != 2))
                    continue
                tn._canonize_between_tids(tid1, tid2, absorb=absorb)

                if equalize_norms:
                    tn.strip_exponent(tid1, equalize_norms)
                    tn.strip_exponent(tid2, equalize_norms)

        if equalize_norms is True:
            # this also redistributes the any collected norm exponent
            tn.equalize_norms_()

        return tn

    gauge_all_canonize_ = functools.partialmethod(
        gauge_all_canonize, inplace=True)

    def gauge_all_simple(
        self,
        max_iterations=5,
        tol=0.0,
        smudge=1e-12,
        power=1.0,
        gauges=None,
        inplace=False,
    ):
        """Iterative gauge all the bonds in this tensor network with a 'simple
        update' like strategy.
        """
        tn = self if inplace else self.copy()

        # every index in the TN
        inds = list(tn.ind_map)

        # the vector 'gauges' that will live on the bonds
        gauges_supplied = gauges is not None
        if not gauges_supplied:
            gauges = {}

        # for retrieving singular values
        info = {}

        # accrue scaling to avoid numerical blow-ups
        nfact = 0.0

        it = 0
        not_converged = True
        while not_converged and it < max_iterations:

            # can only converge if tol > 0.0
            all_converged = tol > 0.0

            for ind in inds:
                try:
                    tid1, tid2 = tn.ind_map[ind]
                except (KeyError, ValueError):
                    # fused multibond (removed) or not a bond (len(tids != 2))
                    continue

                t1 = tn.tensor_map[tid1]
                t2 = tn.tensor_map[tid2]
                lix, bond, rix = tensor_make_single_bond(t1, t2, gauges=gauges)

                # absorb 'outer' gauges into tensors
                inv_gauges = []
                for t, ixs in ((t1, lix), (t2, rix)):
                    for ix in ixs:
                        try:
                            s = (gauges[ix] + smudge)**power
                        except KeyError:
                            continue
                        t.multiply_index_diagonal_(ix, s)
                        # keep track of how to invert gauge
                        inv_gauges.append((t, ix, 1 / s))

                # absorb the inner gauge, if it exists
                if bond in gauges:
                    t1.multiply_index_diagonal_(bond, gauges[bond])

                # perform SVD to get new bond gauge
                tensor_compress_bond(
                    t1, t2, absorb=None, info=info, cutoff=0.0)

                s = info['singular_values']
                smax = s[0]
                new_gauge = s / smax
                nfact = do('log10', smax) + nfact

                if tol > 0.0:
                    # check convergence
                    old_gauge = gauges.get(bond, 1.0)
                    sdiff = do('linalg.norm', old_gauge - new_gauge)
                    all_converged &= sdiff < tol

                # update inner gauge and undo outer gauges
                gauges[bond] = new_gauge
                for t, ix, inv_s in inv_gauges:
                    t.multiply_index_diagonal_(ix, inv_s)

            not_converged = not all_converged
            it += 1

        # redistribute the accrued scaling
        tn.multiply_each_(10**(nfact / tn.num_tensors))

        if not gauges_supplied:
            # absorb all bond gauges
            for ix, s in gauges.items():
                t1, t2 = map(tn.tensor_map.__getitem__, tn.ind_map[ix])
                s_1_2 = s**0.5
                t1.multiply_index_diagonal_(ix, s_1_2)
                t2.multiply_index_diagonal_(ix, s_1_2)

        return tn

    gauge_all_simple_ = functools.partialmethod(gauge_all_simple, inplace=True)

    def gauge_all_random(self, iterations=1, unitary=True, inplace=False):
        """Gauge all the bonds in this network randomly. This is largely for
        testing purposes.
        """
        tn = self if inplace else self.copy()

        for _ in range(iterations):
            for ix, tids in tn.ind_map.items():
                try:
                    tid1, tid2 = tids
                except (KeyError, ValueError):
                    continue

                t1 = tn.tensor_map[tid1]
                t2 = tn.tensor_map[tid2]

                d = t1.ind_size(ix)

                if unitary:
                    G = rand_uni(d, dtype=get_dtype_name(t1.data))
                    G = do('array', G, like=t1.data)
                    Ginv = dag(G)
                else:
                    G = rand_matrix(d, dtype=get_dtype_name(t1.data))
                    G = do('array', G, like=t1.data)
                    Ginv = do("linalg.inv", G)

                t1.gate_(G, ix)
                t2.gate_(Ginv.T, ix)

        return tn

    gauge_all_random_ = functools.partialmethod(gauge_all_random, inplace=True)

    def _gauge_local_tids(
        self,
        tids,
        max_distance=1,
        max_iterations='max_distance',
        method='canonize',
        inwards=False,
        include=None,
        exclude=None,
        **gauge_local_opts
    ):
        """Iteratively gauge all bonds in the local tensor network defined by
        ``tids`` according to one of several strategies.
        """
        if max_iterations == 'max_distance':
            max_iterations = max_distance

        tn_loc = self._select_local_tids(
            tids, max_distance=max_distance, inwards=inwards,
            virtual=True, include=include, exclude=exclude
        )

        if method == "canonize":
            tn_loc.gauge_all_canonize_(
                max_iterations=max_iterations, **gauge_local_opts)
        elif method == "simple":
            tn_loc.gauge_all_simple_(
                max_iterations=max_iterations, **gauge_local_opts)
        elif method == "random":
            tn_loc.gauge_all_random_(**gauge_local_opts)

        return tn_loc

    def gauge_local(
        self,
        tags,
        which='all',
        max_distance=1,
        max_iterations='max_distance',
        method='canonize',
        inplace=False,
        **gauge_local_opts
    ):
        """Iteratively gauge all bonds in the tagged sub tensor network
        according to one of several strategies.
        """
        tn = self if inplace else self.copy()
        tids = self._get_tids_from_tags(tags, which)
        tn._gauge_local_tids(
            tids, max_distance=max_distance, max_iterations=max_iterations,
            method=method, **gauge_local_opts)
        return tn

    gauge_local_ = functools.partialmethod(gauge_local, inplace=True)

    def gauge_simple_insert(self, gauges, smudge=0.0):
        """Insert the simple update style bond gauges found in ``gauges`` if
        they are present in this tensor network. The gauges inserted are also
        returned so that they can be removed later.

        Parameters
        ----------
        gauges : dict[str, array_like]
            The store of bond gauges, the keys being indices and the values
            being the vectors. Only bonds present in this dictionary will be
            gauged.

        Returns
        -------
        outer : list[(Tensor, str, array_like)]
            The sequence of gauges applied to outer indices, each a tuple of
            the tensor, the index and the gauge vector.
        inner : list[((Tensor, Tensor), str, array_like)]
            The sequence of gauges applied to inner indices, each a tuple of
            the two inner tensors, the inner bond and the gauge vector applied.
        """
        # absorb outer gauges fully into single tensor
        outer = []
        for ix in self.outer_inds():
            g = gauges.get(ix, None)
            if g is None:
                continue
            g = (g + smudge * g[0])
            t, = self._inds_get(ix)
            t.multiply_index_diagonal_(ix, g)
            outer.append((t, ix, g))

        # absorb inner gauges half and half into both tensors
        inner = []
        for ix in self.inner_inds():
            g = gauges.get(ix, None)
            if g is None:
                continue
            g = g**0.5
            tl, tr = self._inds_get(ix)
            tl.multiply_index_diagonal_(ix, g)
            tr.multiply_index_diagonal_(ix, g)
            inner.append(((tl, tr), ix, g))

        return outer, inner

    @staticmethod
    def gauge_simple_remove(outer=None, inner=None):
        """Remove the simple update style bond gauges inserted by
        ``gauge_simple_insert``.
        """
        while outer:
            t, ix, g = outer.pop()
            t.multiply_index_diagonal_(ix, g**-1)
        while inner:
            (tl, tr), ix, g = inner.pop()
            ginv = g**-1
            tl.multiply_index_diagonal_(ix, ginv)
            tr.multiply_index_diagonal_(ix, ginv)

    @contextlib.contextmanager
    def gauge_simple_temp(
        self,
        gauges,
        smudge=1e-12,
        ungauge_outer=True,
        ungauge_inner=True,
    ):
        """Context manager that temporarily inserts simple update style bond
        gauges into this tensor network, before optionally ungauging them.

        Parameters
        ----------
        self : TensorNetwork
            The TensorNetwork to be gauge-bonded.
        gauges : dict[str, array_like]
            The store of gauge bonds, the keys being indices and the values
            being the vectors. Only bonds present in this dictionary will be
            gauged.
        ungauge_outer : bool, optional
            Whether to ungauge the outer bonds.
        ungauge_inner : bool, optional
            Whether to ungauge the inner bonds.

        Yields
        ------
        outer : list[(Tensor, int, array_like)]
            The tensors, indices and gauges that were performed on outer
            indices.
        inner : list[((Tensor, Tensor), int, array_like)]
            The tensors, indices and gauges that were performed on inner bonds.

        Examples
        --------

            >>> tn = TN_rand_reg(10, 4, 3)
            >>> tn ^ all
            -51371.66630218866

            >>> gauges = {}
            >>> tn.gauge_all_simple_(gauges=gauges)
            >>> len(gauges)
            20

            >>> tn ^ all
            28702551.673767876

            >>> with gauged_bonds(tn, gauges):
            ...     # temporarily insert gauges
            ...     print(tn ^ all)
            -51371.66630218887

            >>> tn ^ all
            28702551.67376789

        """
        outer, inner = self.gauge_simple_insert(gauges, smudge=smudge)
        try:
            yield outer, inner
        finally:
            self.gauge_simple_remove(outer=outer if ungauge_outer else None,
                                     inner=inner if ungauge_inner else None)

    def _contract_compressed_tid_sequence(
        self,
        seq,
        max_bond=None,
        cutoff=1e-10,
        canonize_distance=0,
        canonize_opts=None,
        canonize_after_distance=0,
        canonize_after_opts=None,
        gauge_boundary_only=False,
        compress_late=True,
        compress_min_size=None,
        compress_opts=None,
        compress_span=False,
        compress_exclude=None,
        equalize_norms=False,
        gauges=None,
        gauge_smudge=1e-6,
        callback_pre_contract=None,
        callback_post_contract=None,
        callback_pre_compress=None,
        callback_post_compress=None,
        callback=None,
        preserve_tensor=False,
        progbar=False,
    ):
        if gauges is True:
            gauges = {}
            self.gauge_all_simple_(gauges=gauges)

        # the boundary - the set of intermediate tensors
        boundary = oset()

        def _do_contraction(tid1, tid2):
            """The inner closure that contracts the two tensors identified by
            ``tid1`` and ``tid2``.
            """
            if callback_pre_contract is not None:
                callback_pre_contract(self, (tid1, tid2))

            # new tensor is now at ``tid2``
            self._contract_between_tids(
                tid1, tid2, equalize_norms=equalize_norms, gauges=gauges,
            )

            # update the boundary
            boundary.add(tid2)

            if callback_post_contract is not None:
                callback_post_contract(self, tid2)

            return tid2, self.tensor_map[tid2]

        # keep track of pairs along the tree - often no point compressing these
        #     (potentially, on some complex graphs, one needs to compress)
        if not compress_span:
            dont_compress_pairs = {frozenset(s[:2]) for s in seq}
        else:
            # else just exclude the next few upcoming contractions, starting
            # with the first
            compress_span = int(compress_span)
            dont_compress_pairs = {
                frozenset(s[:2]) for s in seq[:compress_span]
            }

        def _should_skip_compression(tid1, tid2):
            """The inner closure deciding whether we should compress between
            ``tid1`` and tid2``.
            """
            if (compress_exclude is not None) and (tid2 in compress_exclude):
                # explicitly excluded from compression
                return True

            if frozenset((tid1, tid2)) in dont_compress_pairs:
                # or compressing pair that will be eventually or soon
                # contracted
                return True

            if compress_min_size is not None:
                t1, t2 = self._tids_get(tid1, tid2)
                new_size = t1.size * t2.size
                for ind in t1.bonds(t2):
                    new_size //= t1.ind_size(ind)
                if new_size < compress_min_size:
                    # not going to produce a large tensor so don't bother
                    # compressing
                    return True

        # options relating to locally canonizing around each compression
        if canonize_distance:
            canonize_opts = ensure_dict(canonize_opts)
            canonize_opts.setdefault('equalize_norms', equalize_norms)
            if gauge_boundary_only:
                canonize_opts['include'] = boundary
            else:
                canonize_opts['include'] = None

        # options relating to the compression itself
        compress_opts = ensure_dict(compress_opts)
        compress_opts.setdefault('absorb', 'left')

        # options relating to canonizing around tensors *after* compression
        if canonize_after_distance:
            canonize_after_opts = ensure_dict(canonize_after_opts)
            if gauge_boundary_only:
                canonize_after_opts['include'] = boundary
            else:
                canonize_after_opts['include'] = None

        # allow dynamic compresson options based on distance
        if callable(max_bond):
            chi_fn = max_bond
        else:
            def chi_fn(d):
                return max_bond

        if callable(cutoff):
            eps_fn = cutoff
        else:
            def eps_fn(d):
                return cutoff

        def _compress_neighbors(tid, t, d):
            """Inner closure that compresses tensor ``t`` with identifier
            ``tid`` at distance ``d``, with its neighbors.
            """
            chi = chi_fn(d)
            eps = eps_fn(d)

            if max_bond is None and eps == 0.0:
                # skip compression
                return

            for tid_neighb in self._get_neighbor_tids(tid):

                # first just check for accumulation of small multi-bonds
                t_neighb = self.tensor_map[tid_neighb]
                tensor_fuse_squeeze(t, t_neighb, gauges=gauges)

                if _should_skip_compression(tid, tid_neighb):
                    continue

                # check for compressing large shared (multi) bonds
                if bonds_size(t, t_neighb) > chi:
                    if callback_pre_compress is not None:
                        callback_pre_compress(self, (tid, tid_neighb))

                    self._compress_between_tids(
                        tid,
                        tid_neighb,
                        max_bond=chi,
                        cutoff=eps,
                        canonize_distance=canonize_distance,
                        canonize_opts=canonize_opts,
                        canonize_after_distance=canonize_after_distance,
                        canonize_after_opts=canonize_after_opts,
                        equalize_norms=equalize_norms,
                        gauges=gauges,
                        gauge_smudge=gauge_smudge,
                        **compress_opts
                    )

                    if callback_post_compress is not None:
                        callback_post_compress(self, (tid, tid_neighb))

        num_contractions = len(seq)

        if progbar:
            import tqdm
            max_size = 0.0
            pbar = tqdm.tqdm(total=num_contractions)
        else:
            max_size = pbar = None

        for i in range(num_contractions):
            # tid1 -> tid2 is inwards on the contraction tree, ``d`` is the
            # graph distance from the original region, optional
            tid1, tid2, *maybe_d = seq[i]

            if maybe_d:
                d, = maybe_d
            else:
                d = float('inf')

            if compress_span:
                # only keep track of the next few contractions to ignore
                # (note if False whole seq is already excluded)
                for s in seq[i + compress_span - 1:i + compress_span]:
                    dont_compress_pairs.add(frozenset(s[:2]))

            if compress_late:
                # we compress just before we have to contract involved tensors
                t1, t2 = self._tids_get(tid1, tid2)
                _compress_neighbors(tid1, t1, d)
                _compress_neighbors(tid2, t2, d)

            tid_new, t_new = _do_contraction(tid1, tid2)

            if progbar:
                new_size = math.log2(t_new.size)
                max_size = max(max_size, new_size)
                pbar.set_description(
                    f"log2[SIZE]: {new_size:.2f}/{max_size:.2f}")
                pbar.update()

            if not compress_late:
                # we compress as soon as we produce a new tensor
                _compress_neighbors(tid_new, t_new, d)

            if callback is not None:
                callback(self, tid_new)

        if progbar:
            pbar.close()

        if gauges:
            self.gauge_simple_insert(gauges)

        return maybe_unwrap(
            self,
            preserve_tensor=preserve_tensor,
            equalize_norms=equalize_norms,
        )

    def _contract_around_tids(
        self,
        tids,
        seq=None,
        min_distance=0,
        max_distance=None,
        span_opts=None,
        max_bond=None,
        cutoff=1e-10,
        canonize_distance=0,
        canonize_opts=None,
        gauge_boundary_only=False,
        compress_opts=None,
        equalize_norms=False,
        inplace=True,
        **kwargs,
    ):
        """Contract around ``tids``, by following a greedily generated
        spanning tree, and compressing whenever two tensors in the outer
        'boundary' share more than one index.
        """
        tn = self if inplace else self.copy()

        if seq is None:
            span_opts = ensure_dict(span_opts)
            seq = tn.get_tree_span(
                tids,
                min_distance=min_distance,
                max_distance=max_distance,
                **span_opts)

        canonize_opts = ensure_dict(canonize_opts)
        canonize_opts['exclude'] = oset(itertools.chain(
            canonize_opts.get('exclude', ()), tids
        ))

        return tn._contract_compressed_tid_sequence(
            seq,
            max_bond=max_bond,
            cutoff=cutoff,
            canonize_distance=canonize_distance,
            canonize_opts=canonize_opts,
            gauge_boundary_only=gauge_boundary_only,
            compress_opts=compress_opts,
            compress_exclude=tids,
            equalize_norms=equalize_norms,
            **kwargs)

    def compute_centralities(self):
        import cotengra as ctg
        hg = ctg.get_hypergraph(
            {tid: t.inds for tid, t in self.tensor_map.items()}
        )
        return hg.simple_centrality()

    def most_central_tid(self):
        cents = self.compute_centralities()
        return max((score, tid) for tid, score in cents.items())[1]

    def least_central_tid(self):
        cents = self.compute_centralities()
        return min((score, tid) for tid, score in cents.items())[1]

    def contract_around_center(self, **opts):
        tid_center = self.most_central_tid()
        opts.setdefault("span_opts", {})
        opts["span_opts"].setdefault("distance_sort", "min")
        opts["span_opts"].setdefault("ndim_sort", "max")
        return self.copy()._contract_around_tids([tid_center], **opts)

    def contract_around_corner(self, **opts):
        tid_corner = self.least_central_tid()
        opts.setdefault("span_opts", {})
        opts["span_opts"].setdefault("distance_sort", "max")
        opts["span_opts"].setdefault("ndim_sort", "min")
        return self.copy()._contract_around_tids([tid_corner], **opts)

    def contract_around(
        self,
        tags,
        which='all',
        min_distance=0,
        max_distance=None,
        span_opts=None,
        max_bond=None,
        cutoff=1e-10,
        canonize_distance=0,
        canonize_opts=None,
        gauge_boundary_only=False,
        compress_opts=None,
        equalize_norms=False,
        inplace=False,
        **kwargs
    ):
        """Perform a compressed contraction inwards towards the tensors
        identified by ``tags``.
        """
        tids = self._get_tids_from_tags(tags, which=which)

        return self._contract_around_tids(
            tids,
            min_distance=min_distance,
            max_distance=max_distance,
            span_opts=span_opts,
            max_bond=max_bond,
            cutoff=cutoff,
            canonize_distance=canonize_distance,
            canonize_opts=canonize_opts,
            gauge_boundary_only=gauge_boundary_only,
            compress_opts=compress_opts,
            equalize_norms=equalize_norms,
            inplace=inplace,
            **kwargs)

    contract_around_ = functools.partialmethod(contract_around, inplace=True)

    def contract_compressed(
        self,
        optimize,
        output_inds=None,
        max_bond=None,
        cutoff=1e-10,
        canonize_distance=0,
        canonize_opts=None,
        gauge_boundary_only=False,
        compress_opts=None,
        compress_span=True,
        compress_exclude=None,
        equalize_norms=False,
        callback_pre_contract=None,
        callback_post_contract=None,
        callback_pre_compress=None,
        callback_post_compress=None,
        callback=None,
        progbar=False,
        inplace=False,
        **kwargs
    ):
        tn = self if inplace else self.copy()

        path = tn.contraction_path(optimize, output_inds=output_inds)

        # generate the list of merges (tid1 -> tid2)
        tids = list(tn.tensor_map)
        seq = []
        for i, j in path:
            if i > j:
                i, j = j, i

            tid2 = tids.pop(j)
            tid1 = tids.pop(i)
            tids.append(tid2)

            seq.append((tid1, tid2))

        t = tn._contract_compressed_tid_sequence(
            seq=seq,
            max_bond=max_bond,
            cutoff=cutoff,
            canonize_distance=canonize_distance,
            canonize_opts=canonize_opts,
            gauge_boundary_only=gauge_boundary_only,
            compress_opts=compress_opts,
            compress_span=compress_span,
            compress_exclude=compress_exclude,
            equalize_norms=equalize_norms,
            callback_pre_contract=callback_pre_contract,
            callback_post_contract=callback_post_contract,
            callback_pre_compress=callback_pre_compress,
            callback_post_compress=callback_post_compress,
            callback=callback,
            progbar=progbar,
            **kwargs
        )

        if output_inds and t.inds != output_inds:
            t.transpose_(*output_inds)

        return t

    contract_compressed_ = functools.partialmethod(
        contract_compressed, inplace=True)

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

    def _cut_between_tids(self, tid1, tid2, left_ind, right_ind):
        TL, TR = self.tensor_map[tid1], self.tensor_map[tid2]
        bnd, = bonds(TL, TR)
        TL.reindex_({bnd: left_ind})
        TR.reindex_({bnd: right_ind})

    def cut_between(self, left_tags, right_tags, left_ind, right_ind):
        """Cut the bond between the tensors specified by ``left_tags`` and
        ``right_tags``, giving them the new inds ``left_ind`` and
        ``right_ind`` respectively.
        """
        tid1, = self._get_tids_from_tags(left_tags)
        tid2, = self._get_tids_from_tags(right_tags)
        self._cut_between_tids(tid1, tid2, left_ind, right_ind)

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

        for tid in oset_union(map(self.ind_map.__getitem__, selectors)):
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

    insert_operator_ = functools.partialmethod(insert_operator, inplace=True)

    def _insert_gauge_tids(
        self,
        U,
        tid1,
        tid2,
        Uinv=None,
        tol=1e-10,
        bond=None,
    ):
        t1, t2 = self._tids_get(tid1, tid2)

        if bond is None:
            bond, = t1.bonds(t2)

        if Uinv is None:
            Uinv = do('linalg.inv', U)

            # if we get wildly larger inverse due to singular U, try pseudo-inv
            if vdot(Uinv, Uinv) / vdot(U, U) > 1 / tol:
                Uinv = do('linalg.pinv', U, rcond=tol**0.5)

            # if still wildly larger inverse raise an error
            if vdot(Uinv, Uinv) / vdot(U, U) > 1 / tol:
                raise np.linalg.LinAlgError("Ill conditioned inverse.")

        t1.gate_(Uinv.T, bond)
        t2.gate_(U, bond)

    def insert_gauge(self, U, where1, where2, Uinv=None, tol=1e-10):
        """Insert the gauge transformation ``U^-1 @ U`` into the bond between
        the tensors, ``T1`` and ``T2``, defined by ``where1`` and ``where2``.
        The resulting tensors at those locations will be ``T1 @ U^-1`` and
        ``U @ T2``.

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
        tid1, = self._get_tids_from_tags(where1, which='all')
        tid2, = self._get_tids_from_tags(where2, which='all')
        self._insert_gauge_tids(U, tid1, tid2, Uinv=Uinv, tol=tol)

    # ----------------------- contracting the network ----------------------- #

    def contract_tags(self, tags, which='any', inplace=False, **opts):
        """Contract the tensors that match any or all of ``tags``.

        Parameters
        ----------
        tags : sequence of str
            The list of tags to filter the tensors by. Use ``all`` or ``...``
            (``Ellipsis``) to contract all tensors.
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
        contract, contract_cumulative
        """
        untagged_tn, tagged_ts = self.partition_tensors(
            tags, inplace=inplace, which=which)

        contracting_all = untagged_tn is None
        if not tagged_ts:
            raise ValueError("No tags were found - nothing to contract. "
                             "(Change this to a no-op maybe?)")

        contracted = tensor_contract(
            *tagged_ts, preserve_tensor=not contracting_all, **opts
        )

        if contracting_all:
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
        contract, contract_tags
        """
        tn = self if inplace else self.copy()
        c_tags = oset()

        for tags in tags_seq:
            # accumulate tags from each contractions
            c_tags |= tags_to_oset(tags)

            # peform the next contraction
            tn = tn.contract_tags(c_tags, inplace=True, which='any', **opts)

            if not isinstance(tn, TensorNetwork):
                # nothing more to contract
                break

        return tn

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
        contract_tags, contract_cumulative
        """
        if tags is all:
            return tensor_contract(*self, **opts)

        # this checks whether certain TN classes have a manually specified
        #     contraction pattern (e.g. 1D along the line)
        if self._CONTRACT_STRUCTURED:
            if (tags is ...) or isinstance(tags, slice):
                return self.contract_structured(tags, inplace=inplace, **opts)

        # else just contract those tensors specified by tags.
        return self.contract_tags(tags, inplace=inplace, **opts)

    contract_ = functools.partialmethod(contract, inplace=True)

    def contraction_path(self, optimize=None, **contract_opts):
        """Compute the contraction path, a sequence of (int, int), for
        the contraction of this entire tensor network using path optimizer
        ``optimize``.
        """
        if optimize is None:
            optimize = get_contract_strategy()
        return self.contract(
            all, optimize=optimize, get='path', **contract_opts)

    def contraction_info(self, optimize=None, **contract_opts):
        """Compute the ``opt_einsum.PathInfo`` object decsribing the
        contraction of this entire tensor network using path optimizer
        ``optimize``.
        """
        if optimize is None:
            optimize = get_contract_strategy()
        return self.contract(
            all, optimize=optimize, get='path-info', **contract_opts)

    def contraction_tree(
        self,
        optimize=None,
        output_inds=None,
    ):
        """Return the :class:`cotengra.ContractionTree` corresponding to
        contracting this entire tensor network with path finder ``optimize``.
        """
        import cotengra as ctg

        inputs, output, size_dict = self.get_inputs_output_size_dict(
            output_inds=output_inds)

        if optimize is None:
            optimize = get_contract_strategy()
        if isinstance(optimize, str):
            optimize = oe.paths.get_path_fn(optimize)

        if hasattr(optimize, 'search'):
            return optimize.search(inputs, output, size_dict)

        if callable(optimize):
            path = optimize(inputs, output, size_dict)
        else:
            path = optimize

        tree = ctg.ContractionTree.from_path(
            inputs, output, size_dict, path=path)

        return tree

    def contraction_width(self, optimize=None, **contract_opts):
        """Compute the 'contraction width' of this tensor network. This
        is defined as log2 of the maximum tensor size produced during the
        contraction sequence. If every index in the network has dimension 2
        this corresponds to the maximum rank tensor produced.
        """
        path_info = self.contraction_info(optimize, **contract_opts)
        return math.log2(path_info.largest_intermediate)

    def contraction_cost(self, optimize=None, **contract_opts):
        """Compute the 'contraction cost' of this tensor network. This
        is defined as log10 of the total number of scalar operations during the
        contraction sequence. Multiply by 2 to estimate FLOPS for real dtype,
        and by 8 to estimate FLOPS for complex dtype.
        """
        path_info = self.contraction_info(optimize, **contract_opts)
        return path_info.opt_cost / 2

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

    def aslinearoperator(self, left_inds, right_inds, ldims=None, rdims=None,
                         backend=None, optimize=None):
        """View this ``TensorNetwork`` as a
        :class:`~quimb.tensor.tensor_core.TNLinearOperator`.
        """
        return TNLinearOperator(self, left_inds, right_inds, ldims, rdims,
                                optimize=optimize, backend=backend)

    @functools.wraps(tensor_split)
    def split(self, left_inds, right_inds=None, **split_opts):
        """Decompose this tensor network across a bipartition of outer indices.

        This method matches ``Tensor.split`` by converting to a
        ``TNLinearOperator`` first. Note unless an iterative method is passed
        to ``method``, the full dense tensor will be contracted.
        """
        if right_inds is None:
            oix = self.outer_inds()
            right_inds = tuple(ix for ix in oix if ix not in left_inds)
        T = self.aslinearoperator(left_inds, right_inds)
        return T.split(**split_opts)

    def trace(self, left_inds, right_inds, **contract_opts):
        """Trace over ``left_inds`` joined with ``right_inds``
        """
        tn = self.reindex({u: l for u, l in zip(left_inds, right_inds)})
        return tn.contract_tags(..., **contract_opts)

    def to_dense(self, *inds_seq, to_qarray=True, **contract_opts):
        """Convert this network into an dense array, with a single dimension
        for each of inds in ``inds_seqs``. E.g. to convert several sites
        into a density matrix: ``TN.to_dense(('k0', 'k1'), ('b0', 'b1'))``.
        """
        tags = contract_opts.pop('tags', all)
        t = self.contract(
            tags,
            output_inds=tuple(concat(inds_seq)),
            preserve_tensor=True,
            **contract_opts
        )
        return t.to_dense(*inds_seq, to_qarray=to_qarray)

    @functools.wraps(tensor_network_distance)
    def distance(self, *args, **kwargs):
        return tensor_network_distance(self, *args, **kwargs)

    def fit(
        self,
        tn_target,
        method='als',
        tol=1e-9,
        inplace=False,
        progbar=False,
        **fitting_opts
    ):
        r"""Optimize the entries of this tensor network with respect to a least
        squares fit of ``tn_target`` which should have the same outer indices.
        Depending on ``method`` this calls
        :func:`~quimb.tensor.tensor_core.tensor_network_fit_als` or
        :func:`~quimb.tensor.tensor_core.tensor_network_fit_autodiff`. The
        quantity minimized is:

        .. math::

            D(A, B)
            = | A - B |_{\mathrm{fro}}
            = \mathrm{Tr} [(A - B)^{\dagger}(A - B)]^{1/2}
            = ( \langle A | A \rangle - 2 \mathrm{Re} \langle A | B \rangle|
            + \langle B | B \rangle ) ^{1/2}

        Parameters
        ----------
        tn_target : TensorNetwork
            The target tensor network to try and fit the current one to.
        method : {'als', 'autodiff'}, optional
            Whether to use alternating least squares (ALS) or automatic
            differentiation to perform the optimization. Generally ALS is
            better for simple geometries, autodiff better for complex ones.
        tol : float, optional
            The target norm distance.
        inplace : bool, optional
            Update the current tensor network in place.
        progbar : bool, optional
            Show a live progress bar of the fitting process.
        fitting_opts
            Supplied to either
            :func:`~quimb.tensor.tensor_core.tensor_network_fit_als` or
            :func:`~quimb.tensor.tensor_core.tensor_network_fit_autodiff`.

        Returns
        -------
        tn_opt : TensorNetwork
            The optimized tensor network.

        See Also
        --------
        tensor_network_fit_als, tensor_network_fit_autodiff,
        tensor_network_distance
        """
        check_opt('method', method, ('als', 'autodiff'))
        fitting_opts['tol'] = tol
        fitting_opts['inplace'] = inplace
        fitting_opts['progbar'] = progbar

        if method == 'autodiff':
            return tensor_network_fit_autodiff(self, tn_target, **fitting_opts)
        return tensor_network_fit_als(self, tn_target, **fitting_opts)

    fit_ = functools.partialmethod(fit, inplace=True)

    # --------------- information about indices and dimensions -------------- #

    @property
    def tags(self):
        return oset(self.tag_map)

    def all_inds(self):
        """Return a tuple of all indices (with repetition) in this network.
        """
        return tuple(self.ind_map)

    def ind_size(self, ind):
        """Find the size of ``ind``.
        """
        tid = next(iter(self.ind_map[ind]))
        return self.tensor_map[tid].ind_size(ind)

    def ind_sizes(self):
        """Get dict of each index mapped to its size.
        """
        return {i: self.ind_size(i) for i in self.ind_map}

    def inner_inds(self):
        """Tuple of interior indices, assumed to be any indices that appear
        twice or more (this only holds generally for non-hyper tensor
        networks).
        """
        return tuple(self._inner_inds)

    def outer_inds(self):
        """Tuple of exterior indices, assumed to be any lone indices (this only
        holds generally for non-hyper tensor networks).
        """
        return tuple(self._outer_inds)

    def outer_dims_inds(self):
        """Get the 'outer' pairs of dimension and indices, i.e. as if this
        tensor network was fully contracted.
        """
        return tuple((self.ind_size(i), i) for i in self._outer_inds)

    def compute_contracted_inds(self, *tids, output_inds=None):
        """Get the indices describing the tensor contraction of tensors
        corresponding to ``tids``.
        """
        if output_inds is None:
            output_inds = self._outer_inds

        # number of times each index appears on tensors
        freqs = frequencies(concat(
            self.tensor_map[tid].inds for tid in tids
        ))

        return tuple(
            ix for ix, c in freqs.items() if
            # ind also appears elsewhere -> keep
            (c != len(self.ind_map[ix])) or
            # explicitly in output -> keep
            (ix in output_inds)
        )

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

    def unitize(self, method='qr', allow_no_left_inds=False, inplace=False):
        """
        """
        tn = self if inplace else self.copy()
        for t in tn:
            if t.left_inds is None:
                if allow_no_left_inds:
                    continue
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

    def strip_exponent(self, tid_or_tensor, value=None):
        """Scale the elements of tensor corresponding to ``tid`` so that the
        norm of the array is some value, which defaults to ``1``. The log of
        the scaling factor, base 10, is then accumulated in the ``exponent``
        attribute.

        Parameters
        ----------
        tid : str or Tensor
            The tensor identifier or actual tensor.
        value : None or float, optional
            The value to scale the norm of the tensor to.
        """
        if (value is None) or (value is True):
            value = 1.0

        if isinstance(tid_or_tensor, Tensor):
            t = tid_or_tensor
        else:
            t = self.tensor_map[tid_or_tensor]

        stripped_factor = t.norm() / value
        t.modify(apply=lambda data: data / stripped_factor)
        self.exponent = self.exponent + do('log10', stripped_factor)

    def distribute_exponent(self):
        """Distribute the exponent ``p`` of this tensor network (i.e.
        corresponding to ``tn * 10**p``) equally among all tensors.
        """
        # multiply each tensor by the nth root of 10**exponent
        x = 10**(self.exponent / self.num_tensors)
        self.multiply_each_(x)

        # reset the exponent to zero
        self.exponent = 0.0

    def equalize_norms(self, value=None, inplace=False):
        """Make the Frobenius norm of every tensor in this TN equal without
        changing the overall value if ``value=None``, or set the norm of every
        tensor to ``value`` by scalar multiplication only.

        Parameters
        ----------
        value : None or float, optional
            Set the norm of each tensor to this value specifically. If supplied
            the change in overall scaling will be accumulated in
            ``tn.exponent`` in the form of a base 10 power.
        inplace : bool, optional
            Whether to perform the norm equalization inplace or not.

        Returns
        -------
        TensorNetwork
        """
        tn = self if inplace else self.copy()

        for tid in tn.tensor_map:
            tn.strip_exponent(tid, value=value)

        if value is None:
            tn.distribute_exponent()

        return tn

    equalize_norms_ = functools.partialmethod(equalize_norms, inplace=True)

    def balance_bonds(self, inplace=False):
        """Apply :func:`~quimb.tensor.tensor_contract.tensor_balance_bond` to
        all bonds in this tensor network.

        Parameters
        ----------
        inplace : bool, optional
            Whether to perform the bond balancing inplace or not.

        Returns
        -------
        TensorNetwork
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
                seen[frozenset(tids)].append(ix)

        for tidset, ixs in seen.items():
            if len(ixs) > 1:
                for tid in sorted(tidset):
                    self.tensor_map[tid].fuse_({ixs[0]: ixs})

        return tn

    fuse_multibonds_ = functools.partialmethod(fuse_multibonds, inplace=True)

    def expand_bond_dimension(
        self,
        new_bond_dim,
        rand_strength=0.0,
        inds_to_expand=None,
        inplace=False,
    ):
        """Increase the dimension of bonds to at least ``new_bond_dim``.
        """
        tn = self if inplace else self.copy()

        if inds_to_expand is None:
            # find all 'bonds' - indices connecting two or more tensors
            inds_to_expand = set()
            for ind, tids in tn.ind_map.items():
                if len(tids) >= 2:
                    inds_to_expand.add(ind)
        else:
            inds_to_expand = set(inds_to_expand)

        for t in tn:
            # perform the array expansions
            pads = [
                (0, 0) if ind not in inds_to_expand else
                (0, max(new_bond_dim - d, 0))
                for d, ind in zip(t.shape, t.inds)
            ]

            if rand_strength > 0:
                edata = do('pad', t.data, pads, mode=rand_padder,
                           rand_strength=rand_strength)
            else:
                edata = do('pad', t.data, pads, mode='constant')

            t.modify(data=edata)

        return tn

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

    def rank_simplify(
        self,
        output_inds=None,
        equalize_norms=False,
        cache=None,
        max_combinations=500,
        inplace=False,
    ):
        """Simplify this tensor network by performing contractions that don't
        increase the rank of any tensors.

        Parameters
        ----------
        output_inds : sequence of str, optional
            Explicitly set which indices of the tensor network are output
            indices and thus should not be modified.
        equalize_norms : bool or float
            Actively renormalize the tensors during the simplification process.
            Useful for very large TNs. The scaling factor will be stored as an
            exponent in ``tn.exponent``.
        cache : None or set
            Persistent cache used to mark already checked tensors.
        inplace : bool, optional
            Whether to perform the rand reduction inplace.

        Returns
        -------
        TensorNetwork

        See Also
        --------
        full_simplify, column_reduce, diagonal_reduce
        """
        tn = self if inplace else self.copy()

        if output_inds is None:
            output_inds = tn._outer_inds

        # pairs of tensors we have already checked
        if cache is None:
            cache = set()

        # first parse all tensors
        scalars = []
        count = collections.Counter()
        for tid, t in tuple(tn.tensor_map.items()):

            # remove floating scalar tensors -->
            #     these have no indices so won't be caught otherwise
            if t.ndim == 0:
                tn._pop_tensor(tid)
                scalars.append(t.data)
                continue

            # ... and remove any redundant repeated indices on the same tensor
            t.collapse_repeated_()

            # ... also build the index counter at the same time
            count.update(t.inds)

        # this ensures the output indices are not removed (+1 each)
        count.update(output_inds)

        # special case, everything connected by one index
        trivial = len(count) == 1

        # sorted list of unique indices to check -> start with lowly connected
        def rank_weight(ind):
            return (tn.ind_size(ind), -sum(tn.tensor_map[tid].ndim
                                           for tid in tn.ind_map[ind]))

        queue = oset(sorted(count, key=rank_weight))

        # number of tensors for which there will be more pairwise combinations
        # than max_combinations
        combi_cutoff = int(0.5 * ((8 * max_combinations + 1)**0.5 + 1))

        while queue:
            # get next index
            ind = queue.popright()

            # the tensors it connects
            try:
                tids = tn.ind_map[ind]
            except KeyError:
                # index already contracted alongside another
                continue

            # index only appears on one tensor and not in output -> can sum
            if count[ind] == 1:
                tid, = tids
                t = tn.tensor_map[tid]
                t.sum_reduce_(ind)

                # check if we have created a scalar
                if t.ndim == 0:
                    tn._pop_tensor(tid)
                    scalars.append(t.data)

                continue

            # otherwise check pairwise contractions
            cands = []
            combos_checked = 0

            if len(tids) > combi_cutoff:
                # sort size of the tensors so that when we are limited by
                #     max_combinations we check likely ones first
                tids = sorted(tids, key=lambda tid: tn.tensor_map[tid].ndim)

            for tid_a, tid_b in itertools.combinations(tids, 2):

                ta = tn.tensor_map[tid_a]
                tb = tn.tensor_map[tid_b]

                cache_key = ('rs', tid_a, tid_b, id(ta.data), id(tb.data))
                if cache_key in cache:
                    continue

                combos_checked += 1

                # work out the output indices of candidate contraction
                involved = frequencies(itertools.chain(ta.inds, tb.inds))
                out_ab = []
                deincr = []
                for oix, c in involved.items():
                    if c != count[oix]:
                        out_ab.append(oix)
                        if c == 2:
                            deincr.append(oix)
                    # else this the last occurence of index oix -> remove it

                # check if candidate contraction will reduce rank
                new_ndim = len(out_ab)
                old_ndim = max(ta.ndim, tb.ndim)

                if new_ndim <= old_ndim:
                    res = (new_ndim - old_ndim, tid_a, tid_b, out_ab, deincr)
                    cands.append(res)
                else:
                    cache.add(cache_key)

                if cands and (trivial or combos_checked > max_combinations):
                    # can do contractions in any order
                    # ... or hyperindex is very large, stop checking
                    break

            if not cands:
                # none of the parwise contractions reduce rank
                continue

            _, tid_a, tid_b, out_ab, deincr = min(cands)
            ta = tn._pop_tensor(tid_a)
            tb = tn._pop_tensor(tid_b)
            tab = ta.contract(tb, output_inds=out_ab)

            for ix in deincr:
                count[ix] -= 1

            if not out_ab:
                # handle scalars produced at the end
                scalars.append(tab)
                continue

            tn |= tab

            if equalize_norms:
                tn.strip_exponent(tab, equalize_norms)

            for ix in out_ab:
                # now we need to check outputs indices again
                queue.add(ix)

        if scalars:
            if equalize_norms:
                signs = []
                for s in scalars:
                    signs.append(do("sign", s))
                    tn.exponent += do("log10", do('abs', s))
                scalars = signs

            if tn.num_tensors:
                tn *= prod(scalars)
            else:
                # no tensors left! re-add one with all the scalars
                tn |= Tensor(prod(scalars))

        return tn

    rank_simplify_ = functools.partialmethod(rank_simplify, inplace=True)

    def diagonal_reduce(
        self,
        output_inds=None,
        atol=1e-12,
        cache=None,
        inplace=False,
    ):
        """Find tensors with diagonal structure and collapse those axes. This
        will create a tensor 'hyper' network with indices repeated 2+ times, as
        such, output indices should be explicitly supplied when contracting, as
        they can no longer be automatically inferred. For example:

            >>> tn_diag = tn.diagonal_reduce()
            >>> tn_diag.contract(all, output_inds=[])

        Parameters
        ----------
        output_inds : sequence of str, optional
            Which indices to explicitly consider as outer legs of the tensor
            network and thus not replace. If not given, these will be taken as
            all the indices that appear once.
        atol : float, optional
            When identifying diagonal tensors, the absolute tolerance with
            which to compare to zero with.
        cache : None or set
            Persistent cache used to mark already checked tensors.
        inplace, bool, optional
            Whether to perform the diagonal reduction inplace.

        Returns
        -------
        TensorNetwork

        See Also
        --------
        full_simplify, rank_simplify, antidiag_gauge, column_reduce
        """
        tn = self if inplace else self.copy()

        if cache is None:
            cache = set()

        if output_inds is None:
            output_inds = set(tn._outer_inds)

        queue = list(tn.tensor_map)
        while queue:
            tid = queue.pop()
            t = tn.tensor_map[tid]

            cache_key = ('dr', tid, id(t.data))
            if cache_key in cache:
                continue

            ij = find_diag_axes(t.data, atol=atol)

            # no diagonals
            if ij is None:
                cache.add(cache_key)
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

            # update wherever else the changed index appears (e.g. 'c' above)
            tn.reindex_(ixmap)

            # take the multidimensional diagonal of the tensor
            #     (which now has a repeated index)
            t.collapse_repeated_()

            # tensor might still have other diagonal indices
            queue.append(tid)

        return tn

    diagonal_reduce_ = functools.partialmethod(diagonal_reduce, inplace=True)

    def antidiag_gauge(
        self,
        output_inds=None,
        atol=1e-12,
        cache=None,
        inplace=False,
    ):
        """Flip the order of any bonds connected to antidiagonal tensors.
        Whilst this is just a gauge fixing (with the gauge being the flipped
        identity) it then allows ``diagonal_reduce`` to then simplify those
        indices.

        Parameters
        ----------
        output_inds : sequence of str, optional
            Which indices to explicitly consider as outer legs of the tensor
            network and thus not flip. If not given, these will be taken as
            all the indices that appear once.
        atol : float, optional
            When identifying antidiagonal tensors, the absolute tolerance with
            which to compare to zero with.
        cache : None or set
            Persistent cache used to mark already checked tensors.
        inplace, bool, optional
            Whether to perform the antidiagonal gauging inplace.

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

        if cache is None:
            cache = set()

        done = set()

        queue = list(tn.tensor_map)
        while queue:
            tid = queue.pop()
            t = tn.tensor_map[tid]

            cache_key = ('ag', tid, id(t.data))
            if cache_key in cache:
                continue

            ij = find_antidiag_axes(t.data, atol=atol)

            # tensor not anti-diagonal
            if ij is None:
                cache.add(cache_key)
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

    def column_reduce(
        self,
        output_inds=None,
        atol=1e-12,
        cache=None,
        inplace=False,
    ):
        """Find bonds on this tensor network which have tensors where all but
        one column (of the respective index) is non-zero, allowing the
        'cutting' of that bond.

        Parameters
        ----------
        output_inds : sequence of str, optional
            Which indices to explicitly consider as outer legs of the tensor
            network and thus not slice. If not given, these will be taken as
            all the indices that appear once.
        atol : float, optional
            When identifying singlet column tensors, the absolute tolerance
            with which to compare to zero with.
        cache : None or set
            Persistent cache used to mark already checked tensors.
        inplace, bool, optional
            Whether to perform the column reductions inplace.

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

        if cache is None:
            cache = set()

        queue = list(tn.tensor_map)
        while queue:
            tid = queue.pop()
            t = tn.tensor_map[tid]

            cache_key = ('cr', tid, id(t.data))
            if cache_key in cache:
                continue

            ax_i = find_columns(t.data, atol=atol)

            # no singlet columns
            if ax_i is None:
                cache.add(cache_key)
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

    def split_simplify(
        self,
        atol=1e-12,
        equalize_norms=False,
        cache=None,
        inplace=False,
    ):
        """Find tensors which have low rank SVD decompositions across any
        combination of bonds and perform them.

        Parameters
        ----------
        atol : float, optional
            Cutoff used when attempting low rank decompositions.
        equalize_norms : bool or float
            Actively renormalize the tensors during the simplification process.
            Useful for very large TNs. The scaling factor will be stored as an
            exponent in ``tn.exponent``.
        cache : None or set
            Persistent cache used to mark already checked tensors.
        inplace, bool, optional
            Whether to perform the split simplification inplace.
        """
        tn = self if inplace else self.copy()

        # we don't want to repeatedly check the split decompositions of the
        #     same tensor as we cycle through simplification methods
        if cache is None:
            cache = set()

        for tid, t in tuple(tn.tensor_map.items()):

            # id's are reused when objects go out of scope -> use tid as well
            cache_key = ('sp', tid, id(t.data))
            if cache_key in cache:
                continue

            found = False
            for lix, rix in gen_bipartitions(t.inds):
                tl, tr = t.split(lix, right_inds=rix,
                                 get='tensors', cutoff=atol)
                new_size = max(tl.size, tr.size)
                if new_size < t.size:
                    found = True
                    break

            if found:
                tn._pop_tensor(tid)
                tn |= tl
                tn |= tr

                if equalize_norms:
                    tn.strip_exponent(tl, equalize_norms)
                    tn.strip_exponent(tr, equalize_norms)

            else:
                cache.add(cache_key)

        return tn

    split_simplify_ = functools.partialmethod(split_simplify, inplace=True)

    def gen_loops(self, max_loop_length=None):
        """Generate sequences of tids that represent loops in the TN.

        Parameters
        ----------
        max_loop_length : None or int
            Set the maximum number of tensors that can appear in a loop. If
            ``None``, wait until any loop is found and set that as the
            maximum length.

        Yields
        ------
        tuple[int]
        """
        from cotengra.core import get_hypergraph
        inputs = {tid: t.inds for tid, t in self.tensor_map.items()}
        hg = get_hypergraph(inputs, accel='auto')
        return hg.compute_loops(max_loop_length)

    def tids_are_connected(self, tids):
        """Check whether nodes ``tids`` are connected.

        Parameters
        ----------
        tids : sequence of int
            Nodes to check.

        Returns
        -------
        bool
        """
        enum = range(len(tids))
        groups = dict(zip(enum, enum))
        regions = [
            (oset([tid]), self._get_neighbor_tids(tid))
            for tid in tids
        ]
        for i, j in itertools.combinations(enum, 2):
            mi = groups.get(i, i)
            mj = groups.get(j, j)

            if regions[mi][0] & regions[mj][1]:
                groups[mj] = mi
                regions[mi][0].update(regions[mj][0])
                regions[mi][1].update(regions[mj][1])

        return len(set(groups.values())) == 1

    def compute_shortest_distances(self, tids=None, exclude_inds=()):
        """Compute the minimum graph distances between all or some nodes
        ``tids``.
        """
        if tids is None:
            tids = self.tensor_map
        else:
            tids = set(tids)

        visitors = collections.defaultdict(frozenset)
        for tid in tids:
            # start with target tids having 'visited' themselves only
            visitors[tid] = frozenset([tid])

        distances = {}
        N = math.comb(len(tids), 2)

        for d in itertools.count(1):
            any_change = False
            old_visitors = visitors.copy()

            # only need to iterate over touched region
            for tid in tuple(visitors):
                # at each step, each node sends its current visitors to all
                # neighboring nodes
                current_visitors = old_visitors[tid]
                for next_tid in self._get_neighbor_tids(tid, exclude_inds):
                    visitors[next_tid] |= current_visitors

            for tid in tuple(visitors):
                # check for new visitors -> those with shortest path d
                for diff_tid in visitors[tid] - old_visitors[tid]:
                    any_change = True
                    if tid in tids:
                        distances[tuple(sorted((tid, diff_tid)))] = d

            if (len(distances) == N) or (not any_change):
                # all pair combinations have been computed, or everything
                # converged, presumably due to disconnected subgraphs
                break

        return distances

    def compute_hierarchical_linkage(
        self,
        tids=None,
        method='weighted',
        optimal_ordering=True,
        exclude_inds=(),
    ):
        from scipy.cluster import hierarchy

        if tids is None:
            tids = self.tensor_map
        distances = self.compute_shortest_distances(tids, exclude_inds)

        dinf = 10 * max(distances.values())
        y = [
            distances.get(tuple(sorted((i, j))), dinf)
            for i, j in itertools.combinations(tids, 2)
        ]

        return hierarchy.linkage(
            y, method=method, optimal_ordering=optimal_ordering
        )

    def compute_hierarchical_ssa_path(
        self,
        tids=None,
        method='weighted',
        optimal_ordering=True,
        exclude_inds=(),
        are_sorted=False,
    ):

        L = self.compute_hierarchical_linkage(
            tids, method=method, exclude_inds=exclude_inds,
            optimal_ordering=optimal_ordering)

        sorted_ssa_path = ((int(x[0]), int(x[1])) for x in L)
        if are_sorted:
            return tuple(sorted_ssa_path)

        if tids is None:
            tids = self.tensor_map
        given_idx = {tid: i for i, tid in enumerate(tids)}
        sorted_to_given_idx = {
            i: given_idx[tid] for i, tid in enumerate(sorted(tids))
        }
        return tuple(
            (sorted_to_given_idx.get(x, x), sorted_to_given_idx.get(y, y))
            for x, y in sorted_ssa_path
        )

    def compute_hierarchical_ordering(
        self,
        tids=None,
        method='weighted',
        optimal_ordering=True,
        exclude_inds=(),
    ):
        from scipy.cluster import hierarchy
        if tids is None:
            tids = list(self.tensor_map)
        L = self.compute_hierarchical_linkage(
            tids, method=method, exclude_inds=exclude_inds,
            optimal_ordering=optimal_ordering)
        node2tid = {i: tid for i, tid in enumerate(sorted(tids))}
        return tuple(map(node2tid.__getitem__, hierarchy.leaves_list(L)))

    def pair_simplify(
        self,
        cutoff=1e-12,
        output_inds=None,
        max_inds=10,
        cache=None,
        equalize_norms=False,
        max_combinations=500,
        inplace=False,
        **split_opts,
    ):
        tn = self if inplace else self.copy()

        if output_inds is None:
            output_inds = tn._outer_inds

        queue = list(tn.ind_map)

        def gen_pairs():
            # number of tensors for which there will be more pairwise
            # combinations than max_combinations
            combi_cutoff = int(0.5 * ((8 * max_combinations + 1)**0.5 + 1))

            while queue:
                ind = queue.pop()
                try:
                    tids = tn.ind_map[ind]
                except KeyError:
                    continue

                if len(tids) > combi_cutoff:
                    # sort size of the tensors so that when we are limited by
                    #     max_combinations we check likely ones first
                    tids = sorted(
                        tids, key=lambda tid: tn.tensor_map[tid].ndim)

                for _, (tid1, tid2) in zip(
                    range(max_combinations),
                    itertools.combinations(tids, 2),
                ):
                    if (tid1 in tn.tensor_map) and (tid2 in tn.tensor_map):
                        yield tid1, tid2

        for pair in gen_pairs():

            if cache is not None:
                key = ('pc', frozenset((tid, id(tn.tensor_map[tid].data))
                                       for tid in pair))
                if key in cache:
                    continue

            t1, t2 = tn._tids_get(*pair)
            inds = self.compute_contracted_inds(*pair, output_inds=output_inds)

            if len(inds) > max_inds:
                # don't check exponentially many bipartitions
                continue

            t12 = tensor_contract(t1, t2, output_inds=inds,
                                  preserve_tensor=True)
            current_size = t1.size + t2.size

            cands = []
            for lix, rix in gen_bipartitions(inds):
                tl, tr = t12.split(left_inds=lix, right_inds=rix,
                                   get='tensors', cutoff=cutoff, **split_opts)
                new_size = (tl.size + tr.size)
                if new_size < current_size:
                    cands.append((new_size / current_size, pair, tl, tr))

            if not cands:
                # no decompositions decrease the size
                if cache is not None:
                    cache.add(key)
                continue

            # perform the decomposition that minimizes the new size
            _, pair, tl, tr = min(cands, key=lambda x: x[0])
            for tid in tuple(pair):
                tn._pop_tensor(tid)
            tn |= tl
            tn |= tr

            tensor_fuse_squeeze(tl, tr)
            if equalize_norms:
                tn.strip_exponent(tl, equalize_norms)
                tn.strip_exponent(tr, equalize_norms)

            queue.extend(tl.inds)
            queue.extend(tr.inds)

        return tn

    pair_simplify_ = functools.partialmethod(pair_simplify, inplace=True)

    def loop_simplify(
        self,
        output_inds=None,
        max_loop_length=None,
        max_inds=10,
        cutoff=1e-12,
        loops=None,
        cache=None,
        equalize_norms=False,
        inplace=False,
        **split_opts
    ):
        """Try and simplify this tensor network by identifying loops and
        checking for low-rank decompositions across groupings of the loops
        outer indices.

        Parameters
        ----------
        max_loop_length : None or int, optional
            Largest length of loop to search for, if not set, the size will be
            set to the length of the first (and shortest) loop found.
        cutoff : float, optional
            Cutoff to use for the operator decomposition.
        loops : None, sequence or callable
            Loops to check, or a function that generates them.
        cache : set, optional
            For performance reasons can supply a cache for already checked
            loops.
        inplace : bool, optional
            Whether to replace the loops inplace.
        split_opts
            Supplied to :func:`~quimb.tensor.tensor_core.tensor_split`.

        Returns
        -------
        TensorNetwork
        """
        tn = self if inplace else self.copy()

        if output_inds is None:
            output_inds = tn._outer_inds

        if loops is None:
            loops = tuple(tn.gen_loops(max_loop_length))
        elif callable(loops):
            loops = loops(tn, max_loop_length)

        for loop in loops:
            if any(tid not in tn.tensor_map for tid in loop):
                # some tensors have been compressed away already
                continue

            if cache is not None:
                key = ('lp', frozenset((tid, id(tn.tensor_map[tid].data))
                                       for tid in loop))
                if key in cache:
                    continue

            oix = tn.compute_contracted_inds(*loop, output_inds=output_inds)
            if len(oix) > max_inds:
                continue

            ts = tuple(tn._tids_get(*loop))
            current_size = sum(t.size for t in ts)
            tloop = tensor_contract(*ts, output_inds=oix)

            cands = []
            for left_inds, right_inds in gen_bipartitions(oix):
                if not (
                    tn.tids_are_connected(self._get_tids_from_inds(left_inds))
                    and
                    tn.tids_are_connected(self._get_tids_from_inds(right_inds))
                ):
                    continue

                tl, tr = tensor_split(
                    tloop, left_inds=left_inds, right_inds=right_inds,
                    get='tensors', cutoff=cutoff, **split_opts
                )

                new_size = (tl.size + tr.size)
                if new_size < current_size:
                    cands.append((new_size / current_size, loop, tl, tr))

            if not cands:
                # no decompositions decrease the size
                if cache is not None:
                    cache.add(key)
                continue

            # perform the decomposition that minimizes the new size
            _, loop, tl, tr = min(cands, key=lambda x: x[0])
            for tid in loop:
                tn._pop_tensor(tid)
            tn |= tl
            tn |= tr

            tensor_fuse_squeeze(tl, tr)
            if equalize_norms:
                tn.strip_exponent(tl, equalize_norms)
                tn.strip_exponent(tr, equalize_norms)

        return tn

    loop_simplify_ = functools.partialmethod(loop_simplify, inplace=True)

    def full_simplify(
        self,
        seq='ADCR',
        output_inds=None,
        atol=1e-12,
        equalize_norms=False,
        cache=None,
        inplace=False,
        progbar=False,
        rank_simplify_opts=None,
        loop_simplify_opts=None,
        custom_methods=(),
    ):
        """Perform a series of tensor network 'simplifications' in a loop until
        there is no more reduction in the number of tensors or indices. Note
        that apart from rank-reduction, the simplification methods make use of
        the non-zero structure of the tensors, and thus changes to this will
        potentially produce different simplifications.

        Parameters
        ----------
        seq : str, optional
            Which simplifications and which order to perform them in.

                * ``'A'`` : stands for ``antidiag_gauge``
                * ``'D'`` : stands for ``diagonal_reduce``
                * ``'C'`` : stands for ``column_reduce``
                * ``'R'`` : stands for ``rank_simplify``
                * ``'S'`` : stands for ``split_simplify``
                * ``'L'`` : stands for ``loop_simplify``

            If you want to keep the tensor network 'simple', i.e. with no
            hyperedges, then don't use ``'D'`` (moreover ``'A'`` is redundant).
        output_inds : sequence of str, optional
            Explicitly set which indices of the tensor network are output
            indices and thus should not be modified. If not specified the
            tensor network is assumed to be a 'standard' one where indices that
            only appear once are the output indices.
        atol : float, optional
            The absolute tolerance when indentifying zero entries of tensors
            and performing low-rank decompositions.
        equalize_norms : bool or float
            Actively renormalize the tensors during the simplification process.
            Useful for very large TNs. If `True`, the norms, in the formed of
            stripped exponents, will be redistributed at the end. If an actual
            number, the final tensors will all have this norm, and the scaling
            factor will be stored as a base-10 exponent in ``tn.exponent``.
        cache : None or set
            A persistent cache for each simplification process to mark
            already processed tensors.
        progbar : bool, optional
            Show a live progress bar of the simplification process.
        inplace : bool, optional
            Whether to perform the simplification inplace.

        Returns
        -------
        TensorNetwork

        See Also
        --------
        diagonal_reduce, rank_simplify, antidiag_gauge, column_reduce,
        split_simplify, loop_simplify
        """
        tn = self if inplace else self.copy()
        tn.squeeze_()

        rank_simplify_opts = ensure_dict(rank_simplify_opts)
        loop_simplify_opts = ensure_dict(loop_simplify_opts)

        # all the methods
        if output_inds is None:
            output_inds = self.outer_inds()

        if cache is None:
            cache = set()

        # for the index trick reductions, faster to supply set
        ix_o = set(output_inds)

        # keep simplifying until the number of tensors and indices equalizes
        old_nt, old_ni = -1, -1
        nt, ni = tn.num_tensors, tn.num_indices

        if progbar:
            import tqdm
            pbar = tqdm.tqdm()
            pbar.set_description(f'{nt}, {ni}')

        while (nt, ni) != (old_nt, old_ni):
            for meth in seq:

                if progbar:
                    pbar.update()
                    pbar.set_description(
                        f'{meth} {tn.num_tensors}, {tn.num_indices}')

                if meth in custom_methods:
                    custom_methods[meth](
                        tn, output_inds=output_inds, atol=atol, cache=cache)
                elif meth == 'D':
                    tn.diagonal_reduce_(output_inds=ix_o, atol=atol,
                                        cache=cache)
                elif meth == 'R':
                    tn.rank_simplify_(output_inds=ix_o, cache=cache,
                                      equalize_norms=equalize_norms,
                                      **rank_simplify_opts)
                elif meth == 'A':
                    tn.antidiag_gauge_(output_inds=ix_o, atol=atol,
                                       cache=cache)
                elif meth == 'C':
                    tn.column_reduce_(output_inds=ix_o, atol=atol, cache=cache)
                elif meth == 'S':
                    tn.split_simplify_(atol=atol, cache=cache,
                                       equalize_norms=equalize_norms)
                elif meth == 'L':
                    tn.loop_simplify_(output_inds=ix_o, cutoff=atol,
                                      cache=cache,
                                      equalize_norms=equalize_norms,
                                      **loop_simplify_opts)
                elif meth == 'P':
                    tn.pair_simplify_(output_inds=ix_o, cutoff=atol,
                                      cache=cache,
                                      equalize_norms=equalize_norms,
                                      **loop_simplify_opts)
                else:
                    raise ValueError(f"'{meth}' is not a valid simplify type.")

            old_nt, old_ni = nt, ni
            nt, ni = tn.num_tensors, tn.num_indices

        if equalize_norms:
            if equalize_norms is True:
                # this also redistributes the collected exponents
                tn.equalize_norms_()
            else:
                tn.equalize_norms_(value=equalize_norms)

        if progbar:
            pbar.close()

        return tn

    full_simplify_ = functools.partialmethod(full_simplify, inplace=True)

    def hyperinds_resolve(self, mode='dense', sorter=None, inplace=False):
        """Convert this into a regular tensor network, where all indices
        appear at most twice, by inserting COPY tensor or tensor networks
        for each hyper index.

        Parameters
        ----------
        mode : {'dense', 'mps', 'tree'}, optional
            What type of COPY tensor(s) to insert.
        sorter : None or callable, optional
            If given, a function to sort the indices that a single hyperindex
            will be turned into. Th function is called like
            ``tids.sort(key=sorter)``.
        inplace : bool, optional
            Whether to insert the COPY tensors inplace.

        Returns
        -------
        TensorNetwork
        """
        check_opt('mode', mode, ('dense', 'mps', 'tree'))
        tn = self if inplace else self.copy()

        if sorter == 'centrality':
            from cotengra.cotengra import nodes_to_centrality
            cents = nodes_to_centrality(
                {tid: t.inds for tid, t in tn.tensor_map.items()}
            )

            def sorter(tid):
                return cents[tid]

        if sorter == 'clustering':
            tn_orig = tn.copy()

        ssa_path = None

        copy_tensors = []
        for ix, tids in tuple(tn.ind_map.items()):
            if len(tids) > 2:
                d = tn.ind_size(ix)

                tids = list(tids)
                if sorter == 'clustering':

                    if mode == 'tree':
                        tids.sort()
                        ssa_path = tn_orig.compute_hierarchical_ssa_path(
                            tids, optimal_ordering=False, exclude_inds=(ix,),
                            are_sorted=True)
                    else:
                        tids = tn_orig.compute_hierarchical_ordering(
                            tids, optimal_ordering=True, exclude_inds=(ix,))

                elif sorter is not None:
                    tids.sort(key=sorter)

                # reindex tensors surrounding ind
                copy_inds = []
                for tid in tids:
                    new_ix = rand_uuid()
                    t = tn.tensor_map[tid]
                    t.reindex_({ix: new_ix})
                    copy_inds.append(new_ix)

                # inject new tensor(s) to connect dangling inds
                if mode == 'dense':
                    copy_tensors.append(
                        COPY_tensor(d=d, inds=copy_inds, dtype=t.dtype))
                elif mode == 'mps':
                    copy_tensors.extend(
                        COPY_mps_tensors(d=d, inds=copy_inds, dtype=t.dtype))
                elif mode == 'tree':
                    copy_tensors.extend(
                        COPY_tree_tensors(d=d, inds=copy_inds, dtype=t.dtype,
                                          ssa_path=ssa_path))

        tn.add(copy_tensors)
        return tn

    hyperinds_resolve_ = functools.partialmethod(
        hyperinds_resolve, inplace=True)

    def compress_simplify(
        self,
        output_inds=None,
        atol=1e-6,
        simplify_sequence_a='ADCRS',
        simplify_sequence_b='RPL',
        hyperind_resolve_mode='tree',
        hyperind_resolve_sort='clustering',
        final_resolve=False,
        max_simplification_iterations=100,
        converged_tol=0.01,
        equalize_norms=True,
        progbar=False,
        inplace=False,
    ):
        tn = self if inplace else self.copy()

        if output_inds is None:
            output_inds = self.outer_inds()

        simplify_opts = {
            'atol': atol,
            'equalize_norms': equalize_norms,
            'progbar': progbar,
            'output_inds': output_inds,
            'cache': set(),
        }

        # order of tensors when converting hyperinds
        if callable(hyperind_resolve_sort) or (hyperind_resolve_sort is None):
            sorter = hyperind_resolve_sort
        elif hyperind_resolve_sort == 'centrality':
            from cotengra.cotengra import nodes_to_centrality

            def sorter(tid):
                return cents[tid]
        elif hyperind_resolve_sort == 'random':
            import random

            def sorter(tid):
                return random.random()

        else:
            sorter = hyperind_resolve_sort

        tn.full_simplify_(simplify_sequence_a, **simplify_opts)
        for i in range(max_simplification_iterations):
            nv, ne = tn.num_tensors, tn.num_indices
            if hyperind_resolve_sort == 'centrality':
                # recompute centralities
                cents = nodes_to_centrality(
                    {tid: t.inds for tid, t in tn.tensor_map.items()}
                )
            tn.hyperinds_resolve_(hyperind_resolve_mode, sorter=sorter)
            tn.full_simplify_(simplify_sequence_b, **simplify_opts)
            tn.full_simplify_(simplify_sequence_a, **simplify_opts)
            if (
                (tn.num_tensors == 1) or
                (tn.num_tensors > (1 - converged_tol) * nv and
                 tn.num_indices > (1 - converged_tol) * ne)
            ):
                break

        if final_resolve:
            if hyperind_resolve_sort == 'centrality':
                # recompute centralities
                cents = nodes_to_centrality(
                    {tid: t.inds for tid, t in tn.tensor_map.items()}
                )
            tn.hyperinds_resolve_(hyperind_resolve_mode, sorter=sorter)
            tn.full_simplify_(simplify_sequence_b, **simplify_opts)

        return tn

    compress_simplify_ = functools.partialmethod(
        compress_simplify, inplace=True)

    def max_bond(self):
        """Return the size of the largest bond in this network.
        """
        return max(t.max_dim() for t in self)

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
        return get_common_dtype(*self.arrays)

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
        for tid, t in self.__dict__['tensor_map'].items():
            t.add_owner(self, tid=tid)

    def __str__(self):
        return "{}([{}{}{}])".format(
            self.__class__.__name__,
            os.linesep,
            "".join(["    " + repr(t) + "," + os.linesep
                     for t in self.tensors[:-1]]),
            "    " + repr(self.tensors[-1]) + "," + os.linesep)

    def __repr__(self):
        rep = f"<{self.__class__.__name__}("
        rep += f"tensors={self.num_tensors}"
        rep += f", indices={self.num_indices}"

        return rep + ")>"

    draw = draw_tn
    graph = draw_tn


TNLO_HANDLED_FUNCTIONS = {}


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
    optimize : str, optional
        The path optimizer to use for the 'matrix-vector' contraction.
    backend : str, optional
        The array backend to use for the 'matrix-vector' contraction.
    is_conj : bool, optional
        Whether this object should represent the *adjoint* operator.

    See Also
    --------
    TNLinearOperator1D
    """

    def __init__(self, tns, left_inds, right_inds, ldims=None, rdims=None,
                 optimize=None, backend=None, is_conj=False):
        if backend is None:
            self.backend = get_tensor_linop_backend()
        else:
            self.backend = backend
        self.optimize = optimize

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
        self.tags = oset_union(t.tags for t in self._tensors)

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
                *self._tensors, iT, output_inds=self.left_inds,
                optimize=self.optimize, **self._kws)

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
                *self._tensors, iT, output_inds=o_ix,
                optimize=self.optimize, **self._kws)

        fn = self._contractors[key]
        out_data = fn(*self._ins, in_data, backend=self.backend)

        if self.is_conj:
            out_data = conj(out_data)

        return reshape(out_data, (-1, d))

    def trace(self):
        if 'trace' not in self._contractors:
            tn = TensorNetwork(self._tensors)
            self._contractors['trace'] = tn.trace(
                self.left_inds, self.right_inds, optimize=self.optimize)
        return self._contractors['trace']

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

        return TNLinearOperator(self._tensors, *inds, *dims, is_conj=is_conj,
                                optimize=self.optimize, backend=self.backend)

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
        contract_opts.setdefault('optimize', self.optimize)

        if self.is_conj:
            ts = (t.conj() for t in self._tensors)
        else:
            ts = self._tensors

        if not inds_seq:
            inds_seq = self.left_inds, self.right_inds

        return tensor_contract(*ts, **contract_opts).to_dense(*inds_seq)

    @functools.wraps(tensor_split)
    def split(self, **split_opts):
        return tensor_split(self, left_inds=self.left_inds,
                            right_inds=self.right_inds, **split_opts)

    @property
    def A(self):
        return self.to_dense()

    def astype(self, dtype):
        """Convert this ``TNLinearOperator`` to type ``dtype``.
        """
        return TNLinearOperator(
            (t.astype(dtype) for t in self._tensors),
            left_inds=self.left_inds, right_inds=self.right_inds,
            ldims=self.ldims, rdims=self.rdims,
            optimize=self.optimize, backend=self.backend,
        )

    def __array_function__(self, func, types, args, kwargs):
        if (
            (func not in TNLO_HANDLED_FUNCTIONS) or
            (not all(issubclass(t, self.__class__) for t in types))
        ):
            return NotImplemented
        return TNLO_HANDLED_FUNCTIONS[func](*args, **kwargs)


def tnlo_implements(np_function):
    """Register an __array_function__ implementation for TNLinearOperator
    objects.
    """
    def decorator(func):
        TNLO_HANDLED_FUNCTIONS[np_function] = func
        return func

    return decorator


@tnlo_implements(np.trace)
def _tnlo_trace(x):
    return x.trace()


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

    See Also
    --------
    PTensor
    """

    def __init__(self, fn, params, inds=(), tags=None, left_inds=None):
        super().__init__(
            PArray(fn, params), inds=inds, tags=tags, left_inds=left_inds)

    @classmethod
    def from_parray(cls, parray, inds=(), tags=None, left_inds=None):
        obj = cls.__new__(cls)
        super(PTensor, obj).__init__(
            parray, inds=inds, tags=tags, left_inds=left_inds)
        return obj

    def copy(self):
        """Copy this parametrized tensor.
        """
        return PTensor.from_parray(self._parray.copy(), inds=self.inds,
                                   tags=self.tags, left_inds=self.left_inds)

    @property
    def _data(self):
        """Make ``_data`` read-only and handle conjugation lazily.
        """
        return self._parray.data

    @_data.setter
    def _data(self, x):
        if not isinstance(x, PArray):
            raise TypeError(
                "You can only directly update the data of a ``PTensor`` with "
                "another ``PArray``. You can chain another function with the "
                "``.modify(apply=fn)`` method. Alternatively you can convert "
                "this ``PTensor to a normal ``Tensor`` with "
                "``t.unparametrize()``")
        self._parray = x

    @property
    def data(self):
        return self._data

    @property
    def fn(self):
        return self._parray.fn

    @fn.setter
    def fn(self, x):
        self._parray.fn = x

    @property
    def params(self):
        return self._parray.params

    @params.setter
    def params(self, x):
        self._parray.params = x

    @property
    def shape(self):
        return self._parray.shape

    def _apply_function(self, fn):
        """Apply ``fn`` to the data array of this ``PTensor`` (lazily), by
        composing it with the current parametrized array function.
        """
        self._parray.add_function(fn)

    def conj(self, inplace=False):
        """Conjugate this parametrized tensor - done lazily whenever the
        ``.data`` attribute is accessed.
        """
        t = self if inplace else self.copy()
        t._apply_function(conj)
        return t

    conj_ = functools.partialmethod(conj, inplace=True)

    def unparametrize(self):
        """Turn this PTensor into a normal Tensor.
        """
        return Tensor(self)


class IsoTensor(Tensor):
    """A ``Tensor`` subclass which keeps its ``left_inds`` by default even
    when its data is changed.
    """

    __slots__ = ('_data', '_inds', '_tags', '_left_inds', '_owners')

    def modify(self, **kwargs):
        kwargs.setdefault("left_inds", self.left_inds)
        super().modify(**kwargs)

    def fuse(self, *args, inplace=False, **kwargs):
        t = self if inplace else self.copy()
        t.left_inds = None
        return Tensor.fuse(t, *args, inplace=True, **kwargs)
