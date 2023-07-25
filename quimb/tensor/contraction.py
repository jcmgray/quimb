"""Functions relating to tensor network contraction.
"""
import functools
import itertools
import threading
import contextlib
import collections

import opt_einsum as oe
from opt_einsum.contract import parse_backend
from autoray import infer_backend, shape

from ..utils import concat


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


if parse_backend([0], None) == 'numpy':
    # new enough version to support backend=None

    def _get_contract_expr(eq, *shapes, **kwargs):
        """Get the contraction expression - callable taking raw arrays.
        """
        return oe.contract_expression(eq, *shapes, **kwargs)

else:
    # old: wrap expression to always convert backend=None to 'auto'

    class ConvertBackendKwarg:

        __slots__ = ('fn',)

        def __init__(self, fn):
            self.fn = fn

        def __call__(self, *args, **kwargs):
            backend = kwargs.pop('backend', 'auto')
            if backend is None:
                backend = 'auto'
            return self.fn(*args, backend=backend, **kwargs)

    def _get_contract_expr(eq, *shapes, **kwargs):
        return ConvertBackendKwarg(
            oe.contract_expression(eq, *shapes, **kwargs)
        )


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


def get_contractor(
    eq,
    *shapes,
    cache=True,
    get='expr',
    optimize=None,
    use_cotengra="auto",
    **kwargs
):
    """Get an callable expression that will evaluate ``eq`` based on
    ``shapes``. Cache the result if no constant tensors are involved.

    Parameters
    ----------
    eq : str
        The equation to evaluate, for example ``'a,b->ab'``.
    shapes : tuple of ints
        The shapes of the tensors to be contracted.
    cache : bool, optional
        Whether to cache the contraction, only possible if ``optimize`` is
        not a ``PathOptimizer`` and no constants are specified.
    get : {'expr', 'path', 'info'}, optional
        Whether to return the expression, path, or info.
    optimize : {None, str, path_like, PathOptimizer}, optional
        The contraction path optimization strategy to use.

            - None: use the default strategy,
            - str: use the preset strategy with the given name,
            - path_like: use this exact path,
            - ``opt_einsum.PathOptimizer``: find the path using this optimizer.
            - ``cotengra.HyperOptimizer``: find and perform the contraction
              using ``cotengra.contract_expression``.
            - ``cotengra.ContractionTree``: use this exact tree and perform
              contraction using ``cotengra.contract_expression``.

        Contraction with ``cotengra`` might be a bit more efficient but the
        main reason would be to handle sliced contraction automatically.
    """
    if optimize is None:
        optimize = get_contract_strategy()

    use_cotengra_expression = (
        (get == 'expr') and
        (use_cotengra is not False) and
        (
            (use_cotengra is True) or
            (infer_backend(optimize) == 'cotengra')  # 'auto'
        )
    )

    if use_cotengra_expression:
        # can use more advanced contraction expression with slicing etc.
        import cotengra as ctg
        return ctg.contract_expression(
            eq, *shapes, optimize=optimize, **kwargs)

    # can't cache if using constants
    if 'constants' in kwargs:
        expr_fn = _CONTRACT_FNS['expr', False]
        expr = expr_fn(eq, *shapes, optimize=optimize, **kwargs)
        return expr

    # else make sure shapes are hashable + concrete python ints
    if not (
        isinstance(shapes[0], tuple) and
        isinstance(next(concat(shapes), 1), int)
    ):
        shapes = tuple(tuple(map(int, s)) for s in shapes)

    # and make sure explicit paths are hashable
    if isinstance(optimize, list):
        optimize = tuple(optimize)

    # don't cache path if using a 'single-shot' path-optimizer
    #     (you may want to run these several times, each time improving path)
    cache_path = cache and not isinstance(optimize, oe.paths.PathOptimizer)

    # get the path, unless explicitly given already, whether we cache this is
    # separate from the cache for the actual expression
    if not isinstance(optimize, tuple):
        path_fn = _CONTRACT_FNS['path', cache_path]
        path = path_fn(eq, *shapes, optimize=optimize, **kwargs)
    else:
        path = optimize

    if get == 'expr':
        expr_fn = _CONTRACT_FNS['expr', cache]
        expr = expr_fn(eq, *shapes, optimize=path, **kwargs)
        return expr

    if get == 'path':
        return path

    if get == 'info':
        info_fn = _CONTRACT_FNS['info', cache]
        info = info_fn(eq, *shapes, optimize=path, **kwargs)
        return info


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


def inds_to_symbols(inputs):
    """Map a sequence of inputs terms, containing any hashable indices, to
    single unicode letters, appropriate for einsum.

    Parameters
    ----------
    inputs : sequence of sequence of hashable
        The input indices per tensor.

    Returns
    -------
    symbols : dict[hashable, str]
        The mapping from index to symbol.
    """
    symbols = empty_symbol_map()
    return {
        ix: symbols[ix]
        for term in inputs
        for ix in term
    }


@functools.lru_cache(2**12)
def inds_to_eq(inputs, output=None):
    """Turn input and output indices of any sort into a single 'equation'
    string where each index is a single 'symbol' (unicode character).

    Parameters
    ----------
    inputs : sequence of sequence of hashable
        The input indices per tensor.
    output : sequence of hashable
        The output indices.

    Returns
    -------
    eq : str
        The string to feed to einsum/contract.
    """
    symbols = empty_symbol_map()
    in_str = ("".join(symbols[ix] for ix in inds) for inds in inputs)
    in_str = ",".join(in_str)
    if output is None:
        out_str = "".join(
            ix for ix in symbols.values() if in_str.count(ix) == 1
        )
    else:
        out_str = "".join(symbols[ix] for ix in output)
    return f"{in_str}->{out_str}"


def array_contract(
    arrays,
    inputs,
    output=None,
    optimize=None,
    backend=None,
    **contract_opts
):
    """Contraction interface for raw arrays with arbitrary hashable indices.

    Parameters
    ----------
    arrays : sequence of array_like
        The arrays to contract.
    inputs : sequence of sequence of hashable
        The input indices per tensor.
    output : sequence of hashable, optional
        The output indices, will be computed as every index that appears only
        once in the inputs if not given.
    optimize : None, str, path_like, PathOptimizer, optional
        How to compute the contraction path.
    backend : None, str, optional
        Which backend to use for the contraction.
    contract_opts
        Supplied to ``contract_expression``.

    Returns
    -------
    array_like
    """
    eq = inds_to_eq(inputs, output)
    shapes = tuple(shape(a) for a in arrays)
    f = get_contractor(eq, *shapes, optimize=optimize, **contract_opts)
    return f(*arrays, backend=backend)

try:
    from opt_einsum.contract import infer_backend as _oe_infer_backend
    del _oe_infer_backend
    _CONTRACT_BACKEND = None
    _TENSOR_LINOP_BACKEND = None
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
