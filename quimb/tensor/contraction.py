"""Functions relating to tensor network contraction.
"""
import functools
import itertools
import threading
import contextlib
import collections

import  cotengra as ctg


_CONTRACT_STRATEGY = 'greedy'
_TEMP_CONTRACT_STRATEGIES = collections.defaultdict(list)


def get_contract_strategy():
    r"""Get the default contraction strategy - the option supplied as
    ``optimize`` to ``cotengra``.
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
    ``optimize`` to ``cotengra``.
    """
    global _CONTRACT_STRATEGY
    _CONTRACT_STRATEGY = strategy


@contextlib.contextmanager
def contract_strategy(strategy, set_globally=False):
    """A context manager to temporarily set the default contraction strategy
    supplied as ``optimize`` to ``cotengra``. By default, this only sets the
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


get_symbol = ctg.get_symbol


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
    return ctg.get_symbol_map(inputs)


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


_CONTRACT_BACKEND = None
_TENSOR_LINOP_BACKEND = None
_TEMP_CONTRACT_BACKENDS = collections.defaultdict(list)
_TEMP_TENSOR_LINOP_BACKENDS = collections.defaultdict(list)


def get_contract_backend():
    """Get the default backend used for tensor contractions, via 'cotengra'.

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
    """Set the default backend used for tensor contractions, via 'cotengra'.

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
    contractions, via 'cotengra'. By default, this only sets the contract
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
    'cotengra'. This is different from the default contraction backend as
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
    'cotengra'. This is different from the default contraction backend as
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
    network linear operators, via 'cotengra'. By default, this
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


@functools.wraps(ctg.array_contract)
def array_contract(
    arrays,
    inputs,
    output=None,
    optimize=None,
    backend=None,
    **kwargs,
):
    if optimize is None:
        optimize = get_contract_strategy()
    if backend is None:
        backend = get_contract_backend()
    return ctg.array_contract(
        arrays, inputs, output, optimize=optimize, backend=backend, **kwargs
    )


@functools.wraps(ctg.array_contract_expression)
def array_contract_expression(*args, optimize=None, **kwargs):
    if optimize is None:
        optimize = get_contract_strategy()
    return ctg.array_contract_expression(*args, optimize=optimize, **kwargs)


@functools.wraps(ctg.array_contract_tree)
def array_contract_tree(*args, optimize=None, **kwargs):
    if optimize is None:
        optimize = get_contract_strategy()
    return ctg.array_contract_tree(*args, optimize=optimize, **kwargs)


@functools.wraps(ctg.array_contract_path)
def array_contract_path(*args, optimize=None, **kwargs):
    if optimize is None:
        optimize = get_contract_strategy()
    return ctg.array_contract_path(*args, optimize=optimize, **kwargs)


def array_contract_pathinfo(*args, **kwargs):
    import opt_einsum as oe

    tree = array_contract_tree(*args, **kwargs)

    if tree.sliced_inds:
        import warnings

        warnings.warn(
            "The contraction tree has sliced indices, which are not "
            "supported by opt_einsum. Ignoring them for now."
        )

    shapes = tree.get_shapes()
    path = tree.get_path()
    eq = tree.get_eq()

    return oe.contract_path(eq, *shapes, shapes=True, optimize=path)[1]

