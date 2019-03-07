"""Backend agnostic array operations.
"""

import importlib
import numpy


def infer_backend(x):
    if isinstance(x, numpy.ndarray):
        return 'numpy'
    return x.__class__.__module__.split('.')[0]


# global (non function specific) aliases
_module_aliases = {
    'decimal': 'math',
    'builtins': 'numpy',
    'dask': 'dask.array',
}


# lookup for when functions are elsewhere than the expected module
_submodule_aliases = {}


# lookup for when functions don't have the same name
_func_aliases = {
    ('tensorflow', 'sum'): 'reduce_sum',
    ('tensorflow', 'min'): 'reduce_min',
    ('tensorflow', 'max'): 'reduce_max',
    ('tensorflow', 'mean'): 'reduce_mean',
    ('tensorflow', 'prod'): 'reduce_prod',
}


# actual cache of funtions to use
_funcs = {}


def do(fn, x, *args, **kwargs):
    """Do function named ``fn`` on array ``x``, peforming single dispatch based
    on whichever library defines the class of ``x``.

    Examples
    --------

    Works on numpy arrays:

        >>> import numpy as np
        >>> x_np = np.random.uniform(size=[5])
        >>> y_np = do('sqrt', x_np)
        >>> y_np
        array([0.32464973, 0.90379787, 0.85037325, 0.88729814, 0.46768083])

        >>> type(y_np)
        numpy.ndarray

    Works on cupy arrays:

        >>> import cupy as cp
        >>> x_cp = cp.random.uniform(size=[5])
        >>> y_cp = do('sqrt', x_cp)
        >>> y_cp
        array([0.44541656, 0.88713113, 0.92626237, 0.64080557, 0.69620767])

        >>> type(y_cp)
        cupy.core.core.ndarray

    Works on tensorflow arrays:

        >>> import tensorflow as tf
        >>> x_tf = tf.random.uniform(shape=[5])
        >>> y_tf = do('sqrt', x_tf)
        >>> y_tf
        <tf.Tensor 'Sqrt_1:0' shape=(5,) dtype=float32>

        >>> type(y_tf)
        tensorflow.python.framework.ops.Tensor

    You get the idea.
    """
    backend = infer_backend(x)

    # cached retrieval of correct function for backend
    try:
        lib_fn = _funcs[backend, fn]
    except KeyError:
        # alias for global module
        module = _module_aliases.get(backend, backend)

        # module where function is found for backend
        submodule_name = _submodule_aliases.get((backend, fn), module)

        # parse out extra submodules (e.g. for ``fn='linalg.eigh'``)
        split_fn = fn.split('.')
        submodule_name = '.'.join([submodule_name] + split_fn[:-1])
        only_fn = split_fn[-1]

        # cached lookup of custom name function might take
        fn_name = _func_aliases.get((backend, fn), only_fn)

        # import the function into the cache
        lib = importlib.import_module(submodule_name)
        lib_fn = _funcs[backend, fn] = getattr(lib, fn_name)

    return lib_fn(x, *args, **kwargs)


# --------------------- attribute preferring functions ---------------------- #

def conj(x):
    try:
        return x.conj()
    except AttributeError:
        return do('conj', x)


def real(x):
    try:
        return x.real
    except AttributeError:
        return do('real', x)


def imag(x):
    try:
        return x.imag
    except AttributeError:
        return do('imag', x)


def reshape(x, shape):
    try:
        return x.reshape(shape)
    except AttributeError:
        return do('reshape', x, shape)


# ------------- miscelleneous other backend agnostic functions -------------- #

def iscomplex(x):
    if not hasattr(x, 'dtype'):
        return isinstance(x, complex)
    return 'complex' == x.dtype.name[:7]
