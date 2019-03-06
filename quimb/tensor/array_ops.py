"""Backend agnostic array operations.
"""

import importlib
import functools


def infer_backend(x):
    return x.__class__.__module__.split('.')[0]


# --------------------------------- conj ------------------------------------ #

_conj_aliases = {
    'tensorflow': 'tensorflow.math'
}


@functools.lru_cache()
def get_conj_fn(backend):
    module_name = _conj_aliases.get(backend, backend)
    module = importlib.import_module(module_name)
    return getattr(module, 'conj')


def conj(x):
    try:
        return x.conj()
    except AttributeError:
        return get_conj_fn(infer_backend(x))(x)


# -------------------------------- reshape ---------------------------------- #

_reshape_aliases = {}


@functools.lru_cache()
def get_reshape_fn(backend):
    module_name = _reshape_aliases.get(backend, backend)
    module = importlib.import_module(module_name)
    return getattr(module, 'reshape')


def reshape(x, shape):
    try:
        return x.reshape(shape)
    except AttributeError:
        return get_reshape_fn(infer_backend(x))(x, shape)


def iscomplex(x):
    if not hasattr(x, 'dtype'):
        return isinstance(x, complex)
    return 'complex' == x.dtype.name[:7]


# ---------------------------------- sign ----------------------------------- #

_sign_aliases = {
    'tensorflow': 'tensorflow.math',
}


def builtin_sign(x):
    return -1.0 if x < 0.0 else 1.0


@functools.lru_cache()
def get_sign_fn(backend):
    module_name = _sign_aliases.get(backend, backend)

    if module_name == 'builtins':
        return builtin_sign

    module = importlib.import_module(module_name)
    return getattr(module, 'sign')


def sign(x):
    return get_sign_fn(infer_backend(x))(x)


# ------------------------------- real/imag --------------------------------- #

_real_aliases = {
    'tensorflow': 'tensorflow.math'
}


@functools.lru_cache()
def get_real_fn(backend):
    module_name = _real_aliases.get(backend, backend)
    module = importlib.import_module(module_name)
    return getattr(module, 'real')


def real(x):
    try:
        return x.real
    except AttributeError:
        return get_real_fn(infer_backend(x))(x)


_imag_aliases = {
    'tensorflow': 'tensorflow.math'
}


@functools.lru_cache()
def get_imag_fn(backend):
    module_name = _imag_aliases.get(backend, backend)
    module = importlib.import_module(module_name)
    return getattr(module, 'imag')


def imag(x):
    try:
        return x.imag
    except AttributeError:
        return get_imag_fn(infer_backend(x))(x)
