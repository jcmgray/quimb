"""Backend agnostic array operations.
"""

import importlib
import functools


def infer_backend(x):
    return x.__class__.__module__.split('.')[0]


_CONJ_ALIASES = {
    'tensorflow': 'tensorflow.math'
}


@functools.lru_cache()
def get_conj_fn(backend):
    module_name = _CONJ_ALIASES.get(backend, backend)
    module = importlib.import_module(module_name)
    return getattr(module, 'conj')


def conj(x):
    try:
        return x.conj()
    except AttributeError:
        return get_conj_fn(infer_backend(x))(x)
