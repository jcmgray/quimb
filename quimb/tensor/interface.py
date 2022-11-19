import functools

from ..utils import tree_map
from .tensor_core import Tensor, TensorNetwork


class Placeholder:

    __slots__ = ("shape",)

    def __init__(self, shape):
        self.shape = shape

    def __repr__(self):
        return f"Placeholder(shape={self.shape})"


def pack(obj):
    """Take a tensor or tensor network like object and return a aux needed
    to reconstruct it, and a pytree of raw parameters.

    Parameters
    ----------
    obj : Tensor, TensorNetwork, or similar
        Something that has ``copy``, ``set_params``, and ``get_params``
        methods.

    Returns
    -------
    children : pytree
        A pytree of raw parameter arrays.
    aux : Tensor, TensorNetwork, or similar
        A copy of ``obj`` with all references to the original data removed.
    """
    aux = obj.copy()
    children = aux.get_params()
    placeholders = tree_map(lambda x: Placeholder(x.shape), children)
    aux.set_params(placeholders)
    return children, aux


def unpack(children, aux):
    """Take a aux of a tensor or tensor network like object and a pytree
    of raw parameters and return a new reconstructed object with those
    parameters inserted.

    Parameters
    ----------
    children : pytree
        A pytree of raw parameter arrays, with the same structure as the
        output of ``aux.get_params()``.
    aux : Tensor, TensorNetwork, or similar
        Something that has ``copy``, ``set_params``, and ``get_params``
        methods.

    Returns
    -------
    obj : Tensor, TensorNetwork, or similar
        A copy of ``aux`` with parameters inserted.
    """
    obj = aux.copy()
    obj.set_params(children)
    return obj


# -------------------------------- jax -------------------------------------- #


_JAX_REGISTERED_TN_CLASSES = {}


def jax_register_pytree():
    import jax

    queue = [Tensor, TensorNetwork]
    while queue:
        cls = queue.pop()
        if cls not in _JAX_REGISTERED_TN_CLASSES:
             jax.tree_util.register_pytree_node(cls, pack, unpack)
        queue.extend(cls.__subclasses__())


@functools.lru_cache(1)
def get_jax():
    import jax

    jax_register_pytree()
    return jax
