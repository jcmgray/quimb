"""Tools for interfacing the tensor and tensor network objects with other
libraries.
"""

import functools

from ..utils import tree_map
from .tensor_core import Tensor, TensorNetwork


class Placeholder:

    __slots__ = ("shape",)

    def __init__(self, x):
        self.shape = getattr(x, "shape", None)

    def __repr__(self):
        return f"Placeholder(shape={self.shape})"


def pack(obj):
    """Take a tensor or tensor network like object and return a skeleton needed
    to reconstruct it, and a pytree of raw parameters.

    Parameters
    ----------
    obj : Tensor, TensorNetwork, or similar
        Something that has ``copy``, ``set_params``, and ``get_params``
        methods.

    Returns
    -------
    params : pytree
        A pytree of raw parameter arrays.
    skeleton : Tensor, TensorNetwork, or similar
        A copy of ``obj`` with all references to the original data removed.
    """
    skeleton = obj.copy()
    params = skeleton.get_params()
    placeholders = tree_map(Placeholder, params)
    skeleton.set_params(placeholders)
    return params, skeleton


def unpack(params, skeleton):
    """Take a skeleton of a tensor or tensor network like object and a pytree
    of raw parameters and return a new reconstructed object with those
    parameters inserted.

    Parameters
    ----------
    params : pytree
        A pytree of raw parameter arrays, with the same structure as the
        output of ``skeleton.get_params()``.
    skeleton : Tensor, TensorNetwork, or similar
        Something that has ``copy``, ``set_params``, and ``get_params``
        methods.

    Returns
    -------
    obj : Tensor, TensorNetwork, or similar
        A copy of ``skeleton`` with parameters inserted.
    """
    obj = skeleton.copy()
    obj.set_params(params)
    return obj


# -------------------------------- jax -------------------------------------- #


_JAX_REGISTERED_TN_CLASSES = set()


def jax_pack(obj):
    # jax requires the top level children to be a tuple
    params, aux = pack(obj)
    children = (params,)
    return children, aux


def jax_unpack(aux, children):
    # jax also flips the return order from above
    (params,) = children
    return unpack(params, aux)


def jax_register_pytree():
    import jax

    queue = [Tensor, TensorNetwork]
    while queue:
        cls = queue.pop()
        if cls not in _JAX_REGISTERED_TN_CLASSES:
            jax.tree_util.register_pytree_node(cls, jax_pack, jax_unpack)
            _JAX_REGISTERED_TN_CLASSES.add(cls)
        queue.extend(cls.__subclasses__())


@functools.lru_cache(1)
def get_jax():
    import jax

    jax_register_pytree()
    return jax
