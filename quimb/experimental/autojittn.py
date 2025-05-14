"""Decorator for automatically just in time compiling tensor network functions.

TODO:
- [ ] check and cache on input shapes
"""
import functools

import autoray as ar

from quimb.tensor.interface import pack, unpack


def try_and_get_params(x):
    if hasattr(x, "get_params"):
        return x.get_params()
    return x


class AutojittedTN:
    def __init__(self, fn, decorator=ar.autojit, **decorator_opts):
        self.fn = fn
        self.jit_fn = None
        self.decorator = decorator
        self.decorator_opts = decorator_opts

    def _setup(self, *args, **kwargs):
        # extract all arrays to flat list, keep reference pytree and skeletons
        flat, ref_tree = ar.tree_flatten((args, kwargs), get_ref=True)
        _, skeletons = zip(*map(pack, flat))

        # now decorate the function that takes pytrees of arrays only

        @self.decorator(**self.decorator_opts)
        def pyfn(*pytrees):

            # inject back into the skeletons
            flat = [unpack(p, skel) for p, skel in zip(pytrees, skeletons)]
            args, kwargs = ar.tree_unflatten(flat, ref_tree)

            # call the original function
            result = self.fn(*args, **kwargs)

            # flatten the result to check if it needs unpacking
            flat_out, ref_tree_out = ar.tree_flatten(result, get_ref=True)

            if not any(hasattr(x, "set_params") for x in flat_out):
                # result doesnt need unpacking -> is a pure pytree
                self.ref_tree_out = None
                self.skeletons_out = None
                return result

            # need to unpack after each call
            params_out, skeletons_out = zip(*map(pack, flat_out))
            self.skeletons_out = skeletons_out
            self.ref_tree_out = ref_tree_out
            return params_out

        self.jit_fn = pyfn

    def __call__(self, *args, backend=None, **kwargs):
        if self.jit_fn is None:
            self._setup(*args, **kwargs)

        # extract to sequence of pytrees of arrays
        pytrees = tuple(
            try_and_get_params(x) for x in ar.tree_flatten((args, kwargs))
        )

        # call the traced function
        result = self.jit_fn(*pytrees, backend=backend)

        if self.skeletons_out is None:
            # no need to unpack
            return result

        # unpack into tensor networks instances etc
        flat = [unpack(p, skel) for p, skel in zip(result, self.skeletons_out)]
        return ar.tree_unflatten(flat, self.ref_tree_out)


def autojit_tn(
    fn=None,
    decorator=ar.autojit,
    **decorator_opts,
):
    """Decorate a tensor network function to be just in time compiled / traced.
    This traces solely array operations resulting in a completely static
    computational graph with no side-effects. The resulting function can be
    much faster if called repeatedly with only numeric changes, or hardware
    accelerated if a library such as ``jax`` is used.

    Parameters
    ----------
    fn : callable
        The function to be decorated.
    decorator : callable
        The decorator to use to wrap the underlying array function. For example
        ``jax.jit``. Defaults to ``autoray.autojit``.
    decorator_opts
        Options to pass to the decorator, e.g. ``backend`` for
        ``autoray.autojit``.
    """
    kwargs = {
        "decorator": decorator,
        **decorator_opts,
    }
    if fn is None:
        return functools.partial(autojit_tn, **kwargs)
    return functools.wraps(fn)(AutojittedTN(fn, **kwargs))
