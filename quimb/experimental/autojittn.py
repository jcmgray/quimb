"""Decorator for automatically just in time compiling tensor network functions.

TODO::

    - [ ] go via an intermediate pytree / array function, that could be shared
          e.g. with the TNOptimizer class.

"""
import functools

import autoray as ar



class AutojittedTN:
    """Class to hold the ``autojit_tn`` decorated function callable.
    """

    def __init__(
        self,
        fn,
        decorator=ar.autojit,
        check_inputs=True,
        **decorator_opts
    ):
        self.fn = fn
        self.fn_store = {}
        self.decorator_opts = decorator_opts
        self.check_inputs = check_inputs
        self.decorator = decorator

    def setup_fn(self, tn, *args, **kwargs):
        from quimb.tensor import TensorNetwork

        @self.decorator(**self.decorator_opts)
        def fn_jit(arrays):
            # use separate TN to trace through function
            jtn = tn.copy()

            # insert the tracing arrays
            for t, array in zip(jtn, arrays):
                t.modify(data=array)

            # run function on TN with tracing arrays
            result = self.fn(jtn, *args, **kwargs)

            # check for a inplace tn function
            if isinstance(result, TensorNetwork):
                if result is not jtn:
                    raise ValueError(
                        "If you are compiling a function that returns a"
                        " tensor network it needs to be inplace.")
                self.inplace = True
                return tuple(t.data for t in jtn)
            else:
                # function returns raw scalar/array(s)
                self.inplace = False
                return result

        return fn_jit

    def __call__(self, tn, *args, **kwargs):

        # do we need to generate a new function for these inputs
        if self.check_inputs:
            key = (
                tn.geometry_hash(strict_index_order=True),
                tuple(args),
                tuple(sorted(kwargs.items())),
            )
        else:
            # always use the same function
            key = None

        if key not in self.fn_store:
            self.fn_store[key] = self.setup_fn(tn, *args, **kwargs)
        fn_jit = self.fn_store[key]

        # run the compiled function
        arrays = tuple(t.data for t in tn)
        out = fn_jit(arrays)

        if self.inplace:
            # reinsert output arrays into input TN structure
            for t, array in zip(tn, out):
                t.modify(data=array)
            return tn

        return out


def autojit_tn(
    fn=None,
    decorator=ar.autojit,
    check_inputs=True,
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
        The function to be decorated. It should take as its first argument a
        :class:`~quimb.tensor.tensor_core.TensorNetwork` and return either act
        inplace on it or return a raw scalar or array.
    decorator : callable
        The decorator to use to wrap the underlying array function. For example
        ``jax.jit``. Defaults to ``autoray.autojit``.
    check_inputs : bool, optional
        Whether to check the inputs to the function every call to see if a new
        compiled function needs to be generated. If ``False`` the same compiled
        function will be used for all inputs which might be incorrect. Defaults
        to ``True``.
    decorator_opts
        Options to pass to the decorator, e.g. ``backend`` for
        ``autoray.autojit``.
    """
    kwargs = {
        'decorator': decorator,
        'check_inputs': check_inputs,
        **decorator_opts,
    }
    if fn is None:
        return functools.partial(autojit_tn, **kwargs)
    return functools.wraps(fn)(AutojittedTN(fn, **kwargs))
