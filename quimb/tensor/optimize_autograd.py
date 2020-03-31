"""Support for optimizing tensor networks using ``autograd`` or ``jax`` to
automatically derive gradients for input to scipy.
"""
import re
import functools
import importlib

import tqdm
import numpy as np
from cytoolz import valmap

from .tensor_core import (
    contract_backend,
    Tensor,
    TensorNetwork,
    PTensor,
    utup_intersection,
    tags_to_utup,
)
from .array_ops import iscomplex
from ..core import qarray

if importlib.util.find_spec("jax") is not None:
    _DEFAULT_BACKEND = 'jax'
elif importlib.util.find_spec("tensorflow") is not None:
    _DEFAULT_BACKEND = 'tensorflow'
else:
    _DEFAULT_BACKEND = 'autograd'


_REAL_CONVERSION = {
    'float32': 'float32',
    'float64': 'float64',
    'complex64': 'float32',
    'complex128': 'float64',
}

_COMPLEX_CONVERSION = {
    'float32': 'complex64',
    'float64': 'complex128',
    'complex64': 'complex64',
    'complex128': 'complex128',
}


def equivalent_real_type(x):
    return _REAL_CONVERSION[x.dtype.name]


def equivalent_complex_type(x):
    return _COMPLEX_CONVERSION[x.dtype.name]


class Vectorizer:
    """Object for mapping a sequence of mixed real/complex n-dimensional arrays
    to a single numpy vector and back and forth.
    """

    def __init__(self, arrays):
        self.shapes = [x.shape for x in arrays]
        self.iscomplexes = [iscomplex(x) for x in arrays]
        self.dtypes = [x.dtype for x in arrays]
        self.sizes = [np.prod(s) for s in self.shapes]
        self.d = sum(
            (1 + int(cmplx)) * size
            for size, cmplx in zip(self.sizes, self.iscomplexes)
        )
        self.pack(arrays)

    def pack(self, arrays, name='vector'):

        # scipy's optimization routines require real, double data
        if not hasattr(self, name):
            setattr(self, name, np.empty(self.d, 'float64'))
        x = getattr(self, name)

        i = 0
        for array, size, cmplx in zip(arrays, self.sizes, self.iscomplexes):
            array = np.asarray(array)
            if not cmplx:
                x[i:i + size] = array.reshape(-1)
                i += size
            else:
                real_view = array.reshape(-1).view(equivalent_real_type(array))
                x[i:i + 2 * size] = real_view
                i += 2 * size

        return x

    def unpack(self, vector=None):
        """Turn the single, flat ``vector`` into a sequence of arrays.
        """
        if vector is None:
            vector = self.vector

        i = 0
        arrays = []
        for shape, size, cmplx, dtype in zip(self.shapes, self.sizes,
                                             self.iscomplexes, self.dtypes):
            if not cmplx:
                array = vector[i:i + size]
                array.shape = shape
                i += size
            else:
                array = vector[i:i + 2 * size]
                array = array.view(equivalent_complex_type(array))
                array.shape = shape
                i += 2 * size

            if array.dtype != dtype:
                array = array.astype(dtype)
            arrays.append(array)

        return arrays


def parse_network_to_backend(tn, constant_tags, to_constant):
    tn_ag = tn.copy()
    variables = []

    variable_tag = "__VARIABLE{}__"

    for t in tn_ag:
        # check if tensor has any of the constant tags
        if utup_intersection((t.tags, constant_tags)):
            t.modify(data=to_constant(t.data))
            continue

        if isinstance(t, PTensor):
            data = t.params
        else:
            data = t.data

        # jax doesn't like numpy.ndarray subclasses...
        if isinstance(data, qarray):
            data = data.A

        # append the raw data but mark the corresponding tensor for reinsertion
        variables.append(data)
        t.add_tag(variable_tag.format(len(variables) - 1))

    return tn_ag, variables


def constant_t(t, to_constant):
    ag_t = t.copy()
    ag_t.modify(data=to_constant(ag_t.data))
    return ag_t


def constant_tn(tn, to_constant):
    """Convert a tensor network's arrays to constants.
    """
    ag_tn = tn.copy()
    ag_tn.apply_to_arrays(to_constant)
    return ag_tn


class AutoGradHandler:

    def __init__(self):
        import autograd
        self.autograd = autograd

    def to_variable(self, x):
        return np.array(x)

    def to_constant(self, x):
        return np.array(x)

    def setup_fn(self, fn):
        self._value_and_grad = self.autograd.value_and_grad(fn)

    def value_and_grad(self, arrays):
        loss, grads = self._value_and_grad(arrays)
        return loss, [x.conj() for x in grads]


class JaxHandler:

    def __init__(self, jit_fn=True):
        import jax
        self.jit_fn = jit_fn
        self.jax = jax

    def to_variable(self, x):
        return self.jax.numpy.array(x)

    def to_constant(self, x):
        return self.jax.numpy.array(x)

    def setup_fn(self, fn):
        if self.jit_fn:
            jfn = self.jax.jit(fn)
        else:
            jfn = fn

        self._value_and_grad = self.jax.value_and_grad(jfn)

    def value_and_grad(self, arrays):
        loss, grads = self._value_and_grad(arrays)
        return loss, [x.conj() for x in grads]


class TensorFlowHandler:

    def __init__(self, jit_fn=False, autograph=False):
        import tensorflow
        self.tensorflow = tensorflow
        self.jit_fn = jit_fn
        self.autograph = autograph

    def to_variable(self, x):
        return self.tensorflow.Variable(x)

    def to_constant(self, x):
        return self.tensorflow.constant(x)

    def setup_fn(self, fn):
        if self.jit_fn:
            self._backend_fn = self.tensorflow.function(
                fn, autograph=self.autograph)
        else:
            self._backend_fn = fn

    def value_and_grad(self, arrays):
        variables = [self.to_variable(x) for x in arrays]

        with self.tensorflow.GradientTape() as t:
            result = self._backend_fn(variables)

        grads = [x.numpy() for x in t.gradient(result, variables)]
        loss = result.numpy()

        return loss, grads


class TorchHandler:

    def __init__(self, jit_fn=False, device=None):
        import torch
        self.torch = torch
        self.jit_fn = jit_fn
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device

    def to_variable(self, x):
        return self.torch.tensor(x, requires_grad=True, device=self.device)

    def to_constant(self, x):
        return self.torch.tensor(x, device=self.device)

    def setup_fn(self, fn):
        self._fn = fn
        self._backend_fn = None

    def _setup_backend_fn(self, arrays):
        if self.jit_fn:
            self._backend_fn = self.torch.jit.trace(
                self._fn, example_inputs=[list(map(self.to_constant, arrays))])
        else:
            self._backend_fn = self._fn

    def value_and_grad(self, arrays):
        if self._backend_fn is None:
            self._setup_backend_fn(arrays)

        arrays = [self.to_variable(x) for x in arrays]
        result = self._backend_fn(arrays)
        result.backward()

        loss = result.detach().cpu().numpy()
        grads = [x.grad.cpu().numpy() for x in arrays]

        return loss, grads


_BACKEND_HANDLERS = {
    'autograd': AutoGradHandler,
    'jax': JaxHandler,
    'tensorflow': TensorFlowHandler,
    'torch': TorchHandler,
}


variable_finder = re.compile(r'__VARIABLE(\d+)__')


def inject_(arrays, tn):
    for t in tn:
        for tag in t.tags:
            match = variable_finder.match(tag)
            if match is not None:
                i = int(match.groups(1)[0])

                if isinstance(t, PTensor):
                    t.params = arrays[i]
                else:
                    t.modify(data=arrays[i])

                break


class TNOptimizer:

    def __init__(
        self,
        tn,
        loss_fn,
        norm_fn=None,
        loss_constants=None,
        loss_kwargs=None,
        constant_tags=None,
        loss_target=None,
        optimizer='L-BFGS-B',
        progbar=True,
        bounds=None,
        autograd_backend='AUTO',
        # the rest are ignored for compat
        learning_rate=None,
        learning_decay_steps=None,
        learning_decay_rate=None,
        **backend_opts
    ):
        self.progbar = progbar
        self.optimizer = optimizer
        self.bounds = bounds
        self.constant_tags = tags_to_utup(constant_tags)

        # the object that handles converting to backend + computing gradient
        if autograd_backend.upper() == 'AUTO':
            autograd_backend = _DEFAULT_BACKEND
        self.handler = _BACKEND_HANDLERS[autograd_backend](**backend_opts)
        to_constant = self.handler.to_constant

        # use identity if no nomalization required
        if norm_fn is None:
            def norm_fn(x):
                return x

        self.norm_fn = norm_fn

        # convert constant arrays ahead of time to correct backend
        self.loss_constants = {}
        if loss_constants is not None:
            for k, v in loss_constants.items():
                # check if tensor network supplied
                if isinstance(v, TensorNetwork):
                    # convert it to constant TN
                    self.loss_constants[k] = constant_tn(v, to_constant)
                elif isinstance(v, Tensor):
                    self.loss_constants[k] = constant_t(v, to_constant)
                elif isinstance(v, dict):
                    self.loss_constants[k] = valmap(to_constant, v)
                else:
                    self.loss_constants[k] = to_constant(v)

        self.loss_kwargs = {} if loss_kwargs is None else dict(loss_kwargs)
        self.loss_fn = functools.partial(
            loss_fn, **self.loss_constants, **self.loss_kwargs
        )

        self.loss = float('inf')
        self.loss_best = float('inf')
        self.loss_target = loss_target
        self.losses = []
        self._n = 0

        self.tn_opt, self.variables = parse_network_to_backend(
            tn, self.constant_tags, to_constant
        )

        # this handles storing and packing /  unpacking many arrays as a vector
        self.vctrzr = Vectorizer(self.variables)

        if bounds is not None:
            bounds = (bounds,) * self.vctrzr.d
        self.bounds = bounds

        def func(arrays):
            # set backend explicitly as maybe mixing with numpy arrays
            with contract_backend(autograd_backend):
                inject_(arrays, self.tn_opt)
                return self.loss_fn(self.norm_fn(self.tn_opt))

        self.handler.setup_fn(func)

        def vectorized_value_and_grad(x):
            arrays = self.vctrzr.unpack(x)
            ag_result, ag_grads = self.handler.value_and_grad(arrays)

            self._n += 1
            self.loss = ag_result.item()
            self.losses.append(self.loss)

            vec_grad = self.vctrzr.pack(ag_grads, 'grad')

            return self.loss, vec_grad

        self.vectorized_value_and_grad = vectorized_value_and_grad

    @property
    def nevals(self):
        """The number of gradient evaluations.
        """
        return self._n

    def inject_res_vector_and_return_tn(self):
        arrays = self.vctrzr.unpack(self.res.x)
        inject_(arrays, self.tn_opt)
        self.vctrzr.vector[:] = self.res.x
        tn = self.norm_fn(self.tn_opt.copy())
        tn.drop_tags(t for t in tn.tags if variable_finder.match(t))
        return tn

    def optimize(self, n, tol=None, **options):
        from scipy.optimize import minimize

        try:
            pbar = tqdm.tqdm(total=n, disable=not self.progbar)

            def callback(_):
                pbar.update()
                pbar.set_description(f"{self.loss}")

            self.res = minimize(
                fun=self.vectorized_value_and_grad,
                jac=True,
                x0=self.vctrzr.vector,
                callback=callback,
                tol=tol,
                bounds=self.bounds,
                method=self.optimizer,
                options=dict(
                    maxiter=n,
                    **options,
                )
            )

        finally:
            pbar.close()

        return self.inject_res_vector_and_return_tn()

    def optimize_basinhopping(self, n, nhop, temperature=1.0):
        from scipy.optimize import basinhopping

        try:
            pbar = tqdm.tqdm(total=n * nhop, disable=not self.progbar)

            def callback(x, f, accept):
                if self.loss_target is not None:
                    if self.loss_best < self.loss_target:
                        return True

            def inner_callback(xk):
                self.loss_best = min(self.loss_best, self.loss)
                pbar.update()
                msg = f"{self.loss} [best: {self.loss_best}] "
                pbar.set_description(msg)

            self.res = basinhopping(
                func=self.vectorized_value_and_grad,
                x0=self.vctrzr.vector,
                niter=nhop,
                minimizer_kwargs=dict(
                    jac=True,
                    method=self.optimizer,
                    bounds=self.bounds,
                    callback=inner_callback,
                    options=dict(
                        maxiter=n,
                    )
                ),
                callback=callback,
                T=temperature,
            )

        finally:
            pbar.close()

        return self.inject_res_vector_and_return_tn()
