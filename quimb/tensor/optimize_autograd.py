"""Support for optimizing tensor networks using ``autograd`` or ``jax`` to
automatically derive gradients for input to scipy.
"""
import re
import functools
import importlib

import tqdm
import numpy as np
from autoray import do, to_numpy

from .tensor_core import contract_backend, Tensor, TensorNetwork, PTensor
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

    def unpack(self, vector=None, conj=False):
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
                # XXX: need to conjugate for autograd/jax, convention??
                if conj:
                    array = np.conj(array)
                array.shape = shape
                i += 2 * size

            if array.dtype != dtype:
                array = array.astype(dtype)
            arrays.append(array)

        return arrays


def parse_network_to_ag(tn, constant_tags, backend=_DEFAULT_BACKEND):
    tn_ag = tn.copy()
    variables = []

    variable_tag = "__VARIABLE{}__"

    for t in tn_ag:
        # check if tensor has any of the constant tags
        if t.tags & constant_tags:
            t.modify(data=constant(t.data, backend=backend))
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


def constant(x, backend=_DEFAULT_BACKEND):
    if backend == 'jax' and isinstance(x, qarray):
        x = x.A
    return do('array', x, like=backend)


def constant_t(t, backend=_DEFAULT_BACKEND):
    ag_t = t.copy()
    ag_t.modify(data=constant(ag_t.data, backend=backend))
    return ag_t


def constant_tn(tn, backend=_DEFAULT_BACKEND):
    """Convert a tensor network's arrays to constants.
    """
    ag_tn = tn.copy()
    ag_tn.apply_to_arrays(functools.partial(constant, backend=backend))
    return ag_tn


def tensorflow_value_and_grad(fn):
    """TensorFlow 2 version of ``value_and_grad``.
    """
    import tensorflow as tf

    jit_fn = tf.function(fn, autograph=False)

    def fn_value_and_grad(arrays):
        arrays = [tf.constant(x) for x in arrays]

        # explicitly only watch supplied arrays
        with tf.GradientTape(watch_accessed_variables=False) as t:
            t.watch(arrays)
            result = jit_fn(arrays)

        return result, t.gradient(result, arrays)

    return fn_value_and_grad


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
    ):
        self.progbar = progbar
        self.optimizer = optimizer
        self.bounds = bounds
        self.constant_tags = (
            set() if constant_tags is None else set(constant_tags)
        )

        if autograd_backend.upper() == 'AUTO':
            autograd_backend = _DEFAULT_BACKEND
        self.autograd_backend = autograd_backend

        # use identity if no nomalization required
        if norm_fn is None:
            def norm_fn(x):
                return x

        self.norm_fn = norm_fn

        self.loss_kwargs = {} if loss_kwargs is None else dict(loss_kwargs)

        self.loss_constants = {}
        if loss_constants is not None:
            for k, v in loss_constants.items():
                # check if tensor network supplied
                if isinstance(v, TensorNetwork):
                    # convert it to constant TN
                    self.loss_constants[k] = constant_tn(v, autograd_backend)
                elif isinstance(v, Tensor):
                    self.loss_constants[k] = constant_t(v, autograd_backend)
                else:
                    self.loss_constants[k] = constant(v, autograd_backend)

        self.loss_fn = functools.partial(
            loss_fn, **self.loss_constants, **self.loss_kwargs
        )

        self.loss = np.inf
        self.loss_best = np.inf
        self.loss_target = loss_target
        self.losses = []
        self._n = 0

        self.tn_opt, self.variables = parse_network_to_ag(
            tn, self.constant_tags, backend=self.autograd_backend,
        )

        # this handles storing and packing /  unpacking many arrays as a vector
        self.vctrzr = Vectorizer(self.variables)

        if bounds is not None:
            bounds = (bounds,) * self.vctrzr.d
        self.bounds = bounds

        def func(arrays):
            # need to set backend explicitly as mixing with numpy arrays
            with contract_backend(self.autograd_backend):
                inject_(arrays, self.tn_opt)
                return self.loss_fn(self.norm_fn(self.tn_opt))

        if self.autograd_backend == 'jax':
            import jax
            self.func_val_and_grad = jax.value_and_grad(jax.jit(func))
        elif self.autograd_backend == 'autograd':
            import autograd
            self.func_val_and_grad = autograd.value_and_grad(func)
        elif self.autograd_backend == 'tensorflow':
            self.func_val_and_grad = tensorflow_value_and_grad(func)

        def vectorized_value_and_grad(x):
            arrays = self.vctrzr.unpack(x, conj=True)
            ag_result, ag_grads = self.func_val_and_grad(arrays)
            self._n += 1

            self.loss = to_numpy(ag_result)
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
        inject_(self.vctrzr.unpack(self.res.x, conj=True), self.tn_opt)
        self.vctrzr.vector[:] = self.res.x
        tn = self.norm_fn(self.tn_opt.copy())
        tn.drop_tags(t for t in tn.tags if '__VARIABLE' in t)
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
