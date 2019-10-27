"""Support for optimizing tensor networks using ``autograd`` or ``jax`` to
automatically derive gradients for input to scipy.
"""
import re
import functools
import importlib

import tqdm
import numpy as np

from .tensor_core import contract_backend
from .array_ops import iscomplex
from ..core import qarray


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


def parse_network_to_jax(tn, constant_tags):
    tn_jax = tn.copy()
    variables = []

    variable_tag = "__VARIABLE{}__"

    for t in tn_jax:

        if isinstance(t.data, qarray):
            t.modify(data=np.asarray(t.data))

        # check if tensor has any of the constant tags
        if t.tags & constant_tags:
            continue

        # append the raw data but mark the corresponding tensor for reinsertion
        variables.append(t.data)
        t.add_tag(variable_tag.format(len(variables) - 1))

    return tn_jax, variables


variable_finder = re.compile('__VARIABLE(\d+)__')


def inject_(arrays, tn):
    for t in tn:
        for tag in t.tags:
            match = variable_finder.match(tag)
            if match is not None:
                i = int(match.groups(1)[0])
                break

        t.modify(data=arrays[i])


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
        autograd_backend='AUTO',
        # the rest are ignored for compat
        learning_rate=None,
        learning_decay_steps=None,
        learning_decay_rate=None,
    ):
        self.progbar = progbar
        self.optimizer = optimizer
        self.constant_tags = (set() if constant_tags is None
                              else set(constant_tags))
        if autograd_backend.upper() == 'AUTO':
            if importlib.util.find_spec("jax") is not None:
                autograd_backend = 'jax'
            else:
                autograd_backend = 'autograd'

        self.autograd_backend = autograd_backend

        # use identity if no nomalization required
        if norm_fn is None:
            def norm_fn(x):
                return x

        self.norm_fn = norm_fn

        self.loss_kwargs = {} if loss_kwargs is None else dict(loss_kwargs)
        self.loss_target = loss_target
        self.loss_constants = ({} if loss_constants is None else
                               dict(loss_constants))

        self.loss_fn = functools.partial(
            loss_fn, **self.loss_constants, **self.loss_kwargs
        )
        self.loss = np.inf
        self.loss_best = np.inf
        self._n = 0

        self.tn_opt, self.variables = parse_network_to_jax(
            tn, self.constant_tags
        )

        self.v = Vectorizer(self.variables)

        def func(arrays):
            # need to set backend explicitly as mixing with numpy arrays
            with contract_backend(self.autograd_backend):
                inject_(arrays, self.tn_opt)
                return self.loss_fn(self.norm_fn(self.tn_opt))

        if self.autograd_backend == 'jax':
            import jax
            self.func_jit = jax.jit(func)
            self.grad_jit = jax.jit(jax.grad(func))
        elif self.autograd_backend == 'autograd':
            import autograd
            self.func_jit = func
            self.grad_jit = autograd.grad(func)

        def vectorized_func(x):
            arrays = self.v.unpack(x, conj=True)
            jax_result = self.func_jit(arrays)
            self.loss = np.asarray(jax_result)
            self._n += 1
            return self.loss

        def vectorized_grad(x):
            arrays = self.v.unpack(x, conj=True)
            jax_grads = self.grad_jit(arrays)
            vec_grad = self.v.pack(jax_grads, 'grad')
            return vec_grad

        self.vectorized_func = vectorized_func
        self.vectorized_grad = vectorized_grad

    @property
    def nevals(self):
        """The number of gradient evaluations.
        """
        return self._n

    def inject_res_vector_and_return_tn(self):
        inject_(self.v.unpack(self.res.x, conj=True), self.tn_opt)
        self.v.vector[:] = self.res.x
        return self.norm_fn(self.tn_opt)

    def optimize(self, n, tol=None, **options):
        from scipy.optimize import minimize

        try:
            pbar = tqdm.tqdm(total=n, disable=not self.progbar)

            def callback(_):
                pbar.update()
                pbar.set_description(f"{self.loss}")

            self.res = minimize(
                fun=self.vectorized_func,
                jac=self.vectorized_grad,
                x0=self.v.vector,
                callback=callback,
                tol=tol,
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
                func=self.vectorized_func,
                x0=self.v.vector,
                niter=nhop,
                minimizer_kwargs=dict(
                    jac=self.vectorized_grad,
                    method=self.optimizer,
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
