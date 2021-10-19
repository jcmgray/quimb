"""Support for optimizing tensor networks using automatic differentiation to
automatically derive gradients for input to scipy optimizers.
"""
import re
import warnings
import functools
import importlib
from collections.abc import Iterable

import tqdm
import numpy as np
from autoray import to_numpy, astype, get_dtype_name

from .tensor_core import (
    contract_backend,
    Tensor,
    TensorNetwork,
    PTensor,
    tags_to_oset,
)
from .array_ops import iscomplex
from ..core import qarray
from ..utils import valmap, ensure_dict

if importlib.util.find_spec("jax") is not None:
    _DEFAULT_BACKEND = 'jax'
elif importlib.util.find_spec("tensorflow") is not None:
    _DEFAULT_BACKEND = 'tensorflow'
elif importlib.util.find_spec("torch") is not None:
    _DEFAULT_BACKEND = 'torch'
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

    Parameters
    ----------
    array : sequence of array
        The set of arrays to map into a single real vector.
    """

    def __init__(self, arrays):
        self.shapes = [x.shape for x in arrays]
        self.iscomplexes = [iscomplex(x) for x in arrays]
        self.dtypes = [get_dtype_name(x) for x in arrays]
        self.sizes = [np.prod(s) for s in self.shapes]
        self.d = sum(
            (1 + int(cmplx)) * size
            for size, cmplx in zip(self.sizes, self.iscomplexes)
        )
        self.pack(arrays)

    def pack(self, arrays, name='vector'):
        """Take ``arrays`` and pack their values into attribute `.{name}`, by
        default `.vector`.
        """

        # scipy's optimization routines require real, double data
        if not hasattr(self, name):
            setattr(self, name, np.empty(self.d, 'float64'))
        x = getattr(self, name)

        i = 0
        for array, size, cmplx in zip(arrays, self.sizes, self.iscomplexes):

            if not isinstance(array, np.ndarray):
                array = to_numpy(array)

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

            if get_dtype_name(array) != dtype:
                array = astype(array, dtype)

            arrays.append(array)

        return arrays


_VARIABLE_TAG = "__VARIABLE{}__"
variable_finder = re.compile(r'__VARIABLE(\d+)__')


def _get_tensor_data(t):
    """Simple function to extract tensor data.
    """
    if isinstance(t, PTensor):
        data = t.params
    else:
        data = t.data

    # jax doesn't like numpy.ndarray subclasses...
    if isinstance(data, qarray):
        data = data.A

    return data


def _parse_opt_in(tn, tags, shared_tags, to_constant):
    """Parse a tensor network where tensors are assumed to be constant unless
    tagged.
    """
    tn_ag = tn.copy()
    variables = []

    # tags where each individual tensor should get a separate variable
    individual_tags = tags - shared_tags

    # handle tagged tensors that are not shared
    for t in tn_ag.select_tensors(individual_tags, 'any'):
        # append the raw data but mark the corresponding tensor
        # for reinsertion
        data = _get_tensor_data(t)
        variables.append(data)
        t.add_tag(_VARIABLE_TAG.format(len(variables) - 1))

    # handle shared tags
    for tag in shared_tags:

        var_name = _VARIABLE_TAG.format(len(variables))
        test_data = None

        for t in tn_ag.select_tensors(tag):
            data = _get_tensor_data(t)

            # detect that this tensor is already variable tagged and skip
            # if it is
            if any(variable_finder.match(tag) for tag in t.tags):
                warnings.warn('TNOptimizer warning, tensor tagged with'
                              ' multiple `tags` or `shared_tags`.')
                continue

            if test_data is None:
                # create variable and store data
                variables.append(data)
                test_data = data
            else:
                # check that the shape of the variable's data matches the
                # data of this new tensor
                if test_data.shape != data.shape:
                    raise ValueError('TNOptimizer error, a `shared_tags` tag '
                                     'covers tensors with different numbers of'
                                     ' params.')

            # mark the corresponding tensor for reinsertion
            t.add_tag(var_name)

    # iterate over tensors which *don't* have any of the given tags
    for t in tn_ag.select_tensors(tags, which='!any'):
        t.modify(apply=to_constant)

    return tn_ag, variables


def _parse_opt_out(tn, constant_tags, to_constant,):
    """Parse a tensor network where tensors are assumed to be variables unless
    tagged.
    """
    tn_ag = tn.copy()
    variables = []

    for t in tn_ag:

        if t.tags & constant_tags:
            t.modify(apply=to_constant)
            continue

        # append the raw data but mark the corresponding tensor
        # for reinsertion
        data = _get_tensor_data(t)
        variables.append(data)
        t.add_tag(_VARIABLE_TAG.format(len(variables) - 1))

    return tn_ag, variables


def parse_network_to_backend(
    tn,
    to_constant,
    tags=None,
    shared_tags=None,
    constant_tags=None,
):
    """
    Parse tensor network to:

        - identify the dimension of the optimisation space and the initial
          point of the optimisation from the current values in the tensor
          network,
        - add variable tags to individual tensors so that optimisation vector
          values can be efficiently reinserted into the tensor network.

    There are two different modes:

        - 'opt in' : `tags` (and optionally `shared_tags`) are specified and
          only these tensor tags will be optimised over. In this case
          `constant_tags` is ignored if it is passed,
        - 'opt out' : `tags` is not specified. In this case all tensors will be
          optimised over, unless they have one of `constant_tags` tags.

    Parameters
    ----------
    tn : TensorNetwork
        The initial tensor network to parse.
    to_constant : Callable
        Function that fixes a tensor as constant.
    tags : str, or sequence of str, optional
        Set of opt-in tags to optimise.
    shared_tags : str, or sequence of str, optional
        Subset of opt-in tags to joint optimise i.e. all tensors with tag s in
        shared_tags will correspond to the same optimisation variables.
    constant_tags : str, or sequence of str, optional
        Set of opt-out tags if `tags` not passed.

    Returns
    -------
    tn_ag : TensorNetwork
        Tensor network tagged for reinsertion of optimisation variable values.
    variables : list
        List of variables extracted from ``tn``.
    """
    tags = tags_to_oset(tags)
    shared_tags = tags_to_oset(shared_tags)
    constant_tags = tags_to_oset(constant_tags)

    if tags | shared_tags:
        # opt_in
        if not (tags & shared_tags) == shared_tags:
            tags = tags | shared_tags
            warnings.warn('TNOptimizer warning, some `shared_tags` are missing'
                          ' from `tags`. Automatically adding these missing'
                          ' `shared_tags` to `tags`.')
        if constant_tags:
            warnings.warn('TNOptimizer warning, if `tags` or `shared_tags` are'
                          ' specified then `constant_tags` is ignored - '
                          'consider instead untagging those tensors.')
        return _parse_opt_in(tn, tags, shared_tags, to_constant, )

    # opt-out
    return _parse_opt_out(tn, constant_tags, to_constant, )


def constant_t(t, to_constant):
    ag_t = t.copy()
    ag_t.modify(apply=to_constant)
    return ag_t


def constant_tn(tn, to_constant):
    """Convert a tensor network's arrays to constants.
    """
    ag_tn = tn.copy()
    ag_tn.apply_to_arrays(to_constant)
    return ag_tn


@functools.lru_cache(1)
def get_autograd():
    import autograd
    return autograd


class AutoGradHandler:

    def __init__(self, device='cpu'):
        if device != 'cpu':
            raise ValueError("`autograd` currently is only "
                             "backed by cpu, numpy arrays.")

    def to_variable(self, x):
        return np.asarray(x)

    def to_constant(self, x):
        return np.asarray(x)

    def setup_fn(self, fn):
        autograd = get_autograd()
        self._backend_fn = fn
        self._value_and_grad = autograd.value_and_grad(fn)
        self._hvp = autograd.hessian_vector_product(fn)

    def value(self, arrays):
        return self._backend_fn(arrays)

    def value_and_grad(self, arrays):
        loss, grads = self._value_and_grad(arrays)
        return loss, [x.conj() for x in grads]


@functools.lru_cache(1)
def get_jax():
    import jax
    return jax


class JaxHandler:

    def __init__(self, jit_fn=True, device=None):
        self.jit_fn = jit_fn
        self.device = device

    def to_variable(self, x):
        jax = get_jax()
        return jax.numpy.asarray(x)

    def to_constant(self, x):
        jax = get_jax()
        return jax.numpy.asarray(x)

    def setup_fn(self, fn):
        jax = get_jax()
        if self.jit_fn:
            self._backend_fn = jax.jit(fn, backend=self.device)
            self._value_and_grad = jax.jit(
                jax.value_and_grad(fn), backend=self.device)
        else:
            self._backend_fn = fn
            self._value_and_grad = jax.value_and_grad(fn)

        self._setup_hessp(fn)

    def _setup_hessp(self, fn):
        jax = get_jax()

        def hvp(primals, tangents):
            return jax.jvp(jax.grad(fn), (primals,), (tangents,))[1]

        if self.jit_fn:
            hvp = jax.jit(hvp, device=self.device)

        self._hvp = hvp

    def value(self, arrays):
        jax_arrays = tuple(map(self.to_constant, arrays))
        return to_numpy(self._backend_fn(jax_arrays))

    def value_and_grad(self, arrays):
        loss, grads = self._value_and_grad(arrays)
        return loss, [to_numpy(x.conj()) for x in grads]

    def hessp(self, primals, tangents):
        jax_arrays = self._hvp(primals, tangents)
        return tuple(map(to_numpy, jax_arrays))


@functools.lru_cache(1)
def get_tensorflow():
    import tensorflow
    return tensorflow


class TensorFlowHandler:

    def __init__(
        self,
        jit_fn=False,
        autograph=False,
        experimental_compile=False,
        device=None,
    ):
        self.jit_fn = jit_fn
        self.autograph = autograph
        self.experimental_compile = experimental_compile
        self.device = device

    def to_variable(self, x):
        tf = get_tensorflow()
        if self.device is None:
            return tf.Variable(x)
        with tf.device(self.device):
            return tf.Variable(x)

    def to_constant(self, x):
        tf = get_tensorflow()
        if self.device is None:
            return tf.constant(x)
        with tf.device(self.device):
            return tf.constant(x)

    def setup_fn(self, fn):
        tf = get_tensorflow()
        if self.jit_fn:
            self._backend_fn = tf.function(
                fn,
                autograph=self.autograph,
                experimental_compile=self.experimental_compile)
        else:
            self._backend_fn = fn

    def value(self, arrays):
        tf_arrays = tuple(map(self.to_constant, arrays))
        return to_numpy(self._backend_fn(tf_arrays))

    def value_and_grad(self, arrays):
        tf = get_tensorflow()
        variables = [self.to_variable(x) for x in arrays]

        with tf.GradientTape() as t:
            result = self._backend_fn(variables)
        tf_grads = t.gradient(result, variables)

        grads = [
            # unused variables return as None
            # NB note different convention for conjugation (i.e. none)
            np.zeros_like(arrays[i]) if g is None else to_numpy(g)
            for i, g in enumerate(tf_grads)
        ]
        loss = to_numpy(result)
        return loss, grads


@functools.lru_cache(1)
def get_torch():
    import torch
    return torch


class TorchHandler:

    def __init__(self, jit_fn=False, device=None):
        torch = get_torch()
        self.jit_fn = jit_fn
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device

    def to_variable(self, x):
        torch = get_torch()
        return torch.tensor(x).to(self.device).requires_grad_()

    def to_constant(self, x):
        torch = get_torch()
        return torch.tensor(x).to(self.device)

    def setup_fn(self, fn):
        self._fn = fn
        self._backend_fn = None

    def _setup_backend_fn(self, arrays):
        torch = get_torch()
        if self.jit_fn:
            example_inputs = (tuple(map(self.to_constant, arrays)),)
            self._backend_fn = torch.jit.trace(
                self._fn, example_inputs=example_inputs)
        else:
            self._backend_fn = self._fn

    def value(self, arrays):
        if self._backend_fn is None:
            self._setup_backend_fn(arrays)
        torch_arrays = tuple(map(self.to_constant, arrays))
        return to_numpy(self._backend_fn(torch_arrays))

    def value_and_grad(self, arrays):
        torch = get_torch()

        if self._backend_fn is None:
            self._setup_backend_fn(arrays)

        variables = [self.to_variable(x) for x in arrays]
        result = self._backend_fn(variables)
        torch_grads = torch.autograd.grad(result, variables, allow_unused=True)
        grads = [
            # unused variables return as None
            np.zeros_like(arrays[i]) if g is None else to_numpy(g).conj()
            for i, g in enumerate(torch_grads)
        ]
        loss = to_numpy(result)
        return loss, grads


_BACKEND_HANDLERS = {
    'numpy': AutoGradHandler,
    'autograd': AutoGradHandler,
    'jax': JaxHandler,
    'tensorflow': TensorFlowHandler,
    'torch': TorchHandler,
}


class MultiLossHandler:

    def __init__(self, autodiff_backend, executor=None, **backend_opts):
        self.autodiff_backend = autodiff_backend
        self.backend_opts = backend_opts
        self.executor = executor

        # start just with one, as we don't don't know how many functions yet
        h0 = _BACKEND_HANDLERS[autodiff_backend](**backend_opts)
        self.handlers = [h0]
        # ... but we do need access to `to_constant`
        self.to_constant = h0.to_constant

    def setup_fn(self, funcs):
        fn0, *fns = funcs
        self.handlers[0].setup_fn(fn0)
        for fn in fns:
            h = _BACKEND_HANDLERS[self.autodiff_backend](**self.backend_opts)
            h.setup_fn(fn)
            self.handlers.append(h)

    def _value_seq(self, arrays):
        return sum(h.value(arrays) for h in self.handlers)

    def _value_par_seq(self, arrays):
        futures = [self.executor.submit(h.value, arrays)
                   for h in self.handlers]
        return sum(f.result() for f in futures)

    def value(self, arrays):
        if self.executor is not None:
            return self._value_par(arrays)
        return self._value_seq(arrays)

    def _value_and_grad_seq(self, arrays):
        h0, *hs = self.handlers
        loss, grads = h0.value_and_grad(arrays)
        # need to make arrays writeable for efficient inplace sum
        grads = list(map(np.array, grads))
        for h in hs:
            loss_i, grads_i = h.value_and_grad(arrays)
            loss += loss_i
            for i, g_i in enumerate(grads_i):
                grads[i] += g_i
        return loss, grads

    def _value_and_grad_par(self, arrays):
        futures = [self.executor.submit(h.value_and_grad, arrays)
                   for h in self.handlers]
        results = (f.result() for f in futures)

        # get first result
        loss, grads = next(results)
        grads = list(map(np.array, grads))

        # process remaining results
        for loss_i, grads_i in results:
            loss += loss_i
            for i, g_i in enumerate(grads_i):
                grads[i] += g_i

        return loss, grads

    def value_and_grad(self, arrays):
        if self.executor is not None:
            return self._value_and_grad_par(arrays)
        return self._value_and_grad_seq(arrays)


def inject_(arrays, tn):
    for t in tn:
        for tag in t.tags:
            match = variable_finder.match(tag)
            if match is not None:
                i = int(match.groups(1)[0])

                if isinstance(t, PTensor):
                    t.params = arrays[i]
                else:
                    t.modify(data=arrays[i], left_inds=t.left_inds)

                break


class SGD:
    """Stateful ``scipy.optimize.minimize`` compatible implementation of
    stochastic gradient descent with momentum.

    Adapted from ``autograd/misc/optimizers.py``.
    """

    def __init__(self):
        from scipy.optimize import OptimizeResult
        self.OptimizeResult = OptimizeResult
        self._i = 0
        self._velocity = None

    def get_velocity(self, x):
        if self._velocity is None:
            self._velocity = np.zeros_like(x)
        return self._velocity

    def __call__(self, fun, x0, jac, args=(), learning_rate=0.1, mass=0.9,
                 maxiter=1000, callback=None, bounds=None, **kwargs):

        x = x0
        velocity = self.get_velocity(x)

        for _ in range(maxiter):
            self._i += 1

            g = jac(x)

            if callback and callback(x):
                break

            velocity = mass * velocity - (1.0 - mass) * g
            x = x + learning_rate * velocity

            if bounds is not None:
                x = np.clip(x, bounds[:, 0], bounds[:, 1])

        # save for restart
        self._velocity = velocity

        return self.OptimizeResult(
            x=x, fun=fun(x), jac=g, nit=self._i, nfev=self._i, success=True)


class RMSPROP:
    """Stateful ``scipy.optimize.minimize`` compatible implementation of
    root mean squared prop: See Adagrad paper for details.

    Adapted from ``autograd/misc/optimizers.py``.
    """

    def __init__(self):
        from scipy.optimize import OptimizeResult
        self.OptimizeResult = OptimizeResult
        self._i = 0
        self._avg_sq_grad = None

    def get_avg_sq_grad(self, x):
        if self._avg_sq_grad is None:
            self._avg_sq_grad = np.ones_like(x)
        return self._avg_sq_grad

    def __call__(self, fun, x0, jac, args=(), learning_rate=0.1, gamma=0.9,
                 eps=1e-8, maxiter=1000, callback=None, bounds=None, **kwargs):
        x = x0
        avg_sq_grad = self.get_avg_sq_grad(x)

        for _ in range(maxiter):
            self._i += 1

            g = jac(x)

            if callback and callback(x):
                break

            avg_sq_grad = avg_sq_grad * gamma + g**2 * (1 - gamma)
            x = x - learning_rate * g / (np.sqrt(avg_sq_grad) + eps)

            if bounds is not None:
                x = np.clip(x, bounds[:, 0], bounds[:, 1])

        # save for restart
        self._avg_sq_grad = avg_sq_grad

        return self.OptimizeResult(
            x=x, fun=fun(x), jac=g, nit=self._i, nfev=self._i, success=True)


class ADAM:
    """Stateful ``scipy.optimize.minimize`` compatible implementation of
    ADAM - http://arxiv.org/pdf/1412.6980.pdf.

    Adapted from ``autograd/misc/optimizers.py``.
    """

    def __init__(self):
        from scipy.optimize import OptimizeResult
        self.OptimizeResult = OptimizeResult
        self._i = 0
        self._m = None
        self._v = None

    def get_m(self, x):
        if self._m is None:
            self._m = np.zeros_like(x)
        return self._m

    def get_v(self, x):
        if self._v is None:
            self._v = np.zeros_like(x)
        return self._v

    def __call__(self, fun, x0, jac, args=(), learning_rate=0.001, beta1=0.9,
                 beta2=0.999, eps=1e-8, maxiter=1000, callback=None,
                 bounds=None, **kwargs):
        x = x0
        m = self.get_m(x)
        v = self.get_v(x)

        for _ in range(maxiter):
            self._i += 1

            g = jac(x)

            if callback and callback(x):
                break

            m = (1 - beta1) * g + beta1 * m  # first  moment estimate.
            v = (1 - beta2) * (g**2) + beta2 * v  # second moment estimate.
            mhat = m / (1 - beta1**(self._i))  # bias correction.
            vhat = v / (1 - beta2**(self._i))
            x = x - learning_rate * mhat / (np.sqrt(vhat) + eps)

            if bounds is not None:
                x = np.clip(x, bounds[:, 0], bounds[:, 1])

        # save for restart
        self._m = m
        self._v = v

        return self.OptimizeResult(
            x=x, fun=fun(x), jac=g, nit=self._i, nfev=self._i, success=True)


class NADAM:
    """Stateful ``scipy.optimize.minimize`` compatible implementation of
    NADAM - [Dozat - http://cs229.stanford.edu/proj2015/054_report.pdf].

    Adapted from ``autograd/misc/optimizers.py``.
    """

    def __init__(self):
        from scipy.optimize import OptimizeResult
        self.OptimizeResult = OptimizeResult
        self._i = 0
        self._m = None
        self._v = None
        self._mus = None

    def get_m(self, x):
        if self._m is None:
            self._m = np.zeros_like(x)
        return self._m

    def get_v(self, x):
        if self._v is None:
            self._v = np.zeros_like(x)
        return self._v

    def get_mus(self, beta1):
        if self._mus is None:
            self._mus = [1, beta1 * (1 - 0.5 * 0.96**0.004)]
        return self._mus

    def __call__(self, fun, x0, jac, args=(), learning_rate=0.001, beta1=0.9,
                 beta2=0.999, eps=1e-8, maxiter=1000, callback=None,
                 bounds=None, **kwargs):
        x = x0
        m = self.get_m(x)
        v = self.get_v(x)
        mus = self.get_mus(beta1)

        for _ in range(maxiter):
            self._i += 1

            # this is ``mu[t + 1]`` -> already computed ``mu[t]``
            self._mus.append(beta1 * (1 - 0.5 * 0.96**(0.004 * (self._i + 1))))

            g = jac(x)

            if callback and callback(x):
                break

            gd = g / (1 - np.prod(self._mus[:-1]))
            m = beta1 * m + (1 - beta1) * g
            md = m / (1 - np.prod(self._mus))
            v = beta2 * v + (1 - beta2) * g**2
            vd = v / (1 - beta2**self._i)
            mhat = (1 - self._mus[self._i]) * gd + self._mus[self._i + 1] * md

            x = x - learning_rate * mhat / (np.sqrt(vd) + eps)

            if bounds is not None:
                x = np.clip(x, bounds[:, 0], bounds[:, 1])

        # save for restart
        self._m = m
        self._v = v
        self._mus = mus

        return self.OptimizeResult(
            x=x, fun=fun(x), jac=g, nit=self._i, nfev=self._i, success=True)


_STOC_GRAD_METHODS = {
    'sgd': SGD,
    'rmsprop': RMSPROP,
    'adam': ADAM,
    'nadam': NADAM,
}


def parse_constant_arg(arg, to_constant):
    # check if tensor network supplied
    if isinstance(arg, TensorNetwork):
        # convert it to constant TN
        return constant_tn(arg, to_constant)

    if isinstance(arg, Tensor):
        return constant_t(arg, to_constant)

    if isinstance(arg, dict):
        return valmap(to_constant, arg)

    if isinstance(arg, list):
        return list(map(to_constant, arg))

    if isinstance(arg, tuple):
        return tuple(map(to_constant, arg))

    # assume ``arg`` is a raw array
    return to_constant(arg)


class MakeArrayFn:
    """Class wrapper so picklable.
    """

    __name__ = 'MakeArrayFn'

    def __init__(self, tn_opt, loss_fn, norm_fn, autodiff_backend):
        self.tn_opt = tn_opt
        self.loss_fn = loss_fn
        self.norm_fn = norm_fn
        self.autodiff_backend = autodiff_backend

    def __call__(self, arrays):
        # copy the TN so norm and loss functions can modify in place
        # XXX: make optional for efficiency?
        tn_compute = self.tn_opt.copy()
        inject_(arrays, tn_compute)

        # set backend explicitly as maybe mixing with numpy arrays
        with contract_backend(self.autodiff_backend):
            return self.loss_fn(self.norm_fn(tn_compute))


def identity_fn(x):
    return x


class TNOptimizer:
    """Globally optimize tensors within a tensor network with respect to any
    loss function via automatic differentiation. If parametrized tensors are
    used, optimize the parameters rather than the raw arrays.

    Parameters
    ----------
    tn : TensorNetwork
        The core tensor network structure within which to optimize tensors.
    loss_fn : callable or sequence of callable
        The function that takes ``tn`` (as well as ``loss_constants`` and
        ``loss_kwargs``) and returns a single real 'loss' to be minimized.
        For Hamiltonians which can be represented as a sum over terms, an
        iterable collection of terms (e.g. list) can be given instead. In that
        case each term is evaluated independently and the sum taken as loss_fn.
        This can reduce the total memory requirements or allow for
        parallelization (see ``executor``).
    norm_fn : callable, optional
        A function to call before ``loss_fn`` that prepares or 'normalizes' the
        raw tensor network in some way.
    loss_constants : dict, optional
        Extra tensor networks, tensors, dicts/list/tuples of arrays, or arrays
        which will be supplied to ``loss_fn`` but also converted to the correct
        backend array type.
    loss_kwargs : dict, optional
        Extra options to supply to ``loss_fn`` (unlike ``loss_constants`` these
        are assumed to be simple options that don't need conversion).
    tags : str, or sequence of str, optional
        If supplied, only optimize tensors with any of these tags.
    shared_tags : str, or sequence of str, optional
        If supplied, each tag in ``shared_tags`` corresponds to a group of
        tensors to be optimized together.
    constant_tags : str, or sequence of str, optional
        If supplied, skip optimizing tensors with any of these tags. This
        'opt-out' mode is overridden if either ``tags`` or ``shared_tags`` is
        supplied.
    loss_target : float, optional
        Stop optimizing once this loss value is reached.
    optimizer : str, optional
        Which ``scipy.optimize.minimize`` optimizer to use (the ``'method'``
        kwarg of that function). In addition, ``quimb`` implements a few custom
        optimizers compatible with this interface that you can reference by
        name - ``{'adam', 'nadam', 'rmsprop', 'sgd'}``.
    executor : None or Executor, optional
        To be used with term-by-term Hamiltonians. If supplied, this executor
        is used  to parallelize the evaluation. Otherwise each term is
        evaluated in sequence. It should implement the basic
        concurrent.futures (PEP 3148) interface.
    progbar : bool, optional
        Whether to show live progress.
    bounds : None or (float, float), optional
        Constrain the optimized tensor entries within this range (if the scipy
        optimizer supports it).
    autodiff_backend : {'jax', 'autograd', 'tensorflow', 'torch'}, optional
        Which backend library to use to perform the automatic differentation
        (and computation).
    backend_opts
        Supplied to the backend function compiler and array handler. For
        example ``jit_fn=True`` or ``device='cpu'`` .
    """

    def __init__(
        self,
        tn,
        loss_fn,
        norm_fn=None,
        loss_constants=None,
        loss_kwargs=None,
        tags=None,
        shared_tags=None,
        constant_tags=None,
        loss_target=None,
        optimizer='L-BFGS-B',
        progbar=True,
        bounds=None,
        autodiff_backend='AUTO',
        executor=None,
        **backend_opts
    ):
        self.progbar = progbar
        self.tags = tags
        self.shared_tags = shared_tags
        self.constant_tags = constant_tags

        if autodiff_backend.upper() == 'AUTO':
            autodiff_backend = _DEFAULT_BACKEND
        self._autodiff_backend = autodiff_backend
        self._multiloss = isinstance(loss_fn, Iterable)

        # the object that handles converting to backend + computing gradient
        if self._multiloss:
            # special meta-handler if loss function is sequence to sum
            backend_opts['executor'] = executor
            self.handler = MultiLossHandler(autodiff_backend, **backend_opts)
        else:
            self.handler = _BACKEND_HANDLERS[autodiff_backend](**backend_opts)

        # use identity if no nomalization required
        if norm_fn is None:
            norm_fn = identity_fn
        self.norm_fn = norm_fn

        self.reset(tn, loss_target=loss_target)

        # convert constant arrays ahead of time to correct backend
        self.loss_constants = {
            k: parse_constant_arg(v, self.handler.to_constant)
            for k, v in ensure_dict(loss_constants).items()
        }
        self.loss_kwargs = ensure_dict(loss_kwargs)
        kws = {**self.loss_constants, **self.loss_kwargs}

        # inject these constant options to the loss function(s)
        if self._multiloss:
            # loss is a sum of independent terms
            self.loss_fn = [functools.partial(fn, **kws) for fn in loss_fn]
        else:
            # loss is all in one
            self.loss_fn = functools.partial(loss_fn, **kws)

        # first we wrap the function to convert from array args to TN arg
        #     (i.e. to autodiff library compatible form)
        if self._multiloss:
            array_fn = [MakeArrayFn(self._tn_opt, fn, self.norm_fn,
                                    autodiff_backend) for fn in self.loss_fn]
        else:
            array_fn = MakeArrayFn(
                self._tn_opt, self.loss_fn, self.norm_fn, autodiff_backend)


        # then we pass it to the handler which generates a function that
        # computes both the value and gradients (still in array form)
        self.handler.setup_fn(array_fn)

        # options to do with the minimizer
        self.bounds = bounds
        self.optimizer = optimizer

    def _set_tn(self, tn):
        # work out which tensors to optimize and get the underlying data
        self._tn_opt, variables = parse_network_to_backend(
            tn,
            tags=self.tags,
            shared_tags=self.shared_tags,
            constant_tags=self.constant_tags,
            to_constant=self.handler.to_constant
        )
        # handles storing and packing / unpacking many arrays as a vector
        self.vectorizer = Vectorizer(variables)

    def _reset_tracking_info(self, loss_target=None):
        # tracking info
        self.loss = float('inf')
        self.loss_best = float('inf')
        self.loss_target = loss_target
        self.losses = []
        self._n = 0
        self._pbar = None

    def reset(self, tn=None, clear_info=True, loss_target=None):
        """Reset this optimizer without losing the compiled loss and gradient
        functions.

        Parameters
        ----------
        tn : TensorNetwork, optional
            Set this tensor network as the current state of the optimizer, it
            must exactly match the original tensor network.
        clear_info : bool, optional
            Clear the tracked losses and iterations.
        """
        if tn is not None:
            self._set_tn(tn)
        if clear_info:
            self._reset_tracking_info(loss_target=loss_target)

    def _maybe_init_pbar(self, n):
        if self.progbar:
            self._pbar = tqdm.tqdm(total=n)

    def _maybe_update_pbar(self):
        if self._pbar is not None:
            self._pbar.update()
            self.loss_best = min(self.loss_best, self.loss)
            msg = f"{self.loss:+.12f} [best: {self.loss_best:+.12f}] "
            self._pbar.set_description(msg)

    def _maybe_close_pbar(self):
        if self._pbar is not None:
            self._pbar.close()
            self._pbar = None

    def _check_loss_target(self):
        if (self.loss_target is not None) and (self.loss <= self.loss_target):
            # for scipy terminating optimizer with callback doesn't work
            raise KeyboardInterrupt

    def vectorized_value(self, x):
        """The value of the loss function at vector ``x``.
        """
        self.vectorizer.vector[:] = x
        arrays = self.vectorizer.unpack()
        self.loss = self.handler.value(arrays).item()
        self.losses.append(self.loss)
        self._n += 1
        self._maybe_update_pbar()
        self._check_loss_target()
        return self.loss

    def vectorized_value_and_grad(self, x):
        """The value and gradient of the loss function at vector ``x``.
        """
        self.vectorizer.vector[:] = x
        arrays = self.vectorizer.unpack()
        result, grads = self.handler.value_and_grad(arrays)
        self._n += 1
        self.loss = result.item()
        self.losses.append(self.loss)
        vec_grad = self.vectorizer.pack(grads, 'grad')
        self._maybe_update_pbar()
        self._check_loss_target()
        return self.loss, vec_grad

    def vectorized_hessp(self, x, p):
        """The action of the hessian at point ``x`` on vector ``p``.
        """
        primals = self.vectorizer.unpack(x)
        tangents = self.vectorizer.unpack(p)
        hp_arrays = self.handler.hessp(primals, tangents)
        self._n += 1
        self.losses.append(self.loss)
        self._maybe_update_pbar()
        return self.vectorizer.pack(hp_arrays, 'hp')

    def __repr__(self):
        return (f"<TNOptimizer(d={self.d}, "
                f"backend={self._autodiff_backend})>")

    @property
    def d(self):
        return int(self.vectorizer.d)

    @property
    def nevals(self):
        """The number of gradient evaluations.
        """
        return self._n

    @property
    def optimizer(self):
        """The underlying optimizer that works with the vectorized functions.
        """
        return self._optimizer

    @optimizer.setter
    def optimizer(self, x):
        self._optimizer = x
        if self.optimizer in _STOC_GRAD_METHODS:
            self._method = _STOC_GRAD_METHODS[self.optimizer]()
        else:
            self._method = self.optimizer

    @property
    def bounds(self):
        return self._bounds

    @bounds.setter
    def bounds(self, x):
        if x is not None:
            self._bounds = np.array((x,) * self.vectorizer.d)
        else:
            self._bounds = None

    def get_tn_opt(self):
        """Extract the optimized tensor network, this is a three part process:

            1. inject the current optimized vector into the target tensor
               network,
            2. run it through ``norm_fn``,
            3. drop any tags used to identify variables.

        Returns
        -------
        tn_opt : TensorNetwork
        """
        arrays = tuple(map(self.handler.to_constant, self.vectorizer.unpack()))
        inject_(arrays, self._tn_opt)
        tn = self.norm_fn(self._tn_opt.copy())
        tn.drop_tags(t for t in tn.tags if variable_finder.match(t))

        for t in tn:
            if isinstance(t, PTensor):
                t.params = to_numpy(t.params)
            else:
                t.modify(data=to_numpy(t.data), left_inds=t.left_inds)

        return tn

    def optimize(
        self,
        n,
        tol=None,
        jac=True,
        hessp=False,
        **options
    ):
        """Run the optimizer for ``n`` function evaluations, using
        :func:`scipy.optimize.minimize` as the driver for the vectorized
        computation. Supplying the gradient and hessian vector product is
        controlled by the ``jac`` and ``hessp`` options respectively.

        Parameters
        ----------
        n : int
            Notionally the maximum number of iterations for the optimizer, note
            that depending on the optimizer being used, this may correspond to
            number of function evaluations rather than just iterations.
        tol : None or float, optional
            Tolerance for convergence, note that various more specific
            tolerances can usually be supplied to ``options``, depending on
            the optimizer being used.
        jac : bool, optional
            Whether to supply the jacobian, i.e. gradient, of the loss
            function.
        hessp : bool, optional
            Whether to supply the hessian vector product of the loss function.
        options
            Supplied to :func:`scipy.optimize.minimize`.

        Returns
        -------
        tn_opt : TensorNetwork
        """
        from scipy.optimize import minimize

        if jac:
            fun = self.vectorized_value_and_grad
        else:
            fun = self.vectorized_value

        try:
            self._maybe_init_pbar(n)
            self.res = minimize(
                fun=fun,
                jac=jac,
                hessp=self.vectorized_hessp if hessp else None,
                x0=self.vectorizer.vector,
                tol=tol,
                bounds=self.bounds,
                method=self._method,
                options=dict(maxiter=n, **options),
            )
            self.vectorizer.vector[:] = self.res.x
        except KeyboardInterrupt:
            pass
        finally:
            self._maybe_close_pbar()

        return self.get_tn_opt()

    def optimize_basinhopping(
        self,
        n,
        nhop,
        temperature=1.0,
        jac=True,
        hessp=False,
        **options
    ):
        """Run the optimizer for using :func:`scipy.optimize.basinhopping`
        as the driver for the vectorized computation. This performs ``nhop``
        local optimization each with ``n`` iterations.

        Parameters
        ----------
        n : int
            Number of iterations per local optimization.
        nhop : int
            Number of local optimizations to hop between.
        temperature : float, optional
            H
        options
            Supplied to the inner :func:`scipy.optimize.minimize` call.

        Returns
        -------
        tn_opt : TensorNetwork
        """
        from scipy.optimize import basinhopping

        if jac:
            fun = self.vectorized_value_and_grad
        else:
            fun = self.vectorized_value

        try:
            self._maybe_init_pbar(n * nhop)
            self.res = basinhopping(
                func=fun,
                x0=self.vectorizer.vector,
                niter=nhop,
                minimizer_kwargs=dict(
                    jac=jac,
                    hessp=self.vectorized_hessp if hessp else None,
                    method=self._method,
                    bounds=self.bounds,
                    options=dict(maxiter=n, **options)
                ),
                T=temperature,
            )
            self.vectorizer.vector[:] = self.res.x

        except KeyboardInterrupt:
            pass
        finally:
            self._maybe_close_pbar()

        return self.get_tn_opt()

    def optimize_nlopt(
        self,
        n,
        ftol_rel=None,
        ftol_abs=None,
        xtol_rel=None,
        xtol_abs=None,
    ):
        """Run the optimizer for ``n`` function evaluations, using ``nlopt`` as
        the backend library to run the optimization. Whether the gradient is
        computed depends on which ``optimizer`` is selected, see valid options
        at  https://nlopt.readthedocs.io/en/latest/NLopt_Algorithms/.

        Parameters
        ----------
        n : int
            The maximum number of iterations for the optimizer.
        ftol_rel : float, optional
            Set relative tolerance on function value.
        ftol_abs : float, optional
            Set absolute tolerance on function value.
        xtol_rel : float, optional
            Set relative tolerance on optimization parameters.
        xtol_abs : float, optional
            Set absolute tolerances on optimization parameters.

        Returns
        -------
        tn_opt : TensorNetwork
        """
        import nlopt

        try:
            self._maybe_init_pbar(n)

            def f(x, grad):
                self.vectorizer.vector[:] = x
                arrays = self.vectorizer.unpack()
                if grad.size > 0:
                    result, grads = self.handler.value_and_grad(arrays)
                    grad[:] = self.vectorizer.pack(grads, 'grad')
                else:
                    result = self.handler.value(arrays)
                self._n += 1
                self.loss = result.item()
                self.losses.append(self.loss)
                self._maybe_update_pbar()
                return self.loss

            opt = nlopt.opt(getattr(nlopt, self.optimizer), self.d)
            opt.set_min_objective(f)
            opt.set_maxeval(n)

            if self.bounds is not None:
                opt.set_lower_bounds(self.bounds[:, 0])
                opt.set_upper_bounds(self.bounds[:, 1])

            if self.loss_target is not None:
                opt.set_stopval(self.loss_target)
            if ftol_rel is not None:
                opt.set_ftol_rel(ftol_rel)
            if ftol_abs is not None:
                opt.set_ftol_abs(ftol_abs)
            if xtol_rel is not None:
                opt.set_xtol_rel(xtol_rel)
            if xtol_abs is not None:
                opt.set_xtol_abs(xtol_abs)

            self.vectorizer.vector[:] = opt.optimize(self.vectorizer.vector)

        except (KeyboardInterrupt, RuntimeError):
            pass
        finally:
            self._maybe_close_pbar()

        return self.get_tn_opt()

    def optimize_ipopt(self, n, tol=None, **options):
        """Run the optimizer for ``n`` function evaluations, using ``ipopt`` as
        the backend library to run the optimization via the python package
        ``cyipopt``.

        Parameters
        ----------
        n : int
            The maximum number of iterations for the optimizer.

        Returns
        -------
        tn_opt : TensorNetwork
        """
        from cyipopt import minimize_ipopt

        try:
            self._maybe_init_pbar(n)
            self.res = minimize_ipopt(
                fun=self.vectorized_value_and_grad,
                jac=True,
                x0=self.vectorizer.vector,
                tol=tol,
                bounds=self.bounds,
                method=self._method,
                options=dict(maxiter=n, **options),
            )
            self.vectorizer.vector[:] = self.res.x
        except KeyboardInterrupt:
            pass
        finally:
            self._maybe_close_pbar()

        return self.get_tn_opt()

    def optimize_nevergrad(self, n):
        """Run the optimizer for ``n`` function evaluations, using
        ``nevergrad`` as the backend library to run the optimization. As the
        name suggests, the gradient is not required for this method.

        Parameters
        ----------
        n : int
            The maximum number of iterations for the optimizer.

        Returns
        -------
        tn_opt : TensorNetwork
        """
        import nevergrad as ng

        opt = getattr(ng.optimizers, self.optimizer)(
            parametrization=ng.p.Array(
                init=self.vectorizer.vector,
                lower=self.bounds[:, 0] if self.bounds is not None else None,
                upper=self.bounds[:, 1] if self.bounds is not None else None,
            ),
            budget=n
        )

        try:
            self._maybe_init_pbar(n)
            for _ in range(n):
                x = opt.ask()
                loss = self.vectorized_value(*x.args, **x.kwargs)
                opt.tell(x, loss)
                if self.loss_target is not None:
                    if self.loss < self.loss_target:
                        break

        except KeyboardInterrupt:
            pass
        finally:
            self._maybe_close_pbar()

        # recommendation = opt.minimize(self.vectorized_value)
        recommendation = opt.provide_recommendation()
        self.vectorizer.vector[:] = recommendation.value

        return self.get_tn_opt()

