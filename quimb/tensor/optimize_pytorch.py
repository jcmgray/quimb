import time
import functools

import tqdm

from . import array_ops
from .tensor_core import Tensor, TensorNetwork, utup_intersection, tags_to_utup


_TORCH_DEVICE = None


def _get_torch_and_device(dev=None):
    global _TORCH_DEVICE

    if dev is not None:
        assert dev in ['cpu', 'cuda']
        import torch
        _TORCH_DEVICE = torch, dev

    elif _TORCH_DEVICE is None:
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        _TORCH_DEVICE = torch, device

    return _TORCH_DEVICE


def variable(x):
    torch, device = _get_torch_and_device()
    return torch.tensor(x, requires_grad=True, device=device)


def constant(x):
    torch, device = _get_torch_and_device()
    return torch.tensor(x, requires_grad=False, device=device)


def constant_t(t):
    t_torch = t.copy()
    t_torch.modify(data=constant(t_torch.data))
    return t_torch


def constant_tn(tn):
    const_tn = tn.copy()
    const_tn.apply_to_arrays(constant)
    return const_tn


def parse_network_to_torch(tn, constant_tags):

    tn_torch = tn.copy()
    variables = []
    iscomplex = False

    for t in tn_torch:

        # check if tensor has any of the constant tags
        if utup_intersection((t.tags, constant_tags)):
            t.modify(data=constant(t.data))

        # treat re and im parts as separate variables
        elif array_ops.iscomplex(t.data):
            iscomplex = True
            raise TypeError("Torch does not yet support complex data types.")

        else:
            torch_data = variable(t.data)
            variables.append(torch_data)
            t.modify(data=torch_data)

    return tn_torch, variables, iscomplex


def abs_max_grad(tn):
    """Inspect the gradient of every tensor element in the TensorNetwork
    and return the gradient with the largest magnitude (useful for checking
    convergence)."""

    torch, device = _get_torch_and_device()
    views = []
    for t in tn:
        p = t.data
        if p.grad is None:
            view = p.new(p.numel()).zero_()
        elif p.grad.is_sparse:
            view = p.grad.to_dense().view(-1)
        else:
            view = p.grad.view(-1)
        views.append(view)
    flat_grad = torch.cat(views, 0)
    return flat_grad.abs().max()


class TNOptimizer:
    """Optimize a tensor network's arrays using pytorch.

    Parameters
    ----------
    tn : TensorNetwork
        The tensor network to optimize, every tensor in this will be treated
        as a variable.
    loss_fn : callable
        The target loss function to minimize. Should accept ``tn`` (backed by
        pytorch tensors) or the output of ``norm_fn``.
    norm_fn : callable, optional
        A function to normalize ``tn`` before being passed to ``loss_fn``.
        This offers very
        general support, so it can be used
        to enforce any constraints such as normalization or unitarity. The
        penalty of using this level of generality is that all operations
        that occur within ``norm_fn`` will be added to the computation graph,
        and thus differentiated during backpropagation. Note that due to the
        generality of this function, it does not actually keep the parameters
        under optimization within the normalized manifold, but instead just
        keeps the evaluation of loss_fn within the normalized manifold. This
        can lead to numerical instability in large TNs where the norm may
        explode after many iterations.
    norm_fn_scalar: callable, optional
        A function that accepts ``tn`` and computes the scalar ``x`` by which
        to multiply each tensor in ``tn`` in order to normalize it. If ``tn``
        is normalized by simple scalar multiplication of the tensors, it is
        recommended to use this function instead of ``norm_fn``. In this case,
        the computation of ``x`` and the rescaling of the tensor values can be
        performed outside the scope of the
        differentiable computation graph. This has two important advantages:
        1) It can lead to reductions in cpu time and memory usage when compared
        to ``norm_fn``. 2) It can actually modify the parameters under
        optimization to keep them normalized throughout the procedure,
        increasing numerical stability. If desired, both ``norm_fn`` and
        ``norm_fn_scalar`` can be used at the same time to enforce general
        constraints on ``tn`` and then normalize it with scalar multiplication.
        In this case, note that ``norm_fn_scalar`` is applied first, and then
        ``norm_fn``.
    loss_constants : dict_like, optional
        Extra constant arguments to supply to ``loss_fn`` and be converted to
        tensorflow constant tensors. Can be individual arrays or tensor
        networks.
    loss_kwargs : dict_like, optional
        Other kwargs to supply to ``loss_fn`` that are not arrays or tensor
        networks.
    constant_tags : sequence of str, optional
        Treat any tensors *within* ``tn`` with these tags as constant.
    optimizer : str, optional
        Which optimizer to use, default: ``'Adam'``.
        This should be an optimizer that can be found in the
        ``torch.optim`` submodule.
    learning_rate : float, optional
        The learning rate to apply to use the optimizer with. You can
        dynamically change this between ``optimize`` calls.
    loss_target : float, optional
        If supplied, stop optimizing once this loss is reached.
    tol_change : float, optional
        Convergence threshold on the function value. If the value of the loss
        function changes by less than ``tol_change`` between subsequent
        optimization steps, the optimization will terminate.
    tol_grad : float, optional
        Convergence threshold on the gradients. If the absolute value of the
        gradient with the largest magnitude is smaller than ``tol_grad``, the
        optimization will terminate.
    device : string, optional
        Whether to use ``'cpu'`` or ``'cuda'``, default: ``None``.
        If ``None``, then gpu (``'cuda'``) will be used if it can be
        detected, otherwise ``'cpu'`` will be used.
    progbar : bool, optional
        "True" show live progress of the optimization using a progress bar.
        "False" runs the optimization silently. For simple line-by-line
        printing of the progress, you can set this option to the string
        "simple".


    General usage examples can be seen in the documentation
    for ``tensor.optimize_tensorflow.TNOptimizer``.
    """

    def __init__(
        self,
        tn,
        loss_fn,
        norm_fn=None,
        norm_fn_scalar=None,
        loss_constants=None,
        loss_kwargs=None,
        constant_tags=None,
        optimizer='Adam',
        learning_rate=0.01,
        loss_target=None,
        tol_change=1e-9,
        tol_grad=1e-6,
        device=None,
        progbar=True
    ):
        torch, _ = _get_torch_and_device(device)

        self.tn = tn
        self.optimizer = optimizer
        self._learning_rate = learning_rate
        self.loss_target = loss_target
        self.tol_change = tol_change
        self.tol_grad = tol_grad
        assert progbar in [True, False, 'simple']
        self.progbar = progbar
        self.constant_tags = tags_to_utup(constant_tags)

        if norm_fn is None:
            def norm_fn(x):
                return x

        self.norm_fn = norm_fn
        self.norm_fn_scalar = norm_fn_scalar

        self.loss_constants = {}
        if loss_constants is not None:
            for k, v in loss_constants.items():
                # check if tensor network supplied
                if isinstance(v, TensorNetwork):
                    # convert it to constant pytorch TN
                    self.loss_constants[k] = constant_tn(v)
                elif isinstance(v, Tensor):
                    self.loss_constants[k] = constant_t(v)
                else:
                    self.loss_constants[k] = constant(v)
        self.loss_kwargs = {} if loss_kwargs is None else dict(loss_kwargs)
        self.loss_fn = functools.partial(
            loss_fn, **self.loss_constants, **self.loss_kwargs
        )
        self.loss = None
        self.losses = []
        self._n = 0

        # make pytorch version of network and gather variables etc.
        self.tn_opt, self.variables, self.iscomplex = parse_network_to_torch(
            tn, self.constant_tags
        )

        self.optimizer = getattr(torch.optim, optimizer)(self.variables,
                                                         lr=learning_rate)

    def closure(self):
        self.optimizer.zero_grad()
        # MJO: can't move this out of closure and down into optimizer() along
        # with norm_fn_scalar because we need norm_fn() to be a STRICTLY
        # in-place modification of tn_opt for that to work. I'm not sure how we
        # can guarantee that for a norm_fn() with completely general
        # functionality
        tn = self.norm_fn(self.tn_opt)
        self.loss = self.loss_fn(tn)
        self.loss.backward()
        return self.loss

    @property
    def learning_rate(self):
        return self._learning_rate

    @learning_rate.setter
    def learning_rate(self, lr):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    @property
    def nevals(self):
        """The number of gradient evaluations.
        """
        return self._n

    def _maybe_start_timer(self, max_time):
        if max_time is None:
            return
        else:
            self._time_start = time.time()

    def _time_should_stop(self, max_time):
        if max_time is None:
            return False
        else:
            return (time.time() - self._time_start) > max_time

    def _get_tn_opt_numpy(self):
        tn = self.norm_fn(self.tn_opt.copy())
        tn.drop_tags(t for t in tn.tags if '__VARIABLE' in t)
        tn.apply_to_arrays(lambda x: x.cpu().detach().numpy())
        return tn

    def optimize(self, max_steps, max_time=None):

        # perform the optimization with live progress
        if self.progbar is True:
            pbar = tqdm.tqdm(total=max_steps)
        else:
            pbar = tqdm.tqdm(total=max_steps, disable=True)
        self._maybe_start_timer(max_time)

        try:
            for _ in range(max_steps):
                if self.progbar == 'simple':
                    t0 = time.time()

                # keep the tn actually normalized during the optimization
                # by doing STRICTLY IN-PLACE modification of tn_opt
                if self.norm_fn_scalar is not None:
                    torch, _ = _TORCH_DEVICE
                    with torch.no_grad():
                        fac = self.norm_fn_scalar(self.tn_opt)
                        for tensor in self.tn_opt:
                            tensor._data *= fac

                self.optimizer.step(self.closure)
                self.losses.append(self.loss.item())
                pbar.set_description(f"{self.loss}")
                pbar.update()
                self._n += 1

                if self.progbar == 'simple':
                    t1 = time.time()
                    print(f'loss = {self.loss}, iter time = {t1 - t0}')

                # stopping conditions
                if (self.loss_target is not None):
                    if self.loss < self.loss_target:
                        break

                if self._time_should_stop(max_time):
                    break

                if len(self.losses) > 2:
                    loss_change = abs(self.losses[-1] - self.losses[-2])
                    if loss_change <= self.tol_change:
                        break

                if abs_max_grad(self.tn_opt) <= self.tol_grad:
                    break

                pbar.set_description(f"{self.loss}")

        except Exception as excpt:
            if not isinstance(excpt, KeyboardInterrupt):
                print(excpt)
                raise excpt
        finally:
            pbar.close()

        return self._get_tn_opt_numpy()
