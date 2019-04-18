import time
import functools

import tqdm
import numpy as np

from .tensor_core import TensorNetwork


_TORCH_DEVICE = None


def _get_torch_and_device():
    global _TORCH_DEVICE

    if _TORCH_DEVICE is None:
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


def variable_tn(tn):
    var_tn = tn.copy()
    var_tn.apply_to_arrays(variable)
    return var_tn


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
        if t.tags & constant_tags:
            t.modify(data=constant(t.data))

        # treat re and im parts as separate variables
        elif issubclass(t.dtype.type, np.complexfloating):
            iscomplex = True
            raise TypeError("Torch doesn not yet support complex data types.")

        else:
            torch_data = variable(t.data)
            variables.append(torch_data)
            t.modify(data=torch_data)

    return tn_torch, variables, iscomplex


class TNOptimizer:

    def __init__(
        self,
        tn,
        loss_fn,
        norm_fn=None,
        loss_constants=None,
        loss_kwargs=None,
        constant_tags=None,
        optimizer='Adam',
        learning_rate=0.01,
        loss_target=None,
        progbar=True
    ):
        torch, _ = _get_torch_and_device()

        self.tn = tn
        self.optimizer = optimizer
        self._learning_rate = learning_rate
        self.loss_target = loss_target
        self.progbar = progbar
        self.constant_tags = (set() if constant_tags is None
                              else set(constant_tags))

        # use identity if no nomalization required
        if norm_fn is None:
            def norm_fn(x):
                return x

        self.norm_fn = norm_fn

        self.loss_constants = {}
        if loss_constants is not None:
            for k, v in loss_constants.items():
                # check if tensor network supplied
                if isinstance(v, TensorNetwork):
                    # convert it to constant tensorflow TN
                    self.loss_constants[k] = constant_tn(v)
                else:
                    self.loss_constants[k] = constant(v)
        self.loss_kwargs = {} if loss_kwargs is None else dict(loss_kwargs)
        self.loss_fn = functools.partial(
            loss_fn, **self.loss_constants, **self.loss_kwargs
        )
        self.loss = None
        self._n = 0

        # make tensorflow version of network and gather variables etc.
        self.tn_opt, self.variables, self.iscomplex = parse_network_to_torch(
            tn, self.constant_tags
        )

        self.optimizer = getattr(torch.optim, optimizer)(self.variables,
                                                         lr=learning_rate)

    def closure(self):
        self.optimizer.zero_grad()
        self.loss = self.loss_fn(self.norm_fn(self.tn_opt))
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
        tn_opt_numpy = self.norm_fn(self.tn_opt)
        tn_opt_numpy.apply_to_arrays(lambda x: x.cpu().detach().numpy())
        return tn_opt_numpy

    def optimize(self, max_steps, max_time=None):

        # perform the optimization with live progress
        pbar = tqdm.tqdm(total=max_steps, disable=not self.progbar)
        self._maybe_start_timer(max_time)
        try:
            for _ in range(max_steps):
                self.optimizer.step(self.closure)
                pbar.set_description("{}".format(self.loss))
                pbar.update()
                self._n += 1

                # check if there is a target loss we have reached
                if (self.loss_target is not None):
                    if self.loss < self.loss_target:
                        break
                if self._time_should_stop(max_time):
                    break

                pbar.set_description("{}".format(self.loss))

        except Exception as excpt:
            if not isinstance(excpt, KeyboardInterrupt):
                print(excpt)
                raise excpt
        finally:
            pbar.close()

        return self._get_tn_opt_numpy()
