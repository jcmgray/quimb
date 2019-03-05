import tqdm


_TORCH = None


def get_torch():
    """Get torch and initialize eager mode if not already done.
    """
    global _TORCH
    if _TORCH is None:
        import torch
        _TORCH = torch
    return _TORCH


def variable_tn(tn):
    torch = get_torch()
    var_tn = tn.copy()
    var_tn.apply_to_arrays(lambda t: torch.tensor(t, requires_grad=True))
    return var_tn


def constant_tn(tn):
    torch = get_torch()
    const_tn = tn.copy()
    const_tn.apply_to_arrays(torch.tensor)
    return const_tn


class TNOpt:

    def __init__(self, tn, loss_fn, norm_fn=None,
                 optimizer='Adam',
                 learning_rate=0.01,
                 loss_target=None,
                 progbar=True):
        self.tn = tn
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self._learning_rate = learning_rate
        self.loss_target = loss_target
        self.progbar = progbar

        # use identity if no nomalization required
        if norm_fn is None:
            def norm_fn(x):
                return x

        self.norm_fn = norm_fn

        self.tn_opt = variable_tn(tn)

        torch = get_torch()

        self.optimizer = getattr(
            torch.optim, optimizer
        )([t.data for t in self.tn_opt], lr=learning_rate)

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

    def optimize(self, max_steps, max_time=None):

        # perform the optimization with live progress
        pbar = tqdm.tqdm(total=max_steps, disable=not self.progbar)
        try:
            for _ in range(max_steps):
                self.optimizer.step(self.closure)
                pbar.set_description("{}".format(self.loss))
                pbar.update()

                # check if there is a target loss we have reached
                if (self.loss_target is not None):
                    if self.loss < self.loss_target:
                        break

                # compute a final value of the loss
                self.loss = self.loss_fn(self.norm_fn(self.tn_opt))
                pbar.set_description("{}".format(self.loss))

        finally:
            pbar.close()

        return self.tn_opt
