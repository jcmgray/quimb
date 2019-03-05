"""Support for globally optimizing tensor networks using tensorflow in either
interactive (eager) mode or compiled (session) mode. For computationally heavy
optimizations (complex contractions) the two modes offer similar performance,
but for small to medium contractions graph mode is significantly faster.
"""

from collections import namedtuple

import tqdm
import numpy as np


LazyComplexTF = namedtuple('LazyComplexTF', ['shape', 'real', 'imag'])
"""We need to lazily represent complex tensorflow arrays so that combining the
variable real and complex parts takes place in the gradient tape always when
executing eagerly.
"""


_TENSORFLOW = None
_TENSORFLOW_EAGER = None


def get_tensorflow(eager=None):
    """Get tensorflow and initialize eager mode if tensorflow has not been
    imported yet. If it has, check whether eager mode has been enabled.
    """
    global _TENSORFLOW
    global _TENSORFLOW_EAGER

    if _TENSORFLOW is None:
        # first check if we are importing tensorflow for the first time
        import sys

        try:
            # leave eager decision to whenever tensorflow was imported
            _TENSORFLOW = sys.modules['tensorflow']
            _TENSORFLOW_EAGER = _TENSORFLOW.executing_eagerly()

        except KeyError:
            import tensorflow as tf

            # default to enabling the more intuitive eager mode
            if eager is None:
                eager = True

            if eager:
                tf.enable_eager_execution()

            _TENSORFLOW = tf
            _TENSORFLOW_EAGER = eager

    return _TENSORFLOW


def executing_eagerly():
    global _TENSORFLOW_EAGER
    return _TENSORFLOW_EAGER


def variable_tn(tn):
    """Convert a tensor network's arrays to tensorflow Variables.
    """
    tf = get_tensorflow()
    tf_tn = tn.copy()
    tf_tn.apply_to_arrays(tf.Variable)
    return tf_tn


def constant_tn(tn):
    """Convert a tensor network's arrays to tensorflow constants.
    """
    tf = get_tensorflow()
    tf_tn = tn.copy()
    tf_tn.apply_to_arrays(tf.convert_to_tensor)
    return tf_tn


def parse_network_to_tf(tn):
    tf = get_tensorflow()
    eager = executing_eagerly()

    tn_tf = tn.copy()
    variables = []
    iscomplex = False

    for t in tn_tf:

        # treat re and im parts as separate variables
        if issubclass(t.dtype.type, np.complexfloating):
            tf_re, tf_im = tf.Variable(t.data.real), tf.Variable(t.data.imag)
            variables.extend((tf_re, tf_im))

            # if eager need to delay complex combination until gradient tape
            if eager:
                tf_data = LazyComplexTF(t.data.shape, tf_re, tf_im)
            else:
                tf_data = tf.complex(tf_re, tf_im)

            t.modify(data=tf_data)
            iscomplex = True

        else:
            tf_data = tf.Variable(t.data)
            variables.append(tf_data)
            t.modify(data=tf_data)

    return tn_tf, variables, iscomplex


def evaluate_lazy_complex(tn_tf):
    """Can call this in a gradient tape context to merge real and imaginary
    parts so far lazily represented.
    """
    tf = get_tensorflow()

    tn_eval = tn_tf.copy()

    for t in tn_eval:
        if isinstance(t.data, LazyComplexTF):
            t.modify(data=tf.complex(t.data.real, t.data.imag))

    return tn_eval


def init_uninit_vars(sess):
    """ Initialize all other trainable variables, i.e. those which are
    uninitialized. Allows ``optimize`` to be interatively stopped and started.
    """
    tf = get_tensorflow()

    # names of variables
    uninit_vars = sess.run(tf.report_uninitialized_variables())
    vars_list = {v.decode('utf-8') for v in uninit_vars}

    # variables themselves
    uninit_vars_tf = [v for v in tf.global_variables()
                      if v.name.split(':')[0] in vars_list]

    sess.run(tf.variables_initializer(var_list=uninit_vars_tf))


class TNOpt:
    """Optimize a tensor network's arrays using tensorflow.

    Parameters
    ----------
    tn : TensorNetwork
        The tensor network to optimize, every tensor in this will be treated
        as a variable.
    loss_fn : callable
        The target loss function to minimize. Should accept ``tn`` (backed by
        tensorflow tensors) or the output of ``norm_fn``.
    norm_fn : callable, optional
        A function to normalize ``tn`` before being passed to ``loss_fn`` and
        also to call on the final, optimized tensor network. This can be used
        to enforce constraints such as normalization or unitarity.
    optimizer : str, optional
        Which optimizer to use, default: ``'AdamOptimizer'``.
        This should be an optimizer that can be found in the
        ``tensorflow.train`` submodule. The exception is if ``scipy`` is
        supplied the ``tf.contrib.opt.ScipyOptimizerInterface`` will be used,
        the options of which should be specified in the ``optimize`` call.
        This is mainly useful for the default ``'L-BFGS-B'`` algorithm.
    learning_rate : float, optional
        The learning rate to apply to use the optimizer with. You can
        dynamically change this between ``optimize`` calls.
    loss_target : float, optional
        If supplied, stop optimizing once this loss is reached.
    progbar : bool, optional
        Whether to show live progress of the optimization.


    Examples
    --------

        import tensorflow as tf
        import quimb.tensor as qtn
        from quimb.tensor.optimize_tensorflow import *

    Variational compression of cyclic MPS:

        # the target state we want to compress
        targ = qtn.MPS_rand_state(10, bond_dim=6, cyclic=True, dtype='float32')

        # our intial guess for the compressed state
        psi0 = qtn.MPS_rand_state(10, bond_dim=4, cyclic=True, dtype='float32')

    Note since we are working with real MPS we don't need to worry about
    conjugating etc.

        # first convert our target to a tensorflow constant TN
        tf_targ = constant_tn(targ)
        contract_opts = {'cache': False, 'backend': 'tensorflow'}

        # then define the quantity we want to minimize (-ve. abs overlap)
        def loss(psi):
            overlap = (psi & tf_targ).contract(all, **contract_opts)
            return -tf.abs(overlap)

        # also need to define function that keeps us in the normalized manifold
        def norm(psi):
            nfactor = (psi & psi).contract(all, **contract_opts)
            return psi / nfactor**0.5

    Now we can run the optimizer:

        opt = tf_optimize(psi0, loss, norm, learning_rate=0.1, max_steps=100)
        # -0.930211603: 100%|████████████████| 100/100 [00:14<00:00,  8.20it/s]

    Note we don't expect this to reach 1.0 as for random MPS all the singular
    values are likely significant and so compressing always involves some loss
    of fidelity.

        >>> opt.H @ targ
        0.9302115
    """

    def __init__(self, tn, loss_fn, norm_fn=None,
                 optimizer='AdamOptimizer',
                 learning_rate=0.01,
                 loss_target=None,
                 progbar=True):
        tf = get_tensorflow()

        # make tensorflow version of network and gather variables etc.
        self.tn_opt, self.variables, self.iscomplex = parse_network_to_tf(tn)

        self.progbar = progbar
        self.loss_target = loss_target
        self.learning_rate = learning_rate

        # use identity if no nomalization required
        if norm_fn is None:
            def norm_fn(x):
                return x

        self.norm_fn = norm_fn
        self.loss_fn = loss_fn
        self.loss = None

        if optimizer == 'scipy':
            self.optimizer = 'scipy'
            return

        elif isinstance(optimizer, str):
            if 'Optimizer' not in optimizer:
                optimizer += 'Optimizer'
            self.optimizer = getattr(tf.train, optimizer)(self._get_lr)
        else:
            self.optimizer = optimizer

        if not executing_eagerly():
            # generate the loss and optimizer computational graphs
            self.loss_op = self.loss_fn(self.norm_fn(self.tn_opt))
            self.train_op = self.optimizer.minimize(self.loss_op)

    def _get_lr(self):
        """Defining this callable allows dynamic learning rate for optimizer.
        """
        return self.learning_rate

    def _get_tn_opt_numpy_eager(self):
        """Get normalized, numpy version of optimized tensor network, in eager
        mode.
        """
        if self.iscomplex:
            tn = evaluate_lazy_complex(self.tn_opt)
        else:
            tn = self.tn_opt

        tn_opt_numpy = self.norm_fn(tn)
        tn_opt_numpy.apply_to_arrays(lambda x: x.numpy())
        return tn_opt_numpy

    def _get_tn_opt_numpy_graph(self, sess):
        """Get normalized, numpy version of optimized tensor network.
        """
        tn_opt_numpy = self.norm_fn(self.tn_opt)

        # evaluate all arrays as once for performance reasons
        np_arrays = sess.run([t.data for t in tn_opt_numpy])
        for t, data in zip(tn_opt_numpy, np_arrays):
            t.modify(data=data)

        return tn_opt_numpy

    def _optimize_scipy(self, max_steps, method='L-BFGS-B',
                        sess=None, **kwargs):
        tf = get_tensorflow()

        loss_op = self.loss_fn(self.norm_fn(self.tn_opt))

        kwargs['maxiter'] = max_steps
        if sess is None:
            sess = tf.get_default_session()
        init_uninit_vars(sess)

        pbar = tqdm.tqdm(total=max_steps, disable=not self.progbar)

        def loss_callback(loss_val):
            pbar.set_description("{}".format(loss_val))
            pbar.update()

        try:
            optimizer = tf.contrib.opt.ScipyOptimizerInterface(
                loss_op, options=kwargs, method=method,

            )
            optimizer.minimize(sess, fetches=[loss_op],
                               loss_callback=loss_callback)
        finally:
            pbar.close()

        return self._get_tn_opt_numpy_graph(sess)

    def grad(self):
        tf = get_tensorflow()
        # this is the function that computes the loss and tracks gradient
        with tf.GradientTape() as tape:
            if self.iscomplex:
                tn = evaluate_lazy_complex(self.tn_opt)
            else:
                tn = self.tn_opt
            self.loss = self.loss_fn(self.norm_fn(tn))
        return tape.gradient(self.loss, self.variables)

    def _optimize_eager(self, max_steps):
        tf = get_tensorflow()

        # perform the optimization with live progress
        pbar = tqdm.tqdm(total=max_steps, disable=not self.progbar)
        try:
            for _ in range(max_steps):
                # compute gradient and display loss value
                grads = self.grad()
                pbar.set_description("{}".format(self.loss))

                # performing an optimization step
                self.optimizer.apply_gradients(
                    zip(grads, self.variables),
                    global_step=tf.train.get_or_create_global_step()
                )
                pbar.update()

                # check if there is a target loss we have reached
                if (self.loss_target is not None):
                    if self.loss < self.loss_target:
                        break

        finally:
            pbar.close()

        return self._get_tn_opt_numpy_eager()

    def _optimize_graph(self, max_steps, sess=None):
        if sess is None:
            tf = get_tensorflow()
            sess = tf.get_default_session()

        init_uninit_vars(sess)

        # perform the optimization with live progress
        pbar = tqdm.tqdm(total=max_steps, disable=not self.progbar)
        try:
            for _ in range(max_steps):
                _, self.loss = sess.run([self.train_op, self.loss_op])
                pbar.set_description("{}".format(self.loss))
                pbar.update()

                # check if there is a target loss we have reached
                if (self.loss_target is not None):
                    if self.loss < self.loss_target:
                        break
        finally:
            pbar.close()

        return self._get_tn_opt_numpy_graph(sess)

    def optimize(self, max_steps, **kwargs):
        """
        Returns
        -------
        tn_opt : TensorNetwork
            The optimized tensor network (with arrays converted back to numpy).
        """

        if self.optimizer == 'scipy':
            return self._optimize_scipy(max_steps, **kwargs)
        elif executing_eagerly():
            return self._optimize_eager(max_steps, **kwargs)
        else:
            return self._optimize_graph(max_steps, **kwargs)
