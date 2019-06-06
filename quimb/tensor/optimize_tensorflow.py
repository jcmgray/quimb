"""Support for globally optimizing tensor networks using tensorflow in either
interactive (eager) mode or compiled (session) mode. For computationally heavy
optimizations (complex contractions) the two modes offer similar performance,
but for small to medium contractions graph mode is significantly faster.
"""

import time
import functools
from collections import namedtuple

import tqdm
import numpy as np


from .tensor_core import TensorNetwork


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


def parse_network_to_tf(tn, constant_tags):
    tf = get_tensorflow()
    eager = executing_eagerly()

    tn_tf = tn.copy()
    variables = []
    iscomplex = False

    for t in tn_tf:

        # check if tensor has any of the constant tags
        if t.tags & constant_tags:
            t.modify(data=tf.convert_to_tensor(t.data))

        # treat re and im parts as separate variables
        elif issubclass(t.dtype.type, np.complexfloating):
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


class TNOptimizer:
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
        Which optimizer to use, default: ``'AdamOptimizer'``.
        This should be an optimizer that can be found in the
        ``tensorflow.train`` submodule. The exception is if ``scipy`` is
        supplied the ``tf.contrib.opt.ScipyOptimizerInterface`` will be used,
        the options of which should be specified in the ``optimize`` call.
        This is mainly useful for the default ``'L-BFGS-B'`` algorithm.
    learning_rate : float, optional
        The learning rate to apply to use the optimizer with. You can
        dynamically change this between ``optimize`` calls.
    learning_decay_steps : int, optional
        How many steps to decay the learning rate over.
    learning_decay_rate : float, optional
        How much to decay the learning rate over ``learning_decay_steps``.
    loss_target : float, optional
        If supplied, stop optimizing once this loss is reached.
    progbar : bool, optional
        Whether to show live progress of the optimization.


    Examples
    --------

    .. code:: python3

        import quimb.tensor as qtn
        from quimb.tensor.optimize_tensorflow import TNOptimizer

        # can leave this out, and quimb will invoke tensorflow in
        # eager mode, sacrificing some efficiency
        import tensorflow as tf
        sess = tf.InteractiveSession()

    Variational compression of cyclic MPS, halving its bond dimension:

    .. code:: python3

        # the target state we want to compress
        targ = qtn.MPS_rand_state(n=20, bond_dim=16,
                                  cyclic=True, dtype='complex64')

        # our initial guess for the compressed state
        psi0 = qtn.MPS_rand_state(n=20, bond_dim=8,
                                  cyclic=True, dtype='complex64')

    .. code:: python3

        >>> targ.show()
         16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16
        +--o--o--o--o--o--o--o--o--o--o--o--o--o--o--o--o--o--o--o--o--+
           |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |

        >>> psi0.show()
         8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8
        +-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-+
          | | | | | | | | | | | | | | | | | | | |

        >>> abs(psi0.H @ targ)
        0.0012382069

    .. code:: python3

        # the thing (scalar quantity) we want to *minimize*
        def abs_overlap(psi, target):
            overlap = (psi.H & target) ^ all
            return -abs(overlap)

        # also need to define function that keeps us in the normalized manifold
        def psi_normalize(psi):
            nfactor = (psi.H & psi) ^ all
            return psi / nfactor**0.5

    Now we can set-up the optimizer and run it:

    .. code:: python3

        tnopt = TNOptimizer(
            tn=psi0,
            loss_fn=abs_overlap,
            loss_constants={'target': targ},
            norm_fn=psi_normalize,
            optimizer='Adam',
        )

    .. code:: python3

        >>> psif = tnopt.optimize(400)
        -0.7474728226661682: 100%|███████████| 400/400 [00:07<00:00, 51.74it/s]

        >>> abs(psif.H @ targ)
        0.747473

    You could alter the learning rate and keep running this if satisfactory
    results are not achieved. Note we don't here expect this to reach 1.0 as
    for random MPS all the singular values are likely significant and so
    compressing always involves some loss of fidelity.
    """

    def __init__(
        self,
        tn,
        loss_fn,
        norm_fn=None,
        loss_constants=None,
        loss_kwargs=None,
        constant_tags=None,
        optimizer='AdamOptimizer',
        learning_rate=0.1,
        learning_decay_steps=100,
        learning_decay_rate=0.5,
        loss_target=None,
        progbar=True
    ):
        tf = get_tensorflow()

        self.constant_tags = (set() if constant_tags is None
                              else set(constant_tags))

        # make tensorflow version of network and gather variables etc.
        self.tn_opt, self.variables, self.iscomplex = parse_network_to_tf(
            tn, self.constant_tags
        )

        self.progbar = progbar
        self.loss_target = loss_target

        # set dynamic learning rate
        self.global_step = tf.train.get_or_create_global_step()
        self.learning_rate = tf.train.exponential_decay(
            learning_rate=learning_rate, global_step=self.global_step,
            decay_steps=learning_decay_steps, decay_rate=learning_decay_rate,
        )

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
                    self.loss_constants[k] = tf.convert_to_tensor(v)
        self.loss_kwargs = {} if loss_kwargs is None else dict(loss_kwargs)
        self.loss_fn = functools.partial(
            loss_fn, **self.loss_constants, **self.loss_kwargs
        )
        self.loss = None
        self._n = 0

        if optimizer == 'scipy':
            if executing_eagerly():
                raise ValueError("The tensorflow ``'scipy'`` interface is not "
                                 "available when executing eagerly.")

            self.optimizer = 'scipy'
            return

        elif isinstance(optimizer, str):
            if 'Optimizer' not in optimizer:
                optimizer += 'Optimizer'
            self.optimizer = getattr(tf.train, optimizer)(self.learning_rate)
        else:
            self.optimizer = optimizer

        if not executing_eagerly():
            # generate the loss and optimizer computational graphs
            self.loss_op = self.loss_fn(self.norm_fn(self.tn_opt))
            self.train_op = self.optimizer.minimize(
                self.loss_op, global_step=self.global_step)

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
                        sess=None, max_time=None, **kwargs):
        tf = get_tensorflow()

        loss_op = self.loss_fn(self.norm_fn(self.tn_opt))

        kwargs['maxfun'] = max_steps - 1
        if sess is None:
            sess = tf.get_default_session()
        init_uninit_vars(sess)

        pbar = tqdm.tqdm(total=max_steps, disable=not self.progbar)
        self._maybe_start_timer(max_time)
        timed_out = False

        def step_callback(*_, **__):
            if self._time_should_stop(max_time):
                raise TimeoutError

        def loss_callback(loss_val):
            self._n += 1
            self.loss = loss_val
            pbar.set_description("{}".format(loss_val))
            pbar.update()
            return self._time_should_stop(max_time)

        try:
            optimizer = tf.contrib.opt.ScipyOptimizerInterface(
                loss_op, options=kwargs, method=method,
            )
            optimizer.minimize(sess, fetches=[loss_op],
                               loss_callback=loss_callback,
                               step_callback=step_callback)
        except TimeoutError:
            timed_out = True
        finally:
            pbar.close()

            if timed_out:
                raise TimeoutError("The scipy optimizer currently has no way "
                                   "of extracting the current tensor values "
                                   "mid execution.")

            return self._get_tn_opt_numpy_graph(sess)

    def _grad(self):
        tf = get_tensorflow()
        # this is the function that computes the loss and tracks gradient
        with tf.GradientTape() as tape:
            if self.iscomplex:
                tn = evaluate_lazy_complex(self.tn_opt)
            else:
                tn = self.tn_opt
            self.loss = self.loss_fn(self.norm_fn(tn))
        return tape.gradient(self.loss, self.variables)

    def _optimize_eager(self, max_steps, max_time=None):
        # perform the optimization with live progress
        pbar = tqdm.tqdm(total=max_steps, disable=not self.progbar)
        self._maybe_start_timer(max_time)
        try:
            for _ in range(max_steps):
                # compute gradient and display loss value
                grads = self._grad()
                pbar.set_description("{}".format(self.loss))

                # performing an optimization step
                self.optimizer.apply_gradients(
                    zip(grads, self.variables),
                    global_step=self.global_step,
                )
                pbar.update()
                self._n += 1

                # check stopping criteria
                if (self.loss_target is not None):
                    if self.loss < self.loss_target:
                        break
                if self._time_should_stop(max_time):
                    break

        finally:
            pbar.close()
            return self._get_tn_opt_numpy_eager()

    def _optimize_graph(self, max_steps, max_time=None, sess=None, ):
        if sess is None:
            tf = get_tensorflow()
            sess = tf.get_default_session()

        init_uninit_vars(sess)

        # perform the optimization with live progress
        pbar = tqdm.tqdm(total=max_steps, disable=not self.progbar)
        self._maybe_start_timer(max_time)
        try:
            for _ in range(max_steps):
                _, self.loss = sess.run([self.train_op, self.loss_op])
                pbar.set_description("{}".format(self.loss))
                pbar.update()
                self._n += 1

                # check stopping criteria
                if (self.loss_target is not None):
                    if self.loss < self.loss_target:
                        break
                if self._time_should_stop(max_time):
                    break
        except Exception as e:
            print(e)
        finally:
            pbar.close()
            return self._get_tn_opt_numpy_graph(sess)

    def optimize(self, max_steps, max_time=None, **kwargs):
        """
        Returns
        -------
        tn_opt : TensorNetwork
            The optimized tensor network (with arrays converted back to numpy).
        """
        if self.optimizer == 'scipy':
            kwargs.setdefault('gtol', 1e-12)
            return self._optimize_scipy(max_steps, max_time=max_time, **kwargs)
        elif executing_eagerly():
            return self._optimize_eager(max_steps, max_time=max_time, **kwargs)
        else:
            return self._optimize_graph(max_steps, max_time=max_time, **kwargs)
