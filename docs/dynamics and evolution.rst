##############
Time Evolution
##############

Time evolutions in ``quimb`` are handled by the class :class:`~quimb.evo.Evolution`, which is initialized with a starting state and hamiltonian.

Basic Usage
~~~~~~~~~~~

Set up the :class:`~quimb.evo.Evolution` object with a initial state and hamiltonian.

.. code-block:: python

    >>> p0 = rand_ket(2**10)
    >>> h = ham_heis(10, sparse=True)
    >>> evo = Evolution(p0, h)

Update it in a single shot to a new time and get the state,

.. code-block:: python

    >>> evo.update_to(1)
    >>> evo.pt
    matrix([[ 0.03264482+0.04124479j],
            [ 0.01443932-0.05485965j],
            [ 0.02890706-0.00071998j],
            ...,
            [-0.04548890+0.02058016j],
            [-0.00183926-0.0257674j ],
            [-0.02184933+0.03170824j]])

Lazily generate the state at multiple times,

.. code-block:: python

    >>> for pt in evo.at_times([2, 3, 4]):
    ...     print(expec(pt, p0))
    0.004556606081856398
    0.00363075157285287
    0.007369494258758206


Methods of Updating
~~~~~~~~~~~~~~~~~~~

There are three methods of updating the state:

    - ``Evolution(..., method='integrate')``: use definite integration. Get system at each time step, only need action of Hamiltonian on state. Generally efficient. For pure and mixed states. The additional option ``int_small_step={False, True}`` determines whether a low or high order adaptive stepping scheme is used, giving naturally smaller or larger times steps. See :class:`scipy.integrate.ode` for details, ``False`` corresponds to ``"dop853"``, ``True`` to ``"dopri5"``.

    - ``Evolution(..., method='solve')``. Diagonalize the hamiltonian, which once done, allows quickly updating to arbitrary times. Supports pure and mixed states, recomended for small systems.

    -  ``Evolution(..., method='expm')``: compute the evolved state using the action of the matrix exponential in a 'single shot' style. Only needs action of Hamiltonian, for very large systems can use distributed MPI. Only for pure states.

Computing on the fly
~~~~~~~~~~~~~~~~~~~~

Sometimes, if integrating, it is best to just query the state at time-steps chosen dynamically by the adaptive scheme. This is achieved using the ``compute`` keyword supplied to ``Evolution``. It can also just be a convenient way to set up calculations as well:


.. code-block:: python

    >>> p0 = rand_ket(2**10)
    >>> h = ham_heis(10, sparse=True)
    >>> def calc_t_and_logneg(t, pt):
    ...     ln = logneg_subsys(pt, [2]*10, 0, 1)
    ...     return t, ln
    ...
    >>> evo = Evolution(p0, h, compute=calc_t_and_logneg)
    >>> evo.update_to(1)
    >>> ts, lns = zip(*evo.results)
    >>> ts
    (0.0, 0.06957398962890017, 0.13865533684861908, 0.21450605967375372, 0.29083278799508844, 0.37024226049289344, 0.4474543271078166, 0.5272008246783205, 0.608678805357641, 0.6915947062557095, 0.7749785052178692, 0.8569342998665894, 0.9347788617498614, 1.0)
    >>> lns
    (0.0, 0.27222905881173415, 0.45620792018155404, 0.5593762021046673, 0.5625027885480323, 0.4693229916795102, 0.311228611832485, 0.13832108516057381, 0.03885844451388185, 0.058663924562479174, 0.06616592139197426, 0.0380670545954638, 0.0, 0.0)

If a dict of callables is supplied to ``compute``, (each should take two arguments, the time, and the state, as above), ``Evolution.results`` will itself be a dictionary containing the results of each function at each time step, under the respective key:

.. code-block:: python

    >>> p0 = rand_ket(2**10)
    >>> h = ham_heis(10, sparse=True)
    >>> def calc_t(t, _):
    ...     return t
    ...
    >>> def calc_logneg(_, pt):
    ...     return logneg_subsys(pt, [2]*10, 0, 1)
    ...
    >>> evo = Evolution(p0, h, compute={'t': calc_t, 'ln': calc_logneg})
    >>> evo.update_to(1)
    >>> evo.results['t']
    (0.0, 0.06957398962890017, 0.13865533684861908, 0.21450605967375372, 0.29083278799508844, 0.37024226049289344, 0.4474543271078166, 0.5272008246783205, 0.608678805357641, 0.6915947062557095, 0.7749785052178692, 0.8569342998665894, 0.9347788617498614, 1.0)
    >>> evo.results['ln']
    (0.0, 0.27222905881173415, 0.45620792018155404, 0.5593762021046673, 0.5625027885480323, 0.4693229916795102, 0.311228611832485, 0.13832108516057381, 0.03885844451388185, 0.058663924562479174, 0.06616592139197426, 0.0380670545954638, 0.0, 0.0)
