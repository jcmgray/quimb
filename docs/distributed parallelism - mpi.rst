.. _mpistuff:

###
MPI
###

The ``slepc`` and ``petsc`` parts of ``quimb`` (``eigh(..., k=..., method='slepc')``, ``svds(..., method='slepc')``, ``expm(..., method='slepc')`` etc.), as well as often being faster than ``scipy``, perform the calculations using MPI and are thus suited to clusters with many nodes.

These generally perform best when run on multiple, single-threaded MPI processes, whereas ``numpy`` and ``scipy`` need single-process multi-threaded exectution to parallize (via BLAS).
By default, ``quimb`` thus switches between standard execution and a cached pool of MPI processes when required. It can also run more explicitly under MPI as descibed in :ref:`modes-of-execution`.

If running in a distributed cluster it is also best not to pass full operators to these functions (which will be pickled then transferred), rather it is best to leave the operator unconstructed, whith each worker constructing only the rows it needs.

1. The first aspect is handled by the :class:`~quimb.Lazy` object, which is essentially
   ``functools.partial`` with a `.shape` attribute that must be supplied.
2. The second aspect (only constructing the right rows) is achieved whenever a function takes a
   ``ownership`` argument specifying the slice of rows to construct. Any operators based on
   :func:`~quimb.kron` and :func:`~quimb.ikron` such as the built-in Hamiltonians can do this.

See the :ref:`examples` for a demonstration of this.


.. _modes-of-execution:

Modes of execution
------------------

* Normal, "dynamic MPI mode": main process is not MPI, can run many OMP threads, dynamically spawns a cached mpi pool of processes which are not multithreaded (for efficiency) to do MPI things. Works for interactive sessions and allows OMP parallelism followed by MPI parallelism. Might not work well in a multi-node setting.

* ``quimb-mpi-python``, "mpi4py.futures mode": all processes are MPI, are spawned at start and have single OMP thread. Workers are split off at startup and only master runs the script, passing args and results using pickle to workers. This is achieved using ``<MPI_LAUNCHER> python -m mpi4py.futures <SCRIPT>``. For multi-node execution.

For these first two modes, only one process ever runs the main script and you do not need to think about MPI rank etc. This is unlike:

* ``quimb-mpi-python --syncro``, "syncro mode": All processes are MPI, are spawned at start and have a single OMP thread. All processes run the main script and have thus have access to the arguments submitted to the mpi pool functions without any communication, but split such work in a round-robin way, and broadcast the result to everyone when the Future's result is called. To maintain syncronicity futures cannot be cancelled. For simple multi-node execution.

.. warning::

    In syncro mode, potentially conflicting operations such as IO should be guarded with ``if MPI.COMM_WORLD.Get_rank() == 0`` etc. Additionally, any functions called outside of the MPI pool should be pure to ensure syncronization.


MPI pool
--------

The pool of MPI workers is generated automatically for special functions that require it, but can also be explicitly used with :func:`~quimb.get_mpi_pool` for other simple parallel tasks that will then scale to multi-node cluster settings.

See the :ref:`examples` for a demonstration of this.



An Example
----------

Consider the following script (found in ``docs/examples/ex_mpi_expm_evo.py``):

.. literalinclude:: examples/ex_mpi_expm_evo.py
    :language: py3


If we run the script in normal mode we get:

.. code-block:: bash

    $ python ex_syncro_expm_evo.py
    I am worker 0 of total 1 running main script...
    0: ownership=(0, 65536)
    1: ownership=(65536, 131072)
    2: ownership=(131072, 196608)
    3: ownership=(196608, 262144)
    0: I have final state norm [[1.+0.j]]
    0: sysb=range(5, 10)
    3: sysb=range(6, 11)
    1: sysb=range(7, 12)
    2: sysb=range(8, 13)
    3: sysb=range(9, 14)
    1: sysb=range(10, 15)
    0: sysb=range(11, 16)
    2: sysb=range(12, 17)
    0: I have logneg results: [0.8909014842733883, 0.8909987302898089, 0.8924045900195905, 0.8921292033437735, 0.8912200853252004, 0.8913080757931359, 0.8908582609382703, 0.8924006528057047]

Although the process running the main script prints 0 as its rank, it is not one of the workers (it is '0 of 1'). If we run it in eager mpi mode we get:

.. code-block:: bash

    $ quimb-mpi-python ex_syncro_expm_evo.py
    Launching quimb in mpi4py.futures mode with mpiexec.
    I am worker 0 of total 4 running main script...
    1: ownership=(65536, 131072)
    2: ownership=(131072, 196608)
    3: ownership=(196608, 262144)
    0: ownership=(0, 65536)
    0: I have final state norm [[1.+0.j]]
    1: sysb=range(5, 10)
    2: sysb=range(6, 11)
    3: sysb=range(7, 12)
    3: sysb=range(8, 13)
    2: sysb=range(9, 14)
    1: sysb=range(10, 15)
    3: sysb=range(11, 16)
    2: sysb=range(12, 17)
    0: I have logneg results: [0.8909014842733911, 0.8909987302898126, 0.892404590019593, 0.8921292033437763, 0.8912200853252026, 0.8913080757931393, 0.8908582609382716, 0.8924006528057071]


Note this is essentially the same, apart from the fact that the process running the main script is one now the MPI processes ('0 of 4').

Finally we can run in in 'syncro' mode:

.. code-block:: bash

    $ quimb-mpi-python --syncro ex_syncro_expm_evo.py
    Launching quimb in Syncro mode with mpiexec.
    I am worker 1 of total 4 running main script...
    I am worker 2 of total 4 running main script...
    I am worker 0 of total 4 running main script...
    I am worker 3 of total 4 running main script...
    2: ownership=(131072, 196608)
    1: ownership=(65536, 131072)
    0: ownership=(0, 65536)
    3: ownership=(196608, 262144)
    1: I have final state norm [[1.+0.j]]
    1: sysb=range(6, 11)
    0: I have final state norm [[1.+0.j]]
    0: sysb=range(5, 10)
    2: I have final state norm [[1.+0.j]]
    3: I have final state norm [[1.+0.j]]
    2: sysb=range(7, 12)
    3: sysb=range(8, 13)
    3: sysb=range(12, 17)
    2: sysb=range(11, 16)
    0: sysb=range(9, 14)
    1: sysb=range(10, 15)
    2: I have logneg results: [0.8909014842733911, 0.8909987302898126, 0.892404590019593, 0.8921292033437763, 0.8912200853252026, 0.8913080757931393, 0.8908582609382716, 0.8924006528057071]
    3: I have logneg results: [0.8909014842733911, 0.8909987302898126, 0.892404590019593, 0.8921292033437763, 0.8912200853252026, 0.8913080757931393, 0.8908582609382716, 0.8924006528057071]
    1: I have logneg results: [0.8909014842733911, 0.8909987302898126, 0.892404590019593, 0.8921292033437763, 0.8912200853252026, 0.8913080757931393, 0.8908582609382716, 0.8924006528057071]
    0: I have logneg results: [0.8909014842733911, 0.8909987302898126, 0.892404590019593, 0.8921292033437763, 0.8912200853252026, 0.8913080757931393, 0.8908582609382716, 0.8924006528057071]

Now all workers run the main script, but still correctly split work when a ``slepc`` computation is encountered, and when work is distributed via :func:`~quimb.linalg.mpi_launcher.get_mpi_pool`.
