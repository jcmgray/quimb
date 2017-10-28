#############################
Distributed Parallelism - MPI
#############################



Modes of execution
------------------

* Normal, "dynamic MPI mode": main process is not MPI, can run many OMP threads, dynamically spawns a cached mpi pool of processes which are not multithreaded (for efficiency) to do MPI things. Works for interactive sessions and allows OMP parallelism followed by MPI parallelism. Might not work well in a multi-node setting.

* ``quimb-mpi-python``, "mpi4py.futures mode": all processes are MPI, are spawned at start and have single OMP thread. Workers are split off at startup and only master runs the script, passing args and results using pickle to workers. This is achieved using ``<MPI_LAUNCHER> python -m mpi4py.futures <SCRIPT>``. For multi-node execution.

* ``quimb-mpi-python --syncro``, "syncro mode": All processes are MPI, are spawned at start and have a single OMP thread. All processes run the main script and have thus have access to the arguments submitted to the mpi pool functions without any communication, but split such work in a round-robin way, and broadcast the result to everyone when the Future's result is called. To maintain syncronicity futures cannot be cancelled. For simple multi-node execution.

.. warning::

    In syncro mode, potentially conflicting operations such as IO should be guarded with ``if MPI.COMM_WORLD.Get_rank() == 0`` etc.
