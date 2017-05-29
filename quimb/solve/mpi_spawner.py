"""Manages the spawning of mpi processes to send to the slepc solver.
"""
# TODO: don't send whole matrix? only marginal time savings but memory better.

import os
from .slepc_solver import slepc_seigsys
from .scalapy_solver import scalapy_eigsys


# Work out if already running as mpi
if ('OMPI_COMM_WORLD_SIZE' in os.environ) or ('PMI_SIZE' in os.environ):
    ALREADY_RUNNING_AS_MPI = True
else:
    ALREADY_RUNNING_AS_MPI = False


# Work out the desired total number of workers
for var in ['QUIMB_NUM_MPI_WORKERS',
            'QUIMB_NUM_PROCS',
            'OMPI_UNIVERSE_SIZE',
            'MPI_UNIVERSE_SIZE',
            'OMP_NUM_THREADS']:
    if var in os.environ:
        NUM_MPI_WORKERS = int(os.environ[var])
        NUM_MPI_WORKERS_SET = True
        break
    NUM_MPI_WORKERS_SET = False

if not NUM_MPI_WORKERS_SET:
    import psutil
    NUM_MPI_WORKERS = psutil.cpu_count(logical=False)


class CachedPoolWithShutdown(object):
    """
    """

    def __init__(self, pool_fn):
        self._settings = '__UNINITIALIZED__'
        self._pool_fn = pool_fn

    def __call__(self, num_workers=None, num_threads=1):
        # convert None to default so the cache the same
        if num_workers is None:
            num_workers = NUM_MPI_WORKERS

        # first call
        if self._settings == '__UNINITIALIZED__':
            self._pool = self._pool_fn(num_workers, num_threads)
            self._settings = (num_workers, num_threads)
        # new type of pool requested
        elif self._settings != (num_workers, num_threads):
            self._pool.shutdown()
            self._pool = self._pool_fn(num_workers, num_threads)
            self._settings = (num_workers, num_threads)
        return self._pool


@CachedPoolWithShutdown
def get_mpi_pool(num_workers=None, num_threads=1):
    """
    """
    from mpi4py.futures import MPIPoolExecutor
    return MPIPoolExecutor(num_workers, main=False, delay=1e-2,
                           env={'OMP_NUM_THREADS': str(num_threads),
                                'MPI_UNIVERSE_SIZE': '1'})


def mpi_pool_func(fn, *args,
                  num_workers=None,
                  num_threads=1,
                  mpi_pool=None,
                  spawn_all=not ALREADY_RUNNING_AS_MPI,
                  **kwargs):
    """Automatically wrap a function to be executed in parallel by a
    pool of mpi workers.
    """
    if num_workers is None:
        num_workers = NUM_MPI_WORKERS

    if num_workers == 1:
        kwargs['comm_self'] = True

    if mpi_pool is not None:
        pool = mpi_pool
    else:
        num_workers_to_spawn = num_workers - int(ALREADY_RUNNING_AS_MPI)
        if num_workers_to_spawn > 0:
            pool = get_mpi_pool(num_workers_to_spawn, num_threads)

    # the (non mpi) main process is idle while the workers compute.
    if spawn_all:
        futures = [pool.submit(fn, *args, **kwargs)
                   for _ in range(num_workers)]
        results = (f.result() for f in futures)
        # Get master result, (not always first submitted)
        return next(r for r in results if r is not None)

    # the master process is the master mpi process and contributes
    else:
        for _ in range(num_workers - 1):
            pool.submit(fn, *args, **kwargs)
        return fn(*args, **kwargs)


# ---------------------------------- SLEPC ---------------------------------- #

def slepc_mpi_seigsys_submit_fn(*args, comm_self=False, **kwargs):
    """SLEPc solve function with initial MPI comm find.
    """
    if not comm_self:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
    else:
        comm = None
    return slepc_seigsys(*args, comm=comm, **kwargs)


def slepc_mpi_seigsys(*args, num_workers=None, num_threads=1, **kwargs):
    """Automagically spawn mpi workers to do slepc eigen decomposition.
    """
    return mpi_pool_func(slepc_mpi_seigsys_submit_fn, *args,
                         num_workers=num_workers,
                         num_threads=num_threads, **kwargs)


# --------------------------------- SCALAPY --------------------------------- #

def scalapy_mpi_eigsys_submit_fn(*args, comm_self=False, **kwargs):
    """Scalapy solve function with initial MPI comm find.
    """
    from mpi4py import MPI
    if comm_self:
        comm = MPI.COMM_SELF
    else:
        comm = MPI.COMM_WORLD
    return scalapy_eigsys(*args, comm=comm, **kwargs)


def scalapy_mpi_eigsys(*args, **kwargs):
    """Automagically spawn mpi workers to do slepc eigen decomposition.
    """
    return mpi_pool_func(scalapy_mpi_eigsys_submit_fn, *args, **kwargs)
