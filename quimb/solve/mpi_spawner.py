"""Manages the spawning of mpi processes to send to the slepc solver.
"""
# TODO: don't send whole matrix? only marginal time savings but memory better.

# import functools
from .slepc_solver import slepc_seigsys
from .scalapy_solver import scalapy_eigsys
from ..accel import _NUM_WORKERS


def cached_with_shutdown(fn):
    """Wraps the mpi_pool getter such that successive calls with the same
    arguments return the same pool, but different arguments cause the
    previous pool to be shutdown.
    """
    def wrapped_pool_fn(*args):
        if wrapped_pool_fn.__settings__ == '__UNINITIALIZED__':
            # Pool has not been called, make a new one.
            wrapped_pool_fn.pool = fn(*args)
            wrapped_pool_fn.__settings__ = args

        elif wrapped_pool_fn.__settings__ != args:
            # New settings but old one exists, shut it down and return new one
            wrapped_pool_fn.pool.shutdown()
            wrapped_pool_fn.pool = fn(*args)
            wrapped_pool_fn.__settings__ = args

        return wrapped_pool_fn.pool

    wrapped_pool_fn.__settings__ = '__UNINITIALIZED__'
    return wrapped_pool_fn


@cached_with_shutdown
def get_mpi_pool(num_workers=None, num_threads=1):
    """
    """
    from mpi4py.futures import MPIPoolExecutor

    if num_workers is None:
        num_workers = _NUM_WORKERS

    return MPIPoolExecutor(num_workers, main=False, delay=1e-2,
                           env={'OMP_NUM_THREADS': str(num_threads)})


def mpi_spawn_func(fn, mat, *args,
                   num_workers=None,
                   num_threads=1,
                   mpi_pool=None,
                   **kwargs):
    """Automatically wrap a function to be executed in parallel by a
    pool of spawned mpi workers.
    """
    if num_workers is None:
        num_workers = min(_NUM_WORKERS, mat.shape[0])

    if mpi_pool is None:
        # Check if only one process needed --> don't spawn mpi pool
        if num_workers == 1:
            return fn(mat, *args, **kwargs)

        pool = get_mpi_pool(num_workers, num_threads)

    futures = [pool.submit(fn, mat, *args, **kwargs)
               for _ in range(num_workers)]
    results = (f.result() for f in futures)
    # Get master result, (not always first submitted)
    return next(r for r in results if r is not None)


# ---------------------------------- SLEPC ---------------------------------- #

def slepc_mpi_seigsys_submit_fn(*args, **kwargs):
    """
    """
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    return slepc_seigsys(*args, comm=comm, **kwargs)


def slepc_mpi_seigsys(mat, *args, num_workers=None, num_threads=1, **kwargs):
    """Automagically spawn mpi workers to do slepc eigen decomposition.
    """
    return mpi_spawn_func(slepc_mpi_seigsys_submit_fn, mat, *args,
                          num_workers=num_workers,
                          num_threads=num_threads, **kwargs)


# --------------------------------- SCALAPY --------------------------------- #

def scalapy_mpi_eigsys_submit_fn(*args, **kwargs):
    """
    """
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    return scalapy_eigsys(*args, comm=comm, **kwargs)


def scalapy_mpi_eigsys(mat, *args, **kwargs):
    """Automagically spawn mpi workers to do slepc eigen decomposition.
    """
    return mpi_spawn_func(scalapy_mpi_eigsys_submit_fn, mat, *args, **kwargs)
