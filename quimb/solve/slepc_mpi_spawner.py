from .slepc_solver import slepc_seigsys
from ..accel import _NUM_THREADS


def cached_with_shutdown(fn):
    """
    """
    def wrapped_pool_fn(num_workers, num_threads):
        if wrapped_pool_fn.__settings__ == '__UNINITIALIZED__':
            # Pool has not been called, make a new one.
            wrapped_pool_fn.pool = fn(num_workers, num_threads)
            wrapped_pool_fn.__settings__ = (num_workers, num_threads)

        elif wrapped_pool_fn.__settings__ != (num_workers, num_threads):
            # New settings but old one exists, shut it down and return new one
            wrapped_pool_fn.pool.shutdown()
            wrapped_pool_fn.pool = fn(num_workers, num_threads)
            wrapped_pool_fn.__settings__ = (num_workers, num_threads)

        return wrapped_pool_fn.pool

    wrapped_pool_fn.__settings__ = '__UNINITIALIZED__'
    return wrapped_pool_fn


@cached_with_shutdown
def mpi_pool(num_workers, num_threads):
    """
    """
    from mpi4py.futures import MPIPoolExecutor

    return MPIPoolExecutor(num_workers, main=False, delay=1e-2,
                           env={'OMP_NUM_THREADS': str(num_threads)})


def slepc_mpi_seigsys_master(*args, **kwargs):
    """
    """
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    return slepc_seigsys(*args, comm=comm, **kwargs)


def slepc_mpi_seigsys_worker(*args, **kwargs):
    """
    """
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    slepc_seigsys(*args, comm=comm, **kwargs)


def slepc_mpi_seigsys(*args, num_workers=None, num_threads=1, **kwargs):
    """Automagically spawn mpi workers to do slepc eigen decomposition.
    """
    if num_workers is None:
        num_workers = _NUM_THREADS
    pool = mpi_pool(num_workers, num_threads)
    f = pool.submit(slepc_mpi_seigsys_master, *args, **kwargs)
    for _ in range(num_workers - 1):
        pool.submit(slepc_mpi_seigsys_worker, *args, **kwargs)
    return f.result()
