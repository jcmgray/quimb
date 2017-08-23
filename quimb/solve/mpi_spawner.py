"""Manages the spawning of mpi processes to send to the various solvers.
"""
# TODO: don't send whole matrix? only marginal time savings but memory better.

import os
import functools
from .slepc_solver import (
    slepc_seigsys,
    slepc_svds,
    slepc_mfn_multiply,
)
from .scalapy_solver import eigsys_scalapy


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


class GetMPIBeforeCall(object):
    """Wrap a function to automatically get the correct communicator before
    its called, and to set the `comm_self` kwarg to allow forced self mode.
    """

    def __init__(self, fn):
        self.fn = fn

    def __call__(self, *args,
                 comm_self=False,
                 wait_for_workers=None,
                 **kwargs):
        from mpi4py import MPI

        if not comm_self:
            comm = MPI.COMM_WORLD
        else:
            comm = MPI.COMM_SELF

        if wait_for_workers is not None:
            from time import time
            t0 = time()
            while comm.Get_size() != wait_for_workers:
                if time() - t0 > 2:
                    raise RuntimeError("Timeout while waiting for {} workers "
                                       "to join comm {}."
                                       "".format(wait_for_workers, comm))

        return self.fn(*args, comm=comm, **kwargs)


class SpawnMPIProcessesFunc(object):
    """
    """

    def __init__(self, fn):
        """
        """
        self.fn = fn

    def __call__(self, *args,
                 num_workers=None,
                 num_threads=1,
                 mpi_pool=None,
                 spawn_all=not ALREADY_RUNNING_AS_MPI,
                 **kwargs):
        """Automatically wrap a function to be executed in parallel by a
        pool of mpi workers.

        Parameters
        ----------
            *args
                Supplied to `self.fn`.
            num_workers : int, optional
                How many total process should run function in parallel.
            num_threads : int, optional
                How many (OMP) threads each process should use
            mpi_pool : pool-like, optional
                If not None (default), submit function to this pool.
            spawn_all : bool, optional
                Whether all the parallel processes should be spawned (True), or
                num_workers - 1, so that the current process can also do work.
            **kwargs
                Supplied to `self.fn`.

        Returns
        -------
            `fn` output from the master process.
        """
        if num_workers is None:
            num_workers = NUM_MPI_WORKERS

        if num_workers == 1:
            kwargs['comm_self'] = True
            return self.fn(*args, **kwargs)
        else:
            kwargs['wait_for_workers'] = num_workers

        if mpi_pool is not None:
            pool = mpi_pool
        else:
            num_workers_to_spawn = num_workers - int(ALREADY_RUNNING_AS_MPI)
            if num_workers_to_spawn > 0:
                pool = get_mpi_pool(num_workers_to_spawn, num_threads)

        # the (non mpi) main process is idle while the workers compute.
        if spawn_all:
            futures = [pool.submit(self.fn, *args, **kwargs)
                       for _ in range(num_workers)]
            results = (f.result() for f in futures)
            # Get master result, (not always first submitted)
            return next(r for r in results if r is not None)

        # the master process is the master mpi process and contributes
        else:
            for _ in range(num_workers_to_spawn):
                pool.submit(self.fn, *args, **kwargs)
            return self.fn(*args, **kwargs)


# ---------------------------------- SLEPC ---------------------------------- #

seigsys_slepc_mpi = functools.wraps(slepc_seigsys)(
    GetMPIBeforeCall(slepc_seigsys))
seigsys_slepc_spawn = SpawnMPIProcessesFunc(seigsys_slepc_mpi)

svds_slepc_mpi = functools.wraps(slepc_svds)(
    GetMPIBeforeCall(slepc_svds))
svds_slepc_spawn = SpawnMPIProcessesFunc(svds_slepc_mpi)

mfn_multiply_slepc_mpi = functools.wraps(slepc_mfn_multiply)(
    GetMPIBeforeCall(slepc_mfn_multiply))
mfn_multiply_slepc_spawn = SpawnMPIProcessesFunc(mfn_multiply_slepc_mpi)


# --------------------------------- SCALAPY --------------------------------- #

eigsys_scalapy_mpi = functools.wraps(eigsys_scalapy)(
    GetMPIBeforeCall(eigsys_scalapy))
eigsys_scalapy_spawn = SpawnMPIProcessesFunc(eigsys_scalapy_mpi)
