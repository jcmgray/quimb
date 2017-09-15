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


# Work out if already running as mpi
if ('OMPI_COMM_WORLD_SIZE' in os.environ) or ('PMI_SIZE' in os.environ):
    ALREADY_RUNNING_AS_MPI = True
    if '_QUIMB_MPI_LAUNCHED' not in os.environ:
        raise RuntimeError(
            "For the moment, quimb programs launched explicitly"
            " using MPI need to use `quimb-mpiexec`."
        )

else:
    ALREADY_RUNNING_AS_MPI = False

# Work out the desired total number of workers
for _NUM_MPI_WORKERS_VAR in ['QUIMB_NUM_MPI_WORKERS',
                             'QUIMB_NUM_PROCS',
                             'OMP_NUM_THREADS',
                             ]:
    if _NUM_MPI_WORKERS_VAR in os.environ:
        NUM_MPI_WORKERS = int(os.environ[_NUM_MPI_WORKERS_VAR])
        NUM_MPI_WORKERS_SET = True
        break
    NUM_MPI_WORKERS_SET = False

if not NUM_MPI_WORKERS_SET:
    import psutil
    _NUM_MPI_WORKERS_VAR = 'psutil'
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
        elif ALREADY_RUNNING_AS_MPI and (num_workers != NUM_MPI_WORKERS):
            raise ValueError("Can't specify number of processes when running "
                             "under MPI rather than spawning processes.")

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
                                'QUIMB_NUM_MPI_WORKERS': str(num_workers),
                                '_QUIMB_MPI_LAUNCHED': 'SPAWNED'})


class GetMPIBeforeCall(object):
    """Wrap a function to automatically get the correct communicator before
    its called, and to set the `comm_self` kwarg to allow forced self mode.

    This is called by every mpi process before the function evaluation.
    """

    def __init__(self, fn):
        self.fn = fn

    def __call__(self, *args,
                 comm_self=False,
                 wait_for_workers=None,
                 **kwargs):
        """
        Parameters
        ----------
        *args :
            Supplied to self.fn
        comm_self : bool, optional
            Whether to force use of MPI.COMM_SELF
        wait_for_workers : int, optional
            If set, wait for the communicator to have this many workers, this
            can help to catch some errors regarding expected worker numbers.
        **kwargs :
            Supplied to self.fn
        """
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
                    raise RuntimeError(
                        "Timeout while waiting for {} workers "
                        "to join comm {}.".format(wait_for_workers, comm)
                    )

        return self.fn(*args, comm=comm, **kwargs)


class SpawnMPIProcessesFunc(object):
    """Automatically wrap a function to be executed in parallel by a
    pool of mpi workers.

    This is only called by the master mpi process in manual mode, or only by
    the (non-mpi) spawning process in automatic mode.
    """

    def __init__(self, fn):
        self.fn = fn

    def __call__(self, *args,
                 num_workers=None,
                 num_threads=1,
                 mpi_pool=None,
                 spawn_all=not ALREADY_RUNNING_AS_MPI,
                 **kwargs):
        """
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

        if num_workers == 1:  # no pool or communicator required
            return self.fn(*args, comm_self=True, **kwargs)

        kwargs['wait_for_workers'] = num_workers

        if mpi_pool is not None:
            pool = mpi_pool
        else:
            pool = get_mpi_pool(num_workers, num_threads)

        # the (non mpi) main process is idle while the workers compute.
        if spawn_all:
            futures = [pool.submit(self.fn, *args, **kwargs)
                       for _ in range(num_workers)]
            results = [f.result() for f in futures]

        # the master process is the master mpi process and contributes
        else:
            futures = [pool.submit(self.fn, *args, **kwargs)
                       for _ in range(num_workers - 1)]
            results = ([self.fn(*args, **kwargs)] +
                       [f.result() for f in futures])

        # Get master result, (not always first submitted)
        return next(r for r in results if r is not None)


# ---------------------------------- SLEPC ---------------------------------- #

seigsys_slepc_mpi = functools.wraps(slepc_seigsys)(
    GetMPIBeforeCall(slepc_seigsys))
seigsys_slepc_spawn = functools.wraps(slepc_seigsys)(
    SpawnMPIProcessesFunc(seigsys_slepc_mpi))

svds_slepc_mpi = functools.wraps(slepc_svds)(
    GetMPIBeforeCall(slepc_svds))
svds_slepc_spawn = functools.wraps(slepc_svds)(
    SpawnMPIProcessesFunc(svds_slepc_mpi))

mfn_multiply_slepc_mpi = functools.wraps(slepc_mfn_multiply)(
    GetMPIBeforeCall(slepc_mfn_multiply))
mfn_multiply_slepc_spawn = functools.wraps(slepc_mfn_multiply)(
    SpawnMPIProcessesFunc(mfn_multiply_slepc_mpi))
