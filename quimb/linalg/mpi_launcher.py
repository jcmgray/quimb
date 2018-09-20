"""Manages the spawning of mpi processes to send to the various solvers.
"""

import os
import functools

from .slepc_linalg import (
    eigs_slepc, svds_slepc, mfn_multiply_slepc, ssolve_slepc,
)
from ..core import _NUM_THREAD_WORKERS

# Work out if already running as mpi
if ('OMPI_COMM_WORLD_SIZE' in os.environ) or ('PMI_SIZE' in os.environ):
    ALREADY_RUNNING_AS_MPI = True
    if '_QUIMB_MPI_LAUNCHED' not in os.environ:
        raise RuntimeError(
            "For the moment, quimb programs launched explicitly"
            " using MPI need to use `quimb-mpi-python`.")
    USE_SYNCRO = "QUIMB_SYNCRO_MPI" in os.environ
else:
    ALREADY_RUNNING_AS_MPI = False
    USE_SYNCRO = False

# Work out the desired total number of workers
for _NUM_MPI_WORKERS_VAR in ['QUIMB_NUM_MPI_WORKERS',
                             'QUIMB_NUM_PROCS',
                             'OMPI_COMM_WORLD_SIZE',
                             'PMI_SIZE',
                             'OMP_NUM_THREADS']:
    if _NUM_MPI_WORKERS_VAR in os.environ:
        NUM_MPI_WORKERS = int(os.environ[_NUM_MPI_WORKERS_VAR])
        break
else:
    import psutil
    _NUM_MPI_WORKERS_VAR = 'psutil'
    NUM_MPI_WORKERS = psutil.cpu_count(logical=False)


class SyncroFuture:

    def __init__(self, result, result_rank, comm):
        self._result = result
        self.result_rank = result_rank
        self.comm = comm

    def result(self):
        return self.comm.bcast(self._result, root=self.result_rank)

    @staticmethod
    def cancel():
        raise ValueError("SyncroFutures cannot be cancelled - they are "
                         "submitted in a parallel round-robin fasion where "
                         "each worker immediately computes all its results.")


class SynchroMPIPool:

    def __init__(self):
        import itertools
        from mpi4py import MPI
        self.comm = MPI.COMM_WORLD
        self.size = self.comm.Get_size()
        self.rank = self.comm.Get_rank()
        self.counter = itertools.cycle(range(0, NUM_MPI_WORKERS))

    def submit(self, fn, *args, **kwargs):
        # round robin iterate through ranks
        current_counter = next(self.counter)

        # accept job and compute if have the same rank, else do nothing
        if current_counter == self.rank:
            res = fn(*args, **kwargs)
        else:
            res = None

        # wrap the result in a SyncroFuture, that will broadcast result
        return SyncroFuture(res, current_counter, self.comm)

    def shutdown(self):
        pass


class CachedPoolWithShutdown:
    """Decorator for caching the mpi pool when called with the equivalent args,
    and shutting down previous ones when not needed.
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
    """Get the MPI executor pool, with specified number of processes and
    threads per process.
    """
    if (num_workers == 1) and (num_threads == _NUM_THREAD_WORKERS):
        from concurrent.futures import ProcessPoolExecutor
        return ProcessPoolExecutor(1)

    if USE_SYNCRO:
        return SynchroMPIPool()

    from mpi4py.futures import MPIPoolExecutor
    return MPIPoolExecutor(num_workers, main=False,
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
                        "to join comm {}.".format(wait_for_workers, comm))

        comm.Barrier()
        res = self.fn(*args, comm=comm, **kwargs)
        comm.Barrier()
        return res


class SpawnMPIProcessesFunc(object):
    """Automatically wrap a function to be executed in parallel by a
    pool of mpi workers.

    This is only called by the master mpi process in manual mode, only by
    the (non-mpi) spawning process in automatic mode, or by all processes in
    syncro mode.
    """

    def __init__(self, fn):
        self.fn = fn

    def __call__(self, *args,
                 num_workers=None,
                 num_threads=1,
                 mpi_pool=None,
                 spawn_all=USE_SYNCRO or (not ALREADY_RUNNING_AS_MPI),
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

eigs_slepc_mpi = functools.wraps(eigs_slepc)(
    GetMPIBeforeCall(eigs_slepc))
eigs_slepc_spawn = functools.wraps(eigs_slepc)(
    SpawnMPIProcessesFunc(eigs_slepc_mpi))

svds_slepc_mpi = functools.wraps(svds_slepc)(
    GetMPIBeforeCall(svds_slepc))
svds_slepc_spawn = functools.wraps(svds_slepc)(
    SpawnMPIProcessesFunc(svds_slepc_mpi))

mfn_multiply_slepc_mpi = functools.wraps(mfn_multiply_slepc)(
    GetMPIBeforeCall(mfn_multiply_slepc))
mfn_multiply_slepc_spawn = functools.wraps(mfn_multiply_slepc)(
    SpawnMPIProcessesFunc(mfn_multiply_slepc_mpi))

ssolve_slepc_mpi = functools.wraps(ssolve_slepc)(
    GetMPIBeforeCall(ssolve_slepc))
ssolve_slepc_spawn = functools.wraps(ssolve_slepc)(
    SpawnMPIProcessesFunc(ssolve_slepc_mpi))
