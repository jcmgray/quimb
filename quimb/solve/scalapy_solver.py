import numpy as np
from .slepc_solver import get_default_comm


def squarest_factors(n):
    """Return the pair of integers that factorise `n` but are closest
    to its square root.
    """
    return next(
        (i, n // i)
        for i in range(int(n**0.5), 0, -1)
        if n % i == 0
    )


def get_scalapy_and_params(pn=None, bsz=None, comm=None):
    """
    """
    import scalapy

    if comm is None:
        comm = get_default_comm()

    if pn is None:
        size = comm.Get_size()
        pn = squarest_factors(size)

    if bsz is None:
        bsz = (32, 32)

    scalapy.initmpi(pn, bsz)
    return scalapy, comm, pn, bsz


def convert_mat_to_scalapy(mat, pn=None, bsz=None, comm=None):
    """
    """
    sclp, *_ = get_scalapy_and_params(pn=pn, bsz=bsz, comm=comm)
    smat = sclp.DistributedMatrix(mat.shape, dtype=mat.dtype.type)
    ri, ci = smat.indices()
    smat.local_array[:, :] = mat[ri, ci]
    return smat


def eigsys_scalapy(a, return_vecs=True, pn=None, bsz=None, comm=None,
                   k=None, isherm=True, sort=True, **kwargs):
    """
    """
    if not isherm:
        raise NotImplementedError

    sclp, comm, pn, bsz = get_scalapy_and_params(pn=pn, bsz=bsz, comm=comm)
    sa = convert_mat_to_scalapy(a, pn=pn, bsz=bsz, comm=comm)

    defaults = {'overwrite_a': False,
                'eigvals_only': not return_vecs,
                'eigvals': (0, k - 1) if k else None}

    elev = sclp.eigh(sa, **{**defaults, **kwargs})

    rank = comm.Get_rank()

    if return_vecs:
        elev = elev[0], elev[1].to_global_array(rank=0)

    if rank == 0:
        if return_vecs:
            return elev[0], np.asmatrix(elev[1])
        return elev
