# import functools
import numpy as np
from .slepc_solver import get_default_comm


def get_scalapy(pn=None, bsz=None, comm=None):
    """
    """
    import scalapy

    if comm is None:
        comm = get_default_comm()

    if pn is None:
        size = comm.Get_size()
        pn = int(size**0.5)
        pn = (pn, pn)

    if bsz is None:
        bsz = (32, 32)

    scalapy.initmpi(pn, bsz)
    return scalapy, comm


def convert_to_scalapy(mat, pn=None, bsz=None, comm=None):
    """
    """
    sclp, _ = get_scalapy(pn=pn, bsz=bsz, comm=comm)
    smat = sclp.DistributedMatrix(mat.shape, dtype=mat.dtype.type)
    ri, ci = smat.indices()
    smat.local_array[:, :] = mat[ri, ci]
    return smat


def scalapy_eigsys(a, return_vecs=True, pn=None, bsz=None, comm=None,
                   k=None, **kwargs):
    """
    """
    rank = comm.Get_rank()

    sclp, comm = get_scalapy(pn=pn, bsz=bsz, comm=comm)
    sa = convert_to_scalapy(a, pn=pn, bsz=bsz, comm=comm)

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
