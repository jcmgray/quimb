"""Generate specific tensor network states and operators.
"""
import functools

import numpy as np

from ..accel import make_immutable
from ..linalg.base_linalg import norm_fro_dense
from ..gen.operators import spin_operator, eye
from .tensor_core import Tensor
from .tensor_1d import MatrixProductState, MatrixProductOperator


def rand_tensor(shape, inds, tags=None):
    """Generate a random (complex) tensor with specified shape and inds.
    """
    data = np.random.randn(*shape) + 1.0j * np.random.randn(*shape)
    return Tensor(data=data, inds=inds, tags=tags)


def MPS_rand(n, bond_dim, phys_dim=2,
             site_ind_id='k{}',
             site_tag_id='i{}',
             tags=None,
             bond_name="",
             normalize=True,
             **kwargs):
    """Generate a random matrix product state.

    Parameters
    ----------
    bond_dim : int
        The bond dimension.
    phys_dim : int, optional
        The physical (site) dimensions, defaults to 2.
    site_ind_id : sequence of hashable, or str
        See :class:`~quimb.tensor.tensor_1d.MatrixProductState`.
    site_tag_id=None, optional
        See :class:`~quimb.tensor.tensor_1d.MatrixProductState`.
    tags=None, optional
        See :class:`~quimb.tensor.tensor_1d.MatrixProductState`.
    bond_name : str, optional
        See :class:`~quimb.tensor.tensor_1d.MatrixProductState`.
    """
    shapes = [(bond_dim, phys_dim),
              *((bond_dim, bond_dim, phys_dim),) * (n - 2),
              (bond_dim, phys_dim)]

    arrays = \
        map(lambda x: x / norm_fro_dense(x)**(1 / (x.ndim - 1)),
            map(lambda x: np.random.randn(*x) + 1.0j * np.random.randn(*x),
                shapes))

    rmps = MatrixProductState(arrays, site_ind_id=site_ind_id,
                              bond_name=bond_name, site_tag_id=site_tag_id,
                              tags=tags, **kwargs)

    if normalize:
        rmps.site[-1] /= (rmps.H @ rmps)**0.5

    return rmps


@functools.lru_cache(128)
def mpo_site_ham_heis(j=1.0, bz=0.0):
    """Single site of the spin-1/2 Heisenberg Hamiltonian in MPO form.
    This is cached.

    Parameters
    ----------
    j : float
        (Isotropic) nearest neighbour coupling.
    bz : float
        Magnetic field in Z-direction.

    Returns
    -------
    H : numpy.ndarray
        The tensor, with shape (5, 5, 2, 2).
    """
    H = np.zeros((5, 5, 2, 2), dtype=complex)

    for i, s in enumerate('XYZ'):
        H[1 + i, 0, :, :] = spin_operator(s)
        H[-1, 1 + i, :, :] = j * spin_operator(s)

    H[0, 0, :, :] = eye(2)
    H[4, 4, :, :] = eye(2)
    H[4, 0, :, :] = - bz * spin_operator('Z')

    make_immutable(H)
    return H


def mpo_end_ham_heis_left(j=1.0, bz=0.0):
    """The left most site of a open boundary conditions Heisenberg
    matrix product operator.
    """
    return mpo_site_ham_heis(j=j, bz=bz)[-1, :, :, :]


def mpo_end_ham_heis_right(j=1.0, bz=0.0):
    """The right most site of a open boundary conditions Heisenberg
    matrix product operator.
    """
    return mpo_site_ham_heis(j=j, bz=bz)[:, 0, :, :]


def MPO_ham_heis(n, j=1.0, bz=0.0,
                 upper_ind_id='k{}',
                 lower_ind_id='b{}',
                 site_tag_id='i{}',
                 tags=None,
                 bond_name=""):
    """Heisenberg Hamiltonian in matrix product operator form.
    """
    arrays = (mpo_end_ham_heis_left(j=j, bz=bz),
              *[mpo_site_ham_heis(j=j, bz=bz)] * (n - 2),
              mpo_end_ham_heis_right(j=j, bz=bz))

    HH_mpo = MatrixProductOperator(arrays=arrays,
                                   upper_ind_id=upper_ind_id,
                                   lower_ind_id=lower_ind_id,
                                   site_tag_id=site_tag_id,
                                   tags=tags,
                                   bond_name=bond_name)
    return HH_mpo
