"""Generate specific tensor network states and operators.
"""
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


def build_spin_ham_mpo_tensors(one_site_terms, two_site_terms,
                               S=1 / 2, which=None):
    """Genereate a spin hamiltonian MPO tensor

    Parameters
    ----------
    one_site_terms : sequence of (scalar, operator)
        The terms that act on a single site, each ``operator`` can be a string
        suitable to be sent to :func:`spin_operator` or an actual 2d-array.
    two_site_terms : sequence of (scalar, operator operator)
        The terms that act on two neighbouring sites, each ``operator`` can be
        a string suitable to be sent to :func:`spin_operator` or an actual
        2d-array.
    S : fraction, optional
        What size spin to use, defaults to spin-1/2.
    which : {None, 'L', 'R', 'A'}, optional
        If ``None``, generate the middle tensor, if 'L' a left-end tensor, if
        'R' a right-end tensor and if 'A' all three.

    Returns
    -------
    numpy.ndarray{, numpy.ndarray, numpy.ndarray}
    """
    # local dimension
    D = int(2 * S + 1)
    # bond dimension
    B = len(two_site_terms) + 2

    H = np.zeros((B, B, D, D), dtype=complex)

    # add two-body terms
    for i, (factor, s1, s2) in enumerate(two_site_terms):
        if isinstance(s1, str):
            s1 = spin_operator(s1, S=S)
        if isinstance(s2, str):
            s2 = spin_operator(s2, S=S)
        H[1 + i, 0, :, :] = s1
        H[-1, 1 + i, :, :] = factor * s2

    # add one-body terms
    for factor, s in one_site_terms:
        if isinstance(s, str):
            s = spin_operator(s, S=S)
        H[B - 1, 0, :, :] += factor * s

    H[0, 0, :, :] = eye(D)
    H[B - 1, B - 1, :, :] = eye(D)

    make_immutable(H)

    if which == 'L':
        return H[-1, :, :, :]
    elif which == 'R':
        return H[:, 0, :, :]
    elif which == 'A':
        return H[-1, :, :, :], H, H[:, 0, :, :]

    return H


class MPOSpinHam:
    """
    """

    def __init__(self, S=1 / 2):
        self.S = S
        self.one_site_terms = []
        self.two_site_terms = []

    def add_term(self, factor, *operators):
        """
        """
        if len(operators) == 1:
            self.one_site_terms.append((factor, *operators))
        elif len(operators) == 2:
            self.two_site_terms.append((factor, *operators))
        else:
            raise NotImplementedError("3-body+ terms are not supported yet.")

    def build(self, n, upper_ind_id='k{}', lower_ind_id='b{}',
              site_tag_id='i{}', tags=None, bond_name=""):
        """
        """
        left, middle, right = build_spin_ham_mpo_tensors(
            self.one_site_terms, self.two_site_terms, S=self.S, which='A')

        arrays = (left, *[middle] * (n - 2), right)

        return MatrixProductOperator(arrays=arrays, bond_name=bond_name,
                                     upper_ind_id=upper_ind_id,
                                     lower_ind_id=lower_ind_id,
                                     site_tag_id=site_tag_id, tags=tags)


def MPO_ham_ising(n, j=1.0, bx=0.0,
                  upper_ind_id='k{}',
                  lower_ind_id='b{}',
                  site_tag_id='i{}',
                  tags=None,
                  bond_name=""):
    """Ising Hamiltonian in matrix product operator form.
    """
    H = MPOSpinHam(S=1 / 2)
    H.add_term(j, 'Z', 'Z')
    H.add_term(-bx, 'X')
    return H.build(n, site_tag_id=site_tag_id, tags=tags, bond_name=bond_name,
                   upper_ind_id=upper_ind_id, lower_ind_id=lower_ind_id)


def MPO_ham_XY(n, j=1.0, bz=0.0,
               upper_ind_id='k{}',
               lower_ind_id='b{}',
               site_tag_id='i{}',
               tags=None,
               bond_name=""):
    """XY-Hamiltonian in matrix product operator form.
    """
    H = MPOSpinHam(S=1 / 2)
    H.add_term(j, 'X', 'X')
    H.add_term(j, 'Y', 'Y')
    H.add_term(-bz, 'Z')
    return H.build(n, site_tag_id=site_tag_id, tags=tags, bond_name=bond_name,
                   upper_ind_id=upper_ind_id, lower_ind_id=lower_ind_id)


def MPO_ham_heis(n, j=1.0, bz=0.0,
                 upper_ind_id='k{}',
                 lower_ind_id='b{}',
                 site_tag_id='i{}',
                 tags=None,
                 bond_name=""):
    """Heisenberg Hamiltonian in matrix product operator form.
    """
    H = MPOSpinHam(S=1 / 2)
    H.add_term(j, 'X', 'X')
    H.add_term(j, 'Y', 'Y')
    H.add_term(j, 'Z', 'Z')
    H.add_term(-bz, 'Z')
    return H.build(n, site_tag_id=site_tag_id, tags=tags, bond_name=bond_name,
                   upper_ind_id=upper_ind_id, lower_ind_id=lower_ind_id)
