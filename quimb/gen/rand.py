"""Functions for generating random quantum objects and states.
"""

# TODO: Test density -------------------------------------------------------- #
# TODO: make sure eigen spectrum is correct ... ----------------------------- #

from functools import reduce
import numpy as np
import scipy.sparse as sp
from ..accel import rdmul, dot, matrixify
from ..core import qu, ptr, kron, nmlz


def rand_matrix(d, scaled=True, sparse=False, stype='csr', density=None):
    """Generate a random complex matrix of order `d` with normally distributed
    entries. If `scaled` is `True`, then in the limit of large `d` the
    eigenvalues will be distributed on the unit complex disk.

    Parameters
    ----------
        d: matrix dimension
        scaled: whether to scale the matrices values such that its spectrum
            approximately lies on the unit disk (for dense matrices)
        sparse: whether to produce a sparse matrix
        stype: the type of sparse matrix if so
        density: target density for the sparse matrix

    Returns
    -------
        mat: random matrix
    """
    if sparse:
        # Aim for 10 non-zero values per row, but betwen 1 and d/2
        density = 10 / d if density is None else density
        density = min(max(density, d**-2), 1 - d**-2)
        mat = sp.random(d, d, format=stype, density=density)
        mat.data = np.random.randn(mat.nnz) + 1.0j * np.random.randn(mat.nnz)
    else:
        density = 1.0
        mat = np.random.randn(d, d) + 1.0j * np.random.randn(d, d)
        mat = np.asmatrix(mat)
    if scaled:
        mat /= (2 * d * density)**0.5
    return mat


def rand_herm(d, sparse=False, density=None):
    """Generate a random hermitian matrix of order `d` with normally
    distributed entries. In the limit of large `d` the spectrum will be a
    semi-circular distribution between [-1, 1].
    """
    if sparse:
        density = 10 / d if density is None else density
        density = min(max(density, d**-2), 1 - d**-2)
        density /= 2  # to account of herm construction
    herm = rand_matrix(d, scaled=True, sparse=sparse,
                       density=density) / (2**1.5)
    herm += herm.H
    return herm


def rand_pos(d, sparse=False, density=None):
    """Generate a random positive matrix of order `d`, with normally
    distributed entries. In the limit of large `d` the spectrum will lie
    between [0, 1].
    """
    if sparse:
        density = 10 / d if density is None else density
        density = min(max(density, d**-2), 1 - d**-2)
        density = 0.5 * (density / d)**0.5  # to account for pos construction
    pos = rand_matrix(d, scaled=True, sparse=sparse, density=density)
    return dot(pos, pos.H)


def rand_rho(d, sparse=False, density=None):
    """Generate a random positive matrix of order `d` with normally
    distributed entries and unit trace.
    """
    return nmlz(rand_pos(d, sparse=sparse, density=density))


def rand_ket(d, sparse=False, stype='csr', density=0.01):
    """Generates a ket of length `d` with normally distributed entries.
    """
    if sparse:
        ket = sp.random(d, 1, format=stype, density=density)
        ket.data = np.random.randn(ket.nnz) + 1.0j * np.random.randn(ket.nnz)
    else:
        ket = np.asmatrix(np.random.randn(d, 1) +
                          1.0j * np.random.randn(d, 1))
    return nmlz(ket)


def rand_uni(d):
    """Generate a random unitary matrix of order `d`, distributed according to
    the Haar measure.
    """
    q, r = np.linalg.qr(rand_matrix(d))
    r = np.diagonal(r)
    r = r / np.abs(r)
    return rdmul(q, r)


def rand_haar_state(d):
    """Generate a random state of dimension `d` according to the Haar
    distribution.
    """
    u = rand_uni(d)
    return u[:, 0]


def gen_rand_haar_states(d, reps):
    """Generate many random Haar states, recycling a random unitary matrix
    by using all of its columns (not a good idea?).
    """
    for rep in range(reps):
        cyc = rep % d
        if cyc == 0:
            u = rand_uni(d)
        yield u[:, cyc]


def rand_mix(d, tr_d_min=None, tr_d_max=None, mode='rand'):
    """Constructs a random mixed state by tracing out a random ket
    where the composite system varies in size between 2 and d. This produces
    a spread of states including more purity but has no real meaning.
    """
    if tr_d_min is None:
        tr_d_min = 2
    if tr_d_max is None:
        tr_d_max = d

    m = np.random.randint(tr_d_min, tr_d_max)
    if mode == 'rand':
        psi = rand_ket(d * m)
    elif mode == 'haar':
        psi = rand_haar_state(d * m)

    return ptr(psi, [d, m], 0)


def rand_product_state(n, qtype=None):
    """Generates a ket of `n` many random pure qubits.
    """
    def gen_rand_pure_qubits(n):
        for _ in range(n):
            u = np.random.rand()
            v = np.random.rand()
            phi = 2 * np.pi * u
            theta = np.arccos(2 * v - 1)
            yield qu([[np.cos(theta / 2.0)],
                      [np.sin(theta / 2.0) * np.exp(1.0j * phi)]],
                     qtype=qtype)
    return kron(*gen_rand_pure_qubits(n))


@matrixify
def rand_matrix_product_state(phys_dim, n, bond_dim,
                              cyclic=False, trans_invar=False):
    """Generate a random matrix product state (in dense form).

    Parameters
    ----------
        phys_dim : int
            Physical dimension of each local site.
        n : int
            Number of sites.
        bond_dim : int
            Dimension of the bond (virtual) indices.
        cyclic : bool (optional)
            Whether to impose cyclic boundary conditions on the entanglement
            structure.
        trans_invar : bool (optional)
            Whether to generate a translationally invariant state,
            requires cyclic=True.

    Returns
    -------
        ket : matrix-like
            The random state, with shape (phys_dim**n, 1)

    """
    if trans_invar and not cyclic:
        raise ValueError("State cannot be translationally invariant"
                         "with open boundary conditions.")
    elif trans_invar:
        raise NotImplementedError

    tensor_shp = (bond_dim, phys_dim, bond_dim)

    def gen_tensors():
        for i in range(0, n):
            shape = (tensor_shp[1:] if i == 0 and not cyclic else
                     tensor_shp[:-1] if i == n - 1 and not cyclic else
                     tensor_shp)

            yield np.random.randn(*shape) + 1.0j * np.random.randn(*shape)

    ket_tens = reduce(lambda x, y: np.tensordot(x, y, axes=1), gen_tensors())
    if cyclic:
        ket_tens = np.trace(ket_tens, axis1=0, axis2=-1)

    norm = np.tensordot(ket_tens, ket_tens.conj(), n)
    ket_tens /= norm**0.5
    return ket_tens.reshape((phys_dim**n, 1))


rand_mps = rand_matrix_product_state


def rand_seperable(dims, num_mix=10):
    """Generate a random, mixed, seperable state. E.g rand_seperable([2, 2])
    for a mixed two qubit state with no entanglement.

    Parameters
    ----------
        dims : tuple of int
            The local dimensions across which to be seperable.
        num_mix : int, optional
            How many individual product states to sum together, each with
            random weight.

    Returns
    -------
        np.matrix
            Mixed seperable state.
    """

    def gen_single_sites():
        for dim in dims:
            yield rand_rho(dim)

    weights = np.random.rand(num_mix)

    def gen_single_states():
        for w in weights:
            yield w * kron(*gen_single_sites())

    return sum(gen_single_states()) / np.sum(weights)
