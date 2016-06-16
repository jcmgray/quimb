"""
Functions for generating random quantum objects and states.
"""
import numpy as np
import scipy.sparse as sp
from ..accel import rdmul
from ..core import qjf, ptr, kron, nmlz


def rand_matrix(d, scaled=True, sparse=False, format='csr', density=0.01):
    """
    Generate a random complex matrix of order `d` with normally distributed
    entries. If `scaled` is `True`, then in the limit of large `d` the
    eigenvalues will be distributed on the unit complex disk.
    """
    if sparse:
        mat = sp.random(d, d, format=format, density=density)
        mat.data = np.random.randn(mat.nnz) + 1.0j * np.random.randn(mat.nnz)
    else:
        density = 1.0
        mat = np.random.randn(d, d) + 1.0j * np.random.randn(d, d)
        mat = np.asmatrix(mat)
    if scaled:
        mat /= (2 * d * density)**0.5
    return mat


def rand_herm(d, sparse=False, density=0.01):
    """
    Generate a random hermitian matrix of order `d` with normally distributed
    entries. In the limit of large `d` the spectrum will be a semi-circular
    distribution between [-1, 1].
    """
    density /= 2
    herm = rand_matrix(d, scaled=True, sparse=sparse,
                       density=density) / (2**1.5)
    herm += herm.H
    return herm


def rand_pos(d, sparse=False, density=0.01):
    """
    Generate a random positive matrix of order `d`, with normally distributed
    entries. In the limit of large `d` the spectrum will lie between [0, 1].
    """
    density = (density / d)**0.5
    pos = rand_matrix(d, scaled=True, sparse=sparse, density=density) / 2
    return pos @ pos.H


def rand_rho(d, sparse=False, density=0.01):
    """
    Generate a random positive matrix of order `d` with normally distributed
    entries and unit trace.
    """
    return nmlz(rand_pos(d, sparse=sparse, density=density))


def rand_ket(d, sparse=False, format='csr', density=0.01):
    """
    Generates a ket of length `d` with normally distributed entries.
    """
    if sparse:
        ket = sp.random(d, 1, format=format, density=density)
        ket.data = np.random.randn(ket.nnz) + 1.0j * np.random.randn(ket.nnz)
    else:
        ket = np.asmatrix(np.random.randn(d, 1) +
                          1.0j * np.random.randn(d, 1))
    return nmlz(ket)


def rand_uni(d):
    """
    Generate a random unitary matrix of order `d`, distributed according to
    the Haar measure.
    """
    q, r = np.linalg.qr(rand_matrix(d))
    r = np.diagonal(r)
    r = r / np.abs(r)
    return rdmul(q, r)


def rand_haar_state(d):
    u = rand_uni(d)
    return u[:, 0]


def gen_rand_haar_states(d, reps):
    """
    Generate many random Haar states, recycling a random unitary matrix
    by using all of its columns (not a good idea?).
    """
    for rep in range(reps):
        cyc = rep % d
        if cyc == 0:
            u = rand_uni(d)
        yield u[:, cyc]


def rand_mix(d):
    """
    Constructs a random mixed state by tracing out a random ket
    where the composite system varies in size between 2 and d. This produces
    a spread of states including more purity but has no real meaning.
    """
    m = np.random.randint(2, d+1)
    psi = rand_ket(d * m)
    return ptr(psi, [d, m], 0)


def rand_product_state(n, qtype=None):
    """
    Generates a ket of `n` many random pure qubits.
    """
    def gen_rand_pure_qubits(n):
        for i in range(n):
            u = np.random.rand()
            v = np.random.rand()
            phi = 2 * np.pi * u
            theta = np.arccos(2 * v - 1)
            yield qjf([[np.cos(theta / 2.0)],
                       [np.sin(theta / 2.0) * np.exp(1.0j * phi)]],
                      qtype=qtype)
    return kron(*gen_rand_pure_qubits(n))
