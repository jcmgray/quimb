"""
Functions for generating random quantum objects and states.
"""
import numpy as np
from quijy.core import (qjf, ptr, kron, rdmul, nmlz)


def rand_matrix(n, nmlzd=False):
    """
    Generate a random complex matrix of order `n` with normally-distributed
    coeffs. If `nmlzd` is `True`, then in the limit of large `n` the
    eigenvalues will be distributed on the unit complex disk.
    """
    mat = np.random.randn(n, n) + 1.0j*np.random.randn(n, n)
    if nmlzd:
        mat /= (2 * n)**0.5
    return np.matrix(mat, copy=False)


def rand_herm(n):
    """
    Generates a random hermitian matrix of order `n` with Gaussian entries and
    spectrum of semi-circular distribution between [~ -1, ~ 1].
    """
    herm = rand_matrix(n) / (2**2 * n**0.5)
    return herm + herm.H


def rand_pos(n):
    """
    Generates a random positive matrix of order `n`, with Gaussian entries and
    spectrum between [0, ~1].
    """
    pos = rand_matrix(n) / (2**1.5 * n**0.5)
    return pos * pos.H


def rand_rho(n):
    """
    Generate a random positive matrix of order `n` with Gaussian entries and
    unit trace.
    """
    return nmlz(rand_pos(n))


def rand_ket(n):
    """
    Generates a wavefunction with random, Gaussian coefficients, normalised.
    """
    return qjf(np.random.randn(n, 1) + 1.0j * np.random.randn(n, 1),
               normalized=True)


def rand_uni(n):
    """ Generate a random unitary matrix, distributed according to
    the Haar measure. """
    q, r = np.linalg.qr(rand_matrix(n))
    r = np.diagonal(r)
    r = r / np.abs(r)
    return rdmul(q, r)


def rand_haar_state(n):
    u = rand_uni(n)
    return u[:, 0]


def gen_rand_haar_states(n, reps):
    """
    Generate many random Haar states, recycling a random unitary matrix
    by using all of its columns (not a good idea).
    """
    for rep in range(reps):
        cyc = rep % n
        if cyc == 0:
            u = rand_uni(n)
        yield u[:, cyc]


def rand_mix(n):
    """
    Constructs a random mixed state by tracing out a random gaussian ket
    where the composite system varies in size between 2 and n. This produces
    a spread of states including more purity but has no real meaning.
    """
    m = np.random.randint(2, n+1)
    psi = rand_ket(n*m)
    return ptr(psi, [n, m], 0)


def rand_product_state(n, qtype=None):
    """
    Generates a wavefunction of `n` many random pure qubits.
    """
    # Generator
    def calc_rand_pure_qubits(n):
        for i in range(n):
            u = np.random.rand()
            v = np.random.rand()
            phi = 2 * np.pi * u
            theta = np.arccos(2 * v - 1)
            yield qjf([[np.cos(theta / 2.0)],
                       [np.sin(theta / 2.0) * np.exp(1.0j * phi)]],
                      qtype=qtype)
    return kron(*[x for x in calc_rand_pure_qubits(n)])
