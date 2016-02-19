"""
Functions for generating random quantum objects and states.
"""
import numpy as np
from quijy.core import (qjf, ptr, kron, rdmul, nmlz)


def rand_matrix(d, scaled=False):
    """
    Generate a random complex matrix of order `d` with normally distributed
    entries. If `scaled` is `True`, then in the limit of large `d` the
    eigenvalues will be distributed on the unit complex disk.
    """
    mat = np.random.randn(d, d) + 1.0j*np.random.randn(d, d)
    if scaled:
        mat /= (2 * d)**0.5
    return np.matrix(mat, copy=False)


def rand_herm(d):
    """
    Generate a random hermitian matrix of order `d` with normally distributed
    entries. In the limit of large `d` the spectrum will be a semi-circular distribution between [-1, 1].
    """
    herm = rand_matrix(d) / (2**2 * d**0.5)
    return herm + herm.H


def rand_pos(d):
    """
    Generate a random positive matrix of order `d`, with normally distributed
    entries. In the limit of large `d` the spectrum will lie between [0, 1].
    """
    pos = rand_matrix(d) / (2**1.5 * d**0.5)
    return pos @ pos.H


def rand_rho(d):
    """
    Generate a random positive matrix of order `d` with normally distributed entries and unit trace.
    """
    return nmlz(rand_pos(d))


def rand_ket(d):
    """
    Generates a ket of length `d` with normally distributed entries.
    """
    return qjf(np.random.randn(d, 1) + 1.0j * np.random.randn(d, 1),
               normalized=True)


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
