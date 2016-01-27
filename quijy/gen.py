"""
Functions for generating quantum objects.
TODO: add sparse and qtype to all relevant functions.
"""

import numpy as np
import scipy.sparse as sp
from quijy.core import (qonvert, nrmlz, kron, kronpow, eyepad, eye, trx)


def basis_vec(dir, dim, sparse=False):
    """
    Constructs a unit ket that points in dir of total dimensions dim
    """
    if sparse:
        return sp.csr_matrix(([1.0], ([dir], [0])),
                             dtype=complex, shape=(dim, 1))
    else:
        x = np.zeros([dim, 1], dtype=complex)
        x = qonvert(x)  # turn into matrix
        x[dir] = 1.0
    return x


def sig(xyz, sparse=False):
    """
    Generates one of the three Pauli matrices, 0-X, 1-Y, 2-Z
    """
    if xyz in (1, 'x', 'X'):
        return qonvert([[0, 1], [1, 0]], sparse=sparse)
    elif xyz in (2, 'y', 'Y'):
        return qonvert([[0, -1j], [1j, 0]], sparse=sparse)
    elif xyz in (3, 'z', 'Z'):
        return qonvert([[1, 0], [0, -1]], sparse=sparse)
    elif xyz in (0, 'i', 'I'):
        return qonvert([[1, 0], [0, 1]], sparse=sparse)


def bell_state(s, qtype='ket', sparse=False):
    """
    Generates one of the four bell-states;
    0: phi+, 1: phi-, 2: psi+, 3: psi- (singlet)
    """
    isqr2 = 2.0**-0.5
    if s in (3, 'psi-'):
        return qonvert([0, isqr2, -isqr2, 0], qtype=qtype, sparse=sparse)
    elif s in (0, 'phi+'):
        return qonvert([isqr2, 0, 0, isqr2], qtype=qtype, sparse=sparse)
    elif s in (1, 'phi-'):
        return qonvert([isqr2, 0, 0, -isqr2], qtype=qtype, sparse=sparse)
    elif s in (2, 'psi+'):
        return qonvert([0, isqr2, isqr2, 0], qtype=qtype, sparse=sparse)


def singlet(qtype='ket', sparse=False):
    """ Alias for one of bell-states """
    return bell_state(3, qtype=qtype, sparse=sparse)


def triplets(sparse=False):
    """ Equal mixture of the three triplet bell_states """
    return eye(4, sparse=sparse) - singlet('p', sparse)


def bloch_state(ax, ay, az, purify=False, sparse=False):
    if purify:
        ax, ay, az = np.array([ax, ay, az]) / (ax**2 + ay**2 + az**2)**0.5
    rho = 0.5 * (sig('i') + ax * sig('x') + ay * sig('y') + az * sig('z'))
    return rho if not sparse else qonvert(rho, sparse=sparse)


# functions
def random_psi(n):
    """
    Generates a wavefunction with random coefficients, normalised
    """
    return qonvert(np.random.randn(n, 1) + 1.0j * np.random.randn(n, 1),
                   nrmlz=True)


def random_rho(n):
    """
    Generates a random density matrix of dimension n, no special properties
    other than being guarateed hermitian, positive, and trace 1.
    """
    rho = qonvert(np.random.randn(n, n) + 1.0j * np.random.randn(n, n))
    rho = rho + rho.H
    return nrmlz(rho * rho)


def random_product_state(n, qtype=None):
    """
    Calculates the wavefunction of n many random pure qubits.
    """
    # Generator
    def calc_rand_pure_qubits(n):
        for i in range(n):
            u = np.random.rand()
            v = np.random.rand()
            phi = 2 * np.pi * u
            theta = np.arccos(2 * v - 1)
            yield qonvert([[np.cos(theta / 2.0)],
                           [np.sin(theta / 2.0) * np.exp(1.0j * phi)]],
                          qtype=qtype)
    return kron(*[x for x in calc_rand_pure_qubits(n)])


def neel_state(n):
    binary = '01' * (n / 2)
    binary += (n % 2 == 1) * '0'  # add trailing spin for odd n
    return basis_vec(int(binary, 2), 2 ** n)


def singlet_pairs(n):
    return kronpow(bell_state(3), (n // 2))


def werner_state(p):
    return p * bell_state(3) * bell_state(3).H + (1 - p) * eye(4) / 4


def ghz_state(n, sparse=False):
    return (basis_vec(0, 2**n, sparse=sparse) +
            basis_vec(2**n - 1, 2**n, sparse=sparse))/2.0**0.5


def ham_heis(n, jx=1, jy=1, jz=1, bz=0, ring=False, sparse=False):
    """ Constructs the heisenberg spin 1/2 hamiltonian
    Parameters:
        n: number of spins
        jx, jy, jz: coupling constants, with convention that positive =
        antiferromagnetic
        bz: z-direction magnetic field
        ring: whether to couple the first and last spins
        sparse: whether to return the hamiltonian in sparse form
    Returns:
        ham: hamiltonian as matrix
    """
    dims = [2] * n
    sds = (jx * kron(sig('x', sparse=True), sig('x', sparse=True)) +
           jy * kron(sig('y', sparse=True), sig('y', sparse=True)) +
           jz * kron(sig('z', sparse=True), sig('z', sparse=True)) -
           bz * kron(sig('z', sparse=True), eye(2)))
    # Begin with last spin, not covered by loop
    ham = eyepad(-bz * sig('z', sparse=True), dims, n - 1)
    for i in range(n - 1):
        ham = ham + eyepad(sds, dims[:-1], i)
    if ring:
        ham = ham + eyepad(sig('x', sparse=True), dims, [0, n - 1])  \
                  + eyepad(sig('y', sparse=True), dims, [0, n - 1])  \
                  + eyepad(sig('z', sparse=True), dims, [0, n - 1])
    if not sparse:
        ham = ham.todense()  # always construct sparse though
    return ham


def ham_j1j2(n, j1, j2, bz):
    pass


def ham_majumdar_ghosh(n):
    """ Alias for ham-j1j2. """
    pass
