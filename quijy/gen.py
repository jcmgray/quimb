"""
Functions for generating quantum objects.
TODO: add sparse and qtype to all relevant functions.
"""
# TODO: Graph states, cluster states, multidimensional

import numpy as np
import scipy.sparse as sp
from itertools import product
from quijy.core import (qjf, nmlz, kron, kronpow, eyepad, eye, trx, eyeplace)


def basis_vec(dir, dim, sparse=False):
def basis_vec(dir, dim, sparse=False, **kwargs):
    """
    Constructs a unit vector ket.

    Parameters
    ----------
        dir: which dimension the key should point in
        dim: total number of dimensions
        sparse: return vector as sparse-csr matrix

    Returns:
        x:
    """
    if sparse:
        return sp.csr_matrix(([1.0], ([dir], [0])),
                             dtype=complex, shape=(dim, 1))
    else:
        x = np.zeros([dim, 1], dtype=complex)
        x[dir] = 1.0
    return qjf(x, **kwargs)


def up(**kwargs):
    """ Returns up-state, aka. |0>, +Z eigenstate."""
    return qjf([[1], [0]], **kwargs)

zplus = up


def down(**kwargs):
    """ Returns down-state, aka. |1>, -Z eigenstate."""
    return qjf([[0], [1]], **kwargs)

zminus = down


def plus(**kwargs):
    """ Returns plus-state, aka. |+>, +X eigenstate."""
    return qjf([[2**-0.5], [2**-0.5]], **kwargs)

xplus = plus


def minus(**kwargs):
    """ Returns minus-state, aka. |->, -X eigenstate."""
    return qjf([[2**-0.5], [-2**-0.5]], **kwargs)

xminus = minus


def yplus(**kwargs):
    """ Returns yplus-state, aka. |y+>, +Y eigenstate."""
    return qjf([[2**-0.5], [1.0j / (2**0.5)]], **kwargs)


def yminus(**kwargs):
    """ Returns yplus-state, aka. |y->, -Y eigenstate."""
    return qjf([[2**-0.5], [-1.0j / (2**0.5)]], **kwargs)


def sig(s, **kwargs):
    """
    Generates one of the three Pauli matrices, 0-I, 1-X, 2-Y, 3-Z
    """
    if s in (1, 'x', 'X'):
        return qjf([[0, 1], [1, 0]], **kwargs)
    elif s in (2, 'y', 'Y'):
        return qjf([[0, -1j], [1j, 0]], **kwargs)
    elif s in (3, 'z', 'Z'):
        return qjf([[1, 0], [0, -1]], **kwargs)
    elif s in (0, 'i', 'I'):
        return qjf([[1, 0], [0, 1]], **kwargs)


def bell_state(s, **kwargs):
    """
    Generates one of the four bell-states;
    0: phi+, 1: phi-, 2: psi+, 3: psi- (singlet)
    """
    isqr2 = 2.0**-0.5
    if s in (3, 'psi-'):
        return qjf([[0], [isqr2], [-isqr2], [0]], **kwargs)
    elif s in (0, 'phi+'):
        return qjf([[isqr2], [0], [0], [isqr2]], **kwargs)
    elif s in (1, 'phi-'):
        return qjf([[isqr2], [0], [0], [-isqr2]], **kwargs)
    elif s in (2, 'psi+'):
        return qjf([[0], [isqr2], [isqr2], [0]], **kwargs)


def singlet(**kwargs):
    """ Alias for one of bell-states """
    return bell_state(3, **kwargs)


def triplets(**kwargs):
    """ Equal mixture of the three triplet bell_states """
    return eye(4, **kwargs) - singlet(qtype='dop', **kwargs)


def bloch_state(ax, ay, az, purify=False, **kwargs):
    if purify:
        ax, ay, az = np.array([ax, ay, az]) / (ax**2 + ay**2 + az**2)**0.5
    return sum(0.5 * a * sig(s, **kwargs)
               for a, s in zip((1, ax, ay, az), 'ixyz'))


def rand_ket(n):
    """
    Generates a wavefunction with random coefficients, normalised
    """
    return qjf(np.random.randn(n, 1) + 1.0j * np.random.randn(n, 1),
               normalized=True)


def rand_rho(n):
    """
    Generates a random density matrix of dimension n, no special properties
    other than being guarateed hermitian, positive, and trace 1.
    """
    rho = qjf(np.random.randn(n, n) + 1.0j * np.random.randn(n, n))
    rho = rho + rho.H
    return nmlz(rho * rho)


def rand_mix(n):
    """
    Constructs a random mixed state by tracing out a random gaussian ket
    where the composite system varies in size between 2 and n. This produces
    a spread of states including more purity but has no real grounding.
    """
    m = np.random.randint(2, n+1)
    psi = rand_ket(n*m)
    return trx(psi, [n, m], 0)


def rand_product_state(n, qtype=None):
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
            yield qjf([[np.cos(theta / 2.0)],
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


def w_state(n, sparse=False):
    return sum(basis_vec(2**i, 2**n, sparse=sparse) for i in range(n))/n**0.5


def ham_heis(n, jx=1.0, jy=1.0, jz=1.0, bz=0.0, cyclic=False, sparse=False):
    """ Constructs the heisenberg spin 1/2 hamiltonian
    Parameters:
        n: number of spins
        jx, jy, jz: coupling constants, with convention that positive =
        antiferromagnetic
        bz: z-direction magnetic field
        cyclic: whether to couple the first and last spins
        sparse: whether to return the hamiltonian in sparse form
    Returns:
        ham: hamiltonian as matrix
    """
    dims = [2] * n
    sds = qjf(jx * kron(sig('x'), sig('x')) +
              jy * kron(sig('y'), sig('y')) +
              jz * kron(sig('z'), sig('z')) -
              bz * kron(sig('z'), eye(2)), sparse=True)
    # Begin with last spin, not covered by loop
    ham = eyepad(-bz * sig('z', sparse=True), dims, n - 1)
    for i in range(n - 1):
        ham = ham + eyepad(sds, dims[:-1], i)
    if cyclic:
        ham = ham + eyepad(sig('x', sparse=True), dims, [0, n - 1])  \
                  + eyepad(sig('y', sparse=True), dims, [0, n - 1])  \
                  + eyepad(sig('z', sparse=True), dims, [0, n - 1])
    if not sparse:
        ham = ham.todense()  # always construct sparse though
    return ham


def ham_j1j2(n, j1=1.0, j2=0.5, bz=0.0, cyclic=False, sparse=False):
    """
    Generate the j1-j2 hamiltonian, i.e. next nearest neighbour
    interactions.

    Parameters
    ----------
        n: number of spins
        j1: nearest neighbour coupling strength
        j2: next nearest neighbour coupling strength
        bz: b-field strength in z-direction
        cyclic: cyclic boundary conditions
        sparse: return hamtiltonian as sparse-csr matrix

    Returns
    -------
        ham: Hamtiltonian as matrix
    """
    # TODO: efficient computation by translating operators.
    dims = [2] * n
    coosj1 = np.array([(i, i+1) for i in range(n)])
    coosj2 = np.array([(i, i+2) for i in range(n)])
    s = [sig(i, sparse=True) for i in 'xyz']
    if cyclic:
        coosj1, coosj2 = coosj1 % n, coosj2 % n
    else:
        coosj1 = coosj1[np.all(coosj1 < n, axis=1)]
        coosj2 = coosj2[np.all(coosj2 < n, axis=1)]

    def gen_j1():
        for op, coo in product(s, coosj1):
            yield eyeplace([op], dims, coo, sparse=True)

    def gen_j2():
        for op, coo in product(s, coosj2):
            yield eyeplace([op], dims, coo, sparse=True)

    def gen_bz():
        for i in range(n):
            yield eyeplace([s[2]], dims, i, sparse=True)

    ham = j1 * sum(gen_j1()) + j2 * sum(gen_j2()) + bz * sum(gen_bz())
    return ham if sparse else ham.todense()


def ham_majumdar_ghosh(n, j1=1.0, j2=0.5, **kwargs):
    """ Alias for ham-j1j2. """
    return ham_j1j2(n, j1=j1, j2=j2, **kwargs)


def rand_uni(n):
    x = np.random.randn(n, n) + 1.0j*np.random.randn(n, n)
    x /= 2**0.5
    q, r = np.linalg.qr
