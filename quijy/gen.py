"""
Functions for generating quantum objects.
TODO: add sparse and qtype to all relevant functions.
"""
# TODO: Graph states, cluster states, multidimensional

from itertools import product, permutations
from functools import lru_cache
from math import factorial
import numpy as np
import scipy.sparse as sp
from quijy.core import (qjf, kron, kronpow, eyepad, eye,
                        eyepad, levi_civita)


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
    return qjf([[1],
                [0]], **kwargs)

zplus = up


def down(**kwargs):
    """ Returns down-state, aka. |1>, -Z eigenstate."""
    return qjf([[0],
                [1]], **kwargs)

zminus = down


def plus(**kwargs):
    """ Returns plus-state, aka. |+>, +X eigenstate."""
    return qjf([[2**-0.5],
                [2**-0.5]], **kwargs)

xplus = plus


def minus(**kwargs):
    """ Returns minus-state, aka. |->, -X eigenstate."""
    return qjf([[2**-0.5],
                [-2**-0.5]], **kwargs)

xminus = minus


def yplus(**kwargs):
    """ Returns yplus-state, aka. |y+>, +Y eigenstate."""
    return qjf([[2**-0.5],
                [1.0j / (2**0.5)]], **kwargs)


def yminus(**kwargs):
    """ Returns yplus-state, aka. |y->, -Y eigenstate."""
    return qjf([[2**-0.5],
                [-1.0j / (2**0.5)]], **kwargs)


@lru_cache(maxsize=48)
def sig(xyz, dim=2, **kwargs):
    """
    Generates the spin operators for spin 1/2 or 1.

    Parameters
    ----------
        xyz: which spatial direction
        dim: dimension of spin operator (e.g. 3 for spin-1)

    Returns
    -------
        spin operator, quijified.
    """
    xyzmap = {
        0: 'i', 'i': 'i', 'I': 'i',
        1: 'x', 'x': 'x', 'X': 'x',
        2: 'y', 'y': 'y', 'Y': 'y',
        3: 'z', 'z': 'z', 'Z': 'z'
    }
    opmap = {
        ('x', 2): lambda: qjf([[0, 1],
                               [1, 0]], **kwargs),
        ('x', 3): lambda: qjf([[0, 1, 0],
                               [1, 0, 1],
                               [0, 1, 0]], **kwargs) / 2**0.5,
        ('y', 2): lambda: qjf([[0, -1j],
                               [1j, 0]], **kwargs),
        ('y', 3): lambda: qjf([[0, -1j, 0],
                               [1j, 0, -1j],
                               [0, 1j, 0]], **kwargs) / 2**0.5,
        ('z', 2): lambda: qjf([[1, 0],
                               [0, -1]], **kwargs),
        ('z', 3): lambda: qjf([[1, 0, 0],
                               [0, 0, 0],
                               [0, 0, -1]], **kwargs),
        ('i', 2): lambda: eye(2, **kwargs),
        ('i', 3): lambda: eye(3, **kwargs),
    }
    return opmap[(xyzmap[xyz], dim)]()


@lru_cache(maxsize=8)
def bell_state(s, **kwargs):
    """
    Generates one of the four bell-states;
    0: phi+, 1: phi-, 2: psi+, 3: psi- (singlet)
    """
    keymap = {
        'psi-': 'psi-', 0: 'psi-',
        'psi+': 'psi+', 1: 'psi+',
        'phi-': 'phi-', 2: 'phi-',
        'phi+': 'phi+', 3: 'phi+',
    }
    c = 2.0**-0.5
    statemap = {
        'psi-': lambda: qjf([[0], [c], [-c], [0]], **kwargs),
        'phi+': lambda: qjf([[c], [0], [0], [c]], **kwargs),
        'phi-': lambda: qjf([[c], [0], [0], [-c]], **kwargs),
        'psi+': lambda: qjf([[0], [c], [c], [0]], **kwargs)
    }
    return statemap[keymap[s]]()


def singlet(**kwargs):
    """ Alias for one of bell-states """
    return bell_state('psi-', **kwargs)


def triplets(**kwargs):
    """ Equal mixture of the three triplet bell_states """
    return eye(4, **kwargs) - singlet(qtype='dop', **kwargs)


def bloch_state(ax, ay, az, purify=False, **kwargs):
    if purify:
        ax, ay, az = np.array([ax, ay, az]) / (ax**2 + ay**2 + az**2)**0.5
    return sum(0.5 * a * sig(s, **kwargs)
               for a, s in zip((1, ax, ay, az), 'ixyz'))


def neel_state(n):
    binary = '01' * (n / 2)
    binary += (n % 2 == 1) * '0'  # add trailing spin for odd n
    return basis_vec(int(binary, 2), 2 ** n)


def singlet_pairs(n):
    return kronpow(bell_state(3), (n // 2))


def werner_state(p):
    return p * bell_state(3) @ bell_state(3).H + (1 - p) * eye(4) / 4


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
        ham = ham + eyepad(sds, dims, [i, i + 1])
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
        for coo in coosj1:
            if abs(coo[1] - coo[0]) == 1:  # can sum then tensor
                yield eyepad(sum(op & op for op in s), dims, coo)
            else:  # tensor then sum (slower)
                yield sum(eyepad(op, dims, coo) for op in s)
    def gen_j2():
        for coo in coosj2:
            if abs(coo[1] - coo[0]) == 2:  # can add then tensor
                yield eyepad(sum(op & eye(2) & op for op in s), dims, coo)
            else:
                yield sum(eyepad(op, dims, coo) for op in s)

    gen_bz = (eyepad([s[2]], dims, i) for i in range(n))

    ham = j1 * sum(gen_j1()) + j2 * sum(gen_j2())
    if bz != 0:
        ham += bz * sum(gen_bz)
    return ham if sparse else ham.todense()


def ham_majumdar_ghosh(n, j1=1.0, j2=0.5, **kwargs):
    """ Alias for ham-j1j2. """
    return ham_j1j2(n, j1=j1, j2=j2, **kwargs)


def multi_singlet(n):
    es = [basis_vec(i, n) for i in range(n)]
    vec_perm = permutations(es)
    ind_perm = permutations(range(n))

    def terms():
        for vec, ind in zip(vec_perm, ind_perm):
            yield levi_civita(ind) * kron(*vec)

    return sum(terms()) / factorial(n)**0.5
