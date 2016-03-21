"""
Functions for generating quantum objects.
"""
# TODO: Graph states, cluster states, multidimensional

from itertools import permutations
from functools import lru_cache
from math import factorial
import numpy as np
import scipy.sparse as sp
from ..core import (qjf, kron, kronpow, eye, ldmul, eyepad)
from ..solve import eigsys
from .operators import sig, controlled


def basis_vec(dir, dim, sparse=False, **kwargs):
    """
    Constructs a unit vector ket.

    Parameters
    ----------
        dir: which dimension the key should point in
        dim: total number of dimensions
        sparse: return vector as sparse-csr matrix

    Returns:
    --------
        x: quijified basis vector
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


@lru_cache(maxsize=8)
def bell_state(s, **kwargs):
    """
    Generates one of the four bell-states;
    0: phi+, 1: phi-, 2: psi+, 3: psi- (singlet)
    """
    keymap = {
        'psi-': 'psi-', 0: 'psi-', 's': 'psi-', 'singlet': 'psi-',
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


def thermal_state(ham, beta, precomp_func=False):
    """
    Generate a thermal state of a hamtiltonian.

    Parameters
    ----------
        ham: hamtilonian, either full or tuple of (evals, evecs)
        beta: inverse temperatre of state
        precomp_func: if true, return a function that takes `beta`
            only and is closed over the solved hamiltonian.

    Returns
    -------
        rho_th: density matrix of thermal state, or func to generate such
    """
    if isinstance(ham, (list, tuple)):  # solved already
        l, v = ham
    else:
        l, v = eigsys(ham)
    l = l - min(l)  # offset by min to avoid numeric problems

    def gen_state(b):
        el = np.exp(-b * l)
        el /= np.sum(el)
        return v @ ldmul(el, v.H)

    return gen_state if precomp_func else gen_state(beta)


def neel_state(n):
    binary = '01' * (n // 2)
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


def levi_civita(perm):
    """
    Compute the generalised levi-civita coefficient for a
    permutation of the ints in range(n)
    """
    n = len(perm)
    if n != len(set(perm)):  # infer there are repeated elements
        return 0
    mat = np.zeros((n, n), dtype=np.int32)
    for i, j in zip(range(n), perm):
        mat[i, j] = 1
    return int(np.linalg.det(mat))


def perm_state(ps):
    n = len(ps)
    vec_perm = permutations(ps)
    ind_perm = permutations(range(n))

    def terms():
        for vec, ind in zip(vec_perm, ind_perm):
            yield levi_civita(ind) * kron(*vec)

    return sum(terms()) / factorial(n)**0.5


def graph_state_1d(n, cyclic=True, sparse=False):
    """ Graph State """
    p = kronpow(plus(sparse=sparse), n)
    for i in range(n-1):
        p = eyepad(controlled('z', sparse=True), [2] * n, (i, i+1)) @ p
    if cyclic:
        p = ((eye(2, sparse=True) & eye(2**(n-2), sparse=True) &
              qjf([1, 0], qtype='dop', sparse=True)) +
             (sig('z', sparse=True) & eye(2**(n-2), sparse=True) &
              qjf([0, 1], qtype='dop', sparse=True))) @ p
    return p
