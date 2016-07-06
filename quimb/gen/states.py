"""
Functions for generating quantum objects.
"""
# TODO: Graph states, cluster states, multidimensional

from itertools import permutations
from functools import lru_cache
from math import factorial
import numpy as np
import scipy.sparse as sp
from ..accel import ldmul, dot
from ..core import (qu, kron, kronpow, eye, eyepad)
from ..solve import eigsys
from .operators import sig, controlled


def basis_vec(dir, dim, sparse=False, **kwargs):
    """ Constructs a unit vector ket.

    Parameters
    ----------
        dir: which dimension the key should point in
        dim: total number of dimensions
        sparse: return vector as sparse-csr matrix

    Returns:
    --------
        x: quijified basis vector """
    if sparse:
        return sp.csr_matrix(([1.0], ([dir], [0])),
                             dtype=complex, shape=(dim, 1))
    else:
        x = np.zeros([dim, 1], dtype=complex)
        x[dir] = 1.0
    return qu(x, **kwargs)


def up(**kwargs):
    """ Returns up-state, aka. |0>, +Z eigenstate."""
    return qu([[1], [0]], **kwargs)

zplus = up


def down(**kwargs):
    """ Returns down-state, aka. |1>, -Z eigenstate."""
    return qu([[0], [1]], **kwargs)

zminus = down


def plus(**kwargs):
    """ Returns plus-state, aka. |+>, +X eigenstate."""
    return qu([[2**-0.5], [2**-0.5]], **kwargs)

xplus = plus


def minus(**kwargs):
    """ Returns minus-state, aka. |->, -X eigenstate."""
    return qu([[2**-0.5], [-2**-0.5]], **kwargs)

xminus = minus


def yplus(**kwargs):
    """ Returns yplus-state, aka. |y+>, +Y eigenstate."""
    return qu([[2**-0.5], [1.0j / (2**0.5)]], **kwargs)


def yminus(**kwargs):
    """ Returns yplus-state, aka. |y->, -Y eigenstate."""
    return qu([[2**-0.5], [-1.0j / (2**0.5)]], **kwargs)


def bloch_state(ax, ay, az, purified=False, **kwargs):
    """ Construct qubit density matrix from bloch vector.

    Parameters
    ----------
        ax: x component
        ay: y component
        az: z component
        purified: whether to map vector to surface of bloch sphere

    Returns
    -------
        p: density matrix of qubit 'pointing' in (ax, ay, az) direction. """
    n = (ax**2 + ay**2 + az**2)**.5
    if purified:
        ax, ay, az = (a / n for a in (ax, ay, az))
    return sum(0.5 * a * sig(s, **kwargs)
               for a, s in zip((1, ax, ay, az), "ixyz"))


@lru_cache(maxsize=8)
def bell_state(s, **kwargs):
    """ Generates one of the four bell-states;
    0: phi+, 1: phi-, 2: psi+, 3: psi- (singlet) """
    keymap = {"psi-": "psi-", 0: "psi-", "psim": "psi-",
              "psi+": "psi+", 1: "psi+", "psip": "psi+",
              "phi-": "phi-", 2: "phi-", "phim": "phi-",
              "phi+": "phi+", 3: "phi+", "phip": "phi+"}
    c = 2.**-.5
    statemap = {"psi-": lambda: qu([[0], [c], [-c], [0]], **kwargs),
                "phi+": lambda: qu([[c], [0], [0], [c]], **kwargs),
                "phi-": lambda: qu([[c], [0], [0], [-c]], **kwargs),
                "psi+": lambda: qu([[0], [c], [c], [0]], **kwargs)}
    return statemap[keymap[s]]()


def singlet(**kwargs):
    """ Alias for one of bell-states """
    return bell_state("psi-", **kwargs)


def thermal_state(ham, beta, precomp_func=False):
    """ Generate a thermal state of a hamtiltonian.

    Parameters
    ----------
        ham: hamtilonian, either full or tuple of (evals, evecs)
        beta: inverse temperatre of state
        precomp_func: if true, return a function that takes `beta`
            only and is closed over the solved hamiltonian.

    Returns
    -------
        rho_th: density matrix of thermal state, or func to generate such """
    if isinstance(ham, (list, tuple)):  # solved already
        l, v = ham
    else:
        l, v = eigsys(ham)
    l = l - min(l)  # offset by min to avoid numeric problems

    def gen_state(b):
        el = np.exp(-b * l)
        el /= np.sum(el)
        return dot(v, ldmul(el, v.H))

    return gen_state if precomp_func else gen_state(beta)


def neel_state(n, **kwargs):
    """ Construct Neel state for n spins, i.e. alternating up/down. """
    binary = "01" * (n // 2) + (n % 2 == 1) * "0"
    return basis_vec(int(binary, 2), 2 ** n, **kwargs)


def singlet_pairs(n, **kwargs):
    """ Construct fully dimerised spin chain. """
    return kronpow(bell_state('psi-', **kwargs), (n // 2))


def werner_state(p, **kwargs):
    """ Construct Werner State, i.e. fractional mix of eye with `p` amount of
    singlet """
    return p * bell_state('psi-', qtype="dop", **kwargs) +  \
        (1 - p) * eye(4, **kwargs) / 4


def ghz_state(n, **kwargs):
    """ Construct GHZ state of `n` spins, i.e. equal superposition of all up
    and down. """
    return (basis_vec(0, 2**n, **kwargs) +
            basis_vec(2**n - 1, 2**n, **kwargs)) / 2.**.5


def w_state(n, **kwargs):
    """ Construct W-state for `n` spins, i.e. equal superposition of all
    single spin up states. """
    return sum(basis_vec(2**i, 2**n, **kwargs) for i in range(n))/n**0.5


def levi_civita(perm):
    """ Compute the generalised levi-civita coefficient for a
    permutation of the ints in range(n). """
    n = len(perm)
    if n != len(set(perm)):  # infer there are repeated elements
        return 0
    mat = np.zeros((n, n), dtype=np.int32)
    for i, j in zip(range(n), perm):
        mat[i, j] = 1
    return int(np.linalg.det(mat))


def perm_state(ps):
    """ Construct the anti-symmetric state which is the +- sum of all
    permutations of states `ps`. """
    n = len(ps)
    vec_perm = permutations(ps)
    ind_perm = permutations(range(n))

    def terms():
        for vec, ind in zip(vec_perm, ind_perm):
            yield levi_civita(ind) * kron(*vec)

    return sum(terms()) / factorial(n)**0.5


def graph_state_1d(n, cyclic=True, sparse=False):
    """ Graph State on a line. """
    p = kronpow(plus(sparse=sparse), n)
    for i in range(n-1):
        p = eyepad(controlled("z", sparse=True), [2] * n, (i, i+1)) @ p
    if cyclic:
        p = ((eye(2, sparse=True) & eye(2**(n-2), sparse=True) &
              qu([1, 0], qtype="dop", sparse=True)) +
             (sig("z", sparse=True) & eye(2**(n-2), sparse=True) &
              qu([0, 1], qtype="dop", sparse=True))) @ p
    return p
