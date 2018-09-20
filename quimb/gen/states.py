"""
Functions for generating quantum states.
"""

import itertools
import functools
import math
import numpy as np

from ..core import (dag, ldmul, dot, make_immutable,
                    qu, kron, eye, ikron, kronpow)
from ..linalg.base_linalg import eigh
from .operators import pauli, controlled


def basis_vec(i, dim, ownership=None, **kwargs):
    """Constructs a unit vector ket.

    Parameters
    ----------
    i : int
        Which index should the single non-zero, unit entry.
    dim : int
        Total size of hilbert space.
    sparse : bool, optional
        Return vector as sparse matrix.
    kwargs
        Supplied to ``qu``.

    Returns:
    --------
    vector
        The basis vector.
    """
    if ownership is None:
        shape = (dim, 1)
        x = np.zeros(shape, dtype=complex)
        x[i] = 1.0
    else:
        ri, rf = ownership
        shape = (rf - ri, 1)
        x = np.zeros(shape, dtype=complex)
        if ri <= i < rf:
            x[i - ri] = 1.0

    return qu(x, **kwargs)


@functools.lru_cache(8)
def up(**kwargs):
    """Returns up-state, aka. ``|0>``, +Z eigenstate.
    """
    u = qu([[1], [0]], **kwargs)
    make_immutable(u)
    return u


zplus = up


@functools.lru_cache(8)
def down(**kwargs):
    """Returns down-state, aka. ``|1>``, -Z eigenstate.
    """
    d = qu([[0], [1]], **kwargs)
    make_immutable(d)
    return d


zminus = down


@functools.lru_cache(8)
def plus(**kwargs):
    """Returns plus-state, aka. ``|+>``, +X eigenstate.
    """
    return qu([[2**-0.5], [2**-0.5]], **kwargs)


xplus = plus


@functools.lru_cache(8)
def minus(**kwargs):
    """Returns minus-state, aka. ``|->``, -X eigenstate.
    """
    return qu([[2**-0.5], [-2**-0.5]], **kwargs)


xminus = minus


@functools.lru_cache(8)
def yplus(**kwargs):
    """Returns yplus-state, aka. ``|y+>``, +Y eigenstate.
    """
    return qu([[2**-0.5], [1.0j / (2**0.5)]], **kwargs)


@functools.lru_cache(8)
def yminus(**kwargs):
    """Returns yplus-state, aka. ``|y->``, -Y eigenstate.
    """
    return qu([[2**-0.5], [-1.0j / (2**0.5)]], **kwargs)


def bloch_state(ax, ay, az, purified=False, **kwargs):
    """Construct qubit density operator from bloch vector.

    Parameters
    ----------
    ax : float
        X component of bloch vector.
    ay : float
        Y component of bloch vector.
    az : float
        Z component of bloch vector.
    purified :
        Whether to map vector to surface of bloch sphere.

    Returns
    -------
    Matrix
        Density operator of qubit 'pointing' in (ax, ay, az) direction.
    """
    n = (ax**2 + ay**2 + az**2)**.5
    if purified:
        ax, ay, az = (a / n for a in (ax, ay, az))
    return sum(0.5 * a * pauli(s, **kwargs)
               for a, s in zip((1, ax, ay, az), "ixyz"))


@functools.lru_cache(maxsize=8)
def bell_state(s, **kwargs):
    r"""One of the four bell-states.

    If n = 2**-0.5, they are:

        0. ``'psi-'`` : ``n * ( |01> - |10> )``
        1. ``'psi+'`` : ``n * ( |01> + |10> )``
        2. ``'phi-'`` : ``n * ( |00> - |11> )``
        3. ``'phi+'`` : ``n * ( |00> + |11> )``

    They can be enumerated in this order.

    Parameters
    ----------
    s : str or int
        String of number of state corresponding to above.
    kwargs :
        Supplied to ``qu`` called on state.

    Returns
    -------
    p : immutable vector
        The bell-state ``s``.
    """
    keymap = {"psi-": "psi-", 0: "psi-", "psim": "psi-",
              "psi+": "psi+", 1: "psi+", "psip": "psi+",
              "phi-": "phi-", 2: "phi-", "phim": "phi-",
              "phi+": "phi+", 3: "phi+", "phip": "phi+"}
    c = 2.**-.5
    statemap = {"psi-": lambda: qu([0, c, -c, 0], **kwargs),
                "phi+": lambda: qu([c, 0, 0, c], **kwargs),
                "phi-": lambda: qu([c, 0, 0, -c], **kwargs),
                "psi+": lambda: qu([0, c, c, 0], **kwargs)}
    state = statemap[keymap[s]]()
    make_immutable(state)
    return state


def singlet(**kwargs):
    """Alias for the 'psi-' bell-state.
    """
    return bell_state("psi-", **kwargs)


def thermal_state(ham, beta, precomp_func=False):
    """Generate a thermal state of a Hamiltonian.

    Parameters
    ----------
    ham : operator or (1d-array, 2d-array)
        Hamiltonian, either full or tuple of (evals, evecs).
    beta : float
        Inverse temperature of state.
    precomp_func : bool, optional
        If True, return a function that takes ``beta``
        only and is closed over the solved hamiltonian.

    Returns
    -------
    operator or callable
        Density operator of thermal state, or function to generate such given
        a temperature.
    """
    if isinstance(ham, (list, tuple)):  # solved already
        evals, evecs = ham
    else:
        evals, evecs = eigh(ham)
    evals -= evals.min()  # offset by min to avoid numeric problems

    def gen_state(b):
        el = np.exp(-b * evals)
        el /= np.sum(el)
        return dot(evecs, ldmul(el, dag(evecs)))

    return gen_state if precomp_func else gen_state(beta)


def computational_state(binary, **kwargs):
    """Generate the qubit computational state with ``binary``.

    Parameters
    ----------
    binary : sequence of 0s and 1s
        The binary of the computation state.

    Examples
    --------
    >>> computational_state('101'):
    qarray([[0.+0.j],
            [0.+0.j],
            [0.+0.j],
            [0.+0.j],
            [0.+0.j],
            [1.+0.j],
            [0.+0.j],
            [0.+0.j]])

    >>> qu.computational_state([0, 1], qtype='dop')
    qarray([[0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
            [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j],
            [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
            [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j]])

    See Also
    --------
    MPS_computational_state, basic_vec
    """
    if not isinstance(binary, str):
        binary = "".join(map(str, binary))

    return basis_vec(int(binary, 2), 2 ** len(binary), **kwargs)


def neel_state(n, down_first=False, **kwargs):
    """Construct Neel state for n spins, i.e. alternating up/down.

    Parameters
    ----------
    n : int
        Number of spins.
    down_first : bool, optional
        Whether to start with '1' or '0' first.
    kwargs
        Supplied to ``qu`` called on state.

    See Also
    --------
    computational_state, MPS_neel_state
    """
    binary = "01" * (n // 2) + (n % 2 == 1) * "0"
    if down_first:
        binary = "1" + binary[:-1]

    return computational_state(binary, **kwargs)


def singlet_pairs(n, **kwargs):
    """Construct fully dimerised spin chain.

    I.e. ``bell_state('psi-') & bell_state('psi-') & ...``

    Parameters
    ----------
    n : int
        Number of spins.
    kwargs
        Supplied to ``qu`` called on state.

    Returns
    -------
    vector
    """
    return kronpow(bell_state('psi-', **kwargs), (n // 2))


def werner_state(p, **kwargs):
    """Construct Werner State, i.e. fractional mix of identity with singlet.

    Parameters
    ----------
    p : float
        Singlet Fraction.
    kwargs
        Supplied to :func:`~quimb.qu` called on state.

    Returns
    -------
    qarray
    """
    return (p * bell_state('psi-', qtype="dop", **kwargs) +
            (1 - p) * eye(4, **kwargs) / 4)


def ghz_state(n, **kwargs):
    """Construct GHZ state of `n` spins, i.e. equal superposition of all up
    and down.

    Parameters
    ----------
    n : int
        Number of spins.
    kwargs
        Supplied to ``qu`` called on state.

    Returns
    -------
    vector
    """
    return (basis_vec(0, 2**n, **kwargs) +
            basis_vec(2**n - 1, 2**n, **kwargs)) / 2.**.5


def w_state(n, **kwargs):
    """Construct W-state: equal superposition of all single spin up states.

    Parameters
    ----------
    n : int
        Number of spins.
    kwargs
        Supplied to ``qu`` called on state.

    Returns
    -------
    vector
    """
    return sum(basis_vec(2**i, 2**n, **kwargs) for i in range(n)) / n**0.5


def levi_civita(perm):
    """Compute the generalised levi-civita coefficient for a permutation.

    Parameters
    ----------
    perm : sequence of int
        The permutation, a re-arrangement of ``range(n)``.

    Returns
    -------
    int
        Either -1, 0 or 1.
    """
    n = len(perm)
    if n != len(set(perm)):  # infer there are repeated elements
        return 0
    mat = np.zeros((n, n), dtype=np.int32)
    for i, j in zip(range(n), perm):
        mat[i, j] = 1
    return int(np.linalg.det(mat))


def perm_state(ps):
    """Construct the anti-symmetric state which is the +- sum of all
    tensored permutations of states ``ps``.

    Parameters
    ----------
    ps :  sequence of states
        The states to combine.

    Returns
    -------
    vector or operator
        The permutation state, dimension same as ``kron(*ps)``.

    Examples
    --------
    A singlet is the ``perm_state`` of up and down.

    >>> states = [up(), down()]
    >>> pstate = perm_state(states)
    >>> expec(pstate, singlet())
    1.0
    """
    n = len(ps)
    vec_perm = itertools.permutations(ps)
    ind_perm = itertools.permutations(range(n))

    def terms():
        for vec, ind in zip(vec_perm, ind_perm):
            yield levi_civita(ind) * kron(*vec)

    return sum(terms()) / math.factorial(n)**0.5


def graph_state_1d(n, cyclic=True, sparse=False):
    """Graph State on a line.

    Parameters
    ----------
    n : int
        The number of spins.
    cyclic : bool, optional
        Whether to use cyclic boundary conditions for the graph.
    sparse : bool, optional
        Whether to return a sparse state.

    Returns
    -------
    vector
        The 1d-graph state.
    """
    p = kronpow(plus(sparse=sparse), n)

    for i in range(n - 1):
        p = ikron(controlled("z", sparse=True), [2] * n, (i, i + 1)) @ p

    if cyclic:
        p = ((eye(2, sparse=True) & eye(2**(n - 2), sparse=True) &
              qu([1, 0], qtype="dop", sparse=True)) +
             (pauli("z", sparse=True) & eye(2**(n - 2), sparse=True) &
              qu([0, 1], qtype="dop", sparse=True))) @ p

    return p
