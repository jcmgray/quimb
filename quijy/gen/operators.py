from functools import lru_cache
import numpy as np
from ..core import qjf, eye, kron, eyepad


@lru_cache(maxsize=64)
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
def controlled(s, sparse=False):
    keymap = {
        'x': 'x', 'not': 'x',
        'z': 'z'
        }
    op = sig(keymap[s], sparse=sparse)
    return ((qjf([1, 0], qtype='dop', sparse=sparse) &
             eye(2, sparse=sparse)) +
            (qjf([0, 1], qtype='dop', sparse=sparse) & op))


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
