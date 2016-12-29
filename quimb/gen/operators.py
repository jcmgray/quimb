from functools import lru_cache

from cytoolz import isiterable, concat
import numpy as np
import scipy.sparse as sp

from ..accel import accel, make_immutable
from ..core import qu, eye, kron, eyepad


@lru_cache(maxsize=8)
def sig(xyz, dim=2, **kwargs):
    """Generates the spin operators for spin 1/2 or 1.

    Parameters
    ----------
        xyz: which spatial direction
        dim: dimension of spin operator (e.g. 3 for spin-1)

    Returns
    -------
        spin operator, quijified.
    """
    xyzmap = {0: 'i', 'i': 'i', 'I': 'i',
              1: 'x', 'x': 'x', 'X': 'x',
              2: 'y', 'y': 'y', 'Y': 'y',
              3: 'z', 'z': 'z', 'Z': 'z'}
    opmap = {('i', 2): lambda: eye(2, **kwargs),
             ('x', 2): lambda: qu([[0, 1],
                                   [1, 0]], **kwargs),
             ('y', 2): lambda: qu([[0, -1j],
                                   [1j, 0]], **kwargs),
             ('z', 2): lambda: qu([[1, 0],
                                   [0, -1]], **kwargs),
             ('i', 3): lambda: eye(3, **kwargs),
             ('x', 3): lambda: qu([[0, 1, 0],
                                   [1, 0, 1],
                                   [0, 1, 0]], **kwargs) / 2**.5,
             ('y', 3): lambda: qu([[0, -1j, 0],
                                   [1j, 0, -1j],
                                   [0, 1j, 0]], **kwargs) / 2**.5,
             ('z', 3): lambda: qu([[1, 0, 0],
                                   [0, 0, 0],
                                   [0, 0, -1]], **kwargs)}
    op = opmap[(xyzmap[xyz], dim)]()
    # Operator is cached, so make sure it cannot be modified
    make_immutable(op)
    return op


@lru_cache(maxsize=8)
def controlled(s, sparse=False):
    """Construct a controlled pauli gate for two qubits.
    """
    keymap = {'x': 'x', 'not': 'x',
              'y': 'y',
              'z': 'z'}
    op = ((qu([1, 0], qtype='dop', sparse=sparse) &
           eye(2, sparse=sparse)) +
          (qu([0, 1], qtype='dop', sparse=sparse) &
           sig(keymap[s], sparse=sparse)))
    make_immutable(op)
    return op


@lru_cache(maxsize=8)
def ham_heis(n, j=1.0, bz=0.0, cyclic=True, sparse=False, stype="csr"):
    """Constructs the heisenberg spin 1/2 hamiltonian

    Parameters
    ----------
        n: number of spins
        j: coupling constant(s), with convention that positive =
            antiferromagnetic. Can supply scalar for isotropic coupling or
            vector (jx, jy, jz).
        bz: z-direction magnetic field
        cyclic: whether to couple the first and last spins
        sparse: whether to return the hamiltonian in sparse form
        stype: what format of sparse matrix to return

    Returns
    -------
        ham: hamiltonian as matrix
    """
    # TODO: vector magnetic field
    dims = (2,) * n
    try:
        jx, jy, jz = j
    except TypeError:
        jx = jy = jz = j

    opts = {"sparse": True, "stype": "coo", "coo_build": True}

    sds = qu(jx * kron(sig('x'), sig('x')) +
             jy * kron(sig('y'), sig('y')) +
             jz * kron(sig('z'), sig('z')) -
             bz * kron(sig('z'), eye(2)), sparse=True, stype="coo")
    ham = sum(eyepad(sds, dims, [i, i + 1], **opts) for i in range(n - 1))

    if cyclic:
        ham = ham + sum(eyepad(j * sig(s, sparse=True),
                               dims, [0, n - 1], **opts)
                        for j, s in zip((jx, jy, jz), 'xyz'))
    if bz != 0.0:
        ham = ham + eyepad(-bz * sig('z', sparse=True), dims, n - 1, **opts)

    if not sparse:
        ham = np.asmatrix(ham.todense())
    elif ham.format != stype:
        ham = ham.asformat(stype)
    make_immutable(ham)
    return ham


@lru_cache(maxsize=8)
def ham_j1j2(n, j1=1.0, j2=0.5, bz=0.0, cyclic=True, sparse=False):
    """Generate the j1-j2 hamiltonian, i.e. next nearest neighbour
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
    dims = (2,) * n
    ps = [sig(i, sparse=True) for i in 'xyz']

    coosj1 = np.array([(i, i + 1) for i in range(n)])
    coosj2 = np.array([(i, i + 2) for i in range(n)])
    if cyclic:
        coosj1, coosj2 = coosj1 % n, coosj2 % n
    else:
        coosj1 = coosj1[np.all(coosj1 < n, axis=1)]
        coosj2 = coosj2[np.all(coosj2 < n, axis=1)]

    def j1_terms():
        for coo in coosj1:
            if abs(coo[1] - coo[0]) == 1:  # can sum then tensor (faster)
                yield eyepad(sum(op & op for op in ps), dims, coo)
            else:  # tensor then sum (slower)
                yield sum(eyepad(op, dims, coo) for op in ps)

    def j2_terms():
        for coo in coosj2:
            if abs(coo[1] - coo[0]) == 2:  # can add then tensor (faster)
                yield eyepad(sum(op & eye(2) & op for op in ps), dims, coo)
            else:
                yield sum(eyepad(op, dims, coo) for op in ps)

    gen_bz = (eyepad([ps[2]], dims, i) for i in range(n))

    ham = j1 * sum(j1_terms()) + j2 * sum(j2_terms())
    if bz != 0:
        ham += bz * sum(gen_bz)
    if not sparse:
        ham = ham.todense()
    make_immutable(ham)
    return ham


@accel
def cmbn(n, k):  # pragma: no cover
    """Integer combinatorial factor.
    """
    x = 1.0
    for _ in range(k):
        x *= n / k
        n -= 1
        k -= 1
    return x


def uniq_perms(xs):
    """Generate all the unique permutations of sequence `xs`.
    """
    if len(xs) == 1:
        yield (xs[0],)
    else:
        uniq_xs = set(xs)
        for first_x in uniq_xs:
            rem_xs = list(xs)
            rem_xs.remove(first_x)
            for sub_perm in uniq_perms(rem_xs):
                yield (first_x,) + sub_perm


@lru_cache(maxsize=8)
def zspin_projector(n, sz=0, stype="csr"):
    """Construct the projector onto spin-z subpspaces.

    Parameters
    ----------
        n : int
            Total size of spin system.
        sz : value or sequence of values
            Spin-z value(s) subspace(s) to find projector for.
        stype : str
            Sparse format of the output matrix.
    """
    if not isiterable(sz):
        sz = (sz,)

    p = 0
    all_perms = []

    for s in sz:
        # Number of 'up' spins
        k = n / 2 + s
        if not k.is_integer():
            raise ValueError("{} is not a valid spin half subspace for "
                             "{} spins.".format(s, n))
        k = int(round(k))
        # Size of subspace
        p += int(round(cmbn(n, k)))
        # Find all computational basis states with correct number of 0s and 1s
        base_perm = '0' * (n - k) + '1' * k
        all_perms += [uniq_perms(base_perm)]

    # Coordinates
    cis = tuple(range(p))  # arbitrary basis
    cjs = tuple(int("".join(perm), 2) for perm in concat(all_perms))

    # Construct matrix which prjects only on to these basis states
    prj = sp.coo_matrix((np.ones(p, dtype=complex), (cis, cjs)),
                        shape=(p, 2**n), dtype=complex)
    prj = qu(prj, stype=stype)
    make_immutable(prj)
    return prj


@lru_cache(maxsize=8)
def swap(dim=2, **kwargs):
    """The SWAP operator acting on subsystems of dimension `dim`.
    """
    a = np.identity(dim**2, dtype=complex)
    a = (a.reshape([dim, dim, dim, dim])
          .transpose([0, 3, 1, 2])
          .reshape([dim**2, dim**2]))
    a = qu(a, **kwargs)
    make_immutable(a)
    return a
