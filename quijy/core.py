"""
Core functions for manipulating quantum objects.
"""

import numpy as np
import numpy.linalg as nla
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from numexpr import evaluate as evl
from numba import jit


def isket(p):
    """ Checks if matrix is in ket form, i.e. a column """
    return p.shape[0] > 1 and p.shape[1] == 1  # Column vector check


def isbra(p):
    """ Checks if matrix is in bra form, i.e. a row """
    return p.shape[0] == 1 and p.shape[1] > 1  # Row vector check


def isop(p):
    """ Checks if matrix is an operator, i.e. square """
    m, n = np.shape(p)
    return m == n and m > 1  # Square matrix check


def qonvert(data, qtype=None, sparse=False):
    """ Converts lists to 'quantum' i.e. complex matrices, kets being columns.
    Input:
        data:  list describing entries
        qtype: output type, either 'ket', 'bra' or 'dop' if given
        sparse: convert output to sparse 'csr' format.
    Returns:
        x: numpy or sparse matrix
    * Will unravel an array if 'ket' or 'bra' given.
    * Will conjugate if 'bra' given.
    * Will leave operators as is if 'dop' given.
    """
    x = np.asmatrix(data, dtype=complex)
    sz = np.prod(x.shape)
    if qtype == 'ket':
        x.shape = (sz, 1)
    elif qtype == 'bra':
        x.shape = (1, sz)
        x = np.conj(x)
    elif qtype == 'dop' and not isop(x):
        x = qonvert(x, 'ket') * qonvert(x, 'ket').H
    return sp.csr_matrix(x, dtype=complex) if sparse else x


@jit
def tr(a):
    """ Trace of hermitian matrix (jit version faster than numpy!) """
    x = 0.0
    for i in range(a.shape[0]):
        x += a[i, i].real
    return x
# def tr(a):
#     """ Fallback version for debugging """
#     return np.real(np.trace(a))


def nrmlz(p):
    """ Returns the state p in normalized form """
    return (p / np.sqrt(p.H * p) if isket(p) else
            p / np.sqrt(p * p.H) if isbra(p) else
            p / tr(p))


def eye(n, sparse=False):
    """ Return identity of size n in complex format, optionally sparse"""
    return (sp.eye(n, dtype=complex, format='csr') if sparse else
            np.eye(n, dtype=complex))


@jit
def krnd2(a, b):
    """
    Fast tensor product of two dense arrays (Fast than numpy using jit)
    """
    m, n = a.shape
    p, q = b.shape
    x = np.empty((m * p, n * q), dtype=complex)
    for i in range(m):
        for j in range(n):
            x[i * p:(i + 1)*p, j * q:(j + 1) * q] = a[i, j] * b
    return np.asmatrix(x)
# def krnd2(a, b):  # Fallback for debugging
#     return np.kron(a, b)


def kron(*args):
    """ Tensor product of variable number of arguments.
    Input:
        args: objects to be tensored together
    Returns:
        operator
    The product is performed as (a * (b * (c * ...)))
    """
    a = args[0]
    b = args[1] if len(args) == 2 else  \
        kron(*args[1:])  # Recursively perform kron to 'right'
    return (sp.kron(a, b, 'csr') if (sp.issparse(a) or sp.issparse(b)) else
            krnd2(a, b))


def eyepad(a, dims, inds, sparse=False):
    """ Pad an operator with identities to act on particular subsystem.
    Input:
        a: operator to act
        dims: list of dimensions of subsystems.
        inds: indices of dims to act a on.
        sparse: whether output should be sparse
    Returns:
        b: operator with a acting on each subsystem specified by inds
    Note that the actual numbers in dims[inds] are ignored and the size of
    a is assumed to match. Sparsity of the output can be inferred from
    input.
    """
    sparse = sp.issparse(a)
    inds = np.array(inds, ndmin=1)
    b = eye(np.prod(dims[0:inds[0]]), sparse=sparse)
    for i in range(len(inds) - 1):
        b = kron(b, a)
        pad_size = np.prod(dims[inds[i] + 1:inds[i + 1]])
        b = kron(b, eye(pad_size, sparse=sparse))
    b = kron(b, a)
    pad_size = np.prod(dims[inds[-1] + 1:])
    b = kron(b, eye(pad_size, sparse=sparse))
    return b


def kronpow(a, pow):
    """ Returns 'a' tensored with itself pow times """
    return (1 if pow == 0 else
            a if pow == 1 else
            kron(*[a for i in range(pow)]))


def basis_vec(dir, dim, sparse=False):
    """
    Constructs a unit ket that points in dir of total dimensions dim
    """
    if sparse:
        return sp.coo_matrix(([1.0], ([dir], [0])),
                             dtype=complex, shape=(dim, 1))
    else:
        x = np.zeros([dim, 1], dtype=complex)
        x[dir] = 1.0
    return x


def sig(xyz, sparse=False):
    """
    Generates one of the three Pauli matrices, 0-X, 1-Y, 2-Z
    """
    if xyz in ('x', 'X', 1):
        return qonvert([[0, 1], [1, 0]], sparse=sparse)
    elif xyz in ('y', 'Y', 2):
        return qonvert([[0, -1j], [1j, 0]], sparse=sparse)
    elif xyz in ('z', 'Z', 3):
        return qonvert([[1, 0], [0, -1]], sparse=sparse)
    elif xyz in ('i', 'I', 'id', 0):
        return qonvert([[1, 0], [0, 1]], sparse=sparse)


def evals(a, sort=True):
    """ Find sorted eigenvalues of matrix
    Input:
        a: hermitian matrix
    Returns:
        l: array of eigenvalues, if sorted, by ascending algebraic order
    """
    l = nla.eigvalsh(a)
    return np.sort(l) if sort else l


def ldmul(v, m):
    '''
    Fast left diagonal multiplication of v: vector of diagonal matrix, and m
    '''
    v = v.reshape(np.size(v), 1)
    return evl('v*m')


def groundstate(a):
    """
    # Returns the eigenvecor corresponding to the smallest eigenvalue of a
    """
    l0, v0 = spla.eigsh(a, k=1, which='SA')
    return qonvert(v0, 'ket')


def rdmul(m, v):
    '''
    Fast right diagonal multiplication of v: vector of diagonal matrix, and m
    '''
    v = v.reshape(1, np.size(v))
    return evl('m*v')


def evecs(a, sort=True):
    """ Find sorted eigenvectors of matrix
    Input:
        a: hermitian matrix
    Returns:
        v: eigenvectors as columns of matrix, if sorted, by ascending
        eigenvalue order
    """
    l, v = nla.eigh(a)
    return qonvert(v[:, np.argsort(l)]) if sort else qonvert(v)


def esys(a, sort=True):
    """ Find sorted eigenpairs of matrix
    Input:
        a: hermitian matrix
    Returns:
        l: array of eigenvalues, if sorted, by ascending algebraic order
        v: corresponding eigenvectors as columns of matrix
    """
    l, v = nla.eigh(a)
    if sort:
        sortinds = np.argsort(l)
        return l[sortinds], qonvert(v[:, sortinds])
    else:
        return l, v


def trx(p, dims, keep):
    """ Perform partial trace.
    Input:
        p: state to perform partial trace on, vector or operator
        dims: list of subsystem dimensions
        keep: index of subsytems to keep
    Returns:
        Density matrix of subsytem dimensions dims[keep]
    """
    # Cast as ndarrays for 2D+ reshaping
    if np.size(keep) == np.size(dims):  # keep all subsystems
        if not isop(p):
            return p * p.H  # but return as density operator
        return p
    n = len(dims)
    dims = np.array(dims)
    keep = np.array(keep)
    keep = np.reshape(keep, [keep.size])  # make sure array
    lose = np.delete(np.arange(n), keep)
    dimkeep = np.prod(dims[keep])
    dimlose = np.prod(dims[lose])
    # Permute dimensions into block of keep and block of lose
    perm = np.r_[keep, lose]
    # Apply permutation to state and trace out block of lose
    if not isop(p):  # p = psi
        p = np.array(p)
        p = p.reshape(dims) \
            .transpose(perm) \
            .reshape([dimkeep, dimlose])
        p = np.matrix(p)
        return qonvert(p * p.H)
    else:  # p = rho
        p = np.array(p)
        p = p.reshape(np.r_[dims, dims]) \
            .transpose(np.r_[perm, perm + n]) \
            .reshape([dimkeep, dimlose, dimkeep, dimlose]) \
            .trace(axis1=1, axis2=3)
        return qonvert(p)


def entropy(rho):
    """
    Computes the (von Neumann) entropy of positive matrix rho
    """
    l = evals(rho)
    l = l[np.nonzero(l)]
    return np.sum(-l * np.log2(l))


def mutual_information(p, dims, sysa=0, sysb=1):
    """
    Partitions rho into dims, and finds the mutual information between the
    subsystems at indices sysa and sysb
    """
    if p.isop() or len(dims) > 2:  # mixed combined system
        rhoab = trx(p, dims, [sysa, sysb])
        rhoa = trx(rhoab, np.r_[dims[sysa], dims[sysb]], 0)
        rhob = trx(rhoab, np.r_[dims[sysa], dims[sysb]], 1)
        hab = entropy(rhoab)
        ha = entropy(rhoa)
        hb = entropy(rhob)
    else:  # pure combined system
        hab = 0.0
        rhoa = trx(p, dims, sysa)
        ha = entropy(rhoa)
        hb = ha
    return ha + hb - hab


def chop(x, eps=1.0e-12):
    """
    Sets any values of x smaller than eps (relative to range(x)) to zero
    """
    xr = x.max() - x.min()
    xm = xr * eps
    x[abs(x) < xm] = 0
    return x


def trace_norm(a):
    return np.absolute(evals(a)).sum()


def trace_distance(p, w):
    return 0.5 * trace_norm(p - w)


def partial_transpose(p, dims=[2, 2]):
    """
    Partial transpose of a two qubit dm
    """
    if p.isket():  # state vector supplied
        p = p * p.H
    elif p.isbra():
        p = p.H * p
    p = np.array(p)\
        .reshape(np.concatenate((dims, dims)))  \
        .transpose([2, 1, 0, 3])  \
        .reshape([np.prod(dims), np.prod(dims)])
    return qonvert(p)


def logneg(rho, dims=[2, 2], sysa=0, sysb=1):
    if len(dims) > 2:
        rho = trx(rho, dims, [sysa, sysb])
        dims = [dims[sysa], dims[sysb]]
    e = np.log2(trace_norm(partial_transpose(rho, dims)))
    return max(0.0, e)


def negativity(rho, dims=[2, 2], sysa=0, sysb=1):
    if len(dims) > 2:
        rho = trx(rho, dims, [sysa, sysb])
    n = (trace_norm(partial_transpose(rho)) - 1.0) / 2.0
    return max(0.0, n)


def bell_state(n):
    """
    Generates one of the four bell-states;
    0: phi+, 1: phi-, 2: psi+, 3: psi- (singlet)
    """
    return (qonvert([1, 0, 0, 1], 'ket') / np.sqrt(2.0) if n == 0 else
            qonvert([0, 1, 1, 0], 'ket') / np.sqrt(2.0) if n == 2 else
            qonvert([1, 0, 0, -1], 'ket') / np.sqrt(2.0) if n == 1 else
            qonvert([0, 1, -1, 0], 'ket') / np.sqrt(2.0))


def bloch_state(ax, ay, az, sparse=False, purify=False):
    if purify:
        ax, ay, az = np.array([ax, ay, az])/np.sqrt(ax**2 + ay**2 + az**2)
    rho = 0.5 * (sig('i') + ax * sig('x') + ay * sig('y') + az * sig('z'))
    return rho if not sparse else qonvert(rho, sparse=sparse)


# functions
def random_psi(n):
    """
    Generates a wavefunction with random coefficients, normalised
    """
    psi = 2.0 * np.matrix(np.random.rand(n, 1)) - 1
    psi = psi + 2.0j * np.random.rand(n, 1) - 1.0j
    return nrmlz(psi)


def random_rho(n):
    """
    Generates a random density matrix of dimension n, no special properties
    other than being guarateed hermitian, positive, and trace 1.
    """
    rho = 2.0 * np.matrix(np.random.rand(n, n)) - 1
    rho = rho + 2.0j * np.matrix(np.random.rand(n, n)) - 1.0j
    rho = rho + rho.H
    return nrmlz(rho * rho)


def random_product_state(n):
    x = 1
    for i in range(n):
        u = np.random.rand()
        v = np.random.rand()
        phi = 2 * np.pi * u
        theta = np.arccos(2 * v - 1)
        x = kron(x, np.matrix([[np.cos(theta / 2.0)],
                               [np.exp(1.0j * phi) * np.sin(theta / 2)]]))
    return x


def neel_state(n):
    binary = '01' * (n / 2)
    binary += (n % 2 == 1) * '0'  # add trailing spin for odd n
    return basis_vec(int(binary, 2), 2 ** n)


def singlet_pairs(n):
    return kronpow(bell_state(3), (n / 2))


def werner_state(p):
    return p * bell_state(3) * bell_state(3).H + (1 - p) * np.eye(4) / 4
