# Copyright Johnnie Gray
# A python library for basic quantum stuff
# Requires numpy, scipy
#  NB: kets are column matrices - [[c0], [c1], ..., [cn]]
# TODO: printprogress, pandas, sparse, simulation manager, adaptive SCHRO

import numpy as np
import numpy.linalg as nla
# import scipy.linalg as sla
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from numba import jit
import numexpr as ne

import matplotlib.pyplot as plt
from matplotlib import cm


# class QuEvo(object):
#     """docstring for QuEvo"""
#     def __init__(self,
#                  p_0=None,
#                  ham=None,
#                  method=None,
#                  vals=None,
#                  vecs=None,
#                  solve=False):
#         super(QuEvo, self).__init__()
#         self.p_0 = p_0  # Initial state
#         self.p_t = p_0  # Current state (being with same as initial)
#         self.ham = ham
#         self.method = method
#         self.qtype = 'bra' if isbra(p_0) else 'dop'
#         if solve:
#             self.solve_ham()
#             self.solved = True
#         elif vals is not None and vecs is not None:
#             self.vals = vals
#             self.vecs = vecs
#             self.solved = True
#         else:
#             self.solved = False

#     def solve_ham(self):
#         # Diagonalise hamiltonian
#         self.vals, self.vecs = esys(self.ham)
#         # Find initial state in energy eigenbasis
#         if self.qtype = 'bra':
#             self.pe_0 = self.vecs.H * p_0
#         elif self.qtype = 'dop':
#             self.pe_0 = self.vecs.H * p_0 * self.vecs
#         # Mark solved
#         self.solved = True

#     def update_to(self, t):
#         if method == 'uni':
#             if not self.solved:
#                 self.solve_ham()
#             if self.qtype == 'bra':
#                 pass


def qonvert(data, qtype=None, sparse=False):
    """
    Converts lists to quantum objects as matrices, with kets as columns.
    Assumes correct entries but not shape (i.e. DOES NOT conjugate for 'bra')
    """
    x = np.asmatrix(data, dtype=complex)
    sz = np.prod(x.shape)
    x = (x if qtype is None else
         x.reshape(sz, 1) if qtype == 'ket' else
         x.reshape(1, sz) if qtype == 'bra' else
         qonvert(x, 'ket') * qonvert(x, 'ket').H if qtype == 'dop' else
         x)
    return (sp.csr_matrix(x, dtype=complex) if sparse else
            x)


def isket(p):
    """ Check is q object is in ket form, i.e. columns """
    return p.shape[0] > 1 and p.shape[1] == 1  # Column vector check


def isbra(p):
    """ Check is q object is in bra form, i.e. row """
    return p.shape[0] == 1 and p.shape[1] > 1  # Row vector check


def isop(p):
    """ Checks if q object is an operator, i.e. square """
    return p.shape[0] == p.shape[1]  # Square matrix check


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
    """ Return identity in complex format, and possibly sparse"""
    return (sp.eye(n, dtype=complex, format='csr') if sparse else
            np.eye(n, dtype=complex))


@jit
def krnd2(a, b):
    """
    Fast tensor product of two dense arrays (Fast than numpy because of jit)
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
    """ Tensor product of variable number of arguments"""
    a = args[0]
    b = args[1] if len(args) == 2 else  \
        kron(*args[1:])  # Recursively perform kron
    return (sp.kron(a, b, 'csr') if (sp.issparse(a) or sp.issparse(b)) else
            krnd2(a, b))


def eyepad(a, dims, inds):
    """
    Pads the operators a with identities such that it acts on the subsystems of
    dims specified by inds. Infers sparse/dense from a.
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


def bloch_state(ax, ay, az, sparse=False, purify=False):
    if purify:
        ax, ay, az = np.array([ax, ay, az])/np.sqrt(ax**2 + ay**2 + az**2)
    rho = 0.5 * (sig('i') + ax * sig('x') + ay * sig('y') + az * sig('z'))
    return rho if not sparse else qonvert(rho, sparse=sparse)


def bell_state(n):
    """
    Generates one of the four bell-states;
    0: phi+, 1: phi-, 2: psi+, 3: psi- (singlet)
    """
    return (qonvert([1, 0, 0, 1], 'ket') / np.sqrt(2.0) if n == 0 else
            qonvert([0, 1, 1, 0], 'ket') / np.sqrt(2.0) if n == 2 else
            qonvert([1, 0, 0, -1], 'ket') / np.sqrt(2.0) if n == 1 else
            qonvert([0, 1, -1, 0], 'ket') / np.sqrt(2.0))


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


def groundstate(a):
    """
    # Returns the eigenvecor corresponding to the smallest eigenvalue of a
    """
    l0, v0 = spla.eigsh(a, k=1, which='SA')
    return qonvert(v0, 'ket')


def neel_state(n):
    binary = '01' * (n / 2)
    binary += (n % 2 == 1) * '0'  # add trailing spin for odd n
    return basis_vec(int(binary, 2), 2 ** n)


def singlet_pairs(n):
    return kronpow(bell_state(3), (n / 2))


def werner_state(p):
    return p * bell_state(3) * bell_state(3).H + (1 - p) * np.eye(4) / 4


def evals(a):
    """
    Returns the eigenvalues of a, sorted by ascending algebraic value
    """
    l = nla.eigvalsh(a)
    return np.sort(l)


def evecs(a):
    """
    Returns the eigenvectors of a as columns, sorted in ascending eigenvalue
    order
    """
    l, v = nla.eigh(a)
    sortinds = np.argsort(l)
    return qonvert(v[:, sortinds])


def esys(a):
    """
    Return the eigenvalues and eigenvectors of a, sorted according to
    increasing algebraic value
    """
    l, v = nla.eigh(a)
    sortinds = np.argsort(l)
    return l[sortinds], qonvert(v[:, sortinds])


def trx(p, dims, keep):
    """
    Perform partial trace on p, whose subsystems are given by dims, keeping
    only indices given by keep
    """
    # Cast as ndarrays for 2D+ reshaping
    if np.size(keep) == np.size(dims):
        if not isop(p):
            return p * p.H
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


def rk4_step(y0, f, dt, t=None):
    """
    Performs a 4th order runge-kutta step of length dt according to the
    relation dy/dt = f(t, y). If t is not specified then assumes f = f(y)
    """
    if t is None:
        k1 = f(y0)
        k2 = f(y0 + k1 * dt / 2.0)
        k3 = f(y0 + k2 * dt / 2.0)
        k4 = f(y0 + k3 * dt)
    else:
        k1 = f(y0, t)
        k2 = f(y0 + k1 * dt / 2.0, t + dt / 2.0)
        k3 = f(y0 + k2 * dt / 2.0, t + dt / 2.0)
        k4 = f(y0 + k3 * dt, t + dt)
    return y0 + (k1 + 2.0 * k2 + 2.0 * k3 + k4) * dt / 6.0


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


def ham_heis(n, jx=1, jy=1, jz=1, bz=0, periodic=False, sparse=False):
    """
    Constructs the heisenberg spin 1/2 hamiltonian, defaulting to isotropic
    coupling with no magnetic field and open boundary conditions.
    """
    dims = [2] * n
    sds = (jx * kron(sig('x', True), sig('x', True)) +
           jy * kron(sig('y', True), sig('y', True)) +
           jz * kron(sig('z', True), sig('z', True)) -
           bz * kron(sig('z', True), np.eye(2)))
    # Begin with last spin, not covered by loop
    ham = eyepad(-bz * sig('z', True), dims, n - 1)
    for i in range(n - 1):
        ham = ham + eyepad(sds, dims[:-1], i)
    if periodic:
        ham = ham + eyepad(sig('x', True), dims[:], [0, n - 1])  \
                  + eyepad(sig('y', True), dims[:], [0, n - 1])  \
                  + eyepad(sig('z', True), dims[:], [0, n - 1])
    if not sparse:
        ham = ham.todense()  # always construct sparse though
    return ham


def ezplot(x, y_i, fignum=1, xlog=False, ylog=False, **kwargs):
    """
    Function for automatically plotting multiple sets of data
    """
    # TODO colormap data and legend
    y_i = np.atleast_2d(np.squeeze(y_i))
    dimsy = np.array(np.shape(y_i))
    xaxis = np.argwhere(len(x) == dimsy)[0]  # 0 or 1
    fig = plt.figure(fignum, figsize=(8, 6), dpi=100)
    axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    colors = np.linspace(0, 1, dimsy[1 - xaxis])

    for i in range(dimsy[xaxis - 1]):
        if xaxis:
            y = y_i[i, :]
        else:
            y = y_i[:, i]
        if xlog:
            axes.set_xscale("log")
        if ylog:
            axes.set_yscale("log")
        axes.plot(x, y, '.-', c=cm.jet(colors[i], 1), **kwargs)
    return axes


def ldmul(v, m):
    '''
    Fast left diagonal multiplication of v: vector of diagonal matrix, and m
    '''
    v = v.reshape(np.size(v), 1)
    return ne.evaluate('v*m')


def rdmul(m, v):
    '''
    Fast right diagonal multiplication of v: vector of diagonal matrix, and m
    '''
    v = v.reshape(1, np.size(v))
    return ne.evaluate('m*v')
