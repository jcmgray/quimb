"""
Core functions for manipulating quantum objects.
"""

import numpy as np
import scipy.sparse as sp
from numexpr import evaluate as evl
from numba import jit
from itertools import cycle


def quijify(data, qtype=None, sparse=False, nrmlzd=False, chopd=False):
    """ Converts lists to 'quantum' i.e. complex matrices, kets being columns.
    Input:
        data:  list describing entries
        qtype: output type, either 'ket', 'bra' or 'dop' if given
        sparse: convert output to sparse 'csr' format
        nrmlzd: normalise the output
    Returns:
        x: numpy or sparse matrix
    * Will unravel an array if 'ket' or 'bra' given.
    * Will conjugate if 'bra' given.
    * Will leave operators as is if 'dop' given, but construct one
    them if vector given.
    TODO: convert sparse vector to sparse operator
    """
    p = np.matrix(data, copy=False, dtype=complex)
    if chopd:
        chop(p)
    if qtype is not None:
        sz = np.prod(p.shape)
        if qtype in ('k', 'ket'):
            p.shape = (sz, 1)
        elif qtype in ('b', 'bra'):
            p.shape = (1, sz)
            p = np.conj(p)
        elif qtype in ('p', 'd', 'r', 'rho', 'op', 'dop') and not isop(p):
            p = quijify(p, 'k') * quijify(p, 'k').H
    if nrmlzd:
        p = nrmlz(p)
    return sp.csr_matrix(p, dtype=complex) if sparse else p

qjf = quijify


@jit
def isket(p):
    """ Checks if matrix is in ket form, i.e. a column """
    return p.shape[0] > 1 and p.shape[1] == 1  # Column vector check


@jit
def isbra(p):
    """ Checks if matrix is in bra form, i.e. a row """
    return p.shape[0] == 1 and p.shape[1] > 1  # Row vector check


@jit
def isop(p):
    """ Checks if matrix is an operator, i.e. square """
    m, n = p.shape
    return m == n and m > 1  # Square matrix check


def isherm(a):
    """ Checks if matrix is hermitian, for sparse or dense"""
    if sp.issparse(a):
        # Since sparse, test that no .H elements are not unequal..
        return (a != a.H).nnz == 0
    return np.allclose(a, a.H)


@jit
def trace(a):
    """ Trace of hermitian matrix (jit version faster than numpy!)
    TODO: sparse method
    """
    x = 0.0
    for i in range(a.shape[0]):
        x += a[i, i].real
    return x

tr = trace


def normalize(p):
    """ Returns the state p in normalized form """
    return (p / (p.H * p)[0, 0]**0.5 if isket(p) else
            p / (p * p.H)[0, 0]**0.5 if isbra(p) else
            p / tr(p))

nrmlz = normalize


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
    return np.matrix(x, copy=False)
# def krnd2(a, b):  # Fallback for debugging
#     return kron(a, b)


def kron(*ps):
    """ Tensor product of variable number of arguments.
    Input:
        ps: objects to be tensored together
    Returns:
        operator
    The product is performed as (a * (b * (c * ...)))
    """
    num_p = len(ps)
    a = ps[0]
    if num_p == 1:
        return a
    b = ps[1] if num_p == 2 else  \
        kron(*ps[1:])  # Recursively perform kron to 'right'
    return (sp.kron(a, b, 'csr') if (sp.issparse(a) or sp.issparse(b)) else
            krnd2(a, b))


def kronpow(a, pwr):
    """ Returns 'a' tensored with itself pwr times """
    return (1 if pwr == 0 else
            a if pwr == 1 else
            kron(*[a] * pwr))


def eye(n, sparse=False):
    """ Return identity of size n in complex format, optionally sparse"""
    return (sp.eye(n, dtype=complex, format='csr') if sparse else
            np.eye(n, dtype=complex))


def mapcoords(dims, coos, cyclic=False, trim=None):
    """
    Maps multi-dimensional coordinates and indices to flat arrays in a
    regular way. Wraps or deletes coordinates beyond the system size
    depending on parameters `cyclic` and `trim`.
    INPUTS:
        dims: multi-dim array of systems' internal dimensions
        coos: array of coordinate tuples to convert
        cyclic: whether to automatically wrap coordinates beyond system size or
            delete them.
        trim: if not None, coos will be bound-checked. trim=True will delete
            any coordinates beyond dimensions, trim=False will raise an error.
    OUTPUTS:
        dims: flattened version of dims
        coos: indices mapped to flattened dims
    EXAMPLE:
    >>> dims = ([[10, 11, 12],
                 [13, 14, 15]])
    >>> coos = [(1, 1), (1, 2), (2, 1)]
    >>> ndims, ncoos = flatcoords(dims, coos, cyclic=True)
    >>> ndims[ncoos]
    array([14, 15, 11])
    """
    # Calculate the raveled size of each dimension (i.e. size of 1 incr.)
    shp_dims = np.shape(dims)
    shp_mod = [np.prod(shp_dims[i+1:]) for i in range(len(shp_dims)-1)] + [1]
    coos = np.array(coos)
    if cyclic:
        coos = coos % shp_dims  # (broadcasting dims down columns)
    elif trim is not None:
        if trim:  # delete coordinates which overspill
            coos = coos[np.all(coos == coos % shp_dims, axis=1)]
        elif np.any(coos != coos % shp_dims):
            raise ValueError('Coordinates beyond system dimensions.')
    # Sum contributions from each coordinate & flatten dimensions
    coos = np.sum(shp_mod * coos, axis=1)
    return np.ravel(dims), coos


def eyepad(a, dims, inds, sparse=None):
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
    input if not specified.
    e.g.
    >>> X = sig('x')
    >>> b1 = kron(X, eye(2), X, eye(2))
    >>> b2 = eyepad(X, dims=[2]*4, inds=[0,2])
    >>> allclose(b1, b2)
    True
    """
    sparse = sp.issparse(a) if sparse is None else sparse  # infer sparsity
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


def eyeplace(a, dims, inds, sparse=None):
    """
    Places the operator(s) a 'over' locations inds of dims. Automatically
    placing a large operator over several dimensions is allowed and a list
    of operators can be given which are then applied cyclically.
    INPUTS:
        a: operator or list of operators to put into the tensor space

        dims: dimensions of tensor space, use None to ignore dimension matching
        inds: indices of the dimenions to place operators on
        sparse: whether to construct the new operator in sparse form.
    """
    sparse = sp.issparse(a) if sparse is None else sparse  # infer sparsity
    inds = np.array(inds, ndmin=1)
    ops = cycle(a)

    def gen_ops():
        op = next(ops)
        op_sz = op.shape[0]
        overlap_factor = 1
        for i, dim in enumerate(dims):
            if i in inds:
                if op_sz == overlap_factor * dim or dim is None:
                    yield op
                    op = next(ops)  # reset
                    op_sz = op.shape[0]
                    overlap_factor = 1
                else:
                    # 'merge' dimensions to get bigger size
                    overlap_factor *= dim
            else:
                yield eye(dim, sparse=sparse)

    return kron(*gen_ops())


def partial_trace(p, dims, keep):
    """ Perform partial trace.
    Input:
        p: state to perform partial trace on, vector or operator
        dims: list of subsystem dimensions
        keep: index of subsytems to keep
    Returns:
        Density matrix of subsytem dimensions dims[keep]
    """
    #TODO:  partial trace for sparse matrices
    # Cast as ndarrays for 2D+ reshaping
    if np.size(keep) == np.size(dims):  # keep all subsystems
        if not isop(p):
            return p * p.H  # but return as density operator for consistency
        return p
    n = np.size(dims)
    dims = np.array(dims, ndmin=1)
    keep = np.array(keep, ndmin=1)
    lose = np.delete(range(n), keep)
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
        p = np.matrix(p, copy=True)
        return quijify(p * p.H)
    else:  # p = rho
        p = np.array(p)
        p = p.reshape(np.r_[dims, dims]) \
            .transpose(np.r_[perm, perm + n]) \
            .reshape([dimkeep, dimlose, dimkeep, dimlose]) \
            .trace(axis1=1, axis2=3)
        return quijify(p)

ptr = partial_trace
trx = partial_trace


def chop(x, tol=1.0e-15):
    """
    Sets any values of x smaller than tol (relative to range(x)) to zero.
    Acts in-place on array!
    """
    rnge = abs(x.max() - x.min())
    minm = rnge * tol  # minimum value tolerated
    if sp.issparse(x):
        x.data.real[np.abs(x.data.real) < minm] = 0.0
        x.data.imag[np.abs(x.data.imag) < minm] = 0.0
        x.eliminate_zeros()
    else:
        x.real[np.abs(x.real) < minm] = 0.0
        x.imag[np.abs(x.imag) < minm] = 0.0


def ldmul(v, m):
    '''
    Fast left diagonal multiplication using numexpr
    Args:
        v: vector of diagonal matrix, can be array
        m: matrix
    '''
    v = v.reshape(np.size(v), 1)
    return evl('v*m')


def rdmul(m, v):
    '''
    Fast right diagonal multiplication using numexpr
    Args:
        m: matrix
        v: vector of diagonal matrix, can be array
    '''
    v = v.reshape(1, np.size(v))
    return evl('m*v')


def comm(a, b):
    """ Commutator of two matrices """
    return a * b - b * a


def infer_size(p, base=2):
    """ Infers the size of a state assumed to be made of qubits """
    d = max(p.shape)
    return int(np.log2(d) / np.log2(base))
