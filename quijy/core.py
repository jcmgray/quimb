"""
Core functions for manipulating quantum objects.
"""

from itertools import cycle
from numba import jit
from numexpr import evaluate as evl
import numpy as np
import scipy.sparse as sp


def quijify(data, qtype=None, sparse=False, normalized=False, chopped=False):
    """ Converts lists to 'quantum' i.e. complex matrices, kets being columns.
    * Will unravel an array if 'ket' or 'bra' given.
    * Will conjugate if 'bra' given.
    * Will leave operators as is if 'dop' given, but construct one
    them if vector given with the assumption that it was a ket.

    Parameters
    ----------
        data:  list describing entries
        qtype: output type, either 'ket', 'bra' or 'dop' if given
        sparse: convert output to sparse 'csr' format
        normalized: normalise the output

    Returns
    -------
        x: numpy or sparse matrix
    """
    is_sparse_input = sp.issparse(data)
    if is_sparse_input:
        qob = sp.csr_matrix(data, dtype=complex)
    else:
        qob = np.matrix(data, copy=False, dtype=complex)
    if qtype is not None:
        if qtype in ('k', 'ket'):
            qob.shape = (np.prod(qob.shape), 1)
        elif qtype in ('b', 'bra'):
            qob.shape = (1, np.prod(qob.shape))
            qob = qob.conj()
        elif qtype in ('d', 'r', 'rho', 'op', 'dop') and not isop(qob):
            qob = quijify(qob, 'k') @ quijify(qob, 'k').H
    if chopped:
        chop(qob, inplace=True)
    if normalized:
        normalize(qob, inplace=True)
    return sp.csr_matrix(qob, dtype=complex) if sparse else qob

qjf = quijify


@jit
def isket(qob):
    """ Checks if matrix is in ket form, i.e. a matrix column. """
    return qob.shape[0] > 1 and qob.shape[1] == 1  # Column vector check


@jit
def isbra(qob):
    """ Checks if matrix is in bra form, i.e. a matrix row. """
    return qob.shape[0] == 1 and qob.shape[1] > 1  # Row vector check


@jit
def isop(qob):
    """ Checks if matrix is an operator, i.e. a square matrix. """
    m, n = qob.shape
    return m == n and m > 1  # Square matrix check


def isherm(a):
    """ Checks if matrix is hermitian, for sparse or dense. """
    if sp.issparse(a):
        # Since sparse, test that no .H elements are not unequal..
        return (a != a.H).nnz == 0
    return np.allclose(a, a.H)


@jit
def trace(op):
    """
    Trace of hermitian matrix. This is faster than numpy's
    built-in trace function for real diagonals.
    """
    x = 0.0
    for i in range(op.shape[0]):
        x += op[i, i].real
    return x


def sparse_trace(op):
    """ Trace of sparse hermitian matrix. """
    d = op.diagonal()
    return np.sum(d.real)


def tr(op):
    return sparse_trace(op) if sp.issparse(op) else trace(op)

# Monkey-patch trace methods
np.matrix.tr = trace
sp.csr_matrix.tr = sparse_trace


def normalize(qob, inplace=False):
    """ Returns the state qob in normalized form """
    n_factor = ((qob.H @ qob)[0, 0]**0.5 if isket(qob) else
                (qob @ qob.H)[0, 0]**0.5 if isbra(qob) else
                qob.tr())
    if inplace:
        qob /= n_factor
    else:
        return qob / n_factor

nmlz = normalize

np.matrix.nmlz = nmlz
sp.csr_matrix.nmlz = nmlz


@jit
def kron_dense(a, b):
    """
    Fast tensor product of two dense arrays (Fast than numpy using jit)
    """
    m, n = a.shape
    p, q = b.shape
    x = np.empty((m * p, n * q), dtype=np.complex128)
    for i in range(m):
        for j in range(n):
            x[i * p:(i + 1)*p, j * q:(j + 1) * q] = a[i, j] * b
    return np.matrix(x, copy=False)


def kron(*ops):
    """ Tensor product of variable number of arguments.
    Input:
        ops: objects to be tensored together
    Returns:
        operator
    The product is performed as (a * (b * (c * ...)))
    """
    num_p = len(ops)
    if num_p == 0:
        return 1
    a = ops[0]
    if num_p == 1:
        return a
    b = ops[1] if num_p == 2 else kron(*ops[1:])
    if sp.issparse(a) or sp.issparse(b):
        return sp.kron(a, b, 'csr')
    else:
        return kron_dense(a, b)

# Monkey-patch unused & symbol to tensor product
np.matrix.__and__ = kron
sp.csr_matrix.__and__ = kron


def kronpow(a, pwr):
    """ Returns 'a' tensored with itself pwr times """
    return (1 if pwr == 0 else
            a if pwr == 1 else
            kron(*[a] * pwr))


def eye(n, sparse=False):
    """ Return identity of size n in complex format, optionally sparse"""
    return (sp.eye(n, dtype=complex, format='csr') if sparse else
            qjf(np.eye(n, dtype=complex)))


def mapcoords(dims, coos, cyclic=False, trim=None):
    """
    Maps multi-dimensional coordinates and indices to flat arrays in a
    regular way. Wraps or deletes coordinates beyond the system size
    depending on parameters `cyclic` and `trim`.

    Parameters
    ----------
        dims: multi-dim array of systems' internal dimensions
        coos: array of coordinate tuples to convert
        cyclic: whether to automatically wrap coordinates beyond system size or
            delete them.
        trim: if not None, coos will be bound-checked. trim=True will delete
            any coordinates beyond dimensions, trim=False will raise an error.

    Returns
    -------
        dims: flattened version of dims
        coos: indices mapped to flattened dims

    Examples
    --------
    >>> dims = ([[10, 11, 12],
                 [13, 14, 15]])
    >>> coos = [(1, 1), (1, 2), (2, 1)]
    >>> ndims, ncoos = flatcoords(dims, coos, cyclic=True)
    >>> ndims[ncoos]
    array([14, 15, 11])
    """
    # TODO: compress coords? (argsort and merge identities)
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


def eyepad(op, dims, inds, sparse=None):
    """ Pad an operator with identities to act on particular subsystem.

    Parameters
    ----------
        op: operator to act with
        dims: list of dimensions of subsystems.
        inds: indices of dims to act op on.
        sparse: whether output should be sparse

    Returns
    -------
        bop: operator with op acting on each subsystem specified by inds
    Note that the actual numbers in dims[inds] are ignored and the size of
    op is assumed to match. Sparsity of the output can be inferred from
    input if not specified.

    Examples
    --------
    >>> X = sig('x')
    >>> b1 = kron(X, eye(2), X, eye(2))
    >>> b2 = eyepad(X, dims=[2,2,2,2], inds=[0,2])
    >>> allclose(b1, b2)
    True
    """
    inds = np.array(inds, ndmin=1)
    sparse = sp.issparse(op) if sparse is None else sparse  # infer sparsity
    bop = eye(np.prod(dims[0:inds[0]]), sparse=sparse)
    for i in range(len(inds) - 1):
        bop = kron(bop, op)
        pad_size = np.prod(dims[inds[i] + 1:inds[i + 1]])
        bop = kron(bop, eye(pad_size, sparse=sparse))
    bop = kron(bop, op)
    pad_size = np.prod(dims[inds[-1] + 1:])
    bop = kron(bop, eye(pad_size, sparse=sparse))
    return bop


def eyeplace(ops, dims, inds, sparse=None):
    """
    Places the operator(s) ops 'over' locations inds of dims. Automatically
    placing a large operator over several dimensions is allowed and a list
    of operators can be given which are then applied cyclically.

    Parameters
    ----------
        ops: operator or list of operators to put into the tensor space

        dims: dimensions of tensor space, use None to ignore dimension matching
        inds: indices of the dimenions to place operators on
        sparse: whether to construct the new operator in sparse form.

    Returns
    -------
        Operator such that acts on dims[inds].
    """
    sparse = sp.issparse(ops) if sparse is None else sparse  # infer sparsity
    inds = np.array(inds, ndmin=1)
    ops_cyc = cycle(ops)

    def gen_ops():
        op = next(ops_cyc)
        op_sz = op.shape[0]
        overlap_factor = 1
        for i, dim in enumerate(dims):
            if i in inds:
                if op_sz == overlap_factor * dim or dim is None:
                    yield op
                    op = next(ops_cyc)  # reset
                    op_sz = op.shape[0]
                    overlap_factor = 1
                else:
                    # 'merge' dimensions to get bigger size
                    overlap_factor *= dim
            else:
                yield eye(dim, sparse=sparse)

    return kron(*gen_ops())


def permute_subsystems(p, dims, perm):
    """
    Permute the subsytems of a state.

    Parameters
    ----------
        p: state, vector or operator
        dims: dimensions of the system
        perm: new order of indexes range(len(dims))

    Returns
    -------
        pp: permuted state, vector or operator
    """
    p = np.array(p, copy=False)
    perm = np.array(perm)
    d = np.prod(dims)
    if isop(p):
        p = p.reshape([*dims, *dims]) \
            .transpose([*perm, *(perm+len(dims))]) \
            .reshape((d, d))
    else:
        p = p.reshape(dims) \
            .transpose(perm) \
            .reshape((d, 1))
    return np.matrix(p, copy=False)


def partial_trace(p, dims, keep):
    """ Perform partial trace.

    Parameters
    ----------
        p: state to perform partial trace on, vector or operator
        dims: list of subsystem dimensions
        keep: index of subsytems to keep

    Returns
    -------
        Density matrix of subsytem dimensions dims[keep]
    """
    # TODO:  partial trace for sparse matrices
    # Cast as ndarrays for 2D+ reshaping
    if np.size(keep) == np.size(dims):  # keep all subsystems
        if not isop(p):
            return p @ p.H  # but return as density operator for consistency
        return p
    n = np.size(dims)
    dims = np.array(dims, ndmin=1)
    keep = np.array(keep, ndmin=1)
    lose = np.delete(range(n), keep)
    dimkeep = np.prod(dims[keep])
    dimlose = np.prod(dims[lose])
    # Permute dimensions into block of keep and block of lose
    perm = np.array([*keep, *lose])
    # Apply permutation to state and trace out block of lose
    if not isop(p):  # p = psi
        p = np.array(p)
        p = p.reshape(dims) \
            .transpose(perm) \
            .reshape([dimkeep, dimlose])
        p = np.matrix(p, copy=False)
        return p @ p.H
    else:  # p = rho
        p = np.array(p)
        p = p.reshape((*dims, *dims)) \
            .transpose((*perm, *(perm + n))) \
            .reshape([dimkeep, dimlose, dimkeep, dimlose]) \
            .trace(axis1=1, axis2=3)
        return np.matrix(p, copy=False)

ptr = partial_trace
trx = partial_trace
np.matrix.ptr = partial_trace


def chop(x, tol=1.0e-15, inplace=True):
    """
    Set small values of an array to zero.

    Parameters
    ----------
        x: dense or sparse matrix/array.
        tol: fraction of max(abs(x)) to chop below.
        inplace: whether to act on input array or return copy

    Returns
    -------
        None if inplace else chopped matrix
    """
    minm = np.abs(x).max() * tol  # minimum value tolerated
    if not inplace:
        x = x.copy()
    if sp.issparse(x):
        x.data.real[np.abs(x.data.real) < minm] = 0.0
        x.data.imag[np.abs(x.data.imag) < minm] = 0.0
        x.eliminate_zeros()
    else:
        x.real[np.abs(x.real) < minm] = 0.0
        x.imag[np.abs(x.imag) < minm] = 0.0
    return None if inplace else x


def ldmul(vec, mat):
    '''
    Fast left diagonal multiplication using numexpr,
    faster than numpy for n > ~ 500.

    Parameters
    ----------
        vec: vector of diagonal matrix, can be array
        mat: matrix

    Returns
    -------
        mat: np.matrix
    '''
    d = mat.shape[0]
    vec = vec.reshape(d, 1)
    if d > 500:
        return np.matrix(evl('vec*mat'), copy=False)
    else:
        return np.matrix(np.multiply(vec, mat), copy=False)

def rdmul(mat, vec):
    '''
    Fast right diagonal multiplication using numexpr,
    faster than numpy for n > ~ 500.

    Parameters
    ----------
        mat: matrix
        vec: vector of diagonal matrix, can be array

    Returns
    -------
        mat: np.matrix
    '''
    d = mat.shape[0]
    vec = vec.reshape(1, d)
    if d > 500:
        return np.matrix(evl('mat*vec'), copy=False)
    else:
        return np.matrix(np.multiply(mat, vec), copy=False)


def infer_size(p, base=2):
    """ Infers the size of a state assumed to be made of qubits """
    d = max(p.shape)
    return int(np.log2(d) / np.log2(base))


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
