"""
Core functions for manipulating quantum objects.
"""

from itertools import cycle
from functools import lru_cache
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


def isherm(qob):
    """ Checks if matrix is hermitian, for sparse or dense. """
    if sp.issparse(qob):
        # Since sparse, test that no .H elements are not unequal..
        return (qob != qob.H).nnz == 0
    return np.allclose(qob, qob.H)


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
    return kron(*(a for i in range(pwr)))


@lru_cache(maxsize=10)
def eye(n, sparse=False):
    """ Return identity of size n in complex format, optionally sparse"""
    return (sp.eye(n, dtype=complex, format='csr') if sparse else
            qjf(np.eye(n, dtype=complex)))


def coord_map(dims, coos, cyclic=False, trim=False):
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
        trim: if True, any coordinates beyond dimensions will be deleted,
            overidden by cyclic.

    Returns
    -------
        dims: flattened version of dims
        coos: indices mapped to flattened dims
    """
    # Calculate the raveled size of each dimension (i.e. size of 1 incr.)
    shp_dims = np.shape(dims)
    if len(shp_dims) == 1 and len(np.shape(coos)) == 1:
        coos = [[coo] for coo in coos]
    shp_mod = [np.prod(shp_dims[i+1:], dtype=int)
               for i in range(len(shp_dims))]
    coos = np.array(coos)
    if cyclic:
        coos = coos % shp_dims  # (broadcasting dims down columns)
    elif trim:
        coos = coos[np.all(coos == coos % shp_dims, axis=1)]
    elif np.any(coos != coos % shp_dims):
        raise ValueError('Coordinates beyond system dimensions.')
    # Sum contributions from each coordinate & flatten dimensions
    coos = np.sum(shp_mod * coos, axis=1)
    return np.ravel(dims), coos


def eyepad(ops, dims, inds, sparse=None):
    """
    Places several operators 'over' locations inds of dims. Automatically
    placing a large operator over several dimensions is allowed and a list
    of operators can be given which are then applied cyclically.

    Parameters
    ----------
        ops: operator or list of operators to put into the tensor space
        dims: dimensions of tensor space, use -1 to ignore dimension matching
        inds: indices of the dimenions to place operators on
        sparse: whether to construct the new operator in sparse form.

    Returns
    -------
        Operator such that ops act on dims[inds].
    """
    # TODO: place large operators over disjoint spaces, via permutataion?
    # TODO: allow -1 in dims to auto place *without* ind?
    # TODO: overlap and sort not working together
    if not isinstance(ops, (list, tuple)):
        ops = [ops]
    if np.ndim(dims) > 1:
        dims, inds = mapcoords(dims, inds)
    elif np.ndim(inds) == 0:
        inds = [inds]
    sparse = sp.issparse(ops[0]) if sparse is None else sparse
    inds, ops = zip(*sorted(zip(inds, cycle(ops))))
    inds, ops = set(inds), iter(ops)

    def gen_ops():
        cff_id = 1  # keeps track of compressing adjacent identities
        cff_ov = 1  # keeps track of overlaying op on multiple dimensions
        for ind, dim in enumerate(dims):
            if ind in inds:  # op should be placed here
                if cff_id > 1:  # need preceding identities
                    yield eye(cff_id, sparse=sparse)
                    cff_id = 1  # reset cumulative identity size
                if cff_ov == 1:  # first dim in placement block
                    op = next(ops)
                    sz_op = op.shape[0]
                if cff_ov * dim == sz_op or dim == -1:  # final dim-> place op
                    yield op
                    cff_ov = 1
                else:  # accumulate sub-dims
                    cff_ov *= dim
            elif cff_ov > 1:  # mid placing large operator
                cff_ov *= dim
            else:  # accumulate adjacent identites
                cff_id *= dim
        if cff_id > 1:  # trailing identities
            yield eye(cff_id, sparse=sparse)

    return kron(*gen_ops())


def eyeplace(*args, **kwargs):
    import warnings
    warnings.warn("deprecated", Warning)
    return eyepad(*args, **kwargs)


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
            .transpose([*perm, *(perm + len(dims))]) \
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
