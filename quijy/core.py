"""
Core functions for manipulating quantum objects.
"""

from math import log
from operator import mul
from itertools import cycle, groupby, product
from functools import reduce
import numpy as np
from numpy.matlib import zeros
import scipy.sparse as sp
from numba import jit
from numexpr import evaluate as evl


def quijify(data, qtype=None, sparse=False, normalized=False, chopped=False):
    """ Converts lists to 'quantum' i.e. complex matrices, kets being columns.
    * Will unravel an array if 'ket' or 'bra' given.
    * Will conjugate if 'bra' given.
    * Will leave operators as is if 'dop' given, but construct one
    if vector given with the assumption that it was a ket.

    Parameters
    ----------
        data:  list describing entries
        qtype: output type, either 'ket', 'bra' or 'dop' if given
        sparse: convert output to sparse 'csr' format
        normalized: normalise the output

    Returns
    -------
        x: numpy or sparse matrix """
    is_sparse_input = issparse(data)
    if is_sparse_input:
        qob = sp.csr_matrix(data, dtype=complex)
    else:
        qob = np.matrix(data, copy=False, dtype=complex)
    if qtype is not None:
        if qtype in {"k", "ket"}:
            qob.shape = (np.prod(qob.shape), 1)
        elif qtype in {"b", "bra"}:
            qob.shape = (1, np.prod(qob.shape))
            qob = qob.conj()
        elif qtype in {"d", "r", "rho", "op", "dop"} and not isop(qob):
            qob = quijify(qob, "k") @ quijify(qob, "k").H
    if chopped:
        chop(qob, inplace=True)
    if normalized:
        normalize(qob, inplace=True)
    return sp.csr_matrix(qob, dtype=complex) if sparse else qob

qjf = quijify


def matrixify(foo):
    """ To decorate functions returning ndarrays. """
    return lambda *args, **kwargs: np.asmatrix(foo(*args, **kwargs))


def realify(foo, imag_tol=1.0e-14):
    """ To decorate functions that should return float for small complex. """
    def realified_foo(*args, **kwargs):
        x = foo(*args, **kwargs)
        return x.real if abs(x.imag) < abs(x.real) * imag_tol else x
    return realified_foo


def issparse(x):
    """ Checks if object is scipy sparse format. """
    return isinstance(x, sp.spmatrix)


def isket(qob):
    """ Checks if matrix is in ket form, i.e. a matrix column. """
    return qob.shape[0] > 1 and qob.shape[1] == 1  # Column vector check


def isbra(qob):
    """ Checks if matrix is in bra form, i.e. a matrix row. """
    return qob.shape[0] == 1 and qob.shape[1] > 1  # Row vector check


def isop(qob):
    """ Checks if matrix is an operator, i.e. a square matrix. """
    m, n = qob.shape
    return m == n and m > 1  # Square matrix check


def isherm(qob):
    """ Checks if matrix is hermitian, for sparse or dense. """
    return ((qob != qob.H).nnz == 0 if issparse(qob) else
            np.allclose(qob, qob.H))


def issmall(p, sz=500):
    return p.shape[0] < sz and p.shape[1] < sz


def infer_size(p, base=2):
    """ Infers the size of a state assumed to be made of qubits """
    return int(log(max(p.shape), base))


@realify
@jit(nopython=True)
def trace_dense(op):  # pragma: no cover
    """ Trace of matrix. """
    x = 0.0
    for i in range(op.shape[0]):
        x += op[i, i]
    return x


@realify
def trace_sparse(op):
    """ Trace of sparse matrix. """
    return np.sum(op.diagonal())


def trace(op):
    """ Trace of dense or sparse matrix """
    return trace_sparse(op) if issparse(op) else trace_dense(op)

# Monkey-patch trace methods
tr = trace
np.matrix.tr = trace_dense
sp.csr_matrix.tr = trace_sparse


def normalize(qob, inplace=True):
    """ Returns the state qob in normalized form """
    n_factor = qob.tr() if isop(qob) else inner(qob, qob)**0.25
    if inplace:
        qob /= n_factor
        return qob
    return qob / n_factor

# Monkey-patch normalise methods
nmlz = normalize
np.matrix.nmlz = nmlz
sp.csr_matrix.nmlz = nmlz


@matrixify
@jit(nopython=True)
def kron_dense(a, b):  # pragma: no cover
    """ Fast tensor product of two dense arrays. """
    m, n = a.shape
    p, q = b.shape
    x = np.empty((m * p, n * q), dtype=np.complex128)
    for i in range(m):
        for j in range(n):
            x[i * p:(i + 1)*p, j * q:(j + 1) * q] = a[i, j] * b
    return x


def kron_sparse(a, b):
    """ Sparse tensor product """
    return sp.kron(a, b, format="csr")


def kron(*ops):
    # TODO: scalar?
    # TODO: faster sparse kron using coo?
    # TODO: rename? tensor
    # TODO: parallize?
    # TODO: merge into eyepad with dims=None, coos=None
    """ Tensor product of variable number of arguments.
    Input:
        ops: objects to be tensored together
    Returns:
        operator
    The product is performed as (a * (b * (c * ...))) """
    num_p = len(ops)
    if num_p == 0:
        return 1
    a = ops[0]
    if num_p == 1:
        return a
    b = ops[1] if num_p == 2 else kron(*ops[1:])
    if issparse(a) or issparse(b):
        return kron_sparse(a, b)
    return kron_dense(a, b)

# Monkey-patch unused & symbol to tensor product
np.matrix.__and__ = kron_dense
sp.csr_matrix.__and__ = kron_sparse


def kronpow(a, pwr):
    """ Returns `a` tensored with itself pwr times """
    return kron(*(a for i in range(pwr)))


def coo_map(dims, coos, cyclic=False, trim=False):
    """ Maps multi-dimensional coordinates and indices to flat arrays in a
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
        coos: indices mapped to flattened dims """
    dims = np.asarray(dims)
    shp_dims, n_dims = dims.shape, dims.ndim
    # Calculate 'shape multiplier' for each dimension
    shp_mul = [np.prod(shp_dims[i+1:], dtype=int) for i in range(n_dims)]
    coos = np.asarray(coos).reshape((len(coos), n_dims))
    if cyclic:
        coos = coos % shp_dims  # (broadcasting dims down columns)
    elif trim:
        coos = coos[np.all(coos == coos % shp_dims, axis=1)]
    elif np.any(coos != coos % shp_dims):
        raise ValueError("Coordinates beyond system dimensions.")
    # Sum contributions from each coordinate & flatten dimensions
    coos = np.sum(shp_mul * coos, axis=1)
    return np.ravel(dims), coos


def coo_compress(dims, inds):
    """ Compresses identity spaces together: groups 1D dimensions according to
    whether their index appears in inds, then merges the groups. """
    try:
        inds = {*inds}
        grp_dims = groupby(range(len(dims)), lambda x: x in inds)
    except TypeError:  # single index given
        grp_dims = groupby(range(len(dims)), lambda x: x == inds)
    ks, gs = zip(*(((k, tuple(g)) for k, g in grp_dims)))
    ndims = [reduce(mul, (dims[i] for i in g)) for g in gs]
    ncoos = [i for i in range(len(ndims)) if ks[i]]
    return ndims, ncoos


@matrixify
@jit(nopython=True)
def identity_dense(d):  # pragma: no cover
    """ Returns a dense, complex identity of order d. """
    x = np.zeros((d, d), dtype=np.complex128)
    for i in range(d):
        x[i, i] = 1
    return x


def identity_sparse(d):
    """ Returns a sparse, complex identity of order d. """
    return sp.eye(d, dtype=complex, format="csr")


def identity(d, sparse=False):
    """ Return identity of size d in complex format, optionally sparse"""
    return identity_sparse(d) if sparse else identity_dense(d)

eye = identity


def eyepad(ops, dims, inds, sparse=None):
    # TODO: rename? itensor, tensor
    # TODO: test 2d+ dims and coos
    # TODO: compress coords?
    # TODO: allow -1 in dims to auto place *without* ind?
    """ Places several operators 'over' locations inds of dims. Automatically
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
        Operator such that ops act on dims[inds]. """
    if isinstance(ops, (np.ndarray, sp.spmatrix)):
        ops = [ops]
    if np.ndim(dims) > 1:
        dims, inds = coo_map(dims, inds)
    elif np.ndim(inds) == 0:
        inds = [inds]
    sparse = issparse(ops[0]) if sparse is None else sparse
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


@matrixify
def perm_pad(op, dims, inds):
    # TODO: multiple ops
    # TODO: coo map, coo compress
    # TODO: sparse??
    # TODO: use permute
    """ Advanced tensor placement of operators that allows arbitrary ordering
    such as reversal and interleaving of identities. """
    dims, inds = np.asarray(dims), np.asarray(inds)
    n = len(dims)  # number of subsytems
    sz = np.prod(dims)  # Total size of system
    dims_in = dims[inds]
    sz_in = np.prod(dims_in)  # total size of operator space
    sz_out = sz // sz_in  # total size of identity space
    sz_op = op.shape[0]  # size of individual operator
    n_op = int(log(sz_in, sz_op))  # number of individual operators
    b = np.asarray(kronpow(op, n_op) & eye(sz_out))
    inds_out, dims_out = zip(*((i, x) for i, x in enumerate(dims)
                               if i not in inds))  # inverse of inds
    p = [*inds, *inds_out]  # current order of system
    dims_cur = [*dims_in, *dims_out]
    ip = np.empty(n, dtype=np.int)
    ip[p] = np.arange(n)  # inverse permutation
    return b.reshape([*dims_cur, *dims_cur])  \
            .transpose([*ip, *(ip + n)])  \
            .reshape([sz, sz])


@matrixify
def permute_dense(p, dims, perm):
    """ Permute the subsytems of a dense matrix. """
    p, perm = np.asarray(p), np.asarray(perm)
    d = np.prod(dims)
    if isop(p):
        return p.reshape([*dims, *dims]) \
                .transpose([*perm, *(perm + len(dims))]) \
                .reshape((d, d))
    return p.reshape(dims) \
            .transpose(perm) \
            .reshape((d, 1))


def permute_sparse(a, dims, perm):
    """ Permute the subsytems of a sparse matrix. """
    perm, dims = np.asarray(perm), np.asarray(dims)
    ndims = dims[perm]
    # New dimensions & stride (i.e. product of preceding dimensions)
    odim_stride = np.asarray([np.prod(dims[i+1:])
                              for i, _ in enumerate(dims)])
    ndim_stride = np.asarray([np.prod(ndims[i+1:])
                              for i, _ in enumerate(ndims)])
    # Range of possible coordinates for each subsys
    coos = (tuple(range(dim)) for dim in dims)
    # Complete basis using coordinates for current and new dimensions
    basis = np.asarray(tuple(product(*coos, repeat=1)))
    oinds = np.sum(odim_stride * basis, axis=1)
    ninds = np.sum(ndim_stride * basis[:, perm], axis=1)
    # Construct permutation matrix and apply it to state
    perm_mat = sp.coo_matrix((np.ones(a.shape[0]), (ninds, oinds))).tocsr()
    if isop(a):
        return (perm_mat @ a) @ perm_mat.H
    return perm_mat @ a


def permute(a, dims, perm):
    """ Permute the subsytems of state a.

    Parameters
    ----------
        p: state, vector or operator
        dims: dimensions of the system
        perm: new order of indexes range(len(dims))

    Returns
    -------
        pp: permuted state, vector or operator"""
    if issparse(a):
        return permute_sparse(a, dims, perm)
    return permute_dense(a, dims, perm)


def partial_trace_clever(p, dims, keep):
    # TODO: compress coords?
    # TODO: user tensordot for vec
    """ Perform partial trace.

    Parameters
    ----------
        p: state to perform partial trace on, vector or operator
        dims: list of subsystem dimensions
        keep: index of subsytems to keep

    Returns
    -------
        Density matrix of subsytem dimensions dims[keep] """
    dims, keep = np.array(dims, ndmin=1), np.array(keep, ndmin=1)
    n = len(dims)
    lose = np.delete(range(n), keep)
    sz_keep, sz_lose = np.prod(dims[keep]), np.prod(dims[lose])
    # Permute dimensions into block of keep and block of lose
    perm = np.asarray((*keep, *lose))
    # Apply permutation to state and trace out block of lose
    if not isop(p):  # p = psi
        p = np.asarray(p).reshape(dims) \
            .transpose(perm) \
            .reshape((sz_keep, sz_lose))
        p = np.asmatrix(p)
        return accel_dot(p, p.H)
    else:
        p = np.asarray(p).reshape((*dims, *dims)) \
            .transpose((*perm, *(perm + n))) \
            .reshape((sz_keep, sz_lose, sz_keep, sz_lose)) \
            .trace(axis1=1, axis2=3)
        return np.asmatrix(p)


def trace_lose(p, dims, coo_lose):
    """ Simple partial trace where the single subsytem at coo_lose is traced
    out. """
    p = p if isop(p) else p @ p.H
    dims = np.asarray(dims)
    e = dims[coo_lose]
    a = np.prod(dims[:coo_lose], dtype=int)
    b = np.prod(dims[coo_lose+1:], dtype=int)
    rhos = zeros(shape=(a * b, a * b), dtype=np.complex128)
    for i in range(a * b):
        for j in range(i, a * b):
            rhos[i, j] = trace(p[
                    e * b * (i // b) + (i % b):
                    e * b * (i // b) + (i % b) + (e - 1) * b + 1: b,
                    e * b * (j // b) + (j % b):
                    e * b * (j // b) + (j % b) + (e - 1) * b + 1: b])
            if j != i:
                rhos[j, i] = rhos[i, j].conjugate()
    return rhos


def trace_keep(p, dims, coo_keep):
    """ Simple partial trace where the single subsytem at coo_keep is kept. """
    p = p if isop(p) else p @ p.H
    dims = np.asarray(dims)
    s = dims[coo_keep]
    a = np.prod(dims[:coo_keep], dtype=int)
    b = np.prod(dims[coo_keep+1:], dtype=int)
    rhos = zeros(shape=(s, s), dtype=np.complex128)
    for i in range(s):
        for j in range(i, s):
            for k in range(a):
                rhos[i, j] += trace(p[
                        b * i + s * b * k: b * i + s * b * k + b,
                        b * j + s * b * k: b * j + s * b * k + b])
            if j != i:
                rhos[j, i] = rhos[i, j].conjugate()
    return rhos


def partial_trace_simple(p, dims, coos_keep):
    """ Simple partial trace made up of consecutive single subsystem partial
    traces, augmented by 'compressing' the dimensions each time. """
    p = p if isop(p) else p @ p.H
    dims, coos_keep = coo_compress(dims, coos_keep)
    if len(coos_keep) == 1:
        return trace_keep(p, dims, *coos_keep)
    lmax = max(enumerate(dims),
               key=lambda ix: (ix[0] not in coos_keep)*ix[1])[0]
    p = trace_lose(p, dims, lmax)
    dims = [*dims[:lmax], *dims[lmax+1:]]
    coos_keep = {(ind if ind < lmax else ind - 1) for ind in coos_keep}
    return partial_trace_simple(p, dims, coos_keep)


def partial_trace(p, dims, coos):
    """ Partial trace of a dense or sparse state.

    Parameters
    ----------
        p: state
        dims: list of dimensions of subsystems
        coos: coordinates of subsytems to keep

    Returns
    -------
        rhoab: density matrix of remaining subsytems,"""
    if issparse(p):
        return partial_trace_simple(p, dims, coos)
    return partial_trace_clever(p, dims, coos)

ptr = partial_trace
trx = partial_trace
np.matrix.ptr = partial_trace_clever
sp.csr_matrix.ptr = partial_trace_simple


def chop(x, tol=1.0e-15, inplace=True):
    """ Set small values of an array to zero.

    Parameters
    ----------
        x: dense or sparse matrix/array.
        tol: fraction of max(abs(x)) to chop below.
        inplace: whether to act on input array or return copy

    Returns
    -------
        None if inplace else chopped matrix """
    minm = np.abs(x).max() * tol  # minimum value tolerated
    if not inplace:
        x = x.copy()
    if issparse(x):
        x.data.real[np.abs(x.data.real) < minm] = 0.0
        x.data.imag[np.abs(x.data.imag) < minm] = 0.0
        x.eliminate_zeros()
    else:
        x.real[np.abs(x.real) < minm] = 0.0
        x.imag[np.abs(x.imag) < minm] = 0.0
    return x

# np.matrix.chop = chop
# sp.csr_matrix = chop


# -------------------------------------------------------------------------- #
# Functions accelerated over numpy                                           #
# -------------------------------------------------------------------------- #

@matrixify
@jit(nopython=True)
def accel_mul(x, y):  # pragma: no cover
    """ Accelerated element-wise multiplication of two matrices """
    return x * y


@matrixify
@jit(nopython=True)
def accel_dot(a, b):  # pragma: no cover
    """ Accelerated dot product of two matrices. """
    return a @ b


@realify
@jit(nopython=True)
def accel_vdot(a, b):  # pragma: no cover
    """ Accelerated 'Hermitian' inner product of two vectors. """
    return np.vdot(a.ravel(), b.ravel())


@matrixify
def ldmul(vec, mat):
    """ Accelerated left diagonal multiplication using numexpr,
    faster than numpy for n > ~ 500.

    Parameters
    ----------
        vec: vector of diagonal matrix, can be array
        mat: matrix

    Returns
    -------
        mat: np.matrix """
    d = mat.shape[0]
    vec = vec.reshape(d, 1)
    return evl("vec*mat") if d > 500 else accel_mul(vec, mat)


@matrixify
def rdmul(mat, vec):
    """ Accelerated right diagonal multiplication using numexpr,
    faster than numpy for n > ~ 500.

    Parameters
    ----------
        mat: matrix
        vec: vector of diagonal matrix, can be array

    Returns
    -------
        mat: np.matrix """
    d = mat.shape[0]
    vec = vec.reshape(1, d)
    return evl("mat*vec") if d > 500 else accel_mul(mat, vec)


def inner(a, b):
    """ Operator inner product between a and b, i.e. for vectors it will be the
    absolute overlap squared |<a|b><b|a>|, rather than <a|b>. """
    method = {(0, 0, 0): lambda: abs(accel_vdot(a, b))**2,
              (0, 0, 1): lambda: abs((a.H @ b)[0, 0])**2,
              (0, 1, 0): lambda: accel_vdot(a, accel_dot(b, a)),
              (0, 1, 1): lambda: abs((a.H @ b @ a)[0, 0]),
              (1, 0, 0): lambda: accel_vdot(b, accel_dot(a, b)),
              (1, 0, 1): lambda: abs((b.H @ a @ b)[0, 0]),
              (1, 1, 0): lambda: trace_dense(accel_dot(a, b)),
              (1, 1, 1): lambda: trace_sparse(a @ b)}
    return method[isop(a), isop(b), issparse(a) or issparse(b)]()
