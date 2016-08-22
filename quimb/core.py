"""
Core functions for manipulating quantum objects.
"""
# TODO: move identity, trace to accel

import math
import itertools
import functools

import numpy as np
from numpy.matlib import zeros
import scipy.sparse as sp
from numba import jit, complex128, int64

from .accel import (matrixify, realify, issparse, isop, vdot, dot_dense,
                    kron, kronpow)

_sparse_constructors = {"csr": sp.csr_matrix,
                        "bsr": sp.bsr_matrix,
                        "csc": sp.csc_matrix,
                        "coo": sp.coo_matrix}


def sparse_matrix(data, stype="csr"):
    """ Construct a sparse matrix of a particular format. """
    return _sparse_constructors[stype](data, dtype=complex)


def normalize(qob, inplace=True):
    """ Returns the state qob in normalized form """
    n_factor = qob.tr() if isop(qob) else overlap(qob, qob)**0.25
    if inplace:
        qob /= n_factor
        return qob
    return qob / n_factor

# Monkey-patch normalise methods
nmlz = normalize
np.matrix.nmlz = nmlz
sp.csr_matrix.nmlz = nmlz


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


def quimbify(data, qtype=None, sparse=None, normalized=False,
             chopped=False, stype=None):
    """
    Converts lists to 'quantum' i.e. complex matrices, kets being columns.

    Parameters
    ----------
        data:  list describing entries
        qtype: output type, either 'ket', 'bra' or 'dop' if given
        sparse: convert output to sparse 'csr' format
        normalized: normalise the output
        stype: format of sparse matrix

    Returns
    -------
        x: numpy scipy.sparse matrix.

    Notes
    -----
        1. Will unravel an array if 'ket' or 'bra' given.
        2. Will conjugate if 'bra' given.
        3. Will leave operators as is if 'dop' given, but construct one
            if vector given with the assumption that it was a ket.
    """

    sparse_input = issparse(data)
    sparse_output = (sparse or (sparse_input and sparse is None) or
                     (sparse is None and stype in {"csr", "bsr",
                                                   "csc", "coo"}))
    # Infer output sparse format from input if necessary
    if sparse_input and sparse_output and stype is None:
        stype = data.format

    if qtype is not None:
        # Must be dense to reshape
        data = np.asmatrix(data.A if sparse_input else data, dtype=complex)
        if qtype in {"k", "ket"}:
            data = data.reshape((np.prod(data.shape), 1))
        elif qtype in {"b", "bra"}:
            data = data.reshape((1, np.prod(data.shape))).conj()
        elif qtype in {"d", "r", "rho", "op", "dop"} and not isop(data):
            data = quimbify(data, "k") @ quimbify(data, "k").H
    # Just cast as numpy matrix
    elif not sparse_output:
        data = np.asmatrix(data.A if sparse_input else data, dtype=complex)

    # Check if already sparse matrix, or wanted to be one
    if sparse_output:
        data = sparse_matrix(data, (stype if stype is not None else "csr"))

    # Optionally normalize and chop small components
    if normalized:
        normalize(data, inplace=True)
    if chopped:
        chop(data, inplace=True)

    return data

qu = quimbify
ket = functools.partial(quimbify, qtype='ket')
bra = functools.partial(quimbify, qtype='bra')
dop = functools.partial(quimbify, qtype='dop')
sparse = functools.partial(quimbify, sparse=True)


def infer_size(p, base=2):
    """ Infers the size of a state assumed to be made of qubits """
    return int(math.log(max(p.shape), base))


@realify
@jit(complex128(complex128[:, :]), nopython=True, cache=True)
def _trace_dense(op):  # pragma: no cover
    """ Trace of matrix. """
    x = 0.0
    for i in range(op.shape[0]):
        x += op[i, i]
    return x


@realify
def _trace_sparse(op):
    """ Trace of sparse matrix. """
    return np.sum(op.diagonal())


def trace(op):
    """ Trace of dense or sparse matrix """
    return _trace_sparse(op) if issparse(op) else _trace_dense(op)

# Monkey-patch trace methods
tr = trace
np.matrix.tr = _trace_dense
sp.csr_matrix.tr = _trace_sparse


@matrixify
@jit(complex128[:, :](int64), nopython=True, cache=True)
def _identity_dense(d):  # pragma: no cover
    """ Returns a dense, complex identity of order d. """
    x = np.zeros((d, d), dtype=np.complex128)
    for i in range(d):
        x[i, i] = 1
    return x


def _identity_sparse(d, stype="csr"):
    """ Returns a sparse, complex identity of order d. """
    return sp.eye(d, dtype=complex, format=stype)


def identity(d, sparse=False, stype="csr"):
    """ Return identity of size d in complex format, optionally sparse"""
    return _identity_sparse(d, stype=stype) if sparse else _identity_dense(d)

eye = identity
speye = functools.partial(identity, sparse=True)


def _find_shape_of_nested_int_array(x):
    """ Take a n-nested list/tuple of integers and find its array shape """
    shape = [len(x)]
    sub_x = x[0]
    while not isinstance(sub_x, int):
        shape += [len(sub_x)]
        sub_x = sub_x[0]
    return tuple(shape)


def _dim_map_1d(sza, coos):
    for coo in coos:
        if 0 <= coo < sza:
            yield coo
        else:
            raise ValueError("One or more coordinates out of range.")


def _dim_map_1dtrim(sza, coos):
    return (coo for coo in coos if (0 <= coo < sza))


def _dim_map_1dcyclic(sza, coos):
    return (coo % sza for coo in coos)


def _dim_map_2dcyclic(sza, szb, coos):
    return (szb * (coo[0] % sza) + coo[1] % szb for coo in coos)


def _dim_map_2dtrim(sza, szb, coos):
    for coo in coos:
        x, y = coo
        if 0 <= x < sza and 0 <= y < szb:
            yield szb * x + y


def _dim_map_2d(sza, szb, coos):
    for coo in coos:
        x, y = coo
        if 0 <= x < sza and 0 <= y < szb:
            yield szb * coo[0] + coo[1]
        else:
            raise ValueError("One or more coordinates out of range.")


def _dim_map_nd(szs, coos, cyclic=False, trim=False):
    strides = [1]
    for sz in szs[-1:0:-1]:
        strides.insert(0, sz*strides[0])
    if cyclic:
        coos = ((c % sz for c, sz in zip(coo, szs)) for coo in coos)
    elif trim:
        coos = (c for c in coos if all(x == x % sz for x, sz in zip(c, szs)))
    elif not all(all(c == c % sz for c, sz in zip(coo, szs)) for coo in coos):
        raise ValueError("One or more coordinates out of range.")
    return (sum(c * m for c, m in zip(coo, strides)) for coo in coos)


_dim_mapper_methods = {(1, False, False): _dim_map_1d,
                       (1, False, True): _dim_map_1dtrim,
                       (1, True, False): _dim_map_1dcyclic,
                       (2, False, False): _dim_map_2d,
                       (2, False, True): _dim_map_2dtrim,
                       (2, True, False): _dim_map_2dcyclic}


def dim_map(dims, coos, cyclic=False, trim=False):
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
        coos: indices mapped to flattened dims
    """
    # Figure out shape of dimensions given
    if isinstance(dims, np.ndarray):
        szs = dims.shape
        ndim = dims.ndim
    else:
        szs = _find_shape_of_nested_int_array(dims)
        ndim = len(szs)

    # Ensure `coos` in right format for 1d (i.e. not single tuples)
    if ndim == 1 and not isinstance(coos[0], int):
        coos = (c[0] for c in coos)

    # Map coordinates to indices
    try:
        inds = _dim_mapper_methods[(ndim, cyclic, trim)](*szs, coos)
    except KeyError:
        inds = _dim_map_nd(szs, coos, cyclic, trim)

    # Ravel dims
    while ndim > 1:
        dims = itertools.chain.from_iterable(dims)
        ndim -= 1

    return tuple(dims), tuple(inds)


@jit(nopython=True)
def _dim_compressor(dims, inds):  # pragma: no cover
    """ Helper function for `dim_compress` that does the heavy lifting. """
    blocksize_id = blocksize_op = 1
    autoplace_count = 0
    for i, dim in enumerate(dims):
        if dim < 0:
            if blocksize_op > 1:
                yield (blocksize_op, 1)
                blocksize_op = 1
            elif blocksize_id > 1:
                yield (blocksize_id, 0)
                blocksize_id = 1
            autoplace_count += dim
        elif i in inds:
            if blocksize_id > 1:
                yield (blocksize_id, 0)
                blocksize_id = 1
            elif autoplace_count < 0:
                yield (autoplace_count, 1)
                autoplace_count = 0
            blocksize_op *= dim
        else:
            if blocksize_op > 1:
                yield (blocksize_op, 1)
                blocksize_op = 1
            elif autoplace_count < 0:
                yield (autoplace_count, 1)
                autoplace_count = 0
            blocksize_id *= dim
    yield ((blocksize_op, 1) if blocksize_op > 1 else
           (blocksize_id, 0) if blocksize_id > 1 else
           (autoplace_count, 1))


def dim_compress(dims, inds):
    """ Take some dimensions and target indices and compress both such, i.e.
    merge adjacent identity spaces.

    Parameters
    ----------
        dims: list of systems dimensions
        inds: list of target indices

    Returns
    -------
        dims, inds: new equivalent dimensions and matching indices
    """
    # TODO: turn off ind compress
    # TODO: put yield (autoplace_count, False) --- no need?
    # TODO: handle empty inds = () / [] etc.
    # TODO: don't compress auto (-ve.) so as to allow multiple operators
    if isinstance(inds, int):
        inds = (inds,)
    dims, inds = zip(*_dim_compressor(dims, inds))
    inds = tuple(i for i, b in enumerate(inds) if b)
    return dims, inds


def eyepad(ops, dims, inds, sparse=None, stype=None, coo_build=False):
    # TODO: rename? itensor, tensor
    # TODO: test 2d+ dims and coos
    # TODO: simplify  with compress coords?
    # TODO: allow -1 in dims to auto place *without* ind?
    """
    Tensor product, but padded with identites. Automatically
    placing a large operator over several dimensions is allowed and a list
    of operators can be given which are then applied cyclically.

    Parameters
    ----------
        ops: operator or list of operators to put into the tensor space.
        dims: dimensions of tensor space, use -1 to ignore dimension matching.
        inds: indices of the dimenions to place operators on.
        sparse: whether to construct the new operator in sparse form.

    Returns
    -------
        Operator such that ops act on dims[inds].

    *Notes:*
        1. if len(inds) > len(ops), then ops will be cycled over.
    """

    # Make sure `ops` islist
    if isinstance(ops, (np.ndarray, sp.spmatrix)):
        ops = [ops]

    # Make sure dimensions and coordinates have been flattenened.
    if np.ndim(dims) > 1:
        dims, inds = dim_map(dims, inds)
    # Make sure `inds` is list
    elif np.ndim(inds) == 0:
        inds = [inds]

    # Infer sparsity from list of ops
    if sparse is None:
        sparse = any(issparse(op) for op in ops)

    # Create a sorted list of operators with their matching index
    inds, ops = zip(*sorted(zip(inds, itertools.cycle(ops))))
    inds, ops = set(inds), iter(ops)

    # TODO: refactor this / just use dim_compress
    def gen_ops():
        cff_id = 1  # keeps track of compressing adjacent identities
        cff_ov = 1  # keeps track of overlaying op on multiple dimensions
        for ind, dim in enumerate(dims):
            if ind in inds:  # op should be placed here
                if cff_id > 1:  # need preceding identities
                    yield eye(cff_id, sparse=sparse, stype="coo")
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
            yield eye(cff_id, sparse=sparse, stype="coo")

    return kron(*gen_ops(), stype=stype, coo_build=coo_build)


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
    n_op = int(math.log(sz_in, sz_op))  # number of individual operators
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
def _permute_dense(p, dims, perm):
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


def _permute_sparse(a, dims, perm):
    """ Permute the subsytems of a sparse matrix. """
    perm, dims = np.asarray(perm), np.asarray(dims)
    new_dims = dims[perm]
    # New dimensions & stride (i.e. product of preceding dimensions)
    odim_stride = np.asarray([np.prod(dims[i+1:])
                              for i, _ in enumerate(dims)])
    ndim_stride = np.asarray([np.prod(new_dims[i+1:])
                              for i, _ in enumerate(new_dims)])
    # Range of possible coordinates for each subsys
    coos = (tuple(range(dim)) for dim in dims)
    # Complete basis using coordinates for current and new dimensions
    basis = np.asarray(tuple(itertools.product(*coos, repeat=1)))
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
        return _permute_sparse(a, dims, perm)
    return _permute_dense(a, dims, perm)


def _partial_trace_clever(p, dims, keep):
    # TODO: compress coords?
    # TODO: user tensordot for vec
    # TODO: matrixify
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
        return dot_dense(p, p.H)
    else:
        p = np.asarray(p).reshape((*dims, *dims)) \
            .transpose((*perm, *(perm + n))) \
            .reshape((sz_keep, sz_lose, sz_keep, sz_lose)) \
            .trace(axis1=1, axis2=3)
        return np.asmatrix(p)


def _trace_lose(p, dims, coo_lose):
    """ Simple partial trace where the single subsytem at `coo_lose`
    is traced out. """
    p = p if isop(p) else p @ p.H
    dims = np.asarray(dims)
    e = dims[coo_lose]
    a = np.prod(dims[:coo_lose], dtype=int)
    b = np.prod(dims[coo_lose+1:], dtype=int)
    rhos = zeros(shape=(a * b, a * b), dtype=np.complex128)
    for i in range(a * b):
        for j in range(i, a * b):
            rhos[i, j] = trace(p[
                    e*b*(i//b) + (i % b):
                    e*b*(i//b) + (i % b) + (e-1)*b + 1: b,
                    e*b*(j//b) + (j % b):
                    e*b*(j//b) + (j % b) + (e-1)*b + 1: b])
            if j != i:
                rhos[j, i] = rhos[i, j].conjugate()
    return rhos


def _trace_keep(p, dims, coo_keep):
    """ Simple partial trace where the single subsytem
    at `coo_keep` is kept. """
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
                        b*i + s*b*k: b*i + s*b*k + b,
                        b*j + s*b*k: b*j + s*b*k + b])
            if j != i:
                rhos[j, i] = rhos[i, j].conjugate()
    return rhos


def _partial_trace_simple(p, dims, coos_keep):
    """ Simple partial trace made up of consecutive single subsystem partial
    traces, augmented by 'compressing' the dimensions each time. """
    p = p if isop(p) else p @ p.H
    dims, coos_keep = dim_compress(dims, coos_keep)
    if len(coos_keep) == 1:
        return _trace_keep(p, dims, *coos_keep)
    lmax = max(enumerate(dims),
               key=lambda ix: (ix[0] not in coos_keep)*ix[1])[0]
    p = _trace_lose(p, dims, lmax)
    dims = [*dims[:lmax], *dims[lmax+1:]]
    coos_keep = {(ind if ind < lmax else ind - 1) for ind in coos_keep}
    return _partial_trace_simple(p, dims, coos_keep)


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
        return _partial_trace_simple(p, dims, coos)
    return _partial_trace_clever(p, dims, coos)

ptr = partial_trace
np.matrix.ptr = _partial_trace_clever
sp.csr_matrix.ptr = _partial_trace_simple


def overlap(a, b):
    """ Overlap between a and b, i.e. for vectors it will be the
    absolute overlap squared |<a|b><b|a>|, rather than <a|b>. """
    method = {(0, 0, 0): lambda: abs(vdot(a, b))**2,
              (0, 0, 1): lambda: abs((a.H @ b)[0, 0])**2,
              (0, 1, 0): lambda: vdot(a, dot_dense(b, a)),
              (1, 0, 0): lambda: vdot(b, dot_dense(a, b)),
              (0, 1, 1): realify(lambda: (a.H @ b @ a)[0, 0]),
              (1, 0, 1): realify(lambda: (b.H @ a @ b)[0, 0]),
              (1, 1, 0): lambda: _trace_dense(dot_dense(a, b)),
              (1, 1, 1): lambda: _trace_sparse(a @ b)}
    return method[isop(a), isop(b), issparse(a) or issparse(b)]()
