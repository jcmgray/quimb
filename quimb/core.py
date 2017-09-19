"""Core functions for manipulating quantum objects.
"""
# TODO: move identity, trace to accel
# TODO: refactor eyepad -> itensor

import math
import itertools
import functools

import numpy as np
from numba import jit
from numpy.matlib import zeros
import scipy.sparse as sp

from .accel import (
    njit,
    par_reduce,
    matrixify,
    realify,
    issparse,
    isop,
    vdot,
    dot,
    prod,
    dot_dense,
    kron_dispatch,
    isvec
)
from .utils import deprecated


_SPARSE_CONSTRUCTORS = {"csr": sp.csr_matrix,
                        "bsr": sp.bsr_matrix,
                        "csc": sp.csc_matrix,
                        "coo": sp.coo_matrix}


def sparse_matrix(data, stype="csr"):
    """Construct a sparse matrix of a particular format.

    Parameters
    ----------
    data : array_like
        Fed to scipy.sparse constructor.
    stype : {'csr', 'csc', 'coo', 'bsr'}, optional
        Sparse format.

    Returns
    -------
    scipy sparse matrix
        Of format ``stype``.
    """
    return _SPARSE_CONSTRUCTORS[stype](data, dtype=complex)


def normalize(qob, inplace=True):
    """Normalize a quantum object.

    Parameters
    ----------
    qob : dense or sparse, matrix or vector
        Quantum object to normalize.
    inplace : bool, optional
        Whether to act inplace on the given operator.

    Returns
    -------
    dense or sparse, matrix or vector
        Normalized quantum object.
    """
    n_factor = qob.tr() if isop(qob) else expectation(qob, qob)**0.25
    if inplace:
        qob /= n_factor
        return qob
    return qob / n_factor


def chop(qob, tol=1.0e-15, inplace=True):
    """Set small values of a dense or sparse array to zero.

    Parameters
    ----------
    qob : dense or sparse, matrix or vector
        Quantum object to chop.
    tol : float, optional
        Fraction of ``max(abs(qob))`` to chop below.
    inplace : bool, optional
        Whether to act on input array or return copy.

    Returns
    -------
    dense or sparse, matrix or vector
        Chopped quantum object.
    """
    minm = np.abs(qob).max() * tol  # minimum value tolerated
    if not inplace:
        qob = qob.copy()
    if issparse(qob):
        qob.data.real[np.abs(qob.data.real) < minm] = 0.0
        qob.data.imag[np.abs(qob.data.imag) < minm] = 0.0
        qob.eliminate_zeros()
    else:
        qob.real[np.abs(qob.real) < minm] = 0.0
        qob.imag[np.abs(qob.imag) < minm] = 0.0
    return qob


def quimbify(data, qtype=None, normalized=False, chopped=False,
             sparse=None, stype=None):
    """Converts data to 'quantum' i.e. complex matrices, kets being columns.

    Parameters
    ----------
    data : dense or sparse array_like
        Array describing vector or matrix.
    qtype : {``'ket'``, ``'bra'`` or ``'dop'``}, optional
        Quantum object type output type. Note that if a matrix is given
        as ``data`` and ``'ket'`` or ``'bra'`` as ``qtype``, the matrix
        will be unravelled into a column or row vector.
    sparse : bool, optional
        Whether to convert output to sparse a format.
    normalized : bool, optional
        Whether to normalise the output.
    chopped : bool, optional
        Whether to trim almost zero entries of the output.
    stype : {``'csr'``, ``'csc'``, ``'bsr'``, ``'coo'``}, optional
        Format of output matrix if sparse, defaults to ``'csr'``.

    Returns
    -------
    dense or sparse matrix or vector

    Notes
    -----
    1. Will unravel an array if ``'ket'`` or ``'bra'`` given.
    2. Will conjugate if ``'bra'`` given.
    3. Will leave operators as is if ``'dop'`` given, but construct one if
       vector given with the assumption that it was a ket.

    """

    sparse_input = issparse(data)
    sparse_output = ((sparse) or
                     (sparse_input and sparse is None) or
                     (sparse is None and stype))
    # Infer output sparse format from input if necessary
    if sparse_input and sparse_output and stype is None:
        stype = data.format

    if (qtype is None) and (np.ndim(data) == 1):
        # assume quimbify simple list -> ket
        qtype = 'ket'

    if qtype is not None:
        # Must be dense to reshape
        data = np.asmatrix(data.A if sparse_input else data, dtype=complex)
        if qtype in {"k", "ket"}:
            data = data.reshape((prod(data.shape), 1))
        elif qtype in {"b", "bra"}:
            data = data.reshape((1, prod(data.shape))).conj()
        elif qtype in {"d", "r", "rho", "op", "dop"} and isvec(data):
            data = dot(quimbify(data, "k"), quimbify(data, "k").H)
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
"""Alias of quimbify."""

ket = functools.partial(quimbify, qtype='ket')
"""Convert an object into a ket."""

bra = functools.partial(quimbify, qtype='bra')
"""Convert an object into a bra."""

dop = functools.partial(quimbify, qtype='dop')
"""Convert an object into a density operator."""

sparse = functools.partial(quimbify, sparse=True)
"""Convert an object into sparse form."""


def infer_size(p, base=2):
    """Infer the size, i.e. number of 'sites' in a state.

    Parameters
    ----------
    p : vector or matrix
        An array representing a state with a shape attribute.
    base : int, optional
        Size of the individual states that ``p`` is composed of, e.g. this
        defauts 2 for qubits.

    Returns
    -------
    int
        Number of composite systems.

    Examples
    --------
    >>> infer_size(singlet() & singlet())
    4

    >>> infersize(rand_rho(5**3), base=5)
    3
    """
    sz = math.log(max(p.shape), base)

    if sz % 1 > 1e-13:
        raise ValueError("This state does not seem to be composed of sites"
                         "of equal size {}.".format(base))

    return int(sz)


@realify
@njit
def _trace_dense(op):  # pragma: no cover
    """Trace of a dense matrix.
    """
    x = 0.0j
    for i in range(op.shape[0]):
        x += op[i, i]
    return x


@realify
def _trace_sparse(op):
    """Trace of a sparse matrix.
    """
    return np.sum(op.diagonal())


def trace(mat):
    """Trace of a dense or sparse matrix.

    Parameters
    ----------
    mat : matrix-like
        Complex matrix, dense or sparse.

    Returns
    -------
    x : float
        Trace of ``mat``
    """
    return _trace_sparse(mat) if issparse(mat) else _trace_dense(mat)


@matrixify
@njit
def _identity_dense(d):  # pragma: no cover
    """Returns a dense, complex identity of given dimension.
    """
    x = np.zeros((d, d), dtype=np.complex128)
    for i in range(d):
        x[i, i] = 1
    return x


def _identity_sparse(d, stype="csr"):
    """Returns a sparse, complex identity of order d.
    """
    return sp.eye(d, dtype=complex, format=stype)


def identity(d, sparse=False, stype="csr"):
    """Return identity of size d in complex format, optionally sparse.

    Parameters
    ----------
    d : int
        Dimension of identity.
    sparse : bool, optional
        Whether to output in sparse form.
    stype : str, optional
        If sparse, what format to use.

    Returns
    -------
    id : matrix
        Identity with complex type.
    """
    return _identity_sparse(d, stype=stype) if sparse else _identity_dense(d)


eye = identity
speye = functools.partial(identity, sparse=True)


def kron(*ops, stype=None, coo_build=False, parallel=False):
    """Tensor (kronecker) product of variable number of arguments.

    Parameters
    ----------
    ops : sequence of vectors or matrices
        Objects to be tensored together.
    stype : str, optional
        Desired output format if resultant object is sparse. Should be one
        of {``'csr'``, ``'bsr'``, ``'coo'``, ``'csc'``}. If ``None``, infer
        from input matrices.
    coo_build : bool, optional
        Whether to force sparse construction to use the ``'coo'``
        format (only for sparse matrices in the first place.).

    Returns
    -------
    dense or sparse vector or matrix
        Tensor product of ``*ops``.

    Notes
    -----
     1. The product is performed as ``(a & (b & (c & ...)))``
    """
    opts = {"stype": "coo" if coo_build or stype == "coo" else None}

    if parallel:
        reducer = par_reduce
    else:
        reducer = functools.reduce

    x = reducer(
        functools.partial(kron_dispatch, **opts),
        ops,
    )

    if stype is not None:
        return x.asformat(stype)
    if coo_build or (issparse(x) and x.format == "coo"):
        return x.asformat("csr")
    return x


def kronpow(a, p, stype=None, coo_build=False):
    """Returns `a` tensored with itself `p` times

    Equivalent to ``reduce(lambda x, y: x & y, [a] * p)``.

    Parameters
    ----------
    a : dense or sparse matrix or vector
        Object to tensor power.
    p : int
        Tensor power.
    stype : str, optional
        Desired output format if resultant object is sparse. Should be one
        of {``'csr'``, ``'bsr'``, ``'coo'``, ``'csc'``}.
    coo_build : bool, optional
        Whether to force sparse construction to use the ``'coo'``
        format (only for sparse matrices in the first place.).

    Returns
    -------
    dense or sparse matrix or vector
    """
    return kron(*(a for _ in range(p)), stype=stype, coo_build=coo_build)


def _find_shape_of_nested_int_array(x):
    """Take a n-nested list/tuple of integers and find its array shape.
    """
    shape = [len(x)]
    sub_x = x[0]
    while not isinstance(sub_x, int):
        shape.append(len(sub_x))
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
            yield szb * x + y
        else:
            raise ValueError("One or more coordinates out of range.")


def _dim_map_nd(szs, coos, cyclic=False, trim=False):
    strides = [1]
    for sz in szs[-1:0:-1]:
        strides.insert(0, sz * strides[0])
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
    """Flatten 2d+ dimensions and coordinates.

    Maps multi-dimensional coordinates and indices to flat arrays in a
    regular way. Wraps or deletes coordinates beyond the system size
    depending on parameters ``cyclic`` and ``trim``.

    Parameters
    ----------
    dims : nested tuple of int
        Multi-dim array of systems' internal dimensions.
    coos : list of tuples of int
        Array of coordinate tuples to convert
    cyclic : bool, optional
        Whether to automatically wrap coordinates beyond system size or
        delete them.
    trim : bool, optional
        If True, any coordinates beyond dimensions will be deleted,
        overidden by cyclic.

    Returns
    -------
    flat_dims : tuple
        Flattened version of ``dims``.
    inds : tuple
        Indices corresponding to the original coordinates.

    Examples
    --------

    >>> dims = [[2, 3], [4, 5]]
    >>> coords = [(0, 0), (1, 1)]
    >>> flat_dims, inds = dim_map(dims, coords)
    >>> flat_dims
    (2, 3, 4, 5)
    >>> inds
    (0, 3)

    >>> dim_map(dims, [(2, 0), (-1, 1)], cyclic=True)
    ((2, 3, 4, 5), (0, 3))
    """
    # Figure out shape of dimensions given
    if isinstance(dims, np.ndarray):
        szs = dims.shape
        ndim = dims.ndim
    else:
        szs = _find_shape_of_nested_int_array(dims)
        ndim = len(szs)

    # Ensure `coos` in right format for 1d (i.e. not single tuples)
    if ndim == 1:
        if isinstance(coos, np.ndarray):
            coos = coos.ravel()
        elif not isinstance(coos[0], int):
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
    """Helper function for ``dim_compress`` that does the heavy lifting.
    """
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
    """Compress neighbouring subsytem dimensions.

    Take some dimensions and target indices and compress both, i.e.
    merge adjacent dimensions that are both either in ``dims`` or not. For
    example, if tensoring an operator onto a single site, with many sites
    the identity, treat these as single large identities.

    Parameters
    ----------
    dims : tuple of int
        List of system's dimensions - 1d or flattened (e.g. with
        ``dim_map``).
    inds: tuple of int
        List of target indices, i.e. dimensions not to merge.

    Returns
    -------
    dims : tuple of int
        New compressed dimensions.
    inds : tuple of int
        New indexes corresponding to the compressed dimensions. These are
        guaranteed to now be alternating i.e. either (0, 2, ...) or
        (1, 3, ...).

    Examples
    --------
    >>> dims = [2] * 10
    >>> inds = [3, 4]
    >>> compressed_dims, compressed_inds = dim_compress(dims, inds)
    >>> compressed_dims
    (8, 4, 32)
    >>> compressed_inds
    (1,)
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


def eyepad(ops, dims, inds, sparse=None, stype=None, coo_build=False,
           parallel=False):
    # TODO: rename? itensor, tensor, ikron,
    # TODO: test 2d+ dims and coos
    # TODO: simplify  with compress coords?
    # TODO: allow -1 in dims to auto place *without* ind? one or other
    """Tensor an operator into a larger space by padding with identities.

    Automatically placing a large operator over several dimensions is allowed
    and a list of operators can be given which are then placed cyclically.

    Parameters
    ----------
    op : matrix-like or tuple of matrix-like
        Operator(s) to place into the tensor space. If more than one, these
        are cyclically placed at each of the ``dims`` specified by
        ``inds``.
    dims : tuple of int
        Dimensions of tensor space, use -1 to ignore dimension matching.
    inds : tuple of int
        Indices of the dimensions to place operator on. Each dimension
        specified can be smaller than the size of ``op`` (as long as it
        factorizes it), and can be disjoint from other dimensions when
        ``op`` will be placed.
    sparse : bool, optional
        Whether to construct the new operator in sparse form.
    stype : str, optional
        If sparse, which format to use for the output.
    coo_build : bool, optional
        Whether to build the intermediary matrices using the ``'coo'``
        format - can be faster to build sparse in this way, then
        convert to chosen format, including dense.
    parallel : bool, optional
        Whether to build the operator in parallel using threads (only good
        for big (d > 2**16) operators).

    Returns
    -------
    matrix-like
        Operator such that ops act on ``dims[inds]``.

    See Also
    --------
    perm_eyepad

    Examples
    --------
    Place an operator between two identities:

    >>> IZI = eyepad(sig('z'), [2, 2, 2], 1)
    >>> np.allclose(IZI, eye(2) & sig('z') & eye(2))
    True

    Overlay a large operator on several sites:

    >>> rho_ab = rand_rho(4)
    >>> rho_abc = eyepad(rho_ab, [5, 2, 2, 7], [1, 2])  # overlay both 2s
    >>> rho_abc.shape
    (140, 140)

    Place an operator at specified sites, regardless of size:

    >>> A = rand_herm(5)
    >>> eyepad(A, [2, -1, 2, -1, 2, -1], [1, 3, 5]).shape
    (1000, 1000)
    """

    # Make sure `ops` islist
    if isinstance(ops, (np.ndarray, sp.spmatrix)):
        ops = (ops,)

    # Make sure dimensions and coordinates have been flattenened.
    if np.ndim(dims) > 1:
        dims, inds = dim_map(dims, inds)
    # Make sure `inds` is list
    elif np.ndim(inds) == 0:
        inds = (inds,)

    # Infer sparsity from list of ops
    if sparse is None:
        sparse = any(issparse(op) for op in ops)

    # Create a sorted list of operators with their matching index
    inds, ops = zip(*sorted(zip(inds, itertools.cycle(ops))))
    inds, ops = set(inds), iter(ops)

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

    return kron(*gen_ops(),
                stype=stype,
                coo_build=coo_build,
                parallel=parallel)


@matrixify
def _permute_dense(p, dims, perm):
    """Permute the subsytems of a dense matrix.
    """
    p, perm = np.asarray(p), np.asarray(perm)
    d = prod(dims)
    if isop(p):
        return p.reshape((*dims, *dims)) \
                .transpose((*perm, *(perm + len(dims)))) \
                .reshape((d, d))
    return p.reshape(dims) \
            .transpose(perm) \
            .reshape((d, 1))


def _permute_sparse(a, dims, perm):
    """Permute the subsytems of a sparse matrix.
    """
    perm, dims = np.asarray(perm), np.asarray(dims)

    # New dimensions & stride (i.e. product of preceding dimensions)
    new_dims = dims[perm]
    odim_stride = np.multiply.accumulate(dims[::-1])[::-1] // dims
    ndim_stride = np.multiply.accumulate(new_dims[::-1])[::-1] // new_dims

    # Range of possible coordinates for each subsys
    coos = (tuple(range(dim)) for dim in dims)

    # Complete basis using coordinates for current and new dimensions
    basis = np.asarray(tuple(itertools.product(*coos, repeat=1)))
    oinds = np.sum(odim_stride * basis, axis=1)
    ninds = np.sum(ndim_stride * basis[:, perm], axis=1)

    # Construct permutation matrix and apply it to state
    perm_mat = sp.coo_matrix((np.ones(a.shape[0]), (ninds, oinds))).tocsr()
    if isop(a):
        return dot(dot(perm_mat, a), perm_mat.H)
    return dot(perm_mat, a)


def permute(p, dims, perm):
    """Permute the subsytems of state or opeator.

    Parameters
    ----------
    p : vector or matrix
        State or operator to permute.
    dims : tuple of int
        Internal dimensions of the system.
    perm : tuple of int
        New order of indexes ``range(len(dims))``.

    Returns
    -------
    pp : vector or matrix
        Permuted state or operator.

    See Also
    --------
    perm_eyepad

    Examples
    --------

    >>> IX = speye(2) & sig('X', sparse=True)
    >>> XI = permute(IX, dims=[2, 2], perm=[1, 0])
    >>> np.allclose(XI.A, sig('X') & eye(2))
    True
    """
    if issparse(p):
        return _permute_sparse(p, dims, perm)
    return _permute_dense(p, dims, perm)


def perm_eyepad(op, dims, inds, **kwargs):
    # TODO: multiple ops
    # TODO: coo map, coo compress
    # TODO: sparse, stype, coo_build?
    """Advanced, padded tensor product.

    Construct an operator such that ``op`` acts on ``dims[inds]``, and allow it
    to be arbitrarily split and reversed etc., in other words, permute and then
    tensor it into a larger space.

    Parameters
    ----------
    ops : matrix-like or tuple of matrix-like
        Operator to place into the tensor space.
    dims : tuple of int
        Dimensions of tensor space.
    inds : tuple of int
        Indices of the dimensions to place operators on. If multiple
        operators are specified, ``inds[1]`` corresponds to ``ops[1]`` and
        so on.
    sparse : bool, optional
        Whether to construct the new operator in sparse form.
    stype : str, optional
        If sparse, which format to use for the output.
    coo_build : bool, optional
        Whether to build the intermediary matrices using the ``'coo'``
        format - can be faster to build sparse in this way, then
        convert to chosen format, including dense.

    Returns
    -------
    matrix-like
        Operator such that ops act on ``dims[inds]``.

    See Also
    --------
    eyepad, permute

    Examples
    --------

    Here we take an operator that acts on spins 0 and 1 with X and Z, and
    transform it to act on spins 2 and 0 -- i.e. reverse it and sandwich an
    identity between the two sites it acts on.

    >>> XZ = sig('X') & sig('Z')
    >>> ZIX = perm_eyepad(XZ, dims=[2, 3, 2], inds=[2, 0])
    >>> np.allclose(ZIX, sig('Z') & eye(3) & sig('X'))
    True
    """
    dims, inds = np.asarray(dims), np.asarray(inds)

    # total number of subsytems and size
    n = len(dims)
    sz = prod(dims)

    # dimensions of space where op should be placed, and its total size
    dims_in = dims[inds]
    sz_in = prod(dims_in)

    # construct pre-permuted full operator
    b = eyepad(op, [sz_in, sz // sz_in], 0, **kwargs)

    # inverse of inds
    inds_out, dims_out = zip(
        *((i, x) for i, x in enumerate(dims) if i not in inds))

    # current order and dimensions of system
    p = [*inds, *inds_out]
    dims_cur = (*dims_in, *dims_out)

    # find inverse permutation
    ip = np.empty(n, dtype=np.int)
    ip[p] = np.arange(n)

    return permute(b, dims_cur, ip)


def ind_complement(inds, n):
    """Return the indices below ``n`` not contained in ``inds``.
    """
    return tuple(i for i in range(n) if i not in inds)


def itrace(a, axes=(0, 1)):
    """General tensor trace, i.e. multiple contractions, for a dense array.

    Parameters
    ----------
    a : numpy.ndarray
        Tensor to trace.
    axes : (2,) int or (2,) array of int
        - (2,) int: Perform trace on the two indices listed.
        - (2,) array of int: Trace out first sequence of indices with second
          sequence indices.

    Returns
    -------
    numpy.ndarray
        The tensor remaining after tracing out the specified axes.

    See Also
    --------
    trace, partial_trace

    Examples
    --------
    Trace out a single pair of dimensions:

    >>> a = np.random.rand(2, 3, 4, 2, 3, 4)
    >>> itrace(a, axes=(0, 3)).shape
    (3, 4, 3, 4)

    Trace out multiple dimensions:

    >>> itrace(a, axes=([1, 2], [4, 5])).shape
    (2, 2)
    """
    # Single index pair to trace out
    if isinstance(axes[0], int):
        return np.trace(a, axis1=axes[0], axis2=axes[1])
    elif len(axes[0]) == 1:
        return np.trace(a, axis1=axes[0][0], axis2=axes[1][0])

    # Multiple index pairs to trace out
    gone = set()
    for axis1, axis2 in zip(*axes):
        # Modify indices to adjust for traced out dimensions
        mod1 = sum(x < axis1 for x in gone)
        mod2 = sum(x < axis2 for x in gone)
        gone |= {axis1, axis2}
        a = np.trace(a, axis1=axis1 - mod1, axis2=axis2 - mod2)
    return a


@matrixify
def _partial_trace_dense(p, dims, coo_keep):
    """Perform partial trace of a dense matrix.
    """
    if isinstance(coo_keep, int):
        coo_keep = (coo_keep,)
    if isvec(p):  # p = psi
        p = np.asarray(p).reshape(dims)
        lose = ind_complement(coo_keep, len(dims))
        p = np.tensordot(p, p.conj(), (lose, lose))
        d = int(p.size**0.5)
        return p.reshape((d, d))
    else:
        p = np.asarray(p).reshape((*dims, *dims))
        total_dims = len(dims)
        lose = ind_complement(coo_keep, total_dims)
        lose2 = tuple(ind + total_dims for ind in lose)
        p = itrace(p, (lose, lose2))
    d = int(p.size**0.5)
    return p.reshape((d, d))


def _trace_lose(p, dims, coo_lose):
    """Simple partial trace where the single subsytem at ``coo_lose``
    is traced out.
    """
    p = p if isop(p) else dot(p, p.H)
    dims = np.asarray(dims)
    e = dims[coo_lose]
    a = prod(dims[:coo_lose])
    b = prod(dims[coo_lose + 1:])
    rhos = zeros(shape=(a * b, a * b), dtype=np.complex128)
    for i in range(a * b):
        for j in range(i, a * b):
            i_i = e * b * (i // b) + (i % b)
            i_f = e * b * (i // b) + (i % b) + (e - 1) * b + 1
            j_i = e * b * (j // b) + (j % b)
            j_f = e * b * (j // b) + (j % b) + (e - 1) * b + 1
            rhos[i, j] = trace(p[i_i:i_f:b, j_i:j_f:b])
            if j != i:
                rhos[j, i] = rhos[i, j].conjugate()
    return rhos


def _trace_keep(p, dims, coo_keep):
    """Simple partial trace where the single subsytem
    at ``coo_keep`` is kept.
    """
    p = p if isop(p) else dot(p, p.H)
    dims = np.asarray(dims)
    s = dims[coo_keep]
    a = prod(dims[:coo_keep])
    b = prod(dims[coo_keep + 1:])
    rhos = zeros(shape=(s, s), dtype=np.complex128)
    for i in range(s):
        for j in range(i, s):
            for k in range(a):
                i_i = b * i + s * b * k
                i_f = b * i + s * b * k + b
                j_i = b * j + s * b * k
                j_f = b * j + s * b * k + b
                rhos[i, j] += trace(p[i_i:i_f, j_i:j_f])
            if j != i:
                rhos[j, i] = rhos[i, j].conjugate()
    return rhos


def _partial_trace_simple(p, dims, coo_keep):
    """Simple partial trace made up of consecutive single subsystem partial
    traces, augmented by 'compressing' the dimensions each time.
    """
    p = p if isop(p) else dot(p, p.H)
    dims, coo_keep = dim_compress(dims, coo_keep)
    if len(coo_keep) == 1:
        return _trace_keep(p, dims, *coo_keep)
    lmax = max(enumerate(dims),
               key=lambda ix: (ix[0] not in coo_keep) * ix[1])[0]
    p = _trace_lose(p, dims, lmax)
    dims = (*dims[:lmax], *dims[lmax + 1:])
    coo_keep = {(ind if ind < lmax else ind - 1) for ind in coo_keep}
    return _partial_trace_simple(p, dims, coo_keep)


def partial_trace(p, dims, keep):
    """Partial trace of a dense or sparse state.

    Parameters
    ----------
    p : ket or density matrix
        State to perform partial trace on - can be sparse.
    dims : tuple of int
        List of subsystem dimensions.
    keep : int or tuple of int
        Index or indices of subsytem(s) to keep.

    Returns
    -------
    rho : dense matrix
        Density matrix of subsytem dimensions ``dims[keep]``.

    See Also
    --------
    itrace

    Examples
    --------
    Trace out single subsystem of a ket:

    >>> psi = bell_state('psi-')
    >>> ptr(psi, [2, 2], keep=0)  # expect identity
    matrix([[ 0.5+0.j,  0.0+0.j],
            [ 0.0+0.j,  0.5+0.j]])

    Trace out multiple subsystems of a density matrix:

    >>> rho_abc = rand_rho(3 *4 * 5)
    >>> rho_ab = partial_trace(rho_abc, [3, 4, 5], keep=[0, 1])
    >>> rho_ab.shape
    (12, 12)
    """
    if issparse(p):
        return _partial_trace_simple(p, dims, keep)
    return _partial_trace_dense(p, dims, keep)


_OVERLAP_METHODS = {
    (0, 0, 0): lambda a, b: abs(vdot(a, b))**2,
    (0, 1, 0): lambda a, b: vdot(a, dot_dense(b, a)),
    (1, 0, 0): lambda a, b: vdot(b, dot_dense(a, b)),
    (1, 1, 0): lambda a, b: _trace_dense(dot_dense(a, b)),
    (0, 0, 1): lambda a, b: abs(dot(a.H, b)[0, 0])**2,
    (0, 1, 1): realify(lambda a, b: dot(a.H, dot(b, a))[0, 0]),
    (1, 0, 1): realify(lambda a, b: dot(b.H, dot(a, b))[0, 0]),
    (1, 1, 1): lambda a, b: _trace_sparse(dot(a, b)),
}


def expectation(a, b):
    """'Overlap' between a vector/operator and another vector/operator.

    The 'operator' inner product between ``a`` and ``b``, but also for vectors.
    This means that for consistency:
    * for two vectors it will be the absolute expec squared ``|<a|b><b|a>|``,
    *not* ``<a|b>``
    * for a vector and an operator its will be ``<a|b|a>``
    * for two operators it will be the Hilbert-schmidt inner product
    ``tr(A @ B)``

    In this way ``expectation(a, b) == expectation(dop(a), b) ==
    expectation(dop(a), dop(b))``.

    Parameters
    ----------
    a : vector or operator
        First state or operator - assumed to be ket if vector.
    b : vector or operator
        Second state or operator - assumed to be ket if vector.

    Returns
    -------
    x : float
        'Overlap' between ``a`` and ``b``.
    """
    return _OVERLAP_METHODS[isop(a), isop(b), issparse(a) or issparse(b)](a, b)


expec = expectation
overlap = deprecated(expectation, "'overlap'", "'expectation' or 'expec'")


# --------------------------------------------------------------------------- #
# MONKEY-PATCHES                                                              #
# --------------------------------------------------------------------------- #

# Normalise methods
nmlz = normalize
np.matrix.nmlz = nmlz
sp.csr_matrix.nmlz = nmlz

# Trace methods
tr = trace
np.matrix.tr = _trace_dense
sp.csr_matrix.tr = _trace_sparse
sp.csc_matrix.tr = _trace_sparse
sp.coo_matrix.tr = _trace_sparse
sp.bsr_matrix.tr = _trace_sparse

# Partial trace methods
ptr = partial_trace
np.matrix.ptr = _partial_trace_dense
sp.csr_matrix.ptr = _partial_trace_simple
sp.csc_matrix.ptr = _partial_trace_simple
sp.coo_matrix.ptr = _partial_trace_simple
sp.bsr_matrix.ptr = _partial_trace_simple
