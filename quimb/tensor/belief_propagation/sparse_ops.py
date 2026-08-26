"""Specialized kernels for belief propagation with sparse COO tensors and
dense vector messages. Both ``sparse.COO`` and n-dimensional
``scipy.sparse.coo_array`` are supported.
"""

import numpy as np

from quimb.core import njit


def parse_coo(x):
    """Get the ``(coords, data)`` pair of a COO array, with ``coords`` as a
    tuple of one flat coordinate array per dimension. Returns ``None`` if
    ``x`` is not a COO array with zero fill value. Duplicate coordinates are
    allowed, since every kernel here accumulates additively.
    """
    coords = getattr(x, "coords", None)
    if (coords is None) or (getattr(x, "fill_value", 0) != 0):
        return None
    if not isinstance(coords, tuple):
        # e.g. pydata/sparse stores a single (ndim, nnz) array already
        coords = tuple(coords)
    # numba requires all to have same layout
    coords = tuple(np.ascontiguousarray(c) for c in coords)
    return coords, x.data


def to_dense(x):
    """Convert a possibly sparse array to a dense numpy array."""
    todense = getattr(x, "todense", None)
    if todense is not None:
        x = todense()
    return np.asarray(x)


@njit(nogil=True)  # pragma: no cover
def _compute_all_tensor_messages_coo(coords, data, ms, out):
    """Numba kernel that computes all outgoing messages for COO tensor."""
    ndim = len(coords)
    mvals = np.empty(ndim, dtype=data.dtype)
    after = np.empty(ndim + 1, dtype=data.dtype)

    for n in range(len(data)):
        # for this n'th nnz entry, get the corresponding message values
        for k in range(ndim):
            mvals[k] = ms[k][coords[k][n]]

        # compute prod_{k' > k} m_k'
        after[ndim] = 1.0
        for k in range(ndim - 1, -1, -1):
            after[k] = after[k + 1] * mvals[k]

        # finally compute array entry multiplied by all but one message values
        before = data[n]
        for k in range(ndim):
            out[k][coords[k][n]] += before * after[k + 1]
            before = before * mvals[k]


@njit(nogil=True)  # pragma: no cover
def _contract_all_messages_coo(coords, data, ms, out):
    """Numba kernel that fully contracts a COO tensor with all incoming
    messages."""
    ndim = len(coords)

    for n in range(len(data)):
        val = data[n]
        for k in range(ndim):
            val = val * ms[k][coords[k][n]]
        out[0] += val


def _prepare_coo_ms_for_numba(coo, ms):
    """Make sure COO format tensor and messages are ready for numba kernels."""
    coords, data = coo

    dtype = data.dtype
    for m in ms:
        if m.dtype != dtype:
            dtype = np.promote_types(dtype, m.dtype)

    # every message must have the same type for the kernels to index them
    ms = tuple(np.ascontiguousarray(m, dtype=dtype) for m in ms)

    return coords, np.asarray(data, dtype=dtype), ms, dtype


def compute_all_tensor_messages_coo(coo, ms):
    """Given messages ``ms`` incident to a sparse tensor, compute all the
    outgoing messages, each the contraction of the tensor with every incident
    message but one.

    Parameters
    ----------
    coo : (tuple[array], array)
        The ``(coords, data)`` pair, as returned by ``parse_coo``.
    ms : sequence of array
        The dense vector messages, one per dimension.

    Returns
    -------
    list[array]
    """
    if not ms:
        return []

    coords, data, ms, dtype = _prepare_coo_ms_for_numba(coo, ms)
    out = tuple(np.zeros(m.shape[0], dtype=dtype) for m in ms)
    _compute_all_tensor_messages_coo(coords, data, ms, out)
    return list(out)


def contract_tensor_messages_coo(coo, ms):
    """Contract a sparse tensor with a dense vector message on every
    dimension, to a scalar.

    Parameters
    ----------
    coo : (tuple[array], array)
        The ``(coords, data)`` pair, as returned by ``parse_coo``.
    ms : sequence of array
        The dense vector messages, one per dimension.

    Returns
    -------
    scalar
    """
    if not ms:
        return coo[1].sum()

    coords, data, ms, dtype = _prepare_coo_ms_for_numba(coo, ms)
    out = np.zeros(1, dtype=dtype)
    _contract_all_messages_coo(coords, data, ms, out)
    return out[0]


def sum_all_but_axis_coo(x, axis):
    """Sum the sparse tensor ``x`` over every dimension but ``axis``, returning
    a dense vector. This is the BP 'uniform message' initialization step.
    """
    other = tuple(k for k in range(x.ndim) if k != axis)
    if not other:
        # already 1D
        return to_dense(x)
    return to_dense(x.sum(axis=other))
