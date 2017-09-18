import functools
import numpy as np
try:
    from opt_einsum import contract as einsum
except ImportError:
    from numpy import einsum


@functools.lru_cache(128)
def get_cntrct_inds_ptr_dot(ndim_ab, keep):
    """Find the correct integer contraction labels for ``lazy_ptr_dot``.

    Parameters
    ----------
    ndim_ab : int
        The total number of subsystems (dimensions) in 'ab'.
    keep : sequence of int
        Indexes of the subsytems not to be traced out.

    Returns
    -------
    inds_a_ket : sequence of int
        The tensor index labels for the ket on subsystem 'a'.
    inds_ab_bra : sequence of int
        The tensor index labels for the bra on subsystems 'ab'.
    inds_ab_ket : sequence of int
        The tensor index labels for the ket on subsystems 'ab'.
    """
    inds_a_ket = []
    inds_ab_bra = []
    inds_ab_ket = []

    upper_inds = iter(range(ndim_ab, 2 * ndim_ab))

    for i in range(ndim_ab):
        if i in keep:
            inds_a_ket.append(i)
            inds_ab_bra.append(i)
            inds_ab_ket.append(next(upper_inds))
        else:
            inds_ab_bra.append(i)
            inds_ab_ket.append(i)

    return inds_a_ket, inds_ab_bra, inds_ab_ket


def lazy_ptr_dot(psi_ab, psi_a, dims=None, keep=0):
    """Perform the 'lazy' evalution of ``ptr(psi_ab, ...) @ psi_a``,
    that is, contract the tensor diagram in an efficient way that does not
    necessarily construct the explicit reduced density matrix. In tensor
    diagram notation:
    ``
    +-------+
    | psi_a |   ______
    +_______+  /      \
       a|      |b     |
    +-------------+   |
    |  psi_ab.H   |   |
    +_____________+   |
                      |
    +-------------+   |
    |   psi_ab    |   |
    +_____________+   |
       a|      |b     |
        |      \______/
    ``

    Parameters
    ----------
    psi_ab : ket
        State to partially trace and dot with another ket, with
        size ``prod(dims)``.
    psi_a : ket
        State to act on with the dot product, of size ``prod(dims[keep])``.
    dims : sequence of int, optional
        The sub dimensions of ``psi_ab``, inferred as bipartite if not given,
        i.e. ``(psi_a.size, psi_ab.size // psi_a.size)``.
    keep : int, optional
        Index of the subsystem to keep in the partial trace, assumed to be 0 if
        not given.

    Returns
    -------
    ket
    """
    if dims is None:
        da = psi_a.size
        d = psi_ab.size
        dims = (da, d // da)

    # convert to tuple so can always cache
    keep = (keep,) if isinstance(keep, int) else tuple(keep)

    ndim_ab = len(dims)
    dims_a = [d for i, d in enumerate(dims) if i in keep]

    inds_a_ket, inds_ab_bra, inds_ab_ket = get_cntrct_inds_ptr_dot(
        ndim_ab, keep)

    psi_ab_tensor = np.asarray(psi_ab).reshape(dims)

    return einsum(
        np.asarray(psi_a).reshape(dims_a), inds_a_ket,
        psi_ab_tensor.conjugate(), inds_ab_bra,
        psi_ab_tensor, inds_ab_ket,
    ).reshape(psi_a.shape)


@functools.lru_cache(128)
def get_cntrct_inds_ptr_ppt_dot(ndim_abc, keep_a, keep_b):
    """Find the correct integer contraction labels for ``lazy_ptr_ppt_dot``.

    Parameters
    ----------
    ndim_abb : int
        The total number of subsystems (dimensions) in 'abc'.
    keep_a : sequence of int
        Indexes of the 'a' subsystems to keep.
    keep_b : sequence of int
        Indexes of the 'b' subsystems to keep.

    Returns
    -------
    inds_ab_ket : sequence of int
        The tensor index labels for the ket on subsystems 'ab'.
    inds_abc_bra : sequence of int
        The tensor index labels for the bra on subsystems 'abc'.
    inds_abc_ket : sequence of int
        The tensor index labels for the ket on subsystems 'abc'.
    inds_out : sequence of int
        The tensor indices of the resulting ket, important as these might
        no longer be ordered.
    """
    inds_ab_ket = []
    inds_abc_bra = []
    inds_abc_ket = []
    inds_out = []

    upper_inds = iter(range(ndim_abc, 2 * ndim_abc))

    for i in range(ndim_abc):
        if i in keep_a:
            up_ind = next(upper_inds)
            inds_ab_ket.append(i)
            inds_abc_bra.append(up_ind)
            inds_abc_ket.append(i)
            inds_out.append(up_ind)
        elif i in keep_b:
            up_ind = next(upper_inds)
            inds_ab_ket.append(up_ind)
            inds_abc_bra.append(up_ind)
            inds_abc_ket.append(i)
            inds_out.append(i)
        else:
            inds_abc_bra.append(i)
            inds_abc_ket.append(i)

    return inds_ab_ket, inds_abc_bra, inds_abc_ket, inds_out


def lazy_ptr_ppt_dot(psi_abc, psi_ab, dims, keep_a, keep_b):
    """Perform the 'lazy' evalution of
    ``partial_transpose(ptr(psi_abc, ...)) @ psi_ab``, that is, contract the
    tensor diagram in an efficient way that does not necessarily construct
    the explicit reduced density matrix. For a tripartite system, the partial
    trace is with respect to ``c``, while the partial tranpose is with
    respect to ``a/b``. In tensor diagram notation:
    ``
    +--------------+
    |   psi_ab     |
    +______________+  _____
     a|  ____    |   /     \
      | /   a\  b|   |c    |
      | | +-------------+  |
      | | |  psi_abc.H  |  |
      \ / +-------------+  |
       X                   |
      / \ +-------------+  |
      | | |   psi_abc   |  |
      | | +-------------+  |
      | \____/a  |b  |c    |
     a|          |   \_____/
    ``

    Parameters
    ----------
    psi_ab : ket
        State to partially trace, partially tranpose, then dot with another
        ket, with size ``prod(dims)``.
    psi_a : ket
        State to act on with the dot product, of size
        ``prod(dims[keep_a] + dims[keep_b])``.
    dims : sequence of int
        The sub dimensions of ``psi_abc``.
    keep_a : int
        Index of the subsystems in partition 'a', with respect to *all*
        the dimensions in ``dims``.
    keep_b : int
        Index of the subsystems in partition 'b', with respect to *all*
        the dimensions in ``dims``.

    Returns
    -------
    ket
    """
    # convert to tuple so can always cache
    keep_a = (keep_a,) if isinstance(keep_a, int) else tuple(keep_a)
    keep_b = (keep_b,) if isinstance(keep_b, int) else tuple(keep_b)

    inds_ab_ket, inds_abc_bra, inds_abc_ket, inds_out = \
        get_cntrct_inds_ptr_ppt_dot(len(dims), keep_a, keep_b)

    psi_abc_tensor = np.asarray(psi_abc).reshape(dims)
    dims_ab = [d for i, d in enumerate(dims) if (i in keep_a) or (i in keep_b)]

    # must have ``inds_out`` as resulting indices are not ordered
    # in the same way as input due to partial tranpose.
    return einsum(
        np.asarray(psi_ab).reshape(dims_ab), inds_ab_ket,
        psi_abc_tensor.conjugate(), inds_abc_bra,
        psi_abc_tensor, inds_abc_ket,
        inds_out,
    ).reshape(psi_ab.shape)
