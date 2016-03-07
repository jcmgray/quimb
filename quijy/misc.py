"""
Misc. functions not quantum related.
"""
import numpy as np
import scipy.sparse as sp
from scipy.interpolate import splrep, splev
import xarray as xr
from tqdm import tqdm
from .core import eye, kron


def progbar(it, **kwargs):
    """ tqdm progress bar with deifferent defaults. """
    return tqdm(it, ascii=True, leave=True, **kwargs)


def spline_resample(x, y, n=100, **kwargs):
    ix = np.linspace(x[0], x[-1], n)
    iy = splev(ix, splrep(x, y, **kwargs))
    return ix, iy


def xrmerge(*das, accept_new=False):
    """
    Aggregates xarray Datasets and DataArrays
    """
    da = das[0]
    for new_da in das[1:]:
        # First make sure both datasets have the same variables
        for data_var in new_da.data_vars:
            if data_var not in da.data_vars:
                da[data_var] = np.nan
        # Expand both to have same dimensions, padding with NaN
        da, new_da = xr.align(da, new_da, join='outer')
        # Fill NaNs one way or the other re. accept_new
        da = new_da.fillna(da) if accept_new else da.fillna(new_da)
    return da


def xrgroupby_to_dim(ds, dim):

    def gen_ds():
        for val, d in ds.groupby(dim):
            del d[dim]  # delete grouped labels
            d[dim] = [val]
            d, = xr.broadcast(d)
            yield d

    return xrmerge(*gen_ds())


def coords_compress(dims, inds):
    """
    Compress coordinates/dimensions by combining adjacent identities.
    """
    ndims = np.zeros(len(dims))
    ninds = np.zeros(len(inds))
    j = 0
    k = 0
    idsize = 1
    for i, dim in enumerate(dims):
        if i in inds:
            if idsize > 1:
                ndims[j] = idsize
                idsize = 1
                j += 1
            ninds[k] = j
            ndims[j] = dim
            k += 1
            j += 1
        else:
            idsize *= dim
    if idsize > 1:
        ndims[j] = idsize
        j += 1
    return ndims[:j], ninds[np.argsort(inds)]


def eyepad_old(op, dims, inds, sparse=None):
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
