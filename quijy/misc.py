"""
Misc. functions not quantum related.
"""
import numpy as np
from tqdm import tqdm
import xarray as xr


def progbar(it, **kwargs):
    """ tqdm progress bar with deifferent defaults. """
    return tqdm(it, ascii=True, leave=True, **kwargs)


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
