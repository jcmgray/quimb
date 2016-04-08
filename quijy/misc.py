"""
Misc. functions not quantum related.
"""
import numpy as np
from scipy.interpolate import splrep, splev, PchipInterpolator
import xarray as xr
from tqdm import tqdm


def progbar(it, **kwargs):
    """ tqdm progress bar with deifferent defaults. """
    return tqdm(it, ascii=True, leave=True, **kwargs)


def resample(x, y, n=100, **kwargs):
    ix = np.linspace(x[0], x[-1], n)
    iy = PchipInterpolator(x, y, **kwargs)(ix)
    return ix, iy


def spline_resample(x, y, n=100, **kwargs):
    ix = np.linspace(x[0], x[-1], n)
    iy = splev(ix, splrep(x, y, **kwargs))
    return ix, iy


# -------------------------------------------------------------------------- #
# Convenience functions for working with xarray                              #
# -------------------------------------------------------------------------- #

def xrsmoosh(*dss, accept_new=False):
    """ Aggregates xarray Datasets and DataArrays """
    ds = dss[0]
    for new_ds in dss[1:]:
        # First make sure both datasets have the same variables
        for data_var in new_ds.data_vars:
            if data_var not in ds.data_vars:
                ds[data_var] = np.nan
        # Expand both to have same dimensions, padding with NaN
        ds, new_ds = xr.align(ds, new_ds, join="outer")
        # Fill NaNs one way or the other w.r.t. accept_new
        ds = new_ds.fillna(ds) if accept_new else ds.fillna(new_ds)
    return ds


def xrload(file_name, engine="h5netcdf", load_to_mem=True,
           create_new=True):
    """ Loads a xarray dataset. """
    try:
        ds = xr.open_dataset(file_name, engine=engine)
        if load_to_mem:
            ds.load()
            ds.close()
    except (RuntimeError, OSError) as e:
        if "o such" in str(e) and create_new:
            ds = xr.Dataset()
        else:
            raise e
    return ds


def xrsave(ds, file_name, engine="h5netcdf"):
    """ Saves a xarray dataset. """
    ds.to_netcdf(file_name, engine=engine)


def xrgroupby_to_dim(ds, dim):
    """ Convert a grouped coordinate to dimension. """
    def gen_ds():
        for val, d in ds.groupby(dim):
            del d[dim]  # delete grouped labels
            d[dim] = [val]
            d, = xr.broadcast(d)
            yield d

    return xrsmoosh(*gen_ds())
