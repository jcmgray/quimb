"""
Misc. functions not quantum related.
"""
import numpy as np
from scipy.interpolate import splrep, splev, PchipInterpolator
import xarray as xr
from tqdm import tqdm
from .plot import ilineplot


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
    """
    Aggregates xarray Datasets and DataArrays
    """
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


def time_functions(funcs, func_names, setup_str, sig_str, ns, rep_func=None):
    """ Calculate and plot how a number of functions exec time scales

    Parameters
    ----------
        funcs: list of funcs
        func_names: list of function names
        setup_str: actions to perform before each function
        sig_str: how arguments from setup_str and given to funcs
        ns: range of sizes
        rep_func(n): function for computing the number of repeats

    Returns
    -------
        Plots time vs n for each function. """
    from timeit import Timer

    sz_n = len(ns)
    ts = np.zeros((sz_n, len(funcs)))

    def basic_scaling(n):
        return min(max(int(3 * (2**max(ns))/(2**n)), 1), 10000)

    rep_func = basic_scaling if rep_func is None else rep_func

    for i, n in progbar(enumerate(ns), total=sz_n):
        timers = [Timer(func.__name__+sig_str, 'n='+str(n)+';'+setup_str,
                        globals=globals()) for func in funcs]
        reps = rep_func(n)
        for j, timer in enumerate(timers):
            ts[i, j] = timer.timeit(number=reps)/reps

    ds = xr.Dataset()
    ds.coords['n'] = ns
    ds.coords['func'] = func_names
    ds['t'] = (('n', 'func'), ts)
    ilineplot(ds, 't', 'n', 'func', logy=True)
