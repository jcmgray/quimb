""" Misc. functions not quantum related. """
from functools import partial
from itertools import product
import numpy as np
from scipy.interpolate import splrep, splev, PchipInterpolator
import xarray as xr
from tqdm import tqdm
from .plot import ilineplot


progbar = partial(tqdm, ascii=True)


def param_runner(foo, params, num_progbars=0, _nl=0):
    """ Take a function foo and analyse it over all combinations of named
    variables' values, optionally showing progress.

    Parameters
    ----------
        foo: function to analyse
        params: list of tuples of form ((variable_name, [values]), ...)
        num_progbars: how many levels of nested progress bars to show
        _nl: internal variable used for keeping track of nested level

    Returns
    -------
        data: generator for array (list of lists) of dimension len(params) """
    # TODO: automatic multiprocessing?
    # TODO: inner function with const num_progbars, external pre and post proc
    if _nl == 0:
        if isinstance(params, dict):
            params = params.items()
        params = [*params]
    pname, pvals = params[0]
    if _nl < num_progbars:
        pvals = progbar(pvals, desc=pname)
    for pval in pvals:
        if len(params) == 1:
            yield foo(**{pname: pval})
        else:
            pfoo = partial(foo, **{pname: pval})
            yield [*param_runner(pfoo, params[1:], num_progbars, _nl=_nl+1)]


def sub_split(a, tolist=False):
    """ Split a multi-nested python list at the lowest dimension """
    return (b.astype(type(b.item(0)), copy=False).T
            for b in np.array(a, dtype=object, copy=False).T)


def np_param_runner(foo, params):
    """ Use numpy.vectorize and meshgrid to evaluate a function
    at all combinations of params, may be faster than case_runner but no
    live progress can be shown.

    Parameters
    ----------
        foo: function to evaluate
        params: list of tuples [(parameter, [parameter values]), ... ]

    Returns
    -------
        x: list of arrays, one for each return object of foo,
            each with ndim == len(params)

    # TODO: progbar? """
    params = [*params.items()] if isinstance(params, dict) else params
    prm_names, prm_vals = zip(*params)
    vprm_vals = np.meshgrid(*prm_vals, sparse=True, indexing='ij')
    vfoo = np.vectorize(foo)
    return vfoo(**{n: vv for n, vv in zip(prm_names, vprm_vals)})


def np_param_runner2(foo, params, otypes=None, num_progbars=0):
    """ Use numpy.vectorize and meshgrid to evaluate a function
    at all combinations of params, now with progress bar

    Parameters
    ----------
        foo: function to evaluate
        params: list of tuples [(parameter, [parameter values]), ... ]

    Returns
    -------
        x: list of arrays, one for each return object of foo,
            each with ndim == len(params) """
    # TODO: multiprocess
    params = [*params.items()] if isinstance(params, dict) else params
    pnames, pvals = zip(*params)
    pszs = [len(pval) for pval in pvals]
    pcoos = [[*range(psz)] for psz in pszs]
    first_run = True
    configs = zip(product(*pvals), product(*pcoos))
    for config, coo in progbar(configs, total=np.prod(pszs),
                               disable=num_progbars < 1):
        res = foo(**{n: vv for n, vv in zip(pnames, config)})
        # Use first result to calculate output array
        if first_run:
            multires = isinstance(res, (tuple, list))
            if multires:
                otypes = [type(y) for y in res]
                x = [np.empty(shape=pszs, dtype=otype) for otype in otypes]
            else:
                otype = type(res)
                x = np.empty(shape=pszs, dtype=otype)
            first_run = False
        if multires:
            for sx, y in zip(x, res):
                sx[coo] = y
        else:
            x[coo] = res
    return x


# -------------------------------------------------------------------------- #
# Convenience functions for working with xarray                              #
# -------------------------------------------------------------------------- #

def xr_param_runner(foo, params, result_names, num_progbars=-1, use_np=False):
    """ Take a function foo and analyse it over all combinations of named
    variables values, optionally showing progress and outputing to xarray.

    Parameters
    ----------
        foo: function to analyse
        params: list of tuples of form ((variable_name, [values]), ...)
        result_names, name of dataset's main variable, i.e. the results of foo
        num_progbars: how many levels of nested progress bars to show

    Returns
    -------
        ds: xarray Dataset with appropirate coordinates. """
    params = [*params.items()] if isinstance(params, dict) else params

    data = (np_param_runner(foo, params) if use_np else
            [*param_runner(foo, params, num_progbars=num_progbars)])

    ds = xr.Dataset()
    for var, vals in params:
        ds.coords[var] = [*vals]

    if isinstance(result_names, (list, tuple)):

        for result_name, sdata in zip(result_names,
                                      data if use_np else sub_split(data)):
            ds[result_name] = ([var for var, _ in params], sdata)
    else:
        ds[result_names] = ([var for var, _ in params], data)
    return ds


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
        try:
            ds = xr.open_dataset(file_name, engine=engine)
        except AttributeError as e1:
            if "object has no attribute" in str(e1):
                ds = xr.open_dataset(file_name, engine="netcdf4")
            else:
                raise e1
        if load_to_mem:
            ds.load()
            ds.close()
    except (RuntimeError, OSError) as e2:
        if "o such" in str(e2) and create_new:
            ds = xr.Dataset()
        else:
            raise e2
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


def resample(x, y, n=100, **kwargs):
    ix = np.linspace(x[0], x[-1], n)
    iy = PchipInterpolator(x, y, **kwargs)(ix)
    return ix, iy


def spline_resample(x, y, n=100, **kwargs):
    ix = np.linspace(x[0], x[-1], n)
    iy = splev(ix, splrep(x, y, **kwargs))
    return ix, iy


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
