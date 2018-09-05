"""Miscellenous
"""
import importlib
import itertools


def find_library(x):
    """Check if library is installed.

    Parameters
    ----------
    x : str
        Name of library

    Returns
    -------
    bool
        If library is available.
    """
    return importlib.util.find_spec(x) is not None


def raise_cant_find_library_function(x, extra_msg=None):
    """Return function to flag up a missing necessary library.

    This is simplify the task of flagging optional dependencies only at the
    point at which they are needed, and not earlier.

    Parameters
    ----------
    x : str
        Name of library
    extra_msg : str, optional
        Make the function print this message as well, for additional
        information.

    Returns
    -------
    callable
        A mock function that when called, raises an import error specifying
        the required library.
    """

    def function_that_will_raise(*_, **__):
        error_msg = "The library {} is not installed. ".format(x)
        if extra_msg is not None:
            error_msg += extra_msg
        raise ImportError(error_msg)

    return function_that_will_raise


FOUND_TQDM = find_library('tqdm')
if FOUND_TQDM:
    from tqdm import tqdm

    class _ctqdm(tqdm):
        """A continuous version of tqdm, so that it can be updated with a float
        within some pre-given range, rather than a number of steps.

        Parameters
        ----------
        args : (stop) or (start, stop)
            Stopping point (and starting point if ``len(args) == 2``) of window
            within which to evaluate progress.
        total : int
            The number of steps to represent the continuous progress with.
        kwargs
            Supplied to ``tqdm.tqdm``
        """

        def __init__(self, *args, total=100, **kwargs):
            """
            """
            super(_ctqdm, self).__init__(total=total, unit="%", **kwargs)

            if len(args) == 2:
                self.start, self.stop = args
            else:
                self.start, self.stop = 0, args[0]

            self.range = self.stop - self.start
            self.step = 1

        def cupdate(self, x):
            """'Continuous' update of progress bar.

            Parameters
            ----------
            x : float
                Current position within the range ``[self.start, self.stop]``.
            """
            num_update = int(
                (self.total + 1) * (x - self.start) / self.range - self.step
            )
            if num_update > 0:
                self.update(num_update)
                self.step += num_update

    progbar = tqdm
    continuous_progbar = _ctqdm
else:  # pragma: no cover
    extra_msg = "This is needed to show progress bars."
    progbar = raise_cant_find_library_function("tqdm", extra_msg)
    continuous_progbar = raise_cant_find_library_function("tqdm", extra_msg)


def deprecated(fn, old_name, new_name):

    def new_fn(*args, **kwargs):
        import warnings
        warnings.warn("The {} function is deprecated in favor "
                      "of {}".format(old_name, new_name),
                      Warning)
        return fn(*args, **kwargs)

    return new_fn


def int2tup(x):
    return (x if isinstance(x, tuple) else
            (x,) if isinstance(x, int) else
            tuple(x))


def pairwise(iterable):
    """Iterate over each pair of neighbours in ``iterable``.
    """
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)


def three_line_multi_print(l1, l2, l3, max_width=None):
    if max_width is None:
        import shutil
        max_width, _ = shutil.get_terminal_size()

    if len(l2) <= max_width:
        print(l1)
        print(l2)
        print(l3)
    else:  # pragma: no cover
        max_width -= 10  # for ellipses and pad
        n_lines = (len(l2) - 1) // max_width + 1
        for i in range(n_lines):
            if i == 0:
                print("   ", l1[i * max_width:(i + 1) * max_width], "   ")
                print("   ", l2[i * max_width:(i + 1) * max_width], "...")
                print("   ", l3[i * max_width:(i + 1) * max_width], "   ")
                print(("{:^" + str(max_width) + "}").format("..."))
            elif i == n_lines - 1:
                print("   ", l1[i * max_width:(i + 1) * max_width])
                print("...", l2[i * max_width:(i + 1) * max_width])
                print("   ", l3[i * max_width:(i + 1) * max_width])
            else:
                print("   ", l1[i * max_width:(i + 1) * max_width], "   ")
                print("...", l2[i * max_width:(i + 1) * max_width], "...")
                print("   ", l3[i * max_width:(i + 1) * max_width], "   ")
                print(("{:^" + str(max_width) + "}").format("..."))


def functions_equal(fn1, fn2):
    """Check equality of the code in ``fn1`` and ``fn2``.
    """

    try:
        code1 = fn1.__code__.co_code
    except AttributeError:
        code1 = fn1.__func__.__code__.co_code

    try:
        code2 = fn2.__code__.co_code
    except AttributeError:
        code2 = fn2.__func__.__code__.co_code

    return code1 == code2


def save_to_disk(obj, fname, **dump_opts):
    """Save an object to disk using joblib.dump.
    """
    import joblib
    return joblib.dump(obj, fname, **dump_opts)


def load_from_disk(fname, **load_opts):
    """Load an object form disk using joblib.load.
    """
    import joblib
    return joblib.load(fname, **load_opts)


class Verbosify:  # pragma: no cover
    """Decorator for making functions print their inputs. Simply for
    illustrating a MPI example in the docs.
    """

    def __init__(self, fn, highlight=None, mpi=False):
        self.fn = fn
        self.highlight = highlight
        self.mpi = mpi

    def __call__(self, *args, **kwargs):
        if self.mpi:
            from mpi4py import MPI
            pre_msg = "{}: ".format(MPI.COMM_WORLD.Get_rank())
        else:
            pre_msg = ""

        if self.highlight is None:
            print("{}args {}, kwargs {}".format(pre_msg, args, kwargs))
        else:
            print("{}{}={}".format(pre_msg, self.highlight,
                                   kwargs[self.highlight]))
        return self.fn(*args, **kwargs)


def has_cupy():
    return importlib.util.find_spec("cupy") is not None
