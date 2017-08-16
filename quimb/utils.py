"""Miscellenous
"""
import importlib


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
else:  # pragma: no cover
    tqdm = None


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


if FOUND_TQDM:
    progbar = tqdm
    continuous_progbar = _ctqdm
else:  # pragma: no cover
    extra_msg = "This is needed to show progress bars."
    progbar = raise_cant_find_library_function("tqdm", extra_msg)
    continuous_progbar = raise_cant_find_library_function("tqdm", extra_msg)
