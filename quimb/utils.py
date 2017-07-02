"""Miscellenous
"""
try:
    from tqdm import tqdm
    found_tqdm = True
except ImportError:
    found_tqdm = False


class _ctqdm(tqdm):
    """A continuous version of tqdm, so that it can
    be updated with a float within some pre-given
    range, rather than a number of steps.
    """

    def __init__(self, *args, total=100, **kwargs):
        """
        Parameters
        ----------
            *args : (stop) or (start, stop)
                Stopping point (and starting point if
                len(args) == 2) of window within which
                to evaluate progress.
            total : int
                The number of steps to represent the
                continuous progress with.
            **kwargs
                Supplied to tqdm.tqdm
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
            x :  float
                Current position within the range
                [self.start, self.stop]
        """
        num_update = int(
            (self.total + 1) * (x - self.start) / self.range - self.step
        )
        if num_update > 0:
            self.update(num_update)
            self.step += num_update


class no_tqdm(object):
    """Dummy class for raising no tqdm import error.
    """

    def __init__(self, *args, **kwargs):
        raise ImportError("The library `tqdm` must be installed in order to "
                          "show progress bars.")


if found_tqdm:
    progbar = tqdm
    continuous_progbar = _ctqdm
else:
    progbar = no_tqdm
    continuous_progbar = no_tqdm
