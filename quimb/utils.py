"""Miscellenous
"""
import itertools
import collections
from importlib.util import find_spec


try:
    import cytoolz
    last = cytoolz.last
    concat = cytoolz.concat
    frequencies = cytoolz.frequencies
    partition_all = cytoolz.partition_all
    merge_with = cytoolz.merge_with
    valmap = cytoolz.valmap
    partitionby = cytoolz.partitionby
    concatv = cytoolz.concatv
    partition_all = cytoolz.partition_all
    compose = cytoolz.compose
    identity = cytoolz.identity
    isiterable = cytoolz.isiterable
    unique = cytoolz.unique
    keymap = cytoolz.keymap
except ImportError:
    import toolz
    last = toolz.last
    concat = toolz.concat
    frequencies = toolz.frequencies
    partition_all = toolz.partition_all
    merge_with = toolz.merge_with
    valmap = toolz.valmap
    partitionby = toolz.partitionby
    concatv = toolz.concatv
    partition_all = toolz.partition_all
    compose = toolz.compose
    identity = toolz.identity
    isiterable = toolz.isiterable
    unique = toolz.unique
    keymap = toolz.keymap


_CHECK_OPT_MSG = "Option `{}` should be one of {}, but got '{}'."


def check_opt(name, value, valid):
    if value not in valid:
        raise ValueError(_CHECK_OPT_MSG.format(name, valid, value))


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
    return find_spec(x) is not None


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
        error_msg = f"The library {x} is not installed. "
        if extra_msg is not None:
            error_msg += extra_msg
        raise ImportError(error_msg)

    return function_that_will_raise


FOUND_TQDM = find_library('tqdm')
if FOUND_TQDM:
    from tqdm import tqdm

    class continuous_progbar(tqdm):
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
            kwargs.setdefault('ascii', True)
            super(continuous_progbar, self).__init__(total=total,
                                                     unit="%", **kwargs)

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

    def progbar(*args, **kwargs):
        kwargs.setdefault('ascii', True)
        return tqdm(*args, **kwargs)

else:  # pragma: no cover
    extra_msg = "This is needed to show progress bars."
    progbar = raise_cant_find_library_function("tqdm", extra_msg)
    continuous_progbar = raise_cant_find_library_function("tqdm", extra_msg)


def deprecated(fn, old_name, new_name):

    def new_fn(*args, **kwargs):
        import warnings
        warnings.warn(f"The {old_name} function is deprecated in favor "
                      f"of {new_name}", Warning)
        return fn(*args, **kwargs)

    return new_fn


def int2tup(x):
    return (x if isinstance(x, tuple) else
            (x,) if isinstance(x, int) else
            tuple(x))


def ensure_dict(x):
    """Make sure ``x`` is a ``dict``, creating an empty one if ``x is None``.
    """
    if x is None:
        return {}
    return dict(x)


def pairwise(iterable):
    """Iterate over each pair of neighbours in ``iterable``.
    """
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)


def print_multi_line(*lines, max_width=None):
    if max_width is None:
        import shutil
        max_width, _ = shutil.get_terminal_size()

    max_line_lenth = max(len(ln) for ln in lines)

    if max_line_lenth <= max_width:
        for ln in lines:
            print(ln)

    else:  # pragma: no cover
        max_width -= 10  # for ellipses and pad
        n_lines = len(lines)
        n_blocks = (max_line_lenth - 1) // max_width + 1

        for i in range(n_blocks):
            if i == 0:
                for j, l in enumerate(lines):
                    print(
                        "..." if j == n_lines // 2 else "   ",
                        l[i * max_width:(i + 1) * max_width],
                        "..." if j == n_lines // 2 else "   "
                    )
                print(("{:^" + str(max_width) + "}").format("..."))
            elif i == n_blocks - 1:
                for ln in lines:
                    print("   ", ln[i * max_width:(i + 1) * max_width])
            else:
                for j, ln in enumerate(lines):
                    print(
                        "..." if j == n_lines // 2 else "   ",
                        ln[i * max_width:(i + 1) * max_width],
                        "..." if j == n_lines // 2 else "   ",
                    )
                print(("{:^" + str(max_width) + "}").format("..."))


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
            pre_msg = f"{MPI.COMM_WORLD.Get_rank()}: "
        else:
            pre_msg = ""

        if self.highlight is None:
            print(f"{pre_msg} args {args}, kwargs {kwargs}")
        else:
            print(f"{pre_msg}{self.highlight}={kwargs[self.highlight]}")
        return self.fn(*args, **kwargs)


class oset:
    """An ordered set which stores elements as the keys of dict (ordered as of
    python 3.6). 'A few times' slower than using a set directly for small
    sizes, but makes everything deterministic.
    """

    __slots__ = ('_d',)

    def __init__(self, it=()):
        self._d = dict.fromkeys(it)

    @classmethod
    def _from_dict(cls, d):
        obj = object.__new__(oset)
        obj._d = d
        return obj

    @classmethod
    def from_dict(cls, d):
        """Public method makes sure to copy incoming dictionary.
        """
        return oset._from_dict(d.copy())

    def copy(self):
        return oset.from_dict(self._d)

    def add(self, k):
        self._d[k] = None

    def discard(self, k):
        self._d.pop(k, None)

    def remove(self, k):
        del self._d[k]

    def clear(self):
        self._d.clear()

    def update(self, *others):
        for o in others:
            self._d.update(o._d)

    def union(self, *others):
        u = self.copy()
        u.update(*others)
        return u

    def intersection_update(self, *others):
        if len(others) > 1:
            si = set.intersection(*(set(o._d) for o in others))
        else:
            si = others[0]._d
        self._d = {k: None for k in self._d if k in si}

    def intersection(self, *others):
        n_others = len(others)
        if n_others == 0:
            return self.copy()
        elif n_others == 1:
            si = others[0]._d
        else:
            si = set.intersection(*(set(o._d) for o in others))
        return oset._from_dict({k: None for k in self._d if k in si})

    def difference_update(self, *others):
        if len(others) > 1:
            su = set.union(*(set(o._d) for o in others))
        else:
            su = others[0]._d
        self._d = {k: None for k in self._d if k not in su}

    def difference(self, *others):
        if len(others) > 1:
            su = set.union(*(set(o._d) for o in others))
        else:
            su = others[0]._d
        return oset._from_dict({k: None for k in self._d if k not in su})

    def popleft(self):
        k = next(iter(self._d))
        del self._d[k]
        return k

    def popright(self):
        return self._d.popitem()[0]

    def __eq__(self, other):
        if isinstance(other, oset):
            return self._d == other._d
        return False

    def __or__(self, other):
        return self.union(other)

    def __ior__(self, other):
        self.update(other)
        return self

    def __and__(self, other):
        return self.intersection(other)

    def __iand__(self, other):
        self.intersection_update(other)
        return self

    def __sub__(self, other):
        return self.difference(other)

    def __isub__(self, other):
        self.difference_update(other)
        return self

    def __len__(self):
        return self._d.__len__()

    def __iter__(self):
        return self._d.__iter__()

    def __contains__(self, x):
        return self._d.__contains__(x)

    def __repr__(self):
        return f"oset({list(self._d)})"


class LRU(collections.OrderedDict):
    """Least recently used dict, which evicts old items. Taken from python
    collections OrderedDict docs.
    """

    def __init__(self, maxsize, *args, **kwds):
        self.maxsize = maxsize
        super().__init__(*args, **kwds)

    def __getitem__(self, key):
        value = super().__getitem__(key)
        self.move_to_end(key)
        return value

    def __setitem__(self, key, value):
        if key in self:
            self.move_to_end(key)
        super().__setitem__(key, value)
        if len(self) > self.maxsize:
            oldest = next(iter(self))
            del self[oldest]


def gen_bipartitions(it):
    """Generate all unique bipartitions of ``it``. Unique meaning
    ``(1, 2), (3, 4)`` is considered the same as ``(3, 4), (1, 2)``.
    """
    n = len(it)
    if n:
        for i in range(1, 2**(n - 1)):
            bitstring_repr = f'{i:0>{n}b}'
            l, r = [], []
            for b, x in zip(bitstring_repr, it):
                (l if b == '0' else r).append(x)
            yield l, r
