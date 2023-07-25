"""Misc utility functions.
"""
import functools
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
    partition = cytoolz.partition
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
    partition = toolz.partition
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
    """Check whether ``value`` takes one of ``valid`` options, and raise an
    informative error if not.
    """
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
    """Mark a function as deprecated, and indicate the new name.
    """

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
    """Print multiple lines, with a maximum width.
    """
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


def format_number_with_error(x, err):
    """Given ``x`` with error ``err``, format a string showing the relevant
    digits of ``x`` with two significant digits of the error bracketed, and
    overall exponent if necessary.

    Parameters
    ----------
    x : float
        The value to print.
    err : float
        The error on ``x``.

    Returns
    -------
    str

    Examples
    --------

        >>> print_number_with_uncertainty(0.1542412, 0.0626653)
        '0.154(63)'

        >>> print_number_with_uncertainty(-128124123097, 6424)
        '-1.281241231(64)e+11'

    """
    # compute an overall scaling for both values
    x_exponent = max(
        int(f'{x:e}'.split('e')[1]),
        int(f'{err:e}'.split('e')[1]) + 1,
    )
    # for readability try and show values close to 1 with no exponent
    hide_exponent = (
         # nicer showing 0.xxx(yy) than x.xx(yy)e-1
        (x_exponent in (0, -1)) or
        # also nicer showing xx.xx(yy) than x.xxx(yy)e+1
        ((x_exponent == +1) and (err < abs(x / 10)))
    )
    if hide_exponent:
        suffix = ""
    else:
        x = x / 10**x_exponent
        err = err / 10**x_exponent
        suffix = f"e{x_exponent:+03d}"

    # work out how many digits to print
    # format the main number and bracketed error
    mantissa, exponent = f'{err:.1e}'.split('e')
    mantissa, exponent = mantissa.replace('.', ''), int(exponent)
    return f'{x:.{abs(exponent) + 1}f}({mantissa}){suffix}'


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

    def __deepcopy__(self, memo):
        # always use hashable entries so just take normal copy
        new = self.copy()
        memo[id(self)] = new
        return new

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


TREE_MAP_REGISTRY = {}
TREE_APPLY_REGISTRY = {}
TREE_ITER_REGISTRY = {}


def tree_register_container(cls, mapper, iterator, applier):
    """Register a new container type for use with ``tree_map`` and
    ``tree_apply``.

    Parameters
    ----------
    cls : type
        The container type to register.
    mapper : callable
        A function that takes ``f``, ``tree`` and ``is_leaf`` and returns a new
        tree of type ``cls`` with ``f`` applied to all leaves.
    applier : callable
        A function that takes ``f``, ``tree`` and ``is_leaf`` and applies ``f``
        to all leaves in ``tree``.
    """
    TREE_MAP_REGISTRY[cls] = mapper
    TREE_ITER_REGISTRY[cls] = iterator
    TREE_APPLY_REGISTRY[cls] = applier


IS_CONTAINER_CACHE = {}


def is_not_container(x):
    """The default function to determine if an object is a leaf. This simply
    checks if the object is an instance of any of the registered container
    types.
    """
    try:
        return IS_CONTAINER_CACHE[x.__class__]
    except KeyError:
        isleaf = not any(isinstance(x, cls) for cls in TREE_MAP_REGISTRY)
        IS_CONTAINER_CACHE[x.__class__] = isleaf
        return isleaf


def _tmap_identity(f, tree, is_leaf):
    return tree


TREE_MAPPER_CACHE = {}


def tree_map(f, tree, is_leaf=is_not_container):
    """Map ``f`` over all leaves in ``tree``, returning a new pytree.

    Parameters
    ----------
    f : callable
        A function to apply to all leaves in ``tree``.
    tree : pytree
        A nested sequence of tuples, lists, dicts and other objects.
    is_leaf : callable
        A function to determine if an object is a leaf, ``f`` is only applied
        to objects for which ``is_leaf(x)`` returns ``True``.

    Returns
    -------
    pytree
    """
    if is_leaf(tree):
        return f(tree)

    try:
        return TREE_MAPPER_CACHE[tree.__class__](f, tree, is_leaf)
    except KeyError:
        # reverse so later registered classes take precedence
        for cls, mapper in reversed(TREE_MAP_REGISTRY.items()):
            if isinstance(tree, cls):
                break
        else:
            # neither leaf nor container -> simply return it
            mapper = _tmap_identity
        TREE_MAPPER_CACHE[tree.__class__] = mapper
        return mapper(f, tree, is_leaf)


def empty(tree, is_leaf):
    return iter(())


TREE_ITER_CACHE = {}


def tree_iter(tree, is_leaf=is_not_container):
    """Iterate over all leaves in ``tree``.

    Parameters
    ----------
    f : callable
        A function to apply to all leaves in ``tree``.
    tree : pytree
        A nested sequence of tuples, lists, dicts and other objects.
    is_leaf : callable
        A function to determine if an object is a leaf, ``f`` is only applied
        to objects for which ``is_leaf(x)`` returns ``True``.
    """
    if is_leaf(tree):
        yield tree
        return

    try:
        yield from TREE_ITER_CACHE[tree.__class__](tree, is_leaf)
    except KeyError:
        # reverse so later registered classes take precedence
        for cls, iterator in reversed(TREE_ITER_REGISTRY.items()):
            if isinstance(tree, cls):
                break
        else:
            # neither leaf nor container -> simply ignore it
            iterator = empty
        TREE_ITER_CACHE[tree.__class__] = iterator
        yield from iterator(tree, is_leaf)


def nothing(f, tree, is_leaf):
    pass


TREE_APPLIER_CACHE = {}


def tree_apply(f, tree, is_leaf=is_not_container):
    """Apply ``f`` to all leaves in ``tree``, no new pytree is built.

    Parameters
    ----------
    f : callable
        A function to apply to all leaves in ``tree``.
    tree : pytree
        A nested sequence of tuples, lists, dicts and other objects.
    is_leaf : callable
        A function to determine if an object is a leaf, ``f`` is only applied
        to objects for which ``is_leaf(x)`` returns ``True``.
    """
    if is_leaf(tree):
        f(tree)
        return

    try:
        TREE_APPLIER_CACHE[tree.__class__](f, tree, is_leaf)
    except KeyError:
        # reverse so later registered classes take precedence
        for cls, applier in reversed(TREE_APPLY_REGISTRY.items()):
            if isinstance(tree, cls):
                break
        else:
            # neither leaf nor container -> simply ignore it
            applier = nothing
        TREE_APPLIER_CACHE[tree.__class__] = applier
        applier(f, tree, is_leaf)


class Leaf:

    __slots__ = ()

    def __repr__(self):
        return "Leaf"


Leaf = Leaf()


def is_leaf_object(x):
    return x is Leaf


def tree_flatten(tree, get_ref=False, is_leaf=is_not_container):
    """Flatten ``tree`` into a list of leaves.

    Parameters
    ----------
    tree : pytree
        A nested sequence of tuples, lists, dicts and other objects.
    is_leaf : callable
        A function to determine if an object is a leaf, only objects for which
        ``is_leaf(x)`` returns ``True`` are returned in the flattened list.

    Returns
    -------
    objs : list
        The flattened list of leaf objects.
    (ref_tree) : pytree
        If ``get_ref`` is ``True``, a reference tree, with leaves of None, is
        returned which can be used to reconstruct the original tree.
    """
    objs = []
    if get_ref:
        # return a new tree with Leaf leaves, as well as the flattened list

        def f(x):
            objs.append(x)
            return Leaf

        ref_tree = tree_map(f, tree, is_leaf)
        return objs, ref_tree
    else:
        tree_apply(objs.append, tree, is_leaf)
        return objs


def tree_unflatten(objs, tree, is_leaf=is_leaf_object):
    """Unflatten ``objs`` into a pytree of the same structure as ``tree``.

    Parameters
    ----------
    objs : sequence
        A sequence of objects to be unflattened into a pytree.
    tree : pytree
        A nested sequence of tuples, lists, dicts and other objects, the objs
        will be inserted into a new pytree of the same structure.
    is_leaf : callable
        A function to determine if an object is a leaf, only objects for which
        ``is_leaf(x)`` returns ``True`` will have the next item from ``objs``
        inserted. By default checks for the ``Leaf`` object inserted by
        ``tree_flatten(..., get_ref=True)``.

    Returns
    -------
    pytree
    """
    objs = iter(objs)
    return tree_map(lambda _: next(objs), tree, is_leaf)


def tree_map_tuple(f, tree, is_leaf):
    return tuple(tree_map(f, x, is_leaf) for x in tree)


def tree_iter_tuple(tree, is_leaf):
    for x in tree:
        yield from tree_iter(x, is_leaf)


def tree_apply_tuple(f, tree, is_leaf):
    for x in tree:
        tree_apply(f, x, is_leaf)


tree_register_container(
    tuple, tree_map_tuple, tree_iter_tuple, tree_apply_tuple
)


def tree_map_list(f, tree, is_leaf):
    return [tree_map(f, x, is_leaf) for x in tree]


def tree_iter_list(tree, is_leaf):
    for x in tree:
        yield from tree_iter(x, is_leaf)


def tree_apply_list(f, tree, is_leaf):
    for x in tree:
        tree_apply(f, x, is_leaf)


tree_register_container(list, tree_map_list, tree_iter_list, tree_apply_list)


def tree_map_dict(f, tree, is_leaf):
    return {k: tree_map(f, v, is_leaf) for k, v in tree.items()}


def tree_iter_dict(tree, is_leaf):
    for v in tree.values():
        yield from tree_iter(v, is_leaf)


def tree_apply_dict(f, tree, is_leaf):
    for v in tree.values():
        tree_apply(f, v, is_leaf)


tree_register_container(dict, tree_map_dict, tree_iter_dict, tree_apply_dict)



# a style to use for matplotlib that works with light and dark backgrounds
NEUTRAL_STYLE = {
    'axes.edgecolor': (0.5, 0.5, 0.5),
    'axes.facecolor': (0, 0, 0, 0),
    'axes.grid': True,
    'axes.labelcolor': (0.5, 0.5, 0.5),
    'axes.spines.right': False,
    'axes.spines.top': False,
    'figure.facecolor': (0, 0, 0, 0),
    'grid.alpha': 0.1,
    'grid.color': (0.5, 0.5, 0.5),
    'legend.frameon': False,
    'text.color': (0.5, 0.5, 0.5),
    'xtick.color': (0.5, 0.5, 0.5),
    'xtick.minor.visible': True,
    'ytick.color': (0.5, 0.5, 0.5),
    'ytick.minor.visible': True,
}


def default_to_neutral_style(fn):
    """Wrap a function or method to use the neutral style by default.
    """

    @functools.wraps(fn)
    def wrapper(
        *args,
        style="neutral",
        show_and_close=True,
        **kwargs
    ):
        import matplotlib.pyplot as plt

        if style == "neutral":
            style = NEUTRAL_STYLE
        elif not style:
            style = {}

        with plt.style.context(style):
            out = fn(*args, **kwargs)

            if show_and_close:
                plt.show()
                plt.close()

            return out

    return wrapper
