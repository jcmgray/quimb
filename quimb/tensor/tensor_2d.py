"""Classes and algorithms related to 2D tensor networks.
"""
import re
import random
import functools
from operator import add
from numbers import Integral
from itertools import product, cycle, combinations, count, chain
from collections import defaultdict

from autoray import do, infer_backend, get_dtype_name
import opt_einsum as oe

from ..gen.operators import swap
from ..gen.rand import randn, seed_rand
from ..utils import (
    deprecated,
    print_multi_line,
    check_opt,
    pairwise,
    ensure_dict,
)
from ..utils import progbar as Progbar
from . import array_ops as ops
from .tensor_core import (
    bonds_size,
    bonds,
    oset_union,
    oset,
    rand_uuid,
    tags_to_oset,
    tensor_contract,
    Tensor,
    TensorNetwork,
)
from .tensor_arbgeom import (
    tensor_network_apply_op_vec,
    TensorNetworkGen,
    TensorNetworkGenOperator,
    TensorNetworkGenVector,
)
from .tensor_1d import maybe_factor_gate_into_tensor
from . import decomp


def manhattan_distance(coo_a, coo_b):
    return sum(abs(coo_a[i] - coo_b[i]) for i in range(2))


def nearest_neighbors(coo):
    i, j = coo
    return ((i - 1, j), (i, j - 1), (i, j + 1), (i + 1, j))


def gen_2d_bonds(Lx, Ly, steppers=None, coo_filter=None):
    """Convenience function for tiling pairs of bond coordinates on a 2D
    lattice given a function like ``lambda i, j: (i + 1, j + 1)``.

    Parameters
    ----------
    Lx : int
        The number of rows.
    Ly : int
        The number of columns.
    steppers : callable or sequence of callable, optional
        Function(s) that take args ``(i, j)`` and generate another coordinate,
        thus defining a bond. Only valid steps are taken. If not given,
        defaults to nearest neighbor bonds.
    coo_filter : callable
        Function that takes args ``(i, j)`` and only returns ``True`` if this
        is to be a valid starting coordinate.

    Yields
    ------
    bond : tuple[tuple[int, int], tuple[int, int]]
        A pair of coordinates.

    Examples
    --------

    Generate nearest neighbor bonds:

        >>> for bond in gen_2d_bonds(2, 2, [lambda i, j: (i, j + 1),
        >>>                                 lambda i, j: (i + 1, j)]):
        >>>     print(bond)
        ((0, 0), (0, 1))
        ((0, 0), (1, 0))
        ((0, 1), (1, 1))
        ((1, 0), (1, 1))

    Generate next nearest neighbor digonal bonds:

        >>> for bond in gen_2d_bonds(2, 2, [lambda i, j: (i + 1, j + 1),
        >>>                                 lambda i, j: (i + 1, j - 1)]):
        >>>     print(bond)
        ((0, 0), (1, 1))
        ((0, 1), (1, 0))

    """
    if steppers is None:
        steppers = [
            lambda i, j: (i, j + 1),
            lambda i, j: (i + 1, j),
        ]

    if callable(steppers):
        steppers = (steppers,)

    for i, j in product(range(Lx), range(Ly)):
        if (coo_filter is None) or coo_filter(i, j):
            for stepper in steppers:
                i2, j2 = stepper(i, j)
                if (0 <= i2 < Lx) and (0 <= j2 < Ly):
                    yield (i, j), (i2, j2)


def gen_2d_plaquette(coo0, steps):
    """Generate a plaquette at site ``coo0`` by stepping first in ``steps`` and
    then the reverse steps.

    Parameters
    ----------
    coo0 : tuple
        The coordinate of the first site in the plaquette.
    steps : tuple
        The steps to take to generate the plaquette. Each element should be
        one of ``('x+', 'x-', 'y+', 'y-')``.

    Yields
    ------
    coo : tuple
        The coordinates of the sites in the plaquette, including the last
        site which will be the same as the first.
    """
    x, y = coo0
    smap = {"+": +1, "-": -1}
    step_backs = []
    yield x, y
    for step in steps:
        d, s = step
        x, y = {
            "x": (x + smap[s], y),
            "y": (x, y + smap[s]),
        }[d]
        yield x, y
        step_backs.append(d + "-" if s == "+" else "-")
    for step in step_backs:
        d, s = step
        x, y = {
            "x": (x + smap[s], y),
            "y": (x, y + smap[s]),
        }[d]
        yield x, y


def gen_2d_plaquettes(Lx, Ly, tiling):
    """Generate a tiling of plaquettes in a square 2D lattice.

    Parameters
    ----------
    Lx : int
        The length of the lattice in the x direction.
    Ly : int
        The length of the lattice in the y direction.
    tiling : {'1', '2', 'full'}
        The tiling to use:

        - '1': plaquettes in a checkerboard pattern, such that each edge
            is covered by a maximum of one plaquette.
        - '2' or 'full': dense tiling of plaquettes. All bulk edges will
            be covered twice.

    Yields
    ------
    plaquette : tuple[tuple[int]]
        The coordinates of the sites in each plaquette, including the last
        site which will be the same as the first.
    """
    if str(tiling) == "1":
        for x, y in product(range(Lx), range(Ly)):
            if ((x + y) % 2 == 0) and (x < Lx - 1 and y < Ly - 1):
                yield tuple(gen_2d_plaquette((x, y), ("x+", "y+")))
    elif str(tiling) in ("2", "full"):
        for x, y in product(range(Lx), range(Ly)):
            if x < Lx - 1 and y < Ly - 1:
                yield tuple(gen_2d_plaquette((x, y), ("x+", "y+")))
    else:
        raise ValueError("`tiling` must be one of: '1', '2', 'full'.")


def gen_2d_strings(Lx, Ly):
    """Generate all length-wise strings in a square 2D lattice."""
    for x in range(Lx):
        yield tuple((x, y) for y in range(Ly))
    for y in range(Ly):
        yield tuple((x, y) for x in range(Lx))


class Rotator2D:
    """Object for rotating coordinates and various contraction functions so
    that the core algorithms only have to written once, but nor does the actual
    TN have to be modified.

    Parameters
    ----------
    tn : TensorNetwork2D
        The tensor network to rotate coordinates for.
    xrange : tuple[int, int]
        The range of x-coordinates to range over.
    yrange : tuple[int, int]
        The range of y-coordinates to range over.
    from_which : {'xmin', 'xmax', 'ymin', 'ymax'}
        The direction to sweep from.
    stepsize : int, optional
        The step size to use when sweeping.
    """

    def __init__(self, tn, xrange, yrange, from_which, stepsize=1):
        check_opt("from_which", from_which, {"xmin", "xmax", "ymin", "ymax"})

        if xrange is None:
            xrange = (0, tn.Lx - 1)
        if yrange is None:
            yrange = (0, tn.Ly - 1)

        self.tn = tn
        self.xrange = xrange
        self.yrange = yrange
        self.from_which = from_which
        self.plane = from_which[0]

        if self.plane == "x":
            # -> no rotation needed
            self.imin, self.imax = sorted(xrange)
            self.jmin, self.jmax = sorted(yrange)
            self.x_tag = tn.x_tag
            self.y_tag = tn.y_tag
            self.site_tag = tn.site_tag
        else:  # 'y'
            # -> rotate 90deg
            self.imin, self.imax = sorted(yrange)
            self.jmin, self.jmax = sorted(xrange)
            self.y_tag = tn.x_tag
            self.x_tag = tn.y_tag
            self.site_tag = lambda i, j: tn.site_tag(j, i)

        if "min" in self.from_which:
            # -> sweeps are increasing
            self.sweep = range(self.imin, self.imax + 1, +stepsize)
            self.istep = +stepsize
        else:  # 'max'
            # -> sweeps are decreasing
            self.sweep = range(self.imax, self.imin - 1, -stepsize)
            self.istep = -stepsize

        self.sweep_other = range(self.jmin, self.jmax + 1)

    def get_opposite_env_fn(self):
        """Get the function and location label for contracting boundaries in
        the opposite direction to main sweep.
        """
        return {
            "xmin": (
                functools.partial(
                    self.tn.compute_xmax_environments, yrange=self.yrange
                ),
                "xmax",
            ),
            "xmax": (
                functools.partial(
                    self.tn.compute_xmin_environments, yrange=self.yrange
                ),
                "xmin",
            ),
            "ymin": (
                functools.partial(
                    self.tn.compute_ymax_environments, xrange=self.xrange
                ),
                "ymax",
            ),
            "ymax": (
                functools.partial(
                    self.tn.compute_ymin_environments, xrange=self.xrange
                ),
                "ymin",
            ),
        }[self.from_which]


BOUNDARY_SEQUENCE_VALID = {
    "xmin",
    "xmax",
    "ymin",
    "ymax",
}
BOUNDARY_SEQUENCE_MAP = {
    "b": "xmin",
    "xmin": "xmin",
    "t": "xmax",
    "xmax": "xmax",
    "l": "ymin",
    "ymin": "ymin",
    "r": "ymax",
    "ymax": "ymax",
}


def parse_boundary_sequence(sequence):
    """Ensure ``sequence`` is a tuple of boundary sequence strings from
    ``{'xmin', 'xmax', 'ymin', 'ymax'}``
    """
    if isinstance(sequence, str):
        if sequence in BOUNDARY_SEQUENCE_VALID:
            return (sequence,)
    return tuple(BOUNDARY_SEQUENCE_MAP[d] for d in sequence)


class TensorNetwork2D(TensorNetworkGen):
    r"""Mixin class for tensor networks with a square lattice two-dimensional
    structure, indexed by ``[{row},{column}]`` so that::

                     'Y{j}'
                        v

        i=Lx-1 ●──●──●──●──●──●──   ──●
               |  |  |  |  |  |       |
                     ...
               |  |  |  |  |  | 'I{i},{j}' = 'I3,5' e.g.
        i=3    ●──●──●──●──●──●──
               |  |  |  |  |  |       |
        i=2    ●──●──●──●──●──●──   ──●    <== 'X{i}'
               |  |  |  |  |  |  ...  |
        i=1    ●──●──●──●──●──●──   ──●
               |  |  |  |  |  |       |
        i=0    ●──●──●──●──●──●──   ──●

             j=0, 1, 2, 3, 4, 5    j=Ly-1

    This implies the following conventions:

        * the 'up' bond is coordinates ``(i, j), (i + 1, j)``
        * the 'down' bond is coordinates ``(i, j), (i - 1, j)``
        * the 'right' bond is coordinates ``(i, j), (i, j + 1)``
        * the 'left' bond is coordinates ``(i, j), (i, j - 1)``

    """

    _NDIMS = 2
    _EXTRA_PROPS = (
        "_site_tag_id",
        "_x_tag_id",
        "_y_tag_id",
        "_Lx",
        "_Ly",
    )

    def _compatible_2d(self, other):
        """Check whether ``self`` and ``other`` are compatible 2D tensor
        networks such that they can remain a 2D tensor network when combined.
        """
        return isinstance(other, TensorNetwork2D) and all(
            getattr(self, e) == getattr(other, e)
            for e in TensorNetwork2D._EXTRA_PROPS
        )

    def combine(self, other, *, virtual=False, check_collisions=True):
        """Combine this tensor network with another, returning a new tensor
        network. If the two are compatible, cast the resulting tensor network
        to a :class:`TensorNetwork2D` instance.

        Parameters
        ----------
        other : TensorNetwork2D or TensorNetwork
            The other tensor network to combine with.
        virtual : bool, optional
            Whether the new tensor network should copy all the incoming tensors
            (``False``, the default), or view them as virtual (``True``).
        check_collisions : bool, optional
            Whether to check for index collisions between the two tensor
            networks before combining them. If ``True`` (the default), any
            inner indices that clash will be mangled.

        Returns
        -------
        TensorNetwork2D or TensorNetwork
        """
        new = super().combine(
            other, virtual=virtual, check_collisions=check_collisions
        )
        if self._compatible_2d(other):
            new.view_as_(TensorNetwork2D, like=self)
        return new

    @property
    def Lx(self):
        """The number of rows."""
        return self._Lx

    @property
    def Ly(self):
        """The number of columns."""
        return self._Ly

    @property
    def nsites(self):
        """The total number of sites."""
        return self._Lx * self._Ly

    def site_tag(self, i, j=None):
        """The name of the tag specifiying the tensor at site ``(i, j)``."""
        if j is None:
            i, j = i
        if not isinstance(i, str):
            i = i % self.Lx
        if not isinstance(j, str):
            j = j % self.Ly
        return self.site_tag_id.format(i, j)

    @property
    def x_tag_id(self):
        """The string specifier for tagging each row of this 2D TN."""
        return self._x_tag_id

    def x_tag(self, i):
        if not isinstance(i, str):
            i = i % self.Lx
        return self.x_tag_id.format(i)

    @property
    def x_tags(self):
        """A tuple of all of the ``Lx`` different row tags."""
        return tuple(map(self.x_tag, range(self.Lx)))

    row_tag = deprecated(x_tag, "row_tag", "x_tag")
    row_tags = deprecated(x_tags, "row_tags", "x_tags")

    @property
    def y_tag_id(self):
        """The string specifier for tagging each column of this 2D TN."""
        return self._y_tag_id

    def y_tag(self, j):
        if not isinstance(j, str):
            j = j % self.Ly
        return self.y_tag_id.format(j)

    @property
    def y_tags(self):
        """A tuple of all of the ``Ly`` different column tags."""
        return tuple(map(self.y_tag, range(self.Ly)))

    col_tag = deprecated(y_tag, "col_tag", "y_tag")
    col_tags = deprecated(y_tags, "col_tags", "y_tags")

    def maybe_convert_coo(self, x):
        """Check if ``x`` is a tuple of two ints and convert to the
        corresponding site tag if so.
        """
        if not isinstance(x, str):
            try:
                i, j = map(int, x)
                return self.site_tag(i, j)
            except (ValueError, TypeError):
                pass
        return x

    def _get_tids_from_tags(self, tags, which="all"):
        """This is the function that lets coordinates such as ``(i, j)`` be
        used for many 'tag' based functions.
        """
        tags = self.maybe_convert_coo(tags)
        return super()._get_tids_from_tags(tags, which=which)

    def gen_site_coos(self):
        """Generate coordinates for all the sites in this 2D TN."""
        return product(range(self.Lx), range(self.Ly))

    def gen_bond_coos(self):
        """Generate pairs of coordinates for all the bonds in this 2D TN."""
        return gen_2d_bonds(
            self.Lx,
            self.Ly,
            steppers=[lambda i, j: (i, j + 1), lambda i, j: (i + 1, j)],
        )

    def gen_horizontal_bond_coos(self):
        """Generate all coordinate pairs like ``(i, j), (i, j + 1)``."""
        return gen_2d_bonds(
            self.Lx,
            self.Ly,
            steppers=[
                lambda i, j: (i, j + 1),
            ],
        )

    def gen_horizontal_even_bond_coos(self):
        """Generate all coordinate pairs like ``(i, j), (i, j + 1)`` where
        ``j`` is even, which thus don't overlap at all.
        """
        return gen_2d_bonds(
            self.Lx,
            self.Ly,
            steppers=[
                lambda i, j: (i, j + 1),
            ],
            coo_filter=lambda i, j: j % 2 == 0,
        )

    def gen_horizontal_odd_bond_coos(self):
        """Generate all coordinate pairs like ``(i, j), (i, j + 1)`` where
        ``j`` is odd, which thus don't overlap at all.
        """
        return gen_2d_bonds(
            self.Lx,
            self.Ly,
            steppers=[
                lambda i, j: (i, j + 1),
            ],
            coo_filter=lambda i, j: j % 2 == 1,
        )

    def gen_vertical_bond_coos(self):
        """Generate all coordinate pairs like ``(i, j), (i + 1, j)``."""
        return gen_2d_bonds(
            self.Lx,
            self.Ly,
            steppers=[
                lambda i, j: (i + 1, j),
            ],
        )

    def gen_vertical_even_bond_coos(self):
        """Generate all coordinate pairs like ``(i, j), (i + 1, j)`` where
        ``i`` is even, which thus don't overlap at all.
        """
        return gen_2d_bonds(
            self.Lx,
            self.Ly,
            steppers=[
                lambda i, j: (i + 1, j),
            ],
            coo_filter=lambda i, j: i % 2 == 0,
        )

    def gen_vertical_odd_bond_coos(self):
        """Generate all coordinate pairs like ``(i, j), (i + 1, j)`` where
        ``i`` is odd, which thus don't overlap at all.
        """
        return gen_2d_bonds(
            self.Lx,
            self.Ly,
            steppers=[
                lambda i, j: (i + 1, j),
            ],
            coo_filter=lambda i, j: i % 2 == 1,
        )

    def gen_diagonal_left_bond_coos(self):
        """Generate all coordinate pairs like ``(i, j), (i + 1, j - 1)``."""
        return gen_2d_bonds(
            self.Lx,
            self.Ly,
            steppers=[
                lambda i, j: (i + 1, j - 1),
            ],
        )

    def gen_diagonal_left_even_bond_coos(self):
        """Generate all coordinate pairs like ``(i, j), (i + 1, j - 1)`` where
        ``j`` is even, which thus don't overlap at all.
        """
        return gen_2d_bonds(
            self.Lx,
            self.Ly,
            steppers=[
                lambda i, j: (i + 1, j - 1),
            ],
            coo_filter=lambda i, j: j % 2 == 0,
        )

    def gen_diagonal_left_odd_bond_coos(self):
        """Generate all coordinate pairs like ``(i, j), (i + 1, j - 1)`` where
        ``j`` is odd, which thus don't overlap at all.
        """
        return gen_2d_bonds(
            self.Lx,
            self.Ly,
            steppers=[
                lambda i, j: (i + 1, j - 1),
            ],
            coo_filter=lambda i, j: j % 2 == 1,
        )

    def gen_diagonal_right_bond_coos(self):
        """Generate all coordinate pairs like ``(i, j), (i + 1, j + 1)``."""
        return gen_2d_bonds(
            self.Lx,
            self.Ly,
            steppers=[
                lambda i, j: (i + 1, j + 1),
            ],
        )

    def gen_diagonal_right_even_bond_coos(self):
        """Generate all coordinate pairs like ``(i, j), (i + 1, j + 1)`` where
        ``i`` is even, which thus don't overlap at all.
        """
        return gen_2d_bonds(
            self.Lx,
            self.Ly,
            steppers=[
                lambda i, j: (i + 1, j + 1),
            ],
            coo_filter=lambda i, j: i % 2 == 0,
        )

    def gen_diagonal_right_odd_bond_coos(self):
        """Generate all coordinate pairs like ``(i, j), (i + 1, j + 1)`` where
        ``i`` is odd, which thus don't overlap at all.
        """
        return gen_2d_bonds(
            self.Lx,
            self.Ly,
            steppers=[
                lambda i, j: (i + 1, j + 1),
            ],
            coo_filter=lambda i, j: i % 2 == 1,
        )

    def gen_diagonal_bond_coos(self):
        """Generate all next nearest neighbor diagonal coordinate pairs."""
        return gen_2d_bonds(
            self.Lx,
            self.Ly,
            steppers=[
                lambda i, j: (i + 1, j - 1),
                lambda i, j: (i + 1, j + 1),
            ],
        )

    def valid_coo(self, coo, xrange=None, yrange=None):
        """Check whether ``coo`` is in-bounds.

        Parameters
        ----------
        coo : (int, int, int), optional
            The coordinates to check.
        xrange, yrange : (int, int), optional
            The range of allowed values for the x and y coordinates.

        Returns
        -------
        bool
        """
        if xrange is None:
            xrange = (0, self.Lx - 1)
        if yrange is None:
            yrange = (0, self.Ly - 1)
        return all(mn <= u <= mx for u, (mn, mx) in zip(coo, (xrange, yrange)))

    def get_ranges_present(self):
        """Return the range of site coordinates present in this TN.

        Returns
        -------
        xrange, yrange : tuple[tuple[int, int]]
            The minimum and maximum site coordinates present in each direction.
        """
        xmin = ymin = float("inf")
        xmax = ymax = float("-inf")
        for i, j in self.gen_sites_present():
            xmin = min(i, xmin)
            ymin = min(j, ymin)
            xmax = max(i, xmax)
            ymax = max(j, ymax)
        return (xmin, xmax), (ymin, ymax)

    def __getitem__(self, key):
        """Key based tensor selection, checking for integer based shortcut."""
        return super().__getitem__(self.maybe_convert_coo(key))

    def show(self):
        """Print a unicode schematic of this 2D TN and its bond dimensions."""
        show_2d(self)

    def _repr_info(self):
        info = super()._repr_info()
        info["Lx"] = self.Lx
        info["Ly"] = self.Ly
        info["max_bond"] = self.max_bond()
        return info

    def flatten(self, fuse_multibonds=True, inplace=False):
        """Contract all tensors corresponding to each site into one."""
        tn = self if inplace else self.copy()

        for i, j in self.gen_site_coos():
            tn ^= (i, j)

        if fuse_multibonds:
            tn.fuse_multibonds_()

        return tn.view_as_(TensorNetwork2DFlat, like=self)

    flatten_ = functools.partialmethod(flatten, inplace=True)

    def gen_pairs(
        self,
        xrange=None,
        yrange=None,
        xreverse=False,
        yreverse=False,
        coordinate_order="xy",
        xstep=None,
        ystep=None,
        stepping_order="xy",
        step_only=None,
    ):
        """Helper function for generating pairs of cooordinates for all bonds
        within a certain range, optionally specifying an order.

        Parameters
        ----------
        xrange, yrange : (int, int), optional
            The range of allowed values for the x and y coordinates.
        xreverse, yreverse: bool, optional
            Whether to reverse the order of the x and y sweeps.
        coordinate_order : str, optional
            The order in which to sweep the x and y coordinates. Earlier
            dimensions will change slower. If the corresponding range has
            size 1 then that dimension doesn't need to be specified.
        xstep, ystep : int, optional
            When generating a bond, step in this direction to yield the
            neighboring coordinate. By default, these follow ``xreverse`` and
            ``yreverse`` respectively.
        stepping_order : str, optional
            The order in which to step the x and y coordinates to generate
            bonds. Does not need to include all dimensions.
        step_only : int, optional
            Only perform the ith steps in ``stepping_order``, used to
            interleave canonizing and compressing for example.

        Yields
        ------
        coo_a, coo_b : ((int, int), (int, int))
        """
        if xrange is None:
            xrange = (0, self.Lx - 1)
        if yrange is None:
            yrange = (0, self.Ly - 1)

        # generate the sites and order we will visit them in
        sweeps = {
            "x": (
                range(min(xrange), max(xrange) + 1, +1)
                if not xreverse
                else range(max(xrange), min(xrange) - 1, -1)
            ),
            "y": (
                range(min(yrange), max(yrange) + 1, +1)
                if not yreverse
                else range(max(yrange), min(yrange) - 1, -1)
            ),
        }

        # for convenience, allow subselecting part of stepping_order only
        if step_only is not None:
            stepping_order = stepping_order[step_only]

        # at each step generate the bonds
        if xstep is None:
            xstep = -1 if xreverse else +1
        if ystep is None:
            ystep = -1 if yreverse else +1
        steps = {
            "x": lambda i, j: (i + xstep, j),
            "y": lambda i, j: (i, j + ystep),
        }

        # make sure all coordinates exist - only allow them not to be specified
        # if their range is a unit slice
        for w in "xy":
            if w not in coordinate_order:
                if len(sweeps[w]) > 1:
                    raise ValueError(
                        f"{w} not in coordinate_order and is not size 1."
                    )
                else:
                    # just append -> it won't change order as coord is constant
                    coordinate_order += w
        xi, yi = map(coordinate_order.index, "xy")

        # generate the pairs
        for perm_coo_a in product(*(sweeps[xy] for xy in coordinate_order)):
            coo_a = perm_coo_a[xi], perm_coo_a[yi]
            for xy in stepping_order:
                coo_b = steps[xy](*coo_a)
                # filter out bonds which are out of bounds
                if self.valid_coo(coo_b, xrange, yrange):
                    yield coo_a, coo_b

    def canonize_plane(
        self,
        xrange,
        yrange,
        equalize_norms=False,
        canonize_opts=None,
        **gen_pair_opts,
    ):
        """Canonize every pair of tensors within a subrange, optionally
        specifying a order to visit those pairs in.
        """
        canonize_opts = ensure_dict(canonize_opts)
        canonize_opts.setdefault("equalize_norms", equalize_norms)

        pairs = self.gen_pairs(xrange=xrange, yrange=yrange, **gen_pair_opts)

        for coo_a, coo_b in pairs:
            tag_a = self.site_tag(*coo_a)
            tag_b = self.site_tag(*coo_b)

            # make sure single tensor at each site, skip if none
            try:
                num_a = len(self.tag_map[tag_a])
                if num_a > 1:
                    self ^= tag_a
            except KeyError:
                continue
            try:
                num_b = len(self.tag_map[tag_b])
                if num_b > 1:
                    self ^= tag_b
            except KeyError:
                continue

            self.canonize_between(tag_a, tag_b, **canonize_opts)

    def canonize_row(self, i, sweep, yrange=None, **canonize_opts):
        r"""Canonize all or part of a row.

        If ``sweep == 'right'`` then::

             |  |  |  |  |  |  |         |  |  |  |  |  |  |
            ─●──●──●──●──●──●──●─       ─●──●──●──●──●──●──●─
             |  |  |  |  |  |  |         |  |  |  |  |  |  |
            ─●──●──●──●──●──●──●─  ==>  ─●──>──>──>──>──o──●─ row=i
             |  |  |  |  |  |  |         |  |  |  |  |  |  |
            ─●──●──●──●──●──●──●─       ─●──●──●──●──●──●──●─
             |  |  |  |  |  |  |         |  |  |  |  |  |  |
                .           .               .           .
                jstart      jstop           jstart      jstop

        If ``sweep == 'left'`` then::

             |  |  |  |  |  |  |         |  |  |  |  |  |  |
            ─●──●──●──●──●──●──●─       ─●──●──●──●──●──●──●─
             |  |  |  |  |  |  |         |  |  |  |  |  |  |
            ─●──●──●──●──●──●──●─  ==>  ─●──o──<──<──<──<──●─ row=i
             |  |  |  |  |  |  |         |  |  |  |  |  |  |
            ─●──●──●──●──●──●──●─       ─●──●──●──●──●──●──●─
             |  |  |  |  |  |  |         |  |  |  |  |  |  |
                .           .               .           .
                jstop       jstart          jstop       jstart

        Does not yield an orthogonal form in the same way as in 1D.

        Parameters
        ----------
        i : int
            Which row to canonize.
        sweep : {'right', 'left'}
            Which direction to sweep in.
        jstart : int or None
            Starting column, defaults to whole row.
        jstop : int or None
            Stopping column, defaults to whole row.
        canonize_opts
            Supplied to ``canonize_between``.
        """
        check_opt("sweep", sweep, ("right", "left"))
        self.canonize_plane(
            xrange=(i, i),
            yrange=yrange,
            yreverse=(sweep == "left"),
            **canonize_opts,
        )

    def canonize_column(self, j, sweep, xrange=None, **canonize_opts):
        r"""Canonize all or part of a column.

        If ``sweep='up'`` then::

             |  |  |         |  |  |
            ─●──●──●─       ─●──●──●─
             |  |  |         |  |  |
            ─●──●──●─       ─●──o──●─ istop
             |  |  |   ==>   |  |  |
            ─●──●──●─       ─●──^──●─
             |  |  |         |  |  |
            ─●──●──●─       ─●──^──●─ istart
             |  |  |         |  |  |
            ─●──●──●─       ─●──●──●─
             |  |  |         |  |  |
                .               .
                j               j

        If ``sweep='down'`` then::

             |  |  |         |  |  |
            ─●──●──●─       ─●──●──●─
             |  |  |         |  |  |
            ─●──●──●─       ─●──v──●─ istart
             |  |  |   ==>   |  |  |
            ─●──●──●─       ─●──v──●─
             |  |  |         |  |  |
            ─●──●──●─       ─●──o──●─ istop
             |  |  |         |  |  |
            ─●──●──●─       ─●──●──●─
             |  |  |         |  |  |
                .               .
                j               j

        Does not yield an orthogonal form in the same way as in 1D.

        Parameters
        ----------
        j : int
            Which column to canonize.
        sweep : {'up', 'down'}
            Which direction to sweep in.
        xrange : None or (int, int), optional
            The range of columns to canonize.
        canonize_opts
            Supplied to ``canonize_between``.
        """
        check_opt("sweep", sweep, ("up", "down"))
        self.canonize_plane(
            yrange=(j, j),
            xrange=xrange,
            xreverse=(sweep == "down"),
            **canonize_opts,
        )

    def canonize_row_around(self, i, around=(0, 1)):
        # sweep to the right
        self.canonize_row(i, sweep="right", yrange=(0, min(around)))
        # sweep to the left
        self.canonize_row(i, sweep="left", yrange=(max(around), self.Ly - 1))

    def compress_plane(
        self,
        xrange,
        yrange,
        max_bond=None,
        cutoff=1e-10,
        equalize_norms=False,
        compress_opts=None,
        **gen_pair_opts,
    ):
        """Compress every pair of tensors within a subrange, optionally
        specifying a order to visit those pairs in.
        """
        compress_opts = ensure_dict(compress_opts)
        compress_opts.setdefault("absorb", "right")
        compress_opts.setdefault("equalize_norms", equalize_norms)

        pairs = self.gen_pairs(
            xrange=xrange,
            yrange=yrange,
            **gen_pair_opts,
        )

        for coo_a, coo_b in pairs:
            tag_a = self.site_tag(*coo_a)
            tag_b = self.site_tag(*coo_b)

            # make sure single tensor at each site, skip if none
            try:
                num_a = len(self.tag_map[tag_a])
                if num_a > 1:
                    self ^= tag_a
            except KeyError:
                continue
            try:
                num_b = len(self.tag_map[tag_b])
                if num_b > 1:
                    self ^= tag_b
            except KeyError:
                continue

            self.compress_between(
                tag_a, tag_b, max_bond=max_bond, cutoff=cutoff, **compress_opts
            )

    def compress_row(
        self,
        i,
        sweep,
        yrange=None,
        max_bond=None,
        cutoff=1e-10,
        equalize_norms=False,
        compress_opts=None,
    ):
        r"""Compress all or part of a row.

        If ``sweep == 'right'`` then::

             |  |  |  |  |  |  |         |  |  |  |  |  |  |
            ━●━━●━━●━━●━━●━━●━━●━       ━●━━●━━●━━●━━●━━●━━●━
             |  |  |  |  |  |  |         |  |  |  |  |  |  |
            ━●━━●━━●━━●━━●━━●━━●━  ━━>  ━●━━>──>──>──>──o━━●━ row=i
             |  |  |  |  |  |  |         |  |  |  |  |  |  |
            ━●━━●━━●━━●━━●━━●━━●━       ━●━━●━━●━━●━━●━━●━━●━
             |  |  |  |  |  |  |         |  |  |  |  |  |  |
                .           .               .           .
                jstart      jstop           jstart      jstop

        If ``sweep == 'left'`` then::

             |  |  |  |  |  |  |         |  |  |  |  |  |  |
            ━●━━●━━●━━●━━●━━●━━●━       ━●━━●━━●━━●━━●━━●━━●━
             |  |  |  |  |  |  |         |  |  |  |  |  |  |
            ━●━━●━━●━━●━━●━━●━━●━  ━━>  ━●━━o──<──<──<──<━━●━ row=i
             |  |  |  |  |  |  |         |  |  |  |  |  |  |
            ━●━━●━━●━━●━━●━━●━━●━       ━●━━●━━●━━●━━●━━●━━●━
             |  |  |  |  |  |  |         |  |  |  |  |  |  |
                .           .               .           .
                jstop       jstart          jstop       jstart

        Does not yield an orthogonal form in the same way as in 1D.

        Parameters
        ----------
        i : int
            Which row to compress.
        sweep : {'right', 'left'}
            Which direction to sweep in.
        yrange : tuple[int, int] or None
            The range of columns to compress.
        max_bond : int, optional
            The maximum boundary dimension, AKA 'chi'. The default of ``None``
            means truncation is left purely to ``cutoff`` and is not
            recommended in 2D.
        cutoff : float, optional
            Cut-off value to used to truncate singular values in the boundary
            contraction.
        compress_opts : None or dict, optional
            Supplied to :meth:`compress_between`.
        """
        check_opt("sweep", sweep, ("right", "left"))
        self.compress_plane(
            xrange=(i, i),
            yrange=yrange,
            yreverse=(sweep == "left"),
            max_bond=max_bond,
            cutoff=cutoff,
            equalize_norms=equalize_norms,
            compress_opts=compress_opts,
        )

    def compress_column(
        self,
        j,
        sweep,
        xrange=None,
        max_bond=None,
        cutoff=1e-10,
        equalize_norms=False,
        compress_opts=None,
    ):
        r"""Compress all or part of a column.

        If ``sweep='up'`` then::

             ┃  ┃  ┃         ┃  ┃  ┃
            ─●──●──●─       ─●──●──●─
             ┃  ┃  ┃         ┃  ┃  ┃
            ─●──●──●─       ─●──o──●─  .
             ┃  ┃  ┃   ==>   ┃  |  ┃   .
            ─●──●──●─       ─●──^──●─  . xrange
             ┃  ┃  ┃         ┃  |  ┃   .
            ─●──●──●─       ─●──^──●─  .
             ┃  ┃  ┃         ┃  ┃  ┃
            ─●──●──●─       ─●──●──●─
             ┃  ┃  ┃         ┃  ┃  ┃
                .               .
                j               j

        If ``sweep='down'`` then::

             ┃  ┃  ┃         ┃  ┃  ┃
            ─●──●──●─       ─●──●──●─
             ┃  ┃  ┃         ┃  ┃  ┃
            ─●──●──●─       ─●──v──●─ .
             ┃  ┃  ┃   ==>   ┃  |  ┃  .
            ─●──●──●─       ─●──v──●─ . xrange
             ┃  ┃  ┃         ┃  |  ┃  .
            ─●──●──●─       ─●──o──●─ .
             ┃  ┃  ┃         ┃  ┃  ┃
            ─●──●──●─       ─●──●──●─
             ┃  ┃  ┃         ┃  ┃  ┃
                .               .
                j               j

        Does not yield an orthogonal form in the same way as in 1D.

        Parameters
        ----------
        j : int
            Which column to compress.
        sweep : {'up', 'down'}
            Which direction to sweep in.
        xrange : None or (int, int), optional
            The range of rows to compress.
        max_bond : int, optional
            The maximum boundary dimension, AKA 'chi'. The default of ``None``
            means truncation is left purely to ``cutoff`` and is not
            recommended in 2D.
        cutoff : float, optional
            Cut-off value to used to truncate singular values in the boundary
            contraction.
        compress_opts : None or dict, optional
            Supplied to :meth:`compress_between`.
        """
        check_opt("sweep", sweep, ("up", "down"))
        self.compress_plane(
            yrange=(j, j),
            xrange=xrange,
            xreverse=(sweep == "down"),
            max_bond=max_bond,
            cutoff=cutoff,
            equalize_norms=equalize_norms,
            compress_opts=compress_opts,
        )

    def _contract_boundary_core(
        self,
        xrange,
        yrange,
        from_which,
        max_bond,
        cutoff=1e-10,
        canonize=True,
        layer_tags=None,
        compress_late=True,
        sweep_reverse=False,
        equalize_norms=False,
        compress_opts=None,
        canonize_opts=None,
    ):
        canonize_opts = ensure_dict(canonize_opts)
        canonize_opts.setdefault("absorb", "right")
        compress_opts = ensure_dict(compress_opts)
        compress_opts.setdefault("absorb", "right")

        r2d = Rotator2D(self, xrange, yrange, from_which)
        site_tag = r2d.site_tag
        plane, istep = r2d.plane, r2d.istep

        if layer_tags is None:
            layer_tags = [None]

        for i in r2d.sweep[:-1]:
            for layer_tag in layer_tags:
                for j in r2d.sweep_other:
                    tag1 = site_tag(i, j)  # outer
                    tag2 = site_tag(i + istep, j)  # inner

                    if (tag1 not in self.tag_map) or (
                        tag2 not in self.tag_map
                    ):
                        # allow completely missing sites
                        continue

                    if (layer_tag is None) or len(self.tag_map[tag2]) == 1:
                        # contract *any* tensors with pair of coordinates
                        #
                        #     │  │  │  │  │
                        #     O──O──O──O──O  i+1  │  │  │  │  │
                        #     │  │  │  │  │  -->  O══O══O══O══O
                        #     O──O──O──O──O  i
                        #
                        self.contract_((tag1, tag2), which="any")
                    else:
                        # make sure the exterior sites are a single tensor
                        #
                        #    │ ││ ││ ││ ││ │       │ ││ ││ ││ ││ │   (2 layers)
                        #    A─BA─BA─BA─BA─B       A─BA─BA─BA─BA─B
                        #    │ ││ ││ ││ ││ │  ==>   ╲│ ╲│ ╲│ ╲│ ╲│
                        #    A─BA─BA─BA─BA─B         C══C══C══C══C
                        #
                        if len(self.tag_map[tag1]) > 1:
                            self ^= tag1

                        # contract interior sites from layer ``tag``
                        #
                        #    │ ││ ││ ││ ││ │  (1st contraction if 2 layer tags)
                        #    │ B┼─B┼─B┼─B┼─B
                        #    │╱ │╱ │╱ │╱ │╱
                        #    O══<══<══<══<
                        #
                        self.contract_between(
                            tag1,
                            (tag2, layer_tag),
                            equalize_norms=equalize_norms,
                        )

                        # drop inner site tag merged into outer boundary so
                        # we can still uniquely identify inner tensors
                        if layer_tag != layer_tags[-1]:
                            self[tag1].drop_tags(tag2)

                    if not compress_late:
                        # we immediately compress bonds to all neighboring
                        # tensors, prioritizing memory efficiency
                        #
                        #     │  │  │  │  │
                        #     O══O──O──O──O
                        #      ^  ╲ │  │  │
                        # compress  O──O──O
                        #
                        (tid1,) = self.tag_map[tag1]
                        for tidn in self._get_neighbor_tids(tid1):
                            t1, tn = self._tids_get(tid1, tidn)
                            if bonds_size(t1, tn) > max_bond:
                                self._compress_between_tids(
                                    tidn,
                                    tid1,
                                    max_bond=max_bond,
                                    cutoff=cutoff,
                                    equalize_norms=equalize_norms,
                                    **compress_opts,
                                )

                if compress_late:
                    # we don't compress until the full line of contractions has
                    # been done, prioritizing gauging
                    if canonize:
                        #
                        #     │  │  │  │  │
                        #     O══O══<══<══<
                        #
                        self.canonize_plane(
                            xrange=xrange if plane != "x" else (i, i),
                            xreverse=not sweep_reverse,
                            yrange=yrange if plane != "y" else (i, i),
                            yreverse=not sweep_reverse,
                            equalize_norms=equalize_norms,
                            canonize_opts=canonize_opts,
                        )
                    #
                    #    │  │  │  │  │  -->  │  │  │  │  │  -->  │  │  │  │  │
                    #    >──O══O══O══O  -->  >──>──O══O══O  -->  >──>──>──O══O
                    #    .  .           -->     .  .        -->        .  .
                    #
                    self.compress_plane(
                        xrange=xrange if plane != "x" else (i, i),
                        xreverse=sweep_reverse,
                        yrange=yrange if plane != "y" else (i, i),
                        yreverse=sweep_reverse,
                        max_bond=max_bond,
                        cutoff=cutoff,
                        equalize_norms=equalize_norms,
                        compress_opts=compress_opts,
                    )

    def _contract_boundary_full_bond(
        self,
        xrange,
        yrange,
        from_which,
        max_bond,
        cutoff=0.0,
        method="eigh",
        renorm=False,
        optimize="auto-hq",
        opposite_envs=None,
        equalize_norms=False,
        contract_boundary_opts=None,
    ):
        """Contract the boundary of this 2D TN using the 'full bond'
        environment information obtained from a boundary contraction in the
        opposite direction.

        Parameters
        ----------
        xrange : (int, int) or None, optional
            The range of rows to contract and compress.
        yrange : (int, int)
            The range of columns to contract and compress.
        from_which : {'xmin', 'ymin', 'xmax', 'ymax'}
            Which direction to contract the rectangular patch from.
        max_bond : int
            The maximum boundary dimension, AKA 'chi'. By default used for the
            opposite direction environment contraction as well.
        cutoff : float, optional
            Cut-off value to used to truncate singular values in the boundary
            contraction - only for the opposite direction environment
            contraction.
        method : {'eigh', 'eig', 'svd', 'biorthog'}, optional
            Which similarity decomposition method to use to compress the full
            bond environment.
        renorm : bool, optional
            Whether to renormalize the isometric projection or not.
        optimize : str or PathOptimize, optimize
            Contraction optimizer to use for the exact contractions.
        opposite_envs : dict, optional
            If supplied, the opposite environments will be fetched or lazily
            computed into this dict depending on whether they are missing.
        contract_boundary_opts
            Other options given to the opposite direction environment
            contraction.
        """
        if equalize_norms:
            raise NotImplementedError

        contract_boundary_opts = ensure_dict(contract_boundary_opts)
        contract_boundary_opts.setdefault("max_bond", max_bond)
        contract_boundary_opts.setdefault("cutoff", cutoff)

        # rotate coordinates and sweeps rather than actual TN
        r2d = Rotator2D(self, xrange, yrange, from_which)
        jmin, jmax, istep = r2d.jmin, r2d.jmax, r2d.istep
        y_tag, x_tag, site_tag = r2d.y_tag, r2d.x_tag, r2d.site_tag
        opposite_env_fn, env_location = r2d.get_opposite_env_fn()

        if opposite_envs is None:
            # storage for the top down environments - compute lazily so that a
            #     dict can be supplied *with or without* them precomputed
            opposite_envs = {}

        # now contract in the other direction
        for i in r2d.sweep[:-1]:
            # contract inwards, no compression
            for j in r2d.sweep_other:
                #
                #             j  j+1   ...
                #         │   │   │   │   │   │
                #        =●===●===●───●───●───●─         i + 1
                #    ...        \ │   │   │   │    ...
                #          ->     ●━━━●━━━●━━━●━         i
                #
                self.contract_(
                    [site_tag(i, j), site_tag(i + istep, j)], which="any"
                )

            # form strip of current row and approx top environment
            #     the canonicalization 'compresses' outer bonds
            #
            #     ●━━━●━━━●━━━●━━━●━━━●  i + 2
            #     │   │   │   │   │   │
            #     >--->===●===<===<---<  i + 1
            #       (jmax - jmin) // 2
            #
            row = self.select(x_tag(i))
            row.canonize_around_(y_tag((jmax - jmin) // 2))

            try:
                env = opposite_envs[env_location, i + istep]
            except KeyError:
                # lazy computation of top environements (computes all at once)
                #
                #        ●━━━●━━━●━━━●━━━●━━━●━         i + 1
                #     │  │   │   │   │   │   │    ...
                #     v  ●───●───●───●───●───●─           i
                #        │   │   │   │   │   │
                #                 ...
                #
                opposite_envs.update(opposite_env_fn(**contract_boundary_opts))
                env = opposite_envs[env_location, i + istep]

            ladder = row & env

            # for each pair to compress, form left and right envs from strip
            #
            #            ╭─●━━━●─╮
            #   lenvs[j] ● │   │ ● renvs[j + 1]
            #            ╰─●===●─╯
            #              j  j+1
            #
            lenvs = {jmin + 1: ladder.select(y_tag(jmin))}
            for j in range(jmin + 2, jmax):
                lenvs[j] = ladder.select(y_tag(j - 1)) @ lenvs[j - 1]

            renvs = {jmax - 1: ladder.select(y_tag(jmax))}
            for j in range(jmax - 2, jmin, -1):
                renvs[j] = ladder.select(y_tag(j + 1)) @ renvs[j + 1]

            for j in range(jmin, jmax):
                if (
                    bonds_size(self[site_tag(i, j)], self[site_tag(i, j + 1)])
                    <= max_bond
                ):
                    # no need to form env operator and compress
                    continue

                # for each compression pair make single loop - the bond env
                #
                #       j  j+1
                #     ╭─●━━━●─╮
                #     ● │   │ ●
                #     ╰─●   ●─╯
                #   lcut│   │rcut
                #
                tn_be = TensorNetwork([])
                if j in lenvs:
                    tn_be &= lenvs[j]
                tn_be &= ladder.select_any([y_tag(j), y_tag(j + 1)])
                if j + 1 in renvs:
                    tn_be &= renvs[j + 1]

                lcut = rand_uuid()
                rcut = rand_uuid()
                tn_be.cut_between(
                    site_tag(i, j),
                    site_tag(i, j + 1),
                    left_ind=lcut,
                    right_ind=rcut,
                )

                # form dense environment and find symmetric compressors
                E = tn_be.to_dense([rcut], [lcut], optimize=optimize)

                Cl, Cr = decomp.similarity_compress(
                    E, max_bond, method=method, renorm=renorm
                )

                # insert compressors back in base TN
                #
                #      j       j+1
                #     ━●━━━━━━━━●━ i+1
                #      │        │
                #     =●=Cl──Cr=●= i
                #       <--  -->
                #
                self.insert_gauge(
                    Cr, [site_tag(i, j)], [site_tag(i, j + 1)], Cl
                )

    def _contract_boundary_projector(
        self,
        xrange,
        yrange,
        from_which,
        max_bond=None,
        cutoff=1e-10,
        lazy=False,
        equalize_norms=False,
        optimize="auto-hq",
        compress_opts=None,
    ):
        """Contract the boundary of this 2D tensor network by explicitly
        computing and inserting explicit local projector tensors, which can
        optionally be left uncontracted. Multilayer networks are naturally
        supported.

        Parameters
        ----------
        xrange : tuple
            The range of x indices to contract.
        yrange : tuple
            The range of y indices to contract.
        from_which : {'xmin', 'xmax', 'ymin', 'ymax'}
            From which boundary to contract.
        max_bond : int, optional
            The maximum bond dimension to contract to. If ``None`` (default),
            compression is left to ``cutoff``.
        cutoff : float, optional
            The cutoff to use for boundary compression.
        lazy : bool, optional
            Whether to leave the boundary tensors uncontracted. If ``False``
            (the default), the boundary tensors are contracted and the
            resulting boundary has a single tensor per site.
        equalize_norms : bool, optional
            Whether to actively absorb the norm of modified tensors into
            ``self.exponent``.
        optimize : str or PathOptimizer, optional
            The contract path optimization to use when forming the projector
            tensors.
        compress_opts : dict, optional
            Other options to pass to
            :func:`~quimb.tensor.decomp.svd_truncated`.

        See Also
        --------
        TensorNetwork.insert_compressor_between_regions
        """
        compress_opts = ensure_dict(compress_opts)

        r = Rotator2D(self, xrange, yrange, from_which)
        j0 = r.sweep_other[0]

        for i0, i1 in pairwise(r.sweep):
            # we compute the projectors from an untouched copy
            tn_calc = self.copy()

            for j in r.sweep_other:
                tag_ij = r.site_tag(i0, j)
                tag_ip1j = r.site_tag(i1, j)

                if j != j0:
                    ltags = r.site_tag(i0, j - 1), r.site_tag(i1, j - 1)
                    rtags = (tag_ij, tag_ip1j)
                    #      │         │
                    #    ──O─┐ chi ┌─O──  i+1
                    #      │ └─▷═◁─┘ │
                    #      │ ┌┘   └┐ │
                    #    ──O─┘     └─O──  i
                    #     j-1        j
                    tn_calc.insert_compressor_between_regions(
                        ltags,
                        rtags,
                        new_ltags=ltags,
                        new_rtags=rtags,
                        insert_into=self,
                        max_bond=max_bond,
                        cutoff=cutoff,
                        **compress_opts,
                    )

            if not lazy:
                # contract each pair of boundary tensors with their projectors
                for j in r.sweep_other:
                    self.contract_tags_(
                        (r.site_tag(i0, j), r.site_tag(i1, j)),
                        optimize=optimize,
                    )

            if equalize_norms:
                for t in self.select_tensors(r.x_tag(i1)):
                    self.strip_exponent(t, equalize_norms)

    def contract_boundary_from(
        self,
        xrange,
        yrange,
        from_which,
        max_bond=None,
        *,
        cutoff=1e-10,
        canonize=True,
        mode="mps",
        layer_tags=None,
        sweep_reverse=False,
        compress_opts=None,
        inplace=False,
        **contract_boundary_opts,
    ):
        """Unified entrypoint for contracting any rectangular patch of tensors
        from any direction, with any boundary method.
        """
        check_opt("mode", mode, ("mps", "projector", "full-bond"))

        tn = self if inplace else self.copy()

        # universal options
        contract_boundary_opts["xrange"] = xrange
        contract_boundary_opts["yrange"] = yrange
        contract_boundary_opts["from_which"] = from_which
        contract_boundary_opts["max_bond"] = max_bond

        if mode == "full-bond":
            tn._contract_boundary_full_bond(**contract_boundary_opts)
            return tn

        contract_boundary_opts["cutoff"] = cutoff
        contract_boundary_opts["compress_opts"] = compress_opts

        if mode == "projector":
            tn._contract_boundary_projector(**contract_boundary_opts)
            return tn

        # mode == 'mps' options
        contract_boundary_opts["canonize"] = canonize
        contract_boundary_opts["layer_tags"] = layer_tags
        contract_boundary_opts["sweep_reverse"] = sweep_reverse
        tn._contract_boundary_core(**contract_boundary_opts)

        return tn

    contract_boundary_from_ = functools.partialmethod(
        contract_boundary_from, inplace=True
    )

    def contract_boundary_from_xmin(
        self,
        xrange,
        yrange=None,
        max_bond=None,
        *,
        cutoff=1e-10,
        canonize=True,
        mode="mps",
        layer_tags=None,
        sweep_reverse=False,
        compress_opts=None,
        inplace=False,
        **contract_boundary_opts,
    ):
        r"""Contract a 2D tensor network inwards from the bottom, canonizing
        and compressing (left to right) along the way. If
        ``layer_tags is None`` this looks like::

            a) contract

            │  │  │  │  │
            ●──●──●──●──●       │  │  │  │  │
            │  │  │  │  │  -->  ●══●══●══●══●
            ●──●──●──●──●

            b) optionally canonicalize

            │  │  │  │  │
            ●══●══<══<══<

            c) compress in opposite direction

            │  │  │  │  │  -->  │  │  │  │  │  -->  │  │  │  │  │
            >──●══●══●══●  -->  >──>──●══●══●  -->  >──>──>──●══●
            .  .           -->     .  .        -->        .  .

        If ``layer_tags`` is specified, each then each layer is contracted in
        and compressed separately, resulting generally in a lower memory
        scaling. For two layer tags this looks like::

            a) first flatten the outer boundary only

            │ ││ ││ ││ ││ │       │ ││ ││ ││ ││ │
            ●─○●─○●─○●─○●─○       ●─○●─○●─○●─○●─○
            │ ││ ││ ││ ││ │  ==>   ╲│ ╲│ ╲│ ╲│ ╲│
            ●─○●─○●─○●─○●─○         ●══●══●══●══●

            b) contract and compress a single layer only

            │ ││ ││ ││ ││ │
            │ ○──○──○──○──○
            │╱ │╱ │╱ │╱ │╱
            ●══<══<══<══<

            c) contract and compress the next layer

            ╲│ ╲│ ╲│ ╲│ ╲│
             >══>══>══>══●

        Parameters
        ----------
        xrange : (int, int)
            The range of rows to compress (inclusive).
        yrange : (int, int) or None, optional
            The range of columns to compress (inclusive), sweeping along with
            canonization and compression. Defaults to all columns.
        max_bond : int, optional
            The maximum boundary dimension, AKA 'chi'. The default of ``None``
            means truncation is left purely to ``cutoff`` and is not
            recommended in 2D.
        cutoff : float, optional
            Cut-off value to used to truncate singular values in the boundary
            contraction.
        canonize : bool, optional
            Whether to sweep one way with canonization before compressing.
        mode : {'mps', 'full-bond'}, optional
            How to perform the compression on the boundary.
        layer_tags : None or sequence[str], optional
            If ``None``, all tensors at each coordinate pair
            ``[(i, j), (i + 1, j)]`` will be first contracted. If specified,
            then the outer tensor at ``(i, j)`` will be contracted with the
            tensor specified by ``[(i + 1, j), layer_tag]``, for each
            ``layer_tag`` in ``layer_tags``.
        sweep_reverse : bool, optional
            Which way to perform the compression sweep, which has an effect on
            which tensors end up being canonized. Setting this to true sweeps
            the compression from largest to smallest coordinates.
        compress_opts : None or dict, optional
            Supplied to :meth:`compress_between`.
        inplace : bool, optional
            Whether to perform the contraction inplace or not.

        See Also
        --------
        contract_boundary_from_xmax, contract_boundary_from_ymin,
        contract_boundary_from_ymax
        """
        return self.contract_boundary_from(
            xrange=xrange,
            yrange=yrange,
            from_which="xmin",
            max_bond=max_bond,
            cutoff=cutoff,
            canonize=canonize,
            mode=mode,
            layer_tags=layer_tags,
            sweep_reverse=sweep_reverse,
            compress_opts=compress_opts,
            inplace=inplace,
            **contract_boundary_opts,
        )

    contract_boundary_from_xmin_ = functools.partialmethod(
        contract_boundary_from_xmin, inplace=True
    )

    def contract_boundary_from_xmax(
        self,
        xrange,
        yrange=None,
        max_bond=None,
        *,
        cutoff=1e-10,
        canonize=True,
        mode="mps",
        layer_tags=None,
        inplace=False,
        sweep_reverse=False,
        compress_opts=None,
        **contract_boundary_opts,
    ):
        r"""Contract a 2D tensor network inwards from the top, canonizing and
        compressing (right to left) along the way. If
        ``layer_tags is None`` this looks like::

            a) contract

            ●──●──●──●──●
            |  |  |  |  |  -->  ●══●══●══●══●
            ●──●──●──●──●       |  |  |  |  |
            |  |  |  |  |

            b) optionally canonicalize

            ●══●══<══<══<
            |  |  |  |  |

            c) compress in opposite direction

            >──●══●══●══●  -->  >──>──●══●══●  -->  >──>──>──●══●
            |  |  |  |  |  -->  |  |  |  |  |  -->  |  |  |  |  |
            .  .           -->     .  .        -->        .  .

        If ``layer_tags`` is specified, each then each layer is contracted in
        and compressed separately, resulting generally in a lower memory
        scaling. For two layer tags this looks like::

            a) first flatten the outer boundary only

            ●─○●─○●─○●─○●─○         ●══●══●══●══●
            │ ││ ││ ││ ││ │  ==>   ╱│ ╱│ ╱│ ╱│ ╱│
            ●─○●─○●─○●─○●─○       ●─○●─○●─○●─○●─○
            │ ││ ││ ││ ││ │       │ ││ ││ ││ ││ │

            b) contract and compress a single layer only

            ●══<══<══<══<
            │╲ │╲ │╲ │╲ │╲
            │ ○──○──○──○──○
            │ ││ ││ ││ ││ │

            c) contract and compress the next layer

             ●══●══●══●══●
            ╱│ ╱│ ╱│ ╱│ ╱│

        Parameters
        ----------
        xrange : (int, int)
            The range of rows to compress (inclusive).
        yrange : (int, int) or None, optional
            The range of columns to compress (inclusive), sweeping along with
            canonization and compression. Defaults to all columns.
        max_bond : int, optional
            The maximum boundary dimension, AKA 'chi'. The default of ``None``
            means truncation is left purely to ``cutoff`` and is not
            recommended in 2D.
        cutoff : float, optional
            Cut-off value to used to truncate singular values in the boundary
            contraction.
        canonize : bool, optional
            Whether to sweep one way with canonization before compressing.
        mode : {'mps', 'full-bond'}, optional
            How to perform the compression on the boundary.
        layer_tags : None or str, optional
            If ``None``, all tensors at each coordinate pair
            ``[(i, j), (i - 1, j)]`` will be first contracted. If specified,
            then the outer tensor at ``(i, j)`` will be contracted with the
            tensor specified by ``[(i - 1, j), layer_tag]``, for each
            ``layer_tag`` in ``layer_tags``.
        sweep_reverse : bool, optional
            Which way to perform the compression sweep, which has an effect on
            which tensors end up being canonized. Setting this to true sweeps
            the compression from largest to smallest coordinates.
        compress_opts : None or dict, optional
            Supplied to :meth:`compress_between`.
        inplace : bool, optional
            Whether to perform the contraction inplace or not.

        See Also
        --------
        contract_boundary_from_xmin, contract_boundary_from_ymin,
        contract_boundary_from_ymax
        """
        return self.contract_boundary_from(
            xrange=xrange,
            yrange=yrange,
            from_which="xmax",
            max_bond=max_bond,
            cutoff=cutoff,
            canonize=canonize,
            mode=mode,
            layer_tags=layer_tags,
            sweep_reverse=sweep_reverse,
            compress_opts=compress_opts,
            inplace=inplace,
            **contract_boundary_opts,
        )

    contract_boundary_from_xmax_ = functools.partialmethod(
        contract_boundary_from_xmax, inplace=True
    )

    def contract_boundary_from_ymin(
        self,
        yrange,
        xrange=None,
        max_bond=None,
        *,
        cutoff=1e-10,
        canonize=True,
        mode="mps",
        layer_tags=None,
        sweep_reverse=False,
        compress_opts=None,
        inplace=False,
        **contract_boundary_opts,
    ):
        r"""Contract a 2D tensor network inwards from the left, canonizing and
        compressing (bottom to top) along the way. If
        ``layer_tags is None`` this looks like::

            a) contract

            ●──●──       ●──
            │  │         ║
            ●──●──  ==>  ●──
            │  │         ║
            ●──●──       ●──

            b) optionally canonicalize

            ●──       v──
            ║         ║
            ●──  ==>  v──
            ║         ║
            ●──       ●──

            c) compress in opposite direction

            v──       ●──
            ║         │
            v──  ==>  ^──
            ║         │
            ●──       ^──

        If ``layer_tags`` is specified, each then each layer is contracted in
        and compressed separately, resulting generally in a lower memory
        scaling. For two layer tags this looks like::

            a) first flatten the outer boundary only

            ○──○──           ●──○──
            │╲ │╲            │╲ │╲
            ●─○──○──         ╰─●──○──
             ╲│╲╲│╲     ==>    │╲╲│╲
              ●─○──○──         ╰─●──○──
               ╲│ ╲│             │ ╲│
                ●──●──           ╰──●──

            b) contract and compress a single layer only

               ○──
             ╱╱ ╲
            ●─── ○──
             ╲ ╱╱ ╲
              ^─── ○──
               ╲ ╱╱
                ^─────

            c) contract and compress the next layer

            ●──
            │╲
            ╰─●──
              │╲
              ╰─●──
                │
                ╰──

        Parameters
        ----------
        yrange : (int, int)
            The range of columns to compress (inclusive).
        xrange : (int, int) or None, optional
            The range of rows to compress (inclusive), sweeping along with
            canonization and compression. Defaults to all rows.
        max_bond : int, optional
            The maximum boundary dimension, AKA 'chi'. The default of ``None``
            means truncation is left purely to ``cutoff`` and is not
            recommended in 2D.
        cutoff : float, optional
            Cut-off value to used to truncate singular values in the boundary
            contraction.
        canonize : bool, optional
            Whether to sweep one way with canonization before compressing.
        mode : {'mps', 'full-bond'}, optional
            How to perform the compression on the boundary.
        layer_tags : None or str, optional
            If ``None``, all tensors at each coordinate pair
            ``[(i, j), (i, j + 1)]`` will be first contracted. If specified,
            then the outer tensor at ``(i, j)`` will be contracted with the
            tensor specified by ``[(i + 1, j), layer_tag]``, for each
            ``layer_tag`` in ``layer_tags``.
        sweep_reverse : bool, optional
            Which way to perform the compression sweep, which has an effect on
            which tensors end up being canonized. Setting this to true sweeps
            the compression from largest to smallest coordinates.
        compress_opts : None or dict, optional
            Supplied to :meth:`compress_between`.
        inplace : bool, optional
            Whether to perform the contraction inplace or not.

        See Also
        --------
        contract_boundary_from_xmin, contract_boundary_from_xmax,
        contract_boundary_from_ymax
        """
        return self.contract_boundary_from(
            xrange=xrange,
            yrange=yrange,
            from_which="ymin",
            max_bond=max_bond,
            cutoff=cutoff,
            canonize=canonize,
            mode=mode,
            layer_tags=layer_tags,
            sweep_reverse=sweep_reverse,
            compress_opts=compress_opts,
            inplace=inplace,
            **contract_boundary_opts,
        )

    contract_boundary_from_ymin_ = functools.partialmethod(
        contract_boundary_from_ymin, inplace=True
    )

    def contract_boundary_from_ymax(
        self,
        yrange,
        xrange=None,
        max_bond=None,
        *,
        cutoff=1e-10,
        canonize=True,
        mode="mps",
        layer_tags=None,
        sweep_reverse=False,
        compress_opts=None,
        inplace=False,
        **contract_boundary_opts,
    ):
        r"""Contract a 2D tensor network inwards from the left, canonizing and
        compressing (top to bottom) along the way. If
        ``layer_tags is None`` this looks like::

            a) contract

            ──●──●       ──●
              │  │         ║
            ──●──●  ==>  ──●
              │  │         ║
            ──●──●       ──●

            b) optionally canonicalize

            ──●       ──v
              ║         ║
            ──●  ==>  ──v
              ║         ║
            ──●       ──●

            c) compress in opposite direction

            ──v       ──●
              ║         │
            ──v  ==>  ──^
              ║         │
            ──●       ──^

        If ``layer_tags`` is specified, each then each layer is contracted in
        and compressed separately, resulting generally in a lower memory
        scaling. For two layer tags this looks like::

            a) first flatten the outer boundary only

                ──○──○           ──○──●
                 ╱│ ╱│            ╱│ ╱│
              ──○──○─●         ──○──●─╯
               ╱│╱╱│╱   ==>     ╱│╱╱│
            ──○──○─●         ──○──●─╯
              │╱ │╱            │╱ │
            ──●──●           ──●──╯

            b) contract and compress a single layer only

                ──○
                 ╱ ╲╲
              ──○────v
               ╱ ╲╲ ╱
            ──○────v
               ╲╲ ╱
            ─────●

            c) contract and compress the next layer

                   ╲
                ────v
                 ╲ ╱
              ────v
               ╲ ╱
            ────●

        Parameters
        ----------
        yrange : (int, int)
            The range of columns to compress (inclusive).
        xrange : (int, int) or None, optional
            The range of rows to compress (inclusive), sweeping along with
            canonization and compression. Defaults to all rows.
        max_bond : int, optional
            The maximum boundary dimension, AKA 'chi'. The default of ``None``
            means truncation is left purely to ``cutoff`` and is not
            recommended in 2D.
        cutoff : float, optional
            Cut-off value to used to truncate singular values in the boundary
            contraction.
        canonize : bool, optional
            Whether to sweep one way with canonization before compressing.
        mode : {'mps', 'full-bond'}, optional
            How to perform the compression on the boundary.
        layer_tags : None or str, optional
            If ``None``, all tensors at each coordinate pair
            ``[(i, j), (i, j - 1)]`` will be first contracted. If specified,
            then the outer tensor at ``(i, j)`` will be contracted with the
            tensor specified by ``[(i + 1, j), layer_tag]``, for each
            ``layer_tag`` in ``layer_tags``.
        sweep_reverse : bool, optional
            Which way to perform the compression sweep, which has an effect on
            which tensors end up being canonized. Setting this to true sweeps
            the compression from largest to smallest coordinates.
        compress_opts : None or dict, optional
            Supplied to :meth:`compress_between`.
        inplace : bool, optional
            Whether to perform the contraction inplace or not.

        See Also
        --------
        contract_boundary_from_xmin, contract_boundary_from_xmax,
        contract_boundary_from_ymin
        """
        return self.contract_boundary_from(
            xrange=xrange,
            yrange=yrange,
            from_which="ymax",
            max_bond=max_bond,
            cutoff=cutoff,
            canonize=canonize,
            mode=mode,
            layer_tags=layer_tags,
            sweep_reverse=sweep_reverse,
            compress_opts=compress_opts,
            inplace=inplace,
            **contract_boundary_opts,
        )

    contract_boundary_from_ymax_ = functools.partialmethod(
        contract_boundary_from_ymax, inplace=True
    )

    def _contract_interleaved_boundary_sequence(
        self,
        *,
        contract_boundary_opts=None,
        sequence=None,
        xmin=None,
        xmax=None,
        ymin=None,
        ymax=None,
        max_separation=1,
        max_unfinished=1,
        around=None,
        equalize_norms=False,
        canonize=False,
        canonize_opts=None,
        final_contract=True,
        final_contract_opts=None,
        progbar=False,
        inplace=False,
    ):
        """Unified handler for performing iterleaved contractions in a
        sequence of inwards boundary directions.
        """
        tn = self if inplace else self.copy()

        contract_boundary_opts = ensure_dict(contract_boundary_opts)
        if canonize:
            canonize_opts = ensure_dict(canonize_opts)
            canonize_opts.setdefault("max_iterations", 2)

        if progbar:
            pbar = Progbar()
            pbar.set_description(
                f"contracting boundary, Lx={tn.Lx}, Ly={tn.Ly}"
            )
        else:
            pbar = None

        # set default starting borders
        if any(d is None for d in (xmin, xmax, ymin, ymax)):
            (
                (auto_xmin, auto_xmax),
                (auto_ymin, auto_ymax),
            ) = self.get_ranges_present()

        # location of current boundaries
        boundaries = {
            "xmin": auto_xmin if xmin is None else xmin,
            "xmax": auto_xmax if xmax is None else xmax,
            "ymin": auto_ymin if ymin is None else ymin,
            "ymax": auto_ymax if ymax is None else ymax,
        }
        separations = {
            d: boundaries[f"{d}max"] - boundaries[f"{d}min"] for d in "xy"
        }
        boundary_tags = {
            "xmin": tn.x_tag(boundaries["xmin"]),
            "xmax": tn.x_tag(boundaries["xmax"]),
            "ymin": tn.y_tag(boundaries["ymin"]),
            "ymax": tn.y_tag(boundaries["ymax"]),
        }
        if around is not None:
            if sequence is None:
                sequence = ("xmin", "xmax", "ymin", "ymax")

            target_xmin = min(x[0] for x in around)
            target_xmax = max(x[0] for x in around)
            target_ymin = min(x[1] for x in around)
            target_ymax = max(x[1] for x in around)
            target_check = {
                "xmin": lambda x: x >= target_xmin - 1,
                "xmax": lambda x: x <= target_xmax + 1,
                "ymin": lambda y: y >= target_ymin - 1,
                "ymax": lambda y: y <= target_ymax + 1,
            }

        if sequence is None:
            # contract in both sides along short dimension -> less compression
            if self.Lx >= self.Ly:
                sequence = ("xmin", "xmax")
            else:
                sequence = ("ymin", "ymax")
        else:
            sequence = parse_boundary_sequence(sequence)

        def _is_finished(direction):
            return (
                # two opposing sides have got sufficiently close
                (separations[direction[0]] <= max_separation)
                or (
                    # there is a target region
                    (around is not None)
                    and
                    # and we have reached it
                    target_check[direction](boundaries[direction])
                )
            )

        sequence = [d for d in sequence if not _is_finished(d)]

        while sequence:
            direction = sequence.pop(0)
            if _is_finished(direction):
                # just remove direction from sequence
                continue
            # do a contraction, and keep direction in sequence to try again
            sequence.append(direction)

            if pbar is not None:
                pbar.set_description(
                    f"contracting {direction}, "
                    f"Lx={separations['x'] + 1}, "
                    f"Ly={separations['y'] + 1}"
                )

            if canonize:
                tn.select(boundary_tags[direction]).gauge_all_(**canonize_opts)

            if direction[0] == "x":
                if direction[1:] == "min":
                    xrange = (boundaries["xmin"], boundaries["xmin"] + 1)
                else:  # xmax
                    xrange = (boundaries["xmax"] - 1, boundaries["xmax"])
                yrange = (boundaries["ymin"], boundaries["ymax"])
            else:  # y
                if direction[1:] == "min":
                    yrange = (boundaries["ymin"], boundaries["ymin"] + 1)
                else:  # ymax
                    yrange = (boundaries["ymax"] - 1, boundaries["ymax"])
                xrange = (boundaries["xmin"], boundaries["xmax"])

            tn.contract_boundary_from_(
                xrange=xrange,
                yrange=yrange,
                from_which=direction,
                equalize_norms=equalize_norms,
                **contract_boundary_opts,
            )

            # update the boundaries and separations
            xy, minmax = direction[0], direction[1:]
            separations[xy] -= 1
            if minmax == "min":
                boundaries[direction] += 1
            else:
                boundaries[direction] -= 1

            if pbar is not None:
                pbar.update()

            # check if enough directions are finished -> reached max separation
            if (
                sum(separations[d] > max_separation for d in "xy")
                <= max_unfinished
            ):
                break

        if equalize_norms is True:
            tn.equalize_norms_()

        if pbar is not None:
            pbar.set_description(
                f"contracted boundary, "
                f"Lx={separations['x'] + 1}, "
                f"Ly={separations['y'] + 1}"
            )
            pbar.close()

        if final_contract and (around is None):
            final_contract_opts = ensure_dict(final_contract_opts)
            final_contract_opts.setdefault("optimize", "auto-hq")
            final_contract_opts.setdefault("inplace", inplace)
            return tn.contract(**final_contract_opts)

        return tn

    def contract_boundary(
        self,
        max_bond=None,
        *,
        cutoff=1e-10,
        canonize=True,
        mode="mps",
        layer_tags=None,
        compress_opts=None,
        sequence=None,
        xmin=None,
        xmax=None,
        ymin=None,
        ymax=None,
        max_separation=1,
        around=None,
        equalize_norms=False,
        final_contract=True,
        final_contract_opts=None,
        progbar=None,
        inplace=False,
        **contract_boundary_opts,
    ):
        """Contract the boundary of this 2D tensor network inwards::

            ●──●──●──●       ●──●──●──●       ●──●──●
            │  │  │  │       │  │  │  │       ║  │  │
            ●──●──●──●       ●──●──●──●       ^──●──●       >══>══●       >──v
            │  │ij│  │  ==>  │  │ij│  │  ==>  ║ij│  │  ==>  │ij│  │  ==>  │ij║
            ●──●──●──●       ●══<══<══<       ^──<──<       ^──<──<       ^──<
            │  │  │  │
            ●──●──●──●

        Optionally from any or all of the boundary, in multiple layers, and
        stopping around a region. The default is to contract the boundary from
        the two shortest opposing sides.

        Parameters
        ----------
        around : None or sequence of (int, int), optional
            If given, don't contract the square of sites bounding these
            coordinates.
        max_bond : int, optional
            The maximum boundary dimension, AKA 'chi'. The default of ``None``
            means truncation is left purely to ``cutoff`` and is not
            recommended in 2D.
        cutoff : float, optional
            Cut-off value to used to truncate singular values in the boundary
            contraction.
        canonize : bool, optional
            Whether to sweep one way with canonization before compressing.
        mode : {'mps', 'full-bond'}, optional
            How to perform the compression on the boundary.
        layer_tags : None or sequence of str, optional
            If given, perform a multilayer contraction, contracting the inner
            sites in each layer into the boundary individually.
        compress_opts : None or dict, optional
            Other low level options to pass to
            :meth:`~quimb.tensor.tensor_core.TensorNetwork.compress_between`.
        sequence : sequence of {'xmin', 'xmax', 'ymin', 'ymax'}, optional
            Which directions to cycle throught when performing the inwards
            contractions, i.e. *from* that direction. If ``around`` is
            specified you will likely need all of these! Default is to contract
            from the two shortest opposing sides.
        xmin : int, optional
            The initial bottom boundary row, defaults to 0.
        xmax : int, optional
            The initial top boundary row, defaults to ``Lx - 1``.
        ymin : int, optional
            The initial left boundary column, defaults to 0.
        ymax : int, optional
            The initial right boundary column, defaults to ``Ly - 1``..
        max_separation : int, optional
            If ``around is None``, when any two sides become this far apart
            simply contract the remaining tensor network.
        around : None or sequence of (int, int), optional
            If given, don't contract the square of sites bounding these
            coordinates.
        equalize_norms : bool or float, optional
            Whether to equalize the norms of the boundary tensors after each
            contraction, gathering the overall scaling coefficient, log10, in
            ``tn.exponent``.
        final_contract : bool, optional
            Whether to exactly contract the remaining tensor network after the
            boundary contraction.
        final_contract_opts : None or dict, optional
            Options to pass to
            :meth:`~quimb.tensor.tensor_core.TensorNetwork.contract`,
            ``optimize`` defaults to ``'auto-hq'``.
        progbar : bool, optional
            Whether to show a progress bar.
        inplace : bool, optional
            Whether to perform the contraction in place or not.
        contract_boundary_opts
            Supplied to :meth:`contract_boundary_from`, including compression
            and canonization options.
        """
        contract_boundary_opts["max_bond"] = max_bond
        contract_boundary_opts["mode"] = mode
        contract_boundary_opts["cutoff"] = cutoff
        contract_boundary_opts["canonize"] = canonize
        contract_boundary_opts["layer_tags"] = layer_tags
        contract_boundary_opts["compress_opts"] = compress_opts

        if mode == "full-bond":
            # set shared storage for opposite direction boundary contractions,
            #     this will be lazily filled by _contract_boundary_full_bond
            contract_boundary_opts.setdefault("opposite_envs", {})

        return self._contract_interleaved_boundary_sequence(
            contract_boundary_opts=contract_boundary_opts,
            sequence=sequence,
            xmin=xmin,
            xmax=xmax,
            ymin=ymin,
            ymax=ymax,
            max_separation=max_separation,
            around=around,
            equalize_norms=equalize_norms,
            final_contract=final_contract,
            final_contract_opts=final_contract_opts,
            progbar=progbar,
            inplace=inplace,
        )

    contract_boundary_ = functools.partialmethod(
        contract_boundary, inplace=True
    )

    def contract_mps_sweep(
        self,
        max_bond=None,
        *,
        cutoff=1e-10,
        canonize=True,
        direction=None,
        **contract_boundary_opts,
    ):
        """Contract this 2D tensor network by sweeping an MPS across from one
        side to the other.

        Parameters
        ----------
        max_bond : int, optional
            The maximum boundary dimension, AKA 'chi'. The default of ``None``
            means truncation is left purely to ``cutoff`` and is not
            recommended in 2D.
        cutoff : float, optional
            Cut-off value to used to truncate singular values in the boundary
            contraction.
        canonize : bool, optional
            Whether to sweep one way with canonization before compressing.
        direction : {'xmin', 'xmax', 'ymin', 'ymax'}, optional
            Which direction to sweep from. If ``None`` (default) then the
            shortest boundary is chosen.
        contract_boundary_opts
            Supplied to :meth:`contract_boundary_from`, including compression
            and canonization options.
        """
        if direction is None:
            # choose shortest boundary (i.e. more steps but less compression)
            direction = "xmin" if self.Ly <= self.Lx else "ymin"

        return self.contract_boundary(
            max_bond=max_bond,
            cutoff=cutoff,
            canonize=canonize,
            sequence=[direction],
            **contract_boundary_opts,
        )

    def contract_full_bootstrap(self, n, *, optimize="auto-hq", **kwargs):
        if n < 2:
            raise ValueError(f"``n`` must be at least 2 (got {n}).")

        if self.Lx >= self.Ly:
            fn_a = self.compute_xmax_environments
            fn_b = self.compute_xmin_environments
            mid, lbl_a, lbl_b = self.Ly // 2, "xmax", "xmin"
        else:
            fn_a = self.compute_ymax_environments
            fn_b = self.compute_ymin_environments
            mid, lbl_a, lbl_b = self.Lx // 2, "ymax", "ymin"

        kwargs.setdefault("envs", {})
        envs = kwargs["envs"]
        kwargs["opposite_envs"] = envs
        for _, env_compute in zip(range(1, n), cycle([fn_b, fn_a])):
            env_compute(mode="full-bond", **kwargs)

        tn = envs[lbl_a, mid] | envs[lbl_b, mid + 1]
        return tn.contract(all, optimize=optimize)

    def compute_environments(
        self,
        from_which,
        xrange=None,
        yrange=None,
        max_bond=None,
        *,
        cutoff=1e-10,
        canonize=True,
        mode="mps",
        layer_tags=None,
        dense=False,
        compress_opts=None,
        envs=None,
        **contract_boundary_opts,
    ):
        """Compute the 1D boundary tensor networks describing the environments
        of rows and columns.

        Parameters
        ----------
        from_which : {'xmin', 'xmax', 'ymin', 'ymax'}
            Which boundary to compute the environments from.
        xrange : tuple[int], optional
            The range of rows to compute the environments for.
        yrange : tuple[int], optional
            The range of columns to compute the environments for.
        max_bond : int, optional
            The maximum bond dimension of the environments.
        cutoff : float, optional
            The cutoff for the singular values of the environments.
        canonize : bool, optional
            Whether to canonicalize along each MPS environment before
            compressing.
        mode : {'mps', 'projector', 'full-bond'}, optional
            Which contraction method to use for the environments.
        layer_tags : str or iterable[str], optional
            If this 2D TN is multi-layered (e.g. a bra and a ket), and
            ``mode == 'mps'``, contract and compress each specified layer
            separately, for a cheaper contraction.
        dense : bool, optional
            Whether to use dense tensors for the environments.
        compress_opts : dict, optional
            Other options to pass to
            :func:`~quimb.tensor.tensor_core.tensor_compress_bond`.
        envs : dict, optional
            An existing dictionary to store the environments in.
        contract_boundary_opts
            Other options to pass to
            :meth:`~quimb.tensor.tensor_2d.TensorNetwork2D.contract_boundary_from`.

        Returns
        -------
        envs : dict
            A dictionary of the environments, with keys of the form
            ``(from_which, row_or_col_index)``.
        """
        tn = self.copy()

        r2d = Rotator2D(tn, xrange, yrange, from_which)
        sweep, x_tag = r2d.sweep, r2d.x_tag

        if envs is None:
            envs = {}

        if mode == "full-bond":
            # set shared storage for opposite env contractions
            contract_boundary_opts.setdefault("opposite_envs", {})

        envs[from_which, sweep[0]] = TensorNetwork([])
        first_row = x_tag(sweep[0])
        if dense:
            tn ^= first_row
        envs[from_which, sweep[1]] = tn.select(first_row)

        for i in sweep[2:]:
            iprevprev = i - 2 * sweep.step
            iprev = i - sweep.step
            if dense:
                tn ^= (x_tag(iprevprev), x_tag(iprev))
            else:
                tn.contract_boundary_from_(
                    xrange=(
                        (iprevprev, iprev) if r2d.plane == "x" else r2d.xrange
                    ),
                    yrange=(
                        (iprevprev, iprev) if r2d.plane == "y" else r2d.yrange
                    ),
                    from_which=from_which,
                    max_bond=max_bond,
                    cutoff=cutoff,
                    mode=mode,
                    canonize=canonize,
                    layer_tags=layer_tags,
                    compress_opts=compress_opts,
                    **contract_boundary_opts,
                )

            envs[from_which, i] = tn.select(first_row)

        return envs

    compute_xmin_environments = functools.partialmethod(
        compute_environments, from_which="xmin"
    )
    """Compute the ``self.Lx`` 1D boundary tensor networks describing
    the lower environments of each row in this 2D tensor network. See
    :meth:`~quimb.tensor.tensor_2d.TensorNetwork2D.compute_x_environments`
    for full details.
    """

    compute_xmax_environments = functools.partialmethod(
        compute_environments, from_which="xmax"
    )
    """Compute the ``self.Lx`` 1D boundary tensor networks describing
    the upper environments of each row in this 2D tensor network. See
    :meth:`~quimb.tensor.tensor_2d.TensorNetwork2D.compute_x_environments`
    for full details.
    """

    compute_ymin_environments = functools.partialmethod(
        compute_environments, from_which="ymin"
    )
    """Compute the ``self.Ly`` 1D boundary tensor networks describing
    the left environments of each column in this 2D tensor network. See
    :meth:`~quimb.tensor.tensor_2d.TensorNetwork2D.compute_y_environments`
    for full details.
    """

    compute_ymax_environments = functools.partialmethod(
        compute_environments, from_which="ymax"
    )
    """Compute the ``self.Ly`` 1D boundary tensor networks describing
    the right environments of each column in this 2D tensor network. See
    :meth:`~quimb.tensor.tensor_2d.TensorNetwork2D.compute_y_environments`
    for full details.
    """

    def compute_x_environments(
        self,
        max_bond=None,
        *,
        cutoff=1e-10,
        canonize=True,
        dense=False,
        mode="mps",
        layer_tags=None,
        compress_opts=None,
        envs=None,
        **contract_boundary_opts,
    ):
        r"""Compute the ``2 * self.Lx`` 1D boundary tensor networks describing
        the lower and upper environments of each row in this 2D tensor network,
        *assumed to represent the norm*.

        The top or 'xmax' environment for row ``i`` will be a contraction of
        all rows ``i + 1, i + 2, ...`` etc::

             ●━━━●━━━●━━━●━━━●━━━●━━━●━━━●━━━●━━━●
            ╱ ╲ ╱ ╲ ╱ ╲ ╱ ╲ ╱ ╲ ╱ ╲ ╱ ╲ ╱ ╲ ╱ ╲ ╱ ╲

        The bottom or 'xmin' environment for row ``i`` will be a contraction of
        all rows ``i - 1, i - 2, ...`` etc::

            ╲ ╱ ╲ ╱ ╲ ╱ ╲ ╱ ╲ ╱ ╲ ╱ ╲ ╱ ╲ ╱ ╲ ╱ ╲ ╱
             ●━━━●━━━●━━━●━━━●━━━●━━━●━━━●━━━●━━━●

        Such that
        ``envs['xmax', i] & self.select(self.x_tag(i)) & envs['xmin', i]``
        would look like::

             ●━━━●━━━●━━━●━━━●━━━●━━━●━━━●━━━●━━━●
            ╱ ╲ ╱ ╲ ╱ ╲ ╱ ╲ ╱ ╲ ╱ ╲ ╱ ╲ ╱ ╲ ╱ ╲ ╱ ╲
            o─o─o─o─o─o─o─o─o─o─o─o─o─o─o─o─o─o─o─o
            ╲ ╱ ╲ ╱ ╲ ╱ ╲ ╱ ╲ ╱ ╲ ╱ ╲ ╱ ╲ ╱ ╲ ╱ ╲ ╱
             ●━━━●━━━●━━━●━━━●━━━●━━━●━━━●━━━●━━━●

        And be (an approximation of) the norm centered around row ``i``

        Parameters
        ----------
        max_bond : int, optional
            The maximum boundary dimension, AKA 'chi'. The default of ``None``
            means truncation is left purely to ``cutoff`` and is not
            recommended in 2D.
        cutoff : float, optional
            Cut-off value to used to truncate singular values in the boundary
            contraction.
        canonize : bool, optional
            Whether to sweep one way with canonization before compressing.
        dense : bool, optional
            If true, contract the boundary in as a single dense tensor.
        mode : {'mps', 'full-bond'}, optional
            How to perform the boundary compression.
        layer_tags : None or sequence[str], optional
            If ``None``, all tensors at each coordinate pair
            ``[(i, j), (i + 1, j)]`` will be first contracted. If specified,
            then the outer tensor at ``(i, j)`` will be contracted with the
            tensor specified by ``[(i + 1, j), layer_tag]``, for each
            ``layer_tag`` in ``layer_tags``.
        compress_opts : None or dict, optional
            Supplied to :meth:`compress_between`.
        envs : dict, optional
            Supply an existing dictionary to store the environments in.
        contract_boundary_opts
            Supplied to
            :meth:`~quimb.tensor.tensor_2d.TensorNetwork2D.contract_boundary_from_xmin`
            and
            :meth:`~quimb.tensor.tensor_2d.TensorNetwork2D.contract_boundary_from_xmax`
            .

        Returns
        -------
        x_envs : dict[(str, int), TensorNetwork]
            The two environment tensor networks of row ``i`` will be stored in
            ``x_envs['xmin', i]`` and ``x_envs['xmax', i]``.
        """
        contract_boundary_opts["max_bond"] = max_bond
        contract_boundary_opts["cutoff"] = cutoff
        contract_boundary_opts["canonize"] = canonize
        contract_boundary_opts["mode"] = mode
        contract_boundary_opts["dense"] = dense
        contract_boundary_opts["layer_tags"] = layer_tags
        contract_boundary_opts["compress_opts"] = compress_opts

        if envs is None:
            envs = {}

        self.compute_xmax_environments(envs=envs, **contract_boundary_opts)
        self.compute_xmin_environments(envs=envs, **contract_boundary_opts)

        return envs

    def compute_y_environments(
        self,
        max_bond=None,
        *,
        cutoff=1e-10,
        canonize=True,
        dense=False,
        mode="mps",
        layer_tags=None,
        compress_opts=None,
        envs=None,
        **contract_boundary_opts,
    ):
        r"""Compute the ``2 * self.Ly`` 1D boundary tensor networks describing
        the left ('ymin') and right ('ymax') environments of each column in
        this 2D tensor network, assumed to represent the norm.

        The left or 'ymin' environment for column ``j`` will be a contraction
        of all columns ``j - 1, j - 2, ...`` etc::

            ●<
            ┃
            ●<
            ┃
            ●<
            ┃
            ●<


        The right or 'ymax' environment for row ``j`` will be a contraction of
        all rows ``j + 1, j + 2, ...`` etc::

            >●
             ┃
            >●
             ┃
            >●
             ┃
            >●

        Such that
        ``envs['ymin', j] & self.select(self.y_tag(j)) & envs['ymax', j]``
        would look like::

               ╱o
            ●< o| >●
            ┃  |o  ┃
            ●< o| >●
            ┃  |o  ┃
            ●< o| >●
            ┃  |o  ┃
            ●< o╱ >●

        And be (an approximation of) the norm centered around column ``j``

        Parameters
        ----------
        max_bond : int, optional
            The maximum boundary dimension, AKA 'chi'. The default of ``None``
            means truncation is left purely to ``cutoff`` and is not
            recommended in 2D.
        cutoff : float, optional
            Cut-off value to used to truncate singular values in the boundary
            contraction.
        canonize : bool, optional
            Whether to sweep one way with canonization before compressing.
        dense : bool, optional
            If true, contract the boundary in as a single dense tensor.
        mode : {'mps', 'full-bond'}, optional
            How to perform the boundary compression.
        layer_tags : None or sequence[str], optional
            If ``None``, all tensors at each coordinate pair
            ``[(i, j), (i + 1, j)]`` will be first contracted. If specified,
            then the outer tensor at ``(i, j)`` will be contracted with the
            tensor specified by ``[(i + 1, j), layer_tag]``, for each
            ``layer_tag`` in ``layer_tags``.
        compress_opts : None or dict, optional
            Supplied to :meth:`compress_between`.
        contract_boundary_opts
            Supplied to
            :meth:`~quimb.tensor.tensor_2d.TensorNetwork2D.contract_boundary_from_ymin`
            and
            :meth:`~quimb.tensor.tensor_2d.TensorNetwork2D.contract_boundary_from_ymax`
            .

        Returns
        -------
        y_envs : dict[(str, int), TensorNetwork]
            The two environment tensor networks of column ``j`` will be stored
            in ``y_envs['ymin', j]`` and ``y_envs['ymax', j]``.
        """
        contract_boundary_opts["max_bond"] = max_bond
        contract_boundary_opts["cutoff"] = cutoff
        contract_boundary_opts["canonize"] = canonize
        contract_boundary_opts["mode"] = mode
        contract_boundary_opts["dense"] = dense
        contract_boundary_opts["layer_tags"] = layer_tags
        contract_boundary_opts["compress_opts"] = compress_opts

        if envs is None:
            envs = {}

        self.compute_ymin_environments(envs=envs, **contract_boundary_opts)
        self.compute_ymax_environments(envs=envs, **contract_boundary_opts)

        return envs

    def _compute_plaquette_environments_x_first(
        self,
        x_bsz,
        y_bsz,
        max_bond=None,
        cutoff=1e-10,
        canonize=True,
        layer_tags=None,
        second_dense=None,
        x_envs=None,
        **compute_environment_opts,
    ):
        if second_dense is None:
            second_dense = x_bsz < 2

        # first we contract from either side to produce row environments
        if x_envs is None:
            x_envs = self.compute_x_environments(
                max_bond=max_bond,
                cutoff=cutoff,
                canonize=canonize,
                layer_tags=layer_tags,
                **compute_environment_opts,
            )

        # next we form horizontal strips and contract from both left and right
        #     for each row
        y_envs = dict()
        for i in range(self.Lx - x_bsz + 1):
            #
            #      ●━━━●━━━●━━━●━━━●━━━●━━━●━━━●━━━●━━━●
            #     ╱ ╲ ╱ ╲ ╱ ╲ ╱ ╲ ╱ ╲ ╱ ╲ ╱ ╲ ╱ ╲ ╱ ╲ ╱ ╲
            #     o─o─o─o─o─o─o─o─o─o─o─o─o─o─o─o─o─o─o─o     ┬
            #     | | | | | | | | | | | | | | | | | | | |     ┊ x_bsz
            #     o─o─o─o─o─o─o─o─o─o─o─o─o─o─o─o─o─o─o─o     ┴
            #     ╲ ╱ ╲ ╱ ╲ ╱ ╲ ╱ ╲ ╱ ╲ ╱ ╲ ╱ ╲ ╱ ╲ ╱ ╲ ╱
            #      ●━━━●━━━●━━━●━━━●━━━●━━━●━━━●━━━●━━━●
            #
            row_i = TensorNetwork(
                (
                    x_envs["xmin", i],
                    self.select_any([self.x_tag(i + x) for x in range(x_bsz)]),
                    x_envs["xmax", i + x_bsz - 1],
                )
            ).view_as_(TensorNetwork2D, like=self)
            #
            #           y_bsz
            #           <-->               second_dense=True
            #       ●──      ──●
            #       │          │            ╭──     ──╮
            #       ●── .  . ──●            │╭─ . . ─╮│     ┬
            #       │          │     or     ●         ●     ┊ x_bsz
            #       ●── .  . ──●            │╰─ . . ─╯│     ┴
            #       │          │            ╰──     ──╯
            #       ●──      ──●
            #     'ymin'    'ymax'       'ymin'    'ymax'
            #
            y_envs[i] = row_i.compute_y_environments(
                xrange=(max(i - 1, 0), min(i + x_bsz, self.Lx - 1)),
                max_bond=max_bond,
                cutoff=cutoff,
                canonize=canonize,
                layer_tags=layer_tags,
                dense=second_dense,
                **compute_environment_opts,
            )

        # then range through all the possible plaquettes, selecting the correct
        # boundary tensors from either the column or row environments
        plaquette_envs = dict()
        for i0, j0 in product(
            range(self.Lx - x_bsz + 1), range(self.Ly - y_bsz + 1)
        ):
            # we want to select bordering tensors from:
            #
            #       L──A──A──R    <- A from the row environments
            #       │  │  │  │
            #  i0+1 L──●──●──R
            #       │  │  │  │    <- L, R from the column environments
            #  i0   L──●──●──R
            #       │  │  │  │
            #       L──B──B──R    <- B from the row environments
            #
            #         j0  j0+1
            #
            ymin_coos = ((i0 + x, j0 - 1) for x in range(-1, x_bsz + 1))
            ymin_tags = tuple(
                map(self.site_tag, filter(self.valid_coo, ymin_coos))
            )

            ymax_coos = ((i0 + x, j0 + y_bsz) for x in range(-1, x_bsz + 1))
            ymax_tags = tuple(
                map(self.site_tag, filter(self.valid_coo, ymax_coos))
            )

            xmin_coos = ((i0 - 1, j0 + x) for x in range(y_bsz))
            xmin_tags = tuple(
                map(self.site_tag, filter(self.valid_coo, xmin_coos))
            )

            above_coos = ((i0 + x_bsz, j0 + x) for x in range(y_bsz))
            above_tags = tuple(
                map(self.site_tag, filter(self.valid_coo, above_coos))
            )

            env_ij = TensorNetwork(
                (
                    y_envs[i0]["ymin", j0].select_any(ymin_tags),
                    y_envs[i0]["ymax", j0 + y_bsz - 1].select_any(ymax_tags),
                    x_envs["xmin", i0].select_any(xmin_tags),
                    x_envs["xmax", i0 + x_bsz - 1].select_any(above_tags),
                )
            )

            # finally, absorb any rank-2 corner tensors
            env_ij.rank_simplify_()

            plaquette_envs[(i0, j0), (x_bsz, y_bsz)] = env_ij

        return plaquette_envs

    def _compute_plaquette_environments_y_first(
        self,
        x_bsz,
        y_bsz,
        max_bond=None,
        cutoff=1e-10,
        canonize=True,
        layer_tags=None,
        second_dense=None,
        y_envs=None,
        **compute_environment_opts,
    ):
        if second_dense is None:
            second_dense = y_bsz < 2

        # first we contract from either side to produce column environments
        if y_envs is None:
            y_envs = self.compute_y_environments(
                max_bond=max_bond,
                cutoff=cutoff,
                canonize=canonize,
                layer_tags=layer_tags,
                **compute_environment_opts,
            )

        # next we form vertical strips and contract from both top and bottom
        #     for each column
        x_envs = dict()
        for j in range(self.Ly - y_bsz + 1):
            #
            #        y_bsz
            #        <-->
            #
            #      ╭─╱o─╱o─╮
            #     ●──o|─o|──●
            #     ┃╭─|o─|o─╮┃
            #     ●──o|─o|──●
            #     ┃╭─|o─|o─╮┃
            #     ●──o|─o|──●
            #     ┃╭─|o─|o─╮┃
            #     ●──o╱─o╱──●
            #     ┃╭─|o─|o─╮┃
            #     ●──o╱─o╱──●
            #
            col_j = TensorNetwork(
                (
                    y_envs["ymin", j],
                    self.select_any(
                        [self.y_tag(j + jn) for jn in range(y_bsz)]
                    ),
                    y_envs["ymax", j + y_bsz - 1],
                )
            ).view_as_(TensorNetwork2D, like=self)
            #
            #        y_bsz
            #        <-->        second_dense=True
            #     ●──●──●──●      ╭──●──╮
            #     │  │  │  │  or  │ ╱ ╲ │    'xmax'
            #        .  .           . .                  ┬
            #                                            ┊ x_bsz
            #        .  .           . .                  ┴
            #     │  │  │  │  or  │ ╲ ╱ │    'xmin'
            #     ●──●──●──●      ╰──●──╯
            #
            x_envs[j] = col_j.compute_x_environments(
                yrange=(max(j - 1, 0), min(j + y_bsz, self.Ly - 1)),
                max_bond=max_bond,
                cutoff=cutoff,
                canonize=canonize,
                layer_tags=layer_tags,
                dense=second_dense,
                **compute_environment_opts,
            )

        # then range through all the possible plaquettes, selecting the correct
        # boundary tensors from either the column or row environments
        plaquette_envs = dict()
        for i0, j0 in product(
            range(self.Lx - x_bsz + 1), range(self.Ly - y_bsz + 1)
        ):
            # we want to select bordering tensors from:
            #
            #          A──A──A──A    <- A from the row environments
            #          │  │  │  │
            #     i0+1 L──●──●──R
            #          │  │  │  │    <- L, R from the column environments
            #     i0   L──●──●──R
            #          │  │  │  │
            #          B──B──B──B    <- B from the row environments
            #
            #            j0  j0+1
            #
            ymin_coos = ((i0 + x, j0 - 1) for x in range(x_bsz))
            ymin_tags = tuple(
                map(self.site_tag, filter(self.valid_coo, ymin_coos))
            )

            ymax_coos = ((i0 + x, j0 + y_bsz) for x in range(x_bsz))
            ymax_tags = tuple(
                map(self.site_tag, filter(self.valid_coo, ymax_coos))
            )

            xmin_coos = ((i0 - 1, j0 + x) for x in range(-1, y_bsz + 1))
            xmin_tags = tuple(
                map(self.site_tag, filter(self.valid_coo, xmin_coos))
            )

            xmax_coos = ((i0 + x_bsz, j0 + x) for x in range(-1, y_bsz + 1))
            xmax_tags = tuple(
                map(self.site_tag, filter(self.valid_coo, xmax_coos))
            )

            env_ij = TensorNetwork(
                (
                    y_envs["ymin", j0].select_any(ymin_tags),
                    y_envs["ymax", j0 + y_bsz - 1].select_any(ymax_tags),
                    x_envs[j0]["xmin", i0].select_any(xmin_tags),
                    x_envs[j0]["xmax", i0 + x_bsz - 1].select_any(xmax_tags),
                )
            )

            # finally, absorb any rank-2 corner tensors
            env_ij.rank_simplify_()

            plaquette_envs[(i0, j0), (x_bsz, y_bsz)] = env_ij

        return plaquette_envs

    def compute_plaquette_environments(
        self,
        x_bsz=2,
        y_bsz=2,
        max_bond=None,
        *,
        cutoff=1e-10,
        canonize=True,
        mode="mps",
        layer_tags=None,
        first_contract=None,
        second_dense=None,
        compress_opts=None,
        **compute_environment_opts,
    ):
        r"""Compute all environments like::

            second_dense=False   second_dense=True (& first_contract='columns')

              ●──●                  ╭───●───╮
             ╱│  │╲                 │  ╱ ╲  │
            ●─.  .─●    ┬           ●─ . . ─●    ┬
            │      │    ┊ x_bsz     │       │    ┊ x_bsz
            ●─.  .─●    ┴           ●─ . . ─●    ┴
             ╲│  │╱                 │  ╲ ╱  │
              ●──●                  ╰───●───╯

              <-->                    <->
             y_bsz                   y_bsz

        Use two boundary contractions sweeps.

        Parameters
        ----------
        x_bsz : int, optional
            The size of the plaquettes in the x-direction (number of rows).
        y_bsz : int, optional
            The size of the plaquettes in the y-direction (number of columns).
        max_bond : int, optional
            The maximum boundary dimension, AKA 'chi'. The default of ``None``
            means truncation is left purely to ``cutoff`` and is not
            recommended in 2D.
        cutoff : float, optional
            Cut-off value to used to truncate singular values in the boundary
            contraction.
        canonize : bool, optional
            Whether to sweep one way with canonization before compressing.
        mode : {'mps', 'full-bond'}, optional
            How to perform the boundary compression.
        layer_tags : None or sequence[str], optional
            If ``None``, all tensors at each coordinate pair
            ``[(i, j), (i + 1, j)]`` will be first contracted. If specified,
            then the outer tensor at ``(i, j)`` will be contracted with the
            tensor specified by ``[(i + 1, j), layer_tag]``, for each
            ``layer_tag`` in ``layer_tags``.
        first_contract : {None, 'x', 'y'}, optional
            The environments can either be generated with initial sweeps in
            the row ('x') or column ('y') direction. Generally it makes sense
            to perform this approximate step in whichever is smaller (the
            default).
        second_dense : None or bool, optional
            Whether to perform the second set of contraction sweeps (in the
            rotated direction from whichever ``first_contract`` is) using
            a dense tensor or boundary method. By default this is only turned
            on if the ``bsz`` in the corresponding direction is 1.
        compress_opts : None or dict, optional
            Supplied to :meth:`compress_between`.
        compute_environment_opts
            Supplied to
            :meth:`~quimb.tensor.tensor_2d.TensorNetwork2D.compute_y_environments`
            or
            :meth:`~quimb.tensor.tensor_2d.TensorNetwork2D.compute_x_environments`
            .

        Returns
        -------
        dict[((int, int), (int, int)), TensorNetwork]
            The plaquette environments. The key is two tuples of ints, the
            startings coordinate of the plaquette being the first and the size
            of the plaquette being the second pair.
        """
        if first_contract is None:
            if x_bsz > y_bsz:
                first_contract = "y"
            elif y_bsz > x_bsz:
                first_contract = "x"
            elif self.Lx >= self.Ly:
                first_contract = "x"
            else:
                first_contract = "y"

        compute_env_fn = {
            "x": self._compute_plaquette_environments_x_first,
            "y": self._compute_plaquette_environments_y_first,
        }[first_contract]

        return compute_env_fn(
            x_bsz=x_bsz,
            y_bsz=y_bsz,
            max_bond=max_bond,
            cutoff=cutoff,
            canonize=canonize,
            mode=mode,
            layer_tags=layer_tags,
            compress_opts=compress_opts,
            second_dense=second_dense,
            **compute_environment_opts,
        )

    def coarse_grain_hotrg(
        self,
        direction,
        max_bond=None,
        cutoff=1e-10,
        lazy=False,
        equalize_norms=False,
        optimize="auto-hq",
        compress_opts=None,
        inplace=False,
    ):
        """Coarse grain this tensor network in ``direction`` using HOTRG. This
        inserts oblique projectors between tensor pairs and then optionally
        contracts them into new sites for form a lattice half the size.

        Parameters
        ----------
        direction : {'x', 'y'}
            The direction to coarse grain in.
        max_bond : int, optional
            The maximum bond dimension of the projector pairs inserted.
        cutoff : float, optional
            The cutoff for the singular values of the projector pairs.
        lazy : bool, optional
            Whether to contract the coarse graining projectors or leave them
            in the tensor network lazily. Default is to contract them.
        equalize_norms : bool, optional
            Whether to equalize the norms of the tensors in the coarse grained
            lattice.
        optimize : str, optional
            The optimization method to use when contracting the coarse grained
            lattice, if ``lazy=False``.
        compress_opts : None or dict, optional
            Supplied to
            :meth:`~quimb.tensor.tensor_core.TensorNetwork.insert_compressor_between_regions`.
        inplace : bool, optional
            Whether to perform the coarse graining in place.

        Returns
        -------
        TensorNetwork2D
            The coarse grained tensor network, with size halved in
            ``direction``.

        See Also
        --------
        contract_hotrg, TensorNetwork.insert_compressor_between_regions
        """
        compress_opts = ensure_dict(compress_opts)
        check_opt("direction", direction, ("x", "y"))

        tn = self if inplace else self.copy()
        tn_calc = tn.copy()

        r = Rotator2D(tn, None, None, direction + "min")

        # track new coordinates / tags
        retag_map = {}

        for i in range(r.imin, r.imax + 1, 2):
            next_i_in_lattice = i + 1 <= r.imax

            for j in range(r.jmin, r.jmax + 1):
                #      │         │
                #    ──O─┐ chi ┌─O──  i+1
                #      │ └─▷═◁─┘ │
                #      │ ┌┘   └┐ │
                #    ──O─┘     └─O──  i
                #      │         │
                #     j-1        j
                tag_ij = r.site_tag(i, j)
                tag_ip1j = r.site_tag(i + 1, j)
                new_tag = r.site_tag(i // 2, j)
                retag_map[tag_ij] = new_tag
                if next_i_in_lattice:
                    retag_map[tag_ip1j] = new_tag

                if (j > 0) and next_i_in_lattice:
                    ltags = r.site_tag(i, j - 1), r.site_tag(i + 1, j - 1)
                    rtags = (tag_ij, tag_ip1j)
                    tn_calc.insert_compressor_between_regions(
                        ltags,
                        rtags,
                        new_ltags=ltags,
                        new_rtags=rtags,
                        insert_into=tn,
                        max_bond=max_bond,
                        cutoff=cutoff,
                        **compress_opts,
                    )

            retag_map[r.x_tag(i)] = r.x_tag(i // 2)
            if next_i_in_lattice:
                retag_map[r.x_tag(i + 1)] = r.x_tag(i // 2)

        # then we retag the tensor network and adjust its size
        tn.retag_(retag_map)
        if direction == "x":
            tn._Lx = tn.Lx // 2 + tn.Lx % 2
        else:  # 'y'
            tn._Ly = tn.Ly // 2 + tn.Ly % 2

        # need this since we've fundamentally changed the geometry
        tn.reset_cached_properties()

        if not lazy:
            # contract each pair of tensors with their projectors
            for st in tn.site_tags:
                tn.contract_tags_(st, optimize=optimize)

        if equalize_norms:
            tn.equalize_norms_(value=equalize_norms)

        return tn

    coarse_grain_hotrg_ = functools.partialmethod(
        coarse_grain_hotrg, inplace=True
    )

    def contract_hotrg(
        self,
        max_bond=None,
        *,
        cutoff=1e-10,
        canonize=False,
        canonize_opts=None,
        sequence=("x", "y"),
        max_separation=1,
        max_unfinished=1,
        lazy=False,
        equalize_norms=False,
        final_contract=True,
        final_contract_opts=None,
        progbar=False,
        inplace=False,
        **coarse_grain_opts,
    ):
        """Contract this tensor network using the finite version of HOTRG.
        See https://arxiv.org/abs/1201.1144v4 and
        https://arxiv.org/abs/1905.02351 for the more optimal computaton of the
        projectors used here. The TN is contracted sequentially in
        ``sequence`` directions by inserting oblique projectors between
        plaquettes, and then optionally contracting these new effective sites.
        The algorithm stops when only one direction has a length larger than 2,
        and thus exact contraction can be used.

        Parameters
        ----------
        max_bond : int, optional
            The maximum bond dimension of the projector pairs inserted.
        cutoff : float, optional
            The cutoff for the singular values of the projector pairs.
        canonize : bool, optional
            Whether to canonize all tensors before each contraction,
            via :meth:`gauge_all`.
        canonize_opts : None or dict, optional
            Additional options to pass to
            :meth:`gauge_all`.
        sequence : tuple of str, optional
            The directions to contract in.  Default is to contract in all
            directions.
        max_separation : int, optional
            The maximum distance between sides (i.e. length - 1) of the tensor
            network before that direction is considered finished.
        max_unfinished : int, optional
            The maximum number of directions that can be unfinished (i.e. are
            still longer than max_separation + 1) before the coarse graining
            terminates.
        lazy : bool, optional
            Whether to contract the coarse graining projectors or leave them
            in the tensor network lazily. Default is to contract them.
        equalize_norms : bool or float, optional
            Whether to equalize the norms of all tensors after each
            contraction, gathering the overall scaling coefficient, log10, in
            ``tn.exponent``.
        final_contract : bool, optional
            Whether to exactly contract the remaining tensor network after the
            coarse graining contractions.
        final_contract_opts : None or dict, optional
            Options to pass to :meth:`contract`, ``optimize`` defaults to
            ``'auto-hq'``.
        progbar : bool, optional
            Whether to show a progress bar.
        inplace : bool, optional
            Whether to perform the coarse graining in place.
        coarse_grain_opts
            Additional options to pass to :meth:`coarse_grain_hotrg`.

        Returns
        -------
        TensorNetwork2D
            The contracted tensor network, which will have no more than one
            direction of length > 2.

        See Also
        --------
        coarse_grain_hotrg, contract_ctmrg,
        TensorNetwork.insert_compressor_between_regions
        """
        tn = self if inplace else self.copy()

        if lazy:
            # we are implicitly asking for the tensor network
            final_contract = False

        if canonize:
            canonize_opts = ensure_dict(canonize_opts)
            canonize_opts.setdefault("max_iterations", 2)

        if progbar:
            pbar = Progbar(desc=f"contracting HOTRG, Lx={tn.Lx}, Ly={tn.Ly}")
        else:
            pbar = None

        def _is_finished(direction):
            return getattr(tn, "L" + direction) <= max_separation + 1

        sequence = [d for d in sequence if not _is_finished(d)]
        while sequence:
            direction = sequence.pop(0)
            if _is_finished(direction):
                # just remove direction from sequence
                continue
            # do a contraction, and keep direction in sequence to try again
            sequence.append(direction)

            if pbar is not None:
                pbar.set_description(
                    f"contracting {direction}, Lx={tn.Lx}, Ly={tn.Ly}"
                )

            if canonize:
                tn.gauge_all_(**canonize_opts)

            tn.coarse_grain_hotrg_(
                direction=direction,
                max_bond=max_bond,
                cutoff=cutoff,
                lazy=lazy,
                equalize_norms=equalize_norms,
                **coarse_grain_opts,
            )

            if pbar is not None:
                pbar.update()

            # check if enough directions are finished -> reached max separation
            if sum(not _is_finished(d) for d in "xy") <= max_unfinished:
                break

        if equalize_norms is True:
            # redistribute the exponent equally among all tensors
            tn.equalize_norms_()

        if final_contract:
            final_contract_opts = ensure_dict(final_contract_opts)
            final_contract_opts.setdefault("optimize", "auto-hq")
            final_contract_opts.setdefault("inplace", inplace)
            return tn.contract(**final_contract_opts)

        return tn

    contract_hotrg_ = functools.partialmethod(contract_hotrg, inplace=True)

    def contract_ctmrg(
        self,
        max_bond=None,
        *,
        cutoff=1e-10,
        canonize=False,
        canonize_opts=None,
        lazy=False,
        mode="projector",
        compress_opts=None,
        sequence=("xmin", "xmax", "ymin", "ymax"),
        xmin=None,
        xmax=None,
        ymin=None,
        ymax=None,
        max_separation=1,
        around=None,
        equalize_norms=False,
        final_contract=True,
        final_contract_opts=None,
        progbar=False,
        inplace=False,
        **contract_boundary_opts,
    ):
        """Contract this 2D tensor network using the finite analog of the
        CTMRG algorithm - https://arxiv.org/abs/cond-mat/9507087. The TN is
        contracted sequentially in ``sequence`` directions by inserting oblique
        projectors between boundary pairs, and then optionally contracting
        these new effective sites. The algorithm stops when only one direction
        has a length larger than 2, and thus exact contraction can be used.

        Parameters
        ----------
        max_bond : int, optional
            The maximum bond dimension of the projector pairs inserted.
        cutoff : float, optional
            The cutoff for the singular values of the projector pairs.
        canonize : bool, optional
            Whether to canonize the boundary tensors before each contraction,
            via :meth:`gauge_all`.
        canonize_opts : None or dict, optional
            Additional options to pass to :meth:`gauge_all`.
        lazy : bool, optional
            Whether to contract the coarse graining projectors or leave them
            in the tensor network lazily. Default is to contract them.
        mode : str, optional
            The method to perform the boundary contraction. Defaults to
            ``'projector'``.
        compress_opts : None or dict, optional
            Other low level options to pass to
            :meth:`insert_compressor_between_regions`.
        sequence : sequence of {'xmin', 'xmax', 'ymin', 'ymax'}, optional
            Which directions to cycle throught when performing the inwards
            contractions, i.e. *from* that direction. If ``around`` is
            specified you will likely need all of these! Default is to contract
            in all directions.
        xmin : int, optional
            The initial bottom boundary row, defaults to 0.
        xmax : int, optional
            The initial top boundary row, defaults to ``Lx - 1``.
        ymin : int, optional
            The initial left boundary column, defaults to 0.
        ymax : int, optional
            The initial right boundary column, defaults to ``Ly - 1``..
        max_separation : int, optional
            If ``around is None``, when any two sides become this far apart
            simply contract the remaining tensor network.
        around : None or sequence of (int, int), optional
            If given, don't contract the square of sites bounding these
            coordinates.
        equalize_norms : bool or float, optional
            Whether to equalize the norms of the boundary tensors after each
            contraction, gathering the overall scaling coefficient, log10, in
            ``tn.exponent``.
        final_contract : bool, optional
            Whether to exactly contract the remaining tensor network after the
            boundary contraction.
        final_contract_opts : None or dict, optional
            Options to pass to :meth:`contract`, ``optimize`` defaults to
            ``'auto-hq'``.
        progbar : bool, optional
            Whether to show a progress bar.
        inplace : bool, optional
            Whether to perform the boundary contraction in place.
        contract_boundary_opts
            Additional options to pass to :meth:`contract_boundary_from`.

        Returns
        -------
        scalar or TensorNetwork2D
            Either the fully contracted scalar (if ``final_contract=True`` and
            ``around=None``) or the partially contracted tensor network.

        See Also
        --------
        contract_boundary_from, contract_hotrg,
        TensorNetwork.insert_compressor_between_regions
        """
        contract_boundary_opts["max_bond"] = max_bond
        contract_boundary_opts["cutoff"] = cutoff
        contract_boundary_opts["mode"] = mode
        contract_boundary_opts["compress_opts"] = compress_opts
        contract_boundary_opts["lazy"] = lazy

        if lazy:
            # we are implicitly asking for the tensor network
            final_contract = False

        return self._contract_interleaved_boundary_sequence(
            contract_boundary_opts=contract_boundary_opts,
            canonize=canonize,
            canonize_opts=canonize_opts,
            sequence=sequence,
            xmin=xmin,
            xmax=xmax,
            ymin=ymin,
            ymax=ymax,
            max_separation=max_separation,
            around=around,
            equalize_norms=equalize_norms,
            final_contract=final_contract,
            final_contract_opts=final_contract_opts,
            progbar=progbar,
            inplace=inplace,
        )

    contract_ctmrg_ = functools.partialmethod(contract_ctmrg, inplace=True)


def is_lone_coo(where):
    """Check if ``where`` has been specified as a single coordinate pair."""
    return (len(where) == 2) and (isinstance(where[0], Integral))


def gate_string_split_(
    TG,
    where,
    string,
    original_ts,
    bonds_along,
    reindex_map,
    site_ix,
    info,
    **compress_opts,
):
    # by default this means singuvalues are kept in the string 'blob' tensor
    compress_opts.setdefault("absorb", "right")

    # the outer, neighboring indices of each tensor in the string
    neighb_inds = []

    # tensors we are going to contract in the blob, reindex some to attach gate
    contract_ts = []

    for t, coo in zip(original_ts, string):
        neighb_inds.append(tuple(ix for ix in t.inds if ix not in bonds_along))
        contract_ts.append(t.reindex(reindex_map) if coo in where else t)

    # form the central blob of all sites and gate contracted
    blob = tensor_contract(*contract_ts, TG)

    regauged = []

    # one by one extract the site tensors again from each end
    inner_ts = [None] * len(string)
    i = 0
    j = len(string) - 1

    while True:
        lix = neighb_inds[i]
        if i > 0:
            lix += (bonds_along[i - 1],)

        # the original bond we are restoring
        bix = bonds_along[i]

        # split the blob!
        inner_ts[i], *maybe_svals, blob = blob.split(
            left_inds=lix, get="tensors", bond_ind=bix, **compress_opts
        )

        # if singular values are returned (``absorb=None``) check if we should
        #     return them via ``info``, e.g. for ``SimpleUpdate`
        if maybe_svals and info is not None:
            s = next(iter(maybe_svals)).data
            coo_pair = tuple(sorted((string[i], string[i + 1])))
            info["singular_values", coo_pair] = s

            # regauge the blob but record so as to unguage later
            if i != j - 1:
                blob.multiply_index_diagonal_(bix, s)
                regauged.append((i + 1, bix, s))

        # move inwards along string, terminate if two ends meet
        i += 1
        if i == j:
            inner_ts[i] = blob
            break

        # extract at end of string
        lix = neighb_inds[j]
        if j < len(string) - 1:
            lix += (bonds_along[j],)

        # the original bond we are restoring
        bix = bonds_along[j - 1]

        # split the blob!
        inner_ts[j], *maybe_svals, blob = blob.split(
            left_inds=lix, get="tensors", bond_ind=bix, **compress_opts
        )

        # if singular values are returned (``absorb=None``) check if we should
        #     return them via ``info``, e.g. for ``SimpleUpdate`
        if maybe_svals and info is not None:
            s = next(iter(maybe_svals)).data
            coo_pair = tuple(sorted((string[j - 1], string[j])))
            info["singular_values", coo_pair] = s

            # regauge the blob but record so as to unguage later
            if j != i + 1:
                blob.multiply_index_diagonal_(bix, s)
                regauged.append((j - 1, bix, s))

        # move inwards along string, terminate if two ends meet
        j -= 1
        if j == i:
            inner_ts[j] = blob
            break

    # ungauge the site tensors along bond if necessary
    for i, bix, s in regauged:
        t = inner_ts[i]
        t.multiply_index_diagonal_(bix, s**-1)

    # transpose to match original tensors and update original data
    for to, tn in zip(original_ts, inner_ts):
        tn.transpose_like_(to)
        to.modify(data=tn.data)


def gate_string_reduce_split_(
    TG,
    where,
    string,
    original_ts,
    bonds_along,
    reindex_map,
    site_ix,
    info,
    **compress_opts,
):
    # by default this means singuvalues are kept in the string 'blob' tensor
    compress_opts.setdefault("absorb", "right")

    # indices to reduce, first and final include physical indices for gate
    inds_to_reduce = [(bonds_along[0], site_ix[0])]
    for b1, b2 in pairwise(bonds_along):
        inds_to_reduce.append((b1, b2))
    inds_to_reduce.append((bonds_along[-1], site_ix[-1]))

    # tensors that remain on the string sites and those pulled into string
    outer_ts, inner_ts = [], []
    for coo, rix, t in zip(string, inds_to_reduce, original_ts):
        tq, tr = t.split(
            left_inds=None, right_inds=rix, method="qr", get="tensors"
        )
        outer_ts.append(tq)
        inner_ts.append(tr.reindex_(reindex_map) if coo in where else tr)

    # contract the blob of gate with reduced tensors only
    blob = tensor_contract(*inner_ts, TG)

    regauged = []

    # extract the new reduced tensors sequentially from each end
    i = 0
    j = len(string) - 1

    while True:
        # extract at beginning of string
        lix = bonds(blob, outer_ts[i])
        if i == 0:
            lix.add(site_ix[0])
        else:
            lix.add(bonds_along[i - 1])

        # the original bond we are restoring
        bix = bonds_along[i]

        # split the blob!
        inner_ts[i], *maybe_svals, blob = blob.split(
            left_inds=lix, get="tensors", bond_ind=bix, **compress_opts
        )

        # if singular values are returned (``absorb=None``) check if we should
        #     return them via ``info``, e.g. for ``SimpleUpdate`
        if maybe_svals and info is not None:
            s = next(iter(maybe_svals)).data
            coo_pair = tuple(sorted((string[i], string[i + 1])))
            info["singular_values", coo_pair] = s

            # regauge the blob but record so as to unguage later
            if i != j - 1:
                blob.multiply_index_diagonal_(bix, s)
                regauged.append((i + 1, bix, s))

        # move inwards along string, terminate if two ends meet
        i += 1
        if i == j:
            inner_ts[i] = blob
            break

        # extract at end of string
        lix = bonds(blob, outer_ts[j])
        if j == len(string) - 1:
            lix.add(site_ix[-1])
        else:
            lix.add(bonds_along[j])

        # the original bond we are restoring
        bix = bonds_along[j - 1]

        # split the blob!
        inner_ts[j], *maybe_svals, blob = blob.split(
            left_inds=lix, get="tensors", bond_ind=bix, **compress_opts
        )

        # if singular values are returned (``absorb=None``) check if we should
        #     return them via ``info``, e.g. for ``SimpleUpdate`
        if maybe_svals and info is not None:
            s = next(iter(maybe_svals)).data
            coo_pair = tuple(sorted((string[j - 1], string[j])))
            info["singular_values", coo_pair] = s

            # regauge the blob but record so as to unguage later
            if j != i + 1:
                blob.multiply_index_diagonal_(bix, s)
                regauged.append((j - 1, bix, s))

        # move inwards along string, terminate if two ends meet
        j -= 1
        if j == i:
            inner_ts[j] = blob
            break

    # reabsorb the inner reduced tensors into the sites
    new_ts = [
        tensor_contract(ts, tr, output_inds=to.inds)
        for to, ts, tr in zip(original_ts, outer_ts, inner_ts)
    ]

    # ungauge the site tensors along bond if necessary
    for i, bix, s in regauged:
        t = new_ts[i]
        t.multiply_index_diagonal_(bix, s**-1)

    # update originals
    for to, t in zip(original_ts, new_ts):
        to.modify(data=t.data)


class TensorNetwork2DVector(TensorNetwork2D, TensorNetworkGenVector):
    """Mixin class  for a 2D square lattice vector TN, i.e. one with a single
    physical index per site.
    """

    _EXTRA_PROPS = (
        "_site_tag_id",
        "_x_tag_id",
        "_y_tag_id",
        "_Lx",
        "_Ly",
        "_site_ind_id",
    )

    def site_ind(self, i, j=None):
        """Return the physical index of site ``(i, j)``."""
        if j is None:
            i, j = i
        if not isinstance(i, str):
            i = i % self.Lx
        if not isinstance(j, str):
            j = j % self.Ly
        return self.site_ind_id.format(i, j)

    def reindex_sites(self, new_id, where=None, inplace=False):
        if where is None:
            where = self.gen_sites_present()

        return self.reindex(
            {self.site_ind(*ij): new_id.format(*ij) for ij in where},
            inplace=inplace,
        )

    reindex_sites_ = functools.partialmethod(reindex_sites, inplace=True)

    def phys_dim(self, i=None, j=None):
        """Get the size of the physical indices / a specific physical index."""
        if (i is not None) and (j is not None):
            pix = self.site_ind(i, j)
        else:
            # allow for when some physical indices might have been contracted
            pix = next(iter(ix for ix in self.site_inds if ix in self.ind_map))
        return self.ind_size(pix)

    def gate(
        self,
        G,
        where,
        contract=False,
        tags=None,
        propagate_tags="sites",
        inplace=False,
        info=None,
        long_range_use_swaps=False,
        long_range_path_sequence=None,
        **compress_opts,
    ):
        """Apply the dense gate ``G``, maintaining the physical indices of this
        2D vector tensor network.

        Parameters
        ----------
        G : array_like
            The gate array to apply, should match or be factorable into the
            shape ``(phys_dim,) * (2 * len(where))``.
        where : sequence of tuple[int, int] or tuple[int, int]
            Which site coordinates to apply the gate to.
        contract : {'reduce-split', 'split', False, True}, optional
            How to contract the gate into the 2D tensor network:

                - False: gate is added to network and nothing is contracted,
                  tensor network structure is thus not maintained.
                - True: gate is contracted with all tensors involved, tensor
                  network structure is thus only maintained if gate acts on a
                  single site only.
                - 'split': contract all involved tensors then split the result
                  back into two.
                - 'reduce-split': factor the two physical indices into
                  'R-factors' using QR decompositions on the original site
                  tensors, then contract the gate, split it and reabsorb each
                  side. Much cheaper than ``'split'``.

            The final two methods are relevant for two site gates only, for
            single site gates they use the ``contract=True`` option which also
            maintains the structure of the TN. See below for a pictorial
            description of each method.
        tags : str or sequence of str, optional
            Tags to add to the new gate tensor.
        propagate_tags : {'sites', 'register', True, False}, optional
            If ``contract==False``, which tags to propagate to the new gate
            tensor from the tensors it was applied to:

                - If ``'sites'``, then only propagate tags matching e.g.
                  'I{},{}' and ignore all others. I.e. assuming unitary gates
                  just propagate the causal lightcone.
                - If ``'register'``, then only propagate tags matching the
                  sites of where this gate was actually applied. I.e. ignore
                  the lightcone, just keep track of which 'registers' the gate
                  was applied to.
                - If ``False``, propagate nothing.
                - If ``True``, propagate all tags.

        inplace : bool, optional
            Whether to perform the gate operation inplace on the tensor
            network or not.
        info : None or dict, optional
            Used to store extra optional information such as the singular
            values if not absorbed.
        compress_opts
            Supplied to :func:`~quimb.tensor.tensor_core.tensor_split` for any
            ``contract`` methods that involve splitting. Ignored otherwise.

        Returns
        -------
        G_psi : TensorNetwork2DVector
            The new 2D vector TN like ``IIIGII @ psi`` etc.

        Notes
        -----

        The ``contract`` options look like the following (for two site gates).

        ``contract=False``::

              │   │
              GGGGG
              │╱  │╱
            ──●───●──
             ╱   ╱

        ``contract=True``::

              │╱  │╱
            ──GGGGG──
             ╱   ╱

        ``contract='split'``::

              │╱  │╱          │╱  │╱
            ──GGGGG──  ==>  ──G┄┄┄G──
             ╱   ╱           ╱   ╱
             <SVD>

        ``contract='reduce-split'``::

               │   │             │ │
               GGGGG             GGG               │ │
               │╱  │╱   ==>     ╱│ │  ╱   ==>     ╱│ │  ╱          │╱  │╱
             ──●───●──       ──>─●─●─<──       ──>─GGG─<──  ==>  ──G┄┄┄G──
              ╱   ╱           ╱     ╱           ╱     ╱           ╱   ╱
            <QR> <LQ>                            <SVD>

        For one site gates when one of the 'split' methods is supplied
        ``contract=True`` is assumed.
        """
        check_opt("contract", contract, (False, True, "split", "reduce-split"))

        psi = self if inplace else self.copy()

        if is_lone_coo(where):
            where = (where,)
        else:
            where = tuple(where)
        ng = len(where)

        dp = psi.phys_dim(*where[0])
        tags = tags_to_oset(tags)

        # allow a matrix to be reshaped into a tensor if it factorizes
        #     i.e. (4, 4) assumed to be two qubit gate -> (2, 2, 2, 2)
        G = maybe_factor_gate_into_tensor(G, dp, ng, where)

        site_ix = [psi.site_ind(i, j) for i, j in where]
        # new indices to join old physical sites to new gate
        bnds = [rand_uuid() for _ in range(ng)]
        reindex_map = dict(zip(site_ix, bnds))

        TG = Tensor(G, inds=site_ix + bnds, tags=tags, left_inds=bnds)

        if contract is False:
            #
            #       │   │      <- site_ix
            #       GGGGG
            #       │╱  │╱     <- bnds
            #     ──●───●──
            #      ╱   ╱
            #
            if propagate_tags:
                if propagate_tags == "register":
                    old_tags = oset(map(psi.site_tag, where))
                else:
                    old_tags = oset_union(
                        psi.tensor_map[tid].tags
                        for ind in site_ix
                        for tid in psi.ind_map[ind]
                    )

                if propagate_tags == "sites":
                    # use regex to take tags only matching e.g. 'I4,3'
                    rex = re.compile(psi.site_tag_id.format(r"\d+", r"\d+"))
                    old_tags = oset(filter(rex.match, old_tags))

                TG.modify(tags=TG.tags | old_tags)

            psi.reindex_(reindex_map)
            psi |= TG
            return psi

        if (contract is True) or (ng == 1):
            #
            #       │╱  │╱
            #     ──GGGGG──
            #      ╱   ╱
            #
            psi.reindex_(reindex_map)

            # get the sites that used to have the physical indices
            site_tids = psi._get_tids_from_inds(bnds, which="any")

            # pop the sites, contract, then re-add
            pts = [psi.pop_tensor(tid) for tid in site_tids]
            psi |= tensor_contract(*pts, TG)

            return psi

        # following are all based on splitting tensors to maintain structure
        ij_a, ij_b = where

        # parse the argument specifying how to find the path between
        # non-nearest neighbours
        if long_range_path_sequence is not None:
            # make sure we can index
            long_range_path_sequence = tuple(long_range_path_sequence)
            # if the first element is a str specifying move sequence, e.g.
            #     ('v', 'h')
            #     ('av', 'bv', 'ah', 'bh')  # using swaps
            manual_lr_path = not isinstance(long_range_path_sequence[0], str)
            # otherwise assume a path has been manually specified, e.g.
            #     ((1, 2), (2, 2), (2, 3), ... )
            #     (((1, 1), (1, 2)), ((4, 3), (3, 3)), ...)  # using swaps
        else:
            manual_lr_path = False

        # check if we are not nearest neighbour and need to swap first
        if long_range_use_swaps:
            if manual_lr_path:
                *swaps, final = long_range_path_sequence
            else:
                # find a swap path
                *swaps, final = gen_long_range_swap_path(
                    ij_a, ij_b, sequence=long_range_path_sequence
                )

            # move the sites together
            SWAP = get_swap(
                dp, dtype=get_dtype_name(G), backend=infer_backend(G)
            )
            for pair in swaps:
                psi.gate_(SWAP, pair, contract=contract, absorb="right")

            compress_opts["info"] = info
            compress_opts["contract"] = contract

            # perform actual gate also compressing etc on 'way back'
            psi.gate_(G, final, **compress_opts)

            compress_opts.setdefault("absorb", "both")
            for pair in reversed(swaps):
                psi.gate_(SWAP, pair, **compress_opts)

            return psi

        if manual_lr_path:
            string = long_range_path_sequence
        else:
            string = tuple(
                gen_long_range_path(*where, sequence=long_range_path_sequence)
            )

        # the tensors along this string, which will be updated
        original_ts = [psi[coo] for coo in string]

        # the len(string) - 1 indices connecting the string
        bonds_along = [
            next(iter(bonds(t1, t2))) for t1, t2 in pairwise(original_ts)
        ]

        if contract == "split":
            #
            #       │╱  │╱          │╱  │╱
            #     ──GGGGG──  ==>  ──G┄┄┄G──
            #      ╱   ╱           ╱   ╱
            #
            gate_string_split_(
                TG,
                where,
                string,
                original_ts,
                bonds_along,
                reindex_map,
                site_ix,
                info,
                **compress_opts,
            )

        elif contract == "reduce-split":
            #
            #       │   │             │ │
            #       GGGGG             GGG               │ │
            #       │╱  │╱   ==>     ╱│ │  ╱   ==>     ╱│ │  ╱          │╱  │╱
            #     ──●───●──       ──>─●─●─<──       ──>─GGG─<──  ==>  ──G┄┄┄G──
            #      ╱   ╱           ╱     ╱           ╱     ╱           ╱   ╱
            #    <QR> <LQ>                            <SVD>
            #
            gate_string_reduce_split_(
                TG,
                where,
                string,
                original_ts,
                bonds_along,
                reindex_map,
                site_ix,
                info,
                **compress_opts,
            )

        return psi

    gate_ = functools.partialmethod(gate, inplace=True)

    def compute_norm(
        self,
        layer_tags=("KET", "BRA"),
        **contract_opts,
    ):
        """Compute the norm of this vector via boundary contraction."""
        norm = self.make_norm(layer_tags=layer_tags)
        return norm.contract_boundary(layer_tags=layer_tags, **contract_opts)

    def compute_local_expectation(
        self,
        terms,
        max_bond=None,
        *,
        cutoff=1e-10,
        canonize=True,
        mode="mps",
        layer_tags=("KET", "BRA"),
        normalized=False,
        autogroup=True,
        contract_optimize="auto-hq",
        return_all=False,
        plaquette_envs=None,
        plaquette_map=None,
        **plaquette_env_options,
    ):
        r"""Compute the sum of many local expecations by essentially forming
        the reduced density matrix of all required plaquettes. If you supply
        ``normalized=True`` each expecation is locally normalized, which a) is
        usually more accurate and b) doesn't require a separate normalization
        boundary contraction.

        Parameters
        ----------
        terms : dict[tuple[tuple[int], array]
            A dictionary mapping site coordinates to raw operators, which will
            be supplied to
            :meth:`~quimb.tensor.tensor_2d.TensorNetwork2DVector.gate`. The
            keys should either be a single coordinate - ``(i, j)`` - describing
            a single site operator, or a pair of coordinates -
            ``((i_a, j_a), (i_b, j_b))`` describing a two site operator.
        max_bond : int, optional
            The maximum boundary dimension, AKA 'chi'. The default of ``None``
            means truncation is left purely to ``cutoff`` and is not
            recommended in 2D.
        cutoff : float, optional
            Cut-off value to used to truncate singular values in the boundary
            contraction.
        canonize : bool, optional
            Whether to sweep one way with canonization before compressing.
        mode : {'mps', 'full-bond'}, optional
            How to perform the compression on the boundary.
        layer_tags : None or sequence of str, optional
            If given, perform a multilayer contraction, contracting the inner
            sites in each layer into the boundary individually.
        normalized : bool, optional
            If True, normalize the value of each local expectation by the local
            norm: $\langle O_i \rangle = Tr[\rho_p O_i] / Tr[\rho_p]$.
        autogroup : bool, optional
            If ``True`` (the default), group terms into horizontal and vertical
            sets to be computed separately (usually more efficient) if
            possible.
        contract_optimize : str, optional
            Contraction path finder to use for contracting the local plaquette
            expectation (and optionally normalization).
        return_all : bool, optional
            Whether to the return all the values individually as a dictionary
            of coordinates to tuple[local_expectation, local_norm].
        plaquette_envs : None or dict, optional
            Supply precomputed plaquette environments.
        plaquette_map : None, dict, optional
            Supply the mapping of which plaquettes (denoted by
            ``((x0, y0), (dx, dy))``) to use for which coordinates, it will be
            calculated automatically otherwise.
        plaquette_env_options
            Supplied to :meth:`compute_plaquette_environments` to generate the
            plaquette environments, equivalent to approximately performing the
            partial trace.

        Returns
        -------
        scalar or dict
        """
        norm, ket, bra = self.make_norm(return_all=True)

        if plaquette_envs is None:
            plaquette_env_options["max_bond"] = max_bond
            plaquette_env_options["cutoff"] = cutoff
            plaquette_env_options["canonize"] = canonize
            plaquette_env_options["mode"] = mode
            plaquette_env_options["layer_tags"] = layer_tags

            plaquette_envs = dict()
            for x_bsz, y_bsz in calc_plaquette_sizes(terms.keys(), autogroup):
                plaquette_envs.update(
                    norm.compute_plaquette_environments(
                        x_bsz=x_bsz, y_bsz=y_bsz, **plaquette_env_options
                    )
                )

        if plaquette_map is None:
            # work out which plaquettes to use for which terms
            plaquette_map = calc_plaquette_map(plaquette_envs)

        # now group the terms into just the plaquettes we need
        plaq2coo = defaultdict(list)
        for where, G in terms.items():
            p = plaquette_map[where]
            plaq2coo[p].append((where, G))

        expecs = dict()
        for p in plaq2coo:
            # site tags for the plaquette
            sites = tuple(map(ket.site_tag, plaquette_to_sites(p)))

            # view the ket portion as 2d vector so we can gate it
            ket_local = ket.select_any(sites)
            ket_local.view_as_(TensorNetwork2DVector, like=self)
            bra_and_env = bra.select_any(sites) | plaquette_envs[p]

            with oe.shared_intermediates():
                # compute local estimation of norm for this plaquette
                if normalized:
                    norm_i0j0 = (ket_local | bra_and_env).contract(
                        all, optimize=contract_optimize
                    )
                else:
                    norm_i0j0 = None

                # for each local term on plaquette compute expectation
                for where, G in plaq2coo[p]:
                    expec_ij = (
                        ket_local.gate(G, where, contract=False) | bra_and_env
                    ).contract(all, optimize=contract_optimize)

                    expecs[where] = expec_ij, norm_i0j0

        if return_all:
            return expecs

        if normalized:
            return functools.reduce(add, (e / n for e, n in expecs.values()))

        return functools.reduce(add, (e for e, _ in expecs.values()))

    def normalize(
        self,
        max_bond=None,
        *,
        cutoff=1e-10,
        canonize=True,
        mode="mps",
        layer_tags=("KET", "BRA"),
        balance_bonds=False,
        equalize_norms=False,
        inplace=False,
        **contract_boundary_opts,
    ):
        """Normalize this PEPS.

        Parameters
        ----------
        max_bond : int, optional
            The maximum boundary dimension, AKA 'chi'. The default of ``None``
            means truncation is left purely to ``cutoff`` and is not
            recommended in 2D.
        cutoff : float, optional
            Cut-off value to used to truncate singular values in the boundary
            contraction.
        canonize : bool, optional
            Whether to sweep one way with canonization before compressing.
        mode : {'mps', 'full-bond'}, optional
            How to perform the compression on the boundary.
        layer_tags : None or sequence of str, optional
            If given, perform a multilayer contraction, contracting the inner
            sites in each layer into the boundary individually.
        balance_bonds : bool, optional
            Whether to balance the bonds after normalization, a form of
            conditioning.
        equalize_norms : bool, optional
            Whether to set all the tensor norms to the same value after
            normalization, another form of conditioning.
        inplace : bool, optional
            Whether to perform the normalization inplace or not.
        contract_boundary_opts
            Supplied to :meth:`contract_boundary`, by default, two layer
            contraction will be used.
        """
        contract_boundary_opts["max_bond"] = max_bond
        contract_boundary_opts["cutoff"] = cutoff
        contract_boundary_opts["canonize"] = canonize
        contract_boundary_opts["mode"] = mode
        contract_boundary_opts["layer_tags"] = layer_tags

        norm = self.make_norm()
        nfact = norm.contract_boundary(**contract_boundary_opts)

        n_ket = self.multiply_each(
            nfact ** (-1 / (2 * self.num_tensors)), inplace=inplace
        )

        if balance_bonds:
            n_ket.balance_bonds_()

        if equalize_norms:
            n_ket.equalize_norms_()

        return n_ket

    normalize_ = functools.partialmethod(normalize, inplace=True)


class TensorNetwork2DOperator(TensorNetwork2D, TensorNetworkGenOperator):
    """Mixin class for a 2D square lattice TN operator, i.e. one with both
    'upper' and 'lower' site (physical) indices.
    """

    _EXTRA_PROPS = (
        "_site_tag_id",
        "_x_tag_id",
        "_y_tag_id",
        "_Lx",
        "_Ly",
        "_upper_ind_id",
        "_lower_ind_id",
    )

    def reindex_lower_sites(self, new_id, where=None, inplace=False):
        """Update the lower site index labels to a new string specifier.

        Parameters
        ----------
        new_id : str
            A string with a format placeholder to accept an int, e.g.
            ``"ket{},{}"``.
        where : None or slice
            Which sites to update the index labels on. If ``None`` (default)
            all sites.
        inplace : bool
            Whether to reindex in place.
        """
        if where is None:
            where = self.gen_sites_present()
        return self.reindex(
            {self.lower_ind(i, j): new_id.format(i, j) for i, j in where},
            inplace=inplace,
        )

    reindex_lower_sites_ = functools.partialmethod(
        reindex_lower_sites, inplace=True
    )

    def reindex_upper_sites(self, new_id, where=None, inplace=False):
        """Update the upper site index labels to a new string specifier.

        Parameters
        ----------
        new_id : str
            A string with a format placeholder to accept an int, e.g.
            ``"ket{},{}"``.
        where : None or slice
            Which sites to update the index labels on. If ``None`` (default)
            all sites.
        inplace : bool
            Whether to reindex in place.
        """
        if where is None:
            where = self.gen_sites_present()
        return self.reindex(
            {self.upper_ind(i, j): new_id.format(i, j) for i, j in where},
            inplace=inplace,
        )

    reindex_upper_sites_ = functools.partialmethod(
        reindex_upper_sites, inplace=True
    )

    def lower_ind(self, i, j=None):
        """Get the lower index for a given site."""
        if j is None:
            i, j = i
        if not isinstance(i, str):
            i = i % self.Lx
        if not isinstance(j, str):
            j = j % self.Ly
        return self.lower_ind_id.format(i, j)

    def upper_ind(self, i, j=None):
        """Get the upper index for a given site."""
        if j is None:
            i, j = i
        if not isinstance(i, str):
            i = i % self.Lx
        if not isinstance(j, str):
            j = j % self.Ly
        return self.upper_ind_id.format(i, j)

    def phys_dim(self, i=0, j=0, which="upper"):
        """Get a physical index size of this 2D operator."""
        if which == "upper":
            return self[i, j].ind_size(self.upper_ind(i, j))

        if which == "lower":
            return self[i, j].ind_size(self.lower_ind(i, j))


class TensorNetwork2DFlat(TensorNetwork2D):
    """Mixin class for a 2D square lattice tensor network with a single tensor
    per site, for example, both PEPS and PEPOs.
    """

    _EXTRA_PROPS = (
        "_site_tag_id",
        "_x_tag_id",
        "_y_tag_id",
        "_Lx",
        "_Ly",
    )

    def bond(self, coo1, coo2):
        """Get the name of the index defining the bond between sites at
        ``coo1`` and ``coo2``.
        """
        (b_ix,) = self[coo1].bonds(self[coo2])
        return b_ix

    def bond_size(self, coo1, coo2):
        """Return the size of the bond between sites at ``coo1`` and ``coo2``."""
        b_ix = self.bond(coo1, coo2)
        return self[coo1].ind_size(b_ix)

    def expand_bond_dimension(
        self, new_bond_dim, inplace=True, bra=None, rand_strength=0.0
    ):
        """Increase the bond dimension of this flat, 2D, tensor network,
        padding the tensor data with either zeros or random entries.

        Parameters
        ----------
        new_bond_dim : int
            The new dimension. If smaller or equal to the current bond
            dimension nothing will happend.
        inplace : bool, optional
            Whether to expand in place (the default), or return a new TN.
        bra : TensorNetwork2DFlat, optional
            Expand this TN with the same data also, assuming it to be the
            conjugate, bra, TN.
        rand_strength : float, optional
            If greater than zero, pad the data arrays with gaussian noise of
            this strength.

        Returns
        -------
        tn : TensorNetwork2DFlat
        """
        tn = super().expand_bond_dimension(
            new_bond_dim=new_bond_dim,
            rand_strength=rand_strength,
            inplace=inplace,
        )

        if bra is not None:
            for coo in tn.gen_site_coos():
                bra[coo].modify(data=tn[coo].data.conj())

        return tn

    def compress(
        self,
        max_bond=None,
        cutoff=1e-10,
        equalize_norms=False,
        row_sweep="right",
        col_sweep="up",
        **compress_opts,
    ):
        """Compress all bonds in this flat 2D tensor network.

        Parameters
        ----------
        max_bond : int, optional
            The maximum boundary dimension, AKA 'chi'. The default of ``None``
            means truncation is left purely to ``cutoff`` and is not
            recommended in 2D.
        cutoff : float, optional
            Cut-off value to used to truncate singular values in the boundary
            contraction.
        compress_opts : None or dict, optional
            Supplied to :meth:`compress_between`.
        """
        compress_opts.setdefault("absorb", "both")
        for i in range(self.Lx):
            self.compress_row(
                i,
                sweep=row_sweep,
                max_bond=max_bond,
                cutoff=cutoff,
                equalize_norms=equalize_norms,
                compress_opts=compress_opts,
            )
        for j in range(self.Ly):
            self.compress_column(
                j,
                sweep=col_sweep,
                max_bond=max_bond,
                cutoff=cutoff,
                equalize_norms=equalize_norms,
                compress_opts=compress_opts,
            )


class PEPS(TensorNetwork2DVector, TensorNetwork2DFlat):
    r"""Projected Entangled Pair States object (2D)::


                         ...
             │    │    │    │    │    │
             ●────●────●────●────●────●──
            ╱│   ╱│   ╱│   ╱│   ╱│   ╱│
             │    │    │    │    │    │
             ●────●────●────●────●────●──
            ╱│   ╱│   ╱│   ╱│   ╱│   ╱│
             │    │    │    │    │    │   ...
             ●────●────●────●────●────●──
            ╱│   ╱│   ╱│   ╱│   ╱│   ╱│
             │    │    │    │    │    │
             ●────●────●────●────●────●──
            ╱    ╱    ╱    ╱    ╱    ╱

    Parameters
    ----------
    arrays : sequence of sequence of array
        The core tensor data arrays.
    shape : str, optional
        Which order the dimensions of the arrays are stored in, the default
        ``'urdlp'`` stands for ('up', 'right', 'down', 'left', 'physical').
        Arrays on the edge of lattice are assumed to be missing the
        corresponding dimension.
    tags : set[str], optional
        Extra global tags to add to the tensor network.
    site_ind_id : str, optional
        String specifier for naming convention of site indices.
    site_tag_id : str, optional
        String specifier for naming convention of site tags.
    x_tag_id : str, optional
        String specifier for naming convention of row ('x') tags.
    y_tag_id : str, optional
        String specifier for naming convention of column ('y') tags.
    """

    _EXTRA_PROPS = (
        "_site_tag_id",
        "_x_tag_id",
        "_y_tag_id",
        "_Lx",
        "_Ly",
        "_site_ind_id",
    )

    def __init__(
        self,
        arrays,
        *,
        shape="urdlp",
        tags=None,
        site_ind_id="k{},{}",
        site_tag_id="I{},{}",
        x_tag_id="X{}",
        y_tag_id="Y{}",
        **tn_opts,
    ):
        if isinstance(arrays, PEPS):
            super().__init__(arrays)
            return

        tags = tags_to_oset(tags)
        self._site_ind_id = site_ind_id
        self._site_tag_id = site_tag_id
        self._x_tag_id = x_tag_id
        self._y_tag_id = y_tag_id

        arrays = tuple(tuple(x for x in xs) for xs in arrays)
        self._Lx = len(arrays)
        self._Ly = len(arrays[0])
        tensors = []

        # cache for both creating and retrieving indices
        ix = defaultdict(rand_uuid)

        for i, j in self.gen_site_coos():
            array = arrays[i][j]

            # figure out if we need to transpose the arrays from some order
            #     other than up right down left physical
            array_order = shape
            if i == self.Lx - 1:
                array_order = array_order.replace("u", "")
            if j == self.Ly - 1:
                array_order = array_order.replace("r", "")
            if i == 0:
                array_order = array_order.replace("d", "")
            if j == 0:
                array_order = array_order.replace("l", "")

            # allow convention of missing bonds to be singlet dimensions
            if len(array.shape) != len(array_order):
                array = do("squeeze", array)

            transpose_order = tuple(
                array_order.find(x) for x in "urdlp" if x in array_order
            )
            if transpose_order != tuple(range(len(array_order))):
                array = do("transpose", array, transpose_order)

            # get the relevant indices corresponding to neighbours
            inds = []
            if "u" in array_order:
                inds.append(ix[(i, j), (i + 1, j)])
            if "r" in array_order:
                inds.append(ix[(i, j), (i, j + 1)])
            if "d" in array_order:
                inds.append(ix[(i - 1, j), (i, j)])
            if "l" in array_order:
                inds.append(ix[(i, j - 1), (i, j)])
            inds.append(self.site_ind(i, j))

            # mix site, row, column and global tags

            ij_tags = tags | oset(
                (self.site_tag(i, j), self.x_tag(i), self.y_tag(j))
            )

            # create the site tensor!
            tensors.append(Tensor(data=array, inds=inds, tags=ij_tags))

        super().__init__(tensors, virtual=True, **tn_opts)

    @classmethod
    def from_fill_fn(cls, fill_fn, Lx, Ly, bond_dim, phys_dim=2, **peps_opts):
        """Create a 2D PEPS from a filling function with signature
        ``fill_fn(shape)``.

        Parameters
        ----------
        Lx : int
            The number of rows.
        Ly : int
            The number of columns.
        bond_dim : int
            The bond dimension.
        physical : int, optional
            The physical index dimension.
        peps_opts
            Supplied to :class:`~quimb.tensor.tensor_2d.PEPS`.

        Returns
        -------
        psi : PEPS
        """
        arrays = [[None for _ in range(Ly)] for _ in range(Lx)]

        for i, j in product(range(Lx), range(Ly)):
            shape = []
            if i != Lx - 1:  # bond up
                shape.append(bond_dim)
            if j != Ly - 1:  # bond right
                shape.append(bond_dim)
            if i != 0:  # bond down
                shape.append(bond_dim)
            if j != 0:  # bond left
                shape.append(bond_dim)
            shape.append(phys_dim)

            arrays[i][j] = fill_fn(shape)

        return cls(arrays, **peps_opts)

    @classmethod
    def empty(cls, Lx, Ly, bond_dim, phys_dim=2, like="numpy", **peps_opts):
        """Create an empty 2D PEPS.

        Parameters
        ----------
        Lx : int
            The number of rows.
        Ly : int
            The number of columns.
        bond_dim : int
            The bond dimension.
        physical : int, optional
            The physical index dimension.
        peps_opts
            Supplied to :class:`~quimb.tensor.tensor_2d.PEPS`.

        Returns
        -------
        psi : PEPS

        See Also
        --------
        PEPS.from_fill_fn
        """
        return cls.from_fill_fn(
            lambda shape: do("zeros", shape, like=like),
            Lx,
            Ly,
            bond_dim,
            phys_dim,
            **peps_opts,
        )

    @classmethod
    def ones(cls, Lx, Ly, bond_dim, phys_dim=2, like="numpy", **peps_opts):
        """Create a 2D PEPS whose tensors are filled with ones.

        Parameters
        ----------
        Lx : int
            The number of rows.
        Ly : int
            The number of columns.
        bond_dim : int
            The bond dimension.
        physical : int, optional
            The physical index dimension.
        peps_opts
            Supplied to :class:`~quimb.tensor.tensor_2d.PEPS`.

        Returns
        -------
        psi : PEPS

        See Also
        --------
        PEPS.from_fill_fn
        """
        return cls.from_fill_fn(
            lambda shape: do("ones", shape, like=like),
            Lx,
            Ly,
            bond_dim,
            phys_dim,
            **peps_opts,
        )

    @classmethod
    def rand(
        cls, Lx, Ly, bond_dim, phys_dim=2, dtype=float, seed=None, **peps_opts
    ):
        """Create a random (un-normalized) PEPS.

        Parameters
        ----------
        Lx : int
            The number of rows.
        Ly : int
            The number of columns.
        bond_dim : int
            The bond dimension.
        physical : int, optional
            The physical index dimension.
        dtype : dtype, optional
            The dtype to create the arrays with, default is real double.
        seed : int, optional
            A random seed.
        peps_opts
            Supplied to :class:`~quimb.tensor.tensor_2d.PEPS`.

        Returns
        -------
        psi : PEPS

        See Also
        --------
        PEPS.from_fill_fn
        """
        if seed is not None:
            seed_rand(seed)

        def fill_fn(shape):
            return ops.sensibly_scale(
                ops.sensibly_scale(randn(shape, dtype=dtype))
            )

        return cls.from_fill_fn(
            fill_fn, Lx, Ly, bond_dim, phys_dim, **peps_opts
        )

    def add_PEPS(self, other, inplace=False):
        """Add this PEPS with another."""
        if (self.Lx, self.Ly) != (other.Lx, other.Ly):
            raise ValueError("PEPS must be the same size.")

        peps = self if inplace else self.copy()
        for coo in peps.gen_site_coos():
            t1, t2 = peps[coo], other[coo]

            if set(t1.inds) != set(t2.inds):
                # Need to use bonds to match indices
                reindex_map = {}
                i, j = coo
                if i > 0:
                    pair = ((i - 1, j), (i, j))
                    reindex_map[other.bond(*pair)] = peps.bond(*pair)
                if i < self.Lx - 1:
                    pair = ((i, j), (i + 1, j))
                    reindex_map[other.bond(*pair)] = peps.bond(*pair)
                if j > 0:
                    pair = ((i, j - 1), (i, j))
                    reindex_map[other.bond(*pair)] = peps.bond(*pair)
                if j < self.Ly - 1:
                    pair = ((i, j), (i, j + 1))
                    reindex_map[other.bond(*pair)] = peps.bond(*pair)

                t2 = t2.reindex(reindex_map)

            t1.direct_product_(t2, sum_inds=peps.site_ind(*coo))

        return peps

    add_PEPS_ = functools.partialmethod(add_PEPS, inplace=True)

    def __add__(self, other):
        """PEPS addition."""
        return self.add_PEPS(other, inplace=False)

    def __iadd__(self, other):
        """In-place PEPS addition."""
        return self.add_PEPS(other, inplace=True)

    def show(self):
        """Print a unicode schematic of this PEPS and its bond dimensions."""
        show_2d(self, show_lower=True)


class PEPO(TensorNetwork2DOperator, TensorNetwork2DFlat):
    r"""Projected Entangled Pair Operator object::


                         ...
             │╱   │╱   │╱   │╱   │╱   │╱
             ●────●────●────●────●────●──
            ╱│   ╱│   ╱│   ╱│   ╱│   ╱│
             │╱   │╱   │╱   │╱   │╱   │╱
             ●────●────●────●────●────●──
            ╱│   ╱│   ╱│   ╱│   ╱│   ╱│
             │╱   │╱   │╱   │╱   │╱   │╱   ...
             ●────●────●────●────●────●──
            ╱│   ╱│   ╱│   ╱│   ╱│   ╱│
             │╱   │╱   │╱   │╱   │╱   │╱
             ●────●────●────●────●────●──
            ╱    ╱    ╱    ╱    ╱    ╱

    Parameters
    ----------
    arrays : sequence of sequence of array
        The core tensor data arrays.
    shape : str, optional
        Which order the dimensions of the arrays are stored in, the default
        ``'urdlbk'`` stands for ('up', 'right', 'down', 'left', 'bra', 'ket').
        Arrays on the edge of lattice are assumed to be missing the
        corresponding dimension.
    tags : set[str], optional
        Extra global tags to add to the tensor network.
    upper_ind_id : str, optional
        String specifier for naming convention of upper site indices.
    lower_ind_id : str, optional
        String specifier for naming convention of lower site indices.
    site_tag_id : str, optional
        String specifier for naming convention of site tags.
    x_tag_id : str, optional
        String specifier for naming convention of row ('x') tags.
    y_tag_id : str, optional
        String specifier for naming convention of column ('y') tags.
    """

    _EXTRA_PROPS = (
        "_site_tag_id",
        "_x_tag_id",
        "_y_tag_id",
        "_Lx",
        "_Ly",
        "_upper_ind_id",
        "_lower_ind_id",
    )

    def __init__(
        self,
        arrays,
        *,
        shape="urdlbk",
        tags=None,
        upper_ind_id="k{},{}",
        lower_ind_id="b{},{}",
        site_tag_id="I{},{}",
        x_tag_id="X{}",
        y_tag_id="Y{}",
        **tn_opts,
    ):
        if isinstance(arrays, PEPO):
            super().__init__(arrays)
            return

        tags = tags_to_oset(tags)
        self._upper_ind_id = upper_ind_id
        self._lower_ind_id = lower_ind_id
        self._site_tag_id = site_tag_id
        self._x_tag_id = x_tag_id
        self._y_tag_id = y_tag_id

        arrays = tuple(tuple(x for x in xs) for xs in arrays)
        self._Lx = len(arrays)
        self._Ly = len(arrays[0])
        tensors = []

        # cache for both creating and retrieving indices
        ix = defaultdict(rand_uuid)

        for i, j in product(range(self.Lx), range(self.Ly)):
            array = arrays[i][j]

            # figure out if we need to transpose the arrays from some order
            #     other than up right down left physical
            array_order = shape
            if i == self.Lx - 1:
                array_order = array_order.replace("u", "")
            if j == self.Ly - 1:
                array_order = array_order.replace("r", "")
            if i == 0:
                array_order = array_order.replace("d", "")
            if j == 0:
                array_order = array_order.replace("l", "")

            # allow convention of missing bonds to be singlet dimensions
            if len(array.shape) != len(array_order):
                array = do("squeeze", array)

            transpose_order = tuple(
                array_order.find(x) for x in "urdlbk" if x in array_order
            )
            if transpose_order != tuple(range(len(array_order))):
                array = do("transpose", array, transpose_order)

            # get the relevant indices corresponding to neighbours
            inds = []
            if "u" in array_order:
                inds.append(ix[(i + 1, j), (i, j)])
            if "r" in array_order:
                inds.append(ix[(i, j), (i, j + 1)])
            if "d" in array_order:
                inds.append(ix[(i, j), (i - 1, j)])
            if "l" in array_order:
                inds.append(ix[(i, j - 1), (i, j)])
            inds.append(self.lower_ind(i, j))
            inds.append(self.upper_ind(i, j))

            # mix site, row, column and global tags
            ij_tags = tags | oset(
                (self.site_tag(i, j), self.x_tag(i), self.y_tag(j))
            )

            # create the site tensor!
            tensors.append(Tensor(data=array, inds=inds, tags=ij_tags))

        super().__init__(tensors, virtual=True, **tn_opts)

    @classmethod
    def rand(
        cls,
        Lx,
        Ly,
        bond_dim,
        phys_dim=2,
        herm=False,
        dtype=float,
        seed=None,
        **pepo_opts,
    ):
        """Create a random PEPO.

        Parameters
        ----------
        Lx : int
            The number of rows.
        Ly : int
            The number of columns.
        bond_dim : int
            The bond dimension.
        physical : int, optional
            The physical index dimension.
        herm : bool, optional
            Whether to symmetrize the tensors across the physical bonds to make
            the overall operator hermitian.
        dtype : dtype, optional
            The dtype to create the arrays with, default is real double.
        seed : int, optional
            A random seed.
        pepo_opts
            Supplied to :class:`~quimb.tensor.tensor_2d.PEPO`.

        Returns
        -------
        X : PEPO
        """
        if seed is not None:
            seed_rand(seed)

        arrays = [[None for _ in range(Ly)] for _ in range(Lx)]

        for i, j in product(range(Lx), range(Ly)):
            shape = []
            if i != Lx - 1:  # bond up
                shape.append(bond_dim)
            if j != Ly - 1:  # bond right
                shape.append(bond_dim)
            if i != 0:  # bond down
                shape.append(bond_dim)
            if j != 0:  # bond left
                shape.append(bond_dim)
            shape.append(phys_dim)
            shape.append(phys_dim)

            X = ops.sensibly_scale(
                ops.sensibly_scale(randn(shape, dtype=dtype))
            )

            if herm:
                new_order = list(range(len(shape)))
                new_order[-2], new_order[-1] = new_order[-1], new_order[-2]
                X = (do("conj", X) + do("transpose", X, new_order)) / 2

            arrays[i][j] = X

        return cls(arrays, **pepo_opts)

    rand_herm = functools.partialmethod(rand, herm=True)

    def add_PEPO(self, other, inplace=False):
        """Add this PEPO with another."""
        if (self.Lx, self.Ly) != (other.Lx, other.Ly):
            raise ValueError("PEPOs must be the same size.")

        pepo = self if inplace else self.copy()
        for coo in pepo.gen_site_coos():
            t1, t2 = pepo[coo], other[coo]

            if set(t1.inds) != set(t2.inds):
                # Need to use bonds to match indices
                reindex_map = {}
                i, j = coo
                if i > 0:
                    pair = ((i - 1, j), (i, j))
                    reindex_map[other.bond(*pair)] = pepo.bond(*pair)
                if i < self.Lx - 1:
                    pair = ((i, j), (i + 1, j))
                    reindex_map[other.bond(*pair)] = pepo.bond(*pair)
                if j > 0:
                    pair = ((i, j - 1), (i, j))
                    reindex_map[other.bond(*pair)] = pepo.bond(*pair)
                if j < self.Ly - 1:
                    pair = ((i, j), (i, j + 1))
                    reindex_map[other.bond(*pair)] = pepo.bond(*pair)

                t2 = t2.reindex(reindex_map)

            sum_inds = (pepo.upper_ind(*coo), pepo.lower_ind(*coo))
            t1.direct_product_(t2, sum_inds=sum_inds)

        return pepo

    add_PEPO_ = functools.partialmethod(add_PEPO, inplace=True)

    def __add__(self, other):
        """PEPO addition."""
        return self.add_PEPO(other, inplace=False)

    def __iadd__(self, other):
        """In-place PEPO addition."""
        return self.add_PEPO(other, inplace=True)

    _apply_peps = tensor_network_apply_op_vec

    def apply(self, other, compress=False, **compress_opts):
        """Act with this PEPO on ``other``, returning a new TN like ``other``
        with the same outer indices.

        Parameters
        ----------
        other : PEPS
            The TN to act on.
        compress : bool, optional
            Whether to compress the resulting TN.
        compress_opts
            Supplied to
            :meth:`~quimb.tensor.tensor_2d.TensorNetwork2DFlat.compress`.

        Returns
        -------
        TensorNetwork2DFlat
        """
        if isinstance(other, PEPS):
            return self._apply_peps(other, compress=compress, **compress_opts)

        raise TypeError("Can only apply PEPO to PEPS.")

    def show(self):
        """Print a unicode schematic of this PEPO and its bond dimensions."""
        show_2d(self, show_lower=True, show_upper=True)


def show_2d(tn_2d, show_lower=False, show_upper=False):
    """Base function for printing a unicode schematic of flat 2D TNs."""

    lb = "╱" if show_lower else " "
    ub = "╱" if show_upper else " "

    line0 = " " + (f" {ub}{{:^3}}" * (tn_2d.Ly - 1)) + f" {ub}"
    bszs = [tn_2d.bond_size((0, j), (0, j + 1)) for j in range(tn_2d.Ly - 1)]

    lines = [line0.format(*bszs)]

    for i in range(tn_2d.Lx - 1):
        lines.append(" ●" + ("━━━━●" * (tn_2d.Ly - 1)))

        # vertical bonds
        lines.append(f"{lb}┃{{:<3}}" * tn_2d.Ly)
        bszs = [tn_2d.bond_size((i, j), (i + 1, j)) for j in range(tn_2d.Ly)]
        lines[-1] = lines[-1].format(*bszs)

        # horizontal bonds bottom
        lines.append(" ┃" + (f"{ub}{{:^3}}┃" * (tn_2d.Ly - 1)) + f"{ub}")
        bszs = [
            tn_2d.bond_size((i + 1, j), (i + 1, j + 1))
            for j in range(tn_2d.Ly - 1)
        ]
        lines[-1] = lines[-1].format(*bszs)

    lines.append(" ●" + ("━━━━●" * (tn_2d.Ly - 1)))
    lines.append(f"{lb}    " * tn_2d.Ly)

    print_multi_line(*lines)


def calc_plaquette_sizes(coo_groups, autogroup=True):
    """Find a sequence of plaquette blocksizes that will cover all the terms
    (coordinate pairs) in ``pairs``.

    Parameters
    ----------
    coo_groups : sequence of tuple[tuple[int]] or tuple[int]
        The sequence of 2D coordinates pairs describing terms. Each should
        either be a single 2D coordinate or a sequence of 2D coordinates.
    autogroup : bool, optional
        Whether to return the minimal sequence of blocksizes that will cover
        all terms or merge them into a single ``((x_bsz, y_bsz),)``.

    Return
    ------
    bszs : tuple[tuple[int]]
        Pairs of blocksizes.

    Examples
    --------

    Some nearest neighbour interactions:

        >>> H2 = {None: qu.ham_heis(2)}
        >>> ham = qtn.LocalHam2D(10, 10, H2)
        >>> calc_plaquette_sizes(ham.terms.keys())
        ((1, 2), (2, 1))

        >>> calc_plaquette_sizes(ham.terms.keys(), autogroup=False)
        ((2, 2),)

    If we add any next nearest neighbour interaction then we are going to
    need the (2, 2) blocksize in any case:

        >>> H2[(1, 1), (2, 2)] = 0.5 * qu.ham_heis(2)
        >>> ham = qtn.LocalHam2D(10, 10, H2)
        >>> calc_plaquette_sizes(ham.terms.keys())
        ((2, 2),)

    If we add longer range interactions (non-diagonal next nearest) we again
    can benefit from multiple plaquette blocksizes:

        >>> H2[(1, 1), (1, 3)] = 0.25 * qu.ham_heis(2)
        >>> H2[(1, 1), (3, 1)] = 0.25 * qu.ham_heis(2)
        >>> ham = qtn.LocalHam2D(10, 10, H2)
        >>> calc_plaquette_sizes(ham.terms.keys())
        ((1, 3), (2, 2), (3, 1))

    Or choose the plaquette blocksize that covers all terms:

        >>> calc_plaquette_sizes(ham.terms.keys(), autogroup=False)
        ((3, 3),)

    """
    # get the rectangular size of each coordinate pair
    #     e.g. ((1, 1), (2, 1)) -> (2, 1)
    #          ((4, 5), (6, 7)) -> (3, 3) etc.
    bszs = set()
    for coos in coo_groups:
        if is_lone_coo(coos):
            bszs.add((1, 1))
            continue
        xs, ys = zip(*coos)
        xsz = max(xs) - min(xs) + 1
        ysz = max(ys) - min(ys) + 1
        bszs.add((xsz, ysz))

    # remove block size pairs that can be contained in another block pair size
    #     e.g. {(1, 2), (2, 1), (2, 2)} -> ((2, 2),)
    bszs = tuple(
        sorted(
            b
            for b in bszs
            if not any(
                (b[0] <= b2[0]) and (b[1] <= b2[1]) for b2 in bszs - {b}
            )
        )
    )

    # return each plaquette size separately
    if autogroup:
        return bszs

    # else choose a single blocksize that will cover all terms
    #     e.g. ((1, 2), (3, 2)) -> ((3, 2),)
    #          ((1, 2), (2, 1)) -> ((2, 2),)
    return (tuple(map(max, zip(*bszs))),)


def plaquette_to_sites(p):
    """Turn a plaquette ``((i0, j0), (di, dj))`` into the sites it contains.

    Examples
    --------

        >>> plaquette_to_sites([(3, 4), (2, 2)])
        ((3, 4), (3, 5), (4, 4), (4, 5))
    """
    (i0, j0), (di, dj) = p
    return tuple(
        (i, j) for i in range(i0, i0 + di) for j in range(j0, j0 + dj)
    )


def calc_plaquette_map(plaquettes):
    """Generate a dictionary of all the coordinate pairs in ``plaquettes``
    mapped to the 'best' (smallest) rectangular plaquette that contains them.

    Examples
    --------

    Consider 4 sites, with one 2x2 plaquette and two vertical (2x1)
    and horizontal (1x2) plaquettes each:

        >>> plaquettes = [
        ...     # 2x2 plaquette covering all sites
        ...     ((0, 0), (2, 2)),
        ...     # horizontal plaquettes
        ...     ((0, 0), (1, 2)),
        ...     ((1, 0), (1, 2)),
        ...     # vertical plaquettes
        ...     ((0, 0), (2, 1)),
        ...     ((0, 1), (2, 1)),
        ... ]

        >>> calc_plaquette_map(plaquettes)
        {((0, 0), (0, 1)): ((0, 0), (1, 2)),
         ((0, 0), (1, 0)): ((0, 0), (2, 1)),
         ((0, 0), (1, 1)): ((0, 0), (2, 2)),
         ((0, 1), (1, 0)): ((0, 0), (2, 2)),
         ((0, 1), (1, 1)): ((0, 1), (2, 1)),
         ((1, 0), (1, 1)): ((1, 0), (1, 2))}

    Now every of the size coordinate pairs is mapped to one of the plaquettes,
    but to the smallest one that contains it. So the 2x2 plaquette (specified
    by ``((0, 0), (2, 2))``) would only used for diagonal terms here.
    """
    # sort in descending total plaquette size
    plqs = sorted(plaquettes, key=lambda p: (-p[1][0] * p[1][1], p))

    mapping = dict()
    for p in plqs:
        sites = plaquette_to_sites(p)
        for site in sites:
            mapping[site] = p
        # this will generate all coordinate pairs with ij_a < ij_b
        for ij_a, ij_b in combinations(sites, 2):
            mapping[ij_a, ij_b] = p

    return mapping


def gen_long_range_path(ij_a, ij_b, sequence=None):
    """Generate a string of coordinates, in order, from ``ij_a`` to ``ij_b``.

    Parameters
    ----------
    ij_a : (int, int)
        Coordinate of site 'a'.
    ij_b : (int, int)
        Coordinate of site 'b'.
    sequence : None, iterable of {'v', 'h'}, or 'random', optional
        What order to cycle through and try and perform moves in, 'v', 'h'
        standing for move vertically and horizontally respectively. The default
        is ``('v', 'h')``.

    Returns
    -------
    generator[tuple[int]]
        The path, each element is a single coordinate.
    """
    ia, ja = ij_a
    ib, jb = ij_b
    di = ib - ia
    dj = jb - ja

    # nearest neighbour
    if abs(di) + abs(dj) == 1:
        yield ij_a
        yield ij_b
        return

    if sequence is None:
        poss_moves = cycle(("v", "h"))
    elif sequence == "random":
        poss_moves = (random.choice("vh") for _ in count())
    else:
        poss_moves = cycle(sequence)

    yield ij_a

    for move in poss_moves:
        if abs(di) + abs(dj) == 1:
            yield ij_b
            return

        if (move == "v") and (di != 0):
            # move a vertically
            istep = min(max(di, -1), +1)
            new_ij_a = (ia + istep, ja)
            yield new_ij_a
            ij_a = new_ij_a
            ia += istep
            di -= istep
        elif (move == "h") and (dj != 0):
            # move a horizontally
            jstep = min(max(dj, -1), +1)
            new_ij_a = (ia, ja + jstep)
            yield new_ij_a
            ij_a = new_ij_a
            ja += jstep
            dj -= jstep


def gen_long_range_swap_path(ij_a, ij_b, sequence=None):
    """Generate the coordinates or a series of swaps that would bring ``ij_a``
    and ``ij_b`` together.

    Parameters
    ----------
    ij_a : (int, int)
        Coordinate of site 'a'.
    ij_b : (int, int)
        Coordinate of site 'b'.
    sequence : None, it of {'av', 'bv', 'ah', 'bh'}, or 'random', optional
        What order to cycle through and try and perform moves in, 'av', 'bv',
        'ah', 'bh' standing for move 'a' vertically, 'b' vertically, 'a'
        horizontally', and 'b' horizontally respectively. The default is
        ``('av', 'bv', 'ah', 'bh')``.

    Returns
    -------
    generator[tuple[tuple[int]]]
        The path, each element is two coordinates to swap.
    """
    ia, ja = ij_a
    ib, jb = ij_b
    di = ib - ia
    dj = jb - ja

    # nearest neighbour
    if abs(di) + abs(dj) == 1:
        yield (ij_a, ij_b)
        return

    if sequence is None:
        poss_moves = cycle(("av", "bv", "ah", "bh"))
    elif sequence == "random":
        poss_moves = (random.choice(("av", "bv", "ah", "bh")) for _ in count())
    else:
        poss_moves = cycle(sequence)

    for move in poss_moves:
        if (move == "av") and (di != 0):
            # move a vertically
            istep = min(max(di, -1), +1)
            new_ij_a = (ia + istep, ja)
            yield (ij_a, new_ij_a)
            ij_a = new_ij_a
            ia += istep
            di -= istep

        elif (move == "bv") and (di != 0):
            # move b vertically
            istep = min(max(di, -1), +1)
            new_ij_b = (ib - istep, jb)
            # need to make sure final gate is applied correct way
            if new_ij_b == ij_a:
                yield (ij_a, ij_b)
            else:
                yield (ij_b, new_ij_b)
            ij_b = new_ij_b
            ib -= istep
            di -= istep

        elif (move == "ah") and (dj != 0):
            # move a horizontally
            jstep = min(max(dj, -1), +1)
            new_ij_a = (ia, ja + jstep)
            yield (ij_a, new_ij_a)
            ij_a = new_ij_a
            ja += jstep
            dj -= jstep

        elif (move == "bh") and (dj != 0):
            # move b horizontally
            jstep = min(max(dj, -1), +1)
            new_ij_b = (ib, jb - jstep)
            # need to make sure final gate is applied correct way
            if new_ij_b == ij_a:
                yield (ij_a, ij_b)
            else:
                yield (ij_b, new_ij_b)
            ij_b = new_ij_b
            jb -= jstep
            dj -= jstep

        if di == dj == 0:
            return


def swap_path_to_long_range_path(swap_path, ij_a):
    """Generates the ordered long-range path - a sequence of coordinates - from
    a (long-range) swap path - a sequence of coordinate pairs.
    """
    sites = set(chain(*swap_path))
    return sorted(sites, key=lambda ij_b: manhattan_distance(ij_a, ij_b))


@functools.lru_cache(8)
def get_swap(dp, dtype, backend):
    SWAP = swap(dp, dtype=dtype)
    return do("array", SWAP, like=backend)
