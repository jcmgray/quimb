"""Classes and algorithms related to 2D tensor networks."""

import functools
import warnings
from collections import defaultdict
from itertools import combinations, cycle, product
from numbers import Integral

import autoray as ar

from ...gen.rand import get_rand_fill_fn
from ...utils import (
    check_opt,
    deprecated,
    ensure_dict,
    pairwise,
    print_multi_line,
)
from ...utils import progbar as Progbar
from .. import decomp
from ..environments import (
    find_1d_block,
    gen_compressed_environments,
    gen_exact_environments,
)
from ..tensor_core import (
    Tensor,
    TensorNetwork,
    bonds,
    bonds_size,
    oset,
    parse_site_tag_groups,
    rand_uuid,
    tags_to_oset,
)
from ..tnag.core import (
    LatticeBondMap,
    TensorNetworkGen,
    TensorNetworkGenOperator,
    TensorNetworkGenVector,
    expectations_from_rhos,
    partial_traces_from_environment,
    tensor_network_ag_sum,
)


def nearest_neighbors(coo):
    i, j = coo
    return ((i - 1, j), (i, j - 1), (i, j + 1), (i + 1, j))


def gen_2d_bonds(Lx, Ly, steppers=None, coo_filter=None, cyclic=False):
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

    try:
        cyclic_x, cyclic_y = cyclic
    except (TypeError, ValueError):
        cyclic_x = cyclic_y = cyclic

    def _maybe_wrap_coo(w, Lw, cyclic):
        if 0 <= w < Lw:
            return w
        if cyclic:
            return w % Lw
        return None

    for i, j in product(range(Lx), range(Ly)):
        if (coo_filter is None) or coo_filter(i, j):
            for stepper in steppers:
                i2, j2 = stepper(i, j)

                i2 = _maybe_wrap_coo(i2, Lx, cyclic_x)
                j2 = _maybe_wrap_coo(j2, Ly, cyclic_y)

                if (i2 is not None) and (j2 is not None):
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
            self.is_cyclic_x = tn.is_cyclic_x
            self.is_cyclic_y = tn.is_cyclic_y
        else:  # 'y'
            # -> rotate 90deg
            self.imin, self.imax = sorted(yrange)
            self.jmin, self.jmax = sorted(xrange)
            self.y_tag = tn.x_tag
            self.x_tag = tn.y_tag
            self.site_tag = lambda i, j: tn.site_tag(j, i)
            self.is_cyclic_x = tn.is_cyclic_y
            self.is_cyclic_y = tn.is_cyclic_x

        if "min" in self.from_which:
            # -> sweeps are increasing
            self.sweep = range(self.imin, self.imax + 1, +stepsize)
            self.istep = +stepsize
        else:  # 'max'
            # -> sweeps are decreasing
            self.sweep = range(self.imax, self.imin - 1, -stepsize)
            self.istep = -stepsize

    @functools.cached_property
    def sweep_other(self):
        return range(self.jmin, self.jmax + 1)

    def rotate(self, a, b):
        """Rotate a pair such as a coordinate ``(i, j)`` or a block size
        between the real and rotated frames. This is its own inverse.
        """
        return (a, b) if self.plane == "x" else (b, a)

    @functools.cached_property
    def cyclic_x(self):
        return self.is_cyclic_x(
            (self.jmin + self.jmax) // 2,
            self.imin,
            self.imax,
        )

    @functools.cached_property
    def cyclic_y(self):
        return self.is_cyclic_y(
            (self.imin + self.imax) // 2,
            self.jmin,
            self.jmax,
        )

    def get_jnext(self, j):
        if j == self.jmax:
            if self.cyclic_y:
                # wrap around
                return self.jmin
            # no more steps
            return None
        # normal step
        return j + 1

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
    if isinstance(sequence, str) and sequence in BOUNDARY_SEQUENCE_VALID:
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

        * the 'up' (x+) bond is coordinates ``(i, j), (i + 1, j)``
        * the 'down' (x-) bond is coordinates ``(i, j), (i - 1, j)``
        * the 'right' (y+) bond is coordinates ``(i, j), (i, j + 1)``
        * the 'left' (y-) bond is coordinates ``(i, j), (i, j - 1)``

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

    def has_site(self, site):
        """Whether ``site`` is a valid ``(i, j)`` coordinate of this 2D
        tensor network, with ``0 <= i < Lx`` and ``0 <= j < Ly``.
        """
        if not isinstance(site, tuple) or len(site) != 2:
            return False
        i, j = site
        return (
            isinstance(i, Integral)
            and isinstance(j, Integral)
            and (0 <= i < self.Lx)
            and (0 <= j < self.Ly)
        )

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
            cyclic=(self.is_cyclic_x(), self.is_cyclic_y()),
        )

    def gen_horizontal_bond_coos(self):
        """Generate all coordinate pairs like ``(i, j), (i, j + 1)``."""
        return gen_2d_bonds(
            self.Lx,
            self.Ly,
            steppers=[
                lambda i, j: (i, j + 1),
            ],
            cyclic=(self.is_cyclic_x(), self.is_cyclic_y()),
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
            cyclic=(self.is_cyclic_x(), self.is_cyclic_y()),
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
            cyclic=(self.is_cyclic_x(), self.is_cyclic_y()),
        )

    def gen_vertical_bond_coos(self):
        """Generate all coordinate pairs like ``(i, j), (i + 1, j)``."""
        return gen_2d_bonds(
            self.Lx,
            self.Ly,
            steppers=[
                lambda i, j: (i + 1, j),
            ],
            cyclic=(self.is_cyclic_x(), self.is_cyclic_y()),
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
            cyclic=(self.is_cyclic_x(), self.is_cyclic_y()),
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
            cyclic=(self.is_cyclic_x(), self.is_cyclic_y()),
        )

    def gen_diagonal_left_bond_coos(self):
        """Generate all coordinate pairs like ``(i, j), (i + 1, j - 1)``."""
        return gen_2d_bonds(
            self.Lx,
            self.Ly,
            steppers=[
                lambda i, j: (i + 1, j - 1),
            ],
            cyclic=(self.is_cyclic_x(), self.is_cyclic_y()),
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
            cyclic=(self.is_cyclic_x(), self.is_cyclic_y()),
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
            cyclic=(self.is_cyclic_x(), self.is_cyclic_y()),
        )

    def gen_diagonal_right_bond_coos(self):
        """Generate all coordinate pairs like ``(i, j), (i + 1, j + 1)``."""
        return gen_2d_bonds(
            self.Lx,
            self.Ly,
            steppers=[
                lambda i, j: (i + 1, j + 1),
            ],
            cyclic=(self.is_cyclic_x(), self.is_cyclic_y()),
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
            cyclic=(self.is_cyclic_x(), self.is_cyclic_y()),
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
            cyclic=(self.is_cyclic_x(), self.is_cyclic_y()),
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
            cyclic=(self.is_cyclic_x(), self.is_cyclic_y()),
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

    def is_cyclic_x(self, j=None, imin=None, imax=None):
        """Check if the x dimension is cyclic (periodic), specifically whether
        a bond exists between ``(imin, j)`` and ``(imax, j)``, with default
        values of ``imin = 0`` and ``imax = Lx - 1``, and ``j`` at the center
        of the lattice. If ``imin`` and ``imax`` are adjacent then this is
        considered False, since there is no 'extra' connectivity.
        """
        if imin is None:
            imin = 0
        if imax is None:
            imax = self.Lx - 1

        if abs(imax - imin) <= 1:
            # first and last sites already connected -> a bit undefined
            return False

        if j is None:
            j = self.Ly // 2

        return bool(
            bonds(
                self[self.site_tag(imin, j)],
                self[self.site_tag(imax, j)],
            )
        )

    def is_cyclic_y(self, i=None, jmin=None, jmax=None):
        """Check if the y dimension is cyclic (periodic), specifically whether
        a bond exists between ``(i, jmin)`` and ``(i, jmax)``, with default
        values of ``jmin = 0`` and ``jmax = Ly - 1``, and ``i`` at the center
        of the lattice. If ``jmin`` and ``jmax`` are adjacent then this is
        considered False, since there is no 'extra' connectivity.
        """
        if jmin is None:
            jmin = 0
        if jmax is None:
            jmax = self.Ly - 1

        if abs(jmax - jmin) <= 1:
            # first and last sites already connected -> a bit undefined
            return False

        if i is None:
            i = self.Lx // 2

        return bool(
            bonds(
                self[self.site_tag(i, jmin)],
                self[self.site_tag(i, jmax)],
            )
        )

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

    def flatten(
        self,
        fuse_multibonds=True,
        inplace=False,
    ) -> "TensorNetwork2DFlat":
        """Contract all tensors at each site together, yielding a single tensor
        per site. By default, any multibonds between flattened sites will also
        be fused together. If not already, the resulting tensor network will be
        promoted to a :class:`TensorNetwork2DFlat`.

        Parameters
        ----------
        fuse_multibonds : bool, optional
            Whether to fuse any multibonds that are created by this process.
            Defaults to ``True``.
        inplace : bool, optional
            Whether to modify this tensor network inplace, or return a new
            one. Defaults to ``False``.

        Returns
        -------
        TensorNetwork2DFlat
        """
        tn = super().flatten(fuse_multibonds=fuse_multibonds, inplace=inplace)

        if not isinstance(tn, TensorNetwork2DFlat):
            tn.view_as_(TensorNetwork2DFlat, like=self)

        return tn

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
        compress_opts.setdefault("reduced", "left")
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

    def _contract_boundary_core_via_1d(
        self,
        xrange,
        yrange,
        from_which,
        max_bond,
        cutoff=1e-10,
        method="dm",
        layer_tags=None,
        **compress_opts,
    ):
        from quimb.tensor.tn1d.compress import tensor_network_1d_compress

        r2d = Rotator2D(self, xrange, yrange, from_which)
        site_tag = r2d.site_tag
        istep = r2d.istep

        def _do_compress(site_tags):
            site_tags, untag_groups = parse_site_tag_groups(self, site_tags)
            tn_boundary = self.partition(site_tags, inplace=True)[1]

            tensor_network_1d_compress(
                tn_boundary,
                max_bond=max_bond,
                cutoff=cutoff,
                method=method,
                site_tags=site_tags,
                inplace=True,
                **compress_opts,
            )

            self.add_tensor_network(tn_boundary, virtual=True)

            # also clean tensors replaced during compression
            untag_groups(self)

        # the initial row can contain several layers
        site_tags = [site_tag(r2d.sweep[0], j) for j in r2d.sweep_other]
        if any(len(self.tag_map[st]) > 1 for st in site_tags):
            _do_compress(site_tags)

        if layer_tags is None:
            layer_tags = [None]

        for i in r2d.sweep[:-1]:
            for layer_tag in layer_tags:
                if layer_tag is None:
                    # group all tensors tagged (i,j) OR (i+istep,j)
                    site_tags = [
                        (site_tag(i, j), site_tag(i + istep, j))
                        for j in r2d.sweep_other
                    ]
                else:
                    # group all tagged (i,j) OR ((i+istep,j) AND layer_tag)
                    site_tags = [
                        (site_tag(i, j), (site_tag(i + istep, j), layer_tag))
                        for j in r2d.sweep_other
                    ]

                _do_compress(site_tags)

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
        similarity_method=None,
        renorm=False,
        optimize="auto-hq",
        opposite_envs=None,
        equalize_norms=False,
        contract_boundary_opts=None,
        compress_opts=None,
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
        similarity_method : {'eigh', 'eig', 'svd', 'biorthog'}, optional
            Which similarity decomposition method to use to compress the full
            bond environment, a shortcut for
            ``compress_opts=dict(method=...)``. By default ``'eigh'``.
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
        compress_opts : dict, optional
            Options supplied to
            :func:`~quimb.tensor.decomp.similarity_compress` when compressing
            each full bond environment.
        """
        if equalize_norms:
            raise NotImplementedError

        compress_opts = ensure_dict(compress_opts)
        if similarity_method is not None:
            compress_opts["method"] = similarity_method
        compress_opts.setdefault("method", "eigh")
        compress_opts.setdefault("renorm", renorm)

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
                    E, max_bond, **compress_opts
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
        contract_opts=None,
        reduce_opts=None,
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
        contract_opts : dict, optional
            Explicit options for contracting the projectors to pass to
            :meth:`~quimb.tensor.TensorNetwork.to_dense`. Values
            set here take precedence over any defaults such as ``optimize``.
        reduce_opts : dict, optional
            Explicit options to pass to :func:`squared_op_to_reduced_factor`,
            for example ``method="cholesky"``. Values set here take precedence
            over any defaults.
        compress_opts : dict, optional
            Explicit options to pass to :func:`compute_oblique_projectors`.
            Values set here take precedence over any defaults.

        See Also
        --------
        TensorNetwork.insert_compressor_between_regions
        """
        contract_opts = ensure_dict(contract_opts)
        contract_opts.setdefault("optimize", optimize)

        r = Rotator2D(self, xrange, yrange, from_which)

        for i, inext in pairwise(r.sweep):
            # we compute the projectors from an untouched copy
            tn_calc = self.copy()

            for j in r.sweep_other:
                # this handles cyclic boundary conditions
                jnext = r.get_jnext(j)
                if jnext is not None:
                    ltags = (r.site_tag(i, j), r.site_tag(inext, j))
                    rtags = (r.site_tag(i, jnext), r.site_tag(inext, jnext))
                    #      │         │
                    #    ──O─┐ chi ┌─O──  i+1
                    #      │ └─▷═◁─┘ │
                    #      │ ┌┘   └┐ │
                    #    ──O─┘     └─O──  i
                    #      j        j+1
                    tn_calc.insert_compressor_between_regions(
                        ltags,
                        rtags,
                        new_ltags=ltags,
                        new_rtags=rtags,
                        insert_into=self,
                        max_bond=max_bond,
                        cutoff=cutoff,
                        contract_opts=contract_opts,
                        reduce_opts=reduce_opts,
                        compress_opts=compress_opts,
                    )

            if not lazy:
                # contract each pair of boundary tensors with their projectors
                for j in r.sweep_other:
                    self.contract_tags_(
                        (r.site_tag(i, j), r.site_tag(inext, j)),
                        **contract_opts,
                    )

            if equalize_norms:
                for t in self.select_tensors(r.x_tag(inext)):
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
        method=None,
        layer_tags=None,
        sweep_reverse=False,
        compress_opts=None,
        inplace=False,
        **contract_boundary_opts,
    ):
        """Unified entrypoint for contracting any rectangular patch of tensors
        from any direction, with any boundary ``method``: ``'mps'`` (the
        default), ``'full-bond'``, ``'projector2d'`` or any 1D compression
        method, see
        :func:`~quimb.tensor.tn1d.compress.tensor_network_1d_compress`.
        ``mode`` is a deprecated alias of ``method``.
        """
        method = _parse_boundary_method(method, contract_boundary_opts)

        tn = self if inplace else self.copy()

        # universal options
        contract_boundary_opts["xrange"] = xrange
        contract_boundary_opts["yrange"] = yrange
        contract_boundary_opts["from_which"] = from_which
        contract_boundary_opts["max_bond"] = max_bond
        contract_boundary_opts["compress_opts"] = compress_opts

        if method == "full-bond":
            tn._contract_boundary_full_bond(**contract_boundary_opts)
            return tn

        contract_boundary_opts["cutoff"] = cutoff

        if method == "projector2d":
            tn._contract_boundary_projector(**contract_boundary_opts)
            return tn

        # method == 'mps' options
        contract_boundary_opts["canonize"] = canonize
        contract_boundary_opts["layer_tags"] = layer_tags
        contract_boundary_opts["sweep_reverse"] = sweep_reverse

        if method == "mps":
            tn._contract_boundary_core(**contract_boundary_opts)
            return tn

        tn._contract_boundary_core_via_1d(
            method=method, **contract_boundary_opts
        )
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
        method=None,
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
        method : {'mps', 'full-bond', ...}, optional
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
        method = _parse_boundary_method(method, contract_boundary_opts)
        return self.contract_boundary_from(
            xrange=xrange,
            yrange=yrange,
            from_which="xmin",
            max_bond=max_bond,
            cutoff=cutoff,
            canonize=canonize,
            method=method,
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
        method=None,
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
        method : {'mps', 'full-bond', ...}, optional
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
        method = _parse_boundary_method(method, contract_boundary_opts)
        return self.contract_boundary_from(
            xrange=xrange,
            yrange=yrange,
            from_which="xmax",
            max_bond=max_bond,
            cutoff=cutoff,
            canonize=canonize,
            method=method,
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
        method=None,
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
        method : {'mps', 'full-bond', ...}, optional
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
        method = _parse_boundary_method(method, contract_boundary_opts)
        return self.contract_boundary_from(
            xrange=xrange,
            yrange=yrange,
            from_which="ymin",
            max_bond=max_bond,
            cutoff=cutoff,
            canonize=canonize,
            method=method,
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
        method=None,
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
        method : {'mps', 'full-bond', ...}, optional
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
        method = _parse_boundary_method(method, contract_boundary_opts)
        return self.contract_boundary_from(
            xrange=xrange,
            yrange=yrange,
            from_which="ymax",
            max_bond=max_bond,
            cutoff=cutoff,
            canonize=canonize,
            method=method,
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
        strip_exponent=False,
        equalize_norms="auto",
        final_contract=True,
        final_contract_opts=None,
        optimize="auto-hq",
        progbar=False,
        inplace=False,
    ):
        """Unified handler for performing iterleaved contractions in a
        sequence of inwards boundary directions.
        """
        tn = self if inplace else self.copy()

        contract_boundary_opts = ensure_dict(contract_boundary_opts)

        if equalize_norms == "auto":
            # if we are going to extract exponent at end, assume we
            # should do it throughout the computation as well
            if strip_exponent:
                # but we won't redistribute norms (`True`) during contraction
                equalize_norms = 1.0
            else:
                equalize_norms = False

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
                if self.is_cyclic_x():
                    sequence = ("xmin",)
                else:
                    sequence = ("xmin", "xmax")
            else:
                if self.is_cyclic_y():
                    sequence = ("ymin",)
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
            final_contract_opts.setdefault("optimize", optimize)
            final_contract_opts.setdefault("inplace", inplace)
            final_contract_opts.setdefault("strip_exponent", strip_exponent)
            return tn.contract(**final_contract_opts)

        return tn

    def contract_boundary(
        self,
        max_bond=None,
        *,
        cutoff=1e-10,
        canonize=True,
        method=None,
        layer_tags=None,
        compress_opts=None,
        sequence=None,
        xmin=None,
        xmax=None,
        ymin=None,
        ymax=None,
        max_separation=1,
        max_unfinished=1,
        around=None,
        strip_exponent=False,
        equalize_norms="auto",
        final_contract=True,
        final_contract_opts=None,
        progbar=None,
        inplace=False,
        **contract_boundary_opts,
    ):
        r"""Contract the boundary of this 2D tensor network inwards. By
        default, if contracting to a scalar, this contracts the two shortest
        opposing sides inwards. If `around` is specified, or depending on
        `sequence`, it can contract from in any sequence of directions, and in
        any order like so::

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
        method : {'mps', 'full-bond', ...}, optional
            How to perform the compression on the boundary, can also be any of
            the generic 1D or arbgeom methods.
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
        max_unfinished : int, optional
            If ``around is None``, when this many sides are still not within
            ``max_separation`` simply contract the remaining tensor network.
        around : None or sequence of (int, int), optional
            If given, don't contract the square of sites bounding these
            coordinates.
        strip_exponent : bool, optional
            Whether to strip an overall exponent, log10, from the *final*
            contraction result. If ``True``, a tuple of ``(scalar, exponent)``
            is returned instead of a single scalar.
        equalize_norms : bool, float, or "auto", optional
            Whether to equalize the norms of the boundary tensors after each
            contraction, gathering the overall scaling coefficient, log10, in
            ``tn.exponent``. By default (``"auto"``) this follows
            ``strip_exponent``.
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
        method = _parse_boundary_method(method, contract_boundary_opts)
        contract_boundary_opts["max_bond"] = max_bond
        contract_boundary_opts["method"] = method
        contract_boundary_opts["cutoff"] = cutoff
        contract_boundary_opts["canonize"] = canonize
        contract_boundary_opts["layer_tags"] = layer_tags
        contract_boundary_opts["compress_opts"] = compress_opts

        if method == "full-bond":
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
            max_unfinished=max_unfinished,
            around=around,
            strip_exponent=strip_exponent,
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

    contract_mps_sweep_ = functools.partialmethod(
        contract_mps_sweep, inplace=True
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
            env_compute(method="full-bond", **kwargs)

        tn = envs[lbl_a, mid] | envs[lbl_b, mid + 1]
        return tn.contract(all, optimize=optimize)

    def gen_block_environments(
        self,
        direction,
        blocks,
        max_bond,
        *,
        cyclic=None,
        compress_fn=None,
        schedule="auto",
        method=None,
        layer_tags=None,
        cutoff=None,
        canonize=True,
        optimize="auto-hq",
        equalize_norms=False,
        compress_opts=None,
        **compress_method_opts,
    ):
        """Yield compressed row or column environments as they are ready. Keep
        cached environments only until their last use. See
        :func:`~quimb.tensor.environments.gen_compressed_environments`.

        Parameters
        ----------
        direction : {'x', 'y'}
            Compute environments for blocks of rows or columns respectively.
        blocks : sequence of tuple[int, int]
            The ``(start, size)`` blocks of rows or columns, which can have
            different sizes, all computed together in one sweep. Use
            :func:`~quimb.tensor.environments.all_blocks` to get every block
            of one size.
        max_bond : int or None
            Maximum environment bond dimension, often called 'chi'. Supply
            ``None`` to truncate using only ``cutoff``. This is not recommended
            in 2D.
        cyclic : bool, optional
            Periodicity along ``direction``. By default infer it from the
            network.
        compress_fn : {None, 'ag', '1d', '2d'}, optional
            Which compression function to use along the other direction, see
            :func:`~quimb.tensor.environments.gen_compressed_environments`.
            By default use '1d' if the other direction is open, and 'ag' if
            it is periodic.
        schedule : {'auto', 'tree', 'cut'}, optional
            Environment construction schedule, only relevant if ``cyclic``.
            By default use 'tree', which never compresses two environments
            together. The 'cut' schedule uses fewer compressions but can
            degrade the approximation by combining two already compressed
            environments.
        method : str or callable, optional
            The compression method, supplied to ``compress_fn``. By default
            use its own default method.
        layer_tags : None or sequence[str], optional
            Add the tensors at each row or column one layer at a time in this
            order, compressing after each.
        cutoff : float, optional
            Compression cutoff. By default use the default of the compression
            function.
        canonize : bool or str, optional
            Canonicalization option supplied to the compressor.
        optimize : str, optional
            Contraction path optimizer supplied to the compressor.
        equalize_norms : bool or float, optional
            Whether to equalize tensor norms after each compression.
        compress_opts : dict, optional
            Additional options supplied to the compressor.
        compress_method_opts
            Additional options supplied to the compression method.

        Yields
        ------
        block : tuple[int, int]
            The ``(start, size)`` block.
        environment : TensorNetwork
            The environment of ``block``, which also carries the original
            network's ``.exponent``.
        """
        check_opt("direction", direction, ("x", "y"))
        r2d = Rotator2D(self, None, None, direction + "min")
        if cyclic is None:
            cyclic = r2d.is_cyclic_x()
        if compress_fn is None:
            compress_fn = "ag" if r2d.is_cyclic_y() else "1d"

        environments = gen_compressed_environments(
            self,
            tuple(map(r2d.x_tag, r2d.sweep)),
            tuple(map(r2d.y_tag, r2d.sweep_other)),
            blocks,
            max_bond,
            cyclic=cyclic,
            compress_fn=compress_fn,
            schedule=schedule,
            method=method,
            layer_tags=layer_tags,
            cutoff=cutoff,
            canonize=canonize,
            optimize=optimize,
            equalize_norms=equalize_norms,
            compress_opts=compress_opts,
            **compress_method_opts,
        )
        for block, environment in environments:
            environment.exponent += self.exponent
            yield block, environment

    def compute_block_environments(
        self,
        direction,
        blocks,
        max_bond,
        *,
        cutoff=None,
        method=None,
        **kwargs,
    ):
        """Compute compressed row or column environments for blocks of rows
        or columns. See :meth:`gen_block_environments` for the options, and
        to process them one at a time.

        Parameters
        ----------
        direction : {'x', 'y'}
            Compute environments for blocks of rows or columns respectively.
        blocks : sequence of tuple[int, int]
            The ``(start, size)`` blocks of rows or columns.
        max_bond : int or None
            Maximum environment bond dimension, often called 'chi'. Supply
            ``None`` to truncate using only ``cutoff``. This is not recommended
            in 2D.
        cutoff : float, optional
            Compression cutoff. By default use the default of the compression
            function.
        method : str or callable, optional
            The compression method, see :meth:`gen_block_environments`.
        kwargs
            Supplied to :meth:`gen_block_environments`.

        Returns
        -------
        dict[tuple[int, int], TensorNetwork]
            The environment for each ``(start, size)`` block.
        """
        return dict(
            self.gen_block_environments(
                direction,
                blocks,
                max_bond,
                cutoff=cutoff,
                method=method,
                **kwargs,
            )
        )

    def compute_environments(
        self,
        from_which,
        xrange=None,
        yrange=None,
        max_bond=None,
        *,
        cutoff=1e-10,
        canonize=True,
        method=None,
        layer_tags=None,
        dense=False,
        compress_opts=None,
        envs=None,
        equalize_norms=False,
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
        method : {'mps', 'projector', 'full-bond', ...}, optional
            Which contraction method to use for the environments.
        layer_tags : str or iterable[str], optional
            If this 2D TN is multi-layered (e.g. a bra and a ket), and
            ``method == 'mps'``, contract and compress each specified layer
            separately, for a cheaper contraction.
        dense : bool, optional
            Whether to use dense tensors for the environments.
        compress_opts : dict, optional
            Other options to pass to
            :func:`~quimb.tensor.tensor_core.tensor_compress_bond`.
        envs : dict, optional
            An existing dictionary to store the environments in.
        equalize_norms : bool or float, optional
            Whether to equalize the norms of the boundary tensors after each
            contraction, gathering the overall scaling coefficient, log10, in
            ``tn.exponent``. Note that the boundary tensor networks only
            accumulate *new* exponent generated by the contraction, if the
            original tensor network has a non-zero exponent, you may need to
            use it to get the correct overall scaling. For example:
            ``envs["xmin", i] | tn.select(tn.x_tag(i), with_exponent=True) | envs["xmax", i]``.
        contract_boundary_opts
            Other options to pass to
            :meth:`~quimb.tensor.tn2d.core.TensorNetwork2D.contract_boundary_from`.

        Returns
        -------
        envs : dict
            A dictionary of the environments, with keys of the form
            ``(from_which, row_or_col_index)``.
        """
        method = _parse_boundary_method(method, contract_boundary_opts)
        tn = self.copy()

        r2d = Rotator2D(tn, xrange, yrange, from_which)
        sweep, x_tag = r2d.sweep, r2d.x_tag

        if envs is None:
            envs = {}

        if method == "full-bond":
            # set shared storage for opposite env contractions
            contract_boundary_opts.setdefault("opposite_envs", {})

        envs[from_which, sweep[0]] = TensorNetwork([])
        first_row = x_tag(sweep[0])
        if dense:
            tn ^= first_row
        envs[from_which, sweep[1]] = tn.select(first_row)

        exponent0 = tn.exponent

        for i in sweep[2:]:
            iprevprev = i - 2 * sweep.step
            iprev = i - sweep.step
            if dense:
                tn ^= (x_tag(iprevprev), x_tag(iprev))

                if equalize_norms:
                    tn.equalize_norms_(equalize_norms)

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
                    method=method,
                    canonize=canonize,
                    layer_tags=layer_tags,
                    compress_opts=compress_opts,
                    equalize_norms=equalize_norms,
                    **contract_boundary_opts,
                )

            tn_env_i = tn.select(first_row)
            # we only accumulate exponent generated by the contraction
            tn_env_i.exponent = tn.exponent - exponent0
            envs[from_which, i] = tn_env_i

        return envs

    compute_xmin_environments = functools.partialmethod(
        compute_environments, from_which="xmin"
    )
    """Compute the ``self.Lx`` 1D boundary tensor networks describing
    the lower environments of each row in this 2D tensor network. See
    :meth:`~quimb.tensor.tn2d.core.TensorNetwork2D.compute_x_environments`
    for full details.
    """

    compute_xmax_environments = functools.partialmethod(
        compute_environments, from_which="xmax"
    )
    """Compute the ``self.Lx`` 1D boundary tensor networks describing
    the upper environments of each row in this 2D tensor network. See
    :meth:`~quimb.tensor.tn2d.core.TensorNetwork2D.compute_x_environments`
    for full details.
    """

    compute_ymin_environments = functools.partialmethod(
        compute_environments, from_which="ymin"
    )
    """Compute the ``self.Ly`` 1D boundary tensor networks describing
    the left environments of each column in this 2D tensor network. See
    :meth:`~quimb.tensor.tn2d.core.TensorNetwork2D.compute_y_environments`
    for full details.
    """

    compute_ymax_environments = functools.partialmethod(
        compute_environments, from_which="ymax"
    )
    """Compute the ``self.Ly`` 1D boundary tensor networks describing
    the right environments of each column in this 2D tensor network. See
    :meth:`~quimb.tensor.tn2d.core.TensorNetwork2D.compute_y_environments`
    for full details.
    """

    def compute_x_environments(
        self,
        max_bond=None,
        *,
        cutoff=1e-10,
        canonize=True,
        dense=False,
        method=None,
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
        method : {'mps', 'full-bond', ...}, optional
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
            :meth:`~quimb.tensor.tn2d.core.TensorNetwork2D.contract_boundary_from_xmin`
            and
            :meth:`~quimb.tensor.tn2d.core.TensorNetwork2D.contract_boundary_from_xmax`
            .

        Returns
        -------
        x_envs : dict[(str, int), TensorNetwork]
            The two environment tensor networks of row ``i`` will be stored in
            ``x_envs['xmin', i]`` and ``x_envs['xmax', i]``.
        """
        method = _parse_boundary_method(method, contract_boundary_opts)
        contract_boundary_opts["max_bond"] = max_bond
        contract_boundary_opts["cutoff"] = cutoff
        contract_boundary_opts["canonize"] = canonize
        contract_boundary_opts["method"] = method
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
        method=None,
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
        method : {'mps', 'full-bond', ...}, optional
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
            :meth:`~quimb.tensor.tn2d.core.TensorNetwork2D.contract_boundary_from_ymin`
            and
            :meth:`~quimb.tensor.tn2d.core.TensorNetwork2D.contract_boundary_from_ymax`
            .

        Returns
        -------
        y_envs : dict[(str, int), TensorNetwork]
            The two environment tensor networks of column ``j`` will be stored
            in ``y_envs['ymin', j]`` and ``y_envs['ymax', j]``.
        """
        method = _parse_boundary_method(method, contract_boundary_opts)
        contract_boundary_opts["max_bond"] = max_bond
        contract_boundary_opts["cutoff"] = cutoff
        contract_boundary_opts["canonize"] = canonize
        contract_boundary_opts["method"] = method
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
        y_envs = {}
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
        plaquette_envs = {}
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
        x_envs = {}
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
        plaquette_envs = {}
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
        method=None,
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
        method : {'mps', 'full-bond', ...}, optional
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
            :meth:`~quimb.tensor.tn2d.core.TensorNetwork2D.compute_y_environments`
            or
            :meth:`~quimb.tensor.tn2d.core.TensorNetwork2D.compute_x_environments`
            .

        Returns
        -------
        dict[((int, int), (int, int)), TensorNetwork]
            The plaquette environments. The key is two tuples of ints, the
            startings coordinate of the plaquette being the first and the size
            of the plaquette being the second pair.
        """
        method = _parse_boundary_method(method, compute_environment_opts)
        first_contract = _choose_plaquette_first_contract(
            self, x_bsz, y_bsz, first_contract
        )

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
            method=method,
            layer_tags=layer_tags,
            compress_opts=compress_opts,
            second_dense=second_dense,
            **compute_environment_opts,
        )

    def compute_plaquette_environments_via_envs(
        self,
        x_bsz=2,
        y_bsz=2,
        *,
        max_bond,
        starts=None,
        cyclic=None,
        first_contract=None,
        schedule="auto",
        second_schedule="auto",
        second_dense=None,
        method=None,
        layer_tags=None,
        cutoff=None,
        canonize=True,
        optimize="auto-hq",
        equalize_norms=False,
        compress_opts=None,
        contract_opts=None,
        **compress_method_opts,
    ):
        """Compute plaquette environments for open or periodic boundaries.
        Plaquettes can cross periodic boundaries. Compress in the first
        direction, then contract along each remaining strip. Use
        ``second_dense`` to choose exact or compressed contraction along the
        strips.

        See :meth:`compute_plaquette_environments` for contraction from open
        boundaries.

        Parameters
        ----------
        x_bsz, y_bsz : int, optional
            Plaquette size in each direction.
        max_bond : int or None
            Maximum environment bond dimension, often called 'chi'. Supply
            ``None`` to truncate using only ``cutoff``. This is not recommended
            in 2D.
        starts : sequence of tuple[int, int], optional
            The ``(i, j)`` plaquette starts to compute. By default compute
            every valid start.
        cyclic : bool or tuple[bool, bool], optional
            Periodicity in each direction. By default infer it.
        first_contract : {'x', 'y'}, optional
            Direction to contract approximately first.
        schedule : {'auto', 'tree', 'cut'}, optional
            Construction schedule of the approximate environments in the
            first direction, see :meth:`gen_block_environments`.
        second_schedule : {'auto', 'tree', 'cut'}, optional
            Schedule along each strip. By default use 'cut' for exact
            contraction and 'tree' for compressed contraction.
        second_dense : bool, optional
            Whether to contract along each strip exactly. By default do so only
            for strips of width one. Compress wider strips.
        method : str or callable, optional
            Compression method. By default use the compressor's default for
            each direction. See
            :func:`~quimb.tensor.environments.gen_compressed_environments`.
        layer_tags : None or sequence[str], optional
            Contract the tensors of each plane one layer at a time, in this
            order.
        cutoff : float, optional
            Compression cutoff. By default use the default of the compression
            function.
        canonize : bool or str, optional
            Canonicalization option supplied to the compressor.
        optimize : str, optional
            Contraction path optimizer supplied to the compressor, and the
            default for ``contract_opts``.
        equalize_norms : bool or float, optional
            Whether to equalize tensor norms after each compression.
        compress_opts : dict, optional
            Additional options supplied to the compressor.
        contract_opts : dict, optional
            Options for the exact contractions along each strip, see
            ``second_dense``. ``optimize`` is the default path optimizer.
        compress_method_opts
            Additional options supplied to the compression method.

        Returns
        -------
        dict[((int, int), (int, int)), TensorNetwork]
            Plaquette environments keyed by start and size.
        """
        cyclic_x, cyclic_y = _normalize_2d_cyclic(self, cyclic)
        if not 1 <= x_bsz <= self.Lx:
            raise ValueError("x_bsz must satisfy 1 <= x_bsz <= Lx")
        if not 1 <= y_bsz <= self.Ly:
            raise ValueError("y_bsz must satisfy 1 <= y_bsz <= Ly")
        nx = self.Lx if cyclic_x else self.Lx - x_bsz + 1
        ny = self.Ly if cyclic_y else self.Ly - y_bsz + 1

        if starts is None:
            starts = tuple(product(range(nx), range(ny)))
        else:
            starts = tuple(sorted(set(starts)))

        if not starts:
            return {}
        if any(not (0 <= i < nx and 0 <= j < ny) for i, j in starts):
            raise ValueError(
                f"starts must lie in range(0, {nx}) x range(0, {ny})"
            )

        contract_opts = ensure_dict(contract_opts)
        contract_opts.setdefault("optimize", optimize)
        environment_opts = dict(
            max_bond=max_bond,
            schedule=schedule,
            layer_tags=layer_tags,
            cutoff=cutoff,
            method=method,
            canonize=canonize,
            optimize=optimize,
            equalize_norms=equalize_norms,
            compress_opts=compress_opts,
            **compress_method_opts,
        )
        return dict(
            _gen_plaquette_environments_via_envs(
                self,
                tuple((start, (x_bsz, y_bsz)) for start in starts),
                cyclic_x,
                cyclic_y,
                first_contract,
                second_schedule,
                contract_opts,
                environment_opts,
                second_dense=second_dense,
            )
        )

    def coarse_grain_hotrg(
        self,
        direction,
        max_bond=None,
        cutoff=1e-10,
        canonize=False,
        canonize_opts=None,
        gauge_power=1.0,
        lazy=False,
        strip_exponent=False,
        equalize_norms="auto",
        optimize="auto-hq",
        contract_opts=None,
        reduce_opts=None,
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
        canonize : bool, optional
            Whether to canonize all tensors before computing projectors,
            via :meth:`gauge_all_simple_`.
        canonize_opts : None or dict, optional
            Additional options to pass to :meth:`gauge_all_simple_`.
        gauge_power : float, optional
            If `canonize=True`, the power to which to raise the computed bond
            gauge weights when before computing the compressed projectors.
        lazy : bool, optional
            Whether to contract the coarse graining projectors or leave them
            in the tensor network lazily. Default is to contract them.
        strip_exponent : bool, optional
            If ``True``, enable norm equalization during the coarse graining.
        equalize_norms : bool or "auto", optional
            Whether to equalize the norms of the tensors in the coarse grained
            lattice. By default (``"auto"``) this follows ``strip_exponent``.
        optimize : str, optional
            The optimization method to use when contracting the coarse grained
            lattice, if ``lazy=False``.
        contract_opts : dict, optional
            Explicit options for contracting the projectors to pass to
            :meth:`~quimb.tensor.TensorNetwork.to_dense`. Values
            set here take precedence over any defaults such as ``optimize``.
        reduce_opts : dict, optional
            Explicit options to pass to :func:`squared_op_to_reduced_factor`,
            for example ``method="cholesky"``. Values set here take precedence
            over any defaults.
        compress_opts : dict, optional
            Explicit options to pass to :func:`compute_oblique_projectors`.
            Values set here take precedence over any defaults.
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
        check_opt("direction", direction, ("x", "y"))

        if equalize_norms == "auto":
            # if we are going to extract exponent at end, assume we
            # should do it throughout the computation as well
            if strip_exponent:
                # but we won't redistribute norms (`True`) during contraction
                equalize_norms = 1.0
            else:
                equalize_norms = False

        contract_opts = ensure_dict(contract_opts)
        contract_opts.setdefault("optimize", optimize)

        reduce_opts = ensure_dict(reduce_opts)

        compress_opts = ensure_dict(compress_opts)
        compress_opts.setdefault("max_bond", max_bond)
        compress_opts.setdefault("cutoff", cutoff)

        tn = self if inplace else self.copy()

        if canonize:
            canonize_opts = ensure_dict(canonize_opts)
            # extract simple gauges
            gauges = {}
            tn.gauge_all_simple_(gauges=gauges, **canonize_opts)
            # compute projectors from gauged network
            tn_calc = tn.copy()
            # insert gauges back into target before inserting projectors
            tn.gauge_simple_insert(gauges)
        else:
            gauges = None
            tn_calc = tn.copy()

        r = Rotator2D(tn, None, None, f"{direction}min")

        # track new coordinates / tags
        retag_map = {}

        for i in range(r.imin, r.imax + 1, 2):
            inext = i + 1
            next_i_in_lattice = inext <= r.imax

            for j in r.sweep_other:
                # handles cyclic case
                jnext = r.get_jnext(j)

                #      │         │
                #    ──O─┐ chi ┌─O──  i+1
                #      │ └─▷═◁─┘ │
                #      │ ┌┘   └┐ │
                #    ──O─┘     └─O──  i
                #      │         │
                #      j        j+1
                tag_ij = r.site_tag(i, j)
                tag_ip1j = r.site_tag(inext, j)
                new_tag = r.site_tag(i // 2, j)
                retag_map[tag_ij] = new_tag
                if next_i_in_lattice:
                    retag_map[tag_ip1j] = new_tag

                if next_i_in_lattice and jnext is not None:
                    ltags = (tag_ij, tag_ip1j)
                    rtags = r.site_tag(i, jnext), r.site_tag(inext, jnext)
                    tn_calc.insert_compressor_between_regions(
                        ltags,
                        rtags,
                        insert_into=tn,
                        new_ltags=ltags,
                        new_rtags=rtags,
                        gauges=gauges,
                        contract_opts=contract_opts,
                        reduce_opts=reduce_opts,
                        compress_opts=compress_opts,
                        gauge_power=gauge_power,
                    )

            retag_map[r.x_tag(i)] = r.x_tag(i // 2)
            if next_i_in_lattice:
                retag_map[r.x_tag(inext)] = r.x_tag(i // 2)

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
                tn.contract_tags_(st, **contract_opts)

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
        gauge_power=1.0,
        sequence=("x", "y"),
        max_separation=1,
        max_unfinished=1,
        lazy=False,
        strip_exponent=False,
        equalize_norms="auto",
        optimize="auto-hq",
        contract_opts=None,
        reduce_opts=None,
        compress_opts=None,
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
            Whether to canonize all tensors before computing projectors,
            via :meth:`gauge_all_simple_`.
        canonize_opts : None or dict, optional
            Additional options to pass to :meth:`gauge_all_simple_`.
        gauge_power : float, optional
            If `canonize=True`, the power to which to raise the computed bond
            gauge weights when before computing the compressed projectors.
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
        strip_exponent : bool, optional
            Whether to strip an overall exponent, log10, from the *final*
            contraction result. If ``True``, a tuple of ``(scalar, exponent)``
            is returned instead of a single scalar.
        equalize_norms : bool, float, or "auto", optional
            Whether to equalize the norms of all tensors after each
            contraction, gathering the overall scaling coefficient, log10, in
            ``tn.exponent``. By default (``"auto"``) this follows
            ``strip_exponent``.
        optimize : str or PathOptimizer, optional
            How to optimize the contraction of the projection tensors. Note any
            value in ``contract_opts`` will take precedence over this.
        contract_opts : dict, optional
            Explicit options for contracting the projectors to pass to
            :meth:`~quimb.tensor.TensorNetwork.to_dense`. Values
            set here take precedence over any defaults such as ``optimize``.
        reduce_opts : dict, optional
            Explicit options to pass to :func:`squared_op_to_reduced_factor`,
            for example ``method="cholesky"``. Values set here take precedence
            over any defaults.
        compress_opts : dict, optional
            Explicit options to pass to :func:`compute_oblique_projectors`.
            Values set here take precedence over any defaults.
        final_contract : bool, optional
            Whether to exactly contract the remaining tensor network after the
            coarse graining contractions.
        final_contract_opts : None or dict, optional
            Options to pass to :meth:`contract`, if ``final_contract=True``.
            Defaults to same as ``contract_opts``.
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

        if equalize_norms == "auto":
            # if we are going to extract exponent at end, assume we
            # should do it throughout the computation as well
            if strip_exponent:
                # but we won't redistribute norms (`True`) during contraction
                equalize_norms = 1.0
            else:
                equalize_norms = False

        contract_opts = ensure_dict(contract_opts)
        contract_opts.setdefault("optimize", optimize)

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

            tn.coarse_grain_hotrg_(
                direction=direction,
                max_bond=max_bond,
                canonize=canonize,
                canonize_opts=canonize_opts,
                gauge_power=gauge_power,
                cutoff=cutoff,
                lazy=lazy,
                equalize_norms=equalize_norms,
                contract_opts=contract_opts,
                reduce_opts=reduce_opts,
                compress_opts=compress_opts,
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
            if final_contract_opts is None:
                final_contract_opts = dict(contract_opts)
            else:
                final_contract_opts = ensure_dict(final_contract_opts)
                final_contract_opts.setdefault("optimize", optimize)
            final_contract_opts.setdefault("strip_exponent", strip_exponent)
            return tn.contract(inplace=inplace, **final_contract_opts)

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
        method=None,
        sequence=None,
        xmin=None,
        xmax=None,
        ymin=None,
        ymax=None,
        max_separation=1,
        around=None,
        strip_exponent=False,
        equalize_norms="auto",
        optimize="auto-hq",
        contract_opts=None,
        reduce_opts=None,
        compress_opts=None,
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
        has a length larger than `max_separation`, and thus exact contraction
        can be used.

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
        method : str, optional
            The method to perform the boundary contraction. Defaults to
            ``'projector'``.
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
        strip_exponent : bool, optional
            Whether to strip an overall exponent, log10, from the *final*
            contraction result. If ``True``, a tuple of ``(scalar, exponent)``
            is returned instead of a single scalar.
        equalize_norms : bool, float, or "auto", optional
            Whether to equalize the norms of the boundary tensors after each
            contraction, gathering the overall scaling coefficient, log10, in
            ``tn.exponent``. By default (``"auto"``) this follows
            ``strip_exponent``.
        optimize : str or PathOptimizer, optional
            How to optimize the contraction of the projection tensors. Note any
            value in ``contract_opts`` will take precedence over this.
        contract_opts : dict, optional
            Explicit options for contracting the projectors to pass to
            :meth:`~quimb.tensor.TensorNetwork.to_dense`. Values
            set here take precedence over any defaults such as ``optimize``.
        reduce_opts : dict, optional
            Explicit options to pass to :func:`squared_op_to_reduced_factor`,
            for example ``method="cholesky"``. Values set here take precedence
            over any defaults.
        compress_opts : dict, optional
            Explicit options to pass to :func:`compute_oblique_projectors`.
            Values set here take precedence over any defaults.
        final_contract : bool, optional
            Whether to exactly contract the remaining tensor network after the
            boundary contraction.
        final_contract_opts : None or dict, optional
            Options to pass to :meth:`contract`, if ``final_contract=True``.
            Defaults to same as ``contract_opts``.
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
        contract_opts = ensure_dict(contract_opts)
        contract_opts.setdefault("optimize", optimize)
        if final_contract_opts is None:
            final_contract_opts = contract_opts
        else:
            final_contract_opts = ensure_dict(final_contract_opts)
            final_contract_opts.setdefault("optimize", optimize)

        contract_boundary_opts["max_bond"] = max_bond
        contract_boundary_opts["cutoff"] = cutoff
        method = _parse_boundary_method(
            method, contract_boundary_opts, default="projector"
        )
        contract_boundary_opts["method"] = method
        contract_boundary_opts["lazy"] = lazy
        contract_boundary_opts["canonize"] = canonize
        contract_boundary_opts["canonize_opts"] = canonize_opts
        contract_boundary_opts["contract_opts"] = contract_opts
        contract_boundary_opts["reduce_opts"] = reduce_opts
        contract_boundary_opts["compress_opts"] = compress_opts

        if lazy:
            # we are implicitly asking for the tensor network
            final_contract = False

        if sequence is None:
            sequence = []
            if self.is_cyclic_x():
                sequence.append("xmin")
            else:
                sequence.extend(["xmin", "xmax"])
            if self.is_cyclic_y():
                sequence.append("ymin")
            else:
                sequence.extend(["ymin", "ymax"])

        return self._contract_interleaved_boundary_sequence(
            contract_boundary_opts=contract_boundary_opts,
            sequence=sequence,
            xmin=xmin,
            xmax=xmax,
            ymin=ymin,
            ymax=ymax,
            max_separation=max_separation,
            around=around,
            strip_exponent=strip_exponent,
            equalize_norms=equalize_norms,
            final_contract=final_contract,
            final_contract_opts=final_contract_opts,
            progbar=progbar,
            inplace=inplace,
        )

    contract_ctmrg_ = functools.partialmethod(contract_ctmrg, inplace=True)


def _parse_boundary_method(method, opts, default="mps"):
    """Read the boundary contraction method, including the deprecated ``mode``
    option in ``opts``.
    """
    if "mode" in opts:
        warnings.warn(
            "`mode` is deprecated, use `method` instead.",
            FutureWarning,
            stacklevel=3,
        )
        mode = opts.pop("mode")
        if (method is not None) and (mode == "full-bond"):
            # `method` used to select the full bond similarity decomposition
            opts.setdefault("similarity_method", method)
        method = mode
    if method is None:
        method = default
    return method


def is_lone_coo(where):
    """Check whether ``where`` is a single coordinate pair."""
    return (len(where) == 2) and (isinstance(where[0], Integral))


def _normalize_2d_cyclic(tn, cyclic):
    if cyclic is None:
        return tn.is_cyclic_x(), tn.is_cyclic_y()
    if isinstance(cyclic, bool):
        return cyclic, cyclic
    return tuple(cyclic)


def _choose_plaquette_first_contract(tn, x_bsz, y_bsz, first_contract):
    if first_contract is None:
        if x_bsz > y_bsz:
            first_contract = "y"
        elif (y_bsz > x_bsz) or (tn.Lx >= tn.Ly):
            first_contract = "x"
        else:
            first_contract = "y"
    check_opt("first_contract", first_contract, ("x", "y"))
    return first_contract


def _gen_plaquette_environments_via_envs(
    tn,
    plaquettes,
    cyclic_x,
    cyclic_y,
    first_contract,
    second_schedule,
    contract_opts,
    environment_opts,
    second_dense=None,
):
    """Yield environments for plaquettes ``((i, j), (x_bsz, y_bsz))``.
    Plaquettes can differ in size and cross periodic boundaries.

    First compress along one direction to leave a narrow strip. Unless
    ``first_contract`` is set, choose this direction for each plaquette. Share
    one sweep across all block sizes in each direction.

    Then contract along each strip. By default, contract strips of width one
    exactly and compress wider strips. Set ``second_dense`` to choose
    explicitly.
    """
    # the first direction is 'x' in each rotated frame
    rotators = {d: Rotator2D(tn, None, None, d + "min") for d in "xy"}

    # group strip targets by first direction and first block
    blocks_by_direction = {"x": defaultdict(set), "y": defaultdict(set)}
    for (i, j), (x_bsz, y_bsz) in plaquettes:
        # contract first along the direction that keeps the strip narrow
        direction = _choose_plaquette_first_contract(
            tn, x_bsz, y_bsz, first_contract
        )
        # express the start and size in the rotated frame
        r2d = rotators[direction]
        first, second = r2d.rotate(i, j)
        first_bsz, second_bsz = r2d.rotate(x_bsz, y_bsz)
        blocks_by_direction[direction][first, first_bsz].add(
            (second, second_bsz)
        )

    # share one first sweep across block sizes in each direction
    for direction, second_blocks_by_first in blocks_by_direction.items():
        if not second_blocks_by_first:
            continue

        r2d = rotators[direction]
        first_tag, second_tag = r2d.x_tag, r2d.y_tag
        first_cyclic, second_cyclic = r2d.rotate(cyclic_x, cyclic_y)
        first_length = len(r2d.sweep)

        first_envs = tn.gen_block_environments(
            direction,
            tuple(second_blocks_by_first),
            cyclic=first_cyclic,
            compress_fn="ag" if second_cyclic else "1d",
            **environment_opts,
        )
        second_tags = tuple(map(second_tag, r2d.sweep_other))

        # each first environment is used then dropped as soon as it is ready
        for (first, first_bsz), first_env in first_envs:
            first_block_tags = tuple(
                first_tag(first + d) for d in range(first_bsz)
            )
            # join the block and its environment for the second sweep
            strip = tn.select_any(first_block_tags, virtual=False) | first_env
            second_blocks = tuple(second_blocks_by_first[first, first_bsz])

            # by default only contract strips one plane wide exactly
            if second_dense is None:
                dense = first_bsz < 2
            else:
                dense = second_dense

            if dense:
                second_envs = gen_exact_environments(
                    strip,
                    second_tags,
                    second_blocks,
                    cyclic=second_cyclic,
                    schedule=second_schedule,
                    contract_opts=contract_opts,
                )
            else:
                # compress along the strip
                if first_cyclic:
                    # a single piece, connected to both ends of the block
                    if first_bsz < first_length:
                        first_block_tags += (first_tag(first + first_bsz),)
                else:
                    # a piece either side, unless the block is at an edge
                    if first > 0:
                        first_block_tags = (
                            first_tag(first - 1),
                            *first_block_tags,
                        )
                    if first + first_bsz < first_length:
                        first_block_tags += (first_tag(first + first_bsz),)

                second_envs = gen_compressed_environments(
                    strip,
                    second_tags,
                    first_block_tags,
                    second_blocks,
                    cyclic=second_cyclic,
                    compress_fn="ag" if first_cyclic else "1d",
                    **{**environment_opts, "schedule": second_schedule},
                )

            for (second, second_bsz), second_env in second_envs:
                # add the first environment's edges alongside the plaquette
                target_tags = tuple(
                    second_tag(second + d) for d in range(second_bsz)
                )
                edge_env = first_env.select_any(target_tags, virtual=False)
                environment = TensorNetwork((second_env, edge_env))
                environment.exponent += first_env.exponent
                # return the plaquette in the original coordinates
                p = (
                    r2d.rotate(first, second),
                    r2d.rotate(first_bsz, second_bsz),
                )
                yield p, environment


def _find_plaquette(where, Lx, Ly, cyclic_x, cyclic_y):
    coos = (where,) if is_lone_coo(where) else tuple(where)
    xs, ys = zip(*coos)
    i, x_bsz = find_1d_block(xs, Lx, cyclic_x)
    j, y_bsz = find_1d_block(ys, Ly, cyclic_y)
    return (i, j), (x_bsz, y_bsz)


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
            - 'reduce-split': factor the two physical indices into
              'R-factors' using QR decompositions on the original site
              tensors, then contract the gate, split it and reabsorb each
              side. Much cheaper than ``'split'``.
            - 'split': contract all involved tensors then split the result
              back into two.

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
        if self.has_site(where):
            where = (where,)
        else:
            where = tuple(where)

        # can just use generic arbgeom methods
        return super().gate(
            G=G,
            where=where,
            contract=contract,
            tags=tags,
            propagate_tags=propagate_tags,
            inplace=inplace,
            info=info,
            **compress_opts,
        )

    gate_ = functools.partialmethod(gate, inplace=True)

    def compute_norm(
        self,
        max_bond=None,
        *,
        cutoff=1e-10,
        canonize=True,
        method=None,
        layer_tags=("KET", "BRA"),
        compress_opts=None,
        sequence=None,
        equalize_norms=False,
        progbar=None,
        **contract_opts,
    ):
        """Compute the norm (squared) of this vector via boundary contraction.

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
        method : {'mps', 'full-bond', ...}, optional
            How to perform the compression on the boundary, can also be any of
            the generic 1D or arbgeom methods.
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
        equalize_norms : bool or float, optional
            Whether to equalize the norms of the boundary tensors after each
            contraction, gathering the overall scaling coefficient, log10, in
            ``tn.exponent``.
        progbar : bool, optional
            Whether to show a progress bar.
        contract_opts
            Additional options to pass to
            :meth:`~quimb.tensor.tn2d.core.TensorNetwork2D.contract_boundary`.

        Returns
        -------
        scalar
        """
        method = _parse_boundary_method(method, contract_opts)
        norm = self.make_norm(layer_tags=layer_tags)
        return norm.contract_boundary(
            max_bond=max_bond,
            cutoff=cutoff,
            canonize=canonize,
            method=method,
            layer_tags=layer_tags,
            compress_opts=compress_opts,
            sequence=sequence,
            equalize_norms=equalize_norms,
            progbar=progbar,
            # can perform boundary contraction inplace on new norm network
            inplace=True,
            # but want to unwrap final value, not leave as tensor network
            final_contract_opts={"inplace": False},
            **contract_opts,
        )

    def compute_partial_traces_boundary(
        self,
        wheres,
        max_bond,
        *,
        normalized=True,
        get="matrix",
        cutoff=None,
        canonize=True,
        method=None,
        layer_tags=("KET", "BRA"),
        autogroup=True,
        contract_optimize="auto-hq",
        plaquette_envs=None,
        plaquette_map=None,
        **plaquette_env_options,
    ):
        """Compute reduced density matrices using the boundary contraction
        plaquette environments of :meth:`compute_plaquette_environments`. Only
        for open boundaries.

        Parameters
        ----------
        wheres : sequence of coordinate or sequence of coordinates
            The site or sites to keep for each reduced density matrix, either
            a single coordinate ``(i, j)`` or a pair of coordinates.
        max_bond : int or None
            Maximum environment bond dimension, often called 'chi'. Supply
            ``None`` to truncate using only ``cutoff``. This is not recommended
            in 2D.
        normalized : bool or "return", optional
            Normalize each reduced density matrix to unit trace. If "return",
            give each as ``(rho, trace)`` without dividing by the trace.
            Ignored if ``get="tn"``, which returns the unnormalized network
            without a separate trace.
        get : {'matrix', 'array', 'tensor', 'tn'}, optional
            How to return each reduced density matrix, see
            :meth:`compute_partial_traces`.
        cutoff : float, optional
            Cut-off value to used to truncate singular values in the boundary
            contraction. By default ``1e-10``.
        canonize : bool, optional
            Whether to sweep one way with canonization before compressing.
        method : {'mps', 'full-bond', ...}, optional
            How to perform the compression on the boundary.
        layer_tags : None or sequence of str, optional
            If given, perform a multilayer contraction, contracting the inner
            sites in each layer into the boundary individually.
        autogroup : bool, optional
            If ``True`` (the default), group sites into horizontal and
            vertical sets to be computed separately (usually more efficient)
            if possible.
        contract_optimize : str, optional
            Contraction path finder to use for contracting each local
            reduced density matrix.
        plaquette_envs : None or dict, optional
            Supply precomputed plaquette environments.
        plaquette_map : None, dict, optional
            Supply the mapping of which plaquettes (denoted by
            ``((x0, y0), (dx, dy))``) to use for which coordinates, it will be
            calculated automatically otherwise.
        plaquette_env_options
            Supplied to :meth:`compute_plaquette_environments` to generate the
            plaquette environments.

        Returns
        -------
        dict[coordinate or tuple[coordinate], array or Tensor or TensorNetwork]
            The reduced density matrix for each ``where``, with sites in the
            order of ``where``.
        """
        method = _parse_boundary_method(method, plaquette_env_options)
        if cutoff is None:
            cutoff = 1e-10
        norm, ket, bra = self.make_norm(return_all=True)

        if plaquette_envs is None:
            plaquette_env_options["max_bond"] = max_bond
            plaquette_env_options["cutoff"] = cutoff
            plaquette_env_options["canonize"] = canonize
            plaquette_env_options["method"] = method
            plaquette_env_options["layer_tags"] = layer_tags

            plaquette_envs = {}
            for x_bsz, y_bsz in calc_plaquette_sizes(wheres, autogroup):
                plaquette_envs.update(
                    norm.compute_plaquette_environments(
                        x_bsz=x_bsz, y_bsz=y_bsz, **plaquette_env_options
                    )
                )

        if plaquette_map is None:
            # map each set of sites to a containing plaquette
            plaquette_map = calc_plaquette_map(plaquette_envs)

        # group site sets that share a plaquette
        wheres_by_plaquette = defaultdict(list)
        for where in wheres:
            sites = (where,) if is_lone_coo(where) else tuple(sorted(where))
            key = sites[0] if len(sites) == 1 else sites
            wheres_by_plaquette[plaquette_map[key]].append(where)

        rhos = {}
        for p, p_wheres in wheres_by_plaquette.items():
            tags = tuple(map(ket.site_tag, plaquette_to_sites(p)))
            ket_local = ket.select_any(tags, virtual=False)
            bra_local = bra.select_any(tags, virtual=False)
            rhos.update(
                partial_traces_from_environment(
                    self,
                    p_wheres,
                    ket_local,
                    bra_local,
                    plaquette_envs[p],
                    normalized=normalized,
                    get=get,
                    optimize=contract_optimize,
                )
            )

        return rhos

    def compute_partial_traces_via_envs(
        self,
        wheres,
        max_bond,
        *,
        normalized=True,
        get="matrix",
        autogroup=True,
        cyclic=None,
        first_contract=None,
        schedule="auto",
        second_schedule="auto",
        second_dense=None,
        method=None,
        layer_tags=("KET", "BRA"),
        cutoff=None,
        canonize=True,
        optimize="auto-hq",
        equalize_norms=False,
        compress_opts=None,
        contract_opts=None,
        **compress_method_opts,
    ):
        """Compute reduced density matrices for open or periodic boundaries.
        Use compressed plaquette environments from
        :meth:`compute_plaquette_environments_via_envs`. Share one sweep across
        all plaquette sizes in each first direction, then contract along each
        remaining strip.

        Parameters
        ----------
        wheres : sequence of coordinate or sequence of coordinates
            The site or sites to keep for each reduced density matrix. Sites
            can cross either periodic boundary.
        max_bond : int or None
            Maximum environment bond dimension, often called 'chi'. Supply
            ``None`` to truncate using only ``cutoff``. This is not recommended
            in 2D.
        normalized : bool or "return", optional
            Normalize each reduced density matrix to unit trace. If "return",
            give each as ``(rho, trace)`` without dividing by the trace.
            Ignored if ``get="tn"``, which returns the unnormalized network
            without a separate trace.
        get : {'matrix', 'array', 'tensor', 'tn'}, optional
            How to return each reduced density matrix, see
            :meth:`compute_partial_traces`.
        autogroup : bool, optional
            Use larger requested plaquette sizes to cover smaller ones. For
            requested sizes {3x1, 2x1, 2x2, 1x2, 1x3}, ``autogroup=True``
            uses {3x1, 2x2, 1x3}. With ``autogroup=False``, use 3x3 for all.

            For each set of sites, choose the smallest of these sizes that
            fits, by area. In this example, use 3x1 for the 2x1 sites and
            1x3 for the 1x2 sites. Enlarge the rectangle around the sites to
            that size, keeping its start coordinate. At open boundaries,
            shift it inward if needed to fit the lattice. Reuse an
            environment when both the plaquette size and start match.
        cyclic : bool or tuple[bool, bool], optional
            Periodicity in each direction. By default infer it.
        first_contract : {'x', 'y'}, optional
            Direction to compress first for all plaquettes. By default choose
            it per plaquette to keep the strip narrow, e.g. 'x' for 1x2 and 'y'
            for 2x1. This can require a sweep in each direction.
        schedule : {'auto', 'tree', 'cut'}, optional
            Construction schedule of the approximate environments in the
            first direction, see :meth:`gen_block_environments`.
        second_schedule : {'auto', 'tree', 'cut'}, optional
            Schedule along each strip. By default use 'cut' for exact
            contraction and 'tree' for compressed contraction.
        second_dense : bool, optional
            Whether to contract along each strip exactly. By default do so only
            for strips of width one. Compress wider strips.
        method : str or callable, optional
            Compression method. By default use the compressor's default for
            each direction. See
            :func:`~quimb.tensor.environments.gen_compressed_environments`.
        layer_tags : None or sequence[str], optional
            Contract the tensors of each plane one layer at a time, in this
            order. By default contract the ket and bra layers separately.
        cutoff : float, optional
            Compression cutoff. By default use the default of the compression
            function.
        canonize : bool or str, optional
            Canonicalization option supplied to the compressor.
        optimize : str, optional
            Contraction path optimizer supplied to the compressor, and the
            default for ``contract_opts``.
        equalize_norms : bool or float, optional
            Whether to equalize tensor norms after each compression.
        compress_opts : dict, optional
            Additional options supplied to the compressor.
        contract_opts : dict, optional
            Options for the exact contractions, of the environments along
            each strip if ``second_dense``, and of each reduced density
            matrix. ``optimize`` is the default path optimizer.
        compress_method_opts
            Additional options supplied to the compression method.

        Returns
        -------
        dict[coordinate or tuple[coordinate], array or Tensor or TensorNetwork]
            The reduced density matrix for each ``where``, with sites in the
            order of ``where``.
        """
        cyclic_x, cyclic_y = _normalize_2d_cyclic(self, cyclic)
        norm, ket, bra = self.make_norm(return_all=True)
        plaquettes = {
            where: _find_plaquette(where, self.Lx, self.Ly, cyclic_x, cyclic_y)
            for where in wheres
        }
        sizes = {size for _, size in plaquettes.values()}
        if autogroup:
            # drop sizes contained in another, e.g. 1x2 and 2x1 in 2x2
            sizes = {
                a
                for a in sizes
                if not any(
                    a != b and a[0] <= b[0] and a[1] <= b[1] for b in sizes
                )
            }
        else:
            # a single plaquette size that covers every set of sites
            sizes = {tuple(map(max, zip(*sizes)))}

        for where, ((i, j), (x_bsz, y_bsz)) in plaquettes.items():
            # use the smallest remaining size that contains these sites
            x_bsz, y_bsz = min(
                (b for b in sizes if x_bsz <= b[0] and y_bsz <= b[1]),
                key=lambda b: (b[0] * b[1], b),
            )
            if not cyclic_x:
                i = min(i, self.Lx - x_bsz)
            if not cyclic_y:
                j = min(j, self.Ly - y_bsz)
            plaquettes[where] = (i, j), (x_bsz, y_bsz)

        wheres_by_plaquette = defaultdict(list)
        for where, p in plaquettes.items():
            wheres_by_plaquette[p].append(where)

        contract_opts = ensure_dict(contract_opts)
        contract_opts.setdefault("optimize", optimize)
        environment_opts = dict(
            max_bond=max_bond,
            schedule=schedule,
            layer_tags=layer_tags,
            cutoff=cutoff,
            method=method,
            canonize=canonize,
            optimize=optimize,
            equalize_norms=equalize_norms,
            compress_opts=compress_opts,
            **compress_method_opts,
        )
        plaquette_envs = _gen_plaquette_environments_via_envs(
            norm,
            tuple(wheres_by_plaquette),
            cyclic_x,
            cyclic_y,
            first_contract,
            second_schedule,
            contract_opts,
            environment_opts,
            second_dense=second_dense,
        )

        rhos = {}
        for p, environment in plaquette_envs:
            (i, j), (x_bsz, y_bsz) = p
            tags = tuple(
                ket.site_tag((i + di) % self.Lx, (j + dj) % self.Ly)
                for di in range(x_bsz)
                for dj in range(y_bsz)
            )
            ket_local = ket.select_any(tags, virtual=False)
            bra_local = bra.select_any(tags, virtual=False)
            rhos.update(
                partial_traces_from_environment(
                    self,
                    wheres_by_plaquette[p],
                    ket_local,
                    bra_local,
                    environment,
                    normalized=normalized,
                    get=get,
                    **contract_opts,
                )
            )

        return rhos

    def compute_partial_traces(
        self,
        wheres,
        max_bond,
        *,
        cutoff=None,
        method=None,
        normalized=True,
        get="matrix",
        route=None,
        **kwargs,
    ):
        """Compute many local reduced density matrices at once, each from the
        environment of a plaquette containing its sites.

        Parameters
        ----------
        wheres : sequence of coordinate or sequence of coordinates
            The site or sites to keep for each reduced density matrix.
        max_bond : int or None
            Maximum environment bond dimension, often called 'chi'. Supply
            ``None`` to truncate using only ``cutoff``. This is not recommended
            in 2D.
        cutoff : float, optional
            The cutoff used when compressing the environments. By default use
            the default of the compression function, or ``1e-10`` for
            ``route='boundary'``.
        method : str or callable, optional
            The compression method, see
            :meth:`compute_partial_traces_boundary` for ``route='boundary'``
            and :meth:`compute_partial_traces_via_envs` for ``route='envs'``.
            By default use the default of each.
        normalized : bool or "return", optional
            Normalize each reduced density matrix to unit trace. If "return",
            give each as ``(rho, trace)`` without dividing by the trace.
            Ignored if ``get="tn"``, which returns the unnormalized network
            without a separate trace.
        get : {'matrix', 'array', 'tensor', 'tn'}, optional
            How to return each reduced density matrix:

            - 'matrix': a dense matrix, with the ket sites fused into rows
              and the bra sites fused into columns.
            - 'array': the raw array, with one axis per ket site then one
              axis per bra site.
            - 'tensor': a :class:`~quimb.tensor.tensor_core.Tensor` with the
              ket and bra indices.
            - 'tn': the uncontracted tensor network.
        route : {None, 'boundary', 'envs'}, optional
            How to compute the plaquette environments. By default use
            ``'envs'`` if either direction is periodic and ``'boundary'``
            otherwise.

            - 'boundary': boundary contraction from each side, see
              :meth:`compute_partial_traces_boundary`. Only for open
              boundaries.
            - 'envs': an approximate sweep in one direction, then sweeps
              along each strip in the other direction, see
              :meth:`compute_partial_traces_via_envs`.

        kwargs
            Supplied to the method chosen by ``route``.

        Returns
        -------
        dict[coordinate or tuple[coordinate], array or Tensor or TensorNetwork]
            The reduced density matrix for each ``where``, with sites in the
            order of ``where``.
        """
        if route is None:
            cyclic = self.is_cyclic_x() or self.is_cyclic_y()
            route = "envs" if cyclic else "boundary"
        check_opt("route", route, ("boundary", "envs"))

        if route == "envs":
            fn = self.compute_partial_traces_via_envs
        else:
            fn = self.compute_partial_traces_boundary
        return fn(
            wheres,
            max_bond,
            cutoff=cutoff,
            method=method,
            normalized=normalized,
            get=get,
            **kwargs,
        )

    def partial_trace(
        self,
        keep,
        max_bond,
        *,
        cutoff=None,
        method=None,
        normalized=True,
        get="matrix",
        route=None,
        **kwargs,
    ):
        """Compute the reduced density matrix of the sites ``keep``. See
        :meth:`compute_partial_traces` for the options, and for many sets of
        sites at once.

        Parameters
        ----------
        keep : coordinate or sequence of coordinates
            The site or sites to keep.
        max_bond : int or None
            Maximum environment bond dimension, often called 'chi'. Supply
            ``None`` to truncate using only ``cutoff``. This is not recommended
            in 2D.
        cutoff : float, optional
            The cutoff used when compressing the environments. By default use
            the default of the compression function, or ``1e-10`` for
            ``route='boundary'``.
        method : str or callable, optional
            The compression method, see
            :meth:`compute_partial_traces_boundary` for ``route='boundary'``
            and :meth:`compute_partial_traces_via_envs` for ``route='envs'``.
            By default use the default of each.
        normalized : bool or "return", optional
            Normalize the reduced density matrix to unit trace. If "return",
            give ``(rho, trace)`` without dividing by the trace. Ignored if
            ``get="tn"``, which returns the unnormalized network without a
            separate trace.
        get : {'matrix', 'array', 'tensor', 'tn'}, optional
            How to return the reduced density matrix, see
            :meth:`compute_partial_traces`.
        route : {None, 'boundary', 'envs'}, optional
            How to compute the plaquette environment, see
            :meth:`compute_partial_traces`.
        kwargs
            Supplied to :meth:`compute_partial_traces`.

        Returns
        -------
        array or Tensor or TensorNetwork or (array, float) or (Tensor, float)
        """
        keep = tuple(keep) if is_lone_coo(keep) else tuple(map(tuple, keep))
        return self.compute_partial_traces(
            (keep,),
            max_bond,
            cutoff=cutoff,
            method=method,
            normalized=normalized,
            get=get,
            route=route,
            **kwargs,
        )[keep]

    def compute_local_expectation_via_envs(
        self,
        terms,
        max_bond,
        *,
        cutoff=None,
        method=None,
        normalized=True,
        return_all=False,
        **kwargs,
    ):
        """Compute many local expectations at once, as ``tr(rho G)`` with each
        reduced density matrix ``rho`` from
        :meth:`compute_partial_traces_via_envs`, for open or periodic
        boundaries.

        Parameters
        ----------
        terms : dict[coordinate or tuple[coordinate], array_like]
            The local terms to compute values for. Sites can cross either
            periodic boundary.
        max_bond : int or None
            Maximum environment bond dimension, often called 'chi'. Supply
            ``None`` to truncate using only ``cutoff``. This is not recommended
            in 2D.
        cutoff : float, optional
            Compression cutoff. By default use the default of the compression
            function.
        method : str or callable, optional
            The compression method, see
            :meth:`compute_partial_traces_via_envs`.
        normalized : bool or "return", optional
            Normalize each local reduced density matrix to unit trace. If
            "return" and ``return_all=True``, give each term as
            ``(expec, trace)`` without dividing by the trace.
        return_all : bool, optional
            Whether to return each expectation in ``terms`` separately or sum
            them all together (the default).
        kwargs
            Supplied to :meth:`compute_partial_traces_via_envs`.

        Returns
        -------
        scalar or dict
        """
        return self.compute_local_expectation(
            terms,
            max_bond,
            cutoff=cutoff,
            method=method,
            normalized=normalized,
            return_all=return_all,
            route="envs",
            **kwargs,
        )

    def compute_local_expectation(
        self,
        terms,
        max_bond,
        *,
        cutoff=None,
        method=None,
        normalized=True,
        return_all=False,
        route=None,
        **kwargs,
    ):
        r"""Compute local expectations as ``tr(rho G)``, using reduced density
        matrices from :meth:`compute_partial_traces`. Return their sum by
        default.

        By default normalize each expectation by its local trace,
        :math:`\langle O_i \rangle = Tr[\rho_p O_i] / Tr[\rho_p]`. This avoids
        a separate norm contraction and usually improves accuracy.

        Parameters
        ----------
        terms : dict[coordinate or tuple[coordinate], array_like]
            Local operators keyed by site or sites. Use ``(i, j)`` for one site
            or ``((i_a, j_a), (i_b, j_b), ...)`` for several. Each operator can
            be a matrix or an array with one axis per ket site followed by one
            per bra site.
        max_bond : int or None
            Maximum environment bond dimension, often called 'chi'. Supply
            ``None`` to truncate using only ``cutoff``. This is not recommended
            in 2D.
        cutoff : float, optional
            The cutoff used when compressing the environments. By default use
            the default of the compression function, or ``1e-10`` for
            ``route='boundary'``.
        method : str or callable, optional
            The compression method, see
            :meth:`compute_partial_traces_boundary` for ``route='boundary'``
            and :meth:`compute_partial_traces_via_envs` for ``route='envs'``.
            By default use the default of each.
        normalized : bool or "return", optional
            Normalize each local reduced density matrix to unit trace. If
            "return" and ``return_all=True``, give each term as
            ``(expec, trace)`` without dividing by the trace.
        return_all : bool, optional
            Whether to return each expectation in ``terms`` separately or sum
            them all together (the default).
        route : {None, 'boundary', 'envs'}, optional
            How to compute the plaquette environments, see
            :meth:`compute_partial_traces`. By default use ``'envs'`` if
            either direction is periodic and ``'boundary'`` otherwise.
        kwargs
            Supplied to :meth:`compute_partial_traces_boundary` or
            :meth:`compute_partial_traces_via_envs`, depending on ``route``.

        Returns
        -------
        scalar or dict
        """
        rhos = self.compute_partial_traces(
            tuple(terms),
            max_bond,
            cutoff=cutoff,
            method=method,
            normalized=normalized,
            get="array",
            route=route,
            **kwargs,
        )
        return expectations_from_rhos(
            terms, rhos, normalized=normalized, return_all=return_all
        )

    def local_expectation(
        self,
        G,
        where,
        max_bond,
        *,
        cutoff=None,
        method=None,
        normalized=True,
        route=None,
        **kwargs,
    ):
        """Compute the local expectation ``tr(rho G)`` of operator ``G`` at
        sites ``where``, with ``rho`` from :meth:`partial_trace`. See
        :meth:`compute_local_expectation` for many terms at once.

        Parameters
        ----------
        G : array_like
            The local operator, as a matrix or with one axis per ket site
            then one axis per bra site.
        where : coordinate or sequence of coordinates
            The site or sites to compute the expectation at.
        max_bond : int or None
            Maximum environment bond dimension, often called 'chi'. Supply
            ``None`` to truncate using only ``cutoff``. This is not recommended
            in 2D.
        cutoff : float, optional
            The cutoff used when compressing the environments. By default use
            the default of the compression function, or ``1e-10`` for
            ``route='boundary'``.
        method : str or callable, optional
            The compression method, see
            :meth:`compute_partial_traces_boundary` for ``route='boundary'``
            and :meth:`compute_partial_traces_via_envs` for ``route='envs'``.
            By default use the default of each.
        normalized : bool or "return", optional
            Normalize the reduced density matrix to unit trace. If "return",
            give ``(expec, trace)`` without dividing by the trace.
        route : {None, 'boundary', 'envs'}, optional
            How to compute the plaquette environment, see
            :meth:`compute_partial_traces`.
        kwargs
            Supplied to :meth:`compute_partial_traces`.

        Returns
        -------
        scalar or (scalar, scalar)
        """
        where = (
            tuple(where) if is_lone_coo(where) else tuple(map(tuple, where))
        )
        return self.compute_local_expectation(
            {where: G},
            max_bond,
            cutoff=cutoff,
            method=method,
            normalized=normalized,
            return_all=True,
            route=route,
            **kwargs,
        )[where]

    def normalize(
        self,
        max_bond=None,
        *,
        cutoff=1e-10,
        canonize=True,
        method=None,
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
        method : {'mps', 'full-bond', ...}, optional
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
            Supplied to
            :meth:`~quimb.tensor.tn2d.core.TensorNetwork2D.contract_boundary`,
            by default, two layer contraction will be used.
        """
        method = _parse_boundary_method(method, contract_boundary_opts)
        contract_boundary_opts["max_bond"] = max_bond
        contract_boundary_opts["cutoff"] = cutoff
        contract_boundary_opts["canonize"] = canonize
        contract_boundary_opts["method"] = method
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
    arrays : sequence of sequence of array_like
        The core tensor data arrays.
    shape : str, optional
        Which order the dimensions of the arrays are supplied in, the default
        ``'urdlp'`` stands for ('up', 'right', 'down', 'left', 'physical').
        Arrays on the edge of lattice are assumed to be missing the
        corresponding dimension. Internally, the arrays are stored 'urdlp'.
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

        shape_on_xmin_edge = ar.shape(arrays[0][self._Ly // 2])
        ndim_xmin_edge = len(shape_on_xmin_edge)
        shape_on_ymin_edge = ar.shape(arrays[self._Lx // 2][0])
        ndim_ymin_edge = len(shape_on_ymin_edge)

        cyclicx = (sum(d > 1 for d in shape_on_xmin_edge) == 5) or (
            # handle D=1 PBC case
            (ndim_xmin_edge == 5)
            and (sum(d == 1 for d in shape_on_xmin_edge) == 4)
        )
        cyclicy = (sum(d > 1 for d in shape_on_ymin_edge) == 5) or (
            # handle D=1 PBC case
            (ndim_ymin_edge == 5)
            and (sum(d == 1 for d in shape_on_ymin_edge) == 4)
        )

        # cache for both creating and retrieving bond indices
        bond = LatticeBondMap(self.Lx, self.Ly)
        tensors = []

        for i, j in self.gen_site_coos():
            array = arrays[i][j]

            # figure out if we need to transpose the arrays from some order
            #     other than up right down left physical
            array_order = shape
            if (not cyclicx) and (i == self.Lx - 1):
                array_order = array_order.replace("u", "")
            if (not cyclicy) and (j == self.Ly - 1):
                array_order = array_order.replace("r", "")
            if (not cyclicx) and (i == 0):
                array_order = array_order.replace("d", "")
            if (not cyclicy) and (j == 0):
                array_order = array_order.replace("l", "")

            # allow convention of missing bonds to be singlet dimensions
            if ar.ndim(array) != len(array_order):
                array = ar.do("squeeze", array)

            transpose_order = tuple(
                array_order.find(x) for x in "urdlp" if x in array_order
            )
            if transpose_order != tuple(range(len(array_order))):
                array = ar.do("transpose", array, transpose_order)

            # get the relevant indices corresponding to neighbours
            inds = []
            if "u" in array_order:
                inds.append(bond((i, j), (i + 1, j)))
            if "r" in array_order:
                inds.append(bond((i, j), (i, j + 1)))
            if "d" in array_order:
                inds.append(bond((i, j), (i - 1, j)))
            if "l" in array_order:
                inds.append(bond((i, j), (i, j - 1)))
            inds.append(self.site_ind(i, j))

            # mix site, row, column and global tags
            ij_tags = tags | oset(
                (self.site_tag(i, j), self.x_tag(i), self.y_tag(j))
            )

            # create the site tensor!
            tensors.append(Tensor(data=array, inds=inds, tags=ij_tags))

        super().__init__(tensors, virtual=True, **tn_opts)

    @classmethod
    def from_fill_fn(
        cls,
        fill_fn,
        Lx,
        Ly,
        bond_dim,
        phys_dim=2,
        cyclic=False,
        shape="urdlp",
        **peps_opts,
    ):
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
        phys_dim : int, optional
            The physical index dimension.
        cyclic : bool or tuple[bool, bool], optional
            Whether the lattice is cyclic in the x and y directions.
        shape : str, optional
            How to layout the indices of the tensors, the default is
            ``(up, right, down, left, phys) == 'urdlp'``. This is the order
            of the shape supplied to the filling function.
        peps_opts
            Supplied to :class:`~quimb.tensor.tn2d.core.PEPS`.

        Returns
        -------
        psi : PEPS
        """
        arrays = [[None for _ in range(Ly)] for _ in range(Lx)]

        try:
            cyclicx, cyclicy = cyclic
        except (TypeError, ValueError):
            cyclicx = cyclicy = cyclic

        for i, j in product(range(Lx), range(Ly)):
            shp = []

            for which in shape:
                if (
                    # bond up
                    ((which == "u") and (cyclicx or (i < Lx - 1)))
                    # bond right
                    or ((which == "r") and (cyclicy or (j < Ly - 1)))
                    # bond down
                    or ((which == "d") and (cyclicx or (i > 0)))
                    # bond left
                    or ((which == "l") and (cyclicy or (j > 0)))
                ):
                    shp.append(bond_dim)
                elif which == "p":
                    shp.append(phys_dim)

            arrays[i][j] = fill_fn(shp)

        return cls(arrays, shape=shape, **peps_opts)

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
            Supplied to :class:`~quimb.tensor.tn2d.core.PEPS`.

        Returns
        -------
        psi : PEPS

        See Also
        --------
        PEPS.from_fill_fn
        """
        return cls.from_fill_fn(
            lambda shape: ar.do("zeros", shape, like=like),
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
            Supplied to :class:`~quimb.tensor.tn2d.core.PEPS`.

        Returns
        -------
        psi : PEPS

        See Also
        --------
        PEPS.from_fill_fn
        """
        return cls.from_fill_fn(
            lambda shape: ar.do("ones", shape, like=like),
            Lx,
            Ly,
            bond_dim,
            phys_dim,
            **peps_opts,
        )

    @classmethod
    def zeros(cls, Lx, Ly, bond_dim, phys_dim=2, like="numpy", **peps_opts):
        """Create a 2D PEPS whose tensors are filled with zeros.

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
            Supplied to :class:`~quimb.tensor.tn2d.core.PEPS`.

        Returns
        -------
        psi : PEPS

        See Also
        --------
        PEPS.from_fill_fn
        """
        return cls.from_fill_fn(
            lambda shape: ar.do("zeros", shape, like=like),
            Lx,
            Ly,
            bond_dim,
            phys_dim,
            **peps_opts,
        )

    @classmethod
    def rand(
        cls,
        Lx,
        Ly,
        bond_dim,
        phys_dim=2,
        dist="normal",
        loc=0.0,
        scale=1.0,
        dtype="float64",
        seed=None,
        **peps_opts,
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
        dist : {'normal', 'uniform', 'rademacher', 'exp'}, optional
            Type of random number to generate, defaults to 'normal'.
        loc : float, optional
            An additive offset to add to the random numbers.
        dtype : dtype, optional
            The dtype to create the arrays with, default is real double.
        seed : int, optional
            A random seed.
        peps_opts
            Supplied to :class:`~quimb.tensor.tn2d.core.PEPS`.

        Returns
        -------
        psi : PEPS

        See Also
        --------
        PEPS.from_fill_fn
        """
        fill_fn = get_rand_fill_fn(
            dist=dist,
            loc=loc,
            scale=scale,
            dtype=dtype,
            seed=seed,
        )

        return cls.from_fill_fn(
            fill_fn, Lx, Ly, bond_dim, phys_dim, **peps_opts
        )

    @classmethod
    def product_state(cls, site_map, cyclic=False, **peps_opts):
        """Create a PEPS representing a product state, with explicit bonds of
        dimension 1 between sites.

        Parameters
        ----------
        site_map : dict[tuple[int, int], array] or Sequence[Sequence[array]]
            A mapping of site coordinates to physical vectors, or a 2D array of
            physical vectors. Each vector being a single site state.
        cyclic : bool or tuple[bool, bool], optional
            Whether the lattice is cyclic in the x and y directions.
        peps_opts
            Supplied to :class:`~quimb.tensor.tn2d.core.PEPS`.

        Returns
        -------
        PEPS
        """
        try:
            cyclicx, cyclicy = cyclic
        except (TypeError, ValueError):
            cyclicx = cyclicy = cyclic

        if isinstance(site_map, dict):
            getarray = site_map.get
            Lx = max(i for i, j in site_map) + 1
            Ly = max(j for i, j in site_map) + 1
        else:

            def getarray(ij):
                i, j = ij
                return site_map[i][j]

            Lx = len(site_map)
            Ly = len(site_map[0])

        arrays = [[None for _ in range(Ly)] for _ in range(Lx)]

        for i in range(Lx):
            for j in range(Ly):
                bond_shape = []
                if cyclicx or (i < Lx - 1):  # bond up
                    bond_shape.append(1)
                if cyclicy or (j < Ly - 1):  # bond right
                    bond_shape.append(1)
                if cyclicx or (i > 0):  # bond down
                    bond_shape.append(1)
                if cyclicy or (j > 0):  # bond left
                    bond_shape.append(1)

                ary = getarray((i, j))
                new_shape = (*bond_shape, *ar.shape(ary))
                arrays[i][j] = ar.do("reshape", ary, new_shape)

        return cls(arrays, **peps_opts)

    @classmethod
    def vacuum(cls, Lx, Ly, phys_dim=2, **peps_opts):
        """Create the 'vaccum' state PEPS, i.e. |00...0>.

        Parameters
        ----------
        Lx : int
            The number of rows.
        Ly : int
            The number of columns.
        phys_dim : int, optional
            The physical index dimension.
        peps_opts
            Supplied to :class:`~quimb.tensor.tn2d.core.PEPS.product_state`.

        Returns
        -------
        """
        data = ar.do("array", [1.0] + [0.0] * (phys_dim - 1))
        site_map = {(i, j): data for i in range(Lx) for j in range(Ly)}
        return cls.product_state(site_map, **peps_opts)

    def add_PEPS(self, other, inplace=False):
        return tensor_network_ag_sum(self, other, inplace=inplace)

    add_PEPS_ = functools.partialmethod(add_PEPS, inplace=True)

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
        What order the dimensions of the arrays are supplied in, the default
        ``'urdlbk'`` stands for ('up' / x+, 'right' / y+, 'down' / x-,
        'left' / y-, 'bra', 'ket').
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
    cyclic : None, bool, or tuple[bool, bool], optional
        Whether the lattice is cyclic in the x and y directions. If ``None``
        (default), infer from the array shapes (requires any non-singleton bond
        dimension). Pass an explicit ``bool`` or ``(cyclic_x, cyclic_y)`` to
        override inference; this is needed when bond dimensions are 1 (shape
        inference then cannot detect cyclic boundaries).
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
        cyclic=None,
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

        # cache for both creating and retrieving bond indices
        bond = LatticeBondMap(self.Lx, self.Ly)
        tensors = []

        if cyclic is None:
            # infer boundary conditions from array shapes
            shape_on_xmin_edge = ar.shape(arrays[0][self._Ly // 2])
            shape_on_ymin_edge = ar.shape(arrays[self._Lx // 2][0])
            ndim_xmin_edge = len(shape_on_xmin_edge)
            ndim_ymin_edge = len(shape_on_ymin_edge)

            cyclicx = (sum(d > 1 for d in shape_on_xmin_edge) == 6) or (
                # handle D=1 PBC case
                (ndim_xmin_edge == 6)
                and (sum(d == 1 for d in shape_on_xmin_edge) == 4)
            )
            cyclicy = (sum(d > 1 for d in shape_on_ymin_edge) == 6) or (
                # handle D=1 PBC case
                (ndim_ymin_edge == 6)
                and (sum(d == 1 for d in shape_on_ymin_edge) == 4)
            )
        else:
            try:
                cyclicx, cyclicy = cyclic
            except (TypeError, ValueError):
                cyclicx = cyclicy = cyclic

        for i, j in product(range(self.Lx), range(self.Ly)):
            array = arrays[i][j]

            # figure out if we need to transpose the arrays from some order
            #     other than up right down left physical
            array_order = shape
            if (not cyclicx) and (i == self.Lx - 1):
                array_order = array_order.replace("u", "")
            if (not cyclicy) and (j == self.Ly - 1):
                array_order = array_order.replace("r", "")
            if (not cyclicx) and (i == 0):
                array_order = array_order.replace("d", "")
            if (not cyclicy) and (j == 0):
                array_order = array_order.replace("l", "")

            # allow convention of missing bonds to be singlet dimensions
            if ar.ndim(array) != len(array_order):
                array = ar.do("squeeze", array)

            transpose_order = tuple(
                array_order.find(x) for x in "urdlbk" if x in array_order
            )
            if transpose_order != tuple(range(len(array_order))):
                array = ar.do("transpose", array, transpose_order)

            # get the relevant indices corresponding to neighbours
            inds = []
            if "u" in array_order:
                inds.append(bond((i, j), (i + 1, j)))
            if "r" in array_order:
                inds.append(bond((i, j), (i, j + 1)))
            if "d" in array_order:
                inds.append(bond((i, j), (i - 1, j)))
            if "l" in array_order:
                inds.append(bond((i, j), (i, j - 1)))
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
    def from_fill_fn(
        cls,
        fill_fn,
        Lx,
        Ly,
        bond_dim,
        phys_dim=2,
        cyclic=False,
        shape="urdlbk",
        **pepo_opts,
    ):
        """Create a PEPO and fill the tensor entries with a supplied function
        matching signature ``fill_fn(shape) -> array``.

        Parameters
        ----------
        fill_fn : callable
            A function that takes a shape tuple and returns a data array.
        Lx : int
            The number of rows.
        Ly : int
            The number of columns.
        bond_dim : int
            The bond dimension.
        phys_dim : int, optional
            The physical indices dimension.
        cyclic : bool or tuple[bool, bool], optional
            Whether the lattice is cyclic in the x and y directions.
        shape : str, optional
            How to layout the indices of the tensors, the default is
            ``(up, right, down, left bra, ket) == 'urdlbk'``.
        pepo_opts
            Supplied to :class:`~quimb.tensor.tn2d.core.PEPO`.
        """
        arrays = [[None for _ in range(Ly)] for _ in range(Lx)]

        try:
            cyclicx, cyclicy = cyclic
        except (TypeError, ValueError):
            cyclicx = cyclicy = cyclic

        for i, j in product(range(Lx), range(Ly)):
            shp = []
            for which in shape:
                if (
                    ((which == "u") and (cyclicx or (i < Lx - 1)))
                    or ((which == "r") and (cyclicy or (j < Ly - 1)))
                    or ((which == "d") and (cyclicx or (i > 0)))
                    or ((which == "l") and (cyclicy or (j > 0)))
                ):
                    shp.append(bond_dim)
                elif which in ("b", "k"):
                    shp.append(phys_dim)

            arrays[i][j] = fill_fn(shp)

        return cls(arrays, shape=shape, cyclic=cyclic, **pepo_opts)

    @classmethod
    def rand(
        cls,
        Lx,
        Ly,
        bond_dim,
        phys_dim=2,
        herm=False,
        dist="normal",
        loc=0.0,
        scale=1.0,
        dtype="float64",
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
            Supplied to :class:`~quimb.tensor.tn2d.core.PEPO`.

        Returns
        -------
        X : PEPO
        """
        fill_fn = get_rand_fill_fn(
            dist=dist,
            loc=loc,
            scale=scale,
            dtype=dtype,
            seed=seed,
        )

        if herm:
            _fill_fn_orig = fill_fn

            def fill_fn(shape):
                X = _fill_fn_orig(shape)
                new_order = list(range(len(shape)))
                new_order[-2], new_order[-1] = new_order[-1], new_order[-2]
                return (
                    ar.do("conj", X) + ar.do("transpose", X, new_order)
                ) / 2

        return cls.from_fill_fn(
            fill_fn,
            Lx=Lx,
            Ly=Ly,
            bond_dim=bond_dim,
            phys_dim=phys_dim,
            **pepo_opts,
        )

    rand_herm = functools.partialmethod(rand, herm=True)

    @classmethod
    def zeros(
        cls,
        Lx,
        Ly,
        bond_dim,
        phys_dim=2,
        dtype="float64",
        backend="numpy",
        **pepo_opts,
    ):
        """Create a PEPO with all zero entries.

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
        backend : str, optional
            Which backend to use, default is ``'numpy'``.
        pepo_opts
            Supplied to :class:`~quimb.tensor.tn2d.core.PEPO`.
        """

        def fill_fn(shape):
            return ar.do("zeros", shape, dtype=dtype, like=backend)

        return cls.from_fill_fn(
            fill_fn,
            Lx=Lx,
            Ly=Ly,
            bond_dim=bond_dim,
            phys_dim=phys_dim,
            **pepo_opts,
        )

    def add_PEPO(self, other, inplace=False):
        return tensor_network_ag_sum(self, other, inplace=inplace)

    add_PEPO_ = functools.partialmethod(add_PEPO, inplace=True)

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

    mapping = {}
    for p in plqs:
        sites = plaquette_to_sites(p)
        for site in sites:
            mapping[site] = p
        # this will generate all coordinate pairs with ij_a < ij_b
        for ij_a, ij_b in combinations(sites, 2):
            mapping[ij_a, ij_b] = p

    return mapping


def tensor_network_2d_distance(
    a, b, xAA=None, xAB=None, xBB=None, normalized=False, **kwargs
):
    a = a.copy()
    b = b.copy()
    a.add_tag("__A__")
    b.add_tag("__B__")

    tnAA = a.H.retag_({"__A__": "__Adag__"}) & a
    tnBB = b.H.retag_({"__B__": "__Bdag__"}) & b
    tnAB = b.H & a

    if xAA is None:
        kwargs_aa = kwargs.copy()
        kwargs_aa.setdefault("layer_tags", ["__Adag__", "__A__"])
        xAA = tnAA.contract_boundary(**kwargs_aa)

    if xAB is None:
        kwargs_ab = kwargs.copy()
        kwargs_ab.setdefault("layer_tags", ["__A__", "__B__"])
        xAB = tnAB.contract_boundary(**kwargs_ab)

    if xBB is None:
        kwargs_bb = kwargs.copy()
        kwargs_bb.setdefault("layer_tags", ["__Bdag__", "__B__"])
        xBB = tnBB.contract_boundary(**kwargs_bb)

    dAB = ar.do("abs", xAA - 2 * ar.do("real", xAB) + xBB) ** 0.5

    if normalized:
        dAB *= 2 / (ar.do("abs", xAA) ** 0.5 + ar.do("abs", xBB) ** 0.5)

    return dAB
