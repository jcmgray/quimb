"""Classes and algorithms related to 3D tensor networks.
"""

import functools
import itertools
from operator import add
from numbers import Integral
from collections import defaultdict
from itertools import product, combinations

from autoray import do, dag

from ..utils import check_opt, ensure_dict, pairwise
from ..utils import progbar as Progbar
from ..gen.rand import randn, seed_rand
from . import array_ops as ops
from .tensor_core import (
    bonds_size,
    oset,
    rand_uuid,
    tags_to_oset,
    Tensor,
)
from .tensor_arbgeom import (
    TensorNetworkGen,
    TensorNetworkGenVector,
)


def gen_3d_bonds(Lx, Ly, Lz, steppers=None, coo_filter=None):
    """Convenience function for tiling pairs of bond coordinates on a 3D
    lattice given a function like ``lambda i, j, k: (i + 1, j + 1, k + 1)``.

    Parameters
    ----------
    Lx : int
        The number of x-slices.
    Ly : int
        The number of y-slices.
    Lz : int
        The number of z-slices.
    steppers : callable or sequence of callable
        Function(s) that take args ``(i, j, k)`` and generate another
        coordinate, thus defining a bond. Only valid steps are taken. If not
        given, defaults to nearest neighbor bonds.
    coo_filter : callable
        Function that takes args ``(i, j, k)`` and only returns ``True`` if
        this is to be a valid starting coordinate.

    Yields
    ------
    bond : tuple[tuple[int, int, int], tuple[int, int, int]]
        A pair of coordinates.

    Examples
    --------

    Generate nearest neighbor bonds:

        >>> for bond in gen_3d_bonds(2, 2, 2, [lambda i, j, k: (i + 1, j, k),
        ...                                    lambda i, j, k: (i, j + 1, k),
        ...                                    lambda i, j, k: (i, j, k + 1)]):
        ...     print(bond)
        ((0, 0, 0), (1, 0, 0))
        ((0, 0, 0), (0, 1, 0))
        ((0, 0, 0), (0, 0, 1))
        ((0, 0, 1), (1, 0, 1))
        ((0, 0, 1), (0, 1, 1))
        ((0, 1, 0), (1, 1, 0))
        ((0, 1, 0), (0, 1, 1))
        ((0, 1, 1), (1, 1, 1))
        ((1, 0, 0), (1, 1, 0))
        ((1, 0, 0), (1, 0, 1))
        ((1, 0, 1), (1, 1, 1))
        ((1, 1, 0), (1, 1, 1))

    """
    if steppers is None:
        steppers = [
            lambda i, j, k: (i, j, k + 1),
            lambda i, j, k: (i, j + 1, k),
            lambda i, j, k: (i + 1, j, k),
        ]

    if callable(steppers):
        steppers = (steppers,)

    for i, j, k in product(range(Lx), range(Ly), range(Lz)):
        if (coo_filter is None) or coo_filter(i, j, k):
            for stepper in steppers:
                i2, j2, k2 = stepper(i, j, k)
                if (0 <= i2 < Lx) and (0 <= j2 < Ly) and (0 <= k2 < Lz):
                    yield (i, j, k), (i2, j2, k2)


def gen_3d_plaquette(coo0, steps):
    """Generate a plaquette at site ``coo0`` by stepping first in ``steps`` and
    then the reverse steps.

    Parameters
    ----------
    coo0 : tuple
        The coordinate of the first site in the plaquette.
    steps : tuple
        The steps to take to generate the plaquette. Each element should be
        one of ``('x+', 'x-', 'y+', 'y-', 'z+', 'z-')``.

    Yields
    ------
    coo : tuple
        The coordinates of the sites in the plaquette, including the last
        site which will be the same as the first.
    """
    x, y, z = coo0
    smap = {"+": +1, "-": -1}
    step_backs = []
    yield x, y, z
    for step in steps:
        d, s = step
        x, y, z = {
            "x": (x + smap[s], y, z),
            "y": (x, y + smap[s], z),
            "z": (x, y, z + smap[s]),
        }[d]
        yield x, y, z
        step_backs.append(d + "-" if s == "+" else "-")
    for step in step_backs:
        d, s = step
        x, y, z = {
            "x": (x + smap[s], y, z),
            "y": (x, y + smap[s], z),
            "z": (x, y, z + smap[s]),
        }[d]
        yield x, y, z


def gen_3d_plaquettes(Lx, Ly, Lz, tiling="1"):
    """Generate a tiling of plaquettes in a cubic 3D lattice.

    Parameters
    ----------
    Lx : int
        The length of the lattice in the x direction.
    Ly : int
        The length of the lattice in the y direction.
    Lz : int
        The length of the lattice in the z direction.
    tiling : {'1', '2', '4', 'full'}
        The tiling to use:

        - '1': plaquettes in a sparse checkerboard pattern, such that each edge
            is covered by a maximum of one plaquette.
        - '2': less sparse checkerboard pattern, such that each edge is
            covered by a maximum of two plaquettes.
        - '4' or 'full': dense tiling of plaquettes. All bulk edges will
            be covered four times.

    Yields
    ------
    plaquette : tuple[tuple[int]]
        The coordinates of the sites in each plaquette, including the last
        site which will be the same as the first.
    """
    if isinstance(tiling, int):
        tiling = str(tiling)

    if tiling == "1":
        for x, y, z in itertools.product(range(Lx), range(Ly), range(Lz)):
            if (x % 2 == 0) and (y % 2 == 0) and (x < Lx - 1 and y < Ly - 1):
                yield tuple(gen_3d_plaquette((x, y, z), ("x+", "y+")))
            if (y % 2 == 1) and (z % 2 == 0) and (y < Ly - 1 and z < Lz - 1):
                yield tuple(gen_3d_plaquette((x, y, z), ("y+", "z+")))
            if (z % 2 == 1) and (x % 2 == 1) and (z < Lz - 1 and x < Lx - 1):
                yield tuple(gen_3d_plaquette((x, y, z), ("z+", "x+")))
    elif tiling == "2":
        for x, y, z in itertools.product(range(Lx), range(Ly), range(Lz)):
            if ((x + y) % 2 == 0) and (x < Lx - 1 and y < Ly - 1):
                yield tuple(gen_3d_plaquette((x, y, z), ("x+", "y+")))
            if ((y + z) % 2 == 0) and (y < Ly - 1 and z < Lz - 1):
                yield tuple(gen_3d_plaquette((x, y, z), ("y+", "z+")))
            if ((x + z) % 2 == 1) and (z < Lz - 1 and x < Lx - 1):
                yield tuple(gen_3d_plaquette((x, y, z), ("z+", "x+")))
    elif tiling in ("4", "full"):
        for x, y, z in itertools.product(range(Lx), range(Ly), range(Lz)):
            if x < Lx - 1 and y < Ly - 1:
                yield tuple(gen_3d_plaquette((x, y, z), ("x+", "y+")))
            if y < Ly - 1 and z < Lz - 1:
                yield tuple(gen_3d_plaquette((x, y, z), ("y+", "z+")))
            if z < Lz - 1 and x < Lx - 1:
                yield tuple(gen_3d_plaquette((x, y, z), ("z+", "x+")))
    else:
        raise ValueError(
            "Invalid tiling: {}. Must be one of '1', '2', '4', 'full'."
        )


def gen_3d_strings(Lx, Ly, Lz):
    """Generate all length-wise strings in a cubic 3D lattice."""
    for x, y in itertools.product(range(Lx), range(Ly)):
        yield tuple((x, y, z) for z in range(Lz))
    for y, z in itertools.product(range(Ly), range(Lz)):
        yield tuple((x, y, z) for x in range(Lx))
    for x, z in itertools.product(range(Lx), range(Lz)):
        yield tuple((x, y, z) for y in range(Ly))


class Rotator3D:
    """Object for rotating coordinates and various contraction functions so
    that the core algorithms only have to written once, but nor does the actual
    TN have to be modified.
    """

    def __init__(self, tn, xrange, yrange, zrange, from_which):
        check_opt(
            "from_which",
            from_which,
            {"xmin", "xmax", "ymin", "ymax", "zmin", "zmax"},
        )

        if xrange is None:
            xrange = (0, tn.Lx - 1)
        if yrange is None:
            yrange = (0, tn.Ly - 1)
        if zrange is None:
            zrange = (0, tn.Lz - 1)

        self.xrange = xrange
        self.yrange = yrange
        self.zrange = zrange
        self.from_which = from_which
        self.plane = from_which[0]

        if self.plane == "x":
            # -> no rotation needed
            self.imin, self.imax = sorted(xrange)
            self.jmin, self.jmax = sorted(yrange)
            self.kmin, self.kmax = sorted(zrange)
            self.x_tag = tn.x_tag
            self.y_tag = tn.y_tag
            self.z_tag = tn.z_tag
            self.site_tag = tn.site_tag

        elif self.plane == "y":
            # -> (y, z, x)
            self.imin, self.imax = sorted(yrange)
            self.jmin, self.jmax = sorted(zrange)
            self.kmin, self.kmax = sorted(xrange)
            self.x_tag = tn.y_tag
            self.y_tag = tn.z_tag
            self.z_tag = tn.x_tag
            self.site_tag = lambda i, j, k: tn.site_tag(k, i, j)

        else:  # self.plane == 'z'
            # -> (z, x, y)
            self.imin, self.imax = sorted(zrange)
            self.jmin, self.jmax = sorted(xrange)
            self.kmin, self.kmax = sorted(yrange)
            self.x_tag = tn.z_tag
            self.y_tag = tn.x_tag
            self.z_tag = tn.y_tag
            self.site_tag = lambda i, j, k: tn.site_tag(j, k, i)

        if "min" in self.from_which:
            # -> sweeps are increasing
            self.sweep = range(self.imin, self.imax + 1, +1)
            self.istep = +1
        else:  # 'max'
            # -> sweeps are decreasing
            self.sweep = range(self.imax, self.imin - 1, -1)
            self.istep = -1


# reference for viewing a cube from each direction
#
#      ┌──┐        ┌──┐        ┌──┐        ┌──┐        ┌──┐        ┌──┐
#      │y+│        │z+│        │x-│        │y-│        │z-│        │x+│
#   ┌──┼──┼──┐  ┌──┼──┼──┐  ┌──┼──┼──┐  ┌──┼──┼──┐  ┌──┼──┼──┐  ┌──┼──┼──┐
#   │z-│x-│z+│  │x-│y-│x+│  │y+│z-│y-│  │z-│x+│z+│  │x-│y+│x+│  │y+│z+│y-│
#   └──┼──┼──┘, └──┼──┼──┘, └──┼──┼──┘, └──┼──┼──┘, └──┼──┼──┘, └──┼──┼──┘
#      │y-│        │z-│        │x+│        │y+│        │z+│        │x-│
#      └──┘        └──┘        └──┘        └──┘        └──┘        └──┘


_canonize_plane_opts = {
    "xmin": {
        "yreverse": False,
        "zreverse": False,
        "coordinate_order": "yz",
        "stepping_order": "zy",
    },
    "ymin": {
        "zreverse": False,
        "xreverse": True,
        "coordinate_order": "zx",
        "stepping_order": "xz",
    },
    "zmin": {
        "xreverse": True,
        "yreverse": True,
        "coordinate_order": "xy",
        "stepping_order": "yx",
    },
    "xmax": {
        "yreverse": True,
        "zreverse": True,
        "coordinate_order": "yz",
        "stepping_order": "zy",
    },
    "ymax": {
        "zreverse": True,
        "xreverse": False,
        "coordinate_order": "zx",
        "stepping_order": "xz",
    },
    "zmax": {
        "xreverse": False,
        "yreverse": False,
        "coordinate_order": "xy",
        "stepping_order": "yx",
    },
}


_compress_plane_opts = {
    "xmin": {
        "yreverse": True,
        "zreverse": True,
        "coordinate_order": "yz",
        "stepping_order": "zy",
    },
    "ymin": {
        "zreverse": True,
        "xreverse": False,
        "coordinate_order": "zx",
        "stepping_order": "xz",
    },
    "zmin": {
        "xreverse": False,
        "yreverse": False,
        "coordinate_order": "xy",
        "stepping_order": "yx",
    },
    "xmax": {
        "yreverse": False,
        "zreverse": False,
        "coordinate_order": "yz",
        "stepping_order": "zy",
    },
    "ymax": {
        "zreverse": False,
        "xreverse": True,
        "coordinate_order": "zx",
        "stepping_order": "xz",
    },
    "zmax": {
        "xreverse": True,
        "yreverse": True,
        "coordinate_order": "xy",
        "stepping_order": "yx",
    },
}

BOUNDARY_SEQUENCE_MAP = {
    "xmin": "xmin",
    "xmax": "xmax",
    "ymin": "ymin",
    "ymax": "ymax",
    "zmin": "zmin",
    "zmax": "zmax",
}


def parse_boundary_sequence(sequence):
    if isinstance(sequence, str):
        if sequence in BOUNDARY_SEQUENCE_MAP:
            return (sequence,)
    return tuple(BOUNDARY_SEQUENCE_MAP[s] for s in sequence)


class TensorNetwork3D(TensorNetworkGen):
    _NDIMS = 3
    _EXTRA_PROPS = (
        "_site_tag_id",
        "_x_tag_id",
        "_y_tag_id",
        "_z_tag_id",
        "_Lx",
        "_Ly",
        "_Lz",
    )

    def _compatible_3d(self, other):
        """Check whether ``self`` and ``other`` are compatible 3D tensor
        networks such that they can remain a 3D tensor network when combined.
        """
        return isinstance(other, TensorNetwork3D) and all(
            getattr(self, e) == getattr(other, e)
            for e in TensorNetwork3D._EXTRA_PROPS
        )

    def combine(self, other, *, virtual=False, check_collisions=True):
        """Combine this tensor network with another, returning a new tensor
        network. If the two are compatible, cast the resulting tensor network
        to a :class:`TensorNetwork3D` instance.

        Parameters
        ----------
        other : TensorNetwork3D or TensorNetwork
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
        TensorNetwork3D or TensorNetwork
        """
        new = super().combine(
            other, virtual=virtual, check_collisions=check_collisions
        )
        if self._compatible_3d(other):
            new.view_as_(TensorNetwork3D, like=self)
        return new

    @property
    def Lx(self):
        """The number of x-slices."""
        return self._Lx

    @property
    def Ly(self):
        """The number of y-slices."""
        return self._Ly

    @property
    def Lz(self):
        """The number of z-slices."""
        return self._Lz

    @property
    def nsites(self):
        """The total number of sites."""
        return self._Lx * self._Ly * self._Lz

    def site_tag(self, i, j=None, k=None):
        """The name of the tag specifiying the tensor at site ``(i, j, k)``."""
        if j is None:
            i, j, k = i
        if not isinstance(i, str):
            i = i % self.Lx
        if not isinstance(j, str):
            j = j % self.Ly
        if not isinstance(k, str):
            k = k % self.Lz
        return self.site_tag_id.format(i, j, k)

    @property
    def x_tag_id(self):
        """The string specifier for tagging each x-slice of this 3D TN."""
        return self._x_tag_id

    def x_tag(self, i):
        if not isinstance(i, str):
            i = i % self.Lx
        return self.x_tag_id.format(i)

    @property
    def x_tags(self):
        """A tuple of all of the ``Lx`` different x-slice tags."""
        return tuple(map(self.x_tag, range(self.Lx)))

    @property
    def y_tag_id(self):
        """The string specifier for tagging each y-slice of this 3D TN."""
        return self._y_tag_id

    def y_tag(self, j):
        if not isinstance(j, str):
            j = j % self.Ly
        return self.y_tag_id.format(j)

    @property
    def y_tags(self):
        """A tuple of all of the ``Ly`` different y-slice tags."""
        return tuple(map(self.y_tag, range(self.Ly)))

    @property
    def z_tag_id(self):
        """The string specifier for tagging each z-slice of this 3D TN."""
        return self._z_tag_id

    def z_tag(self, k):
        if not isinstance(k, str):
            k = k % self.Lz
        return self.z_tag_id.format(k)

    @property
    def z_tags(self):
        """A tuple of all of the ``Lz`` different z-slice tags."""
        return tuple(map(self.z_tag, range(self.Lz)))

    def maybe_convert_coo(self, coo):
        """Check if ``coo`` is a tuple of three ints and convert to the
        corresponding site tag if so.
        """
        if not isinstance(coo, str):
            try:
                i, j, k = map(int, coo)
                return self.site_tag(i, j, k)
            except (ValueError, TypeError):
                pass
        return coo

    def _get_tids_from_tags(self, tags, which="all"):
        """This is the function that lets coordinates such as ``(i, j, k)`` be
        used for many 'tag' based functions.
        """
        tags = self.maybe_convert_coo(tags)
        return super()._get_tids_from_tags(tags, which=which)

    def gen_site_coos(self):
        """Generate coordinates for all the sites in this 3D TN."""
        return product(range(self.Lx), range(self.Ly), range(self.Lz))

    def gen_bond_coos(self):
        """Generate pairs of coordinates for all the bonds in this 3D TN."""
        return gen_3d_bonds(
            self.Lx,
            self.Ly,
            self.Lz,
            steppers=[
                lambda i, j, k: (i + 1, j, k),
                lambda i, j, k: (i, j + 1, k),
                lambda i, j, k: (i, j, k + 1),
            ],
        )

    def valid_coo(self, coo, xrange=None, yrange=None, zrange=None):
        """Check whether ``coo`` is in-bounds.

        Parameters
        ----------
        coo : (int, int, int), optional
            The coordinates to check.
        xrange, yrange, zrange : (int, int), optional
            The range of allowed values for the x, y, and z coordinates.

        Returns
        -------
        bool
        """
        if xrange is None:
            xrange = (0, self.Lx - 1)
        if yrange is None:
            yrange = (0, self.Ly - 1)
        if zrange is None:
            zrange = (0, self.Lz - 1)
        return all(
            mn <= u <= mx for u, (mn, mx) in zip(coo, (xrange, yrange, zrange))
        )

    def get_ranges_present(self):
        """Return the range of site coordinates present in this TN.

        Returns
        -------
        xrange, yrange, zrange : tuple[tuple[int, int]]
            The minimum and maximum site coordinates present in each direction.

        Examples
        --------

            >>> tn = qtn.TN3D_rand(4, 4, 4, 2)
            >>> tn_sub = tn.select_local('I1,2,3', max_distance=1)
            >>> tn_sub.get_ranges_present()
            ((0, 2), (1, 3), (2, 3))

        """
        xmin = ymin = zmin = float("inf")
        xmax = ymax = zmax = float("-inf")
        for i, j, k in self.gen_sites_present():
            xmin = min(i, xmin)
            ymin = min(j, ymin)
            zmin = min(k, zmin)
            xmax = max(i, xmax)
            ymax = max(j, ymax)
            zmax = max(k, zmax)
        return (xmin, xmax), (ymin, ymax), (zmin, zmax)

    def __getitem__(self, key):
        """Key based tensor selection, checking for integer based shortcut."""
        return super().__getitem__(self.maybe_convert_coo(key))

    def _repr_info(self):
        info = super()._repr_info()
        info["Lx"] = self.Lx
        info["Ly"] = self.Ly
        info["Lz"] = self.Lz
        info["max_bond"] = self.max_bond()
        return info

    def flatten(self, fuse_multibonds=True, inplace=False):
        """Contract all tensors corresponding to each site into one."""
        tn = self if inplace else self.copy()

        for i, j, k in self.gen_site_coos():
            tn ^= (i, j, k)

        if fuse_multibonds:
            tn.fuse_multibonds_()

        return tn.view_as_(TensorNetwork3DFlat, like=self)

    flatten_ = functools.partialmethod(flatten, inplace=True)

    def gen_pairs(
        self,
        xrange=None,
        yrange=None,
        zrange=None,
        xreverse=False,
        yreverse=False,
        zreverse=False,
        coordinate_order="xyz",
        xstep=None,
        ystep=None,
        zstep=None,
        stepping_order="xyz",
        step_only=None,
    ):
        """Helper function for generating pairs of cooordinates for all bonds
        within a certain range, optionally specifying an order.

        Parameters
        ----------
        xrange, yrange, zrange : (int, int), optional
            The range of allowed values for the x, y, and z coordinates.
        xreverse, yreverse, zreverse : bool, optional
            Whether to reverse the order of the x, y, and z sweeps.
        coordinate_order : str, optional
            The order in which to sweep the x, y, and z coordinates. Earlier
            dimensions will change slower. If the corresponding range has
            size 1 then that dimension doesn't need to be specified.
        xstep, ystep, zstep : int, optional
            When generating a bond, step in this direction to yield the
            neighboring coordinate. By default, these follow ``xreverse``,
            ``yreverse``, and ``zreverse`` respectively.
        stepping_order : str, optional
            The order in which to step the x, y, and z coordinates to generate
            bonds. Does not need to include all dimensions.
        step_only : int, optional
            Only perform the ith steps in ``stepping_order``, used to
            interleave canonizing and compressing for example.

        Yields
        ------
        coo_a, coo_b : ((int, int, int), (int, int, int))
        """
        if xrange is None:
            xrange = (0, self.Lx - 1)
        if yrange is None:
            yrange = (0, self.Ly - 1)
        if zrange is None:
            zrange = (0, self.Lz - 1)

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
            "z": (
                range(min(zrange), max(zrange) + 1, +1)
                if not zreverse
                else range(max(zrange), min(zrange) - 1, -1)
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
        if zstep is None:
            zstep = -1 if zreverse else +1
        steps = {
            "x": lambda i, j, k: (i + xstep, j, k),
            "y": lambda i, j, k: (i, j + ystep, k),
            "z": lambda i, j, k: (i, j, k + zstep),
        }

        # make sure all coordinates exist - only allow them not to be specified
        # if their range is a unit slice
        for w in "xyz":
            if w not in coordinate_order:
                if len(sweeps[w]) > 1:
                    raise ValueError(
                        f"{w} not in coordinate_order and is not size 1."
                    )
                else:
                    # just append -> it won't change order as coord is constant
                    coordinate_order += w
        xi, yi, zi = map(coordinate_order.index, "xyz")

        # generate the pairs
        for perm_coo_a in product(*(sweeps[xyz] for xyz in coordinate_order)):
            coo_a = perm_coo_a[xi], perm_coo_a[yi], perm_coo_a[zi]
            for xyz in stepping_order:
                coo_b = steps[xyz](*coo_a)
                # filter out bonds with are out of bounds
                if self.valid_coo(coo_b, xrange, yrange, zrange):
                    yield coo_a, coo_b

    def canonize_plane(
        self,
        xrange,
        yrange,
        zrange,
        equalize_norms=False,
        canonize_opts=None,
        **gen_pair_opts,
    ):
        """Canonize every pair of tensors within a subrange, optionally
        specifying a order to visit those pairs in.
        """
        canonize_opts = ensure_dict(canonize_opts)
        canonize_opts.setdefault("equalize_norms", equalize_norms)

        pairs = self.gen_pairs(
            xrange=xrange,
            yrange=yrange,
            zrange=zrange,
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

            self.canonize_between(tag_a, tag_b, **canonize_opts)

    def compress_plane(
        self,
        xrange,
        yrange,
        zrange,
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
        compress_opts.setdefault("absorb", "both")
        compress_opts.setdefault("equalize_norms", equalize_norms)

        pairs = self.gen_pairs(
            xrange=xrange,
            yrange=yrange,
            zrange=zrange,
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

    def _contract_boundary_core(
        self,
        xrange,
        yrange,
        zrange,
        from_which,
        max_bond,
        cutoff=1e-10,
        canonize=True,
        canonize_interleave=True,
        layer_tags=None,
        compress_late=True,
        equalize_norms=False,
        compress_opts=None,
        canonize_opts=None,
    ):
        canonize_opts = ensure_dict(canonize_opts)
        canonize_opts.setdefault("absorb", "right")
        compress_opts = ensure_dict(compress_opts)
        compress_opts.setdefault("absorb", "both")

        r3d = Rotator3D(self, xrange, yrange, zrange, from_which)
        site_tag = r3d.site_tag
        plane, istep = r3d.plane, r3d.istep
        jmin, jmax = r3d.jmin, r3d.jmax
        kmin, kmax = r3d.kmin, r3d.kmax

        if canonize_interleave:
            # interleave canonizing and compressing in each direction
            step_onlys = [0, 1]
        else:
            # perform whole sweep of canonizing before compressing
            step_onlys = [None]

        if layer_tags is None:
            layer_tags = [None]

        for i in r3d.sweep[:-1]:
            for layer_tag in layer_tags:
                for j in range(jmin, jmax + 1):
                    for k in range(kmin, kmax + 1):
                        tag1 = site_tag(i, j, k)  # outer
                        tag2 = site_tag(i + istep, j, k)  # inner

                        if (tag1 not in self.tag_map) or (
                            tag2 not in self.tag_map
                        ):
                            # allow completely missing sites
                            continue

                        if (layer_tag is None) or len(self.tag_map[tag2]) == 1:
                            # contract *any* tensors with pair of coordinates
                            self.contract_((tag1, tag2), which="any")
                        else:
                            # ensure boundary is single tensor per site
                            if len(self.tag_map[tag1]) > 1:
                                self ^= tag1

                            # contract specific pair (i.e. one 'inner' layer)
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
                            (tid1,) = self.tag_map[tag1]
                            for tidn in self._get_neighbor_tids(tid1):
                                t1, tn = self._tids_get(tid1, tidn)
                                if bonds_size(t1, tn) > max_bond:
                                    self._compress_between_tids(
                                        tid1,
                                        tidn,
                                        max_bond=max_bond,
                                        cutoff=cutoff,
                                        equalize_norms=equalize_norms,
                                        **compress_opts,
                                    )

                if compress_late:
                    for step_only in step_onlys:
                        if canonize:
                            self.canonize_plane(
                                xrange=xrange if plane != "x" else (i, i),
                                yrange=yrange if plane != "y" else (i, i),
                                zrange=zrange if plane != "z" else (i, i),
                                equalize_norms=equalize_norms,
                                canonize_opts=canonize_opts,
                                step_only=step_only,
                                **_canonize_plane_opts[from_which],
                            )
                        self.compress_plane(
                            xrange=xrange if plane != "x" else (i, i),
                            yrange=yrange if plane != "y" else (i, i),
                            zrange=zrange if plane != "z" else (i, i),
                            max_bond=max_bond,
                            cutoff=cutoff,
                            equalize_norms=equalize_norms,
                            compress_opts=compress_opts,
                            step_only=step_only,
                            **_compress_plane_opts[from_which],
                        )

    def _contract_boundary_projector(
        self,
        xrange,
        yrange,
        zrange,
        from_which,
        max_bond=None,
        cutoff=1e-10,
        canonize=False,
        layer_tags=None,
        lazy=False,
        equalize_norms=False,
        optimize="auto-hq",
        compress_opts=None,
        canonize_opts=None,
    ):
        compress_opts = ensure_dict(compress_opts)

        r = Rotator3D(self, xrange, yrange, zrange, from_which)

        if canonize:
            canonize_opts = ensure_dict(canonize_opts)
            canonize_opts.setdefault("max_iterations", 2)
            self.select(r.x_tag(r.sweep[0])).gauge_all_(**canonize_opts)

        sweep_other = tuple(
            itertools.product(
                range(r.jmin, r.jmax + 1), range(r.kmin, r.kmax + 1)
            )
        )

        for i0, i1 in pairwise(r.sweep):
            # we compute the projectors from an untouched copy
            tn_calc = self.copy()

            for j, k in sweep_other:
                tag_ijk = r.site_tag(i0, j, k)
                tag_ip1jk = r.site_tag(i1, j, k)
                rtags = (tag_ijk, tag_ip1jk)

                poss_ltags = []
                if j != r.jmin:
                    poss_ltags.append(
                        (r.site_tag(i0, j - 1, k), r.site_tag(i1, j - 1, k))
                    )
                if k != r.kmin:
                    poss_ltags.append(
                        (r.site_tag(i0, j, k - 1), r.site_tag(i1, j, k - 1))
                    )

                for ltags in poss_ltags:
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
                for j, k in sweep_other:
                    self.contract_tags_(
                        (r.site_tag(i0, j, k), r.site_tag(i1, j, k)),
                        optimize=optimize,
                    )

            if equalize_norms:
                for t in self.select_tensors(r.x_tag(i1)):
                    self.strip_exponent(t, equalize_norms)

    def contract_boundary_from(
        self,
        xrange,
        yrange,
        zrange,
        from_which,
        max_bond=None,
        *,
        cutoff=1e-10,
        mode="peps",
        equalize_norms=False,
        compress_opts=None,
        canonize_opts=None,
        inplace=False,
        **contract_boundary_opts,
    ):
        """Unified entrypoint for contracting any rectangular patch of tensors
        from any direction, with any boundary method.
        """
        check_opt("mode", mode, ("peps", "projector"))

        tn = self if inplace else self.copy()

        # universal options
        contract_boundary_opts["xrange"] = xrange
        contract_boundary_opts["yrange"] = yrange
        contract_boundary_opts["zrange"] = zrange
        contract_boundary_opts["from_which"] = from_which
        contract_boundary_opts["max_bond"] = max_bond
        contract_boundary_opts["cutoff"] = cutoff
        contract_boundary_opts["equalize_norms"] = equalize_norms
        contract_boundary_opts["compress_opts"] = compress_opts

        if mode == "projector":
            return tn._contract_boundary_projector(**contract_boundary_opts)

        # mode == 'peps' options
        return tn._contract_boundary_core(
            canonize_opts=canonize_opts,
            **contract_boundary_opts,
        )

    contract_boundary_from_ = functools.partialmethod(
        contract_boundary_from, inplace=True
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
        zmin=None,
        zmax=None,
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
                f"contracting boundary, Lx={tn.Lx}, Ly={tn.Ly}, Lz={tn.Lz}"
            )
        else:
            pbar = None

        # set default starting borders
        if any(d is None for d in (xmin, xmax, ymin, ymax, zmin, zmax)):
            (
                (auto_xmin, auto_xmax),
                (auto_ymin, auto_ymax),
                (auto_zmin, auto_zmax),
            ) = self.get_ranges_present()

        # location of current boundaries
        boundaries = {
            "xmin": auto_xmin if xmin is None else xmin,
            "xmax": auto_xmax if xmax is None else xmax,
            "ymin": auto_ymin if ymin is None else ymin,
            "ymax": auto_ymax if ymax is None else ymax,
            "zmin": auto_zmin if zmin is None else zmin,
            "zmax": auto_zmax if zmax is None else zmax,
        }
        separations = {
            d: boundaries[f"{d}max"] - boundaries[f"{d}min"] for d in "xyz"
        }
        boundary_tags = {
            "xmin": tn.x_tag(boundaries["xmin"]),
            "xmax": tn.x_tag(boundaries["xmax"]),
            "ymin": tn.y_tag(boundaries["ymin"]),
            "ymax": tn.y_tag(boundaries["ymax"]),
            "zmin": tn.z_tag(boundaries["zmin"]),
            "zmax": tn.z_tag(boundaries["zmax"]),
        }
        if around is not None:
            target_xmin = min(x[0] for x in around)
            target_xmax = max(x[0] for x in around)
            target_ymin = min(x[1] for x in around)
            target_ymax = max(x[1] for x in around)
            target_zmin = min(x[2] for x in around)
            target_zmax = max(x[2] for x in around)
            target_check = {
                "xmin": lambda x: x >= target_xmin - 1,
                "xmax": lambda x: x <= target_xmax + 1,
                "ymin": lambda y: y >= target_ymin - 1,
                "ymax": lambda y: y <= target_ymax + 1,
                "zmin": lambda z: z >= target_zmin - 1,
                "zmax": lambda z: z <= target_zmax + 1,
            }

        if sequence is None:
            sequence = ("xmin", "xmax", "ymin", "ymax", "zmin", "zmax")
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
                    f"Ly={separations['y'] + 1}, "
                    f"Lz={separations['z'] + 1}"
                )

            if canonize:
                tn.select(boundary_tags[direction]).gauge_all_(**canonize_opts)

            if direction[0] == "x":
                if direction[1:] == "min":
                    xrange = (boundaries["xmin"], boundaries["xmin"] + 1)
                else:  # xmax
                    xrange = (boundaries["xmax"] - 1, boundaries["xmax"])
                yrange = (boundaries["ymin"], boundaries["ymax"])
                zrange = (boundaries["zmin"], boundaries["zmax"])
            elif direction[0] == "y":
                if direction[1:] == "min":
                    yrange = (boundaries["ymin"], boundaries["ymin"] + 1)
                else:  # ymax
                    yrange = (boundaries["ymax"] - 1, boundaries["ymax"])
                xrange = (boundaries["xmin"], boundaries["xmax"])
                zrange = (boundaries["zmin"], boundaries["zmax"])
            else:  # z
                if direction[1:] == "min":
                    zrange = (boundaries["zmin"], boundaries["zmin"] + 1)
                else:  # zmax
                    zrange = (boundaries["zmax"] - 1, boundaries["zmax"])
                xrange = (boundaries["xmin"], boundaries["xmax"])
                yrange = (boundaries["ymin"], boundaries["ymax"])

            # do the contractions!
            tn.contract_boundary_from_(
                xrange=xrange,
                yrange=yrange,
                zrange=zrange,
                from_which=direction,
                equalize_norms=equalize_norms,
                **contract_boundary_opts,
            )

            # update the boundaries and separations
            xyz, minmax = direction[0], direction[1:]
            separations[xyz] -= 1
            if minmax == "min":
                boundaries[direction] += 1
            else:
                boundaries[direction] -= 1

            if pbar is not None:
                pbar.update()

            # check if enough directions are finished -> reached max separation
            if (
                sum(separations[d] > max_separation for d in "xyz")
                <= max_unfinished
            ):
                break

        if equalize_norms is True:
            tn.equalize_norms_()

        if pbar is not None:
            pbar.set_description(
                f"contracted boundary, "
                f"Lx={separations['x'] + 1}, "
                f"Ly={separations['y'] + 1}, "
                f"Lz={separations['z'] + 1}"
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
        mode="peps",
        canonize=True,
        compress_opts=None,
        sequence=None,
        xmin=None,
        xmax=None,
        ymin=None,
        ymax=None,
        zmin=None,
        zmax=None,
        max_separation=1,
        max_unfinished=1,
        around=None,
        equalize_norms=False,
        final_contract=True,
        final_contract_opts=None,
        progbar=False,
        inplace=False,
        **contract_boundary_opts,
    ):
        """ """
        contract_boundary_opts["max_bond"] = max_bond
        contract_boundary_opts["cutoff"] = cutoff
        contract_boundary_opts["mode"] = mode
        contract_boundary_opts["canonize"] = canonize
        contract_boundary_opts["compress_opts"] = compress_opts

        return self._contract_interleaved_boundary_sequence(
            contract_boundary_opts=contract_boundary_opts,
            sequence=sequence,
            xmin=xmin,
            xmax=xmax,
            ymin=ymin,
            ymax=ymax,
            zmin=zmin,
            zmax=zmax,
            max_separation=max_separation,
            max_unfinished=max_unfinished,
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

    def _compute_plane_envs(
        self,
        xrange,
        yrange,
        zrange,
        from_which,
        envs=None,
        storage_factory=None,
        **contract_boundary_opts,
    ):
        """Compute all 'plane' environments for the cube given by
        ``xrange``, ``yrange``, ``zrange``, with direction given by
        ``from_which``.
        """
        tn = self.copy()

        # rotate virtually
        r3d = Rotator3D(tn, xrange, yrange, zrange, from_which)
        plane, p_tag, istep = r3d.plane, r3d.x_tag, r3d.istep

        # 0th plane has no environment, 1st plane's environment is the 0th
        if envs is None:
            if storage_factory is not None:
                envs = storage_factory()
            else:
                envs = {}

        envs[r3d.sweep[1]] = tn.select_any(p_tag(r3d.sweep[0]), virtual=False)

        for i in r3d.sweep[:-2]:
            # contract the boundary in one step
            tn.contract_boundary_from_(
                xrange=xrange if plane != "x" else (i, i + istep),
                yrange=yrange if plane != "y" else (i, i + istep),
                zrange=zrange if plane != "z" else (i, i + istep),
                from_which=from_which,
                **contract_boundary_opts,
            )
            # set the boundary as the environment for the next plane beyond
            envs[i + 2 * istep] = tn.select_any(
                p_tag(i + istep), virtual=False
            )

        return envs

    def _maybe_compute_cell_env(
        self,
        key,
        envs=None,
        storage_factory=None,
        boundary_order=None,
        **contract_boundary_opts,
    ):
        """Recursively compute the necessary 2D, 1D, and 0D environments."""
        if not key:
            # env is the whole TN
            return self.copy()

        if envs is None:
            if storage_factory is not None:
                envs = storage_factory()
            else:
                envs = {}

        if key in envs:
            return envs[key].copy()

        if boundary_order is None:
            scores = {"x": -self.Lx, "y": -self.Ly, "z": -self.Lz}
            boundary_order = sorted(scores, key=scores.__getitem__)

        # check already available parent environments
        for parent_key in itertools.combinations(key, len(key) - 1):
            if parent_key in envs:
                parent_tn = envs[parent_key].copy()
                break
        else:
            # choose which to compute next based on `boundary_order`
            ranked = sorted(key, key=lambda x: boundary_order.index(x[0]))[:-1]
            parent_key = tuple(sorted(ranked))

            # else choose a parent to compute
            parent_tn = self._maybe_compute_cell_env(
                key=parent_key,
                envs=envs,
                storage_factory=storage_factory,
                boundary_order=boundary_order,
                **contract_boundary_opts,
            )

        # need to compute parent first - first set the parents range
        Ls = {"x": self.Lx, "y": self.Ly, "z": self.Lz}
        plane_tags = {"x": self.x_tag, "y": self.y_tag, "z": self.z_tag}
        ranges = {"xrange": None, "yrange": None, "zrange": None}
        for d, s, bsz in parent_key:
            ranges[f"{d}range"] = (max(0, s - 1), min(s + bsz, Ls[d] - 1))

        # then compute the envs for the new direction ``d``
        ((d, _, bsz),) = (x for x in key if x not in parent_key)

        envs_s_min = parent_tn._compute_plane_envs(
            from_which=f"{d}min",
            storage_factory=storage_factory,
            **ranges,
            **contract_boundary_opts,
        )
        envs_s_max = parent_tn._compute_plane_envs(
            from_which=f"{d}max",
            storage_factory=storage_factory,
            **ranges,
            **contract_boundary_opts,
        )

        for s in range(0, Ls[d] - bsz + 1):
            # the central non-boundary slice of tensors
            tags_s = tuple(map(plane_tags[d], range(s, s + bsz)))
            tn_s = parent_tn.select_any(tags_s, virtual=False)

            # the min boundary
            if s in envs_s_min:
                tn_s &= envs_s_min[s]
            # the max boundary
            imax = s + bsz - 1
            if imax in envs_s_max:
                tn_s &= envs_s_max[imax]

            # store the newly created cell along with env
            key_s = tuple(sorted((*parent_key, (d, s, bsz))))
            envs[key_s] = tn_s

        # one of key_s == key
        return envs[key].copy()

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
        direction : {'x', 'y', 'z'}
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
        contract_hotrg, insert_compressor_between_regions
        """
        compress_opts = ensure_dict(compress_opts)
        check_opt("direction", direction, ("x", "y", "z"))

        tn = self if inplace else self.copy()
        tn_calc = tn.copy()

        r = Rotator3D(tn, None, None, None, direction + "min")

        retag_map = {}

        for i in range(r.imin, r.imax + 1, 2):
            next_i_in_lattice = i + 1 <= r.imax
            for j in range(r.jmin, r.jmax + 1):
                for k in range(r.kmin, r.kmax + 1):
                    tag_ijk = r.site_tag(i, j, k)
                    tag_ip1jk = r.site_tag(i + 1, j, k)
                    new_tag = r.site_tag(i // 2, j, k)
                    retag_map[tag_ijk] = new_tag
                    if next_i_in_lattice:
                        retag_map[tag_ip1jk] = new_tag

                    # insert the 'y'-orientated projectors
                    if (j > 0) and next_i_in_lattice:
                        ltags = (
                            r.site_tag(i, j - 1, k),
                            r.site_tag(i + 1, j - 1, k),
                        )
                        rtags = (tag_ijk, tag_ip1jk)
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

                    # insert the 'z'-orientated projectors
                    if (k > 0) and next_i_in_lattice:
                        ltags = (
                            r.site_tag(i, j, k - 1),
                            r.site_tag(i + 1, j, k - 1),
                        )
                        rtags = (tag_ijk, tag_ip1jk)
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
        elif direction == "y":
            tn._Ly = tn.Ly // 2 + tn.Ly % 2
        else:  # direction == "z":
            tn._Lz = tn.Lz // 2 + tn.Lz % 2

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
        cutoff=1e-10,
        canonize=False,
        canonize_opts=None,
        sequence=("x", "y", "z"),
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
        ``sequence`` directions by inserting oblique projectors between tensor
        pairs, and then optionally contracting these new effective sites. The
        algorithm stops when only one direction has a length larger than 2, and
        thus exact contraction can be used.

        Parameters
        ----------
        max_bond : int, optional
            The maximum bond dimension of the projector pairs inserted.
        cutoff : float, optional
            The cutoff for the singular values of the projector pairs.
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
            Whether to equalize the norms of the tensors in the tensor network
            after each coarse graining step.
        final_contract : bool, optional
            Whether to exactly contract the remaining tensor network after the
            coarse graining contractions.
        final_contract_opts : None or dict, optional
            Options to pass to :meth:`contract`, ``optimize`` defaults to
            ``'auto-hq'``.
        inplace : bool, optional
            Whether to perform the coarse graining in place.
        coarse_grain_opts
            Additional options to pass to :meth:`coarse_grain_hotrg`.

        Returns
        -------
        TensorNetwork3D
            The contracted tensor network, which will have no more than one
            directino of length > 2.

        See Also
        --------
        coarse_grain_hotrg, insert_compressor_between_regions
        """
        tn = self if inplace else self.copy()

        if lazy:
            # we are implicitly asking for the tensor network
            final_contract = False

        if canonize:
            canonize_opts = ensure_dict(canonize_opts)
            canonize_opts.setdefault("max_iterations", 2)

        if progbar:
            pbar = Progbar(
                desc=f"contracting HOTRG, Lx={tn.Lx}, Ly={tn.Ly}, Lz={tn.Lz}"
            )
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
                    f"contracting {direction}, "
                    f"Lx={tn.Lx}, Ly={tn.Ly}, Lz={tn.Lz}"
                )

            if canonize:
                tn.gauge_all_(**canonize_opts)

            # do the contractions!
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
            if sum(not _is_finished(d) for d in "xyz") <= max_unfinished:
                break

        if equalize_norms is True:
            # redistribute the exponent equally among all tensors
            tn.equalize_norms_()

        if pbar is not None:
            pbar.set_description(
                f"contracted HOTRG, " f"Lx={tn.Lx}, Ly={tn.Ly}, Lz={tn.Lz}"
            )
            pbar.close()

        if final_contract:
            final_contract_opts = ensure_dict(final_contract_opts)
            final_contract_opts.setdefault("optimize", "auto-hq")
            final_contract_opts.setdefault("inplace", inplace)
            return tn.contract(**final_contract_opts)

        return tn

    contract_hotrg_ = functools.partialmethod(contract_hotrg, inplace=True)


def is_lone_coo(where):
    """Check if ``where`` has been specified as a single coordinate triplet."""
    return (len(where) == 3) and (isinstance(where[0], Integral))


def calc_cell_sizes(coo_groups, autogroup=True):
    # get the rectangular size of each coordinate pair
    bszs = set()
    for coos in coo_groups:
        if is_lone_coo(coos):
            bszs.add((1, 1, 1))
            continue
        xs, ys, zs = zip(*coos)
        xsz = max(xs) - min(xs) + 1
        ysz = max(ys) - min(ys) + 1
        zsz = max(zs) - min(zs) + 1
        bszs.add((xsz, ysz, zsz))

    # remove block size pairs that can be contained in another block pair size
    bszs = tuple(
        sorted(
            b
            for b in bszs
            if not any(
                (b[0] <= b2[0]) and (b[1] <= b2[1]) and (b[2] <= b2[2])
                for b2 in bszs - {b}
            )
        )
    )

    # return each cell size separately
    if autogroup:
        return bszs

    # else choose a single blocksize that will cover all terms
    return (tuple(map(max, zip(*bszs))),)


def cell_to_sites(p):
    """Turn a cell ``((i0, j0, k0), (di, dj, dk))`` into the sites it contains.

    Examples
    --------

        >>> cell_to_sites([(3, 4), (2, 2)])
        ((3, 4), (3, 5), (4, 4), (4, 5))
    """
    (i0, j0, k0), (di, dj, dk) = p
    return tuple(
        (i, j, k)
        for i in range(i0, i0 + di)
        for j in range(j0, j0 + dj)
        for k in range(k0, k0 + dk)
    )


def sites_to_cell(sites):
    """Get the minimum covering cell for ``sites``.

    Examples
    --------

        >>> sites_to_cell([(1, 3, 3), (2, 2, 4)])
        ((1, 2, 3) , (2, 2, 2))
    """
    imin = jmin = kmin = float("inf")
    imax = jmax = kmax = float("-inf")
    for i, j, k in sites:
        imin, jmin, kmin = min(imin, i), min(jmin, j), min(kmin, k)
        imax, jmax, kmax = max(imax, i), max(jmax, j), max(kmax, k)
    x_bsz, y_bsz, z_bsz = imax - imin + 1, jmax - jmin + 1, kmax - kmin + 1
    return (imin, jmin, kmin), (x_bsz, y_bsz, z_bsz)


def calc_cell_map(cells):
    # sort in descending total cell size
    cs = sorted(cells, key=lambda c: (-c[1][0] * c[1][1] * c[1][2], c))

    mapping = dict()
    for c in cs:
        sites = cell_to_sites(c)
        for site in sites:
            mapping[site] = c
        # this will generate all coordinate pairs with ijk_a < ijk_b
        for ijk_a, ijk_b in combinations(sites, 2):
            mapping[ijk_a, ijk_b] = c

    return mapping


class TensorNetwork3DVector(TensorNetwork3D, TensorNetworkGenVector):
    """Mixin class  for a 3D square lattice vector TN, i.e. one with a single
    physical index per site.
    """

    _EXTRA_PROPS = (
        "_site_tag_id",
        "_x_tag_id",
        "_y_tag_id",
        "_z_tag_id",
        "_Lx",
        "_Ly",
        "_Lz",
        "_site_ind_id",
    )

    def site_ind(self, i, j=None, k=None):
        if j is None:
            i, j, k = i
        if not isinstance(i, str):
            i = i % self.Lx
        if not isinstance(j, str):
            j = j % self.Ly
        if not isinstance(k, str):
            k = k % self.Lz
        return self.site_ind_id.format(i, j, k)

    def reindex_sites(self, new_id, where=None, inplace=False):
        if where is None:
            where = self.gen_sites_present()

        return self.reindex(
            {self.site_ind(*coo): new_id.format(*coo) for coo in where},
            inplace=inplace,
        )

    def phys_dim(self, i=None, j=None, k=None):
        """Get the size of the physical indices / a specific physical index."""
        if (i is not None) and (j is not None) and (k is not None):
            pix = self.site_ind(i, j, k)
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
        info=None,
        inplace=False,
        **compress_opts,
    ):
        """Apply a gate ``G`` to sites ``where``, preserving the outer site
        inds.
        """
        if is_lone_coo(where):
            where = (where,)
        else:
            where = tuple(where)

        inds = tuple(map(self.site_ind, where))
        return super().gate_inds(
            G,
            inds,
            contract=contract,
            tags=tags,
            info=info,
            inplace=inplace,
            **compress_opts,
        )

    gate_ = functools.partialmethod(gate, inplace=True)


class TensorNetwork3DFlat(TensorNetwork3D):
    """Mixin class for a 3D square lattice tensor network with a single tensor
    per site, for example, both PEPS and PEPOs.
    """

    _EXTRA_PROPS = (
        "_site_tag_id",
        "_x_tag_id",
        "_y_tag_id",
        "_z_tag_id",
        "_Lx",
        "_Ly",
        "_Lz",
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


class PEPS3D(TensorNetwork3DVector, TensorNetwork3DFlat):
    r"""Projected Entangled Pair States object (3D).

    Parameters
    ----------
    arrays : sequence of sequence of sequence of array
        The core tensor data arrays.
    shape : str, optional
        Which order the dimensions of the arrays are stored in, the default
        ``'urfdlbp'`` stands for ('up', 'right', 'front', down', 'left',
        'behind', 'physical') meaning (x+, y+, z+, x-, y-, z-, physical)
        respectively. Arrays on the edge of lattice are assumed to be missing
        the corresponding dimension.
    tags : set[str], optional
        Extra global tags to add to the tensor network.
    site_ind_id : str, optional
        String specifier for naming convention of site indices.
    site_tag_id : str, optional
        String specifier for naming convention of site tags.
    x_tag_id : str, optional
        String specifier for naming convention of x-slice tags.
    y_tag_id : str, optional
        String specifier for naming convention of y-slice tags.
    z_tag_id : str, optional
        String specifier for naming convention of z-slice tags.
    """

    _EXTRA_PROPS = (
        "_site_tag_id",
        "_x_tag_id",
        "_y_tag_id",
        "_z_tag_id",
        "_Lx",
        "_Ly",
        "_Lz",
        "_site_ind_id",
    )

    def __init__(
        self,
        arrays,
        *,
        shape="urfdlbp",
        tags=None,
        site_ind_id="k{},{},{}",
        site_tag_id="I{},{},{}",
        x_tag_id="X{}",
        y_tag_id="Y{}",
        z_tag_id="Z{}",
        **tn_opts,
    ):
        if isinstance(arrays, PEPS3D):
            super().__init__(arrays)
            return

        tags = tags_to_oset(tags)
        self._site_ind_id = site_ind_id
        self._site_tag_id = site_tag_id
        self._x_tag_id = x_tag_id
        self._y_tag_id = y_tag_id
        self._z_tag_id = z_tag_id

        arrays = tuple(tuple(tuple(z for z in y) for y in x) for x in arrays)
        self._Lx = len(arrays)
        self._Ly = len(arrays[0])
        self._Lz = len(arrays[0][0])
        tensors = []

        # cache for both creating and retrieving indices
        ix = defaultdict(rand_uuid)

        for i, j, k in self.gen_site_coos():
            array = arrays[i][j][k]

            # figure out if we need to transpose the arrays from some order
            #     other than up right front down left behind physical
            array_order = shape
            if i == self.Lx - 1:
                array_order = array_order.replace("u", "")
            if j == self.Ly - 1:
                array_order = array_order.replace("r", "")
            if k == self.Lz - 1:
                array_order = array_order.replace("f", "")
            if i == 0:
                array_order = array_order.replace("d", "")
            if j == 0:
                array_order = array_order.replace("l", "")
            if k == 0:
                array_order = array_order.replace("b", "")

            # allow convention of missing bonds to be singlet dimensions
            if len(array.shape) != len(array_order):
                array = do("squeeze", array)

            transpose_order = tuple(
                array_order.find(x) for x in "urfdlbp" if x in array_order
            )
            if transpose_order != tuple(range(len(array_order))):
                array = do("transpose", array, transpose_order)

            # get the relevant indices corresponding to neighbours
            inds = []
            if "u" in array_order:
                inds.append(ix[(i, j, k), (i + 1, j, k)])
            if "r" in array_order:
                inds.append(ix[(i, j, k), (i, j + 1, k)])
            if "f" in array_order:
                inds.append(ix[(i, j, k), (i, j, k + 1)])
            if "d" in array_order:
                inds.append(ix[(i - 1, j, k), (i, j, k)])
            if "l" in array_order:
                inds.append(ix[(i, j - 1, k), (i, j, k)])
            if "b" in array_order:
                inds.append(ix[(i, j, k - 1), (i, j, k)])
            inds.append(self.site_ind(i, j, k))

            # mix site, slice and global tags
            ijk_tags = tags | oset(
                (
                    self.site_tag(i, j, k),
                    self.x_tag(i),
                    self.y_tag(j),
                    self.z_tag(k),
                )
            )

            # create the site tensor!
            tensors.append(Tensor(data=array, inds=inds, tags=ijk_tags))

        super().__init__(tensors, virtual=True, **tn_opts)

    def permute_arrays(self, shape="urfdlbp"):
        """Permute the indices of each tensor in this PEPS3D to match
        ``shape``. This doesn't change how the overall object interacts with
        other tensor networks but may be useful for extracting the underlying
        arrays consistently. This is an inplace operation.

        Parameters
        ----------
        shape : str, optional
            A permutation of ``'lrp'`` specifying the desired order of the
            left, right, and physical indices respectively.
        """
        steps = {
            "u": lambda i, j, k: (i + 1, j, k),
            "r": lambda i, j, k: (i, j + 1, k),
            "f": lambda i, j, k: (i, j, k + 1),
            "d": lambda i, j, k: (i - 1, j, k),
            "l": lambda i, j, k: (i, j - 1, k),
            "b": lambda i, j, k: (i, j, k - 1),
        }
        for i, j, k in self.gen_site_coos():
            t = self[i, j, k]
            inds = []
            for s in shape:
                if s == "p":
                    inds.append(self.site_ind(i, j, k))
                else:
                    coo2 = steps[s](i, j, k)
                    if self.valid_coo(coo2):
                        t2 = self[coo2]
                        (bix,) = t.bonds(t2)
                        inds.append(bix)
            t.transpose_(*inds)

    @classmethod
    def from_fill_fn(
        cls, fill_fn, Lx, Ly, Lz, bond_dim, phys_dim=2, **peps3d_opts
    ):
        """Create a 3D PEPS from a filling function with signature
        ``fill_fn(shape)``.

        Parameters
        ----------
        Lx : int
            The number of x-slices.
        Ly : int
            The number of y-slices.
        Lz : int
            The number of z-slices.
        bond_dim : int
            The bond dimension.
        physical : int, optional
            The physical index dimension.
        peps_opts
            Supplied to :class:`~quimb.tensor.tensor_3d.PEPS3D`.

        Returns
        -------
        psi : PEPS3D
        """
        arrays = [
            [[None for _ in range(Lz)] for _ in range(Ly)] for _ in range(Lx)
        ]

        for i, j, k in product(range(Lx), range(Ly), range(Lz)):
            shape = []
            if i != Lx - 1:  # bond up
                shape.append(bond_dim)
            if j != Ly - 1:  # bond right
                shape.append(bond_dim)
            if k != Lz - 1:  # bond front
                shape.append(bond_dim)
            if i != 0:  # bond down
                shape.append(bond_dim)
            if j != 0:  # bond left
                shape.append(bond_dim)
            if k != 0:  # bond behind
                shape.append(bond_dim)
            shape.append(phys_dim)

            arrays[i][j][k] = fill_fn(shape)

        return cls(arrays, **peps3d_opts)

    @classmethod
    def empty(
        self,
        Lx,
        Ly,
        Lz,
        bond_dim,
        phys_dim=2,
        like="numpy",
        **peps3d_opts,
    ):
        """Create an empty 3D PEPS.

        Parameters
        ----------
        Lx : int
            The number of x-slices.
        Ly : int
            The number of y-slices.
        Lz : int
            The number of z-slices.
        bond_dim : int
            The bond dimension.
        physical : int, optional
            The physical index dimension.
        peps3d_opts
            Supplied to :class:`~quimb.tensor.tensor_3d.PEPS3D`.

        Returns
        -------
        psi : PEPS3D

        See Also
        --------
        PEPS3D.from_fill_fn
        """
        return self.from_fill_fn(
            lambda shape: do("zeros", shape, like=like),
            Lx,
            Ly,
            Lz,
            bond_dim,
            phys_dim,
            **peps3d_opts,
        )

    @classmethod
    def ones(
        self,
        Lx,
        Ly,
        Lz,
        bond_dim,
        phys_dim=2,
        like="numpy",
        **peps3d_opts,
    ):
        """Create a 3D PEPS whose tensors are filled with ones.

        Parameters
        ----------
        Lx : int
            The number of x-slices.
        Ly : int
            The number of y-slices.
        Lz : int
            The number of z-slices.
        bond_dim : int
            The bond dimension.
        physical : int, optional
            The physical index dimension.
        peps3d_opts
            Supplied to :class:`~quimb.tensor.tensor_3d.PEPS3D`.

        Returns
        -------
        psi : PEPS3D

        See Also
        --------
        PEPS3D.from_fill_fn
        """
        return self.from_fill_fn(
            lambda shape: do("ones", shape, like=like),
            Lx,
            Ly,
            Lz,
            bond_dim,
            phys_dim,
            **peps3d_opts,
        )

    @classmethod
    def rand(
        cls,
        Lx,
        Ly,
        Lz,
        bond_dim,
        phys_dim=2,
        dtype="float64",
        seed=None,
        **peps3d_opts,
    ):
        """Create a random (un-normalized) 3D PEPS.

        Parameters
        ----------
        Lx : int
            The number of x-slices.
        Ly : int
            The number of y-slices.
        Lz : int
            The number of z-slices.
        bond_dim : int
            The bond dimension.
        physical : int, optional
            The physical index dimension.
        dtype : dtype, optional
            The dtype to create the arrays with, default is real double.
        seed : int, optional
            A random seed.
        peps_opts
            Supplied to :class:`~quimb.tensor.tensor_3d.PEPS3D`.

        Returns
        -------
        psi : PEPS3D

        See Also
        --------
        PEPS3D.from_fill_fn
        """
        if seed is not None:
            seed_rand(seed)

        def fill_fn(shape):
            return ops.sensibly_scale(
                ops.sensibly_scale(randn(shape, dtype=dtype))
            )

        return cls.from_fill_fn(
            fill_fn, Lx, Ly, Lz, bond_dim, phys_dim, **peps3d_opts
        )

    def partial_trace_cluster(
        self,
        keep,
        max_bond=None,
        *,
        cutoff=1e-10,
        max_distance=0,
        fillin=0,
        gauges=False,
        flatten=False,
        normalized=True,
        symmetrized="auto",
        get=None,
        **contract_boundary_opts,
    ):
        if is_lone_coo(keep):
            keep = (keep,)

        tags = [self.site_tag(i, j, k) for i, j, k in keep]

        k = self.select_local(
            tags,
            "any",
            max_distance=max_distance,
            fillin=fillin,
            virtual=False,
        )

        k.add_tag("KET")
        if gauges:
            k.gauge_simple_insert(gauges)

        kix = [self.site_ind(i, j, k) for i, j, k in keep]
        bix = [rand_uuid() for _ in kix]

        b = k.H.reindex_(dict(zip(kix, bix))).retag_({"KET": "BRA"})
        rho_tn = k | b
        rho_tn.fuse_multibonds_()

        if get == "tn":
            return rho_tn

        # contract boundaries largest dimension last
        ri, rj, rk = zip(*keep)
        imin, imax = min(ri), max(ri)
        jmin, jmax = min(rj), max(rj)
        kmin, kmax = min(rk), max(rk)
        sequence = sorted(
            [
                ((imax - imin), ("xmin", "xmax")),
                ((jmax - jmin), ("ymin", "ymax")),
                ((kmax - kmin), ("zmin", "zmax")),
            ]
        )
        sequence = [x for s in sequence for x in s[1]]

        rho_t = rho_tn.contract_boundary(
            max_bond=max_bond,
            cutoff=cutoff,
            sequence=sequence,
            layer_tags=None if flatten else ("KET", "BRA"),
            **contract_boundary_opts,
        )

        if symmetrized == "auto":
            symmetrized = not flatten

        rho = rho_t.to_dense(kix, bix)

        # maybe fix up
        if symmetrized:
            rho = (rho + dag(rho)) / 2
        if normalized:
            rho = rho / do("trace", rho)

        return rho

    def partial_trace(
        self,
        keep,
        max_bond=None,
        *,
        cutoff=1e-10,
        canonize=True,
        flatten=False,
        normalized=True,
        symmetrized="auto",
        envs=None,
        storage_factory=None,
        boundary_order=None,
        contract_cell_optimize="auto-hq",
        contract_cell_method="boundary",
        contract_cell_opts=None,
        get=None,
        **contract_boundary_opts,
    ):
        contract_cell_opts = ensure_dict(contract_cell_opts)

        norm = self.make_norm()

        contract_boundary_opts["max_bond"] = max_bond
        contract_boundary_opts["cutoff"] = cutoff
        contract_boundary_opts["canonize"] = canonize
        contract_boundary_opts["layer_tags"] = (
            None if flatten else ("KET", "BRA")
        )
        if symmetrized == "auto":
            symmetrized = not flatten

        # get minimal covering cell, allow single coordinate
        if is_lone_coo(keep):
            keep = (keep,)
        cell = sites_to_cell(keep)

        # get the environment surrounding the cell, allowing reuse via ``envs``
        (i, j, k), (x_bsz, y_bsz, z_bsz) = cell
        key = (("x", i, x_bsz), ("y", j, y_bsz), ("z", k, z_bsz))
        tn_cell = norm._maybe_compute_cell_env(
            key=key,
            envs=envs,
            storage_factory=storage_factory,
            boundary_order=boundary_order,
            **contract_boundary_opts,
        )

        # cut the bonds between target norm sites to make density matrix
        tags = [tn_cell.site_tag(*site) for site in keep]
        kix = [f"k{i},{j},{k}" for i, j, k in keep]
        bix = [f"b{i},{j},{k}" for i, j, k in keep]
        for tag, ind_k, ind_b in zip(tags, kix, bix):
            tn_cell.cut_between((tag, "KET"), (tag, "BRA"), ind_k, ind_b)

        if get == "tn":
            return tn_cell

        if contract_cell_method == "boundary":
            # perform the contract to single tensor as boundary contraction
            # -> still likely far too expensive to contract exactly
            xmin, xmax = max(0, i - 1), min(i + x_bsz, self.Lx - 1)
            ymin, ymax = max(0, j - 1), min(j + y_bsz, self.Ly - 1)
            zmin, zmax = max(0, k - 1), min(k + z_bsz, self.Lz - 1)

            sequence = []
            if i > 0:
                sequence.append("xmin")
            if i < self.Lx - 1:
                sequence.append("xmax")
            if j > 0:
                sequence.append("ymin")
            if j < self.Ly - 1:
                sequence.append("ymax")
            if k > 0:
                sequence.append("zmin")
            if k < self.Lz - 1:
                sequence.append("zmax")

            # contract longest boundary first
            scores = {"x": xmax - xmin, "y": ymax - ymin, "z": zmax - zmin}
            sequence.sort(key=lambda s: scores[s[0]], reverse=True)

            rho_tn = tn_cell.contract_boundary_(
                xmin=xmin,
                xmax=xmax,
                ymin=ymin,
                ymax=ymax,
                zmin=zmin,
                zmax=zmax,
                sequence=sequence,
                optimize=contract_cell_optimize,
                **contract_boundary_opts,
            )
        else:
            contract_cell_opts.setdefault("optimize", contract_cell_optimize)
            contract_cell_opts.setdefault("max_bond", max_bond)
            contract_cell_opts.setdefault("cutoff", cutoff)
            rho_tn = tn_cell.contract_compressed_(
                output_inds=kix + bix, **contract_cell_opts
            )

        # turn into raw array
        rho = rho_tn.to_dense(kix, bix)

        # maybe fix up
        if symmetrized:
            rho = (rho + dag(rho)) / 2
        if normalized:
            rho = rho / do("trace", rho)

        return rho

    def compute_local_expectation(
        self,
        terms,
        max_bond=None,
        *,
        cutoff=1e-10,
        canonize=True,
        flatten=False,
        normalized=True,
        symmetrized="auto",
        return_all=False,
        envs=None,
        storage_factory=None,
        progbar=False,
        **contract_boundary_opts,
    ):
        if envs is None:
            if storage_factory is not None:
                envs = storage_factory()
            else:
                envs = {}

        if progbar:
            items = Progbar(terms.items())
        else:
            items = terms.items()

        expecs = dict()
        for where, G in items:
            rho = self.partial_trace(
                where,
                max_bond=max_bond,
                cutoff=cutoff,
                canonize=canonize,
                flatten=flatten,
                symmetrized=symmetrized,
                normalized=normalized,
                envs=envs,
                storage_factory=storage_factory,
                **contract_boundary_opts,
            )
            expecs[where] = do("tensordot", G, rho, ((0, 1), (1, 0)))

        if return_all:
            return expecs

        return functools.reduce(add, expecs.values())
