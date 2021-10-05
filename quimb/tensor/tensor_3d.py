import functools
from numbers import Integral
from itertools import product, starmap, cycle
from collections import defaultdict

from autoray import do

from ..utils import check_opt, ensure_dict
from ..gen.rand import randn, seed_rand
from . import array_ops as ops
from .tensor_core import (
    Tensor,
    TensorNetwork,
    oset,
    tags_to_oset,
    rand_uuid,
)


def gen_3d_bonds(Lx, Ly, Lz, steppers, coo_filter=None):
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
        coordinate, thus defining a bond.
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

    if callable(steppers):
        steppers = (steppers,)

    for i, j, k in product(range(Lx), range(Ly), range(Lz)):
        if (coo_filter is None) or coo_filter(i, j, k):
            for stepper in steppers:
                i2, j2, k2 = stepper(i, j, k)
                if (0 <= i2 < Lx) and (0 <= j2 < Ly) and (0 <= k2 < Lz):
                    yield (i, j, k), (i2, j2, k2)


class Rotator3D:
    """Object for rotating coordinates and various contraction functions so
    that the core algorithms only have to written once, but nor does the actual
    TN have to be modified.
    """

    def __init__(self, tn, xrange, yrange, zrange, from_which):
        check_opt('from_which', from_which,
                  {'xmin', 'xmax', 'ymin', 'ymax', 'zmin', 'zmax'})

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

        if self.plane == 'x':
            # -> no rotation needed
            self.imin, self.imax = sorted(xrange)
            self.jmin, self.jmax = sorted(yrange)
            self.kmin, self.kmax = sorted(zrange)
            self.x_tag = tn.x_tag
            self.y_tag = tn.y_tag
            self.z_tag = tn.z_tag
            self.site_tag = tn.site_tag

        elif self.plane == 'y':
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

        if 'min' in from_which:
            # -> sweeps are increasing
            self.sweep = range(self.imin, self.imax + 1, + 1)
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
    'xmin': {
        'yreverse': False,
        'zreverse': False,
        'coordinate_order': 'yz',
        'stepping_order': 'zy',
    },
    'ymin': {
        'zreverse': False,
        'xreverse': True,
        'coordinate_order': 'zx',
        'stepping_order': 'xz',
    },
    'zmin': {
        'xreverse': True,
        'yreverse': True,
        'coordinate_order': 'xy',
        'stepping_order': 'yx',
    },
    'xmax': {
        'yreverse': True,
        'zreverse': True,
        'coordinate_order': 'yz',
        'stepping_order': 'zy',
    },
    'ymax': {
        'zreverse': True,
        'xreverse': False,
        'coordinate_order': 'zx',
        'stepping_order': 'xz',
    },
    'zmax': {
        'xreverse': False,
        'yreverse': False,
        'coordinate_order': 'xy',
        'stepping_order': 'yx',
    },
}


_compress_plane_opts = {
    'xmin': {
        'yreverse': True,
        'zreverse': True,
        'coordinate_order': 'yz',
        'stepping_order': 'zy',
    },
    'ymin': {
        'zreverse': True,
        'xreverse': False,
        'coordinate_order': 'zx',
        'stepping_order': 'xz',
    },
    'zmin': {
        'xreverse': False,
        'yreverse': False,
        'coordinate_order': 'xy',
        'stepping_order': 'yx',
    },
    'xmax': {
        'yreverse': False,
        'zreverse': False,
        'coordinate_order': 'yz',
        'stepping_order': 'zy',
    },
    'ymax': {
        'zreverse': False,
        'xreverse': True,
        'coordinate_order': 'zx',
        'stepping_order': 'xz',
    },
    'zmax': {
        'xreverse': True,
        'yreverse': True,
        'coordinate_order': 'xy',
        'stepping_order': 'yx',
    },
}


class TensorNetwork3D(TensorNetwork):

    _EXTRA_PROPS = (
        '_site_tag_id',
        '_x_tag_id',
        '_y_tag_id',
        '_z_tag_id',
        '_Lx',
        '_Ly',
        '_Lz',
    )

    def _compatible_3d(self, other):
        """Check whether ``self`` and ``other`` are compatible 3D tensor
        networks such that they can remain a 3D tensor network when combined.
        """
        return (
            isinstance(other, TensorNetwork3D) and
            all(getattr(self, e) == getattr(other, e)
                for e in TensorNetwork3D._EXTRA_PROPS)
        )

    def __and__(self, other):
        new = super().__and__(other)
        if self._compatible_3d(other):
            new.view_as_(TensorNetwork3D, like=self)
        return new

    def __or__(self, other):
        new = super().__or__(other)
        if self._compatible_3d(other):
            new.view_as_(TensorNetwork3D, like=self)
        return new

    @property
    def Lx(self):
        """The number of x-slices.
        """
        return self._Lx

    @property
    def Ly(self):
        """The number of y-slices.
        """
        return self._Ly

    @property
    def Lz(self):
        """The number of z-slices.
        """
        return self._Lz

    @property
    def nsites(self):
        """The total number of sites.
        """
        return self._Lx * self._Ly * self._Lz

    @property
    def site_tag_id(self):
        """The string specifier for tagging each site of this 3D TN.
        """
        return self._site_tag_id

    def site_tag(self, i, j, k):
        """The name of the tag specifiying the tensor at site ``(i, j, k)``.
        """
        if not isinstance(i, str):
            i = i % self.Lx
        if not isinstance(j, str):
            j = j % self.Ly
        if not isinstance(k, str):
            k = k % self.Lz
        return self.site_tag_id.format(i, j, k)

    @property
    def x_tag_id(self):
        """The string specifier for tagging each x-slice of this 3D TN.
        """
        return self._x_tag_id

    def x_tag(self, i):
        if not isinstance(i, str):
            i = i % self.Lx
        return self.x_tag_id.format(i)

    @property
    def x_tags(self):
        """A tuple of all of the ``Lx`` different x-slice tags.
        """
        return tuple(map(self.x_tag, range(self.Lx)))

    @property
    def y_tag_id(self):
        """The string specifier for tagging each y-slice of this 3D TN.
        """
        return self._y_tag_id

    def y_tag(self, j):
        if not isinstance(j, str):
            j = j % self.Ly
        return self.y_tag_id.format(j)

    @property
    def y_tags(self):
        """A tuple of all of the ``Ly`` different y-slice tags.
        """
        return tuple(map(self.y_tag, range(self.Ly)))

    @property
    def z_tag_id(self):
        """The string specifier for tagging each z-slice of this 3D TN.
        """
        return self._z_tag_id

    def z_tag(self, k):
        if not isinstance(k, str):
            k = k % self.Lz
        return self.z_tag_id.format(k)

    @property
    def z_tags(self):
        """A tuple of all of the ``Lz`` different z-slice tags.
        """
        return tuple(map(self.z_tag, range(self.Lz)))

    @property
    def site_tags(self):
        """All of the ``Lx * Ly`` site tags.
        """
        return tuple(starmap(self.site_tag, self.gen_site_coos()))

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

    def _get_tids_from_tags(self, tags, which='all'):
        """This is the function that lets coordinates such as ``(i, j, k)`` be
        used for many 'tag' based functions.
        """
        tags = self.maybe_convert_coo(tags)
        return super()._get_tids_from_tags(tags, which=which)

    def gen_site_coos(self):
        """Generate coordinates for all the sites in this 3D TN.
        """
        return product(range(self.Lx), range(self.Ly), range(self.Lz))

    def gen_bond_coos(self):
        """Generate pairs of coordinates for all the bonds in this 3D TN.
        """
        return gen_3d_bonds(self.Lx, self.Ly, self.Lz, steppers=[
            lambda i, j, k: (i + 1, j, k),
            lambda i, j, k: (i, j + 1, k),
            lambda i, j, k: (i, j, k + 1),
        ])

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
        return all(mn <= u <= mx for u, (mn, mx) in
                   zip(coo, (xrange, yrange, zrange)))

    def __getitem__(self, key):
        """Key based tensor selection, checking for integer based shortcut.
        """
        return super().__getitem__(self.maybe_convert_coo(key))

    def __repr__(self):
        """Insert number of slices into standard print.
        """
        s = super().__repr__()
        extra = (f', Lx={self.Lx}, Ly={self.Ly}, Lz={self.Lz}, '
                 f'max_bond={self.max_bond()}')
        s = f'{s[:-2]}{extra}{s[-2:]}'
        return s

    def __str__(self):
        """Insert number of slices into standard print.
        """
        s = super().__repr__()
        extra = (f', Lx={self.Lx}, Ly={self.Ly}, Lz={self.Lz}'
                 f'max_bond={self.max_bond()}')
        s = f'{s[:-2]}{extra}{s[-2:]}'
        return s

    def flatten(self, fuse_multibonds=True, inplace=False):
        """Contract all tensors corresponding to each site into one.
        """
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
        coordinate_order='xyz',
        xstep=None,
        ystep=None,
        zstep=None,
        stepping_order='xyz',
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
            'x': (
                range(min(xrange), max(xrange) + 1, +1) if not xreverse else
                range(max(xrange), min(xrange) - 1, -1)
            ),
            'y': (
                range(min(yrange), max(yrange) + 1, +1) if not yreverse else
                range(max(yrange), min(yrange) - 1, -1)
            ),
            'z': (
                range(min(zrange), max(zrange) + 1, +1) if not zreverse else
                range(max(zrange), min(zrange) - 1, -1)
            ),
        }

        # for convenience, allow subselecting part of stepping_order only
        if step_only is not None:
            stepping_order = stepping_order[step_only]

        # stepping_order = stepping_order[::-1]

        # at each step generate the bonds
        if xstep is None:
            xstep = -1 if xreverse else +1
        if ystep is None:
            ystep = -1 if yreverse else +1
        if zstep is None:
            zstep = -1 if zreverse else +1
        steps = {
            'x': lambda i, j, k: (i + xstep, j, k),
            'y': lambda i, j, k: (i, j + ystep, k),
            'z': lambda i, j, k: (i, j, k + zstep),
        }

        # make sure all coordinates exist - only allow them not to be specified
        # if their range is a unit slice
        for w in 'xyz':
            if w not in coordinate_order:
                if len(sweeps[w]) > 1:
                    raise ValueError(
                        f'{w} not in coordinate_order and is not size 1.')
                else:
                    # just append -> it won't change order as coord is constant
                    coordinate_order += w
        xi, yi, zi = map(coordinate_order.index, 'xyz')

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
        **gen_pair_opts
    ):
        """Canonize every pair of tensors within a subrange, optionally
        specifying a order to visit those pairs in.
        """
        canonize_opts = ensure_dict(canonize_opts)
        canonize_opts.setdefault('equalize_norms', equalize_norms)

        pairs = self.gen_pairs(
            xrange=xrange, yrange=yrange, zrange=zrange, **gen_pair_opts,
        )

        for coo_a, coo_b in pairs:
            self.canonize_between(coo_a, coo_b, **canonize_opts)

    def compress_plane(
        self,
        xrange,
        yrange,
        zrange,
        max_bond=None,
        cutoff=1e-10,
        equalize_norms=False,
        compress_opts=None,
        **gen_pair_opts
    ):
        """Compress every pair of tensors within a subrange, optionally
        specifying a order to visit those pairs in.
        """
        compress_opts = ensure_dict(compress_opts)
        compress_opts.setdefault('absorb', 'both')
        compress_opts.setdefault('equalize_norms', equalize_norms)

        pairs = list(self.gen_pairs(
            xrange=xrange, yrange=yrange, zrange=zrange, **gen_pair_opts,
        ))

        for coo_a, coo_b in pairs:
            self.compress_between(coo_a, coo_b, max_bond=max_bond,
                                  cutoff=cutoff, **compress_opts)

    def _contract_boundary_single(
        self,
        xrange,
        yrange,
        zrange,
        from_which,
        max_bond,
        cutoff=1e-10,
        canonize=True,
        canonize_interleave=True,
        layer_tag=None,
        equalize_norms=False,
        compress_opts=None,
        canonize_opts=None,
    ):
        canonize_opts = ensure_dict(canonize_opts)
        canonize_opts.setdefault('absorb', 'right')
        compress_opts = ensure_dict(compress_opts)
        compress_opts.setdefault('absorb', 'both')

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

        for i in r3d.sweep[:-1]:
            for j in range(jmin, jmax + 1):
                for k in range(kmin, kmax + 1):
                    tag1, tag2 = site_tag(i, j, k), site_tag(i + istep, j, k)

                    if layer_tag is None:
                        # contract *any* tensors with pair of coordinates
                        self.contract_((tag1, tag2), which='any')
                    else:
                        # contract specific pair (i.e. only one 'inner' layer)
                        self.contract_between(tag1, (tag2, layer_tag))

            for step_only in step_onlys:
                if canonize:
                    self.canonize_plane(
                        xrange=xrange if plane != 'x' else (i, i),
                        yrange=yrange if plane != 'y' else (i, i),
                        zrange=zrange if plane != 'z' else (i, i),
                        equalize_norms=equalize_norms,
                        canonize_opts=canonize_opts,
                        step_only=step_only,
                        **_canonize_plane_opts[from_which]
                    )
                self.compress_plane(
                    xrange=xrange if plane != 'x' else (i, i),
                    yrange=yrange if plane != 'y' else (i, i),
                    zrange=zrange if plane != 'z' else (i, i),
                    max_bond=max_bond, cutoff=cutoff,
                    equalize_norms=equalize_norms,
                    compress_opts=compress_opts,
                    step_only=step_only,
                    **_compress_plane_opts[from_which]
                )

    def contract_boundary(
        self,
        max_bond=None,
        *,
        cutoff=1e-10,
        canonize=True,
        max_separation=1,
        sequence=None,
        xmin=None,
        xmax=None,
        ymin=None,
        ymax=None,
        zmin=None,
        zmax=None,
        optimize='auto-hq',
        equalize_norms=False,
        compress_opts=None,
        inplace=False,
        **contract_boundary_opts,
    ):
        """
        """
        tn = self if inplace else self.copy()

        contract_boundary_opts['max_bond'] = max_bond
        # contract_boundary_opts['mode'] = mode
        contract_boundary_opts['cutoff'] = cutoff
        contract_boundary_opts['canonize'] = canonize
        # contract_boundary_opts['layer_tags'] = layer_tags
        contract_boundary_opts['compress_opts'] = compress_opts
        contract_boundary_opts['equalize_norms'] = equalize_norms

        # set default starting borders
        if xmin is None:
            xmin = 0
        if xmax is None:
            xmax = tn.Lx - 1
        if ymin is None:
            ymin = 0
        if ymax is None:
            ymax = tn.Ly - 1
        if zmin is None:
            zmin = 0
        if zmax is None:
            zmax = tn.Lz - 1

        if sequence is None:
            sequence = ['xmin', 'xmax', 'ymin', 'ymax', 'zmin', 'zmax']

        finished = {'x': abs(xmax - xmin) <= max_separation,
                    'y': abs(ymax - ymin) <= max_separation,
                    'z': abs(zmax - zmin) <= max_separation}

        for direction in cycle(sequence):

            if sum(finished.values()) >= 2:
                # have reached 'tube' we should contract exactly

                if equalize_norms:
                    tn.equalize_norms_()

                return tn.contract(..., optimize=optimize)

            xyz, minmax = direction[0], direction[1:]
            if finished[xyz]:
                # we have already finished this direction
                continue

            # prepare the sub-cube we will contract and compress
            if xyz == 'x':
                if minmax == 'min':
                    xrange = (xmin, xmin + 1)
                    xmin += 1
                elif minmax == 'max':
                    xrange = (xmax - 1, xmax)
                    xmax -= 1
                finished['x'] = abs(xmax - xmin) <= max_separation
            else:
                xrange = (xmin, xmax)

            if xyz == 'y':
                if minmax == 'min':
                    yrange = (ymin, ymin + 1)
                    ymin += 1
                elif minmax == 'max':
                    yrange = (ymax - 1, ymax)
                    ymax -= 1
                finished['y'] = abs(ymax - ymin) <= max_separation
            else:
                yrange = (ymin, ymax)

            if xyz == 'z':
                if minmax == 'min':
                    zrange = (zmin, zmin + 1)
                    zmin += 1
                elif minmax == 'max':
                    zrange = (zmax - 1, zmax)
                    zmax -= 1
                finished['z'] = abs(zmax - zmin) <= max_separation
            else:
                zrange = (zmin, zmax)

            tn._contract_boundary_single(
                xrange=xrange, yrange=yrange, zrange=zrange,
                from_which=direction, **contract_boundary_opts)


def is_lone_coo(where):
    """Check if ``where`` has been specified as a single coordinate triplet.
    """
    return (len(where) == 3) and (isinstance(where[0], Integral))


class TensorNetwork3DVector(TensorNetwork3D,
                            TensorNetwork):
    """Mixin class  for a 3D square lattice vector TN, i.e. one with a single
    physical index per site.
    """

    _EXTRA_PROPS = (
        '_site_tag_id',
        '_x_tag_id',
        '_y_tag_id',
        '_z_tag_id',
        '_Lx',
        '_Ly',
        '_Lz',
        '_site_ind_id',
    )

    @property
    def site_ind_id(self):
        return self._site_ind_id

    def site_ind(self, i, j, k):
        if not isinstance(i, str):
            i = i % self.Lx
        if not isinstance(j, str):
            j = j % self.Ly
        if not isinstance(k, str):
            k = k % self.Lz
        return self.site_ind_id.format(i, j, k)

    def reindex_sites(self, new_id, where=None, inplace=False):
        if where is None:
            where = self.gen_site_coos()

        return self.reindex(
            {
                self.site_ind(*coo): new_id.format(*coo) for coo in where
            },
            inplace=inplace
        )

    @site_ind_id.setter
    def site_ind_id(self, new_id):
        if self._site_ind_id != new_id:
            self.reindex_sites(new_id, inplace=True)
            self._site_ind_id = new_id

    @property
    def site_inds(self):
        """All of the site inds.
        """
        return tuple(starmap(self.site_ind, self.gen_site_coos()))

    def to_dense(self, *inds_seq, **contract_opts):
        """Return the dense ket version of this 3D vector, i.e. a ``qarray``
        with shape (-1, 1).
        """
        if not inds_seq:
            # just use list of site indices
            return do('reshape', TensorNetwork.to_dense(
                self, self.site_inds, **contract_opts
            ), (-1, 1))

        return TensorNetwork.to_dense(self, *inds_seq, **contract_opts)

    def phys_dim(self, i=None, j=None, k=None):
        """Get the size of the physical indices / a specific physical index.
        """
        if (i is not None) and (j is not None) and (k is not None):
            pix = self.site_ind(i, j, k)
        else:
            # allow for when some physical indices might have been contracted
            pix = next(iter(
                ix for ix in self.site_inds if ix in self.ind_map
            ))
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

        inds = tuple(starmap(self.site_ind, where))
        return super().gate_inds(
            G, inds, contract=contract, tags=tags, info=info, inplace=inplace,
            **compress_opts
        )

    gate_ = functools.partialmethod(gate, inplace=True)


class TensorNetwork3DFlat(TensorNetwork3D,
                          TensorNetwork):
    """Mixin class for a 3D square lattice tensor network with a single tensor
    per site, for example, both PEPS and PEPOs.
    """

    _EXTRA_PROPS = (
        '_site_tag_id',
        '_x_tag_id',
        '_y_tag_id',
        '_z_tag_id',
        '_Lx',
        '_Ly',
        '_Lz',
    )

    def bond(self, coo1, coo2):
        """Get the name of the index defining the bond between sites at
        ``coo1`` and ``coo2``.
        """
        b_ix, = self[coo1].bonds(self[coo2])
        return b_ix

    def bond_size(self, coo1, coo2):
        """Return the size of the bond between sites at ``coo1`` and ``coo2``.
        """
        b_ix = self.bond(coo1, coo2)
        return self[coo1].ind_size(b_ix)


class PEPS3D(TensorNetwork3DVector,
             TensorNetwork3DFlat,
             TensorNetwork3D,
             TensorNetwork):
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
        '_site_tag_id',
        '_x_tag_id',
        '_y_tag_id',
        '_z_tag_id',
        '_Lx',
        '_Ly',
        '_Lz',
        '_site_ind_id',
    )

    def __init__(
        self,
        arrays,
        *,
        shape='urfdlbp',
        tags=None,
        site_ind_id='k{},{},{}',
        site_tag_id='I{},{},{}',
        x_tag_id='X{}',
        y_tag_id='Y{}',
        z_tag_id='Z{}',
        **tn_opts
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
                array_order = array_order.replace('u', '')
            if j == self.Ly - 1:
                array_order = array_order.replace('r', '')
            if k == self.Lz - 1:
                array_order = array_order.replace('f', '')
            if i == 0:
                array_order = array_order.replace('d', '')
            if j == 0:
                array_order = array_order.replace('l', '')
            if k == 0:
                array_order = array_order.replace('b', '')

            # allow convention of missing bonds to be singlet dimensions
            if len(array.shape) != len(array_order):
                array = do('squeeze', array)

            transpose_order = tuple(
                array_order.find(x) for x in 'urfdlbp' if x in array_order
            )
            if transpose_order != tuple(range(len(array_order))):
                array = do('transpose', array, transpose_order)

            # get the relevant indices corresponding to neighbours
            inds = []
            if 'u' in array_order:
                inds.append(ix[(i, j, k), (i + 1, j, k)])
            if 'r' in array_order:
                inds.append(ix[(i, j, k), (i, j + 1, k)])
            if 'f' in array_order:
                inds.append(ix[(i, j, k), (i, j, k + 1)])
            if 'd' in array_order:
                inds.append(ix[(i - 1, j, k), (i, j, k)])
            if 'l' in array_order:
                inds.append(ix[(i, j - 1, k), (i, j, k)])
            if 'b' in array_order:
                inds.append(ix[(i, j, k - 1), (i, j, k)])
            inds.append(self.site_ind(i, j, k))

            # mix site, slice and global tags
            ijk_tags = tags | oset((self.site_tag(i, j, k), self.x_tag(i),
                                   self.y_tag(j), self.z_tag(k)))

            # create the site tensor!
            tensors.append(Tensor(data=array, inds=inds, tags=ijk_tags))

        super().__init__(tensors, virtual=True, **tn_opts)

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
        arrays = [[[None
                    for _ in range(Lz)]
                   for _ in range(Ly)]
                  for _ in range(Lx)]

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
        self, Lx, Ly, Lz, bond_dim, phys_dim=2, like='numpy', **peps3d_opts,
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
            Lx, Ly, Lz, bond_dim, phys_dim, **peps3d_opts
        )

    @classmethod
    def ones(
        self, Lx, Ly, Lz, bond_dim, phys_dim=2, like='numpy', **peps3d_opts,
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
            Lx, Ly, Lz, bond_dim, phys_dim, **peps3d_opts
        )

    @classmethod
    def rand(
        cls, Lx, Ly, Lz, bond_dim, phys_dim=2,
        dtype='float64', seed=None, **peps3d_opts
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
            return ops.sensibly_scale(ops.sensibly_scale(
                randn(shape, dtype=dtype)))

        return cls.from_fill_fn(
            fill_fn, Lx, Ly, Lz, bond_dim, phys_dim, **peps3d_opts
        )
