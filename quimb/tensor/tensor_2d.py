"""Classes and algorithms related to 2D tensor networks.
"""
import re
import random
import functools
from operator import add
from numbers import Integral
from itertools import product, cycle, starmap, combinations, count, chain
from collections import defaultdict

from autoray import do, infer_backend, get_dtype_name
import opt_einsum as oe

from ..gen.operators import swap
from ..gen.rand import randn, seed_rand
from ..utils import print_multi_line, check_opt, pairwise, ensure_dict
from . import array_ops as ops
from .tensor_core import (
    Tensor,
    bonds,
    rand_uuid,
    oset,
    tags_to_oset,
    TensorNetwork,
    tensor_contract,
    oset_union,
    bonds_size,
)
from .tensor_arbgeom import tensor_network_apply_op_vec
from .tensor_1d import maybe_factor_gate_into_tensor
from . import decomp


def manhattan_distance(coo_a, coo_b):
    return sum(abs(coo_a[i] - coo_b[i]) for i in range(2))


def nearest_neighbors(coo):
    i, j = coo
    return ((i - 1, j), (i, j - 1), (i, j + 1), (i + 1, j))


def gen_2d_bonds(Lx, Ly, steppers, coo_filter=None):
    """Convenience function for tiling pairs of bond coordinates on a 2D
    lattice given a function like ``lambda i, j: (i + 1, j + 1)``.

    Parameters
    ----------
    Lx : int
        The number of rows.
    Ly : int
        The number of columns.
    steppers : callable or sequence of callable
        Function(s) that take args ``(i, j)`` and generate another coordinate,
        thus defining a bond.
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

    if callable(steppers):
        steppers = (steppers,)

    for i, j in product(range(Lx), range(Ly)):
        if (coo_filter is None) or coo_filter(i, j):
            for stepper in steppers:
                i2, j2 = stepper(i, j)
                if (0 <= i2 < Lx) and (0 <= j2 < Ly):
                    yield (i, j), (i2, j2)


class Rotator2D:
    """Object for rotating coordinates and various contraction functions so
    that the core algorithms only have to written once, but nor does the actual
    TN have to be modified.
    """

    def __init__(self, tn, xrange, yrange, from_which):
        check_opt('from_which', from_which, {'bottom', 'top', 'left', 'right'})

        if xrange is None:
            xrange = (0, tn.Lx - 1)
        if yrange is None:
            yrange = (0, tn.Ly - 1)

        self.tn = tn
        self.xrange = xrange
        self.yrange = yrange
        self.from_which = from_which

        if self.from_which in {'bottom', 'top'}:
            # -> no rotation needed
            self.imin, self.imax = sorted(xrange)
            self.jmin, self.jmax = sorted(yrange)
            self.row_tag = tn.row_tag
            self.col_tag = tn.col_tag
            self.site_tag = tn.site_tag
        else:  # {'left', 'right'}
            # -> rotate 90deg
            self.imin, self.imax = sorted(yrange)
            self.jmin, self.jmax = sorted(xrange)
            self.col_tag = tn.row_tag
            self.row_tag = tn.col_tag
            self.site_tag = lambda i, j: tn.site_tag(j, i)

        if self.from_which in {'bottom', 'left'}:
            # -> sweeps are increasing
            self.vertical_sweep = range(self.imin, self.imax + 1, +1)
            self.istep = +1
        else:  # {'top', 'right'}
            # -> sweeps are decreasing
            self.vertical_sweep = range(self.imax, self.imin - 1, -1)
            self.istep = -1

    def get_sweep_directions(self, compress_sweep=None):
        """Get the default compress and canonize sweep directions.
        """
        if compress_sweep is None:
            compress_sweep = {
                'right': 'down',
                'left': 'up',
                'top': 'right',
                'bottom': 'left',
            }[self.from_which]
        canonize_sweep = {
            'up': 'down',
            'down': 'up',
            'left': 'right',
            'right': 'left',
        }[compress_sweep]
        return compress_sweep, canonize_sweep

    def get_sweep_fns(self, compress_sweep):
        """Get functions that compress or canonize a single rotated, 'row'.
        """
        comp_sweep, canz_sweep = self.get_sweep_directions(compress_sweep)

        if self.from_which in {'bottom', 'top'}:
            canonize_fn = functools.partial(
                self.tn.canonize_row,
                sweep=canz_sweep, yrange=self.yrange)
            compress_fn = functools.partial(
                self.tn.compress_row,
                sweep=comp_sweep, yrange=self.yrange)
        else:  # {'left', 'right'}
            canonize_fn = functools.partial(
                self.tn.canonize_column,
                sweep=canz_sweep, xrange=self.xrange)
            compress_fn = functools.partial(
                self.tn.compress_column,
                sweep=comp_sweep, xrange=self.xrange)

        return compress_fn, canonize_fn

    def get_contract_boundary_fn(self):
        """Get the function that contracts the boundary in by a single step.
        """
        if self.from_which in {'bottom', 'top'}:

            def fn(i, inext, **kwargs):
                return self.tn.contract_boundary_from_(
                    xrange=(i, inext), yrange=self.yrange,
                    from_which=self.from_which, **kwargs)

        else:  # {'left', 'right'}

            def fn(i, inext, **kwargs):
                return self.tn.contract_boundary_from_(
                    yrange=(i, inext), xrange=self.xrange,
                    from_which=self.from_which, **kwargs)

        return fn

    def get_opposite_env_fn(self):
        """Get the function and location label for contracting boundaries in
        the opposite direction to main sweep.
        """
        return {
            'bottom': (functools.partial(self.tn.compute_top_environments,
                                         yrange=self.yrange), 'top'),
            'top': (functools.partial(self.tn.compute_bottom_environments,
                                      yrange=self.yrange), 'bottom'),
            'left': (functools.partial(self.tn.compute_right_environments,
                                       xrange=self.xrange), 'right'),
            'right': (functools.partial(self.tn.compute_left_environments,
                                        xrange=self.xrange), 'left'),
        }[self.from_which]


class TensorNetwork2D(TensorNetwork):
    r"""Mixin class for tensor networks with a square lattice two-dimensional
    structure, indexed by ``[{row},{column}]`` so that::

                     'COL{j}'
                        v

        i=Lx-1 ●──●──●──●──●──●──   ──●
               |  |  |  |  |  |       |
                     ...
               |  |  |  |  |  | 'I{i},{j}' = 'I3,5' e.g.
        i=3    ●──●──●──●──●──●──
               |  |  |  |  |  |       |
        i=2    ●──●──●──●──●──●──   ──●    <== 'ROW{i}'
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
        '_site_tag_id',
        '_row_tag_id',
        '_col_tag_id',
        '_Lx',
        '_Ly',
    )

    def _compatible_2d(self, other):
        """Check whether ``self`` and ``other`` are compatible 2D tensor
        networks such that they can remain a 2D tensor network when combined.
        """
        return (
            isinstance(other, TensorNetwork2D) and
            all(getattr(self, e) == getattr(other, e)
                for e in TensorNetwork2D._EXTRA_PROPS)
        )

    def __and__(self, other):
        new = super().__and__(other)
        if self._compatible_2d(other):
            new.view_as_(TensorNetwork2D, like=self)
        return new

    def __or__(self, other):
        new = super().__or__(other)
        if self._compatible_2d(other):
            new.view_as_(TensorNetwork2D, like=self)
        return new

    @property
    def Lx(self):
        """The number of rows.
        """
        return self._Lx

    @property
    def Ly(self):
        """The number of columns.
        """
        return self._Ly

    @property
    def nsites(self):
        """The total number of sites.
        """
        return self._Lx * self._Ly

    @property
    def site_tag_id(self):
        """The string specifier for tagging each site of this 2D TN.
        """
        return self._site_tag_id

    def site_tag(self, i, j):
        """The name of the tag specifiying the tensor at site ``(i, j)``.
        """
        if not isinstance(i, str):
            i = i % self.Lx
        if not isinstance(j, str):
            j = j % self.Ly
        return self.site_tag_id.format(i, j)

    @property
    def row_tag_id(self):
        """The string specifier for tagging each row of this 2D TN.
        """
        return self._row_tag_id

    def row_tag(self, i):
        if not isinstance(i, str):
            i = i % self.Lx
        return self.row_tag_id.format(i)

    @property
    def row_tags(self):
        """A tuple of all of the ``Lx`` different row tags.
        """
        return tuple(map(self.row_tag, range(self.Lx)))

    @property
    def col_tag_id(self):
        """The string specifier for tagging each column of this 2D TN.
        """
        return self._col_tag_id

    def col_tag(self, j):
        if not isinstance(j, str):
            j = j % self.Ly
        return self.col_tag_id.format(j)

    @property
    def col_tags(self):
        """A tuple of all of the ``Ly`` different column tags.
        """
        return tuple(map(self.col_tag, range(self.Ly)))

    @property
    def site_tags(self):
        """All of the ``Lx * Ly`` site tags.
        """
        return tuple(starmap(self.site_tag, self.gen_site_coos()))

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

    def _get_tids_from_tags(self, tags, which='all'):
        """This is the function that lets coordinates such as ``(i, j)`` be
        used for many 'tag' based functions.
        """
        tags = self.maybe_convert_coo(tags)
        return super()._get_tids_from_tags(tags, which=which)

    def gen_site_coos(self):
        """Generate coordinates for all the sites in this 2D TN.
        """
        return product(range(self.Lx), range(self.Ly))

    def gen_bond_coos(self):
        """Generate pairs of coordinates for all the bonds in this 2D TN.
        """
        return gen_2d_bonds(self.Lx, self.Ly, steppers=[
            lambda i, j: (i, j + 1),
            lambda i, j: (i + 1, j)
        ])

    def gen_horizontal_bond_coos(self):
        """Generate all coordinate pairs like ``(i, j), (i, j + 1)``.
        """
        return gen_2d_bonds(self.Lx, self.Ly, steppers=[
            lambda i, j: (i, j + 1),
        ])

    def gen_horizontal_even_bond_coos(self):
        """Generate all coordinate pairs like ``(i, j), (i, j + 1)`` where
        ``j`` is even, which thus don't overlap at all.
        """
        return gen_2d_bonds(self.Lx, self.Ly, steppers=[
            lambda i, j: (i, j + 1),
        ], coo_filter=lambda i, j: j % 2 == 0)

    def gen_horizontal_odd_bond_coos(self):
        """Generate all coordinate pairs like ``(i, j), (i, j + 1)`` where
        ``j`` is odd, which thus don't overlap at all.
        """
        return gen_2d_bonds(self.Lx, self.Ly, steppers=[
            lambda i, j: (i, j + 1),
        ], coo_filter=lambda i, j: j % 2 == 1)

    def gen_vertical_bond_coos(self):
        """Generate all coordinate pairs like ``(i, j), (i + 1, j)``.
        """
        return gen_2d_bonds(self.Lx, self.Ly, steppers=[
            lambda i, j: (i + 1, j),
        ])

    def gen_vertical_even_bond_coos(self):
        """Generate all coordinate pairs like ``(i, j), (i + 1, j)`` where
        ``i`` is even, which thus don't overlap at all.
        """
        return gen_2d_bonds(self.Lx, self.Ly, steppers=[
            lambda i, j: (i + 1, j),
        ], coo_filter=lambda i, j: i % 2 == 0)

    def gen_vertical_odd_bond_coos(self):
        """Generate all coordinate pairs like ``(i, j), (i + 1, j)`` where
        ``i`` is odd, which thus don't overlap at all.
        """
        return gen_2d_bonds(self.Lx, self.Ly, steppers=[
            lambda i, j: (i + 1, j),
        ], coo_filter=lambda i, j: i % 2 == 1)

    def gen_diagonal_left_bond_coos(self):
        """Generate all coordinate pairs like ``(i, j), (i + 1, j - 1)``.
        """
        return gen_2d_bonds(self.Lx, self.Ly, steppers=[
            lambda i, j: (i + 1, j - 1),
        ])

    def gen_diagonal_left_even_bond_coos(self):
        """Generate all coordinate pairs like ``(i, j), (i + 1, j - 1)`` where
        ``j`` is even, which thus don't overlap at all.
        """
        return gen_2d_bonds(self.Lx, self.Ly, steppers=[
            lambda i, j: (i + 1, j - 1),
        ], coo_filter=lambda i, j: j % 2 == 0)

    def gen_diagonal_left_odd_bond_coos(self):
        """Generate all coordinate pairs like ``(i, j), (i + 1, j - 1)`` where
        ``j`` is odd, which thus don't overlap at all.
        """
        return gen_2d_bonds(self.Lx, self.Ly, steppers=[
            lambda i, j: (i + 1, j - 1),
        ], coo_filter=lambda i, j: j % 2 == 1)

    def gen_diagonal_right_bond_coos(self):
        """Generate all coordinate pairs like ``(i, j), (i + 1, j + 1)``.
        """
        return gen_2d_bonds(self.Lx, self.Ly, steppers=[
            lambda i, j: (i + 1, j + 1),
        ])

    def gen_diagonal_right_even_bond_coos(self):
        """Generate all coordinate pairs like ``(i, j), (i + 1, j + 1)`` where
        ``i`` is even, which thus don't overlap at all.
        """
        return gen_2d_bonds(self.Lx, self.Ly, steppers=[
            lambda i, j: (i + 1, j + 1),
        ], coo_filter=lambda i, j: i % 2 == 0)

    def gen_diagonal_right_odd_bond_coos(self):
        """Generate all coordinate pairs like ``(i, j), (i + 1, j + 1)`` where
        ``i`` is odd, which thus don't overlap at all.
        """
        return gen_2d_bonds(self.Lx, self.Ly, steppers=[
            lambda i, j: (i + 1, j + 1),
        ], coo_filter=lambda i, j: i % 2 == 1)

    def gen_diagonal_bond_coos(self):
        """Generate all next nearest neighbor diagonal coordinate pairs.
        """
        return gen_2d_bonds(self.Lx, self.Ly, steppers=[
            lambda i, j: (i + 1, j - 1),
            lambda i, j: (i + 1, j + 1),
        ])

    def valid_coo(self, ij):
        """Test whether ``ij`` is in grid for this 2D TN.
        """
        i, j = ij
        return (0 <= i < self.Lx) and (0 <= j < self.Ly)

    def __getitem__(self, key):
        """Key based tensor selection, checking for integer based shortcut.
        """
        return super().__getitem__(self.maybe_convert_coo(key))

    def show(self):
        """Print a unicode schematic of this 2D TN and its bond dimensions.
        """
        show_2d(self)

    def __repr__(self):
        """Insert number of rows and columns into standard print.
        """
        s = super().__repr__()
        extra = f', Lx={self.Lx}, Ly={self.Ly}, max_bond={self.max_bond()}'
        s = f'{s[:-2]}{extra}{s[-2:]}'
        return s

    def __str__(self):
        """Insert number of rows and columns into standard print.
        """
        s = super().__str__()
        extra = f', Lx={self.Lx}, Ly={self.Ly}, max_bond={self.max_bond()}'
        s = f'{s[:-1]}{extra}{s[-1:]}'
        return s

    def flatten(self, fuse_multibonds=True, inplace=False):
        """Contract all tensors corresponding to each site into one.
        """
        tn = self if inplace else self.copy()

        for i, j in self.gen_site_coos():
            tn ^= (i, j)

        if fuse_multibonds:
            tn.fuse_multibonds_()

        return tn.view_as_(TensorNetwork2DFlat, like=self)

    flatten_ = functools.partialmethod(flatten, inplace=True)

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
        check_opt('sweep', sweep, ('right', 'left'))

        if yrange is None:
            yrange = (0, self.Ly - 1)

        if sweep == 'right':
            for j in range(min(yrange), max(yrange), +1):
                self.canonize_between((i, j), (i, j + 1), **canonize_opts)

        else:
            for j in range(max(yrange), min(yrange), -1):
                self.canonize_between((i, j), (i, j - 1), **canonize_opts)

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
        check_opt('sweep', sweep, ('up', 'down'))

        if xrange is None:
            xrange = (0, self.Lx - 1)

        if sweep == 'up':
            for i in range(min(xrange), max(xrange), +1):
                self.canonize_between((i, j), (i + 1, j), **canonize_opts)
        else:
            for i in range(max(xrange), min(xrange), -1):
                self.canonize_between((i, j), (i - 1, j), **canonize_opts)

    def canonize_row_around(self, i, around=(0, 1)):
        # sweep to the right
        self.canonize_row(i, 'right', yrange=(0, min(around)))
        # sweep to the left
        self.canonize_row(i, 'left', yrange=(max(around), self.Ly - 1))

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
            Supplied to
            :meth:`~quimb.tensor.tensor_core.TensorNetwork.compress_between`.
        """
        check_opt('sweep', sweep, ('right', 'left'))
        compress_opts = ensure_dict(compress_opts)
        compress_opts.setdefault('absorb', 'right')
        compress_opts.setdefault('equalize_norms', equalize_norms)

        if yrange is None:
            yrange = (0, self.Ly - 1)

        if sweep == 'right':
            for j in range(min(yrange), max(yrange), +1):
                self.compress_between((i, j), (i, j + 1), max_bond=max_bond,
                                      cutoff=cutoff, **compress_opts)
        else:
            for j in range(max(yrange), min(yrange), -1):
                self.compress_between((i, j), (i, j - 1), max_bond=max_bond,
                                      cutoff=cutoff, **compress_opts)

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
            Supplied to
            :meth:`~quimb.tensor.tensor_core.TensorNetwork.compress_between`.
        """
        check_opt('sweep', sweep, ('up', 'down'))
        compress_opts = ensure_dict(compress_opts)
        compress_opts.setdefault('absorb', 'right')
        compress_opts.setdefault('equalize_norms', equalize_norms)

        if xrange is None:
            xrange = (0, self.Lx - 1)

        if sweep == 'up':
            for i in range(min(xrange), max(xrange), +1):
                self.compress_between((i, j), (i + 1, j), max_bond=max_bond,
                                      cutoff=cutoff, **compress_opts)
        else:
            for i in range(max(xrange), min(xrange), -1):
                self.compress_between((i, j), (i - 1, j), max_bond=max_bond,
                                      cutoff=cutoff, **compress_opts)

    def _contract_boundary_single(
        self,
        xrange,
        yrange,
        from_which,
        max_bond=None,
        cutoff=1e-10,
        canonize=True,
        compress_sweep=None,
        layer_tag=None,
        equalize_norms=False,
        compress_opts=None,
    ):
        # rotate coordinates and sweeps rather than actual TN
        r2d = Rotator2D(self, xrange, yrange, from_which)
        jmin, jmax, istep = r2d.jmin, r2d.jmax, r2d.istep
        site_tag = r2d.site_tag
        compress_fn, canonize_fn = r2d.get_sweep_fns(compress_sweep)

        for i in r2d.vertical_sweep[:-1]:
            #
            #     │  │  │  │  │
            #     ●──●──●──●──●  i+1  │  │  │  │  │
            #     │  │  │  │  │  -->  ●══●══●══●══●
            #     ●──●──●──●──●  i
            #
            for j in range(jmin, jmax + 1):
                tag1, tag2 = site_tag(i, j), site_tag(i + istep, j)

                if layer_tag is None:
                    # contract *any* tensors with pair of coordinates
                    self.contract_((tag1, tag2), which='any')
                else:
                    # contract a specific pair (i.e. only one 'inner' layer)
                    self.contract_between(tag1, (tag2, layer_tag))

            if canonize:
                #
                #     │  │  │  │  │
                #     ●══●══<══<══<
                #
                canonize_fn(i, equalize_norms=equalize_norms)

            #
            #     │  │  │  │  │  -->  │  │  │  │  │  -->  │  │  │  │  │
            #     >──●══●══●══●  -->  >──>──●══●══●  -->  >──>──>──●══●
            #     .  .           -->     .  .        -->        .  .
            #
            compress_fn(i, max_bond=max_bond, cutoff=cutoff,
                        equalize_norms=equalize_norms,
                        compress_opts=compress_opts)

    def _contract_boundary_multi(
        self,
        xrange,
        yrange,
        layer_tags,
        from_which,
        max_bond=None,
        cutoff=1e-10,
        canonize=True,
        compress_sweep=None,
        equalize_norms=False,
        compress_opts=None,
    ):
        # rotate coordinates and sweeps rather than actual TN
        r2d = Rotator2D(self, xrange, yrange, from_which)
        jmin, jmax, istep = r2d.jmin, r2d.jmax, r2d.istep
        site_tag = r2d.site_tag
        contract_single = r2d.get_contract_boundary_fn()

        for i in r2d.vertical_sweep[:-1]:
            # make sure the exterior sites are a single tensor
            #
            #    │ ││ ││ ││ ││ │       │ ││ ││ ││ ││ │   (for two layer tags)
            #    ●─○●─○●─○●─○●─○       ●─○●─○●─○●─○●─○
            #    │ ││ ││ ││ ││ │  ==>   ╲│ ╲│ ╲│ ╲│ ╲│
            #    ●─○●─○●─○●─○●─○         ●══●══●══●══●
            #
            for j in range(jmin, jmax + 1):
                self ^= site_tag(i, j)

            for tag in layer_tags:
                # contract interior sites from layer ``tag``
                #
                #    │ ││ ││ ││ ││ │  (first contraction if two layer tags)
                #    │ ○──○──○──○──○
                #    │╱ │╱ │╱ │╱ │╱
                #    ●══<══<══<══<
                #
                contract_single(
                    i, i + istep, layer_tag=tag,
                    max_bond=max_bond, cutoff=cutoff,
                    canonize=canonize, compress_sweep=compress_sweep,
                    equalize_norms=equalize_norms, compress_opts=compress_opts)

                # so we can still uniqely identify 'inner' tensors, drop inner
                #     site tag merged into outer tensor for all but last tensor
                for j in range(jmin, jmax + 1):
                    inner_tag = site_tag(i + istep, j)
                    if len(self.tag_map[inner_tag]) > 1:
                        self[site_tag(i, j)].drop_tags(inner_tag)

    def _contract_boundary_full_bond(
        self,
        xrange,
        yrange,
        from_which,
        max_bond,
        cutoff=0.0,
        method='eigh',
        renorm=False,
        optimize='auto-hq',
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
        from_which : {'bottom', 'left', 'top', 'right'}
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
        contract_boundary_opts.setdefault('max_bond', max_bond)
        contract_boundary_opts.setdefault('cutoff', cutoff)

        # rotate coordinates and sweeps rather than actual TN
        r2d = Rotator2D(self, xrange, yrange, from_which)
        jmin, jmax, istep = r2d.jmin, r2d.jmax, r2d.istep
        col_tag, row_tag, site_tag = r2d.col_tag, r2d.row_tag, r2d.site_tag
        opposite_env_fn, env_location = r2d.get_opposite_env_fn()

        if opposite_envs is None:
            # storage for the top down environments - compute lazily so that a
            #     dict can be supplied *with or without* them precomputed
            opposite_envs = {}

        # now contract in the other direction
        for i in r2d.vertical_sweep[:-1]:

            # contract inwards, no compression
            for j in range(jmin, jmax + 1):
                #
                #             j  j+1   ...
                #         │   │   │   │   │   │
                #        =●===●===●───●───●───●─         i + 1
                #    ...        \ │   │   │   │    ...
                #          ->     ●━━━●━━━●━━━●━         i
                #
                self.contract_([site_tag(i, j),
                                site_tag(i + istep, j)], which='any')

            # form strip of current row and approx top environment
            #     the canonicalization 'compresses' outer bonds
            #
            #     ●━━━●━━━●━━━●━━━●━━━●  i + 2
            #     │   │   │   │   │   │
            #     >--->===●===<===<---<  i + 1
            #       (jmax - jmin) // 2
            #
            row = self.select(row_tag(i))
            row.canonize_around_(col_tag((jmax - jmin) // 2))

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
            lenvs = {jmin + 1: ladder.select(col_tag(jmin))}
            for j in range(jmin + 2, jmax):
                lenvs[j] = ladder.select(col_tag(j - 1)) @ lenvs[j - 1]

            renvs = {jmax - 1: ladder.select(col_tag(jmax))}
            for j in range(jmax - 2, jmin, -1):
                renvs[j] = ladder.select(col_tag(j + 1)) @ renvs[j + 1]

            for j in range(jmin, jmax):
                if bonds_size(self[site_tag(i, j)],
                              self[site_tag(i, j + 1)]) <= max_bond:
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
                tn_be &= ladder.select_any([col_tag(j), col_tag(j + 1)])
                if j + 1 in renvs:
                    tn_be &= renvs[j + 1]

                lcut = rand_uuid()
                rcut = rand_uuid()
                tn_be.cut_between(site_tag(i, j), site_tag(i, j + 1),
                                  left_ind=lcut, right_ind=rcut)

                # form dense environment and find symmetric compressors
                E = tn_be.to_dense([rcut], [lcut], optimize=optimize)

                Cl, Cr = decomp.similarity_compress(
                    E, max_bond, method=method, renorm=renorm)

                # insert compressors back in base TN
                #
                #      j       j+1
                #     ━●━━━━━━━━●━ i+1
                #      │        │
                #     =●=Cl──Cr=●= i
                #       <--  -->
                #
                self.insert_gauge(
                    Cr, [site_tag(i, j)], [site_tag(i, j + 1)], Cl)

    def contract_boundary_from(
        self,
        xrange,
        yrange,
        from_which,
        max_bond=None,
        *,
        cutoff=1e-10,
        canonize=True,
        mode='mps',
        layer_tags=None,
        compress_sweep=None,
        compress_opts=None,
        inplace=False,
        **contract_boundary_opts,
    ):
        """Unified entrypoint for contracting any rectangular patch of tensors
        from any direction, with any boundary method.
        """
        check_opt('mode', mode, {'mps', 'full-bond'})

        tn = self if inplace else self.copy()

        # universal options
        contract_boundary_opts["xrange"] = xrange
        contract_boundary_opts["yrange"] = yrange
        contract_boundary_opts["from_which"] = from_which
        contract_boundary_opts["max_bond"] = max_bond

        if mode == 'full-bond':
            tn._contract_boundary_full_bond(**contract_boundary_opts)
            return tn

        # mps mode options
        contract_boundary_opts["cutoff"] = cutoff
        contract_boundary_opts["canonize"] = canonize
        contract_boundary_opts["compress_sweep"] = compress_sweep
        contract_boundary_opts["compress_opts"] = compress_opts

        if layer_tags is None:
            tn._contract_boundary_single(**contract_boundary_opts)
        else:
            contract_boundary_opts['layer_tags'] = layer_tags
            tn._contract_boundary_multi(**contract_boundary_opts)

        return tn

    contract_boundary_from_ = functools.partialmethod(
        contract_boundary_from, inplace=True)

    def contract_boundary_from_bottom(
        self,
        xrange,
        yrange=None,
        max_bond=None,
        *,
        cutoff=1e-10,
        canonize=True,
        mode='mps',
        layer_tags=None,
        compress_sweep='left',
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
        compress_sweep : {'left', 'right'}, optional
            Which way to perform the compression sweep, which has an effect on
            which tensors end up being canonized.
        compress_opts : None or dict, optional
            Supplied to
            :meth:`~quimb.tensor.tensor_core.TensorNetwork.compress_between`.
        inplace : bool, optional
            Whether to perform the contraction inplace or not.

        See Also
        --------
        contract_boundary_from_top, contract_boundary_from_left,
        contract_boundary_from_right
        """
        return self.contract_boundary_from(
            xrange=xrange,
            yrange=yrange,
            from_which="bottom",
            max_bond=max_bond,
            cutoff=cutoff,
            canonize=canonize,
            mode=mode,
            layer_tags=layer_tags,
            compress_sweep=compress_sweep,
            compress_opts=compress_opts,
            inplace=inplace,
            **contract_boundary_opts,
        )

    contract_boundary_from_bottom_ = functools.partialmethod(
        contract_boundary_from_bottom, inplace=True)

    def contract_boundary_from_top(
        self,
        xrange,
        yrange=None,
        max_bond=None,
        *,
        cutoff=1e-10,
        canonize=True,
        mode='mps',
        layer_tags=None,
        inplace=False,
        compress_sweep='right',
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
        compress_sweep : {'right', 'left'}, optional
            Which way to perform the compression sweep, which has an effect on
            which tensors end up being canonized.
        compress_opts : None or dict, optional
            Supplied to
            :meth:`~quimb.tensor.tensor_core.TensorNetwork.compress_between`.
        inplace : bool, optional
            Whether to perform the contraction inplace or not.

        See Also
        --------
        contract_boundary_from_bottom, contract_boundary_from_left,
        contract_boundary_from_right
        """
        return self.contract_boundary_from(
            xrange=xrange,
            yrange=yrange,
            from_which="top",
            max_bond=max_bond,
            cutoff=cutoff,
            canonize=canonize,
            mode=mode,
            layer_tags=layer_tags,
            compress_sweep=compress_sweep,
            compress_opts=compress_opts,
            inplace=inplace,
            **contract_boundary_opts,
        )

    contract_boundary_from_top_ = functools.partialmethod(
        contract_boundary_from_top, inplace=True)

    def contract_boundary_from_left(
        self,
        yrange,
        xrange=None,
        max_bond=None,
        *,
        cutoff=1e-10,
        canonize=True,
        mode='mps',
        layer_tags=None,
        compress_sweep='up',
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
        compress_sweep : {'up', 'down'}, optional
            Which way to perform the compression sweep, which has an effect on
            which tensors end up being canonized.
        compress_opts : None or dict, optional
            Supplied to
            :meth:`~quimb.tensor.tensor_core.TensorNetwork.compress_between`.
        inplace : bool, optional
            Whether to perform the contraction inplace or not.

        See Also
        --------
        contract_boundary_from_bottom, contract_boundary_from_top,
        contract_boundary_from_right
        """
        return self.contract_boundary_from(
            xrange=xrange,
            yrange=yrange,
            from_which="left",
            max_bond=max_bond,
            cutoff=cutoff,
            canonize=canonize,
            mode=mode,
            layer_tags=layer_tags,
            compress_sweep=compress_sweep,
            compress_opts=compress_opts,
            inplace=inplace,
            **contract_boundary_opts,
        )

    contract_boundary_from_left_ = functools.partialmethod(
        contract_boundary_from_left, inplace=True)

    def contract_boundary_from_right(
        self,
        yrange,
        xrange=None,
        max_bond=None,
        *,
        cutoff=1e-10,
        canonize=True,
        mode='mps',
        layer_tags=None,
        compress_sweep='down',
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
        compress_sweep : {'down', 'up'}, optional
            Which way to perform the compression sweep, which has an effect on
            which tensors end up being canonized.
        compress_opts : None or dict, optional
            Supplied to
            :meth:`~quimb.tensor.tensor_core.TensorNetwork.compress_between`.
        inplace : bool, optional
            Whether to perform the contraction inplace or not.

        See Also
        --------
        contract_boundary_from_bottom, contract_boundary_from_top,
        contract_boundary_from_left
        """
        return self.contract_boundary_from(
            xrange=xrange,
            yrange=yrange,
            from_which="right",
            max_bond=max_bond,
            cutoff=cutoff,
            canonize=canonize,
            mode=mode,
            layer_tags=layer_tags,
            compress_sweep=compress_sweep,
            compress_opts=compress_opts,
            inplace=inplace,
            **contract_boundary_opts,
        )

    contract_boundary_from_right_ = functools.partialmethod(
        contract_boundary_from_right, inplace=True)

    def contract_boundary(
        self,
        around=None,
        max_bond=None,
        *,
        cutoff=1e-10,
        canonize=True,
        mode='mps',
        layer_tags=None,
        max_separation=1,
        sequence=None,
        bottom=None,
        top=None,
        left=None,
        right=None,
        compress_opts=None,
        equalize_norms=False,
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
        stopping around a region.

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
        max_separation : int, optional
            If ``around is None``, when any two sides become this far apart
            simply contract the remaining tensor network.
        sequence : sequence of {'b', 'l', 't', 'r'}, optional
            Which directions to cycle throught when performing the inwards
            contractions: 'b', 'l', 't', 'r' corresponding to *from the*
            bottom, left, top and right respectively. If ``around`` is
            specified you will likely need all of these!
        bottom : int, optional
            The initial bottom boundary row, defaults to 0.
        top : int, optional
            The initial top boundary row, defaults to ``Lx - 1``.
        left : int, optional
            The initial left boundary column, defaults to 0.
        right : int, optional
            The initial right boundary column, defaults to ``Ly - 1``..
        inplace : bool, optional
            Whether to perform the contraction in place or not.
        compress_opts : None or dict, optional
            Supplied to
            :meth:`~quimb.tensor.tensor_core.TensorNetwork.compress_between`.
        contract_boundary_opts
            Supplied to
            :meth:`~quimb.tensor.tensor_2d.TensorNetwork2D.contract_boundary_from_bottom`,
            :meth:`~quimb.tensor.tensor_2d.TensorNetwork2D.contract_boundary_from_left`,
            :meth:`~quimb.tensor.tensor_2d.TensorNetwork2D.contract_boundary_from_top`,
            or
            :meth:`~quimb.tensor.tensor_2d.TensorNetwork2D.contract_boundary_from_right`,
            including compression and canonization options.
        """
        tn = self if inplace else self.copy()

        contract_boundary_opts['max_bond'] = max_bond
        contract_boundary_opts['mode'] = mode
        contract_boundary_opts['cutoff'] = cutoff
        contract_boundary_opts['canonize'] = canonize
        contract_boundary_opts['layer_tags'] = layer_tags
        contract_boundary_opts['equalize_norms'] = equalize_norms
        contract_boundary_opts['compress_opts'] = compress_opts

        if (mode == 'full-bond'):
            # set shared storage for opposite direction boundary contractions,
            #     this will be lazily filled by _contract_boundary_full_bond
            contract_boundary_opts.setdefault('opposite_envs', {})

        # set default starting borders
        if bottom is None:
            bottom = 0
        if top is None:
            top = tn.Lx - 1
        if left is None:
            left = 0
        if right is None:
            right = tn.Ly - 1

        stop_i_min = stop_i_max = stop_j_min = stop_j_max = None

        if around is not None:
            if sequence is None:
                sequence = 'bltr'
            stop_i_min = min(x[0] for x in around)
            stop_i_max = max(x[0] for x in around)
            stop_j_min = min(x[1] for x in around)
            stop_j_max = max(x[1] for x in around)
        elif sequence is None:
            # contract in along short dimension
            if self.Lx >= self.Ly:
                sequence = 'b'
            else:
                sequence = 'l'

        # keep track of whether we have hit the ``around`` region.
        reached_stop = {direction: False for direction in sequence}

        for direction in cycle(sequence):

            if direction == 'b':
                # for each direction check if we have reached the 'stop' region
                if (around is None) or (bottom + 1 < stop_i_min):
                    tn.contract_boundary_from_bottom_(
                        xrange=(bottom, bottom + 1), yrange=(left, right),
                        compress_sweep='left', **contract_boundary_opts)
                    bottom += 1
                else:
                    reached_stop[direction] = True

            elif direction == 'l':
                if (around is None) or (left + 1 < stop_j_min):
                    tn.contract_boundary_from_left_(
                        xrange=(bottom, top), yrange=(left, left + 1),
                        compress_sweep='up', **contract_boundary_opts)
                    left += 1
                else:
                    reached_stop[direction] = True

            elif direction == 't':
                if (around is None) or (top - 1 > stop_i_max):
                    tn.contract_boundary_from_top_(
                        xrange=(top, top - 1), compress_sweep='right',
                        yrange=(left, right), **contract_boundary_opts)
                    top -= 1
                else:
                    reached_stop[direction] = True

            elif direction == 'r':
                if (around is None) or (right - 1 > stop_j_max):
                    tn.contract_boundary_from_right_(
                        xrange=(bottom, top), yrange=(right, right - 1),
                        compress_sweep='down', **contract_boundary_opts)
                    right -= 1
                else:
                    reached_stop[direction] = True

            else:
                raise ValueError("'sequence' should be an iterable of "
                                 "'b', 'l', 't', 'r' only.")

            if around is None:
                # check if TN has become thin enough to just contract
                thin_strip = (
                    (top - bottom <= max_separation) or
                    (right - left <= max_separation)
                )
                if thin_strip:
                    if equalize_norms is True:
                        tn.equalize_norms_()

                    return tn.contract(all, optimize='auto-hq')

            # check if all directions have reached the ``around`` region
            elif all(reached_stop.values()):
                break

        if equalize_norms is True:
            tn.equalize_norms_()

        return tn

    contract_boundary_ = functools.partialmethod(
        contract_boundary, inplace=True)

    def compute_environments(
        self,
        from_which,
        xrange=None,
        yrange=None,
        max_bond=None,
        *,
        cutoff=1e-10,
        canonize=True,
        mode='mps',
        layer_tags=None,
        dense=False,
        compress_opts=None,
        envs=None,
        **contract_boundary_opts
    ):
        """Compute the ``self.Lx`` 1D boundary tensor networks describing
        the environments of rows and columns.
        """
        tn = self.copy()

        r2d = Rotator2D(tn, xrange, yrange, from_which)
        sweep, row_tag = r2d.vertical_sweep, r2d.row_tag
        contract_boundary_fn = r2d.get_contract_boundary_fn()

        if envs is None:
            envs = {}

        if mode == 'full-bond':
            # set shared storage for opposite env contractions
            contract_boundary_opts.setdefault('opposite_envs', {})

        envs[from_which, sweep[0]] = TensorNetwork([])
        first_row = row_tag(sweep[0])
        if dense:
            tn ^= first_row
        envs[from_which, sweep[1]] = tn.select(first_row)

        for i in sweep[2:]:
            iprevprev = i - 2 * sweep.step
            iprev = i - sweep.step
            if dense:
                tn ^= (row_tag(iprevprev), row_tag(iprev))
            else:
                contract_boundary_fn(
                    iprevprev, iprev,
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

    compute_bottom_environments = functools.partialmethod(
        compute_environments, from_which='bottom')
    """Compute the ``self.Lx`` 1D boundary tensor networks describing
    the lower environments of each row in this 2D tensor network. See
    :meth:`~quimb.tensor.tensor_2d.TensorNetwork2D.compute_row_environments`
    for full details.
    """

    compute_top_environments = functools.partialmethod(
        compute_environments, from_which='top')
    """Compute the ``self.Lx`` 1D boundary tensor networks describing
    the upper environments of each row in this 2D tensor network. See
    :meth:`~quimb.tensor.tensor_2d.TensorNetwork2D.compute_row_environments`
    for full details.
    """

    compute_left_environments = functools.partialmethod(
        compute_environments, from_which='left')
    """Compute the ``self.Ly`` 1D boundary tensor networks describing
    the left environments of each column in this 2D tensor network. See
    :meth:`~quimb.tensor.tensor_2d.TensorNetwork2D.compute_col_environments`
    for full details.
    """

    compute_right_environments = functools.partialmethod(
        compute_environments, from_which='right')
    """Compute the ``self.Ly`` 1D boundary tensor networks describing
    the right environments of each column in this 2D tensor network. See
    :meth:`~quimb.tensor.tensor_2d.TensorNetwork2D.compute_col_environments`
    for full details.
    """

    def compute_row_environments(
        self,
        max_bond=None,
        *,
        cutoff=1e-10,
        canonize=True,
        dense=False,
        mode='mps',
        layer_tags=None,
        compress_opts=None,
        envs=None,
        **contract_boundary_opts
    ):
        r"""Compute the ``2 * self.Lx`` 1D boundary tensor networks describing
        the lower and upper environments of each row in this 2D tensor network,
        *assumed to represent the norm*.

        The 'top' environment for row ``i`` will be a contraction of all
        rows ``i + 1, i + 2, ...`` etc::

             ●━━━●━━━●━━━●━━━●━━━●━━━●━━━●━━━●━━━●
            ╱ ╲ ╱ ╲ ╱ ╲ ╱ ╲ ╱ ╲ ╱ ╲ ╱ ╲ ╱ ╲ ╱ ╲ ╱ ╲

        The 'bottom' environment for row ``i`` will be a contraction of all
        rows ``i - 1, i - 2, ...`` etc::

            ╲ ╱ ╲ ╱ ╲ ╱ ╲ ╱ ╲ ╱ ╲ ╱ ╲ ╱ ╲ ╱ ╲ ╱ ╲ ╱
             ●━━━●━━━●━━━●━━━●━━━●━━━●━━━●━━━●━━━●

        Such that
        ``envs['top', i] & self.select(self.row_tag(i)) & envs['bottom', i]``
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
            Supplied to
            :meth:`~quimb.tensor.tensor_core.TensorNetwork.compress_between`.
        envs : dict, optional
            Supply an existing dictionary to store the environments in.
        contract_boundary_opts
            Supplied to
            :meth:`~quimb.tensor.tensor_2d.TensorNetwork2D.contract_boundary_from_bottom`
            and
            :meth:`~quimb.tensor.tensor_2d.TensorNetwork2D.contract_boundary_from_top`
            .

        Returns
        -------
        row_envs : dict[(str, int), TensorNetwork]
            The two environment tensor networks of row ``i`` will be stored in
            ``row_envs['bottom', i]`` and ``row_envs['top', i]``.
        """
        contract_boundary_opts['max_bond'] = max_bond
        contract_boundary_opts['cutoff'] = cutoff
        contract_boundary_opts['canonize'] = canonize
        contract_boundary_opts['mode'] = mode
        contract_boundary_opts['dense'] = dense
        contract_boundary_opts['layer_tags'] = layer_tags
        contract_boundary_opts['compress_opts'] = compress_opts

        if envs is None:
            envs = {}

        self.compute_top_environments(envs=envs, **contract_boundary_opts)
        self.compute_bottom_environments(envs=envs, **contract_boundary_opts)

        return envs

    def compute_col_environments(
        self,
        max_bond=None,
        *,
        cutoff=1e-10,
        canonize=True,
        dense=False,
        mode='mps',
        layer_tags=None,
        compress_opts=None,
        envs=None,
        **contract_boundary_opts
    ):
        r"""Compute the ``2 * self.Ly`` 1D boundary tensor networks describing
        the left and right environments of each column in this 2D tensor
        network, assumed to represent the norm.

        The 'left' environment for column ``j`` will be a contraction of all
        columns ``j - 1, j - 2, ...`` etc::

            ●<
            ┃
            ●<
            ┃
            ●<
            ┃
            ●<


        The 'right' environment for row ``j`` will be a contraction of all
        rows ``j + 1, j + 2, ...`` etc::

            >●
             ┃
            >●
             ┃
            >●
             ┃
            >●

        Such that
        ``envs['left', j] & self.select(self.col_tag(j)) & envs['right', j]``
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
            Supplied to
            :meth:`~quimb.tensor.tensor_core.TensorNetwork.compress_between`.
        contract_boundary_opts
            Supplied to
            :meth:`~quimb.tensor.tensor_2d.TensorNetwork2D.contract_boundary_from_left`
            and
            :meth:`~quimb.tensor.tensor_2d.TensorNetwork2D.contract_boundary_from_right`
            .

        Returns
        -------
        col_envs : dict[(str, int), TensorNetwork]
            The two environment tensor networks of column ``j`` will be stored
            in ``row_envs['left', j]`` and ``row_envs['right', j]``.
        """
        contract_boundary_opts['max_bond'] = max_bond
        contract_boundary_opts['cutoff'] = cutoff
        contract_boundary_opts['canonize'] = canonize
        contract_boundary_opts['mode'] = mode
        contract_boundary_opts['dense'] = dense
        contract_boundary_opts['layer_tags'] = layer_tags
        contract_boundary_opts['compress_opts'] = compress_opts

        if envs is None:
            envs = {}

        self.compute_left_environments(envs=envs, **contract_boundary_opts)
        self.compute_right_environments(envs=envs, **contract_boundary_opts)

        return envs

    def _compute_plaquette_environments_row_first(
        self,
        x_bsz,
        y_bsz,
        max_bond=None,
        cutoff=1e-10,
        canonize=True,
        layer_tags=None,
        second_dense=None,
        row_envs=None,
        **compute_environment_opts
    ):
        if second_dense is None:
            second_dense = x_bsz < 2

        # first we contract from either side to produce column environments
        if row_envs is None:
            row_envs = self.compute_row_environments(
                max_bond=max_bond, cutoff=cutoff, canonize=canonize,
                layer_tags=layer_tags, **compute_environment_opts)

        # next we form vertical strips and contract from both top and bottom
        #     for each column
        col_envs = dict()
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
            row_i = TensorNetwork((
                row_envs['bottom', i],
                self.select_any([self.row_tag(i + x) for x in range(x_bsz)]),
                row_envs['top', i + x_bsz - 1],
            )).view_as_(TensorNetwork2D, like=self)
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
            #     'left'    'right'       'left'    'right'
            #
            col_envs[i] = row_i.compute_col_environments(
                xrange=(max(i - 1, 0), min(i + x_bsz, self.Lx - 1)),
                max_bond=max_bond, cutoff=cutoff,
                canonize=canonize, layer_tags=layer_tags,
                dense=second_dense, **compute_environment_opts)

        # then range through all the possible plaquettes, selecting the correct
        # boundary tensors from either the column or row environments
        plaquette_envs = dict()
        for i0, j0 in product(range(self.Lx - x_bsz + 1),
                              range(self.Ly - y_bsz + 1)):

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
            left_coos = ((i0 + x, j0 - 1) for x in range(-1, x_bsz + 1))
            left_tags = tuple(
                starmap(self.site_tag, filter(self.valid_coo, left_coos)))

            right_coos = ((i0 + x, j0 + y_bsz) for x in range(-1, x_bsz + 1))
            right_tags = tuple(
                starmap(self.site_tag, filter(self.valid_coo, right_coos)))

            bottom_coos = ((i0 - 1, j0 + x) for x in range(y_bsz))
            bottom_tags = tuple(
                starmap(self.site_tag, filter(self.valid_coo, bottom_coos)))

            above_coos = ((i0 + x_bsz, j0 + x) for x in range(y_bsz))
            above_tags = tuple(
                starmap(self.site_tag, filter(self.valid_coo, above_coos)))

            env_ij = TensorNetwork((
                col_envs[i0]['left', j0].select_any(left_tags),
                col_envs[i0]['right', j0 + y_bsz - 1].select_any(right_tags),
                row_envs['bottom', i0].select_any(bottom_tags),
                row_envs['top', i0 + x_bsz - 1].select_any(above_tags),
            ))

            # finally, absorb any rank-2 corner tensors
            env_ij.rank_simplify_()

            plaquette_envs[(i0, j0), (x_bsz, y_bsz)] = env_ij

        return plaquette_envs

    def _compute_plaquette_environments_col_first(
        self,
        x_bsz,
        y_bsz,
        max_bond=None,
        cutoff=1e-10,
        canonize=True,
        layer_tags=None,
        second_dense=None,
        col_envs=None,
        **compute_environment_opts
    ):
        if second_dense is None:
            second_dense = y_bsz < 2

        # first we contract from either side to produce column environments
        if col_envs is None:
            col_envs = self.compute_col_environments(
                max_bond=max_bond, cutoff=cutoff, canonize=canonize,
                layer_tags=layer_tags, **compute_environment_opts)

        # next we form vertical strips and contract from both top and bottom
        #     for each column
        row_envs = dict()
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
            col_j = TensorNetwork((
                col_envs['left', j],
                self.select_any([self.col_tag(j + jn) for jn in range(y_bsz)]),
                col_envs['right', j + y_bsz - 1],
            )).view_as_(TensorNetwork2D, like=self)
            #
            #        y_bsz
            #        <-->        second_dense=True
            #     ●──●──●──●      ╭──●──╮
            #     │  │  │  │  or  │ ╱ ╲ │    'top'
            #        .  .           . .                  ┬
            #                                            ┊ x_bsz
            #        .  .           . .                  ┴
            #     │  │  │  │  or  │ ╲ ╱ │    'bottom'
            #     ●──●──●──●      ╰──●──╯
            #
            row_envs[j] = col_j.compute_row_environments(
                yrange=(max(j - 1, 0), min(j + y_bsz, self.Ly - 1)),
                max_bond=max_bond, cutoff=cutoff, canonize=canonize,
                layer_tags=layer_tags, dense=second_dense,
                **compute_environment_opts)

        # then range through all the possible plaquettes, selecting the correct
        # boundary tensors from either the column or row environments
        plaquette_envs = dict()
        for i0, j0 in product(range(self.Lx - x_bsz + 1),
                              range(self.Ly - y_bsz + 1)):

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
            left_coos = ((i0 + x, j0 - 1) for x in range(x_bsz))
            left_tags = tuple(
                starmap(self.site_tag, filter(self.valid_coo, left_coos)))

            right_coos = ((i0 + x, j0 + y_bsz) for x in range(x_bsz))
            right_tags = tuple(
                starmap(self.site_tag, filter(self.valid_coo, right_coos)))

            bottom_coos = ((i0 - 1, j0 + x) for x in range(- 1, y_bsz + 1))
            bottom_tags = tuple(
                starmap(self.site_tag, filter(self.valid_coo, bottom_coos)))

            above_coos = ((i0 + x_bsz, j0 + x) for x in range(- 1, y_bsz + 1))
            above_tags = tuple(
                starmap(self.site_tag, filter(self.valid_coo, above_coos)))

            env_ij = TensorNetwork((
                col_envs['left', j0].select_any(left_tags),
                col_envs['right', j0 + y_bsz - 1].select_any(right_tags),
                row_envs[j0]['bottom', i0].select_any(bottom_tags),
                row_envs[j0]['top', i0 + x_bsz - 1].select_any(above_tags),
            ))

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
        mode='mps',
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
        first_contract : {None, 'rows', 'columns'}, optional
            The environments can either be generated with initial sweeps in
            the row or column direction. Generally it makes sense to perform
            this approximate step in whichever is smaller (the default).
        second_dense : None or bool, optional
            Whether to perform the second set of contraction sweeps (in the
            rotated direction from whichever ``first_contract`` is) using
            a dense tensor or boundary method. By default this is only turned
            on if the ``bsz`` in the corresponding direction is 1.
        compress_opts : None or dict, optional
            Supplied to
            :meth:`~quimb.tensor.tensor_core.TensorNetwork.compress_between`.
        compute_environment_opts
            Supplied to
            :meth:`~quimb.tensor.tensor_2d.TensorNetwork2D.compute_col_environments`
            or
            :meth:`~quimb.tensor.tensor_2d.TensorNetwork2D.compute_row_environments`
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
                first_contract = 'columns'
            elif y_bsz > x_bsz:
                first_contract = 'rows'
            elif self.Lx >= self.Ly:
                first_contract = 'rows'
            else:
                first_contract = 'columns'

        compute_env_fn = {
            'rows': self._compute_plaquette_environments_row_first,
            'columns': self._compute_plaquette_environments_col_first,
        }[first_contract]

        return compute_env_fn(
            x_bsz=x_bsz, y_bsz=y_bsz, max_bond=max_bond, cutoff=cutoff,
            canonize=canonize, mode=mode, layer_tags=layer_tags,
            compress_opts=compress_opts, second_dense=second_dense,
            **compute_environment_opts)


def is_lone_coo(where):
    """Check if ``where`` has been specified as a single coordinate pair.
    """
    return (len(where) == 2) and (isinstance(where[0], Integral))


def gate_string_split_(TG, where, string, original_ts, bonds_along,
                       reindex_map, site_ix, info, **compress_opts):

    # by default this means singuvalues are kept in the string 'blob' tensor
    compress_opts.setdefault('absorb', 'right')

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
            left_inds=lix, get='tensors', bond_ind=bix, **compress_opts)

        # if singular values are returned (``absorb=None``) check if we should
        #     return them via ``info``, e.g. for ``SimpleUpdate`
        if maybe_svals and info is not None:
            s = next(iter(maybe_svals)).data
            coo_pair = tuple(sorted((string[i], string[i + 1])))
            info['singular_values', coo_pair] = s

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
            left_inds=lix, get='tensors', bond_ind=bix, **compress_opts)

        # if singular values are returned (``absorb=None``) check if we should
        #     return them via ``info``, e.g. for ``SimpleUpdate`
        if maybe_svals and info is not None:
            s = next(iter(maybe_svals)).data
            coo_pair = tuple(sorted((string[j - 1], string[j])))
            info['singular_values', coo_pair] = s

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


def gate_string_reduce_split_(TG, where, string, original_ts, bonds_along,
                              reindex_map, site_ix, info, **compress_opts):

    # by default this means singuvalues are kept in the string 'blob' tensor
    compress_opts.setdefault('absorb', 'right')

    # indices to reduce, first and final include physical indices for gate
    inds_to_reduce = [(bonds_along[0], site_ix[0])]
    for b1, b2 in pairwise(bonds_along):
        inds_to_reduce.append((b1, b2))
    inds_to_reduce.append((bonds_along[-1], site_ix[-1]))

    # tensors that remain on the string sites and those pulled into string
    outer_ts, inner_ts = [], []
    for coo, rix, t in zip(string, inds_to_reduce, original_ts):
        tq, tr = t.split(left_inds=None, right_inds=rix,
                         method='qr', get='tensors')
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
            left_inds=lix, get='tensors', bond_ind=bix, **compress_opts)

        # if singular values are returned (``absorb=None``) check if we should
        #     return them via ``info``, e.g. for ``SimpleUpdate`
        if maybe_svals and info is not None:
            s = next(iter(maybe_svals)).data
            coo_pair = tuple(sorted((string[i], string[i + 1])))
            info['singular_values', coo_pair] = s

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
            left_inds=lix, get='tensors', bond_ind=bix, **compress_opts)

        # if singular values are returned (``absorb=None``) check if we should
        #     return them via ``info``, e.g. for ``SimpleUpdate`
        if maybe_svals and info is not None:
            s = next(iter(maybe_svals)).data
            coo_pair = tuple(sorted((string[j - 1], string[j])))
            info['singular_values', coo_pair] = s

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


class TensorNetwork2DVector(TensorNetwork2D,
                            TensorNetwork):
    """Mixin class  for a 2D square lattice vector TN, i.e. one with a single
    physical index per site.
    """

    _EXTRA_PROPS = (
        '_site_tag_id',
        '_row_tag_id',
        '_col_tag_id',
        '_Lx',
        '_Ly',
        '_site_ind_id',
    )

    @property
    def site_ind_id(self):
        return self._site_ind_id

    def site_ind(self, i, j):
        if not isinstance(i, str):
            i = i % self.Lx
        if not isinstance(j, str):
            j = j % self.Ly
        return self.site_ind_id.format(i, j)

    def reindex_sites(self, new_id, where=None, inplace=False):
        if where is None:
            where = self.gen_site_coos()

        return self.reindex(
            {
                self.site_ind(*ij): new_id.format(*ij) for ij in where
            },
            inplace=inplace
        )

    reindex_sites_ = functools.partialmethod(reindex_sites, inplace=True)

    @site_ind_id.setter
    def site_ind_id(self, new_id):
        if self._site_ind_id != new_id:
            self.reindex_sites_(new_id)
            self._site_ind_id = new_id

    @property
    def site_inds(self):
        """All of the site inds.
        """
        return tuple(starmap(self.site_ind, self.gen_site_coos()))

    def to_dense(self, *inds_seq, **contract_opts):
        """Return the dense ket version of this 2D vector, i.e. a ``qarray``
        with shape (-1, 1).
        """
        if not inds_seq:
            # just use list of site indices
            return do('reshape', TensorNetwork.to_dense(
                self, self.site_inds, **contract_opts
            ), (-1, 1))

        return TensorNetwork.to_dense(self, *inds_seq, **contract_opts)

    def phys_dim(self, i=None, j=None):
        """Get the size of the physical indices / a specific physical index.
        """
        if (i is not None) and (j is not None):
            pix = self.site_ind(i, j)
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
        propagate_tags='sites',
        inplace=False,
        info=None,
        long_range_use_swaps=False,
        long_range_path_sequence=None,
        **compress_opts
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
        check_opt("contract", contract, (False, True, 'split', 'reduce-split'))

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
                if propagate_tags == 'register':
                    old_tags = oset(starmap(psi.site_tag, where))
                else:
                    old_tags = oset_union(psi.tensor_map[tid].tags
                                          for ind in site_ix
                                          for tid in psi.ind_map[ind])

                if propagate_tags == 'sites':
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
            site_tids = psi._get_tids_from_inds(bnds, which='any')

            # pop the sites, contract, then re-add
            pts = [psi._pop_tensor(tid) for tid in site_tids]
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
                    ij_a, ij_b, sequence=long_range_path_sequence)

            # move the sites together
            SWAP = get_swap(dp, dtype=get_dtype_name(G),
                            backend=infer_backend(G))
            for pair in swaps:
                psi.gate_(SWAP, pair, contract=contract, absorb='right')

            compress_opts['info'] = info
            compress_opts['contract'] = contract

            # perform actual gate also compressing etc on 'way back'
            psi.gate_(G, final, **compress_opts)

            compress_opts.setdefault('absorb', 'both')
            for pair in reversed(swaps):
                psi.gate_(SWAP, pair, **compress_opts)

            return psi

        if manual_lr_path:
            string = long_range_path_sequence
        else:
            string = tuple(gen_long_range_path(
                *where, sequence=long_range_path_sequence))

        # the tensors along this string, which will be updated
        original_ts = [psi[coo] for coo in string]

        # the len(string) - 1 indices connecting the string
        bonds_along = [next(iter(bonds(t1, t2)))
                       for t1, t2 in pairwise(original_ts)]

        if contract == 'split':
            #
            #       │╱  │╱          │╱  │╱
            #     ──GGGGG──  ==>  ──G┄┄┄G──
            #      ╱   ╱           ╱   ╱
            #
            gate_string_split_(
                TG, where, string, original_ts, bonds_along,
                reindex_map, site_ix, info, **compress_opts)

        elif contract == 'reduce-split':
            #
            #       │   │             │ │
            #       GGGGG             GGG               │ │
            #       │╱  │╱   ==>     ╱│ │  ╱   ==>     ╱│ │  ╱          │╱  │╱
            #     ──●───●──       ──>─●─●─<──       ──>─GGG─<──  ==>  ──G┄┄┄G──
            #      ╱   ╱           ╱     ╱           ╱     ╱           ╱   ╱
            #    <QR> <LQ>                            <SVD>
            #
            gate_string_reduce_split_(
                TG, where, string, original_ts, bonds_along,
                reindex_map, site_ix, info, **compress_opts)

        return psi

    gate_ = functools.partialmethod(gate, inplace=True)

    def compute_norm(
        self,
        layer_tags=('KET', 'BRA'),
        **contract_opts,
    ):
        """Compute the norm of this vector via boundary contraction.
        """
        norm = self.make_norm(layer_tags=layer_tags)
        return norm.contract_boundary(layer_tags=layer_tags, **contract_opts)

    def compute_local_expectation(
        self,
        terms,
        max_bond=None,
        *,
        cutoff=1e-10,
        canonize=True,
        mode='mps',
        layer_tags=('KET', 'BRA'),
        normalized=False,
        autogroup=True,
        contract_optimize='auto-hq',
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
            Supplied to
            :meth:`~quimb.tensor.tensor_2d.TensorNetwork2D.compute_plaquette_environments`
            to generate the plaquette environments, equivalent to approximately
            performing the partial trace.

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
                plaquette_envs.update(norm.compute_plaquette_environments(
                    x_bsz=x_bsz, y_bsz=y_bsz, **plaquette_env_options))

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
            sites = tuple(starmap(ket.site_tag, plaquette_to_sites(p)))

            # view the ket portion as 2d vector so we can gate it
            ket_local = ket.select_any(sites)
            ket_local.view_as_(TensorNetwork2DVector, like=self)
            bra_and_env = bra.select_any(sites) | plaquette_envs[p]

            with oe.shared_intermediates():
                # compute local estimation of norm for this plaquette
                if normalized:
                    norm_i0j0 = (
                        ket_local | bra_and_env
                    ).contract(all, optimize=contract_optimize)
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
        mode='mps',
        layer_tags=('KET', 'BRA'),
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
            Supplied to
            :meth:`~quimb.tensor.tensor_2d.TensorNetwork2D.contract_boundary`,
            by default, two layer contraction will be used.
        """
        contract_boundary_opts["max_bond"] = max_bond
        contract_boundary_opts["cutoff"] = cutoff
        contract_boundary_opts["canonize"] = canonize
        contract_boundary_opts["mode"] = mode
        contract_boundary_opts["layer_tags"] = layer_tags

        norm = self.make_norm()
        nfact = norm.contract_boundary(**contract_boundary_opts)

        n_ket = self.multiply_each(
            nfact**(-1 / (2 * self.num_tensors)), inplace=inplace)

        if balance_bonds:
            n_ket.balance_bonds_()

        if equalize_norms:
            n_ket.equalize_norms_()

        return n_ket

    normalize_ = functools.partialmethod(normalize, inplace=True)


class TensorNetwork2DOperator(TensorNetwork2D,
                              TensorNetwork):
    """Mixin class  for a 2D square lattice TN operator, i.e. one with both
    'upper' and 'lower' site (physical) indices.
    """

    _EXTRA_PROPS = (
        '_site_tag_id',
        '_row_tag_id',
        '_col_tag_id',
        '_Lx',
        '_Ly',
        '_upper_ind_id',
        '_lower_ind_id',
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
            where = self.gen_site_coos()
        return self.reindex({
            self.lower_ind(i, j): new_id.format(i, j)
            for i, j in where
        }, inplace=inplace)

    reindex_lower_sites_ = functools.partialmethod(
        reindex_lower_sites, inplace=True)

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
            where = self.gen_site_coos()
        return self.reindex({
            self.upper_ind(i, j): new_id.format(i, j)
            for i, j in where
        }, inplace=inplace)

    reindex_upper_sites_ = functools.partialmethod(
        reindex_upper_sites, inplace=True)

    def _get_lower_ind_id(self):
        return self._lower_ind_id

    def _set_lower_ind_id(self, new_id):
        if new_id == self._upper_ind_id:
            raise ValueError("Setting the same upper and lower index ids will"
                             " make the two ambiguous.")

        if self._lower_ind_id != new_id:
            self.reindex_lower_sites_(new_id)
            self._lower_ind_id = new_id

    lower_ind_id = property(
        _get_lower_ind_id, _set_lower_ind_id,
        doc="The string specifier for the lower phyiscal indices")

    def lower_ind(self, i, j):
        if not isinstance(i, str):
            i = i % self.Lx
        if not isinstance(j, str):
            j = j % self.Ly
        return self.lower_ind_id.format(i, j)

    @property
    def lower_inds(self):
        """All of the lower inds.
        """
        return tuple(starmap(self.lower_ind, self.gen_site_coos()))

    def _get_upper_ind_id(self):
        return self._upper_ind_id

    def _set_upper_ind_id(self, new_id):
        if new_id == self._lower_ind_id:
            raise ValueError("Setting the same upper and lower index ids will"
                             " make the two ambiguous.")

        if self._upper_ind_id != new_id:
            self.reindex_upper_sites_(new_id)
            self._upper_ind_id = new_id

    upper_ind_id = property(
        _get_upper_ind_id, _set_upper_ind_id,
        doc="The string specifier for the upper phyiscal indices")

    def upper_ind(self, i, j):
        if not isinstance(i, str):
            i = i % self.Lx
        if not isinstance(j, str):
            j = j % self.Ly
        return self.upper_ind_id.format(i, j)

    @property
    def upper_inds(self):
        """All of the upper inds.
        """
        return tuple(starmap(self.upper_ind, self.gen_site_coos()))

    def to_dense(self, *inds_seq, **contract_opts):
        """Return the dense matrix version of this 2D operator, i.e. a
        ``qarray`` with shape (d, d).
        """
        if not inds_seq:
            inds_seq = (self.upper_inds, self.lower_inds)

        return TensorNetwork.to_dense(self, *inds_seq, **contract_opts)

    def phys_dim(self, i=0, j=0, which='upper'):
        """Get a physical index size of this 2D operator.
        """
        if which == 'upper':
            return self[i, j].ind_size(self.upper_ind(i, j))

        if which == 'lower':
            return self[i, j].ind_size(self.lower_ind(i, j))


class TensorNetwork2DFlat(TensorNetwork2D,
                          TensorNetwork):
    """Mixin class for a 2D square lattice tensor network with a single tensor
    per site, for example, both PEPS and PEPOs.
    """

    _EXTRA_PROPS = (
        '_site_tag_id',
        '_row_tag_id',
        '_col_tag_id',
        '_Lx',
        '_Ly',
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

    def expand_bond_dimension(self, new_bond_dim, inplace=True, bra=None,
                              rand_strength=0.0):
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
        row_sweep='right',
        col_sweep='up',
        **compress_opts
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
            Supplied to
            :meth:`~quimb.tensor.tensor_core.TensorNetwork.compress_between`.
        """
        compress_opts.setdefault('absorb', 'both')
        for i in range(self.Lx):
            self.compress_row(
                i, sweep=row_sweep, max_bond=max_bond, cutoff=cutoff,
                equalize_norms=equalize_norms, compress_opts=compress_opts)
        for j in range(self.Ly):
            self.compress_column(
                j, sweep=col_sweep, max_bond=max_bond, cutoff=cutoff,
                equalize_norms=equalize_norms, compress_opts=compress_opts)


class PEPS(TensorNetwork2DVector,
           TensorNetwork2DFlat,
           TensorNetwork2D,
           TensorNetwork):
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
    row_tag_id : str, optional
        String specifier for naming convention of row tags.
    col_tag_id : str, optional
        String specifier for naming convention of column tags.
    """

    _EXTRA_PROPS = (
        '_site_tag_id',
        '_row_tag_id',
        '_col_tag_id',
        '_Lx',
        '_Ly',
        '_site_ind_id',
    )

    def __init__(self, arrays, *, shape='urdlp', tags=None,
                 site_ind_id='k{},{}', site_tag_id='I{},{}',
                 row_tag_id='ROW{}', col_tag_id='COL{}', **tn_opts):

        if isinstance(arrays, PEPS):
            super().__init__(arrays)
            return

        tags = tags_to_oset(tags)
        self._site_ind_id = site_ind_id
        self._site_tag_id = site_tag_id
        self._row_tag_id = row_tag_id
        self._col_tag_id = col_tag_id

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
                array_order = array_order.replace('u', '')
            if j == self.Ly - 1:
                array_order = array_order.replace('r', '')
            if i == 0:
                array_order = array_order.replace('d', '')
            if j == 0:
                array_order = array_order.replace('l', '')

            # allow convention of missing bonds to be singlet dimensions
            if len(array.shape) != len(array_order):
                array = do('squeeze', array)

            transpose_order = tuple(
                array_order.find(x) for x in 'urdlp' if x in array_order
            )
            if transpose_order != tuple(range(len(array_order))):
                array = do('transpose', array, transpose_order)

            # get the relevant indices corresponding to neighbours
            inds = []
            if 'u' in array_order:
                inds.append(ix[(i, j), (i + 1, j)])
            if 'r' in array_order:
                inds.append(ix[(i, j), (i, j + 1)])
            if 'd' in array_order:
                inds.append(ix[(i - 1, j), (i, j)])
            if 'l' in array_order:
                inds.append(ix[(i, j - 1), (i, j)])
            inds.append(self.site_ind(i, j))

            # mix site, row, column and global tags

            ij_tags = tags | oset((self.site_tag(i, j),
                                   self.row_tag(i),
                                   self.col_tag(j)))

            # create the site tensor!
            tensors.append(Tensor(data=array, inds=inds, tags=ij_tags))

        super().__init__(tensors, virtual=True, **tn_opts)

    @classmethod
    def from_fill_fn(
        cls, fill_fn, Lx, Ly, bond_dim, phys_dim=2, **peps_opts
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
    def empty(cls, Lx, Ly, bond_dim, phys_dim=2, like='numpy', **peps_opts):
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
            Lx, Ly, bond_dim, phys_dim, **peps_opts
        )

    @classmethod
    def ones(cls, Lx, Ly, bond_dim, phys_dim=2, like='numpy', **peps_opts):
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
            Lx, Ly, bond_dim, phys_dim, **peps_opts
        )

    @classmethod
    def rand(cls, Lx, Ly, bond_dim, phys_dim=2,
             dtype=float, seed=None, **peps_opts):
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
            return ops.sensibly_scale(ops.sensibly_scale(
                randn(shape, dtype=dtype)))

        return cls.from_fill_fn(
            fill_fn, Lx, Ly, bond_dim, phys_dim, **peps_opts
        )

    def add_PEPS(self, other, inplace=False):
        """Add this PEPS with another.
        """
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
        """PEPS addition.
        """
        return self.add_PEPS(other, inplace=False)

    def __iadd__(self, other):
        """In-place PEPS addition.
        """
        return self.add_PEPS(other, inplace=True)

    def show(self):
        """Print a unicode schematic of this PEPS and its bond dimensions.
        """
        show_2d(self, show_lower=True)


class PEPO(TensorNetwork2DOperator,
           TensorNetwork2DFlat,
           TensorNetwork2D,
           TensorNetwork):
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
    row_tag_id : str, optional
        String specifier for naming convention of row tags.
    col_tag_id : str, optional
        String specifier for naming convention of column tags.
    """

    _EXTRA_PROPS = (
        '_site_tag_id',
        '_row_tag_id',
        '_col_tag_id',
        '_Lx',
        '_Ly',
        '_upper_ind_id',
        '_lower_ind_id',
    )

    def __init__(self, arrays, *, shape='urdlbk', tags=None,
                 upper_ind_id='k{},{}', lower_ind_id='b{},{}',
                 site_tag_id='I{},{}', row_tag_id='ROW{}', col_tag_id='COL{}',
                 **tn_opts):

        if isinstance(arrays, PEPO):
            super().__init__(arrays)
            return

        tags = tags_to_oset(tags)
        self._upper_ind_id = upper_ind_id
        self._lower_ind_id = lower_ind_id
        self._site_tag_id = site_tag_id
        self._row_tag_id = row_tag_id
        self._col_tag_id = col_tag_id

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
                array_order = array_order.replace('u', '')
            if j == self.Ly - 1:
                array_order = array_order.replace('r', '')
            if i == 0:
                array_order = array_order.replace('d', '')
            if j == 0:
                array_order = array_order.replace('l', '')

            # allow convention of missing bonds to be singlet dimensions
            if len(array.shape) != len(array_order):
                array = do('squeeze', array)

            transpose_order = tuple(
                array_order.find(x) for x in 'urdlbk' if x in array_order
            )
            if transpose_order != tuple(range(len(array_order))):
                array = do('transpose', array, transpose_order)

            # get the relevant indices corresponding to neighbours
            inds = []
            if 'u' in array_order:
                inds.append(ix[(i + 1, j), (i, j)])
            if 'r' in array_order:
                inds.append(ix[(i, j), (i, j + 1)])
            if 'd' in array_order:
                inds.append(ix[(i, j), (i - 1, j)])
            if 'l' in array_order:
                inds.append(ix[(i, j - 1), (i, j)])
            inds.append(self.lower_ind(i, j))
            inds.append(self.upper_ind(i, j))

            # mix site, row, column and global tags
            ij_tags = tags | oset((self.site_tag(i, j),
                                   self.row_tag(i),
                                   self.col_tag(j)))

            # create the site tensor!
            tensors.append(Tensor(data=array, inds=inds, tags=ij_tags))

        super().__init__(tensors, virtual=True, **tn_opts)

    @classmethod
    def rand(cls, Lx, Ly, bond_dim, phys_dim=2, herm=False,
             dtype=float, seed=None, **pepo_opts):
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

            X = ops.sensibly_scale(ops.sensibly_scale(
                randn(shape, dtype=dtype)))

            if herm:
                new_order = list(range(len(shape)))
                new_order[-2], new_order[-1] = new_order[-1], new_order[-2]
                X = (do('conj', X) + do('transpose', X, new_order)) / 2

            arrays[i][j] = X

        return cls(arrays, **pepo_opts)

    rand_herm = functools.partialmethod(rand, herm=True)

    def add_PEPO(self, other, inplace=False):
        """Add this PEPO with another.
        """
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
        """PEPO addition.
        """
        return self.add_PEPO(other, inplace=False)

    def __iadd__(self, other):
        """In-place PEPO addition.
        """
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
        """Print a unicode schematic of this PEPO and its bond dimensions.
        """
        show_2d(self, show_lower=True, show_upper=True)


def show_2d(tn_2d, show_lower=False, show_upper=False):
    """Base function for printing a unicode schematic of flat 2D TNs.
    """

    lb = '╱' if show_lower else ' '
    ub = '╱' if show_upper else ' '

    line0 = ' ' + (f' {ub}{{:^3}}' * (tn_2d.Ly - 1)) + f' {ub}'
    bszs = [tn_2d.bond_size((0, j), (0, j + 1)) for j in range(tn_2d.Ly - 1)]

    lines = [line0.format(*bszs)]

    for i in range(tn_2d.Lx - 1):
        lines.append(' ●' + ('━━━━●' * (tn_2d.Ly - 1)))

        # vertical bonds
        lines.append(f'{lb}┃{{:<3}}' * tn_2d.Ly)
        bszs = [tn_2d.bond_size((i, j), (i + 1, j)) for j in range(tn_2d.Ly)]
        lines[-1] = lines[-1].format(*bszs)

        # horizontal bonds bottom
        lines.append(' ┃' + (f'{ub}{{:^3}}┃' * (tn_2d.Ly - 1)) + f'{ub}')
        bszs = [tn_2d.bond_size((i + 1, j), (i + 1, j + 1))
                for j in range(tn_2d.Ly - 1)]
        lines[-1] = lines[-1].format(*bszs)

    lines.append(' ●' + ('━━━━●' * (tn_2d.Ly - 1)))
    lines.append(f'{lb}    ' * tn_2d.Ly)

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
    bszs = tuple(sorted(
        b for b in bszs
        if not any(
            (b[0] <= b2[0]) and (b[1] <= b2[1])
            for b2 in bszs - {b}
        )
    ))

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
    return tuple((i, j)
                 for i in range(i0, i0 + di)
                 for j in range(j0, j0 + dj))


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
        poss_moves = cycle(('v', 'h'))
    elif sequence == 'random':
        poss_moves = (random.choice('vh') for _ in count())
    else:
        poss_moves = cycle(sequence)

    yield ij_a

    for move in poss_moves:
        if abs(di) + abs(dj) == 1:
            yield ij_b
            return

        if (move == 'v') and (di != 0):
            # move a vertically
            istep = min(max(di, -1), +1)
            new_ij_a = (ia + istep, ja)
            yield new_ij_a
            ij_a = new_ij_a
            ia += istep
            di -= istep
        elif (move == 'h') and (dj != 0):
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
        poss_moves = cycle(('av', 'bv', 'ah', 'bh'))
    elif sequence == 'random':
        poss_moves = (random.choice(('av', 'bv', 'ah', 'bh')) for _ in count())
    else:
        poss_moves = cycle(sequence)

    for move in poss_moves:
        if (move == 'av') and (di != 0):
            # move a vertically
            istep = min(max(di, -1), +1)
            new_ij_a = (ia + istep, ja)
            yield (ij_a, new_ij_a)
            ij_a = new_ij_a
            ia += istep
            di -= istep

        elif (move == 'bv') and (di != 0):
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

        elif (move == 'ah') and (dj != 0):
            # move a horizontally
            jstep = min(max(dj, -1), +1)
            new_ij_a = (ia, ja + jstep)
            yield (ij_a, new_ij_a)
            ij_a = new_ij_a
            ja += jstep
            dj -= jstep

        elif (move == 'bh') and (dj != 0):
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
    return do('array', SWAP, like=backend)
