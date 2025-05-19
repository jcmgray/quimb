"""Tools for defining and manipulating Hilbert spaces. Some nomenclature:

- *site*: a hashable label for a site in the Hilbert space. This can be
  anything (e.g. tuple[str | int]).
- *register*: a linear index for a site in the Hilbert space. This is an
  integer in the range [0, nsites), and requires an ordering of the sites.
- *configuration*: a mapping from sites to their occupation number or spin
  state. This is a dictionary mapping from site to int.
- *flat configuration*: a flat array of the occupation number or spin state
  of each site in the order given by this Hilbert space. This is a 1D array
  of length nsites with dtype np.uint8, for efficient manipulation with numba
  and numpy.

"""

import functools
import math

import numpy as np

from . import configcore


def parse_edges_to_unique(edges):
    """Given a list of edges, return a sorted list of unique sites and edges.

    Parameters
    ----------
    edges : Iterable[tuple[hashable, hashable]]]
        The edges to parse.

    Returns
    -------
    sites : list of hashable
        The unique sites in the edges, sorted.
    edges : list of (hashable, hashable)
        The unique edges, sorted.
    """
    sites = set()
    uniq_edges = set()
    for i, j in edges:
        if j < i:
            i, j = j, i
        sites.add(i)
        sites.add(j)
        uniq_edges.add((i, j))
    return sorted(sites), sorted(uniq_edges)


def valid_z2_sector(sector):
    """Check if the given sector is valid for Z2 symmetry."""
    return sector in ("even", "odd", 0, 1)


def valid_u1_sector(sector, nsites):
    """Check if the given sector is valid for U1 symmetry."""
    return isinstance(sector, int) and (0 <= sector <= nsites)


def valid_u1u1_sector(sector, nsites):
    """Check if the given sector is valid for U1U1 symmetry."""
    try:
        (na, ka), (nb, kb) = sector
        return (
            isinstance(na, int)
            and isinstance(ka, int)
            and isinstance(nb, int)
            and isinstance(kb, int)
            and (na + nb == nsites)
            and (na >= 0)
            and (nb >= 0)
            and (0 <= ka <= na)
            and (0 <= kb <= nb)
        )
    except (TypeError, ValueError):
        return False


def parse_symmetry_and_sector(nsites, symmetry=None, sector=None):
    if sector is None:
        return None, None

    if symmetry is None:
        # try and infer symmetry from the sector

        if sector in ("even", "odd"):
            symmetry = "Z2"

        elif isinstance(sector, int):
            symmetry = "U1"

        elif valid_u1u1_sector(sector, nsites):
            symmetry = "U1U1"

        else:
            raise ValueError(
                "No `symmetry` provided, and can't infer from `sector`."
            )

    elif symmetry not in ("Z2", "U1", "U1U1"):
        raise ValueError(
            f"Invalid `symmetry` {symmetry}. "
            "Must be one of 'Z2', 'U1', or 'U1U1'."
        )

    if symmetry == "Z2":
        if not valid_z2_sector(sector):
            raise ValueError(
                f"Invalid `sector` {sector} for `symmetry` {symmetry}."
            )
        # convert to int
        sector = {"even": 0, "odd": 1}.get(sector, sector)

    elif symmetry == "U1":
        if not valid_u1_sector(sector, nsites):
            raise ValueError(
                f"Invalid `sector` {sector} for `symmetry` {symmetry}"
                f" and nsites={nsites}."
            )
        # convert to int
        sector = int(sector)

    elif symmetry == "U1U1":
        if not valid_u1u1_sector(sector, nsites):
            raise ValueError(
                f"Invalid `sector` {sector} for `symmetry` "
                f"{symmetry} and nsites={nsites}."
            )
        # convert to tuple of ints
        sector = (
            (int(sector[0][0]), int(sector[0][1])),
            (int(sector[1][0]), int(sector[1][1])),
        )

    return symmetry, sector


class HilbertSpace:
    """Take a set of 'sites' (any sequence of sortable, hashable objects), and
    map this into a 'register' or linearly indexed range, optionally using a
    particular ordering. A symmetry and sector can also be specified, which
    will change the size of the Hilbert space and how the valid configurations
    are enumerated.

    Parameters
    ----------
    sites : int or sequence of hashable objects
        The sites to map into a linear register. If an integer, simply use
        ``range(sites)``.
    order : callable or sequence of hashable objects, optional
        If provided, use this to order the sites. If a callable, it should be a
        sorting key. If a sequence, it should be a permutation of the sites,
        and ``key=order.index`` will be used.
    symmetry : {None, "Z2", "U1", "U1U1"}, optional
        The symmetry of the Hilbert space if any. If `None` and a `sector` is
        provided, the symmetry will be inferred from the sector.
    sector : {None, str, int, tuple[tuple[int, int], tuple[int, int]]}, optional
        The sector of the Hilbert space. If None, no sector is assumed.
    """

    def __init__(
        self,
        sites,
        order=None,
        symmetry=None,
        sector=None,
    ):
        if isinstance(sites, int):
            sites = range(sites)
        if (order is not None) and (not callable(order)):
            order = order.index
        self._order = order
        self._sites = tuple(sorted(sites, key=self._order))
        # linear position to actual site label
        self._mapping_inv = dict(enumerate(self._sites))
        # actual site label to linear position
        self._mapping = {s: i for i, s in self._mapping_inv.items()}

        self._symmetry, self._sector = parse_symmetry_and_sector(
            symmetry=symmetry,
            sector=sector,
            nsites=self.nsites,
        )

        # lazily computed:
        # size of the Hilbert space
        self._size = None
        # storage for pascal table
        self._pt = None

        if self.symmetry is None:
            self._rank_to_flatconfig = functools.partial(
                configcore.rank_to_flatconfig_nosymm,
                n=self.nsites,
            )
            self._flatconfig_to_rank = configcore.flatconfig_to_rank_nosymm

        elif self.symmetry == "Z2":
            self._rank_to_flatconfig = functools.partial(
                configcore.rank_to_flatconfig_z2,
                n=self.nsites,
                p=self._sector,
            )
            self._flatconfig_to_rank = configcore.flatconfig_to_rank_z2

        elif self.symmetry == "U1":
            self._rank_to_flatconfig = functools.partial(
                configcore.rank_to_flatconfig_u1_pascal,
                n=self.nsites,
                k=self.sector,
                pt=self.get_pascal_table(),
            )
            self._flatconfig_to_rank = functools.partial(
                configcore.flatconfig_to_rank_u1_pascal,
                n=self.nsites,
                k=self.sector,
                pt=self.get_pascal_table(),
            )

        elif self.symmetry == "U1U1":
            (na, ka), (nb, kb) = self.sector
            self._rank_to_flatconfig = functools.partial(
                configcore.rank_to_flatconfig_u1u1_pascal,
                na=na,
                ka=ka,
                nb=nb,
                kb=kb,
                pt=self.get_pascal_table(),
            )
            self._flatconfig_to_rank = functools.partial(
                configcore.flatconfig_to_rank_u1u1_pascal,
                na=na,
                ka=ka,
                nb=nb,
                kb=kb,
                pt=self.get_pascal_table(),
            )

    def set_ordering(self, order=None):
        if (order is not None) and (not callable(order)):
            order = order.index
        self._order = order
        self._sites = tuple(sorted(self._sites, key=self._order))
        self._mapping_inv = dict(enumerate(self._sites))
        self._mapping = {s: i for i, s in self._mapping_inv.items()}

    @classmethod
    def from_edges(cls, edges, order=None):
        """Construct a HilbertSpace from a set of edges, which are pairs of
        sites.
        """
        sites, _ = parse_edges_to_unique(edges)
        return cls(sites, order=order)

    @property
    def sites(self):
        """The ordered tuple of all sites in the Hilbert space."""
        return self._sites

    @property
    def sector(self):
        """The sector of the Hilbert space."""
        return self._sector

    @property
    def sector_numba(self):
        """The sector of the Hilbert space, for numba consumption."""
        if self._sector is None:
            return np.array([self.nsites], dtype=np.int64)
        elif self.symmetry == "Z2":
            return np.array([self.nsites, self.sector], dtype=np.int64)
        elif self.symmetry == "U1":
            return np.array([self.nsites, self.sector], dtype=np.int64)
        elif self.symmetry == "U1U1":
            (na, ka), (nb, kb) = self.sector
            return np.array([na, ka, nb, kb], dtype=np.int64)

    @property
    def symmetry(self):
        """The symmetry of the Hilbert space."""
        return self._symmetry

    @property
    def nsites(self):
        """The total number of sites in the Hilbert space."""
        return len(self._sites)

    def get_pascal_table(self):
        """Get a sufficiently large pascal table for this Hilbert space."""
        if self._pt is None:
            if self.symmetry == "U1U1":
                nmax = max(self.sector[0][0], self.sector[1][0])
            else:
                nmax = self.nsites
            self._pt = configcore.build_pascal_table(nmax)
        return self._pt

    @property
    def size(self):
        """Get the size of this Hilbert space, taking into account the
        symmetry and sector.
        """
        if self._size is None:
            if self.symmetry is None:
                self._size = 2**self.nsites

            elif self.symmetry == "Z2":
                self._size = 2 ** (self.nsites - 1)

            elif self.symmetry == "U1":
                self._size = math.comb(self.nsites, self.sector)

            elif self.symmetry == "U1U1":
                (na, ka), (nb, kb) = self.sector
                self._size = math.comb(na, ka) * math.comb(nb, kb)

        return self._size

    def site_to_reg(self, site):
        """Convert a site to a linear register index."""
        return self._mapping[site]

    def reg_to_site(self, reg):
        """Convert a linear register index back to a site."""
        return self._mapping_inv[reg]

    def has_site(self, site):
        """Check if this HilbertSpace contains a given site."""
        return site in self._mapping

    def rank_to_flatconfig(self, rank):
        """Convert a rank (linear index) into a flat configuration.

        Parameters
        ----------
        rank : int
            The rank (linear index) to convert.

        Returns
        -------
        flatconfig : ndarray[uint8]
            A flat configuration, with the occupation number or spin state of
            each site in the order given by this ``HilbertSpace``.
        """
        return self._rank_to_flatconfig(rank)

    def flatconfig_to_rank(self, flatconfig):
        """Convert a flat configuration into a rank (linear index).

        Parameters
        ----------
        flatconfig : ndarray[uint8]
            A flat configuration, with the occupation number or spin state of
            each site in the order given by this ``HilbertSpace``.

        Returns
        -------
        rank : int
            The rank (linear index) of the flat configuration in the Hilbert
            space.
        """
        return self._flatconfig_to_rank(flatconfig)

    def config_to_flatconfig(self, config):
        """Turn a configuration into a flat configuration, assuming the order
        given by this ``HilbertSpace``.

        Parameters
        ----------
        config : dict[hashable, int]
            A dictionary mapping sites to their occupation number / spin.

        Returns
        -------
        flatconfig : ndarray[uint8]
            A flat configuration, with the occupation number or spin state of
            each site in the order given by this ``HilbertSpace``.
        """
        flatconfig = np.empty(self.nsites, dtype=np.uint8)
        for i, site in enumerate(self.sites):
            flatconfig[i] = config[site]
        return flatconfig

    def flatconfig_to_config(self, flatconfig):
        """Turn a flat configuration into a configuration, assuming the order
        given by this ``HilbertSpace``.

        Parameters
        ----------
        flatconfig : ndarray[uint8]
            A flat configuration, with the occupation number or spin state of
            each site in the order given by this ``HilbertSpace``.

        Returns
        -------
        config : dict[hashable, int]
            A dictionary mapping sites to their occupation number / spin state.
        """
        return {site: xi for xi, site in zip(flatconfig, self.sites)}

    def rank_to_config(self, rank):
        """Convert a rank (linear index) into a configuration.

        Parameters
        ----------
        rank : int
            The rank (linear index) to convert.

        Returns
        -------
        config : dict[hashable, int]
            A dictionary mapping sites to their occupation number / spin state.
        """
        flatconfig = self.rank_to_flatconfig(rank)
        return self.flatconfig_to_config(flatconfig)

    def config_to_rank(self, config):
        """Convert a configuration into a rank (linear index).

        Parameters
        ----------
        config : dict[hashable, int]
            A dictionary mapping sites to their occupation number / spin state.

        Returns
        -------
        rank : int
            The rank (linear index) of the configuration in the Hilbert space.
        """
        flatconfig = self.config_to_flatconfig(config)
        return self.flatconfig_to_rank(flatconfig)

    def rand_rank(self, seed=None):
        """Get a random rank (linear index) in the Hilbert space.

        Parameters
        ----------
        seed : None, int or numpy.random.Generator, optional
            The random seed or generator to use. If None, a new generator will
            be created with default settings.

        Returns
        -------
        rank : int64
            A random rank in the Hilbert space.
        """
        rng = np.random.default_rng(seed)
        return rng.integers(0, self.size)

    def rand_flatconfig(self, seed=None):
        """Get a random flat configuration.

        Parameters
        ----------
        seed : None, int or numpy.random.Generator, optional
            The random seed or generator to use. If None, a new generator will
            be created with default settings.

        Returns
        -------
        flatconfig : ndarray[uint8]
            A flat configuration, with the occupation number or spin state of
            each site in the order given by this ``HilbertSpace``.
        """
        r = self.rand_rank(seed=seed)
        return self._rank_to_flatconfig(r)

    def rand_config(self, seed=None):
        """Get a random configuration.

        Parameters
        ----------
        seed : None, int or numpy.random.Generator, optional
            The random seed or generator to use. If None, a new generator will
            be created with default settings.

        Returns
        -------
        config : dict[hashable, np.uint8]
            A dictionary mapping sites to their occupation number / spin state.
        """
        b = self.rand_flatconfig(seed=seed)
        return self.flatconfig_to_config(b)

    def __repr__(self):
        s = "HilbertSpace("
        s += f"nsites={self.nsites}"
        s += f", total_size={self.size:_}"
        if self.symmetry is not None:
            s += f", symmetry={self.symmetry}, sector={self.sector}"
        s += ")"
        return s
