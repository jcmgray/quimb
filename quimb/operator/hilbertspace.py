"""Tools for defining and manipulating Hilbert spaces."""

import functools
import itertools
import math
import numbers

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


def parse_symmetry_and_sector(nsites, sector=None, symmetry=None):
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


def parse_sites_dims(sites, dims):
    """Parse a site and dimension specification.

    Parameters
    ----------
    sites : int, sequence of hashable objects, or dict
        The sites to parse. If an integer, simply use ``range(sites)``. If a
        dict, the keys are the sites and the values are the dimensions.
    dims : int or sequence of int
        The dimensions to parse. If an integer, all sites have the same dim.

    Returns
    -------
    parsed_sites : list of hashable
        The parsed list of sites (as yet unsorted).
    parsed_dims : dict[hashable, int]
        A dictionary mapping each site to its dimension.
    dims_used : set[int]
        The set of unique dimensions used.
    """
    parsed_sites = []
    parsed_dims = {}
    dims_used = set()

    if isinstance(sites, dict):
        # site and dims specified as dict
        for k, v in sites.items():
            parsed_sites.append(k)
            parsed_dims[k] = v
            dims_used.add(v)
        return parsed_sites, parsed_dims, dims_used

    if isinstance(sites, int):
        # simply linearly index n sites
        sites = range(sites)

    if isinstance(dims, numbers.Integral):
        # all sites have the same dim
        dims = itertools.repeat(dims)

    for s, d in zip(sites, dims):
        parsed_sites.append(s)
        parsed_dims[s] = d
        dims_used.add(d)

    return parsed_sites, parsed_dims, dims_used


class HilbertSpace:
    """Take a set of 'sites' (any sequence of sortable, hashable objects), and
    map this into a 'register' or linearly indexed range, optionally using a
    particular ordering. A symmetry and sector can also be specified, which
    will change the size of the Hilbert space and how the valid configurations
    are enumerated.

    Some nomenclature:

    - *site*: a hashable label for a site in the Hilbert space. This can be
      any python object (e.g. tuple[str | int]).
    - *register*: a linear index for a site in the Hilbert space. This is an
      integer in the range [0, nsites), and requires an ordering of the sites.
    - *configuration*: a mapping from sites to their occupation number or spin
      state. This is a dictionary mapping from site to int.
    - *flat configuration*: a flat array of the occupation number or spin state
      of each site in the order given by this Hilbert space (i.e. a mapping of
      register to int). This is a 1D array of length nsites with dtype
      np.uint8, for efficient manipulation with numba and numpy.
    - *rank*: a linear index for a configuration in the Hilbert space, taking
      into account any symmetries and sectors. This is an integer in the range
      [0, size), where size is the size of the Hilbert space given the symmetry
      and sector.

    Parameters
    ----------
    sites : int, sequence of hashable objects, or dict
        The sites to map into a linear register. If an integer, simply use
        ``range(sites)``. If a dict, the keys are the sites and the values are
        the dimensions, in which case the `dims` argument is ignored.
    dims : int or sequence of int, optional
        The local dimensions of each site. If an integer, all sites have the
        same dimension. If a sequence, it should be the same length as `sites`.
        You can also provide the dimensions as part of the `sites` argument
        by passing a dict, in which case this argument is ignored.
    order : bool, sequence[hashable] or callable, optional
        How to order the sites. If `None` or `False` (default), the sites are
        kept in the order supplied. If `True`, the sites are sorted. If a
        sequence, it should be a permutation of the sites, and this will be
        used to order them. If a callable, it should be a sorting key function
        which will be used to order the sites.
    sector : {None, str, int, ((int, int), (int, int))}, optional
        The sector of the Hilbert space. If None, no sector is assumed.
    symmetry : {None, "Z2", "U1", "U1U1"}, optional
        The symmetry of the Hilbert space if any. If `None` and a `sector` is
        provided, the symmetry will be inferred from the sector if possible.
    """

    def __init__(
        self,
        sites,
        dims=2,
        order=None,
        sector=None,
        symmetry=None,
    ):
        self._sites, self._dims, self._dims_used = parse_sites_dims(
            sites=sites,
            dims=dims,
        )
        self.set_ordering(order)

        self._symmetry, self._sector = parse_symmetry_and_sector(
            nsites=self.nsites,
            sector=sector,
            symmetry=symmetry,
        )

        # lazily computed:
        # size of the Hilbert space
        self._size = None
        # storage for dimension of each local site
        self._sizes = None
        # storage for strides
        self._strides = None
        # storage for pascal table
        self._pt = None

        if self._dims_used == {2}:
            # all qubit hilbert space

            if self._symmetry is None:
                self._rank_to_flatconfig = functools.partial(
                    configcore.rank_to_flatconfig_nosymm,
                    n=self.nsites,
                )
                self._flatconfig_to_rank = configcore.flatconfig_to_rank_nosymm

            elif self._symmetry == "Z2":
                self._rank_to_flatconfig = functools.partial(
                    configcore.rank_to_flatconfig_z2,
                    n=self.nsites,
                    p=self._sector,
                )
                self._flatconfig_to_rank = configcore.flatconfig_to_rank_z2

            elif self._symmetry == "U1":
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

            elif self._symmetry == "U1U1":
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

        else:
            # non-qubit hilbert space

            if self._symmetry is not None:
                raise NotImplementedError(
                    "Symmetries are only implemented for "
                    "'qubit' Hilbert spaces (all dims==2)."
                )

            self._rank_to_flatconfig = functools.partial(
                configcore.rank_to_flatconfig_mixed_radix_nosymm,
                sizes=self.get_sizes(),
                strides=self.get_strides(),
            )
            self._flatconfig_to_rank = functools.partial(
                configcore.flatconfig_to_rank_mixed_radix_nosymm,
                strides=self.get_strides(),
            )

    def set_ordering(self, order):
        """Set the ordering of the sites in this Hilbert space.

        Parameters
        ----------
        order : bool, sequence[hashable] or callable, optional
            How to order the sites. If `None` or `False` (default), the sites
            are kept in the order supplied. If `True`, the sites are sorted. If
            a sequence, it should be a permutation of the sites, and this will
            be used to order them. If a callable, it should be a sorting key
            function which will be used to order the sites.
        """
        if order is None or order is False:
            # no sorting
            sites = self._sites
        elif order is True:
            # default sorting
            sites = sorted(self._sites)
        elif not callable(order):
            # assume sequence given directly
            key = {s: i for i, s in enumerate(order)}.get
            sites = sorted(self._sites, key=key)
        else:
            # sorting key
            sites = sorted(self._sites, key=order)
        self._sites = tuple(sites)

        # build forward and backward mapping to 'register' indices
        self._mapping = {}
        self._mapping_inv = {}
        for i, s in enumerate(self._sites):
            self._mapping[s] = i
            self._mapping_inv[i] = s

        # reset invalidated lazily computed values
        self._sizes = None
        self._strides = None

    @classmethod
    def from_edges(cls, edges, order=None):
        """Construct a HilbertSpace from a set of edges, which are pairs of
        sites.

        Parameters
        ----------
        edges : Iterable[tuple[hashable, hashable]]]
            The edges to parse.
        order : bool, sequence[hashable] or callable, optional
            How to order the sites. If `None` or `False` (default), the sites
            are kept in the order supplied. If `True`, the sites are sorted. If
            a sequence, it should be a permutation of the sites, and this will
            be used to order them. If a callable, it should be a sorting key
            function which will be used to order the sites.
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

    def get_sector_numba(self, sector=None, symmetry=None):
        """The sector of the Hilbert space, in 'numba form'. A non-default
        symmetry and sector can be provided.

        Parameters
        ----------
        sector : {None, str, int, ((int, int), (int, int))}, optional
            The sector of the Hilbert space. If None, the default sector is
            used.
        symmetry : {None, "Z2", "U1", "U1U1"}, optional
            The symmetry of the Hilbert space. If None, the default symmetry is
            used, or inferred from the supplied sector if possible.

        Returns
        -------
        sector : ndarray[int64]
            The sector of the Hilbert space, in 'numba form'. This is a 1D
            array of length 1, 2 or 4, depending on the symmetry.
        symmetry : str
            The symmetry of the Hilbert space. This is one of "None", "Z2",
            "U1", or "U1U1".
        """
        if sector is not None:
            symmetry, sector = parse_symmetry_and_sector(
                nsites=self.nsites,
                sector=sector,
                symmetry=symmetry,
            )
        else:
            # use defaults
            sector = self._sector
            symmetry = self._symmetry

        if sector is None:
            sector_nb = np.array([self.nsites], dtype=np.int64)
            symmetry_nb = 0
        elif symmetry == "Z2":
            sector_nb = np.array([self.nsites, sector], dtype=np.int64)
            symmetry_nb = 1
        elif symmetry == "U1":
            sector_nb = np.array([self.nsites, sector], dtype=np.int64)
            symmetry_nb = 2
        elif symmetry == "U1U1":
            (na, ka), (nb, kb) = sector
            sector_nb = np.array([na, ka, nb, kb], dtype=np.int64)
            symmetry_nb = 3

        return sector_nb, symmetry_nb

    @property
    def symmetry(self):
        """The symmetry of the Hilbert space."""
        return self._symmetry

    @property
    def nsites(self):
        """The total number of sites in the Hilbert space."""
        return len(self._sites)

    def get_sizes(self):
        """Get a numpy array of the ordered sizes of each site in the Hilbert
        space.
        """
        if self._sizes is None:
            sizelist = [self._dims[s] for s in self._sites]
            self._sizes = np.array(sizelist, dtype=np.int64)
        return self._sizes

    @property
    def sizes(self):
        """Get a numpy array of the ordered sizes of each site in the Hilbert
        space.
        """
        return self.get_sizes()

    def get_strides(self):
        """Get the strides for each site in the Hilbert space."""
        if self._strides is None:
            self._strides = configcore.calculate_strides(self.get_sizes())
        return self._strides

    @property
    def strides(self):
        """Get the strides for each site in the Hilbert space."""
        return self.get_strides()

    def get_pascal_table(self):
        """Get a sufficiently large pascal table for this Hilbert space."""
        if self._pt is None:
            if self.symmetry == "U1U1":
                nmax = max(self.sector[0][0], self.sector[1][0])
            else:
                nmax = self.nsites
            self._pt = configcore.build_pascal_table(nmax)
        return self._pt

    def get_size(self, sector=None, symmetry=None):
        """Get the size of the Hilbert space, optionally given a non-default
        symmetry and sector.

        Parameters
        ----------
        sector : {None, str, int, ((int, int), (int, int))}, optional
            The sector of the Hilbert space. If None, the default sector is
            used.
        symmetry : {None, "Z2", "U1", "U1U1"}, optional
            The symmetry of the Hilbert space. If None, the default symmetry is
            used, or inferred from the supplied sector if possible.
        """
        if sector is not None:
            symmetry, sector = parse_symmetry_and_sector(
                nsites=self.nsites,
                sector=sector,
                symmetry=symmetry,
            )
        else:
            # use defaults
            sector = self._sector
            symmetry = self._symmetry

        if self._dims_used != {2}:
            if symmetry is not None:
                raise NotImplementedError(
                    "Symmetries are only implemented for "
                    "'qubit' Hilbert spaces (all dims==2)."
                )
            # no symmetry for mixed radix hilbert space
            return np.prod(self.get_sizes())

        if symmetry is None:
            return 2**self.nsites

        if symmetry == "Z2":
            return 2 ** (self.nsites - 1)

        if symmetry == "U1":
            return math.comb(self.nsites, sector)

        if symmetry == "U1U1":
            (na, ka), (nb, kb) = sector
            return math.comb(na, ka) * math.comb(nb, kb)

        raise ValueError(f"Invalid symmetry {symmetry} for sector {sector}.")

    @property
    def size(self):
        """Get the size of this Hilbert space, taking into account the
        default symmetry and sector.
        """
        if self._size is None:
            self._size = self.get_size()
        return self._size

    def site_size(self, site):
        """Get the local dimension of a given site."""
        return self._dims[site]

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
