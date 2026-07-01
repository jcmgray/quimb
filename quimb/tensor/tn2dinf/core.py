"""Infinite, translation-invariant 2D tensor networks on a unit cell: the
flat single-tensor-per-site base and the PEPS wavefunction. See the
subpackage ``__init__`` for the shared vocabulary.
"""

import functools

from autoray import do

from ..tensor_core import Tensor, tensor_gauge_simple_bond
from ..tnag.core import TensorNetworkGen, TensorNetworkGenVector
from .geometry import GeometryInfinite2D, ensure_inf_2d_sites, is_inf_2d_site


class TensorNetworkInfinite2DFlat:
    """Base class for an infinite, translation-invariant 2D tensor network with
    a **single tensor per site** ('flat'), defined by a unit cell
    (``GeometryInfinite2D``). It explicitly stores the ``[-1, 1]^2`` block of
    cells as a ``fragment`` tensor network and keeps two translation-invariance
    registers: ``shared_tensors[site_type]`` and ``shared_indices[bond_type]``.
    Subclasses specialize the per-site legs and the fragment type via
    ``get_site_inds`` (+ ``get_site_shape`` / ``get_site_duals``) and
    ``_new_fragment``, e.g. ``PEPSInfinite2D`` for a wavefunction (physical
    index per site). On its own this base is a bare virtual-bond network, e.g.
    a classical network.

    Parameters
    ----------
    geometry_or_edges : GeometryInfinite2D or sequence of edges
        The unit-cell geometry, or a sequence of edges to build one from.
    site_tag_id : str, optional
        Format string for a site's tensor tag, formatted with the site.
    bond_ind_id : str, optional
        Format string for a bond index, formatted with its two sorted sites.
    site_type_tag_id : str, optional
        Format string for a ``site_type``'s shared tag.
    """

    def __init__(
        self,
        geometry_or_edges,
        site_tag_id="I{}",
        bond_ind_id="b{},{}",
        site_type_tag_id="SITE_TYPE_{}",
    ):
        if isinstance(geometry_or_edges, GeometryInfinite2D):
            self.geometry = geometry_or_edges
        else:
            self.geometry = GeometryInfinite2D(geometry_or_edges)

        self.site_tag_id = site_tag_id
        self.bond_ind_id = bond_ind_id
        self.site_type_tag_id = site_type_tag_id

        # register of shared tensors with the same site_type
        self.shared_tensors = {}
        # register of shared indices with the same bond_type
        self.shared_indices = {}

        # permissible sites to address (the [-1, 1]^2 block)
        sites_neighboring = tuple(
            ((x, y), site_type)
            for x in [-1, 0, +1]
            for y in [-1, 0, +1]
            for site_type in self.site_types
        )

        # the empty fragment scaffold; tensors are added by a constructor
        # (e.g. `.rand` / `.from_fill_fn`), not here. Bond/physical dimensions
        # are defined by those tensors, not stored on the object.
        self.fragment = self._new_fragment(sites_neighboring)
        self._sites = set()

    def _new_fragment(self, sites):
        """Build the empty ``fragment`` tensor network over ``sites``. The base
        is a bare ``TensorNetworkGen`` (virtual bonds only); subclasses choose a
        richer type (and any physical ids).
        """
        return TensorNetworkGen.new(sites=sites, site_tag_id=self.site_tag_id)

    def site_tag(self, site):
        """The tag of the tensor at ``site`` in the fragment tensor network."""
        return self.site_tag_id.format(site)

    def site_type_tag(self, site_type):
        """The shared tag applied to every tensor of ``site_type`` in the
        fragment tensor network.
        """
        return self.site_type_tag_id.format(site_type)

    def get_site_inds(self, site):
        """Index names of ``site``'s tensor, one per bond in
        ``get_site_neighbors`` order. Subclasses append extra legs (e.g. a
        physical index) via ``super().get_site_inds(site)``.
        """
        return tuple(
            self.get_bond_ind(site, nb)
            for nb in self.geometry.get_site_neighbors(site)
        )

    def get_site_shape(self, site, bond_dim):
        """Leg sizes of ``site``'s tensor, in ``get_site_inds`` order: one
        ``bond_dim`` per bond.
        """
        return tuple(bond_dim for _ in self.geometry.get_site_neighbors(site))

    def get_site_duals(self, site):
        """The symmray ``dual`` flag per leg of ``site``'s tensor, in
        ``get_site_inds`` order. A bond leg is ``dual=False`` when ``site`` is
        the first endpoint of the bond's canonical ``bond_type`` and ``True``
        when it is the second, giving opposite duals on the two ends of every
        bond. For symmetric (symmray) builders.
        """
        return tuple(
            self.get_bond_type(site, nb)[0][1] != site[1]
            for nb in self.geometry.get_site_neighbors(site)
        )

    def get_site_tags(self, site):
        """The tags of the tensor at ``site`` in the fragment tensor network:
        the site tag and the site_type tag.
        """
        return (self.site_tag(site), self.site_type_tag(site[1]))

    def _block_fill(self, fill_fn, shape_fn):
        """Populate the whole ``[-1, 1]^2`` block: the first tensor of each
        site_type gets ``fill_fn(shape_fn(site))``, later translates reuse it.
        """
        for cx in [-1, 0, +1]:
            for cy in [-1, 0, +1]:
                for site_type in self.geometry.site_types:
                    site = ((cx, cy), site_type)
                    if site_type in self.shared_tensors:
                        # will be picked up from existing tensor
                        data = None
                    else:
                        data = fill_fn(shape_fn(site))
                    self.add_fragment_site(site, data=data)

    def copy(self, deep=False):
        """Copy this infinite tensor network. The ``geometry`` is shared (it is
        static configuration), while the explicit ``fragment`` and the
        shared-tensor and shared-index registers are copied so the two networks
        can be mutated independently.

        Parameters
        ----------
        deep : bool, optional
            If ``False`` (the default), the underlying numeric data arrays are
            shared between the two networks, matching ``TensorNetwork.copy``. If
            ``True``, the data arrays are copied too.

        Returns
        -------
        TensorNetworkInfinite2DFlat
        """
        new = self.__class__.__new__(self.__class__)

        # static configuration -> shared reference / copied by value
        new.geometry = self.geometry
        new.site_tag_id = self.site_tag_id
        new.bond_ind_id = self.bond_ind_id
        new.site_type_tag_id = self.site_type_tag_id

        # copy the explicit fragment (new Tensor objects, possibly shared data)
        new.fragment = self.fragment.copy(deep=deep)
        new._sites = set(self._sites)

        # rebuild shared_tensors to reference the NEW fragment's tensors,
        # grouped by site_type (intra-group order is irrelevant)
        new.shared_tensors = {}
        for site in sorted(new._sites):
            t = new.fragment[new.site_tag_id.format(site)]
            new.shared_tensors.setdefault(site[1], []).append(t)

        # shared_indices holds only hashables -> copy the nested dicts
        new.shared_indices = {
            bond_type: dict(inds)
            for bond_type, inds in self.shared_indices.items()
        }

        return new

    @property
    def site_types(self):
        return self.geometry.site_types

    @property
    def site_type_tags(self):
        return tuple(
            self.site_type_tag_id.format(st) for st in self.site_types
        )

    def has_fragment_site(self, site):
        """Check if `site` is currently present in the fragment tensor network."""
        return site in self._sites

    def get_bond_ind(self, sitea, siteb):
        """Get the name of the bond index between two sites, invariant to
        which order they are supplied.
        """
        edge_sorted = self.geometry.get_bond_sorted(sitea, siteb)
        return self.bond_ind_id.format(*edge_sorted)

    def get_bond_type(self, sitea, siteb):
        """Get the bond type between two sites, invariant to which order they
        are supplied.
        """
        return self.geometry.get_bond_type(sitea, siteb)

    def add_fragment_site(self, site, data=None):
        """Add, if not already present, ``site`` to the explicit fragment
        tensor network. No-op if ``site`` is already present. The bond and
        physical dimensions are defined by ``data``; ``data`` is required for
        the first site of each site_type and reused (shared) by all its
        translates, so it is ignored for later sites of the same type. Sites
        are normally added by a constructor (``.rand`` / ``.from_fill_fn``).

        Parameters
        ----------
        site : tuple[tuple[int, int], hashable]
            The site to add, as ``(cell, site_type)``. Must lie within the
            ``[-1, 1]^2`` neighbor region of the unit cell.
        data : array_like, optional
            The tensor data for the first site of a new site_type (its legs are
            the bonds in ``get_site_neighbors`` order then any extra legs, see
            ``get_site_shape``). Reused for later translates.
        """
        if self.has_fragment_site(site):
            return

        (x, y), site_type = site
        if not (-1 <= x <= 1 and -1 <= y <= 1):
            # currently only allow sites to be added in the neighboring region
            raise ValueError(f"Site {site} is not in the unit cell.")

        # all leg names (bonds in get_site_neighbors order, then any extra legs)
        inds = self.get_site_inds(site)

        # register each bond index under its bond_type (bonds come first, in
        # get_site_neighbors order, so they align with the front of `inds`)
        for bond_ind, site_neighbor in zip(
            inds, self.geometry.get_site_neighbors(site)
        ):
            bond_type = self.get_bond_type(site, site_neighbor)
            self.shared_indices.setdefault(bond_type, {})[bond_ind] = None

        # generate or retrieve shared data for this site_type
        if site_type not in self.shared_tensors:
            # this is the first tensor of this type -> needs data
            if data is None:
                raise ValueError(
                    f"`data` is required for the first site of type "
                    f"{site_type}."
                )
            self.shared_tensors[site_type] = []
        else:
            data = self.shared_tensors[site_type][0].data

        # create the tensor and add it to the network
        site_type_tag = self.site_type_tag_id.format(site_type)
        tags = [self.site_tag_id.format(site), site_type_tag]
        t = Tensor(data=data, inds=inds, tags=tags)
        self.fragment.add_tensor(t, virtual=True)

        # register under site_type
        self.shared_tensors[site_type].append(t)

        self._sites.add(site)

    def _sync_site(self, site, t=None):
        """Broadcast the data from `site` to every site of that `site_type`.

        The tensor is retrieved from the current fragment tensor network or
        you can explicitly supply it if you have it.
        """
        if t is None:
            t = self.fragment[site]
        site_type = site[1]
        for t_st in self.shared_tensors[site_type]:
            if t_st is not t:
                t_st.modify(data=t.data)

    def _sync_bond(self, bond_type, bond_ind, gauges):
        """Broadcast the gauge on ``bond_ind`` to every index of bond_type."""
        g = gauges[bond_ind]
        for other in self.shared_indices[bond_type]:
            if other != bond_ind:
                gauges[other] = g

    def gauge_all_simple(
        self,
        max_iterations=5,
        tol=0.0,
        smudge=1e-12,
        power=1.0,
        gauges=None,
        fuse_multibonds=False,
        info=None,
        progbar=False,
        inplace=False,
    ):
        """Iterative gauge all the bonds in this tensor network with a 'simple
        update' like strategy. If gauges are not supplied they are initialized
        and then reabsorbed at the end, in which case this method acts as a
        kind of conditioning. More usefully, if you supply `gauges` then they
        will be updated inplace and *not* absorbed back into the tensor
        network, with the assumption that you are using/tracking them
        externally. As the tensors and bond weights are updated, changes are
        propagated to all globally shared bond and site types.

        Parameters
        ----------
        max_iterations : int, optional
            The maximum number of gauging sweeps over all ``bond_types``.
        tol : float, optional
            The convergence tolerance on the singular values. Only enables
            early stopping if greater than 0.0.
        smudge : float, optional
            A small value to add to the singular values when gauging.
        power : float, optional
            A power to raise the singular values to when gauging.
        gauges : dict, optional
            The store of bond gauges, keyed by bond index in the fragment. If
            supplied, it is updated inplace and the gauges are left on the
            bonds. If not, an internal store is used and reabsorbed into the
            tensors at the end.
        fuse_multibonds : bool, optional
            Accepted for signature compatibility, only ``False`` is supported
            (bonds are gauged one ``bond_type`` representative at a time, never
            fused).
        info : dict, optional
            Store extra information about the gauging process in this dict. The
            following keys are filled:

            - 'iterations': the number of sweeps performed.
            - 'max_sdiff': the maximum singular value difference of the final
              sweep (``-1.0`` if no diffs were computed).

        progbar : bool, optional
            Whether to show a progress bar tracking the max singular value
            change per sweep.
        inplace : bool, optional
            Whether to gauge this network inplace or return a gauged copy.

        Returns
        -------
        TensorNetworkInfinite2DFlat
        """
        if fuse_multibonds:
            raise NotImplementedError(
                "Multibonds are gauged per bond_type representative, "
                "fuse_multibonds=True is not supported."
            )

        itn = self if inplace else self.copy()

        gauges_supplied = gauges is not None
        if not gauges_supplied:
            gauges = {}

        if info is None:
            info = {}
        compute_diff = (tol > 0.0) or progbar

        if progbar:
            import tqdm

            pbar = tqdm.tqdm()
        else:
            pbar = None

        it = 0
        unconverged = True
        while unconverged and it < max_iterations:
            # seeding "max_sdiff" tells tensor_gauge_simple_bond to track the
            # running max diff here (can only converge early if tol > 0.0)
            if compute_diff:
                info["max_sdiff"] = -1.0

            for bond_type in itn.geometry.bond_types:
                sitea, siteb = bond_type
                ta = itn.fragment[sitea]
                tb = itn.fragment[siteb]
                bix = itn.get_bond_ind(sitea, siteb)

                tensor_gauge_simple_bond(
                    ta,
                    tb,
                    gauges,
                    bond_ind=bix,
                    fuse_multibonds=False,
                    smudge=smudge,
                    power=power,
                    renorm=True,
                    info=info,
                )

                # propagate the updated tensors and gauge to all translates
                itn._sync_site(sitea, ta)
                itn._sync_site(siteb, tb)
                itn._sync_bond(bond_type, bix, gauges)

            if pbar is not None:
                pbar.update()
                pbar.set_description(f"max|dS|={info['max_sdiff']:.2e}")

            unconverged = (tol == 0.0) or (info["max_sdiff"] > tol)
            it += 1

        if pbar is not None:
            pbar.close()

        if not gauges_supplied:
            # absorb the internal gauges back into the tensors (conditioning)
            itn.gauge_simple_insert(gauges)

        info["iterations"] = it
        # report -1.0 if diffs were never computed (tol == 0.0 and no progbar)
        info.setdefault("max_sdiff", -1.0)
        # internal running diff, remove
        info.pop("sdiff", None)

        return itn

    gauge_all_simple_ = functools.partialmethod(gauge_all_simple, inplace=True)

    def max_bond(self):
        """The largest bond dimension in the fragment."""
        return self.fragment.max_bond()

    def gauge_simple_insert(self, gauges):
        """Absorb the bond ``gauges`` into the tensors, split half-half per
        ``bond_type`` representative and synced to translates. The ``gauges``
        dict is read but not modified (matching
        ``TensorNetwork.gauge_simple_insert``).
        """
        for sitea, siteb in self.geometry.bond_types:
            bix = self.get_bond_ind(sitea, siteb)
            s_half = gauges[bix] ** 0.5
            ta = self.fragment[sitea]
            tb = self.fragment[siteb]
            ta.multiply_index_diagonal_(bix, s_half)
            tb.multiply_index_diagonal_(bix, s_half)
            self._sync_site(sitea, ta)
            self._sync_site(siteb, tb)

    def normalize_simple(self, gauges):
        """Normalize the state and bond ``gauges`` in place, translation
        invariantly: each gauge to unit 2-norm, and each ``site_type`` to unit
        local norm. The local norm is computed on the cell ``(0, 0)``
        representative (always fully interior, so all its bonds carry gauges)
        and the rescaled tensor synced to all translates.
        """
        # normalize the gauges (equal across translates -> stays equal)
        for ix, g in gauges.items():
            gauges[ix] = g / do("linalg.norm", g)

        # normalize each site_type by the local norm of its (0, 0) tensor
        for site_type in self.site_types:
            rep = ((0, 0), site_type)
            t = self.fragment[rep]
            # local norm = norm of the fully gauged single-site tensor
            tg = t.copy()
            for ix in t.inds:
                if ix in gauges:
                    tg.multiply_index_diagonal_(ix, gauges[ix])
            lnorm = tg.norm()
            t.modify(apply=lambda d, _l=lnorm: d / _l)
            self._sync_site(rep)

    def _region_sites(self, where, radius):
        """The set of sites within ``radius`` hops of any site in ``where``."""
        sites = set()
        for site in ensure_inf_2d_sites(where):
            sites |= self.geometry.get_sites_within_radius(site, radius)
        return sites

    def build_fragment(self, sites):
        """Build a standalone fragment tensor network over an arbitrary set of
        ``sites``, filling each with its ``site_type``'s shared data so the
        patch is translation-consistent with this network. Unlike the main
        ``fragment``, ``sites`` are not restricted to the ``[-1, 1]^2`` block;
        this is how the larger neighborhoods needed by ``max_distance > 0``
        clusters and generalized-loop expansions are materialized without
        growing the main fragment. Bonds to sites outside ``sites`` are left
        dangling. Use ``build_fragment_with_gauges`` to also get a tiled gauge
        store for the environment.

        Parameters
        ----------
        sites : iterable[site]
            The sites to include in the fragment.

        Returns
        -------
        TensorNetworkGen
        """
        sites = tuple(sites)
        fragment = self._new_fragment(sites)
        for site in sites:
            site_type = site[1]
            data = self.shared_tensors[site_type][0].data
            tags = self.get_site_tags(site)
            t = Tensor(data=data, inds=self.get_site_inds(site), tags=tags)
            fragment |= t
        return fragment

    def build_fragment_with_gauges(self, sites, gauges):
        """Build a fragment over ``sites`` (see ``build_fragment``) together
        with a copy of ``gauges`` tiled onto every bond present, including the
        dangling boundary bonds, so it can serve as the environment.

        Parameters
        ----------
        sites : iterable[site]
            The sites to include in the fragment.
        gauges : dict
            A bond-gauge store keyed by the main fragment's bond indices. The
            tiled copy is keyed by *this* fragment's own (translated) bond
            indices, each taking the value of its canonical ``bond_type`` gauge
            (the cell (0, 0) representative, always present).

        Returns
        -------
        fragment : TensorNetworkGen
        fragment_gauges : dict
        """
        sites = tuple(sites)
        fragment = self.build_fragment(sites)

        if gauges is None:
            return fragment, None

        fragment_gauges = {}
        for site in sites:
            for neighbor in self.geometry.get_site_neighbors(site):
                bond_type = self.get_bond_type(site, neighbor)
                ind_src = self.get_bond_ind(*bond_type)
                g = gauges.get(ind_src, None)
                if g is not None:
                    ind_dst = self.get_bond_ind(site, neighbor)
                    fragment_gauges[ind_dst] = g
        return fragment, fragment_gauges


class PEPSInfinite2D(TensorNetworkInfinite2DFlat):
    """Infinite 2D PEPS: a translation-invariant wavefunction with one physical
    index per site, on a unit cell (``GeometryInfinite2D``). Adds the physical
    leg, gates, and cluster expectations to ``TensorNetworkInfinite2DFlat``.

    Parameters
    ----------
    geometry_or_edges : GeometryInfinite2D or sequence of edges
        The unit-cell geometry, or a sequence of edges to build one from.
    site_tag_id : str, optional
        Format string for a site's tensor tag, formatted with the site.
    site_ind_id : str, optional
        Format string for a site's physical index, formatted with the site.
    bond_ind_id : str, optional
        Format string for a bond index, formatted with its two sorted sites.
    site_type_tag_id : str, optional
        Format string for a ``site_type``'s shared tag.
    """

    def __init__(
        self,
        geometry_or_edges,
        site_tag_id="I{}",
        site_ind_id="k{}",
        bond_ind_id="b{},{}",
        site_type_tag_id="SITE_TYPE_{}",
    ):
        # set the physical id before the base builds the fragment, the hooks
        # below read it. The fragment starts empty; use `.rand`/`.from_fill_fn`
        # to actually populate it.
        self.site_ind_id = site_ind_id
        super().__init__(
            geometry_or_edges,
            site_tag_id=site_tag_id,
            bond_ind_id=bond_ind_id,
            site_type_tag_id=site_type_tag_id,
        )

    @classmethod
    def from_fill_fn(
        cls,
        fill_fn,
        geometry_or_edges,
        bond_dim,
        phys_dim=2,
        site_tag_id="I{}",
        site_ind_id="k{}",
        bond_ind_id="b{},{}",
        site_type_tag_id="SITE_TYPE_{}",
    ):
        """Build a PEPS, filling each site_type's tensor with
        ``fill_fn(shape) -> array`` (``fill_fn`` first, matching
        ``quimb.tensor.tensor_builder.TN_from_edges_and_fill_fn``).

        Parameters
        ----------
        fill_fn : callable
            Called as ``fill_fn(shape) -> array`` once per ``site_type`` to
            generate its shared tensor data.
        geometry_or_edges : GeometryInfinite2D or sequence of edges
            The unit-cell geometry, or a sequence of edges to build one from.
        bond_dim : int
            The virtual bond dimension.
        phys_dim : int, optional
            The physical dimension.
        site_tag_id, site_ind_id, bond_ind_id, site_type_tag_id : str, optional
            Format strings for tags and indices, see the class docstring.

        Returns
        -------
        PEPSInfinite2D
        """
        self = cls(
            geometry_or_edges,
            site_tag_id=site_tag_id,
            site_ind_id=site_ind_id,
            bond_ind_id=bond_ind_id,
            site_type_tag_id=site_type_tag_id,
        )
        self._block_fill(
            fill_fn,
            lambda site: self.get_site_shape(site, bond_dim, phys_dim),
        )
        return self

    @classmethod
    def rand(
        cls,
        geometry_or_edges,
        bond_dim,
        phys_dim=2,
        *,
        seed=None,
        dtype="float64",
        site_tag_id="I{}",
        site_ind_id="k{}",
        bond_ind_id="b{},{}",
        site_type_tag_id="SITE_TYPE_{}",
        **randn_opts,
    ):
        """Build a PEPS with random dense tensors.

        Parameters
        ----------
        geometry_or_edges : GeometryInfinite2D or sequence of edges
            The unit-cell geometry, or a sequence of edges to build one from.
        bond_dim : int
            The virtual bond dimension.
        phys_dim : int, optional
            The physical dimension.
        seed : int, optional
            Random seed for reproducibility.
        dtype : str, optional
            The data type of the random entries.
        site_tag_id, site_ind_id, bond_ind_id, site_type_tag_id : str, optional
            Format strings for tags and indices, see the class docstring.
        randn_opts
            Supplied to the random fill function.

        Returns
        -------
        PEPSInfinite2D
        """
        from ...gen.rand import get_rand_fill_fn

        fill_fn = get_rand_fill_fn(seed=seed, dtype=dtype, **randn_opts)
        return cls.from_fill_fn(
            fill_fn,
            geometry_or_edges,
            bond_dim,
            phys_dim=phys_dim,
            site_tag_id=site_tag_id,
            site_ind_id=site_ind_id,
            bond_ind_id=bond_ind_id,
            site_type_tag_id=site_type_tag_id,
        )

    def _new_fragment(self, sites):
        return TensorNetworkGenVector.new(
            sites=sites,
            site_tag_id=self.site_tag_id,
            site_ind_id=self.site_ind_id,
        )

    def get_site_inds(self, site):
        # bonds, then the one physical index
        return (*super().get_site_inds(site), self.site_ind_id.format(site))

    def get_site_shape(self, site, bond_dim, phys_dim=2):
        # bond dims, then the physical dim
        return (*super().get_site_shape(site, bond_dim), phys_dim)

    def get_site_duals(self, site):
        # bond duals, then the physical index (ket -> dual=False)
        return (*super().get_site_duals(site), False)

    def copy(self, deep=False):
        new = super().copy(deep=deep)
        new.site_ind_id = self.site_ind_id
        return new

    def gate_simple(
        self,
        G,
        where,
        gauges,
        *,
        max_bond=None,
        cutoff=1e-10,
        renorm=True,
        smudge=1e-12,
        power=1.0,
        path=None,
        info=None,
        inplace=False,
        **gate_opts,
    ):
        """Apply a gate ``G`` at sites ``where`` with local truncated gauging,
        then propagate the updated tensors and bond gauges to all translates.

        Supports one-site gates (``where`` a single site), nearest-neighbor
        two-site gates, and long-range two-site gates. Long-range gates are
        applied as an MPO string along a path of sites between the two
        endpoints (the path must stay within the fragment), updating every
        tensor and bond gauge on the path. Delegates the actual gate to the
        fragment ``TensorNetworkGenVector``.

        Parameters
        ----------
        G : array_like
            The gate to apply (e.g. an exponentiated local Hamiltonian term).
        where : site or sequence[site]
            The site(s) to gate, e.g. ``(site,)`` or a ``bond_type``
            ``(site_a, site_b)``.
        gauges : dict
            Diagonal bond gauges (environments), keyed by bond index. Modified
            in place (the only way to access the updated bond gauges).
        max_bond : int, optional
            The maximum bond dimension to keep.
        cutoff : float, optional
            The singular value cutoff.
        renorm : bool, optional
            Whether to renormalize the new bond gauges before storing them.
        smudge, power : float, optional
            Numerical stabilization of the bond environments.
        path : sequence[site], optional
            For long-range gates, the explicit path of sites to use. If None
            any shortest path will be used.
        inplace : bool, optional
            Whether to gate this network inplace or a copy.

        Returns
        -------
        PEPSInfinite2D
        """
        itn = self if inplace else self.copy()

        if is_inf_2d_site(where):
            where = (where,)
        else:
            where = tuple(where)

        if path is None:
            path = itn.fragment.get_path_between_sites(where[0], where[-1])

        site_types = set()
        for site in path:
            site_type = site[1]
            if site_type in site_types:
                raise NotImplementedError(
                    "Paths cannot revisit site_type, "
                    "consider expanded unit cell."
                )
            site_types.add(site_type)
            itn.add_fragment_site(site)

        if info is None:
            info = {}

        itn.fragment.gate_simple_(
            G=G,
            where=where,
            gauges=gauges,
            max_bond=max_bond,
            cutoff=cutoff,
            renorm=renorm,
            smudge=smudge,
            power=power,
            path=path,
            info=info,
            **gate_opts,
        )

        for sa, sb in zip(path[:-1], path[1:]):
            itn._sync_site(sa)
            bix = itn.get_bond_ind(sa, sb)
            bond_type = itn.get_bond_type(sa, sb)
            itn._sync_bond(bond_type, bix, gauges)
        itn._sync_site(path[-1])

        return itn

    gate_simple_ = functools.partialmethod(gate_simple, inplace=True)

    def get_cluster(self, where, gauges=None, max_distance=0, **kwargs):
        """Get the local wavefunction cluster around ``where``, optionally
        inserting the bond ``gauges`` as the environment. Delegates to a
        fragment ``TensorNetworkGenVector``.

        For ``max_distance=0`` the main ``[-1, 1]^2`` fragment is used. Larger
        clusters are computed on a freshly built fragment tiled out to the
        ``max_distance`` neighborhood of ``where`` (see ``build_fragment``).
        """
        if max_distance == 0:
            return self.fragment.get_cluster(
                where, gauges=gauges, max_distance=0, **kwargs
            )
        fragment, fragment_gauges = self.build_fragment_with_gauges(
            self._region_sites(where, max_distance), gauges
        )
        return fragment.get_cluster(
            where, gauges=fragment_gauges, max_distance=max_distance, **kwargs
        )

    def partial_trace_cluster(
        self, where, gauges=None, max_distance=0, normalized=True, **kwargs
    ):
        """Approximate reduced density matrix at sites ``where``, formed by
        partial-tracing a ``max_distance``-cluster with the bond ``gauges`` as
        the environment. Delegates to a fragment ``TensorNetworkGenVector``
        (``get_cluster`` then ``partial_trace_exact``).

        Parameters
        ----------
        where : sequence[site]
            The sites to keep.
        gauges : dict, optional
            Diagonal bond gauges (environments), keyed by bond index.
        max_distance : int, optional
            The graph distance neighborhood to include (``0`` uses the main
            fragment, ``> 0`` builds a tiled fragment).
        normalized : bool, optional
            Whether to normalize the reduced density matrix.

        Returns
        -------
        array_like
        """
        if max_distance == 0:
            fragment, fragment_gauges = self.fragment, gauges
        else:
            fragment, fragment_gauges = self.build_fragment_with_gauges(
                self._region_sites(where, max_distance), gauges
            )
        return fragment.partial_trace_cluster(
            where,
            gauges=fragment_gauges,
            max_distance=max_distance,
            normalized=normalized,
            **kwargs,
        )

    def local_expectation_cluster(
        self, G, where, gauges=None, max_distance=0, normalized=True, **kwargs
    ):
        """Approximate local expectation of gate ``G`` at sites ``where``,
        using a ``max_distance``-cluster with the bond ``gauges`` as the
        environment. Delegates to a fragment ``TensorNetworkGenVector``.

        Parameters
        ----------
        G : array_like
            The gate to compute the expectation of.
        where : sequence[site]
            The sites to compute the expectation at (e.g. a ``bond_type``).
        gauges : dict, optional
            Diagonal bond gauges (environments), keyed by bond index.
        max_distance : int, optional
            The graph distance neighborhood to include (``0`` uses the main
            fragment, ``> 0`` builds a tiled fragment).
        normalized : bool, optional
            Whether to divide by the local norm (expectation of the identity).

        Returns
        -------
        float
        """
        if max_distance == 0:
            fragment, fragment_gauges = self.fragment, gauges
        else:
            fragment, fragment_gauges = self.build_fragment_with_gauges(
                self._region_sites(where, max_distance), gauges
            )
        return fragment.local_expectation_cluster(
            G,
            where,
            gauges=fragment_gauges,
            max_distance=max_distance,
            normalized=normalized,
            **kwargs,
        )

    def compute_local_expectation_cluster(
        self,
        terms,
        gauges=None,
        max_distance=0,
        normalized=True,
        return_all=False,
        **kwargs,
    ):
        """Sum the local cluster expectations of ``terms`` over the unit cell,
        e.g. to estimate the energy per unit cell from a ``LocalHamInfinite2D``.

        Parameters
        ----------
        terms : LocalHamInfinite2D or dict[bond_type, array_like]
            Anything with an ``.items()`` yielding ``(where, gate)``, where
            ``where`` is a ``bond_type``.
        gauges : dict, optional
            Diagonal bond gauges (environments), keyed by bond index.
        max_distance : int, optional
            The graph distance neighborhood to include (``0`` uses the main
            fragment, ``> 0`` builds a single tiled fragment covering all
            terms).
        normalized : bool, optional
            Whether to locally normalize each term.
        return_all : bool, optional
            If ``True`` return the per-``bond_type`` expectations instead of
            their sum.

        Returns
        -------
        float or dict[bond_type, float]
        """
        if max_distance == 0:
            fragment, fragment_gauges = self.fragment, gauges
        else:
            where_sites = set()
            for where, _ in terms.items():
                where_sites.update(ensure_inf_2d_sites(where))
            fragment, fragment_gauges = self.build_fragment_with_gauges(
                self._region_sites(where_sites, max_distance), gauges
            )
        return fragment.compute_local_expectation_cluster(
            terms,
            gauges=fragment_gauges,
            max_distance=max_distance,
            normalized=normalized,
            return_all=return_all,
            **kwargs,
        )

    def _gloop_region(self, where, gloops):
        """The set of sites needed to evaluate a generalized-loop expansion of
        size ``gloops`` around ``where``. An explicit set of loops contributes
        exactly its sites; an integer max-size or ``None`` (smallest loop) is
        turned into a conservative graph-distance radius (a loop of ``C`` sites
        reaches at most ``~C // 2`` hops out and back).
        """
        if gloops is None:
            # smallest non-trivial loop ~ one unit-cell plaquette
            radius = max(self.geometry.get_cell_size()) + 1
        elif isinstance(gloops, int):
            radius = gloops // 2 + 1
        else:
            # explicit loops: union their sites (plus where), already exact
            sites = set(where)
            for gloop in gloops:
                sites.update(gloop)
            return sites
        return self._region_sites(where, radius)

    def local_expectation_gloop_expand(
        self, G, where, gloops=None, gauges=None, normalized=True, **kwargs
    ):
        """Approximate local expectation of gate ``G`` at sites ``where`` via a
        generalized-loop expansion with the bond ``gauges`` as the environment.
        Computed on a freshly built fragment tiled out far enough to hold the
        loops (see ``build_fragment`` / ``_gloop_region``); delegates to the
        fragment ``TensorNetworkGenVector``.

        Parameters
        ----------
        G : array_like
            The gate to compute the expectation of.
        where : sequence[site]
            The sites to compute the expectation at (e.g. a ``bond_type``).
        gloops : None, int, or sequence[sequence[site]], optional
            The generalized loops to use, or an integer max loop size, or
            ``None`` for the smallest non-trivial loop.
        gauges : dict, optional
            Diagonal bond gauges (environments), keyed by bond index.
        normalized : bool, optional
            Whether and how to normalize the result.

        Returns
        -------
        float
        """

        fragment_sites = self._gloop_region(where, gloops)

        fragment, fragment_gauges = self.build_fragment_with_gauges(
            fragment_sites, gauges
        )
        return fragment.local_expectation_gloop_expand(
            G,
            where,
            gloops=gloops,
            gauges=fragment_gauges,
            normalized=normalized,
            **kwargs,
        )

    def compute_local_expectation_gloop_expand(
        self,
        terms,
        gloops=None,
        *,
        gauges=None,
        normalized=True,
        return_all=False,
        **kwargs,
    ):
        """Sum the generalized-loop-expansion expectations of ``terms`` over the
        unit cell, e.g. an energy per unit cell that is cheaper than (though
        less accurate than) the equivalent-size cluster estimate. Computed on a
        single fragment tiled out to hold the loops around every term; delegates
        to the fragment ``TensorNetworkGenVector``.

        Parameters
        ----------
        terms : LocalHamInfinite2D or dict[bond_type, array_like]
            Anything with an ``.items()`` yielding ``(where, gate)``.
        gloops : None, int, or sequence[sequence[site]], optional
            The generalized loops to use, or an integer max loop size, or
            ``None`` for the smallest non-trivial loop.
        gauges : dict, optional
            Diagonal bond gauges (environments), keyed by bond index.
        normalized : bool, optional
            Whether and how to normalize the result.
        return_all : bool, optional
            If ``True`` return the per-``bond_type`` expectations instead of
            their sum.

        Returns
        -------
        float or dict[bond_type, float]
        """
        where_sites = set()
        for where, _ in terms.items():
            where_sites.update(ensure_inf_2d_sites(where))
        fragment, fragment_gauges = self.build_fragment_with_gauges(
            self._gloop_region(where_sites, gloops), gauges
        )
        return fragment.compute_local_expectation_gloop_expand(
            terms,
            gloops,
            gauges=fragment_gauges,
            normalized=normalized,
            return_all=return_all,
            **kwargs,
        )
