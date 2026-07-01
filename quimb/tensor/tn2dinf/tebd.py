"""Local Hamiltonians and imaginary-time simple update for infinite,
translation-invariant 2D tensor networks (``tn2dinf``).
"""

import collections

from ..tnag.tebd import LocalHamGen, SimpleUpdateGen
from .geometry import GeometryInfinite2D


class LocalHamInfinite2D(LocalHamGen):
    """A local Hamiltonian on an infinite 2D lattice, defined per ``bond_type``
    and ``site_type`` (translation classes) rather than per concrete bond/site.
    Single-site terms are absorbed evenly into the covering two-site terms and
    operations on the terms (matrix exponential, etc.) are cached.

    The Hamiltonian carries its own ``GeometryInfinite2D`` which may differ from
    a wavefunction's geometry as long as the ``site_types`` match (e.g. a
    longer-range Hamiltonian on a nearest-neighbor PEPS).

    Parameters
    ----------
    geometry_or_edges : GeometryInfinite2D or sequence of edges
        The Hamiltonian's geometry (or edges to build one from).
    H2 : array_like or dict[bond_type, array_like]
        The two-site interaction terms. A single array is used as the default
        term for every ``bond_type``. A dict gives per-``bond_type`` terms, with
        ``None`` as an optional default. Keys may be given in either orientation
        (canonicalized via ``geometry.get_bond_type``, flipping the operator if
        needed).
    H1 : array_like or dict[site_type, array_like], optional
        The one-site term(s), absorbed evenly into the covering two-site terms.
        A single array is the default for every ``site_type``. A dict gives
        per-``site_type`` terms, with ``None`` as an optional default.

    Attributes
    ----------
    terms : dict[bond_type, array_like]
        The total effective local term for each ``bond_type``.
    geometry : GeometryInfinite2D
        The Hamiltonian's geometry.
    """

    def __init__(self, geometry_or_edges, H2, H1=None):
        if isinstance(geometry_or_edges, GeometryInfinite2D):
            self.geometry = geometry_or_edges
        else:
            self.geometry = GeometryInfinite2D(geometry_or_edges)

        # caches for not repeating operations (expm, kron, add, etc.)
        self._op_cache = collections.defaultdict(dict)

        bond_types = self.geometry.bond_types
        site_types = self.geometry.site_types

        # two-site terms keyed by bond_type, a single array is the default for
        # every bond_type, a dict gives per-bond_type terms (None as default)
        if hasattr(H2, "shape"):
            default_H2 = H2
            explicit_H2 = {}
        else:
            H2 = dict(H2)
            default_H2 = H2.pop(None, None)
            explicit_H2 = {}
            for where, X in H2.items():
                bond_type = self.geometry.get_bond_type(*where)
                if bond_type not in bond_types:
                    raise ValueError(
                        f"Term {where} (bond_type {bond_type}) is not in the "
                        "geometry."
                    )
                if bond_type[0][1] != where[0][1]:
                    # canonicalization flipped orientation, flip the operator
                    X = self._flip_cached(X)
                if bond_type in explicit_H2:
                    X = self._add_cached(explicit_H2[bond_type], X)
                explicit_H2[bond_type] = X

        self.terms = {}
        for bond_type in bond_types:
            term = explicit_H2.get(bond_type, default_H2)
            if term is None:
                raise ValueError(
                    f"No two-site term supplied for bond_type {bond_type}."
                )
            self.terms[bond_type] = term

        # map each site_type to the (bond_type, slot) pairs covering it
        self._site_type_to_covering = collections.defaultdict(list)
        for bond_type in bond_types:
            (_, site_type_a), (_, site_type_b) = bond_type
            self._site_type_to_covering[site_type_a].append((bond_type, 0))
            self._site_type_to_covering[site_type_b].append((bond_type, 1))

        # one-site terms keyed by site_type (single array -> default for all)
        if H1 is None:
            H1s = {}
        elif hasattr(H1, "shape"):
            H1s = {None: H1}
        else:
            H1s = dict(H1)
        default_H1 = H1s.pop(None, None)
        if default_H1 is not None:
            for site_type in site_types:
                H1s.setdefault(site_type, default_H1)

        # absorb single-site terms evenly into the covering bond_types
        for site_type, H in H1s.items():
            covering = self._site_type_to_covering[site_type]
            n = len(covering)
            if n == 0:
                raise ValueError(
                    f"Site type {site_type} is not coupled to anything."
                )
            H_tensored = (self._op_id_cached(H), self._id_op_cached(H))
            for bond_type, slot in covering:
                self.terms[bond_type] = self._add_cached(
                    self.terms[bond_type],
                    self._div_cached(H_tensored[slot], n),
                )

    @property
    def site_types(self):
        """The unique site_types in the unit cell."""
        return self.geometry.site_types

    @property
    def bond_types(self):
        """The unique bond_types in the unit cell."""
        return self.geometry.bond_types

    @property
    def nsites(self):
        """The number of site_types in the unit cell."""
        return len(self.geometry.site_types)

    def get_gate(self, where):
        """Get the local term for the ``bond_type`` of ``where``, cached."""
        return self.terms[self.geometry.get_bond_type(*where)]

    def get_auto_ordering(self, order="sort", group=False):
        """Ordering of ``bond_types`` into commuting (non site_type-overlapping)
        groups, delegated to the geometry.
        """
        return self.geometry.get_auto_ordering(order=order, group=group)

    def draw(self, *args, **kwargs):
        """Draw the Hamiltonian's geometry."""
        return self.geometry.draw(*args, **kwargs)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"#site_types={len(self.site_types)}, "
            f"#bond_types={len(self.bond_types)})"
        )


class SimpleUpdateInfinite2D(SimpleUpdateGen):
    """Imaginary-time simple update for an infinite 2D PEPS.

    Drives a ``PEPSInfinite2D`` state under a ``LocalHamInfinite2D``,
    reusing quimb's ``SimpleUpdateGen`` for all loop logic (tau schedule, gate
    ordering, periodic gauge equilibration, energy history and convergence).
    The state provides the simple-update surface the driver calls into
    (``gate_simple_``, ``gauge_all_simple_``, ``normalize_simple``,
    ``gauge_simple_insert``, ``compute_local_expectation_cluster``), keyed by
    translation class so updates are shared across the lattice.

    Parameters mirror ``quimb.tensor.tnag.tebd.SimpleUpdateGen``, e.g.::

        su = SimpleUpdateInfinite2D(psi, ham, D=4)
        su.evolve(100, tau=0.3)
        su.evolve(100, tau=0.1)
        psi_final, gauges = su.get_state(absorb_gauges="return")

    Notes
    -----
    - The loop energy is the ``max_distance=0`` cluster estimate;
      ``PEPSInfinite2D`` also offers larger clusters (more accurate) and
      generalized-loop expansions (cheaper than the equivalent-size cluster, but
      less accurate) for measuring the final state.
    - Only ``update="sequential"`` is reliable: the inherited sweep groups gates
      into layers by literal site, whereas the translation-invariant conflict
      rule is by ``site_type``, so ``update="parallel"`` (and per-layer
      equilibration) are not supported yet.
    """

    def __init__(
        self, psi0, ham, *args, compute_energy_per_site=True, **kwargs
    ):
        # default energy to per-site, the standard quoted quantity
        super().__init__(
            psi0,
            ham,
            *args,
            compute_energy_per_site=compute_energy_per_site,
            **kwargs,
        )

    def compute_local_expectation_cluster(self, terms=None, **kwargs):
        """Cluster expectation of ``terms`` on the current state, inserting the
        driver's current gauges as the environment. ``terms`` defaults to
        the Hamiltonian and ``gauges`` to ``self.gauges``, so a bare
        ``su.compute_local_expectation_cluster()`` measures the energy per unit
        cell. Forwards to
        :meth:`PEPSInfinite2D.compute_local_expectation_cluster`, so
        ``max_distance``, ``return_all``, etc. pass straight through.

        Returns the summed (per unit cell) value, not per-site; divide by
        ``self.ham.nsites`` to compare with :attr:`energy`.

        For a meaningful estimate the gauges should be equilibrated with the
        current tensors, with no pending imaginary-time gate, e.g. straight
        after :meth:`evolve` or via :meth:`equilibrate`.
        """
        if terms is None:
            terms = self.ham
        kwargs.setdefault("gauges", self.gauges)
        return self._psi.compute_local_expectation_cluster(terms, **kwargs)

    def compute_local_expectation_gloop_expand(
        self, terms=None, gloops=None, **kwargs
    ):
        """Generalized-loop-expansion expectation of ``terms`` on the current
        state, inserting the driver's current gauges. ``terms`` defaults to
        the Hamiltonian and ``gauges`` to ``self.gauges``. Forwards to
        :meth:`PEPSInfinite2D.compute_local_expectation_gloop_expand`.

        See :meth:`compute_local_expectation_cluster` for the per-cell return
        convention and the equilibration note.
        """
        if terms is None:
            terms = self.ham
        kwargs.setdefault("gauges", self.gauges)
        return self._psi.compute_local_expectation_gloop_expand(
            terms, gloops, **kwargs
        )
