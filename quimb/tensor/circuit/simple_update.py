"""Shared base for the arbitrary-graph simple-update circuit simulators."""

from ..tensor_builder import gen_unique_edges
from .core import CircuitBase
from .gates import parse_to_gate


class CircuitSimpleUpdate(CircuitBase):
    """Common geometry and error scaffolding for the simple-update simulators
    :class:`~quimb.tensor.circuit.CircuitPEPSSimpleUpdate` (a state, evolved
    forwards) and :class:`~quimb.tensor.circuit.CircuitPEPOSimpleUpdate` (a
    local observable, evolved backwards in the Heisenberg picture). Both keep a
    single tensor per site on an arbitrary graph of ``edges``, apply one and
    two site gates with the Vidal-style simple update rule, and only ever hold
    an approximate, gauged tensor network.

    This base handles the edge/site geometry parsing, the ``edges``/``sites``
    views, the ``copy`` skeleton that carries the geometry, the natural
    site-based qubit ordering, and the friendly ``_unsupported`` error for
    representation-invalid methods. Subclasses implement ``_init_state``,
    ``_apply_gate``, ``local_expectation`` and their representation-specific
    surface.
    """

    # subclass-specific hint appended to the ``_unsupported`` error message
    _unsupported_hint = ""

    def _init_geometry(self, edges, gates, psi0, N):
        """Resolve the geometry from explicit ``edges``, else the two-site
        ``gates`` (only inspected here, not applied), else the bonds of an
        existing ``psi0``. Populates ``_edges``, ``_sites``, ``_site_set`` and
        ``_edge_set``. Every site appearing in the edges is included, plus any
        touched by single-qubit gates or present in ``psi0``, padded up to
        ``N``.
        """
        extra_sites = ()
        if edges is None:
            if psi0 is not None:
                edges = tuple(psi0.gen_bond_coos())
            elif gates is not None:
                parsed = [parse_to_gate(g) for g in gates]
                edges = [g.qubits for g in parsed if len(g.qubits) == 2]
                extra_sites = tuple(q for g in parsed for q in g.qubits)
            else:
                raise ValueError(
                    "You must supply one of `edges`, `gates` or `psi0` to "
                    "define the geometry."
                )
        self._edges = tuple(gen_unique_edges(edges))

        sites = set()
        for a, b in self._edges:
            sites.add(a)
            sites.add(b)
        sites.update(extra_sites)
        if psi0 is not None:
            sites.update(psi0.sites)
        if N is not None:
            sites.update(range(N))
        self._sites = tuple(sorted(sites))
        self._site_set = set(self._sites)
        self._edge_set = {frozenset(e) for e in self._edges}

    def copy(self):
        """Copy the circuit, carrying over the geometry that the base
        :class:`CircuitBase` copy does not know about. Subclasses extend this
        via ``super().copy()`` to carry any extra state (e.g. gauges).
        """
        new = super().copy()
        new._edges = self._edges
        new._sites = self._sites
        new._site_set = self._site_set
        new._edge_set = self._edge_set
        return new

    @property
    def edges(self):
        """The unique edges defining the geometry."""
        return self._edges

    @property
    def sites(self):
        """The sites (qubit labels) of the geometry."""
        return self._sites

    def calc_qubit_ordering(self, qubits=None):
        """Natural ordering given by the sites of the geometry."""
        if qubits is None:
            return tuple(self._sites)
        return tuple(sorted(qubits))

    def _unsupported(self, name):
        raise NotImplementedError(
            f"`{name}` is not available for `{type(self).__name__}`, "
            + self._unsupported_hint
        )
