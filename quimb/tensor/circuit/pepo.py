"""PEPO simple-update circuit simulator."""

import numpy as np
from autoray import (
    do,
)

from ...utils import (
    ensure_dict,
)
from ..tensor_builder import (
    TN_from_sites_product_state,
)
from .gates import parse_to_gate
from .simple_update import CircuitSimpleUpdate


class CircuitPEPOSimpleUpdate(CircuitSimpleUpdate):
    r"""Quantum circuit simulator that evolves an observable *backwards* in time,
    in the Heisenberg picture, by representing it as an arbitrary geometry PEPO
    and applying the gates with the simple update rule.

    Rather than evolving a state forwards, gates are simply recorded as they are
    applied; no contraction happens until an expectation value is requested.
    When :meth:`local_expectation` (or :meth:`get_evolved_operator`) is called,
    the local observable is built as a bond dimension 1 PEPO on the supplied
    ``edges`` and the recorded gates are applied in reverse order as
    :math:`O \rightarrow G^\dagger O G`, using
    :func:`~quimb.tensor.tnag.core.tensor_network_ag_gate_simple` (Vidal-style
    gauging plus compression). Gates that fall outside the reverse lightcone of
    the observable are skipped, since :math:`G^\dagger G = 1`. The evolved
    operator is finally projected onto the ``|00...0>`` initial state.

    This is the Heisenberg-picture companion to :class:`CircuitPEPSSimpleUpdate`,
    useful on lattices where evolving the full state is intractable but a single
    local observable can be evolved in a truncated, gauged operator network.

    Parameters
    ----------
    N : int, optional
        The number of qubits. If not given it is inferred from the geometry.
        Supply it to pad the geometry up to ``N`` sites.
    edges : sequence[tuple[hashable, hashable]], optional
        The edges defining the geometry. A bond is placed between each pair of
        sites, and two-qubit gates are only supported on these edges. If not
        given the geometry is inferred from the two-qubit ``gates``.
    gates : sequence, optional
        If ``edges`` is not given, infer the geometry from the two-qubit gates
        in this sequence (the gates are only inspected here, not applied).
    max_bond : int, optional
        The maximum bond dimension to compress the operator to as gates are
        applied during the backwards evolution.
    cutoff : float, optional
        The singular value cutoff to use when compressing.
    gate_contract : str, optional
        How to split a two site gate, see
        :func:`~quimb.tensor.tnag.core.tensor_network_ag_gate_simple`.
    gate_opts : dict, optional
        Default options forwarded to ``gate_simple_`` such as ``max_bond``,
        ``cutoff`` and ``renorm``. This is the single source of truth for the
        compression options; ``max_bond`` and ``cutoff`` are also exposed as
        properties.

    Attributes
    ----------
    edges : tuple[tuple[hashable, hashable]]
        The unique edges defining the geometry.
    sites : tuple[hashable]
        The sites (qubit labels).
    gates : tuple[Gate]
        The gates recorded so far.

    Examples
    --------

        >>> import quimb.tensor as qtn
        >>> edges = [(0, 1), (1, 2), (2, 3)]
        >>> circ = qtn.CircuitPEPOSimpleUpdate(edges=edges, max_bond=16)
        >>> circ.apply_gates(gates)            # no computation happens here
        >>> circ.local_expectation(qu.pauli("Z"), 1)   # evolve + contract here

    See Also
    --------
    CircuitPEPSSimpleUpdate, CircuitMPS
    """

    def __init__(
        self,
        N=None,
        *,
        edges=None,
        gates=None,
        max_bond=None,
        cutoff=1e-10,
        gate_contract="reduce-split",
        gate_opts=None,
        **circuit_opts,
    ):
        # geometry from explicit `edges`, or inferred from the two-site `gates`
        self._init_geometry(edges, gates, None, N)

        # gate_opts is the single source of truth for the compression options
        gate_opts = ensure_dict(gate_opts)
        gate_opts.setdefault("max_bond", max_bond)
        gate_opts.setdefault("cutoff", cutoff)
        gate_opts.setdefault("contract", gate_contract)
        # the operator must not be rescaled during Heisenberg evolution
        gate_opts.setdefault("renorm", False)
        gate_opts.setdefault("smudge", 1e-12)

        circuit_opts.setdefault("tag_gate_numbers", False)
        circuit_opts.setdefault("tag_gate_rounds", False)
        circuit_opts.setdefault("tag_gate_labels", False)

        super().__init__(len(self._sites), None, gate_opts, **circuit_opts)

    @property
    def max_bond(self):
        """The maximum bond dimension to compress to."""
        return self.gate_opts.get("max_bond")

    @max_bond.setter
    def max_bond(self, value):
        self.gate_opts["max_bond"] = value

    @property
    def cutoff(self):
        """The singular value cutoff to use when compressing."""
        return self.gate_opts.get("cutoff")

    @cutoff.setter
    def cutoff(self, value):
        self.gate_opts["cutoff"] = value

    def _init_state(self, N, dtype="complex128"):
        # the base Circuit requires a state; this |00...0> product state is
        # built to satisfy it but is never read - the backwards evolution and
        # the final isel projection onto |00...0> do not use it
        zero = do("array", [1.0, 0.0], dtype=dtype)
        psi = TN_from_sites_product_state({site: zero for site in self._sites})
        for a, b in self._edges:
            psi[a].new_bond(psi[b])
        return psi

    def _apply_gate(self, gate, tags=None, **gate_opts):
        # gates are only recorded here; the actual (backwards) evolution is
        # deferred until an observable is requested
        if gate.controls:
            raise ValueError(
                "Controlled gates are not supported by "
                "`CircuitPEPOSimpleUpdate`. Supply the gate as a full unitary "
                "on its qubits instead."
            )
        if gate.special:
            raise ValueError(
                f"The special gate {gate.label!r} is not supported by "
                "`CircuitPEPOSimpleUpdate`. Supply a gate with an explicit "
                "array acting on sites connected by an edge."
            )

        where = gate.qubits
        if len(where) > 2:
            raise ValueError(
                "`CircuitPEPOSimpleUpdate` only supports one and two site "
                f"gates, but got {len(where)} sites: {where}."
            )
        if len(where) == 1:
            if where[0] not in self._site_set:
                raise ValueError(
                    f"Gate site {where[0]} is not in the geometry."
                )
        elif frozenset(where) not in self._edge_set:
            raise ValueError(
                f"The gate acts on sites {where} which are not a declared "
                "edge, only nearest neighbor gates are allowed."
            )

        self._gates.append(gate)

    def apply_gates(self, gates, progbar=False, **gate_opts):
        if progbar:
            from ...utils import progbar as _progbar

            gates = _progbar(tuple(gates))

        for gate in gates:
            self._apply_gate(parse_to_gate(gate), **gate_opts)

    def _parse_where(self, where):
        if isinstance(where, list):
            where = tuple(where)
        if where in self._site_set:
            # a single site label (which may itself be a tuple, e.g. a coord)
            where = (where,)
        elif not isinstance(where, tuple):
            # a single scalar label that is not a site, wrap it so it fails the
            # membership check below with a clear error rather than `tuple(int)`
            where = (where,)
        for site in where:
            if site not in self._site_set:
                raise ValueError(
                    f"Observable site {site} is not in the geometry."
                )
        if len(where) == 2 and frozenset(where) not in self._edge_set:
            raise ValueError(
                f"Observable on {where} is not a nearest neighbor edge."
            )
        if len(where) > 2:
            raise ValueError(
                "Observables on more than two sites are not supported."
            )
        return where

    def _initial_operator(self, G, where):
        """Build the bond dimension 1 PEPO of ``G`` acting at ``where`` and
        the identity elsewhere, on the circuit geometry.
        """
        from ..tensor_builder import TN_from_edges_and_fill_fn

        eye = np.eye(2, dtype="complex128")

        def fill_fn(shape):
            # shape is (*bond_dims, 2, 2) with all bonds of size 1
            num_bonds = len(shape) - 2
            return eye.reshape((1,) * num_bonds + (2, 2))

        op = TN_from_edges_and_fill_fn(
            fill_fn,
            self._edges,
            D=1,
            phys_dim=2,
            site_ind_id=("k{}", "b{}"),
        )
        # inject the observable on the upper indices (G acting on identity)
        contract = True if len(where) == 1 else self.gate_opts["contract"]
        op.gate_upper_(np.asarray(G), where, contract=contract)
        return op

    def get_evolved_operator(self, G, where, *, max_bond=None, cutoff=None):
        r"""Evolve the local observable ``G`` at ``where`` backwards through the
        recorded circuit, returning the Heisenberg-picture operator
        :math:`U^\dagger G U` as a gauged PEPO. Gates outside the reverse
        lightcone of the observable are skipped.

        Parameters
        ----------
        G : array_like
            The local operator acting on the site(s) in ``where``.
        where : hashable or sequence[hashable]
            The site or sites the operator acts on.
        max_bond, cutoff : optional
            Override the compression options for this call.

        Returns
        -------
        TensorNetworkGenOperator
        """
        where = self._parse_where(where)
        op = self._initial_operator(G, where)

        opts = {**self.gate_opts}
        if max_bond is not None:
            opts["max_bond"] = max_bond
        if cutoff is not None:
            opts["cutoff"] = cutoff

        gauges = {}
        # reverse lightcone: only gates touching the (growing) support of the
        # observable can affect it, the rest cancel as G^dagger G = identity
        support = set(where)
        for gate in reversed(self._gates):
            gate_where = gate.qubits
            if support.isdisjoint(gate_where):
                continue
            support.update(gate_where)
            # dagger as a matrix (reshape first): for a two-qubit tensor with
            # indices (ket0, ket1, bra0, bra1) a plain transpose is wrong, the
            # matrix conjugate-transpose gives the correct (bra, bra, ket, ket)
            arr = np.asarray(gate.array)
            dim = int(round(arr.size**0.5))
            g_dag = arr.reshape(dim, dim).conj().T
            op.gate_simple_(g_dag, gate_where, gauges, **opts)

        op.gauge_simple_insert(gauges)
        return op

    def get_evolved_operator_with_state(
        self, G, where, *, max_bond=None, cutoff=None
    ):
        r"""Return the evolved operator :math:`U^\dagger G U` projected onto the
        ``|00...0>`` initial state on both sides, i.e. the tensor network whose
        full contraction is :math:`\langle 0 | U^\dagger G U | 0 \rangle`. The
        physical indices are projected with ``isel``; the caller can contract
        the returned network however they like.
        """
        op = self.get_evolved_operator(
            G, where, max_bond=max_bond, cutoff=cutoff
        )
        selectors = {}
        for site in self._sites:
            selectors[op.upper_ind(site)] = 0
            selectors[op.lower_ind(site)] = 0
        return op.isel(selectors)

    def local_expectation(
        self,
        G,
        where,
        *,
        max_bond=None,
        cutoff=None,
        optimize="auto-hq",
        **contract_opts,
    ):
        r"""Compute :math:`\langle 0 | U^\dagger G U | 0 \rangle`, the
        expectation of the local operator ``G`` at ``where`` in the state
        prepared by the recorded circuit ``U`` acting on ``|00...0>``.

        Parameters
        ----------
        G : array_like
            The local operator acting on the site(s) in ``where``.
        where : hashable or sequence[hashable]
            The site or sites the operator acts on.
        max_bond, cutoff : optional
            Override the compression options for this call.
        optimize : str, optional
            The contraction path optimizer for the final contraction.
        contract_opts
            Supplied to the final
            :meth:`~quimb.tensor.tensor_core.TensorNetwork.contract`.

        Returns
        -------
        scalar
        """
        tn = self.get_evolved_operator_with_state(
            G, where, max_bond=max_bond, cutoff=cutoff
        )
        return tn.contract(all, optimize=optimize, **contract_opts)

    # this simulator evolves observables, not a state, so the remaining state
    # access methods, supported by the PEPS version, are also unsupported

    def _unsupported(self, name):
        raise NotImplementedError(
            f"`{name}` is not available for `CircuitPEPOSimpleUpdate`, "
            "which evolves an observable in the Heisenberg picture rather "
            "than holding a forward state. Use `local_expectation` for "
            "expectation values, or `get_evolved_operator` / "
            "`get_evolved_operator_with_state` for the evolved operator."
        )

    def get_psi(self):
        self._unsupported("psi")

    def to_dense(self, *args, **kwargs):
        self._unsupported("to_dense")
