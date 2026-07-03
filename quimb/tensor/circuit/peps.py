"""PEPS simple-update circuit simulator."""

from autoray import (
    do,
)

from ...utils import (
    ensure_dict,
)
from ..tensor_builder import (
    TN_from_sites_product_state,
)
from ..tensor_core import (
    Tensor,
)
from .gates import parse_to_gate
from .simple_update import CircuitSimpleUpdate


class CircuitPEPSSimpleUpdate(CircuitSimpleUpdate):
    """Quantum circuit simulation keeping the state as a generic tensor
    network (a "PEPS" defined by an arbitrary graph of ``edges``) and applying
    gates with the simple update rule. The state always keeps a single tensor
    per site, with bonds only along the supplied edges; two-qubit gates are
    only supported on those edges. Bond singular values are tracked as
    Vidal-style gauges, which makes gate application and the computation of
    local expectations cheap and approximate.

    This is useful for circuits on lattices that build up more than 1D worth of
    entanglement, where an exact or MPS simulation is intractable but a
    truncated, gauged tensor network state is a good approximation.

    Parameters
    ----------
    N : int, optional
        The number of qubits in the circuit. If not given it is inferred from
        the geometry. Supply it to pad the geometry up to ``N`` sites,
        including any that have no edges.
    edges : sequence[tuple[int, int]], optional
        The edges defining the geometry of the PEPS. A bond is placed between
        each pair of sites, and two-qubit gates are only supported on these
        edges. Every site appearing in ``edges`` is included. If not given the
        geometry is taken from ``gates`` or ``psi0`` instead.
    gates : sequence, optional
        If ``edges`` is not given, infer the geometry from the two-qubit gates
        in this sequence. The gates are only inspected here, not applied, so
        you still pass them to :meth:`apply_gates` afterwards.
    psi0 : TensorNetworkGenVector, optional
        Supply the initial state directly instead of starting from the
        ``|00...0>`` product state. If ``edges`` is not given the geometry is
        read from the bonds of this state, and the bond gauges are seeded from
        it. Only a single seeding sweep is performed; unlike imaginary time
        simple update the gauge matters immediately, so for an arbitrary
        ``psi0`` you may want to call :meth:`equilibrate` once before applying
        gates.
    max_bond : int, optional
        The maximum bond dimension to truncate to when applying gates.
    cutoff : float, optional
        The singular value cutoff to use when truncating after applying gates.
    renorm : bool, optional
        Whether to renormalize the singular values of a bond after each gate.
        The default ``False`` tracks the norm of the state rather than forcing
        it to one, which is the sensible choice for real time and general
        circuit dynamics. Set ``True`` to instead keep the state normalized
        after every gate, e.g. for the near-identity gates of imaginary time
        evolution.
    gauge_smudge : float, optional
        Small value added to the gauges before they are multiplied in and
        inverted, for numerical stability with very small singular values.
    equilibrate_every : int, optional
        If given, automatically call :meth:`equilibrate` after every this many
        gates have been applied.
    equilibrate_opts : dict, optional
        Default options forwarded to :meth:`equilibrate`.
    gate_opts : dict, optional
        Default options to pass to ``gate_simple_`` such as ``max_bond`` and
        ``cutoff``.
    dtype : str, optional
        If given, ensure the state tensors are cast to this data type.
    to_backend : callable, optional
        If given, apply this function to the state tensors to convert them to a
        particular array backend.
    convert_eager : bool, optional
        Whether to apply the ``dtype`` and ``to_backend`` conversions eagerly
        as each gate is applied. The default ``True`` matches the other running
        simulators (e.g. :class:`CircuitMPS`), since the simple update rule
        contracts each gate into the state immediately rather than building a
        lazy network to contract later.

    Attributes
    ----------
    edges : tuple[tuple[hashable, hashable]]
        The unique edges defining the PEPS geometry.
    sites : tuple[hashable]
        The sites (qubit labels) of the PEPS.
    gauges : dict[str, array]
        The current Vidal-style bond gauges (singular values), keyed by bond
        index, updated in place as gates are applied.

    Notes
    -----
    The gates applied must address qubits using the same labels that appear in
    ``edges``. Two-qubit gates are only supported along an existing edge.

    Examples
    --------

        >>> import quimb.tensor as qtn
        >>> edges = [(0, 1), (1, 2), (0, 3), (1, 4), (2, 5), (3, 4), (4, 5)]
        >>> circ = qtn.CircuitPEPSSimpleUpdate(edges=edges, max_bond=8)
        >>> circ.apply_gates(gates)
        >>> peps = circ.psi

    See Also
    --------
    CircuitMPS, CircuitDense
    """

    _unsupported_hint = (
        "which only ever holds an approximate, gauged tensor network state. "
        "Use `local_expectation` for observables or `psi` to get the gauged "
        "PEPS state and contract or sample it with the approximation you want."
    )

    def __init__(
        self,
        N=None,
        *,
        edges=None,
        gates=None,
        psi0=None,
        max_bond=None,
        cutoff=1e-10,
        renorm=False,
        gauge_smudge=1e-12,
        equilibrate_every=None,
        equilibrate_opts=None,
        gate_opts=None,
        dtype=None,
        to_backend=None,
        convert_eager=True,
        **circuit_opts,
    ):
        # geometry from explicit `edges`, the two-site `gates`, or `psi0` bonds
        self._init_geometry(edges, gates, psi0, N)

        # bond gauges tracked across gate applications
        self.gauges = {}

        # auto re-gauge every this many gates, if given
        self._equilibrate_every = equilibrate_every
        self._equilibrate_opts = ensure_dict(equilibrate_opts)

        gate_opts = ensure_dict(gate_opts)
        gate_opts.setdefault("max_bond", max_bond)
        gate_opts.setdefault("cutoff", cutoff)
        gate_opts.setdefault("renorm", renorm)
        gate_opts.setdefault("smudge", gauge_smudge)

        circuit_opts.setdefault("tag_gate_numbers", False)
        circuit_opts.setdefault("tag_gate_rounds", False)
        circuit_opts.setdefault("tag_gate_labels", False)

        circuit_opts.setdefault("dtype", dtype)
        circuit_opts.setdefault("to_backend", to_backend)
        circuit_opts.setdefault("convert_eager", convert_eager)

        super().__init__(len(self._sites), psi0, gate_opts, **circuit_opts)

        if psi0 is not None:
            # seed the bond gauges from the supplied state
            self._psi.gauge_all_simple_(gauges=self.gauges, max_iterations=1)

    def copy(self):
        """Copy the circuit, including its state, gauges and geometry. The base
        :class:`CircuitSimpleUpdate` copy carries the geometry; the gauges and
        equilibrate options are copied here so the two circuits can be evolved
        independently.
        """
        new = super().copy()
        new.gauges = dict(self.gauges)
        new._equilibrate_every = self._equilibrate_every
        new._equilibrate_opts = dict(self._equilibrate_opts)
        return new

    def _init_state(self, N, dtype="complex128"):
        # |00...0> product state with bond dimension 1 bonds along the edges
        zero = do("array", [1.0, 0.0], dtype=dtype)
        psi = TN_from_sites_product_state({site: zero for site in self._sites})
        for a, b in self.edges:
            psi[a].new_bond(psi[b])
        return psi

    def _apply_gate(self, gate, tags=None, **gate_opts):
        # route gate application through the simple update rule, threading the
        # persistent bond gauges so they stay consistent between gates
        if gate.controls:
            raise ValueError(
                "Controlled gates are not supported by "
                "`CircuitPEPSSimpleUpdate`, since the simple update rule "
                "applies a dense gate array to the sites. Supply the gate as "
                "a full unitary on its qubits instead."
            )
        if gate.special:
            raise ValueError(
                f"The special gate {gate.label!r} is not supported by "
                "`CircuitPEPSSimpleUpdate`. Supply a gate with an explicit "
                "array acting on sites connected by an edge."
            )

        where = gate.qubits
        if len(where) > 2:
            raise ValueError(
                "`CircuitPEPSSimpleUpdate` only supports one and two site "
                f"gates, but got {len(where)} sites: {where}."
            )
        if (len(where) == 2) and (frozenset(where) not in self._edge_set):
            raise ValueError(
                f"The gate acts on sites {where} which are not a declared "
                "edge of the PEPS, only nearest neighbor gates are allowed."
            )

        opts = {**self.gate_opts, **gate_opts}
        opts.pop("contract", None)
        opts.pop("propagate_tags", None)

        G = gate.array
        if self.convert_eager:
            key = id(G)
            if key not in self._backend_gate_cache:
                self._backend_gate_cache[key] = self._maybe_convert(G)
            G = self._backend_gate_cache[key]

        self._psi.gate_simple_(G, where, self.gauges, **opts)
        self._gates.append(gate)

        if self._equilibrate_every and (
            len(self._gates) % self._equilibrate_every == 0
        ):
            self.equilibrate()

    def apply_gates(self, gates, progbar=False, **gate_opts):
        if progbar:
            from ...utils import progbar as _progbar

            gates = tuple(gates)
            gates = _progbar(gates, total=len(gates))
            gates.set_description(f"max_bond={self._psi.max_bond()}")

        for gate in gates:
            gate = parse_to_gate(gate)
            self._apply_gate(gate, **gate_opts)

            if progbar and (gate.total_qubit_count >= 2):
                gates.set_description(f"max_bond={self._psi.max_bond()}")

    def equilibrate(self, **gauge_opts):
        """Re-gauge the whole state with the simple update rule, improving the
        consistency of the tracked bond gauges. This does not change the state
        represented, only the gauge, and can be called periodically between
        rounds of gates to keep the simple update approximation well behaved.

        The default options given at construction via ``equilibrate_opts`` are
        applied first, with any keyword arguments here taking precedence.

        Parameters
        ----------
        gauge_opts
            Supplied to
            :meth:`~quimb.tensor.tensor_core.TensorNetwork.gauge_all_simple_`,
            for example ``max_iterations`` and ``tol``.
        """
        opts = {**self._equilibrate_opts, **gauge_opts}
        opts.setdefault("max_iterations", 100)
        opts.setdefault("tol", 1e-10)
        self._psi.gauge_all_simple_(gauges=self.gauges, **opts)

    def local_expectation(
        self,
        G,
        where,
        *,
        max_distance=0,
        normalized=True,
        **contract_opts,
    ):
        """Compute the local expectation value of operator ``G`` at the site(s)
        ``where``, using the simple update bond gauges to approximate the
        environment beyond ``max_distance``.

        Parameters
        ----------
        G : array_like
            The local operator.
        where : hashable or sequence[hashable]
            The site or sites to compute the expectation at. A single site
            label (which may itself be a tuple, e.g. a 2D coordinate) is
            detected by membership in the set of sites.
        max_distance : int, optional
            How many graph hops of neighboring tensors to include in the local
            cluster used to approximate the reduced density matrix. The default
            ``0`` uses only the target site(s) and their gauges, matching
            :meth:`~quimb.tensor.tnag.core.TensorNetworkGenVector.compute_local_expectation_cluster`.
        normalized : bool, optional
            Whether to normalize by the local norm.
        contract_opts
            Supplied to
            :meth:`~quimb.tensor.tnag.core.TensorNetworkGenVector.compute_local_expectation_cluster`.

        Returns
        -------
        float
        """
        if isinstance(where, list):
            where = tuple(where)
        if where in self._site_set:
            where = (where,)
        else:
            where = tuple(where)
        return self._psi.compute_local_expectation_cluster(
            {where: G},
            gauges=self.gauges,
            max_distance=max_distance,
            normalized=normalized,
            **contract_opts,
        )

    def get_state(self, absorb_gauges=True):
        """Return the current PEPS state, optionally absorbing the bond gauges.

        Parameters
        ----------
        absorb_gauges : bool or "return", optional
            How to handle the tracked Vidal-style bond gauges. If ``True`` (the
            default) the gauges are absorbed, so the returned tensor network is
            the actual wavefunction (up to the simple update approximation). If
            ``False`` the gauges are added to the network as uncontracted
            diagonal tensors. If ``"return"`` the raw gauged network and a copy
            of the gauges are returned separately. The internal state is left
            untouched in every case.

        Returns
        -------
        psi : TensorNetwork
            The current state.
        gauges : dict
            The current gauges, only if ``absorb_gauges == "return"``.
        """
        psi = self._psi.copy()

        if absorb_gauges == "return":
            gauges = dict(self.gauges)
            if not self.convert_eager:
                self._maybe_convert(psi)
            return psi, gauges

        if absorb_gauges:
            # absorb the Vidal-form bond gauges so the returned TN is the state
            psi.gauge_simple_insert(self.gauges)
        else:
            # add the gauges as uncontracted diagonal tensors on their bonds
            for ix, g in self.gauges.items():
                psi |= Tensor(g, inds=[ix])

        if not self.convert_eager:
            self._maybe_convert(psi)
        return psi

    @property
    def psi(self):
        """The PEPS tensor network state, with the simple update bond gauges
        absorbed back in so that it represents the actual wavefunction (a
        proper contraction of ``psi`` gives the state, up to the simple update
        approximation). The internal gauged form is left untouched. Shorthand
        for ``get_state(absorb_gauges=True)``.
        """
        return self.get_state(absorb_gauges=True)

    def to_dense(self, *args, **kwargs):
        """Contract the gauged PEPS into a dense wavefunction, a column-vector
        ``qarray`` of length ``2**N`` ordered like :attr:`sites`, matching the
        output of :meth:`Circuit.to_dense`. This is the actual (approximate)
        state, so the cost grows exponentially with the number of qubits.

        Arguments are forwarded to
        :meth:`~quimb.tensor.tnag.core.TensorNetworkGenVector.to_dense`.
        """
        return self.psi.to_dense(*args, **kwargs)

    def amplitude(self, *args, **kwargs):
        self._unsupported("amplitude")

    def sample(self, *args, **kwargs):
        self._unsupported("sample")

    def sample_chaotic(self, *args, **kwargs):
        self._unsupported("sample_chaotic")

    def sample_chaotic_rehearse(self, *args, **kwargs):
        self._unsupported("sample_chaotic_rehearse")

    def partial_trace(self, *args, **kwargs):
        self._unsupported("partial_trace")

    @property
    def uni(self):
        raise NotImplementedError(
            "`uni` (the dense circuit unitary) is not available for "
            "`CircuitPEPSSimpleUpdate`, which never forms the full unitary. "
            "Apply gates to a state and use `psi` or `local_expectation`."
        )

    def get_psi_reverse_lightcone(self, where, keep_psi0=False):
        # the reverse lightcone is not meaningful for a simple update PEPS,
        # which always keeps the whole state, so just return it
        return self.psi
