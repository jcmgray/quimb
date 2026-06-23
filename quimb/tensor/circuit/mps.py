"""Matrix-product-state circuit simulators."""

import numbers
import warnings

import numpy as np

import quimb as qu

from ...utils import (
    ensure_dict,
)
from .. import array_ops as ops
from ..tensor_builder import (
    MPS_computational_state,
)
from ..tnag.core import TensorNetworkGenVector
from .exact import Circuit
from .gates import parse_to_gate


class CircuitMPS(Circuit):
    """Quantum circuit simulation keeping the state always in a MPS form. If
    you think the circuit will not build up much entanglement, or you just want
    to keep a rigorous handle on how much entanglement is present, this can
    be useful.

    Parameters
    ----------
    N : int, optional
        The number of qubits in the circuit.
    psi0 : TensorNetwork1DVector, optional
        The initial state, assumed to be ``|00000....0>`` if not given. The
        state is always copied and the tag ``PSI0`` added.
    max_bond : int, optional
        The maximum bond dimension to truncate to when applying gates, if any.
        This is simply a shortcut for setting ``gate_opts['max_bond']``.
    cutoff : float, optional
        The singular value cutoff to use when truncating the state.
        This is simply a shortcut for setting ``gate_opts['cutoff']``.
    gate_opts : dict, optional
        Default options to pass to each gate, for example, "max_bond" and
        "cutoff" etc.
    gate_contract : str, optional
        The default method for applying gates. Relevant MPS options are:

        - ``'auto-mps'``: automatically choose a method that maintains the
          MPS form (default). This uses ``'swap+split'`` for 2-qubit gates
          and ``'nonlocal'`` for 3+ qubit gates.
        - ``'swap+split'``: swap nonlocal qubits to be next to each other,
          before applying the gate, then swapping them back
        - ``'nonlocal'``: turn the gate into a potentially nonlocal (sub) MPO
          and apply it directly. See :func:`tensor_network_1d_compress`.

    dtype : str, optional
        The data type to use for the state tensor.
    to_backend : callable, optional
        A function to convert tensor data to a particular backend.
    convert_eager : bool, optional
        Whether to eagerly perform dtype casting and application of
        `to_backend` as gates are supplied, or wait until after the necessary
        TNs for a particular task such as sampling are formed and simplified.
        Eager conversion (`convert_eager=True`) is the default mode for
        MPS simulation, unlike full contraction.
    circuit_opts
        Supplied to :class:`~quimb.tensor.circuit.Circuit`.

    Attributes
    ----------
    psi : MatrixProductState
        The current state of the circuit, always in MPS form.

    Examples
    --------

    Create a circuit object that always uses the "nonlocal" method for
    contracting in gates, and the "dm" compression method within that, using
    a large cutoff and maximum bond dimension::

        circ = qtn.CircuitMPS(
            N=56,
            gate_opts=dict(
                contract="nonlocal",
                method="dm",
                max_bond=1024,
                cutoff=1e-3,
            )
        )

    """

    def __init__(
        self,
        N=None,
        *,
        psi0=None,
        max_bond=None,
        cutoff=1e-10,
        gate_opts=None,
        gate_contract="auto-mps",
        dtype=None,
        to_backend=None,
        convert_eager=True,
        **circuit_opts,
    ):
        gate_opts = ensure_dict(gate_opts)
        gate_opts.setdefault("contract", gate_contract)
        gate_opts.setdefault("propagate_tags", False)
        gate_opts.setdefault("max_bond", max_bond)
        gate_opts.setdefault("cutoff", cutoff)
        # this is used to pass around the canonical form
        gate_opts.setdefault("info", {})

        circuit_opts.setdefault("tag_gate_numbers", False)
        circuit_opts.setdefault("tag_gate_rounds", False)
        circuit_opts.setdefault("tag_gate_labels", False)

        circuit_opts.setdefault("dtype", dtype)
        circuit_opts.setdefault("to_backend", to_backend)
        circuit_opts.setdefault("convert_eager", convert_eager)

        super().__init__(N, psi0, gate_opts, **circuit_opts)

    def _init_state(self, N, dtype="complex128"):
        return MPS_computational_state("0" * N, dtype=dtype)

    def apply_gates(self, gates, progbar=False, **gate_opts):
        if progbar:
            from ...utils import progbar as _progbar

            gates = tuple(gates)
            gates = _progbar(gates, total=len(gates))
            gates.set_description(
                f"max_bond={self._psi.max_bond()}, "
                f"error~={self.error_estimate():.3g}"
            )

        for gate in gates:
            gate = parse_to_gate(gate)
            self._apply_gate(gate, **gate_opts)

            if progbar and (gate.total_qubit_count >= 2):
                # these don't change for single qubit gates
                gates.set_description(
                    f"max_bond={self._psi.max_bond()}, "
                    f"error~={self.error_estimate():.3g}"
                )

    @property
    def psi(self):
        # no squeeze so that bond dims of 1 preserved
        psi = self._psi.copy()
        if not self.convert_eager:
            self._maybe_convert(psi)
        return psi

    @property
    def uni(self):
        raise ValueError(
            "You can't extract the circuit unitary TN from a ``CircuitMPS``."
        )

    def calc_qubit_ordering(self, qubits=None):
        """MPS already has a natural ordering."""
        if qubits is None:
            return tuple(range(self.N))
        else:
            return tuple(sorted(qubits))

    def get_psi_reverse_lightcone(self, where, keep_psi0=False):
        """Override ``get_psi_reverse_lightcone`` as for an MPS the lightcone
        is not meaningful.
        """
        return self.psi

    def sample(
        self,
        C,
        seed=None,
        dtype=None,
        *,
        qubits=None,
        order=None,
        group_size=None,
        max_marginal_storage=None,
        optimize=None,
        backend=None,
        simplify_sequence=None,
        simplify_atol=None,
        simplify_equalize_norms=None,
    ):
        """Sample the MPS circuit ``C`` times.

        Parameters
        ----------
        C : int
            The number of samples to generate.
        seed : None, int, or generator, optional
            A random seed or generator to use for reproducibility.
        """
        unsupported = (
            qubits,
            order,
            group_size,
            max_marginal_storage,
            optimize,
            backend,
            simplify_sequence,
            simplify_atol,
            simplify_equalize_norms,
        )

        if any(x is not None for x in unsupported):
            warnings.warn(
                "Unsupported options for sampling an MPS circuit supplied, "
                "ignoring: " + ", ".join(map(str, unsupported))
            )

        if dtype is not None or not self.convert_eager:
            psi = self._psi.copy()
            self._maybe_convert(psi, dtype)
        else:
            psi = self._psi

        for config, _ in psi.sample(C, seed=seed):
            yield "".join(map(str, config))

    def fidelity_estimate(self):
        r"""Estimate the fidelity of the current state based on its norm, which
        tracks how much the state has been truncated:

        .. math::

            \tilde{F} =
            \left| \langle \psi | \psi \rangle \right|^2
            \approx
            \left|\langle \psi_\mathrm{ideal} | \psi \rangle\right|^2

        See Also
        --------
        error_estimate
        """
        cur_orthog = self.gate_opts["info"].get("cur_orthog", None)

        if cur_orthog is None:
            return abs(self._psi.norm()) ** 2

        cmin, cmax = cur_orthog
        return abs(self._psi[cmin : cmax + 1].norm(tags=all)) ** 2

    def error_estimate(self):
        r"""Estimate the error in the current state based on the norm of the
        discarded part of the state:

        .. math::

            \epsilon = 1 - \tilde{F}

        See Also
        --------
        fidelity_estimate
        """
        return 1 - self.fidelity_estimate()

    def local_expectation(
        self,
        G,
        where,
        normalized=False,
        dtype=None,
        *,
        simplify_sequence=None,
        simplify_atol=None,
        simplify_equalize_norms=None,
        backend=None,
        rehearse=None,
        **contract_opts,
    ):
        """Compute the local expectation value of a local operator at ``where``
        (via forming the reduced density matrix). Note this moves the
        orthogonality around inplace, and records it in `info`.

        Parameters
        ----------
        G : Tensor
            The local operator tensor.
        where : int
            The qubit to compute the expectation value at.
        normalized : bool, optional
            Whether to normalize the expectation value by the norm of the
            state.
        dtype : dtype, optional
            If given, ensure the TN is cast to this dtype before contracting.

        Returns
        -------
        float
        """
        unsupported = (
            simplify_sequence,
            simplify_atol,
            simplify_equalize_norms,
            backend,
            rehearse,
        )

        if any(x is not None for x in unsupported):
            warnings.warn(
                "Unsupported options for computing local_expectation with an "
                "MPS circuit supplied, ignoring: "
                + ", ".join(map(str, unsupported))
            )

        if dtype is not None or not self.convert_eager:
            psi = self._psi.copy()
            self._maybe_convert(psi, dtype)
        else:
            psi = self._psi

        return psi.local_expectation_canonical(
            G,
            where,
            normalized=normalized,
            info=self.gate_opts["info"],
            **contract_opts,
        )


class CircuitPermMPS(CircuitMPS):
    """Quantum circuit simulation keeping the state always in an MPS form, but
    lazily tracking the qubit ordering rather than 'swapping back' qubits after
    applying non-local gates. This can be useful for circuits with no
    expectation of locality. The qubit ordering is always tracked in the
    attribute ``qubits``. The ``psi`` attribute returns the TN with the sites
    reindexed and retagged according to the current qubit ordering, meaning it
    is no longer an MPS. Use `circ.get_psi_unordered()` to get the unpermuted
    MPS and use `circ.qubits` to get the current qubit ordering if you prefer.
    """

    def __init__(
        self,
        N=None,
        psi0=None,
        gate_opts=None,
        gate_contract="swap+split",
        **circuit_opts,
    ):
        gate_opts = ensure_dict(gate_opts)
        gate_opts.setdefault("contract", gate_contract)
        # this is used to pass around the canonical form
        gate_opts.setdefault("info", {})
        super().__init__(N, psi0=psi0, gate_opts=gate_opts, **circuit_opts)
        # keep track of the current qubit ordering
        self.qubits = list(range(self.N))

    def _apply_gate(self, gate, tags=None, **gate_opts):
        # first translate gate qubits to their current 'physical' location
        qubits = gate.qubits
        phys_sites = [self.qubits.index(q) for q in qubits]
        gate = gate.copy_with(qubits=phys_sites)

        # if the gate is non-local, account for swap (without swap back)
        if len(phys_sites) == 2:
            i, j = sorted(phys_sites)
            q = self.qubits.pop(j)
            self.qubits.insert(i + 1, q)
            gate_opts["swap_back"] = False

        super()._apply_gate(gate, tags=tags, **gate_opts)

    def calc_qubit_ordering(self, qubits=None):
        """Given by the current qubit permutation."""
        if qubits is None:
            return tuple(self.qubits)
        else:
            return tuple(sorted(qubits, key=self.qubits.index))

    def get_psi_unordered(self):
        """Return the MPS representing the state but without reordering the
        sites.
        """
        return self._psi.copy()

    def sample(
        self,
        C,
        seed=None,
        dtype=None,
    ):
        """Sample the PermMPS circuit ``C`` times.

        Parameters
        ----------
        C : int
            The number of samples to generate.
        seed : None, int, or generator, optional
            A random seed or generator to use for reproducibility.

        Yields
        ------
        str
            The next sample bitstring.
        """
        if dtype is not None or not self.convert_eager:
            psi = self._psi.copy()
            self._maybe_convert(psi, dtype)
        else:
            psi = self._psi

        # configurations are sampled in physical order, so invert the current
        # physical-site-to-logical-qubit mapping for logical bitstring output
        site_from_qubit = {
            qubit: site for site, qubit in enumerate(self.qubits)
        }
        for config, _ in psi.sample(C, seed=seed):
            yield "".join(
                str(config[site_from_qubit[i]]) for i in range(self.N)
            )

    @property
    def psi(self):
        # need to reindex and retag the MPS
        psi = self._psi.copy()

        psi.view_as_(TensorNetworkGenVector)
        psi.reindex_(
            {
                psi.site_ind(i): psi.site_ind(q)
                for i, q in enumerate(self.qubits)
            }
        )
        psi.retag_(
            {
                psi.site_tag(i): psi.site_tag(q)
                for i, q in enumerate(self.qubits)
            }
        )

        if not self.convert_eager:
            self._maybe_convert(psi)

        return psi

    def to_dense(
        self, reverse=False, optimize="auto-hq", backend=None, dtype=None
    ):
        # contract the qubit-relabeled MPS directly (`self.psi` already maps
        # `site_ind(i)` to logical qubit `i`); no exact-TN simplification
        psi = self.psi
        self._maybe_convert(psi, dtype)
        output_inds = tuple(map(psi.site_ind, range(self.N)))
        if reverse:
            output_inds = output_inds[::-1]
        t = psi.contract(
            all, output_inds=output_inds, optimize=optimize, backend=backend
        )
        k = ops.reshape(t.data, (-1, 1))
        if isinstance(k, np.ndarray):
            k = qu.qarray(k)
        return k

    def amplitude(self, b, optimize="auto-hq", backend=None, dtype=None):
        if len(b) != self.N:
            raise ValueError(
                f"Bit-string {b} length does not "
                f"match number of qubits {self.N}."
            )
        # project each physical index onto its bitstring value first, then
        # contract the resulting scalar network (cheap single-layer amplitude)
        psi = self.psi
        self._maybe_convert(psi, dtype)
        for i, x in zip(range(self.N), b):
            psi.isel_({psi.site_ind(i): int(x)})
        return psi.contract(
            all, output_inds=(), optimize=optimize, backend=backend
        )

    def local_expectation(self, G, where, *args, **kwargs):
        # translate logical qubit(s) to their current physical site(s), since
        # the underlying MPS is stored in permuted (physical) order
        if isinstance(where, numbers.Integral):
            where = self.qubits.index(where)
        else:
            where = tuple(self.qubits.index(w) for w in where)
        return super().local_expectation(G, where, *args, **kwargs)


class CircuitMPSLazy(CircuitMPS):
    """Quantum circuit simulation keeping the state always in an MPS form, but
    lazily applying gates (via sub-MPO representation) and regularly contracting
    and compressing the gates with the state using
    :func:`~quimb.tensor.tn1d.compress.tensor_network_1d_compress`. This is in
    contrast to the TEBD approach of eagerly applying gates and then compressing which
    is used in :class:`CircuitMPS`.

    The periodicity of compression is arbitrary and can be tuned with :attr:`compress_every`,
    and one can find a goldilocks point for fastest runtime. This circuit class is best
    utilized for long-range gates, and provides a speedup over :class:`CircuitMPS` when
    using ``"src"`` compression method.

    Note :attr:`gate_contract` is obsolete for this class as 2+ qubit gates are lazily applied via
    ``"nonlocal"`` and 1 qubit gates are eagerly applied via `True`.

    Parameters
    ----------
    N : int, optional
        The number of qubits in the circuit.
    psi0 : TensorNetwork1DVector, optional
        The initial state, assumed to be ``|00000....0>`` if not given. The
        state is always copied and the tag ``PSI0`` added.
    max_bond : int, optional
        The maximum bond dimension to truncate to when applying gates, if any.
        This is simply a shortcut for setting ``gate_opts['max_bond']``.
    cutoff : float, optional
        The singular value cutoff to use when truncating the state.
        This is simply a shortcut for setting ``gate_opts['cutoff']``.
    gate_opts : dict, optional
        Default options to pass to each gate, for example, "max_bond" and
        "cutoff" etc.
    compress_opts : dict, optional
        Default options to pass to :func:`~quimb.tensor.tn1d.compress.tensor_network_1d_compress`,
        for example, "method", "max_bond" and "cutoff" etc.
    method : str, optional
        The method to use for compressing the state with the gates, passed to
        :func:`~quimb.tensor.tn1d.compress.tensor_network_1d_compress`, e.g.
        ``"dm"`` for density matrix truncation, ``"src"`` for successive
        randomized compression, or ``"direct"`` for direct SVD truncation.
    compress_every : int, optional
        How many gates to apply to any qubit before contracting and compressing
        the state with the gates.

    dtype : str, optional
        The data type to use for the state tensor.
    to_backend : callable, optional
        A function to convert tensor data to a particular backend.
    convert_eager : bool, optional
        Whether to eagerly perform dtype casting and application of
        `to_backend` as gates are supplied, or wait until after the necessary
        TNs for a particular task such as sampling are formed and simplified.
        Eager conversion (`convert_eager=True`) is the default mode for
        MPS simulation, unlike full contraction.
    circuit_opts
        Supplied to :class:`~quimb.tensor.circuit.Circuit`.

    Attributes
    ----------
    psi : MatrixProductState
        The current state of the circuit, always in MPS form.

    Examples
    --------

    Create a circuit object that always uses the "src" compression method
    using maximum bond dimension and compressing when 5 gates are applied
    to any qubit::

        circ = qtn.CircuitMPSLazy(
            N=56,
            max_bond=1024,
            cutoff=0.0,
            compress_every=5,
            method="src",
        )

    """

    def __init__(
        self,
        N=None,
        *,
        psi0=None,
        max_bond=None,
        cutoff=1e-10,
        compress_opts=None,
        method="dm",
        compress_every=2,
        dtype=None,
        to_backend=None,
        convert_eager=True,
        **circuit_opts,
    ):
        super().__init__(
            N,
            psi0=psi0,
            max_bond=max_bond,
            cutoff=cutoff,
            dtype=dtype,
            to_backend=to_backend,
            convert_eager=convert_eager,
            **circuit_opts,
        )
        # separate options for compression step to avoid unknown kwarg errors
        # when passed to `tensor_network_1d_compress`
        # note that `method`, `max_bond`, and `cutoff` are duplicated here
        # as they are also used by the gate application step, so we use
        # setters and getters dedicated to keep them in sync
        self.compress_opts = ensure_dict(compress_opts)
        self.compress_opts.setdefault("max_bond", max_bond)
        self.compress_opts.setdefault("cutoff", cutoff)
        self.compress_opts.setdefault("method", method)
        self._uncompressed_sites = dict()
        self.compress_every = compress_every

    @property
    def max_bond(self):
        return self.compress_opts.get("max_bond", None)

    @max_bond.setter
    def max_bond(self, value):
        self.compress_opts["max_bond"] = value
        self.gate_opts["max_bond"] = value

    @property
    def cutoff(self):
        return self.compress_opts.get("cutoff", 1e-10)

    @cutoff.setter
    def cutoff(self, value):
        self.compress_opts["cutoff"] = value
        self.gate_opts["cutoff"] = value

    @property
    def method(self):
        return self.compress_opts.get("method", None)

    @method.setter
    def method(self, value):
        self.compress_opts["method"] = value

    def _compress(self):
        """Compress the current state by contracting in all gates and then applying
        the specified compression method via
        :func:`~quimb.tensor.tn1d.compress.tensor_network_1d_compress`.
        """
        from ..tn1d.compress import tensor_network_1d_compress

        if not self._uncompressed_sites:
            return

        tensor_network_1d_compress(
            self._psi,
            permute_arrays=False,
            inplace=True,
            **self.compress_opts,
        )

        # `tensor_network_1d_compress` leaves the orthogonality center at the
        # first site, or the last site if `sweep_reverse` is enabled
        if self.compress_opts.get("sweep_reverse", False):
            self.gate_opts["info"]["cur_orthog"] = (self.N - 1, self.N - 1)  # type: ignore
        else:
            self.gate_opts["info"]["cur_orthog"] = (0, 0)

        self._uncompressed_sites.clear()

    def _apply_gate(self, gate, tags=None, **gate_opts):
        gate_qubits = gate.qubits

        if gate.controls:
            gate_qubits = gate.controls + gate_qubits

        # for 1q gates we can eagerly contract given they do not change the MPS
        # structure
        if len(gate_qubits) == 1:
            return super()._apply_gate(gate, tags=tags, **gate_opts)

        min_site, max_site = min(gate_qubits), max(gate_qubits)

        for site in range(min_site, max_site + 1):
            if self._uncompressed_sites.get(site, 0) >= self.compress_every:
                self._compress()
                break

        for site in range(min_site, max_site + 1):
            self._uncompressed_sites[site] = (
                self._uncompressed_sites.get(site, 0) + 1
            )

        return super()._apply_gate(
            gate, tags=tags, contract="nonlocal", method="lazy", **gate_opts
        )

    @property
    def psi(self):
        self._compress()
        return super().psi

    def sample(self, C, *args, **kwargs):
        self._compress()
        yield from super().sample(C, *args, **kwargs)

    def local_expectation(self, G, where, *args, **kwargs):
        self._compress()
        return super().local_expectation(G, where, *args, **kwargs)

    def fidelity_estimate(self):
        self._compress()
        return super().fidelity_estimate()
