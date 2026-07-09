"""Representation-agnostic quantum circuit interface (:class:`CircuitBase`)."""

import collections.abc
import copy
import itertools
import numbers

from autoray import (
    astype,
    get_dtype_name,
)

import quimb as qu

from ...utils import (
    LRU,
    deprecated,
    ensure_dict,
    tree_map,
)
from ...utils import progbar as _progbar
from .. import array_ops as ops
from ..tensor_core import (
    PTensor,
    TensorNetwork,
    get_tags,
    tags_to_oset,
)
from .gates import (
    SPECIAL_GATES,
    Gate,
    apply_controlled_gate,
    parse_to_gate,
)
from .qasm import (
    _is_interface_placeholder,
    _openqasm_eval_expr,
    _placeholder_param_vector,
    parse_openqasm2_file,
    parse_openqasm2_str,
    parse_openqasm2_url,
    parse_openqasm3_str,
    parse_qsim_file,
    parse_qsim_str,
    parse_qsim_url,
)


class CircuitBase:
    """Representation-agnostic interface shared by every circuit simulator.

    Holds the circuit *description* (the list of :class:`Gate` objects, gate
    application front-end and convenience methods, named-parameter management,
    backend/dtype conversion, the ``from_*`` constructors, drawing) plus the
    generic ``_apply_gate`` dispatch. It carries **no** exact-contraction
    machinery, so representation-specific simulators can compose this interface
    without inheriting the exact :class:`Circuit`.

    Subclasses must implement the ``_init_state`` and ``psi`` hooks and may
    override ``calc_qubit_ordering`` (the default is a trivial sorted order).

    Notes
    -----
    The named-parameter methods (``register_named_params``/``get_params``/
    ``set_params``/``update_params_from``) index gate tensors by tag, so they
    are only functional when ``tag_gate_numbers=True`` (the exact
    :class:`Circuit` default). Representations that disable gate-number tagging
    (MPS/PEPS/PEPO) inherit them but they are non-functional there.
    """

    def __init__(
        self,
        N=None,
        psi0=None,
        gate_opts=None,
        gate_contract="auto-split-gate",
        gate_propagate_tags="register",
        tags=None,
        psi0_dtype="complex128",
        psi0_tag="PSI0",
        tag_gate_numbers=True,
        gate_tag_id="GATE_{}",
        tag_gate_rounds=True,
        round_tag_id="ROUND_{}",
        tag_gate_labels=True,
        bra_site_ind_id="b{}",
        dtype=None,
        to_backend=None,
        convert_eager=False,
    ):
        if (N is None) and (psi0 is None):
            raise ValueError("You must supply one of `N` or `psi0`.")

        elif psi0 is None:
            self.N = N
            self._psi = self._init_state(N, dtype=psi0_dtype)

        elif N is None:
            self._psi = psi0.copy()
            self.N = psi0.nsites

        else:
            if N != psi0.nsites:
                raise ValueError("`N` doesn't match `psi0`.")
            self.N = N
            self._psi = psi0.copy()

        self._psi.add_tag(psi0_tag)

        if tags is not None:
            if isinstance(tags, str):
                tags = (tags,)
            for tag in tags:
                self._psi.add_tag(tag)

        self.tag_gate_numbers = tag_gate_numbers
        self.tag_gate_rounds = tag_gate_rounds
        self.tag_gate_labels = tag_gate_labels

        self.dtype = dtype
        self.to_backend = to_backend
        self.convert_eager = convert_eager
        if self.convert_eager:
            self._maybe_convert(self._psi)
        self._backend_gate_cache = {}

        self.gate_opts = ensure_dict(gate_opts)
        self.gate_opts.setdefault("contract", gate_contract)
        self.gate_opts.setdefault("propagate_tags", gate_propagate_tags)
        self._gates = []

        self._ket_site_ind_id = self._psi.site_ind_id
        self._bra_site_ind_id = bra_site_ind_id
        self._gate_tag_id = gate_tag_id
        self._round_tag_id = round_tag_id

        if self._ket_site_ind_id == self._bra_site_ind_id:
            raise ValueError(
                "The 'ket' and 'bra' site ind ids clash : '{}' and '{}".format(
                    self._ket_site_ind_id, self._bra_site_ind_id
                )
            )

        self._sample_n_gates = -1
        self._storage = dict()
        self._sampled_conditionals = dict()
        self._named_params = {}
        self._named_param_exprs = {}

    def copy(self):
        """Copy the circuit and its state."""
        new = object.__new__(self.__class__)
        new.N = self.N
        new._psi = self._psi.copy()
        new.gate_opts = tree_map(lambda x: x, self.gate_opts)
        new.tag_gate_numbers = self.tag_gate_numbers
        new.tag_gate_rounds = self.tag_gate_rounds
        new.tag_gate_labels = self.tag_gate_labels
        new.to_backend = self.to_backend
        new.dtype = self.dtype
        new.convert_eager = self.convert_eager
        new._backend_gate_cache = self._backend_gate_cache
        new._gates = self._gates.copy()
        new._ket_site_ind_id = self._ket_site_ind_id
        new._bra_site_ind_id = self._bra_site_ind_id
        new._gate_tag_id = self._gate_tag_id
        new._round_tag_id = self._round_tag_id
        new._sample_n_gates = self._sample_n_gates
        new._storage = self._storage.copy()
        new._sampled_conditionals = self._sampled_conditionals.copy()
        new._named_params = copy.copy(self._named_params)
        new._named_param_exprs = copy.copy(self._named_param_exprs)
        return new

    def _maybe_convert(self, obj, dtype=None):
        istn = isinstance(obj, TensorNetwork)

        if dtype is None:
            # use default dtype
            dtype = self.dtype

        if dtype is not None:
            # cast array or TN to dtype
            if istn:
                obj.astype_(dtype)
            else:
                if get_dtype_name(obj) != dtype:
                    obj = astype(obj, dtype)

        if self.to_backend is not None:
            # once dtype is enforced, apply to_backend
            # for e.g. gpu transfer etc
            if istn:
                obj.apply_to_arrays(self.to_backend)
            else:
                obj = self.to_backend(obj)

        return obj

    def apply_to_arrays(self, fn):
        """Apply a function to all the arrays in the circuit."""
        self._psi.apply_to_arrays(fn)
        self._named_params = tree_map(fn, self._named_params)

    @staticmethod
    def _normalize_named_param_value(value):
        if _is_interface_placeholder(value):
            return value
        if isinstance(value, numbers.Number):
            return ops.asarray(value)
        return value

    @property
    def named_params(self):
        """Named circuit parameters and their current values."""
        return copy.copy(self._named_params)

    @property
    def named_param_names(self):
        """Names of registered circuit parameters."""
        return tuple(self._named_params)

    @property
    def param_expressions(self):
        """Gate parameter expressions keyed by gate index."""
        return copy.copy(self._named_param_exprs)

    def register_named_params(self, named_params, gate_expressions=None):
        """Register named circuit parameters and gate dependencies.

        Parameters
        ----------
        named_params : sequence[str] or mapping[str, scalar]
            Either names to register, which default to ``nan`` until bound,
            or a mapping supplying initial values.
        gate_expressions : mapping[int, tuple], optional
            Mapping from gate index to the expressions used to generate that
            gate's parameters. Each expression can be a constant, a string
            expression referencing the named parameters, or a callable taking
            the current named parameter mapping.
        """
        if isinstance(named_params, collections.abc.Mapping):
            self._named_params = {
                name: self._normalize_named_param_value(value)
                for name, value in named_params.items()
            }
        else:
            self._named_params = {
                name: self._normalize_named_param_value(float("nan"))
                for name in tuple(named_params)
            }

        if gate_expressions is None:
            gate_expressions = {}

        normalized_gate_expressions = {}
        for i, exprs in gate_expressions.items():
            i = int(i)
            exprs = tuple(exprs)

            if not (0 <= int(i) < len(self._gates)):
                raise ValueError(
                    "Named parameter expressions reference unknown gate "
                    f"index: {i}"
                )

            gate = self._gates[i]
            if not gate.parametrize:
                raise ValueError(
                    "Named parameter expressions require parametrized gate "
                    f"indices, got non-parametrized gate: {i}"
                )

            if len(exprs) != len(gate.params):
                raise ValueError(
                    "Named parameter expression arity does not match gate "
                    f"{i}: expected {len(gate.params)}, got {len(exprs)}"
                )

            normalized_gate_expressions[i] = exprs

        self._named_param_exprs = normalized_gate_expressions
        self._apply_named_param_updates()
        self.clear_storage()

    def _set_gate_params(self, i, params):
        self._psi[self.gate_tag(i)].params = params
        self._gates[i] = self._gates[i].copy_with(params=ops.asarray(params))

    def _apply_named_param_updates(self):
        if not self._named_param_exprs:
            return

        env = dict(self._named_params)
        for i, exprs in self._named_param_exprs.items():
            values = tuple(_openqasm_eval_expr(expr, env) for expr in exprs)
            if any(isinstance(x, str) for x in values):
                raise ValueError(
                    "Named parameter binding left unresolved symbolic values "
                    f"for gate {i}: {values!r}"
                )
            if any(_is_interface_placeholder(x) for x in values):
                values = _placeholder_param_vector(values)
            self._set_gate_params(i, values)

    def get_params(self):
        """Get a pytree - in this case a dict - of all the parameters in the
        circuit.

        Returns
        -------
        dict
            Dictionary containing any named parameters plus any directly
            parametrized gates not driven by named parameter expressions.
        """
        params = dict(self._named_params)
        managed_gates = set(self._named_param_exprs)
        params.update(
            {
                i: self._psi[self.gate_tag(i)].params
                for i, gate in enumerate(self._gates)
                if gate.parametrize and i not in managed_gates
            }
        )
        return params

    def set_params(self, params):
        """Set the parameters of the circuit.

        Parameters
        ----------
        params : dict
            Dictionary mapping gate numbers and/or registered named parameter
            names to new values.
        """
        if params is None:
            params = {}

        named_updates = {k: v for k, v in params.items() if isinstance(k, str)}
        gate_updates = {
            k: v for k, v in params.items() if not isinstance(k, str)
        }

        if named_updates and not self._named_params:
            raise TypeError(
                "String-keyed parameters require registered named parameters."
            )

        extra = set(named_updates) - set(self._named_params)
        if extra:
            raise ValueError(
                "Unknown named parameter values supplied for: "
                + ", ".join(sorted(extra))
            )

        overlap = set(gate_updates) & set(self._named_param_exprs)
        if overlap:
            raise ValueError(
                "Cannot directly set gate parameters managed by named "
                "parameter expressions: "
                + ", ".join(map(str, sorted(overlap)))
            )

        if named_updates:
            self._named_params.update(
                {
                    name: self._normalize_named_param_value(value)
                    for name, value in named_updates.items()
                }
            )
            self._apply_named_param_updates()

        for i, p in gate_updates.items():
            self._set_gate_params(i, p)
        self.clear_storage()

    @classmethod
    def from_qsim_str(cls, contents, progbar=False, **circuit_opts):
        """Generate a ``Circuit`` instance from a 'qsim' string."""
        info = parse_qsim_str(contents)
        qc = cls(info["n"], **circuit_opts)
        qc.apply_gates(info["gates"], progbar=progbar)
        return qc

    @classmethod
    def from_qsim_file(cls, fname, progbar=False, **circuit_opts):
        """Generate a ``Circuit`` instance from a 'qsim' file.

        The qsim file format is described here:
        https://quantumai.google/qsim/input_format.
        """
        info = parse_qsim_file(fname)
        qc = cls(info["n"], **circuit_opts)
        qc.apply_gates(info["gates"], progbar=progbar)
        return qc

    @classmethod
    def from_qsim_url(cls, url, progbar=False, **circuit_opts):
        """Generate a ``Circuit`` instance from a 'qsim' url."""
        info = parse_qsim_url(url)
        qc = cls(info["n"], **circuit_opts)
        qc.apply_gates(info["gates"], progbar=progbar)
        return qc

    from_qasm = deprecated(from_qsim_str, "from_qasm", "from_qsim_str")

    from_qasm_file = deprecated(
        from_qsim_file, "from_qasm_file", "from_qsim_file"
    )

    from_qasm_url = deprecated(from_qsim_url, "from_qasm_url", "from_qsim_url")

    @classmethod
    def from_openqasm2_str(cls, contents, progbar=False, **circuit_opts):
        """Generate a ``Circuit`` instance from an OpenQASM 2.0 string."""
        info = parse_openqasm2_str(contents)
        qc = cls(info["n"], **circuit_opts)
        qc.apply_gates(info["gates"], progbar)
        return qc

    @classmethod
    def from_openqasm2_file(cls, fname, progbar=False, **circuit_opts):
        """Generate a ``Circuit`` instance from an OpenQASM 2.0 file."""
        info = parse_openqasm2_file(fname)
        qc = cls(info["n"], **circuit_opts)
        qc.apply_gates(info["gates"], progbar=progbar)
        return qc

    @classmethod
    def from_openqasm2_url(cls, url, progbar=False, **circuit_opts):
        """Generate a ``Circuit`` instance from an OpenQASM 2.0 url."""
        info = parse_openqasm2_url(url)
        qc = cls(info["n"], **circuit_opts)
        qc.apply_gates(info["gates"], progbar=progbar)
        return qc

    @classmethod
    def from_openqasm3_str(cls, contents, progbar=False, **circuit_opts):
        """Construct a circuit from an OpenQASM 3.0 string.

        Parameters
        ----------
        contents : str
            The OpenQASM 3 source code to parse.
        progbar : bool, optional
            Whether to show a progress bar while applying the parsed gates.
        **circuit_opts
            Options forwarded to the ``Circuit`` constructor.

        Returns
        -------
        Circuit
            A circuit populated with the parsed gates. If symbolic ``input``
            declarations are present, they are registered as generic named
            circuit parameters so that :meth:`set_params` can bind them later.
        """
        info = parse_openqasm3_str(contents)
        qc = cls(info["n"], **circuit_opts)
        qc.apply_gates(info["gates"], progbar=progbar)
        qc.register_named_params(
            {
                name: (value if not isinstance(value, str) else float("nan"))
                for name, value in info["symbols"].items()
            },
            info["expressions"],
        )
        return qc

    @classmethod
    def from_openqasm3_file(cls, fname, progbar=False, **circuit_opts):
        """Construct a circuit from an OpenQASM 3.0 file.

        Parameters
        ----------
        fname : str or path-like
            Path to the OpenQASM 3 file.
        progbar : bool, optional
            Whether to show a progress bar while applying the parsed gates.
        **circuit_opts
            Options forwarded to the ``Circuit`` constructor.

        Returns
        -------
        Circuit
            The parsed circuit instance.
        """
        with open(fname) as f:
            return cls.from_openqasm3_str(
                f.read(), progbar=progbar, **circuit_opts
            )

    @classmethod
    def from_openqasm3_url(cls, url, progbar=False, **circuit_opts):
        """Construct a circuit from an OpenQASM 3.0 URL.

        Parameters
        ----------
        url : str
            URL pointing to an OpenQASM 3 source file.
        progbar : bool, optional
            Whether to show a progress bar while applying the parsed gates.
        **circuit_opts
            Options forwarded to the ``Circuit`` constructor.

        Returns
        -------
        Circuit
            The parsed circuit instance.
        """
        from urllib import request

        return cls.from_openqasm3_str(
            request.urlopen(url).read().decode(),
            progbar=progbar,
            **circuit_opts,
        )

    @classmethod
    def from_gates(cls, gates, N=None, progbar=False, **kwargs):
        """Generate a ``Circuit`` instance from a sequence of gates.

        Parameters
        ----------
        gates : sequence[Gate] or sequence[tuple]
            The sequence of gates to apply.
        N : int, optional
            The number of qubits. If not given, will be inferred from the
            gates.
        progbar : bool, optional
            Whether to show a progress bar.
        kwargs
            Supplied to the ``Circuit`` constructor.
        """
        if N is None:
            gates = tuple(gates)

            N = 0
            for gate in gates:
                gate = parse_to_gate(gate)
                if gate.qubits:
                    N = max(N, max(gate.qubits) + 1)
                if gate.controls:
                    N = max(N, max(gate.controls) + 1)

        qc = cls(N, **kwargs)
        qc.apply_gates(gates, progbar=progbar)
        return qc

    @property
    def gates(self):
        return tuple(self._gates)

    @property
    def num_gates(self):
        return len(self._gates)

    def ket_site_ind(self, i):
        """Get the site index for the given qubit."""
        return self._ket_site_ind_id.format(i)

    def bra_site_ind(self, i):
        """Get the 'bra' site index for the given qubit, if forming an operator."""
        return self._bra_site_ind_id.format(i)

    def gate_tag(self, g):
        """Get the tag for the given gate, indexed linearly."""
        return self._gate_tag_id.format(g)

    def round_tag(self, r):
        """Get the tag for the given round (/layer)."""
        return self._round_tag_id.format(r)

    def _apply_gate(self, gate, tags=None, **gate_opts):
        """Apply a ``Gate`` to this ``Circuit``. This is the main method that
        all calls to apply a gate should go through.

        Parameters
        ----------
        gate : Gate
            The gate to apply.
        tags : str or sequence of str, optional
            Tags to add to the gate tensor(s).
        """
        tags = tags_to_oset(tags)
        if self.tag_gate_numbers:
            tags.add(self.gate_tag(self.num_gates))
        if self.tag_gate_rounds and (gate.round is not None):
            tags.add(self.round_tag(gate.round))
        if self.tag_gate_labels and (gate.tag is not None):
            tags.add(gate.tag)

        # overide any default gate opts
        opts = {**self.gate_opts, **gate_opts}

        if gate.controls:
            # handle extra (low-rank) control structure
            apply_controlled_gate(self._psi, gate, tags=tags, **opts)

        elif gate.special:
            # these are specified as a general function
            SPECIAL_GATES[gate.label](
                self._psi, *gate.params, *gate.qubits, **opts
            )

        else:
            # gate supplied as a matrix/tensor
            G = gate.array

            if self.convert_eager:
                key = id(G)
                if key not in self._backend_gate_cache:
                    self._backend_gate_cache[key] = self._maybe_convert(G)
                G = self._backend_gate_cache[key]

            # apply the gate to the TN!
            self._psi.gate_(G, gate.qubits, tags=tags, **opts)

        # keep track of the gates applied
        self._gates.append(gate)

    def apply_gate(
        self,
        gate_id,
        *gate_args,
        params=None,
        qubits=None,
        controls=None,
        gate_round=None,
        parametrize=None,
        **gate_opts,
    ):
        """Apply a single gate to this tensor network quantum circuit. If
        ``gate_round`` is supplied the tensor(s) added will be tagged with
        ``'ROUND_{gate_round}'``. Alternatively, putting an integer first like
        so::

            circuit.apply_gate(10, 'H', 7)

        Is automatically translated to::

            circuit.apply_gate('H', 7, gate_round=10)

        Parameters
        ----------
        gate_id : Gate, str, or array_like
            Which gate to apply. This can be:

            - A ``Gate`` instance, i.e. with parameters and qubits already
              specified.
            - A string, e.g. ``'H'``, ``'U3'``, etc. in which case
              ``gate_args`` should be supplied with ``(*params, *qubits)``.
            - A raw array, in which case ``gate_args`` should be supplied with
              ``(*qubits,)``.

        gate_args : list[str]
            The arguments to supply to it.
        gate_round : int, optional
            The gate round. If ``gate_id`` is integer-like, will also be taken
            from here, with then ``gate_id, gate_args = gate_args[0],
            gate_args[1:]``.
        gate_opts
            Supplied to the gate function, options here will override the
            default ``gate_opts``.
        """
        gate = parse_to_gate(
            gate_id,
            *gate_args,
            params=params,
            qubits=qubits,
            controls=controls,
            gate_round=gate_round,
            parametrize=parametrize,
        )
        self._apply_gate(gate, **gate_opts)

    def apply_gate_raw(
        self, U, where, controls=None, gate_round=None, **gate_opts
    ):
        """Apply the raw array ``U`` as a gate on qubits in ``where``. It will
        be assumed to be unitary for the sake of computing reverse lightcones.
        """
        gate = Gate.from_raw(U, where, controls=controls, round=gate_round)
        self._apply_gate(gate, **gate_opts)

    def apply_gates(self, gates, progbar=False, **gate_opts):
        """Apply a sequence of gates to this tensor network quantum circuit.

        Parameters
        ----------
        gates : Sequence[Gate] or Sequence[Tuple]
            The sequence of gates to apply.
        gate_opts
            Supplied to :meth:`~quimb.tensor.circuit.Circuit.apply_gate`.
        """
        if progbar:
            from ...utils import progbar as _progbar

            gates = _progbar(gates)

        for gate in gates:
            if isinstance(gate, Gate):
                self._apply_gate(gate, **gate_opts)
            else:
                self.apply_gate(*gate, **gate_opts)

    def h(self, i, gate_round=None, **kwargs):
        self.apply_gate("H", i, gate_round=gate_round, **kwargs)

    def x(self, i, gate_round=None, **kwargs):
        self.apply_gate("X", i, gate_round=gate_round, **kwargs)

    def y(self, i, gate_round=None, **kwargs):
        self.apply_gate("Y", i, gate_round=gate_round, **kwargs)

    def z(self, i, gate_round=None, **kwargs):
        self.apply_gate("Z", i, gate_round=gate_round, **kwargs)

    def s(self, i, gate_round=None, **kwargs):
        self.apply_gate("S", i, gate_round=gate_round, **kwargs)

    def sdg(self, i, gate_round=None, **kwargs):
        self.apply_gate("SDG", i, gate_round=gate_round, **kwargs)

    def t(self, i, gate_round=None, **kwargs):
        self.apply_gate("T", i, gate_round=gate_round, **kwargs)

    def tdg(self, i, gate_round=None, **kwargs):
        self.apply_gate("TDG", i, gate_round=gate_round, **kwargs)

    def sx(self, i, gate_round=None, **kwargs):
        self.apply_gate("SX", i, gate_round=gate_round, **kwargs)

    def sxdg(self, i, gate_round=None, **kwargs):
        self.apply_gate("SXDG", i, gate_round=gate_round, **kwargs)

    def x_1_2(self, i, gate_round=None, **kwargs):
        self.apply_gate("X_1_2", i, gate_round=gate_round, **kwargs)

    def y_1_2(self, i, gate_round=None, **kwargs):
        self.apply_gate("Y_1_2", i, gate_round=gate_round, **kwargs)

    def z_1_2(self, i, gate_round=None, **kwargs):
        self.apply_gate("Z_1_2", i, gate_round=gate_round, **kwargs)

    def w_1_2(self, i, gate_round=None, **kwargs):
        self.apply_gate("W_1_2", i, gate_round=gate_round, **kwargs)

    def hz_1_2(self, i, gate_round=None, **kwargs):
        self.apply_gate("HZ_1_2", i, gate_round=gate_round, **kwargs)

    def cnot(self, i, j, gate_round=None, **kwargs):
        self.apply_gate("CNOT", i, j, gate_round=gate_round, **kwargs)

    def cx(self, i, j, gate_round=None, **kwargs):
        self.apply_gate("CX", i, j, gate_round=gate_round, **kwargs)

    def cy(self, i, j, gate_round=None, **kwargs):
        self.apply_gate("CY", i, j, gate_round=gate_round, **kwargs)

    def cz(self, i, j, gate_round=None, **kwargs):
        self.apply_gate("CZ", i, j, gate_round=gate_round, **kwargs)

    def iswap(self, i, j, gate_round=None, **kwargs):
        self.apply_gate("ISWAP", i, j, **kwargs)

    def iden(self, i, gate_round=None):
        pass

    def swap(self, i, j, gate_round=None, **kwargs):
        self.apply_gate("SWAP", i, j, **kwargs)

    def rx(self, theta, i, gate_round=None, parametrize=False, **kwargs):
        self.apply_gate(
            "RX",
            theta,
            i,
            gate_round=gate_round,
            parametrize=parametrize,
            **kwargs,
        )

    def ry(self, theta, i, gate_round=None, parametrize=False, **kwargs):
        self.apply_gate(
            "RY",
            theta,
            i,
            gate_round=gate_round,
            parametrize=parametrize,
            **kwargs,
        )

    def rz(self, theta, i, gate_round=None, parametrize=False, **kwargs):
        self.apply_gate(
            "RZ",
            theta,
            i,
            gate_round=gate_round,
            parametrize=parametrize,
            **kwargs,
        )

    def u3(
        self,
        theta,
        phi,
        lamda,
        i,
        gate_round=None,
        parametrize=False,
        **kwargs,
    ):
        self.apply_gate(
            "U3",
            theta,
            phi,
            lamda,
            i,
            gate_round=gate_round,
            parametrize=parametrize,
            **kwargs,
        )

    def u2(self, phi, lamda, i, gate_round=None, parametrize=False, **kwargs):
        self.apply_gate(
            "U2",
            phi,
            lamda,
            i,
            gate_round=gate_round,
            parametrize=parametrize,
            **kwargs,
        )

    def u1(self, lamda, i, gate_round=None, parametrize=False, **kwargs):
        self.apply_gate(
            "U1",
            lamda,
            i,
            gate_round=gate_round,
            parametrize=parametrize,
            **kwargs,
        )

    def phase(self, lamda, i, gate_round=None, parametrize=False, **kwargs):
        self.apply_gate(
            "PHASE",
            lamda,
            i,
            gate_round=gate_round,
            parametrize=parametrize,
            **kwargs,
        )

    def cu3(
        self,
        theta,
        phi,
        lamda,
        i,
        j,
        gate_round=None,
        parametrize=False,
        **kwargs,
    ):
        self.apply_gate(
            "CU3",
            theta,
            phi,
            lamda,
            i,
            j,
            gate_round=gate_round,
            parametrize=parametrize,
            **kwargs,
        )

    def cu2(
        self, phi, lamda, i, j, gate_round=None, parametrize=False, **kwargs
    ):
        self.apply_gate(
            "CU2",
            phi,
            lamda,
            i,
            j,
            gate_round=gate_round,
            parametrize=parametrize,
            **kwargs,
        )

    def cu1(self, lamda, i, j, gate_round=None, parametrize=False, **kwargs):
        self.apply_gate(
            "CU1",
            lamda,
            i,
            j,
            gate_round=gate_round,
            parametrize=parametrize,
            **kwargs,
        )

    def cphase(
        self, lamda, i, j, gate_round=None, parametrize=False, **kwargs
    ):
        self.apply_gate(
            "CPHASE",
            lamda,
            i,
            j,
            gate_round=gate_round,
            parametrize=parametrize,
            **kwargs,
        )

    def fsim(
        self, theta, phi, i, j, gate_round=None, parametrize=False, **kwargs
    ):
        self.apply_gate(
            "FSIM",
            theta,
            phi,
            i,
            j,
            gate_round=gate_round,
            parametrize=parametrize,
            **kwargs,
        )

    def fsimg(
        self,
        theta,
        zeta,
        chi,
        gamma,
        phi,
        i,
        j,
        gate_round=None,
        parametrize=False,
        **kwargs,
    ):
        self.apply_gate(
            "FSIMG",
            theta,
            zeta,
            chi,
            gamma,
            phi,
            i,
            j,
            gate_round=gate_round,
            parametrize=parametrize,
            **kwargs,
        )

    def givens(
        self, theta, i, j, gate_round=None, parametrize=False, **kwargs
    ):
        self.apply_gate(
            "GIVENS",
            theta,
            i,
            j,
            gate_round=gate_round,
            parametrize=parametrize,
            **kwargs,
        )

    def givens2(
        self, theta, phi, i, j, gate_round=None, parametrize=False, **kwargs
    ):
        self.apply_gate(
            "GIVENS2",
            theta,
            phi,
            i,
            j,
            gate_round=gate_round,
            parametrize=parametrize,
            **kwargs,
        )

    def xx_plus_yy(
        self, theta, beta, i, j, gate_round=None, parametrize=False, **kwargs
    ):
        self.apply_gate(
            "XXPLUSYY",
            theta,
            beta,
            i,
            j,
            gate_round=gate_round,
            parametrize=parametrize,
            **kwargs,
        )

    def xx_minus_yy(
        self, theta, beta, i, j, gate_round=None, parametrize=False, **kwargs
    ):
        self.apply_gate(
            "XXMINUSYY",
            theta,
            beta,
            i,
            j,
            gate_round=gate_round,
            parametrize=parametrize,
            **kwargs,
        )

    def rxx(self, theta, i, j, gate_round=None, parametrize=False, **kwargs):
        self.apply_gate(
            "RXX",
            theta,
            i,
            j,
            gate_round=gate_round,
            parametrize=parametrize,
            **kwargs,
        )

    def ryy(self, theta, i, j, gate_round=None, parametrize=False, **kwargs):
        self.apply_gate(
            "RYY",
            theta,
            i,
            j,
            gate_round=gate_round,
            parametrize=parametrize,
            **kwargs,
        )

    def rzz(self, theta, i, j, gate_round=None, parametrize=False, **kwargs):
        self.apply_gate(
            "RZZ",
            theta,
            i,
            j,
            gate_round=gate_round,
            parametrize=parametrize,
            **kwargs,
        )

    def crx(self, theta, i, j, gate_round=None, parametrize=False, **kwargs):
        self.apply_gate(
            "CRX",
            theta,
            i,
            j,
            gate_round=gate_round,
            parametrize=parametrize,
            **kwargs,
        )

    def cry(self, theta, i, j, gate_round=None, parametrize=False, **kwargs):
        self.apply_gate(
            "CRY",
            theta,
            i,
            j,
            gate_round=gate_round,
            parametrize=parametrize,
            **kwargs,
        )

    def crz(self, theta, i, j, gate_round=None, parametrize=False, **kwargs):
        self.apply_gate(
            "CRZ",
            theta,
            i,
            j,
            gate_round=gate_round,
            parametrize=parametrize,
            **kwargs,
        )

    def su4(
        self,
        theta1,
        phi1,
        lamda1,
        theta2,
        phi2,
        lamda2,
        theta3,
        phi3,
        lamda3,
        theta4,
        phi4,
        lamda4,
        t1,
        t2,
        t3,
        i,
        j,
        gate_round=None,
        parametrize=False,
        **kwargs,
    ):
        self.apply_gate(
            "SU4",
            theta1,
            phi1,
            lamda1,
            theta2,
            phi2,
            lamda2,
            theta3,
            phi3,
            lamda3,
            theta4,
            phi4,
            lamda4,
            t1,
            t2,
            t3,
            i,
            j,
            gate_round=gate_round,
            parametrize=parametrize,
            **kwargs,
        )

    def ccx(self, i, j, k, gate_round=None, **kwargs):
        self.apply_gate("CCX", i, j, k, gate_round=gate_round, **kwargs)

    def ccnot(self, i, j, k, gate_round=None, **kwargs):
        self.apply_gate("CCNOT", i, j, k, gate_round=gate_round, **kwargs)

    def toffoli(self, i, j, k, gate_round=None, **kwargs):
        self.apply_gate("TOFFOLI", i, j, k, gate_round=gate_round, **kwargs)

    def ccy(self, i, j, k, gate_round=None, **kwargs):
        self.apply_gate("CCY", i, j, k, gate_round=gate_round, **kwargs)

    def ccz(self, i, j, k, gate_round=None, **kwargs):
        self.apply_gate("CCZ", i, j, k, gate_round=gate_round, **kwargs)

    def cswap(self, i, j, k, gate_round=None, **kwargs):
        self.apply_gate("CSWAP", i, j, k, gate_round=gate_round, **kwargs)

    def fredkin(self, i, j, k, gate_round=None, **kwargs):
        self.apply_gate("FREDKIN", i, j, k, gate_round=gate_round, **kwargs)

    def clear_storage(self):
        """Clear all cached data."""
        self._storage.clear()
        self._sampled_conditionals.clear()
        self._marginal_storage_size = 0
        self._sample_n_gates = self.num_gates

    def _maybe_init_storage(self):
        # clear/create the cache if circuit has changed
        if self._sample_n_gates != self.num_gates:
            self.clear_storage()

    def simulate_counts(self, C, seed=None, reverse=False, **to_dense_opts):
        """Simulate measuring all qubits in the computational basis many times.
        Unlike :meth:`~quimb.tensor.circuit.Circuit.sample`, this generates all
        the samples simultaneously using the full wavefunction constructed from
        :meth:`~quimb.tensor.circuit.Circuit.to_dense`, then calling
        :func:`~quimb.calc.simulate_counts`.

        .. warning::

            Because this constructs the full wavefunction it always requires
            exponential memory in the number of qubits, regardless of circuit
            depth and structure.

        Parameters
        ----------
        C : int
            The number of 'experimental runs', i.e. total counts.
        seed : int, optional
            A seed for reproducibility.
        reverse : bool, optional
            Whether to reverse the order of the subsystems, to match the
            convention of qiskit for example.
        to_dense_opts
            Suppled to :meth:`~quimb.tensor.circuit.Circuit.to_dense`.

        Returns
        -------
        results : dict[str, int]
            The number of recorded counts for each
        """
        p_dense = self.to_dense(reverse=reverse, **to_dense_opts)
        return qu.simulate_counts(p_dense, C=C, seed=seed)

    def xeb(
        self,
        samples_or_counts,
        cache=None,
        cache_maxsize=2**20,
        progbar=False,
        **amplitude_opts,
    ):
        """Compute the linear cross entropy benchmark (XEB) for samples or
        counts, amplitude per amplitude.

        Parameters
        ----------
        samples_or_counts : Iterable[str] or Dict[str, int]
            Either the raw bitstring samples or a dict mapping bitstrings to
            the number of counts observed.
        cache : dict, optional
            A dictionary to store the probabilities in, if not supplied
            ``quimb.utils.LRU(cache_maxsize)`` will be used.
        cache_maxsize, optional
            The maximum size of the cache to be used.
        progbar, optional
            Whether to show progress as the bitstrings are iterated over.
        amplitude_opts
            Supplied to :meth:`~quimb.tensor.circuit.Circuit.amplitude`.
        """
        try:
            it = samples_or_counts.items()
        except AttributeError:
            it = zip(samples_or_counts, itertools.repeat(1))

        if progbar:
            it = _progbar(it)

        M = 0
        psum = 0.0

        if cache is None:
            cache = LRU(cache_maxsize)

        for b, cnt in it:
            try:
                p = cache[b]
            except KeyError:
                p = cache[b] = abs(self.amplitude(b, **amplitude_opts)) ** 2
            psum += cnt * p
            M += cnt

        return (2**self.N) / M * psum - 1

    def update_params_from(self, tn):
        """Assuming ``tn`` is a tensor network with tensors tagged ``GATE_{i}``
        corresponding to this circuit (e.g. from ``circ.psi`` or ``circ.uni``)
        but with updated parameters, update the current circuit parameters and
        tensors with those values.

        This is an inplace modification of the ``Circuit``.

        Parameters
        ----------
        tn : TensorNetwork
            The tensor network to find the updated parameters from.
        """
        for i, gate in enumerate(self._gates):
            tag = self.gate_tag(i)
            t = tn[tag]

            # sanity check that tensor(s) `t` correspond to the correct gate
            if gate.tag not in get_tags(t):
                raise ValueError(
                    f"The tensor(s) correponding to gate {i} "
                    f"should be tagged with '{gate.tag}', got {t}."
                )

            # only update gates and tensors if they are parametrizable
            if isinstance(t, PTensor):
                # update the actual tensor
                self._psi[tag].params = t.params

                # update the circuit's gate record
                self._gates[i] = Gate(
                    label=gate.label,
                    params=t.params,
                    qubits=gate.qubits,
                    round=gate.round,
                    parametrize=True,
                )

        self.clear_storage()

    def draw(
        self,
        figsize=None,
        radius=1 / 3,
        drawcolor=(0.5, 0.5, 0.5),
        linewidth=1,
    ):
        """Draw a simple linear schematic of the circuit.

        Parameters
        ----------
        figsize : tuple, optional
            The size of the figure, if not given will be set based on the
            number of gates and qubits.
        radius : float, optional
            The radius of the gates.
        drawcolor : tuple, optional
            The color of the wires.
        linewidth : float, optional
            The linewidth of the wires.

        Returns
        -------
        fig : matplotlib.Figure
            The figure object.
        ax : matplotlib.Axes
            The axis object.
        """
        from quimb.schematic import Drawing, hash_to_color

        if figsize is None:
            figsize = (self.num_gates / 6, self.N / 6)

        d = Drawing(
            figsize=figsize,
            presets=dict(
                wire=dict(
                    color=drawcolor,
                    linewidth=linewidth,
                ),
                gate=dict(
                    radius=radius,
                ),
            ),
        )

        depths = {}
        for i, g in enumerate(self.gates):
            # level = max(depths.get(q, 0) for q in g.qubits) + 1
            level = i

            if len(g.qubits) == 1:
                (q,) = g.qubits
                # draw line from previous gate to this one
                d.line(
                    (depths.get(q, -1) + radius, q),
                    (level - radius, q),
                    preset="wire",
                    zorder=level,
                )
                # draw the gate
                d.marker(
                    (level, q),
                    color=hash_to_color(g.label),
                    zorder=0,
                    preset="gate",
                )
                # record last gate on this qubit
                depths[q] = level
            else:
                # stretch a box over all qubits
                qmin = min(g.qubits)
                qmax = max(g.qubits)
                d.rectangle(
                    (level, qmin),
                    (level, qmax),
                    color=hash_to_color(g.label),
                    zorder=0,
                    alpha=1 / 3,
                    preset="gate",
                )
                for q in g.qubits:
                    # draw markers on each qubit acted on
                    d.marker(
                        (level, q),
                        color=hash_to_color(g.label),
                        zorder=0,
                        preset="gate",
                    )
                    # draw lines from previous gate to this one
                    d.line(
                        (depths.get(q, -1) + radius, q),
                        (level - radius, q),
                        preset="wire",
                        zorder=level,
                    )
                    # record last gate on this qubit
                    depths[q] = level

        # draw final lines to the right
        level = max(depths.values(), default=0) + 1
        for q in depths:
            d.line((depths.get(q, -1), q), (level, q), preset="wire")

        return d.fig, d.ax

    def __repr__(self):
        r = "<Circuit(n={}, num_gates={}, gate_opts={})>"
        return r.format(self.N, self.num_gates, self.gate_opts)

    def _init_state(self, N, dtype="complex128"):
        raise NotImplementedError(
            "Each circuit representation must build its own initial state."
        )

    def get_psi(self):
        """Get a copy of the current state tensor network. This is the single
        method each representation must implement to expose its state, and is
        what the ``psi`` property calls.
        """
        raise NotImplementedError(
            "Each circuit representation must provide its own state via "
            "`get_psi`."
        )

    @property
    def psi(self):
        """Tensor network representation of the current state, a copy, see
        :meth:`get_psi`.
        """
        return self.get_psi()

    def calc_qubit_ordering(self, qubits=None):
        """Default trivial qubit ordering; the exact ``Circuit`` overrides this
        with a lightcone-aware ordering used for sampling.
        """
        if qubits is None:
            return tuple(range(self.N))
        return tuple(sorted(qubits))
