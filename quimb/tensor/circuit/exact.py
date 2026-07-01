"""Exact tensor-network circuit simulators (``Circuit``, ``CircuitDense``)."""

import collections.abc
import copy
import functools
import itertools
import numbers
import operator
import re
import warnings

import numpy as np
from autoray import (
    astype,
    do,
    get_dtype_name,
    reshape,
)

import quimb as qu

from ...utils import (
    LRU,
    deprecated,
    ensure_dict,
    partition_all,
    tree_map,
)
from ...utils import progbar as _progbar
from .. import array_ops as ops
from ..tensor_builder import (
    TN_from_sites_computational_state,
)
from ..tensor_core import (
    PTensor,
    Tensor,
    TensorNetwork,
    get_tags,
    oset_union,
    rand_uuid,
    tags_to_oset,
)
from ..tn1d.core import Dense1D
from ..tnag.core import TensorNetworkGenOperator
from .gates import (
    SPECIAL_GATES,
    Gate,
    apply_controlled_gate,
    parse_to_gate,
    rehearsal_dict,
    sample_bitstring_from_prob_ndarray,
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

# --------------------------- main circuit class ---------------------------- #


class Circuit:
    """Class for simulating quantum circuits using tensor networks. The class
    keeps a list of :class:`Gate` objects in sync with a tensor network
    representing the current state of the circuit.

    Parameters
    ----------
    N : int, optional
        The number of qubits.
    psi0 : TensorNetwork1DVector, optional
        The initial state, assumed to be ``|00000....0>`` if not given. The
        state is always copied and the tag ``PSI0`` added.
    gate_opts : dict_like, optional
        Default keyword arguments to supply to each
        :func:`~quimb.tensor.tn1d.core.gate_TN_1D` call during the circuit.
    gate_contract : str, optional
        Shortcut for setting the default `'contract'` option in `gate_opts`.
    gate_propagate_tags : str, optional
        Shortcut for setting the default `'propagate_tags'` option in
        `gate_opts`.
    tags : str or sequence of str, optional
        Tag(s) to add to the initial wavefunction tensors (whether these are
        propagated to the rest of the circuit's tensors depends on
        ``gate_opts``).
    psi0_dtype : str, optional
        Ensure the initial state has this dtype.
    psi0_tag : str, optional
        Ensure the initial state has this tag.
    tag_gate_numbers : bool, optional
        Whether to tag each gate tensor with its number in the circuit, like
        ``"GATE_{g}"``. This is required for updating the circuit parameters.
    gate_tag_id : str, optional
        The format string for tagging each gate tensor, by default e.g.
        ``"GATE_{g}"``.
    tag_gate_rounds : bool, optional
        Whether to tag each gate tensor with its number in the circuit, like
        ``"ROUND_{r}"``.
    round_tag_id : str, optional
        The format string for tagging each round of gates, by default e.g.
        ``"ROUND_{r}"``.
    tag_gate_labels : bool, optional
        Whether to tag each gate tensor with its gate type label, e.g.
        ``{"X_1/2", "ISWAP", "CCX", ...}``..
    bra_site_ind_id : str, optional
        Use this to label 'bra' site indices when creating certain (mostly
        internal) intermediate tensor networks.
    dtype : str, optional
        A default dtype to perform calculations in. Depending on
        `convert_eager`, this is enforced *after* circuit construction
        and simplification (the default for exact simulation), or eagerly to
        the initial state and as gates are applied (the default for MPS
        simulation).
    to_backend : callable, optional
        If given, apply this function to both the initial state arrays and to
        every gate as it is applied.
    convert_eager : bool, optional
        Whether to eagerly perform dtype casting and application of
        `to_backend` as gates are supplied, or wait until after the necessary
        TNs for a particular task such as sampling are formed and simplified.
        Deferred conversion (`convert_eager=False`) is the default mode for
        full contraction.

    Attributes
    ----------
    psi : TensorNetwork1DVector
        The current circuit wavefunction as a tensor network.
    uni : TensorNetwork1DOperator
        The current circuit unitary operator as a tensor network.
    gates : tuple[Gate]
        The gates in the circuit.

    Examples
    --------

    Create 3-qubit GHZ-state:

        >>> qc = qtn.Circuit(3)
        >>> gates = [
                ('H', 0),
                ('H', 1),
                ('CNOT', 1, 2),
                ('CNOT', 0, 2),
                ('H', 0),
                ('H', 1),
                ('H', 2),
            ]
        >>> qc.apply_gates(gates)
        >>> qc.psi
        <TensorNetwork1DVector(tensors=12, indices=14, L=3, max_bond=2)>

        >>> qc.psi.to_dense().round(4)
        qarray([[ 0.7071+0.j],
                [ 0.    +0.j],
                [ 0.    +0.j],
                [-0.    +0.j],
                [-0.    +0.j],
                [ 0.    +0.j],
                [ 0.    +0.j],
                [ 0.7071+0.j]])

        >>> for b in qc.sample(10):
        ...     print(b)
        000
        000
        111
        000
        111
        111
        000
        111
        000
        000

    See Also
    --------
    Gate
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

    def _init_state(self, N, dtype="complex128"):
        return TN_from_sites_computational_state(
            site_map={i: "0" for i in range(N)}, dtype=dtype
        )

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

        self._psi.squeeze_()

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

    # constant two qubit gates

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

    # special non-tensor gates

    def iden(self, i, gate_round=None):
        pass

    def swap(self, i, j, gate_round=None, **kwargs):
        self.apply_gate("SWAP", i, j, **kwargs)

    # parametrizable gates

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

    @property
    def psi(self):
        """Tensor network representation of the wavefunction."""
        # make sure all same dtype and drop singlet dimensions
        psi = self._psi.copy()
        psi.squeeze_()
        if not self.convert_eager:
            # not converted yet
            self._maybe_convert(psi)
        return psi

    def get_uni(self, transposed=False):
        """Tensor network representation of the unitary operator (i.e. with
        the initial state removed).
        """
        U = self.psi

        if transposed:
            # rename the initial state rand_uuid bonds to 1D site inds
            ixmap = {
                self.ket_site_ind(i): self.bra_site_ind(i)
                for i in range(self.N)
            }
        else:
            ixmap = {}

        # the first `N` tensors should be the tensors of input state
        tids = tuple(U.tensor_map)[: self.N]
        for i, tid in enumerate(tids):
            t = U.pop_tensor(tid)
            (old_ix,) = t.inds

            if transposed:
                ixmap[old_ix] = f"k{i}"
            else:
                ixmap[old_ix] = f"b{i}"

        U.reindex_(ixmap)
        U.view_as_(
            TensorNetworkGenOperator,
            upper_ind_id=self._ket_site_ind_id,
            lower_ind_id=self._bra_site_ind_id,
        )

        return U

    @property
    def uni(self):

        warnings.warn(
            "In future the tensor network returned by ``circ.uni`` will not "
            "be transposed as it is currently, to match the expectation from "
            "``U = circ.uni.to_dense()`` behaving like ``U @ psi``. You can "
            "retain this behaviour with ``circ.get_uni(transposed=True)``.",
            FutureWarning,
        )
        return self.get_uni(transposed=True)

    def get_reverse_lightcone_tags(self, where):
        """Get the tags of gates in this circuit corresponding to the 'reverse'
        lightcone propagating backwards from registers in ``where``.

        Parameters
        ----------
        where : int or sequence of int
            The register or register to get the reverse lightcone of.

        Returns
        -------
        tuple[str]
            The sequence of gate tags (``GATE_{i}``, ...) corresponding to the
            lightcone.
        """
        if isinstance(where, numbers.Integral):
            cone = {where}
        else:
            cone = set(where)

        lightcone_tags = []

        for i, gate in reversed(tuple(enumerate(self._gates))):
            if gate.label == "IDEN":
                continue
            elif gate.controls:
                # TODO: only add if any *targets* in cone, requires changes
                # elsewhere to make sure tensors aren't then missing
                regs = {*gate.controls, *gate.qubits}
                if regs & cone:
                    lightcone_tags.append(self.gate_tag(i))
                    cone |= regs
            elif gate.label == "SWAP":
                i, j = gate.qubits
                i_in_cone = i in cone
                j_in_cone = j in cone
                if i_in_cone:
                    cone.add(j)
                else:
                    cone.discard(j)
                if j_in_cone:
                    cone.add(i)
                else:
                    cone.discard(i)
            else:
                regs = set(gate.qubits)
                if regs & cone:
                    lightcone_tags.append(self.gate_tag(i))
                    cone |= regs

        # initial state is always part of the lightcone
        lightcone_tags.append("PSI0")
        lightcone_tags.reverse()

        return tuple(lightcone_tags)

    def get_psi_reverse_lightcone(self, where, keep_psi0=False):
        """Get just the bit of the wavefunction in the reverse lightcone of
        sites in ``where`` - i.e. causally linked.

        Parameters
        ----------
        where : int, or sequence of int
            The sites to propagate the the lightcone back from, supplied to
            :meth:`~quimb.tensor.circuit.Circuit.get_reverse_lightcone_tags`.
        keep_psi0 : bool, optional
            Keep the tensors corresponding to the initial wavefunction
            regardless of whether they are outside of the lightcone.

        Returns
        -------
        psi_lc : TensorNetwork1DVector
        """
        if isinstance(where, numbers.Integral):
            where = (where,)

        psi = self.psi
        lightcone_tags = self.get_reverse_lightcone_tags(where)
        psi_lc = psi.select_any(lightcone_tags).view_like_(psi)

        if not keep_psi0:
            # these sites are in the lightcone regardless of being alone
            site_inds = set(map(psi.site_ind, where))

            for tid, t in tuple(psi_lc.tensor_map.items()):
                # get all tensors connected to this tensor (incld itself)
                neighbors = oset_union(psi_lc.ind_map[ix] for ix in t.inds)

                # lone tensor not attached to anything - drop it
                # but only if it isn't directly in the ``where`` region
                if (len(neighbors) == 1) and set(t.inds).isdisjoint(site_inds):
                    psi_lc.pop_tensor(tid)

        return psi_lc

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

    def get_psi_simplified(
        self, seq="ADCRS", atol=1e-12, equalize_norms=False
    ):
        """Get the full wavefunction post local tensor network simplification.

        Parameters
        ----------
        seq : str, optional
            Which local tensor network simplifications to perform and in which
            order, see
            :meth:`~quimb.tensor.tensor_core.TensorNetwork.full_simplify`.
        atol : float, optional
            The tolerance with which to compare to zero when applying
            :meth:`~quimb.tensor.tensor_core.TensorNetwork.full_simplify`.
        equalize_norms : bool, optional
            Actively renormalize tensor norms during simplification.

        Returns
        -------
        psi : TensorNetwork1DVector
        """
        self._maybe_init_storage()

        key = ("psi_simplified", seq, atol)
        if key in self._storage:
            return self._storage[key].copy()

        # we simplify and store a copy
        psi = self._psi.copy()
        psi.squeeze_()

        # make sure to keep all outer indices
        output_inds = tuple(map(psi.site_ind, range(self.N)))

        # simplify the state and cache it
        psi.full_simplify_(
            seq=seq,
            atol=atol,
            output_inds=output_inds,
            equalize_norms=equalize_norms,
        )
        self._storage[key] = psi

        # return a copy so we can modify it inplace
        return psi.copy()

    def get_rdm_lightcone_simplified(
        self,
        where,
        seq="ADCRS",
        atol=1e-12,
        equalize_norms=False,
    ):
        """Get a simplified TN of the norm of the wavefunction, with
        gates outside reverse lightcone of ``where`` cancelled, and physical
        indices within ``where`` preserved so that they can be fixed (sliced)
        or used as output indices.

        Parameters
        ----------
        where : int or sequence of int
            The region assumed to be the target density matrix essentially.
            Supplied to
            :meth:`~quimb.tensor.circuit.Circuit.get_reverse_lightcone_tags`.
        seq : str, optional
            Which local tensor network simplifications to perform and in which
            order, see
            :meth:`~quimb.tensor.tensor_core.TensorNetwork.full_simplify`.
        atol : float, optional
            The tolerance with which to compare to zero when applying
            :meth:`~quimb.tensor.tensor_core.TensorNetwork.full_simplify`.
        equalize_norms : bool, optional
            Actively renormalize tensor norms during simplification.

        Returns
        -------
        TensorNetwork
        """
        key = ("rdm_lightcone_simplified", tuple(sorted(where)), seq, atol)
        if key in self._storage:
            return self._storage[key].copy()

        ket_lc = self.get_psi_reverse_lightcone(where)

        k_inds = tuple(map(self.ket_site_ind, where))
        b_inds = tuple(map(self.bra_site_ind, where))

        bra_lc = ket_lc.conj().reindex(dict(zip(k_inds, b_inds)))
        rho_lc = bra_lc | ket_lc

        # don't want to simplify site indices in region away
        output_inds = b_inds + k_inds

        # # simplify the norm and cache it
        rho_lc.full_simplify_(
            seq=seq,
            atol=atol,
            output_inds=output_inds,
            equalize_norms=equalize_norms,
        )
        self._storage[key] = rho_lc

        # return a copy so we can modify it inplace
        return rho_lc.copy()

    def amplitude(
        self,
        b,
        optimize="auto-hq",
        simplify_sequence="ADCRS",
        simplify_atol=1e-12,
        simplify_equalize_norms=True,
        backend=None,
        dtype=None,
        rehearse=False,
    ):
        r"""Get the amplitude coefficient of bitstring ``b``.

        .. math::

            c_b = \langle b | \psi \rangle

        Parameters
        ----------
        b : str or sequence of int
            The bitstring to compute the transition amplitude for.
        optimize : str, optional
            Contraction path optimizer to use for the amplitude, can be a
            non-reusable path optimizer as only called once (though path won't
            be cached for later use in that case).
        simplify_sequence : str, optional
            Which local tensor network simplifications to perform and in which
            order, see
            :meth:`~quimb.tensor.tensor_core.TensorNetwork.full_simplify`.
        simplify_atol : float, optional
            The tolerance with which to compare to zero when applying
            :meth:`~quimb.tensor.tensor_core.TensorNetwork.full_simplify`.
        simplify_equalize_norms : bool, optional
            Actively renormalize tensor norms during simplification.
        backend : str, optional
            Backend to perform the contraction with, e.g. ``'numpy'``,
            ``'cupy'`` or ``'jax'``. Passed to ``cotengra``.
        dtype : str, optional
            Data type to cast the TN to before contraction.
        rehearse : bool or "tn", optional
            If ``True``, generate and cache the simplified tensor network and
            contraction tree but don't actually perform the contraction.
            Returns a dict with keys ``"tn"`` and ``'tree'`` with the tensor
            network that will be contracted and the corresponding contraction
            tree if so.
        """
        self._maybe_init_storage()

        if len(b) != self.N:
            raise ValueError(
                f"Bit-string {b} length does not "
                f"match number of qubits {self.N}."
            )

        fs_opts = {
            "seq": simplify_sequence,
            "atol": simplify_atol,
            "equalize_norms": simplify_equalize_norms,
        }

        # get the full wavefunction simplified
        psi_b = self.get_psi_simplified(**fs_opts)

        # fix the output indices to the correct bitstring
        for i, x in zip(range(self.N), b):
            psi_b.isel_({psi_b.site_ind(i): x})

        # perform a final simplification and cast
        psi_b.full_simplify_(**fs_opts)
        self._maybe_convert(psi_b, dtype)

        if rehearse == "tn":
            return psi_b

        tree = psi_b.contraction_tree(output_inds=(), optimize=optimize)

        if rehearse:
            return rehearsal_dict(psi_b, tree)

        # perform the full contraction with the tree found
        c_b = psi_b.contract(
            all, output_inds=(), optimize=tree, backend=backend
        )

        return c_b

    def amplitude_rehearse(
        self,
        b="random",
        simplify_sequence="ADCRS",
        simplify_atol=1e-12,
        simplify_equalize_norms=True,
        optimize="auto-hq",
        dtype=None,
        rehearse=True,
    ):
        """Perform just the tensor network simplifications and contraction tree
        finding associated with computing a single amplitude (caching the
        results) but don't perform the actual contraction.

        Parameters
        ----------
        b : 'random', str or sequence of int
            The bitstring to rehearse computing the transition amplitude for,
            if ``'random'`` (the default) a random bitstring will be used.
        optimize : str, optional
            Contraction path optimizer to use for the marginal, can be a
            non-reusable path optimizer as only called once (though path won't
            be cached for later use in that case).
        simplify_sequence : str, optional
            Which local tensor network simplifications to perform and in which
            order, see
            :meth:`~quimb.tensor.tensor_core.TensorNetwork.full_simplify`.
        simplify_atol : float, optional
            The tolerance with which to compare to zero when applying
            :meth:`~quimb.tensor.tensor_core.TensorNetwork.full_simplify`.
        simplify_equalize_norms : bool, optional
            Actively renormalize tensor norms during simplification.
        backend : str, optional
            Backend to perform the marginal contraction with, e.g. ``'numpy'``,
            ``'cupy'`` or ``'jax'``. Passed to ``cotengra``.
        dtype : str, optional
            Data type to cast the TN to before contraction.

        Returns
        -------
        dict

        """
        if b == "random":
            b = "r" * self.N

        return self.amplitude(
            b=b,
            optimize=optimize,
            dtype=dtype,
            rehearse=rehearse,
            simplify_sequence=simplify_sequence,
            simplify_atol=simplify_atol,
            simplify_equalize_norms=simplify_equalize_norms,
        )

    amplitude_tn = functools.partialmethod(amplitude_rehearse, rehearse="tn")

    def partial_trace(
        self,
        keep,
        optimize="auto-hq",
        simplify_sequence="ADCRS",
        simplify_atol=1e-12,
        simplify_equalize_norms=True,
        backend=None,
        dtype=None,
        rehearse=False,
    ):
        r"""Perform the partial trace on the circuit wavefunction, retaining
        only qubits in ``keep``, and making use of reverse lightcone
        cancellation:

        .. math::

            \rho_{\bar{q}} = Tr_{\bar{p}}
            |\psi_{\bar{q}} \rangle \langle \psi_{\bar{q}}|

        Where :math:`\bar{q}` is the set of qubits to keep,
        :math:`\psi_{\bar{q}}` is the circuit wavefunction only with gates in
        the causal cone of this set, and :math:`\bar{p}` is the remaining
        qubits.

        Parameters
        ----------
        keep : int or sequence of int
            The qubit(s) to keep as we trace out the rest.
        optimize : str, optional
            Contraction path optimizer to use for the reduced density matrix,
            can be a non-reusable path optimizer as only called once (though
            path won't be cached for later use in that case).
        simplify_sequence : str, optional
            Which local tensor network simplifications to perform and in which
            order, see
            :meth:`~quimb.tensor.tensor_core.TensorNetwork.full_simplify`.
        simplify_atol : float, optional
            The tolerance with which to compare to zero when applying
            :meth:`~quimb.tensor.tensor_core.TensorNetwork.full_simplify`.
        simplify_equalize_norms : bool, optional
            Actively renormalize tensor norms during simplification.
        backend : str, optional
            Backend to perform the marginal contraction with, e.g. ``'numpy'``,
            ``'cupy'`` or ``'jax'``. Passed to ``cotengra``.
        dtype : str, optional
            Data type to cast the TN to before contraction.
        rehearse : bool or "tn", optional
            If ``True``, generate and cache the simplified tensor network and
            contraction tree but don't actually perform the contraction.
            Returns a dict with keys ``"tn"`` and ``'tree'`` with the tensor
            network that will be contracted and the corresponding contraction
            tree if so.

        Returns
        -------
        array or dict
        """

        if isinstance(keep, numbers.Integral):
            keep = (keep,)

        output_inds = tuple(map(self.ket_site_ind, keep)) + tuple(
            map(self.bra_site_ind, keep)
        )

        rho = self.get_rdm_lightcone_simplified(
            where=keep,
            seq=simplify_sequence,
            atol=simplify_atol,
            equalize_norms=simplify_equalize_norms,
        )
        self._maybe_convert(rho, dtype)

        if rehearse == "tn":
            return rho

        tree = rho.contraction_tree(output_inds=output_inds, optimize=optimize)

        if rehearse:
            return rehearsal_dict(rho, tree)

        # perform the full contraction with the tree found
        rho_dense = rho.contract(
            all,
            output_inds=output_inds,
            optimize=tree,
            backend=backend,
        ).data

        return ops.reshape(rho_dense, [2 ** len(keep), 2 ** len(keep)])

    partial_trace_rehearse = functools.partialmethod(
        partial_trace, rehearse=True
    )
    partial_trace_tn = functools.partialmethod(partial_trace, rehearse="tn")

    def local_expectation(
        self,
        G,
        where,
        optimize="auto-hq",
        simplify_sequence="ADCRS",
        simplify_atol=1e-12,
        simplify_equalize_norms=True,
        backend=None,
        dtype=None,
        rehearse=False,
    ):
        r"""Compute the a single expectation value of operator ``G``, acting on
        sites ``where``, making use of reverse lightcone cancellation.

        .. math::

            \langle \psi_{\bar{q}} | G_{\bar{q}} | \psi_{\bar{q}} \rangle

        where :math:`\bar{q}` is the set of qubits :math:`G` acts one and
        :math:`\psi_{\bar{q}}` is the circuit wavefunction only with gates in
        the causal cone of this set. If you supply a tuple or list of gates
        then the expectations will be computed simultaneously.

        Parameters
        ----------
        G : array or sequence[array]
            The raw operator(s) to find the expectation of.
        where : int or sequence of int
            Which qubits the operator acts on.
        optimize : str, optional
            Contraction path optimizer to use for the local expectation,
            can be a non-reusable path optimizer as only called once (though
            path won't be cached for later use in that case).
        simplify_sequence : str, optional
            Which local tensor network simplifications to perform and in which
            order, see
            :meth:`~quimb.tensor.tensor_core.TensorNetwork.full_simplify`.
        simplify_atol : float, optional
            The tolerance with which to compare to zero when applying
            :meth:`~quimb.tensor.tensor_core.TensorNetwork.full_simplify`.
        simplify_equalize_norms : bool, optional
            Actively renormalize tensor norms during simplification.
        backend : str, optional
            Backend to perform the marginal contraction with, e.g. ``'numpy'``,
            ``'cupy'`` or ``'jax'``. Passed to ``cotengra``.
        dtype : str, optional
            Data type to cast the TN to before contraction.
        gate_opts : None or dict_like
            Options to use when applying ``G`` to the wavefunction.
        rehearse : bool or "tn", optional
            If ``True``, generate and cache the simplified tensor network and
            contraction tree but don't actually perform the contraction.
            Returns a dict with keys ``'tn'`` and ``'tree'`` with the tensor
            network that will be contracted and the corresponding contraction
            tree if so.

        Returns
        -------
        scalar, tuple[scalar] or dict
        """
        if isinstance(where, numbers.Integral):
            where = (where,)

        fs_opts = {
            "seq": simplify_sequence,
            "atol": simplify_atol,
            "equalize_norms": simplify_equalize_norms,
        }

        rho = self.get_rdm_lightcone_simplified(where=where, **fs_opts)
        k_inds = tuple(self.ket_site_ind(i) for i in where)
        b_inds = tuple(self.bra_site_ind(i) for i in where)

        if isinstance(G, (list, tuple)):
            # if we have multiple expectations create an extra indexed stack
            nG = len(G)
            G_data = do("stack", G)
            G_data = reshape(G_data, (nG,) + (2,) * 2 * len(where))
            output_inds = (rand_uuid(),)
        else:
            G_data = reshape(G, (2,) * 2 * len(where))
            output_inds = ()

        TG = Tensor(data=G_data, inds=output_inds + b_inds + k_inds)

        rhoG = rho | TG

        rhoG.full_simplify_(output_inds=output_inds, **fs_opts)
        self._maybe_convert(rhoG, dtype)

        if rehearse == "tn":
            return rhoG

        tree = rhoG.contraction_tree(
            output_inds=output_inds, optimize=optimize
        )

        if rehearse:
            return rehearsal_dict(rhoG, tree)

        g_ex = rhoG.contract(
            all,
            output_inds=output_inds,
            optimize=tree,
            backend=backend,
        )

        if isinstance(g_ex, Tensor):
            g_ex = tuple(g_ex.data)

        return g_ex

    local_expectation_rehearse = functools.partialmethod(
        local_expectation, rehearse=True
    )
    local_expectation_tn = functools.partialmethod(
        local_expectation, rehearse="tn"
    )

    def compute_marginal(
        self,
        where,
        fix=None,
        optimize="auto-hq",
        backend=None,
        dtype="complex64",
        simplify_sequence="ADCRS",
        simplify_atol=1e-6,
        simplify_equalize_norms=True,
        rehearse=False,
    ):
        """Compute the probability tensor of qubits in ``where``, given
        possibly fixed qubits in ``fix`` and tracing everything else having
        removed redundant unitary gates.

        Parameters
        ----------
        where : sequence of int
            The qubits to compute the marginal probability distribution of.
        fix : None or dict[int, str], optional
            Measurement results on other qubits to fix.
        optimize : str, optional
            Contraction path optimizer to use for the marginal, can be a
            non-reusable path optimizer as only called once (though path won't
            be cached for later use in that case).
        backend : str, optional
            Backend to perform the marginal contraction with, e.g. ``'numpy'``,
            ``'cupy'`` or ``'jax'``. Passed to ``cotengra``.
        dtype : str, optional
            Data type to cast the TN to before contraction.
        simplify_sequence : str, optional
            Which local tensor network simplifications to perform and in which
            order, see
            :meth:`~quimb.tensor.tensor_core.TensorNetwork.full_simplify`.
        simplify_atol : float, optional
            The tolerance with which to compare to zero when applying
            :meth:`~quimb.tensor.tensor_core.TensorNetwork.full_simplify`.
        simplify_equalize_norms : bool, optional
            Actively renormalize tensor norms during simplification.
        rehearse : bool or "tn", optional
            Whether to perform the marginal contraction or just return the
            associated TN and contraction tree.
        """
        self._maybe_init_storage()

        # index trick to contract straight to reduced density matrix diagonal
        # rho_ii -> p_i (i.e. insert a COPY tensor into the norm)
        output_inds = [self.ket_site_ind(i) for i in where]

        fs_opts = {
            "seq": simplify_sequence,
            "atol": simplify_atol,
            "equalize_norms": simplify_equalize_norms,
        }

        # lightcone region is target qubit plus fixed qubits
        region = set(where)
        if fix is not None:
            region |= set(fix)
        region = tuple(sorted(region))

        # have we fixed or are measuring all qubits?
        final_marginal = len(region) == self.N

        # these both are cached and produce TN copies
        if final_marginal:
            # won't need to partially trace anything -> just need ket
            nm_lc = self.get_psi_simplified(**fs_opts)
        else:
            # can use lightcone cancellation on partially traced qubits
            nm_lc = self.get_rdm_lightcone_simplified(region, **fs_opts)
            # re-connect the ket and bra indices as taking diagonal
            nm_lc.reindex_(
                {self.bra_site_ind(i): self.ket_site_ind(i) for i in region}
            )

        if fix:
            # project (slice) fixed tensors with bitstring
            # this severs the indices connecting bra and ket on fixed sites
            nm_lc.isel_({self.ket_site_ind(i): b for i, b in fix.items()})

        # having sliced we can do a final simplify
        nm_lc.full_simplify_(output_inds=output_inds, **fs_opts)

        # for stability with very small probabilities, scale by average prob
        if fix is not None:
            nfact = 2 ** len(fix)
            if final_marginal:
                nm_lc.multiply_(nfact**0.5, spread_over="all")
            else:
                nm_lc.multiply_(nfact, spread_over="all")

        # cast to desired data type
        self._maybe_convert(nm_lc, dtype)

        if rehearse == "tn":
            return nm_lc

        # NB. the tree isn't *neccesarily* the same each time due to the post
        #     projection full simplify, however there is also the lower level
        #     contraction path cache if the structure generated *is* the same
        #     so still pretty efficient to just overwrite
        tree = nm_lc.contraction_tree(
            output_inds=output_inds,
            optimize=optimize,
        )

        if rehearse:
            return rehearsal_dict(nm_lc, tree)

        # perform the full contraction with the tree found
        p_marginal = abs(
            nm_lc.contract(
                all,
                output_inds=output_inds,
                optimize=tree,
                backend=backend,
            ).data
        )

        if final_marginal:
            # we only did half the ket contraction so need to square
            p_marginal = p_marginal**2

        if fix is not None:
            p_marginal = p_marginal / nfact

        return p_marginal

    compute_marginal_rehearse = functools.partialmethod(
        compute_marginal, rehearse=True
    )
    compute_marginal_tn = functools.partialmethod(
        compute_marginal, rehearse="tn"
    )

    def calc_qubit_ordering(self, qubits=None, method="greedy-lightcone"):
        """Get a order to measure ``qubits`` in, by greedily choosing whichever
        has the smallest reverse lightcone followed by whichever expands this
        lightcone *least*.

        Parameters
        ----------
        qubits : None or sequence of int
            The qubits to generate a lightcone ordering for, if ``None``,
            assume all qubits.

        Returns
        -------
        tuple[int]
            The order to 'measure' qubits in.
        """
        self._maybe_init_storage()

        if qubits is None:
            qubits = tuple(range(self.N))
        else:
            qubits = tuple(sorted(qubits))

        key = ("lightcone_ordering", method, qubits)

        # check the cache first
        if key in self._storage:
            return self._storage[key]

        if method == "greedy-lightcone":
            cone = set()
            lctgs = {
                i: set(self.get_reverse_lightcone_tags(i)) for i in qubits
            }

            order = []
            while lctgs:
                # get the next qubit which adds least num gates to lightcone
                next_qubit = min(lctgs, key=lambda i: len(lctgs[i] - cone))
                cone |= lctgs.pop(next_qubit)
                order.append(next_qubit)

        else:
            # use graph distance based hierachical clustering
            psi = self.get_psi_simplified("R")
            qubit_inds = tuple(map(psi.site_ind, qubits))
            tids = psi._get_tids_from_inds(qubit_inds, "any")
            matcher = re.compile(psi.site_ind_id.format(r"(\d+)"))
            order = []
            for tid in psi.compute_hierarchical_ordering(tids, method=method):
                t = psi.tensor_map[tid]
                for ind in t.inds:
                    for sq in matcher.findall(ind):
                        order.append(int(sq))

        order = self._storage[key] = tuple(order)
        return order

    def _parse_qubits_order(self, qubits=None, order=None):
        """Simply initializes the default of measuring all qubits, and the
        default order, or checks that ``order`` is a permutation of ``qubits``.
        """
        if qubits is None:
            qubits = range(self.N)
        if order is None:
            order = self.calc_qubit_ordering(qubits)
        elif set(qubits) != set(order):
            raise ValueError("``order`` must be a permutation of ``qubits``.")

        return qubits, order

    def _group_order(self, order, group_size=1):
        """Take the qubit ordering ``order`` and batch it in groups of size
        ``group_size``, sorting the qubits (for caching reasons) within each
        group.
        """
        return tuple(
            tuple(sorted(g)) for g in partition_all(group_size, order)
        )

    def get_qubit_distances(self, method="dijkstra", alpha=2):
        """Get a nested dictionary of qubit distances. This is computed from a
        graph representing qubit interactions. The graph has an edge between
        qubits if they are acted on by the same gate, and the distance-weight
        of the edge is exponentially small in the number of gates between them.

        Parameters
        ----------
        method : {'dijkstra', 'resistance'}, optional
            The method to use to compute the qubit distances. See
            :func:`networkx.all_pairs_dijkstra_path_length` and
            :func:`networkx.resistance_distance`.
        alpha : float, optional
            The distance weight between qubits is ``alpha**(num_gates - 1 )``.

        Returns
        -------
        dict[int, dict[int, float]]
            The distance between each pair of qubits, accessed like
            ``distances[q1][q2]``. If two qubits are not connected, the
            distance is missing.
        """
        import networkx as nx

        G = nx.Graph()
        for g in self.gates:
            for q1, q2 in itertools.combinations(g.qubits, 2):
                if G.has_edge(q1, q2):
                    G[q1][q2]["weight"] /= alpha
                else:
                    G.add_edge(q1, q2, weight=1)

        if method == "dijkstra":
            distances = dict(
                nx.all_pairs_dijkstra_path_length(G, weight="weight")
            )
        elif method == "resistance":
            distances = nx.resistance_distance(G, weight="weight")
        else:
            raise ValueError(f"Unknown method {method}.")

        return distances

    def reordered_gates_dfs_clustered(self):
        """Get the gates reordered by a depth first search traversal of the
        multi-qubit gate graph that greedily selects successive gates which
        are 'close' in graph distance, and shifts single qubit gates to be
        adjacent to multi-qubit gates where possible.
        """
        # first we make a directed graph of the multi-qubit gates
        successors = {}
        predecessors = {}
        single_qubit_stacks = {}
        single_qubit_predecessors = {}
        last_gates = {}
        queue = []

        for i, g in enumerate(self.gates):
            if g.total_qubit_count == 1:
                # lazily accumulate single qubit gates
                (q,) = g.qubits
                single_qubit_stacks.setdefault(q, []).append(i)

            else:
                pi = predecessors[i] = []
                sqpi = single_qubit_predecessors[i] = []

                for q in g.qubits:
                    # collect any single qubit gates acting on this qubit
                    sqpi.extend(single_qubit_stacks.pop(q, []))

                    if q in last_gates:
                        # qubit has already been acted on -> have an edge
                        h = last_gates[q]
                        # mark h as a predecessor of i
                        pi.append(h)
                        # mark i as a successor of h
                        successors.setdefault(h, []).append(i)

                    # mark qubit as acted on
                    last_gates[q] = i

                if len(pi) == 0:
                    # no predecessors -> is possible starting multiqubit gate
                    queue.append(i)

        # then we traverse the multi-qubit gates in a depth first, topological
        # order, breaking ties by minimizing the distance between active qubits
        distances = self.get_qubit_distances()

        def gate_distance(i, j):
            qis = self.gates[i].qubits
            qjs = self.gates[j].qubits
            return min(
                distances[q1].get(q2, float("inf")) for q1 in qis for q2 in qjs
            )

        # sort initial queue by qubit with smallest index
        queue.sort(key=lambda i: min(self.gates[i].qubits))
        new_gates = []

        while queue:
            i = queue.pop(0)

            # first flush any single qubit gates acting on the qubits of gate i
            new_gates.extend(
                self.gates[j] for j in single_qubit_predecessors.pop(i, [])
            )
            # then add the gate itself
            new_gates.append(self.gates[i])

            # then remove i as a predecessor of its successors
            for j in successors.pop(i, []):
                pj = predecessors[j]
                pj.remove(i)
                if not pj:
                    # j has no more predecessors -> can be added to queue
                    queue.append(j)

            # check if this is the last time q is acted on,
            # if so flush any remaining single qubit gates
            for q in self.gates[i].qubits:
                if last_gates[q] == i:
                    # qubit has been acted on for the last time
                    new_gates.extend(
                        self.gates[j] for j in single_qubit_stacks.pop(q, [])
                    )

            # sort the queue of possible next gates
            queue.sort(key=lambda k: gate_distance(i, k))

        # flush any remaining single qubit gates
        for q in sorted(single_qubit_stacks):
            new_gates.extend(self.gates[j] for j in single_qubit_stacks.pop(q))

        return new_gates

    def sample(
        self,
        C,
        qubits=None,
        order=None,
        group_size=10,
        max_marginal_storage=2**20,
        seed=None,
        optimize="auto-hq",
        backend=None,
        dtype="complex64",
        simplify_sequence="ADCRS",
        simplify_atol=1e-6,
        simplify_equalize_norms=True,
    ):
        r"""Sample the circuit given by ``gates``, ``C`` times, using lightcone
        cancelling and caching marginal distribution results. This is a
        generator. This proceeds as a chain of marginal computations.

        Assuming we have ``group_size=1``, and some ordering of the qubits,
        :math:`\{q_0, q_1, q_2, q_3, \ldots\}` we first compute:

        .. math::

            p(q_0) = \mathrm{diag} \mathrm{Tr}_{1, 2, 3,\ldots}
            | \psi_{0} \rangle \langle \psi_{0} |

        I.e. simply the probability distribution on a single qubit, conditioned
        on nothing. The subscript on :math:`\psi` refers to the fact that we
        only need gates from the causal cone of qubit 0.
        From this we can sample an outcome, either 0 or 1, if we
        call this :math:`r_0` we can then move on to the next marginal:

        .. math::

            p(q_1 | r_0) = \mathrm{diag} \mathrm{Tr}_{2, 3,\ldots}
            \langle r_0
            | \psi_{0, 1} \rangle \langle \psi_{0, 1} |
            r_0 \rangle

        I.e. the probability distribution of the next qubit, given our prior
        result. We can sample from this to get :math:`r_1`. Then we compute:

        .. math::

            p(q_2 | r_0 r_1) = \mathrm{diag} \mathrm{Tr}_{3,\ldots}
            \langle r_0 r_1
            | \psi_{0, 1, 2} \rangle \langle \psi_{0, 1, 2} |
            r_0 r_1 \rangle

        Eventually we will reach the 'final marginal', which we can compute as

        .. math::

            |\langle r_0 r_1 r_2 r_3 \ldots | \psi \rangle|^2

        since there is nothing left to trace out.

        Parameters
        ----------
        C : int
            The number of times to sample.
        qubits : None or sequence of int, optional
            Which qubits to measure, defaults (``None``) to all qubits.
        order : None or sequence of int, optional
            Which order to measure the qubits in, defaults (``None``) to an
            order based on greedily expanding the smallest reverse lightcone.
            If specified it should be a permutation of ``qubits``.
        group_size : int, optional
            How many qubits to group together into marginals, the larger this
            is the fewer marginals need to be computed, which can be faster at
            the cost of higher memory. The marginal themselves will each be
            of size ``2**group_size``.
        max_marginal_storage : int, optional
            The total cumulative number of marginal probabilites to cache, once
            this is exceeded caching will be turned off.
        seed : None or int, optional
            A random seed, passed to ``numpy.random.seed`` if given.
        optimize : str, optional
            Contraction path optimizer to use for the marginals, shouldn't be
            a non-reusable path optimizer as called on many different TNs.
            Passed to :func:`cotengra.array_contract_tree`.
        backend : str, optional
            Backend to perform the marginal contraction with, e.g. ``'numpy'``,
            ``'cupy'`` or ``'jax'``. Passed to ``cotengra``.
        dtype : str, optional
            Data type to cast the TN to before contraction.
        simplify_sequence : str, optional
            Which local tensor network simplifications to perform and in which
            order, see
            :meth:`~quimb.tensor.tensor_core.TensorNetwork.full_simplify`.
        simplify_atol : float, optional
            The tolerance with which to compare to zero when applying
            :meth:`~quimb.tensor.tensor_core.TensorNetwork.full_simplify`.
        simplify_equalize_norms : bool, optional
            Actively renormalize tensor norms during simplification.

        Yields
        ------
        bitstrings : sequence of str
        """
        # init TN norms, contraction trees, and marginals
        self._maybe_init_storage()

        rng = np.random.default_rng(seed)

        # which qubits and an ordering e.g. (2, 3, 4, 5), (5, 3, 4, 2)
        qubits, order = self._parse_qubits_order(qubits, order)

        # group the ordering e.g. ((5, 3), (4, 2))
        groups = self._group_order(order, group_size)

        result = dict()
        for _ in range(C):
            for where in groups:
                # key - (tuple[int] where, tuple[tuple[int q, str b])
                # value  - marginal probability distribution of `where` given
                #     prior results, as an ndarray
                # e.g. ((2,), ((0, '0'), (1, '0'))): array([1., 0.]), means
                #     prob(qubit2='0')=1 given qubit0='0' and qubit1='0'
                #     prob(qubit2='1')=0 given qubit0='0' and qubit1='0'
                key = (where, tuple(sorted(result.items())))
                if key not in self._sampled_conditionals:
                    # compute p(qs=x | current bitstring)
                    p = self.compute_marginal(
                        where=where,
                        fix=result,
                        optimize=optimize,
                        backend=backend,
                        dtype=dtype,
                        simplify_sequence=simplify_sequence,
                        simplify_atol=simplify_atol,
                        simplify_equalize_norms=simplify_equalize_norms,
                    )
                    p = do("to_numpy", p).astype("float64")
                    p /= p.sum()

                    if self._marginal_storage_size <= max_marginal_storage:
                        self._sampled_conditionals[key] = p
                        self._marginal_storage_size += p.size
                else:
                    p = self._sampled_conditionals[key]

                # the sampled bitstring e.g. '1' or '001010101'
                b_where = sample_bitstring_from_prob_ndarray(p, seed=rng)

                # split back into individual qubit results
                for q, b in zip(where, b_where):
                    result[q] = b

            yield "".join(result[i] for i in qubits)
            result.clear()

    def sample_rehearse(
        self,
        qubits=None,
        order=None,
        group_size=10,
        result=None,
        optimize="auto-hq",
        simplify_sequence="ADCRS",
        simplify_atol=1e-6,
        simplify_equalize_norms=True,
        rehearse=True,
        progbar=False,
    ):
        """Perform the preparations and contraction tree findings for
        :meth:`~quimb.tensor.circuit.Circuit.sample`, caching various
        intermedidate objects, but don't perform the main contractions.

        Parameters
        ----------
        qubits : None or sequence of int, optional
            Which qubits to measure, defaults (``None``) to all qubits.
        order : None or sequence of int, optional
            Which order to measure the qubits in, defaults (``None``) to an
            order based on greedily expanding the smallest reverse lightcone.
        group_size : int, optional
            How many qubits to group together into marginals, the larger this
            is the fewer marginals need to be computed, which can be faster at
            the cost of higher memory. The marginal's size itself is
            exponential in ``group_size``.
        result : None or dict[int, str], optional
            Explicitly check the computational cost of this result, assumed to
            be all zeros if not given.
        optimize : str, optional
            Contraction path optimizer to use for the marginals, shouldn't be
            a non-reusable path optimizer as called on many different TNs.
            Passed to :func:`cotengra.array_contract_tree`.
        simplify_sequence : str, optional
            Which local tensor network simplifications to perform and in which
            order, see
            :meth:`~quimb.tensor.tensor_core.TensorNetwork.full_simplify`.
        simplify_atol : float, optional
            The tolerance with which to compare to zero when applying
            :meth:`~quimb.tensor.tensor_core.TensorNetwork.full_simplify`.
        simplify_equalize_norms : bool, optional
            Actively renormalize tensor norms during simplification.
        progbar : bool, optional
            Whether to show the progress of finding each contraction tree.

        Returns
        -------
        dict[tuple[int], dict]
            One contraction tree object per grouped marginal computation.
            The keys of the dict are the qubits the marginal is computed for,
            the values are a dict containing a representative simplified tensor
            network (key: 'tn') and the main contraction tree (key: 'tree').
        """
        # init TN norms, contraction trees, and marginals
        self._maybe_init_storage()
        qubits, order = self._parse_qubits_order(qubits, order)
        groups = self._group_order(order, group_size)

        if result is None:
            result = {q: "r" for q in qubits}

        fix = {}
        tns_and_trees = {}

        for where in _progbar(groups, disable=not progbar):
            tns_and_trees[where] = self.compute_marginal(
                where=where,
                fix=fix,
                optimize=optimize,
                simplify_sequence=simplify_sequence,
                simplify_atol=simplify_atol,
                simplify_equalize_norms=simplify_equalize_norms,
                rehearse=rehearse,
            )

            # set the result of qubit ``q`` arbitrarily
            for q in where:
                fix[q] = result[q]

        return tns_and_trees

    sample_tns = functools.partialmethod(sample_rehearse, rehearse="tn")

    def sample_chaotic(
        self,
        C,
        marginal_qubits,
        fix=None,
        max_marginal_storage=2**20,
        seed=None,
        optimize="auto-hq",
        backend=None,
        dtype="complex64",
        simplify_sequence="ADCRS",
        simplify_atol=1e-6,
        simplify_equalize_norms=True,
    ):
        r"""Sample from this circuit, *assuming* it to be chaotic. Which is to
        say, only compute and sample correctly from the final marginal,
        assuming that the distribution on the other qubits is uniform.
        Given ``marginal_qubits=5`` for instance, for each sample a random
        bit-string :math:`r_0 r_1 r_2 \ldots r_{N - 6}` for the remaining
        :math:`N - 5` qubits will be chosen, then the final marginal will be
        computed as

        .. math::

            p(q_{N-5}q_{N-4}q_{N-3}q_{N-2}q_{N-1}
            | r_0 r_1 r_2 \ldots r_{N-6})
            =
            |\langle r_0 r_1 r_2 \ldots r_{N - 6} | \psi \rangle|^2

        and then sampled from. Note the expression on the right hand side has
        5 open indices here and so is a tensor, however if ``marginal_qubits``
        is not too big then the cost of contracting this is very similar to
        a single amplitude.

        .. note::

            This method *assumes* the circuit is chaotic, if its not, then the
            samples produced will not be an accurate representation of the
            probability distribution.

        Parameters
        ----------
        C : int
            The number of times to sample.
        marginal_qubits : int or sequence of int
            The number of qubits to treat as marginal, or the actual qubits. If
            an int is given then the qubits treated as marginal will be
            ``circuit.calc_qubit_ordering()[:marginal_qubits]``.
        fix : None or dict[int, str], optional
            Measurement results on other qubits to fix. These will be randomly
            sampled if ``fix`` is not given or a qubit is missing.
        seed : None or int, optional
            A random seed, passed to ``numpy.random.seed`` if given.
        optimize : str, optional
            Contraction path optimizer to use for the marginal, can be a
            non-reusable path optimizer as only called once (though path won't
            be cached for later use in that case).
        backend : str, optional
            Backend to perform the marginal contraction with, e.g. ``'numpy'``,
            ``'cupy'`` or ``'jax'``. Passed to ``cotengra``.
        dtype : str, optional
            Data type to cast the TN to before contraction.
        simplify_sequence : str, optional
            Which local tensor network simplifications to perform and in which
            order, see
            :meth:`~quimb.tensor.tensor_core.TensorNetwork.full_simplify`.
        simplify_atol : float, optional
            The tolerance with which to compare to zero when applying
            :meth:`~quimb.tensor.tensor_core.TensorNetwork.full_simplify`.
        simplify_equalize_norms : bool, optional
            Actively renormalize tensor norms during simplification.

        Yields
        ------
        str
        """
        # init TN norms, contraction trees, and marginals
        self._maybe_init_storage()
        qubits = tuple(range(self.N))

        rng = np.random.default_rng(seed)

        # choose which qubits to treat as marginal - ideally 'towards one side'
        #     to increase contraction efficiency
        if isinstance(marginal_qubits, numbers.Integral):
            marginal_qubits = self.calc_qubit_ordering()[:marginal_qubits]
        where = tuple(sorted(marginal_qubits))

        # we will uniformly sample, and post-select on, the remaining qubits
        fix_qubits = tuple(q for q in qubits if q not in where)

        result = dict()
        for _ in range(C):
            # generate a random bit-string for the fixed qubits
            for q in fix_qubits:
                if (fix is None) or (q not in fix):
                    result[q] = rng.choice(("0", "1"))
                else:
                    result[q] = fix[q]

            # compute the remaining marginal
            key = (where, tuple(sorted(result.items())))
            if key not in self._sampled_conditionals:
                p = self.compute_marginal(
                    where=where,
                    fix=result,
                    optimize=optimize,
                    backend=backend,
                    dtype=dtype,
                    simplify_sequence=simplify_sequence,
                    simplify_atol=simplify_atol,
                    simplify_equalize_norms=simplify_equalize_norms,
                )
                p = do("to_numpy", p).astype("float64")
                p /= p.sum()

                if self._marginal_storage_size <= max_marginal_storage:
                    self._sampled_conditionals[key] = p
                    self._marginal_storage_size += p.size
            else:
                p = self._sampled_conditionals[key]

            # sample a bit-string for the marginal qubits
            b_where = sample_bitstring_from_prob_ndarray(p)

            # split back into individual qubit results
            for q, b in zip(where, b_where):
                result[q] = b

            yield "".join(result[i] for i in qubits)
            result.clear()

    def sample_chaotic_rehearse(
        self,
        marginal_qubits,
        result=None,
        optimize="auto-hq",
        simplify_sequence="ADCRS",
        simplify_atol=1e-6,
        simplify_equalize_norms=True,
        dtype="complex64",
        rehearse=True,
    ):
        """Rehearse chaotic sampling (perform just the TN simplifications and
        contraction tree finding).

        Parameters
        ----------
        marginal_qubits : int or sequence of int
            The number of qubits to treat as marginal, or the actual qubits. If
            an int is given then the qubits treated as marginal will be
            ``circuit.calc_qubit_ordering()[:marginal_qubits]``.
        result : None or dict[int, str], optional
            Explicitly check the computational cost of this result, assumed to
            be all zeros if not given.
        optimize : str, optional
            Contraction path optimizer to use for the marginal, can be a
            non-reusable path optimizer as only called once (though path won't
            be cached for later use in that case).
        simplify_sequence : str, optional
            Which local tensor network simplifications to perform and in which
            order, see
            :meth:`~quimb.tensor.tensor_core.TensorNetwork.full_simplify`.
        simplify_atol : float, optional
            The tolerance with which to compare to zero when applying
            :meth:`~quimb.tensor.tensor_core.TensorNetwork.full_simplify`.
        simplify_equalize_norms : bool, optional
            Actively renormalize tensor norms during simplification.
        dtype : str, optional
            Data type to cast the TN to before contraction.

        Returns
        -------
        dict[tuple[int], dict]
            The contraction path information for the main computation, the key
            is the qubits that formed the final marginal. The value is itself a
            dict with keys ``'tn'`` - a representative tensor network - and
            ``'tree'`` - the contraction tree.
        """

        # init TN norms, contraction trees, and marginals
        self._maybe_init_storage()
        qubits = tuple(range(self.N))

        if isinstance(marginal_qubits, numbers.Integral):
            marginal_qubits = self.calc_qubit_ordering()[:marginal_qubits]
        where = tuple(sorted(marginal_qubits))

        fix_qubits = tuple(q for q in qubits if q not in where)

        if result is None:
            fix = {q: "0" for q in fix_qubits}
        else:
            fix = {q: result[q] for q in fix_qubits}

        rehs = self.compute_marginal(
            where=where,
            fix=fix,
            optimize=optimize,
            simplify_sequence=simplify_sequence,
            simplify_atol=simplify_atol,
            simplify_equalize_norms=simplify_equalize_norms,
            dtype=dtype,
            rehearse=rehearse,
        )

        if rehearse == "tn":
            return rehs

        return {where: rehs}

    sample_chaotic_tn = functools.partialmethod(
        sample_chaotic_rehearse, rehearse="tn"
    )

    def get_gate_by_gate_circuits(self, group_size=10):
        """Get a sequence of circuits by partitioning the gates into groups
        such circuit `i + 1` acts on at most ``group_size`` new qubits compared
        to circuit `i`.

        Parameters
        ----------
        group_size : int, optional
            The maximum number of new qubits that can be acted on by a circuit
            compared to its predecessor.

        Returns
        -------
        Sequence[dict]
            A sequence of dicts, each with keys ``'circuit'`` and ``'where'``,
            where the former is a :class:`~quimb.tensor.circuit.Circuit` and
            the latter the tuple of new qubits that it acts on comparaed to
            the previous circuit.
        """
        circs = [self.__class__(self.N)]
        groups = []
        current_group = set()

        # this ensures that single qubit gates are always adjacent to
        # multi-qubit gates and will thus always be included in the same group
        gates = self.reordered_gates_dfs_clustered()

        for gate in gates:
            # if we were to add next gate, how many new qubits would we have?
            next_group = current_group.union(gate.qubits)
            if len(next_group) > group_size:
                # over the limit: flush a copy of the current circuit and group
                groups.append(tuple(sorted(current_group)))
                circs.append(circs[-1].copy())
                # start a new group
                current_group = set(gate.qubits)
            else:
                # add the gate to the current group
                current_group = next_group
            circs[-1].apply_gate(gate)

        # add the final group corresponding to circs[-1]
        groups.append(tuple(sorted(current_group)))

        return tuple({"circuit": c, "where": g} for c, g in zip(circs, groups))

    def sample_gate_by_gate(
        self,
        C,
        group_size=10,
        seed=None,
        max_marginal_storage=2**20,
        optimize="auto-hq",
        backend=None,
        dtype="complex64",
        simplify_sequence="ADCRS",
        simplify_atol=1e-6,
        simplify_equalize_norms=True,
    ):
        """Sample this circuit using the gate-by-gate method, where we 'evolve'
        a result bitstring by sequentially including more and more gates, at
        each step updating the result by computing a full conditional marginal.
        See "How to simulate quantum measurement without computing marginals"
        by Sergey Bravyi, David Gosset, Yinchen Liu
        (https://arxiv.org/abs/2112.08499). The overall complexity of this is
        guaranteed to be similar to that of computing a single amplitude which
        can be much better than the naive "qubit-by-qubit" (`.sample`) method.
        However, it requires evaluting a number of tensor networks that scales
        linearly with the number of gates which can offset any practical
        advantages for shallow circuits for example.

        Parameters
        ----------
        C : int
            The number of samples to generate.
        group_size : int, optional
            The maximum number of qubits that can be acted on by a circuit
            compared to its predecessor. This will be the dimension of the
            marginal computed at each step.
        seed : None or int, optional
            A random seed, passed to ``numpy.random.seed`` if given.
        max_marginal_storage : int, optional
            The total cumulative number of marginal probabilites to cache, once
            this is exceeded caching will be turned off.
        optimize : str, optional
            Contraction path optimizer to use for the marginals, shouldn't be
            a non-reusable path optimizer as called on many different TNs.
            Passed to :func:`cotengra.array_contract_tree`.
        backend : str, optional
            Backend to perform the marginal contraction with, e.g. ``'numpy'``,
            ``'cupy'`` or ``'jax'``. Passed to ``cotengra``.
        dtype : str, optional
            Data type to cast the TN to before contraction.
        simplify_sequence : str, optional
            Which local tensor network simplifications to perform and in which
            order, see
            :meth:`~quimb.tensor.tensor_core.TensorNetwork.full_simplify`.
        simplify_atol : float, optional
            The tolerance with which to compare to zero when applying
            :meth:`~quimb.tensor.tensor_core.TensorNetwork.full_simplify`.
        simplify_equalize_norms : bool, optional
            Actively renormalize tensor norms during simplification.
        rehearse : bool, optional
            If ``True``, generate and cache the simplified tensor network and
            contraction tree but don't actually perform the contraction.
            Returns a dict with keys ``'tn'`` and ``'tree'`` with the tensor
            network that will be contracted and the corresponding contraction
            tree if so.

        Yields
        ------
        str
        """
        self._maybe_init_storage()

        rng = np.random.default_rng(seed)

        key = ("gate_by_gate_circuits", group_size)
        try:
            circs_wheres = self._storage[key]
        except KeyError:
            circs_wheres = self.get_gate_by_gate_circuits(group_size)
            self._storage[key] = circs_wheres

        for _ in range(C):
            # start with all qubits in the |0> state
            result = {q: "0" for q in range(self.N)}

            for circ_where in circs_wheres:
                # get the next circuit and the new group of qubits
                circ_g = circ_where["circuit"]
                where = circ_where["where"]

                # remove the new group of qubits from our current result
                for q in where:
                    result.pop(q)

                # check if we have already computed the conditional
                key = (where, tuple(sorted(result.items())))

                if key not in circ_g._sampled_conditionals:
                    p = circ_g.compute_marginal(
                        where,
                        fix=result,
                        optimize=optimize,
                        backend=backend,
                        dtype=dtype,
                        simplify_sequence=simplify_sequence,
                        simplify_atol=simplify_atol,
                        simplify_equalize_norms=simplify_equalize_norms,
                    )
                    p /= p.sum()

                    if circ_g._marginal_storage_size <= max_marginal_storage:
                        circ_g._sampled_conditionals[key] = p
                        circ_g._marginal_storage_size += p.size
                else:
                    p = circ_g._sampled_conditionals[key]

                # sample a configuration for our new group
                b_where = sample_bitstring_from_prob_ndarray(p, seed=rng)

                # update the fixed qubits given new group result
                for q, qx in zip(where, b_where):
                    result[q] = qx

            yield "".join(result[i] for i in range(self.N))

    def sample_gate_by_gate_rehearse(
        self,
        group_size=10,
        optimize="auto-hq",
        dtype="complex64",
        simplify_sequence="ADCRS",
        simplify_atol=1e-6,
        simplify_equalize_norms=True,
        rehearse=True,
        progbar=False,
    ):
        """Perform the preparations and contraction tree findings for
        :meth:`~quimb.tensor.circuit.Circuit.sample_gate_by_gate`, caching
        various intermedidate objects, but don't perform the main contractions.

        Parameters
        ----------
        group_size : int, optional
            The maximum number of qubits that can be acted on by a circuit
            compared to its predecessor. This will be the dimension of the
            marginal computed at each step.
        optimize : str, optional
            Contraction path optimizer to use for the marginals, shouldn't be
            a non-reusable path optimizer as called on many different TNs.
            Passed to :func:`cotengra.array_contract_tree`.
        dtype : str, optional
            Data type to cast the TN to before contraction.
        simplify_sequence : str, optional
            Which local tensor network simplifications to perform and in which
            order, see
            :meth:`~quimb.tensor.tensor_core.TensorNetwork.full_simplify`.
        simplify_atol : float, optional
            The tolerance with which to compare to zero when applying
            :meth:`~quimb.tensor.tensor_core.TensorNetwork.full_simplify`.
        simplify_equalize_norms : bool, optional
            Actively renormalize tensor norms during simplification.
        rehearse : True or "tn", optional
            If ``True``, generate and cache the simplified tensor network and
            contraction tree but don't actually perform the contraction. If
            "tn", only generate the simplified tensor networks.

        Returns
        -------
        Sequence[dict] or Sequence[TensorNetwork]
        """
        self._maybe_init_storage()

        key = ("gate_by_gate_circuits", group_size)
        try:
            circs_wheres = self._storage[key]
        except KeyError:
            circs_wheres = self.get_gate_by_gate_circuits(group_size)
            self._storage[key] = circs_wheres

        rehs = []
        result = {q: "0" for q in range(self.N)}

        for circs_wheres in _progbar(circs_wheres, disable=not progbar):
            # get the next circuit and the new group of qubits
            circ_g = circs_wheres["circuit"]
            where = circs_wheres["where"]

            # remove the new group of qubits from our current result
            for q in where:
                result.pop(q)

            r = circ_g.compute_marginal(
                where,
                fix=result,
                optimize=optimize,
                dtype=dtype,
                simplify_sequence=simplify_sequence,
                simplify_atol=simplify_atol,
                simplify_equalize_norms=simplify_equalize_norms,
                rehearse=rehearse,
            )

            if rehearse != "tn":
                r["where"] = where
                r["circuit"] = circ_g

            rehs.append(r)

            # update the fixed qubits with randomly rotated results so we
            # don't get zero probability networks when simplifying
            for q in where:
                result[q] = "r"

        return rehs

    sample_gate_by_gate_tns = functools.partialmethod(
        sample_gate_by_gate_rehearse, rehearse="tn"
    )

    def to_dense(
        self,
        reverse=False,
        optimize="auto-hq",
        simplify_sequence="R",
        simplify_atol=1e-12,
        simplify_equalize_norms=True,
        backend=None,
        dtype=None,
        rehearse=False,
    ):
        """Generate the dense representation of the final wavefunction.

        Parameters
        ----------
        reverse : bool, optional
            Whether to reverse the order of the subsystems, to match the
            convention of qiskit for example.
        optimize : str, optional
            Contraction path optimizer to use for the contraction, can be a
            non-reusable path optimizer as only called once (though path won't
            be cached for later use in that case).
        dtype : dtype or str, optional
            If given, convert the tensors to this dtype prior to contraction.
        simplify_sequence : str, optional
            Which local tensor network simplifications to perform and in which
            order, see
            :meth:`~quimb.tensor.tensor_core.TensorNetwork.full_simplify`.
        simplify_atol : float, optional
            The tolerance with which to compare to zero when applying
            :meth:`~quimb.tensor.tensor_core.TensorNetwork.full_simplify`.
        simplify_equalize_norms : bool, optional
            Actively renormalize tensor norms during simplification.
        backend : str, optional
            Backend to perform the contraction with, e.g. ``'numpy'``,
            ``'cupy'`` or ``'jax'``. Passed to ``cotengra``.
        dtype : str, optional
            Data type to cast the TN to before contraction.
        rehearse : bool, optional
            If ``True``, generate and cache the simplified tensor network and
            contraction tree but don't actually perform the contraction.
            Returns a dict with keys ``'tn'`` and ``'tree'`` with the tensor
            network that will be contracted and the corresponding contraction
            tree if so.

        Returns
        -------
        psi : qarray
            The densely represented wavefunction with ``dtype`` data.
        """
        psi = self.get_psi_simplified(
            seq=simplify_sequence,
            atol=simplify_atol,
            equalize_norms=simplify_equalize_norms,
        )
        self._maybe_convert(psi, dtype)

        if rehearse == "tn":
            return psi

        output_inds = tuple(map(psi.site_ind, range(self.N)))
        if reverse:
            output_inds = output_inds[::-1]

        tree = psi.contraction_tree(output_inds=output_inds, optimize=optimize)

        if rehearse:
            return rehearsal_dict(psi, tree)

        # perform the full contraction with the path found
        psi_tensor = psi.contract(
            all,
            output_inds=output_inds,
            optimize=tree,
            backend=backend,
        ).data

        k = ops.reshape(psi_tensor, (-1, 1))

        if isinstance(k, np.ndarray):
            k = qu.qarray(k)

        return k

    to_dense_rehearse = functools.partialmethod(to_dense, rehearse=True)
    to_dense_tn = functools.partialmethod(to_dense, rehearse="tn")

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

    def schrodinger_contract(self, *args, **contract_opts):
        ntensor = self._psi.num_tensors
        path = [(0, 1)] + [(0, i) for i in reversed(range(1, ntensor - 1))]
        return self.psi.contract(*args, optimize=path, **contract_opts)

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

    def xeb_ex(
        self,
        optimize="auto-hq",
        simplify_sequence="R",
        simplify_atol=1e-12,
        simplify_equalize_norms=True,
        dtype=None,
        backend=None,
        autojit=False,
        progbar=False,
        **contract_opts,
    ):
        """Compute the exactly expected XEB for this circuit. The main feature
        here is that if you supply a cotengra optimizer that searches for
        sliced indices then the XEB will be computed without constructing the
        full wavefunction.

        Parameters
        ----------
        optimize : str or PathOptimizer, optional
            Contraction path optimizer.
        simplify_sequence : str, optional
            Simplifications to apply to tensor network prior to contraction.
        simplify_sequence : str, optional
            Which local tensor network simplifications to perform and in which
            order, see
            :meth:`~quimb.tensor.tensor_core.TensorNetwork.full_simplify`.
        simplify_atol : float, optional
            The tolerance with which to compare to zero when applying
            :meth:`~quimb.tensor.tensor_core.TensorNetwork.full_simplify`.
        dtype : str, optional
            Data type to cast the TN to before contraction.
        backend : str, optional
            Convert tensors to, and then use contractions from, this library.
        autojit : bool, optional
            Apply ``autoray.autojit`` to the contraciton and map-reduce.
        progbar : bool, optional
            Show progress in terms of number of wavefunction chunks processed.
        """
        # get potentially simplified TN of full wavefunction
        psi = self.to_dense_tn(
            simplify_sequence=simplify_sequence,
            simplify_atol=simplify_atol,
            simplify_equalize_norms=simplify_equalize_norms,
            dtype=dtype,
        )

        # find a possibly sliced contraction tree
        output_inds = tuple(map(psi.site_ind, range(self.N)))
        tree = psi.contraction_tree(optimize=optimize, output_inds=output_inds)

        arrays = psi.arrays
        if backend is not None:
            arrays = [do("array", x, like=backend) for x in arrays]

        # perform map-reduce style computation over output wavefunction chunks
        # so we don't need entire wavefunction in memory at same time
        chunks = tree.gen_output_chunks(
            arrays, autojit=autojit, **contract_opts
        )
        if progbar:
            chunks = _progbar(chunks, total=tree.nchunks)

        def f(chunk):
            return do("sum", do("abs", chunk) ** 4)

        if autojit:
            # since we convert the arrays above, the jit backend is
            # automatically inferred
            from autoray import autojit

            f = autojit(f)

        p2sum = functools.reduce(operator.add, map(f, chunks))
        return 2**self.N * p2sum - 1

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


class CircuitDense(Circuit):
    """Quantum circuit simulation keeping the state in full dense form."""

    def __init__(
        self,
        N=None,
        psi0=None,
        gate_opts=None,
        gate_contract=True,
        tags=None,
        convert_eager=True,
        **circuit_opts,
    ):
        gate_opts = ensure_dict(gate_opts)
        gate_opts.setdefault("contract", gate_contract)
        gate_opts.setdefault("convert_eager", convert_eager)
        super().__init__(N, psi0, gate_opts, tags, **circuit_opts)

    @property
    def psi(self):
        t = self._psi ^ ...
        psi = t.as_network()
        psi.view_as_(Dense1D, like=self._psi, L=self.N)
        return psi

    @property
    def uni(self):
        raise ValueError(
            "You can't extract the circuit unitary TN from a ``CircuitDense``."
        )

    def calc_qubit_ordering(self, qubits=None):
        """Qubit ordering doesn't matter for a dense wavefunction."""
        if qubits is None:
            return tuple(range(self.N))
        else:
            return tuple(sorted(qubits))

    def get_psi_reverse_lightcone(self, where, keep_psi0=False):
        """Override ``get_psi_reverse_lightcone`` as for a dense wavefunction
        the lightcone is not meaningful.
        """
        return self.psi
