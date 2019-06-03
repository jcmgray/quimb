import math
import numbers

import quimb as qu
from .tensor_gen import MPS_computational_state


def parse_qasm(qasm):
    """Parse qasm from a string.

    Parameters
    ----------
    qasm : str
        The full string of the qasm file.

    Returns
    -------
    circuit_info : dict
        Information about the circuit:

        - circuit_info['n']: the number of qubits
        - circuit_info['n_gates']: the number of gates in total
        - circuit_info['gates']: list[list[str]], list of gates, each of which
          is a list of strings read from a line of the qasm file.
    """
    lns = qasm.split('\n')
    n = int(lns[0])
    gates = [l.split(" ") for l in lns[1:] if l]
    return {'n': n, 'gates': gates, 'n_gates': len(gates)}


def parse_qasm_file(fname, **kwargs):
    """Parse a qasm file.
    """
    return parse_qasm(open(fname).read(), **kwargs)


def parse_qasm_url(url, **kwargs):
    """Parse a qasm url.
    """
    from urllib import request
    return parse_qasm(request.urlopen(url).read().decode(), **kwargs)


# -------------------------- core gate functions ---------------------------- #

def _merge_tags(tags, gate_opts):
    if isinstance(tags, str):
        tags = {tags}
    else:
        tags = set(tags)
    return tags | set(gate_opts.pop('tags', ()))


def build_gate_1(gate, tags=None):
    """Build a function that applies ``gate`` to a tensor network wavefunction.
    """

    def apply_gate(psi, i, **gate_opts):
        mtags = _merge_tags(tags, gate_opts)
        psi.gate_(gate, int(i), tags=mtags, **gate_opts)

    return apply_gate


def build_gate_2(gate, tags=None):
    """Build a function that applies ``gate`` to a tensor network wavefunction.
    """

    def apply_gate(psi, i, j, **gate_opts):
        mtags = _merge_tags(tags, gate_opts)
        psi.gate_(gate, (int(i), int(j)), tags=mtags, **gate_opts)

    return apply_gate


# rotations take an angle and so don't fit into ``build_gate_1``

def apply_Rx(psi, theta, i, **gate_opts):
    """Apply an X-rotation of ``theta`` to tensor network wavefunction ``psi``.
    """
    mtags = _merge_tags({'RX'}, gate_opts)
    psi.gate_(qu.Rx(float(theta)), int(i), tags=mtags, **gate_opts)


def apply_Ry(psi, theta, i, **gate_opts):
    """Apply a Y-rotation of ``theta`` to tensor network wavefunction ``psi``.
    """
    mtags = _merge_tags({'RY'}, gate_opts)
    psi.gate_(qu.Ry(float(theta)), int(i), tags=mtags, **gate_opts)


def apply_Rz(psi, theta, i, **gate_opts):
    """Apply a Z-rotation of ``theta`` to tensor network wavefunction ``psi``.
    """
    mtags = _merge_tags({'RZ'}, gate_opts)
    psi.gate_(qu.Rz(float(theta)), int(i), tags=mtags, **gate_opts)


def apply_U3(psi, theta, phi, lamda, i, **gate_opts):
    mtags = _merge_tags({'U3'}, gate_opts)
    psi.gate_(qu.U_gate(theta, phi, lamda), int(i), tags=mtags, **gate_opts)


def apply_swap(psi, i, j, **gate_opts):
    itag, jtag = map(psi.site_tag, (i, j))
    psi.reindex_({itag: jtag, jtag: itag})


APPLY_GATES = {
    'RX': apply_Rx,
    'RY': apply_Ry,
    'RZ': apply_Rz,
    'U3': apply_U3,
    'H': build_gate_1(qu.hadamard(), tags='H'),
    'X': build_gate_1(qu.pauli('X'), tags='X'),
    'Y': build_gate_1(qu.pauli('Y'), tags='Y'),
    'Z': build_gate_1(qu.pauli('Z'), tags='Z'),
    'S': build_gate_1(qu.S_gate(), tags='S'),
    'T': build_gate_1(qu.T_gate(), tags='T'),
    'X_1_2': build_gate_1(qu.Rx(math.pi / 2), tags='X_1/2'),
    'Y_1_2': build_gate_1(qu.Ry(math.pi / 2), tags='Y_1/2'),
    'Z_1_2': build_gate_1(qu.Rz(math.pi / 2), tags='Z_1/2'),
    'IDEN': lambda *args, **kwargs: None,
    'CX': build_gate_2(qu.cX(), tags='CX'),
    'CY': build_gate_2(qu.cY(), tags='CY'),
    'CZ': build_gate_2(qu.cZ(), tags='CZ'),
    'CNOT': build_gate_2(qu.CNOT(), tags='CNOT'),
    'SWAP': apply_swap,
}


# --------------------------- main circuit class ---------------------------- #

class Circuit:
    """Class for simulating quantum circuits using tensor networks.

    Parameters
    ----------
    N : int, optional
        The number of qubits.
    psi0 : TensorNetwork1DVector, optional
        The initial state, assumed to be ``|00000....0>`` if not given.
    gate_opts : dict_like, optional
        Default keyword arguments to supply to each
        :func:`~quimb.tensor.tensor_1d.gate_TN_1D` call during the circuit.
    tags : str or sequence of str, optional
        Tag(s) to add to the initial wavefunction tensors (whether these are
        propagated to the rest of the circuit's tensors depends on
        ``gate_opts``).

    Attributes
    ----------
    psi : TensorNetwork1DVector
        The current wavefunction.

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
        >>> qc.apply_circuit(gates)
        >>> qc.psi
        <MatrixProductState(tensors=10, structure='I{}', nsites=3)>

        >>> qc.psi.to_dense().round(4)
        qarray([[ 0.7071+0.j],
                [ 0.    +0.j],
                [ 0.    +0.j],
                [-0.    +0.j],
                [-0.    +0.j],
                [ 0.    +0.j],
                [ 0.    +0.j],
                [ 0.7071+0.j]])
    """

    def __init__(self, N=None, psi0=None, gate_opts=None, tags=None):

        if N is None and psi0 is None:
            raise ValueError("You must supply one of `N` or `psi0`.")

        elif psi0 is None:
            self.N = N
            self._psi = MPS_computational_state('0' * N)

        elif N is None:
            self._psi = psi0.copy()
            self.N = psi0.nsites

        else:
            if N != psi0.nsites:
                raise ValueError("`N` doesn't match `psi0`.")
            self.N = N
            self._psi = psi0.copy()

        if tags is not None:
            if isinstance(tags, str):
                tags = (tags,)
            for tag in tags:
                self._psi.add_tag(tag)

        self.gate_opts = {} if gate_opts is None else dict(gate_opts)
        self.gate_opts.setdefault('contract', 'split-gate')
        self.gate_opts.setdefault('propagate_tags', 'register')
        self.gates = []

    @classmethod
    def from_qasm(cls, qasm, **quantum_circuit_opts):
        """Generate a ``Circuit`` instance from a qasm string.
        """
        info = parse_qasm(qasm)
        qc = cls(info['n'], **quantum_circuit_opts)
        qc.apply_circuit(info['gates'])
        return qc

    @classmethod
    def from_qasm_file(cls, fname, **quantum_circuit_opts):
        """Generate a ``Circuit`` instance from a qasm file.
        """
        info = parse_qasm_file(fname)
        qc = cls(info['n'], **quantum_circuit_opts)
        qc.apply_circuit(info['gates'])
        return qc

    @classmethod
    def from_qasm_url(cls, url, **quantum_circuit_opts):
        """Generate a ``Circuit`` instance from a qasm url.
        """
        info = parse_qasm_url(url)
        qc = cls(info['n'], **quantum_circuit_opts)
        qc.apply_circuit(info['gates'])
        return qc

    def apply_gate(self, gate_id, *gate_args, gate_round=None):
        """Apply a single gate to this tensor network quantum circuit. If
        ``gate_round`` is supplied the tensor(s) added will be tagged with
        ``'ROUND_{gate_round}'``. Alternatively, putting an integer first like
        so::

            circuit.apply_gate(10, 'H', 7)

        Is automatically translated to::

            circuit.apply_gate('H', 7, gate_round=10)

        Parameters
        ----------
        gate_id : str
            Which type of gate to apply.
        gate_args : list[str]
            The argument to supply to it.
        gate_round : int, optional
            The gate round. If ``gate_id`` is integer-like, will also be taken
            from here, with then ``gate_id, gate_args = gate_args[0],
            gate_args[1:]``.
        """

        if (gate_round is not None):
            tags = {'ROUND_{}'.format(gate_round)}
        elif isinstance(gate_id, numbers.Integral) or gate_id.isdigit():
            # gate round given as first entry of qasm line
            tags = {'ROUND_{}'.format(gate_id)}
            gate_id, gate_args = gate_args[0], gate_args[1:]
        else:
            tags = set()

        apply_fn = APPLY_GATES[gate_id.upper()]
        apply_fn(self._psi, *gate_args, tags=tags, **self.gate_opts)
        self.gates.append((gate_id, *gate_args))

    def apply_circuit(self, gates):
        """Apply a sequence of gates to this tensor network quantum circuit.

        Parameters
        ----------
        gates : list[list[str]]
            The sequence of gates to apply.
        """
        for gate in gates:
            self.apply_gate(*gate)

        self._psi.squeeze_()

    def x(self, i, gate_round=None):
        self.apply_gate('X', i, gate_round=gate_round)

    def y(self, i, gate_round=None):
        self.apply_gate('Y', i, gate_round=gate_round)

    def z(self, i, gate_round=None):
        self.apply_gate('Z', i, gate_round=gate_round)

    def h(self, i, gate_round=None):
        self.apply_gate('H', i, gate_round=gate_round)

    def s(self, i, gate_round=None):
        self.apply_gate('S', i, gate_round=gate_round)

    def t(self, i, gate_round=None):
        self.apply_gate('T', i, gate_round=gate_round)

    def iden(self, i, gate_round=None):
        pass

    def rx(self, theta, i, gate_round=None):
        self.apply_gate('RX', theta, i, gate_round=gate_round)

    def ry(self, theta, i, gate_round=None):
        self.apply_gate('RY', theta, i, gate_round=gate_round)

    def rz(self, theta, i, gate_round=None):
        self.apply_gate('RZ', theta, i, gate_round=gate_round)

    def u3(self, theta, phi, lamda, i, gate_round=None):
        self.apply_gate('U3', theta, phi, lamda, i, gate_round=gate_round)

    def cx(self, i, j, gate_round=None):
        self.apply_gate('CX', i, j, gate_round=gate_round)

    def cy(self, i, j, gate_round=None):
        self.apply_gate('CY', i, j, gate_round=gate_round)

    def cz(self, i, j, gate_round=None):
        self.apply_gate('CZ', i, j, gate_round=gate_round)

    def cnot(self, i, j, gate_round=None):
        self.apply_gate('CNOT', i, j, gate_round=gate_round)

    def swap(self, i, j, gate_round=None):
        self.apply_gate('SWAP', i, j)

    @property
    def psi(self):
        """Tensor network representation of the wavefunction.
        """
        # make sure all same dtype and drop singlet dimensions
        psi = self._psi.squeeze()
        psi.astype_(psi.dtype)
        return psi

    def to_dense(self, reverse=False, dtype=None,
                 rank_simplify=False, **contract_opts):
        """Generate the dense representation of the final wavefunction.

        Parameters
        ----------
        reverse : bool, optional
            Whether to reverse the order of the subsystems, to match the
            convention of qiskit for example.
        contract_opts
            Suppled to :func:`~quimb.tensor.tensor_core.tensor_contract`.
        dtype : dtype or str, optional
            If given, convert the tensors to this dtype prior to contraction.
        rank_simplify : bool, optional
            If the network is complex, performing rank simplification first can
            aid the contraction path finding.

        Returns
        -------
        psi : qarray
            The densely represented wavefunction with ``dtype`` data.
        """
        psi = self.psi

        if dtype is not None:
            psi.astype_(dtype)

        if rank_simplify:
            psi.rank_simplify_()

        inds = [psi.site_ind(i) for i in range(self.N)]

        if reverse:
            inds = inds[::-1]

        p_dense = psi.to_dense(inds, tags=all, **contract_opts)
        return p_dense

    def simulate_counts(self, C, seed=None, reverse=False, **contract_opts):
        """Simulate measuring each qubit in the computational basis. See
        :func:`~quimb.calc.simulate_counts`.

        .. warning::

            This currently constructs the full wavefunction in order to sample
            the probabilities accurately.

        Parameters
        ----------
        C : int
            The number of 'experimental runs', i.e. total counts.
        seed : int, optional
            A seed for reproducibility.
        reverse : bool, optional
            Whether to reverse the order of the subsystems, to match the
            convention of qiskit for example.
        contract_opts
            Suppled to :func:`~quimb.tensor.tensor_core.tensor_contract`.

        Returns
        -------
        results : dict[str, int]
            The number of recorded counts for each
        """
        p_dense = self.to_dense(reverse=reverse, **contract_opts)
        return qu.simulate_counts(p_dense, C=C, seed=seed)

    def __repr__(self):
        r = "<Circuit(n={}, n_gates={}, gate_opts={})>"
        return r.format(self.N, len(self.gates), self.gate_opts)


class CircuitMPS(Circuit):
    """Quantum circuit simulation keeping the state always in a MPS form. If
    you think the circuit will not build up much entanglement, or you just want
    to keep a rigorous handle on how much entanglement is present, this can
    be useful.
    """

    def __init__(self, N=None, psi0=None, gate_opts=None, tags=None):
        gate_opts = {} if gate_opts is None else dict(gate_opts)
        gate_opts.setdefault('contract', 'swap+split')
        super().__init__(N, psi0, gate_opts, tags)

    @property
    def psi(self):
        # no squeeze so that bond dims of 1 preserved
        return self._psi


class CircuitDense(Circuit):
    """Quantum circuit simulation keeping the state in full dense form.
    """

    def __init__(self, N=None, psi0=None, gate_opts=None, tags=None):
        gate_opts = {} if gate_opts is None else dict(gate_opts)
        gate_opts.setdefault('contract', True)
        super().__init__(N, psi0, gate_opts, tags)

    @property
    def psi(self):
        return self._psi ^ all
