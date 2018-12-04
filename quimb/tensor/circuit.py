import math
import quimb as qu
from .tensor_gen import MPS_computational_state


def parse_qasm(qasm, strip_round=False):
    """Parse qasm from a string.

    Parameters
    ----------
    qasm : str
        The full string of the qasm file.
    strip_round : bool, optional
        If true, remove the first entry of each line,
        assuming it to the the gate round.

    Returns
    -------
    circuit_info : dict
        Information abou the circuit:

        - circuit_info['n']: the number of qubits
        - circuit_info['n_gates']: the number of gates in total
        - circuit_info['gates']: list[list[str]], list of gates, each of which
          is a list of strings read from a line of the qasm file.
    """
    lns = qasm.split('\n')
    n = int(lns[0])
    gates = [l.split(" ") for l in lns[1:] if l]

    if strip_round:
        gates = [g[1:] for g in gates]

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

def build_gate_1(gate, tags=None):
    """Build a function that applies ``gate`` to a tensor network wavefunction.
    """

    def apply_gate(psi, i, **gate_opts):
        psi.gate_(gate, int(i), tags=tags, **gate_opts)

    return apply_gate


def build_gate_2(gate, tags=None):
    """Build a function that applies ``gate`` to a tensor network wavefunction.
    """

    def apply_gate(psi, i, j, **gate_opts):
        psi.gate_(gate, (int(i), int(j)), tags=tags, **gate_opts)

    return apply_gate


# rotations take an angle and so don't fit into ``build_gate_1``

def apply_Rx(psi, theta, i, **gate_opts):
    """Apply an X-rotation of ``theta`` to tensor network wavefunction ``psi``.
    """
    psi.gate_(qu.Rx(float(theta)), int(i), tags='RX', **gate_opts)


def apply_Ry(psi, theta, i, **gate_opts):
    """Apply a Y-rotation of ``theta`` to tensor network wavefunction ``psi``.
    """
    psi.gate_(qu.Ry(float(theta)), int(i), tags='RY', **gate_opts)


def apply_Rz(psi, theta, i, **gate_opts):
    """Apply a Z-rotation of ``theta`` to tensor network wavefunction ``psi``.
    """
    psi.gate_(qu.Rz(float(theta)), int(i), tags='RZ', **gate_opts)


APPLY_GATES = {
    'RZ': apply_Rx,
    'RY': apply_Ry,
    'RX': apply_Rz,
    'H': build_gate_1(qu.hadamard(), tags='H'),
    'X': build_gate_1(qu.pauli('X'), tags='X'),
    'Y': build_gate_1(qu.pauli('Y'), tags='Y'),
    'Z': build_gate_1(qu.pauli('Z'), tags='Z'),
    'T': build_gate_1(qu.T_gate(), tags='T'),
    'X_1_2': build_gate_1(qu.Rx(math.pi / 2), tags='X_1/2'),
    'Y_1_2': build_gate_1(qu.Ry(math.pi / 2), tags='Y_1/2'),
    'CX': build_gate_2(qu.cX(), tags='CX'),
    'CY': build_gate_2(qu.cY(), tags='CY'),
    'CZ': build_gate_2(qu.cZ(), tags='CZ'),
    'CNOT': build_gate_2(qu.CNOT(), tags='CNOT'),
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
        self.gates = []

    @classmethod
    def from_qasm(cls, qasm, strip_round=False, **quantum_circuit_opts):
        """Generate a ``Circuit`` instance from a qasm string.
        """
        info = parse_qasm(qasm, strip_round=strip_round)
        qc = Circuit(info['n'], **quantum_circuit_opts)
        qc.apply_circuit(info['gates'])
        return qc

    @classmethod
    def from_qasm_file(cls, fname, strip_round=False, **quantum_circuit_opts):
        """Generate a ``Circuit`` instance from a qasm file.
        """
        info = parse_qasm_file(fname, strip_round=strip_round)
        qc = Circuit(info['n'], **quantum_circuit_opts)
        qc.apply_circuit(info['gates'])
        return qc

    @classmethod
    def from_qasm_url(cls, url, strip_round=False, **quantum_circuit_opts):
        """Generate a ``Circuit`` instance from a qasm url.
        """
        info = parse_qasm_url(url, strip_round=strip_round)
        qc = Circuit(info['n'], **quantum_circuit_opts)
        qc.apply_circuit(info['gates'])
        return qc

    def apply_gate(self, gate_id, *gate_args):
        """Apply a single gate to this tensor network quantum circuit.

        Parameters
        ----------
        gate_id : str
            Which type of gate to apply.
        gate_args : list[str]
            The argument to supply to it.
        """
        apply_fn = APPLY_GATES[gate_id.upper()]
        apply_fn(self._psi, *gate_args, **self.gate_opts)
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

    @property
    def psi(self):
        return self._psi.squeeze()
