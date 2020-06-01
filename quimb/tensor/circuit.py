import numbers

import cytoolz
from autoray import do

import quimb as qu
from .tensor_core import get_tags, PTensor, oset, tags_to_oset
from .tensor_gen import MPS_computational_state
from .tensor_1d import TensorNetwork1DVector
from . import array_ops as ops


def _convert_ints_and_floats(x):
    if isinstance(x, str):
        try:
            return int(x)
        except ValueError:
            pass

        try:
            return float(x)
        except ValueError:
            pass

    return x


def _put_registers_last(x):
    # no need to do anything unless parameter (i.e. float) is found last
    if not isinstance(x[-1], float):
        return x

    # swap this last group of floats with the penultimate group of integers
    parts = tuple(cytoolz.partitionby(type, x))
    return tuple(cytoolz.concatv(*parts[:-2], parts[-1], parts[-2]))


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

    lines = qasm.split('\n')
    n = int(lines[0])

    # turn into tuples of python types
    gates = [
        tuple(map(_convert_ints_and_floats, line.strip().split(" ")))
        for line in lines[1:] if line
    ]

    # put registers/parameters in standard order and detect if gate round used
    gates = tuple(map(_put_registers_last, gates))
    round_specified = isinstance(gates[0][0], numbers.Integral)

    return {
        'n': n,
        'gates': gates,
        'n_gates': len(gates),
        'round_specified': round_specified,
    }


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
    return oset.union(*map(tags_to_oset, (tags, gate_opts.pop('tags', None))))


def build_gate_1(gate, tags=None):
    """Build a function that applies ``gate`` to a tensor network wavefunction.
    """

    def apply_constant_single_qubit_gate(psi, i, **gate_opts):
        mtags = _merge_tags(tags, gate_opts)
        psi.gate_(gate, int(i), tags=mtags, **gate_opts)

    return apply_constant_single_qubit_gate


def build_gate_2(gate, tags=None):
    """Build a function that applies ``gate`` to a tensor network wavefunction.
    """

    def apply_constant_two_qubit_gate(psi, i, j, **gate_opts):
        mtags = _merge_tags(tags, gate_opts)
        psi.gate_(gate, (int(i), int(j)), tags=mtags, **gate_opts)

    return apply_constant_two_qubit_gate


# non tensor gates

def apply_swap(psi, i, j, **gate_opts):
    iind, jind = map(psi.site_ind, (int(i), int(j)))
    psi.reindex_({iind: jind, jind: iind})


# parametrizable gates

def rx_gate_param_gen(params):
    phi = params[0]

    c_re = do('cos', phi / 2)
    c_im = do('imag', c_re)
    c = do('complex', c_re, c_im)

    s_im = -do('sin', phi / 2)
    s_re = do('imag', s_im)
    s = do('complex', s_re, s_im)

    data = [[c, s], [s, c]]
    return do('array', data, like=params)


def apply_Rx(psi, theta, i, parametrize=False, **gate_opts):
    """Apply an X-rotation of ``theta`` to tensor network wavefunction ``psi``.
    """
    mtags = _merge_tags('RX', gate_opts)
    if parametrize:
        G = ops.PArray(rx_gate_param_gen, (float(theta),))
    else:
        G = qu.Rx(float(theta))
    psi.gate_(G, int(i), tags=mtags, **gate_opts)


def ry_gate_param_gen(params):
    phi = params[0]

    c_re = do('cos', phi / 2)
    c_im = do('imag', c_re)
    c = do('complex', c_re, c_im)

    s_re = do('sin', phi / 2)
    s_im = do('imag', s_re)
    s = do('complex', s_re, s_im)

    data = [[c, -s], [s, c]]
    return do('array', data, like=params)


def apply_Ry(psi, theta, i, parametrize=False, **gate_opts):
    """Apply a Y-rotation of ``theta`` to tensor network wavefunction ``psi``.
    """
    mtags = _merge_tags('RY', gate_opts)
    if parametrize:
        G = ops.PArray(ry_gate_param_gen, (float(theta),))
    else:
        G = qu.Ry(float(theta))
    psi.gate_(G, int(i), tags=mtags, **gate_opts)


def rz_gate_param_gen(params):
    phi = params[0]

    c_re = do('cos', phi / 2)
    c_im = do('imag', c_re)
    c = do('complex', c_re, c_im)

    s_im = -do('sin', phi / 2)
    s_re = do('imag', s_im)
    s = do('complex', s_re, s_im)

    data = [[c + s, 0], [0, c - s]]
    return do('array', data, like=params)


def apply_Rz(psi, theta, i, parametrize=False, **gate_opts):
    """Apply a Z-rotation of ``theta`` to tensor network wavefunction ``psi``.
    """
    mtags = _merge_tags('RZ', gate_opts)
    if parametrize:
        G = ops.PArray(rz_gate_param_gen, (float(theta),))
    else:
        G = qu.Rz(float(theta))
    psi.gate_(G, int(i), tags=mtags, **gate_opts)


def u3_gate_param_gen(params):
    theta, phi, lamda = params[0], params[1], params[2]

    c2_re = do('cos', theta / 2)
    c2_im = do('imag', c2_re)
    c2 = do('complex', c2_re, c2_im)

    s2_re = do('sin', theta / 2)
    s2_im = do('imag', s2_re)
    s2 = do('complex', s2_re, s2_im)

    el_im = lamda
    el_re = do('imag', el_im)
    el = do('exp', do('complex', el_re, el_im))

    ep_im = phi
    ep_re = do('imag', ep_im)
    ep = do('exp', do('complex', ep_re, ep_im))

    elp_im = lamda + phi
    elp_re = do('imag', elp_im)
    elp = do('exp', do('complex', elp_re, elp_im))

    data = [[c2, -el * s2],
            [ep * s2, elp * c2]]
    return do('array', data, like=params)


def apply_U3(psi, theta, phi, lamda, i, parametrize=False, **gate_opts):
    mtags = _merge_tags('U3', gate_opts)
    if parametrize:
        G = ops.PArray(u3_gate_param_gen, (theta, phi, lamda))
    else:
        G = qu.U_gate(theta, phi, lamda)
    psi.gate_(G, int(i), tags=mtags, **gate_opts)


def fsim_param_gen(params):
    theta, phi = params[0], params[1]

    a_re = do('cos', theta)
    a_im = do('imag', a_re)
    a = do('complex', a_re, a_im)

    b_im = -do('sin', theta)
    b_re = do('imag', b_im)
    b = do('complex', b_re, b_im)

    c_im = -phi
    c_re = do('imag', c_im)
    c = do('exp', do('complex', c_re, c_im))

    data = [[[[1, 0],
              [0, 0]],
             [[0, a],
              [b, 0]]],
            [[[0, b],
              [a, 0]],
             [[0, 0],
              [0, c]]]]

    return do('array', data, like=params)


def apply_fsim(psi, theta, phi, i, j, parametrize=False, **gate_opts):
    mtags = _merge_tags('FSIM', gate_opts)
    if parametrize:
        G = ops.PArray(fsim_param_gen, (theta, phi))
    else:
        G = qu.fsim(theta, phi)
    psi.gate_(G, (int(i), int(j)), tags=mtags, **gate_opts)


GATE_FUNCTIONS = {
    # constant single qubit gates
    'H': build_gate_1(qu.hadamard(), tags='H'),
    'X': build_gate_1(qu.pauli('X'), tags='X'),
    'Y': build_gate_1(qu.pauli('Y'), tags='Y'),
    'Z': build_gate_1(qu.pauli('Z'), tags='Z'),
    'S': build_gate_1(qu.S_gate(), tags='S'),
    'T': build_gate_1(qu.T_gate(), tags='T'),
    'X_1_2': build_gate_1(qu.Xsqrt(), tags='X_1/2'),
    'Y_1_2': build_gate_1(qu.Ysqrt(), tags='Y_1/2'),
    'Z_1_2': build_gate_1(qu.Zsqrt(), tags='Z_1/2'),
    'W_1_2': build_gate_1(qu.Wsqrt(), tags='W_1/2'),
    'HZ_1_2': build_gate_1(qu.Wsqrt(), tags='W_1/2'),
    # constant two qubit gates
    'CNOT': build_gate_2(qu.CNOT(), tags='CNOT'),
    'CX': build_gate_2(qu.cX(), tags='CX'),
    'CY': build_gate_2(qu.cY(), tags='CY'),
    'CZ': build_gate_2(qu.cZ(), tags='CZ'),
    'IS': build_gate_2(qu.iswap(), tags='ISWAP'),
    'ISWAP': build_gate_2(qu.iswap(), tags='ISWAP'),
    # special non-tensor gates
    'IDEN': lambda *args, **kwargs: None,
    'SWAP': apply_swap,
    # parametrizable gates
    'RX': apply_Rx,
    'RY': apply_Ry,
    'RZ': apply_Rz,
    'U3': apply_U3,
    'FS': apply_fsim,
    'FSIM': apply_fsim,
}

ONE_QUBIT_PARAM_GATES = {'RX', 'RY', 'RZ', 'U3'}
TWO_QUBIT_PARAM_GATES = {'FS', 'FSIM'}
ALL_PARAM_GATES = ONE_QUBIT_PARAM_GATES | TWO_QUBIT_PARAM_GATES


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
        >>> qc.apply_gates(gates)
        >>> qc.psi
        <TensorNetwork1DVector(tensors=12, indices=14, nsites=3)>

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
        self.gate_opts.setdefault('contract', 'auto-split-gate')
        self.gate_opts.setdefault('propagate_tags', 'register')
        self.gates = []

        # when we add gates we will modify the TN structure, apart from in the
        # 'swap+split' case, which explicitly maintains an MPS
        if self.gate_opts['contract'] != 'swap+split':
            self._psi.view_as_(TensorNetwork1DVector)

    @classmethod
    def from_qasm(cls, qasm, **quantum_circuit_opts):
        """Generate a ``Circuit`` instance from a qasm string.
        """
        info = parse_qasm(qasm)
        qc = cls(info['n'], **quantum_circuit_opts)
        qc.apply_gates(info['gates'])
        return qc

    @classmethod
    def from_qasm_file(cls, fname, **quantum_circuit_opts):
        """Generate a ``Circuit`` instance from a qasm file.
        """
        info = parse_qasm_file(fname)
        qc = cls(info['n'], **quantum_circuit_opts)
        qc.apply_gates(info['gates'])
        return qc

    @classmethod
    def from_qasm_url(cls, url, **quantum_circuit_opts):
        """Generate a ``Circuit`` instance from a qasm url.
        """
        info = parse_qasm_url(url)
        qc = cls(info['n'], **quantum_circuit_opts)
        qc.apply_gates(info['gates'])
        return qc

    def apply_gate(self, gate_id, *gate_args, gate_round=None, **gate_opts):
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
        gate_opts
            Supplied to the gate function, options here will override the
            default ``gate_opts``.
        """

        # unique tag
        tags = tags_to_oset(f'GATE_{len(self.gates)}')

        # parse which 'round' of gates
        if (gate_round is not None):
            tags.add(f'ROUND_{gate_round}')
        elif isinstance(gate_id, numbers.Integral) or gate_id.isdigit():
            # gate round given as first entry of qasm line
            tags.add(f'ROUND_{gate_id}')
            gate_id, gate_args = gate_args[0], gate_args[1:]

        gate_id = gate_id.upper()
        gate_fn = GATE_FUNCTIONS[gate_id]

        # overide any default gate opts
        opts = {**self.gate_opts, **gate_opts}

        # handle parametrize kwarg for non-parametrizable gates
        if ('parametrize' in opts) and (gate_id not in ALL_PARAM_GATES):
            parametrize = opts.pop('parametrize')
            if parametrize:
                msg = f"The gate '{gate_id}' cannot be parametrized."
                raise ValueError(msg)
            # can pop+ignore if False

        # gate the TN!
        gate_fn(self._psi, *gate_args, tags=tags, **opts)

        # keep track of the gates applied
        self.gates.append((gate_id, *gate_args))

    def apply_gates(self, gates):
        """Apply a sequence of gates to this tensor network quantum circuit.

        Parameters
        ----------
        gates : list[list[str]]
            The sequence of gates to apply.
        """
        for gate in gates:
            self.apply_gate(*gate)

        self._psi.squeeze_()

    def apply_circuit(self, gates):
        import warnings
        msg = ("``apply_circuit`` is deprecated in favour of ``apply_gates``.")
        warnings.warn(msg, DeprecationWarning)
        self.apply_gates(gates)

    def h(self, i, gate_round=None):
        self.apply_gate('H', i, gate_round=gate_round)

    def x(self, i, gate_round=None):
        self.apply_gate('X', i, gate_round=gate_round)

    def y(self, i, gate_round=None):
        self.apply_gate('Y', i, gate_round=gate_round)

    def z(self, i, gate_round=None):
        self.apply_gate('Z', i, gate_round=gate_round)

    def s(self, i, gate_round=None):
        self.apply_gate('S', i, gate_round=gate_round)

    def t(self, i, gate_round=None):
        self.apply_gate('T', i, gate_round=gate_round)

    def x_1_2(self, i, gate_round=None):
        self.apply_gate('X_1_2', i, gate_round=gate_round)

    def y_1_2(self, i, gate_round=None):
        self.apply_gate('Y_1_2', i, gate_round=gate_round)

    def z_1_2(self, i, gate_round=None):
        self.apply_gate('Z_1_2', i, gate_round=gate_round)

    def w_1_2(self, i, gate_round=None):
        self.apply_gate('W_1_2', i, gate_round=gate_round)

    def hz_1_2(self, i, gate_round=None):
        self.apply_gate('HZ_1_2', i, gate_round=gate_round)

    # constant two qubit gates

    def cnot(self, i, j, gate_round=None):
        self.apply_gate('CNOT', i, j, gate_round=gate_round)

    def cx(self, i, j, gate_round=None):
        self.apply_gate('CX', i, j, gate_round=gate_round)

    def cy(self, i, j, gate_round=None):
        self.apply_gate('CY', i, j, gate_round=gate_round)

    def cz(self, i, j, gate_round=None):
        self.apply_gate('CZ', i, j, gate_round=gate_round)

    def iswap(self, i, j, gate_round=None):
        self.apply_gate('ISWAP', i, j)

    # special non-tensor gates

    def iden(self, i, gate_round=None):
        pass

    def swap(self, i, j, gate_round=None):
        self.apply_gate('SWAP', i, j)

    # parametrizable gates

    def rx(self, theta, i, gate_round=None, parametrize=False):
        self.apply_gate('RX', theta, i, gate_round=gate_round,
                        parametrize=parametrize)

    def ry(self, theta, i, gate_round=None, parametrize=False):
        self.apply_gate('RY', theta, i, gate_round=gate_round,
                        parametrize=parametrize)

    def rz(self, theta, i, gate_round=None, parametrize=False):
        self.apply_gate('RZ', theta, i, gate_round=gate_round,
                        parametrize=parametrize)

    def u3(self, theta, phi, lamda, i, gate_round=None, parametrize=False):
        self.apply_gate('U3', theta, phi, lamda, i,
                        gate_round=gate_round, parametrize=parametrize)

    def fsim(self, theta, phi, i, j, gate_round=None, parametrize=False):
        self.apply_gate('FSIM', theta, phi, i, j,
                        gate_round=gate_round, parametrize=parametrize)

    @property
    def psi(self):
        """Tensor network representation of the wavefunction.
        """
        # make sure all same dtype and drop singlet dimensions
        psi = self._psi.squeeze()
        psi.astype_(psi.dtype)
        return psi

    @property
    def uni(self):
        """Tensor network representation of the unitary operator (i.e. with
        the initial state removed).
        """
        U = self.psi

        # rename the initial state rand_uuid bonds to 1D site inds
        ixmap = {f'k{i}': f'b{i}' for i in range(self.N)}

        # the first `N` tensors should be the tensors of input state
        tids = tuple(U.tensor_map)[:self.N]
        for i, tid in enumerate(tids):
            t = U._pop_tensor(tid)
            assert U.site_tag(i) in t.tags
            old_ix, = t.inds
            ixmap[old_ix] = f'k{i}'

        return U.reindex_(ixmap)

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

        p_dense = psi.to_dense(inds, **contract_opts)
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

    def schrodinger_contract(self, *args, **contract_opts):
        ntensor = self._psi.num_tensors
        path = [(0, 1)] + [(0, i) for i in reversed(range(1, ntensor - 1))]
        return self.psi.contract(*args, optimize=path, **contract_opts)

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
        for i, gate in enumerate(self.gates):
            label = gate[0]
            tag = f'GATE_{i}'
            t = tn[tag]

            # sanity check that tensor(s) `t` correspond to the correct gate
            if label not in get_tags(t):
                raise ValueError(f"The tensor(s) correponding to gate {i} "
                                 f"should be tagged with '{label}', got {t}.")

            # only update gates and tensors if they are parametrizable
            if isinstance(t, PTensor):

                # update the actual tensor
                self._psi[tag].params = t.params

                # update the gate entry
                if label in ONE_QUBIT_PARAM_GATES:
                    new_gate = (label, *t.params, gate[-1])
                elif label in TWO_QUBIT_PARAM_GATES:
                    new_gate = (label, *t.params, *gate[-2:])
                else:
                    raise ValueError(f"Didn't recognise '{label}' "
                                     "gate as parametrizable.")

                self.gates[i] = new_gate

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
