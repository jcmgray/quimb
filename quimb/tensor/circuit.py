import numbers

import cytoolz
import numpy as np
from autoray import do

import quimb as qu
from ..utils import progbar as _progbar, oset
from .tensor_core import (get_tags, tags_to_oset, oset_union,
                          PTensor, TensorNetwork)
from .tensor_gen import MPS_computational_state
from .tensor_1d import TensorNetwork1DVector, Dense1D
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


def sample_bitstring_from_prob_ndarray(p):
    """Sample a bitstring from n-dimensional tensor ``p`` of probabilities.

    Examples
    --------

        >>> import numpy as np
        >>> p = np.zeros(shape=(2, 2, 2, 2, 2))
        >>> p[0, 1, 0, 1, 1] = 1.0
        >>> sample_bitstring_from_prob_ndarray(p)
        '01011'
    """
    b = np.random.choice(np.arange(p.size), p=p.flat)
    return f"{b:0>{p.ndim}b}"


# --------------------------- main circuit class ---------------------------- #

class Circuit:
    """Class for simulating quantum circuits using tensor networks.

    Parameters
    ----------
    N : int, optional
        The number of qubits.
    psi0 : TensorNetwork1DVector, optional
        The initial state, assumed to be ``|00000....0>`` if not given. The
        state is always copied and the tag ``PSI0`` added.
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
    """

    def __init__(
        self,
        N=None,
        psi0=None,
        gate_opts=None,
        tags=None,
        psi0_dtype='complex128',
        psi0_tag='PSI0',
    ):

        if N is None and psi0 is None:
            raise ValueError("You must supply one of `N` or `psi0`.")

        elif psi0 is None:
            self.N = N
            self._psi = MPS_computational_state('0' * N, dtype=psi0_dtype)

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

        self.gate_opts = {} if gate_opts is None else dict(gate_opts)
        self.gate_opts.setdefault('contract', 'auto-split-gate')
        self.gate_opts.setdefault('propagate_tags', 'register')
        self.gates = []

        # when we add gates we will modify the TN structure, apart from in the
        # 'swap+split' case, which explicitly maintains an MPS
        if self.gate_opts['contract'] != 'swap+split':
            self._psi.view_as_(TensorNetwork1DVector)

        self._sample_n_gates = -1
        self._sample_norm_cache = None
        self._sample_norm_path_cache = None

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
        psi = self._psi.copy()
        psi.squeeze_()
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

        for i, gate in reversed(tuple(enumerate(self.gates))):
            if isinstance(gate[-2], numbers.Integral):
                regs = set(gate[-2:])
            else:
                regs = set(gate[-1:])

            if regs & cone:
                lightcone_tags.append(f"GATE_{i}")
                cone |= regs

        # initial state is always part of the lightcone
        lightcone_tags.append('PSI0')
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
            # these sites are in the lightcone regardless of gates above them
            site_tags = oset(psi.site_tag(i) for i in where)

            for tid, t in tuple(psi_lc.tensor_map.items()):
                # get all tensors connected to this tensor (incld itself)
                neighbors = oset_union(psi_lc.ind_map[ix] for ix in t.inds)

                # lone tensor not attached to anything - drop it
                # but only if it isn't directly in the ``where`` region
                if (len(neighbors) == 1) and not (t.tags & site_tags):
                    psi_lc._pop_tensor(tid)

        return psi_lc

    def get_norm_lightcone_simplified(self, where, seq='ADCRS', atol=1e-6):
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

        Returns
        -------
        TensorNetwork
        """
        key = (where, seq, atol)
        if key in self._sample_norm_cache:
            return self._sample_norm_cache[key].copy()

        psi_lc = self.get_psi_reverse_lightcone(where)
        norm_lc = psi_lc.H & psi_lc

        # don't want to simplify site indices in region away
        output_inds = tuple(map(psi_lc.site_ind, where))

        # # simplify the norm and cache it
        norm_lc.full_simplify_(seq=seq, atol=atol, output_inds=output_inds)
        self._sample_norm_cache[key] = norm_lc

        # return a copy so we can modify it inplace
        return norm_lc.copy()

    def get_psi_simplified(self, seq='ADCRS', atol=1e-6):
        """Get the full wavefunction post local tensor newtork simplification.

        Parameters
        ----------
        seq : str, optional
            Which local tensor network simplifications to perform and in which
            order, see
            :meth:`~quimb.tensor.tensor_core.TensorNetwork.full_simplify`.
        atol : float, optional
            The tolerance with which to compare to zero when applying
            :meth:`~quimb.tensor.tensor_core.TensorNetwork.full_simplify`.

        Returns
        -------
        TensorNetwork1DVector
        """
        key = (tuple(range(self.N)), seq, atol)
        if key in self._sample_norm_cache:
            return self._sample_norm_cache[key].copy()

        psi = self.psi
        # make sure to keep all outer indices
        output_inds = tuple(map(psi.site_ind, range(self.N)))

        # simplify the state and cache it
        psi.full_simplify_(seq=seq, atol=atol, output_inds=output_inds)
        self._sample_norm_cache[key] = psi

        # return a copy so we can modify it inplace
        return psi.copy()

    def _maybe_init_sampling_caches(self):
        # clear/create the cache if circuit has changed
        if self._sample_n_gates != len(self.gates):
            self._sample_n_gates = len(self.gates)

            # storage
            self._lightcone_orderings = dict()
            self._sample_norm_cache = dict()
            self._sample_norm_path_cache = dict()
            self._sampling_conditionals = dict()
            self._sampling_sliced_contractions = dict()
            self._marginal_storage_size = 0

    def _get_sliced_contractor(
        self,
        info,
        target_size,
        arrays,
        overhead_warn=2.0,
    ):
        key = (info.eq, target_size)
        if key in self._sampling_sliced_contractions:
            sc = self._sampling_sliced_contractions[key]
            sc.arrays = arrays
            return sc

        from cotengra import SliceFinder

        sf = SliceFinder(info, target_size=target_size)
        ix_sl, cost_sl = sf.search()

        if cost_sl.overhead > overhead_warn:
            import warnings
            warnings.warn(
                f"Slicing contraction to size {target_size} has introduced"
                f" an FLOPs overhead of {cost_sl.overhead:.2f}x.")

        sc = sf.SlicedContractor(arrays)
        self._sampling_sliced_contractions[key] = sc
        return sc

    def compute_marginal(
        self,
        where,
        fix=None,
        optimize='auto-hq',
        backend='auto',
        dtype='complex64',
        simplify_sequence='ADCRS',
        simplify_atol=1e-6,
        target_size=None,
        get=None,
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
            Contraction path optimizer to use for the marginal, can be
            a reusable path optimizer as only called once (though path won't be
            cached for later use in that case).
        backend : str, optional
            Backend to perform the marginal contraction with, e.g. ``'numpy'``,
            ``'cupy'`` or ``'jax'``. Passed to ``opt_einsum``.
        dtype : str, optional
            Data type to cast the TN to before contraction.
        simplify_sequence : str, optional
            Which local tensor network simplifications to perform and in which
            order, see
            :meth:`~quimb.tensor.tensor_core.TensorNetwork.full_simplify`.
        simplify_atol : float, optional
            The tolerance with which to compare to zero when applying
            :meth:`~quimb.tensor.tensor_core.TensorNetwork.full_simplify`.
        get : None or 'path-info', optional
            Whether to perform the marginal contraction or just return the
            associated contraction path information.
        target_size : None or int, optional
            The largest size of tensor to allow. If specified and any
            contraction involves tensors bigger than this, 'slice' the
            contraction into independent parts and sum them individually.
            Requires ``cotengra`` currently.
        """
        self._maybe_init_sampling_caches()

        site_ind = self._psi.site_ind

        # index trick to contract straight to reduced density matrix diagonal
        # rho_ii -> p_i (i.e. insert a COPY tensor into the norm)
        output_inds = [site_ind(i) for i in where]

        fs_opts = dict(seq=simplify_sequence, atol=simplify_atol)

        # lightcone region is target qubit plus fixed qubits
        region = set(where)
        if fix is not None:
            region |= set(fix)
        region = tuple(sorted(region))

        # have we fixed or are measuring all qubits?
        final_marginal = (len(region) == self.N)

        # these both are cached and produce TN copies
        if final_marginal:
            # won't need to partially trace anything -> just need ket
            nm_lc = self.get_psi_simplified(**fs_opts)
        else:
            # can use lightcone cancellation on partially traced qubits
            nm_lc = self.get_norm_lightcone_simplified(region, **fs_opts)

        if fix:
            # project (slice) fixed tensors with bitstring
            # this severs the indices connecting bra and ket on fixed sites
            nm_lc.isel_({site_ind(i): int(b) for i, b in fix.items()})

        # having sliced we can do a final simplify
        nm_lc.full_simplify_(output_inds=output_inds, **fs_opts)

        # cast to desired data type
        nm_lc.astype_(dtype)

        # NB. the path isn't *neccesarily* the same each time due to the post
        #     slicing full simplify, however there is also the lower level
        #     contraction path cache if the structure generated *is* the same
        #     so still pretty efficient to just overwrite
        info = self._sample_norm_path_cache[region] = nm_lc.contract(
            all, output_inds=output_inds,
            optimize=optimize, get='path-info'
        )

        if get == 'path-info':
            return info

        if target_size is not None:
            # perform the 'sliced' contraction restricted to ``target_size``
            arrays = tuple(t.data for t in nm_lc)
            sc = self._get_sliced_contractor(info, target_size, arrays)
            p_marginal = abs(sc.contract_all(backend=backend))
        else:
            # perform the full contraction with the path found
            p_marginal = abs(nm_lc.contract(
                all, output_inds=output_inds,
                optimize=info.path, backend=backend
            ).data)

        if final_marginal:
            # we only did half the ket contraction so need to square
            p_marginal = p_marginal**2

        return p_marginal

    def calc_qubit_ordering(self, qubits=None):
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
        self._maybe_init_sampling_caches()

        if qubits is None:
            qubits = tuple(range(self.N))
        else:
            qubits = tuple(sorted(qubits))

        # check the cache first
        if qubits in self._lightcone_orderings:
            return self._lightcone_orderings[qubits]

        cone = set()
        lctgs = {i: set(self.get_reverse_lightcone_tags(i)) for i in qubits}

        order = []
        while lctgs:
            # get the next qubit which adds least num gates to lightcone
            next_qubit = min(lctgs, key=lambda i: len(lctgs[i] - cone))
            cone |= lctgs.pop(next_qubit)
            order.append(next_qubit)

        order = self._lightcone_orderings[qubits] = tuple(order)

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
            tuple(sorted(g))
            for g in cytoolz.partition_all(group_size, order)
        )

    def sample(
        self,
        C,
        qubits=None,
        order=None,
        group_size=1,
        max_marginal_storage=2**20,
        seed=None,
        optimize='auto-hq',
        backend='auto',
        dtype='complex64',
        simplify_sequence='ADCRS',
        simplify_atol=1e-6,
        target_size=None,
    ):
        """Sample the circuit given by ``gates``, ``C`` times, using lightcone
        cancelling and caching marginal distribution results. This is a
        generator.

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
            a reusable path optimizer as called on many different TNs. Passed
            to :func:`opt_einsum.contract_path`. If you want to use a custom
            path optimizer register it with a name using
            ``opt_einsum.paths.register_path_fn`` after which the paths will be
            cached on name.
        backend : str, optional
            Backend to perform the marginal contraction with, e.g. ``'numpy'``,
            ``'cupy'`` or ``'jax'``. Passed to ``opt_einsum``.
        dtype : str, optional
            Data type to cast the TN to before contraction.
        simplify_sequence : str, optional
            Which local tensor network simplifications to perform and in which
            order, see
            :meth:`~quimb.tensor.tensor_core.TensorNetwork.full_simplify`.
        simplify_atol : float, optional
            The tolerance with which to compare to zero when applying
            :meth:`~quimb.tensor.tensor_core.TensorNetwork.full_simplify`.
        target_size : None or int, optional
            The largest size of tensor to allow. If specified and any
            contraction involves tensors bigger than this, 'slice' the
            contraction into independent parts and sum them individually.
            Requires ``cotengra`` currently.

        Yields
        ------
        bitstrings : sequence of str
        """
        # init TN norms, contraction paths, and marginals
        self._maybe_init_sampling_caches()

        # which qubits and an ordering e.g. (2, 3, 4, 5), (5, 3, 4, 2)
        qubits, order = self._parse_qubits_order(qubits, order)

        # group the ordering e.g. ((5, 3), (4, 2))
        groups = self._group_order(order, group_size)

        if seed is not None:
            np.random.seed(seed)

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
                if key not in self._sampling_conditionals:
                    # compute p(qs=x | current bitstring)
                    p = self.compute_marginal(
                        where=where,
                        fix=result,
                        optimize=optimize,
                        backend=backend,
                        dtype=dtype,
                        simplify_sequence=simplify_sequence,
                        simplify_atol=simplify_atol,
                        target_size=target_size,
                    )
                    p = do('to_numpy', p).astype('float64')
                    p /= p.sum()

                    if self._marginal_storage_size <= max_marginal_storage:
                        self._sampling_conditionals[key] = p
                        self._marginal_storage_size += p.size
                else:
                    p = self._sampling_conditionals[key]

                # the sampled bitstring e.g. '1' or '001010101'
                b_where = sample_bitstring_from_prob_ndarray(p)

                # split back into individual qubit results
                for q, b in zip(where, b_where):
                    result[q] = b

            yield "".join(result[i] for i in qubits)
            result.clear()

    def sample_rehearse(
        self,
        qubits=None,
        order=None,
        group_size=1,
        result=None,
        optimize='auto-hq',
        simplify_sequence='ADCRS',
        simplify_atol=1e-6,
        progbar=False,
    ):
        """
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
            a reusable path optimizer as called on many different TNs. Passed
            to :func:`opt_einsum.contract_path`. If you want to use a custom
            path optimizer register it with a name using
            ``opt_einsum.paths.register_path_fn`` after which the paths will be
            cached on name.
        simplify_sequence : str, optional
            Which local tensor network simplifications to perform and in which
            order, see
            :meth:`~quimb.tensor.tensor_core.TensorNetwork.full_simplify`.
        simplify_atol : float, optional
            The tolerance with which to compare to zero when applying
            :meth:`~quimb.tensor.tensor_core.TensorNetwork.full_simplify`.
        progbar : bool, optional
            Whether to show the progress of finding each contraction path.

        Returns
        -------
        infos : tuple[opt_einsum.PathInfo]
            One contraction path info object per grouped marginal computation.
        """
        # init TN norms, contraction paths, and marginals
        self._maybe_init_sampling_caches()
        qubits, order = self._parse_qubits_order(qubits, order)
        groups = self._group_order(order, group_size)

        if result is None:
            result = {q: '0' for q in qubits}

        infos = []
        fix = {}
        for where in _progbar(groups, disable=not progbar):
            infos.append(self.compute_marginal(
                where=where,
                fix=fix,
                optimize=optimize,
                simplify_sequence=simplify_sequence,
                get='path-info',
            ))

            # set the result of qubit ``q`` arbitrarily
            for q in where:
                fix[q] = result[q]

        return tuple(infos)

    def sample_chaotic(
        self,
        C,
        marginal_qubits,
        max_marginal_storage=2**20,
        seed=None,
        optimize='auto-hq',
        backend='auto',
        dtype='complex64',
        simplify_sequence='ADCRS',
        simplify_atol=1e-6,
        target_size=None,
    ):
        """Sample from this circuit, *assuming* it to be chaotic. Which is to
        say, only compute and sample correctly from the final marginal,
        assuming that the distribution on the other qubits is uniform.

        Parameters
        ----------
        C : int
            The number of times to sample.
        marginal_qubits : int or sequence of int
            The number of qubits to treat as marginal, or the actual qubits. If
            an int is given then the qubits treated as marginal will be
            ``circuit.calc_qubit_ordering()[:marginal_qubits]``.
        seed : None or int, optional
            A random seed, passed to ``numpy.random.seed`` if given.
        optimize : str, optional
            Contraction path optimizer to use for the marginals, shouldn't be
            a reusable path optimizer as called on many different TNs. Passed
            to :func:`opt_einsum.contract_path`. If you want to use a custom
            path optimizer register it with a name using
            ``opt_einsum.paths.register_path_fn`` after which the paths will be
            cached on name.
        backend : str, optional
            Backend to perform the marginal contraction with, e.g. ``'numpy'``,
            ``'cupy'`` or ``'jax'``. Passed to ``opt_einsum``.
        dtype : str, optional
            Data type to cast the TN to before contraction.
        simplify_sequence : str, optional
            Which local tensor network simplifications to perform and in which
            order, see
            :meth:`~quimb.tensor.tensor_core.TensorNetwork.full_simplify`.
        simplify_atol : float, optional
            The tolerance with which to compare to zero when applying
            :meth:`~quimb.tensor.tensor_core.TensorNetwork.full_simplify`.
        target_size : None or int, optional
            The largest size of tensor to allow. If specified and any
            contraction involves tensors bigger than this, 'slice' the
            contraction into independent parts and sum them individually.
            Requires ``cotengra`` currently.
        """
        # init TN norms, contraction paths, and marginals
        self._maybe_init_sampling_caches()
        qubits = tuple(range(self.N))

        if seed is not None:
            np.random.seed(seed)

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
                result[q] = np.random.choice(('0', '1'))

            # compute the remaining marginal
            key = (where, tuple(sorted(result.items())))
            if key not in self._sampling_conditionals:
                p = self.compute_marginal(
                    where=where,
                    fix=result,
                    optimize=optimize,
                    backend=backend,
                    dtype=dtype,
                    simplify_sequence=simplify_sequence,
                    simplify_atol=simplify_atol,
                    target_size=target_size,
                )
                p = do('to_numpy', p).astype('float64')
                p /= p.sum()

                if self._marginal_storage_size <= max_marginal_storage:
                    self._sampling_conditionals[key] = p
                    self._marginal_storage_size += p.size
            else:
                p = self._sampling_conditionals[key]

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
        optimize='auto-hq',
        simplify_sequence='ADCRS',
        simplify_atol=1e-6,
    ):
        """Rehearse chaotic sampling (perform just the TN simplifications and
        contraction path finding).

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
            Contraction path optimizer to use for the marginal, can be
            a reusable path optimizer as only called once (though path won't be
            cached for later use in that case).
        simplify_sequence : str, optional
            Which local tensor network simplifications to perform and in which
            order, see
            :meth:`~quimb.tensor.tensor_core.TensorNetwork.full_simplify`.
        simplify_atol : float, optional
            The tolerance with which to compare to zero when applying
            :meth:`~quimb.tensor.tensor_core.TensorNetwork.full_simplify`.

        Returns
        -------
        info : opt_einsum.PathInfo
            The contraction path information for the main computation.
        """

        # init TN norms, contraction paths, and marginals
        self._maybe_init_sampling_caches()
        qubits = tuple(range(self.N))

        if isinstance(marginal_qubits, numbers.Integral):
            marginal_qubits = self.calc_qubit_ordering()[:marginal_qubits]
        where = tuple(sorted(marginal_qubits))

        fix_qubits = tuple(q for q in qubits if q not in where)

        if result is None:
            fix = {q: '0' for q in fix_qubits}
        else:
            fix = {q: result[q] for q in fix_qubits}

        return self.compute_marginal(
            where=where,
            fix=fix,
            optimize=optimize,
            simplify_sequence=simplify_sequence,
            simplify_atol=simplify_atol,
            get='path-info',
        )

    def to_dense(self, reverse=False, dtype=None,
                 simplify_sequence='R', **contract_opts):
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
        simplify_sequence : str, optional
            Which local tensor network simplifications to perform and in which
            order, see
            :meth:`~quimb.tensor.tensor_core.TensorNetwork.full_simplify`.
        contract_opts
            Supplied to
            :meth:`~quimb.tensor.tensor_core.TensorNetwork.to_dense`.

        Returns
        -------
        psi : qarray
            The densely represented wavefunction with ``dtype`` data.
        """
        psi = self.psi

        if dtype is not None:
            psi.astype_(dtype)

        inds = [psi.site_ind(i) for i in range(self.N)]

        if simplify_sequence:
            psi.full_simplify_(simplify_sequence, output_inds=inds)

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

    @property
    def uni(self):
        raise ValueError("You can't extract the circuit unitary "
                         "TN from a ``CircuitMPS``.")

    def calc_qubit_ordering(self, qubits=None):
        """MPS already has a natural ordering.
        """
        if qubits is None:
            return tuple(range(self.N))
        else:
            return tuple(sorted(qubits))

    def get_psi_reverse_lightcone(self, where, keep_psi0=False):
        """Override ``get_psi_reverse_lightcone`` as for an MPS the lightcone
        is not meaningful.
        """
        return self.psi


class CircuitDense(Circuit):
    """Quantum circuit simulation keeping the state in full dense form.
    """

    def __init__(self, N=None, psi0=None, gate_opts=None, tags=None):
        gate_opts = {} if gate_opts is None else dict(gate_opts)
        gate_opts.setdefault('contract', True)
        super().__init__(N, psi0, gate_opts, tags)

    @property
    def psi(self):
        t = self._psi ^ all
        psi = TensorNetwork([t])
        psi.view_as_(Dense1D, like=self._psi)
        return psi

    @property
    def uni(self):
        raise ValueError("You can't extract the circuit unitary "
                         "TN from a ``CircuitDense``.")

    def calc_qubit_ordering(self, qubits=None):
        """Qubit ordering doesn't matter for a dense wavefunction.
        """
        if qubits is None:
            return tuple(range(self.N))
        else:
            return tuple(sorted(qubits))

    def get_psi_reverse_lightcone(self, where, keep_psi0=False):
        """Override ``get_psi_reverse_lightcone`` as for a dense wavefunction
        the lightcone is not meaningful.
        """
        return self.psi
