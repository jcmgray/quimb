"""Tools for quantum circuit simulation using tensor networks.
"""

import re
import math
import numbers
import operator
import functools
import itertools

import numpy as np
from autoray import do, reshape, backend_like

import quimb as qu
from ..utils import progbar as _progbar
from ..utils import oset, partitionby, concatv, partition_all, ensure_dict, LRU
from .tensor_core import (
    get_tags, tags_to_oset, oset_union, tensor_contract,
    PTensor, Tensor, TensorNetwork, rand_uuid,
)
from .tensor_builder import (
    MPS_computational_state, TN_from_sites_computational_state
)
from .tensor_arbgeom import TensorNetworkGenOperator
from .tensor_1d import Dense1D
from . import array_ops as ops


def recursive_stack(x):
    if not isinstance(x, (list, tuple)):
        return x
    return do("stack", tuple(map(recursive_stack, x)))


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
    parts = tuple(partitionby(type, x))
    return tuple(concatv(*parts[:-2], parts[-1], parts[-2]))


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


ALL_GATES = set()
ONE_QUBIT_GATES = set()
TWO_QUBIT_GATES = set()
ALL_PARAM_GATES = set()
ONE_QUBIT_PARAM_GATES = set()
TWO_QUBIT_PARAM_GATES = set()

# the tensor tags to use for each gate (defaults to label)
GATE_TAGS = {}

# the number of qubits a gate acts on
GATE_SIZE = {}

# gates which just require a constant array
CONSTANT_GATES = {}

# gates which are parametrized
PARAM_GATES = {}

# gates which involve a non-array operation such as reindexing only
SPECIAL_GATES = {}


def register_constant_gate(name, G, num_qubits, tag=None):
    if tag is None:
        tag = name
    GATE_TAGS[name] = tag
    CONSTANT_GATES[name] = G
    GATE_SIZE[name] = num_qubits
    if num_qubits == 1:
        ONE_QUBIT_GATES.add(name)
    elif num_qubits == 2:
        TWO_QUBIT_GATES.add(name)
    ALL_GATES.add(name)


def register_param_gate(name, param_fn, num_qubits, tag=None):
    if tag is None:
        tag = name
    GATE_TAGS[name] = tag
    PARAM_GATES[name] = param_fn
    GATE_SIZE[name] = num_qubits
    if num_qubits == 1:
        ONE_QUBIT_GATES.add(name)
        ONE_QUBIT_PARAM_GATES.add(name)
    elif num_qubits == 2:
        TWO_QUBIT_GATES.add(name)
        TWO_QUBIT_PARAM_GATES.add(name)
    ALL_GATES.add(name)
    ALL_PARAM_GATES.add(name)


def register_special_gate(name, fn, num_qubits, tag=None):
    if tag is None:
        tag = name
    GATE_TAGS[name] = tag
    GATE_SIZE[name] = num_qubits
    if num_qubits == 1:
        ONE_QUBIT_GATES.add(name)
    elif num_qubits == 2:
        TWO_QUBIT_GATES.add(name)
    SPECIAL_GATES[name] = fn
    ALL_GATES.add(name)


# constant single qubit gates
register_constant_gate('H', qu.hadamard(), 1)
register_constant_gate('X', qu.pauli('X'), 1)
register_constant_gate('Y', qu.pauli('Y'), 1)
register_constant_gate('Z', qu.pauli('Z'), 1)
register_constant_gate('S', qu.S_gate(), 1)
register_constant_gate('T', qu.T_gate(), 1)
register_constant_gate('X_1_2', qu.Xsqrt(), 1, 'X_1/2')
register_constant_gate('Y_1_2', qu.Ysqrt(), 1, 'Y_1/2')
register_constant_gate('Z_1_2', qu.Zsqrt(), 1, 'Z_1/2')
register_constant_gate('W_1_2', qu.Wsqrt(), 1, 'W_1/2')
register_constant_gate('HZ_1_2', qu.Wsqrt(), 1, 'W_1/2')


# constant two qubit gates
register_constant_gate('CNOT', qu.CNOT(), 2)
register_constant_gate('CX', qu.cX(), 2)
register_constant_gate('CY', qu.cY(), 2)
register_constant_gate('CZ', qu.cZ(), 2)
register_constant_gate('ISWAP', qu.iswap(), 2)
register_constant_gate('IS', qu.iswap(), 2, 'ISWAP')


# single parametrizable gates

def rx_gate_param_gen(params):
    phi = params[0]

    with backend_like(phi):
        c_re = do('cos', phi / 2)
        c_im = do('imag', c_re)
        c = do('complex', c_re, c_im)

        s_im = -do('sin', phi / 2)
        s_re = do('imag', s_im)
        s = do('complex', s_re, s_im)

        return recursive_stack(((c, s), (s, c)))


register_param_gate('RX', rx_gate_param_gen, 1)


def ry_gate_param_gen(params):
    phi = params[0]

    with backend_like(phi):
        c_re = do('cos', phi / 2)
        c_im = do('imag', c_re)
        c = do('complex', c_re, c_im)

        s_re = do('sin', phi / 2)
        s_im = do('imag', s_re)
        s = do('complex', s_re, s_im)

        return recursive_stack(((c, -s), (s, c)))


register_param_gate('RY', ry_gate_param_gen, 1)


def rz_gate_param_gen(params):
    phi = params[0]

    with backend_like(phi):
        c_re = do('cos', phi / 2)
        c_im = do('imag', c_re)
        c = do('complex', c_re, c_im)

        s_im = -do('sin', phi / 2)
        s_re = do('imag', s_im)
        s = do('complex', s_re, s_im)

        # get a 'backend zero'
        z = 0.0 * c_re

        return recursive_stack(((c + s, z), (z, c - s)))


register_param_gate('RZ', rz_gate_param_gen, 1)


def u3_gate_param_gen(params):
    theta, phi, lamda = params[0], params[1], params[2]

    with backend_like(theta):
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

        return recursive_stack(((c2, -el * s2), (ep * s2, elp * c2)))


register_param_gate('U3', u3_gate_param_gen, 1)


def u2_gate_param_gen(params):
    phi, lamda = params[0], params[1]

    with backend_like(phi):
        # get a 'backend one'
        c00 = 0.0 * phi + 1.0

        c01_im = lamda
        c01_re = do('imag', c01_im)
        c01 = - do('exp', do('complex', c01_re, c01_im))

        c10_im = phi
        c10_re = do('imag', c10_im)
        c10 = do('exp', do('complex', c10_re, c10_im))

        c11_im = phi + lamda
        c11_re = do('imag', c11_im)
        c11 = do('exp', do('complex', c11_re, c11_im))

        return recursive_stack(((c00, c01), (c10, c11))) / 2**0.5


register_param_gate('U2', u2_gate_param_gen, 1)


def u1_gate_param_gen(params):
    lamda = params[0]

    with backend_like(lamda):
        # get a 'backend zero'
        c01 = c10 = 0.0 * lamda
        # get a 'backend one'
        c00 = c10 + 1.0

        c11_im = lamda
        c11_re = do('imag', c11_im)
        c11 = do('exp', do('complex', c11_re, c11_im))

        return recursive_stack(((c00, c01), (c10, c11)))


register_param_gate('U1', u1_gate_param_gen, 1)


# two qubit parametrizable gates

def cu3_param_gen(params):
    U3 = u3_gate_param_gen(params)

    with backend_like(U3):
        # get a 'backend zero'
        c0 = 0.0 * U3[0, 0]
        # get a 'backend one'
        c1 = c0 + 1.0

        data = ((((c1, c0), (c0, c0)),
                ((c0, c1), (c0, c0))),
                (((c0, c0), (U3[0, 0], U3[0, 1])),
                ((c0, c0), (U3[1, 0], U3[1, 1]))))

        return recursive_stack(data)


register_param_gate('CU3', cu3_param_gen, 2)


def cu2_param_gen(params):
    U2 = u2_gate_param_gen(params)

    with backend_like(U2):
        # get a 'backend zero'
        c0 = 0.0 * U2[0, 0]
        # get a 'backend one'
        c1 = c0 + 1.0

        data = ((((c1, c0), (c0, c0)),
                ((c0, c1), (c0, c0))),
                (((c0, c0), (U2[0, 0], U2[0, 1])),
                ((c0, c0), (U2[1, 0], U2[1, 1]))))

        return recursive_stack(data)


register_param_gate('CU2', cu2_param_gen, 2)


def cu1_param_gen(params):
    lamda = params[0]

    with backend_like(lamda):
        c11_im = lamda
        c11_re = do('imag', c11_im)
        c11 = do('exp', do('complex', c11_re, c11_im))

        # get a 'backend zero'
        c0 = 0.0 * c11
        # get a 'backend one'
        c1 = c0 + 1.0

        data = ((((c1, c0), (c0, c0)),
                ((c0, c1), (c0, c0))),
                (((c0, c0), (c1, c0)),
                ((c0, c0), (c0, c11))))

        return recursive_stack(data)


register_param_gate('CU1', cu1_param_gen, 2)


def fsim_param_gen(params):
    theta, phi = params[0], params[1]

    with backend_like(theta):
        a_re = do('cos', theta)
        a_im = do('imag', a_re)
        a = do('complex', a_re, a_im)

        b_im = -do('sin', theta)
        b_re = do('imag', b_im)
        b = do('complex', b_re, b_im)

        c_im = -phi
        c_re = do('imag', c_im)
        c = do('exp', do('complex', c_re, c_im))

        # get a 'backend zero'
        c0 = 0.0 * c
        # get a 'backend one'
        c1 = c0 + 1.0

        data = ((((c1, c0), (c0, c0)),
                ((c0, a), (b, c0))),
                (((c0, b), (a, c0)),
                ((c0, c0), (c0, c))))

        return recursive_stack(data)


register_param_gate('FSIM', fsim_param_gen, 2)
register_param_gate('FS', fsim_param_gen, 2, 'FSIM')


def fsimg_param_gen(params):
    theta, zeta, chi, gamma, phi = (
        params[0], params[1], params[2], params[3], params[4]
    )

    with backend_like(theta):
        a11_re = do('cos', theta)
        a11_im = do('imag', a11_re)
        a11 = do('complex', a11_re, a11_im)

        e11_im = -(gamma + zeta)
        e11_re = do('imag', e11_im)
        e11 = do('exp', do('complex', e11_re, e11_im))

        a22_re = do('cos', theta)
        a22_im = do('imag', a22_re)
        a22 = do('complex', a22_re, a22_im)

        e22_im = -(gamma - zeta)
        e22_re = do('imag', e22_im)
        e22 = do('exp', do('complex', e22_re, e22_im))

        a21_re = do('sin', theta)
        a21_im = do('imag', a21_re)
        a21 = do('complex', a21_re, a21_im)

        e21_im = -(gamma - chi)
        e21_re = do('imag', e21_im)
        e21 = do('exp', do('complex', e21_re, e21_im))

        a12_re = do('sin', theta)
        a12_im = do('imag', a12_re)
        a12 = do('complex', a12_re, a12_im)

        e12_im = -(gamma + chi)
        e12_re = do('imag', e12_im)
        e12 = do('exp', do('complex', e12_re, e12_im))

        img_re = do('real', -1.j)
        img_im = do('imag', -1.j)
        img = do('complex', img_re, img_im)

        c_im = -(2 * gamma + phi)
        c_re = do('imag', c_im)
        c = do('exp', do('complex', c_re, c_im))

        # get a 'backend zero'
        c0 = 0.0 * c
        # get a 'backend one'
        c1 = c0 + 1.0

        data = ((((c1, c0), (c0, c0)),
                ((c0, a11 * e11), (a21 * e21 * img, c0))),
                (((c0, a12 * e12 * img), (a22 * e22, c0)),
                ((c0, c0), (c0, c))))

        return recursive_stack(data)


register_param_gate('FSIMG', fsimg_param_gen, 2)


def rzz_param_gen(params):
    r"""
    The gate describing an Ising interaction evolution, or 'ZZ'-rotation.

    .. math::

        \mathrm{RZZ}(\gamma) = \exp(-i \gamma Z_i Z_j)

    """
    gamma = params[0]

    with backend_like(gamma):
        c00 = c11 = do('complex', do('cos', gamma), do('sin', gamma))
        c01 = c10 = do('complex', do('cos', gamma), -do('sin', gamma))

        # get a 'backend zero'
        c0 = 0.0 * c00

        data = ((((c00, c0), (c0, c0)),
                 ((c0, c01), (c0, c0))),
                (((c0, c0), (c10, c0)),
                 ((c0, c0), (c0, c11))))

        return recursive_stack(data)


register_param_gate('RZZ', rzz_param_gen, 2)


def su4_gate_param_gen(params):
    """See https://arxiv.org/abs/quant-ph/0308006 - Fig. 7.
    params:
    #     theta1, phi1, lamda1,
    #     theta2, phi2, lamda2,
    #     theta3, phi3, lamda3,
    #     theta4, phi4, lamda4,
    #     t1, t2, t3,
    """

    TA1 = Tensor(u3_gate_param_gen(params[0:3]), ['a1', 'a0'])
    TA2 = Tensor(u3_gate_param_gen(params[3:6]), ['b1', 'b0'])

    cnot = do('array', qu.CNOT().reshape(2, 2, 2, 2),
              like=params, dtype=TA1.data.dtype)

    TNOTC1 = Tensor(cnot, ['b2', 'a2', 'b1', 'a1'])
    TRz1 = Tensor(rz_gate_param_gen(params[12:13]), inds=['a3', 'a2'])
    TRy2 = Tensor(ry_gate_param_gen(params[13:14]), inds=['b3', 'b2'])
    TCNOT2 = Tensor(cnot, ['a5', 'b4', 'a3', 'b3'])
    TRy3 = Tensor(ry_gate_param_gen(params[14:15]), inds=['b5', 'b4'])
    TNOTC3 = Tensor(cnot, ['b6', 'a6', 'b5', 'a5'])
    TA3 = Tensor(u3_gate_param_gen(params[6:9]), ['a7', 'a6'])
    TA4 = Tensor(u3_gate_param_gen(params[9:12]), ['b7', 'b6'])

    return tensor_contract(
        TA1, TA2, TNOTC1,
        TRz1, TRy2, TCNOT2, TRy3,
        TNOTC3, TA3, TA4,
        output_inds=['a7', 'b7'] + ['a0', 'b0'],
        optimize='auto-hq',
    ).data


register_param_gate('SU4', su4_gate_param_gen, 2)


# special non-tensor gates

def apply_swap(psi, i, j, **gate_opts):

    contract = gate_opts.pop('contract', None)
    if contract == 'swap+split':
        gate_opts.pop('propagate_tags', None)
        psi.swap_sites_with_compress_(i, j, **gate_opts)
    else:
        iind, jind = map(psi.site_ind, (int(i), int(j)))
        psi.reindex_({iind: jind, jind: iind})


register_special_gate('SWAP', apply_swap, 2)
register_special_gate('IDEN', lambda *_, **__: None, 1)


@functools.lru_cache(2**15)
def _cached_param_gate_build(fn, params):
    return fn(params)


class Gate:
    """A simple class for storing the details of a gate.

    Parameters
    ----------
    label : str
        The name or 'identifier' of the gate.
    params : Iterable[float]
        The parameters of the gate.
    qubits : Iterable[int]
        Which qubits the gate acts on.
    round : int, optional
        If given, which round or layer the gate is part of.
    parametrize : bool, optional
        Whether the gate will correspond to a parametrized tensor.
    """

    __slots__ = (
        "_label",
        "_params",
        "_qubits",
        "_round",
        "_parametrize",
        "_tag",
        "_special",
        "_constant",
        "_array",
    )

    def __init__(
        self,
        label,
        params,
        qubits,
        round=None,
        parametrize=False,
    ):
        self._label = label.upper()
        self._params = tuple(params)
        self._qubits = tuple(qubits)
        self._round = int(round) if round is not None else round
        self._parametrize = bool(parametrize)

        self._tag = GATE_TAGS[self._label]
        self._special = self._label in SPECIAL_GATES
        self._constant = self._label in CONSTANT_GATES
        if (self._special or self._constant) and self._parametrize:
            raise ValueError(
                f"Cannot parametrize the gate: {self._label}."
            )
        self._array = None

    @classmethod
    def from_raw(cls, U, qubits, round=None):
        new = object.__new__(cls)
        new._label = f'RAW{id(U)}'
        new._params = 'raw'
        new._qubits = tuple(qubits)
        new._round = int(round) if round is not None else round
        new._special = False
        new._parametrize = isinstance(U, ops.PArray)
        new._tag = None
        new._array = U
        return new

    @property
    def label(self):
        return self._label

    @property
    def params(self):
        return self._params

    @property
    def qubits(self):
        return self._qubits

    @property
    def round(self):
        return self._round

    @property
    def special(self):
        return self._special

    @property
    def parametrize(self):
        return self._parametrize

    @property
    def tag(self):
        return self._tag

    def build_array(self):
        """Build the array representation of the gate.
        """
        if self._special:
            # these don't use an array
            raise ValueError(
                f"{self.label} gates have no array to build."
            )

        if self._constant:
            # simply return the constant array
            return CONSTANT_GATES[self._label]

        # build the array
        param_fn = PARAM_GATES[self._label]
        if self._parametrize:
            # either lazily, as tensor will be parametrized
            return ops.PArray(param_fn, self._params)

        # or cached directly into array
        try:
            return _cached_param_gate_build(param_fn, self._params)
        except TypeError:
            return param_fn(self._params)

    @property
    def array(self):
        if self._array is None:
            self._array = self.build_array()
        return self._array

    def __repr__(self):
        return (
            f"<{self.__class__.__name__}(" +
            f"label={self._label}, " +
            f"params={self._params}, " +
            f"qubits={self._qubits}, " +
            (f"round={self._round}" if self._round is not None else "") +
            (
                f", parametrize={self._parametrize})"
                if self._parametrize else ""
            )
            + ")>"
        )


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


def rehearsal_dict(tn, info):
    return {
        'tn': tn,
        'info': info,
        'W': math.log2(info.largest_intermediate),
        'C': math.log10(info.opt_cost / 2),
    }


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
    psi0_dtype : str, optional
        Ensure the initial state has this dtype.
    psi0_tag : str, optional
        Ensure the initial state has this tag.
    bra_site_ind_id : str, optional
        Use this to label 'bra' site indices when creating certain (mostly
        internal) intermediate tensor networks.

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
    """

    def __init__(
        self,
        N=None,
        psi0=None,
        gate_opts=None,
        tags=None,
        psi0_dtype='complex128',
        psi0_tag='PSI0',
        bra_site_ind_id='b{}',
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

        self.gate_opts = ensure_dict(gate_opts)
        self.gate_opts.setdefault('contract', 'auto-split-gate')
        self.gate_opts.setdefault('propagate_tags', 'register')
        self.gates = []

        self._ket_site_ind_id = self._psi.site_ind_id
        self._bra_site_ind_id = bra_site_ind_id

        if self._ket_site_ind_id == self._bra_site_ind_id:
            raise ValueError(
                "The 'ket' and 'bra' site ind ids clash : "
                "'{}' and '{}".format(self._ket_site_ind_id,
                                      self._bra_site_ind_id))

        self.ket_site_ind = self._ket_site_ind_id.format
        self.bra_site_ind = self._bra_site_ind_id.format

        self._sample_n_gates = -1
        self._storage = dict()
        self._sampled_conditionals = dict()

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

    def _init_state(self, N, dtype='complex128'):
        return TN_from_sites_computational_state(
            site_map={i: '0' for i in range(N)},
            dtype=dtype
        )

    def _apply_gate(self, gate, tags=None, **gate_opts):
        """Apply a ``Gate`` to this ``Circuit``.
        """
        tags = tags_to_oset(tags)
        tags.add(f'GATE_{len(self.gates)}')
        if gate.round is not None:
            tags.add(f'ROUND_{gate.round}')
        if gate.tag is not None:
            tags.add(gate.tag)

        # overide any default gate opts
        opts = {**self.gate_opts, **gate_opts}

        if gate.special:
            # these are specified as a general function
            SPECIAL_GATES[gate.label](
                self._psi, *gate.params, *gate.qubits, **opts
            )
        else:
            # apply the gate to the TN!
            self._psi.gate_(gate.array, gate.qubits, tags=tags, **opts)

        # keep track of the gates applied
        self.gates.append(gate)

    def apply_gate(
        self,
        gate_id,
        *gate_args,
        gate_round=None,
        parametrize=False,
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
        gate_id : str or Gate
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
        if isinstance(gate_id, Gate):
            # already encapuslated
            self._apply_gate(gate_id, **gate_opts)
            return

        if hasattr(gate_id, 'shape') and not isinstance(gate_id, str):
            # raw gate (numpy strings have a shape - ignore those)
            gate = Gate.from_raw(gate_id, gate_args, gate_round)
            self._apply_gate(gate, **gate_opts)
            return

        # else convert from tuple
        if isinstance(gate_id, numbers.Integral) or gate_id.isdigit():
            # gate round given as first entry of qasm line
            gate_round = gate_id
            gate_id, gate_args = gate_args[0], gate_args[1:]
        nq = GATE_SIZE[gate_id.upper()]
        params, qubits, = gate_args[:-nq], gate_args[-nq:]

        gate = Gate(gate_id, params, qubits, gate_round, parametrize)
        self._apply_gate(gate, **gate_opts)

    def apply_gate_raw(
        self,
        U,
        where,
        gate_round=None,
        **gate_opts
    ):
        """Apply the raw array ``U`` as a gate on qubits in ``where``. It will
        be assumed to be unitary for the sake of computing reverse lightcones.
        """
        gate = Gate.from_raw(U, where, gate_round)
        self._apply_gate(gate, **gate_opts)

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

    def apply_circuit(self, gates):  # pragma: no cover
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

    def u2(self, phi, lamda, i, gate_round=None, parametrize=False):
        self.apply_gate('U2', phi, lamda, i,
                        gate_round=gate_round, parametrize=parametrize)

    def u1(self, lamda, i, gate_round=None, parametrize=False):
        self.apply_gate('U1', lamda, i,
                        gate_round=gate_round, parametrize=parametrize)

    def cu3(self, theta, phi, lamda, i, j, gate_round=None, parametrize=False):
        self.apply_gate('CU3', theta, phi, lamda, i, j,
                        gate_round=gate_round, parametrize=parametrize)

    def cu2(self, phi, lamda, i, j, gate_round=None, parametrize=False):
        self.apply_gate('CU2', phi, lamda, i, j,
                        gate_round=gate_round, parametrize=parametrize)

    def cu1(self, lamda, i, j, gate_round=None, parametrize=False):
        self.apply_gate('CU1', lamda, i, j,
                        gate_round=gate_round, parametrize=parametrize)

    def fsim(self, theta, phi, i, j, gate_round=None, parametrize=False):
        self.apply_gate('FSIM', theta, phi, i, j,
                        gate_round=gate_round, parametrize=parametrize)

    def fsimg(self, theta, zeta, chi, gamma, phi, i, j,
              gate_round=None, parametrize=False):
        self.apply_gate('FSIMG', theta, zeta, chi, gamma, phi, i, j,
                        gate_round=gate_round, parametrize=parametrize)

    def rzz(self, theta, i, j, gate_round=None, parametrize=False):
        self.apply_gate('RZZ', theta, i, j,
                        gate_round=gate_round, parametrize=parametrize)

    def su4(
        self,
        theta1, phi1, lamda1,
        theta2, phi2, lamda2,
        theta3, phi3, lamda3,
        theta4, phi4, lamda4,
        t1, t2, t3,
        i, j,
        gate_round=None, parametrize=False
    ):
        self.apply_gate(
            'SU4',
            theta1, phi1, lamda1,
            theta2, phi2, lamda2,
            theta3, phi3, lamda3,
            theta4, phi4, lamda4,
            t1, t2, t3,
            i, j,
            gate_round=gate_round, parametrize=parametrize
        )

    @property
    def psi(self):
        """Tensor network representation of the wavefunction.
        """
        # make sure all same dtype and drop singlet dimensions
        psi = self._psi.copy()
        psi.squeeze_()
        psi.astype_(psi.dtype)
        return psi

    def get_uni(self, transposed=False):
        """Tensor network representation of the unitary operator (i.e. with
        the initial state removed).
        """
        U = self.psi

        if transposed:
            # rename the initial state rand_uuid bonds to 1D site inds
            ixmap = {self.ket_site_ind(i): self.bra_site_ind(i)
                     for i in range(self.N)}
        else:
            ixmap = {}

        # the first `N` tensors should be the tensors of input state
        tids = tuple(U.tensor_map)[:self.N]
        for i, tid in enumerate(tids):
            t = U.pop_tensor(tid)
            old_ix, = t.inds

            if transposed:
                ixmap[old_ix] = f'k{i}'
            else:
                ixmap[old_ix] = f'b{i}'

        U.reindex_(ixmap)
        U.view_as_(
            TensorNetworkGenOperator,
            upper_ind_id=self._ket_site_ind_id,
            lower_ind_id=self._bra_site_ind_id,
        )

        return U

    @property
    def uni(self):
        import warnings
        warnings.warn(
            "In future the tensor network returned by ``circ.uni`` will not "
            "be transposed as it is currently, to match the expectation from "
            "``U = circ.uni.to_dense()`` behaving like ``U @ psi``. You can "
            "retain this behaviour with ``circ.get_uni(transposed=True)``.",
            FutureWarning
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

        for i, gate in reversed(tuple(enumerate(self.gates))):
            if gate.label == 'IDEN':
                continue

            if gate.label == 'SWAP':
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
                continue

            regs = set(gate.qubits)
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

    def _maybe_init_storage(self):
        # clear/create the cache if circuit has changed
        if self._sample_n_gates != len(self.gates):
            self._sample_n_gates = len(self.gates)

            # storage
            self._storage.clear()
            self._sampled_conditionals.clear()
            self._marginal_storage_size = 0

    def _get_sliced_contractor(
        self,
        info,
        target_size,
        arrays,
        overhead_warn=2.0,
    ):
        key = ('sliced_contractor', info.eq, target_size)
        if key in self._storage:
            sc = self._storage[key]
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
        self._storage[key] = sc
        return sc

    def get_psi_simplified(
        self,
        seq='ADCRS',
        atol=1e-12,
        equalize_norms=False
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

        key = ('psi_simplified', seq, atol)
        if key in self._storage:
            return self._storage[key].copy()

        psi = self.psi
        # make sure to keep all outer indices
        output_inds = tuple(map(psi.site_ind, range(self.N)))

        # simplify the state and cache it
        psi.full_simplify_(seq=seq, atol=atol, output_inds=output_inds,
                           equalize_norms=equalize_norms)
        self._storage[key] = psi

        # return a copy so we can modify it inplace
        return psi.copy()

    def get_rdm_lightcone_simplified(
        self,
        where,
        seq='ADCRS',
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
        key = ('rdm_lightcone_simplified', tuple(sorted(where)), seq, atol)
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
        rho_lc.full_simplify_(seq=seq, atol=atol, output_inds=output_inds,
                              equalize_norms=equalize_norms)
        self._storage[key] = rho_lc

        # return a copy so we can modify it inplace
        return rho_lc.copy()

    def amplitude(
        self,
        b,
        optimize='auto-hq',
        simplify_sequence='ADCRS',
        simplify_atol=1e-12,
        simplify_equalize_norms=False,
        backend=None,
        dtype='complex128',
        target_size=None,
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
            Contraction path optimizer to use for the amplitude, can be
            a reusable path optimizer as only called once (though path won't be
            cached for later use in that case).
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
            ``'cupy'`` or ``'jax'``. Passed to ``opt_einsum``.
        dtype : str, optional
            Data type to cast the TN to before contraction.
        target_size : None or int, optional
            The largest size of tensor to allow. If specified and any
            contraction involves tensors bigger than this, 'slice' the
            contraction into independent parts and sum them individually.
            Requires ``cotengra`` currently.
        rehearse : bool or "tn", optional
            If ``True``, generate and cache the simplified tensor network and
            contraction path but don't actually perform the contraction.
            Returns a dict with keys ``"tn"`` and ``'info'`` with the tensor
            network that will be contracted and the corresponding contraction
            path if so.
        """
        self._maybe_init_storage()

        if len(b) != self.N:
            raise ValueError(f"Bit-string {b} length does not "
                             f"match number of qubits {self.N}.")

        fs_opts = {
            'seq': simplify_sequence,
            'atol': simplify_atol,
            'equalize_norms': simplify_equalize_norms,
        }

        # get the full wavefunction simplified
        psi_b = self.get_psi_simplified(**fs_opts)

        # fix the output indices to the correct bitstring
        for i, x in zip(range(self.N), b):
            psi_b.isel_({psi_b.site_ind(i): int(x)})

        # perform a final simplification and cast
        psi_b.full_simplify_(**fs_opts)
        psi_b.astype_(dtype)

        if rehearse == "tn":
            return psi_b

        # get the contraction path info
        info = psi_b.contract(
            all, output_inds=(), optimize=optimize, get='path-info'
        )

        if rehearse:
            return rehearsal_dict(psi_b, info)

        if target_size is not None:
            # perform the 'sliced' contraction restricted to ``target_size``
            arrays = tuple(t.data for t in psi_b)
            sc = self._get_sliced_contractor(info, target_size, arrays)
            c_b = sc.contract_all(backend=backend)
        else:
            # perform the full contraction with the path found
            c_b = psi_b.contract(
                all, output_inds=(), optimize=info.path, backend=backend
            )

        return c_b

    def amplitude_rehearse(
        self,
        b='random',
        simplify_sequence='ADCRS',
        simplify_atol=1e-12,
        simplify_equalize_norms=False,
        optimize='auto-hq',
        dtype='complex128',
        rehearse=True,
    ):
        """Perform just the tensor network simplifications and contraction path
        finding associated with computing a single amplitude (caching the
        results) but don't perform the actual contraction.

        Parameters
        ----------
        b : 'random', str or sequence of int
            The bitstring to rehearse computing the transition amplitude for,
            if ``'random'`` (the default) a random bitstring will be used.
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
        simplify_equalize_norms : bool, optional
            Actively renormalize tensor norms during simplification.
        backend : str, optional
            Backend to perform the marginal contraction with, e.g. ``'numpy'``,
            ``'cupy'`` or ``'jax'``. Passed to ``opt_einsum``.
        dtype : str, optional
            Data type to cast the TN to before contraction.

        Returns
        -------
        dict

        """
        if b == 'random':
            import random
            b = [random.choice('01') for _ in range(self.N)]

        return self.amplitude(
            b=b, optimize=optimize, dtype=dtype, rehearse=rehearse,
            simplify_sequence=simplify_sequence,
            simplify_atol=simplify_atol,
            simplify_equalize_norms=simplify_equalize_norms
        )

    amplitude_tn = functools.partialmethod(amplitude_rehearse, rehearse="tn")

    def partial_trace(
        self,
        keep,
        optimize='auto-hq',
        simplify_sequence='ADCRS',
        simplify_atol=1e-12,
        simplify_equalize_norms=False,
        backend=None,
        dtype='complex128',
        target_size=None,
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
            can be a custom path optimizer as only called once (though path
            won't be cached for later use in that case).
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
            ``'cupy'`` or ``'jax'``. Passed to ``opt_einsum``.
        dtype : str, optional
            Data type to cast the TN to before contraction.
        target_size : None or int, optional
            The largest size of tensor to allow. If specified and any
            contraction involves tensors bigger than this, 'slice' the
            contraction into independent parts and sum them individually.
            Requires ``cotengra`` currently.
        rehearse : bool or "tn", optional
            If ``True``, generate and cache the simplified tensor network and
            contraction path but don't actually perform the contraction.
            Returns a dict with keys ``"tn"`` and ``'info'`` with the tensor
            network that will be contracted and the corresponding contraction
            path if so.

        Returns
        -------
        array or dict
        """

        if isinstance(keep, numbers.Integral):
            keep = (keep,)

        output_inds = (tuple(map(self.ket_site_ind, keep)) +
                       tuple(map(self.bra_site_ind, keep)))

        rho = self.get_rdm_lightcone_simplified(
            where=keep, seq=simplify_sequence, atol=simplify_atol,
            equalize_norms=simplify_equalize_norms,
        ).astype_(dtype)

        if rehearse == "tn":
            return rho

        info = rho.contract(
            all,
            output_inds=output_inds,
            optimize=optimize,
            get='path-info'
        )

        if rehearse:
            return rehearsal_dict(rho, info)

        if target_size is not None:
            # perform the 'sliced' contraction restricted to ``target_size``
            arrays = tuple(t.data for t in rho)
            sc = self._get_sliced_contractor(info, target_size, arrays)
            rho_dense = sc.contract_all(backend=backend)
        else:
            # perform the full contraction with the path found
            rho_dense = rho.contract(
                all, output_inds=output_inds,
                optimize=info.path, backend=backend
            ).data

        return ops.reshape(rho_dense, [2**len(keep), 2**len(keep)])

    partial_trace_rehearse = functools.partialmethod(
        partial_trace, rehearse=True)
    partial_trace_tn = functools.partialmethod(
        partial_trace, rehearse="tn")

    def local_expectation(
        self,
        G,
        where,
        optimize='auto-hq',
        simplify_sequence='ADCRS',
        simplify_atol=1e-12,
        simplify_equalize_norms=False,
        backend=None,
        dtype='complex128',
        target_size=None,
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
        G : array or tuple[array] or list[array]
            The raw operator(s) to find the expectation of.
        where : int or sequence of int
            Which qubits the operator acts on.
        optimize : str, optional
            Contraction path optimizer to use for the local expectation,
            can be a custom path optimizer as only called once (though path
            won't be cached for later use in that case).
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
            ``'cupy'`` or ``'jax'``. Passed to ``opt_einsum``.
        dtype : str, optional
            Data type to cast the TN to before contraction.
        target_size : None or int, optional
            The largest size of tensor to allow. If specified and any
            contraction involves tensors bigger than this, 'slice' the
            contraction into independent parts and sum them individually.
            Requires ``cotengra`` currently.
        gate_opts : None or dict_like
            Options to use when applying ``G`` to the wavefunction.
        rehearse : bool or "tn", optional
            If ``True``, generate and cache the simplified tensor network and
            contraction path but don't actually perform the contraction.
            Returns a dict with keys ``'tn'`` and ``'info'`` with the tensor
            network that will be contracted and the corresponding contraction
            path if so.

        Returns
        -------
        scalar, tuple[scalar] or dict
        """
        if isinstance(where, numbers.Integral):
            where = (where,)

        fs_opts = {
            'seq': simplify_sequence,
            'atol': simplify_atol,
            'equalize_norms': simplify_equalize_norms,
        }

        rho = self.get_rdm_lightcone_simplified(where=where, **fs_opts)
        k_inds = tuple(self.ket_site_ind(i) for i in where)
        b_inds = tuple(self.bra_site_ind(i) for i in where)

        if isinstance(G, (list, tuple)):
            # if we have multiple expectations create an extra indexed stack
            nG = len(G)
            G_data = do('stack', G)
            G_data = reshape(G_data, (nG,) + (2,) * 2 * len(where))
            output_inds = (rand_uuid(),)
        else:
            G_data = reshape(G, (2,) * 2 * len(where))
            output_inds = ()

        TG = Tensor(data=G_data, inds=output_inds + b_inds + k_inds)

        rhoG = rho | TG

        rhoG.full_simplify_(output_inds=output_inds, **fs_opts)
        rhoG.astype_(dtype)

        if rehearse == "tn":
            return rhoG

        info = rhoG.contract(
            all,
            output_inds=output_inds,
            optimize=optimize,
            get='path-info'
        )

        if rehearse:
            return rehearsal_dict(rhoG, info)

        if target_size is not None:
            # perform the 'sliced' contraction restricted to ``target_size``
            arrays = tuple(t.data for t in rhoG)
            sc = self._get_sliced_contractor(info, target_size, arrays)
            g_ex = sc.contract_all(backend=backend)
        else:
            g_ex = rhoG.contract(all, output_inds=output_inds,
                                 optimize=info.path, backend=backend)

        if isinstance(g_ex, Tensor):
            g_ex = tuple(g_ex.data)

        return g_ex

    local_expectation_rehearse = functools.partialmethod(
        local_expectation, rehearse=True)
    local_expectation_tn = functools.partialmethod(
        local_expectation, rehearse="tn")

    def compute_marginal(
        self,
        where,
        fix=None,
        optimize='auto-hq',
        backend=None,
        dtype='complex64',
        simplify_sequence='ADCRS',
        simplify_atol=1e-6,
        simplify_equalize_norms=True,
        target_size=None,
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
        simplify_equalize_norms : bool, optional
            Actively renormalize tensor norms during simplification.
        rehearse : bool or "tn", optional
            Whether to perform the marginal contraction or just return the
            associated TN and contraction path information.
        target_size : None or int, optional
            The largest size of tensor to allow. If specified and any
            contraction involves tensors bigger than this, 'slice' the
            contraction into independent parts and sum them individually.
            Requires ``cotengra`` currently.
        """
        self._maybe_init_storage()

        # index trick to contract straight to reduced density matrix diagonal
        # rho_ii -> p_i (i.e. insert a COPY tensor into the norm)
        output_inds = [self.ket_site_ind(i) for i in where]

        fs_opts = {
            'seq': simplify_sequence,
            'atol': simplify_atol,
            'equalize_norms': simplify_equalize_norms,
        }

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
            nm_lc = self.get_rdm_lightcone_simplified(region, **fs_opts)
            # re-connect the ket and bra indices as taking diagonal
            nm_lc.reindex_({
                self.bra_site_ind(i): self.ket_site_ind(i) for i in region
            })

        if fix:
            # project (slice) fixed tensors with bitstring
            # this severs the indices connecting bra and ket on fixed sites
            nm_lc.isel_({self.ket_site_ind(i): int(b) for i, b in fix.items()})

        # having sliced we can do a final simplify
        nm_lc.full_simplify_(output_inds=output_inds, **fs_opts)

        # for stability with very small probabilities, scale by average prob
        if fix is not None:
            nfact = 2**len(fix)
            if final_marginal:
                nm_lc.multiply_(nfact**0.5, spread_over='all')
            else:
                nm_lc.multiply_(nfact, spread_over='all')

        # cast to desired data type
        nm_lc.astype_(dtype)

        if rehearse == "tn":
            return nm_lc

        # NB. the path isn't *neccesarily* the same each time due to the post
        #     slicing full simplify, however there is also the lower level
        #     contraction path cache if the structure generated *is* the same
        #     so still pretty efficient to just overwrite
        info = nm_lc.contract(
            all, output_inds=output_inds,
            optimize=optimize, get='path-info'
        )

        if rehearse:
            return rehearsal_dict(nm_lc, info)

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

        if fix is not None:
            p_marginal = p_marginal / nfact

        return p_marginal

    compute_marginal_rehearse = functools.partialmethod(
        compute_marginal, rehearse=True)
    compute_marginal_tn = functools.partialmethod(
        compute_marginal, rehearse="tn")

    def calc_qubit_ordering(self, qubits=None, method='greedy-lightcone'):
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

        key = ('lightcone_ordering', method, qubits)

        # check the cache first
        if key in self._storage:
            return self._storage[key]

        if method == 'greedy-lightcone':
            cone = set()
            lctgs = {
                i: set(self.get_reverse_lightcone_tags(i))
                for i in qubits
            }

            order = []
            while lctgs:
                # get the next qubit which adds least num gates to lightcone
                next_qubit = min(lctgs, key=lambda i: len(lctgs[i] - cone))
                cone |= lctgs.pop(next_qubit)
                order.append(next_qubit)

        else:
            # use graph distance based hierachical clustering
            psi = self.get_psi_simplified('R')
            qubit_inds = tuple(map(psi.site_ind, qubits))
            tids = psi._get_tids_from_inds(qubit_inds, 'any')
            matcher = re.compile(psi.site_ind_id.format(r'(\d+)'))
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
            tuple(sorted(g))
            for g in partition_all(group_size, order)
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
        backend=None,
        dtype='complex64',
        simplify_sequence='ADCRS',
        simplify_atol=1e-6,
        simplify_equalize_norms=False,
        target_size=None,
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
        simplify_equalize_norms : bool, optional
            Actively renormalize tensor norms during simplification.
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
        self._maybe_init_storage()

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
                        target_size=target_size,
                    )
                    p = do('to_numpy', p).astype('float64')
                    p /= p.sum()

                    if self._marginal_storage_size <= max_marginal_storage:
                        self._sampled_conditionals[key] = p
                        self._marginal_storage_size += p.size
                else:
                    p = self._sampled_conditionals[key]

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
        simplify_equalize_norms=False,
        rehearse=True,
        progbar=False,
    ):
        """Perform the preparations and contraction path findings for
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
        simplify_equalize_norms : bool, optional
            Actively renormalize tensor norms during simplification.
        progbar : bool, optional
            Whether to show the progress of finding each contraction path.

        Returns
        -------
        dict[tuple[int], dict]
            One contraction path info object per grouped marginal computation.
            The keys of the dict are the qubits the marginal is computed for,
            the values are a dict containing a representative simplified tensor
            network (key: 'tn') and the main contraction path info
            (key: 'info').
        """
        # init TN norms, contraction paths, and marginals
        self._maybe_init_storage()
        qubits, order = self._parse_qubits_order(qubits, order)
        groups = self._group_order(order, group_size)

        if result is None:
            result = {q: '0' for q in qubits}

        fix = {}
        tns_and_infos = {}

        for where in _progbar(groups, disable=not progbar):
            tns_and_infos[where] = self.compute_marginal(
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

        return tns_and_infos

    sample_tns = functools.partialmethod(sample_rehearse, rehearse="tn")

    def sample_chaotic(
        self,
        C,
        marginal_qubits,
        max_marginal_storage=2**20,
        seed=None,
        optimize='auto-hq',
        backend=None,
        dtype='complex64',
        simplify_sequence='ADCRS',
        simplify_atol=1e-6,
        simplify_equalize_norms=False,
        target_size=None,
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
        seed : None or int, optional
            A random seed, passed to ``numpy.random.seed`` if given.
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
        simplify_equalize_norms : bool, optional
            Actively renormalize tensor norms during simplification.
        target_size : None or int, optional
            The largest size of tensor to allow. If specified and any
            contraction involves tensors bigger than this, 'slice' the
            contraction into independent parts and sum them individually.
            Requires ``cotengra`` currently.

        Yields
        ------
        str
        """
        # init TN norms, contraction paths, and marginals
        self._maybe_init_storage()
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
                    target_size=target_size,
                )
                p = do('to_numpy', p).astype('float64')
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
        optimize='auto-hq',
        simplify_sequence='ADCRS',
        simplify_atol=1e-6,
        simplify_equalize_norms=False,
        dtype='complex64',
        rehearse=True,
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
            ``'info'`` - the contraction path information.
        """

        # init TN norms, contraction paths, and marginals
        self._maybe_init_storage()
        qubits = tuple(range(self.N))

        if isinstance(marginal_qubits, numbers.Integral):
            marginal_qubits = self.calc_qubit_ordering()[:marginal_qubits]
        where = tuple(sorted(marginal_qubits))

        fix_qubits = tuple(q for q in qubits if q not in where)

        if result is None:
            fix = {q: '0' for q in fix_qubits}
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
        sample_chaotic_rehearse, rehearse="tn")

    def to_dense(
        self,
        reverse=False,
        optimize='auto-hq',
        simplify_sequence='R',
        simplify_atol=1e-12,
        simplify_equalize_norms=False,
        backend=None,
        dtype=None,
        target_size=None,
        rehearse=False,
    ):
        """Generate the dense representation of the final wavefunction.

        Parameters
        ----------
        reverse : bool, optional
            Whether to reverse the order of the subsystems, to match the
            convention of qiskit for example.
        optimize : str, optional
            Contraction path optimizer to use for the contraction, can be
            a single path optimizer as only called once (though path won't be
            cached for later use in that case).
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
            ``'cupy'`` or ``'jax'``. Passed to ``opt_einsum``.
        dtype : str, optional
            Data type to cast the TN to before contraction.
        target_size : None or int, optional
            The largest size of tensor to allow. If specified and any
            contraction involves tensors bigger than this, 'slice' the
            contraction into independent parts and sum them individually.
            Requires ``cotengra`` currently.
        rehearse : bool, optional
            If ``True``, generate and cache the simplified tensor network and
            contraction path but don't actually perform the contraction.
            Returns a dict with keys ``'tn'`` and ``'info'`` with the tensor
            network that will be contracted and the corresponding contraction
            path if so.

        Returns
        -------
        psi : qarray
            The densely represented wavefunction with ``dtype`` data.
        """
        psi = self.get_psi_simplified(
            seq=simplify_sequence,
            atol=simplify_atol,
            equalize_norms=simplify_equalize_norms
        )

        if dtype is not None:
            psi.astype_(dtype)

        if rehearse == "tn":
            return psi

        output_inds = tuple(map(psi.site_ind, range(self.N)))
        if reverse:
            output_inds = output_inds[::-1]

        # get the contraction path info
        info = psi.contract(
            all, output_inds=output_inds, optimize=optimize, get='path-info'
        )

        if rehearse:
            return rehearsal_dict(psi, info)

        if target_size is not None:
            # perform the 'sliced' contraction restricted to ``target_size``
            arrays = tuple(t.data for t in psi)
            sc = self._get_sliced_contractor(info, target_size, arrays)
            psi_tensor = sc.contract_all(backend=backend)
        else:
            # perform the full contraction with the path found
            psi_tensor = psi.contract(
                all, output_inds=output_inds,
                optimize=info.path, backend=backend
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
                p = cache[b] = abs(self.amplitude(b, **amplitude_opts))**2
            psum += cnt * p
            M += cnt

        return (2 ** self.N) / M * psum - 1

    def xeb_ex(
        self,
        optimize='auto-hq',
        simplify_sequence='R',
        simplify_atol=1e-12,
        simplify_equalize_norms=False,
        dtype=None,
        backend=None,
        autojit=False,
        progbar=False,
        **contract_opts
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
            arrays = [do('array', x, like=backend) for x in arrays]

        # perform map-reduce style computation over output wavefunction chunks
        # so we don't need entire wavefunction in memory at same time
        chunks = tree.gen_output_chunks(
            arrays,
            autojit=autojit,
            **contract_opts
        )
        if progbar:
            chunks = _progbar(chunks, total=tree.nchunks)

        def f(chunk):
            return do('sum', do('abs', chunk)**4)

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
        for i, gate in enumerate(self.gates):
            tag = f'GATE_{i}'
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
                self.gates[i] = Gate(
                    label=gate.label,
                    params=t.params,
                    qubits=gate.qubits,
                    round=gate.round,
                    parametrize=True
                )

    @property
    def num_gates(self):
        return len(self.gates)

    def __repr__(self):
        r = "<Circuit(n={}, num_gates={}, gate_opts={})>"
        return r.format(self.N, self.num_gates, self.gate_opts)


class CircuitMPS(Circuit):
    """Quantum circuit simulation keeping the state always in a MPS form. If
    you think the circuit will not build up much entanglement, or you just want
    to keep a rigorous handle on how much entanglement is present, this can
    be useful.
    """

    def __init__(
        self,
        N=None,
        psi0=None,
        gate_opts=None,
        **circuit_opts,
    ):
        gate_opts = ensure_dict(gate_opts)
        gate_opts.setdefault('contract', 'swap+split')
        super().__init__(N, psi0, gate_opts, **circuit_opts)

    def _init_state(self, N, dtype='complex128'):
        return MPS_computational_state('0' * N, dtype=dtype)

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
        gate_opts = ensure_dict(gate_opts)
        gate_opts.setdefault('contract', True)
        super().__init__(N, psi0, gate_opts, tags)

    @property
    def psi(self):
        t = self._psi ^ all
        psi = t.as_network()
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
