"""Gate definitions, registries, and the ``Gate`` class for circuits."""

import cmath
import functools
import math
import numbers

import numpy as np
from autoray import (
    backend_like,
    do,
    reshape,
)

import quimb as qu

from .. import array_ops as ops
from ..tensor_builder import (
    HTN_CP_operator_from_products,
    MPO_identity_like,
)
from ..tensor_core import (
    Tensor,
    rand_uuid,
    tensor_contract,
)
from ..tn1d.core import MatrixProductOperator


def recursive_stack(x):
    if not isinstance(x, (list, tuple)):
        return x
    return do("stack", tuple(map(recursive_stack, x)))


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


def register_special_gate(name, fn, num_qubits, tag=None, array=None):
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
    if array is not None:
        CONSTANT_GATES[name] = array


# constant single qubit gates
register_constant_gate("H", qu.hadamard(), 1)
register_constant_gate("X", qu.pauli("X"), 1)
register_constant_gate("Y", qu.pauli("Y"), 1)
register_constant_gate("Z", qu.pauli("Z"), 1)
register_constant_gate("S", qu.S_gate(), 1)
register_constant_gate("SDG", qu.S_gate().H, 1)
register_constant_gate("T", qu.T_gate(), 1)
register_constant_gate("TDG", qu.T_gate().H, 1)
register_constant_gate("SX", cmath.rect(1, 0.25 * math.pi) * qu.Xsqrt(), 1)
register_constant_gate(
    "SXDG", cmath.rect(1, -0.25 * math.pi) * qu.Xsqrt().H, 1
)
register_constant_gate("X_1_2", qu.Xsqrt(), 1, "X_1/2")
register_constant_gate("Y_1_2", qu.Ysqrt(), 1, "Y_1/2")
register_constant_gate("Z_1_2", qu.Zsqrt(), 1, "Z_1/2")
register_constant_gate("W_1_2", qu.Wsqrt(), 1, "W_1/2")
register_constant_gate("HZ_1_2", qu.Wsqrt(), 1, "W_1/2")


# constant two qubit gates
register_constant_gate("CX", qu.cX(), 2)
register_constant_gate("CNOT", qu.CNOT(), 2, "CX")
register_constant_gate("CY", qu.cY(), 2)
register_constant_gate("CZ", qu.cZ(), 2)
register_constant_gate("ISWAP", qu.iswap(), 2)
register_constant_gate("IS", qu.iswap(), 2, "ISWAP")


# constant three qubit gates
register_constant_gate("CCX", qu.ccX(), 3)
register_constant_gate("CCNOT", qu.ccX(), 3, "CCX")
register_constant_gate("TOFFOLI", qu.ccX(), 3, "CCX")
register_constant_gate("CCY", qu.ccY(), 3)
register_constant_gate("CCZ", qu.ccZ(), 3)
register_constant_gate("CSWAP", qu.cswap(), 3)
register_constant_gate("FREDKIN", qu.cswap(), 3, "CSWAP")


# single parametrizable gates


def rx_gate_param_gen(params):
    phi = params[0]

    with backend_like(phi):
        # get a real backend zero
        zero = phi * 0.0

        c = do("complex", do("cos", phi / 2), zero)
        s = do("complex", zero, -do("sin", phi / 2))

        return recursive_stack(((c, s), (s, c)))


register_param_gate("RX", rx_gate_param_gen, 1)


def ry_gate_param_gen(params):
    phi = params[0]

    with backend_like(phi):
        # get a real backend zero
        zero = phi * 0.0

        c = do("complex", do("cos", phi / 2), zero)
        s = do("complex", do("sin", phi / 2), zero)

        return recursive_stack(((c, -s), (s, c)))


register_param_gate("RY", ry_gate_param_gen, 1)


def rz_gate_param_gen(params):
    phi = params[0]

    with backend_like(phi):
        # get a real backend zero
        zero = phi * 0.0

        c = do("complex", do("cos", phi / 2), zero)
        s = do("complex", zero, -do("sin", phi / 2))

        # get a complex backend zero
        zero = do("complex", zero, zero)

        return recursive_stack(((c + s, zero), (zero, c - s)))


register_param_gate("RZ", rz_gate_param_gen, 1)


def u3_gate_param_gen(params):
    theta, phi, lamda = params[0], params[1], params[2]

    with backend_like(theta):
        # get a real backend zero
        zero = theta * 0.0

        theta_2 = theta / 2
        c2 = do("complex", do("cos", theta_2), zero)
        s2 = do("complex", do("sin", theta_2), zero)
        el = do("exp", do("complex", zero, lamda))
        ep = do("exp", do("complex", zero, phi))
        elp = do("exp", do("complex", zero, lamda + phi))

        return recursive_stack(((c2, -el * s2), (ep * s2, elp * c2)))


register_param_gate("U3", u3_gate_param_gen, 1)


def u2_gate_param_gen(params):
    phi, lamda = params[0], params[1]

    with backend_like(phi):
        # get a real backend zero
        zero = phi * 0.0

        c01 = -do("exp", do("complex", zero, lamda))
        c10 = do("exp", do("complex", zero, phi))
        c11 = do("exp", do("complex", zero, phi + lamda))

        # get a complex backend zero and backend one
        zero = do("complex", zero, zero)
        one = zero + 1.0

        return recursive_stack(((one, c01), (c10, c11))) / 2**0.5


register_param_gate("U2", u2_gate_param_gen, 1)


def u1_gate_param_gen(params):
    lamda = params[0]

    with backend_like(lamda):
        # get a real backend zero
        zero = lamda * 0.0

        c11 = do("exp", do("complex", zero, lamda))

        # get a complex backend zero and backend one
        zero = do("complex", zero, zero)
        one = zero + 1.0

        return recursive_stack(((one, zero), (zero, c11)))


register_param_gate("U1", u1_gate_param_gen, 1)
register_param_gate("PHASE", u1_gate_param_gen, 1)


# two qubit parametrizable gates


def cu3_param_gen(params):
    U3 = u3_gate_param_gen(params)

    with backend_like(U3):
        # get a 'backend zero'
        zero = 0.0 * U3[0, 0]
        # get a 'backend one'
        one = zero + 1.0

        data = (
            (((one, zero), (zero, zero)), ((zero, one), (zero, zero))),
            (
                ((zero, zero), (U3[0, 0], U3[0, 1])),
                ((zero, zero), (U3[1, 0], U3[1, 1])),
            ),
        )

        return recursive_stack(data)


register_param_gate("CU3", cu3_param_gen, 2)


def cu2_param_gen(params):
    U2 = u2_gate_param_gen(params)

    with backend_like(U2):
        # get a 'backend zero'
        zero = 0.0 * U2[0, 0]
        # get a 'backend one'
        one = zero + 1.0

        data = (
            (((one, zero), (zero, zero)), ((zero, one), (zero, zero))),
            (
                ((zero, zero), (U2[0, 0], U2[0, 1])),
                ((zero, zero), (U2[1, 0], U2[1, 1])),
            ),
        )

        return recursive_stack(data)


register_param_gate("CU2", cu2_param_gen, 2)


def cu1_param_gen(params):
    lamda = params[0]

    with backend_like(lamda):
        # get a real backend zero
        zero = 0.0 * lamda

        c11 = do("exp", do("complex", zero, lamda))

        # get a complex backend zero and backend one
        zero = do("complex", zero, zero)
        one = zero + 1.0

        data = (
            (((one, zero), (zero, zero)), ((zero, one), (zero, zero))),
            (((zero, zero), (one, zero)), ((zero, zero), (zero, c11))),
        )

        return recursive_stack(data)


register_param_gate("CU1", cu1_param_gen, 2)
register_param_gate("CPHASE", cu1_param_gen, 2)


def crx_param_gen(params):
    """Parametrized controlled X-rotation."""
    theta = params[0]

    with backend_like(theta):
        # get a real backend zero
        zero = 0.0 * theta

        ccos = do("complex", do("cos", theta / 2), zero)
        csin = do("complex", zero, -do("sin", theta / 2))

        # get a complex backend zero and backend one
        zero = do("complex", zero, zero)
        one = zero + 1.0

        data = (
            (((one, zero), (zero, zero)), ((zero, one), (zero, zero))),
            (((zero, zero), (ccos, csin)), ((zero, zero), (csin, ccos))),
        )

        return recursive_stack(data)


register_param_gate("CRX", crx_param_gen, 2)


def cry_param_gen(params):
    """Parametrized controlled Y-rotation."""
    theta = params[0]

    with backend_like(theta):
        # get a real backend zero
        zero = 0.0 * theta

        ccos = do("complex", do("cos", theta / 2), zero)
        csin = do("complex", do("sin", theta / 2), zero)

        # get a complex backend zero and backend one
        zero = do("complex", zero, zero)
        one = zero + 1.0

        data = (
            (((one, zero), (zero, zero)), ((zero, one), (zero, zero))),
            (((zero, zero), (ccos, -csin)), ((zero, zero), (csin, ccos))),
        )

        return recursive_stack(data)


register_param_gate("CRY", cry_param_gen, 2)


def crz_param_gen(params):
    """Parametrized controlled Z-rotation."""
    theta = params[0]

    with backend_like(theta):
        # get a real backend zero
        zero = 0.0 * theta

        theta_2 = theta / 2
        c = do("complex", do("cos", theta_2), zero)
        s = do("complex", zero, -do("sin", theta_2))

        # get a complex backend zero and backend one
        zero = do("complex", zero, zero)
        one = zero + 1.0

        data = (
            (((one, zero), (zero, zero)), ((zero, one), (zero, zero))),
            (((zero, zero), (c + s, zero)), ((zero, zero), (zero, c - s))),
        )

        return recursive_stack(data)


register_param_gate("CRZ", crz_param_gen, 2)


def fsim_param_gen(params):
    theta, phi = params[0], params[1]

    with backend_like(theta):
        # get a real backend zero
        zero = theta * 0.0

        a = do("complex", do("cos", theta), zero)
        b = do("complex", zero, -do("sin", theta))
        c = do("exp", do("complex", zero, -phi))

        # get a complex backend zero and backend one
        zero = do("complex", zero, zero)
        one = zero + 1.0

        data = (
            (((one, zero), (zero, zero)), ((zero, a), (b, zero))),
            (((zero, b), (a, zero)), ((zero, zero), (zero, c))),
        )

        return recursive_stack(data)


register_param_gate("FSIM", fsim_param_gen, 2)
register_param_gate("FS", fsim_param_gen, 2, "FSIM")


def fsimg_param_gen(params):
    theta, zeta, chi, gamma, phi = (
        params[0],
        params[1],
        params[2],
        params[3],
        params[4],
    )
    """Parametrized, most general number conserving two qubit gate.
    """

    with backend_like(theta):
        # get a real backend zero
        zero = 0.0 * theta

        cos = do("cos", theta)
        sin = do("sin", theta)

        c11 = do("exp", do("complex", zero, -(gamma + zeta))) * do(
            "complex", cos, zero
        )
        c12 = do("exp", do("complex", zero, -(gamma - chi))) * do(
            "complex", zero, -sin
        )
        c21 = do("exp", do("complex", zero, -(gamma + chi))) * do(
            "complex", zero, -sin
        )
        c22 = do("exp", do("complex", zero, -(gamma - zeta))) * do(
            "complex", cos, zero
        )
        c33 = do("exp", do("complex", zero, -(2 * gamma + phi)))

        # get a complex backend zero and backend one
        zero = do("complex", zero, zero)
        one = zero + 1.0

        data = (
            (((one, zero), (zero, zero)), ((zero, c11), (c12, zero))),
            (((zero, c21), (c22, zero)), ((zero, zero), (zero, c33))),
        )

        return recursive_stack(data)


register_param_gate("FSIMG", fsimg_param_gen, 2)


def givens_param_gen(params):
    theta = params[0]

    with backend_like(theta):
        # get a real backend zero
        zero = 0.0 * theta

        a = do("complex", do("cos", theta), zero)
        b = do("complex", do("sin", theta), zero)

        # get a complex backend zero and backend one
        zero = do("complex", zero, zero)
        one = zero + 1.0

        data = (
            (((one, zero), (zero, zero)), ((zero, a), (-b, zero))),
            (((zero, b), (a, zero)), ((zero, zero), (zero, one))),
        )

        return recursive_stack(data)


register_param_gate("GIVENS", givens_param_gen, num_qubits=2)


def givens2_param_gen(params):
    theta, phi = params[0], params[1]

    with backend_like(theta):
        # get a real backend zero
        zero = 0.0 * theta

        a = do("complex", do("cos", theta), zero)
        b = do("exp", do("complex", zero, phi)) * do(
            "complex", do("sin", theta), zero
        )
        b_conj = do("exp", do("complex", zero, -phi)) * do(
            "complex", do("sin", theta), zero
        )

        # get a complex backend zero and backend one
        zero = do("complex", zero, zero)
        one = zero + 1.0

        data = (
            (((one, zero), (zero, zero)), ((zero, a), (-b, zero))),
            (((zero, b_conj), (a, zero)), ((zero, zero), (zero, one))),
        )

        return recursive_stack(data)


register_param_gate("GIVENS2", givens2_param_gen, num_qubits=2)


def xx_plus_yy_param_gen(params):
    theta, beta = params[0], params[1]

    with backend_like(theta):
        # get a real backend zero
        zero = 0.0 * theta
        half_theta = 0.5 * theta

        a = do("complex", do("cos", half_theta), zero)
        b = do("exp", do("complex", zero, beta)) * do(
            "complex", do("sin", half_theta), zero
        )
        b_conj = do("exp", do("complex", zero, -beta)) * do(
            "complex", do("sin", half_theta), zero
        )

        # get a complex backend zero and backend one
        zero = do("complex", zero, zero)
        one = zero + 1.0

        data = (
            (((one, zero), (zero, zero)), ((zero, a), (-1j * b, zero))),
            (((zero, -1j * b_conj), (a, zero)), ((zero, zero), (zero, one))),
        )

        return recursive_stack(data)


register_param_gate("XXPLUSYY", xx_plus_yy_param_gen, num_qubits=2)


def xx_minus_yy_param_gen(params):
    theta, beta = params[0], params[1]

    with backend_like(theta):
        # get a real backend zero
        zero = 0.0 * theta
        half_theta = 0.5 * theta

        a = do("complex", do("cos", half_theta), zero)
        b = do("exp", do("complex", zero, beta)) * do(
            "complex", do("sin", half_theta), zero
        )
        b_conj = do("exp", do("complex", zero, -beta)) * do(
            "complex", do("sin", half_theta), zero
        )

        # get a complex backend zero and backend one
        zero = do("complex", zero, zero)
        one = zero + 1.0

        data = (
            (((a, zero), (zero, -1j * b_conj)), ((zero, one), (zero, zero))),
            (((zero, zero), (one, zero)), ((-1j * b, zero), (zero, a))),
        )

        return recursive_stack(data)


register_param_gate("XXMINUSYY", xx_minus_yy_param_gen, num_qubits=2)


def rxx_param_gen(params):
    r"""Parametrized two qubit XX-rotation.

    .. math::

        \mathrm{RXX}(\theta) = \exp(-i \frac{\theta}{2} X_i X_j)

    """
    theta = params[0]

    with backend_like(theta):
        # get a real 'backend zero'
        zero = 0.0 * theta

        theta_2 = theta / 2
        ccos = do("complex", do("cos", theta_2), zero)
        csin = do("complex", zero, -do("sin", theta_2))

        # get a complex backend zero
        zero = do("complex", zero, zero)

        data = (
            (((ccos, zero), (zero, csin)), ((zero, ccos), (csin, zero))),
            (((zero, csin), (ccos, zero)), ((csin, zero), (zero, ccos))),
        )

        return recursive_stack(data)


register_param_gate("RXX", rxx_param_gen, 2)


def ryy_param_gen(params):
    r"""Parametrized two qubit YY-rotation.

    .. math::

        \mathrm{RYY}(\theta) = \exp(-i \frac{\theta}{2} Y_i Y_j)

    """
    theta = params[0]

    with backend_like(theta):
        # get a real 'backend zero'
        zero = 0.0 * theta

        theta_2 = theta / 2
        ccos = do("complex", do("cos", theta_2), zero)
        csin = do("complex", zero, do("sin", theta_2))

        # get a complex backend zero
        zero = do("complex", zero, zero)

        data = (
            (((ccos, zero), (zero, csin)), ((zero, ccos), (-csin, zero))),
            (((zero, -csin), (ccos, zero)), ((csin, zero), (zero, ccos))),
        )

        return recursive_stack(data)


register_param_gate("RYY", ryy_param_gen, 2)


def rzz_param_gen(params):
    r"""Parametrized two qubit ZZ-rotation.

    .. math::

        \mathrm{RZZ}(\theta) = \exp(-i \frac{\theta}{2} Z_i Z_j)

    """
    theta = params[0]

    with backend_like(theta):
        # get a real 'backend zero'
        zero = 0.0 * theta

        theta_2 = theta / 2
        c00 = c11 = do("complex", do("cos", theta_2), do("sin", -theta_2))
        c01 = c10 = do("complex", do("cos", theta_2), do("sin", theta_2))

        # get a complex backend zero
        zero = do("complex", zero, zero)

        data = (
            (((c00, zero), (zero, zero)), ((zero, c01), (zero, zero))),
            (((zero, zero), (c10, zero)), ((zero, zero), (zero, c11))),
        )

        return recursive_stack(data)


register_param_gate("RZZ", rzz_param_gen, 2)


def su4_gate_param_gen(params):
    """See https://arxiv.org/abs/quant-ph/0308006 - Fig. 7.
    params:
    #     theta1, phi1, lamda1,
    #     theta2, phi2, lamda2,
    #     theta3, phi3, lamda3,
    #     theta4, phi4, lamda4,
    #     t1, t2, t3,
    """

    TA1 = Tensor(u3_gate_param_gen(params[0:3]), ["a1", "a0"])
    TA2 = Tensor(u3_gate_param_gen(params[3:6]), ["b1", "b0"])

    cnot = do(
        "array",
        qu.CNOT().reshape(2, 2, 2, 2),
        like=params,
        dtype=TA1.data.dtype,
    )

    TNOTC1 = Tensor(cnot, ["b2", "a2", "b1", "a1"])
    TRz1 = Tensor(rz_gate_param_gen(params[12:13]), inds=["a3", "a2"])
    TRy2 = Tensor(ry_gate_param_gen(params[13:14]), inds=["b3", "b2"])
    TCNOT2 = Tensor(cnot, ["a5", "b4", "a3", "b3"])
    TRy3 = Tensor(ry_gate_param_gen(params[14:15]), inds=["b5", "b4"])
    TNOTC3 = Tensor(cnot, ["b6", "a6", "b5", "a5"])
    TA3 = Tensor(u3_gate_param_gen(params[6:9]), ["a7", "a6"])
    TA4 = Tensor(u3_gate_param_gen(params[9:12]), ["b7", "b6"])

    return tensor_contract(
        TA1,
        TA2,
        TNOTC1,
        TRz1,
        TRy2,
        TCNOT2,
        TRy3,
        TNOTC3,
        TA3,
        TA4,
        output_inds=["a7", "b7"] + ["a0", "b0"],
        optimize="auto-hq",
    ).data


register_param_gate("SU4", su4_gate_param_gen, 2)


# special non-tensor gates

_MPS_METHODS = {
    "auto-mps",
    "nonlocal",
    "swap+split",
}


def apply_swap(psi, i, j, **gate_opts):
    contract = gate_opts.pop("contract", None)

    if contract not in _MPS_METHODS:
        # just do swap by lazily reindexing
        iind, jind = map(psi.site_ind, (int(i), int(j)))
        psi.reindex_({iind: jind, jind: iind})

    else:
        # tensors are absorbed so propagate_tags is not needed
        gate_opts.pop("propagate_tags", None)

        if contract == "nonlocal":
            psi.gate_nonlocal_(qu.swap(2), (i, j), **gate_opts)
        else:  # {"swap+split", "auto-mps"}:
            psi.swap_sites_with_compress_(i, j, **gate_opts)


register_special_gate("SWAP", apply_swap, 2, array=qu.swap(2))
register_special_gate("IDEN", lambda *_, **__: None, 1, array=qu.identity(2))


def build_controlled_gate_htn(
    ncontrol,
    gate,
    upper_inds,
    lower_inds,
    tags_each=None,
    tags_all=None,
    bond_ind=None,
):
    """Build a low rank hyper tensor network (CP-decomp like) representation of
    a multi controlled gate.
    """
    ngate = len(gate.qubits)
    gate_shape = (2,) * (2 * ngate)
    array = gate.array.reshape(gate_shape)

    I2 = qu.identity(2, dtype=array.dtype)
    IG = qu.identity(2**ngate, dtype=array.dtype).reshape(gate_shape)
    p1 = qu.down(qtype="dop", dtype=array.dtype)  # |1><1|

    array_seqs = [[I2] * ncontrol + [IG], [p1] * ncontrol + [array - IG]]

    # might need to group indices and tags on the target gate if multi-qubit
    if ngate > 1:
        upper_inds = (*upper_inds[:ncontrol], upper_inds[ncontrol:])
        lower_inds = (*lower_inds[:ncontrol], lower_inds[ncontrol:])
        tags_each = (*tags_each[:ncontrol], tags_each[ncontrol:])

    htn = HTN_CP_operator_from_products(
        array_seqs,
        upper_inds=upper_inds,
        lower_inds=lower_inds,
        tags_each=tags_each,
        tags_all=tags_all,
        bond_ind=bond_ind,
    )

    return htn


def _apply_controlled_gate_mps(psi, gate, tags=None, **gate_opts):
    """Apply a multi-controlled gate to a state represented as an MPS."""
    submpo = gate.build_mpo()
    where = sorted((*gate.controls, *gate.qubits))
    psi.gate_with_submpo_(submpo, where, **gate_opts)


def _apply_controlled_gate_htn(
    psi, gate, tags=None, propagate_tags="register", **gate_opts
):
    assert propagate_tags == "register"

    all_qubits = (*gate.controls, *gate.qubits)
    ncontrol = len(gate.controls)
    ngate = len(gate.qubits)
    ntotal = ncontrol + ngate

    upper_inds = [rand_uuid() for _ in range(ntotal)]
    lower_inds = [rand_uuid() for _ in range(ntotal)]
    tags_sequence = [psi.site_tag(i) for i in all_qubits]

    htn = build_controlled_gate_htn(
        ncontrol,
        gate,
        upper_inds=upper_inds,
        lower_inds=lower_inds,
        tags_each=tags_sequence,
        tags_all=tags,
    )

    psi.gate_inds_with_tn_(
        [psi.site_ind(i) for i in all_qubits],
        htn,
        lower_inds,
        upper_inds,
        **gate_opts,
    )


def _apply_controlled_gate_eager(psi, gate, tags=None, **gate_opts):
    """Apply a multi-controlled gate to a state whose gates are eagerly
    contracted (e.g. a dense statevector): insert the low-rank HTN
    representation of the gate, then contract the resulting tensor network
    back into the dense state. This avoids ever forming the full ``2**(2N)``
    dense operator.
    """
    all_qubits = (*gate.controls, *gate.qubits)
    ntotal = len(all_qubits)
    upper_inds = [rand_uuid() for _ in range(ntotal)]
    lower_inds = [rand_uuid() for _ in range(ntotal)]

    htn = build_controlled_gate_htn(
        len(gate.controls),
        gate,
        upper_inds=upper_inds,
        lower_inds=lower_inds,
        tags_each=[psi.site_tag(i) for i in all_qubits],
        tags_all=tags,
    )
    psi.gate_inds_with_tn_(
        inds=[psi.site_ind(i) for i in all_qubits],
        gate=htn,
        gate_inds_inner=lower_inds,
        gate_inds_outer=upper_inds,
    )
    # contract the whole state back to one tensor; strictly only the acted-on
    # sites' region needs contracting, but that requires carefully specifying
    # output_inds (the HTN hyper-index plus bonds to the rest of the state)
    psi.contract_(output_inds=tuple(psi.outer_inds()))


def apply_controlled_gate(
    psi,
    gate,
    tags=None,
    contract="auto-split-gate",
    propagate_tags="register",
    **gate_opts,
):
    if contract in ("auto-mps", "nonlocal"):
        _apply_controlled_gate_mps(psi, gate, tags=tags, **gate_opts)
    elif contract in ("auto-split-gate", "split-gate"):
        _apply_controlled_gate_htn(
            psi, gate, tags=tags, propagate_tags=propagate_tags, **gate_opts
        )
    elif contract is True:
        # eagerly contracted dense state: form HTN gate and contract it in
        _apply_controlled_gate_eager(psi, gate, tags=tags, **gate_opts)
    else:
        raise ValueError(
            f"Contract method '{contract}' not "
            "supported for multi-controlled gates."
        )


@functools.lru_cache(2**15)
def _cached_param_gate_build(fn, params):
    return fn(params)


class Gate:
    """A simple class for storing the details of a quantum circuit gate.

    Parameters
    ----------
    label : str
        The name or 'identifier' of the gate.
    params : Iterable[float]
        The parameters of the gate.
    qubits : Iterable[int], optional
        Which qubits the gate acts on.
    controls : Iterable[int], optional
        Which qubits are the controls.
    round : int, optional
        If given, which round or layer the gate is part of.
    parametrize : bool, optional
        Whether the gate will correspond to a parametrized tensor.
    """

    __slots__ = (
        "_label",
        "_params",
        "_qubits",
        "_controls",
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
        qubits=None,
        controls=None,
        round=None,
        parametrize=False,
    ):
        self._label = label.upper()

        if self._label not in ALL_GATES:
            raise ValueError(f"Unknown gate: {self._label}.")

        self._params = ops.asarray(params)
        if qubits is None:
            self._qubits = None
        else:
            self._qubits = tuple(qubits)

        if controls is None:
            self._controls = None
        else:
            self._controls = tuple(controls)

        self._round = int(round) if round is not None else round
        self._parametrize = bool(parametrize)

        self._tag = GATE_TAGS[self._label]
        self._special = self._label in SPECIAL_GATES
        self._constant = self._label in CONSTANT_GATES
        if (self._special or self._constant) and self._parametrize:
            raise ValueError(f"Cannot parametrize the gate: {self._label}.")
        self._array = None

    @classmethod
    def from_raw(cls, U, qubits=None, controls=None, round=None):
        new = object.__new__(cls)
        new._label = f"RAW{id(U)}"
        new._params = "raw"
        if qubits is None:
            new._qubits = None
        else:
            new._qubits = tuple(qubits)
        if controls is None:
            new._controls = None
        else:
            new._controls = tuple(controls)
        new._round = int(round) if round is not None else round
        new._special = False
        new._parametrize = isinstance(U, ops.PArray)
        new._tag = None
        new._array = U
        return new

    def copy(self):
        new = object.__new__(self.__class__)
        new._label = self._label
        new._params = self._params
        new._qubits = self._qubits
        new._controls = self._controls
        new._round = self._round
        new._parametrize = self._parametrize
        new._tag = self._tag
        new._special = self._special
        new._constant = self._constant
        new._array = self._array
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

    @qubits.setter
    def qubits(self, qubits):
        if qubits is None:
            self._qubits = None
        else:
            self._qubits = tuple(qubits)

    @property
    def total_qubit_count(self):
        nq = len(self._qubits)
        if self._controls:
            nq += len(self._controls)
        return nq

    @property
    def controls(self):
        return self._controls

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

    def copy_with(self, **kwargs):
        """Take a copy of this gate but with some attributes changed."""
        label = kwargs.get("label", self._label)
        params = kwargs.get("params", self._params)
        qubits = kwargs.get("qubits", self._qubits)
        controls = kwargs.get("controls", self._controls)
        round = kwargs.get("round", self._round)
        parametrize = kwargs.get("parametrize", self._parametrize)

        if isinstance(params, str) and (params == "raw"):
            return self.from_raw(
                U=self._array,
                qubits=qubits,
                controls=controls,
                round=round,
            )
        else:
            return self.__class__(
                label=label,
                params=params,
                qubits=qubits,
                controls=controls,
                round=round,
                parametrize=parametrize,
            )

    def build_array(self):
        """Build the array representation of the gate. For controlled gates
        this *excludes* the control qubits.
        """
        if self._special and (self._label not in CONSTANT_GATES):
            # these don't have an array representation
            raise ValueError(f"{self.label} gates have no array to build.")

        if self._constant:
            # simply return the constant array
            return CONSTANT_GATES[self._label]

        # build the array
        param_fn = PARAM_GATES[self._label]
        if self._parametrize:
            # either lazily, as tensor will be parametrized
            shape = (2,) * (2 * len(self._qubits))
            return ops.PArray(param_fn, self._params, shape=shape)

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

    def build_mpo(self, L=None, **kwargs):
        """Build an MPO representation of this gate."""
        G = self.array

        if L is None:
            L = max((*self.qubits, *self.controls), default=0) + 1

        if not self.controls:
            return MatrixProductOperator.from_dense(
                G, sites=self.qubits, L=L, **kwargs
            )

        IG = qu.identity(2 ** len(self.qubits))
        IG = reshape(IG, G.shape)
        p1 = qu.down(qtype="dop")

        # form (G - 1) on target qubits
        mpo = MatrixProductOperator.from_dense(
            G - IG, sites=self.qubits, L=L, **kwargs
        )

        # take tensor product with |11...><11...| on controls
        mpo.fill_empty_sites_(mode=self.controls, fill_array=p1)

        # add with identity on all qubits
        mpo_I = MPO_identity_like(
            mpo, sites=sorted((*self.qubits, *self.controls))
        )

        return mpo.add_MPO_(mpo_I)

    def __repr__(self):
        return (
            f"<{self.__class__.__name__}("
            + f"label={self._label}, "
            + f"params={self._params}, "
            + f"qubits={self._qubits}"
            + (f", controls={self._controls})" if self._controls else "")
            + (f", round={self._round}" if self._round is not None else "")
            + (
                f", parametrize={self._parametrize})"
                if self._parametrize
                else ""
            )
            + ")>"
        )


def sample_bitstring_from_prob_ndarray(p, seed=None):
    """Sample a bitstring from n-dimensional tensor ``p`` of probabilities.

    Examples
    --------

        >>> import numpy as np
        >>> p = np.zeros(shape=(2, 2, 2, 2, 2))
        >>> p[0, 1, 0, 1, 1] = 1.0
        >>> sample_bitstring_from_prob_ndarray(p)
        '01011'
    """
    rng = np.random.default_rng(seed)
    b = rng.choice(p.size, p=p.ravel())
    return f"{b:0>{p.ndim}b}"


def rehearsal_dict(tn, tree):
    return {
        "tn": tn,
        "tree": tree,
        "W": tree.contraction_width(),
        "C": math.log10(max(tree.contraction_cost(), 1)),
    }


def parse_to_gate(
    gate_id,
    *gate_args,
    params=None,
    qubits=None,
    controls=None,
    gate_round=None,
    parametrize=None,
):
    """Map all types of gate specification into a `Gate` object."""

    if isinstance(gate_id, Gate):
        # already a gate
        if gate_args:
            raise ValueError(
                "You cannot specify ``gate_args`` for an already "
                "encapsulated `Gate` object."
            )

        if any((params, qubits, controls, gate_round, parametrize)):
            raise ValueError(
                "You cannot specify ``controls`` or ``gate_round`` for an "
                "already encapsulated gate - supply directly to the  `Gate` "
                "constructor instead."
            )
        return gate_id

    if isinstance(gate_id, tuple):
        # if given a tuple just unpack it
        if gate_args:
            raise ValueError(
                "You cannot specify ``gate_args`` when supplying a tuple."
            )
        gate_id, gate_args = gate_id[0], gate_id[1:]

    if hasattr(gate_id, "shape") and not isinstance(gate_id, str):
        # raw gate (numpy strings have a shape - ignore those)

        if parametrize is not None:
            raise ValueError(
                "You cannot specify ``parametrize`` for raw gate, supply a "
                "``PArray`` instead."
            )

        return Gate.from_raw(
            U=gate_id,
            qubits=gate_args,
            controls=controls,
            round=gate_round,
        )

    # else gate is specified as a tuple or kwargs

    if isinstance(gate_id, numbers.Integral) or gate_id.isdigit():
        # gate round given as first entry of tuple
        if gate_round is None:
            # explicilty specified ``gate_round`` takes precedence
            gate_round = gate_id
        gate_id, gate_args = gate_args[0], gate_args[1:]

    if parametrize is None:
        parametrize = False

    if gate_args:
        if any((params, qubits)):
            raise ValueError(
                "You cannot specify ``params`` or ``qubits`` "
                "when supplying ``gate_args``."
            )

        nq = GATE_SIZE[gate_id.upper()]
        (
            params,
            qubits,
        ) = (
            gate_args[:-nq],
            gate_args[-nq:],
        )

    else:
        # qubits and params specified directly
        if params is None:
            params = ()

    return Gate(
        label=gate_id,
        params=params,
        qubits=qubits,
        controls=controls,
        round=gate_round,
        parametrize=parametrize,
    )
