"""A convenience import file so that:

    from quimb.gates import *

gives instant access to the standard gate set {I, X, Y, ...}.
"""
import functools
from .gen import operators

I, X, Y, Z = map(operators.pauli, 'IXYZ')
H = operators.hadamard()
S = operators.S_gate()
T = operators.T_gate()

RX = functools.partial(operators.rotation, xyz='x')
RY = functools.partial(operators.rotation, xyz='y')
RZ = functools.partial(operators.rotation, xyz='z')

CNOT = operators.CNOT()
CX = operators.cX()
CY = operators.cY()
CZ = operators.cZ()

SWAP = operators.swap()
ISWAP = operators.iswap()

CCX = operators.ccX()
CCY = operators.ccY()
CCZ = operators.ccZ()

CSWAP = operators.cswap()
TOFFOLI = operators.toffoli()
FREDKIN = operators.fredkin()

U3 = operators.U_gate
fsim = operators.fsim
