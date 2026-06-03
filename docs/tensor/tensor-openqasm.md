# Importing OpenQASM Circuits

`quimb` supports importing both OpenQASM 2 and a practical subset of
OpenQASM 3 into [`Circuit`](quimb.tensor.circuit.Circuit) objects.

## OpenQASM 3 entry points

Use the classmethods on [`Circuit`](quimb.tensor.circuit.Circuit):

- [`Circuit.from_openqasm3_str`](quimb.tensor.circuit.Circuit.from_openqasm3_str)
- [`Circuit.from_openqasm3_file`](quimb.tensor.circuit.Circuit.from_openqasm3_file)
- [`Circuit.from_openqasm3_url`](quimb.tensor.circuit.Circuit.from_openqasm3_url)

These parse the source and immediately build a circuit with the imported
gates:

```python
import quimb.tensor as qtn

qasm = """
OPENQASM 3.0;
include "stdgates.inc";

qubit[2] q;
h q[0];
cx q[0], q[1];
"""

circ = qtn.Circuit.from_openqasm3_str(qasm)
print(circ)
```

## Supported OpenQASM 3 subset

The parser focuses on circuit input. It currently does not support the full QASM 3.0 language. Supported features include:

- `qubit` declarations
- `input` declarations for symbolic gate parameters
- arithmetic expressions such as `pi / 2` or `theta / 2`
- `const`, scalar classical declarations, and array declarations used in
  parameter expressions
- custom gate definitions, including nested custom gates
- register broadcasting such as `h q;` or `cx q, r;`

Ignored with warnings:

- `measure`
- `barrier`
- `gphase`
- plain `bit` declarations used only for classical measurement targets

Unsupported constructs raise `NotImplementedError`, including:

- `output` declarations
- control flow such as `if`, `for`, and `while`
- `reset`
- calibration-related constructs

## Symbolic inputs and rebinding

OpenQASM 3 `input` declarations are preserved on the returned circuit so that
named values can be bound later with [`Circuit.set_params`](quimb.tensor.circuit.Circuit.set_params):

```python
import quimb.tensor as qtn

qasm = """
OPENQASM 3.0;
include "stdgates.inc";

input float theta;
qubit[1] q;
rx(theta) q[0];
"""

circ = qtn.Circuit.from_openqasm3_str(qasm)
circ.set_params({"theta": 0.3})
```

The imported circuit tracks:

- `circ.qasm3_inputs`: declared input names
- `circ.qasm3_symbols`: current symbolic or numeric values
- `circ.qasm3_expressions`: symbolic expressions attached to parameterized gates

All declared inputs must be supplied when binding by name, and unknown names
raise an error.

## OpenQASM 2 compatibility

Equivalent OpenQASM 2 import helpers remain available:

- [`Circuit.from_openqasm2_str`](quimb.tensor.circuit.Circuit.from_openqasm2_str)
- [`Circuit.from_openqasm2_file`](quimb.tensor.circuit.Circuit.from_openqasm2_file)
- [`Circuit.from_openqasm2_url`](quimb.tensor.circuit.Circuit.from_openqasm2_url)

Where the same gate subset is used, the OpenQASM 2 and OpenQASM 3 import paths
produce matching circuits.
