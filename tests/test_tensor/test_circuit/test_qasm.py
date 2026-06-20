import math

import numpy as np
import pytest
from numpy.testing import assert_allclose

import quimb.tensor as qtn
from quimb.tensor.circuit import (
    parse_openqasm3_file,
    parse_openqasm3_str,
    parse_openqasm3_url,
    rx_gate_param_gen,
)
from quimb.tensor.interface import pack, unpack

from ._helpers import graph_to_qsim, rand_reg_graph


def assert_same_gates(circ_a, circ_b):
    assert len(circ_a.gates) == len(circ_b.gates)
    for gate_a, gate_b in zip(circ_a.gates, circ_b.gates):
        assert gate_a.label == gate_b.label
        assert tuple(gate_a.qubits) == tuple(gate_b.qubits)
        assert len(gate_a.params) == len(gate_b.params)
        for pa, pb in zip(gate_a.params, gate_b.params):
            if np.isnan(pa) and np.isnan(pb):
                continue
            assert pa == pytest.approx(pb)


def example_openqasm2_qft():
    return """
    // quantum Fourier transform

    OPENQASM 2.0;
    include "qelib1.inc";

    qreg q[4];
    creg c[4];
    x q[0];
    x q[2];
    barrier q;
    h q[0];
    cu1(pi/2) q[1],q[0];
    h q[1];
    cu1(pi/4) q[2],q[0];
    cu1(pi/2) q[2],q[1];
    /*
    This is a multi line comment.
    */
    h q[2];
    cu1(pi/8) q[3],q[0];
    cu1(pi/4) q[3],q[1];
    cu1(pi/2) q[3],q[2];
    h q[3];

    measure q -> c;
    """


def example_openqasm3_qft():
    return """
    // quantum Fourier transform

    OPENQASM 3.0;
    include "stdgates.inc";

    qubit[4] q;
    bit[4] c;
    x q[0];
    x q[2];
    barrier q;
    h q[0];
    cu1(pi/2) q[1], q[0];
    h q[1];
    cu1(pi/4) q[2], q[0];
    cu1(pi/2) q[2], q[1];
    /*
    This is a multi line comment.
    */
    h q[2];
    cu1(pi/8) q[3], q[0];
    cu1(pi/4) q[3], q[1];
    cu1(pi/2) q[3], q[2];
    h q[3];

    measure q -> c;
    """


class TestCircuitQASM:
    def test_from_qsim(self):
        G = rand_reg_graph(reg=3, n=18, seed=42)
        qsim = graph_to_qsim(G)
        qc = qtn.Circuit.from_qsim_str(qsim)
        assert (qc.psi.H & qc.psi) ^ all == pytest.approx(1.0)

    def test_from_openqasm2(self):
        with pytest.warns(SyntaxWarning) as record:
            qc = qtn.Circuit.from_openqasm2_str(example_openqasm2_qft())
        assert [str(w.message) for w in record] == [
            "Unsupported operation ignored: creg",
            "Unsupported operation ignored: barrier",
            "Unsupported operation ignored: measure",
        ]
        assert (qc.psi.H & qc.psi) ^ all == pytest.approx(1.0)

    def test_from_openqasm3(self):
        with pytest.warns(SyntaxWarning) as record:
            qc = qtn.Circuit.from_openqasm3_str(example_openqasm3_qft())
        assert [str(w.message) for w in record] == [
            "Unsupported operation ignored: bit",
            "Unsupported operation ignored: barrier",
            "Unsupported operation ignored: measure",
        ]
        assert (qc.psi.H & qc.psi) ^ all == pytest.approx(1.0)

    def test_openqasm2_openqasm3_shared_subset_match(self):
        with pytest.warns(SyntaxWarning):
            circ2 = qtn.Circuit.from_openqasm2_str(example_openqasm2_qft())
        with pytest.warns(SyntaxWarning):
            circ3 = qtn.Circuit.from_openqasm3_str(example_openqasm3_qft())
        assert_same_gates(circ2, circ3)
        assert_allclose(circ2.psi.to_dense(), circ3.psi.to_dense())

    def test_openqasm3_symbolic_params(self):
        qasm_str = """
        OPENQASM 3.0;
        include "stdgates.inc";
        input float theta;
        qubit[1] q;
        rx(theta) q[0];
        """
        circ = qtn.Circuit.from_openqasm3_str(qasm_str)
        assert circ.named_param_names == ("theta",)
        assert np.isnan(circ.named_params["theta"])
        assert circ.param_expressions == {0: ("theta",)}
        assert circ.gates[0].label == "RX"
        assert math.isnan(circ.gates[0].params[0])

        circ.set_params({"theta": 0.3})
        assert circ.named_params["theta"] == pytest.approx(0.3)
        assert_allclose(
            np.asarray(circ.psi["GATE_0"].data),
            np.asarray(rx_gate_param_gen((0.3,))),
        )

    def test_openqasm3_custom_gates(self):
        circ = qtn.Circuit.from_openqasm3_str(
            """
        OPENQASM 3.0;
        include "stdgates.inc";
        input float theta;
        qubit[3] q;

        gate hello a, b {
            h a;
            cx a, b;
            u3(theta, 0.2, 0.3) b;
        }

        hello q[0], q[1];
        hello q[2], q[1];
        """
        )
        assert [g.label for g in circ.gates] == [
            "H",
            "CX",
            "U3",
            "H",
            "CX",
            "U3",
        ]
        assert circ.param_expressions[2][0] == "theta"

    def test_openqasm3_named_param_binding_and_copy(self):
        circ = qtn.Circuit.from_openqasm3_str(
            """
        OPENQASM 3.0;
        include "stdgates.inc";
        input float theta;
        qubit[2] q;
        rx(theta) q[0];
        ry(theta / 2) q[1];
        """
        )
        circ2 = circ.copy()

        assert circ2.named_param_names == ("theta",)
        assert np.isnan(circ2.named_params["theta"])
        assert circ2.param_expressions == {
            0: ("theta",),
            1: ("(theta / 2)",),
        }

        circ2.set_params({"theta": 0.6})
        assert circ2.gates[0].params == (pytest.approx(0.6),)
        assert circ2.gates[1].params == (pytest.approx(0.3),)
        assert math.isnan(circ.gates[0].params[0])
        assert math.isnan(circ.gates[1].params[0])
        assert np.isnan(circ.named_params["theta"])

        circ2.set_params({"theta": 0.2})
        assert circ2.gates[0].params == (pytest.approx(0.2),)
        assert circ2.gates[1].params == (pytest.approx(0.1),)

    def test_openqasm3_broadcast_registers(self):
        circ = qtn.Circuit.from_openqasm3_str(
            """
            OPENQASM 3.0;
            include "stdgates.inc";
            qubit[3] q;
            qubit[3] r;
            h q;
            cx q, r;
            """
        )
        assert [(g.label, g.qubits) for g in circ.gates] == [
            ("H", (0,)),
            ("H", (1,)),
            ("H", (2,)),
            ("CX", (0, 3)),
            ("CX", (1, 4)),
            ("CX", (2, 5)),
        ]

    def test_openqasm3_numeric_parse(self):
        circ = qtn.Circuit.from_openqasm3_str(
            """
            OPENQASM 3.0;
            include "stdgates.inc";
            qubit[1] q;
            rx(pi / 2) q[0];
            """
        )
        assert circ.gates[0].params == (pytest.approx(math.pi / 2),)

    def test_openqasm3_parse_file_and_url(self, tmp_path):
        qasm_str = """
        OPENQASM 3.0;
        include "stdgates.inc";
        input float theta;
        qubit[2] q;
        rx(theta) q[0];
        cx q[0], q[1];
        """
        qasm_file = tmp_path / "example.qasm"
        qasm_file.write_text(qasm_str)

        info_str = parse_openqasm3_str(qasm_str)
        info_file = parse_openqasm3_file(qasm_file)
        info_url = parse_openqasm3_url(qasm_file.as_uri())

        assert info_file["n"] == info_str["n"] == info_url["n"]
        assert (
            info_file["n_gates"] == info_str["n_gates"] == info_url["n_gates"]
        )
        assert info_file["inputs"] == info_str["inputs"] == info_url["inputs"]
        assert (
            info_file["symbols"] == info_str["symbols"] == info_url["symbols"]
        )
        assert (
            info_file["expressions"]
            == info_str["expressions"]
            == info_url["expressions"]
        )
        for parsed in (info_file, info_url):
            assert_same_gates(
                qtn.Circuit.from_gates(info_str["gates"], N=info_str["n"]),
                qtn.Circuit.from_gates(parsed["gates"], N=parsed["n"]),
            )

    def test_openqasm3_circuit_from_file_and_url(self, tmp_path):
        qasm_str = """
        OPENQASM 3.0;
        include "stdgates.inc";
        input float theta;
        qubit[2] q;
        rx(theta) q[0];
        cx q[0], q[1];
        """
        qasm_file = tmp_path / "example.qasm"
        qasm_file.write_text(qasm_str)

        circ_str = qtn.Circuit.from_openqasm3_str(qasm_str)
        circ_file = qtn.Circuit.from_openqasm3_file(qasm_file)
        circ_url = qtn.Circuit.from_openqasm3_url(qasm_file.as_uri())

        assert_same_gates(circ_str, circ_file)
        assert_same_gates(circ_str, circ_url)
        assert (
            circ_file.named_param_names
            == circ_str.named_param_names
            == circ_url.named_param_names
        )
        assert (
            set(circ_file.named_params)
            == set(circ_str.named_params)
            == set(circ_url.named_params)
        )
        assert (
            circ_file.param_expressions
            == circ_str.param_expressions
            == circ_url.param_expressions
        )

    def test_openqasm3_named_binding_supports_partial_updates(self):
        circ = qtn.Circuit.from_openqasm3_str(
            """
            OPENQASM 3.0;
            include "stdgates.inc";
            input float theta;
            input float phi;
            qubit[1] q;
            u3(theta, phi, 0.0) q[0];
            """
        )
        circ.set_params({"theta": 0.2})
        assert circ.gates[0].params[0] == pytest.approx(0.2)
        assert math.isnan(circ.gates[0].params[1])
        assert circ.gates[0].params[2] == pytest.approx(0.0)

        circ.set_params({"phi": 0.4})
        assert tuple(circ.gates[0].params) == pytest.approx((0.2, 0.4, 0.0))

    def test_openqasm3_named_binding_empty_dict_preserves_state(self):
        circ = qtn.Circuit.from_openqasm3_str(
            """
            OPENQASM 3.0;
            include "stdgates.inc";
            input float theta;
            qubit[1] q;
            rx(theta) q[0];
            """
        )
        circ.set_params({})
        assert math.isnan(circ.gates[0].params[0])

    def test_openqasm3_array_index_symbolic_binding(self):
        circ = qtn.Circuit.from_openqasm3_str(
            """
            OPENQASM 3.0;
            include "stdgates.inc";
            input float theta;
            array[float, 2] angles = {theta, theta / 2};
            qubit[2] q;
            rx(angles[0]) q[0];
            ry(angles[1]) q[1];
            """
        )
        assert circ.param_expressions == {
            0: ("theta",),
            1: ("(theta / 2)",),
        }

        circ.set_params({"theta": 0.6})
        assert tuple(circ.gates[0].params) == pytest.approx((0.6,))
        assert tuple(circ.gates[1].params) == pytest.approx((0.3,))

    def test_openqasm3_named_binding_rejects_unknown_inputs(self):
        circ = qtn.Circuit.from_openqasm3_str(
            """
            OPENQASM 3.0;
            include "stdgates.inc";
            input float theta;
            qubit[1] q;
            rx(theta) q[0];
            """
        )
        with pytest.raises(ValueError, match="Unknown named parameter values"):
            circ.set_params({"theta": 0.2, "phi": 0.3})

    def test_openqasm3_named_binding_accepts_mixed_keys(self):
        circ = qtn.Circuit.from_openqasm3_str(
            """
            OPENQASM 3.0;
            include "stdgates.inc";
            input float theta;
            qubit[2] q;
            rx(theta) q[0];
            """
        )
        circ.u3(0.1, 0.2, 0.3, 1, parametrize=True)
        circ.set_params({"theta": 0.2, 1: (0.4, 0.5, 0.6)})
        assert tuple(circ.gates[0].params) == pytest.approx((0.2,))
        assert tuple(circ.gates[1].params) == pytest.approx((0.4, 0.5, 0.6))

    def test_openqasm3_get_set_params_roundtrip(self):
        circ = qtn.Circuit.from_openqasm3_str(
            """
            OPENQASM 3.0;
            include "stdgates.inc";
            input float theta;
            qubit[2] q;
            rx(theta) q[0];
            """
        )
        circ.u3(0.1, 0.2, 0.3, 1, parametrize=True)
        circ.set_params({"theta": 0.2, 1: (0.4, 0.5, 0.6)})

        params = circ.get_params()
        assert params["theta"] == pytest.approx(0.2)
        assert tuple(params[1]) == pytest.approx((0.4, 0.5, 0.6))

        circ2 = circ.copy()
        circ2.set_params(params)
        assert tuple(circ2.gates[0].params) == pytest.approx((0.2,))
        assert tuple(circ2.gates[1].params) == pytest.approx((0.4, 0.5, 0.6))

    def test_openqasm3_named_params_pack_unpack_roundtrip(self):
        circ = qtn.Circuit.from_openqasm3_str(
            """
            OPENQASM 3.0;
            include "stdgates.inc";
            input float theta;
            qubit[1] q;
            rx(cos(theta / 2)) q[0];
            """
        )
        circ.set_params({"theta": np.array(0.6)})

        params, skeleton = pack(circ)
        assert params["theta"] == pytest.approx(0.6)

        circ2 = unpack({"theta": np.array(0.2)}, skeleton)
        assert tuple(circ2.gates[0].params) == pytest.approx((math.cos(0.1),))

    def test_openqasm3_named_binding_rejects_direct_managed_gate_override(
        self,
    ):
        circ = qtn.Circuit.from_openqasm3_str(
            """
            OPENQASM 3.0;
            include "stdgates.inc";
            input float theta;
            qubit[1] q;
            rx(theta) q[0];
            """
        )
        with pytest.raises(
            ValueError, match="managed by named parameter expressions"
        ):
            circ.set_params({"theta": 0.2, 0: (0.1,)})

    def test_openqasm3_output_decl_unsupported(self):
        with pytest.raises(NotImplementedError, match="Output declarations"):
            qtn.Circuit.from_openqasm3_str(
                """
                OPENQASM 3.0;
                output float theta;
                qubit[1] q;
                """
            )

    def test_openqasm3_unsupported_ops(self):
        with pytest.raises(NotImplementedError):
            qtn.Circuit.from_openqasm3_str(
                """
                OPENQASM 3.0;
                qubit[1] q;
                reset q[0];
                """
            )

    def test_openqasm3_measure_assignment_warns(self):
        with pytest.warns(SyntaxWarning) as record:
            circ = qtn.Circuit.from_openqasm3_str(
                """
                OPENQASM 3.0;
                include "stdgates.inc";
                qubit[1] q;
                bit c;
                c = measure q[0];
                x q[0];
                """
            )
        assert [str(w.message) for w in record] == [
            "Unsupported operation ignored: bit",
            "Unsupported operation ignored: measure",
        ]
        assert [g.label for g in circ.gates] == ["X"]

    def test_openqasm3_measure_decl_initializer_warns(self):
        with pytest.warns(SyntaxWarning) as record:
            circ = qtn.Circuit.from_openqasm3_str(
                """
                OPENQASM 3.0;
                include "stdgates.inc";
                qubit[1] q;
                bit c = measure q[0];
                x q[0];
                """
            )
        assert [str(w.message) for w in record] == [
            "Unsupported operation ignored: measure"
        ]
        assert [g.label for g in circ.gates] == ["X"]

    def test_openqasm3_custom_gates_overlapping_names(self):
        circ = qtn.Circuit.from_openqasm3_str(
            """
            OPENQASM 3.0;
            include "stdgates.inc";
            input float a;
            qubit[1] q;

            gate foo(a, aa) q {
                u3(aa, a, aa) q;
            }

            foo(0.1, a) q[0];
            """
        )
        assert circ.param_expressions == {0: ("a", 0.1, "a")}
        circ.set_params({"a": 0.2})
        assert tuple(circ.gates[0].params) == pytest.approx((0.2, 0.1, 0.2))

    def test_openqasm3_custom_gates_match_openqasm2(self):
        circ2 = qtn.Circuit.from_openqasm2_str(
            """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[3];

        gate hello a, b {
            h a;
            cx a, b;
            u3(0.1, 0.2, 0.3) b;
        }

        gate world(param1, theta) q {
            u2(theta / 2, param1) q;
            u2(param1, theta / 2) q;
        }

        hello q[0], q[1];
        world(0.1, 0.2) q[2];
        hello q[2], q[1];
        """
        )
        circ3 = qtn.Circuit.from_openqasm3_str(
            """
        OPENQASM 3.0;
        include "stdgates.inc";
        qubit[3] q;

        gate hello a, b {
            h a;
            cx a, b;
            u3(0.1, 0.2, 0.3) b;
        }

        gate world(param1, theta) q {
            u2(theta / 2, param1) q;
            u2(param1, theta / 2) q;
        }

        hello q[0], q[1];
        world(0.1, 0.2) q[2];
        hello q[2], q[1];
        """
        )
        assert_same_gates(circ2, circ3)

    def test_openqasm3_nested_custom_gates_match_openqasm2(self):
        circ2 = qtn.Circuit.from_openqasm2_str(
            """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[3];

        gate cphase(theta) a, b {
            U3(0, 0, theta / 2) a;
            CX a, b;
            U3(0, 0, -theta / 2) b;
            CX a, b;
            U3(0, 0, theta / 2) b;
        }

        gate doublecphase(theta) a, b, c {
            cphase(theta) a, b;
            cphase(theta) b, c;
        }

        doublecphase(0.1) q[0], q[1], q[2];
        doublecphase(0.2) q[2], q[0], q[1];
        """
        )
        circ3 = qtn.Circuit.from_openqasm3_str(
            """
        OPENQASM 3.0;
        include "stdgates.inc";
        qubit[3] q;

        gate cphase(theta) a, b {
            U3(0, 0, theta / 2) a;
            CX a, b;
            U3(0, 0, -theta / 2) b;
            CX a, b;
            U3(0, 0, theta / 2) b;
        }

        gate doublecphase(theta) a, b, c {
            cphase(theta) a, b;
            cphase(theta) b, c;
        }

        doublecphase(0.1) q[0], q[1], q[2];
        doublecphase(0.2) q[2], q[0], q[1];
        """
        )
        assert_same_gates(circ2, circ3)

    def test_openqasm3_gate_name_match_openqasm2(self):
        circ2 = qtn.Circuit.from_openqasm2_str(
            """
        OPENQASM 2.0;
        include "qelib1.inc";
        gate gate_PauliEvolution(param0) q0,q1 { rz(0.2) q0; rz(-0.1) q1; }
        qreg q[2];
        gate_PauliEvolution(0.1) q[0],q[1];
        """
        )
        circ3 = qtn.Circuit.from_openqasm3_str(
            """
        OPENQASM 3.0;
        include "stdgates.inc";
        gate gate_PauliEvolution(param0) q0,q1 { rz(0.2) q0; rz(-0.1) q1; }
        qubit[2] q;
        gate_PauliEvolution(0.1) q[0],q[1];
        """
        )
        assert_same_gates(circ2, circ3)

    def test_openqasm2_custom_gates(self):
        circ = qtn.Circuit.from_openqasm2_str(
            """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[3];

        gate hello a, b {
            h a;
            cx a, b;
            u3(0.1, 0.2, 0.3) b;
        }

        gate world(param1, θ) q
        {
            u2(θ / 2, param1) q;
            u2(param1, θ / 2) q;
        }

        hello q[0], q[1];
        world(0.1, 0.2) q[2];
        hello q[2], q[1];
        """
        )
        assert [g.label for g in circ.gates] == [
            "H",
            "CX",
            "U3",
            "U2",
            "U2",
            "H",
            "CX",
            "U3",
        ]

    def test_openqasm2_custom_nested_gates(self):
        circ = qtn.Circuit.from_openqasm2_str(
            """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[3];

        gate cphase(θ) a, b
        {
            U3(0, 0, θ / 2) a;
            CX a, b;
            U3(0, 0, -θ / 2) b;
            CX a, b;
            U3(0, 0, θ / 2) b;
        }

        gate doublecphase(θ) a, b, c {
            cphase(θ) a, b;
            cphase(θ) b, c;
        }

        doublecphase(0.1) q[0], q[1], q[2];
        doublecphase(0.2) q[2], q[0], q[1];
        """
        )
        assert [g.label for g in circ.gates] == [
            "U3",
            "CX",
            "U3",
            "CX",
            "U3",
        ] * 4

    def test_openqasm2_a_gate_called_gate(self):
        qasm_str = """
        OPENQASM 2.0;
        include "qelib1.inc";
        gate gate_PauliEvolution(param0) q0,q1 { rz(0.2) q0; rz(-0.1) q1; }
        qreg q[2];
        gate_PauliEvolution(0.1) q[0],q[1];
        """
        circ = qtn.Circuit.from_openqasm2_str(qasm_str)
        assert len(circ.gates) == 2
