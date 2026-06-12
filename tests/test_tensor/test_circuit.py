import itertools
import math

import numpy as np
import pytest
from numpy.testing import assert_allclose

import quimb as qu
import quimb.tensor as qtn
from quimb.tensor.circuit import (
    parse_openqasm3_file,
    parse_openqasm3_str,
    parse_openqasm3_url,
    rx_gate_param_gen,
)
from quimb.tensor.interface import pack, unpack


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


def rand_reg_graph(reg, n, seed=None):
    import networkx as nx

    G = nx.random_regular_graph(reg, n, seed=seed)
    return G


def graph_to_qsim(G, gamma0=-0.743043, beta0=0.754082):
    n = G.number_of_nodes()

    # add all the gates
    circ = f"{n}\n"
    for i in range(n):
        circ += f"H {i}\n"
    for i, j in G.edges:
        circ += f"Rzz {gamma0} {i} {j}\n"
    for i in range(n):
        circ += f"Rx {beta0} {i}\n"

    return circ


def random_a2a_circ(L, depth, seed=42):
    rng = np.random.default_rng(seed)

    qubits = np.arange(L)
    gates = []

    for i in range(L):
        gates.append((0, "h", i))

    for d in range(depth):
        rng.shuffle(qubits)

        for i in range(0, L - 1, 2):
            g = rng.choice(["cx", "cy", "cz", "iswap"])
            gates.append((d, g, qubits[i], qubits[i + 1]))

        for q in qubits:
            g = rng.choice(["rx", "ry", "rz"])
            gates.append((d, g, rng.normal(1.0, 0.5), q))

    circ = qtn.Circuit(L)
    circ.apply_gates(gates)

    return circ


def qft_circ(n, swaps=True, **circuit_opts):
    circ = qtn.Circuit(n, **circuit_opts)

    for i in range(n):
        circ.h(i)
        for j, m in zip(range(i + 1, n), itertools.count(2)):
            circ.cu1(2 * math.pi / 2**m, j, i)

    if swaps:
        for i in range(n // 2):
            circ.swap(i, n - i - 1)

    return circ


def swappy_circ(n, depth):
    circ = qtn.Circuit(n)

    for d in range(depth):
        pairs = np.random.permutation(np.arange(n))

        for i in range(n // 2):
            qi = pairs[2 * i]
            qj = pairs[2 * i + 1]

            gate = np.random.choice(["FSIM", "SWAP"])
            if gate == "FSIM":
                params = np.random.randn(2)
            elif gate == "FSIMG":
                params = np.random.randn(5)
            else:
                params = ()

            circ.apply_gate(gate, *params, qi, qj)

    return circ


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


class TestCircuit:
    def test_prepare_GHZ(self):
        qc = qtn.Circuit(3)
        gates = [
            ("H", 0),
            ("H", 1),
            ("CNOT", 1, 2),
            ("CNOT", 0, 2),
            ("H", 0),
            ("H", 1),
            ("H", 2),
        ]
        qc.apply_gates(gates)
        assert qu.expec(qc.psi.to_dense(), qu.ghz_state(3)) == pytest.approx(1)
        counts = qc.simulate_counts(1024)
        assert len(counts) == 2
        assert "000" in counts
        assert "111" in counts
        assert counts["000"] + counts["111"] == 1024

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

    def test_get_params_excludes_named_expression_managed_gate_indices(self):
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

        params = circ.get_params()
        assert set(params) == {"theta", 1}
        assert 0 not in params
        assert np.isnan(params["theta"])
        assert tuple(params[1]) == pytest.approx((0.1, 0.2, 0.3))

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

    def test_circuit_register_named_params_generic(self):
        circ = qtn.Circuit(2)
        circ.rx(np.nan, 0, parametrize=True)
        circ.ry(np.nan, 1, parametrize=True)
        circ.register_named_params(
            {"theta": np.nan},
            {
                0: ("theta",),
                1: ("cos(theta / 2)",),
            },
        )

        circ.set_params({"theta": np.array(0.6)})
        assert tuple(circ.gates[0].params) == pytest.approx((0.6,))
        assert tuple(circ.gates[1].params) == pytest.approx((math.cos(0.3),))

    def test_circuit_register_named_params_sequence_and_callable(self):
        circ = qtn.Circuit(1)
        circ.rx(np.nan, 0, parametrize=True)
        circ.register_named_params(
            ["theta"],
            {0: (lambda env: env["theta"] / 2,)},
        )

        assert circ.named_param_names == ("theta",)
        assert np.isnan(circ.named_params["theta"])
        assert math.isnan(circ.gates[0].params[0])

        circ.set_params({"theta": np.array(0.6)})
        assert tuple(circ.gates[0].params) == pytest.approx((0.3,))

    def test_circuit_set_params_string_keys_require_registration(self):
        circ = qtn.Circuit(1)
        circ.rx(0.1, 0, parametrize=True)

        with pytest.raises(
            TypeError, match="require registered named parameters"
        ):
            circ.set_params({"theta": 0.2})

    def test_circuit_register_named_params_rejects_unknown_gate_index(self):
        circ = qtn.Circuit(1)
        circ.rx(np.nan, 0, parametrize=True)

        with pytest.raises(ValueError, match="unknown gate index: 2"):
            circ.register_named_params({"theta": np.nan}, {2: ("theta",)})

    def test_circuit_register_named_params_rejects_non_parametrized_gate(self):
        circ = qtn.Circuit(1)
        circ.rx(0.1, 0)

        with pytest.raises(ValueError, match="got non-parametrized gate: 0"):
            circ.register_named_params({"theta": np.nan}, {0: ("theta",)})

    def test_circuit_register_named_params_rejects_wrong_arity(self):
        circ = qtn.Circuit(1)
        circ.rx(np.nan, 0, parametrize=True)

        with pytest.raises(ValueError, match="expected 1, got 2"):
            circ.register_named_params(
                {"theta": np.nan},
                {0: ("theta", "theta")},
            )

    def test_circuit_register_named_params_accepts_generator_expressions(self):
        circ = qtn.Circuit(1)
        circ.rx(np.nan, 0, parametrize=True)
        circ.register_named_params(
            {"theta": np.nan},
            {0: (expr for expr in ("theta",))},
        )

        circ.set_params({"theta": np.array(0.6)})
        assert tuple(circ.gates[0].params) == pytest.approx((0.6,))

    def test_circuit_apply_to_arrays_updates_named_params(self):
        circ = qtn.Circuit.from_openqasm3_str(
            """
            OPENQASM 3.0;
            include "stdgates.inc";
            input float theta;
            qubit[1] q;
            rx(theta / 2) q[0];
            """
        )
        circ.set_params({"theta": np.array(0.6, dtype=np.float64)})

        circ.apply_to_arrays(lambda x: np.asarray(x, dtype=np.float32))

        assert circ.get_params()["theta"].dtype == np.float32
        assert circ.psi["GATE_0"].params.dtype == np.float32

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

    @pytest.mark.parametrize(
        "Circ", [qtn.Circuit, qtn.CircuitMPS, qtn.CircuitDense]
    )
    def test_all_gate_methods(self, Circ):
        import random

        g_nq_np = [
            # single qubit
            ("x", 1, 0),
            ("y", 1, 0),
            ("z", 1, 0),
            ("s", 1, 0),
            ("sdg", 1, 0),
            ("t", 1, 0),
            ("tdg", 1, 0),
            ("h", 1, 0),
            ("sx", 1, 0),
            ("sxdg", 1, 0),
            ("iden", 1, 0),
            ("x_1_2", 1, 0),
            ("y_1_2", 1, 0),
            ("z_1_2", 1, 0),
            ("w_1_2", 1, 0),
            ("hz_1_2", 1, 0),
            # single qubit parametrizable
            ("rx", 1, 1),
            ("ry", 1, 1),
            ("rz", 1, 1),
            ("u3", 1, 3),
            ("u2", 1, 2),
            ("u1", 1, 1),
            ("phase", 1, 1),
            # two qubit
            ("cx", 2, 0),
            ("cy", 2, 0),
            ("cz", 2, 0),
            ("cnot", 2, 0),
            ("swap", 2, 0),
            ("iswap", 2, 0),
            # two qubit parametrizable
            ("rxx", 2, 1),
            ("ryy", 2, 1),
            ("rzz", 2, 1),
            ("crx", 2, 1),
            ("cry", 2, 1),
            ("crz", 2, 1),
            ("cu3", 2, 3),
            ("cu2", 2, 2),
            ("cu1", 2, 1),
            ("cphase", 2, 1),
            ("fsim", 2, 2),
            ("fsimg", 2, 5),
            ("givens", 2, 1),
            ("givens2", 2, 2),
            ("xx_plus_yy", 2, 2),
            ("xx_minus_yy", 2, 2),
            ("su4", 2, 15),
        ]
        random.shuffle(g_nq_np)

        psi0 = qtn.MPS_rand_state(2, 2)
        circ = Circ(2, psi0=psi0, tags="PSI0")

        for g, n_q, n_p in g_nq_np:
            args = [
                *np.random.uniform(0, 2 * np.pi, size=n_p),
                *np.random.choice([0, 1], replace=False, size=n_q),
            ]
            getattr(circ, g)(*args)

        assert circ.psi.H @ circ.psi == pytest.approx(1.0)
        assert abs((circ.psi.H & psi0) ^ all) < 0.99999999

    def test_su4(self):
        psi0 = qtn.MPS_rand_state(2, 2)
        circ_a = qtn.Circuit(psi0=psi0)
        params = qu.randn(15)

        circ_a.su4(*params, 0, 1)
        psi_a = circ_a.to_dense()

        circ_b = qtn.Circuit(psi0=psi0)
        (
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
        ) = params
        circ_b.u3(theta1, phi1, lamda1, 0)
        circ_b.u3(theta2, phi2, lamda2, 1)
        circ_b.cnot(1, 0)
        circ_b.rz(t1, 0)
        circ_b.ry(t2, 1)
        circ_b.cnot(0, 1)
        circ_b.ry(t3, 1)
        circ_b.cnot(1, 0)
        circ_b.u3(theta3, phi3, lamda3, 0)
        circ_b.u3(theta4, phi4, lamda4, 1)
        psi_b = circ_b.to_dense()

        assert qu.fidelity(psi_a, psi_b) == pytest.approx(1.0)

    def test_three_qubit_gates(self):
        psi0 = qtn.MPS_rand_state(3, 2)
        circ = qtn.Circuit(psi0=psi0)
        circ.ccx(0, 1, 2)
        circ.cswap(2, 1, 0)
        circ.toffoli(0, 1, 2)
        circ.ccy(1, 0, 2)
        circ.ccz(1, 2, 0)
        circ.fredkin(2, 1, 0)
        psi = circ.psi.to_dense()
        assert qu.expec(psi, psi) == pytest.approx(1.0)

    def test_auto_split_gate(self):
        n = 3
        ops = [
            ("u3", 1.0, 2.0, 3.0, 0),
            ("u3", 2.0, 3.0, 1.0, 1),
            ("u3", 3.0, 1.0, 2.0, 2),
            ("cz", 0, 1),
            ("iswap", 1, 2),
            ("cx", 2, 0),
            ("iswap", 2, 1),
            ("h", 0),
            ("h", 1),
            ("h", 2),
        ]
        cnorm = qtn.Circuit(n, gate_opts=dict(contract="split-gate"))
        cnorm.apply_gates(ops)
        assert cnorm.psi.max_bond() == 4

        cswap = qtn.Circuit(n, gate_opts=dict(contract="swap-split-gate"))
        cswap.apply_gates(ops)
        assert cswap.psi.max_bond() == 4

        cauto = qtn.Circuit(n, gate_opts=dict(contract="auto-split-gate"))
        cauto.apply_gates(ops)
        assert cauto.psi.max_bond() == 2

        assert qu.fidelity(
            cnorm.psi.to_dense(), cswap.psi.to_dense()
        ) == pytest.approx(1.0)
        assert qu.fidelity(
            cswap.psi.to_dense(), cauto.psi.to_dense()
        ) == pytest.approx(1.0)

    @pytest.mark.parametrize("gate2", ["cx", "iswap"])
    def test_circuit_simplify_tensor_network(self, gate2):
        import itertools
        import random

        depth = n = 8

        circ = qtn.Circuit(n)

        def random_single_qubit_layer():
            return [
                (random.choice(["X_1_2", "Y_1_2", "W_1_2"]), i)
                for i in range(n)
            ]

        def even_two_qubit_layer():
            return [(gate2, i, i + 1) for i in range(0, n, 2)]

        def odd_two_qubit_layer():
            return [(gate2, i, i + 1) for i in range(1, n - 1, 2)]

        layering = itertools.cycle(
            [
                random_single_qubit_layer,
                even_two_qubit_layer,
                random_single_qubit_layer,
                odd_two_qubit_layer,
            ]
        )

        for i, layer_fn in zip(range(depth), layering):
            for g in layer_fn():
                circ.apply_gate(*g, gate_round=i)

        psif = qtn.MPS_computational_state("0" * n).squeeze_()
        tn = circ.psi & psif

        c = tn.contract(all)
        cw = tn.contraction_width()

        tn_s = tn.full_simplify()
        assert tn_s.num_tensors < tn.num_tensors
        assert tn_s.num_indices < tn.num_indices
        # need to specify output inds since we now have hyper edges
        c_s = tn_s.contract(all, output_inds=[])
        assert c_s == pytest.approx(c)
        cw_s = tn_s.contraction_width(output_inds=[])
        assert cw_s <= cw

    def test_amplitude(self):
        L = 5
        circ = random_a2a_circ(L, 3)
        psi = circ.to_dense()

        for i in range(2**L):
            b = f"{i:0>{L}b}"
            c = circ.amplitude(b)
            assert c == pytest.approx(psi[i, 0])

    def test_partial_trace(self):
        L = 5
        circ = random_a2a_circ(L, 3)
        psi = circ.to_dense()
        for i in range(L - 1):
            keep = (i, i + 1)
            assert_allclose(
                qu.partial_trace(psi, [2] * 5, keep=keep),
                circ.partial_trace(keep),
                atol=1e-12,
            )

    @pytest.mark.parametrize("group_size", (1, 2, 6))
    def test_sample(self, group_size):
        import collections

        from scipy.stats import power_divergence

        C = 2**10
        L = 5
        circ = random_a2a_circ(L, 3, seed=42)

        psi = circ.to_dense()
        p_exp = abs(psi.reshape(-1)) ** 2
        f_exp = p_exp * C

        counts = collections.Counter(
            circ.sample(C, group_size=group_size, seed=42)
        )
        f_obs = np.zeros(2**L)
        for b, c in counts.items():
            f_obs[int(b, 2)] = c

        assert power_divergence(f_obs, f_exp)[0] < 100

    @pytest.mark.parametrize("group_size", (1, 3))
    def test_sample_gate_by_gate(self, group_size):
        import collections

        from scipy.stats import power_divergence

        C = 2**10
        L = 5
        circ = random_a2a_circ(L, 3, seed=43)

        psi = circ.to_dense()
        p_exp = abs(psi.reshape(-1)) ** 2
        f_exp = p_exp * C

        counts = collections.Counter(
            circ.sample_gate_by_gate(C, group_size=group_size, seed=42)
        )
        f_obs = np.zeros(2**L)
        for b, c in counts.items():
            f_obs[int(b, 2)] = c

        assert power_divergence(f_obs, f_exp)[0] < 100

    def test_sample_chaotic(self):
        import collections

        from scipy.stats import power_divergence

        C = 2**12
        L = 5
        reps = 3
        depth = 2
        goodnesses = [0] * 5

        for i in range(reps):
            circ = random_a2a_circ(L, depth, seed=42 + i)

            psi = circ.to_dense()
            p_exp = abs(psi.reshape(-1)) ** 2
            f_exp = p_exp * C

            for num_marginal in [3, 4, 5]:
                counts = collections.Counter(
                    circ.sample_chaotic(C, num_marginal, seed=42 + i)
                )
                f_obs = np.zeros(2**L)
                for b, c in counts.items():
                    f_obs[int(b, 2)] = c

                goodness = power_divergence(f_obs, f_exp)[0]
                goodnesses[num_marginal - 1] += goodness

        # assert average sampling goodness gets better with larger marginal
        assert sum(goodnesses[i] < goodnesses[i - 1] for i in range(1, L)) == 2

    def test_local_expectation(self):
        import random

        L = 5
        depth = 3
        circ = random_a2a_circ(L, depth)
        psi = circ.to_dense()
        for _ in range(10):
            G = qu.rand_matrix(4)
            i = random.randint(0, L - 2)
            where = (i, i + 1)
            x1 = qu.expec(qu.ikron(G, [2] * L, where), psi)
            x2 = circ.local_expectation(G, where)
            assert x1 == pytest.approx(x2)

    def test_local_expectation_multigate(self):
        circ = qtn.Circuit(2)
        circ.h(0)
        circ.cnot(0, 1)
        circ.y(1)
        Gs = [qu.kronpow(qu.pauli(s), 2) for s in "xyz"]
        exps = circ.local_expectation(Gs, [0, 1])
        assert exps[0] == pytest.approx(-1)
        assert exps[1] == pytest.approx(-1)
        assert exps[2] == pytest.approx(-1)

    def test_local_expectation_len1(self):
        circ = qtn.Circuit(1)
        circ.apply_gate("H", 0, gate_round=0)
        circ.local_expectation([qu.pauli("X")], (0,))

    def test_uni_to_dense(self):
        import cmath

        circ = qft_circ(3)
        U = circ.uni.to_dense()
        w = cmath.exp(2j * math.pi / 2**3)
        ex = 2 ** (-3 / 2) * np.array(
            [
                [w**0, w**0, w**0, w**0, w**0, w**0, w**0, w**0],
                [w**0, w**1, w**2, w**3, w**4, w**5, w**6, w**7],
                [w**0, w**2, w**4, w**6, w**0, w**2, w**4, w**6],
                [w**0, w**3, w**6, w**1, w**4, w**7, w**2, w**5],
                [w**0, w**4, w**0, w**4, w**0, w**4, w**0, w**4],
                [w**0, w**5, w**2, w**7, w**4, w**1, w**6, w**3],
                [w**0, w**6, w**4, w**2, w**0, w**6, w**4, w**2],
                [w**0, w**7, w**6, w**5, w**4, w**3, w**2, w**1],
            ]
        )
        assert_allclose(U, ex)

    def test_swap_lighcones(self):
        circ = qtn.Circuit(3)
        circ.x(0)  # 0
        circ.x(1)  # 1
        circ.x(2)  # 2
        circ.swap(0, 1)  # 3
        circ.cx(1, 2)  # 4
        circ.cx(0, 1)  # 5
        assert circ.get_reverse_lightcone_tags((2,)) == (
            "PSI0",
            "GATE_0",
            "GATE_2",
            "GATE_4",
        )

    def test_swappy_local_expecs(self):
        circ = swappy_circ(4, 4)
        Gs = [qu.rand_matrix(4) for _ in range(3)]
        pairs = [(0, 1), (1, 2), (2, 3)]

        psi = circ.to_dense()
        dims = [2] * 4

        exs = [
            qu.expec(qu.ikron(G, dims, pair), psi)
            for G, pair in zip(Gs, pairs)
        ]
        aps = [circ.local_expectation(G, pair) for G, pair in zip(Gs, pairs)]

        assert_allclose(exs, aps)

    @pytest.mark.parametrize(
        "name, densefn, nparam, nqubit",
        [
            ("rx", qu.Rx, 1, 1),
            ("ry", qu.Ry, 1, 1),
            ("rz", qu.Rz, 1, 1),
            ("u3", qu.U_gate, 3, 1),
            ("fsim", qu.fsim, 2, 2),
            ("fsimg", qu.fsimg, 5, 2),
        ],
    )
    def test_parametrized_gates_rx(self, name, densefn, nparam, nqubit):
        k0 = qu.rand_ket(2**nqubit)
        params = qu.randn(nparam)
        kf = densefn(*params) @ k0
        k0mps = qtn.MatrixProductState.from_dense(k0, [2] * nqubit)
        circ = qtn.Circuit(psi0=k0mps, gate_opts={"contract": False})
        getattr(circ, name)(*params, *range(nqubit), parametrize=True)
        tn = circ.psi
        assert isinstance(tn["GATE_0"], qtn.PTensor)
        assert_allclose(circ.to_dense(), kf)

    def test_apply_raw_gate(self):
        k0 = qu.rand_ket(4)
        psi0 = qtn.MatrixProductState.from_dense(k0, [2] * 2)
        circ = qtn.Circuit(psi0=psi0)
        U = qu.rand_uni(4)
        circ.apply_gate_raw(U, [0, 1], tags="UCUSTOM")
        assert len(circ.gates) == 1
        assert "UCUSTOM" in circ.psi.tags
        assert qu.fidelity(circ.to_dense(), U @ k0) == pytest.approx(1)

    def test_apply_controlled_gate_basic_equiv(self):
        circ = qtn.Circuit(3)
        circ.apply_gate("x", qubits=(2,), controls=(0, 1))
        U = circ.get_uni().to_dense()
        assert_allclose(U, qu.toffoli())

        circ = qtn.Circuit(3)
        circ.apply_gate("swap", qubits=(1, 2), controls=(0,))
        U = circ.get_uni().to_dense()
        assert_allclose(U, qu.fredkin())

    def test_multi_controlled_circuit(self):
        import random

        N = 10
        circ = qtn.Circuit(N)
        regs = list(range(N))
        random.shuffle(regs)
        circ.apply_gate("H", regs[0])
        for i in range(N - 1):
            circ.apply_gate("CNOT", regs[i], regs[i + 1])
        circ.apply_gate("X", N - 1, controls=range(N - 1))
        circ.apply_gate("SWAP", qubits=(N - 2, N - 1), controls=range(N - 2))
        (b,) = circ.sample(1, group_size=3, seed=42)
        assert b[N - 2] == "0"

    @pytest.mark.parametrize("dtype", [None, "complex64", "complex128"])
    @pytest.mark.parametrize("backend", [None, "torch"])
    @pytest.mark.parametrize("dtype_final", [None, "complex64", "complex128"])
    @pytest.mark.parametrize("convert_eager", [True, False])
    def test_conversions(self, dtype, backend, dtype_final, convert_eager):
        if backend == "torch":
            pytest.importorskip("torch")

            def to_backend(x):
                import torch

                return torch.tensor(x)

        else:
            to_backend = None

        circ = qtn.Circuit(
            2, dtype=dtype, to_backend=to_backend, convert_eager=convert_eager
        )
        circ.h(0)
        circ.cx(0, 1)
        circ.y(1)

        if not convert_eager:
            # constructed with default dtype
            assert circ._psi.dtype_name == "complex128"
            assert circ._psi.backend == "numpy"
        else:
            # constructed with this type
            assert circ._psi.dtype == dtype or dtype is None
            if backend == "torch":
                assert circ._psi.backend == "torch"
            else:
                assert circ._psi.backend == "numpy"

        # converted to this type
        if dtype is None:
            expected_default_dtype = "complex128"
        else:
            expected_default_dtype = dtype

        if backend != "torch":
            test_tn_default = circ.amplitude_tn()
            test_tn_explicit = circ.amplitude_tn(dtype=dtype_final)
        else:
            # test a less simplified tensor network
            test_tn_default = circ.partial_trace_tn(
                (1,), simplify_sequence="R"
            )
            test_tn_explicit = circ.partial_trace_tn(
                (1,), simplify_sequence="R", dtype=dtype_final
            )

        assert test_tn_default.dtype_name == expected_default_dtype
        if dtype_final is not None:
            assert test_tn_explicit.dtype_name == dtype_final


class TestCircuitMPS:
    def test_from_qsim_mps_swapsplit(self):
        G = rand_reg_graph(reg=3, n=18, seed=42)
        qsim = graph_to_qsim(G)
        qc = qtn.CircuitMPS.from_qsim_str(qsim)
        assert len(qc.psi.tensors) == 18
        assert (qc.psi.H & qc.psi) ^ all == pytest.approx(1.0)

    def test_multi_controlled_mps_circuit(self):
        N = 10
        rng = np.random.default_rng(42)

        gates = []
        for i in range(N):
            gates.append(
                qtn.Gate(
                    "U3", params=rng.uniform(0, 2 * np.pi, size=3), qubits=[i]
                )
            )
        gates.append(
            qtn.Gate(
                "SU4",
                params=rng.uniform(0, 2 * np.pi, size=15),
                qubits=[6, 2],
                controls=[8, 3, 4, 0],
            )
        )
        for i in range(N):
            gates.append(
                qtn.Gate(
                    "U3", params=rng.uniform(0, 2 * np.pi, size=3), qubits=[i]
                )
            )
        gates.append(
            qtn.Gate.from_raw(
                qu.rand_uni(2**3), qubits=[0, 9, 5], controls=[1, 2, 7]
            )
        )

        circ = qtn.Circuit(N=10)
        circ.apply_gates(gates)
        psi_lazy = circ.psi
        circ = qtn.CircuitMPS(N=10)
        circ.apply_gates(gates)
        mps = circ.psi
        assert mps.norm() == pytest.approx(1.0)
        assert mps.distance_normalized(psi_lazy) < 1e-6

    def test_mps_sampling(self):
        N = 6
        circ = qtn.CircuitMPS(N)
        circ.h(3)
        circ.cx(3, 2)
        circ.cx(2, 1)
        circ.cx(1, 0)
        circ.cx(0, 5)
        circ.cx(5, 4)
        circ.x(4)
        for x in circ.sample(10, seed=42):
            assert x in {"000010", "111101"}

    def test_mps_sampling_seed(self):
        N = 1
        circ = qtn.CircuitMPS(N)
        circ.h(0)
        samples = list(circ.sample(10, seed=1234))
        assert len(set(samples)) == 2

    def test_permmps_sampling(self):
        N = 6
        circ = qtn.CircuitPermMPS(N)
        circ.h(3)
        circ.cx(3, 2)
        circ.cx(2, 1)
        circ.cx(1, 0)
        circ.cx(0, 5)
        circ.cx(5, 4)
        circ.x(4)
        assert circ.qubits != tuple(range(N))
        for x in circ.sample(10, seed=42):
            assert x in {"000010", "111101"}

    def test_permmps_sampling_seed(self):
        N = 1
        circ = qtn.CircuitPermMPS(N)
        circ.h(0)
        samples = list(circ.sample(10, seed=1234))
        assert len(set(samples)) == 2

    def test_permmps_sampling_inverts_qubit_ordering(self):
        circ = qtn.CircuitPermMPS(4)
        circ.x(1)
        circ.cx(0, 3)

        assert circ.qubits == [0, 3, 1, 2]
        assert set(circ.sample(10, seed=42)) == {"0100"}

    def test_lazymps_sampling(self):
        N = 6
        circ = qtn.CircuitLazyMPS(N, max_bond=512)
        circ.h(3)
        circ.cx(3, 2)
        circ.cx(2, 1)
        circ.cx(1, 0)
        circ.cx(0, 5)
        circ.cx(5, 4)
        circ.x(4)
        for x in circ.sample(10, seed=42):
            assert x in {"000010", "111101"}

    def test_lazymps_sampling_seed(self):
        N = 1
        circ = qtn.CircuitLazyMPS(N, max_bond=512)
        circ.h(0)
        samples = list(circ.sample(10, seed=1234))
        assert len(set(samples)) == 2

    def test_performance_lazymps_long_range(self):
        import timeit

        N = 10

        circ = qtn.CircuitLazyMPS(N, max_bond=512, compress_every=4)

        start_time = timeit.default_timer()

        circ.h(0)
        for _ in range(100):
            for i in range(N - 1):
                circ.cx(0, i + 1)

        elapsed_lazymps = timeit.default_timer() - start_time
        lazy_state = circ.psi

        circ = qtn.CircuitMPS(N, max_bond=512, gate_opts=dict(method="src", contract="nonlocal"))

        start_time = timeit.default_timer()
        circ.h(0)
        for _ in range(100):
            for i in range(N - 1):
                circ.cx(0, i + 1)

        elapsed_mps = timeit.default_timer() - start_time
        mps_state = circ.psi

        assert elapsed_lazymps < elapsed_mps, f"LazyMPS took {elapsed_lazymps:.2f}s, MPS took {elapsed_mps:.2f}s"
        assert lazy_state @ mps_state == pytest.approx(1.0)


class TestCircuitPEPSSimpleUpdate:
    def test_requires_geometry(self):
        # none of edges, gates or psi0 given -> cannot define the geometry
        with pytest.raises(ValueError):
            qtn.CircuitPEPSSimpleUpdate()

    def test_geometry_inferred_from_gates(self):
        # passing the two-qubit gates instead of edges should reconstruct the
        # same geometry, and give the same state as the explicit-edges path
        edges = [(0, 1), (1, 2), (0, 2)]
        rng = np.random.default_rng(5)
        gates = [
            qtn.Gate("RY", params=[rng.uniform(0, np.pi)], qubits=[i])
            for i in range(3)
        ]
        gates += [qtn.Gate("CZ", params=(), qubits=list(e)) for e in edges]

        circ_edges = qtn.CircuitPEPSSimpleUpdate(edges=edges, max_bond=16)
        circ_gates = qtn.CircuitPEPSSimpleUpdate(gates=gates, max_bond=16)
        # geometry matches (as undirected edges) and the gates are not applied
        assert {frozenset(e) for e in circ_gates.edges} == {
            frozenset(e) for e in circ_edges.edges
        }
        assert circ_gates.num_gates == 0

        circ_edges.apply_gates(gates)
        circ_gates.apply_gates(gates)
        Z = qu.pauli("Z").astype(complex)
        for i in range(3):
            xe = circ_edges.local_expectation(Z, i, max_distance=3)
            xg = circ_gates.local_expectation(Z, i, max_distance=3)
            assert float(np.real(xg)) == pytest.approx(
                float(np.real(xe)), abs=1e-10
            )

    def test_geometry_inferred_from_psi0(self):
        # build a state, hand it back as psi0, and check the geometry is read
        # from its bonds and expectations are preserved
        edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
        rng = np.random.default_rng(11)
        gates = [
            qtn.Gate("RY", params=[rng.uniform(0, np.pi)], qubits=[i])
            for i in range(4)
        ]
        gates += [qtn.Gate("CZ", params=(), qubits=list(e)) for e in edges]

        circ = qtn.CircuitPEPSSimpleUpdate(edges=edges, max_bond=16)
        circ.apply_gates(gates)
        psi = circ.psi

        circ2 = qtn.CircuitPEPSSimpleUpdate(psi0=psi, max_bond=16)
        assert {frozenset(e) for e in circ2.edges} == {
            frozenset(e) for e in edges
        }
        Z = qu.pauli("Z").astype(complex)
        for i in range(4):
            x1 = circ.local_expectation(Z, i, max_distance=4)
            x2 = circ2.local_expectation(Z, i, max_distance=4)
            assert float(np.real(x2)) == pytest.approx(
                float(np.real(x1)), abs=1e-6
            )

    def test_construction_and_initial_state(self):
        # 2x3 grid of integer-labelled sites
        edges = [(0, 1), (1, 2), (3, 4), (4, 5), (0, 3), (1, 4), (2, 5)]
        circ = qtn.CircuitPEPSSimpleUpdate(edges=edges, max_bond=8)
        # one tensor per site, bonds only on the edges, all bond dim 1
        assert circ.N == 6
        assert circ._psi.max_bond() == 1
        assert circ.psi.norm() == pytest.approx(1.0)

    def test_target_api(self):
        # the exact usage requested in the issue, on a 3x3 grid of integer sites
        ncol = 3
        edges = []
        for r in range(3):
            for c in range(3):
                s = r * ncol + c
                if c + 1 < 3:
                    edges.append((s, s + 1))
                if r + 1 < 3:
                    edges.append((s, s + ncol))
        rng = np.random.default_rng(42)
        gates = [
            qtn.Gate(
                "U3", params=rng.uniform(0, 2 * np.pi, size=3), qubits=[i]
            )
            for i in range(9)
        ]
        gates += [
            qtn.Gate("CZ", params=(), qubits=list(edge)) for edge in edges
        ]
        circ = qtn.CircuitPEPSSimpleUpdate(edges=edges, max_bond=8)
        circ.apply_gates(gates)
        peps = circ.psi
        assert peps.max_bond() <= 8
        assert peps.num_tensors == 9

    def test_matches_exact_on_a_chain(self):
        # on a chain, simple update at large bond is exact, so local
        # expectations should match a dense Circuit to high precision
        N = 5
        edges = qtn.edges_1d_chain(N, cyclic=False)
        rng = np.random.default_rng(0)

        gates = []
        for i in range(N):
            gates.append(
                qtn.Gate("RY", params=[rng.uniform(0, np.pi)], qubits=[i])
            )
        for i, j in edges:
            gates.append(qtn.Gate("CNOT", params=(), qubits=[i, j]))
            gates.append(
                qtn.Gate("RZ", params=[rng.uniform(0, np.pi)], qubits=[j])
            )
            gates.append(qtn.Gate("CNOT", params=(), qubits=[i, j]))

        circ_su = qtn.CircuitPEPSSimpleUpdate(
            edges=edges, max_bond=32, cutoff=1e-12
        )
        circ_su.apply_gates(gates)
        circ_su.equilibrate()

        circ_exact = qtn.Circuit(N=N)
        circ_exact.apply_gates(gates)

        Z = qu.pauli("Z").astype(complex)
        for i in range(N):
            x_su = circ_su.local_expectation(Z, i, max_distance=N)
            x_ex = circ_exact.local_expectation(Z, i)
            assert float(np.real(x_su)) == pytest.approx(
                float(np.real(x_ex)), abs=1e-8
            )

    def test_equilibrate_preserves_expectations(self):
        N = 4
        edges = qtn.edges_1d_chain(N, cyclic=False)
        rng = np.random.default_rng(7)
        gates = [
            qtn.Gate("RY", params=[rng.uniform(0, np.pi)], qubits=[i])
            for i in range(N)
        ]
        for i, j in edges:
            gates.append(qtn.Gate("CNOT", params=(), qubits=[i, j]))

        circ = qtn.CircuitPEPSSimpleUpdate(edges=edges, max_bond=32)
        circ.apply_gates(gates)

        Z = qu.pauli("Z").astype(complex)
        before = [
            float(np.real(circ.local_expectation(Z, i, max_distance=N)))
            for i in range(N)
        ]
        circ.equilibrate()
        after = [
            float(np.real(circ.local_expectation(Z, i, max_distance=N)))
            for i in range(N)
        ]
        for b, a in zip(before, after):
            assert b == pytest.approx(a, abs=1e-8)

    def test_matches_exact_on_2x2_plaquette(self):
        # a 2x2 grid has a 4-cycle, so simple update is genuinely approximate,
        # but a shallow circuit still matches exact closely. checks both
        # local_expectation and the gauge-absorbed psi.
        edges = [(0, 1), (2, 3), (0, 2), (1, 3)]
        rng = np.random.default_rng(3)
        gates = [
            qtn.Gate("RY", params=[rng.uniform(0, np.pi)], qubits=[i])
            for i in range(4)
        ]
        for i, j in edges:
            gates.append(qtn.Gate("CZ", params=(), qubits=[i, j]))

        circ = qtn.CircuitPEPSSimpleUpdate(
            edges=edges, max_bond=16, cutoff=1e-12
        )
        circ.apply_gates(gates)

        circ_exact = qtn.Circuit(N=4)
        circ_exact.apply_gates(gates)

        Z = qu.pauli("Z").astype(complex)
        # gauge-absorbed psi should be a normalized state
        psi = circ.psi
        assert ((psi.H & psi) ^ all) == pytest.approx(1.0, abs=1e-6)

        for i in range(4):
            x_su = circ.local_expectation(Z, i, max_distance=4)
            # measure the same way directly on the returned psi
            x_psi = psi.compute_local_expectation_cluster(
                {(i,): Z}, max_distance=4, normalized=True
            )
            x_ex = circ_exact.local_expectation(Z, i)
            assert float(np.real(x_su)) == pytest.approx(
                float(np.real(x_ex)), abs=1e-6
            )
            assert float(np.real(x_psi)) == pytest.approx(
                float(np.real(x_ex)), abs=1e-6
            )

    def test_psi_access_does_not_mutate_gauges(self):
        edges = [(0, 1), (1, 2)]
        circ = qtn.CircuitPEPSSimpleUpdate(edges=edges, max_bond=8)
        circ.apply_gates(
            [
                qtn.Gate("H", params=(), qubits=[0]),
                qtn.Gate("CNOT", params=(), qubits=[0, 1]),
                qtn.Gate("CNOT", params=(), qubits=[1, 2]),
            ]
        )
        before = {k: np.asarray(v).copy() for k, v in circ.gauges.items()}
        _ = circ.psi
        _ = circ.psi
        assert set(circ.gauges) == set(before)
        for k in before:
            assert_allclose(np.asarray(circ.gauges[k]), before[k])

    def test_controlled_and_exact_methods_raise(self):
        edges = [(0, 1), (1, 2)]
        circ = qtn.CircuitPEPSSimpleUpdate(edges=edges, max_bond=8)
        # controlled gates can't be expressed by the simple update rule
        with pytest.raises(ValueError):
            circ.apply_gates(
                [qtn.Gate("X", params=(), qubits=[2], controls=[0])]
            )
        # exact-state methods are not meaningful for a gauged approximate state
        circ.apply_gates([qtn.Gate("H", params=(), qubits=[0])])
        with pytest.raises(NotImplementedError):
            circ.amplitude("000")
        with pytest.raises(NotImplementedError):
            list(circ.sample(2))
        with pytest.raises(NotImplementedError):
            circ.partial_trace([0])
        with pytest.raises(NotImplementedError):
            circ.sample_chaotic_rehearse()
        with pytest.raises(NotImplementedError):
            circ.uni
        # a two-qubit gate off any declared edge is rejected
        with pytest.raises(ValueError):
            circ.apply_gates([qtn.Gate("CZ", params=(), qubits=[0, 2])])

    def test_to_dense_matches_exact_on_a_chain(self):
        # on a chain at large bond simple update is exact, so the gauged dense
        # contraction should match a dense Circuit up to a global phase
        N = 5
        edges = qtn.edges_1d_chain(N, cyclic=False)
        rng = np.random.default_rng(2)
        gates = [
            (qu.rand_uni(2, seed=int(rng.integers(1 << 30))), i)
            for i in range(N)
        ]
        gates += [
            (qu.rand_uni(4, seed=int(rng.integers(1 << 30))), a, b)
            for a, b in edges
        ]
        circ = qtn.CircuitPEPSSimpleUpdate(
            edges=edges, max_bond=32, cutoff=1e-12
        )
        circ.apply_gates(gates)

        exact = qtn.Circuit(N=N)
        exact.apply_gates(gates)

        k_su = circ.to_dense()
        k_ex = exact.to_dense()
        # column vector of the full state, matching Circuit.to_dense
        assert k_su.shape == k_ex.shape == (2**N, 1)
        assert qu.fidelity(k_su, k_ex) == pytest.approx(1.0, abs=1e-8)

    def test_get_state_absorb_gauges(self):
        # the three absorb_gauges modes should all describe the same state
        edges = [(0, 1), (1, 2), (0, 2)]
        circ = qtn.CircuitPEPSSimpleUpdate(edges=edges, max_bond=8)
        circ.apply_gates(
            [
                qtn.Gate("H", params=(), qubits=[0]),
                qtn.Gate("CNOT", params=(), qubits=[0, 1]),
                qtn.Gate("CZ", params=(), qubits=[1, 2]),
            ]
        )
        # absorbed network is exactly the `psi` property
        psi_absorbed = circ.get_state(absorb_gauges=True)
        assert_allclose(psi_absorbed.to_dense(), circ.psi.to_dense())
        # gauges added but uncontracted should give the same dense state
        psi_uncontracted = circ.get_state(absorb_gauges=False)
        assert qu.fidelity(
            psi_uncontracted.to_dense(), psi_absorbed.to_dense()
        ) == pytest.approx(1.0, abs=1e-10)
        # "return" hands back the raw network plus a copy of the gauges
        raw, gauges = circ.get_state(absorb_gauges="return")
        assert set(gauges) == set(circ.gauges)
        assert gauges is not circ.gauges
        raw.gauge_simple_insert(gauges)
        assert qu.fidelity(
            raw.to_dense(), psi_absorbed.to_dense()
        ) == pytest.approx(1.0, abs=1e-10)

    def test_local_expectation_on_grid_coordinate_sites(self):
        # sites are 2D coordinate tuples; a single-site `where` must not be
        # misread as a two-site operator (regression for the site handling)
        edges = qtn.edges_2d_square(2, 2)
        sites = sorted({s for e in edges for s in e})
        rng = np.random.default_rng(0)
        gates = [
            (qu.rand_uni(2, seed=int(rng.integers(1 << 30))), s) for s in sites
        ]
        gates += [
            (qu.rand_uni(4, seed=int(rng.integers(1 << 30))), a, b)
            for a, b in edges
        ]
        circ = qtn.CircuitPEPSSimpleUpdate(edges=edges, max_bond=16)
        circ.apply_gates(gates)
        circ.equilibrate(max_iterations=300, tol=1e-12)

        qmap = {s: i for i, s in enumerate(sites)}
        ref = qtn.Circuit(N=len(sites))
        for G, *where in gates:
            ref.apply_gate(G, *(qmap[s] for s in where))

        Z = qu.pauli("Z").astype(complex)
        for s in sites:
            x = circ.local_expectation(Z, s, max_distance=2)
            xe = complex(ref.local_expectation(Z, qmap[s]))
            assert complex(x) == pytest.approx(xe, abs=1e-6)

    def test_local_expectation_accepts_list_where(self):
        # a multi-site `where` given as a list must not crash on the set
        # membership check, and should match the tuple form
        edges = [(0, 1), (1, 2)]
        circ = qtn.CircuitPEPSSimpleUpdate(edges=edges, max_bond=8)
        circ.apply_gates(
            [
                qtn.Gate("H", params=(), qubits=[0]),
                qtn.Gate("CNOT", params=(), qubits=[0, 1]),
            ]
        )
        Z = qu.pauli("Z").astype(complex)
        ZZ = qu.kron(Z, Z)
        x_list = circ.local_expectation(ZZ, [0, 1], max_distance=2)
        x_tuple = circ.local_expectation(ZZ, (0, 1), max_distance=2)
        assert complex(x_list) == pytest.approx(complex(x_tuple))

    def test_copy_is_independent(self):
        edges = qtn.edges_2d_square(2, 2)
        circ = qtn.CircuitPEPSSimpleUpdate(edges=edges, max_bond=8)
        rng = np.random.default_rng(3)
        circ.apply_gates(
            [
                (qu.rand_uni(4, seed=int(rng.integers(1 << 30))), a, b)
                for a, b in edges
            ]
        )
        other = circ.copy()
        n_before = circ.num_gates
        # the copy carries the geometry and gauges and is independently usable
        assert set(other.edges) == set(circ.edges)
        assert other.gauges is not circ.gauges
        assert len(other.gauges) == len(circ.gauges)
        a, b = circ.edges[0]
        other.apply_gates([(qu.rand_uni(4), a, b)])
        # mutating the copy must not touch the original
        assert circ.num_gates == n_before
        assert other.num_gates == n_before + 1

    def test_renorm_and_equilibrate_every(self):
        # the renorm and equilibrate_every options should be accepted and keep
        # normalized expectations correct on a chain (where SU is exact)
        N = 5
        edges = qtn.edges_1d_chain(N, cyclic=False)
        rng = np.random.default_rng(1)
        gates = [
            (qu.rand_uni(2, seed=int(rng.integers(1 << 30))), i)
            for i in range(N)
        ]
        gates += [
            (qu.rand_uni(4, seed=int(rng.integers(1 << 30))), a, b)
            for a, b in edges
        ]
        circ = qtn.CircuitPEPSSimpleUpdate(
            edges=edges,
            max_bond=32,
            cutoff=1e-12,
            renorm=True,
            equilibrate_every=3,
        )
        circ.apply_gates(gates)
        circ.equilibrate()

        exact = qtn.Circuit(N=N)
        exact.apply_gates(gates)
        Z = qu.pauli("Z").astype(complex)
        for i in range(N):
            x = circ.local_expectation(Z, i, max_distance=N)
            xe = complex(exact.local_expectation(Z, i))
            assert complex(x) == pytest.approx(xe, abs=1e-8)

    def test_cluster_error_decreases_with_distance(self):
        # on a loopy lattice with an essentially exact (large max_bond) state,
        # a larger cluster must give a more accurate local expectation
        edges = qtn.edges_2d_square(3, 3)
        sites = sorted({s for e in edges for s in e})
        rng = np.random.default_rng(11)
        gates = [
            (qu.rand_uni(2, seed=int(rng.integers(1 << 30))), s) for s in sites
        ]
        gates += [
            (qu.rand_uni(4, seed=int(rng.integers(1 << 30))), a, b)
            for a, b in edges
        ]
        qmap = {s: i for i, s in enumerate(sites)}
        ref = qtn.Circuit(N=len(sites))
        for G, *where in gates:
            ref.apply_gate(G, *(qmap[s] for s in where))
        Z = qu.pauli("Z").astype(complex)
        exact = {s: complex(ref.local_expectation(Z, qmap[s])) for s in sites}

        circ = qtn.CircuitPEPSSimpleUpdate(edges=edges, max_bond=16)
        circ.apply_gates(gates)
        circ.equilibrate(max_iterations=200, tol=1e-10)

        def err(md):
            return max(
                abs(circ.local_expectation(Z, s, max_distance=md) - exact[s])
                for s in sites
            )

        assert err(3) < err(0)

    def test_scales_to_a_larger_lattice(self):
        # the point of a PEPS simulator: run a circuit on a lattice far too
        # large to simulate exactly, with the bond dimension capped
        edges = qtn.edges_2d_square(6, 6)
        max_bond = 8
        circ = qtn.CircuitPEPSSimpleUpdate(edges=edges, max_bond=max_bond)
        sites = sorted({s for e in edges for s in e})
        rng = np.random.default_rng(2)
        gates = [
            (qu.rand_uni(2, seed=int(rng.integers(1 << 30))), s) for s in sites
        ]
        gates += [
            (qu.rand_uni(4, seed=int(rng.integers(1 << 30))), a, b)
            for a, b in edges
        ]
        circ.apply_gates(gates)
        psi = circ.psi
        assert psi.num_tensors == len(sites)
        assert psi.max_bond() <= max_bond


class TestCircuitGen:
    @pytest.mark.parametrize(
        "ansatz,cyclic",
        [
            ("zigzag", False),
            ("brickwork", False),
            ("brickwork", True),
            ("rand", False),
            ("rand", True),
        ],
    )
    @pytest.mark.parametrize("n", [4, 5])
    def test_1D_ansatzes(self, ansatz, cyclic, n):
        depth = 3
        num_pairs = n if cyclic else n - 1

        fn = {
            "zigzag": qtn.circ_ansatz_1D_zigzag,
            "brickwork": qtn.circ_ansatz_1D_brickwork,
            "rand": qtn.circ_ansatz_1D_rand,
        }[ansatz]

        opts = dict(
            n=n,
            depth=3,
            gate_opts=dict(contract=False),
        )
        if cyclic:
            opts["cyclic"] = True
        if ansatz == "rand":
            opts["seed"] = 42

        circ = fn(**opts)
        tn = circ.uni

        # total number of entangling gates
        assert len(tn["CZ"]) == num_pairs * depth

        # number of entangling gates per pair
        for i in range(num_pairs):
            assert len(tn["CZ", f"I{i}", f"I{(i + 1) % n}"]) == depth

        assert all(isinstance(t, qtn.PTensor) for t in tn["U3"])

    def test_qaoa(self):
        G = rand_reg_graph(3, 10, seed=666)
        terms = {(i, j): 1.0 for i, j in G.edges}
        ZZ = qu.pauli("Z") & qu.pauli("Z")

        gammas = [-0.6]
        betas = [-0.4]

        circ1 = qtn.circ_qaoa(terms, 1, gammas, betas)

        energy1 = sum(circ1.local_expectation(ZZ, edge) for edge in terms)
        assert energy1 < -4

        gammas = [-0.4]
        betas = [0.3]

        circ2 = qtn.circ_qaoa(terms, 1, gammas, betas)

        energy2 = sum(circ2.local_expectation(ZZ, edge) for edge in terms)
        assert energy2 > 4
