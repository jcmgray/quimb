import itertools
import math

import numpy as np
import pytest
from numpy.testing import assert_allclose

import quimb as qu
import quimb.tensor as qtn


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
        qc = qtn.Circuit.from_openqasm2_str(example_openqasm2_qft())
        assert (qc.psi.H & qc.psi) ^ all == pytest.approx(1.0)

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


class TestCircuitMPSLazy:
    @staticmethod
    def _random_circuit_gates(N, depth, seed, long_range=False):
        """A brickwork circuit of U3 and SU4 gates, optionally with an extra
        long range two-qubit gate per layer."""
        rng = np.random.default_rng(seed)
        gates = []
        for d in range(depth):
            for i in range(N):
                gates.append(
                    qtn.Gate(
                        "U3", params=rng.uniform(0, 2 * np.pi, 3), qubits=[i]
                    )
                )
            for i in range(d % 2, N - 1, 2):
                gates.append(
                    qtn.Gate(
                        "SU4",
                        params=rng.uniform(0, 2 * np.pi, 15),
                        qubits=[i, i + 1],
                    )
                )
            if long_range:
                a, b = sorted(
                    map(int, rng.choice(N, size=2, replace=False))
                )
                gates.append(
                    qtn.Gate.from_raw(
                        qu.rand_uni(4, seed=int(rng.integers(1 << 30))),
                        qubits=[a, b],
                    )
                )
        return gates

    @pytest.mark.parametrize("method", ["dm", "direct", "zipup", "fit"])
    def test_matches_dense_with_long_range(self, method):
        N = 8
        gates = self._random_circuit_gates(
            N, depth=3, seed=42, long_range=True
        )
        ref = qtn.Circuit(N)
        ref.apply_gates(gates)

        circ = qtn.CircuitMPSLazy(
            N, max_bond=2**N, cutoff=1e-12, method=method
        )
        circ.apply_gates(gates)
        mps = circ.psi

        assert isinstance(mps, qtn.MatrixProductState)
        assert mps.distance_normalized(ref.psi) < 1e-6
        assert circ.fidelity_estimate() == pytest.approx(1.0, abs=1e-6)

    def test_matches_eager_circuitmps(self):
        N = 7
        gates = self._random_circuit_gates(N, depth=3, seed=10)

        eager = qtn.CircuitMPS(N, max_bond=2**N, cutoff=1e-12)
        eager.apply_gates(gates)

        lazy = qtn.CircuitMPSLazy(N, max_bond=2**N, cutoff=1e-12)
        lazy.apply_gates(gates)

        assert lazy.psi.distance_normalized(eager.psi) < 1e-6

    def test_src_method_on_local_circuit(self):
        # the 'src' compression method supports local (non system-spanning)
        # circuits, which is the main use case for lazy compression
        N = 8
        gates = self._random_circuit_gates(N, depth=4, seed=7)
        ref = qtn.Circuit(N)
        ref.apply_gates(gates)

        circ = qtn.CircuitMPSLazy(N, max_bond=2**N, method="src")
        circ.apply_gates(gates)

        assert circ.psi.distance_normalized(ref.psi) < 1e-6

    def test_explicit_long_range_gate(self):
        N = 6
        gates = [
            qtn.Gate("H", params=(), qubits=[0]),
            qtn.Gate.from_raw(qu.rand_uni(4, seed=3), qubits=[0, N - 1]),
            qtn.Gate("CX", params=(), qubits=[N - 1, 2]),
        ]
        ref = qtn.Circuit(N)
        ref.apply_gates(gates)

        circ = qtn.CircuitMPSLazy(N, max_bond=2**N)
        circ.apply_gates(gates)

        assert circ.psi.distance_normalized(ref.psi) < 1e-6

    def test_single_qubit_gates_only(self):
        N = 5
        rng = np.random.default_rng(0)
        gates = [
            qtn.Gate("U3", params=rng.uniform(0, 2 * np.pi, 3), qubits=[i])
            for _ in range(3)
            for i in range(N)
        ]
        ref = qtn.Circuit(N)
        ref.apply_gates(gates)

        circ = qtn.CircuitMPSLazy(N, max_bond=8)
        circ.apply_gates(gates)
        mps = circ.psi

        assert isinstance(mps, qtn.MatrixProductState)
        # single qubit gates never grow the bond dimension
        assert mps.max_bond() == 1
        assert mps.distance_normalized(ref.psi) < 1e-6

    def test_psi_is_bounded_mps(self):
        N = 10
        max_bond = 8
        gates = self._random_circuit_gates(N, depth=6, seed=99)

        circ = qtn.CircuitMPSLazy(N, max_bond=max_bond, method="dm")
        circ.apply_gates(gates)
        mps = circ.psi

        assert isinstance(mps, qtn.MatrixProductState)
        assert mps.max_bond() <= max_bond

    def test_truncation_fidelity_increases_with_bond(self):
        N = 10
        gates = self._random_circuit_gates(N, depth=8, seed=5)

        fids = []
        for max_bond in [2, 4, 8, 16]:
            circ = qtn.CircuitMPSLazy(N, max_bond=max_bond, method="dm")
            circ.apply_gates(gates)
            f = circ.fidelity_estimate()
            assert 0.0 < f <= 1.0 + 1e-9
            assert circ.error_estimate() == pytest.approx(1 - f)
            fids.append(f)

        # raising the bond dimension should never reduce the fidelity
        for lo, hi in zip(fids, fids[1:]):
            assert hi >= lo - 1e-9
        assert fids[-1] > fids[0]

    def test_controlled_gate_then_lazy_keeps_fidelity_valid(self):
        # a controlled gate takes the eager fallback path and records an
        # orthogonality center; later lazy gates trigger a compression that
        # moves the center, so the recorded position must be cleared, else
        # fidelity_estimate and local_expectation read the wrong canonical form
        N = 6
        rng = np.random.default_rng(3)
        gates = [
            qtn.Gate("U3", params=rng.uniform(0, 2 * np.pi, 3), qubits=[i])
            for i in range(N)
        ]
        # controlled gate -> eager fallback path, records cur_orthog
        gates.append(
            qtn.Gate(
                "SU4",
                params=rng.uniform(0, 2 * np.pi, 15),
                qubits=[2, 4],
                controls=[0],
            )
        )
        # then lazily stacked two-qubit gates -> a compression on flush
        for i in range(0, N - 1, 2):
            gates.append(
                qtn.Gate(
                    "SU4",
                    params=rng.uniform(0, 2 * np.pi, 15),
                    qubits=[i, i + 1],
                )
            )

        ref = qtn.Circuit(N)
        ref.apply_gates(gates)
        psi_dense = ref.to_dense()

        circ = qtn.CircuitMPSLazy(
            N, max_bond=2**N, cutoff=1e-12, method="direct"
        )
        circ.apply_gates(gates)

        # no truncation, so the state is essentially exact
        assert circ.fidelity_estimate() == pytest.approx(1.0, abs=1e-6)
        assert circ.error_estimate() == pytest.approx(0.0, abs=1e-6)

        # local expectations must match the dense reference; a stale center
        # would silently shift these
        Z = qu.pauli("Z")
        for q in range(N):
            ref_z = qu.expec(qu.ikron(Z, [2] * N, q), psi_dense).real
            assert circ.local_expectation(Z, q).real == pytest.approx(
                ref_z, abs=1e-6
            )

    def test_lazy_defers_compression(self):
        # a single layer of disjoint two-qubit gates should be stacked and only
        # compressed once, on flush, rather than once per gate
        N = 8
        rng = np.random.default_rng(1)
        circ = qtn.CircuitMPSLazy(N, max_bond=16, method="dm")

        n_compress = 0
        real_compress = circ._compress

        def counting_compress():
            nonlocal n_compress
            if circ._pending:
                n_compress += 1
            real_compress()

        circ._compress = counting_compress

        for i in range(0, N - 1, 2):
            circ.apply_gates(
                [
                    qtn.Gate(
                        "SU4",
                        params=rng.uniform(0, 2 * np.pi, 15),
                        qubits=[i, i + 1],
                    )
                ]
            )
        # nothing compressed while the disjoint layer is being stacked
        assert n_compress == 0

        _ = circ.psi
        # exactly one global compression performed on flush
        assert n_compress == 1

    def test_2d_ising_dynamics(self):
        # small 2D transverse-field Ising Trotter dynamics mapped onto a 1D MPS;
        # the vertical couplings are long range in the 1D ordering
        Lx, Ly = 2, 3
        N = Lx * Ly

        def site(x, y):
            return x * Ly + y

        bonds = []
        for x in range(Lx):
            for y in range(Ly):
                if y + 1 < Ly:
                    bonds.append((site(x, y), site(x, y + 1)))
                if x + 1 < Lx:
                    bonds.append((site(x, y), site(x + 1, y)))

        dt = 0.1
        zz = qu.pauli("Z") & qu.pauli("Z")
        rzz = qu.expm(-1j * dt * zz)
        rx = qu.expm(-1j * dt * qu.pauli("X"))

        gates = []
        for _ in range(3):
            for i in range(N):
                gates.append(qtn.Gate.from_raw(rx, qubits=[i]))
            for a, b in bonds:
                gates.append(qtn.Gate.from_raw(rzz, qubits=[a, b]))

        ref = qtn.Circuit(N)
        ref.apply_gates(gates)

        circ = qtn.CircuitMPSLazy(N, max_bond=2**N, cutoff=1e-12, method="dm")
        circ.apply_gates(gates)

        assert circ.psi.distance_normalized(ref.psi) < 1e-6


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
