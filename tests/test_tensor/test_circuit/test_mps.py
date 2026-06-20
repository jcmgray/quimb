import numpy as np
import pytest

import quimb as qu
import quimb.tensor as qtn

from ._helpers import graph_to_qsim, rand_reg_graph


def factor_pairs(N):
    factors = []
    for i in range(2, int(N**0.5) + 1):
        if N % i == 0:
            factors.append((i, N // i))
    return factors


def random_lattice_gates(N, num_layers=1, angle_range=None, seed=42):
    from itertools import product

    rng = np.random.default_rng(seed)

    if angle_range is None:
        angle_range = (0, 2 * np.pi)

    dimensions = rng.choice(factor_pairs(N))
    rows, cols = dimensions

    rxx = qu.expm(
        -1j * rng.uniform(*angle_range) * (qu.pauli("X") & qu.pauli("X"))
    )
    rzz = qu.expm(
        -1j * rng.uniform(*angle_range) * (qu.pauli("Z") & qu.pauli("Z"))
    )
    ryy = qu.expm(
        -1j * rng.uniform(*angle_range) * (qu.pauli("Y") & qu.pauli("Y"))
    )

    rx = qu.expm(-1j * rng.uniform(*angle_range) * qu.pauli("X"))
    ry = qu.expm(-1j * rng.uniform(*angle_range) * qu.pauli("Y"))
    rz = qu.expm(-1j * rng.uniform(*angle_range) * qu.pauli("Z"))

    single_site_gates = [rx, ry, rz]
    two_site_gates = [rxx, ryy, rzz]

    def site(x, y):
        return x * cols + y

    gates = []

    for _ in range(num_layers):
        for i in range(N):
            gates.append(
                qtn.Gate.from_raw(rng.choice(single_site_gates), qubits=[i])
            )
        for x, y in product(range(rows), range(cols)):
            if y + 1 < cols:
                a, b = site(x, y), site(x, y + 1)
                gates.append(
                    qtn.Gate.from_raw(
                        rng.choice(two_site_gates), qubits=[a, b]
                    )
                )
            if x + 1 < rows:
                a, b = site(x, y), site(x + 1, y)
                gates.append(
                    qtn.Gate.from_raw(
                        rng.choice(two_site_gates), qubits=[a, b]
                    )
                )

    return gates


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
        circ = qtn.CircuitMPSLazy(N)
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
        circ = qtn.CircuitMPSLazy(N)
        circ.h(0)
        samples = list(circ.sample(10, seed=1234))
        assert len(set(samples)) == 2

    @pytest.mark.parametrize("sweep_reverse", [False, True])
    def test_lazymps_local_expectation(self, sweep_reverse):
        circ = qtn.CircuitMPSLazy(
            3, compress_opts=dict(sweep_reverse=sweep_reverse)
        )
        circ.h(0)
        circ.cx(0, 1)
        circ._compress()

        if sweep_reverse:
            assert circ.gate_opts["info"]["cur_orthog"] == (2, 2)
        else:
            assert circ.gate_opts["info"]["cur_orthog"] == (0, 0)

        G = qu.rand_matrix(2)
        expec = circ.local_expectation(G, (1))
        assert circ.gate_opts["info"]["cur_orthog"] == (1, 1)

        psi = circ.to_dense()
        expec_dense = qu.expec(qu.ikron(G, [2, 2, 2], (1)), psi)
        assert expec == pytest.approx(expec_dense)

    @pytest.mark.parametrize("sweep_reverse", [False, True])
    def test_lazymps_fidelity_estimate(self, sweep_reverse):
        gates = random_lattice_gates(10, num_layers=2, seed=1234)

        circ = qtn.CircuitMPSLazy(
            10, max_bond=32, compress_opts=dict(sweep_reverse=sweep_reverse)
        )
        circ.apply_gates(gates)
        bond_32_fidelity = circ.fidelity_estimate()

        assert bond_32_fidelity == pytest.approx(1.0)

        circ = qtn.CircuitMPSLazy(
            10, max_bond=8, compress_opts=dict(sweep_reverse=sweep_reverse)
        )
        circ.apply_gates(gates)
        bond_8_fidelity = circ.fidelity_estimate()

        assert bond_8_fidelity == pytest.approx(0.8, abs=0.05)

        if sweep_reverse:
            assert circ.gate_opts["info"]["cur_orthog"] == (9, 9)
        else:
            assert circ.gate_opts["info"]["cur_orthog"] == (0, 0)

    def test_lazymps_fidelity_with_dense(self):
        rng = np.random.default_rng(42)
        N = 6

        circ = qtn.CircuitMPSLazy(N)
        for i in range(N - 1):
            circ.u3(
                rng.uniform(0, 2 * np.pi),
                rng.uniform(0, 2 * np.pi),
                rng.uniform(0, 2 * np.pi),
                i,
            )
            circ.cx(i, (i + 1))

        checker = qtn.Circuit(N)
        checker = checker.from_gates(circ.gates)

        assert circ.psi.norm() == pytest.approx(1.0)
        assert circ.psi.distance_normalized(checker.psi) < 1e-6

    def test_lazymps_compress_every(self):
        circ = qtn.CircuitMPSLazy(4, max_bond=2, compress_every=3)
        circ.h(0)
        assert circ._uncompressed_sites == {}
        circ.cx(0, 1)
        circ.cx(0, 2)
        circ.cx(1, 3)
        assert circ._uncompressed_sites == {0: 2, 1: 3, 2: 2, 3: 1}
        circ.cx(0, 1)
        assert circ._uncompressed_sites == {0: 1, 1: 1}
        _ = circ.psi
        assert circ._uncompressed_sites == {}

    def test_multi_controlled_lazymps_circuit(self):
        N = 10
        rng = np.random.default_rng(42)

        circ = qtn.CircuitMPSLazy(N)

        circ.apply_gate(
            qtn.Gate(
                "SU4",
                params=rng.uniform(0, 2 * np.pi, size=15),
                qubits=[6, 2],
                controls=[8, 3, 4, 0],
            )
        )
        assert circ._uncompressed_sites == {
            0: 1,
            1: 1,
            2: 1,
            3: 1,
            4: 1,
            5: 1,
            6: 1,
            7: 1,
            8: 1,
        }

        circ.apply_gate(
            qtn.Gate.from_raw(
                qu.rand_uni(2**3),
                qubits=[0, 9, 5],
                controls=[1, 2, 7],
            )
        )
        assert circ._uncompressed_sites == {
            0: 2,
            1: 2,
            2: 2,
            3: 2,
            4: 2,
            5: 2,
            6: 2,
            7: 2,
            8: 2,
            9: 1,
        }

        checker = qtn.Circuit(N=10)
        checker.apply_gates(circ.gates)

        assert circ.psi.norm() == pytest.approx(1.0)
        assert circ.psi.distance_normalized(checker.psi) < 1e-6

    @pytest.mark.parametrize(
        "method", ["dm", "direct", "src", "srcmps", "zipup", "zipup-first"]
    )
    def test_lazymps_2d_long_range_dynamics_without_compression(self, method):
        N = 10

        gates = random_lattice_gates(N, num_layers=4)

        # bond 32 would be sufficient to represent the state exactly, but
        # "zipup" requires bond 64 to be exact
        circ = qtn.CircuitMPSLazy(
            N, max_bond=64, method=method, compress_every=4
        )
        circ.apply_gates(gates)
        lazy_state = circ.psi

        circ = qtn.Circuit(N)
        circ.apply_gates(gates)
        dense_state = circ.psi

        fidelity = np.abs(lazy_state.H @ dense_state) ** 2

        assert fidelity == pytest.approx(1.0, abs=1e-10), (
            f"Fidelity too low for N={N}, method={method}"
        )

    @pytest.mark.parametrize(
        "method", ["dm", "direct", "src", "srcmps", "zipup", "zipup-first"]
    )
    def test_lazymps_2d_long_range_dynamics_with_nontrivial_compression(
        self, method
    ):
        N = 10

        # create a long-range circuit on 10 qubits, which requires bond dimension 16
        # to be represented exactly
        gates = random_lattice_gates(
            N, num_layers=3, angle_range=(0.0, 0.1), seed=1234
        )

        circ = qtn.CircuitMPSLazy(
            N, max_bond=8, method=method, compress_every=4
        )
        circ.apply_gates(gates)
        lazy_state = circ.psi

        circ = qtn.Circuit(N)
        circ.apply_gates(gates)
        dense_state = circ.psi

        if method in {"src"}:
            baseline = 0.95
        elif method in {"srcmps", "zipup-first"}:
            baseline = 0.96
        else:
            baseline = 0.99

        fidelity = np.abs(lazy_state.H @ dense_state) ** 2

        assert fidelity >= baseline, (
            f"Fidelity too low for N={N}, method={method}"
        )
