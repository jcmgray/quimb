import numpy as np
import pytest
from numpy.testing import assert_allclose

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

        psi_lazy = qtn.Circuit.from_gates(gates).psi
        mps = qtn.CircuitMPS.from_gates(gates).psi
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

    def _entangling_gates(self, N=10, depth=6, seed=1):
        rng = np.random.default_rng(seed)
        gates = [("H", i) for i in range(N)]
        for d in range(depth):
            for i in range(d % 2, N - 1, 2):
                gates.append(("CX", i, i + 1))
            for i in range(N):
                gates.append(("RY", float(rng.uniform(0, 2 * np.pi)), i))
        return gates

    def test_max_bond_truncates_the_state(self):
        gates = self._entangling_gates()
        full = qtn.CircuitMPS.from_gates(gates)
        trunc = qtn.CircuitMPS.from_gates(gates, max_bond=4)
        assert trunc.psi.max_bond() == 4
        assert full.psi.max_bond() > 4

    def test_fidelity_and_error_estimate(self):
        gates = self._entangling_gates()
        full = qtn.CircuitMPS.from_gates(gates)
        assert full.fidelity_estimate() == pytest.approx(1.0, abs=1e-10)
        assert full.error_estimate() == pytest.approx(0.0, abs=1e-10)

        trunc = qtn.CircuitMPS.from_gates(gates, max_bond=4)
        f = trunc.fidelity_estimate()
        assert 0.0 < f < 1.0
        # error estimate is the complementary norm loss
        assert trunc.error_estimate() == pytest.approx(1.0 - f, abs=1e-10)

    def test_uni_unsupported(self):
        circ = qtn.CircuitMPS(3)
        circ.h(0)
        with pytest.raises(ValueError):
            circ.uni


class TestCircuitPermMPS:
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

    def _permuting_circuit(self):
        # nonlocal 2-qubit gates make the lazy qubit permutation non-trivial
        gates = [("H", 0), ("CX", 0, 3), ("CX", 1, 4), ("RY", 0.3, 2)]
        return qtn.CircuitPermMPS.from_gates(gates), gates

    def test_qubit_ordering_tracks_permutation(self):
        cp, _ = self._permuting_circuit()
        order = tuple(cp.calc_qubit_ordering())
        assert sorted(order) == list(range(5))  # a genuine permutation
        assert order != tuple(range(5))  # and a non-trivial one

    def test_get_psi_unordered_is_raw_mps(self):
        cp, _ = self._permuting_circuit()
        raw = cp.get_psi_unordered()
        assert isinstance(raw, qtn.MatrixProductState)
        # the public psi is a general tensor network, not an MPS
        assert not isinstance(cp.psi, qtn.MatrixProductState)

    def test_adjacent_su4_matches_exact(self):
        # an adjacent 2q gate induces no permutation, so psi/to_dense are correct
        rng = np.random.default_rng(7)
        su4 = tuple(rng.uniform(0, 2 * np.pi, 15))
        gates = [("H", 0), ("H", 1), ("SU4", *su4, 0, 1)]
        # qubit 2 is idle, so pass N explicitly (from_gates would infer N=2)
        cp = qtn.CircuitPermMPS.from_gates(gates, N=3)
        ce = qtn.Circuit.from_gates(gates, N=3)
        assert tuple(cp.qubits) == (0, 1, 2)  # no permutation induced
        assert abs(qu.fidelity(cp.to_dense(), ce.to_dense())) == pytest.approx(
            1.0, abs=1e-10
        )

    def test_sample_correct_under_permutation(self):
        # sample() inverts the lazy permutation back to logical order
        # logical state is (|000> + |101>)/√2
        cp = qtn.CircuitPermMPS.from_gates([("H", 0), ("CX", 0, 2)])
        assert tuple(cp.qubits) != (0, 1, 2)
        assert set(cp.sample(50, seed=1)) == {"000", "101"}

    def test_controlled_gate_via_controls_kwarg_unsupported(self):
        # controlled gates (controls=) are unsupported under the swap+split path
        cp = qtn.CircuitPermMPS(3)
        cp.h(0)
        with pytest.raises(ValueError):
            cp.apply_gate("X", 2, controls=(0,))

    def test_amplitude_to_dense_correct_under_permutation(self):
        # the lazy permutation is relabelled back to logical order for the
        # exact-contraction entry points: H(0);CX(0,2) is (|000> + |101>)/√2
        cp = qtn.CircuitPermMPS.from_gates([("H", 0), ("CX", 0, 2)])
        ce = qtn.Circuit.from_gates([("H", 0), ("CX", 0, 2)])
        assert tuple(cp.qubits) != (0, 1, 2)  # a non-trivial permutation
        assert cp.amplitude("101") == pytest.approx(ce.amplitude("101"))
        assert cp.amplitude("110") == pytest.approx(0.0)
        assert abs(qu.fidelity(cp.to_dense(), ce.to_dense())) == pytest.approx(
            1.0, abs=1e-10
        )

    def test_observables_correct_under_3cycle_permutation(self):
        # a genuine 3-cycle (not self-inverse) with non-symmetric amplitudes:
        # distinguishes "permutation applied" from "applied twice / not at all"
        N = 4
        gates = [
            ("RY", 0.7, 0),
            ("RY", 1.1, 1),
            ("RY", 0.3, 2),
            ("RY", 0.5, 3),
            ("CX", 0, 2),
            ("CX", 0, 3),
            ("CX", 1, 3),
        ]
        cp = qtn.CircuitPermMPS.from_gates(gates)
        ce = qtn.Circuit.from_gates(gates)
        order = tuple(cp.qubits)
        inverse = [0] * N
        for site, logical in enumerate(order):
            inverse[logical] = site
        # genuine 3-cycle: not identity, and not self-inverse (transposition)
        assert order not in (tuple(range(N)), tuple(inverse))

        assert_allclose(cp.to_dense(), ce.to_dense(), atol=1e-10)
        for i in range(2**N):
            b = f"{i:0{N}b}"
            assert cp.amplitude(b) == pytest.approx(ce.amplitude(b), abs=1e-10)
        assert_allclose(
            cp.partial_trace([0, 1]), ce.partial_trace([0, 1]), atol=1e-10
        )
        for i in range(N):
            assert cp.local_expectation(qu.pauli("Z"), i) == pytest.approx(
                ce.local_expectation(qu.pauli("Z"), i)
            )

    def test_long_range_local_expectation_under_permutation(self):
        # a 2-site observable whose logical qubits land on non-adjacent
        # *physical* sites after the lazy permutation
        N = 6
        gates = [
            ("H", 0),
            ("RY", 0.7, 1),
            ("RY", 1.1, 2),
            ("RY", 0.3, 3),
            ("RY", 0.9, 4),
            ("RY", 0.5, 5),
            ("CX", 0, 2),
            ("CX", 1, 4),
            ("CZ", 0, 5),
            ("RZZ", 0.4, 2, 5),
        ]
        cp = qtn.CircuitPermMPS.from_gates(gates)
        ce = qtn.Circuit.from_gates(gates)
        assert tuple(cp.qubits) != tuple(range(N))  # non-trivial permutation
        ZZ = qu.pauli("Z") & qu.pauli("Z")
        for where in [(0, 5), (1, 4), (0, 3)]:
            assert cp.local_expectation(ZZ, where) == pytest.approx(
                ce.local_expectation(ZZ, where), abs=1e-10
            )

    def test_copy_preserves_qubit_permutation(self):
        gates = [("H", 0), ("CX", 0, 3), ("RY", 0.3, 2)]
        circ = qtn.CircuitPermMPS.from_gates(gates)
        assert tuple(circ.qubits) != tuple(range(4))

        copied = circ.copy()
        assert copied.qubits == circ.qubits
        assert copied.qubits is not circ.qubits
        assert copied.psi.distance_normalized(circ.psi) < 1e-6

        # further gates on the copy leave the original untouched
        qubits_before = list(circ.qubits)
        copied.cx(1, 2)
        assert circ.qubits == qubits_before
        assert circ.num_gates == len(gates)


class TestCircuitMPSLazy:
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

        checker = qtn.Circuit.from_gates(circ.gates)

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

        dense_state = qtn.Circuit.from_gates(gates).psi

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

        dense_state = qtn.Circuit.from_gates(gates).psi

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

    def test_single_qubit_gates_stay_eager(self):
        circ = qtn.CircuitMPSLazy(4)
        for i in range(4):
            circ.h(i)
            circ.rx(0.3, i)
        # eager 1q gates leave nothing pending and never grow a bond
        assert dict(circ._uncompressed_sites) == {}
        assert circ.psi.max_bond() == 1

    def test_two_qubit_gates_deferred_until_flush(self):
        circ = qtn.CircuitMPSLazy(4, max_bond=8, compress_every=10)
        for i in range(3):
            circ.cx(i, i + 1)
        # pending, not yet compressed
        assert dict(circ._uncompressed_sites)

        n_calls = {"n": 0}
        orig = circ._compress

        def counted():
            n_calls["n"] += 1
            return orig()

        circ._compress = counted
        _ = circ.psi  # a state access flushes exactly once
        assert n_calls["n"] == 1
        assert dict(circ._uncompressed_sites) == {}

    def test_special_gates_match_exact(self):
        gates = [("H", 0), ("H", 1), ("CX", 0, 1), ("SWAP", 1, 2), ("IDEN", 0)]
        ce = qtn.Circuit.from_gates(gates)
        cl = qtn.CircuitMPSLazy.from_gates(gates)
        assert abs(qu.fidelity(cl.to_dense(), ce.to_dense())) == pytest.approx(
            1.0, abs=1e-10
        )

    def test_property_setters_sync(self):
        circ = qtn.CircuitMPSLazy(3)
        circ.max_bond = 16
        circ.cutoff = 1e-9
        circ.method = "dm"
        assert circ.max_bond == 16
        assert circ.cutoff == 1e-9
        assert circ.method == "dm"
        # both the per-flush compress opts and the gate-application opts track
        assert circ.gate_opts["max_bond"] == 16
        assert circ.compress_opts["max_bond"] == 16
        assert circ.compress_opts["cutoff"] == 1e-9
        assert circ.compress_opts["method"] == "dm"

    def test_copy_with_pending_gates(self):
        circ = qtn.CircuitMPSLazy(4, max_bond=8, compress_every=4)
        circ.h(0)
        circ.cx(0, 3)
        assert dict(circ._uncompressed_sites)

        copied = circ.copy()
        assert copied.compress_every == circ.compress_every
        assert copied.compress_opts == circ.compress_opts
        assert copied.compress_opts is not circ.compress_opts
        assert copied._uncompressed_sites == circ._uncompressed_sites
        assert copied._uncompressed_sites is not circ._uncompressed_sites

        # the copy represents the same state ...
        psi_orig = circ.psi
        assert copied.psi.distance_normalized(psi_orig) < 1e-6

        # ... and is independent of the original
        copied.x(1)
        assert circ.num_gates == 2
        assert circ.psi.distance_normalized(psi_orig) < 1e-6

    @staticmethod
    def _pending_truncating_ghz():
        # an exact GHZ state is not representable at bond dimension 1, so the
        # configured compression visibly truncates (discarding half the
        # norm), distinguishing the pending exact TN from the compressed MPS
        circ = qtn.CircuitMPSLazy(
            6, max_bond=1, cutoff=0.0, method="dm", compress_every=100
        )
        circ.h(0)
        for i in range(5):
            circ.cx(i, i + 1)
        assert dict(circ._uncompressed_sites)
        return circ

    def test_to_dense_flushes_and_matches_psi(self):
        circ = self._pending_truncating_ghz()
        dense = np.asarray(circ.to_dense()).reshape(-1)
        assert not circ._uncompressed_sites
        k = np.asarray(circ.psi.to_dense()).reshape(-1)
        assert_allclose(dense, k, atol=1e-10)
        assert np.linalg.norm(dense) ** 2 == pytest.approx(0.5)

    def test_amplitude_flushes_and_matches_psi(self):
        circ = self._pending_truncating_ghz()
        amp = circ.amplitude("1" * 6)
        assert not circ._uncompressed_sites
        k = np.asarray(circ.psi.to_dense()).reshape(-1)
        assert amp == pytest.approx(complex(k[-1]), abs=1e-10)

    def test_partial_trace_flushes_and_matches_psi(self):
        circ = self._pending_truncating_ghz()
        rho = np.asarray(circ.partial_trace((0,)))
        assert not circ._uncompressed_sites
        kk = np.asarray(circ.psi.to_dense()).reshape(2, 32)
        assert_allclose(rho, kk @ kk.conj().T, atol=1e-10)

    def test_compute_marginal_flushes_and_matches_psi(self):
        circ = self._pending_truncating_ghz()
        marginal = np.asarray(circ.compute_marginal((0,))).reshape(-1)
        assert not circ._uncompressed_sites
        k = np.asarray(circ.psi.to_dense()).reshape(2, 32)
        assert_allclose(marginal, (np.abs(k) ** 2).sum(axis=1), atol=1e-10)

    def test_accessor_order_independence(self):
        # the sequence from issue #387: to_dense before and after accessing
        # psi must describe the same (compressed) state
        circ = self._pending_truncating_ghz()
        dense_before = np.asarray(circ.to_dense()).reshape(-1)
        k = np.asarray(circ.psi.to_dense()).reshape(-1)
        dense_after = np.asarray(circ.to_dense()).reshape(-1)
        assert_allclose(dense_before, k, atol=1e-10)
        assert_allclose(dense_after, k, atol=1e-10)

    def test_compress_clears_cached_simplified_state(self):
        circ = qtn.CircuitMPSLazy(4, max_bond=2)
        circ.h(0)
        circ.cx(0, 3)
        # a cached simplified state from before the compression is stale
        circ._storage[("psi_simplified", "R", 1e-12)] = circ._psi.copy()
        circ._compress()
        assert circ._storage == {}

    def test_schrodinger_contract_raises(self):
        circ = qtn.CircuitMPSLazy(3, max_bond=2)
        circ.h(0)
        circ.cx(0, 2)
        with pytest.raises(NotImplementedError):
            circ.schrodinger_contract()
