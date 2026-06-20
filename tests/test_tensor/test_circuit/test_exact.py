import itertools
import math

import numpy as np
import pytest
from numpy.testing import assert_allclose

import quimb as qu
import quimb.tensor as qtn


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


class TestCircuitParams:
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
