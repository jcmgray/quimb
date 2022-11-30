import math
import itertools

import pytest
import numpy as np
from numpy.testing import assert_allclose

import quimb as qu
import quimb.tensor as qtn


def rand_reg_graph(reg, n, seed=None):
    import networkx as nx
    G = nx.random_regular_graph(reg, n, seed=seed)
    return G


def graph_to_qasm(G, gamma0=-0.743043, beta0=0.754082):
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
        gates.append((0, 'h', i))

    for d in range(depth):
        rng.shuffle(qubits)

        for i in range(0, L - 1, 2):
            g = rng.choice(['cx', 'cy', 'cz', 'iswap'])
            gates.append((d, g, qubits[i], qubits[i + 1]))

        for q in qubits:
            g = rng.choice(['rx', 'ry', 'rz'])
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

            gate = np.random.choice(['FSIM', 'SWAP'])
            if gate == 'FSIM':
                params = np.random.randn(2)
            elif gate == 'FSIMG':
                params = np.random.randn(5)
            else:
                params = ()

            circ.apply_gate(gate, *params, qi, qj)

    return circ


class TestCircuit:

    def test_prepare_GHZ(self):
        qc = qtn.Circuit(3)
        gates = [
            ('H', 0),
            ('H', 1),
            ('CNOT', 1, 2),
            ('CNOT', 0, 2),
            ('H', 0),
            ('H', 1),
            ('H', 2),
        ]
        qc.apply_gates(gates)
        assert qu.expec(qc.psi.to_dense(), qu.ghz_state(3)) == pytest.approx(1)
        counts = qc.simulate_counts(1024)
        assert len(counts) == 2
        assert '000' in counts
        assert '111' in counts
        assert counts['000'] + counts['111'] == 1024

    def test_from_qasm(self):
        G = rand_reg_graph(reg=3, n=18, seed=42)
        qasm = graph_to_qasm(G)
        qc = qtn.Circuit.from_qasm(qasm)
        assert (qc.psi.H & qc.psi) ^ all == pytest.approx(1.0)

    def test_from_qasm_mps_swapsplit(self):
        G = rand_reg_graph(reg=3, n=18, seed=42)
        qasm = graph_to_qasm(G)
        qc = qtn.CircuitMPS.from_qasm(qasm)
        assert len(qc.psi.tensors) == 18
        assert (qc.psi.H & qc.psi) ^ all == pytest.approx(1.0)

    @pytest.mark.parametrize(
        'Circ', [qtn.Circuit, qtn.CircuitMPS, qtn.CircuitDense]
    )
    def test_all_gate_methods(self, Circ):
        import random

        g_nq_np = [
            # single qubit
            ('x', 1, 0),
            ('y', 1, 0),
            ('z', 1, 0),
            ('s', 1, 0),
            ('t', 1, 0),
            ('h', 1, 0),
            ('iden', 1, 0),
            ('x_1_2', 1, 0),
            ('y_1_2', 1, 0),
            ('z_1_2', 1, 0),
            ('w_1_2', 1, 0),
            ('hz_1_2', 1, 0),
            # single qubit parametrizable
            ('rx', 1, 1),
            ('ry', 1, 1),
            ('rz', 1, 1),
            ('u3', 1, 3),
            ('u2', 1, 2),
            ('u1', 1, 1),
            # two qubit
            ('cx', 2, 0),
            ('cy', 2, 0),
            ('cz', 2, 0),
            ('cnot', 2, 0),
            ('swap', 2, 0),
            ('iswap', 2, 0),
            # two qubit parametrizable
            ('cu3', 2, 3),
            ('cu2', 2, 2),
            ('cu1', 2, 1),
            ('fsim', 2, 2),
            ('fsimg', 2, 5),
            ('rzz', 2, 1),
            ('su4', 2, 15),
        ]
        random.shuffle(g_nq_np)

        psi0 = qtn.MPS_rand_state(2, 2)
        circ = Circ(2, psi0, tags='PSI0')

        for g, n_q, n_p in g_nq_np:
            args = [
                *np.random.uniform(0, 2 * np.pi, size=n_p),
                *np.random.choice([0, 1], replace=False, size=n_q)
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
        (theta1, phi1, lamda1,
         theta2, phi2, lamda2,
         theta3, phi3, lamda3,
         theta4, phi4, lamda4,
         t1, t2, t3,) = params
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

    def test_auto_split_gate(self):

        n = 3
        ops = [
            ('u3', 1., 2., 3., 0),
            ('u3', 2., 3., 1., 1),
            ('u3', 3., 1., 2., 2),
            ('cz', 0, 1),
            ('iswap', 1, 2),
            ('cx', 2, 0),
            ('iswap', 2, 1),
            ('h', 0),
            ('h', 1),
            ('h', 2),
        ]
        cnorm = qtn.Circuit(n, gate_opts=dict(contract='split-gate'))
        cnorm.apply_gates(ops)
        assert cnorm.psi.max_bond() == 4

        cswap = qtn.Circuit(n, gate_opts=dict(contract='swap-split-gate'))
        cswap.apply_gates(ops)
        assert cswap.psi.max_bond() == 4

        cauto = qtn.Circuit(n, gate_opts=dict(contract='auto-split-gate'))
        cauto.apply_gates(ops)
        assert cauto.psi.max_bond() == 2

        assert qu.fidelity(cnorm.psi.to_dense(),
                           cswap.psi.to_dense()) == pytest.approx(1.0)
        assert qu.fidelity(cswap.psi.to_dense(),
                           cauto.psi.to_dense()) == pytest.approx(1.0)

    @pytest.mark.parametrize("gate2", ['cx', 'iswap'])
    def test_circuit_simplify_tensor_network(self, gate2):
        import random
        import itertools

        depth = n = 8

        circ = qtn.Circuit(n)

        def random_single_qubit_layer():
            return [
                (random.choice(['X_1_2', 'Y_1_2', 'W_1_2']), i)
                for i in range(n)
            ]

        def even_two_qubit_layer():
            return [
                (gate2, i, i + 1)
                for i in range(0, n, 2)
            ]

        def odd_two_qubit_layer():
            return [
                (gate2, i, i + 1)
                for i in range(1, n - 1, 2)
            ]

        layering = itertools.cycle([
            random_single_qubit_layer,
            even_two_qubit_layer,
            random_single_qubit_layer,
            odd_two_qubit_layer,
        ])

        for i, layer_fn in zip(range(depth), layering):
            for g in layer_fn():
                circ.apply_gate(*g, gate_round=i)

        psif = qtn.MPS_computational_state('0' * n).squeeze_()
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
            assert_allclose(qu.partial_trace(psi, [2] * 5, keep=keep),
                            circ.partial_trace(keep),
                            atol=1e-12)

    @pytest.mark.parametrize("group_size", (1, 2, 6))
    def test_sample(self, group_size):
        import collections
        from scipy.stats import power_divergence

        C = 2**10
        L = 5
        circ = random_a2a_circ(L, 3)

        psi = circ.to_dense()
        p_exp = abs(psi.reshape(-1))**2
        f_exp = p_exp * C

        counts = collections.Counter(circ.sample(C, group_size=group_size))
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

        for _ in range(reps):
            circ = random_a2a_circ(L, depth)

            psi = circ.to_dense()
            p_exp = abs(psi.reshape(-1))**2
            f_exp = p_exp * C

            for num_marginal in [3, 4, 5]:
                counts = collections.Counter(
                    circ.sample_chaotic(C, num_marginal, seed=666)
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
        Gs = [qu.kronpow(qu.pauli(s), 2) for s in 'xyz']
        exps = circ.local_expectation(Gs, [0, 1])
        assert exps[0] == pytest.approx(-1)
        assert exps[1] == pytest.approx(-1)
        assert exps[2] == pytest.approx(-1)

    def test_uni_to_dense(self):
        import cmath
        circ = qft_circ(3)
        U = circ.uni.to_dense()
        w = cmath.exp(2j * math.pi / 2**3)
        ex = 2**(-3 / 2) * np.array(
            [[w**0, w**0, w**0, w**0, w**0, w**0, w**0, w**0],
             [w**0, w**1, w**2, w**3, w**4, w**5, w**6, w**7],
             [w**0, w**2, w**4, w**6, w**0, w**2, w**4, w**6],
             [w**0, w**3, w**6, w**1, w**4, w**7, w**2, w**5],
             [w**0, w**4, w**0, w**4, w**0, w**4, w**0, w**4],
             [w**0, w**5, w**2, w**7, w**4, w**1, w**6, w**3],
             [w**0, w**6, w**4, w**2, w**0, w**6, w**4, w**2],
             [w**0, w**7, w**6, w**5, w**4, w**3, w**2, w**1]])
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
            'PSI0', 'GATE_0', 'GATE_2', 'GATE_4'
        )

    def test_swappy_local_expecs(self):
        circ = swappy_circ(4, 4)
        Gs = [qu.rand_matrix(4) for _ in range(3)]
        pairs = [(0, 1), (1, 2), (2, 3)]

        psi = circ.to_dense()
        dims = [2] * 4

        exs = [qu.expec(qu.ikron(G, dims, pair), psi)
               for G, pair in zip(Gs, pairs)]
        aps = [circ.local_expectation(G, pair) for G, pair in zip(Gs, pairs)]

        assert_allclose(exs, aps)

    @pytest.mark.parametrize(
        "name, densefn, nparam, nqubit",
        [
            ('rx', qu.Rx, 1, 1),
            ('ry', qu.Ry, 1, 1),
            ('rz', qu.Rz, 1, 1),
            ('u3', qu.U_gate, 3, 1),
            ('fsim', qu.fsim, 2, 2),
            ('fsimg', qu.fsimg, 5, 2),
        ]
    )
    def test_parametrized_gates_rx(self, name, densefn, nparam, nqubit):
        k0 = qu.rand_ket(2**nqubit)
        params = qu.randn(nparam)
        kf = densefn(*params) @ k0
        k0mps = qtn.MatrixProductState.from_dense(k0, [2] * nqubit)
        circ = qtn.Circuit(psi0=k0mps, gate_opts={'contract': False})
        getattr(circ, name)(*params, *range(nqubit), parametrize=True)
        tn = circ.psi
        assert isinstance(tn['GATE_0'], qtn.PTensor)
        assert_allclose(circ.to_dense(), kf)

    def test_apply_raw_gate(self):
        k0 = qu.rand_ket(4)
        psi0 = qtn.MatrixProductState.from_dense(k0, [2] * 2)
        circ = qtn.Circuit(psi0=psi0)
        U = qu.rand_uni(4)
        circ.apply_gate_raw(U, [0, 1], tags='UCUSTOM')
        assert len(circ.gates) == 1
        assert 'UCUSTOM' in circ.psi.tags
        assert qu.fidelity(circ.to_dense(), U @ k0) == pytest.approx(1)


class TestCircuitGen:

    @pytest.mark.parametrize(
        "ansatz,cyclic", [
            ('zigzag', False),
            ('brickwork', False),
            ('brickwork', True),
            ('rand', False),
            ('rand', True),
        ])
    @pytest.mark.parametrize('n', [4, 5])
    def test_1D_ansatzes(self, ansatz, cyclic, n):
        depth = 3
        num_pairs = n if cyclic else n - 1

        fn = {
            'zigzag': qtn.circ_ansatz_1D_zigzag,
            'brickwork': qtn.circ_ansatz_1D_brickwork,
            'rand': qtn.circ_ansatz_1D_rand,
        }[ansatz]

        opts = dict(
            n=n,
            depth=3,
            gate_opts=dict(contract=False),
        )
        if cyclic:
            opts['cyclic'] = True
        if ansatz == 'rand':
            opts['seed'] = 42

        circ = fn(**opts)
        tn = circ.uni

        # total number of entangling gates
        assert len(tn['CZ']) == num_pairs * depth

        # number of entangling gates per pair
        for i in range(num_pairs):
            assert len(tn['CZ', f'I{i}', f'I{(i + 1) % n}']) == depth

        assert all(isinstance(t, qtn.PTensor) for t in tn['U3'])

    def test_qaoa(self):
        G = rand_reg_graph(3, 10, seed=666)
        terms = {(i, j): 1. for i, j in G.edges}
        ZZ = qu.pauli('Z') & qu.pauli('Z')

        gammas = [1.2]
        betas = [-0.4]

        circ1 = qtn.circ_qaoa(terms, 1, gammas, betas)

        energy1 = sum(
            circ1.local_expectation(ZZ, edge)
            for edge in terms
        )
        assert energy1 < -4

        gammas = [0.4]
        betas = [0.3]

        circ2 = qtn.circ_qaoa(terms, 1, gammas, betas)

        energy2 = sum(
            circ2.local_expectation(ZZ, edge)
            for edge in terms
        )
        assert energy2 > 4
