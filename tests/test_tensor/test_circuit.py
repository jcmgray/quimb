import pytest
import numpy as np

import quimb as qu
import quimb.tensor as qtn


def rand_reg_graph(reg, n, seed=None):
    import networkx as nx
    G = nx.random_regular_graph(reg, n, seed=seed)
    return G


def graph_to_circ(G, gamma0=-0.743043, beta0=0.754082):
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
        qasm = graph_to_circ(G)
        qc = qtn.Circuit.from_qasm(qasm)
        assert (qc.psi.H & qc.psi) ^ all == pytest.approx(1.0)

    def test_from_qasm_mps_swapsplit(self):
        G = rand_reg_graph(reg=3, n=18, seed=42)
        qasm = graph_to_circ(G)
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
            # two qubit
            ('cx', 2, 0),
            ('cy', 2, 0),
            ('cz', 2, 0),
            ('cnot', 2, 0),
            ('swap', 2, 0),
            ('iswap', 2, 0),
            # two qubit parametrizable
            ('fsim', 2, 2),
            ('rzz', 2, 1),
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
