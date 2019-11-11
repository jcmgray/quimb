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
        circ += f"CNOT {i} {j}\n"
        circ += f"Rz {gamma0} {j}\n"
        circ += f"CNOT {i} {j}\n"
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

    def test_rand_reg_qaoa(self):
        G = rand_reg_graph(reg=3, n=18, seed=42)
        qasm = graph_to_circ(G)
        qc = qtn.Circuit.from_qasm(qasm)
        assert (qc.psi.H & qc.psi) ^ all == pytest.approx(1.0)

    def test_rand_reg_qaoa_mps_swapsplit(self):
        G = rand_reg_graph(reg=3, n=18, seed=42)
        qasm = graph_to_circ(G)
        qc = qtn.CircuitMPS.from_qasm(qasm)
        assert len(qc.psi.tensors) == 18
        assert (qc.psi.H & qc.psi) ^ all == pytest.approx(1.0)

    @pytest.mark.parametrize(
        'Circ', [qtn.Circuit, qtn.CircuitMPS, qtn.CircuitDense]
    )
    def test_all_gate_methods(self, Circ):
        rots = ['rx', 'ry', 'rz']
        g1s = ['x', 'y', 'z', 's', 't', 'h', 'iden']
        g2s = ['cx', 'cy', 'cz', 'cnot', 'swap']
        g_rand = np.random.permutation(rots + g1s + g2s + ['u3'])

        psi0 = qtn.MPS_rand_state(2, 2)
        circ = Circ(2, psi0)

        for g in g_rand:
            if g == 'u3':
                angles = np.random.uniform(0, 2 * np.pi, size=3)
                i = np.random.choice([0, 1])
                args = (*angles, i)
            elif g in rots:
                theta = np.random.uniform(0, 2 * np.pi)
                i = np.random.choice([0, 1])
                args = (theta, i)
            elif g in g1s:
                i = np.random.choice([0, 1])
                args = (i,)
            elif g in g2s:
                i, j = np.random.permutation([0, 1])
                args = (i, j)

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

        c = tn.contract(all, optimize='greedy')
        cw = tn.contraction_width(optimize='greedy')

        # absorb all low-rank tensors into neighbours
        tn_rs = tn.rank_simplify()

        assert len(tn_rs.tensors) < len(tn.tensors)
        assert all(t.ndim > 2 for t in tn_rs)

        c_rs = tn_rs.contract(all, optimize='greedy')
        assert c == pytest.approx(c_rs)
        cw_rs = tn_rs.contraction_width(optimize='greedy')
        assert cw_rs <= cw

        tn_rs_dr = tn_rs.diagonal_reduce()
        assert len(tn_rs_dr.ind_map) < len(tn_rs.ind_map)

        # need to specify output inds since we now have hyper edges
        c_rs_dr = tn_rs_dr.contract(all, optimize='greedy', output_inds=[])
        assert c_rs_dr == pytest.approx(c)
        cw_rs_dr = tn_rs_dr.contraction_width(optimize='greedy',
                                              output_inds=[])
        assert cw_rs_dr <= cw_rs
