import numpy as np
import pytest
from numpy.testing import assert_allclose

import quimb as qu
import quimb.tensor as qtn


def rand_circuit_gates(sites, edges, n_layers=2, seed=0):
    """Generate a fixed sequence of gates in quimb's ``(array, *qubits)``
    format: each layer is a round of single site gates followed by a round of
    two site gates on the edges.
    """
    rng = np.random.default_rng(seed)
    gates = []
    for _ in range(n_layers):
        for s in sites:
            gates.append((qu.rand_uni(2, seed=int(rng.integers(1 << 30))), s))
        for a, b in edges:
            gates.append(
                (qu.rand_uni(4, seed=int(rng.integers(1 << 30))), a, b)
            )
    return gates


def test_chain_matches_mps_exactly():
    # a 1D chain is a tree, so simple update (with equilibrated gauges) is
    # exact and must agree with an (exact) MPS circuit simulation
    n = 6
    edges = [(i, i + 1) for i in range(n - 1)]
    gates = rand_circuit_gates(range(n), edges, n_layers=2, seed=42)

    cmps = qtn.CircuitMPS(N=n, max_bond=64, cutoff=0.0)
    csu = qtn.CircuitPEPSSimpleUpdate(edges, max_bond=64, cutoff=1e-14)
    cmps.apply_gates(gates)
    csu.apply_gates(gates)

    # restore exact canonical form before measuring
    csu.equilibrate(max_iterations=500, tol=1e-13)

    for i in range(n):
        ref = cmps.local_expectation(qu.pauli("Z"), i, normalized=True)
        val = csu.local_expectation(qu.pauli("Z"), i, max_distance=0)
        assert_allclose(complex(val), complex(ref), atol=1e-10)


def test_psi_is_a_peps_and_truncates_bond():
    edges = qtn.edges_2d_square(3, 3)
    max_bond = 4
    csu = qtn.CircuitPEPSSimpleUpdate(edges, max_bond=max_bond)
    csu.apply_gates(
        rand_circuit_gates(csu.sites, csu.edges, n_layers=3, seed=1)
    )

    psi = csu.psi
    assert isinstance(psi, qtn.TensorNetworkGenVector)
    assert psi.num_tensors == len(csu.sites)
    assert psi.max_bond() <= max_bond


def test_non_edge_and_too_many_sites_raise():
    edges = qtn.edges_2d_square(2, 2)
    csu = qtn.CircuitPEPSSimpleUpdate(edges, max_bond=8)

    # (0, 0) and (1, 1) are diagonal, not a declared edge
    with pytest.raises(ValueError):
        csu.apply_gate(qu.rand_uni(4), (0, 0), (1, 1))

    # three site gates are not supported
    with pytest.raises(ValueError):
        csu.apply_gate(qu.rand_uni(8), (0, 0), (0, 1), (1, 0))


def test_geometry_inferred_from_gates():
    gates = [
        (qu.rand_uni(2), 0),
        (qu.rand_uni(4), 0, 1),
        (qu.rand_uni(4), 1, 2),
    ]
    csu = qtn.CircuitPEPSSimpleUpdate(gates=gates, max_bond=8)
    assert set(csu.sites) == {0, 1, 2}
    assert set(csu.edges) == {(0, 1), (1, 2)}
    # the inferred geometry should accept all the gates
    csu.apply_gates(gates)


def test_copy_is_independent():
    edges = qtn.edges_2d_square(2, 2)
    csu = qtn.CircuitPEPSSimpleUpdate(edges, max_bond=8)
    csu.apply_gates(rand_circuit_gates(csu.sites, csu.edges, seed=3))

    other = csu.copy()
    n_before = csu.num_gates
    other.apply_gate(qu.rand_uni(4), (0, 0), (0, 1))

    # mutating the copy must not affect the original
    assert csu.num_gates == n_before
    assert other.num_gates == n_before + 1
    assert csu.gauges is not other.gauges


def test_psi0_supplied_directly():
    edges = qtn.edges_2d_square(2, 2)
    csu = qtn.CircuitPEPSSimpleUpdate(edges, max_bond=8)
    csu.apply_gates(rand_circuit_gates(csu.sites, csu.edges, seed=5))
    psi0 = csu.psi

    # rebuild a circuit from an existing PEPS and check the geometry matches
    csu2 = qtn.CircuitPEPSSimpleUpdate(psi0=psi0, max_bond=8)
    assert set(csu2.sites) == set(csu.sites)
    assert set(csu2.edges) == set(csu.edges)


def test_loopy_plaquette_matches_exact():
    # on a 2x2 plaquette (a loop) simple update is approximate, but with a
    # full-system cluster (max_distance covering all sites) and no bond
    # truncation it must reproduce the exact result
    edges = qtn.edges_2d_square(2, 2)
    sites = sorted({s for e in edges for s in e})
    gates = rand_circuit_gates(sites, edges, n_layers=1, seed=7)

    # exact reference via a dense circuit on integer qubits
    qmap = {s: i for i, s in enumerate(sites)}
    ref = qtn.Circuit(N=len(sites))
    for G, *where in gates:
        ref.apply_gate(G, *(qmap[s] for s in where))

    csu = qtn.CircuitPEPSSimpleUpdate(edges, max_bond=16)
    csu.apply_gates(gates)
    csu.equilibrate(max_iterations=300, tol=1e-12)

    for s in sites:
        exact = complex(ref.local_expectation(qu.pauli("Z"), qmap[s]))
        val = csu.local_expectation(qu.pauli("Z"), s, max_distance=2)
        assert_allclose(complex(val), exact, atol=1e-10)


def test_loopy_cluster_converges_to_exact():
    # the cluster expectation error should (roughly) decrease as max_distance
    # grows, on a loopy lattice with an essentially exact (large max_bond) state
    edges = qtn.edges_2d_square(3, 3)
    sites = sorted({s for e in edges for s in e})
    gates = rand_circuit_gates(sites, edges, n_layers=2, seed=11)

    qmap = {s: i for i, s in enumerate(sites)}
    ref = qtn.Circuit(N=len(sites))
    for G, *where in gates:
        ref.apply_gate(G, *(qmap[s] for s in where))
    exact = {
        s: complex(ref.local_expectation(qu.pauli("Z"), qmap[s]))
        for s in sites
    }

    csu = qtn.CircuitPEPSSimpleUpdate(edges, max_bond=16)
    csu.apply_gates(gates)
    csu.equilibrate(max_iterations=200, tol=1e-10)

    def err(md):
        return max(
            abs(
                csu.local_expectation(qu.pauli("Z"), s, max_distance=md)
                - exact[s]
            )
            for s in sites
        )

    # the largest cluster must be much more accurate than the smallest
    assert err(3) < err(0) / 5


def test_psi_gauged_keeps_gauges_separate():
    edges = qtn.edges_2d_square(3, 3)
    csu = qtn.CircuitPEPSSimpleUpdate(edges, max_bond=8)
    csu.apply_gates(rand_circuit_gates(csu.sites, csu.edges, seed=2))

    # psi has gauges absorbed, psi_gauged does not - they are different TNs but
    # represent the same physical state up to the (separate) gauges
    assert csu.psi_gauged.num_tensors == len(csu.sites)
    assert len(csu.gauges) == len(csu.edges)


def test_unsupported_exact_methods_raise():
    edges = qtn.edges_2d_square(3, 3)
    csu = qtn.CircuitPEPSSimpleUpdate(edges, max_bond=8)
    csu.apply_gates(rand_circuit_gates(csu.sites, csu.edges, seed=4))

    for method in ("to_dense", "amplitude", "sample"):
        with pytest.raises(NotImplementedError):
            getattr(csu, method)()
    with pytest.raises(ValueError):
        csu.uni
