import numpy as np
import pytest
from numpy.testing import assert_allclose

import quimb as qu
import quimb.tensor as qtn


def build_gates(sites, edges, n_layers, seed):
    """Return circuit gates both as grid-site gates (array, *site) and as
    integer-qubit gates for an exact reference, sharing the same random arrays.
    """
    rng = np.random.default_rng(seed)
    qmap = {s: i for i, s in enumerate(sites)}
    grid, ints = [], []
    for _ in range(n_layers):
        for s in sites:
            U = qu.rand_uni(2, seed=int(rng.integers(1 << 30)))
            grid.append((U, s))
            ints.append((U, qmap[s]))
        for a, b in edges:
            U = qu.rand_uni(4, seed=int(rng.integers(1 << 30)))
            grid.append((U, a, b))
            ints.append((U, qmap[a], qmap[b]))
    return grid, ints, qmap


def exact_expectation(N, int_gates, P, qubit):
    circ = qtn.Circuit(N=N)
    for G, *where in int_gates:
        circ.apply_gate(G, *where)
    return complex(circ.local_expectation(qu.pauli(P), qubit))


def test_chain_matches_exact():
    # a chain is a tree, so simple update is exact for any sufficient max_bond
    sites = list(range(6))
    edges = [(i, i + 1) for i in range(5)]
    grid, ints, qmap = build_gates(sites, edges, n_layers=3, seed=1)

    circ = qtn.CircuitPEPOSimpleUpdate(edges, max_bond=64)
    circ.apply_gates(grid)

    for P in ("X", "Y", "Z"):
        val = complex(circ.local_expectation(qu.pauli(P), 2))
        ref = exact_expectation(6, ints, P, qmap[2])
        assert_allclose(val, ref, atol=1e-10)


def test_loop_exact_at_large_bond_and_truncates():
    # a 2x2 plaquette is loopy; with enough bond to avoid truncation it is
    # exact, and below that there is a controlled truncation error
    edges = qtn.edges_2d_square(2, 2)
    sites = sorted({s for e in edges for s in e})
    grid, ints, qmap = build_gates(sites, edges, n_layers=2, seed=7)

    big = qtn.CircuitPEPOSimpleUpdate(edges, max_bond=64)
    big.apply_gates(grid)
    small = qtn.CircuitPEPOSimpleUpdate(edges, max_bond=4)
    small.apply_gates(grid)

    big_err = small_err = 0.0
    for P in ("X", "Y", "Z"):
        ref = exact_expectation(4, ints, P, qmap[(0, 0)])
        big_err = max(
            big_err,
            abs(complex(big.local_expectation(qu.pauli(P), (0, 0))) - ref),
        )
        small_err = max(
            small_err,
            abs(complex(small.local_expectation(qu.pauli(P), (0, 0))) - ref),
        )

    assert big_err < 1e-10
    assert small_err > 1e-6


def test_apply_gates_is_lazy():
    edges = [(0, 1), (1, 2)]
    circ = qtn.CircuitPEPOSimpleUpdate(edges, max_bond=8)
    grid, _, _ = build_gates([0, 1, 2], edges, n_layers=2, seed=3)
    circ.apply_gates(grid)
    # gates are only recorded, nothing computed
    assert circ.num_gates == len(grid)


def test_lightcone_pruning_ignores_disconnected_gates():
    # observable on site 0; all gates act on the far end (3,4,5), so they are
    # outside the reverse lightcone and must not change <0|Z_0|0> = 1
    sites = list(range(6))
    edges = [(i, i + 1) for i in range(5)]
    far_edges = [(3, 4), (4, 5)]
    grid, _, _ = build_gates([3, 4, 5], far_edges, n_layers=3, seed=9)

    circ = qtn.CircuitPEPOSimpleUpdate(edges, max_bond=8)
    circ.apply_gates(grid)

    val = complex(circ.local_expectation(qu.pauli("Z"), 0))
    assert_allclose(val, 1.0, atol=1e-12)


def test_two_site_observable_matches_exact():
    sites = list(range(5))
    edges = [(i, i + 1) for i in range(4)]
    grid, ints, qmap = build_gates(sites, edges, n_layers=2, seed=11)

    circ = qtn.CircuitPEPOSimpleUpdate(edges, max_bond=64)
    circ.apply_gates(grid)

    ZZ = qu.pauli("Z") & qu.pauli("Z")
    val = complex(circ.local_expectation(ZZ, (1, 2)))

    ref_c = qtn.Circuit(N=5)
    for G, *where in ints:
        ref_c.apply_gate(G, *where)
    ref = complex(ref_c.local_expectation(ZZ, (1, 2)))
    assert_allclose(val, ref, atol=1e-10)


def test_empty_circuit():
    edges = [(0, 1), (1, 2)]
    circ = qtn.CircuitPEPOSimpleUpdate(edges, max_bond=8)
    assert_allclose(
        complex(circ.local_expectation(qu.pauli("Z"), 1)), 1.0, atol=1e-12
    )
    assert_allclose(
        complex(circ.local_expectation(qu.pauli("X"), 1)), 0.0, atol=1e-12
    )


def test_geometry_inferred_from_gates():
    gates = [
        (qu.rand_uni(2), 0),
        (qu.rand_uni(4), 0, 1),
        (qu.rand_uni(4), 1, 2),
    ]
    circ = qtn.CircuitPEPOSimpleUpdate(gates=gates, max_bond=8)
    assert set(circ.sites) == {0, 1, 2}
    assert set(circ.edges) == {(0, 1), (1, 2)}
    circ.apply_gates(gates)
    assert circ.num_gates == 3


def test_validation_and_unsupported():
    edges = qtn.edges_2d_square(2, 2)
    circ = qtn.CircuitPEPOSimpleUpdate(edges, max_bond=8)

    with pytest.raises(ValueError):  # non-edge two-site gate
        circ.apply_gate(qu.rand_uni(4), (0, 0), (1, 1))
    with pytest.raises(ValueError):  # three-site gate
        circ.apply_gate(qu.rand_uni(8), (0, 0), (0, 1), (1, 0))

    for method in ("to_dense", "amplitude", "sample"):
        with pytest.raises(NotImplementedError):
            getattr(circ, method)()
    with pytest.raises(NotImplementedError):
        circ.psi
    with pytest.raises(ValueError):
        circ.uni


def test_copy_is_independent():
    edges = [(0, 1), (1, 2)]
    circ = qtn.CircuitPEPOSimpleUpdate(edges, max_bond=8)
    grid, _, _ = build_gates([0, 1, 2], edges, n_layers=1, seed=2)
    circ.apply_gates(grid)

    other = circ.copy()
    n_before = circ.num_gates
    other.apply_gate(qu.rand_uni(4), 0, 1)

    assert circ.num_gates == n_before
    assert other.num_gates == n_before + 1


def test_2d_grid_shallow_matches_exact():
    # a shallow (bounded lightcone) circuit on a loopy 2D grid should match
    # exact once the bond is large enough to avoid truncation
    from quimb.tensor.tnag.tebd import edge_coloring

    edges = qtn.edges_2d_square(3, 3)
    sites = sorted({s for e in edges for s in e})
    qmap = {s: i for i, s in enumerate(sites)}

    rng = np.random.default_rng(4)
    grid, ints = [], []
    for s in sites:
        U = qu.rand_uni(2, seed=int(rng.integers(1 << 30)))
        grid.append((U, s))
        ints.append((U, qmap[s]))
    for layer in edge_coloring(edges)[:2]:
        for a, b in layer:
            U = qu.rand_uni(4, seed=int(rng.integers(1 << 30)))
            grid.append((U, a, b))
            ints.append((U, qmap[a], qmap[b]))

    circ = qtn.CircuitPEPOSimpleUpdate(edges, max_bond=32)
    circ.apply_gates(grid)

    ref = qtn.Circuit(N=9)
    for G, *where in ints:
        ref.apply_gate(G, *where)

    for P in ("X", "Y", "Z"):
        val = complex(circ.local_expectation(qu.pauli(P), (1, 1)))
        exact = complex(ref.local_expectation(qu.pauli(P), qmap[(1, 1)]))
        assert_allclose(val, exact, atol=1e-10)


def test_large_lattice_local_observable_is_finite():
    # thanks to reverse-lightcone pruning and support-restricted contraction, a
    # local observable of a shallow circuit is tractable far beyond exact sizes
    from quimb.tensor.tnag.tebd import edge_coloring

    L = 16
    edges = qtn.edges_2d_square(L, L)
    sites = sorted({s for e in edges for s in e})

    rng = np.random.default_rng(2)
    grid = [
        (qu.rand_uni(2, seed=int(rng.integers(1 << 30))), s) for s in sites
    ]
    for layer in edge_coloring(edges)[:2]:
        for a, b in layer:
            grid.append(
                (qu.rand_uni(4, seed=int(rng.integers(1 << 30))), a, b)
            )

    circ = qtn.CircuitPEPOSimpleUpdate(edges, max_bond=8)
    circ.apply_gates(grid)
    val = complex(circ.local_expectation(qu.pauli("Z"), (L // 2, L // 2)))
    assert np.isfinite(val.real)
    assert abs(val.imag) < 1e-8
    assert abs(val.real) <= 1.0 + 1e-8


def test_parametrized_2q_gates_match_exact():
    # parametrized two-qubit gates (e.g. SU4, FSIM) are built as (2, 2, 2, 2)
    # arrays rather than (4, 4) matrices, so check the reshape-and-adjoint in
    # the backwards evolution handles them correctly (chain -> exact)
    edges = [(i, i + 1) for i in range(4)]
    rng = np.random.default_rng(13)

    gates = []
    for s in range(5):
        gates.append((qu.rand_uni(2, seed=int(rng.integers(1 << 30))), s))
    for a, b in edges:
        gates.append(("SU4", *rng.uniform(0, 2 * np.pi, size=15), a, b))
        gates.append(("FSIM", *rng.uniform(0, 2 * np.pi, size=2), a, b))

    circ = qtn.CircuitPEPOSimpleUpdate(edges, max_bond=64)
    circ.apply_gates(gates)

    ref = qtn.Circuit(N=5)
    ref.apply_gates(gates)

    for P in ("X", "Y", "Z"):
        assert_allclose(
            complex(circ.local_expectation(qu.pauli(P), 2)),
            complex(ref.local_expectation(qu.pauli(P), 2)),
            atol=1e-10,
        )


def test_gate_opts_is_public_and_get_evolved_operator_returns_operator():
    # gate_opts should be a public dict like the other Circuit classes, and
    # get_evolved_operator should return a single operator carrying its scale
    edges = [(0, 1), (1, 2)]
    circ = qtn.CircuitPEPOSimpleUpdate(edges, max_bond=8, cutoff=1e-12)
    assert circ.gate_opts["max_bond"] == 8
    assert circ.gate_opts["cutoff"] == 1e-12

    circ.apply_gates(build_gates([0, 1, 2], edges, n_layers=1, seed=1)[0])
    op = circ.get_evolved_operator(qu.pauli("Z"), 1)
    assert isinstance(op, qtn.TensorNetworkGenOperator)
