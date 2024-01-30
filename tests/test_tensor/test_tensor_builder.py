import pytest
from numpy.testing import assert_allclose

import quimb as qu
import quimb.tensor as qtn


def test_tn2d_classical_ising_partition_function():
    Lx = 4
    Ly = 5
    coupling = {
        (cooa, coob): float(qu.randn())
        for cooa, coob in qtn.gen_2d_bonds(Lx, Ly)
    }
    h = float(qu.randn())
    tn = qtn.TN2D_classical_ising_partition_function(
        Lx,
        Ly,
        beta=0.44,
        j=lambda cooa, coob: coupling[(cooa, coob)],
        h=h,
        outputs=[(1, 2), (3, 4)],
    )
    assert tn.outer_inds() == ("s1,2", "s3,4")
    htn = qtn.HTN2D_classical_ising_partition_function(
        Lx,
        Ly,
        beta=0.44,
        j=lambda cooa, coob: coupling[(cooa, coob)],
        h=h,
    )
    assert_allclose(
        tn.contract().data,
        htn.contract(output_inds=("s1,2", "s3,4")).data,
    )


def test_tn3d_classical_ising_partition_function():
    Lx, Ly, Lz = 2, 3, 3
    coupling = {
        (cooa, coob): float(qu.randn())
        for cooa, coob in qtn.gen_3d_bonds(Lx, Ly, Lz)
    }
    h = float(qu.randn())
    tn = qtn.TN3D_classical_ising_partition_function(
        Lx,
        Ly,
        Lz,
        beta=0.44,
        j=lambda cooa, coob: coupling[(cooa, coob)],
        h=h,
        outputs=[(1, 0, 2), (0, 2, 1)],
    )
    assert tn.outer_inds() == ("s0,2,1", "s1,0,2")
    htn = qtn.HTN3D_classical_ising_partition_function(
        Lx,
        Ly,
        Lz,
        beta=0.44,
        j=lambda cooa, coob: coupling[(cooa, coob)],
        h=h,
    )
    assert_allclose(
        tn.contract().data,
        htn.contract(output_inds=("s0,2,1", "s1,0,2")).data,
    )


@pytest.mark.parametrize("sites_location", ["side", "diag"])
@pytest.mark.parametrize("outputs", [(), 2, (1, 3)])
def test_all_to_all_classical_partition_functions(sites_location, outputs):
    import numpy as np

    N = 5
    rng = np.random.default_rng(42)
    Jij = {(i, j): rng.normal() for i in range(N) for j in range(i + 1, N)}
    htn = qtn.HTN_classical_partition_function_from_edges(
        edges=Jij.keys(),
        beta=0.179,
        j=Jij,
    )
    Zex = htn.contract(all, output_inds=())

    tn = qtn.TN2D_embedded_classical_ising_partition_function(
        Jij,
        beta=0.179,
        sites_location=sites_location,
        outputs=outputs,
    )

    sites = tuple(tn.gen_sites_present())
    assert len(sites) == N * (N - 1) // 2
    for (i, j) in sites:
        assert i > j

    if isinstance(outputs, tuple):
        assert set(tn.outer_inds()) == {f"s{i}" for i in outputs}
    else:
        assert tn.outer_inds() == (f"s{outputs}",)
        t, = tn._inds_get(f"s{outputs}")
        if sites_location == "side":
            assert "I2,0" in t.tags
        else:
            assert "I2,1" in t.tags
    assert tn.contract(output_inds=()) == pytest.approx(Zex)
