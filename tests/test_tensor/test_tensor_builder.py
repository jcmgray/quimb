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

