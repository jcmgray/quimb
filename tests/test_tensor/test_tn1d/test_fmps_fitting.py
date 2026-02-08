import importlib

import numpy as np
import pytest

requires_symmray = pytest.mark.skipif(
    importlib.util.find_spec("symmray") is None,
    reason="symmray not installed",
)


@requires_symmray
@pytest.fixture
def get_fpeps_and_norm():
    import symmray as sr

    # fPEPS parameters
    Lx = int(4)
    Ly = int(4)
    symmetry = "Z2"
    D = 4
    # Load PEPS
    fpeps = sr.PEPS_fermionic_rand(
        symmetry, Lx, Ly, bond_dim=D, phys_dim=4, seed=42
    )
    fpeps.equalize_norms_()
    for ts in fpeps.tensors:
        ts.data.phase_sync(inplace=True)
    fpeps_norm = fpeps.make_norm()
    benchmark_norm = fpeps_norm.contract_boundary_from_xmax(
        xrange=(0, Lx - 1), max_bond=256, cutoff=0.0, mode="direct"
    ).contract()
    # benchmark_norm = np.float64(9.347604511732736e18)
    return fpeps_norm, benchmark_norm


@requires_symmray
@pytest.mark.parametrize("from_which", ["xmin", "xmax", "ymin", "ymax"])
def test_fmps_mpo_fitting(from_which, get_fpeps_and_norm):
    fpeps_norm, benchmark_norm = get_fpeps_and_norm
    print(f"Benchmark norm: {benchmark_norm}")
    print("Boundary fMPS-MPO fitting contraction test:")
    # contraction bond dimension
    chi = 128
    if from_which == "xmin":
        print("xmin:")
        c_xmin_0 = fpeps_norm.contract_boundary_from_xmin(
            xrange=(0, 1),
            max_bond=chi,
            cutoff=0.0,
            mode="fit",
            tol=1e-5,
            tn_fit="zipup",
            bsz=2,
            max_iterations=6,
        ).contract()
        c_xmin_1 = fpeps_norm.contract_boundary_from_xmin(
            xrange=(0, 2),
            max_bond=chi,
            cutoff=0.0,
            mode="fit",
            tol=1e-5,
            tn_fit="zipup",
            bsz=2,
            max_iterations=6,
        ).contract()
        c_xmin_2 = fpeps_norm.contract_boundary_from_xmin(
            xrange=(0, 1),
            max_bond=chi,
            cutoff=0.0,
            mode="fit",
            tol=1e-5,
            tn_fit="zipup",
            bsz=2,
            max_iterations=5,
        ).contract()
        c_xmin_3 = fpeps_norm.contract_boundary_from_xmin(
            xrange=(0, 2),
            max_bond=chi,
            cutoff=0.0,
            mode="fit",
            tol=1e-5,
            tn_fit="zipup",
            bsz=2,
            max_iterations=5,
        ).contract()
        print(c_xmin_0, np.allclose(c_xmin_0, benchmark_norm, rtol=1e-4))
        print(c_xmin_1, np.allclose(c_xmin_1, benchmark_norm, rtol=1e-4))
        print(c_xmin_2, np.allclose(c_xmin_2, benchmark_norm, rtol=1e-4))
        print(c_xmin_3, np.allclose(c_xmin_3, benchmark_norm, rtol=1e-4))
        assert np.allclose(c_xmin_0, benchmark_norm, rtol=1e-4)
        assert np.allclose(c_xmin_1, benchmark_norm, rtol=1e-4)
        assert np.allclose(c_xmin_2, benchmark_norm, rtol=1e-4)
        assert np.allclose(c_xmin_3, benchmark_norm, rtol=1e-4)
    elif from_which == "xmax":
        print("xmax:")
        c_xmax_0 = fpeps_norm.contract_boundary_from_xmax(
            xrange=(2, 3),
            max_bond=chi,
            cutoff=0.0,
            mode="fit",
            tol=1e-5,
            tn_fit="zipup",
            bsz=2,
            max_iterations=6,
        ).contract()
        c_xmax_1 = fpeps_norm.contract_boundary_from_xmax(
            xrange=(1, 3),
            max_bond=chi,
            cutoff=0.0,
            mode="fit",
            tol=1e-5,
            tn_fit="zipup",
            bsz=2,
            max_iterations=6,
        ).contract()
        c_xmax_2 = fpeps_norm.contract_boundary_from_xmax(
            xrange=(2, 3),
            max_bond=chi,
            cutoff=0.0,
            mode="fit",
            tol=1e-5,
            tn_fit="zipup",
            bsz=2,
            max_iterations=5,
        ).contract()
        c_xmax_3 = fpeps_norm.contract_boundary_from_xmax(
            xrange=(1, 3),
            max_bond=chi,
            cutoff=0.0,
            mode="fit",
            tol=1e-5,
            tn_fit="zipup",
            bsz=2,
            max_iterations=5,
        ).contract()
        print(c_xmax_0, np.allclose(c_xmax_0, benchmark_norm, rtol=1e-4))
        print(c_xmax_1, np.allclose(c_xmax_1, benchmark_norm, rtol=1e-4))
        print(c_xmax_2, np.allclose(c_xmax_2, benchmark_norm, rtol=1e-4))
        print(c_xmax_3, np.allclose(c_xmax_3, benchmark_norm, rtol=1e-4))
        assert np.allclose(c_xmax_0, benchmark_norm, rtol=1e-4)
        assert np.allclose(c_xmax_1, benchmark_norm, rtol=1e-4)
        assert np.allclose(c_xmax_2, benchmark_norm, rtol=1e-4)
        assert np.allclose(c_xmax_3, benchmark_norm, rtol=1e-4)
    elif from_which == "ymin":
        print("ymin:")
        c_ymin_0 = fpeps_norm.contract_boundary_from_ymin(
            yrange=(0, 1),
            max_bond=chi,
            cutoff=0.0,
            mode="fit",
            tol=1e-5,
            tn_fit="zipup",
            bsz=2,
            max_iterations=6,
        ).contract()
        c_ymin_1 = fpeps_norm.contract_boundary_from_ymin(
            yrange=(0, 2),
            max_bond=chi,
            cutoff=0.0,
            mode="fit",
            tol=1e-5,
            tn_fit="zipup",
            bsz=2,
            max_iterations=6,
        ).contract()
        c_ymin_2 = fpeps_norm.contract_boundary_from_ymin(
            yrange=(0, 1),
            max_bond=chi,
            cutoff=0.0,
            mode="fit",
            tol=1e-5,
            tn_fit="zipup",
            bsz=2,
            max_iterations=5,
        ).contract()
        c_ymin_3 = fpeps_norm.contract_boundary_from_ymin(
            yrange=(0, 2),
            max_bond=chi,
            cutoff=0.0,
            mode="fit",
            tol=1e-5,
            tn_fit="zipup",
            bsz=2,
            max_iterations=5,
        ).contract()
        print(c_ymin_0, np.allclose(c_ymin_0, benchmark_norm, rtol=1e-4))
        print(c_ymin_1, np.allclose(c_ymin_1, benchmark_norm, rtol=1e-4))
        print(c_ymin_2, np.allclose(c_ymin_2, benchmark_norm, rtol=1e-4))
        print(c_ymin_3, np.allclose(c_ymin_3, benchmark_norm, rtol=1e-4))
        assert np.allclose(c_ymin_0, benchmark_norm, rtol=1e-4)
        assert np.allclose(c_ymin_1, benchmark_norm, rtol=1e-4)
        assert np.allclose(c_ymin_2, benchmark_norm, rtol=1e-4)
        assert np.allclose(c_ymin_3, benchmark_norm, rtol=1e-4)
    elif from_which == "ymax":
        print("ymax:")
        c_ymax_0 = fpeps_norm.contract_boundary_from_ymax(
            yrange=(2, 3),
            max_bond=chi,
            cutoff=0.0,
            mode="fit",
            tol=1e-5,
            tn_fit="zipup",
            bsz=2,
            max_iterations=6,
        ).contract()
        c_ymax_1 = fpeps_norm.contract_boundary_from_ymax(
            yrange=(1, 3),
            max_bond=chi,
            cutoff=0.0,
            mode="fit",
            tol=1e-5,
            tn_fit="zipup",
            bsz=2,
            max_iterations=6,
        ).contract()
        c_ymax_2 = fpeps_norm.contract_boundary_from_ymax(
            yrange=(2, 3),
            max_bond=chi,
            cutoff=0.0,
            mode="fit",
            tol=1e-5,
            tn_fit="zipup",
            bsz=2,
            max_iterations=5,
        ).contract()
        c_ymax_3 = fpeps_norm.contract_boundary_from_ymax(
            yrange=(1, 3),
            max_bond=chi,
            cutoff=0.0,
            mode="fit",
            tol=1e-5,
            tn_fit="zipup",
            bsz=2,
            max_iterations=5,
        ).contract()
        print(c_ymax_0, np.allclose(c_ymax_0, benchmark_norm, rtol=1e-4))
        print(c_ymax_1, np.allclose(c_ymax_1, benchmark_norm, rtol=1e-4))
        print(c_ymax_2, np.allclose(c_ymax_2, benchmark_norm, rtol=1e-4))
        print(c_ymax_3, np.allclose(c_ymax_3, benchmark_norm, rtol=1e-4))
        assert np.allclose(c_ymax_0, benchmark_norm, rtol=1e-4)
        assert np.allclose(c_ymax_1, benchmark_norm, rtol=1e-4)
        assert np.allclose(c_ymax_2, benchmark_norm, rtol=1e-4)
        assert np.allclose(c_ymax_3, benchmark_norm, rtol=1e-4)
