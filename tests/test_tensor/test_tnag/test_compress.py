import importlib.util

import pytest

import quimb.tensor as qtn
from quimb.tensor.tnag.compress import tensor_network_ag_compress

requires_symmray = pytest.mark.skipif(
    importlib.util.find_spec("symmray") is None,
    reason="symmray not installed",
)


def test_compress_projector_bp_canonize():
    psi = qtn.PEPS.rand(2, 2, bond_dim=2, phys_dim=2, seed=42)
    info = {}

    compressed = tensor_network_ag_compress(
        psi,
        max_bond=1,
        method="projector",
        canonize="bp",
        power=0.75,
        gauge_power=0.75,
        canonize_opts={
            "max_iterations": 100,
            "tol": 1e-8,
            "info": info,
        },
    )

    assert info["converged"]
    assert compressed is not psi
    assert compressed.num_tensors == len(psi.site_tags)
    assert compressed.max_bond() == 1
    assert psi.max_bond() == 2


@requires_symmray
@pytest.mark.parametrize(
    "symmetry",
    [
        "abelian",
        "fermionic-even",
        "fermionic-odd-checker",
        "fermionic-odd-diag",
    ],
)
@pytest.mark.parametrize("canonize", [False, True, "layered", "bp"])
@pytest.mark.parametrize("from_which", ["xmin", "xmax", "ymin", "ymax"])
def test_compress_projector_symmetric(symmetry, canonize, from_which, request):
    """Boundary contracting a scalar symmetric network with locally computed
    projectors should be exact, given enough bond dimension, however the
    network is preconditioned.
    """
    import symmray as sr

    fermionic = symmetry != "abelian"

    if canonize == "bp" and fermionic and from_which in ("xmax", "ymax"):
        # inserting the messages still picks up an overall sign in some
        # regions, leaving the squared operator minus a positive operator,
        # which `compute_reduced_factor` assumes it is not
        request.applymarker(
            pytest.mark.xfail(
                reason="bp message insertion signs some regions", strict=True
            )
        )

    if symmetry == "fermionic-odd-checker":
        # checkerboard of odd sites, keeping the total charge even
        def site_charge(site):
            return (site[0] + site[1]) % 2

    elif symmetry == "fermionic-odd-diag":
        # odd sites on the diagonal, keeping the total charge even
        def site_charge(site):
            return int(site[0] == site[1])

    else:

        def site_charge(site):
            return 0

    Lx = Ly = 4

    tn = sr.TN_abelian_from_edges_rand(
        symmetry="Z2",
        edges=qtn.edges_2d_square(Lx, Ly),
        bond_dim=2,
        phys_dim=None,
        fermionic=fermionic,
        site_tag_id="I{},{}",
        site_charge=site_charge,
        seed=42,
    )
    for i in range(Lx):
        for j in range(Ly):
            tn[f"I{i},{j}"].add_tag(f"X{i}")
            tn[f"I{i},{j}"].add_tag(f"Y{j}")
    tn.view_as_(
        qtn.TensorNetwork2D, Lx=Lx, Ly=Ly, x_tag_id="X{}", y_tag_id="Y{}"
    )

    expected = tn.contract(all, optimize="auto-hq")

    # `contract_boundary` routes this to `tensor_network_ag_compress`
    value = tn.contract_boundary(
        # enough that nothing is discarded
        max_bond=8,
        cutoff=0.0,
        mode="projector",
        canonize=canonize,
        sequence=(from_which,),
    )

    assert value == pytest.approx(expected, rel=1e-10)
