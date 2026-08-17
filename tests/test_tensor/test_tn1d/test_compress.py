import importlib.util

import autoray as ar
import numpy as np
import pytest

import quimb as qu
import quimb.tensor as qtn

from .. import jax_case, pytorch_case

dtypes = ["float32", "float64", "complex64", "complex128"]

requires_symmray = pytest.mark.skipif(
    importlib.util.find_spec("symmray") is None,
    reason="symmray not installed",
)


@pytest.mark.parametrize(
    "method",
    ["src", "src-first", "srcmps", "srcmps-first", "fit", "fit-oversample"],
)
@pytest.mark.parametrize("seed_mode", ["global", "integer", "generator"])
def test_random_seed(method, seed_mode):
    psi = qtn.MPS_rand_state(4, 3, seed=7)
    compress_opts = {
        "max_bond": 2,
        "method": method,
    }
    if method == "fit":
        compress_opts["max_iterations"] = 1
    else:
        compress_opts["cutoff"] = 0.0

    def compress():
        if seed_mode == "global":
            np.random.seed(42)
            seed = None
        elif seed_mode == "integer":
            seed = 42
        else:
            seed = np.random.default_rng(42)
        return qtn.tensor_network_1d_compress(
            psi,
            seed=seed,
            **compress_opts,
        ).to_dense()

    np.testing.assert_allclose(compress(), compress())


@pytest.mark.parametrize("method", ["src-first", "srcmps-first"])
def test_random_oversample_noise_dist(method):
    psi = qtn.MPS_rand_state(4, 3, seed=7)
    a = qtn.tensor_network_1d_compress(
        psi,
        max_bond=2,
        method=method,
        noise_dist="rademacher",
        seed=42,
    )
    b = qtn.tensor_network_1d_compress(
        psi,
        max_bond=2,
        method=method,
        noise_dist="rademacher",
        seed=42,
    )
    np.testing.assert_allclose(a.to_dense(), b.to_dense())


@pytest.mark.parametrize("backend", [jax_case, pytorch_case])
@pytest.mark.parametrize("seed_mode", ["integer", "generator"])
@pytest.mark.parametrize("method", ["src", "srcmps"])
def test_random_backend(method, seed_mode, backend):
    psi = qtn.MPS_rand_state(4, 3, dtype="complex64", seed=7)
    psi.apply_to_arrays(lambda x: ar.do("array", x, like=backend))
    expected = ar.infer_backend_device_dtype(psi[0].data)

    if seed_mode == "integer":
        seed = 42
    else:
        seed = psi.get_namespace().random.default_rng(42)

    compressed = qtn.tensor_network_1d_compress(
        psi,
        max_bond=2,
        cutoff=0.0,
        method=method,
        seed=seed,
    )

    for tensor in compressed:
        assert ar.infer_backend_device_dtype(tensor.data) == expected


@pytest.mark.parametrize("backend", [jax_case, pytorch_case])
@pytest.mark.parametrize("method", ["sdc", "sdc-oversample"])
def test_sdc_backend(method, backend):
    psi = qtn.MPS_rand_state(4, 3, dtype="complex64", seed=7)
    psi.apply_to_arrays(lambda x: ar.do("array", x, like=backend))
    expected = ar.infer_backend_device_dtype(psi[0].data)

    compressed = qtn.tensor_network_1d_compress(
        psi,
        max_bond=2,
        cutoff=0.0,
        method=method,
    )

    for tensor in compressed:
        assert ar.infer_backend_device_dtype(tensor.data) == expected


@requires_symmray
@pytest.mark.parametrize("method", ["sdc", "sdc-oversample"])
@pytest.mark.parametrize("from_which", ["xmin", "xmax", "ymin", "ymax"])
def test_sdc_fermionic(method, from_which):
    import symmray as sr

    peps = sr.PEPS_fermionic_rand("Z2", 3, 3, bond_dim=2, phys_dim=2, seed=42)
    expected = peps.make_norm().contract(all, optimize="auto-hq")

    value = peps.make_norm().contract_boundary(
        max_bond=64,  # exact
        cutoff=0.0,
        mode=method,
        sequence=(from_which,),
    )

    assert value == pytest.approx(expected, rel=1e-10)


@pytest.mark.parametrize("method", ["srcmps", "fit"])
def test_tn_fit(method):
    psi = qtn.MPS_rand_state(4, 3, seed=7)
    tn_fit = qtn.TN_matching(psi, max_bond=2, seed=42)
    compress_opts = {
        "max_bond": 2,
        "method": method,
        "tn_fit": tn_fit,
    }
    if method == "fit":
        compress_opts["max_iterations"] = 1
    else:
        compress_opts["cutoff"] = 0.0

    a = qtn.tensor_network_1d_compress(
        psi,
        seed=1,
        **compress_opts,
    )
    b = qtn.tensor_network_1d_compress(
        psi,
        seed=2,
        **compress_opts,
    )
    np.testing.assert_allclose(a.to_dense(), b.to_dense())


@pytest.mark.parametrize(
    "method",
    [
        "direct",
        "dm",
        "zipup",
        "zipup-first",
        "zipup-oversample",
        "sdc",
        "sdc-oversample",
        "src",
        "src-first",
        "src-oversample",
        "srcmps",
        "srcmps-first",
        "srcmps-oversample",
        "fit",
        "fit-zipup",
        "fit-projector",
        "fit-oversample",
    ],
)
@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("use_input_exponent", [False, True])
@pytest.mark.parametrize("equalize_norms", [False, True, 1.0])
@pytest.mark.parametrize("normalize", [False, True])
def test_basic_compress_double_mpo(
    method,
    dtype,
    use_input_exponent,
    equalize_norms,
    normalize,
):
    L = 8
    phys_dim = 2
    Da = 3
    Db = 2
    max_bond = 6

    # turn case into a deterministic int [0, 2**32-1] for seeding
    seed = qu.utils.hash_kwargs_to_int(
        method=method,
        dtype=dtype,
        use_input_exponent=use_input_exponent,
        equalize_norms=equalize_norms,
        normalize=normalize,
    )

    a = qtn.MPO_rand(
        L,
        bond_dim=Da,
        phys_dim=phys_dim,
        dtype=dtype,
        seed=seed,
        tags="A",
    )
    b = qtn.MPO_rand(
        L,
        bond_dim=Db,
        phys_dim=phys_dim,
        dtype=dtype,
        seed=seed + 1,
        tags="B",
    )
    if use_input_exponent:
        a.exponent = 2.0
        b.exponent = -1.0
    ab = b.gate_upper_with_op_lazy(a)
    if use_input_exponent:
        assert ab.exponent == 1.0
    else:
        assert ab.exponent == 0.0

    c = qtn.tensor_network_1d_compress(
        ab,
        max_bond=max_bond,
        method=method,
        equalize_norms=equalize_norms,
        normalize=normalize,
    )
    assert c.istree()
    assert c.max_bond() == max_bond
    for t in c:
        assert "A" in t.tags and "B" in t.tags

    if (equalize_norms is True) or normalize:
        assert c.exponent == 0.0

    eps = 1e-3 if dtype in ("float32", "complex64") else 1e-6
    if "src" in method:
        # account for noise
        eps *= 5

    if normalize:
        assert c.norm() == pytest.approx(1.0, abs=eps)
        # just use infidelity ~ cosine distance for normalized tensors
        assert c.distance(ab, normalized="infidelity", method="dense") < eps
    else:
        assert c.distance_normalized(ab, method="dense") < eps


@pytest.mark.parametrize(
    "method",
    [
        "direct",
        "dm",
        "zipup",
        "zipup-first",
        "zipup-oversample",
        "sdc",
        "sdc-oversample",
        "src",
        "src-first",
        "src-oversample",
        "srcmps",
        "srcmps-first",
        "srcmps-oversample",
        "fit",
        "fit-zipup",
        "fit-projector",
        "fit-oversample",
    ],
)
@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("sweep_reverse", [False, True])
def test_mps_partial_mpo_apply(method, dtype, sweep_reverse):
    # the sub-MPO has tensors only at `where`, so the lazily gated MPS has a
    # long range (site skipping) bond, which every method should handle via
    # `enforce_1d_like`
    mps = qtn.MPS_rand_state(10, 7, dtype=dtype)
    A = qu.rand_uni(2**3, dtype=dtype)
    where = [8, 4, 5]
    mpo = qtn.MatrixProductOperator.from_dense(A, sites=where)
    new = mps.gate_with_op_lazy(mpo)
    assert (
        qtn.tensor_network_1d_compress(
            new,
            max_bond=32,
            method=method,
            sweep_reverse=sweep_reverse,
            inplace=True,
        )
        is new
    )
    assert new.num_tensors == 10
    eps = 1e-3 if dtype in ("float32", "complex64") else 1e-6
    if "src" in method:
        # account for noise
        eps *= 5
    assert new.distance_normalized(mps.gate(A, where)) == pytest.approx(
        0.0, abs=eps
    )


@pytest.mark.parametrize(
    "method",
    [
        "direct",
        "dm",
        "fit",
        "zipup",
        "zipup-first",
        "sdc",
        "sdc-oversample",
        "src",
        "src-first",
    ],
)
@pytest.mark.parametrize("sweep_reverse", [False, True])
def test_mpo_compress_opts(method, sweep_reverse):
    L = 6
    A = qtn.MPO_rand(L, 2, phys_dim=3, tags="A")
    B = qtn.MPO_rand(L, 3, phys_dim=3, tags="B")
    AB = A.gate_upper_with_op_lazy(B)
    assert AB.num_tensors == 2 * L
    ABc = qtn.tensor_network_1d_compress(
        AB,
        method=method,
        max_bond=5,
        cutoff=1e-6,
        sweep_reverse=sweep_reverse,
        inplace=False,
    )
    assert ABc.num_tensors == L
    assert ABc.num_indices == 2 * L + L - 1
    assert ABc.max_bond() == 5
    if sweep_reverse:
        assert ABc.calc_current_orthog_center() == (L - 1, L - 1)
    else:
        assert ABc.calc_current_orthog_center() == (0, 0)

    for site in range(L):
        assert set(ABc[site].tags) == {"A", "B", f"I{site}"}
