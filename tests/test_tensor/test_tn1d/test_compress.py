import autoray as ar
import numpy as np
import pytest

import quimb as qu
import quimb.tensor as qtn

from .. import (
    bond_orientations,
    jax_case,
    make_symmetric_2d_tn,
    pytorch_case,
    requires_symmray,
    symmetry_cases,
)

dtypes = ["float32", "float64", "complex64", "complex128"]


@pytest.mark.parametrize("method", ["src", "srcmps"])
@pytest.mark.filterwarnings("error:`cutoff` is ignored")
def test_random_sampling_default_cutoff(method):
    from quimb.tensor.tn1d.compress import tensor_network_1d_compress

    psi = qtn.MPS_rand_state(5, 6, seed=42)
    result = tensor_network_1d_compress(
        psi, max_bond=2, method=method, seed=42
    )
    assert result.max_bond() <= 2
    with pytest.warns(UserWarning, match="`cutoff` is ignored"):
        tensor_network_1d_compress(
            psi, max_bond=2, method=method, cutoff=1e-10, seed=42
        )


@pytest.mark.parametrize("method", ["zipup", "sdc", "sdcr"])
@pytest.mark.parametrize("mode", [None, "abs"])
def test_oversample_independent_cutoff_modes(monkeypatch, method, mode):
    from quimb.tensor.tn1d import compress

    name = f"tensor_network_1d_compress_{method}"
    original = getattr(compress, name)
    original_sweep = compress._do_direct_sweep
    modes = []

    def record_intermediate(*args, **kwargs):
        modes.append(kwargs["cutoff_mode"])
        return original(*args, **kwargs)

    def record_sweep(*args, **kwargs):
        modes.append(kwargs["cutoff_mode"])
        return original_sweep(*args, **kwargs)

    monkeypatch.setattr(compress, name, record_intermediate)
    monkeypatch.setattr(compress, "_do_direct_sweep", record_sweep)
    psi = qtn.MPS_rand_state(5, 6, seed=42)
    opts = {} if mode is None else {"cutoff_mode_oversample": mode}
    result = compress.tensor_network_1d_compress(
        psi,
        method=f"{method}-oversample",
        max_bond=2,
        max_bond_oversample=4,
        cutoff_oversample=1e-8,
        **opts,
    )
    assert modes == [mode or "rel", "rsum2"]
    assert result.max_bond() <= 2


@pytest.mark.parametrize("max_bond", [None, 2])
def test_zipup_oversample_default_cutoff(monkeypatch, max_bond):
    from quimb.tensor.tn1d import compress

    original = compress.tensor_network_1d_compress_zipup
    cutoffs = []

    def record(*args, **kwargs):
        cutoffs.append(kwargs["cutoff"])
        return original(*args, **kwargs)

    monkeypatch.setattr(compress, "tensor_network_1d_compress_zipup", record)
    psi = qtn.MPS_rand_state(5, 6, seed=42)
    compress.tensor_network_1d_compress_zipup_oversample(
        psi, max_bond=max_bond, max_bond_oversample=4, cutoff=1e-3
    )
    compress.mps_gate_with_mpo_zipup_first(
        psi,
        qtn.MPO_identity(5),
        max_bond=max_bond,
        max_bond_oversample=4,
        cutoff=1e-3,
    )
    assert cutoffs == ["auto", "auto"]


@pytest.mark.parametrize(
    "method,expected", [("svd", 1e-10), ("svd:rand", 0.0)]
)
def test_zipup_oversample_method_cutoff(monkeypatch, method, expected):
    import functools

    from quimb.tensor import decomp
    from quimb.tensor.tn1d import compress

    original = decomp._SPLIT_FNS[method]
    cutoffs = []

    @functools.wraps(original)
    def record(*args, **kwargs):
        cutoffs.append(kwargs["cutoff"])
        return original(*args, **kwargs)

    monkeypatch.setitem(decomp._SPLIT_FNS, method, record)
    psi = qtn.MPS_rand_state(5, 6, seed=42)
    compress.tensor_network_1d_compress_zipup_oversample(
        psi, max_bond=2, cutoff=1e-3, method=method
    )
    assert cutoffs
    assert cutoffs[:4] == [expected] * 4
    assert set(cutoffs[4:]) == ({1e-3} if method == "svd" else set())


@pytest.mark.parametrize("method", ["zipup", "sdc", "sdcr"])
def test_oversample_randomized_options_stay_intermediate(method):
    from quimb.tensor.tn1d.compress import tensor_network_1d_compress

    psi = qtn.MPS_rand_state(5, 6, seed=42)
    intermediate = {"method": "svd:rand", "oversample": 1}
    final = {"method": "svd", "max_bond": 1}
    result = tensor_network_1d_compress(
        psi,
        method=f"{method}-oversample",
        max_bond=2,
        max_bond_oversample=4,
        seed=42,
        compress_opts=intermediate,
        compress_opts_final=final,
    )
    assert result.max_bond() == 1
    assert intermediate == {"method": "svd:rand", "oversample": 1}
    assert final == {"method": "svd", "max_bond": 1}


@pytest.mark.parametrize("method", ["src", "srcmps", "fit"])
def test_oversample_final_options_precedence(method):
    from quimb.tensor.tn1d.compress import tensor_network_1d_compress

    psi = qtn.MPS_rand_state(5, 6, seed=42)
    result = tensor_network_1d_compress(
        psi,
        method=f"{method}-oversample",
        max_bond=2,
        max_bond_oversample=4,
        seed=42,
        compress_opts={"cutoff_mode": "rsum2", "max_bond": 3},
        compress_opts_final={"method": "svd", "max_bond": 1},
    )
    assert result.max_bond() == 1


# compression options, including fit sweep counts
boundary_options = {
    "direct": {"mode": "direct"},
    "dm": {"mode": "dm"},
    "zipup": {"mode": "zipup"},
    "zipup-oversample": {"mode": "zipup-oversample"},
    "sdc": {"mode": "sdc"},
    "sdc-oversample": {"mode": "sdc-oversample"},
    "sdcr": {"mode": "sdcr"},
    "sdcr-oversample": {"mode": "sdcr-oversample"},
    "fit-bsz1": {
        "mode": "fit",
        "bsz": 1,
        "tn_fit": "zipup",
        "max_iterations": 6,
    },
    "fit-bsz2": {
        "mode": "fit",
        "bsz": 2,
        "tn_fit": "zipup",
        "max_iterations": 6,
    },
    "fit-bsz2-odd-iters": {
        "mode": "fit",
        "bsz": 2,
        "tn_fit": "zipup",
        "max_iterations": 5,
    },
    "fit-bsz1-odd-iters": {
        "mode": "fit",
        "bsz": 1,
        "tn_fit": "zipup",
        "max_iterations": 5,
    },
}


@pytest.fixture(scope="module")
def fpeps_norm_and_benchmark():
    import symmray as sr

    fpeps = sr.PEPS_fermionic_rand("Z2", 4, 4, bond_dim=4, phys_dim=4, seed=42)
    fpeps.equalize_norms_()
    for tensor in fpeps.tensors:
        tensor.data.phase_sync(inplace=True)

    fpeps_norm = fpeps.make_norm()
    benchmark = fpeps_norm.contract(all, optimize="auto-hq")
    return fpeps_norm, benchmark


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
@pytest.mark.parametrize(
    "method", ["sdc", "sdc-oversample", "sdcr", "sdcr-oversample"]
)
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
@pytest.mark.parametrize("method", boundary_options)
@pytest.mark.parametrize("symmetry", symmetry_cases)
@pytest.mark.parametrize("bond_orientation", bond_orientations)
@pytest.mark.parametrize("direction", ["xmin", "xmax", "ymin", "ymax"])
def test_symmetric_boundary_contract(
    method, symmetry, bond_orientation, direction
):
    """Check exact 1D compression of a scalar symmetric 2D network.

    Cover tensor parity, bond orientation, and contraction direction. Use
    enough bond dimension to prevent truncation.
    """
    # use distinct random data for each case
    seed = qu.utils.hash_kwargs_to_int(
        method=method,
        symmetry=symmetry,
        bond_orientation=bond_orientation,
        direction=direction,
    )
    tn = make_symmetric_2d_tn(symmetry, duals=bond_orientation, seed=seed)

    expected = tn.contract(all, optimize="auto-hq")
    value = tn.contract_boundary(
        # prevent truncation
        max_bond=4,
        cutoff=0.0,
        sequence=(direction,),
        **boundary_options[method],
    )
    assert value == pytest.approx(expected, rel=1e-10)


@requires_symmray
@pytest.mark.parametrize(
    "from_which,boundary_range",
    [
        pytest.param("xmin", (0, 2), id="xmin"),
        pytest.param("xmax", (1, 3), id="xmax"),
        pytest.param("ymin", (0, 2), id="ymin"),
        pytest.param("ymax", (1, 3), id="ymax"),
    ],
)
def test_fmps_mpo_fitting(
    from_which,
    boundary_range,
    fpeps_norm_and_benchmark,
):
    fpeps_norm, benchmark = fpeps_norm_and_benchmark
    contract_boundary = getattr(
        fpeps_norm,
        f"contract_boundary_from_{from_which}",
    )
    range_key = "xrange" if from_which.startswith("x") else "yrange"

    result = contract_boundary(
        **{range_key: boundary_range},
        max_bond=128,
        cutoff=0.0,
        mode="fit",
        tol=1e-5,
        tn_fit="zipup",
        bsz=2,
        max_iterations=6,
    ).contract()

    assert result == pytest.approx(benchmark, rel=1e-4)


@requires_symmray
@pytest.mark.parametrize("symmetry", symmetry_cases)
@pytest.mark.parametrize("bond_orientation", bond_orientations)
@pytest.mark.parametrize("direction", ["xmin", "xmax", "ymin", "ymax"])
def test_dm_truncating_matches_direct(symmetry, bond_orientation, direction):
    """Check that truncated DM compression matches direct compression."""
    seed = qu.utils.hash_kwargs_to_int(
        symmetry=symmetry,
        bond_orientation=bond_orientation,
        direction=direction,
    )
    # this network and bond limit force truncation
    tn = make_symmetric_2d_tn(
        symmetry,
        duals=bond_orientation,
        Lx=6,
        Ly=6,
        seed=seed,
    )
    contraction_options = {
        "max_bond": 4,
        "cutoff": 0.0,
        "sequence": (direction,),
    }
    value_dm = tn.contract_boundary(mode="dm", **contraction_options)
    value_direct = tn.contract_boundary(mode="direct", **contraction_options)
    assert value_dm == pytest.approx(value_direct, rel=1e-8)


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
        "zipup-oversample",
        "sdc",
        "sdc-oversample",
        "sdcr",
        "sdcr-oversample",
        "src",
        "src-oversample",
        "srcmps",
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
    if ("src" in method) or ("sdcr" in method):
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
        "zipup-oversample",
        "sdc",
        "sdc-oversample",
        "sdcr",
        "sdcr-oversample",
        "src",
        "src-oversample",
        "srcmps",
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
    eps = 1e-3 if dtype in ("float32", "complex64") else 1e-5
    if ("src" in method) or ("sdcr" in method):
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
        "sdcr",
        "sdcr-oversample",
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


@pytest.mark.parametrize("method", ["direct", "local-early"])
@pytest.mark.parametrize("nested", [False, True])
def test_compress_site_tag_groups(method, nested):
    L = 6
    max_bond = 4
    A = qtn.MPO_rand(L, 3, phys_dim=2, seed=42, tags="A")
    B = qtn.MPS_rand_state(L, 3, phys_dim=2, seed=43, tags="B")
    AB = qtn.tensor_network_apply_op_vec(A, B, contract=False)

    # add a unique tag to each tensor at a site
    for i in range(L):
        AB.select(("A", AB.site_tag(i)), "all").add_tag(f"L{i}")
        AB.select(("B", AB.site_tag(i)), "all").add_tag(f"R{i}")

    if nested:
        # the nested item requires both tags
        site_tags = [(f"L{i}", (f"R{i}", "B")) for i in range(L)]
    else:
        site_tags = [(f"L{i}", f"R{i}") for i in range(L)]

    # retain a virtual view to detect leaked tags
    view = AB.select("A")

    ref = qtn.tensor_network_1d_compress(AB, max_bond=max_bond, method=method)
    ABc = qtn.tensor_network_1d_compress(
        AB, max_bond=max_bond, method=method, site_tags=site_tags
    )

    assert ABc.num_tensors == L
    assert ABc.max_bond() == max_bond
    for src in (ABc, AB, view):
        assert not any(tag.startswith("__GST") for tag in src.tag_map)
    for i in range(L):
        assert set(ABc[f"L{i}"].tags) == {"A", "B", f"I{i}", f"L{i}", f"R{i}"}

    # both forms select the same tensors
    np.testing.assert_allclose(ABc.to_dense(), ref.to_dense(), atol=1e-12)


def test_compress_fit_site_tag_groups_with_initial_guess():
    L = 4
    A = qtn.MPO_rand(L, 3, phys_dim=2, seed=42, tags="A")
    B = qtn.MPS_rand_state(L, 3, phys_dim=2, seed=43, tags="B")
    AB = qtn.tensor_network_apply_op_vec(A, B, contract=False)
    guess = qtn.MPS_rand_state(L, 2, phys_dim=2, seed=44)

    site_tags = []
    for i in range(L):
        left_tag = f"L{i}"
        right_tag = f"R{i}"
        AB.select(("A", AB.site_tag(i)), "all").add_tag(left_tag)
        AB.select(("B", AB.site_tag(i)), "all").add_tag(right_tag)
        guess[i].add_tag((left_tag, right_tag))
        site_tags.append((left_tag, right_tag))

    ABc = qtn.tensor_network_1d_compress(
        AB,
        max_bond=2,
        method="fit",
        site_tags=site_tags,
        tn_fit=guess,
        max_iterations=1,
    )

    assert ABc.num_tensors == L
    for src in (ABc, AB, guess):
        assert not any(tag.startswith("__GST") for tag in src.tag_map)
