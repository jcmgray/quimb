import numpy as np
import pytest
from numpy.testing import assert_allclose

import quimb.operator as qop


def assert_all_matrices_match(
    sob: qop.SparseOperatorBuilder,
    extras=(),
):
    Ad = sob.build_dense()
    As = sob.build_sparse_matrix()
    assert_allclose(Ad, As.toarray(), atol=1e-10)
    Am = sob.build_mpo().to_dense()
    assert_allclose(Ad, Am, atol=1e-10)
    Ae = sob.build_matrix_ikron()
    assert_allclose(Ad, Ae, atol=1e-10)
    for extra in extras:
        assert_allclose(Ae, extra, atol=1e-10)
    return Ae


@pytest.mark.parametrize("n", [1, 2, 3, 5])
@pytest.mark.parametrize("m", [1, 3, 10])
@pytest.mark.parametrize("k", [1, 2, 3])
@pytest.mark.parametrize("kmin", [0, None])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_rand_operator_matrices(n, m, k, kmin, seed):
    if k > n:
        pytest.skip("k > n, skipping test")
    sob = qop.rand_operator(n, m, k, kmin=kmin, seed=seed, ops="xyz+-n")
    assert sob.nsites == n
    assert sob.nterms <= m  # can be less due to repeated terms
    A0 = assert_all_matrices_match(sob)
    sob.pauli_decompose()
    assert_all_matrices_match(sob, extras=[A0])
    sob.pauli_decompose()
    assert_all_matrices_match(sob, extras=[A0])
    sob.pauli_decompose(use_zx=True)
    assert_all_matrices_match(sob, extras=[A0])


class TestU1U1Ordering:
    """U1U1 sectors work whether or not the species are contiguous in the
    register ordering, which sets the jordan-wigner strings.
    """

    def get_hubbard(self, order):
        sites = [(s, c) for s in "↑↓" for c in range(4)]
        hs = qop.HilbertSpace(
            sites, order=order, species=lambda s: s[0], sector=(2, 2)
        )
        return qop.fermi_hubbard_from_edges(
            [(0, 1), (1, 2), (2, 3)], U=2.0, mu=0.3, hilbert_space=hs
        )

    def test_spectrum_independent_of_ordering(self):
        Ab = self.get_hubbard("blocked").build_dense()
        Ai = self.get_hubbard("interleaved").build_dense()
        assert Ab.shape == Ai.shape == (36, 36)
        assert_allclose(Ab, Ab.conj().T, atol=1e-10)
        assert_allclose(Ai, Ai.conj().T, atol=1e-10)
        assert_allclose(
            np.linalg.eigvalsh(Ab), np.linalg.eigvalsh(Ai), atol=1e-10
        )

    def test_sector_spectrum_within_full_space(self):
        full = qop.fermi_hubbard_from_edges(
            [(0, 1), (1, 2), (2, 3)], U=2.0, mu=0.3
        )
        eref = np.linalg.eigvalsh(full.build_dense())
        esub = np.linalg.eigvalsh(
            self.get_hubbard("interleaved").build_dense()
        )
        for e in esub:
            assert np.min(np.abs(eref - e)) < 1e-8

    @pytest.mark.parametrize("order", ["blocked", "interleaved"])
    def test_matrices_and_coupling_agree(self, order):
        H = self.get_hubbard(order)
        hs = H.hilbert_space
        A = H.build_dense()
        assert_allclose(H.build_sparse_matrix().toarray(), A, atol=1e-10)

        x = np.random.default_rng(42).random(hs.size)
        assert_allclose(H.matvec(x), A @ x, atol=1e-10)

        # coupled configs come back in register order, and rebuild each column
        for rank in range(hs.size):
            col = np.zeros(hs.size, dtype=complex)
            fx = hs.rank_to_flatconfig(rank)
            for fy, hxy in zip(*H.flatconfig_coupling(fx)):
                col[hs.flatconfig_to_rank(fy)] += hxy
            assert_allclose(col, A[:, rank], atol=1e-10)

    @pytest.mark.parametrize("order", ["blocked", "interleaved"])
    @pytest.mark.parametrize(
        "sector", [(2, 2), {"↑": 2, "↓": 2}, ((4, 2), (4, 2))]
    )
    def test_sector_supplied_per_call(self, order, sector):
        # regression: the enumeration permutation must not depend on the
        # default symmetry, since a sector can be supplied per call while the
        # coupling map is built once for all of them
        sites = [(s, c) for s in "↑↓" for c in range(4)]
        hs = qop.HilbertSpace(sites, order=order, species=lambda s: s[0])
        assert hs.symmetry is None
        H = qop.fermi_hubbard_from_edges(
            [(0, 1), (1, 2), (2, 3)], U=2.0, mu=0.3, hilbert_space=hs
        )
        A = H.build_dense(sector=sector)
        assert A.shape == (36, 36)
        assert_allclose(A, A.conj().T, atol=1e-10)
        # the sector spectrum sits inside the unrestricted one
        efull = np.linalg.eigvalsh(H.build_dense())
        for e in np.linalg.eigvalsh(A):
            assert np.min(np.abs(efull - e)) < 1e-8

    @pytest.mark.parametrize("order", ["blocked", "interleaved"])
    def test_vmc_paths_agree(self, order):
        H = self.get_hubbard(order)
        hs = H.hilbert_space
        A = H.build_dense()
        gs = np.linalg.eigh(A)[1][:, 0]

        amps = {tuple(hs.rank_to_flatconfig(r)): gs[r] for r in range(hs.size)}
        ev = H.evaluate_exact_flatconfigs(lambda fc: amps[tuple(fc)])
        assert ev.real == pytest.approx(np.linalg.eigvalsh(A)[0])

        camps = {
            tuple(sorted(hs.rank_to_config(r).items())): gs[r]
            for r in range(hs.size)
        }
        ev = H.evaluate_exact_configs(
            lambda c: camps[tuple(sorted(c.items()))]
        )
        assert ev.real == pytest.approx(np.linalg.eigvalsh(A)[0])

    @pytest.mark.parametrize("order", ["blocked", "interleaved"])
    def test_aslinearoperator_and_config_coupling(self, order):
        H = self.get_hubbard(order)
        hs = H.hilbert_space
        A = H.build_dense()

        x = np.random.default_rng(0).random(hs.size)
        assert_allclose(H.aslinearoperator() @ x, A @ x, atol=1e-10)

        # coupled configs are keyed by site, so independent of the ordering
        for rank in (0, hs.size // 2, hs.size - 1):
            col = np.zeros(hs.size, dtype=complex)
            configs, coeffs = H.config_coupling(hs.rank_to_config(rank))
            for config, coeff in zip(configs, coeffs):
                col[hs.config_to_rank(config)] += coeff
            assert_allclose(col, A[:, rank], atol=1e-10)

    @pytest.mark.parametrize("order", ["blocked", "interleaved"])
    @pytest.mark.parametrize("species", [None, "spin"])
    def test_all_builders_share_a_basis(self, order, species):
        # the mpo and ikron builders work in the register ordering, so the
        # unrestricted matrix builders must not use the enumeration one
        sites = [(s, c) for s in "↑↓" for c in range(3)]
        hs = qop.HilbertSpace(
            sites,
            order=order,
            species=(lambda s: s[0]) if species else None,
        )
        H = qop.fermi_hubbard_from_edges(
            [(0, 1), (1, 2)], U=2.0, hilbert_space=hs
        )
        assert_all_matrices_match(H)

    @pytest.mark.parametrize("order", ["blocked", "interleaved"])
    def test_parallel_matches_serial(self, order):
        H = self.get_hubbard(order)
        A = H.build_sparse_matrix()
        assert_allclose(
            H.build_sparse_matrix(parallel=2).toarray(),
            A.toarray(),
            atol=1e-10,
        )
        x = np.random.default_rng(0).random(A.shape[0])
        assert_allclose(H.matvec(x, parallel=2), H.matvec(x), atol=1e-10)


@pytest.mark.parametrize("parallel", [True, 2])
def test_parallel_matches_serial_rand_operator(parallel):
    sob = qop.rand_operator(10, 20, 3, seed=42, ops="xyz+-n")
    A = sob.build_sparse_matrix()
    assert_allclose(
        sob.build_sparse_matrix(parallel=parallel).toarray(),
        A.toarray(),
        atol=1e-10,
    )
    # the operator is complex, so the vector must be too
    rng = np.random.default_rng(0)
    x = rng.random(A.shape[0]) + 1j * rng.random(A.shape[0])
    assert_allclose(
        sob.matvec(x, parallel=parallel), sob.matvec(x), atol=1e-10
    )


def test_matrix_ikron_respects_ordering():
    # regression: the terms are keyed by site, so ikron needs the register
    # of each, not the site label itself
    hs = qop.HilbertSpace([0, 1, 2], order=[2, 1, 0])
    assert hs.site_to_reg(0) == 2
    sob = qop.SparseOperatorBuilder(hilbert_space=hs)
    sob += 1.0, ("z", 0)
    A = sob.build_dense()
    # z on the last register alternates fastest
    assert_allclose(np.diag(A).real, [1, -1] * 4, atol=1e-10)
    assert_allclose(sob.build_matrix_ikron(), A, atol=1e-10)


def test_matrix_ikron_identity_term():
    sob = qop.SparseOperatorBuilder(hilbert_space=qop.HilbertSpace(3))
    sob += 2.0, ("I", 0)
    assert_allclose(sob.build_matrix_ikron(), 2.0 * np.eye(8), atol=1e-10)
