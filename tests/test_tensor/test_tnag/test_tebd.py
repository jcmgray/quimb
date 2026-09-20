import pickle
import warnings

import numpy as np
import pytest
from pytest import approx

import quimb as qu
import quimb.tensor as qtn
from quimb.tensor.tnag.tebd import trotter_schedule


def dense_prod(gates, sites, d=2):
    """The dense operator the ordered sequence of ``gates`` applies, with
    ``sites`` giving the order of the subsystems.
    """
    dims = [d] * len(sites)
    where_to_inds = {site: i for i, site in enumerate(sites)}
    U = qu.eye(d ** len(sites))
    for g in gates:
        inds = [where_to_inds[site] for site in g.where]
        U = qu.pkron(g.U, dims, inds) @ U
    return U


def dense_ham(ham, sites, d=2):
    """The dense hamiltonian ``ham`` represents, with ``sites`` giving the
    order of the subsystems.
    """
    dims = [d] * len(sites)
    where_to_inds = {site: i for i, site in enumerate(sites)}
    H = np.zeros((d ** len(sites),) * 2)
    for where, term in ham.items():
        inds = [where_to_inds[site] for site in where]
        H = H + qu.pkron(term, dims, inds)
    return H


class TestTrotterSchedule:
    def test_order1(self):
        assert trotter_schedule(3, order=1) == [
            (0, 1.0),
            (1, 1.0),
            (2, 1.0),
        ]

    def test_order2_is_palindromic(self):
        assert trotter_schedule(2, order=2) == [(0, 0.5), (1, 1.0), (0, 0.5)]
        sched = trotter_schedule(4, order=2)
        assert sched == list(reversed(sched))

    def test_single_layer(self):
        for order in (1, 2, 4):
            sched = trotter_schedule(1, order=order)
            assert all(k == 0 for k, _ in sched)
            assert sum(frac for _, frac in sched) == approx(1.0)

    @pytest.mark.parametrize("order", [1, 2, 4])
    @pytest.mark.parametrize("nlayers", [1, 2, 3, 5])
    def test_every_layer_gets_unit_total_fraction(self, nlayers, order):
        sched = trotter_schedule(nlayers, order=order)
        for k in range(nlayers):
            total = sum(frac for kk, frac in sched if kk == k)
            assert total == approx(1.0)

    def test_bad_order(self):
        with pytest.raises(ValueError):
            trotter_schedule(2, order=3)


class TestGetTrotterGates:
    def test_structure_obc_order2(self):
        H = qtn.LocalHam1D(L=6, H2=qu.ham_heis(2))
        gates = H.get_trotter_gates(-0.1, order=2, steps=2)

        # even-half, odd, even (fused), odd, even-half
        assert [(g.where, g.frac, g.layer, g.step) for g in gates] == [
            ((0, 1), 0.5, 0, 0),
            ((2, 3), 0.5, 0, 0),
            ((4, 5), 0.5, 0, 0),
            ((3, 4), 1.0, 1, 0),
            ((1, 2), 1.0, 1, 0),
            ((0, 1), 1.0, 2, 0),
            ((2, 3), 1.0, 2, 0),
            ((4, 5), 1.0, 2, 0),
            ((3, 4), 1.0, 3, 1),
            ((1, 2), 1.0, 3, 1),
            ((0, 1), 0.5, 4, 1),
            ((2, 3), 0.5, 4, 1),
            ((4, 5), 0.5, 4, 1),
        ]

    def test_fused_only_has_half_gates_at_the_endpoints(self):
        H = qtn.LocalHam1D(L=8, H2=qu.ham_heis(2))
        gates = H.get_trotter_gates(-0.1, order=2, steps=5)
        halves = [g.layer for g in gates if g.frac != 1.0]
        assert set(halves) == {0, gates[-1].layer}

        # unfused instead has them at every step boundary
        gates = H.get_trotter_gates(
            -0.1, order=2, steps=5, fuse_adjacent=False
        )
        assert len({g.layer for g in gates if g.frac != 1.0}) == 10

    def test_alternate_reverses_every_other_layer(self):
        H = qtn.LocalHam1D(L=6, H2=qu.ham_heis(2))
        wheres = [
            g.where for g in H.get_trotter_gates(-0.1, order=1, alternate=True)
        ]
        assert wheres == [(0, 1), (2, 3), (4, 5), (3, 4), (1, 2)]

        wheres = [
            g.where
            for g in H.get_trotter_gates(-0.1, order=1, alternate=False)
        ]
        assert wheres == [(0, 1), (2, 3), (4, 5), (1, 2), (3, 4)]

    @pytest.mark.parametrize("cyclic", [False, True])
    @pytest.mark.parametrize("L", [4, 5, 6])
    @pytest.mark.parametrize("order", [1, 2, 4])
    def test_fusing_and_alternating_dont_change_the_propagator(
        self, L, cyclic, order
    ):
        H = qtn.LocalHam1D(L=L, H2=qu.ham_heis(2), cyclic=cyclic)
        sites = range(L)
        opts = {"order": order, "steps": 3}
        Ufused = dense_prod(H.get_trotter_gates(-0.13, **opts), sites)
        Uplain = dense_prod(
            H.get_trotter_gates(-0.13, fuse_adjacent=False, **opts), sites
        )
        Useq = dense_prod(
            H.get_trotter_gates(-0.13, alternate=False, **opts), sites
        )
        np.testing.assert_allclose(Ufused, Uplain, atol=1e-12)
        np.testing.assert_allclose(Ufused, Useq, atol=1e-12)

    @pytest.mark.parametrize("cyclic", [False, True])
    @pytest.mark.parametrize("L", [5, 6])
    @pytest.mark.parametrize("order", [1, 2, 4])
    def test_error_scales_with_order(self, L, cyclic, order):
        H = qtn.LocalHam1D(L=L, H2=qu.ham_heis(2), cyclic=cyclic)
        Hd = qu.ham_heis(L, cyclic=cyclic, sparse=False)

        errs = []
        for x in (0.08, 0.04):
            gates = H.get_trotter_gates(-1j * x, order=order)
            U = dense_prod(gates, range(L))
            errs.append(np.linalg.norm(U - qu.expm(-1j * x * Hd)))

        # halving x should reduce the error by 2**(order + 1)
        assert errs[0] / errs[1] == approx(2 ** (order + 1), rel=0.1)

    @pytest.mark.parametrize("order", [1, 2, 4])
    def test_real_time_gates_are_unitary(self, order):
        H = qtn.LocalHam1D(L=5, H2=qu.ham_heis(2), cyclic=True)
        gates = H.get_trotter_gates(-1j * 0.1, order=order, steps=2)
        U = dense_prod(gates, range(5))
        np.testing.assert_allclose(U.conj().T @ U, qu.eye(2**5), atol=1e-12)

    def test_explicit_ordering(self):
        H = qtn.LocalHam1D(L=4, H2=qu.ham_heis(2))
        layers = [((1, 2),), ((0, 1), (2, 3))]
        gates = H.get_trotter_gates(-0.1, order=2, ordering=layers)
        assert [g.where for g in gates] == [
            (1, 2),
            (2, 3),
            (0, 1),
            (1, 2),
        ]
        assert [g.frac for g in gates] == [0.5, 1.0, 1.0, 0.5]

    def test_arbitrary_geometry(self):
        edges = [(0, 1), (1, 2), (2, 0), (2, 3), (3, 4)]
        H2 = qu.ham_heis(2)
        ham = qtn.LocalHamGen(H2={e: H2 for e in edges})
        sites = range(5)
        Hd = dense_ham(ham, sites)

        errs = []
        for x in (0.08, 0.04):
            U = dense_prod(ham.get_trotter_gates(-1j * x, order=2), sites)
            errs.append(np.linalg.norm(U - qu.expm(-1j * x * Hd)))

        assert errs[0] / errs[1] == approx(8, rel=0.1)

    def test_no_terms(self):
        ham = qtn.LocalHamGen(H2={})
        assert ham.get_trotter_gates(-0.1) == ()

    def test_unpacks_as_gate_and_where(self):
        H = qtn.LocalHam1D(L=4, H2=qu.ham_heis(2))
        (g,) = H.get_trotter_gates(-0.1, order=1, ordering=[((0, 1),)])
        U, where = g
        assert where == (0, 1)
        np.testing.assert_allclose(U, g.U)
        assert (g.frac, g.layer, g.step) == (1.0, 0, 0)

        # the attributes should survive a round trip
        g2 = pickle.loads(pickle.dumps(g))
        assert (g2.where, g2.frac, g2.layer, g2.step) == (
            (0, 1),
            1.0,
            0,
            0,
        )


class TestProgressLog:
    def get_su(self, **kwargs):
        edges = qtn.edges_2d_square(2, 3)
        psi = qtn.TN_from_edges_rand(edges, D=2, phys_dim=2)
        ham = qtn.LocalHamGen(H2={e: qu.ham_heis(2) for e in edges})
        return qtn.SimpleUpdateGen(
            psi, ham, D=2, compute_energy_every=1, progbar=False, **kwargs
        )

    def test_writes_data_that_matches_the_run(self, tmp_path):
        su = self.get_su(logdir=tmp_path)
        su.evolve(5)

        payload = qu.load_progress_log(tmp_path)
        assert payload["cls"] == "SimpleUpdateGen"
        assert payload["running"] is False
        assert payload["info"]["n"] == 5
        assert payload["elapsed"] > 0.0
        assert payload["data"]["energies"]["y"] == approx(list(su.energies))
        assert payload["data"]["energies"]["x"] == approx(list(su.energy_ns))
        assert payload["data"]["gauge_diffs"]["yscale"] == "log"

    def test_watch_only_redraws_when_the_file_changes(
        self, tmp_path, monkeypatch
    ):
        import json
        import time

        import quimb.utils_plot as up

        su = self.get_su(logdir=tmp_path)
        su.evolve(3)

        # pretend the run is still going
        fprogress = tmp_path / "progress.json"
        payload = json.loads(fprogress.read_text())
        payload["running"] = True
        fprogress.write_text(json.dumps(payload))

        ndraws = []
        monkeypatch.setattr(
            up, "plot_multi_series_zoom", lambda data, **kw: ndraws.append(1)
        )

        nsleeps = []

        def fake_sleep(interval):
            nsleeps.append(1)
            if len(nsleeps) == 5:
                # the run finishes, rewriting the file
                payload["running"] = False
                fprogress.write_text(json.dumps(payload))

        monkeypatch.setattr(time, "sleep", fake_sleep)
        up.plot_progress_log(tmp_path, watch=True, clear_previous=False)

        assert len(nsleeps) == 5
        # once at the start, then only for the single rewrite
        assert len(ndraws) == 2

    def test_write_survives_a_failing_replace(self, tmp_path, monkeypatch):
        import os
        import time

        # windows raises while a reader has the destination file open
        def failing_replace(src, dst):
            raise PermissionError("file in use")

        su = self.get_su(logdir=tmp_path, log_every=100)
        su.evolve(1)
        before = qu.load_progress_log(tmp_path)

        pauses = []
        monkeypatch.setattr(os, "replace", failing_replace)
        monkeypatch.setattr(time, "sleep", pauses.append)
        su.evolve(1)

        # the run should carry on, leaving no stray temporary files
        assert su.n == 2
        assert qu.load_progress_log(tmp_path) == before
        assert [f.name for f in tmp_path.iterdir()] == ["progress.json"]
        # both writes should back off before giving up
        assert pauses == [0.01, 0.02, 0.04, 0.08, 0.16] * 2

    def test_load_progress_log_accepts_the_file_itself(self, tmp_path):
        su = self.get_su(logdir=tmp_path)
        su.evolve(2)
        assert qu.load_progress_log(tmp_path / "progress.json") == (
            qu.load_progress_log(tmp_path)
        )

    def test_log_every(self, tmp_path):
        seen = []
        su = self.get_su(logdir=tmp_path, log_every=10)
        su.callback = lambda su: seen.append(
            qu.load_progress_log(tmp_path)["info"]["n"]
        )
        su.evolve(12)
        # written once at the start, then only at the tenth sweep
        assert seen == [0] * 9 + [10] * 3
        assert qu.load_progress_log(tmp_path)["info"]["n"] == 12

    def test_log_every_time_spec(self, tmp_path, monkeypatch):
        import time

        now = 1000.0
        monkeypatch.setattr(time, "time", lambda: now)

        seen = []
        su = self.get_su(logdir=tmp_path, log_every="10mins")
        assert su.log_every == "10mins"

        def callback(su):
            nonlocal now
            seen.append(qu.load_progress_log(tmp_path)["info"]["n"])
            # each sweep takes a minute
            now += 60.0

        su.callback = callback
        su.evolve(12)
        # written once at the start, then only after ten minutes have passed
        assert seen == [0] * 10 + [11] * 2
        assert qu.load_progress_log(tmp_path)["info"]["n"] == 12

    def test_stop_file_stops_after_current_sweep(self, tmp_path):
        su = self.get_su(logdir=tmp_path)
        su.evolve(2)
        assert su.n == 2

        (tmp_path / "STOP").touch()
        su.evolve(10)
        assert su.n == 2
        # the state should still be usable, and the file consumed
        assert su.get_state().max_bond() == 2
        assert not (tmp_path / "STOP").exists()

        # a following call should not be stopped by the stale file
        su.evolve(2)
        assert su.n == 4

    def test_stop_file_survives_energy_convergence_check(self, tmp_path):
        # the energy check must not clobber an externally requested stop
        su = self.get_su(logdir=tmp_path, tol_energy_diff=0.0)
        (tmp_path / "STOP").touch()
        su.evolve(10)
        assert su.n == 0

    def test_graceful_interrupt_handler(self):
        import signal

        before = signal.getsignal(signal.SIGINT)
        during = []

        def record(su):
            during.append(signal.getsignal(signal.SIGINT))

        # the handler should be installed during the run, and restored after
        su = self.get_su()
        su.callback = record
        su.evolve(1)
        assert during[0] is not before
        assert signal.getsignal(signal.SIGINT) is before

        # unless turned off
        during.clear()
        su = self.get_su(graceful_interrupt=False)
        su.callback = record
        su.evolve(1)
        assert during[0] is before

    def test_graceful_interrupt_finishes_the_sweep(self):
        import signal

        # note os.kill with SIGINT would simply terminate us on windows
        su = self.get_su()
        su.callback = lambda su: signal.raise_signal(signal.SIGINT)
        su.evolve(10)
        # the sweep should have completed, leaving a usable state
        assert 0 < su.n < 10
        assert su.get_state().max_bond() == 2


class DriverMonitor:
    """Record completed steps and keep a reference to the driver."""

    def __init__(self, su):
        self.su = su
        self.seen = []

    def __call__(self, su):
        self.seen.append(su.n)
        return False


class SweepClock:
    """Advance a fake clock by a minute each sweep, and record the number of
    sweeps the current checkpoint holds.
    """

    def __init__(self, logdir, step=60.0):
        self.logdir = logdir
        self.step = step
        self.now = 1000.0
        self.written = []

    def __call__(self, su):
        if (self.logdir / "checkpoint.pkl").exists():
            saved = qtn.SimpleUpdateGen.from_checkpoint(self.logdir)
            self.written.append(saved.n)
        self.now += self.step
        return False


def constant_energy(_su):
    return -1.234


class TestCheckpointing:
    def get_su(self, **kwargs):
        kwargs.setdefault("progbar", False)
        kwargs.setdefault("compute_energy_every", 1)
        edges = qtn.edges_2d_square(2, 3)
        psi = qtn.TN_from_edges_rand(edges, D=2, phys_dim=2, seed=42)
        ham = qtn.LocalHamGen(H2={e: qu.ham_heis(2) for e in edges})
        return qtn.SimpleUpdateGen(psi, ham, D=2, **kwargs)

    @staticmethod
    def assert_same_state(actual, expected):
        for tensor_actual, tensor_expected in zip(actual._psi, expected._psi):
            np.testing.assert_allclose(
                tensor_actual.data, tensor_expected.data
            )
        assert len(actual.gauges) == len(expected.gauges)
        for gauge_actual, gauge_expected in zip(
            actual.gauges.values(), expected.gauges.values()
        ):
            np.testing.assert_allclose(gauge_actual, gauge_expected)

    def test_default_ordering_can_be_pickled(self):
        su = self.get_su()
        loaded = pickle.loads(pickle.dumps(su))
        assert callable(loaded.ordering)
        loaded.evolve(1)
        assert loaded.n == 1

    def test_resume_warns_if_psi_or_ham_is_supplied(self, tmp_path):
        su = self.get_su(logdir=tmp_path, checkpoint_every=1)
        su.evolve(2)

        with pytest.warns(UserWarning, match="overrides the supplied"):
            resumed = self.get_su(logdir=tmp_path, resume=True)
        assert resumed.n == 2

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            loaded = qtn.SimpleUpdateGen(logdir=tmp_path, resume=True)
        assert loaded.n == 2

    def test_checkpoint_every_time_spec(self, tmp_path, monkeypatch):
        import time

        # a picklable callback, since it gets checkpointed itself
        clock = SweepClock(tmp_path)
        monkeypatch.setattr(time, "time", lambda: clock.now)

        su = self.get_su(
            logdir=tmp_path, checkpoint_every="10mins", callback=clock
        )
        assert su.checkpoint_every == "10mins"
        su.evolve(12)

        # written only once ten minutes have passed, then finally at the end
        assert clock.written == [11, 11]
        assert qtn.SimpleUpdateGen.from_checkpoint(tmp_path).n == 12

    def test_periodic_checkpoint_and_resume(self, tmp_path):
        su = self.get_su(
            logdir=tmp_path,
            checkpoint_every=2,
        )
        su.evolve(5)

        saved = qtn.SimpleUpdateGen.from_checkpoint(tmp_path)
        saved_from_file = qtn.SimpleUpdateGen.from_checkpoint(
            tmp_path / "checkpoint.pkl"
        )
        assert saved_from_file.n == saved.n
        assert saved.n == 5
        resumed = self.get_su(logdir=tmp_path, resume=True)
        assert resumed.n == saved.n
        assert resumed.energies == saved.energies
        assert resumed.energy_ns == saved.energy_ns
        assert resumed.gauge_diffs == saved.gauge_diffs
        self.assert_same_state(resumed, saved)

    def test_resumed_run_matches_uninterrupted_run(self, tmp_path):
        edges = tuple(qtn.edges_2d_square(2, 3))
        full = self.get_su(ordering=edges)
        full.evolve(4)

        split = self.get_su(
            ordering=edges,
            logdir=tmp_path,
            checkpoint_every=2,
        )
        split.evolve(2)
        resumed = self.get_su(
            ordering=edges,
            logdir=tmp_path,
            resume=True,
        )
        resumed.evolve(2)

        assert resumed.n == full.n
        assert resumed.energies == full.energies
        assert resumed.energy_ns == full.energy_ns
        assert resumed.gauge_diffs == full.gauge_diffs
        self.assert_same_state(resumed, full)

    def test_stop_file_writes_checkpoint(self, tmp_path):
        su = self.get_su(
            logdir=tmp_path,
            checkpoint_every=100,
        )
        su.evolve(1)
        (tmp_path / "STOP").touch()
        su.evolve(10)

        saved = qtn.SimpleUpdateGen.from_checkpoint(tmp_path)
        assert saved.n == su.n == 1

    def test_resume_false_ignores_checkpoint(self, tmp_path):
        su = self.get_su(
            logdir=tmp_path,
            checkpoint_every=1,
        )
        su.evolve(1)
        fresh = self.get_su(logdir=tmp_path)
        assert fresh.n == 0

    def test_logdir_does_not_enable_checkpointing(self, tmp_path):
        su = self.get_su(logdir=tmp_path)
        su.evolve(1)
        assert not (tmp_path / "checkpoint.pkl").exists()

    def test_resume_does_not_need_psi_or_ham(self, tmp_path):
        su = self.get_su(logdir=tmp_path, checkpoint_every=1)
        su.evolve(2)

        resumed = qtn.SimpleUpdateGen(logdir=tmp_path, resume=True)
        assert resumed.resumed
        assert resumed.n == 2
        self.assert_same_state(resumed, su)

        resumed.evolve(1)
        assert resumed.n == 3

    def test_resume_skips_the_initial_equilibration(
        self, tmp_path, monkeypatch
    ):
        su = self.get_su(
            logdir=tmp_path, checkpoint_every=1, equilibrate_start=True
        )
        su.evolve(1)

        calls = []
        original = qtn.SimpleUpdateGen.equilibrate

        def record_equilibration(self, *args, **kwargs):
            calls.append(1)
            return original(self, *args, **kwargs)

        monkeypatch.setattr(
            qtn.SimpleUpdateGen,
            "equilibrate",
            record_equilibration,
        )
        qtn.SimpleUpdateGen(logdir=tmp_path, resume=True)
        assert not calls

    def test_starting_fresh_still_needs_psi_and_ham(self, tmp_path):
        with pytest.raises(ValueError, match="psi0"):
            qtn.SimpleUpdateGen(logdir=tmp_path, resume=True)

    def test_resume_with_no_checkpoint_starts_fresh(self, tmp_path):
        su = self.get_su(logdir=tmp_path, resume=True)
        assert su.n == 0

    def test_resume_keeps_session_options(self, tmp_path):
        su = self.get_su(
            logdir=tmp_path,
            checkpoint_every=1,
        )
        su.evolve(1)

        callback = lambda _su: False
        resumed = self.get_su(
            logdir=tmp_path,
            resume=True,
            callback=callback,
            progbar=True,
            plot_every=7,
        )
        assert resumed.callback is callback
        assert resumed.progbar is True
        assert resumed.plot_every == 7

    @pytest.mark.parametrize(
        "kwargs", [{"checkpoint_every": 1}, {"resume": True}]
    )
    def test_checkpointing_requires_logdir(self, kwargs):
        with pytest.raises(ValueError, match="logdir"):
            self.get_su(**kwargs)

    @pytest.mark.parametrize("steps", [4, 5])
    def test_final_checkpoint_after_normal_completion(self, tmp_path, steps):
        su = self.get_su(logdir=tmp_path, checkpoint_every=2)
        su.evolve(steps)

        saved = qtn.SimpleUpdateGen.from_checkpoint(tmp_path)
        assert saved.n == su.n == steps
        assert saved.energies == su.energies
        assert saved.energy_ns == su.energy_ns

    def test_string_ordering_is_unchanged_after_reload(self, tmp_path):
        su = self.get_su(
            ordering="random_sequential",
            logdir=tmp_path,
            checkpoint_every=1,
        )
        su.evolve(1)
        saved = qtn.SimpleUpdateGen.from_checkpoint(tmp_path)
        assert tuple(saved._ordering) == tuple(su._ordering)

    def test_lambda_ordering_is_omitted(self, tmp_path):
        su = self.get_su(
            ordering=lambda: list(qtn.edges_2d_square(2, 3)),
            logdir=tmp_path,
            checkpoint_every=1,
        )
        with pytest.warns(UserWarning, match="cannot be pickled"):
            su.evolve(1)
        assert su.n == 1

        saved = qtn.SimpleUpdateGen.from_checkpoint(tmp_path)
        assert saved.ordering.ham is saved.ham
        saved.evolve(1)
        assert saved.n == 2

    def test_energy_options_come_from_the_checkpoint(self, tmp_path):
        su = self.get_su(
            logdir=tmp_path,
            checkpoint_every=1,
            compute_energy_fn=constant_energy,
            compute_energy_every=3,
        )
        su.evolve(1)

        resumed = self.get_su(
            logdir=tmp_path, resume=True, compute_energy_every=7
        )
        assert resumed.compute_energy_fn is constant_energy
        assert resumed.compute_energy_every == 3

    def test_lambda_energy_fn_falls_back_to_supplied(self, tmp_path):
        su = self.get_su(
            logdir=tmp_path,
            checkpoint_every=1,
            compute_energy_fn=lambda _su: -1.234,
        )
        with pytest.warns(UserWarning, match="`compute_energy_fn`"):
            su.evolve(1)

        replacement = lambda _su: -5.678
        resumed = self.get_su(
            logdir=tmp_path, resume=True, compute_energy_fn=replacement
        )
        assert resumed.compute_energy_fn is replacement

    def test_callback_with_driver_reference_is_checkpointed(self, tmp_path):
        # pickle must preserve the callback's reference back to the driver
        su = self.get_su(logdir=tmp_path, checkpoint_every=1)
        su.callback = DriverMonitor(su)
        su.evolve(2)
        assert su.callback.seen == [1, 2]

        saved = qtn.SimpleUpdateGen.from_checkpoint(tmp_path)
        assert saved.callback is not None
        assert saved.callback.su is saved

    def test_unpicklable_callback_does_not_stop_run(self, tmp_path):
        su = self.get_su(logdir=tmp_path, checkpoint_every=1)
        su.callback = lambda su: False
        with pytest.warns(UserWarning, match="cannot be pickled"):
            su.evolve(2)
        assert su.n == 2
        assert qtn.SimpleUpdateGen.from_checkpoint(tmp_path).n == 2

    def test_moved_checkpoint_uses_new_directory(self, tmp_path):
        old_dir = tmp_path / "old"
        su = self.get_su(logdir=old_dir, checkpoint_every=1)
        su.evolve(1)

        new_dir = tmp_path / "new"
        new_dir.mkdir()
        (new_dir / "checkpoint.pkl").write_bytes(
            (old_dir / "checkpoint.pkl").read_bytes()
        )

        moved = qtn.SimpleUpdateGen.from_checkpoint(new_dir)
        assert moved.logdir == new_dir
        moved.checkpoint_every = 1
        moved.evolve(1)
        assert (new_dir / "progress.json").exists()
        assert qtn.SimpleUpdateGen.from_checkpoint(new_dir).n == 2
        assert qtn.SimpleUpdateGen.from_checkpoint(old_dir).n == 1

    def test_ham_op_cache_is_not_checkpointed(self, tmp_path):
        su = self.get_su(logdir=tmp_path, checkpoint_every=1)
        su.evolve(1)
        assert su.ham._op_cache["expm"]

        # loaded objects have new ids, so the cache must be empty
        saved = qtn.SimpleUpdateGen.from_checkpoint(tmp_path)
        assert not saved.ham._op_cache["expm"]
        saved.evolve(1)
        assert saved.n == 2

    def test_wrong_class_in_checkpoint_raises(self, tmp_path):
        su = self.get_su(logdir=tmp_path, checkpoint_every=1)
        su.evolve(1)
        with pytest.raises(TypeError, match="SimpleUpdateGen"):
            qtn.TEBDGen.from_checkpoint(tmp_path)

    def test_lambda_callback_is_omitted(self, tmp_path):
        callback = lambda _su: False
        su = self.get_su(
            logdir=tmp_path,
            checkpoint_every=1,
            callback=callback,
        )
        with pytest.warns(UserWarning, match="cannot be pickled"):
            su.evolve(1)

        saved = qtn.SimpleUpdateGen.from_checkpoint(tmp_path)
        assert saved.callback is None
