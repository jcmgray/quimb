import time

import pytest

from quimb.utils import (
    Every,
    deprecated,
    oset,
    parse_time_spec,
    raise_cant_find_library_function,
)


class TestLibraryFinding:
    def test_raise_cant_find_library_function(self):
        fn = raise_cant_find_library_function(
            "alibthatisdefinitelynotinstalledasfeasdf"
        )

        with pytest.raises(ImportError):
            fn()

    def test_deprecated(self):
        fn = deprecated(lambda: 2, "old_two", "new_two")

        with pytest.warns(Warning):
            r = fn()

        assert r == 2


class TestOset:
    def test_basic(self):
        xs = oset([3, 1, 2])
        ys = oset([3, 4, 5])
        assert list(xs | ys) == [3, 1, 2, 4, 5]
        assert list(xs & ys) == [3]
        assert list(xs - ys) == [1, 2]

        xc = xs.copy()
        assert xs._d == xc._d
        assert xs._d is not xc._d

        xs |= ys
        assert list(xc) == [3, 1, 2]
        assert list(xs) == [3, 1, 2, 4, 5]

        xs &= oset([5, 4, 2])
        assert list(xs) == [2, 4, 5]
        assert len(xs) == 3

        assert str(xs) == "oset([2, 4, 5])"

        xs.discard(6)
        xs.discard(5)
        assert len(xs) == 2

        with pytest.raises(KeyError):
            xs.remove(5)

        xs.add(10)
        assert 10 in xs
        assert len(xs) == 3

        xs -= xs
        assert not bool(xs)

    def test_multi(self):
        a = oset(range(20))
        c = oset(range(15, 35))
        b = oset(range(10, 30))

        d = oset.union(a, b, c)
        assert len(d) == 35
        d.clear()
        assert len(d) == 0

        d = oset.intersection(a, b, c)
        assert len(d) == 5

        d = oset.difference(a, b, c)
        assert len(d) == 10

        a = oset("abcdefg")
        a.intersection_update(oset("abd"), oset("bdf"))
        assert list(a) == ["b", "d"]

        a = oset("abcdefg")
        a.difference_update(oset("abd"), oset("bdf"))
        assert list(a) == ["c", "e", "g"]


class TestParseTimeSpec:
    @pytest.mark.parametrize(
        "spec,expected",
        [
            (30, 30.0),
            (2.5, 2.5),
            ("30", 30.0),
            ("30s", 30.0),
            ("90 secs", 90.0),
            ("10mins", 600.0),
            ("1.5 hours", 5400.0),
            ("1h30m", 5400.0),
            ("2 days", 172800.0),
            ("1 day 12h", 129600.0),
        ],
    )
    def test_specs(self, spec, expected):
        assert parse_time_spec(spec) == pytest.approx(expected)

    def test_bad_unit(self):
        with pytest.raises(ValueError, match="Unknown time unit"):
            parse_time_spec("10 fortnights")

    def test_no_number(self):
        with pytest.raises(ValueError, match="Could not parse"):
            parse_time_spec("soon")


class TestEvery:
    @pytest.mark.parametrize("spec", [None, 0, False])
    def test_never(self, spec):
        every = Every(spec)
        assert not every
        assert not every.due(1)

    def test_counts(self):
        every = Every(3)
        assert every
        assert [n for n in range(1, 10) if every.due(n)] == [3, 6, 9]

    def test_duration(self, monkeypatch):
        now = 1000.0
        monkeypatch.setattr(time, "time", lambda: now)

        every = Every("10mins")
        assert every
        assert every.spec == "10mins"
        # the first check just starts the clock
        assert not every.due()

        now += 599.0
        assert not every.due()
        now += 1.0
        assert every.due()
        # the clock restarts from the last due check
        assert not every.due()
        now += 600.0
        assert every.due()

    def test_duration_resets(self, monkeypatch):
        now = 1000.0
        monkeypatch.setattr(time, "time", lambda: now)

        every = Every("1min")
        assert not every.due()
        now += 59.0
        # the clock restarts, so those 59 seconds don't count
        every.reset()
        assert not every.due()
        now += 59.0
        assert not every.due()
        now += 1.0
        assert every.due()

    def test_bad_duration(self):
        with pytest.raises(ValueError, match="positive duration"):
            Every("0s")
