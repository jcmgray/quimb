import pytest
from quimb.utils import (
    raise_cant_find_library_function,
    deprecated,
    oset,
)


class TestLibraryFinding:

    def test_raise_cant_find_library_function(self):
        fn = raise_cant_find_library_function(
            "alibthatisdefinitelynotinstalledasfeasdf")

        with pytest.raises(ImportError):
            fn()

    def test_deprecated(self):
        fn = deprecated(lambda: 2, 'old_two', 'new_two')

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

        assert str(xs) == 'oset([2, 4, 5])'

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

        a = oset('abcdefg')
        a.intersection_update(oset('abd'), oset('bdf'))
        assert list(a) == ['b', 'd']

        a = oset('abcdefg')
        a.difference_update(oset('abd'), oset('bdf'))
        assert list(a) == ['c', 'e', 'g']
