import pytest
from quimb.utils import (
    raise_cant_find_library_function,
    deprecated,
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
