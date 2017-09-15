import pytest
from quimb.utils import (
    raise_cant_find_library_function,
)


class TestLibraryFinding:

    def test_raise_cant_find_library_function(self):
        fn = raise_cant_find_library_function(
            "alibthatisdefinitelynotinstalledasfeasdf")

        with pytest.raises(ImportError):

            fn()
