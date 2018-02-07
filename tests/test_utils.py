import pytest
from quimb.utils import (
    raise_cant_find_library_function,
    deprecated,
    functions_equal,
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


class TestFunctionsEqual:

    def test(self):

        def foo1(x):
            return x + 1

        def foo2(yz):
            return yz + 1

        class Foo1:

            def meth1(x):
                return x + 1

        class Foo2:

            def meth2(yz):
                return yz + 1

        # compare function-function
        assert functions_equal(foo1, foo2)

        # compare method-method
        assert functions_equal(Foo1.meth1, Foo2.meth2)

        # compare function-method
        assert functions_equal(foo1, Foo2.meth2)
        assert functions_equal(Foo1.meth1, foo2)
