"""Tools for defining and constructing sparse operators with:

    * arbitrary geometries,
    * numba acceleration,
    * support for symmetic sectors,
    * efficient parallelization,

and optionally producing:

    * sparse matrix form
    * matrix product operators,
    * dict of local gates form
    * VMC 'coupled configs' form

Currently only supports composing operators which are sums of products of
diagonal or anti-diagonal real dimension 2 operators.

TODO::

    - [ ] support for non-diagonal and qudit operators (lower priority)
    - [ ] product of operators generator (e.g. for PEPS DMRG)

DONE::

    - [x] fix sparse matrix being built in opposite direction
    - [x] complex and single precision support (lower priority)
    - [x] use compact bitbasis
    - [x] design interface for HilbertSpace / OperatorBuilder interaction
    - [x] automatic symbolic jordan wigner transformation
    - [x] numba accelerated coupled config
    - [x] general definition and automatic 'symbolic' jordan wigner
    - [x] multithreaded sparse matrix construction
    - [x] LocalHam generator (e.g. for simple update, normal PEPS algs)
    - [x] automatic MPO generator

"""

from .hilbertspace import (
    HilbertSpace,
)
from .models import (
    fermi_hubbard_from_edges,
    fermi_hubbard_spinless_from_edges,
    heisenberg_from_edges,
    rand_operator,
)
from .builder import (
    get_mat,
    SparseOperatorBuilder,
)

__all__ = (
    "fermi_hubbard_from_edges",
    "fermi_hubbard_spinless_from_edges",
    "heisenberg_from_edges",
    "HilbertSpace",
    "rand_operator",
    "SparseOperatorBuilder",
    "get_mat",
)
