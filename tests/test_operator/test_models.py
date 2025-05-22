import pytest

import quimb as qu
import quimb.experimental.operatorbuilder as qop


def test_fermi_hubbard_hex():
    # quspin reference:
    # https://quspin.github.io/QuSpin/examples/example18.html#example18-label
    # hexagonal graph
    edges = [
        ((0, 0), (0, 1)),
        ((0, 0), (1, 0)),
        ((0, 1), (0, 2)),
        ((0, 2), (0, 3)),
        ((0, 2), (1, 2)),
        ((0, 3), (0, 4)),
        ((0, 4), (1, 4)),
        ((1, 0), (1, 1)),
        ((1, 1), (1, 2)),
        ((1, 1), (2, 1)),
        ((1, 2), (1, 3)),
        ((1, 3), (1, 4)),
        ((1, 3), (2, 3)),
        ((1, 4), (1, 5)),
        ((1, 5), (2, 5)),
        ((2, 1), (2, 2)),
        ((2, 2), (2, 3)),
        ((2, 3), (2, 4)),
        ((2, 4), (2, 5)),
    ]
    N_up = 2  # number of spin-up fermions
    N_down = 2  # number of spin-down fermions
    t = 1.0  # tunnelling matrix element
    U = 2.0  # on-site fermion interaction strength
    sob = qop.fermi_hubbard_from_edges(edges, t=t, U=U)
    sector = (
        (sob.nsites // 2, N_up),
        (sob.nsites // 2, N_down),
    )
    assert sob.hilbert_space.get_size(sector) == 14400
    # build sparse matrix
    H = sob.build_sparse_matrix(sector=sector)
    # solve for groundstate
    energy = qu.groundenergy(H)
    assert energy == pytest.approx(-8.67415949)
