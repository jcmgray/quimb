import itertools


def sort_unique(edges):
    """Make sure there are no duplicate edges and that for each
    ``coo_a < coo_b``.
    """
    return tuple(sorted(
        tuple(sorted(edge))
        for edge in set(map(frozenset, edges))
    ))


# ----------------------------------- 2D ------------------------------------ #

def check_2d(coo, Lx, Ly, cyclic):
    """Check ``coo`` in inbounds for a maybe cyclic 2D lattice.
    """
    x, y = coo
    if (not cyclic) and not ((0 <= x < Lx) and (0 <= y < Ly)):
        return
    return (x % Lx, y % Ly)


def edges_2d_square(Lx, Ly, cyclic=False, cells=None):
    """Return the graph edges of a finite 2D square lattice. The nodes
    (sites) are labelled like ``(i, j)``.

    Parameters
    ----------
    Lx : int
        The number of cells along the x-direction.
    Ly : int
        The number of cells along the y-direction.
    cyclic : bool, optional
        Whether to use periodic boundary conditions.
    cells : list, optional
        A list of cells to use. If not given the cells used are
        ``itertools.product(range(Lx), range(Ly))``.

    Returns
    -------
    edges : list[((int, int), (int, int))]
    """
    if cells is None:
        cells = itertools.product(range(Lx), range(Ly))

    edges = []
    for i, j in cells:
        for coob in [(i, j + 1), (i + 1, j)]:
            coob = check_2d(coob, Lx, Ly, cyclic)
            if coob:
                edges.append(((i, j), coob))

    return sort_unique(edges)


def edges_2d_hexagonal(Lx, Ly, cyclic=False, cells=None):
    """Return the graph edges of a finite 2D hexagonal lattice. There are two
    sites per cell, and note the cells do not form a square tiling. The nodes
    (sites) are labelled like ``(i, j, s)`` for ``s`` in ``'AB'``.

    Parameters
    ----------
    Lx : int
        The number of cells along the x-direction.
    Ly : int
        The number of cells along the y-direction.
    cyclic : bool, optional
        Whether to use periodic boundary conditions.
    cells : list, optional
        A list of cells to use. If not given the cells used are
        ``itertools.product(range(Lx), range(Ly))``.

    Returns
    -------
    edges : list[((int, int, str), (int, int, str))]
    """
    if cells is None:
        cells = itertools.product(range(Lx), range(Ly))

    edges = []
    for i, j in cells:
        for *coob, lbl in [
            (i, j, 'B'),
            (i, j - 1, 'B'),
            (i - 1, j, 'B'),
        ]:
            coob = check_2d(coob, Lx, Ly, cyclic)
            if coob:
                edges.append(((i, j, 'A'), (*coob, lbl)))

        for *coob, lbl in [
            (i, j, 'A'),
            (i, j + 1, 'A'),
            (i + 1, j, 'A'),
        ]:
            coob = check_2d(coob, Lx, Ly, cyclic)
            if coob:
                edges.append(((i, j, 'B'), (*coob, lbl)))

    return sort_unique(edges)


def edges_2d_triangular(Lx, Ly, cyclic=False, cells=None):
    """Return the graph edges of a finite 2D triangular lattice. There is a
    single site per cell, and note the cells do not form a square tiling.
    The nodes (sites) are labelled like ``(i, j)``.

    Parameters
    ----------
    Parameters
    ----------
    Lx : int
        The number of cells along the x-direction.
    Ly : int
        The number of cells along the y-direction.
    cyclic : bool, optional
        Whether to use periodic boundary conditions.
    cells : list, optional
        A list of cells to use. If not given the cells used are
        ``itertools.product(range(Lx), range(Ly))``.

    Returns
    -------
    edges : list[((int, int), (int, int))]
    """
    if cells is None:
        cells = itertools.product(range(Lx), range(Ly))

    edges = []
    for i, j in cells:
        for coob in [(i, j + 1), (i + 1, j), (i + 1, j - 1)]:
            coob = check_2d(coob, Lx, Ly, cyclic)
            if coob:
                edges.append(((i, j), coob))

    return sort_unique(edges)


def edges_2d_triangular_rectangular(Lx, Ly, cyclic=False, cells=None):
    """Return the graph edges of a finite 2D triangular lattice tiled in a
    rectangular geometry. There are two sites per rectangular cell. The nodes
    (sites) are labelled like ``(i, j, s)`` for ``s`` in ``'AB'``.

    Parameters
    ----------
    Lx : int
        The number of cells along the x-direction.
    Ly : int
        The number of cells along the y-direction.
    cyclic : bool, optional
        Whether to use periodic boundary conditions.
    cells : list, optional
        A list of cells to use. If not given the cells used are
        ``itertools.product(range(Lx), range(Ly))``.

    Returns
    -------
    edges : list[((int, int, s), (int, int, s))]
    """
    if cells is None:
        cells = itertools.product(range(Lx), range(Ly))

    edges = []
    for i, j in cells:
        for *coob, lbl in [
            (i, j, 'B'),
            (i, j - 1, 'B'),
            (i, j + 1, 'A'),
        ]:
            coob = check_2d(coob, Lx, Ly, cyclic)
            if coob:
                edges.append(((i, j, 'A'), (*coob, lbl)))

        for *coob, lbl in [
            (i + 1, j, 'A'),
            (i, j + 1, 'B'),
            (i + 1, j + 1, 'A'),
        ]:
            coob = check_2d(coob, Lx, Ly, cyclic)
            if coob:
                edges.append(((i, j, 'B'), (*coob, lbl)))

    return sort_unique(edges)


def edges_2d_kagome(Lx, Ly, cyclic=False, cells=None):
    """Return the graph edges of a finite 2D kagome lattice. There are
    three sites per cell, and note the cells do not form a square tiling. The
    nodes (sites) are labelled like ``(i, j, s)`` for ``s`` in ``'ABC'``.

    Parameters
    ----------
    Lx : int
        The number of cells along the x-direction.
    Ly : int
        The number of cells along the y-direction.
    cyclic : bool, optional
        Whether to use periodic boundary conditions.
    cells : list, optional
        A list of cells to use. If not given the cells used are
        ``itertools.product(range(Lx), range(Ly))``.

    Returns
    -------
    edges : list[((int, int, str), (int, int, str))]
    """
    if cells is None:
        cells = itertools.product(range(Lx), range(Ly))

    edges = []
    for i, j in cells:
        for *coob, lbl in [
            (i, j, 'B'),
            (i, j - 1, 'B'),
            (i, j, 'C'),
            (i - 1, j, 'C')
        ]:
            coob = check_2d(coob, Lx, Ly, cyclic)
            if coob:
                edges.append(((i, j, 'A'), (*coob, lbl)))

        for *coob, lbl in [
            (i, j, 'C'),
            (i - 1, j + 1, 'C'),
            (i, j, 'A'),
            (i, j + 1, 'A')
        ]:
            coob = check_2d(coob, Lx, Ly, cyclic)
            if coob:
                edges.append(((i, j, 'B'), (*coob, lbl)))

        for *coob, lbl in [
            (i, j, 'A'),
            (i + 1, j, 'A'),
            (i, j, 'B'),
            (i + 1, j - 1, 'B')
        ]:
            coob = check_2d(coob, Lx, Ly, cyclic)
            if coob:
                edges.append(((i, j, 'C'), (*coob, lbl)))

    return sort_unique(edges)


# ----------------------------------- 3D ------------------------------------ #

def check_3d(coo, Lx, Ly, Lz, cyclic):
    """Check ``coo`` in inbounds for a maybe cyclic 3D lattice.
    """
    x, y, z = coo
    OBC = not cyclic
    inbounds = (0 <= x < Lx) and (0 <= y < Ly) and (0 <= z < Lz)
    if OBC and not inbounds:
        return
    return (x % Lx, y % Ly, z % Lz)


def edges_3d_cubic(Lx, Ly, Lz, cyclic=False, cells=None):
    """Return the graph edges of a finite 3D cubic lattice. The nodes
    (sites) are labelled like ``(i, j, k)``.

    Parameters
    ----------
    Lx : int
        The number of cells along the x-direction.
    Ly : int
        The number of cells along the y-direction.
    Lz : int
        The number of cells along the z-direction.
    cyclic : bool, optional
        Whether to use periodic boundary conditions.
    cells : list, optional
        A list of cells to use. If not given the cells used are
        ``itertools.product(range(Lx), range(Ly), range(Lz))``.

    Returns
    -------
    edges : list[((int, int, int), (int, int, int))]
    """
    if cells is None:
        cells = itertools.product(range(Lx), range(Ly), range(Lz))

    edges = []
    for i, j, k in cells:
        for coob in [(i, j, k + 1), (i, j + 1, k), (i + 1, j, k)]:
            coob = check_3d(coob, Lx, Ly, Lz, cyclic)
            if coob:
                edges.append(((i, j, k), coob))

    return sort_unique(edges)


def edges_3d_pyrochlore(Lx, Ly, Lz, cyclic=False, cells=None):
    """Return the graph edges of a finite 3D pyorchlore lattice. There are
    four sites per cell, and note the cells do not form a cubic tiling. The
    nodes (sites) are labelled like ``(i, j, k, s)`` for ``s`` in ``'ABCD'``.

    Parameters
    ----------
    Lx : int
        The number of cells along the x-direction.
    Ly : int
        The number of cells along the y-direction.
    Lz : int
        The number of cells along the z-direction.
    cyclic : bool, optional
        Whether to use periodic boundary conditions.
    cells : list, optional
        A list of cells to use. If not given the cells used are
        ``itertools.product(range(Lx), range(Ly), range(Lz))``.

    Returns
    -------
    edges : list[((int, int, int, str), (int, int, int, str))]
    """
    if cells is None:
        cells = itertools.product(range(Lx), range(Ly), range(Lz))

    edges = []
    for i, j, k in cells:
        for *coob, lbl in [
            (i, j, k, 'B'),
            (i, j - 1, k, 'B'),
            (i, j, k, 'C'),
            (i - 1, j, k, 'C'),
            (i, j, k, 'D'),
            (i, j, k - 1, 'D'),
        ]:
            coob = check_3d(coob, Lx, Ly, Lz, cyclic)
            if coob:
                edges.append(((i, j, k, 'A'), (*coob, lbl)))

        for *coob, lbl in [
            (i, j, k, 'C'),
            (i - 1, j + 1, k, 'C'),
            (i, j, k, 'D'),
            (i, j + 1, k - 1, 'D'),
            (i, j, k, 'A'),
            (i, j + 1, k, 'A'),
        ]:
            coob = check_3d(coob, Lx, Ly, Lz, cyclic)
            if coob:
                edges.append(((i, j, k, 'B'), (*coob, lbl)))

        for *coob, lbl in [
            (i, j, k, 'D'),
            (i + 1, j, k - 1, 'D'),
            (i, j, k, 'A'),
            (i + 1, j, k, 'A'),
            (i, j, k, 'B'),
            (i + 1, j - 1, k, 'B'),
        ]:
            coob = check_3d(coob, Lx, Ly, Lz, cyclic)
            if coob:
                edges.append(((i, j, k, 'C'), (*coob, lbl)))

        for *coob, lbl in [
            (i, j, k, 'A'),
            (i, j, k + 1, 'A'),
            (i, j, k, 'B'),
            (i, j - 1, k + 1, 'B'),
            (i, j, k, 'C'),
            (i - 1, j, k + 1, 'C'),
        ]:
            coob = check_3d(coob, Lx, Ly, Lz, cyclic)
            if coob:
                edges.append(((i, j, k, 'D'), (*coob, lbl)))

    return sort_unique(edges)


def edges_3d_diamond(Lx, Ly, Lz, cyclic=False, cells=None):
    """Return the graph edges of a finite 3D diamond lattice. There are
    two sites per cell, and note the cells do not form a cubic tiling.  The
    nodes (sites) are labelled like ``(i, j, k, s)`` for ``s`` in ``'AB'``.

    Parameters
    ----------
    Lx : int
        The number of cells along the x-direction.
    Ly : int
        The number of cells along the y-direction.
    Lz : int
        The number of cells along the z-direction.
    cyclic : bool, optional
        Whether to use periodic boundary conditions.
    cells : list, optional
        A list of cells to use. If not given the cells used are
        ``itertools.product(range(Lx), range(Ly), range(Lz))``.

    Returns
    -------
    edges : list[((int, int, int, str), (int, int, int, str))]
    """
    if cells is None:
        cells = itertools.product(range(Lx), range(Ly), range(Lz))

    edges = []
    for i, j, k in cells:
        for *coob, lbl in [
            (i, j, k, 'B'),
        ]:
            coob = check_3d(coob, Lx, Ly, Lz, cyclic)
            if coob:
                edges.append(((i, j, k, 'A'), (*coob, lbl)))

        for *coob, lbl in [
                (i, j, k + 1, 'A'),
                (i, j + 1, k, 'A'),
                (i + 1, j, k, 'A'),
        ]:
            coob = check_3d(coob, Lx, Ly, Lz, cyclic)
            if coob:
                edges.append(((i, j, k, 'B'), (*coob, lbl)))

    return sort_unique(edges)


def edges_3d_diamond_cubic(Lx, Ly, Lz, cyclic=False, cells=None):
    """Return the graph edges of a finite 3D diamond lattice tiled in a cubic
    geometry. There are eight sites per cubic cell. The nodes (sites) are
    labelled like ``(i, j, k, s)`` for ``s`` in ``'ABCDEFGH'``.

    Parameters
    ----------
    Lx : int
        The number of cells along the x-direction.
    Ly : int
        The number of cells along the y-direction.
    Lz : int
        The number of cells along the z-direction.
    cyclic : bool, optional
        Whether to use periodic boundary conditions.
    cells : list, optional
        A list of cells to use. If not given the cells used are
        ``itertools.product(range(Lx), range(Ly), range(Lz))``.

    Returns
    -------
    edges : list[((int, int, int, str), (int, int, int, str))]
    """

    if cells is None:
        cells = itertools.product(range(Lx), range(Ly), range(Lz))

    edges = []
    for i, j, k in cells:
        for *coob, lbl in [
            (i, j, k, 'E'),
        ]:
            coob = check_3d(coob, Lx, Ly, Lz, cyclic)
            if coob:
                edges.append(((i, j, k, 'A'), (*coob, lbl)))

        for *coob, lbl in [
            (i, j, k, 'E'),
            (i, j, k, 'F'),
        ]:
            coob = check_3d(coob, Lx, Ly, Lz, cyclic)
            if coob:
                edges.append(((i, j, k, 'B'), (*coob, lbl)))

        for *coob, lbl in [
            (i, j, k, 'E'),
            (i, j, k, 'G'),
        ]:
            coob = check_3d(coob, Lx, Ly, Lz, cyclic)
            if coob:
                edges.append(((i, j, k, 'C'), (*coob, lbl)))

        for *coob, lbl in [
            (i, j, k, 'E'),
            (i, j, k, 'H'),
        ]:
            coob = check_3d(coob, Lx, Ly, Lz, cyclic)
            if coob:
                edges.append(((i, j, k, 'D'), (*coob, lbl)))

        for *coob, lbl in []:
            coob = check_3d(coob, Lx, Ly, Lz, cyclic)
            if coob:
                edges.append(((i, j, k, 'E'), (*coob, lbl)))

        for *coob, lbl in [
            (i, j + 1, k, 'C'),
            (i + 1, j, k, 'D'),
        ]:
            coob = check_3d(coob, Lx, Ly, Lz, cyclic)
            if coob:
                edges.append(((i, j, k, 'F'), (*coob, lbl)))

        for *coob, lbl in [
            (i + 1, j, k + 1, 'A'),
            (i, j, k + 1, 'B'),
            (i + 1, j, k, 'D'),
        ]:
            coob = check_3d(coob, Lx, Ly, Lz, cyclic)
            if coob:
                edges.append(((i, j, k, 'G'), (*coob, lbl)))

        for *coob, lbl in [
            (i, j + 1, k + 1, 'A'),
            (i, j, k + 1, 'B'),
            (i, j + 1, k, 'C'),
        ]:
            coob = check_3d(coob, Lx, Ly, Lz, cyclic)
            if coob:
                edges.append(((i, j, k, 'H'), (*coob, lbl)))

    return sort_unique(edges)
