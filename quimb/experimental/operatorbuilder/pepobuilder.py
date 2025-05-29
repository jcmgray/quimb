import quimb.tensor as qtn


def make_w_array(Lx, Ly, i, j, A, B, C, cyclic=False, dtype=None):
    import numpy as np

    try:
        is_cyclic_x, is_cyclic_y = cyclic
    except TypeError:
        is_cyclic_x = is_cyclic_y = cyclic

    if dtype is None:
        dtype = np.common_type(A, B, C)

    # virtual bond dimension
    D = 3
    # physical ind size
    p = 2

    I = np.identity(p, dtype=dtype)
    W = np.zeros((D, D, D, D, p, p), dtype=dtype)

    par = 1
    vac = 0
    ex0 = 2

    base_selector = [vac] * 4 + [slice(None)] * 2

    def select(kv=None, ref=None):
        if ref is None:
            selector = list(base_selector)
        else:
            selector = list(ref)
        if kv is not None:
            for k, v in kv.items():
                r = "urdl".index(k)
                selector[r] = v
        return tuple(selector)

    # shape layout:
    # u  r  d  l  k  b
    # basic vacuum term
    W[select()] = I

    din, dout = {
        # starting corner
        # -> no incoming particle required, branches both up and to the right
        (True, True): ("", "ur"),
        # left boundary
        # -> particle comes from below and branches both up and to the right
        (True, False): ("d", "ur"),
        # bottom boundary
        # -> particle comes from the left and continues to the right
        (False, True): ("l", "r"),
        # bulk
        # -> particle comes from the left and continues to the right
        (False, False): ("l", "r"),
    }[j == 0, i == 0]

    # condition on having incoming particle
    sin = select({d: par for d in din})

    # 1-site decay into vacuum
    # for the starting corner this overwrites the vacuum term
    W[sin] = C

    # basic propagation of particle
    for d in dout:
        W[select({d: par}, sin)] = I

    # start of decay into 2-site interaction, requires input
    for d in "ur":
        W[select({d: ex0}, sin)] = A

    # end of decay into 2-site interaction, into vacuum
    W[select({"d": ex0})] = B
    W[select({"l": ex0})] = B

    # for PBC decay can happen next to particle propagation
    if j == 0 and i > 0:
        W[select({"l": ex0, "d": par, "r": par})] = B

    if i == 0 and j > 0:
        W[select({"d": ex0, "u": par, "l": par})] = B

    if i == 0 and j == 0:
        W[select({"d": ex0, "u": par})] = B
        W[select({"l": ex0, "r": par})] = B

    # project vacuum along boundary
    slicer = []
    for isboundary in [
        # up
        (i == Lx - 1 and not is_cyclic_x),
        # right
        (j == Ly - 1 and not is_cyclic_y),
        # down
        (i == 0 and not is_cyclic_x),
        # left
        (j == 0 and not is_cyclic_y),
    ]:
        slicer.append(vac if isboundary else slice(None))

    return W[tuple(slicer)]


def PEPO_nearest_neighbor(
    A, B, C, Lx, Ly, cyclic=False, dtype=None, **pepo_opts
):
    r"""Create a PEPO for a sum of nearest neighbor interactions:

    .. math::

        H = \sum_{\{i,j\}} A_{i} \otimes B_{j} + \sum_i C_{i}

    where A and B define the left and right hand side of an interaction, and C
    is an on-site term. For example, the transverse field ising model can be
    written with `A = -j * Z`, `B = Z`, and `C = -h * X`.

    Parameters
    ----------
    A : array_like
        The left hand side interaction matrix.
    B : array_like
        The right hand side interaction matrix.
    C : array_like
        The on-site interaction matrix.
    Lx : int
        The number of sites in the x-direction.
    Ly : int
        The number of sites in the y-direction.
    dtype : str or dtype, optional
        The data type of the PEPO tensors. If not provided, it will be inferred
        from A, B, and C.
    pepo_opts : dict, optional
        Additional options to pass to the `qtn.PEPO` constructor.

    Returns
    -------
    PEPO
    """
    Ws = [
        [
            make_w_array(Lx, Ly, i, j, A, B, C, cyclic=cyclic, dtype=dtype)
            for j in range(Ly)
        ]
        for i in range(Lx)
    ]

    return qtn.PEPO(
        Ws,
        shape="urdlkb",
        **pepo_opts,
    )
