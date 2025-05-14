import functools

import autoray as ar

import quimb.tensor as qtn
from quimb.tensor.tensor_core import check_opt
from quimb.tensor.tensor_2d import Rotator2D


def CTMRG(
    T, A, C, L, max_bond,
    strip_exponent=False,
):
    """Contract the translationally invariant tensor network given by bulk,
    edge and corner arrays ``T``, ``A`` and ``C`` respectively with side
    length ``L``.

    Parameters
    ----------
    T : array_like with shape (d, d, d, d)
        The bulk tensor, dimensions corresponding to
        (towards corner, away from corner, away from corner, towards corner),
    A : array_like with shape (d, d, d)
        The edge tensor, dimensions corresponding to
        (away from corner, towards corner, towards bulk).
    C : array_like with shape (d, d)
        The corner tensor.
    L : int
        The side length of the lattice to generate.
    max_bond : int
        The maximum bond dimension to use.
    strip_exponent : bool, optional
        Whether to strip an overall exponent while contracting and accrue this
        separately, to avoid overflow.

    Returns
    -------
    Z : float
        The contracted value of the tensor network.
    exponent : float, optional
        The exponent, in log10, accrued while contracting, such that the real
        contracted value is ``Z * 10**exponent``.
    """

    asymmetric = isinstance(A, tuple)
    if asymmetric:
        A, B = A
    else:
        B = A
        # # A = A
        # # perm = (1, 0, 2)
        # perm = (0, 1, 2)
        # B = A.transpose(perm)

    if (L < 4) or (L % 2 != 0):
        raise ValueError("`L` must be an even integer >= 4.")

    if strip_exponent:
        exponent = 0.0
    else:
        exponent = None

    order_a_inwards = True
    order_b_inwards = True

    C_inds = ('c-a', 'c-b')
    if order_a_inwards:
        A_inds = ('c-a', 'a-d', 'a-t')
    else:
        A_inds = ('a-d', 'c-a', 'a-t')
    if order_b_inwards:
        B_inds = ('c-b', 'b-r', 'b-t')
    else:
        B_inds = ('b-r', 'c-b', 'b-t')
    T_inds = ('a-t', 't-d', 't-r', 'b-t')

    AC_inds = ('a-d', 't-d')
    CB_inds = ('b-r', 't-r')

    d = T.shape[0]
    for l in range(2, L - 2, 2):
        #     ┌──┐      ┌──┐
        #     │C ├─c-b──┤B ├──b-r
        #     └┬─┘      └┬─┘
        #      │c-a      │b-t
        #     ┌┴─┐      ┌┴─┐
        #     │A ├─a-t──┤T ├──t-r
        #     └┬─┘      └┬─┘
        #      │a-d      │t-d
        tn_corner = (
            qtn.Tensor(C, inds=C_inds, tags='C') |
            qtn.Tensor(A, inds=A_inds, tags='A') |
            qtn.Tensor(B, inds=B_inds, tags='B') |
            qtn.Tensor(T, inds=T_inds, tags='T')
        )

        # make the projector
        CTM = tn_corner.to_dense(AC_inds, CB_inds)

        if asymmetric:
            s, U = ar.do("linalg.eigh", (CTM + ar.dag(CTM) / 2))
        else:
            # assert qu.isherm(CTM)
            s, U = ar.do("linalg.eigh", CTM)

        k = ar.do("argsort", -ar.do("abs", s))[:max_bond]
        U = U[:, k]
        s = s[k]
        U = U.reshape((-1, d, s.size))

        if asymmetric:
            tn_corner_proj = tn_corner.copy()
            tn_corner_proj |= qtn.Tensor(U, inds=(*AC_inds,'new-d'), tags='U')
            tn_corner_proj |= qtn.Tensor(U, inds=(*CB_inds,'new-r'), tags='V')
            C = tn_corner_proj.to_dense(['new-d'], ['new-r'])
        else:
            C = ar.do("diag", s)

        #      │a-u     │t-u
        #     ┌┴─┐     ┌┴─┐
        #     │A ├─a-t─┤T ├─t-r
        #     └┬─┘     └┬─┘
        #      │a-d     │t-d
        tn_side = (
            (
                qtn.Tensor(A, inds=['a-u', 'a-d', 'a-t'], tags='A')
                if order_a_inwards else
                qtn.Tensor(A, inds=['a-d', 'a-u', 'a-t'], tags='A')
             ) |
            qtn.Tensor(T, inds=['a-t', 't-d', 't-r', 't-u'], tags='T')
        )
        tn_side_proj = tn_side.copy()
        tn_side_proj |= qtn.Tensor(U, inds=['a-u', 't-u', 'new-u'], tags='U')
        tn_side_proj |= qtn.Tensor(U, inds=['a-d', 't-d', 'new-d'], tags='V')
        if order_a_inwards:
            A = tn_side_proj.to_dense(['new-u'], ['new-d'], ['t-r'])
        else:
            A = tn_side_proj.to_dense(['new-d'], ['new-u'], ['t-r'])

        if not asymmetric:
            B = A
        else:
            #          ┌──┐
            #      b-l─┤B ├──b-r
            #          └┬─┘
            #           │b-t
            #          ┌┴─┐
            #      t-l─┤T ├──t-r
            #          └┬─┘
            #           │t-d
            tn_side = (
                (
                    qtn.Tensor(B, inds=['b-l', 'b-r', 'b-t'], tags='B')
                    if order_b_inwards else
                    qtn.Tensor(B, inds=['b-r', 'b-l', 'b-t'], tags='B')
                ) |
                qtn.Tensor(T, inds=['t-l', 't-d', 't-r', 'b-t'], tags='T')
            )
            tn_side_proj = tn_side.copy()
            tn_side_proj |= qtn.Tensor(U, inds=['b-l', 't-l', 'n-l'], tags='U')
            tn_side_proj |= qtn.Tensor(U, inds=['b-r', 't-r', 'n-r'], tags='V')

            if order_b_inwards:
                B = tn_side_proj.to_dense(['n-l'], ['n-r'], ['t-d'])
            else:
                B = tn_side_proj.to_dense(['n-r'], ['n-l'], ['t-d'])

        if strip_exponent:
            # Anorm = ar.do('linalg.norm', A)
            # TODO: work out how to rescale given a different norm each time
            Anorm = 2
            exponent += 4 * l * ar.do('log10', Anorm)
            A = A / Anorm
            B = B / Anorm

            Cnorm = ar.do('linalg.norm', C)
            exponent += 4 * ar.do('log10', Cnorm)
            C = C / Cnorm

    tn_corner = (
        qtn.Tensor(C, inds=C_inds, tags='C') |
        qtn.Tensor(A, inds=A_inds, tags='A') |
        qtn.Tensor(B, inds=B_inds, tags='B') |
        qtn.Tensor(T, inds=T_inds, tags='T')
    )

    # make the projector
    CTM = tn_corner.to_dense(AC_inds, CB_inds)

    return ar.do('trace', CTM @ CTM @ CTM @ CTM), exponent


def coarse_grain_eager(
    self,
    direction,
    compress=True,
    equalize_norms=False,
    inplace=False,
    **compress_opts,
):
    """This contracts pairs of tensors in along ``direction``, and then
    optionally compresses the doubled bonds generated along the other
    direction.
    """
    check_opt("direction", direction, ("x", "y"))
    tn = self if inplace else self.copy()
    r2d = Rotator2D(tn, None, None, direction + "min", stepsize=2)

    # track new coordinates / tags
    retag_map = {}

    # first contract pairs along direction 'x' or 'y'
    for i in r2d.sweep:
        for j in r2d.sweep_other:
            #      │  │  │        │  │  │
            # i+1  │  O──O─       │  │  O─
            #      │╱ │  │   ->   │  │╱ │
            #  i  ═O──O──O─      ═O══O──O─
            #      │  │  │        │  │  │
            #         j              j
            tag_ij = r2d.site_tag(i, j)
            tag_ip1j = r2d.site_tag(i + 1, j)
            tn.contract_between(
                tag_ij, tag_ip1j, equalize_norms=equalize_norms
            )
            new_tag = r2d.site_tag(i // 2, j)
            retag_map[tag_ij] = new_tag
            retag_map[tag_ip1j] = new_tag

            if compress and j > 0:
                #      │  │  │        │  │  │
                # i+1  │  │  O─       │  │  O─
                #      │  │╱ │   ->   │  │╱ │
                #  i  ─O══O──O─      ─O──O──O─
                #      │  │  │        │  │  │
                #    j-1  j         j-1  j
                tag_ijm1 = r2d.site_tag(i, j - 1)
                tn.compress_between(tag_ijm1, tag_ij, **compress_opts)

        retag_map[r2d.x_tag(i)] = r2d.x_tag(i // 2)
        retag_map[r2d.x_tag(i + 1)] = r2d.x_tag(i // 2)

    # then we retag the tensor network and adjust its size
    tn.retag_(retag_map)
    if direction == "x":
        tn._Lx = tn.Lx // 2 + tn.Lx % 2
    else:
        tn._Ly = tn.Ly // 2 + tn.Ly % 2

    return tn


coarse_grain_ = functools.partialmethod(coarse_grain_eager, inplace=True)
