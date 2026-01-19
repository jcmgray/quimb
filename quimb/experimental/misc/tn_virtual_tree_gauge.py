import itertools

from autoray import do

from quimb.tensor import decomp
from quimb.tensor.tensor_core import (
    oset,
    tensor_make_single_bond,
)


def _get_multitree(
    self,
    ind0,
    tid0,
    r,
    exclude=None,
    max_branch=1,
):
    if exclude is None:
        exclude = set()

    t0 = self.tensor_map[tid0]
    region = {tid0}
    ix_boundary = oset(ix for ix in t0.inds if ix != ind0)

    multitree = [[(ind0, tid0, None)]]

    for d in range(1, r + 1):
        next_tids = {}
        for ix in ix_boundary:
            tid_inner, tid_outer = self.ind_map[ix]
            if tid_outer in region:
                tid_inner, tid_outer = tid_outer, tid_inner

            if tid_outer in exclude:
                continue

            if (tid_outer in next_tids) and (d > max_branch):
                continue

            next_tids.setdefault(tid_outer, []).append(
                (ix, tid_outer, tid_inner)
            )

        multitree.extend(next_tids.values())
        region.update(next_tids)
        ix_boundary = oset(
            ix
            for tid in next_tids
            for ix in self.tensor_map[tid].inds
            if ix not in ix_boundary
        )

        if not ix_boundary:
            break

    return multitree


def _compute_multitree_gauge(self, ind0, multitree):
    Gout = None

    for ix_tree in itertools.product(*multitree):
        Gs = {}
        for edge in reversed(ix_tree):
            inner_ix, tid_outer, _ = edge
            t_outer = self.tensor_map[tid_outer]
            other_ix = tuple(oix for oix in t_outer.inds if oix != inner_ix)

            for ix in other_ix:
                if ix in Gs:
                    t_outer = t_outer.gate(Gs[ix], ix)

            new_G = t_outer.compute_reduce_factor("right", other_ix, inner_ix)

            Gs[inner_ix] = new_G / do("linalg.norm", new_G)

        if Gout is None:
            Gout = Gs[ind0]
        else:
            Gout = Gout + Gs[ind0]

    return Gout


def _compress_between_multitree_tids(
    self,
    tidl,
    tidr,
    max_bond,
    cutoff,
    r=2,
    max_branch=2,
    absorb="both",
):
    tl = self.tensor_map[tidl]
    tr = self.tensor_map[tidr]

    _, bix, _ = tensor_make_single_bond(tl, tr)

    multitree_l = _get_multitree(
        self, bix, tidl, r=r, max_branch=max_branch, exclude={tidr}
    )
    multitree_r = _get_multitree(
        self, bix, tidr, r=r, max_branch=max_branch, exclude={tidl}
    )
    Rl = _compute_multitree_gauge(self, bix, multitree_l)
    Rr = _compute_multitree_gauge(self, bix, multitree_r)

    Pl, Pr = decomp.compute_oblique_projectors(
        Rl, Rr.T, max_bond=max_bond, cutoff=cutoff
    )

    tl.gate_(Pl.T, bix)
    tr.gate_(Pr, bix)


def _compress_between_unitree_tids(
    self,
    tidl,
    tidr,
    max_bond,
    cutoff,
    r=2,
    **kwargs,
):
    tl = self.tensor_map[tidl]
    tr = self.tensor_map[tidr]
    _, bix, _ = tensor_make_single_bond(tl, tr)
    unitree = self.get_tree_span([tidl, tidr], max_distance=r)
    Rl, Rr = self._compute_tree_gauges(unitree, [(tidl, bix), (tidr, bix)])
    Pl, Pr = decomp.compute_oblique_projectors(
        Rl, Rr.T, max_bond=max_bond, cutoff=cutoff, **kwargs
    )
    tl.gate_(Pl.T, bix)
    tr.gate_(Pr, bix)
