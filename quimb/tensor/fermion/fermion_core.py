"""Core tensor network tools."""

import contextlib
import copy
import functools
from functools import wraps

import numpy as np
import opt_einsum as oe
import scipy.sparse.linalg as spla
from autoray import conj
from opt_einsum.contract import _tensordot, _transpose, parse_backend

from ...utils import check_opt, oset, valmap
from ..tensor_core import (
    TensorNetwork,
    get_tensor_linop_backend,
    group_inds,
    rand_uuid,
    tags_to_oset,
    tensor_canonize_bond,
    tensor_compress_bond,
    tensor_contract,
    tensor_split,
)
from .block_interface import Constructor
from .block_tools import inv_with_smudge, sqrt
from .tensor_block import (
    BlockTensor,
    BlockTensorNetwork,
    tensor_balance_bond_block_tensor,
    tensor_canonize_bond_block_tensor,
    tensor_compress_bond_block_tensor,
    tensor_split_block_tensor,
)


class FermionSpace:
    def __init__(self, tensor_order=None, virtual=True):
        self.tensor_order = {}
        self._tid_counter = 0
        if tensor_order is not None:
            for tid, (tsr, site) in tensor_order.items():
                self.add_tensor(tsr, tid, site, virtual=virtual)

    def _next_tid(self):
        while self._tid_counter in self.tensor_order:
            self._tid_counter = self._tid_counter + 1
        return self._tid_counter

    @property
    def sites(self):
        """return a list of all the occupied positions"""
        if len(self.tensor_order) == 0:
            return []
        else:
            return [val[1] for val in self.tensor_order.values()]

    def copy(self):
        """Copy the FermionSpace object. Tensor ids and positions will be
        preserved and tensors will be copied
        """
        new_fs = FermionSpace(self.tensor_order, virtual=False)
        new_fs._tid_counter = self._tid_counter
        return new_fs

    def add_tensor(self, tsr, tid=None, site=None, virtual=False):
        """Add a tensor to the current FermionSpace, eg
        01234            0123456
        XXXXX, (6, B) -> XXXXX-B

        Parameters
        ----------
        tsr : FermionTensor
            The fermionic tensor to operate on
        tid : string, optional
            tensor id
        site: int or None, optional
            The position to place the tensor. Tensor will be
            appended to last position if not specified
        virtual: bool, optional
            whether to add the tensor inplace

        """
        if (tid is None) or (tid in self.tensor_order):
            tid = self._next_tid()
        if site is None:
            site = 0 if len(self.sites) == 0 else max(self.sites) + 1
        elif site in self.sites:
            raise ValueError(
                "site:%s occupied, use replace/insert_tensor method" % site
            )

        T = tsr if virtual else tsr.copy()
        self.tensor_order[tid] = (T, site)
        T.set_fermion_owner(self, tid)

    def replace_tensor(self, site, tsr, tid=None, virtual=False):
        """Replace the tensor at a given site, eg
        0123456789            0123456789
        XXXXAXXXXX, (4, B) -> XXXXBXXXXX

        Parameters
        ----------
        site: int
            The position to replace the tensor
        tsr : FermionTensor
            The fermionic tensor to operate on
        tid : string, optional
            rename a new tensor id if provided
        virtual: bool, optional
            whether to replace the tensor inplace
        """
        atid = self.get_tid_from_site(site)
        atsr = self.tensor_order[atid][0]
        T = tsr if virtual else tsr.copy()
        if tid is None or (tid in self.tensor_order and tid != atid):
            tid = atid
        T.set_fermion_owner(self, tid)
        atsr.remove_fermion_owner()
        del self.tensor_order[atid]
        self.tensor_order[tid] = (T, site)

    def insert_tensor(self, site, tsr, tid=None, virtual=False):
        """insert a tensor at a given site, all tensors afterwards
        will be shifted forward by 1, eg,
        012345678            0123456789
        ABCDEFGHI, (4, X) -> ABCDXEFGHI

        Parameters
        ----------
        site: int
            The position to insert the tensor
        tsr : FermionTensor
            The fermionic tensor to operate on
        tid : string, optional
            rename a new tensor id if provided
        virtual: bool, optional
            whether to insert the tensor inplace
        """
        if (tid is None) or (tid in self.tensor_order.keys()):
            tid = self._next_tid()
        if site not in self.sites:
            self.add_tensor(tsr, tid, site=site, virtual=virtual)
        else:
            T = tsr if virtual else tsr.copy()
            T.set_fermion_owner(self, tid)
            for atid, (atsr, asite) in self.tensor_order.items():
                if asite >= site:
                    self.tensor_order.update({atid: (atsr, asite + 1)})
            self.tensor_order.update({tid: (T, site)})

    def insert(self, site, *tsrs, virtual=False):
        """insert a group of tensors at a given site, all tensors afterwards
        will be shifted forward accordingly, eg,
        0123456                0123456789
        ABCDEFG, (4, (X,Y,Z)) -> ABCDXYZEFG

        Parameters
        ----------
        site: int
            The position to begin inserting the tensor
        tsrs : a tuple/list of FermionTensor
            The fermionic tensors to operate on
        virtual: bool, optional
            whether to insert the tensors inplace
        """
        for T in tsrs:
            self.insert_tensor(site, T, virtual=virtual)
            site += 1

    def get_tid_from_site(self, site):
        """Return the tensor id at given site

        Parameters
        ----------
        site: int
            The position to obtain the tensor id
        """
        if site not in self.sites:
            raise KeyError("site:%s not occupied" % site)
        idx = self.sites.index(site)
        return list(self.tensor_order.keys())[idx]

    def get_ind_map(self):
        ind_map = dict()
        for tid, (T, _) in self.tensor_order.items():
            for ind in T.inds:
                if ind not in ind_map:
                    ind_map[ind] = oset([tid])
                else:
                    ind_map[ind] = ind_map[ind].union(oset([tid]))
        return ind_map

    def _reorder_from_dict(self, tid_map):
        """inplace reordering of tensors from a tensor_id/position mapping.
        Pizorn algorithm will be applied during moving

        Parameters
        ----------
        tid_map: dictionary
            Mapping of tensor id to the desired location
        """
        if len(tid_map) == len(self.tensor_order):
            self.reorder_all_(tid_map)
        else:
            tid_lst = list(tid_map.keys())
            des_sites = list(tid_map.values())
            # sort the destination sites to avoid cross-overs during moving
            work_des_sites = sorted(des_sites)[::-1]
            for isite in work_des_sites:
                ind = des_sites.index(isite)
                self.move(tid_lst[ind], isite)

    def reorder_all(self, tid_map, ind_map=None, inplace=False):
        """reordering of tensors from a tensor_id/position mapping. The tid_map
        must contains the mapping for all tensors in this FermionSpace.
        Pizorn algorithm will be applied during moving.

        Parameters
        ----------
        tid_map: dictionary
            Mapping of tensor id to the desired location
        ind_map: dictinary, optional
            Mapping of tensor index to the tensor id
        inplace: bool, optional
            Whether the orordering operation is inplace or not
        """
        fs = self if inplace else self.copy()
        # Compute Global Phase
        if len(tid_map) != len(fs.tensor_order):
            raise ValueError(
                "tid_map must be of equal size as the FermionSpace"
            )
        nsites = len(fs.tensor_order)
        parity_tab = []
        input_tab = []
        free_tids = []
        for isite in range(nsites):
            tid = fs.get_tid_from_site(isite)
            T = fs.tensor_order[tid][0]
            if not T.avoid_phase:
                free_tids.append(tid)
            parity_tab.append(T.parity)
            input_tab.append(tid)

        tid_lst = list(tid_map.keys())
        des_sites = list(tid_map.values())
        global_parity = 0
        for fsite in range(nsites - 1, -1, -1):
            idx = des_sites.index(fsite)
            tid = tid_lst[idx]
            isite = input_tab.index(tid)
            if isite == fsite:
                continue
            global_parity += (
                np.sum(parity_tab[isite + 1 : fsite + 1]) * parity_tab[isite]
            )
            parity_tab[isite : fsite + 1] = parity_tab[
                isite + 1 : fsite + 1
            ] + [parity_tab[isite]]
            input_tab[isite : fsite + 1] = input_tab[isite + 1 : fsite + 1] + [
                input_tab[isite]
            ]

        _global_flip = int(global_parity) % 2 == 1
        if _global_flip:
            if len(free_tids) == 0:
                raise ValueError(
                    "Global flip required on one tensor but all tensors are marked to avoid phase"
                )
            T = fs.tensor_order[free_tids[0]][0]
            T.flip_(global_flip=_global_flip)

        # Compute Local Phase
        if ind_map is None:
            ind_map = fs.get_ind_map()
        else:
            ind_map = ind_map.copy()

        local_flip_info = dict()
        for tid1, fsite1 in tid_map.items():
            T1, isite1 = fs.tensor_order[tid1]
            for ind in T1.inds:
                tids = ind_map.pop(ind, [])
                if len(tids) < 2:
                    continue
                (tid2,) = tids - oset([tid1])
                T2, isite2 = fs.tensor_order[tid2]
                fsite2 = tid_map[tid2]
                if (isite1 - isite2) * (fsite1 - fsite2) < 0:
                    if T1.avoid_phase and T2.avoid_phase:
                        raise ValueError(
                            "relative order for %s and %s changed, local phase can not be avoided"
                            % (tid1, tid2)
                        )
                    else:
                        marked_tid = tid2 if T1.avoid_phase else tid1
                    if marked_tid not in local_flip_info:
                        local_flip_info[marked_tid] = [ind]
                    else:
                        local_flip_info[marked_tid].append(ind)

        for tid, inds in local_flip_info.items():
            T = fs.tensor_order[tid][0]
            T.flip_(local_inds=inds)

        for tid, fsite in tid_map.items():
            T = fs.tensor_order[tid][0]
            fs.tensor_order.update({tid: (T, fsite)})
        return fs

    reorder_all_ = functools.partialmethod(reorder_all, inplace=True)

    def __setitem__(self, site, tsr):
        if site in self.sites:
            self.replace_tensor(site, tsr, virtual=True)
        else:
            self.add_tensor(site, tsr, virtual=True)

    def move(self, tid, des_site):
        """Move a tensor inside this FermionSpace to the specified position with Pizorn algorithm.
        Both local and global phase will be factorized to this single tensor

        Parameters
        ----------
        tid_or_site: string or int
            id or position of the original tensor
        des_site: int
            the position to move the tensor to
        """
        tsr, site = self.tensor_order[tid]
        avoid_phase = tsr.avoid_phase
        if site == des_site:
            return
        move_left = des_site < site
        iterator = (
            range(des_site, site)
            if move_left
            else range(site + 1, des_site + 1)
        )
        shared_inds = []
        tid_lst = [self.get_tid_from_site(isite) for isite in iterator]
        parity = 0
        for itid in tid_lst:
            itsr, isite = self.tensor_order[itid]
            i_shared_inds = list(oset(itsr.inds) & oset(tsr.inds))
            if avoid_phase:
                global_flip = tsr.parity * itsr.parity == 1
                if len(i_shared_inds) > 0 or global_flip:
                    if itsr.avoid_phase:
                        raise ValueError("Two tensors marked to avoid phase")
                    itsr.flip_(
                        global_flip=global_flip, local_inds=i_shared_inds
                    )
            else:
                shared_inds += i_shared_inds
                parity += itsr.parity

            if move_left:
                self.tensor_order[itid] = (itsr, isite + 1)
            else:
                self.tensor_order[itid] = (itsr, isite - 1)

        if not avoid_phase:
            global_parity = (parity % 2) * tsr.data.parity
            global_flip = global_parity == 1
            tsr.flip_(global_flip=global_flip, local_inds=shared_inds)

        self.tensor_order[tid] = (tsr, des_site)

    def move_past(self, tsr, site_range=None):
        """Move an external tensor past the specifed site ranges with Pizorn algorithm.
        Both local and global phase will be factorized to the external tensor.
        The external tensor will not be added to this FermionSpace

        Parameters
        ----------
        tsr: FermionTensor
            the external
        site_range: a tuple of integers, optional
            the range of the tensors to move past, if not specified, will be the whole space
        """
        if site_range is None:
            sites = self.sites
            site_range = (min(sites), max(sites) + 1)
        start, end = site_range
        shared_inds = []
        tid_lst = [
            self.get_tid_from_site(isite) for isite in range(start, end)
        ]
        parity = 0
        for itid in tid_lst:
            itsr, isite = self.tensor_order[itid]
            parity += itsr.parity
            shared_inds += list(oset(itsr.inds) & oset(tsr.inds))
        global_parity = (parity % 2) * tsr.data.parity
        if global_parity != 0:
            tsr.data._global_flip()
        axes = [tsr.inds.index(i) for i in shared_inds]
        if len(axes) > 0:
            tsr.data._local_flip(axes)
        return tsr

    def remove_tensor(self, site):
        """remove a specified tensor at a given site, eg
        012345               01234
        ABCDEF, (3, True) -> ABCEF
        """
        tid = self.get_tid_from_site(site)
        tsr = self.tensor_order[tid][0]
        tsr.remove_fermion_owner()
        del self.tensor_order[tid]

        indent_sites = sorted([isite for isite in self.sites if isite > site])
        tid_lst = [self.get_tid_from_site(isite) for isite in indent_sites]
        for tid in tid_lst:
            tsr, site = self.tensor_order[tid]
            self.tensor_order[tid] = (tsr, site - 1)

    @property
    def H(self):
        """Construct a FermionSpace for the bra state of the tensors"""
        max_site = max(self.sites)
        new_fs = FermionSpace()
        for tid, (tsr, site) in self.tensor_order.items():
            T = tsr.copy()
            T.modify(data=T.data.dagger, inds=T.inds[::-1])
            new_fs.add_tensor(T, tid, max_site - site, virtual=True)
        return new_fs


# --------------------------------------------------------------------------- #
#                                Tensor Funcs                                 #
# --------------------------------------------------------------------------- #
def _launch_fermion_expression(
    expr,
    tensors,
    inplace=False,
    backend="auto",
    preserve_tensor=False,
    **kwargs,
):
    if len(tensors) == 1:
        return tensors[0]
    evaluate_constants = kwargs.pop("evaluate_constants", False)
    if evaluate_constants:
        raise NotImplementedError

    if hasattr(expr, "contraction_list"):
        contraction_list = expr.contraction_list
    else:
        contraction_list = expr.fn.contraction_list
    fs, tid_lst = _dispatch_fermion_space(*tensors, inplace=inplace)
    if inplace:
        tensors = list(tensors)
    else:
        tensors = [fs.tensor_order[tid][0] for tid in tid_lst]

    operands = [Ta.data for Ta in tensors]
    global_phase = 0
    _local_inds = []
    backend = parse_backend(operands, backend)
    # Start contraction loop
    for num, contraction in enumerate(contraction_list):
        inds, idx_rm, einsum_str, _, _ = contraction
        tmp_operands = [tensors.pop(x) for x in inds]
        # Call tensordot (check if should prefer einsum, but only if available)
        input_str, results_index = einsum_str.split("->")
        input_left, input_right = input_str.split(",")
        contract_out = (oset(input_left) | oset(input_right)) - (
            oset(input_left) & oset(input_right)
        )
        if contract_out == oset(results_index):
            Ta, Tb = tmp_operands
            input_str, results_index = einsum_str.split("->")
            tid1, site1 = Ta.get_fermion_info()
            tid2, site2 = Tb.get_fermion_info()
            if site1 < site2:
                fs.move(tid2, site1 + 1)
                input_right, input_left = input_left, input_right
                Ta, Tb = Tb, Ta
            else:
                fs.move(tid1, site2 + 1)
            tensor_result = "".join(
                s for s in input_left + input_right if s not in idx_rm
            )
            # Find indices to contract over
            left_pos, right_pos = [], []
            for s in idx_rm:
                left_pos.append(input_left.find(s))
                right_pos.append(input_right.find(s))

            # Contract!
            new_view = _tensordot(
                Ta.data,
                Tb.data,
                axes=(tuple(left_pos), tuple(right_pos)),
                backend=backend,
            )

            global_phase += Ta.phase.get("global_flip", False) + Tb.phase.get(
                "global_flip", False
            )
            _local_inds.extend(Ta.phase.get("local_inds", []))
            _local_inds.extend(Tb.phase.get("local_inds", []))

            o_ix = [ind for ind in Ta.inds if ind not in Tb.inds] + [
                ind for ind in Tb.inds if ind not in Ta.inds
            ]

            # Build a new view if needed
            if tensor_result != results_index:
                transpose = tuple(map(tensor_result.index, results_index))
                new_view = _transpose(
                    new_view, axes=transpose, backend=backend
                )
                o_ix = [o_ix[ix] for ix in transpose]

            o_tags = oset.union(Ta.tags, Tb.tags)
            if len(o_ix) != 0 or preserve_tensor:
                new_view = Ta.__class__(data=new_view, inds=o_ix, tags=o_tags)
                fs.replace_tensor(min(site1, site2), new_view, virtual=True)
                fs.remove_tensor(min(site1, site2) + 1)
        # Call einsum
        else:
            raise NotImplementedError(
                "Generic Einsum Operations not supported"
            )
        # Append new items and dereference what we can
        tensors.append(new_view)
        del tmp_operands, new_view

    if isinstance(tensors[0], FermionTensor):
        local_inds = []
        for ind in tensors[0].inds:
            if _local_inds.count(ind) == 1:
                local_inds.append(ind)
        global_phase = (global_phase % 2) == 1
        tensors[0].phase = {
            "global_flip": global_phase,
            "local_inds": local_inds,
        }

    return tensors[0]


def is_mergeable(*ts_or_tsn):
    """Check if all FermionTensor or FermionTensorNetwork objects
    are part of the same FermionSpace
    """
    if len(ts_or_tsn) == 1 and isinstance(
        ts_or_tsn, (FermionTensor, FermionTensorNetwork)
    ):
        return True
    fs_lst = []
    site_lst = []
    for obj in ts_or_tsn:
        if isinstance(obj, FermionTensor):
            if obj.fermion_owner is None:
                return False
            fsobj, tid = obj.fermion_owner
            fs_lst.append(hash(fsobj))
            site_lst.append(fsobj.tensor_order[tid][1])
        elif isinstance(obj, FermionTensorNetwork):
            fs_lst.append(hash(obj.fermion_space))
            site_lst.extend(obj.filled_sites)
        else:
            raise TypeError("unable to find fermionspace")

    return all([fs == fs_lst[0] for fs in fs_lst]) and len(
        set(site_lst)
    ) == len(site_lst)


def _dispatch_fermion_space(*tensors, inplace=True):
    """Retrieve the FermionSpace and the associated tensor_ids for the tensors.
    If the given tensors all belong to the same FermionSpace object (fsobj),
    the underlying fsobj will be returned. Otherwise, a new FermionSpace will be created,
    and the tensors will be placed in the same order as the input tensors.

    Parameters
    ----------
    tensors : a tuple or list of FermionTensors
        input_tensors
    inplace: bool
        if not true, a new FermionSpace will be created with all tensors copied.
        so subsequent operations on the fsobj will not alter the input tensors.

    Returns
    -------
    fs : a FermionSpace object
    tid_lst: a list of strings for the tensor_ids
    """

    if is_mergeable(*tensors):
        if isinstance(tensors[0], FermionTensor):
            fs = tensors[0].fermion_owner[0]
        else:
            fs = tensors[0].fermion_space
        if not inplace:
            fs = fs.copy()
        tid_lst = []
        for tsr_or_tn in tensors:
            if isinstance(tsr_or_tn, FermionTensor):
                tid_lst.append(tsr_or_tn.get_fermion_info()[0])
            else:
                tid_lst.append(tsr_or_tn.tensor_map.keys())
    else:
        fs = FermionSpace()
        for tsr_or_tn in tensors[::-1]:
            if isinstance(tsr_or_tn, FermionTensor):
                fs.add_tensor(tsr_or_tn, virtual=inplace)
            elif isinstance(tsr_or_tn, FermionTensorNetwork):
                if not tsr_or_tn.is_continuous():
                    raise ValueError(
                        "Input Network not continous, merge not allowed"
                    )
                for itsr in tsr_or_tn:
                    fs.add_tensor(itsr, virtual=inplace)
        tid_lst = list(fs.tensor_order.keys())[::-1]
    return fs, tid_lst


def _split_and_replace_in_fs(T, insert_gauge=False, **compress_opts):
    compress_opts["get"] = "tensors"
    tensors = T.split(**compress_opts)
    isite = T.get_fermion_info()[1]
    fs = T.fermion_owner[0]
    fs.replace_tensor(
        isite, tensors[-1], tid=rand_uuid(base="_T"), virtual=True
    )
    if insert_gauge and len(tensors) == 3:
        fs.insert_tensor(
            isite + 1, tensors[1], tid=rand_uuid(base="_T"), virtual=True
        )
        offset = 1
    else:
        offset = 0
    fs.insert_tensor(
        isite + 1 + offset, tensors[0], tid=rand_uuid(base="_T"), virtual=True
    )
    return tensors


def _get_gauge_location(Ti, Tj):
    if Ti.get_fermion_info()[1] < Tj.get_fermion_info()[1]:
        flip_pattern = False
        return "front", "back", flip_pattern
    else:
        flip_pattern = True
        return "back", "front", flip_pattern


# --------------------------------------------------------------------------- #
#                                Tensor Class                                 #
# --------------------------------------------------------------------------- #


class FermionTensor(BlockTensor):
    __slots__ = (
        "_data",
        "_inds",
        "_tags",
        "_left_inds",
        "_owners",
        "_fermion_owner",
        "_avoid_phase",
        "_phase",
    )

    def __init__(self, data=1.0, inds=(), tags=None, left_inds=None):
        # a new or copied Tensor always has no owners
        self._fermion_owner = None
        BlockTensor.__init__(
            self, data=data, inds=inds, tags=tags, left_inds=left_inds
        )
        if isinstance(data, FermionTensor):
            if len(data.inds) != 0:
                self._data = data.data.copy()
            self._avoid_phase = data._avoid_phase
            self._phase = data._phase.copy()
        else:
            self._avoid_phase = False
            self._phase = dict()

    @property
    def avoid_phase(self):
        return self._avoid_phase

    @property
    def phase(self):
        return self._phase

    @avoid_phase.setter
    def avoid_phase(self, avoid_phase):
        self._avoid_phase = avoid_phase

    @phase.setter
    def phase(self, phase):
        self._phase = phase

    def set_phase(self, global_flip=False, local_inds=None):
        if local_inds is None:
            local_inds = []
        _global_flip = self.phase.pop("global_flip", False)
        _local_inds = self.phase.pop("local_inds", [])
        self._phase["global_flip"] = _global_flip ^ global_flip
        all_inds = tuple(local_inds) + tuple(_local_inds)
        updated_local_inds = []
        for ind in all_inds:
            count = all_inds.count(ind)
            if count % 2 == 1:
                updated_local_inds.append(ind)
        self._phase["local_inds"] = updated_local_inds

    @property
    def fermion_owner(self):
        return self._fermion_owner

    @property
    def parity(self):
        return self.data.parity

    def modify_tid(self, tid):
        if self.fermion_owner is None:
            return
        fs, old_tid = self.fermion_owner
        if old_tid == tid:
            return
        if tid in fs.tensor_order:
            raise ValueError("tid:%s is already used for another tensor" % tid)
        _, site = fs.tensor_order[old_tid]
        del fs.tensor_order[old_tid]
        fs.tensor_order[tid] = (self, site)
        self.set_fermion_owner(fs, tid)
        if self.owners:
            tn = list(self.owners.values())[0][0]()
            for ind in self.inds:
                tn.ind_map[ind] = (tn.ind_map[ind] - oset([old_tid])) | oset(
                    [tid]
                )
            for tag in self.tags:
                tn.tag_map[tag] = (tn.tag_map[tag] - oset([old_tid])) | oset(
                    [tid]
                )
            del tn.tensor_map[old_tid]
            tn.tensor_map[tid] = self

    def modify(self, **kwargs):
        if "inds" in kwargs and "data" not in kwargs:
            inds = kwargs.get("inds")
            local_inds = self.phase.pop("local_inds", [])
            new_local_inds = []
            for ind in local_inds:
                if ind in self.inds:
                    new_ind = inds[self.inds.index(ind)]
                    new_local_inds.append(new_ind)
            self._phase["local_inds"] = new_local_inds

        super().modify(**kwargs)

    def flip(self, global_flip=False, local_inds=None, inplace=False):
        T = self if inplace else self.copy()
        T.set_phase(global_flip=global_flip, local_inds=local_inds)
        if global_flip:
            T.data._global_flip()
        if local_inds is not None and len(local_inds) > 0:
            axes = [T.inds.index(ind) for ind in local_inds]
            T.data._local_flip(axes)
        return T

    flip_ = functools.partialmethod(flip, inplace=True)

    def copy(self, deep=False):
        """Copy this tensor. Note by default (``deep=False``), the underlying
        array will *not* be copied. The fermion owner will to reset to None
        """
        if deep:
            t = copy.deepcopy(self)
            t.avoid_phase = self.avoid_phase
            t.phase = self.phase.copy()
            t.remove_fermion_owner()
        else:
            t = self.__class__(self, None)
        return t

    def get_fermion_info(self):
        if self.fermion_owner is None:
            return None
        fs, tid = self.fermion_owner
        site = fs.tensor_order[tid][1]
        return (tid, site)

    @fermion_owner.setter
    def fermion_owner(self, fowner):
        self._fermion_owner = fowner

    def set_fermion_owner(self, fs, tid):
        self.fermion_owner = (fs, tid)

    def remove_fermion_owner(self):
        self.fermion_owner = None

    def __and__(self, other):
        """Combine with another ``Tensor`` or ``TensorNetwork`` into a new
        ``TensorNetwork``.
        """
        return FermionTensorNetwork((self, other))

    def __or__(self, other):
        """Combine virtually (no copies made) with another ``Tensor`` or
        ``TensorNetwork`` into a new ``TensorNetwork``.
        """
        return FermionTensorNetwork((self, other), virtual=True)


@tensor_split.register(FermionTensor)
def tensor_split_fermion_tensor(
    T,
    left_inds,
    method="svd",
    get=None,
    absorb="both",
    max_bond=None,
    cutoff=1e-10,
    cutoff_mode="rel",
    renorm=None,
    ltags=None,
    rtags=None,
    stags=None,
    bond_ind=None,
    right_inds=None,
    qpn_info=None,
):
    if get is not None:
        return tensor_split_block_tensor(
            T,
            left_inds,
            method=method,
            get=get,
            absorb=absorb,
            max_bond=max_bond,
            cutoff=cutoff,
            cutoff_mode=cutoff_mode,
            renorm=renorm,
            ltags=ltags,
            rtags=rtags,
            stags=stags,
            bond_ind=bond_ind,
            right_inds=right_inds,
            qpn_info=qpn_info,
        )
    else:
        tensors = tensor_split_block_tensor(
            T,
            left_inds,
            method=method,
            get="tensors",
            absorb=absorb,
            max_bond=max_bond,
            cutoff=cutoff,
            cutoff_mode=cutoff_mode,
            renorm=renorm,
            ltags=ltags,
            rtags=rtags,
            stags=stags,
            bond_ind=bond_ind,
            right_inds=right_inds,
            qpn_info=qpn_info,
        )
        return FermionTensorNetwork(tensors[::-1], check_collisions=False)


def compress_decorator(fn):
    @wraps(fn)
    def wrapper(T1, T2, *args, **kwargs):
        site1 = T1.get_fermion_info()[1]
        site2 = T2.get_fermion_info()[1]
        if site1 < site2:
            fn(T1, T2, *args, **kwargs)
        else:
            absorb = kwargs.pop("absorb")
            kwargs["absorb"] = {
                "left": "right",
                "right": "left",
                "both": "both",
                None: None,
            }[absorb]
            fn(T2, T1, *args, **kwargs)
        return T1, T2

    return wrapper


tensor_compress_bond_fermion = compress_decorator(
    tensor_compress_bond_block_tensor
)
tensor_compress_bond.register(FermionTensor, tensor_compress_bond_fermion)

tensor_canonize_bond_fermion = compress_decorator(
    tensor_canonize_bond_block_tensor
)
tensor_canonize_bond.register(FermionTensor, tensor_canonize_bond_fermion)


# --------------------------------------------------------------------------- #
#                            Tensor Network Class                             #
# --------------------------------------------------------------------------- #


class FermionTensorNetwork(BlockTensorNetwork):
    _EXTRA_PROPS = ()
    _CONTRACT_STRUCTURED = False

    def __init__(self, ts, *, virtual=False, check_collisions=True):
        # short-circuit for copying TensorNetworks
        if isinstance(ts, self.__class__):
            fs = FermionSpace()
            self.tag_map = valmap(lambda tids: tids.copy(), ts.tag_map)
            self.ind_map = valmap(lambda tids: tids.copy(), ts.ind_map)
            self.tensor_map = dict()
            for t in ts:
                tid = t.get_fermion_info()[0]
                t = t.copy()
                self.tensor_map[tid] = t
                self.tensor_map[tid].add_owner(self, tid)
                fs.add_tensor(t, tid=tid, virtual=True)
            self._inner_inds = ts._inner_inds.copy()
            self._outer_inds = ts._outer_inds.copy()
            self._tid_counter = ts._tid_counter
            self.exponent = ts.exponent
            for ep in ts.__class__._EXTRA_PROPS:
                setattr(self, ep, getattr(ts, ep))
            return
        else:
            BlockTensorNetwork.__init__(
                self, ts, virtual=virtual, check_collisions=True
            )

    @property
    def fermion_space(self):
        if len(self.tensor_map) == 0:
            return FermionSpace()
        else:
            return list(self.tensor_map.values())[0].fermion_owner[0]

    @property
    def filled_sites(self):
        return [
            self.fermion_space.tensor_order[tid][1]
            for tid in self.tensor_map.keys()
        ]

    @property
    def H(self):
        tn = self.copy(full=True)
        fs = tn.fermion_space
        max_site = max(fs.sites)
        for tid, (T, site) in fs.tensor_order.items():
            T.modify(data=T.data.dagger, inds=T.inds[::-1])
            fs.tensor_order.update({tid: (T, max_site - site)})
        return tn

    def is_continuous(self):
        """
        Check if sites in the current tensor network are contiguously occupied
        """
        filled_sites = self.filled_sites
        if len(filled_sites) == 0:
            return True
        return (max(filled_sites) - min(filled_sites) + 1) == len(filled_sites)

    def _remove_phase_from_tids(self, tids):
        """
        remove phase information on specified tensors
        """
        tids = tags_to_oset(tids)
        for tid in tids:
            self.tensor_map[tid].phase = dict()

    def _remove_phase_from_tags(self, tags, which="all"):
        tagged_tids = self._get_tids_from_tags(tags, which=which)
        return self._remove_phase_from_tids(tagged_tids)

    def _reorder_tids_like(self, tids, like):
        ntensors = len(self.fermion_space.tensor_order)
        ref_order = dict()
        for tid in tids:
            ref_order[tid] = like.tensor_map[tid].get_fermion_info()[1]
        sort_order = sorted(ref_order, key=lambda k: ref_order[k])
        order_map = dict(
            zip(sort_order, range(ntensors - len(tids), ntensors))
        )
        self._reorder_from_tid(order_map, inplace=True)

    def _split_tensor_tid(self, tid, left_inds, **split_opts):
        t = self._pop_tensor(tid)
        tensors = t.split(left_inds=left_inds, get="tensors", **split_opts)
        fs = self.fermion_space
        site = t.get_fermion_info()[1]
        for i, T in enumerate(tensors[::-1]):
            if i == 0:
                fs.replace_tensor(site, T, virtual=True)
            else:
                fs.insert_tensor(site, T, virtual=True)
            site += 1
            self.add_tensor(T, virtual=True)
        return tensors

    def _refactor_phase_from_tids(self, tids):
        tids = tags_to_oset(tids)
        local_inds = []
        global_flip = 0
        for tid in tids:
            local_inds.extend(self.tensor_map[tid].phase.get("local_inds", []))
            global_flip += self.tensor_map[tid].phase.get("global_flip", False)
        global_flip = (global_flip % 2) == 1
        linked_inds_map = dict()
        for ind in local_inds:
            linked_tids = self.ind_map[ind] - tids
            if linked_tids:
                (output_tid,) = linked_tids
                linked_inds_map[ind] = output_tid
            else:
                raise ValueError(
                    """ can't refactor the local phase on bond %s, either due to:
                        1. The bond has an open indices
                        2. The order of the two tensors sharing
                            this bond needs to be reorderred"""
                    % ind
                )
                return

        for ind, otid in linked_inds_map.items():
            To = self.tensor_map[otid]
            To.flip_(local_inds=(ind,), global_flip=global_flip)
            global_flip = False

        for itid in tids:
            Ti = self.tensor_map[itid]
            Ti.flip_(**Ti.phase)
        if global_flip:
            other_tids = tags_to_oset(self.tensor_map.keys()) - tids
            gtid = list(other_tids)[0]
            self.tensor_map[gtid].flip_(global_flip=True)

    def copy(self, full=False, force=False):
        """For full copy, the tensors and underlying FermionSpace(all tensors in it) will
        be copied. For partial copy, the tensors in this network must be continuously
        placed and a new FermionSpace will be created to hold this continous sector.
        """
        if full:
            fs = self.fermion_space.copy()
            tids = list(self.tensor_map.keys())
            tsr = [fs.tensor_order[tid][0] for tid in tids]
            newtn = FermionTensorNetwork(tsr, virtual=True)
        else:
            if not self.is_continuous() and not force:
                raise TypeError(
                    "Tensors not continuously placed in the network, \
                                partial copy not allowed"
                )
            else:
                newtn = FermionTensorNetwork(self)
        newtn.view_like_(self)
        return newtn

    def __and__(self, other):
        """Combine this tensor network with more tensors, without contracting.
        Copies the tensors.
        """
        if is_mergeable(self, other):
            raise ValueError(
                "the two networks are in the same fermionspace, use self |= other"
            )
        return FermionTensorNetwork((self, other), virtual=False)

    def __or__(self, other):
        """Combine this tensor network with more tensors, without contracting.
        Views the constituent tensors.
        """
        return FermionTensorNetwork((self, other), virtual=True)

    def __iter__(self):
        sorted_sites = sorted(self.filled_sites)
        for isite in sorted_sites:
            tid = self.fermion_space.get_tid_from_site(isite)
            yield self.tensor_map[tid]

    @property
    def tensors(self):
        return tuple([T for T in self])

    def __setitem__(self, tags, tensor):
        """Set the single tensor uniquely associated with ``tags``."""
        tids = self._get_tids_from_tags(tags, which="all")
        if len(tids) != 1:
            raise KeyError(
                "'TensorNetwork.__setitem__' is meant for a single "
                "existing tensor only - found {} with tag(s) '{}'.".format(
                    len(tids), tags
                )
            )

        if not isinstance(tensor, FermionTensor):
            raise TypeError("Can only set value with a new 'FermionTensor'.")

        (tid,) = tids
        site = self.fermion_space.tensor_order[tid][1]
        super()._pop_tensor(tid)
        super().add_tensor(tensor, tid=tid, virtual=True)
        self.fermion_space.replace_tensor(site, tensor, tid=tid, virtual=True)

    def _reorder_from_tid(self, tid_map, inplace=False):
        tn = self if inplace else self.copy(full=True)
        tn.fermion_space._reorder_from_dict(tid_map)
        return tn

    def _select_tids(self, tids, virtual=True):
        """Get a copy or a virtual copy (doesn't copy the tensors) of this
        ``FermionTensorNetwork``, only with the tensors corresponding to ``tids``.
        """
        tn = FermionTensorNetwork(())
        if not virtual:
            # make sure the relative order is consistent as original network
            order_map = dict()
            for tid in tids:
                order_map[tid] = self.tensor_map[tid].get_fermion_info()[1]
            tids = sorted(order_map, key=lambda x: order_map[x])
        for tid in tids:
            tn.add_tensor(self.tensor_map[tid], tid=tid, virtual=virtual)
        tn.view_like_(self)
        return tn

    def add_tensor(self, tsr, tid=None, virtual=False):
        T = tsr if virtual else tsr.copy()
        fs = T.fermion_owner
        if fs is None:
            self.fermion_space.add_tensor(T, tid, virtual=True)
        else:
            if (
                hash(fs[0]) != hash(self.fermion_space)
                and len(self.tensor_map) > 0
            ):
                raise ValueError(
                    "The tensor is not compatible with the current network"
                )
        tid = T.get_fermion_info()[0]
        super().add_tensor(T, tid, virtual=True)

    def add_tensor_network(self, tn, virtual=False, check_collisions=True):
        if virtual:
            if min(len(self.tensor_map), len(tn.tensor_map)) == 0:
                super().add_tensor_network(
                    tn, virtual=virtual, check_collisions=check_collisions
                )
                return
            elif hash(tn.fermion_space) == hash(self.fermion_space):
                if is_mergeable(self, tn):
                    super().add_tensor_network(
                        tn, virtual=True, check_collisions=check_collisions
                    )
                else:
                    raise ValueError(
                        "the two tensornetworks co-share same sites, inplace addition not allowed"
                    )
                return

        if not tn.is_continuous():
            raise ValueError(
                "input tensor network is not contiguously ordered"
            )

        tn = tn if virtual else tn.copy()
        sorted_tensors = []
        for tsr in tn:
            tid = tsr.get_fermion_info()[0]
            sorted_tensors.append([tid, tsr])

        if check_collisions:  # add tensors individually
            # check for matching inner_indices -> need to re-index
            clash_ix = self._inner_inds & tn._inner_inds
            reind = {ix: rand_uuid() for ix in clash_ix}
        else:
            clash_ix = False
            reind = None

        # add tensors, reindexing if necessary
        for tid, tsr in sorted_tensors:
            tsr.remove_fermion_owner()
            if clash_ix and any(i in reind for i in tsr.inds):
                tsr.reindex_(reind)
            self.add_tensor(tsr, virtual=True, tid=tid)

        self.exponent = self.exponent + tn.exponent

    def partition(self, tags, which="any", inplace=False):
        """Split this FTN into two, based on which tensors have any or all of
        ``tags``. Unlike ``partition_tensors``, both results are FTNs which
        inherit the structure of the initial FTN and are still linked to the
        same FermionSpace

        Parameters
        ----------
        tags : sequence of str
            The tags to split the network with.
        which : {'any', 'all'}
            Whether to split based on matching any or all of the tags.
        inplace : bool
            If True, actually remove the tagged tensors from self.

        Returns
        -------
        untagged_tn, tagged_tn : (FermionTensorNetwork, FermionTensorNetwork)
            The untagged and tagged tensor networs.

        See Also
        --------
        partition_tensors, select, select_tensors
        """
        t1 = self if inplace else self.copy(full=True)
        tagged_tids = t1._get_tids_from_tags(tags, which=which)
        t2s = [t1._pop_tensor(tid) for tid in tagged_tids]
        kws = {"check_collisions": False, "virtual": True}
        t2 = FermionTensorNetwork(t2s, **kws)
        t2.view_like_(self)
        return t1, t2

    def partition_tensors(self, tags, inplace=False, which="any"):
        """Split this TN into a list of tensors containing any or all of
        ``tags`` and a ``FermionTensorNetwork`` of the the rest. All
        FermionTensor and FermionTensorNetwork are still linked to the
        same FermionSpace

        Parameters
        ----------
        tags : sequence of str
            The list of tags to filter the tensors by. Use ``...``
            (``Ellipsis``) to filter all.
        inplace : bool, optional
            If true, remove tagged tensors from self, else create a new network
            with the tensors removed.
        which : {'all', 'any'}
            Whether to require matching all or any of the tags.

        Returns
        -------
        (u_tn, t_ts) : (FermionTensorNetwork, tuple of FermionTensors)
            The untagged tensor network, and the sequence of tagged FermionTensors.

        See Also
        --------
        partition, select, select_tensors
        """
        tn = self if inplace else self.copy(full=True)
        return TensorNetwork.partition_tensors(
            tn, tags, inplace=True, which=which
        )

    def _pop_tensor(self, tid, remove_from_fermion_space=False):
        """Remove a tensor from this network, returning said tensor.

        Parameters
        ----------
        tid: str
            tensor identifier
        remove_from_fermion_space: optional, {True, False, 'front', 'end'}
            remove methods:
                - True   : remove tensor from both TN and FermionSpace,
                           keep phase as it is
                - False  : remove tensor from TN,
                           but still linked to fermion space,
                           no change on phase
                - 'front': move tensor to location 0 in FermionSpace,
                           factorize all phase on remaining tensors,
                           then remove tensor from both TN and FermionSpace
                - 'end'  : move tensor to last position in FermionSpace,
                           factorize all phase on remaining tensors,
                           then remove tensor from both TN and FermionSpace
            'front'/'end' can potential fail if the tensor has phase on open indices
        """
        check_opt(
            "remove_from_fermion_space",
            remove_from_fermion_space,
            (True, False, "front", "end"),
        )
        if remove_from_fermion_space:
            t = self.tensor_map[tid]
            tid, site = t.get_fermion_info()
            if remove_from_fermion_space == "front":
                site = 0
                self.fermion_space.move(tid, site)
                self._refactor_phase_from_tids([tid])
            elif remove_from_fermion_space == "end":
                site = len(self.fermion_space.tensor_order) - 1
                self.fermion_space.move(tid, site)
                self._refactor_phase_from_tids([tid])
            self.fermion_space.remove_tensor(site)
        return TensorNetwork._pop_tensor(self, tid)

    def _contract_between_tids(self, tid1, tid2, **contract_opts):
        contract_opts["inplace"] = True
        super()._contract_between_tids(tid1, tid2, **contract_opts)

    def contract_between(self, tags1, tags2, **contract_opts):
        contract_opts["inplace"] = True
        super().contract_between(tags1, tags2, **contract_opts)

    def contract_ind(self, ind, **contract_opts):
        """Contract tensors connected by ``ind``."""
        contract_opts["inplace"] = True
        super().contract_ind(ind, **contract_opts)

    # ----------------------- contracting the network ----------------------- #
    def contract_tags(self, tags, inplace=False, which="any", **opts):
        untagged_tn, tagged_ts = self.partition_tensors(
            tags, inplace=inplace, which=which
        )

        contracting_all = untagged_tn is None
        if not tagged_ts:
            raise ValueError(
                "No tags were found - nothing to contract. "
                "(Change this to a no-op maybe?)"
            )
        opts["inplace"] = True
        opts.setdefault("preserve_tensor", not contracting_all)
        contracted = tensor_contract(*tagged_ts, **opts)

        if contracting_all:
            return contracted

        untagged_tn.add_tensor(contracted, virtual=True)
        return untagged_tn

    def gate_inds(
        self,
        G,
        inds,
        contract=False,
        tags=None,
        info=None,
        inplace=False,
        **compress_opts,
    ):
        check_opt("contract", contract, (False, True, "split", "reduce-split"))

        tn = self if inplace else self.copy()

        if isinstance(inds, str):
            inds = (inds,)
        ng = len(inds)

        # new indices to join old physical sites to new gate
        bnds = tuple([rand_uuid() for _ in range(ng)])
        reindex_map = dict(zip(inds, bnds))

        # tensor representing the gate
        tags = tags_to_oset(tags)
        tG = FermionTensor(
            G.copy(), inds=inds + bnds, tags=tags, left_inds=bnds
        )
        fs = tn.fermion_space

        if contract is False:
            #
            #       │   │      <- site_ix
            #       GGGGG
            #       │╱  │╱     <- bnds
            #     ──●───●──
            #      ╱   ╱
            #
            tn.reindex_(reindex_map)
            tn |= tG
            return tn

        tids = self._get_tids_from_inds(inds, "any")

        fs.add_tensor(tG, virtual=True)

        if (contract is True) or (len(tids) == 1):
            #
            #       │╱  │╱
            #     ──GGGGG──
            #      ╱   ╱
            #
            tn.reindex_(reindex_map)

            # get the sites that used to have the physical indices
            site_tids = tn._get_tids_from_inds(bnds, which="any")

            # pop the sites, contract, then re-add
            pts = [tn._pop_tensor(tid) for tid in site_tids]
            tn |= tensor_contract(*pts, tG, inplace=True)
            return tn

        # get the two tensors and their current shared indices etc.
        ixl, ixr = inds
        tl, tr = tn._inds_get(ixl, ixr)
        if tl.get_fermion_info()[1] > tr.get_fermion_info()[1]:
            ixl, ixr = ixr, ixl
            tl, tr = tr, tl

        bnds_l, (bix,), bnds_r = group_inds(tl, tr)

        tidl, sitel = tl.get_fermion_info()
        tidr, siter = tr.get_fermion_info()
        fermion_info = {tidl: sitel, tidr: siter}

        if contract == "split":
            #
            #       │╱  │╱         │╱  │╱
            #     ──GGGGG──  ->  ──G~~~G──
            #      ╱   ╱          ╱   ╱
            #

            # contract with new gate tensor
            tlGr = tensor_contract(
                tl.reindex_(reindex_map),
                tr.reindex_(reindex_map),
                tG,
                inplace=True,
            )

            tlGr.modify_tid(tidl)
            isite = tlGr.get_fermion_info()[1]
            # decompose back into two tensors
            qpn_info = (tr.net_symmetry, tl.net_symmetry)
            trn, *maybe_svals, tln = tlGr.split(
                left_inds=bnds_r,
                right_inds=bnds_l,
                bond_ind=bix,
                get="tensors",
                qpn_info=qpn_info,
                **compress_opts,
            )

            fs.replace_tensor(isite, tl, virtual=True)
            fs.insert_tensor(isite + 1, tr, tid=tidr, virtual=True)
            revert_index_map = dict(zip(bnds, inds))
            tl.reindex_(revert_index_map)
            tr.reindex_(revert_index_map)

        if contract == "reduce-split":
            # move physical inds on reduced tensors
            #
            #       │   │             │ │
            #       GGGGG             GGG
            #       │╱  │╱   ->     ╱ │ │   ╱
            #     ──●───●──      ──>──●─●──<──
            #      ╱   ╱          ╱       ╱
            #

            tmp_bix_l = rand_uuid()
            tl_Q, tl_R = _split_and_replace_in_fs(
                tl,
                left_inds=None,
                right_inds=[bix, ixl],
                method="qr",
                bond_ind=tmp_bix_l,
                absorb="right",
            )

            tmp_bix_r = rand_uuid()
            tr_L, tr_Q = _split_and_replace_in_fs(
                tr,
                left_inds=[bix, ixr],
                right_inds=None,
                method="qr",
                bond_ind=tmp_bix_r,
                absorb="left",
            )

            # contract reduced tensors with gate tensor
            #
            #          │ │
            #          GGG                │ │
            #        ╱ │ │   ╱    ->    ╱ │ │   ╱
            #     ──>──●─●──<──      ──>──LGR──<──
            #      ╱       ╱          ╱       ╱
            #

            tlGr = tensor_contract(
                tl_R.reindex_(reindex_map),
                tr_L.reindex_(reindex_map),
                tG,
                inplace=True,
            )

            # split to find new reduced factors
            #
            #          │ │                │ │
            #        ╱ │ │   ╱    ->    ╱ │ │   ╱
            #     ──>──LGR──<──      ──>──L=R──<──
            #      ╱       ╱          ╱       ╱
            #
            tr_L, *maybe_svals, tl_R = _split_and_replace_in_fs(
                tlGr,
                left_inds=[tmp_bix_r, ixr],
                right_inds=[tmp_bix_l, ixl],
                bond_ind=bix,
                get="tensors",
                **compress_opts,
            )

            # absorb reduced factors back into site tensors
            #
            #          │ │             │   │
            #        ╱ │ │   ╱         │╱  │╱
            #     ──>──L=R──<──  ->  ──●───●──
            #      ╱       ╱          ╱   ╱
            tln = tensor_contract(tl_Q, tl_R, inplace=True)
            trn = tensor_contract(tr_L, tr_Q, inplace=True)
            lsite = tln.get_fermion_info()[1]
            rsite = trn.get_fermion_info()[1]
            fs.replace_tensor(lsite, tl, tid=tidl, virtual=True)
            fs.replace_tensor(rsite, tr, tid=tidr, virtual=True)

        # if singular values are returned (``absorb=None``) check if we should
        #     return them via ``info``, e.g. for ``SimpleUpdate`
        if maybe_svals and info is not None:
            s = next(iter(maybe_svals)).data
            info["singular_values", (ixl, ixr)] = s

        # update original tensors
        tl.modify(data=tln.transpose_like_(tl).data)
        tr.modify(data=trn.transpose_like_(tr).data)
        # move the tensors to the original locations
        fs._reorder_from_dict(fermion_info)
        return tn

    gate_inds_ = functools.partialmethod(gate_inds, inplace=True)

    def make_norm(
        self,
        mangle_append="*",
        layer_tags=("KET", "BRA"),
        return_all=False,
    ):
        ket = self.copy()
        if len(ket.outer_inds()) == 0:
            return ket
        ket.add_tag(layer_tags[0])

        bra = ket.retag({layer_tags[0]: layer_tags[1]})
        bra = bra.H
        if mangle_append:
            bra.mangle_inner_(mangle_append)
        norm = ket & bra

        if return_all:
            return norm, ket, bra
        return norm

    def gauge_simple_insert(self, gauges):
        # absorb outer gauges fully into single tensor
        outer = []
        inner = []

        if len(self.tensor_map) == len(self.fermion_space.tensor_order):
            full_ind_map = self.ind_map
        else:
            full_ind_map = self.fermion_space.get_ind_map()

        for (ix, iy), g in gauges.items():
            tensors = list(self._inds_get(ix, iy))
            if len(tensors) == 2:
                (tl,) = self._inds_get(ix)
                (tr,) = self._inds_get(iy)
                locl, locr, flip_pattern = _get_gauge_location(tl, tr)
                (bond,) = tl.bonds(tr)
                g = sqrt(g)
                tl.multiply_index_diagonal_(
                    bond, g, location=locl, flip_pattern=flip_pattern
                )
                tr.multiply_index_diagonal_(
                    bond, g, location=locr, flip_pattern=flip_pattern
                )
                inner.append(((tl, tr), bond, g, (locl, locr), flip_pattern))
            elif len(tensors) == 1:
                (tl,) = tensors
                (itid,) = (
                    full_ind_map[iy] if ix in tl.inds else full_ind_map[ix]
                )
                tr = self.fermion_space.tensor_order[itid][0]
                (bond,) = tl.bonds(tr)
                if ix in self.ind_map:
                    locl, _, flip_pattern = _get_gauge_location(tl, tr)
                else:
                    _, locl, flip_pattern = _get_gauge_location(tr, tl)
                tl.multiply_index_diagonal_(
                    bond, g, location=locl, flip_pattern=flip_pattern
                )
                outer.append((tl, bond, g, locl, flip_pattern))
        return outer, inner

    @contextlib.contextmanager
    def gauge_simple_temp(
        self,
        gauges,
        ungauge_outer=True,
        ungauge_inner=True,
    ):
        outer, inner = self.gauge_simple_insert(gauges)
        try:
            yield outer, inner
        finally:
            while ungauge_outer and outer:
                t, ix, g, location, flip_pattern = outer.pop()
                g = inv_with_smudge(g, gauge_smudge=0.0)
                t.multiply_index_diagonal_(
                    ix, g, location=location, flip_pattern=flip_pattern
                )
            while ungauge_inner and inner:
                (tl, tr), ix, g, location, flip_pattern = inner.pop()
                ginv = inv_with_smudge(g, gauge_smudge=0.0)
                tl.multiply_index_diagonal_(
                    ix, ginv, location=location[0], flip_pattern=flip_pattern
                )
                tr.multiply_index_diagonal_(
                    ix, ginv, location=location[1], flip_pattern=flip_pattern
                )

    def __matmul__(self, other):
        """Overload "@" to mean full contraction with another network."""
        return FermionTensorNetwork((self, other)) ^ ...


def _tensors_to_constructors(tensors, inds, inv=True):
    """
    Generate a pyblock3.algebra.fermion.Constructor object
    to allow mapping from vector to tensor and inverse.

    Parameters
    ----------
    tensors: a list/tuple of FermionTensors
        The tensors to gather symmetry information from
    inds: a list/tuple of strings
        The indices of the tensor to construct
    inv: a string of "+" and "-"
        Whether to take the complementary signs
        from the tensor input

    Returns
    -------
    constructor: a pyblock3.algebra.fermion.Constructor object
    """
    string_inv = {"+": "-", "-": "+"}
    pattern = [
        None,
    ] * len(inds)
    bond_infos = [
        None,
    ] * len(inds)
    count = 0
    for T in tensors:
        for ix, ind in enumerate(inds):
            if ind in T.inds:
                ax = T.inds.index(ind)
                bond_infos[ix] = T.data.get_bond_info(ax, flip=False)
                if inv:
                    pattern[ix] = string_inv[T.data.pattern[ax]]
                else:
                    pattern[ix] = T.data.pattern[ax]
                count += 1
        if count == len(inds):
            break
    pattern = "".join(pattern)
    mycon = Constructor.from_bond_infos(bond_infos, pattern)
    return mycon


class FTNLinearOperator(spla.LinearOperator):
    r"""Get a fermionic linear operator - something that replicates the matrix-vector
    operation - for an arbitrary uncontracted  FermionTensorNetwork, e.g::

                 : --O--O--+ +-- :                 --+
                 :   |     | |   :                   |
                 : --O--O--O-O-- :    acting on    --V
                 :   |     |     :                   |
                 : --+     +---- :                 --+
        left_inds^               ^right_inds

    This can then be supplied to scipy's sparse linear algebra routines.
    The ``left_inds`` / ``right_inds`` convention is that the linear operator
    will have shape matching ``(*left_inds, *right_inds)``, so that the
    ``right_inds`` are those that will be contracted in a normal
    matvec / matmat operation::

        _matvec =    --0--v    , _rmatvec =     v--0--
    Note prior to constructing the TNLinearOperator, reordering is needed to move the
    ket site to location 0 and bra site to the last position.

    Parameters
    ----------
    tns : sequence of FermionTensors or FermionTensorNetwork
        A representation of the hamiltonian. If it's a sequence
        of fermionTensors, they must be in the same FermionSpace
    left_inds : sequence of str
        The 'left' inds of the effective hamiltonian network. Usually the
        indices of bra tensor in reverse order (to be consisitent with ket)
    right_inds : sequence of str
        The 'right' inds of the effective hamiltonian network.
        Usually the indices of ket tensor
    target_symmetry: symmetry object in pyblock3.algebra.fermion_symmetry
        The target total symmetry on the right vector
    right_constructor: pyblock3.algebra.fermion.Constructor object, optional
        An object to help map the right vector to a FermionTensor data
    square: bool, optional
        Whether the operator is expected to have same symmetry blocks in
        left/right indices
    optimize : str, optional
        The path optimizer to use for the 'matrix-vector' contraction.
    backend : str, optional
        The array backend to use for the 'matrix-vector' contraction.
    is_conj : bool, optional
        Whether this object should represent the *adjoint* operator.
    location: string, optional
        The relative ordering of the vector with respect to the operator

    See Also
    --------
    TNLinearOperator
    """

    def __init__(
        self,
        tns,
        left_inds,
        right_inds,
        target_symmetry,
        constructor=None,
        optimize="auto",
        backend=None,
        is_conj=False,
        location="back",
    ):
        if backend is None:
            self.backend = get_tensor_linop_backend()
        else:
            self.backend = backend
        self.optimize = optimize
        self.location = location
        self._dq = target_symmetry

        if isinstance(tns, FermionTensorNetwork):
            self._tensors = tns.tensors
        else:
            self._tensors = tuple(tns)

        shape_dict = dict()
        for T in self._tensors:
            shape_dict.update(zip(T.inds, T.shape))
        ish = tuple([shape_dict[ix] for ix in right_inds])
        self._shape = ish

        if constructor is None:
            self.constructor = _tensors_to_constructors(
                self._tensors, right_inds
            )
        else:
            self.constructor = constructor

        self.left_inds, self.right_inds = left_inds, right_inds
        self.tags = oset.union(*(t.tags for t in self._tensors))

        self._kws = {"get": "expression"}

        # if recent opt_einsum specify constant tensors
        if hasattr(oe.backends, "evaluate_constants"):
            self._kws["constants"] = range(len(self._tensors))

        # conjugate inputs/ouputs rather all tensors if necessary
        self.is_conj = is_conj
        self._conj_linop = None
        self._adjoint_linop = None
        self._transpose_linop = None
        self._contractors = dict()

    @property
    def dtype(self):
        return self._tensors[0].dtype

    @property
    def dq(self):
        return self._dq

    @dq.setter
    def dq(self, new_dq):
        self._dq = new_dq

    @property
    def shape(self):
        dim = self.constructor.vector_size(self.dq)
        return (dim, dim)

    def vector_to_tensor(self, vector, dq=None):
        if dq is None:
            dq = self.dq
        tensor = self.constructor.vector_to_tensor(vector, dq)
        tensor.shape = self._shape
        return tensor

    def tensor_to_vector(self, T):
        return self.constructor.tensor_to_vector(T)

    def get_contraction_kits(self):
        fs, tid_lst = _dispatch_fermion_space(*self._tensors, inplace=False)
        tensors = [fs.tensor_order[tid][0] for tid in tid_lst]
        return fs, tensors

    def _matvec(self, vec):
        if self.is_conj:
            vec = conj(vec)
        in_data = self.vector_to_tensor(vec)
        iT = FermionTensor(in_data, inds=self.right_inds)
        fs, tensors = self.get_contraction_kits()
        tensors.append(iT)
        if self.location == "back":
            fs.insert_tensor(0, iT, virtual=True)
        else:
            fs.add_tensor(iT, virtual=True)

        # cache the contractor
        if "matvec" not in self._contractors:
            # generate a expression that acts directly on the data
            self._contractors["matvec"] = tensor_contract(
                *tensors,
                output_inds=self.left_inds,
                optimize=self.optimize,
                **self._kws,
            )

        expr = self._contractors["matvec"]
        out_data = _launch_fermion_expression(
            expr, tensors, backend=self.backend, inplace=True
        )
        out_data = out_data.data
        if self.is_conj:
            out_data = conj(out_data)
        return self.tensor_to_vector(out_data)

    def copy(self, conj=False, transpose=False):
        if transpose:
            left_inds, right_inds = self.right_inds, self.left_inds
            target_symmetry = -self.dq
            location = {"back": "front", "front": "back"}[self.location]
        else:
            left_inds, right_inds = self.left_inds, self.right_inds
            target_symmetry = self.dq
            location = self.location

        if conj:
            is_conj = not self.is_conj
        else:
            is_conj = self.is_conj

        return FTNLinearOperator(
            self._tensors,
            left_inds,
            right_inds,
            target_symmetry,
            optimize=self.optimize,
            backend=self.backend,
            is_conj=is_conj,
            location=location,
        )

    def _adjoint(self):
        if self._adjoint_linop is None:
            self._adjoint_linop = self.copy(conj=True, transpose=True)
        return self._adjoint_linop

    def _transpose(self):
        if self._transpose_linop is None:
            self._transpose_linop = self.copy(transpose=True)
        return self._transpose_linop
