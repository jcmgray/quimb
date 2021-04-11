"""Core Fermionic TensorNetwork Module
Note: The position of Fermionic Tensors inside FermionSpace
    is defined as the its distance to the ket vacuum, eg,
    for |psi> = \hat{Tx} \hat{Ty} \hat{Tz} |0>,
    we have the position for these tensors as
    Tx:2     Ty:1     Tz:0
"""
import numpy as np
import weakref
import functools
from .tensor_core import (Tensor, TensorNetwork, rand_uuid, tags_to_oset,
                          _parse_split_opts, check_opt, _VALID_SPLIT_GET)
from .tensor_core import tensor_contract as _tensor_contract
from ..utils import oset, valmap
from .array_ops import asarray, ndim
from . import fermion_interface

DEFAULT_SYMMETRY = fermion_interface.DEFAULT_SYMMETRY
BondInfo = fermion_interface.BondInfo

def _contract_connected(T1, T2, output_inds=None):
    """Fermionic contraction of two tensors that are adjacent to each other.
    Any shared indexes will be summed over. If the input fermionic tensors
    do not belong to the same FermionSpace, the first tensor is assumed to
    placed after the second tensor, eg \hat{T1} \hat{T2}

    Parameters
    ----------
    T1 : FermionTensor
        The first tensor.
    T2 : FermionTensor
        The second tensor, with matching indices and dimensions to ``T1``.
    output_inds : sequence of str
        If given, the desired order of output indices, else defaults to the
        order they occur in the input indices.

    Returns
    -------
    scalar or FermionTensor
    """
    info1 = T1.get_fermion_info()
    info2 = T2.get_fermion_info()
    t1, t2 = T1, T2
    if info1 is not None and info2 is not None:
        site1, site2 = info1[1], info2[1]
        if abs(site1-site2) != 1:
            raise ValueError("T1 and T2 not adjacently connected in FermionSpace")
        if site1 < site2:
            # if T1 is placed before T2,
            # it shall be parsed as second input to tensordot backend
            t1, t2 = T2, T1
    ainds, binds = t1.inds, t2.inds
    _output_inds = []
    ax_a, ax_b = [], []
    for kia, ia in enumerate(ainds):
        if ia not in binds:
            _output_inds.append(ia)
        else:
            ax_a.append(kia)
            ax_b.append(binds.index(ia))
    for kib, ib in enumerate(binds):
        if ib not in ainds:
            _output_inds.append(ib)
    if output_inds is None: output_inds = _output_inds
    if set(_output_inds) != set(output_inds):
        raise TypeError("specified out_inds not allowed in tensordot, \
                         make sure no summation/Hadamard product appears")

    out = np.tensordot(t1.data, t2.data, axes=[ax_a, ax_b])

    if len(output_inds)==0:
        return out.data[0]

    if output_inds!=_output_inds:
        transpose_order = tuple([_output_inds.index(ia) for ia in output_inds])
        out = out.transpose(transpose_order)
    o_tags = oset.union(*(T1.tags, T2.tags))
    out = FermionTensor(out, inds=output_inds, tags=o_tags)
    return out

def _contract_pairs(fs, tid_or_site1, tid_or_site2, output_inds=None, direction='left'):
    """ Perform pairwise contraction for two tensors in a specified fermion space.
    If the two tensors are not adjacent, move one of the tensors in the given direction.
    Note this could alter the tensors that are in between the two tensors in the fermion space

    Parameters
    ----------
    fs : FermionSpace obj
        the FermionSpace obj that contains the two tensors
    tid_or_site1: a string or an integer
        The string that specifies the id for the first tensor or the site for the first tensor
    tid_or_site2: a string or an integer
        The string that specifies the id for the 2nd tensor or the site for the 2nd tensor
    output_inds: a list of strings
        The list that specifies the output indices and its order
    direction: string "left" or "right"
        The direction to move tensors if the two tensors are not adjacent

    Returns
    -------
    scalar or FermionTensor
    """
    tid1, site1, tsr1 = fs[tid_or_site1]
    tid2, site2, tsr2 = fs[tid_or_site2]

    if not fs.is_adjacent(tid1, tid2):
        fs.make_adjacent(tid1, tid2, direction)

    if direction=="left":
        site1 = min(site1, site2)
    else:
        site1 = max(site1, site2) - 1

    site2 = site1 + 1
    return _contract_connected(tsr1, tsr2, output_inds)

def _fetch_fermion_space(*tensors, inplace=True):
    """ Retrieve the FermionSpace and the associated tensor_ids for the tensors.
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
    if isinstance(tensors, (FermionTensor, FermionTensorNetwork)):
        tensors = (tensors, )

    if is_mergeable(*tensors):
        if isinstance(tensors[0], FermionTensor):
            fs = tensors[0].fermion_owner[1]()
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
        for tsr_or_tn in tensors:
            if isinstance(tsr_or_tn, FermionTensor):
                fs.add_tensor(tsr_or_tn, virtual=inplace)
            elif isinstance(tsr_or_tn, FermionTensorNetwork):
                if not tsr_or_tn.is_continuous():
                    raise ValueError("Input Network not continous, merge not allowed")
                for itsr in tsr_or_tn:
                    fs.add_tensor(itsr, virtual=inplace)
        tid_lst = list(fs.tensor_order.keys())
    return fs, tid_lst

def tensor_contract(*tensors, output_inds=None,
                    direction="left", inplace=False, **contract_opts):
    """ Perform tensor contractions for all given tensors.
    If input tensors do not belong to the same underlying fsobj,
    the position of each tensor will be the same as its order in the input tensor tuple/list.
    Summation and Hadamard product not supported as it's not well defined for fermionic tensors

    Parameters
    ----------
    tensors : a tuple or list of FermionTensors
        input tensors
    output_inds: a list of strings
    direction: string "left" or "right"
        The direction to move tensors if the two tensors are not adjacent
    inplace: bool, optional
        whether to move/contract tensors in place.

    Returns
    -------
    out : a FermionTensor object or a number
    """
    path_info = _tensor_contract(*tensors, get='path-info', **contract_opts)
    fs, tid_lst = _fetch_fermion_space(*tensors, inplace=inplace)
    for conc in path_info.contraction_list:
        pos1, pos2 = conc[0]
        tid1 = tid_lst.pop(pos1)
        tid2 = tid_lst.pop(pos2)
        site1 = fs[tid1][1]
        site2 = fs[tid2][1]
        out = fs._contract_pairs(site1, site2, direction=direction, inplace=True)
        if not isinstance(out, (float, complex)):
            tid_lst.append(out.fermion_owner[2])

    if not isinstance(out, (float, complex)):
        _output_inds = out.inds
        if output_inds is None:
            output_inds = _output_inds
        else:
            output_inds = tuple(output_inds)
        if set(_output_inds) != set(output_inds):
            raise TypeError("specified out_inds not allow in tensordot, \
                         make sure not summation/Hadamard product appears")
        if output_inds!=_output_inds:
            out = out.transpose(*output_inds, inplace=True)
    return out

def tensor_split(
    T,
    left_inds,
    method='svd',
    get=None,
    absorb='both',
    max_bond=None,
    cutoff=1e-10,
    cutoff_mode='rel',
    renorm=None,
    ltags=None,
    rtags=None,
    stags=None,
    bond_ind=None,
    right_inds=None,
    qpn_info=None
):
    """Decompose this Fermionic tensor into two fermionic tensors.

    Parameters
    ----------
    T : FermionTensor
        The fermionic tensor to split.
    left_inds : str or sequence of str
        The index or sequence of inds, which ``T`` should already have, to
        split to the 'left'. You can supply ``None`` here if you supply
        ``right_inds`` instead.
    method : str, optional
        How to split the tensor, only some methods allow bond truncation:

            - ``'svd'``: full SVD, allows truncation.

    get : {None, 'arrays', 'tensors', 'values'}
        If given, what to return instead of a TN describing the split:

            - ``None``: a tensor network of the two (or three) tensors.
            - ``'arrays'``: the raw data arrays (pyblock3.algebra.fermion.FlatFermionTensor) as
              a tuple ``(l, r)`` or ``(l, s, r)`` depending on ``absorb``.
            - ``'tensors '``: the new tensors as a tuple ``(Tl, Tr)`` or
              ``(Tl, Ts, Tr)`` depending on ``absorb``.
            - ``'values'``: only compute and return the singular values ``s``.

    absorb : {'both', 'left', 'right', None}, optional
        Whether to absorb the singular values into both, the left, or the right
        unitary matrix respectively, or neither. If neither (``absorb=None``)
        then the singular values will be returned separately as a 2D FermionTensor.
        If ``get='tensors'`` or ``get='arrays'`` then a tuple like
        ``(left, s, right)`` is returned.
    max_bond : None or int
        If integer, the maxmimum number of singular values to keep, regardless
        of ``cutoff``.
    cutoff : float, optional
        The threshold below which to discard singular values, only applies to
        rank revealing methods (not QR, LQ, or cholesky).
    cutoff_mode : {'sum2', 'rel', 'abs', 'rsum2'}
        Method with which to apply the cutoff threshold:

            - ``'rel'``: values less than ``cutoff * s[0]`` discarded.
            - ``'abs'``: values less than ``cutoff`` discarded.
            - ``'sum2'``: sum squared of values discarded must be ``< cutoff``.
            - ``'rsum2'``: sum squared of values discarded must be less than
              ``cutoff`` times the total sum of squared values.
            - ``'sum1'``: sum values discarded must be ``< cutoff``.
            - ``'rsum1'``: sum of values discarded must be less than
              ``cutoff`` times the total sum of values.

    renorm : {None, bool, or int}, optional
        Whether to renormalize the kept singular values, assuming the bond has
        a canonical environment, corresponding to maintaining the Frobenius
        norm or trace. If ``None`` (the default) then this is automatically
        turned on only for ``cutoff_method in {'sum2', 'rsum2', 'sum1',
        'rsum1'}`` with ``method in {'svd', 'eig', 'eigh'}``.
    ltags : sequence of str, optional
        Add these new tags to the left tensor.
    rtags : sequence of str, optional
        Add these new tags to the right tensor.
    stags : sequence of str, optional
        Add these new tags to the singular value tensor.
    bond_ind : str, optional
        Explicitly name the new bond, else a random one will be generated.
    right_inds : sequence of str, optional
        Explicitly give the right indices, otherwise they will be worked out.
        This is a minor performance feature.

    Returns
    -------
    FermionTensorNetwork or tuple[FermionTensor] or tuple[array] or 1D-array
        Depending on if ``get`` is ``None``, ``'tensors'``, ``'arrays'``, or
        ``'values'``. In the first three cases, if ``absorb`` is set, then the
        returned objects correspond to ``(left, right)`` whereas if
        ``absorb=None`` the returned objects correspond to
        ``(left, singular_values, right)``.
    """
    check_opt('get', get, _VALID_SPLIT_GET)

    if left_inds is None:
        left_inds = oset(T.inds) - oset(right_inds)
    else:
        left_inds = tags_to_oset(left_inds)

    if right_inds is None:
        right_inds = oset(T.inds) - oset(left_inds)

    opts = _parse_split_opts(
           method, cutoff, absorb, max_bond, cutoff_mode, renorm)
    _left_inds = [T.inds.index(i) for i in left_inds]
    _right_inds =[T.inds.index(i) for i in right_inds]

    if method == "svd":
        left, s, right = T.data.tensor_svd(_left_inds, right_idx=_right_inds, qpn_info=qpn_info, **opts)
    else:
        raise NotImplementedError

    if get == 'arrays':
        if absorb is None:
            return left, s, right
        return left, right

    ltags = T.tags | tags_to_oset(ltags)
    rtags = T.tags | tags_to_oset(rtags)
    if bond_ind is None:
        if absorb is None:
            bond_ind = (rand_uuid(),) * 2
        else:
            bond_ind = (rand_uuid(),)
    elif isinstance(bond_ind, str):
        bond_ind = (bond_ind,) * 2

    Tl = FermionTensor(data=left, inds=(*left_inds, bond_ind[0]), tags=ltags)
    Tr = FermionTensor(data=right, inds=(bond_ind[-1], *right_inds), tags=rtags)

    if absorb is None:
        stags = T.tags | tags_to_oset(stags)
        Ts = FermionTensor(data=s, inds=bond_ind, tags=stags)
        tensors = (Tl, Ts, Tr)
    else:
        tensors = (Tl, Tr)

    if get == 'tensors':
        return tensors

    return FermionTensorNetwork(tensors[::-1], check_collisions=False, virtual=True)

def _compress_connected(Tl, Tr, absorb='both', **compress_opts):
    """Compression of two Fermionic tensors that are adjacent to each other.

    Parameters
    ----------
    Tl : FermionTensor
        The left tensor.
    Tr : FermionTensor
        The right tensor, with matching indices and dimensions to ``T1``.
    absorb : {'both', 'left', 'right', None}, optional
        Where to absorb the singular values after decomposition.
    compress_opts :
        Supplied to :func:`~quimb.tensor.fermion.tensor_split`.

    Returns
    -------
    two fermionic Tensors
    """

    if Tl.inds == Tr.inds:
        return Tl, Tr
    left_inds = [ind for ind in Tl.inds if ind not in Tr.inds]
    right_inds = [ind for ind in Tr.inds if ind not in Tl.inds]
    out = _contract_connected(Tl, Tr)
    qpn_info = (Tl.data.dq, Tr.data.dq)
    if Tl.get_fermion_info()[1] < Tr.get_fermion_info()[1]:
        if absorb == "left":
            absorb = "right"
        elif absorb == "right":
            absorb = "left"
        r, l = out.split(left_inds=right_inds, right_inds=left_inds,
                         absorb=absorb, get="tensors", qpn_info=qpn_info, **compress_opts)
    else:
        l, r = out.split(left_inds=left_inds, right_inds=right_inds,
                         absorb=absorb, get="tensors", qpn_info=qpn_info, **compress_opts)
    return l, r

def tensor_compress_bond(
    T1,
    T2,
    absorb='both',
    inplace=True,
    info=None,
    **compress_opts
):
    """compress between the two single fermionic tensors.

    Parameters
    ----------
    T1 : FermionTensor
        The left tensor.
    T2 : FermionTensor
        The right tensor.
    absorb : {'both', 'left', 'right', None}, optional
        Where to absorb the singular values after decomposition.
    info : None or dict, optional
        A dict for returning extra information such as the singular values.
    compress_opts :
        Supplied to :func:`~quimb.tensor.fermion.tensor_split`.
    """
    fs, (tid1, tid2) = _fetch_fermion_space(T1, T2, inplace=inplace)
    site1, site2 = fs[tid1][1], fs[tid2][1]
    fs.make_adjacent(tid1, tid2)
    l, r = _compress_connected(T1, T2, absorb, **compress_opts)
    T1.modify(data=l.data, inds=l.inds)
    T2.modify(data=r.data, inds=r.inds)
    tid_map = {tid1: site1, tid2:site2}
    fs._reorder_from_dict(tid_map)
    return T1, T2

def _canonize_connected(T1, T2, absorb='right', **split_opts):
    """Compression of two Fermionic tensors that are adjacent to each other.

    Parameters
    ----------
    T1 : FermionTensor
        The left tensor.
    T2 : FermionTensor
        The right tensor, with matching indices and dimensions to ``T1``.
    absorb : {'both', 'left', 'right', None}, optional
        Where to absorb the singular values after decomposition.
    split_opts :
        Supplied to :func:`~quimb.tensor.fermion.tensor_split`.

    Returns
    -------
    two fermionic Tensors
    """
    if absorb == 'both':
        return _compress_connected(T1, T2, absorb=absorb, **split_opts)
    if absorb == "left":
        T1, T2 = T2, T1

    shared_ix, left_env_ix = T1.filter_bonds(T2)
    if not shared_ix:
        raise ValueError("The tensors specified don't share an bond.")

    if T1.get_fermion_info()[1] < T2.get_fermion_info()[1]:
        qpn_info = (T1.data.dq.__class__(0), T1.data.dq)
        tRfact, new_T1 = T1.split(shared_ix, get="tensors", qpn_info=qpn_info, **split_opts)
        new_T2 = _contract_connected(T2, tRfact)
    else:
        qpn_info = (T1.data.dq, T1.data.dq.__class__(0))
        new_T1, tRfact = T1.split(left_env_ix, get='tensors', qpn_info=qpn_info, **split_opts)
        new_T2 = _contract_connected(tRfact, T2)

    if absorb == "left":
        return new_T2, new_T1
    else:
        return new_T1, new_T2

def tensor_canonize_bond(T1, T2, absorb='right', **split_opts):
    r"""Inplace 'canonization' of two fermionic tensors. This gauges the bond
    between the two such that ``T1`` is isometric

    Parameters
    ----------
    T1 : FermionTensor
        The tensor to be isometrized.
    T2 : FermionTensor
        The tensor to absorb the R-factor into.
    split_opts
        Supplied to :func:`~quimb.tensor.fermion.tensor_split`, with
        modified defaults of ``method=='svd'`` and ``absorb='right'``.
    """
    check_opt('absorb', absorb, ('left', 'both', 'right'))

    if absorb == 'both':
        return tensor_compress_bond(T1, T2, absorb=absorb, **split_opts)

    fs, (tid1, tid2) = _fetch_fermion_space(T1, T2, inplace=True)
    site1, site2 = fs[tid1][1], fs[tid2][1]

    fs.make_adjacent(tid1, tid2)
    l, r = _canonize_connected(T1, T2, absorb, **split_opts)
    T1.modify(data=l.data, inds=l.inds)
    T2.modify(data=r.data, inds=r.inds)
    tid_map = {tid1: site1, tid2:site2}
    fs._reorder_from_dict(tid_map)

def tensor_balance_bond(t1, t2, smudge=1e-6):
    """Gauge the bond between two tensors such that the norm of the 'columns'
    of the tensors on each side is the same for each index of the bond.

    Parameters
    ----------
    t1 : FermionTensor
        The first tensor, should share a single index with ``t2``.
    t2 : FermionTensor
        The second tensor, should share a single index with ``t1``.
    smudge : float, optional
        Avoid numerical issues by 'smudging' the correctional factor by this
        much - the gauging introduced is still exact.
    """
    from pyblock3.algebra.core import SubTensor
    from pyblock3.algebra.fermion import SparseFermionTensor
    ix, = t1.bonds(t2)
    t1H = t1.H.reindex_({ix: ix+'*'})
    t2H = t2.H.reindex_({ix: ix+'*'})
    out1 = _contract_connected(t1H, t1)
    out2 = _contract_connected(t2H, t2)
    sblk1 = []
    sblk2 = []
    for iblk1 in out1.data.to_sparse():
        for iblk2 in out2.data.to_sparse():
            if iblk1.q_labels != iblk2.q_labels:
                continue
            x = np.diag(np.asarray(iblk1))
            y = np.diag(np.asarray(iblk2))
            s = (x + smudge) / (y + smudge)
            sblk1.append(SubTensor(reduced=np.diag(s**-0.25), q_labels=iblk1.q_labels))
            sblk2.append(SubTensor(reduced=np.diag(s**0.25), q_labels=iblk2.q_labels))

    sign1 = t1.data.pattern[t1.inds.index(ix)]
    sign2 = t2.data.pattern[t2.inds.index(ix)]
    s1_pattern = {"+":"-+", "-":"+-"}[sign1]
    s2_pattern = {"-":"-+", "+":"+-"}[sign2]
    s1 = SparseFermionTensor(blocks=sblk1, pattern=s1_pattern).to_flat()
    s2 = SparseFermionTensor(blocks=sblk2, pattern=s2_pattern).to_flat()
    t1.multiply_index_diagonal_(ix, s1, location="back")
    t2.multiply_index_diagonal_(ix, s2, location="front")

class FermionSpace:
    """A labelled, ordered dictionary. The tensor labels point to the tensor
    and its position inside the fermion space.

    Parameters
    ----------
    tensor_order : dictionary, optional
        tensor_order[tid] = (tensor, site)
    virtual: bool, optional
        whether the FermionSpace should be a *view* onto the tensors it is
        given, or a copy of them.

    Attributes
    ----------
    tensor_map : dict
        Mapping of unique ids to tensors and its location, like``{tensor_id: (tensor, site) ...}``. I.e. this is where the tensors are 'stored' by the FermionSpace.
    """

    def __init__(self, tensor_order=None, virtual=True):
        self.tensor_order = {}
        if tensor_order is not None:
            for tid, (tsr, site) in tensor_order.items():
                self.add_tensor(tsr, tid, site, virtual=virtual)

    @property
    def sites(self):
        """ return a list of all the occupied positions
        """
        if len(self.tensor_order) == 0:
            return []
        else:
            return [val[1] for val in self.tensor_order.values()]

    def is_continuous(self):
        """ Check whether the tensors are continously placed in the Fermion Space
        """
        sites = self.sites
        if len(sites) == 0:
            return True
        else:
            if np.unique(sites).size != len(sites):
                raise ValueError("at least one site is occupied multiple times")
            return len(sites) == (max(sites)-min(sites)+1)

    def copy(self):
        """ Copy the FermionSpace object. Tensor ids and positions will be
        preserved and tensors will be copied
        """
        return FermionSpace(self.tensor_order, virtual=False)

    def add_tensor(self, tsr, tid=None, site=None, virtual=False):
        """ Add a tensor to the current FermionSpace, eg
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
        if (tid is None) or (tid in self.tensor_order.keys()):
            tid = rand_uuid(base="_T")
        if site is None:
            site = 0 if len(self.sites)==0 else max(self.sites) + 1
        if site not in self.sites:
            T = tsr if virtual else tsr.copy()
            self.tensor_order[tid] = (T, site)
            T.set_fermion_owner(self, tid)
        else:
            raise ValueError("site:%s occupied, use replace/insert_tensor method"%site)

    def replace_tensor(self, site, tsr, tid=None, virtual=False):
        """ Replace the tensor at a given site, eg
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
        atid, _, atsr = self[site]
        T = tsr if virtual else tsr.copy()
        if tid is None or (tid in self.tensor_order.keys() and tid != atid):
            tid = atid

        T.set_fermion_owner(self, tid)
        atsr.remove_fermion_owner()
        del self.tensor_order[atid]
        self.tensor_order[tid] = (T, site)

    def insert_tensor(self, site, tsr, tid=None, virtual=False):
        """ insert a tensor at a given site, all tensors afterwards
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
            tid = rand_uuid(base="_T")
        if site not in self.sites:
            self.add_tensor(tsr, tid, site=site, virtual=virtual)
        else:
            T = tsr if virtual else tsr.copy()
            T.set_fermion_owner(self, tid)
            for atid, (atsr, asite) in self.tensor_order.items():
                if asite >= site:
                    self.tensor_order.update({atid: (atsr, asite+1)})
            self.tensor_order.update({tid: (T, site)})


    def insert(self, site, *tsrs, virtual=False):
        """ insert a group of tensors at a given site, all tensors afterwards
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

    def get_tid(self, site):
        """ Return the tensor id at given site

        Parameters
        ----------
        site: int
            The position to obtain the tensor id
        """
        if site not in self.sites:
            raise KeyError("site:%s not occupied"%site)
        idx = self.sites.index(site)
        return list(self.tensor_order.keys())[idx]

    def _reorder_from_dict(self, tid_map):
        """ Reorder tensors from a tensor_id/position mapping.
        Pizorn algorithm will be applied during moving

        Parameters
        ----------
        tid_map: dictionary
            Mapping of tensor id to the desired location
        """

        tid_lst = list(tid_map.keys())
        des_sites = list(tid_map.values())
        # sort the destination sites to avoid cross-overs during moving
        work_des_sites = sorted(des_sites)[::-1]
        for isite in work_des_sites:
            ind = des_sites.index(isite)
            self.move(tid_lst[ind], isite)

    def is_adjacent(self, tid1, tid2):
        """ Check whether two tensors are adjacently placed in the space
        """
        site1 = self.tensor_order[tid1][1]
        site2 = self.tensor_order[tid2][1]
        return abs(site1-site2) == 1

    def __getitem__(self, tid_or_site):
        """Return a tuple of (tensor id, position, tensor) from the tag (tensor id or position)
        """
        if isinstance(tid_or_site, str):
            if tid_or_site not in self.tensor_order.keys():
                raise KeyError("tid:%s not found"%tid_or_site)
            tsr, site = self.tensor_order[tid_or_site]
            return tid_or_site, site, tsr
        elif isinstance(tid_or_site, int):
            if tid_or_site not in self.sites:
                raise KeyError("site:%s not occupied"%tid_or_site)
            tid = self.get_tid(tid_or_site)
            tsr = self.tensor_order[tid][0]
            return tid, tid_or_site, tsr
        else:
            raise ValueError("not a valid key value(tid or site)")

    def __setitem__(self, site, tsr):
        if site in self.sites:
            self.replace_tensor(site, tsr)
        else:
            self.add_tensor(site, tsr)

    def move(self, tid_or_site, des_site):
        """ Move a tensor inside this FermionSpace to the specified position with Pizorn algorithm.
        Both local and global phase will be factorized to this single tensor

        Parameters
        ----------
        tid_or_site: string or int
            id or position of the original tensor
        des_site: int
            the position to move the tensor to
        """

        tid, site, tsr = self[tid_or_site]
        if site == des_site: return
        move_left = (des_site < site)
        iterator = range(des_site, site) if move_left else range(site+1, des_site+1)
        shared_inds = []
        tid_lst = [self[isite][0] for isite in iterator]
        parity = 0
        for itid in tid_lst:
            itsr, isite = self.tensor_order[itid]
            parity += itsr.parity
            shared_inds += list(oset(itsr.inds) & oset(tsr.inds))
            if move_left:
                self.tensor_order[itid] = (itsr, isite+1)
            else:
                self.tensor_order[itid] = (itsr, isite-1)
        global_parity = (parity % 2) * tsr.data.parity
        if global_parity != 0: tsr.data._global_flip()
        axes = [tsr.inds.index(i) for i in shared_inds]
        if len(axes)>0: tsr.data._local_flip(axes)
        self.tensor_order[tid] = (tsr, des_site)

    def move_past(self, tsr, site_range=None):
        """ Move an external tensor past the specifed site ranges with Pizorn algorithm.
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
            site_range = (min(sites), max(sites)+1)
        start, end = site_range
        iterator = range(start, end)
        shared_inds = []
        tid_lst = [self[isite][0] for isite in iterator]
        parity = 0
        for itid in tid_lst:
            itsr, isite = self.tensor_order[itid]
            parity += itsr.parity
            shared_inds += list(oset(itsr.inds) & oset(tsr.inds))
        global_parity = (parity % 2) * tsr.data.parity
        if global_parity != 0: tsr.data._global_flip()
        axes = [tsr.inds.index(i) for i in shared_inds]
        if len(axes)>0: tsr.data._local_flip(axes)
        return tsr

    def make_adjacent(self, tid1, tid2, direction='left'):
        """ Move one tensor in the specified direction to make the two adjacent
        """
        if not self.is_adjacent(tid1, tid2):
            site1 = self.tensor_order[tid1][1]
            site2 = self.tensor_order[tid2][1]
            if site1 == site2: return
            sitemin, sitemax = min(site1, site2), max(site1, site2)
            if direction == 'left':
                self.move(sitemax, sitemin+1)
            elif direction == 'right':
                self.move(sitemin, sitemax-1)
            else:
                raise ValueError("direction %s not recognized"%direction)

    def _contract_pairs(self, tid_or_site1, tid_or_site2, out_inds=None, direction='left', inplace=True):
        """ Contract two tensors in the FermionSpace

        Parameters
        ----------
        tid_or_site1 : string or int
            Tensor id or position for the 1st tensor
        tid_or_site2 : string or int
            Tensor id or position for the 2nd tensor
        out_inds: list of string, optional
            The order for the desired output indices
        direction: string
            The direction to move tensors if the two are not adjacent
        inplace: bool
            Whether to contract/move tensors inplace or in a copied FermionSpace

        Returns
        -------
        scalar or a FermionTensor
        """
        fs = self if inplace else self.copy()
        out  = _contract_pairs(fs, tid_or_site1, tid_or_site2, out_inds, direction)

        if isinstance(out, (float, complex)):
            return out

        site1 = fs[tid_or_site1][1]
        site2 = fs[tid_or_site2][1]

        if direction=="left":
            site1 = min(site1, site2)
        else:
            site1 = max(site1, site2) - 1
        site2 = site1 + 1
        # the output fermion tensor will replace the two input tensors in the space
        fs.replace_tensor(site1, out, virtual=True)
        fs.remove_tensor(site2)

        return out

    def remove_tensor(self, tid_or_site):
        """ remove a specified tensor at a given site, eg
        012345               01234
        ABCDEF, (3, True) -> ABCEF
        """
        tid, site, tsr = self[tid_or_site]
        tsr.remove_fermion_owner()
        del self.tensor_order[tid]

        indent_sites = []
        for isite in self.sites:
            if isite > site:
                indent_sites.append(isite)
        indent_sites = sorted(indent_sites)
        tid_lst = [self.get_tid(isite) for isite in indent_sites]
        for tid in tid_lst:
            tsr, site = self.tensor_order[tid]
            self.tensor_order[tid] = (tsr, site-1)

    @property
    def H(self):
        """ Construct a FermionSpace for the bra state of the tensors
        """
        max_site = max(self.sites)
        new_fs = FermionSpace()
        for tid, (tsr, site) in self.tensor_order.items():
            T = tsr.copy()
            new_data = T.data.dagger
            new_inds = T.inds[::-1]
            T.modify(data=new_data, inds=new_inds)
            new_fs.add_tensor(T, tid, max_site-site, virtual=True)
        return new_fs


class FermionTensor(Tensor):
    """A labelled, tagged ndarray. The index labels are used instead of
    axis numbers to identify dimensions, and are preserved through operations.

    Parameters
    ----------
    data : pyblock3.algebra.fermion.FlatFermionTensor
        The n-dimensional data.
    inds : sequence of str
        The index labels for each dimension. Must match the number of
        dimensions of ``data``.
    tags : sequence of str, optional
        Tags with which to identify and group this tensor. These will
        be converted into a ``oset``.
    left_inds : sequence of str, optional
        Which, if any, indices to group as 'left' indices of an effective
        matrix. This can be useful, for example, when automatically applying
        unitary constraints to impose a certain flow on a tensor network but at
        the atomistic (Tensor) level.
    """
    def __init__(self, data=1.0, inds=(), tags=None, left_inds=None):

        # a new or copied Tensor always has no owners
        self._owners = dict()
        self._fermion_owner = None
        # Short circuit for copying Tensors
        if isinstance(data, self.__class__):
            self._data = data.data.copy()
            self._inds = data.inds
            self._tags = data.tags.copy()
            self._left_inds = data.left_inds
            return

        self._data = data # asarray(data)
        self._inds = tuple(inds)
        self._tags = tags_to_oset(tags)
        self._left_inds = tuple(left_inds) if left_inds is not None else None

        nd = ndim(self._data)
        if nd != len(self.inds):
            raise ValueError(
                f"Wrong number of inds, {self.inds}, supplied for array"
                f" of shape {self._data.shape}.")

        if self.left_inds and any(i not in self.inds for i in self.left_inds):
            raise ValueError(f"The 'left' indices {self.left_inds} are not "
                             f"found in {self.inds}.")

    @property
    def symmetry(self):
        return self.data.symmetry

    @property
    def fermion_owner(self):
        return self._fermion_owner

    @property
    def parity(self):
        return self.data.parity

    def norm(self):
        """Frobenius norm of this tensor.
        """
        return np.linalg.norm(self.data.data, 2)

    def ind_size(self, dim_or_ind):
        if isinstance(dim_or_ind, str):
            if dim_or_ind not in self.inds:
                raise ValueError("%s indice not found in the tensor"%dim_or_ind)
            dim_or_ind = self.inds.index(dim_or_ind)
        ipattern = self.data.pattern[dim_or_ind]
        if ipattern=="+":
            sz = [self.symmetry.from_flat(ix) for ix in self.data.q_labels[:,dim_or_ind]]
        else:
            sz = [-self.symmetry.from_flat(ix) for ix in self.data.q_labels[:,dim_or_ind]]
        sp = self.data.shapes[:,dim_or_ind]
        bond_dict = dict(zip(sz, sp))
        return BondInfo(bond_dict)

    def copy(self, deep=False):
        """Copy this tensor. Note by default (``deep=False``), the underlying
        array will *not* be copied. The fermion owner will to reset to None
        """
        if deep:
            t = copy.deepcopy(self)
            t.remove_fermion_owner()
        else:
            t = self.__class__(self, None)
        return t

    def multiply_index_diagonal(self, ind, x, inplace=False, location="front"):
        """Multiply this tensor by 1D array ``x`` as if it were a diagonal
        tensor being contracted into index ``ind``.
        """
        if location not in ["front", "back"]:
            raise ValueError("invalid for the location of the diagonal")
        t = self if inplace else self.copy(full=True)
        ax = t.inds.index(ind)
        if isinstance(x, FermionTensor):
            x = x.data
        if location=="front":
            out = np.tensordot(x, t.data, axes=((1,), (ax,)))
            transpose_order = list(range(1, ax+1)) + [0] + list(range(ax+1, t.ndim))
        else:
            out = np.tensordot(t.data, x, axes=((ax,),(0,)))
            transpose_order = list(range(ax)) + [t.ndim-1] + list(range(ax, t.ndim-1))
        data = np.transpose(out, transpose_order)
        t.modify(data=data)
        return t

    multiply_index_diagonal_ = functools.partialmethod(
        multiply_index_diagonal, inplace=True)

    def get_fermion_info(self):
        if self.fermion_owner is None:
            return None
        fs, tid = self.fermion_owner[1:]
        return (tid, fs().tensor_order[tid][1])

    def contract(self, *others, output_inds=None, **opts):
        return tensor_contract(self, *others, output_inds=output_inds, **opts)

    @fermion_owner.setter
    def fermion_owner(self, fowner):
        self._fermion_owner = fowner

    def set_fermion_owner(self, fs, tid):
        self.fermion_owner = (hash(fs), weakref.ref(fs), tid)

    def remove_fermion_owner(self):
        self.fermion_owner = None

    def isel(self, selectors, inplace=False):
        raise NotImplementedError

    def expand_ind(self, ind, size):
        raise NotImplementedError

    def new_ind(self, name, size=1, axis=0):
        raise NotImplementedError

    @property
    def shapes(self):
        return self._data.shapes

    @property
    def shape(self):
        """Return the "inflated" shape composed of maximal size for each leg
        """
        shapes = self.shapes
        return tuple(np.amax(shapes, axis=0))

    @functools.wraps(tensor_split)
    def split(self, *args, **kwargs):
        return tensor_split(self, *args, **kwargs)

    def transpose(self, *output_inds, inplace=False):
        """Transpose this tensor. This does not change the physical meaning of
        the operator represented, eg:
        T_{abc}a^{\dagger}b^{\dagger}c^{\dagger} = \tilda{T}_{cab}c^{\dagger}a^{\dagger}b^{\dagger}

        Parameters
        ----------
        output_inds : sequence of str
            The desired output sequence of indices.
        inplace : bool, optional
            Perform the tranposition inplace.

        Returns
        -------
        tt : Tensor
            The transposed tensor.

        See Also
        --------
        transpose_like
        """
        t = self if inplace else self.copy()

        output_inds = tuple(output_inds)  # need to re-use this.

        if set(t.inds) != set(output_inds):
            raise ValueError("'output_inds' must be permutation of the current"
                             f" tensor indices, but {set(t.inds)} != "
                             f"{set(output_inds)}")

        current_ind_map = {ind: i for i, ind in enumerate(t.inds)}
        out_shape = tuple(current_ind_map[i] for i in output_inds)
        t.modify(apply=lambda x: np.transpose(x, out_shape), inds=output_inds)
        return t

    transpose_ = functools.partialmethod(transpose, inplace=True)

    @property
    def H(self):
        """Return the ket of this tensor, eg:
        U_{abc} a^{\dagger}b^{\dagger}c^{\dagger} -> U^{cba\star}cba
        Note this is different from Fermionic transposition
        """
        data = self.data.dagger
        inds = self.inds[::-1]
        tsr = self.copy()
        tsr.modify(data=data, inds=inds)

        return tsr

    def fuse(self, fuse_map, inplace=False):
        raise NotImplementedError

    def unfuse(self, unfuse_map, shape_map, inplace=False):
        raise NotImplementedError

    def squeeze(self, inplace=False):
        raise NotImplementedError

    def norm(self):
        """Frobenius norm of this tensor.
        """
        return np.linalg.norm(self.data.data, 2)

    def symmetrize(self, ind1, ind2, inplace=False):
        raise NotImplementedError

    def unitize(self, left_inds=None, inplace=False, method='qr'):
        raise NotImplementedError

    def randomize(self, dtype=None, inplace=False, **randn_opts):
        raise NotImplementedError

    def flip(self, ind, inplace=False):
        raise NotImplementedError

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

    def graph(self, *args, **kwargs):
        """Plot a graph of this tensor and its indices.
        """
        FermionTensorNetwork((self,)).graph(*args, **kwargs)


def is_mergeable(*ts_or_tsn):
    """Check if all objects(FermionTensor or FermionTensorNetwork)
       are part of the same FermionSpace
    """
    if isinstance(ts_or_tsn, (FermionTensor, FermionTensorNetwork)):
        return True
    fs_lst = []
    site_lst = []
    for obj in ts_or_tsn:
        if isinstance(obj, FermionTensor):
            if obj.fermion_owner is None:
                return False
            hashval, fsobj, tid = obj.fermion_owner
            if fsobj() is None:
                return False
            fs_lst.append(hashval)
            site_lst.append(fsobj()[tid][1])
        elif isinstance(obj, FermionTensorNetwork):
            fs_lst.append(hash(obj.fermion_space))
            site_lst.extend(obj.filled_sites)
        else:
            raise TypeError("unable to find fermionspace")

    return all([fs==fs_lst[0] for fs in fs_lst]) and len(set(site_lst)) == len(site_lst)

class FermionTensorNetwork(TensorNetwork):


    def __init__(self, ts, *,  virtual=False, check_collisions=True):

        if is_mergeable(*ts) and virtual:
            self.tensor_map = dict()
            self.tag_map = dict()
            self.ind_map = dict()
            self.fermion_space = _fetch_fermion_space(*ts)[0]
            self.assemble(ts)
        else:
            if isinstance(ts, FermionTensorNetwork):
                self.tag_map = valmap(lambda tids: tids.copy(), ts.tag_map)
                self.ind_map = valmap(lambda tids: tids.copy(), ts.ind_map)
                self.fermion_space = ts.fermion_space if virtual else ts.fermion_space.copy()
                self.tensor_map = dict()
                for tid, t in ts.tensor_map.items():
                    self.tensor_map[tid] = self.fermion_space[tid][2]
                    self.tensor_map[tid].add_owner(self, tid)
                for ep in ts.__class__._EXTRA_PROPS:
                    setattr(self, ep, getattr(ts, ep))
                self.exponent = ts.exponent
                return

            # internal structure
            self.fermion_space = FermionSpace()
            self.tensor_map = dict()
            self.tag_map = dict()
            self.ind_map = dict()
            self._inner_inds = oset()
            for t in ts:
                self.add(t, virtual=virtual, check_collisions=check_collisions)
            self._inner_inds = None
            self.exponent = 0.0

    def __and__(self, other):
        """Combine this tensor network with more tensors, without contracting.
        Copies the tensors.
        """
        virtual = is_mergeable(self, other)
        return FermionTensorNetwork((self, other), virtual=virtual)

    def __or__(self, other):
        """Combine this tensor network with more tensors, without contracting.
        Views the constituent tensors.
        """
        return FermionTensorNetwork((self, other), virtual=True)

    def __iter__(self):
        sorted_sites = sorted(self.filled_sites)
        for isite in sorted_sites:
            yield self.fermion_space[isite][2]

    def _reorder_from_tid(self, tid_map, inplace=False):
        tn = self if inplace else self.copy(full=True)
        tn.fermion_space._reorder_from_dict(tid_map)
        return tn

    def balance_bonds(self, inplace=False):
        """Apply :func:`~quimb.tensor.fermion.tensor_balance_bond` to
        all bonds in this tensor network.

        Parameters
        ----------
        inplace : bool, optional
            Whether to perform the bond balancing inplace or not.

        Returns
        -------
        TensorNetwork
        """
        tn = self if inplace else self.copy(full=True)

        for ix, tids in tn.ind_map.items():
            if len(tids) != 2:
                continue
            tid1, tid2 = tids
            t1, t2 = [tn.tensor_map[x] for x in (tid1, tid2)]
            tensor_balance_bond(t1, t2)

        return tn

    balance_bonds_ = functools.partialmethod(balance_bonds, inplace=True)

    def assemble_with_tensor(self, tsr):
        if not is_mergeable(self, tsr):
            raise ValueError("tensor not same in the fermion space of the tensor network")
        tid = tsr.fermion_owner[2]
        TensorNetwork.add_tensor(self, tsr, tid, virtual=True)

    def assemble_with_tensor_network(self, tsn):
        if not is_mergeable(self, tsn):
            raise ValueError("the two tensor networks not in the fermion space")
        TensorNetwork.add_tensor_network(self, tsn, virtual=True)

    def assemble(self, t):
        if isinstance(t, (tuple, list)):
            for each_t in t:
                self.assemble(each_t)
            return

        istensor = isinstance(t, FermionTensor)
        istensornetwork = isinstance(t, FermionTensorNetwork)

        if not (istensor or istensornetwork):
            raise TypeError("TensorNetwork should be called as "
                            "`TensorNetwork(ts, ...)`, where each "
                            "object in 'ts' is a Tensor or "
                            "TensorNetwork.")
        if istensor:
            self.assemble_with_tensor(t)
        else:
            self.assemble_with_tensor_network(t)

    def add_tensor(self, tsr, tid=None, virtual=False, site=None):
        if tid is None or tid in self.fermion_space.tensor_order.keys():
            tid = rand_uuid(base="_T")
        if virtual:
            fs  = tsr.fermion_owner
            if fs is not None:
                if fs[0] != hash(self.fermion_space):
                    raise ValueError("the tensor is already is in a different Fermion Space, \
                                      inplace addition not allowed")
                else:
                    if fs[2] in self.tensor_map.keys():
                        raise ValueError("the tensor is already in this TensorNetwork, \
                                          inplace addition not allowed")
                    else:
                        self.assemble_with_tensor(tsr)
            else:
                self.fermion_space.add_tensor(tsr, tid, site, virtual=True)
                TensorNetwork.add_tensor(self, tsr, tid, virtual=True)
        else:
            T = tsr.copy()
            self.fermion_space.add_tensor(T, tid, site, virtual=True)
            TensorNetwork.add_tensor(self, T, tid, virtual=True)

    def add_tensor_network(self, tn, virtual=False, check_collisions=True):
        if virtual:
            if hash(tn.fermion_space) == hash(self.fermion_space):
                if is_mergeable(self, tn):
                    TensorNetwork.add_tensor_network(tn, virtual=virtual, check_collisions=check_collisions)
                else:
                    raise ValueError("the two tensornetworks co-share same sites, inplace addition not allow")
                return

        if not tn.is_continuous():
            raise ValueError("input tensor network is not contiguously ordered")

        sorted_tensors = []
        for tsr in tn:
            tid = tsr.get_fermion_info()[0]
            sorted_tensors.append([tid, tsr])
        # if inplace, fermion_owners need to be removed first to avoid conflicts
        if virtual:
            for tid, tsr in sorted_tensors:
                tsr.remove_fermion_owner()

        if check_collisions:  # add tensors individually
            if getattr(self, '_inner_inds', None) is None:
                self._inner_inds = oset(self.inner_inds())

            # check for matching inner_indices -> need to re-index
            other_inner_ix = oset(tn.inner_inds())
            clash_ix = self._inner_inds & other_inner_ix

            if clash_ix:
                can_keep_ix = other_inner_ix - self._inner_inds
                new_inds = oset(rand_uuid() for _ in range(len(clash_ix)))
                reind = dict(zip(clash_ix, new_inds))
                self._inner_inds.update(new_inds, can_keep_ix)
            else:
                self._inner_inds.update(other_inner_ix)

            # add tensors, reindexing if necessary
            for tid, tsr in sorted_tensors:
                if clash_ix and any(i in reind for i in tsr.inds):
                    tsr = tsr.reindex(reind, inplace=virtual)
                self.add_tensor(tsr, tid=tid, virtual=virtual)

        else:  # directly add tensor/tag indexes
            for tid, tsr in sorted_tensors:
                self.add_tensor(tsr, tid=tid, virtual=virtual)

    def add(self, t, virtual=False, check_collisions=True):
        """Add FermionTensor, FermionTensorNetwork or sequence thereof to self.
        """
        if isinstance(t, (tuple, list)):
            for each_t in t:
                self.add(each_t, virtual=virtual,
                         check_collisions=check_collisions)
            return

        istensor = isinstance(t, FermionTensor)
        istensornetwork = isinstance(t, FermionTensorNetwork)

        if not (istensor or istensornetwork):
            raise TypeError("TensorNetwork should be called as "
                            "`TensorNetwork(ts, ...)`, where each "
                            "object in 'ts' is a Tensor or "
                            "TensorNetwork.")

        if istensor:
            self.add_tensor(t, virtual=virtual)
        else:
            self.add_tensor_network(t, virtual=virtual,
                                    check_collisions=check_collisions)

    def select(self, tags, which='all'):

        tagged_tids = self._get_tids_from_tags(tags, which=which)
        ts = [self.tensor_map[n] for n in tagged_tids]
        tn = FermionTensorNetwork(ts, check_collisions=False, virtual=True)
        tn.view_like_(self)
        return tn

    def __iand__(self, tensor):
        """Inplace, but non-virtual, addition of a Tensor or TensorNetwork to
        this network. It should not have any conflicting indices.
        """
        if is_mergeable(self, tensor):
            self.assemble(tensor)
        else:
            self.add(tensor, virtual=False)
        return self

    def __ior__(self, tensor):
        """Inplace, virtual, addition of a Tensor or TensorNetwork to this
        network. It should not have any conflicting indices.
        """
        if is_mergeable(self, tensor):
            self.assemble(tensor)
        else:
            self.add(tensor, virtual=True)
        return self

    # ------------------------------- Methods ------------------------------- #

    @property
    def filled_sites(self):
        return [self.fermion_space[tid][1] for tid in self.tensor_map.keys()]

    def is_complete(self):
        '''
        Check if the current tensor network contains all the tensors in the fermion space
        '''
        full_tid = self.fermion_space.tensor_order.keys()
        tensor_tid = self.tensor_map.keys()
        return set(full_tid) == set(tensor_tid)

    def is_continuous(self):
        """
        Check if sites in the current tensor network are contiguously occupied
        """
        filled_sites = self.filled_sites
        if len(filled_sites) ==0 : return True
        return (max(filled_sites) - min(filled_sites) + 1) == len(filled_sites)

    def copy(self, full=False):
        """ For full copy, the tensors and underlying FermionSpace(all tensors in it) will
        be copied. For partial copy, the tensors in this network must be continuously
        placed and a new FermionSpace will be created to hold this continous sector.
        """
        if full:
            return self.__class__(self, virtual=False)
        else:
            if not self.is_continuous():
                raise TypeError("Tensors not continuously placed in the network, \
                                partial copy not allowed")
            newtn = FermionTensorNetwork([])
            newtn.add_tensor_network(self)
            newtn.view_like_(self)
            return newtn


    def _pop_tensor(self, tid, remove_from_fs=True):
        """Remove a tensor from this network, returning said tensor.
        """
        # pop the tensor itself
        t = self.tensor_map.pop(tid)

        # remove the tid from the tag and ind maps
        self._remove_tid(t.tags, self.tag_map, tid)
        self._remove_tid(t.inds, self.ind_map, tid)

        # remove this tensornetwork as an owner
        t.remove_owner(self)
        if remove_from_fs:
            self.fermion_space.remove_tensor(tid)
            t.remove_fermion_owner()

        return t


    _pop_tensor_ = functools.partialmethod(_pop_tensor, remove_from_fs=False)

    @property
    def H(self):
        tn = self.copy(full=True)
        fs = tn.fermion_space
        max_site = max(fs.sites)

        for tid, (tsr, site) in fs.tensor_order.items():
            new_data = tsr.data.dagger
            new_inds = tsr.inds[::-1]
            tsr.modify(data=new_data, inds=new_inds)
            fs.tensor_order.update({tid: (tsr, max_site-site)})
        return tn

    def __mul__(self, other):
        raise NotImplementedError

    def __rmul__(self, other):
        raise NotImplementedError

    def __imul__(self, other):
        raise NotImplementedError

    def __truediv__(self, other):
        raise NotImplementedError

    def __itruediv__(self, other):
        raise NotImplementedError

    # ----------------- selecting and splitting the network ----------------- #

    def __setitem__(self, tags, tensor):
        #TODO: FIXME
        """Set the single tensor uniquely associated with ``tags``.
        """
        tids = self._get_tids_from_tags(tags, which='all')
        if len(tids) != 1:
            raise KeyError("'TensorNetwork.__setitem__' is meant for a single "
                           "existing tensor only - found {} with tag(s) '{}'."
                           .format(len(tids), tags))

        if not isinstance(tensor, FermionTensor):
            raise TypeError("Can only set value with a new 'FermionTensor'.")

        tid, = tids
        site = self.fermion_space.tensor_order[tid][1]
        TensorNetwork._pop_tensor(tid)
        TensorNetwork.add_tensor(tensor, tid=tid, virtual=True)
        self.fermion_space.replace_tensor(site, tensor, tid=tid, virtual=True)

    def partition_tensors(self, tags, inplace=False, which='any'):
        """Split this TN into a list of tensors containing any or all of
        ``tags`` and a ``FermionTensorNetwork`` of the the rest.
        The tensors and FermionTensorNetwork remain in the same FermionSpace

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
            The untagged fermion tensor network, and the sequence of tagged Tensors.

        See Also
        --------
        partition, select, select_tensors
        """
        tagged_tids = self._get_tids_from_tags(tags, which=which)

        # check if all tensors have been tagged
        if len(tagged_tids) == self.num_tensors:
            return None, self.tensor_map.values()

        # Copy untagged to new network, and pop tagged tensors from this
        untagged_tn = self if inplace else self.copy(full=True)
        tagged_ts = tuple(map(untagged_tn._pop_tensor_, sorted(tagged_tids)))

        return untagged_tn, tagged_ts

    def partition(self, tags, which='any', inplace=False):
        """Split this TN into two, based on which tensors have any or all of
        ``tags``. Unlike ``partition_tensors``, both results are TNs which
        inherit the structure of the initial TN.

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
        tagged_tids = self._get_tids_from_tags(tags, which=which)

        kws = {'check_collisions': False}
        t1 = self if inplace else self.copy(full=True)
        t2s = [t1._pop_tensor_(tid) for tid in tagged_tids]
        t2 = FermionTensorNetwork(t2s, virtual=True, **kws)
        t2.view_like_(self)
        return t1, t2

    def replace_with_svd(self, where, left_inds, eps, *, which='any',
                         right_inds=None, method='isvd', max_bond=None,
                         absorb='both', cutoff_mode='rel', renorm=None,
                         ltags=None, rtags=None, keep_tags=True,
                         start=None, stop=None, inplace=False):
        raise NotImplementedError

    def contract_between(self, tags1, tags2, **contract_opts):
        """Contract the two tensors specified by ``tags1`` and ``tags2``
        respectively. This is an inplace operation. No-op if the tensor
        specified by ``tags1`` and ``tags2`` is the same tensor.

        Parameters
        ----------
        tags1 :
            Tags uniquely identifying the first tensor.
        tags2 : str or sequence of str
            Tags uniquely identifying the second tensor.
        contract_opts
            Supplied to :func:`~quimb.tensor.fermion.tensor_contract`.
        """
        tid1, = self._get_tids_from_tags(tags1, which='all')
        tid2, = self._get_tids_from_tags(tags2, which='all')
        direction = contract_opts.pop("direction", "left")

        # allow no-op for same tensor specified twice ('already contracted')
        if tid1 == tid2:
            return

        self._pop_tensor_(tid1)
        self._pop_tensor_(tid2)

        out = self.fermion_space._contract_pairs(tid1, tid2, direction=direction, inplace=True)
        if isinstance(out, (float, complex)):
            return out
        else:
            self |= out

    def contract_ind(self, ind, **contract_opts):
        """Contract tensors connected by ``ind``.
        """
        tids = self._get_tids_from_inds(ind)
        if len(tids) <= 1: return
        ts = [self._pop_tensor_(tid) for tid in tids]
        direction = contract_opts.pop("direction", "left")
        out = tensor_contract(*ts, direction=direction, inplace=True)
        if isinstance(out, (float, complex)):
            return out
        else:
            self |= out

    def contract_tags(self, tags, inplace=False, which='any', **opts):

        tids = self._get_tids_from_tags(tags, which='any')
        if len(tids)  == 0:
            raise ValueError("No tags were found - nothing to contract. "
                             "(Change this to a no-op maybe?)")
        elif len(tids) == 1:
            return self

        untagged_tn, tagged_ts = self.partition_tensors(
            tags, inplace=inplace, which=which)


        contracted = tensor_contract(*tagged_ts, inplace=True, **opts)

        if untagged_tn is None:
            return contracted

        untagged_tn.add_tensor(contracted, virtual=True)
        return untagged_tn

    def contract(self, tags=..., inplace=False, **opts):

        if tags is all:
            return tensor_contract(*self, inplace=inplace, **opts)

        # this checks whether certain TN classes have a manually specified
        #     contraction pattern (e.g. 1D along the line)
        if self._CONTRACT_STRUCTURED:
            raise NotImplementedError()

        # else just contract those tensors specified by tags.
        return self.contract_tags(tags, inplace=inplace, **opts)

    def _compress_between_tids(
        self,
        tid1,
        tid2,
        canonize_distance=None,
        canonize_opts=None,
        equalize_norms=False,
        **compress_opts
    ):

        Tl = self.tensor_map[tid1]
        Tr = self.tensor_map[tid2]
        tensor_compress_bond(Tl, Tr, inplace=True, **compress_opts)

        if canonize_distance:
            raise NotImplementedError

        if equalize_norms:
            self.strip_exponent(tid1, equalize_norms)
            self.strip_exponent(tid2, equalize_norms)

    def _canonize_between_tids(
        self,
        tid1,
        tid2,
        equalize_norms=False,
        **canonize_opts,
    ):
        Tl = self.tensor_map[tid1]
        Tr = self.tensor_map[tid2]
        tensor_canonize_bond(Tl, Tr, **canonize_opts)

        if equalize_norms:
            self.strip_exponent(tid1, equalize_norms)
            self.strip_exponent(tid2, equalize_norms)

    def replace_section_with_svd(self, start, stop, eps,
                                 **replace_with_svd_opts):
        raise NotImplementedError

    def convert_to_zero(self):
        raise NotImplementedError

    def compress_all(self, inplace=False, **compress_opts):
        raise NotImplementedError

    def new_bond(self, tags1, tags2, **opts):
        raise NotImplementedError

    def cut_bond(self, bnd, left_ind, right_ind):
        raise NotImplementedError

    def cut_between(self, left_tags, right_tags, left_ind, right_ind):
        raise NotImplementedError

    def cut_iter(self, *inds):
        raise NotImplementedError
