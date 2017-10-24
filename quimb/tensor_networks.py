"""Tensor network tools.
"""

import os
import functools
import operator
import copy
import itertools
import string

from cytoolz import (
    unique,
    concat,
    frequencies,
    partition_all,
    merge_with,
)
import numpy as np
import scipy.sparse.linalg as spla

from .accel import prod, make_immutable, njit
from .linalg.base_linalg import norm_fro_dense
from .gen.operators import spin_operator, eye

try:
    # opt_einsum is highly recommended as until numpy 1.14 einsum contractions
    # do not use BLAS.
    import opt_einsum
    einsum = opt_einsum.contract

    @functools.wraps(opt_einsum.contract_path)
    def einsum_path(*args, optimize='greedy', memory_limit=2**28, **kwargs):
        return opt_einsum.contract_path(
            *args, path=optimize, memory_limit=memory_limit, **kwargs)

except ImportError:

    @functools.wraps(np.einsum)
    def einsum(*args, optimize='greedy', memory_limit=2**28, **kwargs):

        if optimize is False:
            return np.einsum(*args, optimize=False, **kwargs)

        explicit_path = (isinstance(optimize, (tuple, list)) and
                         optimize[0] == 'einsum_path')

        if explicit_path:
            optimize = optimize
        else:
            optimize = (optimize, memory_limit)

        return np.einsum(*args, optimize=optimize, **kwargs)

    @functools.wraps(np.einsum_path)
    def einsum_path(*args, optimize='greedy', memory_limit=2**28, **kwargs):
        return np.einsum_path(
            *args, optimize=(optimize, memory_limit), **kwargs)


# --------------------------------------------------------------------------- #
#                                Tensor Funcs                                 #
# --------------------------------------------------------------------------- #

def set_join(sets):
    """Combine a sequence of sets.
    """
    new_set = set()
    for each_set in sets:
        new_set |= each_set
    return new_set


def _gen_output_inds(all_inds):
    """Generate the output, i.e. unnique, indices from the set ``inds``. Raise
    if any index found more than twice.
    """
    for ind, freq in frequencies(all_inds).items():
        if freq > 2:
            raise ValueError("The index {} appears more "
                             "than twice!".format(ind))
        elif freq == 1:
            yield ind


_einsum_symbols = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
_einsum_symbols_set = set(_einsum_symbols)


def _maybe_map_indices_to_alphabet(a_ix, i_ix, o_ix):
    """``einsum`` need characters a-z,A-Z or equivalent numbers.
    Do this early, and allow *any* index labels.

    Parameters
    ----------
    a_ix : set
        All of the input indices.
    i_ix : sequence of sequence
        The input indices per tensor.
    o_ix : list of int
        The output indices.

    Returns
    -------
    contract_str : str
        The string to feed to einsum/contract.
    """
    if any(i not in _einsum_symbols_set for i in a_ix):
        # need to map inds to alphabet
        if len(a_ix) > len(_einsum_symbols_set):
            raise ValueError("Too many indices to auto-optimize contraction "
                             "for at once, try setting a `contract_strategy` "
                             "or do a manual contraction order using tags.")

        amap = dict(zip(a_ix, _einsum_symbols))
        in_str = ("".join(amap[i] for i in ix) for ix in i_ix)
        out_str = "".join(amap[o] for o in o_ix)

    else:
        in_str = ("".join(ix) for ix in i_ix)
        out_str = "".join(o_ix)

    return ",".join(in_str) + "->" + out_str


class HuskArray(np.ndarray):
    """Just an ndarray with only shape defined, so as to allow caching on shape
    alone.
    """

    def __init__(self, shape):
        self.shape = shape


@functools.lru_cache(1024)
def cache_einsum_path_on_shape(contract_str, *shapes):
    return einsum_path(contract_str, *(HuskArray(shape) for shape in shapes),
                       memory_limit=2**28, optimize='greedy')[0]


def tensor_contract(*tensors, output_inds=None):
    """Efficiently contract multiple tensors, combining their tags.

    Parameters
    ----------
    *tensors : sequence of Tensor
        The tensors to contract.
    output_inds : sequence
        If given, the desired order of output indices, else defaults to the
        order they occur in the input indices.

    Returns
    -------
    scalar or Tensor
    """
    i_ix = tuple(t.inds for t in tensors)  # input indices per tensor
    a_ix = tuple(concat(i_ix))  # list of all input indices

    if output_inds is None:
        # sort output indices  by input order for efficiency and consistency
        o_ix = tuple(x for x in a_ix if x in [*_gen_output_inds(a_ix)])
    else:
        o_ix = output_inds

    # possibly map indices into the 0-52 range needed by einsum
    contract_str = _maybe_map_indices_to_alphabet([*unique(a_ix)], i_ix, o_ix)

    # perform the contraction
    path = cache_einsum_path_on_shape(
        contract_str, *(t.shape for t in tensors))
    o_array = einsum(
        contract_str, *(t.data for t in tensors), optimize=path)

    if not o_ix:
        return o_array

    # unison of all tags
    o_tags = set_join(t.tags for t in tensors)

    return Tensor(array=o_array, inds=o_ix, tags=o_tags)


RAND_UUIDS = map("".join, itertools.product(string.hexdigits, repeat=7))


def rand_uuid(base=""):
    """Return a guaranteed unique, shortish identifier, optional appended
    to ``base``.

    Examples
    --------
    >>> rand_uuid()
    '_2e1dae1b'

    >>> rand_uuid('virt-bond')
    'virt-bond_bf342e68'
    """
    return base + "_" + next(RAND_UUIDS)


@njit  # pragma: no cover
def _array_split_svd(x, tol=-1.0):
    """SVD-decomposition.
    """
    U, s, V = np.linalg.svd(x, full_matrices=False)

    if tol > 0.0:
        s = s[s > tol]
        n_chi = s.size
        U = U[..., :n_chi]
        V = V[:n_chi, ...]

    s **= 0.5
    U = U * s.reshape((1, -1))
    V = s.reshape((-1, 1)) * V

    return U, V


@njit  # pragma: no cover
def dag(x):
    """Hermitian conjugate.
    """
    return np.conjugate(x).T


@njit  # pragma: no cover
def _array_split_eig(x, tol=-1.0):
    """SVD-split via eigen-decomposition.
    """
    if x.shape[0] > x.shape[1]:
        l, V = np.linalg.eigh(dag(x) @ x)
        U = x @ dag(V)

    else:
        l, U = np.linalg.eigh(x @ dag(x))
        V = dag(U) @ x

    if tol > 0.0:
        n_chi = np.sum(l > tol**2)
        U = U[..., :n_chi]
        V = V[:n_chi, ...]

    return U, V


@njit  # pragma: no cover
def _array_split_qr(x, tol):
    """QR-decomposition.
    """
    Q, R = np.linalg.qr(x)
    return Q, R


@njit  # pragma: no cover
def _array_split_lq(x, tol):
    """QR-decomposition.
    """
    Q, L = np.linalg.qr(x.T)
    return L.T, Q.T


class BondError(Exception):
    pass


# --------------------------------------------------------------------------- #
#                                Tensor Class                                 #
# --------------------------------------------------------------------------- #

class Tensor(object):
    """A labelled, tagged ndarray.

    Parameters
    ----------
    data : numpy.ndarray
        The n-dimensions data.
    inds : sequence of hashable
        The index labels for each dimension.
    tags : sequence of hashable
        Tags with which to select and filter from multiple tensors.

    Members
    -------

    """

    def __init__(self, data, inds, tags=None):
        # Short circuit for copying Tensors
        if isinstance(data, Tensor):
            self._data = data.data
            self.inds = data.inds
            self.tags = data.tags.copy()
            return

        self._data = np.asarray(data)
        self.inds = tuple(inds)

        if self._data.ndim != len(self.inds):
            raise ValueError(
                "Wrong number of inds, {}, supplied for array"
                " of shape {}.".format(self.inds, self._data.shape))

        self.tags = (set() if tags is None else
                     {tags} if isinstance(tags, str) else
                     set(tags))

    def copy(self, deep=False):
        """
        """
        if deep:
            return copy.deepcopy(self)
        else:
            return Tensor(self, None)

    def get_data(self):
        return self._data

    def set_data(self, data):
        if data.size != self.size:
            raise ValueError("Cannot set - array size does not match Tensor.")
        elif data.shape != self.shape:
            self._data = data.reshape(self.shape)
        else:
            self._data = data

    data = property(get_data, set_data,
                    doc="The numpy.ndarray with this Tensors' numeric data.")

    def conj(self, inplace=False):
        """Conjugate this tensors data (does nothing to indices).
        """
        if inplace:
            self._data = self._data.conj()
            return self
        else:
            return Tensor(self._data.conj(), self.inds, self.tags)

    @property
    def H(self):
        """Conjugate this tensors data (does nothing to indices).
        """
        return self.conj()

    @property
    def shape(self):
        return self._data.shape

    @property
    def size(self):
        return self._data.size

    def inner_inds(self):
        """
        """
        ind_freqs = frequencies(self.inds)
        return set(i for i in self.inds if ind_freqs[i] == 2)

    def transpose(self, *output_inds, inplace=False):
        """Transpose this tensor.

        Parameters
        ----------
        output_inds : sequence of hashable
            The desired output sequence of indices.

        Returns
        -------
        Tensor
        """
        if inplace:
            tn = self
        else:
            tn = self.copy()

        output_inds = tuple(output_inds)  # need to re-use this.

        if set(tn.inds) != set(output_inds):
            raise ValueError("'output_inds' must be permutation of the "
                             "current tensor indices, but {} != {}"
                             .format(set(tn.inds), set(output_inds)))

        current_ind_map = {ind: i for i, ind in enumerate(tn.inds)}
        out_shape = tuple(current_ind_map[i] for i in output_inds)

        tn._data = tn._data.transpose(*out_shape)
        tn.inds = output_inds
        return tn

    @functools.wraps(tensor_contract)
    def contract(self, *others, memory_limit=2**28, optimize='greedy',
                 output_inds=None):
        return tensor_contract(self, *others, memory_limit=memory_limit,
                               optimize=optimize, output_inds=output_inds)

    def split(self, left_inds, method='svd', tol=None,
              max_bond=None, return_tensors=False):
        """Decompose this tensor into two tensors.

        Parameters
        ----------
        left_inds : sequence of hashable
            The sequence of inds, which ``tensor`` should already have, to
            split to the 'left'.
        method : {'svd', 'eig', 'qr', 'lq'}, optional
            How to split the tensor.
        tol : float, optional
            The tolerance below which to discard singular values, only applies
            to ``method='svd'`` and ``method='eig'``.
        max_bond : int, optional
            If the new bond is larger than this, raise a ``BondError``.
        return_tensors : bool, optional
            If true, return the two tensors rather than the TensorNetwork
            describing the split tensor.

        Returns
        -------
        TensorNetwork or (Tensor, Tensor)
        """
        left_inds = tuple(left_inds)
        right_inds = tuple(x for x in self.inds if x not in left_inds)

        TT = self.transpose(*left_inds, *right_inds)

        left_dims = TT.shape[:len(left_inds)]
        right_dims = TT.shape[len(left_inds):]

        array = TT.data.reshape(prod(left_dims), prod(right_dims))

        if tol is None:
            tol = -1.0

        left, right = {'svd': _array_split_svd,
                       'eig': _array_split_eig,
                       'qr': _array_split_qr,
                       'lq': _array_split_lq}[method](array, tol=tol)

        left = left.reshape(*left_dims, -1)
        right = right.reshape(-1, *right_dims)

        if max_bond is not None:
            if left.shape[-1] > max_bond:
                raise BondError("Maximum bond size exceeded")

        bond_ind = rand_uuid()

        Tl = Tensor(array=left, inds=(*left_inds, bond_ind), tags=self.tags)
        Tr = Tensor(array=right, inds=(bond_ind, *right_inds), tags=self.tags)

        if return_tensors:
            return Tl, Tr

        return TensorNetwork((Tl, Tr), check_collisions=False)

    def reindex(self, index_map, inplace=False):
        """Rename the indices of this tensor, optionally in-place.

        Parameters
        ----------
        index_map : dict-like
            Mapping of pairs ``{old_ind: new_ind, ...}``.
        """
        if inplace:
            new = self
        else:
            new = self.copy()

        new.inds = tuple(index_map.get(ind, ind) for ind in new.inds)
        return new

    def fuse(self, fuse_map, inplace=False):
        """Combine groups of indices into single indices.

        Parameters
        ----------
        fuse_map : dict_like or sequence of tuples.
            Mapping like: ``{new_ind: sequence of existing inds, ...}`` or an
            ordered mapping like ``[(new_ind_1, old_inds_1), ...]`` in which
            case the output tensor's fused inds will be ordered. In both cases
            the new indices are created at the beginning of the tensor's shape.

        Returns
        -------
        Tensor
            The transposed, reshaped and re-labeled tensor.
        """
        if inplace:
            tn = self
        else:
            tn = self.copy()

        if isinstance(fuse_map, dict):
            new_fused_inds, fused_inds = zip(*fuse_map.items())
        else:
            new_fused_inds, fused_inds = zip(*fuse_map)

        unfused_inds = tuple(
            i for i in tn.inds if not
            any(i in fs for fs in fused_inds))

        # transpose tensor to bring groups of fused inds to the beginning
        tn.transpose(*concat(fused_inds), *unfused_inds, inplace=True)

        # for each set of fused dims, group into product, then add remaining
        dims = iter(tn.shape)
        dims = [prod(next(dims) for _ in fs) for fs in fused_inds] + list(dims)

        # create new tensor with new + remaining indices
        tn._data = tn._data.reshape(*dims)
        tn.inds = (*new_fused_inds, *unfused_inds)
        return tn

    def norm(self):
        """Frobenius norm of this tensor.
        """
        return norm_fro_dense(self.data.reshape(-1))

    def almost_equals(self, other, **kwargs):
        """Check if this tensor is almost the same as another.
        """
        same_inds = (set(self.inds) == set(other.inds))
        if not same_inds:
            return False
        otherT = other.transpose(*self.inds)
        return np.allclose(self.data, otherT.data, **kwargs)

    def drop_tags(self, tags=None):
        """Drop certain tags, defaulting to all, from this tensor.
        """
        if tags is None:
            self.tags = set()
        elif isinstance(tags, str):
            self.tags.discard(tags)
        else:
            for tag in tags:
                self.tags.discard(tag)

    def filter_shared_inds(self, other):
        """Sort this tensor's indices into a list of those that it shares and
        doesn't share with another tensor.
        """
        shared = []
        unshared = []
        for i in self.inds:
            if i in other.inds:
                shared.append(i)
            else:
                unshared.append(i)
        return shared, unshared

    def __and__(self, other):
        """Combine with another ``Tensor`` or ``TensorNetwork`` into a new
        ``TensorNetwork``.
        """
        return TensorNetwork((self, other))

    def __matmul__(self, other):
        """Explicitly contract with another tensor.
        """
        return tensor_contract(self, other)

    def __repr__(self):
        return "Tensor(shape={}, inds={}, tags={})".format(
            self.data.shape,
            self.inds,
            self.tags)


# ------------------------- Add ufunc like methods -------------------------- #

def _make_promote_array_func(op, meth_name):

    @functools.wraps(getattr(np.ndarray, meth_name))
    def _promote_array_func(self, other):
        """Use standard array func, but make sure Tensor inds match.
        """
        if isinstance(other, Tensor):
            if set(self.inds) != set(other.inds):
                raise ValueError(
                    "The indicies of these two tensors do not "
                    "match: {} != {}".format(self.inds, other.inds))
            otherT = other.transpose(*self.inds)
            return Tensor(
                array=op(self.data, otherT.data), inds=self.inds,
                tags=self.tags | other.tags)
        return Tensor(array=op(self.data, other),
                      inds=self.inds, tags=self.tags)

    return _promote_array_func


for meth_name, op in [('__add__', operator.__add__),
                      ('__sub__', operator.__sub__),
                      ('__mul__', operator.__mul__),
                      ('__pow__', operator.__pow__),
                      ('__truediv__', operator.__truediv__)]:
    setattr(Tensor, meth_name, _make_promote_array_func(op, meth_name))


def _make_rhand_array_promote_func(op, meth_name):

    @functools.wraps(getattr(np.ndarray, meth_name))
    def _rhand_array_promote_func(self, other):
        """Right hand operations -- no need to check ind equality first.
        """
        return Tensor(array=op(other, self.data),
                      inds=self.inds, tags=self.tags)

    return _rhand_array_promote_func


for meth_name, op in [('__radd__', operator.__add__),
                      ('__rsub__', operator.__sub__),
                      ('__rmul__', operator.__mul__),
                      ('__rpow__', operator.__pow__),
                      ('__rtruediv__', operator.__truediv__)]:
    setattr(Tensor, meth_name, _make_rhand_array_promote_func(op, meth_name))


# --------------------------------------------------------------------------- #
#                            Tensor Network Class                             #
# --------------------------------------------------------------------------- #

_TN_SIMPLE_PROPS = ['contract_strategy', 'nsites', 'contract_bsz']
_TN_DATA_PROPS = ['tensor_index', 'tag_index']


class SiteIndexer(object):
    """
    """

    def __init__(self, tn):
        self.tn = tn

    def __getitem__(self, site):
        if site < 0:
            site = self.tn.nsites + site
        site_tag = self.tn.contract_strategy.format(site)
        return self.tn[site_tag]

    def __setitem__(self, site, tensor):
        if site < 0:
            site = self.tn.nsites + site
        site_tag = self.tn.contract_strategy.format(site)
        self.tn[site_tag] = tensor


class TensorNetwork(object):
    """A collection of (as yet uncontracted) Tensors.

    Parameters
    ----------
    tensors : sequence of Tensor or TensorNetwork
        The objects to combine. The new network will be a *view* onto these
        constituent tensors unless explicitly copied.
    check_collisions : bool, optional
        If True, the default, then Tensors and TensorNetworks with double
        indices which match another Tensor or TensorNetworks double indices
        will have those indices' names mangled. Should be explicily turned off
        when it is known that no collisions will take place -- i.e. when not
        adding any new tensors.
    contract_strategy : str, optional
        A string, with integer format specifier, that describes how to range
        over the network's tags in order to contract it.
    contract_bsz : int, optional
        How many sites to group together when auto contracting. Eg for 3 (with
        the dotted lines denoting vertical strips of tensors to be contracted):

            .....                ........          .....
            O-O-O-O-O-O-O-        /-O-O-O-O-        /-O-
            | | | | | | |   ->   0  | | | |   ->   8  |   ->  etc.
            O-O-O-O-O-O-O-        \-O-O-O-O-        \-O-

        Should not require tensor contractions with more than 52 unique
        indices.
    nsites : int, optional
        The number of sites, if explicitly known. This will be calculated
        using `contract_strategy` if needed but not specified.

    Members
    -------
    tensors : sequence of Tensor
        The tensors in this network.
    tensor_index : dict
        Mapping of names to tensors, like``{tensor_name: tensor, ...}``. I.e.
        this is where the tensors are 'stored' by the network.
    tag_index : dict
        Mapping of tags to a set of tensor names which have those tags. I.e.
        ``{tag: {tensor1, tensor2, ...}}``. Thus to select those tensors one
        might do: ``map(tensor_index.__getitem__, tag_index[tag])``.
    """

    def __init__(self, tensors, *,
                 check_collisions=True,
                 contract_strategy=None,
                 contract_bsz=None,
                 nsites=None):

        self.site = SiteIndexer(self)

        # short-circuit for copying TensorNetworks
        if isinstance(tensors, TensorNetwork):
            self.contract_strategy = tensors.contract_strategy
            self.nsites = tensors.nsites
            self.contract_bsz = tensors.contract_bsz
            self.tag_index = {
                tg: ns.copy() for tg, ns in tensors.tag_index.items()}
            self.tensor_index = {
                n: t.copy() for n, t in tensors.tensor_index.items()}
            return

        self.contract_strategy = contract_strategy
        self.contract_bsz = contract_bsz
        self.nsites = nsites

        self.tensor_index = {}
        self.tag_index = {}

        current_inner_inds = set()

        for t in tensors:

            istensor = isinstance(t, Tensor)
            istensornetwork = isinstance(t, TensorNetwork)

            if not (istensor or istensornetwork):
                raise TypeError("TensorNetwork should be called as "
                                "'TensorNetwork(tensors, ...)', where each "
                                "object in 'tensors' is a Tensor or "
                                "TensorNetwork.")

            if check_collisions:

                # check for matching inner_indices -> need to re-index
                new_inner_inds = set(t.inner_inds())
                if current_inner_inds & new_inner_inds:  # any overlap
                    t.reindex({old: old + "'" for old in new_inner_inds},
                              inplace=True)
                current_inner_inds |= t.inner_inds()

            if istensor:
                self.add_tensor(t)
                continue

            for x in _TN_SIMPLE_PROPS:
                # check whether to inherit ... or compare properties
                if getattr(t, x) is not None:

                    # dont' have prop yet -> inherit
                    if getattr(self, x) is None:
                        setattr(self, x, getattr(t, x))

                    # both have prop, and don't match -> raise
                    elif getattr(t, x) != getattr(self, x):
                        raise ValueError(
                            "Conflicting values found on tensor networks for "
                            "property {}. First value: {}, second value: {}"
                            .format(x, getattr(self, x), getattr(t, x)))

            if check_collisions:
                for name, tensor in t.tensor_index.items():
                    self.add_tensor(tensor, name=name)
            else:
                self.tensor_index.update(t.tensor_index)
                self.tag_index = merge_with(
                    set_join, self.tag_index, t.tag_index)

        # count how many sites if a contract_strategy is given
        if self.contract_strategy:

            if self.nsites is None:
                self.nsites = self.calc_nsites()

            # set default blocksize
            if self.contract_bsz is None:
                self.contract_bsz = 2

    # ------------------------------- Methods ------------------------------- #

    def copy(self, deep=False):
        """Copy this ``TensorNetwork``. If ``deep=False``, (the default), then
        everything but the actual numeric data will be copied.
        """
        if deep:
            return copy.deepcopy(self)
        return self.__class__(self)

    def add_tensor(self, tensor, name=None):
        """Add a single tensor to this network - mangle its name if neccessary.
        """
        # check for name conflict
        if (name is None) or (name in self.tensor_index):
            name = rand_uuid(base="_T")

        # add tensor to the main index
        self.tensor_index[name] = tensor

        # add its name to the relevant tags, or create a new tag
        for tag in tensor.tags:
            try:
                self.tag_index[tag].add(name)
            except KeyError:
                self.tag_index[tag] = {name}

    def pop_tensor(self, name):
        """Remove a tensor from this network, returning said tensor.
        """
        # remove the tensor from the tag index
        for tag in self.tensor_index[name].tags:
            self.tag_index[tag].discard(name)

        # pop the tensor itself
        return self.tensor_index.pop(name)

    def delete_tensor(self, name):
        """Delete a tensor from this network.
        """
        # remove the tensor from the tag index
        for tag in self.tensor_index(name).tags:
            self.tag_index(tag).discard(name)

        # remove the tensor itself
        del self.tensor_index[name]

    def add_tag(self, tag):
        """Add tag to every tensor in this network.
        """
        names = set()
        for n, t in self.tensor_index.items():
            names.add(n)
            t.tags.add(tag)
        self.tag_index[tag] = names

    @property
    def tensors(self):
        return list(self.tensor_index.values())

    def calc_nsites(self):
        """Calculate how many tags there are which match ``contract_strategy``.
        """
        nsites = 0
        while True:
            if self.contract_strategy.format(nsites) in self.tag_index:
                nsites += 1
            else:
                break
        return nsites

    def filter_by_tags(self, tags, inplace=False):
        """Split this TN into a list of tensors containing any of ``tags`` and
        a TensorNetwork of the the rest.

        Parameters
        ----------
        tags : sequence of hashable
            The list of tags to filter the tensors by. Use ``...``
            (``Ellipsis``) to contract all.
        inplace : bool, optional
            If true, remove tagged tensors from self, else create a new network
            with the tensors removed.

        Returns
        -------
        (u_tn, t_ts) : (TensorNetwork, Tensor sequence)
            The untagged tensor network, and the sequence of Tensors to
            contract.
        """

        # contract all
        if tags is ...:
            return None, self.tensor_index.values()

        # Else get the locations of where each tag is found on tensor
        if isinstance(tags, str):
            tagged_names = self.tag_index[tags]
        else:
            tagged_names = set_join(self.tag_index[t] for t in tags)

        # check if all tensors have been tagged
        if len(tagged_names) == len(self.tensor_index):
            return None, self.tensor_index.values()

        # Copy untagged to new network, and pop tagged tensors from this
        if inplace:
            untagged_tn = self
        else:
            untagged_tn = self.copy()
        tagged_ts = tuple(map(untagged_tn.pop_tensor, sorted(tagged_names)))

        return untagged_tn, tagged_ts

    def _contract_tags(self, tags, inplace=False):
        untagged_tn, tagged_ts = self.filter_by_tags(tags, inplace=inplace)

        if not tagged_ts:
            raise ValueError("No tags were found - nothing to contract. "
                             "(Change this to a no-op maybe?)")

        if untagged_tn:
            untagged_tn.add_tensor(tensor_contract(*tagged_ts))
            return untagged_tn

        return tensor_contract(*tagged_ts)

    def cumulative_contract(self, tags_seq, inplace=False):
        """Cumulative contraction of tensor network. Contract the first set of
        tags, then that set with the next set, then both of those with the next
        and so forth. Could also be described as an manually ordered
        contraction of all tags in ``tags_seq``.

        Parameters
        ----------
        tags_seq : sequence of sequence of hashable
            The list of tag-groups to cumulatively contract.

        Returns
        -------
        TensorNetwork, Tensor or Scalar
            The result of the contraction, still a TensorNetwork if the
            contraction was only partial.
        """
        if inplace:
            new_tn = self
        else:
            new_tn = self.copy()
        ctags = set()

        for tags in tags_seq:
            # accumulate tags from each contractions
            if isinstance(tags, str):
                tags = {tags}
            elif isinstance(tags, slice):
                step = 1 if tags.step is None else tags.step
                tags = {self.contract_strategy.format(i)
                        for i in range(tags.start, tags.stop, step)}
            else:
                tags = set(tags)

            ctags |= tags

            # peform the next contraction
            new_tn = new_tn._contract_tags(ctags, inplace=True)

        return new_tn

    def __rshift__(self, *args, **kwargs):
        """Overload of '>>' for TensorNetwork.cumulative_contract.
        """
        return self.cumulative_contract(*args, **kwargs)

    def __irshift__(self, *args, **kwargs):
        """Overload of '>>=' for inplace TensorNetwork.cumulative_contract.
        """
        return self.cumulative_contract(*args, **kwargs)

    def _contract_with_strategy(self, tags, inplace=False):
        # check for all sites
        if tags is ...:
            tags = slice(0, self.nsites)

        if tags.start is None:
            start = 0
        elif tags.start is ...:
            start = self.nsites - 1
        elif tags.start < 0:
            start = self.nsites + tags.start
        else:
            start = tags.start

        if tags.stop is ...:
            stop = self.nsites - 1
        elif tags.stop < 0:
            stop = self.nsites + tags.stop
        else:
            stop = tags.stop

        step = 1 if stop > start else -1

        tags_seq = (self.contract_strategy.format(i)
                    for i in range(start, stop, step))

        # partition sites into `contract_bsz` groups
        if self.contract_bsz > 1:
            tags_seq = partition_all(self.contract_bsz, tags_seq)

        return self.cumulative_contract(tags_seq, inplace=inplace)

    def contract(self, tags=..., inplace=False):
        """Contract some, or all, of the tensors in this network.

        Parameters
        ----------
        tags : sequence of hashable
            Any tensors with any of these tags with be contracted. Set to
            ``...`` (``Ellipsis``) to contract all tensors, the default.

        Returns
        -------
        TensorNetwork, Tensor or Scalar
            The result of the contraction, still a TensorNetwork if the
            contraction was only partial.
        """

        # Check for a structured strategy for performing contraction...
        if self.contract_strategy is not None:
            # ... but only use for total or slice tags
            if (tags is ...) or isinstance(tags, slice):
                return self._contract_with_strategy(tags, inplace=inplace)

        # Else just contract those tensors specified by tags.
        return self._contract_tags(tags, inplace=inplace)

    def __xor__(self, *args, **kwargs):
        """Overload of '^' for TensorNetwork.contract.
        """
        return self.contract(*args, **kwargs)

    def __ixor__(self, *args, **kwargs):
        """Overload of '^=' for inplace TensorNetwork.contract.
        """
        return self.contract(*args, **kwargs)

    def reindex(self, index_map, inplace=False):
        """Rename indices for all tensors in this network, optionally in-place.

        Parameters
        ----------
        index_map : dict-like
            Mapping of pairs ``{old_ind: new_ind, ...}``.
        """
        if inplace:
            new_tn = self
        else:
            new_tn = self.copy()

        for t in new_tn.tensor_index.values():
            t.reindex(index_map, inplace=True)
        return new_tn

    def conj(self, inplace=False):
        """Conjugate all the tensors in this network (leaves all indices).
        """
        if inplace:
            new_tn = self
        else:
            new_tn = self.copy()

        for t in new_tn.tensor_index.values():
            t.conj(inplace=True)

        return new_tn

    @property
    def H(self):
        """Conjugate all the tensors in this network (leaves all indices).
        """
        return self.conj()

    def all_dims_inds(self):
        """Return a list of all dimensions, and the corresponding list of
        indices from the tensor network.
        """
        return zip(*concat(zip(t.shape, t.inds)
                           for t in self.tensor_index.values()))

    def all_inds(self):
        return concat(t.inds for t in self.tensor_index.values())

    def inner_inds(self):
        """Return set of all inner indices, i.e. those that appear twice.
        """
        all_inds = tuple(self.all_inds())
        ind_freqs = frequencies(all_inds)
        return set(i for i in all_inds if ind_freqs[i] == 2)

    def outer_dims_inds(self):
        """Get the 'outer' pairs of dimension and indices, i.e. as if this
        tensor network was fully contracted.
        """
        dims, inds = self.all_dims_inds()
        ind_freqs = frequencies(inds)
        return tuple((d, i) for d, i in zip(dims, inds) if ind_freqs[i] == 1)

    def outer_inds(self):
        """Actual, i.e. exterior, indices of this TensorNetwork.
        """
        return tuple(i for d, i in self.outer_dims_inds())

    def mangle_inner(self):
        index_map = {ind: rand_uuid() for ind in self.inner_inds()}
        self.reindex(index_map, inplace=True)

    @property
    def shape(self):
        """Actual, i.e. exterior, shape of this TensorNetwork.
        """
        return tuple(d for d, i in self.outer_dims_inds())

    def __getitem__(self, tags):
        """Get the single tensor uniquely associated with ``tags``.
        """
        try:
            names = self.tag_index[tags]
        except (KeyError, TypeError):
            names = functools.reduce(
                operator.and_, (self.tag_index[t] for t in tags))

        if len(names) != 1:
            raise KeyError("'TensorNetwork.__getitem__' is meant for a single "
                           "tensor only - found {} with tag(s) '{}'."
                           .format(len(names), tags))

        name, = names
        return self.tensor_index[name]

    def __setitem__(self, tags, tensor):
        """Set the single tensor uniquely associated with ``tags``.
        """
        try:
            names = self.tag_index[tags]
        except (KeyError, TypeError):
            names = functools.reduce(
                operator.and_, (self.tag_index[t] for t in tags))

        if len(names) != 1:
            raise KeyError("'TensorNetwork.__setitem__' is meant for a single "
                           "existing tensor only - found {} with tag(s) '{}'."
                           .format(len(names), tags))

        if not isinstance(tensor, Tensor):
            raise TypeError("Can only set value with a new 'Tensor'.")

        name, = names
        self.tensor_index[name] = tensor

    def __and__(self, other):
        """Combine this tensor network with more tensors, without contracting.
        """
        return TensorNetwork((self, other))

    def __matmul__(self, other):
        """Overload "@" to mean full contraction with another network.
        """
        return TensorNetwork((self, other)) ^ ...

    # ------------------------------ printing ------------------------------- #

    def __repr__(self):
        return "TensorNetwork({}, {}, {})".format(
            repr(self.tensors),
            "contract_strategy='{}'".format(self.contract_strategy) if
            self.contract_strategy is not None else "",
            "nsites={}".format(self.nsites) if
            self.nsites is not None else "")

    def __str__(self):
        return "TensorNetwork([{}{}{}]{}{})".format(
            os.linesep,
            "".join(["    " + repr(t) + "," + os.linesep
                     for t in self.tensors[:-1]]),
            "    " + repr(self.tensors[-1]) + "," + os.linesep,
            ", contract_strategy='{}'".format(self.contract_strategy) if
            self.contract_strategy is not None else "",
            ", nsites={}".format(self.nsites) if
            self.nsites is not None else "")


# --------------------------------------------------------------------------- #
#                          Specific forms on network                          #
# --------------------------------------------------------------------------- #


def make_site_strs(x, nsites):
    """Get a range of site inds or tags based.
    """
    if isinstance(x, str):
        x = tuple(map(x.format, range(nsites)))
    else:
        x = tuple(x)

    return x


class MatrixProductState(TensorNetwork):
    """Initialise a matrix product state, with auto labelling and tagging.

    Parameters
    ----------
    *arrays : sequence of arrays
        The tensor arrays to form into a MPS.
    shape : str, optional
        String specifying layout of the tensors. E.g. 'lrp' (the default)
        indicates the shape corresponds left-bond, right-bond, physical index.
        End tensors have either 'l' or 'r' dropped from the string.
    site_inds : sequence of hashable, or str
        The indices to label the physical dimensions with, if a string is
        supplied, use this to format the indices thus:
        ``map(site_inds.format, range(len(arrays)))``.
        Defaults ``'k0', 'k1', 'k2'...``.
    site_tags : sequence of hashable, or str
        The tags to label each site with, if a string is supplied, use this to
        format the indices thus: ``map(site_tags.format, range(len(arrays)))``.
        Defaults ``'i0', 'i1', 'i2'...``.
    tags : str or sequence of hashable, optional
        Global tags to attach to all tensors.
    bond_name : str, optional
        The base name of the bond indices, onto which uuids will be added.
    """

    def __init__(self, arrays, shape='lrp', site_inds='k{}', site_tags='i{}',
                 tags=None, bond_name="", **kwargs):
        # short-circuit for copying MPSs
        if isinstance(arrays, MatrixProductState):
            super().__init__(arrays)
            self.site_inds = copy.copy(arrays.site_inds)
            self.site_tags = copy.copy(arrays.site_tags)
            return

        arrays = tuple(arrays)
        nsites = len(arrays)

        # process site indices
        self.site_inds = site_inds
        site_inds = make_site_strs(site_inds, nsites)

        # process site tags
        self.site_tags = site_tags
        contract_strategy = site_tags
        site_tags = make_site_strs(site_tags, nsites)

        if tags is not None:
            if isinstance(tags, str):
                tags = {tags}
            else:
                tags = set(tags)

            site_tags = tuple({st} | tags for st in site_tags)

        # TODO: figure out cyclic or not
        # TODO: allow open ends non-cyclic

        # transpose arrays to 'lrp' order.
        lp_ord = tuple(shape.replace('r', "").find(x) for x in 'lp')
        lrp_ord = tuple(shape.find(x) for x in 'lrp')
        rp_ord = tuple(shape.replace('l', "").find(x) for x in 'rp')

        # Do the first tensor seperately.
        next_bond = rand_uuid(base=bond_name)
        tensors = [Tensor(array=arrays[0].transpose(*lp_ord),
                          inds=[next_bond, site_inds[0]],
                          tags=site_tags[0])]
        previous_bond = next_bond

        # Range over the middle tensors
        for array, site_ind, site_tag in zip(arrays[1:-1], site_inds[1:-1],
                                             site_tags[1:-1]):
            next_bond = rand_uuid(base=bond_name)
            tensors.append(Tensor(array=array.transpose(*lrp_ord),
                                  inds=[previous_bond, next_bond, site_ind],
                                  tags=site_tag))
            previous_bond = next_bond

        # Do the last tensor seperately.
        tensors.append(Tensor(array=arrays[-1].transpose(*rp_ord),
                              inds=[previous_bond, site_inds[-1]],
                              tags=site_tags[-1]))

        super().__init__(tensors, contract_strategy=contract_strategy,
                         nsites=nsites, check_collisions=False, **kwargs)

    def reindex_sites(self, pattern, where=None, inplace=False):
        """Update the physical site index labels to a new pattern.

        Parameters
        ----------
        new_site_inds : str
            A string with a format placeholder to accept an int, e.g. "ket{}".
        where : None or slice
            Which sites to update the index labels on. If ``None`` (default)
            all sites.
        inplace : bool
            Whether to reindex in place.
        """
        if where is None:
            start = 0
            stop = self.nsites
        else:
            start = 0 if where.start is None else where.start
            stop = self.nsites if where.stop is ... else where.stop

        return self.reindex({self.site_inds.format(i): pattern.format(i)
                             for i in range(start, stop)}, inplace=inplace)

    def set_site_inds(self, new_site_inds):
        """Reindex all site indices, and set this as the site_inds.
        """
        self.reindex_sites(new_site_inds, inplace=True)
        self.site_inds = new_site_inds

    def left_canonize_site(self, i, bra=None):
        """Left canonize this MPS' ith site, inplace.

        Parameters
        ----------
        i : int
            Which site to canonize. The site at i + 1 also absorbs the
            non-isometric part of the decomposition of site i.
        bra : None or MatrixProductState, optional
            If given, simultaneously left canonize site i of this MPS, assuming
            it to hold the conjugate state.
        """
        T1 = self.site[i]
        T2 = self.site[i + 1]

        t1_inds_set = set(T1.inds)
        t2_inds_set = set(T2.inds)

        old_shared_bond, = t1_inds_set & t2_inds_set
        left_inds = t1_inds_set - t2_inds_set

        Q, R = T1.split(left_inds, method='qr', return_tensors=True)
        R = R @ T2

        new_shared_bond, = (j for j in Q.inds if j not in t1_inds_set)
        Q.reindex({new_shared_bond: old_shared_bond}, inplace=True)
        Q.transpose(*T1.inds, inplace=True)
        R.reindex({new_shared_bond: old_shared_bond}, inplace=True)
        R.transpose(*T2.inds, inplace=True)

        self.site[i]._data = Q._data
        self.site[i + 1]._data = R._data

        if bra is not None:
            bra.site[i]._data = Q._data.conj()
            bra.site[i + 1]._data = R._data.conj()

    def right_canonize_site(self, i, bra=None):
        """Right canonize this MPS' ith site, inplace.

        Parameters
        ----------
        i : int
            Which site to canonize. The site at i - 1 also absorbs the
            non-isometric part of the decomposition of site i.
        bra : None or MatrixProductState, optional
            If given, simultaneously right canonize site i of this MPS,
            assuming it to hold the conjugate state.
        """
        T1 = self.site[i]
        T2 = self.site[i - 1]

        t1_inds_set = set(T1.inds)
        t2_inds_set = set(T2.inds)

        left_inds = t1_inds_set & t2_inds_set
        old_shared_bond, = left_inds

        L, Q = T1.split(left_inds, method='lq', return_tensors=True)
        L = T2 @ L

        new_shared_bond, = (j for j in Q.inds if j not in t1_inds_set)
        L.reindex({new_shared_bond: old_shared_bond}, inplace=True)
        L.transpose(*T2.inds, inplace=True)
        Q.reindex({new_shared_bond: old_shared_bond}, inplace=True)
        Q.transpose(*T1.inds, inplace=True)

        self.site[i - 1]._data = L._data
        self.site[i]._data = Q._data

        if bra is not None:
            bra.site[i - 1]._data = L._data.conj()
            bra.site[i]._data = Q._data.conj()

    def left_canonize(self, start=None, stop=None, normalize=False, bra=None):
        """Left canonize all or a portion of this MPS, such that:

                          i            i
            o-o-o-o-o-o-o-o-         /-o-
            | | | | | | | | ...  ->  | | ...
            o-o-o-o-o-o-o-o-         \-o-

        Parameters
        ----------
        start : int, optional
            If given, the site to start left canonizing at.
        stop : int, optional
            If given, the site to stop left canonizing at.
        normalize : bool, optional
            Whether to normalize the state.
        bra : MatrixProductState, optional
            If supplied, simultaneously left canonize this MPS too, assuming it
            to be the conjugate state.
        """
        if start is None:
            start = 0
        if stop is None:
            stop = self.nsites - 1

        for i in range(start, stop):
            self.left_canonize_site(i, bra=bra)

        if normalize:
            factor = self.site[-1].norm()
            self.site[-1] /= factor
            if bra is not None:
                bra.site[-1] /= factor

    def right_canonize(self, start=None, stop=None, normalize=False, bra=None):
        """Right canonize all or a portion of this MPS, such that:

                 i                          i
                -o-o-o-o-o-o-o-o           -o-\
            ...  | | | | | | | |   ->  ...  | |
                -o-o-o-o-o-o-o-o           -o-/


        Parameters
        ----------
        start : int, optional
            If given, the site to start right canonizing at.
        stop : int, optional
            If given, the site to stop right canonizing at.
        normalize : bool, optional
            Whether to normalize the state.
        bra : MatrixProductState, optional
            If supplied, simultaneously right canonize this MPS too, assuming
            it to be the conjugate state.
        """
        if start is None:
            start = self.nsites - 1
        if stop is None:
            stop = 0

        for i in range(start, stop, -1):
            self.right_canonize_site(i, bra=bra)

        if normalize:
            factor = self.site[0].norm()
            self.site[0] /= factor
            if bra is not None:
                bra.site[0] /= factor

    def canonize(self, orthogonality_center, bra=None):
        """Mixed canonize this MPS.

        Parameters
        ----------
        orthogonality_center : int, optional
            Which site to orthogonalize around.
        bra : MatrixProductState, optional
            If supplied, simultaneously mixed canonize this MPS too, assuming
            it to be the conjugate state.
        """
        self.left_canonize(stop=orthogonality_center, bra=bra)
        self.right_canonize(stop=orthogonality_center, bra=bra)

    def shift_orthogonality_center(self, current, new):
        """Move the orthogonality center of this MPS.

        Parameters
        ----------
        current : int
            The current orthogonality center.
        new : int
            The target orthogonality center.
        """
        if new > current:
            for i in range(current, new):
                self.left_canonize_site(i)
        else:
            for i in range(current, new, -1):
                self.right_canonize_site(i)


class MatrixProductOperator(TensorNetwork):
    """Initialise a matrix product operator, with auto labelling and tagging.

    Parameters
    ----------
    *arrays : sequence of arrays
        The tensor arrays to form into a MPO.
    shape : str, optional
        String specifying layout of the tensors. E.g. 'lrkb' (the default)
        indicates the shape corresponds left-bond, right-bond, ket physical
        index, bra physical index.
        End tensors have either 'l' or 'r' dropped from the string.
    ket_site_inds : sequence of hashable, or str
        The indices to label the ket physical dimensions with, if a string is
        supplied, use this to format the indices thus:
        ``map(ket_site_inds.format, range(len(arrays)))``.
        Defaults ``'k0', 'k1', 'k2'...``.
    bra_site_inds : sequence of hashable, or str
        The indices to label the ket physical dimensions with, if a string is
        supplied, use this to format the indices thus:
        ``map(bra_site_inds.format, range(len(arrays)))``.
        Defaults ``'b0', 'b1', 'b2'...``.
    site_tags : sequence of hashable, or str
        The tags to label each site with, if a string is supplied, use this to
        format the indices thus: ``map(site_tags.format, range(len(arrays)))``.
        Defaults ``'i0', 'i1', 'i2'...``.
    tags : str or sequence of hashable, optional
        Global tags to attach to all tensors.
    bond_name : str, optional
        The base name of the bond indices, onto which uuids will be added.
    """

    def __init__(self, arrays, shape='lrkb', site_tags='i{}', tags=None,
                 ket_site_inds='k{}', bra_site_inds='b{}', bond_name="",
                 **kwargs):
        # short-circuit for copying
        if isinstance(arrays, MatrixProductOperator):
            super().__init__(arrays)
            self.ket_site_inds = copy.copy(arrays.ket_site_inds)
            self.bra_site_inds = copy.copy(arrays.bra_site_inds)
            self.site_tags = copy.copy(arrays.site_tags)
            return

        arrays = tuple(arrays)
        nsites = len(arrays)

        # process site indices
        self.ket_site_inds = ket_site_inds
        self.bra_site_inds = bra_site_inds
        ket_site_inds = make_site_strs(ket_site_inds, nsites)
        bra_site_inds = make_site_strs(bra_site_inds, nsites)

        # process site tags
        self.site_tags = site_tags
        contract_strategy = site_tags
        site_tags = make_site_strs(site_tags, nsites)
        if tags is not None:
            if isinstance(tags, str):
                tags = (tags,)
            else:
                tags = tuple(tags)

            site_tags = tuple((st,) + tags for st in site_tags)

        # transpose arrays to 'lrkb' order.
        lkb_ord = tuple(map(lambda x: shape.replace('r', "").find(x), 'lkb'))
        rkb_ord = tuple(map(lambda x: shape.replace('l', "").find(x), 'rkb'))
        lrkb_ord = tuple(map(shape.find, 'lrkb'))

        # Do the first tensor seperately.
        next_bond = rand_uuid(base=bond_name)
        tensors = [Tensor(array=arrays[0].transpose(*lkb_ord),
                          inds=[next_bond, ket_site_inds[0], bra_site_inds[0]],
                          tags=site_tags[0])]
        previous_bond = next_bond

        # Range over the middle tensors
        for array, ksi, bsi, site_tag in zip(arrays[1:-1],
                                             ket_site_inds[1:-1],
                                             bra_site_inds[1:-1],
                                             site_tags[1:-1]):

            next_bond = rand_uuid(base=bond_name)
            tensors += [Tensor(array=array.transpose(*lrkb_ord),
                               inds=[previous_bond, next_bond, ksi, bsi],
                               tags=site_tag)]
            previous_bond = next_bond

        # Do the last tensor seperately.
        tensors.append(Tensor(array=arrays[-1].transpose(*rkb_ord),
                              inds=[previous_bond,
                                    ket_site_inds[-1],
                                    bra_site_inds[-1]],
                              tags=site_tags[-1]))

        super().__init__(tensors, contract_strategy=contract_strategy,
                         nsites=nsites, check_collisions=False, **kwargs)


# --------------------------------------------------------------------------- #
#                        Specific states and operators                        #
# --------------------------------------------------------------------------- #


def rand_tensor(shape, inds, tags=None):
    """Generate a random (complex) tensor with specified shape.
    """
    array = np.random.randn(*shape) + 1.0j * np.random.randn(*shape)
    return Tensor(array=array, inds=inds, tags=tags)


def MPS_rand(n, bond_dim, phys_dim=2,
             site_inds='k{}',
             site_tags='i{}',
             tags=None,
             bond_name="",
             normalize=True,
             **kwargs):
    """Generate a random matrix product state.

    Parameters
    ----------
    bond_dim : int
        The bond dimension.
    phys_dim : int, optional
        The physical (site) dimensions, defaults to 2.
    site_inds : sequence of hashable, or str
        See :func:`matrix_product_state`.
    site_tags=None, optional
        See :func:`matrix_product_state`.
    tags=None, optional
        See :func:`matrix_product_state`.
    bond_name : str, optional
        See :func:`matrix_product_state`.
    """
    shapes = [(bond_dim, phys_dim),
              *((bond_dim, bond_dim, phys_dim),) * (n - 2),
              (bond_dim, phys_dim)]

    arrays = \
        map(lambda x: x / norm_fro_dense(x)**(1 / (x.ndim - 1)),
            map(lambda x: np.random.randn(*x) + 1.0j * np.random.randn(*x),
                shapes))

    rmps = MatrixProductState(arrays, site_inds=site_inds,
                              bond_name=bond_name, site_tags=site_tags,
                              tags=tags, **kwargs)

    if normalize:
        c = (rmps.H @ rmps)**0.5
        rmps[rmps.contract_strategy.format(n - 1)] /= c

    return rmps


@functools.lru_cache(128)
def mpo_site_ham_heis(j=1.0, bz=0.0):
    """Single site of the spin-1/2 Heisenberg Hamiltonian in MPO form.
    This is cached.

    Parameters
    ----------
    j : float
        (Isotropic) nearest neighbour coupling.
    bz : float
        Magnetic field in Z-direction.

    Returns
    -------
    H : numpy.ndarray
        The tensor, with shape (5, 5, 2, 2).
    """
    H = np.zeros((5, 5, 2, 2), dtype=complex)

    for i, s in enumerate('XYZ'):
        H[1 + i, 0, :, :] = spin_operator(s)
        H[-1, 1 + i, :, :] = j * spin_operator(s)

    H[0, 0, :, :] = eye(2)
    H[4, 4, :, :] = eye(2)
    H[4, 0, :, :] = - bz * spin_operator('Z')

    make_immutable(H)
    return H


def mpo_end_ham_heis_left(j=1.0, bz=0.0):
    """The left most site of a open boundary conditions Heisenberg
    matrix product operator.
    """
    return mpo_site_ham_heis(j=j, bz=bz)[-1, :, :, :]


def mpo_end_ham_heis_right(j=1.0, bz=0.0):
    """The right most site of a open boundary conditions Heisenberg
    matrix product operator.
    """
    return mpo_site_ham_heis(j=j, bz=bz)[:, 0, :, :]


def MPO_ham_heis(n, j=1.0, bz=0.0,
                 ket_site_inds='k{}',
                 bra_site_inds='b{}',
                 site_tags='i{}',
                 tags=None,
                 bond_name=""):
    """Heisenberg Hamiltonian in matrix product operator form.
    """
    arrays = (mpo_end_ham_heis_left(j=j, bz=bz),
              *[mpo_site_ham_heis(j=j, bz=bz)] * (n - 2),
              mpo_end_ham_heis_right(j=j, bz=bz))

    HH_mpo = MatrixProductOperator(arrays=arrays,
                                   ket_site_inds=ket_site_inds,
                                   bra_site_inds=bra_site_inds,
                                   site_tags=site_tags,
                                   tags=tags,
                                   bond_name=bond_name)
    return HH_mpo


def update_with_eff_gs(energy_tn, k, b, i):
    """Find the effective tensor groundstate of:

                  /|\
        o-o-o-o-o- | -o-o-o-o-o-o-o-o
        | | | | |  |  | | | | | | | |
        O-O-O-O-O--O--O-O-O-O-O-O-O-O
        | | | | | i|  | | | | | | | |
        o-o-o-o-o- | -o-o-o-o-o-o-o-o
                  \|/

    And insert it back into the states ``k`` and ``b``, and thus ``energy_tn``.
    """
    eff_ham = (energy_tn ^ slice(0, i) ^ slice(..., i) ^ '__ham__')['__ham__']
    eff_ham.fuse((('lower', b.site[i].inds),
                  ('upper', k.site[i].inds)), inplace=True)
    eff_e, eff_gs = spla.eigs(eff_ham.data, k=1)
    k.site[i].data = eff_gs
    b.site[i].data = eff_gs.conj()
    return eff_e


def dmrg1_sweep(energy_tn, k, b, direction, canonize=True):
    """
    """
    if canonize and direction == 'right':
        k.right_canonize(bra=b)
    elif canonize and direction == 'left':
        k.left_canonize(bra=b)

    if direction == 'right':
        for i in range(0, k.nsites):
            eff_e = update_with_eff_gs(energy_tn, k, b, i)
            if i < k.nsites - 1:
                k.left_canonize_site(i, bra=b)

    elif direction == 'left':
        for i in reversed(range(0, k.nsites)):
            eff_e = update_with_eff_gs(energy_tn, k, b, i)
            if i > 0:
                k.right_canonize_site(i, bra=b)

    return eff_e


def dmrg1(ham, bond_dim, num_sweeps=4):
    ham.add_tag("__ham__")

    k = MPS_rand(ham.nsites, bond_dim)
    k.add_tag("__ket__")

    b = k.H
    b.set_site_inds(ham.bra_site_inds)
    b.add_tag("__bra__")

    energy_tn = (b & ham & k)

    for _ in range(num_sweeps):
        eff_e = dmrg1_sweep(energy_tn, k, b, direction='right')

    return eff_e, k
