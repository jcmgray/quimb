"""Tensor network tools.
"""

import os
import functools
from functools import reduce
import operator
from operator import or_

from cytoolz import concat, frequencies, groupby
import numpy as np

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

def tensor_transpose(tensor, output_inds):
    """Transpose a tensor.

    Parameters
    ----------
    tensor : Tensor
        The tensor to transpose.
    output_inds : sequence of hashable
        The desired output sequence of indices.

    Returns
    -------
    Tensor
    """
    output_inds = tuple(output_inds)  # need to re-use this.

    if set(tensor.inds) != set(output_inds):
        raise ValueError("'output_inds' must be permutation of the "
                         "current tensor indices.")

    current_ind_map = {ind: i for i, ind in enumerate(tensor.inds)}
    out_shape = map(current_ind_map.__getitem__, output_inds)

    return Tensor(tensor.array.transpose(*out_shape),
                  inds=output_inds, tags=tensor.tags)


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
    a_ix : sequence
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
    a_ix = sorted(set(a_ix))

    if any(i not in _einsum_symbols_set for i in a_ix):
        # need to map inds to alphabet

        if len(a_ix) > len(_einsum_symbols_set):
            raise ValueError("Too many indices to auto-optimize contraction "
                             "for at once, try setting a `contract_strategy` "
                             "or manual contraction order using tags.")

        amap = {i: lett for i, lett in zip(a_ix, _einsum_symbols)}
        in_str = map(lambda x: "".join(map(amap.__getitem__, x)), i_ix)
        out_str = "".join(map(amap.__getitem__, o_ix))

    else:
        in_str = map("".join, i_ix)
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


def tensor_contract(*tensors,
                    memory_limit=2**28,
                    optimize='greedy',
                    output_inds=None):
    """Efficiently contract multiple tensors, combining their tags.

    Parameters
    ----------
    *tensors : sequence of Tensor
        The tensors to contract.
    memory_limit : int, optional
        See :py:func:`contract`.
    optimize : str, optional
        See :py:func:`contract`.
    output_inds : sequence
        If given, the desired order of output indices, else defaults to the
        order they occur in the input indices.

    Returns
    -------
    scalar or Tensor
    """
    i_ix = [t.inds for t in tensors]  # input indices per tensor
    a_ix = list(concat(i_ix))  # list of all input indices

    if output_inds is None:
        # sort output indices  by input order for efficiency and consistency
        o_ix = tuple(filter(lambda x: x in set(_gen_output_inds(a_ix)), a_ix))
    else:
        o_ix = output_inds

    # possibly map indices into the 0-52 range needed by einsum
    contract_str = _maybe_map_indices_to_alphabet(a_ix, i_ix, o_ix)

    # perform the contraction
    path = cache_einsum_path_on_shape(contract_str,
                                      *(t.shape for t in tensors))

    o_array = einsum(contract_str, *(t.array for t in tensors),
                     optimize=path)

    if not o_ix:
        return o_array

    # unison of all tags
    o_tags = reduce(or_, (t.tags for t in tensors))

    return Tensor(array=o_array, inds=o_ix, tags=o_tags)


def _gen_rand_uuids():
    """Generate shortish identifiers which are guaranteed unique.
    Will break if more than ~10**12.4 required.
    """
    import uuid
    used = set()
    while True:
        s = str(uuid.uuid4())[:8]
        while s in used:  # pragma: no cover
            s = str(uuid.uuid4())[:8]
        used.add(s)
        yield s


RAND_UUIDS = iter(_gen_rand_uuids())


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
    Q, L = np.linalg.qr(dag(x))
    return dag(L), dag(Q)


class BondError(Exception):
    pass


def tensor_split(tensor, left_inds, bond_name="", method='svd',
                 tol=None, relative=False, max_bond=None):
    """Decompose a tensor into two tensors.

    Parameters
    ----------
    tensor : Tensor
        The tensor to split.
    left_inds : sequence of hashable
        The sequence of inds, which ``tensor`` should already have, to split
    bond_name : str, optional
        The base name to give to the bond between the two resulting tensors.

    Returns
    -------
    TensorNetwork
    """
    left_inds = tuple(left_inds)
    right_inds = tuple(filter(lambda x: x not in left_inds, tensor.inds))

    TT = tensor.transpose(*left_inds, *right_inds)

    left_dims = TT.shape[:len(left_inds)]
    right_dims = TT.shape[len(left_inds):]

    Tf = tensor.fuse({'left': left_inds, 'right': right_inds})

    if tol is None:
        tol = -1.0

    left, right = {'svd': _array_split_svd,
                   'eig': _array_split_eig,
                   'qr': _array_split_qr,
                   'lq': _array_split_lq}[method](Tf.array, tol=tol)

    left = left.reshape(*left_dims, -1)
    right = right.reshape(-1, *right_dims)

    if max_bond is not None:
        if left.shape[-1] > max_bond:
            raise BondError("Maximum bond size exceeded")

    bond_ind = rand_uuid(base=bond_name)

    return TensorNetwork(
        Tensor(array=left, inds=(*left_inds, bond_ind), tags=tensor.tags),
        Tensor(array=right, inds=(bond_ind, *right_inds), tags=tensor.tags),
        check_collisions=False,
    )


# --------------------------------------------------------------------------- #
#                                Tensor Class                                 #
# --------------------------------------------------------------------------- #

class Tensor(object):
    """A labelled, tagged ndarray.

    Parameters
    ----------
    array : numpy.ndarray
        The n-dimensions data.
    inds : sequence of hashable
        The index labels for each dimension.
    tags : sequence of hashable
        Tags with which to select and filter from multiple tensors.
    """

    def __init__(self, array, inds, tags=None):
        self.array = np.asarray(array)
        self.inds = tuple(inds)

        if self.array.ndim != len(self.inds):
            raise ValueError(
                "Wrong number of inds, {}, supplied for array"
                " of shape {}.".format(self.inds, self.array.shape))

        self.tags = (set() if tags is None else
                     {tags} if isinstance(tags, str) else
                     set(tags))

    def conj(self):
        """Conjugate this tensors data (does nothing to indices).
        """
        return Tensor(self.array.conj(), self.inds, self.tags)

    @property
    def H(self):
        """Conjugate this tensors data (does nothing to indices).
        """
        return self.conj()

    @property
    def shape(self):
        return self.array.shape

    @property
    def size(self):
        return self.array.size

    @property
    def inner_inds(self):
        ind_freqs = frequencies(self.inds)
        return tuple(filter(lambda i: ind_freqs[i] == 2, self.inds))

    @functools.wraps(tensor_transpose)
    def transpose(self, *output_inds):
        return tensor_transpose(self, output_inds)

    @functools.wraps(tensor_contract)
    def contract(self, *others, memory_limit=2**28, optimize='greedy',
                 output_inds=None):
        return tensor_contract(self, *others, memory_limit=memory_limit,
                               optimize=optimize, output_inds=output_inds)

    @functools.wraps(tensor_split)
    def split(self, left_inds, bond_name="", method='svd',
              tol=None, relative=False, max_bond=None):
        return tensor_split(self, left_inds=left_inds, bond_name=bond_name,
                            method=method, tol=tol, relative=relative,
                            max_bond=max_bond)

    def reindex(self, index_map, inplace=False):
        """Rename the indices of this tensor, optionally in-place.

        Parameters
        ----------
        index_map : dict-like
            Mapping of pairs ``{old_ind: new_ind, ...}``.
        """
        new_inds = tuple(index_map.get(ind, ind) for ind in self.inds)
        if inplace:
            self.inds = new_inds
            return self
        else:
            return Tensor(array=self.array, inds=new_inds, tags=self.tags)

    def fuse(self, fuse_map):
        """Combine groups of indices into single indices.

        Parameters
        ----------
        fuse_map : dict_like
            Mapping like: ``{new_ind: sequence of existing inds, ...}``.

        Returns
        -------
        Tensor
            The transposed, reshaped and re-labeled tensor.
        """
        fuseds = tuple(fuse_map.values())  # groups of indices to be fused
        unfused_inds = tuple(i for i in self.inds if not
                             any(i in fs for fs in fuseds))

        # transpose tensor to bring groups of fused inds to the beginning
        TT = self.transpose(*concat(fuseds), *unfused_inds)

        # for each set of fused dims, group into product, then add remaining
        dims = iter(TT.shape)
        dims = [prod(next(dims) for _ in fs) for fs in fuseds] + list(dims)

        # create new tensor with new + remaining indices
        new_inds = concat((fuse_map.keys(), unfused_inds))
        return Tensor(TT.array.reshape(*dims), new_inds, self.tags)

    def almost_equals(self, other, **kwargs):
        """Check if this tensor is almost the same as another.
        """
        same_inds = (set(self.inds) == set(other.inds))
        if not same_inds:
            return False
        otherT = other.transpose(*self.inds)
        return np.allclose(self.array, otherT.array, **kwargs)

    def __and__(self, other):
        """Combine with another ``Tensor`` or ``TensorNetwork`` into a new
        ``TensorNetwork``.
        """
        return TensorNetwork(self, other)

    def __matmul__(self, other):
        """Explicitly contract with another tensor.
        """
        return tensor_contract(self, other)

    def __repr__(self):
        return "Tensor(shape={}, inds={}, tags={})".format(
            self.array.shape,
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
                array=op(self.array, otherT.array), inds=self.inds,
                tags=self.tags | other.tags)
        return Tensor(array=op(self.array, other),
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
        return Tensor(array=op(other, self.array),
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

class TensorNetwork(object):
    """A collection of (as yet uncontracted) Tensors.

    Parameters
    ----------
    *tensors : sequence of Tensor or TensorNetwork
        The objects to combine.
    contract_strategy : str, optional
        A string, with integer format specifier, that describes how to range
        over the network's tags in order to contract it.
    nsites : int, optional
        The number of sites, if explicitly known. This will be calculated
        using `contract_strategy` if needed but not specified.
    check_collisions : bool, optional
        If True, the default, then Tensors and TensorNetworks with double
        indices which match another Tensor or TensorNetworks double indices
        will have those indices' names mangled. Should be explicily turned off
        when it is known that no collisions will take place -- i.e. when not
        adding any new tensors.

    Members
    -------
    tensors : sequence of Tensor
        The tensors in this network.
    """

    def __init__(self, *tensors, contract_strategy=None, nsites=None,
                 check_collisions=True):

        self.tensors = []
        self.contract_strategy = contract_strategy
        self.nsites = nsites
        current_inner_inds = set()

        for i, t in enumerate(tensors):

            istensor = isinstance(t, Tensor)
            istensornetwork = isinstance(t, TensorNetwork)

            if not (istensor or istensornetwork):
                raise TypeError("TensorNetwork should be called as "
                                "``TensorNetwork(*tensors)``, where all "
                                "arguments are Tensors or TensorNetworks.")

            if check_collisions:
                # check for matching inner_indices -> need to re-index
                new_inner_inds = set(t.inner_inds)
                if current_inner_inds & new_inner_inds:
                    t = t.reindex({old: old + "-{}".format(i)
                                   for old in new_inner_inds})
                current_inner_inds |= new_inner_inds

            if istensor:
                self.tensors.append(t)
            else:  # assume TensorNetwork
                if t.contract_strategy is not None:
                    # check whether to inherit ...
                    if self.contract_strategy is None:
                        self.contract_strategy = t.contract_strategy
                    # ... or compare contract_strategies
                    else:
                        if t.contract_strategy != self.contract_strategy:
                            raise ValueError(
                                "Conflicting contraction strategies found on "
                                "tensor networks.")

                self.tensors += t.tensors

        # build a map to efficiently locate tensors.
        self.tag_map = self.calc_tag_map()

        # count how many sites if a contract_strategy is given
        if (self.contract_strategy) and (self.nsites is None):
            self.nsites = self.calc_nsites()

    def calc_tag_map(self):
        """Make a dict which maps tags to a list of tensors indices.
        """
        tag_map = {}
        for i, t in enumerate(self.tensors):
            for tag in t.tags:
                if tag in tag_map:
                    tag_map[tag].append(i)
                else:
                    tag_map[tag] = [i]
        return tag_map

    def calc_nsites(self):
        """Calculate how many tags there are which match ``contract_strategy``.
        """
        nsites = 0
        for i in range(10000):
            if self.contract_strategy.format(i) in self.tag_map:
                nsites += 1
            else:
                break
        return nsites

    def filter_by_tags(self, tags):
        """Split this TN into a list of tensors containing any of ``tags`` and
        the rest.

        Parameters
        ----------
        tags : sequence of hashable
            The list of tags to filter the tensors by.

        Returns
        -------
        (untagged, tagged) : (Tensor sequence, Tensor sequence)
            A pair of lists with the untagged and tagged tensors.
        """
        # contract all
        if tags is ...:
            return [], self.tensors

        # Else get the locations of where each tag is found on tensor
        if isinstance(tags, str):
            tagged_locs = set(self.tag_map[tags])
        else:
            tagged_locs = set(concat(map(self.tag_map.__getitem__, tags)))

        # Split a (loc, Tensor) list into tagged and untagged groups
        groups = groupby(lambda x: x[0] in tagged_locs,
                         enumerate(self.tensors))

        # Filter out the locations themselves
        untagged = [i[1] for i in groups.pop(False, [])]
        tagged = [i[1] for i in groups.pop(True, [])]

        return untagged, tagged

    def contract(self, tags=...):
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

        # Check for a structured strategy for performing contraction.
        if self.contract_strategy is not None:
            if tags is ...:  # all sites
                tags = slice(0, self.nsites)

            if isinstance(tags, slice):
                return self >> (self.contract_strategy.format(i) for i in
                                range(tags.start, tags.stop,
                                      1 if tags.step is None else tags.step))

        # Else just contract those tensors specified by tags.
        untagged, tagged = self.filter_by_tags(tags)

        if not tagged:
            raise ValueError("No tags were found - nothing to contract. "
                             "(Change this to a no-op maybe?)")

        if untagged:
            return TensorNetwork(tensor_contract(*tagged), *untagged,
                                 contract_strategy=self.contract_strategy,
                                 nsites=self.nsites, check_collisions=False)

        return tensor_contract(*tagged)

    def cumulative_contract(self, tags_seq):
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
        new_tn = self
        ctags = set()

        for tags in tags_seq:
            # accumulate tags from each contractions
            ctags |= {tags} if isinstance(tags, str) else set(tags)
            # peform the next contraction
            new_tn ^= ctags

        return new_tn

    def reindex(self, index_map, inplace=False):
        """Rename indices for all tensors in this network, optionally in-place.

        Parameters
        ----------
        index_map : dict-like
            Mapping of pairs ``{old_ind: new_ind, ...}``.
        """
        if inplace:
            for t in self.tensors:
                t.reindex(index_map, inplace=True)
            return self
        else:
            return TensorNetwork(*(t.reindex(index_map) for t in self.tensors),
                                 contract_strategy=self.contract_strategy,
                                 nsites=self.nsites, check_collisions=False)

    def conj(self):
        """Conjugate all the tensors in this network (leaves all indices).
        """
        return TensorNetwork(*[t.conj() for t in self.tensors],
                             contract_strategy=self.contract_strategy,
                             nsites=self.nsites, check_collisions=False)

    @property
    def H(self):
        """Conjugate all the tensors in this network (leaves all indices).
        """
        return self.conj()

    @property
    def all_dims_inds(self):
        """Return a list of all dimensions, and the corresponding list of
        indices from the tensor network.
        """
        return zip(*concat(zip(t.shape, t.inds) for t in self.tensors))

    @property
    def all_inds(self):
        return tuple(concat(t.inds for t in self.tensors))

    @property
    def inner_inds(self):
        """Return all inner indices, that is, those that appear twice.
        """
        all_inds = self.all_inds
        ind_freqs = frequencies(all_inds)
        return tuple(filter(lambda i: ind_freqs[i] == 2, all_inds))

    @property
    def outer_dims_inds(self):
        """Get the 'outer' pairs of dimension and indices, i.e. as if this
        tensor network was fully contracted.
        """
        dims, inds = self.all_dims_inds
        ind_freqs = frequencies(inds)
        return tuple((d, i) for d, i in zip(dims, inds) if ind_freqs[i] == 1)

    @property
    def outer_inds(self):
        """Actual, i.e. exterior, shape of this TensorNetwork.
        """
        return tuple(i for d, i in self.outer_dims_inds)

    @property
    def shape(self):
        """Actual, i.e. exterior, shape of this TensorNetwork.
        """
        return tuple(d for d, i in self.outer_dims_inds)

    @property
    def tags(self):
        return reduce(or_, (t.tags for t in self.tensors))

    def __xor__(self, *args, **kwargs):
        """Overload of '^' for TensorNetwork.contract.
        """
        return self.contract(*args, **kwargs)

    def __ixor__(self, *args, **kwargs):
        """Overload of '^=' for inplace TensorNetwork.contract.
        """
        return self.contract(*args, **kwargs)

    def __rshift__(self, *args, **kwargs):
        """Overload of '>>' for TensorNetwork.cumulative_contract.
        """
        return self.cumulative_contract(*args, **kwargs)

    def __irshift__(self, *args, **kwargs):
        """Overload of '>>=' for inplace TensorNetwork.cumulative_contract.
        """
        return self.cumulative_contract(*args, **kwargs)

    def __and__(self, other):
        """Combine this tensor network with more tensors, without contracting.
        """
        return TensorNetwork(self, other)

    def __matmul__(self, other):
        """Overload "@" to mean full contraction with another network.
        """
        return TensorNetwork(self, other) ^ ...

    def __repr__(self):
        return "TensorNetwork({}, {}, {})".format(
            repr(self.tensors),
            "contract_strategy='{}'".format(self.contract_strategy) if
            self.contract_strategy is not None else "",
            "nsites={}".format(self.nsites) if
            self.nsites is not None else "",
        )

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


def matrix_product_state(*arrays,
                         shape='lrp',
                         site_inds=None,
                         site_tags=None,
                         tags=None,
                         bond_name="",):
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

    Returns
    -------
    TensorNetwork
    """
    arrays = tuple(arrays)
    nsites = len(arrays)

    # process site indices
    site_inds = 'k{}' if site_inds is None else site_inds
    site_inds = make_site_strs(site_inds, nsites)

    # process site tags
    site_tags = 'i{}' if site_tags is None else site_tags
    contract_strategy = site_tags
    site_tags = make_site_strs(site_tags, nsites)

    if tags is not None:
        if isinstance(tags, str):
            tags = (tags,)
        else:
            tags = tuple(tags)

        site_tags = tuple((st,) + tags for st in site_tags)

    # TODO: figure out cyclic or not
    # TODO: allow open ends non-cyclic

    lp_ord = tuple(map(lambda x: shape.replace('r', "").find(x), 'lp'))
    rp_ord = tuple(map(lambda x: shape.replace('l', "").find(x), 'rp'))
    lrp_ord = tuple(map(shape.find, 'lrp'))  # transpose arrays to 'lrp' order.

    # Do the first tensor seperately.
    next_bond = rand_uuid(base=bond_name)
    tensors = [Tensor(array=arrays[0].transpose(*lp_ord),
                      inds=[next_bond, site_inds[0]], tags=site_tags[0])]
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

    return TensorNetwork(*tensors, contract_strategy=contract_strategy,
                         nsites=nsites, check_collisions=False)


def matrix_product_operator(*arrays, shape='lrkb',
                            ket_site_inds=None,
                            bra_site_inds=None,
                            site_tags=None,
                            tags=None,
                            bond_name=""):
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

    Returns
    -------
    TensorNetwork
    """
    arrays = tuple(arrays)
    nsites = len(arrays)

    # process site indices
    ket_site_inds = 'k{}' if ket_site_inds is None else ket_site_inds
    ket_site_inds = make_site_strs(ket_site_inds, nsites)
    bra_site_inds = 'b{}' if bra_site_inds is None else bra_site_inds
    bra_site_inds = make_site_strs(bra_site_inds, nsites)

    # process site tags
    site_tags = 'i{}' if site_tags is None else site_tags
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

    return TensorNetwork(*tensors, contract_strategy=contract_strategy,
                         nsites=nsites, check_collisions=False)


# --------------------------------------------------------------------------- #
#                        Specific states and operators                        #
# --------------------------------------------------------------------------- #

def rand_ket_mps(n, bond_dim, phys_dim=2,
                 site_inds=None,
                 site_tags=None,
                 tags=None,
                 bond_name="",
                 normalize=True):
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

    rmps = matrix_product_state(*arrays, site_inds=site_inds,
                                bond_name=bond_name, site_tags=site_tags,
                                tags=tags)

    if normalize:
        c = (rmps.H @ rmps)**0.5
        rmps.tensors[n // 2] /= c

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


def ham_heis_mpo(n, j=1.0, bz=0.0,
                 ket_site_inds=None,
                 bra_site_inds=None,
                 site_tags=None,
                 tags=None,
                 bond_name=""):
    """Heisenberg Hamiltonian in matrix product operator form.
    """
    HH_mpo = matrix_product_operator(
        mpo_end_ham_heis_left(),
        *[mpo_site_ham_heis()] * (n - 2),
        mpo_end_ham_heis_right(),
        ket_site_inds=ket_site_inds,
        bra_site_inds=bra_site_inds,
        site_tags=site_tags,
        tags=tags,
        bond_name=bond_name,
    )
    return HH_mpo
