"""Core tensor network tools.
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

from ..accel import prod, njit
from ..linalg.base_linalg import norm_fro_dense

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
                             "for at once, try setting a `structure` "
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

    return Tensor(data=o_array, inds=o_ix, tags=o_tags)


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


def _array_split_svdvals(x):
    """SVD-decomposition, but return singular values only.
    """
    return np.linalg.svd(x, full_matrices=False, compute_uv=False)


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
        s2, V = np.linalg.eigh(dag(x) @ x)
        U = x @ dag(V)

    else:
        s2, U = np.linalg.eigh(x @ dag(x))
        V = dag(U) @ x

    if tol > 0.0:
        n_chi = np.sum(s2 > tol**2)
        U = U[..., :n_chi]
        V = V[:n_chi, ...]

    return U, V


@njit
def _array_split_svdvals_eig(x):
    """SVD-decomposition via eigen, but return singular values only.
    """
    if x.shape[0] > x.shape[1]:
        s2 = np.linalg.eigvalsh(dag(x) @ x)
    else:
        s2 = np.linalg.eigvalsh(x @ dag(x))
    return s2**0.5


@njit  # pragma: no cover
def _array_split_qr(x, tol):
    """QR-decomposition.
    """
    Q, R = np.linalg.qr(x)
    return Q, R


@njit  # pragma: no cover
def _array_split_lq(x, tol):
    """LQ-decomposition.
    """
    Q, L = np.linalg.qr(x.T)
    return L.T, Q.T


class BondError(Exception):
    pass


def array_direct_product(X, Y, sum_axes=()):
    """Direct product of two numpy.ndarrays.

    Parameters
    ----------
    X : numpy.ndarray
        First tensor.
    Y : numpy.ndarray
        Second tensor, same shape as ``X``.
    sum_axes : sequence of int
        Axes to sum over rather than direct product, e.g. physical indices when
        adding tensor networks.

    Returns
    -------
    Z : numpy.ndarray
        Same shape as ``X`` and ``Y``, but with every dimension doubled in size
        unless it is included in ``sum_axes``.
    """
    if X.shape != Y.shape:
        raise ValueError("Can only add tensors of the "
                         "same shape.")

    if isinstance(sum_axes, int):
        sum_axes = (sum_axes,)

    # parse the intermediate and final shape doubling the size of any axes that
    #   is not to be summed, and preparing slices with which to add X, Y.
    temp_shape = []
    final_shape = []
    selectorX = []
    selectorY = []
    for i, d in enumerate(X.shape):
        if i not in sum_axes:
            temp_shape.append(2)
            final_shape.append(2 * d)
            selectorX.append(0)
            selectorY.append(1)
        else:
            final_shape.append(d)
        temp_shape.append(d)
        selectorX.append(slice(None))
        selectorY.append(slice(None))

    new_type = np.find_common_type((X.dtype, Y.dtype), ())
    Z = np.zeros(temp_shape, dtype=new_type)

    # Add tensors to [0, :, 0, :, ...], [1, :, 1, :, ...] locations, i.e. the
    #   'diagonal' representing the temp doubled axes
    Z[selectorX] += X
    Z[selectorY] += Y

    # Reshape to the same number of original dimensions
    return Z.reshape(final_shape)


def tensor_direct_product(T1, T2, sum_inds=(), inplace=False):
    """Direct product of two Tensors. They must have the same shape, any axes
    included in ``sum_inds`` will be summed over rather than combined.
    Summing over contractions of TensorNetworks equates to contracting a
    TensorNetwork made of direct products of each set of tensors.
    I.e. (a1 @ b1) + (a2 @ b2) == (a1 (+) a2) @ (b1 (+) b2).

    Parameters
    ----------
    T1 : Tensor
        The first tensor.
    T2 : Tensor
        The second tensor, with matching indices and dimensions to ``T1``.
    sum_inds : sequence of hashable, optional
        Axes to sum over rather than combine, e.g. physical indices when
        adding tensor networks.
    inplace : bool, optional
        Whether to modify ``T1`` inplace.

    Returns
    -------
    Tensor
        Like ``T1``, but with each dimension doubled in size if not
        in ``sum_inds``.
    """
    if isinstance(sum_inds, (str, int)):
        sum_inds = (sum_inds,)

    if T2.inds != T1.inds:
        T2 = T2.transpose(*T1.inds)

    sum_axes = tuple(T1.inds.index(ind) for ind in sum_inds)

    if inplace:
        new_T = T1
    else:
        new_T = T1.copy()

    # XXX: add T2s tags?
    new_T._data = array_direct_product(T1.data, T2.data, sum_axes=sum_axes)
    return new_T


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
    def contract(self, *others, output_inds=None):
        return tensor_contract(self, *others, output_inds=output_inds)

    def split(self, left_inds, method='svd', tol=1e-13,
              max_bond=None, get=None):
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
        get : {None, 'tensors', 'values'}
            If given, what to return instead of the TensorNetwork describing
            the split.

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

        if get == 'values':
            return {'svd': _array_split_svdvals,
                    'eig': _array_split_svdvals_eig}[method](array)

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

        Tl = Tensor(data=left, inds=(*left_inds, bond_ind), tags=self.tags)
        Tr = Tensor(data=right, inds=(bond_ind, *right_inds), tags=self.tags)

        if get == 'tensors':
            return Tl, Tr

        return TensorNetwork((Tl, Tr), check_collisions=False)

    def singular_values(self, left_inds, method='svd'):
        """Return the singular values associated with splitting this tensor
        according to ``left_inds``.

        Parameters
        ----------
        left_inds : sequence of hashable
            A subset of this tensors indices that defines 'left'.
        method : {'svd', 'eig'}
            Whether to use the SVD or eigenvalue decomposition to get the
            singular values.

        Returns
        -------
        1d-array
            The singular values.
        """
        return self.split(left_inds=left_inds, method=method, get='values')

    def entropy(self, left_inds, method='svd'):
        """Return the entropy associated with splitting this tensor
        according to ``left_inds``.

        Parameters
        ----------
        left_inds : sequence of hashable
            A subset of this tensors indices that defines 'left'.
        method : {'svd', 'eig'}
            Whether to use the SVD or eigenvalue decomposition to get the
            singular values.

        Returns
        -------
        float
        """
        el = self.singular_values(left_inds=left_inds, method=method)**2
        el = el[el > 0.0]
        return np.sum(-el * np.log2(el))

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
                data=op(self.data, otherT.data), inds=self.inds,
                tags=self.tags | other.tags)
        return Tensor(data=op(self.data, other),
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
        return Tensor(data=op(other, self.data),
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

_TN_SIMPLE_PROPS = ['structure', 'nsites', 'structure_bsz']
_TN_DATA_PROPS = ['tensor_index', 'tag_index']


class SiteIndexer(object):
    """
    """

    def __init__(self, tn):
        self.tn = tn

    def __getitem__(self, site):
        if site < 0:
            site = self.tn.nsites + site
        site_tag = self.tn.structure.format(site)
        return self.tn[site_tag]

    def __setitem__(self, site, tensor):
        if site < 0:
            site = self.tn.nsites + site
        site_tag = self.tn.structure.format(site)
        self.tn[site_tag] = tensor

    def __delitem__(self, site):
        if site < 0:
            site = self.tn.nsites + site
        site_tag = self.tn.structure.format(site)
        del self.tn[site_tag]


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
    structure : str, optional
        A string, with integer format specifier, that describes how to range
        over the network's tags in order to contract it.
    structure_bsz : int, optional
        How many sites to group together when auto contracting. Eg for 3 (with
        the dotted lines denoting vertical strips of tensors to be contracted):

            .....       i        ........ i        ...i.
            O-O-O-O-O-O-O-        /-O-O-O-O-        /-O-
            | | | | | | |   ->   1  | | | |   ->   2  |   ->  etc.
            O-O-O-O-O-O-O-        \-O-O-O-O-        \-O-

        Should not require tensor contractions with more than 52 unique
        indices.
    nsites : int, optional
        The number of sites, if explicitly known. This will be calculated
        using `structure` if needed but not specified.
    virtual : bool, optional
        Whether the TensorNetwork should be a *view* onto the tensors it is
        given, or a copy of them. E.g. if a virtual TN is constructed, any
        changes to a Tensor's indices will propagate to all TNs viewing that
        Tensor.

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
                 structure=None,
                 structure_bsz=None,
                 nsites=None,
                 virtual=False):

        self.site = SiteIndexer(self)

        # short-circuit for copying TensorNetworks
        if isinstance(tensors, TensorNetwork):
            self.structure = tensors.structure
            self.nsites = tensors.nsites
            self.structure_bsz = tensors.structure_bsz
            self.tag_index = {
                tg: nms.copy() for tg, nms in tensors.tag_index.items()}
            self.tensor_index = {nm: tsr if virtual else tsr.copy()
                                 for nm, tsr in tensors.tensor_index.items()}
            return

        self.structure = structure
        self.structure_bsz = structure_bsz
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
                self.add_tensor(t, virtual=virtual)
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
                for nm, tsr in t.tensor_index.items():
                    self.add_tensor(tsr, virtual=virtual, name=nm)
            else:
                for nm, tsr in t.tensor_index.items():
                    self.tensor_index[nm] = tsr if virtual else tsr.copy()
                self.tag_index = merge_with(
                    set_join, self.tag_index, t.tag_index)

        # count how many sites if a structure is given
        if self.structure:
            if self.nsites is None:
                self.nsites = self.calc_nsites()

            # set default blocksize
            if self.structure_bsz is None:
                self.structure_bsz = 2

    def __and__(self, other):
        """Combine this tensor network with more tensors, without contracting.
        Copies the tensors.
        """
        return TensorNetwork((self, other))

    def __or__(self, other):
        """Combine this tensor network with more tensors, without contracting.
        Views the constituent tensors.
        """
        return TensorNetwork((self, other), virtual=True)

    # ------------------------------- Methods ------------------------------- #

    def copy(self, virtual=False, deep=False):
        """Copy this ``TensorNetwork``. If ``deep=False``, (the default), then
        everything but the actual numeric data will be copied.
        """
        if deep:
            return copy.deepcopy(self)
        return self.__class__(self, virtual=virtual)

    def add_tensor(self, tensor, name=None, virtual=False):
        """Add a single tensor to this network - mangle its name if neccessary.
        """
        # check for name conflict
        if name is None:
            name = rand_uuid(base="_T")
        else:
            try:
                self.tensor_index[name]
                name = rand_uuid(base="_T")
            except KeyError:
                # name is fine to keep
                pass

        # add tensor to the main index
        self.tensor_index[name] = tensor if virtual else tensor.copy()

        # add its name to the relevant tags, or create a new tag
        for tag in tensor.tags:
            try:
                self.tag_index[tag].add(name)
            except (KeyError, TypeError):
                self.tag_index[tag] = {name}

    def __iand__(self, tensor):
        """Inplace, but non-virtual, addition of tensor to this network. It
        should not have any conflicting indices.
        """
        self.add_tensor(tensor, virtual=False)
        return self

    def __ior__(self, tensor):
        """Inplace, virtual, addition of tensor to this network. It
        should not have any conflicting indices.
        """
        self.add_tensor(tensor, virtual=True)
        return self

    def pop_tensor(self, name):
        """Remove a tensor from this network, returning said tensor.
        """
        # remove the tensor from the tag index
        for tag in self.tensor_index[name].tags:
            self.tag_index[tag].discard(name)

        # pop the tensor itself
        return self.tensor_index.pop(name)

    def del_tensor(self, name):
        """Delete a tensor from this network.
        """
        # remove the tensor from the tag index
        for tag in self.tensor_index[name].tags:
            tagged_names = self.tag_index[tag]
            tagged_names.discard(name)
            if not tagged_names:
                del self.tag_index[tag]

        # delete the tensor itself
        del self.tensor_index[name]

    def add_tag(self, tag):
        """Add tag to every tensor in this network.
        """
        names = set()
        for n, t in self.tensor_index.items():
            names.add(n)
            t.tags.add(tag)
        self.tag_index[tag] = names

    def drop_tags(self, tags):
        """Remove a tag from any tensors in this network which have it.
        """
        for t in self.tensor_index.values():
            t.drop_tags(tags)
        if isinstance(tags, str):
            del self.tag_index[tags]
        else:
            for tag in tags:
                del self.tag_index[tag]

    @property
    def tags(self):
        return tuple(self.tag_index.keys())

    @property
    def tensors(self):
        return tuple(self.tensor_index.values())

    def __getitem__(self, tags):
        """Get the tensor(s) associated with ``tags``.

        Parameters
        ----------
        tags : str or sequence of str
            The tags used to select the tensor(s)

        Returns
        -------
        Tensor or sequence of Tensors
        """
        try:
            names = self.tag_index[tags]
        except (KeyError, TypeError):
            names = functools.reduce(
                operator.and_, (self.tag_index[t] for t in tags))

        tensors = tuple(self.tensor_index[name] for name in names)

        if len(names) == 1:
            return tensors[0]

        return tensors

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

        # check if tags match, else need to modify TN structure
        if self.tensor_index[name].tags != tensor.tags:
            self.del_tensor(name)
            self.add_tensor(tensor, name, virtual=True)
        else:
            self.tensor_index[name] = tensor

    def __delitem__(self, tags):
        """Delete any tensors associated with ``tags``.
        """
        try:
            names = self.tag_index[tags]
        except (KeyError, TypeError):
            names = functools.reduce(
                operator.and_, (self.tag_index[t] for t in tags))

        for name in copy.copy(names):
            self.del_tensor(name)

    def calc_nsites(self):
        """Calculate how many tags there are which match ``structure``.
        """
        nsites = 0
        while True:
            if self.structure.format(nsites) in self.tag_index:
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
            untagged_tn.add_tensor(tensor_contract(*tagged_ts), virtual=True)
            return untagged_tn

        return tensor_contract(*tagged_ts)

    def parse_tag_slice(self, tag_slice):
        if tag_slice.start is None:
            start = 0
        elif tag_slice.start is ...:
            start = self.nsites - 1
        elif tag_slice.start < 0:
            start = self.nsites + tag_slice.start
        else:
            start = tag_slice.start

        if (tag_slice.stop is ...) or (tag_slice.stop is None):
            stop = self.nsites
        elif tag_slice.stop < 0:
            stop = self.nsites + tag_slice.stop
        else:
            stop = tag_slice.stop

        step = 1 if stop > start else -1
        return start, stop, step

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
            else:
                tags = set(tags)

            ctags |= tags

            # peform the next contraction
            new_tn = new_tn._contract_tags(ctags, inplace=True)

        return new_tn

    def __rshift__(self, tags_seq):
        """Overload of '>>' for TensorNetwork.cumulative_contract.
        """
        return self.cumulative_contract(tags_seq)

    def __irshift__(self, tags_seq):
        """Overload of '>>=' for inplace TensorNetwork.cumulative_contract.
        """
        return self.cumulative_contract(tags_seq, inplace=True)

    def _contract_with_strategy(self, tags, inplace=False):
        # check for all sites
        if tags is ...:
            tags = slice(0, self.nsites)

        tags_seq = (self.structure.format(i)
                    for i in range(*self.parse_tag_slice(tags)))

        # partition sites into `structure_bsz` groups
        if self.structure_bsz > 1:
            tags_seq = partition_all(self.structure_bsz, tags_seq)

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
        if self.structure is not None:
            # ... but only use for total or slice tags
            if (tags is ...) or isinstance(tags, slice):
                return self._contract_with_strategy(tags, inplace=inplace)

        # Else just contract those tensors specified by tags.
        return self._contract_tags(tags, inplace=inplace)

    def __xor__(self, tags):
        """Overload of '^' for TensorNetwork.contract.
        """
        return self.contract(tags)

    def __ixor__(self, tags):
        """Overload of '^=' for inplace TensorNetwork.contract.
        """
        return self.contract(tags, inplace=True)

    def __matmul__(self, other):
        """Overload "@" to mean full contraction with another network.
        """
        return TensorNetwork((self, other)) ^ ...

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

    # --------------- information about indices and dimensions -------------- #

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
        return tuple(di[1] for di in self.outer_dims_inds())

    @property
    def shape(self):
        """Actual, i.e. exterior, shape of this TensorNetwork.
        """
        return tuple(di[0] for di in self.outer_dims_inds())

    # ------------------------------ printing ------------------------------- #

    def __repr__(self):
        return "{}([{}{}{}]{}{})".format(
            self.__class__.__name__,
            os.linesep,
            "".join(["    " + repr(t) + "," + os.linesep
                     for t in self.tensors[:-1]]),
            "    " + repr(self.tensors[-1]) + "," + os.linesep,
            ", structure='{}'".format(self.structure) if
            self.structure is not None else "",
            ", nsites={}".format(self.nsites) if
            self.nsites is not None else "")
