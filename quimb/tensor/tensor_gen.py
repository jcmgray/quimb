"""Generate specific tensor network states and operators.
"""
import math
import functools
import itertools
import collections
from numbers import Integral

import numpy as np
import opt_einsum as oe

from ..core import make_immutable, ikron
from ..utils import deprecated, unique, concat
from ..gen.operators import (
    spin_operator, eye, _gen_mbl_random_factors, ham_heis
)
from ..gen.rand import randn, choice, random_seed_fn, rand_phase
from .tensor_core import (Tensor, new_bond, TensorNetwork, rand_uuid,
                          tensor_direct_product)
from .array_ops import asarray, sensibly_scale, reshape
from .decomp import eigh
from .tensor_arbgeom import TensorNetworkGen, TensorNetworkGenVector
from .tensor_1d import MatrixProductState, MatrixProductOperator
from .tensor_2d import gen_2d_bonds, TensorNetwork2D
from .tensor_3d import TensorNetwork3D
from .tensor_1d_tebd import LocalHam1D
from .tensor_2d_tebd import LocalHam2D
from .tensor_3d_tebd import LocalHam3D


@random_seed_fn
def rand_tensor(shape, inds, tags=None, dtype='float64', left_inds=None):
    """Generate a random tensor with specified shape and inds.

    Parameters
    ----------
    shape : sequence of int
        Size of each dimension.
    inds : sequence of str
        Names of each dimension.
    tags : sequence of str
        Labels to tag this tensor with.
    dtype : {'float64', 'complex128', 'float32', 'complex64'}, optional
        The underlying data type.
    left_inds : sequence of str, optional
        Which, if any, indices to group as 'left' indices of an effective
        matrix. This can be useful, for example, when automatically applying
        unitary constraints to impose a certain flow on a tensor network but at
        the atomistic (Tensor) level.

    Returns
    -------
    Tensor
    """
    data = randn(shape, dtype=dtype)
    return Tensor(data=data, inds=inds, tags=tags, left_inds=left_inds)


@random_seed_fn
def rand_phased(shape, inds, tags=None, dtype=complex):
    """Generate a random tensor with specified shape and inds, and randomly
    'phased' (distributed on the unit circle) data, such that
    ``T.H @ T == T.norm()**2 == T.size``.

    Parameters
    ----------
    shape : sequence of int
        Size of each dimension.
    inds : sequence of str
        Names of each dimension.
    tags : sequence of str
        Labels to tag this tensor with.
    dtype : {'complex128', 'complex64'}, optional
        The underlying data type - can only be complex.

    Returns
    -------
    Tensor
    """
    data = rand_phase(shape, dtype=dtype)
    return Tensor(data=data, inds=inds, tags=tags)


def gen_unique_edges(edges):
    seen = set()
    for node_a, node_b in edges:
        if node_b < node_a:
            node_a, node_b = node_b, node_a
        key = (node_a, node_b)
        if key in seen:
            continue
        yield (node_a, node_b)
        seen.add(key)


def TN_from_edges_and_fill_fn(
    fill_fn,
    edges,
    D,
    phys_dim=None,
    site_tag_id='I{}',
    site_ind_id='k{}',
):
    """Create a tensor network from a sequence of edges defining a graph,
    and a 'fill' function that maps shapes to data.

    Parameters
    ----------
    fill_fn : callable
        A function with signature ``fill_fn(shape) -> array``, used to fill
        each tensor.
    edges : sequence of tuple[hashable, hashable]
        The graph edges, as a sequence of pairs of hashable objects, for
        example integers, representing the nodes. You can redundantly specify
        ``(u, v)`` and ``(v, u)`` and only one edge will be added.
    D : int
        The bond dimension connecting tensors.
    phys_dim : int, optional
        If not ``None``, give each tensor a 'physical', free index of this size
        at each node.
    site_tag_id : str, optional
        String with formatter to tag sites.
    site_ind_id : str, optional
        String with formatter to tag indices (if ``phys_dim`` specified).

    Returns
    -------
    TensorNetworkGen or TensorNetworkGenVector
    """
    terms = collections.defaultdict(list)
    bonds = collections.defaultdict(rand_uuid)

    for node_a, node_b in gen_unique_edges(edges):
        bond = bonds[node_a, node_b]
        # insert at 0 to exactly match geometry of old TN_rand_reg
        terms[node_a].insert(0, bond)
        terms[node_b].insert(0, bond)

    ts = []
    sites = []
    for node, inds in sorted(terms.items(), key=lambda x: x[0]):
        sites.append(node)
        shape = [D] * len(inds)
        if phys_dim is not None:
            inds.append(site_ind_id.format(node))
            shape.append(phys_dim)
        data = fill_fn(shape)
        tags = site_tag_id.format(node)
        ts.append(Tensor(data=data, inds=inds, tags=tags))

    tn = TensorNetwork(ts)

    if phys_dim is not None:
        tn.view_as_(
            TensorNetworkGenVector, sites=sites,
            site_tag_id=site_tag_id, site_ind_id=site_ind_id
        )
    else:
        tn.view_as_(
            TensorNetworkGen, sites=sites,
            site_tag_id=site_tag_id
        )

    return tn


def TN_from_edges_empty(
    edges,
    D,
    phys_dim=None,
    site_tag_id='I{}',
    site_ind_id='k{}',
    dtype='float64',
):
    """Create a tensor network from a sequence of edges defining a graph,
    initialized with empty tensors.

    Parameters
    ----------
    edges : sequence of tuple[hashable, hashable]
        The graph edges, as a sequence of pairs of hashable objects, for
        example integers, representing the nodes. You can redundantly specify
        ``(u, v)`` and ``(v, u)`` and only one edge will be added.
    D : int
        The bond dimension connecting tensors.
    phys_dim : int, optional
        If not ``None``, give each tensor a 'physical', free index of this size
        at each node.
    site_tag_id : str, optional
        String with formatter to tag sites.
    site_ind_id : str, optional
        String with formatter to tag indices (if ``phys_dim`` specified).
    dtype : str, optional
        The data type of the tensors.

    Returns
    -------
    TensorNetworkGen or TensorNetworkGenVector
    """
    def fill_fn(shape):
        return np.empty(shape, dtype=dtype)

    return TN_from_edges_and_fill_fn(
        edges=edges, D=D, fill_fn=fill_fn, phys_dim=phys_dim,
        site_tag_id=site_tag_id, site_ind_id=site_ind_id)


def TN_from_edges_with_value(
    value,
    edges,
    D,
    phys_dim=None,
    site_tag_id='I{}',
    site_ind_id='k{}',
    dtype=None,
):
    """Create a tensor network from a sequence of edges defining a graph,
    initialized with a constant value. This uses ``numpy.broadcast_to`` and
    therefore essentially no memory.

    Parameters
    ----------
    value : scalar
        The value to fill the tensors with.
    edges : sequence of tuple[hashable, hashable]
        The graph edges, as a sequence of pairs of hashable objects, for
        example integers, representing the nodes. You can redundantly specify
        ``(u, v)`` and ``(v, u)`` and only one edge will be added.
    D : int
        The bond dimension connecting tensors.
    phys_dim : int, optional
        If not ``None``, give each tensor a 'physical', free index of this size
        at each node.
    site_tag_id : str, optional
        String with formatter to tag sites.
    site_ind_id : str, optional
        String with formatter to tag indices (if ``phys_dim`` specified).
    dtype : str, optional
        The data type of the tensors.

    Returns
    -------
    TensorNetworkGen or TensorNetworkGenVector
    """
    element = np.array(value, dtype=dtype)

    def fill_fn(shape):
        return np.broadcast_to(element, shape)

    return TN_from_edges_and_fill_fn(
        edges=edges, D=D, fill_fn=fill_fn, phys_dim=phys_dim,
        site_tag_id=site_tag_id, site_ind_id=site_ind_id)


def TN_from_edges_rand(
    edges,
    D,
    phys_dim=None,
    seed=None,
    dtype='float64',
    site_tag_id='I{}',
    site_ind_id='k{}',
):
    """Create a random tensor network with geometry defined from a sequence
    of edges defining a graph.

    Parameters
    ----------
    G : sequence of tuple[node, node]
        The edges defining a graph, each element should be a pair of nodes
        described by hashable objects.
    D : int
        The bond dimension connecting tensors.
    phys_dim : int, optional
        If not ``None``, give each tensor a 'physical', free index of this size
        to mimic a wavefunction of ``len(G)`` sites.
    seed : int, optional
        A random seed.
    site_tag_id : str, optional
        String with formatter to tag sites.
    site_ind_id : str, optional
        String with formatter to tag indices (if ``phys_dim`` specified).

    Returns
    -------
    TensorNetworkGen or TensorNetworkGenVector
    """
    ts = {}

    sites = tuple(sorted(set(concat(edges))))

    for node in sites:
        t = Tensor(tags=site_tag_id.format(node))
        if phys_dim is not None:
            t.new_ind(site_ind_id.format(node), size=phys_dim)
        ts[node] = t

    for node_a, node_b in gen_unique_edges(edges):
        new_bond(ts[node_a], ts[node_b], size=D)

    tn = TensorNetwork(ts.values())
    tn.randomize_(seed=seed, dtype=dtype)

    if phys_dim is not None:
        tn.view_as_(
            TensorNetworkGenVector, sites=sites,
            site_tag_id=site_tag_id, site_ind_id=site_ind_id
        )
    else:
        tn.view_as_(
            TensorNetworkGen, sites=sites,
            site_tag_id=site_tag_id
        )

    return tn


TN_rand_from_edges = deprecated(
    TN_from_edges_rand, "TN_rand_from_edges", "TN_from_edges_rand"
)


def TN_rand_reg(
    n,
    reg,
    D,
    phys_dim=None,
    seed=None,
    dtype='float64',
    site_tag_id='I{}',
    site_ind_id='k{}',
):
    """Create a random regular tensor network.

    Parameters
    ----------
    n : int
        The number of tensors.
    reg : int
        The degree of the tensor network (how many tensors each tensor
        connects to).
    D : int
        The bond dimension connecting tensors.
    phys_dim : int, optional
        If not ``None``, give each tensor a 'physical', free index of this size
        to mimic a wavefunction of ``n`` sites.
    seed : int, optional
        A random seed.
    site_tag_id : str, optional
        String with formatter to tag sites.
    site_ind_id : str, optional
        String with formatter to tag indices (if ``phys_dim`` specified).

    Returns
    -------
    TensorNetwork
    """
    import networkx as nx
    G = nx.random_degree_sequence_graph([reg] * n, seed=seed)
    return TN_from_edges_rand(
        G.edges, D=D, phys_dim=phys_dim, seed=seed, dtype=dtype,
        site_tag_id=site_tag_id, site_ind_id=site_ind_id)


def TN2D_from_fill_fn(
    fill_fn,
    Lx,
    Ly,
    D,
    cyclic=False,
    site_tag_id='I{},{}',
    row_tag_id='ROW{}',
    col_tag_id='COL{}',
):
    """A scalar 2D lattice tensor network with tensors filled by a function.

    Parameters
    ----------
    fill_fn : callable
        A function with signature ``fill_fn(shape) -> array``, used to fill
        each tensor.
    Lx : int
        Length of side x.
    Ly : int
        Length of side y.
    D : int
        The bond dimension connecting sites.
    cyclic : bool or (bool, bool), optional
        Whether to use periodic boundary conditions. X and Y can be specified
        separately using a tuple.
    site_tag_id : str, optional
        String specifier for naming convention of site tags.
    row_tag_id : str, optional
        String specifier for naming convention of row tags.
    col_tag_id : str, optional
        String specifier for naming convention of column tags.

    Returns
    -------
    TensorNetwork2D
    """
    try:
        cyclic_x, cyclic_y = cyclic
    except TypeError:
        cyclic_x = cyclic_y = cyclic

    ts = []
    bonds = collections.defaultdict(rand_uuid)

    for i, j in itertools.product(range(Lx), range(Ly)):
        directions = ""
        inds = []

        if j > 0 or cyclic_y:
            directions += 'l'
            inds.append(bonds[(i, (j - 1) % Ly), (i, j)])
        if j < Ly - 1 or cyclic_y:
            directions += 'r'
            inds.append(bonds[(i, j), (i, (j + 1) % Ly)])
        if i < Lx - 1 or cyclic_x:
            directions += 'u'
            inds.append(bonds[(i, j), ((i + 1) % Lx, j)])
        if i > 0 or cyclic_x:
            directions += 'd'
            inds.append(bonds[((i - 1) % Lx, j), (i, j)])

        shape = (D,) * len(inds)
        data = fill_fn(shape)
        tags = [site_tag_id.format(i, j),
                row_tag_id.format(i), col_tag_id.format(j)]
        ts.append(Tensor(data=data, inds=inds, tags=tags))

    tn = TensorNetwork(ts)

    return tn.view_as_(
        TensorNetwork2D,
        Lx=Lx, Ly=Ly,
        site_tag_id=site_tag_id,
        row_tag_id=row_tag_id,
        col_tag_id=col_tag_id,
    )


def TN2D_empty(
    Lx,
    Ly,
    D,
    cyclic=False,
    site_tag_id='I{},{}',
    row_tag_id='ROW{}',
    col_tag_id='COL{}',
    dtype='float64',
):
    """A scalar 2D lattice tensor network initialized with empty tensors.

    Parameters
    ----------
    Lx : int
        Length of side x.
    Ly : int
        Length of side y.
    D : int
        The bond dimension connecting sites.
    cyclic : bool or (bool, bool), optional
        Whether to use periodic boundary conditions. X and Y can be specified
        separately using a tuple.
    site_tag_id : str, optional
        String specifier for naming convention of site tags.
    row_tag_id : str, optional
        String specifier for naming convention of row tags.
    col_tag_id : str, optional
        String specifier for naming convention of column tags.
    dtype : str, optional
        The data type of the tensors.

    Returns
    -------
    TensorNetwork2D
    """
    def fill_fn(shape):
        return np.empty(shape, dtype=dtype)

    return TN2D_from_fill_fn(
        fill_fn, Lx=Lx, Ly=Ly, D=D, cyclic=cyclic,
        site_tag_id=site_tag_id, row_tag_id=row_tag_id, col_tag_id=col_tag_id,
    )


def TN2D_with_value(
    value,
    Lx,
    Ly,
    D,
    cyclic=False,
    site_tag_id='I{},{}',
    row_tag_id='ROW{}',
    col_tag_id='COL{}',
    dtype=None,
):
    """A scalar 2D lattice tensor network with every element set to ``value``.
    This uses ``numpy.broadcast_to`` and therefore essentially no memory.

    Parameters
    ----------
    value : scalar
        The value to fill the tensors with.
    Lx : int
        Length of side x.
    Ly : int
        Length of side y.
    D : int
        The bond dimension connecting sites.
    cyclic : bool or (bool, bool), optional
        Whether to use periodic boundary conditions. X and Y can be specified
        separately using a tuple.
    site_tag_id : str, optional
        String specifier for naming convention of site tags.
    row_tag_id : str, optional
        String specifier for naming convention of row tags.
    col_tag_id : str, optional
        String specifier for naming convention of column tags.
    dtype : str, optional
        The data type of the tensors.

    Returns
    -------
    TensorNetwork2D
    """
    element = np.array(value, dtype=dtype)

    def fill_fn(shape):
        return np.broadcast_to(element, shape)

    return TN2D_from_fill_fn(
        fill_fn, Lx=Lx, Ly=Ly, D=D, cyclic=cyclic,
        site_tag_id=site_tag_id, row_tag_id=row_tag_id, col_tag_id=col_tag_id,
    )


@random_seed_fn
def TN2D_rand(
    Lx,
    Ly,
    D,
    cyclic=False,
    site_tag_id='I{},{}',
    row_tag_id='ROW{}',
    col_tag_id='COL{}',
    dtype='float64',
):
    """A random scalar 2D lattice tensor network.

    Parameters
    ----------
    Lx : int
        Length of side x.
    Ly : int
        Length of side y.
    D : int
        The bond dimension connecting sites.
    cyclic : bool or (bool, bool), optional
        Whether to use periodic boundary conditions. X and Y can be specified
        separately using a tuple.
    site_tag_id : str, optional
        String specifier for naming convention of site tags.
    row_tag_id : str, optional
        String specifier for naming convention of row tags.
    col_tag_id : str, optional
        String specifier for naming convention of column tags.
    dtype : dtype, optional
        Data type of the random arrays.
    seed : int, optional
        A random seed.

    Returns
    -------
    TensorNetwork2D
    """
    def fill_fn(shape):
        return randn(shape, dtype=dtype)

    return TN2D_from_fill_fn(
        fill_fn, Lx=Lx, Ly=Ly, D=D, cyclic=cyclic,
        site_tag_id=site_tag_id, row_tag_id=row_tag_id, col_tag_id=col_tag_id,
    )


def TN3D_from_fill_fn(
    fill_fn,
    Lx,
    Ly,
    Lz,
    D,
    cyclic=False,
    site_tag_id='I{},{},{}',
    x_tag_id='X{}',
    y_tag_id='Y{}',
    z_tag_id='Z{}',
):
    """A scalar 3D lattice tensor network with tensors filled by a function.

    Parameters
    ----------
    fill_fn : callable
        A function with signature ``fill_fn(shape) -> array``, used to fill
        each tensor.
    Lx : int
        Length of side x.
    Ly : int
        Length of side y.
    Lz : int
        Length of side z.
    D : int
        The bond dimension connecting sites.
    cyclic : bool or (bool, bool, bool), optional
        Whether to use periodic boundary conditions. X, Y and Z can be
        specified separately using a tuple.
    site_tag_id : str, optional
        String formatter specifying how to label each site.
    dtype : dtype, optional
        Data type of the random arrays.

    Returns
    -------
    TensorNetwork3D
    """
    try:
        cyclic_x, cyclic_y, cyclic_z = cyclic
    except TypeError:
        cyclic_x = cyclic_y = cyclic_z = cyclic

    ts = []
    bonds = collections.defaultdict(rand_uuid)

    for i, j, k in itertools.product(range(Lx), range(Ly), range(Lz)):
        directions = ""
        inds = []

        if k > 0 or cyclic_z:
            directions += 'b'
            inds.append(bonds[(i, j, (k - 1) % Lz), (i, j, k)])
        if k < Lz - 1 or cyclic_z:
            directions += 'a'
            inds.append(bonds[(i, j, k), (i, j, (k + 1) % Lz)])
        if j > 0 or cyclic_y:
            directions += 'l'
            inds.append(bonds[(i, (j - 1) % Ly, k), (i, j, k)])
        if j < Ly - 1 or cyclic_y:
            directions += 'r'
            inds.append(bonds[(i, j, k), (i, (j + 1) % Ly, k)])
        if i < Lx - 1 or cyclic_x:
            directions += 'u'
            inds.append(bonds[(i, j, k), ((i + 1) % Lx, j, k)])
        if i > 0 or cyclic_x:
            directions += 'd'
            inds.append(bonds[((i - 1) % Lx, j, k), (i, j, k)])

        shape = (D,) * len(inds)
        data = fill_fn(shape)
        tags = [site_tag_id.format(i, j, k), x_tag_id.format(i),
                y_tag_id.format(j), z_tag_id.format(k)]
        ts.append(Tensor(data=data, inds=inds, tags=tags))

    tn = TensorNetwork(ts)

    return tn.view_as_(
        TensorNetwork3D,
        Lx=Lx, Ly=Ly, Lz=Lz,
        site_tag_id=site_tag_id,
        x_tag_id=x_tag_id,
        y_tag_id=y_tag_id,
        z_tag_id=z_tag_id,
    )


def TN3D_empty(
    Lx,
    Ly,
    Lz,
    D,
    cyclic=False,
    site_tag_id='I{},{},{}',
    x_tag_id='X{}',
    y_tag_id='Y{}',
    z_tag_id='Z{}',
    dtype='float64',
):
    """A scalar 3D lattice tensor network initialized with empty tensors.

    Parameters
    ----------
    Lx : int
        Length of side x.
    Ly : int
        Length of side y.
    Lz : int
        Length of side z.
    D : int
        The bond dimension connecting sites.
    cyclic : bool or (bool, bool, bool), optional
        Whether to use periodic boundary conditions. X, Y and Z can be
        specified separately using a tuple.
    site_tag_id : str, optional
        String formatter specifying how to label each site.
    dtype : dtype, optional
        Data type of the random arrays.
    seed : int, optional
        Random seed.

    Returns
    -------
    TensorNetwork3D
    """
    def fill_fn(shape):
        return np.empty(shape, dtype=dtype)

    return TN3D_from_fill_fn(
        fill_fn, Lx, Ly, Lz, D,
        cyclic=cyclic, site_tag_id=site_tag_id,
        x_tag_id=x_tag_id, y_tag_id=y_tag_id, z_tag_id=z_tag_id,
    )


def TN3D_with_value(
    value,
    Lx,
    Ly,
    Lz,
    D,
    cyclic=False,
    site_tag_id='I{},{},{}',
    x_tag_id='X{}',
    y_tag_id='Y{}',
    z_tag_id='Z{}',
    dtype=None,
):
    """A scalar 2D lattice tensor network with every element set to ``value``.
    This uses ``numpy.broadcast_to`` and therefore essentially no memory.

    Parameters
    ----------
    value : scalar
        The value to fill the tensors with.
    Lx : int
        Length of side x.
    Ly : int
        Length of side y.
    Lz : int
        Length of side z.
    D : int
        The bond dimension connecting sites.
    cyclic : bool or (bool, bool, bool), optional
        Whether to use periodic boundary conditions. X, Y and Z can be
        specified separately using a tuple.
    site_tag_id : str, optional
        String formatter specifying how to label each site.
    dtype : dtype, optional
        Data type of the random arrays.
    seed : int, optional
        Random seed.

    Returns
    -------
    TensorNetwork3D
    """
    element = np.array(value, dtype=dtype)

    def fill_fn(shape):
        return np.broadcast_to(element, shape)

    return TN3D_from_fill_fn(
        fill_fn, Lx, Ly, Lz, D,
        cyclic=cyclic, site_tag_id=site_tag_id,
        x_tag_id=x_tag_id, y_tag_id=y_tag_id, z_tag_id=z_tag_id,
    )


@random_seed_fn
def TN3D_rand(
    Lx,
    Ly,
    Lz,
    D,
    cyclic=False,
    site_tag_id='I{},{},{}',
    x_tag_id='X{}',
    y_tag_id='Y{}',
    z_tag_id='Z{}',
    dtype='float64',
):
    """A random scalar 3D lattice tensor network.

    Parameters
    ----------
    Lx : int
        Length of side x.
    Ly : int
        Length of side y.
    Lz : int
        Length of side z.
    D : int
        The bond dimension connecting sites.
    cyclic : bool or (bool, bool, bool), optional
        Whether to use periodic boundary conditions. X, Y and Z can be
        specified separately using a tuple.
    site_tag_id : str, optional
        String formatter specifying how to label each site.
    dtype : dtype, optional
        Data type of the random arrays.
    seed : int, optional
        Random seed.

    Returns
    -------
    TensorNetwork
    """
    def fill_fn(shape):
        return randn(shape, dtype=dtype)

    return TN3D_from_fill_fn(
        fill_fn, Lx, Ly, Lz, D,
        cyclic=cyclic, site_tag_id=site_tag_id,
        x_tag_id=x_tag_id, y_tag_id=y_tag_id, z_tag_id=z_tag_id,
    )


# ---------------------------- classical models ----------------------------- #

@functools.lru_cache(128)
def classical_ising_S_matrix(beta, j=1.0):
    """The interaction term for the classical ising model.
    """
    S = np.array(
        [[math.exp(+j * beta), math.exp(-j * beta)],
         [math.exp(-j * beta), math.exp(+j * beta)]])
    make_immutable(S)
    return S


@functools.lru_cache(128)
def classical_ising_H_matrix(beta, h=0.0):
    """The magnetic field term for the classical ising model.
    """
    H = np.array([math.exp(-beta * h), math.exp(beta * h)])
    make_immutable(H)
    return H


@functools.lru_cache(128)
def classical_ising_sqrtS_matrix(beta, j=1.0, asymm=None):
    """The sqrt factorized interaction term for the classical ising model.
    If ``j`` is negative you can supply ``asymm='l'`` or ``'r'`` to
    keep the matrix real, but it must be paired with the opposite in a tensor
    network.
    """
    if (j < 0.0) and (asymm is not None):
        Slr = eigh(classical_ising_S_matrix(beta=beta, j=j))
        S_1_2 = {
            'l': Slr[0], 'lT': Slr[0].T,
            'r': Slr[-1], 'rT': Slr[-1].T,
        }[asymm]
    else:
        S_1_2 = np.array(
            [[math.cosh(j * beta)**0.5 + math.sinh(j * beta)**0.5,
              math.cosh(j * beta)**0.5 - math.sinh(j * beta)**0.5],
             [math.cosh(j * beta)**0.5 - math.sinh(j * beta)**0.5,
              math.cosh(j * beta)**0.5 + math.sinh(j * beta)**0.5]]
        ) / 2**0.5
    make_immutable(S_1_2)
    return S_1_2


@functools.lru_cache(128)
def classical_ising_T_matrix(
    beta,
    j=1.0,
    h=0.0,
    directions='lrud',
    asymm=None,
):
    """The single effective TN site for the classical ising model.
    """
    try:
        js = tuple(j)
    except TypeError:
        js = (j,) * len(directions)

    try:
        asymms = tuple(asymm)
    except TypeError:
        asymms = (asymm,) * len(directions)

    arrays = (
        [
            classical_ising_sqrtS_matrix(beta=beta, j=j, asymm=a)
            for j, a in zip(js, asymms)
        ] +
        [classical_ising_H_matrix(beta, h)]
    )
    lhs = ",".join(f'i{x}' for x in directions)
    eq = lhs + ",i->" + directions
    return oe.contract(eq, *arrays)


def HTN2D_classical_ising_partition_function(
    Lx,
    Ly,
    beta,
    h=0.0,
    j=1.0,
    ind_id='s{},{}',
    cyclic=False,
):
    """Hyper tensor network representation of the 2D classical ising model
    partition function. The indices will be shared by 4 or 5 tensors depending
    on whether ``h`` is non-zero. As opposed to the 'normal' tensor network,
    here each classical spin is still a single index, which is easier to
    contract exactly.

    Parameters
    ----------
    Lx : int
        Length of side x.
    Ly : int
        Length of side y.
    beta : float
        The inverse temperature.
    h : float, optional
        The magnetic field strength.
    j : float, optional
        The interaction strength, positive being *ferromagnetic*.
    cyclic : bool or (bool, bool), optional
        Whether to use periodic boundary conditions. X and Y can be specified
        separately using a tuple.
    ind_id : str, optional
        How to label the indices i.e. ``ind_id.format(i, j)``, each of which
        corresponds to a single classical spin.

    Returns
    -------
    TensorNetwork

    See Also
    --------
    TN2D_classical_ising_partition_function
    """
    try:
        cyclic_x, cyclic_y = cyclic
    except TypeError:
        cyclic_x = cyclic_y = cyclic

    if callable(j):
        j_factory = j
    else:
        def j_factory(node_a, node_b):
            return j

    ts = []
    for ni, nj in itertools.product(range(Lx), range(Ly)):

        if ni < Lx - 1 or cyclic_x:
            node_a, node_b = (ni, nj), ((ni + 1) % Lx, nj)
            inds = ind_id.format(*node_a), ind_id.format(*node_b)
            data = classical_ising_S_matrix(
                beta=beta, j=j_factory(node_a, node_b))
            ts.append(Tensor(data, inds=inds))

        if nj < Ly - 1 or cyclic_y:
            node_a, node_b = (ni, nj), (ni, (nj + 1) % Ly)
            inds = ind_id.format(*node_a), ind_id.format(*node_b)
            data = classical_ising_S_matrix(
                beta=beta, j=j_factory(node_a, node_b))
            ts.append(Tensor(data, inds=inds))

        if h != 0.0:
            data = classical_ising_H_matrix(beta=beta, h=h)
            ts.append(Tensor(data, inds=(ind_id.format(ni, nj),)))

    return TensorNetwork(ts)


def HTN3D_classical_ising_partition_function(
    Lx,
    Ly,
    Lz,
    beta,
    j=1.0,
    h=0.0,
    cyclic=False,
    ind_id='s{},{},{}',
):
    """Hyper tensor network representation of the 3D classical ising model
    partition function. The indices will be shared by 6 or 7 tensors depending
    on whether ``h`` is non-zero. As opposed to the 'normal' tensor network,
    here each classical spin is still a single index, which is easier to
    contract exactly.

    Parameters
    ----------
    Lx : int
        Length of side x.
    Ly : int
        Length of side y.
    Lz : int
        Length of side z.
    beta : float
        The inverse temperature.
    j : float, optional
        The interaction strength, positive being *ferromagnetic*.
    h : float, optional
        The magnetic field strength.
    cyclic : bool or (bool, bool, bool), optional
        Whether to use periodic boundary conditions. X, Y and Z can be
        specified separately using a tuple.
    ind_id : str, optional
        How to label the indices i.e. ``ind_id.format(i, j, k)``, each of which
        corresponds to a single classical spin.

    Returns
    -------
    TensorNetwork

    See Also
    --------
    TN3D_classical_ising_partition_function
    """
    try:
        cyclic_x, cyclic_y, cyclic_z = cyclic
    except TypeError:
        cyclic_x = cyclic_y = cyclic_z = cyclic

    if callable(j):
        j_factory = j
    else:
        def j_factory(node_a, node_b):
            return j

    ts = []
    for ni, nj, nk in itertools.product(range(Lx), range(Ly), range(Lz)):

        if ni < Lx - 1 or cyclic_x:
            node_a, node_b = (ni, nj, nk), ((ni + 1) % Lx, nj, nk)
            inds = (ind_id.format(*node_a), ind_id.format(*node_b))
            data = classical_ising_S_matrix(
                beta=beta, j=j_factory(node_a, node_b))
            ts.append(Tensor(data, inds=inds))

        if nj < Ly - 1 or cyclic_y:
            node_a, node_b = (ni, nj, nk), (ni, (nj + 1) % Ly, nk)
            inds = (ind_id.format(*node_a), ind_id.format(*node_b))
            data = classical_ising_S_matrix(
                beta=beta, j=j_factory(node_a, node_b))
            ts.append(Tensor(data, inds=inds))

        if nk < Lz - 1 or cyclic_z:
            node_a, node_b = (ni, nj, nk), (ni, nj, (nk + 1) % Lz)
            inds = (ind_id.format(*node_a), ind_id.format(*node_b))
            data = classical_ising_S_matrix(
                beta=beta, j=j_factory(node_a, node_b))
            ts.append(Tensor(data, inds=inds))

        if h != 0.0:
            data = classical_ising_H_matrix(beta=beta, h=h)
            ts.append(Tensor(data, inds=(ind_id.format(ni, nj, nk),)))

    return TensorNetwork(ts)


def TN2D_classical_ising_partition_function(
    Lx,
    Ly,
    beta,
    j=1.0,
    h=0.0,
    cyclic=False,
    site_tag_id='I{},{}',
    row_tag_id='ROW{}',
    col_tag_id='COL{}',
):
    """The tensor network representation of the 2D classical ising model
    partition function.

    Parameters
    ----------
    Lx : int
        Length of side x.
    Ly : int
        Length of side y.
    beta : float
        The inverse temperature.
    j : float, optional
        The interaction strength, positive being *ferromagnetic*.
    h : float, optional
        The magnetic field strength.
    cyclic : bool or (bool, bool), optional
        Whether to use periodic boundary conditions. X and Y can be specified
        separately using a tuple.
    site_tag_id : str, optional
        String specifier for naming convention of site tags.
    row_tag_id : str, optional
        String specifier for naming convention of row tags.
    col_tag_id : str, optional
        String specifier for naming convention of column tags.

    Returns
    -------
    TensorNetwork2D

    See Also
    --------
    HTN2D_classical_ising_partition_function
    """
    try:
        cyclic_x, cyclic_y = cyclic
    except TypeError:
        cyclic_x = cyclic_y = cyclic

    if callable(j):
        j_factory = j
    else:
        def j_factory(node_a, node_b):
            return j

    ts = []
    bonds = collections.defaultdict(rand_uuid)

    for ni, nj in itertools.product(range(Lx), range(Ly)):
        directions = ""
        inds = []
        js = ()
        asymms = ()

        for inbounds, pair, direction in [
            (nj > 0 or cyclic_y, ((ni, (nj - 1) % Ly), (ni, nj)), 'l'),
            (nj < Ly - 1 or cyclic_y, ((ni, nj), (ni, (nj + 1) % Ly)), 'r'),
            (ni < Lx - 1 or cyclic_x, ((ni, nj), ((ni + 1) % Lx, nj)), 'u'),
            (ni > 0 or cyclic_x, (((ni - 1) % Lx, nj), (ni, nj)), 'd'),
        ]:
            if inbounds:
                js += (j_factory(*pair),)
                directions += direction
                # this is logic for handling negative j without imag tensors
                # i.e. add the left factor if the first instance of bond, right
                # factor if second. If j > 0.0 this doesn't matter anyhow
                asymms += ('l' if pair not in bonds else 'rT',)
                inds.append(bonds[pair])

        ts.append(Tensor(
            data=classical_ising_T_matrix(
                beta=beta, directions=directions, j=js, h=h, asymm=asymms,
            ),
            inds=inds,
            tags=[site_tag_id.format(ni, nj),
                  row_tag_id.format(ni),
                  col_tag_id.format(nj)]))

    tn = TensorNetwork(ts)

    return tn.view_as_(
        TensorNetwork2D,
        Lx=Lx, Ly=Ly,
        site_tag_id=site_tag_id,
        row_tag_id=row_tag_id,
        col_tag_id=col_tag_id,
    )


def TN3D_classical_ising_partition_function(
    Lx,
    Ly,
    Lz,
    beta,
    j=1.0,
    h=0.0,
    cyclic=False,
    site_tag_id='I{},{},{}',
    x_tag_id='X{}',
    y_tag_id='Y{}',
    z_tag_id='Z{}',
):
    """Tensor network representation of the 3D classical ising model
    partition function.

    Parameters
    ----------
    Lx : int
        Length of side x.
    Ly : int
        Length of side y.
    Lz : int
        Length of side z.
    beta : float
        The inverse temperature.
    j : float, optional
        The interaction strength, positive being *ferromagnetic*.
    h : float, optional
        The magnetic field strength.
    cyclic : bool or (bool, bool, bool), optional
        Whether to use periodic boundary conditions. X, Y and Z can be
        specified separately using a tuple.
    site_tag_id : str, optional
        String formatter specifying how to label each site.

    Returns
    -------
    TensorNetwork

    See Also
    --------
    HTN3D_classical_ising_partition_function
    """
    try:
        cyclic_x, cyclic_y, cyclic_z = cyclic
    except TypeError:
        cyclic_x = cyclic_y = cyclic_z = cyclic

    if callable(j):
        j_factory = j
    else:
        def j_factory(node_a, node_b):
            return j

    ts = []
    bonds = collections.defaultdict(rand_uuid)

    for ni, nj, nk in itertools.product(range(Lx), range(Ly), range(Lz)):
        directions = ""
        inds = []
        js = ()
        asymms = ()

        for inbounds, pair, direction in [
            (nk > 0 or cyclic_z,
             ((ni, nj, (nk - 1) % Lz), (ni, nj, nk)), 'b'),
            (nk < Lz - 1 or cyclic_z,
             ((ni, nj, nk), (ni, nj, (nk + 1) % Lz)), 'a'),
            (nj > 0 or cyclic_y,
             ((ni, (nj - 1) % Ly, nk), (ni, nj, nk)), 'l'),
            (nj < Ly - 1 or cyclic_y,
             ((ni, nj, nk), (ni, (nj + 1) % Ly, nk)), 'r'),
            (ni < Lx - 1 or cyclic_x,
             ((ni, nj, nk), ((ni + 1) % Lx, nj, nk)), 'u'),
            (ni > 0 or cyclic_x,
             (((ni - 1) % Lx, nj, nk), (ni, nj, nk)), 'd'),
        ]:
            if inbounds:
                js += (j_factory(*pair),)
                directions += direction
                # this is logic for handling negative j without imag tensors
                # i.e. add the left factor if the first instance of bond, right
                # factor if second. If j > 0.0 this doesn't matter anyhow
                asymms += ('l' if pair not in bonds else 'rT',)
                inds.append(bonds[pair])

        ts.append(Tensor(
            data=classical_ising_T_matrix(
                beta=beta, directions=directions, j=js, h=h, asymm=asymms,
            ),
            inds=inds,
            tags=[
                site_tag_id.format(ni, nj, nk),
                x_tag_id.format(ni),
                y_tag_id.format(nj),
                z_tag_id.format(nk),
            ],
        ))

    tn = TensorNetwork(ts)

    return tn.view_as_(
        TensorNetwork3D,
        Lx=Lx, Ly=Ly, Lz=Lz,
        site_tag_id=site_tag_id,
        x_tag_id=x_tag_id,
        y_tag_id=y_tag_id,
        z_tag_id=z_tag_id,
    )


def HTN_classical_partition_function_from_edges(
    edges,
    beta,
    j=1.0,
    h=0.0,
    site_ind_id="s{}",
    site_tag_id="I{}",
    bond_tag_id="B{},{}",
):
    """Build a hyper tensor network representation of a classical ising model
    partition function by specifying graph edges. There will be a single
    tensor *per interaction* rather than per site, as well as a single tensor
    for each site, if ``h != 0.0``.

    Parameters
    ----------
    edges : sequence of tuple[hashable, hashable]
        The graph edges, as a sequence of pairs of hashable objects, for
        example integers, representing the nodes. You can redundantly specify
        ``(u, v)`` and ``(v, u)`` and only one edge will be added.
    beta : float, optional
        The inverse temperature.
    j : float, or callable, optional
        The interaction strength, positive being *ferromagnetic*. If a
        callable should have the signature ``j(node_a, node_b)`` and return
        a float.
    h : float, or callable, optional
        The magnetic field strength. If a callable should have the
        signature ``h(node)`` and return a float.
    site_ind_id : str, optional
        A string formatter for naming tensor indices like
        ``site_ind_id.format(node)``.
    site_tag_id : str, optional
        A string formatter for naming tensor tags like
        ``site_tag_id.format(node)``.
    bond_tag_id : str, optional
        A string formatter for naming tensor tags like
        ``bond_tag_id.format(node_a, node_b)``.

    Returns
    -------
    TensorNetwork
    """
    if callable(j):
        j_factory = j
    else:
        def j_factory(node_a, node_b):
            return j

    ts = []
    for node_a, node_b in gen_unique_edges(edges):
        data = classical_ising_S_matrix(beta=beta, j=j_factory(node_a, node_b))
        inds = [site_ind_id.format(node_a),
                site_ind_id.format(node_b)]
        tags = [bond_tag_id.format(node_a, node_b),
                site_tag_id.format(node_a),
                site_tag_id.format(node_b)]
        ts.append(Tensor(data=data, inds=inds, tags=tags))

    if h != 0.0:
        if callable(h):
            h_factory = h
        else:
            def h_factory(node):
                return h

        for node in unique(concat(edges)):
            data = classical_ising_H_matrix(beta, h=h_factory(node))
            inds = [site_ind_id.format(node)]
            tags = [site_tag_id.format(node)]
            ts.append(Tensor(data=data, inds=inds, tags=tags))

    return TensorNetwork(ts)


def TN_classical_partition_function_from_edges(
    edges,
    beta,
    j=1.0,
    h=0.0,
    site_tag_id="I{}",
    bond_ind_id="b{},{}",
):
    """Build a regular tensor network representation of a classical ising model
    partition function by specifying graph edges. There will be a single
    tensor per site.

    Parameters
    ----------
    edges : sequence of tuple[hashable, hashable]
        The graph edges, as a sequence of pairs of hashable objects, for
        example integers, representing the nodes. You can redundantly specify
        ``(u, v)`` and ``(v, u)`` and only one edge will be added.
    beta : float, optional
        The inverse temperature.
    j : float, or callable, optional
        The interaction strength, positive being *ferromagnetic*. If a
        callable should have the signature ``j(node_a, node_b)`` and return
        a float.
    h : float, or callable, optional
        The magnetic field strength. If a callable should have the
        signature ``h(node)`` and return a float.
    site_tag_id : str, optional
        A string formatter for naming tensor tags like
        ``site_ind_id.format(node)``.
    bond_ind_id : str, optional
        A string formatter for naming the indices bewteen tensors like
        ``bond_ind_id.format(node_a, node_b)``.

    Returns
    -------
    TensorNetwork
    """
    if callable(j):
        j_factory = j
    else:
        def j_factory(node_a, node_b):
            return j

    to_contract = collections.defaultdict(list)
    ts = []
    for node_a, node_b in gen_unique_edges(edges):
        j_ab = j_factory(node_a, node_b)
        bond_ab = bond_ind_id.format(node_a, node_b)

        # left tensor factor
        data = classical_ising_sqrtS_matrix(beta=beta, j=j_ab, asymm='l')
        inds = [f's{node_a}', bond_ab]
        tags = [site_tag_id.format(node_a)]
        ts.append(Tensor(data=data, inds=inds, tags=tags))

        # right tensor factor
        data = classical_ising_sqrtS_matrix(beta=beta, j=j_ab, asymm='r')
        inds = [bond_ab, f's{node_b}']
        tags = [site_tag_id.format(node_b)]
        ts.append(Tensor(data=data, inds=inds, tags=tags))

        to_contract[f's{node_a}'].append(bond_ab)
        to_contract[f's{node_b}'].append(bond_ab)

    sites = tuple(sorted(set(concat(edges))))

    if h != 0.0:
        if callable(h):
            h_factory = h
        else:
            def h_factory(node):
                return h

        for node in sites:
            data = classical_ising_H_matrix(beta, h=h_factory(node))
            inds = [f's{node}']
            tags = [site_tag_id.format(node)]
            ts.append(Tensor(data=data, inds=inds, tags=tags))
            to_contract[f's{node}'].extend(())

    tn = TensorNetwork(ts)

    for ind, output_inds in to_contract.items():
        tn.contract_ind(ind, output_inds=output_inds)

    tn.view_as_(TensorNetworkGen, sites=sites, site_tag_id=site_tag_id)
    return tn


@functools.lru_cache(128)
def dimer_data(d, cover_count=1, dtype=float):
    shape = [2] * d
    x = np.zeros(shape, dtype=dtype)
    index_sum = np.indices(shape).sum(axis=0)
    x[index_sum == cover_count] = 1
    make_immutable(x)
    return x


def TN_dimer_covering_from_edges(
    edges,
    cover_count=1,
    site_tag_id="I{}",
    bond_ind_id="b{},{}",
    dtype=float,
):
    """Make a tensor network from sequence of graph edges that counts the
    number of ways to cover the graph exactly with dimers. See
    https://arxiv.org/abs/1805.10598 for the construction.

    Parameters
    ----------
    edges : sequence of tuple
        The edges, each item should be a pair of hashable objects describing
        nodes linked.
    cover_count : int, optional
        The exact number of times each node must be 'covered'. For example
        1 for a standard dimer covering or 2 for 'ice rules'.
    site_tag_id : str, optional
        A string formatter for naming tensor tags like
        ``site_ind_id.format(node)``.
    bond_ind_id : str, optional
        A string formatter for naming the indices bewteen tensors like
        ``bond_ind_id.format(node_a, node_b)``.

    Returns
    -------
    TensorNetwork
    """
    nodes2inds = collections.defaultdict(list)
    for ni, nj in edges:
        bond = bond_ind_id.format(ni, nj)
        nodes2inds[ni].append(bond)
        nodes2inds[nj].append(bond)

    ts = []
    for node, inds in nodes2inds.items():
        data = dimer_data(len(inds), cover_count=cover_count, dtype=dtype)
        tag = site_tag_id.format(node)
        ts.append(Tensor(data, inds=inds, tags=tag))

    tn = TensorNetwork(ts)
    sites = tuple(sorted(nodes2inds))
    tn.view_as_(TensorNetworkGen, sites=sites, site_tag_id=site_tag_id)
    return tn


# --------------------------------------------------------------------------- #
#                           Weighted Model Counting                           #
# --------------------------------------------------------------------------- #


def clause_negmask(clause):
    return int("".join('0' if x > 0 else '1' for x in clause), 2)


@functools.lru_cache(128)
def or_clause_data(ndim, m=0, dtype=float, q=2):
    """Get the array representing satisfiability of ``ndim`` clauses with
    unsatisfied condition encoded in ``m``.
    """
    shape = [q] * ndim
    t = np.ones(shape, dtype=dtype)
    t[np.unravel_index(m, shape)] = 0
    return t


def or_clause_tensor(ndim, m, inds, tags=None):
    """Get the tensor representing satisfiability of ``ndim`` clauses with
    unsatisfied condition encoded in ``m`` labelled by ``inds`` and ``tags``.
    """
    data = or_clause_data(ndim, m=m)
    return Tensor(data=data, inds=inds, tags=tags)


def or_clause_mps_tensors(ndim, m, inds, tags=None):
    """Get the set of MPS tensors representing satisfiability of ``ndim``
    clauses with unsatisfied condition encoded in ``m`` labelled by ``inds``
    and ``tags``.
    """
    mps = (
        MPS_computational_state('+' * ndim, tags=tags) * (2**(ndim / 2)) -
        MPS_computational_state(f'{m:0>{ndim}b}', tags=tags)
    )
    mps.reindex_({
        mps.site_ind(i): ind
        for i, ind in enumerate(inds)
    })
    return mps.tensors


@functools.lru_cache(2**10)
def or_clause_parafac_data(ndim, m):
    """Get the set of PARAFAC arrays representing satisfiability of ``ndim``
    clauses with unsatisfied condition encoded in ``m``.
    """
    inds = [f'k{i}' for i in range(ndim)]
    bond = 'b'

    pfc_ones = np.ones((2, 1))
    pfc_up = np.array([[1.], [0.]])
    pfc_dn = np.array([[0.], [1.]])

    ts_ones = [Tensor(data=pfc_ones, inds=[ix, bond]) for ix in inds]

    bmask = f'{m:0>{ndim}b}'
    ts_mask = [Tensor(data=(pfc_dn if b == '1' else pfc_up), inds=[ix, bond])
               for ix, b in zip(inds, bmask)]

    # just need to multiply a single mask tensor by -1
    ts_mask[0] *= -1
    ts = [tensor_direct_product(t1, t2, sum_inds=(ix,))
          for ix, t1, t2 in zip(inds, ts_ones, ts_mask)]

    return tuple(t.data for t in ts)


def clause_parafac_tensors(ndim, m, inds, tags=None):
    """Get the set of PARAFAC tensors representing satisfiability of ``ndim``
    clauses with unsatisfied condition encoded in ``m`` labelled by ``inds``
    and ``tags``.
    """
    bond = rand_uuid()
    return [Tensor(x, inds=[ix, bond], tags=tags)
            for x, ix in zip(or_clause_parafac_data(ndim, m), inds)]


def HTN_from_cnf(fname, mode='parafac'):
    """Create a hyper tensor network from a '.cnf' or '.wcnf' file - i.e. a
    model counting or weighted model counting instance specification.

    Parameters
    ----------
    fname : str
        Path to a '.cnf' or '.wcnf' file.
    mode : {'parafac', 'mps', 'dense', int}, optional
        How to represent the clauses:

            * 'parafac' - `N` rank-2 tensors connected by a single hyper index.
              You could further call :meth:`hyperinds_resolve` for more options
              to convert the hyper index into a (decomposed) COPY-tensor.
            * 'mps' - `N` rank-3 tensors connected along a 1D line.
            * 'dense' - contract the hyper index.
            * int - use the 'parafac' mode, but only if the length of a clause
              is larger than this threshold.

    Returns
    -------
    htn : TensorNetwork
    """

    ts = []
    weights = {}
    weighted = set()
    clause_counter = 1

    with open(fname, 'r') as f:
        for line in f:
            args = line.split()

            # global info, don't need
            if args[0] == 'p':
                # num_vars = int(args[2])
                # num_clauses = int(args[3])
                continue

            # translate mc2021 style weight to normal
            if args[:3] == ['c', 'p', 'weight']:
                args = ('w', *args[3:5])
            # variable weight
            if args[0] == 'w':
                sgn_var, w = args[1:]
                sgn_var = int(sgn_var)
                sgn = '-' if sgn_var < 0 else '+'
                var = str(abs(sgn_var))
                w = float(w)
                weights[var, sgn] = w
                weighted.add(var)
                continue

            # ignore empty lines, other comments and info line
            if (not args) or (args == ['0']) or (args[0][0] in 'c%'):
                continue

            # clause tensor
            clause = tuple(map(int, filter(None, args[:-1])))

            # encode the OR statement with possible negations as int
            m = clause_negmask(clause)
            inds = [str(abs(var)) for var in clause]
            tag = f'CLAUSE{clause_counter}'

            if (
                # parafac mode
                (mode == 'parafac' and len(inds) > 2) or
                # parafac above cutoff size mode
                (isinstance(mode, int) and len(inds) > mode)
            ):
                ts.extend(clause_parafac_tensors(len(inds), m, inds, tag))

            elif mode == 'mps' and len(inds) > 2:
                ts.extend(or_clause_mps_tensors(len(inds), m, inds, tag))

            else:
                # dense
                ts.append(or_clause_tensor(len(inds), m, inds, tag))

            clause_counter += 1

    for var in sorted(weighted):
        wp_specified = (var, '+') in weights
        wm_specified = (var, '-') in weights

        if wp_specified and wm_specified:
            wp, wm = weights[var, '+'], weights[var, '-']
        elif wp_specified:
            wp = weights[var, '+']
            wm = 1 - wp
        elif wm_specified:
            wm = weights[var, '-']
            wp = 1 - wm

        ts.append(Tensor([wm, wp], inds=[var], tags=[f'VAR{var}']))

    return TensorNetwork(ts, virtual=True)


# --------------------------------------------------------------------------- #
#                                    MPSs                                     #
# --------------------------------------------------------------------------- #

@random_seed_fn
def MPS_rand_state(L, bond_dim, phys_dim=2, normalize=True, cyclic=False,
                   dtype='float64', trans_invar=False, **mps_opts):
    """Generate a random matrix product state.

    Parameters
    ----------
    L : int
        The number of sites.
    bond_dim : int
        The bond dimension.
    phys_dim : int, optional
        The physical (site) dimensions, defaults to 2.
    normalize : bool, optional
        Whether to normalize the state.
    cyclic : bool, optional
        Generate a MPS with periodic boundary conditions or not, default is
        open boundary conditions.
    dtype : {float, complex} or numpy dtype, optional
        Data type of the tensor network.
    trans_invar : bool (optional)
        Whether to generate a translationally invariant state,
        requires cyclic=True.
    mps_opts
        Supplied to :class:`~quimb.tensor.tensor_1d.MatrixProductState`.
    """
    if trans_invar and not cyclic:
        raise ValueError("State cannot be translationally invariant with open "
                         "boundary conditions.")

    # check for site varying physical dimensions
    if isinstance(phys_dim, Integral):
        phys_dims = itertools.repeat(phys_dim)
    else:
        phys_dims = itertools.cycle(phys_dim)

    cyc_dim = (bond_dim,) if cyclic else ()

    def gen_shapes():
        yield (*cyc_dim, bond_dim, next(phys_dims))
        for _ in range(L - 2):
            yield (bond_dim, bond_dim, next(phys_dims))
        yield (bond_dim, *cyc_dim, next(phys_dims))

    def gen_data(shape):
        return randn(shape, dtype=dtype)

    if trans_invar:
        array = sensibly_scale(gen_data(next(gen_shapes())))
        arrays = (array for _ in range(L))
    else:
        arrays = map(sensibly_scale, map(gen_data, gen_shapes()))

    rmps = MatrixProductState(arrays, **mps_opts)

    if normalize == 'left':
        rmps.left_canonize(normalize=True)
    elif normalize == 'right':
        rmps.left_canonize(normalize=True)
    elif normalize:
        rmps /= (rmps.H @ rmps)**0.5

    return rmps


def MPS_product_state(arrays, cyclic=False, **mps_opts):
    """Generate a product state in MatrixProductState form, i,e,
    with bond dimension 1, from single site vectors described by ``arrays``.
    """
    cyc_dim = (1,) if cyclic else ()

    def gen_array_shapes():
        yield (*cyc_dim, 1, -1)
        for _ in range(len(arrays) - 2):
            yield (1, 1, -1)
        yield (*cyc_dim, 1, -1)

    mps_arrays = (asarray(array).reshape(*shape)
                  for array, shape in zip(arrays, gen_array_shapes()))

    return MatrixProductState(mps_arrays, shape='lrp', **mps_opts)


def MPS_computational_state(binary, dtype='float64', cyclic=False, **mps_opts):
    """A computational basis state in Matrix Product State form.

    Parameters
    ----------
    binary : str or sequence of int
        String specifying the state, e.g. ``'00101010111'`` or ``[0, 0, 1]``.
    cyclic : bool, optional
        Generate a MPS with periodic boundary conditions or not, default open
        boundary conditions.
    mps_opts
        Supplied to MatrixProductState constructor.
    """

    array_map = {
        '0': np.array([1., 0.], dtype=dtype),
        '1': np.array([0., 1.], dtype=dtype),
        '+': np.array([2**-0.5, 2**-0.5], dtype=dtype),
        '-': np.array([2**-0.5, -2**-0.5], dtype=dtype),
    }

    def gen_arrays():
        for s in binary:
            yield array_map[str(s)]

    return MPS_product_state(tuple(gen_arrays()), cyclic=cyclic, **mps_opts)


def MPS_neel_state(L, down_first=False, dtype='float64', **mps_opts):
    """Generate the neel state in Matrix Product State form.

    Parameters
    ----------
    L : int
        The number of spins.
    down_first : bool, optional
        Whether to start with '1' (down) or '0' (up) first.
    mps_opts
        Supplied to MatrixProductState constructor.
    """
    binary_str = "01" * (L // 2) + (L % 2 == 1) * "0"
    if down_first:
        binary_str = "1" + binary_str[:-1]
    return MPS_computational_state(binary_str, dtype=dtype, **mps_opts)


def MPS_ghz_state(L, dtype='float64', **mps_opts):
    """Build the chi=2 OBC MPS representation of the GHZ state.

    Parameters
    ----------
    L : int
        Number of qubits.
    dtype : {'float64', 'complex128', 'float32', 'complex64'}, optional
        The underlying data type.
    mps_opts
        Supplied to :class:`~quimb.tensor.tensor_1d.MatrixProductState`.
    """

    def gen_arrays():
        yield 2**-0.5 * np.array([[1., 0.],
                                  [0., 1.]]).astype(dtype)

        for i in range(1, L - 1):
            yield np.array([[[1., 0.],
                             [0., 0.]],
                            [[0., 0.],
                             [0., 1.]]]).astype(dtype)

        yield np.array([[1., 0.],
                        [0., 1.]]).astype(dtype)

    return MatrixProductState(gen_arrays(), **mps_opts)


def MPS_w_state(L, dtype='float64', **mps_opts):
    """Build the chi=2 OBC MPS representation of the W state.

    Parameters
    ----------
    L : int
        Number of qubits.
    dtype : {'float64', 'complex128', 'float32', 'complex64'}, optional
        The underlying data type.
    mps_opts
        Supplied to :class:`~quimb.tensor.tensor_1d.MatrixProductState`.
    """

    def gen_arrays():
        yield (np.array([[1., 0.],
                         [0., 1.]]) / L ** 0.5).astype(dtype)

        for i in range(1, L - 1):
            yield np.array([[[1., 0.],
                             [0., 1.]],
                            [[0., 0.],
                             [1., 0.]]]).astype(dtype)

        yield np.array([[0., 1.],
                        [1., 0.]]).astype(dtype)

    return MatrixProductState(gen_arrays(), **mps_opts)


@random_seed_fn
def MPS_rand_computational_state(L, dtype='float64', **mps_opts):
    """Generate a random computation basis state, like '01101001010'.

    Parameters
    ----------
    L : int
        The number of qubits.
    seed : int, optional
        The seed to use.
    dtype : {float, complex} or numpy dtype, optional
        Data type of the tensor network.
    mps_opts
        Supplied to :class:`~quimb.tensor.tensor_1d.MatrixProductState`.
    """
    cstr = (choice(('0', '1')) for _ in range(L))
    return MPS_computational_state(cstr, dtype=dtype, **mps_opts)


def MPS_zero_state(L, bond_dim=1, phys_dim=2, cyclic=False,
                   dtype='float64', **mps_opts):
    """The all-zeros MPS state, of given bond-dimension.

    Parameters
    ----------
    L : int
        The number of sites.
    bond_dim : int, optional
        The bond dimension, defaults to 1.
    phys_dim : int, optional
        The physical (site) dimensions, defaults to 2.
    cyclic : bool, optional
        Generate a MPS with periodic boundary conditions or not, default is
        open boundary conditions.
    dtype : {float, complex} or numpy dtype, optional
        Data type of the tensor network.
    mps_opts
        Supplied to :class:`~quimb.tensor.tensor_1d.MatrixProductState`.
    """
    cyc_dim = (bond_dim,) if cyclic else ()

    def gen_arrays():
        yield np.zeros((*cyc_dim, bond_dim, phys_dim), dtype=dtype)
        for _ in range(L - 2):
            yield np.zeros((bond_dim, bond_dim, phys_dim), dtype=dtype)
        yield np.zeros((bond_dim, *cyc_dim, phys_dim), dtype=dtype)

    return MatrixProductState(gen_arrays(), **mps_opts)


def MPS_sampler(L, dtype=complex, squeeze=True, **mps_opts):
    """A product state for sampling tensor network traces. Seen as a vector it
    has the required property that ``psi.H @ psi == d`` always for hilbert
    space size ``d``.
    """
    arrays = [rand_phase(2, dtype=dtype) for _ in range(L)]
    psi = MPS_product_state(arrays, **mps_opts)
    if squeeze:
        psi.squeeze_()
    return psi


# --------------------------------------------------------------------------- #
#                                    MPOs                                     #
# --------------------------------------------------------------------------- #

def MPO_identity(L, phys_dim=2, dtype='float64', cyclic=False, **mpo_opts):
    """Generate an identity MPO of size ``L``.

    Parameters
    ----------
    L : int
        The number of sites.
    phys_dim : int, optional
        The physical (site) dimensions, defaults to 2.
    dtype : {float, complex} or numpy dtype, optional
        Data type of the tensor network.
    cyclic : bool, optional
        Generate a MPO with periodic boundary conditions or not, default is
        open boundary conditions.
    mpo_opts
        Supplied to :class:`~quimb.tensor.tensor_1d.MatrixProductOperator`.
    """
    II = np.identity(phys_dim, dtype=dtype)
    cyc_dim = (1,) if cyclic else ()

    def gen_arrays():
        yield II.reshape(*cyc_dim, 1, phys_dim, phys_dim)
        for _ in range(L - 2):
            yield II.reshape(1, 1, phys_dim, phys_dim)
        yield II.reshape(1, *cyc_dim, phys_dim, phys_dim)

    return MatrixProductOperator(gen_arrays(), **mpo_opts)


def MPO_identity_like(mpo, **mpo_opts):
    """Return an identity matrix operator with the same physical index and
    inds/tags as ``mpo``.
    """
    return MPO_identity(L=mpo.L, phys_dim=mpo.phys_dim(), dtype=mpo.dtype,
                        site_tag_id=mpo.site_tag_id, cyclic=mpo.cyclic,
                        upper_ind_id=mpo.upper_ind_id,
                        lower_ind_id=mpo.lower_ind_id, **mpo_opts)


def MPO_zeros(L, phys_dim=2, dtype='float64', cyclic=False, **mpo_opts):
    """Generate a zeros MPO of size ``L``.

    Parameters
    ----------
    L : int
        The number of sites.
    phys_dim : int, optional
        The physical (site) dimensions, defaults to 2.
    dtype : {float, complex} or numpy dtype, optional
        Data type of the tensor network.
    cyclic : bool, optional
        Generate a MPO with periodic boundary conditions or not, default is
        open boundary conditions.
    mpo_opts
        Supplied to :class:`~quimb.tensor.tensor_1d.MatrixProductOperator`.

    Returns
    -------
    MatrixProductOperator
    """
    cyc_dim = (1,) if cyclic else ()

    def gen_arrays():
        yield np.zeros((*cyc_dim, 1, phys_dim, phys_dim), dtype=dtype)
        for _ in range(L - 2):
            yield np.zeros((1, 1, phys_dim, phys_dim), dtype=dtype)
        yield np.zeros((1, *cyc_dim, phys_dim, phys_dim), dtype=dtype)

    return MatrixProductOperator(gen_arrays(), **mpo_opts)


def MPO_zeros_like(mpo, **mpo_opts):
    """Return a zeros matrix product operator with the same physical index and
    inds/tags as ``mpo``.

    Parameters
    ----------
    mpo : MatrixProductOperator
        The MPO to copy the shape of.

    Returns
    -------
    MatrixProductOperator
    """
    return MPO_zeros(L=mpo.L, phys_dim=mpo.phys_dim(),
                     dtype=mpo.dtype, site_tag_id=mpo.site_tag_id,
                     upper_ind_id=mpo.upper_ind_id, cyclic=mpo.cyclic,
                     lower_ind_id=mpo.lower_ind_id, **mpo_opts)


def MPO_product_operator(
    arrays,
    cyclic=False,
    **mpo_opts,
):
    """Return an MPO of bond dimension 1 representing the product of raw
    operators given in ``arrays``.

    Parameters
    ----------
    arrays : sequence of 2D array_like
        The operators to form a tensor product of.
    cyclic : bool, optional
        Whether to generate a cyclic MPO or not.
    mpo_opts
        Supplied to :class:`~quimb.tensor.tensor_1d.MatrixProductOperator`.

    Returns
    -------
    MatrixProductOperator
    """
    cyc_dim = (1,) if cyclic else ()

    def gen_arrays():
        array_i, *arrays_mid, array_f = arrays
        yield reshape(array_i, (*cyc_dim, 1, *array_i.shape))
        for array_m in arrays_mid:
            yield reshape(array_m, (1, 1, *array_m.shape))
        yield reshape(array_f, (*cyc_dim, 1, *array_f.shape))

    return MatrixProductOperator(gen_arrays(), shape='lrud', **mpo_opts)


@random_seed_fn
def MPO_rand(L, bond_dim, phys_dim=2, normalize=True, cyclic=False,
             herm=False, dtype='float64', **mpo_opts):
    """Generate a random matrix product state.

    Parameters
    ----------
    L : int
        The number of sites.
    bond_dim : int
        The bond dimension.
    phys_dim : int, optional
        The physical (site) dimensions, defaults to 2.
    normalize : bool, optional
        Whether to normalize the operator such that ``trace(A.H @ A) == 1``.
    cyclic : bool, optional
        Generate a MPO with periodic boundary conditions or not, default is
        open boundary conditions.
    dtype : {float, complex} or numpy dtype, optional
        Data type of the tensor network.
    herm : bool, optional
        Whether to make the matrix hermitian (or symmetric if real) or not.
    mpo_opts
        Supplied to :class:`~quimb.tensor.tensor_1d.MatrixProductOperator`.
    """
    cyc_shp = (bond_dim,) if cyclic else ()

    shapes = [(*cyc_shp, bond_dim, phys_dim, phys_dim),
              *((bond_dim, bond_dim, phys_dim, phys_dim),) * (L - 2),
              (bond_dim, *cyc_shp, phys_dim, phys_dim)]

    def gen_data(shape):
        data = randn(shape, dtype=dtype)
        if not herm:
            return data

        trans = (0, 2, 1) if len(shape) == 3 else (0, 1, 3, 2)
        return data + data.transpose(*trans).conj()

    arrays = map(sensibly_scale, map(gen_data, shapes))

    rmpo = MatrixProductOperator(arrays, **mpo_opts)

    if normalize:
        rmpo /= (rmpo.H @ rmpo)**0.5

    return rmpo


@random_seed_fn
def MPO_rand_herm(L, bond_dim, phys_dim=2, normalize=True,
                  dtype='float64', **mpo_opts):
    """Generate a random hermitian matrix product operator.
    See :class:`~quimb.tensor.tensor_gen.MPO_rand`.
    """
    return MPO_rand(L, bond_dim, phys_dim=phys_dim, normalize=normalize,
                    dtype=dtype, herm=True, **mpo_opts)


# ---------------------------- MPO hamiltonians ----------------------------- #

def maybe_make_real(X):
    """Check if ``X`` is real, if so, convert to contiguous array.
    """
    if np.allclose(X.imag, np.zeros_like(X)):
        return np.ascontiguousarray(X.real)
    return X


def spin_ham_mpo_tensor(one_site_terms, two_site_terms, S=1 / 2,
                        left_two_site_terms=None, which=None, cyclic=False):
    """Generate tensor(s) for a spin hamiltonian MPO.

    Parameters
    ----------
    one_site_terms : sequence of (scalar, operator)
        The terms that act on a single site, each ``operator`` can be a string
        suitable to be sent to :func:`spin_operator` or an actual 2d-array.
    two_site_terms : sequence of (scalar, operator operator)
        The terms that act on two neighbouring sites, each ``operator`` can be
        a string suitable to be sent to :func:`spin_operator` or an actual
        2d-array.
    S : fraction, optional
        What size spin to use, defaults to spin-1/2.
    left_two_site_terms : sequence of (scalar, operator operator), optional
        If the interaction to the left of this site has different spin terms
        then the equivalent list of terms for that site.
    which : {None, 'L', 'R', 'A'}, optional
        If ``None``, generate the middle tensor, if 'L' a left-end tensor, if
        'R' a right-end tensor and if 'A' all three.
    cyclic : bool, optional
        Whether to use periodic boundary conditions - default False.

    Returns
    -------
    numpy.ndarray[, numpy.ndarray, numpy.ndarray]
        The middle, left, right or all three MPO tensors.
    """
    # assume same interaction type everywhere
    if left_two_site_terms is None:
        left_two_site_terms = two_site_terms

    # local dimension
    D = int(2 * S + 1)
    # bond dimension to right
    B = len(two_site_terms) + 2
    # bond dimension to left
    BL = len(left_two_site_terms) + 2

    H = np.zeros((BL, B, D, D), dtype=complex)

    # add one-body terms
    for factor, s in one_site_terms:
        if isinstance(s, str):
            s = spin_operator(s, S=S)
        H[-1, 0, :, :] += factor * s

    # add two-body terms
    for i, (factor, s1, _) in enumerate(two_site_terms):
        if isinstance(s1, str):
            s1 = spin_operator(s1, S=S)
        H[-1, 1 + i, :, :] = factor * s1

    for i, (_, _, s2) in enumerate(left_two_site_terms):
        if isinstance(s2, str):
            s2 = spin_operator(s2, S=S)
        H[i + 1, 0, :, :] = s2

    H[0, 0, :, :] = eye(D)
    H[-1, -1, :, :] = eye(D)

    H = maybe_make_real(H)
    make_immutable(H)

    if which in {None, 'M'}:
        return H

    if cyclic:
        # need special conditions for first MPO matrix
        HL = np.zeros_like(H)
        HL[0, :, :, :] = H[-1, :, :, :]
        HL[1:-1, -1, :, :] = H[1:-1, 0, :, :]
        HR = H
    else:
        HL = H[-1, :, :, :]
        HR = H[:, 0, :, :]

    if which == 'L':
        return HL
    elif which == 'R':
        return HR
    elif which == 'A':
        return HL, H, HR


class _TermAdder:
    """Simple class to allow ``SpinHam1D`` syntax like
    ``builder[i, j] += (1/2, 'Z', 'X')``. This object is temporarily created
    by the getitem call, accumulates the new term, then has its the new
    combined list of terms extracted in the setitem call.
    """

    def __init__(self, terms, nsite):
        self.terms = terms
        self.nsite = nsite

    def __iadd__(self, new):
        if len(new) != self.nsite + 1:
            raise ValueError(
                "New terms should be of the form")

        if self.terms is None:
            self.terms = [new]
        else:
            self.terms += [new]
        return self


class SpinHam1D:
    """Class for easily building custom spin hamiltonians in MPO or LocalHam1D
    form. Currently limited to nearest neighbour interactions (and single site
    terms). It is possible to set 'default' translationally invariant terms,
    but also terms acting on specific sites only (which take precedence).
    It is also possible to build a sparse matrix version of the hamiltonian
    (obviously for small sizes only).

    Parameters
    ----------
    S : float, optional
        The type of spin, defaults to 1/2.
    cyclic : bool, optional
        Whether to use periodic boundary conditions - default is False.

    Examples
    --------
    Initialize the spin hamiltonian builder:

        >>> builder = SpinHam1D(S=3 / 2)

    Add some two-site terms:

        >>> builder += 0.5, '+', '-'
        >>> builder += 0.5, '-', '+'
        >>> builder += 1.0, 'Z', 'Z'

    Add a single site term:

        >>> builder -= 0.3, 'Z'

    Build a MPO version of the hamiltonian for use with DMRG:

        >>> mpo_ham = builder.build_mpo(100)
        >>> mpo_ham
        <MatrixProductOperator(tensors=100, L=100, max_bond=5)>

    Build a LocalHam1D version of the hamiltonian for use with TEBD:

        >>> builder.build_local_ham(100)
        <LocalHam1D(L=100, cyclic=False)>

    You can also set terms for specific sites (this overides any of the
    'default', translationally invariant terms set as above):

        >>> builder[10, 11] += 0.75, '+', '-'
        >>> builder[10, 11] += 0.75, '-', '+'
        >>> builder[10, 11] += 1.5, 'Z', 'Z'

    Or specific one-site terms (which again overides any default
    single site terms set above):

        >>> builder[10] += 3.7, 'Z'
        >>> builder[11] += 0.0, 'I' # '0' term turns off field
    """

    def __init__(self, S=1 / 2, cyclic=False):
        self.S = S
        self.one_site_terms = []
        self.two_site_terms = []
        self.cyclic = cyclic

        # Holders for any non-translationally invariant terms
        self.var_one_site_terms = {}
        self.var_two_site_terms = {}

    def add_term(self, factor, *operators):
        """Add another term to the expression to be built.

        Parameters
        ----------
        factor : scalar
            Scalar factor to multiply this term by.
        *operators : str or array
            The operators to use. Can specify one or two for single or two site
            terms respectively. Can use strings, which are supplied to
            :func:`~quimb.spin_operator`, or actual arrays as long as they have
            the correct dimension.
        """
        if factor == 0.0:
            # no need to add zero terms
            return

        if len(operators) == 1:
            self.one_site_terms.append((factor, *operators))
        elif len(operators) == 2:
            self.two_site_terms.append((factor, *operators))
        else:
            raise NotImplementedError("3-body+ terms are not supported yet.")

    def sub_term(self, factor, *operators):
        """Subtract a term - simple alias that flips sign of ``factor``.
        """
        self.add_term(-factor, *operators)

    def __iadd__(self, term):
        self.add_term(*term)
        return self

    def __isub__(self, term):
        self.sub_term(*term)
        return self

    def __getitem__(self, sites):
        """Part of the machinery that allows terms to be added to specific
        sites like::

            >>> builder[i] += 1/2, 'X'
            >>> builder[45, 46] += 1/2, 'Z', 'Z'

        """
        if isinstance(sites, Integral):
            return _TermAdder(self.var_one_site_terms.get(sites, None), 1)

        i, j = sorted(sites)
        if j - i != 1:
            raise NotImplementedError("Can only add nearest neighbour terms.")

        return _TermAdder(self.var_two_site_terms.get(sites, None), 2)

    def __setitem__(self, sites, value):
        """Part of the machinery that allows terms to be added to specific
        sites like::

            >>> builder[i] += 1/2, 'X'
            >>> builder[45, 46] += 1/2, 'Z', 'Z'

        Could also be called directly with a list of terms like::

            >>> builder[13, 14] = [(1, 'Z', 'Z'), (0.5, 'X', 'Y')]

        Which would overide any terms set so far.
        """
        if isinstance(value, _TermAdder):
            terms = value.terms
        else:
            terms = value

        if isinstance(sites, Integral):
            self.var_one_site_terms[sites] = terms
        else:
            i, j = sorted(sites)
            if j - i != 1:
                raise ValueError("Can only add nearest neighbour terms.")
            self.var_two_site_terms[sites] = terms

    def build_mpo(self, L, upper_ind_id='k{}', lower_ind_id='b{}',
                  site_tag_id='I{}', tags=None, bond_name=""):
        """Build an MPO instance of this spin hamiltonian of size ``L``. See
        also ``MatrixProductOperator``.
        """
        # cache the default term
        t_defs = {}

        def get_default_term(which):
            try:
                return t_defs[which]
            except KeyError:
                t_defs['L'], t_defs[None], t_defs['R'] = spin_ham_mpo_tensor(
                    self.one_site_terms, self.two_site_terms,
                    S=self.S, which='A', cyclic=self.cyclic)
                return t_defs[which]

        def gen_tensors():
            for i in range(L):
                which = {0: 'L', L - 1: 'R'}.get(i, None)

                ij_L = (i - 1, i)
                ij_R = (i, i + 1)

                # check for site/bond specific terms
                var_one = i in self.var_one_site_terms
                var_two = (
                    (ij_L in self.var_two_site_terms) or
                    (ij_R in self.var_two_site_terms)
                )

                if not (var_one or var_two):
                    yield get_default_term(which)
                else:
                    t1s = self.var_one_site_terms.get(i, self.one_site_terms)
                    t2s = self.var_two_site_terms.get(ij_R,
                                                      self.two_site_terms)
                    t2s_L = self.var_two_site_terms.get(ij_L,
                                                        self.two_site_terms)

                    yield spin_ham_mpo_tensor(t1s, t2s, S=self.S,
                                              left_two_site_terms=t2s_L,
                                              which=which, cyclic=self.cyclic)

        return MatrixProductOperator(arrays=gen_tensors(), bond_name=bond_name,
                                     upper_ind_id=upper_ind_id,
                                     lower_ind_id=lower_ind_id,
                                     site_tag_id=site_tag_id, tags=tags)

    def build_sparse(self, L, **ikron_opts):
        """Build a sparse matrix representation of this Hamiltonian.

        Parameters
        ----------
        L : int, optional
            The number of spins to build the matrix for.
        ikron_opts
            Supplied to :func:`~quimb.core.ikron`.

        Returns
        -------
        H : matrix
        """
        ikron_opts.setdefault('sparse', True)

        D = int(2 * self.S + 1)
        dims = [D] * L

        terms = []
        for i in range(L):

            t1s = self.var_one_site_terms.get(i, self.one_site_terms)
            for factor, s in t1s:
                if isinstance(s, str):
                    s = spin_operator(s, S=self.S, sparse=True)
                terms.append(
                    ikron(factor * s, dims, i, **ikron_opts)
                )

            if (i + 1 == L) and (not self.cyclic):
                break

            t2s = self.var_two_site_terms.get((i, i + 1), self.two_site_terms)
            for factor, s1, s2 in t2s:
                if isinstance(s1, str):
                    s1 = spin_operator(s1, S=self.S, sparse=True)
                if isinstance(s2, str):
                    s2 = spin_operator(s2, S=self.S, sparse=True)
                terms.append(
                    ikron([factor * s1, s2], dims, [i, i + 1], **ikron_opts)
                )

        return sum(terms)

    def _get_spin_op(self, factor, *ss):
        if len(ss) == 1:
            s, = ss
            if isinstance(s, str):
                s = spin_operator(s, S=self.S)
            return factor * s

        if len(ss) == 2:
            s1, s2 = ss
            if isinstance(s1, str):
                s1 = spin_operator(s1, S=self.S)
            if isinstance(s2, str):
                s2 = spin_operator(s2, S=self.S)
            return factor * (s1 & s2)

    def _sum_spin_ops(self, terms):
        H = sum(self._get_spin_op(*term) for term in terms)
        H = maybe_make_real(H)
        make_immutable(H)
        return H

    def build_local_ham(self, L=None, **local_ham_1d_opts):
        """Build a nearest neighbour interactor instance of this spin
        hamiltonian of size ``L``. See also
        :class:`~quimb.tensor.tensor_1d_tebd.LocalHam1D`.

        Parameters
        ----------
        L : int, optional
            The number of spins, if the hamiltonian only has two-site terms
            this is optional.

        Returns
        -------
        LocalHam1D
        """
        H1s, H2s = {}, {}

        # add default two site term
        if self.two_site_terms:
            H2s[None] = self._sum_spin_ops(self.two_site_terms)

        # add specific two site terms
        if self.var_two_site_terms:
            for sites, terms in self.var_two_site_terms.items():
                H2s[sites] = self._sum_spin_ops(terms)

        # add default one site term
        if self.one_site_terms:
            H1s[None] = self._sum_spin_ops(self.one_site_terms)

        # add specific one site terms
        if self.var_one_site_terms:
            for site, terms in self.var_one_site_terms.items():
                H1s[site] = self._sum_spin_ops(terms)

        return LocalHam1D(H2=H2s, H1=H1s, L=L,
                          cyclic=self.cyclic, **local_ham_1d_opts)


SpinHam = deprecated(SpinHam1D, 'SpinHam', 'SpinHam1D')


def _ham_ising(j=1.0, bx=0.0, *, S=1 / 2, cyclic=False):
    H = SpinHam1D(S=1 / 2, cyclic=cyclic)
    H += j, 'Z', 'Z'
    H -= bx, 'X'
    return H


def MPO_ham_ising(L, j=1.0, bx=0.0, *, S=1 / 2, cyclic=False, **mpo_opts):
    r"""Ising Hamiltonian in MPO form.

    .. math::

        H_\mathrm{Ising} =
        J \sum_{i} \sigma^Z_i \sigma^Z_{i + 1} -
        B_x \sum_{i} \sigma^X_i

    Note the default convention of antiferromagnetic interactions and spin
    operators not Pauli matrices.

    Parameters
    ----------
    L : int
        The number of sites.
    j : float, optional
        The ZZ interaction strength. Positive is antiferromagnetic.
    bx : float, optional
        The X-magnetic field strength.
    S : {1/2, 1, 3/2, ...}, optional
        The underlying spin of the system, defaults to 1/2.
    cyclic : bool, optional
        Generate a MPO with periodic boundary conditions or not, default is
        open boundary conditions.
    mpo_opts or local_ham_1d_opts
        Supplied to :class:`~quimb.tensor.tensor_1d.MatrixProductOperator`.

    Returns
    -------
    MatrixProductOperator
    """
    H = _ham_ising(j=j, bx=bx, S=S, cyclic=cyclic)
    return H.build_mpo(L, **mpo_opts)


def ham_1d_ising(L=None, j=1.0, bx=0.0, *, S=1 / 2,
                 cyclic=False, **local_ham_1d_opts):
    r"""Ising Hamiltonian in
    :class:`~quimb.tensor.tensor_1d_tebd.LocalHam1D` form.

    .. math::

        H_\mathrm{Ising} =
        J \sum_{i} \sigma^Z_i \sigma^Z_{i + 1} -
        B_x \sum_{i} \sigma^X_i

    Note the default convention of antiferromagnetic interactions and spin
    operators not Pauli matrices.

    Parameters
    ----------
    L : int
        The number of sites.
    j : float, optional
        The ZZ interaction strength. Positive is antiferromagnetic.
    bx : float, optional
        The X-magnetic field strength.
    S : {1/2, 1, 3/2, ...}, optional
        The underlying spin of the system, defaults to 1/2.
    cyclic : bool, optional
        Generate a hamiltonian with periodic boundary conditions or not,
        default is open boundary conditions.
    mpo_opts or local_ham_1d_opts
        Supplied to :class:`~quimb.tensor.tensor_1d.LocalHam1D`.

    Returns
    -------
    LocalHam1D
    """
    H = _ham_ising(j=j, bx=bx, S=S, cyclic=cyclic)
    return H.build_local_ham(L=L, **local_ham_1d_opts)


NNI_ham_ising = deprecated(ham_1d_ising, 'NNI_ham_ising', 'ham_1d_ising')


def _ham_XY(j=1.0, bz=0.0, *, S=1 / 2, cyclic=False):
    H = SpinHam1D(S=S, cyclic=cyclic)

    try:
        jx, jy = j
    except (TypeError, ValueError):
        jx = jy = j

    if jx == jy:
        # easy way to enforce realness
        H += jx / 2, '+', '-'
        H += jx / 2, '-', '+'
    else:
        H += jx, 'X', 'X'
        H += jy, 'Y', 'Y'

    H -= bz, 'Z'

    return H


def MPO_ham_XY(L, j=1.0, bz=0.0, *, S=1 / 2, cyclic=False, **mpo_opts):
    r"""XY-Hamiltonian in MPO form.

    .. math::

        H_\mathrm{XY} =
        \sum_{i} (
            J_X \sigma^X_i \sigma^X_{i + 1} +
            J_Y \sigma^Y_i \sigma^Y_{i + 1}
            )
        - B_x \sum_{i} \sigma^Z_i

    Note the default convention of antiferromagnetic interactions and spin
    operators not Pauli matrices.

    Parameters
    ----------
    L : int
        The number of sites.
    j : float or (float, float), optional
        The XX and YY interaction strength. Positive is antiferromagnetic.
    bz : float, optional
        The Z-magnetic field strength.
    S : {1/2, 1, 3/2, ...}, optional
        The underlying spin of the system, defaults to 1/2.
    cyclic : bool, optional
        Generate a MPO with periodic boundary conditions or not, default is
        open boundary conditions.
    mpo_opts or local_ham_1d_opts
        Supplied to :class:`~quimb.tensor.tensor_1d.MatrixProductOperator`.

    Returns
    -------
    MatrixProductOperator
    """
    H = _ham_XY(j=j, bz=bz, S=S, cyclic=cyclic)
    return H.build_mpo(L, **mpo_opts)


def ham_1d_XY(L=None, j=1.0, bz=0.0, *, S=1 / 2,
              cyclic=False, **local_ham_1d_opts):
    r"""XY-Hamiltonian in
    :class:`~quimb.tensor.tensor_1d_tebd.LocalHam1D` form.

    .. math::

        H_\mathrm{XY} =
        \sum_{i} (
            J_X \sigma^X_i \sigma^X_{i + 1} +
            J_Y \sigma^Y_i \sigma^Y_{i + 1}
            )
        - B_Z \sum_{i} \sigma^Z_i

    Note the default convention of antiferromagnetic interactions and spin
    operators not Pauli matrices.

    Parameters
    ----------
    L : int
        The number of sites.
    j : float or (float, float), optional
        The XX and YY interaction strength. Positive is antiferromagnetic.
    bz : float, optional
        The Z-magnetic field strength.
    S : {1/2, 1, 3/2, ...}, optional
        The underlying spin of the system, defaults to 1/2.
    cyclic : bool, optional
        Generate a hamiltonian with periodic boundary conditions or not,
        default is open boundary conditions.
    local_ham_1d_opts
        Supplied to :class:`~quimb.tensor.tensor_1d.LocalHam1D`.

    Returns
    -------
    LocalHam1D
    """
    H = _ham_XY(j=j, bz=bz, S=S, cyclic=cyclic)
    return H.build_local_ham(L=L, **local_ham_1d_opts)


NNI_ham_XY = deprecated(ham_1d_XY, 'NNI_ham_XY', 'ham_1d_XY')


def _ham_heis(j=1.0, bz=0.0, *, S=1 / 2, cyclic=False):
    H = SpinHam1D(S=S, cyclic=cyclic)

    try:
        jx, jy, jz = j
    except (TypeError, ValueError):
        jx = jy = jz = j

    if jx == jy:
        # easy way to enforce realness
        H += jx / 2, '+', '-'
        H += jx / 2, '-', '+'
    else:
        H += jx, 'X', 'X'
        H += jy, 'Y', 'Y'
    H += jz, 'Z', 'Z'

    H -= bz, 'Z'

    return H


def MPO_ham_heis(L, j=1.0, bz=0.0, *, S=1 / 2, cyclic=False, **mpo_opts):
    r"""Heisenberg Hamiltonian in MPO form.

    .. math::

        H_\mathrm{Heis} =
        \sum_{i} (
            J_X \sigma^X_i \sigma^X_{i + 1} +
            J_Y \sigma^Y_i \sigma^Y_{i + 1} +
            J_Z \sigma^Z_i \sigma^Z_{i + 1}
            )
        - B_Z \sum_{i} \sigma^Z_i

    Note the default convention of antiferromagnetic interactions and spin
    operators not Pauli matrices.

    Parameters
    ----------
    L : int
        The number of sites.
    j : float or (float, float, float), optional
        The XX, YY and ZZ interaction strength. Positive is antiferromagnetic.
    bz : float, optional
        The Z-magnetic field strength.
    S : {1/2, 1, 3/2, ...}, optional
        The underlying spin of the system, defaults to 1/2.
    cyclic : bool, optional
        Generate a MPO with periodic boundary conditions or not, default is
        open boundary conditions.
    mpo_opts
        Supplied to :class:`~quimb.tensor.tensor_1d.MatrixProductOperator`.

    Returns
    -------
    MatrixProductOperator
    """
    H = _ham_heis(j=j, bz=bz, S=S, cyclic=cyclic)
    return H.build_mpo(L, **mpo_opts)


def ham_1d_heis(L=None, j=1.0, bz=0.0, *, S=1 / 2,
                cyclic=False, **local_ham_1d_opts):
    r"""Heisenberg Hamiltonian in
    :class:`~quimb.tensor.tensor_1d_tebd.LocalHam1D` form.

    .. math::

        H_\mathrm{Heis} =
        \sum_{i} (
            J_X \sigma^X_i \sigma^X_{i + 1} +
            J_Y \sigma^Y_i \sigma^Y_{i + 1} +
            J_Z \sigma^Z_i \sigma^Z_{i + 1}
            )
        - B_Z \sum_{i} \sigma^Z_i

    Note the default convention of antiferromagnetic interactions and spin
    operators not Pauli matrices.

    Parameters
    ----------
    L : int
        The number of sites.
    j : float or (float, float, float), optional
        The XX, YY and ZZ interaction strength. Positive is antiferromagnetic.
    bz : float, optional
        The Z-magnetic field strength.
    S : {1/2, 1, 3/2, ...}, optional
        The underlying spin of the system, defaults to 1/2.
    cyclic : bool, optional
        Generate a hamiltonian with periodic boundary conditions or not,
        default is open boundary conditions.
    local_ham_1d_opts
        Supplied to :class:`~quimb.tensor.tensor_1d_tebd.LocalHam1D`.

    Returns
    -------
    LocalHam1D
    """
    H = _ham_heis(j=j, bz=bz, S=S, cyclic=cyclic)
    return H.build_local_ham(L=L, **local_ham_1d_opts)


NNI_ham_heis = deprecated(ham_1d_heis, 'NNI_ham_heis', 'ham_1d_heis')


def MPO_ham_XXZ(L, delta, jxy=1.0, *, S=1 / 2, cyclic=False, **mpo_opts):
    r"""XXZ-Hamiltonian in MPO form.

    .. math::

        H_\mathrm{XXZ} =
        \sum_{i} (
            J_X \sigma^X_i \sigma^X_{i + 1} +
            J_Y \sigma^Y_i \sigma^Y_{i + 1} +
            \Delta \sigma^Z_i \sigma^Z_{i + 1}
            )
        - B_Z \sum_{i} \sigma^Z_i

    Note the default convention of antiferromagnetic interactions and spin
    operators not Pauli matrices.

    Parameters
    ----------
    L : int
        The number of sites.
    delta : float
        The ZZ-interaction strength. Positive is antiferromagnetic.
    jxy : float, optional
        The X- and Y- interaction strength, defaults to 1. Positive is
        antiferromagnetic.
    S : {1/2, 1, 3/2, ...}, optional
        The underlying spin of the system, defaults to 1/2.
    cyclic : bool, optional
        Generate a MPO with periodic boundary conditions or not, default is
        open boundary conditions.
    mpo_opts
        Supplied to :class:`~quimb.tensor.tensor_1d.MatrixProductOperator`.

    Returns
    -------
    MatrixProductOperator
    """
    return MPO_ham_heis(L, j=(jxy, jxy, delta), S=S, cyclic=cyclic, **mpo_opts)


def ham_1d_XXZ(L=None, delta=None, jxy=1.0, *,
               S=1 / 2, cyclic=False, **local_ham_1d_opts):
    r"""XXZ-Hamiltonian in
    :class:`~quimb.tensor.tensor_1d_tebd.LocalHam1D` form.

    .. math::

        H_\mathrm{XXZ} =
        \sum_{i} (
            J_X \sigma^X_i \sigma^X_{i + 1} +
            J_Y \sigma^Y_i \sigma^Y_{i + 1} +
            \Delta \sigma^Z_i \sigma^Z_{i + 1}
            )
        - B_Z \sum_{i} \sigma^Z_i

    Note the default convention of antiferromagnetic interactions and spin
    operators not Pauli matrices.

    Parameters
    ----------
    L : int
        The number of sites.
    delta : float
        The ZZ-interaction strength. Positive is antiferromagnetic.
    jxy : float, optional
        The X- and Y- interaction strength, defaults to 1.  Positive is
        antiferromagnetic.
    S : {1/2, 1, 3/2, ...}, optional
        The underlying spin of the system, defaults to 1/2.
    cyclic : bool, optional
        Generate a hamiltonian with periodic boundary conditions or not,
        default is open boundary conditions.
    local_ham_1d_opts
        Supplied to :class:`~quimb.tensor.tensor_1d_tebd.LocalHam1D`.

    Returns
    -------
    LocalHam1D
    """
    if delta is None:
        raise ValueError("You need to specify ``delta``.")
    return ham_1d_heis(L, j=(jxy, jxy, delta), S=S,
                       cyclic=cyclic, **local_ham_1d_opts)


NNI_ham_XXZ = deprecated(ham_1d_XXZ, 'NNI_ham_XXZ', 'ham_1d_XXZ')


def _ham_bilinear_biquadratic(theta, *, S=1 / 2, cyclic=False):
    H = SpinHam1D(S=S, cyclic=cyclic)

    H += np.cos(theta), 'X', 'X'
    H += np.cos(theta), 'Y', 'Y'
    H += np.cos(theta), 'Z', 'Z'

    XX = spin_operator('X', S=S) @ spin_operator('X', S=S)
    YY = spin_operator('Y', S=S) @ spin_operator('Y', S=S)
    ZZ = spin_operator('Z', S=S) @ spin_operator('Z', S=S)

    H += np.sin(theta), XX, XX
    H += np.sin(theta), XX, YY
    H += np.sin(theta), XX, ZZ
    H += np.sin(theta), YY, XX
    H += np.sin(theta), YY, YY
    H += np.sin(theta), YY, ZZ
    H += np.sin(theta), ZZ, XX
    H += np.sin(theta), ZZ, YY
    H += np.sin(theta), ZZ, ZZ

    return H


def MPO_ham_bilinear_biquadratic(L=None, theta=0, *, S=1 / 2, cyclic=False,
                                 compress=True, **mpo_opts):
    """ Hamiltonian of one-dimensional bilinear biquadratic chain in MPO form,
    see PhysRevB.93.184428.

    Parameters
    ----------
    L : int
        The number of sites.
    theta : float or (float, float), optional
        The parameter for linear and non-linear term of interaction strength,
        defaults to 0.
    S : {1/2, 1, 3/2, ...}, optional
        The underlying spin of the system, defaults to 1/2.
    cyclic : bool, optional
        Generate a hamiltonian with periodic boundary conditions or not,
        default is open boundary conditions.
    mpo_opts
        Supplied to :class:`~quimb.tensor.tensor_1d.MatrixProductOperator`.

    Returns
    -------
    MatrixProductOperator
    """
    H = _ham_bilinear_biquadratic(theta, S=S, cyclic=cyclic)
    H_mpo = H.build_mpo(L, **mpo_opts)
    if compress is True:
        H_mpo.compress(cutoff=1e-12, cutoff_mode='rel' if cyclic else 'sum2')
    return H_mpo


def ham_1d_bilinear_biquadratic(L=None, theta=0, *, S=1 / 2,
                                cyclic=False, **local_ham_1d_opts):
    """ Hamiltonian of one-dimensional bilinear biquadratic chain in LocalHam1D
    form, see PhysRevB.93.184428.

    Parameters
    ----------
    L : int
        The number of sites.
    theta : float or (float, float), optional
        The parameter for linear and non-linear term of interaction strength,
        defaults to 0.
    S : {1/2, 1, 3/2, ...}, optional
        The underlying spin of the system, defaults to 1/2.
    cyclic : bool, optional
        Generate a hamiltonian with periodic boundary conditions or not,
        default is open boundary conditions.
    local_ham_1d_opts
        Supplied to :class:`~quimb.tensor.tensor_1d_tebd.LocalHam1D`.

    Returns
    -------
    LocalHam1D
    """
    H = _ham_bilinear_biquadratic(theta, S=S, cyclic=cyclic)
    return H.build_local_ham(L=L, **local_ham_1d_opts)


NNI_ham_bilinear_biquadratic = deprecated(ham_1d_bilinear_biquadratic,
                                          'NNI_ham_bilinear_biquadratic',
                                          'ham_1d_bilinear_biquadratic')


def _ham_mbl(L, dh, j=1.0, seed=None, S=1 / 2, *, cyclic=False,
             dh_dist='s', dh_dim=1, beta=None):
    # start with the heisenberg builder
    H = _ham_heis(j, S=S, cyclic=cyclic)

    dhds, rs = _gen_mbl_random_factors(L, dh, dh_dim, dh_dist, seed, beta)

    # generate noise, potentially in all directions, each with own strength
    for i in range(L):
        dh_r_xyzs = zip(dhds, rs[:, i], 'XYZ')
        for dh, r, xyz in dh_r_xyzs:
            H[i] += dh * r, xyz

    return H


def MPO_ham_mbl(L, dh, j=1.0, seed=None, S=1 / 2, *, cyclic=False,
                dh_dist='s', dh_dim=1, beta=None, **mpo_opts):
    r"""The many-body-localized spin hamiltonian in MPO form.

    .. math::

        H_\mathrm{MBL} =
        \sum_{i} (
            J_X \sigma^X_i \sigma^X_{i + 1} +
            J_Y \sigma^Y_i \sigma^Y_{i + 1} +
            J_Z \sigma^Z_i \sigma^Z_{i + 1}
            )
        - \sum_{i} h_i \sigma^Z_i

    Note the default convention of antiferromagnetic interactions and spin
    operators not Pauli matrices.

    Parameters
    ----------
    L : int
        Number of spins.
    dh : float
        Random noise strength.
    j : float, or (float, float, float), optional
        Interaction strength(s) e.g. 1 or (1., 1., 0.5).  Positive is
        antiferromagnetic.
    seed : int, optional
        Random number to seed the noise with.
    S : {1/2, 1, 3/2, ...}, optional
        The underlying spin of the system, defaults to 1/2.
    cyclic : bool, optional
        Whether to use periodic boundary conditions - default is False.
    dh_dist : {'s', 'g', 'qp'}, optional
        Whether to use sqaure, guassian or quasiperiodic noise.
    beta : float, optional
        Frequency of the quasirandom noise, only if ``dh_dist='qr'``.
    mpo_opts
        Supplied to :class:`MatrixProductOperator`.

    Returns
    -------
    MatrixProductOperator
    """
    H = _ham_mbl(L, dh=dh, j=j, seed=seed, S=S, cyclic=cyclic,
                 dh_dist=dh_dist, dh_dim=dh_dim, beta=beta)
    return H.build_mpo(L, **mpo_opts)


def ham_1d_mbl(L, dh, j=1.0, seed=None, S=1 / 2, *, cyclic=False,
               dh_dist='s', dh_dim=1, beta=None, **local_ham_1d_opts):
    r"""The many-body-localized spin hamiltonian in
    :class:`~quimb.tensor.tensor_1d_tebd.LocalHam1D` form.

    .. math::

        H_\mathrm{MBL} =
        \sum_{i} (
            J_X \sigma^X_i \sigma^X_{i + 1} +
            J_Y \sigma^Y_i \sigma^Y_{i + 1} +
            J_Z \sigma^Z_i \sigma^Z_{i + 1}
            )
        - \sum_{i} h_i \sigma^Z_i

    Note the default convention of antiferromagnetic interactions and spin
    operators not Pauli matrices.

    Parameters
    ----------
    L : int
        Number of spins.
    dh : float
        Random noise strength.
    j : float, or (float, float, float), optional
        Interaction strength(s) e.g. 1 or (1., 1., 0.5).  Positive is
        antiferromagnetic.
    seed : int, optional
        Random number to seed the noise with.
    S : {1/2, 1, 3/2, ...}, optional
        The underlying spin of the system, defaults to 1/2.
    cyclic : bool, optional
        Whether to use periodic boundary conditions - default is False.
    dh_dist : {'s', 'g', 'qp'}, optional
        Whether to use sqaure, guassian or quasiperiodic noise.
    beta : float, optional
        Frequency of the quasirandom noise, only if ``dh_dist='qr'``.
    local_ham_1d_opts
        Supplied to :class:`LocalHam1D`.

    Returns
    -------
    LocalHam1D
    """
    H = _ham_mbl(L, dh=dh, j=j, seed=seed, S=S, cyclic=cyclic,
                 dh_dist=dh_dist, dh_dim=dh_dim, beta=beta)
    return H.build_local_ham(L, **local_ham_1d_opts)


NNI_ham_mbl = deprecated(ham_1d_mbl, 'NNI_ham_mbl', 'ham_1d_mbl')


def ham_2d_ising(Lx, Ly, j=1.0, bx=0.0, **local_ham_2d_opts):
    r"""Ising Hamiltonian in
    :class:`~quimb.tensor.tensor_2d_tebd.LocalHam2D` form.

    .. math::

        H_\mathrm{Ising} =
        J \sum_{<ij>} \sigma^Z_i \sigma^Z_{j} -
        B_x \sum_{i} \sigma^X_i

    for nearest neighbors :math:`<ij>`. Note the default convention of
    antiferromagnetic interactions and spin operators not Pauli matrices.

    Parameters
    ----------
    Lx : int
        The number of rows.
    Ly : int
        The number of columns.
    j : float, optional
        The ZZ interaction strength. Positive is antiferromagnetic.
    bx : float, optional
        The X-magnetic field strength.
    local_ham_2d_opts
        Supplied to :class:`~quimb.tensor.tensor_2d.LocalHam2D`.

    Returns
    -------
    LocalHam2D
    """
    H2 = ham_heis(2, j=(0, 0, j), cyclic=False)
    if bx == 0.0:
        H1 = None
    else:
        H1 = -bx * spin_operator('X').real
    return LocalHam2D(Lx, Ly, H2=H2, H1=H1, **local_ham_2d_opts)


def ham_2d_heis(Lx, Ly, j=1.0, bz=0.0, **local_ham_2d_opts):
    r"""Heisenberg Hamiltonian in
    :class:`~quimb.tensor.tensor_2d_tebd.LocalHam2D`. form.

    .. math::

        H_\mathrm{Heis} =
        \sum_{<ij>} (
            J_X \sigma^X_i \sigma^X_{j} +
            J_Y \sigma^Y_i \sigma^Y_{j} +
            J_Z \sigma^Z_i \sigma^Z_{j}
            )
        - B_Z \sum_{i} \sigma^Z_{i}

    for nearest neighbors :math:`<ij>`. Note the default convention of
    antiferromagnetic interactions and spin operators not Pauli matrices.

    Parameters
    ----------
    Lx : int
        The number of rows.
    Ly : int
        The number of columns.
    j : float or (float, float, float), optional
        The XX, YY and ZZ interaction strength. Positive is antiferromagnetic.
    bz : float, optional
        The Z-magnetic field strength.
    local_ham_2d_opts
        Supplied to :class:`~quimb.tensor.tensor_2d_tebd.LocalHam2D`.

    Returns
    -------
    LocalHam2D
    """
    H2 = ham_heis(2, j=j, cyclic=False)
    if bz == 0.0:
        H1 = None
    else:
        H1 = -bz * spin_operator('Z').real
    return LocalHam2D(Lx, Ly, H2=H2, H1=H1, **local_ham_2d_opts)


def ham_2d_j1j2(Lx, Ly, j1=1.0, j2=0.5, bz=0.0, **local_ham_2d_opts):
    r"""Heisenberg Hamiltonian in
    :class:`~quimb.tensor.tensor_2d_tebd.LocalHam2D`. form.

    .. math::

        H_\mathrm{Heis} =
        \sum_{<ij>} (
            J_{1,X} \sigma^X_i \sigma^X_{j} +
            J_{1,Y} \sigma^Y_i \sigma^Y_{j} +
            J_{1,Z} \sigma^Z_i \sigma^Z_{j}
            )
        +
        \sum_{<<ij>>} (
            J_{2,X} \sigma^X_i \sigma^X_{j} +
            J_{2,Y} \sigma^Y_i \sigma^Y_{j} +
            J_{2,Z} \sigma^Z_i \sigma^Z_{j}
            )
        - B_Z \sum_{i} \sigma^Z_{i}

    for nearest neighbors :math:`<ij>` and diagonal next nearest neighbors
    :math:`<<ij>>`. Note the default convention of antiferromagnetic
    interactions and spin operators not Pauli matrices.

    Parameters
    ----------
    Lx : int
        The number of rows.
    Ly : int
        The number of columns.
    j2 : float or (float, float, float), optional
        The nearest neighbor  XX, YY and ZZ interaction strength. Positive is
        antiferromagnetic.
    j2 : float or (float, float, float), optional
        The diagonal next nearest nearest XX, YY and ZZ interaction strength.
        Positive is antiferromagnetic.
    bz : float, optional
        The Z-magnetic field strength.
    local_ham_2d_opts
        Supplied to :class:`~quimb.tensor.tensor_2d_tebd.LocalHam2D`.

    Returns
    -------
    LocalHam2D
    """

    # default nearest neighbor interaction
    H2 = {None: ham_heis(2, j=j1, cyclic=False)}

    # diagonal next nearest neighbor interactions
    for bond in gen_2d_bonds(Lx, Ly, steppers=[
        lambda i, j: (i + 1, j - 1),
        lambda i, j: (i + 1, j + 1),
    ]):
        H2[bond] = ham_heis(2, j=j2, cyclic=False)

    # magnetic field
    if bz == 0.0:
        H1 = None
    else:
        H1 = -bz * spin_operator('Z').real

    return LocalHam2D(Lx, Ly, H2=H2, H1=H1, **local_ham_2d_opts)


def ham_3d_heis(Lx, Ly, Lz, j=1.0, bz=0.0, **local_ham_3d_opts):
    r"""Heisenberg Hamiltonian in
    :class:`~quimb.tensor.tensor_3d_tebd.LocalHam3D`. form.

    .. math::

        H_\mathrm{Heis} =
        \sum_{<ij>} (
            J_X \sigma^X_i \sigma^X_{j} +
            J_Y \sigma^Y_i \sigma^Y_{j} +
            J_Z \sigma^Z_i \sigma^Z_{j}
            )
        - B_Z \sum_{i} \sigma^Z_{i}

    for nearest neighbors :math:`<ij>`. Note the default convention of
    antiferromagnetic interactions and spin operators not Pauli matrices.

    Parameters
    ----------
    Lx : int
        The number of x-planes.
    Ly : int
        The number of y-planes.
    Ly : int
        The number of z-planes.
    j : float or (float, float, float), optional
        The XX, YY and ZZ interaction strength. Positive is antiferromagnetic.
    bz : float, optional
        The Z-magnetic field strength.
    local_ham_3d_opts
        Supplied to :class:`~quimb.tensor.tensor_3d_tebd.LocalHam3D`.

    Returns
    -------
    LocalHam3D
    """
    H2 = ham_heis(2, j=j, cyclic=False)
    if bz == 0.0:
        H1 = None
    else:
        H1 = -bz * spin_operator('Z').real
    return LocalHam3D(Lx, Ly, Lz, H2=H2, H1=H1, **local_ham_3d_opts)
