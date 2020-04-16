import random
import itertools
import collections

from autoray import do, to_numpy, infer_backend, get_dtype_name
import tqdm
import numpy as np

from ..core import eye, kron, qarray
from ..linalg.base_linalg import expm
from .graphing import get_colors
from .tensor_core import Tensor


class LocalHam2D:
    """A 2D Hamiltonian represented as local terms. This combines all two site
    and one site terms into a single interaction per lattice pair, and caches
    operations on the terms such as getting their exponentiatial.

    Parameters
    ----------
    Lx : int
        The number of rows.
    Ly : int
        The number of columns.
    H2 : array_like or dict[tuple[tuple[int]], array_like]
        The two site term(s). If a single array is given, assume to be the
        default interaction for all nearest neighbours. If a dict is supplied,
        the keys should represent specific pairs of coordinates like
        ``((ia, ja), (ib, jb))`` with the values the array representing the
        interaction for that pair. A default term for all remaining nearest
        neighbours interactions can still be supplied with the key ``None``.
    H1 : array_like or dict[tuple[int], array_like], optional
        The one site term(s). If a single array is given, assume to be the
        default onsite term for all terms. If a dict is supplied,
        the keys should represent specific coordinates like
        ``(i, j)`` with the values the array representing the local term for
        that site. A default term for all remaining sites can still be supplied
        with the key ``None``.

    Attributes
    ----------
    terms : dict[tuple[tuple[int]], array_like]
        The total effective local term for each interaction (with single site
        terms appropriately absorbed). Each key is a pair of coordinates
        ``ija, ijb`` with ``ija < ijb``.
    """

    def __init__(self, Lx, Ly, H2, H1=None):
        self.Lx = int(Lx)
        self.Ly = int(Ly)

        # caches for not repeating operations / duplicating tensors
        self._op_cache = collections.defaultdict(dict)

        # parse two site terms
        if hasattr(H2, 'shape'):
            # use as default nearest neighbour term
            self.terms = {None: H2}
        else:
            self.terms = dict(H2)

        # convert qarrays (mostly useful for working with jax)
        for key, X in self.terms.items():
            if isinstance(X, qarray):
                self.terms[key] = X.A

        # first combine terms to ensure coo1 < coo2
        for where in tuple(filter(bool, self.terms)):
            coo1, coo2 = where
            if coo1 < coo2:
                continue

            # pop and flip the term
            X12 = self._flip_cached(self.terms.pop(where))

            # add to, or create, term with flipped coos
            new_where = coo2, coo1
            if new_where in self.terms:
                self.terms[new_where] = (
                    self._add_cached(self.terms[new_where], X12)
                )
            else:
                self.terms[new_where] = X12

        # possibly fill in default gates
        default_H2 = self.terms.pop(None, None)
        if default_H2 is not None:
            for i, j in itertools.product(range(self.Lx), range(self.Ly)):
                if i + 1 < self.Lx:
                    where = ((i, j), (i + 1, j))
                    self.terms.setdefault(where, default_H2)
                if j + 1 < self.Ly:
                    where = ((i, j), (i, j + 1))
                    self.terms.setdefault(where, default_H2)

        # make a directory of which single sites are covered by which terms
        #     - to merge them into later
        self._sites_to_covering_terms = collections.defaultdict(list)
        for where in self.terms:
            ij1, ij2 = where
            self._sites_to_covering_terms[ij1].append(where)
            self._sites_to_covering_terms[ij2].append(where)

        # parse one site terms
        if H1 is None:
            H1s = dict()
        elif hasattr(H1, 'shape'):
            # set a default site term
            H1s = {None: H1}
        else:
            H1s = dict(H1)

        # convert qarrays (mostly useful for working with jax)
        for key, X in H1s.items():
            if isinstance(X, qarray):
                H1[key] = X.A

        # possibly set the default single site term
        default_H1 = H1s.pop(None, None)
        if default_H1 is not None:
            for i, j in itertools.product(range(self.Lx), range(self.Ly)):
                H1s.setdefault((i, j), default_H1)

        # now absorb the single site terms evenly into the two site terms
        for (i, j), H in H1s.items():

            # get interacting terms which cover the site
            pairs = self._sites_to_covering_terms[i, j]
            np = len(pairs)
            if np == 0:
                raise ValueError(
                    f"There are no two site terms to add this single site "
                    f"term to - site {(i, j)} is not coupled to anything.")

            # merge the single site term in equal parts into all covering pairs
            H_tensoreds = (self._op_id_cached(H), self._id_op_cached(H))
            for pair in pairs:
                H_tensored = H_tensoreds[pair.index((i, j))]
                self.terms[pair] = (
                    self._add_cached(
                        self.terms[pair],
                        self._div_cached(H_tensored, np)
                    )
                )

    def _flip_cached(self, x):
        cache = self._op_cache['flip']
        key = id(x)
        if key not in cache:
            d = int(x.size**(1 / 4))
            xf = do('reshape', x, (d, d, d, d))
            xf = do('transpose', xf, (1, 0, 3, 2))
            xf = do('reshape', xf, (d * d, d * d))
            cache[key] = xf
        return cache[key]

    def _add_cached(self, x, y):
        cache = self._op_cache['add']
        key = (id(x), id(y))
        if key not in cache:
            cache[key] = x + y
        return cache[key]

    def _div_cached(self, x, y):
        cache = self._op_cache['div']
        key = (id(x), y)
        if key not in cache:
            cache[key] = x / y
        return cache[key]

    def _op_id_cached(self, x):
        cache = self._op_cache['op_id']
        key = id(x)
        if key not in cache:
            xn = to_numpy(x)
            d = int(xn.size**0.5)
            Id = eye(d, dtype=xn.dtype)
            XI = do('array', kron(xn, Id), like=x)
            cache[key] = XI
        return cache[key]

    def _id_op_cached(self, x):
        cache = self._op_cache['id_op']
        key = id(x)
        if key not in cache:
            xn = to_numpy(x)
            d = int(xn.size**0.5)
            Id = eye(d, dtype=xn.dtype)
            IX = do('array', kron(Id, xn), like=x)
            cache[key] = IX
        return cache[key]

    def _expm_cached(self, x, y):
        cache = self._op_cache['expm']
        key = (id(x), y)
        if key not in cache:
            xn = to_numpy(x)
            cache[key] = do('array', expm(xn * y), like=x)
        return cache[key]

    def get_gate(self, where):
        """Get the local term for pair ``where``, cached.
        """
        return self.terms[tuple(where)]

    def get_gate_expm(self, where, x):
        """Get the local term for pair ``where``, matrix exponentiated by
        ``x``, and cached.
        """
        return self._expm_cached(self.get_gate(where), x)

    def apply_to_arrays(self, fn):
        """Apply the function ``fn`` to all the arrays representing terms.
        """
        for k, x in self.terms.items():
            self.terms[k] = fn(x)

    def _nx_color_ordering(self, strategy='smallest_first', interchange=True):
        """Generate a term ordering based on a coloring on the line graph.
        """
        import networkx as nx

        G = nx.Graph()
        for ija, ijb in self.terms:
            G.add_edge(ija, ijb)

        coloring = list(nx.coloring.greedy_color(
            nx.line_graph(G), strategy, interchange=interchange).items())

        # sort into color groups
        coloring.sort(key=lambda coo_color: coo_color[1])

        return [coo for coo, color in coloring]

    def get_auto_ordering(self, order='sort', **kwargs):
        """Get an ordering of the terms to use with TEBD, for example. The
        default is to sort the coordinates then greedily group them into
        commuting sets.

        Parameters
        ----------
        order : {'sort', None, 'random', str}
            How to order the terms *before* greedily grouping them into
            commuting (non-coordinate overlapping) sets. ``'sort'`` will sort
            the coordinate pairs first. ``None`` will use the current order of
            terms which should match the order they were supplied to this
            ``LocalHam2D`` instance.  ``'random'`` will randomly shuffle the
            coordinate pairs before grouping them - *not* the same as returning
            a completely random order. Any other option will be passed as a
            strategy to ``networkx.coloring.greedy_color`` to generate the
            ordering.

        Returns
        -------
        list[tuple[tuple[int]]]
            Sequence of coordinate pairs.
        """
        if order is None:
            pairs = self.terms
        elif order == 'sort':
            pairs = sorted(self.terms)
        elif order == 'random':
            pairs = list(self.terms)
            random.shuffle(pairs)
        else:
            return self._nx_color_ordering(order, **kwargs)

        pairs = {x: None for x in pairs}

        cover = set()
        ordering = list()
        while pairs:
            for pair in tuple(pairs):
                ij1, ij2 = pair
                if (ij1 not in cover) and (ij2 not in cover):
                    ordering.append(pair)
                    pairs.pop(pair)
                    cover.add(ij1)
                    cover.add(ij2)
            cover.clear()

        return ordering

    def __repr__(self):
        s = "<LocalHam2D(Lx={}, Ly={}, num_terms={})>"
        return s.format(self.Lx, self.Ly, len(self.terms))

    def graph(
        self,
        ordering='sort',
        show_norm=True,
        figsize=None,
        fontsize=8,
        legend=True,
        ax=None,
        return_fig=False,
        **kwargs,
    ):
        """Plot this Hamiltonian as a network.

        Parameters
        ----------
        ordering : {'sort', None, 'random'}, optional
            An ordering of the termns, or an argument to be supplied to
            :meth:`quimb.tensor.tensor_2d_tebd.LocalHam2D.get_auto_ordering`
            to generate this automatically.
        show_norm : bool, optional
            Show the norm of each term as edge labels.
        figsize : None or tuple[int], optional
            Size of the figure, defaults to size of Hamiltonian.
        fontsize : int, optional
            Font size for norm labels.
        legend : bool, optional
            Whether to show the legend of which terms are in which group.
        ax : None or matplotlib.Axes, optional
            Add to a existing set of axes.
        return_fig : bool, optional
            Whether to return any newly created figure.
        """
        import matplotlib.pyplot as plt

        if figsize is None:
            figsize = (self.Ly, self.Lx)

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
            ax.axis('off')
            ax.set_aspect('equal')

        if ordering is None or isinstance(ordering, str):
            ordering = self.get_auto_ordering(ordering, **kwargs)

        data = []
        seen = set()
        n = 0
        for ij1, ij2 in ordering:
            if (ij1 in seen) or (ij2 in seen):
                # start a new group
                seen = {ij1, ij2}
                n += 1
            else:
                seen.add(ij1)
                seen.add(ij2)

            ys, xs = zip(ij1, ij2)
            nrm = do('linalg.norm', self.terms[ij1, ij2])

            # set coordinates for label with some offset towards left
            if ij1[1] < ij2[1]:
                x0 = (3 * xs[0] + 2 * xs[1]) / 5
                y0 = (3 * ys[0] + 2 * ys[1]) / 5
            else:
                x0 = (2 * xs[0] + 3 * xs[1]) / 5
                y0 = (2 * ys[0] + 3 * ys[1]) / 5

            data.append((xs, ys, n, x0, y0, nrm))

        num_groups = n + 1
        colors = get_colors(range(num_groups))

        # do the plotting
        for xs, ys, n, x0, y0, nrm in data:
            ax.plot(xs, ys, c=colors[n], linewidth=2 * nrm**0.5)
            if show_norm:
                label = "{:.3f}".format(nrm)
                ax.text(x0, y0, label, c=colors[n], fontsize=fontsize)

        # create legend
        if legend:
            handles = []
            for color in colors.values():
                handles += [plt.Line2D([0], [0], marker='o', color=color,
                                       linestyle='', markersize=10)]

            lbls = [f"Group {i + 1}" for i in range(num_groups)]

            ax.legend(handles, lbls, ncol=max(round(len(handles) / 20), 1),
                      loc='center left', bbox_to_anchor=(1, 0.5))

        if ax is not None:
            return
        elif return_fig:
            return fig
        else:
            plt.show()


class TEBD2D:
    """Generic class for performing two dimensional time evolving block
    decimation, i.e. applying the exponential of a Hamiltonian using
    a product formula that involves applying local exponentiated gates only.

    Parameters
    ----------
    psi0 : TensorNetwork2DVector
        The initial state.
    ham : LocalHam2D
        The Hamtiltonian consisting of local terms.
    tau : float, optional
        The default local exponent, if considered as time real values here
        imply imaginary time.
    max_bond : {'psi0', int, None}, optional
        The maximum bond dimension to keep when applying each gate.
    gate_opts : dict, optional
        Supplied to :meth:`quimb.tensor.tensor_2d.TensorNetwork2DVector.gate`,
        in addition to ``max_bond``. By default ``contract`` is set to
        'reduce-split' and ``cutoff`` is set to ``0.0``.
    ordering : str or tuple[tuple[int]], optional
        How to order the terms, if a string is given then use this as the
        strategy given to
        :meth:`~quimb.tensor.tensor_2d_tebd.LocalHam2D.get_auto_ordering`. An
        explicit list of coordinate pairs can also be given. The default is to
        greedily form an 'edge coloring' based on the sorted list of
        Hamiltonian pair coordinates.
    compute_energy_every : None or int, optional
        How often to compute and record the energy. If a positive integer 'n',
        the energy is computed *before* every nth sweep (i.e. including before
        the zeroth).
    compute_energy_final : bool, optional
        Whether to compute and record the energy at the end of the sweeps
        regardless of the value of ``compute_energy_every``. If you start
        sweeping again then this final energy is the same as the zeroth of the
        next set of sweeps and won't be recomputed.
    compute_energy_opts : dict, optional
        Supplied to
        :meth:`~quimb.tensor.tensor_2d.PEPS.compute_local_expectation`. By
        default ``max_bond`` is set to ``max(8, D**2)`` where ``D`` is the
        maximum bond to use for applying the gate, ``cutoff`` is set to ``0.0``
        and ``normalized`` is set to ``True``.
    compute_energy_fn : callable, optional
        Supply your own function to compute the energy, it should take the
        ``TEBD2D`` object as its only argument.
    callback : callable, optional
        A custom callback to run after every sweep, it should take the
        ``TEBD2D`` object as its only argument. If it returns any value that
        boolean evaluates to ``True`` then terminal the evolution.
    progbar : boolean, optional
        Whether to show a live progress bar during the evolution.

    Attributes
    ----------
    state : TensorNetwork2DVector
        The current state.
    ham : LocalHam2D
        The Hamiltonian being used to evolve.
    energy : float
        The current of the current state, this will trigger a computation if
        the energy at this iteration hasn't been computed yet.
    energies : list[float]
        The energies that have been computed, if any.
    its : list[int]
        The corresponding sequence of iteration numbers that energies have been
        computed at.
    taus : list[float]
        The corresponding sequence of time steps that energies have been
        computed at.
    """

    def __init__(
        self,
        psi0,
        ham,
        tau=0.01,
        max_bond='psi0',
        gate_opts=None,
        ordering='sort',
        compute_energy_every=None,
        compute_energy_final=True,
        compute_energy_opts=None,
        compute_energy_fn=None,
        callback=None,
        progbar=True,
    ):

        self._psi = psi0.copy()
        self.ham = ham
        self.progbar = progbar
        self.callback = callback

        # default time step to use
        self.tau = tau

        # parse gate application options
        if max_bond == 'psi0':
            max_bond = self._psi.max_bond()
        self.gate_opts = (
            dict() if gate_opts is None else
            dict(gate_opts))
        self.gate_opts.setdefault('contract', 'reduce-split')
        self.gate_opts.setdefault('max_bond', max_bond)
        self.gate_opts.setdefault('cutoff', 0.0)

        # parse energy computation options
        self.compute_energy_every = compute_energy_every
        self.compute_energy_final = compute_energy_final
        self.compute_energy_fn = compute_energy_fn
        self.compute_energy_opts = (
            dict() if compute_energy_opts is None else
            dict(compute_energy_opts))
        self.compute_energy_opts.setdefault('max_bond', max(8, max_bond**2))
        self.compute_energy_opts.setdefault('cutoff', 0.0)
        self.compute_energy_opts.setdefault('normalized', True)

        if ordering is None or isinstance(ordering, str):
            self.ordering = self.ham.get_auto_ordering(ordering)
        else:
            self.ordering = tuple(ordering)

        # storage
        self._n = 0
        self.its = []
        self.taus = []
        self.energies = []

    @property
    def n(self):
        """The number of sweeps performed.
        """
        return self._n

    def gate(self, U, where):
        """Perform single gate ``U`` at coordinate pair ``where``.
        """
        self._psi.gate_(U, where, **self.gate_opts)

    def sweep(self):
        """Perform a full sweep of gates at every pair.
        """
        for where in self.ordering:
            U = self.ham.get_gate_expm(where, -self.tau)
            self.gate(U, where)

    def _compute_energy(self):
        """
        """
        if self.its and (self._n == self.its[-1]):
            # only compute if haven't already
            return self.energies[-1]

        if self.compute_energy_fn is not None:
            en = self.compute_energy_fn(self)
        else:
            en = self.state.compute_local_expectation(
                self.ham.terms, **self.compute_energy_opts)

        self.energies.append(en)
        self.taus.append(self.tau)
        self.its.append(self._n)

        return self.energies[-1]

    def _update_progbar(self, pbar):
        desc = f"n={self._n}, tau={self.tau}, energy={float(self.energy):.6f}"
        pbar.set_description(desc)

    def evolve(self, steps, tau=None):
        """
        """
        if tau is not None:
            self.tau = tau

        pbar = tqdm.tqdm(total=steps, disable=not self.progbar)

        try:
            for i in range(steps):
                # possibly compute the energy
                should_compute_energy = (
                    bool(self.compute_energy_every) and
                    (i % self.compute_energy_every == 0))
                if should_compute_energy:
                    self._compute_energy()
                    self._update_progbar(pbar)

                # actually perform the gates
                self.sweep()
                self._n += 1
                pbar.update()

                if self.callback is not None:
                    if self.callback(self):
                        break

            # possibly compute the energy
            if self.compute_energy_final:
                self._compute_energy()
                self._update_progbar(pbar)

        except KeyboardInterrupt:
            # allow the user to interupt early
            pass
        finally:
            pbar.close()

    def get_state(self):
        return self._psi.copy()

    @property
    def state(self):
        """Return a copy of the current state.
        """
        return self.get_state()

    @property
    def energy(self):
        """Return the energy of current state, computing it only if necessary.
        """
        return self._compute_energy()


class SimpleUpdate(TEBD2D):
    """A simple subclass of ``TEBD2D`` that overrides two key methods in order
    to keep 'diagonal gauges' living on the bonds of a PEPS. The gauges are
    stored separately from the main PEPS in the ``gauges`` attribute. Before
    and after a gate is applied they are absorbed and then extracted. When
    accessing the ``state`` attribute they are automatically inserted or you
    can call ``get_state(absorb_gauges=False)`` to lazily add them as hyperedge
    weights only. Reference: https://arxiv.org/abs/0806.3719.

    """ + TEBD2D.__doc__

    def _initialize_gauges(self):
        """Create unit singular values, stored as tensors.
        """
        data00 = self._psi[0, 0].data

        # create the gauges like whatever data array is in the first site.
        backend = infer_backend(data00)
        dtype = get_dtype_name(data00)

        self._gauges = dict()
        for ija, ijb in self._psi.gen_bond_coos():
            bnd = self._psi.bond(ija, ijb)
            d = self._psi.ind_size(bnd)
            Tsval = Tensor(
                do('array', np.ones((d,), dtype=dtype), like=backend),
                inds=[bnd],
                tags=[self._psi.site_tag(*ija), self._psi.site_tag(*ijb)]
            )
            self._gauges[tuple(sorted((ija, ijb)))] = Tsval

    @property
    def gauges(self):
        """The dictionary of bond pair coordinates to Tensors describing the
        weights (``t = gauges[pair]; t.data``) and index
        (``t = gauges[pair]; t.inds[0]``) of all the gauges.
        """
        try:
            return self._gauges
        except AttributeError:
            self._initialize_gauges()
            return self._gauges

    def gate(self, U, where):
        """Like ``TEBD2D.gate`` but absorb and extract the relevant gauges
        before and after each gate application.
        """
        ija, ijb = where
        ia, ja = ija
        ib, jb = ijb

        Ta = self._psi[ija]
        Tb = self._psi[ijb]

        # absorb the 'outer' gauges of each site
        coo_a_neighbours = tuple(filter(
            lambda coo: self._psi.valid_coo(coo) and coo != ijb,
            [(ia + 1, ja), (ia - 1, ja), (ia, ja + 1), (ia, ja - 1)]
        ))
        for coo in coo_a_neighbours:
            Tsval = self.gauges[tuple(sorted((ija, coo)))]
            Ta.multiply_index_diagonal_(Tsval.inds[0], Tsval.data)

        coo_b_neighbours = tuple(filter(
            lambda coo: self._psi.valid_coo(coo) and coo != ija,
            [(ib + 1, jb), (ib - 1, jb), (ib, jb + 1), (ib, jb - 1)]
        ))
        for coo in coo_b_neighbours:
            Tsval = self.gauges[tuple(sorted((ijb, coo)))]
            Tb.multiply_index_diagonal_(Tsval.inds[0], Tsval.data)

        # absorb the bond gauge equally into both sites
        Tsval = self.gauges[ija, ijb]
        bnd, = Tsval.inds
        Ta.multiply_index_diagonal_(bnd, Tsval.data**0.5)
        Tb.multiply_index_diagonal_(bnd, Tsval.data**0.5)

        # perform the gate, retrieving new bond singular values
        info = dict()
        self._psi.gate_(U, where, absorb=None, info=info, **self.gate_opts)

        s = info['singular_values']
        # keep the singular values from blowing up
        s = s / do('sum', s)
        Tsval.modify(data=s)

        # 'extract' the outer gauges again
        for coo in coo_a_neighbours:
            Tsval = self.gauges[tuple(sorted((ija, coo)))]
            Ta.multiply_index_diagonal_(Tsval.inds[0], Tsval.data**-1)
        for coo in coo_b_neighbours:
            Tsval = self.gauges[tuple(sorted((ijb, coo)))]
            Tb.multiply_index_diagonal_(Tsval.inds[0], Tsval.data**-1)

    def get_state(self, absorb_gauges=True):
        """Return the state, with the diagonal bond gauges either absorbed
        equally into the tensors on either side of them
        (``absorb_gauges=True``, the default), or left lazily represented in
        the tensor network with hyperedges (``absorb_gauges=False``).
        """
        psi = self._psi.copy()

        if not absorb_gauges:
            for Tsval in self.gauges.values():
                psi &= Tsval
        else:
            for (ija, ijb), Tsval in self.gauges.items():
                bnd, = Tsval.inds
                Ta = psi[ija]
                Tb = psi[ijb]
                Ta.multiply_index_diagonal_(bnd, Tsval.data**0.5)
                Tb.multiply_index_diagonal_(bnd, Tsval.data**0.5)

        return psi
