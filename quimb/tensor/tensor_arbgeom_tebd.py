import random
import itertools
import collections

from autoray import do, to_numpy, dag

from ..core import eye, kron, qarray
from ..utils import ensure_dict
from ..utils import progbar as Progbar
from .tensor_core import Tensor
from .drawing import get_colors, get_positions


class LocalHamGen:
    """Representation of a local hamiltonian defined on a general graph. This
    combines all two site and one site terms into a single interaction per
    lattice pair, and caches operations on the terms such as getting their
    exponential. The sites (nodes) should be hashable and comparable.

    Parameters
    ----------
    H2 : dict[tuple[node], array_like]
        The interaction terms, with each key being an tuple of nodes defining
        an edge and each value the local hamilotonian term for those two nodes.
    H1 : array_like or dict[node, array_like], optional
        The one site term(s). If a single array is given, assume to be the
        default onsite term for all terms. If a dict is supplied,
        the keys should represent specific coordinates like
        ``(i, j)`` with the values the array representing the local term for
        that site. A default term for all remaining sites can still be supplied
        with the key ``None``.

    Attributes
    ----------
    terms : dict[tuple, array_like]
        The total effective local term for each interaction (with single site
        terms appropriately absorbed). Each key is a pair of coordinates
        ``site_a, site_b`` with ``site_a < site_b``.
    """

    def __init__(self, H2, H1=None):
        # caches for not repeating operations / duplicating tensors
        self._op_cache = collections.defaultdict(dict)

        self.terms = dict(H2)

        # convert qarrays (mostly useful for working with jax)
        for key, X in self.terms.items():
            if isinstance(X, qarray):
                self.terms[key] = self._convert_from_qarray_cached(X)

        self.sites = tuple(
            sorted(set(itertools.chain.from_iterable(self.terms)))
        )

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
                self.terms[new_where] = self._add_cached(
                    self.terms[new_where], X12
                )
            else:
                self.terms[new_where] = X12

        # make a directory of which single sites are covered by which terms
        #     - to merge them into later
        self._sites_to_covering_terms = collections.defaultdict(list)
        for where in self.terms:
            site_a, site_b = where
            self._sites_to_covering_terms[site_a].append(where)
            self._sites_to_covering_terms[site_b].append(where)

        # parse one site terms
        if H1 is None:
            H1s = dict()
        elif hasattr(H1, "shape"):
            # set a default site term
            H1s = {None: H1}
        else:
            H1s = dict(H1)

        # convert qarrays (mostly useful for working with jax)
        for key, X in H1s.items():
            if isinstance(X, qarray):
                H1s[key] = self._convert_from_qarray_cached(X)

        # possibly set the default single site term
        default_H1 = H1s.pop(None, None)
        if default_H1 is not None:
            for site in self.sites:
                H1s.setdefault(site, default_H1)

        # now absorb the single site terms evenly into the two site terms
        for site, H in H1s.items():

            # get interacting terms which cover the site
            pairs = self._sites_to_covering_terms[site]
            num_pairs = len(pairs)
            if num_pairs == 0:
                raise ValueError(
                    f"There are no two site terms to add this single site "
                    f"term to - site {site} is not coupled to anything."
                )

            # merge the single site term in equal parts into all covering pairs
            H_tensoreds = (self._op_id_cached(H), self._id_op_cached(H))
            for pair in pairs:
                H_tensored = H_tensoreds[pair.index(site)]
                self.terms[pair] = self._add_cached(
                    self.terms[pair], self._div_cached(H_tensored, num_pairs)
                )

    @property
    def nsites(self):
        """The number of sites in the system.
        """
        return len(self.sites)

    def items(self):
        """Iterate over all terms in the hamiltonian. This is mostly for
        convenient compatibility with ``compute_local_expectation``.
        """
        return self.terms.items()

    def _convert_from_qarray_cached(self, x):
        cache = self._op_cache["convert_from_qarray"]
        key = id(x)
        if key not in cache:
            cache[key] = x.A
        return cache[key]

    def _flip_cached(self, x):
        cache = self._op_cache["flip"]
        key = id(x)
        if key not in cache:
            d = int(x.size ** (1 / 4))
            xf = do("reshape", x, (d, d, d, d))
            xf = do("transpose", xf, (1, 0, 3, 2))
            xf = do("reshape", xf, (d * d, d * d))
            cache[key] = xf
        return cache[key]

    def _add_cached(self, x, y):
        cache = self._op_cache["add"]
        key = (id(x), id(y))
        if key not in cache:
            cache[key] = x + y
        return cache[key]

    def _div_cached(self, x, y):
        cache = self._op_cache["div"]
        key = (id(x), y)
        if key not in cache:
            cache[key] = x / y
        return cache[key]

    def _op_id_cached(self, x):
        cache = self._op_cache["op_id"]
        key = id(x)
        if key not in cache:
            xn = to_numpy(x)
            d = int(xn.size ** 0.5)
            Id = eye(d, dtype=xn.dtype)
            XI = do("array", kron(xn, Id), like=x)
            cache[key] = XI
        return cache[key]

    def _id_op_cached(self, x):
        cache = self._op_cache["id_op"]
        key = id(x)
        if key not in cache:
            xn = to_numpy(x)
            d = int(xn.size ** 0.5)
            Id = eye(d, dtype=xn.dtype)
            IX = do("array", kron(Id, xn), like=x)
            cache[key] = IX
        return cache[key]

    def _expm_cached(self, x, y):
        cache = self._op_cache["expm"]
        key = (id(x), y)
        if key not in cache:
            el, ev = do("linalg.eigh", x)
            cache[key] = ev @ do("diag", do("exp", el * y)) @ dag(ev)
        return cache[key]

    def get_gate(self, where):
        """Get the local term for pair ``where``, cached.
        """
        return self.terms[tuple(sorted(where))]

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

    def _nx_color_ordering(self, strategy="smallest_first", interchange=True):
        """Generate a term ordering based on a coloring on the line graph.
        """
        import networkx as nx

        G = nx.Graph(tuple(self.terms))

        coloring = list(
            nx.coloring.greedy_color(
                nx.line_graph(G), strategy, interchange=interchange
            ).items()
        )

        # sort into color groups
        coloring.sort(key=lambda coo_color: coo_color[1])

        return [
            # networkx doesn't preserve node order of edge spec
            tuple(sorted(coo)) for
            coo, _ in coloring
        ]

    def get_auto_ordering(self, order="sort", **kwargs):
        """Get an ordering of the terms to use with TEBD, for example. The
        default is to sort the coordinates then greedily group them into
        commuting sets.

        Parameters
        ----------
        order : {'sort', None, 'random', str}
            How to order the terms *before* greedily grouping them into
            commuting (non-coordinate overlapping) sets:

                - ``'sort'`` will sort the coordinate pairs first.
                - ``None`` will use the current order of terms which should
                  match the order they were supplied to this ``LocalHam2D``
                  instance.
                - ``'random'`` will randomly shuffle the coordinate pairs
                  before grouping them - *not* the same as returning a
                  completely random order.
                - ``'random-ungrouped'`` will randomly shuffle the coordinate
                  pairs but *not* group them at all with respect to
                  commutation.

            Any other option will be passed as a strategy to
            ``networkx.coloring.greedy_color`` to generate the ordering.

        Returns
        -------
        list[tuple[node]]
            Sequence of coordinate pairs.
        """
        if order is None:
            pairs = self.terms
        elif order == "sort":
            pairs = sorted(self.terms)
        elif order == "random":
            pairs = list(self.terms)
            random.shuffle(pairs)
        elif order == "random-ungrouped":
            pairs = list(self.terms)
            random.shuffle(pairs)
            return pairs
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
        s = "<LocalHamGen(nsites={}, num_terms={})>"
        return s.format(self.nsites, len(self.terms))

    def draw(
        self,
        ordering="sort",
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
            :meth:`quimb.tensor.tensor_gen_tebd.LocalHamGen.get_auto_ordering`
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
        import networkx as nx
        import matplotlib.pyplot as plt

        if figsize is None:
            L = self.nsites ** 0.5 + 1
            figsize = (L, L)

        ax_supplied = ax is not None
        if not ax_supplied:
            fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
            ax.axis("off")
            ax.set_aspect("equal")

        if ordering is None or isinstance(ordering, str):
            ordering = self.get_auto_ordering(ordering, **kwargs)

        G = nx.Graph()
        seen = set()
        n = 0
        edge_labels = dict()
        for where in ordering:
            site_a, site_b = where
            if (site_a in seen) or (site_b in seen):
                # start a new group
                seen = {site_a, site_b}
                n += 1
            else:
                seen.add(site_a)
                seen.add(site_b)

            nrm = do("linalg.norm", self.terms[where])
            edge_labels[where] = f"{nrm:.2f}"
            G.add_edge(site_a, site_b, norm=nrm, group=n)

        num_groups = n + 1
        colors = get_colors(range(num_groups))

        pos = get_positions(None, G)

        # do the plotting
        nx.draw_networkx_edges(
            G,
            pos=pos,
            width=tuple(2 * x[2]["norm"] ** 0.5 for x in G.edges(data=True)),
            edge_color=tuple(
                colors[x[2]["group"]] for x in G.edges(data=True)
            ),
            alpha=0.8,
            ax=ax,
        )
        nx.draw_networkx_edge_labels(
            G,
            pos=pos,
            edge_labels=edge_labels,
            font_size=fontsize,
            font_color=(0.5, 0.5, 0.5),
            bbox=dict(alpha=0),
            ax=ax,
        )
        nx.draw_networkx_labels(
            G,
            pos,
            font_color=(0.2, 0.2, 0.2),
            font_size=fontsize,
            font_weight="bold",
            ax=ax,
        )

        # create legend
        if legend:
            handles = []
            for color in colors.values():
                handles += [
                    plt.Line2D(
                        [0],
                        [0],
                        marker="o",
                        color=color,
                        linestyle="",
                        markersize=10,
                    )
                ]

            lbls = [f"Group {i + 1}" for i in range(num_groups)]

            ax.legend(
                handles,
                lbls,
                ncol=max(round(len(handles) / 20), 1),
                loc="center left",
                bbox_to_anchor=(1, 0.5),
            )

        if ax_supplied:
            return

        if return_fig:
            return fig

        plt.show()

    graph = draw


class TEBDGen:
    """Generic class for performing time evolving block decimation on an
    arbitrary graph, i.e. applying the exponential of a Hamiltonian using
    a product formula that involves applying local exponentiated gates only.
    """

    def __init__(
        self,
        psi0,
        ham,
        tau=0.01,
        D=None,
        imag=True,
        gate_opts=None,
        ordering=None,
        second_order_reflect=False,
        compute_energy_every=None,
        compute_energy_final=True,
        compute_energy_opts=None,
        compute_energy_fn=None,
        compute_energy_per_site=False,
        callback=None,
        keep_best=False,
        progbar=True,
    ):
        self.imag = imag
        if not imag:
            raise NotImplementedError("Real time evolution not tested yet.")

        self.state = psi0
        self.ham = ham
        self.progbar = progbar
        self.callback = callback

        # default time step to use
        self.tau = tau

        # parse gate application options
        if D is None:
            D = self._psi.max_bond()
        self.gate_opts = ensure_dict(gate_opts)
        self.gate_opts['max_bond'] = D
        self.gate_opts.setdefault('cutoff', 0.0)
        self.gate_opts.setdefault('contract', 'reduce-split')

        # parse energy computation options
        self.compute_energy_opts = ensure_dict(compute_energy_opts)

        self.compute_energy_every = compute_energy_every
        self.compute_energy_final = compute_energy_final
        self.compute_energy_fn = compute_energy_fn
        self.compute_energy_per_site = bool(compute_energy_per_site)

        if ordering is None:

            def dynamic_random():
                return self.ham.get_auto_ordering('random_sequential')

            self.ordering = dynamic_random
        elif isinstance(ordering, str):
            self.ordering = self.ham.get_auto_ordering(ordering)
        elif callable(ordering):
            self.ordering = ordering
        else:
            self.ordering = tuple(ordering)

        self.second_order_reflect = second_order_reflect

        # storage
        self._n = 0
        self.its = []
        self.taus = []
        self.energies = []

        self.keep_best = bool(keep_best)
        self.best = dict(energy=float('inf'), state=None, it=None)

    def sweep(self, tau):
        r"""Perform a full sweep of gates at every pair.

        .. math::

            \psi \rightarrow \prod_{\{ij\}} \exp(-\tau H_{ij}) \psi

        """
        if callable(self.ordering):
            ordering = self.ordering()
        else:
            ordering = self.ordering

        if self.second_order_reflect:
            ordering = tuple(ordering) + tuple(reversed(ordering))
            factor = 2.0
        else:
            factor = 1.0

        for where in ordering:

            if callable(tau):
                U = self.ham.get_gate_expm(where, -tau(where) / factor)
            else:
                U = self.ham.get_gate_expm(where, -tau / factor)

            self.gate(U, where)

    def _update_progbar(self, pbar):
        desc = f"n={self._n}, tau={self.tau}, energy~{float(self.energy):.6f}"
        pbar.set_description(desc)

    def evolve(self, steps, tau=None, progbar=None):
        """Evolve the state with the local Hamiltonian for ``steps`` steps with
        time step ``tau``.
        """
        if tau is not None:
            self.tau = tau

        if progbar is None:
            progbar = self.progbar

        pbar = Progbar(total=steps, disable=self.progbar is not True)

        try:
            for i in range(steps):
                # anything required by both energy and sweep
                self.presweep(i)

                # possibly compute the energy
                should_compute_energy = (
                    bool(self.compute_energy_every) and
                    (i % self.compute_energy_every == 0))
                if should_compute_energy:
                    self._check_energy()
                    self._update_progbar(pbar)

                # actually perform the gates
                self.sweep(self.tau)
                self._n += 1
                pbar.update()

                if self.callback is not None:
                    if self.callback(self):
                        break

            # possibly compute the energy
            if self.compute_energy_final:
                self._check_energy()
                self._update_progbar(pbar)

        except KeyboardInterrupt:
            # allow the user to interupt early
            pass
        finally:
            pbar.close()

    @property
    def state(self):
        """Return a copy of the current state.
        """
        return self.get_state()

    @state.setter
    def state(self, psi):
        self.set_state(psi)

    @property
    def n(self):
        """The number of sweeps performed.
        """
        return self._n

    @property
    def D(self):
        """The maximum bond dimension.
        """
        return self.gate_opts['max_bond']

    @D.setter
    def D(self, value):
        """The maximum bond dimension.
        """
        self.gate_opts['max_bond'] = round(value)

    def _check_energy(self):
        """Logic for maybe computing the energy if needed.
        """
        if self.its and (self._n == self.its[-1]):
            # only compute if haven't already
            return self.energies[-1]

        if self.compute_energy_fn is not None:
            en = self.compute_energy_fn(self)
        else:
            en = self.compute_energy()

        if self.compute_energy_per_site:
            en = en / self.ham.nsites

        self.energies.append(float(en))
        self.taus.append(float(self.tau))
        self.its.append(self._n)

        if self.keep_best and en < self.best['energy']:
            self.best['energy'] = en
            self.best['state'] = self.state
            self.best['it'] = self._n

        return self.energies[-1]

    @property
    def energy(self):
        """Return the energy of current state, computing it only if necessary.
        """
        return self._check_energy()

    # ------- abstract methods that subclasses might want to override ------- #

    def get_state(self):
        """The default method for retrieving the current state - simply a copy.
        Subclasses can override this to perform additional transformations.
        """
        return self._psi.copy()

    def set_state(self, psi):
        """The default method for setting the current state - simply a copy.
        Subclasses can override this to perform additional transformations.
        """
        self._psi = psi.copy()

    def presweep(self, i):
        """Perform any computations required before the sweep (and energy
        computation). For the basic TEBD this is nothing.
        """
        pass

    def gate(self, U, where):
        """Perform single gate ``U`` at coordinate pair ``where``. This is the
        the most common method to override.
        """
        self._psi.gate_(U, where, **self.gate_opts)

    def compute_energy(self):
        """Compute and return the energy of the current state. Subclasses can
        override this with a custom method to compute the energy.
        """
        return self._psi.compute_local_expectation_simple(
            terms=self.ham.terms,
            **self.compute_energy_opts
        )

    def __repr__(self):
        s = "<{}(n={}, tau={}, D={})>"
        return s.format(
            self.__class__.__name__, self.n, self.tau, self.D)


class SimpleUpdateGen(TEBDGen):

    def gate(self, U, where):
        self._psi.gate_simple_(
            U, where, gauges=self.gauges, **self.gate_opts
        )

    def compute_energy(self):
        return self._psi.compute_local_expectation_simple(
            terms=self.ham.terms,
            gauges=self.gauges,
            **self.compute_energy_opts,
        )

    def get_state(self, absorb_gauges=True):
        psi = self._psi.copy()

        if absorb_gauges:
            psi.gauge_simple_insert(self.gauges)
        else:
            for ix, g in self.gauges.items():
                psi |= Tensor(g, inds=[ix])

        return psi

    def set_state(self, psi):
        self._psi = psi.copy()
        self.gauges = {}
        self._psi.gauge_all_simple_(gauges=self.gauges)
