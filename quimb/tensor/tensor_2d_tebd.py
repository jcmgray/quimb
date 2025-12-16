"""Tools for performing TEBD like algorithms on a 2D lattice."""

from itertools import starmap

import numpy as np
import scipy.sparse.linalg as spla
from autoray import conj, dag, do, reshape

from ..utils_plot import default_to_neutral_style
from .contraction import contract_strategy
from .drawing import get_colors
from .optimize import TNOptimizer
from .tensor_2d import (
    PEPS,
    calc_plaquette_map,
    calc_plaquette_sizes,
    gen_2d_bonds,
    plaquette_to_sites,
)
from .tensor_arbgeom_tebd import LocalHamGen, SimpleUpdateGen, TEBDGen


class LocalHam2D(LocalHamGen):
    """A 2D Hamiltonian represented as local terms. This combines all two site
    and one site terms into a single interaction per lattice pair, and caches
    operations on the terms such as getting their exponential.

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

    def __init__(self, Lx, Ly, H2, H1=None, cyclic=False):
        self.Lx = int(Lx)
        self.Ly = int(Ly)

        # parse two site terms
        if hasattr(H2, "shape"):
            # use as default nearest neighbour term
            H2 = {None: H2}
        else:
            H2 = dict(H2)

        # possibly fill in default gates
        default_H2 = H2.pop(None, None)
        if default_H2 is not None:
            for coo_a, coo_b in gen_2d_bonds(
                Lx,
                Ly,
                steppers=[
                    lambda i, j: (i, j + 1),
                    lambda i, j: (i + 1, j),
                ],
                cyclic=cyclic,
            ):
                if (coo_a, coo_b) not in H2 and (coo_b, coo_a) not in H2:
                    H2[coo_a, coo_b] = default_H2

        super().__init__(H2=H2, H1=H1)

    @property
    def nsites(self):
        """The number of sites in the system."""
        return self.Lx * self.Ly

    def __repr__(self):
        s = "<LocalHam2D(Lx={}, Ly={}, num_terms={})>"
        return s.format(self.Lx, self.Ly, len(self.terms))

    @default_to_neutral_style
    def draw(
        self,
        ordering="sort",
        show_norm=True,
        figsize=None,
        fontsize=8,
        legend=True,
        ax=None,
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
        """
        import matplotlib.pyplot as plt

        if figsize is None:
            figsize = (self.Ly, self.Lx)

        ax_supplied = ax is not None
        if not ax_supplied:
            fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
            ax.axis("off")
            ax.set_aspect("equal")
        else:
            fig = None

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

            d = ((xs[1] - xs[0]) ** 2 + (ys[1] - ys[0]) ** 2) ** 0.5
            # offset by the length of bond to distinguish NNN etc.
            #     choose offset direction by parity of first site

            if d > 2**0.5:
                xs = [xi + (-1) ** int(ys[0]) * 0.02 * d for xi in xs]
                ys = [yi + (-1) ** int(xs[0]) * 0.02 * d for yi in ys]

            # set coordinates for label with some offset towards left
            if ij1[1] < ij2[1]:
                lbl_x0 = (3 * xs[0] + 2 * xs[1]) / 5
                lbl_y0 = (3 * ys[0] + 2 * ys[1]) / 5
            else:
                lbl_x0 = (2 * xs[0] + 3 * xs[1]) / 5
                lbl_y0 = (2 * ys[0] + 3 * ys[1]) / 5

            nrm = do("linalg.norm", self.terms[ij1, ij2])

            data.append((xs, ys, n, lbl_x0, lbl_y0, nrm))

        num_groups = n + 1
        colors = get_colors(range(num_groups))

        # do the plotting
        for xs, ys, n, lbl_x0, lbl_y0, nrm in data:
            ax.plot(xs, ys, c=colors[n], linewidth=2 * nrm**0.5)
            if show_norm:
                label = "{:.3f}".format(nrm)
                ax.text(lbl_x0, lbl_y0, label, c=colors[n], fontsize=fontsize)

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

        return fig, ax

    graph = draw


class ComputeEnergyBoundary:
    """Mixin class to add energy computation to TEBD2D classes."""

    def set_default_boundary_opts(self, chi=None):
        # parse energy computation options
        if chi is None:
            chi = max(8, self.D**2)
        self.compute_energy_opts["max_bond"] = chi
        self.compute_energy_opts.setdefault("cutoff", 0.0)
        self.compute_energy_opts.setdefault("normalized", True)

    def compute_energy(self):
        """Compute and return the energy of the current state."""
        return self.state.compute_local_expectation(
            self.ham.terms, **self.compute_energy_opts
        )

    @property
    def chi(self):
        """The bond dimensino to use for contracting the boundary for when
        computing the energy.
        """
        return self.compute_energy_opts["max_bond"]

    @chi.setter
    def chi(self, value):
        self.compute_energy_opts["max_bond"] = round(value)

    def _get_repr_info(self):
        info = super()._get_repr_info()
        info["chi"] = self.chi
        return info


class TEBD2D(TEBDGen, ComputeEnergyBoundary):
    """Generic class for performing two dimensional time evolving block
    decimation, i.e. applying the exponential of a Hamiltonian using
    a product formula that involves applying local exponentiated gates only.
    The only difference between this and `TEBDGen` is that the default energy
    computation uses boundary contraction specific to 2D tensor networks, with
    a default boundary bond dimension of ``max(8, D**2)``.

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
    ordering : str, tuple[tuple[int]], callable, optional
        How to order the terms, if a string is given then use this as the
        strategy given to
        :meth:`~quimb.tensor.tensor_2d_tebd.LocalHam2D.get_auto_ordering`. An
        explicit list of coordinate pairs can also be given. The default is to
        greedily form an 'edge coloring' based on the sorted list of
        Hamiltonian pair coordinates. If a callable is supplied it will be used
        to generate the ordering before each sweep.
    second_order_reflect : bool, optional
        If ``True``, then apply each layer of gates in ``ordering`` forward
        with half the time step, then the same with reverse order.
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
        ``TEBD2D`` object as its only argument. If it returns any value
        that boolean evaluates to ``True`` then terminal the evolution.
    progbar : boolean, optional
        Whether to show a live progress bar during the evolution.
    kwargs
        Extra options for the specific ``TEBD2D`` subclass.

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
    best : dict
        If ``keep_best`` was set then the best recorded energy and the
        corresponding state that was computed - keys ``'energy'`` and
        ``'state'`` respectively.
    """

    def __init__(
        self,
        psi0,
        ham,
        tau=0.01,
        D=None,
        chi=None,
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
        **kwargs,
    ):
        super().__init__(
            psi0=psi0,
            ham=ham,
            tau=tau,
            D=D,
            imag=imag,
            gate_opts=gate_opts,
            ordering=ordering,
            second_order_reflect=second_order_reflect,
            compute_energy_every=compute_energy_every,
            compute_energy_final=compute_energy_final,
            compute_energy_opts=compute_energy_opts,
            compute_energy_fn=compute_energy_fn,
            compute_energy_per_site=compute_energy_per_site,
            callback=callback,
            keep_best=keep_best,
            progbar=progbar,
            **kwargs,
        )
        self.set_default_boundary_opts(chi=chi)


def conditioner(tn, value=None, sweeps=2, balance_bonds=True):
    """ """
    if balance_bonds:
        for _ in range(sweeps - 1):
            tn.balance_bonds_()
            tn.equalize_norms_()
        tn.balance_bonds_()
    tn.equalize_norms_(value=value)


class SimpleUpdate(SimpleUpdateGen, ComputeEnergyBoundary):
    """A simple subclass of ``TEBD2D`` that overrides two key methods in
    order to keep 'diagonal gauges' living on the bonds of a PEPS. The gauges
    are stored separately from the main PEPS in the ``gauges`` attribute.
    Before and after a gate is applied they are absorbed and then extracted.
    When accessing the ``state`` attribute they are automatically inserted or
    you can call ``get_state(absorb_gauges=False)`` to lazily add them as
    hyperedge weights only. Reference: https://arxiv.org/abs/0806.3719.

    Parameters
    ----------
    psi0 : PEPS
        The initial state.
    ham : LocalHam2D
        The Hamtiltonian consisting of local terms.
    tau : float, optional
        The default time step to use.
    D : int, optional
        The maximum bond dimension, by default the current maximum bond of
        ``psi0``.
    chi : int, optional
        The bond dimension to use when computing the energy. By default
        ``max(8, D**2)``.
    cutoff : float, optional
        The singular value cutoff to use when applying gates.
    gate_opts : dict, optional
        Other options to supply to the gate application method,
        :meth:`quimb.tensor.tensor_arbgeom.TensorNetworkGenVector.gate_simple_`.
    ordering : None, str or callable, optional
        The ordering of the terms to apply, by default this will be determined
        automatically. It can be a string to be supplied to
        :meth:`quimb.tensor.tensor_arbgeom_tebd.LocalHamGen.get_auto_ordering`,
        a callable which returns an ordering when called, or a fixed sequence
        of coordinate pairs.
    second_order_reflect : bool, optional
        Whether to use a second order Trotter decomposition by reflecting the
        ordering.
    compute_energy_every : int, optional
        Compute the energy every this many steps.
    compute_energy_final : bool, optional
        Whether to compute the energy at the end.
    compute_energy_opts : dict, optional
        Other options (beyond ``max_bond`` which is set by ``chi``) supplied to
        :meth:`~quimb.tensor.tensor_2d.PEPS.compute_local_expectation`.
    compute_energy_fn : callable, optional
        A custom function to compute the energy, with signature
        ``fn(su: SimpleUpdate)``, where ``su`` is this instance.
    compute_energy_per_site : bool, optional
        Whether to compute the energy per site.
    tol : float, optional
        If not ``None``, stop when either energy difference falls below this
        value, or maximum singluar value changes fall below this value.
    tol_energy_diff : float, optional
        If not ``None``, stop when specifically the energy difference falls
        below this value.
    equilibrate_every : int, optional
        Equilibrate the gauges every this many steps.
    equilibrate_start : bool, optional
        Whether to equilibrate the gauges at the start, regardless of
        ``equilibrate_every``.
    equilibrate_opts : dict, optional
        Default options to supply to the gauge equilibration method, see
        :meth:`quimb.tensor.tensor_core.TensorNetwork.gauge_all_simple`. By
        default `max_iterations` is set to 100 and `tol` to 1e-3.
    callback : callable, optional
        A function to call after each step, with signature
        ``fn(su: SimpleUpdate)``.
    keep_best : bool, optional
        Whether to keep track of the best state and energy. If ``True``, the
        best state found during evolution will be stored in the ``best``
        attribute.
    plot_every : int, optional
        Whether to plot the energy and energy difference every this many steps.
    progbar : bool, optional
        Whether to show a progress bar during evolution.

    Attributes
    ----------
    state : PEPS
        The current state.
    D : int
        The maximum bond dimension.
    n : int
        The number of sweeps performed.
    energy : float
        The energy of the current state, computed only if necessary.
    energies : list[float]
        The history of computed energies.
    energy_diffs : list[float]
        The history of energy differences.
    energy_ns : list[int]
        The iteration numbers at which energies were computed.
    taus : list[float]
        The time steps used at each energy computation.
    best : dict
        If ``keep_best`` is ``True``, this dictionary will contain the best
        energy found during evolution under the key ``'energy'``, the state
        which achieved this energy under the key ``'state'``, and the iteration
        number under the key ``'it'``.
    equilibration_ns : list[int]
        The iteration numbers at which gauge equilibration was performed.
    equilibration_iterations : list[int]
        The number of iterations taken during each gauge equilibration.
    equilibration_max_sdiffs : list[float]
        The maximum singular value difference during each gauge equilibration.
    gauge_diffs : list[float]
        The history of maximum gauge differences after each sweep.
    """

    def __init__(
        self,
        psi0: PEPS,
        ham: LocalHam2D,
        tau=0.01,
        D=None,
        chi=None,
        cutoff=1e-10,
        imag=True,
        gate_opts=None,
        ordering=None,
        second_order_reflect=False,
        update="sequential",
        compute_energy_every=None,
        compute_energy_final=True,
        compute_energy_opts=None,
        compute_energy_fn=None,
        compute_energy_per_site=False,
        tol=None,
        equilibrate_every=None,
        equilibrate_start=True,
        equilibrate_opts=None,
        gauge_diff_period=None,
        callback=None,
        keep_best=False,
        progbar=True,
        **kwargs,
    ):
        super().__init__(
            psi0=psi0,
            ham=ham,
            tau=tau,
            D=D,
            cutoff=cutoff,
            imag=imag,
            gate_opts=gate_opts,
            ordering=ordering,
            second_order_reflect=second_order_reflect,
            update=update,
            compute_energy_every=compute_energy_every,
            compute_energy_final=compute_energy_final,
            compute_energy_opts=compute_energy_opts,
            compute_energy_fn=compute_energy_fn,
            compute_energy_per_site=compute_energy_per_site,
            tol=tol,
            equilibrate_every=equilibrate_every,
            equilibrate_start=equilibrate_start,
            equilibrate_opts=equilibrate_opts,
            gauge_diff_period=gauge_diff_period,
            callback=callback,
            keep_best=keep_best,
            progbar=progbar,
            **kwargs,
        )
        self.set_default_boundary_opts(chi=chi)


def gate_full_update_als(
    ket,
    env,
    bra,
    G,
    where,
    tags_plq,
    steps,
    tol,
    max_bond,
    optimize="auto-hq",
    solver="solve",
    dense=True,
    enforce_pos=False,
    pos_smudge=1e-6,
    init_simple_guess=True,
    condition_tensors=True,
    condition_maintain_norms=True,
    condition_balance_bonds=True,
):
    ket_plq = ket.select_any(tags_plq)
    bra_plq = bra.select_any(tags_plq)

    # this is the full target (copy - not virtual)
    target = ket_plq.gate(G, where, contract=False) | env

    if init_simple_guess:
        ket_plq.gate_(G, where, contract="reduce-split", max_bond=max_bond)
        for site in tags_plq:
            bra_plq[site].modify(data=conj(ket_plq[site].data))

    if condition_tensors:
        conditioner(ket_plq, balance_bonds=condition_balance_bonds)
        for site in tags_plq:
            bra_plq[site].modify(data=conj(ket_plq[site].data))
        if condition_maintain_norms:
            pre_norm = ket_plq[site].norm()

    overlap = bra_plq | target
    norm_plq = bra_plq | env | ket_plq

    xs = dict()
    x_previous = dict()
    previous_cost = None

    with contract_strategy(optimize):
        for i in range(steps):
            for site in tags_plq:
                lix = norm_plq[site, "BRA"].inds[:-1]
                rix = norm_plq[site, "KET"].inds[:-1]
                # remove site tensors and group their indices
                if dense:
                    N = norm_plq.select(site, which="!any").to_dense(lix, rix)

                    if enforce_pos:
                        el, ev = do("linalg.eigh", (N + dag(N)) / 2)
                        el = do("clip", el, pos_smudge, None)
                        N = ev @ do("diag", el) @ dag(ev)

                else:
                    N = norm_plq.select(site, which="!any").aslinearoperator(
                        lix, rix
                    )

                # target vector (remove lower site tensor and contract to vec)
                b = overlap.select((site, "BRA"), which="!all").to_dense(
                    overlap[site, "BRA"].inds[:-1],
                    overlap[site, "BRA"].inds[-1:],
                )

                if solver == "solve":
                    x = do("linalg.solve", N, b)
                elif solver == "lstsq":
                    x = do("linalg.lstsq", N, b, rcond=tol * 1e-3)[0]
                else:
                    # use scipy sparse linalg solvers
                    if solver in ("lsqr", "lsmr"):
                        solver_opts = dict(atol=tol, btol=tol)
                    else:
                        solver_opts = dict(tol=tol)

                    # use current site as initial guess (iterate over site ind)
                    x0 = x_previous.get(site, b)
                    x = np.stack(
                        [
                            getattr(spla, solver)(
                                N, b[..., k], x0=x0[..., k], **solver_opts
                            )[0]
                            for k in range(x0.shape[-1])
                        ],
                        axis=-1,
                    )

                # update the tensors (all 'virtual' TNs above also updated)
                Tk, Tb = ket[site], bra[site]
                Tk.modify(data=reshape(x, Tk.shape))
                Tb.modify(data=reshape(conj(x), Tb.shape))

                # store solution to check convergence
                xs[site] = x

            # after updating both sites check for convergence of tensor entries
            cost_fid = do("trace", do("real", dag(x) @ b))
            cost_norm = do("abs", do("trace", dag(x) @ (N @ x)))
            cost = -2 * cost_fid + cost_norm

            converged = (previous_cost is not None) and (
                abs(cost - previous_cost) < tol
            )
            if converged:
                break

            previous_cost = cost
            for site in tags_plq:
                x_previous[site] = xs[site]

    if condition_tensors:
        if condition_maintain_norms:
            conditioner(
                ket_plq, value=pre_norm, balance_bonds=condition_balance_bonds
            )
        else:
            conditioner(ket_plq, balance_bonds=condition_balance_bonds)
        for site in tags_plq:
            bra_plq[site].modify(data=conj(ket_plq[site].data))


def gate_full_update_autodiff_fidelity(
    ket,
    env,
    bra,
    G,
    where,
    tags_plq,
    steps,
    tol,
    max_bond,
    optimize="auto-hq",
    autodiff_backend="autograd",
    autodiff_optimizer="L-BFGS-B",
    init_simple_guess=True,
    condition_tensors=True,
    condition_maintain_norms=True,
    condition_balance_bonds=True,
    **kwargs,
):
    ket_plq = ket.select_any(tags_plq).view_like_(ket)
    bra_plq = bra.select_any(tags_plq).view_like_(bra)

    # the target sites + gate and also norm (copy - not virtual)
    target = ket_plq.gate(G, where, contract=False) | env

    # make initial guess the simple gate tensors
    if init_simple_guess:
        ket_plq.gate_(G, where, contract="reduce-split", max_bond=max_bond)
        for site in tags_plq:
            bra_plq[site].modify(data=conj(ket_plq[site].data))

    if condition_tensors:
        conditioner(ket_plq, balance_bonds=condition_balance_bonds)
        for site in tags_plq:
            bra_plq[site].modify(data=conj(ket_plq[site].data))
        if condition_maintain_norms:
            pre_norm = ket_plq[site].norm()

    def fidelity(bra_plq):
        for site in tags_plq:
            ket_plq[site].modify(data=conj(bra_plq[site].data))

        fid = (bra_plq | target).contract(all, optimize=optimize)
        norm = (bra_plq | env | ket_plq).contract(all, optimize=optimize)

        return -2 * do("abs", fid) + do("abs", norm)

    tnopt = TNOptimizer(
        bra_plq,
        loss_fn=fidelity,
        tags=tags_plq,
        progbar=False,
        optimizer=autodiff_optimizer,
        autodiff_backend=autodiff_backend,
        **kwargs,
    )
    bra_plq_opt = tnopt.optimize(steps, tol=tol)

    for site in tags_plq:
        new_data = bra_plq_opt[site].data
        ket[site].modify(data=conj(new_data))
        bra[site].modify(data=new_data)

    if condition_tensors:
        if condition_maintain_norms:
            conditioner(
                ket_plq, value=pre_norm, balance_bonds=condition_balance_bonds
            )
        else:
            conditioner(ket_plq, balance_bonds=condition_balance_bonds)
        for site in tags_plq:
            bra_plq[site].modify(data=conj(ket_plq[site].data))


def get_default_full_update_fit_opts():
    """The default options for the full update gate fitting procedure."""
    return {
        # general
        "tol": 1e-10,
        "steps": 20,
        "init_simple_guess": True,
        "condition_tensors": True,
        "condition_maintain_norms": True,
        # alternative least squares
        "als_dense": True,
        "als_solver": "solve",
        "als_enforce_pos": False,
        "als_enforce_pos_smudge": 1e-6,
        # automatic differentation optimizing
        "autodiff_backend": "autograd",
        "autodiff_optimizer": "L-BFGS-B",
    }


def parse_specific_gate_opts(strategy, fit_opts):
    """Parse the options from ``fit_opts`` which are relevant for ``strategy``."""
    gate_opts = {
        "tol": fit_opts["tol"],
        "steps": fit_opts["steps"],
        "init_simple_guess": fit_opts["init_simple_guess"],
        "condition_tensors": fit_opts["condition_tensors"],
        "condition_maintain_norms": fit_opts["condition_maintain_norms"],
    }

    if "als" in strategy:
        gate_opts["solver"] = fit_opts["als_solver"]
        gate_opts["dense"] = fit_opts["als_dense"]
        gate_opts["enforce_pos"] = fit_opts["als_enforce_pos"]
        gate_opts["pos_smudge"] = fit_opts["als_enforce_pos_smudge"]

    elif "autodiff" in strategy:
        gate_opts["autodiff_backend"] = fit_opts["autodiff_backend"]
        gate_opts["autodiff_optimizer"] = fit_opts["autodiff_optimizer"]

    return gate_opts


class FullUpdate(TEBD2D):
    """Implements the 'Full Update' version of 2D imaginary time evolution,
    where each application of a gate is fitted to the current tensors using a
    boundary contracted environment.

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
    ordering : str, tuple[tuple[int]], callable, optional
        How to order the terms, if a string is given then use this as the
        strategy given to
        :meth:`~quimb.tensor.tensor_2d_tebd.LocalHam2D.get_auto_ordering`. An
        explicit list of coordinate pairs can also be given. The default is to
        greedily form an 'edge coloring' based on the sorted list of
        Hamiltonian pair coordinates. If a callable is supplied it will be used
        to generate the ordering before each sweep.
    second_order_reflect : bool, optional
        If ``True``, then apply each layer of gates in ``ordering`` forward
        with half the time step, then the same with reverse order.
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
        ``TEBD2D`` object as its only argument. If it returns any value
        that boolean evaluates to ``True`` then terminal the evolution.
    progbar : boolean, optional
        Whether to show a live progress bar during the evolution.
    fit_strategy : {'als', 'autodiff-fidelity'}, optional
        Core method used to fit the gate application.

            * ``'als'``: alternating least squares
            * ``'autodiff-fidelity'``: local fidelity using autodiff

    fit_opts : dict, optional
        Advanced options for the gate application fitting functions. Defaults
        are inserted and can be accessed via the ``.fit_opts`` attribute.
    compute_envs_every : {'term', 'group', 'sweep', int}, optional
        How often to recompute the environments used to the fit the gate
        application:

            * ``'term'``: every gate
            * ``'group'``: every set of commuting gates (the default)
            * ``'sweep'``: every total sweep
            * int: every ``x`` number of total sweeps

    pre_normalize : bool, optional
        Actively renormalize the state using the computed environments.
    condition_tensors : bool, optional
        Whether to actively equalize tensor norms for numerical stability.
    condition_balance_bonds : bool, optional
        If and when equalizing tensor norms, whether to also balance bonds as
        an additional conditioning.
    contract_optimize : str, optional
        Contraction path optimizer to use for gate + env + sites contractions.

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
    best : dict
        If ``keep_best`` was set then the best recorded energy and the
        corresponding state that was computed - keys ``'energy'`` and
        ``'state'`` respectively.
    fit_opts : dict
        Detailed options for fitting the applied gate.
    """

    def __init__(
        self,
        psi0,
        ham,
        tau=0.01,
        D=None,
        chi=None,
        fit_strategy="als",
        fit_opts=None,
        compute_envs_every=1,
        pre_normalize=True,
        condition_tensors=True,
        condition_balance_bonds=True,
        contract_optimize="auto-hq",
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
        super().__init__(
            psi0=psi0,
            ham=ham,
            tau=tau,
            D=D,
            chi=chi,
            imag=imag,
            gate_opts=gate_opts,
            ordering=ordering,
            second_order_reflect=second_order_reflect,
            compute_energy_every=compute_energy_every,
            compute_energy_final=compute_energy_final,
            compute_energy_opts=compute_energy_opts,
            compute_energy_fn=compute_energy_fn,
            compute_energy_per_site=compute_energy_per_site,
            callback=callback,
            keep_best=keep_best,
            progbar=progbar,
        )

        self.fit_strategy = str(fit_strategy)
        self.fit_opts = get_default_full_update_fit_opts()
        if fit_opts is not None:
            bad_opts = set(fit_opts) - set(self.fit_opts)
            if bad_opts:
                raise ValueError("Invalid fit option(s): {}".format(bad_opts))
            self.fit_opts.update(fit_opts)

        self.pre_normalize = bool(pre_normalize)
        self.contract_optimize = str(contract_optimize)
        self.condition_tensors = bool(condition_tensors)
        self.condition_balance_bonds = bool(condition_balance_bonds)

        self.compute_envs_every = compute_envs_every
        self._env_n = self._env_term_count = self._env_group_count = -1

        self._psi.add_tag("KET")

    @property
    def fit_strategy(self):
        return self._fit_strategy

    @fit_strategy.setter
    def fit_strategy(self, fit_strategy):
        self._gate_fit_fn = {
            "als": gate_full_update_als,
            "autodiff-fidelity": gate_full_update_autodiff_fidelity,
        }[fit_strategy]
        self._fit_strategy = fit_strategy

    def set_state(self, psi):
        self._psi = psi.copy()

        # ensure the final dimension of each tensor is the physical dim
        for tag, ind in zip(self._psi.site_tags, self._psi.site_inds):
            t = self._psi[tag]
            if t.inds[-1] != ind:
                new_inds = [i for i in t.inds if i != ind] + [ind]
                t.transpose_(*new_inds)

    @property
    def compute_envs_every(self):
        return self._compute_envs_every

    @compute_envs_every.setter
    def compute_envs_every(self, x):
        if x == "sweep":
            self._need_to_recompute_envs = lambda: (self._n != self._env_n)
        elif x == "group":
            self._need_to_recompute_envs = lambda: (
                (self._n != self._env_n)
                or (self._group_count != self._env_group_count)
            )
        elif x == "term":
            self._need_to_recompute_envs = lambda: (
                (self._n != self._env_n)
                or (self._group_count != self._env_group_count)
                or (self._term_count != self._env_term_count)
            )
        else:
            x = max(1, int(x))
            self._need_to_recompute_envs = lambda: (self._n >= self._env_n + x)

        self._compute_envs_every = x

    def _maybe_compute_plaquette_envs(self, force=False):
        """Compute and store the plaquette environments for all local terms."""
        # first check if we need to compute the envs
        if not self._need_to_recompute_envs() and not force:
            return

        if self.condition_tensors:
            conditioner(self._psi, balance_bonds=self.condition_balance_bonds)

        # useful to store the bra that went into making the norm
        norm, _, self._bra = self._psi.make_norm(return_all=True)

        envs = dict()
        for x_bsz, y_bsz in calc_plaquette_sizes(self.ham.terms):
            envs.update(
                norm.compute_plaquette_environments(
                    x_bsz=x_bsz,
                    y_bsz=y_bsz,
                    max_bond=self.chi,
                    cutoff=0.0,
                    equalize_norms=True,
                )
            )

        if self.pre_normalize:
            # get the first plaquette env and use it to compute current norm
            p0, env0 = next(iter(envs.items()))
            sites = plaquette_to_sites(p0)
            tags_plq = tuple(starmap(norm.site_tag, sites))
            norm_plq = norm.select_any(tags_plq) | env0

            # contract the local plaquette norm
            nfactor = do(
                "abs", norm_plq.contract(all, optimize=self.contract_optimize)
            )

            # scale the bra and ket and each of the plaquette environments
            self._psi.multiply_(nfactor ** (-1 / 2), spread_over="all")
            self._bra.multiply_(nfactor ** (-1 / 2), spread_over="all")

            # scale the envs, taking into account the number of sites missing
            n = self._psi.num_tensors
            for ((_, _), (di, dj)), env in envs.items():
                n_missing = di * dj
                env.multiply_(
                    nfactor ** (n_missing / n - 1), spread_over="all"
                )

        self.plaquette_envs = envs
        self.plaquette_mapping = calc_plaquette_map(envs)

        self._env_n = self._n
        self._env_group_count = self._group_count
        self._env_term_count = self._term_count

    def presweep(self, i=None):
        """Full update presweep - compute envs and inject gate options."""
        # inject the specific gate options required (do
        # here so user can change options between sweeps)
        self._gate_opts = parse_specific_gate_opts(
            self.fit_strategy, self.fit_opts
        )

        # keep track of number of gates applied, and commutative groups
        self._term_count = 0
        self._group_count = 0
        self._current_group = set()

    def compute_energy(self):
        """Full update compute energy - use the (likely) already calculated
        plaquette environments.
        """
        self._maybe_compute_plaquette_envs(force=self._n != self._env_n)

        return self.state.compute_local_expectation(
            self.ham.terms,
            plaquette_envs=self.plaquette_envs,
            plaquette_mapping=self.plaquette_mapping,
            **self.compute_energy_opts,
        )

    def gate(self, G, where):
        """Apply the gate ``G`` at sites where, using a fitting method that
        takes into account the current environment.
        """
        # check if the new term commutes with those applied so far, this is to
        #     decide if we need to recompute the environments
        swhere = set(where)
        if self._current_group.isdisjoint(swhere):
            # if so add it to the grouping
            self._current_group |= swhere
        else:
            # else increment and reset the grouping
            self._current_group = swhere
            self._group_count += 1

        # get the plaquette containing ``where`` and the sites it contains -
        # these will all be fitted
        self._maybe_compute_plaquette_envs()
        plq = self.plaquette_mapping[tuple(sorted(where))]
        env = self.plaquette_envs[plq]
        tags_plq = tuple(starmap(self._psi.site_tag, plaquette_to_sites(plq)))

        # perform the gate, inplace
        self._gate_fit_fn(
            ket=self._psi,
            env=env,
            bra=self._bra,
            G=G,
            where=where,
            tags_plq=tags_plq,
            max_bond=self.D,
            optimize=self.contract_optimize,
            condition_balance_bonds=self.condition_balance_bonds,
            **self._gate_opts,
        )

        # self._psi.gauge_all_simple_()
        self._psi.equalize_norms_(1.0)
        self._psi.exponent = 0.0

        # increments every gate call regardless
        self._term_count += 1
