import array

from autoray import do
from quimb.tensor import TensorNetwork
from quimb.tensor.decomp import compute_bondenv_projectors
from quimb.tensor.tensor_arbgeom import TensorNetworkGenVector
from quimb.tensor.tensor_arbgeom_tebd import SimpleUpdateGen
from quimb.tensor.tensor_core import tensor_make_single_bond


def compress_between_tids_bondenv_exact(
    self: TensorNetwork,
    tida,
    tidb,
    max_bond,
    cutoff=0.0,
    absorb="both",
    max_iterations=100,
    tol=1e-10,
    solver="solve",
    solver_maxiter=4,
    prenormalize=False,
    condition=True,
    enforce_pos=True,
    pos_smudge=1e-10,
    init="svd",
    info=None,
    optimize="auto-hq",
    **contract_opts,
):
    """Compress the bond between the two tensors identified by ``tida`` and
    ``tidb`` exactly, by computing the full bond environment tensor and
    iteratively fitting compressed (low-rank) projectors to it.

    Parameters
    ----------
    tida : int
        The identifier of the first tensor.
    tidb : int
        The identifier of the second tensor.
    max_bond : int
        The maximum bond dimension to compress to.
    cutoff : float, optional
        The singular value cutoff to use.
    absorb : {'both', 'left', 'right', None}, optional
        How to absorb the effective singular values into the tensors.
    max_iterations : int, optional
        The maximum number of iterations to use when fitting the projectors.
    tol : float, optional
        The target tolerance to reach when fitting the projectors.
    solver : {'solve', None, str}, optional
        The solver to use inside the fitting loop. If None will use a custom
        conjugate gradient method. Else can be any of the iterative solvers
        supported by ``scipy.sparse.linalg`` such as 'gmres', 'bicgstab', etc.
    solver_maxiter : int, optional
        The maximum number of iterations to use for the *inner* solver, i.e.
        per fitting step, only for iterative `solver` args.
    prenormalize : bool, optional
        Whether to prenormalize the environment tensor such that its full
        contraction before compression is 1. Recommended for stability when
        the normalization does not matter.
    condition : bool or "iso", optional
        Whether to condition the projectors after each fitting step. If
        ``True``, their norms will be simply matched. If ``"iso"``, then they
        are gauged each time such that the previous tensor is isometric.
        Recommended for stability.
    enforce_pos : bool, optional
        Whether to enforce the environment tensor to be positive semi-definite
        by symmetrizing and clipping negative eigenvalues. Recommended for
        stability.
    pos_smudge : float, optional
        The value to clip negative eigenvalues to when enforcing positivity,
        relative to the largest eigenvalue.
    init : {'svd', 'eigh', 'random', 'reduced'}, optional
        How to initialize the compression projectors. The options are:

        - 'svd': use a truncated SVD of the environment tensor with the bra
          bond traced out.
        - 'eigh': use a similarity compression of the environment tensor with
          the bra bond traced out.
        - 'random': use random projectors.
        - 'reduced': split the environment into bra and ket parts, then
          canonize one half left and right to get the reduced factors.

    info : dict, optional
        If provided, will store information about the fitting process here.
        The keys 'iterations' and 'distance' will contain the final number of
        iterations and distance reached respectively.
    optimize : str, optional
        Contraction path optimizer to use when forming the bond environment.
    contract_opts
        Other contraction options to pass.
    """
    ta = self.tensor_map[tida]
    tb = self.tensor_map[tidb]

    # get and cut bond
    _, bix, _ = tensor_make_single_bond(ta, tb)
    ta.reindex_({bix: "kl"})
    tb.reindex_({bix: "kr"})

    # contract the 4-index bond tensor
    b = self.conj().reindex_({"kl": "bl", "kr": "br"})
    E = (
        (self | b)
        .contract(
            all,
            output_inds=("kl", "kr", "bl", "br"),
            optimize=optimize,
            **contract_opts,
        )
        .data
    )

    # compute the projectors
    Pl, svals, Pr = compute_bondenv_projectors(
        E,
        max_bond=max_bond,
        cutoff=cutoff,
        max_iterations=max_iterations,
        tol=tol,
        absorb=absorb,
        solver=solver,
        solver_maxiter=solver_maxiter,
        prenormalize=prenormalize,
        condition=condition,
        enforce_pos=enforce_pos,
        pos_smudge=pos_smudge,
        init=init,
        info=info,
    )

    # absorb the projectors to compress the tn!
    ta.gate_(Pl, "kl", transposed=True)
    tb.gate_(Pr, "kr")

    # reconnect bond
    ta.reindex_({"kl": bix})
    tb.reindex_({"kr": bix})

    if (svals is not None) and (info is not None):
        # store the singular values
        info["singular_values"] = svals / do("linalg.norm", svals)


def compress_between_tids_bondenv_cluster(
    self: TensorNetwork,
    tida,
    tidb,
    max_bond,
    cutoff=0.0,
    gauges=None,
    max_distance=1,
    mode="graphdistance",
    fillin=False,
    grow_from="all",
    max_iterations=100,
    tol=1e-10,
    solver="solve",
    solver_maxiter=4,
    prenormalize=False,
    condition=True,
    enforce_pos=True,
    pos_smudge=1e-10,
    init="svd",
    gauge_power=1.0,
    gauge_smudge=1e-10,
    optimize="auto-hq",
    info=None,
    **contract_opts,
):
    """Compress the bond between the two tensors identified by ``tida`` and
    ``tidb`` using a cluster of tensors around them to approximate the bond
    environment tensor.

    Parameters
    ----------
    tida : int
        The identifier of the first tensor.
    tidb : int
        The identifier of the second tensor.
    max_bond : int
        The maximum bond dimension to compress to.
    cutoff : float, optional
        The singular value cutoff to use, once the compressed projectors
        have been fitted.
    gauges : dict[any, array], optional
        Gauges, in terms of singular value vectors, to absorb around the
        cluster.
    max_distance : int, optional
        The maximum distance to the initial tagged region, or if using
        'loopunion' mode, the maximum size of any loop.
    mode : {'graphdistance', 'loopunion'}, optional
        How to select the local tensors, either by graph distance or by
        selecting the union of all loopy regions containing ``tids``.
    fillin : bool or int, optional
        Whether to fill in the local patch with additional tensors, or not.
        `fillin` tensors are those connected by two or more bonds to the
        original local patch, the process is repeated int(fillin) times.
    grow_from : {"all", "any", "alldangle", "anydangle"}, optional
        If mode is 'loopunion', whether each loop should contain *all* of
        the initial tids, or just *any* of them (generating a larger
        region). If 'alldangle' or 'anydangle', then the individual
        loops can contain the target tids even if they are dangling.
    max_iterations : int, optional
        The maximum number of iterations to use when fitting the projectors.
    tol : float, optional
        The target tolerance to reach when fitting the projectors.
    solver : {'solve', None, str}, optional
        The solver to use inside the fitting loop. If None will use a custom
        conjugate gradient method. Else can be any of the iterative solvers
        supported by ``scipy.sparse.linalg`` such as 'gmres', 'bicgstab', etc.
    solver_maxiter : int, optional
        The maximum number of iterations to use for the *inner* solver, i.e.
        per fitting step, only for iterative `solver` args.
    prenormalize : bool, optional
        Whether to prenormalize the environment tensor such that its full
        contraction before compression is 1. Recommended for stability when
        the normalization does not matter.
    condition : bool or "iso", optional
        Whether to condition the projectors after each fitting step. If
        ``True``, their norms will be simply matched. If ``"iso"``, then they
        are gauged each time such that the previous tensor is isometric.
        Recommended for stability.
    enforce_pos : bool, optional
        Whether to enforce the environment tensor to be positive semi-definite
        by symmetrizing and clipping negative eigenvalues. Recommended for
        stability.
    pos_smudge : float, optional
        The value to clip negative eigenvalues to when enforcing positivity,
        relative to the largest eigenvalue.
    init : {'svd', 'eigh', 'random', 'reduced'}, optional
        How to initialize the compression projectors. The options are:

        - 'svd': use a truncated SVD of the environment tensor with the bra
          bond traced out.
        - 'eigh': use a similarity compression of the environment tensor with
          the bra bond traced out.
        - 'random': use random projectors.
        - 'reduced': split the environment into bra and ket parts, then
          canonize one half left and right to get the reduced factors.

    gauge_power : float, optional
        A power to raise the gauge vectors to when inserting.
    gauge_smudge : float, optional
        A small value to add to the gauge vectors to avoid singularities
        when inserting.
    optimize : str, optional
        Contraction path optimizer to use when forming the bond environment.
    info : dict, optional
        If provided, will store information about the fitting process here.
        The keys 'iterations' and 'distance' will contain the final number of
        iterations and distance reached respectively.
    contract_opts
        Other contraction options to pass.
    """
    if info is None:
        info = {}

    if gauges is not None:
        # handle the target bond explicitly, as it is changing size
        ta = self.tensor_map[tida]
        tb = self.tensor_map[tidb]
        _, bix, _ = tensor_make_single_bond(ta, tb)
        g = gauges.pop(bix, None)
        if g is not None:
            gsqrt = do("sqrt", g)
            ta.multiply_index_diagonal_(bix, gsqrt)
            tb.multiply_index_diagonal_(bix, gsqrt)

        # we will extract new gauge directly
        absorb = None
    else:
        absorb = "both"

    k = self._select_local_tids(
        (tida, tidb),
        max_distance=max_distance,
        mode=mode,
        fillin=fillin,
        grow_from=grow_from,
    )

    if gauges is not None:
        outer, inner = k.gauge_simple_insert(
            gauges,
            power=gauge_power,
            smudge=gauge_smudge,
        )
    else:
        outer = inner = None

    compress_between_tids_bondenv_exact(
        k,
        tida,
        tidb,
        absorb=absorb,
        cutoff=cutoff,
        max_bond=max_bond,
        max_iterations=max_iterations,
        tol=tol,
        solver=solver,
        solver_maxiter=solver_maxiter,
        prenormalize=prenormalize,
        condition=condition,
        enforce_pos=enforce_pos,
        pos_smudge=pos_smudge,
        init=init,
        optimize=optimize,
        info=info,
        **contract_opts,
    )

    if gauges is not None:
        k.gauge_simple_remove(outer, inner)
        gauges[bix] = info["singular_values"]


def gate_cluster_(
    self: TensorNetworkGenVector,
    G,
    where,
    max_bond,
    cutoff=0.0,
    gauges=None,
    max_distance=1,
    mode="graphdistance",
    fillin=False,
    grow_from="all",
    max_iterations=100,
    tol=1e-10,
    solver="solve",
    solver_maxiter=4,
    prenormalize=False,
    condition=True,
    enforce_pos=True,
    pos_smudge=1e-10,
    init="svd",
    gauge_power=1.0,
    gauge_smudge=1e-10,
    equalize_norms=None,
    optimize="auto-hq",
    info=None,
    **contract_opts,
):
    """Apply a gate to this tensor network, using a single cluster around the
    target sites to then compress the increased bond dimension.

    Parameters
    ----------
    G : array
        The gate to apply.
    where : sequence of hashable
        The site(s) to apply the gate to.
    max_bond : int
        The maximum bond dimension to compress the bond to.
    cutoff : float, optional
        The singular value cutoff to use, once the compressed projectors
        have been fitted.
    gauges : dict[any, array], optional
        Gauges, in terms of singular value vectors, to absorb around the
        cluster.
    max_distance : int, optional
        The maximum distance to the initial tagged region, or if using
        'loopunion' mode, the maximum size of any loop.
    mode : {'graphdistance', 'loopunion'}, optional
        How to select the local tensors, either by graph distance or by
        selecting the union of all loopy regions containing ``tids``.
    fillin : bool or int, optional
        Whether to fill in the local patch with additional tensors, or not.
        `fillin` tensors are those connected by two or more bonds to the
        original local patch, the process is repeated int(fillin) times.
    grow_from : {"all", "any", "alldangle", "anydangle"}, optional
        If mode is 'loopunion', whether each loop should contain *all* of
        the initial tids, or just *any* of them (generating a larger
        region). If 'alldangle' or 'anydangle', then the individual
        loops can contain the target tids even if they are dangling.
    max_iterations : int, optional
        The maximum number of iterations to use when fitting the projectors.
    tol : float, optional
        The target tolerance to reach when fitting the projectors.
    solver : {'solve', None, str}, optional
        The solver to use inside the fitting loop. If None will use a custom
        conjugate gradient method. Else can be any of the iterative solvers
        supported by ``scipy.sparse.linalg`` such as 'gmres', 'bicgstab', etc.
    solver_maxiter : int, optional
        The maximum number of iterations to use for the *inner* solver, i.e.
        per fitting step, only for iterative `solver` args.
    prenormalize : bool, optional
        Whether to prenormalize the environment tensor such that its full
        contraction before compression is 1. Recommended for stability when
        the normalization does not matter.
    condition : bool or "iso", optional
        Whether to condition the projectors after each fitting step. If
        ``True``, their norms will be simply matched. If ``"iso"``, then they
        are gauged each time such that the previous tensor is isometric.
        Recommended for stability.
    enforce_pos : bool, optional
        Whether to enforce the environment tensor to be positive semi-definite
        by symmetrizing and clipping negative eigenvalues. Recommended for
        stability.
    pos_smudge : float, optional
        The value to clip negative eigenvalues to when enforcing positivity,
        relative to the largest eigenvalue.
    init : {'svd', 'eigh', 'random', 'reduced'}, optional
        How to initialize the compression projectors. The options are:

        - 'svd': use a truncated SVD of the environment tensor with the bra
          bond traced out.
        - 'eigh': use a similarity compression of the environment tensor with
          the bra bond traced out.
        - 'random': use random projectors.
        - 'reduced': split the environment into bra and ket parts, then
          canonize one half left and right to get the reduced factors.

    gauge_power : float, optional
        A power to raise the gauge vectors to when inserting.
    gauge_smudge : float, optional
        A small value to add to the gauge vectors to avoid singularities
        when inserting.
    equalize_norms : bool, optional
        Whether to equalize the norms of the two tensors after compression,
        stripping the exponent.
    optimize : str, optional
        Contraction path optimizer to use when forming the bond environment.
    info : dict, optional
        If provided, will store information about the fitting process here.
        The keys 'iterations' and 'distance' will contain the final number of
        iterations and distance reached respectively.
    contract_opts
        Other contraction options to pass.
    """
    if len(where) == 1:
        # simply contract single site gates in
        return self.gate_(G, where, contract=True)

    taga, tagb = map(self.site_tag, where)
    (tida,) = self._get_tids_from_tags(taga)
    (tidb,) = self._get_tids_from_tags(tagb)

    # apply gate without truncation
    if gauges is not None:
        self.gate_simple_(
            G,
            where,
            gauges=gauges,
            max_bond=None,
            cutoff=0.0,
            power=gauge_power,
            smudge=gauge_smudge,
        )
    else:
        self.gate_(
            G,
            where,
            contract="reduce-split",
            max_bond=None,
            cutoff=0.0,
        )

    compress_between_tids_bondenv_cluster(
        self,
        tida,
        tidb,
        max_bond,
        cutoff=cutoff,
        max_distance=max_distance,
        mode=mode,
        fillin=fillin,
        grow_from=grow_from,
        gauges=gauges,
        max_iterations=max_iterations,
        tol=tol,
        gauge_power=gauge_power,
        gauge_smudge=gauge_smudge,
        solver=solver,
        solver_maxiter=solver_maxiter,
        prenormalize=prenormalize,
        condition=condition,
        enforce_pos=enforce_pos,
        pos_smudge=pos_smudge,
        init=init,
        optimize=optimize,
        info=info,
        **contract_opts,
    )

    if equalize_norms:
        self.strip_exponent(tida, equalize_norms)
        self.strip_exponent(tidb, equalize_norms)


class ClusterUpdateGen(SimpleUpdateGen):
    def __init__(
        self,
        psi0,
        ham,
        max_distance=1,
        mode="graphdistance",
        fillin=0,
        grow_from="all",
        gauge_power=1.0,
        **kwargs,
    ):
        super().__init__(psi0, ham, **kwargs)

        self.gate_opts.setdefault("max_distance", max_distance)
        self.gate_opts.setdefault("mode", mode)
        self.gate_opts.setdefault("fillin", fillin)
        self.gate_opts.setdefault("grow_from", grow_from)
        self.gate_opts.setdefault("gauge_power", gauge_power)
        # we don't use the `contract` option
        self.gate_opts.pop("contract", None)

        # match gauging power in gate and equilibration by default
        self.equilibrate_opts.setdefault("power", gauge_power)

        # track compression performance
        self.max_fit_iterations = array.array("L")
        self.max_fit_distances = array.array("d")

    def presweep(self, *args, **kwargs):
        self.max_fit_iterations.append(0)
        self.max_fit_distances.append(0.0)
        return super().presweep(*args, **kwargs)

    def gate(self, U, where):
        info = {}
        gate_cluster_(
            self._psi,
            U,
            where,
            gauges=self._gauges,
            info=info,
            **self.gate_opts,
        )
        self.max_fit_iterations[-1] = max(
            info["iterations"], self.max_fit_iterations[-1]
        )
        self.max_fit_distances[-1] = max(
            info["distance"], self.max_fit_distances[-1]
        )

    def assemble_plot_data(self):
        data = super().assemble_plot_data()
        data["max_fit_iterations"] = self.max_fit_iterations
        data["max_fit_distances"] = {
            "y": self.max_fit_distances,
            "yscale": "log",
        }
        return data


def compress_between_tids_bondenv_gloop_expand(
    self: TensorNetwork,
    tida,
    tidb,
    max_bond,
    cutoff=0.0,
    gauges=None,
    gloops=None,
    autocomplete=True,
    # autoreduce=True,  # not implemented yet
    grow_from="all",
    normalized=False,
    combine="sum",
    max_iterations=100,
    tol=1e-6,
    gauge_power=1.0,
    gauge_smudge=1e-6,
    prenormalize=False,
    condition=True,
    enforce_pos=True,
    pos_smudge=1e-6,
    solver="solve",
    solver_maxiter=4,
    init="svd",
    equalize_norms=None,
    optimize="auto-hq",
    info=None,
    **contract_opts,
):
    """Compress the bond between the two tensors identified by ``tida`` and
    ``tidb`` using a generalized loop expansion approximation of the bond
    environment tensor.
    """
    from quimb.tensor.belief_propagation import RegionGraph

    ta = self.tensor_map[tida]
    tb = self.tensor_map[tidb]
    _, bix, _ = tensor_make_single_bond(ta, tb)

    if gauges is not None:
        g = gauges.pop(bix, None)
        if g is not None:
            gsqrt = do("sqrt", g)
            ta.multiply_index_diagonal_(bix, gsqrt)
            tb.multiply_index_diagonal_(bix, gsqrt)
        absorb = None
    else:
        absorb = "both"

    gloops = self.get_local_gloops(
        tids=(tida, tidb),
        gloops=gloops,
        grow_from=grow_from,
    )

    # create a region graph, for intersections and counting numbers
    rg = RegionGraph(gloops, autocomplete=autocomplete)
    E = None
    for r in rg.regions:
        # and and possibly gauge region
        k = self._select_tids(r, virtual=False)
        if gauges is not None:
            k.gauge_simple_insert(gauges, power=gauge_power, smudge=gauge_smudge)

        # contract the 4-index bond tensor given by this region
        k._cut_between_tids(tida, tidb, "kl", "kr")
        b = k.conj().reindex_({"kl": "bl", "kr": "br"})
        Er = (
            (k | b)
            .contract(
                all,
                output_inds=("kl", "kr", "bl", "br"),
                optimize=optimize,
                **contract_opts,
            )
            .data
        )

        if normalized:
            Er /= do("einsum", "aabb->", Er)

        if combine == "sum":
            # sum-combine environments, weighting by counting numbers
            # XXX: normalize by trace of Er ?
            cr = rg.get_count(r)
            if cr != 1:
                Er = cr * Er
            E = Er if E is None else E + Er

        elif combine == "prod":
            # product-combine environments, weighting by counting numbers
            cr = rg.get_count(r)
            if cr != 1:
                if cr < 0:
                    Eabs = do("abs", Er)
                    Emax = do("max", Eabs)
                    Etol = Emax * 1e-6
                    # replace zeros
                    Er = Er + Etol * (Eabs < Etol)
                Er = Er**cr
            if E is None:
                E = Er
            else:
                E = E * Er

        else:
            raise ValueError(f"Unrecognized combine method: {combine}")

    # compute the projectors for combined environment
    Pl, maybe_svals, Pr = compute_bondenv_projectors(
        E,
        max_bond=max_bond,
        cutoff=cutoff,
        max_iterations=max_iterations,
        tol=tol,
        absorb=absorb,
        solver=solver,
        solver_maxiter=solver_maxiter,
        prenormalize=prenormalize,
        condition=condition,
        enforce_pos=enforce_pos,
        pos_smudge=pos_smudge,
        init=init,
        info=info,
    )

    # absorb the projectors to compress the tn!
    ta.gate_(Pl, bix, transposed=True)
    tb.gate_(Pr, bix)

    if maybe_svals is not None:
        # update with new, truncated singular values
        gauges[bix] = maybe_svals

    if equalize_norms:
        self.strip_exponent(tida, equalize_norms)
        self.strip_exponent(tidb, equalize_norms)


def gate_gloop_expand_(
    self: TensorNetworkGenVector,
    G,
    where,
    max_bond,
    cutoff=0.0,
    gauges=None,
    gloops=None,
    autocomplete=True,
    # autoreduce=True,
    grow_from="all",
    normalized=False,
    combine="sum",
    max_iterations=100,
    tol=1e-6,
    gauge_power=1.0,
    gauge_smudge=1e-6,
    prenormalize=False,
    condition=True,
    enforce_pos=True,
    pos_smudge=1e-6,
    solver="solve",
    solver_maxiter=4,
    equalize_norms=None,
    optimize="auto-hq",
    info=None,
    **contract_opts,
):
    """Apply a gate to a pair of tensors, then compress the bond between them
    using a generalized loop expansion approximation of the bond environment
    tensor.

    Parameters
    ----------
    G : array
        The gate to apply.
    where : sequence of hashable
        The site(s) to apply the gate to.
    max_bond : int
        The maximum bond dimension to compress the bond to.
    cutoff : float, optional
        The singular value cutoff to use, once the compressed projectors
        have been fitted.
    gauges : dict[any, array], optional
        If provided, the gauge tensors to use for each bond.
    gloops : int, optional
        The maximum number of tensors to include in a single cluster.
    max_iterations : int, optional
        The maximum number of iterations to use when fitting the
        projectors.
    tol : float, optional
        The tolerance to use when fitting the projectors.
    """
    if len(where) == 1:
        # simply contract single site gates in
        return self.gate_(G, where, contract=True)

    taga, tagb = map(self.site_tag, where)
    (tida,) = self._get_tids_from_tags(taga)
    (tidb,) = self._get_tids_from_tags(tagb)

    # apply gate without truncation
    if gauges is not None:
        self.gate_simple_(
            G,
            where,
            gauges=gauges,
            max_bond=None,
            cutoff=0.0,
            power=gauge_power,
            smudge=gauge_smudge,
        )
    else:
        self.gate_(
            G,
            where,
            contract="reduce-split",
            max_bond=None,
            cutoff=0.0,
        )

    compress_between_tids_bondenv_gloop_expand(
        self,
        tida,
        tidb,
        max_bond,
        cutoff=cutoff,
        gloops=gloops,
        autocomplete=autocomplete,
        # autoreduce=autoreduce,
        grow_from=grow_from,
        normalized=normalized,
        combine=combine,
        gauges=gauges,
        max_iterations=max_iterations,
        tol=tol,
        gauge_power=gauge_power,
        gauge_smudge=gauge_smudge,
        prenormalize=prenormalize,
        condition=condition,
        enforce_pos=enforce_pos,
        pos_smudge=pos_smudge,
        solver=solver,
        solver_maxiter=solver_maxiter,
        equalize_norms=equalize_norms,
        optimize=optimize,
        info=info,
        **contract_opts,
    )


class GLoopExpandUpdateGen(SimpleUpdateGen):
    def __init__(
        self,
        psi0,
        ham,
        gloops=None,
        grow_from="all",
        normalized=False,
        combine="sum",
        enforce_pos=True,
        pos_smudge=1e-6,
        gauge_power=1.0,
        **kwargs,
    ):
        super().__init__(psi0, ham, **kwargs)

        self.gate_opts.setdefault("gloops", gloops)
        self.gate_opts.setdefault("grow_from", grow_from)
        self.gate_opts.setdefault("normalized", normalized)
        self.gate_opts.setdefault("combine", combine)
        self.gate_opts.setdefault("enforce_pos", enforce_pos)
        self.gate_opts.setdefault("pos_smudge", pos_smudge)
        self.gate_opts.setdefault("gauge_power", gauge_power)
        # we don't use the `contract` option
        self.gate_opts.pop("contract", None)

        # match gauging power in gate and equilibration by default
        self.equilibrate_opts.setdefault("power", gauge_power)

        # track compression performance
        self.max_fit_iterations = array.array("L")
        self.max_fit_distances = array.array("d")

    def presweep(self, *args, **kwargs):
        self.max_fit_iterations.append(0)
        self.max_fit_distances.append(0.0)
        return super().presweep(*args, **kwargs)

    def gate(self, U, where):
        info = {}
        gate_gloop_expand_(
            self._psi,
            U,
            where,
            gauges=self._gauges,
            info=info,
            **self.gate_opts,
        )
        self.max_fit_iterations[-1] = max(
            info["iterations"], self.max_fit_iterations[-1]
        )
        self.max_fit_distances[-1] = max(
            info["distance"], self.max_fit_distances[-1]
        )

    def assemble_plot_data(self):
        data = super().assemble_plot_data()
        data["max_fit_iterations"] = self.max_fit_iterations
        data["max_fit_distances"] = {
            "y": self.max_fit_distances,
            "yscale": "log",
        }
        return data