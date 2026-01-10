# Changelog

Release notes for `quimb`.

(whats-new-1-12-0)=
## v1.12.0 (unreleased)

**Enhancements:**

- move the experimental `operatorbuilder` module to the main [`quimb.operator`](quimb.operator) module.
- add basic introduction to the operator module - {ref}`operator-basics`
- add new example on tracing tensor network functions {ref}`ex_tracing_tn_functions`
- [`tensor_split`](quimb.tensor.tensor_core.tensor_split): add an `info` kwarg, supplying this with an empty dict or with the entry `'error'` will store the truncation error when using `method in {"svd", "eig"}`.
- update infrastructure for TEBD and SimpleUpdate based algorithms.
- [`schematic.Drawing`](quimb.schematic.Drawing): add [`grid`](quimb.schematic.Drawing.grid), [`grid3d`](quimb.schematic.Drawing.grid3d), [`bezier`](quimb.schematic.Drawing.bezier), [`star`](quimb.schematic.Drawing.star), [`cross`](quimb.schematic.Drawing.cross) and [`zigzag`](quimb.schematic.Drawing.zigzag) methods.
- [`schematic.Drawing`](quimb.schematic.Drawing): add `relative` option to [`arrowhead`](quimb.schematic.Drawing.arrowhead), `shorten` option to [`text_between`](quimb.schematic.Drawing.text_between) and `text_left` and `text_right` options to [`line`](quimb.schematic.Drawing.line).
- add [`Drawing.scale_figsize`](quimb.schematic.Drawing.scale_figsize) for automatically setting the absolute figsize based on placed elements.
- refactor [`TEBDGen`](quimb.tensor.tensor_arbgeom_tebd.TEBDGen) and [`SimpleUpdateGen`](quimb.tensor.tensor_arbgeom_tebd.SimpleUpdateGen)
- update the 2d specific [`SimpleUpdate`](quimb.tensor.tensor_2d_tebd.SimpleUpdate) to use the new infrastructure.
- [`tn.draw()`](quimb.tensor.drawing.draw_tn): show abelian signature if using `symmray` arrays.
- [`tn.draw()`](quimb.tensor.drawing.draw_tn): add `adjust_lims` option
- [`TNOptimizer`](quimb.tensor.optimize.TNOptimizer): allow `autodiff_backend="torch"` with `jit_fn=True` to work with array backends with general pytree parameters, e.g. `symmray` arrays.
- [`tn.gen_gloops`](quimb.tensor.tensor_core.TensorNetwork.gen_gloops) and [`tn.gen_gloops_sites`](quimb.tensor.tensor_arbgeom.TensorNetworkArbgeom.gen_gloops_sites): add `join_overlap` option. When building cluster by joining smaller generalized loops, this option controls how many nodes they need to overlap by to be joined together.
- all message passing routines: add `callback` option
- GBP: allow a message initilization function.
- [`D1BP`](quimb.tensor.belief_propagation.d1bp.D1BP): allow `messages` to be a callable initialization function.
- [`MatrixProductState.gate_nonlocal`](quimb.tensor.tensor_1d.MatrixProductState.gate_nonlocal): add `method="lazy"` option for lazily applying a non-local gate as a sub-MPO without contraction or compression.
- [`LocalHamGen.apply_to_arrays`](quimb.tensor.tensor_arbgeom_tebd.LocalHamGen.apply_to_arrays): support pytree parameter arrays such as `symmray`.
- add [`Tensor.get_namespace`](quimb.tensor.tensor_core.Tensor.get_namespace) and [`TensorNetwork.get_namespace`](quimb.tensor.tensor_core.TensorNetwork.get_namespace) for getting a [reusable data array namespace](https://autoray.readthedocs.io/en/latest/automatic_dispatch.html#namespace-api)
- [`TensorNetwork.isel`](quimb.tensor.tensor_core.TensorNetwork.isel): use `take` where possible to better support e.g. `torch.vmap` across amplitudes.
- [`MatrixProductState.measure`](quimb.tensor.tensor_1d.MatrixProductState.measure), and [`MatrixProductState.sample`](quimb.tensor.tensor_1d.MatrixProductState.sample): add `backend_random` option for specifying which backend to use for random number generation when sampling, this can be set for example to `jax` to make the whole process jittable, but by default is `numpy`, regardless of the actual array backend.

**Bug fixes:**

- fix [`insert_compressor_between_regions`](quimb.tensor.tensor_core.TensorNetwork.insert_compressor_between_regions) when `insert_into is None`.
- tensor network drawing, ensure hyper indices can be specified as `output_inds`.
- fix [`MatrixProductState.measure`](quimb.tensor.tensor_1d.MatrixProductState.measure) when using jax arrays ({issue}`340`).
- fix [`MatrixProductState.measure`](quimb.tensor.tensor_1d.MatrixProductState.measure) when projecting and keeping a site site ({issue}`344`).

(whats-new-1-11-2)=
## v1.11.2 (2025-07-30)

**Enhancements:**

- Update the introduction to tensor contraction docs
- Improve efficiency of 1D structured contractions when default `optimize` is used, especially for large bond dimension overlaps.

**Bug fixes:**

- fixes for MPS and MPO constructors when L=1, ({issue}`314`)
- tensor splitting with absorb="left" now correctly marks left indices.
- [`tn.isel`](quimb.tensor.tensor_core.TensorNetwork.isel): fix bug when value could not be compared to string `"r"`
- truncated svd, make n_chi comparison more robust to different backends


(whats-new-1-11-1)=
## v1.11.1 (2025-06-20)

**Enhancements:**

- add `create_bond` to [`tensor_canonize_bond`](quimb.tensor.tensor_core.tensor_canonize_bond) and [`tensor_compress_bond`](quimb.tensor.tensor_core.tensor_compress_bond) for optionally creating a new bond between two tensors if they don't already share one. Add as a flag to [`TensorNetwork1DFlat.compress`](quimb.tensor.tensor_1d.TensorNetwork1DFlat.compress) and related functions ({issue}`294`).
- add [`ensure_bonds_exist`](quimb.tensor.tensor_1d.TensorNetwork1DFlat.ensure_bonds_exist) for ensuring that all bonds in a 1D flat tensor network exist. Use this in the `permute_arrays` methods and optionally in the `expand_bond_dimension` method.
- [`tn.draw()`](quimb.tensor.drawing.draw_tn): permit empty network, and allow `color=True` to automatically color all tags.
- [`tn.add_tag`](quimb.tensor.tensor_core.TensorNetwork.add_tag): add a `record: Optional[dict]` kwarg, to allow for easy rewinding of temporary tags without tracking the actual networks.
- add [`qu.plot`](quimb.utils_plot.plot) as a quick wrapper for calling `matplotlib.pyplot.plot` with the `quimb` style.
- [`quimb.schematic`](quimb.schematic): add `zorder_delta` kwarg for fine adjustments to layering of objects in approximately the same position.
- [`operatorbuilder`](quimb.operator): big performance improvements and fixes for building matrix representations including Z2 symmetry. Add default `symmetry` and `sector` options that can be overridden at build time. Add lazy (slow, matrix free) 'apply' method. Add `pauli_decompose` transformation. Add experimental PEPO builder for nearest neighbor operators. Add unit tests.

**Bug fixes:**

- Fix [`TensorNetwork2D.compute_plaquette_environments`](quimb.tensor.tensor_2d.TensorNetwork2D.compute_plaquette_environments) for `mode="zipup"` and other boundary contraction methods that use the generic 1D compression algorithms.
- [`parse_openqasm2_str`](quimb.tensor.circuit.parse_openqasm2_str) allow custom gate names to start with the word `gate` ({issue}`312`).
- [`MatrixProductState.gate_with_mpo`](quimb.tensor.tensor_1d.MatrixProductState.gate_with_mpo): fix bug to do with inplace argument ({issue}`313`).


(whats-new-1-11-0)=
## v1.11.0 (2025-05-14)

**Breaking Changes**

- move belief propagation to [`quimb.tensor.belief_propagation`](quimb.tensor.belief_propagation)
- calling [`tn.contract()`](quimb.tensor.tensor_core.TensorNetwork.contract) when an non-zero value has been accrued into `tn.exponent` now automatically re-absorbs that exponent.
- binary tensor operations that would previously have errored now will align and broadcast

**Enhancements:**

- [`Tensor`](quimb.tensor.tensor_core.Tensor): make binary operations (`+, -, *, /, **`) automatically align and broadcast indices. This would previously error.
- [`MatrixProductState.measure`](quimb.tensor.tensor_1d.MatrixProductState.measure): add a `seed` kwarg
- belief propagation, implement DIIS (direct inversion in the iterative subspace)
- belief propagation, unify various aspects such as message normalization and distance.
- belief propagation, add a [`plot`](quimb.tensor.belief_propagation.bp_common.BeliefPropagationCommon.plot) method.
- belief propagation, add a `contract_every` option.
- HV1BP: vectorize both contraction and message initialization
- add [`qu.plot_multi_series_zoom`](quimb.utils_plot.plot_multi_series_zoom) for plotting multiple series with a zoomed inset, useful for various convergence plots such as BP
- add `info` option to [`tn.gauge_all_simple`](quimb.tensor.tensor_core.TensorNetwork.gauge_all_simple) for tracking extra information such as number of iterations and max gauge diffs
- [`Tensor.gate`](quimb.tensor.tensor_core.Tensor.gate): add `transposed` option
- [`TensorNetwork.contract`](quimb.tensor.tensor_core.TensorNetwork.contract): add `strip_exponent` option for return the mantissa and exponent (log10) separately. Compatible with [`contract_tags`](quimb.tensor.tensor_core.TensorNetwork.contract_tags), [`contract_cumulative`](quimb.tensor.tensor_core.TensorNetwork.contract_cumulative), [`contract_compressed`](quimb.tensor.tensor_core.TensorNetwork.contract_compressed) sub modes.
- [`tensor_split`](quimb.tensor.tensor_core.tensor_split): add `matrix_svals` option, if `True` any returned singular values are put into the diagonal of a matrix (by default, `False`, they are returned as a vector).
- add [`Tensor.new_ind_pair_diag`](quimb.tensor.tensor_core.Tensor.new_ind_pair_diag) for expanding an existing index into a pair of new indices, such that the diagonal of the new tensor on those indices is the old tensor.
- [`TNOptimizer`](quimb.tensor.optimize.TNOptimizer): add 'cautious' ADAM
- [`TensorNetwork.pop_tensor`](quimb.tensor.tensor_core.TensorNetwork.pop_tensor): allow `tid` or tags to be specified.
- add an example notebook for converting hyper tensor networks to normal tensor networks, for approximate contraction - {ref}`example-htn-to-2d`
- add "SX" and "SXDG" gates to [`Circuit`](quimb.tensor.circuit.Circuit) ({pull}`#277`)
- add "XXPLUSYY" and "XXPLUSYY" gates to [`Circuit`](quimb.tensor.circuit.Circuit) ({pull}`#279`)
- add progress bar to various `Circuit` methods ({pull}`#288`)
- [`quimb.operator`](quimb.operator): fix MPO building for congested operators ({issue}`296` and {issue}`301`), allow arbitrary dtype ({issue}`289`). Fix building of sparse and matrix representations for non-translationally symmetric operators and operators with trivial (all identity) terms.

**Bug fixes:**

- fix [`MatrixProductState.measure`](quimb.tensor.tensor_1d.MatrixProductState.measure) for `cupy` backend arrays ({issue}`276`).
- fix `linalg.expm` dispatch ({issue}`275`)
- fix 'dm' 1d compress method for disconnected subgraphs
- fix docs source lookup in `quimb.tensor` module
- fix raw gate copying in `Circuit` ({issue}`285`)


(whats-new-1-10-0)=
## v1.10.0 (2024-12-18)

**Enhancements:**

- tensor network fitting: add `method="tree"` for when ansatz is a tree - [`tensor_network_fit_tree`](quimb.tensor.fitting.tensor_network_fit_tree)
- tensor network fitting: fix `method="als"` for complex networks
- tensor network fitting: allow `method="als"` to use a iterative solver suited to much larger tensors, by default a custom conjugate gradient implementation.
- [`tensor_network_distance`](quimb.tensor.fitting.tensor_network_distance) and fitting: support hyper indices explicitly via `output_inds` kwarg
- add [`tn.make_overlap`](quimb.tensor.tensor_core.TensorNetwork.make_overlap) and [`tn.overlap`](quimb.tensor.tensor_core.TensorNetwork.overlap) for computing the overlap between two tensor networks, $\langle O |T \rangle$, with explicit handling of outer indices to address hyper networks. Add `output_inds` to [`tn.norm`](quimb.tensor.tensor_core.TensorNetwork.norm) and [`tn.make_norm`](quimb.tensor.tensor_core.TensorNetwork.make_norm) also, as well as the `squared` kwarg.
- replace all `numba` based paralellism (`prange` and parallel vectorize) with explicit thread pool based parallelism. Should be more reliable and no need to set `NUMBA_NUM_THREADS` anymore. Remove env var `QUIMB_NUMBA_PAR`.
- [`Circuit`](quimb.tensor.circuit.Circuit): add `dtype` and `convert_eager` options. `dtype` specifies what the computation should be performed in. `convert_eager` specifies whether to apply this (and any `to_backend` calls) as soon as gates are applied (the default for MPS circuit simulation) or just prior to contraction (the default for exact contraction simulation).
- [`tn.full_simplify`](quimb.tensor.tensor_core.TensorNetwork.full_simplify): add `check_zero` (by default set of `"auto"`) option which explicitly checks for zero tensor norms when equalizing norms to avoid `log10(norm)` resulting in -inf or nan. Since it creates a data dependency that breaks e.g. `jax` tracing, it is optional.
- [`schematic.Drawing`](quimb.schematic.Drawing): add `shorten` kwarg to [line drawing](quimb.schematic.Drawing.line) and [curve drawing](quimb.schematic.Drawing.curve) and examples to {ref}`schematic`.
- [`TensorNetwork`](quimb.tensor.tensor_core.TensorNetwork): add `.backend` and `.dtype_name` properties.


(whats-new-1-9-0)=
## v1.9.0 (2024-11-19)

**Breaking Changes**

- renamed `MatrixProductState.partial_trace` and `MatrixProductState.ptr` to [MatrixProductState.partial_trace_to_mpo](quimb.tensor.tensor_1d.MatrixProductState.partial_trace_to_mpo) to avoid confusion with other `partial_trace` methods that usually produce a dense matrix.

**Enhancements:**

- add [`Circuit.sample_gate_by_gate`](quimb.tensor.circuit.Circuit.sample_gate_by_gate) and related methods [`CircuitMPS.reordered_gates_dfs_clustered`](quimb.tensor.circuit.Circuit.reordered_gates_dfs_clustered) and [`CircuitMPS.get_qubit_distances`](quimb.tensor.circuit.CircuitMPS.get_qubit_distances) for sampling a circuit using the 'gate by gate' method introduced in https://arxiv.org/abs/2112.08499.
- add [`Circuit.draw`](quimb.tensor.circuit.Circuit.draw) for drawing a very simple circuit schematic.
- [`Circuit`](quimb.tensor.circuit.Circuit): by default turn on `simplify_equalize_norms` and use a `group_size=10` for sampling. This should result in faster and more stable sampling.
- [`Circuit`](quimb.tensor.circuit.Circuit): use `numpy.random.default_rng` for random number generation.
- add [`qtn.circ_a2a_rand`](quimb.tensor.circuit_gen.circ_a2a_rand) for generating random all-to-all circuits.
- expose [`qtn.edge_coloring`](quimb.tensor.tensor_arbgeom_tebd.edge_coloring) as top level function and allow layers to be returned grouped.
- add docstring for [`tn.contract_compressed`](quimb.tensor.tensor_core.TensorNetwork.contract_compressed) and by default pick up important settings from the supplied contraction path optimizer (`max_bond` and `compress_late`)
- add [`Tensor.rand_reduce`](quimb.tensor.tensor_core.Tensor.rand_reduce) for randomly removing a tensor index by contracting a random vector into it. One can also supply the value `"r"` to `isel` selectors to use this.
- add `fit-zipup` and `fit-projector` shorthand methods to the general 1d tensor network compression function
- add [`MatrixProductState.compute_local_expectation`](quimb.tensor.tensor_1d.MatrixProductState.compute_local_expectation) for computing many local expectations for a MPS at once, to match the interface for this method elsewhere. These can either be computed via canonicalization (`method="canonical"`), or via explicit left and right environment contraction (`method="envs"`)
- specialize [`CircuitMPS.local_expectation`](quimb.tensor.circuit.CircuitMPS.local_expectation) to make use of the MPS form.
- add [`PEPS.product_state`](quimb.tensor.tensor_2d.PEPS.product_state) for constructing a PEPS representing a product state.
- add [`PEPS.vacuum`](quimb.tensor.tensor_2d.PEPS.vacuum) for constructing a PEPS representing the vacuum state $|000\ldots0\rangle$.
- add [`PEPS.zeros`](quimb.tensor.tensor_2d.PEPS.zeros) for constructing a PEPS whose entries are all zero.
- [`tn.gauge_all_simple`](quimb.tensor.tensor_core.TensorNetwork.gauge_all_simple): improve scheduling and add `damping` and `touched_tids` options.
- [`qtn.SimpleUpdateGen`](quimb.tensor.tensor_arbgeom_tebd.SimpleUpdateGen): add gauge difference update checking and `tol` and `equilibrate` settings. Update `.plot()` method. Default to a small `cutoff`.
- add [`psi.sample_configuration_cluster`](quimb.tensor.tensor_arbgeom.TensorNetworkGenVector.sample_configuration_cluster) for sampling a tensor network using the simple update or cluster style environment approximation.
- add the new doc {ref}`ex-circuit-sampling`

---


(whats-new-1-8-4)=
## v1.8.4 (2024-07-20)

**Bug fixes:**

- fix for MPS sampling with fixed seed ({issue}`247` and {pull}`248`)
- fix for `mps_gate_with_mpo_lazy` ({issue}`246`).

---


(whats-new-1-8-3)=
## v1.8.3 (2024-07-10)

**Enhancements:**

- support for numpy v2.0 and scipy v1.14
- add MPS sampling: [`MatrixProductState.sample_configuration`](quimb.tensor.tensor_1d.MatrixProductState.sample_configuration) and [`MatrixProductState.sample`](quimb.tensor.tensor_1d.MatrixProductState.sample) (generating multiple samples) and use these for [`CircuitMPS.sample`](quimb.tensor.circuit.CircuitMPS.sample) and [`CircuitPermMPS.sample`](quimb.tensor.circuit.CircuitPermMPS.sample).
- add basic [`.plot()`](quimb.tensor.tensor_arbgeom_tebd.TEBDGen.plot) method for SimpleUpdate classes
- add [`edges_1d_chain`](quimb.tensor.geometry.edges_1d_chain) for generating 1D chain edges
- [operatorbuilder](quimb.operator): better coefficient placement for long range MPO building

---


(whats-new-1-8-2)=
## v1.8.2 (2024-06-12)

**Enhancements:**

- [`TNOptimizer`](quimb.tensor.optimize.TNOptimizer) can now accept an arbitrary pytree (nested combination of dicts, lists, tuples, etc. with `TensorNetwork`, `Tensor` or raw `array_like` objects as the leaves) as the target object to optimize.
- [`TNOptimizer`](quimb.tensor.optimize.TNOptimizer) can now directly optimize [`Circuit`](quimb.tensor.circuit.Circuit) objects, returning a new optimized circuit with updated parameters.
- [`Circuit`](quimb.tensor.circuit.Circuit): add `.copy()`, `.get_params()` and `.set_params()` interface methods.
- Update generic TN optimizer docs.
- add [`tn.gen_inds_loops`](quimb.tensor.tensor_core.TensorNetwork.gen_inds_loops) for generating all loops of indices in a TN.
- add [`tn.gen_inds_connected`](quimb.tensor.tensor_core.TensorNetwork.gen_inds_connected) for generating all connected sets of indices in a TN.
- make SVD fallback error catching more generic ({pull}`#238`)
- fix some windows + numba CI issues.
- [`approx_spectral_function`](quimb.linalg.approx_spectral.approx_spectral_function) add plotting and tracking
- add dispatching to various tensor primitives to allow overriding

---


(whats-new-1-8-1)=
## v1.8.1 (2024-05-06)

**Enhancements:**

- [`CircuitMPS`](quimb.tensor.circuit.CircuitMPS) now supports multi qubit gates, including arbitrary multi-controls (which are treated in a low-rank manner), and faster simulation via better orthogonality center tracking.
- add [`CircuitPermMPS`](quimb.tensor.circuit.CircuitPermMPS)
- add [`MatrixProductState.gate_nonlocal`](quimb.tensor.tensor_1d.MatrixProductState.gate_nonlocal) for applying a gate, supplied as a raw matrix, to a non-local and arbitrary number of sites. The kwarg `contract="nonlocal"` can be used to force this method, or the new option `"auto-mps"` will select this method if the gate is non-local ({issue}`230`)
- add [`MatrixProductState.gate_with_mpo`](quimb.tensor.tensor_1d.MatrixProductState.gate_with_mpo) for applying an MPO to an MPS, and immediately compressing back to MPS form using [`tensor_network_1d_compress`](quimb.tensor.tensor_1d_compress.tensor_network_1d_compress)
- add [`MatrixProductState.gate_with_submpo`](quimb.tensor.tensor_1d.MatrixProductState.gate_with_submpo) for applying an MPO acting only of a subset of sites to an MPS
- add [`MatrixProductOperator.from_dense`](quimb.tensor.tensor_1d.MatrixProductOperator.from_dense) for constructing MPOs from dense matrices, including an only subset of sites
- add [`MatrixProductOperator.fill_empty_sites`](quimb.tensor.tensor_1d.MatrixProductOperator.fill_empty_sites) for 'completing' an MPO which only has tensors on a subset of sites with (by default) identities
-  [`MatrixProductState`](quimb.tensor.tensor_1d.MatrixProductState) and [`MatrixProductOperator`](quimb.tensor.tensor_1d.MatrixProductOperator), now support the ``sites`` kwarg in common constructors, enabling the TN to act on a subset of the full ``L`` sites.
- add [`TensorNetwork.drape_bond_between`](quimb.tensor.tensor_core.TensorNetwork.drape_bond_between) for 'draping' an existing bond between two tensors through a third
- add [`Tensor.new_ind_pair_with_identity`](quimb.tensor.tensor_core.Tensor.new_ind_pair_with_identity)
- TN2D, TN3D and arbitrary geom classical partition function builders ([`TN_classical_partition_function_from_edges`](quimb.tensor.tensor_builder.TN_classical_partition_function_from_edges)) now all support `outputs=` kwarg specifying non-marginalized variables
- add simple dense 1-norm belief propagation algorithm [`D1BP`](quimb.tensor.belief_propagation.d1bp.D1BP)
- add [`qtn.enforce_1d_like`](quimb.tensor.tensor_1d_compress.enforce_1d_like) for checking whether a tensor network is 1D-like, including automatically adding strings of identities between non-local bonds, expanding applicability of [`tensor_network_1d_compress`](quimb.tensor.tensor_1d_compress.tensor_network_1d_compress)
- add [`MatrixProductState.canonicalize`](quimb.tensor.tensor_1d.MatrixProductState.canonicalize) as (by default *non-inplace*) version of `canonize`, to follow the pattern of other tensor network methods. `canonize` is now an alias for `canonicalize_` [note trailing underscore].
- add [`MatrixProductState.left_canonicalize`](quimb.tensor.tensor_1d.MatrixProductState.left_canonicalize) as (by default *non-inplace*) version of `left_canonize`, to follow the pattern of other tensor network methods. `left_canonize` is now an alias for `left_canonicalize_` [note trailing underscore].
- add [`MatrixProductState.right_canonicalize`](quimb.tensor.tensor_1d.MatrixProductState.right_canonicalize) as (by default *non-inplace*) version of `right_canonize`, to follow the pattern of other tensor network methods. `right_canonize` is now an alias for `right_canonicalize_` [note trailing underscore].

**Bug fixes:**

- [`Circuit.apply_gate_raw`](quimb.tensor.circuit.Circuit.apply_gate_raw): fix kwarg bug ({pull}`226`)
- fix for retrieving `opt_einsum.PathInfo` for single scalar contraction ({issue}`231`)


---


(whats-new-1-8-0)=
## v1.8.0 (2024-04-10)

**Breaking Changes**

- all singular value renormalization is turned off by default
- [`TensorNetwork.compress_all`](quimb.tensor.TensorNetwork.compress_all)
  now defaults to using some local gauging


**Enhancements:**

- add `quimb.tensor.tensor_1d_compress.py` with functions for compressing generic
  1D tensor networks (with arbitrary local structure) using various methods.
  The methods are:

  - The **'direct'** method: [`tensor_network_1d_compress_direct`](quimb.tensor.tensor_1d_compress.tensor_network_1d_compress_direct)
  - The **'dm'** (density matrix) method: [`tensor_network_1d_compress_dm`](quimb.tensor.tensor_1d_compress.tensor_network_1d_compress_dm)
  - The **'zipup'** method: [`tensor_network_1d_compress_zipup`](quimb.tensor.tensor_1d_compress.tensor_network_1d_compress_zipup)
  - The **'zipup-first'** method: [`tensor_network_1d_compress_zipup_first`](quimb.tensor.tensor_1d_compress.tensor_network_1d_compress_zipup_first)
  - The 1 and 2 site **'fit'** or sweeping method: [`tensor_network_1d_compress_fit`](quimb.tensor.tensor_1d_compress.tensor_network_1d_compress_fit)
  - ... and some more niche methods for debugging and testing.

  And can be accessed via the unified function [`tensor_network_1d_compress`](quimb.tensor.tensor_1d_compress.tensor_network_1d_compress).
  Boundary contraction in 2D can now utilize any of these methods.
- add `quimb.tensor.tensor_arbgeom_compress.py` with functions for compressing
  arbitrary geometry tensor networks using various methods. The methods are:

  - The **'local-early'** method:
    [`tensor_network_ag_compress_local_early`](quimb.tensor.tensor_arbgeom_compress.tensor_network_ag_compress_local_early)
  - The **'local-late'** method:
    [`tensor_network_ag_compress_local_late`](quimb.tensor.tensor_arbgeom_compress.tensor_network_ag_compress_local_late)
  - The **'projector'** method:
    [`tensor_network_ag_compress_projector`](quimb.tensor.tensor_arbgeom_compress.tensor_network_ag_compress_projector)
  - The **'superorthogonal'** method:
    [`tensor_network_ag_compress_superorthogonal`](quimb.tensor.tensor_arbgeom_compress.tensor_network_ag_compress_superorthogonal)
  - The **'l2bp'** method:
    [`tensor_network_ag_compress_l2bp`](quimb.tensor.tensor_arbgeom_compress.tensor_network_ag_compress_l2bp)

  And can be accessed via the unified function
  [`tensor_network_ag_compress`](quimb.tensor.tensor_arbgeom_compress.tensor_network_ag_compress).
  1D compression can also fall back to these methods.
- support PBC in
  [`tn2d.contract_hotrg`](quimb.tensor.tensor_2d.TensorNetwork2D.contract_hotrg),
  [`tn2d.contract_ctmrg`](quimb.tensor.tensor_2d.TensorNetwork2D.contract_ctmrg),
  [`tn3d.contract_hotrg`](quimb.tensor.tensor_3d.TensorNetwork3D.contract_hotrg) and
  the new function
  [`tn3d.contract_ctmrg`](quimb.tensor.tensor_3d.TensorNetwork3D.contract_ctmrg).
- support PBC in
  [`gen_2d_bonds`](quimb.tensor.tensor_2d.gen_2d_bonds) and
  [`gen_3d_bonds`](quimb.tensor.tensor_3d.gen_3d_bonds), with ``cyclic`` kwarg.
- support PBC in
  [`TN2D_rand_hidden_loop`](quimb.tensor.tensor_builder.TN2D_rand_hidden_loop)
  and
  [`TN3D_rand_hidden_loop`](quimb.tensor.tensor_builder.TN3D_rand_hidden_loop),
  with ``cyclic`` kwarg.
- support PBC in the various base PEPS and PEPO construction methods.
- add [`tensor_network_apply_op_op`](quimb.tensor.tensor_arbgeom.tensor_network_apply_op_op)
  for applying 'operator' TNs to 'operator' TNs.
- tweak [`tensor_network_apply_op_vec`](quimb.tensor.tensor_arbgeom.tensor_network_apply_op_vec)
  for applying 'operator' TNs to 'vector' or 'state' TNs.
- add [`tnvec.gate_with_op_lazy`](quimb.tensor.tensor_arbgeom.TensorNetworkGenVector.gate_with_op_lazy)
  method for applying 'operator' TNs to 'vector' or 'state' TNs like $x \rightarrow A x$.
- add [`tnop.gate_upper_with_op_lazy`](quimb.tensor.tensor_arbgeom.TensorNetworkGenOperator.gate_upper_with_op_lazy)
  method for applying 'operator' TNs to the upper indices of 'operator' TNs like $B \rightarrow A B$.
- add [`tnop.gate_lower_with_op_lazy`](quimb.tensor.tensor_arbgeom.TensorNetworkGenOperator.gate_lower_with_op_lazy)
  method for applying 'operator' TNs to the lower indices of 'operator' TNs like $B \rightarrow B A$.
- add [`tnop.gate_sandwich_with_op_lazy`](quimb.tensor.tensor_arbgeom.TensorNetworkGenOperator.gate_sandwich_with_op_lazy)
  method for applying 'operator' TNs to the upper and lower indices of 'operator' TNs like $B \rightarrow A B A^\dagger$.
- unify all TN summing routines into
  [`tensor_network_ag_sum](quimb.tensor.tensor_arbgeom.tensor_network_ag_sum),
  which allows summing any two tensor networks with matching site tags and
  outer indices, replacing specific MPS, MPO, PEPS, PEPO, etc. summing routines.
- add [`rand_symmetric_array`](quimb.tensor.tensor_builder.rand_symmetric_array),
  [`rand_tensor_symmetric`](quimb.tensor.tensor_builder.rand_tensor_symmetric)
  [`TN2D_rand_symmetric`](quimb.tensor.tensor_builder.TN2D_rand_symmetric)
  for generating random symmetric arrays, tensors and 2D tensor networks.

**Bug fixes:**

- fix scipy sparse monkey patch for scipy>=1.13 ({issue}`222`)
- fix autoblock bug where connected sectors were not being merged ({issue}`223`)


---


(whats-new-1-7-3)=
## v1.7.3 (2024-02-08)

**Enhancements:**

- [qu.randn](quimb.randn): support `dist="rademacher"`.
- support `dist` and other `randn` options in various TN builders.

**Bug fixes:**

- restore fallback (to `scipy.linalg.svd` with driver='gesvd') behavior for truncated SVD with numpy backend.


---


(whats-new-1-7-2)=
## v1.7.2 (2024-01-30)

**Enhancements:**

- add `normalized=True` option to [`tensor_network_distance`](quimb.tensor.tensor_core.tensor_network_distance) for computing the normalized distance between tensor networks: $2 |A - B| / (|A| + |B|)$, which is useful for convergence checks. [`Tensor.distance_normalized`](quimb.tensor.tensor_core.Tensor.distance_normalized) and [`TensorNetwork.distance_normalized`](quimb.tensor.tensor_core.TensorNetwork.distance_normalized) added as aliases.
- add [`TensorNetwork.cut_bond`](quimb.tensor.tensor_core.TensorNetwork.cut_bond) for cutting a bond index

**Bug fixes:**

- removed import of deprecated `numba.generated_jit` decorator.


---


(whats-new-1-7-1)=
## v1.7.1 (2024-01-30)

**Enhancements:**

- add [`TensorNetwork.visualize_tensors`](quimb.tensor.drawing.visualize_tensors)
  for visualizing the actual data entries of an entire tensor network.
- add [`ham.build_mpo_propagator_trotterized`](quimb.tensor.tensor_1d_tebd.LocalHam1D.build_mpo_propagator_trotterized)
  for building a trotterized propagator from a local 1D hamiltonian. This
  also includes updates for creating 'empty' tensor networks using
  [`TensorNetwork.new`](quimb.tensor.tensor_core.TensorNetwork.new), and
  building up gates from empty tensor networks using
  [`TensorNetwork.gate_inds_with_tn`](quimb.tensor.tensor_core.TensorNetwork.gate_inds_with_tn).
- add more options to [`Tensor.expand_ind`](quimb.tensor.tensor_core.Tensor.expand_ind)
  and [`Tensor.new_ind`](quimb.tensor.tensor_core.Tensor.new_ind): repeat
  tiling mode and random padding mode.
- tensor decomposition: make ``eigh_truncated`` backend agnostic.
- [`tensor_compress_bond`](quimb.tensor.tensor_core.tensor_compress_bond): add
  `reduced="left"` and `reduced="right"` modes for when the pair of tensors is
  already in a canonical form.
- add [`qtn.TN2D_embedded_classical_ising_partition_function`](quimb.tensor.tensor_builder.TN2D_embedded_classical_ising_partition_function) for constructing 2D
  (triangular) tensor networks representing all-to-all classical ising
  partition functions.

**Bug fixes:**

- fix bug in [`kruas_op`](quimb.kraus_op) when operator spanned multiple
  subsystems ({issue}`214`)
- fix bug in [`qr_stabilized`](quimb.tensor.decomp.qr_stabilized) when the
  diagonal of `R` has significant imaginary parts.
- fix bug in quantum discord computation when the state was diagonal ({issue}`217`)


---


(whats-new-1-7-0)=
## v1.7.0 (2023-12-08)

**Breaking Changes**

- {class}`~quimb.tensor.Circuit` : remove `target_size` in preparation for
  all contraction specifications to be encapsulated at the contract level (e.g.
  with `cotengra`)
- some TN drawing options (mainly arrow options) have changed due to the
  backend change detailed below.

**Enhancements:**

- [TensorNetwork.draw](quimb.tensor.TensorNetwork.draw): use `quimb.schematic`
  for main `backend="matplotlib"` drawing. Enabling:
    1. multi tag coloring for single tensors
    2. arrows and labels on multi-edges
    3. better sizing of tensors using absolute units
    4. neater single tensor drawing, in 2D and 3D
* add [quimb.schematic.Drawing](quimb.schematic.Drawing) from experimental
  submodule, add example docs at {ref}`schematic`. Add methods `text_between`,
  `wedge`, `line_offset` and other tweaks for future use by main TN drawing.
- upgrade all contraction to use `cotengra` as the backend
- [`Circuit`](quimb.tensor.Circuit) : allow any gate to be controlled by any
  number of qubits.
- [`Circuit`](quimb.tensor.Circuit) : support for parsing `openqasm2`
  specifications now with custom and nested gate definitions etc.
- add [`is_cyclic_x`](quimb.tensor.TensorNetwork2D.is_cyclic_x),
  [`is_cyclic_y`](quimb.tensor.TensorNetwork2D.is_cyclic_y) and
  [`is_cyclic_z`](quimb.tensor.TensorNetwork3D.is_cyclic_z) to
  [TensorNetwork2D](quimb.tensor.TensorNetwork2D) and
  [TensorNetwork3D](quimb.tensor.TensorNetwork3D).
- add [TensorNetwork.compress_all_1d](quimb.tensor.TensorNetwork.compress_all_1d)
  for compressing generic tensor networks that you promise have a 1D topology,
  without casting as a [TensorNetwork1D](quimb.tensor.TensorNetwork1D).
- add [MatrixProductState.from_fill_fn](quimb.tensor.tensor_1d.MatrixProductState.from_fill_fn)
  for constructing MPS from a function that fills the tensors.
- add [Tensor.idxmin](quimb.tensor.Tensor.idxmin) and
  [Tensor.idxmax](quimb.tensor.Tensor.idxmax) for finding the index of the
  minimum/maximum element.
- 2D and 3D classical partition function TN builders: allow output indices.
- [`quimb.tensor.belief_propagation`]([`quimb.tensor.belief_propagation`]):
  add various 1-norm/2-norm dense/lazy BP algorithms.

**Bug fixes:**

- fixed bug where an output index could be removed by squeezing when
  performing tensor network simplifications.


---


(whats-new-1-6-0)=
## v1.6.0 (2023-09-10)

**Breaking Changes**

- Quantum circuit RZZ definition corrected (angle changed by -1/2 to match
  qiskit).

**Enhancements:**

- add OpenQASM 2.0 parsing support: [`Circuit.from_openqasm2_file`](quimb.tensor.Circuit.from_openqasm2_file),
- [`Circuit`](quimb.tensor.Circuit): add RXX, RYY, CRX, CRY, CRZ, toffoli, fredkin, givens gates
- truncate TN pretty html reprentation to 100 tensors for performance
- add [`Tensor.sum_reduce`](quimb.tensor.Tensor.sum_reduce) and [`Tensor.vector_reduce`](quimb.tensor.Tensor.vector_reduce)
- [`contract_compressed`](quimb.tensor.TensorNetwork.contract_compressed), default to 'virtual-tree' gauge
- add [`TN_rand_tree`](quimb.tensor.TN_rand_tree)
- `experimental.operatorbuilder`: fix parallel and heisenberg builder
- make parametrized gate generation even more robost (ensure matching types
  so e.g. tensorflow can be used)

**Bug fixes:**

- fix gauge size check for some backends

---


(whats-new-1-5-1)=
## v1.5.1 (2023-07-28)

**Enhancements:**

- add {func}`~quimb.tensor.tensor_builder.MPS_COPY`.
- add 'density matrix' and 'zip-up' MPO-MPS algorithms.
- add `drop_tags` option to {meth}`~quimb.tensor.tensor_contract`
- {meth}`compress_all_simple`, allow cutoff.
- add structure checking debug methods: {meth}`Tensor.check` and
  {meth}`TensorNetwork.check`.
- add several direction contraction utility functions: {func}`get_symbol`,
  {func}`inds_to_eq` and {func}`array_contract`.

**Bug fixes:**

- {class}`Circuit`: use stack for more robust parametrized gate generation
- fix for {meth}`gate_with_auto_swap` for `i > j`.
- fix bug where calling `tn.norm()` would mangle indices.

---


(whats-new-1-5-0)=
## v1.5.0 (2023-05-03)

**Enhancements**

- refactor 'isometrize' methods including new "cayley", "householder" and
  "torch_householder" methods. See {func}`quimb.tensor.decomp.isometrize`.
- add {meth}`~quimb.tensor.tensor_core.TensorNetwork.compute_reduced_factor`
  and {meth}`~quimb.tensor.tensor_core.TensorNetwork.insert_compressor_between_regions`
  methos, for some RG style algorithms.
- add the `mode="projector"` option for 2D tensor network contractions
- add HOTRG style coarse graining and contraction in 2D and 3D. See
  {meth}`~quimb.tensor.tensor_2d.TensorNetwork2D.coarse_grain_hotrg`,
  {meth}`~quimb.tensor.tensor_2d.TensorNetwork2D.contract_hotrg`,
  {meth}`~quimb.tensor.tensor_3d.TensorNetwork3D.coarse_grain_hotrg`, and
  {meth}`~quimb.tensor.tensor_3d.TensorNetwork3D.contract_hotrg`,
- add CTMRG style contraction for 2D tensor networks:
  {meth}`~quimb.tensor.tensor_2d.TensorNetwork2D.contract_ctmrg`
- add 2D tensor network 'corner double line' (CDL) builders:
  {func}`~quimb.tensor.tensor_builder.TN2D_corner_double_line`
- update the docs to use the [furo](https://pradyunsg.me/furo/) theme,
  [myst_nb](https://myst-nb.readthedocs.io/en/latest/) for notebooks, and
  several other `sphinx` extensions.
- add the `'adabelief'` optimizer to
  {class}`~quimb.tensor.optimize.TNOptimizer` as well as a quick plotter:
  {meth}`~quimb.tensor.optimize.TNOptimizer.plot`
- add initial 3D plotting methods for tensors networks (
  `TensorNetwork.draw(dim=3, backend='matplotlib3d')` or
  `TensorNetwork.draw(dim=3, backend='plotly')`
  ). The new `backend='plotly'` can also be used for 2D interactive plots.
- Update {func}`~quimb.tensor.tensor_builder.HTN_from_cnf` to handle more
  weighted model counting formats.
- Add {func}`~quimb.tensor.tensor_builder.cnf_file_parse`
- Add {func}`~quimb.tensor.tensor_builder.random_ksat_instance`
- Add {func}`~quimb.tensor.tensor_builder.TN_from_strings`
- Add {func}`~quimb.tensor.tensor_builder.convert_to_2d`
- Add {func}`~quimb.tensor.tensor_builder.TN2D_rand_hidden_loop`
- Add {func}`~quimb.tensor.tensor_builder.convert_to_3d`
- Add {func}`~quimb.tensor.tensor_builder.TN3D_corner_double_line`
- Add {func}`~quimb.tensor.tensor_builder.TN3D_rand_hidden_loop`
- various optimizations for minimizing computational graph size and
  construction time.
- add `'lu'`, `'polar_left'` and `'polar_right'` methods to
  {func}`~quimb.tensor.tensor_core.tensor_split`.
- add experimental arbitrary hamilotonian MPO building
- {class}`~quimb.tensor.tensor_core.TensorNetwork`: allow empty constructor
  (i.e. no tensors representing simply the scalar 1)
- {meth}`~quimb.tensor.tensor_core.TensorNetwork.drop_tags`: allow all tags to
  be dropped
- tweaks to compressed contraction and gauging
- add jax, flax and optax example
- add 3D and interactive plotting of tensors networks with via plotly.
- add pygraphiviz layout options
- add {meth}`~quimb.tensor.tensor_core.TensorNetwork.combine` for unified
  handling of combining
  tensor networks potentially with structure
- add HTML colored pretty printing of tensor networks for notebooks
- add `quimb.experimental.cluster_update.py`

**Bug fixes:**

- fix {func}`~quimb.tensor.decomp.qr_stabilized` bug for strictly upper
  triangular R factors.

---


(whats-new-1-4-2)=
## v1.4.2 (2022-11-28)

**Enhancements**

- move from versioneer to to
  [setuptools_scm](https://pypi.org/project/setuptools-scm/) for versioning

---


(whats-new-1-4-1)=
## v1.4.1 (2022-11-28)

**Enhancements**

- unify much functionality from 1D, 2D and 3D into general arbitrary geometry
  class {class}`quimb.tensor.tensor_arbgeom.TensorNetworkGen`
- refactor contraction, allowing using cotengra directly
- add {meth}`~quimb.tensor.tensor_core.Tensor.visualize` for visualizing the
  actual data entries of an arbitrarily high dimensional tensor
- add {class}`~quimb.tensor.circuit.Gate` class for more robust tracking and
  manipulation of gates in quantum {class}`~quimb.tensor.circuit.Circuit`
  simulation
- tweak TN drawing style and layout
- tweak default gauging options of compressed contraction
- add {meth}`~quimb.tensor.tensor_core.TensorNetwork.compute_hierarchical_grouping`
- add {meth}`~quimb.tensor.tensor_core.Tensor.as_network`
- add {meth}`~quimb.tensor.tensor_core.TensorNetwork.inds_size`
- add {meth}`~quimb.tensor.tensor_core.TensorNetwork.get_hyperinds`
- add {meth}`~quimb.tensor.tensor_core.TensorNetwork.outer_size`
- improve {meth}`~quimb.tensor.tensor_core.TensorNetwork.group_inds`
- refactor tensor decompositiona and 'isometrization' methods
- begin supporting pytree specifications in `TNOptimizer`, e.g. for constants
- add `experimental` submodule for new sharing features
- register tensor and tensor network objects with `jax` pytree interface
  ({pull}`150`)
- update CI infrastructure

**Bug fixes:**

> - fix force atlas 2 and `weight_attr` bug ({issue}`126`)
> - allow unpickling of `PTensor` objects ({issue}`128`, {pull}`131`)

---


(whats-new-1-4-0)=
## v1.4.0 (2022-06-14)

**Enhancements**

- Add 2D tensor network support and algorithms
- Add 3D tensor network infrastructure
- Add arbitrary geometry quantum state infrastructure
- Many changes to {class}`~quimb.tensor.optimize.TNOptimizer`
- Many changes to TN drawing
- Many changes to {class}`~quimb.tensor.circuit.Circuit` simulation
- Many improvements to TN simplification
- Make all tag and index operations deterministic
- Add {func}`~quimb.tensor.tensor_core.tensor_network_sum`,
  {func}`~quimb.tensor.tensor_core.tensor_network_distance` and
  {meth}`~quimb.tensor.tensor_core.TensorNetwork.fit`
- Various memory and performance improvements
- Various graph generators and TN builders

---


(whats-new-1-3-0)=
## v1.3.0 (2020-02-18)

**Enhancements**

- Added time dependent evolutions to {class}`~quimb.evo.Evolution` when integrating a pure state - see {ref}`time-dependent-evolution` - as well as supporting `LinearOperator` defined hamiltonians ({pull}`40`).
- Allow the {class}`~quimb.evo.Evolution` callback `compute=` to optionally access the Hamiltonian ({pull}`49`).
- Added {meth}`quimb.tensor.tensor_core.Tensor.randomize` and {meth}`quimb.tensor.tensor_core.TensorNetwork.randomize` to randomize tensor and tensor network entries.
- Automatically squeeze tensor networks when rank-simplifying.
- Add {meth}`~quimb.tensor.tensor_1d.TensorNetwork1DFlat.compress_site` for compressing around single sites of MPS etc.
- Add {func}`~quimb.tensor.tensor_builder.MPS_ghz_state` and {func}`~quimb.tensor.tensor_builder.MPS_w_state` for building bond dimension 2 open boundary MPS reprentations of those states.
- Various changes in conjunction with [autoray](https://github.com/jcmgray/autoray) to improve the agnostic-ness of tensor network operations with respect to the backend array type.
- Add {func}`~quimb.tensor.tensor_core.new_bond` on top of {meth}`quimb.tensor.tensor_core.Tensor.new_ind` and {meth}`quimb.tensor.tensor_core.Tensor.expand_ind` for more graph orientated construction of tensor networks, see {ref}`tn-creation-graph-style`.
- Add the {func}`~quimb.gen.operators.fsim` gate.
- Make the parallel number generation functions use new `numpy 1.17+` functionality rather than `randomgen` (which can still be used as the underlying bit generator) ({pull}`50`)
- TN: rename `contraction_complexity` to {meth}`~quimb.tensor.tensor_core.TensorNetwork.contraction_width`.
- TN: update {meth}`quimb.tensor.tensor_core.TensorNetwork.rank_simplify`, to handle hyper-edges.
- TN: add {meth}`quimb.tensor.tensor_core.TensorNetwork.diagonal_reduce`, to automatically collapse all diagonal tensor axes in a tensor network, introducing hyper edges.
- TN: add {meth}`quimb.tensor.tensor_core.TensorNetwork.antidiag_gauge`, to automatically flip all anti-diagonal tensor axes in a tensor network allowing subsequent diagonal reduction.
- TN: add {meth}`quimb.tensor.tensor_core.TensorNetwork.column_reduce`, to automatically identify tensor axes with a single non-zero column, allowing the corresponding index to be cut.
- TN: add {meth}`quimb.tensor.tensor_core.TensorNetwork.full_simplify`, to iteratively perform all the above simplifications in a specfied order until nothing is left to be done.
- TN: add `num_tensors` and `num_indices` attributes, show `num_indices` in `__repr__`.
- TN: various improvements to the pytorch optimizer ({pull}`34`)
- TN: add some built-in 1D quantum circuit ansatzes:
  {func}`~quimb.tensor.circuit_gen.circ_ansatz_1D_zigzag`,
  {func}`~quimb.tensor.circuit_gen.circ_ansatz_1D_brickwork`, and
  {func}`~quimb.tensor.circuit_gen.circ_ansatz_1D_rand`.
- **TN: add parametrized tensors** {class}`~quimb.tensor.tensor_core.PTensor` and so trainable, TN based quantum circuits -- see {ref}`example-tn-training-circuits`.

**Bug fixes:**

- Fix consistency of {func}`~quimb.calc.fidelity` by making the unsquared version the default for the case when either state is pure, and always return a real number.
- Fix a bug in the 2D system example for when `j != 1.0`
- Add environment variable `QUIMB_NUMBA_PAR` to set whether numba should use automatic parallelization - mainly to fix travis segfaults.
- Make cache import and initilization of `petsc4py` and `slepc4py` more robust.

---


(whats-new-1-2-0)=
## v1.2.0 (2019-06-06)

**Enhancements**

- Added {func}`~quimb.calc.kraus_op` for general, noisy quantum operations
- Added {func}`~quimb.calc.projector` for constructing projectors from observables
- Added {func}`~quimb.calc.measure` for measuring and collapsing quantum states
- Added {func}`~quimb.calc.cprint` pretty printing states in computational basis
- Added {func}`~quimb.calc.simulate_counts` for simulating computational basis counts
- TN: Add {meth}`quimb.tensor.tensor_core.TensorNetwork.rank_simplify`
- TN: Add {meth}`quimb.tensor.tensor_core.TensorNetwork.isel`
- TN: Add {meth}`quimb.tensor.tensor_core.TensorNetwork.cut_iter`
- TN: Add `'split-gate'` gate mode
- TN: Add {class}`~quimb.tensor.optimize_tensorflow.TNOptimizer` for tensorflow based optimization
  of arbitrary, contstrained tensor networks.
- TN: Add {meth}`quimb.tensor.tensor_1d.Dense1D.rand`
- TN: Add {func}`~quimb.tensor.tensor_core.connect` to conveniently set a shared index for tensors
- TN: make many more tensor operations agnostic of the array backend (e.g. numpy, cupy,
  tensorflow, ...)
- TN: allow {func}`~quimb.tensor.tensor_1d.align_TN_1D` to take an MPO as the first argument
- TN: add {meth}`~quimb.tensor.tensor_builder.SpinHam1D.build_sparse`
- TN: add {meth}`quimb.tensor.tensor_core.Tensor.unitize` and {meth}`quimb.tensor.tensor_core.TensorNetwork.unitize` to impose unitary/isometric constraints on tensors specfied using the `left_inds` kwarg
- Many updates to tensor network quantum circuit
  ({class}`quimb.tensor.circuit.Circuit`) simulation including:

  - {class}`quimb.tensor.circuit.CircuitMPS`
  - {class}`quimb.tensor.circuit.CircuitDense`
  - 49-qubit depth 30 circuit simulation example {ref}`quantum-circuit-example`

- Add `from quimb.gates import *` as shortcut to import `X, Z, CNOT, ...`.

- Add {func}`~quimb.gen.operators.U_gate` for parametrized arbitrary single qubit unitary

**Bug fixes:**

- Fix `pkron` for case `len(dims) == len(inds)` ({issue}`17`, {pull}`18`).
- Fix `qarray` printing for older `numpy` versions
- Fix TN quantum circuit bug where Z and X rotations were swapped
- Fix variable bond MPO building ({issue}`22`) and L=2 DMRG
- Fix `norm(X, 'trace')` for non-hermitian matrices
- Add `autoray` as dependency ({issue}`21`)
