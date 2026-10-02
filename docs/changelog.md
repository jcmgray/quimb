# Changelog

Release notes for `quimb`.

## v1.16.0 (unreleased)

### Breaking changes

**Decomposition and compression defaults**

- [`array_split`](#array_split) and low-level rank-revealing decompositions now default to ``cutoff_mode="rel"`` instead of ``"rsum2"``. 1D ``zipup`` compression also defaults to ``"rel"``.
- Randomized SVD and batched decompositions reject active cumulative cutoff modes. Use ``"abs"`` or ``"rel"`` instead.
- [`tensor_network_1d_compress`](#tensor_network_1d_compress) with an arbitrary-geometry ``method`` always returns a 1D network, inserting identities along long-range bonds. To keep the geometry, call [`tensor_network_ag_compress`](#tensor_network_ag_compress) directly, as 2D boundary contraction now does for periodic boundaries, which therefore need an ag ``method``.

**Hilbert spaces and Hubbard models**

- [`HilbertSpace`](#HilbertSpace): site ordering is now immutable. ``set_ordering`` raises ``TypeError``. Use [`with_ordering`](#HilbertSpace.with_ordering) to create a space with a different ordering.
- [`fermi_hubbard_from_edges`](#fermi_hubbard_from_edges): default to ``order="interleaved"``, alternating spins at each coordinate to avoid MPO bond dimensions growing with system size, with slightly slower matrix-vector products. Rebuild anything keyed by rank or flat configuration. Use ``order="blocked"`` for the old layout.

**Partial traces and local expectations**

- 2D [`compute_local_expectation`](#TensorNetwork2DVector.compute_local_expectation): compute ``tr(rho G)`` from each term's reduced density matrix and normalize locally by default (``normalized=True``). With ``return_all=True``, return plain values, or ``(expec, trace)`` pairs with ``normalized="return"``.
- 2D [`compute_local_expectation`](#TensorNetwork2DVector.compute_local_expectation) and the new 2D partial trace, expectation, and environment methods require ``max_bond``. Supply ``None`` explicitly for no limit, which is not recommended in 2D.
- 2D [`partial_trace`](#TensorNetwork2DVector.partial_trace) and [`local_expectation`](#TensorNetwork2DVector.local_expectation) use the boundary or envs routes. For the previous arbitrary-geometry compressed contraction with ``optimize``, call [`TensorNetworkGenVector.partial_trace`](#TensorNetworkGenVector.partial_trace) or [`TensorNetworkGenVector.local_expectation`](#TensorNetworkGenVector.local_expectation) directly.
- Partial traces derive bra index names by replacing a leading ``k`` with ``b``, falling back to random uuids on a name collision. This affects [`partial_trace_exact`](#TensorNetworkGenVector.partial_trace_exact), [`make_reduced_density_matrix`](#TensorNetworkGenVector.make_reduced_density_matrix), and 3D partial traces. See [`get_bra_inds`](#get_bra_inds).
- Cluster expectations default to ``smudge=1e-12`` and ``optimize="auto-hq"``, matching cluster partial traces.

### Enhancements

#### Tensor decompositions

- Split functions accept ``cutoff="auto"``: ``1e-10`` for exact decompositions, disabled for randomized SVD.
- [`svd_rand_truncated`](#svd_rand_truncated): support ``"abs"`` and ``"rel"`` cutoffs on the sketched spectrum, and ``noise_dist="normal"`` or ``"rademacher"`` for the random sketch.
- Batched ``"abs"`` and ``"rel"`` cutoffs treat blocks as one spectrum and retain a common bond dimension. Both modes support renormalization, including for batched decompositions.

#### 1D compression and MPS

- Add successive deterministic compression (``method="sdc"``), ``sdc-oversample``, and randomized variants ``sdcr`` and ``sdcr-oversample``, based on https://arxiv.org/abs/2601.19650.
- ``direct``, ``dm``, ``zipup``, ``sdc``, ``sdcr``, ``src``, ``srcmps``, ``fit``, and their oversampling variants handle long-range bonds directly, supporting ``symmray`` abelian and fermionic tensors (for ``srcmps`` and ``fit``, with a supplied ``tn_fit``).
- ``dm`` uses QR-SVD when a site's density matrix would exceed its rank bound, reducing layered PEPS norm boundary contraction costs (https://arxiv.org/abs/2406.09769), including for fermionic networks.
- ``src``, ``srcmps``, ``fit``, and their oversampling variants accept a ``seed`` or random generator and match the backend's device and dtype (autoray v0.10.0 or newer). By default, use the backend's global random state.
- ``src`` and ``srcmps`` default to ``cutoff=0.0``. ``sdcr`` leaves the cutoff disabled with its default randomized SVD.
- ``zipup``, ``sdc``, and ``sdcr`` oversampling accept ``cutoff_mode_oversample``. Zipup also accepts ``cutoff_oversample="auto"``. All 1D oversampling methods accept ``compress_opts_final`` to configure the final direct sweep separately.
- ``fit`` with ``bsz=1`` supports fermionic tensor networks, but warns for odd-parity tensors because results are likely incorrect.
- Add [`MatrixProductState.from_product`](#MatrixProductState.from_product), also used by [`MPS_product_state`](#MPS_product_state), with support for block-sparse single-site vectors.
- Compression functions accept tag groups in ``site_tags``; see [`parse_site_tag_groups`](#parse_site_tag_groups). The 1D, 2D, and arbitrary-geometry wrappers also accept custom ``method`` callables.

#### Partial traces, environments, and 2D contraction

- Add ``quimb.tensor.environments`` for computing many block or plane environments in one sweep on open or periodic geometries. [`EnvironmentPlan`](#EnvironmentPlan) supports ``"tree"`` and ``"cut"`` schedules; [`gen_exact_environments`](#gen_exact_environments) and [`gen_compressed_environments`](#gen_compressed_environments) yield environments as they become ready. Choose a 1D, 2D, or arbitrary-geometry compressor with ``compress_fn``.
- 1D partial traces and local expectations support periodic MPS with ``route=None``, ``"canonical"``, or ``"envs"``. [`MatrixProductState.partial_trace`](#MatrixProductState.partial_trace) returns a dense reduced density matrix; [`compute_partial_traces`](#MatrixProductState.compute_partial_traces) computes many at once. Add [`local_expectation`](#MatrixProductState.local_expectation) alongside [`compute_local_expectation`](#MatrixProductState.compute_local_expectation), with explicit methods for each route.
- Add [`TensorNetwork1D.gen_block_environments`](#TensorNetwork1D.gen_block_environments) and [`compute_block_environments`](#TensorNetwork1D.compute_block_environments) for exact block environments, and [`is_cyclic`](#TensorNetwork1D.is_cyclic) to check for periodic chains.
- 2D partial traces and local expectations support periodic PEPS with ``route=None``, ``"boundary"``, or ``"envs"`` and options ``max_bond``, ``cutoff``, and ``method``. Add [`TensorNetwork2DVector.partial_trace`](#TensorNetwork2DVector.partial_trace), [`compute_partial_traces`](#TensorNetwork2DVector.compute_partial_traces), and [`local_expectation`](#TensorNetwork2DVector.local_expectation), with explicit methods for each route.
- Add [`TensorNetwork2D.gen_block_environments`](#TensorNetwork2D.gen_block_environments) and [`compute_block_environments`](#TensorNetwork2D.compute_block_environments) for compressed row or column block environments, and [`compute_plaquette_environments_via_envs`](#TensorNetwork2D.compute_plaquette_environments_via_envs) for plaquettes, including those crossing periodic boundaries.
- The 2D ``"envs"`` route accepts ``second_dense`` to control exact contraction of the remaining strip, by default only for strips one plane wide. It also supports ``autogroup`` and ``first_contract``.
- Add the 2D ``"cluster_boundary"`` route: [`partial_trace_cluster_boundary`](#TensorNetwork2DVector.partial_trace_cluster_boundary), [`compute_partial_traces_cluster_boundary`](#TensorNetwork2DVector.compute_partial_traces_cluster_boundary), and [`compute_local_expectation_cluster_boundary`](#TensorNetwork2DVector.compute_local_expectation_cluster_boundary). Control cluster size with ``max_distance``, optionally insert simple-update ``gauges``, and use 1D boundary compression methods. Clusters can cross periodic boundaries.
- [`partial_trace_cluster_boundary`](#TensorNetwork2DVector.partial_trace_cluster_boundary) accepts ``max_separation`` to control uncompressed lines around kept sites. ``0`` compresses the kept line into a boundary; ``0.5`` compresses its ket and bra layers into opposite boundaries. [`contract_boundary`](#TensorNetwork2D.contract_boundary) with ``around`` also respects ``max_separation`` on each side of the target region.
- Add [`compute_partial_traces_exact`](#TensorNetworkGenVector.compute_partial_traces_exact) and [`compute_partial_traces_cluster`](#TensorNetworkGenVector.compute_partial_traces_cluster) for computing many reduced density matrices at once.
- Partial traces accept ``get="tn"`` for an uncontracted network and ``normalized="return"`` for ``(rho, trace)`` without normalization, including exact, cluster, and [`D2BP.partial_trace`](#D2BP.partial_trace) methods.
- 2D [`contract_boundary`](#TensorNetwork2D.contract_boundary) with ``method="full-bond"`` accepts ``compress_opts`` to configure the similarity decomposition, or ``similarity_method`` as a shortcut.

#### Gates and circuits

- Add ``dagger`` and ``transpose`` options to tensor-network gating methods, including [`tensor_network_gate_inds`](#tensor_network_gate_inds), [`tensor_network_gate_sandwich_inds`](#tensor_network_gate_sandwich_inds), [`tensor_network_ag_gate`](#tensor_network_ag_gate), [`tensor_network_ag_gate_simple`](#tensor_network_ag_gate_simple), and [`gate_sandwich_with_op_lazy`](#TensorNetworkGenOperator.gate_sandwich_with_op_lazy). These apply $G^\dagger$ or $G^T$ instead of $G$. [`MatrixProductState.gate_nonlocal`](#MatrixProductState.gate_nonlocal) also accepts ``transpose``.
- Add [`MatrixProductOperator.gate_sandwich_with_auto_swap`](#MatrixProductOperator.gate_sandwich_with_auto_swap) for two-site gate sandwiches that preserve canonical form, including long-range gates, with optional exponent stripping.
- [`CircuitPEPOSimpleUpdate`](#CircuitPEPOSimpleUpdate): add ``dtype``, ``to_backend``, and ``convert_eager`` options, matching [`CircuitPEPSSimpleUpdate`](#CircuitPEPSSimpleUpdate).

#### Time evolution and run management

- Add [`LocalHamGen.get_trotter_gates`](#LocalHamGen.get_trotter_gates) for first-, second-, and fourth-order Trotter approximations over multiple steps, inherited by 1D, 2D, and 3D Hamiltonians. [`TrotterGate`](#TrotterGate) objects unpack as ``U, where``. Consecutive uses of the same commuting layer are fused by default. [`trotter_schedule`](#trotter_schedule) returns the product formula alone.
- [`build_mpo_propagator_trotterized`](#LocalHam1D.build_mpo_propagator_trotterized) and [`build_pepo_propagator_trotterized`](#LocalHam2D.build_pepo_propagator_trotterized) accept ``order`` (default 1). The 1D method also gains ``ordering``, matching the 2D method.
- TEBD and simple-update classes accept ``logdir`` and ``log_every`` to write ``progress.json``. Use [`plot_progress_log`](#plot_progress_log) with ``watch=True`` to monitor a running job, or [`load_progress_log`](#load_progress_log) for raw data.
- These classes accept ``checkpoint_every`` to save ``checkpoint.pkl`` in ``logdir``, including a final checkpoint on normal or graceful completion. Continue with ``resume=True`` or load with ``from_checkpoint``; resumed runs restore state and evolution options, so omit ``psi0`` and ``ham``.
- ``log_every`` and ``checkpoint_every`` accept sweep counts or durations such as ``"10mins"``; see [`parse_time_spec`](#parse_time_spec).
- A ``STOP`` file in ``logdir`` or the first Ctrl-C during [`evolve`](#TEBDSweepMixin.evolve) stops the run after the current sweep. The file is removed once seen; a second interrupt stops immediately. Use ``graceful_interrupt=False`` for the previous interrupt behavior.

#### Belief propagation and loop expansions

- Support fermionic BP projector compression with ``method="projector", canonize="bp"`` and fermionic gates with [`D2BP.gate_`](#D2BP.gate_).
- All BP classes accept ``diis`` and ``damping`` in both ``__init__`` and [`run`](#BeliefPropagationCommon.run). Values passed to ``run`` become the new defaults.
- ``D1BP``, ``D2BP``, ``L1BP``, ``L2BP``, and [`TensorNetwork.gauge_all_simple`](#TensorNetwork.gauge_all_simple) accept ``sweep_order``, with an efficient alternating schedule for trees. See [`SweepScheduler`](#SweepScheduler) for other options.
- Add [`gen_gloops_edge_induced`](#networking.gen_gloops_edge_induced) (``tn.gen_gloops_edge_induced``), yielding [`NetworkPatch`](#NetworkPatch) objects that distinguish loops using different bonds on the same tensors.
- Infinite 2D generalized-loop expectations accept ``max_size`` and ``num_joins``. Reuse their ``info`` cache across loop settings while the state, gauges, and operators remain unchanged.

#### Network construction

- [`TN_from_strings`](#TN_from_strings): add ``join_prefer="short"`` or ``"long"`` to favor smaller or larger loops, and ``join_avoid_loop_length`` (default 2) to avoid loops up to a given length. Set it to 0 to disable loop avoidance. ``join_prefer`` does not support ``join="all"``.
- Add hidden-cactus networks [`TN2D_rand_hidden_cactus`](#TN2D_rand_hidden_cactus), [`TN3D_rand_hidden_cactus`](#TN3D_rand_hidden_cactus), and [`TN_rand_hidden_cactus`](#TN_rand_hidden_cactus) to extend hidden correlations while preserving lattice bond dimensions and low exact contraction cost. Also available through ``TN_from_strings(..., join_trees=True)``.
- Add [`TN_rand_hidden_loop`](#TN_rand_hidden_loop), the arbitrary-graph counterpart of [`TN2D_rand_hidden_loop`](#TN2D_rand_hidden_loop).
- Add [`TensorNetwork.insert_projectors_between_regions`](#TensorNetwork.insert_projectors_between_regions) to insert precomputed projector arrays between regions.
- [`TN_matching`](#TN_matching): add ``edges`` to choose which sites to join, for example a plain chain ignoring long-range bonds.

#### Hilbert spaces and utilities

- [`HilbertSpace`](#HilbertSpace): support ``U1U1`` sectors with any site ordering and a ``species`` argument selecting each site's conserved charge. Accept sector forms ``{species: filling}``, ``(ka, kb)``, and ``((na, ka), (nb, kb))``. ``order="blocked"`` groups species; ``"interleaved"`` alternates them at each position.
- Add [`distance_from_overlaps`](#distance_from_overlaps) for normalized distances and infidelities from three precomputed overlaps.
- [`save_to_disk`](#save_to_disk): add ``atomic`` to prevent partially written targets.

### Deprecations

- [`Tensor.gate`](#Tensor.gate): rename ``transposed`` to ``transpose``. The old name raises ``FutureWarning``.
- 2D boundary contraction, environment, and norm methods: rename ``mode`` to ``method``, with ``FutureWarning`` for the old name. With ``mode="full-bond"``, a ``method`` supplied as well is taken as ``similarity_method``.
- [`MatrixProductState.compute_local_expectation`](#MatrixProductState.compute_local_expectation): rename ``method`` to ``route``, with ``FutureWarning`` for the old name.
- Rename ``MatrixProductState.partial_trace_to_dense_canonical`` to [`partial_trace_canonical`](#MatrixProductState.partial_trace_canonical). The old name still works but raises a warning.
- [`TN_from_strings`](#TN_from_strings): deprecate ``join_avoid_self_loops`` in favor of ``join_avoid_loop_length``.

### Bug fixes

#### Operators and time evolution

- [`SpinHam1D.build_sparse`](#SpinHam1D.build_sparse): include the last term with ``cyclic=True`` ({issue}`419`).
- [`ikron`](#ikron): raise ``ValueError`` for out-of-range or repeated indices, which previously placed fewer operators than supplied without an error.
- [`build_matrix_ikron`](#SparseOperatorBuilder.build_matrix_ikron): fix incorrect matrices with non-identity ordering or non-integer site labels.
- [`HilbertSpace`](#HilbertSpace): prevent incorrect rank-to-configuration mappings after reordering, including invalid configurations for mixed dimensions. Ordering is now immutable, as described above.
- [`TEBD`](#TEBD): fix ``order=4`` to achieve fourth-order accuracy. Correct Suzuki weights include backward evolution during part of each step. Also accept ``order=1``.
- [`build_mpo_propagator_trotterized`](#LocalHam1D.build_mpo_propagator_trotterized): fix cyclic wrapping terms that are not symmetric under a site swap.
- [`FullUpdate`](#FullUpdate): reuse the cached plaquette map in ``compute_energy``.

#### Decompositions and compression

- [`compute_oblique_projectors`](#compute_oblique_projectors): avoid ``inf`` or ``nan`` projectors for rank-deficient environments with ``cutoff=0.0``. The same fix applies to shared diagonal division helpers and [`D2BP.gauge_insert`](#D2BP.gauge_insert) with ``return_gauges="inverse"``.
- [`safe_inverse`](#safe_inverse): fix ``TypeError`` in projector boundary contractions with block-sparse backends such as ``symmray``.
- [`tensor_split`](#tensor_split): fix ``method="svd:rand"`` for complex and single-precision arrays, preserving dtype and device, and ``method="svd:eig"`` with nonzero ``cutoff`` for single precision and Numba.
- Oblique projector compression, including [`L2BP.compress`](#L2BP.compress), CTMRG, and HOTRG: fix ``shape-mismatch`` errors with ``symmray`` block-sparse arrays.
- 3D boundary contraction: remove temporary site tags from replaced tensors, including those shared with cached environments.
- 1D ``fit`` with ``max_iterations=1``: build any initial guess at ``max_bond`` rather than padding a smaller one, and fix a ``TypeError`` when only ``cutoff`` is given.
- 1D ``srcmps``: conjugate the sampling MPS, so a supplied guess helps for complex arrays, and odd-parity fermionic networks work.

#### Fermionic tensors

- 1D compression: fix ``sdc``, ``sdc-oversample``, and ``fit`` for fermionic networks, and ``dm`` for mixed bond orientations and truncated odd-parity norm networks.
- Projector compression (``method="projector"``): fix zero projectors for odd-parity fermionic norm networks with every ``canonize`` option.
- [`D2BP.compress`](#D2BP.compress) and [`D2BP.gauge_symmetric`](#D2BP.gauge_symmetric): preserve positive messages and full-rank identity matrices for fermionic networks.
- [`TensorNetwork.conj`](#TensorNetwork.conj) and [`Tensor.conj`](#Tensor.conj): add ``output_inds`` to select output legs for fermionic conjugation phases, including subnetworks. Fix conjugation of paired and unpaired odd-parity dummy modes.

#### Belief propagation and overlaps

- [`D1BP.contract_loop_series_expansion`](#D1BP.contract_loop_series_expansion) and [`D2BP.contract_loop_series_expansion`](#D2BP.contract_loop_series_expansion): include all distinct loops in each region and correct suppression factors. Multi-excitation corrections raise ``RuntimeError`` if they fail to converge.
- [`D2BP.normalize_tensors`](#D2BP.normalize_tensors) and single-site [`D2BP.gate_`](#D2BP.gate_): keep contractions consistent after updates. Reject distinct supplied networks, and fix message ordering for complex two-site gates.
- [`D2BP.gauge_temp`](#D2BP.gauge_temp): restore gauges when its body raises. Fix relative ``smudge`` scaling for unsorted symmetry-block spectra.
- [`tensor_network_distance`](#tensor_network_distance): remove incorrect dependence of infidelity on relative phase.

#### Partial traces and general tensor operations

- [`partial_trace_exact`](#TensorNetworkGenVector.partial_trace_exact): fix ``AttributeError`` with ``get="tensor"`` and ``normalized=True``.
- Exact, cluster, and arbitrary-geometry partial traces and local expectations accept a single site as ``where`` instead of raising ``TypeError``.
- [`local_expectation_cluster`](#TensorNetworkGenVector.local_expectation_cluster): support ``max_bond`` on MPS.
- 2D [`compute_local_expectation`](#TensorNetwork2DVector.compute_local_expectation): accept reversed two-site terms without ``KeyError`` in the boundary route.
- [`compute_local_expectation_exact`](#TensorNetworkGenVector.compute_local_expectation_exact) and [`compute_local_expectation_cluster`](#TensorNetworkGenVector.compute_local_expectation_cluster): correctly sum normalized expectations with ``normalized="return"`` and ``return_all=False``. Methods that compute terms individually also complete their progress bars.
- [`MPS_product_state`](#MPS_product_state): fix ``TypeError`` for backends requiring a single shape tuple when reshaping single-site vectors.
- [`TensorNetwork.tids_are_connected`](#TensorNetwork.tids_are_connected): make results independent of tensor order.


## v1.15.0 (2026-08-10)

### Breaking changes (1.15.0)

**Circuits**

- [`Circuit.uni`](#Circuit.uni): return the non-transposed unitary, so ``circ.uni.to_dense()`` gives ``U`` acting as ``U @ psi``. Use ``circ.get_uni(transposed=True)`` for the old convention.
- Non-exact circuit simulators now inherit [`CircuitBase`](#CircuitBase) rather than [`Circuit`](#Circuit). Use ``isinstance(circ, CircuitBase)`` for any circuit type. Exact-only methods such as ``get_uni``, ``get_psi_simplified``, and ``sample_gate_by_gate`` are no longer available on these simulators.
- MPS circuit methods operate directly on the current, possibly compressed state and no longer accept ``simplify_*`` options. Unsupported representation-specific methods raise ``NotImplementedError`` consistently.

**Gauging and loop expansions**

- [`TensorNetwork.gauge_all_simple`](#TensorNetwork.gauge_all_simple): arguments after ``max_iterations`` and ``tol``, including ``smudge``, ``power``, and ``damping``, are keyword-only.
- Simple-update gauging: ``smudge`` and ``gauge_smudge`` are relative to the largest gauge value. Results change for unnormalized gauges, but remain unchanged when ``max(g) == 1``.
- Automatic generalized-loop sizing (``max_size=None`` in [`tn.gen_gloops`](#TensorNetwork.gen_gloops), or ``gloops=None`` in loop expansions) covers every target tensor or site that belongs to a loop. Targets outside loops are ignored, with ``UserWarning`` for explicit targets or trees with no loops. Loops larger than the resolved size are excluded. Use ``max_size="min"`` or ``gloops="min"`` for the old behavior.

### Enhancements (1.15.0)

#### Circuit simulators and OpenQASM

- Add [`CircuitPEPSSimpleUpdate`](#CircuitPEPSSimpleUpdate) for arbitrary-geometry PEPS circuits with nearest-neighbor simple-update gates and cluster expectations. Define geometry with ``edges`` or infer it from ``gates`` or ``psi0``; control accuracy with ``max_bond`` and gauge equilibration with ``equilibrate``.
- Add [`CircuitPEPOSimpleUpdate`](#CircuitPEPOSimpleUpdate), a Heisenberg-picture simulator using backward PEPO evolution with simple-update compression and reverse-light-cone filtering. Use ``get_evolved_operator`` for the evolved observable or ``local_expectation`` for its ``|00...0>`` expectation.
- Add [`CircuitBase`](#CircuitBase), the shared base for all circuit simulators. Its ``from_*`` constructors accept representation-specific options.
- Add [`CircuitMPSLazy`](#CircuitMPSLazy) for deferred gate evaluation and periodic compression, which can be more efficient for long-range gates with ``src`` compression. MPS simulators also gain ``compute_marginal`` and ``sample_chaotic``.
- Add OpenQASM 3 parsing through [`from_openqasm3_str`](#CircuitBase.from_openqasm3_str), [`from_openqasm3_file`](#CircuitBase.from_openqasm3_file), and [`from_openqasm3_url`](#CircuitBase.from_openqasm3_url), supporting custom gates, register broadcasting, and symbolic input tracking.
- [`CircuitDense`](#CircuitDense): support controlled gates through ``controls=`` without building the full dense operator.

#### Infinite 2D tensor networks

- Add ``quimb.tensor.tn2dinf`` for infinite-PEPS ground states: [`GeometryInfinite2D`](#GeometryInfinite2D) defines the unit cell, [`PEPSInfinite2D`](#PEPSInfinite2D) stores the state, [`LocalHamInfinite2D`](#LocalHamInfinite2D) defines the Hamiltonian, and [`SimpleUpdateInfinite2D`](#SimpleUpdateInfinite2D) performs imaginary-time simple update. Support dense, Abelian-symmetric, and fermionic ``symmray`` backends, with cluster (``max_distance``) or generalized-loop (``gloops``) expectations. Hamiltonians can include longer-range terms than the state geometry.
- Add [`GeometryInfinite2D.square`](#GeometryInfinite2D.square) for square-lattice unit cells. Set neighbor range with ``couplings`` (neighbor shells or displacement vectors) or ``radius``.

#### Gates, gauges, and lattice bonds

- [`tensor_network_ag_gate_simple`](#tensor_network_ag_gate_simple): support long-range gates between sites without a direct bond through [`tensor_network_ag_gate_simple_long_range`](#tensor_network_ag_gate_simple_long_range).
- [`TensorNetwork.gauge_all_simple`](#TensorNetwork.gauge_all_simple): add ``fuse_multibonds`` to preserve multi-index bonds during gauging. [`tensor_compress_bond`](#tensor_compress_bond) supports explicit bond-index selection; [`SimpleUpdateGen`](#SimpleUpdateGen) preserves multi-index bonds by default.
- Add [`tensor_gauge_simple_bond`](#tensor_gauge_simple_bond) to gauge a single bond using a shared ``gauges`` dictionary.
- [`tensor_canonize_bond`](#tensor_canonize_bond): add ``swap_inds`` to move indices between tensors during canonization and ``bond_ind`` for explicit bond selection.
- Add [`LatticeBondMap`](#LatticeBondMap) for consistent lattice bond indices across open and periodic boundaries.
- [`LocalHamGen.get_auto_ordering`](#LocalHamGen.get_auto_ordering): return commuting layers with ``group=True`` for ``sort``, ``random``, and ``None``. ``order="random-ungrouped"`` groups shuffled terms while preserving their order.

#### Belief propagation: gauging and loop expansions

- Add [`gauge_d2bp`](#gauge_d2bp), [`TensorNetwork.gauge_all_belief_propagation`](#TensorNetwork.gauge_all_belief_propagation), and [`D2BP.gauge_symmetric`](#D2BP.gauge_symmetric) for symmetric gauging with dense 2-norm belief propagation.
- [`D2BP`](#D2BP): add ``power``, relative ``smudge`` conditioning, and [`converge_d2bp`](#converge_d2bp). Support BP messages as projector environments, including ``canonize="bp"`` in [`tensor_network_ag_compress_projector`](#tensor_network_ag_compress_projector).
- [`TensorNetworkGenVector.norm_gloop_expand`](#TensorNetworkGenVector.norm_gloop_expand): add an ``info`` cache, shared with [`compute_local_expectation_gloop_expand`](#TensorNetworkGenVector.compute_local_expectation_gloop_expand) for ``normalized="global"``. Supplied ``gauges`` are copied rather than modified. [`combine_local_contractions`](#combine_local_contractions) gains ``power``.
- [`D1BP.contract_gloop_expand`](#D1BP.contract_gloop_expand): include singleton regions and optionally remove dangling tensors without modifying the target network.

#### Norms, decompositions, and backend conversion

- [`TensorNetwork.norm`](#TensorNetwork.norm), [`norm_gloop_expand`](#TensorNetworkGenVector.norm_gloop_expand), and [`normalize_simple`](#TensorNetworkGen.normalize_simple) accept ``strip_exponent`` to return the norm or normalization factor as separate mantissa and base-10 exponent values.
- [`eigh_truncated`](#eigh_truncated): add ``shift`` for optional diagonal regularization.
- Add [`Tensor.to`](#Tensor.to) and [`TensorNetwork.to`](#TensorNetwork.to) to change backend, dtype, or device (autoray v0.9.0 or newer).
- [`trace_distance`](#trace_distance): speed up Hermitian trace distances using eigenvalues. Use ``isherm=False`` for the previous singular-value calculation.

### Bug fixes (1.15.0)

#### Circuits

- [`CircuitDense`](#CircuitDense): fix ``ValueError`` in ``psi``, ``partial_trace``, and ``local_expectation``. ``get_uni`` raises the same clear ``ValueError`` as ``uni`` because the dense state cannot provide the unitary.
- [`CircuitPermMPS`](#CircuitPermMPS): return ``amplitude``, ``to_dense``, and ``local_expectation`` results in logical qubit order after lazy permutations.
- [`CircuitPermMPS`](#CircuitPermMPS) and [`CircuitMPSLazy`](#CircuitMPSLazy): preserve subclass attributes in ``copy()`` ({issue}`387`).
- [`CircuitMPSLazy`](#CircuitMPSLazy): apply pending gates and configured compression before state accessors run. Results no longer depend on accessor order ({issue}`387`).
- [`CircuitMPS`](#CircuitMPS) and subclasses: raise ``NotImplementedError`` for unsupported ``schrodinger_contract`` instead of an internal ``IndexError`` ({issue}`387`).
- [`Circuit.get_rdm_lightcone_simplified`](#Circuit.get_rdm_lightcone_simplified): prevent ``partial_trace`` and ``local_expectation`` from returning results cached before later gates were applied ({issue}`398`).

#### Belief propagation: messages and loop corrections

- [`TensorNetwork.gen_gloops`](#TensorNetwork.gen_gloops): fix automatic sizing and joins for dangling target tensors. Add ``allow_dangling``, on by default for local expectations and partial traces. ``"alldangle"`` and ``"anydangle"`` remain aliases.
- [`D1BP.contract_loop_series_expansion`](#D1BP.contract_loop_series_expansion) and [`HD1BP.contract_gloop_expand`](#HD1BP.contract_gloop_expand): generate loops with ``gloops=None`` instead of raising ``TypeError``.
- [`D2BP.partial_trace_loop_series_expansion`](#D2BP.partial_trace_loop_series_expansion): fix incorrect reduced density matrices for complex states ({issue}`380`).
- [`D2BP.normalize_tensors`](#D2BP.normalize_tensors): prevent repeated reduced-density-matrix calculations from drifting after rescaling ({issue}`381`).
- [`D2BP.compress`](#D2BP.compress): make later message updates use the compressed tensor data after in-place compression.

#### Compression, lattice boundaries, and norms

- [`tensor_network_1d_compress_src`](#tensor_network_1d_compress_src) and [`tensor_network_1d_compress_srcmps`](#tensor_network_1d_compress_srcmps): fix compression of networks with long-range bonds that skip sites.
- [`enforce_1d_like`](#enforce_1d_like): connect long-range bonds correctly when ``site_tags`` reverse their site order, including ``sweep_reverse=True``.
- [`PEPS`](#PEPS), [`PEPO`](#PEPO), and [`PEPS3D`](#PEPS3D): keep open and periodic bonds distinct for cyclic dimensions of length 1 or 2, including bond dimension 1.
- [`TensorNetwork3D.contract_peps_sweep`](#TensorNetwork3D.contract_peps_sweep): preserve norm exponents and support separate mantissa/exponent returns with ``strip_exponent=True``.
- [`TN_from_strings`](#TN_from_strings) and random hidden-loop networks: fix normalization when the tensor-network exponent is nonzero.
- [`TensorNetwork2DVector.compute_norm`](#TensorNetwork2DVector.compute_norm): always return a scalar.
- [`TensorNetworkGenVector.norm_gloop_expand`](#TensorNetworkGenVector.norm_gloop_expand): include the tensor-network exponent.
- [`TensorNetwork.isel`](#TensorNetwork.isel): support slices.

### Internal (1.15.0)

- Convert ``quimb.tensor.circuit`` from a module to a package, preserving public imports and existing pickles.

### Documentation (1.15.0)

- Sphinx AutoAPI documents each object only where it is defined. Support short suffix links such as [`Tensor`](#Tensor).


## v1.14.0 (2026-05-10)

**Breaking Changes**

- [`tensor_compress_bond`](#tensor_compress_bond): rename input tensor args `ta` and `tb`


**Enhancements:**

- [`D2BP`](#D2BP): support fermionic tensor networks (only computing norm^2 so far, gating/compression need work).
- [`tensor_compress_bond`](#tensor_compress_bond): add `reduce_opts` for controlling the decomposition options used when reducing each tensor before the main truncating decomposition. For example ``reduce_opts={"method": "qr:cholesky"}``.
- add [`TensorNetworkGen.select_sites`](#TensorNetworkGen.select_sites) as a convenience method for selecting a sub network given a list of sites.
- add [`PEPO_product_operator`](#PEPO_product_operator) for bond-dimension-1 PEPOs given by a product of on-site operators, including cyclic boundary conditions via ``cyclic=True`` or ``cyclic=(cyclic_x, cyclic_y)``.
- [`PEPO`](#PEPO): accept explicit ``cyclic`` kwarg in the constructor, to override shape-based boundary-condition inference (required for bond dimension 1 cyclic PEPOs).
- [`TensorNetworkGenOperator`](#TensorNetworkGenOperator): add generic [`apply`](#TensorNetworkGenOperator.apply) (dispatching on operator/vector tensor networks), [`trace`](#TensorNetworkGenOperator.trace) and [`partial_transpose`](#TensorNetworkGenOperator.partial_transpose) methods. These now work for arbitrary geometry operator tensor networks (including MPO and PEPO); ``partial_transpose`` supports arbitrary hashable site labels. ``apply`` also gains an ``inplace`` option that propagates to the *acting* operator rather than the one being acted on.
- add [`TensorNetworkGen.has_site`](#TensorNetworkGen.has_site) to test whether an object is a valid site label of a tensor network. The generic implementation checks membership in the site set; 1D, 2D and 3D tensor networks override it with a fast bounds check.
- add [`LocalHam2D.build_pepo_propagator_trotterized`](#LocalHam2D.build_pepo_propagator_trotterized) for a first-order Trotter decomposition of ``expm(x H)`` as a PEPO. Accepts an `ordering` argument to control the order in which terms are applied.
- [`TensorNetwork.split_simplify`](#TensorNetwork.split_simplify): consider all candidate bipartitions for each tensor and accept the one that minimizes the resulting maximum tensor size, rather than the first reduction found.
- [`contract_hotrg`](#TensorNetwork2D.contract_hotrg), [`coarse_grain_hotrg`](#TensorNetwork2D.coarse_grain_hotrg), their 3D counterparts, and [`tensor_network_ag_compress_projector`](#tensor_network_ag_compress_projector): add `gauge_power` parameter to control the power applied to the bond gauge weights when `canonize=True` before computing the compressed projectors.
- [`RegionGraph`](#RegionGraph): add `get_maximal_regions`, `get_minimal_regions`, and `get_maximal_ancestors` helpers for querying the region hierarchy.

Drawing and schematic updates:

- [`Drawing`](#Drawing): add orthographic projection mode alongside the existing axonometric projection via the new `projection` parameter (replaces `a`/`b`). Named presets include `"orthographic"`, `"axonometric"`, and `"isometric"`.
- [`Drawing.translate`](#Drawing.translate): new context manager to temporarily offset all draw operations in coordinate space (before projection).
- [`Drawing.translate_screen`](#Drawing.translate_screen): new context manager to temporarily offset all draw operations in screen space (after projection).
- [`Drawing.grid3d`](#Drawing.grid3d): automatically select back-facing planes based on projection so grids always appear behind the scene, use readable tick label orientations for all projections, and place axis labels correctly.


**Bug fixes:**

- [`CircuitPermMPS.sample`](#CircuitPermMPS.sample): fix output bitstring ordering when the internal MPS qubit order is permuted ({issue}`327`).
- [`TensorNetwork.split_tensor`](#TensorNetwork.split_tensor): fix handling of `absorb=None`, adding all tensors returned by the split ({issue}`260`).
- [`D2BP.gate_`](#D2BP.gate_): correctly mark touched tensors and rebuild local contraction expressions after applying gates.
- [`contract_hotrg`](#TensorNetwork2D.contract_hotrg) and 3D counterpart: fix bug when specifying `strip_exponent` in `final_contract_opts`.


(whats-new-1-13-0)=
## v1.13.0 (2026-03-19)

**Breaking Changes**

- [`ham_hubbard_hardcore`](#ham_hubbard_hardcore) fix description and sign convention of hopping strength `t`.
- [`heisenberg_from_edges`](#heisenberg_from_edges) fix sign convention of magnetic field terms.
- the [`quimb.tensor`](#tensor) submodule structure has been refactored with [`tn1d`](#tn1d), [`tn2d`](#tn2d), [`tn3d`](#tn3d), and [`tnag`](#tnag) submodules for better organization. Imports from old locations will still work, but are deprecated. Public classes and functions such as [`MatrixProductState`](#MatrixProductState) are directly accessible from the top level `quimb.tensor` module as before.

**Enhancements:**

Major updates to splitting/decomposing individual tensors/arrays:

- add [`array_split`](#array_split) and [`array_svals`](#array_svals) as the primary array-level entry points for matrix decomposition, consolidating dispatch logic that was previously internal to `tensor_core`.
- add [`register_split_driver`](#register_split_driver) and [`register_svals_driver`](#register_svals_driver) decorators for registering custom matrix decomposition methods with `array_split` and `array_svals`.
- allow [`array_split`](#array_split) to handle *batches* of matrices (for most methods).
- [`array_split`](#array_split): automatically detect and forward valid kwargs to underlying decomposition methods.
- [`tensor_split`](#tensor_split) and [`array_split`](#array_split): expand `absorb` options significantly beyond `"left"`, `"both"`, `"right"`, `None` to include `"lorthog"`, `"rorthog"`, `"lfactor"`, `"rfactor"`, `"lsqrt"`, `"rsqrt` and `"s"` for returning partial results (single factors or singular values only). Default changed from `"both"` to `"auto"`, which uses each method's natural default.
- add method `"svd:eig"` with main implementation [`svd_via_eig`](#svd_via_eig) for efficient SVD via hermitian eigen-decomposition, with shortcuts for all absorb modes. This can be faster (especially e.g. on GPU) than the standard SVD, but entails some loss of precision.
- [`tensor_split`](#tensor_split): rename `method` option `"eig"` to `"svd:eig"` to make it clearer that this is an SVD split via eigen-decomposition. `"eig"` remains as a deprecated alias for `"svd:eig"`.
- add method `"svd:rand"` with main implementation [`svd_rand_truncated`](#svd_rand_truncated) for randomized SVD with truncation, with shortcuts for all absorb modes. (This is a new and backend agnostic implementation as opposed to the existing `'rsvd'` method).
- add method `"qr:cholesky"` [`qr_via_cholesky`](#qr_via_cholesky) for efficient QR or LQ like decompositions via cholesky decomposition, with shortcuts for all absorb modes. This can be faster than the standard QR (especially on GPU) but entails some loss of precision.
- [`tensor_split`](#tensor_split) and [`array_split`](#array_split): add `"lsqrt"` and `"rsqrt"` absorb options, update cholesky decomposition to [`cholesky_regularized`](#cholesky_regularized) with `shift` as exposed parameter.
- [`compute_oblique_projectors`](#compute_oblique_projectors): allow `method` kwarg.
- QR decomposition: add `stabilize` kwarg for controlling QR stabilization behavior.
- decomposition methods: various compatibility improvements for JAX backend.

Other enhancements:

- add [`shift`](#shift) and [`clock`](#clock) operators.
- add [`Tensor.isfermionic`](#Tensor.isfermionic) and [`TensorNetwork.isfermionic`](#TensorNetwork.isfermionic) methods.
- add [`Tensor.isblocksparse`](#Tensor.isblocksparse) and [`TensorNetwork.isblocksparse`](#TensorNetwork.isblocksparse) methods.
- add `phase_dual` option to [`TensorNetwork.conj`](#TensorNetwork.conj).
- rename `tensor_network_1d_compress_zipup_first` to [`tensor_network_1d_compress_zipup_oversample`](#tensor_network_1d_compress_zipup_oversample) and standardise `oversample` arguments.
- add [`tensor_network_1d_compress_srcmps_oversample`](#tensor_network_1d_compress_srcmps_oversample) and [`tensor_network_1d_compress_fit_oversample`](#tensor_network_1d_compress_fit_oversample) methods.
- add [`connected_bipartitions`](#quimb.tensor.networking.connected_bipartitions) for finding all connected bipartitions of a tensor network
- [`tn.distribute_exponent`](#TensorNetwork.distribute_exponent): add `new_exponent` option for specifying the new exponent value (default 0.0).
- [`tensor_network_1d_compress`](#tensor_network_1d_compress): correctly handle input networks with non-zero exponents and `equalize_norms`.
- add [`tensor_network_gate_sandwich_inds`](#tensor_network_gate_sandwich_inds) for applying a gate and its conjugate like $G A G^\dagger$ to a tensor network.
- [`tensor_network_ag_gate`](#tensor_network_ag_gate): add `which="sandwich"` option for applying a gate and its conjugate like $G A G^\dagger$ to a tensor network, default to this if the supplied tensor network is a [`TensorNetworkGenOperator`](#TensorNetworkGenOperator).
- add function [`tensor_network_ag_gate_simple`](#tensor_network_ag_gate_simple) for applying a gate to an arbitrary geometry tensor network vector or operator, using simple update style `gauges` to perform any compression.
- [`insert_compressor_between_regions`](#TensorNetwork.insert_compressor_between_regions) and upstream CTMRG/HOTRG methods: add explicit `contract_opts`, `reduce_opts`, and `compress_opts` keyword arguments for fine-grained control.
- [`TensorNetwork2D.contract_boundary`](#TensorNetwork2D.contract_boundary), [`contract_ctmrg`](#TensorNetwork2D.contract_ctmrg), [`contract_hotrg`](#TensorNetwork2D.contract_hotrg), [`coarse_grain_hotrg`](#TensorNetwork2D.coarse_grain_hotrg) and their 3D counterparts: add `strip_exponent` parameter and `equalize_norms="auto"` default.
- [`TensorNetwork3D.contract_hotrg`](#TensorNetwork3D.contract_hotrg): use updated projecting/gauging scheme.
- all compression methods: accept an explicit `compress_opts` kwarg.
- [`tensor_network_ag_compress`](#tensor_network_ag_compress): allow fine-grained control over split options via `compress_opts`.
- [`TensorNetworkGen.flatten`](#TensorNetworkGen.flatten): add arbitrary geometry flatten method, used in 1D/2D/3D.
- [`RegionGraph`](#RegionGraph): various improvements.
- add [`hash_kwargs_to_int`](#hash_kwargs_to_int) utility for hashing keyword arguments to a deterministic integer.

**Bug fixes:**

- fix [`isometrize_qr`](#isometrize_qr) for complex torch arrays ({issue}`346`).
- fix [`right_canonicalize`](#TensorNetwork1DFlat.right_canonicalize) to return the right canonicalized tensor network ({issue}`347`)
- ensure all belief propagation contraction methods correctly propagate the target tensor network's `.exponent`.
- fix cutoff mode bug in [`array_split`](#array_split) decomposition truncation.
- fix [`tensor_network_1d_compress_zipup`](#tensor_network_1d_compress_zipup) `equalize_norms` exponent accumulation.
- fix `final_contract_opts` inplace handling in boundary contraction methods.
- fix [`squared_op_to_reduced_factor`](#squared_op_to_reduced_factor) argument handling.
- fix cholesky decomposition `shift` kwarg forwarding and `absorb="right"` direction.
- fix [`sample_hd1bp`](#sample_hd1bp) sub-progress bar display.
- fix gate tag propagation in [`tensor_network_gate_inds`](#tensor_network_gate_inds).
- handle `equalize_norms` correctly in [TensorNetwork2D.compute_environments](#TensorNetwork2D.compute_environments) ({issue}`352`).


(whats-new-1-12-1)=
## v1.12.1 (2026-01-12)

**Breaking Changes**

- bump minimum required python version to 3.11


**Bug fixes:**

- fix [`SimpleUpdateGen`](#SimpleUpdateGen) mixin inheritance order.
- fix [`insert_compressor_between_regions`](#TensorNetwork.insert_compressor_between_regions) for fermionic tensor networks with bond signature +-.


(whats-new-1-12-0)=
## v1.12.0 (2026-01-09)

**Enhancements:**

- move the experimental `operatorbuilder` module to the main [`quimb.operator`](#operator) module.
- add basic introduction to the operator module - {ref}`operator-basics`
- add new example on tracing tensor network functions {ref}`ex_tracing_tn_functions`
- [`tensor_split`](#tensor_split): add an `info` kwarg, supplying this with an empty dict or with the entry `'error'` will store the truncation error when using `method in {"svd", "svd:eig"}`.
- update infrastructure for TEBD and SimpleUpdate based algorithms.
- [`schematic.Drawing`](#Drawing): add [`grid`](#Drawing.grid), [`grid3d`](#Drawing.grid3d), [`bezier`](#Drawing.bezier), [`star`](#Drawing.star), [`cross`](#Drawing.cross) and [`zigzag`](#Drawing.zigzag) methods.
- [`schematic.Drawing`](#Drawing): add `relative` option to [`arrowhead`](#Drawing.arrowhead), `shorten` option to [`text_between`](#Drawing.text_between) and `text_left` and `text_right` options to [`line`](#Drawing.line).
- add [`Drawing.scale_figsize`](#Drawing.scale_figsize) for automatically setting the absolute figsize based on placed elements.
- refactor [`TEBDGen`](#TEBDGen) and [`SimpleUpdateGen`](#SimpleUpdateGen)
- update the 2d specific [`SimpleUpdate`](#SimpleUpdate) to use the new infrastructure.
- [`tn.draw()`](#draw_tn): show abelian signature if using `symmray` arrays.
- [`tn.draw()`](#draw_tn): add `adjust_lims` option
- [`TNOptimizer`](#TNOptimizer): allow `autodiff_backend="torch"` with `jit_fn=True` to work with array backends with general pytree parameters, e.g. `symmray` arrays.
- [`tn.gen_gloops`](#TensorNetwork.gen_gloops) and [`tn.gen_gloops_sites`](#TensorNetworkGen.gen_gloops_sites): add `join_overlap` option. When building cluster by joining smaller generalized loops, this option controls how many nodes they need to overlap by to be joined together.
- all message passing routines: add `callback` option
- GBP: allow a message initilization function.
- [`D1BP`](#D1BP): allow `messages` to be a callable initialization function.
- [`MatrixProductState.gate_nonlocal`](#MatrixProductState.gate_nonlocal): add `method="lazy"` option for lazily applying a non-local gate as a sub-MPO without contraction or compression.
- [`LocalHamGen.apply_to_arrays`](#LocalHamGen.apply_to_arrays): support pytree parameter arrays such as `symmray`.
- add [`Tensor.get_namespace`](#Tensor.get_namespace) and [`TensorNetwork.get_namespace`](#TensorNetwork.get_namespace) for getting a [reusable data array namespace](https://autoray.readthedocs.io/en/latest/automatic_dispatch.html#namespace-api)
- [`TensorNetwork.isel`](#TensorNetwork.isel): use `take` where possible to better support e.g. `torch.vmap` across amplitudes.
- [`MatrixProductState.measure`](#MatrixProductState.measure), and [`MatrixProductState.sample`](#MatrixProductState.sample): add `backend_random` option for specifying which backend to use for random number generation when sampling, this can be set for example to `jax` to make the whole process jittable, but by default is `numpy`, regardless of the actual array backend.

**Bug fixes:**

- fix [`insert_compressor_between_regions`](#TensorNetwork.insert_compressor_between_regions) when `insert_into is None`.
- tensor network drawing, ensure hyper indices can be specified as `output_inds`.
- fix [`MatrixProductState.measure`](#MatrixProductState.measure) when using jax arrays ({issue}`340`).
- fix [`MatrixProductState.measure`](#MatrixProductState.measure) when projecting and keeping a site site ({issue}`344`).

(whats-new-1-11-2)=
## v1.11.2 (2025-07-30)

**Enhancements:**

- Update the introduction to tensor contraction docs
- Improve efficiency of 1D structured contractions when default `optimize` is used, especially for large bond dimension overlaps.

**Bug fixes:**

- fixes for MPS and MPO constructors when L=1, ({issue}`314`)
- tensor splitting with absorb="left" now correctly marks left indices.
- [`tn.isel`](#TensorNetwork.isel): fix bug when value could not be compared to string `"r"`
- truncated svd, make n_chi comparison more robust to different backends


(whats-new-1-11-1)=
## v1.11.1 (2025-06-20)

**Enhancements:**

- add `create_bond` to [`tensor_canonize_bond`](#tensor_canonize_bond) and [`tensor_compress_bond`](#tensor_compress_bond) for optionally creating a new bond between two tensors if they don't already share one. Add as a flag to [`TensorNetwork1DFlat.compress`](#TensorNetwork1DFlat.compress) and related functions ({issue}`294`).
- add [`ensure_bonds_exist`](#TensorNetwork1DFlat.ensure_bonds_exist) for ensuring that all bonds in a 1D flat tensor network exist. Use this in the `permute_arrays` methods and optionally in the `expand_bond_dimension` method.
- [`tn.draw()`](#draw_tn): permit empty network, and allow `color=True` to automatically color all tags.
- [`tn.add_tag`](#TensorNetwork.add_tag): add a `record: Optional[dict]` kwarg, to allow for easy rewinding of temporary tags without tracking the actual networks.
- add [`qu.plot`](#quimb.utils_plot.plot) as a quick wrapper for calling `matplotlib.pyplot.plot` with the `quimb` style.
- {mod}`quimb.schematic`: add `zorder_delta` kwarg for fine adjustments to layering of objects in approximately the same position.
- [`operatorbuilder`](#operator): big performance improvements and fixes for building matrix representations including Z2 symmetry. Add default `symmetry` and `sector` options that can be overridden at build time. Add lazy (slow, matrix free) 'apply' method. Add `pauli_decompose` transformation. Add experimental PEPO builder for nearest neighbor operators. Add unit tests.

**Bug fixes:**

- Fix [`TensorNetwork2D.compute_plaquette_environments`](#TensorNetwork2D.compute_plaquette_environments) for `mode="zipup"` and other boundary contraction methods that use the generic 1D compression algorithms.
- [`parse_openqasm2_str`](#parse_openqasm2_str) allow custom gate names to start with the word `gate` ({issue}`312`).
- [`MatrixProductState.gate_with_mpo`](#MatrixProductState.gate_with_mpo): fix bug to do with inplace argument ({issue}`313`).


(whats-new-1-11-0)=
## v1.11.0 (2025-05-14)

**Breaking Changes**

- move belief propagation to [`quimb.tensor.belief_propagation`](#quimb.tensor.belief_propagation)
- calling [`tn.contract()`](#TensorNetwork.contract) when an non-zero value has been accrued into `tn.exponent` now automatically re-absorbs that exponent.
- binary tensor operations that would previously have errored now will align and broadcast

**Enhancements:**

- [`Tensor`](#Tensor): make binary operations (`+, -, *, /, **`) automatically align and broadcast indices. This would previously error.
- [`MatrixProductState.measure`](#MatrixProductState.measure): add a `seed` kwarg
- belief propagation, implement DIIS (direct inversion in the iterative subspace)
- belief propagation, unify various aspects such as message normalization and distance.
- belief propagation, add a [`plot`](#BeliefPropagationCommon.plot) method.
- belief propagation, add a `contract_every` option.
- HV1BP: vectorize both contraction and message initialization
- add [`qu.plot_multi_series_zoom`](#plot_multi_series_zoom) for plotting multiple series with a zoomed inset, useful for various convergence plots such as BP
- add `info` option to [`tn.gauge_all_simple`](#TensorNetwork.gauge_all_simple) for tracking extra information such as number of iterations and max gauge diffs
- [`Tensor.gate`](#Tensor.gate): add `transposed` option
- [`TensorNetwork.contract`](#TensorNetwork.contract): add `strip_exponent` option for return the mantissa and exponent (log10) separately. Compatible with [`contract_tags`](#TensorNetwork.contract_tags), [`contract_cumulative`](#TensorNetwork.contract_cumulative), [`contract_compressed`](#TensorNetwork.contract_compressed) sub modes.
- [`tensor_split`](#tensor_split): add `matrix_svals` option, if `True` any returned singular values are put into the diagonal of a matrix (by default, `False`, they are returned as a vector).
- add [`Tensor.new_ind_pair_diag`](#Tensor.new_ind_pair_diag) for expanding an existing index into a pair of new indices, such that the diagonal of the new tensor on those indices is the old tensor.
- [`TNOptimizer`](#TNOptimizer): add 'cautious' ADAM
- [`TensorNetwork.pop_tensor`](#TensorNetwork.pop_tensor): allow `tid` or tags to be specified.
- add an example notebook for converting hyper tensor networks to normal tensor networks, for approximate contraction - {ref}`example-htn-to-2d`
- add "SX" and "SXDG" gates to [`Circuit`](#Circuit) ({pull}`277`)
- add "XXPLUSYY" and "XXPLUSYY" gates to [`Circuit`](#Circuit) ({pull}`279`)
- add progress bar to various `Circuit` methods ({pull}`288`)
- [`quimb.operator`](#operator): fix MPO building for congested operators ({issue}`296` and {issue}`301`), allow arbitrary dtype ({issue}`289`). Fix building of sparse and matrix representations for non-translationally symmetric operators and operators with trivial (all identity) terms.

**Bug fixes:**

- fix [`MatrixProductState.measure`](#MatrixProductState.measure) for `cupy` backend arrays ({issue}`276`).
- fix `linalg.expm` dispatch ({issue}`275`)
- fix 'dm' 1d compress method for disconnected subgraphs
- fix docs source lookup in `quimb.tensor` module
- fix raw gate copying in `Circuit` ({issue}`285`)


(whats-new-1-10-0)=
## v1.10.0 (2024-12-18)

**Enhancements:**

- tensor network fitting: add `method="tree"` for when ansatz is a tree - [`tensor_network_fit_tree`](#tensor_network_fit_tree)
- tensor network fitting: fix `method="als"` for complex networks
- tensor network fitting: allow `method="als"` to use a iterative solver suited to much larger tensors, by default a custom conjugate gradient implementation.
- [`tensor_network_distance`](#tensor_network_distance) and fitting: support hyper indices explicitly via `output_inds` kwarg
- add [`tn.make_overlap`](#TensorNetwork.make_overlap) and [`tn.overlap`](#TensorNetwork.overlap) for computing the overlap between two tensor networks, $\langle O |T \rangle$, with explicit handling of outer indices to address hyper networks. Add `output_inds` to [`tn.norm`](#TensorNetwork.norm) and [`tn.make_norm`](#TensorNetwork.make_norm) also, as well as the `squared` kwarg.
- replace all `numba` based paralellism (`prange` and parallel vectorize) with explicit thread pool based parallelism. Should be more reliable and no need to set `NUMBA_NUM_THREADS` anymore. Remove env var `QUIMB_NUMBA_PAR`.
- [`Circuit`](#Circuit): add `dtype` and `convert_eager` options. `dtype` specifies what the computation should be performed in. `convert_eager` specifies whether to apply this (and any `to_backend` calls) as soon as gates are applied (the default for MPS circuit simulation) or just prior to contraction (the default for exact contraction simulation).
- [`tn.full_simplify`](#TensorNetwork.full_simplify): add `check_zero` (by default set of `"auto"`) option which explicitly checks for zero tensor norms when equalizing norms to avoid `log10(norm)` resulting in -inf or nan. Since it creates a data dependency that breaks e.g. `jax` tracing, it is optional.
- [`schematic.Drawing`](#Drawing): add `shorten` kwarg to [line drawing](#Drawing.line) and [curve drawing](#Drawing.curve) and examples to {ref}`schematic`.
- [`TensorNetwork`](#TensorNetwork): add `.backend` and `.dtype_name` properties.


(whats-new-1-9-0)=
## v1.9.0 (2024-11-19)

**Breaking Changes**

- renamed `MatrixProductState.partial_trace` and `MatrixProductState.ptr` to [MatrixProductState.partial_trace_to_mpo](#MatrixProductState.partial_trace_to_mpo) to avoid confusion with other `partial_trace` methods that usually produce a dense matrix.

**Enhancements:**

- add [`Circuit.sample_gate_by_gate`](#Circuit.sample_gate_by_gate) and related methods [`Circuit.reordered_gates_dfs_clustered`](#Circuit.reordered_gates_dfs_clustered) and [`Circuit.get_qubit_distances`](#Circuit.get_qubit_distances) for sampling a circuit using the 'gate by gate' method introduced in https://arxiv.org/abs/2112.08499.
- add [`CircuitBase.draw`](#CircuitBase.draw) for drawing a very simple circuit schematic.
- [`Circuit`](#Circuit): by default turn on `simplify_equalize_norms` and use a `group_size=10` for sampling. This should result in faster and more stable sampling.
- [`Circuit`](#Circuit): use `numpy.random.default_rng` for random number generation.
- add [`qtn.circ_a2a_rand`](#circ_a2a_rand) for generating random all-to-all circuits.
- expose [`qtn.edge_coloring`](#edge_coloring) as top level function and allow layers to be returned grouped.
- add docstring for [`tn.contract_compressed`](#TensorNetwork.contract_compressed) and by default pick up important settings from the supplied contraction path optimizer (`max_bond` and `compress_late`)
- add [`Tensor.rand_reduce`](#Tensor.rand_reduce) for randomly removing a tensor index by contracting a random vector into it. One can also supply the value `"r"` to `isel` selectors to use this.
- add `fit-zipup` and `fit-projector` shorthand methods to the general 1d tensor network compression function
- add [`MatrixProductState.compute_local_expectation`](#MatrixProductState.compute_local_expectation) for computing many local expectations for a MPS at once, to match the interface for this method elsewhere. These can either be computed via canonicalization (`method="canonical"`), or via explicit left and right environment contraction (`method="envs"`)
- specialize [`CircuitMPS.local_expectation`](#CircuitMPS.local_expectation) to make use of the MPS form.
- add [`PEPS.product_state`](#PEPS.product_state) for constructing a PEPS representing a product state.
- add [`PEPS.vacuum`](#PEPS.vacuum) for constructing a PEPS representing the vacuum state $|000\ldots0\rangle$.
- add [`PEPS.zeros`](#PEPS.zeros) for constructing a PEPS whose entries are all zero.
- [`tn.gauge_all_simple`](#TensorNetwork.gauge_all_simple): improve scheduling and add `damping` and `touched_tids` options.
- [`qtn.SimpleUpdateGen`](#SimpleUpdateGen): add gauge difference update checking and `tol` and `equilibrate` settings. Update `.plot()` method. Default to a small `cutoff`.
- add [`psi.sample_configuration_cluster`](#TensorNetworkGenVector.sample_configuration_cluster) for sampling a tensor network using the simple update or cluster style environment approximation.
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
- add MPS sampling: [`MatrixProductState.sample_configuration`](#MatrixProductState.sample_configuration) and [`MatrixProductState.sample`](#MatrixProductState.sample) (generating multiple samples) and use these for [`CircuitMPS.sample`](#CircuitMPS.sample) and [`CircuitPermMPS.sample`](#CircuitPermMPS.sample).
- add basic [`.plot()`](#TEBDSweepMixin.plot) method for SimpleUpdate classes
- add [`edges_1d_chain`](#edges_1d_chain) for generating 1D chain edges
- [operatorbuilder](#operator): better coefficient placement for long range MPO building

---


(whats-new-1-8-2)=
## v1.8.2 (2024-06-12)

**Enhancements:**

- [`TNOptimizer`](#TNOptimizer) can now accept an arbitrary pytree (nested combination of dicts, lists, tuples, etc. with `TensorNetwork`, `Tensor` or raw `array_like` objects as the leaves) as the target object to optimize.
- [`TNOptimizer`](#TNOptimizer) can now directly optimize [`Circuit`](#Circuit) objects, returning a new optimized circuit with updated parameters.
- [`Circuit`](#Circuit): add `.copy()`, `.get_params()` and `.set_params()` interface methods.
- Update generic TN optimizer docs.
- add [`tn.gen_paths_loops`](#TensorNetwork.gen_paths_loops) for generating all loops of indices in a TN.
- add [`tn.gen_inds_connected`](#TensorNetwork.gen_inds_connected) for generating all connected sets of indices in a TN.
- make SVD fallback error catching more generic ({pull}`238`)
- fix some windows + numba CI issues.
- [`approx_spectral_function`](#approx_spectral_function) add plotting and tracking
- add dispatching to various tensor primitives to allow overriding

---


(whats-new-1-8-1)=
## v1.8.1 (2024-05-06)

**Enhancements:**

- [`CircuitMPS`](#CircuitMPS) now supports multi qubit gates, including arbitrary multi-controls (which are treated in a low-rank manner), and faster simulation via better orthogonality center tracking.
- add [`CircuitPermMPS`](#CircuitPermMPS)
- add [`MatrixProductState.gate_nonlocal`](#MatrixProductState.gate_nonlocal) for applying a gate, supplied as a raw matrix, to a non-local and arbitrary number of sites. The kwarg `contract="nonlocal"` can be used to force this method, or the new option `"auto-mps"` will select this method if the gate is non-local ({issue}`230`)
- add [`MatrixProductState.gate_with_mpo`](#MatrixProductState.gate_with_mpo) for applying an MPO to an MPS, and immediately compressing back to MPS form using [`tensor_network_1d_compress`](#tensor_network_1d_compress)
- add [`MatrixProductState.gate_with_submpo`](#MatrixProductState.gate_with_submpo) for applying an MPO acting only of a subset of sites to an MPS
- add [`MatrixProductOperator.from_dense`](#MatrixProductOperator.from_dense) for constructing MPOs from dense matrices, including an only subset of sites
- add [`MatrixProductOperator.fill_empty_sites`](#MatrixProductOperator.fill_empty_sites) for 'completing' an MPO which only has tensors on a subset of sites with (by default) identities
-  [`MatrixProductState`](#MatrixProductState) and [`MatrixProductOperator`](#MatrixProductOperator), now support the ``sites`` kwarg in common constructors, enabling the TN to act on a subset of the full ``L`` sites.
- add [`TensorNetwork.drape_bond_between`](#TensorNetwork.drape_bond_between) for 'draping' an existing bond between two tensors through a third
- add [`Tensor.new_ind_pair_with_identity`](#Tensor.new_ind_pair_with_identity)
- TN2D, TN3D and arbitrary geom classical partition function builders ([`TN_classical_partition_function_from_edges`](#TN_classical_partition_function_from_edges)) now all support `outputs=` kwarg specifying non-marginalized variables
- add simple dense 1-norm belief propagation algorithm [`D1BP`](#D1BP)
- add [`qtn.enforce_1d_like`](#enforce_1d_like) for checking whether a tensor network is 1D-like, including automatically adding strings of identities between non-local bonds, expanding applicability of [`tensor_network_1d_compress`](#tensor_network_1d_compress)
- add [`MatrixProductState.canonicalize`](#TensorNetwork1DFlat.canonicalize) as (by default *non-inplace*) version of `canonize`, to follow the pattern of other tensor network methods. `canonize` is now an alias for `canonicalize_` [note trailing underscore].
- add [`MatrixProductState.left_canonicalize`](#TensorNetwork1DFlat.left_canonicalize) as (by default *non-inplace*) version of `left_canonize`, to follow the pattern of other tensor network methods. `left_canonize` is now an alias for `left_canonicalize_` [note trailing underscore].
- add [`MatrixProductState.right_canonicalize`](#TensorNetwork1DFlat.right_canonicalize) as (by default *non-inplace*) version of `right_canonize`, to follow the pattern of other tensor network methods. `right_canonize` is now an alias for `right_canonicalize_` [note trailing underscore].

**Bug fixes:**

- [`CircuitBase.apply_gate_raw`](#CircuitBase.apply_gate_raw): fix kwarg bug ({pull}`226`)
- fix for retrieving `opt_einsum.PathInfo` for single scalar contraction ({issue}`231`)


---


(whats-new-1-8-0)=
## v1.8.0 (2024-04-10)

**Breaking Changes**

- all singular value renormalization is turned off by default
- [`TensorNetwork.compress_all`](#TensorNetwork.compress_all)
  now defaults to using some local gauging


**Enhancements:**

- add `quimb.tensor.tn1d.compress.py` with functions for compressing generic
  1D tensor networks (with arbitrary local structure) using various methods.
  The methods are:

  - The **'direct'** method: [`tensor_network_1d_compress_direct`](#tensor_network_1d_compress_direct)
  - The **'dm'** (density matrix) method: [`tensor_network_1d_compress_dm`](#tensor_network_1d_compress_dm)
  - The **'zipup'** method: [`tensor_network_1d_compress_zipup`](#tensor_network_1d_compress_zipup)
  - The **'zipup-oversample'** method: [`tensor_network_1d_compress_zipup_oversample`](#tensor_network_1d_compress_zipup_oversample)
  - The 1 and 2 site **'fit'** or sweeping method: [`tensor_network_1d_compress_fit`](#tensor_network_1d_compress_fit)
  - ... and some more niche methods for debugging and testing.

  And can be accessed via the unified function [`tensor_network_1d_compress`](#tensor_network_1d_compress).
  Boundary contraction in 2D can now utilize any of these methods.
- add `quimb.tensor.tnag.compress.py` with functions for compressing
  arbitrary geometry tensor networks using various methods. The methods are:

  - The **'local-early'** method:
    [`tensor_network_ag_compress_local_early`](#tensor_network_ag_compress_local_early)
  - The **'local-late'** method:
    [`tensor_network_ag_compress_local_late`](#tensor_network_ag_compress_local_late)
  - The **'projector'** method:
    [`tensor_network_ag_compress_projector`](#tensor_network_ag_compress_projector)
  - The **'superorthogonal'** method:
    [`tensor_network_ag_compress_superorthogonal`](#tensor_network_ag_compress_superorthogonal)
  - The **'l2bp'** method:
    [`tensor_network_ag_compress_l2bp`](#tensor_network_ag_compress_l2bp)

  And can be accessed via the unified function
  [`tensor_network_ag_compress`](#tensor_network_ag_compress).
  1D compression can also fall back to these methods.
- support PBC in
  [`tn2d.contract_hotrg`](#TensorNetwork2D.contract_hotrg),
  [`tn2d.contract_ctmrg`](#TensorNetwork2D.contract_ctmrg),
  [`tn3d.contract_hotrg`](#TensorNetwork3D.contract_hotrg) and
  the new function
  [`tn3d.contract_ctmrg`](#TensorNetwork3D.contract_ctmrg).
- support PBC in
  [`gen_2d_bonds`](#gen_2d_bonds) and
  [`gen_3d_bonds`](#gen_3d_bonds), with ``cyclic`` kwarg.
- support PBC in
  [`TN2D_rand_hidden_loop`](#TN2D_rand_hidden_loop)
  and
  [`TN3D_rand_hidden_loop`](#TN3D_rand_hidden_loop),
  with ``cyclic`` kwarg.
- support PBC in the various base PEPS and PEPO construction methods.
- add [`tensor_network_apply_op_op`](#tensor_network_apply_op_op)
  for applying 'operator' TNs to 'operator' TNs.
- tweak [`tensor_network_apply_op_vec`](#tensor_network_apply_op_vec)
  for applying 'operator' TNs to 'vector' or 'state' TNs.
- add [`tnvec.gate_with_op_lazy`](#TensorNetworkGenVector.gate_with_op_lazy)
  method for applying 'operator' TNs to 'vector' or 'state' TNs like $x \rightarrow A x$.
- add [`tnop.gate_upper_with_op_lazy`](#TensorNetworkGenOperator.gate_upper_with_op_lazy)
  method for applying 'operator' TNs to the upper indices of 'operator' TNs like $B \rightarrow A B$.
- add [`tnop.gate_lower_with_op_lazy`](#TensorNetworkGenOperator.gate_lower_with_op_lazy)
  method for applying 'operator' TNs to the lower indices of 'operator' TNs like $B \rightarrow B A$.
- add [`tnop.gate_sandwich_with_op_lazy`](#TensorNetworkGenOperator.gate_sandwich_with_op_lazy)
  method for applying 'operator' TNs to the upper and lower indices of 'operator' TNs like $B \rightarrow A B A^\dagger$.
- unify all TN summing routines into
  [`tensor_network_ag_sum](#tensor_network_ag_sum),
  which allows summing any two tensor networks with matching site tags and
  outer indices, replacing specific MPS, MPO, PEPS, PEPO, etc. summing routines.
- add [`rand_symmetric_array`](#rand_symmetric_array),
  [`rand_tensor_symmetric`](#rand_tensor_symmetric)
  [`TN2D_rand_symmetric`](#TN2D_rand_symmetric)
  for generating random symmetric arrays, tensors and 2D tensor networks.

**Bug fixes:**

- fix scipy sparse monkey patch for scipy>=1.13 ({issue}`222`)
- fix autoblock bug where connected sectors were not being merged ({issue}`223`)


---


(whats-new-1-7-3)=
## v1.7.3 (2024-02-08)

**Enhancements:**

- [qu.randn](#randn): support `dist="rademacher"`.
- support `dist` and other `randn` options in various TN builders.

**Bug fixes:**

- restore fallback (to `scipy.linalg.svd` with driver='gesvd') behavior for truncated SVD with numpy backend.


---


(whats-new-1-7-2)=
## v1.7.2 (2024-01-30)

**Enhancements:**

- add `normalized=True` option to [`tensor_network_distance`](#tensor_network_distance) for computing the normalized distance between tensor networks: $2 |A - B| / (|A| + |B|)$, which is useful for convergence checks. [`Tensor.distance_normalized`](#Tensor.distance_normalized) and [`TensorNetwork.distance_normalized`](#TensorNetwork.distance_normalized) added as aliases.
- add [`TensorNetwork.cut_bond`](#TensorNetwork.cut_bond) for cutting a bond index

**Bug fixes:**

- removed import of deprecated `numba.generated_jit` decorator.


---


(whats-new-1-7-1)=
## v1.7.1 (2024-01-30)

**Enhancements:**

- add [`TensorNetwork.visualize_tensors`](#quimb.tensor.drawing.visualize_tensors)
  for visualizing the actual data entries of an entire tensor network.
- add [`ham.build_mpo_propagator_trotterized`](#LocalHam1D.build_mpo_propagator_trotterized)
  for building a trotterized propagator from a local 1D hamiltonian. This
  also includes updates for creating 'empty' tensor networks using
  [`TensorNetwork.new`](#TensorNetwork.new), and
  building up gates from empty tensor networks using
  [`TensorNetwork.gate_inds_with_tn`](#TensorNetwork.gate_inds_with_tn).
- add more options to [`Tensor.expand_ind`](#Tensor.expand_ind)
  and [`Tensor.new_ind`](#Tensor.new_ind): repeat
  tiling mode and random padding mode.
- tensor decomposition: make ``eigh_truncated`` backend agnostic.
- [`tensor_compress_bond`](#tensor_compress_bond): add
  `reduced="left"` and `reduced="right"` modes for when the pair of tensors is
  already in a canonical form.
- add [`qtn.TN2D_embedded_classical_ising_partition_function`](#TN2D_embedded_classical_ising_partition_function) for constructing 2D
  (triangular) tensor networks representing all-to-all classical ising
  partition functions.

**Bug fixes:**

- fix bug in [`kruas_op`](#kraus_op) when operator spanned multiple
  subsystems ({issue}`214`)
- fix bug in [`qr_stabilized`](#qr_stabilized) when the
  diagonal of `R` has significant imaginary parts.
- fix bug in quantum discord computation when the state was diagonal ({issue}`217`)


---


(whats-new-1-7-0)=
## v1.7.0 (2023-12-08)

**Breaking Changes**

- {class}`.Circuit` : remove `target_size` in preparation for
  all contraction specifications to be encapsulated at the contract level (e.g.
  with `cotengra`)
- some TN drawing options (mainly arrow options) have changed due to the
  backend change detailed below.

**Enhancements:**

- [TensorNetwork.draw](#TensorNetwork.draw): use `quimb.schematic`
  for main `backend="matplotlib"` drawing. Enabling:
    1. multi tag coloring for single tensors
    2. arrows and labels on multi-edges
    3. better sizing of tensors using absolute units
    4. neater single tensor drawing, in 2D and 3D
* add [quimb.schematic.Drawing](#Drawing) from experimental
  submodule, add example docs at {ref}`schematic`. Add methods `text_between`,
  `wedge`, `line_offset` and other tweaks for future use by main TN drawing.
- upgrade all contraction to use `cotengra` as the backend
- [`Circuit`](#Circuit) : allow any gate to be controlled by any
  number of qubits.
- [`Circuit`](#Circuit) : support for parsing `openqasm2`
  specifications now with custom and nested gate definitions etc.
- add [`is_cyclic_x`](#TensorNetwork2D.is_cyclic_x),
  [`is_cyclic_y`](#TensorNetwork2D.is_cyclic_y) and
  [`is_cyclic_z`](#TensorNetwork3D.is_cyclic_z) to
  [TensorNetwork2D](#TensorNetwork2D) and
  [TensorNetwork3D](#TensorNetwork3D).
- add [TensorNetwork.compress_all_1d](#TensorNetwork.compress_all_1d)
  for compressing generic tensor networks that you promise have a 1D topology,
  without casting as a [TensorNetwork1D](#TensorNetwork1D).
- add [MatrixProductState.from_fill_fn](#MatrixProductState.from_fill_fn)
  for constructing MPS from a function that fills the tensors.
- add [Tensor.idxmin](#Tensor.idxmin) and
  [Tensor.idxmax](#Tensor.idxmax) for finding the index of the
  minimum/maximum element.
- 2D and 3D classical partition function TN builders: allow output indices.
- [`quimb.tensor.belief_propagation`](#quimb.tensor.belief_propagation):
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

- add OpenQASM 2.0 parsing support: [`CircuitBase.from_openqasm2_file`](#CircuitBase.from_openqasm2_file),
- [`Circuit`](#Circuit): add RXX, RYY, CRX, CRY, CRZ, toffoli, fredkin, givens gates
- truncate TN pretty html reprentation to 100 tensors for performance
- add [`Tensor.sum_reduce`](#Tensor.sum_reduce) and [`Tensor.vector_reduce`](#Tensor.vector_reduce)
- [`contract_compressed`](#TensorNetwork.contract_compressed), default to 'virtual-tree' gauge
- add [`TN_rand_tree`](#TN_rand_tree)
- `experimental.operatorbuilder`: fix parallel and heisenberg builder
- make parametrized gate generation even more robost (ensure matching types
  so e.g. tensorflow can be used)

**Bug fixes:**

- fix gauge size check for some backends

---


(whats-new-1-5-1)=
## v1.5.1 (2023-07-28)

**Enhancements:**

- add {func}`.MPS_COPY`.
- add 'density matrix' and 'zip-up' MPO-MPS algorithms.
- add `drop_tags` option to {func}`.tensor_core.tensor_contract`
- {meth}`.compress_all_simple`, allow cutoff.
- add structure checking debug methods: {meth}`.Tensor.check` and
  {meth}`.TensorNetwork.check`.
- add several direction contraction utility functions: [`get_symbol`](https://cotengra.readthedocs.io/en/latest/autoapi/cotengra/utils/index.html#cotengra.utils.get_symbol),
  {func}`.inds_to_eq` and {func}`.array_contract`.

**Bug fixes:**

- {class}`.Circuit`: use stack for more robust parametrized gate generation
- fix for {meth}`.gate_with_auto_swap` for `i > j`.
- fix bug where calling `tn.norm()` would mangle indices.

---


(whats-new-1-5-0)=
## v1.5.0 (2023-05-03)

**Enhancements**

- refactor 'isometrize' methods including new "cayley", "householder" and
  "torch_householder" methods. See {func}`.decomp.isometrize`.
- add {meth}`.TensorNetwork.compute_reduced_factor`
  and {meth}`.TensorNetwork.insert_compressor_between_regions`
  methos, for some RG style algorithms.
- add the `mode="projector"` option for 2D tensor network contractions
- add HOTRG style coarse graining and contraction in 2D and 3D. See
  {meth}`.TensorNetwork2D.coarse_grain_hotrg`,
  {meth}`.TensorNetwork2D.contract_hotrg`,
  {meth}`.TensorNetwork3D.coarse_grain_hotrg`, and
  {meth}`.TensorNetwork3D.contract_hotrg`,
- add CTMRG style contraction for 2D tensor networks:
  {meth}`.TensorNetwork2D.contract_ctmrg`
- add 2D tensor network 'corner double line' (CDL) builders:
  {func}`.TN2D_corner_double_line`
- update the docs to use the [furo](https://pradyunsg.me/furo/) theme,
  [myst_nb](https://myst-nb.readthedocs.io/en/latest/) for notebooks, and
  several other `sphinx` extensions.
- add the `'adabelief'` optimizer to
  {class}`.TNOptimizer` as well as a quick plotter:
  {meth}`.TNOptimizer.plot`
- add initial 3D plotting methods for tensors networks (
  `TensorNetwork.draw(dim=3, backend='matplotlib3d')` or
  `TensorNetwork.draw(dim=3, backend='plotly')`
  ). The new `backend='plotly'` can also be used for 2D interactive plots.
- Update {func}`.HTN_from_cnf` to handle more
  weighted model counting formats.
- Add {func}`.cnf_file_parse`
- Add {func}`.random_ksat_instance`
- Add {func}`.TN_from_strings`
- Add {func}`.convert_to_2d`
- Add {func}`.TN2D_rand_hidden_loop`
- Add {func}`.convert_to_3d`
- Add {func}`.TN3D_corner_double_line`
- Add {func}`.TN3D_rand_hidden_loop`
- various optimizations for minimizing computational graph size and
  construction time.
- add `'lu'`, `'polar_left'` and `'polar_right'` methods to
  {func}`.tensor_split`.
- add experimental arbitrary hamilotonian MPO building
- {class}`.TensorNetwork`: allow empty constructor
  (i.e. no tensors representing simply the scalar 1)
- {meth}`.TensorNetwork.drop_tags`: allow all tags to
  be dropped
- tweaks to compressed contraction and gauging
- add jax, flax and optax example
- add 3D and interactive plotting of tensors networks with via plotly.
- add pygraphiviz layout options
- add {meth}`.TensorNetwork.combine` for unified
  handling of combining
  tensor networks potentially with structure
- add HTML colored pretty printing of tensor networks for notebooks
- add `quimb.experimental.cluster_update.py`

**Bug fixes:**

- fix {func}`.qr_stabilized` bug for strictly upper
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
  class {class}`.TensorNetworkGen`
- refactor contraction, allowing using cotengra directly
- add {meth}`.Tensor.visualize` for visualizing the
  actual data entries of an arbitrarily high dimensional tensor
- add {class}`.Gate` class for more robust tracking and
  manipulation of gates in quantum {class}`.Circuit`
  simulation
- tweak TN drawing style and layout
- tweak default gauging options of compressed contraction
- add {meth}`.TensorNetwork.compute_hierarchical_grouping`
- add {meth}`.Tensor.as_network`
- add {meth}`.TensorNetwork.inds_size`
- add {meth}`.TensorNetwork.get_hyperinds`
- add {meth}`.TensorNetwork.outer_size`
- improve {func}`.tensor_core.group_inds`
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
- Many changes to {class}`.TNOptimizer`
- Many changes to TN drawing
- Many changes to {class}`.Circuit` simulation
- Many improvements to TN simplification
- Make all tag and index operations deterministic
- Add {func}`.tensor_network_sum`,
  {func}`.tensor_network_distance` and
  {meth}`.TensorNetwork.fit`
- Various memory and performance improvements
- Various graph generators and TN builders

---


(whats-new-1-3-0)=
## v1.3.0 (2020-02-18)

**Enhancements**

- Added time dependent evolutions to {class}`.Evolution` when integrating a pure state - see {ref}`time-dependent-evolution` - as well as supporting `LinearOperator` defined hamiltonians ({pull}`40`).
- Allow the {class}`.Evolution` callback `compute=` to optionally access the Hamiltonian ({pull}`49`).
- Added {meth}`.Tensor.randomize` and {meth}`.TensorNetwork.randomize` to randomize tensor and tensor network entries.
- Automatically squeeze tensor networks when rank-simplifying.
- Add {meth}`.TensorNetwork1DFlat.compress_site` for compressing around single sites of MPS etc.
- Add {func}`.MPS_ghz_state` and {func}`.MPS_w_state` for building bond dimension 2 open boundary MPS reprentations of those states.
- Various changes in conjunction with [autoray](https://github.com/jcmgray/autoray) to improve the agnostic-ness of tensor network operations with respect to the backend array type.
- Add {func}`.tensor_core.new_bond` on top of {meth}`.Tensor.new_ind` and {meth}`.Tensor.expand_ind` for more graph orientated construction of tensor networks, see {ref}`tn-creation-graph-style`.
- Add the {func}`.operators.fsim` gate.
- Make the parallel number generation functions use new `numpy 1.17+` functionality rather than `randomgen` (which can still be used as the underlying bit generator) ({pull}`50`)
- TN: rename `contraction_complexity` to {meth}`.TensorNetwork.contraction_width`.
- TN: update {meth}`.TensorNetwork.rank_simplify`, to handle hyper-edges.
- TN: add {meth}`.TensorNetwork.diagonal_reduce`, to automatically collapse all diagonal tensor axes in a tensor network, introducing hyper edges.
- TN: add {meth}`.TensorNetwork.antidiag_gauge`, to automatically flip all anti-diagonal tensor axes in a tensor network allowing subsequent diagonal reduction.
- TN: add {meth}`.TensorNetwork.column_reduce`, to automatically identify tensor axes with a single non-zero column, allowing the corresponding index to be cut.
- TN: add {meth}`.TensorNetwork.full_simplify`, to iteratively perform all the above simplifications in a specfied order until nothing is left to be done.
- TN: add `num_tensors` and `num_indices` attributes, show `num_indices` in `__repr__`.
- TN: various improvements to the pytorch optimizer ({pull}`34`)
- TN: add some built-in 1D quantum circuit ansatzes:
  {func}`.circ_ansatz_1D_zigzag`,
  {func}`.circ_ansatz_1D_brickwork`, and
  {func}`.circ_ansatz_1D_rand`.
- **TN: add parametrized tensors** {class}`.PTensor` and so trainable, TN based quantum circuits -- see {ref}`example-tn-training-circuits`.

**Bug fixes:**

- Fix consistency of {func}`.fidelity` by making the unsquared version the default for the case when either state is pure, and always return a real number.
- Fix a bug in the 2D system example for when `j != 1.0`
- Add environment variable `QUIMB_NUMBA_PAR` to set whether numba should use automatic parallelization - mainly to fix travis segfaults.
- Make cache import and initilization of `petsc4py` and `slepc4py` more robust.

---


(whats-new-1-2-0)=
## v1.2.0 (2019-06-06)

**Enhancements**

- Added {func}`.kraus_op` for general, noisy quantum operations
- Added {func}`.projector` for constructing projectors from observables
- Added {func}`.calc.measure` for measuring and collapsing quantum states
- Added {func}`.cprint` pretty printing states in computational basis
- Added {func}`.calc.simulate_counts` for simulating computational basis counts
- TN: Add {meth}`.TensorNetwork.rank_simplify`
- TN: Add {meth}`.TensorNetwork.isel`
- TN: Add {meth}`.TensorNetwork.cut_iter`
- TN: Add `'split-gate'` gate mode
- TN: Add {class}`.TNOptimizer` for tensorflow based optimization
  of arbitrary, contstrained tensor networks.
- TN: Add {meth}`.Dense1D.rand`
- TN: Add {func}`.connect` to conveniently set a shared index for tensors
- TN: make many more tensor operations agnostic of the array backend (e.g. numpy, cupy,
  tensorflow, ...)
- TN: allow {func}`.align_TN_1D` to take an MPO as the first argument
- TN: add {meth}`.SpinHam1D.build_sparse`
- TN: add {meth}`.Tensor.unitize` and {meth}`.TensorNetwork.unitize` to impose unitary/isometric constraints on tensors specfied using the `left_inds` kwarg
- Many updates to tensor network quantum circuit
  ({class}`.Circuit`) simulation including:

  - {class}`.CircuitMPS`
  - {class}`.CircuitDense`
  - 49-qubit depth 30 circuit simulation example {ref}`quantum-circuit-example`

- Add `from quimb.gates import *` as shortcut to import `X, Z, CNOT, ...`.

- Add {func}`.U_gate` for parametrized arbitrary single qubit unitary

**Bug fixes:**

- Fix `pkron` for case `len(dims) == len(inds)` ({issue}`17`, {pull}`18`).
- Fix `qarray` printing for older `numpy` versions
- Fix TN quantum circuit bug where Z and X rotations were swapped
- Fix variable bond MPO building ({issue}`22`) and L=2 DMRG
- Fix `norm(X, 'trace')` for non-hermitian matrices
- Add `autoray` as dependency ({issue}`21`)
