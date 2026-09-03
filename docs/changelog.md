# Changelog

Release notes for `quimb`.

## v1.16.0 (unreleased)

**Breaking Changes:**

- [`HilbertSpace`](#HilbertSpace): site ordering is now immutable. ``set_ordering`` raises ``TypeError``. Use the new [`with_ordering`](#HilbertSpace.with_ordering) method to create a space with a different ordering.
- [`fermi_hubbard_from_edges`](#fermi_hubbard_from_edges): the default is now ``order="interleaved"``. This alternates the spins at each coordinate instead of grouping them. The on-site interaction is then register-local, so the MPO bond dimension does not grow with system size. The cost is one extra Jordan-Wigner Z per hopping term and a ~10% slower matrix-vector product. The register layout changes. Rebuild anything keyed by rank or flat configuration. Use ``order="blocked"`` for the old layout.


**Enhancements:**

- 1D tensor-network compression: add successive deterministic compression (``method="sdc"``) and the ``sdc-oversample``, ``sdcr``, and ``sdcr-oversample`` variants. These methods are based on https://arxiv.org/abs/2601.19650. ``sdc`` forms the low-rank left environments with ``method="svd:eig"``, whereas ``sdcr`` uses a cheap randomized SVD.
- [`MatrixProductOperator.gate_sandwich_with_auto_swap`](#MatrixProductOperator.gate_sandwich_with_auto_swap): apply a two-site gate sandwich and keep the MPO in canonical form. The method tracks the orthogonality center and can strip the center tensor's exponent. For long-range gates, it swaps the sites together and then restores their positions.
- 1D compression: ``src``, ``srcmps``, and ``fit`` now create random tensors with ``autoray.random.array`` (autoray v0.10.0 or newer). The tensors match the device and dtype. These methods and their oversampling variants accept a ``seed`` or random generator. By default, they use the backend's global random state.
- 1D compression: ``fit`` with ``bsz=1`` now supports fermionic tensor networks. It warns if the network contains odd-parity tensors because the result is likely incorrect.
- Gating: add a ``dagger`` option to [`tensor_network_gate_inds`](#tensor_network_gate_inds) (``tn.gate_inds``), [`tensor_network_gate_sandwich_inds`](#tensor_network_gate_sandwich_inds) (``tn.gate_sandwich_inds``), [`tensor_network_ag_gate`](#tensor_network_ag_gate) (``tn.gate``, ``tn.gate_sandwich``, ``tn.gate_upper``, and ``tn.gate_lower``), [`tensor_network_ag_gate_simple`](#tensor_network_ag_gate_simple) (``tn.gate_simple``, including its long-range variant), and [`TensorNetworkGenOperator.gate_sandwich_with_op_lazy`](#TensorNetworkGenOperator.gate_sandwich_with_op_lazy). This option applies $G^\dagger$ instead of $G$. For example, it changes $G A G^\dagger$ to $G^\dagger A G$ for Heisenberg evolution. It avoids manually reshaping and conjugate-transposing tensor gates.
- Gating: add a matching ``transpose`` option to the functions above. It applies $G^T$ instead of $G$, without conjugation. [`MatrixProductState.gate_nonlocal`](#MatrixProductState.gate_nonlocal) also accepts this option and passes it to [`gate_with_submpo`](#MatrixProductState.gate_with_submpo).
- [`Tensor.gate`](#Tensor.gate): rename ``transposed`` to ``transpose`` for consistency with ``dagger`` and the operator-network gating methods. The old name still works but raises ``FutureWarning``.
- [`CircuitPEPOSimpleUpdate`](#CircuitPEPOSimpleUpdate): add ``dtype``, ``to_backend``, and ``convert_eager`` options, as supported by [`CircuitPEPSSimpleUpdate`](#CircuitPEPSSimpleUpdate). The operator is built only when an expectation is computed. At that time, conversion applies to the new identity PEPO, the observable, and each gate array used in the backward evolution.
- [`HilbertSpace`](#HilbertSpace): support ``U1U1`` sectors with any ordering. Each species no longer needs a contiguous block of registers. A new ``species`` argument selects the conserved charge for each site. It also allows the short sector forms ``{species: filling}`` and ``(ka, kb)``, in addition to ``((na, ka), (nb, kb))``.
- [`HilbertSpace`](#HilbertSpace): ``order`` accepts two presets. ``"blocked"`` puts each species in one contiguous block. ``"interleaved"`` alternates the species at each position.
- Add [`LocalHamGen.get_trotter_gates`](#LocalHamGen.get_trotter_gates). It returns local gates for a first-, second-, or fourth-order Trotter approximation to $\exp(x H)$ over any number of steps. Terms are grouped into commuting layers. Each [`TrotterGate`](#TrotterGate) has ``frac``, ``layer``, and ``step`` attributes. It also unpacks as ``U, where``, so ``for U, where in gates: psi.gate_(U, where)`` works. Consecutive uses of the same layer are fused by default. Use [`trotter_schedule`](#trotter_schedule) to get only the product formula. [`LocalHam1D`](#LocalHam1D), [`LocalHam2D`](#LocalHam2D), and [`LocalHam3D`](#LocalHam3D) inherit this method.
- [`build_mpo_propagator_trotterized`](#LocalHam1D.build_mpo_propagator_trotterized) and [`build_pepo_propagator_trotterized`](#LocalHam2D.build_pepo_propagator_trotterized): add an ``order`` option. Its default remains 1. The 1D method also gains the ``ordering`` option already supported by the 2D method. Both methods now use [`get_trotter_gates`](#LocalHamGen.get_trotter_gates) to build the gate sequence.
- Belief propagation: all BP classes now accept ``diis`` and ``damping`` in both ``__init__`` and [`run`](#BeliefPropagationCommon.run). Values passed to ``run`` become the new defaults.
- Add [`gen_gloops_edge_induced`](#gen_gloops_edge_induced) (``tn.gen_gloops_edge_induced``). It yields [`NetworkPatch`](#NetworkPatch) objects that contain each loop's tensors and bonds. Unlike [`gen_gloops`](#gen_gloops), it distinguishes loops that span the same tensors but use different bonds.
- [`TN_from_strings`](#TN_from_strings): add ``join_prefer`` and ``join_avoid_loop_length``. These select string-end pairs by the length of the loop they would close. Length is the number of lines, or the number of tensors after site contraction. ``join_prefer="short"`` makes many small loops. When no loop can close, it prefers shorter string groups. ``"long"`` makes fewer, larger loops. ``join_avoid_loop_length`` sets the maximum length to avoid. It defaults to 2, which avoids loops made from parallel lines on one edge. Set it to 0 to disable loop avoidance. ``join_avoid_self_loops`` is deprecated. ``join_prefer`` does not support ``join="all"``.
- Add hidden-cactus tensor networks: [`TN2D_rand_hidden_cactus`](#TN2D_rand_hidden_cactus), [`TN3D_rand_hidden_cactus`](#TN3D_rand_hidden_cactus), and [`TN_rand_hidden_cactus`](#TN_rand_hidden_cactus). They join hidden loops into a tree. This extends hidden correlations across the graph without changing lattice bond dimensions. Their exact contraction cost remains low. The corresponding [`TN_from_strings`](#TN_from_strings) option is ``join_trees``, off by default.
- Add [`TN_rand_hidden_loop`](#TN_rand_hidden_loop), the arbitrary-graph version of [`TN2D_rand_hidden_loop`](#TN2D_rand_hidden_loop).
- Fix [`tids_are_connected`](#TensorNetwork.tids_are_connected) so its result does not depend on tensor order.
- Add [`MatrixProductState.from_product`](#MatrixProductState.from_product). [`MPS_product_state`](#MPS_product_state) now calls this method. It accepts block-sparse single-site vectors, e.g. from ``symmray``. Each new bond index is non-dual on the left and dual on the right. This makes the two ends contractible.


**Bug fixes:**

- fixed [`SpinHam1D.build_sparse`](#SpinHam1D.build_sparse) with ``cyclic=True`` from missing the last term ({issue}`419`)
- [`ikron`](#ikron): raise ``ValueError`` for indices that are out of range or repeated. These placed fewer operators than given, with no error.
- [`D1BP.contract_loop_series_expansion`](#D1BP.contract_loop_series_expansion) and [`D2BP.contract_loop_series_expansion`](#D2BP.contract_loop_series_expansion): sum all distinct loops in each region, including loops that do not excite every bond. Also use the intensive free energy in loop-series suppression factors. Multi-excitation corrections now raise ``RuntimeError`` if they do not converge.
- [`MPS_product_state`](#MPS_product_state): reshape single-site vectors through the backend-neutral interface. The previous ``x.reshape(*shape)`` call raised ``TypeError`` for array libraries that require a single shape tuple.
- [`build_matrix_ikron`](#SparseOperatorBuilder.build_matrix_ikron): use each term's register, not its site label, as the ``ikron`` index. The previous code silently returned the wrong matrix when these values differed, e.g. with non-identity ordering or non-integer site labels.
- [`HilbertSpace`](#HilbertSpace): prevent site reordering from leaving the rank-to-configuration mappings in the old order. In spaces with mixed dimensions, this decoded ranks into invalid configurations. Site ordering is now immutable, as described above.
- [`compute_oblique_projectors`](#compute_oblique_projectors): damp inverse singular values at ``max(s) * eps`` instead of dividing by them directly. Rank-deficient environments can retain zero singular values when ``cutoff=0.0``. These values previously produced ``inf`` or ``nan`` projectors. The shared diagonal division helpers use the same damping. This also applies to [`D2BP.gauge_insert`](#D2BP.gauge_insert) with ``return_gauges="inverse"``.
- [`safe_inverse`](#safe_inverse): compute a scalar maximum for a single vector. The previous last-axis reduction broadcast the result back. This caused a ``TypeError`` in ``mode="projector"`` boundary contractions with block-sparse backends such as ``symmray``.
- 1D compression: fix ``sdc`` and ``sdc-oversample`` for fermionic tensor networks.
- [`tensor_split`](#tensor_split): fix ``method='svd:rand'`` for complex and single precision arrays: use autoray random.array interface to match dtype and device.
- [`tensor_split`](#tensor_split): fix ``method='svd:eig'`` with nonzero ``cutoff`` for single precision and numba
- 1D compression: fix ``dm`` for fermionic tensor networks, including mixed bond orientations. Its eigendecomposition now keeps the same subspace as a direct SVD.
- 1D compression: fix ``fit`` for fermionic tensor networks. Local updates now include dual-axis phases. Final conjugation now handles the total dummy-mode parity.
- [`D2BP.compress`](#D2BP.compress) and [`D2BP.gauge_symmetric`](#D2BP.gauge_symmetric): preserve positive messages and full-rank identity matrices for fermionic tensor networks.
- [`TensorNetwork.conj`](#TensorNetwork.conj): use ``output_inds`` to select the global output legs for fermionic conjugation phases. This also works for subnetworks.
- [`Tensor.conj`](#Tensor.conj): add the same ``output_inds`` control for a single tensor.
- SRC compression: use QR, not SVD, by default to orthogonalize the sketched columns.
- [`TEBD`](#TEBD): fix ``order=4``, which was only second-order accurate. The Suzuki weight was ``1 / (4 * 4**(1/3))`` instead of ``1 / (4 - 4**(1/3))``. The method now converges at fourth order. For a Heisenberg chain with ``dt=0.05``, it is about 1,000 times more accurate. The correct weights include a negative value, so part of each fourth-order step now evolves backward. [`trotter_schedule`](#trotter_schedule) now supplies the step schedule. ``order=1`` is also accepted.
- [`build_mpo_propagator_trotterized`](#LocalHam1D.build_mpo_propagator_trotterized): apply the wrapping term of a cyclic chain with its sites in the correct order. The old order caused error linear in ``x`` instead of quadratic. Terms that are symmetric under a site swap, such as Heisenberg terms, were not affected.


## v1.15.0 (2026-08-10)

**Breaking Changes:**

- [`Circuit.uni`](#Circuit.uni): return the non-transposed unitary tensor network. Thus, ``circ.uni.to_dense()`` gives ``U``, which acts on a state as ``U @ psi``. The earlier transposed form and its ``FutureWarning`` are removed. Use ``circ.get_uni(transposed=True)`` for the old convention.
- [`TensorNetwork.gauge_all_simple`](#TensorNetwork.gauge_all_simple): arguments after ``max_iterations`` and ``tol`` are now keyword-only. This includes ``smudge``, ``power``, and ``damping``.
- Automatic generalized loop size (``max_size=None`` in [`tn.gen_gloops`](#TensorNetwork.gen_gloops), or ``gloops=None`` in loop expansions): grow loops until they cover every target tensor ID or site. Previously, loop growth stopped at the first and smallest loop. On non-uniform geometries, this could leave some sites without a loop. Targets outside the 2-core cannot be part of a loop and are ignored. A ``UserWarning`` is raised if these targets were explicit or if the network is a tree and has no loops. Loops larger than the resolved size are no longer returned. Use ``max_size="min"`` or ``gloops="min"`` for the old behavior.
- Simple-update gauging: ``smudge`` and ``gauge_smudge`` are now relative to the largest gauge value. They add ``smudge * max(g)`` to the gauge vector instead of ``smudge``. This affects [`tensor_gauge_simple_bond`](#tensor_gauge_simple_bond), [`TensorNetwork.gauge_all_simple`](#TensorNetwork.gauge_all_simple), [`TensorNetwork.gauge_simple_insert`](#TensorNetwork.gauge_simple_insert), [`TensorNetwork.gauge_simple_temp`](#TensorNetwork.gauge_simple_temp), [`tensor_canonize_bond`](#tensor_canonize_bond), [`tensor_compress_bond`](#tensor_compress_bond), [`TensorNetwork.contract_compressed`](#TensorNetwork.contract_compressed), [`tensor_network_ag_gate_simple`](#tensor_network_ag_gate_simple), [`TensorNetworkGenVector.get_cluster`](#TensorNetworkGenVector.get_cluster), [`TensorNetworkGenVector.partial_trace_cluster`](#TensorNetworkGenVector.partial_trace_cluster), and [`CircuitPEPSSimpleUpdate`](#CircuitPEPSSimpleUpdate). Results are unchanged for gauges normalized to ``max(g) == 1``, as is usual. Results change for unnormalized gauges.
- Non-exact circuit simulators no longer subclass the exact [`Circuit`](#Circuit). This applies to [`CircuitMPS`](#CircuitMPS), [`CircuitPermMPS`](#CircuitPermMPS), [`CircuitMPSLazy`](#CircuitMPSLazy), [`CircuitPEPSSimpleUpdate`](#CircuitPEPSSimpleUpdate), and [`CircuitPEPOSimpleUpdate`](#CircuitPEPOSimpleUpdate). They now use the shared [`CircuitBase`](#CircuitBase). Use ``isinstance(circ, CircuitBase)`` to check for any circuit type. Exact-contraction-only methods such as ``get_uni``, ``get_psi_simplified``, and ``sample_gate_by_gate`` are no longer available on these simulators. Supported methods remain available where applicable: ``to_dense``, ``amplitude``, ``sample``, ``partial_trace``, ``local_expectation``, ``simulate_counts``, and ``xeb``. The MPS methods now operate directly on the current, possibly compressed state. They do not perform exact tensor-network simplification and do not accept ``simplify_*`` options. MPS simulators also gain ``compute_marginal`` and ``sample_chaotic``. Methods that a representation cannot support, e.g. ``uni``, now raise ``NotImplementedError`` consistently instead of sometimes raising ``ValueError``.


**Enhancements:**

- Add [`CircuitPEPSSimpleUpdate`](#CircuitPEPSSimpleUpdate), a circuit simulator that stores the state as an arbitrary-geometry PEPS. It applies nearest-neighbor gates with simple-update gauging. A set of ``edges`` defines the geometry. These edges can be inferred from ``gates`` or read from ``psi0``. Use ``max_bond`` to control accuracy and ``equilibrate`` to rebalance gauges at intervals. Local expectations use the cluster approximation.
- Add [`CircuitPEPOSimpleUpdate`](#CircuitPEPOSimpleUpdate), a Heisenberg-picture simulator related to [`CircuitPEPSSimpleUpdate`](#CircuitPEPSSimpleUpdate). It records gates without applying them immediately. For an expectation, it builds the local observable as a bond-dimension-1 PEPO on ``edges``. It then evolves the PEPO backward with simple-update gauging and compression. Gates outside the reverse light cone are skipped. Use ``get_evolved_operator`` to get the evolved operator and ``local_expectation`` to get its ``|00...0>`` expectation.
- Add [`CircuitBase`](#CircuitBase), the shared base for all circuit simulators. It provides gate application, gate helper methods, named-parameter management, ``from_*`` constructors, and drawing. Exact, MPS, PEPS, and PEPO simulators now use this base instead of inheriting the exact [`Circuit`](#Circuit). Each representation implements ``get_psi``, which the ``psi`` property calls. All ``from_*`` constructors accept representation-specific options, such as PEPS or PEPO geometry ``edges``.
- Add ``quimb.tensor.tn2dinf``, a translation-invariant 2D tensor-network family for finding infinite-PEPS ground states with simple update. [`GeometryInfinite2D`](#GeometryInfinite2D) defines a unit cell from ``edges``. Sites have ``(cell, site_type)`` labels, and bonds are grouped into translation classes named ``bond_type``. [`PEPSInfinite2D`](#PEPSInfinite2D) and its base [`TensorNetworkInfinite2DFlat`](#TensorNetworkInfinite2DFlat) store one tensor per site. Shared tensors and indices keep all translations synchronized. [`LocalHamInfinite2D`](#LocalHamInfinite2D) defines one Hamiltonian per ``bond_type``. Its geometry can extend beyond the state geometry, e.g. for longer-range terms on a nearest-neighbor PEPS. [`SimpleUpdateInfinite2D`](#SimpleUpdateInfinite2D) uses [`SimpleUpdateGen`](#SimpleUpdateGen) for imaginary-time simple update. Local expectations use either a cluster approximation (``max_distance``) or a less costly generalized-loop approximation (``gloops``). Dense, Abelian-symmetric, and fermionic ``symmray`` backends are supported.
- Add [`GeometryInfinite2D.square`](#GeometryInfinite2D.square) to create an ``N`` by ``M`` square-lattice unit cell. Sites are placed on zero-based lattice points. Set the neighbor range with ``couplings`` or ``radius``. ``couplings`` can be the number of neighbor shells (``1`` for nearest neighbors, ``2`` to include next-nearest neighbors, etc.) or a list of ``(dx, dy)`` site displacements. ``radius`` sets a cutoff. Both use basis-independent square-lattice distance.
- [`TensorNetwork.gauge_all_simple`](#TensorNetwork.gauge_all_simple): add ``fuse_multibonds``. It updates gauges while preserving multi-index bonds. [`tensor_compress_bond`](#tensor_compress_bond) now supports the required explicit bond-index selection.
- Add [`tensor_gauge_simple_bond`](#tensor_gauge_simple_bond), the single-bond gauging step from [`gauge_all_simple`](#TensorNetwork.gauge_all_simple). It gauges one bond between two tensors against a shared ``gauges`` dictionary.
- [`tensor_network_ag_gate_simple`](#tensor_network_ag_gate_simple) (``gate_simple`` and ``gate_simple_``): support long-range gates between sites without a direct bond. The new [`tensor_network_ag_gate_simple_long_range`](#tensor_network_ag_gate_simple_long_range) splits the gate into a matrix-product operator. It applies the operator along a site path with simple-update gauging and recompression.
- [`tensor_canonize_bond`](#tensor_canonize_bond): add ``swap_inds`` to move one or more indices to the other tensor during canonization, e.g. to move a physical index. Also add ``bond_ind`` for explicit bond-index selection, matching [`tensor_compress_bond`](#tensor_compress_bond).
- [`SimpleUpdateGen`](#SimpleUpdateGen): use ``fuse_multibonds=False`` during gauging. This preserves multi-index bond structure.
- [`LocalHamGen.get_auto_ordering`](#LocalHamGen.get_auto_ordering): support ``group=True`` for the ``sort``, ``random``, and ``None`` strategies. This returns commuting layers. With ``order="random-ungrouped"``, shuffled terms are placed sequentially into commuting layers. This preserves random order instead of filling each layer greedily.
- Add [`LatticeBondMap`](#LatticeBondMap) to assign lattice bond indices consistently across open and periodic boundaries. PEPS, PEPO, PEPS3D, scalar 2D and 3D lattice tensor networks, and classical Ising tensor networks now use it.
- [`eigh_truncated`](#eigh_truncated): add ``shift`` for optional diagonal regularization.
- Add [`CircuitMPSLazy`](#CircuitMPSLazy), an MPS circuit simulator with deferred gate evaluation and periodic automatic compression. For long-range gates with ``src`` compression, it can be more efficient than [`CircuitMPS`](#CircuitMPS).
- [`CircuitBase.from_openqasm3_str`](#CircuitBase.from_openqasm3_str), [`CircuitBase.from_openqasm3_file`](#CircuitBase.from_openqasm3_file), and [`CircuitBase.from_openqasm3_url`](#CircuitBase.from_openqasm3_url): add OpenQASM 3 parsing. This supports custom gates, register broadcasting, and symbolic input tracking.
- [`CircuitDense`](#CircuitDense): accept controlled gates through ``controls=``. The method inserts a low-rank hyper tensor-network form of the gate and contracts it into the dense state. It does not build the full dense operator.
- Add [`gauge_d2bp`](#gauge_d2bp) and [`TensorNetwork.gauge_all_belief_propagation`](#TensorNetwork.gauge_all_belief_propagation). They put any tensor network in the symmetric gauge with dense 2-norm belief propagation. This is equivalent to simple-update gauging with singular values absorbed equally into the two tensors. The new [`D2BP.gauge_symmetric`](#D2BP.gauge_symmetric) method inserts the full-rank oblique projectors for the current messages.
- [`D2BP`](#D2BP): add ``power``, relative ``smudge`` conditioning, and [`converge_d2bp`](#converge_d2bp). Messages can also be used as projector environments. This includes ``canonize="bp"`` in [`tensor_network_ag_compress_projector`](#tensor_network_ag_compress_projector).
- [`TensorNetwork.norm`](#TensorNetwork.norm): add ``strip_exponent``. It returns the norm as separate mantissa and base-10 exponent values. This supports very large or small norms.
- [`TensorNetworkGenVector.norm_gloop_expand`](#TensorNetworkGenVector.norm_gloop_expand): add ``strip_exponent`` and an ``info`` cache for cluster contractions and the site-neighbor map. The cache uses the same keys as [`D1BP.contract_gloop_expand`](#D1BP.contract_gloop_expand). [`TensorNetworkGenVector.compute_local_expectation_gloop_expand`](#TensorNetworkGenVector.compute_local_expectation_gloop_expand) passes it through, so ``normalized="global"`` shares the cache. The method now copies supplied ``gauges`` instead of modifying them. [`TensorNetworkGen.normalize_simple`](#TensorNetworkGen.normalize_simple) also gains ``strip_exponent`` to return the normalization factor as separate mantissa and base-10 exponent values. [`combine_local_contractions`](#combine_local_contractions) gains an overall ``power`` option.
- Add [`Tensor.to`](#Tensor.to) and [`TensorNetwork.to`](#TensorNetwork.to) to change the backend, dtype, or device. This requires autoray v0.9.0 or newer.
- [`D1BP.contract_gloop_expand`](#D1BP.contract_gloop_expand): cache normalized contractions and add all singleton regions, i.e. regions with one tensor. The method can also remove dangling tensors from regions. These operations do not modify the target tensor network.
- [`trace_distance`](#trace_distance): compute the trace norm from the absolute eigenvalues of the difference between two states. Previously, it used singular values. Eigenvalues are valid because the difference is Hermitian. This method is faster, with larger gains at higher dimensions. Use ``isherm=False`` for the previous singular-value calculation.


**Internal:**

- [`gloop_remove_dangling`](#gloop_remove_dangling): use a set instead of a list to test region membership. This is 1.2 to 4 times faster and does not change behavior.
- [`squared_op_to_reduced_factor`](#squared_op_to_reduced_factor): use [`array_split`](#array_split) for the ``cholesky`` method, as for the other methods. This does not change behavior.
- Convert ``quimb.tensor.circuit`` from a module to a package with ``gates``, ``qasm``, ``core``, ``exact``, ``simple_update``, ``mps``, ``peps``, and ``pepo`` submodules. The public ``quimb.tensor.circuit.*`` import paths and each class's ``__module__`` remain unchanged. Existing imports and pickles continue to work. ``core`` contains the new [`CircuitBase`](#CircuitBase). ``simple_update`` contains the shared base for the PEPS and PEPO simple-update simulators. See the inheritance change above.


**Docs:**

- Sphinx AutoAPI now documents each object only where it is defined. For example, it documents ``Tensor`` at ``quimb.tensor.tensor_core``, not also at ``quimb.tensor``. Documentation can use matching short links such as [`Tensor`](#Tensor).


**Bug fixes:**

- [`D1BP.contract_loop_series_expansion`](#D1BP.contract_loop_series_expansion) and [`HD1BP.contract_gloop_expand`](#HD1BP.contract_gloop_expand): fix the default ``gloops=None``. It now generates loops instead of raising ``TypeError``.
- [`TensorNetwork3D.contract_peps_sweep`](#TensorNetwork3D.contract_peps_sweep): track norm exponents through both contraction stages. With ``strip_exponent=True``, the method can also return the mantissa and exponent separately.
- [`TN_from_strings`](#TN_from_strings) and random hidden-loop tensor networks: with ``normalize=True``, apply the tensor-network exponent once instead of twice. Applying it twice gave the wrong scale when normalization produced a nonzero exponent.
- [`CircuitDense`](#CircuitDense): fix ``psi``, ``partial_trace``, and ``local_expectation``. They raised ``ValueError`` because the contracted ``Dense1D`` view did not have its site count.
- [`CircuitPermMPS`](#CircuitPermMPS): make ``amplitude``, ``to_dense``, and ``local_expectation`` return results in logical qubit order after a nontrivial lazy permutation. Previously, only ``sample`` restored logical qubit order.
- [`CircuitPermMPS`](#CircuitPermMPS) and [`CircuitMPSLazy`](#CircuitMPSLazy): make ``copy()`` retain subclass attributes, such as qubit order and compression bookkeeping. Missing attributes made the copies unusable ({issue}`387`).
- [`CircuitMPSLazy`](#CircuitMPSLazy): apply pending gates before inherited state accessors run. This includes ``to_dense``, ``amplitude``, ``partial_trace``, and ``compute_marginal``. These accessors previously returned the exact state and bypassed configured compression. Also invalidate cached simplified states when compression changes the state. Results no longer depend on accessor order ({issue}`387`).
- [`CircuitMPS`](#CircuitMPS) and its subclasses: ``schrodinger_contract`` now raises ``NotImplementedError``. Approximate MPS simulators do not support this method. It previously failed with an internal ``IndexError`` ({issue}`387`).
- [`CircuitDense`](#CircuitDense): ``get_uni`` now raises the same clear ``ValueError`` as ``uni``. The eagerly contracted state cannot provide the unitary.
- [`Circuit.get_rdm_lightcone_simplified`](#Circuit.get_rdm_lightcone_simplified): check that the stored cache is valid before reading it, as in ``get_psi_simplified``. This prevents ``partial_trace`` and ``local_expectation`` from returning results cached before later gates were applied ({issue}`398`).
- [`tensor_network_1d_compress_src`](#tensor_network_1d_compress_src) and [`tensor_network_1d_compress_srcmps`](#tensor_network_1d_compress_srcmps): call [`enforce_1d_like`](#enforce_1d_like), as other 1D compression methods do. This fixes compression for tensor networks with long-range bonds that skip sites, e.g. networks with long-range gates applied lazily.
- [`enforce_1d_like`](#enforce_1d_like): insert identity strings correctly when ``site_tags`` order the tensors of a long-range bond in reverse, e.g. with ``sweep_reverse=True``. Previously, the identities connected to the wrong sites.
- [`PEPS`](#PEPS), [`PEPO`](#PEPO), and [`PEPS3D`](#PEPS3D): keep open and periodic bonds distinct when a cyclic dimension has length 1 or 2. This includes cyclic tensors with bond dimension 1.
- [`TensorNetwork2DVector.compute_norm`](#TensorNetwork2DVector.compute_norm): always return a scalar, not a tensor-network object.
- [`D2BP.partial_trace_loop_series_expansion`](#D2BP.partial_trace_loop_series_expansion): fix loop-series expansion for complex Hermitian BP messages. In ``get_cluster_excited``, boundary messages and inner excitation projectors had reversed ``(ket, bra)`` indices. This gave incorrect reduced density matrices for complex states ({issue}`380`).
- [`TensorNetworkGenVector.norm_gloop_expand`](#TensorNetworkGenVector.norm_gloop_expand): include the tensor network's ``exponent``. Ignoring it gave incorrect values when the norm was stored in the exponent, e.g. after [`equalize_norms`](#TensorNetwork.equalize_norms).
- [`D2BP.normalize_tensors`](#D2BP.normalize_tensors): update cached dual tensors during rescaling. This prevents repeated reduced-density-matrix calculations from drifting ({issue}`381`).
- [`D2BP.compress`](#D2BP.compress): after in-place compression, rebuild cached contraction expressions. Later message updates now use the compressed tensor data instead of stale arrays.
- [`TensorNetwork.isel`](#TensorNetwork.isel): support slice arguments.


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
