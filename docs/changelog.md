# Changelog

Release notes for `quimb`.

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
- [`quimb.experimental.belief_propagation`]([`quimb.experimental.belief_propagation`]):
  add various 1-norm/2-norm dense/lazy BP algorithms.

**Bug fixes:**

- fixed bug where an output index could be removed by squeezing when
  performing tensor network simplifications.


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

(whats-new-1-4-2)=
## v1.4.2 (2022-11-28)

**Enhancements**

- move from versioneer to to
  [setuptools_scm](https://pypi.org/project/setuptools-scm/) for versioning

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
