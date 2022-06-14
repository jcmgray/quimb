Changelog
=========

Release notes for ``quimb``.

.. _whats-new.1.4.0:

v1.4.0 (14th June 2022)
----------------------

**Enhancements**

- Add 2D tensor network support and algorithms
- Add 3D tensor network infrastructure
- Add arbitrary geometry quantum state infrastructure
- Many changes to :class:`TNOptimizer`
- Many changes to TN drawing
- Many changes to :class:`Circuit` simulation
- Many improvements to TN simplification
- Make all tag and index operations deterministic
- Add :func:`~quimb.tensor.tensor_core.tensor_network_sum`,
  :func:`~quimb.tensor.tensor_core.tensor_network_distance` and
  :meth:`~quimb.tensor.tensor_core.TensorNetwork.fit`
- Various memory and performance improvements
- Various graph generators and TN builders


.. _whats-new.1.3.0:

v1.3.0 (18th Feb 2020)
----------------------

**Enhancements**

- Added time dependent evolutions to :class:`~quimb.evo.Evolution` when integrating a pure state - see :ref:`time-dependent-evolution` - as well as supporting ``LinearOperator`` defined hamiltonians (:pull:`40`).
- Allow the :class:`~quimb.evo.Evolution` callback ``compute=`` to optionally access the Hamiltonian (:pull:`49`).
- Added :meth:`quimb.tensor.tensor_core.Tensor.randomize` and :meth:`quimb.tensor.tensor_core.TensorNetwork.randomize` to randomize tensor and tensor network entries.
- Automatically squeeze tensor networks when rank-simplifying.
- Add :meth:`~quimb.tensor.tensor_1d.TensorNetwork1DFlat.compress_site` for compressing around single sites of MPS etc.
- Add :func:`~quimb.tensor.tensor_gen.MPS_ghz_state` and :func:`~quimb.tensor.tensor_gen.MPS_w_state` for building bond dimension 2 open boundary MPS reprentations of those states.
- Various changes in conjunction with `autoray <https://github.com/jcmgray/autoray>`_ to improve the agnostic-ness of tensor network operations with respect to the backend array type.
- Add :func:`~quimb.tensor.tensor_core.new_bond` on top of :meth:`quimb.tensor.tensor_core.Tensor.new_ind` and :meth:`quimb.tensor.tensor_core.Tensor.expand_ind` for more graph orientated construction of tensor networks, see :ref:`tn-creation-graph-style`.
- Add the :func:`~quimb.gen.operators.fsim` gate.
- Make the parallel number generation functions use new `numpy 1.17+` functionality rather than `randomgen` (which can still be used as the underlying bit generator) (:pull:`50`)
- TN: rename ``contraction_complexity`` to :meth:`~quimb.tensor.tensor_core.TensorNetwork.contraction_width`.
- TN: update :meth:`quimb.tensor.tensor_core.TensorNetwork.rank_simplify`, to handle hyper-edges.
- TN: add :meth:`quimb.tensor.tensor_core.TensorNetwork.diagonal_reduce`, to automatically collapse all diagonal tensor axes in a tensor network, introducing hyper edges.
- TN: add :meth:`quimb.tensor.tensor_core.TensorNetwork.antidiag_gauge`, to automatically flip all anti-diagonal tensor axes in a tensor network allowing subsequent diagonal reduction.
- TN: add :meth:`quimb.tensor.tensor_core.TensorNetwork.column_reduce`, to automatically identify tensor axes with a single non-zero column, allowing the corresponding index to be cut.
- TN: add :meth:`quimb.tensor.tensor_core.TensorNetwork.full_simplify`, to iteratively perform all the above simplifications in a specfied order until nothing is left to be done.
- TN: add ``num_tensors`` and ``num_indices`` attributes, show ``num_indices`` in ``__repr__``.
- TN: various improvements to the pytorch optimizer (:pull:`34`)
- TN: add some built-in 1D quantum circuit ansatzes:
  :func:`~quimb.tensor.circuit_gen.circ_ansatz_1D_zigzag`,
  :func:`~quimb.tensor.circuit_gen.circ_ansatz_1D_brickwork`, and
  :func:`~quimb.tensor.circuit_gen.circ_ansatz_1D_rand`.
- **TN: add parametrized tensors** :class:`~quimb.tensor.tensor_core.PTensor` and so trainable, TN based quantum circuits -- see :ref:`example-tn-training-circuits`.

**Bug fixes:**

- Fix consistency of :func:`~quimb.calc.fidelity` by making the unsquared version the default for the case when either state is pure, and always return a real number.
- Fix a bug in the 2D system example for when ``j != 1.0``
- Add environment variable `QUIMB_NUMBA_PAR` to set whether numba should use automatic parallelization - mainly to fix travis segfaults.
- Make cache import and initilization of `petsc4py` and `slepc4py` more robust.

.. _whats-new.1.2.0:

v1.2.0 (6th June 2019)
----------------------

**Enhancements**

- Added :func:`~quimb.calc.kraus_op` for general, noisy quantum operations
- Added :func:`~quimb.calc.projector` for constructing projectors from observables
- Added :func:`~quimb.calc.measure` for measuring and collapsing quantum states
- Added :func:`~quimb.calc.cprint` pretty printing states in computational basis
- Added :func:`~quimb.calc.simulate_counts` for simulating computational basis counts
- TN: Add :meth:`quimb.tensor.tensor_core.TensorNetwork.rank_simplify`
- TN: Add :meth:`quimb.tensor.tensor_core.TensorNetwork.isel`
- TN: Add :meth:`quimb.tensor.tensor_core.TensorNetwork.cut_iter`
- TN: Add ``'split-gate'`` gate mode
- TN: Add :class:`~quimb.tensor.optimize_tensorflow.TNOptimizer` for tensorflow based optimization
  of arbitrary, contstrained tensor networks.
- TN: Add :meth:`quimb.tensor.tensor_1d.Dense1D.rand`
- TN: Add :func:`~quimb.tensor.tensor_core.connect` to conveniently set a shared index for tensors
- TN: make many more tensor operations agnostic of the array backend (e.g. numpy, cupy,
  tensorflow, ...)
- TN: allow :func:`~quimb.tensor.tensor_1d.align_TN_1D` to take an MPO as the first argument
- TN: add :meth:`~quimb.tensor.tensor_gen.SpinHam1D.build_sparse`
- TN: add :meth:`quimb.tensor.tensor_core.Tensor.unitize` and :meth:`quimb.tensor.tensor_core.TensorNetwork.unitize` to impose unitary/isometric constraints on tensors specfied using the ``left_inds`` kwarg
- Many updates to tensor network quantum circuit
  (:class:`quimb.tensor.circuit.Circuit`) simulation including:

  * :class:`quimb.tensor.circuit.CircuitMPS`
  * :class:`quimb.tensor.circuit.CircuitDense`
  * 49-qubit depth 30 circuit simulation example :ref:`quantum-circuit-example`

- Add ``from quimb.gates import *`` as shortcut to import ``X, Z, CNOT, ...``.
- Add :func:`~quimb.gen.operators.U_gate` for parametrized arbitrary single qubit unitary

**Bug fixes:**

- Fix ``pkron`` for case ``len(dims) == len(inds)`` (:issue:`17`, :pull:`18`).
- Fix ``qarray`` printing for older ``numpy`` versions
- Fix TN quantum circuit bug where Z and X rotations were swapped
- Fix variable bond MPO building (:issue:`22`) and L=2 DMRG
- Fix ``norm(X, 'trace')`` for non-hermitian matrices
- Add ``autoray`` as dependency (:issue:`21`)
