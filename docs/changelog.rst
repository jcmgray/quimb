Changelog
=========

Release notes for ``quimb``.


.. _whats-new.1.2.1:

v1.2.1 (unreleased)
-------------------

**Enhancements**

- Added :meth:`quimb.tensor.tensor_core.Tensor.randomize` and :meth:`quimb.tensor.tensor_core.TensorNetwork.randomize` to randomize tensor and tensor network entries.
- Automatically squeeze tensor networks when rank-simplifying
- Add :meth:`~quimb.tensor.tensor_1d.TensorNetwork1DFlat.compress_site` for compressing around single sites of MPS etc.
- Add :func:`~quimb.tensor.tensor_gen.MPS_ghz_state` and :func:`~quimb.tensor.tensor_gen.MPS_w_state` for building bond dimension 2 open boundary MPS reprentations of those states.
- Various changes in conjunction with `autoray <https://github.com/jcmgray/autoray>`_ to improve the agnostic-ness of tensor network operations with respect to the backend array type
- Add :func:`~quimb.tensor.tensor_core.new_bond` on top of :meth:`quimb.tensor.tensor_core.Tensor.new_ind` and :meth:`quimb.tensor.tensor_core.Tensor.expand_ind` for more graph orientated construction of tensor networks, see :ref:`tn-creation-graph-style`
- Add the :func:`~quimb.gen.operators.fsim` gate.

**Bug fixes:**

- Fix consistency of :func:`~quimb.calc.fidelity` by making the unsquared version the default for the case when either state is pure, and always return a real number.
- Fix a bug in the 2D system example for when ``j != 1.0``

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
- TN: add :meth:`~quimb.tensor.tensor_gen.SpinHam.build_sparse`
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
