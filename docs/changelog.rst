Changelog
=========

Release notes for ``quimb``.


.. _whats-new.1.2.0:

v1.2.0 (unreleased)
--------------------

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
