Changelog
=========

Release notes for ``quimb``.


.. _whats-new.1.2.0:

v1.2.0 (unreleased)
--------------------

**Enhancements**

- Added ``kraus_op`` for general, noisy quantum operations
- TN: Many tweaks to :class:`quimb.tensor.circuit.Circuit`
- TN: Add :meth:`quimb.tensor.tensor_core.TensorNetwork.rank_simplify`
- TN: Add :meth:`quimb.tensor.tensor_core.TensorNetwork.isel`
- TN: Add :meth:`quimb.tensor.tensor_core.TensorNetwork.cut_iter`
- TN: Add ``'split-gate'`` gate mode


**Bug fixes:**

- Fix ``pkron`` for case ``len(dims) == len(inds)`` (:issue:`17`, :pull:`18`).
- Fix ``qarray`` printing for older ``numpy`` versions
