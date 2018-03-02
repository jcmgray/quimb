###############
Developer notes
###############


Running the Tests
=================

Testing ``quimb`` requires `pytest <https://docs.pytest.org/en/latest/index.html>`_ and simply involves running ``pytest`` in the root ``quimb`` directory.

The tests can also be run with pre-spawned mpi workers using the command ``quimb-mpi-python -m pytest`` (but not in syncro mode -- see :ref:`mpistuff`).


Adding a function
=================

Steps:

1. Ensure function has numpy style docstring.
2. Make sure function is unit tested.
3. If module, add to ``docs/api.rst`` for autosummarizing.
4. Add to ``quimb/__init__.py`` and ``"__all__"`` if appropriate.


Building the docs locally
=========================

Building the docs requires `sphinx <http://www.sphinx-doc.org/en/stable/>`_ and `sphinx_bootstrap_theme <https://ryan-roemer.github.io/sphinx-bootstrap-theme/>`_

1. To start from scratch, remove ``quimb/docs/_autosummary`` and ``quimb/docs/_build``.
2. Run ``make html`` (``make.bat html`` on windows) in the ``quimb/docs`` folder.
3. Launch page: ``quimb/docs/_build/html/index.html``.
