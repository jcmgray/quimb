###############
Developer notes
###############


Adding a function
=================

Steps:

1. Ensure function has numpy style docstring.
2. Make sure function is unit tested.
3. Document function at least in ``docs/api.rst``.
4. Add to ``quimb/__init__.py`` and ``"__all__"`` if appropriate.


Building the docs locally
=========================

1. To start from scratch, remove ``quimb/docs/_autosummary`` and ``quimb/docs/_build``.
2. Run ``make html`` (``make.bat html`` on windows) in the ``quimb/docs`` folder.
3. Launch page: ``quimb/docs/_build/html/index.html``.
