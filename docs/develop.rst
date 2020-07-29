###############
Developer Notes
###############


Contributing
============

Things to check if new functionality added:

1. Ensure functions are unit tested.
2. Ensure functions have `numpy style docstrings <http://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_numpy.html>`_.
3. Ensure code is PEP8 compliant.
4. If module, add to ``docs/api.rst`` for autosummarizing.
5. Add to ``quimb/__init__.py`` and ``"__all__"`` if appropriate (or the
   tensor network equivalent ``quimb.tensor.__init__.py``).
6. Add to changelog and elsewhere in docs.


Running the Tests
=================

Testing ``quimb`` requires `pytest <https://docs.pytest.org/en/latest/index.html>`_ (as well as ``coverage`` and ``pytest-cov``) and simply involves running ``pytest`` in the root ``quimb`` directory.

The tests can also be run with pre-spawned mpi workers using the command ``quimb-mpi-python -m pytest`` (but not in syncro mode -- see :ref:`mpistuff`).


Building the docs locally
=========================

Building the docs requires `sphinx <http://www.sphinx-doc.org/en/stable/>`_, `pydata-sphinx-theme <https://github.com/pandas-dev/pydata-sphinx-theme>`_, and `nbsphinx <https://nbsphinx.readthedocs.io>`_.

1. To start from scratch, remove ``quimb/docs/_autosummary`` and ``quimb/docs/_build``.
2. Run ``make html`` (``make.bat html`` on windows) in the ``quimb/docs`` folder.
3. Launch the page: ``quimb/docs/_build/html/index.html``.


Minting a Release
=================

``quimb`` uses `versioneer <https://github.com/warner/python-versioneer>`_
to manage versions and releases. The steps to release a new version
on `pypi <https://pypi.org>`_  are as follows:

1. Make sure all tests are passing, as well as the continuous integration
   and readthedocs build.
2. ``git tag`` the release with next ``X.Y.Z`` (n.b. no 'v' prefix).
3. Remove any old builds: ``rm dist/*```
4. Build the tar and wheel ``python setup.py bdist_wheel sdist``
5. Optionally remove the ``build`` folder.
6. Upload using twine: ``twine upload dist/*``
