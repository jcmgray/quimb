# Developer Notes

## Contributing

Things to check if new functionality added:

1. Ensure functions are unit tested.
2. Ensure functions have [numpy style docstrings](http://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_numpy.html).
3. Ensure code is PEP8 compliant.
4. Add to `quimb/__init__.py` and `"__all__"` if appropriate (or the
   tensor network equivalent `quimb.tensor.__init__.py`).
5. Add to changelog and elsewhere in docs.

## Running the Tests

Testing `quimb` requires [pytest](https://docs.pytest.org/en/latest/index.html) (as well as `coverage` and `pytest-cov`) and simply involves running `pytest` in the root `quimb` directory.

The tests can also be run with pre-spawned mpi workers using the command `quimb-mpi-python -m pytest` (but not in syncro mode -- see {ref}`mpistuff`).

## Building the docs locally

Building the docs requires [sphinx](http://www.sphinx-doc.org),
[myst_nb](https://myst-nb.readthedocs.io),
[sphinx-autoapi](https://sphinx-autoapi.readthedocs.io),
[sphinx_copybutton](https://sphinx-copybutton.readthedocs.io),
and [furo](https://github.com/pradyunsg/furo).

1. `cd` into the `quimb/docs` folder.
2. To start from scratch, remove the `_build` folder.
3. Run `sphinx-build -b html . ./_build/html/`.
4. Launch the page: `open _build/html/index.html`.

### Building the DocSet

Building the DocSet requires [doc2dash >= 2.4.1](https://github.com/hynek/doc2dash).

1. To start from scratch, remove `quimb/docs/_build`.
2. Run `make docset` in the `quimb/docs` folder.
3. Open the file `quimb/docs/_build/quimb.docset` to load it to Dash.

Afterwards, in order to update the Dash repository with a the DocSet after a new release:

1. Clone the [Dash-User-Contributions](https://github.com/Kapeli/Dash-User-Contributions).
2. Go to `docsets/quimb`, create a new directory with the version name inside the `versions` dir and copy there the generated DocSet.
3. Edit the `docset.json`: update the `"version"` and add a new element below `"specific_versions"`.
4. Commit and create a new Pull Request.

## Minting a Release

`quimb` uses [setuptools_scm](https://github.com/pypa/setuptools_scm)
to manage version. The steps to release a new version
on [pypi](https://pypi.org)  are as follows:

1. Make sure all tests are passing, as well as the continuous integration
   and readthedocs build.
2. `git tag` the release with next `vX.Y.Z`
3. Push the tag to github: `git push --tags` and github actions will build and
   upload the release to pypi, which will then get picked up by conda-forge.

Alternate manual release steps (after tagging):

3. Remove any old builds: `` rm dist/*` ``
4. Build the tar and wheel `python -m build`
5. Upload using twine: `twine upload dist/*`
