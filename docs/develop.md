# Developer Notes

## Contributing

Contributions to `quimb` are very welcome, whether they are bug reports,
documentation fixes, examples, tests, or new features. If you are planning a
larger change, opening an issue first is often the easiest way to check the
approach before spending too much time on implementation.

Please also read the
[`quimb` Code of Conduct](https://github.com/jcmgray/quimb/blob/main/CODE_OF_CONDUCT.md).

Things to check if new functionality is added:

1. Ensure functions are unit tested.
2. Ensure functions have
   [NumPy-style docstrings](http://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_numpy.html).
3. Ensure code is formatted and linted with `pixi run lint`.
4. Add to `quimb/__init__.py` and `"__all__"` if appropriate (or the
   tensor network equivalent `quimb.tensor.__init__.py`).
5. Add to changelog and elsewhere in docs.


### AI Policy

Please treat the [numpy AI policy](https://numpy.org/devdocs/dev/ai_policy.html) as a rough guide.


## Development Setup

`quimb` uses [pixi](https://pixi.sh) to manage development environments and
reproducible tasks. The environments and tasks are defined in
`pyproject.toml`, which is the source of truth for the commands below.

After cloning the repository, install the pixi environments from the project
root:

```bash
git clone https://github.com/jcmgray/quimb.git
cd quimb
pixi install
```

You can then run project tasks with `pixi run ...`. For example, to run a
short Python command inside the current default test environment:

```bash
pixi run -e testpymid python -c "import quimb; print(quimb.__version__)"
```


## Running the Tests

Testing `quimb` is also handled by pixi tasks. The most common commands are:

```bash
pixi run pytest tests/
pixi run testmatrix
pixi run testtensor
```

The `pytest` task runs in the default test environment. For a narrower check,
pass pytest arguments after the task name:

```bash
pixi run pytest tests/test_utils.py
pixi run pytest tests/test_utils.py::TestOset
pixi run pytest tests/test_utils.py::TestOset::test_basic
```

To run a task in a specific test environment, use `-e`:

```bash
pixi run -e testpyold testmatrix
pixi run -e testpynew testtensor
pixi run -e testjax testtensor
pixi run -e testtorch testtensor
pixi run -e testtensorflow testtensor
pixi run -e testslepc testmatrix
```

The tests can also be run with pre-spawned mpi workers using the command
`quimb-mpi-python -m pytest` (but not in syncro mode -- see {ref}`mpistuff`).


## Formatting the Code

`quimb` uses [`ruff`](https://docs.astral.sh/ruff/) to format imports and code
style. Use the predefined pixi tasks rather than running the tools directly:

```bash
pixi run lint
pixi run format
```

The `format-all` task also runs notebook cleanup with `squeaky`:

```bash
pixi run format-all
```


## Building the docs locally

The documentation dependencies are also managed by pixi. To build, clean, and
serve the docs locally, use:

```bash
pixi run docs
pixi run docs-clean
pixi run docs-serve
```

The local server hosts the built docs at
`http://localhost:8000/`. The generated HTML is in `docs/_build/html/`.


### Building the DocSet

Building the DocSet requires
[doc2dash >= 2.4.1](https://github.com/hynek/doc2dash).

1. To start from scratch, remove `quimb/docs/_build`.
2. Run `make docset` in the `quimb/docs` folder.
3. Open the file `quimb/docs/_build/quimb.docset` to load it to Dash.

Afterwards, in order to update the Dash repository with a the DocSet after a
new release:

1. Clone the
   [Dash-User-Contributions](https://github.com/Kapeli/Dash-User-Contributions).
2. Go to `docsets/quimb`, create a new directory with the version name inside
   the `versions` dir and copy there the generated DocSet.
3. Edit the `docset.json`: update the `"version"` and add a new element below
  `"specific_versions"`.
4. Commit and create a new Pull Request.


## Minting a Release

`quimb` uses [hatch-vcs](https://github.com/ofek/hatch-vcs) to manage version.
The steps to release a new version on [pypi](https://pypi.org) are as follows:

1. Make sure all tests are passing, as well as the continuous integration
   and readthedocs build.
2. `git tag` the release with next `vX.Y.Z`
3. Push the tag to github: `git push --tags` and github actions will build and
   upload the release to [test pypi](https://test.pypi.org/).
4. Mint a github release from the tag, adding release notes from
   `docs/changelog.md`. Github actions will build and upload the release
   to pypi, and conda-forge will pick it up and build the conda package.

Alternate manual release steps (after tagging):

3. Remove any old builds: `` rm dist/*` ``
4. Build the tar and wheel `python -m build`
5. Upload using twine: `twine upload dist/*`
