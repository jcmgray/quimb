# Contributing

Contributions to `quimb` in the form of
[pull requests](https://github.com/jcmgray/quimb/pulls) are very welcome.
Opening an [issue](https://github.com/jcmgray/quimb/issues) first can be useful
for larger changes, design questions, or work that might affect public APIs.

If this is your first time contributing on GitHub, the following guide may be
useful:

- [GitHub - Creating a pull request](https://help.github.com/articles/creating-a-pull-request/)

Please read and follow the [`quimb` Code of Conduct](../CODE_OF_CONDUCT.md).

## Development Setup

`quimb` uses [pixi](https://pixi.sh) for development environments and
predefined project tasks. The environments and commands are defined in
[`pyproject.toml`](../pyproject.toml).

From a fresh clone:

```bash
git clone https://github.com/jcmgray/quimb.git
cd quimb
pixi install
```

## Common Commands

Run tests:

```bash
pixi run pytest tests/
pixi run testmatrix
pixi run testtensor
```

Run a focused test:

```bash
pixi run pytest tests/test_utils.py
pixi run pytest tests/test_utils.py::TestOset
pixi run pytest tests/test_utils.py::TestOset::test_basic
```

Run checks in a specific environment:

```bash
pixi run -e testpyold testmatrix
pixi run -e testpynew testtensor
pixi run -e testjax testtensor
pixi run -e testtorch testtensor
pixi run -e testtensorflow testtensor
pixi run -e testslepc testmatrix
```

Format and lint:

```bash
pixi run lint
pixi run format
```

Build and serve the docs:

```bash
pixi run docs
pixi run docs-serve
```

More developer details are in the
[development guide](https://quimb.readthedocs.io/en/latest/develop.html).

## Contribution Checklist

- [ ] Tests have been added for new functionality.
- [ ] Relevant tests pass with the pixi tasks above.
- [ ] Public functions have
      [NumPy-style docstrings](http://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_numpy.html).
- [ ] Public API additions are exported from the appropriate `__init__.py`.
- [ ] New functionality is documented in `docs/` or demonstrated with an
      example notebook when appropriate.
- [ ] User-facing changes are noted in `docs/changelog.md`.
- [ ] Formatting and lint checks pass with `pixi run lint`.
