#!/bin/sh
set -ex

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# ~~~ New install ~~~ #
if [ ! -d "$HOME/conda/bin" ]; then
  if [ -d "$HOME/conda" ]; then
    rm -rf $HOME/conda
  fi
  echo "Creating new conda installation."
  wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh;
  bash miniconda.sh -b -p $HOME/conda
  export PATH="$HOME/conda/bin:$PATH"
  hash -r
  conda config --set always_yes yes --set changeps1 no
  conda install pyyaml
  conda update -q conda
  conda info -a
  conda env create --file $DIR/requirements-py3.yml
  source activate test-environment
# ~~~ cached install ~~~ #
else
  echo "Using cached conda installation."
  export PATH="$HOME/conda/bin:$PATH"
  hash -r
  conda config --set always_yes yes --set changeps1 no
  conda update -q conda
  source activate test-environment
  conda update -q --all
  pip install -U codeclimate-test-reporter codacy-coverage
  pip install -U --editable=git+https://github.com/dgasmith/opt_einsum.git
  pip uninstall --yes quimb
fi
