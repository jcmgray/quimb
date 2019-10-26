#!/bin/sh
set -ex

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ENV="test-environment-${TRAVIS_PYTHON_VERSION}"

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
  conda env create \
    --name $ENV \
    python=$TRAVIS_PYTHON_VERSION \
    --file $DIR/requirements-py3.yml
  source activate $ENV
# ~~~ cached install ~~~ #
else
  echo "Using cached conda installation."
  export PATH="$HOME/conda/bin:$PATH"
  hash -r
  conda config --set always_yes yes --set changeps1 no
  conda update -q conda
  source activate $ENV
  conda update -q \
    --all \
    python=$TRAVIS_PYTHON_VERSION
  conda env update \
    --file $DIR/requirements-py3.yml \
    python=$TRAVIS_PYTHON_VERSION
  pip install -U codeclimate-test-reporter codacy-coverage
  pip uninstall --yes quimb
fi
