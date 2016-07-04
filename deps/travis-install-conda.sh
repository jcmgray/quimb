#!/bin/sh

export CONDA_PATH=$HOME/conda

if [ ! -d "$CONDA_PATH" ]; then
  wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh;
  bash miniconda.sh -b -p $CONDA_PATH
  export PATH="$CONDA_PATH/bin:$PATH"
  hash -r
  conda config --set always_yes yes --set changeps1 no
  conda update -q conda
  conda info -a
  conda create -q -n test-environment python=$TRAVIS_PYTHON_VERSION numpy scipy numba numexpr coverage
  pip install pytest-cov coveralls
else
  export PATH="$CONDA_PATH/bin:$PATH"
  conda config --set always_yes yes --set changeps1 no
  conda update -q conda
  conda info -a
  source activate test-environment
  conda update -q --all
  pip install -U pytest-cov coveralls
fi
