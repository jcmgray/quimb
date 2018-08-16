from setuptools import setup, find_packages
import versioneer


def readme():
    with open('README.rst') as f:
        import re
        long_desc = f.read()
        # strip out the raw html images
        long_desc = re.sub('\.\. raw::[\S\s]*?>\n\n', "", long_desc)
        return long_desc


setup(
    name='quimb',
    description='Quantum information and many-body library.',
    long_description=readme(),
    url='http://quimb.readthedocs.io',
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    author='Johnnie Gray',
    author_email="john.gray.14@ucl.ac.uk",
    license='MIT',
    packages=find_packages(exclude=['deps', 'tests*']),
    install_requires=[
        'numpy>=1.12',
        'scipy>=1.0.0',
        'numba>=0.39',
        'numexpr>=2.3',
        'psutil>=4.3.1',
        'cytoolz>=0.8.0',
        'tqdm>=4',
        'opt_einsum>=2',
    ],
    extras_require={
        'tensor': [
            'matplotlib',
            'networkx',
        ],
        'advanced_solvers': [
            'mpi4py',
            'petsc4py',
            'slepc4py',
        ],
        'random': [
            'randomgen>=1.14',
        ],
        'tests': [
            'coverage',
            'pytest',
            'pytest-cov',
        ],
        'docs': [
            'sphinx',
            'sphinx_bootstrap_theme',
            'nbsphinx',
            'ipython',
        ],
    },
    scripts=['bin/quimb-mpi-python'],
    python_requires='>=3.5',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
    keywords='quantum physics tensor networks tensors dmrg tebd',
)
