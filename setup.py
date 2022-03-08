from setuptools import setup, find_packages
import versioneer


def readme():
    with open('README.rst') as f:
        import re
        long_desc = f.read()
        # strip out the raw html images
        long_desc = re.sub(r'\.\. raw::[\S\s]*?>\n\n', "", long_desc)
        return long_desc


setup(
    name='quimb',
    description='Quantum information and many-body library.',
    long_description=readme(),
    url='http://quimb.readthedocs.io',
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    author='Johnnie Gray',
    author_email="johnniemcgray@gmail.com",
    license='Apache',
    packages=find_packages(exclude=['deps', 'tests*']),
    install_requires=[
        'numpy>=1.17',
        'scipy>=1.0.0',
        'numba>=0.39',
        'psutil>=4.3.1',
        'cytoolz>=0.8.0',
        'tqdm>=4',
    ],
    extras_require={
        'tensor': [
            'matplotlib>=2.0',
            'networkx>=2.3',
            'opt_einsum>=3.2',
            'autoray>=0.2.0',
            'diskcache>=3.0',
        ],
        'advanced_solvers': [
            'mpi4py',
            'petsc4py',
            'slepc4py',
        ],
        'random': [
            'randomgen>=1.18',
        ],
        'tests': [
            'coverage',
            'pytest',
            'pytest-cov',
        ],
        'docs': [
            'sphinx>=2.0',
            'sphinx-book-theme>=0.1',
            'nbsphinx>=0.4',
            'ipython>=7.0',
            'autoray>=0.2.0',
            'opt_einsum>=3.2',
            'doc2dash>=2.4.1',
        ],
    },
    scripts=['bin/quimb-mpi-python'],
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    keywords='quantum physics tensor networks tensors dmrg tebd',
)
