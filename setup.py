from setuptools import setup, find_packages

setup(
    name='quimb',
    version='0.2.1',
    author='Johnnie Gray',
    license='MIT',
    packages=find_packages(exclude=['deps', 'tests*']),
    install_requires=[
        'numpy>=1.10',
        'scipy>=0.15',
        'numba>=0.22',
        'numexpr>=2.3',
        'psutil>=4.3.1',
        'cytoolz>=0.8.0',
        'tqdm',
    ],
    extras_require={
        'test': [
            'coverage',
            'pytest',
            'pytest-cov',
        ],
        'docs': [
            'sphinx',
            'sphinx_rtd_theme',
            'sphinx_bootstrap_theme',
        ],
        'advanced_solvers': [
            'slepc4py',
        ]
    },
    scripts=['bin/quimb-mpiexec'],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ]
)
