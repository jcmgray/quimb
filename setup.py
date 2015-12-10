from setuptools import setup, find_packages

setup(
    name='quijy',
    version='0.1.1.dev1',
    author='Johnnie Gray',
    license='MIT',
    packages=find_packages(exclude=['contrib', 'docs', 'tests*']),
    install_requires=[
        'numpy>=1.10',
        'numexpr',
        'scipy',
        'numba',
        'numexpr',
        'matplotlib'
        ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.5',
        ]
    )
