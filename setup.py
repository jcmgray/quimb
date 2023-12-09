from setuptools import setup, find_packages


def readme():
    with open("README.md") as f:
        import re

        long_desc = f.read()
        # strip out the raw html images
        long_desc = re.sub(r"\.\. raw::[\S\s]*?>\n\n", "", long_desc)
        return long_desc


setup(
    name="quimb",
    description="Quantum information and many-body library.",
    long_description=readme(),
    long_description_content_type="text/markdown",
    url="http://quimb.readthedocs.io",
    author="Johnnie Gray",
    author_email="johnniemcgray@gmail.com",
    license="Apache",
    packages=find_packages(exclude=["deps", "tests*"]),
    install_requires=[
        "autoray>=0.6.7",
        "cotengra>=0.5.6",
        "cytoolz>=0.8.0",
        "numba>=0.39",
        "numpy>=1.17",
        "psutil>=4.3.1",
        "scipy>=1.0.0",
        "tqdm>=4",
    ],
    extras_require={
        "tensor": [
            "matplotlib>=2.0",
            "networkx>=2.3",
        ],
        "advanced_solvers": [
            "mpi4py",
            "petsc4py",
            "slepc4py",
        ],
        "tests": [
            "coverage",
            "pytest",
            "pytest-cov",
        ],
        "docs": [
            "astroid<3.0.0",
            "autoray>=0.6.7",
            "cotengra>=0.5.3",
            "doc2dash>=2.4.1",
            "furo",
            "ipython!=8.7.0",
            "myst-nb",
            "setuptools_scm",
            "sphinx-autoapi",
            "sphinx-copybutton",
            "sphinx-design",
            "sphinx>=2.0",
        ],
    },
    scripts=["bin/quimb-mpi-python"],
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    keywords="quantum physics tensor networks tensors dmrg tebd",
)
