from setuptools import setup, find_packages

import pathlib

here = pathlib.Path(__file__).parent.resolve()
long_description = (here / "README.md").read_text(encoding="utf-8")

setup(
    name="stablegp",
    version="0.1",
    packages=find_packages(include=["stablegp", "stablegp.*"]),
    install_requires=[
        "torch",
        "numpy",
    ],
    description="Sparse Variational Gaussian Process regression in PyTorch",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/rudolfsg/stablegp",
    author="Rudolfs Grobins",
)
