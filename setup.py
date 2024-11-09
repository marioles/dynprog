# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

with open("README.md") as f:
    readme = f.read()

setup(
    name="dynprog",
    version="0.1.0",
    description="Dynamic programming",
    long_description=readme,
    author="Mario Morales",
    author_email="mario@moralesalfaro.cl",
    url="https://github.com/marioles/dynprog",
    packages=find_packages(exclude=("tests", "docs"))
)
