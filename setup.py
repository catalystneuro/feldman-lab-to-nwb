# -*- coding: utf-8 -*-
from setuptools import setup, find_packages
import os

with open("README.md", "r") as f:
    long_description = f.read()
with open(os.path.join(os.path.abspath(os.path.dirname(__file__)), "requirements.txt")) as f:
    install_requires = f.read().strip().split("\n")
setup(
    name="feldman_lab_to_nwb",
    version="0.0.1",
    description="NWB conversion scripts, functions, and classes for the Feldman lab.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Cody Baker, Alessio Buccino, and Ben Dichter",
    email="ben.dichter@catalystneuro.com",
    packages=find_packages(),
    include_package_data=True,
    install_requires=install_requires,
)
