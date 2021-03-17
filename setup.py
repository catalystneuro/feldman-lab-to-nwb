# -*- coding: utf-8 -*-
from setuptools import setup, find_packages

with open('README.md', 'r') as f:
    long_description = f.read()
setup(
    name='feldman_lab_to_nwb',
    version='0.0.1',
    description='NWB conversion scripts, functions, and classes for the Feldman lab.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Cody Baker, Alessio Buccino, and Ben Dichter',
    email='ben.dichter@catalystneuro.com',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'matplotlib', 'scipy', 'hdf5storage', 'jupyter', 'xlrd', 'h5py', 'pynwb', 'spikeextractors', 'nwbwidgets'
        'nwb-conversion-tools'
    ],
)
