#!/usr/bin/env python
# -*- coding: utf-8 -*-
# This file is part of the Cortix toolkit environment
# https://cortix.org

# Read contents of README file.
from os import path

import setuptools

THIS_DIRECTORY = path.abspath(path.dirname(__file__))

with open(path.join(THIS_DIRECTORY, 'README.md'), encoding='utf-8') as f:
    LONG_DESCRIPTION = f.read()

with open('requirements.txt', 'r') as fh:
    REQ = fh.readlines()

setuptools.setup(
    name='cortix',
    version='1.1.67',
    author="Cortix Computing",
    author_email="valmor_dealmeida@uml.edu",
    description='Cortix is a Python library for network dynamics modeling and HPC simulation.',
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    setup_requires=['wheel'],
    include_package_data=True,
    install_requires=REQ,
    url="https://cortix.org",
    packages=setuptools.find_namespace_packages(),
    keywords = ['simulation', 'math'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        'Topic :: Scientific/Engineering :: Mathematics',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: End Users/Desktop',
        'Intended Audience :: Science/Research',
        'Operating System :: MacOS',
        'Operating System :: Unix',
        'Topic :: Education',
        'Topic :: Utilities'
    ],
    python_requires='>=3.09'
)

