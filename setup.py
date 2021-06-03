"""
SPDX-FileCopyrightText: 2021 Computer Assisted Medical Interventions Group, DKFZ
SPDX-FileCopyrightText: 2021 VISION Lab, Cancer Research UK Cambridge Institute (CRUK CI)
SPDX-License-Identifier: MIT
"""

import setuptools

with open('VERSION', 'r') as readme_file:
    version = readme_file.read()

with open('README.md', 'r') as readme_file:
    long_description = readme_file.read()

with open('requirements.txt', 'r') as requirements_file:
    requirements = requirements_file.read().splitlines()

setuptools.setup(
    name="simpa",
    version=version,
    author="Computer Assisted Medical Interventions (CAMI), DKFZ \n"
           "Cancer Research UK, Cambridge Institute (CRUK CI)",
    description="Simulation and Image Processing for Photoacoustic Imaging",
    long_description=long_description,
    packages=['simpa'],
    install_requires=requirements
)