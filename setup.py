# SPDX-FileCopyrightText: 2021 Computer Assisted Medical Interventions Group, DKFZ
# SPDX-FileCopyrightText: 2021 Janek Groehl
# SPDX-License-Identifier: MIT

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
    url="https://github.com/CAMI-DKFZ/simpa",
    author="Computer Assisted Medical Interventions (CAMI), DKFZ and Janek Groehl",
    description="Simulation and Image Processing for Photonics and Acoustics",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(exclude=["simpa_tests", "simpa_examples", ]),
    install_requires=requirements,
    include_package_data=True
)
