import setuptools

with open('README.md', 'r') as readme_file:
    long_description = readme_file.read()

with open('requirements.txt', 'r') as requirements_file:
    requirements = requirements_file.read().splitlines()

setuptools.setup(
    name="simpa",
    version="0.3.0",
    author="Computer Assisted Medical Interventions (CAMI), DKFZ",
    description="Image Processing for Photoacoustic Imaging",
    long_description=long_description,
    packages=setuptools.find_packages(),
    install_requires=requirements
)