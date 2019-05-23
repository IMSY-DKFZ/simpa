import setuptools

with open("README.md", "r") as readme_file:
    long_description = readme_file.read()

setuptools.setup(
    name="ippai",
    version="0.0.1",
    author="Computer Assisted Medical Interventions, DKFZ",
    description="Image Processing for Photoacoustic Imaging",
    long_description=long_description,
    packages=setuptools.find_packages(),
    install_requires=[
        "numpy",
        "pynrrd"
    ]
)
