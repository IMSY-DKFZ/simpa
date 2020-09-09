#README

The Simulation and Image Processing for Photoacoustic Imaging (SIMPA) toolkit.

## Internal Install Instructions

These install instructions are made under the assumption that you have access to the phabricator simpa project.
When you are reading these instructions there is a 99% chance that is the case (or someone send these instructions
to you).

So, for the 1% of you: Please also follow steps 1 - 3:

1. `git clone https://phabricator.mitk.org/source/simpa.git`
2. `git checkout master`
3. `git pull`

Now open a python instance in the 'simpa' folder that you have just downloaded. Make sure that you have your preferred
virtual environment activated
4. `cd simpa`
5. `python -m setup.py build install`
6. Test if the installation worked by using `python` followed by `import simpa` then `exit()`

If no error messages arise, you are now setup to use simpa in your project.

## Building the documentation

When the installation went fine and you want to make sure that you have the latest documentation
you should do the following steps in a command line:

1. Navigate to the `simpa` source directory (same level where the setup.py is in)
2. Execute the command `sphinx-build -b pdf -a documentation/src documentation`
3. Find the `PDF` file in `documentation/simpa_documantation.pdf`

## Overview

The main use case for the simpa framework is the simulation of photoacoustic images.
However, it can also be used for image processing.

### Simulating photoacoustic images

A basic example on how to use simpa in you project to run an optical forward simulation is given in the 
samples/minimal_optical_simulation.py file.