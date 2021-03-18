# README

The Simulation and Image Processing for Photoacoustic Imaging (SIMPA) toolkit.

## SIMPA Install Instructions

You can install simpa with pip. Simply run:

`pip install simpa`

For a manual installation from the code, please follow steps 1 - 3:

1. `git clone https://github.com/CAMI-DKFZ/simpa.git`
2. `git checkout master`
3. `git pull`

Now open a python instance in the 'simpa' folder that you have just downloaded. Make sure that you have your preferred
virtual environment activated
1. `cd simpa`
2. `pip install -r requirements.txt`
3. `python -m setup.py install`
4. Test if the installation worked by using `python` followed by `import simpa` then `exit()`

If no error messages arise, you are now setup to use simpa in your project.

You also need to manually install the pytorch library to use all features of SIMPA.
To this end, use the pytorch website tool to figure out which version to install:
`https://pytorch.org/get-started/locally/`

## Building the documentation

When the installation went fine and you want to make sure that you have the latest documentation
you should do the following steps in a command line:

1. Navigate to the `simpa` source directory (same level where the setup.py is in)
2. Execute the command `sphinx-build -b pdf -a simpa_documentation/src simpa_documentation`
3. Find the `PDF` file in `simpa_documentation/simpa_documantation.pdf`

## External Tools installation instructions

### mcx (Optical Forward Model)

Either download suitable executables or build yourself from the following sources:

http://mcx.space/

### k-Wave (Acoustic Forward Model)

Please follow the following steps and use the k-Wave install instructions 
for further (and much better) guidance under:

http://www.k-wave.org/

1. Install MATLAB with the core and parallel computing toolboxes activated at the minimum.
2. Download the kWave toolbox
3. Add the kWave toolbox base bath to the toolbox paths in MATLAB
4. If wanted: Download the CPP and CUDA binary files and place them inthe k-Wave/binaries folder
5. Note down the system path to the `matlab` executable file.

## Overview

The main use case for the simpa framework is the simulation of photoacoustic images.
However, it can also be used for image processing.

### Simulating photoacoustic images

A basic example on how to use simpa in you project to run an optical forward simulation is given in the 
samples/minimal_optical_simulation.py file.

## How to contribute

Please find a more detailed description of how to contribute as well as code style references in our **developer_guide.md**

The SIMPA code is written and maintained on a closed git repository that is hosted on a server
of the German Cancer Research Center (DKFZ), Heidelberg, Germany and changes to the develop or master branch are mirrored on Github.
As such, only the current master and develop branch of the repository are open source.

To contribute to SIMPA, please fork the SIMPA github repository and create a pull request with a branch containing your 
suggested changes. The core team developers will then review the suggested changes and integrate these into the code 
base.

Please make sure that you have included unit tests for your code and that all previous tests still run through.

There is a regular SIMPA status meeting at 10:00 CET/CEST and you are very welcome to participate and
raise any issues or suggest new features. You can join the meeting using the following link:

https://meet.google.com/rze-bxej-cvj

Please see the github guidelines for creating pull requests: https://docs.github.com/en/github/collaborating-with-issues-and-pull-requests/about-pull-requests


## Performance profiling

Do you wish to know which parts of the simulation pipeline cost the most amount of time? 
If that is the case then you can use the following commands to profile the execution of your simulation script.
You simply need to replace the `myscript` name with your script name.

`python -m cProfile -o myscript.cprof myscript.py`

`pyprof2calltree -k -i myscript.cprof`