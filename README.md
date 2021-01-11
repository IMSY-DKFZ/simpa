#README

The Simulation and Image Processing for Photoacoustic Imaging (SIMPA) toolkit.

## SIMPA Install Instructions

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

## External Tools installation instructions

### mcx (Optical Forward Model)

Either download suitable executables or build yourself from the following sources:
http://mcx.space/

### k-Wave (Acoustic Forward Model)

Please follow the following steps and use the k-Wave install instructions 
for further (and much better) guidance under http://www.k-wave.org/!

1. Install MATLAB with the core and parallel computing toolboxes activated at the minimum.
2. Download the kWave toolbox
3. Add the kWave toolbox base bath to the toolbox paths in MATLAB
4. If wanted: Download the CPP and CUDA binary files and place them inthe k-Wave/binaries folder
5. Note down the system path to the `matlab` executable file. 

On MATLAB r2020a or newer there is a bug when using the GPU binaries with kWave. Please follow these instructions
http://www.k-wave.org/forum/topic/error-reading-h5-files-when-using-binaries to fix this bug.

### MITK

## Overview

The main use case for the simpa framework is the simulation of photoacoustic images.
However, it can also be used for image processing.

### Simulating photoacoustic images

A basic example on how to use simpa in you project to run an optical forward simulation is given in the 
samples/minimal_optical_simulation.py file.

## Performance profiling

Do you wish to know which parts of the simulation pipeline cost the most amount of time? 
If that is the case then you can use the following commands to profile the execution of your simulation script.
You simply need to replace the `myscript` name with your script name.

`python -m cProfile -o myscript.cprof myscript.py`

`pyprof2calltree -k -i myscript.cprof`