[![Documentation Status](https://readthedocs.org/projects/simpa/badge/?version=develop)](https://simpa.readthedocs.io/en/develop/?badge=develop)
[![Build Status](https://ci.mitk.org/buildStatus/icon?job=SIMPA%2FUnit+Tests+master)](https://ci.mitk.org/job/SIMPA/job/Unit%20Tests%20master/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/CAMI-DKFZ/simpa/blob/master/LICENSE.md)

![Logo](docs/source/images/simpa_logo.png?raw=true "Logo")

# The toolkit for Simulation and Image Processing for Photonics and Acoustics (SIMPA)

SIMPA aims to facilitate realistic image simulation for optical and acoustic imaging modalities by
providing adapters to crucial modelling steps, such as volume generation; optical modelling; acoustic
modelling; and image reconstruction. SIMPA provides a communication layer between various modules
that implement optical and acoustic forward and inverse models.
Non-experts can use the toolkit to create sensible simulations from default parameters in an end-to-end fashion. Domain experts are provided with the functionality to set up a highly customisable
pipeline according to their specific use cases and tool requirements.

* [Getting started](#getting-started)
* [Simulation examples](#simulation-examples)
* [Documentation](#documentation)
* [Contributing](#how-to-contribute)
* [Performance profiling](#performance-profiling)
* [Troubleshooting](#troubleshooting)

The toolkit is still under development and is thus not fully tested and may contain bugs. 
Please report any issues that you find in our Issue Tracker: https://github.com/CAMI-DKFZ/simpa/issues. 
Also make sure to double check all value ranges of the optical and acoustic tissue properties 
and to assess all simulation results for plausibility.

# Getting started

In order to use SIMPA in your project, SIMPA has to be installed as well as the external tools that make the actual simulations possible.
Finally, to connect everything, SIMPA has to find all the binaries of the simulation modules you would like to use.
The SIMPA path management takes care of that.

* [SIMPA installation instructions](#simpa-installation-instructions)
* [External tools installation instructions](#external-tools-installation-instructions)
* [Path Management](#path-management)

## SIMPA installation instructions

The recommended way to install SIMPA is a manual installation from the GitHub repository, please follow steps 1 - 3:

1. `git clone https://github.com/CAMI-DKFZ/simpa.git`
2. `cd simpa`
3. `git checkout master`
4. `git pull`

Now open a python instance in the 'simpa' folder that you have just downloaded. Make sure that you have your preferred
virtual environment activated (we also recommend python 3.8)
1. `pip install -r requirements.txt`
2. `python setup.py install` (for developement: `python setup.py develop`)
3. Test if the installation worked by using `python` followed by `import simpa` then `exit()`

If no error messages arise, you are now setup to use SIMPA in your project.

You can also install SIMPA with pip. Simply run:

`pip install simpa`

You also need to manually install the pytorch library to use all features of SIMPA.
To this end, use the pytorch website tool to figure out which version to install:
[https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)

## External tools installation instructions

In order to get the full SIMPA functionality, you should install all third party toolkits that make the optical and 
acoustic simulations possible. 

### mcx (Optical Forward Model)

Either download suitable executables or build yourself from the following sources:

[http://mcx.space/](http://mcx.space/)

In order to obtain access to all custom sources that we implemented, please build mcx yourself from the
following mcx Github fork:
[https://github.com/CAMI-DKFZ/mcx](https://github.com/CAMI-DKFZ/mcx)

For the installation, please follow the instructions from the original repository.
Please note that there might be compatibility issues using mcx-cl with the MCX Adapter as this use case is not 
being tested and supported by the SIMPA developers.

### k-Wave (Acoustic Forward Model)

Please follow the following steps and use the k-Wave install instructions 
for further (and much better) guidance under:

[http://www.k-wave.org/](http://www.k-wave.org/)

1. Install MATLAB with the core and parallel computing toolboxes activated at the minimum.
2. Download the kWave toolbox
3. Add the kWave toolbox base path to the toolbox paths in MATLAB
4. Download the kWaveArray addition from the link given in this user forum post [http://www.k-wave.org/forum/topic/alpha-version-of-kwavearray-off-grid-sources](http://www.k-wave.org/forum/topic/alpha-version-of-kwavearray-off-grid-sources)
5. Add the kWaveArray folder to the toolbox paths in MATLAB as well
6. If wanted: Download the CPP and CUDA binary files and place them in the k-Wave/binaries folder
7. Note down the system path to the `matlab` executable file.

## Path management

As a pipelining tool that serves as a communication layer between different numerical forward models and
processing tools, SIMPA needs to be configured with the paths to these tools on your local hard drive.
To this end, we have implemented the `PathManager` class that you can import to your project using
`from simpa.utils import PathManager`. The PathManager looks for a `path_config.env` file (just like the
one we provided in the `simpa_examples`) in the following places in this order:
1. The optional path you give the PathManager
2. Your $HOME$ directory
3. The current working directory
4. The SIMPA home directory path

Please follow the instructions in the `path_config.env` file in the `simpa_examples` folder. 

# Simulation examples

To get started with actual simulations, SIMPA provides an [example package](simpa_examples) of simple simulation 
scripts to build your custom simulations upon. The [minimal optical simulation](simpa_examples/minimal_optical_simulation.py)
is a nice start if you have MCX installed.

Generally, the following pseudo code demonstrates the construction and run of a simulation pipeline:

```python
import simpa as sp

# Create general settings 
settings = sp.Settings(general_settings)

# Create specific settings for each pipeline element 
# in the simulation pipeline
settings.set_volume_creation_settings(volume_creation_settings)
settings.set_optical_settings(optical_settings)
settings.set_acoustic_settings(acoustic_settings)
settings.set_reconstruction_settings(reconstruction_settings)

# Set the simulation pipeline
simulation_pipeline = [sp.VolumeCreatorModule(settings),
    sp.OpticalForwardModule(settings),
    sp.AcousticForwardModule(settings),
    sp.ReconstructionModule(settings)]
    
# Choose a PA device with device position in the volume
device = sp.CustomDevice()

# Simulate the pipeline
sp.simulate(simulation_pipeline, settings, device)
```

# Documentation

The updated version of the SIMPA documentation can be found at [https://simpa.readthedocs.io/en/develop](https://simpa.readthedocs.io/en/develop).

## Building the documentation

It is also easily possible to build the SIMPA documentation from scratch.
When the installation succeeded, and you want to make sure that you have the latest documentation
you should do the following steps in a command line:

1. Navigate to the `simpa/docs` directory
2. If you would like the documentation to have the https://readthedocs.org/ style, type `pip install sphinx-rtd-theme`
3. Type `make html`
4. Open the `index.html` file in the `simpa/docs/build/html` directory with your favourite browser.

# How to contribute

Please find a more detailed description of how to contribute as well as code style references in our
[contribution guidelines](CONTRIBUTING.md).

To contribute to SIMPA, please fork the SIMPA github repository and create a pull request with a branch containing your 
suggested changes. The core developers will then review the suggested changes and integrate these into the code 
base.

Please make sure that you have included unit tests for your code and that all previous tests still run through.

There is a regular SIMPA status meeting every Friday on even calendar weeks at 10:00 CET/CEST, and you are very welcome to participate and
raise any issues or suggest new features. If you want to join this meeting, write one of the core developers.

Please see the github guidelines for creating pull requests: [https://docs.github.com/en/github/collaborating-with-issues-and-pull-requests/about-pull-requests](https://docs.github.com/en/github/collaborating-with-issues-and-pull-requests/about-pull-requests)


# Performance profiling

Do you wish to know which parts of the simulation pipeline cost the most amount of time? 
If that is the case then you can use the following commands to profile the execution of your simulation script.
You simply need to replace the `myscript` name with your script name.

`python -m cProfile -o myscript.cprof myscript.py`

`pyprof2calltree -k -i myscript.cprof`

# Troubleshooting

In this section, known problems are listed with their solutions (if available):

## 1. Error reading hdf5-files when using k-Wave binaries:
   
If you encounter an error similar to:

    Error using h5readc
    The filename specified was either not found on the MATLAB path or it contains unsupported characters.

Look up the solution in [this thread of the k-Wave forum](http://www.k-wave.org/forum/topic/error-reading-h5-files-when-using-binaries).  
      