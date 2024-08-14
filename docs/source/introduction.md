# Getting started

In order to use SIMPA in your project, SIMPA has to be installed as well as the external tools that make the actual simulations possible.
Finally, to connect everything, SIMPA has to find all the binaries of the simulation modules you would like to use.
The SIMPA path management takes care of that.

* [SIMPA installation instructions](#simpa-installation-instructions)
* [External tools installation instructions](#external-tools-installation-instructions)
* [Path Management](#path-management)
* [Testing](#run-manual-tests)

## SIMPA installation instructions

The recommended way to install SIMPA is a manual installation from the GitHub repository, please follow steps 1 - 3:

1. `git clone https://github.com/IMSY-DKFZ/simpa.git`
2. `cd simpa`
3. `git checkout main`
4. `git pull`

Now open a python instance in the 'simpa' folder that you have just downloaded. Make sure that you have your preferred
virtual environment activated (we also recommend python 3.10)
1. `pip install .`
2. Test if the installation worked by using `python` followed by `import simpa` then `exit()`

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

Download the latest nightly build of [mcx](http://mcx.space/) on [this page](http://mcx.space/nightly/github/) for your operating system:

- Linux: `mcx-linux-x64-github-latest.zip`
- MacOS: `mcx-macos-x64-github-latest.zip`
- Windows: `mcx-windows-x64-github-latest.zip`

Then extract the files and set `MCX_BINARY_PATH=/.../mcx/bin/mcx` in your path_config.env.

### k-Wave (Acoustic Forward Model)

Please follow the following steps and use the k-Wave install instructions 
for further (and much better) guidance under:

[http://www.k-wave.org/](http://www.k-wave.org/)

1. Install MATLAB with the core, image processing and parallel computing toolboxes activated at the minimum.
2. Download the kWave toolbox (version >= 1.4)
3. Add the kWave toolbox base path to the toolbox paths in MATLAB
4. If wanted: Download the CPP and CUDA binary files and place them in the k-Wave/binaries folder
5. Note down the system path to the `matlab` executable file.

## Path management

As a pipelining tool that serves as a communication layer between different numerical forward models and
processing tools, SIMPA needs to be configured with the paths to these tools on your local hard drive.
You have a couple of options to define the required path variables. 
### Option 1: 
Ensure that the environment variables defined in `simpa_examples/path_config.env.example` are accessible to your script during runtime. This can be done through any method you prefer, as long as the environment variables are accessible through `os.environ`. 
### Option 2:
Import the `PathManager` class to your project using
`from simpa.utils import PathManager`. If a path to a `.env` file is not provided, the `PathManager` looks for a `path_config.env` file (just like the
one we provided in the `simpa_examples/path_config.env.example`) in the following places, in this order:
1. The optional path you give the PathManager
2. Your $HOME$ directory
3. The current working directory
4. The SIMPA home directory path
   
For this option, please follow the instructions in the `simpa_examples/path_config.env.example` file. 

## Run manual tests
To check the success of your installation ot to assess how your contributions affect the Simpa simulation outcomes, you can run the manual tests automatically. Install the testing requirements by doing `pip install .[testing]` and run the `simpa_tests/manual_tests/generate_overview.py` file. This script runs all manual tests and generates both a markdown and an HTML file that compare your results with the reference results.

# Simulation examples

To get started with actual simulations, SIMPA provides an [example package](simpa_examples) of simple simulation 
scripts to build your custom simulations upon. The [minimal optical simulation](minimal_optical_simulation.py)
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

