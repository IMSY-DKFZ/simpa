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

