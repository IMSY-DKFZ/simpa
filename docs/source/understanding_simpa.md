# Understanding SIMPA

## Understanding Tags

### What are Tags?

In SIMPA, Tags are identifiers used to specify and categorize various settings and components within the simulation.
They act as keys in the configuration dictionaries, enabling a clear and organized way to define simulation parameters.
Tags ensure that the configuration is modular, readable, and easy to manage.

### Purpose of Tags

- **Organization**: Tags help in structuring the configuration settings systematically.
- **Flexibility**: They allow users to easily modify and extend configurations.
- **Reusability**: Tags facilitate the reuse of settings across different simulations.

### How Tags Work

Tags are used to identify different components and their settings within the configuration dictionaries. Each component
has a predefined set of tags associated with it. These tags are used to specify the parameters and properties of the 
components.

### What Tags are Available?

The list of Tags available in SIMPA is very extensive (see [simpa.utils](simpa.utils.rst) for full list), due to the
level of customisation available to the user. To get to grips with the more commonly used Tags, we highly recommend
consulting the [examples](simpa_examples.rst).

## Concept of Settings

Settings in SIMPA are configurations that control the behavior of the simulation. They are used to specify parameters
and options for both the overall simulation and individual components of the simulation pipeline. Proper configuration
of these settings is crucial for accurate and efficient simulations. This documentation provides a foundational
understanding of these settings, allowing users to customize their simulations effectively. For more detailed
information on specific settings and components, users are encouraged to refer to the source code and additional
documentation provided within the SIMPA repository.

### Global Settings

Global settings apply to the entire simulation and include parameters that are relevant across multiple components.
These settings typically encompass general simulation properties such as physical constants and overarching simulation
parameters.

#### Example of Global Settings

- `Tags.SPACING_MM`: The voxel spacing in the simulation.
- `Tags.GPU`: Whether there is a GPU available to perform the computation.
- `Tags.WAVELENGTHS`: The wavelengths that will later be simulated.

### Component Settings

Component settings are specific to individual components within the simulation pipeline. Each component can have its own
set of settings that determine how it behaves. These settings allow for fine-grained control over the simulation
process, enabling customization and optimization for specific experimental conditions.

#### Difference Between Global and Component Settings

- **Scope**:
  - Global settings affect the entire simulation framework.
  - Component settings only influence the behavior of their respective components.
  
- **Usage**:
  - Global settings are defined once and applied universally.
  - Component settings are defined for each component individually, allowing for component-specific customization.

#### Implementation
For a given simulation, the overall simulation settings will first be created from the global settings. Then, each
components setting will be added. Overall, a dictionary instance will be created with all of the global settings as well
as the components settings as sub-dictionaries.

## Available Component Settings

The following list describes the available component settings for various components in the SIMPA framework. Each component may have a unique set of settings that control its behavior.

### 1. Volume Creation

Settings for the volume creation component, which defines the method used to create the simulation volume; and therefore
ultimately decides the properties of the simulation volume. It is added to the simulation settings using:
[set_volume_creator_settings](../../simpa/utils/settings.py).

#### Examples of Volume Creation Settings
- `Tags.STRUCTURES`: When using the model based volume creation adapter, sets the structures to be fill the volume.
- `Tags.INPUT_SEGMENTATION_VOLUME`: When using the segmentation based volume creation adapter, the segmentation mapping
will be specified under this tag.

### 2. Acoustic Model

Settings for the acoustic forward model component, which simulates the propagation of acoustic waves. It is added to the
simulation settings using: [set_acoustic_settings](../../simpa/utils/settings.py).

#### Examples of Acoustic Settings
- `Tags.KWAVE_PROPERTY_ALPHA_POWER`: The exponent in the exponential acoustic attenuation law of k-Wave.
- `Tags.RECORDMOVIE`: If true, a movie of the k-Wave simulation will be recorded.

### 3. Optical Model

Settings for the optical model component, which simulates the propagation of light through the medium. It is added to 
the simulation settings using: [set_optical_settings](../../simpa/utils/settings.py).

#### Examples of Optical Settings
- `Tags.OPTICAL_MODEL_NUMBER_PHOTONS`: The number of photons used in the optical simulation.
- `Tags.LASER_PULSE_ENERGY_IN_MILLIJOULE`: The laser pulse energy used in the optical simulation.

### 4. Reconstruction model

Settings for the reconstruction model, which reconstructs the image from the simulated signals. It is added to the
simulation settings using: [set_reconstruction_settings](../../simpa/utils/settings.py).

#### Examples of Reconstruction Settings

- `Tags.RECONSTRUCTION_BMODE_AFTER_RECONSTRUCTION`: Specifies whether an envelope detection should be performed after
reconstruction.
- `Tags.RECONSTRUCTION_PERFORM_BANDPASS_FILTERING`: Whether bandpass filtering should be applied or not.
