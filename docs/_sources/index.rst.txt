
Welcome to the SIMPA documentation!
===================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

.. image:: images/simpa_logo.png
    :width: 12cm

.. mdinclude::  ../../README.md

.. mdinclude::  ../../developer_guide.md

Class references
================

This description details the three principle modules
of the SIMPA toolkit and gives an insight into their constituents.
The core is concerned with providing interfaces for the
simulation tools, while the utils module contains many scripts
and classes to facilitate the use of the simulation pipeline.

Module: core
-------------------

The purpose of the core module is to provide interfaces that facilitate the integration of toolboxes and code for
photoacoustic modeling into a single continuous pipeline.

.. automodule:: simpa.core.simulation
    :members:

Volume creation
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The core contribution of the SIMPA toolkit is the creation of in silico tissue-mimicking
phantoms. This feature is represented by the volume_creation module, that two main volume creation modules:

 - Model-based creation of volumes using a set of rules
 - Segmentation-based creation of volumes

.. automodule:: simpa.core.volume_creation_module
    :members:

Model-based volume creation
""""""""""""""""""""""""""""

.. automodule:: simpa.core.volume_creation_module.volume_creation_module_model_based_adapter
    :members:

Segmentation-based volume creation
"""""""""""""""""""""""""""""""""""

.. automodule:: simpa.core.volume_creation_module.volume_creation_module_segmentation_based_adapter
    :members:

Optical forward modelling
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: simpa.core.optical_simulation_module
    :members:

mcx integration
""""""""""""""""""

.. automodule:: simpa.core.optical_simulation_module.optical_forward_model_mcx_adapter
    :members:

Acoustic forward modelling
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: simpa.core.acoustic_forward_module
    :members:

k-Wave integration
"""""""""""""""""""

.. automodule:: simpa.core.acoustic_forward_module.acoustic_forward_module_k_wave_adapter
    :members:

Image reconstruction
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: simpa.core.reconstruction_module
    :members:

.. automodule:: simpa.core.reconstruction_module.reconstruction_utils
    :members:

Backprojection
"""""""""""""""""""""""""""""""

.. automodule:: simpa.core.reconstruction_module.reconstruction_module_delay_and_sum_adapter
    :members:

Delay-Multiply-And-Sum (DMAS)
"""""""""""""""""""""""""""""""

.. automodule:: simpa.core.reconstruction_module.reconstruction_module_delay_multiply_and_sum_adapter
    :members:

signed Delay-Multiply-And-Sum (sDMAS)
"""""""""""""""""""""""""""""""""""""

.. automodule:: simpa.core.reconstruction_module.reconstruction_module_signed_delay_multiply_and_sum_adapter
    :members:

Time Reversal
""""""""""""""

.. automodule:: simpa.core.reconstruction_module.reconstruction_module_time_reversal_adapter
    :members:

Processing Components
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: simpa.core.processing_components
    :members:

Noise Models
"""""""""""""""""""""""""""""""

.. automodule:: simpa.core.processing_components.noise.gamma_noise
    :members:

.. automodule:: simpa.core.processing_components.noise.gaussian_noise
    :members:

.. automodule:: simpa.core.processing_components.noise.poisson_noise
    :members:

.. automodule:: simpa.core.processing_components.noise.salt_and_pepper_noise
    :members:

.. automodule:: simpa.core.processing_components.noise.uniform_noise
    :members:

Digital device twins
^^^^^^^^^^^^^^^^^^^^^

At every step along the forward simulation, knowledge of the photoacoustic device that is used for the measurements is
needed. This is important to reflect characteristic artefacts and challenges for the respective device.

To this end, we have included digital twins of commonly used devices into the SIMPA core. Additionally, we have included
detection geometries and illumination geometries that can be used to create custom photoacoustic devices for simulation.

Detection Geometries
"""""""""""""""""""""

.. automodule:: simpa.core.device_digital_twins.devices.detection_geometries.detection_geometry_base
    :members:

.. automodule:: simpa.core.device_digital_twins.devices.detection_geometries.curved_array
    :members:

.. automodule:: simpa.core.device_digital_twins.devices.detection_geometries.linear_array
    :members:

.. automodule:: simpa.core.device_digital_twins.devices.detection_geometries.planar_array
    :members:

Illumination Geometries
""""""""""""""""""""""""

.. automodule:: simpa.core.device_digital_twins.devices.illumination_geometries.illumination_geometry_base
    :members:

.. automodule:: simpa.core.device_digital_twins.devices.illumination_geometries.pencil_array_illumination
    :members:

.. automodule:: simpa.core.device_digital_twins.devices.illumination_geometries.pencil_beam_illumination
    :members:

.. automodule:: simpa.core.device_digital_twins.devices.illumination_geometries.slit_illumination
    :members:


Models of real world devices
""""""""""""""""""""""""""""

.. automodule:: simpa.core.device_digital_twins.devices.pa_devices.ithera_msot_acuity
    :members:

.. automodule:: simpa.core.device_digital_twins.devices.pa_devices.ithera_msot_invision
    :members:

.. automodule:: simpa.core.device_digital_twins.devices.pa_devices.ithera_rsom
    :members:

Module: utils
-------------

The utils module contains several general-purpose utility functions whose purpose it is to facilitate
the use of SIMPA. The most important of these is the Tags class, which defines the strings and data types
that have to be used for the keys and values of the settings dictionary.

.. automodule:: simpa.utils.calculate
    :members:

.. automodule:: simpa.utils.constants
    :members:

.. automodule:: simpa.utils.deformation_manager
    :members:

.. automodule:: simpa.utils.dict_path_manager
    :members:

.. automodule:: simpa.utils.path_manager
    :members:

.. automodule:: simpa.utils.settings
    :members:

.. automodule:: simpa.utils.tags
    :members:

.. automodule:: simpa.utils.tissue_properties
    :members:

Libraries
^^^^^^^^^^^^^^^^^^^^^

Another important aspect of the utils class is the libraries that are being provided.
These contain compilations of literature values for the acoustic and optical properties
of commonly used tissue.

.. automodule:: simpa.utils.libraries
    :members:
.. automodule:: simpa.utils.libraries.molecule_library
    :members:
.. automodule:: simpa.utils.libraries.literature_values
    :members:
.. automodule:: simpa.utils.libraries.spectra_library
    :members:
.. automodule:: simpa.utils.libraries.tissue_library
    :members:
.. automodule:: simpa.utils.libraries.structure_library
    :members:

Module: io_handling
-------------------

.. automodule:: simpa.io_handling
    :members:

.. automodule:: simpa.io_handling.io_hdf5
    :members:

.. automodule:: simpa.io_handling.serialization
    :members:

Module: log
-------------------

.. automodule:: simpa.log
    :members:

.. automodule:: simpa.log.file_logger
    :members:


Examples
================

Performing an optical forward simulation
--------------------------------------------------------------------------------------------------------------------

The file can be found in simpa_examples/minimal_optical_simulation.py:

.. literalinclude:: ../../simpa_examples/minimal_optical_simulation.py
    :language: python
    :lines: 1-

Performing a complete forward simulation with acoustic modeling, optical modeling, as well as image reconstruction
--------------------------------------------------------------------------------------------------------------------

The file can be found in simpa_examples/optical_and_acoustic_simulation.py:

.. literalinclude:: ../../simpa_examples/optical_and_acoustic_simulation.py
    :language: python
    :lines: 1-

Defining custom tissue structures and properties
------------------------------------------------

The file can be found in simpa_examples/create_custom_tissues.py:

.. literalinclude:: ../../simpa_examples/create_custom_tissues.py
    :language: python
    :lines: 1-

Defining a custom digital device twin class
------------------------------------------------

The file can be found in simpa_examples/create_a_custom_digital_device_twin.py:

.. literalinclude:: ../../simpa_examples/create_a_custom_digital_device_twin.py
    :language: python
    :lines: 1-

Defining custom tissue types
------------------------------------------------

The file can be found in simpa_examples/create_custom_tissues.py:

.. literalinclude:: ../../simpa_examples/create_custom_tissues.py
    :language: python
    :lines: 1-

Load a segmentation mask and use it to simulate
------------------------------------------------

The file can be found in simpa_examples/segmentation_loader.py:

.. literalinclude:: ../../simpa_examples/segmentation_loader.py
    :language: python
    :lines: 1-