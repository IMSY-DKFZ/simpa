
Welcome to the SIMPA documentation!
=================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

.. image:: images/simpa_logo.png
    :width: 6cm

.. mdinclude::  ../../README.md

.. mdinclude::  developer_guide.md

Examples
================

Performing a simple opical forward simulation
---------------------------------------------

The file can be found in samples/minimal_optical_simulation.py:

.. literalinclude:: ../../samples/minimal_optical_simulation.py
    :language: python
    :lines: 23-

Reading the HDF5 simulation output
----------------------------------

The file can be found in samples/access_saved_PAI_data.py:

.. literalinclude:: ../../samples/access_saved_PAI_data.py
    :language: python
    :lines: 23-

Defining custom tissue structures and properties
------------------------------------------------

The file can be found in samples/create_custom_tissues.py:

.. literalinclude:: ../../samples/create_custom_tissues.py
    :language: python
    :lines: 23-

Class references
================

Module: utils
-------------

.. automodule:: simpa.utils
    :members:

.. automodule:: simpa.utils.libraries
    :members:
.. automodule:: simpa.utils.libraries.chromophore_library
    :members:
.. automodule:: simpa.utils.libraries.literature_values
    :members:
.. automodule:: simpa.utils.libraries.spectra_library
    :members:
.. automodule:: simpa.utils.libraries.tissue_library
    :members:

Module: io_handling
-------------------

.. automodule:: simpa.io_handling
    :members:

.. automodule:: simpa.io_handling.io_hdf5
    :members:

Module: simulate
-------------------

.. automodule:: simpa.simulate
    :members:

.. automodule:: simpa.simulate.simulation
    :members:

.. automodule:: simpa.simulate.tissue_properties
    :members:

.. automodule:: simpa.simulate.volume_creator
    :members:

Conventions
^^^^^^^^^^^^^^^

.. automodule:: simpa.simulate.constants
    :members:

Structures
^^^^^^^^^^^^^^^

.. automodule:: simpa.simulate.structures
    :members:

Forward models
^^^^^^^^^^^^^^^

.. automodule:: simpa.simulate.models
    :members:

.. automodule:: simpa.simulate.models.optical_models
    :members:

.. automodule:: simpa.simulate.models.acoustic_models
    :members:

.. automodule:: simpa.simulate.models.reconstruction_models
    :members:

.. automodule:: simpa.simulate.models.noise_models
    :members:

