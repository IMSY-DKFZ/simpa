
Welcome to IPPAI's documentation!
=================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

.. image:: images/ippai_logo.png
    :width: 6cm

.. mdinclude::  ../../README.md

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

.. automodule:: ippai.utils
.. autoclass:: ippai.utils.MorphologicalTissueProperties
.. autoclass:: ippai.utils.OpticalTissueProperties
.. autoclass:: ippai.utils.StandardProperties

.. autoclass:: ippai.utils.AbsorptionSpectrum
    :members:
.. autoclass:: ippai.utils.AbsorptionSpectrumLibrary
    :members:
.. autofunction:: ippai.utils.view_absorption_spectra

.. autoclass::  Chromophore
    :members:
.. autoclass::  ChromophoreLibrary
    :members:
.. autoclass::  TissueLibrary
    :members:
.. autoclass::  TissueSettingsGenerator
    :members:
