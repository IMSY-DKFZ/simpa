===================================
SIMPA
===================================

.. image:: images/simpa_logo.png
   :width: 12cm
   :align: center

Welcome to the SIMPA documentation!

The toolkit for Simulation and Image Processing for Photonics and Acoustics (SIMPA) aims to facilitate realistic image simulation for optical and acoustic imaging modalities by
providing adapters to crucial modelling steps, such as volume generation; optical modelling; acoustic
modelling; and image reconstruction. SIMPA provides a communication layer between various modules
that implement optical and acoustic forward and inverse models.
Non-experts can use the toolkit to create sensible simulations from default parameters in an end-to-end fashion. Domain experts are provided with the functionality to set up a highly customisable
pipeline according to their specific use cases and tool requirements.

.. toctree::
   :maxdepth: 1
   :caption: SIMPA toolkit

   intro_link
   contributing_link

===================================

.. toctree::
   :maxdepth: 2
   :caption: SIMPA examples

   simpa_examples

===================================

.. toctree::
   :maxdepth: 3
   :caption: Package reference

   simpa.core
   simpa.io_handling
   simpa.log
   simpa.utils
   simpa.visualisation
