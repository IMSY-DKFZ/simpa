# Developer Guide

Dear IPPAI developers.
This Guide is meant to be used as a collection of How-To's to contribute to the framework.

## Adding literature absorption spectra

 The central point, where absorption spectra are collected and handled is in 
 `ippai.utils.libraries.spectra_library.py`. The file comprises the class `AbsorptionSpectrumLibrary`, in which the
 new absorption spectra can be added using the following two steps:
 
 1. In the beginning of the class, there is a bunch of constants that define spectra using the `AbsorptionSpectrum`
 class. Add a new constant here: `NEW_SPECTRUM = AbsorptionSpectrum(absorber_name, wavelengths, absorptions)`. By
 convention, the naming of the constant should be the same as the `absorber_name` field. The `wavelengths` and
 `absorptions` arrays must be of the same length and contain corresponding values.
 
 2. In the `__init__` method of the `AbsorptionSpectrumLibrary` class, the class constants are added to an internal
 list. This has the benefit of enabling the Library class to be iterable. Add your newly added constant field to 
 the list here.
 
 3. Your absorption spectrum is now usable throughout all of ippai and is accessible using the `SPECTRAL_LIBRARY`
 sngleton that can be imported using `from ippai.utils import SPECTRAL_LIBRARY`. 