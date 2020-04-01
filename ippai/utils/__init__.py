# The MIT License (MIT)
#
# Copyright (c) 2018 Computer Assisted Medical Interventions Group, DKFZ
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# First load everything without internal dependencies
from ippai.utils.tags import Tags
from ippai.utils.libraries.literature_values import MorphologicalTissueProperties
from ippai.utils.libraries.literature_values import StandardProperties
from ippai.utils.libraries.literature_values import OpticalTissueProperties

# Then load classes and methods with an <b>increasing</b> amount of internal dependencies.
# If there are import errors in the tests, it is probably due to an incorrect
# initialization order
from ippai.utils.libraries.spectra_library import AbsorptionSpectrumLibrary
from ippai.utils.libraries.spectra_library import AbsorptionSpectrum
from ippai.utils.libraries.spectra_library import SPECTRAL_LIBRARY
from ippai.utils.libraries.spectra_library import view_absorption_spectra

from ippai.utils.libraries.chromophore_library import Chromophore
from ippai.utils.libraries.chromophore_library import ChromophoreLibrary
from ippai.utils.libraries.chromophore_library import CHROMOPHORE_LIBRARY

from ippai.utils.libraries.tissue_library import TissueLibrary
from ippai.utils.libraries.tissue_library import TISSUE_LIBRARY
from ippai.utils.libraries.tissue_library import TissueSettingsGenerator

from ippai.utils.calculate import calculate_oxygenation
from ippai.utils.calculate import calculate_gruneisen_parameter_from_temperature
from ippai.utils.calculate import randomize
from ippai.utils.calculate import randomize_uniform

if __name__ == "__main__":
    view_absorption_spectra()
