# The MIT License (MIT)
#
# Copyright (c) 2018 Computer Assisted Medical Interventions Group, DKFZ
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated simpa_documentation files (the "Software"), to deal
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

from simpa.utils import Tags, SaveFilePaths
from simpa.core.image_reconstruction.MitkBeamformingAdapter import MitkBeamformingAdapter
from simpa.core.image_reconstruction.TimeReversalAdapter import TimeReversalAdapter
from simpa.core.image_reconstruction.TestReconstructionAdapter import TestReconstructionAdapter
from simpa.io_handling.io_hdf5 import save_hdf5


def perform_reconstruction(settings, distortion):
    print("ACOUSTIC FORWARD")

    reconstruction_method = None

    if ((settings[Tags.RECONSTRUCTION_ALGORITHM] == Tags.RECONSTRUCTION_ALGORITHM_DAS) or
        (settings[Tags.RECONSTRUCTION_ALGORITHM] == Tags.RECONSTRUCTION_ALGORITHM_DMAS) or
            (settings[Tags.RECONSTRUCTION_ALGORITHM] == Tags.RECONSTRUCTION_ALGORITHM_SDMAS)):
        reconstruction_method = MitkBeamformingAdapter()
    elif settings[Tags.RECONSTRUCTION_ALGORITHM] == Tags.RECONSTRUCTION_ALGORITHM_TIME_REVERSAL:
        reconstruction_method = TimeReversalAdapter()
    elif settings[Tags.RECONSTRUCTION_ALGORITHM] == Tags.RECONSTRUCTION_ALGORITHM_TEST:
        reconstruction_method = TestReconstructionAdapter()

    reconstruction = reconstruction_method.simulate(settings, distortion)

    reconstruction_output_path = SaveFilePaths.RECONSTRCTION_OUTPUT.\
        format(Tags.ORIGINAL_DATA, settings[Tags.WAVELENGTH])

    if Tags.PERFORM_UPSAMPLING in settings:
        if settings[Tags.PERFORM_UPSAMPLING]:
            reconstruction_output_path = \
                SaveFilePaths.RECONSTRCTION_OUTPUT.format(Tags.UPSAMPLED_DATA, settings[Tags.WAVELENGTH])
    save_hdf5({Tags.RECONSTRUCTED_DATA: reconstruction}, settings[Tags.SIMPA_OUTPUT_PATH],
              reconstruction_output_path)

    return reconstruction_output_path
