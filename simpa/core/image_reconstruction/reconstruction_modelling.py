# The MIT License (MIT)
#
# Copyright (c) 2021 Computer Assisted Medical Interventions Group, DKFZ
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

from simpa.utils import Tags
from simpa.utils.dict_path_manager import generate_dict_path
from simpa.core.image_reconstruction.MitkBeamformingAdapter import MitkBeamformingAdapter
from simpa.core.image_reconstruction.TimeReversalAdapter import TimeReversalAdapter
from simpa.core.image_reconstruction.TestReconstructionAdapter import TestReconstructionAdapter
from simpa.core.image_reconstruction.PyTorchDASAdapter import PyTorchDASAdapter
from simpa.io_handling.io_hdf5 import save_hdf5


def perform_reconstruction(settings: dict) -> str:
    """
    This method is the main entry point to perform image reconstruction using the SIMPA toolkit.
    All information necessary for the respective reconstruction method must be contained in the
    settings dictionary.

    :param settings: a dictionary containing key-value pairs with simulation instructions.
    :returns: the path to the result data in the written HDF5 file.

    """
    reconstruction_method = None

    if ((settings[Tags.RECONSTRUCTION_ALGORITHM]
         == Tags.RECONSTRUCTION_ALGORITHM_DAS)
            or (settings[Tags.RECONSTRUCTION_ALGORITHM]
                == Tags.RECONSTRUCTION_ALGORITHM_DMAS)
            or (settings[Tags.RECONSTRUCTION_ALGORITHM]
                == Tags.RECONSTRUCTION_ALGORITHM_SDMAS)):
        reconstruction_method = MitkBeamformingAdapter()
    elif settings[
            Tags.
            RECONSTRUCTION_ALGORITHM] == Tags.RECONSTRUCTION_ALGORITHM_PYTORCH_DAS:
        reconstruction_method = PyTorchDASAdapter()
    elif settings[
            Tags.
            RECONSTRUCTION_ALGORITHM] == Tags.RECONSTRUCTION_ALGORITHM_TIME_REVERSAL:
        reconstruction_method = TimeReversalAdapter()
    elif settings[
            Tags.
            RECONSTRUCTION_ALGORITHM] == Tags.RECONSTRUCTION_ALGORITHM_TEST:
        reconstruction_method = TestReconstructionAdapter()

    reconstruction = reconstruction_method.simulate(settings)

    reconstruction_output_path = generate_dict_path(Tags.RECONSTRUCTED_DATA, settings[Tags.WAVELENGTH])

    save_hdf5({Tags.RECONSTRUCTED_DATA: reconstruction}, settings[Tags.SIMPA_OUTPUT_PATH],
              reconstruction_output_path)

    return reconstruction_output_path
