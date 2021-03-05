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
from simpa.core.acoustic_simulation.k_wave_adapter import KwaveAcousticForwardModel
from simpa.core.acoustic_simulation.test_acoustic_adapter import TestAcousticAdapter
from simpa.io_handling.io_hdf5 import save_hdf5
from simpa.utils.dict_path_manager import generate_dict_path


def run_acoustic_forward_model(settings):
    """
    This method is the entry method for running an acoustic forward model.
    It is invoked in the *simpa.core.simulation.simulate* method, but can also be called
    individually for the purposes of performing acoustic forward modeling only or in a different context.

    The concrete will be chosen based on the::

        Tags.ACOUSTIC_MODEL

    tag in the settings dictionary.

    :param settings: The settings dictionary containing key-value pairs that determine the simulation.
        Here, it must contain the Tags.ACOUSTIC_MODEL tag and any tags that might be required by the specific
        acoustic model.
    :raises AssertionError: an assertion error is raised if the Tags.ACOUSTIC_MODEL tag is not given or
        points to an unknown acoustic forward model.
    :return: returns the path to the simulated data within the saved HDF5 container.
    """
    adapter = None

    if Tags.ACOUSTIC_MODEL in settings:
        if settings[Tags.ACOUSTIC_MODEL] == Tags.ACOUSTIC_MODEL_K_WAVE:
            adapter = KwaveAcousticForwardModel()
        elif settings[Tags.ACOUSTIC_MODEL] == Tags.ACOUSTIC_MODEL_TEST:
            adapter = TestAcousticAdapter()
    else:
        raise AssertionError("The ACOUSTIC_MODEL tag was not defined in the dictionary")

    if adapter is None:
        raise AssertionError("No known acoustic model was specified by the ACOUSTIC_MODEL tag.")

    data = adapter.simulate(settings)

    acoustic_output_path = generate_dict_path(Tags.TIME_SERIES_DATA, wavelength=settings[Tags.WAVELENGTH])

    save_hdf5(data, settings[Tags.SIMPA_OUTPUT_PATH], acoustic_output_path)

    return acoustic_output_path
