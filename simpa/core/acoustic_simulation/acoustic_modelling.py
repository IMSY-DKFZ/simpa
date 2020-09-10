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

from simpa.utils import Tags, SaveFilePaths
from simpa.core.acoustic_simulation import k_wave_adapter
from simpa.core.acoustic_simulation import test_acoustic_adapter
from simpa.io_handling.io_hdf5 import save_hdf5


def run_acoustic_forward_model(settings):
    print("ACOUSTIC FORWARD")

    adapter = None

    if Tags.ACOUSTIC_MODEL in settings:
        if settings[Tags.ACOUSTIC_MODEL] == Tags.ACOUSTIC_MODEL_K_WAVE:
            adapter = k_wave_adapter
        elif settings[Tags.ACOUSTIC_MODEL] == Tags.ACOUSTIC_MODEL_TEST:
            adapter = test_acoustic_adapter

    data = adapter.simulate(settings)

    acoustic_output_path = SaveFilePaths.ACOUSTIC_OUTPUT.\
        format(Tags.UPSAMPLED_DATA, settings[Tags.WAVELENGTH])

    if Tags.PERFORM_UPSAMPLING in settings:
        if settings[Tags.PERFORM_UPSAMPLING]:
            acoustic_output_path = SaveFilePaths.ACOUSTIC_OUTPUT.\
                format(Tags.UPSAMPLED_DATA, settings[Tags.WAVELENGTH])

    save_hdf5({Tags.TIME_SERIES_DATA: data}, settings[Tags.SIMPA_OUTPUT_PATH],
              acoustic_output_path)

    return acoustic_output_path
