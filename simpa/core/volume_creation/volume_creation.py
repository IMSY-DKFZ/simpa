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

from simpa.utils.settings_generator import Settings
from simpa.utils import Tags, SaveFilePaths
from simpa.io_handling import save_hdf5
from simpa.core.volume_creation.versatile_volume_creator import VersatileVolumeCreator


def run_volume_creation(settings: Settings):
    print("VOLUME CREATION")

    if Tags.VOLUME_CREATOR not in settings:
        raise AssertionError("Tags.VOLUME_CREATOR tag was not specified in the settings. Skipping optical modelling.")

    model = settings[Tags.VOLUME_CREATOR]
    volume_creator_adapter = None

    if model == Tags.VOLUME_CREATOR_VERSATILE:
        volume_creator_adapter = VersatileVolumeCreator()

    volumes = volume_creator_adapter.create_simulation_volume(settings)

    volume_path = SaveFilePaths.SIMULATION_PROPERTIES \
        .format(Tags.ORIGINAL_DATA, str(settings[Tags.WAVELENGTH]))
    save_hdf5(volumes, settings[Tags.SIMPA_OUTPUT_PATH], file_dictionary_path=volume_path)

    return volume_path
