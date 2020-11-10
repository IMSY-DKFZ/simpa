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
from simpa.core.device_digital_twins import DEVICE_MAP


def run_volume_creation(global_settings: Settings):
    print("VOLUME CREATION")

    if Tags.VOLUME_CREATOR not in global_settings:
        raise AssertionError("Tags.VOLUME_CREATOR tag was not specified in the settings. Skipping optical modelling.")

    model = global_settings[Tags.VOLUME_CREATOR]
    volume_creator_adapter = None

    if model == Tags.VOLUME_CREATOR_VERSATILE:
        volume_creator_adapter = VersatileVolumeCreator()

    pa_device = None
    if Tags.DIGITAL_DEVICE in global_settings:
        try:
            pa_device = DEVICE_MAP[global_settings[Tags.DIGITAL_DEVICE]]
            pa_device.check_settings_prerequisites(global_settings)
        except KeyError:
            pa_device = None

    volumes = volume_creator_adapter.create_simulation_volume(global_settings)

    if pa_device is not None:
        volumes, global_settings = pa_device.adjust_simulation_volume_and_settings(volumes, global_settings)

    volume_path = SaveFilePaths.SIMULATION_PROPERTIES \
        .format(Tags.ORIGINAL_DATA, str(global_settings[Tags.WAVELENGTH]))
    save_hdf5(volumes, global_settings[Tags.SIMPA_OUTPUT_PATH], file_dictionary_path=volume_path)

    return volume_path
