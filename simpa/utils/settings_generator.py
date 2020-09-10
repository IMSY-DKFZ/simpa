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
from simpa.io_handling import save_hdf5, load_hdf5


class Settings(dict):
    def __init__(self, dictionary):
        super(Settings, self).__init__()
        for key, value in dictionary.items():
            self[key] = value

    def add_meta_information(self):
        self[Tags.VOLUME_NAME] = "Test_Volume"
        self[Tags.SIMULATION_PATH] = ""
        self[Tags.RANDOM_SEED] = 100
        self[Tags.SPACING_MM] = 0.1
        self[Tags.DIM_VOLUME_X_MM] = 20
        self[Tags.DIM_VOLUME_Y_MM] = 20
        self[Tags.DIM_VOLUME_Z_MM] = 20

    def add_optical_properties(self):
        self[Tags.RUN_OPTICAL_MODEL] = True
        self[Tags.WAVELENGTHS] = [800]
        self[Tags.OPTICAL_MODEL] = Tags.OPTICAL_MODEL_MCX
        self[Tags.OPTICAL_MODEL_NUMBER_PHOTONS] = 1e6
        self[Tags.ILLUMINATION_TYPE] = Tags.ILLUMINATION_TYPE_PENCIL
        self[Tags.ILLUMINATION_POSITION] = [0, 0, 0]
        self[Tags.ILLUMINATION_DIRECTION] = [0, 0.5, 0.5]

    def add_acoustic_properties(self):
        self[Tags.RUN_ACOUSTIC_MODEL] = True
        self[Tags.ACOUSTIC_MODEL] = Tags.ACOUSTIC_MODEL_K_WAVE
        self[Tags.ACOUSTIC_SIMULATION_3D] = False
        self[Tags.PROPERTY_SPEED_OF_SOUND] = 1540
        self[Tags.PROPERTY_DENSITY] = 1000

    def add_reconstruction_properties(self):
        self[Tags.PERFORM_IMAGE_RECONSTRUCTION] = True
        self[Tags.RECONSTRUCTION_ALGORITHM] = Tags.RECONSTRUCTION_ALGORITHM_DAS

    def save(self, path):
        save_hdf5(self, path)

    def load(self, path):
        for key, value in load_hdf5(path).items():
            self[key] = value
