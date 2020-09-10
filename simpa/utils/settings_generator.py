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

    def add_minimal_meta_information(self, volume_name: str = None, simulation_path: str = None,
                                     random_seed: int = None, spacing: float = None, volume_dim_x: (int, float) = None,
                                     volume_dim_y: (int, float) = None, volume_dim_z: (int, float) = None):
        if volume_name is not None:
            self[Tags.VOLUME_NAME] = volume_name
        else:
            self[Tags.VOLUME_NAME] = "Test_Volume"

        if simulation_path is not None:
            self[Tags.SIMULATION_PATH] = simulation_path
        else:
            self[Tags.SIMULATION_PATH] = ""

        if random_seed is not None:
            self[Tags.RANDOM_SEED] = random_seed
        else:
            self[Tags.RANDOM_SEED] = 100

        if spacing is not None:
            self[Tags.SPACING_MM] = spacing
        else:
            self[Tags.SPACING_MM] = 0.1

        if volume_dim_x is not None:
            self[Tags.DIM_VOLUME_X_MM] = volume_dim_x
        else:
            self[Tags.DIM_VOLUME_X_MM] = 20

        if volume_dim_y is not None:
            self[Tags.DIM_VOLUME_Y_MM] = volume_dim_y
        else:
            self[Tags.DIM_VOLUME_Y_MM] = 20

        if volume_dim_z is not None:
            self[Tags.DIM_VOLUME_Z_MM] = volume_dim_z
        else:
            self[Tags.DIM_VOLUME_Z_MM] = 20

    def add_minimal_optical_properties(self, run_optical_model: bool = None, wavelengths: int = None,
                                       optical_model: str = None, photon_number: int = None,
                                       illumination_type: str = None, illumination_position: list = None,
                                       illumination_direction: list = None):

        if run_optical_model is not None:
            self[Tags.RUN_OPTICAL_MODEL] = run_optical_model
        else:
            self[Tags.RUN_OPTICAL_MODEL] = True

        if wavelengths is not None:
            self[Tags.WAVELENGTHS] = wavelengths
        else:
            self[Tags.WAVELENGTHS] = [800]

        if optical_model is not None:
            self[Tags.OPTICAL_MODEL] = optical_model
        else:
            self[Tags.OPTICAL_MODEL] = Tags.OPTICAL_MODEL_MCX

        if photon_number is not None:
            self[Tags.OPTICAL_MODEL_NUMBER_PHOTONS] = photon_number
        else:
            self[Tags.OPTICAL_MODEL_NUMBER_PHOTONS] = 1e6

        if illumination_type is not None:
            self[Tags.ILLUMINATION_TYPE] = illumination_type
        else:
            self[Tags.ILLUMINATION_TYPE] = Tags.ILLUMINATION_TYPE_PENCIL

        if illumination_position is not None:
            self[Tags.ILLUMINATION_POSITION] = illumination_position
        else:
            self[Tags.ILLUMINATION_POSITION] = [0, 0, 0]

        if illumination_position is not None:
            self[Tags.ILLUMINATION_DIRECTION] = illumination_direction
        else:
            self[Tags.ILLUMINATION_DIRECTION] = [0, 0.5, 0.5]

    def add_acoustic_properties(self, run_acoustic_model: bool = None, acoustic_model: str = None,
                                acoustic_simulation_3D: bool = None, speed_of_sound: (int, float) = None,
                                density: (int, float) = None):

        if run_acoustic_model is not None:
            self[Tags.RUN_ACOUSTIC_MODEL] = run_acoustic_model
        else:
            self[Tags.RUN_ACOUSTIC_MODEL] = True

        if acoustic_model is not None:
            self[Tags.ACOUSTIC_MODEL] = acoustic_model
        else:
            self[Tags.ACOUSTIC_MODEL] = Tags.ACOUSTIC_MODEL_K_WAVE

        if acoustic_simulation_3D is not None:
            self[Tags.ACOUSTIC_SIMULATION_3D] = acoustic_simulation_3D
        else:
            self[Tags.ACOUSTIC_SIMULATION_3D] = False

        if speed_of_sound is not None:
            self[Tags.PROPERTY_SPEED_OF_SOUND] = speed_of_sound
        else:
            self[Tags.PROPERTY_SPEED_OF_SOUND] = 1540

        if density is not None:
            self[Tags.PROPERTY_DENSITY] = density
        else:
            self[Tags.PROPERTY_DENSITY] = 1000

    def add_reconstruction_properties(self, perform_image_reconstruction: bool = None,
                                      reconstruction_algorithm: str = None):

        if perform_image_reconstruction is not None:
            self[Tags.PERFORM_IMAGE_RECONSTRUCTION] = perform_image_reconstruction
        else:
            self[Tags.PERFORM_IMAGE_RECONSTRUCTION] = True

        if reconstruction_algorithm is not None:
            self[Tags.RECONSTRUCTION_ALGORITHM] = reconstruction_algorithm
        else:
            self[Tags.RECONSTRUCTION_ALGORITHM] = Tags.RECONSTRUCTION_ALGORITHM_DAS

    def save(self, path):
        save_hdf5(self, path)

    def load(self, path):
        for key, value in load_hdf5(path).items():
            self[key] = value
