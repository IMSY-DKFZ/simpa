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

from simpa.utils import Tags, TISSUE_LIBRARY

from simpa.core.simulation import simulate
from simpa.utils.libraries.structure_library import Background
from simpa.utils.libraries.structure_library import HorizontalLayerStructure
from simpa.utils.libraries.structure_library import TubularStructure
from simpa.utils.settings_generator import Settings
from simpa.utils import create_deformation_settings

import numpy as np

# TODO change these paths to the desired executable and save folder
SAVE_PATH = "D:/bin/"
MCX_BINARY_PATH = "D:/bin/Release/mcx.exe"

VOLUME_WIDTH_IN_MM = 50
VOLUME_HEIGHT_IN_MM = 20
SPACING = 0.432
RANDOM_SEED = 4711


def create_example_tissue(global_settings):
    """
    This is a very simple example script of how to create a tissue definition.
    It contains a muscular background, an epidermis layer on top of the muscles
    and a blood vessel.
    """
    background_dictionary = Settings()
    background_dictionary[Tags.MOLECULE_COMPOSITION] = TISSUE_LIBRARY.constant(0.1, 100.0, 0.9)
    bg = Background(global_settings, background_dictionary)

    muscle_dictionary = Settings()
    muscle_dictionary[Tags.PRIORITY] = 1
    muscle_dictionary[Tags.STRUCTURE_START_MM] = [0, 0, 0]
    muscle_dictionary[Tags.STRUCTURE_END_MM] = [0, 0, 100]
    muscle_dictionary[Tags.MOLECULE_COMPOSITION] = TISSUE_LIBRARY.muscle()
    muscle = HorizontalLayerStructure(global_settings, muscle_dictionary)

    vessel_1_dictionary = Settings()
    vessel_1_dictionary[Tags.PRIORITY] = 3
    vessel_1_dictionary[Tags.STRUCTURE_START_MM] = [25, 0, 10]
    vessel_1_dictionary[Tags.STRUCTURE_END_MM] = [25, 50, 12]
    vessel_1_dictionary[Tags.STRUCTURE_RADIUS] = 3
    vessel_1_dictionary[Tags.MOLECULE_COMPOSITION] = TISSUE_LIBRARY.blood_generic()
    vessel_1 = TubularStructure(global_settings, vessel_1_dictionary)

    epidermis_dictionary = Settings()
    epidermis_dictionary[Tags.PRIORITY] = 8
    epidermis_dictionary[Tags.STRUCTURE_START_MM] = [0, 0, 0]
    epidermis_dictionary[Tags.STRUCTURE_END_MM] = [0, 0, 1]
    epidermis_dictionary[Tags.MOLECULE_COMPOSITION] = TISSUE_LIBRARY.epidermis()
    epidermis = HorizontalLayerStructure(global_settings, epidermis_dictionary)

    tissue_dict = dict()
    tissue_dict[Tags.BACKGROUND] = bg.to_settings()
    tissue_dict["muscle"] = muscle.to_settings()
    tissue_dict["epidermis"] = epidermis.to_settings()
    tissue_dict["vessel_1"] = vessel_1.to_settings()
    return tissue_dict

# Seed the numpy random configuration prior to creating the settings file in
# order to ensure that the same volume
# is generated with the same random seed every time.

np.random.seed(RANDOM_SEED)

settings = {
    # These parameters set the general propeties of the simulated volume
    Tags.RANDOM_SEED: RANDOM_SEED,
    Tags.VOLUME_NAME: "MyVolumeName_"+str(RANDOM_SEED),
    Tags.SIMULATION_PATH: SAVE_PATH,
    Tags.SPACING_MM: SPACING,
    Tags.DIM_VOLUME_Z_MM: VOLUME_HEIGHT_IN_MM,
    Tags.DIM_VOLUME_X_MM: VOLUME_WIDTH_IN_MM,
    Tags.DIM_VOLUME_Y_MM: VOLUME_WIDTH_IN_MM,
    Tags.VOLUME_CREATOR: Tags.VOLUME_CREATOR_VERSATILE,

    # Simulation Device
    Tags.DIGITAL_DEVICE: Tags.DIGITAL_DEVICE_MSOT,

    # The following parameters set the optical forward model
    Tags.RUN_OPTICAL_MODEL: True,
    Tags.WAVELENGTHS: [700],
    Tags.OPTICAL_MODEL_NUMBER_PHOTONS: 1e7,
    Tags.OPTICAL_MODEL_BINARY_PATH: MCX_BINARY_PATH,
    Tags.OPTICAL_MODEL: Tags.OPTICAL_MODEL_MCX,
    Tags.ILLUMINATION_TYPE: Tags.ILLUMINATION_TYPE_MSOT_ACUITY_ECHO,
    Tags.LASER_PULSE_ENERGY_IN_MILLIJOULE: 50,

    # The following parameters tell the script that we do not want any extra
    # modelling steps
    Tags.RUN_ACOUSTIC_MODEL: False,
    Tags.APPLY_NOISE_MODEL: False,
    Tags.PERFORM_IMAGE_RECONSTRUCTION: False,
    Tags.SIMULATION_EXTRACT_FIELD_OF_VIEW: False,

    # Add the volume_creation to be simulated to the tissue

}
settings = Settings(settings)
settings[Tags.SIMULATE_DEFORMED_LAYERS] = True
np.random.seed(RANDOM_SEED)
settings[Tags.DEFORMED_LAYERS_SETTINGS] = create_deformation_settings(bounds_mm=[[0, settings[Tags.DIM_VOLUME_X_MM]],
                                                                                 [0, settings[Tags.DIM_VOLUME_Y_MM]]],
                                                                      maximum_z_elevation_mm=3)
settings[Tags.STRUCTURES] = create_example_tissue(settings)
print("Simulating ", RANDOM_SEED)
import time
timer = time.time()
simulate(settings)
print("Needed", time.time()-timer, "seconds")
# TODO settings[Tags.SIMPA_OUTPUT_PATH]
print("Simulating ", RANDOM_SEED, "[Done]")
