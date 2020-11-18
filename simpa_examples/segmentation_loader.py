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

from simpa.core.simulation import simulate
from simpa.utils.settings_generator import Settings
from simpa.utils import Tags, SegmentationClasses
import nrrd
import numpy as np
from simpa.io_handling import load_hdf5
import matplotlib.pyplot as plt
from simpa.utils.libraries.tissue_library import TISSUE_LIBRARY
from simpa.utils.libraries.molecule_library import MOLECULE_LIBRARY
from simpa.utils.libraries.tissue_library import MolecularCompositionGenerator

label_mask, _ = nrrd.read("D:/labels.nrrd")
label_mask = label_mask[:, :, 0].reshape((256, 1, 128))

segmentation_volume_mask = np.tile(label_mask, (1, 64, 1))

print(np.shape(segmentation_volume_mask))

def segmention_class_mapping():
    ret_dict = dict()
    ret_dict[1] = TISSUE_LIBRARY.blood_generic()
    ret_dict[2] = TISSUE_LIBRARY.epidermis()
    ret_dict[3] = TISSUE_LIBRARY.muscle()
    ret_dict[4] = TISSUE_LIBRARY.mediprene()
    ret_dict[5] = TISSUE_LIBRARY.ultrasound_gel()
    ret_dict[6] = TISSUE_LIBRARY.heavy_water()
    ret_dict[7] = (MolecularCompositionGenerator()
                   .append(MOLECULE_LIBRARY.oxyhemoglobin(0.01))
                   .append(MOLECULE_LIBRARY.deoxyhemoglobin(0.01))
                   .append(MOLECULE_LIBRARY.water(0.98))
                   .get_molecular_composition(SegmentationClasses.COUPLING_ARTIFACT))
    return ret_dict

MCX_BINARY_PATH = "D:/bin/Release/mcx.exe"

settings = Settings()
settings[Tags.SIMULATION_PATH] = "D:/"
settings[Tags.VOLUME_NAME] = "SegmentationTest"
settings[Tags.RANDOM_SEED] = 1234
settings[Tags.WAVELENGTHS] = [700]
settings[Tags.VOLUME_CREATOR] = Tags.VOLUME_CREATOR_SEGMENTATION_BASED
settings[Tags.SPACING_MM] = 0.15625
settings[Tags.DIM_VOLUME_X_MM] = 40
settings[Tags.DIM_VOLUME_Y_MM] = 10
settings[Tags.DIM_VOLUME_Z_MM] = 20
settings[Tags.SIMULATION_EXTRACT_FIELD_OF_VIEW] = False
settings[Tags.INPUT_SEGMENTATION_VOLUME] = segmentation_volume_mask
settings[Tags.SEGMENTATION_CLASS_MAPPING] = segmention_class_mapping()
settings[Tags.RUN_OPTICAL_MODEL] = True
settings[Tags.DIGITAL_DEVICE] = Tags.DIGITAL_DEVICE_MSOT
settings[Tags.RUN_OPTICAL_MODEL] = True
settings[Tags.OPTICAL_MODEL_NUMBER_PHOTONS] = 1e7
settings[Tags.OPTICAL_MODEL_BINARY_PATH] = MCX_BINARY_PATH
settings[Tags.OPTICAL_MODEL] = Tags.OPTICAL_MODEL_MCX
settings[Tags.ILLUMINATION_TYPE] = Tags.ILLUMINATION_TYPE_MSOT_ACUITY_ECHO
settings[Tags.LASER_PULSE_ENERGY_IN_MILLIJOULE] = 50

simulate(settings)

PATH = "D:/SegmentationTest.hdf5"
WAVELENGTH = 700

file = load_hdf5(PATH)

print(file['simulations'].keys())

absorption = (file['simulations']['original_data']['simulation_properties']
              [str(WAVELENGTH)]['mua'])

segmentation = (file['simulations']['original_data']['simulation_properties']
              [str(WAVELENGTH)]['seg'])

shape = np.shape(absorption)
print(shape)

x_pos = int(shape[0]/2)
y_pos = int(shape[1]/2)

plt.figure()
plt.subplot(221)
plt.imshow(np.rot90(absorption[x_pos, :, :], -1))
plt.subplot(222)
plt.imshow(np.rot90(segmentation[x_pos, :, :], -1))
plt.subplot(223)
plt.imshow(np.rot90(absorption[:, y_pos, :], -1))
plt.subplot(224)
plt.imshow(np.rot90(segmentation[:, y_pos, :], -1))
plt.show()