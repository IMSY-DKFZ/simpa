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

from simpa.core.simulation import simulate
from simpa.utils.settings_generator import Settings
from simpa.utils import Tags, SegmentationClasses
import numpy as np
from skimage.data import shepp_logan_phantom
import matplotlib.pyplot as plt
from simpa.utils.libraries.tissue_library import TISSUE_LIBRARY
from simpa.utils.libraries.molecule_library import MOLECULE_LIBRARY
from simpa.utils.libraries.tissue_library import MolecularCompositionGenerator
from scipy.ndimage import zoom

SAVE_FOLDER = "/path/to/save/folder"
MCX_BINARY_PATH = "/path/to/mcx"
PATH = "/path/to/output_file.hdf5"

target_spacing = 1.0

label_mask = shepp_logan_phantom()

label_mask = np.digitize(label_mask, bins=np.linspace(0.0, 1.0, 11), right=True)

print(np.shape(label_mask))
label_mask = np.reshape(label_mask, (400, 1, 400))

input_spacing = 1.0
segmentation_volume_tiled = np.tile(label_mask, (1, 128, 1))
segmentation_volume_mask = np.round(zoom(segmentation_volume_tiled, input_spacing/target_spacing,
                                         order=0)).astype(int)
print(np.shape(segmentation_volume_mask))

plt.figure()
plt.subplot(121)
plt.imshow(np.rot90(segmentation_volume_mask[200, :, :], -1))
plt.subplot(122)
plt.imshow(np.rot90(segmentation_volume_mask[:, 64, :], -1))
plt.show()


def segmention_class_mapping():
    ret_dict = dict()
    ret_dict[0] = TISSUE_LIBRARY.heavy_water()
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
    ret_dict[8] = TISSUE_LIBRARY.heavy_water()
    ret_dict[9] = TISSUE_LIBRARY.heavy_water()
    ret_dict[10] = TISSUE_LIBRARY.heavy_water()
    return ret_dict

settings = {
    Tags.RUN_ACOUSTIC_MODEL: False,
    Tags.APPLY_NOISE_MODEL: False,
    Tags.PERFORM_IMAGE_RECONSTRUCTION: False,
}

settings = Settings(settings)
settings[Tags.SIMULATION_PATH] = SAVE_FOLDER
settings[Tags.VOLUME_NAME] = "SegmentationTest"
settings[Tags.RANDOM_SEED] = 1234
settings[Tags.WAVELENGTHS] = [700]
settings[Tags.VOLUME_CREATOR] = Tags.VOLUME_CREATOR_SEGMENTATION_BASED
settings[Tags.SPACING_MM] = target_spacing
settings[Tags.DIM_VOLUME_X_MM] = 400
settings[Tags.DIM_VOLUME_Y_MM] = 128
settings[Tags.DIM_VOLUME_Z_MM] = 400
settings[Tags.SIMULATION_EXTRACT_FIELD_OF_VIEW] = True
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
