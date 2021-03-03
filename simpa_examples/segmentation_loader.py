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
import nrrd
import numpy as np
from simpa.io_handling import load_hdf5
import matplotlib.pyplot as plt
from simpa.utils.libraries.tissue_library import TISSUE_LIBRARY
from simpa.utils.libraries.molecule_library import MOLECULE_LIBRARY
from simpa.utils.libraries.tissue_library import MolecularCompositionGenerator
from scipy.ndimage import zoom

target_spacing = 0.2

label_mask, _ = nrrd.read("/home/kris/networkdrives/E130-Projekte/Photoacoustics/Studies/20201102_MS-PAB/labeled_data/Study_75_Scan_16_pa_01-labels.nrrd")
print(_)
label_mask = label_mask[:, :, 0].reshape((256, 1, 128))

input_spacing = 0.15625

segmentation_volume_tiled = np.tile(label_mask, (1, 128, 1))
segmentation_volume_mask = np.ones((461, 128, 128+244)) * 6
segmentation_volume_mask[230-128:230+128, :, 244:] = segmentation_volume_tiled

segmentation_volume_mask = np.round(zoom(segmentation_volume_mask, input_spacing/target_spacing,
                                         order=0)).astype(np.int)
segmentation_volume_mask[segmentation_volume_mask == 0] = 5

print(np.shape(segmentation_volume_mask))

plt.figure()
plt.subplot(121)
plt.imshow(np.rot90(segmentation_volume_mask[10, :, :], -1))
plt.subplot(122)
plt.imshow(np.rot90(segmentation_volume_mask[:, 10, :], -1))
plt.show()

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

MCX_BINARY_PATH = "/media/kris/Extreme SSD/simpa/simpa/core/optical_simulation/mcx"

settings = {
    Tags.RUN_ACOUSTIC_MODEL: True,
    Tags.ACOUSTIC_SIMULATION_3D: False,
    Tags.ACOUSTIC_MODEL: Tags.ACOUSTIC_MODEL_K_WAVE,
    Tags.ACOUSTIC_MODEL_BINARY_PATH: "/home/kris/hard_drive/MATLAB/bin/matlab",
    Tags.ACOUSTIC_MODEL_SCRIPT_LOCATION: "/media/kris/Extreme SSD/simpa/simpa/core/acoustic_simulation",
    Tags.GPU: True,

    Tags.PROPERTY_ALPHA_POWER: 1.05,

    Tags.SENSOR_RECORD: "p",
    # Tags.SENSOR_DIRECTIVITY_PATTERN: "pressure",

    Tags.PMLInside: False,
    Tags.PMLSize: [31, 32],
    Tags.PMLAlpha: 1.5,
    Tags.PlotPML: False,
    Tags.RECORDMOVIE: False,
    Tags.MOVIENAME: "visualization_log",
    Tags.ACOUSTIC_LOG_SCALE: True,

    Tags.APPLY_NOISE_MODEL: False,
    Tags.SIMULATION_EXTRACT_FIELD_OF_VIEW: True,

    Tags.PERFORM_IMAGE_RECONSTRUCTION: True,
    Tags.RECONSTRUCTION_ALGORITHM: Tags.RECONSTRUCTION_ALGORITHM_DAS,
    Tags.RECONSTRUCTION_BMODE_METHOD: Tags.RECONSTRUCTION_BMODE_METHOD_HILBERT_TRANSFORM,
    Tags.RECONSTRUCTION_MITK_BINARY_PATH: "/home/kris/hard_drive/MITK/"
                                          "sDMAS-2018.07-2596-g31d1c60d71-linux-x86_64/"
                                          "MITK-experiments/sDMAS-2018.07-2596-g31d1c60d71-linux-x86_64/"
                                          "MitkPABeamformingTool.sh",
    Tags.RECONSTRUCTION_MITK_SETTINGS_XML: "/home/kris/hard_drive/data/pipeline_test/bf_settings.xml",
    Tags.RECONSTRUCTION_OUTPUT_NAME: "/home/kris/hard_drive/data/pipeline_test/test.nrrd",

}

settings = Settings(settings)
settings[Tags.SIMULATION_PATH] = "/media/kris/Extreme SSD/data/simpa_examples"
settings[Tags.VOLUME_NAME] = "SegmentationTest"
settings[Tags.RANDOM_SEED] = 1234
settings[Tags.WAVELENGTHS] = [700]
settings[Tags.VOLUME_CREATOR] = Tags.VOLUME_CREATOR_SEGMENTATION_BASED
settings[Tags.SPACING_MM] = target_spacing
settings[Tags.DIM_VOLUME_X_MM] = 72
settings[Tags.DIM_VOLUME_Y_MM] = 20
settings[Tags.DIM_VOLUME_Z_MM] = 58.125
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

PATH = "/media/kris/Extreme SSD/data/simpa_examples/SegmentationTest.hdf5"
WAVELENGTH = 700

file = load_hdf5(PATH)

print(file['simulations'].keys())

absorption = (file['simulations']['original_data']['simulation_properties']
              [str(WAVELENGTH)]['mua'])

segmentation = (file['simulations']['original_data']['simulation_properties']
              [str(WAVELENGTH)]['seg'])

initial_pressure = (file['simulations']['original_data']
                    ['optical_forward_model_output']
                    [str(WAVELENGTH)]['initial_pressure'])

reconstruction = np.squeeze(
        file["simulations"]["original_data"]["reconstructed_data"][str(WAVELENGTH)]["reconstructed_data"])

real_image, header = nrrd.read("/home/kris/networkdrives/E130-Projekte/Photoacoustics/Studies/20201102_MS-PAB/labeled_data/Study_75_Scan_16_pa_01.nrrd")
print(header)
real_image = real_image[:, :, 0].reshape((256, 128))

shape = np.shape(absorption)
print(shape)

x_pos = int(shape[0]/2)
y_pos = int(shape[1]/2)

plt.figure()
# plt.subplot(131)
# plt.imshow(np.rot90(absorption, -1))
# plt.subplot(132)
# plt.imshow(np.rot90(initial_pressure, -1))
# plt.subplot(133)
# plt.imshow(np.rot90(segmentation, -1))
# plt.show()

plt.subplot(121)
plt.imshow(np.rot90(real_image, -1))
plt.subplot(122)
plt.imshow(np.fliplr(np.rot90(reconstruction, -1))[130:130+100, 80:-80])
plt.show()