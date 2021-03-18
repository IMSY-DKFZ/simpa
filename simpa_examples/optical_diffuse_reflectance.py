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
from simpa.utils.settings_generator import Settings
from simpa.io_handling.io_hdf5 import load_data_field

import numpy as np

# TODO change these paths to the desired executable and save folder
SAVE_PATH = "/home/leo/DKFZ/common_files/test_simpa_simulation"
# MCX_BINARY_PATH = "/home/leo/DKFZ/mcx-v2020/MCXStudio/MCXSuite/mcx/bin/mcx"     # On Linux systems, the .exe ate the end can be omitted.
MCX_BINARY_PATH = "/home/leo/DKFZ/cami-experimental/PAI/MCX/probe_integration/bin/mcx"     # On Linux systems, the .exe ate the end can be omitted.

VOLUME_TRANSDUCER_DIM_IN_MM = 60
VOLUME_PLANAR_DIM_IN_MM = 30
VOLUME_HEIGHT_IN_MM = 60
SPACING = 1
RANDOM_SEED = 471

# If VISUALIZE is set to True, the simulation result will be plotted
VISUALIZE = True


def create_example_tissue(global_settings):
    """
    This is a very simple example script of how to create a tissue definition.
    It contains a muscular background, an epidermis layer on top of the muscles
    and a blood vessel.
    """
    muscle_dictionary = Settings()
    muscle_dictionary[Tags.PRIORITY] = 1
    muscle_dictionary[Tags.STRUCTURE_START_MM] = [0, 0, 20]
    muscle_dictionary[Tags.STRUCTURE_END_MM] = [0, 0, 100]
    muscle_dictionary[Tags.MOLECULE_COMPOSITION] = TISSUE_LIBRARY.muscle()
    muscle_dictionary[Tags.CONSIDER_PARTIAL_VOLUME] = True
    muscle_dictionary[Tags.ADHERE_TO_DEFORMATION] = True
    muscle_dictionary[Tags.STRUCTURE_TYPE] = Tags.HORIZONTAL_LAYER_STRUCTURE

    vessel_1_dictionary = Settings()
    vessel_1_dictionary[Tags.PRIORITY] = 3
    vessel_1_dictionary[Tags.STRUCTURE_START_MM] = [VOLUME_TRANSDUCER_DIM_IN_MM/2, 10, 28]
    vessel_1_dictionary[Tags.STRUCTURE_END_MM] = [VOLUME_TRANSDUCER_DIM_IN_MM/2, 12, 28]
    vessel_1_dictionary[Tags.STRUCTURE_RADIUS_MM] = 3
    vessel_1_dictionary[Tags.MOLECULE_COMPOSITION] = TISSUE_LIBRARY.blood_generic()
    vessel_1_dictionary[Tags.CONSIDER_PARTIAL_VOLUME] = True
    vessel_1_dictionary[Tags.STRUCTURE_TYPE] = Tags.CIRCULAR_TUBULAR_STRUCTURE

    epidermis_dictionary = Settings()
    epidermis_dictionary[Tags.PRIORITY] = 8
    epidermis_dictionary[Tags.STRUCTURE_START_MM] = [0, 0, 20]
    epidermis_dictionary[Tags.STRUCTURE_END_MM] = [0, 0, 21]
    epidermis_dictionary[Tags.MOLECULE_COMPOSITION] = TISSUE_LIBRARY.epidermis()
    epidermis_dictionary[Tags.CONSIDER_PARTIAL_VOLUME] = True
    epidermis_dictionary[Tags.ADHERE_TO_DEFORMATION] = True
    # FIXME:
    epidermis_dictionary[Tags.STRUCTURE_TYPE] = Tags.HORIZONTAL_LAYER_STRUCTURE

    tissue_dict = Settings()
    tissue_dict["muscle"] = muscle_dictionary
    tissue_dict["epidermis"] = epidermis_dictionary
    tissue_dict["vessel_1"] = vessel_1_dictionary
    return tissue_dict

# Seed the numpy random configuration prior to creating the global_settings file in
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
    Tags.DIM_VOLUME_X_MM: VOLUME_TRANSDUCER_DIM_IN_MM,
    Tags.DIM_VOLUME_Y_MM: VOLUME_PLANAR_DIM_IN_MM,
    Tags.VOLUME_CREATOR: Tags.VOLUME_CREATOR_VERSATILE,

    # Simulation Device
    # Tags.DIGITAL_DEVICE: Tags.DIGITAL_DEVICE_MSOT,

    # The following parameters set the optical forward model
    Tags.RUN_OPTICAL_MODEL: True,
    Tags.WAVELENGTHS: [700],
    Tags.OPTICAL_MODEL_NUMBER_PHOTONS: 1e7,
    Tags.OPTICAL_MODEL_BINARY_PATH: MCX_BINARY_PATH,
    Tags.OPTICAL_MODEL: Tags.OPTICAL_MODEL_MCX,
    Tags.ILLUMINATION_TYPE: Tags.ILLUMINATION_TYPE_RING,
    Tags.ILLUMINATION_PARAM1: [7, 10, 45, 120],  # min_r, max_r, angle_1, angle_2
    Tags.ILLUMINATION_PARAM2: [25., 0, 0, 0],  # angle of photon dispersion used when launching photons
    Tags.LASER_PULSE_ENERGY_IN_MILLIJOULE: 50,
    Tags.SAVE_DIFFUSE_REFLECTANCE: True,

    # The following parameters tell the script that we do not want any extra
    # modelling steps
    Tags.RUN_ACOUSTIC_MODEL: False,
    Tags.APPLY_NOISE_MODEL: False,
    Tags.PERFORM_IMAGE_RECONSTRUCTION: False,
    Tags.SIMULATION_EXTRACT_FIELD_OF_VIEW: False,
}

settings = Settings(settings)
settings[Tags.SIMULATE_DEFORMED_LAYERS] = True
settings[Tags.STRUCTURES] = create_example_tissue(settings)

print("Simulating ", RANDOM_SEED)
import time
timer = time.time()
simulate(settings)
print("Needed", time.time()-timer, "seconds")
# TODO global_settings[Tags.SIMPA_OUTPUT_PATH]
print("Simulating ", RANDOM_SEED, "[Done]")

if VISUALIZE:
    from simpa.io_handling.io_hdf5 import load_hdf5
    from simpa.utils import SegmentationClasses
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    PATH = "/home/leo/DKFZ/common_files/test_simpa_simulation/MyVolumeName_471.hdf5"

    file = load_hdf5(PATH)
    settings = Settings(file["settings"])

    if Tags.WAVELENGTH in settings:
        WAVELENGTH = settings[Tags.WAVELENGTH]
    else:
        WAVELENGTH = 700

    fluence = load_data_field(PATH, Tags.OPTICAL_MODEL_FLUENCE, wavelength=700)

    dr = load_data_field(PATH, Tags.OPTICAL_MODEL_DIFFUSE_REFLECTANCE, wavelength=700)
    dr_pos = file['simulations']['optical_forward_model_output'][Tags.SURFACE_LAYER_POSITION]
    depth_map = np.zeros_like(dr)
    depth_map[dr_pos[0], dr_pos[1]] = dr_pos[2]

    initial_pressure = (file['simulations']['optical_forward_model_output']['initial_pressure'][str(WAVELENGTH)])
    absorption = (file['simulations']['simulation_properties']['mua'][str(WAVELENGTH)])
    if 'seg' in file['simulations']['simulation_properties'][str(WAVELENGTH)]:
        segmentation = (file['simulations']['simulation_properties']['seg'][str(WAVELENGTH)])
    else:
        segmentation = np.zeros_like(fluence)
    values = []
    names = []
    for string in SegmentationClasses.__dict__:
        if string[0:2] != "__":
            values.append(SegmentationClasses.__dict__[string])
            names.append(string)

    values = np.asarray(values)
    names = np.asarray(names)
    sort_indexes = np.argsort(values)
    values = values[sort_indexes]
    names = names[sort_indexes]

    colors = [list(np.random.random(3)) for _ in range(len(names))]
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        'Custom cmap', colors, len(names))

    shape = np.shape(initial_pressure)

    x_pos = int(shape[0] / 2)
    y_pos = int(shape[1] / 2)
    z_pos = int(shape[2] / 2)
    plt.figure()
    plt.subplot(241)
    plt.ylabel("X-Z Plane")
    plt.title("Fluence")
    plt.imshow(np.rot90(fluence[:, y_pos, :], -1))
    plt.subplot(242)
    plt.title("Absorption")
    plt.imshow(np.rot90(np.log10(absorption[:, y_pos, :]), -1))
    plt.subplot(243)
    plt.title("Initial Pressure")
    plt.imshow(np.rot90(np.log10(initial_pressure[:, y_pos, :]), -1))
    plt.subplot(244)
    plt.title("Segmentation")
    plt.imshow(np.rot90(segmentation[:, y_pos, :], -1), vmin=values[0], vmax=values[-1], cmap=cmap)
    cbar = plt.colorbar(ticks=values)
    cbar.ax.set_yticklabels(names)
    plt.subplot(245)
    plt.ylabel("Y-Z Plane")
    plt.imshow(np.rot90((fluence[x_pos, :, :]), -1))
    plt.subplot(246)
    plt.imshow(np.rot90(np.log10(absorption[x_pos, :, :]), -1))
    plt.subplot(247)
    plt.imshow(np.rot90(np.log10(initial_pressure[x_pos, :, :]), -1))
    plt.subplot(248)
    plt.imshow(np.rot90(segmentation[x_pos, :, :], -1), vmin=values[0], vmax=values[-1], cmap=cmap)
    cbar = plt.colorbar(ticks=values)
    cbar.ax.set_yticklabels(names)
    plt.show()

    fig = plt.figure()
    plt.subplot(121)
    plt.imshow(np.log(1-dr))
    plt.colorbar()
    plt.title("Diffuse Reflectance")
    plt.subplot(122)
    plt.imshow(depth_map)
    plt.colorbar()
    plt.title("Depth map")
    plt.show()
