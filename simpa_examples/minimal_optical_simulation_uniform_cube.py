# SPDX-FileCopyrightText: 2021 Division of Intelligent Medical Systems, DKFZ
# SPDX-FileCopyrightText: 2021 Janek Groehl
# SPDX-License-Identifier: MIT

# SPDX-FileCopyrightText: 2021 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT


from simpa import Tags
import simpa as sp
import numpy as np

# FIXME temporary workaround for newest Intel architectures
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# TODO: Please make sure that you have set the correct path to MCX binary and SAVE_PATH in the file path_config.env
#  located in the simpa_examples directory
path_manager = sp.PathManager()

VOLUME_TRANSDUCER_DIM_IN_MM = 60
VOLUME_PLANAR_DIM_IN_MM = 30
VOLUME_HEIGHT_IN_MM = 60
SPACING = 0.5
RANDOM_SEED = 471
VOLUME_NAME = "MyVolumeName_"+str(RANDOM_SEED)
SAVE_REFLECTANCE = True
SAVE_PHOTON_DIRECTION = False

# If VISUALIZE is set to True, the simulation result will be plotted
VISUALIZE = True


def create_example_tissue():
    """
    This is a very simple example script of how to create a tissue definition.
    It contains a muscular background, an epidermis layer on top of the muscles
    and a blood vessel.
    """
    background_dictionary = sp.Settings()
    background_dictionary[Tags.MOLECULE_COMPOSITION] = sp.TISSUE_LIBRARY.constant(1e-4, 1e-4, 0.9)
    background_dictionary[Tags.STRUCTURE_TYPE] = Tags.BACKGROUND

    tissue_dict = sp.Settings()
    tissue_dict[Tags.BACKGROUND] = background_dictionary
    return tissue_dict


# Seed the numpy random configuration prior to creating the global_settings file in
# order to ensure that the same volume
# is generated with the same random seed every time.

np.random.seed(RANDOM_SEED)

general_settings = {
    # These parameters set the general properties of the simulated volume
    Tags.RANDOM_SEED: RANDOM_SEED,
    Tags.VOLUME_NAME: VOLUME_NAME,
    Tags.SIMULATION_PATH: path_manager.get_hdf5_file_save_path(),
    Tags.SPACING_MM: SPACING,
    Tags.DIM_VOLUME_Z_MM: VOLUME_HEIGHT_IN_MM,
    Tags.DIM_VOLUME_X_MM: VOLUME_TRANSDUCER_DIM_IN_MM,
    Tags.DIM_VOLUME_Y_MM: VOLUME_PLANAR_DIM_IN_MM,
    Tags.WAVELENGTHS: [500],
    Tags.DO_FILE_COMPRESSION: True
}

settings = sp.Settings(general_settings)

settings.set_volume_creation_settings({
    Tags.STRUCTURES: create_example_tissue()
})
settings.set_optical_settings({
    Tags.OPTICAL_MODEL_NUMBER_PHOTONS: 5e7,
    Tags.OPTICAL_MODEL_BINARY_PATH: path_manager.get_mcx_binary_path(),
    Tags.COMPUTE_DIFFUSE_REFLECTANCE: SAVE_REFLECTANCE,
    Tags.COMPUTE_PHOTON_DIRECTION_AT_EXIT: SAVE_PHOTON_DIRECTION
})

pipeline = [
    sp.ModelBasedVolumeCreationAdapter(settings),
    sp.MCXAdapterReflectance(settings),
]

device = sp.PencilBeamIlluminationGeometry(device_position_mm=np.asarray([VOLUME_TRANSDUCER_DIM_IN_MM/2,
                                                                          VOLUME_PLANAR_DIM_IN_MM/2, 0]))

sp.simulate(pipeline, settings, device)

if VISUALIZE:
    sp.visualise_data(path_to_hdf5_file=path_manager.get_hdf5_file_save_path() + "/" + VOLUME_NAME + ".hdf5",
                      wavelength=settings[Tags.WAVELENGTH],
                      show_initial_pressure=True,
                      show_absorption=True,
                      show_diffuse_reflectance=SAVE_REFLECTANCE,
                      log_scale=True)
