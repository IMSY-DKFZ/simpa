"""
SPDX-FileCopyrightText: 2021 Computer Assisted Medical Interventions Group, DKFZ
SPDX-FileCopyrightText: 2021 VISION Lab, Cancer Research UK Cambridge Institute (CRUK CI)
SPDX-License-Identifier: MIT
"""

from simpa.utils import Tags, TISSUE_LIBRARY
from simpa.core.simulation import simulate
from simpa.utils.settings import Settings
from simpa.core.device_digital_twins import *
import numpy as np
from simpa.simulation_components import VolumeCreationModelModelBasedAdapter, OpticalForwardModelMcxAdapter, \
    GaussianNoiseProcessingComponent, FieldOfViewCroppingProcessingComponent, AcousticForwardModelKWaveAdapter, \
    ImageReconstructionModuleDelayAndSumAdapter
from simpa_examples.create_realistic_forearm_tissue import create_realistic_forearm_tissue
from simpa.io_handling import load_data_field
import matplotlib.pyplot as plt
from simpa.utils.path_manager import PathManager

# FIXME temporary workaround for newest Intel architectures
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# TODO: Please make sure that a valid path_config.env file is located in your home directory, or that you
#  point to the correct file in the PathManager().
path_manager = PathManager()

VOLUME_TRANSDUCER_DIM_IN_MM = 40
VOLUME_PLANAR_DIM_IN_MM = 20
VOLUME_HEIGHT_IN_MM = 20
SPACING = 0.15625
RANDOM_SEED = 1234
VOLUME_NAME = "Optical_Video_"+str(RANDOM_SEED)

# If VISUALIZE is set to True, the simulation result will be plotted
VISUALIZE = True
VIDEO = False
TOTAL_TIME = 5e-9
FRAMES = 1

# Seed the numpy random configuration prior to creating the global_settings file in
# order to ensure that the same volume
# is generated with the same random seed every time.

np.random.seed(RANDOM_SEED)

general_settings = {
    # These parameters set the general propeties of the simulated volume
    Tags.RANDOM_SEED: RANDOM_SEED,
    Tags.VOLUME_NAME: VOLUME_NAME,
    Tags.SIMULATION_PATH: path_manager.get_hdf5_file_save_path(),
    Tags.SPACING_MM: SPACING,
    Tags.DIM_VOLUME_Z_MM: VOLUME_HEIGHT_IN_MM,
    Tags.DIM_VOLUME_X_MM: VOLUME_TRANSDUCER_DIM_IN_MM,
    Tags.DIM_VOLUME_Y_MM: VOLUME_PLANAR_DIM_IN_MM,
    Tags.WAVELENGTHS: [700],
    Tags.GPU: True,
    Tags.LOAD_AND_SAVE_HDF5_FILE_AT_THE_END_OF_SIMULATION_TO_MINIMISE_FILESIZE: True
}

settings = Settings(general_settings)

number_of_knots = 6

deformation_z_elevations = np.tile([-4, -2, 0, -1.4, -1.8, -2.7][::-1], (5, 1))
deformation_z_elevations = np.moveaxis(deformation_z_elevations, 1, 0)
xx, yy = np.meshgrid(np.linspace(0,  (2 * np.sin(0.34 / 40 * 128) * 40), number_of_knots), np.linspace(0, 20, 5),
                     indexing='ij')

deformation_settings = dict()
deformation_settings[Tags.DEFORMATION_X_COORDINATES_MM] = xx
deformation_settings[Tags.DEFORMATION_Y_COORDINATES_MM] = yy
deformation_settings[Tags.DEFORMATION_Z_ELEVATIONS_MM] = deformation_z_elevations

settings.set_volume_creation_settings({
    Tags.SIMULATE_DEFORMED_LAYERS: True,
    Tags.STRUCTURES: create_realistic_forearm_tissue(settings),
    Tags.DEFORMED_LAYERS_SETTINGS: deformation_settings,
    Tags.US_GEL: True
})
settings.set_optical_settings({
    Tags.OPTICAL_MODEL_NUMBER_PHOTONS: 1e7,
    Tags.OPTICAL_MODEL_BINARY_PATH: path_manager.get_mcx_binary_path(),
    Tags.LASER_PULSE_ENERGY_IN_MILLIJOULE: 50,
    Tags.MCX_ASSUMED_ANISOTROPY: 0.9,
    Tags.TOTAL_TIME: TOTAL_TIME,
    Tags.TIME_STEP: TOTAL_TIME/FRAMES,
})
settings.set_acoustic_settings({
    Tags.ACOUSTIC_SIMULATION_3D: False,
    Tags.ACOUSTIC_MODEL_BINARY_PATH: path_manager.get_matlab_binary_path(),
    Tags.PROPERTY_ALPHA_POWER: 0,
    Tags.SENSOR_RECORD: "p",
    Tags.PMLInside: False,
    Tags.PMLSize: [31, 32],
    Tags.PMLAlpha: 1.5,
    Tags.PlotPML: False,
    Tags.RECORDMOVIE: False,
    Tags.MOVIENAME: "visualization_log",
    Tags.ACOUSTIC_LOG_SCALE: True,
    Tags.MODEL_SENSOR_FREQUENCY_RESPONSE: False,
    Tags.INITIAL_PRESSURE_SMOOTHING: True,
})
settings.set_reconstruction_settings({
    Tags.RECONSTRUCTION_PERFORM_BANDPASS_FILTERING: False,
    Tags.ACOUSTIC_MODEL_BINARY_PATH: path_manager.get_matlab_binary_path(),
    Tags.ACOUSTIC_SIMULATION_3D: False,
    Tags.PROPERTY_ALPHA_POWER: 0.00,
    Tags.TUKEY_WINDOW_ALPHA: 0.5,
    Tags.BANDPASS_CUTOFF_LOWPASS: int(8e6),
    Tags.BANDPASS_CUTOFF_HIGHPASS: int(0.1e4),
    Tags.RECONSTRUCTION_BMODE_AFTER_RECONSTRUCTION: True,
    Tags.RECONSTRUCTION_BMODE_METHOD: Tags.RECONSTRUCTION_BMODE_METHOD_HILBERT_TRANSFORM,
    Tags.RECONSTRUCTION_APODIZATION_METHOD: Tags.RECONSTRUCTION_APODIZATION_BOX,
    Tags.RECONSTRUCTION_MODE: Tags.RECONSTRUCTION_MODE_DIFFERENTIAL,
    Tags.SENSOR_RECORD: "p",
    Tags.PMLInside: False,
    Tags.PMLSize: [31, 32],
    Tags.PMLAlpha: 1.5,
    Tags.PlotPML: False,
    Tags.RECORDMOVIE: False,
    Tags.MOVIENAME: "visualization_log",
    Tags.ACOUSTIC_LOG_SCALE: True,
    Tags.PROPERTY_SPEED_OF_SOUND: 1540,
    Tags.PROPERTY_ALPHA_COEFF: 0.01,
    Tags.PROPERTY_DENSITY: 1000,
    Tags.SPACING_MM: SPACING,
    Tags.MODEL_SENSOR_FREQUENCY_RESPONSE: False,
    Tags.INITIAL_PRESSURE_SMOOTHING: False,
})

settings["noise_initial_pressure"] = {
    Tags.NOISE_MEAN: 1,
    Tags.NOISE_STD: 0.05,
    Tags.NOISE_MODE: Tags.NOISE_MODE_MULTIPLICATIVE,
    Tags.DATA_FIELD: Tags.OPTICAL_MODEL_INITIAL_PRESSURE,
    Tags.NOISE_NON_NEGATIVITY_CONSTRAINT: True
}

settings["noise_time_series"] = {
    Tags.NOISE_STD: 25,
    Tags.NOISE_MODE: Tags.NOISE_MODE_ADDITIVE,
    Tags.DATA_FIELD: Tags.TIME_SERIES_DATA
}


pipeline = [
    VolumeCreationModelModelBasedAdapter(settings),
    OpticalForwardModelMcxAdapter(settings),
    GaussianNoiseProcessingComponent(settings, "noise_initial_pressure"),
    AcousticForwardModelKWaveAdapter(settings),
    GaussianNoiseProcessingComponent(settings, "noise_time_series"),
    ImageReconstructionModuleDelayAndSumAdapter(settings),
    FieldOfViewCroppingProcessingComponent(settings),
]

device = MSOTAcuityEcho(device_position_mm=np.array([VOLUME_TRANSDUCER_DIM_IN_MM/2,
                                                     VOLUME_PLANAR_DIM_IN_MM/2,
                                                     -3]),
                        field_of_view_extent_mm=np.asarray([-20 - SPACING/2, 20 + SPACING/2, 0, 0, 0, 20 + SPACING]))

device.update_settings_for_use_of_model_based_volume_creator(settings)

simulate(pipeline, settings, device)

if Tags.WAVELENGTH in settings:
    WAVELENGTH = settings[Tags.WAVELENGTH]
else:
    WAVELENGTH = 700

if VISUALIZE:
    from cv2 import VideoWriter, VideoWriter_fourcc
    from matplotlib import cm
    from PIL import Image
    from simpa.utils.calculate import min_max_normalization
    import nrrd

    p0 = load_data_field(path_manager.get_hdf5_file_save_path() + "/" + VOLUME_NAME + ".hdf5",
                         Tags.OPTICAL_MODEL_INITIAL_PRESSURE, WAVELENGTH)
    recon = load_data_field(path_manager.get_hdf5_file_save_path() + "/" + VOLUME_NAME + ".hdf5",
                            Tags.RECONSTRUCTED_DATA, WAVELENGTH)

    recon = min_max_normalization(recon)
    recon = np.rot90(recon, 3)

    orig_im, header = nrrd.read(REAL_IMAGE_PATH)
    orig_im = min_max_normalization(orig_im[:, :, 0])
    orig_im = np.fliplr(np.rot90(orig_im, 3))

    # plt.subplot(1, 2, 1)
    # plt.imshow(recon)
    # plt.subplot(1, 2, 2)
    # plt.imshow(orig_im)
    # plt.show()

    MORPH_STEPS = 1000

    shape = recon.shape

    width = shape[1]
    height = shape[0]
    seconds = 5
    FPS = int(MORPH_STEPS / seconds)

    fourcc = VideoWriter_fourcc(*'MP42')
    video = VideoWriter(path_manager.get_hdf5_file_save_path() + "/morph_video.avi", fourcc, float(FPS), (width, height))

    for i in range(FPS * seconds):
        frame = ((FPS * seconds) - (i + 1))/(FPS * seconds) * recon + (i + 1)/(FPS * seconds) * orig_im
        frame /= np.max(frame)
        frame = Image.fromarray(np.uint8(cm.viridis(frame) * 255)).convert("RGB")
        frame = np.asarray(frame)
        frame = frame[:, :, ::-1]
        video.write(frame)
    video.release()

    if VIDEO:
        shape = p0.shape

        p0 = p0[:, int(shape[1]/2), :, :]
        p0 = np.rot90(p0, 3, axes=(0, 1))
        shape = p0.shape
        print(shape)

        width = shape[1]
        height = shape[0]
        seconds = 5
        FPS = int(FRAMES / seconds)

        fourcc = VideoWriter_fourcc(*'MP42')
        video = VideoWriter(path_manager.get_hdf5_file_save_path() + "/video.avi", fourcc, float(FPS), (width, height))

        for i in range(FPS * seconds):
            frame = p0[:, :, i]
            frame /= np.max(frame)
            frame = Image.fromarray(np.uint8(cm.viridis(frame)*255)).convert("RGB")
            frame = np.asarray(frame)
            frame = frame[:, :, ::-1]
            video.write(frame)
        video.release()
