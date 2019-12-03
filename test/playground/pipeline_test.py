from ippai.simulate import Tags
from ippai.simulate.simulation import simulate
from ippai.simulate.structures import *
from ippai.simulate.tissue_properties import get_constant_settings
import numpy as np


def create_water_background():
    water_dict = dict()
    water_dict[Tags.STRUCTURE_TYPE] = Tags.STRUCTURE_BACKGROUND
    water_dict[Tags.STRUCTURE_TISSUE_PROPERTIES] = get_constant_settings(mua=1, mus=100, g=0.9)
    water_dict[Tags.STRUCTURE_SEGMENTATION_TYPE] = SegmentationClasses.GENERIC
    return water_dict


def create_upsampling_phantom_parameters(x_min, x_max, z_min, z_max):
    structures_dict = dict()
    structures_dict["background"] = create_water_background()
    structures_dict["dermis"] = create_dermis_layer()
    structures_dict["epidermis"] = create_epidermis_layer()
    structures_dict["vessel"] = create_vessel_tube(x_min=x_min, x_max=x_max, z_min=z_min, z_max=z_max, r_max=5, r_min=5)
    return structures_dict


spacing = 0.3

settings = {
    # Optical forward path settings
    Tags.WAVELENGTHS: [800],
    Tags.RANDOM_SEED: 0,
    Tags.VOLUME_NAME: "Pipeline_test",
    Tags.SIMULATION_PATH: "/home/kris/hard_drive/data/pipeline_test",
    Tags.RUN_OPTICAL_MODEL: True,
    Tags.OPTICAL_MODEL_NUMBER_PHOTONS: 5e7,
    Tags.OPTICAL_MODEL_BINARY_PATH: "/home/kris/hard_drive/ippai/ippai/simulate/models/optical_models/mcx",
    Tags.OPTICAL_MODEL: Tags.MODEL_MCX,
    Tags.RUN_ACOUSTIC_MODEL: True,
    Tags.SIMULATION_EXTRACT_FIELD_OF_VIEW: True,
    Tags.SPACING_MM: spacing,
    Tags.DIM_VOLUME_Z_MM: 35,
    Tags.DIM_VOLUME_X_MM: 80,
    Tags.DIM_VOLUME_Y_MM: 40,
    Tags.STRUCTURES: create_upsampling_phantom_parameters(20, 20, 10.5, 10.5),

    Tags.ILLUMINATION_TYPE: Tags.ILLUMINATION_TYPE_MSOT_ACUITY_ECHO,
    # Tags.ILLUMINATION_POSITION: [50.5, 32.69, 1],
    Tags.ILLUMINATION_DIRECTION: [0, 0.381070, 0.9245460],        # direction of msot acuity
    # Tags.ILLUMINATION_DIRECTION: [0, 0.342027, 0.93969],        # direction of old pasetup
    Tags.ILLUMINATION_PARAM1: [30 / spacing, 0, 0, 0],
    # Tags.ILLUMINATION_PARAM1: [24.5 / spacing, 0, 0, 22.8 / spacing],

    # Upsampling settings

    Tags.PERFORM_UPSAMPLING: True,
    Tags.CROP_IMAGE: True,
    Tags.CROP_POWER_OF_TWO: True,
    Tags.UPSAMPLING_METHOD: Tags.UPSAMPLING_METHOD_NEAREST_NEIGHBOUR,
    Tags.UPSCALE_FACTOR: 2,
    Tags.UPSAMPLING_SCRIPT_LOCATION: "/home/kris/hard_drive/ippai/ippai/simulate/models/acoustic_models",
    Tags.UPSAMPLING_SCRIPT: "upsampling",

    # Acoustic forward path settings

    Tags.ACOUSTIC_MODEL_BINARY_PATH: "matlab",
    Tags.ACOUSTIC_MODEL_SCRIPT_LOCATION: "/home/kris/hard_drive/ippai/ippai/simulate/models/acoustic_models",
    Tags.ACOUSTIC_MODEL_SCRIPT: "simulate",
    Tags.GPU: True,

    Tags.MEDIUM_ALPHA_COEFF_HOMOGENEOUS: False,
    # Tags.MEDIUM_ALPHA_COEFF: 0.1,
    Tags.MEDIUM_ALPHA_POWER: 1.05,
    Tags.MEDIUM_SOUND_SPEED_HOMOGENEOUS: False,
    # Tags.MEDIUM_SOUND_SPEED: 1500,
    Tags.MEDIUM_DENSITY_HOMOGENEOUS: False,
    # Tags.MEDIUM_DENSITY: 7, #"/home/kris/hard_drive/data/k-wave/test_data/test_data/medium_density.npy",

    Tags.SENSOR_RECORD: "p",
    Tags.SENSOR_CENTER_FREQUENCY_MHZ: 7.5e6,
    Tags.SENSOR_BANDWIDTH_PERCENT: 80,
    Tags.SENSOR_DIRECTIVITY_HOMOGENEOUS: True,
    # Tags.SENSOR_DIRECTIVITY_ANGLE: 0,
    # Tags.SENSOR_DIRECTIVITY_ANGLE: "/home/kris/hard_drive/data/pipeline_test/sensor_directivity.npy",
    # 0,   # Most sensitive in x-dir (up/down)
    # Tags.SENSOR_DIRECTIVITY_SIZE_M: 0.001,  # [m]
    Tags.SENSOR_DIRECTIVITY_PATTERN: "pressure",

    Tags.SENSOR_ELEMENT_PITCH_CM: 0.034,
    # Tags.SENSOR_DIRECTIVITY_SIZE_M: 0.00000024,  # [m]
    Tags.SENSOR_SAMPLING_RATE_MHZ: 40,
    Tags.SENSOR_NUM_ELEMENTS: 256,
    Tags.SENSOR_RADIUS_MM: 40,

    Tags.PMLInside: False,
    Tags.PMLSize: [20, 20],
    Tags.PMLAlpha: 2,
    Tags.PlotPML: False,
    Tags.RECORDMOVIE: True,
    Tags.MOVIENAME: "visualization"
}
print("Simulating ", 0)
[settings_path, optical_path, acoustic_path, reconstruction_output_path] = simulate(settings)
print("Simulating ", 0, "[Done]")
