from ippai.simulate import Tags
from ippai.simulate.simulation import simulate
from ippai.simulate.structures import *
from ippai.simulate.tissue_properties import get_constant_settings
import numpy as np


def create_water_background():
    water_dict = dict()
    water_dict[Tags.STRUCTURE_TYPE] = Tags.STRUCTURE_BACKGROUND
    water_dict[Tags.STRUCTURE_TISSUE_PROPERTIES] = get_constant_settings(mua=0.1, mus=100, g=0.9)
    water_dict[Tags.STRUCTURE_SEGMENTATION_TYPE] = SegmentationClasses.ULTRASOUND_GEL_PAD
    return water_dict


def create_upsampling_phantom_parameters(x_min, x_max, z_min, z_max):
    structures_dict = dict()
    structures_dict["background"] = create_water_background()
    num_vessels = np.random.randint(10, 20)
    for vessel_idx in range(num_vessels):
        structures_dict["vessel_"+str(vessel_idx)] = create_vessel_tube(x_min=x_min, x_max=x_max,
                                                               z_min=z_min, z_max=z_max,
                                                               r_max=0.5, r_min=3)
    return structures_dict


for volume_idx in range(0, 1):
    settings = {

        # Basic & geometry settings
        Tags.RANDOM_SEED: volume_idx,
        Tags.VOLUME_NAME: "TestData_" + str(volume_idx).zfill(6),
        Tags.SIMULATION_PATH: "/media/janek/PA DATA/tmp/",
        Tags.SPACING_MM: 0.15,
        Tags.DIM_VOLUME_Z_MM: 21,
        Tags.DIM_VOLUME_X_MM: 40,
        Tags.DIM_VOLUME_Y_MM: 25,
        Tags.AIR_LAYER_HEIGHT_MM: 12,
        Tags.GELPAD_LAYER_HEIGHT_MM: 18,
        Tags.STRUCTURES: create_upsampling_phantom_parameters(0, 40, 2, 21),
        Tags.SIMULATION_EXTRACT_FIELD_OF_VIEW: True,
        Tags.LASER_PULSE_ENERGY_IN_MILLIJOULE: 20.5,

        # Optical forward path settings
        Tags.WAVELENGTHS: [800],
        Tags.RUN_OPTICAL_MODEL: True,
        Tags.OPTICAL_MODEL_NUMBER_PHOTONS: 1e7,
        Tags.OPTICAL_MODEL_BINARY_PATH: "/home/janek/bin/mcx",
        Tags.OPTICAL_MODEL: Tags.MODEL_MCX,

        # Upsampling settings
        Tags.PERFORM_UPSAMPLING: True,
        Tags.CROP_IMAGE: True,
        Tags.CROP_POWER_OF_TWO: True,
        Tags.UPSAMPLING_METHOD: Tags.UPSAMPLING_METHOD_NEAREST_NEIGHBOUR,
        Tags.UPSCALE_FACTOR: 4.0,
        Tags.DL_MODEL_PATH: None,

        # Acoustic forward path settings
        Tags.ACOUSTIC_MODEL_BINARY_PATH: "/home/janek/MATLAB/R2019b/bin/matlab",
        Tags.RUN_ACOUSTIC_MODEL: True,
        Tags.ACOUSTIC_MODEL_SCRIPT_LOCATION: "/home/janek/ippai/ippai/simulate/models/acoustic_models/",
        Tags.ACOUSTIC_MODEL_SCRIPT: "simulate",
        Tags.GPU: True,

        Tags.MEDIUM_ALPHA_COEFF: 0.1,
        Tags.MEDIUM_ALPHA_POWER: 1.5,
        Tags.MEDIUM_SOUND_SPEED: 1540,  #"/home/kris/hard_drive/data/k-wave/test_data/test_data/sound_speed.npy",
        Tags.MEDIUM_DENSITY: 1,  #"/home/kris/hard_drive/data/k-wave/test_data/test_data/medium_density.npy",

        Tags.SENSOR_MASK: 1,  #"/home/kris/hard_drive/data/k-wave/test_data/test_data/sensor_mask.npy",
        Tags.SENSOR_RECORD: "p",
        Tags.SENSOR_CENTER_FREQUENCY_MHZ: 7.5e6,
        Tags.SENSOR_BANDWIDTH_PERCENT: 80,
        Tags.SENSOR_DIRECTIVITY_ANGLE: 0,  #"/home/kris/hard_drive/data/k-wave/test_data/test_data/directivity_angle.npy",
        # 0,   # Most sensitive in x-dir (up/down)
        Tags.SENSOR_DIRECTIVITY_SIZE: 0.001,  # [m]
        Tags.SENSOR_DIRECTIVITY_PATTERN: "pressure",
        Tags.SENSOR_SAMPLING_RATE_MHZ: 66 + 2/3,

        Tags.PMLInside: False,
        Tags.PMLSize: [20, 20],
        Tags.PMLAlpha: 2,
        Tags.PlotPML: False,
        Tags.RECORDMOVIE: True,
        Tags.MOVIENAME: "Movie"
    }
    print("Simulating ", volume_idx)
    [settings_path, optical_path, acoustic_path] = simulate(settings)
    print("Simulating ", volume_idx, "[Done]")
