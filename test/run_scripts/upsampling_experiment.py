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
    structures_dict["dermis"] = create_dermis_layer()
    structures_dict["epidermis"] = create_epidermis_layer()
    structures_dict["vessel"] = create_vessel_tube(x_min=x_min, x_max=x_max, z_min=z_min, z_max=z_max, r_max=1, r_min=1)
    return structures_dict


upsampling_methods = [Tags.UPSAMPLING_METHOD_LANCZOS2,
                      Tags.UPSAMPLING_METHOD_LANCZOS3]
spacings = [0.8, 0.4, 0.2]

root = "/home/kris/hard_drive/cami-experimental/PAI/experiments/Upsampling/"
test_models = ["20191002-123826_Upscalel1_0.8_to_0.4/save/epoch_99.pt",
               "20191002-124021_Upscalel1_0.4_to_0.2/save/epoch_99.pt",
               "20191002-124323_Upscalel1_0.2_to_0.1/save/epoch_99.pt"
               ]


for i in range(0, 9):
    x = 10 * (i % 3 + 1)
    z = 5.25 * (i // 3 + 1)
    for u, method in enumerate(upsampling_methods):
        print(method)
        for [spacing, model_path] in zip(spacings, test_models):
            settings = {
                # Optical forward path settings
                Tags.WAVELENGTHS: [800],
                Tags.RANDOM_SEED: i,
                Tags.VOLUME_NAME: "UpsamplingPhantom_"+str(i) + "/" + method + "/" + str(spacing),
                Tags.SIMULATION_PATH: "/home/kris/hard_drive/data/upsampling_test_spacing_2",
                Tags.RUN_OPTICAL_MODEL: True,
                Tags.OPTICAL_MODEL_NUMBER_PHOTONS: 5e7,
                Tags.OPTICAL_MODEL_BINARY_PATH: "/home/kris/hard_drive/mcx_test/mcx",
                Tags.OPTICAL_MODEL: Tags.MODEL_MCX,
                Tags.RUN_ACOUSTIC_MODEL: True,
                Tags.SIMULATION_EXTRACT_FIELD_OF_VIEW: True,
                Tags.SPACING_MM: spacing,
                Tags.DIM_VOLUME_Z_MM: 21,
                Tags.DIM_VOLUME_X_MM: 40,
                Tags.DIM_VOLUME_Y_MM: 25,
                Tags.AIR_LAYER_HEIGHT_MM: 12,
                Tags.GELPAD_LAYER_HEIGHT_MM: 18,
                Tags.STRUCTURES: create_upsampling_phantom_parameters(x, x, z, z),

                Tags.ILLUMINATION_TYPE: Tags.ILLUMINATION_TYPE_DKFZ_PAUS,
                Tags.ILLUMINATION_DIRECTION: [0, 0.342027, 0.93969],
                Tags.ILLUMINATION_PARAM1: [24.5 / spacing, 0, 0, 22.8 / spacing],

                # Upsampling settings

                Tags.PERFORM_UPSAMPLING: True,
                Tags.CROP_IMAGE: True,
                Tags.CROP_POWER_OF_TWO: True,
                Tags.UPSAMPLING_METHOD: method,
                Tags.UPSCALE_FACTOR: 2,
                Tags.DL_MODEL_PATH: root + model_path,
                Tags.UPSAMPLING_SCRIPT_LOCATION: "/home/kris/hard_drive/ippai/ippai/simulate/models/acoustic_models",
                Tags.UPSAMPLING_SCRIPT: "upsampling",

                # Acoustic forward path settings

                Tags.ACOUSTIC_MODEL_BINARY_PATH: "matlab",
                Tags.ACOUSTIC_MODEL_SCRIPT_LOCATION: "/home/kris/hard_drive/ippai/ippai/simulate/models/acoustic_models",
                Tags.ACOUSTIC_MODEL_SCRIPT: "simulate",
                Tags.GPU: True,

                Tags.MEDIUM_ALPHA_COEFF: 0.1,
                Tags.MEDIUM_ALPHA_POWER: 1.5,
                Tags.MEDIUM_SOUND_SPEED: 1500, #"/home/kris/hard_drive/data/k-wave/test_data/test_data/sound_speed.npy",
                Tags.MEDIUM_DENSITY: 1, #"/home/kris/hard_drive/data/k-wave/test_data/test_data/medium_density.npy",

                Tags.SENSOR_MASK: 1, #"/home/kris/hard_drive/data/k-wave/test_data/test_data/sensor_mask.npy",
                Tags.SENSOR_RECORD: "p",
                Tags.SENSOR_CENTER_FREQUENCY_MHZ: 7.5e6,
                Tags.SENSOR_BANDWIDTH_PERCENT: 80,
                Tags.SENSOR_DIRECTIVITY_ANGLE: 0, #"/home/kris/hard_drive/data/k-wave/test_data/test_data/directivity_angle.npy",
                # 0,   # Most sensitive in x-dir (up/down)
                Tags.SENSOR_DIRECTIVITY_SIZE_M: 0.001,  # [m]
                Tags.SENSOR_DIRECTIVITY_PATTERN: "pressure",

                Tags.PMLInside: False,
                Tags.PMLSize: [20, 20],
                Tags.PMLAlpha: 2,
                Tags.PlotPML: False,
                Tags.RECORDMOVIE: False,
                Tags.MOVIENAME: "UpsamplingPhantom_"+str(i) + "/" + method + "/" + str(spacing) + "/" + "visualization"
            }
            print("Simulating ", i)
            [settings_path, optical_path, acoustic_path, reconstruction_output_path] = simulate(settings)
            print("Simulating ", i, "[Done]")
