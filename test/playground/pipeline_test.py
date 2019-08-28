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


def create_upsampling_phantom_parameters():
    structures_dict = dict()
    structures_dict["background"] = create_water_background()
    structures_dict["dermis"] = create_dermis_layer()
    structures_dict["epidermis"] = create_epidermis_layer()
    structures_dict["vessel"] = create_vessel_tube(x_min=20, x_max=20, z_min=10.5, z_max=10.5, r_max=1, r_min=1)
    return structures_dict


seed_index = 0
while seed_index < 1:
    random_seed = 200000 + seed_index
    seed_index += 1
    np.random.seed(random_seed)

    settings = {
        # Optical forward path settings
        Tags.WAVELENGTHS: [800],
        Tags.RANDOM_SEED: random_seed,
        Tags.VOLUME_NAME: "UpsamplingPhantom_"+str(random_seed).zfill(6),
        Tags.SIMULATION_PATH: "/home/kris/hard_drive/data/pipeline_test",
        Tags.RUN_OPTICAL_MODEL: True,
        Tags.OPTICAL_MODEL_NUMBER_PHOTONS: 1e6,
        Tags.OPTICAL_MODEL_BINARY_PATH: "/home/kris/hard_drive/mcx_test/mcx",
        Tags.OPTICAL_MODEL: Tags.MODEL_MCX,
        Tags.RUN_ACOUSTIC_MODEL: True,
        Tags.SIMULATION_EXTRACT_FIELD_OF_VIEW: True,
        Tags.SPACING_MM: 0.15,
        Tags.DIM_VOLUME_Z_MM: 21,
        Tags.DIM_VOLUME_X_MM: 40,
        Tags.DIM_VOLUME_Y_MM: 25,
        Tags.AIR_LAYER_HEIGHT_MM: 12,
        Tags.GELPAD_LAYER_HEIGHT_MM: 18,
        Tags.STRUCTURES: create_upsampling_phantom_parameters(),

        # Upsampling settings

        Tags.UPSAMPLE: True,
        Tags.CROP_IMAGE: True,
        Tags.CROP_POWER_OF_TWO: True,
        Tags.UPSAMPLING_METHOD: "nearest_neighbor",
        Tags.UPSCALE_FACTOR: 2,

        # Acoustic forward path settings

        Tags.ACOUSTIC_MODEL_SCRIPT: "simulate",
        Tags.GPU: True,

        Tags.MEDIUM_ALPHA_COEFF: 0.1,
        Tags.MEDIUM_ALPHA_POWER: 1.5,
        Tags.MEDIUM_SOUND_SPEED: 1500, #"/home/kris/hard_drive/data/k-wave/test_data/test_data/sound_speed.npy",
        Tags.MEDIUM_DENSITY: 1, #"/home/kris/hard_drive/data/k-wave/test_data/test_data/medium_density.npy",

        Tags.SENSOR_MASK: 1, #"/home/kris/hard_drive/data/k-wave/test_data/test_data/sensor_mask.npy",
        Tags.SENSOR_RECORD: "p",
        Tags.SENSOR_CENTER_FREQUENCY: 7.5e6,
        Tags.SENSOR_BANDWIDTH: 133,
        Tags.SENSOR_DIRECTIVITY_ANGLE: 0, #"/home/kris/hard_drive/data/k-wave/test_data/test_data/directivity_angle.npy",
        # 0,   # Most sensitive in x-dir (up/down)
        Tags.SENSOR_DIRECTIVITY_SIZE: 0.001,  # [m]
        Tags.SENSOR_DIRECTIVITY_PATTERN: "pressure",

        Tags.PMLInside: False,
        Tags.PMLSize: [20, 20],
        Tags.PMLAlpha: 2,
        Tags.PlotPML: False,
        Tags.RECORDMOVIE: True,
        Tags.MOVIENAME: "test"
    }
    print("Simulating ", random_seed)
    [settings_path, optical_path, acoustic_path] = simulate(settings)
    print("Simulating ", random_seed, "[Done]")
