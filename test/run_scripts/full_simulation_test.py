from ippai.simulate import Tags, SegmentationClasses
from ippai.simulate.simulation import simulate
from ippai.simulate.models.reconstruction import perform_reconstruction
import numpy as np
from ippai.simulate.structures import create_vessel_tube, get_constant_settings, create_muscle_background
import numpy as np

VESSELS_MIN = 1
VESSELS_MAX = 1


def create_milk_background():
    water_dict = dict()
    water_dict[Tags.STRUCTURE_TYPE] = Tags.STRUCTURE_BACKGROUND
    water_dict[Tags.STRUCTURE_TISSUE_PROPERTIES] = get_constant_settings(mua=0.1, mus=100, g=0.9)
    water_dict[Tags.STRUCTURE_SEGMENTATION_TYPE] = SegmentationClasses.GENERIC
    return water_dict


def create_single_vessel_phantom_parameters():
    structures_dict = dict()
    x_min = 18
    x_max = 22
    z_min = 4
    z_max = 6
    structures_dict["background"] = create_milk_background()
    structures_dict["blood_tube"] = create_vessel_tube(x_min=x_min, x_max=x_max,
                                                       z_min=z_min, z_max=z_max,
                                                       r_max=2, r_min=3)
    return structures_dict


def create_multi_vessel_phantom_parameters():
    structures_dict = dict()
    x_min = 4
    x_max = 36
    z_min = 4
    z_max = 10
    structures_dict["background"] = create_milk_background()
    num_vessels = np.random.randint(1, 3)
    for i in range(num_vessels):
        structures_dict["blood_tube_" + str(i)] = create_vessel_tube(x_min=x_min, x_max=x_max,
                                                                     z_min=z_min, z_max=z_max,
                                                                     r_max=1, r_min=3)
    return structures_dict


def create_complex_parameters():
    structures_dict = dict()
    x_min = 4
    x_max = 36
    z_min = 4
    z_max = 10
    structures_dict["background"] = create_muscle_background(background_oxy=0.5)
    num_vessels = np.random.randint(VESSELS_MIN, VESSELS_MAX)
    for i in range(num_vessels):
        structures_dict["blood_tube_" + str(i)] = create_vessel_tube(x_min=x_min, x_max=x_max,
                                                                     z_min=z_min, z_max=z_max,
                                                                     r_max=2, r_min=3)
    return structures_dict


for seed_index in range(200, 400):

    random_seed = 10000 + seed_index
    seed_index += 1
    np.random.seed(random_seed)

    settings = {

        # Basic & geometry settings
        Tags.RANDOM_SEED: random_seed,
        Tags.VOLUME_NAME: "Phantom_" + str(random_seed).zfill(6),
        Tags.SIMULATION_PATH: "/media/janek/PA DATA/DS_Simple/",
        Tags.SPACING_MM: 0.3,
        Tags.DIM_VOLUME_Z_MM: 21,
        Tags.DIM_VOLUME_X_MM: 40,
        Tags.DIM_VOLUME_Y_MM: 25,
        Tags.AIR_LAYER_HEIGHT_MM: 12,
        Tags.GELPAD_LAYER_HEIGHT_MM: 18,
        Tags.STRUCTURES: create_multi_vessel_phantom_parameters(),
        Tags.SIMULATION_EXTRACT_FIELD_OF_VIEW: True,
        Tags.LASER_PULSE_ENERGY_IN_MILLIJOULE: 50,

        # Optical forward path settings
        Tags.WAVELENGTHS: [700, 750, 800, 850, 900],
        Tags.WAVELENGTH: 700,
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
        Tags.MEDIUM_SOUND_SPEED: 1540,
        Tags.MEDIUM_DENSITY: 1,

        Tags.SENSOR_MASK: 1,
        Tags.SENSOR_RECORD: "p",
        Tags.SENSOR_CENTER_FREQUENCY_MHZ: 7.5e6,
        Tags.SENSOR_BANDWIDTH_PERCENT: 80,
        Tags.SENSOR_DIRECTIVITY_ANGLE: 0,
        Tags.SENSOR_ELEMENT_PITCH_CM: 0.3,
        Tags.SENSOR_DIRECTIVITY_SIZE_M: 0.002,  # [m]
        Tags.SENSOR_DIRECTIVITY_PATTERN: "pressure",
        Tags.SENSOR_SAMPLING_RATE_MHZ: 66 + 2/3,
        Tags.SENSOR_NUM_ELEMENTS: 128,

        Tags.PMLInside: False,
        Tags.PMLSize: [20, 20],
        Tags.PMLAlpha: 2,
        Tags.PlotPML: False,
        Tags.RECORDMOVIE: False,
        Tags.MOVIENAME: "Movie",

        # Reconstruction settings
        Tags.PERFORM_IMAGE_RECONSTRUCTION: True,
        Tags.RECONSTRUCTION_ALGORITHM: Tags.RECONSTRUCTION_ALGORITHM_SDMAS,
        Tags.RECONSTRUCTION_MITK_BINARY_PATH: "/home/janek/bin/mitk/MitkPABeamformingTool.sh",
        Tags.RECONSTRUCTION_MITK_SETTINGS_XML: "/home/janek/bin/beamformingsettings.xml",
        Tags.RECONSTRUCTION_BMODE_METHOD: Tags.RECONSTRUCTION_BMODE_METHOD_HILBERT_TRANSFORM,

        # Noise settings
        Tags.APPLY_NOISE_MODEL: True,
        Tags.NOISE_MODEL: Tags.NOISE_MODEL_GAUSSIAN,
        Tags.NOISE_MEAN: 0,
        Tags.NOISE_STD: 50,
    }
    print("Simulating ", random_seed)
    [settings_path, optical_path, acoustic_path, reconstruction_path] = simulate(settings)
    #perform_reconstruction(settings, "/media/janek/PA DATA/DS_Forearm/Forearm_17420000/acoustic_forward_model_output_700.npz")
    print("Simulating ", random_seed, "[Done]")