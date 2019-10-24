from ippai.simulate import Tags
from ippai.simulate.simulation import simulate
from ippai.simulate.models.reconstruction import perform_reconstruction
from ippai.simulate.structures import create_unrealistic_forearm_structures
import numpy as np


for seed_index in range(0, 500):

    random_seed = 17420000 + seed_index
    seed_index += 1
    np.random.seed(random_seed)

    relative_shift = ((np.random.random() - 0.5) * 2) * 12.5
    background_oxy = (np.random.random() * 0.2) + 0.4

    settings = {

        # Basic & geometry settings
        Tags.RANDOM_SEED: random_seed,
        Tags.VOLUME_NAME: "Forearm_" + str(random_seed).zfill(6),
        Tags.SIMULATION_PATH: "/media/janek/PA DATA/DS_Forearm/",
        Tags.SPACING_MM: 0.3,
        Tags.DIM_VOLUME_Z_MM: 21,
        Tags.DIM_VOLUME_X_MM: 40,
        Tags.DIM_VOLUME_Y_MM: 25,
        Tags.AIR_LAYER_HEIGHT_MM: 12,
        Tags.GELPAD_LAYER_HEIGHT_MM: 18,
        Tags.STRUCTURES: create_unrealistic_forearm_structures(relative_shift_mm=relative_shift,
                                                               background_oxy=background_oxy,
                                                               radius_factor=1,
                                                               vessel_spawn_probability=0.5),
        Tags.SIMULATION_EXTRACT_FIELD_OF_VIEW: True,
        Tags.LASER_PULSE_ENERGY_IN_MILLIJOULE: 50,

        # Optical forward path settings
        Tags.WAVELENGTHS: [700, 750, 800, 850, 900],
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
        Tags.RECORDMOVIE: True,
        Tags.MOVIENAME: "Movie",

        # Reconstruction settings
        Tags.PERFORM_IMAGE_RECONSTRUCTION: True,
        Tags.RECONSTRUCTION_ALGORITHM: Tags.RECONSTRUCTION_ALGORITHM_DAS,
        Tags.RECONSTRUCTION_MITK_BINARY_PATH: "/home/janek/bin/mitk/MitkPABeamformingTool.sh",
        Tags.RECONSTRUCTION_MITK_SETTINGS_XML: "/home/janek/bin/beamformingsettings.xml",
        Tags.RECONSTRUCTION_BMODE_METHOD: Tags.RECONSTRUCTION_BMODE_METHOD_HILBERT_TRANSFORM
    }
    print("Simulating ", random_seed)
    [settings_path, optical_path, acoustic_path, reconstruction_path] = simulate(settings)
    print("Simulating ", random_seed, "[Done]")
