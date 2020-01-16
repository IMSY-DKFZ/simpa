from ippai.simulate import Tags
from ippai.simulate.simulation import simulate
from ippai.simulate.tissue_properties import get_muscle_settings
from ippai.simulate.structures import create_forearm_structures

import numpy as np


spacings = [0.34]
photons = [1e8]

seed_index = 0
while seed_index < 1000:
    # 465 < 35
    random_seed = 1000 + seed_index
    seed_index += 1
    np.random.seed(random_seed)

    relative_shift = ((np.random.random() - 0.5) * 2) * 12.5 + 26
    background_oxy = (np.random.random() * 0.2) + 0.4
    # wavelength = [np.random.randint(700, 950)]

    for [spacing, num_photons] in zip(spacings, photons):
        settings = {
            Tags.WAVELENGTHS: [800],  # list(range(700, 951, 10)),
            Tags.RANDOM_SEED: random_seed,
            Tags.VOLUME_NAME: "forearm_"+str(random_seed).zfill(6) + "/spacing_{}".format(spacing),
            Tags.SIMULATION_PATH: "/media/kris/Extreme SSD/data/forearm",
            Tags.RUN_OPTICAL_MODEL: True,
            Tags.OPTICAL_MODEL_NUMBER_PHOTONS: num_photons,
            Tags.OPTICAL_MODEL_BINARY_PATH: "/home/kris/hard_drive/ippai/ippai/simulate/models/optical_models/mcx",
            Tags.OPTICAL_MODEL: Tags.MODEL_MCX,
            Tags.RUN_ACOUSTIC_MODEL: True,
            'background_properties': get_muscle_settings(),
            Tags.SIMULATION_EXTRACT_FIELD_OF_VIEW: True,
            Tags.SPACING_MM: spacing,
            Tags.DIM_VOLUME_Z_MM: 45,
            Tags.DIM_VOLUME_X_MM: 90,
            Tags.DIM_VOLUME_Y_MM: 40,

            Tags.STRUCTURES: create_forearm_structures(relative_shift_mm=relative_shift, background_oxy=background_oxy),

            Tags.ILLUMINATION_TYPE: Tags.ILLUMINATION_TYPE_MSOT_ACUITY_ECHO,

            Tags.ILLUMINATION_DIRECTION: [0, 0.381070, 0.9245460],  # direction of msot acuity
            Tags.ILLUMINATION_PARAM1: [30 / spacing, 0, 0, 0],

            # Upsampling settings

            Tags.PERFORM_UPSAMPLING: True,
            Tags.CROP_IMAGE: True,
            Tags.CROP_POWER_OF_TWO: True,
            Tags.UPSAMPLING_METHOD: Tags.UPSAMPLING_METHOD_DEEP_LEARNING,
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
            Tags.SENSOR_CENTER_FREQUENCY_HZ: 4e6,
            Tags.SENSOR_BANDWIDTH_PERCENT: 80,
            Tags.SENSOR_DIRECTIVITY_HOMOGENEOUS: True,
            # Tags.SENSOR_DIRECTIVITY_ANGLE: 0,
            # Tags.SENSOR_DIRECTIVITY_ANGLE: "/home/kris/hard_drive/data/pipeline_test/sensor_directivity.npy",
            # 0,   # Most sensitive in x-dir (up/down)
            Tags.SENSOR_DIRECTIVITY_SIZE_M: 0.00024,  # [m]
            Tags.SENSOR_DIRECTIVITY_PATTERN: "pressure",

            Tags.SENSOR_ELEMENT_PITCH_MM: 0.34,
            # Tags.SENSOR_DIRECTIVITY_SIZE_M: 0.00000024,  # [m]
            Tags.SENSOR_SAMPLING_RATE_MHZ: 40,
            Tags.SENSOR_NUM_ELEMENTS: 256,
            Tags.SENSOR_RADIUS_MM: 40,

            Tags.PMLInside: False,
            Tags.PMLSize: [31, 32],
            Tags.PMLAlpha: 1.5,
            Tags.PlotPML: False,
            Tags.RECORDMOVIE: True,
            Tags.MOVIENAME: "visualization",
            Tags.ACOUSTIC_LOG_SCALE: False,

            # Reconstruction

            Tags.PERFORM_IMAGE_RECONSTRUCTION: True,
            Tags.RECONSTRUCTION_ALGORITHM: Tags.RECONSTRUCTION_ALGORITHM_DAS,
            Tags.RECONSTRUCTION_BMODE_METHOD: Tags.RECONSTRUCTION_BMODE_METHOD_HILBERT_TRANSFORM,
            Tags.RECONSTRUCTION_MITK_BINARY_PATH: "/home/kris/hard_drive/MITK/"
                                                  "sDMAS-2018.07-2596-g31d1c60d71-linux-x86_64/"
                                                  "MITK-experiments/sDMAS-2018.07-2596-g31d1c60d71-linux-x86_64/"
                                                  "MitkPABeamformingTool.sh",
            Tags.RECONSTRUCTION_MITK_SETTINGS_XML: "/home/kris/hard_drive/data/pipeline_test/bf_settings.xml",
            Tags.RECONSTRUCTION_OUTPUT_NAME: "/home/kris/hard_drive/data/pipeline_test/test.nrrd",

            # Noise

            Tags.APPLY_NOISE_MODEL: False,
            Tags.NOISE_MODEL: Tags.NOISE_MODEL_GAUSSIAN,
            Tags.NOISE_MODEL_PATH: "/home/kris/hard_drive/cami-experimental/PAI/MCX/"
                                   "probe_integration/noise_model_msot_acuity.csv"
        }
        print("Simulating ", random_seed)
        [settings_path, optical_path, acoustic_path, reconstructed_path] = simulate(settings)
        print("Simulating ", random_seed, "[Done]")
        break
    break
