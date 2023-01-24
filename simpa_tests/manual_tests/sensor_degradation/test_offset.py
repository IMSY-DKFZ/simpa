import os
import sys
import time
import numpy as np
import simpa as sp
from simpa.utils import Tags

parent_path = os.path.abspath(os.path.join(os.path.dirname(__file__),".."))
sys.path.append(parent_path)
from helpers import create_background, get_reconstruction_settings, NoiseExperiment


### REPRODUCIBILITY COMMENTS ###
# simpa branch: T179_SensorDegradation
# commit hash: e875e0b1f8f61a1335d946998473e28a0d96a81b

# matlab: R2022a Update 3 (9.12.0.1975300) 64-bit (glnxa64)
# k-Wave: v1.3

# mcx commit hash: ed24cccd733a442acf8585da246a69521e2f0ba5
#######

VOLUME_TRANSDUCER_DIM_IN_MM = 75
VOLUME_PLANAR_DIM_IN_MM = 20
VOLUME_HEIGHT_IN_MM = 25
SPACING = 0.2 #0.15625
#WAVELENGTHS = list(np.arange(700, 855, 10))
WAVELENGTHS = [800]

DEVICES = [1,2]

REALISM = 0#1 # 0 = Plain Water without Noise, 1 = Water with Noise, 2 = Forearm
#EXPERIMENT_NAME = "water_experiment"
EXPERIMENT_NAME = "water_experiment_only_offsets"
#EXPERIMENT_NAME = "water_experiment_only_thermalnoise"
#EXPERIMENT_NAME = "water_experiment_only_laserenergy"


try:
    run_by_bash: bool = bool(os.environ["RUN_BY_BASH"])
    print("This runner script is invoked in a bash script!")
except KeyError:
    run_by_bash: bool = False

if run_by_bash:
    RANDOM_SEED_START = int(os.environ['SEED_START'])
    NUMBER_OF_SIMULATIONS = int(os.environ["NUMBER_OF_SIMULATIONS"])
    MCX_SEED = int(os.environ['MCX_SEED'])
    env_path = os.environ['ENV_PATH']
    number_photons = int(os.environ['NUMBER_PHOTONS'])
    reconstruction_algorithm = os.environ["RECONSTRUCTION_ALGORITHM"]
else:
    RANDOM_SEED_START = 0
    NUMBER_OF_SIMULATIONS = 500 # per device
    MCX_SEED = 1234
    env_path = "/home/c738r/simpa"
    number_photons = 1e8
    reconstruction_algorithm = "DAS"

# NOTE: Please make sure that a valid path_config.env file is located in your home directory, or that you
# point to the correct file in the PathManager().
path_manager = sp.PathManager(environment_path=env_path)

SAVE_PATH = path_manager.get_hdf5_file_save_path()

# initialize NoiseExperiment object in order to load noise data and set the SensorDegradation Settings
NoiseExp = NoiseExperiment(experiment_name=EXPERIMENT_NAME, wavelengths=WAVELENGTHS, number_of_simulations=NUMBER_OF_SIMULATIONS)

# Seed the numpy random configuration prior to creating the global_settings file in
# order to ensure that the same volume
# is generated with the same random seed every time.

timer = time.time()

for DEVICE in DEVICES:
    # generate sample list which determines which noise data samples shale be used and read noise data
    np.random.seed(RANDOM_SEED_START) # ensures that sample lists are the same for both devices
    sample_list = NoiseExp.generate_sample_list_and_load_noise_data(device=DEVICE)
    assert len(sample_list) >= NUMBER_OF_SIMULATIONS

    # simulate NUMBER_OF_SIMULATIONS times
    for n, RANDOM_SEED in enumerate(range(RANDOM_SEED_START, RANDOM_SEED_START + NUMBER_OF_SIMULATIONS)):
        # get index of noise sample
        sample_index = sample_list[n]
        sample_index = 12 # uncomment if you want to simulate an example once, do also break at the end in this case
        np.random.seed(RANDOM_SEED)
        if REALISM == 0:
            VOLUME_NAME = "PlainWater_" + str(RANDOM_SEED) # without SensorDegradation
        elif REALISM == 1:
            VOLUME_NAME = "Water_" + str(RANDOM_SEED) + "_" + str(sample_index) # with SensorDegradation
        elif REALISM == 2:
            raise("Nicht fertig überdacht")
            #VOLUME_NAME = "Forearm_" + str(RANDOM_SEED)
        
        if REALISM > 0:
            VOLUME_NAME += f"_y{DEVICE-1}" # denoting the class of the simulated data

        VOLUME_PATH = os.path.join(SAVE_PATH, EXPERIMENT_NAME, VOLUME_NAME)
        os.makedirs(VOLUME_PATH, exist_ok=True)
        for wavelength in WAVELENGTHS:
            np.random.seed(RANDOM_SEED)

            general_settings = {
                        # These parameters set the general properties of the simulated volume
                        Tags.RANDOM_SEED: RANDOM_SEED,
                        Tags.VOLUME_NAME: VOLUME_NAME + f"_Wavelength_{wavelength}",
                        Tags.SIMULATION_PATH: os.path.join(VOLUME_PATH),
                        Tags.SPACING_MM: SPACING,
                        Tags.DIM_VOLUME_Z_MM: VOLUME_HEIGHT_IN_MM,
                        Tags.DIM_VOLUME_X_MM: VOLUME_TRANSDUCER_DIM_IN_MM,
                        Tags.DIM_VOLUME_Y_MM: VOLUME_PLANAR_DIM_IN_MM,
                        Tags.VOLUME_CREATOR: Tags.VOLUME_CREATOR_VERSATILE,
                        Tags.GPU: True,
                        # Tags.SIMULATE_DEFORMED_LAYERS: True,

                        # The following parameters set the optical forward model
                        Tags.WAVELENGTHS: [wavelength],
                        Tags.DO_FILE_COMPRESSION: True
                    }
            settings = sp.Settings(general_settings)
            np.random.seed(RANDOM_SEED)

            # water volume
            settings.set_volume_creation_settings({
                Tags.STRUCTURES: create_background()
            })

            # forearm volume
            """settings.set_volume_creation_settings({
                Tags.STRUCTURES: create_forearm_tissue(settings),
                Tags.SIMULATE_DEFORMED_LAYERS: True,
                Tags.DEFORMED_LAYERS_SETTINGS: sp.create_deformation_settings(
                    bounds_mm=[[0, VOLUME_TRANSDUCER_DIM_IN_MM],
                            [0, VOLUME_PLANAR_DIM_IN_MM]],
                    maximum_z_elevation_mm=3,
                    filter_sigma=0.5,
                    cosine_scaling_factor=1),
                Tags.US_GEL: True,

            })"""

            settings.set_optical_settings({
                Tags.OPTICAL_MODEL_NUMBER_PHOTONS: number_photons,
                Tags.OPTICAL_MODEL_BINARY_PATH: path_manager.get_mcx_binary_path(),
                Tags.LASER_PULSE_ENERGY_IN_MILLIJOULE: 50,
                Tags.MCX_ASSUMED_ANISOTROPY: 0.9,
                Tags.MCX_SEED: MCX_SEED,
            })

            settings.set_acoustic_settings({
                Tags.ACOUSTIC_SIMULATION_3D: False, # makes sense
                Tags.ACOUSTIC_MODEL_BINARY_PATH: path_manager.get_matlab_binary_path(), # makes sense
                Tags.KWAVE_PROPERTY_ALPHA_POWER: 1.05, # NOT DEFAULT VALUE (DEFAULT IS 0.0) TODO ASK
                Tags.KWAVE_PROPERTY_SENSOR_RECORD: "p", # default: 'p'
                Tags.KWAVE_PROPERTY_PlotPML: False, # default: False
                Tags.RECORDMOVIE: False, # default: False
                Tags.MOVIENAME: "visualization_log", # default: 'visualization_log'
                Tags.ACOUSTIC_LOG_SCALE: True, # default: True
                Tags.KWAVE_PROPERTY_PMLInside: False, # default: False
                Tags.KWAVE_PROPERTY_PMLSize: [25, 25], #TODO ASK
                Tags.KWAVE_PROPERTY_PMLAlpha: 1.5, # default: 1.5
                Tags.MODEL_SENSOR_FREQUENCY_RESPONSE: False, # TODO: ASK
                Tags.KWAVE_PROPERTY_INITIAL_PRESSURE_SMOOTHING: False, # TODO: ASK
            })

            settings.set_reconstruction_settings(
                get_reconstruction_settings(matlab_path = path_manager.get_matlab_binary_path(),
                                            spacing=SPACING)
                )

            #settings["noise_initial_pressure"] = {
            #    Tags.NOISE_MEAN: 1,
            #    Tags.NOISE_STD: 0.09,
            #    Tags.NOISE_MODE: Tags.NOISE_MODE_MULTIPLICATIVE,
            #    Tags.DATA_FIELD: Tags.DATA_FIELD_INITIAL_PRESSURE,
            #    Tags.NOISE_NON_NEGATIVITY_CONSTRAINT: True
            #}

            #settings["noise_time_series"] = {
            #    Tags.NOISE_STD: 50,
            #    Tags.NOISE_MODE: Tags.NOISE_MODE_ADDITIVE,
            #    Tags.DATA_FIELD: Tags.DATA_FIELD_TIME_SERIES_DATA
            #}

            # Set settings for noise of given Device
            if REALISM > 0:
                devicenoise_string = f"device_{DEVICE}"
                settings[devicenoise_string] = NoiseExp.get_noise_parameters(device=DEVICE, wavelength=wavelength, sample_index=sample_index)
                       
            # Initialize MSOT Acuity Echo
            device = sp.MSOTAcuityEcho(device_position_mm=np.array([VOLUME_TRANSDUCER_DIM_IN_MM/2,
                                                                    VOLUME_PLANAR_DIM_IN_MM/2,
                                                                    0]),
                                    field_of_view_extent_mm=np.array(
                                        [-20, 20, 0, 0, -4, 16]))
            device.update_settings_for_use_of_model_based_volume_creator(settings)

            if reconstruction_algorithm == "DAS":
                reconstruction_adapter = sp.DelayAndSumAdapter
            elif reconstruction_algorithm == "TR":
                reconstruction_adapter = sp.TimeReversalAdapter
            else:
                raise KeyError("Please select a valid reconstruction algorithm from 'DAS' or 'TR'!")

            if REALISM == 0:
                SIMUATION_PIPELINE = [
                    sp.ModelBasedVolumeCreationAdapter(settings),
                    sp.MCXAdapter(settings),
                    #sp.GaussianNoise(settings, "noise_initial_pressure"),
                    sp.KWaveAdapter(settings),
                    # sp.GaussianNoise(settings, "noise_time_series"),
                    #sp.SensorDegradation(settings, device_string),
                    reconstruction_adapter(settings),
                    sp.FieldOfViewCropping(settings)
                    ]
            elif REALISM > 0:
                SIMUATION_PIPELINE = [
                    sp.ModelBasedVolumeCreationAdapter(settings),
                    sp.MCXAdapter(settings),
                    #sp.GaussianNoise(settings, "noise_initial_pressure"),
                    sp.KWaveAdapter(settings),
                    # sp.GaussianNoise(settings, "noise_time_series"),
                    sp.SensorDegradation(settings, devicenoise_string),
                    reconstruction_adapter(settings),
                    sp.FieldOfViewCropping(settings)
                    ]
          
            sp.simulate(SIMUATION_PIPELINE, settings, device)
        
        break # uncomment if you just one to simulate each device once
    break # uncomment if you just one to simulate only the first device once

print("Needed", time.time() - timer, "seconds")
