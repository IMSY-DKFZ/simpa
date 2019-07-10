from ippai.simulate import Tags
from ippai.simulate.simulation import simulate
from ippai.simulate.tissue_properties import get_muscle_settings
from ippai.simulate.structures import create_unrealistic_forearm_structures

import numpy as np


seed_index = 0
while seed_index < 181:
    random_seed = 40819 + seed_index
    seed_index += 1
    np.random.seed(random_seed)

    relative_shift = ((np.random.random() - 0.5) * 2) * 12.5
    background_oxy = (np.random.random() * 0.2) + 0.4

    settings = {
        Tags.WAVELENGTHS: np.arange(700, 951, 10),
        Tags.RANDOM_SEED: random_seed,
        Tags.VOLUME_NAME: "Forearm_"+str(random_seed).zfill(6),
        Tags.SIMULATION_PATH: "/media/janek/Maxtor/mcx_simulation/",
        Tags.RUN_OPTICAL_MODEL: True,
        Tags.OPTICAL_MODEL_NUMBER_PHOTONS: 1e7,
        Tags.OPTICAL_MODEL_BINARY_PATH: "/home/janek/simulation_test/mcx",
        Tags.OPTICAL_MODEL: Tags.MODEL_MCX,
        Tags.RUN_ACOUSTIC_MODEL: False,
        'background_properties': get_muscle_settings(),
        Tags.SIMULATION_EXTRACT_FIELD_OF_VIEW: True,
        Tags.SPACING_MM: 0.3,
        Tags.DIM_VOLUME_Z_MM: 25,
        Tags.DIM_VOLUME_X_MM: 40,
        Tags.DIM_VOLUME_Y_MM: 25,
        Tags.AIR_LAYER_HEIGHT_MM: 12,
        Tags.GELPAD_LAYER_HEIGHT_MM: 18,
        Tags.STRUCTURES: create_unrealistic_forearm_structures(relative_shift_mm=relative_shift,
                                                               background_oxy=background_oxy,
                                                               radius_factor=1.5,
                                                               vessel_spawn_probability=0.3)
    }
    print("Simulating ", random_seed)
    [settings_path, optical_path, acoustic_path] = simulate(settings)
    print("Simulating ", random_seed, "[Done]")
