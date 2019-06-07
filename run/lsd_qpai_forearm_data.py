from ippai.simulate import Tags
from ippai.simulate.simulation import simulate
from ippai.simulate.tissue_properties import get_muscle_settings
from ippai.simulate.structures import create_forearm_structures

import numpy as np


seed_index = 0
while True:
    random_seed = 11000 + seed_index
    seed_index += 1
    np.random.seed(random_seed)

    relative_shift = ((np.random.random() - 0.5) * 2) * 12.5
    background_oxy = (np.random.random() * 0.6) + 0.2

    settings = {
        Tags.WAVELENGTHS: np.arange(700, 951, 10),
        Tags.RANDOM_SEED: random_seed,
        Tags.VOLUME_NAME: "Forearm_"+str(random_seed).zfill(6),
        Tags.SIMULATION_PATH: "/home/janek/simulation_test/",
        Tags.RUN_OPTICAL_MODEL: True,
        Tags.OPTICAL_MODEL_NUMBER_PHOTONS: 1e7,
        Tags.OPTICAL_MODEL_BINARY_PATH: "/home/janek/mitk-superbuild/MITK-build/bin/MitkMCxyz",
        Tags.OPTICAL_MODEL_PROBE_XML_FILE: "/home/janek/CAMI_PAT_SETUP_V2.xml",
        Tags.OPTICAL_MODEL: Tags.MODEL_MCXYZ,
        Tags.RUN_ACOUSTIC_MODEL: False,
        'background_properties': get_muscle_settings(),
        Tags.SPACING_MM: 0.3,
        Tags.DIM_VOLUME_Z_MM: 30,
        Tags.DIM_VOLUME_X_MM: 40,
        Tags.DIM_VOLUME_Y_MM: 40,
        Tags.AIR_LAYER_HEIGHT_MM: 12,
        Tags.GELPAD_LAYER_HEIGHT_MM: 18,
        Tags.STRUCTURES: create_forearm_structures(relative_shift_mm=relative_shift, background_oxy=background_oxy)
    }

    [settings_path, optical_path, acoustic_path] = simulate(settings)
