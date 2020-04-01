
import sys

sys.path.append("/workplace/ippai/")
from ippai.simulate.simulation import simulate
from ippai.simulate import SegmentationClasses, GeometryClasses
from ippai.simulate.structures import *
from ippai.utils import Tags
from ippai.utils import TISSUE_LIBRARY

#from ippai.simulate.tissue_properties import get_muscle_settings
from ippai.utils import randomize
from itertools import combinations

import pickle
import pandas as pd
import itertools
import random

import numpy as np

WIDTH_IN_MM = 44.2
MAX_NUM_OF_SPLITS = 7
SPACING = 0.34
SEED_INDEX = 0

def create_tissue():
    tissue_dict = dict()
    tissue_dict["background"] = create_muscle_background()
    tissue_dict["epidermis"] = create_epidermis_layer()
    num_splits = np.random.randint(2, (MAX_NUM_OF_SPLITS+1))
    width_increment = WIDTH_IN_MM / num_splits
    for i in range(num_splits):
        left = width_increment * i
        right = width_increment * (i+1)
        tissue_dict["vessel_" + str(i)] = create_vessel_tube(x_min=left+3, x_max=right-3,
                                                             z_min=3, z_max=19, r_min=1, r_max=3)
    return tissue_dict




while SEED_INDEX < 10:
    random_seed = 10000 + SEED_INDEX
    SEED_INDEX += 1
    np.random.seed(random_seed)

    settings = {
        Tags.WAVELENGTHS: np.arange(700, 951, 10),
        Tags.RANDOM_SEED: random_seed,
        Tags.VOLUME_NAME: "Vessels_"+str(random_seed),
        Tags.SIMULATION_PATH: "/workplace/test",
        Tags.RUN_OPTICAL_MODEL: True,
        Tags.OPTICAL_MODEL_NUMBER_PHOTONS: 1e7,
        Tags.OPTICAL_MODEL_BINARY_PATH: "/workplace/ippai/ippai/simulate/models/optical_models/mcx",
        Tags.OPTICAL_MODEL: Tags.MODEL_MCX,
        Tags.ILLUMINATION_TYPE: Tags.ILLUMINATION_TYPE_MSOT_ACUITY_ECHO,
        Tags.ILLUMINATION_DIRECTION: [0, 0.381070, 0.9245460],  # direction of msot acuity
        Tags.ILLUMINATION_PARAM1: [30 / SPACING, 0, 0, 0],  # Width of the MSOT probe
        Tags.LASER_PULSE_ENERGY_IN_MILLIJOULE: 50,
        Tags.RUN_ACOUSTIC_MODEL: False,
        Tags.APPLY_NOISE_MODEL: False,
        Tags.PERFORM_IMAGE_RECONSTRUCTION: False,
        Tags.SIMULATION_EXTRACT_FIELD_OF_VIEW: True,
        Tags.SPACING_MM: SPACING,
        Tags.DIM_VOLUME_Z_MM: WIDTH_IN_MM / 2,
        Tags.DIM_VOLUME_X_MM: WIDTH_IN_MM,
        Tags.DIM_VOLUME_Y_MM: 40,
        Tags.AIR_LAYER_HEIGHT_MM: 0,
        Tags.GELPAD_LAYER_HEIGHT_MM: 0,
        Tags.STRUCTURES: create_tissue()
    }
    print("Simulating ", random_seed)
    simulate(settings)
    print("Simulating ", random_seed, "[Done]")