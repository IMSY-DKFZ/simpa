from ippai.simulate import Tags
from ippai.simulate.simulation import simulate
from ippai.simulate.structures import *
from ippai.simulate.tissue_properties import get_constant_settings
import numpy as np


def create_water_background():
    water_dict = dict()
    water_dict[Tags.STRUCTURE_TYPE] = Tags.STRUCTURE_BACKGROUND
    water_dict[Tags.STRUCTURE_TISSUE_PROPERTIES] = get_constant_settings(mua=1e-5, mus=1e-5, g=1)
    water_dict[Tags.STRUCTURE_SEGMENTATION_TYPE] = SegmentationClasses.ULTRASOUND_GEL_PAD
    return water_dict


def create_agar_phantom():
    phantom_dict = dict()
    phantom_dict[Tags.STRUCTURE_TYPE] = Tags.STRUCTURE_TUBE
    phantom_dict[Tags.STRUCTURE_CENTER_DEPTH_MIN_MM] = 12
    phantom_dict[Tags.STRUCTURE_CENTER_DEPTH_MAX_MM] = 12
    phantom_dict[Tags.STRUCTURE_RADIUS_MIN_MM] = 9.5
    phantom_dict[Tags.STRUCTURE_RADIUS_MAX_MM] = 9.5
    phantom_dict[Tags.STRUCTURE_TUBE_CENTER_X_MIN_MM] = 12
    phantom_dict[Tags.STRUCTURE_TUBE_CENTER_X_MAX_MM] = 12
    phantom_dict[Tags.STRUCTURE_TISSUE_PROPERTIES] = get_settings(w_max=1, w_min=0.5,
                                                                  oxy_min=-1, oxy_max=-1,
                                                                  musp500=5)
    phantom_dict[Tags.STRUCTURE_SEGMENTATION_TYPE] = SegmentationClasses.GENERIC
    return phantom_dict


def create_flow_vessel():
    vessel_dict = dict()
    vessel_dict[Tags.STRUCTURE_TYPE] = Tags.STRUCTURE_TUBE
    vessel_dict[Tags.STRUCTURE_CENTER_DEPTH_MIN_MM] = 12
    vessel_dict[Tags.STRUCTURE_CENTER_DEPTH_MAX_MM] = 12
    vessel_dict[Tags.STRUCTURE_RADIUS_MIN_MM] = 0.5
    vessel_dict[Tags.STRUCTURE_RADIUS_MAX_MM] = 1.25
    vessel_dict[Tags.STRUCTURE_TUBE_CENTER_X_MIN_MM] = 12
    vessel_dict[Tags.STRUCTURE_TUBE_CENTER_X_MAX_MM] = 12
    vessel_dict[Tags.STRUCTURE_TISSUE_PROPERTIES] = get_blood_settings()
    vessel_dict[Tags.STRUCTURE_SEGMENTATION_TYPE] = SegmentationClasses.BLOOD
    return vessel_dict


def create_flow_phantom_parameters():
    structures_dict = dict()
    structures_dict["background"] = create_water_background()
    structures_dict["phantom"] = create_agar_phantom()
    structures_dict["vessel"] = create_flow_vessel()
    return structures_dict


seed_index = 0
while seed_index < 1000:
    random_seed = 200000 + seed_index
    seed_index += 1
    np.random.seed(random_seed)

    settings = {
        Tags.WAVELENGTHS: [660, 664, 680, 684, 694, 700, 708, 715, 730, 735, 760, 770, 775, 779, 800, 850, 950],
        Tags.RANDOM_SEED: random_seed,
        Tags.VOLUME_NAME: "FlowPhantom_"+str(random_seed).zfill(6),
        Tags.SIMULATION_PATH: "/media/janek/Maxtor/flow_phantom_simulation/",
        Tags.RUN_OPTICAL_MODEL: True,
        Tags.OPTICAL_MODEL_NUMBER_PHOTONS: 1e7,
        Tags.OPTICAL_MODEL_BINARY_PATH: "/home/janek/simulation_test/mcx",
        Tags.OPTICAL_MODEL: Tags.MODEL_MCX,
        Tags.RUN_ACOUSTIC_MODEL: False,
        Tags.SIMULATION_EXTRACT_FIELD_OF_VIEW: True,
        Tags.SPACING_MM: 0.3,
        Tags.DIM_VOLUME_Z_MM: 25,
        Tags.DIM_VOLUME_X_MM: 25,
        Tags.DIM_VOLUME_Y_MM: 10,
        Tags.AIR_LAYER_HEIGHT_MM: 0,
        Tags.GELPAD_LAYER_HEIGHT_MM: 0,
        Tags.STRUCTURES: create_flow_phantom_parameters()
    }
    print("Simulating ", random_seed)
    [settings_path, optical_path, acoustic_path] = simulate(settings)
    print("Simulating ", random_seed, "[Done]")
