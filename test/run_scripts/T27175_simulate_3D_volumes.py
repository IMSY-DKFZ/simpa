from ippai.simulate import Tags
from ippai.simulate.simulation import simulate
from ippai.simulate.structures import *
from ippai.simulate.tissue_properties import get_muscle_settings


def create_background():
    muscle_dict = dict()
    muscle_dict[Tags.STRUCTURE_TYPE] = Tags.STRUCTURE_BACKGROUND
    muscle_dict[Tags.STRUCTURE_TISSUE_PROPERTIES] = get_muscle_settings()
    muscle_dict[Tags.STRUCTURE_SEGMENTATION_TYPE] = SegmentationClasses.MUSCLE
    muscle_dict[Tags.STRUCTURE_USE_DISTORTION] = True
    muscle_dict[Tags.STRUCTURE_DISTORTED_PARAM_LIST] = [Tags.KEY_BLOOD, Tags.KEY_OXY, Tags.KEY_WATER]
    muscle_dict[Tags.STRUCTURE_DISTORTION_FREQUENCY_PER_MM] = 2
    return muscle_dict


def create_random_tube():
    tube_dict = dict()
    mua = np.random.random() * 10
    mus = np.random.random() * 100 + 50
    g = np.random.random() * 0.2 + 0.8
    tube_dict[Tags.STRUCTURE_TYPE] = Tags.STRUCTURE_TUBE
    tube_dict[Tags.STRUCTURE_TISSUE_PROPERTIES] = get_constant_settings(mua=mua, mus=mus, g=g)
    tube_dict[Tags.STRUCTURE_USE_DISTORTION] = True
    tube_dict[Tags.STRUCTURE_DISTORTED_PARAM_LIST] = [Tags.KEY_MUA, Tags.KEY_MUS, Tags.KEY_G]
    tube_dict[Tags.STRUCTURE_DISTORTION_FREQUENCY_PER_MM] = 2
    tube_dict[Tags.STRUCTURE_CENTER_DEPTH_MIN_MM] = 0
    tube_dict[Tags.STRUCTURE_CENTER_DEPTH_MAX_MM] = 17
    tube_dict[Tags.STRUCTURE_RADIUS_MIN_MM] = 0.5
    tube_dict[Tags.STRUCTURE_RADIUS_MAX_MM] = 5
    tube_dict[Tags.STRUCTURE_TUBE_CENTER_X_MIN_MM] = 0
    tube_dict[Tags.STRUCTURE_TUBE_CENTER_X_MAX_MM] = 17
    tube_dict[Tags.STRUCTURE_SEGMENTATION_TYPE] = SegmentationClasses.GENERIC
    return tube_dict


def create_random_structures():
    structures_dict = dict()
    structures_dict["background"] = create_background()
    for i in range(5):
        structures_dict["tube_" + str(i).zfill(2)] = create_random_tube()
    return structures_dict


seed_index = 0
while seed_index < 1:
    random_seed = 100000 + seed_index
    seed_index += 1
    np.random.seed(random_seed)
    settings = {
        Tags.WAVELENGTHS: [700, 750, 800, 850, 900, 950],
        Tags.RANDOM_SEED: random_seed,
        Tags.VOLUME_NAME: "RandomVolume_" + str(random_seed).zfill(6),
        Tags.SIMULATION_PATH: "/home/janek/melanie_test/",
        Tags.RUN_OPTICAL_MODEL: True,
        Tags.OPTICAL_MODEL_NUMBER_PHOTONS: 1e7,
        Tags.OPTICAL_MODEL_BINARY_PATH: "/home/janek/simulation_test/mcx",
        Tags.OPTICAL_MODEL: Tags.MODEL_MCX,
        Tags.ILLUMINATION_TYPE: Tags.ILLUMINATION_TYPE_PENCIL,
        Tags.RUN_ACOUSTIC_MODEL: False,
        Tags.SIMULATION_EXTRACT_FIELD_OF_VIEW: False,
        Tags.SPACING_MM: 0.5,
        Tags.DIM_VOLUME_Z_MM: 17,
        Tags.DIM_VOLUME_X_MM: 17,
        Tags.DIM_VOLUME_Y_MM: 17,
        Tags.ILLUMINATION_POSITION: [18, 18, 0.1],
        Tags.ILLUMINATION_DIRECTION: [0, 0, 1],  # direction of msot acuity
        Tags.AIR_LAYER_HEIGHT_MM: 0.5,
        Tags.GELPAD_LAYER_HEIGHT_MM: 0,
        Tags.STRUCTURES: create_random_structures()
    }
    print("Simulating ", random_seed)
    [settings_path, optical_path, acoustic_path, reco_path] = simulate(settings)
    print("Simulating ", random_seed, "[Done]")