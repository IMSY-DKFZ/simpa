
import sys

sys.path.append("/workplace/ippai/")


from ippai.utils import Tags
#from ippai.simulate.constants import SegmentationClasses, GeometryClasses
from ippai.simulate.simulation import simulate
from ippai.simulate.structures import *
from ippai.utils.calculate import randomize

import pickle
import pandas as pd
import itertools
import random
import os
#lst = list(itertools.product([0, 1], repeat=52))


def create_settings_dict():
    settings_dict=dict()
    "This dictionary includes all extrema that can be arranged in the learning process"
    
    settings_dict["randomness"] = True
    settings_dict["background_mua"] = [0, 0.1]
    settings_dict["background_mus"] = [50, 150]
    settings_dict["background_g"] = [0.8, 1]
    settings_dict["background_options"] = [3, 3]

    settings_dict["distortion"] = [1,1]
    settings_dict["distortion_frequency"] = [0, 4] 

    settings_dict["tube_mua"] = [0, 10]
    settings_dict["tube_mus"] = [50, 150]
    settings_dict["tube_g"] = [0.8, 1]
    settings_dict["tube_radius"] = [0.5, 5]
    settings_dict["tube_center_depth"] = [0, 17]
    settings_dict["tube_center_x"] = [0, 17]

    settings_dict["layer_mua"] = [0, 10]
    settings_dict["layer_mus"] = [50, 150]
    settings_dict["layer_g"] = [0.8, 1]
    settings_dict["layer_center_depth"] = [0, 17]
    settings_dict["layer_thickness"] = [0, 5]

    settings_dict["sphere_mua"] = [0, 10]
    settings_dict["sphere_mus"] = [50, 150]
    settings_dict["sphere_g"] = [0.8, 1]
    settings_dict["sphere_radius"] = [0.5, 5]
    settings_dict["sphere_center_depth"] = [0,17]
    settings_dict["sphere_center_x"] = [0,17]
    settings_dict["sphere_center_y"] = [0,17]

    settings_dict["cube_mua"] = [0, 10]
    settings_dict["cube_mus"] = [50, 150]
    settings_dict["cube_g"] = [0.8, 1]
    settings_dict["cube_center_depth"] = [0, 17]
    settings_dict["cube_center_x"] = [0,17]
    settings_dict["cube_center_y"] = [0, 17]
    settings_dict["cube_length_x"] = [0.5, 1.5]
    settings_dict["cube_length_y"] = [0.5, 1.5]
    settings_dict["cube_length_z"] = [0.5, 1.5]

    settings_dict["cubical_tube_mua"] = [0, 10]
    settings_dict["cubical_tube_mus"] = [50, 150]
    settings_dict["cubical_tube_g"] = [0.8, 1]
    settings_dict["cubical_tube_radius"] = [0.5, 5]
    settings_dict["cubical_tube_center_depth"] = [0, 17]
    settings_dict["cubical_tube_center_x"] = [0,17]

    settings_dict["pyramid_mua"] = [0, 10]
    settings_dict["pyramid_mus"] = [50, 150]
    settings_dict["pyramid_g"] = [0.8, 1]
    settings_dict["pyramid_height"] = [0, 10]
    settings_dict["pyramid_center_depth"] = [0,17]
    settings_dict["pyramid_center_x"] = [0,17]
    settings_dict["pyramid_center_y"] = [0,17]
    settings_dict["pyramid_basis_extent"] = [0, 10]
  #  settings_dict["pyramid_orientation_xy"] = [0, 90] 
  #  settings_dict["pyramid_orientation_xz"] = [0, 90]
  #  settings_dict["pyramid_orientation_yz"]= [0, 90] 
    settings_dict["pyramid_orientation"] = [0, 0]
    return settings_dict
  
def create_geometries_dict(): 
    geometries_dict = dict()
    geometries_dict["randomness"] = [True, True]

    geometries_dict["0"] = "air",
    geometries_dict["1"] = "background",
    geometries_dict["2"] = "layer" ,
    geometries_dict["3"] = "tube",
    geometries_dict["4"] = "sphere" ,
    geometries_dict["5"] = "cubical_tube",
    geometries_dict["6"] = "cube",
    geometries_dict["7"] = "pyramid",

    return geometries_dict



# def create_epidermis_layer(settings_dict, background_oxy=0.0):
#     epidermis_dict = dict()
#     if settings_dict["randomness"] == True:
#         epidermis_dict[Tags.STRUCTURE_USE_DISTORTION] = settings_dict["distortion"][0]
#         epidermis_dict[Tags.STRUCTURE_DISTORTION_FREQUENCY_PER_MM] = np.random.random() * (settings_dict["distortion_frequency"][1] - settings_dict["distortion_frequency"][0])+ settings_dict["distortion_frequency"][0]
#         epidermis_dict[Tags.STRUCTURE_CENTER_DEPTH_MIN_MM] = 0
#         epidermis_dict[Tags.STRUCTURE_CENTER_DEPTH_MAX_MM] = 0
#         epidermis_dict[Tags.STRUCTURE_THICKNESS_MIN_MM] = MorphologicalTissueProperties.EPIDERMIS_THICKNESS_MEAN_MM - MorphologicalTissueProperties.EPIDERMIS_THICKNESS_STD_MM
#         epidermis_dict[Tags.STRUCTURE_THICKNESS_MAX_MM] = MorphologicalTissueProperties.EPIDERMIS_THICKNESS_MEAN_MM + MorphologicalTissueProperties.EPIDERMIS_THICKNESS_STD_MM
#         epidermis_dict[Tags.STRUCTURE_CENTER_DEPTH_MM] = randomize(epidermis_dict[Tags.STRUCTURE_CENTER_DEPTH_MIN_MM], epidermis_dict[Tags.STRUCTURE_CENTER_DEPTH_MAX_MM])
#         epidermis_dict[Tags.STRUCTURE_THICKNESS_MM] = randomize(epidermis_dict[Tags.STRUCTURE_THICKNESS_MIN_MM], epidermis_dict[Tags.STRUCTURE_THICKNESS_MAX_MM])
#     else:
#         epidermis_dict[Tags.STRUCTURE_USE_DISTORTION] = settings_dict["distortion"][0]
#         epidermis_dict[Tags.STRUCTURE_DISTORTION_FREQUENCY_PER_MM] = settings_dict["distortion_frequency"][0]
#         epidermis_dict[Tags.STRUCTURE_CENTER_DEPTH_MM] = 0 
#         epidermis_dict[Tags.STRUCTURE_THICKNESS_MM] = MorphologicalTissueProperties.EPIDERMIS_THICKNESS_MEAN_MM
#     epidermis_dict[Tags.STRUCTURE_DISTORTED_PARAM_LIST] = [Tags.KEY_OXY, Tags.KEY_WATER]
#     epidermis_dict[Tags.STRUCTURE_TISSUE_PROPERTIES] = get_epidermis_settings(background_oxy=background_oxy)
#     epidermis_dict[Tags.STRUCTURE_TYPE] = Tags.STRUCTURE_LAYER
#     epidermis_dict[Tags.STRUCTURE_SEGMENTATION_TYPE] = SegmentationClasses.EPIDERMIS
#     epidermis_dict[Tags.STRUCTURE_GEOMETRY_TYPE] = GeometryClasses.LAYER
#     return epidermis_dict


# def create_dermis_layer(settings_dict, background_oxy=0.0):
#     dermis_dict = dict()
#     if settings_dict["randomness"] == True:
#         dermis_dict[Tags.STRUCTURE_USE_DISTORTION] = settings_dict["distortion"][0]
#         dermis_dict[Tags.STRUCTURE_DISTORTION_FREQUENCY_PER_MM] = np.random.random() * (settings_dict["distortion_frequency"][1] - settings_dict["distortion_frequency"][0])+ settings_dict["distortion_frequency"][0]
#         dermis_dict[Tags.STRUCTURE_CENTER_DEPTH_MIN_MM] = 0
#         dermis_dict[Tags.STRUCTURE_CENTER_DEPTH_MAX_MM] = 0
#         dermis_dict[Tags.STRUCTURE_THICKNESS_MIN_MM] = MorphologicalTissueProperties.DERMIS_THICKNESS_MEAN_MM - MorphologicalTissueProperties.DERMIS_THICKNESS_STD_MM
#         dermis_dict[Tags.STRUCTURE_THICKNESS_MAX_MM] = MorphologicalTissueProperties.DERMIS_THICKNESS_MEAN_MM + MorphologicalTissueProperties.DERMIS_THICKNESS_STD_MM
#         dermis_dict[Tags.STRUCTURE_CENTER_DEPTH_MM] = randomize(dermis_dict[Tags.STRUCTURE_CENTER_DEPTH_MIN_MM], dermis_dict[Tags.STRUCTURE_CENTER_DEPTH_MAX_MM])
#         dermis_dict[Tags.STRUCTURE_THICKNESS_MM] = randomize(dermis_dict[Tags.STRUCTURE_THICKNESS_MIN_MM], dermis_dict[Tags.STRUCTURE_THICKNESS_MAX_MM])     
#     else:
#         #todo think about that! 
#         dermis_dict[Tags.STRUCTURE_USE_DISTORTION] = settings_dict["distortion"][0]
#         dermis_dict[Tags.STRUCTURE_DISTORTION_FREQUENCY_PER_MM] =settings_dict["distortion_frequency"][0]
#         dermis_dict[Tags.STRUCTURE_CENTER_DEPTH_MM] = 0 
#         dermis_dict[Tags.STRUCTURE_THICKNESS_MM] = MorphologicalTissueProperties.DERMIS_THICKNESS_MEAN_MM
#     dermis_dict[Tags.STRUCTURE_TISSUE_PROPERTIES] = get_dermis_settings(background_oxy=background_oxy)
#     dermis_dict[Tags.STRUCTURE_DISTORTED_PARAM_LIST] = [Tags.KEY_OXY, Tags.KEY_WATER]
#     dermis_dict[Tags.STRUCTURE_SEGMENTATION_TYPE] = SegmentationClasses.DERMIS
#     dermis_dict[Tags.STRUCTURE_GEOMETRY_TYPE] = GeometryClasses.LAYER
#     dermis_dict[Tags.STRUCTURE_TYPE] = Tags.STRUCTURE_LAYER
#     return dermis_dict


def create_random_structures(settings_dict, geometries_dict):
    number_of_structures_per_sim = 5
    number_of_structures = 7
    structures_dict= dict()
     #   structures_dict["sphere" + str(i).zfill(2)] = create_random_sphere(settings_dict)
    for i in range(0, number_of_structures_per_sim):
        index = int(randomize(1, number_of_structures))
        structure_tag = geometries_dict[str(index)][0]
        structures_dict["structure_" +  str(i).zfill(2)] = eval("create_"+ structure_tag)(settings_dict)
    return structures_dict


def set_validation_settings(path):
    settings_df = pd.DataFrame.from_dict(pd.read_pickle(path + '/' + "boundary_settings_dict.pkl"))
    number_of_variables = settings_df.values.shape[1] -1
    #lst = list(itertools.product([0, 1], repeat=number_of_variables))
    #lst = random.shuffle(lst)
    lst = np.random.randint(2, size=number_of_variables)
    iteration_dict=dict()
    iteration_dict["randomness"] = [False, False]
    for i in range (1, number_of_variables):
        iteration_dict[settings_df.keys()[i]]= settings_df.iloc[lst[i], i],
    return iteration_dict

def create_validation_structures(settings_dict, geometries_dict):
    number_of_structures_per_sim = 10
    number_of_structures = 7
    structures_dict= dict()
    for i in range(0, number_of_structures_per_sim):
        index = int(randomize(1, number_of_structures))
        structure_tag = geometries_dict[str(index)][0]
        structures_dict["structure_" +  str(i).zfill(2)] = eval("create_"+ structure_tag)(settings_dict)
    return structures_dict


"""Configuration of all the simulation volume settings"""

volume_info_path = "/workplace/data/3D/20200401_optical_prop_rough",
settings_dict = create_settings_dict()
geometries_dict = create_geometries_dict()
pickle_out = open(volume_info_path[0] + '/' + "boundary_settings_dict.pkl","wb")
pickle.dump(settings_dict, pickle_out)
pickle_out.close()
pickle_out = open(volume_info_path[0] + '/' + "geometries_dict.pkl","wb")
pickle.dump(geometries_dict, pickle_out)
pickle_out.close()


'''Now, start the simualtion '''
seed_index = 0
while seed_index < 1200:
    random_seed = 100000 + seed_index
    seed_index += 1
    np.random.seed(random_seed)
    settings = {
        Tags.WAVELENGTHS: [700],
        Tags.RANDOM_SEED: random_seed,
        Tags.VOLUME_NAME: "RandomVolume_" + str(random_seed).zfill(6),
        #Tags.SIMULATION_PATH: "/home/melanie/networkdrives/E130-Projekte/Photoacoustics/RawData/20200323_optical_prop_rough",
        Tags.SIMULATION_PATH: volume_info_path[0] + "/train",
        Tags.RUN_OPTICAL_MODEL: True,
        Tags.OPTICAL_MODEL_NUMBER_PHOTONS: 1e7,
        Tags.OPTICAL_MODEL_BINARY_PATH: "/workplace/ippai/ippai/simulate/models/optical_models/mcx",
        Tags.OPTICAL_MODEL: Tags.MODEL_MCX,
        Tags.ILLUMINATION_TYPE: Tags.ILLUMINATION_TYPE_PENCIL,
        Tags.RUN_ACOUSTIC_MODEL: False,
        Tags.SIMULATION_EXTRACT_FIELD_OF_VIEW: False,
        Tags.SPACING_MM: 0.5,
        Tags.DIM_VOLUME_Z_MM: 17,
        Tags.DIM_VOLUME_X_MM: 17,
        Tags.DIM_VOLUME_Y_MM: 17,
        Tags.ILLUMINATION_POSITION: [18, 18, 1],
        Tags.ILLUMINATION_DIRECTION: [0, 0, 1],  # direction of msot acuity
        Tags.AIR_LAYER_HEIGHT_MM: 0.5,
        Tags.GELPAD_LAYER_HEIGHT_MM: 0,
        Tags.STRUCTURE_GAUSSIAN_FILTER: False,
        #Tags.STRUCTURE_GAUSSIAN_FILTER_SIGMA: 1,
        Tags.VOLUME_INFO_PATH: volume_info_path[0],
        Tags.STRUCTURES: create_random_structures(pd.read_pickle(volume_info_path[0] + '/' + "boundary_settings_dict.pkl"), pd.read_pickle(volume_info_path[0] + '/' + "geometries_dict.pkl")) 
        }
    if seed_index < 1000:
        'to get all the structure info and values, check out the volume_info_path'
        try:
        # Create target Directory
            os.mkdir(settings[Tags.SIMULATION_PATH] + '/'+ settings[Tags.VOLUME_NAME])
            print("Directory " , settings[Tags.SIMULATION_PATH] + '/'+ settings[Tags.VOLUME_NAME] ,  " created ") 
        except FileExistsError:
            print("Directory " , settings[Tags.SIMULATION_PATH] + '/'+ settings[Tags.VOLUME_NAME] ,  " already exists")
        'to get all the structure info and values, check out the volume_info_path'
        pickle_out = open(settings[Tags.SIMULATION_PATH] + '/' + settings[Tags.VOLUME_NAME] + "/" +  "structure_train_settings_dict.pkl","wb")
        pickle.dump(settings[Tags.STRUCTURES], pickle_out)
        pickle_out.close()
        
        print("Simulating training set", random_seed)
        [settings_path, optical_path, acoustic_path, reco_path] = simulate(settings)
        print("Simulating training set", random_seed, "[Done]")
    else: 
        val_settings_dict = set_validation_settings(volume_info_path[0])
        settings[Tags.RANDOM_SEED] = random_seed
        settings[Tags.VOLUME_NAME] = "ValVolume_" + str(random_seed).zfill(6)
        settings[Tags.SIMULATION_PATH]= volume_info_path[0] + "/val"
        settings[Tags.STRUCTURES] = create_validation_structures(val_settings_dict, pd.read_pickle(volume_info_path[0] + '/' + "geometries_dict.pkl")) 

        try:
        # Create target Directory
            os.mkdir(settings[Tags.SIMULATION_PATH] + '/'+ settings[Tags.VOLUME_NAME])
            print("Directory " , settings[Tags.SIMULATION_PATH] + '/'+ settings[Tags.VOLUME_NAME] ,  " created ") 
        except FileExistsError:
            print("Directory " , settings[Tags.SIMULATION_PATH] + '/'+ settings[Tags.VOLUME_NAME] ,  " already exists")
        'to get all the structure info and values, check out the volume_info_path'
        pickle_out = open(settings[Tags.SIMULATION_PATH] + '/'+ settings[Tags.VOLUME_NAME] + "/" + "structure_val_settings_dict.pkl","wb")
        pickle.dump(settings[Tags.STRUCTURES], pickle_out)
        pickle_out.close()
        
        print("Simulating ", random_seed)
        [settings_path, optical_path, acoustic_path, reco_path] = simulate(settings)
        print("Simulating ", random_seed, "[Done]")
        