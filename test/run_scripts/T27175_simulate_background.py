
import sys

sys.path.append("/workplace/ippai/")


from ippai.utils import Tags
from ippai.simulate.constants import GeometryClasses
from ippai.simulate.constants import SegmentationClasses
from ippai.simulate.simulation import simulate
from ippai.simulate.structures import *
from ippai.utils.calculate import randomize

import pickle
import pandas as pd
import itertools
import random
import os
#lst = list(itertools.product([0, 1], repeat=52))
# 
# '''this script simulates backgorund only'''
def create_settings_dict():
    settings_dict=dict()
    "This dictionary includes all extrema that can be arranged in the learning process"
    
    settings_dict["randomness"] = True
    settings_dict["background_mua"] = [10e-10, 0.1]
    settings_dict["background_mus"] = [50., 150.]
    settings_dict["background_g"] = [0.8, 1.]
    settings_dict["background_options"] = [3., 3.]

    settings_dict["distortion"] = [0.,0.]
    settings_dict["distortion_frequency"] = [0., 2.] 

    settings_dict["tube_mua"] = [2., 10.]
    settings_dict["tube_mus"] = [50., 150.]
    settings_dict["tube_g"] = [0.8, 5.]
    settings_dict["tube_radius"] = [0.5, 5.]
    settings_dict["tube_center_depth"] = [0., 17.]
    settings_dict["tube_center_x"] = [0., 17.]
    
    return settings_dict
  
def create_geometries_dict(): 
    geometries_dict = dict()
    geometries_dict["randomness"] = [True, True]

    #todo no str
    geometries_dict["0"] = "air",
    geometries_dict["1"] = "background",
    geometries_dict["2"] = "layer" ,
    geometries_dict["3"] = "tube",
    geometries_dict["4"] = "sphere" ,
    geometries_dict["5"] = "cubical_tube",
    geometries_dict["6"] = "cube",
    geometries_dict["7"] = "pyramid",

    return geometries_dict


def create_random_structures(settings_dict, geometries_dict):
    number_of_structures_per_sim = 2
    number_of_structures = 6
    structures_dict= dict()
    structures_dict["structure_00"] = create_background(settings_dict)
     #   structures_dict["sphere" + str(i).zfill(2)] = create_random_sphere(settings_dict)
    for i in range(1, number_of_structures_per_sim):
          index = 3 # int(randomize(2, number_of_structures))
          structure_tag = geometries_dict[str(index)][0]
          structures_dict["structure_" +  str(i).zfill(2)] = eval("create_"+ structure_tag)(settings_dict)
    return structures_dict


def set_validation_settings(path):
    settings_df = pd.DataFrame.from_dict(pd.read_pickle(path + '/' + "boundary_settings_dict.pkl"))
    number_of_variables = settings_df.values.shape[1] -1
    lst = np.random.randint(2, size=number_of_variables)
    iteration_dict=dict()
    iteration_dict["randomness"] = False
    for i in range (1, number_of_variables+1):
        iteration_dict[settings_df.keys()[i]]= np.float(settings_df.iloc[lst[i-1], i]),
    return iteration_dict

def create_validation_structures(settings_dict, geometries_dict):
    number_of_structures_per_sim = 2
    number_of_structures = 6
    structures_dict= dict()
    structures_dict["structure_00"] = create_background(settings_dict)
    for i in range(1, number_of_structures_per_sim):
        index = 3 #int(randomize(2, number_of_structures))
        structure_tag = geometries_dict[str(index)][0]
        structures_dict["structure_" +  str(i).zfill(2)] = eval("create_"+ structure_tag)(settings_dict)
    return structures_dict


"""Configuration of all the simulation volume settings"""

volume_info_path = "/workplace/data/3D/20200403_optical_prop_simple",
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
while seed_index < 2000:
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
    if seed_index <= 1600:
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
        