
import sys
import time 
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
from itertools import product
#lst = list(itertools.product([0, 1], repeat=52))


def create_settings_dict():
    settings_dict=dict()
    "This dictionary includes all extrema that can be arranged in the learning process"
    
    # settings_dict["randomness"] = True
    # settings_dict["background_mua"] = [10e-10, 0.1]
    # settings_dict["background_mus"] = [50., 150.]
    # settings_dict["background_g"] = [0.8, 1.]
    # settings_dict["background_options"] = [3., 3.]

    # settings_dict["tube_mua"] = [2., 10.]
    # settings_dict["tube_mus"] = [50., 150.]
    # settings_dict["tube_g"] = [0.8, 1.]
    # settings_dict["tube_radius"] = [0.5, 5.]

    settings_dict["randomness"] = True
    settings_dict["background_mua"] = [1.0, 1.0]
    settings_dict["background_mus"] = [100., 100.]
    settings_dict["background_g"] = [0.9, 0.9]
    settings_dict["background_options"] = [3., 3.]

    settings_dict["tube_mua_1"] = [4., 4.]
    settings_dict["tube_mua_2"] = [1., 1.]
    settings_dict["tube_mus_1"] = [100., 100.]
    settings_dict["tube_mus_2"] = [100., 100.]
    settings_dict["tube_g"] = [0.9, 0.9]
    settings_dict["tube_radius_1"] = [1., 1.]
    settings_dict["tube_radius_2"] = [2., 2.]

    settings_dict["layer_mua"] = [2., 10.]
    settings_dict["layer_mus"] = [50., 150.]
    settings_dict["layer_g"] = [0.8, 1.]
    settings_dict["layer_thickness"] = [0.2, 5.]

    settings_dict["sphere_mua"] = [2., 10.]
    settings_dict["sphere_mus"] = [50., 150.]
    settings_dict["sphere_g"] = [0.8, 1.]
    settings_dict["sphere_radius"] = [0.5, 5.]

    settings_dict["cube_mua"] = [2., 10.]
    settings_dict["cube_mus"] = [50., 150.]
    settings_dict["cube_g"] = [0.8, 1.]
    settings_dict["cube_length_x"] = [0.5, 3.]
    settings_dict["cube_length_y"] = [0.5, 3.]
    settings_dict["cube_length_z"] = [0.5, 3.]

    settings_dict["cubical_tube_mua"] = [2., 10.]
    settings_dict["cubical_tube_mus"] = [50., 150.]
    settings_dict["cubical_tube_g"] = [0.8, 1.]
    settings_dict["cubical_tube_radius"] = [0.5, 5.]
    settings_dict["cubical_tube_center_depth"] = [0., 17.]
    settings_dict["cubical_tube_center_x"] = [0.,17.]

    settings_dict["pyramid_mua"] = [2., 10.]
    settings_dict["pyramid_mus"] = [50., 150.]
    settings_dict["pyramid_g"] = [0.8, 1]
    settings_dict["pyramid_height"] = [0.5, 10.]
    settings_dict["pyramid_basis_extent"] = [0.5, 10.]
  #  settings_dict["pyramid_orientation_xy"] = [0, 90] 
  #  settings_dict["pyramid_orientation_xz"] = [0, 90]
  #  settings_dict["pyramid_orientation_yz"]= [0, 90] 
    settings_dict["pyramid_orientation"] = [0., 0.]

    settings_dict["distortion"] = [0.,0.]
    settings_dict["distortion_frequency"] = [0., 0.]

    return settings_dict


def create_rd_boundary_settings_dict():
    rd_boundary_settings_dict= dict()
    # rd_boundary_settings_dict["tube_center_depth"] = [0., 17.]
    # rd_boundary_settings_dict["tube_center_x"] = [0., 17.]

    rd_boundary_settings_dict["tube_center_depth_1"] = [2., 2.]
    rd_boundary_settings_dict["tube_center_x_1"] = [2., 2.]

    rd_boundary_settings_dict["tube_center_depth_2"] = [10., 10]
    rd_boundary_settings_dict["tube_center_x_2"] = [10., 10.]

    rd_boundary_settings_dict["layer_center_depth"] = [0., 17.]

    rd_boundary_settings_dict["sphere_center_depth"] = [0.,17.]
    rd_boundary_settings_dict["sphere_center_x"] = [0.,17.]
    rd_boundary_settings_dict["sphere_center_y"] = [0.,17.]

    rd_boundary_settings_dict["cubical_tube_center_depth"] = [0., 17.]
    rd_boundary_settings_dict["cubical_tube_center_x"] = [0.,17.]

    rd_boundary_settings_dict["cube_center_depth"] = [0., 17.]
    rd_boundary_settings_dict["cube_center_x"] = [0.,17.]
    rd_boundary_settings_dict["cube_center_y"] = [0., 17.]

    rd_boundary_settings_dict["pyramid_center_depth"] = [0.,17.]
    rd_boundary_settings_dict["pyramid_center_x"] = [0.,17.]
    rd_boundary_settings_dict["pyramid_center_y"] = [0.,17.]
    return rd_boundary_settings_dict
  
def create_geometries_dict(): 
    geometries_dict = dict()
    geometries_dict[0] = "air",
    geometries_dict[1] = "background",
    geometries_dict[2] = "layer" ,
    geometries_dict[3] = "tube",
    geometries_dict[4] = "sphere" ,
    geometries_dict[5] = "cubical_tube",
    geometries_dict[6] = "cube",
    geometries_dict[7] = "pyramid",
    return geometries_dict

def background_creation(settings_dict, rd_boundary_settings_dict):
    if settings_dict["randomness"] == True:
        background_dict = create_background(mua=randomize(settings_dict["background_mua"][0], settings_dict["background_mua"][1]), mus=randomize(settings_dict["background_mus"][0], settings_dict["background_mus"][1]), g =randomize(settings_dict["background_g"][0], settings_dict["background_g"][1]), distortion=settings_dict["distortion"][0], distortion_frequency=randomize(settings_dict["distortion_frequency"][0], settings_dict["distortion_frequency"][1]))
    else:
        background_dict = create_background(mua=settings_dict["background_mua"], mus=settings_dict["background_mus"], g=settings_dict["background_g"], distortion=settings_dict["distortion"][0], distortion_frequency=settings_dict["distortion_frequency"][0])
    return background_dict

def layer_creation(settings_dict, rd_boundary_settings_dict):
    if settings_dict["randomness"] == True:
        layer_dict = create_layer(mua=randomize(settings_dict["layer_mua"][0], settings_dict["layer_mua"][1]), mus=randomize(settings_dict["layer_mus"][0], settings_dict["layer_mus"][1]), g =randomize(settings_dict["layer_g"][0], settings_dict["layer_g"][1]), distortion=settings_dict["distortion"][0], distortion_frequency=randomize(settings_dict["distortion_frequency"][0], settings_dict["distortion_frequency"][1]),structure_center_depth_mm = randomize(rd_boundary_settings_dict["layer_center_depth"][0], rd_boundary_settings_dict["layer_center_depth"][1]), structure_thickness_mm= randomize(settings_dict["layer_thickness"][0], settings_dict["layer_thickness"][1]))
    else:
        layer_dict = create_layer(mua=settings_dict["layer_mua"], mus=settings_dict["layer_mus"], g=settings_dict["layer_g"], distortion=settings_dict["distortion"][0], distortion_frequency=settings_dict["distortion_frequency"][0], structure_center_depth_mm = randomize(rd_boundary_settings_dict["layer_center_depth"][0], rd_boundary_settings_dict["layer_center_depth"][1]), structure_thickness_mm = settings_dict["layer_thickness"][0])
    return layer_dict

def tube_creation(settings_dict, rd_boundary_settings_dict): 
    if settings_dict["randomness"] == True:
        tube_dict = create_tube(mua=randomize(settings_dict["tube_mua"][0], settings_dict["tube_mua"][1]), mus=randomize(settings_dict["tube_mus"][0], settings_dict["tube_mus"][1]), g =randomize(settings_dict["tube_g"][0], settings_dict["tube_g"][1]), distortion=settings_dict["distortion"][0], distortion_frequency=randomize(settings_dict["distortion_frequency"][0], settings_dict["distortion_frequency"][1]),structure_center_depth_mm = randomize(rd_boundary_settings_dict["tube_center_depth"][0], rd_boundary_settings_dict["tube_center_depth"][1]), structure_radius_mm = randomize(settings_dict["tube_radius"][0], settings_dict["tube_radius"][1]), structure_center_x_mm = randomize(rd_boundary_settings_dict["tube_center_x"][0], rd_boundary_settings_dict["tube_center_x"][1]), structure_center_z_mm = randomize(rd_boundary_settings_dict["tube_center_depth"][0], rd_boundary_settings_dict["tube_center_depth"][1]))
    else: 
        tube_dict = create_tube(mua=settings_dict["tube_mua"], mus=settings_dict["tube_mus"], g=settings_dict["tube_g"], distortion=settings_dict["distortion"][0], distortion_frequency = settings_dict["distortion_frequency"][0], structure_center_depth_mm =  randomize(rd_boundary_settings_dict["tube_center_depth"][0], rd_boundary_settings_dict["tube_center_depth"][1]), structure_radius_mm = settings_dict["tube_radius"][0], structure_center_x_mm = randomize(rd_boundary_settings_dict["tube_center_x"][0], rd_boundary_settings_dict["tube_center_x"][1]), structure_center_z_mm = randomize(rd_boundary_settings_dict["tube_center_depth"][0], rd_boundary_settings_dict["tube_center_depth"][1]))
    return tube_dict

def tube_1_creation(settings_dict, rd_boundary_settings_dict): 
    if settings_dict["randomness"] == True:
        tube_dict = create_tube(mua=randomize(settings_dict["tube_mua_1"][0], settings_dict["tube_mua_1"][1]), mus=randomize(settings_dict["tube_mus_1"][0], settings_dict["tube_mus_1"][1]), g =randomize(settings_dict["tube_g"][0], settings_dict["tube_g"][1]), distortion=settings_dict["distortion"][0], distortion_frequency=randomize(settings_dict["distortion_frequency"][0], settings_dict["distortion_frequency"][1]),structure_center_depth_mm = randomize(rd_boundary_settings_dict["tube_center_depth_1"][0], rd_boundary_settings_dict["tube_center_depth_1"][1]), structure_radius_mm = randomize(settings_dict["tube_radius_1"][0], settings_dict["tube_radius_1"][1]), structure_center_x_mm = randomize(rd_boundary_settings_dict["tube_center_x_1"][0], rd_boundary_settings_dict["tube_center_x_1"][1]), structure_center_z_mm = randomize(rd_boundary_settings_dict["tube_center_depth_1"][0], rd_boundary_settings_dict["tube_center_depth_1"][1]))
    else: 
        tube_dict = create_tube(mua=settings_dict["tube_mua"], mus=settings_dict["tube_mus"], g=settings_dict["tube_g"], distortion=settings_dict["distortion"][0], distortion_frequency = settings_dict["distortion_frequency"][0], structure_center_depth_mm =  randomize(rd_boundary_settings_dict["tube_center_depth"][0], rd_boundary_settings_dict["tube_center_depth"][1]), structure_radius_mm = settings_dict["tube_radius"][0], structure_center_x_mm = randomize(rd_boundary_settings_dict["tube_center_x"][0], rd_boundary_settings_dict["tube_center_x"][1]), structure_center_z_mm = randomize(rd_boundary_settings_dict["tube_center_depth"][0], rd_boundary_settings_dict["tube_center_depth"][1]))
    return tube_dict

def tube_2_creation(settings_dict, rd_boundary_settings_dict): 
    if settings_dict["randomness"] == True:
        tube_dict = create_tube(mua=randomize(settings_dict["tube_mua_2"][0], settings_dict["tube_mua_2"][1]), mus=randomize(settings_dict["tube_mus_2"][0], settings_dict["tube_mus_2"][1]), g =randomize(settings_dict["tube_g"][0], settings_dict["tube_g"][1]), distortion=settings_dict["distortion"][0], distortion_frequency=randomize(settings_dict["distortion_frequency"][0], settings_dict["distortion_frequency"][1]),structure_center_depth_mm = randomize(rd_boundary_settings_dict["tube_center_depth_2"][0], rd_boundary_settings_dict["tube_center_depth_2"][1]), structure_radius_mm = randomize(settings_dict["tube_radius_2"][0], settings_dict["tube_radius_2"][1]), structure_center_x_mm = randomize(rd_boundary_settings_dict["tube_center_x_2"][0], rd_boundary_settings_dict["tube_center_x_2"][1]), structure_center_z_mm = randomize(rd_boundary_settings_dict["tube_center_depth_2"][0], rd_boundary_settings_dict["tube_center_depth_2"][1]))
    else: 
        tube_dict = create_tube(mua=settings_dict["tube_mua"], mus=settings_dict["tube_mus"], g=settings_dict["tube_g"], distortion=settings_dict["distortion"][0], distortion_frequency = settings_dict["distortion_frequency"][0], structure_center_depth_mm =  randomize(rd_boundary_settings_dict["tube_center_depth"][0], rd_boundary_settings_dict["tube_center_depth"][1]), structure_radius_mm = settings_dict["tube_radius"][0], structure_center_x_mm = randomize(rd_boundary_settings_dict["tube_center_x"][0], rd_boundary_settings_dict["tube_center_x"][1]), structure_center_z_mm = randomize(rd_boundary_settings_dict["tube_center_depth"][0], rd_boundary_settings_dict["tube_center_depth"][1]))
    return tube_dict

def sphere_creation(settings_dict, rd_boundary_settings_dict):
    if settings_dict["randomness"] == True:
        sphere_dict = create_sphere(mua=randomize(settings_dict["sphere_mua"][0], settings_dict["sphere_mua"][1]), mus=randomize(settings_dict["sphere_mus"][0], settings_dict["sphere_mus"][1]), g =randomize(settings_dict["sphere_g"][0], settings_dict["sphere_g"][1]), distortion=settings_dict["distortion"][0], distortion_frequency=randomize(settings_dict["distortion_frequency"][0], settings_dict["distortion_frequency"][1]),structure_center_depth_mm = randomize(rd_boundary_settings_dict["sphere_center_depth"][0], rd_boundary_settings_dict["sphere_center_depth"][1]), structure_radius_mm =randomize(settings_dict["sphere_radius"][0], settings_dict["sphere_radius"][1]), structure_center_x_mm=randomize(rd_boundary_settings_dict["sphere_center_x"][0], rd_boundary_settings_dict["sphere_center_x"][1]), structure_center_y_mm=randomize(rd_boundary_settings_dict["sphere_center_y"][0], rd_boundary_settings_dict["sphere_center_y"][1]))
    else:
        sphere_dict = create_sphere(mua=settings_dict["sphere_mua"], mus=settings_dict["sphere_mus"], g=settings_dict["sphere_g"], distortion=settings_dict["distortion"][0], distortion_frequency=settings_dict["distortion_frequency"][0], structure_center_depth_mm=randomize(rd_boundary_settings_dict["sphere_center_depth"][0], rd_boundary_settings_dict["sphere_center_depth"][1]), structure_radius_mm=settings_dict["sphere_radius"][0], structure_center_x_mm=randomize(rd_boundary_settings_dict["sphere_center_x"][0], rd_boundary_settings_dict["sphere_center_x"][1]), structure_center_y_mm=randomize(rd_boundary_settings_dict["sphere_center_y"][0], rd_boundary_settings_dict["sphere_center_y"][1]))
    return sphere_dict

def cubical_tube_creation(settings_dict, rd_boundary_settings_dict):
    if settings_dict["randomness"] == True:
        cubical_tube_dict = create_cubical_tube(mua=randomize(settings_dict["cubical_tube_mua"][0], settings_dict["cubical_tube_mua"][1]), mus=randomize(settings_dict["cubical_tube_mus"][0], settings_dict["cubical_tube_mus"][1]), g =randomize(settings_dict["cubical_tube_g"][0], settings_dict["cubical_tube_g"][1]), distortion=settings_dict["distortion"][0], distortion_frequency=randomize(settings_dict["distortion_frequency"][0], settings_dict["distortion_frequency"][1]), structure_radius_mm=randomize(settings_dict["cubical_tube_radius"][0], settings_dict["cubical_tube_radius"][1]), structure_center_x_mm=randomize(rd_boundary_settings_dict["cubical_tube_center_x"][0], rd_boundary_settings_dict["cubical_tube_center_x"][1]), structure_center_z_mm=randomize(rd_boundary_settings_dict["cubical_tube_center_depth"][0], rd_boundary_settings_dict["cubical_tube_center_depth"][1]))
    else:
        cubical_tube_dict = create_cubical_tube(mua=settings_dict["cubical_tube_mua"], mus=settings_dict["cubical_tube_mus"], g =settings_dict["cubical_tube_g"], distortion=settings_dict["distortion"][0], distortion_frequency=settings_dict["distortion_frequency"][0], structure_radius_mm=settings_dict["cubical_tube_radius"][0], structure_center_x_mm=randomize(rd_boundary_settings_dict["cubical_tube_center_x"][0], rd_boundary_settings_dict["cubical_tube_center_x"][1]), structure_center_z_mm=randomize(rd_boundary_settings_dict["cubical_tube_center_depth"][0], rd_boundary_settings_dict["cubical_tube_center_depth"][1]))
    return cubical_tube_dict

def cube_creation(settings_dict, rd_boundary_settings_dict): 
    if settings_dict["randomness"] == True:
        cube_dict = create_cube(mua=randomize(settings_dict["cube_mua"][0], settings_dict["cube_mua"][1]), mus=randomize(settings_dict["cube_mus"][0], settings_dict["cube_mus"][1]), g =randomize(settings_dict["cube_g"][0], settings_dict["cube_g"][1]), distortion=settings_dict["distortion"][0], distortion_frequency=randomize(settings_dict["distortion_frequency"][0], settings_dict["distortion_frequency"][1]), structure_center_x_mm=randomize(rd_boundary_settings_dict["cube_center_x"][0], rd_boundary_settings_dict["cube_center_x"][1]), structure_center_y_mm=randomize(rd_boundary_settings_dict["cube_center_y"][0], rd_boundary_settings_dict["cube_center_y"][1]), structure_center_z_mm=randomize(rd_boundary_settings_dict["cube_center_depth"][0], rd_boundary_settings_dict["cube_center_depth"][1]), structure_length_x_mm=randomize(settings_dict["cube_length_x"][0], settings_dict["cube_length_x"][1]), structure_length_y_mm=randomize(settings_dict["cube_length_y"][0], settings_dict["cube_length_y"][1]), structure_length_z_mm=randomize(settings_dict["cube_length_z"][0], settings_dict["cube_length_z"][1]))
    else:
        cube_dict = create_cube(mua=settings_dict["cube_mua"], mus=settings_dict["cube_mus"], g=settings_dict["cube_g"], distortion=settings_dict["distortion"][0], distortion_frequency=settings_dict["distortion_frequency"][0], structure_center_x_mm=randomize(rd_boundary_settings_dict["cube_center_x"][0], rd_boundary_settings_dict["cube_center_x"][1]), structure_center_y_mm=randomize(rd_boundary_settings_dict["cube_center_y"][0], rd_boundary_settings_dict["cube_center_y"][1]), structure_center_z_mm=randomize(rd_boundary_settings_dict["cube_center_depth"][0], rd_boundary_settings_dict["cube_center_depth"][1]), structure_length_x_mm=settings_dict["cube_length_x"][0], structure_length_y_mm=settings_dict["cube_length_y"][0], structure_length_z_mm=settings_dict["cube_length_z"][0])
    return cube_dict

def pyramid_creation(settings_dict, rd_boundary_settings_dict):
    if settings_dict["randomness"] == True:
        pyramid_dict = create_pyramid(mua=randomize(settings_dict["pyramid_mua"][0], settings_dict["pyramid_mua"][1]), mus=randomize(settings_dict["pyramid_mus"][0], settings_dict["pyramid_mus"][1]), g =randomize(settings_dict["pyramid_g"][0], settings_dict["pyramid_g"][1]), distortion=settings_dict["distortion"][0], distortion_frequency=randomize(settings_dict["distortion_frequency"][0], settings_dict["distortion_frequency"][1]), structure_center_depth_mm =randomize(rd_boundary_settings_dict["pyramid_center_depth"][0], rd_boundary_settings_dict["pyramid_center_depth"][1]), structure_center_x_mm=randomize(rd_boundary_settings_dict["pyramid_center_x"][0], rd_boundary_settings_dict["pyramid_center_x"][1]), structure_center_y_mm=randomize(rd_boundary_settings_dict["pyramid_center_y"][0], rd_boundary_settings_dict["pyramid_center_y"][1]), strucutre_height_mm=randomize(settings_dict["pyramid_height"][0], settings_dict["pyramid_height"][1]), structure_basis_extent=randomize(settings_dict["pyramid_basis_extent"][0], settings_dict["pyramid_basis_extent"][1]), structure_pyramid_orientation=randomize(settings_dict["pyramid_orientation"][0], settings_dict["pyramid_orientation"][1]))
    else:
        pyramid_dict = create_pyramid(mua=settings_dict["pyramid_mua"], mus=settings_dict["pyramid_mus"], g=settings_dict["pyramid_g"], distortion=settings_dict["distortion"][0], distortion_frequency=settings_dict["distortion_frequency"][0], structure_center_depth_mm =randomize(rd_boundary_settings_dict["pyramid_center_depth"][0], rd_boundary_settings_dict["pyramid_center_depth"][1]), structure_center_x_mm=randomize(rd_boundary_settings_dict["pyramid_center_x"][0], rd_boundary_settings_dict["pyramid_center_x"][1]), structure_center_y_mm=randomize(rd_boundary_settings_dict["pyramid_center_y"][0], rd_boundary_settings_dict["pyramid_center_y"][1]), strucutre_height_mm=settings_dict["pyramid_height"][0], structure_basis_extent=settings_dict["pyramid_basis_extent"][0], structure_pyramid_orientation=settings_dict["pyramid_orientation"][0])
    return pyramid_dict




def create_random_structures(settings_dict, rd_boundary_settings_dict, geometries_dict):
    number_of_structures_per_sim = 1
    number_of_structures = 6
    structures_dict= dict()
    structures_dict["structure_00"] = background_creation(settings_dict, rd_boundary_settings_dict)
    for i in range(1, number_of_structures_per_sim):
          index = 3 #int(randomize(2, 2+ number_of_structures))
          structure_tag = geometries_dict[index][0]
          structures_dict["structure_" +  str(i).zfill(2)] = eval(structure_tag + "_"+ str(i)+"_creation")(settings_dict, rd_boundary_settings_dict)
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

def create_validation_structures(settings_dict, rd_boundary_settings_dict, geometries_dict):
    number_of_structures_per_sim = 1
    number_of_structures = 6
    structures_dict= dict()
    structures_dict["structure_00"] = background_creation(settings_dict, rd_boundary_settings_dict)
    # for i in range(1, number_of_structures_per_sim):
    #     index = 3#int(randomize(2, number_of_structures+2))
    #     structure_tag = geometries_dict[index][0]
    #     structures_dict["structure_" +  str(i).zfill(2)] = eval(structure_tag + "_creation")(settings_dict, rd_boundary_settings_dict)
    return structures_dict


"""Configuration of all the simulation volume settings"""

volume_info_path = "/workplace/data/2D/20201019_optical_prop_tube_RL",
settings_dict = create_settings_dict()
rd_boundary_settings_dict = create_rd_boundary_settings_dict()

geometries_dict = create_geometries_dict()
pickle_out = open(volume_info_path[0] + '/' + "boundary_settings_dict.pkl","wb")
pickle.dump(settings_dict, pickle_out)
pickle_out.close()
pickle_out = open(volume_info_path[0] + '/' + "geometries_dict.pkl","wb")
pickle.dump(geometries_dict, pickle_out)
pickle_out.close()
pickle_out = open(volume_info_path[0] + '/' + "random_boundary_settings_dict.pkl","wb")
pickle.dump(rd_boundary_settings_dict, pickle_out)
pickle_out.close()


'''Now, start the simualtion '''
seed_index = 0
start_time = time.time()
while seed_index < 2:
    random_seed = 100000 + seed_index #random_seed_index_start = random_seed_index_start(last simulation)+ number_simulations(last simulation)
    seed_index += 1
    # r = [1,2,3,4,5,6,7,8]
    r = [0,1]
    perm =list(product(r,repeat=5))
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
        Tags.APPLY_NOISE_MODEL: False,
        Tags.PERFORM_IMAGE_RECONSTRUCTION: False,
        Tags.SIMULATION_EXTRACT_FIELD_OF_VIEW: True,
        Tags.SPACING_MM: 0.5,
        Tags.DIM_VOLUME_Z_MM: 5,
        Tags.DIM_VOLUME_X_MM: 5,
        Tags.DIM_VOLUME_Y_MM: 5,
        Tags.ILLUMINATION_POSITION: [6, 6, 1],
        Tags.ILLUMINATION_DIRECTION: [0, 0, 1],  # direction of msot acuity
        Tags.LASER_PULSE_ENERGY_IN_MILLIJOULE: 50,
        Tags.AIR_LAYER_HEIGHT_MM: 0.5,
        Tags.GELPAD_LAYER_HEIGHT_MM: 0,
        Tags.STRUCTURE_GAUSSIAN_FILTER: False,
        #Tags.STRUCTURE_GAUSSIAN_FILTER_SIGMA: 1,
        Tags.VOLUME_INFO_PATH: volume_info_path[0],
        Tags.STRUCTURES: create_random_structures(pd.read_pickle(volume_info_path[0] + '/' + "boundary_settings_dict.pkl"), pd.read_pickle(volume_info_path[0] + '/' + "random_boundary_settings_dict.pkl"), pd.read_pickle(volume_info_path[0] + '/' + "geometries_dict.pkl")) 
        }
    if seed_index <= 1:
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

        # np.random.seed(seed_index)
        # np.random.seed(0)
        # mua = np.ones((1,1,8,8))#(np.ceil(np.random.rand(1,1,8,8)*8)).astype(np.float32)

        # offset = np.ceil(seed_index/32)
        # mua[0,0,:,0] = np.float32(2.0)
        # mua[0,0,1,2:3] = np.float32(4.0)
        # mua[0,0,3,2:3] = np.float32(4.0)
        # mua[0,0,2,1:5] = np.float32(4.0)

        # index = np.mod(seed_index-1, 32)
        
        # print(perm[index])
        # print(offset)
        # mua[0,0,4,5] = np.float32(perm[index][0]) + offset
        # mua[0,0,5,4] = np.float32(perm[index][1]) + offset
        # mua[0,0,5,5] = np.float32(perm[index][2]) + offset
        # mua[0,0,5,6] = np.float32(perm[index][3]) + offset
        # mua[0,0,6,5] = np.float32(perm[index][4]) + offset

        [settings_path, optical_path, acoustic_path, reco_path] = simulate(settings, mua=None)
        print("Simulating training set", random_seed, "[Done]")
    else: 
        #val_settings_dict = set_validation_settings(volume_info_path[0])
        
        '''here we create a simulation volume with specified mua:
        remarks: 
        - boundary sttings are as always in boundary_settings_dict,
        - the validation settings are in boundary_settings_dict_target'''
        pickle_out = open(volume_info_path[0] + '/' + "boundary_settings_dict.pkl","rb")
        new_dict = pickle.load(pickle_out)
        pickle_out.close()     
        pickle_out  = open(volume_info_path[0] + '/' + "boundary_settings_dict_target.pkl", "wb")
        new_dict["tube_mua_1"] = [7., 7.]
        new_dict["background_mua"] = [1., 1.]
        pickle.dump(new_dict, pickle_out)
        pickle_out.close()    
        settings[Tags.STRUCTURES] = create_random_structures(pd.read_pickle(volume_info_path[0] + '/' + "boundary_settings_dict_target.pkl"), pd.read_pickle(volume_info_path[0] + '/' + "random_boundary_settings_dict.pkl"), pd.read_pickle(volume_info_path[0] + '/' + "geometries_dict.pkl")) 

        settings[Tags.RANDOM_SEED] = random_seed
        settings[Tags.VOLUME_NAME] = "ValVolume_" + str(random_seed).zfill(6)
        settings[Tags.SIMULATION_PATH]= volume_info_path[0] + "/val"
        #settings[Tags.STRUCTURES] = create_validation_structures(val_settings_dict, pd.read_pickle(volume_info_path[0] + '/' + "random_boundary_settings_dict.pkl"), pd.read_pickle(volume_info_path[0] + '/' + "geometries_dict.pkl")) 


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
        
        
        # np.random.seed(seed_index-4000)
        # np.random.seed(0)
        # mua = (np.ceil(np.random.rand(1,1,8,8)*8)).astype(np.float32)
        # mua[0,0,4,4] = (np.ceil(np.random.rand()*8)).astype(np.float32)
      
        [settings_path, optical_path, acoustic_path, reco_path] = simulate(settings, mua=None)
        print("Simulating ", random_seed, "[Done]")
end_time = time.time()
print('Simulation took {:.2f} s'.format(end_time - start_time))
        

'''
remove tube_1_creation and tube_2_creation
add tube_creation
remove eval ( ... + _+ i+ )
remove settings dict mua_1 etc. 
'''
