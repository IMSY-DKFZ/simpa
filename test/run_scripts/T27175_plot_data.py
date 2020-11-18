#est["simulations"]["original_data"]["optical_forward_model_output"]["700"].keys()


import os
import numpy as np
import fnmatch
from shutil import copyfile
import sys
sys.path.append("/workplace/ippai")
from ippai.io_handling import load_hdf5
import matplotlib.pyplot as plt 


""""
configuration
"""
pattern= '*.hdf5'
keys_data='ippai_output'
print('start')

settings_dict = {
    "base_dir": '/workplace/data/3D/20200410_optical_prop_bg_1_tube',
    "data_dir": ('train', 'val',),
    "output_dir": ('preprocessed_data/train_test', 'preprocessed_data/val',),
    "fileparams_lambda": "700",
    "fileparams_data": ('mua', 'mus', 'g',),
    "fileparams_data_path": "simulation_properties", 
    "fileparams_target_path": "optical_forward_model_output", 
    "fileparams_target": ('initial_pressure',),
}
"""
1. preprocess the simulation data files
2. extract the info you want and save it as numpy files
"""
###################################
# 1
###################################
for i in range(0, len(settings_dict["data_dir"])):
    detailed_base_dir=[]
    for root, dirs, files in os.walk(os.path.join(settings_dict["base_dir"], settings_dict["data_dir"][i])):
        for dirname in sorted(dirs):
            for data_root, data_dir, data_files in os.walk(os.path.join(root, dirname)): 
                    for filename in sorted(fnmatch.filter(data_files, pattern)):
                        if keys_data is not None and filename[:-5] in keys_data:  # - because we want to learn 1 wavelength at the moment eg optical_forward_model_output_660
                            detailed_base_dir.append(os.path.join(data_root, filename))
        detailed_base_dir = sorted(detailed_base_dir)

    ###################################
    # 2
    ###################################

    for data_id in range(0, len(detailed_base_dir)):
        channels = len(settings_dict["fileparams_data"]) + len(settings_dict["fileparams_target"])
        #print(channels)
        numpy_array_dict = load_hdf5(detailed_base_dir[data_id])
        array_with_channels = np.empty([channels, int(numpy_array_dict["settings"]["volume_x_dim_mm"]/numpy_array_dict["settings"]["voxel_spacing_mm"])-2, int(numpy_array_dict["settings"]["volume_x_dim_mm"]/numpy_array_dict["settings"]["voxel_spacing_mm"])-2, int(numpy_array_dict["settings"]["volume_x_dim_mm"]/numpy_array_dict["settings"]["voxel_spacing_mm"])-2])
        for idc in range(0, channels):
            if idc < len(settings_dict["fileparams_data"]):
                array_to_save = numpy_array_dict["simulations"]["original_data"][settings_dict["fileparams_data_path"]][settings_dict["fileparams_lambda"]][settings_dict["fileparams_data"][idc]]
                #print(idc)
                # c, x, y, z 
                # the one deletes the pixel where the light source is located
                array_with_channels[idc, :, :, :] = np.log(array_to_save[1:33, 1:33, 2:34])
            else:
                array_to_save_tar = numpy_array_dict["simulations"]["original_data"][settings_dict["fileparams_target_path"]][settings_dict["fileparams_lambda"]][settings_dict["fileparams_target"][idc - len(settings_dict["fileparams_data"])]]
                #print(idc)
                # c, x, y, z 
                # the one deletes the pixel where the light source is located
                array_with_channels[idc, :, :, :] = np.log(array_to_save_tar[1:33, 1:33, 2:34])
        # #Be careful here
        if data_id ==0 or data_id ==10 or data_id==20:
            plt.imsave(os.path.join(settings_dict["base_dir"], settings_dict["data_dir"][i]+"example_mua_"+str(data_id) + "_slice15.png"), array_with_channels[0,:,15,:])  
            plt.imsave(os.path.join(settings_dict["base_dir"], settings_dict["data_dir"][i]+"example_mus_"+str(data_id) + "_slice15.png"), array_with_channels[1,:,15,:])
            plt.imsave(os.path.join(settings_dict["base_dir"], settings_dict["data_dir"][i] + "example_g_"+str(data_id) + "_slice15.png"), array_with_channels[2,:,15,:])
            plt.imsave(os.path.join(settings_dict["base_dir"], settings_dict["data_dir"][i] + "example_p0_"+str(data_id) + "_slice15.png"), array_with_channels[3,:,15,:])
    print("one round plotted [Done]")
