# The MIT License (MIT)
#
# Copyright (c) 2018 Computer Assisted Medical Interventions Group, DKFZ
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import numpy as np
import copy

from ippai.simulate.tissue_properties import TissueProperties
from ippai.utils import Tags, StandardProperties
from ippai.simulate.constants import SegmentationClasses, SaveFilePaths
from ippai.utils.calculate import *

from ippai.io_handling.io_hdf5 import save_hdf5


def create_simulation_volume(settings):
    """
    This method creates a in silico respresentation of a tissue as described in the settings file that is given.
    :param settings: a dictionary containing all relevant Tags for the simulation to be able to instantiate a tissue.
    :return: a path to a npz file containing characteristics of the simulated volume:
            absorption, scattering, anisotropy, oxygenation, and a segmentation mask. All of these are given as 3d
            numpy arrays.
    """

    distortion = None
    if Tags.STRUCTURE_DISTORTED_LAYERS in settings and settings[Tags.STRUCTURE_DISTORTED_LAYERS]:
        if Tags.STRUCTURE_DISTORTED_LAYERS_ELEVATION in settings:
            max_elevation = settings[Tags.STRUCTURE_DISTORTED_LAYERS_ELEVATION]
        else:
            max_elevation = 10
        distortion = create_spline_for_range(0, settings[Tags.DIM_VOLUME_X_MM] + settings[Tags.SPACING_MM],
                                             maximum_y_elevation_mm=max_elevation,
                                             spacing=settings[Tags.SPACING_MM])

    seed = settings[Tags.RANDOM_SEED] + 10
    volumes = create_volumes(settings, seed, distortion=distortion)

    volume_path = SaveFilePaths.SIMULATION_PROPERTIES\
        .format(Tags.ORIGINAL_DATA, str(settings[Tags.WAVELENGTH]))
    save_hdf5(volumes, settings[Tags.IPPAI_OUTPUT_PATH], file_dictionary_path=volume_path)

    if Tags.PERFORM_UPSAMPLING in settings:
        if settings[Tags.PERFORM_UPSAMPLING] is True:

            upsampled_settings = copy.deepcopy(settings)
            upsampled_settings[Tags.UPSAMPLING_RUN] = True
            upsampled_settings[Tags.SPACING_MM] = settings[Tags.SPACING_MM] / settings[Tags.UPSCALE_FACTOR]
            if Tags.ACOUSTIC_SIMULATION_3D not in settings or not settings[Tags.ACOUSTIC_SIMULATION_3D]:
                upsampled_settings[Tags.DIM_VOLUME_Y_MM] = upsampled_settings[Tags.SPACING_MM]
            else:
                upsampled_settings[Tags.DIM_VOLUME_Y_MM] = settings[Tags.UPSCALE_FACTOR] * int(round(settings[Tags.DIM_VOLUME_Y_MM] / settings[Tags.SPACING_MM])) * upsampled_settings[Tags.SPACING_MM]
            upsampled_settings[Tags.DIM_VOLUME_X_MM] = settings[Tags.UPSCALE_FACTOR] * int(round(settings[Tags.DIM_VOLUME_X_MM] / settings[Tags.SPACING_MM])) * upsampled_settings[Tags.SPACING_MM]
            upsampled_settings[Tags.DIM_VOLUME_Z_MM] = settings[Tags.UPSCALE_FACTOR] * int(round(settings[Tags.DIM_VOLUME_Z_MM] / settings[Tags.SPACING_MM])) * upsampled_settings[Tags.SPACING_MM]

            upsampled_volumes = create_volumes(upsampled_settings, seed, distortion=distortion)
            del upsampled_settings[Tags.UPSAMPLING_RUN]

            if Tags.ACOUSTIC_SIMULATION_3D not in settings or not settings[Tags.ACOUSTIC_SIMULATION_3D]:
                for i in upsampled_volumes.keys():
                    if upsampled_volumes[i] is not None:
                        upsampled_volumes[i] = np.squeeze(upsampled_volumes[i])

            upsampled_volume_path = SaveFilePaths.SIMULATION_PROPERTIES\
                .format(Tags.UPSAMPLED_DATA, settings[Tags.WAVELENGTH])
            save_hdf5(upsampled_volumes, settings[Tags.IPPAI_OUTPUT_PATH], file_dictionary_path=upsampled_volume_path)
    np.random.seed(seed + 14)
    return volume_path


def create_volumes(settings, seed, distortion=None):
    tmp_y_dim = settings[Tags.DIM_VOLUME_Y_MM]
    settings[Tags.DIM_VOLUME_Y_MM] = settings[Tags.SPACING_MM]
    np.random.seed(seed)
    volumes = create_empty_volume(settings)
    volumes = add_structures(volumes, settings, distortion=distortion)
    volumes = append_gel_pad(volumes, settings)
    volumes = append_air_layer(volumes, settings)
    if Tags.ILLUMINATION_TYPE in settings:
        if settings[Tags.ILLUMINATION_TYPE] == Tags.ILLUMINATION_TYPE_MSOT_ACUITY_ECHO:
            volumes = append_msot_probe(volumes, settings, distortion)

    volumes = create_gruneisen_map(volumes, settings)

    if Tags.RUN_ACOUSTIC_MODEL in settings:
        if settings[Tags.RUN_ACOUSTIC_MODEL] is True:
            volumes = create_acoustic_properties(volumes, settings)

        for i in volumes.keys():
            if i not in (Tags.PROPERTY_SENSOR_MASK, Tags.PROPERTY_DIRECTIVITY_ANGLE):
                y_slices = int(round(tmp_y_dim / settings[Tags.SPACING_MM]))
                if Tags.UPSAMPLING_RUN in settings and settings[Tags.UPSAMPLING_RUN]:
                    if Tags.ACOUSTIC_SIMULATION_3D not in settings or not settings[Tags.ACOUSTIC_SIMULATION_3D]:
                        y_slices = 1
                    else:
                        y_slices = int(round(int(round(tmp_y_dim / (settings[Tags.SPACING_MM] * settings[Tags.UPSCALE_FACTOR]))) * settings[Tags.UPSCALE_FACTOR]))
                volumes[i] = np.repeat(volumes[i], y_slices, axis=1)
                volumes[i] = np.flip(volumes[i], 1)
            elif volumes[i] is not None:
                tmp_vol = np.zeros(np.shape(volumes[Tags.PROPERTY_ABSORPTION_PER_CM]))

                if Tags.ACOUSTIC_SIMULATION_3D in settings and settings[Tags.ACOUSTIC_SIMULATION_3D]:
                    for sl in range(
                            int(tmp_y_dim / (2 * settings[Tags.SPACING_MM])) - int(13/2 / settings[Tags.SPACING_MM]),
                            int(tmp_y_dim / (2 * settings[Tags.SPACING_MM])) + int(13/2 / settings[Tags.SPACING_MM])+1):
                        tmp_vol[:, sl, :] = volumes[i][:, 0, :]
                else:
                    tmp_vol[:, int(tmp_y_dim / (2 * settings[Tags.SPACING_MM])), :] = volumes[i][:, 0, :]

                volumes[i] = tmp_vol
    settings[Tags.DIM_VOLUME_Y_MM] = tmp_y_dim

    return volumes


def create_gruneisen_map(volumes, settings):
    """
    Creates a map the gruneisenparameter based on the temperature given in Tags.MEDIUM_TEMPERATURE_CELCIUS.
    If no medium temperature is specified, then a standard body temperature of 36°C is assumed.

    :param volumes: The volumes to append the gruneisen parameter to
    :param settings: The settings to extract the temperature from

    :return: the volumes with an appended map of gruneisen parameters of size volumes[0].size
    """
    if Tags.MEDIUM_TEMPERATURE_CELCIUS in settings:
        temperature_celcius = settings[Tags.MEDIUM_TEMPERATURE_CELCIUS]
    else:
        temperature_celcius = StandardProperties.BODY_TEMPERATURE_CELCIUS

    gruneisen_map = np.ones(np.shape(volumes[Tags.PROPERTY_ABSORPTION_PER_CM])) * calculate_gruneisen_parameter_from_temperature(temperature_celcius)
    volumes[Tags.PROPERTY_GRUNEISEN_PARAMETER] = gruneisen_map

    return volumes


def create_acoustic_properties(volumes, settings):
    """
    Creates maps of density, speed of sound and acoustic attenuation based on the segmented mask in volumes.

    :param volumes: The volumes to append the acoustic parameters to
    :param settings: The settings to extract if the medium is homogeneous of if heterogeneous parameters should be set.
    :return: The volumes with appended maps of acoustic parameters of size volumes[0].size
    """

    sizes = volumes[Tags.PROPERTY_ABSORPTION_PER_CM].shape

    # Set speed of sound of the medium

    if Tags.MEDIUM_SOUND_SPEED_HOMOGENEOUS in settings:
        if settings[Tags.MEDIUM_SOUND_SPEED_HOMOGENEOUS] is True:
            if Tags.MEDIUM_SOUND_SPEED in settings:
                sound_speed = np.ones(np.asarray(sizes)) * settings[Tags.MEDIUM_SOUND_SPEED]
            else:
                sound_speed = np.ones(np.asarray(sizes)) * StandardProperties.SPEED_OF_SOUND_GENERIC
            volumes[Tags.PROPERTY_SPEED_OF_SOUND] = sound_speed
        else:
            sound_speed = np.ones(np.asarray(sizes))
            tissue_sound_speeds = [StandardProperties.SPEED_OF_SOUND_AIR,
                                   StandardProperties.SPEED_OF_SOUND_MUSCLE,
                                   StandardProperties.SPEED_OF_SOUND_BONE,
                                   StandardProperties.SPEED_OF_SOUND_BLOOD,
                                   StandardProperties.SPEED_OF_SOUND_SKIN,
                                   StandardProperties.SPEED_OF_SOUND_SKIN,
                                   StandardProperties.SPEED_OF_SOUND_FAT,
                                   StandardProperties.SPEED_OF_SOUND_GEL_PAD,
                                   StandardProperties.SPEED_OF_SOUND_WATER,
                                   StandardProperties.SPEED_OF_SOUND_GENERIC]
            for i in range(-1, 9):
                np.place(sound_speed, volumes[Tags.PROPERTY_SEGMENTATION] == i, tissue_sound_speeds[i])
            volumes[Tags.PROPERTY_SPEED_OF_SOUND] = sound_speed
    else:
        sound_speed = np.ones(np.asarray(sizes)) * StandardProperties.SPEED_OF_SOUND_GENERIC
        volumes[Tags.PROPERTY_SPEED_OF_SOUND] = sound_speed

    # Set density of the medium

    if Tags.MEDIUM_DENSITY_HOMOGENEOUS in settings:
        if settings[Tags.MEDIUM_DENSITY_HOMOGENEOUS] is True:
            if Tags.MEDIUM_DENSITY in settings:
                density = np.ones(np.asarray(sizes)) * settings[Tags.MEDIUM_DENSITY]
            else:
                density = np.ones(np.asarray(sizes)) * StandardProperties.DENSITY_GENERIC
            volumes[Tags.PROPERTY_DENSITY] = density
        else:
            density = np.ones(np.asarray(sizes))
            tissue_densities = [StandardProperties.DENSITY_AIR,
                                StandardProperties.DENSITY_MUSCLE,
                                StandardProperties.DENSITY_BONE,
                                StandardProperties.DENSITY_BLOOD,
                                StandardProperties.DENSITY_SKIN,
                                StandardProperties.DENSITY_SKIN,
                                StandardProperties.DENSITY_FAT,
                                StandardProperties.DENSITY_GEL_PAD,
                                StandardProperties.DENSITY_WATER,
                                StandardProperties.DENSITY_GENERIC]
            for i in range(-1, 9):
                np.place(density, volumes[Tags.PROPERTY_SEGMENTATION] == i, tissue_densities[i])
            volumes[Tags.PROPERTY_DENSITY] = density
    else:
        density = np.ones(np.asarray(sizes)) * StandardProperties.DENSITY_GENERIC
        volumes[Tags.PROPERTY_DENSITY] = density

    # Set attenuation coefficient of the medium

    if Tags.MEDIUM_ALPHA_COEFF_HOMOGENEOUS in settings:
        if settings[Tags.MEDIUM_ALPHA_COEFF_HOMOGENEOUS] is True:
            if Tags.MEDIUM_ALPHA_COEFF in settings:
                alpha_coeff = np.ones(np.asarray(sizes)) * settings[Tags.MEDIUM_ALPHA_COEFF]
            else:
                alpha_coeff = np.ones(np.asarray(sizes)) * StandardProperties.ALPHA_COEFF_GENERIC
            volumes[Tags.PROPERTY_ALPHA_COEFF] = alpha_coeff
        else:
            alpha_coeff = np.ones(np.asarray(sizes))
            tissue_densities = [StandardProperties.ALPHA_COEFF_AIR,
                                StandardProperties.ALPHA_COEFF_MUSCLE,
                                StandardProperties.ALPHA_COEFF_BONE,
                                StandardProperties.ALPHA_COEFF_BLOOD,
                                StandardProperties.ALPHA_COEFF_SKIN,
                                StandardProperties.ALPHA_COEFF_SKIN,
                                StandardProperties.ALPHA_COEFF_FAT,
                                StandardProperties.ALPHA_COEFF_GEL_PAD,
                                StandardProperties.ALPHA_COEFF_WATER,
                                StandardProperties.ALPHA_COEFF_GENERIC]
            for i in range(-1, 9):
                np.place(alpha_coeff, volumes[Tags.PROPERTY_SEGMENTATION] == i, tissue_densities[i])
            volumes[Tags.PROPERTY_ALPHA_COEFF] = alpha_coeff
    else:
        alpha_coeff = np.ones(np.asarray(sizes)) * StandardProperties.ALPHA_COEFF_GENERIC
        volumes[Tags.PROPERTY_ALPHA_COEFF] = alpha_coeff

    return volumes


def append_gel_pad(volumes, global_settings):
    if Tags.GELPAD_LAYER_HEIGHT_MM not in global_settings:
        print("[INFO] Tag", Tags.GELPAD_LAYER_HEIGHT_MM, "not found in settings. Ignoring gel pad.")
        return volumes

    mua = StandardProperties.GELPAD_MUA
    mus = StandardProperties.GELPAD_MUS
    g = StandardProperties.GELPAD_G
    sizes = np.shape(volumes[Tags.PROPERTY_ABSORPTION_PER_CM])
    gelpad_layer_height = int(global_settings[Tags.GELPAD_LAYER_HEIGHT_MM] / global_settings[Tags.SPACING_MM])

    new_mua = np.ones((sizes[0], sizes[1], sizes[2] + gelpad_layer_height)) * mua
    new_mua[:, :, gelpad_layer_height:] = volumes[Tags.PROPERTY_ABSORPTION_PER_CM]

    new_mus = np.ones((sizes[0], sizes[1], sizes[2] + gelpad_layer_height)) * mus
    new_mus[:, :, gelpad_layer_height:] = volumes[Tags.PROPERTY_SCATTERING_PER_CM]

    new_g = np.ones((sizes[0], sizes[1], sizes[2] + gelpad_layer_height)) * g
    new_g[:, :, gelpad_layer_height:] = volumes[Tags.PROPERTY_ANISOTROPY]

    new_oxy = np.ones((sizes[0], sizes[1], sizes[2] + gelpad_layer_height)) * (-1)
    new_oxy[:, :, gelpad_layer_height:] = volumes[Tags.PROPERTY_OXYGENATION]

    new_seg = np.ones((sizes[0], sizes[1], sizes[2] + gelpad_layer_height)) * SegmentationClasses.ULTRASOUND_GEL_PAD
    new_seg[:, :, gelpad_layer_height:] = volumes[Tags.PROPERTY_SEGMENTATION]

    return {Tags.PROPERTY_ABSORPTION_PER_CM: new_mua,
            Tags.PROPERTY_SCATTERING_PER_CM: new_mus,
            Tags.PROPERTY_ANISOTROPY: new_g,
            Tags.PROPERTY_OXYGENATION: new_oxy,
            Tags.PROPERTY_SEGMENTATION: new_seg}


def append_zero_layer(volumes, global_settings):
    """
    MCX allows to record diffuse reflectance at the surface of the 3D volume. For this it is required to append an extra
    layer of zeros at the top of the volume. This method appends a layer with one voxel thickness to the top of the
    volume. The zero layer is appended in the last dimension, if not 3D then exception is thrown.
    :param volumes: dict, contains the volumes to which the zero layer has to be appended, each volume has to be
    previously initialized. Expects 3D volumes.
    :param global_settings: dict, contains all the settings used for the simulations, if key defined by
    :code:`Tags.ZERO_LAYER_HEIGHT_VOXELS` not in dictionary, default value '1' is used.
    :return: dict, new volumes dictionary with zero layer appended to each volume
    """
    sizes = np.shape(volumes[Tags.PROPERTY_ABSORPTION_PER_CM])
    if Tags.ZERO_LAYER_HEIGHT_VOXELS not in global_settings:
        print("[INFO] Tag", Tags.ZERO_LAYER_HEIGHT_VOXELS, "not found in settings. Falling to default: 1 voxel.")
        zero_layer_height = 1
    else:
        zero_layer_height = global_settings[Tags.AIR_LAYER_HEIGHT_MM]

    new_mua = np.zeros((sizes[0], sizes[1], sizes[2] + zero_layer_height))
    new_mua[:, :, zero_layer_height:] = volumes[Tags.PROPERTY_ABSORPTION_PER_CM]

    new_mus = np.zeros((sizes[0], sizes[1], sizes[2] + zero_layer_height))
    new_mus[:, :, zero_layer_height:] = volumes[Tags.PROPERTY_SCATTERING_PER_CM]

    new_g = np.zeros((sizes[0], sizes[1], sizes[2] + zero_layer_height))
    new_g[:, :, zero_layer_height:] = volumes[Tags.PROPERTY_ANISOTROPY]

    new_oxy = np.ones((sizes[0], sizes[1], sizes[2] + zero_layer_height)) * (-1)
    new_oxy[:, :, zero_layer_height:] = volumes[Tags.PROPERTY_OXYGENATION]

    new_seg = np.zeros((sizes[0], sizes[1], sizes[2] + zero_layer_height))
    new_seg[:, :, zero_layer_height:] = volumes[Tags.PROPERTY_SEGMENTATION]

    return {Tags.PROPERTY_ABSORPTION_PER_CM: new_mua,
            Tags.PROPERTY_SCATTERING_PER_CM: new_mus,
            Tags.PROPERTY_ANISOTROPY: new_g,
            Tags.PROPERTY_OXYGENATION: new_oxy,
            Tags.PROPERTY_SEGMENTATION: new_seg}


def append_air_layer(volumes, global_settings):
    mua = StandardProperties.AIR_MUA
    mus = StandardProperties.AIR_MUS
    g = StandardProperties.AIR_G

    sizes = np.shape(volumes[Tags.PROPERTY_ABSORPTION_PER_CM])

    if Tags.AIR_LAYER_HEIGHT_MM not in global_settings:
        print("[INFO] Tag", Tags.AIR_LAYER_HEIGHT_MM, "not found in settings. Ignoring air layer.")
        return volumes

    air_layer_height = int(global_settings[Tags.AIR_LAYER_HEIGHT_MM] / global_settings[Tags.SPACING_MM])

    new_mua = np.ones((sizes[0], sizes[1], sizes[2] + air_layer_height)) * mua
    new_mua[:, :, air_layer_height:] = volumes[Tags.PROPERTY_ABSORPTION_PER_CM]

    new_mus = np.ones((sizes[0], sizes[1], sizes[2] + air_layer_height)) * mus
    new_mus[:, :, air_layer_height:] = volumes[Tags.PROPERTY_SCATTERING_PER_CM]

    new_g = np.ones((sizes[0], sizes[1], sizes[2] + air_layer_height)) * g
    new_g[:, :, air_layer_height:] = volumes[Tags.PROPERTY_ANISOTROPY]

    new_oxy = np.ones((sizes[0], sizes[1], sizes[2] + air_layer_height)) * (-1)
    new_oxy[:, :, air_layer_height:] = volumes[Tags.PROPERTY_OXYGENATION]

    new_seg = np.ones((sizes[0], sizes[1], sizes[2] + air_layer_height)) * SegmentationClasses.AIR
    new_seg[:, :, air_layer_height:] = volumes[Tags.PROPERTY_SEGMENTATION]

    return {Tags.PROPERTY_ABSORPTION_PER_CM: new_mua,
            Tags.PROPERTY_SCATTERING_PER_CM: new_mus,
            Tags.PROPERTY_ANISOTROPY: new_g,
            Tags.PROPERTY_OXYGENATION: new_oxy,
            Tags.PROPERTY_SEGMENTATION: new_seg}


def append_msot_probe(volumes, global_settings, distortion=None):
    mua_water = StandardProperties.AIR_MUA
    mus_water = StandardProperties.AIR_MUS
    g_water = StandardProperties.AIR_G

    mua_mediprene_layer = -np.log(0.85) / 10
    mus_mediprene_layer = -np.log(0.85) - -np.log(0.85) / 10
    g_mediprene_layer = 0.9

    sizes = np.shape(volumes[Tags.PROPERTY_ABSORPTION_PER_CM])

    if Tags.UPSAMPLING_RUN in global_settings and global_settings[Tags.UPSAMPLING_RUN]:
        orig_probe_size = int(round((1 + 42.2) / (global_settings[Tags.SPACING_MM] * global_settings[Tags.UPSCALE_FACTOR])))
        orig_z_dim = int(round(global_settings[Tags.DIM_VOLUME_Z_MM] / (global_settings[Tags.SPACING_MM] * global_settings[Tags.UPSCALE_FACTOR])))
        probe_size = int(round((orig_probe_size + orig_z_dim) * global_settings[Tags.UPSCALE_FACTOR])) - sizes[2]
    else:
        probe_size = int(round((1 + 42.2) / global_settings[Tags.SPACING_MM]))

    mediprene_layer_height = int(round(1 / global_settings[Tags.SPACING_MM]))
    water_layer_height = probe_size - mediprene_layer_height

    new_mua = np.ones((sizes[0], sizes[1], sizes[2] + mediprene_layer_height + water_layer_height)) * mua_water
    new_mua[:, :, water_layer_height + mediprene_layer_height:] = volumes[Tags.PROPERTY_ABSORPTION_PER_CM]
    volumes[Tags.PROPERTY_ABSORPTION_PER_CM] = new_mua

    new_mus = np.ones((sizes[0], sizes[1], sizes[2] + mediprene_layer_height + water_layer_height)) * mus_water
    new_mus[:, :, water_layer_height + mediprene_layer_height:] = volumes[Tags.PROPERTY_SCATTERING_PER_CM]
    volumes[Tags.PROPERTY_SCATTERING_PER_CM] = new_mus

    new_g = np.ones((sizes[0], sizes[1], sizes[2] + mediprene_layer_height + water_layer_height)) * g_water
    new_g[:, :, water_layer_height + mediprene_layer_height:] = volumes[Tags.PROPERTY_ANISOTROPY]
    volumes[Tags.PROPERTY_ANISOTROPY] = new_g

    new_oxy = np.ones((sizes[0], sizes[1], sizes[2] + mediprene_layer_height + water_layer_height)) * (-1)
    new_oxy[:, :, water_layer_height + mediprene_layer_height:] = volumes[Tags.PROPERTY_OXYGENATION]
    volumes[Tags.PROPERTY_OXYGENATION] = new_oxy

    new_seg = np.ones((sizes[0], sizes[1], sizes[2] + mediprene_layer_height + water_layer_height)) * \
              SegmentationClasses.GENERIC
    new_seg[:, :, water_layer_height + mediprene_layer_height:] = volumes[Tags.PROPERTY_SEGMENTATION]
    volumes[Tags.PROPERTY_SEGMENTATION] = new_seg

    if distortion is not None:
        mediprene_layer_settings = {
            Tags.STRUCTURE_CENTER_DEPTH_MAX_MM: 42.2,
            Tags.STRUCTURE_CENTER_DEPTH_MIN_MM: 42.2,
            Tags.STRUCTURE_SEGMENTATION_TYPE: SegmentationClasses.ULTRASOUND_GEL_PAD,
            Tags.STRUCTURE_THICKNESS_MIN_MM: 1,
            Tags.STRUCTURE_THICKNESS_MAX_MM: 1
        }
        volumes, _ = add_layer(volumes, global_settings, mediprene_layer_settings, mua=mua_mediprene_layer,
                               mus=mus_mediprene_layer, g=g_mediprene_layer, oxy=-1,
                               extent_parent_x_z_mm=None, distortion=distortion)

        z_range = range(int(np.round(42.2 / global_settings[Tags.SPACING_MM])),
                        int(np.round((42.2 + 1 - distortion[1]) / global_settings[Tags.SPACING_MM])))
        for y_idx in range(sizes[1]):
            for x_idx in range(sizes[0]):
                for z_idx in z_range:
                    if volumes[Tags.PROPERTY_SEGMENTATION][x_idx, y_idx, z_idx] == SegmentationClasses.ULTRASOUND_GEL_PAD:
                        break
                    else:
                        volumes[Tags.PROPERTY_ABSORPTION_PER_CM][x_idx, y_idx, z_idx] = mua_water
                        volumes[Tags.PROPERTY_SCATTERING_PER_CM][x_idx, y_idx, z_idx] = mus_water
                        volumes[Tags.PROPERTY_ANISOTROPY][x_idx, y_idx, z_idx] = g_water
                        volumes[Tags.PROPERTY_OXYGENATION][x_idx, y_idx, z_idx] = -1
                        volumes[Tags.PROPERTY_SEGMENTATION][x_idx, y_idx, z_idx] = SegmentationClasses.GENERIC

    if Tags.RUN_ACOUSTIC_MODEL in global_settings:
        if global_settings[Tags.RUN_ACOUSTIC_MODEL]:
            sizes = new_seg.shape

            detector_map = np.zeros((sizes[2], sizes[1], sizes[0]))
            detector_directivity = np.zeros(detector_map.shape)

            field_of_view_slice = int(sizes[1] / 2)

            pitch_angle = global_settings[Tags.SENSOR_ELEMENT_PITCH_MM] / global_settings[Tags.SENSOR_RADIUS_MM]
            detector_radius = global_settings[Tags.SENSOR_RADIUS_MM]/global_settings[Tags.SPACING_MM]

            focus = np.asarray([int(round(detector_radius + 11.2 / global_settings[Tags.SPACING_MM])),
                                int(round(detector_map.shape[2] / 2))])

            if distortion is not None:
                focus[0] -= np.round(distortion[1] / (2 * global_settings[Tags.SPACING_MM]))

            if Tags.SENSOR_LINEAR in global_settings and global_settings[Tags.SENSOR_LINEAR]:
                height = int(focus[0] - detector_radius/1.5)
                start = int(round(focus[1] - (int(global_settings[Tags.SENSOR_NUM_ELEMENTS] / 2) * global_settings[
                    Tags.SENSOR_ELEMENT_PITCH_MM] / global_settings[Tags.SPACING_MM])))
                end = int(round(focus[1] + (int(global_settings[Tags.SENSOR_NUM_ELEMENTS] / 2) * global_settings[
                    Tags.SENSOR_ELEMENT_PITCH_MM] / global_settings[Tags.SPACING_MM])))
                for i in range(start, end + 1):
                    detector_map[height, field_of_view_slice, i] = 1
            else:

                for i in range(-int(global_settings[Tags.SENSOR_NUM_ELEMENTS] / 2),
                               int(global_settings[Tags.SENSOR_NUM_ELEMENTS] / 2)):
                    angle = pitch_angle * i  # Convert Pitch to mm
                    y_det = focus[1] + np.sin(angle) * detector_radius
                    z_det = int(round(focus[0] - np.sqrt(detector_radius ** 2 -
                                                         (np.sin(angle) * detector_radius) ** 2)))
                    y_det = int(round(y_det))

                    detector_map[z_det, field_of_view_slice, y_det] = 1

                    if Tags.SENSOR_DIRECTIVITY_HOMOGENEOUS in global_settings:
                        if global_settings[Tags.SENSOR_DIRECTIVITY_HOMOGENEOUS] is True:
                            if Tags.SENSOR_DIRECTIVITY_ANGLE in global_settings:
                                detector_directivity[z_det, field_of_view_slice, y_det] = global_settings[Tags.SENSOR_DIRECTIVITY_ANGLE]
                            else:
                                detector_directivity = None
                        else:
                            detector_directivity[z_det, field_of_view_slice, y_det] = -angle
                    else:
                        detector_directivity = None

            volumes[Tags.PROPERTY_SENSOR_MASK] = np.rot90(detector_map, 1, axes=(0, 2))

            if detector_directivity is not None:
                volumes[Tags.PROPERTY_DIRECTIVITY_ANGLE] = np.rot90(detector_directivity, 1, axes=(0, 2))
            else:
                volumes[Tags.PROPERTY_DIRECTIVITY_ANGLE] = detector_directivity

    return volumes


def create_empty_volume(global_settings):
    voxel_spacing = global_settings[Tags.SPACING_MM]
    volume_x_dim = int(round(global_settings[Tags.DIM_VOLUME_X_MM] / voxel_spacing))
    volume_y_dim = int(round(global_settings[Tags.DIM_VOLUME_Y_MM] / voxel_spacing))
    volume_z_dim = int(round(global_settings[Tags.DIM_VOLUME_Z_MM] / voxel_spacing))
    sizes = (volume_x_dim, volume_y_dim, volume_z_dim)
    absorption_volume = np.zeros(sizes)
    scattering_volume = np.zeros(sizes)
    anisotropy_volume = np.zeros(sizes)
    oxygenation_volume = np.zeros(sizes)
    segmentation_volume = np.zeros(sizes)
    return {Tags.PROPERTY_ABSORPTION_PER_CM: absorption_volume,
            Tags.PROPERTY_SCATTERING_PER_CM: scattering_volume,
            Tags.PROPERTY_ANISOTROPY: anisotropy_volume,
            Tags.PROPERTY_OXYGENATION: oxygenation_volume,
            Tags.PROPERTY_SEGMENTATION: segmentation_volume}


def add_structures(volumes, global_settings, distortion):

    for structure in global_settings[Tags.STRUCTURES]:
        volumes = add_structure(volumes, global_settings[Tags.STRUCTURES][structure], global_settings,
                                distortion=distortion)
    return volumes


def add_structure(volumes, structure_settings, global_settings, extent_x_z_mm=None, distortion=None):
    # TODO check if this is actually how the call should be handeled
    structure_properties = TissueProperties(structure_settings[Tags.STRUCTURE_TISSUE_PROPERTIES])
    [mua, mus, g] = structure_properties.get(global_settings[Tags.WAVELENGTH])
    oxy = calculate_oxygenation(structure_properties)

    if structure_settings[Tags.STRUCTURE_TYPE] == Tags.STRUCTURE_BACKGROUND:
        volumes = set_background(volumes, structure_settings, mua, mus, g, oxy)
        return volumes

    if structure_settings[Tags.STRUCTURE_TYPE] == Tags.STRUCTURE_LAYER:
        volumes, extent_x_z_mm = add_layer(volumes, global_settings, structure_settings, mua, mus, g, oxy,
                                           extent_x_z_mm, distortion=distortion)

    if structure_settings[Tags.STRUCTURE_TYPE] == Tags.STRUCTURE_TUBE:
        volumes, extent_x_z_mm = add_tube(volumes, global_settings, structure_settings, mua, mus, g, oxy,
                                          extent_x_z_mm, distortion=distortion)

    if structure_settings[Tags.STRUCTURE_TYPE] == Tags.STRUCTURE_ELLIPSE:
        volumes, extent_x_z_mm = add_ellipse(volumes, global_settings, structure_settings, mua, mus, g, oxy,
                                             extent_x_z_mm, distortion=distortion)

    if Tags.CHILD_STRUCTURES in structure_settings:
        for child_structure in structure_settings[Tags.CHILD_STRUCTURES]:
            volumes = add_structure(volumes, structure_settings[Tags.CHILD_STRUCTURES][child_structure],
                                    global_settings, extent_x_z_mm)

    return volumes


def set_background(volumes, structure_settings, mua, mus, g, oxy):
    volumes[Tags.PROPERTY_ABSORPTION_PER_CM][:] = mua
    volumes[Tags.PROPERTY_SCATTERING_PER_CM][:] = mus
    volumes[Tags.PROPERTY_ANISOTROPY][:] = g
    volumes[Tags.PROPERTY_OXYGENATION][:] = oxy
    volumes[Tags.PROPERTY_SEGMENTATION][:] = structure_settings[Tags.STRUCTURE_SEGMENTATION_TYPE]
    return volumes


def add_layer(volumes, global_settings, structure_settings, mua, mus, g, oxy, extent_parent_x_z_mm, distortion=None):
    if extent_parent_x_z_mm is None:
        extent_parent_x_z_mm = [0, 0, 0, 0]

    depth_min = structure_settings[Tags.STRUCTURE_CENTER_DEPTH_MIN_MM] + extent_parent_x_z_mm[3]
    depth_max = structure_settings[Tags.STRUCTURE_CENTER_DEPTH_MAX_MM] + extent_parent_x_z_mm[3]
    thickness_min = structure_settings[Tags.STRUCTURE_THICKNESS_MIN_MM]
    thickness_max = structure_settings[Tags.STRUCTURE_THICKNESS_MAX_MM]

    depth_in_voxels = randomize(depth_min, depth_max) / global_settings[Tags.SPACING_MM]
    thickness_in_voxels = randomize(thickness_min, thickness_max) / global_settings[Tags.SPACING_MM]

    sizes = np.shape(volumes[Tags.PROPERTY_ABSORPTION_PER_CM])

    if Tags.STRUCTURE_DISTORTED_LAYERS in global_settings and global_settings[Tags.STRUCTURE_DISTORTED_LAYERS]:
        spline = distortion[0]
        spline_voxel = spline(
            np.arange(sizes[0] * global_settings[Tags.SPACING_MM], step=global_settings[Tags.SPACING_MM]))
        spline_voxel = np.round(spline_voxel / global_settings[Tags.SPACING_MM])
        max_el = np.round(distortion[1] / global_settings[Tags.SPACING_MM])
        depth_in_voxels -= max_el
        elevation_voxel = -copy.deepcopy(max_el)

        it = -1
        fraction = copy.deepcopy(thickness_in_voxels)
        z_range = range(int(depth_in_voxels - elevation_voxel), int(np.around(depth_in_voxels + thickness_in_voxels)))

        for z_idx in z_range:
            for y_idx in range(sizes[1]):
                for x_idx in range(sizes[0]):
                    if spline_evaluator2d_voxel(x_idx, z_idx, spline_voxel, depth_in_voxels, thickness_in_voxels):
                        volumes = set_voxel(volumes, x_idx, y_idx, z_idx, mua, mus, g, oxy,
                                            structure_settings[Tags.STRUCTURE_SEGMENTATION_TYPE])
            fraction -= 1
            it += 1

    else:
        it = -1
        fraction = thickness_in_voxels
        z_range = range(int(depth_in_voxels), int(depth_in_voxels + thickness_in_voxels))
        for z_idx in z_range:
            for y_idx in range(sizes[1]):
                for x_idx in range(sizes[0]):
                    volumes = set_voxel(volumes, x_idx, y_idx, z_idx, mua, mus, g, oxy,
                                        structure_settings[Tags.STRUCTURE_SEGMENTATION_TYPE])
            fraction -= 1
            it += 1

    if fraction > 1e-10:
        for y_idx in range(sizes[1]):
            for x_idx in range(sizes[0]):
                merge_voxel(volumes, x_idx, y_idx, it + 1, mua, mus, g, oxy,
                            structure_settings[Tags.STRUCTURE_SEGMENTATION_TYPE], fraction)
    # FIXME
    extent_parent_x_z_mm = [0, sizes[0] * global_settings[Tags.SPACING_MM],
                            depth_in_voxels * global_settings[Tags.SPACING_MM],
                            (depth_in_voxels + thickness_in_voxels) * global_settings[Tags.SPACING_MM]]

    return volumes, extent_parent_x_z_mm


def add_tube(volumes, global_settings, structure_settings, mua, mus, g, oxy, extent_parent_x_z_mm, distortion=None):
    if extent_parent_x_z_mm is None:
        extent_parent_x_z_mm = [0, 0, 0, 0]

    sizes = np.shape(volumes[Tags.PROPERTY_ABSORPTION_PER_CM])

    radius_min = structure_settings[Tags.STRUCTURE_RADIUS_MIN_MM]
    radius_max = structure_settings[Tags.STRUCTURE_RADIUS_MAX_MM]
    radius_in_mm = randomize(radius_min, radius_max)
    radius_in_voxels = radius_in_mm / global_settings[Tags.SPACING_MM]

    start_x_min = structure_settings[Tags.STRUCTURE_TUBE_CENTER_X_MIN_MM] + \
                  (extent_parent_x_z_mm[0] + extent_parent_x_z_mm[1]) / 2
    start_x_max = structure_settings[Tags.STRUCTURE_TUBE_CENTER_X_MAX_MM] + \
                  (extent_parent_x_z_mm[0] + extent_parent_x_z_mm[1]) / 2
    start_z_min = structure_settings[Tags.STRUCTURE_CENTER_DEPTH_MIN_MM] + \
                  (extent_parent_x_z_mm[2] + extent_parent_x_z_mm[3]) / 2
    start_z_max = structure_settings[Tags.STRUCTURE_CENTER_DEPTH_MAX_MM] + \
                  (extent_parent_x_z_mm[2] + extent_parent_x_z_mm[3]) / 2

    if start_x_min is None:
        start_x_min = radius_in_voxels * global_settings[Tags.SPACING_MM]
    if start_x_max is None:
        start_x_max = (sizes[0] - radius_in_voxels) * global_settings[Tags.SPACING_MM]
    if start_z_min is None:
        start_z_min = radius_in_voxels * global_settings[Tags.SPACING_MM]
    if start_z_max is None:
        start_z_max = (sizes[2] - radius_in_voxels) * global_settings[Tags.SPACING_MM]

    start_in_mm = np.asarray([randomize(start_x_min, start_x_max), 0,
                              randomize(start_z_min, start_z_max)])

    if distortion is not None:
        start_in_mm[2] -= (distortion[1] - distortion[0](start_in_mm[0]))

    start_in_voxels = start_in_mm / global_settings[Tags.SPACING_MM]

    end = np.copy(start_in_voxels)
    start_in_voxels[1] = 0
    end[1] = sizes[1]

    idx_z_start = int(start_in_voxels[2] - radius_in_voxels - 1)
    if idx_z_start < 0:
        idx_z_start = 0
    idx_z_end = int(start_in_voxels[2] + radius_in_voxels + 1)
    if idx_z_end > sizes[2]:
        idx_z_end = sizes[2]
    idx_x_start = int(start_in_voxels[0] - radius_in_voxels - 1)
    if idx_x_start < 0:
        idx_x_start = 0
    idx_x_end = int(start_in_voxels[0] + radius_in_voxels + 1)
    if idx_x_end > sizes[0]:
        idx_x_end = sizes[0]

    for z_idx in range(idx_z_start, idx_z_end):
        for y_idx in range(sizes[1]):
            for x_idx in range(idx_x_start, idx_x_end):
                if fnc_straight_tube(x_idx, y_idx, z_idx, radius_in_voxels, start_in_voxels, end) <= 0:
                    volumes = set_voxel(volumes, x_idx, y_idx, z_idx, mua, mus, g, oxy,
                                        structure_settings[Tags.STRUCTURE_SEGMENTATION_TYPE])

    extent_parent_x_z_mm = [start_in_mm[0] - radius_in_mm, start_in_mm[0] + radius_in_mm,
                            start_in_mm[2] - radius_in_mm, start_in_mm[2] + radius_in_mm]

    return volumes, extent_parent_x_z_mm


def add_ellipse(volumes, global_settings, structure_settings, mua, mus, g, oxy, extent_parent_x_z_mm, distortion=None):
    if extent_parent_x_z_mm is None:
        extent_parent_x_z_mm = [0, 0, 0, 0]

    sizes = np.shape(volumes[0])

    radius_min = structure_settings[Tags.STRUCTURE_RADIUS_MIN_MM]
    radius_max = structure_settings[Tags.STRUCTURE_RADIUS_MAX_MM]
    radius_in_mm = randomize(radius_min, radius_max)

    eccentricity_min = structure_settings[Tags.STRUCTURE_MIN_ECCENTRICITY]
    eccentricity_max = structure_settings[Tags.STRUCTURE_MAX_ECCENTRICITY]

    if eccentricity_max > radius_in_mm * 0.9:
        eccentricity_max = radius_in_mm * 0.9

    e = randomize(eccentricity_min, eccentricity_max)

    radius_z_mm = (radius_in_mm ** 2 + e ** 2) / (2 * radius_in_mm)
    radius_x_mm = radius_in_mm - radius_z_mm

    if np.random.random() < 0.5:
        radius_x_mm = (radius_in_mm ** 2 + e ** 2) / (2 * radius_in_mm)
        radius_z_mm = radius_in_mm - radius_x_mm

    radius_x_in_voxels = radius_x_mm / global_settings[Tags.SPACING_MM]
    radius_z_in_voxels = radius_z_mm / global_settings[Tags.SPACING_MM]

    start_x_min = structure_settings[Tags.STRUCTURE_TUBE_CENTER_X_MIN_MM] + \
                  (extent_parent_x_z_mm[0] + extent_parent_x_z_mm[1]) / 2
    start_x_max = structure_settings[Tags.STRUCTURE_TUBE_CENTER_X_MAX_MM] + \
                  (extent_parent_x_z_mm[0] + extent_parent_x_z_mm[1]) / 2
    start_z_min = structure_settings[Tags.STRUCTURE_CENTER_DEPTH_MIN_MM] + \
                  (extent_parent_x_z_mm[2] + extent_parent_x_z_mm[3]) / 2
    start_z_max = structure_settings[Tags.STRUCTURE_CENTER_DEPTH_MAX_MM] + \
                  (extent_parent_x_z_mm[2] + extent_parent_x_z_mm[3]) / 2

    if start_x_min is None:
        start_x_min = radius_x_in_voxels * global_settings[Tags.SPACING_MM]
    if start_x_max is None:
        start_x_max = (sizes[0] - radius_x_in_voxels) * global_settings[Tags.SPACING_MM]
    if start_z_min is None:
        start_z_min = radius_z_in_voxels * global_settings[Tags.SPACING_MM]
    if start_z_max is None:
        start_z_max = (sizes[2] - radius_z_in_voxels) * global_settings[Tags.SPACING_MM]

    start_in_mm = np.asarray([randomize(start_x_min, start_x_max), 0,
                              randomize(start_z_min, start_z_max)])

    if distortion is not None:
        start_in_mm[2] -= (distortion[1] - distortion[0](start_in_mm[0]))

    start_in_voxels = start_in_mm / global_settings[Tags.SPACING_MM]

    end = np.copy(start_in_voxels)
    start_in_voxels[1] = 0
    end[1] = sizes[1]

    idx_z_start = int(start_in_voxels[2] - radius_z_in_voxels - 1)
    if idx_z_start < 0:
        idx_z_start = 0
    idx_z_end = int(start_in_voxels[2] + radius_z_in_voxels + 1)
    if idx_z_end > sizes[2]:
        idx_z_end = sizes[2]
    idx_x_start = int(start_in_voxels[0] - radius_x_in_voxels - 1)
    if idx_x_start < 0:
        idx_x_start = 0
    idx_x_end = int(start_in_voxels[0] + radius_x_in_voxels + 1)
    if idx_x_end > sizes[0]:
        idx_x_end = sizes[0]

    for z_idx in range(idx_z_start, idx_z_end):
        for y_idx in range(sizes[1]):
            for x_idx in range(idx_x_start, idx_x_end):
                if fnc_straight_ellipse(x_idx, y_idx, z_idx, radius_x_in_voxels, radius_z_in_voxels,
                                        start_in_voxels, end) <= 0:
                    volumes = set_voxel(volumes, x_idx, y_idx, z_idx, mua, mus, g, oxy,
                                        structure_settings[Tags.STRUCTURE_SEGMENTATION_TYPE])

    extent_parent_x_z_mm = [start_in_mm[0] - radius_in_mm, start_in_mm[0] + radius_in_mm,
                            start_in_mm[2] - radius_in_mm, start_in_mm[2] + radius_in_mm]

    return volumes, extent_parent_x_z_mm


def fnc_straight_ellipse(x, y, z, r_x, r_z, X1, X2):
    """
    cartesian representation of a straight tube that goes from position X1 to position X2 with radius r.
    :param x:
    :param y:
    :param z:
    :param r:
    :param X1:
    :param X2:
    :return:
    """

    x_axis = ((y - X1[1]) * (z - X2[2]) - (z - X1[2]) * (y - X2[1])) ** 2 / r_z ** 2
    y_axis = ((z - X1[2]) * (x - X2[0]) - (x - X1[0]) * (z - X2[2])) ** 2
    z_axis = ((x - X1[0]) * (y - X2[1]) - (y - X1[1]) * (x - X2[0])) ** 2 / r_x ** 2
    radius = ((X2[0] - X1[0]) ** 2 + (X2[1] - X1[1]) ** 2 + (X2[2] - X1[2]) ** 2)
    return x_axis + y_axis + z_axis - radius


def fnc_straight_tube(x, y, z, r, X1, X2):
    """
    cartesian representation of a straight tube that goes from position X1 to position X2 with radius r.
    :param x:
    :param y:
    :param z:
    :param r:
    :param X1:
    :param X2:
    :return:
    """
    return ((y - X1[1]) * (z - X2[2]) - (z - X1[2]) * (y - X2[1])) ** 2 + \
           ((z - X1[2]) * (x - X2[0]) - (x - X1[0]) * (z - X2[2])) ** 2 + \
           ((x - X1[0]) * (y - X2[1]) - (y - X1[1]) * (x - X2[0])) ** 2 - \
           r ** 2 * ((X2[0] - X1[0]) ** 2 + (X2[1] - X1[1]) ** 2 + (X2[2] - X1[2]) ** 2)


def merge_voxel(volumes, x_idx, y_idx, z_idx, mua, mus, g, oxy, seg, fraction):
    """
    Updates a voxel position in the volumes by merging the given physical properties with the
    properties already stored in the volumes. The merging is done in a relative manner using the given fraction.

    :param volumes: list of numpy arrays with len(volumes) >= 3
    :param x_idx: integer
    :param y_idx: integer
    :param z_idx: integer
    :param mua: scalar, the optical absorption coefficient in 1/cm
    :param mus: scalar, the optical scattering coefficient in 1/cm
    :param g: scalar, the anisotropy
    :param oxy: scalar, the blood oxygenation in [0, 1]
    :param seg: integer, the tissue segmentation type from SegmentationClasses
    :param fraction: scalar in [0, 1]

    :return: the volumes with the changed properties
    """
    if not np.isscalar(mua):
        if len(mua) > 1:
            volumes[Tags.PROPERTY_ABSORPTION_PER_CM][x_idx, y_idx, z_idx] = volumes[Tags.PROPERTY_ABSORPTION_PER_CM][x_idx, y_idx, z_idx] * (1 - fraction) + \
                                              mua[x_idx, y_idx, z_idx] * fraction
        else:
            volumes[Tags.PROPERTY_ABSORPTION_PER_CM][x_idx, y_idx, z_idx] = volumes[Tags.PROPERTY_ABSORPTION_PER_CM][x_idx, y_idx, z_idx] * (1 - fraction) + mua * fraction
    else:
        volumes[Tags.PROPERTY_ABSORPTION_PER_CM][x_idx, y_idx, z_idx] = volumes[Tags.PROPERTY_ABSORPTION_PER_CM][x_idx, y_idx, z_idx] * (1 - fraction) + mua * fraction

    if not np.isscalar(mus):
        if len(mus) > 1:
            volumes[Tags.PROPERTY_SCATTERING_PER_CM][x_idx, y_idx, z_idx] = volumes[Tags.PROPERTY_SCATTERING_PER_CM][x_idx, y_idx, z_idx] * (1 - fraction) + \
                                              mus[x_idx, y_idx, z_idx] * fraction
        else:
            volumes[Tags.PROPERTY_SCATTERING_PER_CM][x_idx, y_idx, z_idx] = volumes[Tags.PROPERTY_SCATTERING_PER_CM][x_idx, y_idx, z_idx] * (1 - fraction) + mus * fraction
    else:
        volumes[Tags.PROPERTY_SCATTERING_PER_CM][x_idx, y_idx, z_idx] = volumes[Tags.PROPERTY_SCATTERING_PER_CM][x_idx, y_idx, z_idx] * (1 - fraction) + mus * fraction

    if not np.isscalar(g):
        if len(g) > 1:
            volumes[Tags.PROPERTY_ANISOTROPY][x_idx, y_idx, z_idx] = volumes[Tags.PROPERTY_ANISOTROPY][x_idx, y_idx, z_idx] * (1 - fraction) + \
                                              g[x_idx, y_idx, z_idx] * fraction
        else:
            volumes[Tags.PROPERTY_ANISOTROPY][x_idx, y_idx, z_idx] = volumes[Tags.PROPERTY_ANISOTROPY][x_idx, y_idx, z_idx] * (1 - fraction) + g * fraction
    else:
        volumes[Tags.PROPERTY_ANISOTROPY][x_idx, y_idx, z_idx] = volumes[Tags.PROPERTY_ANISOTROPY][x_idx, y_idx, z_idx] * (1 - fraction) + g * fraction

    if oxy is None:
        volumes[Tags.PROPERTY_OXYGENATION][x_idx, y_idx, z_idx] = None
    elif not np.isscalar(oxy):
        if len(oxy) > 1:
            volumes[Tags.PROPERTY_OXYGENATION][x_idx, y_idx, z_idx] = volumes[Tags.PROPERTY_OXYGENATION][x_idx, y_idx, z_idx] * (1 - fraction) + \
                                              oxy[x_idx, y_idx, z_idx] * fraction
        else:
            volumes[Tags.PROPERTY_OXYGENATION][x_idx, y_idx, z_idx] = volumes[Tags.PROPERTY_OXYGENATION][x_idx, y_idx, z_idx] * (1 - fraction) + oxy * fraction
    else:
        volumes[Tags.PROPERTY_OXYGENATION][x_idx, y_idx, z_idx] = volumes[Tags.PROPERTY_OXYGENATION][x_idx, y_idx, z_idx] * (1 - fraction) + oxy * fraction

    volumes[Tags.PROPERTY_SEGMENTATION][x_idx, y_idx, z_idx] = seg
    return volumes


def set_voxel(volumes, x_idx, y_idx, z_idx, mua, mus, g, oxy, seg):
    """
    Sets a voxel position to a specific value in the volume

    :param volumes: list of numpy arrays with len(volumes) >= 3
    :param y_idx: integer
    :param x_idx: integer
    :param z_idx: integer
    :param mua: scalar
    :param mus: scalar
    :param g: scalar

    :return: the volumes with the changed properties
    """
    if not np.isscalar(mua):
        if len(mua) > 1:
            volumes[Tags.PROPERTY_ABSORPTION_PER_CM][x_idx, y_idx, z_idx] = mua[x_idx, y_idx, z_idx]
        else:
            volumes[Tags.PROPERTY_ABSORPTION_PER_CM][x_idx, y_idx, z_idx] = mua
    else:
        volumes[Tags.PROPERTY_ABSORPTION_PER_CM][x_idx, y_idx, z_idx] = mua

    if not np.isscalar(mus):
        if len(mus) > 1:
            volumes[Tags.PROPERTY_SCATTERING_PER_CM][x_idx, y_idx, z_idx] = mus[x_idx, y_idx, z_idx]
        else:
            volumes[Tags.PROPERTY_SCATTERING_PER_CM][x_idx, y_idx, z_idx] = mus
    else:
        volumes[Tags.PROPERTY_SCATTERING_PER_CM][x_idx, y_idx, z_idx] = mus

    if not np.isscalar(g):
        if len(g) > 1:
            volumes[Tags.PROPERTY_ANISOTROPY][x_idx, y_idx, z_idx] = g[x_idx, y_idx, z_idx]
        else:
            volumes[Tags.PROPERTY_ANISOTROPY][x_idx, y_idx, z_idx] = g
    else:
        volumes[Tags.PROPERTY_ANISOTROPY][x_idx, y_idx, z_idx] = g

    if not np.isscalar(oxy):
        if oxy is not None and len(oxy) > 1:
            volumes[Tags.PROPERTY_OXYGENATION][x_idx, y_idx, z_idx] = oxy[x_idx, y_idx, z_idx]
        else:
            volumes[Tags.PROPERTY_OXYGENATION][x_idx, y_idx, z_idx] = oxy
    else:
        volumes[Tags.PROPERTY_OXYGENATION][x_idx, y_idx, z_idx] = oxy

    volumes[Tags.PROPERTY_SEGMENTATION][x_idx, y_idx, z_idx] = seg

    return volumes
