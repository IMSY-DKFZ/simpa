import numpy as np

from ippai.simulate.tissue_properties import TissueProperties
from ippai.simulate import Tags, SegmentationClasses, StandardProperties
from ippai.simulate.utils import randomize


def create_simulation_volume(settings):
    """
    This method creates a in silico respresentation of a tissue as described in the settings file that is given.
    :param settings: a dictionary containing all relevant Tags for the simulation to be able to instantiate a tissue.
    :return: a path to a npz file containing characteristics of the simulated volume:
            absorption, scattering, anisotropy, oxygenation, and a segmentation mask. All of these are given as 3d
            numpy arrays.
    """
    volume_path = settings[Tags.SIMULATION_PATH] + "/" + settings[Tags.VOLUME_NAME] + "/" \
                  + "properties_" + str(settings[Tags.WAVELENGTH]) + "nm.npz"
    volumes = create_empty_volume(settings)
    volumes = add_structures(volumes, settings)
    volumes = append_gel_pad(volumes, settings)
    volumes = append_air_layer(volumes, settings)

    for i in range(len(volumes)):
        volumes[i] = np.flip(volumes[i], 1)

    np.savez(volume_path,
             mua=volumes[0],
             mus=volumes[1],
             g=volumes[2],
             oxy=volumes[3],
             seg=volumes[4])

    return volume_path


def append_gel_pad(volumes, global_settings):

    if Tags.GELPAD_LAYER_HEIGHT_MM not in global_settings:
        print("[INFO] Tag", Tags.GELPAD_LAYER_HEIGHT_MM, "not found in settings. Ignoring gel pad.")
        return volumes

    mua = StandardProperties.AIR_MUA
    mus = StandardProperties.AIR_MUS
    g = StandardProperties.AIR_G
    sizes = np.shape(volumes[0])
    gelpad_layer_height = int(global_settings[Tags.GELPAD_LAYER_HEIGHT_MM] / global_settings[Tags.SPACING_MM])

    new_mua = np.ones((sizes[0], sizes[1], sizes[2] + gelpad_layer_height)) * mua
    new_mua[:, :, gelpad_layer_height:] = volumes[0]

    new_mus = np.ones((sizes[0], sizes[1], sizes[2] + gelpad_layer_height)) * mus
    new_mus[:, :, gelpad_layer_height:] = volumes[1]

    new_g = np.ones((sizes[0], sizes[1], sizes[2] + gelpad_layer_height)) * g
    new_g[:, :, gelpad_layer_height:] = volumes[2]

    new_oxy = np.ones((sizes[0], sizes[1], sizes[2] + gelpad_layer_height)) * (-1)
    new_oxy[:, :, gelpad_layer_height:] = volumes[3]

    new_seg = np.ones((sizes[0], sizes[1], sizes[2] + gelpad_layer_height)) * SegmentationClasses.ULTRASOUND_GEL_PAD
    new_seg[:, :, gelpad_layer_height:] = volumes[4]

    return [new_mua, new_mus, new_g, new_oxy, new_seg]


def append_air_layer(volumes, global_settings):
    mua = StandardProperties.AIR_MUA
    mus = StandardProperties.AIR_MUS
    g = StandardProperties.AIR_G
    sizes = np.shape(volumes[0])
    air_layer_height = int(global_settings[Tags.AIR_LAYER_HEIGHT_MM] / global_settings[Tags.SPACING_MM])

    new_mua = np.ones((sizes[0], sizes[1], sizes[2] + air_layer_height)) * mua
    new_mua[:, :, air_layer_height:] = volumes[0]

    new_mus = np.ones((sizes[0], sizes[1], sizes[2] + air_layer_height)) * mus
    new_mus[:, :, air_layer_height:] = volumes[1]

    new_g = np.ones((sizes[0], sizes[1], sizes[2] + air_layer_height)) * g
    new_g[:, :, air_layer_height:] = volumes[2]

    new_oxy = np.ones((sizes[0], sizes[1], sizes[2] + air_layer_height)) * (-1)
    new_oxy[:, :, air_layer_height:] = volumes[3]

    new_seg = np.ones((sizes[0], sizes[1], sizes[2] + air_layer_height)) * SegmentationClasses.AIR
    new_seg[:, :, air_layer_height:] = volumes[4]

    return [new_mua, new_mus, new_g, new_oxy, new_seg]


def create_empty_volume(global_settings):
    voxel_spacing = global_settings[Tags.SPACING_MM]
    volume_x_dim = int(global_settings[Tags.DIM_VOLUME_X_MM] / voxel_spacing)
    volume_y_dim = int(global_settings[Tags.DIM_VOLUME_Y_MM] / voxel_spacing)
    volume_z_dim = int(global_settings[Tags.DIM_VOLUME_Z_MM] / voxel_spacing)
    sizes = (volume_x_dim, volume_y_dim, volume_z_dim)
    absorption_volume = np.zeros(sizes)
    scattering_volume = np.zeros(sizes)
    anisotropy_volume = np.zeros(sizes)
    oxygenation_volume = np.zeros(sizes)
    segmentation_volume = np.zeros(sizes)
    return [absorption_volume, scattering_volume, anisotropy_volume, oxygenation_volume, segmentation_volume]


def add_structures(volumes, global_settings):
    for structure in global_settings[Tags.STRUCTURES]:
        volumes = add_structure(volumes, global_settings[Tags.STRUCTURES][structure], global_settings)
    return volumes


def add_structure(volumes, structure_settings, global_settings, extent_x_z_mm=None):

    structure_properties = TissueProperties(structure_settings, Tags.STRUCTURE_TISSUE_PROPERTIES,
                                            np.shape(volumes[0]), global_settings[Tags.SPACING_MM])
    [mua, mus, g, oxy] = structure_properties.get(global_settings[Tags.WAVELENGTH])

    if structure_settings[Tags.STRUCTURE_TYPE] == Tags.STRUCTURE_BACKGROUND:
        volumes = add_background(volumes, structure_settings, mua, mus, g, oxy)
        return volumes

    if structure_settings[Tags.STRUCTURE_TYPE] == Tags.STRUCTURE_LAYER:
        volumes, extent_x_z_mm = add_layer(volumes, global_settings, structure_settings, mua, mus, g, oxy, extent_x_z_mm)

    if structure_settings[Tags.STRUCTURE_TYPE] == Tags.STRUCTURE_TUBE:
        volumes, extent_x_z_mm = add_tube(volumes, global_settings, structure_settings, mua, mus, g, oxy, extent_x_z_mm)

    if Tags.CHILD_STRUCTURES in structure_settings:
        for child_structure in structure_settings[Tags.CHILD_STRUCTURES]:
            volumes = add_structure(volumes, structure_settings[Tags.CHILD_STRUCTURES][child_structure],
                                    global_settings, extent_x_z_mm)

    return volumes


def add_background(volumes, structure_settings, mua, mus, g, oxy):
    print("Test")
    volumes[0] += mua
    volumes[1] += mus
    volumes[2] += g
    volumes[3] += oxy
    volumes[4] += structure_settings[Tags.STRUCTURE_SEGMENTATION_TYPE]
    return volumes


def add_layer(volumes, global_settings, structure_settings, mua, mus, g, oxy, extent_parent_x_z_mm):
    print("Adding layer")
    if extent_parent_x_z_mm is None:
        extent_parent_x_z_mm = [0, 0, 0, 0]

    depth_min = structure_settings[Tags.STRUCTURE_DEPTH_MIN_MM] + extent_parent_x_z_mm[3]
    depth_max = structure_settings[Tags.STRUCTURE_DEPTH_MAX_MM] + extent_parent_x_z_mm[3]
    thickness_min = structure_settings[Tags.STRUCTURE_THICKNESS_MIN_MM]
    thickness_max = structure_settings[Tags.STRUCTURE_THICKNESS_MAX_MM]

    depth_in_voxels = randomize(depth_min, depth_max) / global_settings[Tags.SPACING_MM]
    thickness_in_voxels = randomize(thickness_min, thickness_max) / global_settings[Tags.SPACING_MM]

    sizes = np.shape(volumes[0])

    it = -1
    fraction = thickness_in_voxels
    z_range = range(int(depth_in_voxels), int(depth_in_voxels+thickness_in_voxels))
    print(z_range)
    for z_idx in z_range:
        for y_idx in range(sizes[0]):
            for x_idx in range(sizes[1]):
                volumes = set_voxel(volumes, x_idx, y_idx, z_idx, mua, mus, g, oxy,
                                    structure_settings[Tags.STRUCTURE_SEGMENTATION_TYPE])
        fraction -= 1
        it += 1

    if fraction > 1e-10:
        print(fraction)
        for y_idx in range(sizes[0]):
            for x_idx in range(sizes[1]):
                merge_voxel(volumes, x_idx, y_idx, it+1, mua, mus, g, oxy,
                            structure_settings[Tags.STRUCTURE_SEGMENTATION_TYPE], fraction)

    extent_parent_x_z_mm = [0, sizes[1] * global_settings[Tags.SPACING_MM],
                            depth_in_voxels * global_settings[Tags.SPACING_MM],
                            (depth_in_voxels + thickness_in_voxels) * global_settings[Tags.SPACING_MM]]

    return volumes, extent_parent_x_z_mm


def add_tube(volumes, global_settings, structure_settings, mua, mus, g, oxy, extent_parent_x_z_mm):
    print("adding tube")

    if extent_parent_x_z_mm is None:
        extent_parent_x_z_mm = [0, 0, 0, 0]

    sizes = np.shape(volumes[0])

    radius_min = structure_settings[Tags.STRUCTURE_RADIUS_MIN_MM]
    radius_max = structure_settings[Tags.STRUCTURE_RADIUS_MAX_MM]
    radius_in_mm = randomize(radius_min, radius_max)
    radius_in_voxels = radius_in_mm / global_settings[Tags.SPACING_MM]

    start_x_min = structure_settings[Tags.STRUCTURE_TUBE_START_X_MIN_MM] + \
                  (extent_parent_x_z_mm[0] + extent_parent_x_z_mm[1]) / 2
    start_x_max = structure_settings[Tags.STRUCTURE_TUBE_START_X_MAX_MM] + \
                  (extent_parent_x_z_mm[0] + extent_parent_x_z_mm[1]) / 2
    start_z_min = structure_settings[Tags.STRUCTURE_DEPTH_MIN_MM] + \
                  (extent_parent_x_z_mm[2] + extent_parent_x_z_mm[3]) / 2
    start_z_max = structure_settings[Tags.STRUCTURE_DEPTH_MAX_MM] + \
                  (extent_parent_x_z_mm[2] + extent_parent_x_z_mm[3]) / 2

    if start_x_min is None:
        start_x_min = radius_in_voxels * global_settings[Tags.SPACING_MM]
    if start_x_max is None:
        start_x_max = (sizes[1] - radius_in_voxels) * global_settings[Tags.SPACING_MM]
    if start_z_min is None:
        start_z_min = radius_in_voxels * global_settings[Tags.SPACING_MM]
    if start_z_max is None:
        start_z_max = (sizes[2] - radius_in_voxels) * global_settings[Tags.SPACING_MM]

    start_in_mm = np.asarray([randomize(start_x_min, start_x_max), 0,
                              randomize(start_z_min, start_z_max)])

    start_in_voxels = start_in_mm / global_settings[Tags.SPACING_MM]

    end = np.copy(start_in_voxels)
    start_in_voxels[1] = 0
    end[1] = sizes[0]

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
    if idx_x_end > sizes[1]:
        idx_x_end = sizes[1]

    for z_idx in range(idx_z_start, idx_z_end):
        for y_idx in range(sizes[0]):
            for x_idx in range(idx_x_start, idx_x_end):
                if fnc_straight_tube(x_idx, y_idx, z_idx, radius_in_voxels, start_in_voxels, end) <= 0:
                    volumes = set_voxel(volumes, x_idx, y_idx, z_idx, mua, mus, g, oxy,
                                        structure_settings[Tags.STRUCTURE_SEGMENTATION_TYPE])

    extent_parent_x_z_mm = [start_in_mm[0] - radius_in_mm, start_in_mm[0] + radius_in_mm,
                            start_in_mm[2] - radius_in_mm, start_in_mm[2] + radius_in_mm]

    return volumes, extent_parent_x_z_mm


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
    return ((y-X1[1])*(z-X2[2])-(z-X1[2])*(y-X2[1]))**2 + \
           ((z-X1[2])*(x-X2[0])-(x-X1[0])*(z-X2[2]))**2 + \
           ((x-X1[0])*(y-X2[1])-(y-X1[1])*(x-X2[0]))**2 - \
           r**2 * ((X2[0]-X1[0])**2 + (X2[1]-X1[1])**2 + (X2[2]-X1[2])**2)


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
            volumes[0][x_idx, y_idx, z_idx] = volumes[0][x_idx, y_idx, z_idx] * (1-fraction) + \
                                              mua[x_idx, y_idx, z_idx] * fraction
        else:
            volumes[0][x_idx, y_idx, z_idx] = volumes[0][x_idx, y_idx, z_idx] * (1 - fraction) + mua * fraction
    else:
        volumes[0][x_idx, y_idx, z_idx] = volumes[0][x_idx, y_idx, z_idx] * (1-fraction) + mua * fraction

    if not np.isscalar(mus):
        if len(mus) > 1:
            volumes[1][x_idx, y_idx, z_idx] = volumes[1][x_idx, y_idx, z_idx] * (1-fraction) + \
                                              mus[x_idx, y_idx, z_idx] * fraction
        else:
            volumes[1][x_idx, y_idx, z_idx] = volumes[1][x_idx, y_idx, z_idx] * (1 - fraction) + mus * fraction
    else:
        volumes[1][x_idx, y_idx, z_idx] = volumes[1][x_idx, y_idx, z_idx] * (1-fraction) + mus * fraction

    if not np.isscalar(g):
        if len(g) > 1:
            volumes[2][x_idx, y_idx, z_idx] = volumes[2][x_idx, y_idx, z_idx] * (1-fraction) + \
                                              g[x_idx, y_idx, z_idx] * fraction
        else:
            volumes[2][x_idx, y_idx, z_idx] = volumes[2][x_idx, y_idx, z_idx] * (1 - fraction) + g * fraction
    else:
        volumes[2][x_idx, y_idx, z_idx] = volumes[2][x_idx, y_idx, z_idx] * (1-fraction) + g * fraction

    if not np.isscalar(oxy):
        if len(oxy) > 1:
            volumes[3][x_idx, y_idx, z_idx] = volumes[3][x_idx, y_idx, z_idx] * (1-fraction) + \
                                              oxy[x_idx, y_idx, z_idx] * fraction
        else:
            volumes[3][x_idx, y_idx, z_idx] = volumes[3][x_idx, y_idx, z_idx] * (1 - fraction) + oxy * fraction
    else:
        volumes[3][x_idx, y_idx, z_idx] = volumes[3][x_idx, y_idx, z_idx] * (1-fraction) + oxy * fraction

    volumes[4][x_idx, y_idx, z_idx] = seg
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
            volumes[0][x_idx, y_idx, z_idx] = mua[x_idx, y_idx, z_idx]
        else:
            volumes[0][x_idx, y_idx, z_idx] = mua
    else:
        volumes[0][x_idx, y_idx, z_idx] = mua

    if not np.isscalar(mus):
        if len(mus) > 1:
            volumes[1][x_idx, y_idx, z_idx] = mus[x_idx, y_idx, z_idx]
        else:
            volumes[1][x_idx, y_idx, z_idx] = mus
    else:
        volumes[1][x_idx, y_idx, z_idx] = mus

    if not np.isscalar(g):
        if len(g) > 1:
            volumes[2][x_idx, y_idx, z_idx] = g[x_idx, y_idx, z_idx]
        else:
            volumes[2][x_idx, y_idx, z_idx] = g
    else:
        volumes[2][x_idx, y_idx, z_idx] = g

    if not np.isscalar(oxy):
        if len(oxy) > 1:
            volumes[3][x_idx, y_idx, z_idx] = oxy[x_idx, y_idx, z_idx]
        else:
            volumes[3][x_idx, y_idx, z_idx] = oxy
    else:
        volumes[3][x_idx, y_idx, z_idx] = oxy

    volumes[4][x_idx, y_idx, z_idx] = seg

    return volumes
