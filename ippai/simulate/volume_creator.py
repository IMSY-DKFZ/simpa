import numpy as np
from scipy.ndimage import gaussian_filter

from ippai.simulate.tissue_properties import TissueProperties
from ippai.simulate import Tags
from ippai.simulate.utils import randomize


def create_simulation_volume(settings):

    volume_path = settings[Tags.SIMULATION_PATH] + "test_properties.npz"

    volumes = create_empty_volume(settings)
    volumes = add_structures(volumes, settings)
    volumes = append_air_layer(volumes, settings)

    volumes[0] = np.flip(volumes[0], 1)
    volumes[1] = np.flip(volumes[1], 1)
    volumes[2] = np.flip(volumes[2], 1)

    np.savez(volume_path,
             mua=volumes[0],
             mus=volumes[1],
             g=volumes[2])

    return volume_path


def append_air_layer(volumes, global_settings):
    mua = 1e-6
    mus = 1e-6
    g = 1
    sizes = np.shape(volumes[0])
    air_layer_height = int(global_settings[Tags.AIR_LAYER_HEIGHT_MM] / global_settings[Tags.SPACING_MM])
    new_mua = np.ones((sizes[0], sizes[1], sizes[2] + air_layer_height)) * mua
    new_mua[:, :, air_layer_height:] = volumes[0]
    new_mus = np.ones((sizes[0], sizes[1], sizes[2] + air_layer_height)) * mus
    new_mus[:, :, air_layer_height:] = volumes[1]
    new_g = np.ones((sizes[0], sizes[1], sizes[2] + air_layer_height)) * g
    new_g[:, :, air_layer_height:] = volumes[2]
    return [new_mua, new_mus, new_g]


def create_empty_volume(global_settings):
    wavelength = global_settings[Tags.WAVELENGTH]
    voxel_spacing = global_settings[Tags.SPACING_MM]
    volume_y_dim = int(global_settings[Tags.DIM_VOLUME_Y_MM] / voxel_spacing)
    volume_x_dim = int(global_settings[Tags.DIM_VOLUME_X_MM] / voxel_spacing)
    volume_z_dim = int(global_settings[Tags.DIM_VOLUME_Z_MM] / voxel_spacing)
    sizes = (volume_y_dim, volume_x_dim, volume_z_dim)
    background_properties = TissueProperties(global_settings, "background_properties")
    background_properties = background_properties.get(wavelength)
    absorption_volume = np.ones(sizes) * background_properties[0]
    scattering_volume = np.ones(sizes) * background_properties[1]
    anisotropy_volume = np.ones(sizes) * background_properties[2]
    return [absorption_volume, scattering_volume, anisotropy_volume]


def add_structures(volumes, global_settings):
    for structure in global_settings[Tags.STRUCTURES]:
        volumes = add_structure(volumes, global_settings[Tags.STRUCTURES][structure], global_settings)
    return volumes


def add_structure(volumes, structure_settings, global_settings, extent_x_z_mm=None):

    if structure_settings[Tags.STRUCTURE_TYPE] == Tags.STRUCTURE_DISTORTION_MULTIPLICATIVE:
        volumes = apply_distortion_map(volumes, structure_settings, global_settings[Tags.SPACING_MM])
        return volumes

    structure_properties = TissueProperties(structure_settings, Tags.STRUCTURE_TISSUE_PROPERTIES)
    [mua, mus, g] = structure_properties.get(global_settings[Tags.WAVELENGTH])

    if structure_settings[Tags.STRUCTURE_TYPE] == Tags.STRUCTURE_LAYER:
        volumes, extent_x_z_mm = add_layer(volumes, global_settings, structure_settings, mua, mus, g, extent_x_z_mm)

    if structure_settings[Tags.STRUCTURE_TYPE] == Tags.STRUCTURE_TUBE:
        volumes, extent_x_z_mm = add_tube(volumes, global_settings, structure_settings, mua, mus, g, extent_x_z_mm)

    if Tags.CHILD_STRUCTURES in structure_settings:
        for child_structure in structure_settings[Tags.CHILD_STRUCTURES]:
            volumes = add_structure(volumes, structure_settings[Tags.CHILD_STRUCTURES][child_structure],
                                    global_settings, extent_x_z_mm)

    return volumes


def apply_distortion_map(volumes, distortion_dict, spacing):
    sizes = np.shape(volumes[0])
    dist_freq = distortion_dict[Tags.STRUCTURE_DISTORTION_WAVELENGTH_MM]

    if dist_freq is None:
        dist_freq = 1
    else:
        dist_freq = dist_freq / spacing

    dist_min = distortion_dict[Tags.STRUCTURE_DISTORTION_MIN]
    dist_max = distortion_dict[Tags.STRUCTURE_DISTORTION_MAX]

    dist_map = randomize(0.9, 1.1, distribution='normal', size=sizes)
    dist_map = gaussian_filter(dist_map, dist_freq)
    dist_map = ((dist_map - np.mean(dist_map)) / np.std(dist_map)) * \
               ((dist_max - dist_min)/2) + ((dist_max + dist_min)/2)
    print(np.mean(volumes[0]))
    volumes[0] = volumes[0] * dist_map
    print(np.mean(volumes[0]))
    return volumes


def add_layer(volumes, global_settings, structure_settings, mua, mus, g, extent_parent_x_z_mm):
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
                volumes = set_voxel(volumes, y_idx, x_idx, z_idx, mua, mus, g)
        fraction -= 1
        it += 1

    if fraction > 1e-10:
        print(fraction)
        for y_idx in range(sizes[0]):
            for x_idx in range(sizes[1]):
                merge_voxel(volumes, y_idx, x_idx, it+1, mua, mus, g, fraction)

    extent_parent_x_z_mm = [0, sizes[1] * global_settings[Tags.SPACING_MM],
                            depth_in_voxels * global_settings[Tags.SPACING_MM],
                            (depth_in_voxels + thickness_in_voxels) * global_settings[Tags.SPACING_MM]]

    return volumes, extent_parent_x_z_mm


def add_tube(volumes, global_settings, structure_settings, mua, mus, g, extent_parent_x_z_mm):
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
                    volumes = set_voxel(volumes, y_idx, x_idx, z_idx, mua, mus, g)

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


def merge_voxel(volumes, y_idx, x_idx, z_idx, mua, mus, g, fraction):
    """
    Updates a voxel position in the volumes by merging the given physical properties with the
    properties already stored in the volumes. The merging is done in a relative manner using the given fraction.

    :param volumes: list of numpy arrays with len(volumes) >= 3
    :param y_idx: integer
    :param x_idx: integer
    :param z_idx: integer
    :param mua: scalar
    :param mus: scalar
    :param g: scalar
    :param fraction: scalar in [0, 1]

    :return: the volumes with the changed properties
    """
    volumes[0][y_idx, x_idx, z_idx] = volumes[0][y_idx, x_idx, z_idx] * (1-fraction) + mua * fraction
    volumes[1][y_idx, x_idx, z_idx] = volumes[1][y_idx, x_idx, z_idx] * (1-fraction) + mus * fraction
    volumes[2][y_idx, x_idx, z_idx] = volumes[2][y_idx, x_idx, z_idx] * (1-fraction) + g * fraction
    return volumes


def set_voxel(volumes, y_idx, x_idx, z_idx, mua, mus, g):
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
    volumes[0][y_idx, x_idx, z_idx] = mua
    volumes[1][y_idx, x_idx, z_idx] = mus
    volumes[2][y_idx, x_idx, z_idx] = g
    return volumes
