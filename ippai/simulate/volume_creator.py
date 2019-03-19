import numpy as np
from ippai.simulate.tissue_properties import TissueProperties
from ippai.simulate import Tags
from ippai.simulate.utils import randomize


def create_simulation_volume(settings):

    volume_path = settings[Tags.SIMULATION_PATH] + "test_properties.npz"

    volumes = create_empty_volume(settings)
    volumes = add_structures(volumes, settings)
    volumes = append_air_layer(volumes, settings)

    np.savez(volume_path,
             mua=volumes[0],
             mus=volumes[1],
             g=volumes[2])

    return volume_path

def append_air_layer(volumes, settings):
    mua = 1e-6
    mus = 1e-6
    g = 1
    sizes = np.shape(volumes[0])
    air_layer_height = int(settings[Tags.AIR_LAYER_HEIGHT_MM] / settings[Tags.SPACING_MM])
    new_mua = np.ones((sizes[0], sizes[1], sizes[2] + air_layer_height)) * mua
    new_mua[:, :, air_layer_height:] = volumes[0]
    new_mus = np.ones((sizes[0], sizes[1], sizes[2] + air_layer_height)) * mus
    new_mus[:, :, air_layer_height:] = volumes[1]
    new_g = np.ones((sizes[0], sizes[1], sizes[2] + air_layer_height)) * g
    new_g[:, :, air_layer_height:] = volumes[2]
    return [new_mua, new_mus, new_g]

def create_empty_volume(settings):
    wavelength = settings[Tags.WAVELENGTH]
    voxel_spacing = settings[Tags.SPACING_MM]
    volume_y_dim = int(settings[Tags.DIM_VOLUME_Y_MM] / voxel_spacing)
    volume_x_dim = int(settings[Tags.DIM_VOLUME_X_MM] / voxel_spacing)
    volume_z_dim = int(settings[Tags.DIM_VOLUME_Z_MM] / voxel_spacing)
    sizes = (volume_y_dim, volume_x_dim, volume_z_dim)
    background_properties = TissueProperties(settings, "background_properties")
    background_properties = background_properties.get(wavelength)
    absorption_volume = np.ones(sizes) * background_properties[0]
    scattering_volume = np.ones(sizes) * background_properties[1]
    anisotropy_volume = np.ones(sizes) * background_properties[2]
    return [absorption_volume, scattering_volume, anisotropy_volume]


def add_structures(volumes, settings):
    for structure in settings[Tags.STRUCTURES]:
        volumes = add_structure(structure, volumes, settings)
    return volumes


def add_structure(structure, volumes, settings):
    structure_properties = TissueProperties(settings[Tags.STRUCTURES][structure], Tags.STRUCTURE_TISSUE_PROPERTIES)
    [mua, mus, g] = structure_properties.get(settings[Tags.WAVELENGTH])

    if settings[Tags.STRUCTURES][structure][Tags.STRUCTURE_TYPE] == Tags.STRUCTURE_LAYER:
        volumes = add_layer(volumes, settings, structure, mua, mus, g)

    if settings[Tags.STRUCTURES][structure][Tags.STRUCTURE_TYPE] == Tags.STRUCTURE_TUBE:
        volumes = add_tube(volumes, settings, structure, mua, mus, g)

    return volumes


def add_layer(volumes, settings, structure, mua, mus, g):
    print("Adding layer")
    depth_min = settings[Tags.STRUCTURES][structure][Tags.STRUCTURE_DEPTH_MIN_MM]
    depth_max = settings[Tags.STRUCTURES][structure][Tags.STRUCTURE_DEPTH_MAX_MM]
    thickness_min = settings[Tags.STRUCTURES][structure][Tags.STRUCTURE_THICKNESS_MIN_MM]
    thickness_max = settings[Tags.STRUCTURES][structure][Tags.STRUCTURE_THICKNESS_MAX_MM]

    depth_in_voxels = randomize(depth_min, depth_max) / settings[Tags.SPACING_MM]
    thickness_in_voxels = randomize(thickness_min, thickness_max) / settings[Tags.SPACING_MM]

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

    return volumes


def add_tube(volumes, settings, structure, mua, mus, g):
    print("adding tube")
    sizes = np.shape(volumes[0])

    radius_min = settings[Tags.STRUCTURES][structure][Tags.STRUCTURE_RADIUS_MIN_MM]
    radius_max = settings[Tags.STRUCTURES][structure][Tags.STRUCTURE_RADIUS_MAX_MM]
    radius_in_voxels = randomize(radius_min, radius_max) / settings[Tags.SPACING_MM]

    start_x_min = settings[Tags.STRUCTURES][structure][Tags.STRUCTURE_TUBE_START_X_MIN_MM]
    start_x_max = settings[Tags.STRUCTURES][structure][Tags.STRUCTURE_TUBE_START_X_MAX_MM]
    start_y_min = settings[Tags.STRUCTURES][structure][Tags.STRUCTURE_TUBE_START_Y_MIN_MM]
    start_y_max = settings[Tags.STRUCTURES][structure][Tags.STRUCTURE_TUBE_START_Y_MAX_MM]
    start_z_min = settings[Tags.STRUCTURES][structure][Tags.STRUCTURE_TUBE_START_Z_MIN_MM]
    start_z_max = settings[Tags.STRUCTURES][structure][Tags.STRUCTURE_TUBE_START_Z_MAX_MM]

    if start_x_min == -1:
        start_x_min = radius_in_voxels * settings[Tags.SPACING_MM]
    if start_x_max == -1:
        start_x_max = (sizes[1] - radius_in_voxels) * settings[Tags.SPACING_MM]
    if start_y_min == -1:
        start_y_min = radius_in_voxels * settings[Tags.SPACING_MM]
    if start_y_max == -1:
        start_y_max = (sizes[0] - radius_in_voxels) * settings[Tags.SPACING_MM]
    if start_z_min == -1:
        start_z_min = radius_in_voxels * settings[Tags.SPACING_MM]
    if start_z_max == -1:
        start_z_max = (sizes[2] - radius_in_voxels) * settings[Tags.SPACING_MM]

    if start_z_min < settings[Tags.STRUCTURES][structure][Tags.STRUCTURE_DEPTH_MIN_MM]:
        start_z_min = settings[Tags.STRUCTURES][structure][Tags.STRUCTURE_DEPTH_MIN_MM]

    if start_z_max > settings[Tags.STRUCTURES][structure][Tags.STRUCTURE_DEPTH_MAX_MM]:
        start_z_max = settings[Tags.STRUCTURES][structure][Tags.STRUCTURE_DEPTH_MAX_MM]

    start_in_mm = np.asarray([randomize(start_x_min, start_x_max), randomize(start_y_min, start_y_max),
                              randomize(start_z_min, start_z_max)])

    start_in_voxels = start_in_mm / settings[Tags.SPACING_MM]

    if settings[Tags.STRUCTURES][structure][Tags.STRUCTURE_FORCE_ORTHAGONAL_TO_PLANE]:
        end = np.copy(start_in_voxels)
        start_in_voxels[1] = 0
        end[1] = sizes[0]

    idx_z_start = int(start_in_voxels[2] - radius_in_voxels - 1)
    if idx_z_start < 0:
        idx_z_start = 0
    idx_z_end = int(start_in_voxels[2] + radius_in_voxels + 1)
    if idx_z_end > sizes[2]-1:
        idx_z_end = sizes[2]-1
    idx_x_start = int(start_in_voxels[0] - radius_in_voxels - 1)
    if idx_x_start < 0:
        idx_x_start = 0
    idx_x_end = int(start_in_voxels[0] + radius_in_voxels + 1)
    if idx_x_end > sizes[1]-1:
        idx_x_end = sizes[1]-1

    for z_idx in range(idx_z_start, idx_z_end):
        for y_idx in range(sizes[0]):
            for x_idx in range(idx_x_start, idx_x_end):
                if fnc_straight_tube(x_idx, y_idx, z_idx, radius_in_voxels, start_in_voxels, end) <= 0:
                    volumes = set_voxel(volumes, y_idx, x_idx, z_idx, mua, mus, g)
    return volumes


def fnc_straight_tube(x, y, z, r, X1, X2):
    return ((y-X1[1])*(z-X2[2])-(z-X1[2])*(y-X2[1]))**2 + \
           ((z-X1[2])*(x-X2[0])-(x-X1[0])*(z-X2[2]))**2 + \
           ((x-X1[0])*(y-X2[1])-(y-X1[1])*(x-X2[0]))**2 - \
           r**2 * ((X2[0]-X1[0])**2 + (X2[1]-X1[1])**2 + (X2[2]-X1[2])**2)


def merge_voxel(volumes, y_idx, x_idx, z_idx, mua, mus, g, fraction):
    volumes[0][y_idx, x_idx, z_idx] = volumes[0][y_idx, x_idx, z_idx] * (1-fraction) + mua * fraction
    volumes[1][y_idx, x_idx, z_idx] = volumes[1][y_idx, x_idx, z_idx] * (1-fraction) + mus * fraction
    volumes[2][y_idx, x_idx, z_idx] = volumes[2][y_idx, x_idx, z_idx] * (1-fraction) + g * fraction
    return volumes


def set_voxel(volumes, y_idx, x_idx, z_idx, mua, mus, g):
    volumes[0][y_idx, x_idx, z_idx] = mua
    volumes[1][y_idx, x_idx, z_idx] = mus
    volumes[2][y_idx, x_idx, z_idx] = g
    return volumes
