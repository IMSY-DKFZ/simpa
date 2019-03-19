import numpy as np
from ippai.simulate.tissue_properties import TissueProperties
from ippai.simulate import Tags
from ippai.simulate.utils import randomize


def create_simulation_volume(settings):

    volume_path = settings[Tags.SIMULATION_PATH] + "test_properties.npz"

    volumes = create_empty_volume(settings)
    volumes = add_structures(volumes, settings)

    np.savez(volume_path,
             mua=volumes[0],
             mus=volumes[1],
             g=volumes[2])

    return volume_path


def create_empty_volume(settings):
    wavelength = settings[Tags.WAVELENGTH]
    voxel_spacing = settings[Tags.SPACING]
    volume_y_dim = int(settings[Tags.DIM_VOLUME_Y] / voxel_spacing)
    volume_x_dim = int(settings[Tags.DIM_VOLUME_X] / voxel_spacing)
    volume_z_dim = int(settings[Tags.DIM_VOLUME_Z] / voxel_spacing)
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

    return volumes


def add_layer(volumes, settings, structure, mua, mus, g):
    print("Adding layer")
    depth_min = settings[Tags.STRUCTURES][structure][Tags.STRUCTURE_LAYER_DEPTH_MIN]
    depth_max = settings[Tags.STRUCTURES][structure][Tags.STRUCTURE_LAYER_DEPTH_MAX]
    thickness_min = settings[Tags.STRUCTURES][structure][Tags.STRUCTURE_LAYER_THICKNESS_MIN]
    thickness_max = settings[Tags.STRUCTURES][structure][Tags.STRUCTURE_LAYER_THICKNESS_MAX]
    depth_in_voxels = randomize(depth_min, depth_max) / settings[Tags.SPACING]
    thickness_in_voxels = randomize(thickness_min, thickness_max) / settings[Tags.SPACING]

    sizes = np.shape(volumes[0])

    it = -1
    fraction = thickness_in_voxels
    z_range = range(int(depth_in_voxels), int(depth_in_voxels+thickness_in_voxels))
    print(z_range)
    for z_idx in z_range:
        for y_idx in range(sizes[0]):
            for x_idx in range(sizes[0]):
                volumes = set_voxel(volumes, y_idx, x_idx, z_idx, mua, mus, g)
        fraction -= 1
        it += 1

    if fraction > 1e-10:
        for y_idx in range(sizes[0]):
            for x_idx in range(sizes[0]):
                merge_voxel(volumes, y_idx, x_idx, it+1, mua, mus, g, fraction)

    return volumes


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
