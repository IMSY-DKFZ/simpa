import numpy as np
from ippai.simulate.tissue_properties import TissueProperties

def create_simulation_volume(settings):
    volume_path = None

    volume = create_empty_volume(settings)
    print(volume[0][0, 0, 0])
    print(volume[1][0, 0, 0])
    print(volume[2][0, 0, 0])


    return volume_path


def create_empty_volume(settings):
    wavelength = settings["wavelength"]
    voxel_spacing = settings["voxel_spacing_mm"]
    volume_y_dim = int(settings["volume_y_dim_mm"] / voxel_spacing)
    volume_x_dim = int(settings["volume_x_dim_mm"] / voxel_spacing)
    volume_z_dim = int(settings["volume_z_dim_mm"] / voxel_spacing)
    sizes = (volume_y_dim, volume_x_dim, volume_z_dim)
    background_tp = TissueProperties(settings, "background_properties")
    background_properties = background_tp.instantiate_for_wavelength(wavelength)
    absorption_volume = np.ones(sizes) * background_properties[0]
    scattering_volume = np.ones(sizes) * background_properties[1]
    anisotropy_volume = np.ones(sizes) * background_properties[2]
    return [absorption_volume, scattering_volume, anisotropy_volume]
