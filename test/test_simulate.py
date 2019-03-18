from ippai.simulate import simulate
from ippai.simulate.tissue_properties import TissueProperties

tp = TissueProperties(None, None)

settings = {
    'run_optical_forward_model': False,
    'run_acoustic_forward_model': False,
    'background_properties': tp.get_background_settings(),
    'wavelength': 800,
    'voxel_spacing_mm': 0.3,
    'volume_x_dim_mm': 10,
    'volume_y_dim_mm': 10,
    'volume_z_dim_mm': 10
}
print(settings)
simulate(settings)
