# FIXME: Include actual proper tests here!

from ippai.simulate import Tags
from ippai.simulate.simulation import simulate
from ippai.simulate.tissue_properties import get_background_settings
from ippai.simulate.structures import create_forearm_structures
from ippai.simulate.utils import randomize

import matplotlib.pylab as plt
import numpy as np

settings = {
    Tags.WAVELENGTH: 800,
    Tags.SIMULATION_PATH: "/home/janek/simulation_test/",
    Tags.RUN_OPTICAL_MODEL: False,
    Tags.RUN_ACOUSTIC_MODEL: False,
    'background_properties': get_background_settings(),
    Tags.SPACING_MM: 0.3,
    Tags.DIM_VOLUME_Z_MM: 20,
    Tags.DIM_VOLUME_X_MM: 40,
    Tags.DIM_VOLUME_Y_MM: 30,
    Tags.AIR_LAYER_HEIGHT_MM: 12,
    Tags.STRUCTURES: create_forearm_structures()
}
print(settings)
paths = simulate(settings)

mua = np.load(paths[0])['mua']
extent = [0, settings[Tags.DIM_VOLUME_X_MM], settings[Tags.DIM_VOLUME_Z_MM]+settings[Tags.AIR_LAYER_HEIGHT_MM], 0]
plt.imshow(np.rot90(mua[10, :, :], -1), vmin=0, vmax=3, extent=extent)
plt.show()
plt.close()
