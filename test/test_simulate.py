# FIXME: Include actual proper tests here!

from ippai.simulate import Tags
from ippai.simulate.simulation import simulate
from ippai.simulate.tissue_properties import get_background_settings
from ippai.simulate.structures import create_forearm_structures

import matplotlib.pylab as plt
import numpy as np

settings = {
    Tags.WAVELENGTH: 800,
    Tags.SIMULATION_PATH: "/home/janek/simulation_test/",
    Tags.RUN_OPTICAL_MODEL: False,
    Tags.RUN_ACOUSTIC_MODEL: False,
    'background_properties': get_background_settings(),
    Tags.SPACING: 0.3,
    Tags.DIM_VOLUME_Z: 10,
    Tags.DIM_VOLUME_X: 10,
    Tags.DIM_VOLUME_Y: 10,
    Tags.STRUCTURES: create_forearm_structures()
}
print(settings)
paths = simulate(settings)

mua = np.load(paths[0])['mua']

plt.imshow(mua[10, :, :])
plt.show()
plt.close()
