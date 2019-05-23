from ippai.simulate import Tags
from ippai.simulate.models.optical_model import run_optical_forward_model
import matplotlib.pylab as plt
import numpy as np

random_seed=4217

settings = {
    Tags.WAVELENGTHS: [800], #np.arange(700, 951, 10),
    Tags.WAVELENGTH: 800,
    Tags.RANDOM_SEED: random_seed,
    Tags.VOLUME_NAME: "Slab",
    Tags.SIMULATION_PATH: "/home/janek/simulation_test/",
    Tags.RUN_OPTICAL_MODEL: True,
    Tags.OPTICAL_MODEL_NUMBER_PHOTONS: 10000000,
    Tags.OPTICAL_MODEL_BINARY_PATH: "/home/janek/mitk-superbuild/MITK-build/bin/MitkMCxyz",
    #Tags.OPTICAL_MODEL_PROBE_XML_FILE: "/home/janek/CAMI_PAT_SETUP_V2.xml",
    Tags.RUN_ACOUSTIC_MODEL: False,
    Tags.SPACING_MM: 0.5,
    Tags.DIM_VOLUME_Z_MM: 20,
    Tags.DIM_VOLUME_X_MM: 40,
    Tags.DIM_VOLUME_Y_MM: 30,
    Tags.AIR_LAYER_HEIGHT_MM: 12,
}

volume = np.zeros((49, 49, 100, 3))

volume[:, :, :, 0] = 1e-10
volume[:, :, :, 1] = 1e-10
volume[:, :, :, 2] = 0.9
volume[:, :, 30:70, 1] = 10

volume_path = settings[Tags.SIMULATION_PATH]+settings[Tags.VOLUME_NAME]+"/"+settings[Tags.VOLUME_NAME]+"_"+str(settings[Tags.WAVELENGTH])+".npz"

np.savez(volume_path,
         mua=volume[:, :, :, 0],
         mus=volume[:, :, :, 1],
         g=volume[:, :, :, 2],
         )

test = np.load(volume_path)[Tags.PROPERTY_ABSORPTION]

optical_path = run_optical_forward_model(settings, volume_path)

fluence = np.load(optical_path)['fluence']

print(fluence[24, 24, 10])
print(fluence[24, 24, 90])

plt.imshow(fluence[:, 24, :], vmin=0, vmax=0.1)
plt.show()



