import glob
import numpy as np
from ippai.simulate import Tags
import matplotlib.pylab as plt

SIM_PATH = "/home/janek/simulation_test/"

folder_names = glob.glob(SIM_PATH+"/*")
folder_name = folder_names[0]

settings = np.load(folder_name+"/settings.npz")['settings'].item()


wavelengths = settings[Tags.WAVELENGTHS]

muas = [None] * len(wavelengths)
for wl_idx, wavelength in enumerate(wavelengths):
    muas[wl_idx] = np.load(folder_name + "/properties_" + str(wavelength) + "nm.npz")['mua']

muas = np.asarray(muas)

spectrum = np.reshape(muas[:, 30, 26, 32], (-1,))
spectrum = spectrum / np.sum(spectrum)
plt.plot(spectrum)
plt.show()

plt.imshow(np.rot90((muas[25, 30, :, :]), -1))
plt.show()
