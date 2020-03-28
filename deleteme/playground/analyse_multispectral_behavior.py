import glob
import numpy as np
from simulate import Tags
import matplotlib.pylab as plt

SIM_PATH = "/home/janek/simulation_test/"

folder_names = glob.glob(SIM_PATH+"/*")
folder_name = folder_names[0]

settings = np.load(folder_name+"/settings.npz")['settings'].item()


wavelengths = settings[Tags.WAVELENGTHS]

muas = [None] * len(wavelengths)
for wl_idx, wavelength in enumerate(wavelengths):
    muas[wl_idx] = np.load(folder_name + "/properties_" + str(wavelength) + "nm.npz")['mua']

oxy = np.load(folder_name + "/properties_800nm.npz")['oxy']
seg = np.load(folder_name + "/properties_800nm.npz")['seg']

muas = np.asarray(muas)

print(np.shape(muas))

spectrum = np.reshape(muas[:, :, :, 100:], (26, -1))
spectrum = spectrum / np.sum(spectrum)
plt.semilogy(spectrum[:,0:1000], alpha=0.1)
plt.show()

plt.imshow(np.rot90((muas[25, 30, :, :]), -1))
plt.show()
