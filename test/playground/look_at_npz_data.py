import numpy as np
import matplotlib.pylab as plt

wavelength = "700"
PATH = "/media/janek/PA DATA/DS_Forearm/Forearm_17420000/"
data_input = np.load(PATH + "/properties_" + wavelength + "nm.npz")
data_output = np.load(PATH + "/optical_forward_model_output_" + wavelength + ".npz")

fluence = data_output['fluence']
absorption = data_input['mua']

p0 = data_output['initial_pressure']

time_series_data = np.load(PATH + "/noisy_acoustic_model_output_" + wavelength + ".npz")["time_series_data"]
recon = np.load(PATH + "/reconstruction_result_" + wavelength + ".npz")["reconstruction"]

print(np.shape(time_series_data))

plt.subplot(131)
plt.imshow((np.rot90(absorption[:, 100:], -1)))
plt.subplot(132)
plt.imshow((np.rot90(time_series_data[:, 780:], -1)), aspect=0.08)
plt.subplot(133)
plt.imshow((np.rot90(recon[:, 120:, 0], -1)))
plt.show()
