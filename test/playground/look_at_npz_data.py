import numpy as np
import matplotlib.pylab as plt

wavelength = "700"
PATH = "/media/janek/PA DATA/DS_Forearm/Forearm_17420000/"
data_input = np.load(PATH + "/properties_" + wavelength + "nm.npz")
data_output = np.load(PATH + "/optical_forward_model_output_" + wavelength + ".npz")

fluence = data_output['fluence']
absorption = data_input['mua']

p0 = data_output['initial_pressure']

time_series_data = np.load(PATH + "/acoustic_forward_model_output_" + wavelength + ".npz")["time_series_data"]
recon = np.load(PATH + "/reconstruction_result_" + wavelength + ".npz")["reconstruction"]

print(np.shape(time_series_data))

plt.subplot(131)
plt.imshow((np.rot90(p0[:, :], -1)))
plt.subplot(132)
plt.imshow((np.rot90(time_series_data[:, :], -1)), aspect=0.08)
plt.subplot(133)
plt.imshow((np.rot90(recon[:, :, 0], -1)))
plt.show()
