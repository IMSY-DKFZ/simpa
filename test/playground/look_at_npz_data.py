import numpy as np
import matplotlib.pylab as plt

PATH = "/media/janek/PA DATA/tmp/TestData_000000/"
data_input = np.load(PATH + "/properties_800nm.npz")
data_output = np.load(PATH + "/optical_forward_model_output_800.npz")

fluence = data_output['fluence']
absorption = data_input['mua']
p0 = fluence * absorption

time_series_data = np.load(PATH + "/acoustic_forward_model_output.npz")["time_series_data"]

#time_series_data = np.log10(time_series_data - np.min(time_series_data))

print(np.shape(time_series_data))

# plt.subplot(131)
# plt.imshow(np.log10(np.rot90(fluence[:, :], -1)))
# plt.subplot(132)
# plt.imshow(np.log10(np.rot90(absorption[:, :], -1)))
# plt.subplot(133)
# plt.imshow(np.log10(np.rot90(p0[:, :], -1)))
# plt.show()

plt.subplot(121)
plt.imshow(np.log10(np.rot90(p0[:, :], -1)))
plt.subplot(122)
plt.imshow((np.rot90(time_series_data[:, :-10], -1)))
plt.show()
