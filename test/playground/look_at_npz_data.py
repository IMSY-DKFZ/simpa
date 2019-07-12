import numpy as np
import matplotlib.pylab as plt

PATH = "/media/janek/Maxtor/mcx_simulation/Forearm_040948/"
#PATH = "/home/janek/E130-Projekte/Photoacoustics/RawData/20190619_forearm_data_mcx/validation/Forearm_013200/"
#814, 810, 863, 904, 891, 915
data_input = np.load(PATH + "/properties_700nm.npz")
data_output = np.load(PATH + "/optical_forward_model_output_700.npz")

fluence = data_output['fluence']
absorption = data_input['mua']
segmentation = data_input['seg']
p0 = fluence * absorption

print(np.shape(p0))

plt.subplot(131)
plt.imshow(np.log10(np.rot90(fluence[:, :], -1)))
plt.subplot(132)
plt.imshow(np.log10(np.rot90(absorption[:, :], -1)))
plt.subplot(133)
plt.imshow(np.log10(np.rot90(p0[:, :], -1)))
plt.show()

# plt.subplot(131)
# plt.imshow(np.log10(np.rot90(fluence[:, int(np.shape(fluence)[1]/2), :], -1)))
# plt.subplot(132)
# plt.imshow((np.rot90(segmentation[:, int(np.shape(fluence)[1]/2), :], -1)))
# plt.subplot(133)
# plt.imshow(np.log10(np.rot90(p0[:, int(np.shape(fluence)[1]/2), :], -1)))
# plt.show()
