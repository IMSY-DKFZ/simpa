import numpy as np
import matplotlib.pylab as plt

PATH = "/home/janek/E130-Projekte/Photoacoustics/RawData/20190703_upsampling_experiment/Structure_060000/"

#814, 810, 863, 904, 891, 915
data_input = np.load(PATH + "/properties_814nm.npz")
data_output = np.load(PATH + "/optical_forward_model_output_814.npz")

fluence = data_output['fluence']
absorption = data_input['mua']
p0 = fluence * absorption

plt.subplot(131)
plt.imshow(np.log10(np.rot90(fluence[:, int(np.shape(fluence)[1]/2), int(np.shape(fluence)[2]/2):], -1)))
plt.subplot(132)
plt.imshow(np.log10(np.rot90(absorption[:, int(np.shape(absorption)[1]/2), int(np.shape(fluence)[2]/2):], -1)))
plt.subplot(133)
plt.imshow(np.log10(np.rot90(p0[:, int(np.shape(p0)[1]/2), int(np.shape(fluence)[2]/2):], -1)))
plt.show()
