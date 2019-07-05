import numpy as np
import matplotlib.pylab as plt

PATH = "/media/janek/Maxtor/mcx_simulation/Forearm_040501/"
#PATH = "/home/janek/simulation_test/Structure_001000/"
#814, 810, 863, 904, 891, 915
data_input = np.load(PATH + "/properties_900nm.npz")
data_output = np.load(PATH + "/optical_forward_model_output_700.npz")

fluence = data_output['fluence']
absorption = data_input['mua']
p0 = fluence * absorption

print(np.shape(p0))

plt.subplot(131)
plt.imshow(np.log10(np.rot90(fluence[:, int(np.shape(fluence)[1]/2):], -1)))
plt.subplot(132)
plt.imshow(np.log10(np.rot90(absorption[:, int(np.shape(fluence)[1]/2):], -1)))
plt.subplot(133)
plt.imshow(np.log10(np.rot90(p0[:, int(np.shape(fluence)[1]/2):], -1)))
plt.show()
