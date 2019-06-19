import numpy as np
import matplotlib.pylab as plt

PATH = "/home/janek/simulation_test/homogeneous_cube"

data_input = np.load(PATH + "/homogeneous_cube_800.npz")
data_output = np.load(PATH + "/optical_forward_model_output_800.npz")

plt.imshow(np.log10(data_output['fluence'][50, :, :]))
plt.show()