import numpy as np
import matplotlib.pylab as plt
import glob

BASE_FOLDER = "/home/janek/melanie_test/RandomVolume_100000/"
WAVELENGTHS = [700]

folder = BASE_FOLDER
print("Working on " + folder + "...")
for wavelength in WAVELENGTHS:
    wavelength = str(wavelength)
    print(wavelength)

    data_input = np.load(folder + "/properties_" + wavelength + "nm.npz")
    data_output = np.load(folder + "/optical_forward_model_output_" + wavelength + ".npz")

    fluence = data_output['fluence']
    absorption = data_input['mua']
    oxygenation = data_input['oxy']

    p0 = data_output['initial_pressure']

    #time_series_data = np.load(folder + "/noisy_acoustic_model_output_" + wavelength + ".npz")["time_series_data"]
    #recon = np.load(folder + "/reconstruction_result_" + wavelength + ".npz")["reconstruction"]

    #print(np.shape(time_series_data))

    plt.subplot(121)
    plt.imshow(np.log10(np.rot90(p0[:, 17, :], -1)), cmap="gray")
    plt.subplot(122)
    plt.imshow((np.rot90(absorption[:, 17, :], -1)), cmap="magma")
    plt.show()
