# The MIT License (MIT)
#
# Copyright (c) 2018 Computer Assisted Medical Interventions Group, DKFZ
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from simpa.io_handling import load_hdf5, save_hdf5
import matplotlib.pylab as plt
import numpy as np
from simpa.utils import SaveFilePaths

PATH = "path/to/file"
WAVELENGTH = 800  # currently only 800 and 900 are simulated as well

file = load_hdf5(PATH)

print(file['simulations'].keys())

fluence = (file['simulations']['original_data']['optical_forward_model_output']
           [str(WAVELENGTH)]['fluence'])
initial_pressure = (file['simulations']['original_data']
                    ['optical_forward_model_output']
                    [str(WAVELENGTH)]['initial_pressure'])
absorption = (file['simulations']['original_data']['simulation_properties']
              [str(WAVELENGTH)]['mua'])

shape = np.shape(fluence)

if len(shape) > 2:
    plt.figure()
    plt.subplot(231)
    plt.imshow(np.log10(fluence[int(shape[0]/2), :, :]))
    plt.subplot(232)
    plt.imshow(np.log10(absorption[int(shape[0]/2), :, :]))
    plt.subplot(233)
    plt.imshow(np.log10(initial_pressure))
    plt.subplot(234)
    plt.imshow(np.log10(fluence[:, int(shape[1]/2), :]))
    plt.subplot(235)
    plt.imshow(np.log10(absorption[:, int(shape[1]/2), :]))
    plt.subplot(236)
    plt.imshow(np.log10(initial_pressure))
    plt.show()
else:
    plt.figure()
    plt.subplot(131)
    plt.imshow(np.log10(fluence[1:129, -65:-1]))
    plt.subplot(132)
    plt.imshow(np.log10(absorption[1:129, -65:-1]))
    plt.subplot(133)
    plt.imshow(np.log10(initial_pressure[1:129, -65:-1]))
    plt.show()

save_hdf5(file, PATH)