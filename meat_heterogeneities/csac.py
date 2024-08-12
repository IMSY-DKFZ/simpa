# SPDX-FileCopyrightText: 2021 Division of Intelligent Medical Systems, DKFZ
# SPDX-FileCopyrightText: 2021 Janek Groehl
# SPDX-License-Identifier: MIT

import matplotlib.pyplot as plt

import simpa as sp
import nrrd
file = "/home/f762e/Workspace/data/Simulated_data/meat_2/both_exponential_51_0.6_0_0.1_1400_1640_961_1178.hdf5"
sp.VisualiseData(path_to_hdf5_file=file, show_initial_pressure=True,
                 show_blood_volume_fraction=True, show_reconstructed_data=True, wavelengths=[700, 800], log_scale=True)

nrrd_file = "/home/f762e/Workspace/data/Real_data/meat_measurements/meat_study_pa/Scan_50_pa.nrrd"
data, header = nrrd.read(nrrd_file)

plt.imshow(data[:, :, 0, 10])
plt.show()
