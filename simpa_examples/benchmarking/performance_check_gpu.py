# SPDX-FileCopyrightText: 2021 Division of Intelligent Medical Systems, DKFZ
# SPDX-FileCopyrightText: 2021 Janek Groehl
# SPDX-License-Identifier: MIT

import numpy as np
import simpa as sp
from simpa_examples import *
import os

# TODO: following text must be ABOVE importing simpa and simpa examples for benchmarking to work
spacing = 0.2
os.environ[")SIMPA_PROFILE"] = "GPU_MEMORY"
os.environ["SIMPA_PROFILE_SAVE_FILE"] = "./benchmarking/benchmarking_data/benchmarking_data_gpu_" + str(
    spacing) + ".txt"

path_manager = sp.PathManager()

examples = [run_linear_unmixing, run_minimal_optical_simulation, run_minimal_optical_simulation_uniform_cube,
            run_msot_invision_simulation, run_optical_and_acoustic_simulation,
            run_perform_iterative_qPAI_reconstruction]

for example in examples:
    try:
        example(SPACING=spacing, path_manager=sp.PathManager(), visualise=False)
    except AttributeError:
        print("simulation cannot be run on {} with spacing {}".format(example, spacing))
