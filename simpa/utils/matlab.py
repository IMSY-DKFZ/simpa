# SPDX-FileCopyrightText: 2021 Division of Intelligent Medical Systems, DKFZ
# SPDX-FileCopyrightText: 2021 Janek Groehl
# SPDX-License-Identifier: MIT

import inspect
import os
from typing import List
import subprocess


def generate_matlab_cmd(matlab_binary_path: str, simulation_script_path: str, data_path: str,
                        matlab_runtime_path: str = '', matlab_compiled_scripts_path: str = '') -> List[str]:
    # ensure data path is an absolute path
    data_path = os.path.abspath(data_path)

    cmd = list()

    if matlab_runtime_path and matlab_compiled_scripts_path:
        module_loading = list()
        module_loading.append("module")
        module_loading.append("load mcr/23.2")
        subprocess.run(module_loading)

        cmd.append(os.path.join(matlab_compiled_scripts_path, simulation_script_path, f"run_{simulation_script_path}.sh"))
        cmd.append(matlab_runtime_path)
        cmd.append(data_path)
        # cmd.append("&&")
        # cmd.append("module unload")
        # cmd.append("mcr")
    else:
        # get path of calling script to add to matlab path
        base_script_path = os.path.dirname(os.path.abspath(inspect.stack()[1].filename))

        cmd.append(matlab_binary_path)
        cmd.append("-nodisplay")
        cmd.append("-nosplash")
        cmd.append("-automation")
        cmd.append("-wait")
        cmd.append("-r")
        cmd.append(f"addpath('{base_script_path}');{simulation_script_path}('{data_path}');exit;")
    return cmd

def generate_compiled_matlab_scripts(matlab_binary_path, kwave_binary_path, compiled_scripts_path):
    simulation_module = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'core', 'simulation_modules')
    acoustical_path = os.path.join(simulation_module, 'acoustic_forward_module')
    reconstruction_path = os.path.join(simulation_module, 'reconstruction_module')
    cmd = list()

    cmd.append(matlab_binary_path)
    cmd.append("-nodisplay")
    cmd.append("-nosplash")
    cmd.append("-automation")
    cmd.append("-wait")
    cmd.append("-r")

    cmd.append(
        f"mcc -m {os.path.join(acoustical_path, 'simulate_2D.m')} -d {os.path.join(compiled_scripts_path, 'simulate_2D')};"
        f"mcc -m {os.path.join(acoustical_path, 'simulate_3D.m')} -d {os.path.join(compiled_scripts_path, 'simulate_3D')};"
        f"mcc -m {os.path.join(reconstruction_path, 'time_reversal_2D.m')} -d {os.path.join(compiled_scripts_path, 'time_reversal_2D')};"
        f"mcc -m {os.path.join(reconstruction_path, 'time_reversal_3D.m')} -d {os.path.join(compiled_scripts_path, 'time_reversal_3D')};"
        f"exit;")

    subprocess.run(cmd)

if __name__ == "__main__":
    generate_compiled_matlab_scripts("/home/kris/Programs/MATLAB/R2023b/bin/matlab",
                                     "/home/kris/Projects/SIMPA/Repositories/k-wave/k-Wave/binaries/",
                                     "/home/kris/Projects/SIMPA/Binaries/")
