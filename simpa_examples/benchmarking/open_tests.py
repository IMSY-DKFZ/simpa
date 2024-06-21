# SPDX-FileCopyrightText: 2021 Division of Intelligent Medical Systems, DKFZ
# SPDX-FileCopyrightText: 2021 Janek Groehl
# SPDX-License-Identifier: MIT

import os
import numpy as np
from prettytable import PrettyTable


def lines_that_contain(string, fp):
    return [line for line in fp if string in line]


examples = ['linear_unmixing', 'minimal_optical_simulation', 'minimal_optical_simulation_uniform_cube',
            'msot_invision_simulation', 'optical_and_acoustic_simulation',
            'perform_iterative_qPAI_reconstruction']
profiles = ['time', "gpu", "memory"]
spacings = np.arange(0.2, 0.6, 0.2)
benchmarking_dict = {}
for example in examples:
    benchmarking_dict[example] = {}
    for spacing in spacings:
        benchmarking_dict[example][spacing] = {}

info_starts = {"memory": 19, "gpu": 12, "time": 16}
info_ends = {"memory": 29, "gpu": 26, "time": 29}

for profile in profiles:
    for spacing in spacings:
        file_name = "./benchmarking/benchmarking_data/benchmarking_data_"+profile+"_"+str(spacing)+".txt"
        benchmarking_file = open(file_name, 'r')
        current_examples = []

        if profile == 'time':
            example_name_lines = lines_that_contain("File:", benchmarking_file)

            for enl in example_name_lines:
                example = enl.rpartition("simpa_examples/")[2].rpartition(".py")[0]
                if example not in current_examples:
                    current_examples.append(example)
                else:
                    break

        elif profile == 'gpu':
            example_name_lines = lines_that_contain("##", benchmarking_file)

            for enl in example_name_lines:
                example = enl.rpartition("run_")[2].rpartition("\n")[0]
                if example not in current_examples:
                    current_examples.append(example)
                else:
                    break

        if profile == 'memory':
            example_name_lines = lines_that_contain("Filename:", benchmarking_file)

            for enl in example_name_lines:
                example = enl.rpartition("simpa_examples/")[2].rpartition(".py")[0]
                if example not in current_examples:
                    current_examples.append(example)
                else:
                    break

        benchmarking_file = open(file_name, 'r')
        lines_with_sp_simulate = lines_that_contain("sp.simulate", benchmarking_file)

        values = []
        units = []
        examples_counter = 0

        for lwss in lines_with_sp_simulate:
            value = float(lwss[info_starts[profile]:info_ends[profile]])

            unit = str(lwss[info_ends[profile]])
            if profile == "time":
                unit = ""

            example = current_examples[examples_counter]
            benchmarking_dict[example][spacing][profile] = str(value)+unit
            examples_counter += 1
            if examples_counter == len(current_examples):
                examples_counter = 0

# for line_with_sp_simulate in lines_with_sp_simulate:
#     value = float(line_with_sp_simulate[19:29])
#     unit = line_with_sp_simulate[29]
#     benchmarking_dict[file_names[examples_counter]][spacings[spacing_counter]][os.environ['SIMPA_PROFILE']+" value"] = value
#     benchmarking_dict[file_names[examples_counter]][spacings[spacing_counter]][os.environ['SIMPA_PROFILE']+" unit"] = unit
#     examples_counter += 1
#     if examples_counter == 5:
#         spacing_counter += 1
#         examples_counter = 0
#
# for line_with_sp_simulate in lines_with_sp_simulate:
#     value = float(line_with_sp_simulate[16:29])
#     benchmarking_dict[file_names[examples_counter]][spacings[spacing_counter]]["time"] = value
#     examples_counter += 1
#     if examples_counter == 6:
#         spacing_counter += 1
#         examples_counter = 0


table = PrettyTable()
table.field_names = ["Example", "Spacing mm", 'Time μs', "GPU", "MEMORY"]
divider = False
for example, spacing_dict in benchmarking_dict.items():
    subrow_counter = 0
    for spacing, p_dict in spacing_dict.items():
        try:
            p_dict["memory"]
        except KeyError:
            p_dict["memory"] = "N/A"
        if subrow_counter == len(spacing_dict)-1:
            divider = True
        if subrow_counter == 0:
            table.add_row([example, spacing, p_dict['time'], p_dict["gpu"], p_dict["memory"]], divider=divider)
        else:
            table.add_row(["", spacing, p_dict['time'], p_dict["gpu"], p_dict["memory"]], divider=divider)
        subrow_counter += 1
        divider = False
print(table)
