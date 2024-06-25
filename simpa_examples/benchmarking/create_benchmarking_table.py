# SPDX-FileCopyrightText: 2021 Division of Intelligent Medical Systems, DKFZ
# SPDX-FileCopyrightText: 2021 Janek Groehl
# SPDX-License-Identifier: MIT

import os
import numpy as np
from prettytable import PrettyTable


def lines_that_contain(string, fp):
    """
    Function to determine if a string contains a certain word/phrase
    :param string: what to check
    :param fp:
    :return:
    """
    return [line for line in fp if string in line]


examples = ['linear_unmixing', 'minimal_optical_simulation', 'minimal_optical_simulation_uniform_cube',
            'msot_invision_simulation', 'optical_and_acoustic_simulation',
            'perform_iterative_qPAI_reconstruction', 'segmentation_loader']
profiles = ['TIME', "GPU_MEMORY", "MEMORY"]
spacings = np.arange(0.2, 0.6, 0.2)
benchmarking_dict = {}
for example in examples:
    benchmarking_dict[example] = {}
    for spacing in spacings:
        benchmarking_dict[example][spacing] = {}

info_starts = {"MEMORY": 19, "GPU_MEMORY": 12, "TIME": 16}
info_ends = {"MEMORY": 29, "GPU_MEMORY": 26, "TIME": 29}

for profile in profiles:
    for spacing in spacings:
        file_name = "./benchmarking_bash/benchmarking_data_"+profile+"_"+str(spacing)+".txt"
        benchmarking_file = open(file_name, 'r')
        current_examples = []

        if profile == 'TIME':
            example_name_lines = lines_that_contain("File:", benchmarking_file)

            for enl in example_name_lines:
                example = enl.rpartition("simpa_examples/")[2].rpartition(".py")[0]
                if example not in current_examples:
                    current_examples.append(example)
                else:
                    break

        elif profile == 'GPU_MEMORY':
            example_name_lines = lines_that_contain("##", benchmarking_file)

            for enl in example_name_lines:
                example = enl.rpartition("run_")[2].rpartition("\n")[0]
                if example not in current_examples:
                    current_examples.append(example)
                else:
                    break

        if profile == 'MEMORY':
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
            if profile == "TIME":
                unit = ""

            example = current_examples[examples_counter]
            benchmarking_dict[example][spacing][profile] = str(value)+unit
            examples_counter += 1
            if examples_counter == len(current_examples):
                examples_counter = 0


table = PrettyTable()
table.field_names = ["Example", "Spacing mm", 'Time μs', "GPU", "MEMORY"]
divider = False
for example, spacing_dict in benchmarking_dict.items():
    subrow_counter = 0
    for spacing, p_dict in spacing_dict.items():
        try:
            p_dict["MEMORY"]
        except KeyError:
            p_dict["MEMORY"] = "N/A"
        if subrow_counter == len(spacing_dict)-1:
            divider = True
        if subrow_counter == 0:
            table.add_row([example, spacing, p_dict['TIME'], p_dict["GPU_MEMORY"], p_dict["MEMORY"]], divider=divider)
        else:
            table.add_row(["", spacing, p_dict['TIME'], p_dict["GPU_MEMORY"], p_dict["MEMORY"]], divider=divider)
        subrow_counter += 1
        divider = False

table_file_name = "./benchmarking_bash/benchmarking_data_table.txt"
benchmarking_table_file = open(table_file_name, 'w')
benchmarking_table_file.write(table.get_string())
