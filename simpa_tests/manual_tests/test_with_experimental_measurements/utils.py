# SPDX-FileCopyrightText: 2021 Computer Assisted Medical Interventions Group, DKFZ
# SPDX-FileCopyrightText: 2021 Janek Groehl
# SPDX-License-Identifier: MIT

import os
import numpy as np
import pandas as pd


def read_rxt_file(file_path: str) -> (np.ndarray, np.ndarray, np.ndarray, float):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"You need to supply the path to a file, not {file_path}")

    dataframe = pd.read_csv(file_path, " ", skiprows=1, header=None)

    with open(file_path, "r") as metadata_path:
        metadata = metadata_path.readline().split("\t")

    return (np.asarray(dataframe[0].values), np.asarray(dataframe[1].values),
            np.asarray(dataframe[2].values), float(metadata[2]))


def read_reference_spectra(file_path: str) -> (np.ndarray, np.ndarray, float):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"You need to supply the path to a file, not {file_path}")

    data = np.load(file_path)
    return data["mua_mean"], data["mus_mean"], data["g"]
