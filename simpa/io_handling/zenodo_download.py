# SPDX-FileCopyrightText: 2021 Computer Assisted Medical Interventions Group, DKFZ
# SPDX-FileCopyrightText: 2021 Janek Groehl
# SPDX-License-Identifier: MIT

import requests
import json
import wget
import os


def download_from_zenodo(record_id: str, save_dir: str, sandbox: bool = False):
    """
    Downloads a .zip folder from zenodo.org with the specified Record ID.

    :param record_id: Record ID of the specified Zenodo file.
    :param save_dir: Output directory where the Zenodo file will be stored.
    :param sandbox: Boolean to specify whether to download from zenodo.org or sandbox.zenodo.org.
    """

    if sandbox:
        url = 'https://sandbox.zenodo.org/api/records/'
    else:
        url = 'https://zenodo.org/api/records/'

    # create directory, if necessary, then change to it
    os.makedirs(save_dir, exist_ok=True)

    record_id = record_id.strip()

    r = requests.get(url + record_id)

    if r.ok:
        text = json.loads(r.text)
        link = text["files"][0]["links"]["self"]
        wget.download(link, out=save_dir)
