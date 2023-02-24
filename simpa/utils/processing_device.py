# SPDX-FileCopyrightText: 2021 Division of Intelligent Medical Systems, DKFZ
# SPDX-FileCopyrightText: 2021 Janek Groehl
# SPDX-License-Identifier: MIT

from doctest import ELLIPSIS_MARKER
import torch
from simpa.log import Logger
from simpa.utils import Tags, Settings


def get_processing_device(global_settings: Settings = None) -> torch.device:
    """
    Get device (CPU/GPU) for data processing. By default use GPU for fast computation, unless the user manually sets it to CPU. Of course, GPU is only used if available. The user receives a warning if GPU was specified but is not available, in this case processing is done on CPU as fall-back.
    :param global_settings: global settings defined by user
    :type global_settings: Settings
    :return: torch device for processing
    """

    logger = Logger()

    dev = "cuda"  # by default, the GPU is used

    # if the user has specified to use a CPU, do so
    if global_settings is not None:
        if Tags.GPU in global_settings:
            if not global_settings[Tags.GPU]:
                dev = "cpu"

    # if no GPU is available, use the CPU and inform the user
    if dev == "cuda" and not torch.cuda.is_available():
        dev = "cpu"
        if global_settings is not None:
            if Tags.GPU in global_settings:
                logger.warning('Cuda is not available! Check your torch/cuda version. Processing will be done on CPU instead.')

    logger.debug(f"Processing is done on {dev}")

    return torch.device(dev)
