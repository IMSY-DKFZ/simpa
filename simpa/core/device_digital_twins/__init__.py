"""
SPDX-FileCopyrightText: 2021 Computer Assisted Medical Interventions Group, DKFZ
SPDX-FileCopyrightText: 2021 VISION Lab, Cancer Research UK Cambridge Institute (CRUK CI)
SPDX-License-Identifier: MIT
"""

from simpa.utils import Tags
from simpa.core.device_digital_twins.msot_device import MSOTAcuityEcho
from simpa.core.device_digital_twins.rsom_device import RSOMExplorerP50
from simpa.core.device_digital_twins.invision_device import InVision256TF

"""
This DEVICE_MAP can be used in order to obtain appropriate device specifications based on
the desired device design.
"""
DEVICE_MAP = {
    Tags.DIGITAL_DEVICE_MSOT_ACUITY: MSOTAcuityEcho(),
    Tags.DIGITAL_DEVICE_RSOM: RSOMExplorerP50(),
    Tags.DIGITAL_DEVICE_MSOT_INVISION: InVision256TF()
}
