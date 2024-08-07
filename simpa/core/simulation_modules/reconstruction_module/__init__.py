# SPDX-FileCopyrightText: 2021 Division of Intelligent Medical Systems, DKFZ
# SPDX-FileCopyrightText: 2021 Janek Groehl
# SPDX-License-Identifier: MIT
from .reconstruction_adapter_base import ReconstructionAdapterBase

from simpa.utils import Tags, Settings


def create_reconstruction_settings(speed_of_sound_in_m_per_s: int = 1540, time_spacing_in_s: float = 2.5e-8,
                                   sensor_spacing_in_mm: float = 0.1,
                                   recon_mode: str = Tags.RECONSTRUCTION_MODE_PRESSURE,
                                   apodization: str = Tags.RECONSTRUCTION_APODIZATION_BOX) -> Settings:
    """
    Function that creates SIMPA settings for reconstruction convenience function.

    :param speed_of_sound_in_m_per_s: (int) speed of sound in medium in meters per second (default: 1540 m/s)
    :param time_spacing_in_s: (float) time between sampling points in seconds (default: 2.5e-8 s which is equal to 40 MHz)
    :param sensor_spacing_in_mm: (float) space between sensor elements in millimeters (default: 0.1 mm)
    :param recon_mode: SIMPA Tag defining the reconstruction mode - pressure default OR differential
    :param apodization: SIMPA Tag defining the apodization function (default box)
    :return: SIMPA settings
    """

    settings = Settings()
    settings.set_reconstruction_settings({
        Tags.DATA_FIELD_SPEED_OF_SOUND: speed_of_sound_in_m_per_s,
        Tags.SPACING_MM: sensor_spacing_in_mm,
        Tags.RECONSTRUCTION_APODIZATION_METHOD: apodization,
        Tags.RECONSTRUCTION_MODE: recon_mode,
    })
    settings[Tags.K_WAVE_SPECIFIC_DT] = time_spacing_in_s

    return settings
