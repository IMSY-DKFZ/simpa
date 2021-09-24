"""
SPDX-FileCopyrightText: 2021 Computer Assisted Medical Interventions Group, DKFZ
SPDX-FileCopyrightText: 2021 VISION Lab, Cancer Research UK Cambridge Institute (CRUK CI)
SPDX-License-Identifier: MIT
"""

import unittest
import numpy as np
import matplotlib.pyplot as plt
from simpa.core.simulation_modules.reconstruction_module.reconstruction_utils import bandpass_filtering
from simpa.utils import Settings, Tags
from simpa.core.device_digital_twins import LinearArrayDetectionGeometry

Visualize = False


class TestBandpassFilter(unittest.TestCase):

    def test_bandpass(self):

        settings = Settings()
        settings.set_reconstruction_settings({
            Tags.RECONSTRUCTION_PERFORM_BANDPASS_FILTERING: True,
            Tags.TUKEY_WINDOW_ALPHA: 0,
            Tags.BANDPASS_CUTOFF_LOWPASS: int(11000),
            Tags.BANDPASS_CUTOFF_HIGHPASS: int(9000),
        })

        device = LinearArrayDetectionGeometry(
                 sampling_frequency_mhz=1)

        # Calculate base sinus as time series datum
        t_values = np.arange(0, 20, 0.01)

        base_frequency = 1
        base_time_series = np.sin(2 * np.pi * base_frequency * t_values)

        high_frequency = 3
        high_freq_time_series = np.sin(2 * np.pi * high_frequency * t_values)

        low_frequency = 0.1
        low_freq_time_series = np.sin(2 * np.pi * low_frequency * t_values)

        combined_time_series = base_time_series + low_freq_time_series + high_freq_time_series

        filtered_time_series = bandpass_filtering(combined_time_series, settings,
                                                  settings[Tags.RECONSTRUCTION_MODEL_SETTINGS],
                                                  device)

        normalized_filtered_times_series = filtered_time_series/np.max(filtered_time_series)

        assert np.abs(base_time_series - normalized_filtered_times_series).all() < 1e-5

        if Visualize:
            for i, time_series in enumerate([base_time_series, low_freq_time_series, high_freq_time_series,
                                             combined_time_series, normalized_filtered_times_series]):
                plt.subplot(5, 1, i + 1)
                plt.plot(t_values, time_series)
            plt.show()
