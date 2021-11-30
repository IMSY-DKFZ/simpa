# SPDX-FileCopyrightText: 2021 Computer Assisted Medical Interventions Group, DKFZ
# SPDX-FileCopyrightText: 2021 Janek Groehl
# SPDX-License-Identifier: MIT

import unittest
from unittest.case import TestCase
import numpy as np
import matplotlib.pyplot as plt
from simpa.core.simulation_modules.reconstruction_module.reconstruction_utils import bandpass_filtering_with_settings, bandpass_filtering
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

        filtered_time_series_with_settings = bandpass_filtering_with_settings(combined_time_series, settings,
                                                                              settings[Tags.RECONSTRUCTION_MODEL_SETTINGS],
                                                                              device)

        filtered_time_series = bandpass_filtering(combined_time_series, 1e-3, int(11000), int(9000), 0)

        # check if both bandpass filtering methods return the same result
        assert np.array_equal(filtered_time_series, filtered_time_series_with_settings)

        assert (np.abs(base_time_series - filtered_time_series) < 1e-5).all()

        if Visualize:
            labels = ["Base time series", "Low frequency time series", "High frequency time series",
                      "Combined time series", "Filtered time series"]
            for i, time_series in enumerate([base_time_series, low_freq_time_series, high_freq_time_series,
                                             combined_time_series, filtered_time_series]):
                plt.subplot(5, 1, i + 1)
                plt.title(labels[i])
                plt.plot(t_values, time_series)
            plt.tight_layout()
            plt.show()
