# SPDX-FileCopyrightText: 2021 Computer Assisted Medical Interventions Group, DKFZ
# SPDX-FileCopyrightText: 2021 Janek Groehl
# SPDX-License-Identifier: MIT

import unittest
from unittest.case import TestCase
import numpy as np
import matplotlib.pyplot as plt
from simpa.core.simulation_modules.reconstruction_module.reconstruction_utils import tukey_bandpass_filtering_with_settings, tukey_bandpass_filtering, butter_bandpass_filtering, butter_bandpass_filtering_with_settings
from simpa.utils import Settings, Tags
from simpa.core.device_digital_twins import LinearArrayDetectionGeometry

Visualize = False


class TestBandpassFilter(unittest.TestCase):

    def setUp(self) -> None:

        self.settings = Settings()
        self.settings.set_reconstruction_settings({
            Tags.RECONSTRUCTION_PERFORM_BANDPASS_FILTERING: True,
            Tags.TUKEY_WINDOW_ALPHA: 0,
            Tags.BANDPASS_CUTOFF_LOWPASS: int(11000),
            Tags.BANDPASS_CUTOFF_HIGHPASS: int(9000),
            Tags.BUTTERWORTH_FILTER_ORDER: 1,
        })

        self.device = LinearArrayDetectionGeometry(
            sampling_frequency_mhz=1)

        # Calculate base sinus as time series datum
        self.t_values = np.arange(0, 20, 0.01)

        base_frequency = 1
        self.base_time_series = np.sin(2 * np.pi * base_frequency * self.t_values)

        high_frequency = 3
        self.high_freq_time_series = np.sin(2 * np.pi * high_frequency * self.t_values)

        low_frequency = 0.1
        self.low_freq_time_series = np.sin(2 * np.pi * low_frequency * self.t_values)

        self.combined_time_series = self.base_time_series + self.low_freq_time_series + self.high_freq_time_series

    def test_butter_bandpass_filter(self):
        filtered_time_series_with_settings = butter_bandpass_filtering_with_settings(self.combined_time_series, self.settings, self.settings[Tags.RECONSTRUCTION_MODEL_SETTINGS], self.device)


        filtered_time_series_with_resampling = butter_bandpass_filtering(self.combined_time_series, 1e-3, int(11000), int(9000), 0, True)

        filtered_time_series = butter_bandpass_filtering(self.combined_time_series, 1e-3, int(11000), int(9000), 0)

        # check if both bandpass filtering methods return the same result
        assert np.array_equal(filtered_time_series, filtered_time_series_with_settings)

        # compare after 500 steps as the filter does not work perfectly before
        assert (np.abs(self.base_time_series[500:] - filtered_time_series[500:]) < 0.2).all()

        assert (np.abs(self.base_time_series - filtered_time_series_with_resampling) < 1e-1).all()

        if Visualize:
            labels = ["Base time series", "Low frequency time series", "High frequency time series",
                      "Combined time series", "Filtered time series"]
            for i, time_series in enumerate([self.base_time_series, self.low_freq_time_series, self.high_freq_time_series, self.combined_time_series, filtered_time_series]):
                plt.subplot(5, 1, i + 1)
                plt.title(labels[i])
                plt.plot(self.t_values, time_series)
            plt.tight_layout()
            plt.show()

    def test_tukey_bandpass_filter(self):
        filtered_time_series_with_settings = tukey_bandpass_filtering_with_settings(self.combined_time_series, self.settings,
                                                                                    self.settings[Tags.RECONSTRUCTION_MODEL_SETTINGS],
                                                                                    self.device)

        filtered_time_series = tukey_bandpass_filtering(self.combined_time_series, 1e-3, int(11000), int(9000), 0)

        # check if both bandpass filtering methods return the same result
        assert np.array_equal(filtered_time_series, filtered_time_series_with_settings)

        assert (np.abs(self.base_time_series - filtered_time_series) < 1e-5).all()

        if Visualize:
            labels = ["Base time series", "Low frequency time series", "High frequency time series",
                      "Combined time series", "Filtered time series"]
            for i, time_series in enumerate([self.base_time_series, self.low_freq_time_series, self.high_freq_time_series, self.combined_time_series, filtered_time_series]):
                plt.subplot(5, 1, i + 1)
                plt.title(labels[i])
                plt.plot(self.t_values, time_series)
            plt.tight_layout()
            plt.show()

if __name__ == '__main__':
    Visualize = True
    test = TestBandpassFilter()
    test.setUp()
    test.test_tukey_bandpass_filter()
    test.test_butter_bandpass_filter()