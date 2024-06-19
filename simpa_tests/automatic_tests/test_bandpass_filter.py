# SPDX-FileCopyrightText: 2021 Division of Intelligent Medical Systems, DKFZ
# SPDX-FileCopyrightText: 2021 Janek Groehl
# SPDX-License-Identifier: MIT

import unittest
import numpy as np
import matplotlib.pyplot as plt
from simpa.core.simulation_modules.reconstruction_module.reconstruction_utils import tukey_bandpass_filtering_with_settings, tukey_bandpass_filtering, tukey_window_function, butter_bandpass_filtering, butter_bandpass_filtering_with_settings
from simpa.utils import Settings, Tags
from simpa.core.device_digital_twins import LinearArrayDetectionGeometry


class TestBandpassFilter(unittest.TestCase):

    def setUp(self) -> None:

        self.settings = Settings()
        self.settings.set_reconstruction_settings({
            Tags.RECONSTRUCTION_PERFORM_BANDPASS_FILTERING: True,
            Tags.TUKEY_WINDOW_ALPHA: 0,
            Tags.BANDPASS_CUTOFF_LOWPASS_IN_HZ: int(11000),
            Tags.BANDPASS_CUTOFF_HIGHPASS_IN_HZ: int(9000),
            Tags.BUTTERWORTH_FILTER_ORDER: 1,
        })

        self.device = LinearArrayDetectionGeometry(
            sampling_frequency_mhz=1)

        # Calculate base sinus as time series datum
        self.t_step = 0.01  # in ms
        self.t_values = np.arange(0, 20, self.t_step)  # in ms

        base_frequency = 1  # in 10^4 Hz
        self.base_time_series = np.sin(2 * np.pi * base_frequency * self.t_values)

        high_frequency = 3  # in 10^4 Hz
        self.high_freq_time_series = np.sin(2 * np.pi * high_frequency * self.t_values)

        low_frequency = 0.1  # in 10^4 Hz
        self.low_freq_time_series = np.sin(2 * np.pi * low_frequency * self.t_values)

        self.combined_time_series = self.base_time_series + self.low_freq_time_series + self.high_freq_time_series

        ## for tests with random signal ##

        # generate random noisy signal
        random_t_values = np.arange(0, 10, 0.01)
        np.random.seed(4117)
        noisy_frequency = np.random.rand(len(random_t_values))
        self.noisy_signal = np.sin(2 * np.pi * noisy_frequency * random_t_values)

        # filter signal
        self.cutoff_highpass_in_Hz = 40000
        self.cutoff_lowpass_in_Hz = 400000
        self.time_spacing_in_ms = 1e-3
        self.frequencies = np.fft.fftshift(np.fft.fftfreq(len(random_t_values), self.time_spacing_in_ms))  # in kHz

    def test_butter_bandpass_filter(self, show_figure_on_screen=False):
        self.settings.get_reconstruction_settings()[Tags.BANDPASS_FILTER_METHOD] = Tags.BUTTERWORTH_BANDPASS_FILTER
        self.settings.get_reconstruction_settings()[Tags.BUTTERWORTH_FILTER_ORDER] = 1
        filtered_time_series_with_settings = butter_bandpass_filtering_with_settings(
            self.combined_time_series, self.settings, self.settings[Tags.RECONSTRUCTION_MODEL_SETTINGS], self.device)

        filtered_time_series = butter_bandpass_filtering(self.combined_time_series, 1e-3, int(11000), int(9000), 1)

        # check if both bandpass filtering methods return the same result
        assert np.array_equal(filtered_time_series, filtered_time_series_with_settings)

        # compare after 500 steps as the filter does not work perfectly before
        assert (np.abs(self.base_time_series[500:] - filtered_time_series[500:]) < 0.2).all()

        if show_figure_on_screen:
            labels = ["Base time series", "Low frequency time series", "High frequency time series",
                      "Combined time series", "Filtered time series"]
            for i, time_series in enumerate([self.base_time_series, self.low_freq_time_series, self.high_freq_time_series, self.combined_time_series, filtered_time_series]):
                plt.subplot(5, 1, i + 1)
                plt.title(labels[i])
                plt.plot(self.t_values, time_series)
            plt.tight_layout()
            plt.show()

    def test_tukey_window_function(self, show_figure_on_screen=False):
        """
        Test tukey window function for alpha = 0.0 for uneven and even number of time steps.
        Test usual case and edge cases.
        """
        target_size = len(self.t_values)
        delta_f = 1/(target_size * self.t_step/1000)
        frequencies = np.fft.rfftfreq(target_size, self.t_step/1000)

        # examplary cutoffs usual use case
        low_index = int(target_size/8)
        high_index = int(3*target_size/8)
        cutoff_lowpass_in_Hz = int(high_index*delta_f)
        cutoff_highpass_in_Hz = int(low_index*delta_f)
        # reference window
        expected_window = np.zeros_like(frequencies)
        expected_window[low_index:high_index+1] = 1
        # generated window to be tested
        window = tukey_window_function(target_size=target_size, time_spacing_in_ms=self.t_step,
                                       cutoff_lowpass_in_Hz=cutoff_lowpass_in_Hz, cutoff_highpass_in_Hz=cutoff_highpass_in_Hz,
                                       tukey_alpha=0.0)
        if show_figure_on_screen:
            fig, ax = plt.subplots(3, 1, figsize=(6.4*3, 4.8*3), gridspec_kw={'hspace': 0.5})
            ax[0].set_title("Usual use case")
            ax[0].plot(frequencies, expected_window, color="lightgreen", label="reference window")
            ax[0].plot(frequencies, window, linestyle="dashed", color="darkblue", label="generated_window")
            ax[0].set_xlabel("Frequency f [Hz]")
            ax[0].legend()
        # test
        np.array_equal(expected_window, window)

        # check edge case 1: no filtering at all (expect only ones in window)
        cutoff_lowpass_in_Hz = int(10e20)
        cutoff_highpass_in_Hz = int(0)
        # reference window
        expected_window = np.ones_like(frequencies)
        # generated window to be tested
        window = tukey_window_function(target_size=target_size, time_spacing_in_ms=self.t_step,
                                       cutoff_lowpass_in_Hz=cutoff_lowpass_in_Hz, cutoff_highpass_in_Hz=cutoff_highpass_in_Hz,
                                       tukey_alpha=0.0)
        if show_figure_on_screen:
            ax[1].set_title("Edge case: No filtering")
            ax[1].plot(frequencies, expected_window, color="green", label="reference window")
            ax[1].plot(frequencies, window, linestyle="dashed", color="black", label="generated_window")
            ax[1].set_xlabel("Frequency f [Hz]")
        # test
        np.array_equal(expected_window, window)

        # check edge case 2: filtering exactly one represented frequency (expected only one one in window at given frequency)
        low_index = np.random.randint(0, len(frequencies)-1)
        high_index = low_index
        cutoff_lowpass_in_Hz = int(high_index*delta_f)
        cutoff_highpass_in_Hz = int(low_index*delta_f)
        # reference window
        expected_window = np.zeros_like(frequencies)
        expected_window[low_index:high_index+1] = 1
        # generated window to be tested
        window = tukey_window_function(target_size=target_size, time_spacing_in_ms=self.t_step,
                                       cutoff_lowpass_in_Hz=cutoff_lowpass_in_Hz, cutoff_highpass_in_Hz=cutoff_highpass_in_Hz,
                                       tukey_alpha=0.0)
        if show_figure_on_screen:
            ax[2].set_title("Edge case: filtering exactly one frequency")
            ax[2].plot(frequencies, expected_window, color="green", label="reference window")
            ax[2].plot(frequencies, window, linestyle="dashed", color="black", label="generated_window")
            ax[2].set_xlabel("Frequency f [Hz]")
            plt.tight_layout()
            plt.show()
        # test
        np.array_equal(expected_window, window)

    def test_tukey_bandpass_filter(self, show_figure_on_screen=False):
        filtered_time_series_with_settings = tukey_bandpass_filtering_with_settings(self.combined_time_series, self.settings,
                                                                                    self.settings[Tags.RECONSTRUCTION_MODEL_SETTINGS],
                                                                                    self.device)

        filtered_time_series_with_resampling = tukey_bandpass_filtering(
            self.combined_time_series, 1e-3, int(11000), int(9000), 0, True)

        filtered_time_series = tukey_bandpass_filtering(self.combined_time_series, 1e-3, int(11000), int(9000), 0)

        dirac_window_filtered_time_series = tukey_bandpass_filtering(
            self.combined_time_series, 1e-3, int(10000), int(10000), 0)

        # check if both bandpass filtering methods return the same result
        assert np.array_equal(filtered_time_series, filtered_time_series_with_settings)

        assert (np.abs(self.base_time_series - filtered_time_series) < 1e-5).all()

        assert (np.abs(self.base_time_series - dirac_window_filtered_time_series < 1e-5).all())

        assert (np.abs(self.base_time_series - filtered_time_series_with_resampling) < 1e-1).all()

        if show_figure_on_screen:
            labels = ["Base time series", "Low frequency time series", "High frequency time series",
                      "Combined time series", "Filtered time series"]
            for i, time_series in enumerate([self.base_time_series, self.low_freq_time_series, self.high_freq_time_series, self.combined_time_series, filtered_time_series]):
                plt.subplot(5, 1, i + 1)
                plt.title(labels[i])
                plt.plot(self.t_values, time_series)
            plt.tight_layout()
            plt.show()

    def visualize_filtered_spectrum(self, filtered_spectrum: np.ndarray):
        '''
        Visualizes the filtered spectrum and the intervals which are marked as passing or filtering areas.
        '''

        def plot_region(low_limit: int, high_limit: int, label: str = '', color: str = 'red'):
            interval = np.arange(low_limit, high_limit, self.time_spacing_in_ms)
            plt.plot(interval, np.zeros_like(interval), label=label, linewidth=5, color=color)
        # passing intervals
        plot_region(self.cutoff_highpass_in_Hz*self.time_spacing_in_ms, self.cutoff_lowpass_in_Hz*self.time_spacing_in_ms,
                    label="bandwidth of passed filter", color="green")
        plot_region(-self.cutoff_lowpass_in_Hz*self.time_spacing_in_ms, -
                    self.cutoff_highpass_in_Hz*self.time_spacing_in_ms, color="green")

        # filtered intervals
        plot_region(-self.cutoff_highpass_in_Hz*self.time_spacing_in_ms, self.cutoff_highpass_in_Hz*self.time_spacing_in_ms,
                    label="signal should be close to zero", color="red")
        plot_region(self.frequencies[0], -self.cutoff_lowpass_in_Hz*self.time_spacing_in_ms)
        plot_region(self.cutoff_lowpass_in_Hz*self.time_spacing_in_ms, self.frequencies[-1])

        plt.plot(self.frequencies, filtered_spectrum, label="filtered signal")
        plt.xlabel("frequency")
        plt.legend()
        plt.show()

    def test_tukey_filter_with_random_signal(self, show_figure_on_screen=False):

        filtered_signal = tukey_bandpass_filtering(
            self.noisy_signal, self.time_spacing_in_ms, self.cutoff_lowpass_in_Hz, self.cutoff_highpass_in_Hz, tukey_alpha=0.5)

        # compute frequency spectrum
        FILTERED_SIGNAL = np.fft.fftshift(np.fft.fft(filtered_signal))

        # check edge case: no filtering
        not_filtered_signal = tukey_bandpass_filtering(
            self.noisy_signal, self.time_spacing_in_ms, cutoff_highpass_in_Hz=0, cutoff_lowpass_in_Hz=1e40, tukey_alpha=0.0)
        assert np.allclose(self.noisy_signal, not_filtered_signal, atol=1e-12)

        # expected to be close to zero outside of the band
        high_cutoff_point = self.cutoff_highpass_in_Hz*self.time_spacing_in_ms
        low_cutoff_point = self.cutoff_lowpass_in_Hz*self.time_spacing_in_ms
        assert np.allclose(0, FILTERED_SIGNAL[self.frequencies < -low_cutoff_point], atol=1e-5)
        assert np.allclose(0, FILTERED_SIGNAL[self.frequencies > low_cutoff_point], atol=1e-5)
        assert np.allclose(0, FILTERED_SIGNAL[np.where(np.logical_and(
            self.frequencies > -high_cutoff_point, self.frequencies < high_cutoff_point))], atol=1e-5)

        # expected to be not zero within the band
        assert not np.allclose(0, FILTERED_SIGNAL[np.where(np.logical_and(
            self.frequencies > -low_cutoff_point, self.frequencies < -high_cutoff_point))], atol=1e-5)
        assert not np.allclose(0, FILTERED_SIGNAL[np.where(np.logical_and(
            self.frequencies > high_cutoff_point, self.frequencies < low_cutoff_point))], atol=1e-5)

        if show_figure_on_screen:
            self.visualize_filtered_spectrum(FILTERED_SIGNAL)

    def test_butter_filter_with_random_signal(self, show_figure_on_screen=False):

        filtered_signal = butter_bandpass_filtering(
            self.noisy_signal, self.time_spacing_in_ms, self.cutoff_lowpass_in_Hz, self.cutoff_highpass_in_Hz, order=9)

        # compute frequency spectrum
        FILTERED_SIGNAL = np.fft.fftshift(np.fft.fft(filtered_signal))

        # expected to be close to zero outside of the band with some tolerance margin
        high_cutoff_point = self.cutoff_highpass_in_Hz*self.time_spacing_in_ms
        low_cutoff_point = self.cutoff_lowpass_in_Hz*self.time_spacing_in_ms
        assert np.allclose(0, FILTERED_SIGNAL[self.frequencies < -low_cutoff_point*1.1], atol=1)
        assert np.allclose(0, FILTERED_SIGNAL[self.frequencies > low_cutoff_point*1.1], atol=1)
        assert np.allclose(0, FILTERED_SIGNAL[np.where(np.logical_and(
            self.frequencies > -high_cutoff_point/2, self.frequencies < high_cutoff_point/2))], atol=1)

        # expected to be not zero within the band
        assert not np.allclose(0, FILTERED_SIGNAL[np.where(np.logical_and(
            self.frequencies > -low_cutoff_point, self.frequencies < high_cutoff_point))], atol=1e-5)
        assert not np.allclose(0, FILTERED_SIGNAL[np.where(np.logical_and(
            self.frequencies > high_cutoff_point, self.frequencies < low_cutoff_point))], atol=1e-5)

        if show_figure_on_screen:
            self.visualize_filtered_spectrum(FILTERED_SIGNAL)


if __name__ == '__main__':
    test = TestBandpassFilter()
    test.setUp()
    test.test_tukey_bandpass_filter(show_figure_on_screen=False)
    test.test_tukey_window_function(show_figure_on_screen=False)
    test.test_butter_bandpass_filter(show_figure_on_screen=False)
    test.test_tukey_filter_with_random_signal(show_figure_on_screen=False)
    test.test_butter_filter_with_random_signal(show_figure_on_screen=False)
