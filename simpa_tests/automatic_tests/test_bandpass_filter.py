# SPDX-FileCopyrightText: 2021 Division of Intelligent Medical Systems, DKFZ
# SPDX-FileCopyrightText: 2021 Janek Groehl
# SPDX-License-Identifier: MIT

import unittest
import numpy as np
import matplotlib.pyplot as plt
from simpa.core.simulation_modules.reconstruction_module.reconstruction_utils import tukey_bandpass_filtering_with_settings, tukey_bandpass_filtering, butter_bandpass_filtering, butter_bandpass_filtering_with_settings
from simpa.utils import Settings, Tags
from simpa.core.device_digital_twins import LinearArrayDetectionGeometry

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

        ## for tests with random signal ## 

        # generate random noisy signal
        random_t_values = np.arange(0, 10, 0.01)
        np.random.seed(4117)
        noisy_frequency = np.random.rand(len(random_t_values))
        self.noisy_signal = np.sin(2 * np.pi * noisy_frequency * random_t_values)
        
        # filter signal
        self.cutoff_highpass = 40000
        self.cutoff_lowpass = 400000
        self.time_spacing = 1e-3
        self.frequencies = np.fft.fftshift(np.fft.fftfreq(len(random_t_values), self.time_spacing))

    def test_butter_bandpass_filter(self, show_figure_on_screen=False):
        self.settings.get_reconstruction_settings()[Tags.BANDPASS_FILTER_METHOD] = Tags.BUTTERWORTH_BANDPASS_FILTER
        self.settings.get_reconstruction_settings()[Tags.BUTTERWORTH_FILTER_ORDER] = 1
        filtered_time_series_with_settings = butter_bandpass_filtering_with_settings(self.combined_time_series, self.settings, self.settings[Tags.RECONSTRUCTION_MODEL_SETTINGS], self.device)
        
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

    def test_tukey_bandpass_filter(self, show_figure_on_screen=False):
        filtered_time_series_with_settings = tukey_bandpass_filtering_with_settings(self.combined_time_series, self.settings,
                                                                                    self.settings[Tags.RECONSTRUCTION_MODEL_SETTINGS],
                                                                                    self.device)

        filtered_time_series_with_resampling = tukey_bandpass_filtering(self.combined_time_series, 1e-3, int(11000), int(9000), 0, True)

        filtered_time_series = tukey_bandpass_filtering(self.combined_time_series, 1e-3, int(11000), int(9000), 0)

        # check if both bandpass filtering methods return the same result
        assert np.array_equal(filtered_time_series, filtered_time_series_with_settings)

        assert (np.abs(self.base_time_series - filtered_time_series) < 1e-5).all()

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
        
        def plot_region(low_limit: int, high_limit:int, label:str = '', color:str = 'red'):
            interval = np.arange(low_limit, high_limit, self.time_spacing)
            plt.plot(interval, np.zeros_like(interval), label=label, linewidth=5, color=color )
        
        # passing intervals
        plot_region(self.cutoff_highpass*self.time_spacing, self.cutoff_lowpass*self.time_spacing, \
             label="bandwidth of passed filter",color="green")
        plot_region(-self.cutoff_lowpass*self.time_spacing, -self.cutoff_highpass*self.time_spacing,color="green")

        # filtered intervals
        plot_region(-self.cutoff_highpass*self.time_spacing, self.cutoff_highpass*self.time_spacing, \
            label="signal should be close to zero",color="red")
        plot_region(self.frequencies[0], -self.cutoff_lowpass*self.time_spacing)
        plot_region(self.cutoff_lowpass*self.time_spacing, self.frequencies[-1])

        plt.plot(self.frequencies, filtered_spectrum, label="filtered signal")
        plt.xlabel("frequency")
        plt.legend()
        plt.show()

    def test_tukey_filter_with_random_signal(self, show_figure_on_screen=False):

        filtered_signal = tukey_bandpass_filtering(self.noisy_signal, self.time_spacing, self.cutoff_lowpass, self.cutoff_highpass, tukey_alpha=0.5)

        # compute frequency spectrum
        FILTERED_SIGNAL = np.fft.fftshift(np.fft.fft(filtered_signal))

        # expected to be close to zero outside of the band
        high_cutoff_point = self.cutoff_highpass*self.time_spacing
        low_cutoff_point = self.cutoff_lowpass*self.time_spacing
        assert np.allclose(0, FILTERED_SIGNAL[self.frequencies < -low_cutoff_point], atol=1e-5)
        assert np.allclose(0, FILTERED_SIGNAL[self.frequencies > low_cutoff_point], atol=1e-5)
        assert np.allclose(0, FILTERED_SIGNAL[np.where(np.logical_and(self.frequencies>-high_cutoff_point, self.frequencies<high_cutoff_point))], atol=1e-5)
    
        # expected to be not zero within the band
        assert not np.allclose(0, FILTERED_SIGNAL[np.where(np.logical_and(self.frequencies>-low_cutoff_point, self.frequencies<-high_cutoff_point))], atol=1e-5)
        assert not np.allclose(0, FILTERED_SIGNAL[np.where(np.logical_and(self.frequencies>high_cutoff_point, self.frequencies<low_cutoff_point))], atol=1e-5)

        if show_figure_on_screen:
            self.visualize_filtered_spectrum(FILTERED_SIGNAL) 

    def test_butter_filter_with_random_signal(self, show_figure_on_screen=False):

        filtered_signal = butter_bandpass_filtering(self.noisy_signal, self.time_spacing, self.cutoff_lowpass, self.cutoff_highpass, order=9)

        # compute frequency spectrum
        FILTERED_SIGNAL = np.fft.fftshift(np.fft.fft(filtered_signal))

        # expected to be close to zero outside of the band with some tolerance margin
        high_cutoff_point = self.cutoff_highpass*self.time_spacing
        low_cutoff_point = self.cutoff_lowpass*self.time_spacing
        
        assert np.allclose(0, FILTERED_SIGNAL[self.frequencies < -low_cutoff_point*1.1], atol=1)
        assert np.allclose(0, FILTERED_SIGNAL[self.frequencies > low_cutoff_point*1.1], atol=1)
        assert np.allclose(0, FILTERED_SIGNAL[np.where(np.logical_and(self.frequencies>-high_cutoff_point/2, self.frequencies<high_cutoff_point/2))], atol=1)
    
        # expected to be not zero within the band
        assert not np.allclose(0, FILTERED_SIGNAL[np.where(np.logical_and(self.frequencies>-low_cutoff_point, self.frequencies < high_cutoff_point))], atol=1e-5)
        assert not np.allclose(0, FILTERED_SIGNAL[np.where(np.logical_and(self.frequencies>high_cutoff_point, self.frequencies<low_cutoff_point))], atol=1e-5)
        
        if show_figure_on_screen:
            self.visualize_filtered_spectrum(FILTERED_SIGNAL)


if __name__ == '__main__':
    test = TestBandpassFilter()
    test.setUp()
    test.test_tukey_bandpass_filter(show_figure_on_screen=False)
    test.test_butter_bandpass_filter(show_figure_on_screen=False)
    test.test_tukey_filter_with_random_signal(show_figure_on_screen=False)
    test.test_butter_filter_with_random_signal(show_figure_on_screen=False)