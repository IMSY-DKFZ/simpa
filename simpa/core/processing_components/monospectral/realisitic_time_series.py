# SPDX-FileCopyrightText: 2021 Division of Intelligent Medical Systems, DKFZ
# SPDX-FileCopyrightText: 2021 Janek Groehl
# SPDX-License-Identifier: MIT

from simpa.utils import Tags
from simpa.io_handling import load_data_field, save_data_field
from simpa.core.processing_components import ProcessingComponent
from simpa.utils.quality_assurance.data_sanity_testing import assert_array_well_defined
import numpy as np
from typing import Tuple


# TODO: Delete 
import matplotlib.pyplot as plt

# TODO: GENERALIZE
# CURRENT RESTRICTIONS:
# - monospectral normalization (does not include spectral information)
# - membrane base normalization (will not work for )


class SensorDegradation(ProcessingComponent):
    """
    Transform the time series data into real measurement domain
    - laser correct and norm simulated data
    - if TAGS.USE_IN_VITRO_DATA is true: 
        multiply with scaling factor (e.g. in-vitro membrane peak) and laser energy of in-vitro data

    Applies the following degradation techniques to the time series data:
    - broken sensors: sensors that do not record tissue signal but only dark current and thermal noise
    - offsets: each sensor has a specific offset (probably coming from dark current)
    - thermal noise: there is Gaussian noise over time for each sensor

    Component Settings:
       Tags.TRANSFORM_TO_IN_VITRO_DOMAIN
       Tags.SCALING_FACTOR (for example membrane peak of in vitro waterbath data)
       TAGS.IN_VITRO_LASER_ENERGY
       Tags.BROKEN_SENSORS: (for example: np.array([30,94,145]))
       Tags.OFFSETS: (for example: np.ones(256)*2300)
       Tags.THERMAL_NOISES: (default: np.ones(256)*2.5)
    """

    def norm_time_series(self, time_series: np.ndarray, divide_by_maximum: bool = True):
        """
        Apply Laser energy correction and normalize time series using the membrane peak
        """
        # TODO: GENERALIZE THIS FOR MULTISPECTRAL DATA AND TISSUE RELATED SIGNALS (NOT MEMBRANE BASED NORMALIZATION)
        energy = self.global_settings.get_optical_settings()[Tags.LASER_PULSE_ENERGY_IN_MILLIJOULE]

        # laser energy correction
        time_series /= energy

        # norm (such that maximum value is 1) [ASSUME MEMBRANE PEAK NORMALIZATION]
        if divide_by_maximum:
            time_series /= time_series.max()
        return time_series

    def run(self, device) -> None:
        self.logger.info("Applying Sensor Degradation on Time Series Data...")

        # read out time series data
        wavelength = self.global_settings[Tags.WAVELENGTH]
        time_series_data = load_data_field(self.global_settings[Tags.SIMPA_OUTPUT_PATH], Tags.DATA_FIELD_TIME_SERIES_DATA, wavelength)
        
        # 
        self.check_input(time_series_data)

        ##### TODO delete:
        #_, axes = plt.subplots(2, 1)
        #factor = 1#0.8
        #time_limit = int(factor*time_series_data.shape[1])
        #axes[0].imshow(time_series_data[:,:time_limit], aspect="auto")

        if self.component_settings[Tags.TRANSFORM_TO_IN_VITRO_DOMAIN]:
            # laser correct and norm time series data
            time_series_data= self.norm_time_series(time_series_data)

            # transform to in-vitro domain
            time_series_data *= self.component_settings[Tags.SCALING_FACTOR] # scale peaks to in-vitro peaks
            time_series_data *= self.component_settings[Tags.IN_VITRO_LASER_ENERGY] # use in-vitro laser energy
        
        else:
            # laser correct
            time_series_data = self.norm_time_series(time_series_data, divide_by_maximum=False)

        # add noise components, i.e. broken sensors, sensor offsets, sensor thermal noises
        time_series_data = time_series_data*self.working_sensors[:,None] + \
            self.component_settings[Tags.OFFSETS][:,None] + \
            np.random.normal(size=time_series_data.shape) * self.component_settings[Tags.THERMAL_NOISES][:,None]

        # TODO delete
        #axes[1].imshow(time_series_data[:,:time_limit], aspect="auto")
        #plt.show()

        if not (Tags.IGNORE_QA_ASSERTIONS in self.global_settings and Tags.IGNORE_QA_ASSERTIONS):
            assert_array_well_defined(time_series_data)

        # overwrite the time series data
        save_data_field(time_series_data, self.global_settings[Tags.SIMPA_OUTPUT_PATH], Tags.DATA_FIELD_TIME_SERIES_DATA, wavelength)
        self.logger.info("Applying Sensor Degradation on Time Series Data...[Done]")
    
    def check_input(self, time_series_data: np.ndarray):
        (n_sensors, _) = time_series_data.shape

        # indicator function for non-broken sensors
        self.working_sensors = np.ones(n_sensors) 
        self.working_sensors[self.component_settings[Tags.BROKEN_SENSORS]] = 0

        # convert float offset to an array
        if isinstance(self.component_settings[Tags.OFFSETS], (float, list)):
            self.component_settings[Tags.OFFSETS] = np.array(self.component_settings[Tags.OFFSETS])

        # convert float thermal noise to an array
        if isinstance(self.component_settings[Tags.OFFSETS], (float, list)):
            self.component_settings[Tags.THERMAL_NOISES] = np.array(self.component_settings[Tags.THERMAL_NOISES])