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


class SensorEfficiencies(ProcessingComponent):
    """
    Applies degradation techniques (like broken sensors or sensors with less sensitivity) to the time series data.
    Component Settings::
       Tags.DEGRADATED_SENSORS: (default: all)
       Tags.DEGRADATION_FACTORS: (default: 0.1)
       Tags.SHIFT_TS: TODO Probably delete
    """

    def run(self, device) -> None:
        self.logger.info("Applying Sensor Degradation on Time Series Data...")

        # read out time series data
        wavelength = self.global_settings[Tags.WAVELENGTH]
        time_series_data = load_data_field(self.global_settings[Tags.SIMPA_OUTPUT_PATH], Tags.DATA_FIELD_TIME_SERIES_DATA, wavelength)
        
        n_sensors, _ = time_series_data.shape

        # check input
        degradated_sensors, degradation_factors = self.get_indices_and_factors(n_sensors)
        
        # do degradation operations
        self.logger.debug(f"Degrade sensor:\n{degradated_sensors}\nwith factors:\n{degradation_factors}")
        time_series_data_old = time_series_data.copy()#TODO:delete
        # TODO: SPEAK WITH OTHERS
        # looking into real time series data: the data has minimal values of 2000 but simulation only 0.3 or similiar-->
        # add this
        if Tags.SHIFT_TS in self.component_settings.keys():
            time_series_data += self.component_settings[Tags.SHIFT_TS]
        time_series_data[degradated_sensors,:] *= degradation_factors[:, None]

        _, axes = plt.subplots(1,2)
        factor = 0.8
        time_limit = int(factor*time_series_data.shape[1])
        axes[0].imshow(time_series_data_old[:,:time_limit], aspect="auto")
        axes[1].imshow(time_series_data[:,:time_limit], aspect="auto")
        plt.show()


        if not (Tags.IGNORE_QA_ASSERTIONS in self.global_settings and Tags.IGNORE_QA_ASSERTIONS):
            assert_array_well_defined(time_series_data)

        # overwrite the time series data
        save_data_field(time_series_data, self.global_settings[Tags.SIMPA_OUTPUT_PATH], Tags.DATA_FIELD_TIME_SERIES_DATA, wavelength)
        self.logger.info("Applying Sensor Degradation on Time Series Data...[Done]")

    def get_indices_and_factors(self, n_sensors: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Given the Component Settings::
            Tags.DEGRADATED_SENSORS: (default: all)
            Tags.DEGRADATION_FACTORS: (default: uniform(0.9,1.1))
        and the number of total detector sensor elements, 
        return the indices of the degradated sensors and the corresponding factors
        """

        if Tags.DEGRADATED_SENSORS in self.component_settings.keys():
            degradated_sensors = self.component_settings[Tags.DEGRADATED_SENSORS]
            if isinstance(degradated_sensors, (list, np.ndarray)):
                pass
            elif isinstance(degradated_sensors, int):
                degradated_sensors = np.random.choice(n_sensors, size=degradated_sensors)
            elif isinstance(degradated_sensors, float):
                degradated_sensors = np.random.choice(n_sensors, size=int(degradated_sensors*n_sensors))
            else:
                self.logger.critical("Wrong datatype for Tags.DEGRADATED_SENSORS.")

            degradation_factors = np.ones(len(degradated_sensors))
            if Tags.DEGRADATION_FACTORS in self.component_settings.keys() and \
                isinstance(self.component_settings[Tags.DEGRADATION_FACTORS], (float, list, np.ndarray)):
                degradation_factors *= self.component_settings[Tags.DEGRADATION_FACTORS]
            else:
                degradation_factors *= np.random.rand(len(degradated_sensors))*0.2+0.9         
        else:
            self.logger.debug("Whether number or ratio or indices of degradated sensors were given.")
            if Tags.DEGRADATION_FACTORS in self.component_settings.keys():
                if isinstance(self.component_settings[Tags.DEGRADATION_FACTORS], (list, np.ndarray)):
                    degradated_sensors = np.random.choice(n_sensors,
                                                        size=len(self.component_settings[Tags.DEGRADATION_FACTORS])
                                                        )
                    degradation_factors = np.ones(len(degradated_sensors))*self.component_settings[Tags.DEGRADATION_FACTORS]
                elif isinstance(self.component_settings[Tags.DEGRADATION_FACTORS], float):
                    self.logger.debug(f"Change efficiencies of all sensors\
                        with factor {self.component_settings[Tags.DEGRADATION_FACTORS]}.")
                    degradated_sensors = np.random.choice(n_sensors, size=n_sensors)
                    degradation_factors = np.ones(n_sensors)*self.component_settings[Tags.DEGRADATION_FACTORS]
            else:
                self.logger.debug("Change efficiencies of all sensors")
                degradated_sensors = np.random.choice(n_sensors, size=n_sensors)
                degradation_factors = np.ones(n_sensors)*np.random.rand(len(degradated_sensors))*0.2+0.9 
        return degradated_sensors, np.asarray(degradation_factors)