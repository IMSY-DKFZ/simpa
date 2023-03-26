# SPDX-FileCopyrightText: 2021 Division of Intelligent Medical Systems, DKFZ
# SPDX-FileCopyrightText: 2021 Janek Groehl
# SPDX-License-Identifier: MIT

import numpy as np
from simpa.utils import Tags
from simpa.core.processing_components import ProcessingComponent
from simpa.io_handling import load_data_field, save_data_field

class MultiplyEnergy(ProcessingComponent):
    """
    simulate multiplicative noise of p0 coming from energy flucuations of device laser

    Operations:
        - Divide by Tags.LASER_PULSE_ENERGY_IN_MILLIJOULE if given
        - Multiply with Tags.IN_AQUA_LASER_ENERGY_IN_MILLIJOULE

    Component Settings:
       Tags.IN_AQUA_DATA_PATH: path of in-aqua time series data sample containing also the energy
       Tags.IN_AQUA_LASER_ENERGY_IN_MILLIJOULE: laser energy of the corresponding waterbath measurement in mJ
    """

    def run(self, device):
        self.logger.info("Applying Sample Energy Multiplication on Initial Pressure...")
        if Tags.MULTIPLY_ENERGY_ON_PRESSURE_SETTINGS not in self.global_settings:
            self.logger.debug("MultiplyEnergy Settings should be stored in global settings under the tag\
                              Tags.MULTIPLY_ENERGY_ON_PRESSURE_SETTINGS in order to ensure check energy settings\
                              in AddNoisyTimeSeries Component.")

        if Tags.IN_AQUA_DATA_PATH not in self.component_settings:
            self.logger.debug("Tags.IN_AQUA_DATA_PATH should be set for reproducibility.")    

        if Tags.IN_AQUA_LASER_ENERGY_IN_MILLIJOULE in self.component_settings:
            noise_sample_energy = self.component_settings[Tags.IN_AQUA_LASER_ENERGY_IN_MILLIJOULE]
        else:
            msg = f"The field {Tags.IN_AQUA_LASER_ENERGY_IN_MILLIJOULE} must be set in order to use the processing component."
            self.logger.critical(msg)
            raise KeyError(msg)

        # load energy specified in optical simulation settings
        optical_settings = self.global_settings.get_optical_settings()
        if Tags.LASER_PULSE_ENERGY_IN_MILLIJOULE in optical_settings:
            sim_energy = optical_settings[Tags.LASER_PULSE_ENERGY_IN_MILLIJOULE]

            if sim_energy == noise_sample_energy:
                self.logger.info("Same Energy was used already for optical simulation. p0-map is not changed.")
            else:
                # load initial pressure
                wavelength = self.global_settings[Tags.WAVELENGTH]
                p0 = load_data_field(self.global_settings[Tags.SIMPA_OUTPUT_PATH], Tags.DATA_FIELD_INITIAL_PRESSURE, wavelength)
                p0 *= noise_sample_energy/sim_energy
                save_data_field(p0, self.global_settings[Tags.SIMPA_OUTPUT_PATH], Tags.DATA_FIELD_INITIAL_PRESSURE, wavelength)
        
        else:
            msg = f"No laser energy was set for optical simulation.\
                MultiplyEnergy is not meaningful for initial pressure map with non pascal but arbitrary units."
            self.logger.critical(msg)
            raise KeyError(msg)

        self.logger.info("Applying Sample Energy Multiplication on Initial Pressure...[Done]")
    