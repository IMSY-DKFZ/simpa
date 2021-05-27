"""
SPDX-FileCopyrightText: 2021 Computer Assisted Medical Interventions Group, DKFZ
SPDX-FileCopyrightText: 2021 VISION Lab, Cancer Research UK Cambridge Institute (CRUK CI)
SPDX-License-Identifier: MIT
"""
from abc import abstractmethod

from simpa.core.device_digital_twins import DigitalDeviceTwinBase
from simpa.log import Logger
from simpa.utils import Settings


class SimulationModule:
    """
    Defines a simulation module that is callable via the SIMPA core.simulation.simulate method.
    """

    def __init__(self, global_settings: Settings):
        """
        Initialises the SimulationModule given the global settings dictionary.
         :param global_settings: The SIMPA settings dictionary
        """
        self.logger = Logger()
        self.global_settings = global_settings

    @abstractmethod
    def run(self, digital_device_twin: DigitalDeviceTwinBase):
        """
        Executes the respective simulation module

        :param digital_device_twin: The digital twin that can be used by the digital device_twin.
        """
        pass


from .volume_creation_module.volume_creation_module_model_based_adapter import VolumeCreationModelModelBasedAdapter
from .volume_creation_module.volume_creation_module_segmentation_based_adapter import \
    VolumeCreationModuleSegmentationBasedAdapter

from .acoustic_forward_module.acoustic_forward_module_k_wave_adapter import AcousticForwardModelKWaveAdapter

from .optical_simulation_module.optical_forward_model_mcx_adapter import OpticalForwardModelMcxAdapter

from .reconstruction_module.reconstruction_module_time_reversal_adapter import ReconstructionModuleTimeReversalAdapter
from .reconstruction_module.reconstruction_module_delay_and_sum_adapter import \
    ImageReconstructionModuleDelayAndSumAdapter
from .reconstruction_module.reconstruction_module_delay_multiply_and_sum_adapter import \
    ImageReconstructionModuleDelayMultiplyAndSumAdapter
from .reconstruction_module.reconstruction_module_signed_delay_multiply_and_sum_adapter import \
    ImageReconstructionModuleSignedDelayMultiplyAndSumAdapter

from .processing_components import *
