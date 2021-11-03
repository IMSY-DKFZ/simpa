"""
SPDX-FileCopyrightText: 2021 Computer Assisted Medical Interventions Group, DKFZ
SPDX-FileCopyrightText: 2021 VISION Lab, Cancer Research UK Cambridge Institute (CRUK CI)
SPDX-License-Identifier: MIT
"""

from .log import Logger
from .utils import *


from .core.simulation_modules.volume_creation_module.volume_creation_module_model_based_adapter import \
    VolumeCreationModelModelBasedAdapter
from .core.simulation_modules.volume_creation_module.volume_creation_module_segmentation_based_adapter import \
    VolumeCreationModuleSegmentationBasedAdapter
from .core.simulation_modules.optical_simulation_module.optical_forward_model_mcx_adapter import \
    OpticalForwardModelMcxAdapter
from .core.simulation_modules.acoustic_forward_module.acoustic_forward_module_k_wave_adapter import \
    AcousticForwardModelKWaveAdapter
from .core.simulation_modules.reconstruction_module.reconstruction_module_delay_and_sum_adapter import \
    ImageReconstructionModuleDelayAndSumAdapter
from .core.simulation_modules.reconstruction_module.reconstruction_module_delay_multiply_and_sum_adapter import \
    ImageReconstructionModuleDelayMultiplyAndSumAdapter
from .core.simulation_modules.reconstruction_module.reconstruction_module_signed_delay_multiply_and_sum_adapter import \
    ImageReconstructionModuleSignedDelayMultiplyAndSumAdapter
from .core.simulation_modules.reconstruction_module.reconstruction_module_time_reversal_adapter import \
    ReconstructionModuleTimeReversalAdapter

from .core.simulation_modules.reconstruction_module.reconstruction_module_delay_and_sum_adapter import \
    reconstruct_delay_and_sum_pytorch
from .core.simulation_modules.reconstruction_module.reconstruction_module_delay_multiply_and_sum_adapter import \
    reconstruct_delay_multiply_and_sum_pytorch
from .core.simulation_modules.reconstruction_module.reconstruction_module_signed_delay_multiply_and_sum_adapter import \
    reconstruct_signed_delay_multiply_and_sum_pytorch
from .core.simulation_modules.acoustic_forward_module.acoustic_forward_module_k_wave_adapter import \
    perform_k_wave_acoustic_forward_simulation

from simpa.core.processing_components.monospectral.noise import GaussianNoiseProcessingComponent
from simpa.core.processing_components.monospectral.noise import GammaNoiseProcessingComponent
from simpa.core.processing_components.monospectral.noise import PoissonNoiseProcessingComponent
from simpa.core.processing_components.monospectral.noise import SaltAndPepperNoiseProcessingComponent
from simpa.core.processing_components.monospectral.noise import UniformNoiseProcessingComponent
from simpa.core.processing_components.monospectral.field_of_view_cropping import FieldOfViewCroppingProcessingComponent
from simpa.core.processing_components.monospectral.iterative_qPAI_algorithm import IterativeqPAIProcessingComponent
from simpa.core.processing_components.multispectral.linear_unmixing import LinearUnmixingProcessingComponent

from .core.device_digital_twins import *

from .core.simulation import simulate

from .io_handling import load_data_field, load_hdf5, save_data_field, save_hdf5
from .io_handling.zenodo_download import download_from_zenodo
from .io_handling.ipasc import export_to_ipasc

from .visualisation.matplotlib_data_visualisation import visualise_data
