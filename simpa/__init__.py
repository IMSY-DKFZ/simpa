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

from .core.processing_components.field_of_view_cropping import FieldOfViewCroppingProcessingComponent
from .core.processing_components.noise.gamma_noise import GammaNoiseProcessingComponent
from .core.processing_components.noise.gaussian_noise import GaussianNoiseProcessingComponent
from .core.processing_components.noise.poisson_noise import PoissonNoiseProcessingComponent
from .core.processing_components.noise.salt_and_pepper_noise import SaltAndPepperNoiseProcessingComponent
from .core.processing_components.noise.uniform_noise import UniformNoiseProcessingComponent

from .core.device_digital_twins import *

from .core.simulation import simulate

from .visualisation.matplotlib_data_visualisation import visualise_data
