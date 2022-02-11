# SPDX-FileCopyrightText: 2021 Computer Assisted Medical Interventions Group, DKFZ
# SPDX-FileCopyrightText: 2021 Janek Groehl
# SPDX-License-Identifier: MIT

from .utils import *
from .log import Logger

from .core.simulation_modules.volume_creation_module.volume_creation_module_model_based_adapter import \
    ModelBasedVolumeCreationAdapter
from .core.simulation_modules.volume_creation_module.volume_creation_module_segmentation_based_adapter import \
    SegmentationBasedVolumeCreationAdapter
from .core.simulation_modules.optical_simulation_module.optical_forward_model_mcx_adapter import \
    MCXAdapter
from .core.simulation_modules.optical_simulation_module.optical_forward_model_mcx_reflectance_adapter import \
    MCXAdapterReflectance
from .core.simulation_modules.acoustic_forward_module.acoustic_forward_module_k_wave_adapter import \
    KWaveAdapter
from .core.simulation_modules.reconstruction_module.reconstruction_module_delay_and_sum_adapter import \
    DelayAndSumAdapter
from .core.simulation_modules.reconstruction_module.reconstruction_module_delay_multiply_and_sum_adapter import \
    DelayMultiplyAndSumAdapter
from .core.simulation_modules.reconstruction_module.reconstruction_module_signed_delay_multiply_and_sum_adapter import \
    SignedDelayMultiplyAndSumAdapter
from .core.simulation_modules.reconstruction_module.reconstruction_module_time_reversal_adapter import \
    TimeReversalAdapter

from .core.simulation_modules.reconstruction_module.reconstruction_module_delay_and_sum_adapter import \
    reconstruct_delay_and_sum_pytorch
from .core.simulation_modules.reconstruction_module.reconstruction_module_delay_multiply_and_sum_adapter import \
    reconstruct_delay_multiply_and_sum_pytorch
from .core.simulation_modules.reconstruction_module.reconstruction_module_signed_delay_multiply_and_sum_adapter import \
    reconstruct_signed_delay_multiply_and_sum_pytorch
from .core.simulation_modules.acoustic_forward_module.acoustic_forward_module_k_wave_adapter import \
    perform_k_wave_acoustic_forward_simulation

from simpa.core.processing_components.monospectral.noise import GaussianNoise
from simpa.core.processing_components.monospectral.noise import GammaNoise
from simpa.core.processing_components.monospectral.noise import PoissonNoise
from simpa.core.processing_components.monospectral.noise import SaltAndPepperNoise
from simpa.core.processing_components.monospectral.noise import UniformNoise
from simpa.core.processing_components.monospectral.field_of_view_cropping import FieldOfViewCropping
from simpa.core.processing_components.monospectral.iterative_qPAI_algorithm import IterativeqPAI
from simpa.core.processing_components.multispectral.linear_unmixing import LinearUnmixing

from .core.device_digital_twins import *

from .core.simulation import simulate

from .io_handling import load_data_field, load_hdf5, save_data_field, save_hdf5
from .io_handling.zenodo_download import download_from_zenodo
from .io_handling.ipasc import export_to_ipasc

from .visualisation.matplotlib_data_visualisation import visualise_data
from .visualisation.matplotlib_device_visualisation import visualise_device

from .utils.quality_assurance.data_sanity_testing import assert_equal_shapes
from .utils.quality_assurance.data_sanity_testing import assert_array_well_defined
