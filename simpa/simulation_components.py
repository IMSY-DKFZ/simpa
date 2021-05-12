# The MIT License (MIT)
#
# Copyright (c) 2021 Computer Assisted Medical Interventions Group, DKFZ
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated simpa_documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


###################################################
# MODULE ADAPTERS
###################################################

# Tissue Generation
from simpa.core.volume_creation_module.volume_creation_module_segmentation_based_adapter import VolumeCreationModuleSegmentationBasedAdapter
from simpa.core.volume_creation_module.volume_creation_module_model_based_adapter import VolumeCreationModelModelBasedAdapter

# Optical forward modelling
from simpa.core.optical_simulation_module.optical_forward_model_mcx_adapter import OpticalForwardModelMcxAdapter

# Acoustic forward modelling
from simpa.core.acoustic_forward_module.acoustic_forward_module_k_wave_adapter import AcousticForwardModelKWaveAdapter

# Image reconstruction
from simpa.core.reconstruction_module.reconstruction_module_delay_and_sum_adapter import ImageReconstructionModuleDelayAndSumAdapter
from simpa.core.reconstruction_module.reconstruction_module_time_reversal_adapter import ReconstructionModuleTimeReversalAdapter

###################################################
# PROCESSING COMPONENTS
###################################################

from simpa.processing.noise_processing_components import GaussianNoiseProcessingComponent
from simpa.processing.field_of_view_cropping import FieldOfViewCroppingProcessingComponent