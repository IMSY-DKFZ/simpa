# The MIT License (MIT)
#
# Copyright (c) 2018 Computer Assisted Medical Interventions Group, DKFZ
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
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

from simulate.models.acoustic_models.acoustic_modelling import run_acoustic_forward_model
from simulate.models.acoustic_models import AcousticForwardAdapterBase
from simulate.models.acoustic_models.k_wave_adapter import simulate as kWaveAdapter

from simulate.models.optical_models.optical_modelling import run_optical_forward_model
from simulate.models.optical_models import OpticalForwardAdapterBase
from simulate.models.optical_models.mcxyz_adapter import McxyzAdapter
from simulate.models.optical_models.mcx_adapter import McxAdapter

from simulate.models.noise_models.noise_modelling import apply_noise_model_to_reconstructed_data
from simulate.models.noise_models.noise_modelling import apply_noise_model_to_time_series_data
from simulate.models.noise_models import NoiseModelAdapterBase
from simulate.models.noise_models import GaussianNoiseModel

from simulate.models.reconstruction_models.reconstruction_modelling import perform_reconstruction
from simulate.models.reconstruction_models import ReconstructionAdapterBase
from simulate.models.reconstruction_models.MitkBeamformingAdapter import MitkBeamformingAdapter
