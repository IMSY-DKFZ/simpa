from ippai.simulate import Tags
import numpy as np
from ippai.simulate.models.acoustic_models import k_wave_adapter
from ippai.io_handling.io_hdf5 import save_hdf5
import os


def run_acoustic_forward_model(settings, optical_path):
    print("ACOUSTIC FORWARD")

    acoustic_path = (settings[Tags.SIMULATION_PATH] + "/" + settings[Tags.VOLUME_NAME] + "/" +
                     Tags.ACOUSTIC_MODEL_OUTPUT_NAME + "_" + str(settings[Tags.WAVELENGTH]) + ".npz")

    data = k_wave_adapter.simulate(settings, optical_path)

    save_hdf5({"time_series_data": data}, settings[Tags.IPPAI_OUTPUT_PATH],
              "/simulations/upsampled/acoustic_output/")

    return "/simulations/upsampled/acoustic_output/"
