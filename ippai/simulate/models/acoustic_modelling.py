from ippai.simulate import Tags
import numpy as np
from ippai.simulate.models.acoustic_models import k_wave_adapter
import os


def run_acoustic_forward_model(settings, optical_path):
    print("ACOUSTIC FORWARD")

    acoustic_path = settings[Tags.SIMULATION_PATH] + "/" + settings[Tags.VOLUME_NAME] + "/" + \
                    Tags.ACOUSTIC_MODEL_OUTPUT_NAME + ".npz"

    data = k_wave_adapter.simulate(settings, optical_path)

    np.savez(acoustic_path,
             time_series_data=data)

    return acoustic_path
