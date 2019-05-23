from ippai.simulate import Tags
import numpy as np


def run_acoustic_forward_model(simulation_folder, settings):
    # TODO
    acoustic_path = settings[Tags.SIMULATION_PATH] + "/" + settings[Tags.VOLUME_NAME] + "/" + \
                    Tags.ACOUSTIC_MODEL_OUTPUT_NAME + "_" + \
                    str(settings[Tags.WAVELENGTH]) + "nm.npz"

    volumes = [None]

    np.savez(acoustic_path,
             fluence=volumes[0])

    return acoustic_path
