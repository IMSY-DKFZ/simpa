from ippai.simulate import Tags
from ippai.simulate.models.optical_models import mcxyz_adapter
import numpy as np

MODEL_MCXYZ = "mcxyz"


def run_optical_forward_model(settings, optical_properties_path, model=MODEL_MCXYZ):
    # TODO
    optical_output_path = settings[Tags.SIMULATION_PATH] + "/" + settings[Tags.VOLUME_NAME] + "/" + \
                   Tags.OPTICAL_MODEL_OUTPUT_NAME + "_" + \
                   str(settings[Tags.WAVELENGTH]) + ".npz"

    volumes = [None]

    if model == MODEL_MCXYZ:
        volumes = mcxyz_adapter.simulate(optical_properties_path, settings, optical_output_path)

    np.savez(optical_output_path,
             fluence=volumes[0],
             initial_pressure=volumes[1])

    return optical_output_path
