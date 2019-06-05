from ippai.simulate import Tags
from ippai.simulate.models.optical_models import mcxyz_adapter, mxc_adapter
import numpy as np

def run_optical_forward_model(settings, optical_properties_path):
    # TODO
    print("OPTICAL FORWARD")
    optical_output_path = settings[Tags.SIMULATION_PATH] + "/" + settings[Tags.VOLUME_NAME] + "/" + \
                   Tags.OPTICAL_MODEL_OUTPUT_NAME + "_" + \
                   str(settings[Tags.WAVELENGTH]) + ".npz"

    volumes = [None]

    if Tags.OPTICAL_MODEL not in settings:
        raise AssertionError("Tags.OPTICAL_MODEL tag was not specified in the settings. Skipping optical modelling.")

    model = settings[Tags.OPTICAL_MODEL]

    if model == Tags.MODEL_MCXYZ:
        volumes = mcxyz_adapter.simulate(optical_properties_path, settings, optical_output_path)
    if model == Tags.MODEL_MCX:
        volumes = mxc_adapter.simulate(optical_properties_path, settings, optical_output_path)

    np.savez(optical_output_path,
             fluence=volumes[0],
             initial_pressure=volumes[1])

    return optical_output_path
