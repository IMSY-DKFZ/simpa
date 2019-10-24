from ippai.simulate import Tags
from ippai.simulate.models.reconstruction_models.MitkBeamformingAdapter import MitkBeamformingAdapter
import numpy as np


def perform_reconstruction(settings, acoustic_data_path):
    print("ACOUSTIC FORWARD")

    reconstructed_data_save_path = (settings[Tags.SIMULATION_PATH] + "/" + settings[Tags.VOLUME_NAME] + "/" +
                                    Tags.RECONSTRUCTION_OUTPUT_NAME + "_" + str(settings[Tags.WAVELENGTH]) + ".npz")

    reconstruction_method = None

    if settings[Tags.RECONSTRUCTION_ALGORITHM] == Tags.RECONSTRUCTION_ALGORITHM_DAS:
        reconstruction_method = MitkBeamformingAdapter()

    reconstruction = reconstruction_method.simulate(settings, acoustic_data_path)

    np.savez(reconstructed_data_save_path,
             reconstruction=reconstruction)

    return reconstructed_data_save_path
