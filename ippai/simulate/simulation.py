from ippai.simulate import Tags
from ippai.simulate.volume_creator import create_simulation_volume
from ippai.simulate.models.optical_model import run_optical_forward_model
from ippai.simulate.models.acoustic_model import run_acoustic_forward_model
import numpy as np
import os


def simulate(settings):
    """

    :param settings:
    :return:
    """

    wavelengths = settings[Tags.WAVELENGTHS]
    volume_paths = []
    optical_paths = []
    acoustic_paths = []

    path = settings[Tags.SIMULATION_PATH] + "/" + settings[Tags.VOLUME_NAME] + "/"
    if not os.path.exists(path):
        os.mkdir(path)

    np.savez(path + "settings.npz",
             settings=settings)

    for wavelength in wavelengths:

        if settings[Tags.RANDOM_SEED] is not None:
            np.random.seed(settings[Tags.RANDOM_SEED])
        else:
            np.random.seed()

        settings[Tags.WAVELENGTH] = wavelength
        volume_path = create_simulation_volume(settings)
        volume_paths.append(volume_path)

        optical_path = None

        if settings[Tags.RUN_OPTICAL_MODEL]:
            optical_path = run_optical_forward_model(settings, volume_path)
            optical_paths.append(optical_path)

        if settings[Tags.RUN_ACOUSTIC_MODEL]:
            acoustic_path = run_acoustic_forward_model(settings, optical_path)
            acoustic_paths.append(acoustic_path)

    return [volume_paths, optical_paths, acoustic_paths]
