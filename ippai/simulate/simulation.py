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
        os.makedirs(path)

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

        optical_path=None
        acoustic_path = None

        if settings[Tags.RUN_OPTICAL_MODEL]:
            optical_path = run_optical_forward_model(settings, volume_path)
            optical_paths.append(optical_path)

        if settings[Tags.RUN_ACOUSTIC_MODEL]:
            acoustic_path = run_acoustic_forward_model(settings, optical_path)
            acoustic_paths.append(acoustic_path)

        if Tags.SIMULATION_EXTRACT_FIELD_OF_VIEW in settings:
            if settings[Tags.SIMULATION_EXTRACT_FIELD_OF_VIEW]:
                extract_field_of_view(volume_path, optical_path, acoustic_path)

    return [volume_paths, optical_paths, acoustic_paths]


def extract_field_of_view(volume_path, optical_path, acoustic_path):
    if volume_path is not None:
        volume_data = np.load(volume_path)
        mua = volume_data['mua']
        sizes = np.shape(mua)
        mua = mua[:, int(sizes[1]/2), :]
        mus = volume_data['mus'][:, int(sizes[1]/2), :]
        g = volume_data['g'][:, int(sizes[1] / 2), :]
        oxy = volume_data['oxy'][:, int(sizes[1] / 2), :]
        seg = volume_data['seg'][:, int(sizes[1] / 2), :]
        np.savez(volume_path,
                 mua=mua,
                 mus=mus,
                 g=g,
                 oxy=oxy,
                 seg=seg)

    if optical_path is not None:
        optical_data = np.load(optical_path)
        fluence = optical_data['fluence']
        sizes = np.shape(fluence)
        fluence = fluence[:, int(sizes[1] / 2), :]
        initial_pressure = optical_data['initial_pressure'][:, int(sizes[1] / 2), :]
        np.savez(optical_path,
                 fluence=fluence,
                 initial_pressure=initial_pressure)

    if acoustic_path is not None:
        acoustic_data = np.load(acoustic_path)
