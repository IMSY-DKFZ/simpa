from ippai.simulate import Tags
from ippai.simulate.volume_creator import create_simulation_volume
from ippai.simulate.models.optical_modelling import run_optical_forward_model
from ippai.simulate.models.acoustic_modelling import run_acoustic_forward_model
from ippai.simulate.models.noise_modelling import apply_noise_model_to_time_series_data
from ippai.simulate.models.reconstruction import perform_reconstruction
from ippai.process.sampling import upsample
import numpy as np
import os


def simulate(settings):
    """

    :param settings:
    :return:
    """

    wavelengths = settings[Tags.WAVELENGTHS]
    volume_output_paths = []
    optical_output_paths = []
    acoustic_output_paths = []
    reconstruction_output_paths = []

    path = settings[Tags.SIMULATION_PATH] + "/" + settings[Tags.VOLUME_NAME] + "/"
    if not os.path.exists(path):
        os.makedirs(path)

    np.savez(path + "settings.npz",
             settings=settings)

    for wavelength in wavelengths:

        if settings[Tags.RANDOM_SEED] is not None:
            np.random.seed(settings[Tags.RANDOM_SEED])
        else:
            np.random.seed(None)

        settings[Tags.WAVELENGTH] = wavelength
        volume_output_path = create_simulation_volume(settings)
        volume_output_paths.append(volume_output_path)

        optical_output_path = None
        acoustic_output_path = None

        if settings[Tags.RUN_OPTICAL_MODEL]:
            optical_output_path = run_optical_forward_model(settings, volume_output_path)
            optical_output_paths.append(optical_output_path)

        if Tags.SIMULATION_EXTRACT_FIELD_OF_VIEW in settings:
            if settings[Tags.SIMULATION_EXTRACT_FIELD_OF_VIEW]:
                extract_field_of_view(volume_output_path, optical_output_path, acoustic_output_path)

        if Tags.PERFORM_UPSAMPLING in settings:
            if settings[Tags.PERFORM_UPSAMPLING]:
                optical_output_path = upsample(settings, optical_output_path)
                optical_output_paths.append(optical_output_path)

        if settings[Tags.RUN_ACOUSTIC_MODEL]:
            acoustic_output_path = run_acoustic_forward_model(settings, optical_output_path)
            if (Tags.APPLY_NOISE_MODEL in settings) and settings[Tags.APPLY_NOISE_MODEL]:
                acoustic_output_path = apply_noise_model_to_time_series_data(settings, acoustic_output_path)
            acoustic_output_paths.append(acoustic_output_path)

        if Tags.PERFORM_IMAGE_RECONSTRUCTION in settings:
            if settings[Tags.PERFORM_IMAGE_RECONSTRUCTION]:
                reconstruction_output_path = perform_reconstruction(settings, acoustic_output_path)
                reconstruction_output_paths.append(reconstruction_output_path)

    return [volume_output_paths, optical_output_paths, acoustic_output_paths, reconstruction_output_paths]


def extract_field_of_view(volume_path, optical_path, acoustic_path):
    if volume_path is not None:
        volume_data = np.load(volume_path)
        mua = volume_data[Tags.PROPERTY_ABSORPTION_PER_CM]
        sizes = np.shape(mua)
        mua = mua[:, int(sizes[1]/2), :]
        mus = volume_data[Tags.PROPERTY_SCATTERING_PER_CM][:, int(sizes[1]/2), :]
        g = volume_data[Tags.PROPERTY_ANISOTROPY][:, int(sizes[1] / 2), :]
        oxy = volume_data[Tags.PROPERTY_OXYGENATION][:, int(sizes[1] / 2), :]
        seg = volume_data[Tags.PROPERTY_SEGMENTATION][:, int(sizes[1] / 2), :]
        gamma = volume_data[Tags.PROPERTY_GRUNEISEN_PARAMETER]
        if type(gamma) is np.ndarray:
            gamma = gamma[:, int(sizes[1] / 2), :]

        np.savez(volume_path,
                 mua=mua,
                 mus=mus,
                 g=g,
                 oxy=oxy,
                 seg=seg,
                 gamma=gamma)

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
