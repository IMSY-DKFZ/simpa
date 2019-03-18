
from ippai.simulate.volume_creator import create_simulation_volume
from ippai.simulate.optical_model import run_optical_forward_model
from ippai.simulate.acoustic_model import run_acoustic_forward_model


def simulate(settings):
    volume_path = create_simulation_volume(settings)
    optical_path = None
    acoustic_path = None

    if settings['run_optical_forward_model']:
        optical_path = run_optical_forward_model(settings, volume_path)

    if settings['run_acoustic_forward_model']:
        acoustic_path = run_acoustic_forward_model(settings, optical_path)

    return [volume_path, optical_path, acoustic_path]









