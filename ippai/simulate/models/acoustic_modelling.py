from ippai.simulate import Tags, SaveFilePaths
from ippai.simulate.models.acoustic_models import k_wave_adapter
from ippai.io_handling.io_hdf5 import save_hdf5


def run_acoustic_forward_model(settings, optical_path):
    print("ACOUSTIC FORWARD")

    data = k_wave_adapter.simulate(settings, optical_path)

    acoustic_output_path = SaveFilePaths.ACOUSTIC_OUTPUT.format("normal", settings[Tags.WAVELENGTH])
    if Tags.PERFORM_UPSAMPLING in settings:
        if settings[Tags.PERFORM_UPSAMPLING]:
            acoustic_output_path = SaveFilePaths.ACOUSTIC_OUTPUT.format("upsampled", settings[Tags.WAVELENGTH])
    save_hdf5({"time_series_data": data}, settings[Tags.IPPAI_OUTPUT_PATH],
              acoustic_output_path)

    return acoustic_output_path
