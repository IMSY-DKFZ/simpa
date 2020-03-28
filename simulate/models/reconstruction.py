from simulate import Tags, SaveFilePaths
from simulate.models.reconstruction_models.MitkBeamformingAdapter import MitkBeamformingAdapter
from io_handling.io_hdf5 import save_hdf5


def perform_reconstruction(settings, acoustic_data_path):
    print("ACOUSTIC FORWARD")

    reconstruction_method = None

    if ((settings[Tags.RECONSTRUCTION_ALGORITHM] == Tags.RECONSTRUCTION_ALGORITHM_DAS) or
        (settings[Tags.RECONSTRUCTION_ALGORITHM] == Tags.RECONSTRUCTION_ALGORITHM_DMAS) or
            (settings[Tags.RECONSTRUCTION_ALGORITHM] == Tags.RECONSTRUCTION_ALGORITHM_SDMAS)):
        reconstruction_method = MitkBeamformingAdapter()

    reconstruction = reconstruction_method.simulate(settings, acoustic_data_path)

    reconstruction_output_path = SaveFilePaths.RECONSTRCTION_OUTPUT.\
        format(Tags.ORIGINAL_DATA, settings[Tags.WAVELENGTH])

    if Tags.PERFORM_UPSAMPLING in settings:
        if settings[Tags.PERFORM_UPSAMPLING]:
            reconstruction_output_path = \
                SaveFilePaths.RECONSTRCTION_OUTPUT.format(Tags.UPSAMPLED_DATA, settings[Tags.WAVELENGTH])
    save_hdf5({Tags.RECONSTRUCTED_DATA: reconstruction}, settings[Tags.IPPAI_OUTPUT_PATH],
              reconstruction_output_path)

    return reconstruction_output_path
