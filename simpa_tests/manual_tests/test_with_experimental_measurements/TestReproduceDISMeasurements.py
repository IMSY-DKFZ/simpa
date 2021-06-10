"""
SPDX-FileCopyrightText: 2021 Computer Assisted Medical Interventions Group, DKFZ
SPDX-FileCopyrightText: 2021 VISION Lab, Cancer Research UK Cambridge Institute (CRUK CI)
SPDX-License-Identifier: MIT

This test was designed to reproduce the total reflectance and total transmission of light on multiple
wavelengths as determined with a Double integrating Sphere system.

The mu_a and mu_s values were determined with the inverse adding doubling algorithm.
Since the setup is only very vaguely approximated, we do not expect to achieve exactly the same results.

KNOWN ISSUES:
    - Reflection and Transmission are both off by approx. a factor of 3 in this simulation, but the
      multispectral behaviour is matches very well. This is most likely caused by an incorrect
      calculation of the total reflectance and transmission using he fluence map.
"""

from simpa.utils import Tags, TISSUE_LIBRARY, SegmentationClasses
from simpa.core.simulation import simulate
from simpa.utils.settings import Settings
from simpa.utils.libraries.molecule_library import MolecularCompositionGenerator, Molecule
from simpa.utils.libraries.spectra_library import Spectrum, AnisotropySpectrumLibrary
from simpa.io_handling import load_data_field
from simpa.core.device_digital_twins import *
import numpy as np
from simpa.visualisation.matplotlib_data_visualisation import visualise_data
from simpa.core import VolumeCreationModelModelBasedAdapter, OpticalForwardModelMcxAdapter
from simpa_tests.manual_tests.test_with_experimental_measurements.utils import read_reference_spectra, read_rxt_file
import inspect

from simpa.utils.path_manager import PathManager

# FIXME temporary workaround for newest Intel architectures
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# TODO: Please make sure that a valid path_config.env file is located in your home directory, or that you
#  point to the correct file in the PathManager().
path_manager = PathManager()

VOLUME_LENGTH_MM = 20
VOLUME_HEIGHT_IN_MM = 10
SPACING = 0.2
RANDOM_SEED = 471
VOLUME_NAME = "DIS_Test_"+str(RANDOM_SEED)
base_path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

# ##########################################################################
# Choose of of these
# ##########################################################################

# This one works pretty well
# inclusion_path_npz = base_path + "/test_data/background_material_light_spectra.npz"
# inclusion_path_rxt = base_path + "/test_data/background_material_light.rxt"

# This one does not work well
# inclusion_path_npz = base_path + "/test_data/background_material_dark_spectra.npz"
# inclusion_path_rxt = base_path + "/test_data/background_material_dark.rxt"

# This one looks perfect
inclusion_path_npz = base_path + "/test_data/inclusion_material_spectra.npz"
inclusion_path_rxt = base_path + "/test_data/inclusion_material.rxt"

# ##########################################################################
# ##########################################################################

# If VISUALIZE is set to True, the simulation result will be plotted
VISUALIZE = True

inclusion_mua, inclusion_mus, inclusion_g = read_reference_spectra(inclusion_path_npz)
inclusion_wavelengths, inclusion_reflectance, inclusion_transmittance, \
    inclusion_thickness = read_rxt_file(inclusion_path_rxt)

absorption_spectrum = Spectrum("inclusion_absorption", inclusion_wavelengths, inclusion_mua)
scattering_spectrum = Spectrum("inclusion_scattering", inclusion_wavelengths, inclusion_mus)
anisotropy_spectrum = AnisotropySpectrumLibrary.CONSTANT_ANISOTROPY_ARBITRARY(inclusion_g)
transmittance_spectrum = Spectrum("transmittance", inclusion_wavelengths, inclusion_transmittance)
reflectance_spectrum = Spectrum("reflectance", inclusion_wavelengths, inclusion_reflectance)

molecule = Molecule(
    scattering_spectrum=scattering_spectrum,
    absorption_spectrum=absorption_spectrum,
    anisotropy_spectrum=anisotropy_spectrum,
    volume_fraction=1.0
)


def create_measurement_setup(sample_tickness_mm):
    """
    This is a very simple example script of how to create a tissue definition.
    It contains a muscular background, an epidermis layer on top of the muscles
    and a blood vessel.
    """
    background_dictionary = Settings()
    background_dictionary[Tags.MOLECULE_COMPOSITION] = TISSUE_LIBRARY.constant(1e-5, 1, 1.0)
    background_dictionary[Tags.STRUCTURE_TYPE] = Tags.BACKGROUND

    inclusion_tissue = Settings()
    inclusion_tissue[Tags.STRUCTURE_TYPE] = Tags.HORIZONTAL_LAYER_STRUCTURE
    inclusion_tissue[Tags.PRIORITY] = 10
    inclusion_tissue[Tags.STRUCTURE_START_MM] = [0, 0, VOLUME_HEIGHT_IN_MM/2 - sample_tickness_mm/2]
    inclusion_tissue[Tags.STRUCTURE_END_MM] = [0, 0, VOLUME_HEIGHT_IN_MM/2 + sample_tickness_mm/2]
    inclusion_tissue[Tags.MOLECULE_COMPOSITION] = (MolecularCompositionGenerator()
                                                   .append(molecule)
                                                   .get_molecular_composition(segmentation_type=
                                                                              SegmentationClasses.GENERIC))
    inclusion_tissue[Tags.CONSIDER_PARTIAL_VOLUME] = True

    air_tube = Settings()
    air_tube[Tags.STRUCTURE_TYPE] = Tags.CIRCULAR_TUBULAR_STRUCTURE
    air_tube[Tags.PRIORITY] = 9
    air_tube[Tags.STRUCTURE_START_MM] = [VOLUME_LENGTH_MM/2, VOLUME_LENGTH_MM/2, 0]
    air_tube[Tags.STRUCTURE_END_MM] = [VOLUME_LENGTH_MM/2, VOLUME_LENGTH_MM/2, VOLUME_HEIGHT_IN_MM]
    air_tube[Tags.STRUCTURE_RADIUS_MM] = 5.0
    air_tube[Tags.MOLECULE_COMPOSITION] = TISSUE_LIBRARY.constant(1e-5, 1, 1.0)
    air_tube[Tags.CONSIDER_PARTIAL_VOLUME] = True

    absorbing_layer = Settings()

    absorbing_layer[Tags.STRUCTURE_TYPE] = Tags.HORIZONTAL_LAYER_STRUCTURE
    absorbing_layer[Tags.PRIORITY] = 8
    absorbing_layer[Tags.STRUCTURE_START_MM] = [0, 0, VOLUME_HEIGHT_IN_MM / 2 - sample_tickness_mm / 2 - SPACING]
    absorbing_layer[Tags.STRUCTURE_END_MM] = [0, 0, VOLUME_HEIGHT_IN_MM / 2 + sample_tickness_mm / 2 + SPACING]
    absorbing_layer[Tags.MOLECULE_COMPOSITION] = TISSUE_LIBRARY.constant(10000, 0.1, 0.0)
    absorbing_layer[Tags.CONSIDER_PARTIAL_VOLUME] = True

    tissue_dict = Settings()
    tissue_dict[Tags.BACKGROUND] = background_dictionary
    tissue_dict["slab"] = inclusion_tissue
    tissue_dict["boundary"] = absorbing_layer
    tissue_dict["air_tube"] = air_tube
    # tissue_dict["detector"] = detector_tissue
    return tissue_dict


# Seed the numpy random configuration prior to creating the global_settings file in
# order to ensure that the same volume
# is generated with the same random seed every time.
np.random.seed(RANDOM_SEED)

general_settings = {
    # These parameters set the general propeties of the simulated volume
    Tags.RANDOM_SEED: RANDOM_SEED,
    Tags.VOLUME_NAME: VOLUME_NAME,
    Tags.SIMULATION_PATH: path_manager.get_hdf5_file_save_path(),
    Tags.SPACING_MM: SPACING,
    Tags.DIM_VOLUME_Z_MM: VOLUME_HEIGHT_IN_MM,
    Tags.DIM_VOLUME_X_MM: VOLUME_LENGTH_MM,
    Tags.DIM_VOLUME_Y_MM: VOLUME_LENGTH_MM,
    Tags.WAVELENGTHS: [500, 550, 600, 650, 700, 750, 800, 850, 900],
    Tags.LOAD_AND_SAVE_HDF5_FILE_AT_THE_END_OF_SIMULATION_TO_MINIMISE_FILESIZE: True
}

settings = Settings(general_settings)

settings.set_volume_creation_settings({
    Tags.SIMULATE_DEFORMED_LAYERS: True,
    Tags.STRUCTURES: create_measurement_setup(inclusion_thickness)
})
settings.set_optical_settings({
    Tags.OPTICAL_MODEL_NUMBER_PHOTONS: 1e7,
    Tags.OPTICAL_MODEL_BINARY_PATH: path_manager.get_mcx_binary_path(),
    Tags.OPTICAL_MODEL: Tags.OPTICAL_MODEL_MCX,
    Tags.ILLUMINATION_TYPE: Tags.ILLUMINATION_TYPE_PENCIL,
    Tags.LASER_PULSE_ENERGY_IN_MILLIJOULE: 50,
    Tags.MCX_ASSUMED_ANISOTROPY: 0.7,
})

pipeline = [
    VolumeCreationModelModelBasedAdapter(settings),
    OpticalForwardModelMcxAdapter(settings),
]

device = PhotoacousticDevice(device_position_mm=np.asarray([VOLUME_LENGTH_MM / 2 + SPACING,
                                                            VOLUME_LENGTH_MM / 2,
                                                            VOLUME_HEIGHT_IN_MM / 2 - inclusion_thickness / 2 - 2 * SPACING]))
device.add_illumination_geometry(GaussianBeamIlluminationGeometry(beam_radius_mm=4.0))
device.add_illumination_geometry(PencilBeamIlluminationGeometry())

simulate(pipeline, settings, device)

import matplotlib.pyplot as plt

transmittances = []
reflectances = []

for wavelength in settings[Tags.WAVELENGTHS]:

    absorption = load_data_field(path_manager.get_hdf5_file_save_path() + "/" + VOLUME_NAME + ".hdf5",
                                 Tags.PROPERTY_ABSORPTION_PER_CM, wavelength)
    fluence = load_data_field(path_manager.get_hdf5_file_save_path() + "/" + VOLUME_NAME + ".hdf5",
                              Tags.OPTICAL_MODEL_FLUENCE, wavelength)

    start_z = int((VOLUME_HEIGHT_IN_MM / 2 - inclusion_thickness / 2 - 5 * SPACING) / SPACING)
    end_z = int((VOLUME_HEIGHT_IN_MM / 2 + inclusion_thickness / 2 + SPACING) / SPACING)

    fluence_start = fluence[:, :, start_z]
    incident_energy = (1 / SPACING) ** 2 * 100
    fluence_end = fluence[:, :, end_z]

    total_reflection = np.nansum(fluence_start) / incident_energy
    total_transmission = np.nansum(fluence_end) / incident_energy
    transmittances.append(total_transmission)
    reflectances.append((total_reflection))

    print("Incident Energy:", incident_energy)
    print(f"Total Reflection: {total_reflection * 100:.2f}% (expected "
          f"{reflectance_spectrum.get_value_for_wavelength(wavelength) * 100:.2f}%)")
    print(f"Total Transmission: {total_transmission * 100:.2f}% (expected "
          f"{transmittance_spectrum.get_value_for_wavelength(wavelength) * 100:.2f}%)")

visualise_data(path_manager.get_hdf5_file_save_path() + "/" + VOLUME_NAME + ".hdf5", wavelength=800,
               show_segmentation_map=False,
               show_absorption=True,
               show_fluence=True,
               log_scale=True)

measured_transmittance = [transmittance_spectrum.get_value_for_wavelength(wl) for wl in settings[Tags.WAVELENGTHS]]
simulated_transmittance = np.asarray(transmittances)
transmittance_ratio = np.mean(measured_transmittance/simulated_transmittance)
transmittance_ratio_std = np.std(measured_transmittance/simulated_transmittance)
plt.figure()
plt.subplot(1, 2, 1)
plt.title("Transmittance")
plt.plot(settings[Tags.WAVELENGTHS], simulated_transmittance * transmittance_ratio, label="simulation", color="red")
plt.plot(settings[Tags.WAVELENGTHS], measured_transmittance, label="measurement", color="green")
plt.legend(loc="best")

measured_reflectance = [reflectance_spectrum.get_value_for_wavelength(wl) for wl in settings[Tags.WAVELENGTHS]]
simulated_reflectance = np.asarray(reflectances)
reflectance_ratio = np.mean(measured_reflectance / simulated_reflectance)
reflectance_ratio_std = np.std(measured_reflectance / simulated_reflectance)
plt.subplot(1, 2, 2)
plt.title("Reflectance")
plt.plot(settings[Tags.WAVELENGTHS], simulated_reflectance * reflectance_ratio, label="simulation", color="red")
plt.plot(settings[Tags.WAVELENGTHS], measured_reflectance, label="measurement", color="green")
plt.legend(loc="best")

print(f"TRANSMITTANCE RATIO: {transmittance_ratio:.4f} (+/- {transmittance_ratio_std:.4f})")
print(f"REFLECTANCE RATIO: {reflectance_ratio:.4f} (+/- {reflectance_ratio_std:.4f})")

plt.show()
