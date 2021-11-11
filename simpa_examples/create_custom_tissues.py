# SPDX-FileCopyrightText: 2021 Computer Assisted Medical Interventions Group, DKFZ
# SPDX-FileCopyrightText: 2021 Janek Groehl
# SPDX-License-Identifier: MIT

import simpa as sp
import numpy as np


def create_custom_absorber():
    wavelengths = np.linspace(200, 1500, 100)
    absorber = sp.Spectrum(spectrum_name="random absorber",
                           wavelengths=wavelengths,
                           values=np.random.random(
                                      np.shape(wavelengths)))
    return absorber


def create_custom_chromophore(volume_fraction: float = 1.0):
    chromophore = sp.Molecule(
            absorption_spectrum=create_custom_absorber(),
            volume_fraction=volume_fraction,
            scattering_spectrum=sp.ScatteringSpectrumLibrary.CONSTANT_SCATTERING_ARBITRARY(40.0),
            anisotropy_spectrum=sp.AnisotropySpectrumLibrary.CONSTANT_ANISOTROPY_ARBITRARY(0.9)
        )
    return chromophore


def create_custom_tissue_type():

    # First create an instance of a TissueSettingsGenerator
    tissue_settings_generator = sp.MolecularCompositionGenerator()

    water_volume_fraction = 0.4
    blood_volume_fraction = 0.5
    custom_chromophore_volume_fraction = 0.1
    # The volume fraction within every tissue type should sum up to 1.

    oxygenation = 0.4

    # Then append chromophores that you want
    tissue_settings_generator.append(key="oxyhemoglobin",
                                     value=sp.MOLECULE_LIBRARY.oxyhemoglobin(oxygenation * blood_volume_fraction))
    tissue_settings_generator.append(key="deoxyhemoglobin",
                                     value=sp.MOLECULE_LIBRARY.deoxyhemoglobin((1 - oxygenation) * blood_volume_fraction))
    tissue_settings_generator.append(key="water",
                                     value=sp.MOLECULE_LIBRARY.water(water_volume_fraction))
    tissue_settings_generator.append(key="custom",
                                     value=create_custom_chromophore(custom_chromophore_volume_fraction))

    return tissue_settings_generator
