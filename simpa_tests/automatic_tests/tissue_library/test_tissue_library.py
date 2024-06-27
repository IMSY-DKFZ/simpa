# SPDX-FileCopyrightText: 2021 Division of Intelligent Medical Systems, DKFZ
# SPDX-FileCopyrightText: 2021 Janek Groehl
# SPDX-License-Identifier: MIT

import numpy as np
import pytest

from simpa import Spectrum, SegmentationClasses, TISSUE_LIBRARY, Molecule


def test_if_optical_parameter_spectra_are_provided_correct_tissue_definition_is_returned_from_generic_tissue():
    wavelengths_sample = np.arange(400, 800, 25)
    mua_sample = np.linspace(1e-6, 1e-5, wavelengths_sample.shape[0])
    mus_sample = np.linspace(1e-5, 5e-6, wavelengths_sample.shape[0])
    g_sample = np.linspace(0.8, 0.9, wavelengths_sample.shape[0])

    mua_spectrum = Spectrum("Mua", wavelengths_sample, mua_sample)
    mus_spectrum = Spectrum("Mus", wavelengths_sample, mus_sample)
    g_spectrum = Spectrum("g", wavelengths_sample, g_sample)

    actual_tissue = TISSUE_LIBRARY.generic_tissue(mua_spectrum, mus_spectrum, g_spectrum)
    assert actual_tissue.segmentation_type == SegmentationClasses.GENERIC

    assert len(actual_tissue) == 1
    molecule: Molecule = actual_tissue[0]
    assert molecule.volume_fraction == 1.0
    assert molecule.spectrum == mua_spectrum
    assert molecule.scattering_spectrum == mus_spectrum
    assert molecule.anisotropy_spectrum == g_spectrum


def test_if_generic_tissue_is_called_with_invalid_arguments_error_is_raised():
    wavelengths_sample = np.arange(400, 800, 25)
    mua_sample = np.linspace(1e-6, 1e-5, wavelengths_sample.shape[0])
    mus_sample = np.linspace(1e-5, 5e-6, wavelengths_sample.shape[0])
    g_sample = np.linspace(0.8, 0.9, wavelengths_sample.shape[0])

    mua_spectrum = Spectrum("Mua", wavelengths_sample, mua_sample)
    mus_spectrum = Spectrum("Mus", wavelengths_sample, mus_sample)
    g_spectrum = Spectrum("g", wavelengths_sample, g_sample)

    with pytest.raises(AssertionError):
        TISSUE_LIBRARY.generic_tissue(None, mus_spectrum, g_spectrum)

    with pytest.raises(AssertionError):
        TISSUE_LIBRARY.generic_tissue(mua_spectrum, None, g_spectrum)

    with pytest.raises(AssertionError):
        TISSUE_LIBRARY.generic_tissue(mua_spectrum, mus_spectrum, None)


def test_if_optical_parameter_spectra_are_provided_correct_tissue_definition_is_returned_from_constant():
    mua_sample = 1e-5
    mus_sample = 3e-6
    g_sample = 0.85

    actual_tissue = TISSUE_LIBRARY.constant(mua_sample, mus_sample, g_sample)
    assert actual_tissue.segmentation_type == SegmentationClasses.GENERIC

    assert len(actual_tissue) == 1
    molecule: Molecule = actual_tissue[0]
    assert molecule.volume_fraction == 1.0
    assert (molecule.spectrum.values == mua_sample).all()
    assert (molecule.scattering_spectrum.values == mus_sample).all()
    assert (molecule.anisotropy_spectrum.values == g_sample).all()


def test_if_constant_is_called_with_invalid_arguments_error_is_raised():
    mua_sample = 1e-5
    mus_sample = 3e-6
    g_sample = 0.85

    with pytest.raises(TypeError):
        TISSUE_LIBRARY.constant(None, mus_sample, g_sample)

    with pytest.raises(TypeError):
        TISSUE_LIBRARY.constant(mua_sample, None, g_sample)

    with pytest.raises(TypeError):
        TISSUE_LIBRARY.constant(mua_sample, mus_sample, None)
