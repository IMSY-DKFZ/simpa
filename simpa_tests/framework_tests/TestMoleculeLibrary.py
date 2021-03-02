# The MIT License (MIT)
#
# Copyright (c) 2018 Computer Assisted Medical Interventions Group, DKFZ
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated simpa_documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import unittest
from simpa.utils.libraries.tissue_library import MolecularComposition, MolecularCompositionGenerator
from simpa.utils.libraries.literature_values import StandardProperties, OpticalTissueProperties
from simpa.utils.libraries.molecule_library import MOLECULE_LIBRARY
from simpa.utils.libraries.spectra_library import AbsorptionSpectrum
from simpa.utils import SegmentationClasses
from simpa.utils import Tags
from simpa.utils import SPECTRAL_LIBRARY
import numpy as np
from simpa.utils.settings_generator import Settings
from simpa.utils.libraries.structure_library import Background
from simpa.core.volume_creation.versatile_volume_creator import ModelBasedVolumeCreator

def create_background_of_molecule(global_settings, molecule):
    background_structure_dictionary = dict()
    background_structure_dictionary[Tags.PRIORITY] = 0
    background_structure_dictionary[Tags.MOLECULE_COMPOSITION] = molecule
    bg = Background(global_settings, Settings(background_structure_dictionary))
    return bg.to_settings()

def create_test_structure_of_molecule(global_settings, molecule):
    structures_dict = dict()
    structures_dict["background"] = create_background_of_molecule(global_settings, molecule)
    return structures_dict

# MOLECULES = [
#     MOLECULE_LIBRARY.water(),
#     MOLECULE_LIBRARY.oxyhemoglobin(),
#     MOLECULE_LIBRARY.deoxyhemoglobin(),
#     MOLECULE_LIBRARY.melanin(),
#     MOLECULE_LIBRARY.fat(),
#     MOLECULE_LIBRARY.constant_scatterer(), 
#     MOLECULE_LIBRARY.soft_tissue_scatterer(),
#     MOLECULE_LIBRARY.epidermal_scatterer(),
#     MOLECULE_LIBRARY.dermal_scatterer(),
#     MOLECULE_LIBRARY.bone(),
#     MOLECULE_LIBRARY.mediprene(),
#     MOLECULE_LIBRARY.heavy_water(),
#     MOLECULE_LIBRARY.air()
# ]

# SPECTRA = [
#     SPECTRAL_LIBRARY.WATER,
#     SPECTRAL_LIBRARY.OXYHEMOGLOBIN,
#     SPECTRAL_LIBRARY.DEOXYHEMOGLOBIN, 
#     SPECTRAL_LIBRARY.MELANIN,
#     SPECTRAL_LIBRARY.FAT,
#     SPECTRAL_LIBRARY.CONSTANT_ABSORBER_ZERO,
#     SPECTRAL_LIBRARY.CONSTANT_ABSORBER_ZERO,
#     SPECTRAL_LIBRARY.CONSTANT_ABSORBER_ZERO,
#     SPECTRAL_LIBRARY.CONSTANT_ABSORBER_ZERO,
#     SPECTRAL_LIBRARY.CONSTANT_ABSORBER_ZERO,
#     SPECTRAL_LIBRARY.CONSTANT_ABSORBER_ARBITRARY(-np.log(0.85) / 10),
#     SPECTRAL_LIBRARY.CONSTANT_ABSORBER_ARBITRARY(StandardProperties.AIR_MUA),
#     SPECTRAL_LIBRARY.CONSTANT_ABSORBER_ARBITRARY(StandardProperties.AIR_MUA),
# ]

# MOLECULE_NAMES = [
#     'WATER',
#     'BLOOD',
#     'BLOOD', 
#     'SKIN',
#     'FAT',
#     'GENERIC', 
#     'GENERIC',
#     'SKIN',
#     'SKIN', 
#     'BONE', 
#     'GEL_PAD',
#     'HEAVY_WATER', 
#     'AIR',
# ]


class TestMoleculeLibrary(unittest.TestCase):
    def setUp(self):
        print("\n[SetUp]")
    
    def tearDown(self):
        print("\n[TearDown]")

    @staticmethod
    def set_settings():
        random_seed = 4711
        settings = {
            Tags.WAVELENGTHS: [500, 650, 700, 750, 800, 850, 900, 950, 1000],
            Tags.VOLUME_CREATOR: Tags.VOLUME_CREATOR_VERSATILE,
            Tags.RANDOM_SEED: random_seed,
            Tags.VOLUME_NAME: "MoleculePhantom_" + str(random_seed).zfill(6),
            Tags.SIMULATION_PATH: ".",
            Tags.RUN_OPTICAL_MODEL: False,
            Tags.RUN_ACOUSTIC_MODEL: False,
            Tags.SIMULATION_EXTRACT_FIELD_OF_VIEW: False,
            Tags.SPACING_MM: 0.25,
            Tags.DIM_VOLUME_Z_MM: 2,
            Tags.DIM_VOLUME_X_MM: 2,
            Tags.DIM_VOLUME_Y_MM: 2
        }
        return settings

    @staticmethod
    def test_water():
        print("Simulating Water")
        settings=TestMoleculeLibrary.set_settings()
        settings = Settings(settings)
        
        molecule = MolecularCompositionGenerator().append(MOLECULE_LIBRARY.water()).get_molecular_composition(SegmentationClasses.MUSCLE)
        settings[Tags.STRUCTURES] = create_test_structure_of_molecule(settings, molecule)
        spectrum = SPECTRAL_LIBRARY.WATER
            
        MOLECULE_NAME = 'Water'
        print('Molecule: ', MOLECULE_NAME)

        for wavelength in settings[Tags.WAVELENGTHS]:
            settings[Tags.WAVELENGTH] = wavelength
            volume_creator_adapter = ModelBasedVolumeCreator()
            volume = volume_creator_adapter.create_simulation_volume(settings)

            if wavelength ==500:
                assert (volume['mus'] == StandardProperties.WATER_MUS).all()

            assert (volume['density'] == StandardProperties.DENSITY_WATER).all()
            assert (volume['g'] == StandardProperties.WATER_G).all() 
            assert (volume['gamma'] == 0.2004).all()
            assert (volume['mua'] == spectrum.get_absorption_for_wavelength(wavelength)).all()
            assert (volume['sos'] == StandardProperties.SPEED_OF_SOUND_WATER).all()
            print('molecule ', MOLECULE_NAME, 'is ok ')

    @staticmethod
    def test_oxyhemoglobin():
        print("Simulating oxyhemoglobin")
        settings = TestMoleculeLibrary.set_settings()
        settings = Settings(settings)
        
        molecule = MolecularCompositionGenerator().append(MOLECULE_LIBRARY.oxyhemoglobin()).get_molecular_composition(SegmentationClasses.MUSCLE)
        settings[Tags.STRUCTURES] = create_test_structure_of_molecule(settings, molecule)
        spectrum = SPECTRAL_LIBRARY.OXYHEMOGLOBIN
            
        MOLECULE_NAME = 'oxyhemoglobin'
        print('Molecule: ', MOLECULE_NAME)

        for wavelength in settings[Tags.WAVELENGTHS]:
            settings[Tags.WAVELENGTH] = wavelength
            volume_creator_adapter = ModelBasedVolumeCreator()
            volume = volume_creator_adapter.create_simulation_volume(settings)

            if wavelength == 500:
                assert (volume['mus'] == OpticalTissueProperties.MUS500_BLOOD).all()#TODO
            assert (volume['density'] == StandardProperties.DENSITY_BLOOD).all()
            assert (volume['g'] == OpticalTissueProperties.STANDARD_ANISOTROPY).all()#TODO
            assert (volume['gamma'] == 0.2004).all()
            assert (volume['mua'] == spectrum.get_absorption_for_wavelength(wavelength)).all()
            assert (volume['sos'] == StandardProperties.SPEED_OF_SOUND_BLOOD).all()
            print('molecule ', MOLECULE_NAME, 'is ok ')
    
    @staticmethod
    def test_deoxyhemoglobin():
        print("Simulating deoxyhemoglobin")
        settings = TestMoleculeLibrary.set_settings()
        settings = Settings(settings)
        
        molecule = MolecularCompositionGenerator().append(MOLECULE_LIBRARY.deoxyhemoglobin()).get_molecular_composition(SegmentationClasses.MUSCLE)
        settings[Tags.STRUCTURES] = create_test_structure_of_molecule(settings, molecule)
        spectrum = SPECTRAL_LIBRARY.DEOXYHEMOGLOBIN
            
        MOLECULE_NAME = 'deoxyhemoglobin'
        print('Molecule: ', MOLECULE_NAME)

        for wavelength in settings[Tags.WAVELENGTHS]:
            settings[Tags.WAVELENGTH] = wavelength
            volume_creator_adapter = ModelBasedVolumeCreator()
            volume = volume_creator_adapter.create_simulation_volume(settings)

            if wavelength == 500:
                assert (volume['mus'] == OpticalTissueProperties.MUS500_BLOOD).all() #TODO
            assert (volume['density'] == StandardProperties.DENSITY_BLOOD).all()
            assert (volume['g'] == OpticalTissueProperties.STANDARD_ANISOTROPY).all()#TODO
            assert (volume['gamma'] == 0.2004).all()
            assert (volume['mua'] == spectrum.get_absorption_for_wavelength(wavelength)).all()
            assert (volume['sos'] == StandardProperties.SPEED_OF_SOUND_BLOOD).all()
            print('molecule ', MOLECULE_NAME, 'is ok ')

    @staticmethod
    def test_melanin():
        print("Simulating melanin")
        settings = TestMoleculeLibrary.set_settings()
        settings = Settings(settings)
        
        molecule = MolecularCompositionGenerator().append(MOLECULE_LIBRARY.melanin()).get_molecular_composition(SegmentationClasses.MUSCLE)
        settings[Tags.STRUCTURES] = create_test_structure_of_molecule(settings, molecule)
        spectrum = SPECTRAL_LIBRARY.MELANIN
            
        MOLECULE_NAME = 'melanin'
        print('Molecule: ', MOLECULE_NAME)

        for wavelength in settings[Tags.WAVELENGTHS]:
            settings[Tags.WAVELENGTH] = wavelength
            volume_creator_adapter = ModelBasedVolumeCreator()
            volume = volume_creator_adapter.create_simulation_volume(settings)

            if wavelength == 500:
                assert (volume['mus'] == OpticalTissueProperties.MUS500_EPIDERMIS).all() #TODO
            assert (volume['density'] == StandardProperties.DENSITY_SKIN).all()
            assert (volume['g'] == OpticalTissueProperties.STANDARD_ANISOTROPY).all()#TODO
            assert (volume['gamma'] == 0.2004).all()
            assert (volume['mua'] == spectrum.get_absorption_for_wavelength(wavelength)).all()
            assert (volume['sos'] == StandardProperties.SPEED_OF_SOUND_SKIN).all()
            print('molecule ', MOLECULE_NAME, 'is ok ')

    @staticmethod
    def test_fat():
        print("Simulating fat")
        settings = TestMoleculeLibrary.set_settings()
        settings = Settings(settings)
        
        molecule = MolecularCompositionGenerator().append(MOLECULE_LIBRARY.fat()).get_molecular_composition(SegmentationClasses.MUSCLE)
        settings[Tags.STRUCTURES] = create_test_structure_of_molecule(settings, molecule)
        spectrum = SPECTRAL_LIBRARY.FAT
            
        MOLECULE_NAME = 'fat'
        print('Molecule: ', MOLECULE_NAME)

        for wavelength in settings[Tags.WAVELENGTHS]:
            settings[Tags.WAVELENGTH] = wavelength
            volume_creator_adapter = ModelBasedVolumeCreator()
            volume = volume_creator_adapter.create_simulation_volume(settings)

            if wavelength == 500:
                assert (volume['mus'] == OpticalTissueProperties.MUS500_FAT).all() #TODO
            assert (volume['density'] == StandardProperties.DENSITY_FAT).all()
            assert (volume['g'] == OpticalTissueProperties.STANDARD_ANISOTROPY).all()#TODO
            assert (volume['gamma'] == 0.2004).all()
            assert (volume['mua'] == spectrum.get_absorption_for_wavelength(wavelength)).all()
            assert (volume['sos'] == StandardProperties.SPEED_OF_SOUND_FAT).all()
            print('molecule ', MOLECULE_NAME, 'is ok ')

    @staticmethod
    def test_constant_scatterer():
        print("Simulating constant_scatterer")
        settings = TestMoleculeLibrary.set_settings()
        settings = Settings(settings)
        
        molecule = MolecularCompositionGenerator().append(MOLECULE_LIBRARY.constant_scatterer()).get_molecular_composition(SegmentationClasses.MUSCLE)
        settings[Tags.STRUCTURES] = create_test_structure_of_molecule(settings, molecule)
        spectrum = SPECTRAL_LIBRARY.CONSTANT_ABSORBER_ZERO
            
        MOLECULE_NAME = 'constant_scatterer'
        print('Molecule: ', MOLECULE_NAME)

        for wavelength in settings[Tags.WAVELENGTHS]:
            settings[Tags.WAVELENGTH] = wavelength
            volume_creator_adapter = ModelBasedVolumeCreator()
            volume = volume_creator_adapter.create_simulation_volume(settings)

            if wavelength == 500:
                assert (volume['mus'] == 100.0).all() #TODO
            assert (volume['density'] == StandardProperties.DENSITY_GENERIC).all()
            assert (volume['g'] == OpticalTissueProperties.STANDARD_ANISOTROPY).all()#TODO
            assert (volume['gamma'] == 0.2004).all()
            assert (volume['mua'] == spectrum.get_absorption_for_wavelength(wavelength)).all()
            assert (volume['sos'] == StandardProperties.SPEED_OF_SOUND_GENERIC).all()
            print('molecule ', MOLECULE_NAME, 'is ok ')

    @staticmethod
    def test_soft_tissue_scatterer():
        print("Simulating soft_tissue_scatterer")
        settings = TestMoleculeLibrary.set_settings()
        settings = Settings(settings)
        
        molecule = MolecularCompositionGenerator().append(MOLECULE_LIBRARY.soft_tissue_scatterer()).get_molecular_composition(SegmentationClasses.MUSCLE)
        settings[Tags.STRUCTURES] = create_test_structure_of_molecule(settings, molecule)
        spectrum = SPECTRAL_LIBRARY.CONSTANT_ABSORBER_ZERO
            
        MOLECULE_NAME = 'soft_tissue_scatterer'
        print('Molecule: ', MOLECULE_NAME)

        for wavelength in settings[Tags.WAVELENGTHS]:
            settings[Tags.WAVELENGTH] = wavelength
            volume_creator_adapter = ModelBasedVolumeCreator()
            volume = volume_creator_adapter.create_simulation_volume(settings)

            if wavelength == 500:
                assert (volume['mus'] == OpticalTissueProperties.MUS500_BACKGROUND_TISSUE).all() #TODO
            assert (volume['density'] == StandardProperties.DENSITY_GENERIC).all()
            assert (volume['g'] == OpticalTissueProperties.STANDARD_ANISOTROPY).all()#TODO
            assert (volume['gamma'] == 0.2004).all()
            assert (volume['mua'] == spectrum.get_absorption_for_wavelength(wavelength)).all()
            assert (volume['sos'] == StandardProperties.SPEED_OF_SOUND_GENERIC).all()
            print('molecule ', MOLECULE_NAME, 'is ok ')

    @staticmethod
    def test_epidermal_scatterer():
        print("Simulating epidermal_scatterer")
        settings = TestMoleculeLibrary.set_settings()
        settings = Settings(settings)
        
        molecule = MolecularCompositionGenerator().append(MOLECULE_LIBRARY.epidermal_scatterer()).get_molecular_composition(SegmentationClasses.MUSCLE)
        settings[Tags.STRUCTURES] = create_test_structure_of_molecule(settings, molecule)
        spectrum = SPECTRAL_LIBRARY.CONSTANT_ABSORBER_ZERO
            
        MOLECULE_NAME = 'epidermal_scatterer'
        print('Molecule: ', MOLECULE_NAME)

        for wavelength in settings[Tags.WAVELENGTHS]:
            settings[Tags.WAVELENGTH] = wavelength
            volume_creator_adapter = ModelBasedVolumeCreator()
            volume = volume_creator_adapter.create_simulation_volume(settings)

            if wavelength == 500:
                assert (volume['mus'] == OpticalTissueProperties.MUS500_EPIDERMIS).all() #TODO
            assert (volume['density'] == StandardProperties.DENSITY_SKIN).all()
            assert (volume['g'] == OpticalTissueProperties.STANDARD_ANISOTROPY).all()#TODO
            assert (volume['gamma'] == 0.2004).all()
            assert (volume['mua'] == spectrum.get_absorption_for_wavelength(wavelength)).all()
            assert (volume['sos'] == StandardProperties.SPEED_OF_SOUND_SKIN).all()
            print('molecule ', MOLECULE_NAME, 'is ok ')

    @staticmethod
    def test_dermal_scatterer():
        print("Simulating dermal_scatterer")
        settings = TestMoleculeLibrary.set_settings()
        settings = Settings(settings)
        
        molecule = MolecularCompositionGenerator().append(MOLECULE_LIBRARY.dermal_scatterer()).get_molecular_composition(SegmentationClasses.MUSCLE)
        settings[Tags.STRUCTURES] = create_test_structure_of_molecule(settings, molecule)
        spectrum = SPECTRAL_LIBRARY.CONSTANT_ABSORBER_ZERO
            
        MOLECULE_NAME = 'dermal_scatterer'
        print('Molecule: ', MOLECULE_NAME)

        for wavelength in settings[Tags.WAVELENGTHS]:
            settings[Tags.WAVELENGTH] = wavelength
            volume_creator_adapter = ModelBasedVolumeCreator()
            volume = volume_creator_adapter.create_simulation_volume(settings)

            if wavelength == 500:
                assert (volume['mus'] == OpticalTissueProperties.MUS500_DERMIS).all() #TODO
            assert (volume['density'] == StandardProperties.DENSITY_SKIN).all()
            assert (volume['g'] == OpticalTissueProperties.STANDARD_ANISOTROPY).all()#TODO
            assert (volume['gamma'] == 0.2004).all()
            assert (volume['mua'] == spectrum.get_absorption_for_wavelength(wavelength)).all()
            assert (volume['sos'] == StandardProperties.SPEED_OF_SOUND_SKIN).all()
            print('molecule ', MOLECULE_NAME, 'is ok ')
    
    
    @staticmethod
    def test_bone():
        print("Simulating bone")
        settings = TestMoleculeLibrary.set_settings()
        settings = Settings(settings)
        
        molecule = MolecularCompositionGenerator().append(MOLECULE_LIBRARY.bone()).get_molecular_composition(SegmentationClasses.MUSCLE)
        settings[Tags.STRUCTURES] = create_test_structure_of_molecule(settings, molecule)
        spectrum = SPECTRAL_LIBRARY.CONSTANT_ABSORBER_ZERO
            
        MOLECULE_NAME = 'bone'
        print('Molecule: ', MOLECULE_NAME)

        for wavelength in settings[Tags.WAVELENGTHS]:
            settings[Tags.WAVELENGTH] = wavelength
            volume_creator_adapter = ModelBasedVolumeCreator()
            volume = volume_creator_adapter.create_simulation_volume(settings)

            if wavelength == 500:
                assert (volume['mus'] == OpticalTissueProperties.MUS500_BONE).all() #TODO
            assert (volume['density'] == StandardProperties.DENSITY_BONE).all()
            assert (volume['g'] == OpticalTissueProperties.STANDARD_ANISOTROPY).all()#TODO
            assert (volume['gamma'] == 0.2004).all()
            assert (volume['mua'] == spectrum.get_absorption_for_wavelength(wavelength)).all()
            assert (volume['sos'] == StandardProperties.SPEED_OF_SOUND_BONE).all()
            print('molecule ', MOLECULE_NAME, 'is ok ')

    @staticmethod
    def test_mediprene():
        print("Simulating mediprene")
        settings = TestMoleculeLibrary.set_settings()
        settings = Settings(settings)
        
        molecule = MolecularCompositionGenerator().append(MOLECULE_LIBRARY.mediprene()).get_molecular_composition(SegmentationClasses.MUSCLE)
        settings[Tags.STRUCTURES] = create_test_structure_of_molecule(settings, molecule)
        spectrum = SPECTRAL_LIBRARY.CONSTANT_ABSORBER_ARBITRARY(-np.log(0.85) / 10)
            
        MOLECULE_NAME = 'mediprene'
        print('Molecule: ', MOLECULE_NAME)

        for wavelength in settings[Tags.WAVELENGTHS]:
            settings[Tags.WAVELENGTH] = wavelength
            volume_creator_adapter = ModelBasedVolumeCreator()
            volume = volume_creator_adapter.create_simulation_volume(settings)

            if wavelength == 500:
                assert (volume['mus'] == (-np.log(0.85)) - (-np.log(0.85) / 10)).all() #TODO
            assert (volume['density'] == StandardProperties.DENSITY_GEL_PAD).all()
            assert (volume['g'] == OpticalTissueProperties.STANDARD_ANISOTROPY).all()#TODO
            assert (volume['gamma'] == 0.2004).all()
            assert (volume['mua'] == spectrum.get_absorption_for_wavelength(wavelength)).all()
            assert (volume['sos'] == StandardProperties.SPEED_OF_SOUND_GEL_PAD).all()
            print('molecule ', MOLECULE_NAME, 'is ok ')

    @staticmethod
    def test_heavy_water():
        print("Simulating heavy_water")
        settings = TestMoleculeLibrary.set_settings()
        settings = Settings(settings)
        
        molecule = MolecularCompositionGenerator().append(MOLECULE_LIBRARY.heavy_water()).get_molecular_composition(SegmentationClasses.MUSCLE)
        settings[Tags.STRUCTURES] = create_test_structure_of_molecule(settings, molecule)
        spectrum = SPECTRAL_LIBRARY.CONSTANT_ABSORBER_ARBITRARY(StandardProperties.AIR_MUA)
            
        MOLECULE_NAME = 'heavy_water'
        print('Molecule: ', MOLECULE_NAME)

        for wavelength in settings[Tags.WAVELENGTHS]:
            settings[Tags.WAVELENGTH] = wavelength
            volume_creator_adapter = ModelBasedVolumeCreator()
            volume = volume_creator_adapter.create_simulation_volume(settings)

            if wavelength == 500:
                assert (volume['mus'] == StandardProperties.WATER_MUS).all() #TODO
            assert (volume['density'] == StandardProperties.DENSITY_HEAVY_WATER).all()
            assert (volume['g'] == StandardProperties.WATER_G).all()#TODO
            assert (volume['gamma'] == 0.2004).all()
            assert (volume['mua'] == spectrum.get_absorption_for_wavelength(wavelength)).all()
            assert (volume['sos'] == StandardProperties.SPEED_OF_SOUND_HEAVY_WATER).all()
            print('molecule ', MOLECULE_NAME, 'is ok ')

    @staticmethod
    def test_air():
        print("Simulating air")
        settings = TestMoleculeLibrary.set_settings()
        settings = Settings(settings)
        
        molecule = MolecularCompositionGenerator().append(MOLECULE_LIBRARY.air()).get_molecular_composition(SegmentationClasses.MUSCLE)
        settings[Tags.STRUCTURES] = create_test_structure_of_molecule(settings, molecule)
        spectrum = SPECTRAL_LIBRARY.CONSTANT_ABSORBER_ARBITRARY(StandardProperties.AIR_MUA)
            
        MOLECULE_NAME = 'air'
        print('Molecule: ', MOLECULE_NAME)

        for wavelength in settings[Tags.WAVELENGTHS]:
            settings[Tags.WAVELENGTH] = wavelength
            volume_creator_adapter = ModelBasedVolumeCreator()
            volume = volume_creator_adapter.create_simulation_volume(settings)

            if wavelength == 500:
                assert (volume['mus'] == StandardProperties.AIR_MUS).all() #TODO
            assert (volume['density'] == StandardProperties.DENSITY_AIR).all()
            assert (volume['g'] == StandardProperties.AIR_G).all()#TODO
            assert (volume['gamma'] == 0.2004).all()
            assert (volume['mua'] == spectrum.get_absorption_for_wavelength(wavelength)).all()
            assert (volume['sos'] == StandardProperties.SPEED_OF_SOUND_AIR).all()
            print('molecule ', MOLECULE_NAME, 'is ok ')



TestMoleculeLibrary.test_water()
TestMoleculeLibrary.test_oxyhemoglobin()
TestMoleculeLibrary.test_deoxyhemoglobin()
TestMoleculeLibrary.test_melanin()
TestMoleculeLibrary.test_fat()
TestMoleculeLibrary.test_constant_scatterer()
TestMoleculeLibrary.test_soft_tissue_scatterer()
TestMoleculeLibrary.test_epidermal_scatterer()
TestMoleculeLibrary.test_dermal_scatterer()
TestMoleculeLibrary.test_bone()
TestMoleculeLibrary.test_mediprene()
TestMoleculeLibrary.test_heavy_water()
TestMoleculeLibrary.test_air()

