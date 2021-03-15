# The MIT License (MIT)
#
# Copyright (c) 2021 Computer Assisted Medical Interventions Group, DKFZ
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
from simpa.core.volume_creation.versatile_volume_creator import ModelBasedVolumeCreator
from simpa.utils.calculate import calculate_gruneisen_parameter_from_temperature
from simpa_tests.test_utils import create_test_structure_of_molecule, create_background_of_molecule, set_settings


class TestMoleculeLibrary(unittest.TestCase):
    def setUp(self):
        print("\n[SetUp]")
    
    def tearDown(self):
        print("\n[TearDown]")

    def test_water(self):
        print("Simulating Water")
        settings = set_settings()
        confidence_interval=0.01
        settings = Settings(settings)
    
        MOLECULE_NAME = 'Water'  
        print('Molecule: ', MOLECULE_NAME)   
        spectrum = SPECTRAL_LIBRARY.WATER
                
        #setting1: background of molecule
        molecule = MolecularCompositionGenerator().append(MOLECULE_LIBRARY.water()).get_molecular_composition(SegmentationClasses.MUSCLE)
        settings[Tags.STRUCTURES] = create_test_structure_of_molecule(settings, molecule, molecule, molecule, key='setting1')
        for wavelength in settings[Tags.WAVELENGTHS]:
            settings[Tags.WAVELENGTH] = wavelength
            volume_creator_adapter = ModelBasedVolumeCreator()
            volume = volume_creator_adapter.create_simulation_volume(settings)

            #tests if mus, density, g, gamma, mua, and sos are equal to literature values in confidence interval
            if wavelength == 500:
                assert (np.abs(volume['mus'] - StandardProperties.WATER_MUS)<confidence_interval*StandardProperties.WATER_MUS).all()
            assert (np.abs(volume['density'] - StandardProperties.DENSITY_WATER)<confidence_interval *StandardProperties.DENSITY_WATER).all()
            assert (np.abs(volume['g'] - StandardProperties.WATER_G)<confidence_interval*StandardProperties.WATER_G).all() 
            assert (np.abs(volume['gamma'] - calculate_gruneisen_parameter_from_temperature(
                StandardProperties.BODY_TEMPERATURE_CELCIUS))<confidence_interval*calculate_gruneisen_parameter_from_temperature(
                StandardProperties.BODY_TEMPERATURE_CELCIUS)).all()
            assert (np.abs(volume['mua'] - spectrum.get_absorption_for_wavelength(wavelength)<confidence_interval*spectrum.get_absorption_for_wavelength(wavelength))).all()
            assert (np.abs(volume['sos'] - StandardProperties.SPEED_OF_SOUND_WATER)<confidence_interval*StandardProperties.SPEED_OF_SOUND_WATER).all()
        print('molecule ', MOLECULE_NAME, 'in setting 1 (background only) ok ')


        #setting2: vessel of molecule
        molecule1 = MolecularCompositionGenerator().append(MOLECULE_LIBRARY.fat()).get_molecular_composition(SegmentationClasses.MUSCLE)
        molecule2 = MolecularCompositionGenerator().append(MOLECULE_LIBRARY.water()).get_molecular_composition(SegmentationClasses.BLOOD)
        settings[Tags.STRUCTURES] = create_test_structure_of_molecule(settings, molecule1, molecule2, molecule2, key='setting2')
                
        for wavelength in settings[Tags.WAVELENGTHS]:
            settings[Tags.WAVELENGTH] = wavelength
            volume_creator_adapter = ModelBasedVolumeCreator()
            volume = volume_creator_adapter.create_simulation_volume(settings)
    
            #tests if mus, density, g, gamma, mua, and sos are equal to literature values in confidence interval
            if wavelength ==500:
                assert (np.abs(volume['mus'][volume['seg']==3] - StandardProperties.WATER_MUS)<confidence_interval*StandardProperties.WATER_MUS).all()
            assert (np.abs(volume['density'][volume['seg']==3] - StandardProperties.DENSITY_WATER)<confidence_interval *StandardProperties.DENSITY_WATER).all()
            assert (np.abs(volume['g'][volume['seg']==3] - StandardProperties.WATER_G)<confidence_interval*StandardProperties.WATER_G).all() 
            assert (np.abs(volume['gamma'][volume['seg']==3] - calculate_gruneisen_parameter_from_temperature(
                StandardProperties.BODY_TEMPERATURE_CELCIUS))<confidence_interval*calculate_gruneisen_parameter_from_temperature(
                StandardProperties.BODY_TEMPERATURE_CELCIUS)).all()
            assert (np.abs(volume['mua'][volume['seg']==3] - spectrum.get_absorption_for_wavelength(wavelength)<confidence_interval*spectrum.get_absorption_for_wavelength(wavelength))).all()
            assert (np.abs(volume['sos'][volume['seg']==3] - StandardProperties.SPEED_OF_SOUND_WATER)<confidence_interval*StandardProperties.SPEED_OF_SOUND_WATER).all()
        print('molecule ', MOLECULE_NAME, 'in setting 2 (vessel on background) ok ')


        #setting3: vessel of molecule on top of vessel of molecule
        molecule1 = MolecularCompositionGenerator().append(MOLECULE_LIBRARY.fat()).get_molecular_composition(SegmentationClasses.MUSCLE)
        molecule2 = MolecularCompositionGenerator().append(MOLECULE_LIBRARY.melanin()).get_molecular_composition(SegmentationClasses.EPIDERMIS)
        molecule3 = MolecularCompositionGenerator().append(MOLECULE_LIBRARY.water()).get_molecular_composition(SegmentationClasses.BLOOD)
        settings[Tags.STRUCTURES] = create_test_structure_of_molecule(settings, molecule1, molecule2, molecule3, key='setting3')
                
        for wavelength in settings[Tags.WAVELENGTHS]:
            settings[Tags.WAVELENGTH] = wavelength
            volume_creator_adapter = ModelBasedVolumeCreator()
            volume = volume_creator_adapter.create_simulation_volume(settings)

            #tests if mus, density, g, gamma, mua, and sos are equal to literature values in confidence interval
            if wavelength ==500:
                assert (np.abs(volume['mus'][volume['seg']==3] - StandardProperties.WATER_MUS)<confidence_interval*StandardProperties.WATER_MUS).all()
            assert (np.abs(volume['density'][volume['seg']==3] - StandardProperties.DENSITY_WATER)<confidence_interval *StandardProperties.DENSITY_WATER).all()
            assert (np.abs(volume['g'][volume['seg']==3] - StandardProperties.WATER_G)<confidence_interval*StandardProperties.WATER_G).all() 
            assert (np.abs(volume['gamma'][volume['seg']==3] - calculate_gruneisen_parameter_from_temperature(
                StandardProperties.BODY_TEMPERATURE_CELCIUS))<confidence_interval*calculate_gruneisen_parameter_from_temperature(
                StandardProperties.BODY_TEMPERATURE_CELCIUS)).all()
            assert (np.abs(volume['mua'][volume['seg']==3] - spectrum.get_absorption_for_wavelength(wavelength)<confidence_interval*spectrum.get_absorption_for_wavelength(wavelength))).all()
            assert (np.abs(volume['sos'][volume['seg']==3] - StandardProperties.SPEED_OF_SOUND_WATER)<confidence_interval*StandardProperties.SPEED_OF_SOUND_WATER).all()
        print('molecule ', MOLECULE_NAME, 'in setting 3 (vessel on vessel and background) ok ')


################################################
    def test_oxyhemoglobin(self):
        print("Simulating oxyhemoglobin")
        settings = set_settings()
        confidence_interval=0.01
        settings = Settings(settings)
        
        MOLECULE_NAME = 'oxyhemoglobin'
        print('Molecule: ', MOLECULE_NAME)
        spectrum = SPECTRAL_LIBRARY.OXYHEMOGLOBIN
                
        #setting1: background of molecule
        molecule = MolecularCompositionGenerator().append(MOLECULE_LIBRARY.oxyhemoglobin()).get_molecular_composition(SegmentationClasses.MUSCLE)
        settings[Tags.STRUCTURES] = create_test_structure_of_molecule(settings, molecule, molecule, molecule, key='setting1')

        #tests if mus, density, g, gamma, mua, and sos are equal to literature values in confidence interval            
        for wavelength in settings[Tags.WAVELENGTHS]:
            settings[Tags.WAVELENGTH] = wavelength
            volume_creator_adapter = ModelBasedVolumeCreator()
            volume = volume_creator_adapter.create_simulation_volume(settings)

            if wavelength == 500:
                assert (np.abs(volume['mus'] - OpticalTissueProperties.MUS500_BLOOD)<confidence_interval*OpticalTissueProperties.MUS500_BLOOD).all()
            assert (np.abs(volume['density'] - StandardProperties.DENSITY_BLOOD)<confidence_interval*StandardProperties.DENSITY_BLOOD).all()
            assert (np.abs(volume['g'] - OpticalTissueProperties.STANDARD_ANISOTROPY)<confidence_interval*OpticalTissueProperties.STANDARD_ANISOTROPY).all()
            assert (np.abs(volume['gamma'] - calculate_gruneisen_parameter_from_temperature(
                StandardProperties.BODY_TEMPERATURE_CELCIUS))<confidence_interval*calculate_gruneisen_parameter_from_temperature(
                StandardProperties.BODY_TEMPERATURE_CELCIUS)).all()
            assert (np.abs(volume['mua'] - spectrum.get_absorption_for_wavelength(wavelength))<confidence_interval*spectrum.get_absorption_for_wavelength(wavelength)).all()
            assert (np.abs(volume['sos'] - StandardProperties.SPEED_OF_SOUND_BLOOD)<confidence_interval*StandardProperties.SPEED_OF_SOUND_BLOOD).all()
        print('molecule ', MOLECULE_NAME, 'in setting 1 (background only) ok ')


        #setting2: vessel of molecule
        molecule1 = MolecularCompositionGenerator().append(MOLECULE_LIBRARY.water()).get_molecular_composition(SegmentationClasses.MUSCLE)
        molecule2 = MolecularCompositionGenerator().append(MOLECULE_LIBRARY.oxyhemoglobin()).get_molecular_composition(SegmentationClasses.BLOOD)
        settings[Tags.STRUCTURES] = create_test_structure_of_molecule(settings, molecule1, molecule2, molecule2, key='setting2')

        #tests if mus, density, g, gamma, mua, and sos are equal to literature values in confidence interval            
        for wavelength in settings[Tags.WAVELENGTHS]:
            settings[Tags.WAVELENGTH] = wavelength
            volume_creator_adapter = ModelBasedVolumeCreator()
            volume = volume_creator_adapter.create_simulation_volume(settings)

            if wavelength == 500:
                assert (np.abs(volume['mus'][volume['seg']==3] - OpticalTissueProperties.MUS500_BLOOD)<confidence_interval*OpticalTissueProperties.MUS500_BLOOD).all()
            assert (np.abs(volume['density'][volume['seg']==3] - StandardProperties.DENSITY_BLOOD)<confidence_interval*StandardProperties.DENSITY_BLOOD).all()
            assert (np.abs(volume['g'][volume['seg']==3] - OpticalTissueProperties.STANDARD_ANISOTROPY)<confidence_interval*OpticalTissueProperties.STANDARD_ANISOTROPY).all()
            assert (np.abs(volume['gamma'][volume['seg']==3] - calculate_gruneisen_parameter_from_temperature(
                StandardProperties.BODY_TEMPERATURE_CELCIUS))<confidence_interval*calculate_gruneisen_parameter_from_temperature(
                StandardProperties.BODY_TEMPERATURE_CELCIUS)).all()
            assert (np.abs(volume['mua'][volume['seg']==3] - spectrum.get_absorption_for_wavelength(wavelength))<confidence_interval*spectrum.get_absorption_for_wavelength(wavelength)).all()
            assert (np.abs(volume['sos'][volume['seg']==3] - StandardProperties.SPEED_OF_SOUND_BLOOD)<confidence_interval*StandardProperties.SPEED_OF_SOUND_BLOOD).all()
        print('molecule ', MOLECULE_NAME, 'in setting 2 (vessel on background) ok ')
    
        #setting3: vessel of molecule on top of vessel of molecule
        molecule1 = MolecularCompositionGenerator().append(MOLECULE_LIBRARY.water()).get_molecular_composition(SegmentationClasses.MUSCLE)
        molecule2 = MolecularCompositionGenerator().append(MOLECULE_LIBRARY.melanin()).get_molecular_composition(SegmentationClasses.EPIDERMIS)
        molecule3 = MolecularCompositionGenerator().append(MOLECULE_LIBRARY.oxyhemoglobin()).get_molecular_composition(SegmentationClasses.BLOOD)
        settings[Tags.STRUCTURES] = create_test_structure_of_molecule(settings, molecule1, molecule2, molecule3, key='setting3')

        #tests if mus, density, g, gamma, mua, and sos are equal to literature values in confidence interval            
        for wavelength in settings[Tags.WAVELENGTHS]:
            settings[Tags.WAVELENGTH] = wavelength
            volume_creator_adapter = ModelBasedVolumeCreator()
            volume = volume_creator_adapter.create_simulation_volume(settings)

            if wavelength == 500:
                assert (np.abs(volume['mus'][volume['seg']==3] - OpticalTissueProperties.MUS500_BLOOD)<confidence_interval*OpticalTissueProperties.MUS500_BLOOD).all()
            assert (np.abs(volume['density'][volume['seg']==3] - StandardProperties.DENSITY_BLOOD)<confidence_interval*StandardProperties.DENSITY_BLOOD).all()
            assert (np.abs(volume['g'][volume['seg']==3] - OpticalTissueProperties.STANDARD_ANISOTROPY)<confidence_interval*OpticalTissueProperties.STANDARD_ANISOTROPY).all()
            assert (np.abs(volume['gamma'][volume['seg']==3] - calculate_gruneisen_parameter_from_temperature(
                StandardProperties.BODY_TEMPERATURE_CELCIUS))<confidence_interval*calculate_gruneisen_parameter_from_temperature(
                StandardProperties.BODY_TEMPERATURE_CELCIUS)).all()
            assert (np.abs(volume['mua'][volume['seg']==3] - spectrum.get_absorption_for_wavelength(wavelength))<confidence_interval*spectrum.get_absorption_for_wavelength(wavelength)).all()
            assert (np.abs(volume['sos'][volume['seg']==3] - StandardProperties.SPEED_OF_SOUND_BLOOD)<confidence_interval*StandardProperties.SPEED_OF_SOUND_BLOOD).all()
        print('molecule ', MOLECULE_NAME, 'in setting 3 (vessel on vessel and background) ok ')


################################################
    def test_deoxyhemoglobin(self):
        print("Simulating deoxyhemoglobin")
        settings = set_settings()
        confidence_interval=0.01
        settings = Settings(settings)

        MOLECULE_NAME = 'deoxyhemoglobin'
        print('Molecule: ', MOLECULE_NAME)
        spectrum = SPECTRAL_LIBRARY.DEOXYHEMOGLOBIN

        #setting1: background of molecule
        molecule = MolecularCompositionGenerator().append(MOLECULE_LIBRARY.deoxyhemoglobin()).get_molecular_composition(SegmentationClasses.MUSCLE)
        settings[Tags.STRUCTURES] = create_test_structure_of_molecule(settings, molecule, molecule, molecule, key='setting1')
        #tests if mus, density, g, gamma, mua, and sos are equal to literature values in confidence interval  
        for wavelength in settings[Tags.WAVELENGTHS]:
            settings[Tags.WAVELENGTH] = wavelength
            volume_creator_adapter = ModelBasedVolumeCreator()
            volume = volume_creator_adapter.create_simulation_volume(settings)

            if wavelength == 500:
                assert (np.abs(volume['mus'] - OpticalTissueProperties.MUS500_BLOOD)<confidence_interval*OpticalTissueProperties.MUS500_BLOOD).all()
            assert (np.abs(volume['density'] - StandardProperties.DENSITY_BLOOD)<confidence_interval*StandardProperties.DENSITY_BLOOD).all()
            assert (np.abs(volume['g'] - OpticalTissueProperties.STANDARD_ANISOTROPY)<confidence_interval*OpticalTissueProperties.STANDARD_ANISOTROPY).all()
            assert (np.abs(volume['gamma'] - calculate_gruneisen_parameter_from_temperature(
                StandardProperties.BODY_TEMPERATURE_CELCIUS))<confidence_interval*calculate_gruneisen_parameter_from_temperature(
                StandardProperties.BODY_TEMPERATURE_CELCIUS)).all()
            assert (np.abs(volume['mua'] - spectrum.get_absorption_for_wavelength(wavelength))<confidence_interval*spectrum.get_absorption_for_wavelength(wavelength)).all()
            assert (np.abs(volume['sos'] - StandardProperties.SPEED_OF_SOUND_BLOOD)<confidence_interval*StandardProperties.SPEED_OF_SOUND_BLOOD).all()
            print('molecule ', MOLECULE_NAME, 'in setting 1 (background only) ok ')

        #setting2: vessel of molecule
        molecule1 = MolecularCompositionGenerator().append(MOLECULE_LIBRARY.water()).get_molecular_composition(SegmentationClasses.MUSCLE)
        molecule2 = MolecularCompositionGenerator().append(MOLECULE_LIBRARY.deoxyhemoglobin()).get_molecular_composition(SegmentationClasses.BLOOD)
        settings[Tags.STRUCTURES] = create_test_structure_of_molecule(settings, molecule1, molecule2, molecule2, key='setting2')

        #tests if mus, density, g, gamma, mua, and sos are equal to literature values in confidence interval            
        for wavelength in settings[Tags.WAVELENGTHS]:
            settings[Tags.WAVELENGTH] = wavelength
            volume_creator_adapter = ModelBasedVolumeCreator()
            volume = volume_creator_adapter.create_simulation_volume(settings)

            if wavelength == 500:
                assert (np.abs(volume['mus'][volume['seg']==3] - OpticalTissueProperties.MUS500_BLOOD)<confidence_interval*OpticalTissueProperties.MUS500_BLOOD).all()
            assert (np.abs(volume['density'][volume['seg']==3] - StandardProperties.DENSITY_BLOOD)<confidence_interval*StandardProperties.DENSITY_BLOOD).all()
            assert (np.abs(volume['g'][volume['seg']==3] - OpticalTissueProperties.STANDARD_ANISOTROPY)<confidence_interval*OpticalTissueProperties.STANDARD_ANISOTROPY).all()
            assert (np.abs(volume['gamma'][volume['seg']==3] - calculate_gruneisen_parameter_from_temperature(
                StandardProperties.BODY_TEMPERATURE_CELCIUS))<confidence_interval*calculate_gruneisen_parameter_from_temperature(
                StandardProperties.BODY_TEMPERATURE_CELCIUS)).all()
            assert (np.abs(volume['mua'][volume['seg']==3] - spectrum.get_absorption_for_wavelength(wavelength))<confidence_interval*spectrum.get_absorption_for_wavelength(wavelength)).all()
            assert (np.abs(volume['sos'][volume['seg']==3] - StandardProperties.SPEED_OF_SOUND_BLOOD)<confidence_interval*StandardProperties.SPEED_OF_SOUND_BLOOD).all()
        print('molecule ', MOLECULE_NAME, 'in setting 2 (vessel on background) ok ')
    
        #setting3: vessel of molecule on top of vessel of molecule
        molecule1 = MolecularCompositionGenerator().append(MOLECULE_LIBRARY.water()).get_molecular_composition(SegmentationClasses.MUSCLE)
        molecule2 = MolecularCompositionGenerator().append(MOLECULE_LIBRARY.melanin()).get_molecular_composition(SegmentationClasses.EPIDERMIS)
        molecule3 = MolecularCompositionGenerator().append(MOLECULE_LIBRARY.deoxyhemoglobin()).get_molecular_composition(SegmentationClasses.BLOOD)
        settings[Tags.STRUCTURES] = create_test_structure_of_molecule(settings, molecule1, molecule2, molecule3, key='setting3')

        #tests if mus, density, g, gamma, mua, and sos are equal to literature values in confidence interval            
        for wavelength in settings[Tags.WAVELENGTHS]:
            settings[Tags.WAVELENGTH] = wavelength
            volume_creator_adapter = ModelBasedVolumeCreator()
            volume = volume_creator_adapter.create_simulation_volume(settings)

            if wavelength == 500:
                assert (np.abs(volume['mus'][volume['seg']==3] - OpticalTissueProperties.MUS500_BLOOD)<confidence_interval*OpticalTissueProperties.MUS500_BLOOD).all()
            assert (np.abs(volume['density'][volume['seg']==3] - StandardProperties.DENSITY_BLOOD)<confidence_interval*StandardProperties.DENSITY_BLOOD).all()
            assert (np.abs(volume['g'][volume['seg']==3] - OpticalTissueProperties.STANDARD_ANISOTROPY)<confidence_interval*OpticalTissueProperties.STANDARD_ANISOTROPY).all()
            assert (np.abs(volume['gamma'][volume['seg']==3] - calculate_gruneisen_parameter_from_temperature(
                StandardProperties.BODY_TEMPERATURE_CELCIUS))<confidence_interval*calculate_gruneisen_parameter_from_temperature(
                StandardProperties.BODY_TEMPERATURE_CELCIUS)).all()
            assert (np.abs(volume['mua'][volume['seg']==3] - spectrum.get_absorption_for_wavelength(wavelength))<confidence_interval*spectrum.get_absorption_for_wavelength(wavelength)).all()
            assert (np.abs(volume['sos'][volume['seg']==3] - StandardProperties.SPEED_OF_SOUND_BLOOD)<confidence_interval*StandardProperties.SPEED_OF_SOUND_BLOOD).all()
        print('molecule ', MOLECULE_NAME, 'in setting 3 (vessel on vessel and background) ok ')


################################################
    def test_melanin(self):
        print("Simulating melanin")
        settings = set_settings()
        confidence_interval=0.01
        settings = Settings(settings)
        
        MOLECULE_NAME = 'melanin'
        print('Molecule: ', MOLECULE_NAME)
        spectrum = SPECTRAL_LIBRARY.MELANIN
        
        #setting1: background of molecule
        molecule = MolecularCompositionGenerator().append(MOLECULE_LIBRARY.melanin()).get_molecular_composition(SegmentationClasses.MUSCLE)
        settings[Tags.STRUCTURES] = create_test_structure_of_molecule(settings, molecule, molecule, molecule, key='setting1')
        #tests if mus, density, g, gamma, mua, and sos are equal to literature values in confidence interval  
        for wavelength in settings[Tags.WAVELENGTHS]:
            settings[Tags.WAVELENGTH] = wavelength
            volume_creator_adapter = ModelBasedVolumeCreator()
            volume = volume_creator_adapter.create_simulation_volume(settings)

            if wavelength == 500:
                assert (np.abs(volume['mus'] - OpticalTissueProperties.MUS500_EPIDERMIS)<confidence_interval*OpticalTissueProperties.MUS500_EPIDERMIS).all() 
            assert (np.abs(volume['density'] - StandardProperties.DENSITY_SKIN)<confidence_interval*StandardProperties.DENSITY_SKIN).all()
            assert (np.abs(volume['g'] - OpticalTissueProperties.STANDARD_ANISOTROPY)<confidence_interval*OpticalTissueProperties.STANDARD_ANISOTROPY).all()
            assert (np.abs(volume['gamma'] - calculate_gruneisen_parameter_from_temperature(
                StandardProperties.BODY_TEMPERATURE_CELCIUS))<confidence_interval*calculate_gruneisen_parameter_from_temperature(
                StandardProperties.BODY_TEMPERATURE_CELCIUS)).all()
            assert (np.abs(volume['mua'] - spectrum.get_absorption_for_wavelength(wavelength))<confidence_interval*spectrum.get_absorption_for_wavelength(wavelength)).all()
            assert (np.abs(volume['sos'] - StandardProperties.SPEED_OF_SOUND_SKIN)<confidence_interval*StandardProperties.SPEED_OF_SOUND_SKIN).all()
        print('molecule ', MOLECULE_NAME, 'in setting 1 (background only) ok ')

        #setting2: vessel of molecule
        molecule1 = MolecularCompositionGenerator().append(MOLECULE_LIBRARY.water()).get_molecular_composition(SegmentationClasses.MUSCLE)
        molecule2 = MolecularCompositionGenerator().append(MOLECULE_LIBRARY.melanin()).get_molecular_composition(SegmentationClasses.BLOOD)
        settings[Tags.STRUCTURES] = create_test_structure_of_molecule(settings, molecule1, molecule2, molecule2, key='setting2')

        #tests if mus, density, g, gamma, mua, and sos are equal to literature values in confidence interval            
        for wavelength in settings[Tags.WAVELENGTHS]:
            settings[Tags.WAVELENGTH] = wavelength
            volume_creator_adapter = ModelBasedVolumeCreator()
            volume = volume_creator_adapter.create_simulation_volume(settings)

            if wavelength == 500:
                assert (np.abs(volume['mus'][volume['seg']==3] - OpticalTissueProperties.MUS500_EPIDERMIS)<confidence_interval*OpticalTissueProperties.MUS500_EPIDERMIS).all() 
            assert (np.abs(volume['density'][volume['seg']==3] - StandardProperties.DENSITY_SKIN)<confidence_interval*StandardProperties.DENSITY_SKIN).all()
            assert (np.abs(volume['g'][volume['seg']==3] - OpticalTissueProperties.STANDARD_ANISOTROPY)<confidence_interval*OpticalTissueProperties.STANDARD_ANISOTROPY).all()
            assert (np.abs(volume['gamma'][volume['seg']==3] - calculate_gruneisen_parameter_from_temperature(
                StandardProperties.BODY_TEMPERATURE_CELCIUS))<confidence_interval*calculate_gruneisen_parameter_from_temperature(
                StandardProperties.BODY_TEMPERATURE_CELCIUS)).all()
            assert (np.abs(volume['mua'][volume['seg']==3] - spectrum.get_absorption_for_wavelength(wavelength))<confidence_interval*spectrum.get_absorption_for_wavelength(wavelength)).all()
            assert (np.abs(volume['sos'][volume['seg']==3] - StandardProperties.SPEED_OF_SOUND_SKIN)<confidence_interval*StandardProperties.SPEED_OF_SOUND_SKIN).all()
        print('molecule ', MOLECULE_NAME, 'in setting 2 (vessel on background) ok ')
    
        #setting3: vessel of molecule on top of vessel of molecule
        molecule1 = MolecularCompositionGenerator().append(MOLECULE_LIBRARY.water()).get_molecular_composition(SegmentationClasses.MUSCLE)
        molecule2 = MolecularCompositionGenerator().append(MOLECULE_LIBRARY.fat()).get_molecular_composition(SegmentationClasses.EPIDERMIS)
        molecule3 = MolecularCompositionGenerator().append(MOLECULE_LIBRARY.melanin()).get_molecular_composition(SegmentationClasses.BLOOD)
        settings[Tags.STRUCTURES] = create_test_structure_of_molecule(settings, molecule1, molecule2, molecule3, key='setting3')

        #tests if mus, density, g, gamma, mua, and sos are equal to literature values in confidence interval            
        for wavelength in settings[Tags.WAVELENGTHS]:
            settings[Tags.WAVELENGTH] = wavelength
            volume_creator_adapter = ModelBasedVolumeCreator()
            volume = volume_creator_adapter.create_simulation_volume(settings)

            if wavelength == 500:
                assert (np.abs(volume['mus'][volume['seg']==3] - OpticalTissueProperties.MUS500_EPIDERMIS)<confidence_interval*OpticalTissueProperties.MUS500_EPIDERMIS).all() 
            assert (np.abs(volume['density'][volume['seg']==3] - StandardProperties.DENSITY_SKIN)<confidence_interval*StandardProperties.DENSITY_SKIN).all()
            assert (np.abs(volume['g'][volume['seg']==3] - OpticalTissueProperties.STANDARD_ANISOTROPY)<confidence_interval*OpticalTissueProperties.STANDARD_ANISOTROPY).all()
            assert (np.abs(volume['gamma'][volume['seg']==3] - calculate_gruneisen_parameter_from_temperature(
                StandardProperties.BODY_TEMPERATURE_CELCIUS))<confidence_interval*calculate_gruneisen_parameter_from_temperature(
                StandardProperties.BODY_TEMPERATURE_CELCIUS)).all()
            assert (np.abs(volume['mua'][volume['seg']==3] - spectrum.get_absorption_for_wavelength(wavelength))<confidence_interval*spectrum.get_absorption_for_wavelength(wavelength)).all()
            assert (np.abs(volume['sos'][volume['seg']==3] - StandardProperties.SPEED_OF_SOUND_SKIN)<confidence_interval*StandardProperties.SPEED_OF_SOUND_SKIN).all()
        print('molecule ', MOLECULE_NAME, 'in setting 3 (vessel on vessel and background) ok ')


    def test_fat(self):
        print("Simulating fat")
        settings = set_settings()
        confidence_interval=0.01
        settings = Settings(settings)

        MOLECULE_NAME = 'fat'
        print('Molecule: ', MOLECULE_NAME)
        spectrum = SPECTRAL_LIBRARY.FAT
                
        #setting1: background of molecule
        molecule = MolecularCompositionGenerator().append(MOLECULE_LIBRARY.fat()).get_molecular_composition(SegmentationClasses.MUSCLE)
        settings[Tags.STRUCTURES] = create_test_structure_of_molecule(settings, molecule, molecule, molecule, key='setting1')
        #tests if mus, density, g, gamma, mua, and sos are equal to literature values in confidence interval  
        for wavelength in settings[Tags.WAVELENGTHS]:
            settings[Tags.WAVELENGTH] = wavelength
            volume_creator_adapter = ModelBasedVolumeCreator()
            volume = volume_creator_adapter.create_simulation_volume(settings)

            if wavelength == 500:
                assert (np.abs(volume['mus'] - OpticalTissueProperties.MUS500_FAT)<confidence_interval*OpticalTissueProperties.MUS500_FAT).all() 
            assert (np.abs(volume['density'] - StandardProperties.DENSITY_FAT)<confidence_interval*StandardProperties.DENSITY_FAT).all()
            assert (np.abs(volume['g'] - OpticalTissueProperties.STANDARD_ANISOTROPY)<confidence_interval*OpticalTissueProperties.STANDARD_ANISOTROPY).all()
            assert (np.abs(volume['gamma'] - calculate_gruneisen_parameter_from_temperature(
                StandardProperties.BODY_TEMPERATURE_CELCIUS))<confidence_interval*calculate_gruneisen_parameter_from_temperature(
                StandardProperties.BODY_TEMPERATURE_CELCIUS)).all()
            assert (np.abs(volume['mua'] - spectrum.get_absorption_for_wavelength(wavelength))<confidence_interval*spectrum.get_absorption_for_wavelength(wavelength)).all()
            assert (np.abs(volume['sos'] - StandardProperties.SPEED_OF_SOUND_FAT)<confidence_interval*StandardProperties.SPEED_OF_SOUND_FAT).all()
        print('molecule ', MOLECULE_NAME, 'in setting 1 (background only) ok ')

        #setting2: vessel of molecule
        molecule1 = MolecularCompositionGenerator().append(MOLECULE_LIBRARY.water()).get_molecular_composition(SegmentationClasses.MUSCLE)
        molecule2 = MolecularCompositionGenerator().append(MOLECULE_LIBRARY.fat()).get_molecular_composition(SegmentationClasses.BLOOD)
        settings[Tags.STRUCTURES] = create_test_structure_of_molecule(settings, molecule1, molecule2, molecule2, key='setting2')

        #tests if mus, density, g, gamma, mua, and sos are equal to literature values in confidence interval            
        for wavelength in settings[Tags.WAVELENGTHS]:
            settings[Tags.WAVELENGTH] = wavelength
            volume_creator_adapter = ModelBasedVolumeCreator()
            volume = volume_creator_adapter.create_simulation_volume(settings)

            if wavelength == 500:
                assert (np.abs(volume['mus'][volume['seg']==3] - OpticalTissueProperties.MUS500_FAT)<confidence_interval*OpticalTissueProperties.MUS500_FAT).all() 
            assert (np.abs(volume['density'][volume['seg']==3] - StandardProperties.DENSITY_FAT)<confidence_interval*StandardProperties.DENSITY_FAT).all()
            assert (np.abs(volume['g'][volume['seg']==3] - OpticalTissueProperties.STANDARD_ANISOTROPY)<confidence_interval*OpticalTissueProperties.STANDARD_ANISOTROPY).all()
            assert (np.abs(volume['gamma'][volume['seg']==3] - calculate_gruneisen_parameter_from_temperature(
                StandardProperties.BODY_TEMPERATURE_CELCIUS))<confidence_interval*calculate_gruneisen_parameter_from_temperature(
                StandardProperties.BODY_TEMPERATURE_CELCIUS)).all()
            assert (np.abs(volume['mua'][volume['seg']==3] - spectrum.get_absorption_for_wavelength(wavelength))<confidence_interval*spectrum.get_absorption_for_wavelength(wavelength)).all()
            assert (np.abs(volume['sos'][volume['seg']==3] - StandardProperties.SPEED_OF_SOUND_FAT)<confidence_interval*StandardProperties.SPEED_OF_SOUND_FAT).all()
        print('molecule ', MOLECULE_NAME, 'in setting 2 (vessel on background) ok ')
    
        #setting3: vessel of molecule on top of vessel of molecule
        molecule1 = MolecularCompositionGenerator().append(MOLECULE_LIBRARY.water()).get_molecular_composition(SegmentationClasses.MUSCLE)
        molecule2 = MolecularCompositionGenerator().append(MOLECULE_LIBRARY.melanin()).get_molecular_composition(SegmentationClasses.EPIDERMIS)
        molecule3 = MolecularCompositionGenerator().append(MOLECULE_LIBRARY.fat()).get_molecular_composition(SegmentationClasses.BLOOD)
        settings[Tags.STRUCTURES] = create_test_structure_of_molecule(settings, molecule1, molecule2, molecule3, key='setting3')

        #tests if mus, density, g, gamma, mua, and sos are equal to literature values in confidence interval            
        for wavelength in settings[Tags.WAVELENGTHS]:
            settings[Tags.WAVELENGTH] = wavelength
            volume_creator_adapter = ModelBasedVolumeCreator()
            volume = volume_creator_adapter.create_simulation_volume(settings)

            if wavelength == 500:
                assert (np.abs(volume['mus'][volume['seg']==3] - OpticalTissueProperties.MUS500_FAT)<confidence_interval*OpticalTissueProperties.MUS500_FAT).all() 
            assert (np.abs(volume['density'][volume['seg']==3] - StandardProperties.DENSITY_FAT)<confidence_interval*StandardProperties.DENSITY_FAT).all()
            assert (np.abs(volume['g'][volume['seg']==3] - OpticalTissueProperties.STANDARD_ANISOTROPY)<confidence_interval*OpticalTissueProperties.STANDARD_ANISOTROPY).all()
            assert (np.abs(volume['gamma'][volume['seg']==3] - calculate_gruneisen_parameter_from_temperature(
                StandardProperties.BODY_TEMPERATURE_CELCIUS))<confidence_interval*calculate_gruneisen_parameter_from_temperature(
                StandardProperties.BODY_TEMPERATURE_CELCIUS)).all()
            assert (np.abs(volume['mua'][volume['seg']==3] - spectrum.get_absorption_for_wavelength(wavelength))<confidence_interval*spectrum.get_absorption_for_wavelength(wavelength)).all()
            assert (np.abs(volume['sos'][volume['seg']==3] - StandardProperties.SPEED_OF_SOUND_FAT)<confidence_interval*StandardProperties.SPEED_OF_SOUND_FAT).all()
        print('molecule ', MOLECULE_NAME, 'in setting 3 (vessel on vessel and background) ok ')


    def test_constant_scatterer(self):
        print("Simulating constant_scatterer")
        settings = set_settings()
        confidence_interval=0.01
        settings = Settings(settings)
        
        MOLECULE_NAME = 'constant_scatterer'
        print('Molecule: ', MOLECULE_NAME)
        spectrum = SPECTRAL_LIBRARY.CONSTANT_ABSORBER_ZERO

        #setting1: background of molecule
        molecule = MolecularCompositionGenerator().append(MOLECULE_LIBRARY.constant_scatterer()).get_molecular_composition(SegmentationClasses.MUSCLE)
        settings[Tags.STRUCTURES] = create_test_structure_of_molecule(settings, molecule, molecule, molecule, key='setting1')   
        #tests if mus, density, g, gamma, mua, and sos are equal to literature values in confidence interval  
        for wavelength in settings[Tags.WAVELENGTHS]:
            settings[Tags.WAVELENGTH] = wavelength
            volume_creator_adapter = ModelBasedVolumeCreator()
            volume = volume_creator_adapter.create_simulation_volume(settings)

            if wavelength == 500:
                assert (np.abs(volume['mus'] - 100.0)<confidence_interval*100.0).all() 
            assert (np.abs(volume['density'] - StandardProperties.DENSITY_GENERIC)<confidence_interval*StandardProperties.DENSITY_GENERIC).all()
            assert (np.abs(volume['g'] - OpticalTissueProperties.STANDARD_ANISOTROPY)<confidence_interval*OpticalTissueProperties.STANDARD_ANISOTROPY).all()
            assert (np.abs(volume['gamma'] - calculate_gruneisen_parameter_from_temperature(
                StandardProperties.BODY_TEMPERATURE_CELCIUS))<confidence_interval*calculate_gruneisen_parameter_from_temperature(
                StandardProperties.BODY_TEMPERATURE_CELCIUS)).all()
            assert (np.abs(volume['mua'] - spectrum.get_absorption_for_wavelength(wavelength))<confidence_interval*spectrum.get_absorption_for_wavelength(wavelength)).all()
            assert (np.abs(volume['sos'] - StandardProperties.SPEED_OF_SOUND_GENERIC)<confidence_interval*StandardProperties.SPEED_OF_SOUND_GENERIC).all()
        print('molecule ', MOLECULE_NAME, 'in setting 1 (background only) ok ')

        #setting2: vessel of molecule
        molecule1 = MolecularCompositionGenerator().append(MOLECULE_LIBRARY.water()).get_molecular_composition(SegmentationClasses.MUSCLE)
        molecule2 = MolecularCompositionGenerator().append(MOLECULE_LIBRARY.constant_scatterer()).get_molecular_composition(SegmentationClasses.BLOOD)
        settings[Tags.STRUCTURES] = create_test_structure_of_molecule(settings, molecule1, molecule2, molecule2, key='setting2')

        #tests if mus, density, g, gamma, mua, and sos are equal to literature values in confidence interval            
        for wavelength in settings[Tags.WAVELENGTHS]:
            settings[Tags.WAVELENGTH] = wavelength
            volume_creator_adapter = ModelBasedVolumeCreator()
            volume = volume_creator_adapter.create_simulation_volume(settings)

            if wavelength == 500:
                assert (np.abs(volume['mus'][volume['seg']==3] - 100.0)<confidence_interval*100.0).all() 
            assert (np.abs(volume['density'][volume['seg']==3] - StandardProperties.DENSITY_GENERIC)<confidence_interval*StandardProperties.DENSITY_GENERIC).all()
            assert (np.abs(volume['g'][volume['seg']==3] - OpticalTissueProperties.STANDARD_ANISOTROPY)<confidence_interval*OpticalTissueProperties.STANDARD_ANISOTROPY).all()
            assert (np.abs(volume['gamma'][volume['seg']==3] - calculate_gruneisen_parameter_from_temperature(
                StandardProperties.BODY_TEMPERATURE_CELCIUS))<confidence_interval*calculate_gruneisen_parameter_from_temperature(
                StandardProperties.BODY_TEMPERATURE_CELCIUS)).all()
            assert (np.abs(volume['mua'][volume['seg']==3] - spectrum.get_absorption_for_wavelength(wavelength))<confidence_interval*spectrum.get_absorption_for_wavelength(wavelength)).all()
            assert (np.abs(volume['sos'][volume['seg']==3] - StandardProperties.SPEED_OF_SOUND_GENERIC)<confidence_interval*StandardProperties.SPEED_OF_SOUND_GENERIC).all()
        print('molecule ', MOLECULE_NAME, 'in setting 2 (vessel on background) ok ')
    
        #setting3: vessel of molecule on top of vessel of molecule
        molecule1 = MolecularCompositionGenerator().append(MOLECULE_LIBRARY.water()).get_molecular_composition(SegmentationClasses.MUSCLE)
        molecule2 = MolecularCompositionGenerator().append(MOLECULE_LIBRARY.melanin()).get_molecular_composition(SegmentationClasses.EPIDERMIS)
        molecule3 = MolecularCompositionGenerator().append(MOLECULE_LIBRARY.constant_scatterer()).get_molecular_composition(SegmentationClasses.BLOOD)
        settings[Tags.STRUCTURES] = create_test_structure_of_molecule(settings, molecule1, molecule2, molecule3, key='setting3')

        #tests if mus, density, g, gamma, mua, and sos are equal to literature values in confidence interval            
        for wavelength in settings[Tags.WAVELENGTHS]:
            settings[Tags.WAVELENGTH] = wavelength
            volume_creator_adapter = ModelBasedVolumeCreator()
            volume = volume_creator_adapter.create_simulation_volume(settings)

            if wavelength == 500:
                assert (np.abs(volume['mus'][volume['seg']==3] - 100.0)<confidence_interval*100.0).all() 
            assert (np.abs(volume['density'][volume['seg']==3] - StandardProperties.DENSITY_GENERIC)<confidence_interval*StandardProperties.DENSITY_GENERIC).all()
            assert (np.abs(volume['g'][volume['seg']==3] - OpticalTissueProperties.STANDARD_ANISOTROPY)<confidence_interval*OpticalTissueProperties.STANDARD_ANISOTROPY).all()
            assert (np.abs(volume['gamma'][volume['seg']==3] - calculate_gruneisen_parameter_from_temperature(
                StandardProperties.BODY_TEMPERATURE_CELCIUS))<confidence_interval*calculate_gruneisen_parameter_from_temperature(
                StandardProperties.BODY_TEMPERATURE_CELCIUS)).all()
            assert (np.abs(volume['mua'][volume['seg']==3] - spectrum.get_absorption_for_wavelength(wavelength))<confidence_interval*spectrum.get_absorption_for_wavelength(wavelength)).all()
            assert (np.abs(volume['sos'][volume['seg']==3] - StandardProperties.SPEED_OF_SOUND_GENERIC)<confidence_interval*StandardProperties.SPEED_OF_SOUND_GENERIC).all()
        print('molecule ', MOLECULE_NAME, 'in setting 3 (vessel on vessel and background) ok ')


################################################
    def test_soft_tissue_scatterer(self):
        print("Simulating soft_tissue_scatterer")
        settings = set_settings()
        confidence_interval=0.01
        settings = Settings(settings)
        
        MOLECULE_NAME = 'soft_tissue_scatterer'
        print('Molecule: ', MOLECULE_NAME)
        spectrum = SPECTRAL_LIBRARY.CONSTANT_ABSORBER_ZERO
        
        #setting1: background of molecule
        molecule = MolecularCompositionGenerator().append(MOLECULE_LIBRARY.soft_tissue_scatterer()).get_molecular_composition(SegmentationClasses.MUSCLE)
        settings[Tags.STRUCTURES] = create_test_structure_of_molecule(settings, molecule, molecule, molecule, key='setting1')   
        #tests if mus, density, g, gamma, mua, and sos are equal to literature values in confidence interval  
        for wavelength in settings[Tags.WAVELENGTHS]:
            settings[Tags.WAVELENGTH] = wavelength
            volume_creator_adapter = ModelBasedVolumeCreator()
            volume = volume_creator_adapter.create_simulation_volume(settings)

            if wavelength == 500:
                assert (np.abs(volume['mus'] - OpticalTissueProperties.MUS500_BACKGROUND_TISSUE)<confidence_interval*OpticalTissueProperties.MUS500_BACKGROUND_TISSUE).all() 
            assert (np.abs(volume['density'] - StandardProperties.DENSITY_GENERIC)<confidence_interval*StandardProperties.DENSITY_GENERIC).all()
            assert (np.abs(volume['g'] - OpticalTissueProperties.STANDARD_ANISOTROPY)<confidence_interval*OpticalTissueProperties.STANDARD_ANISOTROPY).all()
            assert (np.abs(volume['gamma'] - calculate_gruneisen_parameter_from_temperature(
                StandardProperties.BODY_TEMPERATURE_CELCIUS))<confidence_interval*calculate_gruneisen_parameter_from_temperature(
                StandardProperties.BODY_TEMPERATURE_CELCIUS)).all()
            assert (np.abs(volume['mua'] - spectrum.get_absorption_for_wavelength(wavelength))<confidence_interval*spectrum.get_absorption_for_wavelength(wavelength)).all()
            assert (np.abs(volume['sos'] - StandardProperties.SPEED_OF_SOUND_GENERIC)<confidence_interval*StandardProperties.SPEED_OF_SOUND_GENERIC).all()
        print('molecule ', MOLECULE_NAME, 'in setting 1 (background only) ok ')

        #setting2: vessel of molecule
        molecule1 = MolecularCompositionGenerator().append(MOLECULE_LIBRARY.water()).get_molecular_composition(SegmentationClasses.MUSCLE)
        molecule2 = MolecularCompositionGenerator().append(MOLECULE_LIBRARY.soft_tissue_scatterer()).get_molecular_composition(SegmentationClasses.BLOOD)
        settings[Tags.STRUCTURES] = create_test_structure_of_molecule(settings, molecule1, molecule2, molecule2, key='setting2')

        #tests if mus, density, g, gamma, mua, and sos are equal to literature values in confidence interval            
        for wavelength in settings[Tags.WAVELENGTHS]:
            settings[Tags.WAVELENGTH] = wavelength
            volume_creator_adapter = ModelBasedVolumeCreator()
            volume = volume_creator_adapter.create_simulation_volume(settings)

            if wavelength == 500:
                assert (np.abs(volume['mus'][volume['seg']==3] - OpticalTissueProperties.MUS500_BACKGROUND_TISSUE)<confidence_interval*OpticalTissueProperties.MUS500_BACKGROUND_TISSUE).all() 
            assert (np.abs(volume['density'][volume['seg']==3]  - StandardProperties.DENSITY_GENERIC)<confidence_interval*StandardProperties.DENSITY_GENERIC).all()
            assert (np.abs(volume['g'][volume['seg']==3]  - OpticalTissueProperties.STANDARD_ANISOTROPY)<confidence_interval*OpticalTissueProperties.STANDARD_ANISOTROPY).all()
            assert (np.abs(volume['gamma'][volume['seg']==3]  - calculate_gruneisen_parameter_from_temperature(
                StandardProperties.BODY_TEMPERATURE_CELCIUS))<confidence_interval*calculate_gruneisen_parameter_from_temperature(
                StandardProperties.BODY_TEMPERATURE_CELCIUS)).all()
            assert (np.abs(volume['mua'][volume['seg']==3]  - spectrum.get_absorption_for_wavelength(wavelength))<confidence_interval*spectrum.get_absorption_for_wavelength(wavelength)).all()
            assert (np.abs(volume['sos'][volume['seg']==3]  - StandardProperties.SPEED_OF_SOUND_GENERIC)<confidence_interval*StandardProperties.SPEED_OF_SOUND_GENERIC).all()
        print('molecule ', MOLECULE_NAME, 'in setting 2 (vessel on background) ok ')
    
        #setting3: vessel of molecule on top of vessel of molecule
        molecule1 = MolecularCompositionGenerator().append(MOLECULE_LIBRARY.water()).get_molecular_composition(SegmentationClasses.MUSCLE)
        molecule2 = MolecularCompositionGenerator().append(MOLECULE_LIBRARY.melanin()).get_molecular_composition(SegmentationClasses.EPIDERMIS)
        molecule3 = MolecularCompositionGenerator().append(MOLECULE_LIBRARY.soft_tissue_scatterer()).get_molecular_composition(SegmentationClasses.BLOOD)
        settings[Tags.STRUCTURES] = create_test_structure_of_molecule(settings, molecule1, molecule2, molecule3, key='setting3')

        #tests if mus, density, g, gamma, mua, and sos are equal to literature values in confidence interval            
        for wavelength in settings[Tags.WAVELENGTHS]:
            settings[Tags.WAVELENGTH] = wavelength
            volume_creator_adapter = ModelBasedVolumeCreator()
            volume = volume_creator_adapter.create_simulation_volume(settings)

            if wavelength == 500:
                assert (np.abs(volume['mus'][volume['seg']==3] - OpticalTissueProperties.MUS500_BACKGROUND_TISSUE)<confidence_interval*OpticalTissueProperties.MUS500_BACKGROUND_TISSUE).all() 
            assert (np.abs(volume['density'][volume['seg']==3]  - StandardProperties.DENSITY_GENERIC)<confidence_interval*StandardProperties.DENSITY_GENERIC).all()
            assert (np.abs(volume['g'][volume['seg']==3]  - OpticalTissueProperties.STANDARD_ANISOTROPY)<confidence_interval*OpticalTissueProperties.STANDARD_ANISOTROPY).all()
            assert (np.abs(volume['gamma'][volume['seg']==3]  - calculate_gruneisen_parameter_from_temperature(StandardProperties.BODY_TEMPERATURE_CELCIUS))<confidence_interval*calculate_gruneisen_parameter_from_temperature(
                StandardProperties.BODY_TEMPERATURE_CELCIUS)).all()
            assert (np.abs(volume['mua'][volume['seg']==3]  - spectrum.get_absorption_for_wavelength(wavelength))<confidence_interval*spectrum.get_absorption_for_wavelength(wavelength)).all()
            assert (np.abs(volume['sos'][volume['seg']==3]  - StandardProperties.SPEED_OF_SOUND_GENERIC)<confidence_interval*StandardProperties.SPEED_OF_SOUND_GENERIC).all()
        print('molecule ', MOLECULE_NAME, 'in setting 3 (vessel on vessel and background) ok ')


################################################
    def test_epidermal_scatterer(self):
        print("Simulating epidermal_scatterer")
        settings = set_settings()
        confidence_interval=0.01
        settings = Settings(settings)
        
        MOLECULE_NAME = 'epidermal_scatterer'
        print('Molecule: ', MOLECULE_NAME)
        spectrum = SPECTRAL_LIBRARY.CONSTANT_ABSORBER_ZERO

        #setting1: background of molecule
        molecule = MolecularCompositionGenerator().append(MOLECULE_LIBRARY.epidermal_scatterer()).get_molecular_composition(SegmentationClasses.MUSCLE)
        settings[Tags.STRUCTURES] = create_test_structure_of_molecule(settings, molecule, molecule, molecule, key='setting1')   
        #tests if mus, density, g, gamma, mua, and sos are equal to literature values in confidence interval  
        for wavelength in settings[Tags.WAVELENGTHS]:
            settings[Tags.WAVELENGTH] = wavelength
            volume_creator_adapter = ModelBasedVolumeCreator()
            volume = volume_creator_adapter.create_simulation_volume(settings)
   
            if wavelength == 500:
                assert (np.abs(volume['mus'] - OpticalTissueProperties.MUS500_EPIDERMIS)<confidence_interval*OpticalTissueProperties.MUS500_EPIDERMIS).all() 
            assert (np.abs(volume['density'] - StandardProperties.DENSITY_SKIN)<confidence_interval*StandardProperties.DENSITY_SKIN).all()
            assert (np.abs(volume['g'] - OpticalTissueProperties.STANDARD_ANISOTROPY)<confidence_interval*OpticalTissueProperties.STANDARD_ANISOTROPY).all()
            assert (np.abs(volume['gamma'] - calculate_gruneisen_parameter_from_temperature(
                StandardProperties.BODY_TEMPERATURE_CELCIUS))<confidence_interval*calculate_gruneisen_parameter_from_temperature(
                StandardProperties.BODY_TEMPERATURE_CELCIUS)).all()
            assert (np.abs(volume['mua'] - spectrum.get_absorption_for_wavelength(wavelength))<confidence_interval*spectrum.get_absorption_for_wavelength(wavelength)).all()
            assert (np.abs(volume['sos'] - StandardProperties.SPEED_OF_SOUND_SKIN)<confidence_interval*StandardProperties.SPEED_OF_SOUND_SKIN).all()
        print('molecule ', MOLECULE_NAME, 'in setting 1 (background only) ok ')

        #setting2: vessel of molecule
        molecule1 = MolecularCompositionGenerator().append(MOLECULE_LIBRARY.water()).get_molecular_composition(SegmentationClasses.MUSCLE)
        molecule2 = MolecularCompositionGenerator().append(MOLECULE_LIBRARY.epidermal_scatterer()).get_molecular_composition(SegmentationClasses.BLOOD)
        settings[Tags.STRUCTURES] = create_test_structure_of_molecule(settings, molecule1, molecule2, molecule2, key='setting2')

        #tests if mus, density, g, gamma, mua, and sos are equal to literature values in confidence interval            
        for wavelength in settings[Tags.WAVELENGTHS]:
            settings[Tags.WAVELENGTH] = wavelength
            volume_creator_adapter = ModelBasedVolumeCreator()
            volume = volume_creator_adapter.create_simulation_volume(settings)
           
            if wavelength == 500:
                assert (np.abs(volume['mus'][volume['seg']==3] - OpticalTissueProperties.MUS500_EPIDERMIS)<confidence_interval*OpticalTissueProperties.MUS500_EPIDERMIS).all() 
            assert (np.abs(volume['density'][volume['seg']==3] - StandardProperties.DENSITY_SKIN)<confidence_interval*StandardProperties.DENSITY_SKIN).all()
            assert (np.abs(volume['g'][volume['seg']==3] - OpticalTissueProperties.STANDARD_ANISOTROPY)<confidence_interval*OpticalTissueProperties.STANDARD_ANISOTROPY).all()
            assert (np.abs(volume['gamma'][volume['seg']==3] - calculate_gruneisen_parameter_from_temperature(
                StandardProperties.BODY_TEMPERATURE_CELCIUS))<confidence_interval*calculate_gruneisen_parameter_from_temperature(
                StandardProperties.BODY_TEMPERATURE_CELCIUS)).all()
            assert (np.abs(volume['mua'][volume['seg']==3] - spectrum.get_absorption_for_wavelength(wavelength))<confidence_interval*spectrum.get_absorption_for_wavelength(wavelength)).all()
            assert (np.abs(volume['sos'][volume['seg']==3] - StandardProperties.SPEED_OF_SOUND_SKIN)<confidence_interval*StandardProperties.SPEED_OF_SOUND_SKIN).all()
        print('molecule ', MOLECULE_NAME, 'in setting 2 (vessel on background) ok ')
    
        #setting3: vessel of molecule on top of vessel of molecule
        molecule1 = MolecularCompositionGenerator().append(MOLECULE_LIBRARY.water()).get_molecular_composition(SegmentationClasses.MUSCLE)
        molecule2 = MolecularCompositionGenerator().append(MOLECULE_LIBRARY.melanin()).get_molecular_composition(SegmentationClasses.EPIDERMIS)
        molecule3 = MolecularCompositionGenerator().append(MOLECULE_LIBRARY.epidermal_scatterer()).get_molecular_composition(SegmentationClasses.BLOOD)
        settings[Tags.STRUCTURES] = create_test_structure_of_molecule(settings, molecule1, molecule2, molecule3, key='setting3')

        #tests if mus, density, g, gamma, mua, and sos are equal to literature values in confidence interval            
        for wavelength in settings[Tags.WAVELENGTHS]:
            settings[Tags.WAVELENGTH] = wavelength
            volume_creator_adapter = ModelBasedVolumeCreator()
            volume = volume_creator_adapter.create_simulation_volume(settings)
           
            if wavelength == 500:
                assert (np.abs(volume['mus'][volume['seg']==3] - OpticalTissueProperties.MUS500_EPIDERMIS)<confidence_interval*OpticalTissueProperties.MUS500_EPIDERMIS).all() 
            assert (np.abs(volume['density'][volume['seg']==3] - StandardProperties.DENSITY_SKIN)<confidence_interval*StandardProperties.DENSITY_SKIN).all()
            assert (np.abs(volume['g'][volume['seg']==3] - OpticalTissueProperties.STANDARD_ANISOTROPY)<confidence_interval*OpticalTissueProperties.STANDARD_ANISOTROPY).all()
            assert (np.abs(volume['gamma'][volume['seg']==3] - calculate_gruneisen_parameter_from_temperature(
                StandardProperties.BODY_TEMPERATURE_CELCIUS))<confidence_interval*calculate_gruneisen_parameter_from_temperature(
                StandardProperties.BODY_TEMPERATURE_CELCIUS)).all()
            assert (np.abs(volume['mua'][volume['seg']==3] - spectrum.get_absorption_for_wavelength(wavelength))<confidence_interval*spectrum.get_absorption_for_wavelength(wavelength)).all()
            assert (np.abs(volume['sos'][volume['seg']==3] - StandardProperties.SPEED_OF_SOUND_SKIN)<confidence_interval*StandardProperties.SPEED_OF_SOUND_SKIN).all()
        print('molecule ', MOLECULE_NAME, 'in setting 3 (vessel on vessel and background) ok ')


################################################
    def test_dermal_scatterer(self):
        print("Simulating dermal_scatterer")
        settings = set_settings()
        confidence_interval=0.01
        settings = Settings(settings)

        MOLECULE_NAME = 'dermal_scatterer'
        print('Molecule: ', MOLECULE_NAME)        
        spectrum = SPECTRAL_LIBRARY.CONSTANT_ABSORBER_ZERO
            
        #setting1: background of molecule
        molecule = MolecularCompositionGenerator().append(MOLECULE_LIBRARY.dermal_scatterer()).get_molecular_composition(SegmentationClasses.MUSCLE)
        settings[Tags.STRUCTURES] = create_test_structure_of_molecule(settings, molecule, molecule, molecule, key='setting1')   
        #tests if mus, density, g, gamma, mua, and sos are equal to literature values in confidence interval  
        for wavelength in settings[Tags.WAVELENGTHS]:
            settings[Tags.WAVELENGTH] = wavelength
            volume_creator_adapter = ModelBasedVolumeCreator()
            volume = volume_creator_adapter.create_simulation_volume(settings)
            if wavelength == 500:
                assert (np.abs(volume['mus'] - OpticalTissueProperties.MUS500_DERMIS)<confidence_interval*OpticalTissueProperties.MUS500_DERMIS).all() 
            assert (np.abs(volume['density'] - StandardProperties.DENSITY_SKIN)<confidence_interval*StandardProperties.DENSITY_SKIN).all()
            assert (np.abs(volume['g'] - OpticalTissueProperties.STANDARD_ANISOTROPY)<confidence_interval*OpticalTissueProperties.STANDARD_ANISOTROPY).all()
            assert (np.abs(volume['gamma'] - calculate_gruneisen_parameter_from_temperature(
                StandardProperties.BODY_TEMPERATURE_CELCIUS))<confidence_interval*calculate_gruneisen_parameter_from_temperature(
                StandardProperties.BODY_TEMPERATURE_CELCIUS)).all()
            assert (np.abs(volume['mua'] - spectrum.get_absorption_for_wavelength(wavelength))<confidence_interval*spectrum.get_absorption_for_wavelength(wavelength)).all()
            assert (np.abs(volume['sos'] - StandardProperties.SPEED_OF_SOUND_SKIN)<confidence_interval*StandardProperties.SPEED_OF_SOUND_SKIN).all()
        print('molecule ', MOLECULE_NAME, 'in setting 1 (background only) ok ')

        #setting2: vessel of molecule
        molecule1 = MolecularCompositionGenerator().append(MOLECULE_LIBRARY.water()).get_molecular_composition(SegmentationClasses.MUSCLE)
        molecule2 = MolecularCompositionGenerator().append(MOLECULE_LIBRARY.dermal_scatterer()).get_molecular_composition(SegmentationClasses.BLOOD)
        settings[Tags.STRUCTURES] = create_test_structure_of_molecule(settings, molecule1, molecule2, molecule2, key='setting2')

        #tests if mus, density, g, gamma, mua, and sos are equal to literature values in confidence interval            
        for wavelength in settings[Tags.WAVELENGTHS]:
            settings[Tags.WAVELENGTH] = wavelength
            volume_creator_adapter = ModelBasedVolumeCreator()
            volume = volume_creator_adapter.create_simulation_volume(settings)
           
            if wavelength == 500:
                assert (np.abs(volume['mus'][volume['seg']==3] - OpticalTissueProperties.MUS500_DERMIS)<confidence_interval*OpticalTissueProperties.MUS500_DERMIS).all() 
            assert (np.abs(volume['density'][volume['seg']==3] - StandardProperties.DENSITY_SKIN)<confidence_interval*StandardProperties.DENSITY_SKIN).all()
            assert (np.abs(volume['g'][volume['seg']==3] - OpticalTissueProperties.STANDARD_ANISOTROPY)<confidence_interval*OpticalTissueProperties.STANDARD_ANISOTROPY).all()
            assert (np.abs(volume['gamma'][volume['seg']==3] - calculate_gruneisen_parameter_from_temperature(
                StandardProperties.BODY_TEMPERATURE_CELCIUS))<confidence_interval*calculate_gruneisen_parameter_from_temperature(
                StandardProperties.BODY_TEMPERATURE_CELCIUS)).all()
            assert (np.abs(volume['mua'][volume['seg']==3] - spectrum.get_absorption_for_wavelength(wavelength))<confidence_interval*spectrum.get_absorption_for_wavelength(wavelength)).all()
            assert (np.abs(volume['sos'][volume['seg']==3] - StandardProperties.SPEED_OF_SOUND_SKIN)<confidence_interval*StandardProperties.SPEED_OF_SOUND_SKIN).all()
        print('molecule ', MOLECULE_NAME, 'in setting 2 (vessel on background) ok ')
    
        #setting3: vessel of molecule on top of vessel of molecule
        molecule1 = MolecularCompositionGenerator().append(MOLECULE_LIBRARY.water()).get_molecular_composition(SegmentationClasses.MUSCLE)
        molecule2 = MolecularCompositionGenerator().append(MOLECULE_LIBRARY.melanin()).get_molecular_composition(SegmentationClasses.EPIDERMIS)
        molecule3 = MolecularCompositionGenerator().append(MOLECULE_LIBRARY.dermal_scatterer()).get_molecular_composition(SegmentationClasses.BLOOD)
        settings[Tags.STRUCTURES] = create_test_structure_of_molecule(settings, molecule1, molecule2, molecule3, key='setting3')

        #tests if mus, density, g, gamma, mua, and sos are equal to literature values in confidence interval            
        for wavelength in settings[Tags.WAVELENGTHS]:
            settings[Tags.WAVELENGTH] = wavelength
            volume_creator_adapter = ModelBasedVolumeCreator()
            volume = volume_creator_adapter.create_simulation_volume(settings)
           
            if wavelength == 500:
                assert (np.abs(volume['mus'][volume['seg']==3] - OpticalTissueProperties.MUS500_DERMIS)<confidence_interval*OpticalTissueProperties.MUS500_DERMIS).all() 
            assert (np.abs(volume['density'][volume['seg']==3] - StandardProperties.DENSITY_SKIN)<confidence_interval*StandardProperties.DENSITY_SKIN).all()
            assert (np.abs(volume['g'][volume['seg']==3] - OpticalTissueProperties.STANDARD_ANISOTROPY)<confidence_interval*OpticalTissueProperties.STANDARD_ANISOTROPY).all()
            assert (np.abs(volume['gamma'][volume['seg']==3] - calculate_gruneisen_parameter_from_temperature(
                StandardProperties.BODY_TEMPERATURE_CELCIUS))<confidence_interval*calculate_gruneisen_parameter_from_temperature(
                StandardProperties.BODY_TEMPERATURE_CELCIUS)).all()
            assert (np.abs(volume['mua'][volume['seg']==3] - spectrum.get_absorption_for_wavelength(wavelength))<confidence_interval*spectrum.get_absorption_for_wavelength(wavelength)).all()
            assert (np.abs(volume['sos'][volume['seg']==3] - StandardProperties.SPEED_OF_SOUND_SKIN)<confidence_interval*StandardProperties.SPEED_OF_SOUND_SKIN).all()
        print('molecule ', MOLECULE_NAME, 'in setting 3 (vessel on vessel and background) ok ')


################################################
    def test_bone(self):
        print("Simulating bone")
        settings = set_settings()
        confidence_interval = 0.01
        settings = Settings(settings)

        MOLECULE_NAME = 'bone'
        print('Molecule: ', MOLECULE_NAME)        
        spectrum = SPECTRAL_LIBRARY.CONSTANT_ABSORBER_ARBITRARY(OpticalTissueProperties.BONE_ABSORPTION)

        #setting1: background of molecule
        molecule = MolecularCompositionGenerator().append(MOLECULE_LIBRARY.bone())\
            .get_molecular_composition(SegmentationClasses.MUSCLE)
        settings[Tags.STRUCTURES] = create_test_structure_of_molecule(settings, molecule,
                                                                      molecule, molecule,
                                                                      key='setting1')
        #tests if mus, density, g, gamma, mua, and sos are equal to literature values in confidence interval  
        for wavelength in settings[Tags.WAVELENGTHS]:
            settings[Tags.WAVELENGTH] = wavelength
            volume_creator_adapter = ModelBasedVolumeCreator()
            volume = volume_creator_adapter.create_simulation_volume(settings)
   
            if wavelength == 500:
                assert (np.abs(volume['mus'] - OpticalTissueProperties.MUS500_BONE) <=
                        confidence_interval*OpticalTissueProperties.MUS500_BONE).all()
            assert (np.abs(volume['density'] - StandardProperties.DENSITY_BONE) <=
                    confidence_interval*StandardProperties.DENSITY_BONE).all()
            assert (np.abs(volume['g'] - OpticalTissueProperties.STANDARD_ANISOTROPY) <=
                    confidence_interval*OpticalTissueProperties.STANDARD_ANISOTROPY).all()
            assert (np.abs(volume['gamma'] - calculate_gruneisen_parameter_from_temperature(
                StandardProperties.BODY_TEMPERATURE_CELCIUS)) <=
                    confidence_interval*calculate_gruneisen_parameter_from_temperature(
                    StandardProperties.BODY_TEMPERATURE_CELCIUS)).all()
            assert (np.abs(volume['mua'] - spectrum.get_absorption_for_wavelength(wavelength)) <=
                    confidence_interval*spectrum.get_absorption_for_wavelength(wavelength)).all()
            assert (np.abs(volume['sos'] - StandardProperties.SPEED_OF_SOUND_BONE) <=
                    confidence_interval*StandardProperties.SPEED_OF_SOUND_BONE).all()
        print('molecule ', MOLECULE_NAME, 'in setting 1 (background only) ok ')

        #setting2: vessel of molecule
        molecule1 = MolecularCompositionGenerator().append(MOLECULE_LIBRARY.water()).get_molecular_composition(SegmentationClasses.MUSCLE)
        molecule2 = MolecularCompositionGenerator().append(MOLECULE_LIBRARY.bone()).get_molecular_composition(SegmentationClasses.BLOOD)
        settings[Tags.STRUCTURES] = create_test_structure_of_molecule(settings, molecule1, molecule2, molecule2, key='setting2')

        for wavelength in settings[Tags.WAVELENGTHS]:
            settings[Tags.WAVELENGTH] = wavelength
            volume_creator_adapter = ModelBasedVolumeCreator()
            volume = volume_creator_adapter.create_simulation_volume(settings)
           
            if wavelength == 500:
                assert (np.abs(volume['mus'][volume['seg'] == 3] - OpticalTissueProperties.MUS500_BONE) <
                        confidence_interval*OpticalTissueProperties.MUS500_BONE).all()
            assert (np.abs(volume['density'][volume['seg'] == 3] - StandardProperties.DENSITY_BONE) <
                    confidence_interval*StandardProperties.DENSITY_BONE).all()
            assert (np.abs(volume['g'][volume['seg'] == 3] - OpticalTissueProperties.STANDARD_ANISOTROPY) <
                    confidence_interval*OpticalTissueProperties.STANDARD_ANISOTROPY).all()
            assert (np.abs(volume['gamma'][volume['seg'] == 3] - calculate_gruneisen_parameter_from_temperature(
                StandardProperties.BODY_TEMPERATURE_CELCIUS)) <
                    confidence_interval*calculate_gruneisen_parameter_from_temperature(
                StandardProperties.BODY_TEMPERATURE_CELCIUS)).all()
            assert (np.abs(volume['mua'][volume['seg'] == 3] - spectrum.get_absorption_for_wavelength(wavelength)) <
                    confidence_interval*spectrum.get_absorption_for_wavelength(wavelength)).all()
            assert (np.abs(volume['sos'][volume['seg'] == 3] - StandardProperties.SPEED_OF_SOUND_BONE) <
                    confidence_interval*StandardProperties.SPEED_OF_SOUND_BONE).all()
        print('molecule ', MOLECULE_NAME, 'in setting 2 (vessel on background) ok ')
    
        #setting3: vessel of molecule on top of vessel of molecule
        molecule1 = MolecularCompositionGenerator().append(MOLECULE_LIBRARY.water()).get_molecular_composition(SegmentationClasses.MUSCLE)
        molecule2 = MolecularCompositionGenerator().append(MOLECULE_LIBRARY.melanin()).get_molecular_composition(SegmentationClasses.EPIDERMIS)
        molecule3 = MolecularCompositionGenerator().append(MOLECULE_LIBRARY.bone()).get_molecular_composition(SegmentationClasses.BLOOD)
        settings[Tags.STRUCTURES] = create_test_structure_of_molecule(settings, molecule1, molecule2, molecule3, key='setting3')

        for wavelength in settings[Tags.WAVELENGTHS]:
            settings[Tags.WAVELENGTH] = wavelength
            volume_creator_adapter = ModelBasedVolumeCreator()
            volume = volume_creator_adapter.create_simulation_volume(settings)
           
            if wavelength == 500:
                assert (np.abs(volume['mus'][volume['seg'] == 3] - OpticalTissueProperties.MUS500_BONE) <
                        confidence_interval*OpticalTissueProperties.MUS500_BONE).all()
            assert (np.abs(volume['density'][volume['seg'] == 3] - StandardProperties.DENSITY_BONE) <
                    confidence_interval*StandardProperties.DENSITY_BONE).all()
            assert (np.abs(volume['g'][volume['seg'] == 3] - OpticalTissueProperties.STANDARD_ANISOTROPY) <
                    confidence_interval*OpticalTissueProperties.STANDARD_ANISOTROPY).all()
            assert (np.abs(volume['gamma'][volume['seg'] == 3] - calculate_gruneisen_parameter_from_temperature(
                StandardProperties.BODY_TEMPERATURE_CELCIUS)) <
                    confidence_interval*calculate_gruneisen_parameter_from_temperature(
                    StandardProperties.BODY_TEMPERATURE_CELCIUS)).all()
            assert (np.abs(volume['mua'][volume['seg'] == 3] - spectrum.get_absorption_for_wavelength(wavelength)) <
                    confidence_interval*spectrum.get_absorption_for_wavelength(wavelength)).all()
            assert (np.abs(volume['sos'][volume['seg'] == 3] - StandardProperties.SPEED_OF_SOUND_BONE) <
                    confidence_interval*StandardProperties.SPEED_OF_SOUND_BONE).all()
        print('molecule ', MOLECULE_NAME, 'in setting 3 (vessel on vessel and background) ok ')


    def test_mediprene(self):
        print("Simulating mediprene")
        settings = set_settings()
        confidence_interval=0.01
        settings = Settings(settings)
        
        MOLECULE_NAME = 'mediprene'
        print('Molecule: ', MOLECULE_NAME)
        spectrum = SPECTRAL_LIBRARY.CONSTANT_ABSORBER_ARBITRARY(-np.log(0.85) / 10)
        
        #setting1: background of molecule
        molecule = MolecularCompositionGenerator().append(MOLECULE_LIBRARY.mediprene()).get_molecular_composition(SegmentationClasses.MUSCLE)
        settings[Tags.STRUCTURES] = create_test_structure_of_molecule(settings, molecule, molecule, molecule, key='setting1')      
        #tests if mus, density, g, gamma, mua, and sos are equal to literature values in confidence interval  
        for wavelength in settings[Tags.WAVELENGTHS]:
            settings[Tags.WAVELENGTH] = wavelength
            volume_creator_adapter = ModelBasedVolumeCreator()
            volume = volume_creator_adapter.create_simulation_volume(settings)
   
            if wavelength == 500:
                assert (np.abs(volume['mus'] - ((-np.log(0.85)) - (-np.log(0.85) / 10)))<confidence_interval*((-np.log(0.85)) - (-np.log(0.85) / 10))).all() 
            assert (np.abs(volume['density'] - StandardProperties.DENSITY_GEL_PAD)<confidence_interval*StandardProperties.DENSITY_GEL_PAD).all()
            assert (np.abs(volume['g'] - OpticalTissueProperties.STANDARD_ANISOTROPY)<confidence_interval*OpticalTissueProperties.STANDARD_ANISOTROPY).all()
            assert (np.abs(volume['gamma'] - calculate_gruneisen_parameter_from_temperature(
                StandardProperties.BODY_TEMPERATURE_CELCIUS))<confidence_interval*calculate_gruneisen_parameter_from_temperature(
                StandardProperties.BODY_TEMPERATURE_CELCIUS)).all()
            assert (np.abs(volume['mua'] - spectrum.get_absorption_for_wavelength(wavelength))<confidence_interval*spectrum.get_absorption_for_wavelength(wavelength)).all()
            assert (np.abs(volume['sos'] - StandardProperties.SPEED_OF_SOUND_GEL_PAD)<confidence_interval*StandardProperties.SPEED_OF_SOUND_GEL_PAD).all()
        print('molecule ', MOLECULE_NAME, 'in setting 1 (background only) ok ')

        #setting2: vessel of molecule
        molecule1 = MolecularCompositionGenerator().append(MOLECULE_LIBRARY.water()).get_molecular_composition(SegmentationClasses.MUSCLE)
        molecule2 = MolecularCompositionGenerator().append(MOLECULE_LIBRARY.mediprene()).get_molecular_composition(SegmentationClasses.BLOOD)
        settings[Tags.STRUCTURES] = create_test_structure_of_molecule(settings, molecule1, molecule2, molecule2, key='setting2')

        #tests if mus, density, g, gamma, mua, and sos are equal to literature values in confidence interval            
        for wavelength in settings[Tags.WAVELENGTHS]:
            settings[Tags.WAVELENGTH] = wavelength
            volume_creator_adapter = ModelBasedVolumeCreator()
            volume = volume_creator_adapter.create_simulation_volume(settings)
           
            if wavelength == 500:
                assert (np.abs(volume['mus'][volume['seg']==3] - ((-np.log(0.85)) - (-np.log(0.85) / 10)))<confidence_interval*((-np.log(0.85)) - (-np.log(0.85) / 10))).all() 
            assert (np.abs(volume['density'][volume['seg']==3] - StandardProperties.DENSITY_GEL_PAD)<confidence_interval*StandardProperties.DENSITY_GEL_PAD).all()
            assert (np.abs(volume['g'][volume['seg']==3] - OpticalTissueProperties.STANDARD_ANISOTROPY)<confidence_interval*OpticalTissueProperties.STANDARD_ANISOTROPY).all()
            assert (np.abs(volume['gamma'][volume['seg']==3] - calculate_gruneisen_parameter_from_temperature(
                StandardProperties.BODY_TEMPERATURE_CELCIUS))<confidence_interval*calculate_gruneisen_parameter_from_temperature(
                StandardProperties.BODY_TEMPERATURE_CELCIUS)).all()
            assert (np.abs(volume['mua'][volume['seg']==3] - spectrum.get_absorption_for_wavelength(wavelength))<confidence_interval*spectrum.get_absorption_for_wavelength(wavelength)).all()
            assert (np.abs(volume['sos'][volume['seg']==3] - StandardProperties.SPEED_OF_SOUND_GEL_PAD)<confidence_interval*StandardProperties.SPEED_OF_SOUND_GEL_PAD).all()
        print('molecule ', MOLECULE_NAME, 'in setting 2 (vessel on background) ok ')
    
        #setting3: vessel of molecule on top of vessel of molecule
        molecule1 = MolecularCompositionGenerator().append(MOLECULE_LIBRARY.water()).get_molecular_composition(SegmentationClasses.MUSCLE)
        molecule2 = MolecularCompositionGenerator().append(MOLECULE_LIBRARY.melanin()).get_molecular_composition(SegmentationClasses.EPIDERMIS)
        molecule3 = MolecularCompositionGenerator().append(MOLECULE_LIBRARY.mediprene()).get_molecular_composition(SegmentationClasses.BLOOD)
        settings[Tags.STRUCTURES] = create_test_structure_of_molecule(settings, molecule1, molecule2, molecule3, key='setting3')

        #tests if mus, density, g, gamma, mua, and sos are equal to literature values in confidence interval            
        for wavelength in settings[Tags.WAVELENGTHS]:
            settings[Tags.WAVELENGTH] = wavelength
            volume_creator_adapter = ModelBasedVolumeCreator()
            volume = volume_creator_adapter.create_simulation_volume(settings)
           
            if wavelength == 500:
                assert (np.abs(volume['mus'][volume['seg']==3] - ((-np.log(0.85)) - (-np.log(0.85) / 10)))<confidence_interval*((-np.log(0.85)) - (-np.log(0.85) / 10))).all() 
            assert (np.abs(volume['density'][volume['seg']==3] - StandardProperties.DENSITY_GEL_PAD)<confidence_interval*StandardProperties.DENSITY_GEL_PAD).all()
            assert (np.abs(volume['g'][volume['seg']==3] - OpticalTissueProperties.STANDARD_ANISOTROPY)<confidence_interval*OpticalTissueProperties.STANDARD_ANISOTROPY).all()
            assert (np.abs(volume['gamma'][volume['seg']==3] - calculate_gruneisen_parameter_from_temperature(
                StandardProperties.BODY_TEMPERATURE_CELCIUS))<confidence_interval*calculate_gruneisen_parameter_from_temperature(
                StandardProperties.BODY_TEMPERATURE_CELCIUS)).all()
            assert (np.abs(volume['mua'][volume['seg']==3] - spectrum.get_absorption_for_wavelength(wavelength))<confidence_interval*spectrum.get_absorption_for_wavelength(wavelength)).all()
            assert (np.abs(volume['sos'][volume['seg']==3] - StandardProperties.SPEED_OF_SOUND_GEL_PAD)<confidence_interval*StandardProperties.SPEED_OF_SOUND_GEL_PAD).all()
        print('molecule ', MOLECULE_NAME, 'in setting 3 (vessel on vessel and background) ok ')


################################################
    def test_heavy_water(self):
        print("Simulating heavy_water")
        settings = set_settings()
        confidence_interval=0.01
        settings = Settings(settings)

        MOLECULE_NAME = 'heavy_water'
        print('Molecule: ', MOLECULE_NAME)        
        spectrum = SPECTRAL_LIBRARY.CONSTANT_ABSORBER_ARBITRARY(StandardProperties.HEAVY_WATER_MUA)
        
        #setting1: background of molecule
        molecule = MolecularCompositionGenerator().append(MOLECULE_LIBRARY.heavy_water()).get_molecular_composition(SegmentationClasses.MUSCLE)
        settings[Tags.STRUCTURES] = create_test_structure_of_molecule(settings, molecule, molecule, molecule, key='setting1')   
        #tests if mus, density, g, gamma, mua, and sos are equal to literature values in confidence interval  
        for wavelength in settings[Tags.WAVELENGTHS]:
            settings[Tags.WAVELENGTH] = wavelength
            volume_creator_adapter = ModelBasedVolumeCreator()
            volume = volume_creator_adapter.create_simulation_volume(settings)

            if wavelength == 500:
                assert (np.abs(volume['mus'] - StandardProperties.WATER_MUS)<confidence_interval*StandardProperties.WATER_MUS).all() 
            assert (np.abs(volume['density'] - StandardProperties.DENSITY_HEAVY_WATER)<confidence_interval*StandardProperties.DENSITY_HEAVY_WATER).all()
            assert (np.abs(volume['g'] - StandardProperties.WATER_G)<confidence_interval*StandardProperties.WATER_G).all()
            assert (np.abs(volume['gamma'] - calculate_gruneisen_parameter_from_temperature(
                StandardProperties.BODY_TEMPERATURE_CELCIUS))<confidence_interval*calculate_gruneisen_parameter_from_temperature(
                StandardProperties.BODY_TEMPERATURE_CELCIUS)).all()
            assert (np.abs(volume['mua'] - spectrum.get_absorption_for_wavelength(wavelength))<confidence_interval*spectrum.get_absorption_for_wavelength(wavelength)).all()
            assert (np.abs(volume['sos'] - StandardProperties.SPEED_OF_SOUND_HEAVY_WATER)<confidence_interval*StandardProperties.SPEED_OF_SOUND_HEAVY_WATER).all()
        print('molecule ', MOLECULE_NAME, 'in setting 1 (background only) ok ')

        #setting2: vessel of molecule
        molecule1 = MolecularCompositionGenerator().append(MOLECULE_LIBRARY.water()).get_molecular_composition(SegmentationClasses.MUSCLE)
        molecule2 = MolecularCompositionGenerator().append(MOLECULE_LIBRARY.heavy_water()).get_molecular_composition(SegmentationClasses.BLOOD)
        settings[Tags.STRUCTURES] = create_test_structure_of_molecule(settings, molecule1, molecule2, molecule2, key='setting2')

        #tests if mus, density, g, gamma, mua, and sos are equal to literature values in confidence interval            
        for wavelength in settings[Tags.WAVELENGTHS]:
            settings[Tags.WAVELENGTH] = wavelength
            volume_creator_adapter = ModelBasedVolumeCreator()
            volume = volume_creator_adapter.create_simulation_volume(settings)
           
            if wavelength == 500:
                assert (np.abs(volume['mus'][volume['seg']==3] - StandardProperties.WATER_MUS)<confidence_interval*StandardProperties.WATER_MUS).all() 
            assert (np.abs(volume['density'][volume['seg']==3] - StandardProperties.DENSITY_HEAVY_WATER)<confidence_interval*StandardProperties.DENSITY_HEAVY_WATER).all()
            assert (np.abs(volume['g'][volume['seg']==3] - StandardProperties.WATER_G)<confidence_interval*StandardProperties.WATER_G).all()
            assert (np.abs(volume['gamma'][volume['seg']==3] - calculate_gruneisen_parameter_from_temperature(
                StandardProperties.BODY_TEMPERATURE_CELCIUS))<confidence_interval*calculate_gruneisen_parameter_from_temperature(
                StandardProperties.BODY_TEMPERATURE_CELCIUS)).all()
            assert (np.abs(volume['mua'][volume['seg']==3] - spectrum.get_absorption_for_wavelength(wavelength))<confidence_interval*spectrum.get_absorption_for_wavelength(wavelength)).all()
            assert (np.abs(volume['sos'][volume['seg']==3] - StandardProperties.SPEED_OF_SOUND_HEAVY_WATER)<confidence_interval*StandardProperties.SPEED_OF_SOUND_HEAVY_WATER).all()
        print('molecule ', MOLECULE_NAME, 'in setting 2 (vessel on background) ok ')
    
        #setting3: vessel of molecule on top of vessel of molecule
        molecule1 = MolecularCompositionGenerator().append(MOLECULE_LIBRARY.water()).get_molecular_composition(SegmentationClasses.MUSCLE)
        molecule2 = MolecularCompositionGenerator().append(MOLECULE_LIBRARY.melanin()).get_molecular_composition(SegmentationClasses.EPIDERMIS)
        molecule3 = MolecularCompositionGenerator().append(MOLECULE_LIBRARY.heavy_water()).get_molecular_composition(SegmentationClasses.BLOOD)
        settings[Tags.STRUCTURES] = create_test_structure_of_molecule(settings, molecule1, molecule2, molecule3, key='setting3')

        #tests if mus, density, g, gamma, mua, and sos are equal to literature values in confidence interval            
        for wavelength in settings[Tags.WAVELENGTHS]:
            settings[Tags.WAVELENGTH] = wavelength
            volume_creator_adapter = ModelBasedVolumeCreator()
            volume = volume_creator_adapter.create_simulation_volume(settings)
           
            if wavelength == 500:
                assert (np.abs(volume['mus'][volume['seg']==3] - StandardProperties.WATER_MUS)<confidence_interval*StandardProperties.WATER_MUS).all() 
            assert (np.abs(volume['density'][volume['seg']==3] - StandardProperties.DENSITY_HEAVY_WATER)<confidence_interval*StandardProperties.DENSITY_HEAVY_WATER).all()
            assert (np.abs(volume['g'][volume['seg']==3] - StandardProperties.WATER_G)<confidence_interval*StandardProperties.WATER_G).all()
            assert (np.abs(volume['gamma'][volume['seg']==3] - calculate_gruneisen_parameter_from_temperature(
                StandardProperties.BODY_TEMPERATURE_CELCIUS))<confidence_interval*calculate_gruneisen_parameter_from_temperature(
                StandardProperties.BODY_TEMPERATURE_CELCIUS)).all()
            assert (np.abs(volume['mua'][volume['seg']==3] - spectrum.get_absorption_for_wavelength(wavelength))<confidence_interval*spectrum.get_absorption_for_wavelength(wavelength)).all()
            assert (np.abs(volume['sos'][volume['seg']==3] - StandardProperties.SPEED_OF_SOUND_HEAVY_WATER)<confidence_interval*StandardProperties.SPEED_OF_SOUND_HEAVY_WATER).all()
        print('molecule ', MOLECULE_NAME, 'in setting 3 (vessel on vessel and background) ok ')


################################################
    def test_air(self):
        print("Simulating air")
        settings = set_settings()
        confidence_interval=0.01
        settings = Settings(settings)

        MOLECULE_NAME = 'air'
        print('Molecule: ', MOLECULE_NAME)    
        spectrum = SPECTRAL_LIBRARY.CONSTANT_ABSORBER_ARBITRARY(StandardProperties.AIR_MUA)

        #setting1: background of molecule
        molecule = MolecularCompositionGenerator().append(MOLECULE_LIBRARY.air()).get_molecular_composition(SegmentationClasses.MUSCLE)
        settings[Tags.STRUCTURES] = create_test_structure_of_molecule(settings, molecule, molecule, molecule, key='setting1')   
        #tests if mus, density, g, gamma, mua, and sos are equal to literature values in confidence interval  
        for wavelength in settings[Tags.WAVELENGTHS]:
            settings[Tags.WAVELENGTH] = wavelength
            volume_creator_adapter = ModelBasedVolumeCreator()
            volume = volume_creator_adapter.create_simulation_volume(settings)

            if wavelength == 500:
                assert (np.abs(volume['mus'] - StandardProperties.AIR_MUS)<confidence_interval*StandardProperties.AIR_MUS).all() 
            assert (np.abs(volume['density'] - StandardProperties.DENSITY_AIR)<confidence_interval*StandardProperties.DENSITY_AIR).all()
            assert (np.abs(volume['g'] - StandardProperties.AIR_G)<confidence_interval*StandardProperties.AIR_G).all()
            assert (np.abs(volume['gamma'] - calculate_gruneisen_parameter_from_temperature(
                StandardProperties.BODY_TEMPERATURE_CELCIUS))<confidence_interval*calculate_gruneisen_parameter_from_temperature(
                StandardProperties.BODY_TEMPERATURE_CELCIUS)).all()
            assert (np.abs(volume['mua'] - spectrum.get_absorption_for_wavelength(wavelength))<confidence_interval*spectrum.get_absorption_for_wavelength(wavelength)).all()
            assert (np.abs(volume['sos'] - StandardProperties.SPEED_OF_SOUND_AIR)<confidence_interval*StandardProperties.SPEED_OF_SOUND_AIR).all()
        print('molecule ', MOLECULE_NAME, 'in setting 1 (background only) ok ')

        #setting2: vessel of molecule
        molecule1 = MolecularCompositionGenerator().append(MOLECULE_LIBRARY.water()).get_molecular_composition(SegmentationClasses.MUSCLE)
        molecule2 = MolecularCompositionGenerator().append(MOLECULE_LIBRARY.air()).get_molecular_composition(SegmentationClasses.BLOOD)
        settings[Tags.STRUCTURES] = create_test_structure_of_molecule(settings, molecule1, molecule2, molecule2, key='setting2')

        #tests if mus, density, g, gamma, mua, and sos are equal to literature values in confidence interval            
        for wavelength in settings[Tags.WAVELENGTHS]:
            settings[Tags.WAVELENGTH] = wavelength
            volume_creator_adapter = ModelBasedVolumeCreator()
            volume = volume_creator_adapter.create_simulation_volume(settings)
           
            if wavelength == 500:
                assert (np.abs(volume['mus'][volume['seg']==3] - StandardProperties.AIR_MUS)<confidence_interval*StandardProperties.AIR_MUS).all() 
            assert (np.abs(volume['density'][volume['seg']==3] - StandardProperties.DENSITY_AIR)<confidence_interval*StandardProperties.DENSITY_AIR).all()
            assert (np.abs(volume['g'][volume['seg']==3] - StandardProperties.AIR_G)<confidence_interval*StandardProperties.AIR_G).all()
            assert (np.abs(volume['gamma'][volume['seg']==3] - calculate_gruneisen_parameter_from_temperature(
                StandardProperties.BODY_TEMPERATURE_CELCIUS))<confidence_interval*calculate_gruneisen_parameter_from_temperature(
                StandardProperties.BODY_TEMPERATURE_CELCIUS)).all()
            assert (np.abs(volume['mua'][volume['seg']==3] - spectrum.get_absorption_for_wavelength(wavelength))<confidence_interval*spectrum.get_absorption_for_wavelength(wavelength)).all()
            assert (np.abs(volume['sos'][volume['seg']==3] - StandardProperties.SPEED_OF_SOUND_AIR)<confidence_interval*StandardProperties.SPEED_OF_SOUND_AIR).all()
        print('molecule ', MOLECULE_NAME, 'in setting 2 (vessel on background) ok ')
    
        #setting3: vessel of molecule on top of vessel of molecule
        molecule1 = MolecularCompositionGenerator().append(MOLECULE_LIBRARY.water()).get_molecular_composition(SegmentationClasses.MUSCLE)
        molecule2 = MolecularCompositionGenerator().append(MOLECULE_LIBRARY.melanin()).get_molecular_composition(SegmentationClasses.EPIDERMIS)
        molecule3 = MolecularCompositionGenerator().append(MOLECULE_LIBRARY.air()).get_molecular_composition(SegmentationClasses.BLOOD)
        settings[Tags.STRUCTURES] = create_test_structure_of_molecule(settings, molecule1, molecule2, molecule3, key='setting3')

        #tests if mus, density, g, gamma, mua, and sos are equal to literature values in confidence interval            
        for wavelength in settings[Tags.WAVELENGTHS]:
            settings[Tags.WAVELENGTH] = wavelength
            volume_creator_adapter = ModelBasedVolumeCreator()
            volume = volume_creator_adapter.create_simulation_volume(settings)
           
            if wavelength == 500:
                assert (np.abs(volume['mus'][volume['seg']==3] - StandardProperties.AIR_MUS)<confidence_interval*StandardProperties.AIR_MUS).all() 
            assert (np.abs(volume['density'][volume['seg']==3] - StandardProperties.DENSITY_AIR)<confidence_interval*StandardProperties.DENSITY_AIR).all()
            assert (np.abs(volume['g'][volume['seg']==3] - StandardProperties.AIR_G)<confidence_interval*StandardProperties.AIR_G).all()
            assert (np.abs(volume['gamma'][volume['seg']==3] - calculate_gruneisen_parameter_from_temperature(
                StandardProperties.BODY_TEMPERATURE_CELCIUS))<confidence_interval*calculate_gruneisen_parameter_from_temperature(
                StandardProperties.BODY_TEMPERATURE_CELCIUS)).all()
            assert (np.abs(volume['mua'][volume['seg']==3] - spectrum.get_absorption_for_wavelength(wavelength))<confidence_interval*spectrum.get_absorption_for_wavelength(wavelength)).all()
            assert (np.abs(volume['sos'][volume['seg']==3] - StandardProperties.SPEED_OF_SOUND_AIR)<confidence_interval*StandardProperties.SPEED_OF_SOUND_AIR).all()
        print('molecule ', MOLECULE_NAME, 'in setting 3 (vessel on vessel and background) ok ')