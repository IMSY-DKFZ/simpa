from simpa import Tags
import simpa as sp
import os
import numpy as np

def create_background():
    """
    only background
    """
    background_dictionary = sp.Settings()
    water = sp.MolecularCompositionGenerator()
    water.append(sp.MoleculeLibrary.water())
    background_dictionary[Tags.MOLECULE_COMPOSITION] = water.get_molecular_composition(segmentation_type=8)# sp.TISSUE_LIBRARY.constant(1e-10, 1e-10, 1.0)
    background_dictionary[Tags.STRUCTURE_TYPE] = Tags.BACKGROUND

    tissue_dict = sp.Settings()
    tissue_dict[Tags.BACKGROUND] = background_dictionary
    return tissue_dict

def get_reconstruction_settings(matlab_path, spacing):
    return {
            Tags.RECONSTRUCTION_PERFORM_BANDPASS_FILTERING: True, # makes sense
            Tags.ACOUSTIC_MODEL_BINARY_PATH: matlab_path,
            # Tags.ACOUSTIC_SIMULATION_3D: True,
            Tags.KWAVE_PROPERTY_ALPHA_POWER: 1.05, # NOT DEFAULT VALUE (DEFAULT IS 0.0) TODO ASK
            Tags.TUKEY_WINDOW_ALPHA: 0.5, # default: 0.5
            Tags.BANDPASS_CUTOFF_LOWPASS: int(8e6), # default: int(8e6)
            Tags.BANDPASS_CUTOFF_HIGHPASS: int(0.1e6), # default: int(0.1e6)
            Tags.RECONSTRUCTION_BMODE_AFTER_RECONSTRUCTION: True, # makes sense
            Tags.RECONSTRUCTION_BMODE_METHOD: Tags.RECONSTRUCTION_BMODE_METHOD_HILBERT_TRANSFORM, # makes sense
            Tags.RECONSTRUCTION_APODIZATION_METHOD: Tags.RECONSTRUCTION_APODIZATION_BOX, # makes sense
            Tags.RECONSTRUCTION_MODE: Tags.RECONSTRUCTION_MODE_PRESSURE, #Tags.RECONSTRUCTION_MODE_DIFFERENTIAL
            Tags.KWAVE_PROPERTY_SENSOR_RECORD: "p", # default: 'p'
            Tags.KWAVE_PROPERTY_PlotPML: False, # default: False
            Tags.RECORDMOVIE: False, # default: False
            Tags.MOVIENAME: "visualization_log", # default: 'visualization_log'
            Tags.ACOUSTIC_LOG_SCALE: True, # default: True
            Tags.KWAVE_PROPERTY_PMLInside: False, # default: False
            Tags.KWAVE_PROPERTY_PMLSize: [31, 32], # TODO ASK
            Tags.KWAVE_PROPERTY_PMLAlpha: 1.5, # default: 1.5
            Tags.DATA_FIELD_SPEED_OF_SOUND: 1540, # makes sense
            Tags.SPACING_MM: spacing, # default: 0.5
            Tags.MODEL_SENSOR_FREQUENCY_RESPONSE: False # TODO ASK
        }

def get_noise_data(data_path: str, wavelenghts: list, number_of_simulations: int):
    noise_data = np.load(os.path.join(data_path), allow_pickle=True)
    noise_data = noise_data.item()
    noise_data_wavelenghts = [key for key in noise_data.keys() if type(key) == int] # only integer keys are wavelengts
    n_samples = []
    for wv in list(set(wavelenghts) & set(noise_data_wavelenghts)):
        assert noise_data[wv]["offsets"].shape[1] == noise_data[wv]["thermal_noises"].shape[1] \
            and noise_data[wv]["membrane_peaks"].shape == noise_data[wv]["energies"].shape \
            and len(noise_data[wv]["membrane_peaks"]) == noise_data[wv]["offsets"].shape[1]
        n_samples.append(np.shape(noise_data[wv]["offsets"])[1]) # assumes that second dimension of offsets are the samples #TODO generalize
    n_samples_min = np.min(n_samples) # read out the minimum number of samples for all wavelenghts that are stored in the data and are wanted
    sample_list = np.random.choice(list(range(n_samples_min)), size=number_of_simulations)
    return noise_data, sample_list

class NoiseExperiment():

    def __init__(self, experiment_name: str, wavelengths: list, number_of_simulations: int):
        self.experiment = experiment_name
        self.wavelengths = wavelengths
        self.number_of_simulations = number_of_simulations
    
    def generate_sample_list_and_load_noise_data(self, device: int):
        if self.experiment == "water_experiment" \
           or self.experiment == "water_experiment_only_offsets"\
           or self.experiment == "water_experiment_only_thermalnoise"\
           or self.experiment == "water_experiment_only_laserenergy":
            IN_VITRO_DATA_PATH = "/home/c738r/data/real_data/PreProcessedData/20221027_DemineralizedWaterbath/offsets_thermalnoises_800nm.npy"
            NOISE_DATA_PATH = "/home/c738r/data/real_data/PreProcessedData/20220907_2D-PAD_Erlangen/offsets_thermalnoises_800nm.npy"
            if device == 1:
                self.in_vitro_data, self.sample_list = get_noise_data(IN_VITRO_DATA_PATH, self.wavelengths, self.number_of_simulations)
            elif device == 2:
                self.noise_data_pad, self.sample_list = get_noise_data(NOISE_DATA_PATH,  self.wavelengths, self.number_of_simulations)
        else:
            raise(Exception("Not specified noise data for given Experiment"))
        return self.sample_list

    def get_noise_parameters(self, device:int, wavelength, sample_index: int):
        """
        :param device: device index
        :param wavelength: wavelength in nm
        :param n: sample index
        """
        if self.experiment == "water_experiment":
            if device == 1: # Using waterbath measurements acquired by REZ-device
                return {
                    Tags.TRANSFORM_TO_IN_VITRO_DOMAIN: True,
                    Tags.SCALING_FACTOR: self.in_vitro_data[wavelength]["membrane_peaks"][sample_index],
                    Tags.IN_VITRO_LASER_ENERGY: self.in_vitro_data[wavelength]["energies"][sample_index],
                    Tags.BROKEN_SENSORS: np.array([30, 94, 145]),
                    Tags.OFFSETS: self.in_vitro_data[wavelength]["offsets"][:,sample_index],
                    Tags.THERMAL_NOISES: self.in_vitro_data[wavelength]["thermal_noises"][:,sample_index]
                    }
            elif device==2: # Using 2D-PAD-Data acquired by Erlangen-device
                return {
                    Tags.TRANSFORM_TO_IN_VITRO_DOMAIN: True,
                    Tags.SCALING_FACTOR: self.noise_data_pad[wavelength]["membrane_peaks"][sample_index], #mean value of device 1
                    Tags.IN_VITRO_LASER_ENERGY: self.noise_data_pad[wavelength]["energies"][sample_index], #mean value of device 1
                    Tags.BROKEN_SENSORS: np.array([30, 48, 47, 158]),
                    Tags.OFFSETS: self.noise_data_pad[wavelength]["offsets"][:,sample_index],
                    Tags.THERMAL_NOISES: self.noise_data_pad[wavelength]["thermal_noises"][:,sample_index]
                    }
        elif self.experiment == "water_experiment_only_offsets":
            if device == 1: # Using waterbath measurements acquired by REZ-device BUT ONLY OFFSETS
                return {
                    Tags.TRANSFORM_TO_IN_VITRO_DOMAIN: True,
                    Tags.SCALING_FACTOR: self.in_vitro_data[wavelength]["membrane_peaks"][sample_index],
                    Tags.IN_VITRO_LASER_ENERGY: self.in_vitro_data[wavelength]["energies"][sample_index],
                    #Tags.BROKEN_SENSORS: np.array([30, 94, 145]),
                    Tags.OFFSETS: self.in_vitro_data[wavelength]["offsets"][:,sample_index]
                    #Tags.THERMAL_NOISES: self.in_vitro_data[wavelength]["thermal_noises"][:,sample_index]
                    }
            elif device==2: # Using 2D-PAD-Data acquired by Erlangen-device BUT ONLY OFFSETS
                return {
                    Tags.TRANSFORM_TO_IN_VITRO_DOMAIN: True,
                    Tags.SCALING_FACTOR: self.noise_data_pad[wavelength]["membrane_peaks"][sample_index], #mean value of device 1
                    Tags.IN_VITRO_LASER_ENERGY: self.noise_data_pad[wavelength]["energies"][sample_index], #mean value of device 1
                    #Tags.BROKEN_SENSORS: np.array([30, 48, 47, 158]),
                    Tags.OFFSETS: self.noise_data_pad[wavelength]["offsets"][:,sample_index]
                    #Tags.THERMAL_NOISES: self.noise_data_pad[wavelength]["thermal_noises"][:,sample_index]
                    }
        elif self.experiment == "water_experiment_only_thermalnoise":
            if device == 1: # Using waterbath measurements acquired by REZ-device BUT ONLY OFFSETS
                return {
                    Tags.TRANSFORM_TO_IN_VITRO_DOMAIN: True,
                    Tags.SCALING_FACTOR: self.in_vitro_data[wavelength]["membrane_peaks"][sample_index],
                    Tags.IN_VITRO_LASER_ENERGY: self.in_vitro_data[wavelength]["energies"][sample_index],
                    #Tags.BROKEN_SENSORS: np.array([30, 94, 145]),
                    #Tags.OFFSETS: self.in_vitro_data[wavelength]["offsets"][:,sample_index],
                    Tags.THERMAL_NOISES: self.in_vitro_data[wavelength]["thermal_noises"][:,sample_index]
                    }
            elif device==2: # Using 2D-PAD-Data acquired by Erlangen-device BUT ONLY OFFSETS
                return {
                    Tags.TRANSFORM_TO_IN_VITRO_DOMAIN: True,
                    Tags.SCALING_FACTOR: self.noise_data_pad[wavelength]["membrane_peaks"][sample_index], #mean value of device 1
                    Tags.IN_VITRO_LASER_ENERGY: self.noise_data_pad[wavelength]["energies"][sample_index], #mean value of device 1
                    #Tags.BROKEN_SENSORS: np.array([30, 48, 47, 158]),
                    #Tags.OFFSETS: self.noise_data_pad[wavelength]["offsets"][:,sample_index],
                    Tags.THERMAL_NOISES: self.noise_data_pad[wavelength]["thermal_noises"][:,sample_index]
                }
        elif self.experiment == "water_experiment_only_laserenergy":
            if device == 1: # Using waterbath measurements acquired by REZ-device BUT ONLY OFFSETS
                return {
                    Tags.TRANSFORM_TO_IN_VITRO_DOMAIN: True,
                    Tags.SCALING_FACTOR: self.in_vitro_data[wavelength]["membrane_peaks"][sample_index],
                    Tags.IN_VITRO_LASER_ENERGY: self.in_vitro_data[wavelength]["energies"][sample_index],
                    }
            elif device==2: # Using 2D-PAD-Data acquired by Erlangen-device BUT ONLY OFFSETS
                return {
                    Tags.TRANSFORM_TO_IN_VITRO_DOMAIN: True,
                    Tags.SCALING_FACTOR: self.noise_data_pad[wavelength]["membrane_peaks"][sample_index], #mean value of device 1
                    Tags.IN_VITRO_LASER_ENERGY: self.noise_data_pad[wavelength]["energies"][sample_index], #mean value of device 1
                }
        else:
            raise(Exception("No setting for given Experiment_name"))
