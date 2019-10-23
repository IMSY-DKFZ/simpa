from ippai.simulate import Tags
from ippai.simulate.models.optical_models.mcx_adapter import McxAdapter
from ippai.simulate.models.optical_models.mcxyz_adapter import McxyzAdapter
import numpy as np


def run_optical_forward_model(settings, optical_properties_path):
    # TODO
    print("OPTICAL FORWARD")
    optical_output_path = settings[Tags.SIMULATION_PATH] + "/" + settings[Tags.VOLUME_NAME] + "/" + \
                   Tags.OPTICAL_MODEL_OUTPUT_NAME + "_" + \
                   str(settings[Tags.WAVELENGTH]) + ".npz"

    volumes = [None]

    if Tags.OPTICAL_MODEL not in settings:
        raise AssertionError("Tags.OPTICAL_MODEL tag was not specified in the settings. Skipping optical modelling.")

    model = settings[Tags.OPTICAL_MODEL]
    forward_model_implementation = None

    if model == Tags.MODEL_MCXYZ:
        forward_model_implementation = McxyzAdapter()
    if model == Tags.MODEL_MCX:
        forward_model_implementation = McxAdapter()

    fluence = forward_model_implementation.simulate(optical_properties_path, settings)

    optical_properties = np.load(optical_properties_path)
    absoprtion = optical_properties[Tags.PROPERTY_ABSORPTION_PER_CM]
    gruneisen_parameter = optical_properties[Tags.PROPERTY_GRUNEISEN_PARAMETER]

    if Tags.LASER_PULSE_ENERGY_IN_MILLIJOULE in settings:
        units = Tags.UNITS_PRESSURE
        # Initial pressure should be given in units of Pascale
        conversion_factor = 1e6  # 1 J/cm^3 = 10^6 N/m^2 = 10^6 Pa
        initial_pressure = (absoprtion * fluence * gruneisen_parameter *
                            settings[Tags.LASER_PULSE_ENERGY_IN_MILLIJOULE] * conversion_factor)
    else:
        units = Tags.UNITS_ARBITRARY
        initial_pressure = absoprtion * fluence

    np.savez(optical_output_path,
             fluence=fluence,
             initial_pressure=initial_pressure,
             units=units)

    return optical_output_path
