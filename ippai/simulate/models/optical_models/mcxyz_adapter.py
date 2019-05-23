import nrrd
import numpy as np
import subprocess
from ippai.simulate import Tags
import os

def simulate(optical_properties_path, settings, optical_output_path):
    optical_properties = np.load(optical_properties_path)
    mua = optical_properties[Tags.PROPERTY_ABSORPTION]
    mus = optical_properties[Tags.PROPERTY_SCATTERING]
    g = optical_properties[Tags.PROPERTY_ANISOTROPY]

    #musp = mus * (1-g)

    shape = np.shape(mus)
    mcxyz_input = np.zeros((shape[0], shape[1], shape[2], 3))
    mcxyz_input[:, :, :, 0] = mua
    mcxyz_input[:, :, :, 1] = mus * (1-g)
    mcxyz_input[:, :, :, 2] = g
    tmp_input_path = optical_properties_path.replace(".npz", ".nrrd")
    output_path = optical_output_path.replace(".npz", ".nrrd")
    nrrd.write(tmp_input_path, mcxyz_input, {'space dimension': 4,
                                             'dimension': 4,
                                             'space directions': [[settings[Tags.SPACING_MM]/10, 0, 0, 0],
                                                         [0, settings[Tags.SPACING_MM]/10, 0, 0],
                                                         [0, 0, settings[Tags.SPACING_MM]/10, 0],
                                                         [0, 0, 0, 1]],
                                             'mcflag': 4, 'launchflag': 0, 'boundaryflag': 1,
                                             'launchPointX': 0, 'launchPointY': 0})

    cmd = list()
    cmd.append(settings[Tags.OPTICAL_MODEL_BINARY_PATH])
    cmd.append("-i")
    cmd.append(tmp_input_path)
    cmd.append("-o")
    cmd.append(output_path)
    if Tags.OPTICAL_MODEL_NUMBER_PHOTONS in settings:
        cmd.append("-n")
        cmd.append(str(settings[Tags.OPTICAL_MODEL_NUMBER_PHOTONS]))
    if Tags.OPTICAL_MODEL_PROBE_XML_FILE in settings:
        cmd.append("-p")
        cmd.append(str(settings[Tags.OPTICAL_MODEL_PROBE_XML_FILE]))

    subprocess.run(cmd)

    [fluence, meta] = nrrd.read(output_path)
    print(meta)
    p0 = fluence * mua

    # os.remove(tmp_input_path)
    #os.remove(optical_output_path)

    return [fluence, p0]




