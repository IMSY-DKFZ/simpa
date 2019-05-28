import numpy as np
import struct
import subprocess
from ippai.simulate import Tags
import matplotlib.pyplot as plt
import json
import nibabel

def simulate(optical_properties_path, settings, optical_output_path):
    optical_properties = np.load(optical_properties_path)
    absorption = optical_properties[Tags.PROPERTY_ABSORPTION_PER_CM] * 0.1
    scattering = optical_properties[Tags.PROPERTY_SCATTERING_PER_CM] * 0.1
    op_array = np.asarray([absorption,
                           scattering])

    [_, nx, ny, nz] = np.shape(op_array)

    #TODO: Fix saving of file in correct format or fix usage in mcx

    # op_array = op_array.astype(np.float32)
    # op_array = op_array.T
    # op_array = np.moveaxis(op_array, 0, 1)

    print(np.shape(op_array))

    optical_properties_list = list(np.reshape(op_array, op_array.size, "F"))

    # plt.imshow(op_array[0, int(nx / 2), :, :])
    # plt.show()

    mcx_input = struct.pack("f" * len(optical_properties_list), *optical_properties_list)
    tmp_input_path = settings[Tags.SIMULATION_PATH] + "/" + settings[Tags.VOLUME_NAME]+".bin"
    input_file = open(tmp_input_path, "wb")
    input_file.write(mcx_input)

    output_file = settings[Tags.SIMULATION_PATH] + "/" + settings[Tags.VOLUME_NAME]+"_output"
    #write settings to json

    settings_dict = {
        "Session": {
        "ID": output_file,
        "DoAutoThread": 1,
        "Photons": settings[Tags.OPTICAL_MODEL_NUMBER_PHOTONS]
         },
	"Forward": {
		"T0": 0,
		"T1": 5e-09,
		"Dt": 5e-09
	},
	"Optode": {
		"Source": {
			"Pos": [int(nx/2)+1,int(ny/2)+1,0],
			"Dir": [0,0,1]
		}
	},
	"Domain": {
		"OriginType": 0,
        "LengthUnit": settings[Tags.SPACING_MM],
		"Media": [
			{
				"mua": 0,
				"mus": 0,
				"g": 1,
				"n": 1
			},
			{
				"mua": 0,
				"mus": 0,
				"g": 0.9,
				"n": 1.37
			}
		],
		"MediaFormat": "muamus_float",
		"Dim": [nx, ny, nz],
		"VolumeFile": settings[Tags.SIMULATION_PATH] + "/" + settings[Tags.VOLUME_NAME]+".bin"
	}}

    json_filename = settings[Tags.SIMULATION_PATH] + "/" + settings[Tags.VOLUME_NAME]+".json"
    with open(json_filename, "w") as json_file:
        json.dump(settings_dict, json_file)

    # cmd = list()
    # cmd.append(settings[Tags.OPTICAL_MODEL_BINARY_PATH])
    # cmd.append("-f")
    # cmd.append(json_filename)
    # cmd.append("-M")
    # cmd.append("1")
    #
    # subprocess.run(cmd)

    cmd = list()
    cmd.append(settings[Tags.OPTICAL_MODEL_BINARY_PATH])
    cmd.append("-f")
    cmd.append(json_filename)
    cmd.append("-O")
    cmd.append("F")

    subprocess.run(cmd)

    # Read output

    f = open(output_file+".mc2", 'rb')
    data = f.read()
    data = struct.unpack('%df' % (len(data) / 4), data)
    fluence = np.asarray(data).reshape([nx, ny, nz, 1], order='F')
    fluence = np.squeeze(fluence, 3)



    p0 = fluence * absorption

    # data = nibabel.load(settings[Tags.SIMULATION_PATH] + "/" + settings[Tags.VOLUME_NAME]+"_output_vol.nii").get_fdata()
    # print(np.shape(data))
    #
    # plt.imshow(data[:, int(nx / 2), :, 0])
    # plt.show()

    return [fluence, p0]

