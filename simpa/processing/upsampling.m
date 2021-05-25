%%SPDX-FileCopyrightText: 2021 Computer Assisted Medical Interventions Group, DKFZ
%%SPDX-FileCopyrightText: 2021 VISION Lab, Cancer Research UK Cambridge Institute (CRUK CI)
%%SPDX-License-Identifier: MIT

function [] = upsampling(image_data_path, settings_path)

settings = jsondecode(fileread(settings_path));

initial_pressure = readNPY(image_data_path);
upsampled_init = imresize(initial_pressure, settings.upscale_factor, settings.upsampling_method);

writeNPY(upsampled_init, settings.output_file)

end