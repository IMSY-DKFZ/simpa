%%SPDX-FileCopyrightText: 2021 Computer Assisted Medical Interventions Group, DKFZ
%%SPDX-FileCopyrightText: 2021 Janek Groehl
%%SPDX-License-Identifier: MIT

function [] = time_reversal_3D(acoustic_path)

%% In case of an error, make sure the matlab scripts exits anyway
clean_up = onCleanup(@exit);

%% Read settings file
data = load(acoustic_path);

settings = data.settings;

%% Read time_series_data
time_series_data = data.time_series_data;

%% Define kWaveGrid

[Nx, Ny, Nz] = size(data.sensor_mask);
if isfield(settings, 'sample') == true
    if settings.sample == true
        dx = double(settings.voxel_spacing_mm)/(double(settings.upscale_factor) * 1000);
    else
        dx = double(settings.voxel_spacing_mm)/1000;    % convert from mm to m
    end
else
    dx = double(settings.voxel_spacing_mm)/1000;    % convert from mm to m
end
kgrid = kWaveGrid(Nx, dx, Ny, dx, Nz, dx);
source.p0 = 0;

%% Define medium

% if a field of the struct "data" is given which describes the sound speed, the array is loaded and is used as medium.sound_speed
if isfield(data, 'sos') == true
    medium.sound_speed = double(data.sos);
else
    medium.sound_speed = 1540;
end

% if a field of the struct "data" is given which describes the attenuation, the array is loaded and is used as medium.alpha_coeff
if isfield(data, 'alpha_coeff') == true
 medium.alpha_coeff = double(data.alpha_coeff);
else
 medium.alpha_coeff = 0.01;
end

medium.alpha_power = double(settings.medium_alpha_power); % b for a * MHz ^ b

% if a field of the struct "data" is given which describes the density, the array is loaded and is used as medium.density
if isfield(data, 'density') == true
    medium.density = double(data.density);
else
    medium.density = 1000*ones(Nx, Ny, Nz);
end

kgrid.setTime(settings.Nt, settings.dt)

%sound_speed_ref = min(min(medium.sound_speed));
%kgrid.dt = 1 / (settings.sensor_sampling_rate_mhz * 10^6);
%kgrid.Nt = ceil((sqrt((Nx*dx)^2+(Ny*dx)^2) / sound_speed_ref) / kgrid.dt);
%kgrid.t_array = makeTime(kgrid, medium.sound_speed, 0.3);	% time array with
% CFL number of 0.3 (advised by manual)
% Using makeTime, dt = CFL*dx/medium.sound_speed and the total
% time is set to the time it would take for an acoustic wave to travel
% across the longest grid diagonal.

%% Define sensor

sensor.mask = data.sensor_mask;

% if a field of the struct "data" is given which describes the sensor directivity angles, the array is loaded and is used as sensor.directivity_angle
%if isfield(data, 'directivity_angle') == true
%    sensor.directivity_angle = data.directivity_angle;
%end
%
%if isfield(data, 'directivity_size')
%    sensor.directivity_size = settings.sensor_directivity_size;
%end

% sensor.directivity_pattern = settings.sensor_directivity_pattern;

% model sensor frequency response
if isfield(settings, 'model_sensor_frequency_response') == true
    if settings.model_sensor_frequency_response == true
        center_freq = double(settings.sensor_center_frequency); % [Hz]
        bandwidth = double(settings.sensor_bandwidth); % [%]
        sensor.frequency_response = [center_freq, bandwidth];
    end
end

sensor.time_reversal_boundary_data = time_series_data;

%% Computation settings

if settings.gpu == true
    datacast = 'gpuArray-single';
else
    datacast = 'single';
end

input_args = {'DataCast', datacast, 'PMLInside', settings.pml_inside, ...
              'PMLAlpha', settings.pml_alpha, 'PMLSize', 'auto', ...
              'PlotPML', settings.plot_pml, 'RecordMovie', settings.record_movie, ...
              'MovieName', settings.movie_name, 'PlotScale', [0, 1]};

if settings.gpu == true
    reconstructed_data = kspaceFirstOrder3DG(kgrid, medium, source, sensor, input_args{:});
    reconstructed_data = gather(reconstructed_data);
else
    reconstructed_data = kspaceFirstOrder3D(kgrid, medium, source, sensor, input_args{:});
end

%% Write data to mat array
save(strcat(acoustic_path, 'tr.mat'), 'reconstructed_data')

end