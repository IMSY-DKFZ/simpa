%%SPDX-FileCopyrightText: 2021 Computer Assisted Medical Interventions Group, DKFZ
%%SPDX-FileCopyrightText: 2021 VISION Lab, Cancer Research UK Cambridge Institute (CRUK CI)
%%SPDX-License-Identifier: MIT

function [] = simulate_3D(optical_path)

%% Read settings file

data = load(optical_path);
settings = data.settings;

%% Read initial pressure

source.p0 = data.initial_pressure;

%% Define kWaveGrid

% add 2 pixel "gel" to reduce Fourier artifact
GEL_LAYER_HEIGHT = 0;

%source.p0 = padarray(source.p0, [GEL_LAYER_HEIGHT 0], 0, 'pre');
[Nx, Ny, Nz] = size(source.p0);
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

%% Define medium

% if a field of the struct "data" is given which describes the sound speed, the array is loaded and is used as medium.sound_speed
if isfield(data, 'sos') == true
    medium.sound_speed = data.sos;
    % add 2 pixel "gel" to reduce Fourier artifact
%    medium.sound_speed = padarray(medium.sound_speed, [GEL_LAYER_HEIGHT 0], 'replicate', 'pre');
else
    medium.sound_speed = 1540;
end

% if a field of the struct "data" is given which describes the attenuation, the array is loaded and is used as medium.alpha_coeff
if isfield(data, 'alpha_coeff') == true
 medium.alpha_coeff = data.alpha_coeff;
 % add 2 pixel "gel" to reduce Fourier artifact
% medium.alpha_coeff = padarray(medium.alpha_coeff, [GEL_LAYER_HEIGHT 0], 'replicate', 'pre');
else
 medium.alpha_coeff = 0.01;
end

medium.alpha_power = double(settings.medium_alpha_power); % b for a * MHz ^ b

% if a field of the struct "data" is given which describes the density, the array is loaded and is used as medium.density
if isfield(data, 'density') == true
    medium.density = data.density;
    % add 2 pixel "gel" to reduce Fourier artifact
%    medium.density = padarray(medium.density, [GEL_LAYER_HEIGHT 0], 'replicate', 'pre');
else
    medium.density = 1000*ones(Nx, Ny, Nz);
end

%sound_speed_ref = min(min(medium.sound_speed));
%kgrid.dt = 1 / (settings.sensor_sampling_rate_mhz * 10^6);
%kgrid.Nt = ceil((sqrt((Nx*dx)^2+(Ny*dx)^2) / sound_speed_ref) / kgrid.dt);
kgrid.t_array = makeTime(kgrid, medium.sound_speed, 0.3);	% time array with
% CFL number of 0.3 (advised by manual)
% Using makeTime, dt = CFL*dx/medium.sound_speed and the total
% time is set to the time it would take for an acoustic wave to travel
% across the longest grid diagonal.

%% Define sensor

% if a field of the struct "data" is given which describes the sensor mask, the array is loaded and is used as sensor.mask
if isfield(data, 'sensor_mask')
    sensor.mask = data.sensor_mask;
    fprintf("Number of elements in sensor mask: %d", sum(sensor.mask,'all'))
end

% if a field of the struct "data" is given which describes the sensor directivity angles, the array is loaded and is used as sensor.directivity_angle
if isfield(data, 'directivity_angle')
    sensor.directivity_angle = data.directivity_angle;
end

if isfield(data, 'directivity_size')
    sensor.directivity_size = double(settings.sensor_directivity_size);
end

%sensor.directivity_pattern = settings.sensor_directivity_pattern;

%% Computation settings

if settings.gpu == true
    datacast = 'gpuArray-single';
else
    datacast = 'single';
end
% max_pressure = max(max(initial_pressure));

input_args = {'DataCast', datacast, 'PMLInside', settings.pml_inside, ...
              'PMLAlpha', double(settings.pml_alpha), 'PMLSize', 'auto', ...
              'PlotPML', settings.plot_pml, 'RecordMovie', settings.record_movie, ...
              'MovieName', settings.movie_name, 'PlotScale', [-1, 1], 'LogScale', settings.acoustic_log_scale};

if settings.gpu == true
    time_series_data = kspaceFirstOrder3DG(kgrid, medium, source, sensor, input_args{:});
    time_series_data = gather(time_series_data);
else
    time_series_data = kspaceFirstOrder3D(kgrid, medium, source, sensor, input_args{:});
end

% define the frequency response of the sensor elements, gaussian shape with
center_freq = double(settings.sensor_center_frequency); % in units of [Hz]
bandwidth = double(settings.sensor_bandwidth); % in units of [%]
num_elements = size(time_series_data, 2);
num_samples = size(time_series_data(1).p, 4);
ts_array = zeros(num_elements, num_samples);
for i = 1:num_elements
   ts_array(i, :) = time_series_data(i).p;
end
time_series_data = gaussianFilter(ts_array, 1/kgrid.dt, center_freq, bandwidth);
%time_series_data = time_series_data.';

%% Write data to mat array
save(strcat(optical_path), 'time_series_data')
time_step = kgrid.dt;
number_time_steps = kgrid.Nt;
save(strcat(optical_path, 'dt.mat'), 'time_step', 'number_time_steps');

end