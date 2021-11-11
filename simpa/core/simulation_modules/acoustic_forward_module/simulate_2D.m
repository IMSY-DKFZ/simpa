%%SPDX-FileCopyrightText: 2021 Computer Assisted Medical Interventions Group, DKFZ
%%SPDX-FileCopyrightText: 2021 Janek Groehl
%%SPDX-License-Identifier: MIT

function [] = simulate_2D(optical_path)

%% In case of an error, make sure the matlab scripts exits anyway
clean_up = onCleanup(@exit);

%% Read settings file

data = load(optical_path);
settings = data.settings;

%% Read initial pressure

source.p0 = data.initial_pressure;

% Choose if the initial pressure should be smoothed before simulation
if isfield(settings, 'initial_pressure_smoothing') == true
    p0_smoothing = settings.initial_pressure_smoothing;
else
    p0_smoothing = true;
end

%% Define kWaveGrid

% add N pixel "gel" to reduce Fourier artifact
GEL_LAYER_HEIGHT = 3;

source.p0 = padarray(source.p0, [GEL_LAYER_HEIGHT 0], 0, 'pre');
[Nx, Ny] = size(source.p0);
disp(Nx);
disp(Ny);
if isfield(settings, 'sample') == true
    if settings.sample == true
        dx = double(settings.voxel_spacing_mm) / (double(settings.upscale_factor) * 1000);
    else
        dx = double(settings.voxel_spacing_mm) / 1000;    % convert from mm to m
    end
else
    dx = double(settings.voxel_spacing_mm) / 1000;    % convert from mm to m
end
kgrid = kWaveGrid(Nx, dx, Ny, dx);

%% Define medium

% if a field of the struct "data" is given which describes the sound speed, the array is loaded and is used as medium.sound_speed
if isfield(data, 'sos') == true
    medium.sound_speed = data.sos;
    medium.sound_speed = padarray(medium.sound_speed, [GEL_LAYER_HEIGHT 0], 'replicate', 'pre');
else
    medium.sound_speed = 1540;
end

% if a field of the struct "data" is given which describes the attenuation, the array is loaded and is used as medium.alpha_coeff
if isfield(data, 'alpha_coeff') == true
 medium.alpha_coeff = data.alpha_coeff;
 medium.alpha_coeff = padarray(medium.alpha_coeff, [GEL_LAYER_HEIGHT 0], 'replicate', 'pre');
else
 medium.alpha_coeff = 0.01;
end

medium.alpha_power = double(settings.medium_alpha_power); % b for a * MHz ^ b
medium.alpha_mode = 'no_dispersion';

% if a field of the struct "data" is given which describes the density, the array is loaded and is used as medium.density
if isfield(data, 'density') == true
    medium.density = data.density;
    medium.density = padarray(medium.density, [GEL_LAYER_HEIGHT 0], 'replicate', 'pre');
else
    medium.density = 1000 * ones(Nx, Ny);
end

%% Sampling rate

% load sampling rate from settings
dt = 1.0 / double(settings.sensor_sampling_rate_mhz * 1000000);

% Simulate as many time steps as a wave takes to traverse diagonally through the entire tissue
Nt = round((sqrt(Ny*Ny+Nx*Nx)*dx / mean(medium.sound_speed, 'all')) / dt);

estimated_cfl_number = dt / dx * mean(medium.sound_speed, 'all');
disp(estimated_cfl_number);

% smaller time steps are better for numerical stability in time progressing simulations
% A minimum CFL of 0.3 is advised in the kwave handbook.
% In case we specify something larger, we use a higher sampling rate than anticipated.
% Otherwise we simulate with the target sampling rate
if estimated_cfl_number < 0.3
    kgrid.setTime(Nt, dt);
    disp("Setting custom time!");
else
    kgrid.t_array = makeTime(kgrid, medium.sound_speed, 0.3);
    disp("Making time!");
end

%% Define sensor

% create empty array
karray = kWaveArray;

elem_pos = data.sensor_element_positions/1000;

% In case some detectors are defined at zeros or with negative values out
% of bounds, correct all of them with minimum needed correction of
% spacing dx.

min_x_pos = find(elem_pos(1, :) <= 0);
min_y_pos = find(elem_pos(2, :) <= 0);
x_correction = 0;
y_correction = 0;
if size(min_x_pos) > 0
   x_correction = dx;
end

if size(min_y_pos) > 0
   y_correction = dx;
end

elem_pos(1, :) = elem_pos(1, :) - 0.5 * kgrid.x_size + x_correction + dx * GEL_LAYER_HEIGHT;
elem_pos(2, :) = elem_pos(2, :) - 0.5 * kgrid.y_size + y_correction;

num_elements = size(elem_pos, 2);

element_width = double(settings.detector_element_width_mm)/1000;
angles = data.directivity_angle;

if isfield(settings, 'sensor_radius_mm') == true
    radius_of_curv = double(settings.sensor_radius_mm)/1000;
end


% add elements to the array

for ind = 1:num_elements
  x = elem_pos(1, ind);
  y = elem_pos(2, ind);
  alpha = angles(1, ind);
  x = x + 0.5 * (element_width*sin(alpha));
  y = y + 0.5 * (element_width*cos(alpha));
  karray.addRectElement([x, y], element_width, 0.00001, [angles(1, ind)]);
end

% assign binary mask from karray to the sensor mask
sensor.mask = karray.getArrayBinaryMask(kgrid);

% model sensor frequency response
if isfield(settings, 'model_sensor_frequency_response') == true
    if settings.model_sensor_frequency_response == true
        center_freq = double(settings.sensor_center_frequency); % [Hz]
        bandwidth = double(settings.sensor_bandwidth); % [%]
        sensor.frequency_response = [center_freq, bandwidth];
    end
end

%% Computation settings

if settings.gpu == true
    datacast = 'gpuArray-single';
else
    datacast = 'single';
end

input_args = {'DataCast', datacast, 'PMLInside', settings.pml_inside, ...
               'PMLAlpha', settings.pml_alpha, 'PMLSize', 'auto', ...
               'PlotPML', settings.plot_pml, 'RecordMovie', settings.record_movie, ...
               'MovieName', settings.movie_name, 'PlotScale', [-1, 1], 'LogScale', settings.acoustic_log_scale, ...
               'Smooth', p0_smoothing};

if settings.gpu == true
    time_series_data = kspaceFirstOrder2DG(kgrid, medium, source, sensor, input_args{:});
    time_series_data = gather(time_series_data);
else
    time_series_data = kspaceFirstOrder2D(kgrid, medium, source, sensor, input_args{:});
end

% combine data to give one trace per physical array element
time_series_data = karray.combineSensorData(kgrid, time_series_data);

%% Write data to mat array
save(optical_path, 'time_series_data')%, '-v7.3')
time_step = kgrid.dt
number_time_steps = kgrid.Nt
save(strcat(optical_path, 'dt.mat'), 'time_step', 'number_time_steps');

end
