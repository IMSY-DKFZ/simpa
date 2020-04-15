%% The MIT License (MIT)
%%
%% Copyright (c) 2018 Computer Assisted Medical Interventions Group, DKFZ
%%
%% Permission is hereby granted, free of charge, to any person obtaining a copy
%% of this software and associated documentation files (the "Software"), to deal
%% in the Software without restriction, including without limitation the rights
%% to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
%% copies of the Software, and to permit persons to whom the Software is
%% furnished to do so, subject to the following conditions:
%%
%% The above copyright notice and this permission notice shall be included in all
%% copies or substantial portions of the Software.
%%
%% THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
%% IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
%% FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
%% AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
%% LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
%% OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
%% SOFTWARE.

function [] = simulate_3D(settings, optical_path)

%% Read settings file
settings = jsondecode(fileread(settings));  % read settings as json file

%% Read initial pressure
data = load(optical_path);
source.p0 = data.initial_pressure;

%% Define kWaveGrid

% add 2 pixel "gel" to reduce Fourier artifact
GEL_LAYER_HEIGHT = 0;

%source.p0 = padarray(source.p0, [GEL_LAYER_HEIGHT 0], 0, 'pre');
[Nx, Ny, Nz] = size(source.p0);
if settings.sample == true
    dx = settings.voxel_spacing_mm/(settings.upscale_factor * 1000);
else
    dx = settings.voxel_spacing_mm/1000;    % convert from mm to m
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

medium.alpha_power = settings.medium_alpha_power; % b for a * MHz ^ b

% if a field of the struct "data" is given which describes the density, the array is loaded and is used as medium.density
if isfield(data, 'density') == true
    medium.density = data.density;
    % add 2 pixel "gel" to reduce Fourier artifact
%    medium.density = padarray(medium.density, [GEL_LAYER_HEIGHT 0], 'replicate', 'pre');
else
    medium.density = ones(Nx, Ny);
end

%sound_speed_ref = min(min(medium.sound_speed));
%kgrid.dt = 1 / (settings.sensor_sampling_rate_mhz * 10^6);
%kgrid.Nt = ceil((sqrt((Nx*dx)^2+(Ny*dx)^2) / sound_speed_ref) / kgrid.dt);
kgrid.t_array = makeTime(kgrid, medium.sound_speed, 0.15);	% time array with
% CFL number of 0.3 (advised by manual)
% Using makeTime, dt = CFL*dx/medium.sound_speed and the total
% time is set to the time it would take for an acoustic wave to travel
% across the longest grid diagonal.

%% Define sensor

% if a field of the struct "data" is given which describes the sensor mask, the array is loaded and is used as sensor.mask
if isfield(data, 'sensor_mask') == true
    sensor.mask = data.sensor_mask;
    % add 2 pixel "gel" to reduce Fourier artifact
%    sensor.mask = padarray(sensor.mask, [GEL_LAYER_HEIGHT 0], 0, 'pre');
else
%    num_elements = settings.sensor_num_elements;
%    element_spacing = Ny / num_elements;
%    sensor.mask = zeros(Nx, Ny);
%    sensor.mask(3, round(element_spacing/2):round(element_spacing):Ny) = 1;
%    writeNPY(sensor.mask, "/home/kris/hard_drive/data/pipeline_test/Pipeline_test/sensor_mask.npy");
end

% if a field of the struct "data" is given which describes the sensor directivity angles, the array is loaded and is used as sensor.directivity_angle
if isfield(data, 'directivity_angle') == true
    sensor.directivity_angle = data.directivity_angle;
    % add 2 pixel "gel" to reduce Fourier artifact
%    sensor.directivity_angle = padarray(sensor.directivity_angle, [GEL_LAYER_HEIGHT 0], 0, 'pre');
end

if isfield(data, 'directivity_size')
    sensor.directivity_size = settings.sensor_directivity_size;
end

%sensor.directivity_pattern = settings.sensor_directivity_pattern;

% define the frequency response of the sensor elements, gaussian shape with
% FWHM = bandwidth*center_freq

center_freq = settings.sensor_center_frequency; % [Hz]
bandwidth = settings.sensor_bandwidth; % [%]
sensor.frequency_response = [center_freq, bandwidth];

%% Computation settings

if settings.gpu == true
    datacast = 'gpuArray-single';
else
    datacast = 'single';
end
% max_pressure = max(max(initial_pressure));

input_args = {'DataCast', datacast, 'PMLInside', settings.pml_inside, ...
              'PMLAlpha', settings.pml_alpha, 'PMLSize', 'auto', ...
              'PlotPML', settings.plot_pml, 'RecordMovie', settings.record_movie, ...
              'MovieName', settings.movie_name, 'PlotScale', [-1, 1], 'LogScale', settings.acoustic_log_scale};

if settings.gpu == true
    sensor_data_3D = kspaceFirstOrder3DG(kgrid, medium, source, sensor, input_args{:});
    sensor_data_3D = gather(sensor_data_3D);
else
    sensor_data_3D = kspaceFirstOrder3D(kgrid, medium, source, sensor, input_args{:});
end

%% Write data to numpy array
save(strcat(optical_path, '.mat'), 'sensor_data_2D')
time_step = kgrid.dt;
save(strcat(optical_path, 'dt.mat'), 'time_step');

end