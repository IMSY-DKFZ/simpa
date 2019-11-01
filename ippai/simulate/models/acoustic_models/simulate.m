function [] = simulate(settings, optical_path)

%% Read settings file
settings = jsondecode(fileread(settings));  % read settings as json file

%% Read initial pressure
data = unzip(optical_path);    % unzip optical forward model
% from .npz file to "fluence" (data{1}) and "initial_pressure" (data{2})
initial_pressure = rot90(readNPY(data{2}), 3);  % rotate initial pressure 270°
initial_pressure = fliplr(initial_pressure)
source.p0 = initial_pressure;

%% Define kWaveGrid

% add 2 pixel "gel" to reduce Fourier artifact
GEL_LAYER_HEIGHT = 2

source.p0 = padarray(source.p0, [GEL_LAYER_HEIGHT 0], 0, 'pre');
[Nx, Ny] = size(source.p0);
if settings.sample == true
    dx = settings.voxel_spacing_mm/(settings.upscale_factor * 1000);
else
    dx = settings.voxel_spacing_mm/1000;    % convert from mm to m
end
kgrid = kWaveGrid(Nx, dx, Ny, dx);

%% Define medium

% if a path to a file is given which describes the sound speed in a .npy
% file, the file is loaded and is used as medium.sound_speed
if ischar(settings.medium_sound_speed) == 1 
    medium.sound_speed = readNPY(settings.medium_sound_speed);
    % add 2 pixel "gel" to reduce Fourier artifact
    medium.sound_speed = padarray(medium.sound_speed, [GEL_LAYER_HEIGHT 0], 'replicate', 'pre');
else
    medium.sound_speed = settings.medium_sound_speed;
end

medium.alpha_coeff = settings.medium_alpha_coeff; % a
medium.alpha_power = settings.medium_alpha_power; % b for a * MHz ^ b

% if a path to a file is given which describes the medium density in a .npy
% file, the file is loaded and is used as medium.medium_density
if ischar(settings.medium_density) == 1 
    medium.density = readNPY(settings.medium_density);
    % add 2 pixel "gel" to reduce Fourier artifact
    medium.density = padarray(medium.density, [GEL_LAYER_HEIGHT 0], 'replicate', 'pre');
else
    medium.density = ones(Nx, Ny);
end

kgrid.dt = 1 / (settings.sensor_sampling_rate_mhz * 10^6)
kgrid.Nt = ceil((sqrt((Nx*dx)^2+(Ny*dx)^2) / medium.sound_speed) / kgrid.dt)
% kgrid.t_array = makeTime(kgrid, medium.sound_speed, 0.3);	% time array with
% CFL number of 0.3 (advised by manual)
% Using makeTime, dt = CFL*dx/medium.sound_speed and the total
% time is set to the time it would take for an acoustic wave to travel 
% across the longest grid diagonal.

%% Define sensor

% if a path to a file is given which describes the sensor mask in a .npy
% file, the file is loaded and is used as sensor.mask
if ischar(settings.sensor_mask) == 1 
    sensor.mask = readNPY(settings.sensor_mask);
    % add 2 pixel "gel" to reduce Fourier artifact
    sensor.mask = padarray(sensor.mask, [GEL_LAYER_HEIGHT 0], 0, 'pre');
else
    num_elements = settings.sensor_num_elements
    spacing = Ny / num_elements
    sensor.mask = zeros(Nx, Ny);
    sensor.mask(2, spacing/2:spacing:Ny) = 1;
end

% if a path to a file is given which describes the sensor directivity angles in a .npy
% file, the file is loaded and is used as sensor.directivity_angle
if ischar(settings.sensor_directivity_angle) == 1 
    sensor.directivity_angle = readNPY(settings.sensor_directivity_angle);
    % add 2 pixel "gel" to reduce Fourier artifact
    sensor.directivity_angle = padarray(sensor.directivity_angle, [GEL_LAYER_HEIGHT 0], 0, 'pre');
else
    dir_angles = settings.sensor_directivity_angle;
    sensor.directivity_angle = zeros(Nx, Ny);
    sensor.directivity_angle(sensor.mask == 1) = dir_angles;
end

sensor.directivity_size = settings.sensor_directivity_size;

sensor.directivity_pattern = settings.sensor_directivity_pattern;

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
max_pressure = max(max(initial_pressure))

input_args = {'DataCast', datacast, 'PMLInside', settings.pml_inside, ...
              'PMLAlpha', settings.pml_alpha, 'PMLSize', settings.pml_size, ...
              'PlotPML', settings.plot_pml, 'RecordMovie', settings.record_movie, ...
              'MovieName', settings.movie_name, 'PlotScale', [-max_pressure/2, max_pressure/2]};

sensor_data_2D = kspaceFirstOrder2D(kgrid, medium, source, sensor, input_args{:});

if settings.gpu == true
    sensor_data_2D = gather(sensor_data_2D);
end

%% Write data to numpy array
writeNPY(sensor_data_2D, settings.output_file);

end