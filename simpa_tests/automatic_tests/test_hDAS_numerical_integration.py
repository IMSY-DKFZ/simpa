#!/usr/bin/env python
# coding: utf-8

# In[1]:
from simpa.utils import Tags
import torch
from torch.distributions.uniform import Uniform
import numpy as np
from simpa.utils.settings import Settings
from simpa.log.file_logger import Logger
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from simpa.utils.calculate import bilinear_interpolation
from simpa.core.simulation_modules.reconstruction_module.reconstruction_utils import calculate_delays_for_heterogen_sos
from simpa.core.simulation_modules.reconstruction_module.reconstruction_utils import calculate_delays_for_heterogen_sos_OLD


# In[2]:
global_settings = Settings()
global_settings[Tags.DIM_VOLUME_X_MM] = 20
global_settings[Tags.DIM_VOLUME_Z_MM] = 40
global_settings[Tags.SPACING_MM] = 0.1

xdim_glob = int(global_settings[Tags.DIM_VOLUME_X_MM]/global_settings[Tags.SPACING_MM])
ydim_glob = int(global_settings[Tags.DIM_VOLUME_Z_MM]/global_settings[Tags.SPACING_MM])

print(f"Global Dimensions {xdim_glob} x {ydim_glob}")

logger = Logger()

torch_device = torch.device("cuda")

# FOV

xdim = xdim_glob-2 #xdim_glob     # pixels (xdim_mm = 10 mm)
ydim = ydim_glob-2#ydim_glob        # pixels (ydim_mm = 15 mm)
print(f"FOV Dimensions {xdim} x {ydim}")
zdim = 1
xdim_start = -xdim//2
xdim_end = xdim  + xdim_start
ydim_start = -ydim//2
ydim_end = ydim + ydim_start
zdim_start = 1
zdim_end = 1

time_spacing_in_ms = 0.2

spacing_in_mm = 0.1


device_base_position_mm = np.array([global_settings[Tags.DIM_VOLUME_X_MM]/2, global_settings[Tags.SPACING_MM], global_settings[Tags.DIM_VOLUME_Z_MM]/2])  

random_sensors = False
if random_sensors: 
    # sample sensors
    sensor_distri = Uniform(low=torch.tensor([-global_settings[Tags.DIM_VOLUME_X_MM]/2+global_settings[Tags.SPACING_MM]/4,
                                              0,
                                              -global_settings[Tags.DIM_VOLUME_Z_MM]/2]+global_settings[Tags.SPACING_MM]/4).to(torch_device),
                            high=torch.tensor([global_settings[Tags.DIM_VOLUME_X_MM]/2-global_settings[Tags.SPACING_MM]/4,
                                               0.01,
                                               global_settings[Tags.DIM_VOLUME_Z_MM]/2]-global_settings[Tags.SPACING_MM]/4).to(torch_device))
    n_sensor_elements = 2
    sensor_positions = sensor_distri.sample((n_sensor_elements,))
else:
    n_sensor_elements = 2
    sensor_positions = torch.tensor([[xdim/4*spacing_in_mm, 1e-5, -ydim/4*spacing_in_mm],
                                     [-xdim/2*spacing_in_mm, 1e-5,  ydim/2*spacing_in_mm]], device=torch_device)


# In[3]:
# TODO: To investigate
grid_points_middle = True # should be true
center_FOV_x = True # should be true


# * grid_points_middle = True: 
#     
#         assumes that the position of the first sos pixel is at [global_spacing/2 mm, global_spacing/2 mm] and is spanned over [0 mm,0 mm] to [global_spacing mm, global_spacing mm]
#         
#         
# * grid_points_middle = False: 
#         
#         assumes that the position of the first sos pixel is at [0 mm ,0 mm]  and is spanned over [-global_spacing/2 mm, -global_spacing/2 mm] to [global_spacing/2 mm, global_spacing/2 mm]

# In[4]:
# for gradients
c = 1
k = 100

# for quadratic and gaussian
centre = [global_settings[Tags.DIM_VOLUME_X_MM]/2, global_settings[Tags.DIM_VOLUME_Z_MM]/2]
#centre  = [0,0]

#scalarfield_type = "constant"
#scalarfield_type = "horizontal_gradient"
#scalarfield_type = "vertical_gradient"
scalarfield_type = "quadratic"

if scalarfield_type == "constant":
    scalarfield = np.ones((xdim_glob,ydim_glob)) * c
elif scalarfield_type == "horizontal_gradient":
    x_pos_mm = np.arange(xdim_glob)*global_settings[Tags.SPACING_MM]
    if grid_points_middle:
        x_pos_mm += 0.5*global_settings[Tags.SPACING_MM]
    scalarfield = np.ones((xdim_glob,ydim_glob)) * (k*x_pos_mm[:,None] + c)
elif scalarfield_type == "vertical_gradient":
    y_pos_mm = np.arange(ydim_glob)*global_settings[Tags.SPACING_MM]
    if grid_points_middle:
        y_pos_mm += 0.5*global_settings[Tags.SPACING_MM] 
    scalarfield = np.ones((xdim_glob,ydim_glob)) * (k*y_pos_mm[None,:] + c)
elif scalarfield_type == "quadratic":
    x_pos_mm = np.arange(xdim_glob)*global_settings[Tags.SPACING_MM]
    y_pos_mm = np.arange(ydim_glob)*global_settings[Tags.SPACING_MM]
    if grid_points_middle:
        x_pos_mm += 0.5*global_settings[Tags.SPACING_MM]
        y_pos_mm += 0.5*global_settings[Tags.SPACING_MM]
    xgrid, ygrid = np.meshgrid(x_pos_mm, y_pos_mm) 
    scalarfield = (xgrid-centre[0])**2+(ygrid-centre[1])**2 + c
    scalarfield = scalarfield.T # in order to be shape xdim_glob x ydim_glob
else:
    raise("Error")

scalarfield_3d = scalarfield[:,None,:]

speed_of_sound_in_m_per_s = 1/scalarfield_3d


# In[5]:
sens_pix = (sensor_positions.cpu().numpy()+device_base_position_mm)/global_settings[Tags.SPACING_MM]
device_pix = device_base_position_mm/global_settings[Tags.SPACING_MM]

xdim_start_glob = xdim_start + device_pix[0]
ydim_start_glob = ydim_start + device_pix[2]

plt.figure(figsize=(6,12))
ax = plt.gca()
plt.imshow(scalarfield.T)
plt.colorbar()


# Create a Rectangle patch
rect = patches.Rectangle((xdim_start_glob,ydim_start_glob),xdim, ydim, linewidth=1,edgecolor='grey', facecolor='none', linestyle="--", label="FOV")
# Add the patch to the Axes
ax.add_patch(rect)

plt.scatter(sens_pix[:,0], sens_pix[:,2], color="red", label="sensors")
plt.scatter(device_pix[0], device_pix[2], color="orange", label="device base position")

plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), facecolor='whitesmoke')

"""
plt.figure()
plt.imshow(1/scalarfield)
plt.colorbar()
plt.show()""";


# In[6]:


def get_xyz(xdim, xdim_start, ydim, ydim_start, zdim, zdim_start, n_sensor_elements, torch_device, center_FOV_x):
    if center_FOV_x:
        x_offset = 0.5 if xdim % 2 == 0 else 0  # to ensure pixels are symmetrically arranged around the 0 like the
        # sensor positions, add an offset of 0.5 pixels if the dimension is even
        print(x_offset)
    else:
        x_offset = 0

    x = xdim_start + torch.arange(xdim, device=torch_device, dtype=torch.float32) + x_offset
    y = ydim_start + torch.arange(ydim, device=torch_device, dtype=torch.float32)
    if zdim == 1:
        """xx, yy, zz, jj = torch.meshgrid(torch.arange(xdim_start, xdim_end, device=torch_device),
                                        torch.arange(ydim_start, ydim_end, device=torch_device),
                                        torch.arange(zdim, device=torch_device),
                                        torch.arange(n_sensor_elements, device=torch_device))"""
        z = torch.arange(zdim, device=torch_device, dtype=torch.float32)
    else:
        """xx, yy, zz, jj = torch.meshgrid(torch.arange(xdim_start, xdim_end, device=torch_device),
                                        torch.arange(ydim_start, ydim_end, device=torch_device),
                                        torch.arange(zdim_start, zdim_end, device=torch_device),
                                        torch.arange(n_sensor_elements, device=torch_device))"""
        z = zdim_start + torch.arange(zdim, device=torch_device, dtype=torch.float32)
    #j = torch.arange(n_sensor_elements, device=torch_device, dtype=torch.float32)
    return x, y, z


# In[7]:


x, y, z = get_xyz(xdim, xdim_start, ydim, ydim_start, zdim, zdim_start, n_sensor_elements, torch_device, center_FOV_x)


# # [
# ad grid_sample function:
# 
# according to: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
# 
# align_corners (bool, optional) – Geometrically, we consider the pixels of the input as squares rather than points. If set to True, the extrema (-1 and 1) are considered as referring to the center points of the input’s corner pixels. If set to False, they are instead considered as referring to the corner points of the input’s corner pixels, making the sampling more resolution agnostic. This option parallels the align_corners option in interpolate(), and so whichever option is used here should also be used there to resize the input image before grid sampling. Default: False
# # ]

# In[8]:


def calc_delays_(sensor_positions: torch.tensor, xdim: int, ydim: int, zdim: int,
                 xdim_start: int, xdim_end: int, ydim_start: int, ydim_end: int,
                 zdim_start: int, zdim_end: int, spacing_in_mm: float,
                 time_spacing_in_ms: float, speed_of_sound_in_m_per_s: np.ndarray,
                 n_sensor_elements: int, global_settings: Settings, 
                 device_base_position_mm: np.ndarray, logger: Logger,
                 torch_device: torch.device,
                 grid_points_middle: bool, center_FOV_x: bool) -> torch.tensor:
    """
    Returns the delays indicating which time series data has to be summed up taking a heterogenous 
    speed-of-sound-map into account, i.e. performing a line integral over the inverse speed-of-sound map.
    
    :param sensor_positions: (torch tensor) sensor positions in mm in the FOV coordinate system,
                                            where the origin is at the device base position
    :param xdim: (int) number of x values of sources in FOV
    :param ydim: (int) number of y values of sources in FOV
    :param zdim: (int) number of z values of sources in FOV
    :param xdim_start: (int) x-index of starting pixel in FOV
    :param ydim_start: (int) y-index of starting pixel in FOV
    :param zdim_start: (int) z-index of starting pixel in FOV
    :param xdim_end: (int) x-index of stopping pixel in FOV
    :param ydim_end: (int) y-index of stopping pixel in FOV
    :param zdim_end: (int) z-index of stopping pixel in FOV
    :param spacing_in_mm: (float) spacing of voxels in reconstructed image (FOV) in mm
    :param time_spacing_in_ms: (float) temporal spacing of the time series data in ms
    :param speed_of_sound_in_m_per_s: (np.ndarray) (heterogenous) speed-of-sound map
    :param n_sensor_elements: (int) number of sensor elements of the given device
    :param global settings: (Settings) settings for the whole simulation
    :param device_base_position_mm: (np.ndarray) position of the device base in mm
    :param logger: (logger) logger instance in order to log and print warnings/errors/debug hints
    :param torch_device: (torch.device) either cpu or cuda GPU device used for the tensors

    :return: (torch tensor) delays indices indicating which time series data
             one has to sum up
    """
    logger.debug("Considering heterogenous SoS-map in reconstruction algorithm")

    STEPS_MIN = 2
    mixed_coordinate_system_calculation = False #TODO: decide for one implementation,  ########################################### setze auf false wenn vergleich mit analytical solution
    # so far mixed_coordinate_system_calculation is more accurate!!!!
    
    #TODO: OUTSOURCE
    x, y, z = get_xyz(xdim, xdim_start, ydim, ydim_start, zdim, zdim_start, n_sensor_elements, torch_device, center_FOV_x)
    
    if zdim == 1:
        # get relevant sos slice
        transducer_plane = int(round((device_base_position_mm[1]/global_settings[Tags.SPACING_MM]))) - 1
        # take wanted sos slice and invert values
        sos_map = 1/speed_of_sound_in_m_per_s[:,transducer_plane,:]
        # convert it to tensor and desired shape
        sos_tensor = torch.from_numpy(sos_map) # IMPORTANT: not transposed!
        sos_tensor = sos_tensor.to(torch_device)

        # calculate absolute positions, where origin is  the origin of the whole volume (volume coordinate system)
        xx, yy = torch.meshgrid(x, y)

        device_base_position_mm = torch.from_numpy(device_base_position_mm)
        device_base_position_mm = device_base_position_mm.to(torch_device)

        if mixed_coordinate_system_calculation: #TODO: MORE ACCURATE
            xx = xx*spacing_in_mm
            yy = yy*spacing_in_mm
            sensor_positions = sensor_positions[:,::2] # leave out 3rd dimension

            # calculate the step size of the different rays
            source_pos_in_mm = torch.dstack([xx,yy])
            
            #if grid_points_middle:
            #    source_pos_in_mm += spacing_in_mm/2
            
            ds = (source_pos_in_mm[:, :, None, :] - sensor_positions[None, None, :, :]).pow(2).sum(-1).sqrt()              

            device_base_position_xy_mm = device_base_position_mm[::2] # leave out second component
            ########################################################################################################################################
            # in order to have the same like grid_sample function
            if grid_points_middle:
                source_pos_scaled = source_pos_in_mm/global_settings[Tags.SPACING_MM] - 0.5 # global pixel indices ##########################
                sensor_positions_scaled = sensor_positions/global_settings[Tags.SPACING_MM] - 0.5  ##########################################
            else: # in order to have the same like analytical function without shifting
                source_pos_scaled = source_pos_in_mm/global_settings[Tags.SPACING_MM] # global pixel indices
                sensor_positions_scaled = sensor_positions/global_settings[Tags.SPACING_MM]  
            #########################################################################################################################################
            del source_pos_in_mm
            del sensor_positions
        else:
            # global position according to whole image in mm
            xx = xx*spacing_in_mm + device_base_position_mm[0]
            yy = yy*spacing_in_mm + device_base_position_mm[2]
            sensor_positions = sensor_positions + device_base_position_mm
            sensor_positions = sensor_positions[:,::2] # leave out 3rd dimension

            # calculate the distances ds of all sources and detector sensors size (i.e. for all rays)
            source_pos_in_mm = torch.dstack([xx,yy])
            #if grid_points_middle:
            #    source_pos_in_mm += spacing_in_mm/2
            
            ds = (source_pos_in_mm[:, :, None, :] - sensor_positions[None, None, :, :]).pow(2).sum(-1).sqrt()        
            
            ########################
            # in order to have the same like grid_sample function
            if grid_points_middle:
                source_pos_scaled = source_pos_in_mm/global_settings[Tags.SPACING_MM] - 0.5 # global pixel indices
                sensor_positions_scaled = sensor_positions/global_settings[Tags.SPACING_MM] - 0.5  
            else: # in order to have the same like analytical function without shifting
                source_pos_scaled = source_pos_in_mm/global_settings[Tags.SPACING_MM] # global pixel indices
                sensor_positions_scaled = sensor_positions/global_settings[Tags.SPACING_MM]  
            ###############################################################################
            del source_pos_in_mm
            del sensor_positions

        # calculate the number of steps per ray, we want that for the longest ray that the ray is sampled at least
        # at every pixel
        steps = int(max((xdim**2+ydim**2)**0.5 + 0.5, ds.max()/(2**0.5*spacing_in_mm) + 0.5, STEPS_MIN))
        logger.debug(f"Using {steps} steps for numerical calculation of the line integrals")
        # steps        
        s = torch.linspace(0, 1, steps, device=torch_device)

        delays = torch.zeros(xdim, ydim, n_sensor_elements, dtype=torch.float64, device=torch_device)

        # loop over the sensors because vectorization of the loop needs too much memory
        for sensor in range(n_sensor_elements):
            # calculate the sampled positions (#steps samples) for all rays coming from all source points to a given sensor
            rays = (1-s[None, None, :, None])* source_pos_scaled[:, :, None, :] +  \
                (s[None, None, :, None] * sensor_positions_scaled[None, None, sensor, None, :]) # shape: xdim x ydim x #steps x 2           
            rays = rays.reshape(xdim*ydim*steps,2)
            # calculate the corresponding interpolated sos values at those points
            interpolated_sos = bilinear_interpolation(sos_tensor, rays[:,0], rays[:,1])
            print(interpolated_sos)
            print(rays)
            interpolated_sos = interpolated_sos.reshape(xdim*ydim, steps)
            # integrate the interpolated inverse sos-values
            integrals = torch.trapezoid(interpolated_sos)
            integrals = integrals.reshape((xdim, ydim))
            # calculate the delays by multiplying it with the step size ds/(steps-1) and divide it by the time spacing
            # in order to get the delay indices for the time series data
            delays[:,:, sensor] = integrals*ds[:,:,sensor]/((steps-1) * time_spacing_in_ms)

        return delays.unsqueeze(2), torch.arange(n_sensor_elements, device=torch_device), ds # xdim x ydim x 1 x #sensors

    else:
        logger.warning("3-dimensional heterogenous SoS reconstruction is not implemented yet.")


# $$
# \int_\gamma f ds = \int_0^1 f(\gamma(t))\cdot||\dot{\gamma}||\cdot\text{d}t \\
# $$
# For source positions $(x_S, y_S)$ and sensor positions $(x_E, y_E)$ we can parametrize a ray by
# $$
# \gamma(t) = \left(\begin{array}{c} x_S(1-t) + x_E \\ y_S (1-t) + y_E t \end{array}\right) \\
# \Rightarrow ||\dot{\gamma}(t)|| = \left|\left|\begin{array}{c} -x_S + x_E \\ -y_S + y_E \end{array}\right|\right| = \sqrt{(x_E-x_S)^2 + (y_E-y_S)^2} = D
# $$
# Thus using the distance between source point and sensor position the integral is given by:
# $$
# D \cdot \int_0^1 f(\gamma(t)) \cdot \text{d}t
# $$

# In[9]:


def analytical_solution(scalarfield_type, k, c, xdim, xdim_start, xdim_end, ydim_start, ydim_end, spacing_in_mm,
                        device_base_position_mm, sensor_positions, time_spacing_in_ms,
                        center_FOV_x):
    
    print(scalarfield_type)
    
    if center_FOV_x:
        x_offset = 0.5 if xdim % 2 == 0 else 0  # to ensure pixels are symmetrically arranged around the 0 like the
        # sensor positions, add an offset of 0.5 pixels if the dimension is even
        print("x_offset", x_offset)
    else:
        x_offset = 0
    
    x = xdim_start + torch.arange(xdim, device=torch_device, dtype=torch.float32) + x_offset
    y = ydim_start + torch.arange(ydim, device=torch_device, dtype=torch.float32)
    
    xx, yy = torch.meshgrid(x, y)
    source_positions = torch.dstack([xx*spacing_in_mm + device_base_position_mm[0], yy*spacing_in_mm + device_base_position_mm[2]])
    
    device_base_position_mm = torch.from_numpy(device_base_position_mm).to(torch_device)
    sensor_positions = sensor_positions + device_base_position_mm
    sensor_positions = sensor_positions[:,::2] # leave out 3rd dimension
      
    delays = torch.zeros(xdim, ydim, n_sensor_elements, dtype=torch.float64, device=torch_device)
    ds = torch.zeros(xdim, ydim, n_sensor_elements, dtype=torch.float64, device=torch_device)
    
    if scalarfield_type == "constant":
        for x_index in range(xdim):
            for y_index in range(ydim):
                for n in range(n_sensor_elements):
                    x_s, y_s = source_positions[x_index, y_index]
                    x_e, y_e = sensor_positions[n]
                    # distance
                    D = ((x_e-x_s)**2 + (y_e-y_s)**2)**0.5
                    delays[x_index, y_index, n] = D*c
                    ds[x_index, y_index, n] = D
    
    elif scalarfield_type == "horizontal_gradient":
        for x_index in range(xdim):
            for y_index in range(ydim):
                for n in range(n_sensor_elements):
                    x_s, y_s = source_positions[x_index, y_index]
                    x_e, y_e = sensor_positions[n]
                    # distance
                    D = ((x_e-x_s)**2 + (y_e-y_s)**2)**0.5
                    delays[x_index, y_index, n] = D*(c+k/2*(x_s+x_e))
                    ds[x_index, y_index, n] = D
    
    elif scalarfield_type == "vertical_gradient":
        for x_index in range(xdim):
            for y_index in range(ydim):
                for n in range(n_sensor_elements):
                    x_s, y_s = source_positions[x_index, y_index]
                    x_e, y_e = sensor_positions[n]
                    # distance
                    D = ((x_e-x_s)**2 + (y_e-y_s)**2)**0.5
                    delays[x_index, y_index, n] = D*(c+k/2*(y_s+y_e))
                    ds[x_index, y_index, n] = D
    
    elif scalarfield_type == "quadratic":
        for x_index in range(xdim):
            for y_index in range(ydim):
                for n in range(n_sensor_elements):
                    x_s, y_s = source_positions[x_index, y_index]
                    x_e, y_e = sensor_positions[n]
                    # distance
                    delta_x = x_e-x_s
                    delta_y = y_e-y_s
                    D = (delta_x**2 + delta_y**2)**0.5
                    ds[x_index, y_index, n] = D
                    
                    tilde_x = x_s-centre[0]
                    tilde_y = y_s-centre[1]                  
                    delays[x_index, y_index, n] = D*(tilde_x**2+tilde_y**2+c + delta_x*tilde_x+delta_y*tilde_y +1/3*D**2)
    
    return delays/time_spacing_in_ms, ds


# In[10]:


delays_num, ds_num = calculate_delays_for_heterogen_sos_OLD(sensor_positions, xdim, ydim, zdim, x, y, z, spacing_in_mm, time_spacing_in_ms,
                                                    speed_of_sound_in_m_per_s, n_sensor_elements, global_settings,
                                                    device_base_position_mm, logger, torch_device,
                                                    grid_points_middle, debug = True)

delays_num, ds_num = delays_num.squeeze().cpu().numpy(), ds_num.cpu().numpy()


# In[11]:


# using own interpolation
delays_num2, ds_num2 = calculate_delays_for_heterogen_sos(sensor_positions, xdim, ydim, zdim, x, y, z, spacing_in_mm, time_spacing_in_ms,
                                                    speed_of_sound_in_m_per_s, n_sensor_elements, global_settings,
                                                    device_base_position_mm, logger, torch_device,
                                                    grid_points_middle, debug = True)
delays_num2, ds_num2 = delays_num2.squeeze().cpu().numpy(), ds_num2.cpu().numpy()


# In[12]:


delays_ana, ds_ana = analytical_solution(scalarfield_type, k, c, xdim, xdim_start, xdim_end, ydim_start, ydim_end, spacing_in_mm,
                                 device_base_position_mm, sensor_positions, time_spacing_in_ms, center_FOV_x)
delays_ana, ds_ana = delays_ana.cpu().numpy(), ds_ana.cpu().numpy()


# In[13]:


print(sensor_positions.cpu().numpy())
print((sensor_positions.cpu().numpy() + device_base_position_mm)[:, ::2])
print((sensor_positions.cpu().numpy() + device_base_position_mm)[:, ::2]/global_settings[Tags.SPACING_MM])
print(device_base_position_mm)


# In[14]:


difference = np.abs(delays_num-delays_ana)
difference2 = np.abs(delays_num2-delays_ana)


# In[15]:


cols = 3
fig, axes = plt.subplots(n_sensor_elements, cols, figsize=(8*cols,6*n_sensor_elements))

for n in range(n_sensor_elements):
    im1 = axes[n,0].imshow(delays_num[:,:,n])
    plt.colorbar(im1, ax=axes[n, 0])
    im2 = axes[n,1].imshow(delays_ana[:,:,n])
    plt.colorbar(im2, ax=axes[n, 1])
    im3 = axes[n,2].imshow(delays_num2[:,:,n])
    plt.colorbar(im3, ax=axes[n, 2])

    #dif = axes[n,2].imshow(difference[:,:,n], cmap = "magma")
    #plt.colorbar(dif, ax=axes[n, 2])


# In[16]:


fig, axes = plt.subplots(n_sensor_elements, 2, figsize=(8*(cols-1),6*n_sensor_elements))

for n in range(n_sensor_elements):
    dif1 = axes[n,0].imshow(difference[:,:,n])
    plt.colorbar(dif1, ax=axes[n, 0])
    dif2 = axes[n,1].imshow(difference2[:,:,n])
    plt.colorbar(dif2, ax=axes[n, 1])


# ### Note: For grid_points_middle=True
# if the FOV is the same like the global volume dimensions, then the numerical and the analytical solution will differ for the boundary FOV-pixel, because they differ in how they treat the boundary of the sos-map:
# - the numerical solution assigns the global position [0 mm, 0 mm] the nearest sos-value at the position [spacing/2 mm, spacing/2 mm] (because we do not have any other neigbors between which one can interpolate (we clamp the boundary values to the nearest one)
# - the analytical solution assumes no step in the sos-map and assumes that the sos-scalarfield-function goes on also after the boundary.

# In[17]:


print("abs. diff of torch interpolation and analytical solution")
difference.max(), difference.mean(), difference.min()


# In[18]:


print("abs. diff of own interpolation and analytical solution")
difference2.max(), difference2.mean(), difference2.min()


# In[19]:


print("Distance differences")
np.abs(ds_ana - ds_num).max(), np.abs(ds_ana - ds_num2).max()


# In[20]:


fig, axes = plt.subplots(n_sensor_elements, 3, figsize=(6*cols,8*n_sensor_elements))

integral_num = delays_num[:,:,:]/ds_num[:,:,:]
integral_ana = delays_ana[:,:,:]/ds_ana[:,:,:]
integral_num2 = delays_num2[:,:,:]/ds_num2[:,:,:]

for n in range(n_sensor_elements):
    im1 = axes[n,0].imshow(integral_num[:,:,n].T)
    plt.colorbar(im1, ax=axes[n, 0])
    im2 = axes[n,1].imshow(integral_ana[:,:,n].T)
    plt.colorbar(im2, ax=axes[n, 1])
    im3 = axes[n,2].imshow(integral_num2[:,:,n].T)
    plt.colorbar(im3, ax=axes[n, 2])


# In[21]:


fig, axes = plt.subplots(1, 2, figsize=(28,8))

axes[0].hist(np.abs(integral_num-integral_ana).reshape(-1), bins=20)
axes[0].set_title(f"max = {np.abs(integral_num-integral_ana).max()}\nmean={np.abs(integral_num-integral_ana).mean()}\nmin={np.abs(integral_num-integral_ana).min()}", fontsize=16)

axes[1].hist(np.abs(integral_num2-integral_ana).reshape(-1), bins=20)
axes[1].set_title(f"max = {np.abs(integral_num2-integral_ana).max()}\nmean={np.abs(integral_num2-integral_ana).mean()}\nmin={np.abs(integral_num2-integral_ana).min()}", fontsize=16)
plt.show()
