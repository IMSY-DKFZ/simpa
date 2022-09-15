import unittest
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

import sys
class TesthDASNumericalIntegration(unittest.TestCase):
    """
    
    """

    def setUp(self, xglob = 400, yglob = 300, spac_glob = 0.1, spac_fov = 0.1):
        # Global Volume
        self.global_settings = Settings()

        self.xdim_glob = int(xglob)
        self.ydim_glob = int(yglob)

        self.global_settings[Tags.DIM_VOLUME_X_MM] = self.xdim_glob * spac_glob
        self.global_settings[Tags.DIM_VOLUME_Z_MM] = self.ydim_glob * spac_glob
        self.global_settings[Tags.SPACING_MM] = spac_glob

        print(f"Global Dimensions {self.xdim_glob} x {self.ydim_glob}")

        # FOV
        self.xdim = self.xdim_glob//2 #xdim_glob     # pixels (xdim_mm = 10 mm)
        self.ydim = self.ydim_glob//2 #ydim_glob        # pixels (ydim_mm = 15 mm)
        self.xdim = np.clip(self.xdim, 2, self.xdim_glob-2) # for xdim >= xdim_glob-1: boundary miscalculations in the analytical solution
        self.ydim = np.clip(self.ydim, 2, self.ydim_glob-2)
        self.zdim = 1
        self.xdim_start = -self.xdim//2
        self.xdim_end = self.xdim  + self.xdim_start
        self.ydim_start = -self.ydim//2
        self.ydim_end = self.ydim + self.ydim_start
        self.zdim_start = 1
        self.zdim_end = 1

        print(f"FOV Dimensions {self.xdim} x {self.ydim} \
               from [{self.xdim_start},{self.ydim_start}] to [{self.xdim_end},{self.ydim_end}]")

        self.time_spacing_in_ms = 1 # we just one to compare the integration
        self.spacing_in_mm = spac_fov

        self.logger = Logger()
        self.torch_device = torch.device("cuda")
        
        self.device_base_position_mm = np.array([self.global_settings[Tags.DIM_VOLUME_X_MM]/2,
                                                 self.global_settings[Tags.SPACING_MM],
                                                 self.global_settings[Tags.DIM_VOLUME_Z_MM]/2])  

        # Simulate sensors
        RANDOM_SENSORS = False # Should be False for Reproducibility or True for test random sensors
        self.n_sensor_elements = 2
        if RANDOM_SENSORS: 
            # sample sensors
            sensor_distri = Uniform(
                low=torch.tensor([-self.global_settings[Tags.DIM_VOLUME_X_MM]/2+self.global_settings[Tags.SPACING_MM]/4,
                                  0,
                                  -self.global_settings[Tags.DIM_VOLUME_Z_MM]/2]+self.global_settings[Tags.SPACING_MM]/4).to(self.torch_device),
                high=torch.tensor([self.global_settings[Tags.DIM_VOLUME_X_MM]/2-self.global_settings[Tags.SPACING_MM]/4,
                                   0.01,
                                   self.global_settings[Tags.DIM_VOLUME_Z_MM]/2]-self.global_settings[Tags.SPACING_MM]/4).to(self.torch_device))
            self.sensor_positions = sensor_distri.sample((self.n_sensor_elements,))
        else:
            self.sensor_positions = torch.tensor([[self.xdim/4*self.spacing_in_mm, 1e-5, -self.ydim/4*self.spacing_in_mm],
                                                  [-self.xdim/2*self.spacing_in_mm, 1e-5,  self.ydim/2*self.spacing_in_mm]],
                                                  device=self.torch_device)
            self.n_sensor_elements = 2

        # Account for centered source points and for the thing that pixel positions correspond to the centre of the pixel
        self.grid_points_middle = True # should be true
        self.center_FOV_x = True # should be true

        # * grid_points_middle = True: 
        #     
        #         assumes that the position of the first sos pixel is at [global_spacing/2 mm, global_spacing/2 mm] and is spanned over [0 mm,0 mm] to [global_spacing mm, global_spacing mm]
        #         
        #         
        # * grid_points_middle = False: 
        #         
        #         assumes that the position of the first sos pixel is at [0 mm ,0 mm]  and is spanned over [-global_spacing/2 mm, -global_spacing/2 mm] to [global_spacing/2 mm, global_spacing/2 mm]

        # ### Note: For grid_points_middle=True
        # if the FOV is the same like the global volume dimensions, then the numerical and the analytical solution will differ for the boundary FOV-pixel, because they differ in how they treat the boundary of the sos-map:
        # - the numerical solution assigns the global position [0 mm, 0 mm] the nearest sos-value at the position [spacing/2 mm, spacing/2 mm] (because we do not have any other neigbors between which one can interpolate (we clamp the boundary values to the nearest one)
        # - the analytical solution assumes no step in the sos-map and assumes that the sos-scalarfield-function goes on also after the boundary.

        self.scalarfield_type_list = ["constant", "horizontal_gradient", "vertical_gradient", "quadratic"]

    def create_scalarfield(self, scalarfield_type="horizontal_gradient")-> None:
        """
        creates heterogenous maps
        """
        if scalarfield_type == "constant":
            self.c = 1500
            self.k = 1

            self.scalarfield = np.ones((self.xdim_glob,self.ydim_glob)) * self.c
        elif scalarfield_type == "horizontal_gradient":
            self.c = 1400
            self.k = int(200/self.global_settings[Tags.DIM_VOLUME_X_MM])

            x_pos_mm = np.arange(self.xdim_glob)*self.global_settings[Tags.SPACING_MM]
            if self.grid_points_middle:
                x_pos_mm += 0.5*self.global_settings[Tags.SPACING_MM]
            self.scalarfield = np.ones((self.xdim_glob,self.ydim_glob)) * (self.k*x_pos_mm[:,None] + self.c)
        elif scalarfield_type == "vertical_gradient":
            self.c = 1400
            self.k = int(200/self.global_settings[Tags.DIM_VOLUME_Z_MM])

            y_pos_mm = np.arange(self.ydim_glob)*self.global_settings[Tags.SPACING_MM]
            if self.grid_points_middle:
                y_pos_mm += 0.5*self.global_settings[Tags.SPACING_MM] 
            self.scalarfield = np.ones((self.xdim_glob,self.ydim_glob)) * (self.k*y_pos_mm[None,:] + self.c)
        elif scalarfield_type == "quadratic":
            self.c = 100
            self.k = 1
            self.centre = [self.global_settings[Tags.DIM_VOLUME_X_MM]/2, self.global_settings[Tags.DIM_VOLUME_Z_MM]/2]

            x_pos_mm = np.arange(self.xdim_glob)*self.global_settings[Tags.SPACING_MM]
            y_pos_mm = np.arange(self.ydim_glob)*self.global_settings[Tags.SPACING_MM]
            if self.grid_points_middle:
                x_pos_mm += 0.5*self.global_settings[Tags.SPACING_MM]
                y_pos_mm += 0.5*self.global_settings[Tags.SPACING_MM]
            xgrid, ygrid = np.meshgrid(x_pos_mm, y_pos_mm) 
            scalarfield = (xgrid-self.centre[0])**2+(ygrid-self.centre[1])**2 + self.c
            self.scalarfield = scalarfield.T # in order to be shape xdim_glob x ydim_glob
        else:
            raise("Error")

        scalarfield_3d = self.scalarfield[:,None,:]
        self.speed_of_sound_in_m_per_s = 1/scalarfield_3d

    def get_xyz(self, center_FOV_x):
        """
        helper function to compute the positions of the FOV points(=source points) taking into account the symmetry in x
        direction (off by one fix by Tom)
        """
        if center_FOV_x:
            x_offset = 0.5 if self.xdim % 2 == 0 else 0  # to ensure pixels are symmetrically arranged around the 0 like the
            # sensor positions, add an offset of 0.5 pixels if the dimension is even
        else:
            x_offset = 0

        x = self.xdim_start + torch.arange(self.xdim, device=self.torch_device, dtype=torch.float32) + x_offset
        y = self.ydim_start + torch.arange(self.ydim, device=self.torch_device, dtype=torch.float32)
        if self.zdim == 1:
            z = torch.arange(self.zdim, device=self.torch_device, dtype=torch.float32)
        else:
            z = self.zdim_start + torch.arange(self.zdim, device=self.torch_device, dtype=torch.float32)
        return x, y, z

    def analytical_solution(self, scalarfield_type):
    
        print(scalarfield_type)
              
        xx, yy = torch.meshgrid(self.x, self.y)
        source_positions = torch.dstack([xx*self.spacing_in_mm + self.device_base_position_mm[0], yy*self.spacing_in_mm + self.device_base_position_mm[2]])
        
        device_base_position_mm = torch.from_numpy(self.device_base_position_mm).to(self.torch_device)
        sensor_positions = self.sensor_positions + device_base_position_mm
        sensor_positions = sensor_positions[:,::2] # leave out 3rd dimension
        
        delays = torch.zeros(self.xdim, self.ydim, self.n_sensor_elements, dtype=torch.float64, device=self.torch_device)
        ds = torch.zeros(self.xdim, self.ydim, self.n_sensor_elements, dtype=torch.float64, device=self.torch_device)
        
        if scalarfield_type == "constant":
            for x_index in range(self.xdim):
                for y_index in range(self.ydim):
                    for n in range(self.n_sensor_elements):
                        x_s, y_s = source_positions[x_index, y_index]
                        x_e, y_e = sensor_positions[n]
                        # distance
                        D = ((x_e-x_s)**2 + (y_e-y_s)**2)**0.5
                        delays[x_index, y_index, n] = D*self.c
                        ds[x_index, y_index, n] = D
        
        elif scalarfield_type == "horizontal_gradient":
            for x_index in range(self.xdim):
                for y_index in range(self.ydim):
                    for n in range(self.n_sensor_elements):
                        x_s, y_s = source_positions[x_index, y_index]
                        x_e, y_e = sensor_positions[n]
                        # distance
                        D = ((x_e-x_s)**2 + (y_e-y_s)**2)**0.5
                        delays[x_index, y_index, n] = D*(self.c+self.k/2*(x_s+x_e))
                        ds[x_index, y_index, n] = D
        
        elif scalarfield_type == "vertical_gradient":
            for x_index in range(self.xdim):
                for y_index in range(self.ydim):
                    for n in range(self.n_sensor_elements):
                        x_s, y_s = source_positions[x_index, y_index]
                        x_e, y_e = sensor_positions[n]
                        # distance
                        D = ((x_e-x_s)**2 + (y_e-y_s)**2)**0.5
                        delays[x_index, y_index, n] = D*(self.c+self.k/2*(y_s+y_e))
                        ds[x_index, y_index, n] = D
        
        elif scalarfield_type == "quadratic":
            for x_index in range(self.xdim):
                for y_index in range(self.ydim):
                    for n in range(self.n_sensor_elements):
                        x_s, y_s = source_positions[x_index, y_index]
                        x_e, y_e = sensor_positions[n]
                        # distance
                        delta_x = x_e-x_s
                        delta_y = y_e-y_s
                        D = (delta_x**2 + delta_y**2)**0.5
                        ds[x_index, y_index, n] = D
                        
                        tilde_x = x_s-self.centre[0]
                        tilde_y = y_s-self.centre[1]                  
                        delays[x_index, y_index, n] = D*(tilde_x**2+tilde_y**2+self.c + delta_x*tilde_x+delta_y*tilde_y +1/3*D**2)
        
        return delays/self.time_spacing_in_ms, ds

    def test(self, visualize, scalarfield_type = None) -> None:
        self.x, self.y, self.z = self.get_xyz(self.center_FOV_x)

        if scalarfield_type == None:
            for scalarfield_type in self.scalarfield_type_list:
                print("\n############################################\nScalarfield", scalarfield_type, "\n############################################")
                self.run_for_given_type(visualize=visualize, scalarfield_type=scalarfield_type)
        else:
            print("\n############################################\nScalarfield", scalarfield_type, "\n############################################")
            self.run_for_given_type(visualize=visualize, scalarfield_type=scalarfield_type)

    def run_for_given_type(self, visualize=True, scalarfield_type="horizontal_gradient"):

        self.create_scalarfield(scalarfield_type)



        self.delays_num, self.ds_num, self.interpols = calculate_delays_for_heterogen_sos_OLD(
                                    self.sensor_positions, self.xdim, self.ydim, self.zdim, self.x, self.y, self.z, self.spacing_in_mm,
                                    self.time_spacing_in_ms, self.speed_of_sound_in_m_per_s, self.n_sensor_elements, self.global_settings,
                                    self.device_base_position_mm, self.logger, self.torch_device,
                                    self.grid_points_middle,
                                    global_ds_calc = True, get_ds = True, get_interpols=True, verbose = False, uneven_steps = True
                                    )

        self.delays_num, self.ds_num = self.delays_num.squeeze().cpu().numpy(), self.ds_num.cpu().numpy()

        # using own interpolation
        self.delays_num2, self.ds_num2, self.interpols2 = calculate_delays_for_heterogen_sos(
                                    self.sensor_positions, self.xdim, self.ydim, self.zdim, self.x, self.y, self.z, self.spacing_in_mm,
                                    self.time_spacing_in_ms, self.speed_of_sound_in_m_per_s, self.n_sensor_elements, self.global_settings,
                                    self.device_base_position_mm, self.logger, self.torch_device,
                                    self.grid_points_middle,
                                    global_ds_calc = True, get_ds = True, get_interpols=True, verbose = False, uneven_steps = True
                                    )
        self.delays_num2, self.ds_num2 = self.delays_num2.squeeze().cpu().numpy(), self.ds_num2.cpu().numpy()

        # Analytical Integration
        self.delays_ana, self.ds_ana = self.analytical_solution(scalarfield_type)
        self.delays_ana, self.ds_ana = self.delays_ana.cpu().numpy(), self.ds_ana.cpu().numpy()

        # Calculate differences
        self.difference = np.abs(self.delays_num - self.delays_ana)
        self.difference2 = np.abs(self.delays_num2 - self.delays_ana)
        self.difference_nums = np.abs(self.delays_num - self.delays_num2)

        print("Absolute difference of delays")
        print(" - numerical delays (grid_sample interpol.) and analytical delays")
        print(f"    max={self.difference.max()}\n    mean={self.difference.mean()}\n    min={self.difference.min()}")
        print(" - numerical delays (own interpol.) and analytical delays")
        print(f"    max={self.difference2.max()}\n    mean={self.difference2.mean()}\n    min={self.difference2.min()}")
        print(" - numerical delays (own interpol.) and numerical delays (grid_sample interpol.)")
        print(f"    max={self.difference_nums.max()}\n    mean={self.difference_nums.mean()}\n    min={self.difference_nums.min()}")

        # TODO: ############################ ACTUAL TEST #########################
        self.tolerarance = 0.5
        if scalarfield_type == "quadratic":
            self.tolerance = 2
        #assert difference.max() < self.tolerance
        #assert difference2.max() < self.tolerance
        ####################################

        ds_diff_num = np.abs(self.ds_ana - self.ds_num)
        ds_diff_num2 = np.abs(self.ds_ana - self.ds_num2)
        if ds_diff_num.max() > 0.0 or ds_diff_num2.max() > 0.0:
            print("Maximal distance differences")
            print(" - numerical delays (grid_sample interpol.) and analytical ds")
            print(f"    max={ds_diff_num.max()}")
            print(" - numerical delays (own interpol.) and analytical ds") 
            print(f"    max={ds_diff_num2.max()}")

        # Integrals
        # avoid division by 0
        self.ds_num[self.ds_num==0] = 42
        self.ds_num2[self.ds_num2==0] = 42
        self.ds_ana[self.ds_ana==0] = 42
        self.integral_num = self.delays_num[:,:,:]/self.ds_num[:,:,:]
        self.integral_ana = self.delays_ana[:,:,:]/self.ds_ana[:,:,:]
        self.integral_num2 = self.delays_num2[:,:,:]/self.ds_num2[:,:,:]
        # Integration differences
        self.int_difference = np.abs(self.integral_num - self.integral_ana)
        self.int_difference2 = np.abs(self.integral_num2 - self.integral_ana)
        self.int_difference_num = np.abs(self.integral_num - self.integral_num2)
        print("Absolute difference of integrals")
        print(" - numerical integrals (grid_sample interpol.) and analytical integrals")
        print(f"    max={self.int_difference.max()}\n    mean={self.int_difference.mean()}\n    min={self.int_difference.min()}")
        print(" - numerical integrals (own interpol.) and analytical integrals")
        print(f"    max={self.int_difference2.max()}\n    mean={self.int_difference2.mean()}\n    min={self.int_difference2.min()}")
        print(" - numerical integrals (own interpol.) and numerical integrals (grid_sample interpol.)")
        print(f"    max={self.int_difference_num.max()}\n    mean={self.int_difference_num.mean()}\n    min={self.int_difference_num.min()}")

        print("Interpolation")
        print("grid sample")
        print(self.interpols)
        print("own")
        print(self.interpols2)
        print("deviations")
        print(np.abs(self.interpols-self.interpols2))
        print("    max difference = ", np.abs(self.interpols-self.interpols2).max())

        if visualize:
            self.visualize_setting()
            self.visualize_delays()
            self.visualize_differences()
            self.visualize_delay_difference_hist()
            self.visualize_integral_differences()
            self.visualize_integral_difference_hist()

    def visualize_setting(self) -> None:
        """
        visualizes the scalarfield, the sensors, the FOV and the device base position 
        """           
        sens_pix = (self.sensor_positions.cpu().numpy()+self.device_base_position_mm)/self.global_settings[Tags.SPACING_MM]
        device_pix = self.device_base_position_mm/self.global_settings[Tags.SPACING_MM]

        FOV_global_pix_factor = self.spacing_in_mm/self.global_settings[Tags.SPACING_MM]
        xdim_start_glob = self.xdim_start*FOV_global_pix_factor + device_pix[0]
        ydim_start_glob = self.ydim_start*FOV_global_pix_factor + device_pix[2]

        plt.figure("Scalarfield", figsize=(6,12))
        ax = plt.gca()
        plt.imshow(self.scalarfield.T, extent=[0, self.xdim_glob, self.ydim_glob, 0])
        plt.colorbar()
        # Create a Rectangle patch
        rect = patches.Rectangle((xdim_start_glob,ydim_start_glob),self.xdim*FOV_global_pix_factor, self.ydim*FOV_global_pix_factor,
                                linewidth=1, edgecolor='grey', facecolor='none', linestyle="--", label="FOV")
        # Add the patch to the Axes
        ax.add_patch(rect)
        plt.scatter(sens_pix[:,0], sens_pix[:,2], color="red", label="sensors")
        plt.scatter(device_pix[0], device_pix[2], color="orange", label="device base position")
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), facecolor='whitesmoke')
        plt.show()

    def visualize_delays(self) -> None:
        cols = 3
        fig, axes = plt.subplots(self.n_sensor_elements, cols, figsize=(8*cols,6*self.n_sensor_elements))
        fig.canvas.manager.set_window_title("Delays of the FOV points")
        for n in range(self.n_sensor_elements):
            im1 = axes[n,0].imshow(self.delays_num[:,:,n])
            plt.colorbar(im1, ax=axes[n, 0])
            im2 = axes[n,1].imshow(self.delays_ana[:,:,n])
            plt.colorbar(im2, ax=axes[n, 1])
            im3 = axes[n,2].imshow(self.delays_num2[:,:,n])
            plt.colorbar(im3, ax=axes[n, 2])
        plt.show()

    def visualize_differences(self) -> None:
        cols = 3
        fig, axes = plt.subplots(self.n_sensor_elements, 2, figsize=(8*(cols-1),6*self.n_sensor_elements))
        fig.canvas.manager.set_window_title("Delay Differences")
        for n in range(self.n_sensor_elements):
            dif1 = axes[n,0].imshow(self.difference[:,:,n])
            plt.colorbar(dif1, ax=axes[n, 0])
            dif2 = axes[n,1].imshow(self.difference2[:,:,n])
            plt.colorbar(dif2, ax=axes[n, 1])
        plt.show()

    def visualize_delay_difference_hist(self) -> None:
        fig, axes = plt.subplots(1, 2, figsize=(28,8))
        fig.canvas.manager.set_window_title("Analysis of the Delay Differences")
        fig.suptitle('Delay Differences', fontsize=18)

        axes[0].hist(self.difference.reshape(-1), bins=20)
        axes[0].set_title(f"grid_sample()\nmax = {self.difference.max()}\nmean={self.difference.mean()}\nmin={self.difference.min()}", fontsize=16)

        axes[1].hist(self.difference2.reshape(-1), bins=20)
        axes[1].set_title(f"own interpol.\nmax = {self.difference2.max()}\nmean={self.difference2.mean()}\nmin={self.difference2.min()}", fontsize=16)
        plt.show()

    def visualize_integral_differences(self) -> None:
        cols = 3 
        fig, axes = plt.subplots(self.n_sensor_elements, 3, figsize=(6*cols,8*self.n_sensor_elements))
        fig.canvas.manager.set_window_title("Integrals")
        for n in range(self.n_sensor_elements):
            im1 = axes[n,0].imshow(self.integral_num[:,:,n].T)
            plt.colorbar(im1, ax=axes[n, 0])
            im2 = axes[n,1].imshow(self.integral_ana[:,:,n].T)
            plt.colorbar(im2, ax=axes[n, 1])
            im3 = axes[n,2].imshow(self.integral_num2[:,:,n].T)
            plt.colorbar(im3, ax=axes[n, 2])
        plt.show()

    def visualize_integral_difference_hist(self) -> None:      
        fig, axes = plt.subplots(1, 2, figsize=(28,8))
        fig.canvas.manager.set_window_title("Integral Differences")
        fig.suptitle('Integral Differences', fontsize=18)
        axes[0].hist(self.int_difference.reshape(-1), bins=20)
        axes[0].set_title(f"grid_sample()\nmax = {self.int_difference.max()}\nmean={self.int_difference.mean()}\nmin={self.int_difference.min()}", fontsize=16)

        axes[1].hist(self.int_difference2.reshape(-1), bins=20)
        axes[1].set_title(f"own interpol.\nmax = {self.int_difference2.max()}\nmean={self.int_difference2.mean()}\nmin={self.int_difference2.min()}", fontsize=16)
        plt.show()


if __name__ == "__main__":
    test = TesthDASNumericalIntegration()
    test.setUp(4,8,0.5,0.5)
    test.test(visualize=True, scalarfield_type="horizontal_gradient")

    #test.setUp()
    #test.test(visualize=False)