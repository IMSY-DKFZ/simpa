import unittest
from simpa.utils import Tags
import torch
from torch.distributions.uniform import Uniform
import numpy as np
from simpa.utils.settings import Settings
from simpa.log.file_logger import Logger
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import Tuple
from simpa.core.simulation_modules.reconstruction_module.reconstruction_utils import calculate_delays_for_heterogen_sos

class TesthDASNumericalIntegration(unittest.TestCase):
    """
    This test is an automatic test, however one can also run the script and visualize the results and get more detailed prints
    This test tests the heterogenous delay calculation hDAS.
    usage:
    - run automatic tests with minimal prints
        test.setUp()
        test.test()
    - run tests for given scalarfield type showing plots and printing informations about the differences of delays, ds and integrals
        test.setUp()
        test.test(visualize=True, verbosity=["delays", "ds", "integrals"], scalarfield_type="horizontal_gradient")
    - visualize a specific scalarfield type and the corresponding SoS-map
        test.setUp()
        test.create_scalarfield("quadratic")
        test.visualize_setting(show=True)
    """

    def setUp(self, xdim_glob: int = 400, ydim_glob: int = 500, spac_glob: float = 0.1, xdim_fov: int = None, ydim_fov: int = None,
              spac_fov: float = 0.1, PAI_realistic_units: bool = True) -> None:
        """
        sets up volume, FOV, sensor positions, physical properties
        """
        # Global Volume
        self.global_settings = Settings()

        self.xdim_glob = int(xdim_glob)
        self.ydim_glob = int(ydim_glob)

        self.global_settings[Tags.DIM_VOLUME_X_MM] = self.xdim_glob * spac_glob
        self.global_settings[Tags.DIM_VOLUME_Z_MM] = self.ydim_glob * spac_glob
        self.global_settings[Tags.SPACING_MM] = spac_glob

        print(f"Global Dimensions {self.xdim_glob} x {self.ydim_glob}")

        # FOV
        self.xdim = int(xdim_glob)//2 if xdim_fov == None else xdim_fov # self.xdim_glob//2 #xdim_glob     # pixels (xdim_mm = 10 mm)
        self.ydim = int(ydim_glob)//2 if ydim_fov == None else ydim_fov #self.ydim_glob//2 #ydim_glob        # pixels (ydim_mm = 15 mm)
        # Note: 
        # if FOV dimensions == Global volume dimensions: numerical and the analytical solution will differ for the boundary FOV-pixel,
        # because they differ in how they treat the boundary of the SoS-map:
        #  - the numerical solution assigns the global position [0 mm, 0 mm] the nearest sos-value at the position [spacing/2 mm, spacing/2 mm] 
        #    (because we do not have any other neigbors between which one can interpolate [we clamp the boundary values to the nearest one])
        #  - the analytical solution assumes no step in the SoS-map and assumes that the SoS-scalarfield-function goes on also after the boundary.
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

        self.PAI_realistic_units = PAI_realistic_units
        self.time_spacing_in_ms = 2.5e-8 if self.PAI_realistic_units else 1 # 1 if we just one to compare the integration
        self.spacing_in_mm = spac_fov

        self.logger = Logger()
        self.torch_device = torch.device("cpu") #torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        
        self.device_base_position_mm = np.array([self.global_settings[Tags.DIM_VOLUME_X_MM]/2,
                                                 self.global_settings[Tags.SPACING_MM],
                                                 self.global_settings[Tags.DIM_VOLUME_Z_MM]/2])  

        # Simulate sensors
        self.RANDOM_SENSORS = False # Should be False for Reproducibility or True to test random sensors
        self.n_sensor_elements = 10
        if self.RANDOM_SENSORS: 
            # sample sensors
            sensor_distri = Uniform(
                low=torch.tensor([-self.global_settings[Tags.DIM_VOLUME_X_MM]/2+self.global_settings[Tags.SPACING_MM]/4,
                                  0,
                                  -self.global_settings[Tags.DIM_VOLUME_Z_MM]/2+self.global_settings[Tags.SPACING_MM]/4],
                                   dtype=torch.float64, device=self.torch_device),
                high=torch.tensor([self.global_settings[Tags.DIM_VOLUME_X_MM]/2-self.global_settings[Tags.SPACING_MM]/4,
                                   0.01,
                                   self.global_settings[Tags.DIM_VOLUME_Z_MM]/2-self.global_settings[Tags.SPACING_MM]/4],
                                   dtype=torch.float64, device = self.torch_device))
            self.sensor_positions = sensor_distri.sample((self.n_sensor_elements,))
        else:
            self.sensor_positions = torch.tensor([[self.xdim/4*self.spacing_in_mm, 1e-5, -self.ydim/4*self.spacing_in_mm],
                                                  [-self.xdim/2*self.spacing_in_mm, 1e-5,  self.ydim/2*self.spacing_in_mm]],
                                                  device=self.torch_device)
            self.n_sensor_elements = 2

        self.scalarfield_type_list = ["constant", "horizontal_gradient", "vertical_gradient", "quadratic"]

    def create_scalarfield(self, scalarfield_type: str = "horizontal_gradient")-> None:
        """
        creates heterogenous maps
        """
        if scalarfield_type == "constant":
            self.c = 1/1500 if self.PAI_realistic_units else 1500 #1500
            self.k = 1

            self.scalarfield = np.ones((self.xdim_glob,self.ydim_glob)) * self.c
        elif scalarfield_type == "horizontal_gradient":
            """
            going from range 1/1600 up to 1/1379.31 if self.PAI_realistic units otherwise go from 1400 to 1600
            """
            self.c = 1/1600 if self.PAI_realistic_units else 1400
            self.k = 1e-4/self.global_settings[Tags.DIM_VOLUME_X_MM] if self.PAI_realistic_units else int(200/self.global_settings[Tags.DIM_VOLUME_X_MM])

            x_pos_mm = np.arange(self.xdim_glob)*self.global_settings[Tags.SPACING_MM]
            # pixel [0,0] denotes to position [spacing/2, spacing/2]
            x_pos_mm += 0.5*self.global_settings[Tags.SPACING_MM]
            self.scalarfield = np.ones((self.xdim_glob,self.ydim_glob)) * (self.k*x_pos_mm[:,None] + self.c)
        elif scalarfield_type == "vertical_gradient":
            """
            going from range 1/1600 up to 1/1379.31 if self.PAI_realistic units otherwise go from 1400 to 1600
            """
            self.c = 1/1600 if self.PAI_realistic_units else 1400
            self.k = 1e-4/self.global_settings[Tags.DIM_VOLUME_Z_MM] if self.PAI_realistic_units else int(200/self.global_settings[Tags.DIM_VOLUME_Z_MM])

            y_pos_mm = np.arange(self.ydim_glob)*self.global_settings[Tags.SPACING_MM]
            # pixel [0,0] denotes to position [spacing/2, spacing/2]
            y_pos_mm += 0.5*self.global_settings[Tags.SPACING_MM] 
            self.scalarfield = np.ones((self.xdim_glob,self.ydim_glob)) * (self.k*y_pos_mm[None,:] + self.c)
        elif scalarfield_type == "quadratic":
            self.c = None # see below
            self.k = None # see below
            self.centre = [self.global_settings[Tags.DIM_VOLUME_X_MM]/2, self.global_settings[Tags.DIM_VOLUME_Z_MM]/2]

            x_pos_mm = np.arange(self.xdim_glob)*self.global_settings[Tags.SPACING_MM]
            y_pos_mm = np.arange(self.ydim_glob)*self.global_settings[Tags.SPACING_MM]
            # pixel [0,0] denotes to position [spacing/2, spacing/2]
            x_pos_mm += 0.5*self.global_settings[Tags.SPACING_MM]
            y_pos_mm += 0.5*self.global_settings[Tags.SPACING_MM]
            xgrid, ygrid = np.meshgrid(x_pos_mm, y_pos_mm) 
            quadratic = (xgrid-self.centre[0])**2+(ygrid-self.centre[1])**2
            #scale
            max_mm = max(self.global_settings[Tags.DIM_VOLUME_Z_MM]/2, self.global_settings[Tags.DIM_VOLUME_Z_MM]/2)
            self.k = 0.00008/max_mm**2 if self.PAI_realistic_units else 1
            #shift
            self.c = 1/1650 if self.PAI_realistic_units else 1
            scalarfield = self.k*quadratic+self.c
            self.scalarfield = scalarfield.T # in order to be shape xdim_glob x ydim_glob
        else:
            raise("Error")

        scalarfield_3d = self.scalarfield[:,None,:]
        self.speed_of_sound_in_m_per_s = 1/scalarfield_3d

    def get_xyz(self) -> Tuple[torch.tensor, torch.tensor, torch.tensor]:
        """
        helper function to compute the positions of the FOV points(=source points) taking into account the symmetry in x
        direction (off by one fix by Tom)
        """
        x_offset = 0.5 if self.xdim % 2 == 0 else 0  # to ensure pixels are symmetrically arranged around the 0 like the
        # sensor positions, add an offset of 0.5 pixels if the dimension is even
        
        x = self.xdim_start + torch.arange(self.xdim, device=self.torch_device, dtype=torch.float32) + x_offset
        y = self.ydim_start + torch.arange(self.ydim, device=self.torch_device, dtype=torch.float32)
        if self.zdim == 1:
            z = torch.arange(self.zdim, device=self.torch_device, dtype=torch.float32)
        else:
            z = self.zdim_start + torch.arange(self.zdim, device=self.torch_device, dtype=torch.float32)
        return x, y, z

    def analytical_solution(self, scalarfield_type: str) -> Tuple[torch.tensor, torch.tensor]:
        """
        Calculates the delays using an analytical formula (derived by hand) and also returns the used distances
        """
        print(f"\nCalculate analytical solution for {scalarfield_type} scalarfield\n")
        xx, yy = torch.meshgrid(self.x, self.y)
        source_positions = torch.dstack([xx*self.spacing_in_mm, yy*self.spacing_in_mm])
        device_base_position_mm = self.device_base_position_mm.copy()
        device_base_position_mm_x = device_base_position_mm[0]
        device_base_position_mm_y = device_base_position_mm[2]

        sensor_positions = self.sensor_positions.clone()
        sensor_positions = sensor_positions[:,::2] # leave out 3rd dimension
        
        delays = torch.zeros(self.xdim, self.ydim, self.n_sensor_elements, dtype=torch.float64, device=self.torch_device)
        ds_ref = torch.zeros(self.xdim, self.ydim, self.n_sensor_elements, dtype=torch.float64, device=self.torch_device)
        
        if scalarfield_type == "constant":
            for x_index in range(self.xdim):
                for y_index in range(self.ydim):
                    for n in range(self.n_sensor_elements):
                        x_s, y_s = source_positions[x_index, y_index]
                        x_e, y_e = sensor_positions[n]
                        # distance calculated in FOV system (invariant under translation)
                        D = ((x_e-x_s)**2 + (y_e-y_s)**2)**0.5
                        # Calculate delays
                        delays[x_index, y_index, n] = D*self.c
                        ds_ref[x_index, y_index, n] = D
        
        elif scalarfield_type == "horizontal_gradient":
            for x_index in range(self.xdim):
                for y_index in range(self.ydim):
                    for n in range(self.n_sensor_elements):
                        x_s, y_s = source_positions[x_index, y_index]
                        x_e, y_e = sensor_positions[n]
                        # distance calculated in FOV system
                        D = ((x_e-x_s)**2 + (y_e-y_s)**2)**0.5
                        # go into global coordinate system
                        x_s_glob = x_s + device_base_position_mm_x 
                        x_e_glob = x_e + device_base_position_mm_x
                        #y_s_glob = y_s + device_base_position_mm_y
                        #y_e_glob = y_e + device_base_position_mm_y
                        # Calculate delays
                        delays[x_index, y_index, n] = D*(self.c+self.k/2*(x_s_glob+x_e_glob))
                        ds_ref[x_index, y_index, n] = D
        
        elif scalarfield_type == "vertical_gradient":
            for x_index in range(self.xdim):
                for y_index in range(self.ydim):
                    for n in range(self.n_sensor_elements):
                        x_s, y_s = source_positions[x_index, y_index]
                        x_e, y_e = sensor_positions[n]
                        # distance calculated in FOV system
                        D = ((x_e-x_s)**2 + (y_e-y_s)**2)**0.5
                        # go into global coordinate system
                        #x_s_glob = x_s + device_base_position_mm_x 
                        #x_e_glob = x_e + device_base_position_mm_x
                        y_s_glob = y_s + device_base_position_mm_y
                        y_e_glob = y_e + device_base_position_mm_y
                        # Calculate delays
                        delays[x_index, y_index, n] = D*(self.c+self.k/2*(y_s_glob+y_e_glob))
                        ds_ref[x_index, y_index, n] = D
        
        elif scalarfield_type == "quadratic":
            for x_index in range(self.xdim):
                for y_index in range(self.ydim):
                    for n in range(self.n_sensor_elements):
                        x_s, y_s = source_positions[x_index, y_index]
                        x_e, y_e = sensor_positions[n]
                        # distances calculated in FOV system
                        delta_x = x_e-x_s
                        delta_y = y_e-y_s
                        D = (delta_x**2 + delta_y**2)**0.5
                        # go into global coordinate system
                        x_s_glob = x_s + device_base_position_mm_x 
                        #x_e_glob = x_e + device_base_position_mm_x
                        y_s_glob = y_s + device_base_position_mm_y
                        #y_e_glob = y_e + device_base_position_mm_y
                        # Calculate delays
                        tilde_x = x_s_glob-self.centre[0]
                        tilde_y = y_s_glob-self.centre[1]                  
                        delays[x_index, y_index, n] = D*self.k*(tilde_x**2+tilde_y**2 + delta_x*tilde_x + delta_y*tilde_y + 1/3*D**2) \
                                                      + D*self.c
                        ds_ref[x_index, y_index, n] = D
        
        del source_positions, sensor_positions
        return delays/self.time_spacing_in_ms, ds_ref

    def test(self, visualize=False, verbosity: list = [], scalarfield_type: str = None) -> None:
        """
        test function to automatically test for given or specified scalarfield types
        """
        self.x, self.y, self.z = self.get_xyz()

        if scalarfield_type == None:
            for scalarfield_type in self.scalarfield_type_list:
                print("\n#######################################\nScalarfield", scalarfield_type, "\n#######################################")
                self.run_for_given_type(visualize=visualize, verbosity=verbosity, scalarfield_type=scalarfield_type)
        else:
            print("\n#######################################\nScalarfield", scalarfield_type, "\n#######################################")
            self.run_for_given_type(visualize=visualize, verbosity=verbosity, scalarfield_type=scalarfield_type)

    def run_for_given_type(self, visualize: bool =True, verbosity: list = ["delays", "ds", "integrals"],
                           scalarfield_type: str = "horizontal_gradient") -> None:
        """
        run test for given scalarfield type
        """

        self.create_scalarfield(scalarfield_type)

        print(f"Calculate numerical solution for {scalarfield_type} scalarfield")
        self.delays_num, self.ds_num = calculate_delays_for_heterogen_sos(
                                    self.sensor_positions, self.xdim, self.ydim, self.zdim, self.x, self.y, self.z, self.spacing_in_mm,
                                    self.time_spacing_in_ms, self.speed_of_sound_in_m_per_s, self.n_sensor_elements, self.global_settings,
                                    self.device_base_position_mm, self.logger, self.torch_device,
                                    get_ds = True
                                    )

        self.delays_num, self.ds_num = self.delays_num.squeeze().cpu().numpy(), self.ds_num.cpu().numpy()

        # Analytical Integration
        self.delays_ana, self.ds_ana = self.analytical_solution(scalarfield_type)
        self.delays_ana, self.ds_ana = self.delays_ana.cpu().numpy(), self.ds_ana.cpu().numpy()

        # Calculate differences of the delays
        self.difference = np.abs(self.delays_num - self.delays_ana)

        # Calculate differences of the distances
        ds_diff_num = np.abs(self.ds_ana - self.ds_num)

        self.tolerances = {"relative": {
                                "constant": 1e-15,
                                "horizontal_gradient": 1e-8,
                                "vertical_gradient": 1e-8,
                                "quadratic": 1e-5
                                },
                            "absolute": {
                                "constant": 1e-9,
                                "horizontal_gradient": 1e-2,
                                "vertical_gradient": 1e-2,
                                "quadratic": 1 if not self.RANDOM_SENSORS else 10
                                }
                            }
        
        print("Delays:")
        # test whether relative differences are in the accecpted range defined above
        torch.testing.assert_close(self.delays_num, self.delays_ana, rtol=self.tolerances["relative"][scalarfield_type], atol=42)
        print("  Relative Difference < ", self.tolerances["relative"][scalarfield_type])

        # Test for absolute differences
        if self.PAI_realistic_units:
            torch.testing.assert_close(self.delays_num, self.delays_ana, rtol=42, atol=self.tolerances["absolute"][scalarfield_type])
            print("  Absolute Difference < ", self.tolerances["absolute"][scalarfield_type])

        if "delays" in verbosity:
            print("Absolute difference of delays")
            print(f"    max={self.difference.max()}\n    mean={self.difference.mean()}\n    min={self.difference.min()}")
            print(" - numerical delays (own interpol.) and analytical delays")
            print("Relative difference of delays (divided by analytical delays)")
            print(f"    max={(self.difference/self.delays_ana).max()}\n    mean={(self.difference/self.delays_ana).mean()}\n    min={(self.difference/self.delays_ana).min()}")


        print("Distances of Sources and Sensors:")
        # test whether the difference is 0
        torch.testing.assert_close(self.ds_num, self.ds_ana, rtol=0, atol=0)
        print("  Absolute Difference is 0")

        if "ds" in verbosity:
            print("Distance differences")
            print(f"    max={ds_diff_num.max()}\n    mean={ds_diff_num.mean()}\n    min={ds_diff_num.min()}")

        if "integrals" in verbosity or visualize:
            # Compute the Integrals
            # avoid division by 0
            self.ds_num[self.ds_num==0] = 42
            self.ds_ana[self.ds_ana==0] = 42
            self.integral_num = self.delays_num[:,:,:]/self.ds_num[:,:,:]
            self.integral_ana = self.delays_ana[:,:,:]/self.ds_ana[:,:,:]
            # Integration differences
            self.int_difference = np.abs(self.integral_num - self.integral_ana)
            if "integrals" in verbosity:
                print("Absolute difference of integrals")
                print(f"    max={self.int_difference.max()}\n    mean={self.int_difference.mean()}\n    min={self.int_difference.min()}")

        if visualize:
            self.visualize_setting()
            self.visualize_delays()
            self.visualize_differences()
            self.visualize_delay_difference_hist()
            self.visualize_integrals()
            self.visualize_integral_difference_hist()
            plt.show()

        return None

    def visualize_setting(self, show: bool = False) -> None:
        """
        visualizes the scalarfield, the sensors, the FOV and the device base position 
        """           
        sens_pix = (self.sensor_positions.cpu().numpy()+self.device_base_position_mm)/self.global_settings[Tags.SPACING_MM]
        device_pix = self.device_base_position_mm/self.global_settings[Tags.SPACING_MM]

        FOV_global_pix_factor = self.spacing_in_mm/self.global_settings[Tags.SPACING_MM]
        xdim_start_glob = self.xdim_start*FOV_global_pix_factor + device_pix[0]
        ydim_start_glob = self.ydim_start*FOV_global_pix_factor + device_pix[2]

        fig, axes = plt.subplots(1, 2, figsize=(8*2,12))
        fig.canvas.manager.set_window_title("Scalarfield and SoS-map")
        im1 = axes[0].imshow(self.scalarfield.T, extent=[0, self.xdim_glob, self.ydim_glob, 0])
        axes[0].set_xlabel("x")
        axes[0].set_ylabel("y")
        axes[0].set_title("Scalarfield")
        plt.colorbar(im1, ax=axes[0])
        # Create a Rectangle patch
        rect = patches.Rectangle((xdim_start_glob,ydim_start_glob),self.xdim*FOV_global_pix_factor, self.ydim*FOV_global_pix_factor,
                                linewidth=1, edgecolor='grey', facecolor='none', linestyle="--", label="FOV")
        # Add the patch to the Axes
        axes[0].add_patch(rect)
        axes[0].scatter(sens_pix[:,0], sens_pix[:,2], color="red", label="sensors")
        axes[0].scatter(device_pix[0], device_pix[2], color="orange", label="device base position", marker="x")
        axes[0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.08), facecolor='whitesmoke')

        im2 = plt.imshow(1/self.scalarfield.T, extent=[0, self.xdim_glob, self.ydim_glob, 0])
        axes[1].set_xlabel("x")
        axes[1].set_ylabel("y")
        axes[1].set_title("SoS-map = 1/Scalarfield")
        plt.colorbar(im2, ax=axes[1])

        if show:
            plt.show()

    def visualize_delays(self, show: bool = False) -> None:
        """
        visualize the numerically and analytically calculated delays for all sensors
        """
        cols = 2
        fig, axes = plt.subplots(self.n_sensor_elements, cols, figsize=(8*cols,6*self.n_sensor_elements))
        fig.canvas.manager.set_window_title("Delays of the FOV points")
        for n in range(self.n_sensor_elements):
            im1 = axes[n,0].imshow(self.delays_num[:,:,n])
            axes[n, 0].set_title(f"Numerical (Sensor {n+1})")
            axes[n, 0].set_ylabel("x")
            axes[n, 0].set_xlabel("y")
            plt.colorbar(im1, ax=axes[n, 0])
            im2 = axes[n,1].imshow(self.delays_ana[:,:,n])
            axes[n, 1].set_title(f"Analytical (Sensor {n+1})")
            axes[n, 1].set_ylabel("x")
            axes[n, 1].set_xlabel("y")
            plt.colorbar(im2, ax=axes[n, 1])
        if show:
            plt.show()

    def visualize_differences(self, show: bool = False) -> None:
        """
        visualize the difference between the numerically and analytically calculated delays
        """
        fig, axes = plt.subplots(1, self.n_sensor_elements, figsize=(8*self.n_sensor_elements,6))
        fig.canvas.manager.set_window_title("Delay Differences")
        for n in range(self.n_sensor_elements):
            dif1 = axes[n].imshow(self.difference[:,:,n], cmap="hot")
            axes[n].set_title(f"Sensor {n+1}")
            axes[n].set_ylabel("x")
            axes[n].set_xlabel("y")
            plt.colorbar(dif1, ax=axes[n])
        if show:
            plt.show()

    def visualize_delay_difference_hist(self, show: bool = False) -> None:
        """
        plot a histogram of the delay differences
        """
        plt.figure("Histogram of Delay Differences")
        plt.hist(self.difference.reshape(-1), bins=20, color="red")
        plt.title(f"max = {self.difference.max()}\nmean={self.difference.mean()}\nmin={self.difference.min()}", fontsize=16)
        if show:
            plt.show()

    def visualize_integrals(self, show: bool = False) -> None:
        """
        plot the integrals, i.e. delays/ds
        """
        cols = 2
        fig, axes = plt.subplots(self.n_sensor_elements, cols, figsize=(6*cols,8*self.n_sensor_elements))
        fig.canvas.manager.set_window_title("Integrals")
        for n in range(self.n_sensor_elements):
            im1 = axes[n,0].imshow(self.integral_num[:,:,n].T)
            axes[n, 0].set_title(f"Numerical (Sensor {n+1})")
            axes[n, 0].set_ylabel("x")
            axes[n, 0].set_xlabel("y")
            plt.colorbar(im1, ax=axes[n, 0])
            im2 = axes[n,1].imshow(self.integral_ana[:,:,n].T)
            axes[n, 1].set_title(f"Analytical (Sensor {n+1})")
            axes[n, 1].set_ylabel("x")
            axes[n, 1].set_xlabel("y")
            plt.colorbar(im2, ax=axes[n, 1])
        if show:
            plt.show()

    def visualize_integral_difference_hist(self, show: bool = False) -> None:      
        """
        plot a histogram of the integral differences
        """
        plt.figure("Integral Differences")
        plt.hist(self.int_difference.reshape(-1), bins=20, color="red")
        plt.title(f"max = {self.int_difference.max()}\nmean={self.int_difference.mean()}\nmin={self.int_difference.min()}", fontsize=16)
        if show:
            plt.show()

if __name__ == "__main__":
    test = TesthDASNumericalIntegration()

    # Normal usage
    test.setUp()
    test.test()

    # More verbose prints
    #test.setUp()
    #test.test(verbosity=["delays", "ds", "integrals"])

    # Visualize scalarfield 
    #test.setUp()
    #test.create_scalarfield("quadratic")
    #test.visualize_setting(show=True)

    # Print and plot all quantities for given scalarfield
    #test.setUp()
    #test.test(visualize=True, verbosity=["delays", "ds", "integrals"], scalarfield_type="horizontal_gradient")

