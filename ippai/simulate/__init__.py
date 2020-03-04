import numpy as np


class Tags:

    """
    General settings
    """
    SIMULATION_PATH = "simulation_path"
    VOLUME_NAME = "volume_name"
    WAVELENGTHS = "wavelengths"
    WAVELENGTH = "wavelength"
    RANDOM_SEED = "random_seed"
    TISSUE_PROPERTIES_OUPUT_NAME = "properties"
    SIMULATION_EXTRACT_FIELD_OF_VIEW = "extract_field_of_view"
    GPU = "gpu"
    MEDIUM_TEMPERATURE_CELCIUS = "medium_temperature"

    """
    Optical model settings
    """
    RUN_OPTICAL_MODEL = 'run_optical_forward_model'
    OPTICAL_MODEL_OUTPUT_NAME = "optical_forward_model_output"
    OPTICAL_MODEL_BINARY_PATH = "optical_model_binary_path"
    OPTICAL_MODEL_NUMBER_PHOTONS = "optical_model_number_of_photons"
    OPTICAL_MODEL_PROBE_XML_FILE = "optical_model_probe_xml_file"  # TODO rename PROBE -> ILLUMINATION
    LASER_PULSE_ENERGY_IN_MILLIJOULE = "laser_pulse_energy_in_millijoule"

    ILLUMINATION_TYPE = "optical_model_illumination_type"

    # Illumination parameters
    ILLUMINATION_POSITION = "illumination_position"
    ILLUMINATION_DIRECTION = "illumination_direction"
    ILLUMINATION_PARAM1 = "illumination_param1"
    ILLUMINATION_PARAM2 = "illumination_param2"
    TIME_STEP = "time_step"

    # Supported illumination types - implemented in mcx
    ILLUMINATION_TYPE_PENCIL = "pencil"
    ILLUMINATION_TYPE_DISK = "disk"
    ILLUMINATION_TYPE_SLIT = "slit"
    ILLUMINATION_TYPE_GAUSSIAN = "gaussian"
    ILLUMINATION_TYPE_PATTERN = "pattern"
    ILLUMINATION_TYPE_PATTERN_3D = "pattern3d"
    ILLUMINATION_TYPE_FOURIER = "fourier"
    ILLUMINATION_TYPE_FOURIER_X = "fourierx"
    ILLUMINATION_TYPE_FOURIER_X_2D = "fourierx2d"

    ILLUMINATION_TYPE_DKFZ_PAUS = "pasetup"  # TODO more explanatory rename of pasetup
    ILLUMINATION_TYPE_MSOT_ACUITY_ECHO = "msot_acuity_echo"

    # Supported optical models
    OPTICAL_MODEL = "optical_model"
    MODEL_MCXYZ = "mcxyz"
    MODEL_MCX = "mcx"

    # Supported acoustic models
    ACOUSTIC_MODEL = "acoustic_model"
    MODEL_K_WAVE = "kwave"
    ACOUSTIC_MODEL_SCRIPT = "acoustic_model_script"
    ACOUSTIC_MODEL_SCRIPT_LOCATION = "acoustic_model_script_location"

    """
    Acoustic model settings
    """

    RUN_ACOUSTIC_MODEL = 'run_acoustic_forward_model'
    ACOUSTIC_MODEL_BINARY_PATH = "acoustic_model_binary_path"
    ACOUSTIC_MODEL_OUTPUT_NAME = "acoustic_forward_model_output"
    ACOUSTIC_SIMULATION_PATH = "acoustic_simulation_path"
    RECORDMOVIE = "record_movie"
    MOVIENAME = "movie_name"
    ACOUSTIC_PLOT_SCALE = "acoustic_plot_scale"
    ACOUSTIC_LOG_SCALE = "acoustic_log_scale"
    TIME_SERIES_DATA = "time_series_data"
    TIME_SERIES_DATA_NOISE = "time_series_data_noise"

    # Reconstruction settings
    PERFORM_IMAGE_RECONSTRUCTION = "perform_image_reconstruction"
    RECONSTRUCTION_OUTPUT_NAME = "reconstruction_result"
    RECONSTRUCTION_ALGORITHM = "reconstruction_algorithm"
    RECONSTRUCTION_ALGORITHM_DAS = "DAS"
    RECONSTRUCTION_ALGORITHM_DMAS = "DMAS"
    RECONSTRUCTION_ALGORITHM_SDMAS = "sDMAS"
    RECONSTRUCTION_MITK_BINARY_PATH = "reconstruction_mitk_binary_path"
    RECONSTRUCTION_MITK_SETTINGS_XML = "reconstruction_mitk_settings_xml"
    RECONSTRUCTION_BMODE_METHOD = "reconstruction_bmode_method"
    RECONSTRUCTION_BMODE_METHOD_ABS = "Abs"
    RECONSTRUCTION_BMODE_METHOD_HILBERT_TRANSFORM = "EnvelopeDetection"
    RECONSTRUCTED_DATA = "reconstructed_data"
    RECONSTRUCTED_DATA_NOISE = "reconstructed_data_noise"

    """
    Upsampling settings
    """

    CROP_IMAGE = "crop_image"
    CENTER_CROP = "center_crop"
    CROP_POWER_OF_TWO = "crop_power_of_two"
    PERFORM_UPSAMPLING = "sample"
    UPSAMPLING_METHOD = "upsampling_method"
    UPSAMPLING_METHOD_DEEP_LEARNING = "deeplearning"
    UPSAMPLING_METHOD_NEAREST_NEIGHBOUR = "nearestneighbour"
    UPSAMPLING_METHOD_BILINEAR = "bilinear"
    UPSAMPLING_METHOD_LANCZOS2 = "lanczos2"
    UPSAMPLING_METHOD_LANCZOS3 = "lanczos3"
    UPSAMPLING_SCRIPT = "upsampling_script"
    UPSAMPLING_SCRIPT_LOCATION = "upsampling_script_location"
    UPSCALE_FACTOR = "upscale_factor"
    DL_MODEL_PATH = "dl_model_path"

    # physical property volume types
    PROPERTY_ABSORPTION_PER_CM = 'mua'
    PROPERTY_SCATTERING_PER_CM = 'mus'
    PROPERTY_ANISOTROPY = 'g'
    PROPERTY_OXYGENATION = 'oxy'
    PROPERTY_SEGMENTATION = 'seg'
    PROPERTY_GRUNEISEN_PARAMETER = 'gamma'
    PROPERTY_SPEED_OF_SOUND = "sos"

    # Air layer
    AIR_LAYER = "airlayer"
    AIR_LAYER_HEIGHT_MM = "air_layer_height"

    # Gel Pad Layer
    GELPAD_LAYER = "gelpad"
    GELPAD_LAYER_HEIGHT_MM = "gelpad_layer_height_mm"

    # Volume geometry settings
    SPACING_MM = "voxel_spacing_mm"
    DIM_VOLUME_X_MM = "volume_x_dim_mm"
    DIM_VOLUME_Y_MM = "volume_y_dim_mm"
    DIM_VOLUME_Z_MM = "volume_z_dim_mm"

    # 2D Acoustic Medium Properties
    MEDIUM_SOUND_SPEED_HOMOGENEOUS = "medium_sound_speed_homogeneous"
    MEDIUM_SOUND_SPEED = "medium_sound_speed"
    MEDIUM_DENSITY_HOMOGENEOUS = "medium_density_homogeneous"
    MEDIUM_DENSITY = "medium_density"
    MEDIUM_ALPHA_COEFF_HOMOGENEOUS = "medium_alpha_coeff_homogeneous"
    MEDIUM_ALPHA_COEFF = "medium_alpha_coeff"
    MEDIUM_ALPHA_POWER = "medium_alpha_power"
    MEDIUM_NONLINEARITY = "medium_nonlinearity"

    # PML parameters

    PMLSize = "pml_size"
    PMLAlpha = "pml_alpha"
    PMLInside = "pml_inside"
    PlotPML = "plot_pml"

    # Acoustic Sensor Properties
    SENSOR_MASK = "sensor_mask"
    SENSOR_RECORD = "sensor_record"
    SENSOR_CENTER_FREQUENCY_HZ = "sensor_center_frequency"
    SENSOR_BANDWIDTH_PERCENT = "sensor_bandwidth"
    SENSOR_DIRECTIVITY_HOMOGENEOUS = "sensor_directivity_homogeneous"
    SENSOR_DIRECTIVITY_ANGLE = "sensor_directivity_angle"
    SENSOR_DIRECTIVITY_SIZE_M = "sensor_directivity_size"
    SENSOR_DIRECTIVITY_PATTERN = "sensor_directivity_pattern"
    SENSOR_ELEMENT_PITCH_MM = "sensor_element_pitch"
    SENSOR_SAMPLING_RATE_MHZ = "sensor_sampling_rate_mhz"
    SENSOR_NUM_ELEMENTS = "sensor_num_elements"
    SENSOR_ARC = "sensor_arc"
    SENSOR_RADIUS_MM = "sensor_radius_mm"

    # Noise properties
    APPLY_NOISE_MODEL = "apply_noise_model"
    NOISE_MODEL = "noise_model"
    NOISE_MODEL_GAUSSIAN = "noise_model_gaussian"
    NOISE_MEAN = "noise_mean"
    NOISE_STD = "noise_std"
    NOISE_MODEL_OUTPUT_NAME = "noise_model_output"
    NOISE_MODEL_PATH = "noise_model_path"

    # Constant Tissue Properties
    KEY_CONSTANT_PROPERTIES = "constant_properties"
    KEY_MUA = "mua"
    KEY_MUS = "mus"
    KEY_G = "g"

    # Tissue Properties Settings
    KEY_BLOOD = "B"
    KEY_B_MIN = "B_min"
    KEY_B_MAX = "B_max"
    KEY_WATER = "W"
    KEY_W_MAX = "w_max"
    KEY_W_MIN = "w_min"
    KEY_FAT = "F"
    KEY_F_MAX = "f_max"
    KEY_F_MIN = "f_min"
    KEY_MELANIN = "M"
    KEY_M_MAX = "m_max"
    KEY_M_MIN = "m_min"
    KEY_OXY = "OXY"
    KEY_OXY_MAX = "oxy_max"
    KEY_OXY_MIN = "oxy_min"
    KEY_MUSP500 = "musp500"
    KEY_F_RAY = "f_ray"
    KEY_B_MIE = "b_mie"
    KEY_ANISOTROPY = "anisotropy"

    # Structures
    STRUCTURES = "structures"
    CHILD_STRUCTURES = "child_structures"
    STRUCTURE_TYPE = "structure_type"
    STRUCTURE_SEGMENTATION_TYPE = "structure_segmentation_type"
    STRUCTURE_TISSUE_PROPERTIES = "structure_tissue_properties"

    STRUCTURE_CENTER_DEPTH_MIN_MM = "structure_depth_min_mm"
    STRUCTURE_CENTER_DEPTH_MAX_MM = "structure_depth_max_mm"

    STRUCTURE_USE_DISTORTION = "structure_distortion_multiplicative"
    STRUCTURE_DISTORTED_PARAM_LIST = "structure_distorted_param_list"
    STRUCTURE_DISTORTION_FREQUENCY_PER_MM = "structure_distortion_wavelength_mm"

    STRUCTURE_BACKGROUND = "structure_background"

    STRUCTURE_LAYER = "structure_layer"
    STRUCTURE_THICKNESS_MIN_MM = "structure_thickness_min_mm"
    STRUCTURE_THICKNESS_MAX_MM = "structure_thickness_max_mm"

    STRUCTURE_TUBE = "structure_tube"
    STRUCTURE_RADIUS_MIN_MM = "structure_radius_min_mm"
    STRUCTURE_RADIUS_MAX_MM = "structure_radius_max_mm"
    STRUCTURE_FORCE_ORTHAGONAL_TO_PLANE = "structure_force_orthagonal_to_plane"
    STRUCTURE_TUBE_CENTER_X_MIN_MM = "structure_tube_start_x_min_mm"
    STRUCTURE_TUBE_CENTER_X_MAX_MM = "structure_tube_start_x_max_mm"

    STRUCTURE_ELLIPSE = "structure_ellipse"
    STRUCTURE_MIN_ECCENTRICITY = "structure_eccentricity_min"
    STRUCTURE_MAX_ECCENTRICITY = "structure_eccentricity_max"

    UNITS_ARBITRARY = "arbitrary_unity"
    UNITS_PRESSURE = "newton_per_meters_squared"


class SegmentationClasses:
    """
    The segmentation classes define which "tissue types" are modelled in the simulation volumes.
    """
    GENERIC = -1
    AIR = 0
    MUSCLE = 1
    BONE = 2
    BLOOD = 3
    EPIDERMIS = 4
    DERMIS = 5
    FAT = 6
    ULTRASOUND_GEL_PAD = 7
    WATER = 8


class StandardProperties:
    """
    This class contains a listing of default parameters and options with the aim to reduce the amount
    """
    AIR_MUA = 1e-10
    AIR_MUS = 1e-10
    AIR_G = 1
    GELPAD_MUA = 1e-10
    GELPAD_MUS = 1e-10
    GELPAD_G = 1

    # @book{marx2013rosen,
    #   title={Rosen's Emergency Medicine-Concepts and Clinical Practice E-Book},
    #   author={Marx, John and Walls, Ron and Hockberger, Robert},
    #   year={2013},
    #   publisher={Elsevier Health Sciences}
    # }
    BODY_TEMPERATURE_CELCIUS = 37.0

    # @techreport{hasgall2018database,
    #     title = {IT’IS Database for thermal and electromagnetic parameters of biological tissues.
    #     Version 4.0, May 15, 2018. doi: 10.13099},
    #     author = {Hasgall, PA and Di Gennaro, F and Baumgartner, C and Neufeld, E and Lloyd, B and Gosselin,
    #               MC and Payne, D and Klingenb{\"o}ck, A and Kuster, N},
    #     year = {2018},
    #     institution = {VIP21000 - 04 - 0.Onl: www.itis.ethz.ch / database}
    # }

    DENSITY_GENERIC = 1.000 # kg/l
    DENSITY_AIR = 0.001
    DENSITY_MUSCLE = 1.090
    DENSITY_BONE = 1.908
    DENSITY_BLOOD = 1.050
    DENSITY_SKIN = 1.109
    DENSITY_FAT = 0.911
    DENSITY_GEL_PAD = 0.890
    DENSITY_WATER = 1.000

    SPEED_OF_SOUND_GENERIC = 1540   # m/s
    SPEED_OF_SOUND_AIR = 343
    SPEED_OF_SOUND_MUSCLE = 1588.4
    SPEED_OF_SOUND_BONE = 3500      # Cortical bone
    SPEED_OF_SOUND_BLOOD = 1578.2
    SPEED_OF_SOUND_SKIN = 1624.0
    SPEED_OF_SOUND_FAT = 1440.2
    SPEED_OF_SOUND_GEL_PAD = 1583
    SPEED_OF_SOUND_WATER = 1482.3

    ALPHA_COEFF_GENERIC = 0.02   # dB/m/MHz
    ALPHA_COEFF_AIR = 1.3579e+05
    ALPHA_COEFF_MUSCLE = 14.1579
    ALPHA_COEFF_BONE = 29.7723    # Cortical bone
    ALPHA_COEFF_BLOOD = 2.8174
    ALPHA_COEFF_SKIN = 11.5470
    ALPHA_COEFF_FAT = 9.1534
    ALPHA_COEFF_GEL_PAD = 0.5
    ALPHA_COEFF_WATER = 0.0138


class OpticalTissueProperties:
    """
    This class contains a listing of optical tissue parameters as reported in literature.
    The listing is not the result of a meta analysis, but rather uses the best fitting paper at
    the time pf implementation.
    Each of the fields is annotated with a literature reference or a descriptions of how the particular
    values were derived for tissue modelling.
    """

    # Background oxygenation assumed arbitrarily, to cover a large range of oxygenation values
    BACKGROUND_OXYGENATION = 0.8
    BACKGROUND_OXYGENATION_VARIATION = 0.1

    # Venous blood parameters taken from the referenced literature. <60% SvO2 were reported as critical. Normal values
    # are reported as 70%.
    # @article{molnar2018monitoring,
    #   title={Monitoring of tissue oxygenation: an everyday clinical challenge},
    #   author={Molnar, Zsolt and Nemeth, Marton},
    #   journal={Frontiers in medicine},
    #   volume={4},
    #   pages={247},
    #   year={2018},
    #   publisher={Frontiers}
    # }
    VENOUS_OXYGENATION = 0.7
    VENOUS_OXYGENATION_VARIATION = 0.1

    # Arterial blood parameters taken from the referenced literature.
    # @article{merrick1976continuous,
    #   title={Continuous, non-invasive measurements of arterial blood oxygen levels},
    #   author={Merrick, Edwin B and Hayes, Thomas J},
    #   journal={Hewlett-packard J},
    #   volume={28},
    #   number={2},
    #   pages={2--9},
    #   year={1976}
    # }
    ARTERIAL_OXYGENATION = 0.95
    ARTERIAL_OXYGENATION_VARIATION = 0.05

    # Tissue Property derived from the meta analysis by Steve Jacques in 2013:
    # @article{jacques2013optical,
    #   title={Optical properties of biological tissues: a review},
    #   author={Jacques, Steven L},
    #   journal={Physics in Medicine \& Biology},
    #   volume={58},
    #   number={11},
    #   pages={R37},
    #   year={2013},
    #   publisher={IOP Publishing}
    # }
    MUSP500_BACKGROUND_TISSUE = 19.1  # Table 2: Average over all other soft tissue
    FRAY_BACKGROUND_TISSUE = 0.153  # Table 2: Average over all other soft tissue
    BMIE_BACKGROUND_TISSUE = 1.091  # Table 2: Average over all other soft tissue

    MUSP500_EPIDERMIS = 66.7  # Table 1; Salomatina et al 2006; One value for epidermis
    FRAY_EPIDERMIS = 0.29  # Table 1; Salomatina et al 2006; One value for epidermis
    BMIE_EPIDERMIS = 0.689  # Table 1; Salomatina et al 2006; One value for epidermis
    MUSP500_DERMIS = 43.6  # Table 1; Salomatina et al 2006; One value for dermis
    FRAY_DERMIS = 0.41  # Table 1; Salomatina et al 2006; One value for dermis
    BMIE_DERMIS = 0.69  # Table 1; Salomatina et al 2006; One value for dermis
    MUSP500_FAT = 19.3  # Table 2 average fatty tissue
    FRAY_FAT = 0.174  # Table 2 average fatty tissue
    BMIE_FAT = 0.447  # Table 2 average fatty tissue
    MUSP500_BLOOD = 22  # Table 1 Alexandrakis et al 2005
    FRAY_BLOOD = 0.66  # Table 1 Alexandrakis et al 2005
    BMIE_BLOOD = 0  # Table 1 Alexandrakis et al 2005
    MUSP500_BONE = 15.3  # Table 2 Mean for bone
    FRAY_BONE = 0.022  # Table 2 Mean for bone
    BMIE_BONE = 0.326  # Table 2 Mean for bone
    STANDARD_ANISOTROPY = 0.9  # Average anisotropy of measured values presented in paper

    # Water content of bone:
    # @article{timmins1977bone,
    #   title={Bone water},
    #   author={Timmins, PA and Wall, JC},
    #   journal={Calcified tissue research},
    #   volume={23},
    #   number={1},
    #   pages={1--5},
    #   year={1977},
    #   publisher={Springer}
    # }
    WATER_VOLUME_FRACTION_BONE_MEAN = 0.19
    WATER_VOLUME_FRACTION_BONE_STD = 0.01

    # Adult body composition derived values
    # @article{forbes1953composition,
    #   title={The composition of the adult human body as determined by chemical analysis},
    #   author={Forbes, RM and Cooper, AR and Mitchell, HH and others},
    #   journal={J Biol Chem},
    #   volume={203},
    #   number={1},
    #   pages={359--366},
    #   year={1953}
    # }
    WATER_VOLUME_FRACTION_SKIN = 0.58
    WATER_VOLUME_FRACTION_HUMAN_BODY = 0.68

    # Muscle tissue blood volume fraction:
    # @article{vankana1998mechanical,
    #   title={Mechanical blood-tissue interaction in contracting muscles: a model study},
    #   author={Vankana, WJ and Huyghe, Jacques M and van Donkelaar, Corrinus C and Drost, Maarten R and Janssen,
    # Jan D and Huson, A},
    #   journal={Journal of Biomechanics},
    #   volume={31},
    #   number={5},
    #   pages={401--409},
    #   year={1998},
    #   publisher={Elsevier}
    # }
    BLOOD_VOLUME_FRACTION_MUSCLE_TISSUE = 0.01  # Value of arterial bvf at t0 in fig 3.

    # Mean and spread calculated from all ethnicities from figure 2C, averaged over both
    # photoexposed and photoprotected samples.
    # @article{alaluf2002ethnic,
    #   title={Ethnic variation in melanin content and composition in photoexposed and photoprotected human skin},
    #   author={Alaluf, Simon and Atkins, Derek and Barrett, Karen and Blount, Margaret and Carter,
    # Nik and Heath, Alan},
    #   journal={Pigment Cell Research},
    #   volume={15},
    #   number={2},
    #   pages={112--118},
    #   year={2002},
    #   publisher={Wiley Online Library}
    # }
    MELANIN_VOLUME_FRACTION_MEAN = 0.022
    MELANIN_VOLUME_FRACTION_STD = 0.01


class MorphologicalTissueProperties:
    """
    This class contains a listing of morphological tissue parameters as reported in literature.
    The listing is not the result of a meta analysis, but rather uses the best fitting paper at
    the time pf implementation.
    Each of the fields is annotated with a literature reference or a descriptions of how the particular
    values were derived for tissue modelling.
    """
    # Radial and ulnar artery diameter reference:
    # @article{ashraf2010size,
    #   title={Size of radial and ulnar artery in local population},
    #   author={Ashraf, Tariq and Panhwar, Ziauddin and Habib, Sultana and Memon, Muhammad Anis and Shamsi,
    # Fahad and Arif, Javed},
    #   journal={JPMA-Journal of the Pakistan Medical Association},
    #   volume={60},
    #   number={10},
    #   pages={817},
    #   year={2010}
    # }
    RADIAL_ARTERY_DIAMETER_MEAN_MM = 2.25
    RADIAL_ARTERY_DIAMETER_STD_MM = 0.4
    ULNAR_ARTERY_DIAMETER_MEAN_MM = 2.35
    ULNAR_ARTERY_DIAMETER_STD_MM = 0.35

    # Accompanying veins diameter reference. They specifically only mention the ulnar accompanying vein properties.
    # We assume a non-significant similarity for the radial accompanying vein.
    # @incollection{yang_ulnar_2018,
    #   title = {Ulnar {Artery} to {Superficial} {Arch} {Bypass} with a {Vein} {Graft}},
    #   booktitle = {Operative {Techniques}: {Hand} and {Wrist} {Surgery}},
    #   author = {Yang, Guang and Chung, Kevin C.},
    #   year = {2018},
    #   doi = {10.1016/B978-0-323-40191-3.00081-0},
    #   pages = {732--737},
    # }
    RADIAL_VEIN_DIAMETER_MEAN_MM = 1
    RADIAL_VEIN_DIAMETER_STD_MM = 0.2
    ULNAR_VEIN_DIAMETER_MEAN_MM = 1
    ULNAR_VEIN_DIAMETER_STD_MM = 0.2

    # Median artery diameter reference (at the P2 point):
    # @article{hubmer2004posterior,
    #   title={The posterior interosseous artery in the distal part of the forearm. Is the term ‘recurrent branch of
    # the anterior interosseous artery’justified?},
    #   author={Hubmer, Martin G and Fasching, Thomas and Haas, Franz and Koch, Horst and Schwarzl, Franz and Weiglein,
    # Andreas and Scharnagl, Erwin},
    #   journal={British journal of plastic surgery},
    #   volume={57},
    #   number={7},
    #   pages={638--644},
    #   year={2004},
    #   publisher={Elsevier}
    # }
    MEDIAN_ARTERY_DIAMETER_MEAN_MM = 0.6
    MEDIAN_ARTERY_DIAMETER_STD_MM = 0.25

    # TODO CITE
    # Assumption: about half the size of the radial and ulnar accompanying veins due to size of the respective
    # artery in comparison to the radial and ulna artery
    MEDIAN_VEIN_DIAMETER_MEAN_MM = 0.5
    MEDIAN_VEIN_DIAMETER_STD_MM = 0.1

    # Thickness of the dermis and epidermis approximated with values for the hand. Averaged for
    # @article{oltulu2018measurement,
    #   title={Measurement of epidermis, dermis, and total skin thicknesses from six different body regions
    # with a new ethical histometric technique},
    #   author={Oltulu, Pembe and Ince, Bilsev and Kokbudak, Naile and Findik, Sidika and Kilinc, Fahriye and others},
    #   journal={Turkish Journal of Plastic Surgery},
    #   volume={26},
    #   number={2},
    #   pages={56},
    #   year={2018},
    #   publisher={Medknow Publications}
    # }
    DERMIS_THICKNESS_MEAN_MM = 2.3
    DERMIS_THICKNESS_STD_MM = 1.2
    EPIDERMIS_THICKNESS_MEAN_MM = 0.22
    EPIDERMIS_THICKNESS_STD_MM = 0.1

    # Distance of radius and ulnar at resting position, when bones are not crossed
    # @article{christensen1968study,
    #   title={A study of the interosseous distance between the radius and ulna during rotation of the forearm},
    #   author={Christensen, John B and Adams, John P and Cho, KO and Miller, Lawrence},
    #   journal={The Anatomical Record},
    #   volume={160},
    #   number={2},
    #   pages={261--271},
    #   year={1968},
    #   publisher={Wiley Online Library}
    # }
    RADIUS_ULNA_BONE_SEPARATION_MEAN_MM = 32
    RADIUS_ULNA_BONE_POSITION_STD_MM = 2

    # Subcutaneous veins depth measurements are extrapolated from graphs in table 3.
    # The diameter measurement are supposed to resemble the approximate range from figure 15.
    # @article{goh2017subcutaneous,
    #   title={Subcutaneous veins depth measurement using diffuse reflectance images},
    #   author={Goh, CM and Subramaniam, R and Saad, NM and Ali, SA and Meriaudeau, F},
    #   journal={Optics express},
    #   volume={25},
    #   number={21},
    #   pages={25741--25759},
    #   year={2017},
    #   publisher={Optical Society of America}
    # }
    SUBCUTANEOUS_VEIN_DEPTH_MEAN_MM = 1.5
    SUBCUTANEOUS_VEIN_DEPTH_STD_MM = 0.7
    SUBCUTANEOUS_VEIN_DIAMETER_MEAN_MM = 0.8
    SUBCUTANEOUS_VEIN_DIAMETER_STD_MM = 0.6

    # The following properties were experimentally determined based on data sets provided by Janek Gröhl
    # (Photoacoustic forearm images) and André Klein (Forearm CT images from full body CTs)
    RADIAL_ARTERY_DEPTH_MEAN_MM = 9
    RADIAL_ARTERY_DEPTH_STD_MM = 1
    ULNAR_ARTERY_DEPTH_MEAN_MM = 8
    ULNAR_ARTERY_DEPTH_STD_MM = 1
    DISTANCE_RADIAL_AND_ULNA_ARTERY_MEAN_MM = 30
    DISTANCE_RADIAL_AND_ULNA_ARTERY_STD_MM = 5
    RADIUS_BONE_DIAMETER_MEAN_MM = 20
    RADIUS_BONE_DIAMETER_STD_MM = 2
    ULNA_BONE_DIAMETER_MEAN_MM = 15
    ULNA_BONE_DIAMETER_STD_MM = 2
    MEDIAN_ARTERY_DEPTH_MEAN_MM = 19
    MEDIAN_ARTERY_DEPTH_STD_MM = 1
    ACCOMPANYING_VEIN_MEDIAN_DISTANCE_MEAN_MM = 1
    ACCOMPANYING_VEIN_MEDIAN_DISTANCE_STD_MM = 0.2
    ACCOMPANYING_VEIN_DISTANCE_MEAN_MM = 2.5
    ACCOMPANYING_VEIN_DISTANCE_STD_MM = 0.4
    ACCOMPANYING_VEIN_DEPTH_STD_MM = 1.5
    RADIUS_BONE_DEPTH_MEAN_MM = 22
    RADIUS_BONE_DEPTH_STD_MM = 2
    ULNA_BONE_DEPTH_MEAN_MM = 22
    ULNA_BONE_DEPTH_STD_MM = 2

    # Arbitrary position constants based on the respective coordinate systems
    RADIAL_ARTERY_X_POSITION_MEAN_MM = 2.5
    ULNAR_ARTERY_X_POSITION_MEAN_MM = RADIAL_ARTERY_X_POSITION_MEAN_MM + DISTANCE_RADIAL_AND_ULNA_ARTERY_MEAN_MM
    MEDIAN_ARTERY_X_POSITION_MEAN_MM = RADIAL_ARTERY_X_POSITION_MEAN_MM + DISTANCE_RADIAL_AND_ULNA_ARTERY_MEAN_MM / 2
    ARTERY_X_POSITION_UNCERTAINTY_MM = DISTANCE_RADIAL_AND_ULNA_ARTERY_STD_MM / np.sqrt(2)
