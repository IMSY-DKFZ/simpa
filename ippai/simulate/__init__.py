import numpy as np

class Tags:

    # General settings
    SIMULATION_PATH = "simulation_path"
    VOLUME_NAME = "volume_name"
    WAVELENGTHS = "wavelengths"
    WAVELENGTH = "wavelength"
    RANDOM_SEED = "random_seed"
    TISSUE_PROPERTIES_OUPUT_NAME = "properties"

    # Optical model settings
    RUN_OPTICAL_MODEL = 'run_optical_forward_model'
    OPTICAL_MODEL_OUTPUT_NAME = "optical_forward_model_output"
    OPTICAL_MODEL_BINARY_PATH = "optical_model_binary_path"
    OPTICAL_MODEL_NUMBER_PHOTONS = "optical_model_number_of_photons"
    OPTICAL_MODEL_PROBE_XML_FILE = "optical_model_probe_xml_file"

    # Supported optical models
    OPTICAL_MODEL = "optical_model"
    MODEL_MCXYZ = "mcxyz"
    MODEL_MCX = "mcx"

    # Acoustic model settings
    RUN_ACOUSTIC_MODEL = 'run_acoustic_forward_model'
    ACOUSTIC_MODEL_OUTPUT_NAME = "acoustic_forward_model_output"

    # physical property volume types
    PROPERTY_ABSORPTION_PER_CM = 'mua'
    PROPERTY_SCATTERING_PER_CM = 'mus'
    PROPERTY_ANISOTROPY = 'g'
    PROPERTY_OXYGENATION = 'sO2'
    PROPERTY_SEGMENTATION = 'segmentation'

    # Air layer
    AIR_LAYER_HEIGHT_MM = "air_layer_height"

    # Gel Pad Layer
    GELPAD_LAYER_HEIGHT_MM = "gelpad_layer_height_mm"

    # Volume geometry settings
    SPACING_MM = "voxel_spacing_mm"
    DIM_VOLUME_X_MM = "volume_x_dim_mm"
    DIM_VOLUME_Y_MM = "volume_y_dim_mm"
    DIM_VOLUME_Z_MM = "volume_z_dim_mm"

    # Tissue Properties Settings
    KEY_B = "B"
    KEY_B_MIN = "B_min"
    KEY_B_MAX = "B_max"
    KEY_W = "W"
    KEY_W_MAX = "w_max"
    KEY_W_MIN = "w_min"
    KEY_F = "F"
    KEY_F_MAX = "f_max"
    KEY_F_MIN = "f_min"
    KEY_M = "M"
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

    STRUCTURE_DEPTH_MIN_MM = "structure_depth_min_mm"
    STRUCTURE_DEPTH_MAX_MM = "structure_depth_max_mm"

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
    STRUCTURE_TUBE_START_X_MIN_MM = "structure_tube_start_x_min_mm"
    STRUCTURE_TUBE_START_X_MAX_MM = "structure_tube_start_x_max_mm"


class SegmentationClasses:
    """
    The segmentation classes define which "tissue types" are modelled in the simulation volumes.
    """
    AIR = 0
    MUSCLE = 1
    BONE = 2
    BLOOD = 3
    EPIDERMIS = 4
    DERMIS = 5
    FAT = 6
    ULTRASOUND_GEL_PAD = 7


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
    STANDARD_ANISOTROPY = 0.9

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
