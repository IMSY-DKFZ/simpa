# The MIT License (MIT)
#
# Copyright (c) 2018 Computer Assisted Medical Interventions Group, DKFZ
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


class StandardProperties:
    """
    This class contains a listing of default parameters and options with the aim to reduce the amount
    """
    AIR_MUA = 1e-10
    AIR_MUS = 1e-10
    AIR_G = 1.0
    GELPAD_MUA = 1e-10
    GELPAD_MUS = 1e-10
    GELPAD_G = 1.0

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

    SPEED_OF_SOUND_GENERIC = 1540.0   # m/s
    SPEED_OF_SOUND_AIR = 343.0
    SPEED_OF_SOUND_MUSCLE = 1588.4
    SPEED_OF_SOUND_BONE = 3500.0      # Cortical bone
    SPEED_OF_SOUND_BLOOD = 1578.2
    SPEED_OF_SOUND_SKIN = 1624.0
    SPEED_OF_SOUND_FAT = 1440.2
    SPEED_OF_SOUND_GEL_PAD = 1583.0
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
    BACKGROUND_OXYGENATION = 0.5
    BACKGROUND_OXYGENATION_VARIATION = 0.5

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
    MUSP500_BLOOD = 22.0  # Table 1 Alexandrakis et al 2005
    FRAY_BLOOD = 0.66  # Table 1 Alexandrakis et al 2005
    BMIE_BLOOD = 0.0  # Table 1 Alexandrakis et al 2005
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
    # RADIAL_ARTERY_DIAMETER_MEAN_MM = 2.25
    # RADIAL_ARTERY_DIAMETER_STD_MM = 0.4
    # ULNAR_ARTERY_DIAMETER_MEAN_MM = 2.35
    # ULNAR_ARTERY_DIAMETER_STD_MM = 0.35

    # Kris approximation
    RADIAL_ARTERY_DIAMETER_MEAN_MM = 2.7
    RADIAL_ARTERY_DIAMETER_STD_MM = 0.4
    ULNAR_ARTERY_DIAMETER_MEAN_MM = 3.0
    ULNAR_ARTERY_DIAMETER_STD_MM = 0.4

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
    RADIAL_VEIN_DIAMETER_MEAN_MM = 1.0
    RADIAL_VEIN_DIAMETER_STD_MM = 0.2
    ULNAR_VEIN_DIAMETER_MEAN_MM = 1.0
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
    RADIUS_ULNA_BONE_SEPARATION_MEAN_MM = 32.0
    RADIUS_ULNA_BONE_POSITION_STD_MM = 2.0

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
    # RADIAL_ARTERY_DEPTH_MEAN_MM = 9
    # RADIAL_ARTERY_DEPTH_STD_MM = 1
    # ULNAR_ARTERY_DEPTH_MEAN_MM = 8
    # ULNAR_ARTERY_DEPTH_STD_MM = 1
    # Kris approximation for realistic forearm images:
    RADIAL_ARTERY_DEPTH_MEAN_MM = 6.0
    RADIAL_ARTERY_DEPTH_STD_MM = 0.5
    ULNAR_ARTERY_DEPTH_MEAN_MM = 6.0
    ULNAR_ARTERY_DEPTH_STD_MM = 0.5

    # DISTANCE_RADIAL_AND_ULNA_ARTERY_MEAN_MM = 30
    # DISTANCE_RADIAL_AND_ULNA_ARTERY_STD_MM = 5
    # Kris approx
    DISTANCE_RADIAL_AND_ULNA_ARTERY_MEAN_MM = 15.0
    DISTANCE_RADIAL_AND_ULNA_ARTERY_STD_MM = 5.0
    RADIUS_BONE_DIAMETER_MEAN_MM = 20.0
    RADIUS_BONE_DIAMETER_STD_MM = 2.0
    ULNA_BONE_DIAMETER_MEAN_MM = 15.0
    ULNA_BONE_DIAMETER_STD_MM = 2.0
    MEDIAN_ARTERY_DEPTH_MEAN_MM = 19.0
    MEDIAN_ARTERY_DEPTH_STD_MM = 1.0
    ACCOMPANYING_VEIN_MEDIAN_DISTANCE_MEAN_MM = 1.0
    ACCOMPANYING_VEIN_MEDIAN_DISTANCE_STD_MM = 0.2
    ACCOMPANYING_VEIN_DISTANCE_MEAN_MM = 2.5
    ACCOMPANYING_VEIN_DISTANCE_STD_MM = 0.4
    ACCOMPANYING_VEIN_DEPTH_STD_MM = 1.5
    RADIUS_BONE_DEPTH_MEAN_MM = 22.0
    RADIUS_BONE_DEPTH_STD_MM = 2.0
    ULNA_BONE_DEPTH_MEAN_MM = 22.0
    ULNA_BONE_DEPTH_STD_MM = 2.0

    # Arbitrary position constants based on the respective coordinate systems
    RADIAL_ARTERY_X_POSITION_MEAN_MM = 7.5
    ULNAR_ARTERY_X_POSITION_MEAN_MM = RADIAL_ARTERY_X_POSITION_MEAN_MM + DISTANCE_RADIAL_AND_ULNA_ARTERY_MEAN_MM
    MEDIAN_ARTERY_X_POSITION_MEAN_MM = RADIAL_ARTERY_X_POSITION_MEAN_MM + DISTANCE_RADIAL_AND_ULNA_ARTERY_MEAN_MM / 2
    ARTERY_X_POSITION_UNCERTAINTY_MM = DISTANCE_RADIAL_AND_ULNA_ARTERY_STD_MM / (2**(1/2))
