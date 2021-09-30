from simpa.utils import Tags, Settings, TISSUE_LIBRARY
from simpa.utils.libraries.structure_library import *


def create_realistic_forearm_tissue(settings):
    x_dim = settings[Tags.DIM_VOLUME_X_MM]
    y_dim = settings[Tags.DIM_VOLUME_Y_MM]
    z_dim = settings[Tags.DIM_VOLUME_Z_MM]

    background_dictionary = Settings()
    background_dictionary[Tags.MOLECULE_COMPOSITION] = TISSUE_LIBRARY.constant(1e-10, 1e-10, 1.0)
    background_dictionary[Tags.STRUCTURE_TYPE] = Tags.BACKGROUND

    tissue_dict = Settings()
    tissue_dict[Tags.BACKGROUND] = background_dictionary
    tissue_dict["muscle"] = define_horizontal_layer_structure_settings(z_start_mm=1.5, thickness_mm=100,
                                                                       molecular_composition=
                                                                       TISSUE_LIBRARY.soft_tissue(blood_volume_fraction=0.05),
                                                                       priority=1,
                                                                       consider_partial_volume=True,
                                                                       adhere_to_deformation=True)
    tissue_dict["epidermis"] = define_horizontal_layer_structure_settings(z_start_mm=1.5, thickness_mm=0.05,
                                                                          molecular_composition=
                                                                          TISSUE_LIBRARY.epidermis(0.01),
                                                                          priority=8,
                                                                          consider_partial_volume=True,
                                                                          adhere_to_deformation=True)
    tissue_dict["main_artery"] = define_circular_tubular_structure_settings(
        tube_start_mm=[x_dim/2 - 4.4, 0, 5.5],
        tube_end_mm=[x_dim/2 - 4.4, y_dim, 5.5],
        molecular_composition=TISSUE_LIBRARY.blood(0.99),
        radius_mm=1.25, priority=3, consider_partial_volume=True,
        adhere_to_deformation=True
    )
    tissue_dict["accomp_vein_1"] = define_elliptical_tubular_structure_settings(
        tube_start_mm=[x_dim/2 - 6.8, 0, 5.6],
        tube_end_mm=[x_dim/2 - 6.8, y_dim, 5.6],
        molecular_composition=TISSUE_LIBRARY.blood(0.9),
        radius_mm=0.6, priority=3, consider_partial_volume=True,
        adhere_to_deformation=True,
        eccentricity=0.8,
    )
    tissue_dict["accomp_vein_2"] = define_elliptical_tubular_structure_settings(
        tube_start_mm=[x_dim / 2 - 1.25, 0, 5.6],
        tube_end_mm=[x_dim / 2 - 1.25, y_dim, 5.6],
        molecular_composition=TISSUE_LIBRARY.blood(0.6),
        radius_mm=0.65, priority=3, consider_partial_volume=True,
        adhere_to_deformation=True,
        eccentricity=0.9,
    )

    tissue_dict["vessel_3"] = define_elliptical_tubular_structure_settings(
        tube_start_mm=[x_dim - 6.125, 0, 3.5],
        tube_end_mm=[x_dim - 6.125, y_dim, 3.5],
        molecular_composition=TISSUE_LIBRARY.blood(0.99),
        radius_mm=0.65, priority=3, consider_partial_volume=True,
        adhere_to_deformation=False,
        eccentricity=0.93,
    )

    tissue_dict["vessel_4"] = define_elliptical_tubular_structure_settings(
        tube_start_mm=[5.2, 0, 4.5],
        tube_end_mm=[5.2, y_dim, 4.5],
        molecular_composition=TISSUE_LIBRARY.blood(0.5),
        radius_mm=0.1, priority=3, consider_partial_volume=True,
        adhere_to_deformation=False,
        eccentricity=0,
    )

    tissue_dict["vessel_5"] = define_elliptical_tubular_structure_settings(
        tube_start_mm=[x_dim - 10.4, 0, 6],
        tube_end_mm=[x_dim - 10.4, y_dim, 6],
        molecular_composition=TISSUE_LIBRARY.blood(0.5),
        radius_mm=0.1, priority=3, consider_partial_volume=True,
        adhere_to_deformation=False,
        eccentricity=0,
    )

    tissue_dict["vessel_6"] = define_elliptical_tubular_structure_settings(
        tube_start_mm=[x_dim - 14.4, 0, 3],
        tube_end_mm=[x_dim - 14.4, y_dim, 3],
        molecular_composition=TISSUE_LIBRARY.soft_tissue(blood_volume_fraction=0.2),
        radius_mm=0.1, priority=3, consider_partial_volume=True,
        adhere_to_deformation=False,
        eccentricity=0.99,
    )

    return tissue_dict