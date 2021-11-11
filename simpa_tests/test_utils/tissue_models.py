# SPDX-FileCopyrightText: 2021 Computer Assisted Medical Interventions Group, DKFZ
# SPDX-FileCopyrightText: 2021 Janek Groehl
# SPDX-License-Identifier: MIT

import simpa as sp
from simpa import Tags


def create_simple_tissue_model(transducer_dim_in_mm: float, planar_dim_in_mm: float):
    """
    This is a very simple example script of how to create a tissue definition.
    It contains a muscular background, an epidermis layer on top of the muscles
    and a blood vessel.
    """
    bg_oxy = 0.5
    v1_oxy = 1.0
    v2_oxy = 0.0
    background_dictionary = sp.Settings()
    background_dictionary[Tags.MOLECULE_COMPOSITION] = (sp.MolecularCompositionGenerator()
                .append(sp.MOLECULE_LIBRARY.oxyhemoglobin(bg_oxy))
                .append(sp.MOLECULE_LIBRARY.deoxyhemoglobin(1 - bg_oxy))
                .get_molecular_composition(sp.SegmentationClasses.BLOOD))

    background_dictionary[Tags.STRUCTURE_TYPE] = Tags.BACKGROUND

    tissue_dict = sp.Settings()
    tissue_dict[Tags.BACKGROUND] = background_dictionary

    tissue_dict["vessel_1"] = sp.define_circular_tubular_structure_settings(
        tube_start_mm=[transducer_dim_in_mm / 2 - 10, 0, 5],
        tube_end_mm=[transducer_dim_in_mm / 2 - 10, planar_dim_in_mm, 5],
        molecular_composition=(sp.MolecularCompositionGenerator()
                .append(sp.MOLECULE_LIBRARY.oxyhemoglobin(v1_oxy))
                .append(sp.MOLECULE_LIBRARY.deoxyhemoglobin(1 - v1_oxy))
                .get_molecular_composition(sp.SegmentationClasses.BLOOD)),
        radius_mm=2, priority=3, consider_partial_volume=True,
        adhere_to_deformation=False
    )
    tissue_dict["vessel_2"] = sp.define_circular_tubular_structure_settings(
        tube_start_mm=[transducer_dim_in_mm / 2, 0, 10],
        tube_end_mm=[transducer_dim_in_mm / 2, planar_dim_in_mm, 10],
        molecular_composition=(sp.MolecularCompositionGenerator()
                .append(sp.MOLECULE_LIBRARY.oxyhemoglobin(v2_oxy))
                .append(sp.MOLECULE_LIBRARY.deoxyhemoglobin(1 - v2_oxy))
                .get_molecular_composition(sp.SegmentationClasses.BLOOD)),
        radius_mm=3, priority=3, consider_partial_volume=True,
        adhere_to_deformation=False
    )
    return tissue_dict
