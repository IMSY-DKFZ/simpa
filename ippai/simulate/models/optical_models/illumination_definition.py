from ippai.simulate import Tags


def define_illumination(settings, nx, ny, nz):
    if settings[Tags.OPTICAL_MODEL] is Tags.MODEL_MCX:
        return define_illumination_mcx(settings, nx, ny, nz)
    if settings[Tags.OPTICAL_MODEL] is Tags.MODEL_MCXYZ:
        return define_illumination_mcxyz(settings, nx, ny, nz)


def define_illumination_mcx(settings, nx, ny, nz):
    if Tags.ILLUMINATION_TYPE not in settings:
        source_type = Tags.ILLUMINATION_TYPE_PENCIL
    else:
        source_type = settings[Tags.ILLUMINATION_TYPE]

    if settings[Tags.ILLUMINATION_TYPE] == Tags.ILLUMINATION_TYPE_MSOT_ACUITY_ECHO:
        source_position = [int(nx/2.0) + 0.5, int(ny/2.0 - 17.81/settings[Tags.SPACING_MM]) + 0.5, 1]
    elif Tags.ILLUMINATION_POSITION not in settings:
        source_position = [int(nx / 2.0) + 0.5, int(ny / 2.0) + 0.5, 1]
    else:
        source_position = settings[Tags.ILLUMINATION_POSITION]

    if Tags.ILLUMINATION_DIRECTION not in settings:
        source_direction = [0, 0, 1]
    else:
        source_direction = settings[Tags.ILLUMINATION_DIRECTION]

    if Tags.ILLUMINATION_PARAM1 not in settings:
        source_param1 = [0, 0, 0, 0]
    else:
        source_param1 = settings[Tags.ILLUMINATION_PARAM1]

    if Tags.ILLUMINATION_PARAM2 not in settings:
        source_param2 = [0, 0, 0, 0]
    else:
        source_param2 = settings[Tags.ILLUMINATION_PARAM2]

    return {
        "Type": source_type,
        "Pos": source_position,
        "Dir": source_direction,
        "Param1": source_param1,
        "Param2": source_param2
    }


def define_illumination_mcxyz(settings, nx, ny, nz):
    # TODO
    raise NotImplementedError("niy - only supports mcx :D")
