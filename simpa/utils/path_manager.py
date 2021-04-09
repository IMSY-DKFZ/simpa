import os


class Path_manager():
    def __init__(self):
        pass

    @staticmethod
    def get_path():
        SAVE_PATH = get_os('SAVE_PATH', default='/path/to/save/folder', required=True)
        MCX_BINARY_PATH = get_os('MCX_BINARY_PATH', default='/path/to/mcx.exe', required=False)
        MATLAB_PATH = get_os('MATLAB_PATH', default='/path/to/matlab.exe', required=False)
        ACOUSTIC_MODEL_SCRIPT = get_os('ACOUSTIC_MODEL_SCRIPT', default='/path/to/simpa/core/acoustic_simulation', required=False)

        return SAVE_PATH, MCX_BINARY_PATH, MATLAB_PATH, ACOUSTIC_MODEL_SCRIPT

    
def get_os(name, default=None, required=False):
    os_var = os.environ.get(name)
    if os_var is None and default is not None:
        os_var = default
    return os_var

