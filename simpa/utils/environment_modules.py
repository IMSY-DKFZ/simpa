import os
import subprocess


def use_environment_module(env_module: str, module_version: str, module_option: str):
    if module_option not in ["load", "unload"]:
        raise ValueError("Choose either 'load' or 'unload' as module option!")
    module_command = list()
    module_command.append("bash")
    module_command.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "use_env_module.sh"))
    module_command.append(module_option)
    subprocess.run(module_command)
