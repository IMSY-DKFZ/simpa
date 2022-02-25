# SPDX-FileCopyrightText: 2021 Computer Assisted Medical Interventions Group, DKFZ
# SPDX-FileCopyrightText: 2021 Janek Groehl
# SPDX-License-Identifier: MIT

import os
import glob

current_dir = os.path.dirname(os.path.realpath(__file__))
if current_dir[-6:] == "source":
    folder_level = "../"
else:
    folder_level = ""
rst_files = glob.glob(os.path.join(current_dir, "simpa*.rst"))


for rst_file in rst_files:
    new_lines = list()
    file = open(rst_file, "r")
    lines = file.readlines()
    lines_iterator = iter(lines)
    for line in lines_iterator:
        if line == 'Submodules\n':
            next(lines_iterator, None)
            next(lines_iterator, None)
            next(lines_iterator, None)
        elif line == 'Subpackages\n':
            next(lines_iterator, None)
            next(lines_iterator, None)
        elif " package" in line:
            new_line = line.replace(" package", "")
            modules = new_line.split(".")
            new_lines.append(modules[-1])
        else:
            new_lines.append(line)
    file.close()
    file = open(rst_file, "w")
    file.writelines(new_lines)
    file.close()

if os.path.exists(os.path.join(current_dir, "simpa_examples.rst")):
    os.remove(os.path.join(current_dir, "simpa_examples.rst"))
simpa_examples_rst_file = open(os.path.join(current_dir, "simpa_examples.rst"), "w")
simpa_examples_rst_file.write("simpa\_examples\n=========================================\n\n.. toctree::\n   :maxdepth: 2\n\n")
examples = glob.glob(os.path.join(current_dir, "../" + folder_level + "simpa_examples/*.py"))
for example in examples:
    example_file_name = example.split("/")[-1]
    if example_file_name == "__init__.py":
        continue
    example_file_name = example_file_name.replace(".py", "")
    example_file_name_rst = example_file_name + ".rst"
    if os.path.exists(os.path.join(current_dir, example_file_name_rst)):
        os.remove(os.path.join(current_dir, example_file_name_rst))
    example_rst_file = open(os.path.join(current_dir, example_file_name_rst), "a")
    example_rst_file.write("{}\n=========================================\n\n.. literalinclude:: ../{}\n   :language: python\n   :lines: 1-\n\n".format(example_file_name, f'../simpa_examples/{example_file_name}.py'))
    example_rst_file.close()
    simpa_examples_rst_file.writelines("   {}\n".format(example_file_name))

simpa_examples_rst_file.close()

with open(os.path.join(current_dir, "../" + folder_level + "README.md"), "r") as in_file:
    readme_lines = in_file.readlines()

if os.path.exists(os.path.join(current_dir, "introduction.md")):
    os.remove(os.path.join(current_dir, "introduction.md"))
start_line_idx = readme_lines.index("# Getting started\n")
end_line_idx = readme_lines.index("# Documentation\n")

for l, readme_line in enumerate(readme_lines):
    if "[minimal optical simulation](simpa_examples/minimal_optical_simulation.py)" in readme_line:
        readme_lines[l] = readme_line.replace(
            "[minimal optical simulation](simpa_examples/minimal_optical_simulation.py)",
            "[minimal optical simulation](minimal_optical_simulation.py)")
intro_md = open(os.path.join(current_dir, "introduction.md"), "a")
intro_md.writelines(readme_lines[start_line_idx:end_line_idx])
intro_md.close()
