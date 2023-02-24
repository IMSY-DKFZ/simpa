# SPDX-FileCopyrightText: 2021 Division of Intelligent Medical Systems, DKFZ
# SPDX-FileCopyrightText: 2021 Janek Groehl
# SPDX-License-Identifier: MIT

import os
import sys
print(sys.path)
import inspect # TODO use getdoc or getsource
#import unittest # TODO: delete
from mdutils.mdutils import MdUtils
from simpa_tests import manual_test_classes
#import simpa_tests.manual_tests.acoustic_forward_models.KWaveAcousticForwardConvenienceFunction as specific_module # TODO delete
#import simpa_tests.manual_tests as test_module # TODO delete
import ast
import importlib
import logging


class GenerateOverview():
    """
    Runs all scripts automatically and takes the created images and compares it with reference images.

    """

    def __init__(self, verbose: bool = False, save_path: str = None):
        self.verbosity = verbose
        self.import_path = "simpa_tests.manual_tests"
        self.current_dir = os.path.dirname(os.path.realpath(__file__)) # directory of this script, i.e ~/simpa/simpa_tests/manual_tests
        self.file_name = os.path.basename(__file__)
        self.reference_figures_path = os.path.join(self.current_dir, "reference_figures/")
        if save_path == None:
            self.save_path = os.path.join(self.current_dir, "figures/")
        else:
            self.save_path = save_path
        if not os.path.isdir(self.save_path):
            print(f"Created {self.save_path} directory")
            os.mkdir(self.save_path)
        self.ref_path = os.path.join(self.current_dir, "figures_ref/")

        self.mdFile = MdUtils(file_name='manual_tests_overview', title='<u>Overview of Manual Test Results</u>')

        # TODO delete the following
        self.scripts_to_neglect = []
        self.scripts_to_neglect += ["PointSourceReconstruction.py", "KWaveAcousticForwardConvenienceFunction.py"] # TODO: let script not stop for not running manual tests and mention this errors in overview
        self.scripts_to_neglect += ["DelayAndSumReconstruction.py", "SignedDelayMultiplyAndSumReconstruction.py", "DelayMultiplyAndSumReconstruction.py",
                                   "TimeReversalReconstruction.py", "MinimalKWaveTest.py", "ReproduceDISMeasurements.py",
                                   "QPAIReconstruction.py", "TestLinearUnmixingVisual.py", "SegmentationLoader.py", "SimulationWithMSOTInvision.py",
                                   "VisualiseDevices.py", "AbsorptionAndScatteringWithinHomogenousMedium.py", "ComputeDiffuseReflectance.py", 
                                   "CompareMCXResultsWithDiffusionTheory.py", "AbsorptionAndScatteringWithInifinitesimalSlabExperiment.py"]
        self.scripts_to_neglect.remove("DelayAndSumReconstruction.py")
        self.scripts_to_neglect.remove("MinimalKWaveTest.py")
        self.scripts_to_neglect.remove("SegmentationLoader.py")
        #self.scripts_to_neglect = []

    def clean_dir(self, dir):
        """
        Deletes "__pycache__" and "__init__.py" strings from dir list

        :param dir: list of files or directories
        :type dir: list

        :return: cleaned list of directories
        """

        to_delete = ["__pycache__", "__init__.py", self.file_name, "test_data", "utils.py", "manual_tests_overview.md",
         "mdutils_example_temp.py", "Example_Markdown.md", "figures", "reference_figures"]

        for name in to_delete:
            try:
                dir.remove(name)
            except ValueError:
                pass 

    def run_manual_tests(self):
        """
        
        """
        print("HERE", self.scripts_to_neglect)

        directories = os.listdir(self.current_dir)
        directories.sort()
        self.clean_dir(directories)
 
        for dir_ in directories:
            print() #TODO delete this print
            print(dir_) #TODO delete this print
            self.mdFile.new_header(level=1, title=dir_)
            files = os.listdir(os.path.join(self.current_dir, dir_))
            files.sort()
            self.clean_dir(files)

            
            
            for file in files:
                print() #TODO delete this print
                print(" ", file) #TODO delete this print

                if file in self.scripts_to_neglect: #TODO automatically let also try not working manual tests and print this in overview!
                    print(file, "has bug and has to be neglected")
                    continue

                self.mdFile.new_header(level=2, title=file)
                
                global_path = os.path.join(self.current_dir, dir_, file)            
                module_name = ".".join([self.import_path, dir_, file.split(".")[0]])

                # execute all manual test scripts
                try:
                    print(f"  import module {module_name}") #TODO delete this print
                    module = importlib.import_module(module_name)

                    # run all test classes of the current python source code
                    with open(global_path, 'r', encoding='utf-8') as source:
                        print("  parsing code") # TODO: delete this print
                        p = ast.parse(source.read())
                        classes = [node.name for node in ast.walk(p) if isinstance(node, ast.ClassDef)]
                        for class_name in classes:
                            print("    Run", class_name)
                            
                            class_ = getattr(module, class_name)

                            # run the manual test
                            test_object = class_()
                            if not self.verbosity:
                                print()
                                self.deafen(test_object.run_test, show_figure_on_screen=False, save_path=self.save_path)
                                print("HERE")
                            else:
                                test_object.run_test(show_figure_on_screen=False, save_path=self.save_path)
                except Exception as e:
                    logging.warning(f"Error Name: {type(e).__name__}")
                    logging.warning(f"Error Message: {e}")
                    self.mdFile.write(f"<font color=red>ERROR occured:</font><br>Error: {type(e).__name__}<br>Error message: {e}")

                # Comparing reference image and new generated image
                try:
                    reference_folder = os.path.join(self.reference_figures_path, os.path.splitext(file)[0])
                    ref_img_list = os.listdir(reference_folder)
                    ref_img_list.sort()
                    for img_name in ref_img_list:
                        ref_img_path = os.path.join(reference_folder, img_name)
                        img_path = os.path.join(self.save_path, img_name)
                        self.create_comparison_html_table(ref_img_path, img_path)
                except:
                    self.mdFile.write("Could not load any figures.")
                
                #break
            #break 


        # Create a table of contents
        self.mdFile.new_table_of_contents(table_title='Contents', depth=2)
        print("saving md file in os.getcwd()=", os.getcwd())
        self.mdFile.create_md_file()

    # Helper Functions
    def create_pdf(self):
        mdFile = MdUtils(file_name='manual_tests_overview', title='Overview of Manual Test Results')
        mdFile.new_header(level=1, title='Overview of Manual Test results') 

        print("saving md file in os.getcwd()=", os.getcwd())
        mdFile.create_md_file()

    def create_comparison_html_table(self, img1_path=None, img2_path=None):
        specify_size = False
        if specify_size:
            self.mdFile.write(f"""
<table>
    <tr>
        <td>Reference</td>
        <td>Generated</td>
    </tr>
    <tr>
        <td> <img src={img1_path} width=270 height=480></td>
        <td> <img src={img2_path} width=270 height=480></td>
    </tr>
</table>
""")
        else:
            self.mdFile.write(f"""
<table>
    <tr>
        <td>Reference</td>
        <td>Generated</td>
    </tr>
    <tr>
        <td> <img src={img1_path}></td>
        <td> <img src={img2_path}></td>
    </tr>
</table>
""")

    def deafen(self, method, **kwargs):
        os.system("set -v")
        logging.disable(logging.CRITICAL)
        real_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")
        method(**kwargs)
        sys.stdout = real_stdout
        logging.disable(logging.DEBUG)
        os.system("set +v")


if __name__ == '__main__':
    automatic_manual_tests = GenerateOverview()
    automatic_manual_tests.run_manual_tests()