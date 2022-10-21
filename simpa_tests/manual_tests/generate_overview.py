# SPDX-FileCopyrightText: 2021 Division of Intelligent Medical Systems, DKFZ
# SPDX-FileCopyrightText: 2021 Janek Groehl
# SPDX-License-Identifier: MIT

import os
import sys
import inspect
#import unittest # TODO: delete
from mdutils.mdutils import MdUtils
from simpa_tests import manual_test_classes
import simpa_tests.manual_tests.acoustic_forward_models.KWaveAcousticForwardConvenienceFunction as specific_module
import simpa_tests.manual_tests as test_module
import ast
import importlib

# Run all manual tests automatically and save the figures

class GenerateOverview_unittest():
    """
    TODO: delete
    deprecated
    geht nicht da die manual tests keine unittest objects sind
    """

    def run(self):
        #suite = unittest.TestSuite()

        for test_class in manual_test_classes:

            print("\n\n", test_class, "\n\n")

            #suite.addTests(unittest.defaultTestLoader.loadTestsFromName(test_class))
            #unittest.TextTestRunner().run(suite)


class GenerateOverview():
    """

    """

    def __init__(self):
        self.import_path = "simpa_tests.manual_tests"
        self.current_dir = os.path.dirname(os.path.realpath(__file__)) # directory of this script, i.e ~/simpa/simpa_tests/manual_tests
        self.save_path = os.path.join(self.current_dir, "figures/")
        self.file_name = os.path.basename(__file__)

        if not os.path.isdir(self.save_path):
            print(f"Created {self.save_path} directory")
            os.mkdir(self.save_path)

    
    def clean_dir(self, dir):
        """
        Deletes "__pycache__" and "__init__.py" strings from dir list

        :param dir: list of files or directories
        """

        to_delete = ["__pycache__", "__init__.py", self.file_name, "test_data", "utils.py"]

        for name in to_delete:
            try:
                dir.remove(name)
            except ValueError:
                pass 

    def run_manual_tests(self):
        """
        
        """

        directories = os.listdir(self.current_dir)
        self.clean_dir(directories)
 
        for dir_ in directories:
            print()
            print(dir_)
            files = os.listdir(os.path.join(self.current_dir, dir_))
            self.clean_dir(files)
            
            for file in files:
                print()
                print(" ", file)
                # execute all manual test scripts
                global_path = os.path.join(self.current_dir, dir_, file)
                
                if False:
                    # to run the script
                    os.system(f"python3 {global_path}")
                

                module_name = ".".join([self.import_path, dir_, file.split(".")[0]])

                if False:
                    module = __import__(module_name)
                else:
                    module = importlib.import_module(module_name)

                # run all test classes of the current python source code
                with open(global_path, 'r', encoding='utf-8') as source:
                    p = ast.parse(source.read())
                    classes = [node.name for node in ast.walk(p) if isinstance(node, ast.ClassDef)]
                    for class_name in classes:
                        print("   Run", class_name)
                        class_ = getattr(module, class_name)

                        # run the manual test
                        test_object = class_()
                        test_object.run_test(show_figure_on_screen=False)#, save_path=self.save_path) # TODO: uncomment
                break
            break  

    def create_pdf(self):
        mdFile = MdUtils(file_name='manual_tests_overview', title='Overview of Manual Test Results')
        mdFile.new_header(level=1, title='Overview of Manual Test results') 

        print("saving md file in os.getcwd()=", os.getcwd())
        mdFile.create_md_file()


if __name__ == '__main__':
    automatic_manual_tests = GenerateOverview()
    automatic_manual_tests.run_manual_tests()
    #automatic_manual_tests.create_pdf()