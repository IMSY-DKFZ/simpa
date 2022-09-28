# SPDX-FileCopyrightText: 2021 Division of Intelligent Medical Systems, DKFZ
# SPDX-FileCopyrightText: 2021 Janek Groehl
# SPDX-License-Identifier: MIT

import os
import sys
import inspect
#import unittest
from mdutils.mdutils import MdUtils
from simpa_tests import manual_test_classes
import simpa_tests.manual_tests.acoustic_forward_models.KWaveAcousticForwardConvenienceFunction as specific_module
import simpa_tests.manual_tests as test_module
import ast

#mdFile = MdUtils(file_name='manual_tests_overview', title='Overview of Manual Test Results')

# Run all manual tests automatically and save the figures

class GenerateOverview_unittest():
    """
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
    Assumes that the class is named like the file itself.
    """

    def __init__(self):
        self.import_path = "simpa_tests.manual_tests"
        self.current_dir = os.path.dirname(os.path.realpath(__file__))
        self.file_name = os.path.basename(__file__)

    
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

        directories = os.listdir(self.current_dir)

        self.clean_dir(directories)
 
        #print(dir(test_module))

        #print(dir(specific_module))

        for dir_ in directories:
            files = os.listdir(os.path.join(self.current_dir, dir_))

            self.clean_dir(files)#

            
            for file in files:

                # execute all manual test scripts
                global_path = os.path.join(self.current_dir, dir_, file)
                if False:
                    
                    os.system(f"python3 {global_path}")



                with open(global_path, 'r', encoding='utf-8') as source:
                    p = ast.parse(source.read())
                    classes = [node.name for node in ast.walk(p) if isinstance(node, ast.ClassDef)]
                    print(classes)
                # ['test', 'test2', 'inner_class']



                #class_name = os.path.splitext(file)[0]       
                #module_name = ".".join([self.import_path, dir_, class_name])
                
                #print(module_name)
                #
                #print(module_name)
                #print(dir(module_name))
                #print()


                #module = __import__(module_name, fromlist=[class_name])
                #class_object = getattr(module, class_name)
                #print("\n\n\nRun ", class_name)
                       
                #print(sys.modules[module_name])
                #for name, obj in inspect.getmembers(sys.modules[module_name]):
                    #if inspect.isclass(obj):
                        #print(obj)
                
                
                #test = class_object()
                #test.run_test(show_figure_on_screen=False)



if __name__ == '__main__':
    automatic_manual_tests = GenerateOverview()
    automatic_manual_tests.run_manual_tests()