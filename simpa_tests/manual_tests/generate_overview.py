# SPDX-FileCopyrightText: 2021 Division of Intelligent Medical Systems, DKFZ
# SPDX-FileCopyrightText: 2021 Janek Groehl
# SPDX-License-Identifier: MIT

import ast
import glob
import importlib
import inspect
import logging
import os
import shutil
import sys
import zipfile

import pypandoc
import requests
from mdutils.mdutils import MdUtils


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
        self.md_name = 'manual_tests_overview'
        self.mdFile = MdUtils(file_name=self.md_name, title='<u>Overview of Manual Test Results</u>')
        self.set_style()

        # TODO fix issue in PointSourceReconstruction.py file (make it consistent with the other manual tests)
        self.scripts_to_neglect = ["PointSourceReconstruction.py"]
        # For testing this file one can ignore the time consuming scripts
        """self.scripts_to_neglect += ["DelayAndSumReconstruction.py", 
                                   "SignedDelayMultiplyAndSumReconstruction.py", "DelayMultiplyAndSumReconstruction.py",
                                   "TimeReversalReconstruction.py", "MinimalKWaveTest.py", "ReproduceDISMeasurements.py",
                                   "QPAIReconstruction.py", "TestLinearUnmixingVisual.py", "SegmentationLoader.py", "SimulationWithMSOTInvision.py",
                                   "VisualiseDevices.py", "AbsorptionAndScatteringWithinHomogenousMedium.py", "ComputeDiffuseReflectance.py", 
                                   "CompareMCXResultsWithDiffusionTheory.py", "AbsorptionAndScatteringWithInifinitesimalSlabExperiment.py"]"""
        
    def download_reference_images(self):
        """
        removes the current reference figures directory and downloads the latest references from nextcloud
        """
        ref_imgs_path = os.path.join(self.current_dir,"reference_figures")
        if os.path.exists(ref_imgs_path):
            # Remove the directory
            shutil.rmtree(ref_imgs_path)
            print(f'Directory {ref_imgs_path} removed successfully.')
        # nextcloud url with the reference images
        self.nextcloud_url = "https://hub.dkfz.de/s/Xb96SFXbmiE5Fk8" # shared "reference_figures" folder on nextcloud
        # Specify the local directory to save the files
        zip_filepath = os.path.join(self.current_dir, "downloaded.zip")#'downloaded_folder.zip')       
        # Construct the download URL based on the public share link
        download_url = self.nextcloud_url.replace('/s/', '/index.php/s/') + '/download'
        # Send a GET request to download the file
        response = requests.get(download_url)

        if response.status_code == 200:
            # Save the file
            with open(zip_filepath, 'wb') as f:
                f.write(response.content)
            print(f'File downloaded successfully and stored at {zip_filepath}.')
        else:
            print(f'Failed to download file. Status code: {response.status_code}')

        # Open the zip file
        with zipfile.ZipFile(zip_filepath, 'r') as zip_ref:
            # Extract all the contents into the specified directory
            zip_ref.extractall(self.current_dir)

        print(f'Files extracted to {self.current_dir}')

        # Remove the zip file after extraction
        os.remove(zip_filepath)
        print(f'{zip_filepath} removed successfully.')

    def clean_dir(self, dir):
        """
        Deletes scripts from dir list that shall not be runned

        :param dir: list of files or directories
        :type dir: list

        :return: None but updateds dir to be a cleaned list without scripts that can not be runned
        """

        # do not execute the following files in the manual_tests folder
        to_be_ignored = ["__pycache__", "__init__.py", self.file_name, "test_data", "utils.py",
                         "manual_tests_overview.md", "manual_tests_overview.pdf", "manual_tests_overview.html",
                         "figures", "reference_figures", "path_config.env"]

        for name in to_be_ignored:
            try:
                dir.remove(name)
            except ValueError:
                pass 

    def run_manual_tests(self, run_tests: bool = True):
        """
        runs all the scripts and creates md file with the results figures

        :param run_tests: if true scripts are executed
        :type run_tests: bool

        :return: None
        """
        print("NEGLECT THE FOLLOWING FILES", self.scripts_to_neglect)

        directories = os.listdir(self.current_dir)
        directories.sort()
        self.clean_dir(directories)
 
        for dir_num, dir_ in enumerate(directories):
            print("\n" , dir_)
            dir_title = f"{dir_num+1}. " + dir_.replace("_", " ").capitalize()
            self.mdFile.new_header(level=1, title=dir_title)
            files = os.listdir(os.path.join(self.current_dir, dir_))
            files.sort()
            self.clean_dir(files)

            # iterate through scripts
            for file_num, file in enumerate(files):
                print("\n ", file) 

                if file in self.scripts_to_neglect:
                    print(file, "has bug or is not compatible and has to be neglected")
                    continue
                
                file_title = f"{dir_num+1}.{file_num+1} " + file.split(".py")[0]
                self.mdFile.new_header(level=2, title=file_title)
                
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

                            # write class documentation string in the markdown file
                            class_doc = inspect.getdoc(class_)
                            self.mdFile.write("- <b>Description:</b><br>")
                            self.mdFile.write(str(class_doc))

                            # run the manual test
                            test_object = class_()
                            if run_tests:
                                if not self.verbosity:
                                    self.deafen(test_object.run_test, show_figure_on_screen=False, save_path=self.save_path)
                                else:
                                    test_object.run_test(show_figure_on_screen=False, save_path=self.save_path)
                except Exception as e:
                    logging.warning(f"Error Name: {type(e).__name__}")
                    logging.warning(f"Error Message: {e}")
                    self.mdFile.write(f"\n- <font color=red><b>ERROR occured:</b></font><br>- Error: {type(e).__name__}<br>- Error message: {e}\n")

                # Write comparison of reference image and new generated image in markdown file
                self.mdFile.write("\n- <b>Comparison of reference and generated image:</b><br>\n")
                try:
                    reference_folder = os.path.join(self.reference_figures_path, os.path.splitext(file)[0])
                    ref_img_list = glob.glob(os.path.join(reference_folder, "*.png"))
                    if len(ref_img_list) == 0:
                        logging.warning("No reference image found")
                    ref_img_list.sort()
                    for ref_img_path in ref_img_list:
                        img_name = os.path.basename(ref_img_path)
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
    def set_style(self):
        self.mdFile.write("""
<style>
img {
max-width: 100%;
height: auto;
}
h1 {
    margin-top: 42px;
}
.click-zoom-left input[type=checkbox] {
    display: none
}
.click-zoom-left img {
    /* margin: 100px; */
    transition: transform 0.25s ease;
    cursor: zoom-in
}
.click-zoom-left input[type=checkbox]:checked~img {
    transform: translate(50%, 100%) scale(2);
    cursor: zoom-out
}

.click-zoom-right input[type=checkbox] {
    display: none
}
.click-zoom-right img {
    /* margin: 100px; */
    transition: transform 0.25s ease;
    cursor: zoom-in
}
.click-zoom-right input[type=checkbox]:checked~img {
    transform: translate(-50%, -100%) scale(2);
    cursor: zoom-out
}
</style>
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

    def create_html(self):
        try:
            print("saving html file in os.getcwd()=", os.getcwd())
            with open(os.path.join(os.getcwd(), self.md_name+".html"), "w") as html_file:
                text = pypandoc.convert_text(self.mdFile.get_md_text(), "html", format="md", extra_args=['--markdown-headings=atx'])
                updated_text = ""
                for row in text.split("\n"):
                    if 'href="#' in row:
                        id1 = row.find("#")
                        id2 = row.find("-")
                        row = row.replace(row[id1+1:id2+1], "")
                        updated_text += (row+"\n")
                    elif self.reference_figures_path in row:
                        updated_text += '<div class="click-zoom-left">\n<label>\n<input type="checkbox">\n'
                        updated_text += row
                        updated_text += '</label>\n</div>'
                    elif self.save_path in row:
                        updated_text += '<div class="click-zoom-right">\n<label>\n<input type="checkbox">\n'
                        updated_text += row
                        updated_text += '</label>\n</div>'
                    else:
                        updated_text += (row+"\n")
                html_file.write(updated_text)
            #pypandoc.convert_file(self.md_name + ".md", 'html', outputfile=self.md_name + '.html')
        except Exception as e:
            print("probably you should pip install pypandoc, pypandoc_binary.") 
            print(e)

if __name__ == '__main__':
    automatic_manual_tests = GenerateOverview()
    automatic_manual_tests.download_reference_images()
    automatic_manual_tests.run_manual_tests(run_tests=True)
    automatic_manual_tests.create_html()