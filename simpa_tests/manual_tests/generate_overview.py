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
from importlib.metadata import version

import pypandoc
import requests
from mdutils.mdutils import MdUtils

from simpa.log import Logger


class GenerateOverview():
    """
    Runs all scripts automatically and takes the created images and compares it with reference images.
    """

    def __init__(self, verbose: bool = False, save_path: str = None):
        self.verbosity = verbose
        self.logger = Logger()
        self.import_path = "simpa_tests.manual_tests"
        # directory of this script, i.e ~/simpa/simpa_tests/manual_tests
        self.current_dir = os.path.dirname(os.path.realpath(__file__))
        self.file_name = os.path.basename(__file__)
        self.reference_figures_path = os.path.join(self.current_dir, "reference_figures/")
        if save_path == None:
            self.save_path = os.path.join(self.current_dir, "figures/")
        else:
            self.save_path = save_path
        os.makedirs(self.save_path, exist_ok=True)
        self.logger.debug(f"Created {self.save_path} directory")
        self.md_name = 'manual_tests_overview'
        self.mdFile = MdUtils(file_name=self.md_name, title='<u>Overview of Manual Test Results</u>')
        self.set_style()

        # If you manually want to neglect a specific manual test enter the python script name here
        self.scripts_to_neglect = []

    def download_reference_images(self):
        """
        Removes the current reference figures directory and downloads the latest references from nextcloud.

        :return: None
        """
        ref_imgs_path = os.path.join(self.current_dir, "reference_figures")
        if os.path.exists(ref_imgs_path):
            # Remove the directory
            shutil.rmtree(ref_imgs_path)
        # nextcloud url with the reference images
        self.nextcloud_url = "https://hub.dkfz.de/s/Xb96SFXbmiE5Fk8"  # shared "reference_figures" folder on nextcloud
        # Specify the local directory to save the files
        zip_filepath = os.path.join(self.current_dir, "downloaded.zip")
        # Construct the download URL based on the public share link
        download_url = self.nextcloud_url.replace('/s/', '/index.php/s/') + '/download'
        # Send a GET request to download the file
        self.logger.debug(f'Download folder with reference figures from nextcloud...')
        response = requests.get(download_url)
        if response.status_code == 200:
            # Save the file
            with open(zip_filepath, 'wb') as f:
                f.write(response.content)
            self.logger.debug(f'File downloaded successfully and stored at {zip_filepath}.')
        else:
            self.logger.critical(f'Failed to download file. Status code: {response.status_code}')
            raise requests.exceptions.HTTPError(f'Failed to download file. Status code: {response.status_code}')

        # Open the zip file
        with zipfile.ZipFile(zip_filepath, 'r') as zip_ref:
            # Extract all the contents into the specified directory
            zip_ref.extractall(self.current_dir)

        self.logger.debug(f'Files extracted to {self.current_dir}')

        # Remove the zip file after extraction
        os.remove(zip_filepath)
        self.logger.debug(f'{zip_filepath} removed successfully.')

    def clean_dir(self, dir):
        """
        Removes scripts from the directory list that should not be run.

        :param dir: List of files or directories.
        :type dir: list

        :return: None
        """

        # do not execute the following files in the manual_tests folder
        to_be_ignored = ["__pycache__", "__init__.py", self.file_name, "test_data", "utils.py",
                         "manual_tests_overview.md", "manual_tests_overview.pdf", "manual_tests_overview.html",
                         "figures", "reference_figures", "path_config.env", "version.txt"]

        for name in to_be_ignored:
            try:
                dir.remove(name)
            except ValueError:
                pass

    def log_version(self):
        """
        Logs the current 'simpa' version to a file and compares it with a reference version.

        :return: None
        """
        self.simpa_version = version("simpa")
        with open(os.path.join(self.save_path, "simpa_version.txt"), "w") as file:
            file.write(self.simpa_version)

        ref_version_path = os.path.join(self.reference_figures_path, "simpa_version.txt")
        try:
            with open(ref_version_path, 'r') as file:
                reference_sp_version = file.read()
            self.mdFile.write(f"""
<b>SIMPA versions:</b><br>\n\n
<table>
    <tr>
        <td>Reference simpa version:</td>
        <td>{reference_sp_version}</td>
    </tr>
    <tr>
        <td>Your simpa version:</td>
        <td>{self.simpa_version}</td>
    </tr>
</table>
""")
            if self.simpa_version != reference_sp_version:
                self.logger.debug(
                    "Your simpa version does not match with the simpa version used for generating the reference figures")
        except FileNotFoundError:
            self.logger.warning(f"The reference simpa version file at {ref_version_path} was not found")
        except IOError:
            self.logger.warning(f"An error occurred while reading the file at {ref_version_path}")

    def run_manual_tests(self, run_tests: bool = True):
        """
        runs all the scripts and creates md file with the results figures

        :param run_tests: if true scripts are executed
        :type run_tests: bool

        :return: None
        """
        self.logger.debug(f"Neglect the following files: {self.scripts_to_neglect}")

        directories = os.listdir(self.current_dir)
        directories.sort()
        self.clean_dir(directories)

        for dir_num, dir_ in enumerate(directories):
            self.logger.debug(f"Enter dir: {dir_}")
            dir_title = f"{dir_num+1}. " + dir_.replace("_", " ").capitalize()
            self.mdFile.new_header(level=1, title=dir_title)
            files = os.listdir(os.path.join(self.current_dir, dir_))
            files.sort()
            self.clean_dir(files)

            # iterate through scripts
            for file_num, file in enumerate(files):
                self.logger.debug(f"Enter file: {file}")
                test_save_path = os.path.join(self.save_path, file.split(".py")[0] + "/")
                os.makedirs(test_save_path, exist_ok=True)

                if file in self.scripts_to_neglect:
                    self.logger.debug(f"{file} has bug or is not compatible and has to be neglected")
                    continue

                file_title = f"{dir_num+1}.{file_num+1} " + file.split(".py")[0]
                self.mdFile.new_header(level=2, title=file_title)

                global_path = os.path.join(self.current_dir, dir_, file)
                module_name = ".".join([self.import_path, dir_, file.split(".")[0]])

                # execute all manual test scripts
                try:
                    self.logger.debug(f"import module {module_name}")
                    module = importlib.import_module(module_name)

                    # run all test classes of the current python source code
                    with open(global_path, 'r', encoding='utf-8') as source:
                        p = ast.parse(source.read())
                        classes = [node.name for node in ast.walk(p) if isinstance(node, ast.ClassDef)]
                        for class_name in classes:
                            self.logger.debug(f"Run {class_name}")

                            class_ = getattr(module, class_name)

                            # write class documentation string in the markdown file
                            class_doc = inspect.getdoc(class_)
                            self.mdFile.write("- <b>Description:</b><br>")
                            self.mdFile.write(str(class_doc))

                            # run the manual test
                            test_object = class_()
                            if run_tests:
                                if not self.verbosity:
                                    self.deafen(test_object.run_test, show_figure_on_screen=False,
                                                save_path=test_save_path)
                                else:
                                    test_object.run_test(show_figure_on_screen=False, save_path=test_save_path)
                except Exception as e:
                    self.logger.warning(f"Error Name: {type(e).__name__}")
                    self.logger.warning(f"Error Message: {e}")
                    self.mdFile.write(
                        f"\n- <font color=red><b>ERROR occured:</b></font><br>- Error: {type(e).__name__}<br>- Error message: {e}\n")

                # Write comparison of reference image and new generated image in markdown file
                self.mdFile.write("\n- <b>Comparison of reference and generated image:</b><br>\n")
                try:
                    reference_folder = os.path.join(self.reference_figures_path, os.path.splitext(file)[0])
                    ref_img_list = glob.glob(os.path.join(reference_folder, "*.png"))
                    if len(ref_img_list) == 0:
                        self.logger.warning("No reference image found")
                    ref_img_list.sort()
                    for ref_img_path in ref_img_list:
                        img_name = os.path.basename(ref_img_path)
                        img_path = os.path.join(test_save_path, img_name)
                        self.create_comparison_html_table(ref_img_path, img_path)
                except:
                    self.mdFile.write("Could not load any figures.")

        # Create a table of contents
        self.mdFile.new_table_of_contents(table_title='Contents', depth=2)
        self.logger.debug(f"Saving md file in {os.getcwd()=}")
        self.mdFile.create_md_file()

    # Helper Functions
    def create_comparison_html_table(self, img1_path=None, img2_path=None):
        """
        Creates an HTML table to compare two images, with optional size specification.

        :param img1_path: Path to the first image.
        :type img1_path: str or None
        :param img2_path: Path to the second image.
        :type img2_path: str or None

        :return: None
        """
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
        """
        Writes CSS styles to the Markdown file for image and header formatting, including zoom functionality.

        :return: None
        """
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
        """
        Suppresses output and logging temporarily while executing a specified method.

        :param method: The method to execute with suppressed output and logging.
        :type method: callable
        :param kwargs: Keyword arguments to pass to the method.
        :type kwargs: dict

        :return: None
        """

        os.system("set -v")
        self.logger._logger.setLevel(logging.CRITICAL)
        real_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")
        method(**kwargs)
        sys.stdout = real_stdout
        self.logger._logger.setLevel(logging.DEBUG)
        os.system("set +v")

    def create_html(self):
        """
        Creates an HTML table to compare generated and reference figures.

        :return: None
        """
        try:
            self.logger.debug(f"Saving html file in {os.getcwd()=}")
            with open(os.path.join(os.getcwd(), self.md_name+".html"), "w") as html_file:
                text = pypandoc.convert_text(self.mdFile.get_md_text(), "html", format="md",
                                             extra_args=['--markdown-headings=atx'])
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
            # pypandoc.convert_file(self.md_name + ".md", 'html', outputfile=self.md_name + '.html')
        except Exception as e:
            self.logger.warning("Check installation of needed requirements (pypandoc, pypandoc_binary).")


if __name__ == '__main__':
    automatic_manual_tests = GenerateOverview()
    automatic_manual_tests.download_reference_images()
    automatic_manual_tests.log_version()
    automatic_manual_tests.run_manual_tests(run_tests=True)
    automatic_manual_tests.create_html()
