#IPPAI

The Image Processing for Photoacoustic Imaging (IPPAI) toolkit.

## Internal Install Instructions

These install instructions are made under the assumption that you have access to the phabricator ippai project.
When you are reading these instructions there is a 99% chance that is the case (or someone send these instructions
to you).

So, for the 1% of you: Please also follow steps 1 - 3:

1. `git clone https://phabricator.mitk.org/source/ippai.git`
2. `git checkout master`
3. `git pull`

Now open a python instance in the 'ippai' folder that you have just downloaded. Make sure that you have your preferred
virtual environment activated
4. `cd ippai`
5. `python -m setup.py build install`
6. Test if the installation worked by using `python` followed by `import ippai` then `exit()`

If no error messages arise, you are now setup to use ippai in your project.

## Overview

The main use case for the ippai framework is the simulation of photoacoustic images.
However, it can also be used for image processing.

### Simulating photoacoustic images