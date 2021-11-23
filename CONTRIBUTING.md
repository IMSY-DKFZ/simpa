# Contributing to SIMPA

First of all: Thank you for your participation and help! It is much appreciated!
We are convinced that your contributions will help making SIMPA better, more robust, and 
reliable for everyone!

This Guide is meant to be used as a collection of How-To's to contribute to the framework.
In case you have any questions, do not hesitate to get in touch with the members of the core development team:

Kris K. Dreher (k.dreher@dkfz-heidelberg.de)

Janek Groehl (janek.grohl@cruk.cam.ac.uk)



## How to contribute

The SIMPA code is written and maintained on a closed repository that is hosted on a server
of the German Cancer Research Center and changes to the develop or master branch are mirrored on Github 
(https://github.com/CAMI-DKFZ/simpa/). As such, only the current master and develop branch of the repository are 
open source.

To make us aware of an issue, please create an issue on the SIMPA github repository.

To contribute to SIMPA, please fork the SIMPA github repository and create a pull request with a branch containing your 
suggested changes. The core team developers will then review the suggested changes and integrate these into the code 
base.

Please make sure that you have included unit tests for your code and that all previous tests still run through.

There is a regular SIMPA status meeting every Friday on even calendar weeks at 10:00 CET/CEST and you are
very welcome to participate and raise any issues or suggest new features. You can obtain the meeting links from the core
developer team.
We also have a Slack workspace that you can join if you are interested to contribute.

Please see the github guidelines for creating pull requests:
https://docs.github.com/en/github/collaborating-with-issues-and-pull-requests/about-pull-requests

## Coding style

When writing code for SIMPA, please use the PEP 8 python coding conventions (https://www.python.org/dev/peps/pep-0008/)
and consider using the following structures in your code in order to make a new
developer or someone external always know exactly what to expect.

- Classnames are written in camel-case notation `ClassName`
- Function names are written in small letter with `_` as the delimiter `function_name`
- Function parameters are always annotated with their type `arg1: type = default`
- Only use primitive types as defaults. If a non-primitive type is used, then the default should be `None` and
 the parameter should be initialized in the beginning of a function.
- A single line of code should not be longer than 120 characters.
- Functions should follow the following simple structure:
  1. Input validation (arguments all not `None`, correct type, and acceptable value ranges?)
  2. Processing (clean handling of errors that might occur)
  3. Output generation (sanity checking of the output before handing it off to the caller)

## Documenting your code
Only documented code will appear in the sphinx generated documentation.

A class should be documented using the following syntax:


    class ClassName(Superclass):
        """
        Explain how the class is used and what it does.
        """

For functions, a lot of extra attributes can be added to the documentation:


    def function_name(self, arg1:type = default, arg2:type = default) -> return_type:
        """
        Explain how the function is used and what it does.
        
        :param arg1: type, value range, Null acceptable?
        :param arg2: type, value range, Null acceptable?
        :returns: type, value range, does it return Null?
        :raises ExceptionType: explain when and why this exception is raised
        """

# List of Contributors

In the following table we list the people that have contributd to the SIMPA toolkit.
People might be listed several times if they have contributed while being affiliated
to different institutions. The contributors are sorted by end date
and then by last name.

|Year|Name|Affiliation|
|---|---|---|
|2019 - today | Kris Dreher | German Cancer Research Center |
|2020 - today | Janek Groehl | Cancer Research UK, Cambridge Institute |
|2021 - today | Niklas Holzwarth | German Cancer Research Center |
|2020 - today | Leonardo Menjivar | German Cancer Research Center |
|2021 - today | Tom Rix | German Cancer Research Center |
|2020 - today | Melanie Schellenberg | German Cancer Research Center |
|2021 - today | Patricia Vieten | German Cancer Research Center |
|2019 - 2020 | Janek Groehl | German Cancer Research Center |
