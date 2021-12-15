# Contributing to SIMPA

First of all: Thank you for your participation and help! It is much appreciated!
We are convinced that your contributions will help to make SIMPA better, more robust, and 
reliable for everyone!

This Guide is meant to be used as a collection of How-Tos to contribute to the framework.
In case you have any questions, do not hesitate to get in touch with the members of the core development team:

Kris K. Dreher (k.dreher@dkfz-heidelberg.de)

Janek Groehl (janek.grohl@cruk.cam.ac.uk)

* [How to contribute](#how-to-contribute)
* [Documenting your code](#documenting-your-code)
* [Coding style](#coding-style)

## How to contribute

If you are interested in contributing to SIMPA don’t hesitate to contact the core development team early on 
to discuss your ideas and find the best strategy for integration. 
There is also a regular SIMPA status meeting every Friday on even calendar weeks at 10:00 CET/CEST, 
and you are very welcome to participate and raise any issues or discuss your or suggest new features. 
Once you reached out to us, you will be provided with the information on how to join.
In general the following steps are involved during a contribution:

### Contribution process
1.	Create feature request / bug report on the [SIMPA issues page](https://github.com/CAMI-DKFZ/simpa/issues)
2.	Discuss potential contribution with core development team
3.	Fork the [SIMPA repository](https://github.com/CAMI-DKFZ/simpa)
4.	Create feature branch from develop using the naming convention T<Issue#>_<FeatureName>, 
      where <Issue#> represent the number github assigned the created issue and <FeatureName> describes 
      what is being developed in CamelCaseNotation.
    Examples: `T13_FixSimulatorBug`, `T27_AddNewSimulator`

5.	Perform test driven development on feature branch. 
      A new implemented feature / a bug fix should be accompanied by a test. 
      Additionally, all previously existing tests must still pass after the contribution. 
6.	Once development is finished, create a pull request including your changes. 
      For more information on how to create pull request, see GitHub's [about pull requests](https://docs.github.com/en/github/collaborating-with-issues-and-pull-requests/about-pull-requests).
7.	A member of the core development team will review your pull request and potentially require further changes 
      (see [Contribution review and integration](#contribution-review-and-integration)). 
      Once all remarks have been resolved, your changes will be merged into the develop branch.

For each contribution, please make sure to consider the following:

### Contribution origin, rights, and sign-off

In addition to your actual contribution, we need to ensure that you actually have the right to make that contribution.
SIMPA is licensed under a MIT license consequently your work must be licensed under a compatible license. 
If you write new code or if you are allowed to re-license it, you might want to use SIMPA’s license to ease integration.
In case your contribution contains software patented by you, it should be licensed under the Apache 2.0 license. 
For all contributions involving patented software, 
consider getting in touch with the SIMPA developers early on to discuss potential issues.
To confirm the contribution rights and track and acknowledge individual contributions, 
we utilize a “sign-off” procedure on contributions submitted to SIMPA. 
Each commit / patch needs to be equipped with a “sign-off” statement 
which certifies that you wrote the contribution or otherwise have the right to pass it on as open-source 
as defined in the following:

```text
Developer's Certificate of Origin 1.1

By making a contribution to this project, I certify that:

a)	The contribution was created in whole or in part by me and 
        I have the right to submit it under the open source license indicated in the file; or
b)	The contribution is based upon previous work that, to the best of my knowledge, 
        is covered under an appropriate open source license and I have the right under that license to submit 
        that work with modifications, whether created in whole or in part by me, under the same open source license 
        (unless I am permitted to submit under a different license), as indicated in the file; or
c)	The contribution was provided directly to me by some other person who certified (a), (b) or (c) 
        and I have not modified it.
d)	I understand and agree that this project and the contribution are public and that a record of the contribution 
        (including all personal information I submit with it, including my sign-off) is maintained indefinitely 
        and may be redistributed consistent with this project or the open source license(s) involved.
```

The sign-off statement needs to be at least a line containing your full name, and a valid e-mail address assigned to you:
 
`Signed-off-by: YOUR_NAME <YOUR-EMAIL>`

We recommend making use of the `--signoff` option of the git commit command:
`git commit --signoff “YOUR COMMIT MESSAGE”`
In case already existing commits need to be signed off, you can make use of the `--amend` option:

`git commit --amend --signoff`

### Contribution review and integration
To ensure correctness and high quality of the submitted code, each contribution will be reviewed by a member of the core development team regarding among others the following aspects:
- The code is correct and implements the described feature / fixes the described issue.
- The code follows the [SIMPA coding style](#coding-style)
- The code is [documented appropriately](#documenting-your-code)
- The code is covered by sensible unit tests that pass upon submission
- The contribution does not lead to side effects in other parts of the toolkit (e.g. failing tests)
Once the reviewer is content with the contribution, the changes will be integrated into the code base.

### Contact 

There is a regular SIMPA status meeting every Friday on even calendar weeks at 10:00 CET/CEST, and you are
very welcome to participate and raise any issues or suggest new features. You can obtain the meeting links from the core
developer team.
We also have a Slack workspace that you can join if you are interested to contribute.

## Coding style

When writing code for SIMPA, please use the [PEP 8](https://www.python.org/dev/peps/pep-0008/) python coding conventions
and consider using the following structures in your code in order to make a new
developer or someone external always know exactly what to expect.

- Class names are written in camel-case notation `ClassName`
- Function names are written in lowercase with `_` as the delimiter `function_name`
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
        Description of how the class is used and what it does.
        List all attributes in the following fashion:

        Attributes:
            att_1 (att_1_type): description for att_1
            att_2 (att_2_type): description for att_2
        """

For functions, a lot of extra attributes can be added to the documentation:


    def function_name(self, arg1 = default, arg2 = default) -> return_type:
        """
        Explain how the function is used and what it does.
        
        :param arg1: value range, Null acceptable?
        :type arg1: type_of_arg1
        :param arg2: value range, Null acceptable?
        :type arg2: type_of_arg2
        :returns: value range, does it return Null?
        :rtype: return type
        :raises ExceptionType: explain when and why this exception is raised
        """

# List of Contributors

In the following table we list the people that have contributed to the SIMPA toolkit.
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
