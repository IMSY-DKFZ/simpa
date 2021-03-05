# SIMPA Release Checklist

This checklist lists all things that should be fulfilled before publishing
a new SIMPA release and guides through all the things that have to be done
for the publication process.

## Release Types

We differentiate between major releases *M*, minor releases *m*, and
arbitrary in-between-steps *a*. Together, these make up the release
version number M.m.a.

We do not have a clear definition on when a major release is being
achieved, but every published release must have an incremented
release number to ensure an unambiguous identification of a release
version.

## Release Prerequisites

The following things *must* be done and passed to allow for publication
of the new release:

1. Run a code inspection to test for any compile errors, such as
   function calls with the wrong parameters or unresolved references.
2. Run of all *framework_tests* for example using the do_coverage.py
   script without any unexpected failures.
3. Execute the tests in the *simulation_tests* test suite and ensure
   that the results are okay.
4. Follow the instructions of any other checklist in the *checklists*
   folder.
5. Execute all examples of the *simpa_examples* folder and ensure
   they run through.
6. Make sure the SIMPA version number is adjusted in the framework.   

## Release Steps

1. Commit the code to the master branch and ensure that the Jenkins
build runs successfully.
2. Open Github mirror page and validate that the  changes are pushed.
3. Create a release tag with the correct version on Github.
4. Update the code on zenodo.
5. Upload the code to pypi.
      