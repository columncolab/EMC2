================================
Contributor's Guide
================================

Your effort to help improve EMC² are much appreciated.
The following short guide provides general instructions on how you can contribute.

Bug Report and Bug Fixes
=========================

1. **Identifying Bugs**: If you encounter a bug, please check opened and closed issues to see if it's already reported. If not, you're encouraged to report it.

2. **Reporting Bugs**: Create a new issue with a clear and concise title. Include details such as:
   - Steps to reproduce
   - Expected behavior
   - Actual behavior
   - Screenshots (if applicable)
   - Environment details (OS, version of EMC², etc.)

3. **Fixing Bugs**: After reporting a bug, if you wish to fix it, please comment on the issue stating your intention. This avoids duplication of efforts.

4. **Submitting Bug Fixes**: Once fixed, submit a Pull Request (PR) referencing the bug issue number in your PR description.

Implementing New Features
==========================

1. **Feature Discussions**: Before implementation, you are encouraged to initiate a discussion through an issue.

2. **Design and Document**: Document your approach.

3. **Submit PR**: Once ready, submit a PR with detailed notes about the feature, relevant tests, and any necessary documentation updates.

Writing and Editing Documentation
==================================

1. **Documentation Contribution**: Enhance our project documentation by updating docstrings (see the **Python File Setup**
section below), guides, and README files.

2. **Clarity and Grammar**: Ensure your documentation is clear and free from grammatical errors.

3. **Submit Changes**: Use a PR to submit your documentation updates and include relevant sections that your documentation affects.

Feedback Submission
====================

1. If you have ideas or suggestions or for more structured proposals, please create a new issue.

Forking and Cloning EMC²
=========================

1. **Fork the Repository**: Click on "Fork" to create your copy of the EMC² repository.

2. **Clone the Repository**: Clone your fork to your local machine using::

    git clone https://github.com/ARM-DOE/EMC2.git

Git Branches and Setting Origin/Upstream
=========================================

1. **Create a New Branch**: Always create a new branch to work on::

    git checkout -b new-branch-name

2. **Set Upstream**: Configure the upstream repository::

    git remote add upstream https://github.com/ARM-DOE/EMC2.git

3. **Syncing Your Fork**: Regularly sync your fork with the upstream repository::

    git fetch upstream
    git merge upstream/main

Code Style
===========

1. **Follow PEP 8**: Ensure your code adheres to PEP8 style guidelines (see https://www.python.org/dev/peps/pep-0008/).

2. **Use Linters**: Employ linters such as `flake8`, `black`, or `pylint` to check your code style. Each linter has its
unique features. For example, while `black` only focuses on code formatting, `pylint` also checks for errors and provide
refactoring suggestions. Installation of those linters is straightforward. For example, to install `pylint`, one could use::

    conda install pylint
    pip install pylint


Python File Setup
=================

1. **File description**: In case you create a new module under a new `.py` file, the top of that Python script should include a brief description.
For example:

.. code-block:: python

        """
        This module contains the Model class and example Models for your use.

        """

2. **Module import order**: Following this short description, imports should be specified in an order that follows PEP8 standards:

        1. Standard library imports.
        2. Related third party imports.
        3. Local application/library specific imports.

3. **Function documentation**: Following a function's def line, but before the function code, a doc
string is required to describe input parameters and returned objects, provide references or
other helpful information. These documentation standards follow the NumPy documentation style.

For more on the NumPy documentation style:

- https://numpydoc.readthedocs.io/en/latest/format.html#docstring-standard

For example:

.. code-block:: python


        def calc_radar_bulk(instrument, model, is_conv, p_values, z_values, atm_ext, OD_from_sfc=True,
                            hyd_types=None, mie_for_ice=False, **kwargs):
            """
            Calculates the radar stratiform or convective reflectivity and attenuation
            in a sub-columns using bulk scattering LUTs assuming geometric scatterers
            (radiation scheme logic).
            Effective radii for each hydrometeor class must be provided (in model.ds).

            Parameters
            ----------
            instrument: Instrument
                The instrument to simulate. The instrument must be a lidar.
            model: Model
                The model to generate the parameters for.
            is_conv: bool
                True if the cell is convective
            p_values: ndarray
                model output pressure array in Pa.
            z_values: ndarray
                model output height array in m.
            atm_ext: ndarray
                atmospheric attenuation per layer (dB/km).
            OD_from_sfc: bool
                If True, then calculate optical depth from the surface.
            hyd_types: list or None
                list of hydrometeor names to include in calcuation. using default Model subclass types if None.
            mie_for_ice: bool
                If True, using bulk mie caculation LUTs. Otherwise, currently using the bulk C6
                scattering LUTs for 8-column severly roughned aggregate.
            Additonal keyword arguments are passed into
            :py:func:`emc2.simulator.lidar_moments.accumulate_attenuation`.

            Returns
            -------
            model: :func:`emc2.core.Model`
                The model with the added simulated lidar parameters.

            """

Unit Testing
=============

1. **Write Tests**: Ensure all new features and bug fixes come with unit tests that demonstrate the expected behavior.
The test functions should include assertion statements that check calculated vs. expected value(s), for example.

.. code-block:: python

        def test_lambda_mu():
            # We have a cloud with a constant N, increasing LWC
            # Therefore, if dispersion is fixed, slope should decrease with LWC
            # N_0 will also increases since it is directly proportional to lambda

            my_model = emc2.core.model.TestConvection()
            my_model = emc2.simulator.psd.calc_mu_lambda(my_model, hyd_type="cl", calc_dispersion=False)
            my_ds = my_model.ds
            assert np.all(my_ds["mu"] == 1 / 0.09)
            diffs = np.diff(my_ds["lambda"])
            diffs = diffs[np.isfinite(diffs)]
            assert np.all(diffs < 0)
            diffs = np.diff(my_ds["N_0"])
            diffs = diffs[np.isfinite(diffs)]
            assert np.all(diffs < 0)

2. **Testing Framework**: Use `pytest` to verify functionality.

3. **Run Your Tests**: Validate your changes before submitting by running::

    pytest tests/

Summary: Adding Changes to GitHub
=================================

1. **Commit Your Changes**: Use clear commit messages.

2. **Push to Your Fork**: Push your branch to your GitHub fork.

3. **Open a Pull Request**: Go to the EMC² repository and click "New Pull Request."

Thank you!
