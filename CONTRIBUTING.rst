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

1. **Documentation Contribution**: Enhance our project documentation by updating docstrings, guides, and README files.

2. **Clarity and Grammar**: Ensure your documentation is clear and free from grammatical errors.

3. **Submit Changes**: Use a PR to submit your documentation updates and include relevant sections that your documentation affects.

Feedback Submission
====================

1. If you have ideas or suggestions or for more structured proposals, please create a new issue.

Forking and Cloning EMC²
=========================

1. **Fork the Repository**: Click on "Fork" to create your copy of the EMC² repository.

2. **Clone the Repository**: Clone your fork to your local machine using::

    git clone https://github.com/columncolab/EMC2.git

Git Branches and Setting Origin/Upstream
=========================================

1. **Create a New Branch**: Always create a new branch to work on::

    git checkout -b new-branch-name

2. **Set Upstream**: Configure the upstream repository::

    git remote add upstream https://github.com/columncolab/EMC2.git

3. **Syncing Your Fork**: Regularly sync your fork with the upstream repository::

    git fetch upstream
    git merge upstream/main

Code Style
===========

1. **Follow PEP 8**: Ensure your code adheres to PEP 8 style guidelines.

2. **Use Linters**: Employ linters such as `flake8`, `black`, or `pylint` to check your code style.

Unit Testing
=============

1. **Write Tests**: Ensure all new features and bug fixes come with unit tests that demonstrate the expected behavior.

2. **Testing Framework**: Use `pytest` to verify functionality.

3. **Run Your Tests**: Validate your changes before submitting by running::

    pytest tests/

Summary: Adding Changes to GitHub
=================================

1. **Commit Your Changes**: Use clear commit messages.

2. **Push to Your Fork**: Push your branch to your GitHub fork.

3. **Open a Pull Request**: Go to the EMC² repository and click "New Pull Request."

Thank you!
