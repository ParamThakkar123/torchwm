"""Evaluation and interactive-play entry points used by ``torchwm eval`` / ``torchwm play``.

These used to live in the repository's ``scripts/`` directory, which is not part
of the wheel, so the installed CLI could only run them from a source checkout.
Modules are imported lazily by the CLI because they pull in OpenCV and the
evaluation networks.
"""
