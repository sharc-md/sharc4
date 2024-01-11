import os

import pytest
from SHARC_MOLCAS import SHARC_MOLCAS
from utils import expand_path

PATH = expand_path("$SHARC/../tests/interface")


def test_molcasversion():
    tests = [
        (os.path.join(PATH, "inputs/molcas/version1"), (18, 0)),
        (os.path.join(PATH, "inputs/molcas/version2"), (23, 6)),
        (os.path.join(PATH, "inputs/molcas/version3"), (20, 10)),
    ]

    for i, ref in tests:
        assert SHARC_MOLCAS().get_molcas_version(i) == ref


def test_molcasversion1():
    tests = [PATH, os.path.join(PATH, "inputs/molcas/version4")]

    for i in tests:
        with pytest.raises((ValueError, FileNotFoundError)):
            SHARC_MOLCAS().get_molcas_version(i)