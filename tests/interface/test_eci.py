import os

import pytest
from SHARC_ECI import SHARC_ECI
from utils import expand_path

PATH = expand_path("$SHARC/../tests/interface")

@pytest.mark.skip()
def test_template():
    tests = ["template1", "template2", "template3", "template16"]

    for t in tests:
        try:
            SHARC_ECI().read_template(os.path.join(PATH, "inputs/eci/template", t))
        except Exception as excinfo:
            pytest.fail(f"Unexpected exception raised: {excinfo}, {t}")

@pytest.mark.skip()
def test_template2():
    tests = [
        "template4",
        "template5",
        "template6",
        "template7",
        "template8",
        "template9",
        "template10",
        "template11",
        "template12",
        "template13",
        "template14",
        "template15",
        "template17"
    ]

    for t in tests:
        with pytest.raises(ValueError):
            SHARC_ECI().read_template(os.path.join(PATH, "inputs/eci/template", t))
