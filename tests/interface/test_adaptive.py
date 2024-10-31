import os

import pytest
from SHARC_ADAPTIVE import SHARC_ADAPTIVE
from utils import expand_path, InDir

PATH = expand_path("$SHARC/../tests/interface")


def test_template_fail():
    tests = [
        os.path.join(PATH, "inputs/adaptive/template_fail1"),
        os.path.join(PATH, "inputs/adaptive/template_fail2"),
        os.path.join(PATH, "inputs/adaptive/template_fail3"),
        os.path.join(PATH, "inputs/adaptive/template_fail4"),
        os.path.join(PATH, "inputs/adaptive/template_fail5"),
        os.path.join(PATH, "inputs/adaptive/template_fail6"),
    ]

    for templ in tests:
        with pytest.raises((ValueError, TypeError)):
            SHARC_ADAPTIVE().read_template(templ)


def test_template():
    tests = [os.path.join(PATH, "inputs/adaptive/template_pass")]

    for templ in tests:
        SHARC_ADAPTIVE().read_template(templ)


def test_run():
    tests = [
        (os.path.join(PATH, "inputs/adaptive/run/run1.template"), os.path.join(PATH, "inputs/adaptive/run/QM1.in")),
        (os.path.join(PATH, "inputs/adaptive/run/run2.template"), os.path.join(PATH, "inputs/adaptive/run/QM1.in")),
        ]

    for templ, qmin in tests:
        with InDir(os.path.join(PATH, "inputs/adaptive/run")):
            test_interface = SHARC_ADAPTIVE()
            test_interface.setup_mol(qmin)
            test_interface._read_resources = True
            test_interface.read_template(templ)
            test_interface.setup_interface()
            test_interface.read_requests(qmin)
            test_interface.set_coords(qmin)
            test_interface.run()
            test_interface.getQMout()


def test_run_fail():
    tests = [
        (os.path.join(PATH, "inputs/adaptive/run/run_fail1.template"), os.path.join(PATH, "inputs/adaptive/run/QM1.in")),
        (os.path.join(PATH, "inputs/adaptive/run/run_fail2.template"), os.path.join(PATH, "inputs/adaptive/run/QM1.in")),
    ]

    for templ, qmin in tests:
        with InDir(os.path.join(PATH, "inputs/adaptive/run")):
            test_interface = SHARC_ADAPTIVE()
            test_interface.setup_mol(qmin)
            test_interface._read_resources = True
            test_interface.read_template(templ)
            test_interface.setup_interface()
            with pytest.raises(ValueError):
                test_interface.read_requests(qmin)
                test_interface.set_coords(qmin)
                test_interface.run()
                test_interface.getQMout()
