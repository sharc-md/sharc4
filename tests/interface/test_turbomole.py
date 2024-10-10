import os

import pytest
from SHARC_TURBOMOLE import SHARC_TURBOMOLE
from utils import expand_path

PATH = "$SHARC/../tests/interface"


def test_template_fail():
    tests = [
        "inputs/turbomole/template/auxbasis-basislib",
        "inputs/turbomole/template/basis",
        "inputs/turbomole/template/method",
        "inputs/turbomole/template/dipolelvl1",
        "inputs/turbomole/template/dipolelvl2",
        "inputs/turbomole/template/scf",
        "inputs/turbomole/template/spin-scale",
    ]

    for test in tests:
        test_interface = SHARC_TURBOMOLE()
        with pytest.raises(ValueError):
            test_interface.read_template(expand_path(os.path.join(PATH, test)))


def test_template():
    tests = [
        "inputs/turbomole/template/method1",
        "inputs/turbomole/template/method2",
        "inputs/turbomole/template/method3",
        "inputs/turbomole/template/dipole0",
        "inputs/turbomole/template/dipole1",
        "inputs/turbomole/template/dipole2",
        "inputs/turbomole/template/spin-scale-dp0",
        "inputs/turbomole/template/spin-scale-dp1",
        "inputs/turbomole/template/scs",
        "inputs/turbomole/template/sos",
        "inputs/turbomole/template/dscf",
        "inputs/turbomole/template/ridft",
    ]

    for test in tests:
        test_interface = SHARC_TURBOMOLE()
        test_interface.read_template(expand_path(os.path.join(PATH, test)))


def test_request_fail():
    tests = [
        ("inputs/turbomole/qmin/QM1.in", {"method": "cc2"}),
        ("inputs/turbomole/qmin/QM2.in", {}),  # > Kr
        ("inputs/turbomole/qmin/QM1.in", {"spin-scaling": "lt-sos"}),
        ("inputs/turbomole/qmin/QM1.in", {}),  # no orca
    ]

    for qmin, templ in tests:
        test_interface = SHARC_TURBOMOLE()
        test_interface.setup_mol(expand_path(os.path.join(PATH,qmin)))
        test_interface.QMin.template.update(templ)
        test_interface._read_template = True
        test_interface._read_resources = True
        with pytest.raises(ValueError):
            test_interface.read_requests(expand_path(os.path.join(PATH,qmin)))
