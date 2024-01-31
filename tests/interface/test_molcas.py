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


def test_gettasks():
    tests = [
        (
            os.path.join(PATH, "inputs/molcas/tasks/QM1.in"),
            os.path.join(PATH, "inputs/molcas/tasks/template1"),
            [
                ["gateway"],
                ["seward"],
                ["rasscf", 1, 4, False, False],
                ["copy", "MOLCAS.JobIph", "MOLCAS.1.JobIph"],
                ["copy", "MOLCAS.RasOrb", "MOLCAS.1.RasOrb"],
                ["copy", "MOLCAS.rasscf.molden", "MOLCAS.1.molden"],
                ["link", "MOLCAS.1.JobIph", "JOBOLD"],
                ["rasscf", 1, 4, True, False],
                ["mclr", 0.0001, "sala=2"],
                ["alaska"],
                ["link", "MOLCAS.1.JobIph", "JOBOLD"],
                ["rasscf", 1, 4, True, False],
                ["mclr", 0.0001, "sala=1"],
                ["alaska"],
                ["link", "MOLCAS.1.JobIph", "JOBOLD"],
                ["rasscf", 1, 4, True, False],
                ["mclr", 0.0001, "sala=3"],
                ["alaska"],
                ["link", "MOLCAS.1.JobIph", "JOBOLD"],
                ["rasscf", 1, 4, True, False],
                ["mclr", 0.0001, "sala=4"],
                ["alaska"],
                ["link", "MOLCAS.1.JobIph", "JOB001"],
                ["rassi", "dm", [4]],
                ["rasscf", 2, 2, False, False],
                ["copy", "MOLCAS.JobIph", "MOLCAS.2.JobIph"],
                ["copy", "MOLCAS.RasOrb", "MOLCAS.2.RasOrb"],
                ["copy", "MOLCAS.rasscf.molden", "MOLCAS.2.molden"],
                ["link", "MOLCAS.2.JobIph", "JOBOLD"],
                ["rasscf", 2, 2, True, False],
                ["mclr", 0.0001, "sala=1"],
                ["alaska"],
                ["link", "MOLCAS.2.JobIph", "JOBOLD"],
                ["rasscf", 2, 2, True, False],
                ["mclr", 0.0001, "sala=2"],
                ["alaska"],
                ["link", "MOLCAS.2.JobIph", "JOB001"],
                ["rassi", "dm", [2]],
                ["link", "MOLCAS.1.JobIph", "JOB001"],
                ["link", "MOLCAS.2.JobIph", "JOB002"],
                ["rassi", "soc", [4, 2]],
            ],
        )
    ]

    for qmin, templ, ref in tests:
        test_interface = SHARC_MOLCAS()
        test_interface.setup_mol(qmin)
        test_interface.read_template(templ)
        test_interface.QMin.save["step"] = 0
        test_interface._read_resources = True
        test_interface.read_requests(qmin)
        test_interface.setup_interface()

        tasks = test_interface._gen_tasklist(test_interface.QMin)
        assert tasks == ref
