import os
import shutil

import h5py
import numpy as np
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


def test_get_features():
    tests = [
        (os.path.join(PATH, "inputs/molcas/features/bin1/"), (False, True, True)),
        (os.path.join(PATH, "inputs/molcas/features/bin2/"), (True, True, True)),
        (os.path.join(PATH, "inputs/molcas/features/bin3/"), (False, True, False)),
        (os.path.join(PATH, "inputs/molcas/features/bin4/"), (False, False, False)),
    ]

    for path, ref in tests:
        test_interface = SHARC_MOLCAS()
        test_interface.QMin.resources["molcas"] = path
        test_interface._get_molcas_features()
        assert (test_interface._wfa, test_interface._hdf5, test_interface._mpi) == ref


def test_get_features2():
    """
    HDF, MPI should be false if ldd not installed
    """
    old_environ = dict(os.environ)
    os.environ.update({"PATH": "/tmp"})
    test_interface = SHARC_MOLCAS()
    test_interface.QMin.resources["molcas"] = os.path.join(PATH, "inputs/molcas/features/bin2/")
    test_interface._get_molcas_features()
    os.environ.clear()
    os.environ.update(old_environ)
    assert (test_interface._wfa, test_interface._hdf5, test_interface._mpi) == (True, False, False)


def test_generate_schedule():
    """
    Test if joblist contains all job keys
    """
    tests = [
        (
            os.path.join(PATH, "inputs/molcas/schedule/QM1.in"),
            [{"master"}, {"grad_1_1", "grad_1_2", "grad_1_3", "grad_1_4", "grad_2_1", "grad_2_2"}],
        ),
        (
            os.path.join(PATH, "inputs/molcas/schedule/QM2.in"),
            [
                {"master"},
                {
                    "grad_1_1",
                    "grad_1_2",
                    "grad_1_3",
                    "grad_1_4",
                    "grad_2_1",
                    "grad_2_2",
                    "nacdr_1_1_1_2",
                    "nacdr_1_1_1_3",
                    "nacdr_1_1_1_4",
                    "nacdr_1_2_1_3",
                    "nacdr_1_2_1_4",
                    "nacdr_1_3_1_4",
                    "nacdr_2_1_2_2",
                },
            ],
        ),
        (
            os.path.join(PATH, "inputs/molcas/schedule/QM3.in"),
            [{"master"}, {"grad_1_1", "grad_1_2", "grad_1_3", "grad_1_4", "grad_2_1", "grad_2_2", "nacdr_1_1_1_2"}],
        ),
        (
            os.path.join(PATH, "inputs/molcas/schedule/QM4.in"),
            [{"master"}, {"grad_1_1", "grad_2_1", "nacdr_1_1_1_2"}],
        ),
        (
            os.path.join(PATH, "inputs/molcas/schedule/QM5.in"),
            [{"master"}],
        ),
    ]

    for qmin, ref in tests:
        test_interface = SHARC_MOLCAS()
        test_interface.setup_mol(qmin)
        test_interface._read_template = True
        test_interface._read_resources = True
        test_interface.setup_interface()
        test_interface.read_requests(qmin)
        schedule = test_interface._generate_schedule()
        assert len(schedule) == len(ref)
        for idx, i in enumerate(ref):
            assert schedule[idx].keys() == i


def test_gettasks_init():
    # Test different requests from INIT
    # Currently no always_guess, always_orb_init
    tests = [
        (
            os.path.join(PATH, "inputs/molcas/tasks/QM1.in"),
            os.path.join(PATH, "inputs/molcas/tasks/template_casscf"),
            {
                "master": (
                    0,
                    [
                        ["gateway"],
                        ["seward"],
                        ["rasscf", 1, 4, False, False],
                        ["copy", "MOLCAS.RasOrb", "MOLCAS.1.RasOrb"],
                        ["copy", "MOLCAS.rasscf.molden", "MOLCAS.1.molden"],
                        ["copy", "MOLCAS.JobIph", "MOLCAS.1.JobIph"],
                        ["rasscf", 2, 2, False, False],
                        ["copy", "MOLCAS.RasOrb", "MOLCAS.2.RasOrb"],
                        ["copy", "MOLCAS.rasscf.molden", "MOLCAS.2.molden"],
                        ["copy", "MOLCAS.JobIph", "MOLCAS.2.JobIph"],
                        ["link", "MOLCAS.1.JobIph", "JOB001"],
                        ["rassi", "dm", [4]],
                        ["link", "MOLCAS.2.JobIph", "JOB001"],
                        ["rassi", "dm", [2]],
                        ["link", "MOLCAS.1.JobIph", "JOB001"],
                        ["link", "MOLCAS.2.JobIph", "JOB002"],
                        ["rassi", "soc", [4, 2]],
                    ],
                ),
                "grad_1_1": (
                    1,
                    [
                        ["copy", "$master_path/MOLCAS.1.JobIph", "JOBOLD"],
                        ["rasscf", 1, 4, True, False],
                        ["mclr", 0.0001, "sala=1"],
                        ["alaska"],
                    ],
                ),
            },
        ),
        (
            os.path.join(PATH, "inputs/molcas/tasks/QM2.in"),
            os.path.join(PATH, "inputs/molcas/tasks/template_casscf"),
            {
                "master": (
                    0,
                    [
                        ["gateway"],
                        ["seward"],
                        ["rasscf", 1, 4, False, False],
                        ["copy", "MOLCAS.JobIph", "MOLCAS.1.JobIph"],
                        ["rasscf", 2, 2, False, False],
                        ["copy", "MOLCAS.JobIph", "MOLCAS.2.JobIph"],
                    ],
                )
            },
        ),
        (
            os.path.join(PATH, "inputs/molcas/tasks/QM3.in"),
            os.path.join(PATH, "inputs/molcas/tasks/template_casscf"),
            {
                "master": (
                    0,
                    [
                        ["gateway"],
                        ["seward"],
                        ["rasscf", 1, 4, False, False],
                        ["copy", "MOLCAS.JobIph", "MOLCAS.1.JobIph"],
                        ["rasscf", 2, 2, False, False],
                        ["copy", "MOLCAS.JobIph", "MOLCAS.2.JobIph"],
                    ],
                ),
                "grad_1_2": (
                    1,
                    [
                        ["copy", "$master_path/MOLCAS.1.JobIph", "JOBOLD"],
                        ["rasscf", 1, 4, True, False],
                        ["mclr", 0.0001, "sala=2"],
                        ["alaska"],
                    ],
                ),
            },
        ),
        (
            os.path.join(PATH, "inputs/molcas/tasks/QM4.in"),
            os.path.join(PATH, "inputs/molcas/tasks/template_casscf"),
            {
                "master": (
                    0,
                    [
                        ["gateway"],
                        ["seward"],
                        ["rasscf", 1, 4, False, False],
                        ["copy", "MOLCAS.JobIph", "MOLCAS.1.JobIph"],
                        ["rasscf", 2, 2, False, False],
                        ["copy", "MOLCAS.JobIph", "MOLCAS.2.JobIph"],
                        ["link", "MOLCAS.1.JobIph", "JOB001"],
                        ["rassi", "dm", [4]],
                        ["link", "MOLCAS.2.JobIph", "JOB001"],
                        ["rassi", "dm", [2]],
                        ["link", "MOLCAS.1.JobIph", "JOB001"],
                        ["link", "MOLCAS.2.JobIph", "JOB002"],
                        ["rassi", "soc", [4, 2]],
                    ],
                ),
                "grad_1_1": (
                    1,
                    [
                        ["copy", "$master_path/MOLCAS.1.JobIph", "JOBOLD"],
                        ["rasscf", 1, 4, True, False],
                        ["mclr", 0.0001, "sala=1"],
                        ["alaska"],
                    ],
                ),
                "nacdr_1_1_1_2": (
                    1,
                    [
                        ["copy", "$master_path/MOLCAS.1.JobIph", "MOLCAS.1.JobIph"],
                        ["link", "MOLCAS.1.JobIph", "JOBOLD"],
                        ["rasscf", 1, 4, True, False],
                        ["mclr", 0.0001, "nac=1 2"],
                        ["alaska"],
                        ["link", "MOLCAS.1.JobIph", "JOB001"],
                        ["link", "MOLCAS.JobIph", "JOB002"],
                        ["rassi", "overlap", [4, 4]],
                    ],
                ),
                "nacdr_1_1_1_3": (
                    1,
                    [
                        ["copy", "$master_path/MOLCAS.1.JobIph", "MOLCAS.1.JobIph"],
                        ["link", "MOLCAS.1.JobIph", "JOBOLD"],
                        ["rasscf", 1, 4, True, False],
                        ["mclr", 0.0001, "nac=1 3"],
                        ["alaska"],
                        ["link", "MOLCAS.1.JobIph", "JOB001"],
                        ["link", "MOLCAS.JobIph", "JOB002"],
                        ["rassi", "overlap", [4, 4]],
                    ],
                ),
            },
        ),
        (
            os.path.join(PATH, "inputs/molcas/tasks/QM1.in"),
            os.path.join(PATH, "inputs/molcas/tasks/template_caspt2"),
            {
                "master": (
                    0,
                    [
                        ["gateway"],
                        ["seward"],
                        ["rasscf", 1, 4, False, False],
                        ["copy", "MOLCAS.RasOrb", "MOLCAS.1.RasOrb"],
                        ["copy", "MOLCAS.rasscf.molden", "MOLCAS.1.molden"],
                        ["caspt2", 1, 4, "caspt2"],
                        ["copy", "MOLCAS.JobMix", "MOLCAS.1.JobIph"],
                        ["rasscf", 2, 2, False, False],
                        ["copy", "MOLCAS.RasOrb", "MOLCAS.2.RasOrb"],
                        ["copy", "MOLCAS.rasscf.molden", "MOLCAS.2.molden"],
                        ["caspt2", 2, 2, "caspt2"],
                        ["copy", "MOLCAS.JobMix", "MOLCAS.2.JobIph"],
                        ["link", "MOLCAS.1.JobIph", "JOB001"],
                        ["rassi", "dm", [4]],
                        ["link", "MOLCAS.2.JobIph", "JOB001"],
                        ["rassi", "dm", [2]],
                        ["link", "MOLCAS.1.JobIph", "JOB001"],
                        ["link", "MOLCAS.2.JobIph", "JOB002"],
                        ["rassi", "soc", [4, 2]],
                    ],
                ),
            },
        ),
        (
            os.path.join(PATH, "inputs/molcas/tasks/QM4.in"),
            os.path.join(PATH, "inputs/molcas/tasks/template_mscaspt2"),
            {
                "master": (
                    0,
                    [
                        ["gateway"],
                        ["seward"],
                        ["rasscf", 1, 4, False, False],
                        ["caspt2", 1, 4, "ms-caspt2"],
                        ["copy", "MOLCAS.JobMix", "MOLCAS.1.JobIph"],
                        ["rasscf", 2, 2, False, False],
                        ["caspt2", 2, 2, "ms-caspt2"],
                        ["copy", "MOLCAS.JobMix", "MOLCAS.2.JobIph"],
                        ["link", "MOLCAS.1.JobIph", "JOB001"],
                        ["rassi", "dm", [4]],
                        ["link", "MOLCAS.2.JobIph", "JOB001"],
                        ["rassi", "dm", [2]],
                        ["link", "MOLCAS.1.JobIph", "JOB001"],
                        ["link", "MOLCAS.2.JobIph", "JOB002"],
                        ["rassi", "soc", [4, 2]],
                    ],
                ),
                "grad_1_3": (
                    1,
                    [
                        ["copy", "$master_path/MOLCAS.1.JobIph", "JOBOLD"],
                        ["rasscf", 1, 4, True, False],
                        ["caspt2", 1, 4, "ms-caspt2", "GRDT\nrlxroot = 3"],
                        ["mclr", 0.0001],
                        ["alaska"],
                    ],
                ),
                "nacdr_2_1_2_2": (
                    1,
                    [
                        ["copy", "$master_path/MOLCAS.2.JobIph", "MOLCAS.2.JobIph"],
                        ["link", "MOLCAS.2.JobIph", "JOBOLD"],
                        ["rasscf", 2, 2, True, False],
                        ["caspt2", 2, 2, "ms-caspt2", "GRDT\nnac = 1 2"],
                        ["alaska", 1, 2],
                        ["link", "MOLCAS.2.JobIph", "JOB001"],
                        ["link", "MOLCAS.JobIph", "JOB002"],
                        ["rassi", "overlap", [2, 2]],
                    ],
                ),
            },
        ),
        (
            os.path.join(PATH, "inputs/molcas/tasks/QM1.in"),
            os.path.join(PATH, "inputs/molcas/tasks/template_cmspdft"),
            {
                "master": (
                    0,
                    [
                        ["gateway"],
                        ["seward"],
                        ["rasscf", 1, 4, False, False, ["CMSI"]],
                        ["copy", "MOLCAS.RasOrb", "MOLCAS.1.RasOrb"],
                        ["copy", "MOLCAS.rasscf.molden", "MOLCAS.1.molden"],
                        ["mcpdft", ["KSDFT=t:pbe", "noGrad", "MSPDFT", "WJOB", "CMMI=0", "CMSS=Do_Rotate.txt", "CMTH=1.0d-10"]],
                        ["copy", "MOLCAS.JobIph", "MOLCAS.1.JobIph"],
                        ["rasscf", 2, 2, False, False, ["CMSI"]],
                        ["copy", "MOLCAS.RasOrb", "MOLCAS.2.RasOrb"],
                        ["copy", "MOLCAS.rasscf.molden", "MOLCAS.2.molden"],
                        ["mcpdft", ["KSDFT=t:pbe", "noGrad", "MSPDFT", "WJOB", "CMMI=0", "CMSS=Do_Rotate.txt", "CMTH=1.0d-10"]],
                        ["copy", "MOLCAS.JobIph", "MOLCAS.2.JobIph"],
                        ["link", "MOLCAS.1.JobIph", "JOB001"],
                        ["rassi", "dm", [4]],
                        ["link", "MOLCAS.2.JobIph", "JOB001"],
                        ["rassi", "dm", [2]],
                        ["link", "MOLCAS.1.JobIph", "JOB001"],
                        ["link", "MOLCAS.2.JobIph", "JOB002"],
                        ["rassi", "soc", [4, 2]],
                    ],
                ),
                # TODO
                "grad_1_1": (
                    1,
                    [
                        ["copy", "$master_path/MOLCAS.1.JobIph", "JOBOLD"],
                        #["copy", f"{os.getcwd()}/SAVE/Do_Rotate.1.txt", "Do_Rotate.txt"],
                        ["rasscf", 1, 4, True, False, ["RLXROOT=1", "CMSI"]],
                        ["mcpdft", ["KSDFT=t:pbe", "GRAD", "MSPDFT", "WJOB"]],
                        ["alaska", 1],
                    ],
                ),
            },
        ),
        (
            os.path.join(PATH, "inputs/molcas/tasks/QM1.in"),
            os.path.join(PATH, "inputs/molcas/tasks/template_xmscaspt2"),
            {
                "master": (
                    0,
                    [
                        ["gateway"],
                        ["seward"],
                        ["rasscf", 1, 4, False, False],
                        ["copy", "MOLCAS.RasOrb", "MOLCAS.1.RasOrb"],
                        ["copy", "MOLCAS.rasscf.molden", "MOLCAS.1.molden"],
                        ["caspt2", 1, 4, "xms-caspt2"],
                        ["copy", "MOLCAS.JobMix", "MOLCAS.1.JobIph"],
                        ["rasscf", 2, 2, False, False],
                        ["copy", "MOLCAS.RasOrb", "MOLCAS.2.RasOrb"],
                        ["copy", "MOLCAS.rasscf.molden", "MOLCAS.2.molden"],
                        ["caspt2", 2, 2, "xms-caspt2"],
                        ["copy", "MOLCAS.JobMix", "MOLCAS.2.JobIph"],
                        ["link", "MOLCAS.1.JobIph", "JOB001"],
                        ["rassi", "dm", [4]],
                        ["link", "MOLCAS.2.JobIph", "JOB001"],
                        ["rassi", "dm", [2]],
                        ["link", "MOLCAS.1.JobIph", "JOB001"],
                        ["link", "MOLCAS.2.JobIph", "JOB002"],
                        ["rassi", "soc", [4, 2]],
                    ],
                ),
                "grad_1_1": (
                    1,
                    [
                        ["copy", "$master_path/MOLCAS.1.JobIph", "JOBOLD"],
                        ["rasscf", 1, 4, True, False],
                        ["caspt2", 1, 4, "xms-caspt2", "GRDT\nrlxroot = 1"],
                        ["mclr", 0.0001],
                        ["alaska"],
                    ],
                ),
            },
        ),
    ]

    with open("SAVE/MOLCAS.1.JobIph", "w") as one, open("SAVE/MOLCAS.2.JobIph", "w") as two:
        one.write("1")
        two.write("1")
    for qmin, templ, ref in tests:
        test_interface = SHARC_MOLCAS()
        test_interface.setup_mol(qmin)
        test_interface.read_template(templ)
        test_interface._read_resources = True
        test_interface.read_requests(qmin)
        test_interface.setup_interface()
        test_interface.QMin.scheduling["schedule"] = test_interface._generate_schedule()
        for k, v in ref.items():
            tasks = test_interface._gen_tasklist(test_interface.QMin.scheduling["schedule"][v[0]][k])
            try:
                assert tasks == v[1], f"Key: {k}, QMin: {qmin}"
            except Exception as exc:
                shutil.rmtree(os.path.join(os.getcwd(), "SAVE"))
                raise RuntimeError(exc)
    shutil.rmtree(os.path.join(os.getcwd(), "SAVE"))


@pytest.mark.skip()
def test_gettasks():
    tests = [
        (
            os.path.join(PATH, "inputs/molcas/tasks/QM5.in"),
            os.path.join(PATH, "inputs/molcas/tasks/template_casscf"),
            {
                "master": (
                    0,
                    [
                        ["gateway"],
                        ["seward"],
                        ["link", f"{os.getcwd()}/SAVE/MOLCAS.1.JobIph.0", "JOBOLD"],
                        ["rasscf", 1, 4, True, False],
                        ["copy", "MOLCAS.JobIph", "MOLCAS.1.JobIph"],
                        ["link", f"{os.getcwd()}/SAVE/MOLCAS.2.JobIph.0", "JOBOLD"],
                        ["rasscf", 2, 2, True, False],
                        ["copy", "MOLCAS.JobIph", "MOLCAS.2.JobIph"],
                        ["link", f"{os.getcwd()}/SAVE/MOLCAS.1.JobIph.0", "JOB001"],
                        ["link", "MOLCAS.1.JobIph", "JOB002"],
                        ["rassi", "overlap", [4, 4]],
                        ["link", f"{os.getcwd()}/SAVE/MOLCAS.2.JobIph.0", "JOB001"],
                        ["link", "MOLCAS.2.JobIph", "JOB002"],
                        ["rassi", "overlap", [2, 2]],
                        ["link", "MOLCAS.1.JobIph", "JOB001"],
                        ["link", "MOLCAS.2.JobIph", "JOB002"],
                        ["rassi", "soc", [4, 2]],
                    ],
                ),
                "grad_1_1": (
                    1,
                    [
                        ["link", "MOLCAS.1.JobIph", "JOBOLD"],
                        ["rasscf", 1, 4, True, False],
                        ["mclr", 0.0001, "sala=1"],
                        ["alaska"],
                    ],
                ),
            },
        ),
        (
            os.path.join(PATH, "inputs/molcas/tasks/QM5.in"),
            os.path.join(PATH, "inputs/molcas/tasks/template_xmscaspt2"),
            {
                "master": (
                    0,
                    [
                        ["gateway"],
                        ["seward"],
                        ["link", f"{os.getcwd()}/SAVE/MOLCAS.1.JobIph.0", "JOBOLD"],
                        ["rasscf", 1, 4, True, False],
                        ["caspt2", 1, 4, "xms-caspt2"],
                        ["copy", "MOLCAS.JobMix", "MOLCAS.1.JobIph"],
                        ["link", f"{os.getcwd()}/SAVE/MOLCAS.2.JobIph.0", "JOBOLD"],
                        ["rasscf", 2, 2, True, False],
                        ["caspt2", 2, 2, "xms-caspt2"],
                        ["copy", "MOLCAS.JobMix", "MOLCAS.2.JobIph"],
                        ["link", f"{os.getcwd()}/SAVE/MOLCAS.1.JobIph.0", "JOB001"],
                        ["link", "MOLCAS.1.JobIph", "JOB002"],
                        ["rassi", "overlap", [4, 4]],
                        ["link", f"{os.getcwd()}/SAVE/MOLCAS.2.JobIph.0", "JOB001"],
                        ["link", "MOLCAS.2.JobIph", "JOB002"],
                        ["rassi", "overlap", [2, 2]],
                        ["link", "MOLCAS.1.JobIph", "JOB001"],
                        ["link", "MOLCAS.2.JobIph", "JOB002"],
                        ["rassi", "soc", [4, 2]],
                    ],
                ),
                "grad_1_1": (
                    1,
                    [
                        ["link", "MOLCAS.1.JobIph", "JOBOLD"],
                        ["rasscf", 1, 4, True, False],
                        ["caspt2", 1, 4, "xms-caspt2", "GRDT\nrlxroot = 1"],
                        ["mclr", 0.0001],
                        ["alaska"],
                    ],
                ),
            },
        ),
    ]

    with open(f"{os.path.abspath(os.curdir)}/SAVE/STEP", "w", encoding="utf-8") as file:
        file.write("0")
    for qmin, templ, ref in tests:
        test_interface = SHARC_MOLCAS()
        test_interface.setup_mol(qmin)
        test_interface.read_template(templ)
        test_interface._read_resources = True
        test_interface.QMin.save["step"] = 0
        test_interface.read_requests(qmin)
        test_interface.setup_interface()
        test_interface.QMin.scheduling["schedule"] = test_interface._generate_schedule()
        for k, v in ref.items():
            tasks = test_interface._gen_tasklist(test_interface.QMin.scheduling["schedule"][v[0]][k])
            try:
                assert tasks == v[1], f"Key: {k}, QMin: {qmin}"
            except Exception as exc:
                # shutil.rmtree(os.path.join(os.getcwd(), "SAVE"))
                raise exc
    shutil.rmtree(os.path.join(os.getcwd(), "SAVE"))


def test_write_geom():
    tests = [
        (
            os.path.join(PATH, "inputs/molcas/geoms/QM1.in"),
            "3\n\nS1  0.000000 0.000000 0.000841\nH2  -0.000000 0.974799 1.182934\nH3  -0.000000 -0.974799 1.182934\n",
        ),
        (
            os.path.join(PATH, "inputs/molcas/geoms/QM2.in"),
            "10\n\nS1  0.000000 0.000000 0.000000\nC2  0.000000 0.000000 -1.587500\nH3  0.952500 0.000000 -2.116700\nC4  -1.267509 0.000000 -2.291716\nH5  -1.834145 -0.889165 -2.019263\nH6  -1.834145 0.889165 -2.019263\nC7  -1.026071 0.000000 -3.721474\nH8  -1.978014 0.000000 -4.250365\nH9  -0.459435 0.889165 -3.993927\nH10  -0.459435 -0.889165 -3.993927\n",
        ),
        (
            os.path.join(PATH, "inputs/molcas/geoms/QM3.in"),
            "2\n\nI1  0.000000 0.000000 0.000000\nBr2  3.230431 0.000000 0.000000\n",
        ),
        (
            os.path.join(PATH, "inputs/molcas/geoms/QM4.in"),
            "97\n\nC1  3.069079 0.978249 6.081524\nC2  2.262218 0.424504 5.087449\nN3  1.063849 0.973853 4.770988\nC4  0.647373 2.065730 5.426430\nC5  1.403004 2.666685 6.424552\nC6  2.636466 2.112766 6.758791\nC7  2.622002 -0.769724 4.309934\nN8  1.683199 -1.176891 3.421728\nC9  1.917820 -2.257839 2.665695\nC10  3.095337 -2.988177 2.757550\nC11  4.068672 -2.579680 3.666237\nC12  3.827710 -1.456961 4.450212\nRu13  0.000003 0.000001 3.318321\nN14  -1.063846 -0.973850 4.770987\nC15  -2.262218 -0.424504 5.087441\nC16  -3.069084 -0.978252 6.081509\nC17  -2.636473 -2.112769 6.758778\nC18  -1.403007 -2.666684 6.424546\nC19  -0.647372 -2.065727 5.426430\nC20  -2.622000 0.769724 4.309924\nN21  -1.683194 1.176892 3.421722\nC22  -1.917812 2.257840 2.665687\nC23  -3.095332 2.988175 2.757537\nC24  -4.068671 2.579676 3.666219\nC25  -3.827710 1.456958 4.450195\nN26  0.898511 0.998075 1.728900\nC27  0.482076 0.527305 0.536316\nC28  0.948561 1.035115 -0.677736\nC29  1.902076 2.073895 -0.604914\nN30  2.311982 2.519221 0.589379\nC31  1.811327 1.982375 1.689728\nC32  0.475555 0.518390 -1.917957\nC33  -0.475549 -0.518394 -1.917957\nC34  -0.948554 -1.035117 -0.677735\nC35  -0.482067 -0.527307 0.536317\nC36  -1.902070 -2.073897 -0.604911\nC37  -2.409116 -2.628458 -1.846036\nC38  -1.933453 -2.112008 -3.089889\nC39  -0.962307 -1.050816 -3.127001\nC40  -2.422463 -2.650269 -4.306104\nC41  -3.370463 -3.684855 -4.241115\nC42  -3.829419 -4.181363 -3.025138\nC43  -3.353062 -3.658424 -1.828523\nC44  -0.481150 -0.526059 -4.351055\nC45  -0.970832 -1.064924 -5.589904\nC46  -1.931406 -2.118651 -5.578132\nC47  0.962309 1.050814 -3.127002\nC48  0.481149 0.526059 -4.351055\nC49  0.970828 1.064925 -5.589905\nC50  0.491153 0.542230 -6.822505\nC51  -0.491161 -0.542227 -6.822504\nC52  -0.978202 -1.087695 -8.017319\nC53  -1.913018 -2.115033 -8.005229\nC54  -2.385873 -2.624505 -6.802486\nC55  2.409119 2.628457 -1.846039\nC56  1.933455 2.112007 -3.089892\nC57  2.422462 2.650269 -4.306107\nC58  1.931402 2.118653 -5.578134\nN59  -2.311975 -2.519223 0.589382\nC60  -1.811319 -1.982376 1.689731\nN61  -0.898503 -0.998076 1.728901\nC62  3.370462 3.684855 -4.241119\nC63  3.829420 4.181363 -3.025143\nC64  3.353065 3.658423 -1.828527\nC65  0.978191 1.087700 -8.017321\nC66  1.913007 2.115039 -8.005231\nC67  2.385865 2.624508 -6.802489\nH68  -4.565073 -4.983562 -3.015360\nH69  2.278225 2.524075 -8.945705\nH70  -2.278239 -2.524068 -8.945702\nH71  4.565074 4.983562 -3.015366\nH72  -3.254481 -2.555683 7.537408\nH73  -5.004682 3.126174 3.764056\nH74  3.254470 2.555678 7.537425\nH75  5.004682 -3.126179 3.764079\nH76  -0.319615 2.468772 5.136400\nH77  1.020786 3.552746 6.926073\nH78  4.025367 0.525320 6.328989\nH79  1.133291 -2.545707 1.970204\nH80  3.237443 -3.859422 2.122374\nH81  4.574323 -1.117230 5.162569\nH82  -4.025375 -0.525325 6.328969\nH83  -1.020790 -3.552745 6.926069\nH84  0.319619 -2.468766 5.136406\nH85  -4.574326 1.117225 5.162549\nH86  -3.237436 3.859420 2.122360\nH87  -1.133281 2.545710 1.970201\nH88  2.155825 2.366181 2.649657\nH89  -2.155817 -2.366182 2.649660\nH90  -3.700308 -4.034694 -0.868713\nH91  -3.767044 -4.120775 -5.153844\nH92  3.700313 4.034693 -0.868717\nH93  3.767041 4.120777 -5.153849\nH94  0.629926 0.715147 -8.976225\nH95  3.117363 3.426986 -6.831335\nH96  -3.117371 -3.426983 -6.831331\nH97  -0.629940 -0.715141 -8.976225\n",
        ),
    ]

    for geom, ref in tests:
        test_interface = SHARC_MOLCAS()
        test_interface.setup_mol(geom)
        test_interface.set_coords(geom)
        assert test_interface._write_geom(test_interface.QMin.molecule["elements"], test_interface.QMin.coords["coords"]) == ref


@pytest.mark.skip()
def test_write_input():
    tests = [
        (
            os.path.join(PATH, "inputs/molcas/inputfile/QM.in"),
            os.path.join(PATH, "inputs/molcas/inputfile/template_casscf"),
            (
                f"&GATEWAY\nCOORD=MOLCAS.xyz\nGROUP=NOSYM\nBASIS=cc-pvdz\nAMFI\nangmom\n0 0 0\nBASLIB\n{os.getcwd()}/basisset\n\nRICD\nCDTHreshold=0.0001\n\n&SEWARD\nDOANA\n\n&RASSCF\nSPIN=1\nNACTEL=8 0 0\nINACTIVE=5\nRAS2=6\nITERATIONS=200,100\nCIROOT=4 5; 1 2 3 4 ;1 1 1 1 \nORBLISTING=NOTHING\nPRWF=0.1\nTHRS=1.0e-10 1.0e-06 1.0e-06\n\n>> COPY MOLCAS.JobIph MOLCAS.1.JobIph\n\n>> COPY MOLCAS.rasscf.molden MOLCAS.1.molden\n\n>> COPY MOLCAS.1.JobIph JOB001\n\n&RASSI\nNROFJOBIPHS\n1 4\n1 2 3 4\nMEIN\n\n&RASSCF\nSPIN=2\nNACTEL=7 0 0\nINACTIVE=5\nRAS2=6\nITERATIONS=200,100\nCIROOT=2 2 1\nORBLISTING=NOTHING\nPRWF=0.1\nTHRS=1.0e-10 1.0e-06 1.0e-06\n\n>> COPY MOLCAS.JobIph MOLCAS.2.JobIph\n\n>> COPY MOLCAS.rasscf.molden MOLCAS.2.molden\n\n>> COPY MOLCAS.2.JobIph JOB001\n\n&RASSI\nNROFJOBIPHS\n1 2\n1 2\nMEIN\n\n>> COPY MOLCAS.1.JobIph JOB001\n\n>> COPY MOLCAS.2.JobIph JOB002\n\n&RASSI\nNROFJOBIPHS\n2 4 2\n1 2 3 4\n1 2\nMEIN\nSPINORBIT\nSOCOUPLING=0.0d0\nEJOB\n\n",
                ">> COPY MOLCAS.1.JobIph JOBOLD\n\n>> COPY MOLCAS.1.JobIph JOBOLD\n\n&RASSCF\nSPIN=1\nNACTEL=8 0 0\nINACTIVE=5\nRAS2=6\nITERATIONS=200,100\nCIROOT=4 5; 1 2 3 4 ;1 1 1 1 \nORBLISTING=NOTHING\nPRWF=0.1\nTHRS=1.0e-10 1.0e-06 1.0e-06\nJOBIPH\n\n&MCLR\nTHRESHOLD=0.000100\nsala=1\n\n&ALASKA\n\n>> COPY MOLCAS.2.JobIph JOBOLD\n\n",
                ">> COPY MOLCAS.1.JobIph JOBOLD\n\n>> COPY MOLCAS.1.JobIph JOBOLD\n\n&RASSCF\nSPIN=1\nNACTEL=8 0 0\nINACTIVE=5\nRAS2=6\nITERATIONS=200,100\nCIROOT=4 5; 1 2 3 4 ;1 1 1 1 \nORBLISTING=NOTHING\nPRWF=0.1\nTHRS=1.0e-10 1.0e-06 1.0e-06\nJOBIPH\n\n&MCLR\nTHRESHOLD=0.000100\nnac=1 2\n\n&ALASKA\n\n>> COPY MOLCAS.1.JobIph JOB001\n\n>> COPY MOLCAS.JobIph JOB002\n\n>> COPY MOLCAS.1.JobIph JOB002\n\n&RASSI\nNROFJOBIPHS\n2 4 4\n1 2 3 4\n1 2 3 4\nMEIN\nOVERLAPS\n\n>> COPY MOLCAS.2.JobIph JOBOLD\n\n>> COPY MOLCAS.2.JobIph JOB001\n\n>> COPY MOLCAS.JobIph JOB002\n\n>> COPY MOLCAS.2.JobIph JOB002\n\n&RASSI\nNROFJOBIPHS\n2 2 2\n1 2\n1 2\nMEIN\nOVERLAPS\n\n",
            ),
        )
    ]
    for qmin, template, ref in tests:
        test_interface = SHARC_MOLCAS()
        test_interface.setup_mol(qmin)
        test_interface.read_template(template)
        test_interface._read_resources = True
        test_interface.setup_interface()
        test_interface.read_requests(qmin)
        test_interface.QMin.scheduling["schedule"] = test_interface._generate_schedule()
        master = test_interface.QMin.scheduling["schedule"][0]["master"]
        assert test_interface._write_input(test_interface._gen_tasklist(master), master) == ref[0]
        grad = test_interface.QMin.scheduling["schedule"][1]["grad_1_1"]
        # print(test_interface._write_input(test_interface._gen_tasklist(grad),grad))
        assert test_interface._write_input(test_interface._gen_tasklist(grad), grad) == ref[1]
        nac = test_interface.QMin.scheduling["schedule"][1]["nacdr_1_1_1_2"]


def test_get_energy():
    tests = [
        (
            os.path.join(PATH, "inputs/molcas/tasks/QM1.in"),
            "casscf",
            os.path.join(PATH, "inputs/molcas/output/hdmsoc420casscf"),
        ),
        (
            os.path.join(PATH, "inputs/molcas/tasks/QM1.in"),
            "caspt2",
            os.path.join(PATH, "inputs/molcas/output/hdmsoc420caspt2"),
        ),
        (
            os.path.join(PATH, "inputs/molcas/tasks/QM1.in"),
            "cms-pdft",
            os.path.join(PATH, "inputs/molcas/output/hdmsoc420cmspdft"),
        ),
        (
            os.path.join(PATH, "inputs/molcas/tasks/QM1.in"),
            "ms-caspt2",
            os.path.join(PATH, "inputs/molcas/output/hdmsoc420mscaspt2"),
        ),
        (
            os.path.join(PATH, "inputs/molcas/tasks/QM1.in"),
            "xms-caspt2",
            os.path.join(PATH, "inputs/molcas/output/hdmsoc420xmscaspt2"),
        ),
    ]

    for qmin, method, output in tests:
        test_interface = SHARC_MOLCAS()
        test_interface.setup_mol(qmin)
        test_interface.QMin.template["method"] = method

        hdf = h5py.File(f"{output}.h5", "r")
        with open(f"{output}.out", "r", encoding="utf-8") as ascii_out:
            assert np.allclose(test_interface._get_energy(hdf), test_interface._get_energy(ascii_out.read()))


def test_get_socs():
    tests = [
        (
            os.path.join(PATH, "inputs/molcas/tasks/QM1.in"),
            "casscf",
            os.path.join(PATH, "inputs/molcas/output/hdmsoc420casscf"),
        ),
        (
            os.path.join(PATH, "inputs/molcas/tasks/QM1.in"),
            "caspt2",
            os.path.join(PATH, "inputs/molcas/output/hdmsoc420caspt2"),
        ),
        (
            os.path.join(PATH, "inputs/molcas/tasks/QM1.in"),
            "cms-pdft",
            os.path.join(PATH, "inputs/molcas/output/hdmsoc420cmspdft"),
        ),
        (
            os.path.join(PATH, "inputs/molcas/tasks/QM1.in"),
            "ms-caspt2",
            os.path.join(PATH, "inputs/molcas/output/hdmsoc420mscaspt2"),
        ),
        (
            os.path.join(PATH, "inputs/molcas/tasks/QM1.in"),
            "xms-caspt2",
            os.path.join(PATH, "inputs/molcas/output/hdmsoc420xmscaspt2"),
        ),
    ]

    for qmin, method, output in tests:
        test_interface = SHARC_MOLCAS()
        test_interface.setup_mol(qmin)
        test_interface.QMin.template["method"] = method
        test_interface.setup_interface()

        hdf = h5py.File(f"{output}.h5", "r")
        with open(f"{output}.out", "r", encoding="utf-8") as ascii_out:
            ref_ascii = test_interface._get_socs(ascii_out.read())
            ref_hdf = test_interface._get_socs(hdf)
            np.einsum("ii->i", ref_hdf)[:] = 0.0
            assert np.allclose(ref_ascii, ref_hdf)


def test_get_grad():
    tests = [
        (
            os.path.join(PATH, "inputs/molcas/output/grad1"),
            np.array(
                [
                    [-1.93061522031230e-14, 2.07963771793602e-03, -7.58071656304145e-02],
                    [-4.92596610394539e-14, 1.68703340325114e-02, 3.66415199618992e-02],
                    [6.85658132425765e-14, -1.89499717504480e-02, 3.91656456685157e-02],
                ]
            ),
        ),
        (
            os.path.join(PATH, "inputs/molcas/output/grad2"),
            np.array(
                [
                    [-1.93061522031230e+14, 2.07963771793602e-03, -7.58071656304145e-02],
                    [-4.92596610394539e-14, 1.68703340325114e-02, 3.66415199618992e-02],
                    [6.85658132425765e-14, -1.89499717504480e-02, 3.91656456685157e-02],
                ]
            ),
        ),
        (
            os.path.join(PATH, "inputs/molcas/output/grad3"),
            np.array(
                [
                    [-1.93061522031230e-14, 2.07963771793602e-03, -7.58071656304145e-02],
                    [-4.92596610394539e-14, 1.68703340325114e-02, 3.66415199618992e-02],
                    [-4.92596610394539e-14, 1.68703340325114e-02, 3.66415199618992e-02],
                    [6.85658132425765e-14, -1.89499717504480e-02, 3.91656456685157e-02],
                    [-4.92596610394539e-14, 1.68703340325114e-02, 3.66415199618992e-02],
                    [-4.92596610394539e-14, 1.68703340325114e-02, 3.66415199618992e-02],
                ]
            ),
        )
    ]

    for grad, ref in tests:
        test_interface = SHARC_MOLCAS()
        with open(grad, "r", encoding="utf-8") as file:
            assert np.allclose(test_interface._get_grad(file.read()), ref)
