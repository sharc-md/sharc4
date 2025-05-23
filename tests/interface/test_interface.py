import pytest
import os
from utils import expand_path
from SHARC_INTERFACE import SHARC_INTERFACE
import shutil

SHARC_INTERFACE.__abstractmethods__ = set()

PATH = "$SHARC/../tests/interface"


def get_states(path: str, states: list):
    test_interface = SHARC_INTERFACE()

    test_interface.setup_mol(path)
    assert test_interface.QMin.molecule["states"] == states


def set_requests(path: str, requests: dict):
    test_interface = SHARC_INTERFACE()
    test_interface.setup_mol(path)
    test_interface._read_template = True
    test_interface._read_resources = True
    test_interface.read_requests(path)
    shutil.rmtree(os.path.join(os.getcwd(), "SAVE"))
    for k, v in requests.items():
        assert test_interface.QMin.requests[k] == v, ValueError(f"{k}")


def read_resources(path: str, params: dict, whitelist: list):
    test_interface = SHARC_INTERFACE()
    test_interface.QMin.resources.types = {
        "int_key": int,
        "float_key": float,
        "key1": str,
        "key2": list,
        "key4": bool,
        "key5": list,
        "key6": dict,
    }
    test_interface.QMin.resources.data = {
        "int_key": None,
        "float_key": None,
        "key1": None,
        "key2": None,
        "key4": None,
        "key5": None,
        "key6": {},
    }
    test_interface._setup_mol = True
    test_interface.read_resources(path, whitelist)
    for k, v in params.items():
        assert test_interface.QMin.resources[k] == v


def test_states1():
    tests = [("inputs/QM1.in", [3, 1, 5]), ("inputs/QM3.in", [0, 0, 0, 0, 9, 9])]
    for path, state in tests:
        get_states(os.path.join(expand_path(PATH), path), state)


def test_states2():
    tests = [("inputs/QM_failstate1.in", []), ("inputs/QM_failstate2.in", []), ("inputs/QM2.in", [])]
    for path, state in tests:
        with pytest.raises(ValueError):
            get_states(os.path.join(expand_path(PATH), path), state)


def test_requests1():
    tests = [
        (
            "inputs/QM1.in",
            {
                "h": True,
                "soc": True,
                "dm": True,
                "grad": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
                "nacdr": None,
                "overlap": False,
                "phases": False,
                "ion": True,
                "socdr": False,
                "dmdr": False,
                "multipolar_fit": None,
                "theodore": True,
            },
        ),
        (
            "inputs/QM3.in",
            {
                "h": True,
                "soc": False,
                "dm": True,
                "grad": list(range(1, 100)),
                "nacdr": ["all"],
                "overlap": False,
                "phases": False,
                "ion": False,
                "socdr": True,
                "dmdr": False,
                "multipolar_fit": None,
                "theodore": False,
            },
        ),
        (
            "inputs/QM4.in",
            {
                "h": True,
                "soc": True,
                "dm": True,
                "grad": list(range(1, 21)),
                "nacdr": [[1, 2]],
                "overlap": False,
                "phases": False,
                "ion": True,
                "socdr": False,
                "dmdr": False,
                "multipolar_fit": None,
                "theodore": True,
            },
        ),
        (
            "inputs/QM0.in",
            {
                "h": True,
                "soc": True,
                "dm": True,
                "grad": [1, 2],
                "nacdr": None,
                "overlap": False,
                "phases": False,
                "ion": True,
                "socdr": False,
                "dmdr": False,
                "multipolar_fit": None,
                "theodore": True,
            },
        ),
        (
            "inputs/QM8.in",
            {
                "h": True,
                "soc": True,
                "dm": True,
                "grad": [10],
                "nacdr": None,
                "overlap": False,
                "phases": False,
                "ion": True,
                "socdr": False,
                "dmdr": False,
                "multipolar_fit": None,
                "theodore": True,
            },
        ),
        (
            "inputs/QM9.in",
            {
                "h": True,
                "soc": True,
                "dm": True,
                "grad": None,
                "nacdr": [[1, 10]],
                "overlap": False,
                "phases": False,
                "ion": True,
                "socdr": False,
                "dmdr": False,
                "multipolar_fit": None,
                "theodore": True,
            },
        ),
        (
            "inputs/QM10.in",
            {
                "h": True,
                "soc": True,
                "dm": True,
                "grad": [1, 2, 3],
                "nacdr": [[1, 10]],
                "overlap": False,
                "phases": False,
                "ion": True,
                "socdr": False,
                "dmdr": False,
                "multipolar_fit": None,
                "theodore": True,
            },
        ),
    ]
    for path, req in tests:
        set_requests(os.path.join(expand_path(PATH), path), req)


def test_requests2():
    tests = [
        ("inputs/QM_failreq1.in", {}),
        ("inputs/QM_failreq2.in", {}),
        ("inputs/QM_failreq3.in", {}),
        ("inputs/QM2.in", {}),
        ("inputs/QM_failreq4.in", {}),
        ("inputs/QM_failreq5.in", {}),
        ("inputs/QM_failreq6.in", {}),
    ]

    for path, req in tests:
        with pytest.raises((AssertionError, ValueError, RuntimeError)):
            set_requests(os.path.join(expand_path(PATH), path), req)


def test_driver_requests():
    tests = [
        (
            os.path.join(expand_path(PATH), "inputs/QM5.in"),
            {"tasks": "step  0 SOC DM", "grad": "all", "nacdr": ""},
            {"h": True, "soc": True, "dm": True, "grad": [1, 2, 3, 4, 5, 6], "nacdr": None, "overlap": False, "phases": False},
            0,
        ),
        (
            os.path.join(expand_path(PATH), "inputs/QM5.in"),
            {"tasks": "step  1 SOC DM OVERLAP PHASES", "grad": "all", "nacdr": ""},
            {"h": True, "soc": True, "dm": True, "grad": [1, 2, 3, 4, 5, 6], "nacdr": None, "overlap": True, "phases": True},
            1,
        ),
        (
            os.path.join(expand_path(PATH), "inputs/QM5.in"),
            {"tasks": "step  0 SOC DM", "grad": "all", "nacdr": "NACDR"},
            {
                "h": True,
                "soc": True,
                "dm": True,
                "grad": [1, 2, 3, 4, 5, 6],
                "nacdr": [
                    (1, 1),
                    (1, 2),
                    (1, 3),
                    (1, 4),
                    (1, 5),
                    (1, 6),
                    (1, 7),
                    (1, 8),
                    (1, 9),
                    (1, 10),
                    (1, 11),
                    (1, 12),
                    (2, 1),
                    (2, 2),
                    (2, 3),
                    (2, 4),
                    (2, 5),
                    (2, 6),
                    (2, 7),
                    (2, 8),
                    (2, 9),
                    (2, 10),
                    (2, 11),
                    (2, 12),
                    (3, 1),
                    (3, 2),
                    (3, 3),
                    (3, 4),
                    (3, 5),
                    (3, 6),
                    (3, 7),
                    (3, 8),
                    (3, 9),
                    (3, 10),
                    (3, 11),
                    (3, 12),
                    (4, 1),
                    (4, 2),
                    (4, 3),
                    (4, 4),
                    (4, 5),
                    (4, 6),
                    (4, 7),
                    (4, 8),
                    (4, 9),
                    (4, 10),
                    (4, 11),
                    (4, 12),
                    (5, 1),
                    (5, 2),
                    (5, 3),
                    (5, 4),
                    (5, 5),
                    (5, 6),
                    (5, 7),
                    (5, 8),
                    (5, 9),
                    (5, 10),
                    (5, 11),
                    (5, 12),
                    (6, 1),
                    (6, 2),
                    (6, 3),
                    (6, 4),
                    (6, 5),
                    (6, 6),
                    (6, 7),
                    (6, 8),
                    (6, 9),
                    (6, 10),
                    (6, 11),
                    (6, 12),
                    (7, 1),
                    (7, 2),
                    (7, 3),
                    (7, 4),
                    (7, 5),
                    (7, 6),
                    (7, 7),
                    (7, 8),
                    (7, 9),
                    (7, 10),
                    (7, 11),
                    (7, 12),
                    (8, 1),
                    (8, 2),
                    (8, 3),
                    (8, 4),
                    (8, 5),
                    (8, 6),
                    (8, 7),
                    (8, 8),
                    (8, 9),
                    (8, 10),
                    (8, 11),
                    (8, 12),
                    (9, 1),
                    (9, 2),
                    (9, 3),
                    (9, 4),
                    (9, 5),
                    (9, 6),
                    (9, 7),
                    (9, 8),
                    (9, 9),
                    (9, 10),
                    (9, 11),
                    (9, 12),
                    (10, 1),
                    (10, 2),
                    (10, 3),
                    (10, 4),
                    (10, 5),
                    (10, 6),
                    (10, 7),
                    (10, 8),
                    (10, 9),
                    (10, 10),
                    (10, 11),
                    (10, 12),
                    (11, 1),
                    (11, 2),
                    (11, 3),
                    (11, 4),
                    (11, 5),
                    (11, 6),
                    (11, 7),
                    (11, 8),
                    (11, 9),
                    (11, 10),
                    (11, 11),
                    (11, 12),
                    (12, 1),
                    (12, 2),
                    (12, 3),
                    (12, 4),
                    (12, 5),
                    (12, 6),
                    (12, 7),
                    (12, 8),
                    (12, 9),
                    (12, 10),
                    (12, 11),
                    (12, 12),
                ],
                "overlap": False,
                "phases": False,
            },
            0,
        ),
    ]
    for qmin, tasks, ref, step in tests:
        with open("SAVE/STEP", "w", encoding="utf-8") as file:
            file.write(str(step))
        test_interface = SHARC_INTERFACE()
        test_interface.setup_mol(qmin)
        # test_interface.QMin.save["step"] = 1
        test_interface._set_driver_requests(tasks)
        for k, v in ref.items():
            assert test_interface.QMin.requests[k] == v
    shutil.rmtree(os.path.join(os.getcwd(), "SAVE"))


def test_driver_requests2():
    tests = [
        (
            os.path.join(expand_path(PATH), "inputs/request_test1.in"),
            {"h": True, "soc": True, "dm": True, "grad": [1, 2, 3, 4, 5, 6]},
        ),
        (
            os.path.join(expand_path(PATH), "inputs/request_test2.in"),
            {"h": True, "grad": [1, 2, 3], "theodore": True},
        ),
        (
            os.path.join(expand_path(PATH), "inputs/request_test3.in"),
            {"h": True, "grad": [1, 2, 3], "multipolar_fit": ["all"]},
        ),
        (
            os.path.join(expand_path(PATH), "inputs/request_test4.in"),
            {"h": True, "grad": [1, 2, 3], "density_matrices": ["all"]},
        )
    ]

    for qmfile, qmdict in tests:
        file_test = SHARC_INTERFACE()
        dict_test = SHARC_INTERFACE()

        file_test.setup_mol(qmfile)
        dict_test.setup_mol(qmfile)

        file_test._read_template = True
        dict_test._read_template = True
        file_test._read_resources = True
        dict_test._read_resources = True

        file_test.read_requests(qmfile)
        dict_test.read_requests(qmdict)

        for (key, val_file), val_dict in zip(file_test.QMin.requests.items(), dict_test.QMin.requests.values()):
            assert val_file == val_dict, key


def test_resources1():
    tests = [
        ("inputs/interface_resources1", {"key1": "test", "key2": ["test1", "test2"], "key4": True}, []),
        ("inputs/interface_resources2", {"key1": "test2", "key2": [["test1", "test2"], ["test3", "test4"]]}, ["key2"]),
        ("inputs/interface_resources3", {"int_key": 13123, "float_key": -3.0}, []),
        ("inputs/interface_resources4", {"key1": "test2", "key2": ["test4"]}, []),
        ("inputs/interface_resources5", {"key1": "test1", "key2": ["test3", "test4"]}, []),
        ("inputs/interface_resources6", {"key1": "test2", "key2": [["test1"], ["test2"], ["test3"], ["test4"]]}, ["key2"]),
        ("inputs/interface_resources7", {"key1": "test2", "key2": [["test1", "test2"], ["test3", "test4"]]}, []),
        ("inputs/interface_resources8", {"key1": "test2", "key2": ["test1", "test2", "test3", "test4"]}, []),
        ("inputs/interface_resources10", {"key2": [[1, 2], ["3"]], "key5": [[1], [2]]}, ["key2"]),
        ("inputs/interface_resources11", {"key6": {"Fe": "1.81", "C": "1.2", "N": "2.0"}}, []),
        ("inputs/interface_resources12", {"key6": {"Fe": "1.81", "C": "1.2", "N": "2.0"}}, []),
    ]
    for path, params, whitelist in tests:
        read_resources(os.path.join(expand_path(PATH), path), params, whitelist)


def test_resources2():
    tests = [
        ("inputs/interface_resources9", {}, []),
    ]
    for path, params, whitelist in tests:
        with pytest.raises(ValueError):
            read_resources(os.path.join(expand_path(PATH), path), params, whitelist)


def test_save_resources():
    tests = [
        ("inputs/interface/save_resources1", {"always_orb_init": False, "always_guess": False}),
        ("inputs/interface/save_resources2", {"always_orb_init": True, "always_guess": True}),
        ("inputs/interface/save_resources3", {"always_orb_init": False, "always_guess": True}),
        ("inputs/interface/save_resources4", {"always_orb_init": True, "always_guess": False}),
    ]

    for input, ref in tests:
        test_interface = SHARC_INTERFACE()
        test_interface._setup_mol = True
        test_interface.read_resources(os.path.join(expand_path(PATH), input))
        for key, val in ref.items():
            assert test_interface.QMin.save[key] == val
