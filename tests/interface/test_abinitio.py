import os
import shutil

import numpy as np
import pytest
from SHARC_ABINITIO import SHARC_ABINITIO
from utils import expand_path

SHARC_ABINITIO.__abstractmethods__ = set()

PATH = expand_path("$SHARC/../tests/interface")


def setup_interface(qmin: str, template: str, maps: dict):
    test = SHARC_ABINITIO()
    test.setup_mol(qmin)
    test.read_template(template)
    test._read_resources = True
    test.QMin.resources["wfoverlap"] = ""
    test.read_requests(qmin)
    test.setup_interface()
    print(test.QMin.molecule["charge"])

    for k, v in maps.items():
        assert test.QMin.maps[k] == v, ValueError(f"{test.QMin.maps[k]} {k}")


def test_setupinterface1():
    tests = [
        (
            "inputs/QM1.in",
            "inputs/abinitio_template1",
            {
                "gradmap": {(1, 2), (2, 1), (3, 4), (3, 1), (1, 1), (3, 3), (3, 2), (1, 3), (3, 5)},
                "chargemap": {1: 0, 2: 1, 3: 0},
                "nacmap": None,
            },
        ),
        (
            "inputs/QM1c.in",
            "inputs/abinitio_template2",
            {
                "gradmap": {(1, 2), (2, 1), (3, 4), (3, 1), (1, 1), (3, 3), (3, 2), (1, 3), (3, 5)},
                "chargemap": {1: 0, 2: -1, 3: 0},
                "nacmap": None,
            },
        ),
        (
            "inputs/QM3.in",
            "inputs/abinitio_template3",
            {
                "gradmap": {
                    (6, 1),
                    (5, 4),
                    (5, 1),
                    (5, 7),
                    (6, 4),
                    (6, 7),
                    (5, 2),
                    (6, 2),
                    (5, 5),
                    (6, 5),
                    (5, 8),
                    (6, 8),
                    (5, 3),
                    (5, 6),
                    (6, 6),
                    (5, 9),
                    (6, 3),
                    (6, 9),
                },
                "chargemap": {1: 0, 2: 1, 3: 0, 4: 1, 5: 0, 6: 1},
                "nacmap": {
                    (5, 1, 5, 4),
                    (6, 3, 6, 4),
                    (5, 6, 5, 8),
                    (6, 7, 6, 9),
                    (6, 3, 6, 7),
                    (6, 2, 6, 5),
                    (5, 3, 5, 4),
                    (5, 5, 5, 6),
                    (5, 3, 5, 7),
                    (6, 2, 6, 8),
                    (6, 6, 6, 7),
                    (5, 5, 5, 9),
                    (5, 2, 5, 5),
                    (5, 4, 5, 7),
                    (5, 2, 5, 8),
                    (6, 1, 6, 5),
                    (6, 1, 6, 2),
                    (6, 1, 6, 8),
                    (6, 5, 6, 7),
                    (6, 4, 6, 5),
                    (6, 4, 6, 8),
                    (6, 8, 6, 9),
                    (5, 1, 5, 3),
                    (5, 1, 5, 9),
                    (5, 7, 5, 9),
                    (5, 1, 5, 6),
                    (5, 6, 5, 7),
                    (6, 3, 6, 6),
                    (6, 3, 6, 9),
                    (6, 7, 6, 8),
                    (5, 3, 5, 6),
                    (6, 2, 6, 4),
                    (5, 5, 5, 8),
                    (5, 3, 5, 9),
                    (6, 6, 6, 9),
                    (6, 2, 6, 7),
                    (5, 2, 5, 4),
                    (6, 5, 6, 6),
                    (5, 4, 5, 6),
                    (5, 2, 5, 7),
                    (6, 1, 6, 4),
                    (6, 5, 6, 9),
                    (5, 4, 5, 9),
                    (6, 1, 6, 7),
                    (6, 4, 6, 7),
                    (5, 1, 5, 2),
                    (5, 7, 5, 8),
                    (5, 1, 5, 5),
                    (5, 1, 5, 8),
                    (5, 6, 5, 9),
                    (6, 3, 6, 5),
                    (6, 2, 6, 3),
                    (6, 3, 6, 8),
                    (6, 6, 6, 8),
                    (6, 2, 6, 6),
                    (5, 3, 5, 5),
                    (5, 5, 5, 7),
                    (5, 2, 5, 3),
                    (5, 3, 5, 8),
                    (6, 2, 6, 9),
                    (5, 4, 5, 5),
                    (5, 2, 5, 6),
                    (6, 1, 6, 3),
                    (6, 5, 6, 8),
                    (5, 4, 5, 8),
                    (5, 2, 5, 9),
                    (5, 8, 5, 9),
                    (6, 1, 6, 6),
                    (6, 1, 6, 9),
                    (6, 4, 6, 9),
                    (6, 4, 6, 6),
                    (5, 1, 5, 7),
                },
            },
        ),
        (
            "inputs/QM5.in",
            "inputs/abinitio_template1",
            {
                "gradmap": {(1, 2), (3, 3), (3, 1), (1, 1), (1, 3), (3, 2)},
                "chargemap": {1: 0, 2: 1, 3: 0},
                "nacmap": {(1, 1, 1, 3), (3, 2, 3, 3), (3, 1, 3, 3), (1, 1, 1, 2), (1, 2, 1, 3), (3, 1, 3, 2)},
            },
        ),
        (
            "inputs/QM6.in",
            "inputs/abinitio_template1",
            {
                "gradmap": {(1, 2), (2, 1), (3, 4), (3, 1), (1, 1), (3, 3), (3, 2), (1, 3), (3, 5)},
                "chargemap": {1: 0, 2: 1, 3: 0},
                "nacmap": set(),
            },
        ),
    ]

    for qmin, template, maps in tests:
        setup_interface(os.path.join(PATH, qmin), os.path.join(PATH, template), maps)


def test_setupinterface2():
    tests = [
        ("inputs/QM3f.in", "inputs/abinitio_template1", {}),
        ("inputs/QM2.in", "inputs/abinitio_template1", {"gradmap": None, "chargemap": {}, "nacmap": set()}),
    ]

    for qmin, template, maps in tests:
        with pytest.raises(ValueError):
            setup_interface(os.path.join(PATH, qmin), os.path.join(PATH, template), maps)


def test_clean_savedir():
    tmp_dir = os.path.join(PATH, "savedir_test")

    tests = [
        (["dfgdb.brebr.4", "dfgdb.3.5", "5555.7"], 3, 8, ["dfgdb.3.5", "5555.7"]),
        (["QQMM.gbw.4", "dfgdb.3.5", "5555.7"], -1, 8, ["QQMM.gbw.4", "dfgdb.3.5", "5555.7"]),
        (["QQMM.gbw.4", "dfgdb.3.5", "5555.7"], 1, 8, ["5555.7"]),
        (["QQMM.gbw.4", "dfgdb.3.5", "5555.7"], 1, 9, []),
        (["rrrtnrnr.log", "dfgdb.brebr.4", "dfgdb.3.5", "5555.7"], 3, 8, ["rrrtnrnr.log", "dfgdb.3.5", "5555.7"]),
    ]

    for files, retain, step, res in tests:
        # Create temp files
        os.mkdir(tmp_dir)
        for file in files:
            with open(os.path.join(tmp_dir, file), "a"):
                os.utime(os.path.join(tmp_dir, file))

        SHARC_ABINITIO.clean_savedir(tmp_dir, retain, step)
        try:
            assert os.listdir(tmp_dir) == res
        finally:
            shutil.rmtree(tmp_dir)


def test_parsedyson():
    tests = [
        (
            "inputs/dyson1",
            np.array(
                [
                    [0.9401441027, 0.8906030759, 0.9307425454, 0.0043114302],
                    [0.4511414880, 0.0234786094, 0.0009732790, 0.3944154081],
                    [0.0180273519, 0.4250909955, 0.0047639010, 0.7392838275],
                    [0.0546332129, 0.0858993467, 0.3193591589, 0.0014270452],
                ]
            ),
        ),
        (
            "inputs/dyson2",
            np.array(
                [
                    [0.8944273988, 0.0004782934, 0.0723134962, 0.0009021681],
                    [0.0896794811, 0.0007445378, 0.8279340441, 0.0003716648],
                    [0.0062383550, 0.9479518302, 0.0023316326, 0.0106589898],
                    [0.4213051340, 0.0041931949, 0.0339798806, 0.0002505716],
                ]
            ),
        ),
        ("inputs/dyson3", np.array([[0.9401441027]])),
        ("inputs/dyson4", np.array([[0.9401441027, 0.9307595123], [0.4511457142, 0.0009661697]])),
    ]

    test_interface = SHARC_ABINITIO()
    for wfovlp, ref in tests:
        assert np.allclose(test_interface.get_dyson(os.path.join(PATH, wfovlp)), ref)
