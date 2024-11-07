import os
import shutil

import pytest
from SHARC_ORCA import SHARC_ORCA
from utils import expand_path, mkdir
from constants import FROZENS

PATH = expand_path("$SHARC/../tests/interface")


def setup_interface(path: str, maps: dict):
    test_interface = SHARC_ORCA()
    test_interface.setup_mol(path)
    test_interface._read_resources = True
    test_interface._read_template = True
    test_interface.QMin.resources["wfoverlap"] = ""
    test_interface.read_requests(path)
    test_interface.setup_interface()
    for k, v in maps.items():
        assert test_interface.QMin.maps[k] == v, test_interface.QMin.maps[k]


def build_jobs(path: str, template: str, maps: dict):
    test_interface = SHARC_ORCA()
    test_interface.setup_mol(path)
    test_interface._read_resources = True
    test_interface.read_template(template)
    test_interface.read_requests(path)
    test_interface.setup_interface()
    for k, v in maps.items():
        assert test_interface.QMin.control[k] == v, test_interface.QMin.control[k]


def get_energy(outfile: str, template: str, qmin: str, mults: list, energies: dict, orcaver: tuple = (5, 0, 4)):
    test_interface = SHARC_ORCA()
    test_interface.setup_mol(qmin)
    test_interface._read_resources = True
    test_interface.QMin.resources["wfoverlap"] = ""
    test_interface.read_template(template)
    test_interface.setup_interface()
    test_interface.read_requests(qmin)
    test_interface.QMin.resources["orcaversion"] = orcaver
    with open(outfile, "r", encoding="utf-8") as file:
        parsed = test_interface._get_energy(file.read(), mults)
        for k, v in parsed.items():
            assert v == pytest.approx(energies[k])


@pytest.mark.dependency()
def test_orcaversion():
    test_interface = SHARC_ORCA()
    test_interface._setup_mol = True
    test_interface.read_resources(os.path.join(PATH, "inputs/orcapath"))
    assert isinstance(SHARC_ORCA.get_orca_version(test_interface.QMin.resources["orcadir"]), tuple)


def test_requests1():
    tests = ["inputs/QM2.in", "inputs/QM3.in", "inputs/QM4.in"]  # , "inputs/orca_requests_fail"]
    for i in tests:
        with pytest.raises(ValueError):
            test_interface = SHARC_ORCA()
            test_interface.setup_mol(os.path.join(PATH, i))
            test_interface._read_template = True
            test_interface._read_resources = True
            test_interface.read_requests(os.path.join(PATH, i))


def test_requests2():
    tests = ["inputs/orca_requests"]
    for i in tests:
        test_interface = SHARC_ORCA()
        test_interface.setup_mol(os.path.join(PATH, i))
        test_interface._read_template = True
        test_interface._read_resources = True
        test_interface.QMin.resources["wfoverlap"] = ""
        test_interface.read_requests(os.path.join(PATH, i))


def test_maps():
    tests = [
        (
            "inputs/QM0.in",
            {
                "multmap": {1: 1, 3: 1, -1: [1, 3], 2: 2, -2: [2]},
                "ionmap": [(1, 1, 2, 2), (2, 2, 3, 1)],
                "gsmap": {
                    1: 1,
                    2: 1,
                    3: 1,
                    4: 4,
                    5: 4,
                    6: 6,
                    7: 6,
                    8: 1,
                    9: 1,
                    10: 1,
                    11: 1,
                    12: 1,
                    13: 1,
                    14: 1,
                    15: 1,
                    16: 1,
                    17: 1,
                    18: 1,
                    19: 1,
                    20: 1,
                    21: 1,
                    22: 1,
                },
            },
        )
    ]

    for path, maps in tests:
        setup_interface(os.path.join(PATH, path), maps)


@pytest.mark.dependency(depends=["test_orcaversion"])
def test_resources():
    test_pass = ["inputs/orcapath"]
    test_fail = ["inputs/orcapath_fail", "inputs/orcapath_fail2"]

    for i in test_pass:
        test_interface = SHARC_ORCA()
        test_interface._setup_mol = True
        test_interface.read_resources(os.path.join(PATH, i))

    for i in test_fail:
        with pytest.raises(ValueError):
            test_interface = SHARC_ORCA()
            test_interface._setup_mol = True
            test_interface.read_resources(os.path.join(PATH, i))


def test_energies():
    tests = [
        (
            "inputs/orca1.out",
            "inputs/orca_template",
            "inputs/QM1.in",
            [1, 3],
            {
                (1, 1): -550.164846079,
                (1, 2): -550.065349079,
                (1, 3): -550.051038079,
                (1, 4): -549.960953079,
                (1, 5): -549.902495079,
                (3, 1): -550.096449079,
                (3, 2): -550.090568079,
                (3, 3): -550.074080079,
                (3, 4): -549.942447079,
                (3, 5): -549.936124079,
            },
            (5, 0, 4),
        ),
        (
            "inputs/orca6_1.out",
            "inputs/orca_template",
            "inputs/QM1.in",
            [1, 3],
            {
                (1, 1): -550.164846079,
                (1, 2): -550.065349079,
                (1, 3): -550.051038079,
                (1, 4): -549.960953079,
                (1, 5): -549.902495079,
                (3, 1): -550.096449079,
                (3, 2): -550.090568079,
                (3, 3): -550.074080079,
                (3, 4): -549.942447079,
                (3, 5): -549.936124079,
            },
            (6, 0, 0),
        ),
        ("inputs/orca1-2.out", "inputs/orca_template", "inputs/QM1.in", [2], {(2, 1): -549.725632289}, (5, 0, 4)),
        ("inputs/orca6_1-2.out", "inputs/orca_template", "inputs/QM1.in", [2], {(2, 1): -549.725632289}, (6, 0, 0)),
        (
            "inputs/orca3.out",
            "inputs/orca_template",
            "inputs/orca3.in",
            [2],
            {
                (2, 1): -549.725632289,
                (2, 2): -549.691766289,
                (2, 3): -549.690712289,
                (2, 4): -549.639773289,
                (2, 5): -549.631470289,
            },
            (5, 0, 4),
        ),
        (
            "inputs/orca6_3.out",
            "inputs/orca_template",
            "inputs/orca3.in",
            [2],
            {
                (2, 1): -549.725632289,
                (2, 2): -549.691766289,
                (2, 3): -549.690712289,
                (2, 4): -549.639773289,
                (2, 5): -549.631470289,
            },
            (6, 0, 0),
        ),
        (
            "inputs/orca4.out",
            "inputs/orca_template",
            "inputs/orca4.in",
            [4],
            {(4, 1): -549.649784479, (4, 2): -549.641911479},
            (5, 0, 4),
        ),
        (
            "inputs/orca6_4.out",
            "inputs/orca_template",
            "inputs/orca4.in",
            [4],
            {(4, 1): -549.649784479, (4, 2): -549.641911479},
            (6, 0, 0),
        ),
    ]
    for outfile, template, qmin, mults, energies, orca in tests:
        get_energy(
            os.path.join(PATH, outfile),
            os.path.join(PATH, template),
            os.path.join(PATH, qmin),
            mults,
            energies,
            orca,
        )


def test_buildjobs1():
    tests = [
        (
            "inputs/orca3.in",
            "inputs/job_template1",
            {"joblist": [2], "states_to_do": [0, 5], "jobs": {2: {"mults": [2], "restr": False}}},
        ),
        (
            "inputs/orca4.in",
            "inputs/job_template1",
            {
                "joblist": [2, 4],
                "states_to_do": [0, 2, 0, 2],
                "jobs": {2: {"mults": [2], "restr": False}, 4: {"mults": [4], "restr": False}},
            },
        ),
        (
            "inputs/orca5.in",
            "inputs/job_template1",
            {
                "joblist": [1, 2],
                "states_to_do": [4, 2, 3],
                "jobs": {1: {"mults": [1, 3], "restr": True}, 2: {"mults": [2], "restr": False}},
            },
        ),
        (
            "inputs/orca6.in",
            "inputs/job_template2",
            {
                "joblist": [2, 3],
                "states_to_do": [0, 2, 3],
                "jobs": {2: {"mults": [2], "restr": False}, 3: {"mults": [3], "restr": False}},
            },
        ),
        (
            "inputs/orca7.in",
            "inputs/job_template2",
            {
                "joblist": [1, 2, 3],
                "states_to_do": [1, 2, 3],
                "jobs": {1: {"mults": [1], "restr": True}, 2: {"mults": [2], "restr": False}, 3: {"mults": [3], "restr": False}},
            },
        ),
    ]

    for path, template, maps in tests:
        build_jobs(os.path.join(PATH, path), os.path.join(PATH, template), maps)


def test_buildjobs2():
    tests = [
        ("inputs/orca9.in", "inputs/job_template3", {}),
        ("inputs/orca8.in", "inputs/job_template2", {}),
    ]

    for path, template, maps in tests:
        with pytest.raises(ValueError):
            build_jobs(os.path.join(PATH, path), os.path.join(PATH, template), maps)


@pytest.mark.dependency(depends=["test_orcaversion"])
def test_read_mos():
    tests = [
        ("inputs/QM5.in", "inputs/abinitio_template1", "inputs/read_mo1", "inputs/mos1", 1),
        ("inputs/orca3.in", "inputs/abinitio_template1", "inputs/read_mo2", "inputs/mos2", 2),
    ]

    for qmin, template, gbw, mos, job in tests:
        test_interface = SHARC_ORCA()
        test_interface.setup_mol(os.path.join(PATH, qmin))
        test_interface.read_template(os.path.join(PATH, template))
        test_interface.QMin.resources["orcadir"] = expand_path("$ORCADIR")
        test_interface._read_resources = True
        test_interface.setup_interface()
        test_interface.QMin.molecule["frozcore"] = sum(map(lambda x: FROZENS[x], test_interface.QMin.molecule["elements"]))
        with open(os.path.join(PATH, mos), "r", encoding="utf-8") as file:
            ref_mos = file.read()
            assert test_interface._get_mos(os.path.join(PATH, gbw), job) == ref_mos
            os.remove(os.path.join(PATH, gbw, "fragovlp.out"))
            os.remove(os.path.join(PATH, gbw, "fragovlp.err"))


def test_get_dets():
    tests = [
        ("inputs/orca_dets_input", "inputs/orca_cis1", 1, 1, "inputs/orca_dets1"),
        ("inputs/orca_dets_input", "inputs/orca_cis2", 2, 2, "inputs/orca_dets2"),
        ("inputs/orca_dets_input", "inputs/orca_cis3", 1, 3, "inputs/orca_dets3"),
        ("inputs/orca_dets_input4", "inputs/orca_cis4", 1, 1, "inputs/orca_dets4"),
    ]

    for qmin, cis, job, mult, det in tests:
        test_interface = SHARC_ORCA()
        test_interface.setup_mol(os.path.join(PATH, qmin))
        test_interface._read_template = True
        test_interface.read_resources(os.path.join(PATH, "inputs/ORCA.resources"))
        test_interface.setup_interface()
        with open(os.path.join(PATH, det), "r", encoding="utf-8") as file:
            ref_det = file.read().split("\n")
            test_det = test_interface.get_dets_from_cis(os.path.join(PATH, cis), job)[f"dets.{mult}"].split("\n")
            assert test_det[0] == ref_det[0]
            for test, ref in zip(test_det[1:-1], ref_det[1:-1]):
                test_split = test.split()
                ref_split = ref.split()
                assert test_split[0] == ref_split[0]
                for test_val, ref_val in zip(test_split[1:], ref_split[1:]):
                    assert float(test_val) == pytest.approx(float(ref_val), abs=1e-7)


def test_ao_matrix():
    tests = [("inputs/aooverl1gbw", "inputs/aooverl1"), ("inputs/aooverl2gbw", "inputs/aooverl2")]

    test_interface = SHARC_ORCA()
    test_interface.QMin.resources["orcadir"] = expand_path("$ORCADIR")

    for gbw, ovl in tests:
        ao_overl = test_interface._get_ao_matrix(os.path.join(PATH, gbw))
        os.remove(os.path.join(PATH, gbw, "fragovlp.out"))
        os.remove(os.path.join(PATH, gbw, "fragovlp.err"))
        with open(os.path.join(PATH, ovl), "r") as ref:
            assert ao_overl == ref.read()


def test_ao_matrix_overlap():
    tests = [
        ("inputs/orca_overlap/aooverl1", "gbw1.1.2", "gbw1.1.1"),
        ("inputs/orca_overlap/aooverl2", "gbw2.1.2", "gbw2.1.1"),
        ("inputs/orca_overlap/aooverl3", "gbw3.2.2", "gbw3.2.1"),
    ]

    test_interface = SHARC_ORCA()
    test_interface.QMin.resources["orcadir"] = expand_path("$ORCADIR")
    for aooverl, gbw1, gbw2 in tests:
        ao_overl = test_interface._get_ao_matrix(os.path.join(PATH, "inputs/orca_overlap"), gbw1, gbw2, 15, True)
        os.remove(os.path.join(PATH, "inputs/orca_overlap", "fragovlp.out"))
        os.remove(os.path.join(PATH, "inputs/orca_overlap", "fragovlp.err"))
        with open(os.path.join(PATH, aooverl), "r") as ref:
            assert ref.read() == ao_overl


def test_template():
    tests = [
        (
            "inputs/orca_templatetest1",
            {
                "paddingstates": None,
                "no_tda": False,
                "basis": "6-31G",
                "auxbasis": None,
                "functional": "b3lyp",
                "dispersion": "D3",
                "ri": "rijcosx",
                "scf": None,
                "keys": "tightscf zora",
                "paste_input_file": None,
                "frozen": -1,
                "maxiter": 700,
                "hfexchange": -1.0,
                "intacc": -1.0,
                "unrestricted_triplets": False,
                "basis_per_element": ["F", "cc-pvqz"],
                "basis_per_atom": None,
                "ecp_per_element": None,
            },
        ),
        (
            "inputs/orca_templatetest2",
            {
                "paddingstates": None,
                "no_tda": False,
                "basis": "6-31G",
                "auxbasis": None,
                "functional": "b3lyp",
                "dispersion": "D3",
                "ri": "rijcosx",
                "scf": None,
                "keys": "tightscf zora",
                "paste_input_file": None,
                "frozen": -1,
                "maxiter": 700,
                "hfexchange": -1.0,
                "intacc": -1.0,
                "unrestricted_triplets": False,
                "basis_per_element": ["F", "cc-pvqz"],
                "basis_per_atom": ["2", "cc-pvtz", "1", "cc-pvtz"],
                "ecp_per_element": None,
            },
        ),
        (
            "inputs/orca_templatetest3",
            {
                "paddingstates": None,
                "no_tda": False,
                "basis": "cc-pvdz",
                "auxbasis": "cc-pvdz/j",
                "functional": "b3lyp",
                "dispersion": "D3",
                "ri": "rijcosx",
                "scf": None,
                "keys": "tightscf zora",
                "paste_input_file": None,
                "frozen": -1,
                "maxiter": 700,
                "hfexchange": -1.0,
                "intacc": -1.0,
                "unrestricted_triplets": False,
                "basis_per_element": ["F", "cc-pvqz"],
                "basis_per_atom": ["2", "cc-pvtz", "1", "cc-pvtz"],
                "ecp_per_element": None,
            },
        ),
    ]
    for template, ref in tests:
        test_interface = SHARC_ORCA()
        test_interface.setup_mol(os.path.join(PATH, "inputs/QM1.in"))
        test_interface.read_template(os.path.join(PATH, template))
        for k, v in ref.items():
            assert test_interface.QMin["template"][k] == v


def test_orb_init():
    test_interface = SHARC_ORCA()
    test_interface.setup_mol(os.path.join(PATH, "inputs", "copy_gbw", "QM.in"))
    test_interface.read_template(os.path.join(PATH, "inputs", "orca_templatetest1"))
    test_interface.read_resources(os.path.join(PATH, "inputs", "ORCA.resources"))
    test_interface.setup_interface()

    tests = [
        (1, 1, "2d8b1ce3415044bb9d4c79e929caa9e1"),
        (0, 1, "b5059c85870b4c8db7da31857353797a"),
        (1, 2, "df3e3a79149d4ffaa8884d30229879b8"),
        (0, 2, "cded86a151644dd9817e38e6f0a436bd"),
        (4, 1, "9557466025f24c8e9377fc4e7aef2a28"),
        (4, 2, "dd6aaae7deaf4da3b9d5c1fca3ceecfc"),
    ]

    for step, job, check in tests:
        test_interface.QMin.control["jobid"] = job
        test_interface.QMin.save["step"] = step
        test_interface.QMin.save["savedir"] = os.path.join(PATH, "inputs", "copy_gbw")
        mkdir(os.path.join(PATH, "inputs", "copy_gbw", "test"))
        test_interface._copy_gbw(test_interface.QMin, os.path.join(PATH, "inputs", "copy_gbw", "test"))
        file = open(os.path.join(PATH, "inputs", "copy_gbw", "test", "ORCA.gbw"), "r", encoding="utf-8").read()
        shutil.rmtree(os.path.join(PATH, "inputs", "copy_gbw", "test"))
        assert file == check
