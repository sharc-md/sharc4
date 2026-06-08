from qmout import QMout
from qmin import QMin
from utils import expand_path, readfile
import os
import pickle
import numpy as np

PATH = expand_path("$SHARC/../tests/interface/inputs/qmout")


def test_read_qmout():
    """
    This test is to check if QM.out files can be loaded.
    No validations done here!
    """
    tests = [
        os.path.join(PATH, "qmout_ion.out"),
        os.path.join(PATH, "qmout_theodore.out"),
        os.path.join(PATH, "qmout_overlap.out"),
        os.path.join(PATH, "qmout_notes.out"),
        os.path.join(PATH, "qmout_nacdr.out"),
        os.path.join(PATH, "qmout_dens.out"),
    ]

    for test in tests:
        QMout(test)


def test_parsing():
    tests = [
        (os.path.join(PATH, "421_h_soc_dm_grad_ion"), os.path.join(PATH, "421_h_soc_dm_grad_ion_ref")),  # plus notes (turbomole)
        (os.path.join(PATH, "421_h_soc_dm_grad_ion2"), os.path.join(PATH, "421_h_soc_dm_grad_ion_ref")),  # no notes
        (os.path.join(PATH, "421_h_theodore"), os.path.join(PATH, "421_h_theodore_ref")),  # no notes
        (os.path.join(PATH, "410_h_multipolar"), os.path.join(PATH, "410_h_multipolar_ref")),  # plus densities
        (os.path.join(PATH, "400_h_nacdr"), os.path.join(PATH, "400_h_nacdr_ref")),
    ]

    for qmout, ref in tests:
        parsed = QMout(qmout)
        with open(ref, "rb") as f:
            reference = pickle.load(f)
        for k, v in parsed.__dict__.items():
            if isinstance(v, np.ndarray):
                assert np.allclose(v, reference[k])
            elif isinstance(v, list):
                if k == "prop2d" or k == "prop1d":
                    for i, j in zip(v, reference[k]):
                        assert i[0] == j[0], k
                        if isinstance(i[1], np.ndarray):
                            assert np.allclose(i[1], j[1]), k
                        else:
                            assert i[1] == j[1]
                else:
                    assert v == reference[k]
            elif isinstance(v, dict):
                for k2, v2 in v.items():
                    if isinstance(v2, np.ndarray):
                        assert np.allclose(v2, reference[k][k2])
                    else:
                        assert v2 == str(reference[k][k2])
            elif v:
                assert v == reference[k], k


def test_writing():
    tests = [
        (os.path.join(PATH, "421_h_soc_dm_grad_ion"), {"h": True, "soc": True, "dm": True, "grad": [1], "ion": True}),
        (os.path.join(PATH, "421_h_soc_dm_grad_ion2"), {"h": True, "soc": True, "dm": True, "grad": [1], "ion": True}),
        (os.path.join(PATH, "421_h_theodore"), {"h": True, "theodore": True}),
        (os.path.join(PATH, "422_h_dm_multipolar_dens_mol"), {"h": True, "dm": True, "multipolar_fit": [1], "density_matrices": [1], "mol": True}),
        (os.path.join(PATH, "400_h_nacdr"), {"nacdr": [1]}),
    ]

    for qmout, req in tests:
        test = QMout(qmout)
        test.multipolar_fit_settings = " order: 2, grid: lebedev, firstlayer: 1.4, density: 4.0, layers: 4"
        requests = QMin().requests
        requests.update(req)
        a = test.write(None, requests).splitlines(True)
        b = readfile(qmout)
        for k, (la, lb) in enumerate(zip(b, a)):
            if la != lb:
                raise AssertionError(f"First diff at line {k+1}:\ntest: {la!r}\n ref: {lb!r}")
        if len(a) != len(b):
            raise AssertionError(f"Different length: test={len(a)} ref={len(b)}")
