from qmout import QMout
from utils import expand_path
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
        #(os.path.join(PATH, "421_h_soc_dm_grad_ion"), os.path.join(PATH, "421_h_soc_dm_grad_ion_ref")), # plus notes (turbomole)
        (os.path.join(PATH, "421_h_soc_dm_grad_ion2"), os.path.join(PATH, "421_h_soc_dm_grad_ion_ref")), # no notes
        (os.path.join(PATH, "421_h_theodore"), os.path.join(PATH, "421_h_theodore_ref")), # no notes
        (os.path.join(PATH, "410_h_multipolar"), os.path.join(PATH, "410_h_multipolar_ref")), # plus densities
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
                        assert v2 == reference[k][k2]
            elif v:
                assert v == reference[k], k