from qmout import QMout
from utils import expand_path
import os

PATH = expand_path("$SHARC/../tests/interface/inputs")


def test_read_qmout():
    """
    This test is to check if QM.out files can be loaded.
    No validations done here!
    """
    tests = [os.path.join(PATH, "qmout_ion.out"), os.path.join(PATH, "qmout_theodore.out")]

    for test in tests:
        QMout(test)
