from SHARC_ABINITIO import SHARC_ABINITIO
import datetime

__all__ = ["SHARC_GAUSSIAN"]

AUTHORS = ""
VERSION = ""
VERSIONDATE = datetime.datetime(2023, 8, 29)
NAME = "GAUSSIAN"
DESCRIPTION = ""

CHANGELOGSTRING = """
"""


class SHARC_GAUSSIAN(SHARC_ABINITIO):
    """
    SHARC interface for gaussian
    """

    _version = VERSION
    _versiondate = VERSIONDATE
    _authors = AUTHORS
    _changelogstring = CHANGELOGSTRING
    _name = NAME
    _description = DESCRIPTION

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Add template keys
        self.QMin.template.update(
            {
                "basis": "6-31G",
                "functional": "PBEPBE",
                "dispersion": None,
                "scrf": None,
                "grid": None,
                "denfit": False,
                "scf": None,
                "no_tda": False,
                "unrestricted_triplets": False,
                "iop": None,
            }
        )
        self.QMin.template.types.update(
            {
                "basis": str,
                "functional": str,
                "dispersion": str,
                "scrf": str,
                "grid": str,
                "denfit": bool,
                "scf": str,
                "no_tda": bool,
                "unrestricted_triplets": bool,
                "iop": str,
            }
        )

        # Add resource keys
        self.QMin.resources.update(
            {
                "groot": None,
                "wfoverlap": None,
                "wfthres": None,
                "numfrozcore": 0,
                "numocc": None,
            }
        )

        self.QMin.resources.types.update(
            {
                "groot": str,
                "wfoverlap": str,
                "wfthres": float,
                "numfrozcore": int,
                "numocc": int,
            }
        )
