import datetime
import os
import re
from io import TextIOWrapper

from SHARC_ABINITIO import SHARC_ABINITIO
from utils import convert_list, expand_path

AUTHORS = ""
VERSION = ""
VERSIONDATE = datetime.datetime(2023, 8, 29)
NAME = "MOLCAS"
DESCRIPTION = ""

CHANGELOGSTRING = """
"""

all_features = set(
    [
        "h",
        "dm",
        "soc",
        "grad",
        "ion",
        "overlap",
        "phases",
        "multipolar_fit"
        # raw data request
        "basis_set",
        "wave_functions",
        "density_matrices",
    ]
)


class SHARC_MOLCAS(SHARC_ABINITIO):
    """
    SHARC interface for MOLCAS
    """

    _version = VERSION
    _versiondate = VERSIONDATE
    _authors = AUTHORS
    _changelogstring = CHANGELOGSTRING
    _name = NAME
    _description = DESCRIPTION

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # Add resource keys
        self.QMin.resources.update(
            {
                "molcas": None,
                "wfoverlap": None,
                "mpi_parallel": False,
                "schedule_scaling": 0.6,
                "delay": 0.0,
                "always_orb_init": False,
                "always_guess": False,
                "savedir": None,
            }
        )

        self.QMin.resources.types.update(
            {
                "molcas": str,
                "wfoverlap": str,
                "mpi_parallel": bool,
                "schedule_scaling": float,
                "delay": float,
                "always_orb_init": bool,
                "always_guess": bool,
                "savedir": str,
            }
        )

        # Add template keys
        self.QMin.template.update(
            {
                "basis": None,
                "baslib": None,
                "nactel": None,
                "ras2": None,
                "inactive": None,
                "roots": list(range(8)),
                "rootpad": list(range(8)),
                "method": "casscf",
                "functional": "tpbe",
                "douglas-kroll": False,
                "ipea": 0.25,
                "imaginary": 0.0,
                "frozen": -1,
                "cholesky": False,
                "gradaccudefault": 1e-4,
                "gradaccumax": 1e-2,
                "pcmset": None,
                "pcmstate": None,
                "iterations": [200, 100],
            }
        )

        self.QMin.template.update(
            {
                "basis": str,
                "baslib": str,
                "nactel": int,
                "ras2": int,
                "inactive": int,
                "roots": list,
                "rootpad": list,
                "method": str,
                "functional": str,
                "douglas-kroll": bool,
                "ipea": float,
                "imaginary": float,
                "frozen": int,
                "cholesky": bool,
                "gradaccudefault": float,
                "gradaccumax": float,
                "pcmset": (dict, list),
                "pcmstate": list,
                "iterations": list,
            }
        )

    @staticmethod
    def version() -> str:
        return SHARC_MOLCAS._version

    @staticmethod
    def versiondate() -> datetime.datetime:
        return SHARC_MOLCAS._versiondate

    @staticmethod
    def changelogstring() -> str:
        return SHARC_MOLCAS._changelogstring

    @staticmethod
    def authors() -> str:
        return SHARC_MOLCAS._authors

    @staticmethod
    def name() -> str:
        return SHARC_MOLCAS._name

    @staticmethod
    def description() -> str:
        return SHARC_MOLCAS._description

    @staticmethod
    def about() -> str:
        return f"{SHARC_MOLCAS._name}\n{SHARC_MOLCAS._description}"

    def get_features(self, KEYSTROKES: TextIOWrapper | None = None) -> set[str]:
        """return availble features

        ---
        Parameters:
        KEYSTROKES: object as returned by open() to be used with question()
        """
        return all_features

    def get_infos(self, INFOS: dict, KEYSTROKES: TextIOWrapper | None = None) -> dict:
        """prepare INFOS obj

        ---
        Parameters:
        INFOS: dictionary with all previously collected infos during setup
        KEYSTROKES: object as returned by open() to be used with question()
        """
        return INFOS

    def create_restart_files(self) -> None:
        pass

    def read_resources(self, resources_file: str = "MOLCAS.resources", kw_whitelist: list[str] | None = None) -> None:
        super().read_resources(resources_file, kw_whitelist)

        # Path to MOLCAS
        if not self.QMin.resources["molcas"]:
            self.log.error(f"molcas key not found in {resources_file}")
            raise ValueError()

        self.QMin.resources["molcas"] = expand_path(self.QMin.resources["molcas"])
        os.environ["MOLCAS"] = self.QMin.resources["molcas"]

        if self.get_molcas_version(self.QMin.resources["molcas"]) < (18, 0):
            self.log.error("This version of SHARC-MOLCAS is only compatible with MOLCAS 18 or higher!")
            raise ValueError()

        # MOLCAS driver
        driver = os.path.join(self.QMin.resources["molcas"], "bin", "pymolcas")
        if os.path.isfile(driver):
            self.QMin.resources.update({"driver": driver})

        driver = os.path.join(self.QMin.resources["molcas"], "bin", "molcas.exe")
        if os.path.isfile(driver) and "driver" not in self.QMin.resources:
            self.QMin.resources.update({"driver": driver})

        if "driver" not in self.QMin.resources:
            self.log.error(f"No driver found in {os.path.join(self.QMin.resources['molcas'], 'bin')}")
            raise ValueError()

        # WFOVERLAP
        if self.QMin.resources["wfoverlap"]:
            self.QMin.resources["wfoverlap"] = expand_path(self.QMin.resources["wfoverlap"])

        # Check orb init and guess
        if self.QMin.resources["always_guess"] and self.QMin.resources["always_orb_init"]:
            self.log.error("always_guess and always_orb_init cannot be used together!")
            raise ValueError()

    def read_template(self, template_file: str = "MOLCAS.template", kw_whitelist: list[str] | None = None) -> None:
        super().read_template(template_file, kw_whitelist)

        # Roots
        self.QMin.template["roots"] = convert_list(self.QMin.template["roots"])
        self.QMin.template["rootpad"] = convert_list(self.QMin.template["rootpad"])

        if not all(map(lambda x: x >= 0, [*self.QMin.template["roots"], *self.QMin.template["rootpad"]])):
            self.log.error("roots and rootpad must contain positive integers.")
            raise ValueError()

        for idx, val in enumerate(self.QMin.template["roots"]):
            if val < self.QMin.molecule["states"][idx]:
                self.log.error(f"Too few states in state-averaging in multiplicity {idx+1}!")
                raise ValueError()

        for idx, val in enumerate(reversed(self.QMin.template["roots"])):
            if val == 0:
                self.QMin.template["roots"] = self.QMin.template["roots"][: -1 * idx]
                break

        self.QMin.template["rootpad"] = self.QMin.template["rootpad"][: len(self.QMin.template["roots"])]

        # Path to baslib
        if self.QMin.template["baslib"]:
            self.QMin.template["baslib"] = expand_path(self.QMin.template["baslib"])

        # Iterations
        match len(self.QMin.template["iterations"]):
            case 2:
                self.QMin.template["iterations"] = convert_list(self.QMin.template["iterations"])
            case 1:
                self.QMin.template["iterations"] = [int(self.QMin.template["iterations"][0]), 100]
            case _:
                self.log.error(f"{self.QMin.template['iterations']} is not a valid iteration value!")
                raise ValueError()

        # PCM
        if isinstance(self.QMin.template["pcmset"], list):
            if len(self.QMin.template["pcmset"]) != 3:
                self.log.error("pcmset must contain three parameter!")
                raise ValueError()

            self.QMin.template["pcmset"] = {
                "solvent": self.QMin.template["pcmset"][0],
                "aare": float(self.QMin.template["pcmset"][1]),
                "r-min": float(self.QMin.template["pcmset"][2]),
            }

        # Check for basis and cas settings
        for i in ["basis", "nactel", "ras2", "inactive"]:
            if not self.QMin.template[i]:
                self.log.error(f"Key {i} is missing in template file!")
                raise ValueError()

        # Validate method
        if self.QMin.template["method"] not in ["casscf", "caspt2", "ms-caspt2", "mc-pdft", "xms-pdft", "cms-pdft"]:
            self.log.error(f"{self.QMin.template['method']} is not a valid method!")
            raise ValueError()

        # TODO: gradmode
        # gradmode 0 = one file; gradmode 1 = multiple files

    @staticmethod
    def get_molcas_version(path: str) -> tuple[int, int]:
        """
        Get version number of MOLCAS

        path:   Path to MOLCAS directory
        """

        with open(os.path.join(path, ".molcasversion"), "r", encoding="utf-8") as version_file:
            version = re.match(r"v(\d+)\.(\d+)", version_file.read())
            if not version:
                raise ValueError(f"No MOLCAS version found in {os.path.join(path, '.molcasversion')}")
        return int(version.group(1)), int(version.group(2))
