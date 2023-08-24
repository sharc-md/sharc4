from SHARC_INTERFACE import SHARC_INTERFACE
from qmin import QMin
from abc import abstractmethod
from io import TextIOWrapper
from typing import List, Dict
from datetime import date

all_features = {
    "h",
    "soc",
    "dm",
    "grad",
    "nacdr",
    "overlap",
    "phases",
    "ion",
    "dmdr",
    "socdr",
    "multipolar_fit",
    "theodore",
    "point_charges",
    # raw data request
    "basis_set",
    "wave_functions",
    "density_matrices",
}


class SHARC_ABINITIO(SHARC_INTERFACE):
    """
    Abstract base class for ab-initio interfaces
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    @abstractmethod
    def authors(self) -> str:
        return "Severin Polonius, Sebastian Mai"

    @abstractmethod
    def version(self) -> str:
        return "3.0"

    @abstractmethod
    def versiondate(self) -> date:
        return date(2021, 7, 15)

    @staticmethod
    @abstractmethod
    def name() -> str:
        return "base"

    @staticmethod
    @abstractmethod
    def description() -> str:
        return "Abstract base class for SHARC interfaces."

    @staticmethod
    @abstractmethod
    def changelogstring() -> str:
        return "This is the changelog string"

    @staticmethod
    @abstractmethod
    def about() -> str:
        return "Name and description of the interface"

    @abstractmethod
    def get_features(self, KEYSTROKES: TextIOWrapper = None) -> set:
        """return availble features

        ---
        Parameters:
        KEYSTROKES: object as returned by open() to be used with question()
        """
        return all_features

    @abstractmethod
    def get_infos(self, INFOS: dict, KEYSTROKES: TextIOWrapper = None) -> dict:
        """prepare INFOS obj

        ---
        Parameters:
        INFOS: dictionary with all previously collected infos during setup
        KEYSTROKES: object as returned by open() to be used with question()
        """
        return INFOS

    @abstractmethod
    def prepare(self, INFOS: dict, dir: str):
        "setup the calculation in directory 'dir'"
        return

    @abstractmethod
    def print_qmin(self) -> None:
        pass

    @abstractmethod
    def execute_from_qmin(self, workdir: str, qmin: QMin):
        """
        Erster Schritt, setup_workdir ( inputfiles schreiben, orbital guesses kopieren, xyz, pc)
        Programm aufrufen (z.b. run_program)
        checkstatus(), check im workdir ob rechnung erfolgreich
            if success: pass
            if not: try again or return error
        postprocessing of workdir files (z.b molden file erzeugen)
        """

    @abstractmethod
    def read_template(self, template_file: str) -> None:
        super().read_template(template_file)

    @abstractmethod
    def read_resources(self, resources_file: str, kw_whitelist: List[str] = []) -> None:
        super().read_resources(resources_file, kw_whitelist)

    @abstractmethod
    def read_requests(self, requests_file: str = "QM.in") -> None:
        super().read_requests(requests_file)

    @abstractmethod
    def write_step_file(self) -> None:
        super().write_step_file()

    @abstractmethod
    def printQMout(self):
        super().writeQMout()

    @abstractmethod
    def setup_interface(self):
        pass

    @abstractmethod
    def getQMout(self):
        pass

    @abstractmethod
    def create_restart_files(self):
        pass

    def run_program(
        self, workdir: str, cmd: str, out: str, err: str, keepfiles: List[str]
    ) -> int:
        return error_code

    def runjobs(self, schedule: List[Dict[str, QMin]]):
        """ """

    @abstractmethod
    def run(self):
        """
        request & other logic
            requestmaps anlegen
            pfade für verschiedene orbital restart files
        make schedule
        runjobs()
        run_wfoverlap (braucht input files)
        run_theodore
        save directory handling
        """
