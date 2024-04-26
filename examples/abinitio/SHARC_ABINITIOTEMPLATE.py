import datetime
import os
from io import TextIOWrapper

from numpy import ndarray
from qmin import QMin
from SHARC_ABINITIO import SHARC_ABINITIO
from utils import mkdir

__all__ = ["SHARC_INTERFACENAME"]  # Only export interface class


# This will be shown in the header when running a single point or sharc.x
AUTHORS = "<your name>"
VERSION = "<interface version>"
VERSIONDATE = datetime.datetime(1999, 1, 1)
# This will be shown in the setup scripts
NAME = "<interface name>"
DESCRIPTION = "<interface description>"

CHANGELOGSTRING = "<changelog>"

# All supported properties of interface
all_features = set(["<features>"])  # h, dm, soc, grad, ...


class SHARC_INTERFACENAME(SHARC_ABINITIO):
    """
    Doc string of your interface
    """

    _version = VERSION
    _versiondate = VERSIONDATE
    _authors = AUTHORS
    _changelogstring = CHANGELOGSTRING
    _name = NAME
    _description = DESCRIPTION

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # Define all class variables here
        self._need_this_later = None

        # Add extra keys for resources and template that is not already defined in QMin
        self.QMin.resources.update({"new_key": None})
        # Keys will be parsed as strings, for automatic casting add a type
        self.QMin.resources.types.update({"new_key": str})  # Define type of key

        # Same for template
        self.QMin.template.update({"template_key": None})
        self.QMin.template.types.update({"template_key": int})

    @staticmethod
    def version() -> str:
        return SHARC_INTERFACENAME._version

    @staticmethod
    def versiondate() -> datetime.datetime:
        return SHARC_INTERFACENAME._versiondate

    @staticmethod
    def changelogstring() -> str:
        return SHARC_INTERFACENAME._changelogstring

    @staticmethod
    def authors() -> str:
        return SHARC_INTERFACENAME._authors

    @staticmethod
    def name() -> str:
        return SHARC_INTERFACENAME._name

    @staticmethod
    def description() -> str:
        return SHARC_INTERFACENAME._description

    @staticmethod
    def about() -> str:
        return f"{SHARC_INTERFACENAME._name}\n{SHARC_INTERFACENAME._description}"

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
        # Setup things that should be asked during setup here
        return INFOS

    def prepare(self, INFOS: dict, dir_path: str) -> None:
        # Copy files that are needed for interface in setup
        pass

    def create_restart_files(self) -> None:
        pass

    def execute_from_qmin(self, workdir: str, qmin: QMin) -> tuple[int, datetime.timedelta]:
        """
        Do QM calculation
        will be called in SHARC_ABINITIO.runjobs()
        """

        # Setup workdir
        mkdir(workdir)

        # Copy restart from savedir and input files needed for calculation here

        # Run QM
        starttime = datetime.datetime.now()
        exec_str = "<command to run QM program>"
        exit_code = self.run_program(
            workdir, exec_str, os.path.join(workdir, "interfacename.log"), os.path.join(workdir, "interfacename.err")
        )
        endtime = datetime.datetime.now()

        # Maybe some errorhandling in case exit_code != 0

        # Post processing, molden file, wfoverlap det/mo files, ...

        # Copy restart files to savedir
        return exit_code, endtime - starttime
    
    def _create_aoovl(self) -> None:
        # Create AO overlap between geometry A and B if overlaps are needed/possible
        pass

    def read_template(self, template_file: str = "INTERFACENAME.template", kw_whitelist: list[str] | None = None) -> None:
        super().read_template(template_file, kw_whitelist)

        # Validate and/or process custom template keys here

    def read_resources(self, resources_file: str = "INTERFACENAME.resources", kw_whitelist: list[str] | None = None) -> None:
        super().read_resources(resources_file, kw_whitelist)

        # Validate and/or process custom resources keys here

    def read_requests(self, requests_file: str = "QM.in") -> None:
        super().read_requests(requests_file)

        for req, val in self.QMin.requests.items():
            if val and req != "retain" and req not in all_features:
                raise ValueError(f"Found unsupported request {req}.")

        # Do some request related checks here

    def setup_interface(self) -> None:
        super().setup_interface()

        # Setup stuff that needs to be done after read_template and read_resources

    def getQMout(self) -> dict[str, ndarray]:
        # Parse requested properties from outputs and populate QMout object
        return self.QMout

    def run(self) -> None:
        starttime = datetime.datetime.now()

        # Build schedule executed by runjobs here
        
        # Execute schedule, execute_from_qmin will be run inside runjobs
        self.runjobs(self.QMin.scheduling["schedule"])

        # Run overlap calc here if needed
        if self.QMin.requests["overlap"]:
            self._run_wfoverlap()
        # Do ion/dyson calc and everything that has to be done after the actual QM calc

        self.log.debug("All jobs finished successful")

        self.QMout["runtime"] = datetime.datetime.now() - starttime

if __name__ == "__main__":
    SHARC_INTERFACENAME().main()