#!/usr/bin/env python3
import datetime
import os
from io import TextIOWrapper

from numpy import ndarray
from qmin import QMin
from SHARC_ABINITIO import SHARC_ABINITIO
from utils import mkdir


# ---------------------------------| Infos |---------------------------------------------------------------------------

#TODO: Change INTERFACENAME to your desired name

__all__ = ["SHARC_INTERFACENAME"]  # Only export interface class


#TODO: This will be shown in the header when running a single point or sharc.x
AUTHORS = "<your name>"
VERSION = "<interface version>"
VERSIONDATE = datetime.datetime(1999, 1, 1)
#TODO: This will be shown in the setup scripts
NAME = "<interface name>"
DESCRIPTION = "<interface description>"

CHANGELOGSTRING = "<changelog>"

#TODO: Set the supported properties of interface
all_features = set(
    [
        #TODO: requests that your interface can fullfill. Delete the ones that cannot be used. 
        "h",
        "dm",
        "grad",
        "nacdr",
        "overlap",
        "molden",
        # Rest of the possible requests:
        # "phases",
        # "ion",
        # "soc",
        # "multipolar_fit",
        # "theodore",
        # "density_matrices",
        # "retain",
        # "molden",
    ]
)



# ---------------------------------| Template/Resources Definition |----------------------------------------------------

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

        #TODO: Define all class variables here
        self._need_this_later = None

        #TODO: Add extra keys for resources and template that is not already defined in QMin
        self.QMin.resources.update(
            {
                #Resources that your QC-program and interface need go here.
                #TODO:
                "programdir": None, # Path to the executable of the QC-program
                "wfthres": 1.0,     # Norm of wave function for writing determinant files
                "numocc": None,     # Number of orbitals to not ionize from in Dyson calculations
            }
        )
        #TODO: Keys will be parsed as strings, for automatic casting add a type
        self.QMin.resources.types.update(
            {
                #TODO:
                "programdir": str,
                "wfthres": float,
                "numocc": int,
            }
        )

        #TODO: Same for template
        self.QMin.template.update(
            {
                #Program specific information goes here. This data is read from the <INTERFACE_NAME>.template file.
                #You can put whatever you need here.We have put some examples here.
                #TODO: Examples:
                "basis": "6-31G",
                "functional": "PBE",
                "maxscf": 5000,
                "nocc": 0,
                "nunocc": 0,
                "act_orbs": [1],
                "charge": 0,
            }
        )
        self.QMin.template.types.update(
            {
                #TODO:
                "basis": str,
                "functional": str,
                "maxscf": int,
                "nocc": int,
                "nunocc": int,
                "act_orbs": list,
                "charge": int,
            }
        )

# ---------------------------------| Standard Methods |------------------------------------------------------------

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

# ---------------------------------| Initialization |------------------------------------------------------------------

    def read_template(self, template_file: str = "INTERFACENAME.template", kw_whitelist: list[str] | None = None) -> None:
        super().read_template(template_file, kw_whitelist)

        #TODO: Validate and/or process custom template keys here

    def read_resources(self, resources_file: str = "INTERFACENAME.resources", kw_whitelist: list[str] | None = None) -> None:
        super().read_resources(resources_file, kw_whitelist)

        #TODO: Validate and/or process custom resources keys here

    def setup_interface(self) -> None:
        super().setup_interface()

        #TODO: Setup stuff that needs to be done after read_template and read_resources

    
    def read_requests(self, requests_file: str = "QM.in") -> None:
        super().read_requests(requests_file)

        for req, val in self.QMin.requests.items():
            if val and req != "retain" and req not in all_features:
                raise ValueError(f"Found unsupported request {req}.")

        #TODO: Do some request related checks here. Only important for hybrid interfaces.


# ---------------------------------| Run Schedule |---------------------------------------------------------------------
# and WFoverlap/Theodore + Care for Restart Information

    def run(self) -> None:
        starttime = datetime.datetime.now()

        #TODO: Build schedule executed by runjobs here
        

        # Execute schedule, execute_from_qmin will be run inside runjobs
        self.runjobs(self.QMin.scheduling["schedule"])

        # Run overlap calc here if needed
        if self.QMin.requests["overlap"]:
            self._run_wfoverlap()
        
        #TODO: ion/dyson calc and everything that has to be done after the actual QM calc

        self.log.debug("All jobs finished successful")

        self.QMout["runtime"] = datetime.datetime.now() - starttime
    
    def create_restart_files(self) -> None:
        pass


# ---------------------------------| Make Schedule |---------------------------------------------------------------------

    def _gen_schedule(self) -> None:
        """
        Generates scheduling from joblist
        """
        pass


# ---------------------------------| Execute Program |-----------------------------------------------------------------

    def execute_from_qmin(self, workdir: str, qmin: QMin) -> tuple[int, datetime.timedelta]:
        """
        Do QM calculation
        will be called in SHARC_ABINITIO.runjobs()
        """

        # Setup workdir
        mkdir(workdir)

        #TODO: Copy restart from savedir and input files needed for calculation here

        # Run QM
        starttime = datetime.datetime.now()
        exec_str = "<command to run QM program>"
        exit_code = self.run_program(
            workdir, exec_str, os.path.join(workdir, "interfacename.log"), os.path.join(workdir, "interfacename.err")
        )
        endtime = datetime.datetime.now()

        #TODO: Maybe some errorhandling in case exit_code != 0

        #TODO: Post processing, molden file, wfoverlap det/mo files, ...

        #TODO: Copy restart files to savedir
        return exit_code, endtime - starttime
    

# ---------------------------------| Get Data |-----------------------------------------------------------------------

    def getQMout(self) -> dict[str, ndarray]:
        #TODO: Parse requested properties from outputs and populate QMout object. You can make as many parsing and helper functions as you want.
        
        return self.QMout
    

# ---------------------------------| Setup Related |------------------------------------------------------------------

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
        #TODO: Setup things that should be asked during setup here
        return INFOS

    def prepare(self, INFOS: dict, dir_path: str) -> None:
        #TODO: Copy files that are needed for interface in setup
        pass


# ---------------------------------| Additional Methods |------------------------------------------------------------

#TODO: Put all of your extra methods in here. They all should start with and underscore "_". For example a method that parses the gradients from the output-file should be called _get_grad().



# ---------------------------------| Main Function |--------------------------------------------------------------------       

if __name__ == "__main__":
    SHARC_INTERFACENAME().main()