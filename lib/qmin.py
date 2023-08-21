import logging
import os
import re
import sys
from collections import UserDict

from error import Error

__all__ = ["QMin"]

logging.basicConfig(
    format="{levelname:<9s} [{filename}:{funcName}():{lineno}] {message}",
    level=logging.INFO,
    style="{",
)
logger = logging.getLogger(__name__)


class QMinBase(UserDict):
    """
    Base class for custom dictionary used in QMin
    """

    def __setitem__(self, key, value):
        # Check if new values has the correct type (if available)
        if key in self.types and not isinstance(value, self.types[key]):
            raise Error(
                f"{key} should be of type {self.types[key]} but is {type(value)}"
            )
        self.data[key] = value

    def __getitem__(self, key):
        return self.data[key]

    def __str__(self) -> str:
        return "".join("{}: {}\n".format(k, v) for k, v in self.data.items())


class QMinMolecule(QMinBase):
    """
    Custom dictionary for the molecule section
    """

    def __init__(self):
        # Set data dictionary and dictionary of default types
        self.data = {
            "comment": None,
            "natom": None,
            "elements": None,
            "unit": None,
            "factor": None,
            "states": None,
            "nstates": None,
            "nmstates": None,
            "qmmm": False,
            "npc": None,
            # Ab initio interfaces
            "Atomcharge": None,
            "frozcore": None,
        }

        self.types = {
            "comment": str,
            "natom": int,
            "elements": list,
            "unit": str,
            "factor": float,
            "states": list,
            "nstates": int,
            "nmstates": int,
            "qmmm": bool,
            "npc": int,
            "Atomcharge": int,
            "frozcore": int,
        }


class QMinCoords(QMinBase):
    """
    Custom dictionary for the coords section
    """

    def __init__(self):
        # Set data dictionary and dictionary of default types
        self.data = {"coords": None, "pccoords": None, "pccharce": None}
        self.types = {"coords": list, "pccoords": list, "pccharce": list}


class QMinSave(QMinBase):
    """
    Custom dictionary for the save section
    """

    def __init__(self):
        # Set data dictionary and dictionary of default types
        self.data = {
            "savedir": os.path.join(os.getcwd(), "SAVE"),
            "step": None,
            "previous_step": None,
            "init": False,
            "newstep": False,
            "samestep": False,
            "restart": False,
            # Ab initio interfaces
            "always_guess": False,
            "always_orb_init": False,
        }
        # TODO: lookup missing types
        self.types = {
            "savedir": str,
            "step": int,
            "previous_step": int,
            "init": bool,
            "newstep": bool,
            "samestep": bool,
            "restart": bool,
            "always_guess": bool,
            "always_orb_init": bool,
        }


class QMinRequests(QMinBase):
    """
    Custom dictionary for the requests section
    """

    def __init__(self):
        # Set data dictionary and dictionary of default types
        self.data = {
            "h": False,
            "soc": False,
            "dm": False,
            "grad": None,
            "nacdr": [],
            "overlap": False,
            "phases": False,
            "ion": False,
            "socdr": False,
            "dmdr": False,
            "multipolar_fit": False,
            "theodore": False,
            "cleanup": False,
            "backup": False,
            "molden": False,
            "savestuff": False,
            "nooverlap": False,
        }
        self.types = {
            "h": bool,
            "soc": bool,
            "dm": bool,
            "grad": list,
            "nacdr": list,
            "overlap": bool,
            "phases": bool,
            "ion": bool,
            "socdr": bool,
            "dmdr": bool,
            "multipolar_fit": bool,
            "theodore": bool,
            "cleanup": bool,
            "backup": bool,
            "molden": bool,
            "savestuff": bool,
            "nooverlap": bool,
        }


class QMinMaps(QMinBase):
    """
    Custom dictionary for the maps section
    """

    def __init__(self):
        self.data = {
            "statemap": None,
            "mults": None,
            "gsmap": None,
            "gradmap": None,
            "nacmap": None,
            "densmap": None,
            "ionmap": None,
            # Ab initio interfaces
            "chargemap": None,
            "multmap": None,
        }

        self.types = {
            "statemap": dict,
            # "mults": set,
            "gsmap": dict,
            "gradmap": set,
            "nacmap": set,
            "densmap": set,
            "ionmap": list,
            "chargemap": dict,
            "multmap": dict,
        }


class QMinResources(QMinBase):
    """
    Custom dictionary for the resources section
    """

    def __init__(self):
        self.data = {
            "pwd": os.getcwd(),
            "cwd": os.getcwd(),
            "scratchdir": os.path.join(os.getcwd(), "SCRATCH"),
            "ncpu": 1,
            "ngpu": None,
            "memory": 100,
            # For ab initio interfaces with external code
            "delay": 0,
            "schedule_scaling": 0,
            # Overlaps (for ab initio)
            "wfoverlap": "",
            "wfthres": None,
            "numfrozcore": None,
            "numocc": None,
            # Theodore resources
            "theodir": "",
            "theodore_prop": None,
            "theodore_fragment": None,
            "theodore_n": 0,
            # Resources for RESP
            "resp_layers": None,
            "resp_tdm_fit_order": None,
            "resp_density": None,
            "resp_first_layer": None,
            "resp_shells": None,
            "resp_grid": None,
            # Paths to ab initio programs
            "molprodir": None,
            "molcasdir": None,
            "orcadir": None,
            "gaussiandir": None,
            "turbomoledir": None,
            "bageldir": None,
            "columbusdir": None,
            "amsdir": None,
            "tinkerdir": None,
            "openmmdir": None,
            "cobrammdir": None,
            # Interface specific
            "molcas_driver": None,
            "mpi_parallel": False,
            "runc": None,
            "scmlicense": None,
            "scm_tmpdir": None,
            "pyquante": None,
        }
        # TODO: lookup missing types
        self.types = {
            "pwd": str,
            "cwd": str,
            "scratchdir": str,
            "ncpu": int,
            "ngpu": int,
            "memory": int,
            "delay": float,
            "schedule_scaling": float,
            "wfoverlap": str,
            "wfthres": float,
            "numfrozcore": int,
            "numocc": int,
            "theodir": str,
            "theodore_prop": list,
            "theodore_fragment": list,
            "theodore_n": int,
            # "resp_layers": None,
            # "resp_tdm_fit_order": None,
            # "resp_density": None,
            # "resp_first_layer": None,
            # "resp_shells": None,
            # "resp_grid": None,
            "molprodir": str,
            "molcasdir": str,
            "orcadir": str,
            "gaussiandir": str,
            "turbomoledir": str,
            "bageldir": str,
            "columbusdir": str,
            "amsdir": str,
            "tinkerdir": str,
            "openmmdir": str,
            "cobrammdir": str,
            "molcas_driver": str,
            "mpi_parallel": bool,
            "runc": str,
            "scmlicense": str,
            "scm_tmpdir": str,
            "pyquante": str,
        }


class QMinTemplate(QMinBase):
    """
    Custom dictionary for the template section
    """

    def __init__(self):
        """
        Just read anything from template file and process it in
        the interface?
        """
        # self.data = {
        #    "neglected_gradients": None,
        #    "dipolelevel": None,
        #    "template_file": None,
        # }
        #
        # self.types = {
        #    "neglected_gradients": str,
        #    "dipolelevel": int,
        #    "template_file": str,
        # }
        self.data = {}
        self.types = {}


class QMinControl(QMinBase):
    """
    Custom dictionary for the control section
    """

    def __init__(self):
        self.data = {
            "jobid": None,
            "workdir": None,
            "master": False,
            "gradonly": False,
            "states_to_do": None,
        }

        self.types = {
            "jobid": int,
            "workdir": str,
            "master": bool,
            "gradonly": bool,
            "states_to_do": list,
        }


class QMinInterface(QMinBase):
    """
    Custom dictionary for the interface section
    """

    def __init__(self):
        self.data = {}
        self.types = {}


class QMinScheduling(QMinBase):
    """
    Custom dictionary for the scheduling section
    """

    def __init__(self):
        self.data = {}
        self.types = {}


class QMinBackwards(QMinBase):
    """
    Custom dictionary for the backwards section
    """

    def __init__(self):
        self.data = {}
        self.types = {}


class QMin:
    """
    The QMin object carries all information relevant to the execution of a SHARC interface.
    """

    interface = QMinInterface()
    molecule = QMinMolecule()
    coords = QMinCoords()
    save = QMinSave()
    requests = QMinRequests()
    maps = QMinMaps()
    resources = QMinResources()
    template = QMinTemplate()
    scheduling = QMinScheduling()
    control = QMinControl()
    backwards = QMinBackwards()

    def __getitem__(self, key):
        return getattr(self, key)

    def __str__(self):
        return f"""\
Interface: 
{self.interface}
Molecule: 
{self.molecule}
Coords: 
{self.coords}
Save: 
{self.save}
Requests: 
{self.requests}
Maps: 
{self.maps}
Resources: 
{self.resources}
Template: 
{self.template}
Scheduling: 
{self.scheduling}
Control: 
{self.control}
Backwards: 
{self.backwards}"""


if __name__ == "__main__":
    pass
