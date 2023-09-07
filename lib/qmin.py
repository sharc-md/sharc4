import os
from collections import UserDict
from copy import deepcopy

import numpy as np

__all__ = ["QMin"]


class QMinBase(UserDict):
    """
    Base class for custom dictionary used in QMin
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data = {}
        self.types = {}

    def __setitem__(self, key, value):
        # Check if new values has the correct type (if available)
        if key in self.types and not isinstance(value, self.types[key]):
            raise TypeError(
                f"{key} should be of type {self.types[key]} but is {type(value)}"
            )
        self.data[key] = value

    def __getitem__(self, key):
        return self.data[key]

    def __str__(self) -> str:
        return "".join(f"{k}: {v}\n" for k, v in self.data.items())


class QMinMolecule(QMinBase):
    """
    Custom dictionary for the molecule section

        "comment": str,
        "natom": int,
        "elements": list,
        "unit": str,
        "factor": float,
        "states": list,
        "nstates": int,
        "nmstates": int,
        "point_charges": bool,
        "npc": int,
        "Atomcharge": int,
        "frozcore": int,

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
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
            "point_charges": False,
            "npc": 0,
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
            "point_charges": bool,
            "npc": int,
            "Atomcharge": int,
            "frozcore": int,
        }


class QMinCoords(QMinBase):
    """
    Custom dictionary for the coords section

        "coords": (np.ndarray, list),
        "pccoords": (np.ndarray, list),
        "pccharge": (np.ndarray, list)
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Set data dictionary and dictionary of default types
        self.data = {"coords": None, "pccoords": None, "pccharge": None}
        self.types = {
            "coords": (np.ndarray, list),
            "pccoords": (np.ndarray, list),
            "pccharge": (np.ndarray, list),
        }


class QMinSave(QMinBase):
    """
    Custom dictionary for the save section

        "savedir": str,
        "step": int,
        "previous_step": int,
        "init": bool,
        "newstep": bool,
        "samestep": bool,
        "always_guess": bool,
        "always_orb_init": bool,
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Set data dictionary and dictionary of default types
        self.data = {
            "savedir": os.path.join(os.getcwd(), "SAVE"),
            "step": None,
            "previous_step": None,
            "init": False,
            "newstep": False,
            "samestep": False,
            # Ab initio interfaces
            "always_guess": False,
            "always_orb_init": False,
        }
        self.types = {
            "savedir": str,
            "step": int,
            "previous_step": int,
            "init": bool,
            "newstep": bool,
            "samestep": bool,
            "always_guess": bool,
            "always_orb_init": bool,
        }


class QMinRequests(QMinBase):
    """
    Custom dictionary for the requests section

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
        "multipolar_fit": list,
        "theodore": bool,
        "cleanup": bool,
        "backup": str,
        "molden": bool,
        "savestuff": bool,self.QMin.resources.types[
                                param[0]
                            ](param[1])
        "nooverlap": bool,
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Set data dictionary and dictionary of default types
        self.data = {
            "h": False,
            "soc": False,
            "dm": False,
            "grad": None,
            "nacdr": None,
            "overlap": False,
            "phases": False,
            "ion": False,
            "socdr": False,
            "dmdr": False,
            "multipolar_fit": None,
            "theodore": False,
            # Pseudorequests
            "cleanup": False,
            "backup": None,
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
            "multipolar_fit": list,
            "theodore": bool,
            "cleanup": bool,
            "backup": str,
            "molden": bool,
            "savestuff": bool,
            "nooverlap": bool,
        }


class QMinMaps(QMinBase):
    """
    Custom dictionary for the maps section

        "statemap": dict,
        "mults": set,
        "gsmap": dict,
        "gradmap": set,
        "nacmap": set,
        "densmap": set,
        "ionmap": list,
        "chargemap": dict,
        "multmap": dict,
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
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
            "mults": set,
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

        "pwd": str,
        "cwd": str,
        "scratchdir": str,
        "ncpu": int,
        "ngpu": int,
        "memory": int,
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data = {
            "pwd": os.getcwd(),
            "cwd": os.getcwd(),
            "scratchdir": os.path.join(os.getcwd(), "SCRATCH"),
            "ncpu": 1,
            "ngpu": None,
            "memory": 1000,
        }
        self.types = {
            "pwd": str,
            "cwd": str,
            "scratchdir": str,
            "ncpu": int,
            "ngpu": int,
            "memory": int,
        }


class QMinControl(QMinBase):
    """
    Custom dictionary for the control section


        "jobid": int,
        "workdir": str,
        "master": bool,
        "gradonly": bool,
        "states_to_do": list,
        "jobs": dict,
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data = {
            "jobid": None,
            "workdir": None,
            "master": False,
            "gradonly": False,
            "states_to_do": None,
            "jobs": None,
        }

        self.types = {
            "jobid": int,
            "workdir": str,
            "master": bool,
            "gradonly": bool,
            "states_to_do": list,
            "jobs": dict,
        }


class QMin:
    """
    The QMin object carries all information relevant to the execution of a SHARC interface.
    """

    interface: QMinBase
    molecule: QMinMolecule
    coords: QMinCoords
    save: QMinSave
    requests: QMinRequests
    maps: QMinMaps
    resources: QMinResources
    template: QMinBase
    scheduling: QMinBase
    control: QMinControl

    def __init__(self):
        self.interface = QMinBase()
        self.molecule = QMinMolecule()
        self.coords = QMinCoords()
        self.save = QMinSave()
        self.requests = QMinRequests()
        self.maps = QMinMaps()
        self.resources = QMinResources()
        self.template = QMinBase()
        self.scheduling = QMinBase()
        self.control = QMinControl()

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
{self.control}"""

    def copy(self):
        """
        Return copy of QMin object
        """
        return deepcopy(self)


if __name__ == "__main__":
    pass
