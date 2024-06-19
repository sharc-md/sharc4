import os
from collections import UserDict
from copy import deepcopy
from pyscf import gto

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
            raise TypeError(f"{key} should be of type {self.types[key]} but is {type(value)}")
        self.data[key] = value

    def __getitem__(self, key):
        return self.data[key]

    def __str__(self) -> str:
        return "".join(f"{k}: {v}\n" for k, v in self.data.items())

    def __deepcopy__(self, memo):
        qmin_copy = self.__class__()
        memo[id(self)] = qmin_copy
        for k in self.data.keys():
            k_type = type(self.data[k])
            if self.data[k] is None:
                if k in self.types:
                    qmin_copy.types[k] = self.types[k]
                qmin_copy.data[k] = None
                continue
            qmin_copy.types[k] = k_type
            match k_type.__name__:  # returns the simple name of a type -> list[int] = 'list'
                case "int" | "float" | "bool" | "str":  # immutable data types (ref changes upon change)
                    qmin_copy.data[k] = self.data[k]
                case "list" | "dict" | "tuple":
                    qmin_copy.data[k] = deepcopy(self.data[k], memo)
                case "ndarray" | "Mole":  # use defined copy functions for these types
                    qmin_copy.data[k] = self.data[k].copy()
                case _:
                    qmin_copy.data[k] = deepcopy(self.data[k], memo)
        return qmin_copy


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
        "Ubasis" : ndarray[float,2]

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
            "Ubasis": np.zeros((2, 2)),
            "mol": None,
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
            "Ubasis": np.ndarray,
            "mol": gto.Mole,
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
        "retain": str,
        "molden": bool,
        "savestuff": bool,
        "nooverlap": bool,
        "mol": bool,
        "density_matrices": list,
        "dyson_orbitals" : list,
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
            "retain": 5,
            "molden": False,
            "savestuff": False,
            "nooverlap": False,
            "mol": False,
            "density_matrices": None,
            "dyson_orbitals": None,
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
            "retain": int,
            "molden": bool,
            "savestuff": bool,
            "nooverlap": bool,
            "basis_set": bool,
            "density_matrices": list,
            "dyson_orbitals": list,
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
        "retain": int,
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
            "retain": 5,
        }
        self.types = {
            "pwd": str,
            "cwd": str,
            "scratchdir": str,
            "ncpu": int,
            "ngpu": int,
            "memory": int,
            "retain": int,
        }


class QMinControl(QMinBase):
    """
    Custom dictionary for the control section


        "jobid": int,
        "workdir": str,
        "master": bool,
        "gradonly": bool,
        "densonly": bool,
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
            "densonly": False,
            "states_to_do": None,
            "jobs": None,
        }

        self.types = {
            "jobid": int,
            "workdir": str,
            "master": bool,
            "gradonly": bool,
            "densonly": bool,
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
        self.basis = True

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, k, v):
        setattr(self, k, v)

    # def __contains__(self, key):
    # return self.__dict__.__contains__(key)

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
{self.scheduling if "scheduling" in self.__dict__ else None}
Control: 
{self.control}"""

    def __deepcopy__(self, memo, full: bool = False):
        """
        Return copy of QMin object
        """
        qmin_copy = self.__class__.__new__(self.__class__)
        memo[id(self)] = qmin_copy
        for sub in filter(lambda x: not x.startswith("__"), dir(self)):
            if not full and sub == "scheduling":
                qmin_copy[sub] = QMinBase()
                continue
            qmin_copy[sub] = deepcopy(self[sub], memo)
        return qmin_copy


if __name__ == "__main__":
    pass
