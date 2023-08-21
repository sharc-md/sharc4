import logging
import os
import re
import sys
from collections import UserDict
from typing import List, Union

import numpy as np
from constants import ATOMCHARGE, BOHR_TO_ANG, FROZENS
from error import Error
from utils import itmult, itnmstates, parse_xyz, readfile

__all__ = ["QMin"]

logging.basicConfig(
    format="{levelname:<9s} [{filename}:{funcName}():{lineno}] {message}",
    level=logging.INFO,
    style="{",
)
logger = logging.getLogger(__name__)


def expand_path(path: str) -> str:
    """
    Expand variables in path, error out if variable is not resolvable
    """
    expand = os.path.abspath(os.path.expanduser(os.path.expandvars(path)))
    assert "$" not in expand, f"Undefined env variable in {expand}"
    return expand


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

    def format(self) -> str:
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

    def __init__(
        self,
        debug: bool = False,
        no_print: bool = False,
        persistent: bool = False,
        dry_run: bool = False,
    ):
        self.interface["debug"] = debug
        self.interface["no_print"] = no_print
        self.interface["persistent"] = persistent
        self.interface["dry_run"] = dry_run

        if debug:
            logger.setLevel(logging.DEBUG)
        elif no_print:
            logger.setLevel(logging.WARNING)

        self._setup_mol = False
        self._read_template = False
        self._read_resources = False
        self._setsave = False

        logger.debug("QMin object initialized.")

    def __getitem__(self, key):
        if not self._setup_mol:
            raise Error("QMin not properly initialized. Run setup_mol first!")
        return getattr(self, key)

    def __str__(self):
        return f"""\
Interface: 
{self.interface.format()}
Molecule: 
{self.molecule.format()}
Coords: 
{self.coords.format()}
Save: 
{self.save.format()}
Requests: 
{self.requests.format()}
Maps: 
{self.maps.format()}
Resources: 
{self.resources.format()}
Template: 
{self.template.format()}
Scheduling: 
{self.scheduling.format()}
Control: 
{self.control.format()}
Backwards: 
{self.backwards.format()}"""

    def setup_mol(self, qmin_file: str) -> None:
        """
        Sets up the molecular system from a `QM.in` file.
        parses the elements, states, and savedir and prepare the QMin object accordingly.

        qmin_file:  Path to QM.in file.
        """
        # TODO: option to inizialize from driver and check qmmm
        logger.debug("Seting up molecule from %s", qmin_file)

        if self._setup_mol:
            logger.warning(
                "setup_mol() was already called! Continue setup with %s", qmin_file
            )

        qmin_lines = readfile(qmin_file)
        self.molecule["comment"] = qmin_lines[1]

        try:
            natom = int(qmin_lines[0])
        except ValueError:
            raise Error("first line must contain the number of atoms!", 2)
        if len(qmin_lines) < natom + 4:
            raise Error(
                'Input file must contain at least:\nnatom\ncomment\ngeometry\nkeyword "states"\nat least one task',
                3,
            )
        self.molecule["elements"] = list(
            map(lambda x: parse_xyz(x)[0], (qmin_lines[2 : natom + 2]))
        )
        self.molecule["Atomcharge"] = sum(
            map(lambda x: ATOMCHARGE[x], self.molecule["elements"])
        )
        self.molecule["frozcore"] = sum(
            map(lambda x: FROZENS[x], self.molecule["elements"])
        )
        self.molecule["natom"] = len(self.molecule["elements"])

        # replaces all comments with white space. filters all empty lines
        filtered = filter(
            lambda x: not re.match(r"^\s*$", x),
            map(
                lambda x: re.sub(r"#.*$", "", x),
                qmin_lines[self.molecule["natom"] + 2 :],
            ),
        )

        # naively parse all key argument pairs from QM.in
        for line in filtered:
            llist = line.split(None, 1)
            key = llist[0].lower()
            if key == "states":
                # also does update nmstates, nstates, statemap
                self.parseStates(llist[1])
            elif key == "unit":
                unit = llist[1].strip().lower()
                if unit in ["bohr", "angstrom"]:
                    self.molecule["unit"] = unit
                    self.molecule["factor"] = (
                        1.0 if unit == "bohr" else 1.0 / BOHR_TO_ANG
                    )
                else:
                    raise Error("unknown unit specified", 23)
            elif key == "savedir":
                self._setsave = True
                self.save["savedir"] = llist[1].strip()
                logger.debug("SAVEDIR set to %s", self.save["savedir"])

        if not isinstance(self.save["savedir"], str):
            self.save["savedir"] = "./SAVEDIR/"
            logger.debug("Setting default SAVEDIR")

        self.save["savedir"] = expand_path(self.save["savedir"])

        if not isinstance(self.molecule["unit"], str):
            logger.warning('No "unit" specified in QMin! Assuming Bohr')
            self.molecule["unit"] = "bohr"

        self._setup_mol = True

        logger.debug("Setup successful.")

    def parseStates(self, states: str) -> None:
        """
        Setup states, statemap and everything related
        """
        res = {}
        try:
            res["states"] = list(map(int, states.split()))
        except (ValueError, IndexError):
            # get traceback of currently handled exception
            tb = sys.exc_info()[2]
            raise Error(
                'Keyword "states" has to be followed by integers!', 37
            ).with_traceback(tb)
        reduc = 0
        for i in reversed(res["states"]):
            if i == 0:
                reduc += 1
            else:
                break
        for i in range(reduc):
            del res["states"][-1]
        nstates = 0
        nmstates = 0
        for i in range(len(res["states"])):
            nstates += res["states"][i]
            nmstates += res["states"][i] * (i + 1)
        self.maps["statemap"] = {
            i + 1: [*v] for i, v in enumerate(itnmstates(res["states"]))
        }
        self.molecule["nstates"] = nstates
        self.molecule["nmstates"] = nmstates
        self.molecule["states"] = res["states"]

    def read_resources(self, resources_file: str) -> None:
        """
        Reads a resource file and assigns parameters to
        self.resources. Parameters are only checked by type (if available),
        sanity checks need to be done in specific interface. If multiple entries
        of a parameter with one value are in the file, the latest value will be saved.

        resources_file: Path to resource file.
        """
        logger.debug("Reading resource file %s", resources_file)

        if not self._setup_mol:
            raise Error(
                "Interface is not set up for this template. Call setup_mol with the QM.in file first!",
                23,
            )

        if self._read_resources:
            logger.warning(
                "Resources already read! Overwriting with %s", resources_file
            )

        # Set ncpu from env variables, gets overwritten if in resources
        if "ncpu" in os.environ:
            self.resources["ncpu"] = (
                int(os.getenv("ncpu")) if int(os.getenv("ncpu")) > 0 else 1
            )
            logger.info(
                'Found env variable ncpu=%s, resources["ncpu"] set to %s',
                os.getenv("ncpu"),
                self.resources["ncpu"],
            )

        with open(resources_file, "r", encoding="utf-8") as rcs_file:
            for line in rcs_file:
                # Ignore comments and empty lines
                if re.match(r"^\w+", line):
                    # Remove comments and assign values
                    param = re.sub(r"#.*$", "", line).split()
                    # Expand to fullpath if ~ or $ in string
                    param = [
                        expand_path(x) if re.match(r"\~|\$", x) else x for x in param
                    ]
                    if len(param) == 1:
                        self.resources[param[0]] = True
                    elif len(param) == 2:
                        # Check if savedir already specified in QM.in
                        if param[0] == "savedir":
                            if not self._setsave:
                                self.save["savedir"] = param[1]
                                logger.debug("SAVEDIR set to %s", self.save["savedir"])
                            else:
                                logger.info(
                                    "SAVEDIR is already set and will not be overwritten!"
                                )
                            continue
                        # Cast to correct type if available
                        if param[0] in self.resources.keys():
                            self.resources[param[0]] = self.resources.types[param[0]](
                                param[1]
                            )
                        else:
                            self.resources[param[0]] = param[1]
                    else:
                        # If key already exists extend list with values
                        if (
                            param[0] in self.resources.keys()
                            and self.resources[param[0]]
                        ):
                            self.resources[param[0]].extend(list(param[1:]))
                        else:
                            self.resources[param[0]] = list(param[1:])
        self._read_resources = True

    def read_template(self, template_file: str) -> None:
        """
        Reads a template file and assigns parameters to
        self.template. No sanity checks at all, has to be done
        in the interface. If multiple entries
        of a parameter with one value are in the file, the latest value will be saved.

        template_file:  Path to template file
        """
        logger.debug("Reading template file %s", template_file)

        if self._read_template:
            logger.warning("Template already read! Overwriting with %s", template_file)

        with open(template_file, "r", encoding="utf-8") as tmpl_file:
            for line in tmpl_file:
                # Ignore comments and empty lines
                if re.match(r"^\w+", line):
                    # Remove comments and assign values
                    param = re.sub(r"#.*$", "", line).split()
                    if len(param) == 1:
                        self.template[param[0]] = True
                    elif len(param) == 2:
                        self.template[param[0]] = param[1]
                    else:
                        self.template[param[0]] = list(param[1:])

        # Check if charge is set and generate chargemap and ionmap
        if "charge" not in self.template.keys():
            self.template["charge"] = [
                i % 2 for i in range(len(self.molecule["states"]))
            ]
            logger.info(
                "charge not specified setting default, %s", self.template["charge"]
            )

        if "paddingstates" not in self.template.keys():
            self.template["paddingstates"] = [0 for _ in self.molecule["states"]]
            logger.info(
                "paddingstates not specified setting default, %s",
                self.template["paddingstates"],
            )

        logger.debug("Setting up maps")
        self.maps["chargemap"] = {
            idx + 1: int(chrg) for (idx, chrg) in enumerate(self.template["charge"])
        }

        self.control["states_to_do"] = [
            v + int(self.template["paddingstates"][i]) if v > 0 else v
            for i, v in enumerate(self.molecule["states"])
        ]

        if "unrestricted_triplets" not in self.template.keys():
            if len(self.molecule["states"]) >= 3 and self.molecule["states"][2] > 0:
                self.control["states_to_do"][0] = max(self.molecule["states"][0], 1)
                req = max(self.molecule["states"][0] - 1, self.molecule["states"][2])
                self.control["states_to_do"][0] = req + 1
                self.control["states_to_do"][2] = req

        jobs = {}
        if self.control["states_to_do"][0] > 0:
            jobs[1] = {"mults": [1], "restr": True}
        if (
            len(self.control["states_to_do"]) >= 2
            and self.control["states_to_do"][1] > 0
        ):
            jobs[2] = {"mults": [2], "restr": False}
        if (
            len(self.control["states_to_do"]) >= 3
            and self.control["states_to_do"][2] > 0
        ):
            if (
                "unrestricted_triplets" not in self.template.keys()
                and self.control["states_to_do"][0] > 0
            ):
                jobs[1]["mults"].append(3)
            else:
                jobs[3] = {"mults": [3], "restr": False}
        self.maps["mults"] = jobs

        if len(self.control["states_to_do"]) >= 4:
            for imult, nstate in enumerate(self.control["states_to_do"][3:]):
                if nstate > 0:
                    # jobs[len(jobs)+1]={'mults':[imult+4],'restr':False}
                    jobs[imult + 4] = {"mults": [imult + 4], "restr": False}

        multmap = {}
        for ijob, job in jobs.items():
            for imult in job["mults"]:
                multmap[imult] = ijob
            multmap[-(ijob)] = job["mults"]
        multmap[1] = 1
        self.maps["multmap"] = multmap

        ionmap = []
        for m1 in itmult(self.molecule["states"]):
            job1 = multmap[m1]
            el1 = self.maps["chargemap"][m1]
            for m2 in itmult(self.molecule["states"]):
                if m1 >= m2:
                    continue
                job2 = multmap[m2]
                el2 = self.maps["chargemap"][m2]
                # print m1,job1,el1,m2,job2,el2
                if abs(m1 - m2) == 1 and abs(el1 - el2) == 1:
                    ionmap.append((m1, job1, m2, job2))
        self.maps["ionmap"] = ionmap

        gsmap = {}
        for i in range(self.molecule["nmstates"]):
            m1, s1, ms1 = tuple(self.maps["statemap"][i + 1])
            gs = (m1, 1, ms1)
            job = multmap[m1]
            if m1 == 3 and jobs[job]["restr"]:
                gs = (1, 1, 0.0)
            for j in range(self.molecule["nmstates"]):
                m2, s2, ms2 = tuple(self.maps["statemap"][j + 1])
                if (m2, s2, ms2) == gs:
                    break
            gsmap[i + 1] = j + 1
        self.maps["gsmap"] = gsmap
        self._read_template = True

    def set_coords(self, xyz: Union[str, List, np.ndarray]) -> None:
        """
        Sets coordinates, qmmm and pccharge from file or list/array
        xyz: path to xyz file or list/array with coords
        """
        if isinstance(xyz, str):
            lines = readfile(xyz)
            try:
                natom = int(lines[0])
            except ValueError:
                raise Error("first line must contain the number of atoms!", 2)
            self.coords["coords"] = (
                np.asarray([parse_xyz(x)[1] for x in lines[2 : natom + 2]], dtype=float)
                * self.molecule["factor"]
            )
        elif isinstance(xyz, (list, np.ndarray)):
            self.coords["coords"] = np.asarray(xyz) * self.molecule["factor"]
        else:
            raise NotImplementedError(
                "'set_coords' is only implemented for str, list[list[float]] or numpy.ndarray type"
            )
        # TODO: Implement qmmm and pccharge

    def read_requests(self, requests_file: str = "QM.in") -> None:
        """
        Reads QM.in file and parses requests
        """
        # TODO: pc file? densmap only for multipolar fit?
        assert (
            self._read_template
        ), "Interface is not set up correctly. Call read_template with the .template file first!"
        assert (
            self._read_resources
        ), "Interface is not set up correctly. Call read_resources with the .resources file first!"

        logger.debug("Reading requests from %s", requests_file)

        # Reset requests
        self.requests = QMinRequests()
        self.save["init"] = False
        self.save["samestep"] = False
        self.save["newstep"] = False
        self.save["restart"] = False

        # Parse QM.in and setup request dict
        with open(requests_file, "r", encoding="utf-8") as requests:
            # Skip xyz part
            atoms = next(requests)
            for _ in range(int(atoms) + 1):
                next(requests)

            nac_select = False

            for line in requests:
                # Check for valid keywords, remove comments
                if re.match(r"^\w", line):
                    params = re.sub(r"#.*$", "", line).split()

                    # Parse NACDR if requested
                    if params[0].casefold() == "nacdr":
                        logger.debug("Parsing request %s", params)
                        if len(params) > 1 and params[1].casefold() == "select":
                            nac_select = True
                        else:
                            self.requests["nacdr"] = ["all"]
                        continue
                    if nac_select:
                        if params[0].casefold() == "end":
                            nac_select = False
                        else:
                            assert (
                                len(params) == 2
                            ), "NACs have to be given in state pairs!"
                            logger.debug("Adding state pair %s to NACDR list", params)
                            self.requests["nacdr"].append(params)
                        continue

                    # Parse every other request
                    if params[0].casefold() in (
                        *self.requests.keys(),
                        "init",
                        "samestep",
                        "restart",
                        "newstep",
                    ):
                        logger.debug("Parsing request %s", params)
                        self._set_requests(params)

            assert not nac_select, "No end keyword found after nacdr select!"
        self._step_logic()
        self._request_logic()

        if self.requests["backup"]:
            logger.debug("Setting up backup directories")

    def _request_logic(self) -> None:
        """
        Checks for conflicting options, generates requested maps
        and sets path variables according to requests
        """
        logger.debug("Starting request logic")

        if not os.path.exists(self.save["savedir"]):
            logger.debug("Creating savedir %s", self.save["savedir"])
            os.mkdir(self.save["savedir"])

        if self.requests["phases"] and not self.requests["overlap"]:
            logger.info("Found phases in requests, set overlap to true")
            self.requests["overlap"] = True

        if (
            self.requests["ion"] or self.requests["overlap"]
        ) and self.__class__.__name__ != "LVC":
            assert os.path.isfile(
                self.resources["wfoverlap"]
            ), "Missing path to wfoverlap.x in resources file!"

        assert not (
            self.requests["overlap"] and self.save["init"]
        ), '"overlap" and "phases" cannot be calculated in the first timestep! Delete either "overlap" or "init"'

        if self.requests["theodore"]:
            assert os.path.isdir(
                self.resources["theodir"]
            ), "Give path to the TheoDORE installation directory in resources file!"
            os.environ["THEODIR"] = self.resources["theodir"]
            os.environ["PYTHONPATH"] += (
                os.pathsep
                + os.path.join(self.resources["theodir"], "lib")
                + os.pathsep
                + self.resources["theodir"]
            )
            self.resources["theodore_n"] = (
                len(self.resources["theodore_prop"])
                + len(self.resources["theodore_fragment"]) ** 2
            )

        # Setup gradmap, densmap and nacmap
        gradmap = set()
        if isinstance(self.requests["grad"], list):
            logger.debug("Building gradmap")
            gradmap = set(
                {tuple(self.maps["statemap"][i][0:2]) for i in self.requests["grad"]}
            )
        self.maps["gradmap"] = gradmap

        densmap = set()
        if isinstance(self.requests["multipolar_fit"], list):
            logger.debug("Building densmap")
            densmap = set(
                {
                    tuple(self.maps["statemap"][i][0:2])
                    for i in self.requests["multipolar_fit"]
                }
            )
        self.maps["densmap"] = densmap

        nacmap = set()
        if len(self.requests["nacdr"]) > 0 and self.requests["nacdr"][0] != "all":
            logger.debug("Building nacmap")
            for i in self.requests["nacdr"]:
                s1 = self.maps["statemap"][int(i[0])]
                s2 = self.maps["statemap"][int(i[1])]
                if s1[0] != s2[0] or s1 == s2:
                    continue
                if s1[1] > s2[1]:
                    continue
                nacmap.add(tuple(s1 + s2))
        self.maps["nacmap"] = nacmap

    def _step_logic(self) -> None:
        """
        Performs step logic
        """
        logger.debug("Starting step logic")

        # TODO: implement previous_step from driver
        last_step = None
        stepfile = os.path.join(self.save["savedir"], "STEP")
        if os.path.isfile(stepfile):
            logger.debug("Found stepfile %s", stepfile)
            last_step = int(readfile(stepfile)[0])

        if not self.save["step"]:
            if last_step:
                self.save["newstep"] = True
                self.save["step"] = last_step + 1
            else:
                self.save["init"] = True
                self.save["step"] = 0
            return

        if not last_step:
            assert (
                self.save["step"] == 0
            ), f'Specified step ({self.save["step"]}) could not be restarted from!\nCheck your savedir and "STEP" file in {self.save["savedir"]}'
            self.save["init"] = True
        elif self.save["step"] == -1:
            self.save["newstep"] = True
            self.save["step"] = last_step + 1
        elif self.save["step"] == last_step:
            self.save["samestep"] = True
        elif self.save["step"] == last_step + 1:
            self.save["newstep"] = True
        else:
            raise Error(
                f'Determined last step ({last_step}) from savedir and specified step ({self.save["step"]}) do not fit!\nPrepare your savedir and "STEP" file accordingly before starting again or choose "step -1" if you want to proceed from last successful step!'
            )

    def _set_requests(self, request: list) -> None:
        """
        Setup requests and do basic sanity checks
        """
        if request[0].casefold() in self.requests.keys():
            if request[0].casefold() == "h" and len(request) == 1:
                self.requests["h"] = True
            elif request[0].casefold() == "grad":
                if len(request) > 1 and request[1].casefold() != "all":
                    self.requests["grad"] = [int(i) for i in request[1:]]
                    return
                self.requests["grad"] = [
                    i + 1 for i in range(self.molecule["nmstates"])
                ]
            elif request[0].casefold() == "soc":
                if sum(i > 0 for i in self.molecule["states"]) < 2:
                    logger.warning(
                        "SOCs requestet but only 1 multiplicity given! Disable SOCs"
                    )
                    return
                self.requests["soc"] = True
            elif request[0].casefold() == "multipolar_fit":
                if len(request > 1):
                    self.requests["multipolar_fit"] = sorted(request[1:])
                    return
                self.requests["multipolar_fit"] = [
                    i + 1 for i in range(self.molecule["nmstates"])
                ]
            else:
                self.requests[request[0].casefold()] = True
        else:
            self.save[request[0].casefold()] = True

    def setup_run(self) -> None:
        pass

    def run(self):
        pass

    def get_QMout(self):
        pass

    def finalize(self):
        pass


if __name__ == "__main__":
    qmin = QMin(debug=True)
    interface = "MOLCAS"
    qmin.setup_mol(
        f"/user/sascha/development/eci/sharc_main/examples/SHARC_{interface}/QM.in"
        # "/user/sascha/development/eci/densitytest/gaussian/QM.in"
        #"/user/sascha/development/eci/soctest"
    )
    qmin.read_resources(
        f"/user/sascha/development/eci/sharc_main/examples/SHARC_{interface}/{interface}.resources"
        # "/user/sascha/development/eci/densitytest/gaussian/GAUSSIAN.resources"
    )
    qmin.read_template(
        f"/user/sascha/development/eci/sharc_main/examples/SHARC_{interface}/{interface}.template"
        # "/user/sascha/development/eci/densitytest/gaussian/GAUSSIAN.template"
    )
    qmin.read_requests(
        f"/user/sascha/development/eci/sharc_main/examples/SHARC_{interface}/QM.in"
        #"/user/sascha/development/eci/soctest"
    )
    print(qmin.requests.format())
    print(qmin.maps.format())
    print(qmin.save.format())
    # print(qmin)
