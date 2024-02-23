#!/usr/bin/env python3

# ******************************************
#
#    SHARC Program Suite
#
#    Copyright (c) 2019 University of Vienna
#
#    This file is part of SHARC.
#
#    SHARC is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    SHARC is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    inside the SHARC manual.  If not, see <http://www.gnu.org/licenses/>.
#
# ******************************************


# IMPORTS
# external
import ast
import os
import re
import sys
from abc import ABC, abstractmethod
from datetime import date
from io import TextIOWrapper
from socket import gethostname
from textwrap import wrap
from typing import Any
from dataclasses import dataclass

import numpy as np

# internal
from constants import ATOMCHARGE, BOHR_TO_ANG, FROZENS
from logger import SHARCPRINT, TRACE, CustomFormatter, logging
from qmin import QMin
from qmout import QMout
from utils import clock, expand_path, itnmstates, parse_xyz, readfile, writefile, electronic_state

np.set_printoptions(linewidth=400, formatter={"float": lambda x: f"{x: 9.7}"})
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


class SHARC_INTERFACE(ABC):
    """
    Abstract Base Class for SHARC interfaces

    persistent:     Changes from stateless to statefull
    logname:        Name of the logger
    logfile:        Filename for logger output
    loglevel:       Set loglevel
    """

    # internal status indicators    # are these needed?
    _setup_mol = False
    _read_resources = False
    _read_template = False
    _states = None
    density_recipes = None
    _DEBUG = False
    _PRINT = True

    # TODO: set Debug and Print flag
    # TODO: set persistant flag for file-io vs in-core

    def __init__(
        self,
        persistent=False,
        logname: str | None = None,
        logfile: str | None = None,
        loglevel: int = logging.root.level,
    ):
        # all the output from the calculation will be stored here
        self.QMout = QMout()
        self.clock = clock()
        self.persistent = persistent
        self.QMin = QMin()
        self.density_recipes = {}
        self._setsave = False

        logname = logname if isinstance(logname, str) else self.name()
        self.log = logging.getLogger(logname)
        self.log.propagate = False
        self.log.handlers = []
        self.log.setLevel(loglevel)
        hdlr = (
            logging.FileHandler(filename=logfile, mode="w", encoding="utf-8")
            if isinstance(logfile, str)
            else logging.StreamHandler(sys.stdout)
        )
        hdlr._name = logname + "Handler"
        hdlr.setFormatter(CustomFormatter())

        self.log.addHandler(hdlr)
        self.log.print = self.sharcprint
        self.log.trace = self.trace

    def sharcprint(self, msg, *args, **kwargs):
        """
        Log 'msg % args' with severity 'SHARCPRINT'.

        To pass exception information, use the keyword argument exc_info with
        a true value, e.g.
        """
        self.log.log(SHARCPRINT, msg, *args, **kwargs)

    def trace(self, msg, *args, format=True, **kwargs):
        """
        Log 'msg % args' with severity 'SHARCPRINT'.

        use to log extensive runtime information (this is even lower than DEBUG)

        To pass exception information, use the keyword argument exc_info with
        a true value, e.g.
        """
        if not format:
            kwargs.update({"extra": {"simple": True}})
        self.log.log(TRACE, msg, *args, **kwargs)

    @staticmethod
    @abstractmethod
    def authors() -> str:
        """
        Return authors of interface
        """
        return "Severin Polonius, Sebastian Mai"

    @staticmethod
    @abstractmethod
    def version() -> str:
        """
        Return version of interface
        """
        return "4.0"

    @staticmethod
    @abstractmethod
    def versiondate() -> date:
        """
        Return creation date of interface
        """
        return date(2021, 7, 15)

    @staticmethod
    @abstractmethod
    def name() -> str:
        """
        Return name of interface
        """
        return "base"

    @staticmethod
    @abstractmethod
    def description() -> str:
        """
        Return interface description
        """
        return "Abstract base class for SHARC interfaces."

    @staticmethod
    @abstractmethod
    def changelogstring() -> str:
        """
        Return changelog of interface
        """
        return "This is the changelog string"

    @abstractmethod
    def get_features(self, KEYSTROKES: TextIOWrapper | None = None) -> set:
        """return availble features

        ---
        Parameters:
        KEYSTROKES: object as returned by open() to be used with question()
        """
        return all_features

    @abstractmethod
    def get_infos(self, INFOS: dict, KEYSTROKES: TextIOWrapper | None = None) -> dict:
        """communicate requests from setup and asks for additional paths or info

        The `INFOS` dict holds all global informations like paths to programs
        and requests in `INFOS['needed_requests']`

        all interface specific information like additional files etc should be stored
        in the interface intance itself.

        use the `question()` function from the `utils` module and write the answers
        into `KEYSTROKES`

        Parameters:
        ---
        INFOS
            dict[str]: dictionary with all previously collected infos during setup
        KEYSTROKES
            str: object as returned by open() to be used with question()
        """
        return INFOS

    @abstractmethod
    def prepare(self, INFOS: dict, dir_path: str):
        """
        prepares the folder for an interface calculation

        Parameters
        ----------
        INFOS
            dict[str]: dictionary with all infos from the setup script
        dir_path
            str: *relative* path to the directory to setup (can be appended to `scratchdir`)
        """
        return

    def print_qmin(self) -> None:
        """
        Print contents of QMin object
        """
        self.log.info(f"{self.QMin}")

    def main(self) -> None:
        """
        main routine for all interfaces.
        This routine containes all functions that will be accessed when any interface is calculating
        a single point. All of these functions have to be defined in the derived class if not
        available in this base class.
        """

        args = sys.argv
        self.clock = clock()
        self.printheader()
        if len(args) != 2:
            self.log.info(
                f"Usage:,\n./SHARC_{self.name()}.py <QMin>\nversion: {self.version()}\ndate: {self.versiondate():%d.%m.%Y}\nchangelog: {self.changelogstring()}"
            )
            sys.exit(1)
        QMinfilename = sys.argv[1]

        # --- the following are called once inside a driver ---
        # set up the system (i.e. molecule, states, unit...)
        self.setup_mol(QMinfilename)
        # read in the resources available for this computation (program path, cores, memory)
        self.read_resources(f"{self.name()}.resources")
        # read in the specific template file for the interface with all keywords
        self.read_template(f"{self.name()}.template")
        # setup internal state for the computation
        self.setup_interface()

        # --- the following are called per time step inside a driver ---
        # read the property requests that have to be calculated
        self.read_requests(QMinfilename)
        # set the coordinates of the molecular system
        self.set_coords(QMinfilename)
        # print qmin
        self.print_qmin()
        # perform the calculation and parse the output, do subsequent calculations with other tools
        self.run()
        # get output as requested
        self.getQMout()

        # Remove old data
        self.clean_savedir(self.QMin.save["savedir"], self.QMin.requests["retain"], self.QMin.save["step"])

        # writes a STEP file in the SAVEDIR (marks this step as succesfull)
        self.write_step_file()

        # printing and output generation
        self.log.info(self.formatQMout())
        self.QMout["runtime"] = self.clock.measuretime(log=self.log.info)
        self.writeQMout(filename=QMinfilename)

    @abstractmethod
    def read_template(self, template_file: str, kw_whitelist: list[str] | None = None) -> None:
        """
        Reads a template file and assigns parameters to
        self.QMin.template. No sanity checks at all, has to be done
        in the interface. If multiple entries
        of a parameter with one value are in the file, the latest value will be saved.

        template_file:  Path to template file
        """
        self.log.debug(f"Reading template file {template_file}")

        if self._read_template:
            self.log.warning(f"Template already read! Overwriting with {template_file}")

        self.QMin.template.update(self._parse_raw(template_file, self.QMin.template.types, kw_whitelist))

        self._read_template = True
        if "charge" not in self.QMin.template or self.QMin.template["charge"] is None:
            charge = [i % 2 for i in range(len(self.QMin.molecule['states']))]
        else:
            charge = self.QMin.template["charge"]
        
        # Start TOMI
        self.states = []
        for S, nstates in enumerate(self.QMin.molecule['states']):
            c = charge[S]
            for N in range(nstates):
                for M in range(-S, S + 1, 2):
                    self.states.append(electronic_state( Z=c, S=S, M=M, N=N+1, C={} ) )  # This is the moment in which states get their pointers
        # End TOMI
        

    @staticmethod
    def clean_savedir(path: str, retain: int, step: int) -> None:
        """
        Remove older files than step-retain

        path:       Path to savedir
        retain:     Number of timesteps to keep (-1 = all)
        step:       Current step
        """

    @abstractmethod
    def run(self) -> None:
        """
        Do request & other logic and calculations here
        """

    @abstractmethod
    def setup_interface(self) -> None:
        """
        Prepare the interface for calculations
        """

    @abstractmethod
    def getQMout(self) -> dict[str, np.ndarray]:
        """
        Return QMout object
        """

    #@abstractmethod
    #def dyson_orbitals_with_other(self,other):
    #    """
    #    Calculates Dyson orbitals between self and other.
    #    Presumably it will be implemented in SHARC_ABINITIO subclass and in each individual FAST or HYBRID interface
    #    """

    @abstractmethod
    def create_restart_files(self) -> None:
        """
        Create restart files
        """

    def set_coords(self, xyz: str | list | np.ndarray, pc: bool = False) -> None:
        """
        Sets coordinates, qmmm and pccharge from file or list/array
        xyz: path to xyz file or list/array with coords
        pc: Set point charge coordinates
        """
        key = "coords" if not pc else "pccoords"
        if isinstance(xyz, str):
            lines = readfile(xyz)
            try:
                natom = int(lines[0])
            except ValueError as error:
                raise ValueError("first line must contain the number of atoms!") from error
            self.QMin.coords[key] = (
                np.asarray([parse_xyz(x)[1] for x in lines[2 : natom + 2]], dtype=float) * self.QMin.molecule["factor"]
            )
        elif isinstance(xyz, (list, np.ndarray)):
            self.QMin.coords[key] = np.asarray(xyz) * self.QMin.molecule["factor"]
        else:
            raise NotImplementedError("'set_coords' is only implemented for str, list[list[float]] or numpy.ndarray type")

    def setup_mol(self, qmin_file: str) -> None:
        """
        Sets up the molecular system from a `QM.in` file.
        parses the elements, states, and savedir and prepare the QMin object accordingly.

        qmin_file:  Path to QM.in file.
        """
        self.log.debug(f"Setting up molecule from {qmin_file}")

        if self._setup_mol:
            self.log.warning(
                f"setup_mol() was already called! Continue setup with {qmin_file}",
            )

        qmin_lines = readfile(qmin_file)
        self.QMin.molecule["comment"] = qmin_lines[1]

        try:
            natom = int(qmin_lines[0])
        except ValueError as err:
            raise ValueError("first line must contain the number of atoms!") from err

        self.QMin.molecule["elements"] = list(map(lambda x: parse_xyz(x)[0], (qmin_lines[2 : natom + 2])))
        self.QMin.molecule["Atomcharge"] = sum(map(lambda x: ATOMCHARGE[x], self.QMin.molecule["elements"]))
        # self.QMin.molecule["frozcore"] = sum(map(lambda x: FROZENS[x], self.QMin.molecule["elements"]))
        self.QMin.molecule["frozcore"] = 0
        self.QMin.molecule["natom"] = len(self.QMin.molecule["elements"])

        # replaces all comments with white space. filters all empty lines
        filtered = filter(
            lambda x: not re.match(r"^\s*$", x),
            map(
                lambda x: re.sub(r"#.*$", "", x),
                qmin_lines[self.QMin.molecule["natom"] + 2 :],
            ),
        )

        # naively parse all key argument pairs from QM.in
        for line in filtered:
            llist = line.split(None, 1)
            key = llist[0].lower()
            if key == "states":
                # also does update nmstates, nstates, statemap
                states_dict = self.parseStates(llist[1])
                if len(states_dict["states"]) < 1:
                    self.log.error("Number of states must be > 0!")
                    raise ValueError()
                self.QMin.maps["statemap"] = states_dict["statemap"]
                self.QMin.molecule["nstates"] = states_dict["nstates"]
                self.QMin.molecule["nmstates"] = states_dict["nmstates"]
                self.QMin.molecule["states"] = states_dict["states"]

            elif key == "unit":
                unit = llist[1].strip().lower()
                if unit in ["bohr", "angstrom"]:
                    self.QMin.molecule["unit"] = unit
                    self.QMin.molecule["factor"] = 1.0 if unit == "bohr" else 1.0 / BOHR_TO_ANG
                else:
                    raise ValueError("unknown unit specified")
            elif key == "savedir":
                self._setsave = True
                self.QMin.save["savedir"] = llist[1].strip()
                self.log.debug(f"SAVEDIR set to {self.QMin.save['savedir']}")
            elif key == "point_charges":
                self.QMin.molecule["point_charges"] = True
                pcfile = expand_path(llist[1].strip())
                self.log.debug(f"Reading point charges from {pcfile}")

                # Read pcfile and assign charges and coordinates
                pccharge = []
                pccoords = []
                for pcharges in map(lambda x: x.split(), readfile(pcfile)):
                    pccoords.append(
                        [
                            float(pcharges[0]) * self.QMin.molecule["factor"],
                            float(pcharges[1]) * self.QMin.molecule["factor"],
                            float(pcharges[2]) * self.QMin.molecule["factor"],
                        ]
                    )
                    pccharge.append(float(pcharges[3]))

                self.QMin.coords["pccoords"] = pccoords
                self.QMin.coords["pccharge"] = pccharge
                self.QMin.molecule["npc"] = len(pccharge)

        if self.QMin.molecule["factor"] is None:
            self.log.warning("No Unit specified assuming Angstrom!")
            self.QMin.molecule["factor"] = 1.0 / BOHR_TO_ANG
            self.QMin.molecule["unit"] = "angstrom"

        if not isinstance(self.QMin.save["savedir"], str):
            self.QMin.save["savedir"] = "./SAVEDIR/"
            self.log.debug("Setting default SAVEDIR")

        self.QMin.save["savedir"] = expand_path(self.QMin.save["savedir"])

        if not isinstance(self.QMin.molecule["unit"], str):
            self.log.warning('No "unit" specified in QMin! Assuming Bohr')
            self.QMin.molecule["unit"] = "bohr"

        if all((val is None for val in self.QMin.molecule.values())):
            raise ValueError(
                """Input file must contain at least:
                natom
                comment
                geometry
                keyword "states"
                at least one task"""
            )

        self._setup_mol = True

        self.log.debug("Setup successful.")

    def parseStates(self, states: str) -> dict[str, Any]:
        """
        Setup states, statemap and everything related
        """
        res = {}
        try:
            res["states"] = list(map(int, states.split()))
        except (ValueError, IndexError) as err:
            raise ValueError('Keyword "states" has to be followed by integers!', 37) from err
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
        if any(map(lambda x: x < 0, res["states"])):
            raise ValueError("States must be positive numbers!")
        for i in range(len(res["states"])):
            nstates += res["states"][i]
            nmstates += res["states"][i] * (i + 1)
        return {
            "nstates": nstates,
            "nmstates": nmstates,
            "states": res["states"],
            "statemap": {i + 1: [*v] for i, v in enumerate(itnmstates(res["states"]))},
        }

    @abstractmethod
    def read_resources(self, resources_file: str, kw_whitelist: list[str] | None = None) -> None:
        """
        Reads a resource file and assigns parameters to
        self.QMin.resources. Parameters are only checked by type (if available),
        sanity checks need to be done in specific interface. If multiple entries
        of a parameter with one value are in the file, the latest value will be saved.

        resources_file: Path to resource file.
        kw_whitelist:   Whitelist for keywords (with multiple values) that do not get
                        overwritten when keyword multiple times in resources_file,
                        instead the list will be extended
        """
        self.log.debug(f"Reading resource file {resources_file}")
        kw_whitelist = [] if not kw_whitelist else kw_whitelist

        if not self._setup_mol:
            raise RuntimeError("Interface is not set up for this template. Call setup_mol first!")

        if self._read_resources:
            self.log.warning(f"Resources already read! Overwriting with {resources_file}")

        # Set ncpu from env variables, gets overwritten if in resources
        priority_order = ["SLURM_NTASKS_PER_NODE", " NSLOTS"]
        for prio in priority_order:
            if prio in os.environ:
                self.QMin.resources["ncpu"] = max(1, int(os.environ[prio]))
                self.log.info(
                    f'Found env variable ncpu={os.environ[prio]}, resources["ncpu"] set to {self.QMin.resources["ncpu"]}',
                )
                break

        raw_dict = self._parse_raw(
            resources_file,
            {**self.QMin.resources.types, "savedir": str, "always_guess": bool, "always_orb_init": bool},
            kw_whitelist,
        )

        if "savedir" in raw_dict:
            if not self._setsave:
                self.QMin.save["savedir"] = expand_path(raw_dict["savedir"])
                self.log.debug(
                    f"SAVEDIR set to {self.QMin.save['savedir']}",
                )
            else:
                self.log.info("SAVEDIR is already set and will not be overwritten!")
            del raw_dict["savedir"]
        if "always_guess" in raw_dict:
            self.QMin.save["always_guess"] = True
            del raw_dict["always_guess"]
        if "always_orb_init" in raw_dict:
            self.QMin.save["always_orb_init"] = True
            del raw_dict["always_orb_init"]
        self.QMin.resources.update(raw_dict)

        self._read_resources = True

    def _parse_raw(self, file: str, types_dict: dict, kw_whitelist=None) -> dict:
        """
        parse the content of a keyword-argument file (.resources, .template)

        Args:
            file: file to parse from
            types_dict: dictionary with keywords and their respective types
            kw_whitelist list: list with keywords that should be appended upon multiple encounters

        Raises:
            RuntimeError:

        Returns:

        """
        kw_whitelist = [] if not kw_whitelist else kw_whitelist

        lines = readfile(file)
        # replaces all comments with white space. filters all empty lines
        filtered = filter(lambda x: not re.match(r"^\s*$", x), map(lambda x: re.sub(r"#.*$", "", x).strip(), lines))
        file_str = "\n".join(filtered)
        if not file_str or file_str.isspace():  # check if there is only whitespace left!
            return {}

        # concat all lines for select keyword:
        # 1 join lines to full file string,
        # 2 match all select/start ... end blocks,
        # 3 replace all \n with ',' in the matches,
        # 4 return matches between [' and ']
        def format_match(x: re.Match) -> str:
            return x.group(3) + " " + re.sub(r"\n+", "','", "['{}']".format(x.group(4)))

        lines = re.sub(r"(s(elect|tart)\s+)(.+)\n([\w\d\s\n]*)(\nend)", format_match, file_str).split("\n")
        # Store all encountered keywords to warn for duplicates
        keyword_list = set()
        out_dict = {}
        for line in lines:
            # assign values
            param = line.split(maxsplit=1)
            if param[0] in keyword_list and param[0] not in kw_whitelist:
                self.log.warning(f"Multiple entries of {param[0]} in {file}")

            keyword_list.add(param[0])

            match param:
                case [key] if key in types_dict:
                    key_type = types_dict[key]
                    if key_type is bool:
                        out_dict[key] = True
                    else:
                        self.log.error(f"resources keyword '{key}' is type {key_type} but has no value!")
                        raise RuntimeError()

                case [key, val] if key in types_dict:
                    key_type = types_dict[key]
                    if isinstance(key_type, tuple):
                        key_type, _ = key_type
                    if key_type is list:
                        if key not in out_dict or key not in kw_whitelist:
                            out_dict[key] = []
                        if val[0] == "[":
                            raw_value = ast.literal_eval(val)
                            # check if matrix
                            if isinstance(raw_value[0], str):
                                raw_value = [
                                    entry if len(entry) > 1 else entry[0] for entry in map(lambda x: x.split(), raw_value)
                                ]
                        else:
                            raw_value = val.split()
                        out_dict[key].append(raw_value)
                    elif key_type is str:
                        out_dict[key] = expand_path(val) if re.match(r"\~|\$", val) else val
                    elif key_type is tuple:
                        out_dict[key] = (v for v in val)
                    elif key_type is bool:
                        if isinstance(val, str):
                            if val.lower() == "false":
                                out_dict[key] = False
                            elif val.lower() == "true":
                                out_dict[key] = True
                            else:
                                raise ValueError(f"Boolian value for '{key}': {val} cannot be interpreted as a Boolian!")
                    else:
                        out_dict[key] = key_type(val)

                case _:
                    self.log.warning(f"'{param[0]}' not in {file.split('.')[-1]} keywords: {', '.join(types_dict.keys())}")
                    # raise ValueError(f"'{param[0]}' not in {file.split('.')[-1]} keywords: {', '.join(types_dict.keys())}")

        # sanitize lists:
        for key, key_type in types_dict.items():
            if key in out_dict and key_type == list and len(out_dict[key]) == 1 and isinstance(out_dict[key][0], list):
                out_dict[key] = out_dict[key][0]

        return out_dict

    def read_requests(self, requests_file: str = "QM.in") -> None:
        """
        Reads QM.in file and parses requests
        """
        # TODO: pc file? densmap only for multipolar fit?
        assert self._read_template, "Interface is not set up correctly. Call read_template with the .template file first!"
        assert self._read_resources, "Interface is not set up correctly. Call read_resources with the .resources file first!"

        self.log.debug(f"Reading requests from {requests_file}")

        # Reset requests
        self.QMin.requests = QMin().requests
        self.QMin.save["init"] = False
        self.QMin.save["samestep"] = False
        self.QMin.save["newstep"] = False
        self.QMin.save["restart"] = False

        # read file and skip geometry
        lines = readfile(requests_file)[self.QMin.molecule["natom"] + 2 :]
        # replaces all comments with white space. filters all empty lines
        filtered = filter(lambda x: not re.match(r"^\s*$", x), map(lambda x: re.sub(r"#.*$", "", x).strip(), lines))
        file_str = "\n".join(filtered)
        if not file_str or file_str.isspace():  # check if there is only whitespace left!
            return {}

        # concat all lines for select keyword:
        # 1 join lines to full file string,
        # 2 match all select/start ... end blocks,
        # 3 replace all \n with ',' in the matches,
        # 4 return matches between [' and ']
        def format_match(x: re.Match) -> str:
            return x.group(3) + " " + re.sub(r"\n+", "','", "['{}']".format(x.group(4)))

        lines = re.sub(r"(s(elect|tart)\s+)(.+)\n([\w\d\s\n]*)(\nend)", format_match, file_str).split("\n")

        for line in lines:
            match line.lower().split(maxsplit=1):
                case [key] if key in (*self.QMin.requests.keys(), "step"):
                    self.log.debug(f"Parsing request {key}")
                    self._set_request((key, None))
                case ["select" | "start", key]:
                    self.log.error(f"line with '{line}' found but no 'end' keyword!")
                    raise ValueError(f"line with '{line}' found but no 'end' keyword!")
                case [key, val] if key in (*self.QMin.requests.keys(), "step"):
                    self.log.debug(f"Parsing request {key} {val}")
                    if val[0] == "[":
                        raw_value = ast.literal_eval(val)
                        # check if matrix
                        if isinstance(raw_value[0], str):
                            raw_value = [entry if len(entry) > 1 else entry[0] for entry in map(lambda x: x.split(), raw_value)]
                    else:
                        raw_value = val.split() if len(val.split()) > 1 else val
                    self._set_request((key, raw_value))
                case ["backup"]:
                    self.log.warning("'backup' request is deprecated, use 'retain <number of steps>' instead!")
                case ["init" | "newstep" | "samestep" | "restart"]:
                    pass
                case ["unit" | "states", _]:
                    pass
                case _:
                    self.log.warning(f"request '{line}' not specified! Will not be applied!")

        self._step_logic()
        self._request_logic()

    def _step_logic(self) -> None:
        """
        Performs step logic
        """
        self.log.debug("Starting step logic")

        # TODO: implement previous_step from driver
        last_step = None
        stepfile = os.path.join(self.QMin.save["savedir"], "STEP")
        self.log.debug(f"stepfile {stepfile}")
        if os.path.isfile(stepfile):
            self.log.debug(f"Found stepfile {stepfile}")
            last_step = int(readfile(stepfile)[0])

        if not self.QMin.save["step"]:
            if last_step is not None:
                self.QMin.save["newstep"] = True
                self.QMin.save["step"] = last_step + 1
            else:
                self.QMin.save["init"] = True
                self.QMin.save["step"] = 0
            return

        if last_step is None:
            assert (
                self.QMin.save["step"] == 0
            ), f'Specified step ({self.QMin.save["step"]}) could not be restarted from!\nCheck your savedir and "STEP" file in {self.QMin.save["savedir"]}'
            self.QMin.save["init"] = True
        elif self.QMin.save["step"] == -1:
            self.QMin.save["newstep"] = True
            self.QMin.save["step"] = last_step + 1
        elif self.QMin.save["step"] == last_step:
            self.QMin.save["samestep"] = True
        elif self.QMin.save["step"] == last_step + 1:
            self.QMin.save["newstep"] = True
        else:
            self.log.error(
                f"""Determined last step ({last_step}) from savedir and specified step ({self.QMin.save["step"]}) do not fit!
                Prepare your savedir and "STEP" file accordingly before starting again or choose "step -1" if you want to proceed from last successful step!"""
            )
            raise RuntimeError()

    def _set_driver_requests(self, requests: dict) -> None:
        # delete all old requests
        self.QMin.requests = QMin().requests
        self.log.debug(f"getting requests {requests} step: {self.QMin.save['step']}")
        # logic for raw tasks object from pysharc interface
        if "tasks" in requests and isinstance(requests["tasks"], str):
            requests.update({k.lower(): True for k in requests["tasks"].split()})
            if "soc" in requests:
                requests["h"] = True
            del requests["tasks"]
        for task in ["nacdr", "overlap", "grad", "ion"]:
            if task in requests and isinstance(requests[task], str):
                if requests[task] == "":  # removes task from dict if {'task': ''}
                    del requests[task]
                elif task == requests[task].lower() or requests[task] == "all":
                    requests[task] = [i + 1 for i in range(self.QMin.molecule["nstates"])]
                else:
                    requests[task] = [int(i) for i in requests[task].split()]

        if self.QMin.save["step"] == 0:
            for req in ["overlap", "phases"]:
                if req in requests:
                    requests[req] = False
        self.log.debug(f"setting requests {requests}")
        self.QMin.requests.update(requests)
        for i in ["init", "newstep", "samestep"]:
            self.QMin.save[i] = False
        # if restart:
        # self._step_logic()
        self._request_logic()

    def _set_request(self, request: list[str]) -> None:
        """
        Setup requests and do basic sanity checks
        """
        req = request[0]
        if req in self.QMin.requests.keys():
            match request:
                case ["grad", None]:
                    self.QMin.requests[req] = [i + 1 for i in range(self.QMin.molecule["nmstates"])]
                case ["grad", "all"]:
                    self.QMin.requests[req] = [i + 1 for i in range(self.QMin.molecule["nmstates"])]
                case ["nacdr" | "multipolar_fit" | "density_matrices", None]:
                    self.QMin.requests[req] = ["all"]
                case ["nacdr" | "multipolar_fit" | "density_matrices", "all"]:
                    self.QMin.requests[req] = ["all"]
                case ["grad", value]:
                    self.QMin.requests[req] = sorted(list(map(int, request[1])))
                    if max(self.QMin.requests[req]) > self.QMin.molecule["nmstates"]:
                        self.log.error(f"Requested {req} higher than total number of states!")
                        raise ValueError()
                    if min(self.QMin.requests[req]) <= 0:
                        self.log.error(f"Requested {req} must be greather than 0!")
                        raise ValueError()
                    if len(self.QMin.requests[req]) != len(set(self.QMin.requests[req])):
                        self.log.error(f"Duplicate {req} requested!")
                        raise ValueError()
                case ["nacdr", value]:
                    self.QMin.requests[req] = sorted([[int(x) for x in y] for y in value])
                    if not all(len(x) == 2 for x in self.QMin.requests[req]):
                        raise ValueError(f"'{req}' not set correctly! Needs to to be nx2 matrix not {self.QMin.requests[req]}")
                case ["density_matrices" | "multipolar_fit", value]:
                    self.QMin.requests[req] = value
                case ["soc", None]:
                    if len(self.QMin.molecule["states"]) < 2:
                        self.log.warning("SOCs requested but only singlets given! Disable SOCs")
                        return
                    self.QMin.requests["soc"] = True
                    self.QMin.requests["h"] = True
                case ["retain", _]:
                    self.QMin.requests[req] = int(request[1])
                case _:
                    self.QMin.requests[req] = True
        else:
            match request[0]:
                case "step":
                    self.QMin.save[req] = int(request[1])
                case _:
                    self.QMin.save[req] = True

    def _request_logic(self) -> None:
        """
        Checks for conflicting options, generates requested maps
        and sets path variables according to requests
        """
        self.log.debug("Starting request logic")

        if not os.path.isdir(self.QMin.save["savedir"]):
            self.log.debug(f"Creating savedir {self.QMin.save['savedir']}")
            os.mkdir(self.QMin.save["savedir"])

        assert not (
            (self.QMin.requests["overlap"] or self.QMin.requests["phases"]) and self.QMin.save["init"]
        ), '"overlap" and "phases" cannot be calculated in the first timestep!'

    def write_step_file(self) -> None:
        """
        Write current step into stepfile (only if cleanup not requested)
        """
        if self.QMin.requests["cleanup"]:
            return
        stepfile = os.path.join(self.QMin.save["savedir"], "STEP")
        writefile(stepfile, str(self.QMin.save["step"]))

    def writeQMout(self, filename: str = "QM.out") -> None:
        """
        Writes the requested quantities to the file which SHARC reads in.
        """
        outfilename = os.path.splitext(filename)[0] + ".out"
        self.log.info(f"===> Writing output to file {outfilename} in SHARC Format\n")
        self.QMout.write(outfilename, self.QMin.requests)

    def formatQMout(self) -> str:
        """If PRINT, prints a summary of all requested QM output values.
        Matrices are formatted using printcomplexmatrix, vectors using printgrad.
        """
        return self.QMout.formatQMout(self.QMin, DEBUG=self._DEBUG)

    def printQMout(self) -> None:
        """
        Prints formatted QMout data
        """
        self.log.info(self.formatQMout())

    def printheader(self) -> None:
        """Prints the formatted header of the log file. Prints version number and version date
        Takes nothing, returns nothing."""

        self.log.info(f"{self.clock.starttime} {gethostname()} {os.getcwd()}")
        rule = "=" * 76
        lines = [
            f"  {rule}",
            "",
            f"SHARC - {self.name()} - Interface",
            "",
            f"Authors: {self.authors()}",
            "",
            f"Version: {self.version()}",
            f"Date: {self.versiondate():%d.%m.%Y}",
            "",
            f"  {rule}",
        ]
        # wraps Authors line in case its too long
        lines[4:5] = wrap(lines[4], width=70)
        lines[1:-1] = map(lambda s: "||{:^76}||".format(s), lines[1:-1])
        self.log.info("\n".join(lines))



