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
from copy import deepcopy
from datetime import date, datetime
import math
import sys
import os
import glob
import re
import shutil
import ast
import numpy as np
import subprocess as sp
from abc import ABC, abstractmethod
from typing import Union, List

# from functools import reduce, singledispatchmethod
from socket import gethostname
from textwrap import wrap
import logging
import logging.config

# internal
from printing import printcomplexmatrix, printgrad, printtheodore
from utils import *
from constants import *
from qmin import QMin


def expand_path(path: str) -> str:
    """
    Expand variables in path, error out if variable is not resolvable
    """
    expand = os.path.abspath(os.path.expanduser(os.path.expandvars(path)))
    assert "$" not in expand, f"Undefined env variable in {expand}"
    return expand


class CustomFormatter(logging.Formatter):
    err_fmt = "ERROR: %(msg)s"
    dbg_fmt = "DEBUG: %(msg)s"
    info_fmt = "%(msg)s"

    def format(self, record):
        # Replace the original format with one customized by logging level
        if record.levelno == logging.DEBUG:
            self._fmt = CustomFormatter.dbg_fmt

        elif record.levelno == logging.INFO:
            self._fmt = CustomFormatter.info_fmt

        elif record.levelno == logging.ERROR:
            self._fmt = CustomFormatter.err_fmt

        # Call the original formatter class to do the grunt work
        formatter = logging.Formatter(self._fmt)

        return formatter.format(record)


fmt = CustomFormatter()
hdlr = logging.StreamHandler(sys.stdout)

hdlr.setFormatter(fmt)
logging.root.addHandler(hdlr)
logging.root.setLevel(logging.DEBUG)


class SHARC_INTERFACE(ABC):
    # internal status indicators
    _setup_mol = False
    _read_resources = False
    _read_template = False
    _DEBUG = False
    _PRINT = True

    # TODO: set Debug and Print flag
    # TODO: set persistant flag for file-io vs in-core
    def __init__(self, persistent=False):
        # all the output from the calculation will be stored here
        self.QMout = {}
        self.persistent = persistent
        self.QMin = QMin()
        self._setup_mol = False
        self._read_resources = False
        self._setsave = False

    @abstractmethod
    def authors(self) -> str:
        return "Severin Polonius, Sebastian Mai"

    @abstractmethod
    def version(self) -> str:
        return "3.0"

    @abstractmethod
    def versiondate(self) -> date:
        return date(2021, 7, 15)

    @abstractmethod
    def name(self) -> str:
        return "base"

    @abstractmethod
    def description(self) -> str:
        return "Abstract base class for SHARC interfaces."

    @abstractmethod
    def changelogstring(self) -> str:
        return "This is the changelog string"

    def main(self):
        """
        main routine for all interfaces.
        This routine containes all functions that will be accessed when any interface is calculating a single point.
        All of these functions have to be defined in the derived class if not available in this base class
        """

        args = sys.argv
        self.clock = clock()
        self.printheader()
        if len(args) != 2:
            print(
                "Usage:",
                f"./SHARC_{self.name} <QMin>",
                f"version: {self.version}",
                f"date: {self.versiondate}",
                f"changelog: {self.changelogstring}",
                sep="\n",
            )
            sys.exit(106)
        QMinfilename = sys.argv[1]
        # set up the system (i.e. molecule, states, unit...)
        self.setup_mol(QMinfilename)
        # read in the resources available for this computation (program path, cores, memory)
        self.read_resources(f"{self.name}.resources")
        # read in the specific template file for the interface with all keywords
        self.read_template(f"{self.name}.template")
        # set the coordinates of the molecular system
        self.set_coords(QMinfilename)
        # read the property requests that have to be calculated
        self.read_requests(QMinfilename)
        # setup internal state for the computation
        self.setup_run()

        # perform the calculation and parse the output, do subsequent calculations with other tools
        self.run()

        # get output as requested
        self.getQMout()

        # backup data if requested
        if self.QMin.requests["backup"]:
            self.backupdata(self.QMin.requests["backup"])
        # writes a STEP file in the SAVEDIR (marks this step as succesfull)
        self.write_step_file()

        # printing and output generation
        if self._PRINT or self._DEBUG:
            self.printQMout()
        self.QMout["runtime"] = self.clock.measuretime()
        self.writeQMout()

    @abstractmethod
    def read_template(self, template_file: str) -> None:
        """
        Reads a template file and assigns parameters to
        self.template. No sanity checks at all, has to be done
        in the interface. If multiple entries
        of a parameter with one value are in the file, the latest value will be saved.

        template_file:  Path to template file
        """
        logging.debug(f"Reading template file {template_file}")

        if self._read_template:
            logging.warning(f"Template already read! Overwriting with {template_file}")

        with open(template_file, "r", encoding="utf-8") as tmpl_file:
            for line in tmpl_file:
                # Ignore comments and empty lines
                if re.match(r"^\w+", line):
                    # Remove comments and assign values
                    param = re.sub(r"#.*$", "", line).split()
                    if len(param) == 1:
                        self.QMin.template[param[0]] = True
                    elif len(param) == 2:
                        self.QMin.template[param[0]] = param[1]
                    else:
                        self.QMin.template[param[0]] = list(param[1:])

        self._read_template = True

    @abstractmethod
    def run(self):
        pass

    @abstractmethod
    def getQMout(self):
        pass

    def set_coords(self, xyz: Union[str, List, np.ndarray]) -> None:
        """
        Sets coordinates, qmmm and pccharge from file or list/array
        xyz: path to xyz file or list/array with coords
        """
        if isinstance(xyz, str):
            lines = readfile(xyz)
            try:
                natom = int(lines[0])
            except ValueError as error:
                raise ValueError(
                    "first line must contain the number of atoms!"
                ) from error
            self.QMin.coords["coords"] = (
                np.asarray([parse_xyz(x)[1] for x in lines[2 : natom + 2]], dtype=float)
                * self.QMin.molecule["factor"]
            )
        elif isinstance(xyz, (list, np.ndarray)):
            self.QMin.coords["coords"] = np.asarray(xyz) * self.QMin.molecule["factor"]
        else:
            raise NotImplementedError(
                "'set_coords' is only implemented for str, list[list[float]] or numpy.ndarray type"
            )

    def setup_mol(self, qmin_file: str) -> None:
        """
        Sets up the molecular system from a `QM.in` file.
        parses the elements, states, and savedir and prepare the QMin object accordingly.

        qmin_file:  Path to QM.in file.
        """
        logging.debug(f"Setting up molecule from {qmin_file}")

        if self._setup_mol:
            logging.warning(
                f"setup_mol() was already called! Continue setup with {qmin_file}",
            )

        qmin_lines = readfile(qmin_file)
        self.QMin.molecule["comment"] = qmin_lines[1]

        try:
            natom = int(qmin_lines[0])
        except ValueError:
            raise Error("first line must contain the number of atoms!", 2)
        if len(qmin_lines) < natom + 4:
            raise Error(
                'Input file must contain at least:\nnatom\ncomment\ngeometry\nkeyword "states"\nat least one task',
                3,
            )
        self.QMin.molecule["elements"] = list(
            map(lambda x: parse_xyz(x)[0], (qmin_lines[2 : natom + 2]))
        )
        self.QMin.molecule["Atomcharge"] = sum(
            map(lambda x: ATOMCHARGE[x], self.QMin.molecule["elements"])
        )
        self.QMin.molecule["frozcore"] = sum(
            map(lambda x: FROZENS[x], self.QMin.molecule["elements"])
        )
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
                self.parseStates(llist[1])
            elif key == "unit":
                unit = llist[1].strip().lower()
                if unit in ["bohr", "angstrom"]:
                    self.QMin.molecule["unit"] = unit
                    self.QMin.molecule["factor"] = (
                        1.0 if unit == "bohr" else 1.0 / BOHR_TO_ANG
                    )
                else:
                    raise Error("unknown unit specified", 23)
            elif key == "savedir":
                self._setsave = True
                self.QMin.save["savedir"] = llist[1].strip()
                logging.debug(f"SAVEDIR set to {self.QMin.save['savedir']}")

        if not isinstance(self.QMin.save["savedir"], str):
            self.QMin.save["savedir"] = "./SAVEDIR/"
            logging.debug("Setting default SAVEDIR")

        self.QMin.save["savedir"] = expand_path(self.QMin.save["savedir"])

        if not isinstance(self.QMin.molecule["unit"], str):
            logging.warning('No "unit" specified in QMin! Assuming Bohr')
            self.QMin.molecule["unit"] = "bohr"

        self._setup_mol = True

        logging.debug("Setup successful.")

    def parseStates(self, states: str) -> None:
        """
        Setup states, statemap and everything related
        """
        res = {}
        try:
            res["states"] = list(map(int, states.split()))
        except (ValueError, IndexError):
            raise ValueError('Keyword "states" has to be followed by integers!', 37)
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
        self.QMin.maps["statemap"] = {
            i + 1: [*v] for i, v in enumerate(itnmstates(res["states"]))
        }
        self.QMin.molecule["nstates"] = nstates
        self.QMin.molecule["nmstates"] = nmstates
        self.QMin.molecule["states"] = res["states"]

    @abstractmethod
    def read_resources(self, resources_file: str, kw_whitelist: list = []) -> None:
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
        logging.debug(f"Reading resource file {resources_file}")

        if not self._setup_mol:
            raise Error(
                "Interface is not set up for this template. Call setup_mol with the QM.in file first!",
                23,
            )

        if self._read_resources:
            logging.warning(
                f"Resources already read! Overwriting with {resources_file}"
            )

        # Set ncpu from env variables, gets overwritten if in resources
        priority_order = ["SLURM_NTASKS_PER_NODE", " NSLOTS"]
        for pr in priority_order:
            if pr in os.environ:
                self.QMin.resources["ncpu"] = max(1, int(os.environ[pr]))
                logging.info(
                    f'Found env variable ncpu={os.environ[pr]}, resources["ncpu"] set to {self.QMin.resources["ncpu"]}',
                )
                break

        with open(resources_file, "r", encoding="utf-8") as rcs_file:
            # Store all encountered keywords to warn for duplicates
            keyword_list = []
            for line in rcs_file:
                # Ignore comments and empty lines
                if re.match(r"^\w+", line):
                    # Remove comments and assign values
                    param = re.sub(r"#.*$", "", line).split()
                    # Expand to fullpath if ~ or $ in string
                    param = [
                        expand_path(x) if re.match(r"\~|\$", x) else x for x in param
                    ]

                    # Check for duplicates in keyword_list
                    if param[0] in keyword_list:
                        logging.warning(
                            f"Multiple entries of {param[0]} in {resources_file}"
                        )
                    keyword_list.append(param[0])

                    if len(param) == 1:
                        self.QMin.resources[param[0]] = True
                    elif len(param) == 2:
                        # Check if savedir already specified in QM.in
                        if param[0] == "savedir":
                            if not self._setsave:
                                self.QMin.save["savedir"] = param[1]
                                logging.debug(
                                    f"SAVEDIR set to {self.QMin.save['savedir']}", 
                                )
                            else:
                                logging.info(
                                    "SAVEDIR is already set and will not be overwritten!"
                                )
                            continue
                        # Cast to correct type if available
                        if param[0] in self.QMin.resources.keys():
                            self.QMin.resources[param[0]] = self.QMin.resources.types[
                                param[0]
                            ](param[1])
                        else:
                            self.QMin.resources[param[0]] = param[1]
                    else:
                        # If whitelisted key already exists extend list with values
                        if (
                            param[0] in self.QMin.resources.keys()
                            and self.QMin.resources[param[0]]
                            and param[0] in kw_whitelist
                        ):
                            logging.debug(f"Extend white listed parameter {param[0]}")
                            self.QMin.resources[param[0]].extend(list(param[1:]))
                        else:
                            logging.warning(f"Parameter list {param} overwritten!")
                            self.QMin.resources[param[0]] = list(param[1:])
        self._read_resources = True

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

        logging.debug(f"Reading requests from {requests_file}")

        # Reset requests
        self.QMin.requests = QMin().requests
        self.QMin.save["init"] = False
        self.QMin.save["samestep"] = False
        self.QMin.save["newstep"] = False
        self.QMin.save["restart"] = False

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
                        logging.debug(f"Parsing request {params}", )
                        if len(params) > 1 and params[1].casefold() == "select":
                            nac_select = True
                        else:
                            self.QMin.requests["nacdr"] = ["all"]
                        continue
                    if nac_select:
                        if params[0].casefold() == "end":
                            nac_select = False
                        else:
                            assert (
                                len(params) == 2
                            ), "NACs have to be given in state pairs!"
                            logging.debug(f"Adding state pair {params} to NACDR list")
                            self.QMin.requests["nacdr"].append(params)
                        continue

                    # Parse every other request
                    if params[0].casefold() in (
                        *self.QMin.requests.keys(),
                        "init",
                        "samestep",
                        "restart",
                        "newstep",
                    ):
                        logging.debug(f"Parsing request {params}")
                        self._set_requests(params)

            assert not nac_select, "No end keyword found after nacdr select!"
        self._step_logic()
        self._request_logic()

        if self.QMin.requests["backup"]:
            logging.debug("Setting up backup directories")

    def _step_logic(self) -> None:
        """
        Performs step logic
        """
        logging.debug("Starting step logic")

        # TODO: implement previous_step from driver
        last_step = None
        stepfile = os.path.join(self.QMin.save["savedir"], "STEP")
        if os.path.isfile(stepfile):
            logging.debug(f"Found stepfile {stepfile}")
            last_step = int(readfile(stepfile)[0])

        if not self.QMin.save["step"]:
            if last_step:
                self.QMin.save["newstep"] = True
                self.QMin.save["step"] = last_step + 1
            else:
                self.QMin.save["init"] = True
                self.QMin.save["step"] = 0
            return

        if not last_step:
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
            raise Error(
                f'Determined last step ({last_step}) from savedir and specified step ({self.QMin.save["step"]}) do not fit!\nPrepare your savedir and "STEP" file accordingly before starting again or choose "step -1" if you want to proceed from last successful step!'
            )

    def _set_requests(self, request: list) -> None:
        """
        Setup requests and do basic sanity checks
        """
        if request[0].casefold() in self.QMin.requests.keys():
            if request[0].casefold() == "h" and len(request) == 1:
                self.QMin.requests["h"] = True
            elif request[0].casefold() == "grad":
                if len(request) > 1 and request[1].casefold() != "all":
                    self.QMin.requests["grad"] = [int(i) for i in request[1:]]
                    return
                self.QMin.requests["grad"] = [
                    i + 1 for i in range(self.QMin.molecule["nmstates"])
                ]
            elif request[0].casefold() == "soc":
                if sum(i > 0 for i in self.QMin.molecule["states"]) < 2:
                    logging.warning(
                        "SOCs requestet but only 1 multiplicity given! Disable SOCs"
                    )
                    return
                self.QMin.requests["soc"] = True
            elif request[0].casefold() == "multipolar_fit":
                if len(request > 1):
                    self.QMin.requests["multipolar_fit"] = sorted(request[1:])
                    return
                self.QMin.requests["multipolar_fit"] = [
                    i + 1 for i in range(self.QMin.molecule["nmstates"])
                ]
            else:
                self.QMin.requests[request[0].casefold()] = True
        else:
            self.QMin.save[request[0].casefold()] = True

    def _request_logic(self) -> None:
        """
        Checks for conflicting options, generates requested maps
        and sets path variables according to requests
        """
        logging.debug("Starting request logic")

        if not os.path.exists(self.QMin.save["savedir"]):
            logging.debug(f"Creating savedir {self.QMin.save['savedir']}")
            os.mkdir(self.QMin.save["savedir"])

        if self.QMin.requests["phases"] and not self.QMin.requests["overlap"]:
            logging.info("Found phases in requests, set overlap to true")
            self.QMin.requests["overlap"] = True

        if (
            self.QMin.requests["ion"] or self.QMin.requests["overlap"]
        ) and self.__class__.__name__ != "LVC":
            assert os.path.isfile(
                self.QMin.resources["wfoverlap"]
            ), "Missing path to wfoverlap.x in resources file!"

        assert not (
            self.QMin.requests["overlap"] and self.QMin.save["init"]
        ), '"overlap" and "phases" cannot be calculated in the first timestep! Delete either "overlap" or "init"'

    @abstractmethod
    def write_step_file(self):
        pass

    @abstractmethod
    def writeQMout(self):
        pass

    @abstractmethod
    def printQMout(self):
        pass

    # ============================PRINTING ROUTINES========================== #

    def printheader(self):
        """Prints the formatted header of the log file. Prints version number and version date
        Takes nothing, returns nothing."""

        print(self.clock.starttime, gethostname(), os.getcwd())
        rule = "=" * 76
        lines = [
            f"  {rule}",
            "",
            f"SHARC - {self.name} - Interface",
            "",
            f"Authors: {self.authors}",
            "",
            f"Version: {self.version}",
            "Date: {:%d.%m.%Y}".format(self.versiondate),
            "",
            f"  {rule}",
        ]
        # wraps Authors line in case its too long
        lines[4:5] = wrap(lines[4], width=70)
        lines[1:-1] = map(lambda s: "||{:^76}||".format(s), lines[1:-1])
        print(*lines, sep="\n")
        print("\n")


class SHARC_ABINITIO(SHARC_INTERFACE):
    @abstractmethod
    def create_restart_files(self):
        pass

    def read_resources(self):
        super().read_resources()


if __name__ == "__main__":
    logging.info("hello from log!")
    logging.debug("this is a debug")
    logging.error("EROROR")
