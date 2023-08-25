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
import os
import re
import sys
from abc import ABC, abstractmethod
from datetime import date
from io import TextIOWrapper

# from functools import reduce, singledispatchmethod
from socket import gethostname
from textwrap import wrap
from typing import List, Union

import numpy as np

# internal
from constants import ATOMCHARGE, FROZENS, BOHR_TO_ANG
from logger import logging, CustomFormatter, SHARCPRINT
from qmin import QMin
from qmout import QMout
from utils import readfile, writefile, clock, parse_xyz, itnmstates, expand_path

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

    persistent:     Does something
    logname:        Name of the logger
    logfile:        Filename for logger output
    loglevel:       Set loglevel
    """

    # internal status indicators
    _setup_mol = False
    _read_resources = False
    _read_template = False
    _DEBUG = False
    _PRINT = True

    # TODO: set Debug and Print flag
    # TODO: set persistant flag for file-io vs in-core

    def __init__(
        self,
        persistent=False,
        logname: str = None,
        logfile: str = None,
        loglevel: int = logging.INFO,
    ):
        # all the output from the calculation will be stored here
        self.QMout = QMout()
        self.clock = clock()
        self.persistent = persistent
        self.QMin = QMin()
        self._setup_mol = False
        self._read_resources = False
        self._setsave = False

        logname = self.name() if logname is None else logname
        self.log = logging.getLogger(logname)
        self.log.propagate = False
        self.log.handlers = []
        self.log.setLevel(loglevel)
        hdlr = (
            logging.StreamHandler(sys.stdout)
            if logfile is None
            else logging.FileHandler(filename=logfile, mode="w", encoding="utf-8")
        )
        hdlr._name = logname + 'Handler'
        hdlr.setFormatter(CustomFormatter())

        self.log.addHandler(hdlr)
        self.log.print = self.sharcprint

    def sharcprint(self, msg, *args, **kwargs):
        """
        Log 'msg % args' with severity 'SHARCPRINT'.

        To pass exception information, use the keyword argument exc_info with
        a true value, e.g.
        """
        self.log.log(SHARCPRINT, msg, *args, **kwargs)

    @staticmethod
    @abstractmethod
    def authors() -> str:
        return "Severin Polonius, Sebastian Mai"

    @staticmethod
    @abstractmethod
    def version() -> str:
        return "3.0"

    @staticmethod
    @abstractmethod
    def versiondate() -> date:
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

    def print_qmin(self) -> None:
        self.log.info(f"{self.QMin}")

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
                f"./SHARC_{self.name()} <QMin>",
                f"version: {self.version()}",
                f"date: {self.versiondate()}",
                f"changelog: {self.changelogstring()}",
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
        self.setup_interface()
        # print qmin
        self.print_qmin()
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
        # if self._PRINT or self._DEBUG:
        #     string = self.formatQMout()
        self.log.info(self.formatQMout())
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
        self.log.debug(f"Reading template file {template_file}")

        if self._read_template:
            self.log.warning(f"Template already read! Overwriting with {template_file}")

        with open(template_file, "r", encoding="utf-8") as tmpl_file:
            for line in tmpl_file:
                # Ignore comments and empty lines
                if re.match(r"^(\s*)\w+", line):
                    # Remove comments and assign values
                    param = re.sub(r"#.*$", "", line).split()
                    if len(param) == 1:
                        self.QMin.template[param[0]] = True
                    elif len(param) == 2:
                        if param[0] in self.QMin.template.types.keys():
                            self.QMin.template.types[param[0]](param[1])
                        else:
                            self.QMin.template[param[0]] = param[1]
                    else:
                        self.QMin.template[param[0]] = list(param[1:])

        self._read_template = True

    @abstractmethod
    def run(self):
        pass

    @abstractmethod
    def setup_interface(self):
        pass

    @abstractmethod
    def getQMout(self):
        pass

    @abstractmethod
    def create_restart_files(self):
        pass

    def set_coords(self, xyz: Union[str, List, np.ndarray], pc: bool = False) -> None:
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
                raise ValueError(
                    "first line must contain the number of atoms!"
                ) from error
            self.QMin.coords[key] = (
                << << << < HEAD
                np.asarray([parse_xyz(x)[1] for x in lines[2: natom + 2]], dtype=float) ==
                == ===
                np.asarray([parse_xyz(x)[1] for x in lines[2: natom + 2]], dtype=float) >>
                >>>> > 4199fad0362739dbc01aadc51ce65c2436a5908e *
                self.QMin.molecule["factor"]
            )
        elif isinstance(xyz, (list, np.ndarray)):
            self.QMin.coords[key] = np.asarray(xyz) * self.QMin.molecule["factor"]
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
        self.log.debug(f"Setting up molecule from {qmin_file}")

        if self._setup_mol:
            self.log.warning(
                f"setup_mol() was already called! Continue setup with {qmin_file}",
            )

        qmin_lines = readfile(qmin_file)
        self.QMin.molecule["comment"] = qmin_lines[1]

        try:
            natom = int(qmin_lines[0])
        except ValueError as e:
            raise ValueError("first line must contain the number of atoms!") from e
        if len(qmin_lines) < natom + 4:
            raise RuntimeError(
                'Input file must contain at least:\nnatom\ncomment\ngeometry\nkeyword "states"\nat least one task'
            )
        self.QMin.molecule["elements"] = list(
            map(lambda x: parse_xyz(x)[0], (qmin_lines[2: natom + 2]))
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
                qmin_lines[self.QMin.molecule["natom"] + 2:],
            ),
        )

        # naively parse all key argument pairs from QM.in
        for line in filtered:
            llist = line.split(None, 1)
            key = llist[0].lower()
            if key == "states":
                # also does update nmstates, nstates, statemap
                states_dict = self.parseStates(llist[1])
                self.QMin.maps["statemap"] = states_dict["statemap"]
                self.QMin.molecule["nstates"] = states_dict["nstates"]
                self.QMin.molecule["nmstates"] = states_dict["nmstates"]
                self.QMin.molecule["states"] = states_dict["states"]
            elif key == "unit":
                unit = llist[1].strip().lower()
                if unit in ["bohr", "angstrom"]:
                    self.QMin.molecule["unit"] = unit
                    self.QMin.molecule["factor"] = (
                        1.0 if unit == "bohr" else 1.0 / BOHR_TO_ANG
                    )
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

        self._setup_mol = True

        self.log.debug("Setup successful.")

    def parseStates(self, states: str) -> dict:
        """
        Setup states, statemap and everything related
        """
        res = {}
        try:
            res["states"] = list(map(int, states.split()))
        except (ValueError, IndexError) as e:
            raise ValueError(
                'Keyword "states" has to be followed by integers!', 37
            ) from e
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
        return {
            "nstates": nstates,
            "nmstates": nmstates,
            "states": res["states"],
            "statemap": {i + 1: [*v] for i, v in enumerate(itnmstates(res["states"]))},
        }

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
        self.log.debug(f"Reading resource file {resources_file}")

        if not self._setup_mol:
            raise RuntimeError(
                "Interface is not set up for this template. Call setup_mol with the QM.in file first!"
            )

        if self._read_resources:
            self.log.warning(
                f"Resources already read! Overwriting with {resources_file}"
            )

        # Set ncpu from env variables, gets overwritten if in resources
        priority_order = ["SLURM_NTASKS_PER_NODE", " NSLOTS"]
        for pr in priority_order:
            if pr in os.environ:
                self.QMin.resources["ncpu"] = max(1, int(os.environ[pr]))
                self.log.info(
                    f'Found env variable ncpu={os.environ[pr]}, resources["ncpu"] set to {self.QMin.resources["ncpu"]}',
                )
                break

        with open(resources_file, "r", encoding="utf-8") as rcs_file:
            # Store all encountered keywords to warn for duplicates
            keyword_list = []
            for line in rcs_file:
                # Ignore comments and empty lines
                if re.match(r"^(\s*)\w+", line):
                    # Remove comments and assign values
                    param = re.sub(r"#.*$", "", line).split()
                    # Expand to fullpath if ~ or $ in string
                    param = [
                        expand_path(x) if re.match(r"\~|\$", x) else x for x in param
                    ]

                    # Check for duplicates in keyword_list
                    if param[0] in keyword_list:
                        self.log.warning(
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
                                self.log.debug(
                                    f"SAVEDIR set to {self.QMin.save['savedir']}",
                                )
                            else:
                                self.log.info(
                                    "SAVEDIR is already set and will not be overwritten!"
                                )
                            continue
                        # Cast to correct type if available
                        if param[0] in self.QMin.resources.types:
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
                            self.log.debug(f"Extend white listed parameter {param[0]}")
                            self.QMin.resources[param[0]].extend(list(param[1:]))
                        else:
                            self.log.warning(f"Parameter list {param} overwritten!")
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

        self.log.debug(f"Reading requests from {requests_file}")

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
            nacdr = []
            for line in requests:
                # Check for valid keywords, remove comments
                line = re.sub(r"#.*$", "", line)
                if re.match(r"^(\s*)\w", line):
                    params = line.split()

                    # Parse NACDR if requested
                    if params[0].casefold() == "nacdr":
                        self.log.debug(
                            f"Parsing request {params}",
                        )
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
                            self.log.debug(f"Adding state pair {params} to NACDR list")
                            nacdr.append(params)
                        continue

                    # Parse every other request
                    if params[0].casefold() in (
                        *self.QMin.requests.keys(),
                        "step",
                    ):
                        self.log.debug(f"Parsing request {params}")
                        self._set_requests(params)

            assert not nac_select, "No end keyword found after nacdr select!"
            if nacdr:
                self.QMin.requests["nacdr"] = nacdr
        self._step_logic()
        self._request_logic()

        if self.QMin.requests["backup"]:
            self.log.debug("Setting up backup directories")

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
            self.log.error(
                f'Determined last step ({last_step}) from savedir and specified step ({self.QMin.save["step"]}) do not fit!\nPrepare your savedir and "STEP" file accordingly before starting again or choose "step -1" if you want to proceed from last successful step!'
            )
            raise RuntimeError()

    def _set_driver_requests(self, requests: dict):
        # delete all old requests
        self.QMin.requests = QMin().requests
        self.log.debug(f"getting requests {requests} step: {self.QMin.save['step']}")
        # logic for raw tasks object from pysharc interface
        if 'tasks' in requests and type(requests['tasks']) is str:
            requests.update({k.lower(): True for k in requests['tasks'].split()})
            del requests['tasks']
        for task in ['nacdr', 'overlap', 'grad', 'ion']:
            if task in requests and type(requests[task]) is str:
                if requests[task] == '':    # removes task from dict if {'task': ''}
                    del requests[task]
                elif task == requests[task].lower() or requests[task] == 'all':
                    requests[task] = [i + 1 for i in range(self.QMin.molecule['nstates'])]
                else:
                    requests[task] = [int(i) for i in requests[task].split()]

        if self.QMin.save['step'] == 0:
            for r in ['overlap', 'phases']:
                if r in requests:
                    requests[r] = False
        self.log.debug(f"setting requests {requests}")
        self.QMin.requests.update(requests)
        for i in ['init', 'newstep', 'samestep']:
            self.QMin.save[i] = False
        self._request_logic()

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
                    self.log.warning(
                        "SOCs requested but only 1 multiplicity given! Disable SOCs"
                    )
                    return
                self.QMin.requests["soc"] = True
            elif request[0].casefold() == "multipolar_fit":
                if len(request) > 1:
                    self.QMin.requests["multipolar_fit"] = sorted(request[1:])
                    return
                self.QMin.requests["multipolar_fit"] = [
                    i + 1 for i in range(self.QMin.molecule["nmstates"])
                ]
            else:
                self.QMin.requests[request[0].casefold()] = True
        elif request[0].casefold() == "step":
            self.QMin.save[request[0].casefold()] = int(request[1])
        else:
            self.QMin.save[request[0].casefold()] = True

    def _request_logic(self) -> None:
        """
        Checks for conflicting options, generates requested maps
        and sets path variables according to requests
        """
        self.log.debug("Starting request logic")

        if not os.path.exists(self.QMin.save["savedir"]):
            self.log.debug(f"Creating savedir {self.QMin.save['savedir']}")
            os.mkdir(self.QMin.save["savedir"])

        assert not (
            (self.QMin.requests["overlap"] or self.QMin.requests["phases"])
            and self.QMin.save["init"]
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
        k = filename.rfind(".")
        if k == -1:
            outfilename = filename + ".out"
        else:
            outfilename = filename[:k] + ".out"
        self.log.info(
            "===> Writing output to file %s in SHARC Format\n" % (outfilename)
        )
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

    # ============================PRINTING ROUTINES========================== #

    def printheader(self):
        """Prints the formatted header of the log file. Prints version number and version date
        Takes nothing, returns nothing."""

        print(self.clock.starttime, gethostname(), os.getcwd())
        rule = "=" * 76
        lines = [
            f"  {rule}",
            "",
            f"SHARC - {self.name()} - Interface",
            "",
            f"Authors: {self.authors()}",
            "",
            f"Version: {self.version()}",
            "Date: {:%d.%m.%Y}".format(self.versiondate()),
            "",
            f"  {rule}",
        ]
        # wraps Authors line in case its too long
        lines[4:5] = wrap(lines[4], width=70)
        lines[1:-1] = map(lambda s: "||{:^76}||".format(s), lines[1:-1])
        print(*lines, sep="\n")
        print("\n")


if __name__ == "__main__":
    logging.info("hello from log!")
    logging.debug("this is a debug")
    logging.error("EROROR")
