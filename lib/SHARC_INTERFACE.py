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
from logger import log as logging

# internal
from printing import printcomplexmatrix, printgrad, printtheodore
from utils import *
from constants import *
from qmin import QMin



all_features = {'h',
                 'soc',
                 'dm',
                 'grad',
                 'nacdr',
                 'overlap',
                 'phases',
                 'ion',
                 'dmdr',
                 'socdr',
                 'multipolar_fit',
                 'theodore',
                 'point_charges',
                # raw data request
                 'basis_set',
                 'wave_functions',
                 'density_matrices',
                 }


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
    def get_features(self) -> set:
        "return availble features"
        return all_features

    @abstractmethod
    def get_infos(self, INFOS: dict) -> dict:
        "prepare INFOS obj"
        return INFOS

    @abstractmethod
    def prepare(self, INFOS: dict, dir: str):
        "setup the calculation in directory 'dir'"
        return


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
    def setup_run(self):
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
                np.asarray([parse_xyz(x)[1] for x in lines[2: natom + 2]], dtype=float) *
                self.QMin.molecule["factor"]
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
                        if param[0] in self.QMin.resources.types.keys():
                            self.QMin.resources[param[0]] = self.QMin.resources.types[
                                param[0]
                            ](param[1])
                        else:
                            self.QMin.resources[param[0]] = param[1]
                    else:
                        # If whitelisted key already exists extend list with values
                        if (
                            param[0] in self.QMin.resources.keys() and
                            self.QMin.resources[param[0]] and
                            param[0] in kw_whitelist
                        ):
                            logging.debug(f"Extend white listed parameter {param[0]}")
                            self.QMin.resources[param[0]].extend(list(param[1:]))
                        else:
                            logging.warning(f"Parameter list {param} overwritten!")
                            self.QMin.resources[param[0]] = list(param[1:])
        self._read_resources = True

    @abstractmethod
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
                        logging.debug(
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

        assert not (
            (self.QMin.requests["overlap"] or self.QMin.requests["phases"])
            and self.QMin.save["init"]
        ), '"overlap" and "phases" cannot be calculated in the first timestep!'

    @abstractmethod
    def write_step_file(self) -> None:
        """
        Write current step into stepfile (only if cleanup not requested)
        """
        if self.QMin.requests["cleanup"]:
            return
        stepfile = os.path.join(self.QMin.save["savedir"], "STEP")
        writefile(stepfile, str(self.QMin.save["step"]))

    @abstractmethod
    def writeQMout(self) -> None:
        """
        Writes the requested quantities to the file which SHARC reads in.
        """
        logging.info("Writing output to QM.out in SHARC format.")
        string = ""
        if self.QMin.requests["h"] or self.QMin.requests["soc"]:
            string += self.writeQMoutsoc()
        if self.QMin.requests["dm"]:
            string += self.writeQMoutdm()
        if self.QMin.requests["grad"]:
            string += self.writeQMoutgrad()
        if self.QMin.requests["overlap"]:
            string += self.writeQMoutnacsmat()
        if self.QMin.requests["nacdr"]:
            string += self.writeQMoutnacana()
        if self.QMin.requests["socdr"]:
            string += self.writeQMoutsocdr()
        if self.QMin.requests["dmdr"]:
            string += self.writeQMoutdmdr()
        if self.QMin.requests["ion"]:
            string += self.writeQMoutprop()
        if self.QMin.requests["theodore"] and QMin["template"]["qmmm"]:
            string += self.writeQMoutTHEODORE()
        if self.QMin.requests["phases"]:
            string += self.writeQmoutPhases()
        if self.QMin.requests["multipolar_fit"]:
            string += self.writeQMoutmultipolarfit()
        string += self.writeQMouttime()
        outfile = os.path.join(self.QMin.resources["pwd"], "QM.out")
        writefile(outfile, string)

    def writeQMoutsoc(self):
        """
        Generates a string with the Spin-Orbit Hamiltonian in SHARC format.

        The string starts with a ! followed by a flag specifying the type of data.
        In the next line, the dimensions of the matrix are given, followed by nmstates blocks of nmstates elements.
        Blocks are separated by a blank line.
        """
        nmstates = self.QMin.molecule["nmstates"]
        string = ""
        string += "! %i Hamiltonian Matrix (%ix%i, complex)\n" % (1, nmstates, nmstates)
        string += "%i %i\n" % (nmstates, nmstates)
        for i in range(nmstates):
            for j in range(nmstates):
                string += "%s %s " % (
                    eformat(self.QMout["h"][i][j].real, 12, 3),
                    eformat(self.QMout["h"][i][j].imag, 12, 3),
                )
            string += "\n"
        string += "\n"
        return string

    def writeQMoutdm(self):
        """
        Generates a string with the Dipole moment matrices in SHARC format.

        The string starts with a ! followed by a flag specifying the type of data.
        In the next line, the dimensions of the matrix are given, followed by nmstates blocks of nmstates elements.
        Blocks are separated by a blank line. The string contains three such matrices.
        """
        QMout = self.QMout
        nmstates = self.QMin.molecule["nmstates"]
        string = ""
        string += "! %i Dipole Moment Matrices (3x%ix%i, complex)\n" % (
            2,
            nmstates,
            nmstates,
        )
        for xyz in range(3):
            string += "%i %i\n" % (nmstates, nmstates)
            for i in range(nmstates):
                for j in range(nmstates):
                    string += "%s %s " % (
                        eformat(QMout["dm"][xyz][i][j].real, 12, 3),
                        eformat(QMout["dm"][xyz][i][j].imag, 12, 3),
                    )
                string += "\n"
            string += ""
        return string

    def writeQMoutgrad(self):
        """
        Generates a string with the Gradient vectors in SHARC format.

        The string starts with a ! followed by a flag specifying the type of data.
        On the next line, natom and 3 are written, followed by the gradient, with one line per atom and
        a blank line at the end. Each MS component shows up (nmstates gradients are written).
        """

        QMout = self.QMout
        states = self.QMin.molecule["states"]
        nmstates = self.QMin.molecule["nmstates"]
        natom = self.QMin.molecule["natom"]
        string = ""
        string += "! %i Gradient Vectors (%ix%ix3, real)\n" % (3, nmstates, natom)
        i = 0
        for imult, istate, ims in itnmstates(states):
            string += "%i %i ! m1 %i s1 %i ms1 %i\n" % (natom, 3, imult, istate, ims)
            for atom in range(natom):
                for xyz in range(3):
                    string += "%s " % (eformat(QMout["grad"][i][atom][xyz], 12, 3))
                string += "\n"
            string += ""
            i += 1
        return string

    def writeQMoutnacsmat(self):
        """
        Generates a string with the adiabatic-diabatic transformation matrix in SHARC format.

        The string starts with a ! followed by a flag specifying the type of data.
        In the next line, the dimensions of the matrix are given, followed by nmstates blocks of nmstates elements.
        Blocks are separated by a blank line.
        """

        QMout = self.QMout
        nmstates = self.QMin.molecule["nmstates"]
        string = ""
        string += "! %i Overlap matrix (%ix%i, complex)\n" % (6, nmstates, nmstates)
        string += "%i %i\n" % (nmstates, nmstates)
        for j in range(nmstates):
            for i in range(nmstates):
                string += "%s %s " % (
                    eformat(QMout["overlap"][j][i].real, 12, 3),
                    eformat(QMout["overlap"][j][i].imag, 12, 3),
                )
            string += "\n"
        string += "\n"
        return string

    def writeQMoutnacana(self):
        """
        Generates a string with the NAC vectors in SHARC format.

        The string starts with a ! followed by a flag specifying the type of data.
        On the next line, natom and 3 are written, followed by the gradient, with one line per atom and
         a blank line at the end. Each MS component shows up (nmstates x nmstates vectors are written).
        """

        QMout = self.QMout
        states = self.QMin.molecule["states"]
        nmstates = self.QMin.molecule["nmstates"]
        natom = self.QMin.molecule["natom"]
        string = ""
        string += "! %i Non-adiabatic couplings (ddr) (%ix%ix%ix3, real)\n" % (
            5,
            nmstates,
            nmstates,
            natom,
        )
        i = 0
        for imult, istate, ims in itnmstates(states):
            j = 0
            for jmult, jstate, jms in itnmstates(states):
                # string+='%i %i ! %i %i %i %i %i %i\n' % (natom,3,imult,istate,ims,jmult,jstate,jms)
                string += "%i %i ! m1 %i s1 %i ms1 %i   m2 %i s2 %i ms2 %i\n" % (
                    natom,
                    3,
                    imult,
                    istate,
                    ims,
                    jmult,
                    jstate,
                    jms,
                )
                for atom in range(natom):
                    for xyz in range(3):
                        string += "%s " % (
                            eformat(QMout["nacdr"][i][j][atom][xyz], 12, 3)
                        )
                    string += "\n"
                string += ""
                j += 1
            i += 1
        return string

    def writeQMoutsocdr(self):
        """
        Generates a string with the SOCDR vectors in SHARC format.

        The string starts with a ! followed by a flag specifying the type of data.
        On the next line, natom and 3 are written, followed by the gradient, with one line per atom and
         a blank line at the end. Each MS component shows up (nmstates x nmstates vectors are written).
        """
        QMout = self.QMout
        states = self.QMin.molecule["states"]
        nmstates = self.QMin.molecule["nmstates"]
        natom = self.QMin.molecule["natom"]
        string = ""
        string += "! %i Spin-Orbit coupling derivatives (%ix%ix3x%ix3, complex)\n" % (
            13,
            nmstates,
            nmstates,
            natom,
        )
        i = 0
        for imult, istate, ims in itnmstates(states):
            j = 0
            for jmult, jstate, jms in itnmstates(states):
                string += "%i %i ! m1 %i s1 %i ms1 %i   m2 %i s2 %i ms2 %i\n" % (
                    natom,
                    3,
                    imult,
                    istate,
                    ims,
                    jmult,
                    jstate,
                    jms,
                )
                for atom in range(natom):
                    for xyz in range(3):
                        string += "%s %s " % (
                            eformat(QMout["socdr"][i][j][atom][xyz].real, 12, 3),
                            eformat(QMout["socdr"][i][j][atom][xyz].imag, 12, 3),
                        )
                string += "\n"
                string += ""
                j += 1
            i += 1
        string += "\n"
        return string

    def writeQMoutdmdr(self):
        """
        Generates a string with the DMDR vectors in SHARC format.

        The string starts with a ! followed by a flag specifying the type of data.
        On the next line, natom and 3 are written, followed by the gradient, with one line per atom and
         a blank line at the end. Each MS component shows up (nmstates x nmstates vectors are written).
        """
        QMout = self.QMout
        states = self.QMin.molecule["states"]
        nmstates = self.QMin.molecule["nmstates"]
        natom = self.QMin.molecule["natom"]
        string = ""
        string += "! %i Dipole moment derivatives (%ix%ix3x%ix3, real)\n" % (
            12,
            nmstates,
            nmstates,
            natom,
        )
        i = 0
        for imult, istate, ims in itnmstates(states):
            j = 0
            for jmult, jstate, jms in itnmstates(states):
                for ipol in range(3):
                    string += (
                        "%i %i ! m1 %i s1 %i ms1 %i   m2 %i s2 %i ms2 %i   pol %i\n"
                        % (natom, 3, imult, istate, ims, jmult, jstate, jms, ipol)
                    )
                    for atom in range(natom):
                        for xyz in range(3):
                            string += "%s " % (
                                eformat(QMout["dmdr"][ipol][i][j][atom][xyz], 12, 3)
                            )
                        string += "\n"
                    string += ""
                j += 1
            i += 1
        string += "\n"
        return string

    def writeQMoutprop(self):
        """
        Generates a string with the Spin-Orbit Hamiltonian in SHARC format.

        The string starts with a ! followed by a flag specifying the type of data.
        In the next line, the dimensions of the matrix are given, followed by nmstates blocks of nmstates elements.
        Blocks are separated by a blank line.
        """

        QMout = self.QMout
        nmstates = self.QMin.molecule["nmstates"]
        string = ""
        string += "! %i Property Matrix (%ix%i, complex)\n" % (11, nmstates, nmstates)
        string += "%i %i\n" % (nmstates, nmstates)
        for i in range(nmstates):
            for j in range(nmstates):
                string += "%s %s " % (
                    eformat(QMout["prop"][i][j].real, 12, 3),
                    eformat(QMout["prop"][i][j].imag, 12, 3),
                )
            string += "\n"
        string += "\n"

        # print(property matrices (flag 20) in new format)
        string += "! %i Property Matrices\n" % (20)
        string += "%i    ! number of property matrices\n" % (1)

        string += "! Property Matrix Labels (%i strings)\n" % (1)
        string += "Dyson norms\n"

        string += "! Property Matrices (%ix%ix%i, complex)\n" % (1, nmstates, nmstates)
        string += "%i %i   ! Dyson norms\n" % (nmstates, nmstates)
        for i in range(nmstates):
            for j in range(nmstates):
                string += "%s %s " % (
                    eformat(QMout["prop"][i][j].real, 12, 3),
                    eformat(QMout["prop"][i][j].imag, 12, 3),
                )
            string += "\n"
        string += "\n"
        return string

    def writeQMoutmultipolarfit(self):
        """
        Generates a string with the fitted RESP charges for each pair of states specified.

        The string starts with a! followed by a flag specifying the type of data.
        Each line starts with the atom number (starting at 1), state i and state j.
        If i ==j: fit for single state, else fit for transition multipoles.
        One line per atom and a blank line at the end.
        """

        QMout = self.QMout
        states = self.QMin.molecule["states"]
        nmstates = self.QMin.molecule["nmstates"]
        natom = self.QMin.molecule["natom"]
        fits = self.QMin.resources["multipolar_fit"]
        resp_layers = self.QMin.resources["resp_layers"]
        resp_density = self.QMin.resources["resp_density"]
        resp_flayer = self.QMin.resources["resp_first_layer"]
        resp_order = self.QMin.resources["resp_fit_order"]
        resp_grid = self.QMin.resources["resp_grid"]
        setting_str = f" settings [order grid firstlayer density layers] {resp_order} {resp_grid} {resp_flayer} {resp_density} {resp_layers}"
        string = f"! 22 Atomwise multipolar density representation fits for states ({nmstates}x{nmstates}x{natom}x10) {setting_str}\n"

        for i, (imult, istate, ims) in zip(range(nmstates), itnmstates(states)):
            for j, (jmult, jstate, jms) in zip(range(nmstates), itnmstates(states)):
                string += f"{natom} 10 ! m1 {imult} s1 {istate} ms1 {ims: 3.1f}   m2 {jmult} s2 {jstate} ms2 {jms: 3.1f}\n"
                entry = np.zeros((natom, 10))
                if ims != jms or imult != jmult:
                    pass  # ensures that entry stays full of zeros
                elif (imult, istate, jmult, jstate) in fits:
                    fit = fits[(imult, istate, jmult, jstate)]
                    entry[
                        :, : fit.shape[1]
                    ] = fit  # catch cases where fit is not full order
                elif (jmult, jstate, imult, istate) in fits:
                    fit = fits[(jmult, jstate, imult, istate)]
                    entry[
                        :, : fit.shape[1]
                    ] = fit  # catch cases where fit is not full order
                string += (
                    "\n".join(
                        map(
                            lambda x: " ".join(map(lambda y: "{: 10.8f}".format(y), x)),
                            entry,
                        )
                    )
                    + "\n"
                )

                string += ""
        return string

    def writeQMoutTHEODORE(self):
        """
        Write Theodore output
        """

        QMout = self.QMout
        nmstates = self.QMin.molecule["nmstates"]
        nprop = self.QMin["resources"]["theodore_n"]
        if self.QMin["template"]["qmmm"]:
            nprop += len(QMout["qmmm"]["MMEnergy_terms"])
        if nprop <= 0:
            return "\n"

        string = ""

        string += "! %i Property Vectors\n" % (21)
        string += "%i    ! number of property vectors\n" % (nprop)

        string += "! Property Vector Labels (%i strings)\n" % (nprop)
        descriptors = []
        if "theodore" in QMin:
            for i in QMin["resources"]["theodore_prop"]:
                descriptors.append("%s" % i)
                string += descriptors[-1] + "\n"
            for i in range(len(QMin["resources"]["theodore_fragment"])):
                for j in range(len(QMin["resources"]["theodore_fragment"])):
                    descriptors.append("Om_{%i,%i}" % (i + 1, j + 1))
                    string += descriptors[-1] + "\n"
        if QMin["template"]["qmmm"]:
            for label in sorted(QMout["qmmm"]["MMEnergy_terms"]):
                descriptors.append(label)
                string += label + "\n"

        string += "! Property Vectors (%ix%i, real)\n" % (nprop, nmstates)
        if "theodore" in QMin:
            for i in range(QMin["resources"]["theodore_n"]):
                string += "! TheoDORE descriptor %i (%s)\n" % (i + 1, descriptors[i])
                for j in range(nmstates):
                    string += "%s\n" % (eformat(QMout["theodore"][j][i].real, 12, 3))
        if QMin["template"]["qmmm"]:
            for label in sorted(QMout["qmmm"]["MMEnergy_terms"]):
                string += "! QM/MM energy contribution (%s)\n" % (label)
                for j in range(nmstates):
                    string += "%s\n" % (
                        eformat(QMout["qmmm"]["MMEnergy_terms"][label], 12, 3)
                    )
        string += "\n"

        return string

    def writeQmoutPhases(self):
        """"
        Write phases output
        """
        QMout = self.QMout
        string = "! 7 Phases\n%i ! for all nmstates\n" % (
            self.QMin.molecule["nmstates"]
        )
        for i in range(QMin["nmstates"]):
            string += "%s %s\n" % (
                eformat(QMout["phases"][i].real, 9, 3),
                eformat(QMout["phases"][i].imag, 9, 3),
            )
        return string

    def writeQMouttime(self):
        """
        Generates a string with the quantum mechanics total runtime in SHARC format.

        The string starts with a ! followed by a flag specifying the type of data.
        In the next line, the runtime is given.
        """

        QMout = self.QMout
        string = "! 8 Runtime\n%s\n" % (eformat(QMout["runtime"], 9, 3))
        return string

    @abstractmethod
    def printQMout(self):
        '''If PRINT, prints a summary of all requested QM output values.
        Matrices are formatted using printcomplexmatrix, vectors using printgrad.
        '''
        QMout = self.QMout

        states = self.QMin.molecule['states']
        nmstates = self.QMin.molecule['nmstates']
        natom = self.QMin.molecule['natom']
        print('===> Results:\n')
        # Hamiltonian matrix, real or complex
        if self.QMin.requests['h'] or self.QMin.requests['soc']:
            eshift = math.ceil(QMout['h'][0][0].real)
            print('=> Hamiltonian Matrix:\nDiagonal Shift: %9.2f' % (eshift))
            matrix = deepcopy(QMout['h'])
            for i in range(nmstates):
                matrix[i][i] -= eshift
            printcomplexmatrix(matrix, states)
        # Dipole moment matrices
        if self.QMin.requests['dm']:
            print('=> Dipole Moment Matrices:\n')
            for xyz in range(3):
                print('Polarisation %s:' % (IToPol[xyz]))
                matrix = QMout['dm'][xyz]
                printcomplexmatrix(matrix, states)
        # Gradients
        if self.QMin.requests['grad']:
            print('=> Gradient Vectors:\n')
            istate = 0
            for imult, i, ms in itnmstates(states):
                print('%s\t%i\tMs= % .1f:' % (IToMult[imult], i, ms))
                printgrad(QMout['grad'][istate], natom, self.QMin.molecule['elements'], self._DEBUG)
                istate += 1
        # Overlaps
        if self.QMin.requests['overlap']:
            print('=> Overlap matrix:\n')
            matrix = QMout['overlap']
            printcomplexmatrix(matrix, states)
            if 'phases' in QMout:
                print('=> Wavefunction Phases:\n')
                for i in range(nmstates):
                    print('% 3.1f % 3.1f' % (QMout['phases'][i].real, QMout['phases'][i].imag))
                print('\n')
        # Spin-orbit coupling derivatives
        if self.QMin.requests['socdr']:
            print('=> Spin-Orbit Gradient Vectors:\n')
            istate = 0
            for imult, i, ims in itnmstates(states):
                jstate = 0
                for jmult, j, jms in itnmstates(states):
                    print('%s\t%i\tMs= % .1f -- %s\t%i\tMs= % .1f:' % (IToMult[imult], i, ims, IToMult[jmult], j, jms))
                    printgrad(QMout['socdr'][istate][jstate], natom, QMin['geo'])
                    jstate += 1
                istate += 1
        # Dipole moment derivatives
        if self.QMin.requests['dmdr']:
            print('=> Dipole moment derivative vectors:\n')
            istate = 0
            for imult, i, msi in itnmstates(states):
                jstate = 0
                for jmult, j, msj in itnmstates(states):
                    if imult == jmult and msi == msj:
                        for ipol in range(3):
                            print(
                                '%s\tStates %i - %i\tMs= % .1f\tPolarization %s:' %
                                (IToMult[imult], i, j, msi, IToPol[ipol])
                            )
                            printgrad(QMout['dmdr'][ipol][istate][jstate], natom, QMin['geo'])
                    jstate += 1
                istate += 1
        # Property matrix (dyson norms)
        if self.QMin.requests['ion'] and 'prop' in QMout:
            print('=> Property matrix:\n')
            matrix = QMout['prop']
            printcomplexmatrix(matrix, states)
        # TheoDORE
        if self.QMin.requests['theodore']:
            print('=> TheoDORE results:\n')
            matrix = QMout['theodore']
            printtheodore(matrix, QMin)
        sys.stdout.flush()

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
