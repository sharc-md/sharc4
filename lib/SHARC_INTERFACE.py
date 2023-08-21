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

class CustomFormatter(logging.Formatter):
    err_fmt  = "ERROR: %(msg)s"
    dbg_fmt  = "DEBUG: %(msg)s"
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
    def read_template(self, template_filename):
        pass

    @abstractmethod
    def run(self):
        pass

    @abstractmethod
    def getQMout(self):
        pass

    @abstractmethod
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
                raise ValueError("first line must contain the number of atoms!") from error
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
        
    @abstractmethod
    def setup_mol(self):
        pass

    @abstractmethod
    def read_resources(self):
        pass

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
        '''Prints the formatted header of the log file. Prints version number and version date
        Takes nothing, returns nothing.'''

        print(self.clock.starttime, gethostname(), os.getcwd())
        rule = '=' * 76
        lines = [
            f'  {rule}', '', f'SHARC - {self.name} - Interface', '', f'Authors: {self.authors}', '',
            f'Version: {self.version}', 'Date: {:%d.%m.%Y}'.format(self.versiondate), '', f'  {rule}'
        ]
        # wraps Authors line in case its too long
        lines[4:5] = wrap(lines[4], width=70)
        lines[1:-1] = map(lambda s: '||{:^76}||'.format(s), lines[1:-1])
        print(*lines, sep='\n')
        print('\n')

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