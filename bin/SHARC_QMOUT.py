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
import datetime
import os
import shutil
from io import TextIOWrapper

import numpy as np
from logger import log as logging
from qmout import QMout

# internal
from SHARC_FAST import SHARC_FAST
from utils import Error, expand_path, question

AUTHORS = "Sebastian Mai"
VERSION = "4.0"
VERSIONDATE = datetime.datetime(2023, 8, 29)
NAME = "SHARC constant data interface"
DESCRIPTION = "     FAST interface for frozen-nuclei dynamics (constant E/SOC/DM, unit overlaps, zero grad/NAC)"

CHANGELOGSTRING = """
"""

all_features = set(
    [
        "h",
        "soc",
        "dm",
        "grad",
        "nacdr",
        "overlap",
        "multipolar_fit",
        "phases",
        # "ion",
        # "theodore",
        "dmdr",
        "socdr",
    ]
)


class SHARC_QMOUT(SHARC_FAST):
    """
    QM.out interface
    """

    _version = VERSION
    _versiondate = VERSIONDATE
    _authors = AUTHORS
    _changelogstring = CHANGELOGSTRING
    _name = NAME
    _description = DESCRIPTION

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setup_info = None
        self.QMout2 = None

    @staticmethod
    def version() -> str:
        return SHARC_QMOUT._version

    @staticmethod
    def versiondate() -> str:
        return SHARC_QMOUT._versiondate

    @staticmethod
    def changelogstring() -> str:
        return SHARC_QMOUT._changelogstring

    @staticmethod
    def authors() -> str:
        return SHARC_QMOUT._authors

    def get_features(self, KEYSTROKES: TextIOWrapper | None = None) -> set:
        "return availble features"
        return all_features

    def get_infos(self, INFOS: dict, KEYSTROKES: TextIOWrapper | None = None) -> dict:
        "prepare INFOS obj"
        path = question(
            "Please provide path to QM.out file",
            str,
            default="QM.out",
            KEYSTROKES=KEYSTROKES,
            autocomplete=True,
        )
        linking = question("Sym-link the file? (no = copy)?", bool, default=False, KEYSTROKES=KEYSTROKES)
        self.setup_info = {}
        self.setup_info["path"] = expand_path(path)
        self.setup_info["link"] = linking
        return INFOS

    def prepare(self, INFOS: dict, dir_path: str) -> None:
        "setup the folders"
        if self.setup_info["link"]:
            os.symlink(self.setup_info["path"], os.path.join(dir_path, "QMout.template"))
        else:
            shutil.copy(self.setup_info["path"], os.path.join(dir_path, "QMout.template"))

    @staticmethod
    def name() -> str:
        return SHARC_QMOUT._name

    @staticmethod
    def description() -> str:
        return SHARC_QMOUT._description

    @staticmethod
    def about() -> str:
        return f"{SHARC_QMOUT._name}\n{SHARC_QMOUT._description}"

    def create_restart_files(self):
        pass

    def getQMout(self) -> QMout:
        """
        Generate QMout for all requested requests
        """
        # allocate
        requests = set()
        for k, v in self.QMin.requests.items():
            if v in (None, False, []):
                continue
            requests.add(k)
        self.QMout.allocate(
            self.QMin.molecule["states"],
            self.QMin.molecule["natom"],
            self.QMin.molecule["npc"],
            requests,
        )
        if self.QMin.requests["h"] or self.QMin.requests["soc"]:
            self.QMout["h"] = self.QMout2["h"]

        if self.QMin.requests["dm"]:
            self.QMout["dm"] = self.QMout2["dm"]

        if self.QMin.requests["overlap"]:
            np.fill_diagonal(self.QMout["overlap"], 1.0)

        if self.QMin.requests["phases"]:
            self.QMout["phases"] = [complex(1.0, 0.0) for i in range(self.QMout.nmstates)]

        # if self.QMin.requests["ion"]:
        # self.QMout["prop2d"] = self.QMout2["prop2d"]

        # if self.QMin.requests["theodore"]:
        # self.QMout["prop1d"] = self.QMout2["prop1d"]

        if self.QMin.requests["multipolar_fit"]:
            self.QMout["multipolar_fit"] = self.QMout2["multipolar_fit"]

        return self.QMout

    def run(self) -> None:
        pass

    def setup_interface(self):
        # read the file
        self.QMout2 = QMout(filepath="QMout.template")
        self.QMout["notes"]["QMout"] = "Notes were not transferred."
        # check the file
        if any(
            [
                self.QMin.molecule["states"] != self.QMout2.states,
                self.QMin.molecule["natom"] != self.QMout2.natom,
                self.QMin.molecule["npc"] != self.QMout2.npc,
            ]
        ):
            self.log.error("QMin.molecule and QM.out file are inconsistent")
            raise ValueError()

    def read_resources(self, resources_file: str | None = None, kw_whitelist: list[str] | None = None) -> None:
        """
        Do nothing version of read_resources, takes nothing, returns nothing.
        """
        if not self._setup_mol:
            raise Error("Interface is not setup, call setup_mol first!")

        if self._read_resources:
            logging.warning("Resource file already read.")
        self._read_resources = True

    def read_template(self, template_file: str | None = None, kw_whitelist: list[str] | None = None) -> None:
        """
        Do nothing version of read_template, takes nothing, returns nothing.
        """
        if self._read_template:
            logging.warning("Template file already read.")
        self._read_template = True


if __name__ == "__main__":
    SHARC_QMOUT().main()
