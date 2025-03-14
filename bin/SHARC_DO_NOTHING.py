#!/usr/bin/env python3

# ******************************************
#
#    SHARC Program Suite
#
#    Copyright (c) 2025 University of Vienna
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
from io import TextIOWrapper
from typing import Optional

import numpy as np
from logger import log as logging

# internal
from SHARC_FAST import SHARC_FAST
from utils import Error, makecmatrix, question

__all__ = ["SHARC_DO_NOTHING"]

AUTHORS = "Sebastian Mai"
VERSION = "4.0"
VERSIONDATE = datetime.datetime(2025, 4, 1)
NAME = "SHARC Do Nothing Interface"
DESCRIPTION = "     FAST interface for testing (zero E/grad/NAC/DM, unity overlaps/phases)"

CHANGELOGSTRING = """
"""

class SHARC_DO_NOTHING(SHARC_FAST):
    """
    Do nothing interface
    """

    _version = VERSION
    _versiondate = VERSIONDATE
    _authors = AUTHORS
    _changelogstring = CHANGELOGSTRING
    _name = NAME
    _description = DESCRIPTION
    _step = 0

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setup_info = ""

    @staticmethod
    def version() -> str:
        return SHARC_DO_NOTHING._version

    @staticmethod
    def versiondate() -> datetime.datetime:
        return SHARC_DO_NOTHING._versiondate

    @staticmethod
    def changelogstring() -> str:
        return SHARC_DO_NOTHING._changelogstring

    @staticmethod
    def authors() -> str:
        return SHARC_DO_NOTHING._authors

    def get_features(self, KEYSTROKES: Optional[TextIOWrapper] = None) -> set[str]:
        "return availble features"
        return {
            "h",
            "soc",
            "dm",
            "grad",
            "nacdr",
            "overlap",
            "multipolar_fit",
            "phases",
            "ion",
            "theodore",
            "dmdr",
            "socdr",
            "point_charges",
        }

    def get_infos(
        self, INFOS: dict, KEYSTROKES: Optional[TextIOWrapper] = None
    ) -> dict:
        "prepare INFOS obj"
        self.setup_info = question(
            "Please provide your favorite dish!",
            str,
            default="Pizza",
            KEYSTROKES=KEYSTROKES,
            autocomplete=False,
        )
        return INFOS

    def prepare(self, INFOS: dict, dir_path: str) -> None:
        "setup the folders"
        fpath = os.path.join(dir_path, "Food")
        with open(fpath, "w", encoding="utf-8") as file:
            file.write(self.setup_info)

    @staticmethod
    def name() -> str:
        return SHARC_DO_NOTHING._name

    @staticmethod
    def description() -> str:
        return SHARC_DO_NOTHING._description

    @staticmethod
    def about() -> str:
        return "Name and description of the interface"

    def dyson_orbitals_with_other(self, other):
        pass

    def create_restart_files(self):
        pass

    def getQMout(self) -> dict[str, np.ndarray]:
        """
        Generate QMout for all requested requests
        """
        nmstates = self.QMin.molecule["nmstates"]

        # Allocate arrays in QMout
        requests = set()
        for key, val in self.QMin.requests.items():
            if val in (None, False, []):
                continue
            requests.add(key)

        self.QMout.allocate(
            self.QMin.molecule["states"],
            self.QMin.molecule["natom"],
            self.QMin.molecule["npc"],
            requests,
        )

        if self.QMin.requests["overlap"]:
            np.fill_diagonal(self.QMout["overlap"], 1.0)

        if self.QMin.requests["phases"]:
            self.QMout["phases"] = [complex(1.0, 0.0) for i in range(nmstates)]

        if self.QMin.requests["ion"]:
            self.QMout["prop2d"] = [("Dyson norms", makecmatrix(nmstates, nmstates))]

        if self.QMin.requests["theodore"]:
            self.QMout["prop1d"] = [("Om", [0.0 for i in range(nmstates)])]

        if self.QMin.molecule["point_charges"]:
            self.QMout["prop0d"] = [("MMen", 0.0)]

        self.QMout["notes"]["Do nothing"] = "This is a note."

        return self.QMout

    def write_step_file(self):
        pass

    def run(self):
        pass

    def setup_interface(self):
        pass

    def read_resources(
        self, resources_file: Optional[str] = None, kw_whitelist: Optional[list] = None
    ) -> None:
        """
        Do nothing version of read_resources, takes nothing, returns nothing.
        """
        if not self._setup_mol:
            raise Error("Interface is not setup, call setup_mol first!")

        if self._read_resources:
            logging.warning("Resource file already read.")
        self._read_resources = True

    def read_template(self, template_file: Optional[str] = None) -> None:
        """
        Do nothing version of read_template, takes nothing, returns nothing.
        """
        if self._read_template:
            logging.warning("Template file already read.")
        self._read_template = True


if __name__ == "__main__":
    np.set_printoptions(linewidth=400, formatter={"float": lambda x: f"{x: 9.7}"})
    test = SHARC_DO_NOTHING()
    test.main()
