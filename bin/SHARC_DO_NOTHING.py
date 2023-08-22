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
from typing import Dict

import numpy as np
from logger import log as logging
# internal
from SHARC_INTERFACE import SHARC_INTERFACE
from utils import Error, makecmatrix

authors = "Sebastian Mai"
version = "3.0"
versiondate = datetime.datetime(2023, 8, 29)
name = "SHARC Do Nothing Interface"
description = "Zero energies/gradients/couplings/etc and unity overlap matrices/phases."

changelogstring = """
"""
np.set_printoptions(linewidth=400, formatter={"float": lambda x: f"{x: 9.7}"})

all_features = {}


class SHARC_DO_NOTHING(SHARC_INTERFACE):
    """
    Do nothing interface
    """

    _version = version
    _versiondate = versiondate
    _authors = authors
    _changelogstring = changelogstring
    _name = name
    _description = description
    _step = 0

    def __init__(self):
        super().__init__()
        self._read_template = False
        self._read_resources = False
        self._setup_mol = False

    def version(self) -> str:
        return self._version

    def versiondate(self) -> str:
        return self._versiondate

    def changelogstring(self) -> str:
        return self._changelogstring

    def authors(self) -> str:
        return self._authors

    def get_features(self) -> dict:
        "return availble features"
        return all_features

    def prepare(self, INFOS: dict):
        "setup the folders"
        return

    def get_infos(self, INFOS: dict) -> dict:
        "prepare INFOS obj"
        return INFOS

    @staticmethod
    def name() -> str:
        return SHARC_DO_NOTHING._name

    @staticmethod
    def description() -> str:
        return SHARC_DO_NOTHING._description

    @staticmethod
    def about() -> str:
        return "Name and description of the interface"

    def create_restart_files(self):
        pass

    def getQMout(self) -> Dict[str, np.ndarray]:
        """
        Generate QMout for all requested requests
        """
        QMout = self.QMout
        nmstates = self.QMin.molecule["nmstates"]
        natom = self.QMin.molecule["natom"]

        if self.QMin.requests["h"] or self.QMin.requests["soc"]:
            if "h" not in QMout:
                QMout["h"] = makecmatrix(nmstates, nmstates)

        if self.QMin.requests["dm"]:
            if "dm" not in QMout:
                QMout["dm"] = [makecmatrix(nmstates, nmstates) for i in range(3)]

        if self.QMin.requests["grad"]:
            if "grad" not in QMout:
                QMout["grad"] = [
                    [[0.0 for i in range(3)] for j in range(natom)]
                    for k in range(nmstates)
                ]
            # TODO: point charges

        if self.QMin.requests["overlap"]:
            if "overlap" not in QMout:
                QMout["overlap"] = makecmatrix(nmstates, nmstates)

        if self.QMin.requests["phases"]:
            if "phases" not in QMout:
                QMout["phases"] = [complex(1.0, 0.0) for i in range(nmstates)]

        if self.QMin.requests["ion"]:
            if "prop" not in QMout:
                QMout["prop"] = makecmatrix(nmstates, nmstates)

        if self.QMin.requests["nacdr"]:
            if "nacdr" not in QMout:
                QMout["nacdr"] = [
                    [
                        [[0.0 for i in range(3)] for j in range(natom)]
                        for k in range(nmstates)
                    ]
                    for l in range(nmstates)
                ]
        if self.QMin.requests["dmdr"]:
            if "dmdr" not in QMout:
                QMout["dmdr"] = [
                    [
                        [
                            [[0.0 for _ in range(3)] for _ in range(natom)]
                            for _ in range(3)
                        ]
                        for _ in range(nmstates)
                    ]
                    for _ in range(nmstates)
                ]

        return QMout

    def writeQMout(self):
        super().writeQMout()

    def printQMout(self):
        pass

    def write_step_file(self):
        pass

    def run(self):
        pass

    def setup_run(self):
        pass

    def read_resources(
        self, resources_file: str = None, kw_whitelist: list = None
    ) -> None:
        """
        Do nothing version of read_resources, takes nothing, returns nothing.
        """
        if not self._setup_mol:
            raise Error("Interface is not setup, call setup_mol first!")

        if self._read_resources:
            logging.warning("Resource file already read.")
        self._read_resources = True

    def read_template(self, template_file: str = None) -> None:
        """
        Do nothing version of read_template, takes nothing, returns nothing.
        """
        if self._read_template:
            logging.warning("Template file already read.")
        self._read_template = True

    def read_requests(self, requests_file: str = "QM.in") -> None:
        """
        Read and check if requests are supported
        """
        super().read_requests(requests_file)
        if any([self.QMin.requests["theodore"], self.QMin.requests["socdr"]]):
            raise Error("SOCDR and theodore not supported!")


if __name__ == "__main__":
    interface = "MOLPRO"
    test = SHARC_DO_NOTHING()
    test.setup_mol(
        f"/user/sascha/development/eci/sharc_main/examples/SHARC_{interface}/QM.in"
    )
    test.read_resources()
    test.read_template()
    test.read_requests(
        f"/user/sascha/development/eci/sharc_main/examples/SHARC_{interface}/QM.in"
    )
    # test.main()
    print(test.getQMout())
    test.writeQMout()
    print(test.QMin)
