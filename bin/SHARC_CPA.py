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


import datetime
import os
import shutil
from io import TextIOWrapper

import numpy as np
import yaml
from SHARC_HYBRID import SHARC_HYBRID
from utils import InDir, expand_path, question, mkdir

__all__ = ["SHARC_CPA"]

AUTHORS = "Marco Romanelli"
VERSION = "1.0"
VERSIONDATE = datetime.datetime(2025, 6, 4)
NAME = "CPA"
DESCRIPTION = "HYBRID interface for performing Classical-Path-Approximation (CPA) dynamics. Coding is based on ASE_DB"

CHANGELOGSTRING = """ This hybrid interface only request the ground-state gradients to each child interface and 
return to the driver call always the ground-state gradient for each excited-state. This is meant to be
used for Classical-Path-Approximation (CPA) surface-hopping dynamics.
see J. Chem. Theory Comput. 2013, 9, 4959−4972 for more details on CPA.
"""


class SHARC_CPA(SHARC_HYBRID):
    """
    Hybrid interface for Classical-Path-Approximation (CPA) dynamics. 
    more details on CPA: J. Chem. Theory Comput. 2013, 9, 4959−4972
    """

    _version = VERSION
    _versiondate = VERSIONDATE
    _authors = AUTHORS
    _changelogstring = CHANGELOGSTRING
    _name = NAME
    _description = DESCRIPTION

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Define template
        self.QMin.template.update(
            {"reference": None }
        )
        self.QMin.template.types.update(
            {"reference": dict }
        )

        # Template interface structure
        self._interface_templ = {
            "interface": str,  # Name of SHARC interface
            "args": list,  # Init arguments for child
            "kwargs": dict,  # Keyword args for child
        }

        self.template_file = None

    def read_resources(self, resources_file="CPA.resources", kw_whitelist=None):
        self._read_resources = True

    def read_template(self, template_file="CPA.template", kw_whitelist=None):
        self.log.debug(f"Parsing template file {template_file}")

        # TODO: sanity checks
        # Open template_file file and parse yaml
        with open(template_file, "r", encoding="utf-8") as tmpl_file:
            tmpl_dict = yaml.safe_load(tmpl_file)
            self.log.debug(f"Parsing yaml file:\n{tmpl_dict}")

        if "reference" not in tmpl_dict:
            self.log.error("Reference interface has to be defined!")
            raise ValueError

        for k, v in self._interface_templ.items():
            if k not in tmpl_dict["reference"]:
                self.log.error(f"Key {k} not found.")
                raise ValueError
            if not isinstance(tmpl_dict["reference"][k], v):
                self.log.error(f"Value of key {k} must be of type {v}")
                raise ValueError
        
        self.QMin.template["reference"] = tmpl_dict["reference"]

        # Instantiate reference
        child = self.QMin.template["reference"]
        self.instantiate_children({"reference": (child["interface"], child["args"], child["kwargs"])})

        self._read_template = True

    def setup_interface(self):
        super().setup_interface()
        with InDir("QM_"+self.QMin.template["reference"]["interface"]):
            self._kindergarden["reference"].setup_mol(self.QMin)
            self._kindergarden["reference"].read_resources()
            self._kindergarden["reference"].read_template()
            self._kindergarden["reference"].setup_interface()
            self.log.debug("maps debugging of child")
            self.log.debug(self._kindergarden["reference"].QMin.maps)
            self.log.debug("debugging child scratchdir and savedir")
            self.log.debug(self._kindergarden["reference"].QMin.resources["scratchdir"])
            self.log.debug(self._kindergarden["reference"].QMin.save["savedir"])

    def create_restart_files(self):
        self._kindergarden["reference"].create_restart_files()

    def run(self):
        with InDir("QM_"+self.QMin.template["reference"]["interface"]):
            self._kindergarden["reference"].run()

    def getQMout(self):
        self.QMout = self._kindergarden["reference"].getQMout() #QMout from child is takes as in, only gradients are adjusted below -> CPA
        self.log.debug("nmstates requested to SHARC_CPA.py")
        self.log.debug(self.QMin.molecule["nmstates"])
        if len(self.QMout.grad.shape)==2: #Means child can only provide 1 gradient, so the GS one.
            self.QMout["grad"]=np.array([self.QMout["grad"] for i in range(self.QMin.molecule["nmstates"])]) #Each excited state has some GS gradient
        else:
            self.QMout["grad"]=np.array([self.QMout["grad"][0] for i in range(self.QMin.molecule["nmstates"])]) #retaining only GS gradient if child computes all
        return self.QMout

    def read_requests(self, requests_file="QM.in"):
        super().read_requests(requests_file)
        requests = {}
        for k,v in self.QMin.requests.items():  #to modify requests for child interfaces, only GS gradient can be requested from childs
            if v:
                if k == "grad":
                    requests[k]=[1]
                    continue
                requests[k]=v
        self.log.debug(requests)
        self._kindergarden["reference"].read_requests(requests)

    def set_coords(self, xyz, pc=False):
        super().set_coords(xyz, pc)
        self._kindergarden["reference"].set_coords(xyz, pc)

    @staticmethod
    def authors() -> str:
        return SHARC_CPA._authors

    @staticmethod
    def version() -> str:
        return SHARC_CPA._version

    @staticmethod
    def versiondate():
        return SHARC_CPA._versiondate

    @staticmethod
    def name() -> str:
        return SHARC_CPA._name

    @staticmethod
    def description() -> str:
        return SHARC_CPA._description

    @staticmethod
    def changelogstring() -> str:
        return SHARC_CPA._changelogstring

    def get_features(self, KEYSTROKES: TextIOWrapper | None = None) -> set:
        if not self._read_template:
            self.template_file = question(
                "Please specify the path to your CPA.template file", str, KEYSTROKES=KEYSTROKES, default="CPA.template"
            )

            self.read_template(self.template_file)

        child_features = self._kindergarden["reference"].get_features(KEYSTROKES=KEYSTROKES)
        self.log.debug("debugging child features")
        self.log.debug(child_features)
        return set(child_features)

    def get_infos(self, INFOS: dict, KEYSTROKES: TextIOWrapper | None = None) -> dict:
        self.log.info("=" * 80)
        self.log.info(f"{'||':<78}||")
        self.log.info(f"||{'CPA interface setup':=^76}||\n{'||':<78}||")
        self.log.info("=" * 80)
        self.log.info("\n")

        self.log.info(f"\n{' Setting up child interface ':=^80s}\n")
        self._kindergarden["reference"].QMin.molecule["states"] = INFOS["states"]
        self._kindergarden["reference"].get_infos(INFOS, KEYSTROKES=KEYSTROKES)
        return INFOS

    def prepare(self, INFOS: dict, dir_path: str):
        if "link_files" in INFOS:
            os.symlink(expand_path(self.template_file), os.path.join(dir_path, self.name() + ".template"))
        else:
            shutil.copy(self.template_file, os.path.join(dir_path, self.name() + ".template"))

        if not self.QMin.save["savedir"]:
            self.log.warning("savedir not specified, setting savedir to current directory!")
            self.QMin.save["savedir"] = os.getcwd() #Nothing is gonna save anyway from the SHARC_CPA,py call

        # Calling child prepare routine. Important to specify correct directory where SHARC_VASP.py is gonna be called
        # We don't care about child's scratchdir and savedir here because those are gonna be read in through child.resources otherwise warning will be raised.
        qmdir=os.pathjoin(dir_path,"QM_"+self.QMin.template["reference"]["interface"])
        mkdir(qmdir) 
        self._kindergarden["reference"].prepare(INFOS, qmdir)


if __name__ == "__main__":
    SHARC_CPA().main()
