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
DESCRIPTION = "   HYBRID interface for performing Classical-Path-Approximation (CPA) dynamics." 

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

        # Update template keys
        self.QMin.template.update({"interface": None})
        self.QMin.template.types.update({"interface": dict})
        self.template_file = None
        self.resources_file = None

    def read_resources(self, resources_file: str = "CPA.resources", kw_whitelist: list[str] | None = None) -> None:
        if os.path.isfile(resources_file):
            return super().read_resources(resources_file, kw_whitelist)
        self.log.info("No resource file found. Loading defaults.")
        self._read_resources = True

    def read_template(self, template_file: str = "CPA.template", kw_whitelist: list[str] | None = None) -> None:
        self.log.debug(f"Parsing template file {template_file}")

        # Open template_file and parse yaml
        with open(template_file, "r", encoding="utf-8") as tmpl_file:
            tmpl_dict = yaml.safe_load(tmpl_file)
            self.log.debug(f"Parsing yaml file:\n{tmpl_dict}")

        #Some sanity checks of template file
        if "name" not in tmpl_dict["interface"]:
            self.log.error("No name defined in interface!")
            raise ValueError
        if not isinstance(tmpl_dict["interface"]["name"], str):
            self.log.error("Name must be defined as string!")
            raise ValueError
        if "args" not in tmpl_dict["interface"]:
            tmpl_dict["interface"]["args"] = []
        if not isinstance(tmpl_dict["interface"]["args"], list):
            self.log.error("Args must be a list!")
            raise ValueError
        if "kwargs" not in tmpl_dict["interface"]:
            tmpl_dict["interface"]["kwargs"] = {}
        if not isinstance(tmpl_dict["interface"]["kwargs"], dict):
            self.log.error("Kwargs must be a dictionary!")
            raise ValueError

        self.QMin.template["interface"] = tmpl_dict["interface"]
        self._read_template = True

    def setup_interface(self):
        super().setup_interface()

        with InDir("QM_"+self.QMin.template["interface"]["name"]):
            child = self.QMin.template["interface"]
            self.instantiate_children({"child": (child["name"], child["args"], child["kwargs"])})
            self._kindergarden["child"].setup_mol(self.QMin)
            self._kindergarden["child"].read_resources()
            self._kindergarden["child"].read_template()
            self._kindergarden["child"].QMin.save["savedir"] = os.path.join(self.QMin.save["savedir"], "QM_"+self.QMin.template["interface"]["name"])
            self._kindergarden["child"].QMin.resources["scratchdir"] = os.path.join(self.QMin.resources["scratchdir"], "QM_"+self.QMin.template["interface"]["name"])
            self._kindergarden["child"].QMin.resources["pwd"] = os.path.join(self.QMin.resources["pwd"], "QM")
            self._kindergarden["child"].QMin.resources["cwd"] = os.path.join(self.QMin.resources["cwd"], "QM")
            self._kindergarden["child"].setup_interface()
            #Debugging
            self.log.debug("maps debugging of child")
            self.log.debug(self._kindergarden["child"].QMin.maps)

    def create_restart_files(self) -> None:
        self._kindergarden["child"].create_restart_files()

    def write_step_file(self):
        if not self.persistent:
            super().write_step_file()
        else:
            self.savedict["last_step"] = self.QMin.save["step"]
        self._kindergarden["child"].write_step_file()

    def run(self) -> None:
        with InDir("QM_"+self.QMin.template["interface"]["name"]):
            self._kindergarden["child"].run()

    def getQMout(self):
        self.QMout = self._kindergarden["child"].getQMout() #QMout from child is takes as in, only gradients are adjusted below -> CPA
        self.log.debug("nmstates requested to SHARC_CPA.py")
        self.log.debug(self.QMin.molecule["nmstates"])
        if self.QMin.requests["grad"] is not None:
            if len(self.QMout.grad.shape)==2: #Means child can only provide 1 gradient, so the GS one.
                self.QMout["grad"]=np.array([self.QMout["grad"] for i in range(self.QMin.molecule["nmstates"])]) #Each excited state has same GS gradient
            else:
                self.QMout["grad"]=np.array([self.QMout["grad"][0] for i in range(self.QMin.molecule["nmstates"])]) #retaining only GS gradient if child computes all
        return self.QMout

    def read_requests(self, requests_file="QM.in"):
        super().read_requests(requests_file)
        requests = dict(filter(lambda item: item[1] is not None, self.QMin.requests.items()))
        requests.update(
            {
                "grad": [1], #CPA only requests GS gradient from child.
                "step": self.QMin.save["step"],
            }
        )
        self.log.debug("Requests after handling of CPA interface")
        self.log.debug(requests)
        self._kindergarden["child"].read_requests(requests)

    def set_coords(self, xyz: np.ndarray | list | str, pc: bool = False) -> None:
        super().set_coords(xyz, pc)
        self._kindergarden["child"].set_coords(xyz, pc)

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
            if os.path.isfile("CPA.template"):
                self.log.info("Found CPA.template in current directory")
                if question("Use this template file?", bool, KEYSTROKES=KEYSTROKES, default=True):
                    self.template_file = "CPA.template"
                else:
                    self.template_file = question(
                        "Please specify the path to your CPA.template file", str, KEYSTROKES=KEYSTROKES, default="CPA.template", autocomplete=True)
                    while not os.path.isfile(self.template_file) :
                        self.log.info(f"File {self.template_file} does not exist!")
                        self.template_file = question(
                            "Please specify the path to your CPA.template file", str, KEYSTROKES=KEYSTROKES, default="CPA.template",autocomplete=True)
            else:
                self.template_file = question(
                    "Please specify the path to your CPA.template file", str, KEYSTROKES=KEYSTROKES, default="CPA.template", autocomplete=True)
                while not os.path.isfile(self.template_file) :
                    self.log.info(f"File {self.template_file} does not exist!")
                    self.template_file = question(
                        "Please specify the path to your CPA.template file", str, KEYSTROKES=KEYSTROKES, default="CPA.template",autocomplete=True)
            
            self.read_template(self.template_file)
            #Instantiate child
            child = self.QMin.template["interface"]
            kindergarden = {"child": (child["name"], child["args"], child["kwargs"])}
            self.instantiate_children(kindergarden)

        features = set({"overlap", "phases"}) | self._kindergarden["child"].get_features()
        if features.isdisjoint({"h", "grad"}):
            self.log.error("Child interface must support at least h, and grad request!")
            raise ValueError
        self.log.debug("debugging child features")
        self.log.debug(features)

        return features

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

    def prepare(self, INFOS: dict, dir_path: str) -> None:
        if "link_files" in INFOS:
            os.symlink(expand_path(self.template_file), os.path.join(dir_path, self.name() + ".template"))
        else:
            shutil.copy(self.template_file, os.path.join(dir_path, self.name() + ".template"))

        if not self.QMin.save["savedir"]:
            self.log.warning("savedir not specified, setting savedir to current directory!")
            self.QMin.save["savedir"] = os.getcwd()

        # folder setup and savedir
        self._kindergarden["child"].QMin.save["savedir"] = self.QMin.save["savedir"]
        self._kindergarden["child"].QMin.resources["scratchdir"] = self.QMin.resources["scratchdir"]
        mkdir(child_path := os.path.join(dir_path, "QM"), force=False)
        self._kindergarden["child"].prepare(INFOS, child_path)




if __name__ == "__main__":
    SHARC_CPA().main()
