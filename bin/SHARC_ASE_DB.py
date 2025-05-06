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

import ase
import numpy as np
import yaml
from ase.db import connect
from SHARC_HYBRID import SHARC_HYBRID
from utils import expand_path, question

__all__ = ["SHARC_ASE_DB"]

AUTHORS = "Sascha Mausenberger"
VERSION = "4.0"
VERSIONDATE = datetime.datetime(2024, 12, 2)
NAME = "ASE_DB"
DESCRIPTION = "   HYBRID interface for saveing data to ASE db"

CHANGELOGSTRING = """
"""


class SHARC_ASE_DB(SHARC_HYBRID):
    """
    Adaptive sampling interface for SHARC 4.0
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
            {"reference": None, "props_to_save": None, "ase_file": None, "format": "sharc", "output_steps": 1}
        )
        self.QMin.template.types.update(
            {"reference": dict, "props_to_save": list, "ase_file": str, "format": str, "output_steps": int}
        )

        # Template interface structure
        self._interface_templ = {
            "interface": str,  # Name of SHARC interface
            "args": list,  # Init arguments for child
            "kwargs": dict,  # Keyword args for child
        }

        self.template_file = None

    def read_resources(self, resources_file="ASE_DB.resources", kw_whitelist=None):
        self._read_resources = True

    def read_template(self, template_file="ASE_DB.template", kw_whitelist=None):
        self.log.debug(f"Parsing template file {template_file}")

        # TODO: sanity checks
        # Open template_file file and parse yaml
        with open(template_file, "r", encoding="utf-8") as tmpl_file:
            tmpl_dict = yaml.safe_load(tmpl_file)
            self.log.debug(f"Parsing yaml file:\n{tmpl_dict}")

        if "props_to_save" not in tmpl_dict:
            self.log.error("Template file must contain props_to_save list!")
            raise ValueError
        self.QMin.template["props_to_save"] = tmpl_dict["props_to_save"]

        if "ase_file" not in tmpl_dict:
            self.log.error("Template file has to contain ase_file!")
            raise ValueError
        self.QMin.template["ase_file"] = tmpl_dict["ase_file"]

        if "reference" not in tmpl_dict:
            self.log.error("Reference interface has to be defined!")
            raise ValueError

        if "format" in tmpl_dict:
            if tmpl_dict["format"].lower() not in ("sharc", "spainn"):
                self.log.error(f"{tmpl_dict['format']} is not a valid format! Either use SHARC or SPAINN.")
                raise ValueError

        if "output_steps" in tmpl_dict:
            self.QMin.template["output_steps"] = tmpl_dict["output_steps"]

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
        self._kindergarden["reference"].setup_mol(self.QMin)
        self._kindergarden["reference"].read_resources()
        self._kindergarden["reference"].read_template()
        self._kindergarden["reference"].setup_interface()

    def create_restart_files(self):
        self._kindergarden["reference"].create_restart_files()

    def run(self):
        self._kindergarden["reference"].QMin.coords["coords"] = self.QMin.coords["coords"].copy()
        for key, value in self.QMin.requests.items():
            if value is not None:
                self._kindergarden["reference"].QMin.requests[key] = value
        self._kindergarden["reference"].QMin.save["step"] = self.QMin.save["step"]
        self._kindergarden["reference"]._step_logic()
        self._kindergarden["reference"]._request_logic()
        self._kindergarden["reference"].run()

    def getQMout(self):
        self.QMout = self._kindergarden["reference"].getQMout()
        if self.QMin.save["step"] % self.QMin.template["output_steps"] != 0:
            return self.QMout

        with connect(self.QMin.template["ase_file"]) as db:
            if self.QMin.template["format"].lower() == "spainn":
                data = {}
                for prop in self.QMin.template["props_to_save"]:
                    match prop:
                        case "h":
                            data["energy"] = np.einsum("ii->i", self.QMout[prop])
                        case "grad":
                            data["forces"] = -np.einsum("ijk->jik", self.QMout[prop])
                        case "nacdr":
                            data["nacs"] = np.einsum("ijk->jik", self.QMout[prop])
                        case "dm":
                            idx = np.triu_indices(self.QMin.molecule["nmstates"])
                            data["dipoles"] = self.QMout[prop][idx]
                        case "grad_pc":
                            data["forces_pc"] = -np.einsum("ijk->jik", self.QMout[prop])
                        case "nacdr_pc":
                            data["nacs_pc"] = np.einsum("ijk->jik", self.QMout[prop])
                        case _:
                            data[prop] = self.QMout[prop]
                if self.QMin.molecule["point_charges"]:
                    data["external_charge_positions"] = self.QMin.coords["pccoords"]
                    data["external_charges"] = self.QMin.coords["pccharge"]
            else:
                if self.QMin.molecule["point_charges"]:
                    data["pccoords"] = self.QMin.coords["pccoords"]
                    data["pccharge"] = self.QMin.coords["pccharge"]
                data = {k: self.QMout[k] for k in self.QMin.template["props_to_save"]}

            db.write(ase.Atoms(symbols=self.QMin.molecule["elements"], positions=self.QMin.coords["coords"]), data=data)
        return self.QMout

    def read_requests(self, requests_file="QM.in"):
        super().read_requests(requests_file)
        self._kindergarden["reference"].read_requests(requests_file)

    def set_coords(self, xyz, pc=False):
        super().set_coords(xyz, pc)
        self._kindergarden["reference"].set_coords(xyz, pc)

    @staticmethod
    def authors() -> str:
        return SHARC_ASE_DB._authors

    @staticmethod
    def version() -> str:
        return SHARC_ASE_DB._version

    @staticmethod
    def versiondate():
        return SHARC_ASE_DB._versiondate

    @staticmethod
    def name() -> str:
        return SHARC_ASE_DB._name

    @staticmethod
    def description() -> str:
        return SHARC_ASE_DB._description

    @staticmethod
    def changelogstring() -> str:
        return SHARC_ASE_DB._changelogstring

    def get_features(self, KEYSTROKES: TextIOWrapper | None = None) -> set:
        if not self._read_template:
            self.template_file = question(
                "Please specify the path to your ASE_DB.template file", str, KEYSTROKES=KEYSTROKES, default="ASE_DB.template"
            )

            self.read_template(self.template_file)

        child_features = self._kindergarden["reference"].get_features(KEYSTROKES=KEYSTROKES)
        self.log.debug(child_features)
        return set(child_features)

    def get_infos(self, INFOS: dict, KEYSTROKES: TextIOWrapper | None = None) -> dict:
        self.log.info("=" * 80)
        self.log.info(f"{'||':<78}||")
        self.log.info(f"||{'ASE_DB interface setup':=^76}||\n{'||':<78}||")
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
            self.QMin.save["savedir"] = os.getcwd()

        # folder setup and savedir
        self._kindergarden["reference"].QMin.save["savedir"] = self.QMin.save["savedir"]
        self._kindergarden["reference"].QMin.resources["scratchdir"] = self.QMin.resources["scratchdir"]
        self._kindergarden["reference"].prepare(INFOS, dir_path)


if __name__ == "__main__":
    SHARC_ASE_DB().main()
