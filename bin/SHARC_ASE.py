#!/usr/bin/env python3
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

__all__ = ["SHARC_ASE"]

AUTHORS = "Sascha Mausenberger"
VERSION = "4.0"
VERSIONDATE = datetime.datetime(2024, 12, 2)
NAME = "ASE"
DESCRIPTION = "HYBRID interface for saveing data to ASE db"

CHANGELOGSTRING = """
"""

# TODO: features
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
    "mol",
    "wave_functions",
    "density_matrices",
}


class SHARC_ASE(SHARC_HYBRID):
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
        self.QMin.template.update({"reference": None, "props_to_save": None, "ase_file": None, "format": "sharc"})
        self.QMin.template.types.update({"reference": dict, "props_to_save": list, "ase_file": str, "format": str})

        # Template interface structure
        self._interface_templ = {
            "interface": str,  # Name of SHARC interface
            "args": list,  # Init arguments for child
            "kwargs": dict,  # Keyword args for child
        }

        self.resources_file = None
        self.template_file = None

    def read_resources(self, resources_file="ASE.resources", kw_whitelist=None):
        return super().read_resources(resources_file, kw_whitelist)

    def read_template(self, template_file="ASE.template", kw_whitelist=None):
        self.log.debug(f"Parsing template file {template_file}")

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
        self._kindergarden["reference"].run()

    def getQMout(self):
        self.QMout = self._kindergarden["reference"].getQMout()

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
        return SHARC_ASE._authors

    @staticmethod
    def version() -> str:
        return SHARC_ASE._version

    @staticmethod
    def versiondate():
        return SHARC_ASE._versiondate

    @staticmethod
    def name() -> str:
        return SHARC_ASE._name

    @staticmethod
    def description() -> str:
        return SHARC_ASE._description

    @staticmethod
    def changelogstring() -> str:
        return SHARC_ASE._changelogstring

    def get_features(self, KEYSTROKES: TextIOWrapper | None = None) -> set:
        if not self._read_template:
            self.template_file = question(
                "Please specify the path to your ASE.template file", str, KEYSTROKES=KEYSTROKES, default="ASE.template"
            )

            self.read_template(self.template_file)

        child_features = self._kindergarden["reference"].get_features(KEYSTROKES=KEYSTROKES)
        self.log.debug(child_features)
        return set(child_features)

    def get_infos(self, INFOS: dict, KEYSTROKES: TextIOWrapper | None = None) -> dict:
        self.log.info("=" * 80)
        self.log.info(f"{'||':<78}||")
        self.log.info(f"||{'ASE interface setup':=^76}||\n{'||':<78}||")
        self.log.info("=" * 80)
        self.log.info("\n")

        if question("Do you have an ASE.resources file?", bool, KEYSTROKES=KEYSTROKES, autocomplete=False, default=False):
            self.resources_file = question(
                "Specify path to ASE.resources", str, KEYSTROKES=KEYSTROKES, autocomplete=True, default="ASE.resources"
            )

        self.log.info(f"\n{' Setting up child interface ':=^80s}\n")
        self._kindergarden["reference"].QMin.molecule["states"] = INFOS["states"]
        self._kindergarden["reference"].get_infos(INFOS, KEYSTROKES=KEYSTROKES)
        return INFOS

    def prepare(self, INFOS: dict, dir_path: str):
        if "link_files" in INFOS:
            os.symlink(expand_path(self.template_file), os.path.join(dir_path, self.name() + ".template"))
            if "resources_file" in self.__dict__:
                os.symlink(expand_path(self.resources_file), os.path.join(dir_path, self.name() + ".resources"))
        else:
            shutil.copy(self.template_file, os.path.join(dir_path, self.name() + ".template"))
            if "resources_file" in self.__dict__:
                shutil.copy(self.resources_file, os.path.join(dir_path, self.name() + ".resources"))

        if not self.QMin.save["savedir"]:
            self.log.warning("savedir not specified, setting savedir to current directory!")
            self.QMin.save["savedir"] = os.getcwd()

        # folder setup and savedir
        self._kindergarden["reference"].QMin.save["savedir"] = self.QMin.save["savedir"]
        self._kindergarden["reference"].QMin.resources["scratchdir"] = self.QMin.resources["scratchdir"]
        self._kindergarden["reference"].prepare(INFOS, os.getcwd())


if __name__ == "__main__":
    SHARC_ASE().main()
