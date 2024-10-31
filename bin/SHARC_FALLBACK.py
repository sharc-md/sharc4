#!/usr/bin/env python3
import datetime
import os
from io import TextIOWrapper

import yaml
from SHARC_HYBRID import SHARC_HYBRID
from utils import InDir

__all__ = ["SHARC_FALLBACK"]

AUTHORS = "Sascha Mausenberger"
VERSION = "4.0"
VERSIONDATE = datetime.datetime(2024, 10, 31)
NAME = "Fallback"
DESCRIPTION = "SHARC 4.0 interface for Fallback"

CHANGELOGSTRING = """
"""

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


class SHARC_FALLBACK(SHARC_HYBRID):
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

        self._trial_interface = None
        self._fallback_interface = None
        self._trial_failed = False

        # Define template keys
        self.QMin.template.update({"trial_interface": None, "fallback_interface": None})
        self.QMin.template.types.update({"trial_interface": dict, "fallback_interface": dict})

        # Template interface structure
        self._interface_templ = {
            "interface": str,  # Name of SHARC interface
            "args": list,  # Init arguments for child
            "kwargs": dict,  # Keyword args for child
        }

    @staticmethod
    def authors() -> str:
        return SHARC_FALLBACK._authors

    @staticmethod
    def version() -> str:
        return SHARC_FALLBACK._version

    @staticmethod
    def versiondate():
        return SHARC_FALLBACK._versiondate

    @staticmethod
    def name() -> str:
        return SHARC_FALLBACK._name

    @staticmethod
    def description() -> str:
        return SHARC_FALLBACK._description

    @staticmethod
    def changelogstring() -> str:
        return SHARC_FALLBACK._changelogstring

    def get_features(self, KEYSTROKES: TextIOWrapper | None = None) -> set:
        return all_features

    def get_infos(self, INFOS: dict, KEYSTROKES: TextIOWrapper | None = None) -> dict:
        return None

    def prepare(self, INFOS: dict, dir_path: str):
        return

    def read_resources(self, resources_file="FALLBACK.resources", kw_whitelist=None):
        return super().read_resources(resources_file, kw_whitelist)

    def read_template(self, template_file="FALLBACK.template", kw_whitelist=None):
        self.log.debug(f"Parsing template file {template_file}")

        # Open template_file file and parse yaml
        with open(template_file, "r", encoding="utf-8") as tmpl_file:
            tmpl_dict = yaml.safe_load(tmpl_file)
            self.log.debug(f"Parsing yaml file:\n{tmpl_dict}")

        if "trial_interface" not in tmpl_dict or "fallback_interface" not in tmpl_dict:
            self.log.error("Template file must contain trial_interface and fallback_interface!")
            raise ValueError

        # Check interface dict
        for k, v in self._interface_templ.items():
            if k not in tmpl_dict["trial_interface"]:
                self.log.error(f"Key {k} not found in trial_interface.")
            if k not in tmpl_dict["fallback_interface"]:
                self.log.error(f"Key {k} not found in fallback_interface.")
                raise ValueError
            if not isinstance(tmpl_dict["trial_interface"][k], v):
                self.log.error(f"Value of key {k} in trial_interface must be of type {v}")
                raise ValueError
            if not isinstance(tmpl_dict["fallback_interface"][k], v):
                self.log.error(f"Value of key {k} in fallback_interface must be of type {v}")
                raise ValueError
        self.QMin.template["fallback_interface"] = tmpl_dict["fallback_interface"]
        self.QMin.template["trial_interface"] = tmpl_dict["trial_interface"]
        self._read_template = True

    def run(self):
        self._trial_failed = False
        try:
            with InDir(os.path.join(self.QMin.resources["pwd"], "trial_interface")):
                self._trial_interface.run()
                self._trial_interface.getQMout()
        except:  # pylint: disable=bare-except
            self.log.info("Trial interface failed, running fallback.")
            self._trial_failed = True
            with InDir(os.path.join(self.QMin.resources["pwd"], "fallback_interface")):
                self._fallback_interface.run()
                self._fallback_interface.getQMout()

    def create_restart_files(self):
        super().create_restart_files()
        self._trial_interface.create_restart_files()
        self._fallback_interface.create_restart_files()

    def setup_interface(self):
        if not os.path.isdir(path := os.path.join(self.QMin.resources["pwd"], "fallback_interface")):
            self.log.error(f"{path} does not exist!")
            raise ValueError
        if not os.path.isdir(path := os.path.join(self.QMin.resources["pwd"], "trial_interface")):
            self.log.error(f"{path} does not exist!")
            raise ValueError

        # Instantiate trial and fallback interfaces
        trial = self.QMin.template["trial_interface"]
        fallback = self.QMin.template["fallback_interface"]

        with InDir(os.path.join(self.QMin.resources["pwd"], "trial_interface")):
            self._trial_interface = self._load_interface(trial["interface"])(trial["args"], trial["kwargs"])
            self._trial_interface.setup_mol(self.QMin)
            self._trial_interface.read_resources()
            self._trial_interface.read_template()
            self._trial_interface.setup_interface()

        with InDir(os.path.join(self.QMin.resources["pwd"], "fallback_interface")):
            self._fallback_interface = self._load_interface(fallback["interface"])(fallback["args"], fallback["kwargs"])
            self._fallback_interface.setup_mol(self.QMin)
            self._fallback_interface.read_resources()
            self._fallback_interface.read_template()
            self._fallback_interface.setup_interface()

    def read_requests(self, requests_file="QM.in"):
        super().read_requests(requests_file)
        self._trial_interface.read_requests(requests_file)
        self._fallback_interface.read_requests(requests_file)

    def set_coords(self, xyz, pc=False):
        super().set_coords(xyz, pc)
        self._trial_interface.set_coords(xyz, pc)
        self._fallback_interface.set_coords(xyz, pc)

    def getQMout(self):
        if self._trial_failed:
            self.QMout = self._fallback_interface.QMout
            return self.QMout
        self.QMout = self._trial_interface.QMout
        return self.QMout


if __name__ == "__main__":
    SHARC_FALLBACK().main()
