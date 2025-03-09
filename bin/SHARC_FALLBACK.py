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
import shutil

import yaml
from SHARC_HYBRID import SHARC_HYBRID
from utils import InDir, expand_path, mkdir, question

__all__ = ["SHARC_FALLBACK"]

AUTHORS = "Sascha Mausenberger"
VERSION = "4.0"
VERSIONDATE = datetime.datetime(2024, 10, 31)
NAME = "FALLBACK"
DESCRIPTION = "   HYBRID interface for calling a fallback interface if primary interface fails"

CHANGELOGSTRING = """
"""

# all_features = {
#     "h",
#     "soc",
#     "dm",
#     "grad",
#     "nacdr",
#     "overlap",
#     "phases",
#     "ion",
#     "dmdr",
#     "socdr",
#     "multipolar_fit",
#     "theodore",
#     "point_charges",
#     # raw data request
#     "mol",
#     "wave_functions",
#     "density_matrices",
# }


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
        self._nfails = 0
        self._nsuccesses = 0
        self._nfails_total = 0

        # Define template keys
        self.QMin.template.update(
            {"trial_interface": None, "fallback_interface": None, "stop_at_nfails": 2, "reset_fail_counter": 1}
        )
        self.QMin.template.types.update(
            {"trial_interface": dict, "fallback_interface": dict, "stop_at_nfails": int, "reset_fail_counter": int}
        )

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

    # ----------------------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------------

    def get_features(self, KEYSTROKES: TextIOWrapper | None = None) -> set:
        # return all_features
        if not self._read_template:
            self.template_file = question(
                "Please specify the path to your FALLBACK.template file", str, KEYSTROKES=KEYSTROKES, default="FALLBACK.template"
            )
            self.read_template(self.template_file)
        if self._trial_interface is None:
            prog = self.QMin.template["trial_interface"]["interface"]
            self._trial_interface = self._load_interface(prog)()
        if self._fallback_interface is None:
            prog = self.QMin.template["fallback_interface"]["interface"]
            self._fallback_interface = self._load_interface(prog)()

        trial_features = self._trial_interface.get_features(KEYSTROKES=KEYSTROKES)
        fallback_features = self._fallback_interface.get_features(KEYSTROKES=KEYSTROKES)
        own_features = trial_features & fallback_features - set(["overlap", "phases"])
        self.log.debug(own_features)  # log features
        return set(own_features)

    def get_infos(self, INFOS: dict, KEYSTROKES: TextIOWrapper | None = None) -> dict:
        # Setup some output to log
        self.log.info("=" * 80)
        self.log.info(f"{'||':<78}||")
        self.log.info(f"||{'FALLBACK interface setup':^76}||\n{'||':<78}||")
        self.log.info("=" * 80)
        self.log.info("\n")

        # Get the infos from the child
        self.log.info(f"{' Setting up Trial interface ':=^80s}\n")
        self._trial_interface.get_infos(INFOS, KEYSTROKES=KEYSTROKES)

        # Get the infos from the child
        self.log.info(f"{' Setting up Fallback interface ':=^80s}\n")
        self._fallback_interface.get_infos(INFOS, KEYSTROKES=KEYSTROKES)

        return INFOS

    def prepare(self, INFOS: dict, dir_path: str):
        QMin = self.QMin

        # template
        if "link_files" in INFOS:
            os.symlink(expand_path(self.template_file), os.path.join(dir_path, self.name() + ".template"))
        else:
            shutil.copy(self.template_file, os.path.join(dir_path, self.name() + ".template"))

        # make empty resources file
        path = os.path.join(dir_path, "FALLBACK.resources")
        open(path, "a").close()

        # setup dirs
        traildir = os.path.join(dir_path, "trial_interface")
        fallbackdir = os.path.join(dir_path, "fallback_interface")
        mkdir(traildir)
        mkdir(fallbackdir)

        self._trial_interface.prepare(INFOS, traildir)
        self._fallback_interface.prepare(INFOS, fallbackdir)

    # ----------------------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------------

    def read_resources(self, resources_file="FALLBACK.resources", kw_whitelist=None):
        self._read_resources = True

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

    # ----------------------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------------

    def run(self):
        self._trial_failed = False
        self._trial_interface.QMin.save["step"] = self.QMin.save["step"] - self._nfails_total
        self._fallback_interface.QMin.save["step"] = self._nfails_total
        try:
            with InDir("trial_interface"):
                self._trial_interface.run()
                self._trial_interface.getQMout()
        except:  # pylint: disable=bare-except
            self.log.info("Trial interface failed, running fallback.")
            self._trial_failed = True
            self._nfails_total += 1
            self._nfails += 1
            if self._nfails > self.QMin.template["stop_at_nfails"]:
                raise
            with InDir("fallback_interface"):
                self._fallback_interface.run()
                self._fallback_interface.getQMout()
        else:
            self._nsuccesses += 1
            if self._nsuccesses >= self.QMin.template["reset_fail_counter"]:
                self._nfails = 0
        

    def create_restart_files(self):
        super().create_restart_files()
        self._trial_interface.create_restart_files()
        self._fallback_interface.create_restart_files()

    def write_step_file(self):
        super().write_step_file()
        self._trial_interface.write_step_file()
        self._fallback_interface.write_step_file()

    def clean_savedir(self):
        super().clean_savedir()
        self._trial_interface.clean_savedir()
        self._fallback_interface.clean_savedir()

    # ----------------------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------------

    def setup_interface(self):
        if not os.path.isdir("fallback_interface"):
            self.log.error("Path fallback_interface does not exist!")
            raise ValueError
        if not os.path.isdir("trial_interface"):
            self.log.error("Path trial_interface does not exist!")
            raise ValueError

        # Counter of fails
        self._nfails = 0
        self._nsuccesses = 0
        self._nfails_total = 0

        # Instantiate trial and fallback interfaces
        trial = self.QMin.template["trial_interface"]
        fallback = self.QMin.template["fallback_interface"]

        with InDir("trial_interface"):
            self._trial_interface = self._load_interface(trial["interface"])(*trial["args"], **trial["kwargs"])
            self._trial_interface.setup_mol(self.QMin)
            self._trial_interface.read_resources()
            self._trial_interface.read_template()
            self._trial_interface.setup_interface()

        with InDir("fallback_interface"):
            self._fallback_interface = self._load_interface(fallback["interface"])(*fallback["args"], **fallback["kwargs"])
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
