#!/usr/bin/env python3
import datetime
import os
import shutil
from io import TextIOWrapper

import numpy as np
from SHARC_FAST import SHARC_FAST
from spainn.calculator import SPaiNNulator
from utils import link, question, expand_path

__all__ = ["SHARC_SPAINN"]

AUTHORS = "Sascha Mausenberger"
VERSION = "4.0"
VERSIONDATE = datetime.datetime(2024, 10, 15)
NAME = "SPAINN"
DESCRIPTION = "SHARC interface for SPaiNN"

CHANGELOGSTRING = """
"""

all_features = set(
    [
        "h",
        "dm",
        "grad",
        "nacdr",
    ]
)


class SHARC_SPAINN(SHARC_FAST):
    """
    SHARC interface for SPaiNN
    """

    _version = VERSION
    _versiondate = VERSIONDATE
    _authors = AUTHORS
    _changelogstring = CHANGELOGSTRING
    _name = NAME
    _description = DESCRIPTION

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # Add resource keys
        self.QMin.resources.update({"modelpath": None})
        self.QMin.resources.types.update({"modelpath": str})

        # Add template keys
        self.QMin.template.update(
            {"cutoff": 10.0, "nac_key": "smooth_nacs", "properties": ["energy", "forces", "smooth_nacs", "dipoles"]}
        )
        self.QMin.template.types.update({"cutoff": float, "nac_key": str, "properties": list})

        self.spainnulator = None
        self._resources_file = None
        self._template_file = None

    @staticmethod
    def version() -> str:
        return SHARC_SPAINN._version

    @staticmethod
    def versiondate() -> datetime.datetime:
        return SHARC_SPAINN._versiondate

    @staticmethod
    def changelogstring() -> str:
        return SHARC_SPAINN._changelogstring

    @staticmethod
    def authors() -> str:
        return SHARC_SPAINN._authors

    @staticmethod
    def name() -> str:
        return SHARC_SPAINN._name

    @staticmethod
    def description() -> str:
        return SHARC_SPAINN._description

    @staticmethod
    def about() -> str:
        return f"{SHARC_SPAINN._name}\n{SHARC_SPAINN._description}"

    def get_features(self, KEYSTROKES: TextIOWrapper | None = None) -> set[str]:
        """return availble features

        ---
        Parameters:
        KEYSTROKES: object as returned by open() to be used with question()
        """
        return all_features

    def get_infos(self, INFOS: dict, KEYSTROKES: TextIOWrapper | None = None) -> dict:
        self.log.info("=" * 80)
        self.log.info(f"{'||':<78}||")
        self.log.info(f"||{'SPAINN interface setup':^76}||\n{'||':<78}||")
        self.log.info("=" * 80)
        self.log.info("\n")
        if os.path.isfile("SPAINN.template"):
            self.log.info("Found SPAINN.template in current directory")
            if question("Use this template file?", bool, KEYSTROKES=KEYSTROKES, default=True):
                self._template_file = "SPAINN.template"
        else:
            self.log.info("Specify a path to a SPAINN template file.")
            while not os.path.isfile(template_file := question("Template path:", str, KEYSTROKES=KEYSTROKES)):
                self.log.info(f"File {template_file} does not exist!")
            self._template_file = template_file

        if question("Do you have a SPAINN.resources file?", bool, KEYSTROKES=KEYSTROKES, autocomplete=False, default=False):
            while not os.path.isfile(
                resources_file := question("Specify path to SPAINN.resources", str, KEYSTROKES=KEYSTROKES, autocomplete=True)
            ):
                self.log.info(f"File {resources_file} does not exist!")
            self._resources_file = resources_file
        else:
            self.log.info(f"{'SPAINN ressource usage':-^60}\n")
            self.setupINFOS["modelpath"] = question("Specify path to SPaiNN model: ", str, KEYSTROKES=KEYSTROKES)

    def prepare(self, INFOS: dict, dir_path: str):
        create_file = link if INFOS["link_files"] else shutil.copy
        if not self._resources_file:
            with open(os.path.join(dir_path, "SPAINN.resources"), "w", encoding="utf-8") as file:
                if "modelpath" in self.setupINFOS:
                    file.write(f"modelpath {self.setupINFOS['modelpath']}\n")
                else:
                    self.log.error("Modelpath not specified!")
                    raise ValueError
        else:
            create_file(expand_path(self._resources_file), os.path.join(dir_path, "SPAINN.resources"))
        create_file(expand_path(self._template_file), os.path.join(dir_path, "SPAINN.template"))

    def read_resources(self, resources_file="SPAINN.resources", kw_whitelist=None):
        return super().read_resources(resources_file, kw_whitelist)

    def read_template(self, template_file="SPAINN.template", kw_whitelist=None):
        return super().read_template(template_file, kw_whitelist)

    def setup_interface(self):
        super().setup_interface()
        self.spainnulator = SPaiNNulator(
            atom_types=self.QMin.molecule["elements"],
            modelpath=self.QMin.resources["modelpath"],
            cutoff=self.QMin.template["cutoff"],
            nac_key=self.QMin.template["nac_key"],
            n_states={"n_singlets": self.QMin.molecule["states"][0], "n_triplets": 0},
            properties=self.QMin.template["properties"],
        )

    def create_restart_files(self):
        pass

    def run(self):
        pass

    def getQMout(self):
        requests = set()
        for key, val in self.QMin.requests.items():
            if not val:
                continue
            requests.add(key)

        self.log.debug("Allocate space in QMout object")
        self.QMout.allocate(
            states=self.QMin.molecule["states"],
            natom=self.QMin.molecule["natom"],
            npc=self.QMin.molecule["npc"],
            requests=requests,
        )

        prediction = self.spainnulator.calculate(self.QMin.coords["coords"])
        if self.QMin.requests["h"]:
            self.QMout["h"] = np.asarray(prediction["h"])

        if self.QMin.requests["grad"]:
            self.QMout["grad"] = np.asarray(prediction["grad"])

        if self.QMin.requests["nacdr"]:
            self.QMout["nacdr"] = np.asarray(prediction["nacdr"])

        if self.QMin.requests["dm"]:
            self.QMout["dm"] = np.asarray(prediction["dm"])

        return self.QMout


if __name__ == "__main__":
    SHARC_SPAINN().main()
