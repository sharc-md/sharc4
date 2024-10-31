#!/usr/bin/env python3
import datetime
import os
from importlib import import_module
from io import TextIOWrapper

import numpy as np
import yaml
from SHARC_HYBRID import SHARC_HYBRID
from utils import InDir

__all__ = ["SHARC_ADAPTIVE"]

AUTHORS = "Sascha Mausenberger"
VERSION = "4.0"
VERSIONDATE = datetime.datetime(2024, 10, 17)
NAME = "Adaptive sampling"
DESCRIPTION = "SHARC 4.0 interface for adaptive sampling"

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


class SHARC_ADAPTIVE(SHARC_HYBRID):
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
            {
                "thresholds": {},  # Contains threshold values for one or more properties
                "error_function": "mae",  # MAE, MSE, RMSE, ...
                "exit_on_fail": True,  # Raise exception if threshold exceeded
                "write_geoms": True,  # Write current geometry to file if threshold exceeded
                "geom_file": "geoms.xyz",  # Name of geom file
                "interfaces": [],  # List of child parameters
                "custom_error": {},  # Dictionary with custom loss function
            }
        )
        self.QMin.template.types.update(
            {
                "thresholds": dict,
                "error_function": str,
                "exit_on_fail": bool,
                "write_geoms": bool,
                "geom_file": str,
                "interfaces": list,
                "custom_error": dict,
            }
        )

        # Resources interface structure
        self._interface_templ = {
            "label": str,  # Label of the child
            "interface": str,  # Name of SHARC interface
            "args": list,  # Init arguments for child
            "kwargs": dict,  # Keyword args for child
        }

        # Supported properties
        self._valid_props = ("h", "soc", "dm", "grad", "nacdr")

        # Loss functions
        self._error_function = {
            "mae": SHARC_ADAPTIVE._mae,
            "mae_max": SHARC_ADAPTIVE._mae_max,
            "mse": SHARC_ADAPTIVE._mse,
            "mse_max": SHARC_ADAPTIVE._mse_max,
            "rmse": SHARC_ADAPTIVE._rmse,
        }

    @staticmethod
    def _rmse(a: np.ndarray, b: np.ndarray) -> float:
        return np.sqrt(SHARC_ADAPTIVE._mse(a, b))

    @staticmethod
    def _mse_max(a: np.ndarray, b: np.ndarray) -> float:
        return np.square(a - b).max()

    @staticmethod
    def _mse(a: np.ndarray, b: np.ndarray) -> float:
        return np.square(a - b).mean()

    @staticmethod
    def _mae_max(a: np.ndarray, b: np.ndarray) -> float:
        return np.abs(a - b).max()

    @staticmethod
    def _mae(a: np.ndarray, b: np.ndarray) -> float:
        return np.abs(a - b).mean()

    def _import_loss(self, file: str, function: str) -> callable:
        try:
            module = import_module(file)
        except (ModuleNotFoundError, ImportError, TypeError):
            self.log.error(f"{file} could not be imported!")
            raise
        try:
            loss = getattr(module, function)
        except AttributeError as exc:
            self.log.error(f"Function {function} not found in {module}")
            raise AttributeError from exc
        return loss

    def read_resources(self, resources_file="ADAPTIVE.resources", kw_whitelist=None):
        super().read_resources(resources_file, kw_whitelist)

    def read_template(self, template_file="ADAPTIVE.template", kw_whitelist=None):
        self.log.debug(f"Parsing template file {template_file}")

        # Open template_file file and parse yaml
        with open(template_file, "r", encoding="utf-8") as tmpl_file:
            tmpl_dict = yaml.safe_load(tmpl_file)
            self.log.debug(f"Parsing yaml file:\n{tmpl_dict}")

        # At least threshold must be included
        if "thresholds" not in tmpl_dict:
            self.log.error("Teplate file must contain thresholds!")
            raise ValueError

        for k, v in tmpl_dict["thresholds"].items():
            if k.lower() not in self._valid_props:
                self.log.error(f"{k} is not a supported property, supported properties are {self._valid_props}")
                raise ValueError
            try:
                self.QMin.template["thresholds"][k] = float(v)
            except ValueError as exc:
                self.log.error(f"Invalid threshold value {v} for {k}, value must be float!")
                raise ValueError from exc
        for key in ("error_function", "exit_on_fail", "write_geoms", "geom_file"):
            if key in tmpl_dict:
                self.QMin.template[key] = tmpl_dict[key]

        # Check custom loss
        if "custom_error" in tmpl_dict:
            if tmpl_dict["custom_error"].keys() != {"name", "file", "function"}:
                self.log.error("custom_error dictionary must contain keys name, file and function!")
                raise ValueError
            self._error_function[tmpl_dict["custom_error"]["name"]] = self._import_loss(
                tmpl_dict["custom_error"]["file"], tmpl_dict["custom_error"]["function"]
            )

        if self.QMin.template["error_function"].lower() not in self._error_function:
            self.log.error("Invalid error function!")
            raise ValueError

        if "interfaces" not in tmpl_dict:
            self.log.error("Resources file does not contain interfaces list!")
            raise ValueError

        if len(tmpl_dict["interfaces"]) != 2:  # TODO: more than two interfaces
            self.log.error("Currently this interface only works with two childs!")
            raise ValueError
        if len(tmpl_dict["interfaces"]) < 2:
            self.log.error("Interfaces list in resource file must contain at least two entries!")
            raise ValueError

        # Check interface dict
        for k, v in self._interface_templ.items():
            for child in tmpl_dict["interfaces"]:
                if k not in child:
                    self.log.error(f"Key {k} not found.")
                    raise ValueError
                if not isinstance(child[k], v):
                    self.log.error(f"Value of key {k} must be of type {v}")
                    raise ValueError
        self.QMin.template["interfaces"] = tmpl_dict["interfaces"]

        self._read_template = True

    def setup_interface(self):
        """
        Prepare the interfaces for calculations
        """
        # Instantiate children
        for child in self.QMin.template["interfaces"]:
            # Check if child directory exists
            if not os.path.isdir(path := os.path.join(self.QMin.resources["pwd"], child["label"])):
                self.log.error(f"{path} does not exist!")
                raise ValueError
            with InDir(path):
                self.instantiate_children({child["label"]: (child["interface"], child["args"], child["kwargs"])})

        for child, instance in self._kindergarden.items():
            self.log.debug(f"Setup child {child}")
            with InDir(os.path.join(self.QMin.resources["pwd"], child)):
                instance.setup_mol(self.QMin)
                instance.read_resources()
                instance.read_template()
                instance.setup_interface()

    def read_requests(self, requests_file="QM.in"):
        super().read_requests(requests_file)

        for req in self.QMin.template["thresholds"]:
            if not self.QMin.requests[req]:
                self.log.error(f"Threshold for {req} specified, but is not requested!")
                raise ValueError

        for k, v in self._kindergarden.items():
            self.log.debug(f"Set requests to child {k}")
            v.read_requests(requests_file)

    def set_coords(self, xyz, pc=False):
        super().set_coords(xyz, pc)
        for k, v in self._kindergarden.items():
            self.log.debug(f"Set coords to child {k}")
            v.set_coords(xyz, pc)

    def run(self):
        self.run_children(self.log, self._kindergarden, self.QMin.resources["ncpu"])

    def getQMout(self):
        """
        Check if thresholds for specified properties are exceeded
        Return QMout object of first child
        """
        for prop, thres in self.QMin.template["thresholds"].items():
            child = iter(self._kindergarden.values())
            err_func = self._error_function[self.QMin.template["error_function"]]
            match prop:
                case "h":
                    error = err_func(np.einsum("ii->i", next(child).QMout["h"]), np.einsum("ii->i", next(child).QMout["h"]))
                case "soc":
                    # Set diagonals to zero, then calculate error
                    qmin1 = next(child).QMout["h"].copy()
                    qmin2 = next(child).QMout["h"].copy()
                    np.einsum("ii->i", qmin1)[:] = 0.0
                    np.einsum("ii->i", qmin2)[:] = 0.0
                    error = err_func(qmin1, qmin2)
                case "nacdr" | "dm":
                    reference = next(child).QMout
                    error = err_func(reference[prop], self.phase_correct(reference[prop], next(child).QMout[prop]))
                case _:
                    error = err_func(next(child).QMout[prop], next(child).QMout[prop])
            if error > thres:
                self.log.info(f"Threshold for {prop} exceeded, error = {error}")
                if self.QMin.template["write_geoms"]:
                    with open(self.QMin.template["geom_file"], "a", encoding="utf-8") as geom:
                        geom.write(f"{self.QMin.molecule['natom']}\n\n")
                        for atom, coords in zip(self.QMin.molecule["elements"], self.QMin.coords["coords"]):
                            geom.write(f"{atom}\t{coords[0]:15.10f}\t{coords[1]:15.10f}\t{coords[2]:15.10f}\n")
                if self.QMin.template["exit_on_fail"]:
                    raise ValueError
            else:
                self.log.debug(f"Error of {prop} is {error}")

        self.QMout = next(iter(self._kindergarden.values())).QMout
        return self.QMout

    def create_restart_files(self):
        for child in self._kindergarden.values():
            child.create_restart_files()

    @staticmethod
    def phase_correct(reference: np.ndarray, secondary: np.ndarray) -> np.ndarray:
        """
        Adjust phase from secondary to reference and return array
        """
        # Make copy of secondary
        corrected = secondary.copy()
        for idx, (ref, sec) in enumerate(zip(reference, secondary)):
            if np.vdot(sec, ref) < 0:
                corrected[idx] *= -1
        return corrected

    @staticmethod
    def authors() -> str:
        return SHARC_ADAPTIVE._authors

    @staticmethod
    def version() -> str:
        return SHARC_ADAPTIVE._version

    @staticmethod
    def versiondate():
        return SHARC_ADAPTIVE._versiondate

    @staticmethod
    def name() -> str:
        return SHARC_ADAPTIVE._name

    @staticmethod
    def description() -> str:
        return SHARC_ADAPTIVE._description

    @staticmethod
    def changelogstring() -> str:
        return SHARC_ADAPTIVE._changelogstring

    def get_features(self, KEYSTROKES: TextIOWrapper | None = None) -> set:
        return all_features

    def get_infos(self, INFOS: dict, KEYSTROKES: TextIOWrapper | None = None) -> dict:
        return None

    def prepare(self, INFOS: dict, dir_path: str):
        return


if __name__ == "__main__":
    SHARC_ADAPTIVE().main()
