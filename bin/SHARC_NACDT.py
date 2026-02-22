#!/usr/bin/env python3

# ******************************************
#
#    SHARC Program Suite
#
#    Copyright (c) 2026 University of Vienna
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
from constants import au2fs
from qmout import QMout
from SHARC_HYBRID import SHARC_HYBRID
from utils import InDir, expand_path, mkdir, question

__all__ = ["SHARC_NACDT"]

AUTHORS = "Sascha Mausenberger"
VERSION = "4.0"
VERSIONDATE = datetime.datetime(2026, 2, 6)
NAME = "NACDT"
DESCRIPTION = "   HYBRID interface for calculating overlaps from TDC"

CHANGELOGSTRING = """
"""


# Direct translation from sharc matrix.f90:408 dlowdin to Python with ChatGPT
def dtransform(A_ss: np.ndarray, U_ss: np.ndarray, mode: str) -> np.ndarray:
    """
    Returns transformed matrix (does NOT modify inputs).

    mode == 'utau'  →  U^T A U
    mode == 'uaut'  →  U A U^T
    """
    A = np.asarray(A_ss, dtype=np.float64)
    U = np.asarray(U_ss, dtype=np.float64)

    if mode == "utau":
        return U.T @ A @ U
    elif mode == "uaut":
        return U @ A @ U.T
    else:
        raise ValueError("Unknown transformation mode in dtransform")
def ddiagonalize(A_ss: np.ndarray):
    """
    Diagonalizes symmetric matrix A_ss.

    Returns:
        A_diag  → diagonal matrix with eigenvalues (ascending)
        U_ss    → eigenvector matrix such that
                  original A = U diag(w) U^T
    """
    A = np.asarray(A_ss, dtype=np.float64)

    w, U_ss = np.linalg.eigh(A)  # ascending eigenvalues
    A_diag = np.diag(w)

    return A_diag, U_ss
def dnormalize(A_ss: np.ndarray) -> np.ndarray:
    """
    Returns a column-normalized copy of A_ss.
    """
    A = np.asarray(A_ss, dtype=np.float64)

    norms = np.sqrt(np.sum(A * A, axis=0))
    return A / norms
def dlowdin(A_ss: np.ndarray) -> np.ndarray:
    """
    Löwdin symmetric orthogonalization.

    Returns a new orthonormalized matrix.
    Does NOT modify input.
    """
    A = np.asarray(A_ss, dtype=np.float64)

    # 1) S = A^T A
    S_ss = A.T @ A

    # 2) Diagonalize S
    S_diag, U_ss = ddiagonalize(S_ss)

    # 3) Inverse square root of eigenvalues
    diag_vals = np.diag(S_diag)
    inv_sqrt = 1.0 / np.sqrt(diag_vals)
    S_inv_sqrt_diag = np.diag(inv_sqrt)

    # 4) Transform back: U * S^{-1/2} * U^T
    S_inv_sqrt = dtransform(S_inv_sqrt_diag, U_ss, "uaut")

    # 5) A <- A * S^{-1/2}
    A_orth = A @ S_inv_sqrt

    # 6) Normalize columns
    A_orth = dnormalize(A_orth)

    return A_orth

# End of ChatGPT


class SHARC_NACDT(SHARC_HYBRID):
    """
    SHARC interface for Frenkel exciton model
    """

    _version = VERSION
    _versiondate = VERSIONDATE
    _authors = AUTHORS
    _changelogstring = CHANGELOGSTRING
    _name = NAME
    _description = DESCRIPTION

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # Update template keys
        self.QMin.template.update({"interface": None, "dt": 0.5 / au2fs})
        self.QMin.template.types.update({"interface": dict, "dt": float})

        self.template_file = None
        self.resources_file = None

        self._grad_list = None

    @staticmethod
    def description() -> str:
        return SHARC_NACDT._description

    @staticmethod
    def version() -> str:
        return SHARC_NACDT._version

    @staticmethod
    def name() -> str:
        return SHARC_NACDT._name

    @staticmethod
    def versiondate() -> datetime.datetime:
        return SHARC_NACDT._versiondate

    @staticmethod
    def changelogstring() -> str:
        return SHARC_NACDT._changelogstring

    @staticmethod
    def authors() -> str:
        return SHARC_NACDT._authors

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

    def get_infos(self, INFOS: dict, KEYSTROKES: TextIOWrapper | None = None) -> dict:
        self.log.info("=" * 80)
        self.log.info(f"{'||':<78}||")
        self.log.info(f"||{'NACDT interface setup': ^76}||\n{'||':<78}||")
        self.log.info("=" * 80)
        self.log.info("\n")

        self.log.info(f"\n{' Setting up child interface ':=^80s}\n")
        self._kindergarden["child"].QMin.molecule["states"] = INFOS["states"]
        self._kindergarden["child"].get_infos(INFOS, KEYSTROKES=KEYSTROKES)
        return INFOS

    def get_features(self, KEYSTROKES: TextIOWrapper | None = None) -> set[str]:
        if not self._read_template:
            self.template_file = expand_path(
                question(
                    "Please specify the path to your NACDT.template file",
                    str,
                    KEYSTROKES=KEYSTROKES,
                    default="NACDT.template",
                )
            )
            self.read_template(self.template_file)
            frag = self.QMin.template["interface"]
            kindergarden = {"child": (frag["name"], frag["args"], frag["kwargs"])}
            self.instantiate_children(kindergarden)

        features = set({"overlap", "phases"}) | self._kindergarden["child"].get_features()
        if features.isdisjoint({"h", "grad"}):
            self.log.error("Child interface must support at least h, and grad request!")
            raise ValueError
        return features

    def create_restart_files(self) -> None:
        if self.persistent:
            super().write_step_file()
        else:
            super().create_restart_files()
        self._kindergarden["child"].create_restart_files()

    def write_step_file(self):
        if not self.persistent:
            super().write_step_file()
        else:
            self.savedict["last_step"] = self.QMin.save["step"]
        self._kindergarden["child"].write_step_file()

    def set_coords(self, xyz: np.ndarray | list | str, pc: bool = False) -> None:
        super().set_coords(xyz, pc)
        self._kindergarden["child"].set_coords(xyz, pc)

    def read_resources(self, resources_file: str = "NACDT.resources", kw_whitelist: list[str] | None = None) -> None:
        if os.path.isfile(resources_file):
            return super().read_resources(resources_file, kw_whitelist)
        self.log.info("No resource file found. Loading defaults.")
        self._read_resources = True

    def read_template(self, template_file: str = "NACDT.template", kw_whitelist: list[str] | None = None) -> None:
        self.log.debug(f"Parsing template file {template_file}")

        # Open template_file and parse yaml
        with open(template_file, "r", encoding="utf-8") as tmpl_file:
            tmpl_dict = yaml.safe_load(tmpl_file)
            self.log.debug(f"Parsing yaml file:\n{tmpl_dict}")

        if "name" not in tmpl_dict["interface"]:
            self.log.error("No name defined in interace!")
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
        if "dt" in tmpl_dict:
            if not isinstance(tmpl_dict["dt"], float):
                self.log.error("dt must be a float!")
                raise ValueError
            self.QMin.template["dt"] = tmpl_dict["dt"] / au2fs

        self.QMin.template["interface"] = tmpl_dict["interface"]

        self._read_template = True

    def setup_interface(self) -> None:
        super().setup_interface()

        with InDir("QM"):
            child = self.QMin.template["interface"]
            self.instantiate_children({"child": (child["name"], child["args"], child["kwargs"])})

            self._kindergarden["child"].setup_mol(self.QMin)
            self._kindergarden["child"].QMin.save["savedir"] = os.path.join(self.QMin.save["savedir"], "QM")
            self._kindergarden["child"].read_resources()
            self._kindergarden["child"].read_template()
            self._kindergarden["child"].QMin.resources["scratchdir"] = os.path.join(self.QMin.resources["scratchdir"], "QM")
            self._kindergarden["child"].QMin.resources["pwd"] = os.path.join(self.QMin.resources["pwd"], "QM")
            self._kindergarden["child"].QMin.resources["cwd"] = os.path.join(self.QMin.resources["cwd"], "QM")
            self._kindergarden["child"].setup_interface()

        self._grad_list = list(range(1, self.QMin.molecule["nmstates"] + 1))

    def run(self) -> None:
        with InDir("QM"):
            self._kindergarden["child"].run()

    def read_requests(self, requests_file="QM.in"):
        super().read_requests(requests_file)

        requests = dict(filter(lambda item: item[1] is not None, self.QMin.requests.items()))
        requests.update(
            {
                "overlap": False,
                "phases": False,
                "grad": self._grad_list,
                "h": True,
                "step": self.QMin.save["step"],
            }
        )
        self._kindergarden["child"].read_requests(requests)

    def getQMout(self) -> QMout:
        self.QMout = self._kindergarden["child"].getQMout()
        step = self.QMin.save["step"]

        veloc = np.zeros((self.QMin.molecule["natom"], 3))
        if self.QMin.coords["veloc"] is not None:
            veloc = self.QMin.coords["veloc"]

        # TDC calculation by first order gradient difference (see qm.f90 QM_processing)
        nacdt = np.zeros((self.QMin.molecule["nmstates"], self.QMin.molecule["nmstates"]))
        if self.QMin.requests["overlap"] or self.QMin.requests["phases"]:
            # Load prev. step properties
            if not self.persistent:
                data = np.load(os.path.join(self.QMin.save["savedir"], f"arrays.{self.QMin.save['step'] - 1}"))
                grad_old, veloc_old, nacdt_old = data["grad"], data["veloc"], data["nacdt"]
                data.close()
            else:
                grad_old, veloc_old, nacdt_old = self.savedict[step - 1]

            # d(dV)/dt = d(dV)/dR * veloc
            gv_old = np.einsum("sad,ad->s", grad_old, veloc_old)
            gv = np.einsum("sad,ad->s", self.QMout.grad, veloc)

            eh = (gv - gv_old) / self.QMin.template["dt"]

            sum_state = 0
            for states in self.QMin.molecule["states"]:
                first = sum_state
                sum_state += states
                last = sum_state

                for i in range(first, last):
                    for j in range(i + 1, last):
                        denom = self.QMout.h[j, j].real - self.QMout.h[i, i].real
                        if denom == 0.0:
                            fmag = (eh[j] - eh[i]) / 1.0e-8
                        else:
                            fmag = (eh[j] - eh[i]) / denom

                        nacdt[i, j] = 0.5 * np.sqrt(fmag) if fmag > 0.0 else 0.0
                        nacdt[j, i] = -nacdt[i, j]

            # Overlaps from TDC
            overlaps = nacdt_old * self.QMin.template["dt"]
            np.fill_diagonal(overlaps, 0.0)

            for istate in range(self.QMin.molecule["nmstates"]):
                row = overlaps[istate, :]
                overlap_sum = np.sum(row**2)

                if overlap_sum > 1.0:
                    row /= overlap_sum

                overlaps[istate, istate] = np.sqrt(max(0.0, 1.0 - overlap_sum))

            if self.QMin.requests["overlap"]:
                self.QMout.overlap = dlowdin(overlaps)
            if self.QMin.requests["phases"]:
                self.QMout.phases = np.einsum("ii->i", overlaps).copy()
                self.QMout.phases[self.QMout.phases > 0] = 1
                self.QMout.phases[self.QMout.phases < 0] = -1

        # Save properties of current time step
        if not self.persistent:
            with open(os.path.join(self.QMin.save["savedir"], f"arrays.{self.QMin.save['step']}"), "wb") as f:
                np.savez(
                    f,
                    grad=self.QMout.grad,
                    veloc=veloc,
                    nacdt=nacdt,
                )
        else:
            self.savedict[step] = (self.QMout.grad, veloc, nacdt)
        return self.QMout


if __name__ == "__main__":
    SHARC_NACDT().main()
