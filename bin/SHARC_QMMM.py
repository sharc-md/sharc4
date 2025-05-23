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


import copy

# IMPORTS
# external
import datetime
import os
import shutil
from io import TextIOWrapper

import numpy as np

# internal
from constants import NUMBERS

# from factory import factory
from SHARC_HYBRID import SHARC_HYBRID
from SHARC_INTERFACE import SHARC_INTERFACE
from utils import ATOM, InDir, expand_path, itnmstates, mkdir, question, readfile

VERSION = "4.0"
VERSIONDATE = datetime.datetime(2023, 8, 24)

CHANGELOGSTRING = """
"""
np.set_printoptions(linewidth=400)


class SHARC_QMMM(SHARC_HYBRID):
    _version = VERSION
    _versiondate = VERSIONDATE
    _changelogstring = CHANGELOGSTRING
    _step = 0
    _qm_s = 0.3
    _mm_s = 1 - _qm_s

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Add template keys
        self.QMin.template.update(
            {
                "qmmm_table": "QMMM.table",
                "qm-program": None,
                "mm-program": None,
                "embedding": "subtractive",
                "qm-dir": None,
                "mml-dir": "MML",
                "mms-dir": "MMS",
                "mm_dipole": False,
            }
        )
        self.QMin.template.types.update(
            {
                "qmmm_table": str,
                "qm-program": str,
                "mm-program": str,
                "embedding": str,
                "qm-dir": str,
                "mml-dir": str,
                "mms-dir": str,
                "mm_dipole": bool,
            }
        )

        self.qm_interface = None
        self.mml_interface = None
        self.mms_interface = None
        self.atoms = None
        self.qm_ids = None
        self.mm_ids = None
        self._linkatoms = None
        self._num_mm = None
        self._num_qm = None
        self.mm_links = None
        self.non_link_mm = None

    @staticmethod
    def description():
        return "   HYBRID interface for QM/MM (electrostatic embedding, link atom scheme)"

    @staticmethod
    def version():
        return SHARC_QMMM._version

    def get_infos(self, INFOS, KEYSTROKES: TextIOWrapper | None = None) -> dict:
        self.log.info("=" * 80)
        self.log.info(f"{'||':<78}||")
        self.log.info(f"||{'QMMM interface setup':=^76}||\n{'||':<78}||")
        self.log.info("=" * 80)
        self.log.info("\n")

        # self.template_file = question("Specify path to QMMM.template", str, KEYSTROKES=KEYSTROKES, autocomplete=True)

        # self.read_template(self.template_file)
        if question("Do you have an QMMM.resources file?", bool, KEYSTROKES=KEYSTROKES, autocomplete=False, default=False):
            self.resources_file = question(
                "Specify path to QMMM.resources", str, KEYSTROKES=KEYSTROKES, autocomplete=True, default="QMMM.resources"
            )

        self.log.info(f"\n{' Setting up QM-interface ':=^80s}\n")
        self.qm_interface.QMin.molecule["states"] = INFOS["states"]
        self.qm_interface.get_infos(INFOS, KEYSTROKES=KEYSTROKES)
        self.log.info(f"\n{' Setting up MML-interface (whole system) ':=^80s}\n")
        self.mml_interface.QMin.molecule["states"] = [1]
        self.mml_interface.get_infos(INFOS, KEYSTROKES=KEYSTROKES)
        if self.QMin.template["embedding"] == "subtractive":
            self.log.info(f"\n{' Setting up MMS-interface (qm system) ':=^80s}\n")
            self.mms_interface.QMin.molecule["states"] = [1]
            self.mms_interface.get_infos(INFOS, KEYSTROKES=KEYSTROKES)

        return INFOS

    @staticmethod
    def name() -> str:
        return "QMMM"

    def prepare(self, INFOS, dir_path) -> None:
        QMin = self.QMin

        if "link_files" in INFOS:
            os.symlink(expand_path(self.template_file), os.path.join(dir_path, self.name() + ".template"))
            os.symlink(
                expand_path(self.QMin.template["qmmm_table"]),
                os.path.join(dir_path, os.path.split(self.QMin.template["qmmm_table"])[1]),
            )
            if "resources_file" in self.__dict__:
                os.symlink(expand_path(self.resources_file), os.path.join(dir_path, self.name() + ".resources"))
        else:
            shutil.copy(self.template_file, os.path.join(dir_path, self.name() + ".template"))
            shutil.copy(
                expand_path(self.QMin.template["qmmm_table"]),
                os.path.join(dir_path, os.path.split(self.QMin.template["qmmm_table"])[1]),
            )
            if "resources_file" in self.__dict__:
                shutil.copy(self.resources_file, os.path.join(dir_path, self.name() + ".resources"))

        # if not QMin.save["savedir"]:
        #     self.log.warning("savedir not specified in QM.in, setting savedir to current directory!")
        #     QMin.save["savedir"] = os.getcwd()

        qmdir = dir_path + f"/{QMin.template['qm-dir']}"
        mkdir(qmdir)
        mmldir = dir_path + f"/{QMin.template['mml-dir']}"
        mkdir(mmldir)
        mmsdir = dir_path + f"/{QMin.template['mms-dir']}"
        mkdir(mmsdir)

        # folder setup and savedir
        qm_savedir = os.path.join(dir_path, QMin.save["savedir"], "QM_" + QMin.template["qm-program"].upper())
        # self.log.debug(f"qm_savedir {qm_savedir}")
        # if not os.path.isdir(qm_savedir):
        #     mkdir(qm_savedir)
        self.qm_interface.QMin.save["savedir"] = qm_savedir
        self.qm_interface.QMin.resources["scratchdir"] = os.path.join(
            QMin.resources["scratchdir"], "QM_" + QMin.template["qm-program"].upper()
        )
        self.qm_interface.prepare(INFOS, qmdir)

        mml_savedir = os.path.join(dir_path, QMin.save["savedir"], "MML_" + QMin.template["mm-program"].upper())
        # if not os.path.isdir(mml_savedir):
        #     mkdir(mml_savedir)
        self.mml_interface.QMin.save["savedir"] = mml_savedir
        self.mml_interface.QMin.resources = os.path.join(
            QMin.resources["scratchdir"], "MML_" + QMin.template["mm-program"].upper()
        )
        self.mml_interface.prepare(INFOS, mmldir)

        if QMin.template["embedding"] == "subtractive":
            mms_savedir = os.path.join(dir_path, QMin.save["savedir"], "MMS_" + QMin.template["mm-program"].upper())
            # if not os.path.isdir(mms_savedir):
            #     mkdir(mms_savedir)
            self.mms_interface.QMin.save["savedir"] = mms_savedir
            self.mms_interface.QMin.resources["scratchdir"] = os.path.join(
                QMin.resources["scratchdir"], "MMS_" + QMin.template["mm-program"].upper()
            )
            self.mms_interface.prepare(INFOS, mmsdir)

    @staticmethod
    def about():
        pass

    @staticmethod
    def versiondate():
        return SHARC_QMMM._versiondate

    @staticmethod
    def changelogstring():
        return SHARC_QMMM._changelogstring

    @staticmethod
    def authors() -> str:
        return "Severin Polonius"

    # TODO: update for other embeddings
    def get_features(self, KEYSTROKES: TextIOWrapper | None = None) -> set:

        if not self._read_template:
            self.template_file = question(
                "Please specify the path to your QMMM.template file", str, KEYSTROKES=KEYSTROKES, default="QMMM.template"
            )

            self.read_template(self.template_file)

        qm_features = self.qm_interface.get_features(KEYSTROKES=KEYSTROKES)
        mm_features = self.mml_interface.get_features(KEYSTROKES=KEYSTROKES)

        qmmm_features = {feat for feat in qm_features}
        if "point_charges" in qmmm_features:
            qmmm_features.remove("point_charges")
        else:
            self.log.error("Your QM interface needs to be able to include point charges in its calculations!")
            raise RuntimeError

        if "multipolar_fit" not in mm_features:
            self.log.error("Your MM interface needs to be able to provide point charges!")
            raise RuntimeError

        if "grad" in qmmm_features and "grad" not in mm_features:
            qmmm_features.remove("grad")

        if "h" in qmmm_features and "h" not in mm_features:
            qmmm_features.remove("h")
        self.log.debug(qmmm_features)

        return set(qmmm_features)

    def read_template(self, template_file="QMMM.template", kw_whitelist: list[str] | None = None) -> None:
        super().read_template(template_file, kw_whitelist)

        # check
        # allowed_embeddings = ["additive", "subtractive"]
        allowed_embeddings = ["subtractive"]
        if self.QMin.template["embedding"] not in allowed_embeddings:
            self.log.error(
                f"Chosen embedding \"{self.QMin.template['embedding']}\" is not available (available: {', '.join(str(i) for i in all)})"
            )
            raise RuntimeError()

        required: set = {
            "qm-program",
            "mm-program",
        }

        if not required.issubset(self.QMin.template.keys()):
            self.log.error(
                '"{}" not specified in {}'.format(
                    '", "'.join(filter(lambda x: x not in self.QMin.template, required)), template_file
                )
            )
            raise RuntimeError()

        # --- create interfaces ---

        # QM interface
        self.qm_interface: SHARC_INTERFACE = self._load_interface(self.QMin.template["qm-program"])(
            persistent=self.persistent, logname=f"QM {self.QMin.template['qm-program']}", loglevel=self.log.level
        )
        self.qm_interface.QMin.molecule["states"] = copy.copy(self.QMin.molecule["states"])

        # MML interface
        self.mml_interface: SHARC_INTERFACE = self._load_interface(self.QMin.template["mm-program"])(
            persistent=self.persistent, logname=f"MML {self.QMin.template['mm-program']}", loglevel=self.log.level
        )
        self.mml_interface.QMin.molecule["states"] = [1]

        # MMS interface
        if self.QMin.template["embedding"] == "subtractive":
            self.mms_interface: SHARC_INTERFACE = self._load_interface(self.QMin.template["mm-program"])(
                persistent=self.persistent, logname=f"MMS {self.QMin.template['mm-program']}", loglevel=self.log.level
            )
            self.mms_interface.QMin.molecule["states"] = [1]

        # start processing template
        if not self.QMin.template["qm-dir"]:
            self.QMin.template["qm-dir"] = self.qm_interface.name()
            self.log.info(f"'qm-dir not set in template setting to name of program: {self.QMin.template['qm-dir']}")
        qmmm_table = []

        # check is qmmm_table is relative or absolute path
        if not os.path.isabs(self.QMin.template["qmmm_table"]):
            #  path from location of template
            self.QMin.template["qmmm_table"] = os.path.join(os.path.dirname(template_file), self.QMin.template["qmmm_table"])
        if not os.path.isfile(self.QMin.template["qmmm_table"]):
            self.log.error(f"{self.QMin.template['qmmm_table']} not found! Specify 'qmmm_table' in template!")
            raise RuntimeError()
        for line in readfile(self.QMin.template["qmmm_table"]):
            qmmm_table.append(line.split())

        self.atoms = [
            ATOM(i, v[0].lower() == "qm", v[1], [0.0, 0.0, 0.0], set(map(lambda x: int(x) - 1, v[2:])))
            for i, v in enumerate(qmmm_table)
        ]

        # sanitize mmatoms
        # set links
        self.qm_ids = []
        self.mm_ids = []
        self._linkatoms = set()  # set to hold tuple
        for i in self.atoms:
            for jd in i.bonds:
                # jd = j - 1
                if i.id == jd:
                    self.log.error(f"Atom bound to itself:\n{i}")
                    raise RuntimeError()
                j = self.atoms[jd]
                if i.id not in j.bonds:
                    j.bonds.add(i.id)
                if i.qm != j.qm:
                    self._linkatoms.add((i.id, j.id) if i.qm else (j.id, i.id))
            # sort out qm atoms
            self.qm_ids.append(i.id) if i.qm else self.mm_ids.append(i.id)
        self._num_qm = len(self.qm_ids)
        self._num_mm = len(self.mm_ids)
        self.mm_links = set(mm for _, mm in self._linkatoms)  # set of all mm_ids in link bonds (deleted in point charges!)

        # check of linkatoms: map linkatoms to sets of unique qm and mm ids: decreased number -> Error
        if len(self._linkatoms) > len(set(map(lambda x: x[0], self._linkatoms))):
            self.log.error("Some QM atom is involved in more than one link bond!")
            raise RuntimeError()
        if len(self._linkatoms) > len(set(map(lambda x: x[1], self._linkatoms))):
            self.log.error("Some MM atom is involved in more than one link bond!")
            raise RuntimeError()
        self._linkatoms = list(self._linkatoms)
        self._read_template = True

    def read_resources(self, resources_filename="QMMM.resources"):
        if not os.path.isfile(resources_filename):
            self.log.warning(f"File '{resources_filename}' not found! Continuuing without appling any settings")
            self._read_resources = True
            return
        super().read_resources(resources_filename)
        self._read_resources = True

    def setup_interface(self):
        # obtain the statemap
        self.QMin.maps["statemap"] = {i + 1: [*v] for i, v in enumerate(itnmstates(self.QMin.molecule["states"]))}
        # prepare info for both interfaces
        el = self.QMin.molecule["elements"]
        # TODO: check whether self.atoms[i].symbol and el are the same
        n_link = len(self._linkatoms)
        qm_el = [self.atoms[i].symbol for i in self.qm_ids] + ["H"] * n_link

        # TODO: Would be better to call setup_mol() in the following, but that is a bit difficult for this interface

        # ----- QM interface -----
        # setup mol for qm
        qm_savedir = os.path.join(self.QMin.save["savedir"], "QM_" + self.QMin.template["qm-program"].upper())
        if not os.path.isdir(qm_savedir):
            mkdir(qm_savedir)
        self.qm_interface.setup_mol(
            {
                "states": self.QMin.molecule["states"],
                "charge": self.QMin.molecule["charge"],
                "NAtoms": self._num_qm + n_link,
                "IAn": [NUMBERS[a] for a in qm_el],
                "retain": f"retain {self.QMin.requests['retain']}",
                "savedir": qm_savedir,
            }
        )
        self.qm_interface.QMin.molecule["npc"] = self._num_mm
        self.qm_interface.QMin.molecule["point_charges"] = True

        # read template and resources
        with InDir(self.QMin.template["qm-dir"]) as _:
            self.qm_interface.read_resources()
            self.qm_interface.read_template()
            self.qm_interface.setup_interface()

        # ----- MML interface -----
        # setup mol for mml
        mml_savedir = os.path.join(self.QMin.save["savedir"], "MML_" + self.QMin.template["mm-program"].upper())
        if not os.path.isdir(mml_savedir):
            mkdir(mml_savedir)
        self.mml_interface.setup_mol(
            {
                "states": [1],
                "charge": [0],
                "NAtoms": self.QMin.molecule["natom"],
                "IAn": [NUMBERS[a] for a in el],
                "retain": f"retain {self.QMin.requests['retain']}",
                "savedir": mml_savedir,
            }
        )
        with InDir(self.QMin.template["mml-dir"]) as _:
            self.mml_interface.read_resources()
            self.mml_interface.read_template()
            self.mml_interface.setup_interface()

        # ----- MMS interface -----
        # switch for subtractive
        if self.QMin.template["embedding"] == "subtractive":
            mms_savedir = os.path.join(self.QMin.save["savedir"], "MMS_" + self.QMin.template["mm-program"].upper())
            if not os.path.isdir(mms_savedir):
                mkdir(mms_savedir)
            # setup mol for mms
            mms_el = [self.atoms[i].symbol for i in self.qm_ids]
            mms_el += [
                self.atoms[x[1]].symbol for x in self._linkatoms
            ]  # add symbols of link atoms (original element -> proper bonded terms in MM calc)

            self.mml_interface.setup_mol(
                {
                    "states": [1],
                    "charge": [0],
                    "NAtoms": self._num_qm + n_link,
                    "IAn": [NUMBERS[a] for a in mms_el],
                    "retain": f"retain {self.QMin.requests['retain']}",
                    "savedir": mml_savedir,
                }
            )

            # read template and resources
            with InDir(self.QMin.template["mms-dir"]) as _:
                self.mms_interface.read_resources()
                self.mms_interface.read_template()
                self.mms_interface.setup_interface()

        return

    def run(self):
        qm_coords = np.array([self.QMin.coords["coords"][i].copy() for i in self.qm_ids])
        if len(self._linkatoms) > 0:
            # get linkatom coords
            def get_link_coord(link: tuple) -> np.ndarray[float]:
                qm_id, mm_id = link
                return self.QMin.coords["coords"][qm_id] * self._qm_s + self.QMin.coords["coords"][mm_id] * self._mm_s

            link_coords = np.array([get_link_coord(link) for link in self._linkatoms])
            self.qm_interface.QMin.coords["coords"] = np.vstack((qm_coords, link_coords))
        else:
            self.qm_interface.QMin.coords["coords"] = qm_coords
        if self.QMin.template["embedding"] == "subtractive":
            if len(self._linkatoms) > 0:
                mmlink_indices = [i[1] for i in self._linkatoms]
            else:
                mmlink_indices = []
            self.qm_and_mmlink_indices = sorted(self.qm_ids + mmlink_indices)
            mms_coords = np.array([self.QMin.coords["coords"][i].copy() for i in self.qm_and_mmlink_indices])

        self.mml_interface.QMin.coords["coords"] = self.QMin.coords["coords"].copy()
        # setting requests for qm and mm regions based on the QMMM requests

        all_requests = {k: v for (k, v) in self.QMin.requests.items() if v is not None}

        for key, value in all_requests.items():
            match key:
                # if both interfaces have to compute a property
                case "h" | "dm" | "multipolar_fit":
                    self.qm_interface.QMin.requests[key] = value
                    self.mml_interface.QMin.requests[key] = value
                    if self.QMin.template["embedding"] == "subtractive":
                        self.mms_interface.QMin.requests[key] = value
                # grad is special since we should always only have one state for the MM regions
                case "grad":
                    self.qm_interface.QMin.requests[key] = value
                    self.mml_interface.QMin.requests[key] = value
                    if self.QMin.template["embedding"] == "subtractive":
                        self.mms_interface.QMin.requests[key] = [1]
                # for properties, which should only be computed with QM
                case _:
                    self.qm_interface.QMin.requests[key] = value
        self.qm_interface.QMin.save["step"] = self.QMin.save["step"]
        self.qm_interface._step_logic()
        self.qm_interface._request_logic()
        self.mml_interface.QMin.save["step"] = self.QMin.save["step"]
        self.mml_interface._step_logic()
        self.mml_interface._request_logic()
        if self.QMin.template["embedding"] == "subtractive":
            self.mms_interface.QMin.save["step"] = self.QMin.save["step"]
            self.mms_interface._step_logic()
            self.mms_interface._request_logic()

        # always set this request as these charges are required for the calculation of the point charges
        self.mml_interface.QMin.requests["multipolar_fit"] = [1]

        # calc mm
        with InDir(self.QMin.template["mml-dir"]) as _:
            self.mml_interface.run()
            self.mml_interface.getQMout()
            self.mml_interface.write_step_file()
            # is analogous to the density fit from QM interfaces -> generated upon same request
            raw_pc = self.mml_interface.QMout["multipolar_fit"][(self.mml_interface.states[0], self.mml_interface.states[0])][
                :, 0
            ]

        if self.QMin.template["embedding"] == "subtractive":
            self.mms_interface.QMin.coords["coords"] = mms_coords
            with InDir(self.QMin.template["mms-dir"]) as _:
                self.mms_interface.run()
                self.mms_interface.getQMout()
                self.mms_interface.write_step_file()

        # redistribution of mm pc of link atom (charge is not the same in qm calc but pc would be too close)
        for _, mmid in self._linkatoms:
            atom: ATOM = self.atoms[mmid]
            # -> redistribute charge to neighboring atoms (look in old ORCA line 1300)
            neighbor_ids = [x.id for x in map(lambda y: self.atoms[y], atom.bonds) if not x.qm]
            chrg = raw_pc[mmid] / len(neighbor_ids)
            for nb in neighbor_ids:
                raw_pc[nb] += chrg
            raw_pc[mmid] = 0.0
        # self.non_link_mm = [i for i in self.mm_ids if i not in self.mm_links]  # shallow copy
        self.non_link_mm = [i for i in self.mm_ids]  # shallow copy
        # calc qm
        # pc: list[list[float]] = each pc is x, y, z, qpc[p[mmid][1]][3] = 0.  # set the charge of the mm atom to zero
        self.qm_interface.QMin.coords["pccoords"] = self.QMin.coords["coords"][self.non_link_mm, :]
        if "SCALE_POINT_CHARGES" in os.environ:
            raw_pc = raw_pc * float(os.environ["SCALE_POINT_CHARGES"])
        self.qm_interface.QMin.coords["pccharge"] = raw_pc[self.non_link_mm]

        with InDir(self.QMin.template["qm-dir"]) as _:
            self.qm_interface.run()
            self.qm_interface.getQMout()
            self.qm_interface.write_step_file()

    def getQMout(self):
        qmQMout = self.qm_interface.QMout
        self.QMout.states = self.QMin.molecule["states"]
        self.QMout.charges = self.QMin.molecule["charge"][:]
        self.QMout.nstates = self.QMin.molecule["nstates"]
        self.QMout.nmstates = self.QMin.molecule["nmstates"]
        self.QMout.natom = self.QMin.molecule["natom"]
        self.QMout.npc = self.QMin.molecule["npc"]
        self.QMout.point_charges = False

        # for debugging contributions
        set_qm_to_zero = False
        set_mms_to_zero = False
        set_mml_to_zero = False

        mm_e = float(self.mml_interface.QMout["h"][0][0])
        if set_mml_to_zero:
            mm = e = 0.0
        if self.QMin.template["embedding"] == "subtractive":
            mms_e = float(self.mms_interface.QMout["h"][0][0])
            if set_mms_to_zero:
                mms_e = 0.0
            mm_e -= mms_e

        self.QMout["prop0d"].append(("MM Energy", mm_e))
        # Hamiltonian
        if self.qm_interface.QMin.requests["h"]:
            self.QMout.h = qmQMout.h.copy()
            if set_qm_to_zero:
                self.QMout.h = np.zeros_like(self.QMout.h)
            self.QMout.h += np.eye(self.QMout.h.shape[0], dtype=float) * mm_e

        # gen output
        if self.QMin.requests["grad"]:
            qm_grad = qmQMout.grad
            if set_qm_to_zero:
                qm_grad = np.zeros_like(qm_grad)
            mm_grad = self.mml_interface.QMout.grad[0]
            if set_mml_to_zero:
                mm_grad = np.zeros_like(mm_grad)

            if self.QMin.template["embedding"] == "subtractive":
                mms_grad = self.mms_interface.QMout["grad"][0]
                if set_mms_to_zero:
                    mms_grad = np.zeros_like(mms_grad)
                mm_grad[self.qm_and_mmlink_indices, :] -= mms_grad[:, :]

            # init gradient as mm_gradient
            grad = np.full((self.QMin.molecule["nmstates"], self.QMin.molecule["natom"], 3), mm_grad[None, ...])
            grad[:, self.qm_ids, :] += qm_grad[:, : self._num_qm, :]
            # linkatoms come after qm atoms
            for n, _ in enumerate(self._linkatoms):
                qm_id, mm_id = self._linkatoms[n]
                grad[:, mm_id, :] += self._mm_s * qm_grad[:, n + self._num_qm, :]
                grad[:, qm_id, :] += self._qm_s * qm_grad[:, n + self._num_qm, :]

            if "grad_pc" in qmQMout:  # apply pc grad
                if set_qm_to_zero:
                    qmQMout.grad_pc = np.zeros_like(qmQMout.grad_pc)
                grad[:, self.non_link_mm, :] += qmQMout.grad_pc
            else:
                self.log.warning("No 'grad_pc' in QMout of QM interface!")

            self.QMout.grad = grad

        if self.QMin.requests["nacdr"]:
            nacdr = np.zeros(
                (self.QMin.molecule["nmstates"], self.QMin.molecule["nmstates"], self.QMin.molecule["natom"], 3), dtype=float
            )
            nacdr[:, :, self.qm_ids, :] += self.qm_interface.QMout.nacdr

            for n, _ in enumerate(self._linkatoms):  # linkatoms come after qm atoms
                qm_id, mm_id = self._linkatoms[n]
                nacdr[:, :, mm_id, :] += self._mm_s * self.qm_interface.QMout.nacdr[:, :, n + self._num_qm, :]
                nacdr[:, :, qm_id, :] += self._qm_s * self.qm_interface.QMout.nacdr[:, :, n + self._num_qm, :]

            if "nacdr_pc" in qmQMout:
                nacdr[:, :, self.non_link_mm, :] += qmQMout.nacdr_pc
            else:
                self.log.warning("No 'nacdr_pc' in QMout of QM interface!")

            self.QMout.nacdr = nacdr

        if self.QMin.requests["dm"]:
            self.QMout.dm = self.qm_interface.QMout.dm.copy()
            if self.QMin.template["mm_dipole"]:
                np.einsum("xii->xi", self.QMout.dm)[...] += self.mml_interface.QMout.dm[:, 0, 0]

        if self.QMin.requests["overlap"]:
            self.QMout.overlap = self.qm_interface.QMout.overlap

        # potentially put out other contributions and properties...
        for i in ["ion", "prop", "theodore"]:
            if i in qmQMout:
                self.QMout[i] = qmQMout[i]
        self.QMout.runtime = self.clock.measuretime()
        return self.QMout

    def create_restart_files(self):
        self.qm_interface.create_restart_files()
        self.mml_interface.create_restart_files()
        if self.QMin.template["embedding"] == "subtractive":
            self.mms_interface.create_restart_files()

    def clean_savedir(self):
        super().clean_savedir()
        self.qm_interface.clean_savedir()
        self.mml_interface.clean_savedir()
        if self.QMin.template["embedding"] == "subtractive":
            self.mms_interface.clean_savedir()

    # def _step_logic(self):
    #     super()._step_logic()
    #     self.qm_interface._step_logic()
    #     self.mml_interface._step_logic()
    #     if self.QMin.template["embedding"] == "subtractive":
    #         self.mms_interface._step_logic()

    # def write_step_file(self):
    #     super().write_step_file()
    #     self.qm_interface.write_step_file()
    #     self.mml_interface.write_step_file()
    #     if self.QMin.template["embedding"] == "subtractive":
    #         self.mms_interface.write_step_file()

    # def update_step(self, step: int = None):
    #     super().update_step(step)
    #     self.qm_interface.update_step(step)
    #     self.mml_interface.update_step(step)
    #     if self.QMin.template["embedding"] == "subtractive":
    #         self.mms_interface.update_step(step)


if __name__ == "__main__":
    from logger import loglevel

    try:
        qmmm = SHARC_QMMM(loglevel=loglevel)
        qmmm.main()
    except KeyboardInterrupt:
        print("\nCTRL+C makes me a sad SHARC ;-(")
