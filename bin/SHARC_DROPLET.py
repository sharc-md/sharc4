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


# IMPORTS
# external
import datetime
import math
import os
import shutil
from io import TextIOWrapper

import numpy as np
import yaml
# internal
from constants import au2a, au2newton, n_avogadro
from SHARC_HYBRID import SHARC_HYBRID
from SHARC_INTERFACE import SHARC_INTERFACE
from utils import InDir, expand_path, mkdir, question

VERSION = "4.0"
VERSIONDATE = datetime.datetime(2025, 4, 1)

CHANGELOGSTRING = """
"""


class SHARC_DROPLET(SHARC_HYBRID):
    _version = VERSION
    _versiondate = VERSIONDATE
    _changelogstring = CHANGELOGSTRING
    _step = 0

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Define attributes here
        self.children_instantiated = False

        # Define template keys and types here
        self.QMin.template.update({"child": None, "droplet_potentials": None})
        self.QMin.template.types.update({"child": dict, "droplet_potentials": list})

        # Simple template for child interfaces
        # See read_template
        self._child_template = {
            "interface": str,  # Name of the SHARC interface
            "dir": str,  # folder to run the child
            "args": list,  # Arguments for child interface
            "kwargs": dict,  # Keyword arguments for child interface
        }

        self.child_interface = None

    @staticmethod
    def description():
        return "   HYBRID interface for adding positional harmonic restraints (droplets, tethers, anchors)"

    @staticmethod
    def version():
        return SHARC_DROPLET._version

    @staticmethod
    def name() -> str:
        return "DROPLET"

    @staticmethod
    def about():
        pass

    @staticmethod
    def versiondate():
        return SHARC_DROPLET._versiondate

    @staticmethod
    def changelogstring():
        return SHARC_DROPLET._changelogstring

    @staticmethod
    def authors() -> str:
        return "Sebastian Mai"

    def get_features(self, KEYSTROKES: TextIOWrapper | None = None) -> set:

        if not self._read_template:
            while True:
                self.child_interfacename = question(
                    "Please specify the name of the child interface", str, KEYSTROKES=KEYSTROKES, default=None, autocomplete=False
                )
                try:
                    # make the child _load_interface
                    self.child_interface: SHARC_INTERFACE = self._load_interface(self.child_interfacename.upper())(
                        persistent=self.persistent, logname=f"QM {self.child_interfacename}", loglevel=self.log.level
                    )
                    self.child_interface.QMin.molecule["states"] = self.QMin.molecule["states"]
                    break
                except:
                    self.log.info("This did not work.")

        child_features = self.child_interface.get_features(KEYSTROKES=KEYSTROKES)
        self.log.debug(child_features)
        # we do not currently support point charges as feature, because want to avoid
        # the ambiguity of having to assign gradients to the charges
        # This means that DROPLET always comes above QMMM
        return set(child_features) - set(["point_charges"])

    def get_infos(self, INFOS, KEYSTROKES: TextIOWrapper | None = None) -> dict:
        self.log.info("=" * 80)
        self.log.info(f"{'||':<78}||")
        self.log.info(f"||{'DROPLET interface setup': ^76}||\n{'||':<78}||")
        self.log.info("=" * 80)
        self.log.info("\n")

        if question("Do you have an DROPLET.resources file?", bool, KEYSTROKES=KEYSTROKES, autocomplete=False, default=False):
            self.resources_file = question(
                "Specify path to DROPLET.resources", str, KEYSTROKES=KEYSTROKES, autocomplete=True, default="DROPLET.resources"
            )

        # --- Setup droplet potentials ---
        while True:
            n_droplets = question(
                "How many droplet/tether potentials do you want to define?", int, KEYSTROKES=KEYSTROKES, default=[1]
            )[0]
            if 0 <= n_droplets:
                break
            else:
                self.log.info("Must be >=0!")
        self.setupINFOS["droplet_potentials"] = []

        for i in range(n_droplets):
            self.log.info(f"\n--- Defining droplet potential #{i+1} ---\n")
            name = f"droplet{i+1}"

            if question(
                "Do you want to calculate the parameters from density/system size?", bool, KEYSTROKES=KEYSTROKES, default=False
            ):
                density = question(
                    "Specify the density of your solvent [g/mL] (default: water at 298K)",
                    float,
                    KEYSTROKES=KEYSTROKES,
                    default=[0.9974],
                )[0]
                press_pascal = (
                    question(
                        "Specify the desired pressure at the surface of the droplet in bar",
                        float,
                        KEYSTROKES=KEYSTROKES,
                        default=[1],
                    )[0]
                    * 100_000
                )
                wokness = question(
                    "On a scale from 1 (harmonic) to 0 (hard wall), how fast should the potential increase?",
                    float,
                    KEYSTROKES=KEYSTROKES,
                    default=[0.2],
                )[0]
                molar_mass = question(
                    "Specify the molar mass of your solvent [g/mol] (default: water)",
                    float,
                    KEYSTROKES=KEYSTROKES,
                    default=[18.01528],
                )[0]
                n_mol = question("How many molecules are in your simulation?", int, KEYSTROKES=KEYSTROKES)[0]

                r_drop = (3 * (n_mol * (1 / (n_avogadro * (1000 * density / molar_mass) * 1e-27))) / (4 * math.pi)) ** (1 / 3)
                r_off = r_drop * (1 - wokness)
                Rcut = r_off
                k_force = (press_pascal * 1e-20 * 4 * math.pi * r_drop**2) / (r_drop - r_off) / (au2newton / au2a)

                self.log.info(f"→ droplet_radius = {Rcut:.3f} Å; force constant = {k_force:.6e} Hartree/Bohr²")
            else:
                Rcut = question("Specify the cutoff radius Rcut (Å)", float, KEYSTROKES=KEYSTROKES)[0]
                k_force = question("Specify the force constant k (Hartree/Bohr²)", float, KEYSTROKES=KEYSTROKES)[0]

            while True:
                origin = question("Specify the origin in Å", float, KEYSTROKES=KEYSTROKES, default=[0.0, 0.0, 0.0])
                if len(origin) == 3:
                    break
                else:
                    self.log.info("Requires three floats!")

            # atoms
            if question("Should all atoms be affected?", bool, KEYSTROKES=KEYSTROKES, default=True):
                atoms = "all"
            else:
                atoms = question("Specify atoms (list of indices starting at 1)", int, KEYSTROKES=KEYSTROKES, ranges=True)

            self.setupINFOS["droplet_potentials"].append(
                {
                    "name": name,
                    "Rcut": Rcut,
                    "k": k_force,
                    "origin": origin,
                    "atoms": atoms,
                }
            )

        # --- Setup child interface ---
        self.log.info(f"\n{' Setting up child interface ':=^80s}\n")
        self.child_interface.QMin.molecule["states"] = INFOS["states"]
        self.child_interface.get_infos(INFOS, KEYSTROKES=KEYSTROKES)

        return INFOS

    def prepare(self, INFOS, dir_path) -> None:

        QMin = self.QMin

        # --- Create child definition ---
        child_block = {
            "name": self.child_interfacename,
            "interface": self.child_interfacename,
            "dir": self.child_interfacename,
            "args": [],
            "kwargs": {},
        }

        # --- Build template dict ---
        template_data = {
            "child": child_block,
            "droplet_potentials": self.setupINFOS.get("droplet_potentials", []),
        }

        # --- Write YAML template ---
        template_file = os.path.join(dir_path, self.name() + ".template")
        with open(template_file, "w") as f:
            yaml.dump(template_data, f, sort_keys=False)

        # --- Handle resources ---
        if "resources_file" in self.__dict__:
            if "link_files" in INFOS:
                os.symlink(expand_path(self.resources_file), os.path.join(dir_path, self.name() + ".resources"))
            else:
                shutil.copy(self.resources_file, os.path.join(dir_path, self.name() + ".resources"))

        if not QMin.save["savedir"]:
            self.log.warning("savedir not specified, setting savedir to current directory!")
            QMin.save["savedir"] = os.getcwd()

        # --- Child directory setup ---
        qmdir = dir_path + f"/{self.child_interfacename.upper()}"
        mkdir(qmdir)

        # folder setup and savedir
        qm_savedir = os.path.join(dir_path, QMin.save["savedir"], "QM_" + self.child_interfacename.upper())
        self.log.debug(f"qm_savedir {qm_savedir}")
        self.child_interface.QMin.save["savedir"] = qm_savedir
        self.child_interface.QMin.resources["scratchdir"] = os.path.join(
            QMin.resources["scratchdir"], "QM_" + self.child_interfacename.upper()
        )
        self.child_interface.prepare(INFOS, qmdir)

    def read_template(self, template_file="DROPLET.template", kw_whitelist=None):
        # It is recommended to use yaml format for hybrid templates.
        # Especially for multi child hybrids it simplifies template parsing
        with open(template_file, "r", encoding="utf-8") as tmpl_file:
            tmpl_dict = yaml.safe_load(tmpl_file)

            if "child" in tmpl_dict:
                child_config = tmpl_dict["child"]
                if "name" not in child_config:
                    child_config["name"] = "child1"
                child_config.setdefault("args", [])  # add empty if not there
                child_config.setdefault("kwargs", {})  # add empty if not there
                for k, expected_type in self._child_template.items():
                    if k not in child_config or not isinstance(child_config[k], expected_type):
                        name = child_config["name"]
                        self.log.error(
                            f"Child '{name}' is missing key '{k}' " f"or value is not of type {expected_type.__name__}"
                        )
                        raise ValueError
            else:
                self.log.error("No children defined in template.")
                raise ValueError

        # When the checks passed, the yaml dictionary can be asigned to QMin
        self.QMin.template["child"] = tmpl_dict["child"]
        self.child_interfacename = next(iter(self.QMin.template["child"]["name"]))

        # make the child _load_interface
        self.child_interface: SHARC_INTERFACE = self._load_interface(self.QMin.template["child"]["interface"])(
            *child_config["args"], **child_config["kwargs"]
        )

        # check directory
        if not self.QMin.template["child"]["dir"]:
            self.QMin.template["child"]["dir"] = self.child_interface.name()
            self.log.info(f"'child-dir not set in template setting to name of program: {self.QMin.template['child']}")

        # go through the droplet potentials and parse them
        # - replace "all" by list of all atoms
        # - handle center of mass origin
        # also check if all entries are there
        processed = []
        for droplet in tmpl_dict["droplet_potentials"]:
            name = droplet.get("name", "<unnamed>")
            # --- Rcut ---
            Rcut = droplet.get("Rcut")/au2a
            if not isinstance(Rcut, (int, float)) or Rcut < 0:
                raise ValueError(f"Droplet potential '{name}': Rcut must be a nonnegative float")
            # --- k ---
            k = droplet.get("k")
            if not isinstance(k, (int, float)):
                raise ValueError(f"Droplet potential '{name}': k must be a float")
            if k < 0:
                self.log.warning(f"Droplet potential '{name}': k is negative!")

            # --- atoms ---
            natom = self.QMin.molecule["natom"]
            atoms = droplet.get("atoms")
            if atoms is None:
                atoms = "all"
            if isinstance(atoms, str):
                if atoms.lower() == "all":
                    atoms = list(range(1, natom + 1))
                else:
                    raise ValueError(f"Droplet '{name}': atoms string must be 'all', got '{atoms}'")
            elif isinstance(atoms, list) and all(isinstance(x, int) for x in atoms):
                if not all(1 <= x <= natom for x in atoms):
                    raise ValueError(f"Droplet '{name}': atoms list must be integers between 1 and {natom}")
            else:
                raise ValueError(f"Droplet '{name}': atoms must be 'all' or a list of integers")
            atoms = [a - 1 for a in atoms]

            # --- origin ---
            origin = droplet.get("origin")
            if origin is None:
                origin = [0.0, 0.0, 0.0]
            if isinstance(origin, str):
                if origin.lower() == "origin":
                    origin = [0.0, 0.0, 0.0]
                elif origin.lower() == "com":
                    origin = "com"
                else:
                    raise ValueError(f"Droplet '{name}': origin string must be 'com', got '{origin}'")
            elif isinstance(origin, list) and len(origin) == 3 and all(isinstance(x, (int, float)) for x in origin):
                origin = [float(x)/au2a for x in origin]
            else:
                raise ValueError(f"Droplet '{name}': origin must be a string or list of 3 floats")

            # --- done for this potential ---
            processed.append(
                {
                    "name": name,
                    "Rcut": float(Rcut),
                    "k": float(k),
                    "origin": origin,
                    "atoms": atoms,
                }
            )
        self.QMin.template["droplet_potentials"] = processed

        # Indicate that read_template was called. This has to be done if
        self._read_template = True

    def read_resources(self, resources_filename="DROPLET.resources"):
        if not os.path.isfile(resources_filename):
            self.log.warning(f"File '{resources_filename}' not found! Continuuing without appling any settings")
            self._read_resources = True
            return
        super().read_resources(resources_filename)

    def _check_charge(self):
        """
        Do not check charge for total system
        """

    def setup_interface(self):
        # prepare info for child interface
        # setup mol for qm
        self.child_interface.setup_mol(self.QMin)

        qm_savedir = os.path.join(self.QMin.save["savedir"], "QM_" + self.QMin.template["child"]["name"].upper())
        if not os.path.isdir(qm_savedir):
            mkdir(qm_savedir)
        # read template and resources
        self.log.debug(self.QMin.template["child"]["dir"])
        with InDir(self.QMin.template["child"]["dir"]) as _:
            self.child_interface.read_resources()
            self.child_interface.QMin.save["savedir"] = qm_savedir  # overwrite savedir
            self.child_interface.read_template()
            self.child_interface.setup_interface()

    def run(self):

        # set origins for "com" option. This will only happen the first time run() is called
        coords = self.QMin.coords["coords"]
        for potential in self.QMin.template["droplet_potentials"]:
            if isinstance(potential["origin"], str) and potential["origin"].lower() == "com":
                com = np.mean(coords[potential["atoms"], :], axis=0)
                potential["origin"] = com

        # run child
        with InDir(self.QMin.template["child"]["dir"]):
            self.child_interface.run()

    def getQMout(self):
        self.QMout = self.child_interface.getQMout()

        # compute restraint energy and gradient
        E = []
        if self.QMin.requests["grad"]:
            Grad = []
        coords = self.QMin.coords["coords"]

        # iterate over restraints
        for potential in self.QMin.template["droplet_potentials"]:
            Rcut = potential["Rcut"]
            kcon = potential["k"]
            origin = np.asarray(potential["origin"], dtype=float)
            atoms = potential["atoms"]
            e = 0.0
            if self.QMin.requests["grad"]:
                grad = np.zeros_like(coords)

            for iatom in atoms:
                xyz = coords[iatom]
                Ri = np.sqrt(np.sum((xyz - origin) ** 2))
                if Ri >= Rcut:
                    e += kcon / 2 * (Ri - Rcut) ** 2
                    if self.QMin.requests["grad"]:
                        grad[iatom, :] += kcon * (Ri - Rcut) * (xyz - origin) / Ri

            # save the computed results
            E.append(e)
            if self.QMin.requests["grad"]:
                Grad.append(grad)

        # compound all restraints
        Etot = sum(E)
        if self.QMin.requests["grad"]:
            Gradtot = np.zeros_like(self.QMout["grad"][0])
            for grad in Grad:
                Gradtot += grad

        # apply to QMout
        if self.QMin.requests["h"]:
            self.QMout.h = self.child_interface.QMout.h.copy()
            self.QMout.h += np.eye(self.QMout.h.shape[0], dtype=float) * Etot

        if self.QMin.requests["grad"]:
            self.QMout.grad = self.child_interface.QMout.grad.copy()
            for i, _ in enumerate(self.QMout.grad):
                self.QMout.grad[i] += Gradtot
                # TODO: set gradients to zero that were not originally requested, but keep M_S sublevels

        self.QMout.runtime = self.clock.measuretime(self.log.debug)
        return self.QMout

    def write_step_file(self):
        super().write_step_file()
        self.child_interface.write_step_file()

    def read_requests(self, requests_file="QM.in"):
        super().read_requests(requests_file)
        self.child_interface.read_requests(requests_file)

    def create_restart_files(self):
        self.child_interface.create_restart_files()

    def clean_savedir(self):
        super().clean_savedir()
        self.child_interface.clean_savedir()

    def set_coords(self, xyz, pc=False):
        super().set_coords(xyz, pc)
        self.child_interface.set_coords(xyz, pc)


if __name__ == "__main__":
    from logger import loglevel

    try:
        DROPLET = SHARC_DROPLET(loglevel=loglevel)
        DROPLET.main()
    except KeyboardInterrupt:
        print("\nCTRL+C makes me a sad SHARC ;-(")
