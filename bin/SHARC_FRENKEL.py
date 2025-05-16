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

import numpy as np
import yaml
from constants import NUMBERS
from SHARC_HYBRID import SHARC_HYBRID
from utils import InDir, expand_path

__all__ = ["SHARC_FRENKEL"]

AUTHORS = "Sascha Mausenberger"
VERSION = "4.0"
VERSIONDATE = datetime.datetime(2025, 5, 7)
NAME = "FRENKEL"
DESCRIPTION = "   HYBRID interface for Frenkel exciton model"

CHANGELOGSTRING = """
"""

all_features = set(["h", "grad", "point_charges"])


class SHARC_FRENKEL(SHARC_HYBRID):
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
        self.QMin.template.update({"fragments": None, "embedding": None})
        self.QMin.template.types.update({"fragments": dict, "embedding": dict})

        # Template interface structure
        self._interface_templ = {
            "interface": str,  # Name of SHARC interface
            "args": list,  # Init arguments for child
            "kwargs": dict,  # Keyword args for child
            "atoms": str,  # List/Range of atoms
            "states": list,  # List of states
            "charges": list,  # List of charges
        }

        # Interface for electrostatic embedding
        self._embedding_interface = None

        # Keep track of total site states to preallocate Hamiltonian
        self._total_site_states = 1  # GS prod

    @staticmethod
    def description():
        return SHARC_FRENKEL._description

    @staticmethod
    def version():
        return SHARC_FRENKEL._version

    @staticmethod
    def name() -> str:
        return SHARC_FRENKEL._name

    @staticmethod
    def versiondate():
        return SHARC_FRENKEL._versiondate

    @staticmethod
    def changelogstring():
        return SHARC_FRENKEL._changelogstring

    @staticmethod
    def authors() -> str:
        return SHARC_FRENKEL._authors

    def get_features(self, KEYSTROKES=None):
        return all_features

    def read_template(self, template_file="FRENKEL.template", kw_whitelist=None):
        self.log.debug(f"Parsing template file {template_file}")

        # Open template_file and parse yaml
        with open(template_file, "r", encoding="utf-8") as tmpl_file:
            tmpl_dict = yaml.safe_load(tmpl_file)
            self.log.debug(f"Parsing yaml file:\n{tmpl_dict}")

        assert len(tmpl_dict["fragments"]) > 1, "At least two fragments have to be defined!"

        for name, frag in tmpl_dict["fragments"].items():
            # Set default value for args, kwargs and charges
            if "args" not in frag:
                tmpl_dict["fragments"][name]["args"] = []
            if "kwargs" not in frag:
                tmpl_dict["fragments"][name]["kwargs"] = {}
            if "charges" not in frag:
                self.log.info(f"No charge defined for fragment {name}, set default 0.")
                tmpl_dict["fragments"][name]["charges"] = [0]

            # Check if all parameters are present and of correct type
            for k, v in self._interface_templ.items():
                if k not in frag:
                    self.log.error(f"{k} has to be defined in fragment {name}")
                    raise ValueError
                if not isinstance(frag[k], v):
                    self.log.error(f"Value of key {k} in fragment {name} must be of type {v}")
                    raise ValueError

            # Convert atoms string to list
            tmpl_dict["fragments"][name]["atoms"] = sorted(
                {
                    n
                    for part in frag["atoms"].split(",")
                    for n in (range(int(part.split("-")[0]), int(part.split("-")[1]) + 1) if "-" in part else [int(part)])
                }
            )
            # Increment total site states excluding site gs
            self._total_site_states += frag["states"][0] - 1

            # Check if states >= 2 and only singlets
            assert (n_states := sum(frag["states"])) == frag["states"][0], "Only singlet states are supported!"
            assert n_states > 1, f"Too few states for fragment {name}!"

        self.log.debug(f"Total number of site states {self._total_site_states}")

        # Setup embedding interface
        if "embedding" in tmpl_dict:
            if "interface" not in tmpl_dict["embedding"]:
                self.log.error("Interface has to be defined in embedding!")
                raise ValueError
            if "args" not in tmpl_dict["embedding"]:
                tmpl_dict["embedding"]["args"] = []
            if "kwargs" not in tmpl_dict["embedding"]:
                tmpl_dict["embedding"]["kwargs"] = {}

        self.QMin.template.update(tmpl_dict)
        self._read_template = True

    def read_resources(self, resources_file="FRENKEL.resources", kw_whitelist=None):
        super().read_resources(resources_file)

    def setup_interface(self):
        super().setup_interface()

        # Check if number of requested states is doable
        assert (
            n_singlets := self.QMin.molecule["states"][0]
        ) <= self._total_site_states, f"Requested more states than possible ({self._total_site_states})"

        assert sum(self.QMin.molecule["states"]) == n_singlets, "This interface only supports singlet states!"

        kindergarden = {
            name: (frag["interface"], frag["args"], frag["kwargs"]) for name, frag in self.QMin.template["fragments"].items()
        }
        self.instantiate_children(kindergarden)

        # Setup QMin
        for name, frag in self.QMin.template["fragments"].items():
            self.log.debug(f"Setup fragment {name}")

            assert (
                "multipolar_fit" in self._kindergarden[name].get_features()
            ), f"{frag['interface']} does not support multipolar_fit request!"

            self._kindergarden[name].setup_mol(
                {
                    "states": frag["states"],
                    "charge": frag["charges"],
                    "NAtoms": len(frag["atoms"]),
                    "IAn": [NUMBERS[self.QMin.molecule["elements"][a]] for a in frag["atoms"]],
                    "retain": f"retain {self.QMin.requests['retain']}",
                    "savedir": expand_path(os.path.join(self.QMin.save["savedir"], name)),
                    "point_charges": self.QMin.molecule["point_charges"],
                }
            )
            # Set point charges if requested
            if self.QMin.molecule["point_charges"]:
                self._kindergarden[name].QMin.coords["pccharge"] = self.QMin.coords["pccharge"]
                self._kindergarden[name].QMin.molecule["npc"] = self.QMin.molecule["npc"]
                self._kindergarden[name].set_coords(self.QMin.coords["pccoords"], True)

            # Setup template, resources, interface
            with InDir(name):
                self._kindergarden[name].read_resources()
                self._kindergarden[name].read_template()
                self._kindergarden[name].setup_interface()

            # Set scratchdir
            self._kindergarden[name].QMin.resources["scratchdir"] = expand_path(
                os.path.join(self.QMin.resources["scratchdir"], name)
            )

            # Adapt pwd/cwd
            self._kindergarden[name].QMin.resources["pwd"] = expand_path(os.path.join(self.QMin.resources["pwd"], name))
            self._kindergarden[name].QMin.resources["cwd"] = expand_path(os.path.join(self.QMin.resources["cwd"], name))

        # Setup embedding
        if self.QMin.template["embedding"]:
            self._embedding_interface = self._load_interface(self.QMin.template["embedding"]["interface"])(
                self.QMin.template["embedding"]["args"], self.QMin.template["embedding"]["kwargs"]
            )

            self._embedding_interface.setup_mol(
                {
                    "states": [1],
                    "charge": self.QMin.molecule["charge"],
                    "NAtoms": self.QMin.molecule["natom"],
                    "IAn": [NUMBERS[a] for a in self.QMin.molecule["elements"]],
                    "retain": f"retain {self.QMin.requests['retain']}",
                    "savedir": expand_path(os.path.join(self.QMin.save["savedir"], "embedding")),
                }
            )

            with InDir("embedding"):
                self._embedding_interface.read_resources()
                self._embedding_interface.read_template()
                self._embedding_interface.setup_interface()
        # TODO: does it need an embedding child for each fragment?

    def set_coords(self, xyz, pc=False):
        super().set_coords(xyz, pc)
        # Set coords for fragments
        for name, frag in self.QMin.template["fragments"].items():
            if pc:
                self._kindergarden[name].set_coords(xyz, pc)
                continue

            coords = np.zeros((len(frag["atoms"]), 3), dtype=float)
            for idx, a in enumerate(frag["atoms"]):
                # Always Bohr as it comes from parent
                coords[idx] = self.QMin.coords["coords"][a]
            self._kindergarden[name].set_coords(coords, pc)

        # Set coords for embedding
        if self._embedding_interface and not pc:
            self._embedding_interface.set_coords(xyz, pc)

    def read_requests(self, requests_file="QM.in"):
        super().read_requests(requests_file)

        if self.QMin.requests["grad"] and self.QMin.template["embedding"] is None:
            self.log.error("Gradients only available with embedding!")
            raise ValueError

        if self._embedding_interface:
            requests = {"h": True, "multipolar_fit": ["all"], "step": self.QMin.save["step"]}
            if self.QMin.requests["grad"] is not None:
                requests["grad"] = [1]
            self._embedding_interface.read_requests(requests)

            # Check if fragment children can do point charges
            for name, child in self._kindergarden.items():
                assert "point_charges" in child.get_features(), f"Fragment {name} does not support point charges!"

        for iface in self._kindergarden.values():
            requests = {"h": True, "multipolar_fit": ["all"], "step": self.QMin.save["step"]}
            if self.QMin.requests["grad"] is not None:
                requests["grad"] = list(range(1, iface.QMin.molecule["nstates"]))
            iface.read_requests(requests)

    def run(self):
        if self._embedding_interface:
            self._embedding_interface.run()
            embedding_charges = self._embedding_interface.QMout.multipolar_fit.squeeze()

            for name, child in self._kindergarden.items():
                pccharge = np.zeros(embedding_charges.shape[0] - child.QMin.molecule["natom"])
                pccoords = np.zeros((pccharge.shape[0], 3))
                iter_gen = iter(range(pccharge.shape[0]))

                for idx, charge in enumerate(embedding_charges):
                    if idx in self.QMin.template["fragments"][name]["atoms"]:
                        continue
                    pccharge[it_idx := next(iter_gen)] = charge
                    pccoords[it_idx, :] = self.QMin.coords["coords"][idx, :]

                child.QMin.coords["pccharge"] = pccharge
                child.QMin.coords["pccoords"] = pccoords
                child.QMin.molecule["point_charges"] = True
                # TODO: add external pc
        self.run_children(self.log, self._kindergarden, self.QMin.resources["ncpu"])

    def _get_exciton_energies(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Construct and diagonalize exciton Hamiltonian
        returns eigenvalues and eigenvectors
        """
        # Initialize hamiltonian, assign energies
        total_states = 1  # 1 GS prod state
        total_gs_energy = 0
        hamiltonian = np.zeros((self._total_site_states, self._total_site_states), dtype=float)

        for name, frag in self._kindergarden.items():
            site_states = frag.QMin.molecule["states"][0] - 1  # Exclude site GS
            total_states += site_states
            site_gs_energy = frag.QMout.h[0, 0].real
            total_gs_energy += site_gs_energy
            site_energies = np.einsum("ii->i", frag.QMout.h.real)
            self.log.debug(f"Site energies for {name} {site_energies}")

            np.einsum("ii->i", hamiltonian)[total_states - site_states : total_states] = site_energies[1:] - site_gs_energy
        np.einsum("ii->i", hamiltonian)[:] += total_gs_energy  # Add GS prod energy

        # Calculate excitonic couplings
        fragment_list = list(self._kindergarden.keys())

        cnt_i = 1
        for idx, a in enumerate(fragment_list):
            states_a = self._kindergarden[a].QMin.molecule["states"][0] - 1
            coords_a = self._kindergarden[a].QMin.coords["coords"]

            cnt_i += states_a
            cnt_j = 1

            for jdx, b in enumerate(fragment_list):
                states_b = self._kindergarden[b].QMin.molecule["states"][0] - 1

                # Skip upper diagonal
                cnt_j += states_b
                if idx <= jdx:
                    continue

                # Calculate inverse distance matrix for fragment A and B (atoms_a x atoms_b)
                diff = coords_a[:, np.newaxis, :] - self._kindergarden[b].QMin.coords["coords"][np.newaxis, :, :]
                r_ab = 1 / np.sqrt(np.einsum("ijk,ijk->ij", diff, diff))

                # Create monopole matrices (states x natoms)
                s_a, s_b = self._kindergarden[a].states, self._kindergarden[b].states
                monopoles_a = np.stack([self._kindergarden[a].QMout.multipolar_fit[(s_a[0], k)][:, 0] for k in s_a[1:]])
                monopoles_b = np.stack([self._kindergarden[b].QMout.multipolar_fit[(s_b[0], k)][:, 0] for k in s_b[1:]])

                hamiltonian[cnt_i - states_a : cnt_i, cnt_j - states_b : cnt_j] = np.einsum(
                    "ia,jb,ab->ij", monopoles_a, monopoles_b, r_ab
                )
        return np.linalg.eigh(hamiltonian)

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

        energies, _ = self._get_exciton_energies()
        np.einsum("ii->i", self.QMout.h)[:] = energies[: self.QMin.molecule["states"][0]]
        return self.QMout

    def create_restart_files(self):
        super().create_restart_files()
        for child in self._kindergarden.values():
            child.create_restart_files()

    def get_infos(self, INFOS, KEYSTROKES=None):
        pass

    def prepare(self, INFOS, dir_path):
        pass


if __name__ == "__main__":
    SHARC_FRENKEL().main()
