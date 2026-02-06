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
import pickle
import shutil
from io import TextIOWrapper

import numpy as np
import yaml
from constants import NUMBERS
from SHARC_HYBRID import SHARC_HYBRID
from SHARC_INTERFACE import SHARC_INTERFACE
from utils import InDir, expand_path, link, mkdir, question

__all__ = ["SHARC_FRENKEL"]

AUTHORS = "Sascha Mausenberger"
VERSION = "4.0"
VERSIONDATE = datetime.datetime(2025, 5, 7)
NAME = "FRENKEL"
DESCRIPTION = "   HYBRID interface for Frenkel exciton model"

CHANGELOGSTRING = """
"""


def generate_cube_data(
    kindergarden_values: list[SHARC_INTERFACE], coeffs: np.ndarray, max_states: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate per-atom effective transition charges for all exciton states

    kindergarden_values:    List of kindergarden instances
    coeffs:                 Exciton wave function
    max_states:             Save states from first excited to max_states
    """
    atoms_per_frag = [f.QMin.molecule["natom"] for f in kindergarden_values]
    exc_per_frag = [f.QMin.molecule["states"][0] - 1 for f in kindergarden_values]
    all_coords = np.concatenate([f.QMin.coords["coords"] for f in kindergarden_values])

    atom_charges = []
    for frag in kindergarden_values:
        atom_charges.append([NUMBERS[a] for a in frag.QMin.molecule["elements"]])
    atom_charges = np.concatenate(atom_charges, axis=0)

    trans_charges_per_atom = np.zeros((coeffs.shape[0], int(np.sum(atoms_per_frag))))

    offset_at = 0
    offset_exc = 0
    for frag, nat, nexc in zip(kindergarden_values, atoms_per_frag, exc_per_frag):
        local_charge = np.stack([frag.QMout.multipolar_fit[(frag.states[0], state)][:, 0] for state in frag.states[1:]])
        trans_charges_per_atom[offset_exc : offset_exc + nexc, offset_at : offset_at + nat] = local_charge
        offset_at += nat
        offset_exc += nexc

    all_trans_charges = coeffs[:, :max_states].T @ trans_charges_per_atom
    return all_trans_charges, all_coords, atom_charges


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
        self.QMin.template.update({"fragments": None, "embedding": None, "embedding_lj": None})
        self.QMin.template.types.update({"fragments": dict, "embedding": dict, "embedding_lj": dict})

        # Update resource keys
        self.QMin.resources.update({"save_cube": False})
        self.QMin.resources.types.update({"save_cube": bool})

        # Template interface structure
        self._interface_templ = {
            "interface": str,  # Name of SHARC interface
            "args": list,  # Init arguments for child
            "kwargs": dict,  # Keyword args for child
            "atoms": str,  # List/Range of atoms
            "states": list,  # List of states
            "charges": list,  # List of charges
        }

        self.template_file = None
        self._resources_file = None

        # Interface for electrostatic embedding
        self._embedding_interface = None
        self._embedding_lj = None  # LJ repulsion

        # Keep track of total site states to preallocate Hamiltonian
        self._total_site_states = 1  # GS prod
        self._allow_coupling = False  # Allow NACs and overlaps
        self._n_fragments = None

        # Precompute atoms and states
        self._atoms = None
        self._states = None

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
        if not self._read_template:
            self.template_file = expand_path(
                question(
                    "Please specify the path to your FRENKEL.template file",
                    str,
                    KEYSTROKES=KEYSTROKES,
                    default="FRENKEL.template",
                )
            )
            self.read_template(self.template_file)
            kindergarden = {
                name: (frag["interface"], frag["args"], frag["kwargs"]) for name, frag in self.QMin.template["fragments"].items()
            }
            self.instantiate_children(kindergarden)

        all_features = set(["h", "grad", "point_charges", "dm", "overlap", "phases", "nacdr"])
        for child in self._kindergarden.values():
            all_features &= child.get_features(KEYSTROKES=KEYSTROKES)
        self.log.debug(f"Features: {all_features}")
        all_features.add("theodore")
        if self._allow_coupling:
            all_features.add("nacdr")
            all_features.add("overlap")
            all_features.add("phases")
        return all_features

    def read_template(self, template_file="FRENKEL.template", kw_whitelist=None):
        self.log.debug(f"Parsing template file {template_file}")

        # Open template_file and parse yaml
        with open(template_file, "r", encoding="utf-8") as tmpl_file:
            tmpl_dict = yaml.safe_load(tmpl_file)
            self.log.debug(f"Parsing yaml file:\n{tmpl_dict}")

        assert len(tmpl_dict["fragments"]) > 1, "At least two fragments have to be defined!"

        self._atoms = []
        self._states = []
        for name, frag in tmpl_dict["fragments"].items():
            # Set default value for args, kwargs and charges
            if "args" not in frag:
                frag["args"] = []
            if "kwargs" not in frag:
                frag["kwargs"] = {}
            if "charges" not in frag:
                self.log.info(f"No charge defined for fragment {name}, set default 0.")
                frag["charges"] = [0]

            # Check if all parameters are present and of correct type
            for k, v in self._interface_templ.items():
                if k not in frag:
                    self.log.error(f"{k} has to be defined in fragment {name}")
                    raise ValueError
                if not isinstance(frag[k], v):
                    self.log.error(f"Value of key {k} in fragment {name} must be of type {v}")
                    raise ValueError

            # Convert atoms string to array
            frag["atoms"] = np.array(
                sorted(
                    {
                        n
                        for part in frag["atoms"].split(",")
                        for n in (range(int(part.split("-")[0]), int(part.split("-")[1]) + 1) if "-" in part else [int(part)])
                    }
                )
            )
            self._atoms.append(frag["atoms"])
            self._states.append(frag["states"][0] - 1)
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
        if "embedding_lj" in tmpl_dict:
            if "interface" not in tmpl_dict["embedding_lj"]:
                self.log.error("Interface has to be defined in embedding!")
                raise ValueError
            if "args" not in tmpl_dict["embedding_lj"]:
                tmpl_dict["embedding_lj"]["args"] = []
            if "kwargs" not in tmpl_dict["embedding_lj"]:
                tmpl_dict["embedding_lj"]["kwargs"] = {}

        self.QMin.template.update(tmpl_dict)
        # NACs available if all fragments only have 1 excited state
        if all(frag["states"][0] < 3 for frag in self.QMin.template["fragments"].values()):
            self._allow_coupling = True
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
        self._n_fragments = len(self._kindergarden)

        # Setup QMin
        for name, frag in self.QMin.template["fragments"].items():
            self.log.debug(f"Setup fragment {name}")

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
                self._kindergarden[name].set_pccharge(self.QMin.coords["pccharge"])
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
            self._embedding_interface.QMin.molecule["unit"] = self.QMin.molecule["unit"]
            self._embedding_interface.QMin.molecule["factor"] = self.QMin.molecule["factor"]

            with InDir("embedding"):
                self._embedding_interface.read_resources()
                self._embedding_interface.read_template()
                self._embedding_interface.setup_interface()
            self._embedding_interface.QMin.resources["scratchdir"] = expand_path(
                os.path.join(self.QMin.resources["scratchdir"], "embedding")
            )

        if self.QMin.template["embedding_lj"]:
            self._embedding_lj = self._load_interface(self.QMin.template["embedding_lj"]["interface"])(
                self.QMin.template["embedding_lj"]["args"], self.QMin.template["embedding_lj"]["kwargs"]
            )

            self._embedding_lj.setup_mol(
                {
                    "states": [1],
                    "charge": self.QMin.molecule["charge"],
                    "NAtoms": self.QMin.molecule["natom"],
                    "IAn": [NUMBERS[a] for a in self.QMin.molecule["elements"]],
                    "retain": f"retain {self.QMin.requests['retain']}",
                    "savedir": expand_path(os.path.join(self.QMin.save["savedir"], "embedding_lj")),
                }
            )
            self._embedding_lj.QMin.molecule["unit"] = self.QMin.molecule["unit"]
            self._embedding_lj.QMin.molecule["factor"] = self.QMin.molecule["factor"]

            with InDir("embedding_lj"):
                self._embedding_lj.read_resources()
                self._embedding_lj.read_template()
                self._embedding_lj.setup_interface()
            self._embedding_lj.QMin.resources["scratchdir"] = expand_path(
                os.path.join(self.QMin.resources["scratchdir"], "embedding_lj")
            )
        # TODO: does it need an embedding child for each fragment?

    def set_coords(self, xyz, pc=False):
        super().set_coords(xyz, pc)
        # Set coords for fragments
        for name, frag in self.QMin.template["fragments"].items():
            if pc:
                self._kindergarden[name].set_coords(xyz * self.QMin.molecule["factor"], pc)
                continue
            self._kindergarden[name].set_coords(self.QMin.coords["coords"][frag["atoms"]], pc)

        # Set coords for embedding
        if self._embedding_interface and not pc:
            self._embedding_interface.set_coords(xyz, pc)
        if self._embedding_lj and not pc:
            self._embedding_lj.set_coords(xyz, pc)

    def read_requests(self, requests_file="QM.in"):
        super().read_requests(requests_file)

        if self._embedding_interface:
            self._embedding_interface.read_requests({"h": True, "multipolar_fit": ["all"], "step": self.QMin.save["step"]})

            # Check if fragment children can do point charges
            for name, child in self._kindergarden.items():
                assert "point_charges" in child.get_features(), f"Fragment {name} does not support point charges!"

        if self._embedding_lj:
            self._embedding_lj.read_requests({"h": True, "grad": [0], "step": self.QMin.save["step"]})

        for iface in self._kindergarden.values():
            requests = {"h": True, "multipolar_fit": ["all"], "step": self.QMin.save["step"]}
            if self.QMin.requests["grad"] or (self.QMin.requests["nacdr"] and not self._allow_coupling):
                requests["grad"] = list(range(1, iface.QMin.molecule["nstates"] + 1))
            if self.QMin.requests["nacdr"] and not self._allow_coupling:
                requests["nacdr"] = "all"
            if (self.QMin.requests["overlap"] or self.QMin.requests["phases"]) and not self._allow_coupling:
                requests["overlap"] = True
            iface.read_requests(requests)

    def run(self):
        if self._embedding_interface:
            self._embedding_interface.run()
            embedding_charges = self._embedding_interface.QMout.multipolar_fit[
                (self._embedding_interface.states[0], self._embedding_interface.states[0])
            ][:, 0]

            for name, child in self._kindergarden.items():
                atoms = self.QMin.template["fragments"][name]["atoms"]
                child.set_pccharges(embedding_charges[~atoms])
                child.set_coords(self.QMin.coords["coords"][~atoms, :], True)
                child.QMin.molecule["point_charges"] = True
                # TODO: add external pc
        if self._embedding_lj and self.QMin.requests["grad"]:
            self._embedding_lj.run()
        self.run_children(self.log, self._kindergarden, self.QMin.resources["ncpu"])

    def _wfa(self, wavefunction: np.ndarray) -> list[list[str, np.ndarray]]:
        """
        Calculate excitonic wave function descriptors
        D_A = SUM_i c_Ai²
        PR = SUM_A D_A / SUM_A D_A²
        """
        np.square(wavefunction, out=wavefunction)

        descriptors = []
        site_index = 0
        da_sum = np.zeros(self._total_site_states)
        da_sq_sum = np.zeros(self._total_site_states)

        for fragment, child in self._kindergarden.items():
            num_states = child.QMin.molecule["states"][0]
            da_array = np.zeros(self._total_site_states)

            # Sum over the states of this fragment (excluding index 0)
            da_array[1:] = np.sum(wavefunction[1:, 1 + site_index : site_index + num_states], axis=1)

            da_sum += da_array
            da_sq_sum += da_array**2
            site_index += num_states - 1

            descriptors.append([f"D_{fragment}", da_array])

        with np.errstate(divide="ignore", invalid="ignore"):
            pr_array = da_sum / da_sq_sum
            pr_array[~np.isfinite(pr_array)] = 0  # Replace NaN or inf with 0
            descriptors.append(["PR", pr_array])

        return descriptors

    def _get_exciton_energies(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Construct and diagonalize exciton Hamiltonian
        returns eigenvalues and eigenvectors
        """
        frags = list(self._kindergarden.items())
        coords = [f.QMin.coords["coords"] for _, f in frags]
        states = self._states

        hamiltonian = np.zeros((self._total_site_states, self._total_site_states), dtype=float)
        # Create 0->n transition monopole matrices (states x natoms)
        monopoles = [np.stack([f.QMout.multipolar_fit[(f.states[0], k)][:, 0] for k in f.states[1:]]) for _, f in frags]

        cnt_i = 1
        for idx, (_, a) in enumerate(frags):
            cnt_i += states[idx]
            cnt_j = 1

            # Add site gs energy to GS prod energy
            np.einsum("ii->i", hamiltonian)[:] += (gs_en := a.QMout.h[0, 0].real)
            np.einsum("ii->i", hamiltonian)[cnt_i - states[idx] : cnt_i] += np.einsum("ii->i", a.QMout.h[1:, 1:]).real - gs_en

            if idx == self._n_fragments - 1:  # Last fragment has no off diagonal
                break

            for jdx, _ in enumerate(frags):
                # Skip lower diagonal
                cnt_j += states[jdx]
                if idx >= jdx:
                    continue

                # Calculate inverse distance matrix for fragment A and B (atoms_a x atoms_b)
                diff = coords[idx][:, np.newaxis, :] - coords[jdx][np.newaxis, :, :]
                r2 = np.einsum("...k,...k->...", diff, diff)
                r_ab = 1.0 / np.sqrt(r2)

                hamiltonian[cnt_i - states[idx] : cnt_i, cnt_j - states[jdx] : cnt_j] = monopoles[idx] @ r_ab @ monopoles[jdx].T
        return np.linalg.eigh(hamiltonian, UPLO="u")

    def _get_derivatives(self) -> np.ndarray:
        """
        Calculate derivative of Hamiltonian (Hellmann-Feynman theorem)
        dE/dR ~ site-state gradient + dV/dR, assuming transition charges
        are not a function of R
        """
        frags = list(self._kindergarden.items())
        coords = [f.QMin.coords["coords"] for _, f in frags]
        states = self._states
        atoms = self._atoms

        hamiltonian_dr = np.zeros((self._total_site_states, self._total_site_states, self.QMin.molecule["natom"], 3))
        monopoles = [np.stack([f.QMout.multipolar_fit[(f.states[0], k)][:, 0] for k in f.states[1:]]) for _, f in frags]

        state_cnt = 1
        for idx, (_, a) in enumerate(frags):

            # Add GS gradient to GS prod. gradient and excited site gradients to diagonal
            np.einsum("iijk->ijk", hamiltonian_dr)[:, atoms[idx], :] = a.QMout.grad[0]
            np.einsum("iijk->ijk", hamiltonian_dr)[state_cnt : state_cnt + states[idx], atoms[idx], :] = a.QMout.grad[1:]

            # Create 0->n transition monopole matrices (states x natoms)
            if idx == self._n_fragments - 1:  # Last fragment has no off diagonal
                break

            state_cnt += states[idx]
            state_cnt_b = 1
            for jdx, _ in enumerate(frags):
                # Skip lower diagonal
                state_cnt_b += states[jdx]
                if idx >= jdx:
                    continue

                # d/dR(1/|R_a-R_b|) = -R_a-R_b/|R_a-R_b|**3
                diff = coords[idx][:, np.newaxis, :] - coords[jdx][np.newaxis, :, :]
                r2 = np.einsum("...k,...k->...", diff, diff)
                inv_r3 = 1.0 / (r2 * np.sqrt(r2))
                r_ab = diff * inv_r3[..., None]

                # dV/dR for atoms on fragment A and B
                d_va = -np.einsum("ia,jb,abk->ijak", monopoles[idx], monopoles[jdx], r_ab, optimize=True)
                d_vb = np.einsum("ia,jb,abk->ijbk", monopoles[idx], monopoles[jdx], r_ab, optimize=True)

                # Fill off diagonals dH_ij=dH_ji
                hamiltonian_dr[state_cnt - states[idx] : state_cnt, state_cnt_b - states[jdx] : state_cnt_b, atoms[idx], :] = d_va
                hamiltonian_dr[state_cnt_b - states[jdx] : state_cnt_b, state_cnt - states[idx] : state_cnt, atoms[idx], :] = (
                    d_va.swapaxes(0, 1)
                )
                hamiltonian_dr[state_cnt - states[idx] : state_cnt, state_cnt_b - states[jdx] : state_cnt_b, atoms[jdx], :] = d_vb
                hamiltonian_dr[state_cnt_b - states[jdx] : state_cnt_b, state_cnt - states[idx] : state_cnt, atoms[jdx], :] = (
                    d_vb.swapaxes(0, 1)
                )
        return hamiltonian_dr

    def _get_exciton_dipoles(self, coeffs: np.ndarray) -> np.ndarray:
        """
        Calculate (transition) dipole moments of exciton states

        coeffs: n_states x n_states array of eigenvectors from Hamiltonian
        """
        dipoles = np.zeros((3, self._total_site_states, self._total_site_states))

        state_cnt = 1
        for a in self._kindergarden.values():
            states_a = a.QMin.molecule["states"][0] - 1
            coords_a = a.QMin.coords["coords"]

            # Add GS prod. dipole moment
            dipoles[:, 0, 0] += (
                gs_dp := np.einsum("i,ij->j", a.QMout.multipolar_fit[(a.states[0], a.states[0])][:, 0], coords_a)
            )

            for idx, s1 in enumerate(a.states[1:]):
                gsle = np.einsum("i,ij->j", a.QMout.multipolar_fit[(a.states[0], s1)][:, 0], coords_a)
                dipoles[:, 0, state_cnt + idx] = gsle
                dipoles[:, state_cnt + idx, 0] = gsle
                for jdx, s2 in enumerate(a.states[1:]):
                    dipoles[:, state_cnt + idx, state_cnt + jdx] = np.einsum(
                        "i,ik->k", a.QMout.multipolar_fit[(s1, s2)][:, 0], coords_a
                    ) - (gs_dp if idx == jdx else 0.0)

            state_cnt += states_a
        return np.einsum("pi,kpq,qj->kij", coeffs, dipoles, coeffs, optimize=True)

    def _get_exciton_overlaps(self, prev_coeffs: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
        """
        Calculate overlaps of excitonic states <c(t)|S_sites|c(t+dt)>

        pref_coeffs:    n_states x n_states array of eigenvectors from Hamiltonian
                        from last step
        coeffs:         n_states x n_states array of eigenvectors from Hamiltonian
                        from current step
        """
        if self._allow_coupling:
            return prev_coeffs.T @ coeffs

        site_overlaps = np.eye(self._total_site_states)

        state_cnt = 1
        for site in self._kindergarden.values():
            state_cnt += (n_states := site.QMin.molecule["states"][0] - 1)
            site_overlaps[state_cnt - n_states : state_cnt, state_cnt - n_states : state_cnt] = site.QMout.overlap[1:, 1:]
        return prev_coeffs.T @ site_overlaps @ coeffs

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
        nstates = self.QMin.molecule["states"][0]

        energies, coeffs = self._get_exciton_energies()
        np.einsum("ii->i", self.QMout.h)[:] = energies[:nstates]

        # Save eigenvectors for overlap calculations
        with open(os.path.join(self.QMin.save["savedir"], f"eigenvectors.{self.QMin.save['step']}"), "wb") as f:
            np.save(f, coeffs)

        if self.QMin.requests["grad"] or self.QMin.requests["nacdr"]:
            # dH/dR including site gradients
            hamiltonian_dr = self._get_derivatives()
            if self.QMin.requests["nacdr"]:
                # Add intra site NACs to block diaginal if available
                # <i|dH/dR|j> * (E_i - E_j)
                gap = energies[None, :] - energies[:, None]
                if not self._allow_coupling:
                    s_cnt = 1
                    for label, frag in self._kindergarden.items():
                        nacs = frag.QMout.nacdr
                        atoms = self.QMin.template["fragments"][label]["atoms"]
                        states = s_cnt + frag.QMin.molecule["states"][0] - 1
                        hamiltonian_dr[s_cnt:states, s_cnt:states, atoms, :] += (
                            nacs[1:, 1:, :, :] * gap[s_cnt:states, s_cnt:states, None, None]
                        )
                        s_cnt += states - s_cnt
            if self.QMin.requests["grad"]:
                self.QMout.grad = np.einsum("in,jn,ijkl->nkl", coeffs, coeffs, hamiltonian_dr, optimize=True)[:nstates, :, :]
                if self._embedding_lj:
                    self.QMout.grad += self._embedding_lj.getQMout()["grad"]

            if self.QMin.requests["nacdr"]:
                # <i|dH/dR|j> / (E_i - E_j)
                # Make sure diagonal is 0 after division
                gap[np.diag_indices_from(gap)] = np.inf
                np.divide(hamiltonian_dr, gap[:, :, None, None], out=hamiltonian_dr)
                ct = np.ascontiguousarray(coeffs.T)
                tmp = np.tensordot(hamiltonian_dr, coeffs, axes=([1], [0]))
                out = np.tensordot(ct, tmp, axes=([1], [0]))
                self.QMout.nacdr = out.transpose(0, 3, 1, 2)[:nstates, :nstates, :, :]

        if self.QMin.requests["dm"]:
            self.QMout.dm[:, :nstates, :nstates] = self._get_exciton_dipoles(coeffs)[:, :nstates, :nstates]

        if self.QMin.requests["overlap"] or self.QMin.requests["phases"]:
            prev_coeffs = np.load(os.path.join(self.QMin.save["savedir"], f"eigenvectors.{self.QMin.save['step']-1}"))
            overlap = self._get_exciton_overlaps(prev_coeffs, coeffs)[:nstates, :nstates]
            if self.QMin.requests["overlap"]:
                self.QMout.overlap = overlap
            if self.QMin.requests["phases"]:
                self.QMout.phases = np.einsum("ii->i", overlap).copy()
                self.QMout.phases[self.QMout.phases > 0] = 1
                self.QMout.phases[self.QMout.phases < 0] = -1

        if self.QMin.requests["theodore"]:
            self.QMout.prop1d.extend(self._wfa(coeffs.copy()))

        if self.QMin.resources["save_cube"]:
            cube_data = generate_cube_data(self._kindergarden.values(), coeffs, nstates)
            with open(os.path.join(self.QMin.save["savedir"], f"cube_data.{self.QMin.save['step']}"), "wb") as f:
                pickle.dump(cube_data, f, protocol=pickle.HIGHEST_PROTOCOL)

        self.QMout["runtime"] = self.clock.measuretime(False)
        return self.QMout

    def create_restart_files(self):
        super().create_restart_files()
        for child in self._kindergarden.values():
            child.create_restart_files()

    def write_step_file(self):
        super().write_step_file()
        if self._embedding_interface:
            self._embedding_interface.write_step_file()
        if self._embedding_lj:
            self._embedding_lj.write_step_file()

    def get_infos(self, INFOS: dict, KEYSTROKES: TextIOWrapper | None = None) -> dict:
        self.log.info("=" * 80)
        self.log.info(f"{'||':<78}||")
        self.log.info(f"||{'FRENKEL interface setup': ^76}||\n{'||':<78}||")
        self.log.info("=" * 80)
        self.log.info("\n")

        if not self.template_file:
            if question("Do you have an FRENKEL.template file?", bool, KEYSTROKES=KEYSTROKES, autocomplete=False, default=False):
                while not os.path.isfile(
                    (template_file := question("Specify the path:", str, KEYSTROKES=KEYSTROKES, default="ORCA.resources"))
                ):
                    self.log.info(f"file at {template_file} does not exist!")
                self.template_file = expand_path(template_file)
                self.read_template(template_file)
            kindergarden = {
                name: (frag["interface"], frag["args"], frag["kwargs"]) for name, frag in self.QMin.template["fragments"].items()
            }
            self.instantiate_children(kindergarden)

        if question("Do you have an FRENKEL.resources file?", bool, KEYSTROKES=KEYSTROKES, autocomplete=False, default=False):
            self._resources_file = expand_path(
                question(
                    "Specify path to FRENKEL.resources",
                    str,
                    KEYSTROKES=KEYSTROKES,
                    autocomplete=True,
                    default="FRENKEL.resources",
                )
            )

        if self.QMin.template["embedding"]:
            self._embedding_interface = self._load_interface(self.QMin.template["embedding"]["interface"])(
                self.QMin.template["embedding"]["args"], self.QMin.template["embedding"]["kwargs"]
            )
            self._embedding_interface.log = self.log
            self.log.info("=" * 80)
            self.log.info(f"{'||':<78}||")
            self.log.info(f"||{'Embedding interface setup': ^76}||\n{'||':<78}||")
            self.log.info("=" * 80)
            self.log.info("\n")
            self._embedding_interface.get_infos(INFOS, KEYSTROKES=KEYSTROKES)

        self.log.info("=" * 80)
        self.log.info(f"{'||':<78}||")
        self.log.info(f"||{'Child interface setup': ^76}||\n{'||':<78}||")
        self.log.info("=" * 80)
        self.log.info("\n")
        for child, instance in self._kindergarden.items():
            self.log.info(f"Setting up interface {child}")
            instance.log = self.log
            instance.get_infos(INFOS, KEYSTROKES=KEYSTROKES)
        return INFOS

    def prepare(self, INFOS: dict, dir_path: str) -> None:
        create_file = link if INFOS["link_files"] else shutil.copy

        create_file(self.template_file, os.path.join(dir_path, "FRENKEL.template"))
        if self._resources_file:
            create_file(self._resources_file, os.path.join(dir_path, "FRENKEL.resources"))

        for child, instance in self._kindergarden.items():
            child_dir = os.path.join(dir_path, child)
            mkdir(child_dir)
            instance.prepare(INFOS, child_dir)

        if self.QMin.template["embedding"]:
            child_dir = os.path.join(dir_path, "embedding")
            mkdir(child_dir)
            self._embedding_interface.prepare(INFOS, child_dir)


if __name__ == "__main__":
    SHARC_FRENKEL().main()
