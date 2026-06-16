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

# This script calculates QC results for a system described by the LVC model
#
# Reads QM.in
# Calculates SOC matrix, dipole moments, gradients, nacs and overlaps
# Writes these back to QM.out

# IMPORTS
# external
import os
import sys
import datetime
import numpy as np
import re
import torch

# internal
from SHARC_FAST import SHARC_FAST
from utils import phase_correction
from io import TextIOWrapper
from constants import au2a, au2eV
# from kabsch import kabsch_w as kabsch, kabsch_w_with_deriv
# from numba import njit

authors = "Sebastian Mai"
version = "4.0"
versiondate = datetime.datetime(2025, 10, 1)

changelogstring = """
"""
np.set_printoptions(linewidth=400, formatter={"float": lambda x: f"{x: 9.7}"}, threshold=sys.maxsize)


class SHARC_NAI(SHARC_FAST):
    # _do_kabsch = False
    _diagonalize = True
    # _gammas = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Add resource keys
        self.QMin.resources.update({"diagonalize": True, "keep_U": False})
        self.QMin.resources.types.update(
            {
                "diagonalize": bool,
                "keep_U": bool,
            }
        )

    @staticmethod
    def name():
        return "NaI"

    @staticmethod
    def version():
        return version

    @staticmethod
    def versiondate():
        return versiondate

    @staticmethod
    def changelogstring():
        return changelogstring

    @staticmethod
    def authors():
        return authors

    @staticmethod
    def about():
        return "Interface for calculations with an NaI analytical model"

    @staticmethod
    def description():
        return "     FAST interface for an NaI analytical model"

    def read_template(self, template_filename="NAI.template"):
        self._read_template = True
        return

    def read_resources(self, resources_filename="NAI.resources"):
        if not os.path.isfile(resources_filename):
            self.log.warning("NAI.resources not found; continuing without further settings.")
            self._read_resources = True
            return

        super().read_resources(resources_filename)
        if "diagonalize" in self.QMin.resources:
            self._diagonalize = self.QMin.resources["diagonalize"]

    def setup_interface(self):
        super().setup_interface()

        if self.QMin.molecule["natom"] != 2:
            raise ValueError("Only for two atoms!")
        if self.QMin.molecule["nmstates"] != 2 or self.QMin.molecule["nstates"] != 2:
            raise ValueError("Only for two singlet states!")

        atoms_lower = [a.lower() for a in self.QMin.molecule["elements"]]
        if atoms_lower.count("na") == 1 and atoms_lower.count("i") == 1:
            pass
        else:
            raise ValueError("Only for NaI!")

        if self.persistent:
            for file in os.listdir(self.QMin.save["savedir"]):
                if re.match(r"^U\.npy\.\d+$", file):
                    step = int(file.split('.')[-1])
                    ufile = os.path.join(self.QMin.save["savedir"], file)
                    self.savedict[step] = {'U': np.load(ufile).reshape( (self.QMin.molecule['nmstates'], self.QMin.molecule['nmstates']) )}
                

    def getQMout(self):
        self.QMout["runtime"] = self.clock.measuretime(False)
        return self.QMout



    def run(self):

        req_nmstates = 2
        req_states = 2
        nmstates = 2
        nstates = 2
        natoms = 2

        self._U = np.zeros((req_nmstates, req_nmstates), dtype=float)
        Hd = np.zeros((req_nmstates, req_nmstates), dtype=float)

        coords = self.QMin.coords["coords"]  

        # --- constants (converted to torch tensors for autograd) ---
        a2ev = torch.tensor(au2eV)   # eV/au
        a2au = torch.tensor(au2a)       # A/au
        A2   = 2760.0 / a2ev             # eV
        B2   = 2.398 / a2ev**(1./8.) / a2au              # eV^(1/8) A
        C2   = 11.3 / a2ev / a2au**6               # eV A^6
        lp   = 0.408 / a2au**3              # A^3
        lm   = 6.431 / a2au**3              # A^3
        ro   = 0.3489 / a2au             # A
        de   = 2.075 / a2ev              # eV
        e2   = 1. #14.3996 / a2ev / a2au                # au
        #
        A1   = 0.813 / a2ev              # eV
        b1   = 4.08 * a2au               # A^-1
        R0   = 2.67 / a2au               # A
        #
        A12  = 0.055 / a2ev              # eV
        b12  = 0.6931 * a2au**2             # A^-2
        Rx   = 6.93 / a2au               # A

        with torch.enable_grad():
            coords_p = torch.from_numpy(coords)
            coords_p.requires_grad = bool(self.QMin.requests["grad"])
            Ri = coords_p[0]
            Rj = coords_p[1]
            Rij = torch.norm(Ri - Rj)   # already in bohr

            # Hamiltonian elements should be in Hartree
            H11 = (
                (A2 + (B2 / Rij) ** 8)
                * torch.exp(-Rij / ro)
                - e2 / Rij
                - e2 * (lp + lm) / (2.0 * Rij**4)
                - C2 / Rij**6
                - 2.0 * e2 * lp * lm / Rij**7
                + de
            )

            H12 = A12 * torch.exp(-b12 * ((Rij - Rx) ** 2))

            H22 = A1 * torch.exp(-b1 * (Rij - R0))

            Hd = torch.stack(
                [torch.stack([H11, H12]),
                torch.stack([H12, H22])]
            )

            E, U = torch.linalg.eigh(Hd)

            energies = []
            forces = []

            for i in range(len(E)):
                e = E[i]
                if self.QMin.requests["grad"]:
                    coords_p.grad = None          # clear previous gradients
                    e.backward(retain_graph=True) # compute gradient of this state
                    grad = coords_p.grad.detach().clone().numpy()
                    forces.append(grad)          # minus gradient = force
                energies.append(e.detach().numpy())
            energies = np.array(energies)          # shape (n_states,)
            if self.QMin.requests["grad"]:
                forces = np.stack(forces, axis=0)  # shape (n_states, n_atoms, 3)


        # Had
        H_ad = torch.diag(E).detach().numpy()

        # overlaps
        U_np = U.detach().numpy()  
        if self.QMin.requests["overlap"] or self.QMin.requests["phases"]:
            if self.QMin.save["step"] == 0:
                pass
            elif self.persistent:
                Uold = self.savedict[self.QMin.save['step']-1]["U"]
            else:
                Uold = np.load(os.path.join(self.QMin.save["savedir"], f"U.npy.{self.QMin.save['step']-1}")).reshape(self._U.shape)
            overlap = Uold.T @ U_np
            if self.QMin.requests["phases"]:
                _, phases = phase_correction(overlap)

        # store old overlaps
        if not self.QMin.save["samestep"]:
            # store U matrix
            if self.persistent:
                self.savedict[self.QMin.save['step']] = {'U': np.copy(U_np)}
                # self.savedict["last_step"] = self.QMin.save['step']
            else:
                with open(os.path.join(self.QMin.save["savedir"], f"U.npy.{self.QMin.save['step']}"), 'wb') as f:
                    np.save(f, U_np)  # writes a binary file (can be read with numpy.load())
            
            # keep all U matrices 
            if self.QMin.resources["keep_U"]:
                if "all_U" not in self.__dict__:
                    self.all_U = []
                self.all_U.append(U_np)

        # dipoles
        Rvec = Rj-Ri
        Rdir = Rvec / torch.norm(Rvec)  # unit vector
        # Dipole in diabatic basis: zero diagonals, 0.1 a.u. along Rij for off-diagonal
        dtype = U.dtype
        device = U.device
        mu_d = torch.zeros(2, 2, 3, dtype=dtype, device=device)
        mu_d[0, 1] = 0.1 * Rdir
        mu_d[1, 0] = 0.1 * Rdir
        mu_ad = torch.zeros_like(mu_d)
        for k in range(3):   # x, y, z
            mu_ad[:,:,k] = U.T @ mu_d[:,:,k] @ U   # adiabatic basis
        mu_ad = mu_ad.permute(2, 0, 1)  # now shape = (3,2,2)
        dipoles = mu_ad.detach().numpy()



        # ======================================== assign to QMout =========================================
        self.log.debug(f"requests: {self.QMin.requests}")
        self.QMout.states = [req_states]
        self.QMout.nstates = self.QMin.molecule["nstates"]
        self.QMout.nmstates = self.QMin.molecule["nmstates"]
        self.QMout.natom = self.QMin.molecule["natom"]
        self.QMout.npc = self.QMin.molecule["npc"]
        self.QMout.point_charges = False

        self.QMout.h = H_ad
        self.QMout.dm = dipoles
        if self.QMin.requests["overlap"]:
            self.QMout.overlap = overlap
        if self.QMin.requests["phases"]:
            self.QMout.phases = phases
        if self.QMin.requests["grad"]:
            self.QMout.grad = forces

        # TODO: later
        # if self.QMin.requests["nacdr"]:
        #     self.QMout.nacdr = nacdr
        #     if do_pc:
        #         self.QMout.nacdr_pc = np.einsum("kij->ijk", nacdr_pc).reshape((req_nmstates, req_nmstates, -1, 3))

        return

    def create_restart_files(self):
        super().create_restart_files()
        if self.persistent:
            for istep in self.savedict:
                if not isinstance(istep,int):
                    continue
                with open( os.path.join(self.QMin.save["savedir"], f'U.npy.{istep}'), 'wb') as f:
                    np.save(f, self.savedict[istep]["U"])  # writes a binary file (can be read with numpy.load())

            if self.QMin.resources["keep_U"]:
                all_U = np.array(self.all_U)
                np.save(os.path.join(self.QMin.save["savedir"], f"U_0-{self.QMin.save['step']}.npy"), all_U)
        # else: nothing is done because run() has already saved the U matrix

    def get_features(self, KEYSTROKES: TextIOWrapper = None) -> set:
        return {
            "h",
            "dm",
            "grad",
            # "nacdr",
            "overlap",
            "phases",
        }

    def get_infos(self, INFOS: dict, KEYSTROKES: TextIOWrapper = None) -> dict:
        self.log.info("=" * 80)
        self.log.info(f"{'||':<78}||")
        self.log.info(f"||{'NaI interface setup': ^76}||\n{'||':<78}||")
        self.log.info("=" * 80)
        self.log.info("\n")

        return INFOS

    def dyson_orbitals_with_other(self, other):
        raise NotImplementedError()

    def prepare(self, INFOS: dict, dir_path: str):
        pass
        # super().prepare(INFOS, dir_path)

if __name__ == "__main__":
    from logger import loglevel

    lvc = SHARC_NAI(loglevel=loglevel)
    lvc.main()
