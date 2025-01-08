#!/usr/bin/env python3

# ******************************************
#
#    SHARC Program Suite
#
#    Copyright (c) 2019 University of Vienna
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

# internal
from SHARC_FAST import SHARC_FAST
from utils import readfile, writefile, question, expand_path, phase_correction
from io import TextIOWrapper
from constants import U_TO_AMU
from kabsch import kabsch_w as kabsch, kabsch_w_with_deriv
from numba import njit

authors = "Sebastian Mai and Severin Polonius"
version = "4.0"
versiondate = datetime.datetime(2023, 8, 25)

changelogstring = """
"""
np.set_printoptions(linewidth=400, formatter={"float": lambda x: f"{x: 9.7}"}, threshold=sys.maxsize)


class SHARC_LVC(SHARC_FAST):
    _read_resources = True
    _do_kabsch = False
    _diagonalize = True
    _gammas = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Add resource keys
        self.QMin.resources.update({"do_kabsch": False, "diagonalize": True, "keep_U": False})
        self.QMin.resources.types.update(
            {
                "do_kabsch": bool,
                "diagonalize": bool,
                "keep_U": bool,
            }
        )

    @staticmethod
    def name():
        return "LVC"

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
        return "Interface for calculations with linear vibronic coupling models"

    @staticmethod
    def description():
        return "     FAST interface for linear/quadratic vibronic coupling models (LVC, QVC, LVC/MM)"

    def read_template(self, template_filename="LVC.template"):
        f = open(os.path.abspath(template_filename), "r")
        V0file = f.readline()[:-1]
        self.read_V0(expand_path(V0file))
        self.parsed_states = self.parseStates(f.readline())
        self.template_states = self.parsed_states["states"]
        states = self.QMin.molecule["states"]

        if (len(states) > len(self.template_states)) or any(a > b for (a, b) in zip(states, self.template_states)):
            self.log.error(
                f'states from QM.in and nstates from LVC.template are inconsistent! {self.QMin.molecule["states"]} != {self.template_states}'
            )
            raise ValueError(
                f'impossible to calculate {self.QMin.molecule["states"]} with template holding {self.template_states}'
            )
        if any(a < b for (a, b) in zip(states, self.template_states)):
            self.log.warning(f"Calculating with {self.template_states} but returning {states}")

        natom = self.QMin.molecule["natom"]
        r3N = 3 * natom
        nmstates = self.parsed_states["nmstates"]
        states = self.parsed_states["states"]

        self._H_i = {im: np.zeros((r3N, n, n), dtype=float) for im, n in enumerate(states) if n != 0}
        self._G = {im: np.zeros((n, n, r3N, r3N), dtype=float) for im, n in enumerate(states) if n != 0}
        self._h = {im: np.zeros((n, n), dtype=float) for im, n in enumerate(states) if n != 0}
        self._dipole = np.zeros((3, nmstates, nmstates), dtype=complex)
        self._soc = np.zeros((nmstates, nmstates), dtype=complex)
        self._U = np.zeros((nmstates, nmstates), dtype=float)
        self._Q = np.zeros(r3N, float)
        xyz = {"X": 0, "Y": 1, "Z": 2}
        soc_real = True
        dipole_real = True
        line = f.readline()
        if line.startswith("charge"):
            charges = [int(x) for x in line.split()[1:]]
            c1 = np.array(charges)
            c2 = np.array(self.QMin.molecule["charge"])
            st = np.array(self.QMin.molecule["states"])
            # zero-pad shorter one
            max_len = max(len(c1), len(c2), len(st))
            c1 = np.pad(c1, (0, max_len - len(c1)), mode='constant', constant_values=0)
            c2 = np.pad(c2, (0, max_len - len(c2)), mode='constant', constant_values=0)
            st = np.pad(st, (0, max_len - len(st)), mode='constant', constant_values=0)
            # remove charge for irrelevant multiplicities
            mask = np.array(st) != 0
            c1 = c1[mask]
            c2 = c2[mask]
            if not np.array_equal(c1,c2):
                self.log.error('Charges from request and in template are not consistent!')
                raise RuntimeError()
            line = f.readline()
        self.QMout.charges = self.QMin.molecule["charge"]
        # NOTE: possibly assign whole array with index accessor (numpy)
        if line == "epsilon\n":
            z = int(f.readline()[:-1])

            def a(x):
                v = f.readline().split()
                return (int(v[0]) - 1, int(v[1]) - 1, float(v[2]))

            for im, s, v in map(a, range(z)):
                self._h[im][s, s] += v
            line = f.readline()

        if line == "eta\n":
            z = int(f.readline()[:-1])
            for im, si, sj, v in map(
                lambda v: (int(v[0]) - 1, int(v[1]) - 1, int(v[2]) - 1, float(v[3])),
                map(lambda _: f.readline().split(), range(z)),
            ):
                self._h[im][si, sj] += v
            line = f.readline()

        if line == "kappa\n":
            z = int(f.readline()[:-1])

            def b(_):
                v = f.readline().split()
                return (int(v[0]) - 1, int(v[1]) - 1, int(v[2]) - 1, float(v[3]))

            for im, s, i, v in map(b, range(z)):
                self._H_i[im][i, s, s] = v
            line = f.readline()

        if line == "lambda\n":
            z = int(f.readline()[:-1])

            def c(_):
                v = f.readline().split()
                return (int(v[0]) - 1, int(v[1]) - 1, int(v[2]) - 1, int(v[3]) - 1, float(v[4]))

            for im, si, sj, i, v in map(c, range(z)):
                self._H_i[im][i, si, sj] = v
                self._H_i[im][i, sj, si] = v
            line = f.readline()

        if line == "lambda_soc\n":
            z = int(f.readline()[:-1])
            self._lambda_soc = np.zeros((nmstates, nmstates, r3N), dtype=complex)
            lambda_soc_real = True

            def c(_):
                v = f.readline().split()
                return (int(v[0]) - 1, int(v[1]) - 1, int(v[2]) - 1, (1j if v[3] == "I" else 1.0) * float(v[4]))

            for si, sj, i, v in map(c, range(z)):
                if lambda_soc_real and isinstance(v, complex):
                    lambda_soc_real = False

                self._lambda_soc[si, sj, i] += v
                self._lambda_soc[sj, si, i] += v
            line = f.readline()

        if line == "gamma\n":
            z = int(f.readline()[:-1])
            self._gammas = z != 0
            self.log.info(f"including Gammas in calculation: {self._gammas}")
            self.log.debug(f"gammas: {self._gammas}")

            def d(_):
                v = f.readline().split()
                return (int(v[0]) - 1, int(v[1]) - 1, int(v[2]) - 1, int(v[3]) - 1, int(v[4]) - 1, float(v[5]))

            for im, si, sj, n, m, v in map(d, range(z)):
                self._G[im][sj, si, n, m] = v
                self._G[im][si, sj, n, m] = v
            line = f.readline()

        while line:
            if not line.split():
                line = f.readline()
                continue
            factor = 1j if line.split()[-1] == "I" else 1
            if "SOC" in line:
                if factor != 1:
                    soc_real = False
                line = f.readline()
                i = 0
                self.log.debug(f"Reading SOC {factor}")
                while i < nmstates:
                    self._soc[i, :] += np.asarray(line.split(), dtype=float) * factor
                    i += 1
                    line = f.readline()
            elif line[:2] == "DM":
                j = xyz[line[2]]
                if factor != 1:
                    dipole_real = False
                line = f.readline()
                i = 0
                while i < nmstates:
                    self._dipole[j, i, :] += np.asarray(line.split(), dtype=float) * factor
                    i += 1
                    line = f.readline()
            elif "Multipolar Density Fit" in line:
                line = f.readline()
                n_fits = int(line)
                self.log.debug(f"multipolar_fit {n_fits}")
                self._fits = {im: np.zeros((n, n, natom, 10), dtype=float) for im, n in enumerate(states) if n != 0}

                def d(_):
                    v = f.readline().split()
                    return (int(v[0]) - 1, int(v[1]) - 1, int(v[2]) - 1, int(v[3]), v[4:])

                for im, si, sj, i, v in map(d, range(n_fits)):
                    if i >= natom:
                        continue
                    n = len(v)
                    dens = np.array([float(x) for x in v[:n]])
                    self._fits[im][si, sj, i, :n] = dens
                    self._fits[im][sj, si, i, :n] = dens
            else:
                line = f.readline()
        f.close()
        # setting type as necessary (converting type through view and reshape is a lot faster that simple astype
        # assignemnt)
        if soc_real:
            self._soc = np.reshape(self._soc.view(float), self._soc.shape + (2,))[:, :, 0]
        if "_lambda_soc" in self.__dict__ and lambda_soc_real:
            self._lambda_soc = self._lambda_soc.real.copy()
        if dipole_real:
            self._dipole = np.reshape(self._dipole.view(float), self._dipole.shape + (2,))[:, :, :, 0]

        # if self.QMin.save["init"]:
        # SHARC_FAST.checkscratch(self.QMin.save["savedir"])
        self._read_template = True
        return

    # TODO: rework to buffer files
    def read_V0(self, filename):
        QMin = self.QMin
        lines = readfile(filename)
        it = 1
        elem = QMin.molecule["elements"]
        rM = list(
            map(lambda x: [x[0]] + [float(y) for y in x[2:]], map(lambda x: x.split(), lines[it : it + QMin.molecule["natom"]]))
        )
        v0_elem = [x[0] for x in rM]
        if v0_elem != elem:
            raise ValueError(f"inconsistent atom labels in QM.in and {filename}:\n{elem}\n{v0_elem}")
        rM = np.asarray([x[1:] for x in rM], dtype=float)
        self._ref_coords = rM[:, :-1]
        self._masses = rM[:, -1]
        tmp = np.sqrt(rM[:, -1] * U_TO_AMU)
        self._Msa = np.asarray([tmp, tmp, tmp]).flatten(order="F")
        it += QMin.molecule["natom"] + 1
        self._Om = np.asarray(lines[it].split(), dtype=float)
        it += 2
        self._Km = np.asarray([x.split() for x in lines[it:]], dtype=float).T * self._Msa
        return

    def read_resources(self, resources_filename="LVC.resources"):
        if not os.path.isfile(resources_filename):
            self.log.warning("LVC.resources not found; continuing without further settings.")
            self._read_resources = True
            return

        super().read_resources(resources_filename)
        self._do_kabsch = self.QMin.resources["do_kabsch"]
        if "diagonalize" in self.QMin.resources:
            self._diagonalize = self.QMin.resources["diagonalize"]

    def setup_interface(self):
        super().setup_interface()
        if self.persistent:
            for file in os.listdir(self.QMin.save["savedir"]):
                if re.match(r"^U\.npy\.\d+$", file):
                    step = int(file.split('.')[-1])
                    ufile = os.path.join(self.QMin.save["savedir"], file)
                    self.savedict[step] = {'U': np.load(ufile).reshape( (self.QMin.molecule['nmstates'], self.QMin.molecule['nmstates']) )}
                

    def getQMout(self):
        return self.QMout

    @staticmethod
    @njit(cache=True, fastmath=True)
    def get_mult_prefactors(pc_coord_diff, pc_inv_dist_A_B, r_inv3, r_inv5):
        # precalculated dist matrix
        # pc_inv_dist_A_B = 1 / np.sqrt(np.sum((pc_coord_diff) ** 2, axis=0))  # distance matrix n_coord (A), n_pc (B)
        R = pc_coord_diff
        # r_inv3 = pc_inv_dist_A_B**3
        # r_inv5 = pc_inv_dist_A_B**5
        # full stack of factors for the multipole expansion
        # .,   x, y, z,   xx, yy, zz, xy, xz, yz
        return np.stack(
            (
                pc_inv_dist_A_B,  # .
                R[0, ...] * r_inv3,  # x
                R[1, ...] * r_inv3,  # y
                R[2, ...] * r_inv3,  # z
                R[0, ...] * R[0, ...] * r_inv5 * 0.5,  # xx
                R[1, ...] * R[1, ...] * r_inv5 * 0.5,  # yy
                R[2, ...] * R[2, ...] * r_inv5 * 0.5,  # zz
                R[0, ...] * R[1, ...] * r_inv5,  # xy
                R[0, ...] * R[2, ...] * r_inv5,  # xz
                R[1, ...] * R[2, ...] * r_inv5,  # yz
            )
        )

    @staticmethod
    @njit(cache=True, fastmath=True)
    def get_mult_prefactors_deriv(pc_coord_diff, pc_inv_dist_A_B, r_inv3, r_inv5):
        # pc_inv_dist_A_B = 1 / np.sqrt(np.sum((pc_coord_diff) ** 2, axis=0))  # distance matrix n_coord (A), n_pc (B)
        R = pc_coord_diff
        # r_inv3 = pc_inv_dist_A_B**3
        # r_inv5 = pc_inv_dist_A_B**5
        r_inv7 = pc_inv_dist_A_B**7
        R_sq = R**2
        # full stack of factors for the multipole expansion
        # order 0,   x, y, z,   xx, yy, zz, xy, xz, yz
        return np.stack(
            (  # derivatives in x direction
                -R[0, ...] * r_inv3,  # -Rx/R^3
                (-2 * R_sq[0, ...] + R_sq[1, ...] + R_sq[2, ...]) * r_inv5,  # (-2Rx2+Ry2+Rz2)/R5
                -3 * R[1, ...] * R[0, ...] * r_inv5,  # -3RyRx/R5
                -3 * R[2, ...] * R[0, ...] * r_inv5,  # -3RzRx/R5
                -R[0, ...] * (1.5 * R_sq[0, ...] - (R_sq[1, ...] + R_sq[2, ...])) * r_inv7,  # -Rx(5Rx2-2R2)/2R7
                -2.5 * R_sq[1, ...] * R[0, ...] * r_inv7,  # -5Ry2Rx/2R7
                -2.5 * R_sq[2, ...] * R[0, ...] * r_inv7,  # -5Rz2Rx/2R7
                R[1, ...] * (-4 * R_sq[0, ...] + R_sq[1, ...] + R_sq[2, ...]) * r_inv7,  # Ry(-4Rx2+R2)/R7
                R[2, ...] * (-4 * R_sq[0, ...] + R_sq[1, ...] + R_sq[2, ...]) * r_inv7,  # Rz(-4Rx2+R2)/R7
                -5 * R[0, ...] * R[1, ...] * R[2, ...] * r_inv7,  # -5RxRyRz/R7
                # derivatives in y direction
                -R[1, ...] * r_inv3,  # -Ry/R^3
                -3 * R[0, ...] * R[1, ...] * r_inv5,  # -3RxRy/R5
                (-2 * R_sq[1, ...] + R_sq[0, ...] + R_sq[2, ...]) * r_inv5,  # (-2Ry2+Rx2+Rz2)/R5
                -3 * R[2, ...] * R[1, ...] * r_inv5,  # -3RzRy/R5
                -2.5 * R_sq[0, ...] * R[1, ...] * r_inv7,  # -5Rx2Ry/2R7
                -R[1, ...] * (1.5 * R_sq[1, ...] - (R_sq[0, ...] + R_sq[2, ...])) * r_inv7,  # -Ry(5Ry2-2R2)/2R7
                -2.5 * R_sq[2, ...] * R[1, ...] * r_inv7,  # -5Rz2Ry/2R7
                R[0, ...] * (-4 * R_sq[1, ...] + R_sq[0, ...] + R_sq[2, ...]) * r_inv7,  # Rx(-4Ry2+R2)/R7
                -5 * R[0, ...] * R[1, ...] * R[2, ...] * r_inv7,  # -5RxRyRz/R7
                R[2, ...] * (-4 * R_sq[1, ...] + R_sq[0, ...] + R_sq[2, ...]) * r_inv7,  # Rz(-4Ry2+R2)/R7
                # derivatives in z direction
                -R[2, ...] * r_inv3,  # -Rz/R^3
                -3 * R[0, ...] * R[2, ...] * r_inv5,  # -3RxRz/R5
                -3 * R[1, ...] * R[2, ...] * r_inv5,  # -3RyRz/R5
                (-2 * R_sq[2, ...] + R_sq[1, ...] + R_sq[0, ...]) * r_inv5,  # (-2Rz2+Rx2+Rz2)/R5
                -2.5 * R_sq[0, ...] * R[2, ...] * r_inv7,  # -5Rx2Rz/2R7
                -2.5 * R_sq[1, ...] * R[2, ...] * r_inv7,  # -5Ry2Rz/2R7
                -R[2, ...] * (1.5 * R_sq[2, ...] - (R_sq[1, ...] + R_sq[0, ...])) * r_inv7,  # -Rz(5Rz2-2R2)/2R7
                -5 * R[0, ...] * R[1, ...] * R[2, ...] * r_inv7,  # -5RxRyRz/R7
                R[0, ...] * (-4 * R_sq[2, ...] + R_sq[1, ...] + R_sq[0, ...]) * r_inv7,  # Rx(-5Rz2+R2)/R7
                R[1, ...] * (-4 * R_sq[2, ...] + R_sq[1, ...] + R_sq[0, ...]) * r_inv7,  # Ry(-5Rz2+R2)/R7
            )
        ).reshape((3, 10, pc_coord_diff.shape[1], pc_coord_diff.shape[2]))

    @staticmethod
    def rotate_multipoles(q, Trot):
        dip = q[..., 1:4].copy()
        # res = np.zeros_like(q)
        dip = dip @ Trot
        quad = np.zeros((*q.shape[:-1], 3, 3))
        # [0,1,2,0,0,1],[0,1,2,1,2,2]
        quad[..., [0, 1, 2], [0, 1, 2]] = q[..., 4:7]
        quad[..., [0, 0, 1], [1, 2, 2]] = 0.5 * q[..., 7:]
        quad[..., [1, 2, 2], [0, 0, 1]] = quad[..., [0, 0, 1], [1, 2, 2]]
        quad = Trot.T @ quad @ Trot
        return np.concatenate(
            (q[..., 0, None], dip, quad[..., [0, 1, 2], [0, 1, 2]], 2 * quad[..., [0, 0, 1], [1, 2, 2]]), axis=-1
        )

    @staticmethod
    def rotate_multipoles_deriv(q, Trot, dTrot):
        dip = q[..., 1:4].copy()
        # res = np.zeros_like(q)
        dip = np.einsum("ijax,kxy->kijay", dip, dTrot, casting="no", optimize=True)
        quad = np.zeros((*q.shape[:-1], 3, 3))
        # [0,1,2,0,0,1],[0,1,2,1,2,2]
        quad[..., [0, 1, 2], [0, 1, 2]] = q[..., 4:7]
        quad[..., [0, 0, 1], [1, 2, 2]] = 0.5 * q[..., 7:]
        quad[..., [1, 2, 2], [0, 0, 1]] = quad[..., [0, 0, 1], [1, 2, 2]]
        quad = np.einsum(
            "kxm,ijaxy,yn->kijamn", dTrot, quad, Trot, casting="no", optimize=["einsum_path", (1, 2), (0, 1)]
        ) + np.einsum("xm,ijaxy,kyn->kijamn", Trot, quad, dTrot, casting="no", optimize=["einsum_path", (0, 1), (0, 1)])
        return np.concatenate((dip, quad[..., [0, 1, 2], [0, 1, 2]], 2 * quad[..., [0, 0, 1], [1, 2, 2]]), axis=-1)

    def run(self):
        do_pc = self.QMin.molecule["point_charges"]
        do_derivs = self.QMin.requests["grad"] or self.QMin.requests["nacdr"]

        weights = self._masses
        # NOTE: do not calculate all nacs and grads only requested!!
        req_nmstates = self.QMin.molecule["nmstates"]
        req_states = self.QMin.molecule["states"]
        nmstates = self.parsed_states["nmstates"]
        states = self.parsed_states["states"]

        # conditionally turn on kabsch as flag (do_pc for additional logic)
        do_kabsch = True if do_pc else self._do_kabsch
        self.clock.starttime = datetime.datetime.now()
        self._U = np.zeros((nmstates, req_nmstates), dtype=float)
        Hd = np.zeros((req_nmstates, req_nmstates), dtype=float)
        natom = self.QMin.molecule["natom"]
        r3N = 3 * self.QMin.molecule["natom"]
        coords: np.ndarray = self.QMin.coords["coords"].copy()
        coords_ref_basis = coords
        if do_kabsch:
            if do_derivs:
                self._Trot, self._com_ref, self._com_coords, dTrot = kabsch_w_with_deriv(self._ref_coords, coords, weights)
                dc = np.eye(r3N).reshape((natom, 3, natom, 3))
                dcom = np.einsum("b,axby->xby", weights, dc) / sum(weights)
                dcom = dcom.reshape((3, r3N))
                dc = dc.reshape(natom, 3, r3N)
                coords_deriv = np.einsum("ax,kyx->kay", (coords - self._com_coords), dTrot) + np.einsum(
                    "axk,yx->kay", (dc - dcom[None, ...]), self._Trot
                )
                coords_deriv = coords_deriv.reshape((r3N, r3N))
            else:
                self._Trot, self._com_ref, self._com_coords = kabsch(self._ref_coords, coords, weights)
            coords_ref_basis = (coords - self._com_coords) @ self._Trot.T + self._com_ref
        # sanity check for coordinates - check if centre of mass is conserved
        elif self.QMin.save["step"] == 0:
            self._Trot, self._com_ref, self._com_coords = kabsch(self._ref_coords, coords, weights)
            if not np.allclose(self._com_ref, self._com_coords, rtol=1e-3) or not np.allclose(
                np.diag(self._Trot), np.ones(3, dtype=float), rtol=1e-5
            ):
                raise RuntimeError(
                    "Misaligned geometry without activated Kabsch algorithm! -> check you input structure or activate Kabsch"
                )

        # kabsch is necessary with point charges
        if do_pc:
            fits_rot = {im: self.rotate_multipoles(fits, self._Trot) for im, fits in self._fits.items()}
            self.pc_chrg = np.array(self.QMin.coords["pccharge"])  # n_pc, 1
            pc_coord = np.array(self.QMin.coords["pccoords"])  # n_pc, 1

            n_pc = pc_coord.shape[0]
            # matrix of position differences (for gradient calc) 3, n_coord (A), n_pc (B)
            pc_coord_diff = np.full((3, coords.shape[0], n_pc), coords.T[:, :, None]) - pc_coord.T[:, None, :]
            pc_inv_dist_A_B = 1 / np.sqrt(np.sum((pc_coord_diff) ** 2, axis=0))  # distance matrix n_coord (A), n_pc (B)
            r_inv3 = pc_inv_dist_A_B**3
            r_inv5 = pc_inv_dist_A_B**5
            mult_prefactors = self.get_mult_prefactors(pc_coord_diff, pc_inv_dist_A_B, r_inv3, r_inv5)
            mult_prefactors_pc = np.einsum("b,yab->ay", self.pc_chrg, mult_prefactors)
            del mult_prefactors

        # Build full H and diagonalize
        self._Q = np.sqrt(self._Om) * (self._Km @ (coords_ref_basis.flatten() - self._ref_coords.flatten()))
        self._V = self._Om * self._Q
        V0 = 0.5 * (self._V @ self._Q)
        start = 0  # starting index for blocks
        start_req = 0  # starting index for blocks
        # TODO what if I want to get gradients only ? i.e. samestep
        for im, n_req in filter(lambda x: x[1] != 0, enumerate(req_states)):
            n = states[im]
            stop = start + n
            stop_req = start_req + n_req

            H = self._h[im] + np.identity(n) * V0
            H += np.einsum("kij, k->ij", self._H_i[im], self._Q)
            if self._gammas:
                H += np.einsum("n,ijnm,m->ij", self._Q, self._G[im], self._Q, casting="no", optimize=True)
            if do_pc:
                H += np.einsum("ijay,ay->ij", fits_rot[im], mult_prefactors_pc, casting="no", optimize=True)
            if self._diagonalize:
                eigen_values, eigen_vec = np.linalg.eigh(H, UPLO="U")
                np.einsum("ii->i", Hd)[start_req:stop_req] = eigen_values[:n_req]
            else:
                eigen_vec = np.identity(n, dtype=float)
                Hd[start_req:stop_req, start_req:stop_req] = H[:n_req]
            self._U[start:stop, start_req:stop_req] = eigen_vec[:, :n_req]

            for x in range(im):  # fills in blocks for other magnetic quantum numbers
                s1_req = start_req + n_req * (x + 1)
                s2_req = s1_req + n_req
                s1 = start + n * (x + 1)
                s2 = s1 + n
                self._U[s1:s2, s1_req:s2_req] += self._U[start:stop, start_req:stop_req]
                if self._diagonalize:
                    np.einsum("ii->i", Hd)[s1_req:s2_req] = eigen_values[:n_req]
                else:
                    Hd[s1_req:s2_req, s1_req:s2_req] = H[:n_req, :n_req]

            start += n * (im + 1)
            start_req += n_req * (im + 1)

        if do_kabsch:
            if do_pc and do_derivs:
                pc_grad = np.zeros((self.pc_chrg.shape[0] * 3, req_nmstates))

                mult_prefactors_deriv = self.get_mult_prefactors_deriv(pc_coord_diff, pc_inv_dist_A_B, r_inv3, r_inv5)

                fits_deriv = {}
                for im, n in filter(lambda x: x[1] != 0, enumerate(states)):
                    fits_deriv[im] = self.rotate_multipoles_deriv(self._fits[im], self._Trot, dTrot)

                mult_prefactors_deriv_pc = np.einsum("xyab,b->abxy", mult_prefactors_deriv, self.pc_chrg)
                del mult_prefactors_deriv

            if do_derivs:
                dQ_dr = np.sqrt(self._Om)[..., None] * (self._Km @ coords_deriv.T)
                dQ_dr = dQ_dr.T
                del coords_deriv

            # rotate and diagonalize the fits already to decrease comp effort in case of less adia states
            if do_pc and (do_derivs or self.QMin.requests["multipolar_fit"]):
                if self._diagonalize:
                    dfits = {}
                    dfits_deriv = {}
                    start_req = 0
                    start = 0
                    for im, n_req in filter(lambda x: x[1] != 0, enumerate(req_states)):
                        n = states[im]
                        stop_req = start_req + n_req
                        stop = start + n
                        u = self._U[start:stop, start_req:stop_req]
                        dfits[im] = np.einsum("ijay,in,jm->mnay", fits_rot[im], u, u, optimize=True)
                        if do_derivs:
                            dfits_deriv[im] = np.einsum("kijay,in,jm->kmnay", fits_deriv[im], u, u, optimize=True)
                        start += n * (im + 1)
                        start_req += n_req * (im + 1)
                else:
                    dfits = {}
                    dfits_deriv = {}
                    for im, n_req in filter(lambda x: x[1] != 0, enumerate(req_states)):
                        dfits[im] = fits_rot[im][..., :n_req, :n_req]
                        if do_derivs:
                            dfits_deriv[im] = fits_deriv[im][:, :n_req, :n_req, ...]

        elif do_derivs:
            dQ_dr = np.sqrt(self._Om)[None, ...] * self._Km.T
        elif self.QMin.requests["multipolar_fit"]:
            dfits = self._fits

        # GRADS and NACS
        if self.QMin.requests["nacdr"]:
            self.log.debug("start calculating nacdr")
            # Build full derivative matrix
            start = 0  # starting index for blocks
            start_req = 0  # starting index for blocks
            nacdr = np.zeros((r3N, req_nmstates, req_nmstates), float)
            grad = np.zeros((r3N, req_nmstates))
            if do_pc:
                nacdr_pc = np.zeros((n_pc * 3, req_nmstates, req_nmstates), float)

            for im, n_req in filter(lambda x: x[1] != 0, enumerate(req_states)):
                n = states[im]
                stop = start + n
                stop_req = start_req + n_req

                u = self._U[start:stop, start_req:stop_req]
                dlvc = np.zeros((r3N, n*n))
                dlvc[:, ::n+1] += self._V[:, None]
                dlvc = dlvc.reshape((r3N, n, n))
                # np.einsum("kii->ki", dlvc)[...] += self._V[
                    # ..., None
                # ]  # fills diagonal on matrix with shape (r3N,nmstates,nmstates)
                dlvc += self._H_i[im]
                dlvc = np.einsum("kij,lk->lij", dlvc, dQ_dr, casting="no", optimize=True)
                if self._gammas:
                    dlvc += np.einsum("n,ijnm,lm->lij", self._Q, self._G[im], dQ_dr, casting="no", optimize=True)
                if self._diagonalize:
                    # dlvc = u.T @ dlvc @ u
                    dlvc = np.einsum("kij,im,jn->kmn", dlvc, u, u, casting="no", optimize=["einsum_path", (0, 1), (0, 1)])
                else:
                    dlvc = dlvc[..., :n_req, :n_req]

                if do_pc:
                    # calculate derivative of electrostic interaction
                    if "_dcoulomb_path" not in self.__dict__:
                        self._dcoulomb_path = np.einsum_path(
                            "abxy,ijay->abxij", mult_prefactors_deriv_pc, dfits[im], optimize="optimal"
                        )[0]
                    dcoulomb: np.ndarray = np.einsum(
                        "abxy,ijay->abxij", mult_prefactors_deriv_pc, dfits[im], optimize=self._dcoulomb_path
                    )
                    # add derivative to lvc derivative summed ofe all point charges
                    dlvc += np.einsum("abxij->axij", dcoulomb).reshape((r3N, n_req, n_req))
                    # add the derivative of the multipoles
                    dlvc += np.einsum("ay,kijay->kij", mult_prefactors_pc[..., 1:], dfits_deriv[im], casting="no", optimize=True)
                    # calculate the pc derivatives
                    pc_derivative = -np.einsum("abxij->bxij", dcoulomb).reshape((-1, n_req, n_req))
                    del dcoulomb

                # transform gradients to adiabatic basis
                eV = Hd.flat[:: req_nmstates + 1]
                # energy weighting of the nacs
                tmp = np.full((n_req, n_req), eV[start_req:stop_req]).T
                tmp -= eV[start_req:stop_req]
                idx = tmp != float(0)
                tmp[idx] **= -1

                nacdr[:, start_req:stop_req, start_req:stop_req] = np.einsum(
                    "ji,kij->kij", tmp, dlvc, casting="no", optimize=True
                )
                grad[..., start_req:stop_req] = np.einsum("kii->ki", dlvc)
                if do_pc:
                    pc_grad[..., start_req:stop_req] = np.einsum("kii->ki", pc_derivative)
                    nacdr_pc[..., start_req:stop_req, start_req:stop_req] = np.einsum(
                        "ji,kij->kij", tmp, pc_derivative, casting="no", optimize=True
                    )
                # fills in blocks for other magnetic quantum numbers
                for s1 in map(lambda x: start_req + n_req * (x + 1), range(im)):
                    s2 = s1 + n_req
                    nacdr[:, s1:s2, s1:s2] = nacdr[:, start_req:stop_req, start_req:stop_req]
                    grad[..., s1:s2] = grad[..., start_req:stop_req]
                    if do_pc:
                        nacdr_pc[:, s1:s2, s1:s2] = nacdr_pc[:, start_req:stop_req, start_req:stop_req]
                        pc_grad[..., s1:s2] = pc_grad[..., start_req:stop_req]
                start += n * (im + 1)
                start_req += n_req * (im + 1)

        # calculate only gradients
        if self.QMin.requests["grad"] and not self.QMin.requests["nacdr"]:
            self.log.debug("start calculating gradients")
            grad = np.zeros((r3N, req_nmstates))
            start = 0
            start_req = 0
            for im, n_req in filter(lambda x: x[1] != 0, enumerate(req_states)):
                n = states[im]
                stop = start + n
                stop_req = start_req + n_req

                u = self._U[start:stop, start_req:stop_req]
                grad_lvc = np.full((r3N, n_req), self._V[..., None])
                if self._diagonalize:
                    if self._gammas:
                        h_i = self._H_i[im] + np.einsum("m,ijnm->nij", self._Q, self._G[im], casting="no", optimize=True)
                        grad_lvc += np.einsum("kij,in,jn->kn", h_i, u, u, casting="no", optimize=True)
                    else:
                        grad_lvc += np.einsum("kij,in,jn->kn", self._H_i[im], u, u, casting="no", optimize=True)
                else:
                    grad_lvc += np.einsum("kii->ki", self._H_i[im][:, :n_req, :n_req])
                    if self._gammas:
                        grad_lvc += np.einsum("n,iinm->mi", self._Q, self._G[im], casting="no", optimize=True)
                if do_pc:
                    idfits = np.einsum("iiay->iay", dfits[im])
                    idfits_deriv = np.einsum("kiiay->kiay", dfits_deriv[im])
                grad_lvc = np.einsum("lk,ki->li", dQ_dr, grad_lvc)
                if do_pc:
                    # calculate derivative of electrostic interaction
                    if "_dcoulomb_grad_path" not in self.__dict__:
                        self._dcoulomb_grad_path = np.einsum_path(
                            "abxy,iay->abxi", mult_prefactors_deriv_pc, idfits, optimize="optimal"
                        )[0]
                    dcoulomb: np.ndarray = np.einsum(
                        "abxy,iay->abxi", mult_prefactors_deriv_pc, idfits, casting="no", optimize=self._dcoulomb_grad_path
                    )
                    # add derivative to lvc derivative summed ofe all point charges
                    grad_lvc += np.einsum("abxi->axi", dcoulomb).reshape((r3N, n_req))
                    # add the derivative of the multipoles
                    grad_lvc += np.einsum("ay,kiay->ki", mult_prefactors_pc[..., 1:], idfits_deriv, casting="no", optimize=True)
                    # calculate the pc derivatives
                    pc_grad[..., start_req:stop_req] = -np.einsum("abxi->bxi", dcoulomb).reshape((-1, n_req))
                    del dcoulomb
                grad[..., start_req:stop_req] += grad_lvc
                # fills in blocks for other magnetic quantum numbers
                for s1 in map(lambda x: start_req + n_req * (x + 1), range(im)):
                    s2 = s1 + n_req
                    grad[..., s1:s2] += grad_lvc
                    if do_pc:
                        pc_grad[..., s1:s2] = pc_grad[..., start_req:stop_req]
                start += n * (im + 1)
                start_req += n_req * (im + 1)

        if self.QMin.requests["overlap"] or self.QMin.requests["phases"]:
            if self.QMin.save["step"] == 0:
                pass
            elif self.persistent:
                Uold = self.savedict[self.QMin.save['step']-1]["U"]
            else:
                Uold = np.load(os.path.join(self.QMin.save["savedir"], f"U.npy.{self.QMin.save['step']-1}")).reshape(self._U.shape)
            overlap = Uold.T @ self._U
            if self.QMin.requests["phases"]:
                _, phases = phase_correction(overlap)

        # OVERLAP
        if not self.QMin.save["samestep"]:
            # store U matrix
            if self.persistent:
                self.savedict[self.QMin.save['step']] = {'U': np.copy(self._U)}
                # self.savedict["last_step"] = self.QMin.save['step']
            else:
                with open(os.path.join(self.QMin.save["savedir"], f"U.npy.{self.QMin.save['step']}"), 'wb') as f:
                    np.save(f, self._U)  # writes a binary file (can be read with numpy.load())
            
            # keep all U matrices 
            # TODO: could be removed because is done by retain mechanism
            if self.QMin.resources["keep_U"]:
                if "all_U" not in self.__dict__:
                    self.all_U = []
                self.all_U.append(self._U)

        # ========================== Prepare results ========================================
        if self.QMin.requests["soc"]:

            if "_lambda_soc" in self.__dict__:
                self.log.debug("adding linear derivatives of soc")
                soc = np.einsum("ijk,k->ij", self._lambda_soc, self._Q)
                soc = np.add(self._soc, soc, casting='safe')
                adia_soc = np.einsum('in,ij,jm->nm', self._U, soc, self._U, casting='safe', optimize=["einsum_path", (0,1),(0,1)])
                self.log.debug(f"soc sanity check: {adia_soc.dtype} {self._lambda_soc.dtype}")
            else:
                adia_soc = np.einsum('in,ij,jm->nm', self._U, self._soc, self._U, casting='safe', optimize=["einsum_path", (0,1),(0,1)])
                self.log.debug(f"soc sanity check: {adia_soc.dtype} {self._soc.dtype}")

            if adia_soc.dtype == complex:
                Hd = Hd.astype(complex)
            Hd += adia_soc

        dipole = (
            np.einsum("in,kij,jm->knm", self._U, self._dipole, self._U, casting="no", optimize=True)
            if self._diagonalize
            else self._dipole
        )

        if do_kabsch:
            dipole = np.einsum("inm,ij->jnm", dipole, self._Trot)

        if self.QMin.requests["grad"]:
            grad = grad.T.reshape((req_nmstates, self.QMin.molecule["natom"], 3))
        if self.QMin.requests["nacdr"]:
            nacdr = np.einsum("kij->ijk", nacdr).reshape((req_nmstates, req_nmstates, self.QMin.molecule["natom"], 3))

        if self.QMin.requests["multipolar_fit"]:
            multipolar_fit = {}
            for s1, s2 in self.QMin.requests["multipolar_fit"]:
                im = s1.S
                multipolar_fit[(s1, s2)] = dfits[im][s1.N - 1, s2.N - 1, ...]

        # ======================================== assign to QMout =========================================
        self.log.debug(f"requests: {self.QMin.requests}")
        self.QMout.states = req_states
        self.QMout.nstates = self.QMin.molecule["nstates"]
        self.QMout.nmstates = self.QMin.molecule["nmstates"]
        self.QMout.natom = self.QMin.molecule["natom"]
        self.QMout.npc = self.QMin.molecule["npc"]
        self.QMout.point_charges = do_pc
        self.QMout.h = Hd
        self.QMout.dm = dipole
        if self.QMin.requests["overlap"]:
            self.QMout.overlap = overlap
        if self.QMin.requests["phases"]:
            self.QMout.phases = phases
        if self.QMin.requests["grad"]:
            self.QMout.grad = grad
        if self.QMin.requests["nacdr"]:
            self.QMout.nacdr = nacdr
            if do_pc:
                self.QMout.nacdr_pc = np.einsum("kij->ijk", nacdr_pc).reshape((req_nmstates, req_nmstates, -1, 3))
        if do_pc and do_derivs:
            self.QMout.grad_pc = pc_grad.T.reshape((req_nmstates, -1, 3))
        if self.QMin.requests["multipolar_fit"]:
            self.QMout.multipolar_fit = multipolar_fit

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
            "soc",
            "dm",
            "grad",
            "nacdr",
            "overlap",
            "multipolar_fit",
            "point_charges",
            "phases",
        }

    def get_infos(self, INFOS: dict, KEYSTROKES: TextIOWrapper = None) -> dict:
        self.log.info("=" * 80)
        self.log.info(f"{'||':<78}||")
        self.log.info(f"||{'LVC interface setup': ^76}||\n{'||':<78}||")
        self.log.info("=" * 80)
        self.log.info("\n")

        if os.path.isfile('LVC.template'):
            default = 'LVC.template'
        else:
            default = None
        self.template_file = question("Specify path to LVC.template", str, KEYSTROKES=KEYSTROKES, autocomplete=True, default=default)
        while not os.path.isfile(self.template_file):
            self.template_file = question(
                f"'{self.template_file}' not found!\nSpecify path to LVC.template", str, KEYSTROKES=KEYSTROKES, autocomplete=True
            )

        # Check template for Soc and multipoles and states
        soc_found = False
        mfit_found = False
        dm_found = False
        with open(self.template_file, "r") as f:
            for line in f:
                if "SOC" in line or "lambda_soc" in line:
                    soc_found = True
                if "DM" in line:
                    dm_found = True
                if "Multipolar Density Fit" in line:
                    mfit_found = True
        if "soc" in INFOS["needed_requests"] and not soc_found:
            self.log.error(f"Requested SOC calculation but 'SOC' keyword not found in {self.template_file}")
            raise RuntimeError()

        if ("multipolar_fit" in INFOS["needed_requests"] or "point_charges" in INFOS["needed_requests"]) and not mfit_found:
            self.log.error(
                f"Calculation with 'point_charges' and/or 'multipolar_fit' requested but 'Multipolar Density Fit' not found in {self.template_file}"
            )
            raise RuntimeError()

        if "dm" in INFOS["needed_requests"] and not dm_found:
            self.log.error(f"Calculation of dipole moment requested but 'DM' keyword not found in {self.template_file}")
            raise RuntimeError()

        if question("Do you have an LVC.resources file?", bool, KEYSTROKES=KEYSTROKES, autocomplete=False, default=False):
            self.resources_file = question("Specify path to LVC.resources", str, KEYSTROKES=KEYSTROKES, autocomplete=True)
        else:
            self.wants_kabsch = question("Do you want to use the Kabsch algorithm?", bool, KEYSTROKES=KEYSTROKES, default=True)

        return INFOS

    def dyson_orbitals_with_other(self, other):
        raise NotImplementedError()

    def prepare(self, INFOS: dict, dir_path: str):
        super().prepare(INFOS, dir_path)
        if not "resources_file" in self.__dict__ or not self.resources_file:
            if "wants_kabsch" in self.__dict__ and self.wants_kabsch:
                string = 'do_kabsch true\n'
                writefile(os.path.join(dir_path, self.name() + ".resources"), string)

if __name__ == "__main__":
    from logger import loglevel

    lvc = SHARC_LVC(loglevel=loglevel)
    lvc.main()
