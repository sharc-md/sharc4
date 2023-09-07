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

# internal
from SHARC_FAST import SHARC_FAST
from utils import readfile, mkdir, question, expand_path
import shutil
from io import TextIOWrapper
from constants import U_TO_AMU, MASSES
from kabsch import kabsch_w as kabsch

authors = 'Sebastian Mai and Severin Polonius'
version = '4.0'
versiondate = datetime.datetime(2023, 8, 25)

changelogstring = '''
'''
np.set_printoptions(linewidth=400, formatter={'float': lambda x: f'{x: 9.7}'})


class SHARC_LVC(SHARC_FAST):

    _read_resources = True
    _do_kabsch = False
    _diagonalize = True
    _step = 0


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
        return "Linear Vibronic Coupling model calculations"

    def read_template(self, template_filename='LVC.template'):

        f = open(os.path.abspath(template_filename), 'r')
        V0file = f.readline()[:-1]
        self.read_V0(os.path.abspath(V0file))
        self.parsed_states = self.parseStates(f.readline())
        self.template_states = self.parsed_states['states']
        states = self.QMin.molecule['states']

        if (len(states) > len(self.template_states)) or any(a > b for (a, b) in zip(states, self.template_states)):
            self.log.error(f'states from QM.in and nstates from LVC.template are inconsistent! {self.QMin.molecule["states"]} != {states}')
            raise ValueError(f'impossible to calculate {self.QMin.molecule["states"]} with template holding {self.template_states}')
        if any(a < b for (a, b) in zip(states, self.template_states)):
            self.log.warning(f"Calculating with {self.template_states} but returning {states}")

        natom = self.QMin.molecule['natom']
        r3N = 3 * natom
        nmstates = self.parsed_states['nmstates']
        states = self.parsed_states['states']

        self._H_i = {im: np.zeros((n, n, r3N), dtype=float) for im, n in enumerate(states) if n != 0}
        self._epsilon = {im: np.zeros(n, dtype=float) for im, n in enumerate(states) if n != 0}
        self._eV = {im: np.zeros(n, dtype=float) for im, n in enumerate(states) if n != 0}
        self._dipole = np.zeros((3, nmstates, nmstates), dtype=complex)
        self._soc = np.zeros((nmstates, nmstates), dtype=complex)
        self._U = np.zeros((nmstates, nmstates), dtype=float)
        self._Q = np.zeros(r3N, float)
        xyz = {'X': 0, 'Y': 1, 'Z': 2}
        soc_real = True
        dipole_real = True
        line = f.readline()
        # NOTE: possibly assign whole array with index accessor (numpy)
        if line == 'epsilon\n':
            z = int(f.readline()[:-1])

            def a(x):
                v = f.readline().split()
                return (int(v[0]) - 1, int(v[1]) - 1, float(v[2]))

            for im, s, v in map(a, range(z)):
                self._epsilon[im][s] += v
        if f.readline() == 'kappa\n':
            z = int(f.readline()[:-1])

            def b(_):
                v = f.readline().split()
                return (int(v[0]) - 1, int(v[1]) - 1, int(v[2]) - 1, float(v[3]))

            for im, s, i, v in map(b, range(z)):
                self._H_i[im][s, s, i] = v
        if f.readline() == 'lambda\n':
            z = int(f.readline()[:-1])

            def c(_):
                v = f.readline().split()
                return (int(v[0]) - 1, int(v[1]) - 1, int(v[2]) - 1, int(v[3]) - 1, float(v[4]))

            for im, si, sj, i, v in map(c, range(z)):
                self._H_i[im][si, sj, i] = v
                self._H_i[im][sj, si, i] = v
        line = f.readline()
        while len(line) != 0:
            factor = 1j if line[-2] == 'I' else 1
            if line[:3] == 'SOC':
                if factor != 1:
                    soc_real = False
                line = f.readline()
                i = 0
                while len(line.split()) == nmstates:
                    self._soc[i, :] += np.asarray(line.split(), dtype=float) * factor
                    i += 1
                    line = f.readline()
            elif line[:2] == 'DM':
                j = xyz[line[2]]
                if factor != 1:
                    dipole_real = False
                line = f.readline()
                i = 0
                while len(line.split()) == nmstates:
                    self._dipole[j, i, :] += np.asarray(line.split(), dtype=float) * factor
                    i += 1
                    line = f.readline()
            elif 'Multipolar Density Fit' in line:
                line = f.readline()
                n_fits = int(line)
                self._fits = {im: np.zeros((n, n, natom, 10), dtype=float) for im, n in enumerate(states) if n != 0}

                def d(_):
                    v = f.readline().split()
                    return (int(v[0]) - 1, int(v[1]) - 1, int(v[2]) - 1, int(v[3]), v[4:])

                for im, si, sj, i, v in map(d, range(n_fits)):
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
            self._soc = np.reshape(self._soc.view(float), self._soc.shape + (2, ))[:, :, 0]
        if dipole_real:
            self._dipole = np.reshape(self._dipole.view(float), self._dipole.shape + (2, ))[:, :, :, 0]

        if self.QMin.save['init']:
            SHARC_FAST.checkscratch(self.QMin.save['savedir'])
        self._read_template = True
        return

    # TODO: rework to buffer files
    def read_V0(self, filename):
        QMin = self.QMin
        lines = readfile(filename)
        it = 1
        elem = QMin.molecule['elements']
        rM = list(
            map(lambda x: [x[0]] + [float(y) for y in x[2:]], map(lambda x: x.split(), lines[it:it + QMin.molecule['natom']]))
        )
        v0_elem = [x[0] for x in rM]
        if v0_elem != elem:
            raise ValueError(f'inconsistent atom labels in QM.in and {filename}:\n{elem}\n{v0_elem}')
        rM = np.asarray([x[1:] for x in rM], dtype=float)
        self._ref_coords = rM[:, :-1]
        self._masses = rM[:, -1]
        tmp = np.sqrt(rM[:, -1] * U_TO_AMU)
        self._Msa = np.asarray([tmp, tmp, tmp]).flatten(order='F')
        it += QMin.molecule['natom'] + 1
        self._Om = np.asarray(lines[it].split(), dtype=float)
        it += 2
        self._Km = np.asarray([x.split() for x in lines[it:]], dtype=float).T * self._Msa
        return

    def read_resources(self, resources_filename="LVC.resources"):
        if not os.path.isfile(resources_filename):
            self.log.warning("LVC.resources not found; continuuing without further settings.")
            self._read_resources = True
            return

        super().read_resources(resources_filename)
        if "do_kabsch" in self.QMin.resources:
            self._do_kabsch = True
        #  if "diagonalize" in self.QMin.resources:
            #  self._diagonalize = True


    def setup_interface(self):
        if self.QMin.requests['overlap']:
            if self.QMin.save['step'] == 0:
                self.QMout.overlap = np.identity(self.parsed_states['nmstates'], dtype=float)
            else:
                self.log.debug(f'restarting: getting overlap from file {self.QMin.save["step"]-1}.out')
                self.QMout.overlap = np.fromfile(
                    os.path.join(self.QMin.save['savedir'], f'U_{self.QMin.save["step"]-1}.out'), dtype=float
                ).reshape(self._U.shape).T @ self._U

    def getQMout(self):
        return self.QMout

    @staticmethod
    def get_mult_prefactors(pc_coord_diff):
        # precalculated dist matrix
        pc_inv_dist_A_B = 1 / np.sqrt(np.sum((pc_coord_diff)**2, axis=2))    # distance matrix n_coord (A), n_pc (B)
        R = pc_coord_diff
        r_inv3 = pc_inv_dist_A_B**3
        r_inv5 = pc_inv_dist_A_B**5
        # full stack of factors for the multipole expansion
        # .,   x, y, z,   xx, yy, zz, xy, xz, yz
        return np.stack(
            (
                pc_inv_dist_A_B,    # .
                R[..., 0] * r_inv3,    # x
                R[..., 1] * r_inv3,    # y
                R[..., 2] * r_inv3,    # z
                R[..., 0] * R[..., 0] * r_inv5 * 0.5,    # xx
                R[..., 1] * R[..., 1] * r_inv5 * 0.5,    # yy
                R[..., 2] * R[..., 2] * r_inv5 * 0.5,    # zz
                R[..., 0] * R[..., 1] * r_inv5,    # xy
                R[..., 0] * R[..., 2] * r_inv5,    # xz
                R[..., 1] * R[..., 2] * r_inv5    # yz
            )
        )

    @staticmethod
    def get_mult_prefactors_deriv(pc_coord_diff):
        pc_inv_dist_A_B = 1 / np.sqrt(np.sum((pc_coord_diff)**2, axis=2))    # distance matrix n_coord (A), n_pc (B)
        R = pc_coord_diff
        r_inv3 = pc_inv_dist_A_B**3
        r_inv5 = pc_inv_dist_A_B**5
        r_inv7 = pc_inv_dist_A_B**7
        R_sq = R**2
        # full stack of factors for the multipole expansion
        # order 0,   x, y, z,   xx, yy, zz, xy, xz, yz
        return np.stack(
            (   # derivatives in x direction
                -R[..., 0] * r_inv3,  # -Rx/R^3
                (-2 * R_sq[..., 0] + R_sq[..., 1] + R_sq[..., 2]) * r_inv5,  # (-2Rx2+Ry2+Rz2)/R5
                -3 * R[..., 1] * R[..., 0] * r_inv5,  # -3RyRx/R5
                -3 * R[..., 2] * R[..., 0] * r_inv5,  # -3RzRx/R5
                -R[..., 0] * (1.5 * R_sq[..., 0] - (R_sq[..., 1] + R_sq[..., 2])) * r_inv7,  # -Rx(5Rx2-2R2)/2R7
                -2.5 * R_sq[..., 1] * R[..., 0] * r_inv7,  # -5Ry2Rx/2R7
                -2.5 * R_sq[..., 2] * R[..., 0] * r_inv7,  # -5Rz2Rx/2R7
                R[..., 1] * (-4 * R_sq[..., 0] + R_sq[..., 1] + R_sq[..., 2]) * r_inv7,  # Ry(-4Rx2+R2)/R7
                R[..., 2] * (-4 * R_sq[..., 0] + R_sq[..., 1] + R_sq[..., 2]) * r_inv7,  # Rz(-4Rx2+R2)/R7
                -5 * R[..., 0] * R[..., 1] * R[..., 2] * r_inv7,  # -5RxRyRz/R7
                # derivatives in y direction
                -R[..., 1] * r_inv3,  # -Ry/R^3
                -3 * R[..., 0] * R[..., 1] * r_inv5,  # -3RxRy/R5
                (-2 * R_sq[..., 1] + R_sq[..., 0] + R_sq[..., 2]) * r_inv5,  # (-2Ry2+Rx2+Rz2)/R5
                -3 * R[..., 2] * R[..., 1] * r_inv5,  # -3RzRy/R5
                -2.5 * R_sq[..., 0] * R[..., 1] * r_inv7,  # -5Rx2Ry/2R7
                -R[..., 1] * (1.5 * R_sq[..., 1] - (R_sq[..., 0] + R_sq[..., 2])) * r_inv7,  # -Ry(5Ry2-2R2)/2R7
                -2.5 * R_sq[..., 2] * R[..., 1] * r_inv7,  # -5Rz2Ry/2R7
                R[..., 0] * (-4 * R_sq[..., 1] + R_sq[..., 0] + R_sq[..., 2]) * r_inv7,  # Rx(-4Ry2+R2)/R7
                -5 * R[..., 0] * R[..., 1] * R[..., 2] * r_inv7,  # -5RxRyRz/R7
                R[..., 2] * (-4 * R_sq[..., 1] + R_sq[..., 0] + R_sq[..., 2]) * r_inv7,  # Rz(-4Ry2+R2)/R7
                # derivatives in z direction
                -R[..., 2] * r_inv3,  # -Rz/R^3
                -3 * R[..., 0] * R[..., 2] * r_inv5,  # -3RxRz/R5
                -3 * R[..., 1] * R[..., 2] * r_inv5,  # -3RyRz/R5
                (-2 * R_sq[..., 2] + R_sq[..., 1] + R_sq[..., 0]) * r_inv5,  # (-2Rz2+Rx2+Rz2)/R5
                -2.5 * R_sq[..., 0] * R[..., 2] * r_inv7,  # -5Rx2Rz/2R7
                -2.5 * R_sq[..., 1] * R[..., 2] * r_inv7,  # -5Ry2Rz/2R7
                -R[..., 2] * (1.5 * R_sq[..., 2] - (R_sq[..., 1] + R_sq[..., 0])) * r_inv7,  # -Rz(5Rz2-2R2)/2R7
                -5 * R[..., 0] * R[..., 1] * R[..., 2] * r_inv7,  # -5RxRyRz/R7
                R[..., 0] * (-4 * R_sq[..., 2] + R_sq[..., 1] + R_sq[..., 0]) * r_inv7,  # Rx(-5Rz2+R2)/R7
                R[..., 1] * (-4 * R_sq[..., 2] + R_sq[..., 1] + R_sq[..., 0]) * r_inv7,  # Ry(-5Rz2+R2)/R7
            )
        ).reshape((3, 10, pc_coord_diff.shape[0], pc_coord_diff.shape[1]))

    @staticmethod
    def rotate_multipoles(q, Trot):
        res = q.copy()
        res[..., 1:4] = res[..., 1:4] @ Trot
        quad = np.zeros((*[*res.shape][:-1], 3, 3))
        # [0,1,2,0,0,1],[0,1,2,1,2,2]
        quad[..., [0, 1, 2], [0, 1, 2]] = res[..., 4:7]
        quad[..., [0, 0, 1], [1, 2, 2]] = 0.5 * res[..., 7:]
        quad[..., [1, 2, 2], [0, 0, 1]] = quad[..., [0, 0, 1], [1, 2, 2]]
        quad = Trot.T @ quad @ Trot
        res[..., 4:7] = quad[..., [0, 1, 2], [0, 1, 2]]
        res[..., 7:] = 2 * quad[..., [0, 0, 1], [1, 2, 2]]
        return res

    def run(self):
        do_pc = self.QMin.molecule["point_charges"]
        weights = self._masses
        # NOTE: do not calculate all nacs and grads only requested!!
        req_nmstates = self.QMin.molecule['nmstates']
        req_states = self.QMin.molecule['states']
        nmstates = self.parsed_states['nmstates']
        states = self.parsed_states['states']

        # conditionally turn on kabsch as flag (do_pc for additional logic)
        do_kabsch = True if do_pc else self._do_kabsch
        self.clock.starttime = datetime.datetime.now()
        self._U = np.zeros((nmstates, nmstates), dtype=float)
        Hd = np.zeros((nmstates, nmstates), dtype=self._soc.dtype)
        r3N = 3 * self.QMin.molecule['natom']
        coords: np.ndarray = self.QMin.coords['coords'].copy()
        coords_ref_basis = coords
        if do_kabsch:
            self._Trot, self._com_ref, self._com_coords = kabsch(self._ref_coords, coords, weights)
            coords_ref_basis = (coords - self._com_coords) @ self._Trot.T + self._com_ref
        # sanity check for coordinates - check if centre of mass is conserved
        elif self.QMin.save['step'] == 0:
            self._Trot, self._com_ref, self._com_coords = kabsch(self._ref_coords, coords, weights)
            if not np.allclose(self._com_ref, self._com_coords, rtol=1e-3) or \
               not np.allclose(np.diag(self._Trot), np.ones(3, dtype=float), rtol=1e-5):
                raise RuntimeError('Misaligned geometry without activated Kabsch algorithm! -> check you input structure or activate Kabsch')

        # kabsch is necessary with point charges
        if do_pc:
            # weights = [MASSES[i] for i in self.QMin['elements']]
            # self._Trot, self._com_ref, self._com_coords = kabsch(self._ref_coords, coords, weights)
            self._fits_rot = {im: self.rotate_multipoles(fits, self._Trot) for im, fits in self._fits.items()}
            # coords_ref_basis = (coords - self._com_coords) @ self._Trot.T + self._com_ref
            self.pc_chrg = np.array(self.QMin.coords['pccharge'])    # n_pc, 1
            pc_coord = np.array(self.QMin.coords['pccoords'])    # n_pc, 1
            # matrix of position differences (for gradient calc) n_coord (A), n_pc (B), 3
            pc_coord_diff = np.full((coords.shape[0], pc_coord.shape[0], 3), coords[:, None, :]) - pc_coord
            mult_prefactors = self.get_mult_prefactors(pc_coord_diff)
            mult_prefactors_pc = np.einsum('b,yab->yab', self.pc_chrg, mult_prefactors)
            del mult_prefactors
        # Build full H and diagonalize
        self._Q = np.sqrt(self._Om) * (self._Km @ (coords_ref_basis.flatten() - self._ref_coords.flatten()))
        self._V = self._Om * self._Q
        V0 = 0.5 * (self._V) @ self._Q
        start = 0    # starting index for blocks
        # TODO what if I want to get gradients only ? i.e. samestep
        for im, n in filter(lambda x: x[1] != 0, enumerate(states)):
            H = np.diag(self._epsilon[im] + V0)
            H += self._H_i[im] @ self._Q
            if do_pc:
                # assert np.allclose(ene, ene_v, rtol=1e-8)
                H += np.einsum('ijay,yab->ij', self._fits_rot[im], mult_prefactors_pc, casting='no', optimize=True)
            stop = start + n
            if self._diagonalize:
                eigen_values, self._U[start:stop, start:stop] = np.linalg.eigh(H, UPLO='U')
                np.einsum('ii->i', Hd)[start:stop] = eigen_values
            else:
                self._U[start:stop, start:stop] = np.identity(n, dtype=float)
                Hd[start:stop, start:stop] = H

            for s1 in map(
                lambda x: start + n * (x + 1), range(im)
            ):    # fills in blocks for other magnetic quantum numbers
                s2 = s1 + n
                self._U[s1:s2, s1:s2] = self._U[start:stop, start:stop]
                if self._diagonalize:
                    np.einsum('ii->i', Hd)[s1:s2] = eigen_values
                else:
                    Hd[s1:s2, s1:s2] = H

            start += n * (im + 1)
        grad = np.zeros((nmstates, r3N))

        if do_kabsch:
            if do_pc:
                pc_grad = np.zeros((nmstates, self.pc_chrg.shape[0], 3))

                mult_prefactors_deriv = self.get_mult_prefactors_deriv(pc_coord_diff)

                fits_deriv = {
                    im: np.zeros((n, n, self.QMin.molecule['natom'], 9, self.QMin.molecule['natom'], 3))
                    for im, n in filter(lambda x: x[1] != 0, enumerate(states))
                }

                mult_prefactors_deriv_pc = np.einsum('xyab,b->xyab', mult_prefactors_deriv, self.pc_chrg)
                del mult_prefactors_deriv

        # numerically calculate the derivatives of the coordinates in the reference system with respect ot the sharc coords
            shift = 0.0005
            multiplier = 1 / (2 * shift)

            coords_deriv = np.zeros((r3N, r3N))
            for a in range(self.QMin.molecule['natom']):
                for x in range(3):
                    for f, m in [(1, multiplier), (-1, -multiplier)]:
                        c = np.copy(coords)
                        c[a, x] += f * shift
                        Trot, com_ref, com_c = kabsch(self._ref_coords, c, weights)
                        c_rot = (c - com_c) @ Trot.T + com_ref
                        coords_deriv[..., a * 3 + x] += (m * c_rot).flat
                        if do_pc:
                            for im, n in filter(lambda x: x[1] != 0, enumerate(states)):
                                fits_rot = self.rotate_multipoles(self._fits[im], Trot)
                                fits_deriv[im][..., a, x] += m * fits_rot[..., 1:]

            dQ_dr = np.sqrt(self._Om)[..., None] * (self._Km @ coords_deriv)
            del coords_deriv
        else:
            dQ_dr = np.sqrt(self._Om)[..., None] * self._Km

        # GRADS and NACS
        if self.QMin.requests['nacdr']:
            # Build full derivative matrix
            start = 0    # starting index for blocks
            nacdr = np.zeros((nmstates, nmstates, r3N), float)
            if do_pc:
                nacdr_pc = np.zeros((nmstates, nmstates, self.QMin.molecule['npc'], 3), float)

            for im, n in filter(lambda x: x[1] != 0, enumerate(states)):
                stop = start + n
                u = self._U[start:stop, start:stop]
                dlvc = np.zeros((n, n, r3N))
                np.einsum('iik->ik',
                          dlvc)[...] += self._V[None,
                                                ...]    # fills diagonal on matrix with shape (nmstates,nmstates, r3N)
                dlvc += self._H_i[im]
                dlvc = np.einsum('ijk,kl->ijl', dlvc, dQ_dr, casting='no', optimize=True)
                if do_pc:
                    # calculate derivative of electrostic interaction
                    if "_dcoulomb_path" in self.__dict__:
                        self._dcoulomb_path = np.einsum_path('xyab,ijay->ijabx', mult_prefactors_deriv_pc, self._fits_rot[im], optimize='optimal')[0]
                    dcoulomb: np.ndarray = np.einsum('xyab,ijay->ijabx', mult_prefactors_deriv_pc, self._fits_rot[im], optimize=self._dcoulomb_path)
                    # add derivative to lvc derivative summed ofe all point charges
                    dlvc += np.einsum('ijabx->ijax', dcoulomb).reshape((n, n, r3N))
                    # add the derivative of the multipoles
                    if "_dlvc_path" in self.__dict__:
                        self._dlvc_path = np.einsum_path(
                            'yab,ijaymx->ijmx', mult_prefactors_pc[1:, ...], fits_deriv[im], optimize='optimal'
                        )[0]
                    dlvc += np.einsum(
                        'yab,ijaymx->ijmx', mult_prefactors_pc[1:, ...], fits_deriv[im], casting='no', optimize=self._dlvc_path
                    ).reshape((n, n, r3N))
                    # calculate the pc derivatives
                    pc_derivative = -np.einsum('ijabx->ijbx', dcoulomb)
                    del dcoulomb
                    if self._diagonalize:
                        if "_pc_derivative_nac_path" in self.__dict__:
                            self._pc_derivative_nac_path = np.einsum_path('ijbx,im,jn->mnbx', pc_derivative, u, u, optimize='optimal')[0]
                        pc_derivative = np.einsum('ijbx,im,jn->mnbx', pc_derivative, u, u, casting='no', optimize=self._pc_derivative_nac_path)

                # transform gradients to adiabatic basis
                if self._diagonalize:
                    if "_dlvc_nac_diag_path" not in self.__dict__:
                        self._dlvc_nac_diag_path = np.einsum_path('ijk,im,jn->mnk', dlvc, u, u, optimize='optimal')[0]
                    dlvc = np.einsum('ijk,im,jn->mnk', dlvc, u, u, casting='no', optimize=self._dlvc_nac_diag_path)

                if Hd.dtype == complex:
                    eV = np.reshape(Hd.view(float), (nmstates * nmstates, 2))[::nmstates + 1, 0]
                else:
                    eV = Hd.flat[::nmstates + 1]
                cast = complex if Hd.dtype == complex else float
                # energy weighting of the nacs
                tmp = np.full((n, n), eV[start:stop]).T
                tmp -= eV[start:stop]
                idx = tmp != cast(0)
                tmp[idx] **= -1

                nacdr[start:stop, start:stop, ...] = dlvc
                nacdr[start:stop, start:stop, :] = np.einsum(
                    'ji,ijk->ijk', tmp, nacdr[start:stop, start:stop, :], casting='no', optimize=True
                )
                if do_pc:
                    nacdr_pc[start:stop, start:stop, ...] = pc_derivative
                    nacdr_pc[start:stop, start:stop, :] = np.einsum(
                        'ji,ijbx->ijbx', tmp, nacdr_pc[start:stop, start:stop, :], casting='no', optimize=True
                    )
                # fills in blocks for other magnetic quantum numbers
                for s1 in map(lambda x: start + n * (x + 1), range(im)):
                    s2 = s1 + n
                    nacdr[s1:s2, s1:s2, :] = nacdr[start:stop, start:stop, :]
                    if do_pc:
                        nacdr_pc[s1:s2, s1:s2, :] = nacdr_pc[start:stop, start:stop, :]
                start += n * (im + 1)
            grad = np.einsum('iik->ik', nacdr)
            if do_pc:
                pc_grad += np.einsum('mmbx->mbx', nacdr_pc)

        # calculate only gradients
        if self.QMin.requests['grad']:
            grad = np.zeros((nmstates, r3N))
            start = 0
            for im, n in filter(lambda x: x[1] != 0, enumerate(states)):
                stop = start + n
                u = self._U[start:stop, start:stop]
                grad_lvc = np.full((n, r3N), self._V[None, ...])
                if self._diagonalize:
                    grad_lvc += np.einsum('ijk,in,jn->nk', self._H_i[im], u, u, casting='no', optimize=True)
                    if do_pc:
                        if "_fits_r_path" not in self.__dict__:
                            self._fits_r_path = np.einsum_path('ijay,in,jn->nay', self._fits_rot[im], u, u, optimize='optimal')[0]
                        fits_r = np.einsum('ijay,in,jn->nay', self._fits_rot[im], u, u, casting='no', optimize=self._fits_r_path)
                        if "_dfits_path" not in self.__dict__:
                            self._dfits_path = np.einsum_path('ijaymx,in,jn->naymx', fits_deriv[im], u, u, optimize='optimal')[0]
                        dfits = np.einsum('ijaymx,in,jn->naymx', fits_deriv[im], u, u, casting='no', optimize=True)
                else:
                    grad_lvc += np.einsum('iik->ik', self._H_i[im])
                    if do_pc:
                        fits_r = np.einsum('iiay->iay', self._fits_rot[im])
                        dfits = np.einsum('iiaymx->iaymx', fits_deriv[im])
                grad_lvc = grad_lvc @ dQ_dr
                if do_pc:
                    # calculate derivative of electrostic interaction
                    if "_dcoulomb_grad_path" not in self.__dict__:
                        self._dcoulomb_grad_path = np.einsum_path('xyab,iay->iabx', mult_prefactors_deriv_pc, fits_r, optimize='optimal')[0]
                    dcoulomb: np.ndarray = np.einsum('xyab,iay->iabx', mult_prefactors_deriv_pc, fits_r, casting='no', optimize=self._dcoulomb_grad_path)
                    # add derivative to lvc derivative summed ofe all point charges
                    grad_lvc += np.einsum('iabx->iax', dcoulomb).reshape((n, r3N))
                    # add the derivative of the multipoles
                    if "_grad_lvc_path" not in self.__dict__:
                        self._grad_lvc_path = np.einsum_path(
                            'yab,iaymx->imx', mult_prefactors_pc[1:, ...], dfits, optimize='optimal'
                        )[0]
                    grad_lvc += np.einsum(
                        'yab,iaymx->imx', mult_prefactors_pc[1:, ...], dfits, casting='no', optimize=self._grad_lvc_path
                    ).reshape((n, r3N))
                    # calculate the pc derivatives
                    pc_grad[start:stop, ...] = -np.einsum('iabx->ibx', dcoulomb)
                    del dcoulomb
                grad[start:stop, ...] += grad_lvc
                # fills in blocks for other magnetic quantum numbers
                for s1 in map(lambda x: start + n * (x + 1), range(im)):
                    s2 = s1 + n
                    grad[s1:s2, ...] += grad_lvc
                    if do_pc:
                        pc_grad[s1:s2, ...] = pc_grad[start:stop, ...]
                start += n * (im + 1)

        if self.QMin.requests['overlap']:
            if self.QMin.save['step'] == 0:
                pass
            elif self.persistent:
                overlap = self._Uold.T @ self._U
            else:
                overlap = np.fromfile(os.path.join(self.QMin.save['savedir'], 'Uold.out'),
                                      dtype=float).reshape(self._U.shape).T @ self._U

        # OVERLAP
        if self.persistent:
            self._Uold = np.copy(self._U)
        else:
            self._U.tofile(
                os.path.join(self.QMin.save['savedir'], 'Uold.out')
            )    # writes a binary file (can be read with numpy.fromfile())


        # ========================== Prepare results ========================================
        Hd += self._U.T @ self._soc @ self._U


        dipole = np.einsum('ni,kij,jm->knm', self._U.T, self._dipole, self._U, casting='no', optimize=True) if self._diagonalize else self._dipole
        if do_kabsch:
            #  self._QMin['coords'] = self._QMin['coords']
            dipole = (np.einsum('inm,ij->jnm', dipole, self._Trot))

        grad = grad.reshape((nmstates, self.QMin.molecule['natom'], 3))
        if self.QMin.requests['nacdr']:
            nacdr = nacdr.reshape((nmstates, nmstates, self.QMin.molecule['natom'], 3))

        if self.QMin.requests['multipolar_fit']:
            multipolar_fit = np.zeros((nmstates, nmstates, self.QMin.molecule['natom'], 10), dtype=float)
            start = 0
            for im, n in filter(lambda x: x[1] != 0, enumerate(states)):
                stop = start + n
                u = self._U[start:stop, start:stop]
                if do_kabsch or do_pc:
                    adia_fit = np.einsum('in,ijay,jm->nmay', u, self._fits_rot[im], u, optimize=True, casting='no')
                else:
                    adia_fit = np.einsum('in,ijay,jm->nmay', u, self._fits[im], u, optimize=True, casting='no')
                multipolar_fit[start:stop, start:stop, ...] = adia_fit
                for s1 in map(lambda x: start + n * (x + 1), range(im)):
                    s2 = s1 + n
                    multipolar_fit[s1:s2, s1:s2, ...] = adia_fit
                start += n * (im + 1)

        # ======================================== assign to QMout =========================================
        if states == req_states:
            self.log.debug(f"requests: {self.QMin.requests}")
            self.QMout.states = req_states
            self.QMout.nstates = self.QMin.molecule['nstates']
            self.QMout.nmstates = self.QMin.molecule['nmstates']
            self.QMout.natom = self.QMin.molecule['natom']
            self.QMout.npc = self.QMin.molecule['npc']
            self.QMout.point_charges = do_pc
            self.QMout.h = Hd
            self.QMout.dm = dipole
            if self.QMin.requests['overlap']:
                self.QMout.overlap = overlap
            if self.QMin.requests['grad']:
                self.QMout.grad = grad
            if self.QMin.requests['nacdr']:
                self.QMout.nacdr = nacdr
                if do_pc:
                    self.QMout.nacdr_pc = nacdr_pc
            if do_pc:
                self.log.debug("assign pcgrad")
                self.QMout.grad_pc = pc_grad
            if self.QMin.requests['multipolar_fit']:
                self.QMout.multipolar_fit = multipolar_fit
        else:
            self.log.info(f"returnung subset of states {states} -> {req_states}")
            #  raise NotImplementedError("Calculating with less states is not yet implemented")
            self.QMout.allocate(req_states, self.QMin.molecule['natom'], self.QMin.molecule['npc'], self.QMin.requests)
            matrices = [(self.QMout.h, H, 2), (self.QMout.dm, dipole, 1)]
            if self.QMin.requests['overlap']:
                matrices.append((self.QMout.overlap, overlap, 2))
            if self.QMin.requests['grad']:
                matrices.append((self.QMout.grad, grad, 1))
            if self.QMin.requests['nacdr']:
                matrices.append((self.QMout.nacdr, nacdr, 2))
                if do_pc:
                    matrices.append((self.QMout.nacdr_pc, nacdr_pc, 2))
            if do_pc:
                matrices.append((self.QMout.grad_pc, pc_grad, 1))
            if self.QMin.requests['multipolar_fit']:
                matrices.append((self.QMout.multipolar_fit, multipolar_fit))

            start = 0
            start_qm = 0
            for im, (n, nr) in filter(lambda x: x[1][1] != 0, enumerate(zip(states, req_states))):
                stop_qm = start + nr
                for (qm_mat, mat, dim) in matrices:
                    if dim == 1:
                        qm_mat[start_qm:stop_qm, ...] = mat[start:stop, ...]
                    else:
                        qm_mat[start_qm:stop_qm, start_qm:stop_qm, ...] = mat[start:stop, start:stop, ...]

                for x in range(1, im):
                    s1 = start + n * (x + 1)
                    s1_qm = start_qm + nr * (x + 1)
                #  for s1 in map(lambda x: start + n * (x + 1), range(im)):
                    s2 = s1 + nr
                    s2_qm = s1_qm + nr
                    for (qm_mat, mat, dim) in matrices:
                        if dim == 1:
                            qm_mat[s1_qm:s2_qm, ...] = mat[s1:s2, ...]
                        else:
                            qm_mat[s1_qm:s2_qm, s1_qm:s2_qm, ...] = mat[s1:s2, s1:s2, ...]
                start += n * (im + 1)
                start_qm += nr * (im + 1)

        self._step += 1
        return

    def create_restart_files(self):
        self._U.tofile(
            os.path.join(self.QMin.save['savedir'], f'U_{self.QMin.save["step"]}.out')
        )    # writes a binary file (can be read with numpy.fromfile())


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
        }

    def get_infos(self, INFOS: dict, KEYSTROKES: TextIOWrapper = None) -> dict:

        self.log.info("=" * 80)
        self.log.info(f"{'||':<78}||")
        self.log.info(f"||{'LVC interface setup':=^76}||\n{'||':<78}||")
        self.log.info("=" * 80)
        self.log.info("\n")

        self.template_file = question("Specify path to LVC.template", str, KEYSTROKES=KEYSTROKES, autocomplete=True)

        # Check template for Soc and multipoles and states
        soc_found = False
        mfit_found = False
        dm_found = False
        with open(self.template_file, 'r') as f:
            for line in f:
                if 'SOC' in line:
                    soc_found = True
                if 'DM' in line:
                    dm_found = True
                if 'Multipolar Density Fit' in line:
                    mfit_found = True
        if 'soc' in INFOS['needed_requests'] and not soc_found:
            self.log.error(f"Requested SOC calculation but 'SOC' keyword not found in {self.template_file}")
            raise RuntimeError()

        if ('multipolar_fit' in INFOS['needed_requests'] or 'point_charges' in INFOS['needed_requests']) and not mfit_found:
            self.log.error(f"Calculation with 'point_charges' and/or 'multipolar_fit' requested but 'Multipolar Density Fit' not found in {self.template_file}")
            raise RuntimeError()

        if 'dm' in INFOS['needed_requests'] and not dm_found:
            self.log.error(f"Calculation of dipole moment requested but 'DM' keyword not found in {self.template_file}")
            raise RuntimeError()

        if question("Do you have an LVC.resources file?", bool, KEYSTROKES=KEYSTROKES, autocomplete=False, default=False):
            self.resources_file = question("Specify path to LVC.resources", str, KEYSTROKES=KEYSTROKES, autocomplete=True)

        return INFOS


if __name__ == '__main__':
    from logger import loglevel
    lvc = SHARC_LVC(loglevel=loglevel)
    lvc.main()
