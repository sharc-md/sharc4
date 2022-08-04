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
import sys
import datetime
import numpy as np

# internal
from SHARC_INTERFACE import INTERFACE
from utils import *
from constants import U_TO_AMU, MASSES
from kabsch import kabsch_w as kabsch
from resp import Cartesian2sperical

authors = 'Sebastian Mai and Severin Polonius'
version = '3.0'
versiondate = datetime.datetime(2021, 7, 29)

changelogstring = '''
'''
np.set_printoptions(linewidth=400, formatter={'float': lambda x: f'{x: 9.7}'})


class LVC(INTERFACE):

    _version = version
    _versiondate = versiondate
    _authors = authors
    _changelogstring = changelogstring
    _read_resources = True
    _do_kabsch = True
    _diagonalize = True
    _step = 0

    @property
    def version(self):
        return self._version

    @property
    def versiondate(self):
        return self._versiondate

    @property
    def changelogstring(self):
        return self._changelogstring

    @property
    def authors(self):
        return self._authors

    def read_template(self, template_filename='LVC.template'):
        QMin = self._QMin
        QMin['template'] = {'qmmm': False, 'cobramm': False}
        r3N = 3 * QMin['natom']
        natom = QMin['natom']
        nmstates = QMin['nmstates']

        f = open(os.path.abspath(template_filename), 'r')
        V0file = f.readline()[:-1]
        self.read_V0(os.path.abspath(V0file))
        parsed_states = INTERFACE.parseStates(f.readline())
        states = parsed_states['states']
        if states != QMin['states']:
            print(f'states from QM.in and nstates from LVC.template are inconsistent! {QMin["states"]} != {states}')
            sys.exit(25)

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

        if 'init' in self._QMin:
            INTERFACE.checkscratch(self._QMin['savedir'])
        self._read_template = True
        return

    # TODO: rework to buffer files
    def read_V0(self, filename):
        QMin = self.QMin
        lines = readfile(filename)
        it = 1
        elem = QMin['elements']
        rM = list(
            map(lambda x: [x[0]] + [float(y) for y in x[2:]], map(lambda x: x.split(), lines[it:it + QMin['natom']]))
        )
        if [x[0] for x in rM] != elem:
            raise Error(f'inconsistent atom labels in QM.in and {filename}:\n{rM[0]}\n{elem}')
        rM = np.asarray([x[1:] for x in rM], dtype=float)
        self._ref_coords = rM[:, :-1]
        tmp = np.sqrt(rM[:, -1] * U_TO_AMU)
        self._Msa = np.asarray([tmp, tmp, tmp]).flatten(order='F')
        it += QMin['natom'] + 1
        self._Om = np.asarray(lines[it].split(), dtype=float)
        it += 2
        self._Km = np.asarray([x.split() for x in lines[it:]], dtype=float).T * self._Msa
        return

    def read_resources(self, resources_filename="LVC.resources"):
        pass

    def setup_run(self):
        pass

    def _request_logic(self):
        QMin = self._QMin
        # prepare savedir
        if not os.path.isdir(QMin['savedir']):
            mkdir(QMin['savedir'])

        possibletasks = {'h', 'soc', 'dm', 'grad', 'overlap'}
        tasks = possibletasks & QMin.keys()
        if len(tasks) == 0:
            raise Error(f'No tasks found! Tasks are {possibletasks}.', 39)

        if 'h' not in tasks and 'soc' not in tasks:
            QMin['h'] = True

        if 'soc' in tasks and (len(QMin['states']) < 3 or QMin['states'][2] <= 0):
            del QMin['soc']
            QMin['h'] = True
            print('HINT: No triplet states requested, turning off SOC request.')

        if 'overlap' in tasks and 'init' in tasks:
            raise Error(
                '"overlap" and "phases" cannot be calculated in the first timestep! Delete either "overlap" or "init"',
                43
            )

        if not tasks.isdisjoint({'h', 'soc', 'dm', 'grad'}) and 'overlap' in tasks:
            QMin['h'] = True

        if 'pc_file' in QMin:
            QMin['point_charges'] = [
                [float(x[0]) * self._factor,
                 float(x[1]) * self._factor,
                 float(x[2]) * self._factor,
                 float(x[3])] for x in map(lambda x: x.split(), readfile(QMin['pc_file']))
            ]

    def getQMout(self):
        return self._QMout

    @staticmethod
    def get_mult_prefactors(coords, pc_coords):
        # matrix of position differences (for gradient calc) n_coord (A), n_pc (B), 3
        pc_coord_diff = np.full((coords.shape[0], pc_coords.shape[0], 3), coords[:, None, :]) - pc_coords
        # print(self.pc_coord_diff[1,0,0])
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
    def get_mult_prefactors_deriv(coords, pc_coords):
        pc_coord_diff = np.full((coords.shape[0], pc_coords.shape[0], 3), coords[:, None, :]) - pc_coords
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
        ).reshape((3, 10, coords.shape[0], pc_coords.shape[0]))

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

    # NOTE: potentially do kabsch on reference coords and normal modes (if nmstates**2 > 3*natom)
    def run(self):
        do_pc = 'point_charges' in self._QMin
        self.clock.starttime = datetime.datetime.now()
        nmstates = self._QMin['nmstates']
        self._U = np.zeros((nmstates, nmstates), dtype=float)
        Hd = np.zeros((nmstates, nmstates), dtype=self._soc.dtype)
        states = self._QMin['states']
        r3N = 3 * self._QMin['natom']
        coords: np.ndarray = self._QMin['coords'].copy()
        if self._do_kabsch:
            weights = [MASSES[i] for i in self._QMin['elements']]
            self._Trot, self._com_ref, self._com_coords = kabsch(self._ref_coords, coords, weights)
            self._fits_rot = {im: self.rotate_multipoles(fits, self._Trot) for im, fits in self._fits.items()}
            coords_ref_basis = (coords - self._com_coords) @ self._Trot.T + self._com_ref
        if do_pc:
            pc = np.array(self.QMin['point_charges'])    # pc: list[list[float]] = each pc is x, y, z, q
            pc_coord = pc[:, :3]    # n_pc, 3
            self.pc_chrg = pc[:, 3].reshape((-1, 1))    # n_pc, 1
            mult_prefactors = self.get_mult_prefactors(coords, pc_coord)
            mult_prefactors_pc = np.einsum('b,yab->yab', self.pc_chrg.flat, mult_prefactors)
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
                H += np.einsum('ijay,yab->ij', self._fits_rot[im], mult_prefactors_pc)
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

            start = stop
        # GRADS and NACS
        if 'nacdr' in self._QMin:
            # Build full derivative matrix
            start = 0    # starting index for blocks
            dE = np.zeros((nmstates * nmstates, r3N), float)
            dE[::nmstates + 1, :] += self._V    # fills diagonal on matrix with shape (nmstates,nmstates, r3N)
            dE = dE.reshape((nmstates, nmstates, r3N))
            for im, n in filter(lambda x: x[1] != 0, enumerate(states)):
                stop = start + n
                dE[start:stop, start:stop, :] += self._H_i[im]
                for s1 in map(
                    lambda x: start + n * (x + 1), range(im)
                ):    # fills in blocks for other magnetic quantum numbers
                    s2 = s1 + n
                    dE[s1:s2, s1:s2, :] = dE[start:stop, start:stop, :]
                start = stop
            dE = np.einsum('ijr,in->njr', dE, self._U, casting='no', optimize=True)
            dE = np.einsum('njr,jm->nmr', dE, self._U, casting='no', optimize=True)
            dE = np.einsum('mnr,r->nmr', dE, np.sqrt(self._Om), casting='no', optimize=True)
            dE = np.einsum('ij,kli->klj', self._Km, dE, casting='no', optimize=True)
            grad = np.einsum('nnl->nl', dE)    # gradients in cartesian basis
            nacdr = dE
        else:

            shift = 0.0005
            multiplier = 1 / (2 * shift)
            weights = [MASSES[i] for i in self._QMin['elements']]
            num_deriv = {
                im: np.zeros((n, n, self._QMin['natom'], 3))
                for im, n in filter(lambda x: x[1] != 0, enumerate(states))
            }
            num_deriv_lvc = {
                im: np.zeros((n, n, self._QMin['natom'], 3))
                for im, n in filter(lambda x: x[1] != 0, enumerate(states))
            }
            num_deriv_lvc_ii = {
                im: np.zeros((n, n, self._QMin['natom'], 3))
                for im, n in filter(lambda x: x[1] != 0, enumerate(states))
            }
            num_deriv_lvc_ij = {
                im: np.zeros((n, n, self._QMin['natom'], 3))
                for im, n in filter(lambda x: x[1] != 0, enumerate(states))
            }
            num_deriv_coulomb = {
                im: np.zeros((n, n, self._QMin['natom'], 3))
                for im, n in filter(lambda x: x[1] != 0, enumerate(states))
            }

            for a in range(self._QMin['natom']):
                for x in range(3):
                    for f, m in [(1, multiplier), (-1, -multiplier)]:
                        c = np.copy(coords)
                        c[a, x] += f * shift
                        Trot, com_ref, com_c = kabsch(self._ref_coords, c, weights)
                        c_rot = (c - com_c) @ Trot.T + com_ref
                        # print((c_rot.flatten() - self._ref_coords.flatten())[:3])
                        Q = np.sqrt(self._Om) * (self._Km @ (c_rot.flatten() - self._ref_coords.flatten()))
                        V = self._Om * Q
                        V0 = 0.5 * (V) @ Q
                        mp = self.get_mult_prefactors(c, pc_coord)
                        start = 0
                        for im, n in filter(lambda z: z[1] != 0, enumerate(states)):
                            fits_rot = self.rotate_multipoles(self._fits[im], Trot)
                            stop = start + n
                            H = np.diag(self._epsilon[im] + V0)
                            num_deriv_lvc_ii[im][..., a, x] += m * H
                            num_deriv_lvc_ij[im][..., a, x] += m * (self._H_i[im] @ Q)
                            H += self._H_i[im] @ Q
                            num_deriv_lvc[im][..., a, x] += m * H
                            H_c = np.einsum('ijay,yab,b->ij', fits_rot, mp, self.pc_chrg.flat)
                            num_deriv_coulomb[im][..., a, x] += m * H_c
                            H += H_c
                            # hd = np.einsum('in,ij,jm->nm', u, H, u, casting='no', optimize=True)
                            num_deriv[im][..., a, x] += m * H
            #                 for s1 in map(
            #                     lambda z: start + n * (z + 1), range(im)
            #                 ):    # fills in blocks for other magnetic quantum numbers
            #                     s2 = s1 + n
            #                     grad[s1:s2, s1:s2, a, x] += m * hd
            #                 start = stop
            shift = 0.005
            multiplier = 1 / (2 * shift)

            coords_deriv = np.zeros((r3N, r3N))
            # Km_deriv = np.zeros((r3N, self._QMin['natom'], 3, self._QMin['natom'], 3))
            # Km = self._Km.copy().reshape((r3N, self._QMin['natom'], 3)) 
            dQ_dr = np.zeros((r3N, self._QMin['natom'], 3))
            for a in range(self._QMin['natom']):
                for x in range(3):
                    for f, m in [(1, multiplier), (-1, -multiplier)]:
                        c = np.copy(coords)
                        c[a, x] += f * shift
                        Trot, com_ref, com_c = kabsch(self._ref_coords, c, weights)
                        c_rot = (c - com_c) @ Trot.T + com_ref
                        coords_deriv[..., a * (self._QMin['natom'] - 1) + x] += (m * c_rot).flat
                        # Km_deriv[..., a, x] += m * Km @ Trot.T
                        # dQ_dr[..., a, x] += m * np.sqrt(self._Om) * (self._Km @ (c_rot.flatten() - self._ref_coords.flatten()))
                        # coords_deriv[..., a, x] += m * ( - com_c) @ Trot.T + com_ref
            

            # coords_deriv = coords_deriv.reshape((r3N, r3N))
            np.savetxt('coords_deriv.txt', coords_deriv)
            # Km_deriv = Km_deriv.reshape((r3N, r3N, r3N))
            dQ_dr = dQ_dr.reshape((r3N, r3N))

            grad = np.zeros((nmstates, r3N))
            grad2 = np.zeros((nmstates, nmstates, r3N))
            grad2 = {
                im: np.zeros((n, n, r3N))
                for im, n in filter(lambda x: x[1] != 0, enumerate(states))
            }
            # Km_sharc = np.einsum('kl,lm->km', self._Km, coords_deriv)
            # self._Q = np.sqrt(self._Om) * (self._Km @ (coords_ref_basis.flatten() - self._ref_coords.flatten()))
            # coords_div = (coords_ref_basis.flatten() - self._ref_coords.flatten())
            dQ_dr = np.sqrt(self._Om)[..., None] * (self._Km @ coords_deriv)
            # dQ_dr += np.einsum('m,mkl,k->ml', np.sqrt(self._Om), Km_deriv, coords_div)
            V_sharc = (self._Om * self._Q)
            # start = 0
            for im, n in filter(lambda x: x[1] != 0, enumerate(states)):
                stop = start + n
                # g = grad2[start:stop, :]
                np.einsum('iik->ik', grad2[im])[...] += V_sharc[None, ...]
                grad2[im] += self._H_i[im].copy()
                # u = self._U[start:stop, start:stop]
                # hd = np.einsum('im,ijk,jm->mk', u, h, u, casting='no', optimize=True)
                # g += h
                grad2[im] = grad2[im] @ dQ_dr
            print('soll')
            print(num_deriv_lvc[0][0, 0, 0, :])
            print('ist')
            print(grad2[0][0, 0, :3])
            # assert False

        if do_pc:
            self.pc_grad = np.zeros((nmstates, self.pc_chrg.shape[0], 3))

            mult_prefactors_deriv = self.get_mult_prefactors_deriv(coords, pc_coord)

            shift = 0.0005
            multiplier = 1 / (2 * shift)
            weights = [MASSES[i] for i in self._QMin['elements']]

            fits_deriv = {
                im: np.zeros((n, n, self._QMin['natom'], 9, self._QMin['natom'], 3))
                for im, n in filter(lambda x: x[1] != 0, enumerate(states))
            }

            # num_deriv = {im: np.zeros((n, n, self._QMin['natom'], 3))
            #     for im, n in filter(lambda x: x[1] != 0, enumerate(states))}

            for a in range(self._QMin['natom']):
                for x in range(3):
                    for f, m in [(1, multiplier), (-1, -multiplier)]:
                        c = np.copy(coords)
                        c[a, x] += f * shift
                        Trot, _, _ = kabsch(self._ref_coords, c, weights)
                        for im, n in filter(lambda x: x[1] != 0, enumerate(states)):
                            fits_rot = self.rotate_multipoles(self._fits[im], Trot)
                            fits_deriv[im][..., a, x] += m * fits_rot[..., 1:]
                            # mp = self.get_mult_prefactors(c, pc_coord)
                            # num_deriv[im][..., a, x] += m * np.einsum('ijay,yab,b->ij', fits_rot, mp, self.pc_chrg.flat)

            mult_prefactors_deriv_pc = np.einsum('xyab,b->xyab', mult_prefactors_deriv, self.pc_chrg.flat)
            del mult_prefactors_deriv

            # grad = grad.reshape((nmstates, self._QMin['natom'], 3))
            # grad = np.einsum('mni,ij-> mnj', grad, self._Trot)
            # grad = grad.reshape((nmstates, -1))
            start = 0
            for im, n in filter(lambda x: x[1] != 0, enumerate(states)):
                stop = start + n
                # gradients on point charges
                u = self._U[start:stop, start:stop]

                derivative: np.ndarray = np.einsum('xyab,ijay->ijabx', mult_prefactors_deriv_pc, self._fits_rot[im])

                atom_derivative_ana = np.einsum('ijabx->ijax', derivative)

                rot_deriv_ana = np.einsum(
                    'yab,ijaymx->ijmx', mult_prefactors_pc[1:, ...], fits_deriv[im]
                )
                atom_derivative_ana += rot_deriv_ana
                print('soll coulomb')
                print(num_deriv_coulomb[im][0,0,0,:])
                print('ist coulomb')
                print(atom_derivative_ana[0,0,0,:])

                print()
                print('soll total')
                print(num_deriv[im][0, 0, 0, :])
                print('ist total')
                atom_derivative_ana = atom_derivative_ana.reshape((n, n, -1))
                atom_derivative_ana +=  grad2[im]
                print(atom_derivative_ana[0,0, :3])

                atom_derivative = atom_derivative_ana.reshape((n,n, self._QMin['natom'], 3))
                # atom_derivative = num_deriv[im]
                pc_derivative = -np.einsum('ijabx->ijbx', derivative)
                # print(pc_derivative[0,0,:])
                del derivative
                pc_derivative_trans = np.einsum('ijbx,im,jn->mnbx', pc_derivative, u, u)
                self.pc_grad[start:stop, ...] += np.einsum('mmbx->mbx', pc_derivative_trans)
                atom_derivative = np.einsum('ijax,im,jn->mnax', atom_derivative, u, u).reshape((n, n, -1))
                # atom_derivative = atom_derivative.reshape((n, n, -1))

                grad[start:stop, ...] = np.einsum('mmk->mk', atom_derivative)
                # print(grad[start:stop, :3])

                np.einsum('iik->ik', atom_derivative)[...] = np.zeros(
                    (n, self.QMin['natom'] * 3), dtype=float
                )    # set diagonal to zero
                if 'nacdr' in self._QMin:
                    nacdr[start:stop, start:stop, ...] += atom_derivative
                for s1 in map(lambda x: start + n * (x + 1), range(im)):
                    s2 = s1 + n
                    # add diagonal to grad and off diagonals to nacdr
                    grad[s1:s2, ...] = grad[start:stop, ...]
                    self.pc_grad[s1:s2, ...] = self.pc_grad[start:stop, ...]
                    if 'nacdr' in self._QMin:
                        nacdr[s1:s2, s1:s2, ...] = nacdr[start:stop, start:stop, ...]
                start = stop

        # do the energy weighting of the nacdrs
        if 'nacdr' in self._QMin:
            start = 0    # starting index for blocks
            if Hd.dtype == complex:
                eV = np.reshape(Hd.view(float), (nmstates * nmstates, 2))[::nmstates + 1, 0]
            else:
                eV = Hd.flat[::nmstates + 1]
            cast = complex if Hd.dtype == complex else float
            for im, n in filter(lambda x: x[1] != 0, enumerate(states)):
                stop = start + n
                tmp = np.full((n, n), eV[start:stop]).T
                tmp -= eV[start:stop]
                idx = tmp != cast(0)
                tmp[idx] **= -1
                nacdr[start:stop, start:stop, :] = np.einsum(
                    'ji,ijk->ijk', tmp, nacdr[start:stop, start:stop, :], casting='no', optimize=True
                )
                for s1 in map(
                    lambda x: start + n * (x + 1), range(im)
                ):    # fills in blocks for other magnetic quantum numbers
                    s2 = s1 + n
                    nacdr[s1:s2, s1:s2, :] = nacdr[start:stop, start:stop, :]
                start = stop

        if 'overlap' in self._QMin:
            if 'init' in self._QMin:
                overlap = np.identity(nmstates, dtype=float)
            elif 'restart' in self._QMin:
                overlap = np.fromfile(
                    os.path.join(self._QMin['savedir'], f'U_{self._QMin["step"]-1}.out'), dtype=float
                ).reshape(self._U.shape).T @ self._U
            elif self._persistent:
                overlap = self._Uold.T @ self._U
            else:
                overlap = np.fromfile(os.path.join(self._QMin['savedir'], 'Uold.out'),
                                      dtype=float).reshape(self._U.shape).T @ self._U
            self._QMout['overlap'] = overlap.tolist()

        if self._persistent:
            self._Uold = np.copy(self._U)
        else:
            self._U.tofile(
                os.path.join(self._QMin['savedir'], 'Uold.out')
            )    # writes a binary file (can be read with numpy.fromfile())
        # OVERLAP
        Hd += self._U.T @ self._soc @ self._U
        self._QMout['h'] = Hd.tolist()
        if self._do_kabsch:
            self._QMin['coords'] = self._QMin['coords']
            dipole = np.einsum('ni,kij,jm->knm', self._U.T, self._dipole, self._U, casting='no', optimize='optimal')
            self._QMout['dm'] = (np.einsum('inm,ij->jnm', dipole, self._Trot)).tolist()
            grad = grad.reshape((nmstates, self._QMin['natom'], 3))
            # self._QMout['grad'] = (np.einsum('mni,ij-> mnj', grad, self._Trot)).tolist()
            self._QMout['grad'] = grad.tolist()
            if 'nacdr' in self._QMin:
                nacdr = nacdr.reshape((nmstates, nmstates, self._QMin['natom'], 3))
                # self._QMout['nacdr'] = np.einsum('mnki,ij->mnkj', nacdr, self._Trot).tolist()
                self._QMout['nacdr'] = nacdr.tolist()
            if do_pc:
                # self._QMout['pc_grad'] = np.einsum('mni,ij-> mnj', self.pc_grad, self._Trot).tolist()
                self._QMout['pc_grad'] = self.pc_grad.tolist()
        else:
            self._QMout['dm'] = np.einsum(
                'ni,kij,jm->knm', self._U.T, self._dipole, self._U, casting='no', optimize='optimal'
            ).tolist()
            self._QMout['grad'] = grad.reshape((nmstates, self._QMin['natom'], 3)).tolist()
            if 'nacdr' in self._QMin:
                self._QMout['nacdr'] = nacdr.reshape((nmstates, nmstates, self._QMin['natom'], 3)).tolist()
            if do_pc:
                self._QMout['pc_grad'] = self.pc_grad.tolist()

        # self._QMout['runtime'] = self.clock.measuretime()
        self._step += 1
        return

    def create_restart_files(self):
        self._U.tofile(
            os.path.join(self._QMin['savedir'], f'U_{self._QMin["step"]}.out')
        )    # writes a binary file (can be read with numpy.fromfile())

    def main(self):
        name = self.__class__.__name__
        args = sys.argv
        if len(args) != 2:
            print(
                'Usage:',
                f'./SHARC_{name} <QMin>',
                f'version: {self.version}',
                f'date: {self.versiondate}',
                f'changelog: {self.changelogstring}',
                sep='\n'
            )
            sys.exit(106)
        QMinfilename = sys.argv[1]
        pwd = os.getcwd()
        self.printheader()
        self.setup_mol(os.path.join(pwd, QMinfilename))
        self.read_template()
        self.set_coords(os.path.join(pwd, QMinfilename))
        self.read_requests(os.path.join(pwd, QMinfilename))
        self.run()
        self.write_step_file()
        # if PRINT or DEBUG:
        #     self.printQMout()
        self.writeQMout()


if __name__ == '__main__':
    lvc = LVC()
    lvc.main()
