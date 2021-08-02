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
from os import stat
from posixpath import join
import sys
import time
import datetime
import numpy as np

# internal
from SHARC_INTERFACE import INTERFACE
from utils import *
from constants import U_TO_AMU

authors = 'Sebastian Mai and Severin Polonius'
version = '3.0'
versiondate = datetime.datetime(2021, 7, 29)

changelogstring = '''
'''
np.set_printoptions(linewidth=400)

class LVC(INTERFACE):

    _version = version
    _versiondate = versiondate
    _authors = authors
    _changelogstring = changelogstring

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
        r3N = 3 * QMin['natom']
        nmstates = QMin['nmstates']

        f = open(os.path.abspath(template_filename), 'r')
        V0file = f.readline()[:-1]
        self.read_V0(os.path.abspath(V0file))
        states = [int(s) for s in f.readline().split()]
        if states != QMin['states']:
            print(f'states from QM.in and nstates from LVC.template are inconsistent! {QMin["states"]} != {states}')
            sys.exit(25)

        self._H = {im: np.zeros((n, n), dtype=float) for im, n in enumerate(states) if n != 0}
        self._H_i = {im: np.zeros((n, n, r3N), dtype=float) for im, n in enumerate(states) if n != 0}
        self._epsilon = {im: np.zeros(n, dtype=float) for im, n in enumerate(states) if n != 0}
        self._eV = {im: np.zeros(n, dtype=float) for im, n in enumerate(states) if n != 0}
        self._dipole = np.zeros((3, nmstates, nmstates), dtype=complex)
        self._soc = np.zeros((nmstates, nmstates), dtype=complex)
        self._U = np.zeros((nmstates, nmstates), dtype=complex)
        self._Q = np.zeros(r3N, float)
        xyz = {'X': 0, 'Y': 1, 'Z': 2}
        soc_real = True
        dipole_real = True
        line = f.readline()
        # NOTE: possibly assign whole arry with index accessor (numpy)
        if line == 'epsilon\n':
            z = int(f.readline()[:-1])

            def a(x):
                v = f.readline().split()
                return (int(v[0]) - 1, int(v[1]) - 1, float(v[2]))
            # for im, s, v in map(lambda v: [int(v[0]) - 1, int(v[1]) - 1, float(v[2])], [f.readline().split() for _ in range(z)]):
            for im, s, v in map(a, range(z)):
                self._epsilon[im][s] += v
        if f.readline() == 'kappa\n':
            z = int(f.readline()[:-1])

            def b(_):
                v = f.readline().split()
                return (int(v[0]) - 1, int(v[1]) - 1, int(v[2]) - 1, float(v[3]))
            # for im, s, i, v in map(lambda v: [int(v[0]) - 1, int(v[1]) - 1, int(v[2]) - 1, float(v[3])], [f.readline().split() for _ in range(z)]):
            for im, s, i, v in map(b, range(z)):
                self._H_i[im][s, s, i] = v
        if f.readline() == 'lambda\n':
            z = int(f.readline()[:-1])
            
            def c(_):
                v = f.readline().split()
                return (int(v[0]) - 1, int(v[1]) - 1, int(v[2]) - 1, int(v[3]) - 1, float(v[4]))
            # for im, si, sj, i, v in map(lambda v: [int(v[0]) - 1, int(v[1]) - 1, int(v[2]) - 1, int(v[3]) - 1, float(v[4])], [f.readline().split() for _ in range(z)]):
            for im, si, sj, i, v in map(c, range(z)):
                self._H_i[im][si, sj, i] = v
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
            else:
                line = f.readline()
        f.close()
        # setting type as necessary (converting type through view and reshape is a lot faster that simple astype assignemnt)
        if soc_real:
            self._soc = np.reshape(self._soc.view(float), self._soc.shape + (2,))[:, :, 0]
        if dipole_real:
            self._dipole = np.reshape(self._dipole.view(float), self._dipole.shape + (2,))[:, :, :, 0]
        # timing (BIG): 0.59
        return

    def read_V0(self, filename):
        QMin = self.QMin
        lines = readfile(filename)
        it = 1
        elem = QMin['elements']
        rM = list(map(lambda x: [x[0]] + [float(y) for y in x[2:]], map(lambda x: x.split(), lines[it:it + QMin['natom']])))
        if [x[0] for x in rM] != elem:
            raise Error(f'inconsistent atom labels in Qm.in and {filename}:\n{rM[:,0]}\n{elem}')
        rM = np.asarray([x[1:] for x in rM], dtype=float)
        self._disp = rM[:, :-1]
        tmp = np.sqrt(rM[:, -1] * U_TO_AMU)
        self._Msa = np.asarray([tmp, tmp, tmp]).flatten(order='F')
        it += QMin['natom'] + 1
        self._Om = np.asarray(lines[it].split(), dtype=float)
        it += 2
        self._Km = np.asarray([x.split() for x in lines[it:]], dtype=float) * self._Msa
        return


    def read_resources(self, resources_filename):
        return

    def run(self):
        # displacements
        self.clock.starttime = datetime.datetime.now()
        S = time.time_ns()
        nmstates = self._QMin['nmstates']
        Hd = np.zeros((nmstates, nmstates), dtype=complex)
        states = self._QMin['states']
        self._Q += np.sqrt(self._Om) * (self._Km @ (self._QMin['coords'].flatten() - self._disp.flatten()))
        V0 = 0.5 * (self._Om @ self._Q)
        print(V0)
        start = 0  # starting index for blocks
        for im, n in enumerate(states):
            if n == 0:
                continue
            H = self._H[im]
            np.einsum('ii->i', H)[:] = self._epsilon[im] + V0
            H += self._H_i[im] @ self._Q
            # print(H)
            stop = start + n
            np.einsum('ii->i', Hd)[start:stop], self._U[start:stop, start:stop] = np.linalg.eigh(H, UPLO='U')
            for s1 in map(lambda x: start + n * (x + 1), range(im)):  # fills in blocks for other magnetic quantum numbers
                s2 = s1 + n
                self._U[s1:s2, s1:s2] = self._U[start:stop, start:stop]
                np.einsum('ii->i', Hd)[s1:s2] = np.einsum('ii->i', Hd)[start:stop]
            start = stop
        Hd += self._U.T @ self._soc @ self._U
        dE = {}
        dQ = np.sqrt(self._Om) * self._Km
        dV0 = self._Om @ dQ
        for im, H in self._H_i.items():
            print(self._H_i[im].shape, dQ.shape)
            dE[im] =  np.einsum('ijk,kl', self._H_i[im], dQ)
            np.einsum('iij->ij', dE[im])[:, :] += dV0 
        
        print('run:', (time.time_ns() - S) * 1.e-9)
        self.clock.measuretime()
        return
