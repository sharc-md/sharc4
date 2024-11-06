#!/usr/bin/env python3

import numpy as np
import sys
from optparse import OptParseError, OptionParser
#import readline
import re
import os
import shutil
from constants import BOHR_TO_ANG, U_TO_AMU

# ======================================================================================================================
# ======================================================================================================================
# ======================================================================================================================


class Traj:
    def __init__(self, filename, retatom=-1):
        f = open(filename)
        self.data = f.readlines()
        f.close()
        self.iline = 0
        self.natom = int(self.data[0])
        self.ngeom = len(self.data) / (self.natom + 2)
        if retatom == -1:
            self.retatom = self.natom
        else:
            self.retatom = retatom
        self.el = []
        for i in range(self.retatom):
            self.el.append(self.data[2 + i].split()[0])

    def nextgeom(self):
        c = []
        try:
            title = self.data[self.iline + 1]
            for i in range(self.retatom):
                line = self.data[i + self.iline + 2].split()
                x = []
                for j in range(3):
                    x.append(float(line[j + 1]) / BOHR_TO_ANG)
                c.append(x)
        except IndexError:
            raise EOFError
        self.iline += self.natom + 2
        c = np.array(c)
        cc = self.centroid(c)
        c -= cc
        return c[0:self.retatom], title

    def centroid(self, X):
        return sum(X) / len(X)


# ======================================================================================================================



def find_lines(nlines, match: str, strings: list[str]):
    smatch = match.lower().split()
    iline = -1
    while True:
        iline += 1
        if iline == len(strings):
            return []
        line = strings[iline].lower().split()
        if tuple(line) == tuple(smatch):
            return strings[iline + 1:iline + nlines + 1]


# =========================================================


def read_V0(QMin, SH2LVC, fname):
    """"
  Reads information about the ground-state potential from V0.txt.
  Returns the displacement vector.
  """
    try:
        f = open(fname)
    except IOError:
        print(f'Input file {fname} not found.')
        sys.exit(20)
    v0 = f.readlines()
    f.close()

    # read the coordinates and compute Cartesian displacements
    disp = []    # displacement as one 3N-vector
    SH2LVC['Ms'] = []    # Squareroot masses in a.u.
    geom = QMin['geom']
    tmp = find_lines(QMin['natom'], 'Geometry', v0)
    for i in range(QMin['natom']):
        s = tmp[i].lower().split()
        if s[0] != geom[i][0].lower():
            print(s[0], geom[i][0])
            print(f'Inconsistent atom labels in QM.in and {fname}!')
            sys.exit(21)
        disp += [geom[i][1] - float(s[2]), geom[i][2] - float(s[3]), geom[i][3] - float(s[4])]
        SH2LVC['Ms'] += 3 * [(float(s[5]) * U_TO_AMU)**.5]

    # Frequencies (a.u.)
    tmp = find_lines(1, 'Frequencies', v0)
    if tmp == []:
        print('No Frequencies defined in %s!' % fname)
        sys.exit(22)
    SH2LVC['Om'] = [float(o) for o in tmp[0].split()]

    # Normal modes in mass-weighted coordinates
    tmp = find_lines(len(SH2LVC['Om']), 'Mass-weighted normal modes', v0)
    if tmp == []:
        print('No normal modes given in %s!' % fname)
        sys.exit(23)
    SH2LVC['V'] = [map(float, line.split()) for line in tmp]    # transformation matrix

    return disp


# ======================================================================================================================
# ======================================================================================================================
# ======================================================================================================================


def main():
    parser = OptionParser()
    parser.add_option('-v', '--v-file', dest='vname', default='V0.txt', help='path to the V0.txt file')
    options, args = parser.parse_args()

    vname = os.path.expandvars(options.vname)
    if not os.path.isfile(vname):
        print(f'{vname} file not found! Please specifiy correct path with "-v <path>"')

    trajname = args[0]
    trajname = os.path.expandvars(trajname)
    if not os.path.isfile(trajname):
        print(f'{trajname} file not found!')

    TRAJ = Traj(trajname)

    dt = 0.5

    it = -1
    while True:
        try:
            t, title = TRAJ.nextgeom()
        except EOFError:
            break
        it += 1
        QMin = {'natom': len(t), 'geom': []}
        for i in range(len(t)):
            QMin['geom'].append([TRAJ.el[i]] + t[i].tolist())

        SH2LVC = {}

        disp = read_V0(QMin, SH2LVC, vname)

        # Transform the coordinates to dimensionless mass-weighted normal modes
        r3N = range(3 * QMin['natom'])
        Om = SH2LVC['Om']
        MR = [SH2LVC['Ms'][i] * disp[i] for i in r3N]
        MRV = [0. for i in r3N]
        for i in r3N:
            MRV[i] = sum(MR[j] * SH2LVC['V'][j][i] for j in r3N)
        Q = [MRV[i] * Om[i]**0.5 for i in r3N]

        string = '%6.2f ' % (dt * it)
        for i in Q:
            string += ' %12.9f' % i
        print(string, title.strip())


# ======================================================================================================================

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        sys.stdout.write('\nCtrl+C makes me a sad SHARC ;-(\n')
        quit(0)
