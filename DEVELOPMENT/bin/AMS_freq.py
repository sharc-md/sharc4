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


from scm.plams import KFFile
import sys
import os

try:
    import numpy
except ImportError:
    print('The kffile module required to read AMS binary files needs numpy. Please install numpy and then try again')
    sys.exit()

adf = os.path.expandvars('$AMSHOME')
sys.path.append(adf + '/scripting')

# ================================================


def read_t21(filename):
    f = KFFile(filename)

    try:
        f.sections()
    except IndexError:
        print('File does not seem to be a rkf file...')
        return None

    Freq = f.read("Vibrations", "Frequencies[cm-1]")

    natom = int(f.read("Geometry", "nr of atoms"))

    Atomsymbols = f.read("Molecule", "AtomSymbols").split()

    xyz = f.read("Geometry", "xyz InputOrder")
    FreqCoord = []
    FreqCoord = [[Atomsymbols[n // 3]] + xyz[n:n + 3] for n in range(0, len(xyz), 3)]  # makes chunks of length 3 (i.e. x y z per atom) into list
    nModes = f.read("Vibrations", "nNormalModes")

    # Reads the normal modes
    # TODO: the excluded rigid normal modes can be found in "Vibrations": "NoWeightRigidMode(i)" i ranging from 1 - 6
    Modes = {}
    for i in range(nModes):
        mode = f.read("Vibrations", "NoWeightNormalMode({})".format(i + 1))  # Lists a normal mode as x0, y0, z0, x1, y1, z1, x2 ....
        Modes[i] = [mode[n:n + 3] for n in range(0, len(mode), 3)]  # makes chunks of length 3 (i.e. displacements of single atom) into list

    Int = {}

    F = {'FREQ': Freq,
         'FR-COORD': FreqCoord,
         'FR-NORM-COORD': Modes,
         'INT': Int,
         'natom': natom}
    return F

# ================================================


def read_AMSout(filename):
    data = open(filename).readlines()

    iline = -1
    while True:
        iline += 1
        if iline >= len(data):
            print('Could not find Frequencies output!')
            return None
        line = data[iline]
        if 'F R E Q U E N C' in line:
            break
    iline += 10
    FreqCoord = []
    natom = 0
    while True:
        iline += 1
        line = data[iline]
        if '----' in line:
            break
        s = line.split()
        try:
            atom = [s[1], float(s[2]), float(s[3]), float(s[4])]
        except IndexError:
            print('Could not find optimized coordinates!')
            return None
        FreqCoord.append(atom)
        natom += 1

    while True:
        iline += 1
        line = data[iline]
        if 'Vibrations and Normal Modes  ***  (cartesian coordinates, NOT mass-weighted)  ***' in line:
            break
    iline += 6
    Modes = {}
    x = 0
    y = 0
    while True:
        iline += 1
        line = data[iline]
        if 'List of All Frequencies:' in line:
            break
        s = line.split()
        if '----' in line:
            y = len(s)
            x += y
            for i in range(y):
                Modes[x - y + i] = []
        if len(s) <= 3:
            continue
        for i in range(y):
            m = [float(s[1 + 3 * i]), float(s[2 + 3 * i]), float(s[3 + 3 * i])]
            Modes[x - y + i].append(m)

    iline += 8
    Int = {}
    Freq = []
    x = 0
    while True:
        iline += 1
        line = data[iline]
        s = line.split()
        if len(s) == 0:
            break
        Freq.append(float(s[0]))
        Int[x] = float(s[2])
        x += 1

    F = {'FREQ': Freq,
         'FR-COORD': FreqCoord,
         'FR-NORM-COORD': Modes,
         'INT': Int,
         'natom': natom}
    return F

# ================================================


def format_molden(F):
    string = '[MOLDEN FORMAT]\n[FREQ]\n'
    for i in F['FREQ']:
        string += '%8.2f\n' % i
    string += '[FR-COORD]\n'
    for atom in F['FR-COORD']:
        string += '%4s  %12.9f  %12.9f  %12.9f\n' % tuple(atom)
    string += '[FR-NORM-COORD]\n'
    for i in range(len(F['FR-NORM-COORD'])):
        mode = F['FR-NORM-COORD'][i]
        string += 'Vibration %4i\n' % (i + 1)
        for m in mode:
            string += '  %12.9f  %12.9f  %12.9f\n' % tuple(m)
    if len(F['INT']) > 0:
        string += '[INT]\n'
        for i in range(len(F['FR-NORM-COORD'])):
            if i in F['INT']:
                string += '%12.6f\n' % F['INT'][i]
            else:
                string += '%12.6f\n' % 0.
    return string


# ================================================
filename = sys.argv[1]

F = read_t21(filename)
if not F:
    F = read_AMSout(filename)
if not F:
    print('Could not parse file %s!' % filename)
    sys.exit(1)

string = format_molden(F)
outfile = open(filename + '.molden', 'w')
outfile.write(string)
outfile.close()



# save the shell command
command = 'python ' + ' '.join(sys.argv)
f = open('KEYSTROKES.AMS_freq', 'w')
f.write(command)
f.close()
