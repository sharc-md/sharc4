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

# Script for the conversion of
# Gaussian output files to frequency files
# for molden. Only works with freq=hpmodes

from sys import argv
from constants import NUMBERS, au2a

INV_NUMBERS = {v: k for k, v in NUMBERS.items()}


# Check arguments
if len(argv) != 2:
    print("Usage: GAUSSIAN_freq.py <gaussian.log>\n:")
    print("Convert Gaussian output file to molden file for")
    print("normal mode visualisation.")
    exit()

name, gaussian_file = argv

try:
    lines = open(gaussian_file, 'r').readlines()
except IOError:
    print("Could not open %s." % gaussian_file)
    exit()

# check if file is sucessfully completed orca file:
is_gaussian = False
finished = False
if "Entering Gaussian System" in lines[0]:
    is_gaussian = True
for line in lines:
    if 'hpmodes' in line.lower():
        finished = True

if not is_gaussian:
    print("File %s is not in gaussian output format (probably)!" % gaussian_file)
    exit()
elif is_gaussian and not finished:
    print("Run the job with freq=hpmodes!")
    exit()
elif is_gaussian and finished:
    print("Reading data from file %s..." % gaussian_file)

# Standard orientation: (from bottom)
# get coordinates
for iline, line in enumerate(lines[::-1]):
    original_index = len(lines) - 1 - iline
    if "Standard orientation:" in line:
        break
iline = original_index + 4
coords = []
while True:
    iline += 1
    line = lines[iline]
    if '---' in line:
        break
    s = line.split()
    atom = [ INV_NUMBERS[int(s[1])], float(s[3]), float(s[4]), float(s[5]) ]
    coords.append(atom)
natom = len(coords)
nfreq = 3*natom-6


# freq, int, modes
for iline, line in enumerate(lines):
    if "Harmonic frequencies (cm**-1), IR intensities (KM/Mole), Raman scattering" in line:
        break
iline += 4
freqs = []
modes = []
ints = []
while True:
    if "Harmonic frequencies (cm**-1), IR intensities (KM/Mole), Raman scattering" in lines[iline]:
        break
    s = lines[iline].split()
    nhere = len(s)
    for imode in range(nhere):
        freq =   float(lines[iline+2  ].split()[2+imode])
        ir   =   float(lines[iline+5  ].split()[3+imode])
        mode = [ float(lines[iline+7+i].split()[3+imode]) for i in range(3*natom) ]
        freqs.append(freq)
        ints.append(ir)
        modes.append(mode)
    iline += 7+3*natom

# generate molden file
out_file = gaussian_file + '.molden'
out = open(out_file, 'w')
out.write("[MOLDEN FORMAT]\n")
# write frequencies
out.write("[FREQ]\n")
for freq in freqs:
    out.write(str(freq) + '\n')
# write coordinates block (A.U.)
out.write("[FR-COORD]\n")
for coord in coords:
    out.write(coord[0] + ' ' + ' '.join([str(i/au2a) for i in coord[1:4]]) + '\n')
# write normal modes:
out.write("[FR-NORM-COORD]\n")
for i in range(nfreq):
    out.write("vibration %d\n" % (i + 1))
    for j in range(len(modes[i])):
        out.write(str(modes[i][j]) + ' ')
        if (j + 1) % 3 == 0:
            out.write('\n')
out.write('[INT]\n')
for i in range(nfreq):
    out.write('%16.15f\n' % ints[i])
out.close()
print("Molden output written to %s" % out_file)



