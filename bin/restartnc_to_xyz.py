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

# Script for the calculation of Wigner distributions from molden frequency files
#
# usage python restartnc_to_xyz.py -t <time step>  <prmtop file>  <restartnc file>

import sys
from netCDF4 import Dataset
import numpy as np
from itertools import chain
import datetime
import os
from optparse import OptionParser

from constants import au2fs, ANG_TO_BOHR, AMBERVEL_TO_AU, IAn2AName
from utils import readfile




# =========================================================
# some constants
DEBUG = False

version = '4.0'
versiondate = datetime.date(2024, 11, 30)








def main(prmtop, rst_file, dt, ang, app, gv, ini):
    # get number of atoms and elements
    natom = -1
    with open(prmtop, "r") as f:
        line = f.readline()
        while line:
            if "%FLAG POINTERS" in line:
                f.readline()
                natom = int(f.readline().split()[0])
            elif "%FLAG ATOMIC_NUMBER" in line:
                f.readline()
                if natom == -1:
                    print("natom not read!")
                    exit(1)
                nl = (natom - 1) // 10 + 1
                numbers = [int(x) for x in chain(*map(lambda x: f.readline().split(), range(nl)))]
                symbols = [IAn2AName[x] for x in numbers]
            elif "%FLAG MASS" in line:
                f.readline()
                if natom == -1:
                    print("natom not read!")
                    exit(1)
                nl = (natom - 1) // 5 + 1
                masses = [float(x) for x in chain(*map(lambda x: f.readline().split(), range(nl)))]
            line = f.readline()

    with Dataset(rst_file) as dat:
        geom_rst = dat.variables["coordinates"]
        na = dat.dimensions["atom"].size
        nx = dat.dimensions["spatial"].size
        if na != natom:
            print("natom do not match!")
            print(f"From prmtop: {natom}")
            print(f"From NetCDF: {na}")
            exit(1)
        veloc_rst = dat.variables["velocities"]  # angstrom/picosecond, NOT Amber velocity units!

        geom = np.array(geom_rst[:], dtype=np.float32).reshape((na, nx)) * ANG_TO_BOHR
        veloc = np.array(veloc_rst[:], dtype=np.float32).reshape((na, nx)) * ANG_TO_BOHR /1000 * au2fs

        # move geom halve a timestep back (AMBER uses leapfrog!!!)
        geom = geom - 0.5 * dt * veloc / au2fs

        if gv:
            # write geom and veloc files
            with open("geom", "w") as g:
                for s, n, c, m in zip(symbols, numbers, geom, masses):
                    g.write(f"{s:5s} {float(n): 12.8f}   {c[0]: 12.8f} {c[1]: 12.8f} {c[2]: 12.8f}  {m: 12.8f}\n")
            np.savetxt("veloc", veloc, fmt="% 12.8f")

        else:
            if ini:
                # print one initcond to stdout
                string = 'Index     0\nAtoms\n'
                for s, n, c, m, v in zip(symbols, numbers, geom, masses, veloc):
                    string += f"{s:2s} {float(n): 5.1f} {c[0]: 12.8f} {c[1]: 12.8f} {c[2]: 12.8f} {m: 12.8f} {v[0]: 12.8f} {v[1]: 12.8f} {v[2]: 12.8f}\n"
                string += "States\n"
                string += "Ekin      % 16.12f a.u.\n" % (0.)
                string += "Epot_harm % 16.12f a.u.\n" % (0.)
                string += "Epot      % 16.12f a.u.\n" % (0.)
                string += "Etot_harm % 16.12f a.u.\n" % (0.)
                string += "Etot      % 16.12f a.u.\n" % (0.)
                string += "\n\n"
                print(string)
            else:
                # print xyz and possibly requests to stdout
                string = '%i\n' % natom
                if ang:
                    factor = 1. / ANG_TO_BOHR
                    string += f'File {rst_file}, rewound by {dt/2}fs, in Angstrom\n'
                else:
                    factor = 1.
                    string += f'File {rst_file}, rewound by {dt/2}fs, in Bohrs\n'
                for s, c in zip(symbols, geom * factor):
                    string += f"{s:5s} {c[0]: 12.8f} {c[1]: 12.8f} {c[2]: 12.8f} \n"

                # append from QM.in file
                if app:
                    f = 'QM.in'
                    if os.path.isfile(f):
                        ff = readfile(f)
                        for line in ff:
                            if 'angstrom' in line.lower() and not ang:
                                print("QM.in file has 'unit angstrom', but Bohr conversion was requested. Use --angstrom option")
                                sys.exit(1)
                        fake_natom = int(ff[0])
                        string += ''.join(ff[fake_natom+2:])
                print(string)



if __name__ == "__main__":

    parser = OptionParser()
    parser.add_option("-t", "--timestep", dest="dt", type="float", help="specify the timestep in fs")
    parser.add_option("-a", "--angstrom", dest='ang', action='store_true', help="Output in Angstrom (default in Bohr)")
    parser.add_option("-q", "--append_qmin", dest='app', action='store_true', help="Append request lines from file 'QM.in'")
    parser.add_option("-i", "--initconds", dest='ini', action='store_true', help="Produce output to append to initconds instead of xyz")
    parser.add_option("-g", "--geom_veloc", dest='gv', action='store_true', help="Produce geom and veloc files instead of writing to stdout")

    (options, args) = parser.parse_args()
    if len(args) == 0:
        parser.print_usage()
    if options.dt is None:
        print("specify the timestep!!")
        exit(1)
    if options.gv:
        if options.ang:
            sys.stderr.write("Ignoring -a flag while -g is active")
        if options.app:
            sys.stderr.write("Ignoring -q flag while -g is active")
        if options.ini:
            sys.stderr.write("Ignoring -i flag while -g is active")
    if options.ini:
        if options.ang:
            sys.stderr.write("Ignoring -a flag while -i is active")
        if options.app:
            sys.stderr.write("Ignoring -q flag while -i is active")
    main(*args, options.dt, options.ang, options.app, options.gv, options.ini)
















