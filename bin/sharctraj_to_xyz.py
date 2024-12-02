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

from constants import au2fs, ANG_TO_BOHR, U_TO_AMU, IAn2AName
from utils import readfile




# =========================================================
# some constants
DEBUG = False

version = '4.0'
versiondate = datetime.date(2024, 11, 30)








def main(geomfile, ncfile, step, ang, app, gv, ini):
    # get number of atoms, elements, numbers, masses from geom file
    dat = readfile(geomfile)
    numbers = []
    symbols = []
    masses = []
    for line in dat:
        s = line.split()
        symbols.append( s[0] )
        numbers.append( float(s[1]) )
        masses.append( float(s[5]) )
    natom = len(masses)

    # look into NetCDF file
    with Dataset(ncfile) as dat:

        # get dimensions and data
        nstep, natom2, nspat = dat.variables["geom"].shape
        geom_rst = dat.variables["geom"]
        veloc_rst = dat.variables["veloc"]

        # figure out which step we want
        geom  = np.array( geom_rst[step, :, :], dtype=np.float32).reshape(3, natom)
        veloc = np.array(veloc_rst[step, :, :], dtype=np.float32).reshape(3, natom)
        geom  = np.einsum("xa->ax", geom)
        veloc = np.einsum("xa->ax", veloc)


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
                    string += f'File {ncfile}\n'
                else:
                    factor = 1.
                    string += f'File {ncfile}\n'
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
    # parser.add_option("-t", "--timestep", dest="dt", type="float", help="specify the timestep in fs")
    parser.add_option("-s", "--step", dest="step", type="int", default=-1, help="specify the timestep to extract (negative numbers are counted from the end)")
    parser.add_option("-a", "--angstrom", dest='ang', action='store_true', help="Output in Angstrom (default in Bohr)")
    parser.add_option("-q", "--append_qmin", dest='app', action='store_true', help="Append request lines from file 'QM.in'")
    parser.add_option("-i", "--initconds", dest='ini', action='store_true', help="Produce output to append to initconds instead of xyz")
    parser.add_option("-g", "--geom_veloc", dest='gv', action='store_true', help="Produce geom and veloc files instead of writing to stdout")

    (options, args) = parser.parse_args()
    if len(args) <= 1:
        parser.print_usage()
    # if options.dt is None:
    #     print("specify the timestep!!")
    #     exit(1)
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
    main(args[0], args[1], options.step, options.ang, options.app, options.gv, options.ini)
















