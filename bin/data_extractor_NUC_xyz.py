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

from constants import au2fs, ANG_TO_BOHR, U_TO_AMU, IAn2AName
from utils import readfile
from setup_from_prmtop import expand_str_to_list




# =========================================================
# some constants
DEBUG = False

version = '4.0'
versiondate = datetime.date(2025, 4, 1)








def main(geomfile, ncfile, qm_atoms):
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

    # Convert 1-based QM atom indices to 0-based
    qm_atom_indices = sorted(set(i - 1 for i in qm_atoms if 1 <= i <= natom))
    if qm_atoms:
        sys.stderr.write(f"Restricting XYZ output to atoms: {qm_atoms}\n")
    if qm_atom_indices:
        symbols = [symbols[i] for i in qm_atom_indices]

    # look into NetCDF file
    with Dataset(ncfile) as dat:

        # get dimensions and data
        nstep, natom2, nspat = dat.variables["geom"].shape
        geom_rst = dat.variables["geom"]
        # veloc_rst = dat.variables["veloc"]
        sys.stderr.write("nframe from NetCDF: "+str(nstep)+'\n')
        sys.stderr.write("natom  from NetCDF: "+str(natom2)+'\n')
        sys.stderr.write("ndim   from NetCDF: "+str(nspat)+'\n')
        if nstep == 0:
            print("No steps found. Data can only extracted from finished trajectories.")
            sys.exit(0)

        # figure out which step we want
        geom  = np.array( geom_rst[:, :, :], dtype=np.float32).reshape(nstep, 3, natom)
        # veloc = np.array(veloc_rst[:, :, :], dtype=np.float32).reshape(nstep, 3, natom)
        geom  = np.einsum("sxa->sax", geom)
        # veloc = np.einsum("sxa->sax", veloc)

        # Filter atoms if needed
        natom_print = natom
        if qm_atom_indices:
            geom = geom[:, qm_atom_indices, :]
            natom_print = len(qm_atom_indices)


        # print xyz
        string = ''
        for istep in range(nstep):
            string += '%i\nStep %i\n' % (natom_print,istep)
            factor = 1. / ANG_TO_BOHR
            for s, c in zip(symbols, geom[istep,:,:] * factor):
                string += f"{s:5s} {c[0]: 12.8f} {c[1]: 12.8f} {c[2]: 12.8f} \n"

        print(string)



if __name__ == "__main__":

    parser = OptionParser()
    parser.add_option(
    "-q",
    "--qm-list",
    type="str",
    default="",
    dest="qm_list",
    help="Specify 'QM' atoms as list starting from 1 (e.g. 1~3 5 8~12 20)\ndefault=\"\"",
    )

    (options, args) = parser.parse_args()
    if len(args) <= 1:
        parser.print_usage()
        print("Required positional arguments: <geom> <output.dat.nc>")
        quit()
    qm_atoms = expand_str_to_list(options.qm_list)
    main(args[0], args[1], qm_atoms)
















