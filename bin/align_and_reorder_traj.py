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

# Script for obtaining geometries where surface hops occured
#
# usage: python crossing.py

import sys
import re
import os
import shutil
import subprocess as sp
import datetime
from time import perf_counter_ns

from netCDF4 import Dataset
import numpy as np
import numpy.ma as ma


from utils import question, readfile
from kabsch import kabsch_w
from constants import U_TO_AMU, MASSES, BOHR_TO_ANG, au2fs

# =========================================================0
# some constants
DEBUG = False

# ======================================================================= #

version = '2.1'
versiondate = datetime.date(2019, 9, 1)

# ======================================================================================================================
# ======================================================================================================================
# ======================================================================================================================


def displaywelcome():
    print('Script for hopping geometry extraction started...\n')
    string = '\n'
    string += '  ' + '=' * 80 + '\n'
    string += '||' + '{:^80}'.format('') + '||\n'
    string += '||' + '{:^80}'.format('Aligning coordinates/velocities for SHARC trajectories') + '||\n'
    string += '||' + '{:^80}'.format('') + '||\n'
    string += '||' + '{:^80}'.format('Author: Sebastian Mai') + '||\n'
    string += '||' + '{:^80}'.format('') + '||\n'
    string += '||' + '{:^80}'.format('Version:' + version) + '||\n'
    string += '||' + '{:^80}'.format(versiondate.strftime("%d.%m.%y")) + '||\n'
    string += '||' + '{:^80}'.format('') + '||\n'
    string += '  ' + '=' * 80 + '\n\n'
    string += '''
This script reads output.dat.nc and sharc_traj_xyz.nc files to produce a set of
NetCDF output files with ailgned geometries/velocities from each time step
  '''
    print(string)

# ======================================================================================================================
# ======================================================================================================================
# ======================================================================================================================

def read_xyz(xyz):
    lines = readfile(xyz)
    natom = int(lines[0])
    geom = []
    symbols = []
    for line in lines[2:]:
        s, x, y, z = line.split()
        geom.append([float(x), float(y), float(z)])
        symbols.append(s)
    if len(geom) == natom:
        return np.array(geom), symbols
    else:
        raise ValueError(f"Number of atoms ({len(geom)}) does not match the specified count ({natom}).")


# ======================================================================================================================

global KEYSTROKES
old_question = question
def question(question, typefunc, default=None, autocomplete=True, ranges=False):
    return old_question(
        question=question, typefunc=typefunc, KEYSTROKES=KEYSTROKES, default=default, autocomplete=autocomplete, ranges=ranges
    )

def open_keystrokes():
    global KEYSTROKES
    KEYSTROKES = open('KEYSTROKES.tmp', 'w')


def close_keystrokes():
    KEYSTROKES.close()
    shutil.move('KEYSTROKES.tmp', 'KEYSTROKES.align_and_reorder_traj')

# ======================================================================================================================
# ======================================================================================================================
# ======================================================================================================================


def get_general():
    ''''''

    INFOS = {}

    # Path to trajectories
    print('{:-^60}'.format('Paths to trajectories'))
    print('\nPlease enter the paths to all directories containing the "TRAJ_0XXXX" directories.\nE.g. S_2 and S_3. \nPlease enter one path at a time, and type "end" to finish the list.')
    count = 0
    paths = []
    while True:
        path = question('Path: ', str, 'end')
        if path == 'end':
            if len(paths) == 0:
                print('No path yet!')
                continue
            print('')
            break
        path = os.path.expanduser(os.path.expandvars(path))
        if not os.path.isdir(path):
            print('Does not exist or is not a directory: %s' % (path))
            continue
        if path in paths:
            print('Already included.')
            continue
        ls = os.listdir(path)
        print(ls)
        for i in ls:
            if 'TRAJ' in i:
                count += 1
        print('Found %i subdirectories in total.\n' % count)
        paths.append(path)
    INFOS['paths'] = paths
    print('Total number of subdirectories: %i\n\n' % (count))

    # get valid trajectories
    forbidden = ['crashed', 'running', 'dead', 'dont_analyze']
    valid_paths = []
    print('Checking the directories...')
    for idir in INFOS['paths']:
        ls = sorted(os.listdir(idir))
        for itraj in ls:
            path = os.path.join(idir,itraj)
            if not os.path.isdir(path):
                continue
            if not re.match(r'^TRAJ_\d{5}$', itraj):
                continue
            lstraj = os.listdir(path)
            valid = True
            for i in lstraj:
                if i.lower() in forbidden:
                    valid = False
                    break
            if not valid:
                continue
            valid_paths.append(path)
    print('Number of trajectories: %i' % (len(valid_paths)))
    if len(valid_paths) == 0:
        print('No valid trajectories found, exiting...')
        sys.exit(0)
    INFOS['valid_paths'] = valid_paths


    # decide on file
    possible = set(['output.dat.nc', 'sharc_traj_xyz.nc'])
    ls = set(os.listdir(INFOS['valid_paths'][0]))
    can_do = possible & ls
    print('{:-^60}'.format('Coordinate data file'))
    print("\nFrom which file do you want to read out the coordinate data?")
    print("Possible files: %s\n" % can_do)
    if 'sharc_traj_xyz.nc' in can_do:
        default = 'sharc_traj_xyz.nc' 
    else:
        default = 'output.dat.nc'
    while True:
        file = question("File?", str, default = default )
        if file in can_do:
            break
        else:
            print("File not possible")
    INFOS['coord_files'] = file


    # reference geometry
    print('{:-^60}'.format('Alignment and geometry file'))
    print("\nThis script can align the geometries in different manners.")
    INFOS['align'] = question("Do you want to align?", bool, default = True)
    if INFOS['align']:
        # ref geometry
        while True:
            try:
                file = question("\nPath to reference geometry (xyz format)?", str)
                xyz, sym = read_xyz(file)
            except ValueError:
                print("Incorrect number of atoms in file.")
                continue
            except KeyboardInterrupt:
                raise
            except:
                print("Could not read from file.")
                continue
            break
        INFOS['ref_xyz'] = xyz
        INFOS['ref_sym'] = sym


        # ref geometry atom mapping
        default = [ i+1 for i in range(len(xyz)) ]
        INFOS['ref_map'] = question("\nTo which atoms in the trajectories do the reference atoms correspond?", int, default=default)


        # perspective
        print("\nThis script can align the different time steps in two ways:")
        print("- Molecule's perspective: Alignment is done for each time step individually")
        print("- Solvent's perspective: Alignment is done for each time step, but using the shift/rotation matrix of the first step")
        INFOS['mol_persp'] = question("Do you want to use the Molecule's perspective?", bool, default = True)


    # coord or veloc
    print('{:-^60}'.format('Output files'))
    print("\nThis script can write coordinates and velocities.")
    INFOS['write_coord'] = question("Do you want to write coordinates?", bool, default=True)
    INFOS['write_veloc'] = question("Do you want to write velocities?", bool, default=False)


    return INFOS

# ======================================================================================================================
# ======================================================================================================================
# ======================================================================================================================

# def kabsch_w(a: np.ndarray, b: np.ndarray, weights) -> np.ndarray:
#     if a.ndim != 2 or a.shape[-1] != 3:
#         raise ValueError("Expected input `a` to have shape (N, 3), " "got {}".format(a.shape))
#     if b.ndim != 2 or b.shape[-1] != 3:
#         raise ValueError("Expected input `b` to have shape (N, 3), " "got {}.".format(b.shape))

#     if a.shape != b.shape:
#         raise ValueError(
#             "Expected inputs `a` and `b` to have same shapes" ", got {} and {} respectively.".format(a.shape, b.shape)
#         )

#     weights = np.asarray(weights)
#     if weights.ndim != 1 or weights.shape[0] != a.shape[0]:
#         raise ValueError(f"Expected input `weights` to have shape {a.shape[0]}")
#     # shift to centroid
#     M = sum(weights)
#     a_s = weights @ a / M
#     b_s = weights @ b / M
#     B = np.einsum("j,ji->ji", weights, a - a_s)
#     B = np.einsum("ji,jk->ik", B, b - b_s)
#     u, s, vT = np.linalg.svd(B)

#     B = u @ vT
#     # Correct improper rotation if necessary (as in Kabsch algorithm)
#     if np.linalg.det(u @ vT) < 0:
#         s[-1] = -s[-1]
#         u[:, -1] = -u[:, -1]
#         B = u @ vT

#     if s[1] + s[2] < 1e-16 * s[0]:
#         print("Optimal rotation is not uniquely or poorly defined " "for the given sets of vectors.")

#     return B, a_s, b_s


# ======================================================================================================================


def do_calc(INFOS):

    # make list of files and get dimensions
    natoms = set()
    nsteps = set()
    filename_list_and_steps = {}
    for ipath in INFOS['valid_paths']:
        f = os.path.join(ipath, INFOS['coord_files'])
        if os.path.isfile(f):
            with Dataset(f) as file:
                ns, na, nx = file.variables["geom"].shape
            filename_list_and_steps[f] = (ns, na, nx)
            natoms.add(na)
            nsteps.add(ns)

    # get dimensions
    if len(natoms)>1:
        print("Number of atoms is not consistent!")
        sys.exit()
    if len(nsteps)>1:
        print('WARNING: Number of time steps inconsistent. Later files will have fewer trajectories.')
    natom = next(iter(natoms))
    nstep = max(nsteps)
    nfile = len(filename_list_and_steps)

    # prepare reference in numpy
    T_mats_0 = np.repeat(np.eye(3)[np.newaxis, :, :], nfile, axis=0) # rotation matrices initialized to unit matrices
    com_coord_0 = np.zeros((nfile, 3))  # COM shift
    com_ref_0 = np.zeros((nfile, 3))    # COM shift reference
    masses = [MASSES[s.title()] / U_TO_AMU for s in INFOS['ref_sym']]
    ref_geom = INFOS['ref_xyz']
    ref_map = np.array(INFOS['ref_map']) - 1

    # timing
    start_time = perf_counter_ns()
    last__time = perf_counter_ns()

    # main loop
    for istep in range(nstep):

        # progress
        print('\n=== Time step %i ===' % istep)

        # main containers
        all_geom = np.zeros((nfile, natom, 3), dtype=float)

        # go through the files
        sys.stdout.write("Files done (of %i): " % nfile)
        for itraj, filename in enumerate(sorted(filename_list_and_steps.keys())):
            sys.stdout.write('#')
            # print("-> File: %s [%i]" % (filename, istep))
            with Dataset(filename) as f:
                # read and convert coordinates
                geom = f.variables["geom"]
                geom = np.array(geom[istep], dtype=float).reshape(3, natom)
                geom = np.einsum("xa->ax", geom) * BOHR_TO_ANG # convert to angstrom to align and write

                # make alignment data
                if INFOS['align']:
                    if istep == 0 or INFOS['mol_persp']:
                        T_mats_0[itraj, ...], com_ref_0[itraj, ...], com_coord_0[itraj, ...] = kabsch_w(ref_geom, geom[ref_map, :], masses)

                # align
                all_geom[itraj, ...] = (geom - com_coord_0[itraj, None, ...]) @ T_mats_0[itraj, ...].T + com_ref_0[itraj, None, ...]
        sys.stdout.write('\n')

        if INFOS['write_coord']:
            # all geometries acquired, now write them
            if INFOS['mol_persp']:
                out_filename = f"frame_coord_mol_pers_{istep:05d}.nc"
            else:
                out_filename = f"frame_coord_sol_pers_{istep:05d}.nc"
            print(" => Writing File: %s" % out_filename)
            with Dataset(out_filename, "w", format="NETCDF3_64BIT_OFFSET") as out:
                out.title = "SHARCTRAJ to AMBERTRAJ"
                out.application = "AMBER"
                out.programVersion = "V5.1.0"
                out.Conventions = "AMBER"
                out.ConventionVersion = "1.0"
                frame = out.createDimension("frame", None)
                spatial = out.createDimension("spatial", 3)
                atom = out.createDimension("atom", natom)
                v_spatial = out.createVariable("spatial", "|S1", ("spatial",))
                v_spatial[:] = ma.array([b"x", b"y", b"z"], mask=False, fill_value=b"N/A", dtype="|S1")
                v_coord = out.createVariable("coordinates", "f4", ("frame", "atom", "spatial"))
                v_coord.units = "angstrom"
                v_coord[:] = all_geom[...]



        # if velocities are needed, go through the files and read them
        # rotation matrices are already there, so no need to touch geometries again
        if INFOS['write_veloc']:

            # reuse the same container
            all_geom = np.zeros((nfile, natom, 3), dtype=float)

            # go through the files
            sys.stdout.write("Files done (of %i): " % nfile)
            for itraj, filename in enumerate(sorted(filename_list_and_steps.keys())):
                sys.stdout.write('#')
                # print("-> File: %s [%i] for velocity" % (filename, istep))
                with Dataset(filename) as f:
                    # read and convert coordinates
                    veloc = f.variables["veloc"]
                    veloc = np.array(veloc[istep], dtype=float).reshape(3, natom)
                    veloc = np.einsum("xa->ax", veloc) * BOHR_TO_ANG * 1000 / au2fs # convert to angstrom/picosecond

                    # align
                    all_geom[itraj, ...] = veloc @ T_mats_0[itraj, ...].T # COM shifts are not applied to velocities
            sys.stdout.write('\n')
            
            # all geometries acquired, now write them
            if INFOS['mol_persp']:
                out_filename = f"frame_veloc_mol_pers_{istep:05d}.nc"
            else:
                out_filename = f"frame_veloc_sol_pers_{istep:05d}.nc"
            print(" => Writing File: %s" % out_filename)
            with Dataset(out_filename, "w", format="NETCDF3_64BIT_OFFSET") as out:
                out.title = "SHARCTRAJ to AMBERTRAJ"
                out.application = "AMBER"
                out.programVersion = "V5.1.0"
                out.Conventions = "AMBER"
                out.ConventionVersion = "1.0"
                frame = out.createDimension("frame", None)
                spatial = out.createDimension("spatial", 3)
                atom = out.createDimension("atom", natom)
                v_spatial = out.createVariable("spatial", "|S1", ("spatial",))
                v_spatial[:] = ma.array([b"x", b"y", b"z"], mask=False, fill_value=b"N/A", dtype="|S1")
                v_veloc = out.createVariable("velocities", "f4", ("frame", "atom", "spatial"))
                v_veloc.units = "angstrom/picosecond"
                v_veloc[:] = all_geom[...]

        # timings
        final_time = perf_counter_ns()
        timing = (final_time - last__time) * 1e-9
        nfiles = 1 + int(INFOS['write_veloc'])
        print(f"Wall time: {timing: 15.4}sec ({timing/nfile/nfiles: 15.4}sec per file access)")
        last__time = final_time
    
    # total timing
    timing = (final_time - start_time) * 1e-9
    nfiles = 1 + int(INFOS['write_veloc'])
    print(f"Total wall time: {timing: 15.4}sec ({timing/nfile/nstep/nfiles: 15.4}sec per file access)")


# ======================================================================================================================
# ======================================================================================================================
# ======================================================================================================================


# ======================================================================================================================
# ======================================================================================================================
# ======================================================================================================================

def main():
    '''Main routine'''

    usage = '''
python align_and_reorder_traj.py

This interactive program reads an Amber NetCDF file and computes a 
kernel density estimation of certain atoms.
It writes the density to an dx file
'''

    description = ''
    displaywelcome()
    open_keystrokes()

    INFOS = get_general()

    print('{:#^60}\n'.format('Full input'))
    for item in INFOS:
        if not item == 'statemap':
            print(item, ' ' * (25 - len(item)), INFOS[item])
    print('')
    calc = question('Do you want to do the specified analysis?', bool, True)
    print('')

    if calc:
        do_calc(INFOS)

    close_keystrokes()


# ======================================================================================================================

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\nCtrl+C makes me a sad SHARC ;-(\n')
        quit(0)















