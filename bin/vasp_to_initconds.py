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
# usage python wigner.py [-n <NUMBER>] <MOLDEN-FILE>

import copy
import random
import sys
import datetime
import re
from optparse import OptionParser
from constants import au2fs, U_TO_AMU, ANG_TO_BOHR, MASSES_VASP, NUMBERS
import os
import numpy as np
import ase

try:
    from py4vasp import Calculation
except ImportError:
    raise ImportError("This scripts needs py4vasp, please install it.")
try:
    import mdtraj as md
except ImportError:
    raise ImportError("This scripts needs mdtraj, please install it.")

# =========================================================
# some constants
DEBUG = False

version = '4.0'
versiondate = datetime.date(2025, 4, 1)

# =========================================================

def readfile(filename):
    try:
        f = open(filename)
        out = f.readlines()
        f.close()
    except IOError:
        print('File %s does not exist!' % (filename))
        sys.exit(13)
    return out

# ======================================================================= #

def writefile(filename, content):
    # content can be either a string or a list of strings
    try:
        f = open(filename, 'w')
        if isinstance(content, list):
            for line in content:
                f.write(line)
        elif isinstance(content, str):
            f.write(content)
        else:
            print('Content %s cannot be written to file!' % (content))
            sys.exit(14)
        f.close()
    except IOError:
        print('Could not write to file %s!' % (filename))
        sys.exit(15)

# ======================================================================================================================

def try_read(l, index, typefunc, default):
    try:
        if typefunc == bool:
            return 'True' == l[index]
        else:
            return typefunc(l[index])
    except IndexError:
        return typefunc(default)
    except ValueError:
        print('Could not initialize object!')
        quit(1)

# ======================================================================================================================

class ATOM:
    def __init__(self, symb='??', num=0., coord=[0., 0., 0.], m=0., veloc=[0., 0., 0.]):
        self.symb = symb
        self.num = num
        self.coord = coord
        self.mass = m
        self.veloc = veloc
        self.Ekin = 0.5 * self.mass * sum([self.veloc[i]**2 for i in range(3)])

    def init_from_str(self, initstring=''):
        f = initstring.split()
        self.symb = try_read(f, 0, str, '??')
        self.num = try_read(f, 1, float, 0.)
        self.coord = [try_read(f, i, float, 0.) for i in range(2, 5)]
        self.mass = try_read(f, 5, float, 0.) * U_TO_AMU
        self.veloc = [try_read(f, i, float, 0.) for i in range(6, 9)]
        self.Ekin = 0.5 * self.mass * sum([self.veloc[i]**2 for i in range(3)])

    def __str__(self):
        s = '%2s % 5.1f ' % (self.symb, self.num)
        s += '% 12.8f % 12.8f % 12.8f ' % tuple(self.coord)
        s += '% 12.8f ' % (self.mass / U_TO_AMU)
        s += '% 12.8f % 12.8f % 12.8f' % tuple(self.veloc)
        return s

    def EKIN(self):
        self.Ekin = 0.5 * self.mass * sum([self.veloc[i]**2 for i in range(3)])
        return self.Ekin

    def geomstring(self):
        s = '  %2s % 5.1f % 12.8f % 12.8f % 12.8f % 12.8f' % (self.symb, self.num, self.coord[0], self.coord[1], self.coord[2], self.mass / U_TO_AMU)
        return s

    def velocstring(self):
        s = ' ' * 11 + '% 12.8f % 12.8f % 12.8f' % tuple(self.veloc)
        return s

# ======================================================================================================================

class INITCOND:
    def __init__(self, atomlist=[], eref=0., epot_harm=0.):
        self.atomlist = atomlist
        self.eref = eref
        self.Epot_harm = epot_harm
        self.natom = len(atomlist)
        self.Ekin = sum([atom.Ekin for atom in self.atomlist])
        self.statelist = []
        self.nstate = 0
        self.Epot = epot_harm
        self.molecule_format = "native"

    def addstates(self, statelist):
        self.statelist = statelist
        self.nstate = len(statelist)
        self.Epot = self.statelist[0].e - self.eref

    def init_from_file(self, f, eref, index):
        while True:
            line = f.readline()
            # if 'Index     %i' % (index) in line:
            if re.search(r'Index\\s+%i' % (index), line):
                break
        f.readline()        # skip one line, where "Atoms" stands
        atomlist = []
        while True:
            line = f.readline()
            if 'States' in line:
                break
            atom = ATOM()
            atom.init_from_str(line)
            atomlist.append(atom)
        statelist = []
        while True:
            line = f.readline()
            if 'Ekin' in line:
                break
            state = STATE()
            state.init_from_str(line)
            statelist.append(state)
        while not line == '\n' and not line == '':
            line = f.readline()
            if 'epot_harm' in line.lower():
                epot_harm = float(line.split()[1])
                break
        self.atomlist = atomlist
        self.eref = eref
        self.Epot_harm = epot_harm
        self.natom = len(atomlist)
        self.Ekin = sum([atom.Ekin for atom in self.atomlist])
        self.statelist = statelist
        self.nstate = len(statelist)
        if self.nstate > 0:
            self.Epot = self.statelist[0].e - self.eref
        else:
            self.Epot = epot_harm

    def __str__(self):
        s = 'Atoms\n'
        for atom in self.atomlist:
            s += str(atom) + '\n'
        s += 'States\n'
        for state in self.statelist:
            s += str(state) + '\n'
        s += 'Ekin      % 16.12f a.u.\n' % (self.Ekin)
        s += 'Epot_harm % 16.12f a.u.\n' % (self.Epot_harm)
        s += 'Epot      % 16.12f a.u.\n' % (self.Epot)
        s += 'Etot_harm % 16.12f a.u.\n' % (self.Epot_harm + self.Ekin)
        s += 'Etot      % 16.12f a.u.\n' % (self.Epot + self.Ekin)
        s += '\n\n'
        return s

# ======================================================================================================================

def get_center_of_mass(molecule):
    """This function returns a list containing the center of mass
of a molecule."""
    mass = 0.0
    for atom in molecule:
        mass += atom.mass
    com = [0.0 for xyz in range(3)]
    for atom in molecule:
        for xyz in range(3):
            com[xyz] += atom.coord[xyz] * atom.mass / mass
    return com


def restore_center_of_mass(ic,com_eq):
    """This function restores the center of mass for the distorted
geometry of an initial condition aligning it to the equilibrium structure COM"""

    # caluclate center of mass for initial condition of molecule
    com_distorted = get_center_of_mass(ic)
    # get difference vector and restore original center of mass
    diff = [com_eq[xyz] - com_distorted[xyz] for xyz in range(3)]
    for atom in ic:
        for xyz in range(3):
            atom.coord[xyz] += diff[xyz]


def remove_translations(ic):
    """This function calculates the movement of the center of mass
of an initial condition for a small timestep and removes this vector
from the initial condition's velocities."""
    # get center of mass at t = 0.0
    com = get_center_of_mass(ic)
    # get center of mass at t = dt = 0.01
    ic2 = copy.deepcopy(ic)
    dt = 0.01
    for atom in ic2:
        for xyz in range(3):
            atom.coord[xyz] += dt * atom.veloc[xyz]
    com2 = get_center_of_mass(ic2)
    # calculate velocity of center of mass and remove it
    v_com = [(com2[xyz] - com[xyz]) / dt for xyz in range(3)]
    for atom in ic:
        for xyz in range(3):
            atom.veloc[xyz] -= v_com[xyz]
        atom.EKIN()
    if DEBUG:
        # check if v_com now is really zero
        # get center of mass at t = 0.0
        com = get_center_of_mass(ic)
        # get center of mass at t = dt = 1.0
        ic2 = copy.deepcopy(ic)
        dt = 1.0
        for atom in ic2:
            for xyz in range(3):
                atom.coord[xyz] += dt * atom.veloc[xyz]
        com2 = get_center_of_mass(ic2)
        # calculate velocity of center of mass and remove it
        v_com = [(com2[xyz] - com[xyz]) / dt for xyz in range(3)]
        print(v_com)

def det(m):
    """This function calculates the determinant of a 3x3 matrix."""
    return m[0][0] * m[1][1] * m[2][2] + m[0][1] * m[1][2] * m[2][0] \
        + m[0][2] * m[1][0] * m[2][1] - m[0][0] * m[1][2] * m[2][1] \
        - m[0][1] * m[1][0] * m[2][2] - m[0][2] * m[1][1] * m[2][0]

def inverted(m):
    """This function calculates the inverse of a 3x3 matrix."""
    norm = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) \
        + m[0][1] * (m[1][2] * m[2][0] - m[1][0] * m[2][2]) \
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
    m_inv = [[0.0 for i in range(3)] for j in range(3)]
    m_inv[0][0] = (m[1][1] * m[2][2] - m[1][2] * m[2][1]) / norm
    m_inv[0][1] = (m[0][2] * m[2][1] - m[0][1] * m[2][2]) / norm
    m_inv[0][2] = (m[0][1] * m[1][2] - m[0][2] * m[1][1]) / norm
    m_inv[1][0] = (m[1][2] * m[2][0] - m[1][0] * m[2][2]) / norm
    m_inv[1][1] = (m[0][0] * m[2][2] - m[0][2] * m[2][0]) / norm
    m_inv[1][2] = (m[0][2] * m[1][0] - m[0][0] * m[1][2]) / norm
    m_inv[2][0] = (m[1][0] * m[2][1] - m[1][1] * m[2][0]) / norm
    m_inv[2][2] = (m[0][1] * m[2][0] - m[0][0] * m[2][1]) / norm
    m_inv[2][2] = (m[0][0] * m[1][1] - m[0][1] * m[1][0]) / norm
    return m_inv

def matmul(m1, m2):
    """This function multiplies two NxN matrices m1 and m2."""
    # get dimensions of resulting matrix
    n = len(m1)
    # calculate product
    result = [[0.0 for i in range(n)] for j in range(n)]
    for i in range(n):
        for j in range(n):
            for k in range(n):
                result[i][j] += m1[i][k] * m2[k][j]
    return result

def cross_prod(a, b):
    """This function calculates the cross product of two
3 dimensional vectors."""
    result = [0.0 for i in range(3)]
    result[0] = a[1] * b[2] - b[1] * a[2]
    result[1] = a[2] * b[0] - a[0] * b[2]
    result[2] = a[0] * b[1] - b[0] * a[1]
    return result

def linmapping(lm, y):
    z = [0.0 for i in range(3)]
    z[0] = lm[0][0] * y[0] + lm[0][1] * y[1] + lm[0][2] * y[2]
    z[1] = lm[1][0] * y[0] + lm[1][1] * y[1] + lm[1][2] * y[2]
    z[2] = lm[2][0] * y[0] + lm[2][1] * y[1] + lm[2][2] * y[2]
    return z

def remove_rotations(ic):
    # copy initial condition object
    ictmp = copy.deepcopy(ic)
    # move center of mass to coordinates (0, 0, 0)
    com = get_center_of_mass(ic)
    for atom in ictmp:
        for xyz in range(3):
            atom.coord[xyz] -= com[xyz]
    # calculate moment of inertia tensor
    I = [[0.0 for i in range(3)] for j in range(3)]
    for atom in ictmp:
        I[0][0] += atom.mass * (atom.coord[1]**2 + atom.coord[2]**2)
        I[1][1] += atom.mass * (atom.coord[0]**2 + atom.coord[2]**2)
        I[2][2] += atom.mass * (atom.coord[0]**2 + atom.coord[1]**2)
        I[0][1] -= atom.mass * atom.coord[0] * atom.coord[1]
        I[0][2] -= atom.mass * atom.coord[0] * atom.coord[2]
        I[1][2] -= atom.mass * atom.coord[1] * atom.coord[2]
    I[1][0] = I[0][1]
    I[2][0] = I[0][2]
    I[2][1] = I[1][2]
    if det(I) > 0.01:  # checks if I is invertible
        ch = matmul(I, inverted(I))
        # calculate angular momentum
        ang_mom = [0.0 for i in range(3)]
        for atom in ictmp:
            mv = [0.0 for i in range(3)]
            for xyz in range(3):
                mv[xyz] = atom.mass * atom.veloc[xyz]
            L = cross_prod(mv, atom.coord)
            for xyz in range(3):
                ang_mom[xyz] -= L[xyz]
        # calculate angular velocity
        ang_vel = linmapping(inverted(I), ang_mom)
        for i, atom in enumerate(ictmp):
            v_rot = cross_prod(ang_vel, atom.coord)  # calculate rotational velocity
            for xyz in range(3):
                ic[i].veloc[xyz] -= v_rot[xyz]  # remove rotational velocity
    else:
        print('WARNING: moment of inertia tensor is not invertible')

# ======================================================================================================================

def ask_for_masses():
    print('''
Option -m used, please enter non-default masses:
+ number mass           add non-default mass <mass> for atom <number> (counting starts at 1)
- number                remove non-default mass for atom <number> (default mass will be used)
show                    show non-default atom masses
end                     finish input for non-default masses
''')
    MASS_LIST = {}
    while True:
        line = input()
        if 'end' in line:
            break
        if 'show' in line:
            s = '-----------------------\nAtom               Mass\n'
            for i in MASS_LIST:
                s += '% 4i %18.12f\n' % (i, MASS_LIST[i])
            s += '-----------------------'
            print(s)
            continue
        if '+' in line:
            f = line.split()
            if len(f) < 3:
                continue
            try:
                num = int(f[1])
                mass = float(f[2])
            except ValueError:
                continue
            MASS_LIST[num] = mass * U_TO_AMU
            continue
        if '-' in line:
            f = line.split()
            if len(f) < 2:
                continue
            try:
                num = int(f[1])
            except ValueError:
                continue
            del MASS_LIST[num]
            continue
    return MASS_LIST

# ======================================================================================================================
def get_mass(symb, number, MASSLIST):
    if number in MASSLIST:
        return MASSLIST[number]
    else:
        try:
            return MASSES_VASP[symb]
        except KeyError:
            print('No default mass for atom %s' % (symb))
            sys.exit(1)

# ======================================================================================================================

def random_initcond(INFOS):
    '''
    Generating initial conditions for SHARC by random sampling of the last specified frames of a VASP MD trajectory.
    '''
    #Random indexes for sampling Ninit initial conditions over last Nframes of the trajectory. First index is always 0 for reference geometry
    index=[0]+random.sample(range(1, len(INFOS["xyz"])), INFOS["NINIT"]) 
    INFOS["veloc"]=INFOS["veloc"][index] #Sampled velocities
    INFOS["xyz"]=INFOS["xyz"][index] #Sampled coordinates
    print("Random sampling performed")
    
    return

def every_initcond(INFOS):
    '''
    Generating initial conditions for SHARC by sampling one initial condition every NFRAMES/n frame.
    In total n initial conditions out of the processed frames are generated.
    '''
   
    print(f"Sampling one snapshot every {len(INFOS["xyz"])//INFOS["NINIT"]}")
    # Compute equally spaced indices +0 ensures the first frame is always included as reference.
    index = [0] + [int(round(i)) for i in np.linspace(1,  len(INFOS["xyz"])- 1, INFOS["NINIT"])]
    # Use the indices to slice your arrays
    INFOS["veloc"] = INFOS["veloc"][index]  # Sampled velocities
    INFOS["xyz"]   = INFOS["xyz"][index]    # Sampled coordinates

    return

def get_coords(INFOS):
    
    ATOMS = INFOS["elements"]
    natom = len(ATOMS)
    # initialize arrays
    ic_list = []
    igeom = 0
    # go through the data
    for xyz,velocity in zip(INFOS['xyz'],INFOS["veloc"]):
        atomlist = []
        for iatom in range(natom):
            symb = ATOMS[iatom]
            num = NUMBERS[symb]
            vel = [0., 0., 0.]
            mass = get_mass(symb,iatom+1,INFOS["masslist"])
            atomlist.append(ATOM(symb, num, xyz[iatom], mass, vel))
            atomlist[iatom].veloc = velocity[iatom]
            atomlist[iatom].EKIN()
        igeom += 1
        if igeom == 1:
            INFOS['COM_EQ']=get_center_of_mass(atomlist) #Computing COM of eq.structure for later alignment by restore_center_of_mass
        if not INFOS['KTR'] and igeom > 1:
            restore_center_of_mass(atomlist,INFOS['COM_EQ'])
            remove_translations(atomlist)
            remove_rotations(atomlist)
        if igeom == 1: #reference geometry if equilibrium structure, leading to ICOND_00000
            molecule = INITCOND(atomlist, 0., 0.)
        else:
            ic_list.append(INITCOND(atomlist, 0., 0.))

    return molecule, ic_list


# ======================================================================================================================
# ======================================================================================================================
# ======================================================================================================================

def create_initial_conditions_string(molecule, ic_list, eref=0.0):
    """This function converts an list of initial conditions into a string."""
    ninit = len(ic_list)
    natom = ic_list[0].natom
    representation = 'None'
    # eref
    eharm = 0.
    # for mode in modes:
    # eharm+=mode['freq']*0.5
    string = '''SHARC Initial conditions file, version %s
Ninit     %i
Natom     %i
Repr      %s
Eref      %18.10f
Eharm     %18.10f

Equilibrium
''' % (version, ninit, natom, representation, eref, eharm)
    for atom in molecule.atomlist:
        string += str(atom) + '\n'
    string += '\n\n'

    for i, ic in enumerate(ic_list):
        string += 'Index     %i\n%s' % (i + 1, str(ic))
   
    return string

# ======================================================================================================================


def make_dyn_file(ic_list, filename):
    fl = open(filename, 'w')
    string = ''
    for i, ic in enumerate(ic_list):
        string += '%i\n%i\n' % (ic.natom, i)
        for atom in ic.atomlist:
            string += '%s' % (atom.symb)
            for j in range(3):
                string += ' %f' % (atom.coord[j] / ANG_TO_BOHR)
            string += '\n'
    fl.write(string)
    fl.close()

# ======================================================================================================================

def main():
    '''Main routine'''

    # command line option setup
    usage ='''
    vasp_to_initconds.py MD_folder --flags

    MD_folder -> Path to directory that contains VASP MD.

    This script generate a set of initial conditions (initconds file) for SHARC-VASP dynamics reading a MD trajectory computed with VASP.
    It analyzes the last N frames specified by the user with the flag -f. By default all frames are processed.
    py4vasp and mdtraj python packages have to be installed in the user's python environment. 
    Two options are supported for generating initial conditions:
    1) Random sampling of n initial conditions from the last N frames processed (--random).
    2) Sampling of n initial conditions from the last N frames by diving the total number of processed frames by n. 
       One snapshot is taken for each subgroup, so every NFRAMES/n. (--every).
    '''
    description = ''
    parser = OptionParser(usage=usage, description=description)
    parser.add_option('-f', dest='frames', type=int, nargs=1, default=None, help="N. of last frames to read from trajectory. (Default all)")
    parser.add_option('-n', dest='init', type=int, nargs=1, default=10, help="N. of initial conditions to generate (Default 10)")
    parser.add_option('--eq', dest='eq', type=str, nargs=1, default=None, help="Path to POSCAR of equilibrium structure, for ICOND_00000")
    parser.add_option('--random', dest='random', action='store_true', help="Select n random initial conditions from the input frames")
    parser.add_option('--every', dest='every', action='store_true', help="Select one initial condition every NFRAMES/n")

    parser.add_option('-o', dest='o', type=str, nargs=1, default='initconds', help="Output filename (string, default=""initconds"")")
    parser.add_option('-x', dest='X', action='store_true', help="Generate a xyz file with the sampled geometries in addition to the initconds file")
    parser.add_option('-m', dest='m', action='store_true', help="Enter non-default atom masses")
    parser.add_option('--keep_trans_rot', dest='KTR', action='store_true', help="Keep translational and rotational components")

    # arg processing
    (options, args) = parser.parse_args()
    if len(args) == 0:
        print(usage)
        quit(1)

    # options
    INFOS = {}
    INFOS['VASPDIR'] = args[0]

    # Checking INCAR for MD run.
    with open(os.path.join(INFOS["VASPDIR"],"INCAR"), 'r') as file:
        INCAR=file.read()
        
         #Getting timestep from INCAR, not so relevant for this script.
        pattern=rf'\s*POTIM\s*=\s*(\d*\.\d*).*\n'
        if re.search(pattern,INCAR,re.IGNORECASE): 
            timestep=float(re.search(pattern,INCAR).group(1))
            INFOS['timestep'] = timestep
        else:
            print("No timestep (POTIM) was found in the INCAR. Not a big issue here, are you sure though you have run an MD dynamics with VASP?")
        
        #Checking if VELOCITY=.true. is in INCAR. Must be there to read velocities!!
        pattern = rf'\s*VELOCITY\s*=\s*(\.true\.)\s*\n'
        if re.search(pattern,INCAR,re.IGNORECASE): 
            pass
        else:
            print("ERROR: No VELOCITY=.TRUE. was found in the INCAR. Atomic velocities from MD cannot be read. Please redo the MD run with this flag.")
            sys.exit(1)

    INFOS['outfile'] = options.o
    INFOS['masslist'] = {}
    if options.m:
        INFOS['masslist'] = ask_for_masses()
    INFOS['KTR'] = options.KTR
    INFOS['NFRAMES']=options.frames
    INFOS['NINIT']=options.init
    INFOS['EQ']=options.eq
    if INFOS['EQ'] is None:
        print('ERROR: please specify the path to the POSCAR of the equilibrium structure. This is necessary for ICOND_00000 and setting up the other ICOND_* properly')
        sys.exit(1) 
    else:
        if not os.path.isfile(INFOS['EQ']):
            print('ERROR: Wrong path for equilibrium structure POSCAR file. File does not exist.')
            sys.exit(1) 

    INFOS['random']=options.random
    INFOS['every']=options.every

    #Reading of initial VASP MD trajectroy
    NM_TO_BOHR=ANG_TO_BOHR*10
    calc=Calculation.from_path(INFOS['VASPDIR'])
    if INFOS['NFRAMES'] is None:
        traj=calc.structure[:].to_mdtraj()
        data=calc.velocity[:].read()
    else:
        traj=calc.structure[-INFOS['NFRAMES']:].to_mdtraj()
        data=calc.velocity[-INFOS['NFRAMES']:].read()

    #Reading equilibrium structure (POSCAR)
    eq_poscar=ase.io.read(INFOS['EQ']) 
    eq_coords=eq_poscar.get_positions()*ANG_TO_BOHR #ASE gives back Angstrom!
    eq_vel=np.zeros(shape=traj.xyz.shape[-2:])
    #Adding one dimension only for Vstacking below
    eq_coords=eq_coords[None,:,:]
    eq_vel=eq_vel[None,:,:]
    #Updating dictionary with trajectory information
    INFOS["veloc"]=np.vstack((eq_vel,data["velocities"]*ANG_TO_BOHR*au2fs)) #VASP velocities are in Ang./fs. First is equilibrium geometry at 0 K.
    INFOS["xyz"]=np.vstack((eq_coords,traj.xyz*NM_TO_BOHR)) #Transforming coordinates to Bohr upon MDTraj reading. MDTraj works with nm. First is equilibrium geometry.
    INFOS["lattice_vectors"]=data["structure"]["lattice_vectors"] #Lattice vectors. Do not change with NVT simulations.
    INFOS["elements"]=data["structure"]["elements"] #List of atom labels
    
    print('''Initial condition generation started...
VASPDIR with MD data                             = "%s"
OUTPUT file                                      = "%s"
POSCAR of equilibrium structure                  = "%s"
Number of initial conditions to generate         = %i
Number of MD frames read                         = %i''' % (INFOS['VASPDIR'],INFOS['outfile'],INFOS['EQ'],INFOS["NINIT"],len(INFOS["veloc"])-1))

    #Either clustering or random sampling below
    if INFOS['random']:
        random_initcond(INFOS)
    elif INFOS['every']:
        every_initcond(INFOS)

    # Initial conditions writing
    molecule, ic_list = get_coords(INFOS)
    outfile = open(INFOS['outfile'], 'w')
    outstring = create_initial_conditions_string(molecule, ic_list)
    outfile.write(outstring)
    outfile.close()
    if options.X:
        make_dyn_file(ic_list, options.o + '.xyz')
    # save the shell command
    command = 'python ' + ' '.join(sys.argv)
    f = open('KEYSTROKES.vasp_to_initconds', 'w')
    f.write(command)
    f.close()

# ======================================================================================================================

if __name__ == '__main__':
    main()
