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

# Interactive script for the setup of dynamics calculations for SHARC
#
# usage: python setup_traj.py

import math
import sys
import re
import os
import numpy as np
import stat
import shutil
import datetime
import random
import json
from optparse import OptionParser
from socket import gethostname
import cProfile
import subprocess as sp
from logger import log
from scipy.spatial.transform import Rotation as R
# import factory
from utils import question, itnmstates, expand_path, readfile, link
from numba import njit
from constants import IToMult, U_TO_AMU, HARTREE_TO_EV
from SHARC_INTERFACE import SHARC_INTERFACE
from SHARC_QMOUT import SHARC_QMOUT
from qmout import QMout


# =========================================================0
PI = math.pi
# log.root.setLevel(log.DEBUG)

version = "4.0"
versionneeded = [0.2, 1.0, 2.0, 2.1, float(version)]
versiondate = datetime.date(2019, 9, 1)


global KEYSTROKES
old_question = question
def question(question, typefunc, default=None, autocomplete=True, ranges=False):
    return old_question(
        question=question, typefunc=typefunc, KEYSTROKES=KEYSTROKES, default=default, autocomplete=autocomplete, ranges=ranges
    )


# ======================================================================= #

# General NAMD methods in SHARC: TSH and SCP
Method={
    1: {'name':        'tsh',
        'description': 'Trajectory surface hopping dynamics using single surface potential'
        },
    2: {'name':        'scp',
        'description': 'Semi-classical Ehrenfest dynamics using self-consistent potential'
        }
}

# Couplings to propagate the el-TDSE: 
Couplings = {
    1: {"name": "nacdt",   "description": "DDT     =  < a|d/dt|b >            Hammes-Schiffer-Tully scheme", "required": ["nacdt"]},
    2: {"name": "nacdr",   "description": "DDR     =  < a|d/dR|b >            Original Tully scheme       ", "required": ["nacdr"]},
    3: {'name': "ktdc",    "description": "ktdc    = sqrt(D2(dV)/dt2/(dV))/2  Curvature Driven TDC scheme ", "required": []},
    4: {"name": "overlap", "description": "overlap = < a(t0)|b(t) >           Local Diabatization scheme  ", "required": ["overlap"]},
}

# Nonadiabatic coupling-like vectors to propagate nuclei with SCP (not relevant for TSH)
Neom = {
    1: {'name':        'ddr',
        'description': 'Nuclear EOM propagation with NACdR   ',
        "required": ["nacdr"],
        },
    2: {'name':        'gdiff',
        'description': 'Nuclear EOM propagation effective NAC based on gradient difference    ',
        "required": []
        }
}

# Velocity-Verlet integrator to be used
Integrator={
    1: {'name':        'avv',
        'description': 'adaptive timestep Velocity-Verlet integrator',
        'forbidden': ["overlap", "phases"]
        },
    2: {'name':        'fvv',
        'description': 'fixed timestep Velocity-Verlet integrator',
        'forbidden': []
        }
}

# Gradient mixing protocol for TSH in diagonal basis or for SCP
GradCorrect={
    1: {'name':        'none',
        'description': 'mixed gradients are calculated as linear combination of MCH gradients only',
        'required':   []
        },
    2: {'name':        'ngt',
        'description': 'mixed gradients are calculated by correction of MCH gradients with non-adiabatic coupling vector',
        'required':   ['nacdr']
        },
    3: {'name':        'tdh',
        'description': 'mixed gradients are calculated by rescaling of the MCH gradients according to time derivatives in diagonal and MCH representations',
        'required':   []  # TODO: what is required?
        }
}

# How to rescale the kinetic energy vector after a hop. 
# TODO: Does this apply to SCP or only to TSH?
EkinCorrect={
    1: {'name':             'none',
        'description':      'Do not conserve total energy. Hops are never frustrated.',
        'description_refl': 'Do not reflect at a frustrated hop.',
        'required':   []
        },
    2: {'name':             'parallel_vel',
        'description':      'Adjust kinetic energy by rescaling the velocity vectors. Often sufficient.',
        'description_refl': 'Reflect the full velocity vector.',
        'required':   []
        },
    3: {'name':             'parallel_pvel',
        'description':      'Adjust kinetic energy only with the component of the velocity vector along the vibrational velocity vector.',
        'description_refl': 'Reflect the vibrational velocity vector.',
        'required':   []
        },
    4: {'name':             'parallel_nac',
        'description':      'Adjust kinetic energy only with the component of the velocity vector along the non-adiabatic coupling vector.',
        'description_refl': 'Reflect only the component of the velocity vector along the non-adiabatic coupling vector.',
        'required':   ['nacdr']
        },
    5: {'name':             'parallel_diff',
        'description':      'Adjust kinetic energy only with the component of the velocity vector along the gradient difference vector.',
        'description_refl': 'Reflect only the component of the velocity vector along the gradient difference vector.',
        'required':   []
        },
    6: {'name':             'parallel_pnac',
        'description':      'Adjust kinetic energy only with the component of the velocity vector along the projected non-adiabatic coupling vector.',
        'description_refl': 'Reflect only the component of the velocity vector along the projected non-adiabatic coupling vector.',
        'required':   ['nacdr']
        },
    7: {'name':             'parallel_enac',
        'description':      'Adjust kinetic energy only with the component of the velocity vector along the effective non-adiabatic coupling vector.',
        'description_refl': 'Reflect only the component of the velocity vector along the effective non-adiabatic coupling vector.',
        'required':   []
        },
    8: {'name':             'parallel_penac',
        'description':      'Adjust kinetic energy only with the component of the velocity vector along the projected effective non-adiabatic coupling vector.',
        'description_refl': 'Reflect only the component of the velocity vector along the projected effective non-adiabatic coupling vector.',
        'required':   []
        }
}

# Decoherence schemes for TSH
DecoherencesTSH = {
    1: {'name': 'none',
        'description': 'No decoherence correction.',
        'required': [],
        'params': ''
        },
    2: {'name': 'edc',
        'description': 'Energy-based decoherence scheme (Granucci, Persico, Zoccante).',
        'required': [],
        'params': '0.1'
        },
    3: {'name': 'afssh',
        'description': 'Augmented fewest-switching surface hopping (Jain, Alguire, Subotnik).',
        'required': [],
        'params': ''
        }
}

# Decoherence schemes for SCP
DecoherencesSCP={
  1: {'name':             'none',
      'description':      'No decoherence correction.',
      'required':   [],
      'params':     ''
     },
  2: {'name':             'dom',
      'description':      'Decay of Mixing (Zhu, Nangia, Jasper, Truhlar).',
      'required':   [],
      'params':     ''
     }
}

# Decoherence time formulas for SCP
DecotimeSCP={
    1: {'name':             'csdm',
        'description':      'Original CSDM method (Zhu, Nangia, Jasper, Truhlar)'
        },
    2: {'name':             'scdm',
        'description':      'SCDM method (Zhu, Jasper, Truhlar)'
        },
    3: {'name':             'edc',
        'description':      'energy based decoherence (Granucci, Persico, Zoccante)'
        },
    4: {'name':             'sd',
        'description':      'stochastic decoherence time (Jasper, Truhlar)'
        },
    5: {'name':             'fp1',
        'description':      'force momentum method 1 (Shu, Zhang, Truhlar, underdevelopment)'
        },
    6: {'name':             'fp2',
        'description':      'force momentum method 2 (Shu, Zhang, Truhlar, underdevelopment)'
        }
}

# Surface hopping schemes for TSH
HoppingSchemes = {
    1: {"name": "off", "description": "Surface hops off."},
    2: {"name": "sharc", "description": "Standard SHARC surface hopping probabilities (Mai, Marquetand, Gonzalez)."},
    3: {"name": "gfsh", "description": "Global flux surface hopping probabilities (Wang, Trivedi, Prezhdo)."},
}

# Pointer state switching schemes for SCP
SwitchingSchemes={
    1: {'name':             'off',
        'description':      'Surface switchings off.'
        },
    2: {'name':             'CSDM',
        'description':      'Coherent switching with decay of mixing (Shu, Zhang, Mai, Sun, Truhlar, Gonzalez).'
        }
}


# ======================================================================================================================
# ======================================================================================================================
# ======================================================================================================================


def try_read(a, index, typefunc, default):
    try:
        if typefunc == bool:
            return "True" == a[index]
        else:
            return typefunc(a[index])
    except IndexError:
        return typefunc(default)
    except ValueError:
        log.info("Could not initialize object!")
        quit(1)


# ======================================================================================================================


class ATOM:
    def __init__(self, symb="??", num=0.0, coord=[0.0, 0.0, 0.0], m=0.0, veloc=[0.0, 0.0, 0.0]):
        self.symb = symb
        self.num = num
        self.coord = coord
        self.mass = m
        self.veloc = veloc
        self.Ekin = 0.5 * self.mass * sum([self.veloc[i] ** 2 for i in range(3)])

    def init_from_str(self, initstring=""):
        f = initstring.split()
        self.symb = try_read(f, 0, str, "??")
        self.num = try_read(f, 1, float, 0.0)
        self.coord = [try_read(f, i, float, 0.0) for i in range(2, 5)]
        self.mass = try_read(f, 5, float, 0.0) * U_TO_AMU
        self.veloc = [try_read(f, i, float, 0.0) for i in range(6, 9)]
        self.Ekin = 0.5 * self.mass * sum([self.veloc[i] ** 2 for i in range(3)])

    def __str__(self):
        s = "%2s % 5.1f " % (self.symb, self.num)
        s += "% 12.8f % 12.8f % 12.8f " % tuple(self.coord)
        s += "% 12.8f " % (self.mass / U_TO_AMU)
        s += "% 12.8f % 12.8f % 12.8f" % tuple(self.veloc)
        return s

    def EKIN(self):
        self.Ekin = 0.5 * self.mass * sum([self.veloc[i] ** 2 for i in range(3)])
        return self.Ekin

    def geomstring(self):
        s = "  %2s % 5.1f % 12.8f % 12.8f % 12.8f % 12.8f" % (
            self.symb,
            self.num,
            self.coord[0],
            self.coord[1],
            self.coord[2],
            self.mass / U_TO_AMU,
        )
        return s

    def velocstring(self):
        s = " " * 11 + "% 12.8f % 12.8f % 12.8f" % tuple(self.veloc)
        return s


# ======================================================================================================================


class STATE:
    def __init__(self, i=0, e=0.0, eref=0.0, dip=[0.0, 0.0, 0.0]):
        self.i = i
        self.e = e.real
        self.eref = eref.real
        self.dip = dip
        self.Excited = False
        self.Eexc = self.e - self.eref
        self.Fosc = (2.0 / 3.0 * self.Eexc * sum([i * i.conjugate() for i in self.dip])).real
        if self.Eexc == 0.0:
            self.Prob = 0.0
        else:
            self.Prob = self.Fosc / self.Eexc**2

    def init_from_str(self, initstring):
        f = initstring.split()
        self.i = try_read(f, 0, int, 0)
        self.e = try_read(f, 1, float, 0.0)
        self.eref = try_read(f, 2, float, 0.0)
        self.dip = [complex(try_read(f, i, float, 0.0), try_read(f, i + 1, float, 0.0)) for i in [3, 5, 7]]
        self.Excited = try_read(f, 11, bool, False)
        self.Eexc = self.e - self.eref
        self.Fosc = (2.0 / 3.0 * self.Eexc * sum([i * i.conjugate() for i in self.dip])).real
        if self.Eexc == 0.0:
            self.Prob = 0.0
        else:
            self.Prob = self.Fosc / self.Eexc**2

    def __str__(self):
        s = "%03i % 18.10f % 18.10f " % (self.i, self.e, self.eref)
        for i in range(3):
            s += "% 12.8f % 12.8f " % (self.dip[i].real, self.dip[i].imag)
        s += "% 12.8f % 12.8f %s" % (self.Eexc * HARTREE_TO_EV, self.Fosc, self.Excited)
        return s

    # def Excite(self, max_Prob, erange):
    #     try:
    #         Prob = self.Prob / max_Prob
    #     except ZeroDivisionError:
    #         Prob = -1.0
    #     if not (erange[0] <= self.Eexc <= erange[1]):
    #         Prob = -1.0
    #     self.Excited = random.random() < Prob


# ======================================================================================================================


class INITCOND:
    def __init__(self, atomlist=[], eref=0.0, epot_harm=0.0):
        self.atomlist = atomlist
        self.eref = eref
        self.Epot_harm = epot_harm
        self.natom = len(atomlist)
        self.Ekin = sum([atom.Ekin for atom in self.atomlist])
        self.statelist = []
        self.nstate = 0
        self.Epot = epot_harm

    def addstates(self, statelist):
        self.statelist = statelist
        self.nstate = len(statelist)
        self.Epot = self.statelist[0].e - self.eref

    def init_from_file(self, f, eref, index):
        while True:
            line = f.readline()
            # if 'Index     %i' % (index) in line:
            if re.search(r"Index\s+%i" % (index), line):
                break
            if line == "\n":
                continue
            if line == "":
                log.info("Initial condition %i not found in file %s" % (index, f.name))
                quit(1)
        f.readline()  # skip one line, where "Atoms" stands
        atomlist = []
        self.Ekin = 0.
        while True:
            line = f.readline()
            if "States" in line:
                break
            m, vx, vy, vz = line.split()[-4:]
            self.Ekin += 0.5 * float(m) * U_TO_AMU * (float(vx) ** 2 + float(vy) ** 2 + float(vz) ** 2)
            atomlist.append(line)
        statelist = []
        while True:
            line = f.readline()
            if "Ekin" in line:
                break
            state = STATE()
            state.init_from_str(line)
            statelist.append(state)
        epot_harm = 0.0
        while not line == "\n" and not line == "":
            line = f.readline()
            if "epot_harm" in line.lower():
                epot_harm = float(line.split()[1])
                break
        self.atomlist = atomlist
        self.eref = eref
        self.Epot_harm = epot_harm
        self.natom = len(atomlist)
        # self.Ekin = sum([atom.Ekin for atom in self.atomlist])
        self.statelist = statelist
        self.nstate = len(statelist)
        if self.nstate > 0:
            self.Epot = self.statelist[0].e - self.eref
        else:
            self.Epot = epot_harm

    def __str__(self):
        s = "Atoms\n"
        for atom in self.atomlist:
            s += str(atom) + "\n"
        s += "States\n"
        for state in self.statelist:
            s += str(state) + "\n"

        s += "Ekin      % 16.12f a.u.\n" % (self.Ekin)
        s += "Epot_harm % 16.12f a.u.\n" % (self.Epot_harm)
        s += "Epot      % 16.12f a.u.\n" % (self.Epot)
        s += "Etot_harm % 16.12f a.u.\n" % (self.Epot_harm + self.Ekin)
        s += "Etot      % 16.12f a.u.\n" % (self.Epot + self.Ekin)
        s += "\n\n"
        return s


# ======================================================================================================================
# ======================================================================================================================
# ======================================================================================================================


def check_initcond_version(string, must_be_excited=False):
    if "sharc initial conditions file" not in string.lower():
        return False
    f = string.split()
    for i, field in enumerate(f):
        if "version" in field.lower():
            try:
                v = float(f[i + 1])
                if v not in versionneeded:
                    return False
            except IndexError:
                return False
    if must_be_excited:
        if "excited" not in string.lower():
            return False
    return True


# ======================================================================================================================


def displaywelcome():
    log.info("Script for setup of SHARC trajectories started...\n")
    string = "\n"
    string += "  " + "=" * 80 + "\n"
    input = [
        " ",
        "Setup trajectories for SHARC dynamics",
        " ",
        "Authors: Sebastian Mai, Philipp Marquetand, Severin Polonius",
        " ",
        "Version: %s" % (version),
        "Date: %s" % (versiondate.strftime("%d.%m.%y")),
        " ",
    ]
    for inp in input:
        string += "||{:^80}||\n".format(inp)
    string += "  " + "=" * 80 + "\n\n"
    string += """
This script automatizes the setup of the input files for SHARC dynamics.
  """
    log.info(string)


# ======================================================================================================================
# ======================================================================================================================
# ======================================================================================================================


def open_keystrokes():
    global KEYSTROKES
    KEYSTROKES = open("KEYSTROKES.tmp", "w")


def close_keystrokes():
    KEYSTROKES.close()
    shutil.move("KEYSTROKES.tmp", "KEYSTROKES.setup_laser_excitation")


# ===================================


class init_string:
    def __init__(self):
        self.strings = []
        self.nst = 0
        self.width = 100
        self.group = 10
        self.groups = (self.width - 1) // self.group + 1
        self.nrow = 1
        self.lastrow = 0

    def add(self, s):
        self.strings.append(s)
        self.nst += 1
        self.nrow = (self.nst - 1) // self.width + 1
        self.lastrow = self.nst % self.width
        if self.lastrow == 0:
            self.lastrow = self.width

    def reset(self):
        self.strings = []
        self.nst = 0
        self.nrow = 1
        self.lastrow = 0

    def __str__(self):
        nw = int(math.log(self.nst) // math.log(10) + 1.1)
        s = " " * (nw + 2)
        fs = "%%%ii" % (nw)
        for i in range(self.groups):
            s += " " * (self.group - nw + 1) + fs % ((i + 1) * self.group)
        s += "\n"
        s += " " * (nw + 2)
        for i in range(self.groups):
            s += " "
            for j in range(self.group - 1):
                s += " "
            s += "|"
        s += "\n"
        index = 0
        for i in range(self.nrow):
            s += fs % (i * self.width) + " | "
            for j in range(self.width):
                try:
                    s += self.strings[index]
                except IndexError:
                    return s
                index += 1
                if (j + 1) % self.group == 0:
                    s += " "
            s += "\n"
        s += "\n"
        return s


# ======================================================================================================================


# def analyze_initconds(initlist, INFOS):
#     if INFOS["show_content"]:
#         log.info("Contents of the initconds file:")
#         log.info(
#             """\nLegend:
# ?       Geometry and Velocity
# .       not selected
# #       selected
# """
#         )
#     n_hasexc = []
#     n_issel = []
#     display = init_string()
#     for state in range(INFOS["nstates"]):
#         if INFOS["show_content"]:
#             log.info("State %i:" % (state + 1))
#         display.reset()
#         n_hasexc.append(0)
#         n_issel.append(0)
#         for i in initlist:
#             if len(i.statelist) < state + 1:
#                 display.add("?")
#             else:
#                 n_hasexc[-1] += 1
#                 if i.statelist[state].Excited:
#                     display.add("#")
#                     n_issel[-1] += 1
#                 else:
#                     display.add(".")
#         if INFOS["show_content"]:
#             log.info(display)
#     log.info("Number of excited states and selections:")
#     log.info("State    #InitCalc       #Selected")
#     for i in range(len(n_hasexc)):
#         s = "% 5i        % 5i           % 5i" % (i + 1, n_hasexc[i], n_issel[i])
#         if not INFOS["isactive"][i]:
#             s += "  inactive"
#         log.info(s)
#     return n_issel


# ======================================================================================================================


def get_initconds(INFOS):
    """"""

    INFOS["initf"].seek(0)  # rewind the initf file
    initlist = []
    log.info("Reading initconds file")
    width_bar = 80
    for icond in range(1, INFOS["ninit"] + 1):
        # done = width_bar * (icond) // INFOS["ninit"]
        # sys.stdout.write("\r  Progress: [" + "=" * done + " " * (width_bar - done) + "] %3i%%" % (done * 100 // width_bar))
        initcond = INITCOND()
        initcond.init_from_file(INFOS["initf"], INFOS["eref"], icond)
        initlist.append(initcond)
    log.info("\nNumber of initial conditions in file:       %5i" % (INFOS["ninit"]))

    INFOS["initlist"] = initlist
    # INFOS["n_issel"] = [True]+[False]*(INFOS["nstates"]-1)  # analyze_initconds(initlist, INFOS)
    return INFOS

def get_laser(INFOS):
    laser = np.genfromtxt(INFOS["laserfile"])
    if laser.shape[1] != 8:
        print("Laser file does not match specifications!")
        raise IOError
    else:
        laser_tsteps, laser_freqs = laser[:, 0], laser[:, -1]
        Er, Ei = laser[:, 1:6:2], laser[:, 2:7:2]
    return laser_tsteps, laser_freqs, Er, Ei 

def random_seed():
    print("{:-^60}".format("Random number seed") + "\n")
    print('Please enter a random number generator seed (type "!" to initialize the RNG from the system time).')
    while True:
        line = question("RNG Seed: ", str, "!", False)
        if line == "!":
            random.seed()
            break
        try:
            rngseed = int(line)
            random.seed(rngseed)
        except ValueError:
            print('Please enter an integer or "!".')
            continue
        break
    print("")
    return rngseed

@njit
def transform_fields(Rmat, Er=None, Ei=None, Br=None, Bi=None, Egradr=None, Egradi=None):
    tsteps = Er.shape[0]
    if Er is not None:
        Er_rot = np.empty_like(Er)
        Ei_rot = np.empty_like(Ei)
    # if B != None:
    #     B_rot = np.empty_like(B)
    # if Egrad != None:
    #     Egrad_rot = np.empty_like(Egrad)

    for t in range(tsteps):
        Er_rot[t] = np.ascontiguousarray(Er[t]) @ np.ascontiguousarray(Rmat)
        Ei_rot[t] = np.ascontiguousarray(Ei[t]) @ np.ascontiguousarray(Rmat)
        # B_rot[t] = B[t] @ Rmat
        # Egrad_rot[t] = Rmat @ Egrad_rot[t] @ Rmat.T
    # return E_rot, B_rot, Egrad_rot
    return Er_rot, Ei_rot


# def custom_formatter(val: float):                                           
#     """                                                                     
#     Formats the laser fields files' values in defined scientific notation   
#     Args:                                                                   
#        x (int):                                                            
#     Returns:                                                                
#       Formatted laser fields files' values                                 
#     """                                                                     
#     # assert isinstance(val, float), "val must be a float!"                 
#     if val!=0.0:                                                            
#         if np.abs(val)<1E-99:                                               
#             val=0.0                                                         
#     val_form = '{:.8e}'.format(val)  # Format with 3 digits for the exponent
#     mantissa, exponent = val_form.split('e')                                
#     sign = '  ' if float(mantissa) >= 0 else ' '  # Check if positive       
#     return f'{sign}{mantissa}E{exponent[0]}{exponent[1:].zfill(2)}'         


# def fast_formatter(arr):
#     arr = arr.copy()
#     arr[np.abs(arr) < 1e-99] = 0.0
# 
#     base_fmt = np.char.mod('%.8e', arr.ravel())  # flatten for formatting
# 
#     parts = np.char.split(base_fmt, 'e')
#     mantissa = np.array([p[0] for p in parts])
#     exponent = np.array([p[1] for p in parts])
# 
#     exponent = np.char.zfill(exponent, 3)
#     exponent = np.char.upper(exponent)
# 
#     signs = np.where(mantissa.astype(float) >= 0, '  ', ' ')
#     formatted = np.char.add(signs, mantissa)
#     formatted = np.char.add(formatted, 'E')
#     formatted = np.char.add(formatted, exponent)
# 
#     return formatted.reshape(arr.shape)
# 
# 
def write_fields(output_name, laser_tsteps, laser_freqs, E=None):
    rot_laser_fields = np.empty((E[0].shape[0], 8))
    rot_laser_fields[:, 1:6:2], rot_laser_fields[:, 2:7:2] = E
    rot_laser_fields[:, 0], rot_laser_fields[:, 7] = laser_tsteps, laser_freqs

    # formatted_laser_file = np.empty(
    #     (len(rot_laser_fields), rot_laser_fields.shape[1] + 1),
    #     dtype="U16"
    # )
    # formatted_laser_file[:, 0] = " "
    # formatted_laser_file[:, 1:] = fast_formatter(rot_laser_fields)

    np.savetxt(output_name, rot_laser_fields, fmt="%1.8E", delimiter=" ", comments="")
# 
# # def write_fields(output_name, laser_tsteps, laser_freqs, E=None):
# #     rot_laser_fields = np.empty((E[0].shape[0], 8))
# #     rot_laser_fields[:, 1:6:2], rot_laser_fields[:, 2:7:2] = E
# #     rot_laser_fields[:, 0], rot_laser_fields[:, 7] = laser_tsteps, laser_freqs
# #     vectorized_formatter = np.vectorize(custom_formatter)  
# #     formatted_laser_file = np.empty((len(rot_laser_fields), len(rot_laser_fields[0]) + 1), dtype="U16")                  
# #     formatted_laser_file[:, 0] = " "  # First column filled with space                                                   
# #     formatted_laser_file[:, 1:] = vectorized_formatter(rot_laser_fields)                                                 
# #     # head=''.join(head)                                                                                                   
# #     np.savetxt(output_name, formatted_laser_file, fmt="%s", delimiter="", comments='')        


# ======================================================================================================================


# def check_laserfile(filename, nsteps, dt):
#     log.info('Laser file must have %i steps and a time step of %f fs.' % (nsteps,dt))
#     try:
#         f = open(filename)
#         data = f.readlines()
#         f.close()
#     except IOError:
#         log.info("Could not open laser file %s" % (filename))
#         return False
#     n = 0
#     for line in data:
#         if len(line.split()) >= 8:
#             n += 1
#         else:
#             break
#     if n < nsteps:
#         log.info("File %s has only %i timesteps, %i steps needed!" % (filename, n, nsteps))
#         return False
#     for i in range(int(nsteps) - 1):
#         t0 = float(data[i].split()[0])
#         t1 = float(data[i + 1].split()[0])
#         if abs(abs(t1 - t0) - dt) > 1e-6:
#             log.info("Time step wrong in file %s at line %i." % (filename, i + 1))
#             return False
#     return True


def get_laser_time(filename):
    data = readfile(filename)
    print( float(data[-1].split()[0]), len(data)-1, float(data[-1].split()[0])/(len(data)-1))  # -float(data[-2].split()[0])  # tmax, nsteps, dtstep 
    return float(data[-1].split()[0]), len(data)-1, float(data[-1].split()[0])/(len(data)-1)  # -float(data[-2].split()[0])  # tmax, nsteps, dtstep 
    
# ======================================================================================================================
# ======================================================================================================================
# ======================================================================================================================


def get_general(INFOS):
    """This routine questions from the user some general information:
    - initconds file path        
    - ICONDS directory           
    - laser file path            
    - initial state              
    - representation (SHARC/MCH) 
    - run script information     
    """

    string = "\n  " + "=" * 80 + "\n"
    string += "||" + f"{'Initial conditions':^80}" + "||\n"
    string += "  " + "=" * 80 + "\n\n"
    log.info(string)
    log.info(
        """\nThis script reads the initial conditions (geometries, velocities, initial excited state)
from the initcond files as provided by wigner.py.
"""
    )

    # open the initconds file
    try:
        initfile = "initconds"
        initf = open(initfile)
        line = initf.readline()
        if check_initcond_version(line, must_be_excited=False):
            log.info('Initial conditions file "initconds" detected. Do you want to use this?')
            if not question('Use file "initconds"?', bool, True):
                initf.close()
                raise IOError
        else:
            initf.close()
            raise IOError
    except IOError:
        log.info("Please enter the filename of the initial conditions file.")
        while True:
            initfile = question("Initial conditions filename:", str, "initconds")
            initfile = os.path.expanduser(os.path.expandvars(initfile))
            if os.path.isdir(initfile):
                log.info("Is a directory: %s" % (initfile))
                continue
            if not os.path.isfile(initfile):
                log.info("File does not exist: %s" % (initfile))
                continue
            try:
                initf = open(initfile, "r")
            except IOError:
                log.info("Could not open: %s" % (initfile))
                continue
            line = initf.readline()
            if check_initcond_version(line, must_be_excited=False):
                break
            else:
                log.info("File does not contain initial conditions!")
                continue
    # read the header
    INFOS["ninit"] = int(initf.readline().split()[1])
    log.info("\nFile %s contains %i initial conditions." % (initfile, INFOS["ninit"]))
    while True:
        INFOS["icond_sel"] = question("Which initial conditions do you want to take? ", int, ranges=True)
        INFOS["icond_sel"] = list(set(INFOS["icond_sel"]))  # remove duplicates
        if not all(1 <= num <= INFOS["ninit"] for num in INFOS["icond_sel"]):
            log.info(all(1 <= num <= INFOS["ninit"] for num in INFOS["icond_sel"]))
            continue
        else:
            log.info("Chose %s initconds" % INFOS["icond_sel"])
            break
    INFOS["natom"] = int(initf.readline().split()[1])
    log.info("Number of atoms is %i" % (INFOS["natom"]))
    INFOS["repr"] = initf.readline().split()[1]
    if INFOS["repr"].lower() == "mch":
        INFOS["diag"] = False
        INFOS["repr"] = "MCH"
    else:
        INFOS["diag"] = True
        INFOS["repr"] = "diag"

    INFOS["eref"] = float(initf.readline().split()[1])
    INFOS["eharm"] = float(initf.readline().split()[1])

    # # get guess for number of states
    # line = initf.readline()
    # if "states" in line.lower():
    #     states = []
    #     li = line.split()
    #     for i in range(1, len(li)):
    #         states.append(int(li[i]))
    #     guessstates = states
    # else:
    #     guessstates = None
    
    # Equi-block from excite.py
    while True:
        line = initf.readline()
        if "Equilibrium" in line:
            break
        if line == "":
            print("File malformatted! No equilibrium geometry!")
            quit(1)
    equi = []
    for i in range(INFOS["natom"]):
        line = initf.readline()
        # atom = ATOM()
        # atom.init_from_str(line)
        equi.append(line)
    INFOS["equi"] = equi

    log.info("Reference energy %16.12f a.u." % (INFOS["eref"]))
    log.info("Excited states are in %s representation.\n" % (["MCH", "diagonal"][INFOS["diag"]]))
    initf.seek(0)  # rewind the initf file
    INFOS["initf"] = initf

    # Number of states
    # log.info(
    #     "\nPlease enter the number of states as a list of integers\ne.g. 3 0 3 for three singlets, zero doublets and three triplets."
    # )
    # while True:
    #     states = question("Number of states:", int, guessstates)
    #     if len(states) == 0:
    #         continue
    #     if any(i < 0 for i in states):
    #         log.info("Number of states must be positive!")
    #         continue
    #     break
    # log.info("")
    # nstates = 0
    # for mult, i in enumerate(states):
    #     nstates += (mult + 1) * i

    # print("\nPlease enter the molecular charge for each chosen multiplicity\ne.g. 0 +1 0 for neutral singlets and triplets and cationic doublets.")
    # default = [i % 2 for i in range(len(states))]
    # while True:
    #     charges = question("Molecular charges per multiplicity:", int, default)
    #     if not states:
    #         continue
    #     if len(charges) != len(states):
    #         print("Charges array must have same length as states array")
    #         continue
    #     break

    # log.info("Number of states: " + str(states))
    # log.info("Total number of states: %i\n" % (nstates))
    if os.path.isdir(INFOS["path"]):
        num = int(0)  # Should open the QM.out file from ICOND00000 as standard
        qmoutfile = os.path.join(INFOS["path"], 'ICOND_%05i' % num, 'QM.out')
        if not os.path.isfile(qmoutfile):
            log.error('Can not find QM.out file for %s' % INFOS["path"])
    else:
        qmoutfile = INFOS["path"]
        if not os.path.isfile(qmoutfile):
            log.error('Can not find QM.out file for %s' % INFOS["path"])
    QMoutfile = QMout(qmoutfile)
    INFOS["charge"] = QMoutfile.charges 
    INFOS["states"] = QMoutfile.states
    INFOS["nstates"] = QMoutfile.nstates
    log.info(
        "Number of states and molecular charge for each chosen multiplicity: %s %s" % (INFOS["charge"], INFOS["states"])
    )
    # obtain the statemap
    statemap = {}
    i = 1
    for imult, istate, ims in itnmstates(INFOS["states"]):
        statemap[i] = [imult, istate, ims]
        i += 1
    INFOS["statemap"] = statemap

    # get active states
    INFOS["actstates"] = INFOS["states"]
    isactive = []
    for imult in range(len(INFOS["states"])):
        for ims in range(imult + 1):
            for istate in range(INFOS["states"][imult]):
                isactive.append((istate + 1 <= INFOS["actstates"][imult]))
    INFOS["isactive"] = isactive
    log.info("")

    # ask whether initfile content is shown
    # INFOS["show_content"] = question("Do you want to see the content of the initconds file?", bool, False)

    # read initlist, analyze it and log.info(content (all in get_initconds))
    INFOS["initf"] = initf
    INFOS = get_initconds(INFOS)
    initf.close()

    # Generate random example for setup-states, according to Leti's wishes
    exampleset = set()
    nactive = sum(INFOS["isactive"])
    while len(exampleset) < min(3, nactive):
        i = random.randint(1, INFOS["nstates"])
        if INFOS["isactive"][i - 1]:
            exampleset.add(i)
    exampleset = list(exampleset)
    exampleset.sort()
    string1 = ""
    string2 = ""
    j = 0
    for i in exampleset:
        j += 1
        if j == len(exampleset) and len(exampleset) > 1:
            string1 += str(i)
            string2 += "and " + str(i)
        else:
            string1 += str(i) + " "
            string2 += str(i) + ", "

    # ask for states to setup
    log.info(
        "\nPlease enter a list specifying for which excited states trajectories should be set-up\ne.g. %s to select states %s."
        % (string1, string2)
    )
    defsetupstates = []
    nmax = 0
    for i, active in enumerate(INFOS["isactive"]):
        #if active and INFOS["n_issel"][i] > 0:
        defsetupstates.append(i + 1)
        nmax += 1  # INFOS["n_issel"][i]
    if nmax <= 0:
        log.info("\nZero trajectories can be set up!")
        sys.exit(1)
    while True:
        setupstates = question("States to setup the dynamics:", int, defsetupstates, ranges=True)
        valid = True
        for i in setupstates:
            if i > INFOS["nstates"]:
                log.info("There are only %i states!" % (INFOS["nstates"]))
                valid = False
                continue
            if i < 0:
                valid = False
                continue
            if not INFOS["isactive"][i - 1]:
                log.info("State %i is inactive!" % (i))
                valid = False
        if not valid:
            continue
        INFOS["setupstates"] = set(setupstates)
        # log.info(INFOS["n_issel"])
        log.info(INFOS["setupstates"])
        log.info(INFOS["isactive"])
        # nsetupable = sum([INFOS["n_issel"][i - 1] for i in INFOS["setupstates"] if INFOS["isactive"][i - 1]])
        nsetupable = sum([INFOS["isactive"][i - 1] for i in INFOS["setupstates"]])
        log.info("\nThere can be %i trajector%s set up.\n" % (nsetupable, ["y", "ies"][nsetupable != 1]))
        if nsetupable == 0:
            continue
        break


    return INFOS




def get_requests(INFOS, interface: SHARC_INTERFACE) -> list[str]:
    """get requests for every single point"""
    interface.QMin.molecule['states'] = INFOS['states']
    int_features = interface.get_features(KEYSTROKES=KEYSTROKES)
    log.info("\nThe following features are available from this interface:")
    log.info(int_features)
    
    INFOS["needed_requests"] = set()

    # Dynamics options
    string = "\n  " + "=" * 80 + "\n"
    string += "||" + f"{'Surface Hopping dynamics settings':^80}" + "||\n"
    string += "  " + "=" * 80 + "\n\n"
    log.info(string)


    # Method
    # log.info(f"{'Nonadiabatic dynamics method':-^60}" + "\n")
    # log.info('Please choose the dynamics method you want to employ.')
    # cando = list(Method)
    # for i in Method:
    #     log.info('%i\t%s' % (i, Method[i]['description']))
    # while True:
    #     dyn=question('Method:',int,[1])[0]
    #     if dyn in Method and dyn in cando:
    #         break
    #     else:
    #         log.info('Please input one of the following: %s!' % ([i for i in cando]))
    INFOS['method']='tsh'
    # TODO: is SCP requiring any features?
    INFOS["needed_requests"].add("h")
    INFOS["needed_requests"].add("grad")
    INFOS["needed_requests"].add("dm")
    log.info('Dynamics method: %s' % INFOS['method'])

    # Simulation time and timestep
    log.info(
        """Please specify the file containing the complete laser field. The timestep in the file and the length of the file must fit to the simulation time, time step and number of substeps given above.
            Laser files can be created using $SHARC/laser.x
        """
        )
    while True:
        INFOS["laserfile"] = os.path.abspath(question("Laser filename:", str))
        log.info(INFOS["laserfile"])
        if not os.path.isfile(INFOS["laserfile"]):
            log.info("File %s does not exist!" % (INFOS["laserfile"]))
            continue
        else:
            break
    INFOS["rand_laser_pol"] = question("Do you want to have an isotropic laser polarization distribution:", bool, True)
    INFOS["tmax"], INFOS["nsteps"], INFOS["dtstep"] = get_laser_time(INFOS["laserfile"])
    log.info("Total simulation time: %f"  % INFOS["tmax"])
    log.info("\nSimulation will have %i timesteps." % (INFOS["dtstep"]))


    # Integrator
    INFOS['integrator'] = int(2)    
    log.info("Integrator: %s " % Integrator[INFOS['integrator']]["name"])
    # number of substeps
    INFOS["nsubstep"] = int(1) 
    log.info("NSubstep: %s " % INFOS["nsubstep"])
    # whether to kill relaxed trajectories
    INFOS["kill"] = "False"
    log.info("")


    log.info("\n" + f"{'Dynamics settings':-^60}")


    # SHARC or MCH
    log.info(
        "\nDo you want to perform the dynamics in the diagonal representation (SHARC dynamics) or in the MCH representation (regular TSH/SCP)?"
    )
    surf = question("SHARC dynamics?", bool, True)
    if INFOS['method']=='tsh':
        INFOS['surf'] = ['mch', 'diagonal'][surf]

    elif INFOS['method']=='scp':
        INFOS['surf'] = 'diagonal'
        if surf==True:
            INFOS['pointer_basis'] = 'diag'
            INFOS['neom_rep'] = 'diag' 
        else:
            INFOS['pointer_basis'] = 'diag'
            INFOS['neom_rep'] = 'MCH'



    states = INFOS["states"]
    log.info("states found in get_requests")
    # Setup SOCs
    if len(states) > 1:
        if "soc" in int_features:
            log.info("Do you want to include spin-orbit couplings in the dynamics?\n")
            soc = question("Spin-Orbit calculation?", bool, True)
            if soc:
                log.info("Will calculate spin-orbit matrix.")
        else:
            log.info("Interface cannot provide SOCs: not calculating spin-orbit matrix.")
            soc = False
    else:
        log.info("Only singlets specified: not calculating spin-orbit matrix.")
        soc = False
    log.info("")
    INFOS["soc"] = soc
    if INFOS["soc"]:
        INFOS["needed_requests"].add("soc")



    # Coupling
    log.info("\nPlease choose the quantities to describe non-adiabatic effects between the states:")
    INFOS["coupling"] = 4
    INFOS["needed_requests"].update(Couplings[4]["required"])

    # Phase tracking
    INFOS["phases_from_interface"] = False

    # Gradient correction (only for diagonal PESs)
    if INFOS["surf"] == "diagonal":
        INFOS["gradcorrect"] = 1
        INFOS["needed_requests"].update(GradCorrect[1]["required"])
    else:
        num = next((k for k, v in GradCorrect.items() if v["name"] == "none"), None)
        INFOS["gradcorrect"] = 1
        INFOS["needed_requests"].update(GradCorrect[1]["required"])
    INFOS["needed_requests"].update(GradCorrect[INFOS["gradcorrect"]]["required"])




    #===============================
    # Begin Surface hopping details
    #=============================== 
    if INFOS['method']=='tsh':
        # Kinetic energy modification
        INFOS["ekincorrect"] = 1
        if INFOS["ekincorrect"]:
            for i in EkinCorrect[INFOS["ekincorrect"]]["required"]:
                INFOS["needed_requests"].add(i)
        # frustrated reflection
        INFOS["reflect"] = 1
        if INFOS["reflect"]:
            for i in EkinCorrect[INFOS["ekincorrect"]]["required"]:
                INFOS["needed_requests"].add(i)
        # decoherence
        INFOS["decoherence"] = [DecoherencesTSH[1]["name"], DecoherencesTSH[1]["params"]]
        for i in DecoherencesTSH[1]["required"]:
            INFOS["needed_requests"].add(i)

        # surface hopping scheme
        INFOS["hopping"] = HoppingSchemes[1]["name"]

        # Forced hops to lowest state
        INFOS["force_hops"] = False 
        INFOS["force_hops_dE"] = 9999.0

        # TODO: move out of the TSH/SCP if's
        # Scaling
        INFOS["scaling_for_sharc"] = False

        # TODO: move out of the TSH/SCP if's
        # Damping
        INFOS["damping"] = False

        # TODO: move out of the TSH/SCP if's?
        # atommask
        INFOS["atommaskarray"] = None

    #===============================
    # End Surface hopping details
    #===============================


    log.info(f"\n\n{'Settings for large systems':-^60}\n")


    INFOS["dipolegrad"] = False

    # Setup Dyson computation
    INFOS["ion"] = False

    string = "\n  " + "=" * 80 + "\n"
    string += "||" + f"{'Interface setup':^80}" + "||\n"
    string += "  " + "=" * 80 + "\n\n"
    log.info(string)

    return INFOS


def get_trajectory_info(INFOS) -> dict:
    INFOS["pysharc"] = True 

    # NetCDF
    INFOS["netcdf"] = True
    INFOS["netcdf_separate"] = True

    # strides
    log.info("\nDo you want to modify the output.dat writing stride?")
    if INFOS["netcdf_separate"]:
        log.info("\nNOTE: This stride will only affect electronic data in output.dat.nc")
    INFOS["stride"] = [1]

    # separate nuclear stride
    INFOS["stride_nuclear"] = [INFOS["nsteps"]]

    # Add some simple keys
    INFOS["log.infolevel"] = 2
    INFOS["cwd"] = os.getcwd()
    log.info("")

    return INFOS


# ======================================================================================================================
# ======================================================================================================================
# ======================================================================================================================
# ======================================================================================================================
# ======================================================================================================================
# ======================================================================================================================
# ======================================================================================================================
# ======================================================================================================================
# ======================================================================================================================
# ======================================================================================================================
# ======================================================================================================================


def get_runscript_info(INFOS):
    """"""

    string = "\n  " + "=" * 80 + "\n"
    string += "||" + f"{'Run mode setup':^80}" + "||\n"
    string += "  " + "=" * 80 + "\n\n"
    log.info(string)

    # run script
    log.info(f"{'Run script':-^60}" + "\n")
    log.info(
        """This script can generate the run scripts for each initial condition in two modes:

  - In mode 1, the calculation is run in subdirectories of the current directory.

  - In mode 2, the input files are transferred to another directory (e.g. a local scratch directory), the calculation is run there, results are copied back and the temporary directory is deleted. Note that this temporary directory is not the same as the "scratchdir" employed by the interfaces.

Note that in any case this script will create the input subdirectories in the current working directory.
"""
    )
    log.info("In case of mode 1, the calculations will be run in:\n%s\n" % (INFOS["cwd"]))
    here = question("Use mode 1 (i.e., calculate here)?", bool, True)
    if here:
        INFOS["here"] = True
        INFOS["copydir"] = INFOS["cwd"]
    else:
        INFOS["here"] = False
        log.info("\nWhere do you want to perform the calculations? Note that this script cannot check whether the path is valid.")
        INFOS["copydir"] = question("Run directory?", str)
    log.info("")

    # submission script
    log.info(f"{'Submission script':-^60}" + "\n")
    log.info(
        """During the setup, a script for running all initial conditions sequentially in batch mode is generated. Additionally, a queue submission script can be generated for all initial conditions.
"""
    )
    qsub = question("Generate submission script?", bool, False)
    if not qsub:
        INFOS["qsub"] = False
    else:
        INFOS["qsub"] = True
        log.info(
            '\nPlease enter a queue submission command, including possibly options to the queueing system,\ne.g. for SGE: "qsub -q queue.q -S /bin/bash -cwd" (Do not type quotes!).'
        )
        INFOS["qsubcommand"] = question("Submission command?", str, None, False)
        INFOS["proj"] = question("Project Name:", str, None, False)

    log.info("")
    return INFOS


# ======================================================================================================================
# ======================================================================================================================
# ======================================================================================================================


def make_directory(iconddir):
    """Creates a directory"""

    if os.path.isfile(iconddir):
        log.info("\nWARNING: %s is a file!" % (iconddir))
        return -1
    if os.path.isdir(iconddir):
        if len(os.listdir(iconddir)) == 0:
            return 0
        else:
            log.info("\nWARNING: %s/ is not empty!" % (iconddir))
            if "overwrite" not in globals():
                global overwrite
                overwrite = question("Do you want to overwrite files in this and all following directories? ", bool, False)
            if overwrite:
                return 0
            else:
                return -1
    else:
        try:
            os.mkdir(iconddir)
        except OSError:
            log.info("\nWARNING: %s cannot be created!" % (iconddir))
            return -1
        return 0


# ======================================================================================================================
def json_info(INFOS):
    setup_laser_excitation_info_filename= "%s/setup_laser_excitation.json" % (os.getcwd())

    # open writable json file
    try:
        setup_laser_excitation_info = open(setup_laser_excitation_info_filename, "w")
    except IOError:
        log.info(
            "IOError during opening writeable %s. Quitting."
            % (setup_laser_excitation_info_filename)
        )
        quit(1)

    INFOS["initf"] = INFOS["initf"].name  # write INFOS to info file
    INFOS.pop("initlist")
    INFOS["needed_requests"] = list(INFOS["needed_requests"])
    INFOS["statemap"] = list(INFOS["statemap"])
    INFOS["setupstates"] = list(INFOS["setupstates"])
    json.dump(INFOS, setup_laser_excitation_info, sort_keys=True, indent=4)
    setup_laser_excitation_info.close()

    return 


def writeSHARCinput(INFOS, initobject, iconddir, istate, laser_tsteps, laser_freqs, Er, Ei, rng_gen, rot_vec_arr, ask=False):
    inputfname = iconddir + "/input"
    try:
        inputf = open(inputfname, "w")
    except IOError:
        log.info("IOError during writeSHARCinput, iconddir=%s\n%s" % (iconddir, inputfname))
        quit(1)

    s = 'printlevel 2\n\ngeomfile "geom"\nveloc external\nvelocfile "veloc"\n\n'   
    s += "nstates "
    for nst in INFOS["states"]:
        s += "%i " % nst
    s += "\nactstates "
    for nst in INFOS["actstates"]:
        s += "%i " % nst
    s += "\ncharge "
    for nst in INFOS["charge"]:
        s += "%i " % nst
    s += "\nstate %i %s\n" % (istate, ["mch", "diag"][INFOS["diag"]])
    s += "coeff auto\n"
    s += "rngseed %i\n\n" % (0)
    s += "ezero %18.10f\n" % (INFOS["eref"])

    s += "tmax %f\nstepsize %f\nnsubsteps %i\n" % (INFOS["tmax"], INFOS["dtstep"], INFOS["nsubstep"])
    s += 'integrator %s\n' % (Integrator[INFOS['integrator']]["name"])
    # if Integrator[INFOS['integrator']]["name"] == 'avv':
    #     s += 'convthre %s\n' % (INFOS['convthre'])
    s += "\n"


    # general dynamics settings
    s += 'method %s\n' % (INFOS['method'])
    s += "surf %s\n" % (INFOS["surf"])
    s += "coupling %s\n" % (Couplings[INFOS["coupling"]]["name"])
    s += 'nogradcorrect\n'

    # TSH settings
    if INFOS['method'] == 'tsh':
        s += 'ekincorrect %s\n' % (EkinCorrect[INFOS['ekincorrect']]['name'])
        s += 'reflect_frustrated %s\n' % (EkinCorrect[INFOS['reflect']]['name'])
        s += 'decoherence_scheme %s\n' % (INFOS['decoherence'][0])
        if INFOS['decoherence'][1]:
            s += 'decoherence_param %s\n' % (INFOS['decoherence'][1])
        s += 'hopping_procedure %s\n' % (INFOS['hopping'])
        if INFOS['force_hops']:
            s += 'force_hop_to_gs %f\n' % (INFOS['force_hops_dE'])
        if INFOS['scaling_for_sharc']:
            s += 'scaling %f\n' % (INFOS['scaling_for_sharc'])
        if INFOS['damping'] is not False:
            s += 'dampeddyn %f\n' % (INFOS['damping'])
        if INFOS['phases_from_interface']:
            s += 'phases_from_interface\n'
        if "atommaskarray" in INFOS and INFOS["atommaskarray"] is not None:
            s += '\natommask external\natommaskfile "atommask"\n\n'

    s += "notrack_phase\n"

    if INFOS["select_directly"]:
        s += "select_directly\n"

    if not INFOS["soc"]:
        s += "nospinorbit\n"

    # NetCDF or ASCII
    out = "netcdf_separate_nuc"
    s += "output_format %s\n" % out

    # stride
    if "stride" in INFOS:
        s += "output_dat_steps"
        for i in INFOS["stride"]:
            s += " %i" % i
        s += "\n"

    # stride for separate nuclei
    if INFOS["netcdf_separate"]:
        if "stride_nuclear" in INFOS:
            s += "output_dat_steps_nuc"
            for i in INFOS["stride_nuclear"]:
                s += " %i" % i
            s += "\n"

    s += "\n"

    # laser
    s += "laser external\n"
    s += 'laserfile "laser"\n'
    s += "laserfilepath %s\n" %(INFOS["laserfile"])
    s += "\n"

    # let user look at input and add extra stuff
    if ask:
        if question("\n\nDo you want to see the input for the first trajectory?", bool, default=False):
            log.info(f"{'generated input for ' + iconddir:=^80}")
            log.info("-"*80)
            log.info(s)
            log.info("-"*80)
        if question("Do you want to add keywords to the input of all trajectories?", bool, default=False):
            INFOS["all_additions"] = []
            addition = " "
            while addition != "end":
                INFOS["all_additions"].append(addition)
                addition = question("Type the keyword and value you want to add (terminate by typing 'end')", str, default='end')

    if "all_additions" in INFOS:
        s += "\n".join(INFOS["all_additions"])

    inputf.write(s)
    inputf.close()

    # geometry file
    geomfname = iconddir + "/geom"
    geomf = open(geomfname, "w")
    for atom in initobject.atomlist:
        geomf.write(atom[:60] + "\n")
    geomf.close()

    # velocity file
    velocfname = iconddir + "/veloc"
    velocf = open(velocfname, "w")
    for atom in initobject.atomlist:
        velocf.write(atom[60:])
    velocf.close()

    # laser file
    laserfname = iconddir + "/laser"
    sharcpath = os.getenv('SHARC')   
    if sharcpath is None:                                                                
       print('Please set $SHARC to the directory containing the SHARC executables!')
       sys.exit(1)
    if INFOS["rand_laser_pol"]:
        rot = R.random(random_state=rng_gen)
        Rmat = rot.as_matrix()
        rot_vec_arr = np.vstack((rot_vec_arr, rot.as_rotvec()))
        trans_fields = transform_fields(Rmat, Er=Er, Ei=Ei, Br=None, Bi=None, Egradr=None, Egradi=None) 
        write_fields(laserfname, laser_tsteps, laser_freqs, E=trans_fields)
        # align_laser(laser_file=INFOS["laserfile"], rot_matrix=None, output_name=laserfname, random_no=random.randint(1, int(1E6)), no_print=True)
    else: 
        link(INFOS["laserfile"], laserfname)
        # shutil.copy(INFOS["laserfile"], laserfname)
   
    # atommask file
    if "atommaskarray" in INFOS and INFOS['atommaskarray'] is not None:
        atommfname = iconddir + "/atommask"
        atommf = open(atommfname, "w")
        for i, atom in enumerate(initobject.atomlist):
            if i + 1 in INFOS["atommaskarray"]:
                atommf.write("T\n")
            else:
                atommf.write("F\n")
        atommf.close()

    return rot_vec_arr


# ======================================================================================================================


def writeRunscript(INFOS, iconddir, interface):
    """writes the runscript in each subdirectory"""
    try:
        runscript = open("%s/run.sh" % (iconddir), "w")
    except IOError:
        log.info("IOError during writeRunscript, iconddir=%s" % (iconddir))
        quit(1)
    if "proj" in INFOS:
        projname = "%4s_%5s" % (INFOS["proj"][0:4], iconddir[-6:-1])
    else:
        projname = "traj_%5s" % (iconddir[-6:-1])

    # ================================
    intstring = ""
    if "amsbashrc" in INFOS:
        intstring = ". %s\nexport PYTHONPATH=$AMSHOME/scripting:$PYTHONPATH" % (INFOS["amsbashrc"])

    # ================================
    if INFOS["pysharc"]:
        driver = ("_".join(interface.__class__.__name__.split("_")[1:])).lower()
        exestring = ". $SHARC/sharcvars.sh\n$SHARC/driver.py -i %s input" % driver
    else:
        exestring = "$SHARC/sharc.x input"

    # ================================ for here mode
    if INFOS["here"]:
        string = """#!/usr/bin/env bash

echo "%s"

%s

PRIMARY_DIR=%s/%s

cd $PRIMARY_DIR

%s
""" % (
            projname,
            intstring,
            INFOS["cwd"],
            iconddir,
            exestring,
        )
    #
    # ================================ for remote mode
    else:
        string = """#!/usr/bin/env bash

# $-N %s
""" % (
            projname
        )
        if INFOS["qsub"]:
            string += "#$ -v USER_EPILOG=%s/epilog.sh" % (iconddir)

        string += """
%s

PRIMARY_DIR=%s/%s
COPY_DIR=%s/%s

mkdir -p $COPY_DIR
cp -r $PRIMARY_DIR/* $COPY_DIR
cd $COPY_DIR
echo $HOSTNAME > $PRIMARY_DIR/host_info
echo $(pwd) >> $PRIMARY_DIR/host_info
echo $(date) >> $PRIMARY_DIR/host_info

%s
err=$?

cp -r $COPY_DIR/output.* $COPY_DIR/restart.* $COPY_DIR/restart/ $PRIMARY_DIR

if [ $err == 0 ];
then
  rm -r $COPY_DIR
else
  echo "The calculation crashed at
date = $(date)
with error code $err.
Please inspect the trajectory on
host = $HOSTNAME
in
dir  = $(pwd)
" > $PRIMARY_DIR/README
fi
""" % (
            intstring,
            INFOS["cwd"],
            iconddir,
            INFOS["copydir"],
            iconddir,
            exestring,
        )

    runscript.write(string)
    runscript.close()
    filename = iconddir + "/run.sh"
    os.chmod(filename, os.stat(filename).st_mode | stat.S_IXUSR)

    # also write an epilog script
    if not INFOS["here"] and INFOS["qsub"]:
        try:
            episcript = open(iconddir + "/epilog.sh", "w")
            string = """#/bin/bash

PRIMARY_DIR=%s/%s
COPY_DIR=%s/%s

cp $COPY_DIR/output.* $COPY_DIR/restart.* $PRIMARY_DIR
rm -r $COPY_DIR
""" % (
                INFOS["cwd"],
                iconddir,
                INFOS["copydir"],
                iconddir,
            )
            episcript.write(string)
            episcript.close()
        except IOError:
            log.info("Could not write epilog script for %s." % (iconddir))
    return


# ======================================================================================================================
# ======================================================================================================================
# ======================================================================================================================


def get_iconddir(istate, INFOS):
    if INFOS["diag"]:
        dirname = "State_%i" % (istate)
    else:
        mult, state, ms = INFOS["statemap"][istate]
        dirname = IToMult[mult] + "_%i" % (state - (mult == 1 or mult == 2))
    return dirname


# ====================================


def setup_all(INFOS, interface: SHARC_INTERFACE):
    """This routine sets up the directories for the initial calculations."""

    string = "\n  " + "=" * 80 + "\n"
    string += "||" + f"{'Setting up directories...':^80}" + "||\n"
    string += "  " + "=" * 80 + "\n\n"
    log.info(string)
    INFOS["setupstates_names"] = []
    all_run = open("all_run_traj.sh", "w")
    string = "#!/bin/bash\n\nCWD=%s\n\n" % (INFOS["cwd"])
    all_run.write(string)
    if INFOS["qsub"]:
        all_qsub = open("all_qsub_traj.sh", "w")
        string = "#!/bin/bash\n\nCWD=%s\n\n" % (INFOS["cwd"])
        all_qsub.write(string)

    for istate in INFOS["setupstates"]:
        dirname = get_iconddir(istate, INFOS)
        io = make_directory(dirname)
        if io != 0:
            log.info("Could not make directory %s" % (dirname))
            quit(1)


    ask = True
    laser_tsteps, laser_freqs, Er, Ei = get_laser(INFOS)
    rng_gen = np.random.default_rng(seed=INFOS["rng_seed_laser"])
    rot_vec_arr = np.empty((0,3))
    for istate in INFOS["setupstates"]:
        width = 50
        ntraj = len(INFOS["icond_sel"])  # INFOS["ntraj"]
        idone = 0

        initlist = INFOS["initlist"]
        log.info("Trajectory setup for initial state %i" % istate)
        for ic, icond in enumerate(INFOS["icond_sel"]):
            # if len(initlist[icond - 1].statelist) < istate:
            #     continue
            # if not initlist[icond - 1].statelist[istate - 1].Excited:
            #     continue

            done = (ic+1) * width // ntraj
            sys.stdout.write("\rProgress: [" + "=" * done + " " * (width - done) + "] %3i%%" % (done * 100 // width))

            dirname = get_iconddir(istate, INFOS) + "/TRAJ_%05i" % (icond)
             
            io = make_directory(dirname)
            if io != 0:
                log.info("Skipping initial condition %i %i!" % (istate, icond))
                continue

            rot_vec_arr = writeSHARCinput(INFOS, initlist[icond - 1], dirname, istate, laser_tsteps, laser_freqs, Er, Ei, rng_gen, rot_vec_arr, ask=ask)
            ask = False
            io = make_directory(dirname + "/QM")
            io += make_directory(dirname + "/restart")
            if io != 0:
                log.info("Could not make QM or restart directory!")
                continue
            interface.prepare(INFOS, dirname + "/QM")
            qmoutfile = "ICOND_%05i/QM.out" % (icond)
            # prevent symlinks error
            if os.path.realpath(os.path.join(INFOS["path"], qmoutfile)) !=  os.path.realpath(dirname+"/QM/QMout.template"): 
                shutil.copy(os.path.join(INFOS["path"], qmoutfile), dirname+"/QM/QMout.template")

            writeRunscript(INFOS, dirname, interface)

            string = "cd $CWD/%s/\nbash run.sh\ncd $CWD\necho %s >> DONE\n" % (dirname, dirname)
            all_run.write(string)
            if INFOS["qsub"]:
                string = "cd $CWD/%s/\n%s run.sh\ncd $CWD\n" % (dirname, INFOS["qsubcommand"])
                all_qsub.write(string)

        sys.stdout.write("\n")
        INFOS["setupstates_names"].append(get_iconddir(istate, INFOS))
    log.info("\n\n%i trajectories setup, last initial condition was %i in state %i.\n" % (ntraj, icond, istate))
    setup_stat = open("setup_traj.status", "a+")
    string = """*** %s %s %s
  First index:          %i
  Last index:           %i
  Trajectories:         %i
  State of last traj.:  %i

""" % (
                datetime.datetime.now(),
                gethostname(),
                os.getcwd(),
                # TODO Delete First index / Last index - if trajs are e.g. 1 2 3 6 12 
                min(INFOS["icond_sel"]),  # INFOS["firstindex"],
                max(INFOS["icond_sel"]),
                ntraj,
                istate,
            )
    setup_stat.write(string)
    setup_stat.close()
    all_run.close()
    filename = "all_run_traj.sh"
    os.chmod(filename, os.stat(filename).st_mode | stat.S_IXUSR)
    if INFOS["qsub"]:
        all_qsub.close()
        filename = "all_qsub_traj.sh"
        os.chmod(filename, os.stat(filename).st_mode | stat.S_IXUSR)
    np.savetxt("rot_vec_arr", rot_vec_arr)
    log.info("\n")


# ======================================================================================================================
# ======================================================================================================================
# ======================================================================================================================




def main():
    """Main routine"""

    usage = """
python setup_traj.py

This interactive program prepares SHARC dynamics calculations.
"""

    description = ""
    parser = OptionParser(usage=usage, description=description)

    displaywelcome()
    open_keystrokes()
    INFOS = {"select_directly": True}  # deactivate in get_infos within interface!

    chosen_interface: SHARC_INTERFACE = SHARC_QMOUT()
    INFOS = chosen_interface.get_infos(INFOS, KEYSTROKES)  # get patho of  QM.out file or to folder containing ICOND folders
    INFOS["path"] = chosen_interface.setupINFOS["path"]
    INFOS["rng_seed_laser"] = random_seed()
    INFOS = get_general(INFOS)  
    INFOS = get_requests(INFOS, chosen_interface)
    INFOS = get_trajectory_info(INFOS)
    INFOS = get_runscript_info(INFOS)
    log.info("\n" + f"{'Full input':#^60}" + "\n")
    for item in INFOS:
        if "initlist" not in item:
            log.info(f"{item:<25} {INFOS[item]}")
    log.info("")
    setup = question("Do you want to setup the specified calculations?", bool, True)
    log.info("")    
    if setup:
        INFOS["link_files"] = False
        if question("Do you want to link the interface files?", bool, default=False, autocomplete=False):
            INFOS["link_files"] = True
        setup_all(INFOS, chosen_interface)
    json_info(INFOS) 
    close_keystrokes()


# ======================================================================================================================
if __name__ == "__main__":
    try:
            main()
    except KeyboardInterrupt:
        log.info("\nCtrl+C makes me a sad SHARC ;-(\n")
        quit(0)
