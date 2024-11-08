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
import stat
import shutil
import datetime
import random
from optparse import OptionParser
from socket import gethostname

from logger import log
import factory
from utils import question, itnmstates, expand_path
from constants import IToMult, U_TO_AMU, HARTREE_TO_EV
from SHARC_INTERFACE import SHARC_INTERFACE

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

    def Excite(self, max_Prob, erange):
        try:
            Prob = self.Prob / max_Prob
        except ZeroDivisionError:
            Prob = -1.0
        if not (erange[0] <= self.Eexc <= erange[1]):
            Prob = -1.0
        self.Excited = random.random() < Prob


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
    shutil.move("KEYSTROKES.tmp", "KEYSTROKES.setup_traj")


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


def analyze_initconds(initlist, INFOS):
    if INFOS["show_content"]:
        log.info("Contents of the initconds file:")
        log.info(
            """\nLegend:
?       Geometry and Velocity
.       not selected
#       selected
"""
        )
    n_hasexc = []
    n_issel = []
    display = init_string()
    for state in range(INFOS["nstates"]):
        if INFOS["show_content"]:
            log.info("State %i:" % (state + 1))
        display.reset()
        n_hasexc.append(0)
        n_issel.append(0)
        for i in initlist:
            if len(i.statelist) < state + 1:
                display.add("?")
            else:
                n_hasexc[-1] += 1
                if i.statelist[state].Excited:
                    display.add("#")
                    n_issel[-1] += 1
                else:
                    display.add(".")
        if INFOS["show_content"]:
            log.info(display)
    log.info("Number of excited states and selections:")
    log.info("State    #InitCalc       #Selected")
    for i in range(len(n_hasexc)):
        s = "% 5i        % 5i           % 5i" % (i + 1, n_hasexc[i], n_issel[i])
        if not INFOS["isactive"][i]:
            s += "  inactive"
        log.info(s)
    return n_issel


# ======================================================================================================================


def get_initconds(INFOS):
    """"""

    INFOS["initf"].seek(0)  # rewind the initf file
    initlist = []
    log.info("Reading initconds file")
    width_bar = 80
    for icond in range(1, INFOS["ninit"] + 1):
        done = width_bar * (icond) // INFOS["ninit"]
        sys.stdout.write("\r  Progress: [" + "=" * done + " " * (width_bar - done) + "] %3i%%" % (done * 100 // width_bar))
        initcond = INITCOND()
        initcond.init_from_file(INFOS["initf"], INFOS["eref"], icond)
        initlist.append(initcond)
    log.info("\nNumber of initial conditions in file:       %5i" % (INFOS["ninit"]))

    INFOS["initlist"] = initlist
    INFOS["n_issel"] = analyze_initconds(initlist, INFOS)
    return INFOS


# ======================================================================================================================


def check_laserfile(filename, nsteps, dt):
    log.info('Laser file must have %i steps and a time step of %f fs.' % (nsteps,dt))
    try:
        f = open(filename)
        data = f.readlines()
        f.close()
    except IOError:
        log.info("Could not open laser file %s" % (filename))
        return False
    n = 0
    for line in data:
        if len(line.split()) >= 8:
            n += 1
        else:
            break
    if n < nsteps:
        log.info("File %s has only %i timesteps, %i steps needed!" % (filename, n, nsteps))
        return False
    for i in range(int(nsteps) - 1):
        t0 = float(data[i].split()[0])
        t1 = float(data[i + 1].split()[0])
        if abs(abs(t1 - t0) - dt) > 1e-6:
            log.info("Time step wrong in file %s at line %i." % (filename, i + 1))
            return False
    return True


# ======================================================================================================================
# ======================================================================================================================
# ======================================================================================================================


def get_general(INFOS):
    """This routine questions from the user some general information:
    - initconds file
    - number of states
    - number of initial conditions
    - interface to use"""

    string = "\n  " + "=" * 80 + "\n"
    string += "||" + f"{'Initial conditions':^80}" + "||\n"
    string += "  " + "=" * 80 + "\n\n"
    log.info(string)
    log.info(
        """\nThis script reads the initial conditions (geometries, velocities, initial excited state)
from the initconds.excited files as provided by excite.py.
"""
    )

    # open the initconds file
    try:
        initfile = "initconds.excited"
        initf = open(initfile)
        line = initf.readline()
        if check_initcond_version(line, must_be_excited=True):
            log.info('Initial conditions file "initconds.excited" detected. Do you want to use this?')
            if not question('Use file "initconds.excited"?', bool, True):
                initf.close()
                raise IOError
        else:
            initf.close()
            raise IOError
    except IOError:
        log.info("Please enter the filename of the initial conditions file.")
        while True:
            initfile = question("Initial conditions filename:", str, "initconds.excited")
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
            if check_initcond_version(line, must_be_excited=True):
                break
            else:
                log.info("File does not contain initial conditions!")
                continue
    # read the header
    INFOS["ninit"] = int(initf.readline().split()[1])
    INFOS["natom"] = int(initf.readline().split()[1])
    log.info("\nFile %s contains %i initial conditions." % (initfile, INFOS["ninit"]))
    log.info("Number of atoms is %i" % (INFOS["natom"]))
    INFOS["repr"] = initf.readline().split()[1]
    if INFOS["repr"].lower() == "mch":
        INFOS["diag"] = False
    else:
        INFOS["diag"] = True
    INFOS["eref"] = float(initf.readline().split()[1])
    INFOS["eharm"] = float(initf.readline().split()[1])

    # get guess for number of states
    line = initf.readline()
    if "states" in line.lower():
        states = []
        li = line.split()
        for i in range(1, len(li)):
            states.append(int(li[i]))
        guessstates = states
    else:
        guessstates = None

    log.info("Reference energy %16.12f a.u." % (INFOS["eref"]))
    log.info("Excited states are in %s representation.\n" % (["MCH", "diagonal"][INFOS["diag"]]))
    initf.seek(0)  # rewind the initf file
    INFOS["initf"] = initf

    # Number of states
    log.info(
        "\nPlease enter the number of states as a list of integers\ne.g. 3 0 3 for three singlets, zero doublets and three triplets."
    )
    while True:
        states = question("Number of states:", int, guessstates)
        if len(states) == 0:
            continue
        if any(i < 0 for i in states):
            log.info("Number of states must be positive!")
            continue
        break
    log.info("")
    nstates = 0
    for mult, i in enumerate(states):
        nstates += (mult + 1) * i

    print("\nPlease enter the molecular charge for each chosen multiplicity\ne.g. 0 +1 0 for neutral singlets and triplets and cationic doublets.")
    default = [i % 2 for i in range(len(states))]
    while True:
        charges = question("Molecular charges per multiplicity:", int, default)
        if not states:
            continue
        if len(charges) != len(states):
            print("Charges array must have same length as states array")
            continue
        break

    log.info("Number of states: " + str(states))
    log.info("Total number of states: %i\n" % (nstates))
    INFOS["states"] = states
    INFOS["nstates"] = nstates
    INFOS["charge"] = charges
    # obtain the statemap
    statemap = {}
    i = 1
    for imult, istate, ims in itnmstates(INFOS["states"]):
        statemap[i] = [imult, istate, ims]
        i += 1
    INFOS["statemap"] = statemap

    # get active states
    if question("Do you want all states to be active?", bool, True):
        INFOS["actstates"] = INFOS["states"]
    else:
        log.info(
            "\nPlease enter the number of ACTIVE states as a list of integers\ne.g. 3 0 3 for three singlets, zero doublets and three triplets."
        )
        while True:
            actstates = question("Number of states:", int)
            if len(actstates) != len(INFOS["states"]):
                log.info("Length of nstates and actstates must match!")
                continue
            valid = True
            for i, nst in enumerate(actstates):
                if not 0 <= nst <= INFOS["states"][i]:
                    log.info(
                        "Number of active states of multiplicity %i must be between 0 and the number of states of this multiplicity (%i)!"
                        % (i + 1, INFOS["states"][i])
                    )
                    valid = False
            if not valid:
                continue
            break
        INFOS["actstates"] = actstates
    isactive = []
    for imult in range(len(INFOS["states"])):
        for ims in range(imult + 1):
            for istate in range(INFOS["states"][imult]):
                isactive.append((istate + 1 <= INFOS["actstates"][imult]))
    INFOS["isactive"] = isactive
    log.info("")

    # ask whether initfile content is shown
    INFOS["show_content"] = question("Do you want to see the content of the initconds file?", bool, False)

    # read initlist, analyze it and log.info(content (all in get_initconds))
    INFOS["initf"] = initf
    INFOS = get_initconds(INFOS)

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
        if active and INFOS["n_issel"][i] > 0:
            defsetupstates.append(i + 1)
            nmax += INFOS["n_issel"][i]
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
        nsetupable = sum([INFOS["n_issel"][i - 1] for i in INFOS["setupstates"] if INFOS["isactive"][i - 1]])
        log.info("\nThere can be %i trajector%s set up.\n" % (nsetupable, ["y", "ies"][nsetupable != 1]))
        if nsetupable == 0:
            continue
        break

    # select range within initconds file
    # only start index needed, end index is determined by number of trajectories
    log.info("Please enter the index of the first initial condition in the initconds file to be setup.")
    while True:
        firstindex = question("Starting index:", int, [1])[0]
        if not 0 < firstindex <= INFOS["ninit"]:
            log.info("Please enter an integer between %i and %i." % (1, INFOS["ninit"]))
            continue
        nsetupable = 0
        for i, initcond in enumerate(INFOS["initlist"]):
            if i + 1 < firstindex:
                continue
            for state in set(setupstates):
                try:
                    nsetupable += initcond.statelist[state - 1].Excited
                except IndexError:
                    break
        log.info(
            "\nThere can be %i trajector%s set up, starting in %i states."
            % (nsetupable, ["y", "ies"][nsetupable != 1], len(INFOS["setupstates"]))
        )
        if nsetupable == 0:
            continue
        break
    INFOS["firstindex"] = firstindex

    # Number of trajectories
    log.info("\nPlease enter the total number of trajectories to setup.")
    while True:
        ntraj = question("Number of trajectories:", int, [nsetupable])[0]
        if not 1 <= ntraj <= nsetupable:
            log.info("Please enter an integer between %i and %i." % (1, nsetupable))
            continue
        break
    INFOS["ntraj"] = ntraj

    # Random number seed
    log.info('\nPlease enter a random number generator seed (type "!" to initialize the RNG from the system time).')
    while True:
        line = question("RNG Seed: ", str, "!", False)
        if line == "!":
            random.seed()
            break
        try:
            rngseed = int(line)
            random.seed(rngseed)
        except ValueError:
            log.info('Please enter an integer or "!".')
            continue
        break
    log.info("")

    return INFOS


def get_interface() -> SHARC_INTERFACE:
    "asks for interface and instantiates it"
    log.info("")
    log.info("{:-^60}".format("Choose the quantum chemistry interface"))
    log.info("\nPlease specify the quantum chemistry interface (enter any of the following numbers):")
    Interfaces = factory.get_available_interfaces()
    possible_numbers = []
    for i, (name, interface, possible) in enumerate(Interfaces):
        if not possible:
            log.info("% 3i %-20s %s" % (i+1, name, interface))
        else:
            log.info("% 3i %-20s %s" % (i+1, name, interface.description()))
            possible_numbers.append(i+1)
    log.info("")
    while True:
        num = question("Interface number:", int)[0]
        if num in possible_numbers:
            break
        else:
            log.info("Please input one of the following: %s!" % (possible_numbers))
    log.info("")
    log.info("The following interface was selected:")
    log.info("% 3i %-20s %s" % (num, Interfaces[num-1][0], Interfaces[num-1][1].description()))
    return Interfaces[num-1][1]


def get_requests(INFOS, interface: SHARC_INTERFACE) -> list[str]:
    """get requests for every single point"""
    interface.QMin.molecule['states'] = INFOS['states']
    int_features = interface.get_features(KEYSTROKES=KEYSTROKES)
    log.debug(int_features)
    
    INFOS["needed_requests"] = set()

    # Dynamics options
    string = "\n  " + "=" * 80 + "\n"
    string += "||" + f"{'Surface Hopping dynamics settings':^80}" + "||\n"
    string += "  " + "=" * 80 + "\n\n"
    log.info(string)


    # Method
    log.info(f"{'Nonadiabatic dynamics method':-^60}" + "\n")
    log.info('Please choose the dynamics method you want to employ.')
    cando = list(Method)
    for i in Method:
        log.info('%i\t%s' % (i, Method[i]['description']))
    while True:
        dyn=question('Method:',int,[1])[0]
        if dyn in Method and dyn in cando:
            break
        else:
            log.info('Please input one of the following: %s!' % ([i for i in cando]))
    INFOS['method']=Method[dyn]['name']
    # TODO: is SCP requiring any features?
    INFOS["needed_requests"].add("h")
    INFOS["needed_requests"].add("grad")
    INFOS["needed_requests"].add("dm")


    # Simulation time
    log.info(f"{'Simulation time':-^60}" + "\n")
    log.info("Please enter the total simulation time.")
    while True:
        num2 = question("Simulation time (fs):", float, [1000.0])[0]
        if num2 <= 0:
            log.info("Simulation time must be positive!")
            continue
        break
    INFOS["tmax"] = num2


    # Timestep
    log.info("\nPlease enter the simulation timestep (0.5 fs recommended).")
    while True:
        dt = question("Simulation timestep (fs):", float, [0.5])[0]
        if dt <= 0:
            log.info("Simulation timestep must be positive!")
            continue
        break
    INFOS["dtstep"] = dt
    log.info("\nSimulation will have %i timesteps." % (num2 // dt + 1))


    # Integrator
    log.info('\nPlease choose the integrator you want to use')
    cando = list(Integrator)
    for i in Integrator:
        log.info('%i\t%s' % (i, Integrator[i]['description']))
    while True:
        itg=question('Integrator:',int,[2])[0]
        if itg in Integrator and itg in cando:
            break
        else:
            log.info('Please input one of the following: %s!' % ([i for i in cando]))
    INFOS['integrator'] = itg    #Integrator[itg]['name']
    # some integrators do not work with all requests
    for forbidden in Integrator[INFOS['integrator']]["forbidden"]:
        if forbidden in int_features:
            log.info('Integrator is not compatible with feature "%s"' % forbidden)
            int_features.remove(forbidden)


    # convergence threshold
    if Integrator[INFOS['integrator']]["name"] == 'avv':
        while True:
            conv=question('Convergence threshold (eV):',float,[0.00005])[0]
            if conv<=0:
                log.info('Must be positive!')
                continue
            break
        log.info('\nConvergence threshold: %f.' % (conv))
        INFOS['convthre']=conv


    # number of substeps
    log.info("\nPlease enter the number of substeps for propagation (25 recommended).")
    while True:
        nsubstep = question("Nsubsteps:", int, [25])[0]
        if nsubstep <= 0:
            log.info("Enter a positive integer!")
            continue
        break
    INFOS["nsubstep"] = nsubstep


    # whether to kill relaxed trajectories
    log.info("\nThe trajectories can be prematurely terminated after they run for a certain time in the lowest state. ")
    INFOS["kill"] = question("Do you want to prematurely terminate trajectories?", bool, False)
    if INFOS["kill"]:
        while True:
            tkill = question("Kill after (fs):", float, [10.0])[0]
            if tkill <= 0:
                log.info("Must be positive!")
                continue
            break
        INFOS["killafter"] = tkill
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
    available = []
    for i in Couplings:
        if set(Couplings[i]["required"]).issubset(int_features):
            available.append(i)
    for i in Couplings:
        log.info("%i\t%s%s" % (i, Couplings[i]["description"], ["(not available)", ""][i in available]))
    log.info('')
    default = [available[-1]]
    while True:
        num = question("Coupling number:", int, default)[0]
        if num in Couplings and num in available:
            break
        else:
            log.info("Please input one of the following: %s!" % available)
    INFOS["coupling"] = num
    INFOS["needed_requests"].update(Couplings[num]["required"])


    # Phase tracking
    INFOS["phases_from_interface"] = False
    if Couplings[INFOS["coupling"]]["name"] != "overlap":
        if "phases" in int_features:
            INFOS["phases_from_interface"] = question("Do you want to track wavefunction phases through overlaps?", bool, True)
            if INFOS["phases_from_interface"]:
                INFOS["needed_requests"].add("phases")


    # Gradient correction (only for diagonal PESs)
    if INFOS["surf"] == "diagonal":
        available = []
        for i in GradCorrect:
            if set(GradCorrect[i]["required"]).issubset(int_features):
                available.append(i)
        for i in GradCorrect:
            log.info("%i\t%s%s" % (i, GradCorrect[i]["description"], ["(not available)", ""][i in available]))
        log.info('')
        # recommend ngt if nacdr are already calculated
        recommended = [available[0]]
        priority = ["tdh", "none", "ngt"]
        for name in priority:
            num = next((k for k, v in GradCorrect.items() if v["name"] == name), None)  # TODO: check if that works
            if set(GradCorrect[num]["required"]).issubset(INFOS["needed_requests"]):
                recommended = [num]
        while True:
            num = question("Coupling number:", int, recommended)[0]
            if num in GradCorrect and num in available:
                break
            else:
                log.info("Please input one of the following: %s!" % available)
        INFOS["gradcorrect"] = num
        INFOS["needed_requests"].update(GradCorrect[num]["required"])
    else:
        num = next((k for k, v in GradCorrect.items() if v["name"] == "none"), None)
        INFOS["gradcorrect"] = num
        INFOS["needed_requests"].update(GradCorrect[num]["required"])
    INFOS["needed_requests"].update(GradCorrect[INFOS["gradcorrect"]]["required"])




    #===============================
    # Begin Surface hopping details
    #=============================== 
    if INFOS['method']=='tsh':
        # Kinetic energy modification
        log.info(
            "\nDuring a surface hop, the kinetic energy has to be modified in order to conserve total energy. There are several options to that:"
        )
        cando = []
        for i in EkinCorrect:
            recommended = len(EkinCorrect[i]["required"]) == 0 or set(EkinCorrect[num]["required"]).issubset(INFOS["needed_requests"])
            possible = all([j in int_features for j in EkinCorrect[i]["required"]])
            if possible:
                cando.append(i)
            if not possible:
                log.info("%i\t%s%s" % (i, EkinCorrect[i]["description"], "\n\t(not possible)"))
            else:
                log.info("%i\t%s%s" % (i, EkinCorrect[i]["description"], ["\n\t(extra computational cost)", ""][recommended]))
        while True:
            ekinc = question("EkinCorrect:", int, [2])[0]
            if ekinc in EkinCorrect and ekinc in cando:
                break
            else:
                log.info("Please input one of the following: %s!" % ([i for i in cando]))
        INFOS["ekincorrect"] = ekinc
        if INFOS["ekincorrect"]:
            for i in EkinCorrect[INFOS["ekincorrect"]]["required"]:
                INFOS["needed_requests"].add(i)



        # frustrated reflection
        log.info(
            "\nIf a surface hop is refused (frustrated) due to insufficient energy, the velocity can either be left unchanged or reflected:"
        )
        cando = []
        for i in EkinCorrect:
            recommended = len(EkinCorrect[i]["required"]) == 0 or set(EkinCorrect[num]["required"]).issubset(INFOS["needed_requests"])
            possible = all([j in int_features for j in EkinCorrect[i]["required"]])
            if possible:
                cando.append(i)
            if not possible:
                log.info("%i\t%s%s" % (i, EkinCorrect[i]["description_refl"], "\n\t(not possible)"))
            else:
                log.info("%i\t%s%s" % (i, EkinCorrect[i]["description_refl"], ["\n\t(extra computational cost)", ""][recommended]))
        while True:
            reflect = question("Reflect frustrated:", int, [1])[0]
            if reflect in EkinCorrect and reflect in cando:
                break
            else:
                log.info("Please input one of the following: %s!" % ([i for i in cando]))
        INFOS["reflect"] = reflect
        if INFOS["reflect"]:
            for i in EkinCorrect[INFOS["ekincorrect"]]["required"]:
                INFOS["needed_requests"].add(i)


        # decoherence
        log.info("\nPlease choose a decoherence correction for the %s states:" % (["MCH", "diagonal"][INFOS["surf"] == "diagonal"]))
        cando = []
        for i in DecoherencesTSH:
            recommended = len(DecoherencesTSH[i]["required"]) == 0 or set(DecoherencesTSH[num]["required"]).issubset(INFOS["needed_requests"])
            possible = all([j in int_features for j in DecoherencesTSH[i]["required"]])
            if possible:
                cando.append(i)
            if not possible:
                log.info("%i\t%s%s" % (i, DecoherencesTSH[i]["description"], "\n\t(not possible)"))
            else:
                log.info("%i\t%s%s" % (i, DecoherencesTSH[i]["description"], ["\n\t(extra computational cost)", ""][recommended]))
        while True:
            decoh = question("Decoherence scheme:", int, [2])[0]
            if decoh in DecoherencesTSH and decoh in cando:
                break
            else:
                log.info("Please input one of the following: %s!" % ([i for i in cando]))
        INFOS["decoherence"] = [DecoherencesTSH[decoh]["name"], DecoherencesTSH[decoh]["params"]]
        for i in DecoherencesTSH[decoh]["required"]:
            INFOS["needed_requests"].add(i)


        # surface hopping scheme
        log.info("\nPlease choose a surface hopping scheme for the %s states:" % (["MCH", "diagonal"][INFOS["surf"] == "diagonal"]))
        cando = list(HoppingSchemes)
        for i in HoppingSchemes:
            log.info("%i\t%s" % (i, HoppingSchemes[i]["description"]))
        while True:
            hopping = question("Hopping scheme:", int, [2])[0]
            if hopping in HoppingSchemes and hopping in cando:
                break
            else:
                log.info("Please input one of the following: %s!" % ([i for i in cando]))
        INFOS["hopping"] = HoppingSchemes[hopping]["name"]


        # Forced hops to lowest state
        log.info("\nDo you want to perform forced hops to the lowest state based on a energy gap criterion?")
        log.info("(Note that this ignores spin multiplicity)")
        INFOS["force_hops"] = question("Forced hops to ground state?", bool, False)
        if INFOS["force_hops"]:
            INFOS["force_hops_dE"] = abs(question("Energy gap threshold for forced hops (eV):", float, [0.1])[0])
        else:
            INFOS["force_hops_dE"] = 9999.0


        # TODO: move out of the TSH/SCP if's
        # Scaling
        log.info("\nDo you want to scale the energies and gradients?")
        scal = question("Scaling?", bool, False)
        if scal:
            while True:
                fscal = question("Scaling factor (>0.0): ", float)[0]
                if fscal <= 0:
                    log.info("Please enter a positive real number!")
                    continue
                break
            INFOS["scaling_for_sharc"] = fscal
        else:
            INFOS["scaling_for_sharc"] = False


        # TODO: move out of the TSH/SCP if's
        # Damping
        log.info("\nDo you want to damp the dynamics (Kinetic energy is reduced at each timestep by a factor)?")
        damp = question("Damping?", bool, False)
        if damp:
            while True:
                fdamp = question("Scaling factor (0-1): ", float)[0]
                if not 0 <= fdamp <= 1:
                    log.info("Please enter a real number 0<=r<=1!")
                    continue
                break
            INFOS["damping"] = fdamp
        else:
            INFOS["damping"] = False


        # TODO: move out of the TSH/SCP if's?
        # atommask
        INFOS["atommaskarray"] = None
        if (INFOS["decoherence"][0] == "edc") or (INFOS["ekincorrect"] == 2) or (INFOS["reflect"] == 2):
            log.info("\nDo you want to use an atom mask for velocity rescaling or decoherence?")
            if question("Atom masking?", bool, False):
                log.info(
                    '\nPlease enter all atom indices (start counting at 1) of the atoms which should considered for velocity rescaling and dechoerence. \nRemember that you can also enter ranges (e.g., "-1~-3  5  11~21").'
                )
                #      log.info('\nPlease enter all atom indices (start counting at 1) of the atoms which should be masked. \nRemember that you can also enter ranges (e.g., "-1~-3  5  11~21").')
                arr = question("Masked atoms:", int, ranges=True)
                INFOS["atommaskarray"] = []
                for i in arr:
                    if 1 <= i <= INFOS["natom"]:
                        INFOS["atommaskarray"].append(i)


        # TODO: move out of the TSH/SCP if's? Or set sel_g/sel_t nonetheless?
        # selection of gradients (only for SHARC) and NACs (only if NAC=ddr)
        log.info("\n" + f"{'Selection of Gradients and NACs':-^60}" + "\n")
        log.info(
            """In order to speed up calculations, SHARC is able to select which gradients and NAC vectors it has to calculate at a certain timestep. The selection is based on the energy difference between the state under consideration and the classical occupied state.
    """
        )
        if INFOS["surf"] == "diagonal":
            if INFOS["soc"]:
                sel_g = question("Select gradients?", bool, False)
            else:
                sel_g = True
        else:
            sel_g = False
        INFOS["sel_g"] = sel_g
        if "nacdr" in INFOS["needed_requests"]:
            sel_t = question("Select non-adiabatic couplings?", bool, False)
        else:
            sel_t = False
        INFOS["sel_t"] = sel_t
        if sel_g or sel_t:
            if not sel_t and not INFOS["soc"]:
                INFOS["eselect"] = 0.001
                log.info("\nSHARC dynamics without SOC and NAC: setting minimal selection threshold.")
            else:
                log.info(
                    "\nPlease enter the energy difference threshold for the selection of gradients and non-adiabatic couplings (in eV). (0.5 eV recommended, or even larger if SOC is strong in this system.)"
                )
                eselect = question("Selection threshold (eV):", float, [0.5])[0]
                INFOS["eselect"] = abs(eselect)

    #===============================
    # End Surface hopping details
    #===============================

    #========================================
    # Begin Self-Consistent Potential Methods details 
    #========================================
    if INFOS['method']=='scp':

        # Nuclear EOM
        print('\nPlease choose the nuclear EOM propagator for SCP:')
        cando=list(Neom)
        for i in Neom:
            print('%i\t%s' % (i, Neom[i]['description']))
        while True:
            if INFOS['coupling']==3:
                eom=question('Neom:',int,[2])[0]
            else:
                eom=question('Neom:',int,[1])[0]
            if eom in Neom and eom in cando:
                break
            else:
                print('Please input one of the following: %s!' % (cando))
        INFOS['neom'] = Neom[eom]['name']

        # decoherence
        print('\nPlease choose a decoherence correction for the %s states:' % (['MCH','diagonal'][INFOS['surf']=='diagonal']))
        cando=[]
        for i in DecoherencesSCP:
            recommended = len(DecoherencesSCP[i]["required"]) == 0 or set(DecoherencesSCP[num]["required"]).issubset(INFOS["needed_requests"])
            possible = all([j in int_features for j in DecoherencesSCP[i]["required"]])
            if possible:
                cando.append(i)
            if not possible:
                print('%i\t%s%s' % (i, DecoherencesSCP[i]['description'],'\n\t(not possible)' ))
            else:
                print('%i\t%s%s' % (i, DecoherencesSCP[i]['description'],['\n\t(extra computational cost)',''][ recommended ]))
        while True:
            decoh=question('Decoherence scheme:',int,[2])[0]
            if decoh in DecoherencesSCP and decoh in cando:
                break
            else:
                print('Please input one of the following: %s!' % ([i for i in cando]))
        INFOS['decoherence']=[DecoherencesSCP[decoh]['name'],DecoherencesSCP[decoh]['params']]
        for i in DecoherencesSCP[decoh]["required"]:
            INFOS["needed_requests"].add(i)

        # surface switching scheme for decay of mixing methods
        if INFOS['decoherence'][0]=='dom':
            print('\nPlease choose a surface switching scheme for the %s states:' % (['MCH','diagonal'][INFOS['surf']=='diagonal']))
            cando=list(SwitchingSchemes)
            for i in SwitchingSchemes:
                print('%i\t%s' % (i, SwitchingSchemes[i]['description']))
            while True:
                switching=question('Switching scheme:',int,[2])[0]
                if switching in SwitchingSchemes and switching in cando:
                    break
                else:
                    print('Please input one of the following: %s!' % ([i for i in cando]))
            INFOS['switching'] = SwitchingSchemes[switching]['name']


        # decoherence time method
        if INFOS['decoherence'][0]=='dom':
            print('\nPlease choose a decoherence time scheme:')
            cando=list(DecotimeSCP)
            for i in DecotimeSCP:
                print('%i\t%s' % (i, DecotimeSCP[i]['description']))
            while True:
                decotimemethod=question('Decoherence time scheme:',int,[1])[0]
                if decotimemethod in DecotimeSCP and decotimemethod in cando:
                    break
                else:
                    print('Please input one of the following: %s!' % ([i for i in cando]))
            INFOS['decotime'] = DecotimeSCP[decotimemethod]['name']


        # gaussian width parameter for Decoherence time scheme=fp2
        if INFOS['decotime']=='fp2':
            width=question('Gaussian width (bohr^-2):',float,[6.0])[0]
            if width<=0:
                print('Must be positive!')
            print('\nGaussian width: %f.' % (width))
            INFOS['width']=width

        # Damping
        print('\nDo you want to damp the dynamics (Kinetic energy is reduced at each timestep by a factor)?')
        damp=question('Damping?',bool,False)
        if damp:
            while True:
                fdamp=question('Scaling factor (0-1): ',float)[0]
                if not 0<=fdamp<=1:
                    print('Please enter a real number 0<=r<=1!')
                    continue
                break
            INFOS['damping']=fdamp
        else:
            INFOS['damping']=False

    #===========================================
    # End Self-Consistent Potential Methods details 
    #===========================================


    # rattle file
    log.info(f"\n\n{'RATTLE':-^60}")
    INFOS["rattle"] = question("Do you want to constrain some bond lengths (via a RATTLE)?", bool, default=False)
    if INFOS["rattle"]:
        INFOS["rattlefile"] = question("specify path to rattle file: ", str, default="rattle", autocomplete=True)


    # thermostat
    INFOS["use_thermostat"] = question("Do you want to use a thermostat?", bool, False)
    if INFOS["use_thermostat"]:
        INFOS["thermostat"] = question("Specify the thermostat (available: 'langevin')", str, default="langevin").lower()
        INFOS["thermostat_temp"] = question("Please specify the desired temperature in Kelvin:", float, default=[298.15])[0]
        if INFOS["thermostat"] == "langevin":
            INFOS["thermostat_rng"] = question("Please enter an rng seed:", int, default=[1234])[0]
            INFOS["thermostat_friction"] = question("Please enter the friction coefficient [fs^-1]:", float, default=[0.02])[0]
            log.debug("regions not implemented")
            # if question("Do you want to use ", bool, False)


    # droplet potential
    if question("Do you want to use a droplet force?", bool, default=False):
        INFOS["droplet"] = True
        if question("Do you want to calculate the parameters from system size?", bool, default=False):
            density = question("Specify the density of your solvent [g/mL] (default: water at 298K)", float, default=[0.9974])[0]
            press_pascal = (
                question("Specify the desired pressure at the surface of the droplet in bar", float, default=[1])[0] * 100_000
            )
            wokness = question(
                "On a scale from 1 being a harmonic potential and 0 being a solid wall, how fast should the potential increase?",
                float,
                default=[0.2],
            )[0]
            molar_mass = question("Specify the molar-mass of your solvent [g/mol] (default: water)", float, default=[18.01528])[0]
            n_mol = question("How many molecules are in you simulation?", int)[0]
            r_drop = (
                (3 * (n_mol * (1 / (6.022_140_857e23 * (1000 * density / molar_mass) * 1e-27))) / (4 * math.pi)) ** (1 / 3)
            )
            r_off = r_drop * (1 - wokness)
            INFOS["droplet_radius"] = r_off
            INFOS["droplet_force"] = (press_pascal * 1e-20 * 4 * math.pi * r_drop ** 2) / (r_drop - r_off) / (
                8.2387235e-8 * 1.889726125
            )  # force in N/ang to Hartree/Bohr**2
            log.info(
                f"droplet_radius (potential free radius) = {INFOS['droplet_radius']} Å; droplet force {INFOS['droplet_force']} Hartree/Bohr^2"
            )
        else:
            INFOS["droplet_force"] = question("Specify the force in Hartree/Bohr^2", float)[0]
            INFOS["droplet_radius"] = question(
                "Specify the offset-radius for the droplet (beyond which the force is applied) [Angstrom]", float
            )[0]
        if question("Should all atoms be affected by the droplet force?", bool, default=True):
            INFOS["droplet_atoms"] = "all"
        else:
            INFOS["droplet_atoms"] = question("Specify the atom affected by the droplet force (list of atom indexes starting at 1)", int, ranges=True)


    # tether
    if question("Do you want to use a tether? (restraints groupg of atoms to a certian absolute coordinate)", bool, default=False):
        INFOS["tether"] = True
        INFOS["tether_force"] = question("Specify the force in Hartree/Bohr^2", float)[0]
        while True:
            INFOS["tether_position"] = question("Specify the tether position as 'x y z' in Å", float, default=[0., 0., 0.])
            if len(INFOS["tether_position"]) != 3:
                log.info("Please specify three numbers 'x y z' !")
                continue
            break
        INFOS["tether_radius"] = question(
            "Specify the offset-radius for the tether (beyond which the force is applied) (Angstrom)", float
        )[0]
        if question("Should all atoms be affected by the tether force?", bool, default=False):
            INFOS["tether_atoms"] = "all"
        else:
            INFOS["tether_atoms"] = question("Specify the atoms affected by the tether force (list of atom indexes starting at 1)", int, ranges=True)


    # Laser file
    log.info("\n\n" + f"{'Laser file':-^60}" + "\n")
    INFOS["laser"] = question("Do you want to include a laser field in the simulation?", bool, False)
    if INFOS["laser"]:
        log.info(
            """Please specify the file containing the complete laser field. The timestep in the file and the length of the file must fit to the simulation time, time step and number of substeps given above.

Laser files can be created using $SHARC/laser.x
"""
        )
        if os.path.isfile("laser"):
            if check_laserfile(
                "laser", int(INFOS["tmax"] / INFOS["dtstep"] * INFOS["nsubstep"] + 1), INFOS["dtstep"] / INFOS["nsubstep"]
            ):
                log.info('Valid laser file "laser" detected. ')
                usethisone = question("Use this laser file?", bool, True)
                if usethisone:
                    INFOS["laserfile"] = "laser"
        if "laserfile" not in INFOS:
            while True:
                filename = question("Laser filename:", str)
                if not os.path.isfile(filename):
                    log.info("File %s does not exist!" % (filename))
                    continue
                if check_laserfile(
                    filename, (INFOS["tmax"] / INFOS["dtstep"] * INFOS["nsubstep"] + 1), INFOS["dtstep"] / INFOS["nsubstep"]
                ):
                    break
            INFOS["laserfile"] = filename
        # only the analytical interface can do dipole gradients
        if "dipolegrad" in int_features:
            INFOS["dipolegrad"] = question("Do you want to use dipole moment gradients?", bool, False)
        else:
            INFOS["dipolegrad"] = False
        log.info("")
    else:
        INFOS["dipolegrad"] = False
    if INFOS["dipolegrad"]:
        INFOS["needed_requests"].add("dmdr")


    # Setup Dyson computation
    INFOS["ion"] = False
    if "dyson" in int_features:
        n = [0, 0]
        for i, j in enumerate(INFOS["states"]):
            n[i % 2] += j
        if n[0] >= 1 and n[1] >= 1:
            log.info("\n" + f"{'Ionization probability by Dyson norms':-^60}" + "\n")
            log.info("Do you want to compute Dyson norms between neutral and ionic states?")
            INFOS["ion"] = question("Dyson norms?", bool, False)
            if INFOS["ion"]:
                INFOS["needed_requests"].add("ion")


    # Setup theodore
    if "theodore" in int_features:
        log.info("\n" + f"{'TheoDORE wave function analysis':-^60}" + "\n")
        log.info("Do you want to run TheoDORE to obtain one-electron descriptors for the electronic wave functions?")
        INFOS["theodore"] = question("TheoDORE?", bool, False)
        if INFOS["theodore"]:
            INFOS["needed_requests"].add("theodore")


    string = "\n  " + "=" * 80 + "\n"
    string += "||" + f"{'Interface setup':^80}" + "||\n"
    string += "  " + "=" * 80 + "\n\n"
    log.info(string)

    return INFOS


def get_trajectory_info(INFOS) -> dict:

    # PYSHARC
    string = "\n  " + "=" * 80 + "\n"
    string += "||" + f"{'PYSHARC':^80}" + "||\n"
    string += "  " + "=" * 80 + "\n"
    log.info(string)
    pysharc_possible = True
    if Integrator[INFOS['integrator']]["name"] == 'avv':
        log.info("Pysharc not possible with adaptive time step integrator.")
        pysharc_possible = False
    if pysharc_possible:
        log.info("\nThe chosen interface can be run very efficiently with PYSHARC.")
        log.info("PYSHARC runs the SHARC dynamics directly within Python (with C and Fortran extension)")
        log.info("with minimal file I/O for maximum performance.")
        INFOS["pysharc"] = question("Setup for PYSHARC?", bool, True)
    else:
        INFOS["pysharc"] = False


    # Dynamics options
    string = "\n  " + "=" * 80 + "\n"
    string += "||" + f"{'Content of output.dat files':^80}" + "||\n"
    string += "  " + "=" * 80 + "\n"
    log.info(string)


    # NetCDF
    log.info("\nSHARC or PYSHARC can produce output in ASCII format (all features supported currently)")
    log.info("or in NetCDF format (more efficient file I/O, some features currently not supported).")
    INFOS["netcdf"] = question("Write output in NetCDF format?", bool, INFOS["pysharc"])


    # options for writing to output.dat
    log.info("\nDo you want to write the gradients to the output.dat file ?")
    write_grad = question("Write gradients?", bool, False)
    if write_grad:
        INFOS["write_grad"] = True
    else:
        INFOS["write_grad"] = False

    log.info("\nDo you want to write the non-adiabatic couplings (NACs) to the output.dat file ?")
    write_NAC = question("Write NACs?", bool, False)
    if write_NAC:
        INFOS["write_NAC"] = True
    else:
        INFOS["write_NAC"] = False

    log.info("\nDo you want to write property matrices to the output.dat file  (e.g., Dyson norms)?")
    if "ion" in INFOS and INFOS["ion"]:
        INFOS["write_property2d"] = question("Write property matrices?", bool, True)
    else:
        INFOS["write_property2d"] = question("Write property matrices?", bool, False)

    log.info("\nDo you want to write property vectors to the output.dat file  (e.g., TheoDORE results)?")
    if "theodore" in INFOS and INFOS["theodore"]:
        INFOS["write_property1d"] = question("Write property vectors?", bool, True)
    else:
        INFOS["write_property1d"] = question("Write property vectors?", bool, False)

    log.info("\nDo you want to write the overlap matrix to the output.dat file ?")
    INFOS["write_overlap"] = question("Write overlap matrix?", bool, (Couplings[INFOS["coupling"]]["name"] == "overlap"))

    log.info("\nDo you want to modify the output.dat writing stride?")
    stride = question("Modify stride?", bool, False)
    if stride:
        INFOS["stride"] = []
        stride = question('Enter the  *INITIAL*   output stride (e.g., "1"=write every step)', int, [1])
        INFOS["stride"].extend(stride)
        stride = question(
            'Enter the *SUBSEQUENT* output stride (e.g., "10 2"=write every second step starting at step 10)', int, [0, 1]
        )
        INFOS["stride"].extend(stride)
        stride = question(
            'Enter the   *FINAL*    output stride (e.g., "100 10"=write every tenth step starting at step 100)', int, [0, 1]
        )
        INFOS["stride"].extend(stride)
    else:
        INFOS["stride"] = [1]


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


def writeSHARCinput(INFOS, initobject, iconddir, istate, ask=False):
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
    s += "rngseed %i\n\n" % (random.randint(-32768, 32767))
    s += "ezero %18.10f\n" % (INFOS["eref"])

    s += "tmax %f\nstepsize %f\nnsubsteps %i\n" % (INFOS["tmax"], INFOS["dtstep"], INFOS["nsubstep"])
    s += 'integrator %s\n' % (Integrator[INFOS['integrator']]["name"])
    if Integrator[INFOS['integrator']]["name"] == 'avv':
        s += 'convthre %s\n' % (INFOS['convthre'])
    if INFOS["kill"]:
        s += "killafter %f\n" % (INFOS["killafter"])
    s += "\n"


    # general dynamics settings
    s += 'method %s\n' % (INFOS['method'])
    s += "surf %s\n" % (INFOS["surf"])
    s += "coupling %s\n" % (Couplings[INFOS["coupling"]]["name"])
    if GradCorrect[INFOS['gradcorrect']]['name'] == 'none':
        s += 'nogradcorrect\n'
    else:
        s += 'gradcorrect %s\n' % (GradCorrect[INFOS['gradcorrect']]['name'])

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

    if INFOS['method'] == 'scp':
        s += 'pointer_basis %s\n' % (INFOS['pointer_basis'])
        s += 'neom_rep %s\n' % (INFOS['neom_rep'])
        s += 'neom %s\n' % (INFOS['neom'])
        s += 'decoherence_scheme %s\n' % (INFOS['decoherence'][0])
        if INFOS['decoherence'][1]:
            s += 'decoherence_param %s\n' % (INFOS['decoherence'][1])
        if INFOS['decoherence'][0]=='dom':
            s += 'switching_procedure %s\n' % (INFOS['switching'])
        if INFOS['decoherence'][0]=='dom':
            s += 'decotime_method %s\n' % (INFOS['decotime'])
            if INFOS['decotime']=='fp2':
                s += 'gaussian_width %s\n' % (INFOS['width'])
        if INFOS['damping']:
            s += 'dampeddyn %f\n' % (INFOS['damping'])

    if INFOS["pysharc"]:
        s += "notrack_phase\n"

    if INFOS["sel_g"]:
        s += "grad_select\n"
    else:
        s += "grad_all\n"
    if INFOS["sel_t"]:
        s += "nac_select\n"
    else:
        if "nacdr" in INFOS["needed_requests"]:
            s += "nac_all\n"
    if "eselect" in INFOS:
        s += "eselect %f\n" % (INFOS["eselect"])

    if INFOS["select_directly"]:
        s += "select_directly\n"

    if not INFOS["soc"]:
        s += "nospinorbit\n"

    if INFOS["write_grad"]:
        s += "write_grad\n"
    if INFOS["write_NAC"]:
        s += "write_nacdr\n"
    if INFOS["write_overlap"]:
        s += "write_overlap\n"
    if INFOS["write_property1d"]:
        s += "write_property1d\n"
        if "theodore.count" in INFOS:
            s += "n_property1d %i\n" % (INFOS["theodore.count"])
        else:
            s += "n_property1d %i\n" % (1)
    if INFOS["write_property2d"]:
        s += "write_property2d\n"
        s += "n_property2d %i\n" % (1)

    # NetCDF or ASCII
    if INFOS["netcdf"]:
        out = "netcdf"
    else:
        out = "ascii"
    s += "output_format %s\n" % out

    # stride
    if "stride" in INFOS:
        s += "output_dat_steps"
        for i in INFOS["stride"]:
            s += " %i" % i
        s += "\n"

    # rattle
    if INFOS["rattle"]:
        s += f'rattle\nrattlefile "{INFOS["rattlefile"].split("/")[-1]}"\n'

    s += "\n"

    # laser
    if INFOS["laser"]:
        s += "laser external\n"
        s += 'laserfile "laser"\n'
        if INFOS["dipolegrad"]:
            s += "dipole_gradient\n"
        s += "\n"

    # Dyson norms
    if "ion" in INFOS and INFOS["ion"]:
        s += "ionization\n"
        s += "ionization_step 1\n"

    # TheoDORE
    if "theodore" in INFOS and INFOS["theodore"]:
        s += "theodore\n"
        s += "theodore_step 1\n"

    # Thermostat
    if INFOS["use_thermostat"]:
        s += f"thermostat {INFOS['thermostat']}\n"
        s += f"temperature {INFOS['thermostat_temp']:.2f}\n"
        if INFOS["thermostat"] == "langevin":
            s += f"rngseed_thermostat {INFOS['thermostat_rng']}\n"
            s += f"thermostat_const {INFOS['thermostat_friction']}\n"
        s += "\n"

    # Droplet and tether
    if 'droplet' in INFOS and 'tether' in INFOS:
        s += "restrictive_potential droplet_tether\n"
    elif 'droplet' in INFOS:
        s += "restrictive_potential droplet\n"
    elif 'tether' in INFOS:
        s += "restrictive_potential tether\n"

    if 'droplet' in INFOS:
        s += f"restricted_droplet_force {INFOS['droplet_force']: 8.6e}\n"
        s += f"restricted_droplet_radius {INFOS['droplet_radius']: .6f}\n"
        if type(INFOS['droplet_atoms']) is list:
            if len(INFOS['droplet_atoms']) < 11:
                s += f"restricted_droplet_atoms {' '.join(map(str, INFOS['droplet_atoms']))}\n"
            else:
                with open(iconddir + "/droplet_atoms", "w") as f:
                    f.write("\n".join(map(lambda x: "T" if x in INFOS['droplet_atoms'] else "F", range(1,
                            len(initobject.atomlist)+1))))
                s += "restricted_droplet_atoms file \"droplet_atoms\"\n"
        else:
            s += f"restricted_droplet_atoms {INFOS['droplet_atoms']}\n"

    if 'tether' in INFOS:
        s += f"tethering_force {INFOS['tether_force']: 8.6e}\n"
        s += f"tethering_radius {INFOS['tether_radius']: 8.6f}\n"
        if type(INFOS['tether_atoms']) is list:
            if len(INFOS['tether_atoms']) < 11:
                s += f"tether_at {' '.join(map(str, INFOS['tether_atoms']))}\n"
            else:
                with open(iconddir + "/tether_atoms", "w") as f:
                    f.write("\n".join(map(lambda x: "T" if x in INFOS['tether_atoms'] else "F", range(1,
                            len(initobject.atomlist)+1))))
                s += "tether_at file \"tether_atoms\"\n"
        else:
            s += f"tether_at {INFOS['tether_atoms']}\n"
        s += f"tethering_position {' '.join(map(lambda x: f'{x: 8.6f}', INFOS['tether_position']))}\n"



    # let user look at input and add extra stuff
    if ask:
        if question("Do you want to see the input for the first trajectory?", bool, default=False):
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
    if INFOS["laser"]:
        laserfname = iconddir + "/laser"
        shutil.copy(INFOS["laserfile"], laserfname)

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

    return


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

    width = 50
    ntraj = INFOS["ntraj"]
    idone = 0
    finished = False

    initlist = INFOS["initlist"]
    ask = True

    for icond in range(INFOS["firstindex"], INFOS["ninit"] + 1):
        for istate in INFOS["setupstates"]:
            if len(initlist[icond - 1].statelist) < istate:
                continue
            if not initlist[icond - 1].statelist[istate - 1].Excited:
                continue

            idone += 1

            done = idone * width // ntraj
            sys.stdout.write("\rProgress: [" + "=" * done + " " * (width - done) + "] %3i%%" % (done * 100 // width))

            dirname = get_iconddir(istate, INFOS) + "/TRAJ_%05i/" % (icond)
            io = make_directory(dirname)
            if io != 0:
                log.info("Skipping initial condition %i %i!" % (istate, icond))
                continue

            writeSHARCinput(INFOS, initlist[icond - 1], dirname, istate, ask=ask)
            ask = False
            io = make_directory(dirname + "/QM")
            io += make_directory(dirname + "/restart")
            if io != 0:
                log.info("Could not make QM or restart directory!")
                continue
            interface.prepare(INFOS, dirname + "/QM")
            
            if INFOS["pysharc"]:
                run_qm = open(dirname + "/QM/runQM.sh", "w")
                string = "cd QM\n$SHARC/%s.py QM.in >> QM.log 2>>QM.err\nerr=$?\n\nexit $err" % (interface.__class__.__name__)                
                run_qm.write(string)                               

            writeRunscript(INFOS, dirname, interface)
            if INFOS["rattle"]:
                shutil.copy(expand_path(INFOS["rattlefile"]), os.path.join(dirname, INFOS["rattlefile"].split("/")[-1]))

            string = "cd $CWD/%s/\nbash run.sh\ncd $CWD\necho %s >> DONE\n" % (dirname, dirname)
            all_run.write(string)
            if INFOS["qsub"]:
                string = "cd $CWD/%s/\n%s run.sh\ncd $CWD\n" % (dirname, INFOS["qsubcommand"])
                all_qsub.write(string)

            if idone == ntraj:
                finished = True
                break
        if finished:
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
                INFOS["firstindex"],
                icond,
                ntraj,
                istate,
            )
            setup_stat.write(string)
            setup_stat.close()
            break

    all_run.close()
    filename = "all_run_traj.sh"
    os.chmod(filename, os.stat(filename).st_mode | stat.S_IXUSR)
    if INFOS["qsub"]:
        all_qsub.close()
        filename = "all_qsub_traj.sh"
        os.chmod(filename, os.stat(filename).st_mode | stat.S_IXUSR)

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

    INFOS = get_general(INFOS)
    chosen_interface: SHARC_INTERFACE = get_interface()()
    INFOS = get_requests(INFOS, chosen_interface)
    INFOS = chosen_interface.get_infos(INFOS, KEYSTROKES)
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

    close_keystrokes()


# ======================================================================================================================
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log.info("\nCtrl+C makes me a sad SHARC ;-(\n")
        quit(0)
