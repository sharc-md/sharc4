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

import math
import re
import os
import sys
import shutil
import random
import datetime
import time

import numpy as np

from constants import CM_TO_HARTREE, HARTREE_TO_EV, U_TO_AMU, BOHR_TO_ANG
from qmout import QMout
from utils import itnmstates, question as question_def


# =========================================================0
# some constants
DEBUG = False
PI = math.pi

version = "4.0"
versionneeded = [0.2, 1.0, 2.0, 2.1, 3.0, float(version)]
versiondate = datetime.date(2025, 4, 1)

# ======================================================================================================================
# ======================================================================================================================
# ======================================================================================================================


def try_read(l, index, typefunc, default):
    try:
        if typefunc == bool:
            return "True" == l[index]
        else:
            return typefunc(l[index])
    except IndexError:
        return typefunc(default)
    except ValueError:
        print("Could not initialize object!")
        quit(1)


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
        if erange[0] <= self.Eexc <= erange[1]:
            self.Excited = random.random() < (self.Prob / max_Prob)
        else:
            self.Excited = False


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
            if line == "\n":
                continue
            # if 'Index     %i' % (index) in line:
            if line.startswith("Index") and int(line.split()[-1]) == index:
                break
            if line == "":
                print("Initial condition %i not found in file %s" % (index, f.name))
                quit(1)
        f.readline()  # skip one line, where "Atoms" stands
        atomlist = []
        self.Ekin = 0.0
        while True:
            line = f.readline()
            if line.startswith("States"):
                break
            m, vx, vy, vz = line.split()[-4:]
            self.Ekin += 0.5 * float(m) * U_TO_AMU * (float(vx) ** 2 + float(vy) ** 2 + float(vz) ** 2)
            atomlist.append(line)
        # statelist = []
        while True:
            line = f.readline()
            if line.startswith("Ekin"):
                break
            # state = STATE()
            # state.init_from_str(line)
            # statelist.append(state)
        epot_harm = 0.0
        while line and line != "\n":
            line = f.readline()
            if "epot_harm" in line.lower():
                epot_harm = float(line.split()[1])
                break
        self.atomlist = atomlist
        self.eref = eref
        self.Epot_harm = epot_harm
        self.natom = len(atomlist)
        # self.Ekin = sum([atom.Ekin for atom in self.atomlist])
        # self.statelist = statelist
        # self.nstate = len(statelist)
        # if self.nstate > 0:
            # self.Epot = self.statelist[0].e - self.eref
        # else:
            # self.Epot = epot_harm

    def __str__(self):
        s = "Atoms\n" + "".join(self.atomlist)
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


def get_statemap(states):
    statemap = {}
    i = 1
    for imult, istate, ims in itnmstates(states):
        statemap[i] = [imult, istate, ims]
        i += 1
    return statemap


def print_statemap(statemap, diag=False):
    n = len(statemap)
    if diag:
        s = "# State map for diagonal states:\n#State\tQuant\n"
        for i in range(1, n + 1):
            s += "%i\t%i\n" % (i, i)
    else:
        s = "# State map for MCH states:\n#State\tMult\tM_s\tQuant\n"
        for i in range(1, n + 1):
            (mult, state, ms) = statemap[i]
            s += "%i\t%i\t%+3.1f\t%i\n" % (i, mult, ms, state)
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
    string = "\n"
    string += "  " + "=" * 80 + "\n"
    string += "||" + "{:^80}".format("") + "||\n"
    string += "||" + "{:^80}".format("Excite initial conditions for SHARC") + "||\n"
    string += "||" + "{:^80}".format("") + "||\n"
    string += "||" + "{:^80}".format("Author: Sebastian Mai") + "||\n"
    string += "||" + "{:^80}".format("") + "||\n"
    string += "||" + "{:^80}".format("Version:" + version) + "||\n"
    string += "||" + "{:^80}".format(versiondate.strftime("%d.%m.%y")) + "||\n"
    string += "||" + "{:^80}".format("") + "||\n"
    string += "  " + "=" * 80 + "\n\n"
    string += """
This script automatizes to read-out the results of initial excited-state calculations for SHARC.
It calculates oscillator strength (in MCH and diagonal basis) and stochastically
determines initial states for trajectories.
  """
    print(string)


# ======================================================================================================================
# ======================================================================================================================
# ======================================================================================================================


def open_keystrokes():
    global KEYSTROKES
    KEYSTROKES = open("KEYSTROKES.tmp", "w")


def close_keystrokes():
    KEYSTROKES.close()
    shutil.move("KEYSTROKES.tmp", "KEYSTROKES.excite")


# ===================================


def question(question, typefunc, default=None, autocomplete=True, ranges=False):
    return question_def(question, typefunc, KEYSTROKES, default, autocomplete, ranges)


# ======================================================================================================================
# ======================================================================================================================
# ======================================================================================================================


def get_infos(INFOS):
    """This routine asks for the paths of the initconds file and ICONDS directory, for energy window and the representation."""

    print("{:-^60}".format("Initial conditions file") + "\n")
    # open the initconds file
    try:
        initfile = "initconds"
        initf = open(initfile)
        line = initf.readline()
        if check_initcond_version(line):
            print('Initial conditions file "initconds" detected. Do you want to use this?')
            if not question('Use file "initconds"?', bool, True):
                initf.close()
                raise IOError
        else:
            initf.close()
            raise IOError
    except IOError:
        print("\nIf you do not have an initial conditions file, prepare one with wigner.py!\n")
        print("Please enter the filename of the initial conditions file.")
        while True:
            initfile = question("Initial conditions filename:", str, "initconds")
            initfile = os.path.expanduser(os.path.expandvars(initfile))
            if os.path.isdir(initfile):
                print("Is a directory: %s" % (initfile))
                continue
            if not os.path.isfile(initfile):
                print("File does not exist: %s" % (initfile))
                continue
            try:
                initf = open(initfile, "r")
            except IOError:
                print("Could not open: %s" % (initfile))
                continue
            line = initf.readline()
            if check_initcond_version(line):
                break
            else:
                print("File does not contain initial conditions!")
                continue
    # read the header
    INFOS["ninit"] = int(initf.readline().split()[1])
    INFOS["natom"] = int(initf.readline().split()[1])
    INFOS["repr"] = initf.readline().split()[1]
    INFOS["eref"] = float(initf.readline().split()[1])
    INFOS["eharm"] = float(initf.readline().split()[1])

    # get guess for number of states
    line = initf.readline()
    if "states" in line.lower():
        states = []
        l = line.split()
        for i in range(1, len(l)):
            states.append(int(l[i]))
        INFOS["states"] = states
    else:
        INFOS["states"] = None

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
    initf.seek(0)  # rewind the initf file
    INFOS["initf"] = initf
    print('\nFile "%s" contains %i initial conditions.' % (initfile, INFOS["ninit"]))
    print("Number of atoms is %i\n" % (INFOS["natom"]))

    print("{:-^60}".format("Generate excited state lists") + "\n")
    print(
        """Using the following options, excited state lists can be added to the initial conditions:

1       Generate a list of dummy states
2       Read excited-state information from ab initio calculations (from setup_init.py)"""
    )
    allowed = [1, 2]
    guess_gen = [2]
    if any([i in INFOS["repr"].lower() for i in ["mch", "diag"]]):
        allowed.append(3)
        print("3       Keep existing excited-state information")
        guess_gen = [3]
    print("")
    while True:
        INFOS["gen_list"] = question("How should the excited-state lists be generated?", int, guess_gen)[0]
        if not INFOS["gen_list"] in allowed:
            print("Please give one of the following integer: %s" % (allowed))
            continue
        break

    if INFOS["gen_list"] == 1:
        INFOS["read_QMout"] = False
        INFOS["make_list"] = True
    elif INFOS["gen_list"] == 2:
        INFOS["read_QMout"] = True
        INFOS["make_list"] = False
    elif INFOS["gen_list"] == 3:
        INFOS["read_QMout"] = False
        INFOS["make_list"] = False

    if INFOS["read_QMout"]:
        print("Please enter the path to the directory containing the ICOND subdirectories.")
        while True:
            path = question("Path to ICOND directories:", str)
            path = os.path.expanduser(os.path.expandvars(path))
            path = os.path.abspath(path)
            if not os.path.isdir(path):
                print("Is not a directory or does not exist: %s" % (path))
                continue
            else:
                ls = os.listdir(path)
                n = 0
                for i in ls:
                    if "ICOND" in i:
                        n += 1
                if n == 0:
                    print("Does not contain any ICOND directories: %s" % (path))
                    continue
                else:
                    break
        print("\n%s\nDirectory contains %i subdirectories." % (path, n))
        if n < INFOS["ninit"] + 1:
            print("There are more initial conditions in %s." % (initfile))
        INFOS["iconddir"] = path
        INFOS["ncond"] = n
        print("")

    if INFOS["make_list"]:
        print(
            "\nPlease enter the number of states as a list of integers\ne.g. 3 0 3 for three singlets, zero doublets and three triplets."
        )
        while True:
            states = question("Number of states:", int)
            if len(states) == 0:
                continue
            if any(i < 0 for i in states):
                print("Number of states must be positive!")
                continue
            break
        print("")
        nstates = 0
        for mult, i in enumerate(states):
            nstates += (mult + 1) * i
        print("Number of states: " + str(states))
        print("Total number of states: %i\n" % (nstates))
        print("")
        INFOS["states"] = states
        INFOS["nstates"] = nstates

    if INFOS["make_list"] or INFOS["read_QMout"]:
        print("{:-^60}".format("Excited-state representation"))
        if INFOS["read_QMout"]:
            print(
                """\nThis script can calculate the excited-state energies and oscillator strengths in two representations.
These representations are:
- MCH representation: Only the diagonal elements of the Hamiltonian are taken into account. The states are the spin-free states as calculated in the quantum chemistry code. This option should be used if the ground state is spin-pure.
- diagonal representation: The Hamiltonian including spin-orbit coupling is diagonalized. The states are spin-corrected, fully adiabatic. Note that for this the excited-state calculations have to include spin-orbit couplings. 
"""
            )
        else:
            print(
                """\nThis script needs to set the electronic state representation.
There are two representations:
- MCH representation: Only the diagonal elements of the Hamiltonian are taken into account. The states are the spin-free states as calculated in the quantum chemistry code. This option should be used if the ground state is spin-mixed.
- diagonal representation: The Hamiltonian including spin-orbit coupling is diagonalized. The states are spin-corrected, fully adiabatic. Note that for this the excited-state calculations have to include spin-orbit couplings. 
"""
            )
        INFOS["diag"] = question("Do you want to use the diagonal representation (True=diag, False=MCH)?", bool, default = False)
        if INFOS["diag"] and INFOS["read_QMout"]:
            qmfilename = INFOS["iconddir"] + "/ICOND_00000/QM.in"
            if os.path.isfile(qmfilename):
                soc_there = False
                qmfile = open(qmfilename, "r")
                for line in qmfile:
                    if "soc" in line.lower():
                        soc_there = True
                qmfile.close()
                if not soc_there:
                    print(
                        "\nDiagonal representation specified, but \n%s\n says there are no SOCs in the QM.out files.\nUsing MCH representation."
                        % (qmfilename)
                    )
                    INFOS["diag"] = False
                    time.sleep(2)
            else:
                print("Could not determine whether calculations include SOC.")
        print("")
        if INFOS["diag"]:
            INFOS["repr"] = "diag"
        else:
            INFOS["repr"] = "MCH"

        if INFOS["read_QMout"]:
            qmfilename = INFOS["iconddir"] + "/ICOND_00000/QM.in"
            if os.path.isfile(qmfilename):
                qmfile = open(qmfilename, "r")
                for line in qmfile:
                    if re.search(r"^\\s?ion\\s?", line.lower()):
                        INFOS["ion"] = question("Use ionization probabilities instead of dipole moments?", bool, False)
                    if "states" in line.lower():
                        states = []
                        l = line.split()
                        for i in range(1, len(l)):
                            states.append(int(l[i]))
                        INFOS["states"] = states
                qmfile.close()
            if "ion" not in INFOS:
                INFOS["ion"] = False

        print("\n" + "{:-^60}".format("Reference energy") + "\n")
        if INFOS["read_QMout"]:
            qmfilename = INFOS["iconddir"] + "/ICOND_00000/QM.out"
        if INFOS["make_list"]:
            eref_from_file = question("Do you have conducted an ab initio calculation at the equilibrium geometry?", bool)
            if eref_from_file:
                while True:
                    qmfilename = question("Path to the QM.out file of the calculation:", str)
                    if not os.path.isfile(qmfilename):
                        print("File %s does not exist!" % (qmfilename))
                        continue
                    break
            else:
                qmfilename = ""
        if os.path.isfile(qmfilename):
            qmout = QMout(filepath=qmfilename)
            H = qmout.h
            DM = qmout.dm
            if H is not None:
                if INFOS["diag"]:
                    P = qmout.ion
                    eig, U = np.linalg.eigh(H)
                    Ucon = np.conjugate(U)
                    DM = np.einsum("kij,in,jm->knm", DM, Ucon, U)
                    P = np.einsum("kij,in,jm->knm", P, Ucon, U)
                INFOS["eref"] = H[0][0].real
                print("Reference energy read from file \n%s" % (qmfilename))
                print("E_ref= %16.12f" % (INFOS["eref"]))
        else:
            print("\nPlease enter the ground state equilibrium energy in hartree.")
            INFOS["eref"] = question("Reference energy (hartree): ", float)[0]
        print("")

    print("\n" + "{:-^60}".format("Excited-state selection") + "\n")
    print(
        """Using the following options, the excited states can be flagged as valid initial states for dynamics:

1       Unselect all initial states
2       Provide a list of desired initial states"""
    )
    allowed = [1, 2]
    guess_gen = [2]
    if not INFOS["make_list"]:
        print("3       Simulate delta-pulse excitation based on excitation energies and oscillator strengths")
        allowed.append(3)
        guess_gen = [3]
    if not INFOS["make_list"] and not INFOS["read_QMout"]:
        print("4       Keep selection (i.e., only print statistics on the excited states and exit)")
        allowed.append(4)
    print("")
    while True:
        INFOS["excite"] = question("How should the excited states be flagged?", int, guess_gen)[0]
        if not INFOS["excite"] in allowed:
            print("Please give one of the following integer: %s" % (allowed))
            continue
        break
    print("")

    if INFOS["excite"] == 1:
        INFOS["allowed"] = set()
        INFOS["erange"] = [-2.0, -1.0]

    if INFOS["excite"] == 3 or (INFOS["excite"] == 2 and not INFOS["make_list"]) or INFOS["excite"] == 4:
        print("\n" + "{:-^60}".format("Excitation window"))
        if INFOS["excite"] == 4:
            print("\nEnter the energy window for counting.")
        else:
            print("\nEnter the energy window for exciting the trajectories.")
        while True:
            erange = question("Range (eV):", float, [0.0, 10.0])
            if erange[0] >= erange[1]:
                print("Range empty!")
                continue
            break
        print("\nScript will allow excitations only between %f eV and %f eV.\n" % (erange[0], erange[1]))
        erange[0] /= HARTREE_TO_EV
        erange[1] /= HARTREE_TO_EV
        INFOS["erange"] = erange

    INFOS["diabatize"] = False
    if INFOS["excite"] == 2:
        print("\n" + "{:-^60}".format("Considered states"))

        if INFOS["read_QMout"] and INFOS["repr"] == "MCH":
            qmfilename = INFOS["iconddir"] + "/ICOND_00001/QM.in"
            if os.path.isfile(qmfilename):
                qmfile = open(qmfilename, "r")
                for line in qmfile:
                    if re.search(r"^\\s?overlap\\s?", line.lower()):
                        print(
                            "\nThe vertical excitation calculations were done with overlaps with a reference.\nReference overlaps can be used to obtain diabatic states.\n"
                        )
                        INFOS["diabatize"] = question(
                            "Do you want to specify the initial states in a diabatic picture?", bool, False
                        )
                qmfile.close()

        print(
            """\nPlease give a list of all states which should be
flagged as valid initial states for the dynamics.
Note that this is applied to all initial conditions."""
        )
        if INFOS["diabatize"]:
            print(
                "\nNOTE: These numbers are interpreted as diabatic states.\nThe diabatic basis is the set of states computed in ICOND_00000/.\nPlease carefully analyze these states to decide which diabatic states to request."
            )
            # print('NOTE: You can only enter one initial state.')
        if "states" in INFOS:
            diago = INFOS["repr"] == "diag"
            print(print_statemap(get_statemap(INFOS["states"]), diag=diago))

        while True:
            allowed_states = question("List of initial states:", int, ranges=True)
            if any([i <= 0 for i in allowed_states]):
                print("State indices must be positive!")
                continue
            # if INFOS['diabatize']:
            # if len(allowed_states)>1:
            # print('Only one initial state allowed!')
            # continue
            break
        INFOS["allowed"] = set(allowed_states)
        if "erange" not in INFOS:
            INFOS["erange"] = [float("-inf"), float("inf")]

    if INFOS["read_QMout"]:
        print("{:-^60}".format("Considered states") + "\n")
        print(
            "From which state should the excitation originate (for computation of excitation energies and oscillator strength)?"
        )
        INFOS["initstate"] = question("Lower state for excitation?", int, [1])[0] - 1
    else:
        INFOS["initstate"] = 0

    if INFOS["excite"] == 3:
        if "states" in INFOS:
            diago = INFOS["repr"] == "diag"
            print(print_statemap(get_statemap(INFOS["states"]), diag=diago))
        allstates = question("Do you want to include all states in the selection?", bool, True)
        if allstates:
            INFOS["allowed"] = set()
        else:
            print("\nPlease enter the states which you want to EXCLUDE from the selection procedure.")
            a = question("Excluded states:", int, ranges=True)
            INFOS["allowed"] = set([-i for i in a])
        print("")

    if INFOS["excite"] == 3:
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

    if INFOS["excite"] == 4:
        INFOS["allowed"] = set()

    return INFOS


# ======================================================================================================================
# ======================================================================================================================
# ======================================================================================================================


def get_initconds(INFOS):
    """"""

    print("Reading initial condition file ...")
    if not INFOS["read_QMout"] and not INFOS["make_list"]:
        INFOS["initf"].seek(0)
        while True:
            line = INFOS["initf"].readline()
            if "Repr" in line:
                INFOS["diag"] = line.split()[1].lower() == "diag"
                INFOS["repr"] = line.split()[1]
            if "Eref" in line:
                INFOS["eref"] = float(line.split()[1])
                break

    initlist = []
    width_bar = 50
    for icond in range(1, INFOS["ninit"] + 1):
        initcond = INITCOND()
        initcond.init_from_file(INFOS["initf"], INFOS["eref"], icond)
        initlist.append(initcond)
        done = width_bar * (icond) // INFOS["ninit"]
        sys.stdout.write("\r  Progress: [" + "=" * done + " " * (width_bar - done) + "] %3i%%" % (done * 100 // width_bar))
    print("\nNumber of initial conditions in file:       %5i" % (INFOS["ninit"]))
    return initlist


# ======================================================================================================================


def make_list(INFOS, initlist):
    print("\nMaking dummy states ...")
    width_bar = 50
    for icond in range(1, INFOS["ninit"] + 1):
        estates = []
        for istate in range(INFOS["nstates"]):
            estates.append(STATE(i=istate + 1))
        initlist[icond - 1].addstates(estates)
        done = width_bar * (icond) // INFOS["ninit"]
        sys.stdout.write("\r  Progress: [" + "=" * done + " " * (width_bar - done) + "] %3i%%" % (done * 100 // width_bar))
    print("\nNumber of initial conditions where states were added:   %5i" % (INFOS["ninit"]))
    return initlist


# ======================================================================================================================


def get_QMout(INFOS, initlist):
    """"""

    print("\nReading QM.out data ...")
    ncond = 0
    initstate = INFOS["initstate"]
    width_bar = 50
    for icond in range(1, INFOS["ninit"] + 1):
        # look for a QM.out file
        qmfilename = INFOS["iconddir"] + "/ICOND_%05i/QM.out" % (icond)
        done = width_bar * (icond) // INFOS["ninit"]
        sys.stdout.write("\r  Progress: [" + "=" * done + " " * (width_bar - done) + "] %3i%%" % (done * 100 // width_bar))
        if not os.path.isfile(qmfilename):
            # print('No QM.out for ICOND_%05i!' % (icond))
            continue
        ncond += 1
        qmout = QMout(filepath=qmfilename)
        H = qmout.h
        DM = qmout.dm
        if INFOS["diag"]:
            P = qmout.ion
            eig, U = np.linalg.eigh(H)
            Ucon = np.conjugate(U)
            DM = np.einsum("kij,in,jm->knm", DM, Ucon, U)
            P = np.einsum("kij,in,jm->knm", P, Ucon, U)
        if INFOS["diabatize"]:
            Smat = qmout.overlap
            thres = 0.5
            # string=''
            N = Smat[0][0].real ** 2
            for i in range(len(Smat)):
                for j in range(len(Smat[0])):
                    Smat[i][j] = Smat[i][j].real ** 2 / N
                    # string+='%5.3f  ' % Smat[i][j]
                # string+='\n'
            # print(string)
            Diabmap = {}
            for i in range(len(Smat)):
                j = Smat[i].index(max(Smat[i]))
                if Smat[i][j] >= thres:
                    Diabmap[i] = j
            # print(icond,Diabmap)
        # generate list of excited states
        estates = []
        for istate in range(len(H)):
            if INFOS["ion"]:
                dip = [math.sqrt(abs(P[initstate][istate])), 0, 0]
            else:
                dip = [DM[i][initstate][istate] for i in range(3)]
            estate = STATE(len(estates) + 1, H[istate][istate], H[initstate][initstate], dip)
            estates.append(estate)
        initlist[icond - 1].addstates(estates)
        if INFOS["diabatize"]:
            initlist[icond - 1].Diabmap = Diabmap
    print("\nNumber of initial conditions with QM.out:   %5i" % (ncond))
    return initlist


# ======================================================================================================================


def excite(INFOS, initlist):
    emin = INFOS["erange"][0]
    emax = INFOS["erange"][1]
    if not INFOS["excite"] == 4:
        if INFOS["excite"] == 3:
            # get the maximum oscillator strength
            maxprob = 0
            probs = np.zeros((len(initlist), len(initlist[0].statelist)), dtype=float)
            for i, icond in enumerate(initlist):
                if icond.statelist == []:
                    continue
                for j, jstate in enumerate(icond.statelist):
                    if emin <= jstate.Eexc <= emax:
                        if -(j + 1) not in INFOS["allowed"]:
                            probs[i, j] = jstate.Prob
                            if jstate.Prob > maxprob:
                                maxprob = jstate.Prob
            np.save("initconds_props.npy", probs)
            
        # set the excitation flags
        print("\nSelecting initial states ...")
        width_bar = 50
        nselected = 0
        for i, icond in enumerate(initlist):
            done = width_bar * (i + 1) // len(initlist)
            sys.stdout.write("\r  Progress: [" + "=" * done + " " * (width_bar - done) + "] %3i%%" % (done * 100 // width_bar))
            if icond.statelist == []:
                continue
            else:
                if INFOS["excite"] == 1:
                    for jstate in icond.statelist:
                        jstate.Excited = False
                elif INFOS["excite"] == 2:
                    if INFOS["diabatize"]:
                        Diabmap = icond.Diabmap
                        # print(i,Diabmap)
                        allowed = []
                        for q in INFOS["allowed"]:
                            if q - 1 in Diabmap:
                                allowed.append(Diabmap[q - 1] + 1)
                    else:
                        allowed = INFOS["allowed"]
                    for j, jstate in enumerate(icond.statelist):
                        if emin <= jstate.Eexc <= emax and j + 1 in allowed:
                            jstate.Excited = True
                            nselected += 1
                        else:
                            jstate.Excited = False
                elif INFOS["excite"] == 3:
                    # and excite
                    for j, jstate in enumerate(icond.statelist):
                        if emin <= jstate.Eexc <= emax:
                            if maxprob > 0 and -(j + 1) not in INFOS["allowed"]:
                                jstate.Excite(maxprob, INFOS["erange"])
                                if jstate.Excited:
                                    nselected += 1
                            else:
                                jstate.Excited = False
                        else:
                            jstate.Excited = False
        print("\nNumber of initial states:                   %5i" % (nselected))

    # statistics
    maxprob = 0.0
    nexc = [0]
    ninrange = [0]
    ntotal = [0]
    for i, icond in enumerate(initlist):
        if icond.statelist == []:
            continue
        else:
            for j, jstate in enumerate(icond.statelist):
                if j + 1 > len(ntotal):
                    ntotal.append(0)
                if j + 1 > len(ninrange):
                    ninrange.append(0)
                if j + 1 > len(nexc):
                    nexc.append(0)
                ntotal[j] += 1
                if emin <= jstate.Eexc <= emax:
                    ninrange[j] += 1
                if jstate.Excited:
                    nexc[j] += 1
    print("\nNumber of initial conditions excited:")
    print("State   Selected   InRange   Total")
    for i in range(len(ntotal)):
        print("  % 3i       % 4i      % 4i    % 4i" % (i + 1, nexc[i], ninrange[i], ntotal[i]))
    return initlist


# ======================================================================================================================
# ======================================================================================================================
# ======================================================================================================================


def writeoutput(initlist, INFOS):
    outfilename = INFOS["initf"].name + ".excited"
    if os.path.isfile(outfilename):
        overw = question("Overwrite %s? " % (outfilename), bool, False)
        print("")
        if overw:
            try:
                outf = open(outfilename, "w")
            except IOError:
                print("Could not open: %s" % (outfilename))
                outf = None
        else:
            outf = None
        if not outf:
            while True:
                outfilename = question("Please enter the output filename: ", str)
                try:
                    outf = open(outfilename, "w")
                except IOError:
                    print("Could not open: %s" % (outfilename))
                    continue
                break
    else:
        outf = open(outfilename, "w")

    print("Writing output to %s ..." % (outfilename))

    outf.write(
        """SHARC Initial conditions file, version %s   <Excited>
Ninit     %i
Natom     %i
Repr      %s
Eref      %18.10f
Eharm     %18.10f
"""
        % (version, INFOS["ninit"], INFOS["natom"], INFOS["repr"], INFOS["eref"], INFOS["eharm"])
    )
    string = ""
    if INFOS["states"]:
        string += "States    "
        for n in INFOS["states"]:
            string += "%i " % (n)
    string += "\n\n\nEquilibrium\n"
    string += "".join(INFOS["equi"])
    string += "\n\n"
    outf.write(string)

    # for atom in INFOS['equi']:
    # string += str(atom) + '\n'

    for i, icond in enumerate(initlist):
        outf.write("Index     %i\n%s" % (i + 1, str(icond)))
    # outf.write(string)
    outf.close()


# ======================================================================================================================
# ======================================================================================================================
# ======================================================================================================================


def main():
    """Main routine"""

    usage = """
python excite.py

This interactive script reads out initconds files and QM.out files from excitation calculations and combines these
information to determine which initial conditions are bright enough for a dynamics simulation.
"""
    displaywelcome()
    open_keystrokes()

    INFOS = {}
    INFOS = get_infos(INFOS)

    print("\n\n" + "{:#^60}".format("Full input") + "\n")
    for item in INFOS:
        if not item == "equi":
            print(item, " " * (25 - len(item)), INFOS[item])
    print("")
    go_on = question("Do you want to continue?", bool, True)
    if not go_on:
        quit(0)
    print("")

    initlist = get_initconds(INFOS)

    if INFOS["read_QMout"]:
        initlist = get_QMout(INFOS, initlist)
    if INFOS["make_list"]:
        initlist = make_list(INFOS, initlist)
    initlist = excite(INFOS, initlist)

    if not INFOS["excite"] == 4:
        writeoutput(initlist, INFOS)
    else:
        print("Nothing done, will not write output.")

    close_keystrokes()


# ======================================================================================================================


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nCtrl+C makes me a sad SHARC ;-(\n")
        quit(0)
