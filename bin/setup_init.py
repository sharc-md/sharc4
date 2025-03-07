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

# Interactive script for the setup of initial condition excitation calculations for SHARC
#
# usage: python setup_init.py

import math
import sys
import re
import os
import stat
import shutil
import datetime
from optparse import OptionParser
import readline
import time
import ast
import random
import factory

from constants import IToMult, U_TO_AMU, HARTREE_TO_EV
from logger import log
from utils import question
from SHARC_INTERFACE import SHARC_INTERFACE

# =========================================================
# some constants
PI = math.pi

version = "4.0"
versionneeded = [0.2, 1.0, 2.0, 2.1, float(version)]
versiondate = datetime.date(2023, 8, 24)
global KEYSTROKES
old_question = question


def question(question, typefunc, default=None, autocomplete=True, ranges=False):
    return old_question(
        question=question, typefunc=typefunc, KEYSTROKES=KEYSTROKES, default=default, autocomplete=autocomplete, ranges=ranges
    )


# ======================================================================================================================
# ======================================================================================================================
# ======================================================================================================================


def try_read(word, index, typefunc, default):
    try:
        return typefunc(word[index])
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
        self.dip = [try_read(f, i, float, 0.0) for i in range(3, 6)]
        self.Excited = try_read(f, 2, bool, False)
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
        s += "% 12.8f % 12.8f %s" % (self.Eexc * HARTREE_TO_EV, self.Fosc, self.excited)
        return s

    def Excite(self, max_Prob, erange):
        try:
            Prob = self.Prob / max_Prob
        except ZeroDivisionError:
            Prob = -1.0
        if not (erange[0] <= self.Eexc <= erange[1]):
            Prob = -1.0
        self.excited = random.random() < Prob


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
        while True:
            line = f.readline()
            if "States" in line:
                break
            atom = ATOM()
            atom.init_from_str(line)
            atomlist.append(atom)
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
        self.Ekin = sum([atom.Ekin for atom in self.atomlist])
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
    log.info("Script for setup of initial conditions started...\n")
    string = "\n"
    string += "  " + "=" * 80 + "\n"
    input = [
        " ",
        "Setup trajectories for SHARC dynamics",
        " ",
        "Authors: Sebastian Mai, Severin Polonius",
        " ",
        "Version: %s" % (version),
        "Date: %s" % (versiondate.strftime("%d.%m.%y")),
        " ",
    ]
    for inp in input:
        string += "||{:^80}||\n".format(inp)
    string += "  " + "=" * 80 + "\n\n"
    string += """
This script automatizes the setup of excited-state calculations for initial conditions
for SHARC dynamics.
  """
    log.info(string)


# ======================================================================================================================


def open_keystrokes():
    global KEYSTROKES
    KEYSTROKES = open("KEYSTROKES.tmp", "w")


def close_keystrokes():
    KEYSTROKES.close()
    shutil.move("KEYSTROKES.tmp", "KEYSTROKES.setup_init")


# ===================================

# ======================================================================================================================
# ======================================================================================================================
# ======================================================================================================================
# ======================================================================================================================
# ======================================================================================================================
# ======================================================================================================================
# ======================================================================================================================


def get_general(INFOS):
    """This routine questions from the user some general information:
    - initconds file
    - number of states
    - number of initial conditions
    - interface to use"""

    log.info(f'{"Initial conditions file":-^60s}' + "\n")
    # open the initconds file
    try:
        initfile = "initconds"
        initf = open(initfile)
        line = initf.readline()
        if check_initcond_version(line):
            log.info('Initial conditions file "initconds" detected. Do you want to use this?')
            if not question('Use file "initconds"?', bool, True):
                initf.close()
                raise IOError
        else:
            initf.close()
            raise IOError
    except IOError:
        log.info("\nIf you do not have an initial conditions file, prepare one with wigner.py!\n")
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
            if check_initcond_version(line):
                break
            else:
                log.info("File does not contain initial conditions!")
                continue
    # read the header
    ninit = int(initf.readline().split()[1])
    natom = int(initf.readline().split()[1])
    INFOS["ninit"] = ninit
    INFOS["natom"] = natom
    initf.seek(0)  # rewind the initf file
    INFOS["initf"] = initf
    log.info('\nFile "%s" contains %i initial conditions.' % (initfile, ninit))
    log.info("Number of atoms is %i\n" % (natom))
    log.info(f"{'Range of initial conditions':-^60}")
    log.info(
        "\nPlease enter the range of initial conditions for which an excited-state calculation should be performed as two integers separated by space."
    )
    while True:
        irange = question("Initial condition range:", int, [1, ninit])
        if len(irange) != 2:
            log.info("Enter two numbers separated by spaces!")
            continue
        if irange[0] > irange[1]:
            log.info("Range empty!")
            continue
        if irange[0] == irange[1] == 0:
            log.info("Only preparing calculation at equilibrium geometry!")
            break
        if irange[1] > ninit:
            log.info("There are only %i initial conditions in file %s!" % (ninit, initfile))
            continue
        if irange[0] <= 0:
            log.info("Only positive indices allowed!")
            continue
        break
    log.info("\nScript will use initial conditions %i to %i (%i in total).\n" % (irange[0], irange[1], irange[1] - irange[0] + 1))
    INFOS["irange"] = irange

    log.info(f"{'Number of states and charge':-^60}")
    log.info(
        "\nPlease enter the number of states as a list of integers\ne.g. 3 0 3 for three singlets, zero doublets and three triplets."
    )
    while True:
        states = question("Number of states:", int)
        if len(states) == 0:
            continue
        if any(i < 0 for i in states):
            log.info("Number of states must be positive!")
            continue
        break
    log.info("")

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

    nstates = 0
    for mult, i in enumerate(states):
        nstates += (mult + 1) * i
    log.info("Number of states: " + str(states))
    log.info("Total number of states: %i\n" % (nstates))
    INFOS["states"] = states
    INFOS["nstates"] = nstates
    INFOS["charge"] = charges

    log.info("")
    return INFOS


def get_interface() -> SHARC_INTERFACE:
    "asks for interface and instantiates it"
    Interfaces = factory.get_available_interfaces()
    log.info("")
    log.info("{:-^60}".format("Choose the quantum chemistry interface"))
    log.info("\nPlease specify the quantum chemistry interface (enter any of the following numbers):")
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
    interface.QMin.molecule["states"] = INFOS["states"]
    int_features = interface.get_features(KEYSTROKES=KEYSTROKES)
    log.info("\nThe following features are available from this interface:")
    log.info(int_features)

    INFOS["needed_requests"] = set(["h", "dm"])
    states = INFOS["states"]

    # Setup SOCs
    log.info("\n" + f"{'Spin-orbit couplings (SOCs)':-^60}" + "\n")
    if len(states) > 1:
        if "soc" in int_features:
            log.info("Do you want to compute spin-orbit couplings?\n")
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

    # Setup Dyson spectra
    if "ion" in int_features:
        n = [0, 0]
        for i, j in enumerate(states):
            n[i % 2] += j
        if n[0] >= 1 and n[1] >= 1:
            log.info("\n" + f"{'Ionization probability by Dyson norms':-^60}" + "\n")
            log.info("Do you want to compute Dyson norms between neutral and ionic states?")
            INFOS["ion"] = question("Dyson norms?", bool, False)
            if INFOS["ion"]:
                INFOS["needed_requests"].add("ion")

    # Setup initconds with reference overlap
    if "overlap" in int_features:
        log.info("\n" + f"{'Overlaps to reference states':-^60}" + "\n")
        log.info(
            "Do you want to compute the overlaps between the states at the equilibrium geometry and the states at the initial condition geometries?"
        )
        INFOS["refov"] = question("Reference overlaps?", bool, False)
        if INFOS["refov"]:
            INFOS["needed_requests"].add("overlap")

    # Setup theodore
    if "theodore" in int_features:
        log.info("\n" + f"{'TheoDORE wave function analysis':-^60}" + "\n")
        log.info("Do you want to run TheoDORE to obtain one-electron descriptors for the electronic wave functions?")
        INFOS["theodore"] = question("TheoDORE?", bool, False)
        if INFOS["theodore"]:
            INFOS["needed_requests"].add("theodore")

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


def get_runscript_info(INFOS):
    """"""

    string = "\n  " + "=" * 80 + "\n"
    string += "||" + f"{'Run mode setup':^80}" + "||\n"
    string += "  " + "=" * 80 + "\n\n"
    log.info(string)

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
    else:
        INFOS["here"] = False
        log.info("\nWhere do you want to perform the calculations? Note that this script cannot check whether the path is valid.")
        INFOS["copydir"] = question("Run directory?", str)
    log.info("")

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


def writeQMin(INFOS, iconddir):
    icond = int(iconddir[-6:-1])
    try:
        qmin = open("%s/QM.in" % (iconddir), "w")
    except IOError:
        log.info("IOError during writeQMin, icond=%s" % (iconddir))
        quit(1)
    string = "%i\nInitial condition %s\n" % (INFOS["natom"], iconddir)

    if icond > 0:
        searchstring = r"Index\s+%i" % (icond)
    else:
        searchstring = "Equilibrium"
    rewinded = False
    while True:
        try:
            line = INFOS["initf"].readline()
        except EOFError:
            if not rewinded:
                rewinded = True
                INFOS["initf"].seek(0)
            else:
                log.info("Could not find Initial condition %i!" % (icond))
                quit(1)
        # if searchstring in line:
        if re.search(searchstring, line):
            break
    if icond > 0:
        line = INFOS["initf"].readline()  # skip one line
    for iatom in range(INFOS["natom"]):
        line = INFOS["initf"].readline()
        s = line.split()
        string += "%s %s %s %s\n" % (s[0], s[2], s[3], s[4])

    string += "unit bohr\nstates "
    for i in INFOS["states"]:
        string += "%i " % (i)
    string += "\n"
    string += "charge "
    for i in INFOS["charge"]:
        string += "%i " % (i)
    string += "\n"

    if "refov" in INFOS and INFOS["refov"]:
        if icond == 0:
            string += "step 0\nsavedir ./SAVE/\n"
        else:
            string += "overlap\nsavedir ./SAVE/\n"
    else:
        string += "step 0\n"

    if INFOS["soc"]:
        string += "\nSOC\n"
    else:
        string += "\nH\n"
    string += "DM\n"
    if "ion" in INFOS and INFOS["ion"]:
        string += "ion\n"
    if "theodore" in INFOS and INFOS["theodore"]:
        string += "theodore\n"

    qmin.write(string)
    qmin.close()
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
        projname = "init_%5s" % (iconddir[-6:-1])

    # ================================
    intstring = ""
    if "adfrc" in INFOS:
        intstring = ". %s\nexport PYTHONPATH=$ADFHOME/scripting:$PYTHONPATH" % (INFOS["adfrc"])
    elif "amsbashrc" in INFOS:
        intstring = ". %s\nexport PYTHONPATH=$AMSHOME/scripting:$PYTHONPATH" % (INFOS["amsbashrc"])

    # ================================
    if ("refov" in INFOS and INFOS["refov"]) and iconddir != "ICOND_00000/":
        refstring = """
if [ -d ../ICOND_00000/SAVE ];
then
  if [ -d ./SAVE ];
  then
    rm -r ./SAVE
  fi
  cp -r ../ICOND_00000/SAVE ./
else
  echo "Should do a reference overlap calculation, but the reference data in ../ICOND_00000/ seems not OK."
  exit 1
fi
"""
    else:
        refstring = ""

    # generate run scripts here
    # ================================ for here mode
    if INFOS["here"]:
        string = """#!/bin/bash

# $-N %s

%s

PRIMARY_DIR=%s/%s/

cd $PRIMARY_DIR
%s

$SHARC/%s.py QM.in > QM.log 2> QM.err
""" % (
            projname,
            intstring,
            INFOS["cwd"],
            iconddir,
            refstring,
            interface.__class__.__name__,
        )
    #
    # ================================ for remote mode
    else:
        string = """#!/bin/bash

# $-N %s

%s

PRIMARY_DIR=%s/%s/
COPY_DIR=%s/%s/

cd $PRIMARY_DIR
%s

mkdir -p $COPY_DIR
cp -r $PRIMARY_DIR/* $COPY_DIR
cd $COPY_DIR

$SHARC/%s QM.in >> QM.log 2>> QM.err

cp -r $COPY_DIR/QM.* $COPY_DIR/SAVE/ $PRIMARY_DIR
rm -r $COPY_DIR
""" % (
            projname,
            intstring,
            INFOS["cwd"],
            iconddir,
            INFOS["copydir"],
            iconddir,
            refstring,
            interface.__class__.__name__,
        )

    # ================================
    runscript.write(string)
    runscript.close()
    filename = "%s/run.sh" % (iconddir)
    os.chmod(filename, os.stat(filename).st_mode | stat.S_IXUSR)
    return


# ======================================================================================================================
# ======================================================================================================================
# ======================================================================================================================


def setup_equilibrium(INFOS, interface: SHARC_INTERFACE):
    # iconddir='ICOND_%05i/' % (0)
    # exists=os.path.isfile(iconddir+'/QM.out')
    exists = False
    if not exists:
        iconddir = "ICOND_%05i/" % (0)
        io = make_directory(iconddir)
        if io != 0:
            log.info("Skipping initial condition %s!" % (iconddir))
            return

        writeQMin(INFOS, iconddir)
        interface.prepare(INFOS, iconddir)
        writeRunscript(INFOS, iconddir, interface)
    return exists


# ======================================================================================================================
# ======================================================================================================================
# ======================================================================================================================


def setup_all(INFOS, interface: SHARC_INTERFACE):
    """This routine sets up the directories for the initial calculations."""

    string = "\n  " + "=" * 80 + "\n"
    string += "||" + f"{'Setting up directories...':^80}" + "||\n"
    string += "  " + "=" * 80 + "\n\n"
    log.info(string)

    all_run = open("all_run_init.sh", "w")
    string = "#/bin/bash\n\nCWD=%s\n\n" % (INFOS["cwd"])
    all_run.write(string)
    if INFOS["qsub"]:
        all_qsub = open("all_qsub_init.sh", "w")
        string = "#/bin/bash\n\nCWD=%s\n\n" % (INFOS["cwd"])
        all_qsub.write(string)

    width = 50
    ninit = INFOS["irange"][1] - INFOS["irange"][0] + 1
    idone = 0

    EqExists = setup_equilibrium(INFOS, interface)
    if not EqExists:
        iconddir = "ICOND_%05i/" % (0)
        string = "cd $CWD/%s/\nbash run.sh\ncd $CWD\necho %s >> DONE\n" % (iconddir, iconddir)
        all_run.write(string)
        if INFOS["qsub"]:
            string = "cd $CWD/%s/\n%s run.sh\ncd $CWD\n" % (iconddir, INFOS["qsubcommand"])
            all_qsub.write(string)

    if INFOS["irange"] != [0, 0]:
        for icond in range(INFOS["irange"][0], INFOS["irange"][1] + 1):
            iconddir = "ICOND_%05i/" % (icond)
            idone += 1
            done = int((idone / ninit) * width)

            io = make_directory(iconddir)
            if io != 0:
                log.info("Skipping initial condition %s!" % (iconddir))
                continue

            writeQMin(INFOS, iconddir)
            interface.prepare(INFOS, iconddir)
            writeRunscript(INFOS, iconddir, interface)

            string = "cd $CWD/%s/\nbash run.sh\ncd $CWD\necho %s >> DONE\n" % (iconddir, iconddir)
            all_run.write(string)
            if INFOS["qsub"]:
                string = "cd $CWD/%s/\n%s run.sh\ncd $CWD\n" % (iconddir, INFOS["qsubcommand"])
                all_qsub.write(string)
            sys.stdout.write("\rProgress: [" + "=" * done + " " * (width - done) + "] %3i%%" % (done * 100 // width))
            sys.stdout.flush()

    all_run.close()
    filename = "all_run_init.sh"
    os.chmod(filename, os.stat(filename).st_mode | stat.S_IXUSR)
    if INFOS["qsub"]:
        all_qsub.close()
        filename = "all_qsub_init.sh"
        os.chmod(filename, os.stat(filename).st_mode | stat.S_IXUSR)

    log.info("\n")


# ======================================================================================================================
# ======================================================================================================================
# ======================================================================================================================


def main():
    """Main routine"""

    usage = """
python setup_init.py

This interactive program prepares the initial excited-state calculations for SHARC.
As input it takes the initconds file, number of states and range of initconds.

Afterwards, it asks for the interface used and goes through the preparation depending on the interface.
"""

    description = ""
    parser = OptionParser(usage=usage, description=description)

    displaywelcome()
    open_keystrokes()
    INFOS = {}
    INFOS["cwd"] = os.getcwd()

    INFOS = get_general(INFOS)
    chosen_interface: SHARC_INTERFACE = get_interface()()
    INFOS = get_requests(INFOS, chosen_interface)
    INFOS = chosen_interface.get_infos(INFOS, KEYSTROKES)
    INFOS = get_runscript_info(INFOS)

    log.info("\n" + f"{'Full input':#^60}" + "\n")
    for item in INFOS:
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
