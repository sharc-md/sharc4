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

import datetime
import os
import sys
import json
import math
import numpy as np
import subprocess as sp
from scipy.linalg import fractional_matrix_power
from itertools import starmap, chain
from optparse import OptionParser
from qmout import QMout
import random

from constants import IToMult, U_TO_AMU, HARTREE_TO_EV
from utils import itnmstates, readfile, question as question_def
from printing import printheader
from logger import log 
import shutil

def open_keystrokes():
    global KEYSTROKES
    KEYSTROKES = open("KEYSTROKES.tmp", "w")


def close_keystrokes():
    KEYSTROKES.close()
    shutil.move("KEYSTROKES.tmp", "KEYSTROKES.excite")

# ===================================


def question(question, typefunc, default=None, autocomplete=True, ranges=False):
    return question_def(question, typefunc, KEYSTROKES, default, autocomplete, ranges)


np.set_printoptions(linewidth=800, formatter={"float": lambda x: f"{x.real: 7.5e}"}, threshold=sys.maxsize)


def json_load_byteified(file_handle):
    return _byteify(json.load(file_handle, object_hook=_byteify), ignore_dicts=True)


def json_loads_byteified(json_text):
    return _byteify(json.loads(json_text, object_hook=_byteify), ignore_dicts=True)


def _byteify(data, ignore_dicts=False):
    # if this is a list of values, return list of byteified values
    if isinstance(data, list):
        return [_byteify(item, ignore_dicts=True) for item in data]
    # if this is a dictionary, return dictionary of byteified keys and values
    # but only if we haven't already byteified it
    if isinstance(data, dict) and not ignore_dicts:
        return {_byteify(key, ignore_dicts=True): _byteify(value, ignore_dicts=True) for key, value in data.items()}
    # if it's anything else, return it in its original form
    return data


# ======================================================================= #


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



# ======================================================================= #


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


# ======================================================================= #


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


# ======================================================================= #


version = "4.0"
versionneeded = [0.2, 1.0, 2.0, 2.1, float(version)]
versiondate = datetime.date(2024, 4, 1)


# ======================================================================= #

pthresh = 1.0e-5**2

# ======================================================================= #


def displaywelcome():
    lines = [
        f"Compute LVC parameters",
        "",
        f"Authors: Severin Polonius, Sebastian Mai, Simon Kropf",
        "",
        f"Version: {version}",
        "Date: {:%d.%m.%Y}".format(versiondate),
    ]

    print("Script for setup of displacements started...\n")
    printheader(lines)
    string = "This script automatizes the setup of excited-state calculations for displacements\nfor SHARC dynamics."
    print(string)


# ======================================================================= #


def run_data_extractor(initstate, INFOS):
    """
    Extract output.dat in every TRAJ folder for every setupstate
    """
    forbidden = ['crashed', 'running', 'dead', 'dont_analyze']
    dirname = INFOS["setupstates_names"][initstate]  # get_iconddir(initstate, INFOS)  # State directory containing trajectories 
    req_files_traj = ["geom", "input", "laser", "run.sh", "veloc"]
    req_folders_traj = ["QM", "restart"]
    # run the data extractor, if necessary
    # first check whether $SHARC contains the extractor
    print('Running data_extractor...')
    sharcpath = os.getenv('SHARC')
    if sharcpath is None:
        print('Please set $SHARC to the directory containing the SHARC executables!')
        sys.exit(1)
    else:
        if not os.path.isfile(sharcpath + '/data_extractor.x'):
            print('$SHARC does not contain data_extractor.x!')
            sys.exit(1)
        else:
            if not all(traj.startswith("TRAJ_") for traj in os.listdir(dirname)):  # check, if all trajectories start with TRAJ_ 
                log.info("Not all trajectories for state %s start with 'TRAJ_'" % dirname)
                # sys.exit(1)
            if not ["TRAJ_%05i" % int(el) for el in INFOS["icond_sel"]] == os.listdir(dirname):
                log.info("Not all trajectories for the selected initial conditions exist!")
                # sys.exit(1)
            for itraj in os.listdir(dirname):  # cycle through all TRAJ_directories in initstate folder
                traj_path = os.path.join(dirname, itraj)
                update = True
                if INFOS["diag"]:
                    if os.path.isfile(os.path.join(traj_path, "output_data/coeff_diag.out")):
                        update = False
                        break
                    extract_arg = "-cd"
                else: 
                    if os.path.isfile(os.path.join(traj_path, "output_data/coeff_MCH.out")):
                        update = False 
                        break
                    extract_arg = "-cm"
                if any([os.path.isfile(os.path.join(traj_path, forbid_file)) for forbid_file in forbidden]):
                    update = False
                # if not all(os.path.isfile(os.path.join(traj_path, file) for file in req_files_traj)):
                #     log.info("Trajectory setup files are missing for %s" % traj_path)
                #     # sys.exit(1)
                # if not all(os.path.isdir(os.path.join(traj_path, folder) for folder in req_folders_traj)):
                #     log.info("Trajectory setup folders are missing for %s" % traj_path)
                #     # sys.exit(1)
                # check whether output_data/expec.out is newer than output.dat
                # TODO New check for corrupted trajectory folders
                # if not os.path.isfile(path + '/output_data/expec.out'):
                #     update = True
                # if not update:
                #     time_dat = os.path.getmtime(path + '/output.dat')
                #     time_expec = os.path.getmtime(path + '/output_data/expec.out')
                #     if time_dat > time_expec or INFOS['run_extractor_full']:
                #         update = True
                if update:
                    os.chdir(traj_path)
                    if INFOS["netcdf"]:
                        if os.path.isfile("output.dat.nc"):
                            io = sp.call(sharcpath + '/data_extractor_NetCDF.x %s output.dat > /dev/null 2> /dev/null' % extract_arg, shell=True)
                            if io != 0:
                                print('WARNING: extractor call failed for %s! Exit code %i' % (traj_path, io))
                        else:
                            log.info("No file 'output.dat.nc' in %s. Quitting." % traj_path)
                            sys.exit(1)
                    elif INFOS["ascii"]:
                        if os.path.isfile("output.dat"):
                            io = sp.call(sharcpath + '/data_extractor.x %s output.dat > /dev/null 2> /dev/null' % extract_arg, shell=True)
                            if io != 0:
                                print('WARNING: extractor call failed for %s! Exit code %i' % (traj_path, io))
                        else:
                            log.info("No file 'output.dat.nc' in %s. Quitting." % traj_path)
                            sys.exit(1)
                    else:
                        pass
                    os.chdir(INFOS["cwd"])
    print('Extraction finished!\n')


# ======================================================================= #


def gfsh_probs(istate, icond, INFOS):
    print(INFOS["max_prob"])
    if INFOS["diag"]:
        coeff_file = "coeff_diag.out"
    else:
        coeff_file = "coeff_MCH.out"
    coeff_data =  np.genfromtxt(get_iconddir(istate, INFOS) + "/TRAJ_%05i/output_data/%s" %(icond, coeff_file), comments="#")
    # coeffs = np.zeros((coeff_data.shape[0], int((coeff_data.shape[1]-2)/2)), dtype=complex)
    # TODO Decide, whether coeff_data.shape makes more sense or tsteps, setupstates
    coeffs = np.zeros((INFOS["nsteps"]+1, INFOS["nstates"]), dtype=complex)  # nstates
    time_steps, states = coeffs.shape
    for n_columns in range(0, len(coeffs[1])):
        coeffs[:, int(n_columns)] = coeff_data[:, int(2*(n_columns+1))] + 1.j*coeff_data[:, int(2*(n_columns+1)+1)]
    exc_prob = np.zeros((time_steps, int(states)))
    exc_prob_tdiff = np.zeros((time_steps-1, int(states)))
    max_prob = np.zeros(time_steps-1)
    rho = np.zeros_like(coeffs, dtype=float)
    rho[:, 1:] = np.abs(coeffs[:, 1:])**2/INFOS["max_prob"]  # excited state coefficients
    # rho[:, 0] =  np.abs(coeffs[:, 0])**2  # ground state coefficients
    # norm = np.sum(rho, axis=1)
    # rho=rho/norm[:, np.newaxis]
    rho[:, 0] = 1.-np.sum(rho[:, 1:], axis=1)
    # print(f"pmax_gfsh_probs: {max_prob}")
    # nominator of 8.151
    gs_fac = (1 - (rho[1:, 0] / rho[:-1, 0]))  # [time, ground state]
    # Loop calculates the maximum in the denominator of 8.151
    for tstep in range(time_steps-1):
        for exc_state in range(states):
            # if exc_state==0:
            #    continue
            # else:
            exc_prob_tdiff[tstep, exc_state] = rho[tstep+1, exc_state] - rho[tstep, exc_state]
            if exc_prob_tdiff[tstep, exc_state] < 0.:  #  if difference is negative -> negative added to sum
                max_prob[tstep] -= exc_prob_tdiff[tstep, exc_state]
    # Loop calculates result of 8.151
    for tstep in range(time_steps-1):
        if gs_fac[tstep]>0:
            for exc_state in range(states):
                if exc_state==0:
                    continue
                if exc_prob_tdiff[tstep, exc_state] < 0.:
                    continue
                else:
                    exc_prob[tstep, exc_state] = np.max([0, gs_fac[tstep]*exc_prob_tdiff[tstep, exc_state]/max_prob[tstep]])
    data_to_save = np.column_stack((range(time_steps), exc_prob))
    np.savetxt("exc_state_%05i_traj_%05i" % (istate, icond), data_to_save, fmt="%.2e", delimiter="\t")
    return exc_prob


# ======================================================================= #


def compute_max_prob(INFOS):
    """
    Extract output.dat in every TRAJ folder for every setupstate
    """
    # check if trajectory was run? Otherwise skip
    pmax = 0.  # initialize maximum probability over all initial states and initial conditions (TRAJS) to ever leave the initial state
    # run the data extractor, if necessary
    # first check whether $SHARC contains the extractor
    if INFOS["diag"]:
        coeff_file = "coeff_diag.out"
    else:
        coeff_file = "coeff_MCH.out"
    print('Running data_extractor...')
    for istate in INFOS["setupstates"]:  
        for icond in INFOS["icond_sel"]:
            print(istate, icond)
            p_traj = 0.  # Initialize the accumulated probability to leave the initial state for this trajectory
            coeff_data =  np.genfromtxt(get_iconddir(istate, INFOS) + "/TRAJ_%05i/output_data/%s" %(icond, coeff_file), comments="#")
            # TODO: probably would suffice to only take the initial state - defining a big array is not needed
            # coeffs = np.zeros((coeff_data.shape[0], int((coeff_data.shape[1]-2)/2)), dtype=complex)
            # for n_columns in range(0, len(coeffs[1])):
            # coeffs[:, int(n_columns)] = coeff_data[:, int(2*(n_columns+1))] + 1.j*coeff_data[:, int(2*(n_columns+1)+1)]
            coeff_init = coeff_data[:, int(2*(istate-1+1))] + 1.j*coeff_data[:, int(2*(istate-1+1)+1)]  # skip first two columns containing time and c**2, but istates start at 1
            p_init = np.abs(coeff_init)**2
            for tstep in range(len(coeff_init)-1):
                p_init_tdiff = p_init[tstep]-p_init[tstep+1]
                if p_init_tdiff > 0.:
                    p_traj += p_init_tdiff  # if the population of the initial state gets less, add this to the p_traj
            if p_traj > pmax:
                pmax = p_traj
    return pmax 


# ======================================================================= #


def get_initconds(INFOS):
    """
    """
    # Skip first commented block, as INFOS["read_QMout"] is true

    # print("Reading initial condition file ...")
    # if not INFOS["read_QMout"] and not INFOS["make_list"]:
    #     INFOS["initf"].seek(0)
    #     while True:
    #         line = INFOS["initf"].readline()
    #         if "Repr" in line:
    #             INFOS["diag"] = line.split()[1].lower() == "diag"
    #             INFOS["repr"] = line.split()[1]
    #         if "Eref" in line:
    #             INFOS["eref"] = float(line.split()[1])
    #             break

    initlist = []
    width_bar = 50
    for icond in range(1, INFOS["ninit"] + 1):
        initcond = INITCOND()
        initf = open(INFOS["initf"]) 
        initcond.init_from_file(initf, INFOS["eref"], icond)
        initf.close()
        initlist.append(initcond)
        done = width_bar * (icond) // INFOS["ninit"]
        sys.stdout.write("\r  Progress: [" + "=" * done + " " * (width_bar - done) + "] %3i%%" % (done * 100 // width_bar))
    print("\nNumber of initial conditions in file:       %5i" % (INFOS["ninit"]))
    return initlist


# ======================================================================= #


def get_QMout(INFOS, initstate, initlist):
    """"""

    print("\nReading QM.out data ...")
    ncond = 0
    width_bar = 50
    print("iconds %i" % INFOS["ninit"])
    for icond in range(1, INFOS["ninit"] + 1):
        # look for a QM.out file
        qmfilename = INFOS["path"]+"/ICOND_%05i/QM.out" % (icond)
        done = width_bar * (icond) // INFOS["ninit"]
        sys.stdout.write("\r  Progress: [" + "=" * done + " " * (width_bar - done) + "] %3i%%" % (done * 100 // width_bar))
        if not os.path.isfile(qmfilename):
            print('No QM.out for ICOND_%05i!' % (icond))
            continue
        ncond += 1
        qmout = QMout(filepath=qmfilename)
        H = qmout.h
        DM = qmout.dm
        estates = []
        for istate in range(len(H)):
            dip = [DM[i][initstate][istate] for i in range(3)]
            estate = STATE(len(estates) + 1, H[istate][istate], H[initstate][initstate], dip)
            estates.append(estate)
        print("TESTgetqmout")
        initlist[icond - 1].addstates(estates)
    print("\nNumber of initial conditions with QM.out:   %5i" % (ncond))
    return initlist


# ======================================================================= #


def get_iconddir(istate, INFOS):
    if INFOS["diag"]:
        dirname = "State_%i" % (istate)
    else:
        mult, state, ms = INFOS["statemap"][istate]
        dirname = IToMult[mult] + "_%i" % (state - (mult == 1 or mult == 2))
    return dirname


# ======================================================================= #
# ======================================================================= #
# ======================================================================= #


def writeoutput(initlist, istate, INFOS):
    dirname = get_iconddir(istate+1, INFOS)
    outfilename = INFOS["initf"] + "_" + dirname + ".excited"
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

def excite(INFOS, initlist, exc_list, setupstate):
    print("\nSelecting initial states ...")
    width_bar = 50
    nselected = 0
    for i, icond in enumerate(initlist):
        done = width_bar * (i + 1) // len(initlist)
        sys.stdout.write("\r  Progress: [" + "=" * done + " " * (width_bar - done) + "] %3i%%" % (done * 100 // width_bar))
        if icond.statelist == []:
            continue
        elif i+1 not in INFOS["icond_sel"]:  # Did not select icond for laser excitation
            log.info("Initial condition %i not selected!" % i)
        else:
            if exc_list[setupstate, i, 1] != 0:
                log.info(icond.statelist)
                for j, jstate in enumerate(icond.statelist):
                    if exc_list[setupstate, i, 1]==j:
                        jstate.Excited = True
                    else:
                        jstate.Excited = False
    #for i, icond in enumerate(initlist)
    #        # get the maximum oscillator strength
    #        maxprob = 0
    #        probs = np.zeros((len(initlist), len(initlist[0].statelist)), dtype=float)
    #        for i, icond in enumerate(initlist):
    #            if icond.statelist == []:
    #                continue
    #            for j, jstate in enumerate(icond.statelist):
    #                if emin <= jstate.Eexc <= emax:
    #                    if -(j + 1) not in INFOS["allowed"]:
    #                        probs[i, j] = jstate.Prob
    #                        if jstate.Prob > maxprob:
    #                            maxprob = jstate.Prob
    #        np.save("initconds_props.npy", probs)

    #    # set the excitation flags
    #    print("\nSelecting initial states ...")
    #    width_bar = 50
    #    nselected = 0
    #    for i, icond in enumerate(initlist):
    #        done = width_bar * (i + 1) // len(initlist)
    #        sys.stdout.write("\r  Progress: [" + "=" * done + " " * (width_bar - done) + "] %3i%%" % (done * 100 // width_bar))
    #        if icond.statelist == []:
    #            continue
    #        else:
    #            if INFOS["excite"] == 1:
    #                for jstate in icond.statelist:
    #                    jstate.Excited = False
    #            elif INFOS["excite"] == 2:
    #                if INFOS["diabatize"]:
    #                    Diabmap = icond.Diabmap
    #                    # print(i,Diabmap)
    #                    allowed = []
    #                    for q in INFOS["allowed"]:
    #                        if q - 1 in Diabmap:
    #                            allowed.append(Diabmap[q - 1] + 1)

    #                else:
    #                    allowed = INFOS["allowed"]
    #                for j, jstate in enumerate(icond.statelist):
    #                    if emin <= jstate.Eexc <= emax and j + 1 in allowed:
    #                        jstate.Excited = True
    #                        nselected += 1
    #                    else:
    #                        jstate.Excited = False
    #            elif INFOS["excite"] == 3:
    #                # and excite
    #                for j, jstate in enumerate(icond.statelist):
    #                    if emin <= jstate.Eexc <= emax:
    #                        if maxprob > 0 and -(j + 1) not in INFOS["allowed"]:
    #                            jstate.Excite(maxprob, INFOS["erange"])
    #                            if jstate.Excited:
    #                                nselected += 1
    #                        else:
    #                            jstate.Excited = False
    #                    else:
    #                        jstate.Excited = False
    #    print("\nNumber of initial states:                   %5i" % (nselected))

    # statistics
    nexc = [0]
    ntotal = [0]
    for i, icond in enumerate(initlist):
        if icond.statelist == []:
            continue
        else:
            for j, jstate in enumerate(icond.statelist):
                if j + 1 > len(ntotal):
                    ntotal.append(0)
                if j + 1 > len(nexc):
                    nexc.append(0)
                ntotal[j] += 1
                if jstate.Excited:
                    nexc[j] += 1
    print("\nNumber of initial conditions excited:")
    print("State   Selected     Total")
    for i in range(len(ntotal)):
        print("  % 3i      % 4i      % 4i" % (i + 1, nexc[i], ntotal[i]))
    return initlist



# ======================================================================================================================
# ======================================================================================================================
# ======================================================================================================================

def main():
    """Main routine"""
    script_name = sys.argv[0].split("/")[-1]

    usage = """python %s""" % (script_name)

    parser = OptionParser(usage=usage, description="")
    displaywelcome()
    open_keystrokes()
    is_other_dir = len(sys.argv) == 2 and os.path.isdir(sys.argv[1])
    # load INFOS object from file
    setup_laser_excitation_info_filename = os.path.join(sys.argv[1], "setup_laser_excitiation.json") if is_other_dir else "setup_laser_excitation.json"

    try:
        with open(setup_laser_excitation_info_filename, "r") as setup_laser_excitation_info:
            INFOS = json_load_byteified(setup_laser_excitation_info)
            setup_laser_excitation_info.close()
    except IOError:
        print("IOError during opening readable %s - file. Quitting." % (setup_laser_excitation_info_filename))
        quit(1)
    INFOS["rng_seed"] = random_seed()
    # INFOS = get_initconds(INFOS) - all necessary information in json file?
    # TODO: check that the istates are correctly called - they start with 1, not 0
    initlist = []
    for i, istate in enumerate(INFOS["setupstates"]):
        initlist.append(get_initconds(INFOS))
        initlist[i] = get_QMout(INFOS, i, initlist[i])  # adding excited state information
        log.info([print("TESTOI %s" % el.statelist for el in initlist[i])])
        run_data_extractor(i, INFOS)

    INFOS["max_prob"] = 1  #  compute_max_prob(INFOS)  # Compute maximum probability to leave initial state
    print(INFOS["max_prob"])
    exc_probs = np.zeros((len(INFOS["setupstates"]), len(INFOS["icond_sel"]), INFOS["nsteps"]+1, INFOS["nstates"]))
    exc_probs_cumsum = np.zeros_like(exc_probs) 
    exc_list = np.zeros((len(INFOS["setupstates"]), len(INFOS["icond_sel"]), 2))
    log.info(exc_list.shape)
    for istate in INFOS["setupstates"]:
        for icond in INFOS["icond_sel"]:
            jump_to_next = False
            exc_probs[istate-1, icond-1, :, :] = gfsh_probs(istate, icond, INFOS)
            exc_probs_cumsum[istate-1, icond-1, :, :] = np.cumsum(exc_probs[istate-1, icond-1, :, :], axis=1) 
            for tstep in range(0, INFOS["nsteps"]+1):
                if jump_to_next:
                    continue
                for exc_state in range(1, INFOS["nstates"]+1):
                    if jump_to_next:
                        continue
                    if random.random() <= exc_probs_cumsum[istate-1, icond-1, tstep, exc_state-1]:
                        print(exc_state)
                        exc_list[istate-1, icond-1, :] = tstep, exc_state 
                        print(exc_list[istate-1, icond-1, :], istate, icond, tstep)
                        jump_to_next = True
    print("FINIHSED")                    
    for i, istate in enumerate(INFOS["setupstates"]):
        #print(i, istate)
        initlist[i] = excite(INFOS, initlist[i], exc_list, i)
        writeoutput(initlist[i], i, INFOS) 
    # set manually for old calcs
    # INFOS['ignore_problematic_states'] = True
    close_keystrokes()


# ======================================================================= #


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nCtrl+C occured. Exiting.\n")
        sys.exit()
