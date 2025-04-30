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
    shutil.move("KEYSTROKES.tmp", "KEYSTROKES.excite_laser_excitation")

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
        self.ExcTime = ""
        self.IState = "" 
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
        self.ExcTime = try_read(f, 12, str, "")
        self.IState = try_read(f, 13, str, "")
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
        try:
            s += "% 12.8f % 12.8f %s % 03i % 12.8f" % (self.Eexc * HARTREE_TO_EV, self.Fosc, self.Excited, self.IState, self.ExcTime)
        except:
            s += "% 12.8f % 12.8f %s % s % s" % (self.Eexc * HARTREE_TO_EV, self.Fosc, self.Excited, self.IState, self.ExcTime)
        return s

    # def Excite(self, max_Prob, erange):
    #     if erange[0] <= self.Eexc <= erange[1]:
    #         self.Excited = random.random() < (self.Prob / max_Prob)
    #     else:
    #         self.Excited = False


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

    def get_coeff(self, coeff, coeff_save):
        self.coeff = coeff
        self.coeff_save = coeff_save 

    def __str__(self):
        s = "Atoms\n" + "".join(self.atomlist)
        s += "States\n"
        for state in self.statelist:
            s += str(state) + "\n"
        if np.any([self.statelist[state].Excited for state in range(0, len(self.statelist))]) and self.coeff_save:
            s += "Coefficients\n"
            for ist in range(0, self.nstate):
                if not self.statelist[ist].Excited:  # TODO: Double-excitations of a trajectory (in principle) not covered.
                    continue
                s += f"Coef {ist+1:03d}\n" 
                for jst in range(0, self.nstate):
                    s += "%03i " % (jst+1)
                    for k in range(0, 2):  # complex number 
                        s += "% 18.10f " % self.coeff[jst, k]
                    s += "\n"
        else:
            pass

        s += "Ekin      % 16.12f a.u.\n" % (self.Ekin)
        s += "Epot_harm % 16.12f a.u.\n" % (self.Epot_harm)
        s += "Epot      % 16.12f a.u.\n" % (self.Epot)
        s += "Etot_harm % 16.12f a.u.\n" % (self.Epot_harm + self.Ekin)
        s += "Etot      % 16.12f a.u.\n" % (self.Epot + self.Ekin)
        s += "\n\n"
        return s


# ======================================================================= #


version = "1.0"
versionneeded = [0.2, 1.0, 2.0, 2.1, float(version)]
versiondate = datetime.date(2025, 5, 1)


# ======================================================================= #

pthresh = 1.0e-5**2

# ======================================================================= #


def displaywelcome():
    lines = [
        f"Compute excitation probabilities and excitation times",
        "",
        f"Authors: Lorenz Grünewald, Sebastian Mai",
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
    width_bar = 80
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
            if not all(traj.startswith("TRAJ_") for traj in filter(os.path.isdir, os.listdir(dirname))):  # check, if all trajectories start with TRAJ_ 
                log.info(os.listdir(dirname))
                log.info("Not all trajectories for state %s start with 'TRAJ_'" % dirname)
                # sys.exit(1)
            if not all(os.path.exists(dirname+"/"+traj) for traj in ["TRAJ_%05i" % int(el) for el in INFOS["icond_sel"]]):
                log.info("Not all trajectories for the selected initial conditions exist!")
                # sys.exit(1)
            for itraj, traj in enumerate(["TRAJ_%05i" % int(el) for el in INFOS["icond_sel"]]):  # cycle through all TRAJ_directories in initstate folder
                done = width_bar * (itraj+1) // len(INFOS["icond_sel"])
                sys.stdout.write("\r  Progress: [" + "=" * done + " " * (width_bar - done) + "] %3i%%" % (done * 100 // width_bar))
                traj_path = os.path.join(dirname, traj)
                update = True
                if INFOS["diag"]:
                    if os.path.isfile(os.path.join(traj_path, "output_data/coeff_diag.out")):
                        # log.info("Already existing 'coeff_diag.out' for %s in %s" % (traj, INFOS["setupstates_names"][initstate]))
                        update = False
                        continue
                    else:
                        extract_arg = "-cd"
                else: 
                    if os.path.isfile(os.path.join(traj_path, "output_data/coeff_MCH.out")):
                        # log.info("Already existing 'coeff_MCH.out' for %s in %s" % (traj, INFOS["setupstates_names"][initstate]))
                        update = False 
                        continue
                    else:
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

    sys.stdout.write("\n")

# ======================================================================= #


def gfsh_probs(istate, ic, INFOS):
    # istate, ic are counter, not states / TRAJ names
    if INFOS["diag"]:
        coeff_file = "coeff_diag.out"
    else:
        coeff_file = "coeff_MCH.out"
    # coeffs = np.zeros((coeff_data.shape[0], int((coeff_data.shape[1]-2)/2)), dtype=complex)
    # TODO Decide, whether coeff_data.shape makes more sense or tsteps, setupstates
    # Check, whether every file has same number of time steps, in case data_extractor failed in the middle of extracting
    #coeffs = np.zeros((INFOS["nsteps"]+1, INFOS["nstates"]), dtype=complex)  # nstates
    coeffs = np.zeros((INFOS["nsteps"]+1, sum(INFOS["states"][i] * (i + 1) for i in range(len(INFOS["states"])))), dtype=complex)  # nstates*multiplicity
    time_steps, states = coeffs.shape
    for n_columns in range(0, len(coeffs[1])):
        coeffs[:, int(n_columns)] = INFOS["coeff_data"][ic, istate, :, int(2*(n_columns+1))] + 1.j*INFOS["coeff_data"][ic, istate, :, int(2*(n_columns+1)+1)]
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
    #data_to_save = np.column_stack((range(time_steps), exc_prob))
    #np.savetxt("exc_state_%05i_traj_%05i" % (istate, ic+1), data_to_save, fmt="%.5e", delimiter="\t")
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

    INFOS["coeff_data"] = np.zeros((INFOS["ninit"], len(INFOS["setupstates"]), INFOS["nsteps"]+1, (2+2*int(INFOS["nstates"]))))  # initconds, initstates, timesteps, states
    for i, istate in enumerate(INFOS["setupstates"]):  
        for j, jcond in enumerate(INFOS["icond_sel"]):
            #  p_traj = 0.  # Initialize the accumulated probability to leave the initial state for this trajectory
            INFOS["coeff_data"][j, i, :, :] =  np.genfromtxt(get_iconddir(istate, INFOS) + "/TRAJ_%05i/output_data/%s" %(jcond, coeff_file), comments="#")
            # TODO: probably would suffice to only take the initial state - defining a big array is not needed
            # coeffs = np.zeros((coeff_data.shape[0], int((coeff_data.shape[1]-2)/2)), dtype=complex)
            # for n_columns in range(0, len(coeffs[1])):
            # coeffs[:, int(n_columns)] = coeff_data[:, int(2*(n_columns+1))] + 1.j*coeff_data[:, int(2*(n_columns+1)+1)]
            coeff_init = INFOS["coeff_data"][j, i, :, int(2*(istate-1+1))] + 1.j*INFOS["coeff_data"][j, i, :, int(2*(istate-1+1)+1)]  # skip first two columns containing time and c**2, but istates start at 1
            pstay = 1. 
            p_init = np.abs(coeff_init)**2
            for tstep in range(len(coeff_init)-1):
                pstay *= 1. - (max(0, 1-p_init[tstep+1]/p_init[tstep]))
                # if istate==5:
                #     if icond==3:
                #         print(tstep, pstay, p_init[tstep])
                # pstay *= 1. - (max(0, p_init[tstep]-p_init[tstep+1]))
            pleave = 1.-pstay
            # print("ISTATE: %i, ICOND: %i, pleave: %f" % (istate, icond, pleave))
            if pleave > pmax:
                pmax = pleave
            # p_traj = 1
            # for tstep in range(len(coeff_init)-1):
            #     p_init_tdiff = p_init[tstep]-p_init[tstep+1]
            #     if p_init_tdiff > 0:
            #         p_traj *= (1-p_init_tdiff)  # prob to never hop
            # if 1-p_traj > pmax:  # prob to hop at least once
            #     pmax = 1-p_traj
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
    # for ic, icond in enumerate(INFOS["icond_sel"]):
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
    for icond in range(1, INFOS["ninit"] + 1):
    # for ic, icond in enumerate(INFOS["icond_sel"]):
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
        initlist[icond-1].addstates(estates)
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


def read_coeff(INFOS, statelist, exc_list, istate):
    INFOS["start_coeff"] = question("Should the coefficients be stored for the full dynamics run? \n" + \
                                     "(0: No, 1: Yes, at the hopping times, 2: Yes, at the end of the electron-only dynamics, 3: Yes, at another time)", int, [0], False)
    if INFOS["start_coeff"][0] == 1:
        INFOS["exc_time_bool"] = True
        INFOS["coeff_bool"] = True
    elif INFOS["start_coeff"][0] == 2:
        INFOS["exc_time_bool"] = True
        INFOS["coeff_bool"] = True
    elif INFOS["start_coeff"][0] == 3:
        INFOS["exc_time_bool"] = True
        INFOS["coeff_bool"] = True
        while True:
            INFOS["coeff_time"] = question("The dynamics was set up from 0.0fs to %.2f fs. Which coefficients should be taken to initialize the full dynamics run? (0, %.2f)" \
                % (INFOS["tmax"]-INFOS["dtstep"], INFOS["tmax"]-INFOS["dtstep"]), str, str(INFOS["tmax"]-INFOS["dtstep"]), False)
            if float(INFOS["coeff_time"]) >= 0.0 and float(INFOS["coeff_time"]) <= INFOS["tmax"]-INFOS["dtstep"]:
                INFOS["coeff_time_idx"] = str(int(np.round(float(INFOS["coeff_time"])/INFOS["dtstep"], 0))) 
                break
            else:
                continue
    else:
        INFOS["exc_time_bool"] = True
        INFOS["coeff_bool"] = False
        INFOS["coeff_time_idx"] = str(np.nan) 
    coeff = np.zeros((INFOS["ninit"], INFOS["nstates"], INFOS["nstates"], 2))  # NTRAJ, NSTATES (only @ excited states filled)k NSTATES, COMPLEX 
    for itraj, traj in enumerate(["TRAJ_%05i" % int(el) for el in INFOS["icond_sel"]]):  # cycle through all TRAJ_directories in initstate folder
        if int(exc_list[istate, itraj, 2]) == 1:  # IF STATE/TRAJ COMBINATION IS EXCITED
            # done = width_bar * (itraj) // len(INFOS["icond_sel"])
            # sys.stdout.write("\r  Progress: [" + "=" * done + " " * (width_bar - done) + "] %3i%%" % (done * 100 // width_bar))
            try:
                match INFOS["start_coeff"][0]:
                    case 0:  # Coeff from pure state 
                        pass
                    case 1:  # Coeffs from hopping time 
                        coeff[itraj, INFOS["setupstates"][istate]-1, :, 0] = INFOS["coeff_data"][itraj, istate, int(exc_list[istate, itraj, 0]), 2::2]
                        coeff[itraj, INFOS["setupstates"][istate]-1, :, 1] = INFOS["coeff_data"][itraj, istate, int(exc_list[istate, itraj, 0]), 3::2]
                    case 2:  # Coeffs from last timestep 
                        coeff[itraj, INFOS["setupstates"][istate]-1, :, 0] = INFOS["coeff_data"][itraj, istate, -1, 2::2]
                        coeff[itraj, INFOS["setupstates"][istate]-1, :, 1] = INFOS["coeff_data"][itraj, istate, -1, 3::2]
                    case 3:  # Coeffs from custom timestep
                        coeff[itraj, INFOS["setupstates"][istate]-1, :, 0] = INFOS["coeff_data"][itraj, istate, int(INFOS["coeff_time_idx"]), 2::2]
                        coeff[itraj, INFOS["setupstates"][istate]-1, :, 1] = INFOS["coeff_data"][itraj, istate, int(INFOS["coeff_time_idx"]), 3::2]
            except OSError:
                print(f"Trajectory {traj} does not exist for setup state {istate}!")
        print(coeff[itraj, INFOS["setupstates"][istate]-1, :, :])
        print(istate, itraj, int(exc_list[istate, itraj, 0]), coeff.shape)
        statelist[itraj].get_coeff(coeff[itraj, INFOS["setupstates"][istate]-1, :, :], INFOS["coeff_bool"])
    return statelist    

# ======================================================================= #
# ======================================================================= #
# ======================================================================= #


def writeoutput(initlist, istate, INFOS):
    dirname = get_iconddir(INFOS["setupstates"][istate], INFOS)
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
excitation_times     %s
explicit_coefficients     %s
"""
        % (version, INFOS["ninit"], INFOS["natom"], INFOS["repr"], INFOS["eref"], INFOS["eharm"], INFOS["exc_time_bool"], INFOS["coeff_bool"])
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

    for ic, icond in enumerate(initlist):
        outf.write("Index     %i\n%s" % (ic + 1, str(icond)))
    # outf.write(string)
    outf.close()
    return 0


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


def sample_number():
    print("{:-^60}".format("Sample iterations of initial conditions") + "\n")
    print('Please enter a the number of iterations to sample the initial conditions.')
    while True:
        line = question("Sample iterations: ", int, [1], False)
        try:
            sample_number = int(line[0])
        except ValueError:
            print('Please enter an integer.')
            continue
        break
    print("")
    return sample_number


def excite(INFOS, initlist, exc_list, setupstate):
    print("\nSelecting initial states ...")
    width_bar = 50
    # for i, icond in enumerate(initlist):
    for ic, icond in enumerate(INFOS["icond_sel"]):
        # done = width_bar * (i + 1) // len(initlist)
        done = width_bar * (ic + 1) // len(INFOS["icond_sel"])
        sys.stdout.write("\r  Progress: [" + "=" * done + " " * (width_bar - done) + "] %3i%%" % (done * 100 // width_bar))
        if initlist[ic].statelist == []:
            continue
        else:
            if exc_list[setupstate, ic, 2]:
                for j, jstate in enumerate(initlist[ic].statelist):
                    if exc_list[setupstate, ic, 1]==j:
                        jstate.Excited = True
                        jstate.ExcTime = exc_list[setupstate, ic, 0]*INFOS["tmax"]/INFOS["nsteps"]
                        jstate.IState = setupstate+1
                    else:
                        jstate.Excited = False
                        jstate.ExcTime = ""
                        jstate.IState = ""

    # statistics
    nexc = [0]
    ntotal = [0]
    for ic, icond in enumerate(initlist):
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
        # TODO Instead of ntotal, write number of initconds in TRAJ folders
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
    INFOS["sample_number"] = sample_number()
    # INFOS = get_initconds(INFOS) - all necessary information in json file?
    # TODO: check that the istates are correctly called - they start with 1, not 0
    initlist = []
    for i, istate in enumerate(INFOS["setupstates"]):
        initlist.append(get_initconds(INFOS))
        initlist[i] = get_QMout(INFOS, i, initlist[i])  # adding excited state information
        run_data_extractor(i, INFOS)

    INFOS["max_prob"] = compute_max_prob(INFOS)  # Compute maximum probability to leave initial state
    print("Computed pmax = %.2f" % INFOS["max_prob"])
    exc_probs = np.zeros((len(INFOS["setupstates"]), len(INFOS["icond_sel"]), INFOS["nsteps"]+1, sum(INFOS["states"][i] * (i + 1) for i in range(len(INFOS["states"])))))
    exc_probs_cumsum = np.zeros_like(exc_probs) 
    exc_list = np.zeros((len(INFOS["setupstates"]), len(INFOS["icond_sel"]), 3))  # last index: 0: exc.time, 1: to which state was excited, 2: excitation or not
    double_exc = False
    for isa, isample in enumerate(range(INFOS["sample_number"])):  # Sample up to the point, where double excitations occur; To increase probabilities
        isa_exc_list = np.zeros_like(exc_list)
        if double_exc:
            continue
        for ist, istate in enumerate(INFOS["setupstates"]):
            if double_exc:
                continue
            for ic, icond in enumerate(INFOS["icond_sel"]):
                if double_exc:
                    continue
                jump_to_next = False
                if isa == 0:
                    exc_probs[ist, ic, :, :] = gfsh_probs(ist, ic, INFOS)
                    exc_probs_cumsum[ist, ic, :, :] = np.cumsum(exc_probs[ist, ic, :, :], axis=1) 
                random_probs = []
                for tstep in range(0, INFOS["nsteps"]+1):
                    no_random = random.random()
                    random_probs.append(no_random)
                    if jump_to_next:
                        continue
                    for exc_state in range(1, sum(INFOS["states"][i] * (i + 1) for i in range(len(INFOS["states"])))+1):
                        if jump_to_next:
                            continue
                        if no_random <= exc_probs_cumsum[ist, ic, tstep, exc_state-1]:
                            isa_exc_list[ist, ic, :] = tstep, exc_state-1, 1.0 
                            jump_to_next = True
                if isa_exc_list[ist, ic, 2] == 1.0 and exc_list[ist, ic, 2] == 1.0:
                    print("Double excitation @ run: %i. Only keep excitations up to sampling iteration %i" %(isa+1, isa))
                    print
                    double_exc = True  # Skip all further iterations of isa - one TRAJ would be excited twice
        exc_list += isa_exc_list
    for ist, istate in enumerate(INFOS["setupstates"]):
        initlist[ist] = read_coeff(INFOS, initlist[ist], exc_list, istate-1)  # istate -1, because it could also be state 5 that is active
        initlist[ist] = excite(INFOS, initlist[ist], exc_list[:, :, :], ist)
        writeoutput(initlist[ist], ist, INFOS) 
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
