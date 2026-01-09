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

import datetime
import os
import sys
import json
import copy
# import math
import numpy as np
import subprocess as sp
import scipy.constants as const
# from scipy.linalg import fractional_matrix_power
# from itertools import starmap, chain
from optparse import OptionParser
from qmout import QMout
import random

from constants import IToMult, U_TO_AMU, HARTREE_TO_EV
from utils import itnmstates, readfile, question as question_def
from printing import printheader
from scipy.signal import butter, filtfilt
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
    if isinstance(data, list):
        return [_byteify(item, ignore_dicts=True) for item in data]
    if isinstance(data, dict) and not ignore_dicts:
        return {_byteify(key, ignore_dicts=True): _byteify(value, ignore_dicts=True) for key, value in data.items()}
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
    def __init__(self, i=0, e=0.0, eref=0.0, dip=[0.0, 0.0, 0.0], magdip=[0.0, 0.0, 0.0], elquad=[[0] * 3 for _ in range(3)]):
        self.i = i
        self.e = e.real
        self.eref = eref.real
        self.dip = dip
        self.magdip = magdip
        self.elquad = elquad
        self.Excited = False
        self.ExcTime = ""
        self.IState = "" 
        self.Eexc = self.e - self.eref
        self.Fosc = (2.0 / 3.0 * self.Eexc * sum([i * i.conjugate() for i in self.dip])).real
        # Magnetic dipole contribution (10.1063/1.4766359)
        self.Fosc += (2/3.*self.Eexc * sum([np.imag(i)**2 for i in self.magdip])).real
        # Electric quadrupole contribution (10.1063/1.4766359)
        Q = np.array(self.elquad, dtype=complex).reshape(3, 3)
        trace_Q = np.trace(Q)
        quad_term = np.sum(np.abs(Q)**2) - (1.0/3.0) * np.abs(trace_Q)**2
        self.Fosc += (1.0/20.0) * const.alpha**2 * self.Eexc**3 * quad_term
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
        self.magdip = [complex(try_read(f, i, float, 0.0), try_read(f, i + 1, float, 0.0)) for i in [3, 5, 7]]
        self.elquad = [complex(try_read(f, i, float, 0.0), try_read(f, i + 1, float, 0.0)) for i in [3, 5, 7]]
        self.Excited = try_read(f, 11, bool, False)
        self.ExcTime = try_read(f, 12, str, "")
        self.IState = try_read(f, 13, str, "")
        self.Eexc = self.e - self.eref
        self.Fosc = (2.0 / 3.0 * self.Eexc * sum([i * i.conjugate() for i in self.dip])).real
        # Magnetic dipole contribution (10.1063/1.4766359)
        self.Fosc += (2/3.*self.Eexc * sum([np.imag(i)**2 for i in self.magdip])).real
        # Electric quadrupole contribution (10.1063/1.4766359)
        Q = np.array(self.elquad, dtype=complex).reshape(3, 3)
        trace_Q = np.trace(Q)
        quad_term = np.sum(np.abs(Q)**2) - (1.0/3.0) * np.abs(trace_Q)**2
        self.Fosc += (1.0/20.0) * const.alpha**2 * self.Eexc**3 * quad_term
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
        while True:
            line = f.readline()
            if line.startswith("Ekin"):
                break
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


def run_data_extractor(setupstate_list, INFOS):
    """
    Extract output.dat in every TRAJ folder for every setupstate
    """
    forbidden = ['crashed', 'running', 'dead', 'dont_analyze']
    for idx_setupstate, setupstate in enumerate(setupstate_list):
        dirname = INFOS["setupstates_names"][idx_setupstate]  # get_iconddir(initstate, INFOS)  # State directory containing trajectories 
        req_files_traj = ["geom", "input", "laser", "run.sh", "veloc"]
        req_folders_traj = ["QM", "restart"]
        width_bar = 80
        print(f"\nRunning data_extractor for State {INFOS['setupstates'][idx_setupstate]} ...")
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
                    if os.path.isfile(os.path.join(traj_path, "output_data/coeff_MCH.out")):
                        # log.info("Already existing 'coeff_MCH.out' for %s in %s" % (traj, INFOS["setupstates_names"][initstate]))
                        update = False 
                        continue
                    else:
                        extract_arg = "-cm"
                    if any([os.path.isfile(os.path.join(traj_path, forbid_file)) for forbid_file in forbidden]):
                        update = False
                    # check whether output_data/expec.out is newer than output.dat
                    # TODO New check for corrupted trajectory folders
                    # if not os.path.isfile(path + '/output_data/expec.out'):
                    #     update = True
                    if not update:
                        time_dat = os.path.getmtime(os.path.join(traj_path, 'output.dat'))
                        time_expec = os.path.getmtime(os.path.join(traj_path, 'output_data/coeff_MCH.out'))
                        if time_dat > time_expec or INFOS['run_extractor_full']:
                            update = True
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


def smooth_population(INFOS):
    # Check for the coefficient basis
    done = 0
    width_bar = 50
    print("\nNumber of initial conditions in file:       %5i" % (INFOS["ninit"]))
    coeff_file = "coeff_MCH.out"
    if INFOS["smoothing"]:
        INFOS["freq_cutoff"] = question("Which cutoff energy should be taken for the lowpass filter: (a.u.)", float, [INFOS["eq_energies"][-1]-INFOS["eq_energies"][0]])[0]
        order = 4
        dt = INFOS["tmax"]/INFOS["nsteps"]
        nyquist = 1 / (2.*dt)
        wn = INFOS["freq_cutoff"] / nyquist
        b, a = butter(order, wn, btype='low')
        padlen = 3 * max(len(a), len(b))
        print(f"Padding length in fs: {padlen*dt}")
    INFOS["coeff_data"] = np.zeros((len(INFOS["setupstates"]), INFOS["ninit"], INFOS["nsteps"]+1, (2+2*len(INFOS["statemap"]))))  # initconds, initstates, timesteps, states
    INFOS["rho_data"] = np.zeros((len(INFOS["setupstates"]), INFOS["ninit"], INFOS["nsteps"]+1, (len(INFOS["statemap"]))))  # initconds, initstates, timesteps, states
    if INFOS["smoothing"]:
        print("Smoothing populations!")
        for i_setupstate, setupstate in INFOS["setupstates"]-1:
            for j, jcond in enumerate(INFOS["icond_sel"]):
                # TRAJ, INITSTATE, TIME, STATEMAP
                traj_path = os.path.join(INFOS["setupstates_names"][i_setupstate], "/TRAJ_%05i/output_data" %(jcond))
                INFOS["coeff_data"][i_setupstate, j, :, :] = np.genfromtxt(os.path.join(traj_path, coeff_file), comments="#")
                INFOS["rho_data"][i_setupstate, j, :, :] = np.abs(INFOS["coeff_data"][i_setupstate, j, :, 2::2] + 1.j*INFOS["coeff_data"][i_setupstate, j, :, 3::2])**2  # skip first two columns containing time and c**2, but istates start at 1
                # Apply Low-pass Butterworth filter 
                INFOS["rho_data"][i_setupstate, j, :, :] = filtfilt(b, a, INFOS["rho_data"][i_setupstate, j, :, :], axis=0, padtype="constant", padlen=padlen)
                if os.access(traj_path, os.W_OK):
                    data = np.column_stack((INFOS["coeff_data"][i_setupstate, 0, :, 0], np.sum(INFOS["rho_data"][i_setupstate, j, :, :], axis=-1), INFOS["rho_data"][i_setupstate, j, :, :]))
                    header = (
                    "# " + " ".join([f"{i+1:>20}" for i in range(data.shape[1])]) + "\n"
                    "#               Time |           Sum c**2 |  === pop_mch ===>\n"
                    "#               [fs] |                 [] |                 [] |\n"
                    )
                    with open(os.path.join(traj_path, "pop_smooth_mch.out"), "w") as f:
                        f.write(header)
                        for row in data:
                            line = "  " + " ".join([f"{val: .12E}" for val in row])
                            f.write(line + "\n")
                else:
                    if j==0:
                        print("Smoothed population files are not written due to lack of permission!")
                done = width_bar * (j) // (len(INFOS["icond_sel"]))
                sys.stdout.write("\r  Progress: [" + "=" * done + " " * (width_bar - done) + "] %3i%%" % (done * 100 // width_bar))
    else:
        print("No smoothing applied to populations!")
        for idx_setupstate, setupstate in enumerate(INFOS["setupstates"]):
            for j, jcond in enumerate(INFOS["icond_sel"]):
                traj_path = os.path.join(INFOS["setupstates_names"][idx_setupstate], "TRAJ_%05i/output_data" %(jcond))  # TRAJ, INITSTATE, TIME, STATEMAP
                INFOS["coeff_data"][idx_setupstate, j, :, :] = np.genfromtxt(os.path.join(traj_path, coeff_file), comments="#")
                INFOS["rho_data"][idx_setupstate, j, :, :] = np.abs(INFOS["coeff_data"][idx_setupstate, j, :, 2::2] + 1.j*INFOS["coeff_data"][idx_setupstate, j, :, 3::2])**2  # skip first two columns containing time and c**2, but istates start at 1
    return INFOS


# ======================================================================= #


def gfsh_probs(rho, ist, ic, INFOS):
    exc_pop = np.zeros((INFOS["nsteps"]+1, len(INFOS["statemap"])))
    exc_pop_tdiff = np.zeros((INFOS["nsteps"], len(INFOS["statemap"])))
    max_pop = np.zeros(INFOS["nsteps"])
    gs_fac = (1 - (rho[1:, ist] / rho[:-1, ist]))  # [time, ground state]
    for tstep in range(INFOS["nsteps"]):
        for exc_state in range(len(INFOS["statemap"])):
            exc_pop_tdiff[tstep, exc_state] = rho[tstep+1, exc_state] - rho[tstep, exc_state]
            if exc_pop_tdiff[tstep, exc_state] < 0.:  #  if difference is negative -> negative added to sum
                max_pop[tstep] -= exc_pop_tdiff[tstep, exc_state]
    for tstep in range(INFOS["nsteps"]):
        if gs_fac[tstep]>0:
            for exc_state in range(len(INFOS["statemap"])):
                if exc_state==ist:
                    continue
                if exc_pop_tdiff[tstep, exc_state] < 0.:
                    continue
                else:
                    exc_pop[tstep, exc_state] = np.max([0, gs_fac[tstep]*exc_pop_tdiff[tstep, exc_state]/max_pop[tstep]])
    return exc_pop


# ======================================================================= #


def compute_max_prob(INFOS, rho_read=np.array([])):
    """
    Extract output.dat in every TRAJ folder for single setupstate
    """
    pmax = np.zeros(len(INFOS["setupstates"]))  # initialize maximum probability over all initial states and initial conditions (TRAJS) to ever leave the initial state
    if rho_read.shape[0] != 0:  # if not the default rho_read
        for setupstate in INFOS["setupstates"]:
            pleave_arr = np.zeros(len(INFOS["icond_sel"]))
            for j, jcond in enumerate(INFOS["icond_sel"]):
                # TODO: is INFOS["setup_states"] correct, or should one take the current state of each individual traj?
                p_init = rho_read[j, :, setupstate-1]  # take the rho_list icond j and setup state j (ground state)
                pstay = 1.
                for tstep in range(INFOS["nsteps"]):                     
                    pstay *= 1. - (max(0, 1-p_init[tstep+1]/p_init[tstep]))
                pleave_arr[j] = 1.-pstay                                    
    else:
        for idx_setupstate, setupstate in enumerate(INFOS["setupstates"]):
            pleave_arr = np.zeros(len(INFOS["icond_sel"]))
            for j, jcond in enumerate(INFOS["icond_sel"]):
                rho_init = INFOS["rho_data"][idx_setupstate, j, :, setupstate-1] 
                pstay = 1. 
                p_init = rho_init
                for tstep in range(len(rho_init)-1):
                    pstay *= 1. - (max(0, 1-p_init[tstep+1]/p_init[tstep]))
                pleave_arr[j] = 1.-pstay
            pmax[idx_setupstate] = np.max(pleave_arr)*INFOS["renorm_scale_fac"]
    # print("PMAX", pmax)
    # print("Initconds index for pmax:",np.argmax(pleave_arr))
    return pmax  #, pleave_arr


# ======================================================================= #


def get_initconds(INFOS):
    """
    """
    initlist = []
    width_bar = 50
    for ic, icond in enumerate(INFOS["icond_sel"]):
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

    print(f"\nReading QM.out data of state {initstate+1} ...")
    ncond = 0
    width_bar = 50
    eq_qmout = QMout(filepath=INFOS["path"]+"/ICOND_00000/QM.out")
    INFOS["eq_energies"] = np.einsum("ii->i", eq_qmout.h.real)  
    for ic, icond in enumerate(INFOS["icond_sel"]):
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
        if "mdeqm" in INFOS["needed_requests"]:
            MDM = qmout.mdm
            EQM = qmout.eqm
        estates = []
        for istate in range(len(H)):
            dip = [DM[i][initstate][istate] for i in range(3)]
            if "mdeqm" in INFOS["needed_requests"]:
                magdip = [MDM[i][initstate][istate] for i in range(3)]
                elquad = [EQM[i][j][initstate][istate] for i in range(3) for j in range(3)]
                estate = STATE(len(estates) + 1, H[istate][istate], H[initstate][initstate], dip, magdip, elquad)
            else:
                estate = STATE(len(estates) + 1, H[istate][istate], H[initstate][initstate], dip)
            estates.append(estate)
        initlist[icond-1].addstates(estates)
    print("\nNumber of initial conditions with QM.out:   %5i" % (ncond))
    return initlist, INFOS


# ======================================================================= #


def get_iconddir(istate, INFOS):
    if INFOS["diag"]:  # For the naming of the folder, the initial 
        dirname = "State_%i" % (istate)
    else:
        mult, state, ms = INFOS["statemap"][istate]
        dirname = IToMult[mult] + "_%i" % (state - (mult == 1 or mult == 2))
    return dirname


def read_coeff(INFOS, setup_statelist, exc_list):
    # TODO: Make compatible with reading in smoothed rho instead of coeff
    # INFOS["start_coeff"] = question("Should the coefficients be stored for the full dynamics run? \n" + \
    #                                  "(0: No, 1: Yes, at the hopping times, 2: Yes, at the end of the electron-only dynamics, 3: Yes, at another time)", int, [0], False)
    # if INFOS["start_coeff"][0] == 1:
    INFOS["start_coeff"] = [1]
    INFOS["exc_time_bool"] = True
    INFOS["coeff_bool"] = False
    # elif INFOS["start_coeff"][0] == 2:
    #     INFOS["exc_time_bool"] = True
    #     INFOS["coeff_bool"] = True
    # elif INFOS["start_coeff"][0] == 3:
    #     INFOS["exc_time_bool"] = True
    #     INFOS["coeff_bool"] = True
    #     while True:
    #         INFOS["coeff_time"] = question("The dynamics was set up from 0.0fs to %.2f fs. Which coefficients should be taken to initialize the full dynamics run? (0, %.2f)" \
    #             % (INFOS["tmax"]-INFOS["dtstep"], INFOS["tmax"]-INFOS["dtstep"]), str, str(INFOS["tmax"]-INFOS["dtstep"]), False)
    #         if float(INFOS["coeff_time"]) >= 0.0 and float(INFOS["coeff_time"]) <= INFOS["tmax"]-INFOS["dtstep"]:
    #             INFOS["coeff_time_idx"] = str(int(np.round(float(INFOS["coeff_time"])/INFOS["dtstep"], 0))) 
    #             break
    #         else:
    #             continue
    # else:
    #     INFOS["exc_time_bool"] = True
    #     INFOS["coeff_bool"] = False
    #     INFOS["coeff_time_idx"] = str(np.nan) 
    coeff = np.zeros((len(INFOS["setupstates"]), INFOS["ninit"], len(INFOS["statemap"]), 2))  # NTRAJ, NSTATES (only @ excited states filled)k NSTATES, COMPLEX 
    for idx_setupstate, setupstate in enumerate(INFOS["setupstates"]):
        print(len(setup_statelist))
        for itraj, traj in enumerate(["TRAJ_%05i" % int(el) for el in INFOS["icond_sel"]]):  # cycle through all TRAJ_directories in initstate folder
            if int(exc_list[idx_setupstate, itraj, 2]) == 1:  # IF TRAJ COMBINATION IS EXCITED
                try:
                    match INFOS["start_coeff"][0]:
                        case 0:  # Coeff from pure state 
                            pass
                        # case 1:  # Coeffs from hopping time 
                        #     coeff[itraj, :, 0] = INFOS["coeff_data"][itraj, istate, int(exc_list[istate, itraj, 0]), 2::2]
                        #     coeff[itraj, :, 1] = INFOS["coeff_data"][itraj, istate, int(exc_list[istate, itraj, 0]), 3::2]
                        # case 2:  # Coeffs from last timestep 
                        #     coeff[itraj, :, 0] = INFOS["coeff_data"][itraj, istate, -1, 2::2]
                        #     coeff[itraj, :, 1] = INFOS["coeff_data"][itraj, istate, -1, 3::2]
                        # case 3:  # Coeffs from custom timestep
                        #     coeff[itraj, :, 0] = INFOS["coeff_data"][itraj, istate, int(INFOS["coeff_time_idx"]), 2::2]
                        #     coeff[itraj, :, 1] = INFOS["coeff_data"][itraj, istate, int(INFOS["coeff_time_idx"]), 3::2]
                except OSError:
                    print(f"Trajectory {traj} does not exist for setup state {INFOS['setupstate'][idx_setupstate]}!")
            setup_statelist[idx_setupstate][itraj].get_coeff(coeff[idx_setupstate, itraj], INFOS["coeff_bool"])
    return setup_statelist


# ======================================================================= #

# TODO: Make pleave_arr for every setupstate
# def write_probabilities(rho, initlist, exc_list, INFOS):
#     pmax, pleave_arr = compute_max_prob(INFOS, rho)  # returns list of pmax for all setupstates
#     n_states = sum(INFOS["states"][i] * (i + 1) for i in range(len(INFOS["states"])))
#     n_trajs = len(INFOS["icond_sel"])
# 
#     exc_energies_all = np.zeros((n_trajs, n_states))
#     osc_strengths_all = np.zeros((n_trajs, n_states))
# 
#     for ic, icond in enumerate(INFOS["icond_sel"]):
#         for j, jstate in enumerate(initlist[ic].statelist):
#             exc_energies_all[ic, j] = jstate.Eexc
#             osc_strengths_all[ic, j] = jstate.Fosc
# 
#     with open("probabilities.txt", "w") as f:
#         header = ["#No.TRAJ", "ptotk_old", "ptotk_new"]
#         header += [f"Eexc{j+1}" for j in range(n_states)]
#         header += [f"Fosc{j+1}" for j in range(n_states)]
#         header += ["Exc", "Exc_State", "Exc_Time"]
#         f.write(", ".join(header) + "\n")
# 
#         n_digits = 5  
#         for k in range(n_trajs):
#             traj_label = f"TRAJ_{INFOS['icond_sel'][k]:0{n_digits}d}"
#             row = [traj_label, f"{INFOS['pleave_arr_old'][k]:.16f}", f"{pleave_arr[k]:.16f}"]
#             row += [f"{exc_energies_all[k, m]:.16f}" for m in range(n_states)]
#             row += [f"{osc_strengths_all[k, m]:.16f}" for m in range(n_states)]
#             row += [f"{exc_list[k, 2]:.16f}"] 
#             row += [f"{exc_list[k, 1]:.16f}"] 
#             row += [f"{exc_list[k, 0]:.12e}"] 
#             f.write("".join(s.rjust(20) for s in row) + "\n")


def writeoutput(setupstate_initlist, INFOS):
    for idx_setupstate, setupstate in enumerate(setupstate_initlist): 
        dirname = get_iconddir(INFOS["setupstates"][idx_setupstate], INFOS)
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
    
        for ic, icond in enumerate(setupstate):
            for j, jstate in enumerate(setupstate[ic].statelist):
                if jstate.Excited:
                    print(ic, j, jstate.IState, jstate.ExcTime)
            outf.write("Index     %i\n%s" % (ic + 1, str(icond)))
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


def scale_pmax():
    print("{:-^60}".format("Renormalization of hopping probabilities") + "\n")
    print('Please enter a number greater than 1 (full renormalization). High values result in lower excitation yields.')
    while True:
        renorm_scale_fac = question("Renormalization scaling factor: ", float, [1.], False)[0]
        if not renorm_scale_fac>=1.:
            print("Must be >=1!")
            continue
        break
    print("")
    return renorm_scale_fac


def sample_number():
    # print("{:-^60}".format("Sample iterations of initial conditions") + "\n")
    # print('Please enter a the number of iterations to sample the initial conditions.')
    # while True:
    #     line = question("Sample iterations: ", int, [1], False)
    #     try:
    #         sample_number = int(line[0])
    #     except ValueError:
    #         print('Please enter an integer.')
    #         continue
    #     break
    # print("")
    return 1


def excite(INFOS, setupstate_initlist, exc_list):
    width_bar = 50
    for idx_setupstate, setupstate in enumerate(setupstate_initlist):
        print(f"\nSelecting initial states for setup state {INFOS['setupstates'][idx_setupstate]} ...")
        for ic, icond in enumerate(INFOS["icond_sel"]):
            done = width_bar * (ic + 1) // len(INFOS["icond_sel"])
            sys.stdout.write("\r  Progress: [" + "=" * done + " " * (width_bar - done) + "] %3i%%" % (done * 100 // width_bar))
            if setupstate[ic].statelist == []:
                continue
            else:
                if exc_list[idx_setupstate, ic, 2]:
                    for j, jstate in enumerate(setupstate[ic].statelist):
                        if exc_list[idx_setupstate, ic, 1]-1==j:
                            jstate.Excited = True
                            jstate.IState = exc_list[idx_setupstate, ic, 1] 
                            jstate.ExcTime = exc_list[idx_setupstate, ic, 0]*INFOS["tmax"]/INFOS["nsteps"]
                        else:
                            jstate.Excited = False
                            jstate.ExcTime = ""
                            jstate.IState = ""

        nexc = [0]
        ntotal = [0]
        for ic, icond in enumerate(setupstate):
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
        print(f"\nNumber of initial conditions excited for setupstate {INFOS['setupstates'][idx_setupstate]}:")
        print("State   Selected     Total")
        for i in range(len(ntotal)):
            # TODO Instead of ntotal, write number of initconds in TRAJ folders
            print("  % 3i      % 4i      % 4i" % (i + 1, nexc[i], ntotal[i]))
        setupstate_initlist[idx_setupstate] = setupstate
    return setupstate_initlist



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
    INFOS["renorm_scale_fac"] = scale_pmax()
    INFOS["smoothing"] = question("Should the population be smoothed before analysis?", bool, True) 
    INFOS["max_hops"] = question("What is the max. allowed number of hops (including back-hops)", int, [99999])[0] 
    INFOS["sample_number"] = sample_number()
    initlist = get_initconds(INFOS)
    setupstate_initlist = [
        copy.deepcopy(get_QMout(INFOS, setupstate-1, initlist)[0])
        for setupstate in INFOS["setupstates"]
    ]
    run_data_extractor(setupstate_initlist, INFOS)
    INFOS = smooth_population(INFOS)
    INFOS["max_prob"] = np.max(compute_max_prob(INFOS))  # Compute maximum probability to leave initial state for all setupstates
    print("Computed pmax = %.5f" % INFOS["max_prob"])
    exc_pop = np.zeros((len(INFOS["setupstates"]), len(INFOS["icond_sel"]), INFOS["nsteps"]+1, len(INFOS["statemap"])))
    exc_pop_cumsum = np.zeros_like(exc_pop) 
    exc_list = np.zeros((len(INFOS["setupstates"]), len(INFOS["icond_sel"]), 3))  # last index: 0: exc.time, 1: from which state was excited, 2: excitation or not
    hop_logs = []
    hop_header = ("# traj hop_idx time_fs from_state to_state dE_eV fosc rand p_all_states prob_this\n")
    exc_list[:, :, 1] = np.zeros((len(INFOS["setupstates"]), len(INFOS["icond_sel"])))  # Initialize the initstate for the excitations
    rho_renorm = np.zeros_like(exc_pop)
    rho_renorm[:, :, :, :] = np.abs(INFOS["rho_data"])/INFOS["max_prob"]  # readout rho_data is already squared - setupstates, ...
    for idx_setupstate, setupstate in enumerate(INFOS["setupstates"]):
        rho_renorm[idx_setupstate, :, :, INFOS["setupstates"][idx_setupstate]-1] = 1.-np.sum(rho_renorm[idx_setupstate, :, :, :], axis=-1)+rho_renorm[idx_setupstate, :, :, INFOS["setupstates"][idx_setupstate]-1]
        for ic, icond in enumerate(INFOS["icond_sel"]):
            exc_pop[idx_setupstate, ic, :, :] = gfsh_probs(rho_renorm[idx_setupstate, ic, :, :], INFOS["setupstates"][idx_setupstate]-1, ic, INFOS)  # time, exc_state
            exc_pop_cumsum[idx_setupstate, ic, :, :] = np.cumsum(exc_pop[idx_setupstate, ic, :, :], axis=1) 
            random_probs = []
            hop_idx = 0
            to_state = 0
            for tstep in range(0, INFOS["nsteps"]+1):
                no_random = random.random()
                random_probs.append(no_random)
                # TODO: prevent from "Hops" from setupstate to the same setupstate
                for exc_state in range(1, len(INFOS["statemap"])+1):
                    if no_random <= exc_pop_cumsum[idx_setupstate, ic, tstep, exc_state-1] and hop_idx <=INFOS["max_hops"]:
                        hop_idx +=1 
                        to_state = exc_state - 1
                        time_fs = INFOS["tmax"] * tstep / INFOS["nsteps"]
                        dE = initlist[ic].statelist[to_state].Eexc - initlist[ic].statelist[INFOS["setupstates"][idx_setupstate]-1].Eexc
                        fosc = initlist[ic].statelist[to_state].Fosc
                        probs_all = exc_pop[idx_setupstate, ic, tstep, :]
                        prob_this = probs_all[to_state]

                        hop_logs.append({
                            "traj": icond,
                            "hop_idx": hop_idx,
                            "time_fs": time_fs,
                            "from": INFOS['setupstates'][idx_setupstate]-1,
                            "to": to_state,
                            "dE_ev": dE * const.physical_constants["Hartree energy in eV"][0],
                            "fosc": fosc,
                            "rand": no_random,
                            "probs_all": probs_all.copy(),  # optional, can skip to save memory
                            "prob_this": prob_this,
                            "last_hop": 0
                        })
                        exc_list[idx_setupstate, ic, :] = tstep, INFOS["setupstates"][idx_setupstate], 1.0 
                        exc_pop[idx_setupstate, ic, :, :] = gfsh_probs(rho_renorm[idx_setupstate, ic, :, :], INFOS["setupstates"][idx_setupstate]-1, ic, INFOS)  # time, exc_state
                        exc_pop_cumsum[idx_setupstate, ic, :, :] = np.cumsum(exc_pop[idx_setupstate, ic, :, :], axis=1) 
            if to_state != 0:
                hop_logs[-1]["last_hop"] = 1
        print(f"Finished EOE rescaling for State {INFOS['setupstates'][idx_setupstate]}!")


    try:
        with open("hop.log", "w") as f:
            f.write(hop_header)
            for h in hop_logs:
                line = (
                    f"{h['traj']:6d} {h['hop_idx']:6d} "
                    f"{h['time_fs']:10.4f} {h['from']+1:4d} {h['to']+1:4d} "
                    f"{h['dE_ev']: 12.6e} {h['fosc']:12.6e} {h['rand']:12.6e} "
                    f"{' '.join(f'{p:.4e}' for p in h['probs_all'])} "
                    f"{h['prob_this']:.4e}"
                    f"{h['last_hop']:6d}\n"
                )
                f.write(line)
        print(f"\n Hop log written to hop.log")
    except Exception as e:
        print(f"Could not write hop log: {e}")

    setupstate_initlist = read_coeff(INFOS, setupstate_initlist, exc_list)  # istate -1, because it could also be state 5 that is active
    setupstate_initlist = excite(INFOS, setupstate_initlist, exc_list)
    writeoutput(setupstate_initlist, INFOS) 
    # write_probabilities(rho_renorm, initlist, exc_list, INFOS)

    close_keystrokes()


# ======================================================================= #


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nCtrl+C occured. Exiting.\n")
        sys.exit()
