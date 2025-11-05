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

# Command line script to use promdens to excite an initconds file (with state info)
#
# usage: python excite_PDA.py [options] initconds.excited

import re
import sys
import numpy as np
from logger import log
from constants import D2au, U_TO_AMU, HARTREE_TO_EV, au2fs
import argparse
import subprocess
import sys
from printing import printheader
import os
import datetime
import shutil



# =========================== initconds file reading and writing =================

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
            atomlist.append(line.strip())
        statelist = []
        while True:
            line = f.readline()
            if "Ekin" in line:
                break
            state = STATE()
            state.init_from_str(line)
            statelist.append(state)
        for istate,state in enumerate(statelist):
            if state.e == state.eref:
                break
        for state in statelist:
            state.IState = istate+1
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
            s += "% 12.8f % 12.8f %s % 03i % s" % (self.Eexc * HARTREE_TO_EV, self.Fosc, self.Excited, self.IState, self.ExcTime)
        return s


def read_ic(filename):
    """
    Read a SHARC initial condition file into a list of INITCOND objects + metadata dict.
    """
    initlist = []
    INFOS = {}
    INFOS["initf"] = filename

    with open(filename, "r") as f:
        header = f.readline()
        if not header.startswith("SHARC Initial conditions file"):
            raise ValueError("Not a valid SHARC IC file")

        # parse header keywords
        while True:
            line = f.readline()
            if line.strip() == "":
                break
            key, *vals = line.split()
            key = key.strip()
            if key.lower() == "ninit":
                INFOS["ninit"] = int(vals[0])
            elif key.lower() == "natom":
                INFOS["natom"] = int(vals[0])
            elif key.lower() == "states":
                INFOS["states"] = [int(i) for i in vals]
            elif key.lower() == "repr":
                INFOS["repr"] = vals[0]
            elif key.lower() == "eref":
                INFOS["eref"] = float(vals[0])
            elif key.lower() == "eharm":
                INFOS["eharm"] = float(vals[0])
            elif key.lower().startswith("excitation_times"):
                INFOS["exc_time_bool"] = vals[0]
            elif key.lower().startswith("explicit_coefficients"):
                INFOS["coeff_bool"] = vals[0]

        while True:
            line = f.readline()
            if "Equilibrium" in line:
                break
        equi = []
        for i in range(INFOS["natom"]):
            line = f.readline()
            # atom = ATOM()
            # atom.init_from_str(line)
            equi.append(line)
        INFOS["equi"] = equi

        # Now loop over IC blocks
        width_bar = 50
        for idx in range(INFOS["ninit"]):
            ic = INITCOND()
            ic.init_from_file(f, INFOS["eref"], index=idx+1)
            initlist.append(ic)
            done = width_bar * (idx+1) // INFOS["ninit"]
            sys.stdout.write("\r  Progress: [" + "=" * done + " " * (width_bar - done) + "] %3i%%" % (done * 100 // width_bar))
        print()
    INFOS["n_exc_states"] = len(initlist[0].statelist)

    # provide some details
    print("Number of initial conditions from header: %i" % INFOS["ninit"])
    print("Number of initial conditions from list: %i" % len(initlist))
    print("Number of atoms: %i" % INFOS["natom"])
    print("Number of states: %s" % str(INFOS["states"]))
    print()

    return initlist, INFOS

# ==================================== PDA interface =========================

def write_promdens_input(initlist, filename, nstates):
    with open(filename, "w") as f:
        f.write("#index    dE01 (a.u.)  |mu_01| (Debye)  ....\n")
        for idx, ic in enumerate(initlist, start=1):
            #print(idx, ic)
            line = f"{idx:5d}"
            # write all states, even the initial one
            if len(ic.statelist) != nstates:
                print(f"Initial condition {idx} does not have {nstates} states -> Ignored")
                continue
            for st in ic.statelist:
                #print(st)
                dE = st.Eexc  # in a.u.
                mu = np.linalg.norm([c for c in st.dip]) / D2au # in D
                if dE == 0.:
                    mu = 0.
                line += f"   {dE:12.8f}   {mu:8.4f}"
            f.write(line + "\n")

# ==================

def run_promdens(promdens_in, args, nstates):

    if shutil.which("promdens") is None:
        print("ERROR: 'promdens' is not found in your PATH. ")
        print("You can install it with 'pip install promdens'.")
        sys.exit(1)

    # Forward all remaining args to the external script
    cmd = ["promdens", str(promdens_in)]

    # Insert your automatically managed parameters
    cmd += [
        # "--nsamples", str(0),
        "--nstates", str(nstates),
        "--energy_unit", "a.u.",
        "--tdm_unit", "debye",
        "--file_type", "file",
    ]

    # Forward everything else
    if args.rest:
        cmd += args.rest

    print("Running:", " ".join(cmd))
    result = subprocess.run(cmd, check=False)


# ==================

def parse_promdens_output(initlist, dt):

    print("#############################################################")
    print("#                  End of PROMDENS execution                #")
    print("#############################################################")
    print("... back in the wrapper. Now reading the promdens output and writing initconds file...")
    print("- considering each initial condition/excited state only once")
    print(f"- rounding starting times from promdens to multiples of {dt}fs")
    print()

    # --- Step 1: Read PDA file and collect matches ---
    # We'll map (index, el_state-1) -> excitation time in fs
    pda_file = "pda.dat"
    excitation_data = {}

    skip_count = 0
    with open(pda_file, "r") as f:
        for line in f:
            if line.strip().startswith("#") or not line.strip():
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            try:
                index = int(parts[0])
                exc_time_fs = float(parts[1]) * au2fs
                exc_time_fs = round(exc_time_fs / dt) * dt
                el_state = int(parts[2])
                if (index, el_state) in excitation_data:
                    print(f"An initial condition and initial state was picked more than once: {(index, el_state)} -> Skipped")
                    skip_count += 1
                else:
                    excitation_data[(index, el_state)] = exc_time_fs
            except ValueError:
                continue

    if skip_count > 0:
        print("\nWARNING: Some initial conditions/excited states were picked more than once!")
        print("Consider reducing --nsamples")
        print()

    print(f"Read {len(excitation_data)} excitation entries from PDA file.")
    print(excitation_data)
    
    # --- Step 2: Update initlist ---
    for (idx, el_state), exc_time_fs in excitation_data.items():
        initlist[idx-1].statelist[el_state-1].Excited = True
        initlist[idx-1].statelist[el_state-1].ExcTime = exc_time_fs



# ==================


version = "4.0"
versionneeded = [0.2, 1.0, 2.0, 2.1, float(version)]
versiondate = datetime.date(2025, 11, 1)


# ==================


def displaywelcome():
    lines = [
        f"Compute excitation probabilities and excitation times via promdens",
        "",
        f"Authors: Lorenz Grünewald, Sebastian Mai",
        "",
        f"Version: {version}",
        "Date: {:%d.%m.%Y}".format(versiondate),
    ]

    printheader(lines)
    string = "This script is a wrapper for running SHARC initconds through promdens."
    print(string)
        
# ==================

def writeoutput(initlist, outfile, INFOS):
    if outfile:
        outfilename = str(outfile)
    else:
        outfilename = INFOS["initf"] + ".excited"
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
        % (version, INFOS["ninit"], INFOS["natom"], INFOS["repr"], INFOS["eref"], INFOS["eharm"], True, False)
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

    for ic, icond in enumerate(initlist):
        outf.write("Index     %i\n%s" % (ic + 1, str(icond)))
    outf.close()
    return 0


# add a hook into print_help to also print the help of prom_dens
class MyParser(argparse.ArgumentParser):
    def print_help(self):
        super().print_help()
        print("\n\nHelp message of promdens --help is following.")
        print("Note that --nsamples, --nstates, --energy_unit, --tdm_unit, --file_type, and --plot are ignored by excite_from_promdens.py")
        print("\n\n--- promdens --help ---\n")
        if shutil.which("promdens"):
            subprocess.run(["promdens", "--help"], check=False)
        else:
            print("promdens not found in PATH.")



# ==================================== Own code =========================

def main():

    # Create wrapper parser
    parser = MyParser(
        description="Command line script to use promdens to excite an initconds file (with state info)"
    )
    parser.add_argument("ic_file", help="Path to the initconds.excited file")
    parser.add_argument(
        "-o", "--output",
        help="Name of the output initconds file (optional). "
    )
    parser.add_argument("--dt", type=float, default=0.5, help="Time step (fs) intended for later dynamics")
    # --- Settings that the user cannot set for promdens ---
    parser.add_argument("--nsamples", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--nstates", type=int, default=1, help=argparse.SUPPRESS)
    parser.add_argument("--energy_unit", default="a.u.", help=argparse.SUPPRESS)
    parser.add_argument("--tdm_unit", default="debye", help=argparse.SUPPRESS)
    parser.add_argument("--file_type", default="file", help=argparse.SUPPRESS)
    parser.add_argument("--plot", default="file", help=argparse.SUPPRESS)
    parser.add_argument(
        "rest", nargs=argparse.REMAINDER,
        help="Arguments to pass to the external PDA script"
    )
    args = parser.parse_args()


    # 1. Read IC file (using your INITCOND logic)
    ic_file = args.ic_file
    initlist, INFOS = read_ic(ic_file) 
    # 2. Write promdens input
    promdens_in = "input_file.dat"
    write_promdens_input(initlist, promdens_in, INFOS["n_exc_states"])
    # 3. Run promdens
    run_promdens(promdens_in, args, INFOS["n_exc_states"])
    # 4. Parse output and update ICs
    parse_promdens_output(initlist, args.dt)
    # 5. Write updated excited ICs
    writeoutput(initlist, args.output, INFOS=INFOS)


if __name__ == "__main__":
    try:
        displaywelcome()
        main()
    except KeyboardInterrupt:
        print("\nCtrl+C makes me a sad SHARC ;-(\n")
        quit(0)
