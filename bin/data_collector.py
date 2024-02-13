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

import copy
import math
import sys
import os
import shutil
import datetime
from itertools import islice
import numpy as np
from scipy import stats
from utils import question, expand_path
from printing import printheader

# =========================================================0
# some constants
DEBUG = False
PI = math.pi

version = "4.0"
versiondate = datetime.date(2024, 1, 26)


PIPELINE = """
**************************************************************************************************
Pipeline:
=========
            Collecting
                 |
                 1-------\
                 |       |
                 |   Smoothing
                 |       |
                 |<------/
                 |
  /--------------2--------\
  |                       |
  |                 Synchronizing
  |                       |
  |          /------------3--------------\
  |          |                           |
  |          |                     Convoluting(X)
  |          |                           |
  |          4-------\\                   6---------\
  |          |       |                   |         |
  |          |   Averaging               |     Summing(Y)
  |          |       |                   |         |
  |          |<------/                   |<--------/
  |          |                           |
  |          5-------\\                   7---------\
  |          |       |                   |         |
  |          |   Statistics              |   Integrating(X)
  |          |       |                   |         |
  |          |<------/                   |<--------/
  |          |                           |
  |          |                           8---------\
  |          |                           |         |
  |          |               /-----------9         |
  |          |               |           |         |
  |          |         Integrating(T)    |   Convoluting(T)
  |          |               |           |         |
  |          |               \\---------->|         |
  |          |                           |         |
  |          |                           |<--------/
  |          |                           |
  |          |<-------------------------10
  |          |                           |
Type1      Type2                       Type3

Procedure Explanations:
=======================
- Collecting:
  extracts the requested columns from the files, creating a Type1 dataset

- Smoothing:
  applies some smoothing procedure to each trajectory independently, creating a new Type1 dataset

- Synchronizing:
  merges all trajectories into one dataset with a common time axis, creating a Type2 dataset

- Averaging:
  for each X and Y column, compute average and standard deviation across all trajectories, creating a new Type2 dataset

- Statistics:
  for each X and Y column, compute average and standard deviation across all trajectories and all time steps, creating a new Type2 dataset

- Convoluting(X):
  merges all trajectories into a dataset with common time and X axes, using a convolution along the X axis; creates a Type3 dataset

- Sum(Y):
  if multiple Y values are present, sum them up for each T and X, creating a new Type3 dataset

- Integrating(X):
  integrates along the X axis within some specified bounds, creating a new Type3 dataset

- Convoluting(T):
  applies a convolution along the time axis, creating a new Type3 dataset

- Integrating(T):
  performs a cumulative summation along the time axis (such that the final integral is stored in the last time step), creating a new Type3 dataset

Dataset Explanations:
====================
- Type1 dataset:
  Independent trajectories with possibly different time axes
  (not intended for plotting)
  ***
##i  path                 time   x1   x2  ...   y1   y1  ...
  0  TRAJ_00001/filename   0.0  1.6  3.1  ...  0.1  6.1  ...
  0  TRAJ_00001/filename   0.5  1.7  2.7  ...  0.1  6.0  ...
  0  TRAJ_00001/filename   1.0  1.9  2.2  ...  0.2  6.1  ...
  ...

  1  TRAJ_00002/filename   0.0  1.7  2.9  ...  0.2  6.2  ...
  1  TRAJ_00002/filename   1.0  1.8  2.3  ...  0.1  6.3  ...
  1  TRAJ_00002/filename   2.0  1.7  1.6  ...  0.1  6.1  ...
  ...
  ***


- Type2 dataset:
  Data with a common time axis, with possibly missing entries for some trajectories
  (can be plotted as hair figures, etc)
  ***
##       <-- TRAJ_00001 -->  ...  <-- TRAJ_00002 -->  ...
##time   x1   y1   x2   y2  ...   x1   y1   x2   y2  ...
  0.0    1.6  0.1  3.1  6.1  ...  1.7  0.2  2.9  6.2  ...
  0.5    1.7  0.1  2.7  6.0  ...  nan  nan  nan  nan  ...
  1.0    1.9  0.2  2.2  6.1  ...  1.8  0.1  2.3  6.3  ...
  ...
  ***


- Type3 dataset:
  Common time and X axes, Y values obtained by convolution
  (can be plotted as 3D plots)
  ***
##time   x   y1  y2   ...
  0.0  1.2  0.2  0.0  ...
  0.0  1.3  0.3  0.0  ...
  0.0  1.4  0.5  0.0  ...
  0.0  1.5  0.7  0.0  ...
  ...

  0.5  1.2  0.2  0.0  ...
  0.5  1.3  0.3  0.0  ...
  0.5  1.4  0.5  0.0  ...
  0.5  1.5  0.7  0.0  ...
  ...
  ***
**************************************************************************************************
"""


# =============================================================================================== #
# =============================================================================================== #
# =========================================== general routines ================================== #
# =============================================================================================== #
# =============================================================================================== #


class gauss:
    def __init__(self, fwhm):
        self.fwhm = fwhm
        self.c = -4.0 * np.log(2.0) / fwhm**2  # this factor needs to be evaluated only once

    def ev(self, A, x0, x):
        return A * np.exp(self.c * (x - x0) ** 2)  # this routine does only the necessary calculations


class lorentz:
    def __init__(self, fwhm):
        self.fwhm = fwhm
        self.c = 0.25 * fwhm**2

    def ev(self, A, x0, x):
        return A / ((x - x0) ** 2 / self.c + 1)


class boxfunction:
    def __init__(self, fwhm):
        self.fwhm = fwhm
        self.w = 0.5 * fwhm

    def ev(self, A, x0, x):
        res = np.zeros_like(x)
        res[abs(x - x0) < self.w] = A
        return res


class lognormal:
    def __init__(self, fwhm):
        self.fwhm = fwhm

    def ev(self, A, x0, x):
        if x <= 0 or x0 <= 0:
            return 0.0
        # for lognormal distribution, the factor for the exponent depends on x0
        c = (np.log((self.fwhm + np.sqrt(self.fwhm**2 + 4.0 * x0**2)) / (2.0 * x0))) ** 2
        # note that the function does not take a value of A at x0
        # instead, the function is normalized such that its maximum will have a value of A (at x<=x0)
        return A * x0 / x * np.exp(-c / (4.0 * np.log(2.0)) - np.log(2.0) * (np.log(x) - np.log(x0)) ** 2 / c)


kernels = {
    1: {"f": gauss, "description": "Gaussian function", "factor": 1.0},
    2: {"f": lorentz, "description": "Lorentzian function", "factor": 2.0},
    3: {"f": boxfunction, "description": "Rectangular window function", "factor": 0.6},
    4: {"f": lognormal, "description": "Log-normal function", "factor": 1.5},
}


class spectrum:
    def __init__(self, npts, emin, emax, fwhm, lineshape):
        self.npts = npts
        if lineshape == 1:
            self.f = gauss(fwhm)
        elif lineshape == 2:
            self.f = lorentz(fwhm)
        elif lineshape == 3:
            self.f = boxfunction(fwhm)
        elif lineshape == 4:
            self.f = lognormal(fwhm)
        self.en = np.fromiter(
            map(lambda i: emin + float(i) / self.npts * (emax - emin), range(self.npts + 1)), dtype=float, count=self.npts + 1
        )  # the energy grid needs to be calculated only once
        self.spec = np.zeros((self.npts + 1), dtype=float)

    def add(self, A, x0):
        if A == 0.0:
            return
        self.spec += self.f.ev(A, x0, self.en)


def in_pairs(iterable):
    it = iter(iterable)
    while batch := tuple(islice(it, 2)):
        yield batch


# ======================================================================================================================
# ======================================================================================================================
# ======================================================================================================================


def displaywelcome():
    print("Script for data collecting started...\n\n")
    lines = [
        "Reading table data from SHARC dynamics",
        "",
        "Authors: Sebastian Mai, Severin Polonius",
        "Version:" + version,
        versiondate.strftime("%d.%m.%y"),
    ]
    printheader(lines)
    print(
        """
This script collects table data from SHARC trajectories, smooths them, synchronizes them,
convolutes them, and computes averages and similar statistics.
  """
    )


# ======================================================================================================================
# ======================================================================================================================
# ======================================================================================================================


def open_keystrokes():
    global KEYSTROKES
    KEYSTROKES = open("KEYSTROKES.tmp", "w")


def close_keystrokes():
    KEYSTROKES.close()
    shutil.move("KEYSTROKES.tmp", "KEYSTROKES.data_collector")


# ===================================


global KEYSTROKES
old_question = question


def question(question, typefunc, default=None, autocomplete=True, ranges=False):
    return old_question(
        question=question, typefunc=typefunc, KEYSTROKES=KEYSTROKES, default=default, autocomplete=autocomplete, ranges=ranges
    )


# ======================================================================================================================
# ======================================================================================================================
# ======================================================================================================================


class histogram:
    def __init__(self, binlist):
        """binlist must be a list of floats
        Later, all floats x with binlist[i-1]<x<=binlist[i] will return i"""
        self.binlist = sorted(binlist)
        self.len = len(binlist) + 1

    def put(self, x):
        i = 0
        for el in self.binlist:
            if x <= el:
                return i
            else:
                i += 1
        return i

    def __repr__(self):
        s = "Histogram object: "
        for i in self.binlist:
            s += "%f " % (i)
        return s


def replace_middle_column_name(name, replace_str):
    v, desc, num = name.split()
    return f"{v} {desc:<6s} {num:>3s}"


def prepend_middle_column_name(name, addition):
    v, desc, num = name.split()
    return f"{v} {addition+desc:<6s} {num:>3s}"


# ======================================================================================================================
# ======================================================================================================================
# ======================================================================================================================


def get_general():
    """"""

    INFOS = {}

    # -------------------------------- Running data_extractor or geo.py ----------------------------
    # TODO

    # ---------------------------------------- File selection --------------------------------------

    print("{:-^60}".format("Paths to trajectories"))
    print(
        '\nPlease enter the paths to all directories containing the "TRAJ_0XXXX" directories.\nE.g. Sing_2/ and Sing_3/. \nPlease enter one path at a time, and type "end" to finish the list.'
    )
    count = 0
    paths = []
    while True:
        path = question("Path: ", str, "end")
        if path == "end":
            if len(paths) == 0:
                print("No path yet!")
                continue
            print("")
            break
        path = expand_path(path)
        if not os.path.isdir(path):
            print("Does not exist or is not a directory: %s" % (path))
            continue
        if path in paths:
            print("Already included.")
            continue
        ls = os.listdir(path)
        print(ls)
        for i in ls:
            if "TRAJ" in i or "ICOND" in i:
                count += 1
        print("Found %i subdirectories in total.\n" % count)
        paths.append(path)
    INFOS["paths"] = paths
    print("Total number of subdirectories: %i\n" % (count))

    # make list of TRAJ paths
    width = 50
    forbidden = ["crashed", "running", "dead", "dont_analyze"]
    dirs = []
    ntraj = 0
    print("Checking the directories...")
    for idir in INFOS["paths"]:
        ls = os.listdir(idir)
        for itraj in sorted(ls):
            if "TRAJ_" not in itraj and "ICOND_" not in itraj:
                continue
            path = idir + "/" + itraj
            if not os.path.isdir(path):
                continue
            s = path + " " * (width - len(path))
            lstraj = os.listdir(path)
            valid = True
            for i in lstraj:
                if i.lower() in forbidden:
                    s += "DETECTED FILE %s" % (i.lower())
                    # print(s)
                    valid = False
                    break
            if not valid:
                continue
            s += "OK"
            # print(s)
            ntraj += 1
            dirs.append(os.path.relpath(path))
    print("Number of trajectories: %i" % (ntraj))
    if ntraj == 0:
        print("No valid trajectories found, exiting...")
        sys.exit(0)

    print("\nDo you want to see all common files before specifying the filepath to analyse?:")
    if question("Yes or no?:", bool, default=True):
        # check the dirs
        print("Checking for common files...")
        allfiles = {}
        exclude_dirs = {"SCRATCH", "SAVE", "QM", "restart", "MMS", "MML"}
        exclude = [
            "template",
            "resources",
            "runQM.sh",
            "QM.in",
            "QM.out",
            "QM.log",
            "QM.err",
            "output.dat",
            "output.dat.nc",
            "output.log",
            "output.xyz",
            "output.dat.ext",
            "input",
            "geom",
            "veloc",
            "coeff",
            "atommask",
            "laser",
            "run.sh",
            "restart",
            ".*init",
            "STOP",
            "CRASHED",
            "RUNNING",
            "DONT_ANALYZE",
            "table" "driver",
            "rattle",
        ]
        for d in dirs:
            for dirpath, dirnames, filenames in os.walk(d, topdown=True):
                dirnames[:] = [
                    d for d in dirs if d not in exclude_dirs
                ]  # from https://stackoverflow.com/questions/19859840/excluding-directories-in-os-walk
                for f in filter(lambda x: not any(ex in x for ex in exclude), filenames):
                    line = os.path.join(os.path.relpath(dirpath, d), f)
                    if line in allfiles:
                        allfiles[line] += 1
                    else:
                        allfiles[line] = 1
        allfiles = {k: v for k, v in allfiles.items() if v >= 2}

        print("\nList of files common to the trajectory directories:\n")
        print("%6s %20s   %s" % ("Index", "Number of appearance", "Relative file path"))
        print("-" * 58)
        allfiles_index = {}
        for iline, line in enumerate(sorted(allfiles)):
            allfiles_index[iline] = line
            print("%6i %20i   %s" % (iline, allfiles[line], line))

        # choose one of these files
        print("\nPlease give the relative file path of the file you want to collect:")
        while True:
            string = question("File path or index:", str, "0", False)
            try:
                string = allfiles_index[int(string)]
            except ValueError:
                pass
            if string in allfiles:
                INFOS["filepath"] = string
                break
            else:
                print("I did not understand %s" % string)

        # make list of files
        allfiles = []
        for d in dirs:
            f = os.path.join(d, INFOS["filepath"])
            # if os.path.isfile(f):
            allfiles.append(f)
        INFOS["allfiles"] = allfiles
    else:
        print("\nPlease give the relative file path of the file you want to collect:")
        while True:
            INFOS["filepath"] = question("File path:", str, ".", False)
            absent = []
            allfiles = []
            print("Checking if file exists in directories...")
            for i, d in enumerate(dirs):
                done = 50 * (i + 1) // len(dirs)
                sys.stdout.write("\r  Progress: [" + "=" * done + " " * (50 - done) + "] %3i%%" % (done * 100 / 50))
                f = os.path.join(d, INFOS["filepath"])
                if os.path.isfile(f):
                    allfiles.append(f)
                else:
                    absent.append(d)
            sys.stdout.write("\n")
            if len(absent) != 0:
                print(f"\n{INFOS['filepath']} is absent in {absent}")
                if question("Continue anyway?", bool, False):
                    break
            else:
                break

        INFOS["allfiles"] = allfiles

    # ---------------------------------------- Columns --------------------------------------

    print("\n" + "{:-^60}".format("Data columns") + "\n")
    # get number of columns
    filename = allfiles[0]
    testfile = open(filename, "r")
    for line in testfile:
        if "#" not in line:
            ncol = len(line.split())
            break
    testfile.close()
    print("Number of columns in the file:   %i" % (ncol))
    INFOS["ncol"] = ncol

    # select columns
    print("\nPlease select the data columns for the analysis:")
    print("For T column: \n  only enter one (positive) column index. \n  If 0, the line number will be used instead.")
    print(
        "For X column: \n  enter one or more column indices. \n  If 0, all entries of that column will be set to 1. \n  If negative, the read numbers will be multiplied by -1."
    )
    print(
        "For Y column: \n  enter as many column indices as for X. \n  If 0, all entries of that column will be set to 1. \n  If negative, the read numbers will be multiplied by -1."
    )
    print("")
    while True:
        INFOS["colT"] = question("T column (time):", int, [1])[0]
        if 0 <= INFOS["colT"] <= ncol:
            # 0:   use line number (neglecting commented or too short lines)
            # 1-n: use that line for time data
            break
        else:
            print("Please enter a number between 0 and %i!" % ncol)
    while True:
        INFOS["colX"] = question("X columns:", int, [2], ranges=True)
        if all([-ncol <= x <= ncol for x in INFOS["colX"]]):
            INFOS["nX"] = len(INFOS["colX"])
            break
        else:
            print("Please enter a set of numbers between %i and %i!" % (-ncol, ncol))
    while True:
        default = [0 for i in INFOS["colX"]]
        INFOS["colY"] = question("Y columns:", int, default, ranges=True)
        if all([-ncol <= x <= ncol for x in INFOS["colY"]]) and len(INFOS["colY"]) == len(INFOS["colX"]):
            INFOS["nY"] = len(INFOS["colY"])
            break
        else:
            print("Please enter a set of %i numbers between %i and %i!" % (len(INFOS["colX"]), -ncol, ncol))

    print("Selected columns:")
    print("T: %s     X: %s    Y: %s\n" % (str(INFOS["colT"]), str(INFOS["colX"]), str(INFOS["colY"])))

    # ---------------------------------------- Analysis procedure --------------------------------------

    print("{:-^60}".format("Analysis procedure") + "\n")
    show = question("Show possible workflow options?", bool, True)
    if show:
        print("\nThe following diagram shows which workflows are possible with this script:")
        print(PIPELINE)

    # Question 1
    print("\n" + "{:-^40}".format("1 Smoothing") + "\n")
    if question("Do you want to apply smoothing to the individual trajectories?", bool, False):
        print("\nChoose one of the following smoothing functions:")
        for i in sorted(kernels):
            print("%i  %s" % (i, kernels[i]["description"]))
        while True:
            i = question("Choose one of the functions:", int, [1])[0]
            if i in kernels:
                break
            else:
                print("Choose one of the following: %s" % (list(kernels)))
        w = question("Choose width of the smoothing function (in units of column %i):" % (INFOS["colT"]), float, [10.0])[0]
        INFOS["smoothing"] = {"function": kernels[i]["f"](w)}
    else:
        INFOS["smoothing"] = {}

    # Question 2
    print("\n" + "{:-^40}".format("2 Synchronizing") + "\n")
    if question("Do you want to synchronize the data?", bool, True):
        INFOS["synchronizing"] = True
    else:
        INFOS["synchronizing"] = False

    # first branching
    INFOS["averaging"] = {}
    INFOS["statistics"] = {}
    INFOS["convolute_X"] = {}
    INFOS["sum_Y"] = False
    INFOS["integrate_X"] = {}
    INFOS["convolute_T"] = {}
    INFOS["integrate_T"] = {}
    INFOS["type3_to_type2"] = False

    # Question 3
    if INFOS["synchronizing"]:
        print("\n" + "{:-^40}".format("3 Convoluting along X") + "\n")
        if question("Do you want to apply convolution in X direction?", bool, False):
            print("\nChoose one of the following convolution kernels:")
            for i in sorted(kernels):
                print("%i  %s" % (i, kernels[i]["description"]))
            while True:
                kern = question("Choose one of the functions:", int, [1])[0]
                if kern in kernels:
                    break
                else:
                    print("Choose one of the following: %s" % (list(kernels)))
            w = question("Choose width of the smoothing function (in units of the X columns):", float, [1.0])[0]
            INFOS["convolute_X"] = {"function": kernels[kern]["f"](w)}
            # print('Choose the size of the grid along X:')
            INFOS["convolute_X"]["npoints"] = question("Size of the grid along X:", int, [25])[0]
            print("\nChoose minimum and maximum of the grid along X:")
            print("Enter either a single number a (X grid from  xmin-a*width  to  xmax+a*width)")
            print("        or two numbers a and b (X grid from  a  to  b)")
            INFOS["convolute_X"]["xrange"] = question("Xrange:", float, [kernels[kern]["factor"]])
            if len(INFOS["convolute_X"]["xrange"]) > 2:
                INFOS["convolute_X"]["xrange"] = INFOS["convolute_X"]["xrange"][:2]

    # Question 4
    if INFOS["synchronizing"] and not INFOS["convolute_X"]:
        print("\n" + "{:-^40}".format("4 Averaging") + "\n")
        if question("Do you want to average the data columns across all trajectories?", bool, False):
            print("Choose one of the following options:")
            print("%i  %s" % (1, "Arithmetic average and standard deviation"))
            print("%i  %s" % (2, "Geometric average and standard deviation"))
            while True:
                av = question("Choose one of the options:", int, [1])[0]
                if av in [1, 2]:
                    break
                else:
                    print("Choose one of the following: %s" % ("[1, 2]"))
            if av == 1:
                INFOS["averaging"] = {"mean": mean_arith, "stdev": stdev_arith}
            elif av == 2:
                INFOS["averaging"] = {"mean": mean_geom, "stdev": stdev_geom}

    # Question 4
    if INFOS["synchronizing"] and not INFOS["convolute_X"]:
        print("\n" + "{:-^40}".format("5 Total statistics") + "\n")
        if question("Do you want to compute the total mean and standard deviation over all time steps?", bool, False):
            print("Choose one of the following options:")
            print("%i  %s" % (1, "Arithmetic average and standard deviation"))
            print("%i  %s" % (2, "Geometric average and standard deviation"))
            while True:
                av = question("Choose one of the options:", int, [1])[0]
                if av in [1, 2]:
                    break
                else:
                    print("Choose one of the following: %s" % ("[1, 2]"))
            if av == 1:
                INFOS["statistics"] = {"mean": mean_arith, "stdev": stdev_arith}
            elif av == 2:
                INFOS["statistics"] = {"mean": mean_geom, "stdev": stdev_geom}

    # Question 6
    if INFOS["synchronizing"] and INFOS["convolute_X"]:
        print("\n" + "{:-^40}".format("6 Sum over all Y") + "\n")
        INFOS["sum_Y"] = question("Do you want to sum up all Y values?", bool, False)

    # Question 7
    if INFOS["synchronizing"] and INFOS["convolute_X"]:
        print("\n" + "{:-^40}".format("7 Integrate along X") + "\n")
        if question("Do you want to integrate in X direction?", bool, False):
            print("Please specify the lower and upper bounds for the integration:")
            while True:
                INFOS["integrate_X"]["xrange"] = question("Xmin and Xmax:", float, [0.0, 10.0])
                if len(INFOS["integrate_X"]["xrange"]) >= 2:
                    INFOS["integrate_X"]["xrange"] = INFOS["integrate_X"]["xrange"][:2]
                    break

    # Question 8
    if INFOS["synchronizing"] and INFOS["convolute_X"]:
        print("\n" + "{:-^40}".format("8 Convoluting along T") + "\n")
        if question("Do you want to apply convolution in T direction?", bool, False):
            print("Choose one of the following convolution kernels:")
            for i in sorted(kernels):
                print("%i  %s" % (i, kernels[i]["description"]))
            while True:
                kern = question("Choose one of the functions:", int, [1])[0]
                if kern in kernels:
                    break
                else:
                    print("Choose one of the following: %s" % (list(kernels)))
            w = question("Choose width of the smoothing function (in units of the T column):", float, [25.0])[0]
            INFOS["convolute_T"] = {"function": kernels[kern]["f"](w)}
            # print('Choose the size of the grid along X:')
            INFOS["convolute_T"]["npoints"] = question("Size of the grid along T:", int, [200])[0]
            print("\nChoose minimum and maximum of the grid along T:")
            print("Enter either a single number a (T grid from  xmin-a*width  to  xmax+a*width)")
            print("        or two numbers a and b (T grid from  a  to  b)")
            INFOS["convolute_T"]["xrange"] = question("Trange:", float, [kernels[kern]["factor"]])
            if len(INFOS["convolute_T"]["xrange"]) > 2:
                INFOS["convolute_T"]["xrange"] = INFOS["convolute_T"]["xrange"][:2]

    # Question 9
    if INFOS["synchronizing"] and INFOS["convolute_X"] and not INFOS["convolute_T"]:
        print("\n" + "{:-^40}".format("9 Integrating along T") + "\n")
        INFOS["integrate_T"] = question("Do you want to integrate in T direction?", bool, False)

    # Question 10
    if INFOS["synchronizing"] and INFOS["convolute_X"]:
        print("\n" + "{:-^40}".format("10 Convert to Type2 dataset") + "\n")
        print("If you performed integration along X, the data might be better formatted as Type2 dataset.")
        recommend = bool(INFOS["integrate_X"])
        INFOS["type3_to_type2"] = question("Do you want to output as Type2 dataset?", bool, recommend)

    # pprint.pprint(INFOS)
    # sys.exit(1)

    return INFOS


# ======================================================================================================================
# ======================================================================================================================
# ======================================================================================================================


def do_calc(INFOS):
    outindex = 0
    outstring = ""

    print("\n\n>>>>>>>>>>>>>>>>>>>>>> Started data analysis\n")

    # ---------------------- collect data -------------------------------
    print("Collecting the data ...")
    all_data = collect_data(INFOS)
    outindex = 1
    filename = make_filename(outindex, INFOS, outstring)
    print('>>>> Writing output to file "%s"...\n' % filename)
    write_type1(filename, all_data, INFOS)

    # ---------------------- apply temporal smoothing -------------------------------
    if INFOS["smoothing"]:
        print("Applying temporal smoothing ...")
        all_data = smoothing_xy(INFOS, all_data)
        outindex = 1
        outstring += "_sm"
        filename = make_filename(outindex, INFOS, outstring)
        print('>>>> Writing output to file "%s"...\n' % filename)
        write_type1(filename, all_data, INFOS)

    # ---------------------- apply synchronization -------------------------------
    if INFOS["synchronizing"]:
        print("Synchronizing temporal data ...")
        all_data2 = synchronize(all_data)

        n_names = len(INFOS["colnames"]) // 2
        INFOS["colnames"] = INFOS["colnames"][:n_names] * all_data2["arr"].shape[1] + INFOS["colnames"][n_names:] * all_data2["arr"].shape[1]

        outindex = 2
        outstring += "_sy"
        filename = make_filename(outindex, INFOS, outstring)
        print('>>>> Writing output to file "%s"...\n' % filename)
        write_type2(filename, all_data2, INFOS)

    # ---------------------- compute averages --------------------
    if INFOS["averaging"]:
        print("Computing averages ...")
        all_data2 = calc_average(INFOS, all_data2)
        outindex = 2
        outstring += "_av"
        filename = make_filename(outindex, INFOS, outstring)
        print('>>>> Writing output to file "%s"...\n' % filename)
        write_type2(filename, all_data2, INFOS, write_count=True)

    # ---------------------- compute averages --------------------
    if INFOS["statistics"]:
        print("Computing total statistics ...")
        all_data2 = calc_statistics(INFOS, all_data2)
        outindex = 2
        outstring += "_st"
        filename = make_filename(outindex, INFOS, outstring)
        print('>>>> Writing output to file "%s"...\n' % filename)
        write_type2(filename, all_data2, INFOS, write_count=True)

    # ---------------------- convoluting X --------------------
    if INFOS["convolute_X"]:
        print("Convoluting data (along X column) ...")
        all_data3 = do_x_convolution(INFOS, all_data2)
        outindex = 3
        outstring += "_cX"
        filename = make_filename(outindex, INFOS, outstring)
        print('>>>> Writing output to file "%s"...\n' % filename)
        write_type3(filename, all_data3, INFOS)

    # ---------------------- convoluting X --------------------
    if INFOS["sum_Y"]:
        print("Summing all Y values ...")
        all_data3 = do_y_summation(INFOS, all_data3)
        outindex = 3
        outstring += "_sY"
        filename = make_filename(outindex, INFOS, outstring)
        print('>>>> Writing output to file "%s"...\n' % filename)
        write_type3(filename, all_data3, INFOS)

    # ---------------------- integrating X --------------------
    if INFOS["convolute_X"] and INFOS["integrate_X"]:
        print("Integrating data (along X column) ...")
        all_data3 = integrate_X(INFOS, all_data3)
        outindex = 3
        outstring += "_iX"
        filename = make_filename(outindex, INFOS, outstring)
        print('>>>> Writing output to file "%s"...\n' % filename)
        write_type3(filename, all_data3, INFOS)

    # ---------------------- convoluting T --------------------
    if INFOS["convolute_X"] and INFOS["convolute_T"]:
        print("Convoluting data (along T column) ...")
        all_data3 = do_t_convolution(INFOS, all_data3)
        outindex = 3
        outstring += "_cT"
        filename = make_filename(outindex, INFOS, outstring)
        print('>>>> Writing output to file "%s"...\n' % filename)
        write_type3(filename, all_data3, INFOS)

    # ---------------------- integrating T --------------------
    if INFOS["convolute_X"] and INFOS["integrate_T"]:
        print("Integrating data (along T column) ...")
        all_data3 = integrate_T(INFOS, all_data3)
        outindex = 3
        outstring += "_iT"
        filename = make_filename(outindex, INFOS, outstring)
        print('>>>> Writing output to file "%s"...\n' % filename)
        write_type3(filename, all_data3, INFOS)

    # ---------------------- integrating T --------------------
    if INFOS["convolute_X"] and INFOS["type3_to_type2"]:
        print("Converting to Type2 dataset ...")
        data2 = type3_to_type2(INFOS, all_data3)
        outindex = 2
        outstring += "_cv"
        filename = make_filename(outindex, INFOS, outstring)
        print('>>>> Writing output to file "%s"...\n' % filename)
        write_type2(filename, data2, INFOS)

    print("Finished!")

    return INFOS


# ===============================================


def make_filename(outindex, INFOS, outstring):
    filename = "collected_data_%i_" % (INFOS["colT"])
    for i in INFOS["colX"]:
        filename += "%i" % i
    filename += "_"
    for i in INFOS["colY"]:
        filename += "%i" % i
    filename += outstring + ".type%i.txt" % (outindex)
    if len(filename) >= 255:
        filename = filename[:15] + "..." + filename[-35:]
    return filename


# ======================================================================================================================
# ======================================================================================================================
# ======================================================================================================================


def collect_data(INFOS):
    all_data = {}
    width_bar = 50
    columns = [i - 1 for i in INFOS["colX"]] + [i - 1 for i in INFOS["colY"]]

    colnames = [f"X Column {i:3d}" for i in INFOS["colX"]] + [
        f"Y Column {iy:3d}" for iy in INFOS["colY"]
    ]
    # print(columns, colnames)
    read_cols = sorted(set(filter(lambda x: x >= 0, [INFOS["colT"] - 1] + columns)))
    indices_data = list(map(read_cols.index, filter(lambda c: c in read_cols, columns)))
    indices_arr = [i for i, col in enumerate(columns) if col >= 0]

    for i, file in enumerate(INFOS["allfiles"]):
        done = width_bar * (i + 1) // len(INFOS["allfiles"])
        sys.stdout.write("\r  Progress: [" + "=" * done + " " * (width_bar - done) + "] %3i%%" % (done * 100 / width_bar))
        data = np.loadtxt(
            file,
            # delimiter="",
            dtype=float,
            comments="#",
            usecols=read_cols,
        )
        arr = np.ones((data.shape[0], len(INFOS["colX"]) * 2), dtype=float)
        # correct the order
        if INFOS["colT"] == 0:
            time = np.linspace(0, data.shape[0] - 1, data.shape[0], dtype=float)
        else:
            time = data[:, read_cols.index(INFOS["colT"] - 1)]
        
        arr[:, indices_arr] = data[:, indices_data]
        # arr is aranged over time, XorY, columns -> this makes it easy to access pairs of columns and data points
        all_data[file] = {"arr": arr.reshape(data.shape[0], 2, -1), "time": time}

    sys.stdout.write("\n")
    INFOS["columns"] = columns
    INFOS["colnames"] = colnames
    return all_data


def smoothing_xy(INFOS, data1: dict):
    data2 = {}
    f = INFOS["smoothing"]["function"]
    width_bar = 50
    for i, filekey in enumerate(data1.keys()):
        done = width_bar * (i + 1) / len(data1)
        sys.stdout.write("\r  Progress: [" + "=" * done + " " * (width_bar - done) + "] %3i%%" % (done * 100 / width_bar))
        sys.stdout.flush()

        time = data1[filekey]["time"]
        arr = data1[filekey]["arr"].reshape((len(time), -1))
        arr2 = arr.copy()

        for it, time1 in enumerate(time):
            temp_conv = f.ev(1.0, time1, time)
            scaled_temp_conv = temp_conv[:, None] * arr
            mask = temp_conv > 0.0
            n = np.sum(temp_conv[mask])
            s = np.sum(scaled_temp_conv[mask], axis=0)
            arr2 = s / n
        data2[filekey]["arr"] = arr2.reshape(data1[filekey]["arr"].shape)
        data2[filekey]["time"] = time
    sys.stdout.write("\n")
    return data2


# ===========================================


def synchronize(all_data):
    """
    creates dataframe with multicolumn and synchronizes times (fill value = NaN)

          1       2         3
          f1      f2        f3
          X  Y    X    Y    X    Y
    Time
    1     2  0  NaN  NaN  2.0  0.0
    4     5  1  5.0  1.0  NaN  NaN
    7     8  2  8.0  2.0  8.0  2.0
    """
    all_times = set()
    for filekey in sorted(all_data.keys()):
        if (len(all_data[filekey]["time"]) != len(all_times) and all_data[filekey]["time"][0] not in all_times and all_data[filekey]["time"][1] not in all_times):
            all_times = all_times.union(set((all_data[filekey]["time"] * 10000).astype(int)))
    all_times = np.array(sorted(all_times))
    all_times_idx = {k: i for i, k in enumerate(all_times)}
    
    file_keys = sorted(all_data.keys())
    arr = np.full((len(file_keys), len(all_times), 2, all_data[filekey]["arr"].shape[2]), np.nan)
    width_bar = 50
    for i, fk in enumerate(file_keys):
        done = width_bar * (i + 1) // len(file_keys)
        sys.stdout.write("\r  Progress: [" + "=" * done + " " * (width_bar - done) + "] %3i%%" % (done * 100 // width_bar))
        if (
                len(all_data[fk]["time"]) == len(all_times) and
                np.isclose(all_data[fk]["time"][0], all_times[0]) and
                np.isclose(all_data[fk]["time"][1], all_times[1], rtol=1e-8)):
            arr[i, ...] = all_data[fk]["arr"]
        else:
            idx = [all_times_idx[t] for t in (all_data[filekey]["time"] * 10000).astype(int)]
            arr[i, idx, ...] = all_data[fk]["arr"]

    # arr has shape time, files, XorY, cols
    return {"arr": np.einsum('ftxc->tfxc', arr), "time": all_times}


# ===========================================


def calc_average(INFOS, all_data):
    "calculates averages and stdevs over trajectory axis"
    nt, nf, _, nc = all_data["arr"].shape
    arr = all_data["arr"]
    # times, XorY, MeanorStdev, cols
    calc_data = np.zeros((nt, 2, 2, nc), dtype=float)
    cols = []
    for ic, col_id in enumerate(INFOS["colnames"]):
        if col_id.split()[2] == "0":
            continue
        cols.append(ic)
        ic = ic % nc  # wrap around XorY


        calc_data[:, :, 0, ic] = INFOS["averaging"]["mean"](arr[..., ic], axis=1)
        calc_data[:, :, 1, ic] = INFOS["averaging"]["stdev"](arr[..., ic], axis=1)

    count = np.sum(~np.isnan(arr), axis=1)

    INFOS["colnames"] = [replace_middle_column_name(INFOS["colnames"][c], "Mean") for c in cols]
    INFOS["colnames"].extend([replace_middle_column_name(INFOS["colnames"][c], "Stdev") for c in cols])
    INFOS["colnames"].append("Count")
    return {"arr": calc_data, "time": all_data["time"], "count": count}


# ===========================================


def calc_statistics(INFOS, all_data):
    "calculates averages and stdevs over expanding time axis"
    nt, nf, _, nc = all_data["arr"].shape
    arr = all_data["arr"]
    # times, XorY, MeanorStdev, cols
    calc_data = np.zeros((nf, 2, 2, nc), dtype=float)
    for it in range(nt):

        calc_data[it, :, 0, :] = INFOS["statistics"]["mean"](arr[:it + 1, ...], axis=0)
        calc_data[it, :, 1, :] = INFOS["statistics"]["stdev"](arr[:it + 1, ...], axis=0)

    INFOS["colnames"] = [prepend_middle_column_name(c, "Mean") for c in INFOS["colnames"]]
    INFOS["colnames"].extend([prepend_middle_column_name(c, "Stdev") for c in INFOS["colnames"]])
    return {"arr": calc_data, "time": all_data["time"]}


# ===========================================


def do_x_convolution(INFOS, all_data):
    # set up xrange
    width = INFOS["convolute_X"]["function"].fwhm
    xmin = np.min(all_data["arr"][:, :, 0, :])
    xmax = np.max(all_data["arr"][:, :, 0, :])
    if not INFOS["convolute_X"]["xrange"]:
        xmin = xmin - 2.0 * width
        xmax = xmax + 2.0 * width
    elif len(INFOS["convolute_X"]["xrange"]) == 2:
        xmin = INFOS["convolute_X"]["xrange"][0]
        xmax = INFOS["convolute_X"]["xrange"][1]
    elif len(INFOS["convolute_X"]["xrange"]) == 1:
        xmin = xmin - INFOS["convolute_X"]["xrange"][0] * width
        xmax = xmax + INFOS["convolute_X"]["xrange"][0] * width

    width_bar = 50

    arr = all_data["arr"]
    nt, nf, _, nc = arr.shape

    nps = INFOS["convolute_X"]["npoints"]
    ene_grid = np.linspace(xmin, xmax, nps)
    conv_func = INFOS["convolute_X"]["function"].ev
    max_A = np.max(all_data["arr"][:, :, 1, :])
    max_gauss = conv_func(max_A, ene_grid[nps // 2], ene_grid)
    idx = np.nonzero(max_gauss > 1e-10)[0]
    idx_w_g = max(idx) - min(idx)
    upper = nps - idx_w_g//2
    lower = idx_w_g//2



    conv_data = np.zeros((nt, nc, nps), dtype=float)
    for it in range(nt):

        done = width_bar * (it + 1) // nt
        sys.stdout.write("\r  Progress: [" + "=" * done + " " * (width_bar - done) + "] %3i%%" % (done * 100 // width_bar))

        for col in range(nc):
            conv_it_col = conv_data[it, col, :]
            for idx in range(nf):
                A = arr[it, idx, 1, col]
                x0 = arr[it, idx, 0, col]
                if np.isnan(A) or np.isnan(x0):
                    continue

                xidx = np.searchsorted(ene_grid, x0)
                if xidx < lower:
                    id1 = 0
                    id2 = xidx + idx_w_g
                elif xidx > upper:
                    id1 = xidx - idx_w_g
                    id2 = nps
                else:
                    id1, id2 = xidx - idx_w_g, xidx + idx_w_g
                conv_it_col[id1:id2] += conv_func(A, x0, ene_grid[id1:id2])


    sys.stdout.write("\n")

    INFOS["colnames"] = [
        f"Conv({x},{y})" for x, y in map(
            lambda c: (INFOS["colnames"][c].split()[2], INFOS["colnames"][c + nc].split()[2]), range(nc)
        )
    ]

    return {"arr": conv_data, "time": all_data["time"], "x_axis": ene_grid}


# ===========================================


def do_t_convolution(INFOS, all_data):
    # set up trange
    width = INFOS["convolute_T"]["function"].fwhm
    arr = all_data["arr"]
    time = all_data["time"]
    tmin = min(time)
    tmax = max(time)
    if not INFOS["convolute_T"]["xrange"]:
        tmin = tmin - 2.0 * width
        tmax = tmax + 2.0 * width
    elif len(INFOS["convolute_T"]["xrange"]) == 2:
        tmin = INFOS["convolute_T"]["xrange"][0]
        tmax = INFOS["convolute_T"]["xrange"][1]
    elif len(INFOS["convolute_T"]["xrange"]) == 1:
        tmin = tmin - INFOS["convolute_T"]["xrange"][0] * width
        tmax = tmax + INFOS["convolute_T"]["xrange"][0] * width

    # do convolution
    nt, nc, nx = arr.shape
    nps = INFOS["convolute_T"]["npoints"]
    t_grid = np.linspace(tmin, tmax, nps)
    conv_t = np.zeros((nc, nx, nps), dtype=float)
    conv_func = INFOS["convolute_T"]["function"].ev

    width_bar = 50

    old_it = -1
    with np.nditer(arr, flags=["multi_index"], op_flags=["readonly"], casting="no") as iter:
        for v in iter:
            it, ic, ix = iter.multi_index
            if old_it != it:
                done = width_bar * (it + 1) // nt
                sys.stdout.write("\r  Progress: [" + "=" * done + " " * (width_bar - done) + "] %3i%%" % (done * 100 // width_bar))
                old_it = it

            conv_t[ic, ix, :] += conv_func(v, time[it], t_grid)

    all_data["arr"] = np.einsum("cxt->tcx", conv_t)
    all_data["time"] = t_grid
    return all_data



# ===========================================


def integrate_T(INFOS, all_data):
    # do cumulative sum for all x values

    calc_arr = all_data["arr"].copy()
    nt, nc, nx = calc_arr.shape

    width_bar = 50
    for it in range(1, nt):
        done = width_bar * (it + 1) // nt
        sys.stdout.write("\r  Progress: [" + "=" * done + " " * (width_bar - done) + "] %3i%%" % (done * 100 // width_bar))
        calc_arr[it, ...] += calc_arr[it - 1, ...]
    sys.stdout.write("\n")

    return {**all_data, "arr": calc_arr}


# ===========================================


def integrate_X(INFOS, all_data):
    # sum up for all x values below a, between a and b, and above b
    xmin = INFOS["integrate_X"]["xrange"][0]
    xmax = INFOS["integrate_X"]["xrange"][1]
    xvalues = np.linspace(-1, 1, 3)
    bins = [xmin, xmax]
    arr = all_data["arr"]
    nt, nc, nx = arr.shape
    idx = np.digitize(arr, bins=bins)
    int_x_arr = np.zeros((nt, nc, 3), dtype=float)
    np.add.at(int_x_arr, idx, arr)

    return {**all_data, "arr": int_x_arr, "x_axis": xvalues}


# ===========================================


def do_y_summation(INFOS, all_data):
    INFOS["colnames"] = ["Time", "X_axis", "Y_sum"]
    return {**all_data, "arr": np.sum(all_data["arr"], axis=1)}


# ===========================================


def type3_to_type2(INFOS, all_data):
    arr = all_data["arr"]
    nt, nc, nx = arr.shape
    return {f'X={x:8f}': {"arr": arr[..., ix], "time": all_data["time"]} for ix, x in enumerate(all_data["X_axis"])}


# ======================================================================================================================


def mean_arith(data: np.ndarray, axis=1):
    if data.shape[axis] == 1:
        return data
    return np.mean(data, axis=axis)


# ======================================== #


def stdev_arith(data: np.ndarray, axis=1):
    return data.std(axis=axis)


# ======================================== #


def mean_geom(data: np.ndarray, axis=1):
    if data.shape[axis] == 1:
        return data
    return np.exp(np.log(data).mean(axis=axis))


# ======================================== #


def stdev_geom(data: np.ndarray, axis=1):
    res = np.full((data.shape[0]), np.nan, dtype=float)
    for i, row in enumerate(data):
        row_arr = row.to_numpy()
        res[i] = stats.gstd(row_arr[row_arr > 0], axis=axis)
    return res


# ======================================== #


# ======================================================================================================================


def write_type1(filename, all_data, INFOS):
    # make header
    longest = max([len(key) for key in all_data])
    string = "#    1 " + " " * (longest - 1) + "2" + " " * 15 + "3"
    for i in range(2 * len(INFOS["colX"])):
        string += "             %3i" % (i + 4)
    # make data string

    with open(filename, "w") as f:
        for i, filekey in enumerate(sorted(all_data)):

            time = all_data[filekey]["time"]
            out = all_data[filekey]["arr"].reshape((time.shape[0], -1))

            if i == 0:
                f.write(string + "\n")
                columns = ["Index", " " * (longest - 8) + "Filename", "Time"] + INFOS["colnames"]
                f.write("#" + "  ".join(map(lambda x: f"{x:>14s}", columns)) + "\n")
            for idx, (t, c) in enumerate(zip(time, out)):
                f.write(f"{i:>15d} {filekey:>14s} {t: 14.8E}  " + "  ".join(map(lambda x: f"{x: 14.8E}", c)) + "\n")
            f.write("\n")
    return


# ======================================================================================================================


def write_type2(filename, all_data, INFOS, write_count=False):
    n_cols = len(INFOS["colnames"]) + 1 + (1 if write_count else 0)
    arr = all_data["arr"][:, :, 0, :].reshape(all_data["arr"].shape[0], -1)
    with open(filename, "w") as f:
        f.write("#" + " ".join(map(lambda i: f"{i + 1:14d} ", range(n_cols))) + "\n")
        # sort multiindex to level!
        # get columns
        column_names = INFOS["colnames"]
        f.write("#" + "  ".join(map(lambda i: f"{i:>14s}", ["Time"] + column_names)) + "\n")
        if write_count:
            count = all_data["count"]
            for idx, (t, c) in enumerate(zip(all_data["time"], arr)):
                f.write(f"{t: 14.8E} " + "  ".join(map(lambda x: f"{x: 14.8E}", c)) + f"  {count[idx]: 14.8E}" + "\n")
        else:
            for idx, (t, c) in enumerate(zip(all_data["time"], arr)):
                f.write(f"{t: 14.8E} " + " ".join(map(lambda x: f"{x: 14.8E}", c)) + "\n")

    return


# ======================================================================================================================


def write_type3(filename, all_data, INFOS):
    # make header
    arr = all_data["arr"]
    nt, nc, nps = arr.shape
    with open(filename, "w") as f:
        f.write(f"#{1:>14d} " + " ".join(map(lambda i: f"{i + 2:>15d}", range(nc + 1))) + "\n")
        f.write(f"#{'Time':>14s} " + " ".join(map(lambda i: f"{i:>15s}", ["X_axis"] + INFOS["colnames"])) + "\n")
        for it, t in enumerate(all_data["time"]):
            for ix, x in enumerate(all_data["x_axis"]):
                c = arr[it, :, ix]
                f.write(f"{t: 14.8E} {x: 14.8E} " + " ".join(map(lambda x: f"{x: 14.8E}", c)) + "\n")
            f.write("\n")

 


# ======================================================================================================================


def readType1(strings):
    print("Type1 cannot be read currently!")
    sys.exit(1)
    # data1={}
    # for line in strings:
    # s=line.split()
    # if len(s)<1:
    # continue
    # key=s[1]
    # values=tuple( [ float(i) for i in s[2:] ] )
    # if not key in data1:
    # data1[key]=[]
    # data1[key].append(values)
    # for key in data:
    # data1[key].sort(key=lambda x: x[0])
    # return data1


# ======================================================================================================================

# ======================================================================================================================


def readType3(strings):
    print("Type3 cannot be read currently!")
    sys.exit(1)


# ======================================================================================================================
# ======================================================================================================================
# ======================================================================================================================


def main():
    """
    python data_collector.py

    This interactive program reads table information from SHARC trajectories.
    """

    displaywelcome()
    open_keystrokes()

    INFOS = get_general()

    print("\n\n{:#^80}\n".format("Full input"))
    for item in INFOS:
        print(item, " " * (25 - len(item)), INFOS[item])
    print("")
    calc = question("Do you want to do the specified analysis?", bool, True)
    print("")

    if calc:
        INFOS = do_calc(INFOS)

    close_keystrokes()


# ======================================================================================================================

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nCtrl+C makes me a sad SHARC ;-(\n")
        quit(0)
