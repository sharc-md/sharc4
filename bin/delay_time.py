#!/usr/bin/env python3

# ******************************************
#
#    SHARC Program Suite
#
#    Copyright (c) 2026 University of Vienna
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
versiondate = datetime.date(2025, 4, 1)


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
    shutil.move("KEYSTROKES.tmp", "KEYSTROKES.delay_time")


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



def get_general():
    """"""

    INFOS = {}

    # ---------------------------------------- Trajectory selection --------------------------------------

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
        exclude = {
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
            "table",
            "driver",
            "rattle",
        }
        for d in dirs:
            for dirpath, dirnames, filenames in os.walk(d, topdown=True):
                dirnames[:] = set(dirnames) - exclude_dirs # from https://stackoverflow.com/questions/19859840/excluding-directories-in-os-walk
                # filenames2 = set(filenames) - exclude    # that is more efficient but only works with exact matches
                # for f in filenames2:
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


        # ---------------------------------------- Choose file 1 --------------------------------------
        print("\n{:-^60}".format("File 1"))

        # choose one of these files
        print("\nPlease give the relative file path of the 'File 1' you want to analyze:")
        while True:
            string = question("File 1 path or index:", str, "0", False)
            try:
                string = allfiles_index[int(string)]
            except ValueError:
                pass
            except KeyError:
                print('I did not understand %s' % string)
                continue
            if string in allfiles:
                INFOS["file1path"] = string
                break
            else:
                print("I did not understand %s" % string)

        # make list of files
        allfiles1 = []
        for d in dirs:
            f = os.path.join(d, INFOS["file1path"])
            if os.path.isfile(f):
                allfiles1.append(f)
        INFOS["allfiles1"] = allfiles1


        # ---------------------------------------- Choose file 2 --------------------------------------
        print("\n{:-^60}".format("File 2"))

        # choose one of these files
        print("\nPlease give the relative file path of the 'File 2' you want to analyze:")
        while True:
            string = question("File 2 path or index:", str, "0", False)
            try:
                string = allfiles_index[int(string)]
            except ValueError:
                pass
            except KeyError:
                print('I did not understand %s' % string)
                continue
            if string in allfiles:
                INFOS["file2path"] = string
                break
            else:
                print("I did not understand %s" % string)

        # make list of files
        allfiles2 = []
        for d in dirs:
            f = os.path.join(d, INFOS["file2path"])
            if os.path.isfile(f):
                allfiles2.append(f)
        INFOS["allfiles2"] = allfiles2

    else:
        # ---------------------------------------- Choose file 1 --------------------------------------
        print("\n{:-^60}".format("File 1"))

        print("\nPlease give the relative file path of the 'File 1' you want to analyze:")
        while True:
            INFOS["file1path"] = question("File 1 path:", str, ".", False)
            absent = []
            allfiles1 = []
            print("Checking if file exists in directories...")
            for i, d in enumerate(dirs):
                done = 50 * (i + 1) // len(dirs)
                sys.stdout.write("\r  Progress: [" + "=" * done + " " * (50 - done) + "] %3i%%" % (done * 100 / 50))
                f = os.path.join(d, INFOS["file1path"])
                if os.path.isfile(f):
                    allfiles1.append(f)
                else:
                    absent.append(d)
            sys.stdout.write("\n")
            if len(absent) != 0:
                print(f"\n{INFOS['file1path']} is absent in {absent}")
                if question("Continue anyway?", bool, False):
                    break
            else:
                break

        INFOS["allfiles1"] = allfiles1

        # ---------------------------------------- Choose file 2 --------------------------------------
        print("\n{:-^60}".format("File 2"))

        print("\nPlease give the relative file path of the 'File 2' you want to analyze:")
        while True:
            INFOS["file2path"] = question("File 2 path:", str, ".", False)
            absent = []
            allfiles2 = []
            print("Checking if file exists in directories...")
            for i, d in enumerate(dirs):
                done = 50 * (i + 1) // len(dirs)
                sys.stdout.write("\r  Progress: [" + "=" * done + " " * (50 - done) + "] %3i%%" % (done * 100 / 50))
                f = os.path.join(d, INFOS["file2path"])
                if os.path.isfile(f):
                    allfiles2.append(f)
                else:
                    absent.append(d)
            sys.stdout.write("\n")
            if len(absent) != 0:
                print(f"\n{INFOS['file2path']} is absent in {absent}")
                if question("Continue anyway?", bool, False):
                    break
            else:
                break

        INFOS["allfiles2"] = allfiles2



    # print(INFOS["allfiles1"])
    # print(INFOS["allfiles2"])
    # ---------------------------------------- Columns --------------------------------------

    # get number of columns from file 1
    ncol1 = None
    for filename in allfiles1:
        testfile = open(filename, "r")
        for line in testfile:
            if "#" not in line:
                ncol1 = len(line.split())
                break
        testfile.close()
        if ncol1 is not None:
            break
    print("Number of columns in the file:   %i" % (ncol1))
    INFOS["ncol1"] = ncol1

    # get number of columns from file 2
    ncol2 = None
    for filename in allfiles2:
        testfile = open(filename, "r")
        for line in testfile:
            if "#" not in line:
                ncol2 = len(line.split())
                break
        testfile.close()
        if ncol2 is not None:
            break
    print("Number of columns in the file:   %i" % (ncol2))
    INFOS["ncol2"] = ncol2



    # ---------------------------------------- Assign Column File 1 --------------------------------------

    print("\n" + "{:-^60}".format("Data columns File 1") + "\n")

    # select columns
    print("\nPlease select the data columns for the analysis:")
    print("For T column: \n  only enter one (positive) column index. \n  If 0, the line number will be used instead.")
    print(
        "For X column: \n  enter one column index."
    )
    print("")
    while True:
        INFOS["colT1"] = question("T column (time) for File 1:", int, [1])[0]
        if 0 <= INFOS["colT1"] <= ncol1:
            # 0:   use line number (neglecting commented or too short lines)
            # 1-n: use that line for time data
            break
        else:
            print("Please enter a number between 0 and %i!" % ncol1)
    while True:
        INFOS["colX1"] = question("X column (value) for File 1:", int, [2])[0]
        if 0 <= INFOS["colX1"] <= ncol1:
            break
        else:
            print("Please enter a number between 0 and %i!" % ncol1)

    print("Selected columns:")
    print("T: %s     X: %s\n" % (str(INFOS["colT1"]), str(INFOS["colX1"])))




    # ---------------------------------------- Assign Column File 2 --------------------------------------

    print("\n" + "{:-^60}".format("Data columns File 2") + "\n")

    # select columns
    print("\nPlease select the data columns for the analysis:")
    print("For T column: \n  only enter one (positive) column index. \n  If 0, the line number will be used instead.")
    print(
        "For X column: \n  enter one column index."
    )
    print("")
    while True:
        INFOS["colT2"] = question("T column (time) for File 2:", int, [1])[0]
        if 0 <= INFOS["colT2"] <= ncol2:
            # 0:   use line number (neglecting commented or too short lines)
            # 1-n: use that line for time data
            break
        else:
            print("Please enter a number between 0 and %i!" % ncol2)
    while True:
        INFOS["colX2"] = question("X column (value) for File 2:", int, [2])[0]
        if 0 <= INFOS["colX2"] <= ncol2:
            break
        else:
            print("Please enter a number between 0 and %i!" % ncol2)

    print("Selected columns:")
    print("T: %s     X: %s\n" % (str(INFOS["colT2"]), str(INFOS["colX2"])))


    # ---------------------------------------- Assign 1 Threshold --------------------------------------

    print("\n" + "{:-^60}".format("Event detection for File 1") + "\n")

    print("\nThe event for File 1 will be detected through detecting a crossing of a numerical threshold (either from below or above)")
    print("Enter the corresponding threshold below")
    print("")

    INFOS["thres1"] = question("Threshold for File 1", float, [0.5])[0]


    # ---------------------------------------- Assign 2 Threshold --------------------------------------

    print("\n" + "{:-^60}".format("Event detection for File 2") + "\n")

    print("\nThe event for File 2 will be detected through detecting a crossing of a numerical threshold (either from below or above)")
    print("Enter the corresponding threshold below")
    print("")

    INFOS["thres2"] = question("Threshold for File 2", float, [0.5])[0]


    # ---------------------------------------- Event detection method --------------------------------------

    INFOS["options"] = {"mode": "simple"}
    print("\n" + "{:-^60}".format("Event detection Method") + "\n")

    # print list
    print("\nChoose an option for the event detection.")
    methods = {"simple": {}, "persistence": {"persistence_time": (float, [20.], "in fs")}}
    print("Possible methods:")
    for key in methods:
        options = methods[key]
        options_string = ','.join(options) if options else "none"
        print("- '%s' (options: %s)" % (key, options_string))

    # ask for method
    while True:
        key = question("Method", str, "simple", autocomplete=False)
        if not key in methods:
            print("Not a valid method")
            continue
        break
    INFOS["options"] = {"mode": key}

    # ask for options
    for option in methods[key]:
        parts = methods[key][option]
        if len(parts) > 2:
            a = question("Set option '%s' (%s)" % (option, ','.join(parts[2:])), parts[0], parts[1])
        else:
            a = question("Set option '%s'" % option, parts[0], parts[1])
        if parts[0] is int or parts[0] is float:
            a = a[0]
        INFOS["options"][option] = a

    return INFOS

# ======================================================================================================================
# ======================================================================================================================
# ======================================================================================================================


def load_column(file, colT, colX, ncol):
    """
    Loads time and value columns from a trajectory file using NumPy.
    
    Parameters:
        file (str): Path to the file.
        colT (int): Index (0-based) of the time column.
        colX (int): Index (0-based) of the value column.
        ncol (int): Expected number of columns in the file.
    
    Returns:
        t (np.ndarray): Time values.
        x (np.ndarray): Parameter values.
    """
    try:
        # Use genfromtxt to skip commented lines and handle missing data gracefully
        data = np.genfromtxt(
            file,
            comments="#",
            usecols=(colT-1, colX-1),
            invalid_raise=False
        )
        if data.ndim == 1:  # Happens when only one valid line
            data = np.array([data])

        t = data[:, 0]
        x = data[:, 1]
        
        return t, x

    except Exception as e:
        raise RuntimeError(f"Failed to read file {file}: {e}")



# ======================================================================================================================
# ======================================================================================================================
# ======================================================================================================================

def detect_event(Tarray, Xarray, threshold, options):
    """
    Detects the first event where X crosses the threshold.
    
    Parameters:
        Tarray (np.ndarray): Time array (can be non-uniform).
        Xarray (np.ndarray): Value array.
        threshold (float): Threshold for event detection.
        options (dict): Dictionary with options (currently supports 'mode').

    Returns:
        time (float or np.nan): Estimated time of crossing (midpoint between before/after).
        direction (str or None): 'up' if crossing from below, 'down' if crossing from above, None if not found.
    """
    mode = options["mode"]

    if mode == "simple":
        for i in range(1, len(Xarray)):
            x_prev, x_curr = Xarray[i - 1], Xarray[i]
            t_prev, t_curr = Tarray[i - 1], Tarray[i]

            # Check if crossing occurs
            if x_prev < threshold <= x_curr:
                direction = +1
                time = (t_prev + t_curr) / 2
                return time, direction
            elif x_prev > threshold >= x_curr:
                direction = -1
                time = (t_prev + t_curr) / 2
                return time, direction
    
    elif mode == "persistence":
        persistence_time = options.get("persistence_time", 0.0)
        candidate_start_idx = None
        candidate_direction = 0

        for i in range(1, len(Xarray)):
            x_prev, x_curr = Xarray[i - 1], Xarray[i]
            t_prev, t_curr = Tarray[i - 1], Tarray[i]

            # Check for tentative crossing up
            if candidate_start_idx is None:
                if x_prev < threshold <= x_curr:
                    candidate_start_idx = i - 1
                    candidate_direction = +1
                elif x_prev > threshold >= x_curr:
                    candidate_start_idx = i - 1
                    candidate_direction = -1

            if candidate_start_idx is not None:
                # Check if signal stayed on correct side of threshold
                # For upward crossing, X must stay >= threshold
                # For downward crossing, X must stay <= threshold

                if candidate_direction == +1 and x_curr < threshold:
                    # Fell back below threshold before persistence time reached
                    candidate_start_idx = None
                    candidate_direction = 0
                elif candidate_direction == -1 and x_curr > threshold:
                    # Rose back above threshold before persistence time reached
                    candidate_start_idx = None
                    candidate_direction = 0
                else:
                    # Check elapsed time since candidate start
                    elapsed = t_curr - Tarray[candidate_start_idx]
                    if elapsed >= persistence_time:
                        # Event confirmed, return time of first crossing (average between candidate_start_idx and candidate_start_idx+1)
                        time = (Tarray[candidate_start_idx] + Tarray[candidate_start_idx + 1]) / 2
                        return time, candidate_direction
    else:
        raise NotImplementedError(f"Mode '{mode}' is not implemented yet.")

    # No crossing found
    return float('nan'), 0




# ======================================================================================================================
# ======================================================================================================================
# ======================================================================================================================



def do_calc(INFOS):

    # options = {"mode": "simple"}
    # options = {"mode": "persistence", "persistence_time": 20.0}
    options = INFOS["options"]

    # initialize results
    results = []
    string = f"#{'i':>4} {'t1/fs':>10} {'t2/fs':>10} {'dt/fs':>10} {'dir1':>5} {'dir2':>5} {'File 1':>30} {r'File 2':>30}"
    print(string)
    outfile = open("delay_times.out","w")
    outfile.write(string+'\n')
    outfile.flush() 

    # loop over trajectories
    for i, (file1, file2) in enumerate(zip(INFOS["allfiles1"], INFOS["allfiles2"])):

        try:
            # load
            T1, X1 = load_column(file1, INFOS["colT1"], INFOS["colX1"], INFOS["ncol1"])
            T2, X2 = load_column(file2, INFOS["colT2"], INFOS["colX2"], INFOS["ncol2"])

            # detect events
            t1, direction1 = detect_event(T1, X1, INFOS["thres1"], options )
            t2, direction2 = detect_event(T2, X2, INFOS["thres2"], options )

            # store results
            results.append({
                "traj": i,
                "file1": file1,
                "file2": file2,
                "t1": t1,
                "t2": t2,
                "dt": t2 - t1 if not (np.isnan(t1) or np.isnan(t2)) else np.nan,
                "dir1": direction1,
                "dir2": direction2
            })

            # print
            r=results[-1]
            string = f" {r['traj']:4d} {r['t1']:10.2f} {r['t2']:10.2f} {r['dt']:10.2f} {r['dir1']:5d} {r['dir2']:5d} {r['file1']:30s} {r['file2']:30s}"
            print(string)
            outfile.write(string+'\n')
            if i%100 == 0:
                outfile.flush()
        except KeyboardInterrupt:
            print("Skipping rest of the analysis, proceeding with print out...")
            break
    print('Output written to "delay_times.out"')
    outfile.close()

    # histogram
    valid_dt = np.array([r["dt"] for r in results if not np.isnan(r["dt"])])
    def auto_bin_width(data):
        # Freedman–Diaconis rule
        q75, q25 = np.percentile(data, [75 ,25])
        iqr = q75 - q25
        bin_width = 2 * iqr * len(data) ** (-1/3)
        return bin_width if bin_width > 0 else (np.max(data) - np.min(data)) / 10

    bin_width = auto_bin_width(valid_dt)
    bins = np.arange(np.min(valid_dt), np.max(valid_dt) + bin_width, bin_width)

    hist, bin_edges = np.histogram(valid_dt, bins=bins)
    histfile = open("delay_histogram.out",'w')
    for count, left, right in zip(hist, bin_edges[:-1], bin_edges[1:]):
        string = f"{left:10.2f} - {right:10.2f} fs: {count}"
        print(string)
        histfile.write(string+'\n')
    print('Output written to "delay_histogram.out"')
    histfile.close()




    return INFOS



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












