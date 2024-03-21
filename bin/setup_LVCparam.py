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

# Interactive script for the setup of displacement calculations for SHARC
#
# usage: python setup_LVCparam.py

import readline
import datetime
import sys
import re
import os
import stat
import shutil
from optparse import OptionParser

# for ordered dictionary output
from collections import OrderedDict

# to easily write/read data structure to/from file
# import pickle
import json
import factory
from logger import log
from constants import IToMult, U_TO_AMU
from SHARC_INTERFACE import SHARC_INTERFACE

# ======================================================================================================================
# ======================================================================================================================
# ======================================================================================================================
# ======================================================================================================================
# ======================================================================================================================
# ======================================================================================================================


version = "4.0"
versionneeded = [0.2, 1.0, 2.0, 2.1, float(version)]
versiondate = datetime.date(2024, 1, 1)

# ======================================================================================================================


def make_directory(displacement_dir):
    """Creates a directory"""

    if os.path.isfile(displacement_dir):
        print("\nWARNING: %s is a file!" % (displacement_dir))
        return -1

    if os.path.isdir(displacement_dir):
        if len(os.listdir(displacement_dir)) == 0:
            return 0
        else:
            print("\nWARNING: %s/ is not empty!" % (displacement_dir))

            if "overwrite" not in globals():
                global overwrite
                overwrite = question("Do you want to overwrite files in this folder? ", bool, False)

            if overwrite:
                return 0
            else:
                return -1
    else:
        try:
            os.mkdir(displacement_dir)
        except OSError:
            print("\nWARNING: %s cannot be created!" % (displacement_dir))
            return -1
        return 0


# ======================================================================================================================


def displaywelcome():
    print("Script for setup of LVC parametrization started...\n")
    string = "\n"
    string += "  " + "=" * 80 + "\n"
    string += "||" + "{:^80}".format("") + "||\n"
    string += "||" + "{:^80}".format("LVC parametrization for SHARC dynamics") + "||\n"
    string += "||" + "{:^80}".format("") + "||\n"
    string += "||" + "{:^80}".format("Author: Simon Kropf, Sebastian Mai, Severin Polonius") + "||\n"
    string += "||" + "{:^80}".format("") + "||\n"
    string += "||" + "{:^80}".format("Version:" + version) + "||\n"
    string += "||" + "{:^80}".format(versiondate.strftime("%d.%m.%y")) + "||\n"
    string += "||" + "{:^80}".format("") + "||\n"
    string += "  " + "=" * 80 + "\n\n"
    string += (
        "This script automatizes the setup of excited-state calculations\nin order to parametrize LVC models for SHARC dynamics."
    )
    print(string)


def open_keystrokes():
    global KEYSTROKES
    KEYSTROKES = open("KEYSTROKES.tmp", "w")


def close_keystrokes():
    KEYSTROKES.close()
    shutil.move("KEYSTROKES.tmp", "KEYSTROKES.setup_LVCparam")


def question(question, typefunc, default=None, autocomplete=True, ranges=False):
    if typefunc == int or typefunc == float:
        if default is not None and not isinstance(default, list):
            print("Default to int or float question must be list!")
            quit(1)
    if typefunc == str and autocomplete:
        readline.set_completer_delims(" \t\n;")
        readline.parse_and_bind("tab: complete")  # activate autocomplete
    else:
        readline.parse_and_bind("tab: ")  # deactivate autocomplete

    while True:
        s = question
        if default is not None:
            if typefunc == bool or typefunc == str:
                s += " [%s]" % (str(default))
            elif typefunc == int or typefunc == float:
                s += " ["
                for i in default:
                    s += str(i) + " "
                s = s[:-1] + "]"
        if typefunc == str and autocomplete:
            s += " (autocomplete enabled)"
        if typefunc == int and ranges:
            s += " (range comprehension enabled)"
        s += " "

        line = input(s)
        line = re.sub("#.*$", "", line).strip()
        if not typefunc == str:
            line = line.lower()

        if line == "" or line == "\n":
            if default is not None:
                KEYSTROKES.write(line + " " * (40 - len(line)) + " #" + s + "\n")
                return default
            else:
                continue

        if typefunc == bool:
            posresponse = ["y", "yes", "true", "t", "ja", "si", "yea", "yeah", "aye", "sure", "definitely"]
            negresponse = ["n", "no", "false", "f", "nein", "nope"]
            if line in posresponse:
                KEYSTROKES.write(line + " " * (40 - len(line)) + " #" + s + "\n")
                return True
            elif line in negresponse:
                KEYSTROKES.write(line + " " * (40 - len(line)) + " #" + s + "\n")
                return False
            else:
                print("I didn't understand you.")
                continue

        if typefunc == str:
            KEYSTROKES.write(line + " " * (40 - len(line)) + " #" + s + "\n")
            return line

        if typefunc == float:
            # float will be returned as a list
            f = line.split()
            try:
                for i in range(len(f)):
                    f[i] = typefunc(f[i])
                KEYSTROKES.write(line + " " * (40 - len(line)) + " #" + s + "\n")
                return f
            except ValueError:
                print("Please enter floats!")
                continue

        if typefunc == int:
            # int will be returned as a list
            f = line.split()
            out = []
            try:
                for i in f:
                    if ranges and "~" in i:
                        q = i.split("~")
                        for j in range(int(q[0]), int(q[1]) + 1):
                            out.append(j)
                    else:
                        out.append(int(i))
                KEYSTROKES.write(line + " " * (40 - len(line)) + " #" + s + "\n")
                return out
            except ValueError:
                if ranges:
                    print('Please enter integers or ranges of integers (e.g. "-3~-1  2  5~7")!')
                else:
                    print("Please enter integers!")
                continue


def print_INFOS(INFOS):
    print("\n" + "{:#^60}".format("Full input") + "\n")

    for item in INFOS:
        i = 0  # counter for very long lists we do not want full output)

        if isinstance(INFOS[item], list) or isinstance(INFOS[item], tuple):
            first = True
            for elem in INFOS[item]:
                if i >= 10:
                    break
                if first:
                    print(item, " " * (25 - len(item) - 1), elem)
                    first = False
                else:
                    print(" " * 25, elem)

                i += 1

        elif isinstance(INFOS[item], dict):
            first = True
            for k, v in INFOS[item].items():
                if i >= 10:
                    break
                if first:
                    print(item, " " * (25 - len(item)) + "%s: %s" % (k, v))
                    first = False
                else:
                    print(" " * 25 + " %s: %s" % (k, v))

                i += 1
        else:
            print(item, " " * (25 - len(item) - 1), INFOS[item])

        if i >= 10:
            print(" " * 25, ".")
            print(" " * 25, "(" + str(len(INFOS[item]) - i) + " more)")
            print(" " * 25, ".")
    return


def reduce_big_list_to_short_str(big_list):
    """
    Takes possibly big lists of numbers (e. g. list of all normal modes)
    and reduces them to short string for clean output to user

    e. g.: [7 8 9 12 13 14 17 20 21 22 23 25 26 28 29 30] => '[7~9 12~14 17 20~23 25 26 28~30]'

    returns shortened list string
    """

    # if empty return [None]
    if not big_list:
        return [None]

    # start list_string, sort bit_list, set vars
    short_list_str = "("
    big_list = sorted(big_list)
    i_start, i_current, i_lastadded = 0, 0, 0

    # while index is within big_list
    while i_current < len(big_list) - 1:
        # check if next element is within range & continue
        if big_list[i_current] + 1 == big_list[i_current + 1]:
            i_current += 1
            continue
        # range ended - create shortened string
        else:
            # no range just one number alone
            if i_current == i_start:
                short_list_str += "%i " % big_list[i_current]
            # range detected - shorten it
            else:
                # special case for 2 neighbour numbers
                if big_list[i_start] + 1 == big_list[i_current]:
                    short_list_str += "%i %i " % (big_list[i_start], big_list[i_current])
                else:
                    short_list_str += "%i~%i " % (big_list[i_start], big_list[i_current])

            # set vars accordingly for next run
            i_current += 1
            i_start = i_current
            i_lastadded = i_current

    # code above will always leave out last range/number - add it here (that's why we need i_lastadded)
    if i_lastadded != i_current:
        # special case again for 2 neighbouring numbers
        if big_list[i_start] + 1 == big_list[i_current]:
            short_list_str += "%i %i" % (big_list[i_start], big_list[i_current])
        else:
            short_list_str += "%i~%i" % (big_list[i_start], big_list[i_current])
    else:
        short_list_str += str(big_list[i_current])

    # close bracket & return
    short_list_str += ")"
    return short_list_str


def reduce_displacement_dictionary_to_output_str(big_dictionary):
    """
    used for shortening displacement dict therefore can never be empty

    e. g.
    for:
        OrderedDict({ 7: 0.05,  8: 0.05,  9: 0.05, 12: 0.05, 13: 0.05,
                     14: 0.04, 15: 0.05, 16: 0.05, 19: 0.03, 21: 0.03,
                     22: 0.03, 23: 0.03, 27: 0.01 })

    returns:
        [7~9 12 13 15 16]: 0.05
        [14]: 0.04
        [19 21~23]: 0.03
        [27]: 0.01

    returns string with ranges for same displacements
    """
    output_str = ""

    # getting all different displacements
    displacement_list = []
    for normal_mode, displacement in big_dictionary.items():
        if displacement not in displacement_list:
            displacement_list.append(displacement)

    # running through all different displacements
    normal_mode_list_big = []
    for displacement in displacement_list:
        # adding all normal modes with same displacement to list
        for normal_mode, disp in big_dictionary.items():
            if displacement == disp:
                normal_mode_list_big.append(normal_mode)

        # reduce all normal modes with same displacement and add it to nice output
        output_str += "%s: %g\n" % (reduce_big_list_to_short_str(normal_mode_list_big), displacement)
        normal_mode_list_big = []

    return output_str


# ======================================================================================================================
# ======================================================================================================================
# ======================================================================================================================
# ======================================================================================================================
# ======================================================================================================================
# ======================================================================================================================
def get_interface() -> SHARC_INTERFACE:
    "asks for interface and instantiates it"
    Interfaces = factory.get_available_interfaces()
    log.info("{:-^60}".format("Choose the quantum chemistry interface"))
    log.info("\nPlease specify the quantum chemistry interface (enter any of the following numbers):")
    possible_numbers = []
    for i, (name, interface) in enumerate(Interfaces):
        if type(interface) == str:
            log.info("%i\t%s: %s" % (i, name, interface))
        else:
            log.info("%i\t%s: %s" % (i, name, interface.description()))
            possible_numbers.append(i)
    log.info("")
    while True:
        num = question("Interface number:", int)[0]
        if num in possible_numbers:
            break
        else:
            log.info("Please input one of the following: %s!" % (possible_numbers))
    log.info("")
    return Interfaces[num][1]


def get_V0_and_states():
    """
    This routine questions from the user some general information:
    - V0.txt file
    - # of displacements
    - one/two-sided derivation
    - number of states
    - QM package
    - SOCs

    returns INFOS dictionary
    """
    INFOS = {}

    ## -------------------- getting & reading V0.txt -------------------- ##
    print("{:-^60}".format("V0.txt file") + "\n")

    # check if V0 exists
    v0file = "V0.txt"
    try:
        if os.path.isfile(v0file):
            print('Ground-state file "V0.txt" detected. Do you want to use this?')
            if not question('Use file "V0.txt"?', bool, True):
                raise IOError
        else:
            raise IOError
    except IOError:
        print("\nIf you do not have an ground-state file, prepare one with 'wigner.py -l <molden-file>'!\n")
        print("Please enter the filename of the ground-state file.")
        while True:
            v0file = question("Ground-state filename:", str, "V0.txt")
            v0file = os.path.expanduser(os.path.expandvars(v0file))
            if os.path.isdir(v0file):
                print("Is a directory: %s" % (v0file))
                continue
            if not os.path.isfile(v0file):
                print("File does not exist: %s" % (v0file))
                continue
            if os.path.isfile(v0file):
                break

    INFOS["v0f"] = os.path.abspath(v0file)

    # read V0.txt
    with open(v0file, "r") as v0f:
        content = v0f.readlines()
        v0f.close()

    INFOS = read_V0(INFOS, content)

    # output to user
    print(
        '\nFile "%s" contains %i atoms and we will use %i frequencies/normal modes.\n(others are zero)\n'
        % (v0file, len(INFOS["atoms"]), len(INFOS["frequencies"]))
    )

    ## -------------------- number of states -------------------- ##
    print("{:-^60}".format("Number of states"))
    print(
        "\nPlease enter the number of states as a list of integers\ne.g. 3 0 3 for three singlets, zero doublets and three triplets."
    )

    while True:
        states = question("Number of states:", int)

        if not states:
            continue
        if any(i < 0 for i in states):
            print("Number of states must be positive!")
            continue
        break

    nstates = 0
    for mult, i in enumerate(states):
        nstates += (mult + 1) * i

    # output to user
    print("\nNumber of states: %s" % (states))
    print("Total number of states: %i\n" % (nstates))

    # saving input
    INFOS["states"] = states
    INFOS["nstates"] = nstates

    return INFOS


def get_setup_info(INFOS, interface: SHARC_INTERFACE):
    features = interface.get_features(KEYSTROKES)
    states = INFOS["states"]

    INFOS["needed_requests"] = []

    ## -------------------- Setup SOCs -------------------- ##
    print("{:-^60}".format("Spin-orbit couplings (SOCs)") + "\n")
    if len(states) > 1:
        if "soc" in features:
            print("Do you want to compute spin-orbit couplings?\n")
            soc = question("Spin-Orbit calculation?", bool, True)
            if soc:
                print("Will calculate spin-orbit matrix.")
        else:
            print("Interface cannot provide SOCs: not calculating spin-orbit matrix.")
            soc = False
    else:
        print("Only singlets specified: not calculating spin-orbit matrix.")
        soc = False
    print("")

    # save input
    INFOS["soc"] = soc
    if INFOS["soc"]:
        INFOS["needed_requests"].append("soc")

    ## -------------------- whether to do gradients or numerical -------------------- ##
    print("{:-^60}".format("Analytical gradients") + "\n")

    INFOS["ana_grad"] = question("Do you want to use analytical gradients for kappa terms?", bool, True)

    print("\nAnalytical gradients for kappas: %r\n" % INFOS["ana_grad"])

    INFOS["gammas"] = False
    if question("Do you want to calculate second order terms (gammas)?", bool, False):
        if question("Gammas from five-point stencils (two more displacements at 2*h per normal mode)?", bool, False):
            INFOS["gammas"] = "five-point stencil"
        elif question("Gammas from hessian of diabatized gradients at displacements?", bool, False):
            INFOS["gammas"] = "hessian from diabatized gradients"
        elif question("Gammas from hessian from second-order central of energies?", bool, False):
            INFOS["gammas"] = "second order central"
        else:
            print("\nNot including gamma values\n")
    if INFOS["gammas"]:
        if question("Do you want to calculate gamma terms ony for certain states?", bool, default=False):
            INFOS["gamma_selected_states"] = {}
            for imult, n in enumerate(INFOS["states"]):
                if n == 0:
                    continue
                states = reduce_big_list_to_short_str(list(range(n)))
                selected = question(f"Which states do you want to include for {IToMult[imult + 1]}s {states}?", int, ranges=True)
                if len(selected) == 0:
                    selected = INFOS["states"]
                INFOS["gamma_selected_states"][str(imult)] = selected

            print("\nGamma terms will be calculated for the following states:", INFOS["gamma_selected_states"])

    ## -------------------- whether to do gradients or numerical -------------------- ##
    print("{:-^60}".format("Analytical nonadiabatic coupling vectors") + "\n")
    if "nacdr" in features:
        INFOS["ana_nac"] = question("Do you want to use analytical nonadiabatic coupling vectors for lambda terms?", bool, False)
    else:
        INFOS["ana_nac"] = False

    print("Do you want to use analytical nonadiabatic coupling vectors for lambdas: %r\n" % INFOS["ana_nac"])
    if INFOS["ana_nac"]:
        INFOS["needed_requests"].append("nacdr")

    ## -------------------- Whether to do overlaps -------------------- ##
    if (not INFOS["ana_grad"]) or (not INFOS["ana_nac"]):
        do_overlaps = True
    else:
        do_overlaps = False

    if do_overlaps:
        if "overlap" not in features:
            print("Interface cannot provide overlaps, numerical mode not possible!")
            sys.exit(1)

        INFOS["do_overlaps"] = True
        INFOS["needed_requests"].append("overlap")
    else:
        INFOS["do_overlaps"] = False

    ## -------------------- select normal modes -------------------- ##
    print("{:-^60}".format("Normal modes") + "\n")

    if not question("Do you want to make LVC parameters for all normal modes?", bool, True):
        not_using = question(
            "Which of the normal modes do you NOT want to use? %s" % (reduce_big_list_to_short_str(INFOS["normal_modes"].keys())),
            int,
            ranges=True,
        )

        # delete selected normal modes & jump over non-existing ones
        for k in not_using:
            if not (k in INFOS["normal_modes"].keys()):
                print("Normal mode %i doesn't exist. Going on with the rest." % (k))
            else:
                del INFOS["normal_modes"][k]

    # output to user
    print("\nWe will use the following normal modes: %s\n" % (reduce_big_list_to_short_str(INFOS["normal_modes"].keys())))

    if INFOS["gammas"]:
        all_modes = INFOS["normal_modes"]
        gamma_modes = question(
            f"For which normal modes do you want to calculate gamma terms {reduce_big_list_to_short_str(all_modes)}",
            int,
            ranges=True,
        )
        if len(gamma_modes) == 0:
            gamma_modes = all_modes
        INFOS["gamma_normal_modes"] = list(map(str, gamma_modes))
        print("\nThe following modes will be used for gamma calculations", reduce_big_list_to_short_str(gamma_modes))

    ## ----------------------------------------------------------------- ##
    ## -------------------- Infos for displacements -------------------- ##
    if INFOS["do_overlaps"]:
        ## -------------------- reading displacements -------------------- ##
        print("{:-^60}".format("Displacements") + "\n")
        displacement_magnitudes = OrderedDict(sorted({i: 0.05 for i in INFOS["normal_modes"].keys()}.items(), key=lambda t: t[0]))

        # getting user input
        if question("Do you want to use other displacements than the default of [0.05]?", bool, False):
            while True:
                print("")
                displacement = abs(question("Displacement magnitude:", float, [0.05])[0])

                if displacement == 0:
                    print("Only non-zero displacement magnitudes are allowed.")
                    continue

                indexes = question(
                    "For which normal modes do you want to use the displacement of %g? %s:"
                    % (displacement, reduce_big_list_to_short_str(INFOS["normal_modes"].keys())),
                    int,
                    ranges=True,
                )

                # check if normal mode of input exists and add it to displacement_magnitudes dict
                for k in indexes:
                    if k in INFOS["normal_modes"].keys():
                        displacement_magnitudes[k] = displacement
                    else:
                        print("Normal mode %i doesn't exist. Therefore can't set displacement. Going on with the rest." % (k))

                print("")
                if not question("Do you want to define more displacements?", bool, False):
                    break

        # saving input
        INFOS["displacement_magnitudes"] = displacement_magnitudes

        # output to user
        print(
            "\nScript will use displacement magnitudes of:\n\n%s \n"
            % (reduce_displacement_dictionary_to_output_str(displacement_magnitudes))
        )

        ## -------------------- ignore problematic states -------------------- ##
        if not (INFOS["ana_grad"] and INFOS["ana_nac"]):
            print("{:-^60}".format("Intruder states"))

            print(
                """
  Intruder states can be detected by small overlap matrix elements.
  Affected numerical kappa/lambda terms will be ignored and not written to the parameter file.
"""
            )
            INFOS["ignore_problematic_states"] = not question("Do you want to check for intruder states?", bool, True)

            print("\nIgnore problematic states: %r\n" % INFOS["ignore_problematic_states"])

        else:
            INFOS["ignore_problematic_states"] = False

        ## -------------------- one/two-sided derivation -------------------- ##
        print("{:-^60}".format("One-/Two-sided derivation"))
        print("\nOne-/Two-sided derivation of normal modes.")

        # getting user input
        one_sided_derivations = {}

        derivations = question(
            "Choose for which normal modes you want to use one-sided derivation %s:"
            % (reduce_big_list_to_short_str(INFOS["normal_modes"])),
            int,
            [None],
            ranges=True,
        )

        # check if normal mode of input exists and add it to one_sided_derivation dict
        if derivations != [None]:
            for k in derivations:
                if k in INFOS["normal_modes"].keys():
                    one_sided_derivations[k] = True
                else:
                    print("Normal mode %i doesn't exist. Therefore can't set one-sided derivation. Going on with the rest." % (k))

        # saving input
        INFOS["one-sided_derivations"] = OrderedDict(sorted(one_sided_derivations.items(), key=lambda t: t[0]))

        # output to user
        print("\nOne-sided derivation will be used on: %s\n" % (reduce_big_list_to_short_str(one_sided_derivations.keys())))

    ## ----------------------Multipolar fit ---------------------------- ##
    INFOS["multipolar_fit"] = question(
        "Do you want to fit an atomwise multipolar density representation for each state?", bool, False
    )

    ## -------------------- Calculate displacements -------------------- ##
    INFOS = calculate_displacements(INFOS)

    ## -------------------- Create path information -------------------- ##
    INFOS["result_path"] = "DSPL_RESULTS"
    INFOS["paths"] = {}
    INFOS["paths"]["000_eq"] = "DSPL_%03i_eq" % (0)
    if INFOS["do_overlaps"]:
        INFOS["paths"].update({k: "DSPL_%s_%s" % tuple(k.split("_")) for k in INFOS["displacements"].keys()})
    # sort dict
    INFOS["paths"] = OrderedDict(sorted(INFOS["paths"].items(), key=lambda t: t[0]))

    INFOS["cwd"] = os.getcwd()
    return INFOS


# ======================================================================================================================


def read_V0(INFOS, content):
    """
    reads V0.txt and puts the data into the INFOS dictionary

    returns INFOS dictionary
    """
    # current header
    header = ""
    # set headers of V0.txt file
    headers = {"geo": "Geometry\n", "freq": "Frequencies\n", "mwnmodes": "Mass-weighted normal modes\n"}
    # init list/dicts
    INFOS["atoms"], frequencies, normal_modes = [], {}, {}

    for line in content:
        # check if within atom lines
        if header == headers["geo"] and line != headers["freq"]:
            elements = line.strip().split()

            # add atom
            INFOS["atoms"].append(
                {
                    "atom": elements[0],
                    "e-": int(float(elements[1])),
                    "coords [bohr]": (float(elements[2]), float(elements[3]), float(elements[4])),
                    "mass (amu)": float(elements[5]) * U_TO_AMU,
                }
            )

        # check if within frequencies and add them
        if header == headers["freq"] and line != headers["mwnmodes"]:
            frequencies = {i + 1: freq for i, freq in zip(range(len(line.split())), map(float, line.strip().split()))}

            # init number of normal_modes with lists, as to be able to append them later
            for i in range(len(frequencies)):
                normal_modes[i + 1] = []

        # within normal modes
        if header == headers["mwnmodes"]:
            elements = line.strip().split()

            # every column is a normal mode - assign accordingly
            for i in range(len(elements)):
                normal_modes[i + 1].append(float(elements[i]))

        # change current header
        if line in headers.values():
            header = line

    # save as ordered dict for nice output later on
    INFOS["normal_modes"] = OrderedDict(
        sorted({i: normal_mode for i, normal_mode in normal_modes.items() if frequencies[i] != 0}.items(), key=lambda t: t[0])
    )
    INFOS["frequencies"] = OrderedDict(
        sorted({i: freq for i, freq in frequencies.items() if freq != 0}.items(), key=lambda t: t[0])
    )

    return INFOS


# ======================================================================================================================


def get_runscript_info(INFOS):
    """
    Gets all the necessary information from the user for the runscripts

    returns INFOS dictionary
    """

    string = "\n  " + "=" * 80
    string += "\n||" + "{:^80}".format("Run mode setup") + "||"
    string += "\n  " + "=" * 80 + "\n\n"
    print(string)

    print("{:-^60}".format("Run script") + "\n")
    print(
        """  This script can generate the run scripts for each initial condition in two modes:

    - In mode 1, the calculation is run in subdirectories of the current directory.

    - In mode 2, the input files are transferred to another directory (e.g. a local
      scratch directory), the calculation is run there, results are copied back and
      the temporary directory is deleted. Note that this temporary directory is not
      the same as the "scratchdir" employed by the interfaces.

  Note that in any case this script will create the input subdirectories in the
  current working directory."""
    )

    print("\n  In case of mode 1, the calculations will be run in: '%s'\n" % (INFOS["cwd"]))
    INFOS["here"] = question("Use mode 1 (i.e., calculate here)?", bool, True)

    if not INFOS["here"]:
        print("\nWhere do you want to perform the calculations? Note that this script cannot check\nwhether the path is valid.")
        INFOS["copydir"] = question("Run directory?", str)
    print("")

    print("{:-^60}".format("Submission script") + "\n")
    print(
        "During the setup, a script for running all initial conditions sequentially in batch\nmode is generated. Additionally, a queue submission script can be generated for all\ninitial conditions."
    )
    INFOS["qsub"] = question("Generate submission script?", bool, False)

    if INFOS["qsub"]:
        INFOS["qsub"] = True
        print(
            '\nPlease enter a queue submission command, including possibly options to the queueing\nsystem, e.g. for SGE: "qsub -q queue.q -S /bin/bash -cwd" (Do not type quotes!).'
        )
        INFOS["qsubcommand"] = question("Submission command?", str, None, False)
        INFOS["proj"] = question("Project Name:", str, None, False)

    print("")
    return INFOS


# ======================================================================================================================


def calculate_displacements(INFOS):
    """
    Calculates all displacements to set up the calculations
    and the full transformation matrix of dimensionless mass-weighted coordinates

    returns INFOS dictionary
    """

    # dividing normal modes by sqrt(frequency)
    fw_normal_modes = {}
    for i, normal_mode in INFOS["normal_modes"].items():
        fw_normal_modes[i] = [nm / (INFOS["frequencies"][i] ** 0.5) for nm in normal_mode]

    # dividing the normal modes by sqrt(atom_mass)
    fmw_normal_modes = {}
    for i, fw_normal_mode in fw_normal_modes.items():
        j = 0
        fmw_normal_mode = []
        for atom in INFOS["atoms"]:
            fmw_normal_mode.append(fw_normal_mode[j] / (atom["mass (amu)"] ** 0.5))
            fmw_normal_mode.append(fw_normal_mode[j + 1] / (atom["mass (amu)"] ** 0.5))
            fmw_normal_mode.append(fw_normal_mode[j + 2] / (atom["mass (amu)"] ** 0.5))
            j += 3

        fmw_normal_modes[i] = fmw_normal_mode

    # writing frequency and mass weighted normal modes to dict
    INFOS["fmw_normal_modes"] = fmw_normal_modes

    # writing displacements by multiplying normal modes with displacement magnitudes
    displacements = {}
    if INFOS["do_overlaps"]:
        for k, normal_mode in fmw_normal_modes.items():
            displacements[f"{k:03d}_p"] = [nm * INFOS["displacement_magnitudes"][k] for nm in normal_mode]

            # for two sided derivation
            if k not in INFOS["one-sided_derivations"] or str(k) in INFOS["gamma_normal_modes"]:
                displacements[f"{k:03d}_n"] = [nm * INFOS["displacement_magnitudes"][k] * (-1) for nm in normal_mode]
            if INFOS["gammas"] == "five-point stencil" and str(k) in INFOS["gamma_normal_modes"]:
                displacements[f"{k:03d}_p2"] = [2 * nm * INFOS["displacement_magnitudes"][k] for nm in normal_mode]
                displacements[f"{k:03d}_n2"] = [-2 * nm * INFOS["displacement_magnitudes"][k] for nm in normal_mode]

    # sort displacements and return
    INFOS["displacements"] = OrderedDict(sorted(displacements.items(), key=lambda t: t[0]))
    return INFOS


# ======================================================================================================================


def write_QM_in(INFOS, displacement_key, displacement_value, displacement_dir):
    """
    Writes QM.in file for displacement calculations
    """

    # open writable QM.in file
    try:
        qmin = open("%s/QM.in" % (displacement_dir), "w")
    except IOError:
        print("IOError during write_QM_in, displacement_dir=%s" % (displacement_dir))
        quit(1)

    # number of atoms
    string = "%i\n" % (len(INFOS["atoms"]))

    # number of current initial condition
    string += "Displacement %s\n" % displacement_key

    # add eq
    if displacement_key == "000_eq":
        for atom in INFOS["atoms"]:
            string += "%s %f %f %f\n" % (
                atom["atom"],
                atom["coords [bohr]"][0],
                atom["coords [bohr]"][1],
                atom["coords [bohr]"][2],
            )

    # for non eq add displacements
    else:
        i = 0
        for atom in INFOS["atoms"]:
            string += "%s %f %f %f\n" % (
                atom["atom"],
                atom["coords [bohr]"][0] + INFOS["displacements"][displacement_key][i],
                atom["coords [bohr]"][1] + INFOS["displacements"][displacement_key][i + 1],
                atom["coords [bohr]"][2] + INFOS["displacements"][displacement_key][i + 2],
            )
            i += 3

    # unit def
    string += "unit bohr\n"

    # states def
    string += "states "
    for i in INFOS["states"]:
        string += "%i " % (i)
    string += "\n"

    # eq: init ; displacement: overlap
    if displacement_key == "000_eq":
        string += "step 0\n"
    else:
        string += "overlap\n"
        # string += 'cleanup\n'
        string += "step 1\n"

    # set savedir
    string += "savedir ./SAVE/\n"

    # spin orbit coupling
    if INFOS["soc"]:
        string += "\nSOC\n"
    else:
        string += "\nH\n"

    # dipole moment
    string += "DM\n"

    # gradient
    if displacement_key == "000_eq" and INFOS["ana_grad"]:
        string += "GRAD\n"
    elif INFOS["gammas"] == "hessian from diabatized gradients":
        string += "GRAD\n"
    if displacement_key == "000_eq" and INFOS["ana_nac"]:
        string += "NACDR\n"

    # if theodore set it
    if displacement_key == "000_eq" and "theodore" in INFOS and INFOS["theodore"]:
        string += "theodore\n"

    # if multipolar fit is requested
    if displacement_key == "000_eq" and INFOS["multipolar_fit"]:
        string += "multipolar_fit all\n"

    qmin.write(string)
    qmin.close()


# ======================================================================================================================


def write_runscript(INFOS, interface: SHARC_INTERFACE, displacement_dir):
    """
    writes the runscript in each subdirectory
    """

    filename = "%s/run.sh" % (displacement_dir)
    try:
        runscript = open(filename, "w")
    except IOError:
        print("IOError during write_runscript, displacement_dir = %s" % (displacement_dir))
        quit(1)

    if "proj" in INFOS:
        projname = "%4s_%5s" % (INFOS["proj"][0:4], displacement_dir[-6:-1])
    else:
        projname = "init_%5s" % (displacement_dir[-6:-1])

    intstring = ""
    if "amsbashrc" in INFOS:
        intstring = ". %s\nexport PYTHONPATH=$AMSHOME/scripting:$PYTHONPATH" % (INFOS["amsbashrc"])

    refstring = ""
    if displacement_dir != INFOS["result_path"] + "/" + INFOS["paths"]["000_eq"]:
        refstring = (
            'if [ -d ../../%s/SAVE ];\nthen\n  if [ -d ./SAVE ];\n  then\n    rm -r ./SAVE\n  fi\n  cp -r ../../%s/SAVE ./\nelse\n  echo "Should do a reference overlap calculation, but the reference data in ../../%s/ seems not OK."\n  exit 1\nfi'
            % (
                INFOS["result_path"] + "/" + INFOS["paths"]["000_eq"],
                INFOS["result_path"] + "/" + INFOS["paths"]["000_eq"],
                INFOS["result_path"] + "/" + INFOS["paths"]["000_eq"],
            )
        )

    # generate run scripts here
    # for here mode
    if INFOS["here"]:
        string = f"""#!/bin/bash

# $-N {projname}

{intstring}

PRIMARY_DIR={INFOS['cwd']}/{displacement_dir}/

cd $PRIMARY_DIR
{refstring}

$SHARC/{interface.__class__.__name__}.py QM.in > QM.log 2> QM.err
"""
    # for remote mode
    else:
        string = f"""#!/bin/bash

# $-N {projname}

{intstring}

PRIMARY_DIR={INFOS['cwd']}/{displacement_dir}/
COPY_DIR={INFOS['copydir']}/{displacement_dir}/

cd $PRIMARY_DIR
{refstring}

mkdir -p $COPY_DIR
cp -r $PRIMARY_DIR/* $COPY_DIR
cd $COPY_DIR

$SHARC/{interface.__class__.__name__}.py QM.in > QM.log 2> QM.err

cp -r $COPY_DIR/QM.* $COPY_DIR/SAVE/ $PRIMARY_DIR
rm -r $COPY_DIR
"""
    # run, close & make executable
    runscript.write(string)
    runscript.close()
    os.chmod(filename, os.stat(filename).st_mode | stat.S_IXUSR)

    return


# ======================================================================================================================


def write_displacement_info(INFOS):
    """
    Writes displacement info to log-file and INFOS to info-file
    """

    displacement_log_filename = "%s/displacements.log" % (INFOS["result_path"])
    displacement_info_filename = "%s/displacements.json" % (INFOS["result_path"])

    # open writable displacements.info file
    try:
        displacement_log = open(displacement_log_filename, "w")
        displacement_info = open(displacement_info_filename, "w")
    except IOError:
        print(
            "IOError during opening writeable %s or %s - file. Quitting."
            % (displacement_log_filename, displacement_info_filename)
        )
        quit(1)

    # write INFOS to info file
    # pickle.dump(INFOS, displacement_info)
    json.dump(INFOS, displacement_info, sort_keys=True, indent=4)

    # writing header
    displacement_log.write(
        "natoms: %i\nstates: %s\nnstates: %i\n\nnormal_mode displacement_magnitude p/n path\n"
        % (len(INFOS["atoms"]), str(INFOS["states"])[1:-1].replace(",", ""), INFOS["nstates"])
    )

    # writing eq
    displacement_log.write(
        "-%s-%s-%s%s/\n" % (" " * 11, " " * 22, " " * 3, INFOS["result_path"] + "/" + INFOS["paths"]["000_eq"])
    )

    # writing all displacements to log file
    if INFOS["do_overlaps"]:
        for displacement_key, v in INFOS["displacements"].items():
            normal_mode = int(displacement_key.split("_")[0])

            displacement_log.write(
                "%i%s%g%s%c%s%s\n"
                % (
                    normal_mode,
                    " " * (12 - len(str(normal_mode))),
                    INFOS["displacement_magnitudes"][normal_mode],
                    " " * (23 - len(str(INFOS["displacement_magnitudes"][normal_mode]))),
                    displacement_key[-1],
                    " " * 3,
                    INFOS["result_path"] + "/" + INFOS["paths"][displacement_key],
                )
            )

    displacement_log.close()
    displacement_info.close()


# ======================================================================================================================

# def write_frozen_cores(INFOS):
# for path in INFOS['paths'].values():
# relative_path = INFOS['result_path'] + '/' + path

# for file_name in os.listdir(relative_path):
# if file_name.endswith('.resources'):
# try:
# file = open(relative_path + '/' + file_name, 'a')
# file.write('numfrozcore %i\n' % INFOS['frozen_cores'])
# file.close()
# except IOError:
# print('Interface resource file was not found in %s\nCouldn\'t write frozen cores!' % (path))

# ======================================================================================================================


def setup_equilibrium(INFOS, interface: SHARC_INTERFACE):
    """
    Sets up the eq condition
    """

    displacement_dir = INFOS["result_path"] + "/" + INFOS["paths"]["000_eq"]
    io = make_directory(displacement_dir)

    # eq condition already exists
    if io != 0:
        print("Skipping equlibrium %s!" % (displacement_dir))
        return True

    # create eq condition
    write_QM_in(INFOS, "000_eq", None, displacement_dir)
    interface.prepare(INFOS, displacement_dir)
    write_runscript(INFOS, interface, displacement_dir)

    return False


# ======================================================================================================================


def setup_all(INFOS, interface: SHARC_INTERFACE):
    """
    This routine sets up the directories for the initial calculations.
    """
    INFOS["link_files"] = False

    if make_directory(INFOS["result_path"]) == -1:
        print("Results folder will not be created or overwritten. Quitting.")
        quit(1)

    string = "\n  " + "=" * 80
    string += "\n||" + "{:^80}".format("Setting up directories...") + "||"
    string += "\n  " + "=" * 80 + "\n\n"
    print(string)

    # define local variables
    all_run_filename = "%s/all_run_dspl.sh" % (INFOS["result_path"])
    all_qsub_filename = "%s/all_qsub_dspl.sh" % (INFOS["result_path"])

    # write current working directory to all_run_dspl.sh & all_qsub_dspl.sh
    all_run = open(all_run_filename, "w")
    string = "#/bin/bash\n\nCWD=%s\n\n" % (INFOS["cwd"])
    all_run.write(string)

    # add queueing script if wanted
    if INFOS["qsub"]:
        all_qsub = open(all_qsub_filename, "w")
        string = "#/bin/bash\n\nCWD=%s\n\n" % (INFOS["cwd"])
        all_qsub.write(string)

    # if eq doesn't exist yet, set it up & add it to the run_all* files
    if not setup_equilibrium(INFOS, interface):
        eq_dir = INFOS["result_path"] + "/" + INFOS["paths"]["000_eq"]

        string = "cd $CWD/%s/\nbash run.sh\ncd $CWD\necho %s >> %s/DONE\n" % (eq_dir, eq_dir, INFOS["result_path"])
        all_run.write(string)

        if INFOS["qsub"]:
            string = "cd $CWD/%s/\n%s run.sh\ncd $CWD\n" % (eq_dir, INFOS["qsubcommand"])
            all_qsub.write(string)

    # set up all displacement calculations
    if INFOS["do_overlaps"]:
        dispacements_done = 0
        width_progressbar = 50
        number_of_displacements = len(INFOS["displacements"].keys())

        # iterating through displacements
        for displacement_key, displacement_value in INFOS["displacements"].items():
            displacement_dir = INFOS["result_path"] + "/" + INFOS["paths"][displacement_key]

            dispacements_done += 1
            done = dispacements_done * width_progressbar // number_of_displacements

            sys.stdout.write(
                "\rProgress: [" + "=" * done + " " * (width_progressbar - done) + "] %3i%%" % (done * 100 // width_progressbar)
            )
            sys.stdout.flush()

            if make_directory(displacement_dir) != 0:
                print("Skipping displacement %s!" % (displacement_dir))
                continue

            # write QM.in, interfaces & runscript for displacement
            write_QM_in(INFOS, displacement_key, displacement_value, displacement_dir)
            # getattr(interfaces, [Interfaces[INFOS['interface']]['prepare_routine']][0])(INFOS, displacement_dir)
            interface.prepare(INFOS, displacement_dir)
            write_runscript(INFOS, interface, displacement_dir)

            string = "cd $CWD/%s/\nbash run.sh\ncd $CWD\necho %s >> %s/DONE\n" % (
                displacement_dir,
                displacement_dir,
                INFOS["result_path"],
            )
            all_run.write(string)

            if INFOS["qsub"]:
                string = "cd $CWD/%s/\n%s run.sh\ncd $CWD\n" % (displacement_dir, INFOS["qsubcommand"])
                all_qsub.write(string)

    # close filehandlers & make executable
    all_run.close()
    os.chmod(all_run_filename, os.stat(all_run_filename).st_mode | stat.S_IXUSR)

    if INFOS["qsub"]:
        all_qsub.close()
        os.chmod(all_qsub_filename, os.stat(all_qsub_filename).st_mode | stat.S_IXUSR)

    # write displacement info
    write_displacement_info(INFOS)

    # write frozen cores
    # if INFOS['frozen_cores'] != -1:
    # write_frozen_cores(INFOS)

    print("\n")


# ======================================================================================================================


def main():
    """
    Main routine
    """

    usage = """python setup_displacement.py"""

    parser = OptionParser(usage=usage, description="")

    displaywelcome()
    open_keystrokes()

    # get general information for calcultion
    INFOS = get_V0_and_states()
    chosen_interface: SHARC_INTERFACE = get_interface()()
    chosen_interface.QMin.molecule["states"] = INFOS["states"]
    INFOS = get_setup_info(INFOS, chosen_interface)
    chosen_interface.get_infos(INFOS, KEYSTROKES)
    # get interface info - use reflection to get chosen routine
    INFOS = get_runscript_info(INFOS)
    # get the interface

    print_INFOS(INFOS)

    if question("Do you want to setup the specified calculations?", bool, True):
        setup_all(INFOS, chosen_interface)

    close_keystrokes()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nCtrl+C makes me a sad SHARC ;-(\n")
        quit(0)
