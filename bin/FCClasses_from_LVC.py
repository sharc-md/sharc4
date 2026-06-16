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


import numpy as np
import os
import shutil
import sys
import datetime
# import argparse
from collections import defaultdict
from utils import question, readfile, writefile
from printing import printheader
from constants import au2eV, au2rcm

HARTREE_TO_CM1 = au2rcm
HARTREE_TO_EV = au2eV

version = "4.1"
versiondate = datetime.date(2025, 9, 1)

# ======================================================================================================================
# ======================================================================================================================
# ======================================================================================================================


def displaywelcome():
    lines = [
        "Creating FCClasses3 input from LVC models",
        "",
        "Authors: Diksha",
        "Version:" + version,
        versiondate.strftime("%d.%m.%y"),
    ]
    printheader(lines)
    print(
        """
This script converts an LVC model into input files for FCClasses3 to generate vibronic spectra
  """
    )

# ======================================================================================================================

def open_keystrokes():
    global KEYSTROKES
    KEYSTROKES = open("KEYSTROKES.tmp", "w")

def close_keystrokes():
    KEYSTROKES.close()
    shutil.move("KEYSTROKES.tmp", "KEYSTROKES.FCClasses_from_LVC")

global KEYSTROKES
old_question = question

def question(question, typefunc, default=None, autocomplete=True, ranges=False):
    return old_question(
        question=question, typefunc=typefunc, KEYSTROKES=KEYSTROKES, default=default, autocomplete=autocomplete, ranges=ranges
    )

# ======================================================================================================================
# ======================================================================================================================
# ======================================================================================================================


# ====== Prompt Utilities ======
def prompt_path(prompt_text, default_path):
    while True:
        path = question(prompt_text, str, default=default_path, autocomplete=True)
        if os.path.isfile(path):
            return path
        else:
            print("File not found.")

def prompt_state(prompt_text, default=None, choices=None):
    if choices:
        print(f"Available states: {choices}")
    while True:
        choice = question(prompt_text, int, default=[default])[0]
        if choice in choices:
            return choice
        else:
            print(f"Invalid choice. Please choose from: {choices}")

# ====== FCclasses Setup Core ======
def read_v0(v0_path):
    atoms, masses, mode_vectors = [], [], []
    freqs = []
    section = None
    with open(v0_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.lower() == "geometry":
                section = "geometry"
                continue
            elif line.lower().startswith("frequencies"):
                section = "frequencies"
                continue
            elif line.lower().startswith("mass-weighted normal modes"):
                section = "modes"
                continue

            if section == "geometry":
                parts = line.split()
                atoms.append(parts[0])
                masses.append(float(parts[5]))
            elif section == "frequencies":
                freqs.extend([float(x) for x in line.split()])
            elif section == "modes":
                mode_vectors.extend([float(x) for x in line.split()])
    num_atoms = len(atoms)
    num_modes = len(mode_vectors) // (3 * num_atoms)
    mode_matrix = np.array(mode_vectors).reshape(num_modes, 3 * num_atoms)
    return atoms, np.array(masses), mode_matrix, freqs

# ---------------------------------------
def read_kappas(lvc_template_path):
    kappa_dict = defaultdict(list)
    in_kappa_block = False
    kappa_lines_remaining = 0

    with open(lvc_template_path, "r") as f:
        for line in f:
            line = line.strip()

            if line.lower() == "kappa":
                in_kappa_block = True
                continue
            if in_kappa_block and kappa_lines_remaining == 0:
                try:
                    kappa_lines_remaining = int(line)
                except ValueError:
                    break
                continue
            if in_kappa_block and kappa_lines_remaining > 0:
                parts = line.split()
                if len(parts) == 4:
                    _, state, mode, value = parts
                    kappa_dict[int(state)].append((int(mode) - 1, float(value)))
                    kappa_lines_remaining -= 1
    return kappa_dict

# ---------------------------------------
def extract_de_ev(lvc_template_path, final_state):
    with open(lvc_template_path, "r") as f:
        lines = f.readlines()

    epsilon_index = next((i for i, line in enumerate(lines) if line.strip().lower() == "epsilon"), None)
    if epsilon_index is None:
        raise ValueError("No 'epsilon' section found in LVC.template.")

    n_states = int(lines[epsilon_index + 1].strip())

    for i in range(epsilon_index + 2, epsilon_index + 2 + n_states):
        parts = lines[i].strip().split()
        if len(parts) >= 3 and int(parts[1]) == final_state:
            return float(parts[2]) * HARTREE_TO_EV
    raise ValueError(f"Final state index {final_state} not found in epsilon section.")

# ---------------------------------------
def get_available_states_from_epsilon(lvc_template_path):
    with open(lvc_template_path, "r") as f:
        lines = f.readlines()
    epsilon_index = next((i for i, line in enumerate(lines) if line.strip().lower() == "epsilon"), None)
    n_singlets = int(lines[epsilon_index + 1].strip())
    return list(range(1, n_singlets + 1))

# ---------------------------------------
def normalize_gradient(gradient, frequencies_cm1):
    frequencies_au = np.array(frequencies_cm1) / HARTREE_TO_CM1
    frequencies_au = np.where(frequencies_au > 1e-12, frequencies_au, 1e-12)
    return gradient * np.sqrt(frequencies_au)

# ---------------------------------------
def get_dipole_blocks(lvc_template_path):
    dmx, dmy, dmz = [], [], []
    section = None
    found = {"dmx": False, "dmy": False, "dmz": False}

    with open(lvc_template_path, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("DMX R"):
                section = "dmx"
                found["dmx"] = True
                continue
            elif line.startswith("DMY R"):
                section = "dmy"
                found["dmy"] = True
                continue
            elif line.startswith("DMZ R"):
                section = "dmz"
                found["dmz"] = True
                continue
            elif section and (line == "" or line.lower().startswith(("lambda", "/", "kappa","multipolar","spin"))):
                section = None
                continue

            if section == "dmx":
                dmx.append([float(x) for x in line.split()])
            elif section == "dmy":
                dmy.append([float(x) for x in line.split()])
            elif section == "dmz":
                dmz.append([float(x) for x in line.split()])

    max_size = max(len(dmx), len(dmy), len(dmz))
    size = max_size if max_size > 0 else 2

    if not found["dmx"]:
        print("DMX R block not found. Using zero matrix.")
        dmx = np.zeros((size, size)).tolist()
    if not found["dmy"]:
        print("DMY R block not found. Using zero matrix.")
        dmy = np.zeros((size, size)).tolist()
    if not found["dmz"]:
        print("DMZ R block not found. Using zero matrix.")
        dmz = np.zeros((size, size)).tolist()

    return dmx, dmy, dmz

# ---------------------------------------
def write_dipole_xyz_rows(dmx, dmy, dmz, n_modes, output_path="eldip_fchk"):
    rows = np.zeros((n_modes, 3))
    try:
        rows[0, 0] = dmx[0][1]
        rows[0, 1] = dmy[0][1]
        rows[0, 2] = dmz[0][1]
    except IndexError:
        print("Could not extract d00 (S0->S0). Row 0 remains zeros.")
    try:
        rows[1, 0] = dmx[0][1]
        rows[1, 1] = dmy[0][1]
        rows[1, 2] = dmz[0][1]
    except IndexError:
        print("Could not extract d10 (S1->S0). Row 1 remains zeros.")
    np.savetxt(output_path, rows, fmt="%.8E")

# ---------------------------------------
def write_fcc_inp(nvib, de_ev):
    content = f"""$$$
PROPERTY     =   OPA
MODEL        =   VG
DIPOLE       =   FC
DE           =   {de_ev:.5f}
TEMP         =   0.00
BROADFUN     =   GAU
HWHM         =   0.005
METHOD       =   TD
NVIB         =   {nvib}
NORMALMODES  =   IMPLICIT
FREQ1_FILE   =  freq1.txt
FREQ2_FILE   =  freq2.txt
GRAD2_FILE   =  grad2.txt
ELDIP_FILE   =  eldip_fchk
DUSCH_FILE   =  IDENTITY
"""
    writefile("fcc.inp", content)
    # with open("fcc.inp", "w") as f:
        # f.write(content)

# ---------------------------------------
def run_converter(v0_path, lvc_path, initial_state, final_state):
    atoms, masses, mode_matrix, freqs = read_v0(v0_path)
    nvib = len(freqs) - 6
    freqs_cm1 = [f * HARTREE_TO_CM1 for f in freqs[6:] if f > 1e-8]

    np.savetxt("freq1.txt", freqs_cm1, fmt="%12.5f")
    np.savetxt("freq2.txt", freqs_cm1, fmt="%12.5f")

    kappa_data = read_kappas(lvc_path)

    for state, fname in [(initial_state, "grad1.txt"), (final_state, "grad2.txt")]:
        if state not in kappa_data:
            print(f"State {state} not found in kappa.")
            sys.exit(1)
        full_kappa = np.zeros(len(freqs))
        for mode_idx, value in kappa_data[state]:
            full_kappa[mode_idx] = value
        grad = normalize_gradient(full_kappa[6:], freqs_cm1)
        np.savetxt(fname, grad, fmt="%.8e")

    dmx, dmy, dmz = get_dipole_blocks(lvc_path)
    write_dipole_xyz_rows(dmx, dmy, dmz, nvib)

    de_ev = extract_de_ev(lvc_path, final_state)
    write_fcc_inp(nvib, de_ev)

    print("All files generated successfully.")
    print("Output: freq1.txt  freq2.txt  grad1.txt  grad2.txt  eldip_fchk  fcc.inp")
    print("To run FCclasses:")
    print("    export OMP_NUM_THREADS=N")
    print("    fcclasses3 fcc.inp")
    print("To plot the spectrum subsequently, use spec_Int_TD.dat")


# ======================================================================================================================
# ======================================================================================================================
# ======================================================================================================================

def main():
    displaywelcome()
    open_keystrokes()

    v0_path = prompt_path("Enter path to V0.txt:", "V0.txt")
    lvc_path = prompt_path("Enter path to LVC.template:", "LVC.template")
    available_states = get_available_states_from_epsilon(lvc_path)
    initial_state = prompt_state("Select initial state index:", default=1, choices=available_states)
    final_choices = [s for s in available_states if s != initial_state]
    final_state = prompt_state("Select final state index:", default=2, choices=final_choices)

    run_converter(v0_path, lvc_path, initial_state, final_state)
    
    close_keystrokes()

# ======================================================================================================================
if __name__ == "__main__":
    main()

