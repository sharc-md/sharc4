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

# Script for printing excitation energies, oscillator strengths and other quantities from QM.out file
#
# usage python QMout_print.py [options] <QM.out>

import argparse
import os
import sys

import numpy as np
from constants import HARTREE_TO_EV, IToMult, alpha
from qmout import QMout
from utils import itnmstates


def transform(H, DM, MDM, EQM):
    """transforms the H and DM matrices in the representation where H is diagonal."""
    eig, U = np.linalg.eigh(H)
    Ucon = U.conj().T

    H[:] = 0
    np.fill_diagonal(H, eig.astype(complex))

    if DM is not None:
        DM = Ucon @ DM @ U
    if MDM is not None:
        MDM = Ucon @ MDM @ U
    if EQM is not None:
        EQM = Ucon @ EQM @ U
    return H, DM, MDM, EQM, U


# ========================== Main Code =============================== #


def main():

    usage = """
QMout_print.py [options] QM.out

This script reads a QM.out file from a SHARC interface and prints
excitation energies and oscillator strengths.
"""

    description = ""

    parser = argparse.ArgumentParser(usage=usage, description=description)
    parser.add_argument("inputfile", help="Input file")
    parser.add_argument("-e", type=float, default=0.0, help="Absolute energy shift (float, default: compute relative energies)")
    parser.add_argument("-D", action="store_true", help="Diagonalize")
    parser.add_argument("-S", type=int, default=1, help="Initial state (default: lowest=1)")
    parser.add_argument("-L", action="store_true", help="Format in a single line")
    parser.add_argument("-I", action="store_true", default=False, help="Use Dyson norms instead of oscillator strengths")
    parser.add_argument("-M", action="store_true", default=False, help="Include magnetic dipoles/electric quadrupoles if present")

    options = parser.parse_args()
    ezero = options.e
    initial = options.S - 1
    target_list = {1, 2}  # h, dm

    if options.I:
        if options.D:
            raise ValueError("-I and -D are not compatible.")
        target_list.add(20)  # prop2d

    if options.M:
        target_list.update({41, 42})

    qmout = QMout(options.inputfile, flags=target_list)
    nmstates = qmout.nmstates
    states = qmout.states

    # check if Dyson norms are there
    if options.I:
        for i in qmout.prop2d:
            if i[0] == "ion":
                ion = i[1]
                break
        else:
            raise ValueError("ION not found!")
    
    # Check if MDM and EQM are there
    qmout.mdm = getattr(qmout, "mdm", None)
    qmout.edm = getattr(qmout, "eqm", None)
    if options.M:
        if qmout.mdm is None or qmout.edm is None:
            raise ValueError("-M but no magnetic dipoles/electric quadrupoles in file!")

    # obtain the statemap
    statemap = {}
    i = 1
    for imult, istate, ims in itnmstates(qmout.states):
        statemap[i] = [imult, istate, ims]
        i += 1

    # print header
    if not options.L:
        sys.stderr.write(f"Number of states: {states}\n")
        sys.stderr.write(
            f"{'State':>5s}  {'Label':>11s} {'E (E_h)':>16s} "
            f"{'dE (eV)':>12s} {(['f_osc', 'Dys norm'][options.I]):>12s}   {'Spin':>6s}\n"
        )

    # transform and prepare quantities
    if options.D:
        h, dm, mdm, eqm, U = transform(qmout.h, qmout.dm, qmout.mdm, qmout.edm)
    else:
        h = qmout.h
        try:
            dm = qmout.dm
        except AttributeError:
            dm = np.zeros((3, nmstates, nmstates), dtype=complex)

    # initialize quantum numbers
    m = np.array([statemap[i + 1][0] for i in range(nmstates)], dtype=int)
    s = np.array([statemap[i + 1][1] for i in range(nmstates)], dtype=int)
    ms = np.array([statemap[i + 1][2] for i in range(nmstates)], dtype=float)


    # get a list of the to-be-printed states and their labels
    indices = []
    labels = []
    if options.D:
        # --- diagonalized representation ---
        for istate in range(nmstates):

            w = np.abs(U[:, istate]) ** 2
            jbest = int(np.argmax(w))

            m_best = int(m[jbest])
            s_best = int(s[jbest])

            label = f"{IToMult[m_best][0]:>10s}{(s_best - (m_best <= 2)):02d}"

            indices.append(istate)
            labels.append(label)
    else:
        # --- original representation ---
        ok = (-2.0 * ms + 1.0) == m

        for i in range(nmstates):
            if ok[i]:
                label = f"{IToMult[m[i]][0]:>10s}{(s[i] - (m[i] <= 2)):02d}"

                indices.append(i)
                labels.append(label)


    # compute values
    fosc = []
    energies = np.real(np.diag(h))
    ref = ezero if ezero != 0.0 else energies[initial]

    for idx, label in zip(indices, labels):

        # energy
        e = float(energies[idx])
        de = (e - ref) * HARTREE_TO_EV

        # spin
        if options.D:
            w = np.abs(U[:, idx]) ** 2
            spin = float(m @ w)
        else:
            spin = float(m[idx])

        # oscillator strength
        if options.I:
            f = float(np.real(ion[idx][initial]))
        else:
            d = np.real(dm[:, idx, initial])
            f = (2.0 / 3.0) * (e - ref) * float(d @ d)
            if options.M:
                mdm = np.imag(qmout.mdm[:, idx, initial])
                f += (2.0 / 3.0) * (e - ref) * float(mdm @ mdm)
                eqm = qmout.eqm[:, :, idx, initial]
                quad_term = np.sum(np.abs(eqm) ** 2) - (1 / 3.0) * np.abs(np.trace(eqm)) ** 2
                f += (1.0 / 20.0) * alpha**2 * (e - ref) ** 3 * quad_term
        fosc.append(f)

        # print if not one-line output
        if not options.L:
            line = (
                f"{idx+1:5d} {label} "
                f"{e:16.10f} {de:12.8f} {f:12.8f}   {spin:6.4f}"
            )
            if idx == initial:
                line += " #initial state"
            print(line)

    # print one-line output
    if options.L:
        cwd = os.path.basename(os.getcwd()).split("_")[-1]

        if options.D:
            indices = range(nmstates)
        else:
            indices = [i for i in range(nmstates) if (-2 * statemap[i + 1][2] + 1) == statemap[i + 1][0]]

        parts = [cwd]
        parts += [f"{energies[i]:16.10f}" for i in indices]
        parts += [f"{fosc[i]:12.8f}" for i in indices]

        print(" ".join(parts))


if __name__ == "__main__":
    main()
