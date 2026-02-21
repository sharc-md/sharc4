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
from constants import HARTREE_TO_EV, IToMult
from qmout import QMout
from utils import itnmstates


def transform(H, DM):
    """transforms the H and DM matrices in the representation where H is diagonal."""
    eig, U = np.linalg.eigh(H)
    Ucon = U.conj().T

    H[:] = 0
    np.fill_diagonal(H, eig.astype(complex))

    if DM is not None:
        DM = Ucon @ DM @ U
    return H, DM, U


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
    parser.add_argument("-e", type=float, default=0.0, help="Absolute energy shift (float, default=compute relative energies)")
    parser.add_argument("-D", action="store_true", help="Diagonalize")
    parser.add_argument("-S", type=int, default=1, help="Initial state (Lowest=1)")
    parser.add_argument("-L", action="store_true", help="Format in a single line")
    parser.add_argument("-I", action="store_true", default=False, help="Use Dyson norms instead of oscillator strengths")

    options = parser.parse_args()
    ezero = options.e
    initial = options.S - 1
    target_list = {1, 2}  # h, dm

    if options.I:
        if options.D:
            print("-I and -D are not compatible.")
            sys.exit()
        target_list.add(20)  # prop2d

    qmout = QMout(options.inputfile, flags=target_list)
    nmstates = qmout.nmstates
    states = qmout.states
    if options.I:
        for i in qmout.prop2d:
            if i[0] == "ion":
                ion = i[1]
                break
        else:
            raise ValueError("ION not found!")

    # obtain the statemap
    statemap = {}
    i = 1
    for imult, istate, ims in itnmstates(qmout.states):
        statemap[i] = [imult, istate, ims]
        i += 1

    if not options.L:
        sys.stderr.write(f"{options.inputfile} {nmstates} {target_list}\n")
        sys.stderr.write(f"Number of states: {states}\n")
        sys.stderr.write(
            f"{'State':>5s}  {'Label':>11s} {'E (E_h)':>16s} "
            f"{'dE (eV)':>12s} {(['f_osc', 'Dys norm'][options.I]):>12s}   {'Spin':>6s}\n"
        )

    if options.D:
        h, dm, U = transform(qmout.h, qmout.dm)
    else:
        h = qmout.h
        try:
            dm = qmout.dm
        except AttributeError:
            dm = np.zeros((3, nmstates, nmstates), dtype=complex)

    fosc = []
    energies = np.real(np.diag(h))
    m = np.array([statemap[i + 1][0] for i in range(nmstates)], dtype=int)
    s = np.array([statemap[i + 1][1] for i in range(nmstates)], dtype=int)
    ms = np.array([statemap[i + 1][2] for i in range(nmstates)], dtype=float)
    if options.D:
        for istate in range(nmstates):
            e = float(energies[istate])

            w = np.abs(U[:, istate]) ** 2
            spin = float(m @ w)

            jbest = int(np.argmax(w))
            m_best = int(m[jbest])
            s_best = int(s[jbest])

            d = np.real(dm[:, istate, initial])
            f = (2.0 / 3.0) * (e - float(energies[initial])) * float(d @ d)
            fosc.append(f)

            ref = ezero if ezero != 0.0 else float(energies[initial])
            de = (e - ref) * HARTREE_TO_EV

            line = (
                f"{istate + 1:5d} {IToMult[m_best][0]:>10s}{(s_best - (m_best <= 2)):02d} "
                f"{e:16.10f} {de:12.8f} {f:12.8f}   {spin:6.4f}"
            )
            if istate == initial:
                line += " #initial state"
            if not options.L:
                print(line)
    else:
        ref = ezero if ezero != 0.0 else energies[initial]
        de = (energies - ref) * HARTREE_TO_EV
        ok = (-2.0 * ms + 1.0) == m

        for i, e in enumerate(energies):
            f = (
                float(np.real(ion[i][initial]))
                if options.I
                else (2.0 / 3.0) * (e - energies[initial]) * float((np.real(dm[:, i, initial]) ** 2).sum())
            )
            fosc.append(f)

            if not options.L and ok[i]:
                line = (
                    f"{i+1:5d} {IToMult[m[i]][0]:>10s}{(s[i] - (m[i] <= 2)):02d} "
                    f"{float(e):16.10f} {float(de[i]):12.8f} {f:12.8f}   {float(m[i]):6.4f}"
                )
                if i == initial:
                    line += " #initial state"
                print(line)

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
