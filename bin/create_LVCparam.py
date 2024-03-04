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
import numpy as np
from scipy import linalg
from optparse import OptionParser

from constants import IToMult
from utils import itnmstates, readfile
from printing import printheader
from qmout import QMout

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


if sys.version_info[0] != 3:
    print("This is a script for Python 3!")
    sys.exit(0)

version = "2.1"
versionneeded = [0.2, 1.0, 2.0, 2.1, float(version)]
versiondate = datetime.date(2019, 9, 1)


# ======================================================================= #

pthresh = 1.0e-5**2

# ======================================================================= #


def displaywelcome():
    lines = [
        f"Compute LVC parameters",
        "",
        f"Authors: Simon Kropf, Sebastian Mai, Severin Polonius",
        "",
        f"Version: {version}",
        "Date: {:%d.%m.%Y}".format(versiondate),
    ]

    print("Script for setup of displacements started...\n")
    printheader(lines)
    string = "This script automatizes the setup of excited-state calculations for displacements\nfor SHARC dynamics."
    print(string)


# ======================================================================= #


def read_QMout(path, nstates, natom, request):
    targets = {
        "h": {"flag": 1, "type": complex, "dim": (nstates, nstates)},
        "dm": {"flag": 2, "type": complex, "dim": (3, nstates, nstates)},
        "grad": {"flag": 3, "type": float, "dim": (nstates, natom, 3)},
        "nacdr": {"flag": 5, "type": float, "dim": (nstates, nstates, natom, 3)},
        "overlap": {"flag": 6, "type": complex, "dim": (nstates, nstates)},
        "multipolar_fit": {"flag": 22, "type": float, "dim": (nstates, nstates, natom, 10)},
    }

    # read QM.out
    lines = readfile(path)

    # obtain all targets
    QMout = {}
    for t in targets:
        if t in request:
            iline = -1
            while True:
                iline += 1
                if iline >= len(lines):
                    print('Could not find "%s" (flag "%i") in file %s!' % (t, targets[t]["flag"], path))
                    sys.exit(11)
                line = lines[iline]
                if "! %i" % (targets[t]["flag"]) in line:
                    # store settings of multipolar fit
                    if t == "multipolar_fit":
                        QMout["multipolar_fit_settings"] = line.split("states")[-1].strip()
                    break
            values = []
            # =========== single matrix
            if len(targets[t]["dim"]) == 2:
                iline += 1
                for irow in range(targets[t]["dim"][0]):
                    iline += 1
                    line = lines[iline].split()
                    if targets[t]["type"] == complex:
                        row = [complex(float(line[2 * i]), float(line[2 * i + 1])) for i in range(targets[t]["dim"][1])]
                    elif targets[t]["type"] == float:
                        row = [float(line[i]) for i in range(targets[t]["dim"][1])]
                    else:
                        row = line
                    values.append(row)
            # =========== list of matrices
            elif len(targets[t]["dim"]) == 3:
                for iblocks in range(targets[t]["dim"][0]):
                    iline += 1
                    block = []
                    for irow in range(targets[t]["dim"][1]):
                        iline += 1
                        line = lines[iline].split()
                        if targets[t]["type"] == complex:
                            row = [complex(float(line[2 * i]), float(line[2 * i + 1])) for i in range(targets[t]["dim"][2])]
                        elif targets[t]["type"] == float:
                            row = [float(line[i]) for i in range(targets[t]["dim"][2])]
                        else:
                            row = line
                        block.append(row)
                    values.append(block)
            # =========== matrix of matrices
            elif len(targets[t]["dim"]) == 4:
                for iblocks in range(targets[t]["dim"][0]):
                    sblock = []
                    for jblocks in range(targets[t]["dim"][1]):
                        iline += 1
                        block = []
                        for irow in range(targets[t]["dim"][2]):
                            iline += 1
                            line = lines[iline].split()
                            if targets[t]["type"] == complex:
                                row = [complex(float(line[2 * i]), float(line[2 * i + 1])) for i in range(targets[t]["dim"][3])]
                            elif targets[t]["type"] == float:
                                row = [float(line[i]) for i in range(targets[t]["dim"][3])]
                            else:
                                row = line
                            block.append(row)
                        sblock.append(block)
                    values.append(sblock)
            QMout[t] = np.array(values)

    return QMout


# ======================================================================= #


def LVC_complex_mat(header, mat, deldiag=False, oformat=" % .7e"):
    rnonzero = False
    inonzero = False

    rstr = header + " R\n"
    istr = header + " I\n"
    for i in range(len(mat)):
        for j in range(len(mat)):
            val = mat[i][j].real
            if deldiag and i == j:
                val = 0.0
            rstr += oformat % val
            if val * val > pthresh:
                rnonzero = True

            val = mat[i][j].imag
            if deldiag and i == j:
                val = 0.0
            istr += oformat % val
            if val * val > pthresh:
                inonzero = True

        rstr += "\n"
        istr += "\n"

    retstr = ""
    if rnonzero:
        retstr += rstr
    if inonzero:
        retstr += istr

    return retstr


# ======================================================================= #


def loewdin_orthonormalization(A):
    S = A.T @ A
    e, v = np.linalg.eigh(S)
    idx = e > 1e-15
    S_lo = np.dot(v[:, idx] / np.sqrt(e[idx]), v[:, idx].conj().T)
    return A @ S_lo
    # return np.matmul(A, linalg.fractional_matrix_power(np.matmul(A.T, A), -0.5))


def loewdin_orthonormalization_old(A):
    """
    returns loewdin orthonormalized matrix
    """

    # S = A^T * A
    S = np.dot(A.T, A)

    # S^d = U^T * S * U
    S_diag_only, U = np.linalg.eigh(S)

    # calculate the inverse sqrt of the diagonal matrix
    S_diag_only_inverse_sqrt = [1.0 / (float(d) ** 0.5) for d in S_diag_only]
    S_diag_inverse_sqrt = np.diag(S_diag_only_inverse_sqrt)

    # calculate inverse sqrt of S
    S_inverse_sqrt = np.dot(np.dot(U, S_diag_inverse_sqrt), U.T)

    # calculate loewdin orthonormalized matrix
    A_lo = np.dot(A, S_inverse_sqrt)

    # normalize A_lo
    A_lo = A_lo.T
    length = len(A_lo)
    A_lon = np.zeros((length, length), dtype=complex)

    for i in range(length):
        norm_of_col = np.linalg.norm(A_lo[i])
        # A_lon[i] = [e / (norm_of_col**0.5) for e in A_lo[i]][0]
        A_lon[i] = A_lo[i] / np.sqrt(norm_of_col)

    return A_lon.T


# ======================================================================= #


def partition_matrix(matrix, multiplicity, states):
    """
    return the first partitioned matrix of the given multiplicity

    e. g.: (3 0 2) states

      [111, 121, 131,   0,   0,   0,   0,   0,   0]       returns for multiplicity of 1:
      [112, 122, 132,   0,   0,   0,   0,   0,   0]             [111, 121, 131]
      [113, 123, 133,   0,   0,   0,   0,   0,   0]             [112, 122, 132]
      [  0,   0,   0, 311, 321,   0,   0,   0,   0]             [113, 123, 133]
      [  0,   0,   0, 312, 322,   0,   0,   0,   0] ====>
      [  0,   0,   0,   0,   0, 311, 321,   0,   0]       returns for multiplicity of 3:
      [  0,   0,   0,   0,   0, 312, 322,   0,   0]               [311, 321]
      [  0,   0,   0,   0,   0,   0,   0, 311, 321]               [312, 322]
      [  0,   0,   0,   0,   0,   0,   0, 312, 322]

      123 ^= 1...multiplicity
             2...istate
             3...jstate
    """
    # get start index based on given multiplicity
    start_index = sum((s * (i + 1) for i, s in enumerate(states[: multiplicity - 1])))

    # size of the partition ^= state for given multiplicity
    size = states[multiplicity - 1]

    # return [x[start_index : start_index + size] for x in matrix[start_index : start_index + size]]
    return matrix[start_index : start_index + size, start_index : start_index + size, ...]


# ======================================================================= #


def phase_correction(matrix):
    U = matrix.real.copy()
    det = np.linalg.det(U)
    if det < 0:
        U[:, 0] *= -1.0  # this row/column convention is correct

    # sweeps
    l = len(U)
    sweeps = 0
    while True:
        done = True
        for j in range(l):
            for k in range(j + 1, l):
                delta = 3.0 * (U[j, j] ** 2 + U[k, k] ** 2)
                delta += 6.0 * U[j, k] * U[k, j]
                delta += 8.0 * (U[k, k] + U[j, j])
                delta -= 3.0 * (U[j, :] @ U[:, j] + U[k, :] @ U[:, k])
                # for i in range(l):
                # delta -= 3.0 * (U[j, i] * U[i, j] + U[k, i] * U[i, k])
                if delta < -1e-15:  # needs proper threshold towards 0
                    U[:, j] *= -1.0  # this row/column convention is correct
                    U[:, k] *= -1.0  # this row/column convention is correct
                    done = False
        sweeps += 1
        if done:
            break
    return U


def phase_correction_old(matrix):
    length = len(matrix)
    # phase_corrected_matrix = [[0.0 for x in range(length)] for x in range(length)]
    phase_corrected_matrix = np.zeros_like(matrix)

    for i in range(length):
        diag = matrix[i][i].real

        # look if diag is significant and negative & switch phase
        if diag**2 > 0.5 and diag < 0:
            for j in range(length):
                phase_corrected_matrix[j][i] = matrix[j][i] * -1
        # otherwise leave values as is
        else:
            for j in range(length):
                phase_corrected_matrix[j][i] = matrix[j][i]

    return phase_corrected_matrix


# ======================================================================= #


def check_overlap_diagonal(matrix, states, normal_mode, displacement):
    """
    Checks for problematic states (diagonals**2 of overlap matrix smaller than 0.5)
    """
    problematic_states = {}

    for imult in range(len(states)):
        part_matrix = partition_matrix(matrix, imult + 1, states)

        for state in range(len(part_matrix)):
            sum_column = sum([part_matrix[j][state] ** 2 for j in range(len(part_matrix))]).real
            if sum_column < 0.5:
                print("* Problematic state %i in %i%s: %s" % (state + 1, int(normal_mode), displacement, IToMult[imult + 1]))
                problematic_states[str(normal_mode) + displacement] = imult + 1

    return problematic_states


# ======================================================================= #


def calculate_W_dQi(H, S, e_ref):
    """
    Calculates the displacement matrix
    """

    # get diagonalised hamiltonian
    # H = np.diag([e - e_ref for e in np.diag(H)])

    # S = phase_correction(S)

    # do loewdin orthonorm. on overlap matrix
    U = loewdin_orthonormalization(S)
    U = phase_correction(U)

    # <old|new><new|new><new|old> -> <old|old>
    return np.dot(np.dot(U, H), U.T) - np.eye(H.shape[0]) * e_ref


# ======================================================================= #


def write_LVC_template(INFOS):
    lvc_template_content = "%s\n" % (INFOS["v0f"])
    lvc_template_content += str(INFOS["states"])[1:-1].replace(",", "") + "\n"

    # print INFOS

    # print some infos
    print("\nData extraction started ...")
    print("Number of states:", INFOS["nstates"])
    print("Number of atoms:", len(INFOS["atoms"]))
    print("Kappas:", ["numerical", "analytical"][INFOS["ana_grad"]])
    print("Lambdas:", ["numerical", "analytical"][INFOS["ana_nac"]])
    if "gammas" in INFOS and INFOS["gammas"]:
        print(f"Gammas: {INFOS['gammas']}")
    print()
    print("Reading files ...")
    print()

    # extract data from central point
    requests = ["h", "dm"]
    if INFOS["ana_grad"]:
        requests.append("grad")
    if INFOS["ana_nac"]:
        requests.append("nacdr")
    if "multipolar_fit" in INFOS and INFOS["multipolar_fit"]:
        requests.append("multipolar_fit")
    path = os.path.join(INFOS["paths"]["0eq"], "QM.out")
    print("reading QMout_eq at:", path)
    # QMout_eq = read_QMout(path, INFOS["nstates"], len(INFOS["atoms"]), requests)
    QMout_eq = QMout(path, INFOS["states"], len(INFOS["atoms"]), npc=0)
    print(", ".join(requests))
    for normal_mode in INFOS["fmw_normal_modes"].keys():
        INFOS["fmw_normal_modes"][normal_mode] = np.array(INFOS["fmw_normal_modes"][normal_mode])

    # ------------------ epsilon ----------------------
    epsilon_str_list = []

    i = 0
    e_ref = QMout_eq.h[0, 0]

    # run through all multiplicities
    for imult in range(len(INFOS["states"])):
        # partition matrix for every multiplicity
        partition = partition_matrix(QMout_eq.h, imult + 1, INFOS["states"])

        # run over diagonal and get epsilon values
        for istate in range(len(partition)):
            epsilon_str_list.append("%3i %3i % .10f\n" % (imult + 1, istate + 1, (partition[istate][istate] - e_ref).real))

    # add results to template string
    lvc_template_content += "epsilon\n"
    lvc_template_content += "%i\n" % (len(epsilon_str_list))
    lvc_template_content += "".join(sorted(epsilon_str_list))

    # ------------------- kappa -----------------------
    nkappa = 0
    kappa_str_list = []

    # run through all possible states
    if INFOS["ana_grad"]:
        start = 0
        for imult, nsi in enumerate(INFOS["states"]):
            if nsi == 0:
                continue
            # puts the gradient matrix into a list, has form: [ x, y, z, x, y, z, x, y, z]
            gradient = QMout_eq.grad[start : start + nsi, ...].reshape((nsi, -1))

            # runs through normal modes
            for normal_mode in INFOS["fmw_normal_modes"].keys():
                # calculates kappa from normal modes and grad
                kappas = np.dot(INFOS["fmw_normal_modes"][normal_mode], gradient.T)

                # writes kappa to result string
                for i, k in enumerate(kappas):
                    if k**2 > pthresh:
                        kappa_str_list.append("%3i %3i %5i % .5e\n" % (imult + 1, i + 1, int(normal_mode), k))
                        nkappa += 1
            start += nsi

    # ------------------------ lambda --------------------------
    lam = 0
    nlambda = 0
    lambda_str_list = []

    if INFOS["ana_nac"]:
        for i, sti in enumerate(itnmstates(INFOS["states"])):
            imult, istate, ims = sti

            if ims != (imult - 1) / 2.0:
                continue

            for j, stj in enumerate(itnmstates(INFOS["states"])):
                jmult, jstate, jms = stj

                if jms != (jmult - 1) / 2.0:
                    continue

                if i >= j:
                    continue

                if imult != jmult:
                    continue

                if ims != jms:
                    continue

                nacvector = QMout_eq.nacdr[i, j].flat

                # runs through normal modes
                for normal_mode in INFOS["fmw_normal_modes"].keys():
                    # calculates lambd from normal modes and grad
                    dE = (QMout_eq.h[j][j] - QMout_eq.h[i][i]).real
                    # lambd = sum([INFOS["fmw_normal_modes"][normal_mode][ixyz] * nacvector[ixyz] for ixyz in r3N]) * dE
                    lambd = np.dot(INFOS["fmw_normal_modes"][normal_mode], nacvector) * dE

                    # writes lambd to result string
                    if lambd**2 > pthresh:
                        # lambda_str_list.append('%3i %3i %5i % .5e\n' % (imult, istate, int(normal_mode), lambd))
                        lambda_str_list.append("%3i %3i %3i %3i % .5e\n" % (imult, istate, jstate, int(normal_mode), lambd))
                        nlambda += 1

    # ------------------------ numerical kappas and lambdas --------------------------

    if not (INFOS["ana_nac"] and INFOS["ana_grad"]):
        if "displacements" not in INFOS:
            print('No displacement info found in "displacements.json"!')
            sys.exit(1)

        if not INFOS["ana_nac"] and not INFOS["ana_grad"]:
            whatstring = "kappas and lambdas"
        elif not INFOS["ana_grad"]:
            whatstring = "kappas"
        elif not INFOS["ana_nac"]:
            whatstring = "lambdas"

        # running through all normal modes
        for normal_mode, v in INFOS["normal_modes"].items():
            twosided = False

            # get pos displacement
            pos_displ_mag = INFOS["displacement_magnitudes"][normal_mode]

            # get hamiltonian & overlap matrix from QM.out
            path = os.path.join(INFOS["paths"][str(normal_mode) + "p"], "QM.out")
            # requests = ["h", "overlap"]
            QMout_pos = QMout(path, INFOS["states"], len(INFOS["atoms"]), 0)

            # check diagonal of S & print warning
            INFOS["problematic_mults"] = check_overlap_diagonal(QMout_pos.overlap, INFOS["states"], normal_mode, "p")

            # calculate displacement matrix
            # pos_W_dQi = calculate_W_dQi(pos_H, pos_S, e_ref)

            # Check for two-sided differentiation
            if str(normal_mode) + "n" in INFOS["displacements"]:
                twosided = True
                # get neg displacement
                neg_displ_mag = INFOS["displacement_magnitudes"][normal_mode]

                # get hamiltonian & overlap matrix from QM.out
                path = os.path.join(INFOS["paths"][str(normal_mode) + "n"], "QM.out")
                QMout_neg = QMout(path, INFOS["states"], len(INFOS["atoms"]), 0)

                # check diagonal of S & print warning if wanted
                INFOS["problematic_mults"].update(check_overlap_diagonal(QMout_neg.overlap, INFOS["states"], normal_mode, "n"))

                # calculate displacement matrix

            # Loop over multiplicities to get kappas and lambdas
            for imult in filter(lambda x: INFOS["states"][x] != 0, range(len(INFOS["states"]))):
                # checking problematic states
                if INFOS["ignore_problematic_states"]:
                    if str(normal_mode) + "p" in INFOS["problematic_mults"]:
                        if INFOS["problematic_mults"][str(normal_mode) + "p"] == imult + 1:
                            print("Not producing %s for normal mode: %s" % (whatstring, normal_mode))
                            continue
                    if str(normal_mode) + "n" in INFOS["problematic_mults"]:
                        if twosided and INFOS["problematic_mults"][str(normal_mode) + "n"] == imult + 1:
                            print(
                                "! Not producing %s for multiplicity %i for normal mode: %s"
                                % (whatstring, imult + 1, normal_mode)
                            )
                            continue

                # partition matrices
                pos_H = partition_matrix(QMout_pos.h, imult + 1, INFOS["states"])
                pos_S = partition_matrix(QMout_pos.overlap, imult + 1, INFOS["states"])
                pos_partition = calculate_W_dQi(pos_H, pos_S, e_ref)
                # pos_partition = partition_matrix(pos_W_dQi, imult + 1, INFOS["states"])
                if twosided:
                    neg_H = partition_matrix(QMout_neg.h, imult + 1, INFOS["states"])
                    neg_S = partition_matrix(QMout_neg.overlap, imult + 1, INFOS["states"])
                    neg_partition = calculate_W_dQi(neg_H, neg_S, e_ref)
                    # neg_partition = partition_matrix(neg_W_dQi, imult + 1, INFOS["states"])
                partition_length = len(pos_partition)

                # get lambdas and kappas
                for i in range(partition_length):
                    if not INFOS["ana_grad"]:
                        if not twosided:
                            kappa = pos_partition[i][i].real / pos_displ_mag
                        else:
                            kappa = (pos_partition[i][i] - neg_partition[i][i]).real / (pos_displ_mag + neg_displ_mag)
                        if kappa**2 > pthresh:
                            kappa_str_list.append("%3i %3i %5i % .5e\n" % (imult + 1, i + 1, int(normal_mode), kappa))
                            nkappa += 1

                    if not INFOS["ana_nac"]:
                        for j in range(partition_length):
                            if i >= j:
                                continue
                            if not twosided:
                                lam = pos_partition[i][j].real / pos_displ_mag
                            else:
                                lam = (pos_partition[i][j] - neg_partition[i][j]).real / (pos_displ_mag + neg_displ_mag)
                            if lam**2 > pthresh:
                                lambda_str_list.append(
                                    "%3i %3i %3i %3i % .5e\n" % (imult + 1, i + 1, j + 1, int(normal_mode), lam)
                                )
                                nlambda += 1

    # --------------- GAMMA --------------
    # approximation from second order central
    gamma_str_list = []
    if "gammas" in INFOS and INFOS["gammas"] == "second order central":
        if "displacements" not in INFOS:
            print('No displacement info found in "displacements.json"!')
            sys.exit(1)
        # running through all normal modes
        for normal_mode, v in INFOS["normal_modes"].items():
            # Check for two-sided differentiation
            if not str(normal_mode) + "n" in INFOS["displacements"]:
                break

            # get pos displacement
            pos_displ_mag = INFOS["displacement_magnitudes"][normal_mode]

            # get hamiltonian & overlap matrix from QM.out
            path = os.path.join(INFOS["paths"][str(normal_mode) + "p"], "QM.out")
            requests = ["h", "overlap"]
            pos_H, pos_S = read_QMout(path, INFOS["nstates"], len(INFOS["atoms"]), requests).values()

            # check diagonal of S & print warning
            INFOS["problematic_mults"] = check_overlap_diagonal(pos_S, INFOS["states"], normal_mode, "p")

            # calculate displacement matrix
            pos_W_dQi = calculate_W_dQi(pos_H, pos_S, e_ref)

            # get neg displacement
            neg_displ_mag = INFOS["displacement_magnitudes"][normal_mode]

            # get hamiltonian & overlap matrix from QM.out
            path = os.path.join(INFOS["paths"][str(normal_mode) + "n"], "QM.out")
            requests = ["h", "overlap"]
            neg_H, neg_S = read_QMout(path, INFOS["nstates"], len(INFOS["atoms"]), requests).values()

            # check diagonal of S & print warning if wanted
            INFOS["problematic_mults"].update(check_overlap_diagonal(neg_S, INFOS["states"], normal_mode, "n"))

            # calculate displacement matrix
            neg_W_dQi = calculate_W_dQi(neg_H, neg_S, e_ref)

            # Loop over multiplicities to get kappas and lambdas
            for imult in range(len(INFOS["states"])):
                # checking problematic states
                if INFOS["ignore_problematic_states"]:
                    if str(normal_mode) + "p" in INFOS["problematic_mults"]:
                        if INFOS["problematic_mults"][str(normal_mode) + "p"] == imult + 1:
                            print("Not producing %s for normal mode: %s" % (whatstring, normal_mode))
                            continue
                    if str(normal_mode) + "n" in INFOS["problematic_mults"]:
                        if twosided and INFOS["problematic_mults"][str(normal_mode) + "n"] == imult + 1:
                            print(
                                "! Not producing %s for multiplicity %i for normal mode: %s"
                                % (whatstring, imult + 1, normal_mode)
                            )
                            continue

                # partition matrices
                eq_partition = partition_matrix(QMout_eq.h, imult + 1, INFOS["states"])
                pos_partition = partition_matrix(pos_W_dQi, imult + 1, INFOS["states"])
                if twosided:
                    neg_partition = partition_matrix(neg_W_dQi, imult + 1, INFOS["states"])
                partition_length = len(pos_partition)

                # get lambdas and kappas
                for i in range(partition_length):
                    if not INFOS["ana_nac"]:
                        for j in range(partition_length):
                            if i > j:
                                continue
                            omeg = (pos_partition[i][j] - 2 * eq_partition[i][j] + neg_partition[i][j]).real / (
                                pos_displ_mag + neg_displ_mag
                            ) ** 2
                            if omeg**2 > pthresh:
                                gamma_str_list.append(
                                    "%3i %3i %3i %3i %3i % .5e\n"
                                    % (imult + 1, i + 1, j + 1, int(normal_mode), int(normal_mode), omeg)
                                )
    # calculate gammas from approximatin the hessian through diabatized gradients at displacements and equilibrium geometry
    print("gammas", INFOS["gammas"])
    check_gamma = {}
    if "gammas" in INFOS and INFOS["gammas"] == "hessian from diabatized gradients":
        # SCHEDULE:
        for normal_mode in INFOS["normal_modes"].keys():
            path = os.path.join(INFOS["paths"][str(normal_mode) + "p"], "QM.out")
            QMout_pos = QMout(path, INFOS["states"], len(INFOS["atoms"]), 0)

            path = os.path.join(INFOS["paths"][str(normal_mode) + "n"], "QM.out")
            QMout_neg = QMout(path, INFOS["states"], len(INFOS["atoms"]), 0)

            displ_mag = INFOS["displacement_magnitudes"][normal_mode]

            # Loop over multiplicities to get kappas and lambdas
            start = 0
            for imult, nsi in enumerate(INFOS["states"]):
                if nsi == 0:
                    continue
                part_grad_pos = QMout_pos.grad[start : start + nsi, ...].reshape((nsi, -1))
                part_ovl = QMout_pos.overlap[start : start + nsi, start : start + nsi]

                # do loewdin orthonorm. on overlap matrix
                part_ovl = loewdin_orthonormalization(part_ovl)

                part_ovl = phase_correction(part_ovl)

                part_grad_neg = QMout_neg.grad[start : start + nsi, ...].reshape((nsi, -1))
                part_ovl = QMout_pos.overlap[start : start + nsi, start : start + nsi]

                # do loewdin orthonorm. on overlap matrix
                part_ovl = loewdin_orthonormalization(part_ovl)

                part_ovl = phase_correction(part_ovl)

                for derivate_mode in INFOS["normal_modes"].keys():
                    nac_from_grad_pos = np.diag(np.einsum("k,ik->i", INFOS["fmw_normal_modes"][derivate_mode], part_grad_pos))
                    diab_nac_pos = np.diag(part_ovl @ nac_from_grad_pos @ part_ovl.T)

                    nac_from_grad_neg = np.diag(np.einsum("k,ik->i", INFOS["fmw_normal_modes"][derivate_mode], part_grad_neg))
                    diab_nac_neg = np.diag(part_ovl @ nac_from_grad_neg @ part_ovl.T)

                    gammas = (diab_nac_pos - diab_nac_neg).real / (
                        displ_mag * 4.0
                    )  # gammas are 0.5*f''(x)/dQidQj (https://doi.org/10.1142/9789812565464_0007)

                    if normal_mode == derivate_mode:
                        gammas -= INFOS["frequencies"][normal_mode] * 0.5

                    if normal_mode == derivate_mode and start == 0:
                        print(
                            "sanity check: difference in S0 frequency of mode:",
                            normal_mode,
                            f"{(gammas[0] * 2)/4.55633590401805e-06: .1f}cm-1",
                        )
                    if start == 0:
                        gammas[0] = 0.0

                    check_gamma[(imult, normal_mode, derivate_mode)] = gammas
                    if (imult, derivate_mode, normal_mode) in check_gamma and normal_mode != derivate_mode:
                        gammas = (
                            check_gamma[(imult, normal_mode, derivate_mode)] + check_gamma[(imult, derivate_mode, normal_mode)]
                        ) * 0.5
                    elif normal_mode != derivate_mode:
                        continue

                    # print(gammas[0], [f"{i:3d}" for i in np.where(gammas > 4.55633590401805e-06)[0] /])
                    gamma_str_list.extend(
                        list(
                            map(
                                lambda i: f"{imult + 1:3d} {i+1:3d} {i+1:3d} {int(normal_mode):3d} {int(derivate_mode):3d} {gammas[i]: .7e}\n",
                                np.where(gammas > 4.55633590401805e-06)[0],
                            )
                        )
                    )
                start += nsi

        # get gradient and overlap
        # gradient -> approximate nac -> diagonalize
        # approximate hessian

    # add results to template string
    lvc_template_content += "kappa\n"
    lvc_template_content += "%i\n" % (nkappa)
    lvc_template_content += "".join(sorted(kappa_str_list))

    lvc_template_content += "lambda\n"
    lvc_template_content += "%i\n" % (nlambda)
    lvc_template_content += "".join(sorted(lambda_str_list))

    if len(gamma_str_list) != 0:
        lvc_template_content += "gamma\n"
        lvc_template_content += "%i\n" % (len(gamma_str_list))
        lvc_template_content += "".join(sorted(gamma_str_list))

    # ----------------------- matrices ------------------------------
    if INFOS["soc"]:
        if INFOS["soc_file"]:
            print("Reading SOCs from file:", INFOS["soc_file"])
            print("Beware that it will be not checked for complete compatability!")
            QMout_soc = QMout(filepath=INFOS["soc_file"], states=INFOS["states"], natom=len(INFOS["atoms"]), npc=0)
            print(
                "sanity check: RSMD of adiabatic energies:", np.sqrt(np.mean((np.diag(QMout_soc.h) - np.diag(QMout_eq.h)) ** 2))
            )
            lvc_template_content += LVC_complex_mat("SOC", QMout_soc.h, deldiag=True)
        else:
            lvc_template_content += LVC_complex_mat("SOC", QMout_eq.h, deldiag=True)
    lvc_template_content += LVC_complex_mat("DMX", QMout_eq.dm[0])
    lvc_template_content += LVC_complex_mat("DMY", QMout_eq.dm[1])
    lvc_template_content += LVC_complex_mat("DMZ", QMout_eq.dm[2])

    # --------------------- multipolar fit ---------------------------
    if "multipolar_fit" in QMout_eq:
        fit = QMout_eq.multipolar_fit
        settings = QMout_eq.notes["multipolar_fit"] if "multipolar_fit" in QMout_eq.notes else ""
        mat_string = ""
        n_entries = 0
        for (s_i, s_j), fit in QMout_eq["multipolar_fit"].items():
            if s_i > s_j:
                continue
            for atom in range(len(INFOS["atoms"])):  # get mults
                n_entries += 1
                nums = "".join(map(lambda x: f"{x: 12.8f}", fit[atom, :]))
                # print(f"{s_i.S} {s_i.N + 1:2} {s_j.N + 1:2} {atom:3}    {nums}\n")
                mat_string += f"{s_i.S + 1} {s_i.N:2} {s_j.N:2} {atom:3}    {nums}\n"
        # for m_i, n_i in enumerate(INFOS["states"]):  # get mults
        #     fit_block = partition_matrix(fit, m_i + 1, INFOS["states"])
        #     for s_i in range(n_i):
        #         for s_j in range(n_i):
        #             if s_i > s_j:
        #                 continue
        #             for atom in range(len(INFOS["atoms"])):  # get mults
        #                 n_entries += 1
        #                 nums = "".join(map(lambda x: f"{x: 12.8f}", fit_block[s_i, s_j, atom, :]))
        #                 mat_string += f"{m_i + 1} {s_i + 1:2} {s_j + 1:2} {atom:3}    {nums}\n"
        lvc_template_content += f"Multipolar Density Fit {settings}\n{n_entries}\n{mat_string}"

    # -------------------- write to file ----------------------------
    print("\nFinished!\nLVC parameters written to file: LVC.template\n")
    lvc_template = open("LVC.template", "w")
    lvc_template.write(lvc_template_content)
    lvc_template.close()


# ======================================================================= #
# ======================================================================= #
# ======================================================================= #


def main():
    """Main routine"""
    script_name = sys.argv[0].split("/")[-1]

    usage = """python %s""" % (script_name)

    displaywelcome()
    is_other_dir = len(sys.argv) == 2 and os.path.isdir(sys.argv[1])
    # load INFOS object from file
    displacement_info_filename = os.path.join(sys.argv[1], "displacements.json") if is_other_dir else "displacements.json"

    try:
        with open(displacement_info_filename, "r") as displacement_info:
            INFOS = json_load_byteified(displacement_info)
            displacement_info.close()
    except IOError:
        print("IOError during opening readable %s - file. Quitting." % (displacement_info_filename))
        quit(1)

    # set manually for old calcs
    # INFOS['ignore_problematic_states'] = True
    if is_other_dir:
        for k, v in INFOS["paths"].items():
            INFOS["paths"][k] = os.path.join(sys.argv[1], v)

    write_LVC_template(INFOS)


# ======================================================================= #


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nCtrl+C occured. Exiting.\n")
        sys.exit()
