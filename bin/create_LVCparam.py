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
from scipy.linalg import fractional_matrix_power
from itertools import starmap, chain

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


version = "4.0"
versionneeded = [0.2, 1.0, 2.0, 2.1, float(version)]
versiondate = datetime.date(2024, 4, 1)


# ======================================================================= #

pthresh = 1.0e-5**2

# ======================================================================= #


def displaywelcome():
    lines = [
        f"Compute LVC parameters",
        "",
        f"Authors: Severin Polonius, Sebastian Mai, Simon Kropf",
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
    # S = A.T @ A
    # e, v = np.linalg.eigh(S)
    # idx = e > 1e-15
    # S_lo = np.dot(v[:, idx] / np.sqrt(e[idx]), v[:, idx].conj().T)
    # return A @ S_lo
    return A @ fractional_matrix_power(A.T @ A, -0.5)


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

def phase_correction_do_nothing(matrix):
    return matrix.real.copy()


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
    # return U @ H @ U.T


# ======================================================================= #


def write_LVC_template(INFOS, template_name):
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
    path = os.path.join(INFOS["paths"]["000_eq"], "QM.out")
    print("reading QMout_eq at:", path)
    # QMout_eq = read_QMout(path, INFOS["nstates"], len(INFOS["atoms"]), requests)
    flags = {1, 2}
    if INFOS["ana_grad"]:
        flags.add(3)
    if INFOS["ana_nac"]:
        flags.add(5)
    if INFOS["multipolar_fit"]:
        flags.add(22)
    QMout_eq = QMout(path, INFOS["states"], len(INFOS["atoms"]), npc=0, flags=flags)
    lvc_template_content += "charge " + " ".join(map(str, QMout_eq.charges)) + "\n"
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
    kappas_dict = {}

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

                kappas_dict[(imult, int(normal_mode))] = {}
                # writes kappa to result string
                for i, k in enumerate(kappas):
                    kappas_dict[(imult, int(normal_mode))][i] = k
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

    if not INFOS["ana_nac"] or not INFOS["ana_grad"]:
        if "displacements" not in INFOS:
            print('No displacement info found in "displacements.json"!')
            sys.exit(1)
        whatstring = "kappas and lambdas"

        # running through all normal modes
        for normal_mode, v in INFOS["normal_modes"].items():
            twosided = False

            # get pos displacement
            pos_displ_mag = INFOS["displacement_magnitudes"][normal_mode]

            # get hamiltonian & overlap matrix from QM.out
            path = os.path.join(INFOS["paths"][f"{normal_mode:>03s}_{'p'}"], "QM.out")
            # requests = ["h", "overlap"]
            print("reading displaced QMout at:", path)
            QMout_pos = QMout(path, INFOS["states"], len(INFOS["atoms"]), 0)

            # check diagonal of S & print warning
            INFOS["problematic_mults"] = check_overlap_diagonal(QMout_pos.overlap, INFOS["states"], normal_mode, "p")

            # calculate displacement matrix
            # pos_W_dQi = calculate_W_dQi(pos_H, pos_S, e_ref)

            # Check for two-sided differentiation
            if f"{normal_mode:>03s}_{'n'}" in INFOS["paths"]:
                twosided = True
                # get neg displacement
                neg_displ_mag = INFOS["displacement_magnitudes"][normal_mode]

                # get hamiltonian & overlap matrix from QM.out
                path = os.path.join(INFOS["paths"][f"{normal_mode:>03s}_{'n'}"], "QM.out")
                print("reading displaced QMout at:", path)
                QMout_neg = QMout(path, INFOS["states"], len(INFOS["atoms"]), 0)

                # check diagonal of S & print warning if wanted
                INFOS["problematic_mults"].update(check_overlap_diagonal(QMout_neg.overlap, INFOS["states"], normal_mode, "n"))

                # calculate displacement matrix

            # Loop over multiplicities to get kappas and lambdas
            # Loop over multiplicities
            start = 0
            for imult, nsi in enumerate(INFOS["states"]):
                if nsi == 0:
                    continue
                part_h_pos = QMout_pos.h[start : start + nsi, start : start + nsi].real
                part_ovl_pos = QMout_pos.overlap[start : start + nsi, start : start + nsi].real
                part_ovl_pos = loewdin_orthonormalization(part_ovl_pos)
                part_ovl_pos = phase_correction(part_ovl_pos)
                pos_partition = part_ovl_pos @ part_h_pos @ part_ovl_pos.T

                part_h_neg = QMout_neg.h[start : start + nsi, start : start + nsi].real
                part_ovl_neg = QMout_neg.overlap[start : start + nsi, start : start + nsi].real
                part_ovl_neg = loewdin_orthonormalization(part_ovl_neg)
                part_ovl_neg = phase_correction(part_ovl_neg)
                neg_partition = part_ovl_neg @ part_h_neg @ part_ovl_neg.T
                # checking problematic states
                if INFOS["ignore_problematic_states"]:
                    if str(normal_mode) + "p" in INFOS["problematic_mults"]:
                        if INFOS["problematic_mults"][f"{normal_mode:>03s}_{'p'}"] == imult + 1:
                            print("Not producing %s for normal mode: %s" % (whatstring, normal_mode))
                            continue
                    if str(normal_mode) + "n" in INFOS["problematic_mults"]:
                        if twosided and INFOS["problematic_mults"][f"{normal_mode:>03s}_{'n'}"] == imult + 1:
                            print(
                                "! Not producing %s for multiplicity %i for normal mode: %s"
                                % (whatstring, imult + 1, normal_mode)
                            )
                            continue

                # partition matrices
                # pos_H = partition_matrix(QMout_pos.h, imult + 1, INFOS["states"])
                # pos_S = partition_matrix(QMout_pos.overlap, imult + 1, INFOS["states"])
                # pos_partition = calculate_W_dQi(pos_H, pos_S, e_ref)
                # pos_partition = partition_matrix(pos_W_dQi, imult + 1, INFOS["states"])
                # if twosided:
                #     neg_H = partition_matrix(QMout_neg.h, imult + 1, INFOS["states"])
                #     neg_S = partition_matrix(QMout_neg.overlap, imult + 1, INFOS["states"])
                #     neg_partition = calculate_W_dQi(neg_H, neg_S, e_ref)
                #     # neg_partition = partition_matrix(neg_W_dQi, imult + 1, INFOS["states"])
                partition_length = len(pos_partition)
                if not INFOS["ana_grad"]:
                    kappas_dict[(imult, int(normal_mode))] = {}

                # get lambdas and kappas
                for i in range(partition_length):
                    if not INFOS["ana_grad"]:
                        if not twosided:
                            kappa = pos_partition[i][i].real / pos_displ_mag
                        else:
                            kappa = (pos_partition[i][i] - neg_partition[i][i]).real / (pos_displ_mag + neg_displ_mag)
                        if kappa**2 > pthresh:
                            kappas_dict[(imult, int(normal_mode))][i] = kappa
                            kappa_str_list.append("%3i %3i %5i % .5e\n" % (imult + 1, i + 1, int(normal_mode), kappa))
                            nkappa += 1

                    if not INFOS["ana_nac"]:
                        for j in range(partition_length):
                            if i >= j:
                                continue
                            if not twosided:
                                lam = pos_partition[i][j].real / pos_displ_mag
                            else:
                                lam = pos_partition[i][j].real / (pos_displ_mag + neg_displ_mag) - neg_partition[i][j].real / (
                                    pos_displ_mag + neg_displ_mag
                                )
                            if lam**2 > pthresh:
                                lambda_str_list.append(
                                    "%3i %3i %3i %3i % .5e\n" % (imult + 1, i + 1, j + 1, int(normal_mode), lam)
                                )
                                nlambda += 1
                start += nsi

    # ------------------------ lambda_soc --------------------------

    if INFOS["lambda_soc"]:
        print("calculating derviatives of SOCs in linear approx")
        if "displacements" not in INFOS:
            print('No displacement info found in "displacements.json"!')
            sys.exit(1)
        lambda_soc_str_list = []
        # running through all normal modes
        for normal_mode, v in INFOS["normal_modes"].items():
            twosided = False

            # get pos displacement
            pos_displ_mag = INFOS["displacement_magnitudes"][normal_mode]

            # get hamiltonian & overlap matrix from QM.out
            path = os.path.join(INFOS["paths"][f"{normal_mode:>03s}_{'p'}"], "QM.out")
            QMout_pos = QMout(path, INFOS["states"], len(INFOS["atoms"]), 0, flags={1, 6})

            # check diagonal of S & print warning
            INFOS["problematic_mults"] = check_overlap_diagonal(QMout_pos.overlap, INFOS["states"], normal_mode, "p")

            # Check for two-sided differentiation
            if f"{normal_mode:>03s}_{'n'}" in INFOS["paths"]:
                twosided = True
                # get neg displacement
                neg_displ_mag = INFOS["displacement_magnitudes"][normal_mode]

                # get hamiltonian & overlap matrix from QM.out
                path = os.path.join(INFOS["paths"][f"{normal_mode:>03s}_{'n'}"], "QM.out")
                QMout_neg = QMout(path, INFOS["states"], len(INFOS["atoms"]), 0, flags={1, 6})

                # check diagonal of S & print warning if wanted
                INFOS["problematic_mults"].update(check_overlap_diagonal(QMout_neg.overlap, INFOS["states"], normal_mode, "n"))

                # calculate displacement matrix

            # Loop over multiplicities to get kappas and lambdas
            # Loop over multiplicities
            start = 0
            for imult, nsi in enumerate(INFOS["states"]):
                if nsi != 0:
                    for _ in range(imult + 1):
                        QMout_pos.h[start : start + nsi, start : start + nsi] = 0.0 + 0.0j
                        if twosided:
                            QMout_neg.h[start : start + nsi, start : start + nsi] = 0.0 + 0.0j
                        start += nsi

            part_ovl_pos = loewdin_orthonormalization(QMout_pos.overlap)
            part_ovl_pos = phase_correction(part_ovl_pos)
            pos_partition = part_ovl_pos @ QMout_pos.h @ part_ovl_pos.T

            if twosided:
                part_ovl_neg = loewdin_orthonormalization(QMout_neg.overlap)
                part_ovl_neg = phase_correction(part_ovl_neg)
                neg_partition = part_ovl_neg @ QMout_neg.h @ part_ovl_neg.T

            if twosided:
                fac = pos_displ_mag + neg_displ_mag
                lambda_soc = pos_partition / fac - neg_partition / fac
            else:
                lambda_soc = pos_partition / pos_displ_mag

            lambda_soc = np.stack((lambda_soc.real, lambda_soc.imag), axis=-1)

            lambda_soc_str_list.extend(
                list(
                    starmap(
                        lambda i, j, c: f"{i+1:3d} {j+1:3d} {int(normal_mode):3d} {'RI'[c]} {lambda_soc[i,j,c]: .7e}\n",
                        filter(lambda x: x[0] < x[1], zip(*np.where(abs(lambda_soc) > 1e-7))),
                    )
                )
            )

    # --------------- GAMMA --------------
    gamma_str_list = []
    # five point stencil
    if "gammas" in INFOS and INFOS["gammas"] == "five-point stencil":
        if "displacements" not in INFOS:
            print('No displacement info found in "displacements.json"!')
            sys.exit(1)
        # running through all normal modes
        for normal_mode in INFOS["gamma_normal_modes"]:
            displ_mag = INFOS["displacement_magnitudes_gamma"][normal_mode]
            m = 2.0 if f"{normal_mode:>03s}_{'p2'}" in INFOS["paths_gamma"] else 1.0
            if normal_mode + "_2" in INFOS["displacement_magnitudes_gamma"]:
                m = INFOS["displacement_magnitudes_gamma"][normal_mode + "_2"] / displ_mag
            print(displ_mag, m)
            freq = INFOS["frequencies"][normal_mode]
            necessary_displ = math.sqrt(2 * 2 * 4.55633590401805e-06 / freq)  # displ necessary for energy difference of 2cm-1
            if (necessary_displ - displ_mag * m) / necessary_displ > 0.1:  # 10% tolerance for
                print(
                    f"skipping normal mode {normal_mode} ({freq/4.55633590401805e-06: 0.1f}cm-1)! Insufficient maximum displacment {displ_mag * m: .2f} vs {necessary_displ: .2f}!"
                )
                continue

            path = os.path.join(INFOS["paths_gamma"][f"{normal_mode:>03s}_{'p'}"], "QM.out")
            QMout_pos = QMout(path, INFOS["states"], len(INFOS["atoms"]), 0)
            print(path)

            if f"{normal_mode:>03s}_{'p2'}" not in INFOS["paths_gamma"]:
                path = os.path.join(INFOS["paths_gamma"][f"{normal_mode:>03s}_{'p'}"], "QM.out")
            else:
                path = os.path.join(INFOS["paths_gamma"][f"{normal_mode:>03s}_{'p2'}"], "QM.out")
            QMout_pos2 = QMout(path, INFOS["states"], len(INFOS["atoms"]), 0)
            print(path)

            path = os.path.join(INFOS["paths_gamma"][f"{normal_mode:>03s}_{'n'}"], "QM.out")
            QMout_neg = QMout(path, INFOS["states"], len(INFOS["atoms"]), 0)
            print(path)

            if f"{normal_mode:>03s}_{'n2'}" not in INFOS["paths_gamma"]:
                path = os.path.join(INFOS["paths_gamma"][f"{normal_mode:>03s}_{'n'}"], "QM.out")
            else:
                path = os.path.join(INFOS["paths_gamma"][f"{normal_mode:>03s}_{'n2'}"], "QM.out")
            QMout_neg2 = QMout(path, INFOS["states"], len(INFOS["atoms"]), 0)
            print(path)

            # Loop over multiplicities
            start = 0
            for imult, nsi in enumerate(INFOS["states"]):
                if nsi == 0:
                    continue
                states = INFOS["gamma_selected_states"][str(imult)]
                # states = list(range(INFOS["states"][imult]))
                part_h_pos = QMout_pos.h[start : start + nsi, start : start + nsi].real
                part_ovl_pos = QMout_pos.overlap[start : start + nsi, start : start + nsi].real
                part_ovl_pos = loewdin_orthonormalization(part_ovl_pos)
                part_ovl_pos = phase_correction(part_ovl_pos)
                diab_h_pos = part_ovl_pos @ part_h_pos @ part_ovl_pos.T

                part_h_pos2 = QMout_pos2.h[start : start + nsi, start : start + nsi].real
                part_ovl_pos2 = QMout_pos2.overlap[start : start + nsi, start : start + nsi].real
                part_ovl_pos2 = loewdin_orthonormalization(part_ovl_pos2)
                part_ovl_pos2 = phase_correction(part_ovl_pos2)
                diab_h_pos2 = part_ovl_pos2 @ part_h_pos2 @ part_ovl_pos2.T

                part_h_neg = QMout_neg.h[start : start + nsi, start : start + nsi].real
                part_ovl_neg = QMout_neg.overlap[start : start + nsi, start : start + nsi].real
                part_ovl_neg = loewdin_orthonormalization(part_ovl_neg)
                part_ovl_neg = phase_correction(part_ovl_neg)
                diab_h_neg = part_ovl_neg @ part_h_neg @ part_ovl_neg.T

                part_h_neg2 = QMout_neg2.h[start : start + nsi, start : start + nsi].real
                part_ovl_neg2 = QMout_neg2.overlap[start : start + nsi, start : start + nsi].real
                part_ovl_neg2 = loewdin_orthonormalization(part_ovl_neg2)
                part_ovl_neg2 = phase_correction(part_ovl_neg2)
                diab_h_neg2 = part_ovl_neg2 @ part_h_neg2 @ part_ovl_neg2.T

                part_eq_h = QMout_eq.h[start : start + nsi, start : start + nsi].real

                # correct for kappas already included
                k_m_n = kappas_dict[(imult, int(normal_mode))]
                for s, k in k_m_n.items():
                    diab_h_neg2[s, s] += 2 * k * displ_mag
                    diab_h_pos2[s, s] -= 2 * k * displ_mag
                    diab_h_neg[s, s] += k * displ_mag
                    diab_h_pos[s, s] -= k * displ_mag

                # five-point stencil (m=2): (-E2p + 16*E1p - 30*E0 + 16*E1n - E2n) / 12h**2
                # five-point stencil for variable n f(x+mh) and f(x+mh):
                #    (-Emp + m^4 Ep - (2m^4-2)E0 + m^4 En - Emn) / ((m^4 - m^2)h^2)
                # m4 = m**4
                # neg_contr = (diab_h_pos2 + (2*m4-2) * part_eq_h + diab_h_neg2) / ((m4-m**2) * displ_mag**2)
                # pos_contr = ((m4) * diab_h_pos + (m4) * diab_h_neg) / ((m4-m**2) * displ_mag**2)
                # gammas = pos_contr - neg_contr
                # gammas = gammas * 1e-4

                # Test mean of all four points
                gammas = (m**2 * (diab_h_pos + diab_h_neg - 2 * part_eq_h) + diab_h_pos2 + diab_h_neg2 - 2 * part_eq_h) / (
                    2 * m**2 * displ_mag**2
                )
                # test only outer points
                # gammas = (diab_h_pos2 + diab_h_neg2 - 2 * part_eq_h) / (m**2 * displ_mag**2)
                # test with more weight on outer points
                # gammas = (m**2*(diab_h_pos + diab_h_neg - 2* part_eq_h) + 3*diab_h_pos2 + 3*diab_h_neg2 - 6*part_eq_h)/(4*m**2*displ_mag**2)

                # gammas = 0.5 * (np.diag(gammas) - INFOS["frequencies"][normal_mode])
                gammas = 0.5 * gammas
                np.einsum("ii->i", gammas)[...] -= 0.5 * INFOS["frequencies"][normal_mode]
                # gammas = 0.5 * (np.diag(gammas))

                if start == 0:
                    freq = INFOS["frequencies"][normal_mode]
                    gfreq_dev = (gammas[0, 0] * 2) / 4.55633590401805e-06
                    dev_per = gfreq_dev / (freq / 4.55633590401805e-06)
                    print(
                        "sanity check: difference in S0 frequency of mode:",
                        normal_mode,
                        f"{gfreq_dev: .1f}cm-1",
                        f"{dev_per:.1%}",
                    )
                    gammas[0] = 0
                gammas = gammas[states, ...][..., states]
                print(
                    "Problematic gammas:",
                    [
                        (normal_mode, imult, n, f"{(g * 2 + INFOS['frequencies'][normal_mode])/4.55633590401805e-06: .1f}cm-1")
                        for n, g in enumerate(np.diag(gammas))
                        if (g * 2 + INFOS["frequencies"][normal_mode]) / 4.55633590401805e-06 < 0
                    ],
                )

                gamma_str_list.extend(
                    list(
                        starmap(
                            lambda i, j: f"{imult + 1:3d} {states[i]+1:3d} {states[j]+1:3d} {int(normal_mode):3d} {int(normal_mode):3d} {gammas[i,j]: .7e}\n",
                            zip(*np.where(abs(gammas) > 4.55633590401805e-06)),
                        )
                    )
                )
                # gammas = np.diag(gammas)
                # gamma_str_list.extend(
                # list(
                # map(
                # lambda i: f"{imult + 1:3d} {states[i]+1:3d} {states[i]+1:3d} {int(normal_mode):3d} {int(normal_mode):3d} {gammas[i]: .7e}\n",
                # np.where(abs(gammas) > 4.55633590401805e-06)[0],
                # )
                # )
                # )
                start += nsi

    # approximation from second order central
    if "gammas" in INFOS and INFOS["gammas"] == "second order central":
        if "displacements" not in INFOS:
            print('No displacement info found in "displacements.json"!')
            sys.exit(1)
        # running through all normal modes
        for normal_mode in INFOS["gamma_normal_modes"]:
            freq = INFOS["frequencies"][normal_mode]
            path = os.path.join(INFOS["paths"][f"{normal_mode:>03s}_{'p'}"], "QM.out")
            QMout_pos = QMout(path, INFOS["states"], len(INFOS["atoms"]), 0)

            path = os.path.join(INFOS["paths"][f"{normal_mode:>03s}_{'n'}"], "QM.out")
            QMout_neg = QMout(path, INFOS["states"], len(INFOS["atoms"]), 0)

            displ_mag = INFOS["displacement_magnitudes"][normal_mode]

            # Loop over multiplicities
            start = 0
            for imult, nsi in enumerate(INFOS["states"]):
                if nsi == 0:
                    continue
                states = INFOS["gamma_selected_states"][str(imult)]
                part_h_pos = QMout_pos.h[start : start + nsi, start : start + nsi].real
                part_ovl_pos = QMout_pos.overlap[start : start + nsi, start : start + nsi].real

                # do loewdin orthonorm. on overlap matrix

                part_ovl_pos = loewdin_orthonormalization(part_ovl_pos)

                part_ovl_pos = phase_correction(part_ovl_pos)

                diab_h_pos = part_ovl_pos @ part_h_pos @ part_ovl_pos.T
                # diab_h_pos = calculate_W_dQi(part_h_pos, part_ovl_pos, e_ref)

                part_h_neg = QMout_neg.h[start : start + nsi, start : start + nsi].real
                part_ovl_neg = QMout_neg.overlap[start : start + nsi, start : start + nsi].real

                # do loewdin orthonorm. on overlap matrix
                part_ovl_neg = loewdin_orthonormalization(part_ovl_neg)

                part_ovl_neg = phase_correction(part_ovl_neg)

                diab_h_neg = part_ovl_neg @ part_h_neg @ part_ovl_neg.T
                # diab_h_neg = calculate_W_dQi(part_h_neg, part_ovl_neg, e_ref)

                part_eq_h = QMout_eq.h[start : start + nsi, start : start + nsi].real
                gammas = (diab_h_pos - 2 * part_eq_h + diab_h_neg) / (displ_mag) ** 2
                gammas = 0.5 * (np.diag(gammas) - INFOS["frequencies"][normal_mode])
                gammas = gammas[states]
                check = np.where(np.abs(gammas * 2) / freq > 0.5)[0]
                if len(check) > 0:
                    print(
                        f"WARNING: gammas wrong for states in {imult}",
                        [states[x] for x in check],
                        np.array2string(
                            (gammas[check] * 2) / 4.55633590401805e-06, formatter={"float": lambda x: f"{x: 9.1f}cm-1"}
                        ),
                        np.array2string(gammas[check] * 2 / freq, formatter={"float": lambda x: f"{x*100: 4.1f}%"}),
                    )
                if start == 0:
                    print(
                        "sanity check: difference in S0 frequency of mode:",
                        normal_mode,
                        f"{(gammas[0] * 2)/4.55633590401805e-06: .1f}cm-1",
                    )
                    gammas[0] = 0
                print(
                    "adding gammas for states ",
                    normal_mode,
                    imult,
                    [
                        [states[i], gammas[i] * 2 / freq]
                        for i in np.where((np.abs(gammas) > 4.55633590401805e-06) & (np.abs(gammas * 2) / freq < 0.5))[0]
                    ],
                )
                # gammas = np.where(np.abs(gammas * 2) / freq > 0.5, 0, gammas)

                gamma_str_list.extend(
                    list(
                        map(
                            lambda i: f"{imult + 1:3d} {states[i]+1:3d} {states[i]+1:3d} {int(normal_mode):3d} {int(normal_mode):3d} {gammas[i]: .7e}\n",
                            np.where(abs(gammas) > 4.55633590401805e-06)[0],
                        )
                    )
                )

                start = start + nsi

    # calculate gammas from approximatin the hessian through diabatized gradients at displacements and equilibrium geometry
    print("gammas", INFOS["gammas"])
    check_gamma = {}
    if "gammas" in INFOS and INFOS["gammas"] == "hessian from diabatized gradients":
        # SCHEDULE:
        for normal_mode in INFOS["gamma_normal_modes"]:
            freq = INFOS["frequencies"][normal_mode]
            path = os.path.join(INFOS["paths"][f"{normal_mode:>03s}_{'p'}"], "QM.out")
            QMout_pos = QMout(path, INFOS["states"], len(INFOS["atoms"]), 0)

            path = os.path.join(INFOS["paths"][f"{normal_mode:>03s}_{'n'}"], "QM.out")
            QMout_neg = QMout(path, INFOS["states"], len(INFOS["atoms"]), 0)

            displ_mag = INFOS["displacement_magnitudes"][normal_mode]

            # Loop over multiplicities to get kappas and lambdas
            start = 0
            for imult, nsi in enumerate(INFOS["states"]):
                if nsi == 0:
                    continue
                states = INFOS["gamma_selected_states"][str(imult)]

                # POS
                part_grad_pos = QMout_pos.grad[start : start + nsi, ...].reshape((nsi, -1))
                part_ovl_pos = QMout_pos.overlap[start : start + nsi, start : start + nsi].real

                nac_grad_pos = np.zeros((nsi, nsi, part_grad_pos.shape[-1]), dtype=float)
                np.einsum("iik->ik", nac_grad_pos)[...] += part_grad_pos[...]  # fill diag with grads

                # do loewdin orthonorm. on overlap matrix
                part_ovl_pos = loewdin_orthonormalization(part_ovl_pos)
                part_ovl_pos = phase_correction(part_ovl_pos)

                diab_grad_pos = np.einsum("in,nmk,im->ik", part_ovl_pos, nac_grad_pos, part_ovl_pos)

                # NEG
                part_grad_neg = QMout_neg.grad[start : start + nsi, ...].reshape((nsi, -1))
                part_ovl_neg = QMout_neg.overlap[start : start + nsi, start : start + nsi].real

                nac_grad_neg = np.zeros((nsi, nsi, part_grad_neg.shape[-1]), dtype=float)
                np.einsum("iik->ik", nac_grad_neg)[...] += part_grad_neg[...]  # fill diag with grads

                # do loewdin orthonorm. on overlap matrix
                part_ovl_neg = loewdin_orthonormalization(part_ovl_neg)
                part_ovl_neg = phase_correction(part_ovl_neg)

                diab_grad_neg = np.einsum("in,nmk,im->ik", part_ovl_neg, nac_grad_neg, part_ovl_neg)

                # diff_grad = diab_grad_pos - diab_grad_neg

                for derivate_mode in INFOS["gamma_normal_modes"]:
                    # nac_from_grad_pos = np.diag(np.einsum("k,ik->i", INFOS["fmw_normal_modes"][derivate_mode], part_grad_pos))
                    # diab_nac_pos = np.diag(part_ovl_pos @ nac_from_grad_pos @ part_ovl_pos.T)

                    # nac_from_grad_neg = np.diag(np.einsum("k,ik->i", INFOS["fmw_normal_modes"][derivate_mode], part_grad_neg))
                    # diab_nac_neg = np.diag(part_ovl_neg @ nac_from_grad_neg @ part_ovl_neg.T)

                    # diff grad from xyz to Q
                    # gammas are 0.5*f''(x)/dQidQj (https://doi.org/10.1142/9789812565464_0007)
                    diab_grad_pos_Q = np.einsum("k,ik->i", INFOS["fmw_normal_modes"][derivate_mode], diab_grad_pos) / (
                        2.0 * displ_mag
                    )
                    diab_grad_neg_Q = np.einsum("k,ik->i", INFOS["fmw_normal_modes"][derivate_mode], diab_grad_neg) / (
                        2.0 * displ_mag
                    )
                    gammas = diab_grad_pos_Q - diab_grad_neg_Q
                    gammas = 0.5 * gammas[states]

                    # gammas = (diab_nac_pos - diab_nac_neg).real / (
                    # displ_mag * 4.0
                    # )

                    if normal_mode == derivate_mode:
                        gammas -= freq * 0.5
                        # print(f"normal_mode {normal_mode}:", INFOS["frequencies"][normal_mode] / 4.55633590401805e-06)
                        check = np.where(np.abs(gammas * 2) / freq > 0.5)[0]
                        if len(check) > 0:
                            print(
                                f"WARNING: gammas wrong for states in {normal_mode}: {imult}",
                                [states[x] for x in check],
                                np.array2string(
                                    (gammas[check] * 2) / 4.55633590401805e-06, formatter={"float": lambda x: f"{x: 9.1f}cm-1"}
                                ),
                                np.array2string(gammas[check] * 2 / freq, formatter={"float": lambda x: f"{x*100: 4.1f}%"}),
                            )

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
                        # print(
                        # normal_mode,
                        # derivate_mode,
                        # np.linalg.norm(
                        # check_gamma[(imult, normal_mode, derivate_mode)]
                        # - check_gamma[(imult, derivate_mode, normal_mode)]
                        # ),
                        # )
                        gammas = (
                            check_gamma[(imult, normal_mode, derivate_mode)] + check_gamma[(imult, derivate_mode, normal_mode)]
                        ) * 0.5
                    elif normal_mode != derivate_mode:
                        continue
                    else:
                        check = np.where(np.abs(gammas * 2) / freq > 0.5)[0]
                        if len(check) > 0:
                            print(
                                f"WARNING: gammas wrong for states in {normal_mode}: {imult}",
                                [states[x] for x in check],
                                np.array2string(
                                    (gammas[check] * 2) / 4.55633590401805e-06, formatter={"float": lambda x: f"{x: 9.1f}cm-1"}
                                ),
                                np.array2string(gammas[check] * 2 / freq, formatter={"float": lambda x: f"{x*100: 4.1f}%"}),
                            )
                        print(
                            "adding gammas for states ",
                            normal_mode,
                            derivate_mode,
                            imult,
                            [
                                [states[i], gammas[i] * 2 / freq]
                                for i in np.where((np.abs(gammas) > 4.55633590401805e-06) & (np.abs(gammas * 2) / freq < 0.5))[0]
                            ],
                        )
                        gammas = np.where(np.abs(gammas * 2) / freq > 0.5, 0, gammas)
                    # gammas = 0.5*gammas

                    gamma_str_list.extend(
                        list(
                            map(
                                lambda i: f"{imult + 1:3d} {states[i]+1:3d} {states[i]+1:3d} {int(normal_mode):3d} {int(derivate_mode):3d} {gammas[i]: .7e}\n",
                                np.where(np.abs(gammas) > 4.55633590401805e-06)[0],
                            )
                        )
                    )
                start += nsi

    # add results to template string
    lvc_template_content += "kappa\n"
    lvc_template_content += "%i\n" % (nkappa)
    lvc_template_content += "".join(sorted(kappa_str_list))

    lvc_template_content += "lambda\n"
    lvc_template_content += "%i\n" % (nlambda)
    lvc_template_content += "".join(sorted(lambda_str_list))

    if INFOS["lambda_soc"]:
        # add results to template string
        lvc_template_content += "lambda_soc\n"
        lvc_template_content += "%i\n" % (len(lambda_soc_str_list))
        lvc_template_content += "".join(sorted(lambda_soc_str_list))

    if len(gamma_str_list) != 0:
        lvc_template_content += "gamma\n"
        lvc_template_content += "%i\n" % (len(gamma_str_list))
        lvc_template_content += "".join(sorted(gamma_str_list))

    # ----------------------- matrices ------------------------------
    if INFOS["soc"]:
        if "soc_file" in INFOS and INFOS["soc_file"]:
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
            if "no_transition_multipoles" in INFOS and INFOS["no_transition_multipoles"] and s_i != s_j:
                # TODO: This option is not written into the json by setup_LVCparam.py
                continue
            if s_i > s_j:
                continue
            for atom in range(len(INFOS["atoms"])):  # get mults
                n_entries += 1
                nums = "".join(map(lambda x: f"{x: 12.8f}", fit[atom, :]))
                # print(f"{s_i.S} {s_i.N + 1:2} {s_j.N + 1:2} {atom:3}    {nums}\n")
                mat_string += f"{s_i.S + 1} {s_i.N:2} {s_j.N:2} {atom:3}    {nums}\n"
        lvc_template_content += f"Multipolar Density Fit {settings}\n{n_entries}\n{mat_string}"

    # -------------------- write to file ----------------------------
    print("\nFinished!\nLVC parameters written to file:  " + template_name + "\n")
    lvc_template = open(template_name, "w")
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
    template_name = "LVC.template"
    #print(len(sys.argv))
    if len(sys.argv) == 3:
        template_name = sys.argv[2]
    if is_other_dir:
        for k, v in INFOS["paths"].items():
            INFOS["paths"][k] = os.path.join(sys.argv[1], v)

    write_LVC_template(INFOS, template_name)


# ======================================================================= #


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nCtrl+C occured. Exiting.\n")
        sys.exit()
