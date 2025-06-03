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


import argparse
# from netCDF4 import Dataset
# from numba import njit
import numpy as np
import sys
# from utils import readfile
# import time
import os
from os.path import abspath, realpath
import math
from scipy.integrate import simpson



def make_form_factor_func(params):
    a_vals = params["a"]
    b_vals = params["b"]
    c_val = params["c"] if params["c"] is not None else 0.0

    def form_factor(G):
        q_term = (G / (4 * math.pi)) ** 2
        sum_term = sum(a * math.exp(-b * q_term) for a, b in zip(a_vals, b_vals))
        return c_val + sum_term

    return form_factor


def parse_atomic_data(file_path):
    data = {}

    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if line.startswith("#") or not line:
            continue  # Skip header or empty lines

        parts = line.split()
        element = parts[0].lower()
        numbers = list(map(float, parts[1:]))

        a_values = numbers[0::2][:4]  # Every other starting from 0, 4 elements
        b_values = numbers[1::2][:4]  # Every other starting from 1, 4 elements
        c_value = numbers[8]  # 9th value

        data[element] = {
            "a": a_values,
            "b": b_values,
            "c": c_value
        }

    return data



def load_histogram(file_path, cutoff, value_column):
    try:
        data = np.loadtxt(file_path)
    except Exception as e:
        sys.exit(f"Error reading file {file_path}: {e}")

    r = data[:, 0]
    col_index = value_column - 1
    if col_index >= data.shape[1]:
        sys.exit(f"Column {value_column} not found in {file_path}")
    h = data[:, col_index]

    mask = r <= cutoff
    return r[mask], h[mask]



def compute_SQ(element_data, element_alpha, element_beta, file_hist, file_hist_ref, R_cutoff, Q_max, Q_points, hist_column):
    f_alpha = make_form_factor_func(element_data[element_alpha])
    f_beta = make_form_factor_func(element_data[element_beta])

    r_vals, H = load_histogram(file_hist, R_cutoff, hist_column)
    _, H_ref = load_histogram(file_hist_ref, R_cutoff, hist_column)

    H_diff = H - H_ref

    Q_vals = np.linspace(0.01, Q_max, Q_points)
    S_Q = []

    for Q in Q_vals:
        fq_a = f_alpha(Q)
        fq_b = f_beta(Q)

        sin_term = np.sin(Q * r_vals) / (Q * r_vals)
        integrand = H_diff * sin_term
        integral = simpson(integrand, x = r_vals)

        S_val = fq_a * fq_b * integral
        S_Q.append(S_val)

    return Q_vals, np.array(S_Q)



def main():
    sharc_env = os.environ.get("SHARC")
    default_data_path = os.path.join(sharc_env, "..", "lib", "formfactor_gaussian.txt") if sharc_env else None
    if not os.path.isfile(default_data_path):
        default_data_path = None
    # the data file is taken from https://lampz.tugraz.at/~hadley/ss1/crystaldiffraction/atomicformfactors/formfactors.php

    parser = argparse.ArgumentParser(description="Compute S(Q) from pair distribution functions and atomic form factors.")
    if default_data_path:
        parser.add_argument('--data', default=default_data_path, help='File path for atomic form factor data')
    else:
        parser.add_argument('--data', required=True, help='File path for atomic form factor data')
    parser.add_argument('--alpha', required=True, help='Element alpha key')
    parser.add_argument('--beta', required=True, help='Element beta key')
    parser.add_argument('--hist', required=True, help='File path for H_alpha_beta(r)')
    parser.add_argument('--hist-ref', required=True, help='File path for H_alpha_beta^ref(r)')
    parser.add_argument('--column', type=int, default=2, help='Column index to use from histogram files (1-based, default=2)')
    parser.add_argument('--rcut', type=float, default=10.0, help='Cutoff radius R (default=10.0)')
    parser.add_argument('--qmax', type=float, default=15.0, help='Maximum Q value (default=15.0)')
    parser.add_argument('--qpoints', type=int, default=200, help='Number of Q points to compute (default=200)')

    args = parser.parse_args()

    # # Example hardcoded data - replace with file loading if needed
    # element_data = {
    #     "H": {
    #         "a": [0.489918, 0.262003, 0.196767, 0.049879],
    #         "b": [20.6593, 7.74039, 49.5519, 2.20159],
    #         "c": 0.001305
    #     },
    #     "O": {
    #         "a": [3.0485, 2.2868, 1.5463, 0.867],  # Example oxygen values
    #         "b": [13.2771, 5.7011, 0.3239, 32.9089],
    #         "c": 0.2508
    #     }
    #     # Add more elements as needed
    # }

    if args.qmax > 25.0:
        if realpath(abspath(args.data)) == realpath(abspath(default_data_path)):
            raise ValueError("Q_max >25 Å⁻¹: not supported by default form factor data file.")
        else:
            print("Q_max >25 Å⁻¹: please make sure that the used form factor data supports this.")

    element_data = parse_atomic_data(args.data)
    if args.alpha not in element_data or args.beta not in element_data:
        sys.exit(f"Element keys must be in dataset. Found keys: {list(element_data.keys())}")

    Q_vals, S_vals = compute_SQ(
        element_data,
        element_alpha=args.alpha.lower(),
        element_beta=args.beta.lower(),
        file_hist=args.hist,
        file_hist_ref=args.hist_ref,
        R_cutoff=args.rcut,
        Q_max=args.qmax,
        Q_points=args.qpoints,
        hist_column=args.column
    )

    for q, s in zip(Q_vals, S_vals):
        print(f"{q:.5f} {s:.5f}")

if __name__ == "__main__":
    main()
