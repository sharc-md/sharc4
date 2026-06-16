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

import os
import re
from pathlib import Path
import argparse
import numpy as np

# === Convolution kernels ===
class gauss:
    def __init__(self, fwhm):
        self.c = -4.0 * np.log(2.0) / fwhm**2
    def ev(self, A, x0, x):
        return A * np.exp(self.c * (x - x0)**2)

class lorentz:
    def __init__(self, fwhm):
        self.c = 0.25 * fwhm**2
    def ev(self, A, x0, x):
        return A / (((x - x0)**2) / self.c + 1)

class boxfunction:
    def __init__(self, fwhm):
        self.w = 0.5 * fwhm
    def ev(self, A, x0, x):
        res = np.zeros_like(x)
        res[np.abs(x - x0) < self.w] = A
        return res

class lognormal:
    def __init__(self, fwhm):
        self.fwhm = fwhm
    def ev(self, A, x0, x):
        if np.any(x <= 0) or x0 <= 0:
            return np.zeros_like(x)
        c = (np.log((self.fwhm + np.sqrt(self.fwhm**2 + 4.0 * x0**2)) / (2.0 * x0)))**2
        return A * x0 / x * np.exp(-c / (4.0 * np.log(2.0)) - np.log(2.0) * (np.log(x) - np.log(x0))**2 / c)

kernels = {
    1: {"f": gauss, "description": "Gaussian function"},
    2: {"f": lorentz, "description": "Lorentzian function"},
    3: {"f": boxfunction, "description": "Rectangular window"},
    4: {"f": lognormal, "description": "Log-normal function"},
}

# === Timing helpers ===
def parse_output_dat_steps(line):
    nums = list(map(int, line.split()[1:]))
    steps_info = []
    if len(nums) == 1:
        steps_info.append((nums[0], None))
    else:
        steps_info.append((nums[0], nums[1]))
        for i in range(2, len(nums) - 1, 2):
            steps_info.append((nums[i], nums[i + 1]))
        if len(nums) % 2 == 1:  # odd count => last stride has no max_step
            steps_info.append((nums[-1], None))
    return steps_info


def build_get_time_func(steps_info, stepsize):
    """
    Build a function mapping file index to time based on steps_info and stepsize.

    steps_info: list of (stride, max_step) tuples
    stepsize: multiplier for each step
    """
    # Precompute breakpoints
    breakpoints = []
    for i, (stride, max_step) in enumerate(steps_info):
        if max_step is None:
            break
        if i == 0:
            highest = max_step // stride
        else:
            highest = (max_step - steps_info[i-1][1]) // stride
        breakpoints.append(highest)
    breakpoints.append(np.inf)  # sentinel for last segment

    def get_time(index):
        step = 0
        for i, b in enumerate(breakpoints):
            cumulative = sum(breakpoints[:i+1])
            if index >= cumulative:
                step += b * steps_info[i][0]
            else:
                step += (index - sum(breakpoints[:i])) * steps_info[i][0]
                break
        return step * stepsize

    return get_time

# === File matching ===
def match_files(pattern):
    if '/' in pattern:
        raise ValueError("Pattern cannot include directories. Change to the directory where the files are located.")
    m = re.match(r'^(.*)%0(\d+)i(.*)$', pattern)
    if not m:
        raise ValueError("Pattern must contain %0Ni for integer index.")
    prefix, width, suffix = m.groups()
    width = int(width)
    regex = re.compile(r'^' + re.escape(prefix) + r'(\d{' + str(width) + r'})' + re.escape(suffix) + r'$')
    files = []
    for fname in os.listdir('.'):
        match = regex.match(fname)
        if match:
            files.append((int(match.group(1)), os.path.join(os.path.dirname(prefix) or '.', fname)))
    return sorted(files, key=lambda x: x[0])


# === Main script ===
def main():
    ap = argparse.ArgumentParser(description="Combine and optionally convolve scattering data.")
    ap.add_argument("pattern", help="File pattern, e.g. 'scatter_%%05i.out'")
    group = ap.add_mutually_exclusive_group()
    group.add_argument("--input-file", help="File containing output_dat_steps and stepsize")
    group.add_argument("--time-step", type=float, help="Fixed time step if metadata missing")
    ap.add_argument("--x-col", type=int, default=1, help="X column (1-based)")
    ap.add_argument("--y-col", type=int, default=2, help="Y column (1-based)")
    ap.add_argument("--output", required=True, help="Output file")
    ap.add_argument("--convolution", type=int, choices=kernels.keys(), help="Convolution kernel ID", default=None)
    ap.add_argument("--fwhm", type=float, help="FWHM for convolution kernel (in fs)")
    ap.add_argument("--tmin", type=float, help="Min time for convolution grid (in fs)")
    ap.add_argument("--tmax", type=float, help="Max time for convolution grid (in fs)")
    ap.add_argument("--tpoints", type=int, help="Number of points in convolution grid")
    ap.add_argument("--ref_av_n", type=int, default=1, help="Use average of the first N time steps as reference")
    args = ap.parse_args()

    # Time mapping
    if args.input_file:
        with open(args.input_file) as f:
            output_dat_steps = None
            stepsize = None
            for line in f:
                if line.startswith("output_dat_steps"):
                    output_dat_steps = parse_output_dat_steps(line)
                elif line.startswith("stepsize"):
                    stepsize = float(line.split()[1])
        if output_dat_steps is None or stepsize is None:
            raise ValueError("input-file missing required timing data.")
        get_time = build_get_time_func(output_dat_steps, stepsize)
    elif args.time_step:
        get_time = lambda i: i * args.time_step
    else:
        get_time = lambda i: i * 1.0

    # Match and read files
    files = match_files(args.pattern)
    if not files:
        raise ValueError("No matching files found.")

    data = {}
    # first_time_vals = {}
    x_values = None
    times = []

    for idx, fname in files:
        time = get_time(idx)
        times.append(time)
        xy = np.loadtxt(fname, comments='#', usecols=(args.x_col-1, args.y_col-1))
        if x_values is None:
            x_values = xy[:, 0]
        elif not np.allclose(x_values, xy[:, 0], equal_nan=True):
            raise ValueError(f"X mismatch in file: {fname}")
        y_vals = xy[:, 1]
        for x, y in zip(x_values, y_vals):
            data.setdefault(x, {})[time] = y
            # if x not in first_time_vals:
            #     first_time_vals[x] = y

    first_times = sorted(times)[:args.ref_av_n]  # first N times
    first_time_vals = {}
    for x in x_values:
        vals = [data[x][t] for t in first_times if t in data[x]]
        first_time_vals[x] = np.mean(vals)

    # Convolution or raw output
    if args.convolution:
        if not (args.fwhm and args.tmin is not None and args.tmax is not None and args.tpoints):
            raise ValueError("Convolution requires --fwhm, --tmin, --tmax, --tpoints.")
        # if args.input_file:
        #     raise ValueError("Convolution not compatible with non-uniform stride.")
        kernel_class = kernels[args.convolution]["f"]
        kernel = kernel_class(args.fwhm)
        t_grid = np.linspace(args.tmin, args.tmax, args.tpoints)
        first_time_vals={}
        with open(args.output, "w") as f_out:
            for t in t_grid:
                for x in x_values:
                    times = np.array(sorted(data[x].keys()))
                    y_raw = np.array([data[x][tt] for tt in times])
                    # Convolve
                    y_conv = np.sum([kernel.ev(y_raw[j], times[j], t) for j in range(len(times))])
                    y_weight = np.sum([kernel.ev(1., times[j], t) for j in range(len(times))])
                    f_out.write(f"{t:.6f} {x:.6f} {y_conv/y_weight:.6e}\n")
                f_out.write("\n")
    else:
        with open(args.output, "w") as f_out:
            for t in sorted({tt for x in x_values for tt in data[x].keys()}):
                for x in x_values:
                    y = data[x].get(t, np.nan)
                    f_out.write(f"{t:.6f} {x:.6f} {y:.6e} {first_time_vals[x]:.6e}\n")
                f_out.write("\n")

    print(f"Finished writing to {args.output}")

if __name__ == "__main__":
    main()
