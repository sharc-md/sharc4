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


from optparse import OptionParser
from netCDF4 import Dataset
from numba import njit
import numpy as np
import sys
from utils import readfile
import time
import os



def get_indices(maskfile):
    mask = readfile(maskfile)
    if "Frame    AtomNum Atom   ResNum  Res   MolNum" in mask[0]:
        # mask file from cpptraj
        indices = set()
        for line in mask:
            if '#' in line:
                continue
            indices.add(int(line.split()[1])-1)
            mask_indices = np.array(sorted(indices))
    else:
        # raw file
        mask_indices = np.loadtxt(maskfile, dtype = int) - 1
    return mask_indices

def validate_mask(mask, maskname, natom):
    mask_set = set(mask)
    if len(mask) != len(mask_set):
        raise ValueError(f"Mask '{maskname}' contains duplicate atom indices.")
    if not all(0 <= idx < natom for idx in mask):
        invalid = [idx for idx in mask if idx < 0 or idx >= natom]
        raise ValueError(f"Mask '{maskname}' contains out-of-range indices: {invalid}")


def process_frame_broadcast(c1, c2, hist, nhist, rhist, same_group):
    # (natom1, natom2, 3)
    dist_vectors = c1[:, np.newaxis, :] - c2[np.newaxis, :, :]
    dist_vectors_sq = dist_vectors ** 2
    dist_squared = np.sum(dist_vectors_sq, axis=-1)

    if same_group:
        for i in range(dist_squared.shape[0]):
            dist_squared[i, i] = np.inf

    # normalized component weights = cos²
    for i in range(3):
        dist_vectors_sq[..., i] /= dist_squared

    dist = np.sqrt(dist_squared)
    # with np.errstate(invalid='ignore'):
    indices = np.floor((dist / rhist) * nhist).astype(np.int64)
    indices = np.clip(indices, 0, nhist)

    indices_flat = indices.flatten()
    for idir in range(3):
        values = dist_vectors_sq[..., idir].flatten()
        for k in range(indices_flat.size):
            hist[idir, indices_flat[k]] += values[k]

try:
    from numba import njit
    process_frame_broadcast = njit(process_frame_broadcast, cache = True)
    print("Using numba-accelerated frame processing.")
except ImportError:
    print("Numba not found — using slower fallback mode.")



def main(infile, maskfile1, maskfile2, outfile, options):

    start = time.time()

    # get data
    with Dataset(infile) as f:
        nframe, natom, nspatial = f.variables["coordinates"].shape
        data = np.array(f.variables["coordinates"])

    # mask file
    # TODO: format from cpptraj mask command
    mask_indices1 = get_indices(maskfile1)
    mask_indices2 = get_indices(maskfile2)
    n1 = len(mask_indices1)  # natom1
    n2 = len(mask_indices2)  # natom2
    same_group = (os.path.abspath(maskfile1) == os.path.abspath(maskfile2))
    if not same_group:
        overlap = set(mask_indices1) & set(mask_indices2)
        if overlap:
            raise ValueError(f"Mask files contain overlapping atom indices: {sorted(overlap)}")
    validate_mask(mask_indices1, maskfile1, natom)
    validate_mask(mask_indices2, maskfile2, natom)

    # process options
    nhist = options.n 
    rhist = options.w * options.n

    # setup data
    # setup the histogram here
    hist = np.zeros( (3,nhist+1), dtype=float)

    bin_edges = np.linspace(0, rhist, nhist + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    shell_volumes = 4/3 * np.pi * (bin_edges[1:]**3 - bin_edges[:-1]**3)  # volume of spherical shells
    if same_group:
        npairs = n1 * (n1 - 1)
    else:
        npairs = n1 * n2

    # apply mask  (nframe, natom, nspatial)
    coord1 = data[:, mask_indices1, :]
    coord2 = data[:, mask_indices2, :]

    print(f"Entering main loop after {time.time() - start:.1f}s")

    # compute RDFs
    try:
        for iframe in range(nframe):
            process_frame_broadcast(coord1[iframe], coord2[iframe], hist, nhist, rhist, same_group)
            if (iframe+1) % 100 == 0:
                print(f"Processed frame {iframe+1}/{nframe} after {time.time() - start:.1f}s")
        print(f"Processed frame {iframe+1}/{nframe} after {time.time() - start:.1f}s")
    except KeyboardInterrupt:
        print(f"Interrupted after processed frame {iframe+1}/{nframe} after {time.time() - start:.1f}s")
        nframe = iframe+1
        print(f"Will write output now assuming nframe = {nframe}")

    # write to txt file
    if options.rawhist:
        norm_factor = nframe
        for idir in range(3):
            hist[idir, :-1] /= (norm_factor)
    else:
        norm_factor = nframe * npairs
        for idir in range(3):
            hist[idir, :-1] /= (norm_factor * shell_volumes)
    with open(outfile, 'w') as f:
        for i in range(nhist):
            # Write the distance and the corresponding accumulated value to the file
            f.write(f"{bin_centers[i]:12.6f} {sum(hist[:,i]):15.8e} {hist[0,i]:15.8e} {hist[1,i]:15.8e} {hist[2,i]:15.8e}\n")



# =================================================================================
# =================================================================================
# =================================================================================

if __name__ == "__main__":

    parser = OptionParser()
    parser.add_option("-w", "--cell_width",  dest="w", type="float", default=0.1, help="specify the cell width in Angstrom")
    parser.add_option("-n", "--cell_number", dest="n", type="int",   default=100,  help="specify the number of cells")
    parser.add_option("-r", "--rawhist", dest='rawhist', action='store_true', help="Return raw histograms rather than normalized RDFs")


    (options, args) = parser.parse_args()
    if len(args) < 3:
        parser.print_usage()
        sys.exit()
    infile, maskfile1, maskfile2, outfile = args[0:4]
    print("\nRunning Cartesian-weighted RDF with the following options:")
    print(f"  Input file:          {infile}")
    print(f"  Mask file 1:         {maskfile1}")
    print(f"  Mask file 2:         {maskfile2}")
    print(f"  Output file:         {outfile}")
    print(f"  Bin width (-w):      {options.w:.3f} Å")
    print(f"  Number of bins (-n): {options.n}")
    if options.rawhist:
        print(f"  Mode (-r)          : Raw histograms (normalized by frame number)")
    else:
        print(f"  Mode (-r)          : RDF (normalized by frame number, atom pairs, and shell volume)")
    print()
    main(infile, maskfile1, maskfile2, outfile, options)














