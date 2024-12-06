#!/usr/bin/env python3


from optparse import OptionParser
from netCDF4 import Dataset
import numba as nb
import numpy as np
import sys
from utils import readfile



def print_dx(name, data, origin, count, delta):
    with open(name, "w") as f:
        f.write(f"object 1 class gridpositions counts {count} {count} {count}\n")
        f.write("origin " + " ".join(map(lambda x: f"{x: 5.2f}", origin)) + "\n")
        f.write(f"delta {delta:4.2f} 0 0\n")
        f.write(f"delta 0 {delta:4.2f} 0\n")
        f.write(f"delta 0 0 {delta:4.2f}\n")
        f.write(f"object 2 class gridconnections counts {count} {count} {count}\n")
        f.write(f"object 3 class array type double rank 0 items {count**3} data follows\n")

        f.write("\n".join([" ".join(map(lambda x: f"{x:f}", data[i : i + 3])) for i in range(0, len(data), 3)]))
        f.write('\nobject "density [A^-3]" class field\n')


def get_c(fwhm):
    return -4.0 * np.log(2.0) / fwhm**2  # this factor needs to be evaluated only once


@nb.jit(cache=True, fastmath=True)
def loops_gauss_3d(traj, xyz0, xyz0_2, g_stop, c=-11.090354888959125):
    values = np.zeros((xyz0.shape[0]))
    for i_t in range(traj.shape[0]):
        for i_a in range(traj.shape[1]):
            xyz = traj[i_t, i_a, :].copy()
            if np.any(xyz > g_stop):
                continue
            cov = xyz @ xyz.T
            sq_sum = -2 * xyz0 @ xyz
            sq_sum += xyz0_2
            sq_sum += cov
            values += np.exp(c * sq_sum)
    return values




def main(infile, maskfile, outfile, options):

    # process options
    g_len = options.n
    width = options.w
    fwhm = options.f
    c = get_c(fwhm)

    # setup data
    g_stop = g_len * width / 2
    X, Y, Z = np.mgrid[-g_stop : g_stop : g_len * 1j, 
                       -g_stop : g_stop : g_len * 1j, 
                       -g_stop : g_stop : g_len * 1j]
    grid3D = np.stack((X.flatten(), Y.flatten(), Z.flatten()), axis=-1).astype(np.float32)
    grid3D_2 = np.einsum("ix,ix->i", grid3D, grid3D)

    # mask file
    # TODO: format from cpptraj mask command
    mask = readfile(maskfile)
    if "Frame    AtomNum Atom   ResNum  Res   MolNum" in mask[0]:
        # mask file from cpptraj
        indices = set()
        for line in mask:
            if '#' in line:
                continue
            indices.add(line.split()[1]-1)
            mask_indices = np.array(sorted(indices))
    else:
        # raw file
        mask_indices = np.loadtxt(maskfile, dtype = int) - 1

    # get data
    # TODO: could do this with slicing for lower memory usage
    with Dataset(infile) as f:
        # nframe, natom, nspat = f.variables["coordinates"].shape
        data = np.array(f.variables["coordinates"])

    # apply mask
    data = data[:, mask_indices, :]

    # compute KDE
    dens = loops_gauss_3d(data, grid3D, grid3D_2, g_stop, c)

    # write to dx file
    print_dx(
            outfile,
            dens,
            [-g_stop + width / 2, -g_stop + width / 2, -g_stop + width / 2],
            g_len,
            width,
        )




# =================================================================================
# =================================================================================
# =================================================================================


def parse_origin(option, opt_str, value, parser):
    try:
        # Split the input and convert to floats
        coords = list(map(float, value.split(',')))
        if len(coords) != 3:
            raise ValueError
        setattr(parser.values, option.dest, coords)
    except ValueError:
        raise ValueError(f"Invalid format for {opt_str}. Must be 'x,y,z'.")

# =================================================================================

if __name__ == "__main__":

    parser = OptionParser()
    parser.add_option("-w", "--cell_width",  dest="w", type="float", default=0.5, help="specify the cell width in Angstrom")
    parser.add_option("-n", "--cell_number", dest="n", type="int",   default=40,  help="specify the number of cells")
    parser.add_option("-f", "--fwhm",        dest="f", type="float", default=0.5, help="specify the FWHM of the convolution function in Angstrom")
    parser.add_option("-c", "--center",      dest="c", action="callback", callback=parse_origin, help="specify the center of the grid")
    # parser.add_option("-a", "--angstrom", dest='ang', action='store_true', help="Output in Angstrom (default in Bohr)")

    (options, args) = parser.parse_args()
    if len(args) < 3:
        parser.print_usage()
        sys.exit()
    infile, maskfile, outfile = args[0:3]
    main(infile, maskfile, outfile, options)














