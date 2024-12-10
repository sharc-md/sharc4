#!/usr/bin/python2

from __future__ import print_function
import numpy
import sys
from optparse import OptionParser
#import readline
# import re
import os
# import shutil
import copy
import numpy as np

from utils import readfile
import kabsch
from constants import U_TO_AMU, BOHR_TO_ANG
from setup_from_prmtop import expand_str_to_list




# ======================================================================================================================
# ======================================================================================================================
# ======================================================================================================================

class XYZIterator:
    def __init__(self, filename, buffered=True):
        self.filename = filename
        self.buffered = buffered
        self.file = None
        self.buffer = None
        self.current_index = 0

        # Open the file
        self.file = open(self.filename, 'r')

        # Read the first two lines to initialize natom and elements
        self.natom = int(self.file.readline().strip())
        self.next_comment_line = self.file.readline()  # Skip the comment line
        self.comment_line = ""

        # Buffered mode: Read the entire file into memory
        if self.buffered:
            self.buffer = self.file.readlines()
            self.file.close()
            self.file = None

        # Extract elements from the first geometry
        self.elements = []
        if self.buffered:
            lines = self.buffer[:self.natom]
        else:
            lines = [self.file.readline().strip() for _ in range(self.natom)]
            self.file.seek(0)
            self.file.readline()
            self.file.readline()
        self.elements = [line.split()[0] for line in lines]

    def __iter__(self):
        return self

    def __next__(self):
        if self.buffered:
            if self.current_index >= len(self.buffer):
                raise StopIteration

            # Extract lines for the current geometry
            start = self.current_index
            end = start + self.natom
            lines = self.buffer[start:end]
            self.current_index += self.natom + 2  # Move to the next geometry
            self.comment_line = copy.copy(self.next_comment_line)
            try:
                self.next_comment_line = self.buffer[self.current_index-1]
            except IndexError:
                self.next_comment_line = ""
        else:
            # Read lines for the current geometry from the file
            lines = [self.file.readline().strip() for _ in range(self.natom)]
            if not lines[0]:  # EOF reached
                raise StopIteration
            # Skip the blank line and comment line
            self.comment_line = copy.copy(self.next_comment_line)
            self.file.readline()
            self.next_comment_line = self.file.readline()

        # Convert coordinates to a numpy array in Bohr
        coordinates = np.array([
            [float(value) for value in line.split()[1:]]
            for line in lines
        ]) / BOHR_TO_ANG

        return coordinates

    def __del__(self):
        # Ensure the file is closed when the object is deleted
        if self.file and not self.file.closed:
            self.file.close()




# ======================================================================================================================
# ======================================================================================================================
# ======================================================================================================================

def read_V0(filename):
    # read
    lines = readfile(filename)
    # get number of atoms
    index = next((i for i, s in enumerate(lines) if "frequencies" in s.lower()), -1)
    if index < 0:
      raise ValueError
    natom = index -1
    # prepare
    V0 = {}
    it = 1
    # get ref structure
    rM = list(
        map(lambda x: [x[0]] + [float(y) for y in x[2:]], map(lambda x: x.split(), lines[it : it + natom]))
    )
    rM = np.asarray([x[1:] for x in rM], dtype=float)
    # assign and keep on reading
    V0["ref_coords"] = rM[:, :-1]
    V0["masses"] = rM[:, -1]
    tmp = np.sqrt(rM[:, -1] * U_TO_AMU)
    V0["Msa"] = np.asarray([tmp, tmp, tmp]).flatten(order="F")
    it += natom + 1
    V0["Om"] = np.asarray(lines[it].split(), dtype=float)
    it += 2
    V0["Km"] = np.asarray([x.split() for x in lines[it:]], dtype=float).T * V0["Msa"]
    return V0



# ======================================================================================================================
# ======================================================================================================================
# ======================================================================================================================

def main():

    parser = OptionParser()
    parser.add_option('-p', dest='p', type=int, nargs=1, default=4, help="number of decimals (default=4)")
    parser.add_option('-w', dest='f', type=int, nargs=1, default=20, help="field width (default=20)")
    parser.add_option('-g', dest='g', type="string", nargs=1, default="output.xyz", help="geometry file in xyz format (default=output.xyz)")
    parser.add_option('-v', dest='v', type="string", nargs=1, default="V0.txt", help="V0.txt file with the normal mode definitions")
    parser.add_option('-t', dest='t', type=float, nargs=1, default=1.0, help="timestep between successive geometries is fs (default=1.0 fs)")
    parser.add_option('-T', dest='T', type=int, nargs=1, default=0, help="start counting the timesteps at T (default=0)")
    parser.add_option('-k', dest='k', action='store_true', help="Switch on aligning via the Kabsch algorithm")
    parser.add_option('-b', dest='b', action='store_true', help="Switch on buffered reading")
    parser.add_option(
        "-q",
        "--qm-list",
        type="str",
        default="",
        dest="qm_list",
        help="Specify 'QM' atoms as list starting from 1 (e.g. 1~3,5,8~12,20)\ndefault=\"\"",
    )

    (options, args) = parser.parse_args()

    # open files
    file_size_mb = os.path.getsize(options.g) / (1024 * 1024)
    # buffered = file_size_mb<50.
    buffered = options.b
    TRAJ = XYZIterator(options.g, buffered = buffered)
    V0 = read_V0(options.v)
    ref_coords = V0["ref_coords"]

    # parse arguments
    if options.qm_list:
        qm_list = expand_str_to_list(options.qm_list)
        qm_list = np.array(qm_list) - 1
    else:
        qm_list = np.arange(TRAJ.natom)

    # iteration
    for igeom, geom in enumerate(TRAJ):
        # process coordinates
        coords = geom[qm_list]
        if options.k:
            B, a_s, b_s = kabsch.kabsch(ref_coords, coords)
            coords_in_ref_frame = (coords - b_s) @ B.T + a_s
        else:
            coords_in_ref_frame = coords
        
        # compute normal mode coordinates
        Q = np.sqrt(V0["Om"]) * (V0["Km"] @ (coords_in_ref_frame.flatten() - ref_coords.flatten()))

        # format output
        time = (igeom + options.T) * options.t
        string = "%6.2f " % time
        for i in Q:
            string += ' %12.9f' % i
        print(string + ' ' + TRAJ.comment_line.strip())



# ======================================================================================================================

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        sys.stdout.write('\nCtrl+C makes me a sad SHARC ;-(\n')
        quit(0)
