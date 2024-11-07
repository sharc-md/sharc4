#!/usr/bin/env python3

# Crude script for the conversion of
# orca output files to frequency files
# for molden. Only the last occurring
# instances of coordinates, frequencies
# and normal modes are used.

from sys import argv

# Check arguments
if len(argv) != 2:
    print("Usage: ORCA_hess_freq.py <orca.hess>\n:")
    print("Convert orca Hessian file to molden file for")
    print("normal mode visualisation.")
    exit()

name, orca_file = argv

try:
    lines = open(orca_file, 'r').readlines()
except IOError:
    print("Could not open %s." % orca_file)
    exit()

# check if file is sucessfully completed orca file:
is_orca = False
finished = False
if lines[1].strip() == "$orca_hessian_file":
    is_orca = True
finished = True

if not is_orca:
    print("File %s is not in orca output format (probably)!" % orca_file)
    exit()
elif is_orca and not finished:
    print("The job either has crashed or has not finished yet.")
    exit()
elif is_orca and finished:
    print("Reading data from file %s..." % orca_file)

# get coordinates
for iline, line in enumerate(lines):
    if "$atoms" in line:
        break
natom = int(lines[iline+1])
coords = []
for iatom in range(natom):
    s = lines[iline + 2 + iatom].split()
    atom = [ s[0], float(s[2]), float(s[3]), float(s[4]) ]
    coords.append(atom)
nfreq = 3*natom

# get frequencies
freqs = []
for iline, line in enumerate(lines):
    if "$vibrational_frequencies" in line:
        break
nfreq2 = int(lines[iline+1])
for ifreq in range(nfreq):
    f = float(lines[iline+2+ifreq].split()[-1])
    freqs.append(f)

# get modes
modes = []
for iline, line in enumerate(lines):
    if "$normal_modes" in line:
        break
for imode in range(nfreq):
    mode = []
    for iatom in range(natom*3):
        irow = iline + 3 + iatom + (imode//5)*(natom*3+1)
        icol = 1 + imode%5
        #print(irow,icol)
        c = float(lines[irow].split()[icol])
        mode.append(c)
    modes.append(mode)

# get intensities
ints = []
for iline, line in enumerate(lines):
    if "$ir_spectrum" in line:
        break
for imode in range(nfreq):
    s = lines[iline+2+imode].split()
    ir = float(s[2])
    ints.append(ir)




# generate molden file
out_file = orca_file + '.molden'
out = open(out_file, 'w')
out.write("[MOLDEN FORMAT]\n")
# write frequencies
out.write("[FREQ]\n")
for freq in freqs:
    out.write(str(freq) + '\n')
# write coordinates block (A.U.)
out.write("[FR-COORD]\n")
for coord in coords:
    out.write(coord[0] + ' ' + ' '.join([str(i) for i in coord[1:4]]) + '\n')
# write normal modes:
out.write("[FR-NORM-COORD]\n")
for i in range(nfreq):
    out.write("vibration %d\n" % (i + 1))
    for j in range(len(modes[i])):
        out.write(str(modes[i][j]) + ' ')
        if (j + 1) % 3 == 0:
            out.write('\n')
out.write('[INT]\n')
for i in range(nfreq):
    out.write('%16.9f\n' % ints[i])
out.close()
print("Molden output written to %s" % out_file)






# kate: indent-width 4
