#!/usr/bin/env python

import os, sys, argparse
import numpy as np
import re
from constants import ANG_TO_BOHR

def vibration_from_outcar(file_outcar,file_poscar,file_out,remove_rotations):
    '''
    Read vibration eigenvectors and eigenvalues from OUTCAR and create molden file for SHARC wigner.py
    '''
   
    #Getting n. of modes, nmodes
    with open(file_outcar,"r") as f:
        outcar=f.read()
    pattern=rf"\s+Degrees\s+of\s+freedom\s+DOF\s+=\s+(.*?)\n"
    nmodes=int(re.search(pattern,outcar).group(1))
    
    #Getting modes frequency and displacement
    start_marker=r"\s+Eigenvectors and eigenvalues of the dynamical matrix\n"
    start_marker=start_marker + r" \-+\n \n \n"
    end_marker=r" \n Finite differences"
    pattern = rf'{start_marker}(.*?){end_marker}'
    match = re.search(pattern, outcar, re.DOTALL).group(1).splitlines()
    data=[i for i in match if not i.isspace()]
    chunk=len(data)/nmodes
    if chunk.is_integer():
        chunk=int(chunk)
        index=[i for i in range(0,len(data),chunk)] #list of indexes, each one pointing to one vibrational mode block
    else:
        print(f"The number of frequencies in the output does not match with those that are analyzed, check OUTCAR!")
        sys.exit()
    freqs=[]
    modes=[]
    modes_to_discard=[]
    for n,i in enumerate(index):
        pattern1=r"\s*"+str(n+1)+r"\s+f\s*="
        pattern2=r"\s*"+str(n+1)+r"\s+f/i\s*="
        if re.search(pattern1,data[i]) is not None:
            match=r".*2PiTHz (.*?) cm-1"
            freqs.append(float(re.search(match,data[i]).group(1)))
            modes.append(data[i+2:i+chunk])
        elif re.search(pattern2,data[i]) is not None:
            modes_to_discard.append(n)
            match=r".*2PiTHz (.*?) cm-1"
            freqs.append(float(re.search(match,data[i]).group(1)))
            modes.append(data[i+2:i+chunk])
        else:
            print(f"Something went wrong with frequency mode n.{n}. Please check your VASP freq calculation. Check OUTCAR")
            sys.exit()
    #Sorting out mode blocks and generating numpy arrays out of lists.
    modes_tmp=modes.copy()
    modes=[]
    for i in range(len(modes_tmp)):
        tmp=[j.split() for j in modes_tmp[i]]
        modes.append(tmp)

    freqs=np.array(freqs,dtype=np.float64) 
    modes=np.array(modes,dtype=np.float64)

    # Checking for more than 3 imaginary freuquencies, meaning that the calculation went wrong.
    if len(modes_to_discard) > 3:
        print("WARNING: You have more than 3 imaginary frequencies in your VASP output. \nThese modes will be ignored but" \
                " something went wrong in your calculation, check your OUTCAR.\n" \
                "Only 3 (translational modes) are expected with imaginary frequencies because of finite difference errors.\n")
        for i in modes_to_discard:
            freqs[i]=0.0
            modes[i,:,3:]=0.0
    #Setting to zero the 3 translational modes 
    freqs = np.flip(freqs, axis=0)
    modes = np.flip(modes, axis=0)
    freqs[0:3]=0.0 #3 translations to zero
    modes[0:3,:,3:]=0.0
    #Remove rotarions if selected. Only for isolated system in a vacuum box.
    if remove_rotations:
        print("You have selected to remove rotational degrees of freedom. You must be simulating an isolate system in a vacuum box.\n")
        freqs[3:6]=0.0
        modes[3:6,:,3:]=0.0
    
    ##### Writing out molden file for SHARC #####
    
    #reading list of elements
    with open(file_poscar,"r") as f:
        tmp=f.readlines()
    if "!" in tmp[5].split():
        ind=tmp[5].split().index("!")
        el=tmp[5].split()[:ind]
    else:
        el=tmp[5].split()
    if "!" in tmp[6].split():
        ind=tmp[6].split().index("!")
        el_n=tmp[6].split()[:ind]
    else:
        el_n=tmp[6].split()
    elements=[]
    for i,j in zip(el,el_n):
        for k in range(int(j)):
            elements.append(i)
    # Writing of the molden file
    with open(file_out,"w") as f:
        f.write("[MOLDEN FORMAT]\n")
        f.write("[FREQ]\n")
        for i in freqs:
            f.write(f"{i:.2f}\n")
        f.write("[FR-COORD]\n")
        for i in range(len(elements)):
            f.write(f"{elements[i]} {modes[0][i,0]*ANG_TO_BOHR:.6f} {modes[0][i,1]*ANG_TO_BOHR:.6f} {modes[0][i,2]*ANG_TO_BOHR:.6f}\n") #bohr for geometry coordinates
        f.write("[FR-NORM-COORD]\n")
        for n,i in enumerate(modes):
            f.write(f"vibration {n+1}\n")
            for j in i:
                f.write(f"{j[3]:.6f} {j[4]:.6f} {j[5]:.6f}\n") #Mode displacements are normalized in VASP -> adimensional
    return print(f"{file_out} has been written out correctly\n")

def parse_cml_args(cml):
    '''
    command line parser.
    '''
    description='''
    This script generates a molden file which has to be used by wigner.py to generate the initial conditions (initconds file).
    The user has to provide the path to the OUTCAR file containg a converged VASP frequency calculation as well as location of the POSCAR file.
    '''

    arg = argparse.ArgumentParser(description=description,add_help=True,formatter_class=argparse.RawDescriptionHelpFormatter)
    arg.add_argument('-o', dest='outcar', action='store', type=str,
                     default='OUTCAR',
                     help='Location of VASP OUTCAR file containing frequency calculation.')
    
    arg.add_argument('-p', dest='poscar', action='store', type=str,
                     default='POSCAR',
                     help='Location of VASP POSCAR file used for frequency calculation.')
    
    arg.add_argument('-f', dest='file_out', action='store', type=str,
                     default='vasp.molden',
                     help='Name of output molden file. Default is vasp.molden')
    
    arg.add_argument('--remove_rot', dest='RR', action='store_true',
                     help='Remove rotational degrees of freedom. Only necessary when simulating non-periodic systems in a vacuum box.')
    
    return arg.parse_args(cml)

def main(cml):
    arg = parse_cml_args(cml)
    print("\nGenerating molden file from VASP frequency calculation\n")
    vibration_from_outcar(arg.outcar,arg.poscar,arg.file_out,arg.RR)

if __name__ == "__main__":
    main(sys.argv[1:])
