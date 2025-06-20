#!/usr/bin/env python

import os, sys, argparse
import numpy as np
import re

def vibration_from_outcar(file_outcar='OUTCAR',file_poscar='POSCAR',file_out='vasp.molden'):
    '''
    Read vibration eigenvectors and eigenvalues from OUTCAR and create molden file for SHARC wigner.py
    Low frequency modes below specified threshold are neglected.
    '''
   
    #Conversion factors
    ang2au=1.8897259886 
    #Getting n. of modes, nmodes
    with open(file_outcar,"r") as f:
        outcar=f.read()
    pattern=rf"\s+Degree of freedom:\s+\d+\/(.*?)\n"
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
    counter=0
    for n,i in enumerate(index):
        pattern1=r"\s+"+str(n+1)+r"\s+f"
        pattern2=r"\s+"+str(n+1)+r"\s+f/i"
        if re.search(pattern1,data[i]) is not None:
            match=r".*2PiTHz (.*?) cm-1"
            freqs.append(float(re.search(match,data[i]).group(1)))
            modes.append(data[i+2:i+chunk])
        elif re.search(pattern2,data[i]) is not None: 
            counter=counter+1
            match=r".*2PiTHz (.*?) cm-1"
            freqs.append(float(re.search(match,data[i]).group(1)))
            modes.append(data[i+2:i+chunk])
        else:
            print("Something went wrong with your VASP freq calculation. Check OUTCAR")
            sys.exit()
    if counter > 3:
        print("You have more than 3 imaginary frequencies in your VASP output." \
                "Something went wrong, check your OUTCAR. Only 3 (translational modes) are expected")
        sys.exit()
    modes_tmp=modes.copy()
    modes=[]
    for i in range(len(modes_tmp)):
        tmp=[j.split() for j in modes_tmp[i]]
        modes.append(tmp)
    #Setting to zero the 3 translational modes 
    freqs.reverse() #reordering from lowest frequency first
    modes.reverse()
    freqs=np.array(freqs,dtype=np.float64) 
    modes=np.array(modes,dtype=np.float64)
    freqs[0:3]=0.0 #3 translations to zero
    modes[0:3,:,3:]=0.0
    
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
            f.write(f"{elements[i]} {modes[0][i,0]*ang2au:.6f} {modes[0][i,1]*ang2au:.6f} {modes[0][i,2]*ang2au:.6f}\n") #bohr for geometry coordinates
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
    arg = argparse.ArgumentParser(add_help=True)
    
    arg.add_argument('-o', dest='outcar', action='store', type=str,
                     default='OUTCAR',
                     help='Location of VASP OUTCAR file containing frequency calculation.')
    
    arg.add_argument('-p', dest='poscar', action='store', type=str,
                     default='POSCAR',
                     help='Location of VASP POSCAR file used for frequency calculation.')
    
    arg.add_argument('-f', dest='file_out', action='store', type=str,
                     default='vasp.molden',
                     help='Name of output molden file. Default is vasp.molden')
    
    return arg.parse_args(cml)

def main(cml):
    arg = parse_cml_args(cml)
    print("\nGenerating molden file from VASP frequency calculation\n")
    vibration_from_outcar(arg.outcar,arg.poscar,arg.file_out)

if __name__ == "__main__":
    main(sys.argv[1:])
