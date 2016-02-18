#!/usr/bin/env python

import sys
import os
import re
import string
import math
import imp
adf=os.path.expandvars('$ADFHOME')
sys.path.append(adf+'/scripting')
import kf

try:
  import numpy
except ImportError:
  print 'The kf module required to read ADF binary files needs numpy. Please install numpy and then try again'
  sys.exit()

filename = sys.argv[1]
file1 = kf.kffile(filename)

outfile = open(filename[:-3]+'molden','w')

Freq = file1.read("Freq","Frequencies")
FreqCoord = file1.read("Freq","xyz")
Normalmodes= file1.read("Freq","Normalmodes")
NrAtom = file1.read("Freq","nr of atoms")
atomtype_a=file1.read("Geometry","fragmenttype")
atomtype=atomtype_a.tolist()
atomtype_index_a=file1.read("Geometry","fragment and atomtype index")
atomtype_index_b=atomtype_index_a.tolist()
atomtype_index=atomtype_index_b[int(NrAtom):]

Atomsymbs=[]
for a in range(0,int(NrAtom)):
    b=atomtype_index[a]-1
    c=atomtype[b]
    Atomsymbs.append(c)

nmodes = int(NrAtom) * 3

outfile.write("[MOLDEN FORMAT]\n")

outfile.write(' [FREQ] \n')
for i in range(0,int(nmodes)):
    x = i+1
    if x <= int(len(Freq)):
       outfile.write(' %6.2f \n' % (float(Freq[i])))
    else:
       outfile.write(' 0.00 \n')

a=-3
outfile.write(' [FR-COORD] \n')
for j in range(0,int(NrAtom)):
    Symb =str(Atomsymbs[j])
    a = a+3
    outfile.write(Symb + '   %4.12f  %4.12f  %4.12f \n' %(float(FreqCoord[a]), float(FreqCoord[a+1]), float(FreqCoord[a+2])))

o=-3

outfile.write(' [FR-NORM-COORD] \n')
for l in range(0,int(nmodes)):
    m = l+1
    outfile.write('Vibration     '+ str(m) + '\n')
    if m <= int(len(Freq)):
        for n in range(0,int(NrAtom)):
            o = o + 3
            outfile.write('  %4.8f   %4.8f   %4.8f \n' % (float(Normalmodes[o]), float(Normalmodes[o+1]), float(Normalmodes[o+2])))
    else:
        p = -3
        for n in range(0,int(NrAtom)):
            p= p+3
            outfile.write('  %4.8f   %4.8f   %4.8f \n' % (float(Normalmodes[p]), float(Normalmodes[p+1]), float(Normalmodes[p+2])))

file1.close()
outfile.close()
