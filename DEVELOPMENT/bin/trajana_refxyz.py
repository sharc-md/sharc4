#!/usr/bin/env python

import sys
import os
import re
import string
import math
import imp

au2a=0.529177211

filename = sys.argv[1]
file1=open(filename)

outfile = open('ref.xyz','w')

f=file1.readlines()

coord_start=0
coord_end=0
i=-1
for line in f:
    i=i+1
    start=re.search('\[FR-COORD\]',line)
    end=re.search('\[FR-NORM-COORD\]',line)
    if start !=None:
       coord_start=i+1
    if end !=None:
       coord_end=i
numat=coord_end-coord_start

outfile.write('%i \n\n'%(numat))

for lines in f[coord_start:coord_end]:
    (Symb,xcoord,ycoord,zcoord)=lines.split()
    xcoord_a=float(xcoord)*au2a
    ycoord_a=float(ycoord)*au2a
    zcoord_a=float(zcoord)*au2a
    outfile.write(Symb + '   %4.12f  %4.12f  %4.12f \n' %(float(xcoord_a), float(ycoord_a), float(zcoord_a)))
    

file1.close()
outfile.close()
