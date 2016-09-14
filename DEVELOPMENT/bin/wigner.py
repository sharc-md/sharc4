#!/usr/bin/env python2

# Script for the calculation of Wigner distributions from molden frequency files
# 
# usage python wigner.py [-n <NUMBER>] <MOLDEN-FILE>

import copy
import math
import cmath
import random
import sys
import datetime
from optparse import OptionParser

# =========================================================0
# compatibility stuff

if sys.version_info[0]!=2:
  print 'This is a script for Python 2!'
  sys.exit(0)

if sys.version_info[1]<5:
  def any(iterable):
    for element in iterable:
      if element:
        return True
    return False

  def all(iterable):
    for element in iterable:
      if not element:
        return False
    return True



# some constants
DEBUG = False
CM_TO_HARTREE = 1./219474.6     #4.556335252e-6 # conversion factor from cm-1 to Hartree
HARTREE_TO_EV = 27.211396132    # conversion factor from Hartree to eV
U_TO_AMU = 1./5.4857990943e-4            # conversion from g/mol to amu
ANG_TO_BOHR = 1./0.529177211    #1.889725989      # conversion from Angstrom to bohr
PI = math.pi

version='1.0'
versiondate=datetime.date(2014,10,8)


NUMBERS = {'H':  1, 'He': 2,
'Li': 3, 'Be': 4, 'B':  5, 'C':  6,  'N': 7,  'O': 8, 'F':  9, 'Ne':10,
'Na':11, 'Mg':12, 'Al':13, 'Si':14,  'P':15,  'S':16, 'Cl':17, 'Ar':18,
'K': 19, 'Ca':20, 
'Sc':21, 'Ti':22, 'V': 23, 'Cr':24, 'Mn':25, 'Fe':26, 'Co':27, 'Ni':28, 'Cu':29, 'Zn':30,
'Ga':31, 'Ge':32, 'As':33, 'Se':34, 'Br':35, 'Kr':36, 
'Rb':37, 'Sr':38,
'Y':39,  'Zr':40, 'Nb':41, 'Mo':42, 'Tc':43, 'Ru':44, 'Rh':45, 'Pd':46, 'Ag':47, 'Cd':48,
'In':49, 'Sn':50, 'Sb':51, 'Te':52,  'I':53, 'Xe':54,
'Cs':55, 'Ba':56,
'La':57, 'Hf':72, 'Ta':73,  'W':74, 'Re':75, 'Os':76, 'Ir':77, 'Pt':78, 'Au':79, 'Hg':80,
'Tl':81, 'Pb':82, 'Bi':83, 'Po':84, 'At':85, 'Rn':86
}

# Atomic Weights of the most common isotopes
# Masses are from MOLPRO, except where noted
# Also very comprehensive: http://www.nist.gov/pml/data/comp.cfm
MASSES = {'H' :   1.00782 * U_TO_AMU,
          'He':   4.00260 * U_TO_AMU,
          'Li':   7.01600 * U_TO_AMU,
          'Be':   9.01218 * U_TO_AMU,
          'B' :  11.00931 * U_TO_AMU,
          'C' :  12.00000 * U_TO_AMU,
          'N' :  14.00307 * U_TO_AMU,
          'O' :  15.99491 * U_TO_AMU,
          'F' :  18.99840 * U_TO_AMU,
          'Ne':  19.99244 * U_TO_AMU,
          'Na':  22.98980 * U_TO_AMU,
          'Mg':  23.98504 * U_TO_AMU,
          'Al':  26.98153 * U_TO_AMU,
          'Si':  27.97693 * U_TO_AMU,
          'P' :  30.97376 * U_TO_AMU,
          'S' :  31.97207 * U_TO_AMU,
          'Cl':  34.96885 * U_TO_AMU,
          'Ar':  39.96238 * U_TO_AMU,
          'K' :  38.96371 * U_TO_AMU,
          'Ca':  39.96259 * U_TO_AMU,
          'Sc':  44.95592 * U_TO_AMU,
          'Ti':  47.94795 * U_TO_AMU,
          'V' :  50.94400 * U_TO_AMU,
          'Cr':  51.94050 * U_TO_AMU,
          'Mn':  54.93800 * U_TO_AMU,
          'Fe':  55.93490 * U_TO_AMU,
          'Co':  58.93320 * U_TO_AMU,
          'Ni':  57.93534 * U_TO_AMU,
          'Cu':  62.92960 * U_TO_AMU,
          'Zn':  63.92910 * U_TO_AMU,
          'Ga':  68.92570 * U_TO_AMU,
          'Ge':  73.92190 * U_TO_AMU,
          'As':  74.92160 * U_TO_AMU,
          'Se':  79.91650 * U_TO_AMU,
          'Br':  78.91830 * U_TO_AMU,
          'Kr':  83.80000 * U_TO_AMU,
          'Rb':  84.91170 * U_TO_AMU,
          'Sr':  87.90560 * U_TO_AMU,
          'Y' :  88.90590 * U_TO_AMU,
          'Zr':  89.90430 * U_TO_AMU,
          'Nb':  92.90600 * U_TO_AMU,
          'Mo':  97.90550 * U_TO_AMU,
          'Tc':  98.90620 * U_TO_AMU,
          'Ru': 101.90370 * U_TO_AMU,
          'Rh': 102.90480 * U_TO_AMU,
          'Pd': 105.90320 * U_TO_AMU,
          'Ag': 106.90509 * U_TO_AMU,
          'Cd': 113.90360 * U_TO_AMU,
          'In': 114.90410 * U_TO_AMU,
          'Sn': 119.90220 * U_TO_AMU,   # MOLPRO library is wrong
          'Sb': 120.90380 * U_TO_AMU,
          'Te': 129.90670 * U_TO_AMU,
          'I' : 126.90440 * U_TO_AMU,
          'Xe': 131.90420 * U_TO_AMU,
          'Cs': 132.90510 * U_TO_AMU,
          'Ba': 137.90500 * U_TO_AMU,
          'La': 138.90610 * U_TO_AMU,
          'Hf': 179.94680 * U_TO_AMU,
          'Ta': 180.94800 * U_TO_AMU,
          'W' : 183.95100 * U_TO_AMU,
          'Re': 186.95600 * U_TO_AMU,
          'Os': 190.20000 * U_TO_AMU,
          'Ir': 192.96330 * U_TO_AMU,
          'Pt': 194.96480 * U_TO_AMU,
          'Au': 196.96660 * U_TO_AMU,
          'Hg': 201.97060 * U_TO_AMU,
          'Tl': 204.97450 * U_TO_AMU,
          'Pb': 207.97660 * U_TO_AMU,
          'Bi': 208.98040 * U_TO_AMU,
          'Po': 208.98250 * U_TO_AMU,
          'At': 209.98715 * U_TO_AMU,   # MOLPRO library is wrong
          'Rn': 210.99060 * U_TO_AMU}   # MOLPRO library is wrong

# Isotopes used for the masses
ISOTOPES={'H' : 'H-1' ,
          'He': 'He-4',
          'Li': 'Li-7',
          'Be': 'Be-9*',
          'B' : 'B_11' ,
          'C' : 'C-12' ,
          'N' : 'N-14' ,
          'O' : 'O-16' ,
          'F' : 'F-19*' ,
          'Ne': 'Ne-20',
          'Na': 'Na-23*',
          'Mg': 'Mg-24',
          'Al': 'Al-27*',
          'Si': 'Si-28',
          'P' : 'P-31*' ,
          'S' : 'S-32' ,
          'Cl': 'Cl-35',
          'Ar': 'Ar-40',
          'K' : 'K-39' ,
          'Ca': 'Ca-40',
          'Sc': 'Sc-45*',
          'Ti': 'Ti-48',
          'V' : 'V-51' ,
          'Cr': 'Cr-52',
          'Mn': 'Mn-55*',
          'Fe': 'Fe-56',
          'Co': 'Co-59*',
          'Ni': 'Ni-58',
          'Cu': 'Cu-63',
          'Zn': 'Zn-64',
          'Ga': 'Ga-68',
          'Ge': 'Ge-74',
          'As': 'As-75*',
          'Se': 'Se-80',
          'Br': 'Br-79',
          'Kr': 'Kr-84',
          'Rb': 'Rb-85',
          'Sr': 'Sr-88',
          'Y' : 'Y-89*' ,
          'Zr': 'Zr-90',
          'Nb': 'Nb-93*',
          'Mo': 'Mo-98',
          'Tc': 'Tc-99',
          'Ru': 'Ru-102',
          'Rh': 'Rh-103*',
          'Pd': 'Pd-106',
          'Ag': 'Ag-107',
          'Cd': 'Cd-114',
          'In': 'In-115',
          'Sn': 'Sn-120',
          'Sb': 'Sb-121',
          'Te': 'Te-130',
          'I' : 'I-127*' ,
          'Xe': 'Xe-132',
          'Cs': 'Cs-133*',
          'Ba': 'Ba-138',
          'La': 'La-139',
          'Hf': 'Hf-180',
          'Ta': 'Ta-181',
          'W' : 'W-184' ,
          'Re': 'Re-187',
          'Os': 'Os-192',
          'Ir': 'Ir-193',
          'Pt': 'Pt-195',
          'Au': 'Au-197*',
          'Hg': 'Hg-202',
          'Tl': 'Tl-205',
          'Pb': 'Pb-208',
          'Bi': 'Bi-209*',
          'Po': 'Po-209',
          'At': 'At-210',
          'Rn': 'Rn-211'}



# thresholds
LOW_FREQ = 10.0 # threshold in cm^-1 for ignoring rotational and translational low frequencies

# ======================================================================================================================
# ======================================================================================================================
# ======================================================================================================================


def try_read(l,index,typefunc,default):
  try:
    if typefunc==bool:
      return 'True'==l[index]
    else:
      return typefunc(l[index])
  except IndexError:
    return typefunc(default)
  except ValueError:
    print 'Could not initialize object!'
    quit(1)

# ======================================================================================================================

class ATOM:
  def __init__(self,symb='??',num=0.,coord=[0.,0.,0.],m=0.,veloc=[0.,0.,0.]):
    self.symb  = symb
    self.num   = num
    self.coord = coord
    self.mass  = m
    self.veloc = veloc
    self.Ekin=0.5*self.mass * sum( [ self.veloc[i]**2 for i in range(3) ] )

  def init_from_str(self,initstring=''):
    f=initstring.split()
    self.symb  =   try_read(f,0,str,  '??')
    self.num   =   try_read(f,1,float,0.)
    self.coord = [ try_read(f,i,float,0.) for i in range(2,5) ]
    self.mass  =   try_read(f,5,float,0.)*U_TO_AMU
    self.veloc = [ try_read(f,i,float,0.) for i in range(6,9) ]
    self.Ekin=0.5*self.mass * sum( [ self.veloc[i]**2 for i in range(3) ] )

  def __str__(self):
    s ='%2s % 5.1f '               % (self.symb, self.num)
    s+='% 12.8f % 12.8f % 12.8f '  % tuple(self.coord)
    s+='% 12.8f '                  % (self.mass/U_TO_AMU)
    s+='% 12.8f % 12.8f % 12.8f'   % tuple(self.veloc)
    return s

  def EKIN(self):
    self.Ekin=0.5*self.mass * sum( [ self.veloc[i]**2 for i in range(3) ] )
    return self.Ekin

  def geomstring(self):
    s='  %2s % 5.1f % 12.8f % 12.8f % 12.8f % 12.8f' % (self.symb,self.num,self.coord[0],self.coord[1],self.coord[2],self.mass/U_TO_AMU)
    return s

  def velocstring(self):
    s=' '*11+'% 12.8f % 12.8f % 12.8f' % tuple(self.veloc)
    return s

# ======================================================================================================================

class STATE:
  def __init__(self,i=0,e=0.,eref=0.,dip=[0.,0.,0.]):
    self.i       = i
    self.e       = e.real
    self.eref    = eref.real
    self.dip     = dip
    self.Excited = False
    self.Eexc    = self.e-self.eref
    self.Fosc    = (2./3.*self.Eexc*sum( [i*i.conjugate() for i in self.dip] ) ).real
    if self.Eexc==0.:
      self.Prob  = 0.
    else:
      self.Prob  = self.Fosc/self.Eexc**2

  def init_from_str(self,initstring):
    f=initstring.split()
    self.i       =   try_read(f,0,int,  0 )
    self.e       =   try_read(f,1,float,0.)
    self.eref    =   try_read(f,2,float,0.)
    self.dip     = [ complex( try_read(f,i,float,0.),try_read(f,i+1,float,0.) ) for i in [3,5,7] ]
    self.Excited =   try_read(f,11,bool, False)
    self.Eexc    = self.e-self.eref
    self.Fosc    = (2./3.*self.Eexc*sum( [i*i.conjugate() for i in self.dip] ) ).real
    if self.Eexc==0.:
      self.Prob  = 0.
    else:
      self.Prob  = self.Fosc/self.Eexc**2

  def __str__(self):
    s ='%03i % 18.10f % 18.10f ' % (self.i,self.e,self.eref)
    for i in range(3):
      s+='% 12.8f % 12.8f ' % (self.dip[i].real,self.dip[i].imag)
    s+='% 12.8f % 12.8f %s' % (self.Eexc*HARTREE_TO_EV,self.Fosc,self.Excited)
    return s

  def Excite(self,max_Prob,erange):
    try:
      Prob=self.Prob/max_Prob
    except ZeroDivisionError:
      Prob=-1.
    if not (erange[0] <= self.Eexc <= erange[1]):
      Prob=-1.
    self.Excited=(random.random() < Prob)

# ======================================================================================================================

class INITCOND:
  def __init__(self,atomlist=[],eref=0.,epot_harm=0.):
    self.atomlist=atomlist
    self.eref=eref
    self.Epot_harm=epot_harm
    self.natom=len(atomlist)
    self.Ekin=sum( [atom.Ekin for atom in self.atomlist] )
    self.statelist=[]
    self.nstate=0
    self.Epot=epot_harm

  def addstates(self,statelist):
    self.statelist=statelist
    self.nstate=len(statelist)
    self.Epot=self.statelist[0].e-self.eref

  def init_from_file(self,f,eref,index):
    while True: 
      line=f.readline()
      if 'Index     %i' % (index) in line:
        break
    f.readline()        # skip one line, where "Atoms" stands
    atomlist=[]
    while True:
      line=f.readline()
      if 'States' in line:
        break
      atom=ATOM()
      atom.init_from_str(line)
      atomlist.append(atom)
    statelist=[]
    while True:
      line=f.readline()
      if 'Ekin' in line:
        break
      state=STATE()
      state.init_from_str(line)
      statelist.append(state)
    while not line=='\n' and not line=='':
      line=f.readline()
      if 'epot_harm' in line.lower():
        epot_harm=float(line.split()[1])
        break
    self.atomlist=atomlist
    self.eref=eref
    self.Epot_harm=epot_harm
    self.natom=len(atomlist)
    self.Ekin=sum( [atom.Ekin for atom in self.atomlist] )
    self.statelist=statelist
    self.nstate=len(statelist)
    if self.nstate>0:
      self.Epot=self.statelist[0].e-self.eref
    else:
      self.Epot=epot_harm

  def __str__(self):
    s='Atoms\n'
    for atom in self.atomlist:
      s+=str(atom)+'\n'
    s+='States\n'
    for state in self.statelist:
      s+=str(state)+'\n'
    s+='Ekin      % 16.12f a.u.\n' % (self.Ekin)
    s+='Epot_harm % 16.12f a.u.\n' % (self.Epot_harm)
    s+='Epot      % 16.12f a.u.\n' % (self.Epot)
    s+='Etot_harm % 16.12f a.u.\n' % (self.Epot_harm+self.Ekin)
    s+='Etot      % 16.12f a.u.\n' % (self.Epot+self.Ekin)
    s+='\n\n'
    return s


# ======================================================================================================================
# ======================================================================================================================
# ======================================================================================================================
def ask_for_masses():
  print '''
Option -m used, please enter non-default masses:
+ number mass           add non-default mass <mass> for atom <number>
- number                remove non-default mass for atom <number> (default mass will be used)
show                    show non-default atom masses
end                     finish input for non-default masses
'''
  MASS_LIST={}
  while True:
    line=raw_input()
    if 'end' in line:
      break
    if 'show' in line:
      s='-----------------------\nAtom               Mass\n'
      for i in MASS_LIST:
        s+='% 4i %18.12f\n' % (i,MASS_LIST[i])
      s+='-----------------------'
      print s
      continue
    if '+' in line:
      f=line.split()
      if len(f)<3:
        continue
      try:
        num=int(f[1])
        mass=float(f[2])
      except ValueError:
        continue
      MASS_LIST[num]=mass*U_TO_AMU
      continue
    if '-' in line:
      f=line.split()
      if len(f)<2:
        continue
      try:
        num=int(f[1])
      except ValueError:
        continue
      del MASS_LIST[num]
      continue
  return MASS_LIST


# ======================================================================================================================

def get_mass(symb,number):
  if 'MASS_LIST' in globals() and number in MASS_LIST:
    return MASS_LIST[number]
  else:
    try:
      return MASSES[symb]
    except KeyError:
      print 'No default mass for atom %s' % (symb)
      quit(1)


# ======================================================================================================================

def import_from_molpro(filename,scaling):
  """This function imports the atomic coordinates and normal modes
from a MOLPRO frequency calculation.
It returns the atomic coordinates in a list where each entry is a
dictionary containing the symbol and xyz-coordinates of a single
atom. The normal modes are also returned in a list where each entry
is a dictionary describing one mode by its frequency and the
movement vector for each atom of the molecule."""

  f=open(filename)
  data=f.readlines()
  f.close()

  # find coordinate block
  iline=0
  while not 'FREQUENCIES * CALCULATION OF NORMAL MODES' in data[iline]:
    iline+=1
    if iline==len(data):
      print 'Could not find coordinates in %s!' % (filename)
      quit(1)
  # get atoms
  iline+=7
  natom=0
  molecule=[]
  while not data[iline]=='\n':
    f=data[iline].split()
    symb=f[1].lower().title()
    num=float(f[2])
    coord=[ float(f[i+3]) for i in range(3) ]
    natom+=1
    mass=get_mass(symb,natom)
    whichatoms.append(symb)
    molecule.append(ATOM(symb,num,coord,mass))
    iline+=1

  # obtain all frequencies, including low ones
  found_freq = True
  modes = [] # list of normal modes
  for ln, line in enumerate(data): # iterate through the file line by line
    if line.find('Wavenumbers [cm-1]') != -1 and found_freq:
      # read frequencies
      line = line.split()
      for i, freq in enumerate(line[2:]):
        if float(freq) > LOW_FREQ: # ignore low frequencies
          mode = {}
          mode['freq'] = float(freq) * CM_TO_HARTREE * scaling
          mode['move'] = []
          hindex = i + 1 # horizontal index
          for j, atom in enumerate(molecule):
            vindex = ln + (j+1)*3 # at this line the data for the oscillation of the single atom starts
            x = float(data[vindex].split()[hindex])
            y = float(data[vindex+1].split()[hindex])
            z = float(data[vindex+2].split()[hindex])
            mode['move'].append( [x, y, z] )
          # transform coordinates of displacement into mass weighted coordinates
          # see: http://www.molpro.net/piperdavidl/molpro-user/2010-November/004035.html
          # first of all, norm molpro movement vectors
          norm = 0.0
          for j, atom in enumerate(molecule):
            for xyz in range(3):
              norm += mode['move'][j][xyz]**2
          norm = math.sqrt(norm)
          for j, atom in enumerate(molecule):
            for xyz in range(3):
              mode['move'][j][xyz] /= norm
          # now calculate the movement vectors, taking the mass of the atoms into account
          mu_reduced = 0.0
          for j, atom in enumerate(molecule):
            for xyz in range(3):
              mu_reduced += mode['move'][j][xyz]**2 / MASSES[atom.symb]
          mu_reduced = math.sqrt(mu_reduced)
          for j, atom in enumerate(molecule):
            for xyz in range(3):
              mode['move'][j][xyz] *= mu_reduced
          mode['mu_red'] = mu_reduced
          modes.append(mode)
  #print molecule
  #print modes
  return molecule, modes

# ======================================================================================================================

def import_from_molden(filename,scaling):
  '''This function imports atomic coordinates and normal modes from a MOLDEN
file. Returns molecule and modes as the other function does.
'''
  f=open(filename)
  data=f.readlines()
  f.close()

  # find coordinate block
  iline=0
  while not 'FR-COORD' in data[iline]:
    iline+=1
    if iline==len(data):
      print 'Could not find coordinates in %s!' % (filename)
      print 'Perhaps this is a MOLPRO output, then please use -M option.'
      quit(1)
  # get atoms
  iline+=1
  natom=0
  molecule=[]
  while not '[' in data[iline]:
    f=data[iline].split()
    symb=f[0].lower().title()
    num=NUMBERS[symb]
    coord=[ float(f[i+1]) for i in range(3) ]
    natom+=1
    mass=get_mass(symb,natom)
    whichatoms.append(symb)
    molecule.append(ATOM(symb,num,coord,mass))
    iline+=1

  # find number of frequencies
  iline=-1
  nmodes=-1
  while True:
    iline+=1
    if iline==len(data):
      nmodes=3*natom
      break
    line=data[iline]
    if 'N_FREQ' in line:
      nmodes=int(data[iline+1])
      break

  # warn, if too few normal modes were found
  if nmodes<3*natom:
    print '*'*51+'\nWARNING: Less than 3*N_atom normal modes extracted!\n'+'*'*51+'\n'

  # obtain all frequencies, including low ones
  iline=0
  modes=[]
  while not '[FREQ]' in data[iline]:
    iline+=1
  iline+=1
  for imode in range(nmodes):
    try:
      mode={'freq':float(data[iline+imode])*CM_TO_HARTREE * scaling}
      modes.append(mode)
    except ValueError:
      print '*'*51+'\nWARNING: Less than 3*N_atom normal modes, but no [N_FREQ] keyword!\n'+'*'*51+'\n'
      nmodes=imode
      break

  # obtain normal coordinates
  iline=0
  while not 'FR-NORM-COORD' in data[iline]:
    iline+=1
  iline+=1
  for imode in range(nmodes):
    iline+=1
    move=[]
    for iatom in range(natom):
      f=data[iline].split()
      move.append([ float(f[i]) for i in range(3) ])
      iline+=1
    modes[imode]['move']=move
    # normalization stuff
    norm = 0.0
    for j, atom in enumerate(molecule):
      for xyz in range(3):
        norm += modes[imode]['move'][j][xyz]**2
    norm = math.sqrt(norm)
    if norm!=0.0:
      for j, atom in enumerate(molecule):
        for xyz in range(3):
          modes[imode]['move'][j][xyz] /= norm
    elif modes[imode]['freq']>=LOW_FREQ*CM_TO_HARTREE:
      print 'WARNING: Displacement vector of mode %i is null vector. Ignoring this mode!' % (imode+1)
      modes[imode]['freq']=0.
    # now calculate the movement vectors, taking the mass of the atoms into account
    mu_reduced = 0.0
    for j, atom in enumerate(molecule):
      for xyz in range(3):
        mu_reduced += modes[imode]['move'][j][xyz]**2 / atom.mass
    mu_reduced = math.sqrt(mu_reduced)
    for j, atom in enumerate(molecule):
      for xyz in range(3):
        modes[imode]['move'][j][xyz] *= mu_reduced
    modes[imode]['mu_red'] = mu_reduced

  # delete low modes
  newmodes=[]
  for imode in range(nmodes):
    if modes[imode]['freq']<0.:
      print 'Detected negative frequency!'
    if modes[imode]['freq']>=LOW_FREQ*CM_TO_HARTREE:
      newmodes.append(modes[imode])
  modes=newmodes
  return molecule, modes

# ======================================================================================================================
# ======================================================================================================================
# ======================================================================================================================

def factorial(n):
    """This function recursively calculates the factorial of n."""
    if n == 0:
        return 1
    elif n == 1:
        return 1
    else:
        return n * factorial(n-1)

def laguerre(n, x):
    """This function recursively calculates the value of the nth order
Laguerre polynomial at point x."""
    if n == 0:
        return 1.0
    elif n == 1:
        return 1.0 - x
    else:
        L = 1.0 / n * ( (2.0*(n-1.0) + 1.0 - x) * laguerre(n-1, x)
                        - (n-1.0) * laguerre(n-2, x) )
        return L

def wigner(Q, P, n=0):
    """This function calculates the Wigner distribution for 
a single one-dimensional harmonic oscillator.
Q contains the dimensionless coordinate of the
oscillator and P contains the corresponding momentum.
n is the number of the vibrational state (default 0).
The function returns a probability for this set of parameters."""
    if n == 0: # vibrational ground state
        return math.exp(-Q**2) * math.exp(-P**2)
    else: # vibrational excited state
        rhosquare = 2.0 * (P**2 + Q**2)
        W = (-1.0)**n * laguerre(n,rhosquare) \
          * math.exp(-rhosquare/2.0)
        return W

def plot_wigner_functions():
    """This function creates several output files for the first 11
Wigner functions for Q and P in the interval [-3, +3].
It also plots the wigner distribution for Q from -6 to +6
for the first 11 vibrational states."""
    grid = [0.06*i for i in range(-100, 101)]
    outstring = ''
    for Q in grid:
        outstring += '%6.2f' % Q
        for n in range(11):
            # integrate over P
            W_tot = 0.0
            for P in grid:
                W_tot += wigner(Q, P, n)
            outstring += ' % 12.8e' % W_tot
        outstring += '\n'
    outfile = open('wignerplot.out', 'w')
    outfile.write(outstring)
    outfile.close()
    
    grid = [0.06*i for i in range(-100, 101)]
    outstring = ''
    for Q in grid:
        outstring += '%6.2f' % Q
        for n in range(11):
            outstring += ' % 12.8e' % laguerre(n, Q)
        outstring += '\n'
    outfile = open('laguerreplot.out', 'w')
    outfile.write(outstring)
    outfile.close()

    grid = [0.06*i for i in range(-100, 101)]
    for n in range(11):
        outstring = ''
        for Q in grid:
            for P in grid:
                W = wigner(Q, P, n)
                outstring += '%6.2f %6.2f % 12.8e\n' % (Q, P, W)
            outstring += '\n'
        filename = 'wigner%02i.out' % n
        outfile = open(filename, 'w')
        outfile.write(outstring)
        outfile.close()

def get_center_of_mass(molecule):
    """This function returns a list containing the center of mass
of a molecule."""
    mass = 0.0
    for atom in molecule:
        mass += atom.mass
    com = [0.0 for xyz in range(3)]
    for atom in molecule:
        for xyz in range(3):
            com[xyz] += atom.coord[xyz] * atom.mass / mass
    return com

def restore_center_of_mass(molecule, ic):
    """This function restores the center of mass for the distorted
geometry of an initial condition."""
    # calculate original center of mass
    com = get_center_of_mass(molecule)
    # caluclate center of mass for initial condition of molecule
    com_distorted = get_center_of_mass(ic)
    # get difference vector and restore original center of mass
    diff = [com[xyz] - com_distorted[xyz] for xyz in range(3)]
    for atom in ic:
        for xyz in range(3):
            atom.coord[xyz] += diff[xyz]

def remove_translations(ic):
    """This function calculates the movement of the center of mass
of an initial condition for a small timestep and removes this vector
from the initial condition's velocities."""
    # get center of mass at t = 0.0
    com = get_center_of_mass(ic)
    # get center of mass at t = dt = 0.01
    ic2 = copy.deepcopy(ic)
    dt = 0.01
    for atom in ic2:
        for xyz in range(3):
            atom.coord[xyz] += dt*atom.veloc[xyz]
    com2 = get_center_of_mass(ic2)
    # calculate velocity of center of mass and remove it
    v_com = [ (com2[xyz]-com[xyz])/dt for xyz in range(3) ]
    for atom in ic:
        for xyz in range(3):
            atom.veloc[xyz] -= v_com[xyz]
    if DEBUG:
        # check if v_com now is really zero
        # get center of mass at t = 0.0
        com = get_center_of_mass(ic)
        # get center of mass at t = dt = 1.0
        ic2 = copy.deepcopy(ic)
        dt = 1.0
        for atom in ic2:
            for xyz in range(3):
                atom.coord[xyz] += dt*atom.veloc[xyz]
        com2 = get_center_of_mass(ic2)
        # calculate velocity of center of mass and remove it
        v_com = [ (com2[xyz]-com[xyz])/dt for xyz in range(3) ]
        print v_com


def det(m):
    """This function calculates the determinant of a 3x3 matrix."""
    return m[0][0]*m[1][1]*m[2][2] + m[0][1]*m[1][2]*m[2][0] \
         + m[0][2]*m[1][0]*m[2][1] - m[0][0]*m[1][2]*m[2][1] \
         - m[0][1]*m[1][0]*m[2][2] - m[0][2]*m[1][1]*m[2][0]

def inverted(m):
    """This function calculates the inverse of a 3x3 matrix."""
    norm = m[0][0] * (m[1][1]*m[2][2] - m[1][2]*m[2][1]) \
         + m[0][1] * (m[1][2]*m[2][0] - m[1][0]*m[2][2]) \
         + m[0][2] * (m[1][0]*m[2][1] - m[1][1]*m[2][0])
    m_inv = [[0.0 for i in range(3)] for j in range(3)]
    m_inv[0][0] = (m[1][1]*m[2][2] - m[1][2]*m[2][1]) / norm
    m_inv[0][1] = (m[0][2]*m[2][1] - m[0][1]*m[2][2]) / norm
    m_inv[0][2] = (m[0][1]*m[1][2] - m[0][2]*m[1][1]) / norm
    m_inv[1][0] = (m[1][2]*m[2][0] - m[1][0]*m[2][2]) / norm
    m_inv[1][1] = (m[0][0]*m[2][2] - m[0][2]*m[2][0]) / norm
    m_inv[1][2] = (m[0][2]*m[1][0] - m[0][0]*m[1][2]) / norm
    m_inv[2][0] = (m[1][0]*m[2][1] - m[1][1]*m[2][0]) / norm
    m_inv[2][2] = (m[0][1]*m[2][0] - m[0][0]*m[2][1]) / norm
    m_inv[2][2] = (m[0][0]*m[1][1] - m[0][1]*m[1][0]) / norm
    return m_inv

def matmul(m1, m2):
    """This function multiplies two NxN matrices m1 and m2."""
    # get dimensions of resulting matrix
    n = len(m1)
    # calculate product
    result = [[0.0 for i in range(n)] for j in range(n)]
    for i in range(n):
        for j in range(n):
            for k in range(n):
                result[i][j] += m1[i][k]*m2[k][j]
    return result

def cross_prod(a, b):
    """This function calculates the cross product of two
3 dimensional vectors."""
    result = [0.0 for i in range(3)]
    result[0] = a[1]*b[2] - b[1]*a[2]
    result[1] = a[2]*b[0] - a[0]*b[2]
    result[2] = a[0]*b[1] - b[0]*a[1]
    return result

def linmapping(lm, y):
    z = [0.0 for i in range(3)]
    z[0] = lm[0][0]*y[0] + lm[0][1]*y[1] + lm[0][2]*y[2]
    z[1] = lm[1][0]*y[0] + lm[1][1]*y[1] + lm[1][2]*y[2]
    z[2] = lm[2][0]*y[0] + lm[2][1]*y[1] + lm[2][2]*y[2]
    return z

def remove_rotations(ic):
    # copy initial condition object
    ictmp = copy.deepcopy(ic)
    # move center of mass to coordinates (0, 0, 0)
    com = get_center_of_mass(ic)
    for atom in ictmp:
        for xyz in range(3):
            atom.coord[xyz] -= com[xyz]
    # calculate moment of inertia tensor
    I = [[0.0 for i in range(3)] for j in range(3)]
    for atom in ictmp:
        I[0][0] += atom.mass*(atom.coord[1]**2 + atom.coord[2]**2)
        I[1][1] += atom.mass*(atom.coord[0]**2 + atom.coord[2]**2)
        I[2][2] += atom.mass*(atom.coord[0]**2 + atom.coord[1]**2)
        I[0][1] -= atom.mass * atom.coord[0] * atom.coord[1]
        I[0][2] -= atom.mass * atom.coord[0] * atom.coord[2]
        I[1][2] -= atom.mass * atom.coord[1] * atom.coord[2]
    I[1][0] = I[0][1]
    I[2][0] = I[0][2]
    I[2][1] = I[1][2]
    if det(I) > 0.01: # checks if I is invertible
        ch = matmul(I, inverted(I))
        # calculate angular momentum
        ang_mom = [0.0 for i in range(3)]
        for atom in ictmp:
            mv = [0.0 for i in range(3)]
            for xyz in range(3):
                mv[xyz] = atom.mass * atom.veloc[xyz]
            L = cross_prod(mv, atom.coord)
            for xyz in range(3):
                ang_mom[xyz] -= L[xyz]
        # calculate angular velocity
        ang_vel = linmapping(inverted(I), ang_mom)
        for i,atom in enumerate(ictmp):
            v_rot = cross_prod(ang_vel, atom.coord) # calculate rotational velocity
            for xyz in range(3):
                ic[i].veloc[xyz] -= v_rot[xyz] # remove rotational velocity
    else:
        print 'WARNING: moment of inertia tensor is not invertible'

def constrain_displacement(molecule, ic, threshold=0.5):
    """This function ensures, that each atom of a generated initial
condition is not displaced further, than a given threshold from its
original position. Threshold is given in bohr."""
    for i, atom in enumerate(molecule):
        diff_vector = [ic[i].coord[xyz]-atom.coord[xyz]
                       for xyz in range(3)]
        displacement = 0.0
        for xyz in range(3):
            displacement += diff_vector[xyz]**2
        displacement = math.sqrt(displacement)
        if displacement > threshold:
            if DEBUG:
                print 'displacment for atom %i %s is %f' % (i, atom.symb, displacement)
            # shorten diff_vector to length of threshold
            for xyz in range(3):
                diff_vector[xyz] /= displacement/threshold
            # apply changes to initial condition
            for xyz in range(3):
                ic[i]['coords'][xyz] = atom.coord[xyz] \
                                       + diff_vector[xyz]

def sample_initial_condition(molecule, modes):
  """This function samples a single initial condition from the
modes and atomic coordinates by the use of a Wigner distribution.
The first atomic dictionary in the molecule list contains also
additional information like kinetic energy and total harmonic
energy of the sampled initial condition.
Method is based on L. Sun, W. L. Hase J. Chem. Phys. 133, 044313
(2010) nonfixed energy, independent mode sampling."""
  # copy the molecule in equilibrium geometry
  atomlist = copy.deepcopy(molecule) # initialising initial condition object
  Epot = 0.0
  for atom in atomlist:
    atom.veloc = [0.0, 0.0, 0.0] # initialise velocity lists
  for mode in modes: # for each uncoupled harmonatomlist oscillator
    while True:
      # get random Q and P in the interval [-3,+3]
      # this interval is good for vibrational ground state
      # should be increased for higher states
      random_Q = random.random()*6.0 - 3.0
      random_P = random.random()*6.0 - 3.0
      # calculate probability for this set of P and Q with Wigner distr.
      probability = wigner(random_Q, random_P)
      if probability>1. or probability<0.:
        print 'WARNING: wrong probability %f detected!' % (probability)
      elif probability > random.random():
        break # coordinates accepted
    # now transform the dimensionless coordinate into a real one
    # paper says, that freq_factor is sqrt(2*PI*freq)
    # molpro directly gives angular frequency (2*PI is not needed)
    freq_factor = math.sqrt(mode['freq'])
    # Higher frequencies give lower displacements and higher momentum.
    # Therefore scale random_Q and random_P accordingly:
    random_Q /= freq_factor 
    random_P *= freq_factor 
    # add potential energy of this mode to total potential energy
    Epot += 0.5 * mode['freq']**2 * random_Q**2
    for i, atom in enumerate(atomlist): # for each atom
      for xyz in range(3): # and each direction
        # distort geometry according to normal mode movement
        # factor of sqrt(1/2) is necessary to distribute energy evenly over
        # kinetatomlist and potential energy
        atom.coord[xyz] += random_Q * mode['move'][i][xyz] * math.sqrt(0.5)
        # add velocity
        atom.veloc[xyz] += random_P * mode['move'][i][xyz] * math.sqrt(0.5)
      atom.EKIN()
  if not KTR:
    restore_center_of_mass(molecule, atomlist)
    remove_translations(atomlist)
    remove_rotations(atomlist)

  ic = INITCOND(atomlist,0.,Epot)
  return ic

# ======================================================================================================================
# ======================================================================================================================
# ======================================================================================================================

#def initial_condition_to_string(ic):
    #"""This function converts an initial condition into a formatted
#string and returns it."""
    #outstring = 'Geometry of molecule (in bohr):\n'
    #for atom in ic.atomlist:
        #outstring += ' %2s %5.1f %12.8f %12.8f %12.8f %12.8f\n' \
                     #% (atom.symb, NUMBERS[atom.symb],
                        #atom.coord[0], atom.coord[1],
                        #atom.coord[2], atom.mass/U_TO_AMU)
    #outstring += 'Velocities of the single atoms (a.u.):\n'
    #for atom in ic.atomlist:
        #outstring += '%12.8f %12.8f %12.8f\n' % (atom.veloc[0],
                                #atom.veloc[1], atom.veloc[2])
    #for key in sorted(ic[0].keys(), key=str.lower):
        #if key not in ('veloc', 'symbol', 'coords', 'mass') and not key.startswith('Excited state'):
            #outstring += '%s: %12.8f a.u. ( %12.8f eV)\n' \
                          #% (key, ic[0][key], ic[0][key]*HARTREE_TO_EV)
        #elif key.startswith('Excited state'):
            #outstring += '%s:\n' % key
            #for exckey in sorted(ic[0][key].keys(), key=str.lower):
                #if exckey in ('E_final', 'E_exc'):
                    #outstring += '    %s: %12.8f a.u. ( %12.8f eV)\n' \
                     #% (exckey, ic[0][key][exckey], ic[0][key][exckey]*HARTREE_TO_EV)
                #elif exckey in ('Osc'):
                    #outstring += '    %s: %12.8f\n' % (exckey, ic[0][key][exckey])
                #elif exckey in ('Excitation'):
                    #outstring += '    %s: %s\n' % (exckey, ic[0][key][exckey])
                #else:
                    #print exckey, key, ic[0][key][exckey]
    #return outstring

# ======================================================================================================================

def create_initial_conditions_string(molecule, modes, ic_list, eref=0.0):
  """This function converts an list of initial conditions into a string."""
  ninit=len(ic_list)
  natom=ic_list[0].natom
  representation='None'
  #eref
  eharm=0.
  for mode in modes:
    eharm+=mode['freq']*0.5
  string='''SHARC Initial conditions file, version %s
Ninit     %i
Natom     %i
Repr      %s
Eref      %18.10f
Eharm     %18.10f

Equilibrium
''' % (version,ninit,natom,representation,eref,eharm)
  for atom in molecule:
    string+=str(atom)+'\n'
  string+='\n\n'

  for i, ic in enumerate(ic_list):
    string += 'Index     %i\n%s' % (i+1, str(ic))
  return string

# ======================================================================================================================

def create_initial_conditions_list(amount, molecule, modes):
    """This function creates 'amount' initial conditions from the
data given in 'molecule' and 'modes'. Output is returned 
as a list containing all initial condition objects."""
    ic_list = []
    for i in range(1,amount+1): # for each requested initial condition
        # sample the initial condition
        ic = sample_initial_condition(molecule, modes)
        ic_list.append(ic)
    return ic_list

# ======================================================================================================================

def make_dyn_file(states, ic_list):
  if not os.path.exists('init_geoms'):
    os.mkdir('init_geoms')
  for state in range(states):
    fl=open('init_geoms/state_%i.xyz' % (state+1),'w')
    string=''
    for i,ic in enumerate(ic_list):
      if 'Excited state %i' % (state) in ic[0] and ic[0]['Excited state %i' % (state)]['Excitation']:
        string+='%i\n%i\n' % (len(ic),i)
        for atom in ic:
          string+='%s' % (atom.symb)
          for j in range(3):
            string+=' %f' % (atom.coord[j]/ANG_TO_BOHR)
          string+='\n'
    fl.write(string)
    fl.close()

# ======================================================================================================================
# ======================================================================================================================
# ======================================================================================================================

def main():
  '''Main routine'''

  usage='''
Wigner.py [options] filename.molden

This script reads a MOLDEN file containing frequencies and normal modes [1]
and generates a Wigner distribution of geometries and velocities.

The creation of the geometries and velocities is based on the 
sampling of the Wigner distribution of a quantum harmonic oscillator, 
as described in [2] (non-fixed energy, independent mode sampling).

[1] http://www.cmbi.ru.nl/molden/molden_format.html
[2] L. Sun, W. L. Hase: J. Chem. Phys. 133, 044313 (2010)
'''

  description=''

  parser = OptionParser(usage=usage, description=description)
  parser.add_option('-n', dest='n', type=int, nargs=1, default=3, help="Number of geometries to be generated (integer, default=3)")
  parser.add_option('-r', dest='r', type=int, nargs=1, default=16661, help="Seed for the random number generator (integer, default=16661)")
  parser.add_option('-o', dest='o', type=str, nargs=1, default='initconds', help="Output filename (string, default=""initconds"")")
  parser.add_option('--MOLPRO', dest='M', action='store_true',help="Assume a MOLPRO frequency file (default=assume MOLDEN file)")
  parser.add_option('-m', dest='m', action='store_true',help="Enter non-default atom masses")
  parser.add_option('-s', dest='s', type=float, nargs=1, default=1.0, help="Scaling factor for the energies (float, default=1.0)")
  parser.add_option('--keep_trans_rot', dest='KTR', action='store_true',help="Keep translational and rotational components")
  (options, args) = parser.parse_args()
  random.seed(options.r)
  amount=options.n
  if len(args)==0:
    print usage
    quit(1)
  filename=args[0]
  outfile=options.o
  nondefmass=options.m
  scaling=options.s
  if not options.M:
    options.M=False

  print '''Initial condition generation started...
%s file                  = "%s"
OUTPUT file                  = "%s"
Number of geometries         = %i
Random number generator seed = %i''' % (['MOLDEN','MOLPRO'][options.M], filename, outfile, options.n, options.r)
  if nondefmass:
    global MASS_LIST
    MASS_LIST = ask_for_masses()
  else:
    print ''
  if scaling!=1.0:
    print 'Scaling factor               = %f\n' % (scaling)

  global KTR
  KTR=options.KTR

  global whichatoms
  whichatoms=[]
  if options.M:
    molecule, modes = import_from_molpro(filename, scaling)
  else:
    molecule, modes = import_from_molden(filename, scaling)

  string='Geometry:\n'
  for atom in molecule:
    string+=str(atom)[:61]+'\n'
  string+='Assumed Isotopes: '
  for i in set(whichatoms):
    string+=ISOTOPES[i]+' '
  string+='\nIsotopes with * are pure isotopes.\n'
  print string

  string='Frequencies (cm^-1) used in the calculation:\n'
  for i,mode in enumerate(modes):
    string+='%4i %12.4f\n' % (i+1,mode['freq']/CM_TO_HARTREE)
  print string

  #print 'Generating %i initial conditions' % amount
  ic_list = create_initial_conditions_list(amount, molecule, modes)
  #print 'Writing output to initconds'
  outfile = open(outfile, 'w')
  outstring = create_initial_conditions_string(molecule, modes, ic_list)
  outfile.write(outstring)
  outfile.close()

  # save the shell command
  command='python '+' '.join(sys.argv)
  f=open('KEYSTROKES.wigner','w')
  f.write(command)
  f.close()

# ======================================================================================================================

if __name__ == '__main__':
    main()
