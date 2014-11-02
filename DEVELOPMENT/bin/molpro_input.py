#!/usr/bin/env python2

# Script for the calculation of Wigner distributions from molden frequency files
# 
# usage 

import copy
import math
import sys
import re
import os
import stat
import shutil
import datetime
from optparse import OptionParser
import readline
import time

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
BOHR_TO_ANG=0.529177211
PI = math.pi

version='1.0'
versiondate=datetime.date(2014,10,8)


# List of atomic numbers until Rn, with Lanthanoids missing (1-57, 72-86)
NUMBERS = {'H':  1, 'He': 2,
'Li': 3, 'Be': 4, 'B':  5, 'C':  6,  'N': 7,  'O': 8, 'F':  9, 'Ne':10,
'Na':11, 'Mg':12, 'Al':13, 'Si':14,  'P':15,  'S':16, 'Cl':17, 'Ar':18,
'K': 19, 'Ca':20, 
'Sc':21, 'Ti':22, 'V': 23, 'Cr':24, 'Mn':25, 'Fe':26, 'Co':27, 'Ni':28, 'Cd':29, 'Zn':30,
'Ge':31, 'Ga':32, 'As':33, 'Se':34, 'Br':35, 'Kr':36, 
'Rb':37, 'Sr':38,
'Y':39,  'Zr':40, 'Nb':41, 'Mo':42, 'Tc':43, 'Ru':44, 'Rh':45, 'Pd':46, 'Ag':47, 'Cd':48,
'In':49, 'Sn':50, 'Sb':51, 'Te':52,  'I':53, 'Xe':54,
'Cs':55, 'Ba':56,
'La':57, 'Hf':72, 'Ta':73,  'W':74, 'Re':75, 'Os':76, 'Ir':77, 'Pt':78, 'Au':79, 'Hg':80,
'Tl':81, 'Pb':82, 'Bi':83, 'Po':84, 'At':85, 'Rn':86
}


MASSES = {'H' :   1.00782,
          'He':   4.00260,
          'Li':   7.01600,
          'Be':   9.01218,
          'B' :  11.00931,
          'C' :  12.00000,
          'N' :  14.00307,
          'O' :  15.99491,
          'F' :  18.99840,
          'Ne':  19.99244,
          'Na':  22.98980,
          'Mg':  23.98504,
          'Al':  26.98153,
          'Si':  27.97693,
          'P' :  30.97376,
          'S' :  31.97207,
          'Cl':  34.96885,
          'Ar':  39.96238,
          'K' :  38.96371,
          'Ca':  39.96259,
          'Sc':  44.95592,
          'Ti':  47.94795,
          'V' :  50.94400,
          'Cr':  51.94050,
          'Mn':  54.93800,
          'Fe':  55.93490,
          'Co':  58.93320,
          'Ni':  57.93534,
          'Cu':  62.92960,
          'Zn':  63.92910,
          'Ga':  68.92570,
          'Ge':  73.92190,
          'As':  74.92160,
          'Se':  79.91650,
          'Br':  78.91830,
          'Kr':  83.80000,
          'Rb':  84.91170,
          'Sr':  87.90560,
          'Y' :  88.90590,
          'Zr':  89.90430,
          'Nb':  92.90600,
          'Mo':  97.90550,
          'Tc':  98.90620,
          'Ru': 101.90370,
          'Rh': 102.90480,
          'Pd': 105.90320,
          'Ag': 106.90509,
          'Cd': 113.90360,
          'In': 114.90410,
          'Sn': 119.90220,   # MOLPRO library is wrong
          'Sb': 120.90380,
          'Te': 129.90670,
          'I' : 126.90440,
          'Xe': 131.90420,
          'Cs': 132.90510,
          'Ba': 137.90500,
          'La': 138.90610,
          'Hf': 179.94680,
          'Ta': 180.94800,
          'W' : 183.95100,
          'Re': 186.95600,
          'Os': 190.20000,
          'Ir': 192.96330,
          'Pt': 194.96480,
          'Au': 196.96660,
          'Hg': 201.97060,
          'Tl': 204.97450,
          'Pb': 207.97660,
          'Bi': 208.98040,
          'Po': 208.98250,
          'At': 209.98715,   # MOLPRO library is wrong
          'Rn': 210.99060}   # MOLPRO library is wrong

# ======================================================================================================================
# ======================================================================================================================
# ======================================================================================================================

def centerstring(string,n,pad=' '):
  l=len(string)
  if l>=n:
    return string
  else:
    return  pad*((n-l+1)/2)+string+pad*((n-l)/2)

def displaywelcome():
  string='\n'
  string+='  '+'='*80+'\n'
  string+='||'+centerstring('',80)+'||\n'
  string+='||'+centerstring('MOLPRO Input file generator',80)+'||\n'
  string+='||'+centerstring('',80)+'||\n'
  string+='||'+centerstring('Author: Sebastian Mai',80)+'||\n'
  string+='||'+centerstring('',80)+'||\n'
  string+='||'+centerstring('Version:'+version,80)+'||\n'
  string+='||'+centerstring(versiondate.strftime("%d.%m.%y"),80)+'||\n'
  string+='||'+centerstring('',80)+'||\n'
  string+='  '+'='*80+'\n\n'
  string+='''
This script allows to quickly create MOLPRO input files for single-points calculations,
ground state optimizations, frequency calculations and SA-CASSCF calculations. 
It also generates MOLPRO.template files to be used with the SHARC-MOLPRO Interface.
  '''
  print string

# ======================================================================================================================
# ======================================================================================================================
# ======================================================================================================================


def open_keystrokes():
  global KEYSTROKES
  KEYSTROKES=open('KEYSTROKES.tmp','w')

def close_keystrokes():
  KEYSTROKES.close()
  shutil.move('KEYSTROKES.tmp','KEYSTROKES.molpro_input')

# ===================================

def question(question,typefunc,default=None,autocomplete=True):
  if typefunc==int or typefunc==float:
    if not default==None and not isinstance(default,list):
      print 'Default to int or float question must be list!'
      quit(1)
  if typefunc==str and autocomplete:
    readline.set_completer_delims(' \t\n;')
    readline.parse_and_bind("tab: complete")    # activate autocomplete
  else:
    readline.parse_and_bind("tab: ")            # deactivate autocomplete

  while True:
    s=question
    if default!=None:
      if typefunc==bool or typefunc==str:
        s+= ' [%s]' % (str(default))
      elif typefunc==int or typefunc==float:
        s+= ' ['
        for i in default:
          s+=str(i)+' '
        s=s[:-1]+']'
    if typefunc==str and autocomplete:
      s+=' (autocomplete enabled)'
    s+=' '

    line=raw_input(s)
    line=re.sub('#.*$','',line).strip()
    if not typefunc==str:
      line=line.lower()

    if line=='' or line=='\n':
      if default!=None:
        KEYSTROKES.write(line+' '*(40-len(line))+' #'+s+'\n')
        return default
      else:
        continue

    if typefunc==bool:
      posresponse=['y','yes','true', 'ja',  'si','yea','yeah','aye','sure','definitely']
      negresponse=['n','no', 'false','nein',     'nope']
      if line in posresponse:
        KEYSTROKES.write(line+' '*(40-len(line))+' #'+s+'\n')
        return True
      elif line in negresponse:
        KEYSTROKES.write(line+' '*(40-len(line))+' #'+s+'\n')
        return False
      else:
        print 'I didn''t understand you.'
        continue

    if typefunc==str:
      KEYSTROKES.write(line+' '*(40-len(line))+' #'+s+'\n')
      return line

    if typefunc==int or typefunc==float:
      # int and float will be returned as a list
      f=line.split()
      try:
        for i in range(len(f)):
          f[i]=typefunc(f[i])
        KEYSTROKES.write(line+' '*(40-len(line))+' #'+s+'\n')
        return f
      except ValueError:
        if typefunc==int:
          i=1
        elif typefunc==float:
          i=2
        print 'Please enter a %s' % ( ['string','integer','float'][i] )
        continue

# ======================================================================================================================
# ======================================================================================================================
# ======================================================================================================================

def show_massses(masslist):
  s='Number\tType\tMass\n'
  for i,atom in enumerate(masslist):
    s+='%i\t%2s\t%12.9f %s\n' % (i+1,atom[0],atom[1], ['','*'][atom[1]!=MASSES[atom[0]]])
  print s

def ask_for_masses(masslist):
  print '''
Please enter non-default masses:
+ number mass           use non-default mass <mass> for atom <number>
- number                remove non-default mass for atom <number> (default mass will reinstated)
show                    show atom masses
end                     finish input for non-default masses
'''
  show_massses(masslist)
  while True:
    line=question('Change an atoms mass:',str,'end',False)
    if 'end' in line:
      break
    if 'show' in line:
      show_massses(masslist)
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
      if not 0<=num<=len(masslist):
        print 'Atom %i does not exist!' % (num)
        continue
      masslist[num-1][1]=mass
      continue
    if '-' in line:
      f=line.split()
      if len(f)<2:
        continue
      try:
        num=int(f[1])
      except ValueError:
        continue
      if not 0<=num<=len(masslist):
        print 'Atom %i does not exist!' % (num)
        continue
      masslist[num-1][1]=MASSES[masslist[num-1][0]]
      continue
  return masslist

# ======================================================================================================================
# ======================================================================================================================
# ======================================================================================================================

def get_infos():
  '''Asks for the settings of the calculation:
- type (single point, optimization+freq or MOLPRO.template
- level of theory
- basis set
- douglas kroll
- memory
- geometry

specific:
- opt: freq?
- CASSCF: docc, act'''

  INFOS={}

  # Type of calculation
  print centerstring('Type of calculation',60,'-')
  print '''\nThis script generates input for the following types of calculations:
  1       Single point calculations (HF, DFT, MP2, SS/SA-CASSCF)
  2       Optimizations & Frequency calculations (HF, DFT, MP2, SS/SA-CASSCF)
  3       MOLPRO.template file for dynamics (SA-CASSCF)
Please enter the number corresponding to the type of calculation.
'''
  while True:
    ctype=question('Type of calculation:',int)[0]
    if not ctype in [1,2,3]:
      print 'Enter an integer (1-3)!'
      continue
    break
  INFOS['ctype']=ctype
  freq=False
  if ctype==2:
    freq=question('Frequency calculation?',bool,True)
  INFOS['freq']=freq
  print ''


  guessnact=None
  guessnorb=None
  guessnelec=None
  guessbase=None
  guessstates=None
  guessmem=None


  # Geometry
  print centerstring('Geometry',60,'-')
  if ctype==3:
    print '\nNo geometry necessary for MOLPRO.template generation\n'
    INFOS['geom']=None
    # see whether a MOLPRO.input file is there, where we can take the number of electrons from
    nelec=0
    try:
      molproinput=open('MOLPRO.input','r')
      for line in molproinput:
        if 'wf,' in line and not './wf,' in line:
          guessnelec=[int(line.split(',')[1])]
          mult=int(line.split(',')[3])
        if 'state,' in line:
          if guessstates==None:
            guessstates=[]
          nstate=int(line.split(',')[1])
          for i in range(mult-len(guessstates)):
            guessstates.append(0)
          guessstates.append(nstate)
        if 'closed,' in line:
          nclosed=int(line.split(',')[1])
        if 'occ,' in line:
          nocc=int(line.split(',')[1])
        if 'basis=' in line:
          guessbase=line.split('=')[1].strip()
        if 'memory' in line:
          guessmem=[int(line.split(',')[1])/125]
      try:
        guessnorb=[nocc-nclosed]
        guessnact=[guessnelec[0]-2*nclosed]
      except:
        pass
    except (IOError,ValueError):
      pass
    # continue with asking for number of electrons
    while True:
      nelec=question('Number of electrons: ',int,guessnelec,False)[0]
      if nelec<=0:
        print 'Enter a positive number!'
        continue
      break
    INFOS['nelec']=nelec
  else:
    print '\nPlease specify the geometry file (xyz format, Angstroms):'
    while True:
      path=question('Geometry filename:',str,'geom.xyz')
      try:
        gf=open(path,'r')
      except IOError:
        print 'Could not open: %s' % (path)
        continue
      g=gf.readlines()
      gf.close()
      try:
        natom=int(g[0])
      except ValueError:
        print 'Malformatted: %s' % (path)
        continue
      geom=[]
      ncharge=0
      fine=True
      for i in range(natom):
        try:
          line=g[i+2].split()
        except IndexError:
          print 'Malformatted: %s' % (path)
          fine=False
        try:
          atom=[line[0],float(line[1]),float(line[2]),float(line[3])]
        except (IndexError,ValueError):
          print 'Malformatted: %s' % (path)
          fine=False
          continue
        geom.append(atom)
        try:
          ncharge+=NUMBERS[atom[0]]
        except KeyError:
          print 'Atom type %s not supported!' % (atom[0])
          fine=False
      if not fine:
        continue
      else:
        break
    print 'Number of atoms: %i\nNuclear charge: %i\n' % (natom,ncharge)
    INFOS['geom']=geom
    INFOS['ncharge']=ncharge
    INFOS['natom']=natom
    print 'Enter the total (net) molecular charge:'
    while True:
      charge=question('Charge:',int,[0])[0]
      break
    INFOS['nelec']=ncharge-charge
    print 'Number of electrons: %i\n' % (ncharge-charge)

  # Masses
  if INFOS['freq']:
    # make default mass list
    masslist=[]
    for atom in geom:
      masslist.append( [atom[0],MASSES[atom[0]]] )
    # ask
    INFOS['nondefmass']=not question('Use standard masses (most common isotope)?',bool,True)
    if INFOS['nondefmass']:
      INFOS['masslist']=ask_for_masses(masslist)
    else:
      INFOS['masslist']=masslist

  # Level of theory
  print '\n'+centerstring('Level of theory',60,'-')
  print '''\nSupported by this script are:
  1       HF
  2       DFT %s
  3       MP2 %s
  4       SS-CASSCF
  5       SA-CASSCF %s
''' % tuple(3*[['','(Only numerical frequencies)'][INFOS['freq']]])
  if ctype==3:
    ltype=5
    print 'Choosing SA-CASSCF for MOLPRO.template generation.'
  else:
    while True:
      ltype=question('Level of theory:',int)[0]
      if not ltype in [1,2,3,4,5]:
        print 'Enter an integer (1-5)!'
        continue
      break
  INFOS['ltype']=ltype

  # DFT
  if ltype==2:
    print '''Commons functionals and their names in MOLPRO:
  B3LYP      B3LYP, B3LYP3, B3LYP5
  BP86       B-P
  PBE        PBE
  PBE0       PBE0
'''
    func=question('Functional:',str,None,False)
    INFOS['dft.func']=func
    disp=question('Dispersion correction? ',bool)
    INFOS['dft.disp']=disp

  # basis set
  print '\nPlease enter the basis set.'
  cadpac=(ctype==2 and ltype+freq>=5) or ctype==3
  if ctype==2 and ltype+freq>=5:
    print 'For SA-CASSCF Optimizations/Frequencies and SS-CASSCF Frequencies,\nonly segmented basis sets are allowed.'
  if ctype==3:
    print 'For MOLPRO.template generation, only segmented basis sets are allowed.'
  print '''Common available basis sets:
  Pople:     6-31G**, 6-311G, 6-31+G, 6-31G(d,p), ...
  Dunning:   cc-pVXZ, aug-cc-pVXZ, cc-pVXZ-DK, ...    %s
  Turbomole: def2-SV(P), def2-SVP, def2-TZVP, ...
  ANO:       ROOS                                     %s''' % (['','not available'][cadpac],['','not available'][cadpac])
  basis=question('Basis set:',str,guessbase,False)
  INFOS['basis']=basis

  # douglas kroll
  dk=question('Douglas-Kroll scalar-relativistic integrals?',bool,True)
  INFOS['DK']=dk

  # CASSCF
  if ltype>=4:
    print '\n'+centerstring('CASSCF Settings',60,'-')+'\n'
    while True:
      nact=question('Number of active electrons:',int,guessnact)[0]
      if nact<=0:
        print 'Enter a positive number!'
        continue
      if (INFOS['nelec']-nact)%2!=0:
        print 'nelec-nact must be even!'
        continue
      if INFOS['nelec']<nact:
        print 'Number of active electrons cannot be larger than total number of electrons!'
        continue
      break
    INFOS['cas.nact']=nact
    while True:
      norb=question('Number of active orbitals:',int,guessnorb)[0]
      if norb<=0:
        print 'Enter a positive number!'
        continue
      if norb>2*nact:
        print 'norb cannot be larger than 2*nact!'
        continue
      break
    INFOS['cas.norb']=norb

  if ltype<5:
    print '\nPlease enter the multiplicity (1=singlet, 2=doublet, 3=triplet, ...)'
    while True:
      mult=question('Multiplicity:',int,[1])[0]
      if mult<=0:
        print 'Enter a positive number!'
        continue
      if (INFOS['nelec']-mult-1)%2!=0:
        print 'Nelec is %i, so mult cannot be %i' % (INFOS['nelec'],mult)
        continue
      break
    INFOS['mult']=mult
    if ltype==4:
      INFOS['cas.nstates']=[0 for i in range(mult)]
      INFOS['cas.nstates'][mult-1]=1
      INFOS['maxmult']=mult
  elif ltype==5:
    print 'Please enter the number of states as a list of integers\ne.g. 3 0 3 for three singlets, zero doublets and three triplets.'
    states=question('Number of states:',int,guessstates)
    maxmult=len(states)
    for i in range(maxmult):
      n=states[i]
      if (not i%2==INFOS['nelec']%2) and int(n)>0:
        print 'Nelec is %i. Ignoring states with mult=%i!' % (INFOS['nelec'], i+1)
        states[i]=0
      if n<0:
        states[i]=0
    s='Accepted number of states:'
    for i in states:
      s+=' %i' % (i)
    print s
    INFOS['maxmult']=len(states)
    INFOS['cas.nstates']=states
    if ctype==2:
      print '\nPlease specify the state to optimize\ne.g. 3 2 for the second triplet state.'
      while True:
        rmult,rstate=tuple(question('Root:',int,[1,1]))
        if not 1<=rmult<=INFOS['maxmult']:
          print '%i must be between 1 and %i!' % (rmult,INFOS['maxmult'])
          continue
        if not 1<=rstate<=states[rmult-1]:
          print 'Only %i states of mult %i' % (states[rmult-1],rmult)
          continue
        break
      INFOS['cas.root']=[rmult,rstate]
    if ctype==1 and maxmult>1:
      INFOS['soci']=question('Do Spin-Orbit CASCI after CASSCF?',bool,False)

  print '\n'+centerstring('Memory',60,'-')
  print '\nRecommendation: for small systems: 100-300 MB, for medium-sized systems: 1000-2000 MB\n'
  mem=abs(question('Memory in MB: ',int,guessmem)[0])
  mem=max(mem,50)
  INFOS['mem']=mem

  print ''

  return INFOS

# ======================================================================================================================
# ======================================================================================================================
# ======================================================================================================================

def setup_input(INFOS):
  ''''''

  if INFOS['ctype']==3:
    inpf='MOLPRO.template'
  else:
    inpf='MOLPRO.input'
  print 'Writing input to %s' % (inpf)
  try:
    inp=open(inpf,'w')
  except IOError:
    print 'Could not open %s for write!' % (inpf)
    quit(1)

  s='***,%s generated by molpro_input.py Version %s\n' % (inpf,version)
  s+='memory,%i,k\n\n' % (INFOS['mem']*125)     # convert to Mega-Words
  if INFOS['ctype']<3:
    s+='file,1,./integrals,scratch\n'
    s+='file,2,./wf,new   ! remove ",new" if you want to restart\n'
  s+='\n\nprint,orbitals,civectors;\n\n'
  if INFOS['DK']:
    s+='dkroll=1\ndkho=2\n'
  s+='basis=%s\n\n' % (INFOS['basis'])

  if INFOS['geom']:
    s+='nosym\nangstrom\ngeometry={\n'
    for iatom,atom in enumerate(INFOS['geom']):
      s+='%s%i % 16.9f % 16.9f % 16.9f\n' % (atom[0],iatom+1,atom[1],atom[2],atom[3])
    s+='}\n\n'
  if INFOS['freq']:
    s+='mass,isotope\n'
    for iatom,atom in enumerate(INFOS['geom']):
      s+='mass,,%s%i=%f\n' % (atom[0],iatom+1,INFOS['masslist'][iatom][1])
    s+='mass,print\n\n'

  if INFOS['ltype']==1:
    if INFOS['nelec']%2==0:
      s+='{hf'
    else:
      s+='{uhf'
    if INFOS['mult']!=1 or INFOS['ncharge']!=INFOS['nelec']:
      s+='\nwf,%i,1,%i\n' % (INFOS['nelec'],INFOS['mult']-1)
    s+='};\n\n'
  elif INFOS['ltype']==2:
    if INFOS['nelec']%2==0:
      s+='{ks'
    else:
      s+='{uks'
    s+=',%s' % (INFOS['dft.func'])
    if INFOS['dft.disp']:
      s+=';disp'
    if INFOS['mult']!=1 or INFOS['ncharge']!=INFOS['nelec']:
      s+='\nwf,%i,1,%i\n' % (INFOS['nelec'],INFOS['mult']-1)
    s+='};\n\n'
  elif INFOS['ltype']==3:
    if INFOS['nelec']%2==0:
      s+='{hf'
    else:
      s+='{uhf'
    if INFOS['mult']!=1 or INFOS['ncharge']!=INFOS['nelec']:
      s+='\nwf,%i,1,%i\n' % (INFOS['nelec'],INFOS['mult']-1)
    if INFOS['nelec']%2==0:
      s+='};\n{mp2};\n\n'
    else:
      s+='};\n{ump2};\n\n'
  elif INFOS['ltype']>=4:
    s+='{casscf\n'
    s+='frozen,0\nclosed,%i\n' % ((INFOS['nelec']-INFOS['cas.nact'])/2)
    s+='occ,%i\n' % (INFOS['cas.norb']+(INFOS['nelec']-INFOS['cas.nact'])/2)
    if INFOS['ctype']<3:
      s+='!start,2140.2       ! uncomment if restarting\n'
      s+='orbital,2140.2\n'
    if INFOS['ctype']==1:
      s+='!rotate,-1.1,-1.1   ! uncomment if rotating orbitals\n'
    for i,n in enumerate(INFOS['cas.nstates']):
      if n==0:
        continue
      s+='wf,%i,1,%i\n' % (INFOS['nelec'],i)
      s+='state,%i\n' % (n)
      s+='weight'+',1'*n+'\n'

    if INFOS['ctype']==2:
      if INFOS['ltype']==5:
        s+='\ncpmcscf,grad,state=%i.1,ms2=%i,record=5001.2,accu=1e-7\n' % (INFOS['cas.root'][1],INFOS['cas.root'][0]-1)
      if INFOS['ltype']==4 and INFOS['freq']:
        s+='\ncpmcscf,hess,accu=1e-4\n'
    s+='};\n\n'

  if INFOS['ctype']==2:
    s+='{optg,maxit=50};\n'
    if INFOS['freq']:
      s+='{frequencies};\n'
    s+='\n'

  if INFOS['ctype']==1:
    s+='PUT,MOLDEN,geom.molden\n'
  elif INFOS['ctype']==2:
    if INFOS['freq']:
      s+='PUT,MOLDEN,freq.molden\n'
    else:
      s+='PUT,MOLDEN,opt.molden\n'

  if 'soci' in INFOS and INFOS['soci']:
    s+='\n\n'
    for i,n in enumerate(INFOS['cas.nstates']):
      if n==0:
        continue
      s+='{ci\norbital,2140.2\nsave,%i.2\nnoexc\ncore,%i\n' % (6001+i,(INFOS['nelec']-INFOS['cas.nact'])/2)
      s+='wf,%i,%i,%i\nstate,%i\n}\n\n' % (INFOS['nelec'],1,i+1,n)
    s+='{ci\nhlsmat,amfi'
    for i,n in enumerate(INFOS['cas.nstates']):
      if n==0:
        continue
      s+=',%i.2' % (6001+i)
    s+='\nprint,hls=1\n}\n\n'

  s+='\n\n---\n'
  s+='!Infos:\n'
  s+='!%s@%s\n' % (os.environ['USER'],os.environ['HOSTNAME'])
  s+='!Date: %s\n' % (datetime.datetime.now())
  s+='!Current directory: %s\n\n' % (os.getcwd())

  inp.write(s)

# ======================================================================================================================
# ======================================================================================================================
# ======================================================================================================================

def set_runscript(INFOS):

  if INFOS['ctype']>=3:
    return

  print ''
  if not question('Runscript?',bool,True):
    return
  print ''

  # MOLPRO executable
  print centerstring('Path to MOLPRO',60,'-')+'\n'
  path=os.getenv('MOLPRO')
  path=os.path.expanduser(os.path.expandvars(path))
  if not path.endswith('/molpro'):
    path+='/molpro'
  if path!='':
    print 'Environment variable $MOLPRO detected:\n$MOLPRO=%s\n' % (path)
    if question('Do you want to use this MOLPRO installation?',bool,True):
      INFOS['molpro']=path
  if not 'molpro' in INFOS:
    print '\nPlease specify path to MOLPRO directory (SHELL variables and ~ can be used, will be expanded when interface is started).\n'
    INFOS['molpro']=question('Path to MOLPRO:',str)
  print ''


  # Scratch directory
  print centerstring('Scratch directory',60,'-')+'\n'
  print 'Please specify an appropriate scratch directory. This will be used to temporally store the integrals. The scratch directory will be deleted after the calculation. Remember that this script cannot check whether the path is valid, since you may run the calculations on a different machine. The path will not be expanded by this script.'
  INFOS['scratchdir']=question('Path to scratch directory:',str)+'/WORK'
  print ''

  runscript='run_MOLPRO.sh'
  print 'Writing run script %s' % (runscript)
  try:
    runf=open(runscript,'w')
  except IOError:
    print 'Could not write %s' (runscript)
    return

  string='''#!/bin/bash

PRIMARY_DIR=%s
SCRATCH_DIR=%s
cd $PRIMARY_DIR
mkdir -p $SCRATCH_DIR

%s MOLPRO.input -W$PRIMARY_DIR -I$SCRATCH_DIR -d$SCRATCH_DIR

rm -r $SCRATCH_DIR  ''' % (os.getcwd(), INFOS['scratchdir'], INFOS['molpro'])

  runf.write(string)
  runf.close()
  os.chmod(runscript, os.stat(runscript).st_mode | stat.S_IXUSR)

# ======================================================================================================================
# ======================================================================================================================
# ======================================================================================================================

def main():
  '''Main routine'''

  usage='''
python molpro_input.py

This interactive program prepares a MOLPRO input file for ground state optimizations and frequency calculations with HF, DFT, MP2 and CASSCF. It also generates input for SA-CASSCF excited-state calculations (MOLPRO.template files to be used with the SHARC-MOLPRO interface).
'''

  description=''
  parser = OptionParser(usage=usage, description=description)

  displaywelcome()
  open_keystrokes()

  INFOS=get_infos()

  print centerstring('Full input',60,'#')+'\n'
  for item in INFOS:
    print item, ' '*(15-len(item)), INFOS[item]
  print ''

  setup_input(INFOS)
  set_runscript(INFOS)
  print '\nFinished\n'

  close_keystrokes()

# ======================================================================================================================

if __name__ == '__main__':
  try:
    main()
  except KeyboardInterrupt:
    print '\nCtrl+C makes me a sad SHARC ;-(\n'
    quit(0)