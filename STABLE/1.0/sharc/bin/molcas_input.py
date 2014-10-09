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
  string+='||'+centerstring('MOLCAS template file generator',80)+'||\n'
  string+='||'+centerstring('',80)+'||\n'
  string+='||'+centerstring('Author: Sebastian Mai',80)+'||\n'
  string+='||'+centerstring('',80)+'||\n'
  string+='||'+centerstring('Version:'+version,80)+'||\n'
  string+='||'+centerstring(versiondate.strftime("%d.%m.%y"),80)+'||\n'
  string+='||'+centerstring('',80)+'||\n'
  string+='  '+'='*80+'\n\n'
  string+='''
This script allows to quickly create template files to be used with the SHARC-MOLCAS interface.
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
  shutil.move('KEYSTROKES.tmp','KEYSTROKES.molcas_input')

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


  INFOS['ctype']=3
  INFOS['freq']=False


  # Geometry
  print centerstring('Geometry',60,'-')
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

  ltype=5
  INFOS['ltype']=ltype

  # basis set
  print '\nPlease enter the basis set.'
  basis=question('Basis set:',str,autocomplete=False)
  INFOS['basis']=basis


  # CASSCF
  if ltype>=4:
    print '\n'+centerstring('CASSCF Settings',60,'-')+'\n'
    while True:
      nact=question('Number of active electrons:',int)[0]
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
      norb=question('Number of active orbitals:',int)[0]
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
    states=question('Number of states:',int)
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
    if INFOS['ctype']==2:
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

  print ''

  return INFOS

# ======================================================================================================================
# ======================================================================================================================
# ======================================================================================================================

def setup_input(INFOS):
  ''''''

  inpf='MOLCAS.template'
  print 'Writing input to %s' % (inpf)
  try:
    inp=open(inpf,'w')
  except IOError:
    print 'Could not open %s for write!' % (inpf)
    quit(1)

  s='basis %s\n' % (INFOS['basis'])

  if INFOS['ltype']>=4:
    s+='ras2 %i\n' % (INFOS['cas.norb'])
    s+='nactel %i\n' % (INFOS['cas.nact'])
    s+='inactive %i\n' % ((INFOS['nelec']-INFOS['cas.nact'])/2)
    for i,n in enumerate(INFOS['cas.nstates']):
      if n==0:
        continue
      s+='spin %i roots %i\n' % (i+1,n)

  s+='\n\n'
  s+='* Infos:\n'
  s+='* %s@%s\n' % (os.environ['USER'],os.environ['HOSTNAME'])
  s+='* Date: %s\n' % (datetime.datetime.now())
  s+='* Current directory: %s\n\n' % (os.getcwd())

  inp.write(s)

# ======================================================================================================================
# ======================================================================================================================
# ======================================================================================================================

def main():
  '''Main routine'''

  usage='''
python molcas_input.py

This interactive program prepares template files for the SHARC-MOLCAS interface.
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
  print '\nFinished\n'

  close_keystrokes()

# ======================================================================================================================

if __name__ == '__main__':
  try:
    main()
  except KeyboardInterrupt:
    print '\nCtrl+C makes me a sad SHARC ;-(\n'
    quit(0)