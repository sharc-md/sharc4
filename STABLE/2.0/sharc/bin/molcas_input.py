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
versiondate=datetime.date(2015,1,23)


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
'La':57, 
'Ce':58, 'Pr':59, 'Nd':60, 'Pm':61, 'Sm':62, 'Eu':63, 'Gd':64, 'Tb':65, 'Dy':66, 'Ho':67, 'Er':68, 'Tm':69, 'Yb':70, 'Lu':71,
'Hf':72, 'Ta':73,  'W':74, 'Re':75, 'Os':76, 'Ir':77, 'Pt':78, 'Au':79, 'Hg':80,
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


IToMult={1: 'Singlet',
         2: 'Doublet',
         3: 'Triplet',
         4: 'Quartet',
         5: 'Quintet',
         6: 'Sextet',
         7: 'Septet',
         8: 'Octet',
         'Singlet': 1,
         'Doublet': 2,
         'Triplet': 3,
         'Quartet': 4,
         'Quintet': 5,
         'Sextet': 6,
         'Septet': 7,
         'Octet': 8
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
  string+='  '+'='*    80+'\n'
  string+='||'+centerstring('',80)+'||\n'
  string+='||'+centerstring('MOLCAS Input file generator',80)+'||\n'
  string+='||'+centerstring('',80)+'||\n'
  string+='||'+centerstring('Author: Sebastian Mai',80)+'||\n'
  string+='||'+centerstring('',80)+'||\n'
  string+='||'+centerstring('Version:'+version,80)+'||\n'
  string+='||'+centerstring(versiondate.strftime("%d.%m.%y"),80)+'||\n'
  string+='||'+centerstring('',80)+'||\n'
  string+='  '+'='*    80+'\n\n'
  string+='''
This script allows to quickly create MOLCAS input files for single-points calculations
on the SA-CASSCF and (MS-)CASPT2 levels of theory. 
It also generates MOLCAS.template files to be used with the SHARC-MOLCAS Interface.
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

def question(question,typefunc,default=None,autocomplete=True,ranges=False):
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
    if typefunc==int and ranges:
      s+=' (range comprehension enabled)'
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
      posresponse=['y','yes','true', 't', 'ja',  'si','yea','yeah','aye','sure','definitely']
      negresponse=['n','no', 'false', 'f', 'nein', 'nope']
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

    if typefunc==float:
      # float will be returned as a list
      f=line.split()
      try:
        for i in range(len(f)):
          f[i]=typefunc(f[i])
        KEYSTROKES.write(line+' '*(40-len(line))+' #'+s+'\n')
        return f
      except ValueError:
        print 'Please enter floats!'
        continue

    if typefunc==int:
      # int will be returned as a list
      f=line.split()
      out=[]
      try:
        for i in f:
          if ranges and '~' in i:
            q=i.split('~')
            for j in range(int(q[0]),int(q[1])+1):
              out.append(j)
          else:
            out.append(int(i))
        KEYSTROKES.write(line+' '*(40-len(line))+' #'+s+'\n')
        return out
      except ValueError:
        if ranges:
          print 'Please enter integers or ranges of integers (e.g. "-3~-1  2  5~7")!'
        else:
          print 'Please enter integers!'
        continue

# ======================================================================================================================
# ======================================================================================================================
# ======================================================================================================================

def show_massses(masslist):
  s='Number\tType\tMass\n'
  for i,atom in enumerate(masslist):
    s+='%i\t%2s\t%12.9f %s\n' % (i+1,atom[0],atom[1], ['','*    '][atom[1]!=MASSES[atom[0]]])
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
- type (single point, optimization+freq or MOLCAS.template
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
  1       Single point calculations (RASSCF, CASPT2)
  2       Optimizations & Frequency calculations (RASSCF, CASPT2)
  3       MOLCAS.template file for SHARC dynamics (SA-CASSCF)
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
  guessninact=None
  guessnelec=None
  guessnorb=None
  guessbase=None
  guessstates=[0]*8
  guessspin=None


  # Geometry
  print centerstring('Geometry',60,'-')
  if ctype==3:
    print '\nNo geometry necessary for MOLCAS.template generation\n'
    INFOS['geom']=None
    # see whether a MOLCAS.input file is there, where we can take the number of electrons from
    nelec=0
    try:
      molproinput=open('MOLCAS.input','r')
      for line in molproinput:
        #print line.strip()
        if 'basis' in line.lower():
          guessbase=line.split()[-1]
        if 'nactel' in line.lower():
          guessnact=int(line.split()[-1].split(',')[0])
        if 'inact' in line.lower():
          guessninact=int(line.split()[-1])
        if 'ras2' in line.lower():
          guessnorb=int(line.split()[-1])
        if 'spin' in line.lower():
          guessspin=int(line.split()[-1])
        if 'ciroot' in line.lower():
          s=int(line.split()[-1].split(',')[0])
          guessstates[guessspin-1]=s
      try:
        for istate in range(len(guessstates)-1,-1,-1):
          if guessstates[istate]==0:
            guessstates.pop()
          else:
            break
        #print guessnorb,guessnact,guessninact,guessnelec
        if guessninact!=None and guessnact!=None:
          guessnelec=[2*guessninact+guessnact]
        if guessnorb!=None:
          guessnorb=[guessnorb]
        if guessnact!=None:
          guessnact=[guessnact]
        if guessninact!=None:
          guessninact=[guessninact]
        #print guessnorb,guessnact,guessninact,guessnelec
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
          line[0]=re.sub('[0-9]','',line[0])
          atom=[line[0],float(line[1]),float(line[2]),float(line[3])]
        except (IndexError,ValueError):
          print 'Malformatted: %s' % (path)
          fine=False
          continue
        geom.append(atom)
        try:
          ncharge+=NUMBERS[atom[0].title()]
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
    INFOS['charge']=charge
    INFOS['nelec']=ncharge-charge
    print 'Number of electrons: %i\n' % (ncharge-charge)

  # Masses
  if INFOS['freq']:
    # make default mass list
    masslist=[]
    for atom in geom:
      masslist.append( [atom[0],MASSES[atom[0]]] )
    # ask
    #INFOS['nondefmass']=not question('Use standard masses (most common isotope)?',bool,True)
    #if INFOS['nondefmass']:
      #INFOS['masslist']=ask_for_masses(masslist)
    #else:
    INFOS['masslist']=masslist

  # Level of theory
  print '\n'+centerstring('Level of theory',60,'-')
  print '''\nSupported by this script are:
  1       RASSCF
  2       CASPT2 %s
''' % (['','(Only numerical gradients)'][INFOS['freq']])
  #if ctype==3:
    #ltype=1
    #print 'Choosing RASSCF for MOLCAS.template generation.'
  #else:
  while True:
    ltype=question('Level of theory:',int)[0]
    if not ltype in [1,2]:
      print 'Enter an integer (1-2)!'
      continue
    break
  INFOS['ltype']=ltype


  # basis set
  print '\nPlease enter the basis set.'
  print '''Common available basis sets:
  Pople:     6-31G**, 6-311G, 6-31+G, 6-31G(d,p), ...    %s
  Dunning:   cc-pVXZ, aug-cc-pVXZ, cc-pVXZ-DK, ...    
  ANO:       ANO-S-vdzp, ANO-L, ANO-RCC                   ''' % (['','(Not available)'][ctype==3])
  basis=question('Basis set:',str,guessbase,False)
  INFOS['basis']=basis
  INFOS['cholesky']=question('Use Cholesky decomposition?',bool,False)

  # douglas kroll
  dk=question('Douglas-Kroll scalar-relativistic integrals?',bool,True)
  INFOS['DK']=dk

  # CASSCF
  if ltype>=1:
    print '\n'+centerstring('CASSCF Settings',60,'-')+'\n'
    while True:
      nact=question('Number of active electrons:',int,guessnact)[0]
      if nact<=0:
        print 'Enter a positive number larger than zero!'
        continue
      if INFOS['nelec']<nact:
        print 'Number of active electrons cannot be larger than total number of electrons!'
        continue
      if (INFOS['nelec']-nact)%2!=0:
        print 'nelec-nact must be even!'
        continue
      break
    INFOS['cas.nact']=nact
    while True:
      norb=question('Number of active orbitals:',int,guessnorb)[0]
      if norb<=0:
        print 'Enter a positive number!'
        continue
      if 2*norb<=nact:
        print 'norb must be larger than nact/2!'
        continue
      break
    INFOS['cas.norb']=norb

  if ltype>=1:
    print 'Please enter the number of states for state-averaging as a list of integers\ne.g. 3 0 2 for three singlets, zero doublets and two triplets.'
    while True:
      states=question('Number of states:',int,guessstates)
      maxmult=len(states)
      for i in range(maxmult):
        n=states[i]
        if (not i%2==INFOS['nelec']%2) and int(n)>0:
          print 'Nelec is %i. Ignoring states with mult=%i!' % (INFOS['nelec'], i+1)
          states[i]=0
        if n<0:
          states[i]=0
      if sum(states)==0:
        print 'No states!'
        continue
      break
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
      INFOS['opt.root']=[rmult,rstate]
      print 'Optimization: Only performing one RASSCF for %ss.' % (IToMult[rmult])
      for imult in range(len(INFOS['cas.nstates'])):
        if INFOS['cas.nstates'][imult]==0:
          continue
        if imult+1!=rmult:
          INFOS['cas.nstates'][imult]=0
      s='Accepted number of states:'
      for i in INFOS['cas.nstates']:
        s+=' %i' % (i)
      print s

  if ltype>1:
    print '\n'+centerstring('CASPT2 Settings',60,'-')+'\n'
    if ctype==1:
      INFOS['pt2.multi']=question('Multi-state CASPT2?',bool,True)
    else:
      INFOS['pt2.multi']=True
    INFOS['pt2.ipea']=not question('Set IPEA shift to zero?',bool,False)
    INFOS['pt2.imag']=question('Imaginary level shift?',float,[0.0])[0]





  print '\n'+centerstring('Further Settings',60,'-')+'\n'

  if ctype==1 and maxmult>1:
    INFOS['soc']=question('Do Spin-Orbit RASSI?',bool,False)
  else:
    INFOS['soc']=False

  print ''

  return INFOS

# ======================================================================================================================
# ======================================================================================================================
# ======================================================================================================================

def setup_input(INFOS):
  ''''''

  # choose file name
  if INFOS['ctype']==3:
    inpf='MOLCAS.template'
  else:
    inpf='MOLCAS.input'
  print 'Writing input to %s' % (inpf)
  try:
    inp=open(inpf,'w')
  except IOError:
    print 'Could not open %s for write!' % (inpf)
    quit(1)


  # template generation
  if INFOS['ctype']==3:
    s='basis %s\n' % (INFOS['basis'])
    s+='ras2 %i\n' % (INFOS['cas.norb'])
    s+='nactel %i\n' % (INFOS['cas.nact'])
    s+='inactive %i\n' % ((INFOS['nelec']-INFOS['cas.nact'])/2)
    s+='roots'
    for i,n in enumerate(INFOS['cas.nstates']):
      s+=' %i ' % (n)
    s+='\n\n'
    if not INFOS['DK']:
      s+='no-douglas-kroll\n'
    if INFOS['cholesky']:
      s+='cholesky\n'
    if INFOS['ltype']==1:
      s+='method CASSCF\n'
    elif INFOS['ltype']==2:
      if not INFOS['pt2.ipea']:
        s+='ipea 0.00\n'
      s+='imaginary %4.2f\n' % INFOS['pt2.imag']
      if INFOS['pt2.multi']:
        s+='method MS-CASPT2\n'
      else:
        s+='method CASPT2\n'
    s+='\n\n'
    s+='#     Infos:\n'
    s+='#     %s@%s\n' % (os.environ['USER'],os.environ['HOSTNAME'])
    s+='#     Date: %s\n' % (datetime.datetime.now())
    s+='#     Current directory: %s\n\n' % (os.getcwd())
    inp.write(s)
    return


  # input file generation
  s='**     %s generated by molcas_input.py Version %s\n\n' % (inpf,version)
  s+='&GATEWAY\n'
  if INFOS['geom']:
    s+='COORD\n%i\n\n' % (len(INFOS['geom']))
    for iatom,atom in enumerate(INFOS['geom']):
      s+='%s%i % 16.9f % 16.9f % 16.9f\n' % (atom[0],iatom+1,atom[1],atom[2],atom[3])
  if INFOS['basis']:
    s+='GROUP = nosym\nTITLE = Molcas-%s\nBASIS = %s\n' % (['SP','Opt',''][INFOS['ctype']-1],INFOS['basis'])


  if INFOS['ctype']==2:
    s+='\n\n**     ================ Optimization ================\n\n'
    s+='>> LINK FORCE %sOrbitals.RasOrb INPORB\n' % (IToMult[INFOS['opt.root'][0]])
    s+='>>> DO WHILE\n'


  s+='\n&SEWARD\n'
  if INFOS['DK']:
    s+='EXPERT\nR02O\n'
  if INFOS['soc']:
    s+='* If using MOLCAS v>=8.1, move the AMFI keyword to the &GATEWAY section.\n'
    s+='AMFI\n'
  if INFOS['cholesky']:
    s+='CHOLESKY\n'


  if INFOS['ctype']==1:
    if not INFOS['DK'] and INFOS['nelec']%2==0 and INFOS['charge']==0:
      s+='\n\n&SCF\n\n'
    else:
      s+='\n\n** For DKH integrals or with ions, MOLCAS SCF seems to not work properly.\n*&SCF\n\n'


  ijobiph=0
  for imult,nstate in enumerate(INFOS['cas.nstates']):
    if nstate==0:
      continue
    mult=imult+1
    ijobiph+=1

    if INFOS['ctype']==1:
      s+='\n\n**     ================ %s states ================\n\n' % (IToMult[mult])
      s+='**     Uncomment the following line in order to restart the orbitals:\n'
      s+='*      >> LINK FORCE %sOrbitals.RasOrb INPORB\n' % (IToMult[mult])

    s+='''
&RASSCF
SPIN   = %i
NACTEL = %i,0,0
INACT  = %i
RAS2   = %i
CIROOT = %i,%i,1
'''% (mult,
       INFOS['cas.nact'],
       (INFOS['nelec']-INFOS['cas.nact'])/2,
       INFOS['cas.norb'],
       nstate,nstate)
    if INFOS['ctype']==1:
      s+='''**     Uncomment the following line in order to restart the orbitals:
*LUMORB
**     Uncomment the following lines in order to change the orbital order:
*ALTER
*1
*1 1 2
'''
    if INFOS['ctype']==2:
      s+='LUMORB\n'
      if INFOS['ltype']==1 and nstate>1:
        s+='RLXROOT = %i\n' % (INFOS['opt.root'][1])
    if INFOS['ctype']==1:
      s+='''
>> SAVE $Project.rasscf.molden %sOrbitals.molden
>> SAVE $Project.RasOrb %sOrbitals.RasOrb
''' % (IToMult[mult],IToMult[mult])

    if INFOS['ltype']>1:
      s+='''
&CASPT2
SHIFT      = 0.0
IMAGINARY  = %4.2f
IPEASHIFT  = %4.2f
MAXITER    = 120
* If using MOLCAS v>=8.1, uncomment the following line to get CASPT2 properties (dipole moments):
*PROP
''' % (INFOS['pt2.imag'],[0.,0.25][INFOS['pt2.ipea']])
      if not INFOS['pt2.multi']:
        s+='NOMULT\n' 
      s+='MULTISTATE = %i %s\n' % (nstate, ' '.join([str(i+1) for i in range(nstate)]))
      if INFOS['ctype']==2:
        #s+='LUMORB\n'
        if INFOS['ltype']==2 and nstate>1:
          s+='RLXROOT = %i\n' % (INFOS['opt.root'][1])

    if INFOS['ctype']==1:
      if INFOS['ltype']==1:
        s+='\n>> SAVE $Project.JobIph JOB%03i\n\n' % (ijobiph)
      elif INFOS['ltype']==2:
        s+='\n>> SAVE $Project.JobMix JOB%03i\n\n' % (ijobiph)



  if INFOS['ctype']==2:
    s+='\n&SLAPAF\n>>> ENDDO\n'
  if INFOS['freq']:
    s+='\n**     ================ Frequencies ================\n'
    s+='\n&MCKINLEY\n'
    #s+='\n&MCLR\nMASS\n'
    #for iatom,atom in enumerate(INFOS['geom']):
      #s+='%s%i = %f\n' % (atom[0],iatom+1,INFOS['masslist'][iatom][1])


  if INFOS['ctype']==1:
    s+='\n\n**     ================ Final RASSI calculation ================\n\n'

    s+='&RASSI\nNROFJOBIPHS\n'
    njobiph=[]
    for nstate in INFOS['cas.nstates']:
      if nstate>0:
        njobiph.append(nstate)
    s+='%i %s\n' % (len(njobiph),' '.join([str(i) for i in njobiph]))
    ijobiph=0
    for imult,nstate in enumerate(INFOS['cas.nstates']):
      if nstate==0:
        continue
      mult=imult+1
      ijobiph+=1
      s+='%s\n' % (' '.join([str(i+1) for i in range(nstate)]))
    s+='CIPR\n'
    if INFOS['ltype']==2:
      s+='EJOB\n'
    if INFOS['soc']:
      s+='SPINORBIT\nSOCOUPLING = 0.0\n'

    s+='**If you want to calculate transition densities for TheoDORE:\n'
    s+='*       *Uncomment the corresponding block in the run script.\n'
    if INFOS['ltype']==2:
      s+='*       *Delete the EJOB keyword above.\n'
    s+='*       *Uncomment the following line:\n'
    s+='*TRD1\n'

  s+='\n\n'
  s+='*     Infos:\n'
  s+='*     %s at %s\n' % (os.environ['USER'],os.environ['HOSTNAME'])
  s+='*     Date: %s\n' % (datetime.datetime.now())
  s+='*     Current directory: %s\n\n' % (os.getcwd())

  inp.write(s)

# ======================================================================================================================
# ======================================================================================================================
# ======================================================================================================================

def set_runscript(INFOS):

  if INFOS['ctype']==3:
    # no run script for template generation
    return

  print ''
  if not question('Runscript?',bool,True):
    return
  print ''

  # MOLCAS executable
  print centerstring('Path to MOLCAS',60,'-')+'\n'
  path=os.getenv('MOLCAS')
  if path!=None:
    path=os.path.expanduser(os.path.expandvars(path))
    print 'Environment variable $MOLCAS detected:\n$MOLCAS=%s\n' % (path)
    if question('Do you want to use this MOLCAS installation?',bool,True):
      INFOS['molcas']=path
  if not 'molcas' in INFOS:
    print '\nPlease specify path to MOLCAS directory (SHELL variables and ~ can be used, will be expanded when interface is started).\n'
    INFOS['molcas']=question('Path to MOLCAS:',str)
  print ''


  # Scratch directory
  print centerstring('Scratch directory',60,'-')+'\n'
  print 'Please specify an appropriate scratch directory. This will be used to run the calculation. Remember that this script cannot check whether the path is valid, since you may run the calculation on a different machine. The path will not be expanded by this script.'
  INFOS['scratchdir']=question('Path to scratch directory:',str)+'/WORK'
  print ''
  # Keep scratch directory
  INFOS['delete_scratch']=question('Delete scratch directory after calculation?',bool,False)

  # Memory
  print '\n'+centerstring('Memory',60,'-')
  print '\nRecommendation: for small systems: 100-300 MB, for medium-sized systems: 1000-2000 MB\n'
  mem=abs(question('Memory in MB: ',int,[500])[0])
  # always give at least 50 MB
  mem=max(mem,50)
  INFOS['mem']=mem

  # make job name
  cwd=os.path.split(os.getcwd())[-1][0:6]
  if len(cwd)<6:
    cwd='_'*(6-len(cwd))+cwd
  cwd='MCAS'+cwd

  string='''#!/bin/bash
#$ -N %s
#$ -S /bin/bash
#$ -cwd

PRIMARY_DIR=%s
SCRATCH_DIR=%s

export MOLCAS=%s
export MOLCASMEM=%i
export MOLCASDISK=0
export MOLCASRAMD=0
export MOLCAS_MOLDEN=ON

#export MOLCAS_CPUS=1
#export OMP_NUM_THREADS=1

export Project="MOLCAS"
export HomeDir=$PRIMARY_DIR
export CurrDir=$PRIMARY_DIR
export WorkDir=$SCRATCH_DIR/$Project/
ln -sf $WorkDir $CurrDir/WORK

cd $HomeDir
mkdir -p $WorkDir

''' % (cwd,
       os.getcwd(), 
       INFOS['scratchdir'], 
       INFOS['molcas'],
       INFOS['mem'])

  for imult,nstate in enumerate(INFOS['cas.nstates']):
    if nstate==0:
      continue
    mult=imult+1
    string+='cp $HomeDir/%sOrbitals.RasOrb $WorkDir\n' % (IToMult[mult])

  string+='\n$MOLCAS/bin/molcas.exe MOLCAS.input &> $CurrDir/MOLCAS.log\n\n'

  for imult,nstate in enumerate(INFOS['cas.nstates']):
    if nstate==0:
      continue
    mult=imult+1
    string+='cp $WorkDir/%sOrbitals.* $HomeDir\n' % (IToMult[mult])

  string+='#mkdir -p $HomeDir/TRD/\n#cp $WorkDir/TRD2_* $HomeDir/TRD/\n'

  if INFOS['delete_scratch']:
    string+='\nrm -r $SCRATCH_DIR\n'


  runscript='run_MOLCAS.sh'
  print 'Writing run script %s' % (runscript)
  try:
    runf=open(runscript,'w')
  except IOError:
    print 'Could not write %s' (runscript)
    return
  runf.write(string)
  runf.close()
  os.chmod(runscript, os.stat(runscript).st_mode | stat.S_IXUSR)

# ======================================================================================================================
# ======================================================================================================================
# ======================================================================================================================

def warnings(INFOS):
  print centerstring(' WARNINGS ',62,'*')
  print '*'+' '*60+'*'
  if INFOS['ctype']==1:
    if INFOS['DK']:
      print '*'+centerstring('Douglas-Kroll: Will not do SCF!',60,' ')+'*'
      print '*'+' '*60+'*'
    if INFOS['nelec']%2!=0:
      print '*'+centerstring('Odd number of electrons: Will not do SCF!',60,' ')+'*'
      print '*'+' '*60+'*'
    if INFOS['charge']!=0:
      print '*'+centerstring('Nonzero charge: Will not do SCF!',60,' ')+'*'
      print '*'+' '*60+'*'
  print '*'+' '*60+'*'
  print '*'*62

# ======================================================================================================================
# ======================================================================================================================
# ======================================================================================================================

def main():
  '''Main routine'''

  usage='''
python molcas_input.py

This interactive program prepares a MOLCAS input file for ground state optimizations and frequency calculations with HF, DFT, MP2 and CASSCF. It also generates MOLCAS.template files to be used with the SHARC-MOLCAS interface.
'''

  description=''
  parser = OptionParser(usage=usage, description=description)

  displaywelcome()
  open_keystrokes()

  INFOS=get_infos()

  print centerstring('Full input',60,'#')+'\n'
  for item in INFOS:
    print item, ' '*    (15-len(item)), INFOS[item]
  print ''

  setup_input(INFOS)
  set_runscript(INFOS)
  print '\nFinished\n'

  close_keystrokes()

  warnings(INFOS)

# ======================================================================================================================

if __name__ == '__main__':
  try:
    main()
  except KeyboardInterrupt:
    print '\nCtrl+C makes me a sad SHARC ;-(\n'
    quit(0)